"""
The normal operator, and what it was assembled from.

A linear Gaussian problem has two normal operators, one per formalism:

.. code-block:: text

    data space   N = A Q A* + R           on the data space
    model space  N = Q^-1 + A* R^-1 A     on the model space

and either can be assembled and handed to a solver. But a solver is not the
only thing that wants one. A *preconditioner* wants one too, and the useful
preconditioners for this operator are not generic: they exploit the fact that
it is ``A Q A* + R`` and not merely some positive definite operator. v1's
diagonal preconditioner uses ``<v, A Q A* v> == <A* v, Q A* v>``, which needs
``A`` and ``Q`` apart; the Woodbury one inverts through the opposite space,
which needs all three. Assembling ``N`` into a single operator destroys exactly
the structure they need, which is why in v1 they had to live as methods on the
inversion class, the only place that still held the parts.

So the operator carries its factors. Generic preconditioners — Jacobi,
spectral, banded — see a :class:`LinearOperator` and are unchanged; the
structure-aware ones read ``forward``, ``prior_covariance`` and
``error_covariance`` off it. The preconditioners then stay free-standing and
reusable rather than being bound to one inversion class, and an inversion
exposes its normal operator instead of a growing family of preconditioner
factories. See DESIGN.md section 23.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal

from ..algebra.operators import LinearOperator
from ..algebra.spaces import HilbertSpace
from ..probability.gaussian import GaussianMeasure
from ..traits import Traits

__all__ = [
    "FactoredNormalOperator",
    "NormalOperator",
    "Formalism",
    "choose_formalism",
]

Formalism = Literal["auto", "model_space", "data_space"]


def choose_formalism(
    model_space: HilbertSpace,
    data_space: HilbertSpace,
    /,
    *,
    formalism: Formalism = "auto",
) -> Literal["model_space", "data_space"]:
    """Which space to assemble the normal equations in.

    A purely computational choice — the two give the same answer — so ``auto``
    takes whichever space is smaller, falling back to the data space when a
    dimension is unavailable.

    ``auto`` is offered and is *not* the default anywhere. Comparing dimensions
    says nothing about whether the model-space route is available at all: it
    needs ``Q^-1``, and a function-space prior often has none. It also reads a
    discretisation's size as though it were the problem's. Data spaces can be
    large; model spaces are usually larger still, so the data space is where
    the normal equations belong unless there is a reason otherwise, and the
    model-space formalism is kept for when there *is* one — an overdetermined
    problem with a cheap precision, or a surrogate. See DESIGN.md §18.6, §27.

    Args:
        model_space: where the models live.
        data_space: where the data live.
        formalism: ``"model_space"``, ``"data_space"``, or ``"auto"`` to pick
            by dimension. Read the paragraph above before reaching for
            ``"auto"``: it compares sizes, which is not the question.

    Returns:
        The formalism to use.

    Raises:
        ValueError: for an unrecognised formalism.
    """
    if formalism not in ("auto", "model_space", "data_space"):
        raise ValueError(
            f"The formalism is 'auto', 'model_space' or 'data_space', got "
            f"{formalism!r}."
        )
    if formalism != "auto":
        return formalism
    try:
        return "model_space" if model_space.dim <= data_space.dim else "data_space"
    except (AttributeError, NotImplementedError):  # pragma: no cover
        return "data_space"


class FactoredNormalOperator(LinearOperator):
    """A normal operator that still knows what it was assembled from.

    The contract a structure-aware preconditioner needs, and the reason it is
    written down rather than left implicit: there are two normal operators in
    this package — the Gaussian :class:`NormalOperator` and the Tikhonov
    :class:`~pygeoinf2.inference.tikhonov.TikhonovNormalOperator` — and a
    preconditioner that checked for one of them by name would silently refuse
    the other. Both are ``A ... A* + ...``; both can be preconditioned the same
    way.

    Subclasses supply ``forward``, ``prior_covariance`` and
    ``error_covariance``; the spaces follow from the first.
    """

    @property
    @abstractmethod
    def formalism(self) -> Literal["model_space", "data_space"]:
        """Which space this was assembled in."""

    @property
    @abstractmethod
    def forward(self) -> LinearOperator:
        """The forward operator ``A``."""

    @property
    @abstractmethod
    def prior_covariance(self) -> LinearOperator:
        """``Q``: the prior covariance, or the one a damping stands for."""

    @property
    @abstractmethod
    def error_covariance(self) -> LinearOperator | None:
        """``R``, or None when the problem is noise-free."""

    @property
    def model_space(self) -> HilbertSpace:
        """The model space, which the forward operator maps out of."""
        return self.forward.domain

    @property
    def data_space(self) -> HilbertSpace:
        """The data space, which the forward operator maps into."""
        return self.forward.codomain

    @property
    def has_error(self) -> bool:
        """Whether the problem carries a data error measure."""
        return self.error_covariance is not None


class NormalOperator(FactoredNormalOperator):
    """The normal operator of a linear Gaussian problem, with its factors.

    Behaves as the assembled operator everywhere an operator is wanted, and
    additionally exposes ``forward``, ``prior``, ``error`` and the covariances
    and precisions taken from them.
    """

    def __init__(
        self,
        forward: LinearOperator,
        prior: GaussianMeasure,
        /,
        *,
        error: GaussianMeasure | None = None,
        formalism: Formalism = "data_space",
    ) -> None:
        """
        Args:
            forward: the forward operator ``A``.
            prior: the prior ``Q`` on the model space. In the model-space
                formalism its precision is needed, not merely its covariance.
            error: the data error measure ``R``. Omitted means the data are
                exact, which only the data-space formalism can express — see
                below.
            formalism: which space to assemble in. ``"auto"`` takes the
                smaller, except that it takes the data space whenever there is
                no error measure, because that is then the only one available.

        Raises:
            ValueError: if the prior or error lives on the wrong space; if the
                chosen formalism needs a covariance or a precision that was not
                supplied; or if the model-space formalism is asked for with no
                error measure.

        **With no error measure there is no model-space formalism.** The two
        assemblies are

        .. code-block:: text

            data space    A Q A* + R          exact data is R == 0
            model space   Q^-1 + A* R^-1 A    exact data is R^-1 == infinity

        and dropping the ``R`` terms gives ``A Q A*`` on one side but
        ``Q^-1 + A* A`` on the other. The second is not the noise-free problem:
        it is the problem with ``R == I``, a unit-variance error model wearing
        the label "no error". The two posteriors genuinely differ — measured
        mean difference 0.86 against a mean of norm 2.6 on a random 12-by-8
        problem — so offering both would be offering two answers to one
        question. v1 does exactly that, silently.

        Conditioning on exact data is a limit the model-space form cannot take,
        so it is refused rather than approximated.
        """
        if prior.domain != forward.domain:
            raise ValueError(
                f"The prior lives on {prior.domain!r}, but the forward "
                f"operator has model space {forward.domain!r}."
            )
        if error is not None and error.domain != forward.codomain:
            raise ValueError(
                f"The error measure lives on {error.domain!r}, but the forward "
                f"operator has data space {forward.codomain!r}."
            )
        if error is None:
            # Exact data has no model-space assembly at all (see the docstring),
            # so "auto" must not be allowed to wander into one, and an explicit
            # request has to be refused rather than quietly given R == I.
            if formalism == "model_space":
                raise ValueError(
                    "The model-space formalism needs an error measure. Without "
                    "one the data are exact, and 'Q^-1 + A* A' is not that "
                    "problem: it is the problem with R == I, whose posterior is "
                    "a different measure. Use the data-space formalism, which "
                    "expresses exact data as R == 0, or supply the error "
                    "measure you mean."
                )
            chosen = "data_space"
        else:
            chosen = choose_formalism(
                forward.domain, forward.codomain, formalism=formalism
            )

        if chosen == "data_space":
            if prior.covariance is None:
                raise ValueError(
                    "The data-space formalism needs the prior covariance Q, "
                    "and this prior was given only a precision."
                )
            assembled = forward @ prior.covariance @ forward.adjoint
            if error is not None:
                if error.covariance is None:
                    raise ValueError(
                        "The data-space formalism needs the error covariance R, "
                        "and this measure was given only a precision."
                    )
                assembled = assembled + error.covariance
            space = forward.codomain
        else:
            if prior.precision is None:
                raise ValueError(
                    "The model-space formalism needs the prior precision Q^-1. "
                    "Supply one, or obtain it by damping with "
                    "GaussianMeasure.with_regularized_inverse, or assemble in "
                    "the data space instead."
                )
            # error is not None on this branch: the guard above sends the
            # error-free case to the data space.
            if error.precision is None:
                raise ValueError(
                    "The model-space formalism needs the error precision "
                    "R^-1, and this measure was given only a covariance."
                )
            assembled = prior.precision + forward.adjoint @ error.precision @ forward
            space = forward.domain

        # A sum of positive definite operators, which the trait algebra cannot
        # deduce through the addition, so it is claimed here where the reason
        # is known.
        super().__init__(
            space,
            space,
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )
        self._assembled = assembled
        self._forward = forward
        self._prior = prior
        self._error = error
        self._formalism = chosen

    def _value(self, x: Any) -> Any:
        return self._assembled(x)

    def _adjoint_value(self, y: Any) -> Any:
        return self._assembled(y)

    # ----------------------------------------------------------------- #
    #                             The factors                           #
    # ----------------------------------------------------------------- #

    @property
    def formalism(self) -> Literal["model_space", "data_space"]:
        """Which space this was assembled in."""
        return self._formalism

    @property
    def forward(self) -> LinearOperator:
        """The forward operator ``A``."""
        return self._forward

    @property
    def prior(self) -> GaussianMeasure:
        """The prior measure, whose covariance is ``Q``."""
        return self._prior

    @property
    def error(self) -> GaussianMeasure | None:
        """The data error measure, whose covariance is ``R``, or None."""
        return self._error

    @property
    def has_error(self) -> bool:
        """Whether the problem carries a data error measure."""
        return self._error is not None

    @property
    def model_space(self) -> HilbertSpace:
        """The model space, which the forward operator maps out of."""
        return self._forward.domain

    @property
    def data_space(self) -> HilbertSpace:
        """The data space, which the forward operator maps into."""
        return self._forward.codomain

    @property
    def prior_covariance(self) -> LinearOperator:
        """``Q``."""
        return self._prior.covariance

    @property
    def prior_precision(self) -> LinearOperator | None:
        """``Q^-1``, or None when the prior has no precision."""
        return self._prior.precision

    @property
    def error_covariance(self) -> LinearOperator | None:
        """``R``, or None for a noise-free problem."""
        return None if self._error is None else self._error.covariance

    @property
    def error_precision(self) -> LinearOperator | None:
        """``R^-1``, or None for a noise-free problem."""
        return None if self._error is None else self._error.precision

    @property
    def assembled(self) -> LinearOperator:
        """The plain operator, with the structure forgotten.

        Rarely wanted — this object already *is* the assembled operator. It is
        here for the case where an operator of exactly this type would be
        picked up by a structure-aware routine that should not fire.
        """
        return self._assembled

    def __repr__(self) -> str:
        if "_formalism" not in self.__dict__:  # pragma: no cover
            return "NormalOperator(<under construction>)"
        parts = "A Q A* + R" if self.has_error else "A Q A*"
        if self._formalism == "model_space":
            parts = "Q^-1 + A* R^-1 A" if self.has_error else "Q^-1 + A* A"
        return f"NormalOperator({parts}, {self._formalism}, " f"dim {self.domain.dim})"

    # ----------------------------------------------------------------- #
    #                     Surrogates and the update                     #
    # ----------------------------------------------------------------- #

    def surrogate(
        self,
        /,
        *,
        forward: LinearOperator | None = None,
        prior: GaussianMeasure | None = None,
        error: GaussianMeasure | None = None,
        formalism: Formalism | None = None,
    ) -> "NormalOperator":
        """The same normal operator with any of its factors replaced.

        The point of a surrogate is to be *cheap*: a smoother forward operator,
        a coarser discretisation, a stationary prior standing in for a
        non-stationary one, a diagonal error covariance. Its inverse is then a
        preconditioner for the true operator, and correctness never depends on
        how close the surrogate is — only the iteration count does.

        The surrogate may live on a **different model space**. In the tomography
        example it is a sphere of a sixth the degree, with its own prior and its
        own path-average operator, and only the data space is shared. That is
        exactly why the data-space formalism is the one that survives the
        substitution: ``A Q A* + R`` acts on the data space whatever the model
        space is.

        Args:
            forward: a cheaper ``A``. May map from a different model space,
                in which case *prior* must be given too.
            prior: a cheaper ``Q``, on the surrogate's own model space.
            error: a cheaper ``R``, on the shared data space.
            formalism: which space to assemble in. Inherited if omitted.

        Returns:
            The surrogate normal operator.

        Raises:
            ValueError: if the surrogate does not share the data space, or if
                a new forward operator is given on a different model space
                without a prior to match it -- the prior it would otherwise
                inherit lives somewhere else.
        """
        replacement_forward = self._forward if forward is None else forward
        replacement_prior = self._prior if prior is None else prior
        replacement_error = self._error if error is None else error
        if forward is not None and prior is None:
            if replacement_forward.domain != self._prior.domain:
                raise ValueError(
                    f"The surrogate forward operator has model space "
                    f"{replacement_forward.domain!r}, but the prior it would "
                    f"inherit lives on {self._prior.domain!r}. A surrogate on "
                    f"a different model space needs its own prior."
                )
        if replacement_forward.codomain != self.data_space:
            raise ValueError(
                f"A surrogate must share the data space: this one maps into "
                f"{replacement_forward.codomain!r} rather than "
                f"{self.data_space!r}."
            )
        return NormalOperator(
            replacement_forward,
            replacement_prior,
            error=replacement_error,
            formalism=self._formalism if formalism is None else formalism,
        )

    def weighted_adjoint(self) -> LinearOperator:
        """``A* R^-1``, or ``A*`` when the problem is noise-free.

        The piece the model-space formalism applies to the data residual.

        Returns:
            The operator from the data space to the model space.

        Raises:
            ValueError: if there is an error measure but it carries no
                precision. The model-space formalism is built on ``R^-1``, and
                a measure that cannot supply one cannot be used that way --
                the data-space formalism needs only ``R`` itself.
        """
        if self._error is None:
            return self._forward.adjoint
        precision = self._error.precision
        if precision is None:
            raise ValueError(
                "The error measure has no precision R^-1, which the "
                "model-space formalism needs."
            )
        return self._forward.adjoint @ precision

    def gain(self, inverse: LinearOperator, /) -> LinearOperator:
        """The Kalman gain, given an inverse of this operator.

        ``Q A* N^-1`` in the data-space formalism, ``N^-1 A* R^-1`` in the
        model-space one. Both map data residuals to model updates, and they are
        the same operator written two ways.
        """
        if self._formalism == "data_space":
            return self.prior_covariance @ self._forward.adjoint @ inverse
        return inverse @ self.weighted_adjoint()

    def posterior_covariance(
        self, inverse: LinearOperator, gain: LinearOperator, /
    ) -> LinearOperator:
        """``Q - K A Q``, which in the model-space formalism is just ``N^-1``."""
        if self._formalism == "model_space":
            return inverse
        prior_covariance = self.prior_covariance
        # A Schur complement, so positive semidefinite -- a posterior
        # covariance always is. The trait algebra cannot see that through a
        # difference, so the claim is made where the reason is known.
        return (prior_covariance - gain @ self._forward @ prior_covariance).with_traits(
            Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE
        )

    def right_hand_side(self, residual: Any, /) -> Any:
        """The right-hand side of ``N w = v`` for a shifted data residual.

        The residual itself in the data-space formalism; ``A* R^-1`` applied to
        it in the model-space one.
        """
        if self._formalism == "data_space":
            return residual
        return self.weighted_adjoint()(residual)
