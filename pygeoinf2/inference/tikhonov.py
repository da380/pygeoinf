"""
The damped normal operator, as a one-parameter family.

Tikhonov least squares has two normal operators, as the Gaussian case does:

.. code-block:: text

    model space  N(t) = A* R^-1 A + t I
    data space   N(t) = A A* + t R

and they are, exactly, the Gaussian normal operators of
:mod:`~pygeoinf2.inference.normal` with an isotropic prior — ``Q^-1 == t I`` in
the model space, ``Q == (1/t) I`` in the data space, where the two factors of
``1/t`` cancel between the gain and the operator, so the estimator is the same
mapping and not merely a proportional one.

They are kept apart anyway, and not because the identity is doubtful. A
:class:`~pygeoinf2.inference.normal.NormalOperator` is one assembly of one
problem. This is a *family*: its whole purpose is to be walked along, because
the damping is what a discrepancy search solves for and what an L-curve sweeps.
An object whose point is the sweep should say so in its type, and the warm
starting that makes a sweep affordable has nowhere to live on a single
assembly.

Reading ``t`` as a prior variance is also a claim about what regularisation
means, and one that a damping does not have to make.

See DESIGN.md section 24.
"""

from __future__ import annotations

from typing import Any, Literal

from ..algebra.operators import LinearOperator
from ..algebra.spaces import HilbertSpace
from ..numerics.root_find import DampedSolves
from ..numerics.solvers import CholeskySolver, LinearSolver
from ..probability.gaussian import GaussianMeasure
from ..traits import Traits
from .normal import Formalism, choose_formalism

__all__ = ["TikhonovNormalOperator", "TikhonovFamily"]


class TikhonovNormalOperator(LinearOperator):
    """``N(t)`` at one damping, carrying the factors it was built from.

    The same bargain as :class:`~pygeoinf2.inference.normal.NormalOperator`:
    it behaves as the assembled operator, and it still knows ``A``, ``R`` and
    ``t``, so a structure-aware preconditioner can be built against it.
    """

    def __init__(
        self,
        forward: LinearOperator,
        damping: float,
        /,
        *,
        error: GaussianMeasure | None = None,
        formalism: Formalism = "auto",
    ) -> None:
        """
        Args:
            forward: the forward operator ``A``.
            damping: the Tikhonov parameter ``t``, non-negative.
            error: the data error measure ``R``. Omitted means unweighted
                least squares.
            formalism: which space to assemble in; ``"auto"`` takes the smaller.
        """
        if damping < 0.0:
            raise ValueError(f"The damping must be non-negative, got {damping}.")
        if error is not None and error.domain != forward.codomain:
            raise ValueError(
                f"The error measure lives on {error.domain!r}, but the forward "
                f"operator has data space {forward.codomain!r}."
            )
        chosen = choose_formalism(forward.domain, forward.codomain, formalism=formalism)

        if chosen == "model_space":
            precision = None if error is None else error.precision
            if error is not None and precision is None:
                raise ValueError(
                    "The model-space formalism needs the error precision R^-1, "
                    "and this measure was given only a covariance."
                )
            weighted = (
                forward.adjoint if precision is None else forward.adjoint @ precision
            )
            assembled = weighted @ forward
            shift = LinearOperator.identity(forward.domain)
            space = forward.domain
        else:
            assembled = forward @ forward.adjoint
            if error is None:
                shift = LinearOperator.identity(forward.codomain)
            else:
                if error.covariance is None:
                    raise ValueError(
                        "The data-space formalism needs the error covariance R, "
                        "and this measure was given only a precision."
                    )
                shift = error.covariance
            weighted = forward.adjoint
            space = forward.codomain

        base = assembled
        if damping > 0.0:
            assembled = assembled + damping * shift

        # Positive definite is a *claim*, and at zero damping it is the
        # caller's: an undamped normal operator is definite exactly when the
        # forward operator has full rank in the relevant space, which nothing
        # here can know. Damped, it is definite outright. Claiming it either
        # way keeps the two cases one code path and lets a direct solver fail
        # loudly rather than being refused in advance.
        super().__init__(
            space,
            space,
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )

        self._assembled = assembled
        self._base = base
        self._shift = shift
        self._weighted = weighted
        self._forward = forward
        self._error = error
        self._damping = float(damping)
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
    def damping(self) -> float:
        """The Tikhonov parameter ``t``."""
        return self._damping

    @property
    def model_space(self) -> HilbertSpace:
        """The model space."""
        return self._forward.domain

    @property
    def data_space(self) -> HilbertSpace:
        """The data space."""
        return self._forward.codomain

    @property
    def error(self) -> GaussianMeasure | None:
        """The data error measure, or None."""
        return self._error

    @property
    def has_error(self) -> bool:
        """Whether the problem carries a data error measure."""
        return self._error is not None

    @property
    def error_covariance(self) -> LinearOperator | None:
        """``R``, or None."""
        return None if self._error is None else self._error.covariance

    @property
    def error_precision(self) -> LinearOperator | None:
        """``R^-1``, or None."""
        return None if self._error is None else self._error.precision

    @property
    def prior_covariance(self) -> LinearOperator:
        """``(1 / t) I``: the prior this damping is, read as one.

        Supplied so that the preconditioners written for a Gaussian normal
        operator apply here unchanged — the identity of the module docstring,
        used rather than argued about. Undefined at zero damping, which is the
        unregularised case and has no prior reading.
        """
        if self._damping <= 0.0:
            raise ValueError(
                "Zero damping corresponds to no prior at all, so there is no "
                "prior covariance to report. Damp the problem, or use the "
                "factors directly."
            )
        return (1.0 / self._damping) * LinearOperator.identity(self.model_space)

    @property
    def prior_precision(self) -> LinearOperator:
        """``t I``."""
        return self._damping * LinearOperator.identity(self.model_space)

    @property
    def base(self) -> LinearOperator:
        """The undamped part: ``A* R^-1 A`` or ``A A*``."""
        return self._base

    @property
    def shift(self) -> LinearOperator:
        """What the damping multiplies: ``I`` or ``R``."""
        return self._shift

    @property
    def assembled(self) -> LinearOperator:
        """The plain operator, with the structure forgotten."""
        return self._assembled

    def __repr__(self) -> str:
        if "_formalism" not in self.__dict__:  # pragma: no cover
            return "TikhonovNormalOperator(<under construction>)"
        if self._formalism == "model_space":
            parts = "A* R^-1 A + t I" if self.has_error else "A* A + t I"
        else:
            parts = "A A* + t R" if self.has_error else "A A* + t I"
        return (
            f"TikhonovNormalOperator({parts}, t={self._damping:g}, "
            f"{self._formalism}, dim {self.domain.dim})"
        )

    # ----------------------------------------------------------------- #
    #                        Right-hand side                            #
    # ----------------------------------------------------------------- #

    def right_hand_side(self, data: Any, /) -> Any:
        """The right-hand side of ``N(t) w == v``, from observed data.

        The data shifted by the error's expectation, then ``A* R^-1`` applied
        in the model-space formalism and nothing in the data-space one.
        """
        space = self.data_space
        shifted = data
        if self._error is not None:
            shifted = space.subtract(data, self._error.expectation)
        if self._formalism == "model_space":
            return self._weighted(shifted)
        return shifted

    def model_from(self, solution: Any, /) -> Any:
        """The model, from the solution of ``N(t) w == v``.

        The solution itself in the model-space formalism; ``A*`` applied to it
        in the data-space one, where the unknown solved for is the dual
        variable rather than the model.
        """
        if self._formalism == "model_space":
            return solution
        return self._forward.adjoint(solution)

    def surrogate(
        self,
        /,
        *,
        forward: LinearOperator | None = None,
        error: GaussianMeasure | None = None,
        damping: float | None = None,
        formalism: Formalism | None = None,
    ) -> "TikhonovNormalOperator":
        """The same operator with any of its factors replaced by cheap ones."""
        replacement = self._forward if forward is None else forward
        if replacement.codomain != self.data_space:
            raise ValueError(
                f"A surrogate must share the data space: this one maps into "
                f"{replacement.codomain!r} rather than {self.data_space!r}."
            )
        return TikhonovNormalOperator(
            replacement,
            self._damping if damping is None else damping,
            error=self._error if error is None else error,
            formalism=self._formalism if formalism is None else formalism,
        )


class TikhonovFamily:
    """``t -> N(t)``, with the solves along a sweep warm-started.

    What a discrepancy search walks along, and what an L-curve samples. Holding
    the family rather than one assembly is what lets consecutive solves start
    from each other: in a bisection the multipliers converge, so by the end each
    solve is a small correction to the last rather than a fresh problem.

    The undamped parts are built once. Only the sum is reassembled per damping,
    and that is free — it is an expression, not a matrix.
    """

    def __init__(
        self,
        forward: LinearOperator,
        /,
        *,
        error: GaussianMeasure | None = None,
        solver: LinearSolver | None = None,
        formalism: Formalism = "auto",
    ) -> None:
        """
        Args:
            forward: the forward operator ``A``.
            error: the data error measure ``R``.
            solver: how to invert ``N(t)``. Cholesky by default. An *iterative*
                solver is what makes warm starting mean anything; a direct one
                refactorises at every damping and reports zero iterations,
                which is the honest signal that it is doing so.
            formalism: which space to assemble in.
        """
        self._forward = forward
        self._error = error
        self._solver = CholeskySolver() if solver is None else solver
        self._formalism = choose_formalism(
            forward.domain, forward.codomain, formalism=formalism
        )
        # Built once, at zero damping, to obtain the two fixed pieces.
        self._template = TikhonovNormalOperator(
            forward, 0.0, error=error, formalism=self._formalism
        )
        self._solves = DampedSolves(
            self._template.base,
            self._template.shift,
            self._solver,
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )

    @property
    def formalism(self) -> Literal["model_space", "data_space"]:
        """Which space the family is assembled in."""
        return self._formalism

    @property
    def forward(self) -> LinearOperator:
        """The forward operator ``A``."""
        return self._forward

    @property
    def error(self) -> GaussianMeasure | None:
        """The data error measure, or None."""
        return self._error

    @property
    def solver(self) -> LinearSolver:
        """How each member is inverted."""
        return self._solver

    def at(self, damping: float, /) -> TikhonovNormalOperator:
        """The member of the family at one damping."""
        return TikhonovNormalOperator(
            self._forward,
            damping,
            error=self._error,
            formalism=self._formalism,
        )

    def right_hand_side(self, data: Any, /) -> Any:
        """The right-hand side, which does not depend on the damping."""
        return self._template.right_hand_side(data)

    def model_from(self, solution: Any, /) -> Any:
        """The model, from a solution of the normal equations."""
        return self._template.model_from(solution)

    def solve(self, damping: float, right_hand_side: Any, /, *, x0: Any = None) -> Any:
        """One member's solve, warm-started from *x0*.

        Returns the solver's own result, so the iteration count is visible;
        that is the only way to tell a warm start that is working from one that
        silently is not.
        """
        return self._solves.solve(damping, right_hand_side, x0=x0)

    def model(self, damping: float, data: Any, /, *, x0: Any = None) -> tuple[Any, Any]:
        """The model at one damping, and the raw solution to warm-start from."""
        result = self.solve(damping, self.right_hand_side(data), x0=x0)
        return self.model_from(result.solution), result
