"""
An estimator is a mapping from data to an answer.

Three kinds, distinguished only by what ``__call__`` returns: a point, a
measure, or a set. Which kind you get follows the description of
uncertainty: no prior gives a point, a prior measure with a noise measure
gives a measure, a prior set with a noise set gives a set.

The property space is the model space, or a space reached from it by an
operator ``T``. An inverse problem is an inference problem with ``T == identity``, so
:meth:`Estimator.push_forward` is the whole of the difference: applying ``T``
to a point, pushing a measure through it, taking the image of a set.

A point estimator that is also an :class:`~pygeoinf2.algebra.operators.Operator`
joins the operator algebra: it can be composed, differentiated, and used as
the forward map of a further inference. :class:`LinearPointEstimator` is the
affine case, with its error budget attached; :class:`ComposedPointEstimator`
is any point estimator followed by a property operator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..algebra.operators import (
    AffineOperator,
    Linearization,
    LinearOperator,
    Operator,
)
from ..algebra.spaces import HilbertSpace
from ..probability.base import ProbabilityMeasure
from ..probability.gaussian import GaussianMeasure
from ..traits import Traits, congruence_traits

__all__ = [
    "Estimator",
    "PointEstimator",
    "MeasureEstimator",
    "SetEstimator",
    "LinearPointEstimator",
    "ComposedPointEstimator",
    "GaussianEstimator",
]


class Estimator(ABC):
    """A mapping from data to an answer about a property of the model."""

    @property
    @abstractmethod
    def data_space(self) -> HilbertSpace:
        """The space the data live in."""

    @property
    @abstractmethod
    def property_space(self) -> HilbertSpace:
        """The space the answer is about — the model space, or a property space."""

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """The answer, given data."""

    @abstractmethod
    def push_forward(self, operator: LinearOperator, /) -> "Estimator":
        """The estimator of a property of the model, rather than the model."""


class PointEstimator(Estimator):
    """An estimator returning a single vector of the property space.

    The base of every point-valued method, including those that are also
    operators from the data space: the affine ones are
    :class:`LinearPointEstimator`, and the ones nonlinear in the data stay
    :class:`~pygeoinf2.algebra.operators.Operator` subclasses carrying their
    exact derivative, so an inference can be built on top of them by the
    chain rule. A subclass that is also an operator lists this class first
    and the operator class second, and defines ``__call__`` itself, since
    the abstract one here would otherwise shadow the operator's.
    """


class MeasureEstimator(Estimator):
    """An estimator returning a probability measure on the property space."""


class SetEstimator(Estimator):
    """An estimator returning a subset of the property space."""


class LinearPointEstimator(PointEstimator, AffineOperator):
    """A point estimator that is affine in the data, so also an operator.

    It is an :class:`~pygeoinf2.algebra.operators.AffineOperator`, ``d -> B d +
    t``, and carries its error budget alongside the estimate itself. With
    ``A`` the forward operator, ``T`` the property operator and ``e`` the data
    error, of expectation ``mu_e``, the estimate of ``T m`` from data
    ``d == A m + e`` is in error by

    .. code-block:: text

        B d + t - T m == [B (A m + mu_e) + t - T m] + B (e - mu_e)

    The first term is the bias, the error the estimate makes on average; the
    second is the noise, of mean zero. An estimator that removes the error's
    expectation has ``t == -B mu_e`` and the bias is ``(B A - T) m``; one
    confined to an affine subspace keeps a constant part as well. The
    budget is two operators, each returned by a method of the same name:

    :meth:`resolution`
        ``B A``, the averaging kernel: how the estimate sees the true model.
        Its departure from ``T`` is what makes the bias, which :meth:`bias`
        evaluates for a given model.
    :meth:`propagated_covariance`
        ``B R B*``, the data error mapped into the property space: the
        covariance of the second term.

    The second term is a statement about the noise and the budget bounds it.
    The first is a statement about the model, and bounding it takes a prior:
    under a prior measure the complete answer is the posterior, under a prior
    set the feasible property set. The budget is not an uncertainty on its
    own, and this class does not turn it into one.
    """

    def __init__(
        self,
        operator: LinearOperator,
        /,
        *,
        translation: Any = None,
        lazy: bool = False,
        forward_operator: LinearOperator | None = None,
        error: ProbabilityMeasure | None = None,
        property_operator: LinearOperator | None = None,
    ) -> None:
        """
        Args:
            operator: ``B``, mapping data to the property space.
            translation: an offset added to ``B d``, a vector of the
                property space; zero if omitted. With *lazy*, a
                zero-argument callable computing it on first use instead.
            lazy: take *translation* as such a callable. An estimator whose
                offset costs a solve passes one, so that building it does
                not pay for the solve.
            forward_operator: ``A``, kept so ``resolution`` can be formed.
            error: the data error, kept so its covariance can be propagated
                and its expectation taken in :meth:`bias`.
            property_operator: ``T``, from the model space to the property
                space, which :meth:`bias` subtracts from the resolution. The
                identity when omitted, the estimator then being of the model
                itself.

        Raises:
            ValueError: if ``T`` is omitted but ``B`` does not land in the
                model space ``A`` acts on, so that there is no identity to
                take, or if ``T`` is given on the wrong spaces.
        """
        if translation is None:
            super().__init__(operator, operator.codomain.zero())
        else:
            super().__init__(operator, translation, lazy=lazy)
        if property_operator is None:
            if (
                forward_operator is not None
                and forward_operator.domain != operator.codomain
            ):
                raise ValueError(
                    "The estimator lands in a space other than the model space "
                    "of its forward operator, so it estimates a property; pass "
                    "property_operator= to say which."
                )
            property_operator = LinearOperator.identity(operator.codomain)
        elif property_operator.codomain != operator.codomain or (
            forward_operator is not None
            and property_operator.domain != forward_operator.domain
        ):
            raise ValueError(
                "The property operator must map the model space of the forward "
                "operator to the space the estimator lands in."
            )
        self._operator = operator
        self._forward_operator = forward_operator
        self._error = error
        self._property_operator = property_operator

    @property
    def operator(self) -> LinearOperator:
        """``B`` itself, without the translation."""
        return self._operator

    @property
    def property_operator(self) -> LinearOperator:
        """``T``, the property the estimate is of; the identity for the model itself."""
        return self._property_operator

    @property
    def data_space(self) -> HilbertSpace:
        """``B``'s domain."""
        return self._operator.domain

    @property
    def property_space(self) -> HilbertSpace:
        """``B``'s codomain."""
        return self._operator.codomain

    def __call__(self, data: Any) -> Any:
        """The estimate ``B d + t``."""
        return self._value(data)

    def resolution(self) -> LinearOperator:
        """``B A``: how the estimate sees the model, the part of it that varies with the truth.

        The resolution operator, which says what is actually being estimated:
        it is the property operator ``T`` only for a perfectly resolved
        problem, and its departure from ``T`` is the blurring the data
        impose. It is not a bound on the error: the error it describes, ``(B
        A - T) m``, depends on the model, and saying how large that can be
        takes a prior.

        Returns:
            The operator from the model space to the property space.

        Raises:
            ValueError: if the estimator was built without a forward
                operator, in which case there is no ``A`` to compose with.
        """
        if self._forward_operator is None:
            raise ValueError(
                "This estimator was built without a forward operator, so it "
                "cannot form its resolution."
            )
        return self._operator @ self._forward_operator

    def bias(self, model: Any, /) -> Any:
        """``B (A m + mu_e) + t - T m``: the error the estimate makes on a given model, on average.

        The term the propagated covariance leaves out: the estimate of the
        data the model predicts, at the error's expectation ``mu_e``, less
        the property the estimate is of. It is ``(B A - T) m`` for an
        estimator that removes the error's expectation, so zero for every
        model only when the problem is perfectly resolved. It needs a model
        to evaluate, which is the point: without a statement about the model
        there is no bound on it.

        Args:
            model: the model to evaluate the bias at.

        Returns:
            A vector of the property space.

        Raises:
            ValueError: if the estimator was built without a forward
                operator.
        """
        if self._forward_operator is None:
            raise ValueError(
                "This estimator was built without a forward operator, so it "
                "cannot form its bias."
            )
        expected = self._forward_operator(model)
        if self._error is not None:
            expected = self.data_space.add(expected, self._error.expectation)
        return self.property_space.subtract(
            self._value(expected), self._property_operator(model)
        )

    def propagated_covariance(self) -> LinearOperator:
        """``B R B*``: the data error, as a covariance on the property space.

        What the noise becomes once it has been through the estimator: the
        covariance of ``B e``, the second term of the error. It is not the
        uncertainty of the estimate, because the first term, the bias, is
        missing from it, and that term is bounded by a prior rather than by
        the noise. A Gaussian with this covariance about the estimate looks
        like a posterior and is not one.

        Returns:
            The covariance operator on the property space.

        Raises:
            ValueError: if the estimator was built without a data error
                measure, there being no ``R`` to propagate.
        """
        if self._error is None:
            raise ValueError("This estimator was built without a data error measure.")
        return self._operator @ self._error.covariance @ self._operator.adjoint

    def push_forward(self, operator: LinearOperator, /) -> "LinearPointEstimator":
        """The estimator of ``T m`` rather than of ``m``.

        Args:
            operator: the property map ``T``, from the current property
                space.

        Returns:
            The estimator ``T B d + T t``, with its budget composed likewise.
            The offset ``T t`` is formed on first use, so pushing forward an
            estimator whose offset is still unevaluated does not evaluate it.
        """
        return LinearPointEstimator(
            operator @ self._operator,
            translation=lambda: operator(self.translation),
            lazy=True,
            forward_operator=self._forward_operator,
            error=self._error,
            property_operator=operator @ self._property_operator,
        )


class ComposedPointEstimator(PointEstimator, Operator):
    """A point estimator followed by a property operator: ``T`` applied to the estimate.

    What :meth:`PointEstimator.push_forward` returns for an estimator that is
    not affine in the data, so that composing with ``T`` has no simpler form.
    When the inner estimator is an operator with a derivative, the composed
    one is too, by the chain rule: ``T`` after the inner derivative.
    """

    def __init__(self, estimator: PointEstimator, operator: LinearOperator, /) -> None:
        """
        Args:
            estimator: the point estimator, of the model or of a property.
            operator: ``T``, from the estimator's property space.

        Raises:
            ValueError: if ``T`` does not act on the estimator's property
                space.
        """
        if operator.domain != estimator.property_space:
            raise ValueError(
                "The property operator must act on the estimator's property space."
            )
        super().__init__(estimator.data_space, operator.codomain)
        self._estimator = estimator
        self._operator = operator

    @property
    def estimator(self) -> PointEstimator:
        """The estimator being pushed forward."""
        return self._estimator

    @property
    def property_operator(self) -> LinearOperator:
        """``T``, applied to its estimate."""
        return self._operator

    @property
    def data_space(self) -> HilbertSpace:
        """The inner estimator's data space."""
        return self._estimator.data_space

    @property
    def property_space(self) -> HilbertSpace:
        """``T``'s codomain."""
        return self._operator.codomain

    @property
    def has_derivative(self) -> bool:
        """True when the inner estimator is an operator carrying a derivative."""
        return isinstance(self._estimator, Operator) and self._estimator.has_derivative

    def __call__(self, data: Any, /, **kwargs: Any) -> Any:
        """``T`` applied to the inner estimate.

        Args:
            data: the observations.
            **kwargs: passed to the inner estimator's call, such as the
                ``start=`` of an estimator that searches.

        Returns:
            A vector of the property space.
        """
        return self._operator(self._estimator(data, **kwargs))

    def _value(self, data: Any) -> Any:
        return self._operator(self._estimator(data))

    def _derivative(self, data: Any) -> LinearOperator:
        return self._operator @ self._inner_operator().derivative(data)

    def _linearize(self, data: Any) -> Linearization:
        inner = self._inner_operator().at(data)
        return Linearization(
            data, self._operator(inner.value), self._operator @ inner.derivative
        )

    def _inner_operator(self) -> Operator:
        if not isinstance(self._estimator, Operator):
            raise NotImplementedError(
                f"{type(self._estimator).__name__} is not an operator, so the "
                "composed estimator carries no derivative."
            )
        return self._estimator

    def push_forward(self, operator: LinearOperator, /) -> "ComposedPointEstimator":
        """The estimator of ``S T m``: one composition, not a chain of them.

        Args:
            operator: ``S``, from the current property space.

        Returns:
            The inner estimator followed by ``S T``.
        """
        return ComposedPointEstimator(self._estimator, operator @ self._operator)


class _PushedMeasureEstimator(MeasureEstimator):
    """A measure estimator whose every answer is pushed through an operator.

    What ``push_forward`` returns for a measure estimator with no simpler
    form for the pushed estimator: the measure for the data is computed and
    then pushed forward, which a Gaussian and a Gaussian mixture both do in
    closed form.
    """

    def __init__(
        self, estimator: MeasureEstimator, operator: LinearOperator, /
    ) -> None:
        """
        Args:
            estimator: the measure estimator, of the model or of a property.
            operator: ``T``, from the estimator's property space.

        Raises:
            ValueError: if ``T`` does not act on the estimator's property
                space.
        """
        if operator.domain != estimator.property_space:
            raise ValueError(
                f"The property operator acts on {operator.domain!r}, not on "
                f"the estimator's space {estimator.property_space!r}."
            )
        self._estimator = estimator
        self._operator = operator

    @property
    def data_space(self) -> HilbertSpace:
        """The inner estimator's data space."""
        return self._estimator.data_space

    @property
    def property_space(self) -> HilbertSpace:
        """``T``'s codomain."""
        return self._operator.codomain

    def __call__(self, data: Any, /, **kwargs: Any) -> ProbabilityMeasure:
        """The inner estimator's measure for the data, pushed through ``T``.

        Args:
            data: the observations.
            **kwargs: passed to the inner estimator's call.

        Returns:
            A measure on the property space.
        """
        return self._estimator(data, **kwargs).push_forward(self._operator)

    def push_forward(self, operator: LinearOperator, /) -> "_PushedMeasureEstimator":
        """The estimator of ``S T m``: one composition, not a chain of them."""
        return _PushedMeasureEstimator(self._estimator, operator @ self._operator)


class GaussianEstimator(MeasureEstimator):
    """A measure-valued estimator whose covariance does not depend on the data.

    Which is the case for every linear-Gaussian method: only the expectation
    moves, and affinely. So the object is a pair -- an affine expectation map
    and a fixed covariance -- and pushing it through a property operator maps
    the first and conjugates the second.
    """

    def __init__(
        self,
        expectation_operator: AffineOperator,
        covariance: LinearOperator,
        /,
        *,
        centered_sample: Any = None,
    ) -> None:
        """
        Args:
            expectation_operator: data to the posterior expectation.
            covariance: the posterior covariance, on the property space.
            centered_sample: a callable taking a generator and returning one
                draw of the *fluctuation* about the expectation. Supplied when
                the posterior can be sampled without a covariance factor --
                which for a linear Gaussian inversion it can, by
                randomize-then-optimize. Carried by the estimator so that
                :meth:`push_forward` maps it through with everything else.

        Raises:
            ValueError: if the covariance does not act on the codomain of the
                expectation operator.
        """
        if covariance.domain != expectation_operator.codomain:
            raise ValueError(
                "The covariance must act on the space the mean map lands in."
            )
        self._expectation_operator = expectation_operator
        self._covariance = covariance
        self._centered_sample = centered_sample

    @property
    def expectation_operator(self) -> AffineOperator:
        """The affine map from data to the posterior expectation."""
        return self._expectation_operator

    @property
    def covariance(self) -> LinearOperator:
        """The posterior covariance, the same for every data vector."""
        return self._covariance

    @property
    def data_space(self) -> HilbertSpace:
        """The mean map's domain."""
        return self._expectation_operator.domain

    @property
    def property_space(self) -> HilbertSpace:
        """The space the posterior lives on."""
        return self._expectation_operator.codomain

    @property
    def can_sample(self) -> bool:
        """Whether the posteriors this produces can be drawn from."""
        return self._centered_sample is not None

    def __call__(self, data: Any) -> GaussianMeasure:
        """The posterior measure for this data.

        Samplable whenever the estimator is, which for a linear Gaussian
        inversion is whenever the prior and the data errors are: the draw is
        randomize-then-optimize and needs no factor of the posterior
        covariance.
        """
        return GaussianMeasure(
            self.property_space,
            expectation=self._expectation_operator(data),
            covariance=self._covariance,
            sample=self._centered_sample,
        )

    def push_forward(self, operator: LinearOperator, /) -> "GaussianEstimator":
        """The posterior for ``T m``: expectation through ``T``, covariance conjugated.

        The covariance is the operator ``T C T*``; nothing is formed on the
        model space.

        **Which of the two to reach for.** This pushes the *estimator*, and is
        what to use for a property of data not yet seen -- one estimator, many
        datasets, one property. For a property of a posterior already in hand,
        push the *measure*: ``estimator(data).push_forward(T)`` gives the
        identical measure -- same mean, same ``T C T*``, same sampler -- for no
        solve at all, where ``estimator.push_forward(T)(data)`` solves the
        normal equations again for data that have already been inverted.

        The sampler travels with it: a draw of ``T m`` is ``T`` applied to a
        draw of ``m``. So a property posterior can be sampled exactly when the
        model posterior can, which is what a non-linear property of it needs.

        Args:
            operator: the property map ``T``.

        Returns:
            The estimator of ``T m``.
        """
        pushed = None
        if self._centered_sample is not None:

            def pushed(rng: Any, _sample: Any = self._centered_sample) -> Any:
                return operator(_sample(rng))

        # A congruence T C T* preserves self-adjointness and semidefiniteness,
        # and definiteness when T is onto; that is claimed here rather than
        # left to the composition node. The node infers it only when its
        # flattened factor list reads as a product F ... F* with the adjoints
        # mirrored about the middle; a covariance written ``(I - K A) Q``
        # contributes two factors, and the mirror image is lost.
        covariance = (operator @ self._covariance @ operator.adjoint).with_traits(
            congruence_traits(
                self._covariance.traits,
                outer_surjective=Traits.SURJECTIVE in operator.traits,
            )
        )
        return GaussianEstimator(
            operator @ self._expectation_operator, covariance, centered_sample=pushed
        )
