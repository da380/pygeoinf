"""
An estimator is a mapping from data to an answer.

Three kinds, distinguished only by what ``__call__`` returns: a point, a
measure, or a set. Which kind you get is fixed by the *prior* — no prior gives
a point, a distribution gives a distribution, a constraint set gives a
constraint set — and that rule is what says these three exist and stops a
fourth being invented.

The target is the model space, or a property space reached by an operator
``T``. An inverse problem is an inference problem with ``T == identity``, so
:meth:`Estimator.push_forward` is the whole of the difference: applying ``T``
to a point, pushing a measure through it, taking the image of a set.

See DESIGN.md section 18.7.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..algebra.operators import AffineOperator, LinearOperator
from ..algebra.spaces import HilbertSpace
from ..probability.base import ProbabilityMeasure
from ..probability.gaussian import GaussianMeasure

__all__ = [
    "Estimator",
    "PointEstimator",
    "MeasureEstimator",
    "SetEstimator",
    "LinearPointEstimator",
    "GaussianEstimator",
]


class Estimator(ABC):
    """A mapping from data to an answer about the target."""

    @property
    @abstractmethod
    def data_space(self) -> HilbertSpace:
        """The space the data live in."""

    @property
    @abstractmethod
    def target_space(self) -> HilbertSpace:
        """The space the answer is about — the model space, or a property space."""

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """The answer, given data."""

    @abstractmethod
    def push_forward(self, operator: LinearOperator, /) -> "Estimator":
        """The estimator of a property of the model, rather than the model."""


class PointEstimator(Estimator):
    """An estimator returning a single vector of the target space."""


class MeasureEstimator(Estimator):
    """An estimator returning a probability measure on the target space."""


class SetEstimator(Estimator):
    """An estimator returning a subset of the target space."""


class LinearPointEstimator(AffineOperator):
    """A point estimator that is affine in the data, so also an operator.

    It joins the existing algebra rather than wrapping it, and carries what the
    method actually produces alongside the estimate itself:

    ``resolution``
        ``B A``, the averaging kernel — how the estimate sees the true model.
        This is the *output* of a Backus-Gilbert method, not a diagnostic
        added afterwards.
    ``propagated_covariance``
        ``B R B*``, the data error mapped into the target space.
    """

    def __init__(
        self,
        operator: LinearOperator,
        /,
        *,
        translation: Any = None,
        forward_operator: LinearOperator | None = None,
        error: ProbabilityMeasure | None = None,
    ) -> None:
        """
        Args:
            operator: ``B``, mapping data to the target space.
            translation: an offset added to ``B d``; zero if omitted.
            forward_operator: ``A``, kept so ``resolution`` can be formed.
            error: the data error, kept so its covariance can be propagated.
        """
        super().__init__(
            operator,
            operator.codomain.zero() if translation is None else translation,
        )
        self._operator = operator
        self._forward_operator = forward_operator
        self._error = error

    @property
    def operator(self) -> LinearOperator:
        """``B`` itself, without the translation."""
        return self._operator

    @property
    def data_space(self) -> HilbertSpace:
        """``B``'s domain."""
        return self._operator.domain

    @property
    def target_space(self) -> HilbertSpace:
        """``B``'s codomain."""
        return self._operator.codomain

    @property
    def resolution(self) -> LinearOperator:
        """``B A``: what the estimate reports when the truth is ``m``."""
        if self._forward_operator is None:
            raise AttributeError(
                "This estimator was built without a forward operator, so it "
                "cannot form its resolution."
            )
        return self._operator @ self._forward_operator

    def propagated_covariance(self) -> LinearOperator:
        """``B R B*``: the data error, as a covariance on the target space."""
        if self._error is None:
            raise AttributeError(
                "This estimator was built without a data error measure."
            )
        return self._operator @ self._error.covariance @ self._operator.adjoint

    def push_forward(self, operator: LinearOperator, /) -> "LinearPointEstimator":
        """The estimator of ``T m`` rather than of ``m``."""
        return LinearPointEstimator(
            operator @ self._operator,
            translation=operator(self.translation),
            forward_operator=self._forward_operator,
            error=self._error,
        )

    def as_measure(self) -> "GaussianEstimator":
        """The measure-valued estimator this one induces.

        Mean ``B d``, covariance ``B R B*``. One of the bridges of §18.7: a
        linear point estimator, a Gaussian estimator and a set estimator are
        three readings of the same object, not three implementations.
        """
        return GaussianEstimator(self, self.propagated_covariance())


PointEstimator.register(LinearPointEstimator)


class GaussianEstimator(MeasureEstimator):
    """A measure-valued estimator whose covariance does not depend on the data.

    Which is the case for every linear-Gaussian method: only the mean moves,
    and affinely. So the object is a pair, and pushing it through a property
    operator is one line.
    """

    def __init__(self, mean_map: AffineOperator, covariance: LinearOperator, /) -> None:
        """
        Args:
            mean_map: data to the posterior mean.
            covariance: the posterior covariance, on the target space.
        """
        if covariance.domain != mean_map.codomain:
            raise ValueError(
                "The covariance must act on the space the mean map lands in."
            )
        self._mean_map = mean_map
        self._covariance = covariance

    @property
    def mean_map(self) -> AffineOperator:
        """The affine map from data to the posterior mean."""
        return self._mean_map

    @property
    def covariance(self) -> LinearOperator:
        """The posterior covariance, the same for every data vector."""
        return self._covariance

    @property
    def data_space(self) -> HilbertSpace:
        """The mean map's domain."""
        return self._mean_map.domain

    @property
    def target_space(self) -> HilbertSpace:
        """The space the posterior lives on."""
        return self._mean_map.codomain

    def __call__(self, data: Any) -> GaussianMeasure:
        """The posterior measure for this data."""
        return GaussianMeasure(
            self.target_space,
            expectation=self._mean_map(data),
            covariance=self._covariance,
        )

    def push_forward(self, operator: LinearOperator, /) -> "GaussianEstimator":
        """The posterior for ``T m``: mean through ``T``, covariance conjugated.

        Worth reaching for even when the model-space posterior is available,
        since ``T C T*`` on a small property space is cheaper than forming
        ``C`` on the model space at all.
        """
        return GaussianEstimator(
            operator @ self._mean_map,
            operator @ self._covariance @ operator.adjoint,
        )
