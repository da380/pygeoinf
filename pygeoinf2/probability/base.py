"""
Probability measures on Hilbert spaces.

The base class asks for very little — a way to draw a sample — and treats
everything else as optional: an expectation, a covariance, a log density, its
gradient. A measure that can only be sampled is still a measure, and that is
the case nonlinear inference actually presents.

Two things are groundwork for the nonlinear side rather than for the linear
one. ``grad_log_density`` returns a **vector**, not a functional, which is a
direct dividend of Riesz-identifying the spaces and is what MALA, HMC and any
gradient-based posterior exploration need. And ``push_forward`` is closed for
the Gaussian-plus-affine case but otherwise returns something that can still be
*sampled* — draw from the base, apply the map — even with no closed density.

See DESIGN.md section 7.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING, Sequence

from numpy.random import Generator

from ..algebra.direct_sum import BlockDiagonalLinearOperator, DirectSum
from ..algebra.operators import AffineOperator, LinearOperator, Operator
from ..algebra.spaces import HilbertSpace

if TYPE_CHECKING:
    pass

__all__ = ["ProbabilityMeasure", "ProductMeasure", "PushForwardMeasure", "product"]


class ProbabilityMeasure[X](ABC):
    """A probability measure on a Hilbert space."""

    def __init__(self, domain: HilbertSpace[X]) -> None:
        self._domain = domain

    @property
    def domain(self) -> HilbertSpace[X]:
        """The space the measure lives on."""
        return self._domain

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._domain!r})"

    # ----------------------------------------------------------------- #
    #                             Sampling                              #
    # ----------------------------------------------------------------- #

    @abstractmethod
    def sample(self, *, rng: Generator | None = None) -> X:
        """Draw one sample.

        The generator is always explicit. v1 draws from NumPy's legacy global
        state, so no result involving sampling is reproducible.
        """

    def samples(self, n: int, *, rng: Generator | None = None) -> list[X]:
        """Draw ``n`` independent samples."""
        if n < 0:
            raise ValueError("n must be non-negative.")
        return [self.sample(rng=rng) for _ in range(n)]

    def sample_expectation(self, n: int, *, rng: Generator | None = None) -> X:
        """The sample mean of ``n`` draws."""
        return self._domain.mean(self.samples(n, rng=rng))

    # ----------------------------------------------------------------- #
    #                              Moments                              #
    # ----------------------------------------------------------------- #

    @property
    def expectation(self) -> X | None:
        """The mean, or None when it is not available in closed form."""
        return None

    @property
    def covariance(self) -> LinearOperator[X, X] | None:
        """The covariance operator, or None when unavailable.

        When present it is self-adjoint and positive semidefinite, and says so
        in its traits.
        """
        return None

    @property
    def has_expectation(self) -> bool:
        """True when the mean is available in closed form."""
        return self.expectation is not None

    @property
    def has_covariance(self) -> bool:
        """True when the covariance is available in closed form."""
        return self.covariance is not None

    # ----------------------------------------------------------------- #
    #                             Densities                             #
    # ----------------------------------------------------------------- #

    def directional_covariance(self, u: Any, v: Any, /) -> float:
        """``Cov((x, u), (x, v))``, the covariance of two linear readings."""
        return self.domain.inner_product(self.covariance(u), v)

    def directional_variance(self, u: Any, /) -> float:
        """``Var((x, u))``, the variance of one linear reading."""
        return self.directional_covariance(u, u)

    def two_point_covariance(self, point: Any, /) -> Any:
        """The covariance function anchored at a point, as a field.

        ``y -> Cov(x(point), x(y))``. It is ``C u`` with ``u`` the representer
        of evaluation at ``point``, because ``(C u, u_y)`` is exactly the value
        of ``C u`` at ``y`` — so the whole function comes from one application
        of the covariance rather than one per pair.

        Needs a domain whose points can be evaluated at, so it is defined for a
        space of functions and not for a space of coefficients.
        """
        space = self.domain
        dirac = getattr(space, "dirac", None)
        if dirac is None:
            raise TypeError(
                f"{type(space).__name__} has no evaluation functional, so a "
                "two-point covariance is not defined on it."
            )
        return self.covariance(dirac(point).representer)

    def log_density(self, x: X) -> float:
        """The log density at ``x``, up to an additive constant."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a log density."
        )

    def grad_log_density(self, x: X) -> X:
        """The gradient of the log density — a **vector** in the domain.

        Not a functional: under Riesz identification the representer is the
        natural object, and it is what a gradient-based sampler steps along.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a log-density gradient."
        )

    @property
    def has_log_density(self) -> bool:
        """True when a log density is available."""
        return type(self).log_density is not ProbabilityMeasure.log_density

    @property
    def has_grad_log_density(self) -> bool:
        """True when a log-density gradient is available."""
        return type(self).grad_log_density is not ProbabilityMeasure.grad_log_density

    # ----------------------------------------------------------------- #
    #                          Transformations                          #
    # ----------------------------------------------------------------- #

    def affine_map(
        self,
        operator: LinearOperator,
        /,
        *,
        translation: object | None = None,
    ) -> ProbabilityMeasure:
        """The law of ``A X + b``."""
        if operator.domain != self._domain:
            raise ValueError(
                f"Cannot map: the operator's domain {operator.domain!r} is not "
                f"the measure's domain {self._domain!r}."
            )
        specialised = self._combine_affine(operator, translation)
        if specialised is not None:
            return specialised
        return PushForwardMeasure(
            self,
            operator if translation is None else AffineOperator(operator, translation),
        )

    def push_forward(self, operator: Operator) -> ProbabilityMeasure:
        """The law of ``F(X)``.

        Exact for a Gaussian under an affine map. Otherwise the result can
        still be sampled, which is the minimum nonlinear inference needs.
        """
        if isinstance(operator, AffineOperator):
            return self.affine_map(
                operator.linear_part, translation=operator.translation
            )
        if isinstance(operator, LinearOperator):
            return self.affine_map(operator)
        if operator.domain != self._domain:
            raise ValueError(
                f"Cannot push forward: the operator's domain "
                f"{operator.domain!r} is not the measure's domain "
                f"{self._domain!r}."
            )
        return PushForwardMeasure(self, operator)

    def __rmatmul__(self, operator: Operator) -> ProbabilityMeasure:
        """``A @ mu`` is the law of ``A X``."""
        if not isinstance(operator, Operator):
            return NotImplemented
        return self.push_forward(operator)

    def translate(self, vector: X) -> ProbabilityMeasure:
        """The law of ``X + v``."""
        return self.affine_map(
            LinearOperator.identity(self._domain), translation=vector
        )

    # ----------------------------------------------------------------- #
    #                      Specialisation protocol                      #
    # ----------------------------------------------------------------- #
    #
    # As for operators: returning None means "no specialisation, fall back".
    # A measure family that is closed under these operations -- an invariant
    # Gaussian on a symmetric space, say -- overrides them to stay in its
    # class, because degrading to a generic measure loses the closed-form
    # divergences and norms that made the family worth having.

    def _combine_affine(
        self, operator: LinearOperator, translation: object | None
    ) -> ProbabilityMeasure | None:
        """Specialise ``A X + b``. Return None to fall back to a pushforward."""
        return None

    def _combine_add(self, other: ProbabilityMeasure) -> ProbabilityMeasure | None:
        return None

    def _combine_scale(self, alpha: float) -> ProbabilityMeasure | None:
        return None

    # ----------------------------------------------------------------- #
    #                              Algebra                              #
    # ----------------------------------------------------------------- #

    def __add__(self, other: ProbabilityMeasure) -> ProbabilityMeasure:
        """The law of ``X + Y`` for **independent** ``X`` and ``Y``."""
        if not isinstance(other, ProbabilityMeasure):
            return NotImplemented
        if self._domain != other.domain:
            raise ValueError("Measures must share a domain to be added.")
        for candidate in (self._combine_add(other), other._combine_add(self)):
            if candidate is not None:
                return candidate
        return _IndependentSum(self, other)

    def __sub__(self, other: ProbabilityMeasure) -> ProbabilityMeasure:
        if not isinstance(other, ProbabilityMeasure):
            return NotImplemented
        return self + (-1.0 * other)

    def __mul__(self, alpha: float) -> ProbabilityMeasure:
        """The law of ``alpha X``, so the covariance scales by ``alpha^2``."""
        if not isinstance(alpha, (int, float)):
            return NotImplemented
        specialised = self._combine_scale(float(alpha))
        if specialised is not None:
            return specialised
        return self.affine_map(float(alpha) * LinearOperator.identity(self._domain))

    def __rmul__(self, alpha: float) -> ProbabilityMeasure:
        return self.__mul__(alpha)

    def __truediv__(self, alpha: float) -> ProbabilityMeasure:
        if alpha == 0.0:
            raise ZeroDivisionError("Cannot divide a measure by zero.")
        return self.__mul__(1.0 / float(alpha))


class PushForwardMeasure[X, Y](ProbabilityMeasure[Y]):
    """The law of ``F(X)`` for a general operator ``F``.

    Samples, and nothing else. There is no closed-form density, and the
    moments are not claimed — asking for them returns ``None`` rather than a
    silently approximate answer. Use ``sample_expectation`` when an empirical
    estimate is what is wanted.
    """

    def __init__(self, base: ProbabilityMeasure[X], operator: Operator) -> None:
        super().__init__(operator.codomain)
        self._base = base
        self._operator = operator

    @property
    def base(self) -> ProbabilityMeasure[X]:
        """The measure being pushed forward."""
        return self._base

    @property
    def operator(self) -> Operator:
        """The map being applied."""
        return self._operator

    def sample(self, *, rng: Generator | None = None) -> Y:
        """Draw from the base measure and apply the map."""
        return self._operator(self._base.sample(rng=rng))

    def __repr__(self) -> str:
        return f"PushForwardMeasure({self._operator!r})"


class _IndependentSum[X](ProbabilityMeasure[X]):
    """The law of ``X + Y`` for independent measures with no shared structure."""

    def __init__(
        self, left: ProbabilityMeasure[X], right: ProbabilityMeasure[X]
    ) -> None:
        super().__init__(left.domain)
        self._left = left
        self._right = right

    def sample(self, *, rng: Generator | None = None) -> X:
        """Draw from each summand independently and add."""
        return self._domain.add(self._left.sample(rng=rng), self._right.sample(rng=rng))

    @property
    def expectation(self) -> X | None:
        """The sum of the summands' means, when both are available."""
        if not (self._left.has_expectation and self._right.has_expectation):
            return None
        return self._domain.add(self._left.expectation, self._right.expectation)

    @property
    def covariance(self) -> LinearOperator[X, X] | None:
        """The sum of the summands' covariances, the parts being independent."""
        if not (self._left.has_covariance and self._right.has_covariance):
            return None
        return self._left.covariance + self._right.covariance


class ProductMeasure[X](ProbabilityMeasure[tuple]):
    """Independent factors on a direct sum.

    v1 has this only for Gaussians (``GaussianMeasure.from_direct_sum``), but
    the independent product of *any* samplable measures is samplable — which is
    what the joint model needs when the prior is not Gaussian, and so is
    exactly the case the nonlinear work runs into.
    """

    def __init__(
        self,
        factors: Sequence[ProbabilityMeasure],
        /,
        *,
        labels: Sequence[str] | None = None,
    ) -> None:
        factors = tuple(factors)
        if not factors:
            raise ValueError("A product measure needs at least one factor.")
        super().__init__(DirectSum([f.domain for f in factors], labels=labels))
        self._factors = factors

    @property
    def factors(self) -> tuple[ProbabilityMeasure, ...]:
        """The factors, in order."""
        return self._factors

    def sample(self, *, rng: Generator | None = None) -> tuple:
        """One independent draw from each factor."""
        return tuple(factor.sample(rng=rng) for factor in self._factors)

    @property
    def expectation(self) -> tuple | None:
        """The factors' means, when every factor has one."""
        if not all(factor.has_expectation for factor in self._factors):
            return None
        return tuple(factor.expectation for factor in self._factors)

    @property
    def covariance(self) -> LinearOperator | None:
        """Block diagonal, because the factors are independent."""
        if not all(factor.has_covariance for factor in self._factors):
            return None
        return BlockDiagonalLinearOperator(
            [factor.covariance for factor in self._factors]
        )

    def factor(self, key: int | str) -> ProbabilityMeasure:
        """The factor at a label or index."""
        return self._factors[self._domain.index(key)]

    def __repr__(self) -> str:
        return f"ProductMeasure({len(self._factors)} factors)"


def product(
    measures: Sequence[ProbabilityMeasure],
    /,
    *,
    labels: Sequence[str] | None = None,
) -> ProbabilityMeasure:
    """The independent product of measures, collapsed where it can be.

    A product of Gaussians is Gaussian, with block-diagonal covariance; anything
    else is a :class:`ProductMeasure`.
    """
    from .gaussian import GaussianMeasure

    measures = tuple(measures)
    if measures and all(isinstance(m, GaussianMeasure) for m in measures):
        return GaussianMeasure.from_product(measures, labels=labels)
    return ProductMeasure(measures, labels=labels)
