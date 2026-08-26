"""
Convex subsets, and the three views of one.

A convex set, its indicator functional and its support function are the same
object seen from three directions, and this module makes that explicit:

- ``set.contains(x)`` — the set as a predicate;
- ``set.indicator()`` — a ``Functional`` whose proximal operator *is* the
  projection, so a convex set drops straight into ``ProximalGradient``;
- ``set.support_function()`` — the convex-analysis view, with its own closed
  algebra.

v1 keeps these in three places: the sets in ``subsets``, the support functions
in ``convex_analysis``, and nothing that ties an indicator to a proximal
method. Tying them together is most of the modernisation.

``project`` here means the **metric projection**: the nearest point of the set,
which leaves a point already inside where it is. That is what a proximal method
needs. v1's ``HalfSpace.project`` instead goes to the bounding hyperplane
whichever side the point is on, which is a different map and is not idempotent
— using it as a projection would quietly move feasible points.

Everything with a closed form is coordinate-free, because the closed forms are
written with norms and inner products. Where there is no closed form — the
projection onto a general ellipsoid, for instance — the set says so rather than
offering an approximation under the same name.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from ..algebra.operators import Functional, LinearOperator
from ..algebra.spaces import HilbertSpace
from ..numerics.convex import SupportFunction
from ..traits import Traits
from .sets import Subset

__all__ = [
    "ConvexSet",
    "Ball",
    "HalfSpace",
    "Hyperplane",
    "Ellipsoid",
]


class ConvexSet(Subset):
    """A closed convex subset, which knows its own nearest point."""

    @abstractmethod
    def project(self, x: Any, /) -> Any:
        """The nearest point of the set, in the space's own norm.

        Idempotent, and the identity on points already inside. Both properties
        are what make it usable as a proximal operator, and both are checked by
        ``testing.check_projection``.
        """

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """True when the point is its own projection.

        A default that every convex set inherits: a point is in a closed set
        exactly when the nearest point of that set is itself. Subsets with a
        cheaper test override it.
        """
        distance = self._domain.norm(self._domain.subtract(x, self.project(x)))
        return distance <= rtol * max(self._domain.norm(x), 1.0)

    def support_function(self) -> SupportFunction:
        """``h(y) == sup { (y, x) : x in this set }``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a support function."
        )

    def indicator(self) -> Functional:
        """The functional that is zero on the set and infinite off it.

        Its proximal operator is the projection, so this is how a hard
        constraint enters ``ProximalGradient`` or any other proximal method.
        """
        return _SetIndicator(self)


class _SetIndicator(Functional):
    """The indicator of a convex set, whose prox is the set's projection."""

    def __init__(self, subset: ConvexSet, /) -> None:
        super().__init__(subset.domain)
        self._subset = subset

    @property
    def subset(self) -> ConvexSet:
        """The set being indicated."""
        return self._subset

    def _value(self, x: Any) -> float:
        return 0.0 if self._subset.contains(x) else float("inf")

    @property
    def has_prox(self) -> bool:
        """True: the projection is the proximal operator."""
        return True

    def prox(self, x: Any, step: float, /) -> Any:
        """The projection, which does not depend on the step."""
        return self._subset.project(x)

    def conjugate(self) -> Functional:
        """The set's support function, which is the conjugate of its indicator."""
        return self._subset.support_function()

    def __repr__(self) -> str:
        return f"Indicator({self._subset!r})"


class Ball(ConvexSet):
    """``{ x : ||x - centre|| <= radius }``."""

    def __init__(
        self,
        domain: HilbertSpace,
        /,
        *,
        radius: float = 1.0,
        centre: Any = None,
    ) -> None:
        """
        Args:
            domain: the space.
            radius: the radius, which must be positive.
            centre: the centre. Defaults to zero.
        """
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        super().__init__(domain)
        self._radius = float(radius)
        self._centre = domain.zero() if centre is None else centre

    def translate(self, vector: Any, /) -> "Ball":
        """The same ball, moved by a vector."""
        return Ball(
            self.domain,
            radius=self._radius,
            centre=self.domain.add(self._centre, vector),
        )

    @property
    def radius(self) -> float:
        """The radius."""
        return self._radius

    @property
    def centre(self) -> Any:
        """The centre."""
        return self._centre

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """Cheaper than the default: one norm rather than a projection."""
        offset = self._domain.norm(self._domain.subtract(x, self._centre))
        return offset <= self._radius * (1.0 + rtol)

    def project(self, x: Any, /) -> Any:
        """Rescale the offset from the centre to the radius, if it exceeds it."""
        space = self._domain
        offset = space.subtract(x, self._centre)
        distance = space.norm(offset)
        if distance <= self._radius:
            return space.copy(x)
        return space.axpy(self._radius / distance, offset, space.copy(self._centre))

    def support_function(self) -> SupportFunction:
        """``radius ||y|| + (centre, y)``."""
        return SupportFunction.of_ball(
            self._domain, radius=self._radius, centre=self._centre
        )

    def indicator(self) -> Functional:
        """The ball's indicator, with its closed-form proximal operator."""
        from ..numerics.convex import BallIndicator

        return BallIndicator(self._domain, radius=self._radius, centre=self._centre)

    def __repr__(self) -> str:
        return f"Ball(radius={self._radius})"


class Hyperplane(ConvexSet):
    """``{ x : (normal, x) == offset }``.

    Convex, closed, and with a projection in closed form — which is why it
    belongs here rather than with the general subsets.
    """

    def __init__(
        self, domain: HilbertSpace, normal: Any, /, *, offset: float = 0.0
    ) -> None:
        """
        Args:
            domain: the space.
            normal: the normal vector, which must be nonzero.
            offset: the level.
        """
        super().__init__(domain)
        squared = domain.squared_norm(normal)
        if squared == 0.0:
            raise ValueError("The normal vector must be nonzero.")
        self._normal = normal
        self._offset = float(offset)
        self._squared_norm = squared

    @property
    def normal(self) -> Any:
        """The normal vector."""
        return self._normal

    @property
    def offset(self) -> float:
        """The level the normal is set equal to."""
        return self._offset

    def _residual(self, x: Any) -> float:
        return self._domain.inner_product(self._normal, x) - self._offset

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """True when the point satisfies the equation to tolerance."""
        scale = max(
            abs(self._offset), self._domain.norm(x) * np.sqrt(self._squared_norm), 1.0
        )
        return abs(self._residual(x)) <= rtol * scale

    def project(self, x: Any, /) -> Any:
        """Remove the component of the residual along the normal."""
        space = self._domain
        return space.axpy(
            -self._residual(x) / self._squared_norm, self._normal, space.copy(x)
        )

    def __repr__(self) -> str:
        return f"Hyperplane(offset={self._offset})"


class HalfSpace(ConvexSet):
    """``{ x : (normal, x) <= offset }``."""

    def __init__(
        self, domain: HilbertSpace, normal: Any, /, *, offset: float = 0.0
    ) -> None:
        """
        Args:
            domain: the space.
            normal: the outward normal, which must be nonzero.
            offset: the level.
        """
        super().__init__(domain)
        squared = domain.squared_norm(normal)
        if squared == 0.0:
            raise ValueError("The normal vector must be nonzero.")
        self._normal = normal
        self._offset = float(offset)
        self._squared_norm = squared

    @property
    def normal(self) -> Any:
        """The outward normal."""
        return self._normal

    @property
    def offset(self) -> float:
        """The level the normal is bounded by."""
        return self._offset

    @property
    def boundary(self) -> Hyperplane:
        """The bounding hyperplane."""
        return Hyperplane(self._domain, self._normal, offset=self._offset)

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """True when the inequality holds to tolerance."""
        residual = self._domain.inner_product(self._normal, x) - self._offset
        scale = max(abs(self._offset), 1.0)
        return residual <= rtol * scale

    def project(self, x: Any, /) -> Any:
        """The metric projection, which leaves feasible points alone.

        Note the ``max(0, .)``: a point already inside is returned unchanged.
        Projecting onto the boundary regardless — which is what v1 does — is a
        different map, is not idempotent, and would move feasible points in a
        proximal iteration.
        """
        space = self._domain
        excess = space.inner_product(self._normal, x) - self._offset
        if excess <= 0.0:
            return space.copy(x)
        return space.axpy(-excess / self._squared_norm, self._normal, space.copy(x))

    def __repr__(self) -> str:
        return f"HalfSpace(offset={self._offset})"


class Ellipsoid(ConvexSet):
    """``{ x : (x - centre, P (x - centre)) <= 1 }`` for a positive definite ``P``.

    Given by its **precision** rather than its covariance, because that is what
    a membership test needs and what a credible region is naturally expressed
    with: the set of points whose Mahalanobis distance is at most one.

    A support function is available in closed form given the covariance. A
    *projection* is not: it requires solving a scalar secular equation with a
    linear solve at every step, so it is not offered under the same name as the
    closed forms elsewhere in this module.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        precision: LinearOperator,
        /,
        *,
        centre: Any = None,
        covariance: LinearOperator | None = None,
    ) -> None:
        """
        Args:
            domain: the space.
            precision: a positive definite operator on it.
            centre: the centre. Defaults to zero.
            covariance: the inverse of the precision, if it is known. Supplying
                it is what makes the support function available.
        """
        super().__init__(domain)
        required = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        if required & precision.traits != required:
            raise ValueError(
                f"The precision must claim {required!s}; it claims "
                f"{precision.traits!s}."
            )
        if precision.domain != domain or precision.codomain != domain:
            raise ValueError(f"The precision must be an operator on {domain!r}.")
        self._precision = precision
        self._covariance = covariance
        self._centre = domain.zero() if centre is None else centre

    def translate(self, vector: Any, /) -> "Ellipsoid":
        """The same ellipsoid, moved by a vector."""
        return Ellipsoid(
            self.domain,
            self._precision,
            centre=self.domain.add(self._centre, vector),
            covariance=self._covariance,
        )

    @property
    def precision(self) -> LinearOperator:
        """The operator defining the Mahalanobis distance."""
        return self._precision

    @property
    def centre(self) -> Any:
        """The centre."""
        return self._centre

    def mahalanobis_squared(self, x: Any, /) -> float:
        """``(x - centre, P (x - centre))``."""
        offset = self._domain.subtract(x, self._centre)
        return self._domain.inner_product(self._precision(offset), offset)

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """True when the Mahalanobis distance is at most one."""
        return self.mahalanobis_squared(x) <= 1.0 + rtol

    def project(self, x: Any, /) -> Any:
        """Not available in closed form.

        The nearest point on an ellipsoid satisfies a secular equation in a
        scalar multiplier, and each evaluation of it needs a linear solve. That
        is a genuine algorithm rather than a formula, so it is not offered here
        under the same name as the closed forms; use a proximal method on the
        indicator's smooth surrogate, or intersect with a ball instead.
        """
        raise NotImplementedError(
            "An ellipsoid has no closed-form projection: it requires solving a "
            "secular equation with a linear solve per iteration. Ball, "
            "HalfSpace and Hyperplane do have one."
        )

    def support_function(self) -> SupportFunction:
        """``(centre, y) + sqrt((y, C y))`` with ``C`` the covariance."""
        if self._covariance is None:
            raise NotImplementedError(
                "An ellipsoid's support function needs its covariance, the "
                "inverse of the precision. Pass covariance= to supply it."
            )
        return _EllipsoidSupport(self._domain, self._covariance, self._centre)

    def __repr__(self) -> str:
        return f"Ellipsoid({self._domain!r})"


class _EllipsoidSupport(SupportFunction):
    """The support function of an ellipsoid given by its covariance."""

    def __init__(
        self, domain: HilbertSpace, covariance: LinearOperator, centre: Any, /
    ) -> None:
        super().__init__(domain)
        self._covariance = covariance
        self._centre = centre

    def _value(self, y: Any) -> float:
        quadratic = self._domain.inner_product(self._covariance(y), y)
        return self._domain.inner_product(self._centre, y) + float(
            np.sqrt(max(quadratic, 0.0))
        )

    def _maximiser(self, y: Any) -> Any:
        space = self._domain
        mapped = self._covariance(y)
        norm = np.sqrt(max(space.inner_product(mapped, y), 0.0))
        if norm == 0.0:
            return space.copy(self._centre)
        return space.axpy(1.0 / norm, mapped, space.copy(self._centre))
