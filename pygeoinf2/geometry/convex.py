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
    "Polytope",
    "BallSurface",
    "EllipsoidSurface",
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

    def support_maximiser(self, direction: Any, /) -> Any:
        """The point of the set attaining ``h(direction)``.

        A *subgradient* of the support function, and the reason it is worth
        having: a nonsmooth minimisation of a support function needs one at
        every step, and for the sets that have closed forms so does this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a support maximiser."
        )

    def indicator(self) -> Functional:
        """The functional that is zero on the set and infinite off it.

        Its proximal operator is the projection, so this is how a hard
        constraint enters ``ProximalGradient`` or any other proximal method.
        """
        return _SetIndicator(self)

    @classmethod
    def from_support_function(
        cls,
        domain: HilbertSpace,
        oracle: Any,
        /,
        *,
        maximiser: Any = None,
    ) -> "ConvexSet":
        """A convex set given only by its support function.

        A closed convex set is *determined* by its support function
        (Rockafellar 13.1-13.2), so this loses nothing in principle. What it
        loses in practice is cheapness: every question has to go through the
        oracle, and one call may be an optimisation.

        This is how the feasible property set of §18.3 arrives. Its support
        function is a dual minimisation and there is no other description of
        it, so the set is the oracle.

        Args:
            domain: the space.
            oracle: called with a direction, returning ``h(q)``.
            maximiser: optionally, called with a direction and returning the
                point of the set attaining the supremum. Supplying it is what
                makes an *inner* approximation available as well as an outer
                one.
        """
        return _OracleSet(domain, oracle, maximiser=maximiser)

    def __add__(self, other: Any) -> Any:
        """The Minkowski sum, whose support function is the sum of theirs.

        ``h_{A+B} == h_A + h_B``, exactly, which is the one operation on convex
        sets that support functions make trivial. Route (b) of §18.3 returns a
        sum of two ellipsoids and has no simpler description.
        """
        if not isinstance(other, ConvexSet):
            return NotImplemented
        if other.domain != self.domain:
            raise ValueError("Both sets must live on the same space.")
        return _MinkowskiSum(self, other)

    def translate(self, vector: Any, /) -> "ConvexSet":
        """The set moved by a vector."""
        return _Translated(self, vector)


class _Translated(ConvexSet):
    """A convex set moved by a vector."""

    def __init__(self, base: ConvexSet, vector: Any, /) -> None:
        super().__init__(base.domain)
        self._base = base
        self._vector = vector

    def project(self, x: Any, /) -> Any:
        """Move in, project, move back."""
        inner = self._base.project(self.domain.subtract(x, self._vector))
        return self.domain.add(inner, self._vector)

    def support_function(self) -> SupportFunction:
        """``h_{K+v}(q) == h_K(q) + (v, q)``."""
        return _ShiftedSupport(self._base.support_function(), self._vector)

    def __repr__(self) -> str:
        return f"Translated({self._base!r})"


class _ShiftedSupport(SupportFunction):
    """The support function of a translated set."""

    def __init__(self, base: SupportFunction, vector: Any, /) -> None:
        super().__init__(base.domain)
        self._base = base
        self._vector = vector

    def _value(self, y: Any) -> float:
        return self._base(y) + self.domain.inner_product(self._vector, y)


class _OracleSet(ConvexSet):
    """A convex set known only through its support function."""

    def __init__(self, domain: HilbertSpace, oracle: Any, /, *, maximiser: Any) -> None:
        super().__init__(domain)
        self._oracle = oracle
        self._maximiser = maximiser

    @property
    def has_maximiser(self) -> bool:
        """Whether a point attaining the supremum can be produced."""
        return self._maximiser is not None

    def maximiser(self, direction: Any, /) -> Any:
        """The point of the set furthest along a direction."""
        if self._maximiser is None:
            raise AttributeError(
                "This set was given a support function but no maximiser, so it "
                "can bound itself from outside but not exhibit a point."
            )
        return self._maximiser(direction)

    def support_function(self) -> SupportFunction:
        """The oracle, as a functional."""
        return _OracleSupport(self.domain, self._oracle)

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """Not decidable from a support function alone.

        ``(q, x) <= h(q)`` for *every* ``q`` is membership, and no finite
        number of directions establishes it. What can be had is a certificate
        of *non*-membership — see :meth:`outside`.
        """
        raise NotImplementedError(
            "Membership needs every direction. Use outside() for a "
            "certificate of exclusion, or polytope() for an outer bound."
        )

    def outside(self, x: Any, directions: Any, /, *, rtol: float = 1e-9) -> bool:
        """True when some direction proves the point is not in the set.

        One-sided, and honestly so: a ``True`` is a proof, and a ``False`` is
        only the absence of one among the directions tried.
        """
        for direction in directions:
            pairing = self.domain.inner_product(direction, x)
            if pairing > self._oracle(direction) * (1.0 + rtol) + rtol:
                return True
        return False

    def project(self, x: Any, /) -> Any:
        """Not available from a support function alone."""
        raise NotImplementedError(
            "A support function does not give a projection. Bound the set with "
            "polytope() and project onto that."
        )

    def polytope(self, directions: Any, /) -> "Polytope":
        """A certified outer bound, one half-space per direction (§18.4)."""
        return Polytope(
            self.domain,
            [
                HalfSpace(self.domain, direction, offset=self._oracle(direction))
                for direction in directions
            ],
            outer=True,
        )

    def __repr__(self) -> str:
        return f"ConvexSet.from_support_function({self.domain!r})"


class _OracleSupport(SupportFunction):
    """A support function that is whatever the oracle says."""

    def __init__(self, domain: HilbertSpace, oracle: Any, /) -> None:
        super().__init__(domain)
        self._oracle = oracle

    def _value(self, y: Any) -> float:
        return float(self._oracle(y))


class _MinkowskiSum(ConvexSet):
    """``A + B``, known through the sum of their support functions."""

    def __init__(self, first: ConvexSet, second: ConvexSet, /) -> None:
        super().__init__(first.domain)
        self._first = first
        self._second = second

    @property
    def summands(self) -> tuple[ConvexSet, ConvexSet]:
        """The two sets being added."""
        return self._first, self._second

    def support_function(self) -> SupportFunction:
        """``h_A + h_B``, exactly."""
        return self._first.support_function() + self._second.support_function()

    def project(self, x: Any, /) -> Any:
        """Not generally available: the sum of two projections is not one."""
        raise NotImplementedError(
            "A Minkowski sum has no projection in closed form, even when its "
            "summands do. Bound it with a polytope from its support function."
        )

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """Not generally decidable without an optimisation."""
        raise NotImplementedError(
            "Membership of a Minkowski sum needs an optimisation over the "
            "splitting of the point between the summands."
        )

    def __repr__(self) -> str:
        return f"MinkowskiSum({self._first!r}, {self._second!r})"


class Polytope(ConvexSet):
    """An intersection of half-spaces, which remembers which side it is on.

    §18.4's requirement made into a type. A polytope built from support
    function values *contains* the set it approximates; one built as the convex
    hull of feasible points is *contained* in it. Reporting the second as
    though it were the answer is the mistake BGP's Figure 4 is about, and the
    two are not interchangeable — so ``outer`` is not optional.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        half_spaces: Any,
        /,
        *,
        outer: bool,
    ) -> None:
        """
        Args:
            domain: the space.
            half_spaces: the constraints, as :class:`HalfSpace` objects.
            outer: ``True`` if this contains the set it approximates,
                ``False`` if it is contained in it.
        """
        super().__init__(domain)
        self._half_spaces = tuple(half_spaces)
        if not self._half_spaces:
            raise ValueError("A polytope needs at least one half-space.")
        if any(plane.domain != domain for plane in self._half_spaces):
            raise ValueError("Every half-space must live on the same space.")
        self._outer = bool(outer)

    @property
    def half_spaces(self) -> tuple:
        """The constraints."""
        return self._half_spaces

    @property
    def is_outer(self) -> bool:
        """Whether this contains the set it approximates, or is contained by it."""
        return self._outer

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """Whether the point satisfies every constraint."""
        return all(
            plane.contains(x, rtol=rtol) if hasattr(plane, "contains") else False
            for plane in self._half_spaces
        )

    def project(self, x: Any, /, *, iterations: int = 200) -> Any:
        """The nearest point, by cyclic projection onto the half-spaces.

        Dykstra's algorithm would be needed for the exact nearest point;
        alternating projection converges to *a* point of the intersection,
        which is what a feasibility question wants. The distinction is named
        rather than hidden.
        """
        current = x
        for _ in range(iterations):
            previous = current
            for plane in self._half_spaces:
                current = plane.project(current)
            if self.domain.norm(self.domain.subtract(current, previous)) <= 1e-12 * max(
                self.domain.norm(current), 1.0
            ):
                break
        return current

    def __and__(self, other: "Polytope") -> "Polytope":
        """Both sets of constraints, which tightens an outer bound."""
        if self._outer != other.is_outer:
            raise ValueError(
                "An outer and an inner polytope cannot be intersected: the "
                "result would bound nothing."
            )
        return Polytope(
            self.domain,
            self._half_spaces + other.half_spaces,
            outer=self._outer,
        )

    def __repr__(self) -> str:
        side = "outer" if self._outer else "inner"
        return f"Polytope({len(self._half_spaces)} half-spaces, {side})"


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

    def support_maximiser(self, direction: Any, /) -> Any:
        """``centre + radius * direction / ||direction||``."""
        length = self.domain.norm(direction)
        if length == 0.0:
            return self._centre
        return self.domain.add(
            self._centre, self.domain.scale(self._radius / length, direction)
        )

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

    def support_maximiser(self, direction: Any, /) -> Any:
        """``centre + C q / sqrt((C q, q))``, which needs the covariance."""
        if self._covariance is None:
            raise AttributeError(
                "This ellipsoid was built without a covariance, so it cannot "
                "exhibit the point attaining its support."
            )
        image = self._covariance(direction)
        scale = np.sqrt(self.domain.inner_product(image, direction))
        if scale == 0.0:
            return self._centre
        return self.domain.add(self._centre, self.domain.scale(1.0 / scale, image))

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


class BallSurface(Subset):
    """The *surface* of a ball: ``{ x : ||x - c|| == r }``.

    Not convex, so it has no support function and no projection in the sense
    :class:`ConvexSet` means — but the nearest point on it is still well
    defined everywhere except the centre, and it is what a norm constraint of
    the form ``||x|| == r`` describes. Constrained optimisation is where this
    is wanted: an equality constraint, not an inequality.
    """

    def __init__(
        self, domain: HilbertSpace, /, *, radius: float = 1.0, centre: Any = None
    ) -> None:
        """
        Args:
            domain: the space.
            radius: the radius, which must be positive.
            centre: the centre. Defaults to zero.
        """
        super().__init__(domain)
        if radius <= 0.0:
            raise ValueError(f"The radius must be positive, got {radius}.")
        self._radius = float(radius)
        self._centre = domain.zero() if centre is None else centre

    @property
    def radius(self) -> float:
        """The radius."""
        return self._radius

    @property
    def centre(self) -> Any:
        """The centre."""
        return self._centre

    def contains(self, x: Any, /, *, tolerance: float = 1e-9) -> bool:
        """Whether a point lies on the surface, to a relative tolerance.

        Unlike a solid set, membership here is a measure-zero condition, so it
        is only ever meaningful up to a tolerance and the tolerance is a named
        argument rather than a hidden constant.
        """
        distance = self.domain.norm(self.domain.subtract(x, self._centre))
        return abs(distance - self._radius) <= tolerance * self._radius

    def project(self, x: Any, /) -> Any:
        """The nearest point on the surface.

        Undefined at the centre, where every point of the surface is equally
        near, and that is raised rather than resolved by an arbitrary choice.
        """
        offset = self.domain.subtract(x, self._centre)
        distance = self.domain.norm(offset)
        if distance == 0.0:
            raise ValueError(
                "The centre is equidistant from the whole surface, so it has "
                "no nearest point."
            )
        return self.domain.add(
            self._centre, self.domain.scale(self._radius / distance, offset)
        )

    def sample(self, /, *, rng: Any = None) -> Any:
        """A point drawn uniformly over the surface.

        White noise projected outward: the standard construction, and the
        reason it is white noise rather than ``random`` is that only the former
        is isotropic in a space with a non-trivial metric.
        """
        return self.project(
            self.domain.add(self._centre, self.domain.white_noise(rng=rng))
        )

    def __repr__(self) -> str:
        return f"BallSurface({self.domain!r}, radius={self._radius})"


class EllipsoidSurface(Subset):
    """The boundary of an ellipsoid: ``{ x : (P (x - c), x - c) == 1 }``."""

    def __init__(
        self,
        domain: HilbertSpace,
        precision: LinearOperator,
        /,
        *,
        centre: Any = None,
    ) -> None:
        """
        Args:
            domain: the space.
            precision: a positive definite operator on it.
            centre: the centre. Defaults to zero.
        """
        super().__init__(domain)
        required = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        if required & precision.traits != required:
            raise ValueError(
                f"The precision must claim {required!s}; it claims "
                f"{precision.traits!s}."
            )
        self._precision = precision
        self._centre = domain.zero() if centre is None else centre

    @property
    def precision(self) -> LinearOperator:
        """The shape operator."""
        return self._precision

    @property
    def centre(self) -> Any:
        """The centre."""
        return self._centre

    def contains(self, x: Any, /, *, tolerance: float = 1e-9) -> bool:
        """Whether a point lies on the surface, to a relative tolerance."""
        offset = self.domain.subtract(x, self._centre)
        value = self.domain.inner_product(self._precision(offset), offset)
        return abs(value - 1.0) <= tolerance

    def __repr__(self) -> str:
        return f"EllipsoidSurface({self.domain!r})"
