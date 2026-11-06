"""
Convex subsets, and the three views of one.

A convex set, its indicator functional and its support function are the same
object seen from three directions, and this module makes that explicit:

- ``set.contains(x)`` — the set as a predicate;
- ``set.indicator()`` — a ``Functional`` whose proximal operator *is* the
  projection, so a convex set drops straight into ``ProximalGradient``;
- ``set.support_function()`` — the convex-analysis view, with its own closed
  algebra.

``project`` here means the **metric projection**: the nearest point of the set,
which leaves a point already inside where it is. That is what a proximal method
needs. Going to the bounding hyperplane whichever side the point is on would be
a different map, not idempotent, and would quietly move feasible points.

Everything with a closed form is coordinate-free, because the closed forms are
written with norms and inner products. Where there is no closed form, a set
either computes the answer by a stated algorithm -- the projection onto an
ellipsoid is Newton's method on a secular equation, onto an intersection
Dykstra's algorithm -- or refuses, as the projection onto a Minkowski sum does,
rather than offering an approximation under the same name.
"""

from __future__ import annotations

import threading
from abc import abstractmethod
from typing import Any, Literal, Sequence

import numpy as np

from ..algebra.operators import Functional, LinearOperator
from ..algebra.spaces import HilbertSpace
from ..numerics.convex import SupportFunction
from ..traits import Traits
from .sets import Subset

__all__ = [
    "ConvexIntersection",
    "Polytope",
    "BallSurface",
    "EllipsoidSurface",
    "ConvexSet",
    "Ball",
    "HalfSpace",
    "Hyperplane",
    "Ellipsoid",
    "sum_of_ellipsoids",
    "enclosing_ellipsoid",
]


class ConvexSet(Subset):
    """A closed convex subset, and what it can say about itself.

    A convex set has up to four descriptions, and a given set has some of
    them: **membership**, whether a point is in it; a **projection**, its
    nearest point; a **support function**, ``sup (q, x)`` over the set, with
    a **maximizer** attaining it; and a **level function** ``f`` with the set
    ``{ x : f(x) <= level }``. The last two are the characterizations that
    matter for a set known through algorithms rather than formulas -- a
    support function bounds the set from outside one direction at a time, a
    level function decides points and gives inner bounds -- and a set may
    carry both, or one, or neither. Each is declared by a ``has_*`` property,
    meaning that the description is supplied cheaply; the closed-form sets
    here have all four, and the combinators carry what their parts allow.

    Two descriptions can be *converted* into when absent, at the cost of an
    optimization per evaluation, and the methods do so rather than refuse:
    a set with a differentiable level function gets a support function and
    a maximizer, by maximizing the pairing under the level constraint with
    one multiplier; and a set with a support function and a maximizer gets
    a projection, hence membership, by the dual of the nearest-point
    problem, a strongly convex minimization over the same space that is
    affordable when the space is small -- a property space, not a model
    space. The ``has_*`` flags stay false for a converted description, so
    that generic code can tell cheap from computed.
    """

    # ----------------------------------------------------------------- #
    #                          What this set can do                     #
    # ----------------------------------------------------------------- #

    @property
    def has_projection(self) -> bool:
        """Whether :meth:`project` is available."""
        return True

    @property
    def computes_support_function(self) -> bool:
        """Whether :meth:`support_function` answers, cheaply or by conversion."""
        return self.has_support_function or self.has_smooth_level_function

    @property
    def computes_maximizer(self) -> bool:
        """Whether :meth:`support_maximizer` answers, cheaply or by conversion."""
        return self.has_maximizer or self.has_smooth_level_function

    def intersect(self, other: Subset, /) -> Subset:
        """The intersection, which stays *convex* when the other set is.

        Convexity is preserved by intersection, and saying so is what keeps
        the projection available -- and with it every proximal method, the
        primal-dual route, and any use as a prior. Falls back to the general
        :class:`~pygeoinf.geometry.sets.Intersection` otherwise, which tests
        membership and nothing else.
        """
        if isinstance(other, ConvexSet):
            return ConvexIntersection([self, other])
        return super().intersect(other)

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

        Args:
            x: a vector of the space.
            rtol: how far outside still counts as inside, as a distance
                relative to the larger of the norms of the point and of its
                projection, so it means the same thing whatever the units
                are.

        Returns:
            Whether the set contains the point.
        """
        space = self._domain
        nearest = self.project(x)
        distance = space.norm(space.subtract(x, nearest))
        return distance <= rtol * max(space.norm(x), space.norm(nearest))

    def support_function(self) -> SupportFunction:
        """``h(y) == sup { (y, x) : x in this set }``.

        Returns:
            The support function, as a functional on the space.

        A set with a differentiable level function and no closed-form
        support function gets one by conversion: each value is the
        maximization of the pairing under the level constraint, a scalar
        root find over one multiplier with a convex minimization inside,
        and infinite in a direction the set is unbounded in.

        Raises:
            NotImplementedError: for a set with neither; see
                :attr:`has_support_function`. An intersection has none in
                closed form, and the feasible property set computes its own
                by its algorithms.
        """
        if self.has_smooth_level_function:
            return _LevelSupport(
                self.domain,
                self.level_function(),
                self.level,
                start=getattr(self, "center", None),
            )
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a support function."
        )

    def support_maximizer(self, direction: Any, /) -> Any:
        """The point of the set attaining ``h(direction)``.

        A *subgradient* of the support function, and the reason it is worth
        having: a nonsmooth minimization of a support function needs one at
        every step, and for the sets that have closed forms so does this.

        Args:
            direction: the direction to maximize along.

        Returns:
            A point of the set attaining the support value.

        Raises:
            NotImplementedError: for a set with no closed form and no
                differentiable level function to convert from.
            ValueError: if the set is unbounded in the direction.
        """
        if self.has_smooth_level_function:
            return self.support_function().subgradient(direction)
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a support maximizer."
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
        maximizer: Any = None,
        membership: Any = None,
    ) -> "ConvexSet":
        """A convex set given only by its support function.

        A closed convex set is *determined* by its support function
        (Rockafellar 13.1-13.2), so this loses nothing in principle. What it
        loses in practice is cheapness: every question has to go through the
        oracle, and one call may be an optimization.

        The distance algorithm of a feasible property set uses it for a set
        whose support values are computed by its other algorithms, so that
        the set is the oracle.

        Args:
            domain: the space.
            oracle: called with a direction, returning ``h(q)``.
            maximizer: optionally, called with a direction and returning the
                point of the set attaining the supremum. Supplying it is what
                makes an *inner* approximation available as well as an outer
                one.
            membership: optionally, called as ``membership(x, rtol)`` and
                returning whether ``x`` is in the set. A support function
                cannot decide membership, so a set that has a separate test
                for it hands it in here, and then answers both questions.
        """
        return _OracleSet(domain, oracle, maximizer=maximizer, membership=membership)

    def __add__(self, other: Any) -> Any:
        """The Minkowski sum, whose support function is the sum of theirs.

        ``h_{A+B} == h_A + h_B``, exactly, which is the one operation on convex
        sets that support functions make trivial. The sum has a support
        function and a maximizer when both summands do, and neither a
        projection nor a membership test.
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

    def support_maximizer(self, direction: Any, /) -> Any:
        """The base's maximizer, moved."""
        return self.domain.add(self._base.support_maximizer(direction), self._vector)

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """Membership of the moved point in the base.

        Args:
            x: a vector of the space.
            rtol: passed to the base's test.

        Returns:
            Whether the base contains ``x`` moved back.
        """
        return self._base.contains(self.domain.subtract(x, self._vector), rtol=rtol)

    @property
    def has_membership(self) -> bool:
        """Whether :meth:`contains` decides membership."""
        return self._base.has_membership

    @property
    def has_projection(self) -> bool:
        """Whether :meth:`project` is available."""
        return self._base.has_projection

    @property
    def has_support_function(self) -> bool:
        """Whether :meth:`support_function` is available."""
        return self._base.has_support_function

    @property
    def has_maximizer(self) -> bool:
        """Whether :meth:`support_maximizer` can exhibit a point."""
        return self._base.has_maximizer

    @property
    def has_level_function(self) -> bool:
        """Whether a level function describes the set."""
        return self._base.has_level_function

    def level_function(self) -> Functional:
        """``x -> f(x - v)``, the base's level function moved with the set."""
        base, vector, space = self._base.level_function(), self._vector, self.domain
        return Functional.from_callables(
            space,
            lambda x: base(space.subtract(x, vector)),
            gradient=(
                (lambda x: base.gradient(space.subtract(x, vector)))
                if base.has_derivative
                else None
            ),
            hessian=(
                (lambda x: base.hessian(space.subtract(x, vector)))
                if base.has_hessian
                else None
            ),
        )

    @property
    def level(self) -> float:
        """The level the level function is bounded by."""
        return self._base.level

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

    def __init__(
        self,
        domain: HilbertSpace,
        oracle: Any,
        /,
        *,
        maximizer: Any,
        membership: Any = None,
    ) -> None:
        super().__init__(domain)
        self._oracle = oracle
        self._maximizer = maximizer
        self._membership = membership

    @property
    def has_membership(self) -> bool:
        """Whether the set can decide membership, as well as bound itself."""
        return self._membership is not None

    @property
    def has_projection(self) -> bool:
        """False: a support function does not give a projection."""
        return False

    @property
    def has_support_function(self) -> bool:
        """True: that is what the set was given."""
        return True

    @property
    def has_maximizer(self) -> bool:
        """Whether a point attaining the supremum can be produced."""
        return self._maximizer is not None

    def support_maximizer(self, direction: Any, /) -> Any:
        """The maximizer, under the base's name."""
        return self.maximizer(direction)

    def maximizer(self, direction: Any, /) -> Any:
        """The point of the set furthest along a direction.

        Args:
            direction: the direction to maximize along.

        Returns:
            The maximizing point.

        Raises:
            NotImplementedError: unless the support function was built with a
                maximizer. A support *value* does not determine the point that
                attains it.
        """
        if self._maximizer is None:
            raise NotImplementedError(
                "This set was given a support function but no maximizer, so it "
                "can bound itself from outside but not exhibit a point."
            )
        return self._maximizer(direction)

    def support_function(self) -> SupportFunction:
        """The oracle, as a functional, carrying the maximizer when one was given."""
        return SupportFunction.of_oracle(
            self.domain, self._oracle, maximizer=self._maximizer
        )

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """Membership, when the set was given a test for it; otherwise refused.

        A support function alone cannot decide it: ``(q, x) <= h(q)`` for
        *every* ``q`` is membership, and no finite number of directions
        establishes it. What can be had from the support function is a
        certificate of *non*-membership — see :meth:`outside`. A set built
        with ``membership=`` answers exactly, by that test.

        Without a test, membership is decided through the nearest-point
        problem's dual, a strongly convex minimization over the space with
        one support evaluation per step, when a maximizer is available: see
        :meth:`project`.

        Args:
            x: a vector of the space.
            rtol: passed to the membership test, or the distance's tolerance
                relative to the larger of the norms of the point and of its
                projection.

        Raises:
            NotImplementedError: if there is neither a membership test nor a
                maximizer. Returning a guess would be worse than refusing.
        """
        if self._membership is not None:
            return bool(self._membership(x, rtol))
        if self._maximizer is None:
            raise NotImplementedError(
                "Membership needs every direction, and this set has no "
                "maximizer to convert its support function with. Use outside() "
                "for a certificate of exclusion, or polytope() for an outer bound."
            )
        space = self._domain
        nearest, distance, _ = _project_by_support(self, x)
        return distance <= rtol * max(space.norm(x), space.norm(nearest))

    def outside(self, x: Any, directions: Any, /, *, rtol: float = 1e-9) -> bool:
        """True when some direction proves the point is not in the set.

        One-sided, and honestly so: a ``True`` is a proof, and a ``False`` is
        only the absence of one among the directions tried. The point is
        past the supporting hyperplane of a direction ``q`` when
        ``(q, x) > h(q) + rtol * max(|h(q)|, ||q|| ||x||)``: relative to the
        larger of the support value and the size of the pairing itself, so
        that it means the same whatever the units and whatever the sign of
        the support.

        Args:
            x: the point to test.
            directions: the directions to try.
            rtol: how far past a supporting hyperplane the point must lie
                before it counts as outside it, relative to that scale.
        """
        space = self.domain
        size = space.norm(x)
        for direction in directions:
            pairing = space.inner_product(direction, x)
            bound = self._oracle(direction)
            scale = max(abs(bound), space.norm(direction) * size)
            if pairing > bound + rtol * scale:
                return True
        return False

    def project(self, x: Any, /) -> Any:
        """The nearest point, by the dual of the nearest-point problem.

        ``½ d(x)² == max over q of (q, x) − h(q) − ½ ||q||²``, whose maximizer
        is ``x − P(x)``; a strongly convex problem over the space, solved by
        a bundle method with one support evaluation and one maximizer per
        step. Affordable on a small space; on a model space it is the wrong
        tool, which is why :attr:`has_projection` stays false.

        Raises:
            NotImplementedError: if the set has no maximizer, without which
                the dual has no subgradient.
        """
        if self._maximizer is None:
            raise NotImplementedError(
                "A support function without a maximizer gives no projection. "
                "Bound the set with polytope() and project onto that."
            )
        return _project_by_support(self, x)[0]

    def polytope(self, directions: Any, /) -> "Polytope":
        """A certified outer bound, one half-space per direction."""
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


class _LevelSupport(SupportFunction):
    """The support function of ``{ f <= c }`` by one multiplier, for a differentiable ``f``.

    ``h(q) == sup { (q, x) : f(x) <= c }``. For a multiplier ``λ > 0`` the
    minimizer ``x_λ`` of ``f(x) − (q, x) / λ`` has ``f(x_λ)`` decreasing in
    ``λ``, so the multiplier at which ``f(x_λ) == c`` is a monotone root
    find, and ``x_λ`` there attains the supremum. When ``f(x_λ)`` stays
    above ``c`` as ``λ`` falls, no point of the set attains a finite
    supremum in that direction and the value is infinite.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        functional: Functional,
        level: float,
        /,
        *,
        start: Any = None,
    ) -> None:
        """
        Args:
            domain: the space.
            functional: ``f``, convex with a gradient.
            level: ``c``.
            start: where each inner minimization begins; a point of the
                set, such as its center. The origin if omitted.
        """
        super().__init__(domain)
        self._functional = functional
        self._level = float(level)
        self._start = start
        # One remembered answer per thread, so that a value and the
        # maximizer asked for the same direction in turn cost one solve.
        self._memo = threading.local()

    def _solve(self, direction: Any) -> tuple[float, Any]:
        last = getattr(self._memo, "last", None)
        if last is not None and last[0] is direction:
            return last[1]
        from ..numerics.optimization import LBFGS, NewtonCG
        from ..numerics.root_find import Evaluation, monotone_root

        space, f, level = self.domain, self._functional, self._level
        optimizer = (
            NewtonCG(forcing=1e-10, rtol=1e-12, gtol=0.0, ftol=1e-15)
            if f.has_hessian
            else LBFGS(rtol=1e-12, gtol=0.0, max_iterations=2000)
        )
        start = self._start
        if start is None:
            start = space.zero()
        if space.norm(direction) == 0.0:
            # The zero direction: the supremum is zero, attained anywhere
            # in the set, and the minimizer of f is a point of it.
            answer = (0.0, optimizer.minimize(f, start).minimizer)
            self._memo.last = (direction, answer)
            return answer

        def model(multiplier: float) -> Any:
            weight = 1.0 / multiplier

            def value(x: Any) -> float:
                return float(f(x)) - weight * space.inner_product(direction, x)

            def gradient(x: Any) -> Any:
                return space.axpy(-weight, direction, space.copy(f.gradient(x)))

            hessian = (lambda x: f.hessian(x)) if f.has_hessian else None
            objective = Functional.from_callables(
                space, value, gradient=gradient, hessian=hessian
            )
            return optimizer.minimize(objective, start).minimizer

        root = monotone_root(
            lambda multiplier, _: Evaluation(float(f(model(multiplier)))),
            level,
            decreasing=True,
            max_iterations=80,
            rtol=1e-9,
            atol=0.0,
            warm_start=False,
        )
        if root.exhausted == "low":
            answer: tuple[float, Any] = (float("inf"), None)
        else:
            point = model(root.argument)
            answer = (float(space.inner_product(direction, point)), point)
        self._memo.last = (direction, answer)
        return answer

    def _value(self, y: Any) -> float:
        return self._solve(y)[0]

    def _maximizer(self, y: Any) -> Any:
        point = self._solve(y)[1]
        if point is None:
            raise ValueError(
                "The set is unbounded in this direction; no point attains the "
                "support."
            )
        return point


def _project_by_support(
    subset: ConvexSet, x: Any, /, *, method: Any = None
) -> tuple[Any, float, Any]:
    """The nearest point of a set from its support function and maximizer.

    ``½ d(x)² == max over q of (q, x) − h(q) − ½ ||q||²`` (the Fenchel dual of
    the nearest-point problem), attained at ``q* == x − P(x)``. The
    minimization of the negated objective is strongly convex with subgradient
    ``p̂(q) − x + q``, ``p̂`` the maximizer, and a bundle method takes it. The
    bundle stops on the value, which is quadratic in the error of ``q``, so
    the projection is accurate to about the square root of its relative gap
    tolerance, in units of the distance.

    Returns:
        ``(projection, distance, result)``.
    """
    from ..numerics.convex import ProximalBundleMethod

    space = subset.domain
    support = subset.support_function()

    def value(q: Any) -> float:
        return (
            float(support(q)) - space.inner_product(q, x) + 0.5 * space.squared_norm(q)
        )

    def subgradient(q: Any) -> Any:
        return space.axpy(1.0, q, space.subtract(support.subgradient(q), x))

    objective = Functional.from_callables(space, value)
    optimizer = method or ProximalBundleMethod(tolerance=1e-12, max_iterations=500)
    result = optimizer.minimize(objective, space.zero(), subgradient=subgradient)
    q = result.minimizer
    return space.subtract(x, q), float(space.norm(q)), result


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

    def support_maximizer(self, direction: Any, /) -> Any:
        """The sum of the parts' maximizers, which attains the sum of supports."""
        return self.domain.add(
            self._first.support_maximizer(direction),
            self._second.support_maximizer(direction),
        )

    @property
    def has_membership(self) -> bool:
        """Whether :meth:`contains` decides membership."""
        return False

    @property
    def has_projection(self) -> bool:
        """Whether :meth:`project` is available."""
        return False

    @property
    def has_support_function(self) -> bool:
        """Whether :meth:`support_function` is available."""
        return self._first.has_support_function and self._second.has_support_function

    @property
    def has_maximizer(self) -> bool:
        """Whether :meth:`support_maximizer` can exhibit a point."""
        return self._first.has_maximizer and self._second.has_maximizer

    def project(self, x: Any, /) -> Any:
        """Not generally available: the sum of two projections is not one.

        Raises:
            NotImplementedError: always. Projecting onto a Minkowski sum is
                its own optimization, not a composition of the parts'.
        """
        raise NotImplementedError(
            "A Minkowski sum has no projection in closed form, even when its "
            "summands do. Bound it with a polytope from its support function."
        )

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """Not generally decidable without an optimization.

        Args:
            x: a vector of the space.
            rtol: unused.

        Raises:
            NotImplementedError: always. Deciding whether a point is a sum of
                one from each part is a feasibility problem.
        """
        raise NotImplementedError(
            "Membership of a Minkowski sum needs an optimization over the "
            "splitting of the point between the summands."
        )

    def outer_ellipsoid(
        self, *, criterion: Literal["trace", "volume"] = "trace"
    ) -> Ellipsoid:
        """An ellipsoid containing the sum, when every summand is a ball or an ellipsoid.

        See :func:`sum_of_ellipsoids`: the sum of ellipsoids is not one, and
        this is the smallest ellipsoid of a family that contains it.

        Args:
            criterion: what the ellipsoid is smallest in -- the sum of its
                squared semi-axes, or its volume.

        Raises:
            TypeError: if a summand is not a ball or an ellipsoid carrying
                its covariance.
        """
        covariances, center = [], self.domain.zero()
        for part in _flatten_sum(self):
            covariances.append(_covariance_of(part))
            center = self.domain.add(center, part.center)
        return sum_of_ellipsoids(
            self.domain, covariances, center=center, criterion=criterion
        )

    def __repr__(self) -> str:
        return f"MinkowskiSum({self._first!r}, {self._second!r})"


def _flatten_sum(subset: ConvexSet) -> list[ConvexSet]:
    if isinstance(subset, _MinkowskiSum):
        first, second = subset.summands
        return _flatten_sum(first) + _flatten_sum(second)
    return [subset]


def _covariance_of(subset: ConvexSet) -> LinearOperator:
    """The covariance of a ball or an ellipsoid, in the ellipsoid's normalized units."""
    if isinstance(subset, Ball):
        return (LinearOperator.identity(subset.domain) * subset.radius**2).with_traits(
            Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )
    if isinstance(subset, Ellipsoid) and subset.covariance is not None:
        return subset.covariance
    raise TypeError(
        "Each summand must be a ball or an ellipsoid carrying its "
        f"covariance; got {subset!r}."
    )


def _component_matrix(space: HilbertSpace, operator: LinearOperator) -> np.ndarray:
    """The matrix of an operator on a small space, column by column."""
    return np.column_stack(
        [space.to_components(operator(space.basis_vector(j))) for j in range(space.dim)]
    )


def _ellipsoid_from_components(
    space: HilbertSpace, covariance: np.ndarray, center: Any
) -> Ellipsoid:
    """The ellipsoid with this covariance, given as a component matrix."""
    traits = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
    try:
        precision = np.linalg.inv(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "The ellipsoid is flat: its covariance is singular."
        ) from error
    return Ellipsoid(
        space,
        LinearOperator.from_matrix(
            space, space, precision, form="components", traits=traits
        ),
        center=center,
        covariance=LinearOperator.from_matrix(
            space, space, covariance, form="components", traits=traits
        ),
    )


def sum_of_ellipsoids(
    space: HilbertSpace,
    covariances: Sequence[LinearOperator],
    /,
    *,
    center: Any = None,
    criterion: Literal["trace", "volume"] = "trace",
) -> Ellipsoid:
    """An ellipsoid containing the Minkowski sum of ellipsoids with these covariances.

    The sum of ``{ C_i^½ u : |u| <= 1 }`` over ``i`` is not an ellipsoid, but
    for any weights ``α_i > 0`` summing to one it lies in the ellipsoid of
    covariance ``Σ C_i / α_i``, by the Cauchy-Schwarz inequality applied to
    the sum. Minimizing the trace of that covariance -- the sum of the
    squared semi-axes -- has the closed form ``α_i`` proportional to
    ``sqrt(tr C_i)``; for two summands it is ``(1 + 1/p) C₁ + (1 + p) C₂``
    with ``p == sqrt(tr C₁ / tr C₂)``. Minimizing the volume is a scalar
    minimization in ``log p`` for two summands, of a function that is
    unimodal.

    This is what a linear certificate's set reduces to on quadratic sets:
    one ellipsoid about the estimate in place of a sum of two.

    Args:
        space: the space the ellipsoids lie on; small, as the covariances
            are assembled as matrices on it.
        covariances: the summands' covariances, as operators on the space.
            A summand that is a point has covariance zero and may be left
            out; one that is given is dropped.
        center: the sum of the summands' centers. The origin if omitted.
        criterion: what the ellipsoid is smallest in -- the sum of its
            squared semi-axes, or its volume.

    Returns:
        The ellipsoid, carrying its covariance.

    Raises:
        ValueError: if no covariance is nonzero, the criterion is unknown,
            or the sum is flat; NotImplementedError for the volume
            criterion with more than two summands.
    """
    if criterion not in ("trace", "volume"):
        raise ValueError(f"The criterion is 'trace' or 'volume', got {criterion!r}.")
    center = space.zero() if center is None else center
    matrices = [_component_matrix(space, covariance) for covariance in covariances]
    traces = [float(np.trace(matrix)) for matrix in matrices]
    kept = [(matrix, trace) for matrix, trace in zip(matrices, traces) if trace > 0.0]
    if not kept:
        raise ValueError("Every covariance is zero: the sum is a point.")
    if len(kept) == 1:
        return _ellipsoid_from_components(space, kept[0][0], center)
    if criterion == "trace":
        roots = [np.sqrt(trace) for _, trace in kept]
        total = sum(roots)
        combined = total * sum(matrix / root for (matrix, _), root in zip(kept, roots))
        return _ellipsoid_from_components(space, combined, center)
    if len(kept) > 2:
        raise NotImplementedError(
            "The volume criterion is available for two summands; the trace "
            "criterion for any number."
        )
    from scipy.optimize import minimize_scalar

    (first, trace_1), (second, trace_2) = kept

    def log_volume(x: float) -> float:
        p = np.exp(x)
        sign, log_det = np.linalg.slogdet((1.0 + 1.0 / p) * first + (1.0 + p) * second)
        return float(log_det) if sign > 0.0 else np.inf

    guess = 0.5 * np.log(trace_1 / trace_2)
    result = minimize_scalar(
        log_volume,
        bounds=(guess - 20.0, guess + 20.0),
        method="bounded",
        options=dict(xatol=1e-8),
    )
    p = float(np.exp(result.x))
    return _ellipsoid_from_components(
        space, (1.0 + 1.0 / p) * first + (1.0 + p) * second, center
    )


def enclosing_ellipsoid(
    space: HilbertSpace,
    points: Sequence[Any],
    /,
    *,
    tolerance: float = 1e-7,
    max_iterations: int = 10_000,
) -> Ellipsoid:
    """The minimum-volume ellipsoid enclosing a set of points, by Khachiyan's algorithm.

    A description of the points' convex hull and nothing more: applied to
    the contact points of a set it encloses the inner hull, which lies
    inside the set, and says nothing about the set's extent beyond them.
    An outer bound of the set is :meth:`Polytope.inscribed_ellipsoid` on
    the polytope of its support values, or the set's own
    ``outer_ellipsoid``.

    The algorithm is the Frank-Wolfe iteration on the dual, with away
    steps (Todd and Yildirim, 2007): weights on the points, moved towards
    the point furthest outside the current ellipsoid or away from the
    weighted point furthest inside, until both are within the tolerance.
    The ellipsoid is then inflated to the furthest point, so that it
    encloses every point whatever the tolerance; the tolerance only
    governs how far from the smallest it is. It is dense in the dimension
    of the space and linear in the number of points per iteration.

    Args:
        space: the space the points lie on; small.
        points: the points, at least one more than the dimension and not
            all on one hyperplane.
        tolerance: on the relative excess of the furthest point.
        max_iterations: the iteration budget.

    Returns:
        The ellipsoid, carrying its covariance.

    Raises:
        ValueError: if the points do not span the space.
    """
    components = np.stack([space.to_components(point) for point in points]).astype(
        float
    )
    count, dim = components.shape
    if count <= dim:
        raise ValueError(
            f"{count} points cannot span {dim} dimensions; an enclosing "
            "ellipsoid needs at least one more point than the dimension."
        )
    lifted = np.column_stack([components, np.ones(count)])
    weights = np.full(count, 1.0 / count)
    target = dim + 1.0
    for _ in range(max_iterations):
        gram = lifted.T @ (weights[:, None] * lifted)
        try:
            solved = np.linalg.solve(gram, lifted.T)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "The points lie on a hyperplane and span no volume."
            ) from error
        distances = np.einsum("ij,ji->i", lifted, solved)
        furthest = int(np.argmax(distances))
        carried = np.flatnonzero(weights > 0.0)
        nearest = int(carried[np.argmin(distances[carried])])
        excess = distances[furthest] - target
        deficit = target - distances[nearest]
        if max(excess, deficit) <= tolerance * target:
            break
        if excess >= deficit:
            step = excess / (target * (distances[furthest] - 1.0))
            weights *= 1.0 - step
            weights[furthest] += step
        else:
            share = weights[nearest]
            step = min(
                share / (1.0 - share),
                deficit / (target * max(distances[nearest] - 1.0, 1e-300)),
            )
            weights *= 1.0 + step
            weights[nearest] = max(weights[nearest] - step, 0.0)
    center_components = weights @ components
    scatter = components.T @ (weights[:, None] * components) - np.outer(
        center_components, center_components
    )
    # In components the ellipsoid is (x − c)ᵀ M (x − c) <= 1 with
    # M == scatter⁻¹ / dim; that is the Galerkin matrix of the precision.
    try:
        precision_galerkin = np.linalg.inv(scatter) / dim
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "The points lie on a hyperplane and span no volume."
        ) from error
    offsets = components - center_components
    inflation = float(
        np.max(np.einsum("ij,jk,ik->i", offsets, precision_galerkin, offsets))
    )
    precision_galerkin /= inflation
    gram_matrix = space.gram_matrix()
    covariance = (inflation * dim) * scatter @ gram_matrix
    traits = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
    return Ellipsoid(
        space,
        LinearOperator.from_matrix(
            space, space, precision_galerkin, form="galerkin", traits=traits
        ),
        center=space.from_components(center_components),
        covariance=LinearOperator.from_matrix(
            space, space, covariance, form="components", traits=traits
        ),
    )


def _dykstra(space: Any, parts: Sequence[Any], x: Any, max_iterations: int) -> Any:
    """The nearest point of an intersection, by Dykstra's algorithm.

    Cycling the projections on their own -- alternating projection -- reaches
    *a* point of the intersection, not the nearest one. Dykstra is the same
    cycle carrying one correction vector per part: each is added back before
    its projection and re-formed after it, which is what makes the limit the
    projection onto the intersection rather than merely a point of it.

    Args:
        space: the space the sets live in.
        parts: the sets, each of which must project.
        x: the point to project.
        max_iterations: the maximum number of cycles, each one projection per part.

    Returns:
        The nearest point, to the accuracy the cycle reached.
    """
    corrections = [space.zero() for _ in parts]
    current = space.copy(x)
    for _ in range(max_iterations):
        start = space.copy(current)
        for index, part in enumerate(parts):
            shifted = space.add(current, corrections[index])
            current = part.project(shifted)
            corrections[index] = space.subtract(shifted, current)
        if space.norm(space.subtract(current, start)) <= 1e-14 * max(
            space.norm(current), 1.0
        ):
            break
    return current


class Polytope(ConvexSet):
    """An intersection of half-spaces, which remembers which side it is on.

    A polytope built from support function values *contains* the set it
    approximates; one built as the convex hull of feasible points is
    *contained* in it. Reporting the second as though it were the answer
    understates the set, and the two are not interchangeable -- so ``outer``
    is not optional.
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

        Raises:
            TypeError: if a constraint is not a :class:`HalfSpace`.
            ValueError: if no half-spaces are given, or one lives on another
                space.
        """
        super().__init__(domain)
        self._half_spaces = tuple(half_spaces)
        if not self._half_spaces:
            raise ValueError("A polytope needs at least one half-space.")
        for plane in self._half_spaces:
            if not isinstance(plane, HalfSpace):
                raise TypeError(
                    f"A polytope is built from HalfSpace objects, got "
                    f"{type(plane).__name__}."
                )
        if any(plane.domain != domain for plane in self._half_spaces):
            raise ValueError("Every half-space must live on the same space.")
        self._outer = bool(outer)

    @property
    def half_spaces(self) -> tuple:
        """The constraints."""
        return self._half_spaces

    @property
    def has_level_function(self) -> bool:
        """Whether a level function describes the set."""
        return True

    def level_function(self) -> Functional:
        """``max_i ((n_i, x) - b_i)``, at level zero; the active plane's normal is a subgradient."""
        space, planes = self._domain, self._half_spaces

        def excesses(x: Any) -> list[float]:
            return [
                space.inner_product(plane.normal, x) - plane.offset for plane in planes
            ]

        return Functional.from_callables(
            space,
            lambda x: max(excesses(x)),
            subgradient=lambda x: space.copy(
                planes[int(np.argmax(excesses(x)))].normal
            ),
        )

    @property
    def level(self) -> float:
        """The level the level function is bounded by."""
        return 0.0

    @property
    def is_outer(self) -> bool:
        """Whether this contains the set it approximates, or is contained by it."""
        return self._outer

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """Whether the point satisfies every constraint.

        Args:
            x: a vector of the space.
            rtol: how far outside still counts as inside. Scaled by the set's
                own size, so it means the same thing whatever the units are.

        Returns:
            Whether the set contains the point.
        """
        return all(plane.contains(x, rtol=rtol) for plane in self._half_spaces)

    def project(self, x: Any, /, *, max_iterations: int = 1000) -> Any:
        """The nearest point of the polytope, by Dykstra's algorithm.

        Cycling the half-space projections on their own -- alternating
        projection -- reaches *a* feasible point, not the nearest one. On
        ``{x <= 0} and {x + y <= 0}`` from ``(1, 0.5)`` it stops at
        ``(-0.25, 0.25)`` at squared distance 1.625, where the nearest point is
        the origin at 1.25. That matters because :meth:`indicator` hands this
        to a proximal method as a prox, and a prox that is not the projection
        gives the wrong fixed point.

        Dykstra is the same cycle carrying one correction vector per
        constraint: each is added back before its projection and re-formed
        after it, which is what makes the limit the projection onto the
        intersection rather than merely a point of it.

        Args:
            x: the point to project.
            max_iterations: the maximum number of cycles, each one projection
                per half-space. The cycle stops itself earlier once it has
                converged. An under-converged projection is a wrong prox, so
                the cap is generous.

        Returns:
            The nearest point of the polytope, to the accuracy the cycle
            reached.
        """
        return _dykstra(self.domain, self._half_spaces, x, max_iterations)

    def __and__(self, other: Subset) -> Subset:
        """Both sets of constraints, which tightens an outer bound.

        With another polytope, one polytope carrying both sets of
        half-spaces; with any other set, the general intersection, as
        :meth:`intersect` gives.

        Raises:
            ValueError: for an outer and an inner polytope.
        """
        if not isinstance(other, Polytope):
            return self.intersect(other)
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

    def inscribed_ellipsoid(
        self, *, tolerance: float = 1e-8, max_iterations: int = 200
    ) -> Ellipsoid:
        """The maximum-volume ellipsoid inside the polytope.

        A description of the polytope, not of the set it bounds: an outer
        polytope's inscribed ellipsoid need not contain the set and need
        not lie inside it. It is the ellipsoid a plot of the polytope
        reads off -- its center and its principal widths -- and a
        description that stays stable as more half-spaces are added.

        The ellipsoid ``{ c + B u : |u| <= 1 }`` lies in a half-space
        ``(n, x) <= b`` exactly when ``|B n| + (n, c) <= b``, so the
        maximum volume is ``max log det B`` under one such constraint per
        half-space, a convex program in the ``dim (dim + 3) / 2`` entries
        of ``B`` and ``c`` (Boyd and Vandenberghe, section 8.4.2). It is
        solved by Newton's method on the logarithmic barrier, from the
        Chebyshev center, and is dense in the dimension of the space.

        Args:
            tolerance: on the duality gap of the barrier method, relative
                to the number of half-spaces.
            max_iterations: the budget of Newton steps, over all barrier
                stages.

        Returns:
            The ellipsoid, carrying its covariance.

        Raises:
            ValueError: if the polytope is empty, flat or unbounded, or the
                Newton iteration runs out of budget.
        """
        from scipy.optimize import linprog

        space = self.domain
        gram_matrix = space.gram_matrix()
        normals = np.stack(
            [
                gram_matrix @ space.to_components(plane.normal)
                for plane in self._half_spaces
            ]
        )
        offsets = np.array([plane.offset for plane in self._half_spaces], dtype=float)
        dim = space.dim
        count = len(self._half_spaces)
        if count < dim + 1:
            raise ValueError(
                f"{count} half-spaces cannot bound {dim} dimensions; the polytope "
                "is unbounded."
            )
        lengths = np.linalg.norm(normals, axis=1)
        # Bounded in every coordinate direction, or not at all.
        for j in range(dim):
            for sign in (1.0, -1.0):
                extreme = linprog(
                    sign * np.identity(dim)[j],
                    A_ub=normals,
                    b_ub=offsets,
                    bounds=[(None, None)] * dim,
                    method="highs",
                )
                if extreme.status == 3:
                    raise ValueError(
                        "The polytope is unbounded; it has no largest ellipsoid."
                    )
        # The Chebyshev center: the largest ball inside, as a start.
        program = linprog(
            np.append(np.zeros(dim), -1.0),
            A_ub=np.column_stack([normals, lengths]),
            b_ub=offsets,
            bounds=[(None, None)] * dim + [(0.0, None)],
            method="highs",
        )
        if program.status == 3:
            raise ValueError("The polytope is unbounded; it has no largest ellipsoid.")
        if program.status != 0 or program.x[-1] <= 0.0:
            raise ValueError("The polytope is empty or flat; it has no interior.")
        center = program.x[:-1]
        radius = program.x[-1]

        # A basis of the symmetric matrices, orthonormal in the Frobenius norm.
        basis = []
        for k in range(dim):
            for ell in range(k, dim):
                matrix = np.zeros((dim, dim))
                if k == ell:
                    matrix[k, k] = 1.0
                else:
                    matrix[k, ell] = matrix[ell, k] = 1.0 / np.sqrt(2.0)
                basis.append(matrix)
        basis = np.stack(basis)
        size = len(basis)
        # Columns: for each half-space, E_k n_i for every basis matrix.
        images = np.einsum("kpq,iq->ipk", basis, normals)  # (count, dim, size)

        def unpack(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return np.einsum("k,kpq->pq", z[:size], basis), z[size:]

        def slacks(z: np.ndarray) -> np.ndarray | None:
            shape, center = unpack(z)
            try:
                np.linalg.cholesky(shape)
            except np.linalg.LinAlgError:
                return None
            reach = np.linalg.norm(normals @ shape, axis=1)
            slack = offsets - normals @ center - reach
            return slack if np.all(slack > 0.0) else None

        def objective(z: np.ndarray, weight: float) -> float:
            shape, _ = unpack(z)
            slack = slacks(z)
            if slack is None:
                return np.inf
            sign, log_det = np.linalg.slogdet(shape)
            if sign <= 0.0:
                return np.inf
            return -weight * float(log_det) - float(np.sum(np.log(slack)))

        def derivatives(z: np.ndarray, weight: float) -> tuple[np.ndarray, np.ndarray]:
            shape, _ = unpack(z)
            slack = slacks(z)
            inverse = np.linalg.inv(shape)
            gradient = np.zeros(size + dim)
            hessian = np.zeros((size + dim, size + dim))
            # −weight log det B.
            gradient[:size] = -weight * np.einsum("pq,kqp->k", inverse, basis)
            products = np.einsum("pq,kqr->kpr", inverse, basis)
            hessian[:size, :size] = weight * np.einsum(
                "kpq,lqp->kl", products, products
            )
            # −Σ log s_i with s_i == b_i − (n_i, c) − |B n_i|.
            reaches = normals @ shape  # rows v_i == B n_i
            for i in range(count):
                v = reaches[i]
                g = np.linalg.norm(v)
                w = images[i]  # (dim, size): E_k n_i by column
                d = w.T @ v / g  # ∂|B n_i| / ∂z_k
                ds = np.concatenate([-d, -normals[i]])
                second = (w.T @ w - np.outer(d, d)) / g  # ∂²|B n_i|
                gradient -= ds / slack[i]
                hessian += np.outer(ds, ds) / slack[i] ** 2
                hessian[:size, :size] += second / slack[i]
            return gradient, hessian

        z = np.concatenate(
            [np.einsum("kpq,pq->k", basis, 0.5 * radius * np.identity(dim)), center]
        )
        weight = 1.0
        steps = 0
        while True:
            # Newton's method on the barrier at this weight.
            while True:
                gradient, hessian = derivatives(z, weight)
                try:
                    step = -np.linalg.solve(hessian, gradient)
                except np.linalg.LinAlgError as error:
                    raise ValueError(
                        "The Newton system is singular; the polytope may be flat."
                    ) from error
                decrement = -float(gradient @ step)
                # The objective carries the weight, so the stop does too.
                if decrement / 2.0 <= 1e-12 * weight:
                    break
                value = objective(z, weight)
                length = 1.0
                while (
                    objective(z + length * step, weight)
                    > value - 0.25 * length * decrement
                ):
                    length *= 0.5
                    if length < 1e-10:
                        break
                if length < 1e-10:
                    # Rounding has caught up with the line search: nothing
                    # further to resolve at this weight.
                    break
                z = z + length * step
                steps += 1
                if steps >= max_iterations:
                    raise ValueError(
                        f"The inscribed ellipsoid did not converge in {max_iterations} "
                        "Newton steps."
                    )
            if 1.0 / weight <= tolerance:
                break
            weight *= 20.0
        shape, center = unpack(z)
        # In components the ellipsoid is (x − c)ᵀ B⁻² (x − c) <= 1: the
        # Galerkin matrix of the precision is B⁻².
        shape_inverse = np.linalg.inv(shape)
        traits = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        return Ellipsoid(
            space,
            LinearOperator.from_matrix(
                space,
                space,
                shape_inverse @ shape_inverse,
                form="galerkin",
                traits=traits,
            ),
            center=space.from_components(center),
            covariance=LinearOperator.from_matrix(
                space,
                space,
                shape @ shape @ gram_matrix,
                form="components",
                traits=traits,
            ),
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
    """``{ x : ||x - center|| <= radius }``."""

    def __init__(
        self,
        domain: HilbertSpace,
        /,
        *,
        radius: float = 1.0,
        center: Any = None,
    ) -> None:
        """
        Args:
            domain: the space.
            radius: the radius, which must not be negative. Zero is allowed and
                gives the single point at the center -- the degenerate case that
                says "exactly this", which is what error-free data are. There
                ``project`` and ``support_maximizer`` return the center and
                ``contains`` admits it alone.
            center: the center. Defaults to zero.

        Raises:
            ValueError: if the radius is negative.
        """
        if radius < 0.0:
            raise ValueError("radius must not be negative.")
        super().__init__(domain)
        self._radius = float(radius)
        self._center = domain.zero() if center is None else center

    def support_maximizer(self, direction: Any, /) -> Any:
        """``center + radius * direction / ||direction||``."""
        length = self.domain.norm(direction)
        if length == 0.0:
            return self._center
        return self.domain.add(
            self._center, self.domain.scale(self._radius / length, direction)
        )

    @property
    def covariance(self) -> LinearOperator:
        """``r² I``: the ball as the ellipsoid ``{(C⁻¹(x − c), x − c) <= 1}``.

        What makes a ball interchangeable with an :class:`Ellipsoid` wherever
        a quadratic set's shape operator is read; the identity carries the
        space's own metric.
        """
        return (LinearOperator.identity(self._domain) * self._radius**2).with_traits(
            Traits.POSITIVE_DEFINITE
        )

    @property
    def precision(self) -> LinearOperator:
        """``I / r²``: the inverse of :attr:`covariance`.

        Raises:
            ValueError: for a ball of zero radius, the single point at the
                center, whose quadratic form is not finite.
        """
        if self._radius <= 0.0:
            raise ValueError(
                "A ball of zero radius is a single point and has no precision."
            )
        return (
            LinearOperator.identity(self._domain) * (1.0 / self._radius**2)
        ).with_traits(Traits.POSITIVE_DEFINITE)

    def translate(self, vector: Any, /) -> "Ball":
        """The same ball, moved by a vector."""
        return Ball(
            self.domain,
            radius=self._radius,
            center=self.domain.add(self._center, vector),
        )

    @property
    def radius(self) -> float:
        """The radius."""
        return self._radius

    @property
    def center(self) -> Any:
        """The center."""
        return self._center

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """Cheaper than the default: one norm rather than a projection.

        Args:
            x: a vector of the space.
            rtol: how far outside still counts as inside, relative to the
                largest of the radius, ``||x||`` and ``||center||``: the sizes
                of the quantities compared, so it means the same thing
                whatever the units are, and a ball of radius zero still
                contains its own center after rounding.

        Returns:
            Whether the set contains the point.
        """
        space = self._domain
        offset = space.norm(space.subtract(x, self._center))
        scale = max(self._radius, space.norm(x), space.norm(self._center))
        return offset <= self._radius + rtol * scale

    def project(self, x: Any, /) -> Any:
        """Rescale the offset from the center to the radius, if it exceeds it."""
        space = self._domain
        offset = space.subtract(x, self._center)
        distance = space.norm(offset)
        if distance <= self._radius:
            return space.copy(x)
        return space.axpy(self._radius / distance, offset, space.copy(self._center))

    @property
    def boundary(self) -> "BallSurface":
        """The sphere bounding it, as a set in its own right.

        This is what an equality constraint ``||x - c|| == r`` describes:
        :class:`BallSurface` projects onto it and samples uniformly over it,
        neither of which the solid ball does.

        Returns:
            The bounding surface.

        Raises:
            ValueError: for a ball of zero radius. That set is the single
                point at the center, and its boundary in the ambient space is
                itself -- not a surface, which is why this refuses rather than
                returning something with a radius of zero.
        """
        return BallSurface(self._domain, radius=self._radius, center=self._center)

    def support_function(self) -> SupportFunction:
        """``radius ||y|| + (center, y)``."""
        return SupportFunction.of_ball(
            self._domain, radius=self._radius, center=self._center
        )

    @property
    def has_support_function(self) -> bool:
        """Whether :meth:`support_function` is available."""
        return True

    @property
    def has_maximizer(self) -> bool:
        """Whether :meth:`support_maximizer` can exhibit a point."""
        return True

    @property
    def has_level_function(self) -> bool:
        """Whether a level function describes the set."""
        return True

    def level_function(self) -> Functional:
        """The squared distance to the center, at level ``radius^2``."""
        space, center = self._domain, self._center
        return Functional.from_callables(
            space,
            lambda x: space.squared_norm(space.subtract(x, center)),
            gradient=lambda x: space.scale(2.0, space.subtract(x, center)),
            hessian=lambda x: (LinearOperator.identity(space) * 2.0).with_traits(
                Traits.POSITIVE_DEFINITE
            ),
        )

    @property
    def level(self) -> float:
        """The level the level function is bounded by."""
        return self._radius**2

    def indicator(self) -> Functional:
        """The ball's indicator, with its closed-form proximal operator."""
        from ..numerics.convex import BallIndicator

        return BallIndicator(self._domain, radius=self._radius, center=self._center)

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

        Raises:
            ValueError: for a zero normal.
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
        """True when the point satisfies the equation to tolerance.

        Args:
            x: a vector of the space.
            rtol: how far outside still counts as inside, relative to the
                larger of ``|offset|`` and ``||normal|| ||x||``, the sizes of
                the two sides of the equation, so it means the same thing
                whatever the units are.

        Returns:
            Whether the set contains the point.
        """
        scale = max(
            abs(self._offset), self._domain.norm(x) * np.sqrt(self._squared_norm)
        )
        return abs(self._residual(x)) <= rtol * scale

    def project(self, x: Any, /) -> Any:
        """Remove the component of the residual along the normal."""
        space = self._domain
        return space.axpy(
            -self._residual(x) / self._squared_norm, self._normal, space.copy(x)
        )

    @property
    def boundary(self) -> "Hyperplane":
        """Itself: a hyperplane has empty interior, so every point is a boundary point."""
        return self

    def support_function(self) -> SupportFunction:
        """``alpha * offset`` for a direction ``alpha * normal``, else ``+inf``.

        Finite only along the normal, either way: the plane is bounded in
        that one direction and unbounded in every other.
        """
        return SupportFunction.of_hyperplane(
            self._domain, self._normal, offset=self._offset
        )

    @property
    def has_support_function(self) -> bool:
        """Whether :meth:`support_function` is available."""
        return True

    @property
    def has_maximizer(self) -> bool:
        """Whether :meth:`support_maximizer` can exhibit a point."""
        return True

    def support_maximizer(self, direction: Any, /) -> Any:
        """The plane's point of least norm, ``(offset / (normal, normal)) normal``.

        Every point of the plane attains the support when it is finite; this
        is the canonical one.

        Raises:
            ValueError: if the support is infinite in this direction.
        """
        return self.support_function().subgradient(direction)

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

        Raises:
            ValueError: for a zero normal.
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
        """True when the inequality holds to tolerance.

        Scaled by the same two quantities :meth:`Hyperplane.contains` uses,
        which is what makes a point on the boundary of a half-space and a
        point on its bounding hyperplane agree about whether they are on it.
        The offset alone is not enough: a constraint through the origin has
        offset zero, and the tolerance would then not scale with the point at
        all.

        Args:
            x: a vector of the space.
            rtol: how far outside still counts as inside, relative to the
                larger of ``|offset|`` and ``||normal|| ||x||``, so it means
                the same thing whatever the units are.

        Returns:
            Whether the set contains the point.
        """
        residual = self._domain.inner_product(self._normal, x) - self._offset
        scale = max(
            abs(self._offset),
            self._domain.norm(x) * self._domain.norm(self._normal),
        )
        return residual <= rtol * scale

    def project(self, x: Any, /) -> Any:
        """The metric projection, which leaves feasible points alone.

        A point already inside is returned unchanged. Projecting onto the
        boundary regardless would be a different map, not idempotent, and
        would move feasible points in a proximal iteration.
        """
        space = self._domain
        excess = space.inner_product(self._normal, x) - self._offset
        if excess <= 0.0:
            return space.copy(x)
        return space.axpy(-excess / self._squared_norm, self._normal, space.copy(x))

    def support_function(self) -> SupportFunction:
        """``alpha * offset`` for a direction ``alpha * normal``, ``alpha >= 0``, else ``+inf``.

        An unbounded set has an extended-real support function, and this one
        is infinite in every direction but the outward normal's; in an
        intersection that is the right contribution, since ``min_i h_i`` is
        then decided by the bounded parts.
        """
        return SupportFunction.of_half_space(
            self._domain, self._normal, offset=self._offset
        )

    @property
    def has_support_function(self) -> bool:
        """Whether :meth:`support_function` is available."""
        return True

    @property
    def has_maximizer(self) -> bool:
        """Whether :meth:`support_maximizer` can exhibit a point."""
        return True

    @property
    def has_level_function(self) -> bool:
        """Whether a level function describes the set."""
        return True

    def level_function(self) -> Functional:
        """The linear form ``(normal, .)``, at level ``offset``."""
        from ..algebra.operators import LinearFunctional

        return LinearFunctional.from_representer(self._domain, self._normal)

    @property
    def level(self) -> float:
        """The level the level function is bounded by."""
        return self._offset

    def support_maximizer(self, direction: Any, /) -> Any:
        """The boundary plane's point of least norm, ``(offset / (normal, normal)) normal``.

        Every point of the boundary attains the support when it is finite;
        this is the canonical one.

        Raises:
            ValueError: if the support is infinite in this direction.
        """
        return self.support_function().subgradient(direction)

    def __repr__(self) -> str:
        return f"HalfSpace(offset={self._offset})"


_INVERSE_RTOL = 1e-6
"""How closely, relative to ``(x, C x)``, a covariance must invert its precision on the probe."""


def _check_inverse(
    space: HilbertSpace, precision: LinearOperator, covariance: LinearOperator
) -> None:
    """Refuse a covariance that is not the precision's inverse, on one probe.

    With ``y == C x`` for a fixed random ``x``, ``(P y, y) == (x, y)`` when
    ``P C == I``. The quadratic form rather than ``P C x`` itself keeps the
    test indifferent to the conditioning of an ellipsoid long in some
    directions and thin in others, while a covariance off by a scale, or of
    another shape, fails it.

    Raises:
        ValueError: if the two differ by more than a millionth of ``(x, y)``.
    """
    probe = space.random(rng=np.random.default_rng(0))
    image = covariance(probe)
    expected = float(space.inner_product(probe, image))
    found = float(space.inner_product(precision(image), image))
    if not abs(found - expected) <= _INVERSE_RTOL * abs(expected):
        raise ValueError(
            "The covariance is not the precision's inverse: with y == C x on a "
            f"probe x, (P y, y) is {found:.6g} where (x, y) is {expected:.6g}."
        )


class Ellipsoid(ConvexSet):
    """``{ x : (x - center, P (x - center)) <= 1 }`` for a positive semidefinite ``P``.

    Given by its **precision** rather than its covariance, because that is what
    a membership test needs and what a credible region is naturally expressed
    with: the set of points whose Mahalanobis distance is at most one.

    A support function is available in closed form given the covariance;
    without one it is computed by the level-function conversion, a solve
    with the precision per evaluation. The projection has no closed form:
    :meth:`project` solves a scalar secular equation by Newton's method,
    with linear solves at every step.

    A precision that claims only positive *semi*-definiteness defines an
    unbounded set -- a bound on a semi-norm, or an ``L²`` amplitude bound on a
    Sobolev space -- which is admitted: membership, the level function and
    the projection are unchanged, there is no covariance to give, and the
    support function is infinite in the directions the set is unbounded in.
    :attr:`is_bounded` says which case this is.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        precision: LinearOperator,
        /,
        *,
        center: Any = None,
        covariance: LinearOperator | None = None,
        factor: LinearOperator | None = None,
        radius: float | None = None,
    ) -> None:
        """
        Args:
            domain: the space.
            precision: a self-adjoint positive semidefinite operator on it;
                positive definite for a bounded ellipsoid.
            center: the center. Defaults to zero.
            radius: when the ellipsoid is a ball of radius ``r`` in the norm
                an operator ``B`` defines, so that the precision is
                ``B / r^2``, that ``r``, for reading back; it changes
                nothing about the set.
            covariance: the inverse of the precision, if it is known. Supplying
                it is what makes the support function available. It is
                checked on one probe ``x``, at the cost of one application of
                each operator: with ``y == C x``, ``(P y, y)`` must equal
                ``(x, y)`` to a millionth, which a covariance that is not the
                precision's inverse fails, a scaled one included.
            factor: a factor ``L`` of the covariance, ``C == L L*``, if one
                is known -- a Gaussian measure usually carries one. The
                support function then costs one adjoint application of the
                factor rather than one of the covariance. The maximizer still
                needs the covariance.

        Raises:
            ValueError: if the precision does not claim to be self-adjoint
                and positive semidefinite, an operator is on the wrong
                space, or the covariance is not the precision's inverse on
                the probe.
        """
        super().__init__(domain)
        required = Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE
        if required & precision.traits != required:
            raise ValueError(
                f"The precision must claim {required!s}; it claims "
                f"{precision.traits!s}."
            )
        if precision.domain != domain or precision.codomain != domain:
            raise ValueError(f"The precision must be an operator on {domain!r}.")
        if factor is not None and factor.codomain != domain:
            raise ValueError(f"The factor must map into {domain!r}.")
        if covariance is not None:
            if covariance.domain != domain or covariance.codomain != domain:
                raise ValueError(f"The covariance must be an operator on {domain!r}.")
            _check_inverse(domain, precision, covariance)
        self._precision = precision
        self._covariance = covariance
        self._factor = factor
        self._center = domain.zero() if center is None else center
        if radius is not None and radius <= 0.0:
            raise ValueError("A radius must be positive.")
        self._radius = None if radius is None else float(radius)

    @property
    def radius(self) -> float:
        """The radius of the ball this ellipsoid is in its operator's norm.

        Raises:
            ValueError: if the ellipsoid was not built as such a ball.
        """
        if self._radius is None:
            raise ValueError(
                "This ellipsoid was not built as a ball in an operator's norm, "
                "so it has no radius; its shape is its precision."
            )
        return self._radius

    def support_maximizer(self, direction: Any, /) -> Any:
        """``center + C q / sqrt((C q, q))`` given the covariance; converted without it.

        Args:
            direction: the direction to maximize along.

        Returns:
            The point of the ellipsoid furthest along it.

        Raises:
            ValueError: if the set is unbounded in the direction.
        """
        if self._covariance is None:
            return super().support_maximizer(direction)
        image = self._covariance(direction)
        scale = np.sqrt(self.domain.inner_product(image, direction))
        if scale == 0.0:
            return self._center
        return self.domain.add(self._center, self.domain.scale(1.0 / scale, image))

    def translate(self, vector: Any, /) -> "Ellipsoid":
        """The same ellipsoid, moved by a vector."""
        moved = Ellipsoid(
            self.domain,
            self._precision,
            center=self.domain.add(self._center, vector),
            factor=self._factor,
            radius=self._radius,
        )
        # The covariance was checked when this one was built.
        moved._covariance = self._covariance
        return moved

    @property
    def covariance(self) -> LinearOperator | None:
        """The inverse of the precision, if it was given; ``None`` otherwise."""
        return self._covariance

    @property
    def factor(self) -> LinearOperator | None:
        """A factor ``L`` of the covariance, ``C == L L*``, if one was given."""
        return self._factor

    @property
    def precision(self) -> LinearOperator:
        """The operator defining the Mahalanobis distance."""
        return self._precision

    @property
    def center(self) -> Any:
        """The center."""
        return self._center

    @property
    def boundary(self) -> "EllipsoidSurface":
        """The surface ``(P (x - c), x - c) == 1``, as a set in its own right.

        The surface carries the same precision and center, so it needs no
        covariance and exists whether or not this ellipsoid was given one.

        Returns:
            The bounding surface.
        """
        return EllipsoidSurface(self._domain, self._precision, center=self._center)

    @property
    def has_support_function(self) -> bool:
        """Only with the covariance in hand."""
        return self._covariance is not None

    @property
    def has_maximizer(self) -> bool:
        """Whether :meth:`support_maximizer` can exhibit a point."""
        return self._covariance is not None

    @property
    def has_level_function(self) -> bool:
        """Whether a level function describes the set."""
        return True

    @property
    def is_bounded(self) -> bool:
        """Whether the precision claims positive definiteness, which bounds the set."""
        return bool(Traits.POSITIVE_DEFINITE & self._precision.traits)

    def level_function(self) -> Functional:
        """The Mahalanobis form ``(P (x - c), x - c)``, at level one.

        Its Hessian ``2 P`` claims what the precision claims: positive
        definite for a bounded ellipsoid, semidefinite otherwise.
        """
        space, precision, center = self._domain, self._precision, self._center
        definite = Traits.POSITIVE_DEFINITE if self.is_bounded else Traits.NONE
        traits = Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE | definite
        return Functional.from_callables(
            space,
            self.mahalanobis_squared,
            gradient=lambda x: space.scale(2.0, precision(space.subtract(x, center))),
            hessian=lambda x: (precision * 2.0).with_traits(traits),
        )

    @property
    def level(self) -> float:
        """The level the level function is bounded by."""
        return 1.0

    def mahalanobis_squared(self, x: Any, /) -> float:
        """``(x - center, P (x - center))``."""
        offset = self._domain.subtract(x, self._center)
        return self._domain.inner_product(self._precision(offset), offset)

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """True when the Mahalanobis distance is at most one.

        Args:
            x: a vector of the space.
            rtol: how far outside still counts as inside. Scaled by the set's
                own size, so it means the same thing whatever the units are.

        Returns:
            Whether the set contains the point.
        """
        return self.mahalanobis_squared(x) <= 1.0 + rtol

    def project(
        self,
        x: Any,
        /,
        *,
        solver: Any = None,
        rtol: float = 1e-12,
        max_iterations: int = 100,
    ) -> Any:
        """The nearest point of the ellipsoid, by Newton on the secular equation.

        There is no closed form. The nearest point satisfies
        ``y + lambda P y == x - c`` for a multiplier ``lambda >= 0``, so
        ``y(lambda) == (I + lambda P)^-1 (x - c)`` and the constraint becomes a
        scalar equation ``phi(lambda) == (P y, y) - 1 == 0``. Each evaluation
        of ``phi`` costs a linear solve.

        ``phi`` is decreasing in ``lambda``, so Newton from zero converges from
        above and cannot overshoot into negative multipliers. A point already
        inside is returned unchanged, which is where ``lambda == 0``.

        **Matrix-free by default.** Each Newton step solves with
        ``I + lambda P`` twice, once for ``phi'`` and once for the new point,
        and each solve is conjugate gradients on that operator, which is
        positive definite: applications of ``P`` and nothing formed. The
        second solve starts from the first-order predictor ``y + d lambda
        y'``, which is what the derivative solve just computed, so it is
        usually a correction. On a small space a direct solver can be passed
        instead, at the cost of assembling and factorizing ``I + lambda P``
        twice per step.

        Args:
            x: the point to project.
            solver: how to invert ``I + lambda P`` -- a linear solver, or a
                callable taking the operator and returning one. Conjugate
                gradients at ``rtol=1e-12`` by default, so that the constraint
                can be met to *rtol*.
            rtol: on the constraint residual.
            max_iterations: the Newton cap.

        Returns:
            The nearest point of the ellipsoid.
        """
        from ..algebra.operators import LinearOperator
        from ..numerics.solvers import CGSolver, resolve_solver
        from ..traits import Traits

        space = self.domain
        offset = space.subtract(x, self._center)
        if float(space.inner_product(self._precision(offset), offset)) <= 1.0:
            return space.copy(x)

        chosen = resolve_solver(solver, self._precision, default=CGSolver(rtol=1e-12))
        identity = LinearOperator.identity(space)

        def shifted(multiplier: float) -> LinearOperator:
            return (identity + multiplier * self._precision).with_traits(
                Traits.POSITIVE_DEFINITE
            )

        multiplier = 0.0
        point = offset
        for _ in range(max_iterations):
            weighted = self._precision(point)
            residual = float(space.inner_product(weighted, point)) - 1.0
            if abs(residual) <= rtol:
                break
            # y'(lambda) == -(I + lambda P)^-1 P y, and phi' == 2 (P y, y').
            slope = space.negative(chosen(shifted(multiplier)).solve(weighted).solution)
            derivative = 2.0 * float(space.inner_product(weighted, slope))
            if derivative == 0.0:  # pragma: no cover - a degenerate ellipsoid
                break
            advanced = max(multiplier - residual / derivative, 0.0)
            guess = space.axpy(advanced - multiplier, slope, space.copy(point))
            point = chosen(shifted(advanced)).solve(offset, x0=guess).solution
            multiplier = advanced

        return space.add(self._center, point)

    def support_function(self) -> SupportFunction:
        """``(center, y) + sqrt((y, C y))`` with ``C`` the covariance; converted without it.

        Returns:
            The support function: closed form given the covariance, and
            otherwise the level-function conversion, one solve with the
            precision per evaluation and infinite where the set is
            unbounded.
        """
        if self._covariance is None:
            return super().support_function()
        return _EllipsoidSupport(
            self._domain, self._covariance, self._center, factor=self._factor
        )

    def __repr__(self) -> str:
        return f"Ellipsoid({self._domain!r})"


class _EllipsoidSupport(SupportFunction):
    """The support function of an ellipsoid given by its covariance.

    ``(center, y) + sqrt((C y, y))``; with a factor ``C == L L*`` in hand the
    root is ``||L* y||``, one adjoint application of the factor and no
    clamp, since a norm cannot go negative by rounding.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        covariance: LinearOperator,
        center: Any,
        /,
        *,
        factor: LinearOperator | None = None,
    ) -> None:
        super().__init__(domain)
        self._covariance = covariance
        self._center = center
        self._factor = factor

    def _value(self, y: Any) -> float:
        if self._factor is not None:
            pulled = self._factor.adjoint(y)
            root = self._factor.domain.norm(pulled)
        else:
            quadratic = self._domain.inner_product(self._covariance(y), y)
            root = float(np.sqrt(max(quadratic, 0.0)))
        return self._domain.inner_product(self._center, y) + root

    def _maximizer(self, y: Any) -> Any:
        space = self._domain
        mapped = self._covariance(y)
        norm = np.sqrt(max(space.inner_product(mapped, y), 0.0))
        if norm == 0.0:
            return space.copy(self._center)
        return space.axpy(1.0 / norm, mapped, space.copy(self._center))


class BallSurface(Subset):
    """The *surface* of a ball: ``{ x : ||x - c|| == r }``.

    Not convex, so it has no support function and no projection in the sense
    :class:`ConvexSet` means — but the nearest point on it is still well
    defined everywhere except the center, and it is what a norm constraint of
    the form ``||x|| == r`` describes. Constrained optimization is where this
    is wanted: an equality constraint, not an inequality.
    """

    def __init__(
        self, domain: HilbertSpace, /, *, radius: float = 1.0, center: Any = None
    ) -> None:
        """
        Args:
            domain: the space.
            radius: the radius, which must be positive.
            center: the center. Defaults to zero.

        Raises:
            ValueError: for a radius that is not positive.
        """
        super().__init__(domain)
        if radius <= 0.0:
            raise ValueError(f"The radius must be positive, got {radius}.")
        self._radius = float(radius)
        self._center = domain.zero() if center is None else center

    @property
    def radius(self) -> float:
        """The radius."""
        return self._radius

    @property
    def center(self) -> Any:
        """The center."""
        return self._center

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """Whether a point lies on the surface, to a relative tolerance.

        Unlike a solid set, membership here is a measure-zero condition, so it
        is only ever meaningful up to a tolerance and the tolerance is a named
        argument rather than a hidden constant.

        Args:
            x: the point to test.
            rtol: the relative tolerance on the radius.

        Returns:
            True when the point lies on the surface to that tolerance.
        """
        distance = self.domain.norm(self.domain.subtract(x, self._center))
        return abs(distance - self._radius) <= rtol * self._radius

    def project(self, x: Any, /) -> Any:
        """The nearest point on the surface.

        Undefined at the center, where every point of the surface is equally
        near, and that is raised rather than resolved by an arbitrary choice.

        Args:
            x: the point to project.

        Returns:
            The nearest point of the surface.

        Raises:
            ValueError: at the center, where every point of the surface is
                equally near and there is no nearest one to return.
        """
        offset = self.domain.subtract(x, self._center)
        distance = self.domain.norm(offset)
        if distance == 0.0:
            raise ValueError(
                "The center is equidistant from the whole surface, so it has "
                "no nearest point."
            )
        return self.domain.add(
            self._center, self.domain.scale(self._radius / distance, offset)
        )

    def sample(self, /, *, rng: Any = None) -> Any:
        """A point drawn uniformly over the surface.

        White noise projected outward: the standard construction, and the
        reason it is white noise rather than ``random`` is that only the former
        is isotropic in a space with a non-trivial metric.
        """
        return self.project(
            self.domain.add(self._center, self.domain.white_noise(rng=rng))
        )

    @property
    def boundary(self) -> "BallSurface":
        """Itself: a sphere has empty interior, so every point is a boundary point."""
        return self

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
        center: Any = None,
    ) -> None:
        """
        Args:
            domain: the space.
            precision: a positive definite operator on it.
            center: the center. Defaults to zero.

        Raises:
            ValueError: if the precision does not claim to be self-adjoint
                and positive definite.
        """
        super().__init__(domain)
        required = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        if required & precision.traits != required:
            raise ValueError(
                f"The precision must claim {required!s}; it claims "
                f"{precision.traits!s}."
            )
        self._precision = precision
        self._center = domain.zero() if center is None else center

    @property
    def precision(self) -> LinearOperator:
        """The shape operator."""
        return self._precision

    @property
    def center(self) -> Any:
        """The center."""
        return self._center

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """Whether a point lies on the surface, to a relative tolerance.

        Args:
            x: the point to test.
            rtol: the tolerance on the Mahalanobis value, whose target is one.

        Returns:
            True when the point lies on the surface to that tolerance.
        """
        offset = self.domain.subtract(x, self._center)
        value = self.domain.inner_product(self._precision(offset), offset)
        return abs(value - 1.0) <= rtol

    @property
    def boundary(self) -> "EllipsoidSurface":
        """Itself: the surface has empty interior, so every point is a boundary point."""
        return self

    def __repr__(self) -> str:
        return f"EllipsoidSurface({self.domain!r})"


class ConvexIntersection(ConvexSet):
    """The points in every one of several *convex* sets.

    An intersection of convex sets is convex, and keeping it a
    :class:`ConvexSet` -- rather than the general
    :class:`~pygeoinf.geometry.sets.Intersection`, which only tests
    membership -- keeps the projection, and with it proximal steps and use as
    a prior.

    The projection is Dykstra's algorithm over the parts, the same machinery
    :class:`Polytope` uses over its half-spaces -- a polytope being the special
    case where every part is a half-space.

    **There is no support function, only an upper bound.** ``min_i h_i`` bounds
    the intersection's support from above and is not equal to it: the
    maximizing point of one set need not lie in the others. It is available
    as :meth:`support_bound`, and :meth:`support_function` raises rather than
    return it as the truth.
    """

    def __init__(self, subsets: Sequence[ConvexSet], /) -> None:
        """
        Args:
            subsets: convex sets in a common space. Nested intersections are
                flattened.

        Raises:
            ValueError: if none are given.
            TypeError: if any is not convex.
        """
        parts: list[ConvexSet] = []
        for subset in subsets:
            if isinstance(subset, ConvexIntersection):
                parts.extend(subset.subsets)
            else:
                parts.append(subset)
        if not parts:
            raise ValueError("An intersection needs at least one subset.")
        for part in parts:
            if not isinstance(part, ConvexSet):
                raise TypeError(
                    f"Every part must be convex for the intersection to be; "
                    f"got {type(part).__name__}. Use Intersection for the "
                    "general case, which tests membership and nothing else."
                )
        super().__init__(parts[0].domain)
        for part in parts[1:]:
            self._check_domain(part)
        self._subsets = tuple(parts)

    @property
    def subsets(self) -> tuple[ConvexSet, ...]:
        """The intersected sets."""
        return self._subsets

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """True when every part contains the point.

        Args:
            x: a vector of the space.
            rtol: how far outside still counts as inside. Scaled by the set's
                own size, so it means the same thing whatever the units are.

        Returns:
            Whether the set contains the point.
        """
        return all(part.contains(x, rtol=rtol) for part in self._subsets)

    def project(self, x: Any, /, *, max_iterations: int = 1000) -> Any:
        """The nearest point of the intersection, by Dykstra's algorithm.

        Args:
            x: the point to project.
            max_iterations: the maximum number of cycles, each one projection per
                part. A part that projects iteratively -- an ellipsoid, say --
                makes each of those a small solve of its own. The cycle stops
                itself earlier once it has converged.

        Returns:
            The nearest point, to the accuracy the cycle reached.
        """
        return _dykstra(self.domain, self._subsets, x, max_iterations)

    def support_bound(self, direction: Any, /) -> float:
        """``min_i h_i(direction)``: an *upper bound* on the support value.

        Every part contains the intersection, so each part's support value
        bounds it from above and the least of them is the best such bound. It
        is not the support function: the point of one set attaining its own
        support need not lie in the others, and then the bound is strict.

        A part with no support function -- an ellipsoid given without its
        covariance, say -- is left out. The intersection lies inside every
        remaining part, so the least of their values is still an upper bound,
        if a looser one. An unbounded part such as a half-space contributes
        ``+inf`` in most directions, which leaves the minimum to the bounded
        parts.

        Args:
            direction: the direction to bound in.

        Returns:
            The bound.

        Raises:
            NotImplementedError: if no part has a support function, in which
                case there is no bound to give.
        """
        bounds = [
            float(part.support_function()(direction))
            for part in self._subsets
            if part.has_support_function
        ]
        if not bounds:
            raise NotImplementedError(
                "No part of this intersection has a support function, so "
                "there is no upper bound to report."
            )
        return min(bounds)

    @property
    def has_level_function(self) -> bool:
        """When every part has one: the set is where all of them are within level."""
        return all(part.has_level_function for part in self._subsets)

    def level_function(self) -> Functional:
        """``max_i (f_i(x) - level_i)``, at level zero.

        The active part's subgradient is one of the maximum's, when the
        part's level function has one.

        Raises:
            NotImplementedError: unless every part has a level function.
        """
        if not self.has_level_function:
            raise NotImplementedError(
                "An intersection has a level function only when every part does."
            )
        space = self._domain
        functionals = [part.level_function() for part in self._subsets]
        levels = [part.level for part in self._subsets]

        def excesses(x: Any) -> list[float]:
            return [f(x) - level for f, level in zip(functionals, levels)]

        subgradient = None
        if all(f.has_subgradient for f in functionals):

            def subgradient(x: Any) -> Any:
                return functionals[int(np.argmax(excesses(x)))].subgradient(x)

        return Functional.from_callables(
            space, lambda x: max(excesses(x)), subgradient=subgradient
        )

    @property
    def level(self) -> float:
        """The level the level function is bounded by."""
        return 0.0

    def support_function(self) -> SupportFunction:
        """Not available.

        Raises:
            NotImplementedError: always. There is no formula for the support
                function of an intersection. :meth:`support_bound` gives the
                upper bound ``min_i h_i``; a feasible property set, whose
                support values are computed by minimization, is the way to
                the true value.
        """
        raise NotImplementedError(
            "An intersection has no closed-form support function. Use "
            "support_bound() for the upper bound min_i h_i, or pose it as a "
            "FeasiblePropertySet, whose algorithms compute the true value."
        )

    def __repr__(self) -> str:
        return f"ConvexIntersection({', '.join(repr(s) for s in self._subsets)})"
