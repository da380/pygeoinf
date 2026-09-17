"""Subsets, convex sets and subspaces."""

import numpy as np
import pytest

from pygeoinf2 import EuclideanSpace, LinearOperator, Traits
from pygeoinf2.geometry import (
    AffineSubspace,
    ConvexSet,
    Ball,
    BallSurface,
    Ellipsoid,
    EmptySet,
    HalfSpace,
    Hyperplane,
    Intersection,
    LinearSubspace,
    OrthogonalProjector,
    Polytope,
    Union,
    UniversalSet,
)
from pygeoinf2.numerics.convex import ProximalGradient, SquaredDistance, SupportFunction
from pygeoinf2.numerics.solvers import CholeskySolver, LinearSolver
from pygeoinf2.symmetric_space import Sobolev
from pygeoinf2.testing import (
    check_operator,
    check_projection,
    check_traits,
)

from .conftest import make_dense_metric_space, make_weighted_space
from .doubles import OpaqueSpace


@pytest.fixture
def X():
    return Sobolev((16,), 2.0, 0.3)


class TestSetAlgebra:
    def test_the_trivial_sets(self, X, rng):
        x = X.random(rng=rng)
        assert not EmptySet(X).contains(x)
        assert UniversalSet(X).contains(x)

    def test_complement(self, X, rng):
        ball = Ball(X, radius=1.0)
        outside = ball.complement()
        inside_point = X.zero()
        assert ball.contains(inside_point)
        assert not outside.contains(inside_point)

    def test_a_double_complement_returns_the_original(self, X):
        ball = Ball(X, radius=1.0)
        assert ball.complement().complement() is ball

    def test_intersection_and_union(self, X, rng):
        small, large = Ball(X, radius=0.5), Ball(X, radius=5.0)
        point = X.scale(1.0 / max(X.norm(X.random(rng=rng)), 1e-30), X.random(rng=rng))
        both = small & large
        either = small | large
        assert either.contains(point)
        assert both.contains(point) == small.contains(point)

    def test_nested_operations_flatten(self, X):
        """And an intersection of *convex* sets stays convex, so it keeps its
        projection -- and with it every proximal method that needs one."""
        from pygeoinf2.geometry.convex import ConvexIntersection

        balls = [Ball(X, radius=r) for r in (1.0, 2.0, 3.0)]
        combined = (balls[0] & balls[1]) & balls[2]
        assert isinstance(combined, ConvexIntersection)
        assert len(combined.subsets) == 3

    def test_intersecting_with_a_general_set_falls_back(self, X):
        """Convexity is what buys the projection; without it there is nothing
        to keep, and the plain Intersection tests membership and no more."""
        from pygeoinf2.geometry.sets import Intersection, Subset

        class Blob(Subset):
            def contains(self, x, /, *, rtol=1e-9):
                return True

        combined = Ball(X, radius=1.0) & Blob(X)
        assert isinstance(combined, Intersection)
        assert not hasattr(combined, "project")

    def test_mismatched_domains_are_refused(self, X):
        with pytest.raises(ValueError, match="share a domain"):
            Ball(X) & Ball(EuclideanSpace(3))

    def test_an_empty_combination_is_refused(self):
        with pytest.raises(ValueError, match="at least one"):
            Union([])

    def test_it_works_without_coordinates(self, rng):
        space = OpaqueSpace(np.array([1.0, 4.0, 9.0]))
        ball = Ball(space, radius=2.0)
        assert ball.contains(space.zero())
        assert isinstance(ball & UniversalSet(space), Intersection)


class TestLevelSets:
    """A set from a functional and a level, v1's ``SublevelSet`` and
    ``LevelSet``, which v2 had no way to build."""

    @pytest.fixture(params=[make_weighted_space, make_dense_metric_space])
    def space(self, request):
        return request.param()

    @pytest.fixture
    def squared_norm(self, space):
        from pygeoinf2.algebra.operators import Functional

        return Functional.from_callables(space, lambda x: space.squared_norm(x))

    def test_membership_is_the_inequality_in_the_metric(self, space, squared_norm, rng):
        """On a dense Gram the squared norm is not the squared component
        norm, so the set is the ball in the space, not in the components."""
        from pygeoinf2.geometry import SublevelSet

        ball = SublevelSet(squared_norm, level=4.0)
        assert ball.functional is squared_norm
        assert ball.level == 4.0
        for _ in range(20):
            x = space.random(rng=rng)
            assert ball.contains(x) == (space.norm(x) <= 2.0)
        assert ball.contains(space.zero())
        assert ball.contains(space.scale(2.0 / space.norm(x), x))

    def test_the_tolerance_scales_with_the_level(self, space, squared_norm, rng):
        from pygeoinf2.geometry import SublevelSet

        x = space.random(rng=rng)
        on = space.scale(2.0 / space.norm(x), x)
        just_out = space.scale(1.0 + 1e-7, on)
        assert not SublevelSet(squared_norm, level=4.0).contains(just_out)
        assert SublevelSet(squared_norm, level=4.0).contains(just_out, rtol=1e-6)

    def test_the_boundary_is_the_level_set(self, space, squared_norm, rng):
        from pygeoinf2.geometry import LevelSet, SublevelSet

        ball = SublevelSet(squared_norm, level=4.0)
        sphere = ball.boundary
        assert isinstance(sphere, LevelSet)
        assert sphere.level == 4.0
        x = space.random(rng=rng)
        on = space.scale(2.0 / space.norm(x), x)
        assert sphere.contains(on)
        assert not sphere.contains(space.scale(0.5, on))
        assert not sphere.contains(space.scale(1.5, on))
        assert sphere.contains(space.scale(1.0 + 1e-10, on))
        # And the level set has empty interior, so it is its own boundary.
        assert sphere.boundary is sphere

    def test_the_level_set_agrees_with_the_sphere(self, space, rng):
        """The same set two ways: ``BallSurface`` and the level set of the
        norm, which must contain the same points."""
        from pygeoinf2.algebra.operators import Functional
        from pygeoinf2.geometry import LevelSet

        norm = Functional.from_callables(space, lambda x: space.norm(x))
        sphere = LevelSet(norm, level=1.5)
        surface = BallSurface(space, radius=1.5)
        for _ in range(10):
            x = space.random(rng=rng)
            for scale in (1.5 / space.norm(x), 1.0):
                point = space.scale(scale, x)
                assert sphere.contains(point) == surface.contains(point)

    def test_it_composes_with_the_set_algebra(self, space, squared_norm, rng):
        from pygeoinf2.geometry import SublevelSet

        shell = SublevelSet(squared_norm, level=4.0) & ~SublevelSet(
            squared_norm, level=1.0
        )
        x = space.random(rng=rng)
        assert shell.contains(space.scale(1.5 / space.norm(x), x))
        assert not shell.contains(space.scale(0.5 / space.norm(x), x))
        assert not shell.contains(space.scale(2.5 / space.norm(x), x))


class TestBoundaryOnTheBase:
    """``boundary`` is on the base, so generic code can ask; a set with no
    description of its boundary refuses rather than raising AttributeError."""

    def test_the_trivial_sets(self, X):
        assert isinstance(UniversalSet(X).boundary, EmptySet)
        empty = EmptySet(X)
        assert empty.boundary is empty

    def test_a_set_and_its_complement_share_a_boundary(self, X, rng):
        ball = Ball(X, radius=1.0)
        outside = ~ball
        assert isinstance(outside.boundary, BallSurface)
        x = X.random(rng=rng)
        on = X.scale(1.0 / X.norm(x), x)
        assert outside.boundary.contains(on)

    def test_a_surface_is_its_own_boundary(self, X, rng):
        for thin in (
            Hyperplane(X, X.random(rng=rng), offset=0.3),
            BallSurface(X, radius=1.0),
            Ellipsoid(X, LinearOperator.identity(X)).boundary,
        ):
            assert thin.boundary is thin

    def test_an_intersection_refuses_rather_than_lacking_the_attribute(self, X):
        combined = Ball(X, radius=1.0) & HalfSpace(X, X.basis_vector(0))
        with pytest.raises(NotImplementedError, match="boundary"):
            combined.boundary
        with pytest.raises(NotImplementedError, match="boundary"):
            (Ball(X, radius=1.0) | Ball(X, radius=2.0)).boundary


class TestConvexityCheck:
    """v1's randomised ``ConvexSubset.check``, now on the functional in
    ``testing``, where the other sampled axioms live."""

    def test_a_convex_functional_passes(self, X, rng):
        from pygeoinf2.testing import check_convexity

        check_convexity(SquaredDistance(X), rng=rng)
        check_convexity(Ball(X, radius=1.0).support_function(), rng=rng)
        check_convexity(Ball(X, radius=1.0).indicator(), rng=rng)

    def test_a_non_convex_one_fails(self, X, rng):
        from pygeoinf2.algebra.operators import Functional
        from pygeoinf2.testing import check_convexity

        concave = Functional.from_callables(X, lambda x: -X.squared_norm(x))
        with pytest.raises(AssertionError, match="convex"):
            check_convexity(concave, rng=rng)

    def test_the_metric_is_the_spaces_own(self, rng):
        """The combination is taken in the space and the values are the
        space's own norm, on a dense Gram, where the component norm would
        give a different functional."""
        from pygeoinf2.algebra.operators import Functional
        from pygeoinf2.testing import check_convexity

        space = make_dense_metric_space()
        check_convexity(
            Functional.from_callables(space, lambda x: space.norm(x) ** 3), rng=rng
        )


class TestDeclaredCapabilities:
    """A convex set says what it can do -- membership, projection, support
    function, maximiser, level function -- so generic code asks rather than
    catches. The closed forms have all of them; the combinators carry what
    their parts allow."""

    @pytest.fixture(params=[make_weighted_space, make_dense_metric_space])
    def space(self, request):
        return request.param()

    @staticmethod
    def flags(subset):
        return (
            subset.has_membership,
            subset.has_projection,
            subset.has_support_function,
            subset.has_maximiser,
            subset.has_level_function,
        )

    def test_the_closed_forms_have_everything(self, space, rng):
        precision = LinearOperator.identity(space) * 0.25
        for subset in (
            Ball(space, radius=1.5, centre=space.random(rng=rng)),
            HalfSpace(space, space.random(rng=rng), offset=0.3),
            Ellipsoid(
                space, precision, covariance=LinearOperator.identity(space) * 4.0
            ),
        ):
            assert self.flags(subset) == (True, True, True, True, True)
        plane = Hyperplane(space, space.random(rng=rng), offset=0.3)
        assert self.flags(plane) == (True, True, True, True, False)
        # An ellipsoid without its covariance has no support side.
        assert self.flags(Ellipsoid(space, precision)) == (
            True,
            True,
            False,
            False,
            True,
        )

    def test_the_level_function_describes_the_set(self, space, rng):
        """``f(x) <= level`` exactly where ``contains`` says so, on the
        boundary included."""
        precision = LinearOperator.identity(space) * 0.25
        for subset in (
            Ball(space, radius=1.5, centre=space.random(rng=rng)),
            HalfSpace(space, space.random(rng=rng), offset=0.3),
            Ellipsoid(space, precision),
        ):
            f, level = subset.level_function(), subset.level
            for _ in range(12):
                x = space.scale(2.0, space.random(rng=rng))
                assert (f(x) <= level * (1.0 + 1e-9) + 1e-12) == subset.contains(x)
            outside = space.scale(5.0, space.random(rng=rng))
            while subset.contains(outside):
                outside = space.scale(5.0, space.random(rng=rng))
            on = subset.project(outside)
            assert f(on) == pytest.approx(level, abs=1e-8)
            # And the gradient is the space's, checked against a finite step.
            x = space.random(rng=rng)
            step = space.random(rng=rng)
            h = 1e-6
            numerical = (f(space.axpy(h, step, space.copy(x))) - f(x)) / h
            assert space.inner_product(f.gradient(x), step) == pytest.approx(
                numerical, rel=1e-4, abs=1e-6
            )

    def test_the_combinators_carry_what_their_parts_allow(self, space, rng):
        ball = Ball(space, radius=1.0)
        # A half-space has no translate of its own, so this is the generic
        # translated set carrying its base's descriptions.
        wall = HalfSpace(space, space.random(rng=rng), offset=0.3)
        shift = space.random(rng=rng)
        moved = wall.translate(shift)
        assert self.flags(moved) == (True, True, True, True, True)
        x = space.random(rng=rng)
        assert moved.level_function()(x) == pytest.approx(
            wall.level_function()(space.subtract(x, shift))
        )
        assert moved.contains(x) == wall.contains(space.subtract(x, shift))
        total = ball + Ball(space, radius=2.0)
        assert self.flags(total) == (False, False, True, True, False)
        q = space.random(rng=rng)
        assert space.inner_product(total.support_maximiser(q), q) == pytest.approx(
            total.support_function()(q)
        )
        half = HalfSpace(space, space.random(rng=rng), offset=0.3)
        both = ball & half
        assert self.flags(both) == (True, True, False, False, True)
        f = both.level_function()
        for _ in range(12):
            y = space.scale(1.5, space.random(rng=rng))
            assert (f(y) <= 1e-12) == both.contains(y)
        oracle = ConvexSet.from_support_function(space, ball.support_function())
        assert self.flags(oracle) == (False, False, True, False, False)
        knowing = ConvexSet.from_support_function(
            space,
            ball.support_function(),
            maximiser=ball.support_maximiser,
            membership=lambda z, rtol: ball.contains(z, rtol=rtol),
        )
        assert self.flags(knowing) == (True, False, True, True, False)
        assert knowing.contains(space.zero())

    def test_an_oracle_sets_support_function_keeps_its_maximiser(self, space, rng):
        """The set was given a maximiser; the functional it hands back must
        carry it, or the subgradient a minimiser asks for is lost."""
        ball = Ball(space, radius=1.0)
        knowing = ConvexSet.from_support_function(
            space, ball.support_function(), maximiser=ball.support_maximiser
        )
        h = knowing.support_function()
        q = space.random(rng=rng)
        assert h.has_subgradient
        assert (
            space.norm(space.subtract(h.subgradient(q), ball.support_maximiser(q)))
            < 1e-12
        )
        bare = ConvexSet.from_support_function(space, ball.support_function())
        assert not bare.support_function().has_subgradient
        with pytest.raises(NotImplementedError):
            bare.support_function().subgradient(q)

    def test_a_polytopes_level_function_is_the_largest_excess(self, space, rng):
        from pygeoinf2.geometry.convex import Polytope

        planes = [HalfSpace(space, space.random(rng=rng), offset=0.5) for _ in range(3)]
        box = Polytope(space, planes, outer=True)
        assert box.has_level_function
        f = box.level_function()
        for _ in range(12):
            y = space.scale(2.0, space.random(rng=rng))
            assert (f(y) <= 1e-12) == box.contains(y)

    def test_the_intersection_bound_reads_the_flags(self, space, rng):
        """A part without a support function contributes nothing; a part
        whose support function *fails* is no longer silently skipped."""
        ball = Ball(space, radius=1.0)
        precision = LinearOperator.identity(space)
        bare = Ellipsoid(space, precision)  # no covariance: no support function
        combined = ball & bare
        q = space.random(rng=rng)
        assert combined.support_bound(q) == pytest.approx(ball.support_function()(q))


class TestConvexProjections:
    """``project`` means the nearest point, not a map onto the boundary."""

    def sets(self, X, rng):
        normal = X.random(rng=rng)
        return [
            ("ball", Ball(X, radius=1.5)),
            ("offset ball", Ball(X, radius=1.0, centre=X.random(rng=rng))),
            ("half-space", HalfSpace(X, normal, offset=0.4)),
            ("hyperplane", Hyperplane(X, normal, offset=0.4)),
        ]

    def test_they_are_metric_projections(self, X, rng):
        for name, subset in self.sets(X, rng):
            check_projection(subset, rng=rng)

    def test_a_feasible_point_is_left_alone(self, X, rng):
        """The property v1's HalfSpace.project does not have."""
        normal = X.random(rng=rng)
        half_space = HalfSpace(X, normal, offset=10.0)
        inside = X.zero()
        assert half_space.contains(inside)
        assert X.norm(X.subtract(half_space.project(inside), inside)) < 1e-12

        # A boundary projection would move it onto the plane instead.
        boundary = half_space.boundary
        assert X.norm(X.subtract(boundary.project(inside), inside)) > 1e-6

    def test_the_ball_projection_lands_on_the_sphere(self, X, rng):
        ball = Ball(X, radius=0.7)
        far = X.scale(50.0 / max(X.norm(X.random(rng=rng)), 1e-30), X.random(rng=rng))
        assert X.norm(ball.project(far)) == pytest.approx(0.7)

    def test_a_zero_normal_is_refused(self, X):
        with pytest.raises(ValueError, match="nonzero"):
            HalfSpace(X, X.zero())
        with pytest.raises(ValueError, match="nonzero"):
            Hyperplane(X, X.zero())

    def test_a_negative_radius_is_refused(self, X):
        with pytest.raises(ValueError, match="not be negative"):
            Ball(X, radius=-1.0)

    def test_a_zero_radius_is_the_single_point_at_the_centre(self, X, rng):
        """The degenerate ball says "exactly this", which is what error-free
        data are. Refusing it is what stopped the Backus routes running with no
        error measure at all."""
        centre = X.random(rng=rng)
        point = Ball(X, radius=0.0, centre=centre)
        elsewhere = X.add(centre, X.random(rng=rng))

        assert point.contains(centre)
        assert not point.contains(elsewhere)
        assert X.norm(X.subtract(point.project(elsewhere), centre)) < 1e-12
        # Its support function is the point support (centre, y).
        direction = X.random(rng=rng)
        assert point.support_function()(direction) == pytest.approx(
            X.inner_product(centre, direction)
        )


class TestThreeViewsOfOneSet:
    """A set, its indicator and its support function are one object."""

    def test_the_indicator_prox_is_the_projection(self, X, rng):
        for subset in (
            Ball(X, radius=1.2),
            HalfSpace(X, X.random(rng=rng), offset=0.3),
            Hyperplane(X, X.random(rng=rng), offset=0.3),
        ):
            indicator = subset.indicator()
            x = X.random(rng=rng)
            assert indicator.has_prox
            assert X.norm(X.subtract(indicator.prox(x, 1.0), subset.project(x))) < 1e-12

    def test_the_indicator_is_zero_inside_and_infinite_outside(self, X, rng):
        ball = Ball(X, radius=1.0)
        indicator = ball.indicator()
        assert indicator(X.zero()) == 0.0
        far = X.scale(100.0 / max(X.norm(X.random(rng=rng)), 1e-30), X.random(rng=rng))
        assert indicator(far) == float("inf")

    def test_the_support_function_matches(self, X, rng):
        ball = Ball(X, radius=2.0)
        support = ball.support_function()
        assert isinstance(support, SupportFunction)
        y = X.random(rng=rng)
        assert support(y) == pytest.approx(2.0 * X.norm(y))

    def test_the_conjugate_of_the_indicator_is_the_support(self, X, rng):
        subset = Ball(X, radius=1.7)
        y = X.random(rng=rng)
        assert subset.indicator().conjugate()(y) == pytest.approx(
            subset.support_function()(y)
        )

    def test_a_set_drops_into_a_proximal_method(self, X, rng):
        """The payoff: a hard constraint needs no extra machinery."""
        centre = X.random(rng=rng)
        constraint = Ball(X, radius=0.25)
        result = ProximalGradient(max_iterations=2000, gtol=1e-14).minimise(
            SquaredDistance(X, centre=centre),
            X.random(rng=rng),
            nonsmooth=constraint.indicator(),
        )
        assert constraint.contains(result.minimiser, rtol=1e-6)
        assert X.norm(result.minimiser) == pytest.approx(0.25, rel=1e-6)


class TestEllipsoid:
    @pytest.fixture
    def ellipsoid(self, X, rng):
        precision = X.invariant_operator(lambda values: 1.0 + values)
        covariance = precision.inverse
        return Ellipsoid(X, precision, covariance=covariance)

    def test_membership_is_the_mahalanobis_distance(self, X, ellipsoid, rng):
        assert ellipsoid.contains(X.zero())
        x = X.random(rng=rng)
        scaled = X.scale(1.0 / np.sqrt(ellipsoid.mahalanobis_squared(x)), x)
        assert ellipsoid.contains(scaled, rtol=1e-8)
        assert not ellipsoid.contains(X.scale(1.001, scaled))

    def test_the_support_function_is_the_covariance_norm(self, X, ellipsoid, rng):
        support = ellipsoid.support_function()
        y = X.random(rng=rng)
        expected = np.sqrt(X.inner_product(ellipsoid.precision.inverse(y), y))
        assert support(y) == pytest.approx(expected)

    def test_the_maximiser_attains_the_supremum(self, X, ellipsoid, rng):
        support = ellipsoid.support_function()
        y = X.random(rng=rng)
        maximiser = support.subgradient(y)
        assert ellipsoid.contains(maximiser, rtol=1e-8)
        assert X.inner_product(maximiser, y) == pytest.approx(support(y))

    def test_the_projection_lands_on_the_boundary(self, X, ellipsoid, rng):
        """It used to raise. A set that cannot project cannot be used by
        anything needing a proximal step -- the primal-dual route, a proximal
        method, an intersection by Dykstra -- so a few linear solves is a
        better answer than not being available.

        Newton on the secular equation ``(P y, y) == 1``, which is exact to
        twelve digits."""
        for _ in range(4):
            point = X.random(rng=rng)
            if ellipsoid.contains(point):
                continue
            projected = ellipsoid.project(point)
            offset = X.subtract(projected, ellipsoid.centre)
            assert X.inner_product(
                ellipsoid._precision(offset), offset
            ) == pytest.approx(1.0, abs=1e-10)

    def test_a_point_inside_is_left_where_it_is(self, X, ellipsoid):
        centre = ellipsoid.centre
        assert X.norm(X.subtract(ellipsoid.project(centre), centre)) < 1e-14

    def test_it_is_the_nearest_point(self, X, ellipsoid, rng):
        """Checked against a constrained optimiser, not merely against the
        constraint: landing on the boundary is necessary and not sufficient."""
        from scipy.optimize import minimize

        def constraint(components):
            """The ellipsoid's own, through the space's inner product -- which
            on a non-diagonal metric is not the component dot product."""
            offset = X.subtract(X.from_components(components), ellipsoid.centre)
            return 1.0 - X.inner_product(ellipsoid._precision(offset), offset)

        for _ in range(3):
            point = X.scale(3.0, X.random(rng=rng))
            if ellipsoid.contains(point):
                continue
            projected = ellipsoid.project(point)

            reference = minimize(
                lambda z: X.squared_norm(X.subtract(X.from_components(z), point)),
                X.to_components(projected),
                constraints=[{"type": "ineq", "fun": constraint}],
            )
            assert X.norm(X.subtract(projected, point)) == pytest.approx(
                X.norm(X.subtract(X.from_components(reference.x), point)),
                rel=1e-5,
            )

    def test_the_projection_is_matrix_free_by_default(self, rng, monkeypatch):
        """Conjugate gradients on ``I + lambda P``, nothing extracted. A
        Cholesky factorisation used to be the default, twice per Newton step:
        3.5 s at dimension 1500, and 31 s with 54 000 applications when the
        precision had to be probed for its matrix. A direct solver by name
        gives the same point."""
        from pygeoinf2.numerics.solvers import CholeskySolver

        for space in (EuclideanSpace(40), make_dense_metric_space(40)):
            root = rng.normal(size=(40, 40)) / np.sqrt(40)
            components = np.linalg.solve(
                space.gram_matrix(), root @ root.T + 0.5 * np.identity(40)
            )
            precision = LinearOperator.from_matrix(
                space,
                space,
                components,
                traits=Traits.POSITIVE_DEFINITE,
                form="components",
            )
            ellipsoid = Ellipsoid(space, precision)
            point = space.scale(3.0, space.random(rng=rng))

            with monkeypatch.context() as patched:
                patched.setattr(
                    LinearOperator,
                    "matrix",
                    lambda *a, **k: (_ for _ in ()).throw(AssertionError("dense")),
                )
                projected = ellipsoid.project(point)
            assert ellipsoid.mahalanobis_squared(projected) == pytest.approx(
                1.0, abs=1e-10
            )
            direct = ellipsoid.project(point, solver=CholeskySolver())
            assert space.norm(space.subtract(projected, direct)) < 1e-8 * space.norm(
                direct
            )

    def test_an_indefinite_precision_is_refused(self, X, rng):
        bad = LinearOperator.self_adjoint(X, lambda x: x)
        with pytest.raises(ValueError, match="must claim"):
            Ellipsoid(X, bad)

    def test_the_support_function_needs_the_covariance(self, X):
        precision = X.invariant_operator(lambda values: 1.0 + values)
        with pytest.raises(NotImplementedError, match="covariance"):
            Ellipsoid(X, precision).support_function()


class TestBoundaries:
    """A solid set knows its surface. v1's ``Ball.boundary`` and
    ``Ellipsoid.boundary`` were not restored with the rest of the set algebra,
    and the surfaces they return are what an *equality* constraint is -- the
    ball surface projects onto it and samples uniformly over it, which the
    solid ball does not."""

    def test_a_balls_boundary_is_its_surface(self, X, rng):
        from pygeoinf2.geometry.convex import BallSurface

        centre = X.random(rng=rng)
        ball = Ball(X, radius=1.3, centre=centre)
        surface = ball.boundary
        assert isinstance(surface, BallSurface)
        assert surface.radius == ball.radius
        assert X.norm(X.subtract(surface.centre, centre)) < 1e-14

        # It is the boundary in the sense that matters: on it, not in it.
        outside = X.add(centre, X.scale(4.0, X.random(rng=rng)))
        landed = surface.project(outside)
        assert surface.contains(landed)
        assert ball.contains(landed, rtol=1e-9)
        assert surface.contains(surface.sample(rng=rng))
        # and the centre is in the ball but not on its boundary.
        assert ball.contains(centre) and not surface.contains(centre)

    def test_a_point_has_no_surface(self, X, rng):
        """A ball of zero radius is the single point at its centre. Its
        boundary in the ambient space is itself, which is not a surface, so
        this refuses rather than returning a radius of zero."""
        with pytest.raises(ValueError, match="must be positive"):
            Ball(X, radius=0.0, centre=X.random(rng=rng)).boundary

    def test_an_ellipsoids_boundary_is_its_surface(self, X, rng):
        from pygeoinf2.geometry.convex import EllipsoidSurface

        precision = X.invariant_operator(lambda values: 1.0 + values)
        ellipsoid = Ellipsoid(X, precision, covariance=precision.inverse)
        surface = ellipsoid.boundary
        assert isinstance(surface, EllipsoidSurface)
        assert surface.precision is ellipsoid.precision

        point = X.random(rng=rng)
        landed = ellipsoid.project(X.scale(4.0, point))
        assert surface.contains(landed, rtol=1e-8)
        assert not surface.contains(ellipsoid.centre)

    def test_it_needs_no_covariance(self, X):
        """The surface is defined by the precision, so an ellipsoid built
        without a covariance -- which cannot give its support function -- can
        still give its boundary."""
        precision = X.invariant_operator(lambda values: 1.0 + values)
        assert Ellipsoid(X, precision).boundary is not None


class TestProjectors:
    def test_a_projector_carries_its_structure(self, X, rng):
        projector = OrthogonalProjector.from_basis(
            X, [X.random(rng=rng) for _ in range(4)]
        )
        assert Traits.SELF_ADJOINT & projector.traits
        assert Traits.IDEMPOTENT & projector.traits
        assert Traits.POSITIVE_SEMIDEFINITE & projector.traits  # by closure
        check_operator(projector, rng=rng)
        check_traits(projector, rng=rng)

    def test_the_complement_is_a_projector(self, X, rng):
        """Not a generic difference, which would forget it is idempotent."""
        projector = OrthogonalProjector.from_basis(
            X, [X.random(rng=rng) for _ in range(3)]
        )
        complement = projector.complement()
        assert isinstance(complement, OrthogonalProjector)
        assert Traits.IDEMPOTENT & complement.traits
        check_traits(complement, rng=rng)
        assert complement.complement() is projector

    def test_they_sum_to_the_identity(self, X, rng):
        projector = OrthogonalProjector.from_basis(
            X, [X.random(rng=rng) for _ in range(3)]
        )
        x = X.random(rng=rng)
        assert (
            X.norm(X.subtract(X.add(projector(x), projector.complement()(x)), x))
            < 1e-10
        )

    def test_an_empty_basis_projects_to_zero(self, X, rng):
        projector = OrthogonalProjector.from_basis(X, [])
        assert X.norm(projector(X.random(rng=rng))) == pytest.approx(0.0)

    def test_it_is_coordinate_free(self, rng):
        space = OpaqueSpace(np.array([1.0, 4.0, 9.0, 0.25]))
        projector = OrthogonalProjector.from_basis(
            space, [space.random(rng=rng) for _ in range(2)]
        )
        check_operator(projector, rng=rng)
        check_traits(projector, rng=rng)


class TestSubspaces:
    @pytest.fixture
    def problem(self, X, rng):
        Y = EuclideanSpace(4)
        A = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(4, X.dim)), form="components"
        )
        return X, Y, A

    def test_a_span(self, X, rng):
        vectors = [X.random(rng=rng) for _ in range(5)]
        subspace = LinearSubspace.from_basis(X, vectors)
        check_projection(subspace, rng=rng)
        assert subspace.dimension() == 5
        for vector in vectors:
            assert X.norm(X.subtract(subspace.project(vector), vector)) < 1e-9

    def test_a_kernel(self, problem, rng):
        X, Y, A = problem
        subspace = LinearSubspace.from_kernel(A)
        check_projection(subspace, rng=rng)
        assert np.max(np.abs(A(subspace.project(X.random(rng=rng))))) < 1e-8
        assert subspace.dimension() == X.dim - 4

    def test_the_dimension_uses_the_component_trace(self, rng):
        """``sum (P e_i, e_i)`` is the trace only on an orthonormal basis.

        On a weighted space it is the Galerkin diagonal and means nothing,
        which is the derivative-and-gradient confusion in another costume.
        """
        for space in (Sobolev((16,), 2.0, 0.3), EuclideanSpace(16)):
            vectors = [space.random(rng=rng) for _ in range(3)]
            assert LinearSubspace.from_basis(space, vectors).dimension() == 3

    def test_the_orthogonal_complement(self, X, rng):
        subspace = LinearSubspace.from_basis(X, [X.random(rng=rng) for _ in range(4)])
        complement = subspace.complement()
        assert isinstance(complement, LinearSubspace)
        assert complement.dimension() == X.dim - 4
        x = X.random(rng=rng)
        assert abs(X.inner_product(subspace.project(x), complement.project(x))) < 1e-9

    def test_an_affine_subspace_solves_the_equation(self, problem, rng):
        X, Y, A = problem
        value = Y.random(rng=rng)
        subspace = AffineSubspace.from_linear_equation(A, value)
        check_projection(subspace, rng=rng)
        assert np.allclose(A(subspace.project(X.random(rng=rng))), value, atol=1e-8)

    def test_its_translation_is_the_minimum_norm_solution(self, problem, rng):
        X, Y, A = problem
        value = Y.random(rng=rng)
        subspace = AffineSubspace.from_linear_equation(A, value)
        assert np.allclose(A(subspace.translation), value, atol=1e-8)
        for _ in range(10):
            other = subspace.project(X.random(rng=rng))
            assert X.norm(subspace.translation) <= X.norm(other) + 1e-8

    def test_the_tangent_of_an_affine_subspace(self, problem, rng):
        X, Y, A = problem
        subspace = AffineSubspace.from_linear_equation(A, Y.random(rng=rng))
        tangent = subspace.tangent
        assert isinstance(tangent, LinearSubspace)
        assert np.max(np.abs(A(tangent.project(X.random(rng=rng))))) < 1e-8

    def test_a_linear_constraint_in_a_proximal_method(self, problem, rng):
        """A subspace is a convex set, so it constrains like any other."""
        X, Y, A = problem
        value = Y.random(rng=rng)
        subspace = AffineSubspace.from_linear_equation(A, value)
        result = ProximalGradient(max_iterations=2000, gtol=1e-14).minimise(
            SquaredDistance(X, centre=X.random(rng=rng)),
            X.random(rng=rng),
            nonsmooth=subspace.indicator(),
        )
        assert np.allclose(A(result.minimiser), value, atol=1e-7)


class TestOneNormalInverse:
    """``(A A*)^-1`` is built once and shared.

    Three constructions need it -- the kernel projector, the minimum-norm
    translation and the pseudo-inverse -- and each used to build its own. With
    an iterative solver that is a wasted object; with a direct one it is the
    matrix of ``A A*`` extracted and factorised again each time, and a subspace
    built from an equation did that twice in one constructor call.
    """

    class Counting(LinearSolver):
        """A solver that records how often it was asked to invert."""

        def __init__(self):
            self.inner = CholeskySolver()
            self.count = 0

        def _invert(self, operator):
            self.count += 1
            return self.inner(operator)

    @pytest.fixture
    def problem(self, rng):
        space = EuclideanSpace(40)
        codomain = EuclideanSpace(6)
        operator = LinearOperator.from_matrix(
            space, codomain, rng.normal(size=(6, 40)), form="components"
        )
        return space, codomain, operator

    def test_an_affine_subspace_builds_it_once(self, problem, rng):
        space, codomain, operator = problem
        solver = self.Counting()
        subspace = AffineSubspace.from_linear_equation(
            operator, codomain.random(rng=rng), solver=solver
        )
        assert solver.count == 1
        subspace.pseudo_inverse()
        assert solver.count == 1
        subspace.with_constraint_value(codomain.random(rng=rng))
        assert solver.count == 1
        subspace.with_translation(space.random(rng=rng)).pseudo_inverse()
        assert solver.count == 1

    def test_a_kernel_builds_it_once(self, problem):
        _, _, operator = problem
        solver = self.Counting()
        subspace = LinearSubspace.from_kernel(operator, solver=solver)
        assert solver.count == 1
        subspace.pseudo_inverse()
        assert solver.count == 1

    def test_sharing_it_does_not_change_the_answers(self, problem, rng):
        """The projector is the same object in `with_constraint_value` because
        the kernel does not move; the translation does."""
        space, codomain, operator = problem
        first = codomain.random(rng=rng)
        second = codomain.random(rng=rng)
        subspace = AffineSubspace.from_linear_equation(
            operator, first, solver=CholeskySolver()
        )
        moved = subspace.with_constraint_value(second)
        assert np.allclose(operator(moved.translation), second, atol=1e-9)
        assert np.allclose(
            operator(moved.project(space.random(rng=rng))), second, atol=1e-9
        )
        check_projection(moved, rng=rng)
        # and the pseudo-inverse is still the minimum-norm right inverse.
        recovered = subspace.pseudo_inverse()(second)
        assert np.allclose(operator(recovered), second, atol=1e-9)
        assert space.norm(recovered) <= space.norm(moved.translation) + 1e-9


class TestTheMetricEntersEveryProjection:
    """The metric rule, applied to the sets.

    Everything in this module -- a projection, a support function, a support
    maximiser, a subspace's dimension -- is written with ``inner_product``,
    ``norm`` and adjoints, and every one of those expressions is also correct
    on components when the Gram matrix is the identity. Only a non-diagonal
    Gram tells the two apart, and the rest of this file runs on a Sobolev
    space over a box, whose metric is diagonal in the basis it uses.

    So these repeat the substance of the checks above on
    ``make_dense_metric_space``, and against closed forms rather than by
    sampling wherever there is one.
    """

    @pytest.fixture
    def space(self):
        return make_dense_metric_space(6)

    @pytest.fixture
    def normal(self, space, rng):
        return space.random(rng=rng)

    def test_the_ball(self, space, rng):
        """``h(y) = r ||y|| + (c, y)`` in the *space's* norm, and the point
        attaining it is on the sphere of that norm."""
        centre = space.random(rng=rng)
        ball = Ball(space, radius=1.4, centre=centre)
        check_projection(ball, rng=rng)

        direction = space.random(rng=rng)
        support = ball.support_function()
        expected = 1.4 * space.norm(direction) + space.inner_product(centre, direction)
        assert support(direction) == pytest.approx(expected)

        maximiser = ball.support_maximiser(direction)
        assert space.inner_product(maximiser, direction) == pytest.approx(expected)
        assert space.norm(space.subtract(maximiser, centre)) == pytest.approx(1.4)

        # And the projection is the closed form, not the component one: on a
        # dense Gram those differ.
        far = space.add(centre, space.scale(9.0, direction))
        offset = space.subtract(far, centre)
        landed = space.axpy(1.4 / space.norm(offset), offset, space.copy(centre))
        assert space.norm(space.subtract(ball.project(far), landed)) < 1e-12

    def test_the_half_space_and_the_hyperplane(self, space, normal, rng):
        half = HalfSpace(space, normal, offset=0.4)
        plane = Hyperplane(space, normal, offset=0.4)
        check_projection(half, rng=rng)
        check_projection(plane, rng=rng)

        x = space.random(rng=rng)
        excess = space.inner_product(normal, x) - 0.4
        # The step is the residual over ||n||^2 in the space's inner product.
        step = excess / space.squared_norm(normal)
        expected = space.axpy(-step, normal, space.copy(x))
        assert space.norm(space.subtract(plane.project(x), expected)) < 1e-12
        if excess > 0.0:
            assert space.norm(space.subtract(half.project(x), expected)) < 1e-12
        else:
            assert space.norm(space.subtract(half.project(x), x)) < 1e-12

    def test_the_ball_surface(self, space, rng):
        """Its ``sample`` is white noise projected outward, and white noise is
        the only isotropic draw on a space with a metric."""
        from pygeoinf2.geometry.convex import BallSurface

        surface = BallSurface(space, radius=2.0, centre=space.random(rng=rng))
        for _ in range(5):
            assert surface.contains(surface.sample(rng=rng))
            assert surface.contains(surface.project(space.random(rng=rng)))

    def test_the_ellipsoid(self, space, rng):
        """The precision is self-adjoint on this space -- which is symmetry of
        its *Galerkin* matrix, not of its components -- and the support
        function is the covariance norm in the space's inner product."""
        root = rng.normal(size=(space.dim, space.dim))
        precision = LinearOperator.from_matrix(
            space,
            space,
            root @ root.T + space.dim * np.identity(space.dim),
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
            form="galerkin",
        )
        check_traits(precision, rng=rng)
        covariance = CholeskySolver()(precision)
        ellipsoid = Ellipsoid(space, precision, covariance=covariance)

        direction = space.random(rng=rng)
        support = ellipsoid.support_function()
        expected = np.sqrt(space.inner_product(covariance(direction), direction))
        assert support(direction) == pytest.approx(expected)

        maximiser = ellipsoid.support_maximiser(direction)
        assert space.inner_product(maximiser, direction) == pytest.approx(expected)
        assert ellipsoid.contains(maximiser, rtol=1e-8)

        # Newton on the secular equation, with a solve per step, on a dense
        # Gram: the projection lands on the boundary and is the nearest point.
        point = space.scale(6.0, space.random(rng=rng))
        projected = ellipsoid.project(point)
        assert ellipsoid.mahalanobis_squared(projected) == pytest.approx(1.0, abs=1e-9)
        for _ in range(30):
            other = ellipsoid.project(
                space.add(projected, space.scale(0.05, space.random(rng=rng)))
            )
            assert (
                space.norm(space.subtract(other, point))
                >= space.norm(space.subtract(projected, point)) - 1e-9
            )

    def test_the_projector_and_the_dimension(self, space, rng):
        """``P`` is self-adjoint *in the space's inner product*, and the
        dimension is the trace of its component matrix -- not the sum of
        ``(P e_i, e_i)``, which on this space is the Galerkin diagonal and
        means nothing."""
        vectors = [space.random(rng=rng) for _ in range(3)]
        projector = OrthogonalProjector.from_basis(space, vectors)
        check_operator(projector, rng=rng)
        check_traits(projector, rng=rng)

        subspace = LinearSubspace(projector)
        assert subspace.dimension() == 3
        assert subspace.complement().dimension() == space.dim - 3
        for vector in vectors:
            assert space.norm(space.subtract(subspace.project(vector), vector)) < 1e-9

        galerkin = sum(
            space.inner_product(projector(space.basis_vector(i)), space.basis_vector(i))
            for i in range(space.dim)
        )
        assert abs(galerkin - 3.0) > 0.1, "the fixture no longer distinguishes them"

    def test_the_affine_subspace(self, space, rng):
        """The translation is the minimum-norm solution in the space's norm,
        which is where the adjoint -- and so the metric -- enters."""
        codomain = EuclideanSpace(2)
        operator = LinearOperator.from_matrix(
            space, codomain, rng.normal(size=(2, space.dim)), form="components"
        )
        value = codomain.random(rng=rng)
        subspace = AffineSubspace.from_linear_equation(operator, value)
        check_projection(subspace, rng=rng)

        assert np.allclose(operator(subspace.translation), value, atol=1e-8)
        for _ in range(20):
            other = subspace.project(space.random(rng=rng))
            assert space.norm(subspace.translation) <= space.norm(other) + 1e-8
        assert subspace.tangent.dimension() == space.dim - 2

    def test_the_intersection(self, space, rng):
        """Dykstra's corrections are vectors of the space, so the whole loop
        is in its metric."""
        combined = Ball(space, radius=1.0) & HalfSpace(
            space, space.basis_vector(0), offset=-0.2
        )
        check_projection(combined, rng=rng)


class TestPolytopeProjection:
    """The nearest point of an intersection of half-spaces, by Dykstra."""

    def test_it_is_the_nearest_point_not_merely_a_feasible_one(self):
        """The counterexample that showed cyclic projection was not a
        projection: on ``{x <= 0}`` and ``{x + y <= 0}`` from ``(1, 0.5)`` it
        returned ``(-0.25, 0.25)`` at squared distance 1.625, where the origin
        is feasible at 1.25."""
        space = EuclideanSpace(2)
        polytope = Polytope(
            space,
            [
                HalfSpace(space, np.array([1.0, 0.0])),
                HalfSpace(space, np.array([1.0, 1.0])),
            ],
            outer=True,
        )
        point = np.array([1.0, 0.5])
        projected = polytope.project(point)

        assert projected == pytest.approx(np.zeros(2), abs=1e-10)
        assert space.norm(space.subtract(point, projected)) ** 2 == pytest.approx(1.25)

    @pytest.mark.parametrize(
        "build", [lambda: EuclideanSpace(3), make_dense_metric_space]
    )
    def test_it_satisfies_the_projection_axioms(self, build, rng):
        """Including on a non-diagonal Gram, where 'nearest' is nearest in the
        space's own norm rather than in components."""
        space = build()
        polytope = Polytope(
            space,
            [
                HalfSpace(space, space.from_components(normal), offset=offset)
                for normal, offset in zip(
                    [
                        np.array([1.0, 0.2, -0.3]),
                        np.array([-0.4, 1.0, 0.1]),
                        np.array([0.2, -0.5, 1.0]),
                    ],
                    [-0.2, 0.1, -0.3],
                )
            ],
            outer=True,
        )
        check_projection(polytope, rng=rng)

    def test_the_indicators_prox_is_that_projection(self, rng):
        """The reason it has to be the projection: a proximal method takes this
        as the prox, and a prox that is not the projection has the wrong fixed
        point."""
        space = EuclideanSpace(2)
        polytope = Polytope(
            space,
            [
                HalfSpace(space, np.array([1.0, 0.0])),
                HalfSpace(space, np.array([1.0, 1.0])),
            ],
            outer=True,
        )
        point = np.array([1.0, 0.5])
        assert polytope.indicator().prox(point, 0.7) == pytest.approx(
            polytope.project(point), abs=1e-10
        )


class TestHalfSpaceSupport:
    """v1's ``HalfSpaceSupportFunction``: extended-real valued, finite only
    along the outward normal, with the boundary's least-norm point as the
    maximiser. v2's ``HalfSpace`` inherited the base refusal."""

    @pytest.fixture(params=[make_weighted_space, make_dense_metric_space])
    def space(self, request):
        return request.param()

    def test_along_the_normal_it_is_the_scaled_offset(self, space, rng):
        normal = space.random(rng=rng)
        half = HalfSpace(space, normal, offset=0.4)
        h = half.support_function()
        assert h(normal) == pytest.approx(0.4)
        assert h(space.scale(2.5, normal)) == pytest.approx(1.0)
        assert h(space.zero()) == 0.0

    def test_elsewhere_it_is_infinite(self, space, rng):
        """Including against the normal: the pairing grows along the ray into
        the set, so the supremum is infinite, not ``-offset``."""
        normal = space.random(rng=rng)
        half = HalfSpace(space, normal, offset=0.4)
        h = half.support_function()
        assert h(space.random(rng=rng)) == float("inf")
        assert h(space.scale(-1.0, normal)) == float("inf")

    def test_parallel_is_decided_in_the_metric(self, rng):
        """On a dense Gram a direction with the normal's *components* is not
        parallel to it in the space, and one that is has different
        components; the test is on the residual in the space's own norm."""
        space = make_dense_metric_space()
        normal = space.from_components(np.array([1.0, 0.0, 0.0]))
        h = HalfSpace(space, normal, offset=1.0).support_function()
        assert h(normal) == pytest.approx(1.0)
        # Nearly parallel, within the tolerance: the tiny residual is scaled
        # by the direction's norm, so a large multiple still counts.
        nearly = space.axpy(1e-14, space.basis_vector(1), space.scale(1e6, normal))
        assert h(nearly) == pytest.approx(1e6)
        assert h(space.axpy(1e-6, space.basis_vector(1), space.copy(normal))) == float(
            "inf"
        )

    def test_the_maximiser_is_the_least_norm_boundary_point(self, space, rng):
        normal = space.random(rng=rng)
        half = HalfSpace(space, normal, offset=0.4)
        direction = space.scale(3.0, normal)
        point = half.support_maximiser(direction)

        assert half.boundary.contains(point)
        assert space.inner_product(point, direction) == pytest.approx(
            half.support_function()(direction)
        )
        # Least norm: the plane's nearest point to the origin.
        assert (
            space.norm(space.subtract(point, half.boundary.project(space.zero())))
            < 1e-12
        )
        # And the subgradient route agrees, which is what a bundle method uses.
        assert (
            space.norm(
                space.subtract(half.support_function().subgradient(direction), point)
            )
            < 1e-12
        )

    def test_an_unbounded_direction_has_no_maximiser(self, space, rng):
        half = HalfSpace(space, space.random(rng=rng), offset=0.4)
        with pytest.raises(ValueError, match="infinite"):
            half.support_maximiser(space.random(rng=rng))

    def test_the_hyperplane_is_finite_both_ways(self, space, rng):
        normal = space.random(rng=rng)
        plane = Hyperplane(space, normal, offset=0.4)
        h = plane.support_function()
        assert h(normal) == pytest.approx(0.4)
        assert h(space.scale(-2.0, normal)) == pytest.approx(-0.8)
        assert h(space.random(rng=rng)) == float("inf")
        point = plane.support_maximiser(space.scale(-2.0, normal))
        assert plane.contains(point)
        assert space.inner_product(point, space.scale(-2.0, normal)) == pytest.approx(
            -0.8
        )

    def test_it_composes_with_the_set_algebra(self, space, rng):
        """A translated half-space shifts the finite value and leaves the
        infinite ones infinite; a Minkowski sum with a ball is infinite off
        the normal and the ball's support plus the offset along it."""
        normal = space.random(rng=rng)
        shift = space.random(rng=rng)
        half = HalfSpace(space, normal, offset=0.4)
        moved = half.translate(shift).support_function()
        assert moved(normal) == pytest.approx(0.4 + space.inner_product(shift, normal))
        assert moved(space.random(rng=rng)) == float("inf")

        fat = (half + Ball(space, radius=2.0)).support_function()
        assert fat(normal) == pytest.approx(0.4 + 2.0 * space.norm(normal))
        assert fat(space.random(rng=rng)) == float("inf")

    def test_two_half_spaces_bound_each_other_along_a_shared_normal(self, space, rng):
        """An intersection of half-spaces alone used to have no bound at
        all; now it has one wherever a part's support is finite."""
        normal = space.random(rng=rng)
        slab = HalfSpace(space, normal, offset=0.4) & HalfSpace(
            space, space.scale(-1.0, normal), offset=0.1
        )
        assert slab.support_bound(normal) == pytest.approx(0.4)
        assert slab.support_bound(space.scale(-1.0, normal)) == pytest.approx(0.1)
        assert slab.support_bound(space.random(rng=rng)) == float("inf")

    def test_a_zero_normal_is_refused_by_the_support_function_too(self, space):
        from pygeoinf2.numerics.convex import SupportFunction

        with pytest.raises(ValueError, match="nonzero"):
            SupportFunction.of_half_space(space, space.zero())


class TestConvexIntersection:
    """An intersection of convex sets is convex, and v2 returned a plain
    Intersection for it -- which knows only how to test membership. That loses
    the projection, and with it every proximal method, the primal-dual route,
    and any use as a prior."""

    @pytest.fixture
    def parts(self, X):
        return Ball(X, radius=1.0), HalfSpace(X, X.basis_vector(0), offset=-0.2)

    def test_it_projects(self, X, parts, rng):
        """Against a constrained optimiser. Dykstra, not alternating
        projection: the latter reaches a point of the intersection, not the
        nearest one."""
        from scipy.optimize import minimize

        ball, half = parts
        combined = ball & half

        for _ in range(3):
            point = X.scale(3.0, X.random(rng=rng))
            projected = combined.project(point)
            assert combined.contains(projected, rtol=1e-6)

            reference = minimize(
                lambda z: X.squared_norm(X.subtract(X.from_components(z), point)),
                X.to_components(projected),
                constraints=[
                    {
                        "type": "ineq",
                        "fun": lambda z: 1.0 - X.norm(X.from_components(z)),
                    },
                    {
                        "type": "ineq",
                        "fun": lambda z: -0.2
                        - X.inner_product(X.basis_vector(0), X.from_components(z)),
                    },
                ],
            )
            assert X.norm(X.subtract(projected, point)) == pytest.approx(
                X.norm(X.subtract(X.from_components(reference.x), point)),
                rel=1e-4,
            )

    def test_the_support_function_is_refused_and_the_bound_is_not(self, X, parts):
        """``min_i h_i`` bounds the support from above and is not equal to it:
        the point of one set attaining its own support need not lie in the
        others. v1 returned it as the support function."""
        ball, half = parts
        combined = ball & half

        with pytest.raises(NotImplementedError, match="support_bound"):
            combined.support_function()
        # The half-space is unbounded in this direction and its support is
        # infinite, so the minimum is the ball's own support, which is its
        # radius times the direction's norm -- not the radius, this space's
        # basis not being orthonormal.
        direction = X.basis_vector(1)
        assert half.support_function()(direction) == float("inf")
        assert combined.support_bound(direction) == pytest.approx(
            ball.support_function()(direction)
        )

    def test_a_non_convex_part_is_refused(self, X):
        from pygeoinf2.geometry.convex import ConvexIntersection
        from pygeoinf2.geometry.sets import Subset

        class Blob(Subset):
            def contains(self, x, /, *, rtol=1e-9):
                return True

        with pytest.raises(TypeError, match="convex"):
            ConvexIntersection([Ball(X, radius=1.0), Blob(X)])

    def test_an_empty_intersection_is_refused(self):
        from pygeoinf2.geometry.convex import ConvexIntersection

        with pytest.raises(ValueError, match="at least one"):
            ConvexIntersection([])

    def test_it_can_be_used_where_a_projection_is_needed(self, X, parts, rng):
        """The point of the change: a proximal method takes the indicator's
        prox, which is the projection."""
        ball, half = parts
        indicator = (ball & half).indicator()
        point = X.scale(2.0, X.random(rng=rng))
        assert (ball & half).contains(indicator.prox(point, 1.0), rtol=1e-6)
