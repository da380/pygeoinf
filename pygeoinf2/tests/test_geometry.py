"""Subsets, convex sets and subspaces."""

import numpy as np
import pytest

from pygeoinf2 import EuclideanSpace, LinearOperator, Traits
from pygeoinf2.geometry import (
    AffineSubspace,
    Ball,
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
from pygeoinf2.symmetric_space import Sobolev
from pygeoinf2.testing import (
    check_operator,
    check_projection,
    check_traits,
)

from .conftest import make_dense_metric_space
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
        balls = [Ball(X, radius=r) for r in (1.0, 2.0, 3.0)]
        combined = (balls[0] & balls[1]) & balls[2]
        assert isinstance(combined, Intersection)
        assert len(combined.subsets) == 3

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

    def test_the_projection_says_it_is_not_available(self, X, ellipsoid, rng):
        """Rather than offering an approximation under the same name."""
        with pytest.raises(NotImplementedError, match="secular equation"):
            ellipsoid.project(X.random(rng=rng))

    def test_an_indefinite_precision_is_refused(self, X, rng):
        bad = LinearOperator.self_adjoint(X, lambda x: x)
        with pytest.raises(ValueError, match="must claim"):
            Ellipsoid(X, bad)

    def test_the_support_function_needs_the_covariance(self, X):
        precision = X.invariant_operator(lambda values: 1.0 + values)
        with pytest.raises(NotImplementedError, match="covariance"):
            Ellipsoid(X, precision).support_function()


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
