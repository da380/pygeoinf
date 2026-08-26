"""Convex functionals and non-smooth methods."""

import numpy as np
import pytest

from pygeoinf2.algebra.operators import (
    Functional,
    LinearFunctional,
    LinearOperator,
)
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.numerics.convex import (
    BallIndicator,
    NormFunctional,
    ProximalGradient,
    ProximalPoint,
    SquaredDistance,
    SubgradientDescent,
    SupportFunction,
)

from .conftest import make_weighted_space
from .doubles import OpaqueSpace


def least_squares(space, matrix, data):
    """``0.5 ||A x - d||^2``, with an ill-conditioned ``A`` to make it hard."""
    codomain = EuclideanSpace(matrix.shape[0])
    A = LinearOperator.from_component_matrix(space, codomain, matrix)

    def value(x):
        residual = matrix @ space.to_components(x) - data
        return 0.5 * float(residual @ residual)

    def derivative(x):
        residual = matrix @ space.to_components(x) - data
        return LinearFunctional.from_derivative_components(space, matrix.T @ residual)

    return Functional.from_callables(space, value, derivative=derivative), A


class TestSquaredDistance:
    def test_value_and_gradient(self, rng):
        space = make_weighted_space()
        centre = space.random(rng=rng)
        f = SquaredDistance(space, centre=centre)
        x = space.random(rng=rng)

        assert f(x) == pytest.approx(
            0.5 * space.squared_norm(space.subtract(x, centre))
        )
        assert (
            space.norm(space.subtract(f.gradient(x), space.subtract(x, centre))) < 1e-12
        )

    def test_its_hessian_is_the_identity(self, rng):
        """Whatever the metric, which is the whole point of DESIGN.md 5.6."""
        space = make_weighted_space()
        f = SquaredDistance(space, centre=space.random(rng=rng))
        x, v = space.random(rng=rng), space.random(rng=rng)
        assert space.norm(space.subtract(f.hessian(x)(v), v)) < 1e-12

    def test_the_prox_is_a_contraction_towards_the_centre(self, rng):
        space = make_weighted_space()
        centre = space.random(rng=rng)
        f = SquaredDistance(space, centre=centre)
        x = space.random(rng=rng)
        for step in (0.1, 1.0, 10.0):
            proximal = f.prox(x, step)
            expected = space.scale(
                1.0 / (1.0 + step), space.axpy(step, centre, space.copy(x))
            )
            assert space.norm(space.subtract(proximal, expected)) < 1e-12


class TestNormFunctional:
    def test_the_prox_shrinks_along_the_vector(self, rng):
        """Coordinate-free: a norm and a scaling, no basis anywhere."""
        space = make_weighted_space()
        f = NormFunctional(space, weight=0.5)
        x = space.random(rng=rng)
        step = 0.3

        proximal = f.prox(x, step)
        shrinkage = max(0.0, 1.0 - step * 0.5 / space.norm(x))
        assert space.norm(space.subtract(proximal, space.scale(shrinkage, x))) < 1e-12

    def test_the_prox_reaches_zero_for_a_large_step(self, rng):
        space = make_weighted_space()
        f = NormFunctional(space, weight=1.0)
        x = space.random(rng=rng)
        assert space.norm(f.prox(x, 10.0 * space.norm(x))) == pytest.approx(0.0)

    def test_the_prox_solves_its_own_definition(self, rng):
        """Check the closed form against the variational problem it solves."""
        space = EuclideanSpace(4)
        f = NormFunctional(space, weight=0.7)
        x = space.random(rng=rng)
        step = 0.4
        proximal = f.prox(x, step)

        def objective(y):
            return f(y) + space.squared_norm(space.subtract(y, x)) / (2.0 * step)

        best = objective(proximal)
        for _ in range(200):
            perturbed = space.axpy(1e-3, space.random(rng=rng), space.copy(proximal))
            assert objective(perturbed) >= best - 1e-9

    def test_the_subgradient_at_the_origin_is_admissible(self, rng):
        space = make_weighted_space()
        f = NormFunctional(space, weight=1.0)
        assert space.norm(f.subgradient(space.zero())) == pytest.approx(0.0)

    def test_the_subgradient_has_the_right_norm(self, rng):
        space = make_weighted_space()
        f = NormFunctional(space, weight=2.5)
        assert space.norm(f.subgradient(space.random(rng=rng))) == pytest.approx(2.5)

    def test_a_non_positive_weight_is_refused(self):
        with pytest.raises(ValueError, match="positive"):
            NormFunctional(EuclideanSpace(3), weight=0.0)


class TestBallIndicator:
    def test_the_prox_projects(self, rng):
        space = make_weighted_space()
        indicator = BallIndicator(space, radius=0.7)
        outside = space.scale(
            5.0 / space.norm(space.random(rng=rng)), space.random(rng=rng)
        )
        projected = indicator.prox(outside, 1.0)
        assert space.norm(projected) == pytest.approx(0.7)

    def test_a_point_inside_is_left_alone(self, rng):
        space = make_weighted_space()
        indicator = BallIndicator(space, radius=10.0)
        x = space.random(rng=rng)
        assert space.norm(space.subtract(indicator.prox(x, 1.0), x)) < 1e-12

    def test_the_value_is_infinite_outside(self, rng):
        space = make_weighted_space()
        indicator = BallIndicator(space, radius=0.5)
        x = space.random(rng=rng)
        far = space.scale(100.0 / space.norm(x), x)
        assert indicator(far) == float("inf")

    def test_the_conjugate_is_the_support_function(self, rng):
        space = make_weighted_space()
        indicator = BallIndicator(space, radius=2.0)
        conjugate = indicator.conjugate()
        y = space.random(rng=rng)
        assert conjugate(y) == pytest.approx(2.0 * space.norm(y))


class TestSupportFunctions:
    def test_a_ball(self, rng):
        space = make_weighted_space()
        h = SupportFunction.of_ball(space, radius=3.0)
        y = space.random(rng=rng)
        assert h(y) == pytest.approx(3.0 * space.norm(y))

    def test_the_subgradient_attains_the_supremum(self, rng):
        space = make_weighted_space()
        h = SupportFunction.of_ball(space, radius=3.0)
        y = space.random(rng=rng)
        maximiser = h.subgradient(y)
        assert space.norm(maximiser) == pytest.approx(3.0)
        assert space.inner_product(maximiser, y) == pytest.approx(h(y))

    def test_a_point(self, rng):
        space = make_weighted_space()
        point = space.random(rng=rng)
        h = SupportFunction.of_point(space, point)
        y = space.random(rng=rng)
        assert h(y) == pytest.approx(space.inner_product(point, y))

    def test_a_minkowski_sum(self, rng):
        """The algebra is closed, which is why the class exists."""
        space = make_weighted_space()
        centre = space.random(rng=rng)
        total = SupportFunction.of_ball(space, radius=2.0) + SupportFunction.of_point(
            space, centre
        )
        assert isinstance(total, SupportFunction)
        y = space.random(rng=rng)
        assert total(y) == pytest.approx(
            2.0 * space.norm(y) + space.inner_product(centre, y)
        )

    def test_a_positive_scaling(self, rng):
        space = make_weighted_space()
        h = 3.0 * SupportFunction.of_ball(space, radius=2.0)
        assert isinstance(h, SupportFunction)
        y = space.random(rng=rng)
        assert h(y) == pytest.approx(6.0 * space.norm(y))

    def test_a_linear_image(self, rng):
        """h_{A K}(y) == h_K(A* y)."""
        X, Y = make_weighted_space(), EuclideanSpace(3)
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, X.dim)))
        h = SupportFunction.of_ball(X, radius=1.0).compose_with(A)
        y = Y.random(rng=rng)
        assert h(y) == pytest.approx(X.norm(A.adjoint(y)))

    def test_a_mismatched_operator_is_refused(self, rng):
        """The operator must map *out of* the space the set lives in."""
        X, Y = make_weighted_space(), EuclideanSpace(3)
        A = LinearOperator.from_component_matrix(Y, Y, np.identity(3))
        with pytest.raises(ValueError, match="must map out of"):
            SupportFunction.of_ball(X).compose_with(A)


class TestProximalGradient:
    def test_it_solves_a_problem_with_a_known_answer(self, rng):
        """min 0.5||x-c||^2 + w||x|| has the shrinkage of c as its minimiser."""
        space = make_weighted_space()
        centre = space.random(rng=rng)
        smooth = SquaredDistance(space, centre=centre)
        penalty = NormFunctional(space, weight=0.3)

        result = ProximalGradient(max_iterations=2000, gtol=1e-14).minimise(
            smooth, space.random(rng=rng), nonsmooth=penalty
        )
        assert result.converged
        expected = penalty.prox(centre, 1.0)
        assert space.norm(space.subtract(result.minimiser, expected)) < 1e-8

    def test_acceleration_helps_an_ill_conditioned_problem(self, rng):
        """Compared on a fixed budget, which is the meaningful comparison.

        Neither method converges on this problem within any reasonable number
        of iterations -- that is what ill-conditioned means -- so comparing
        iterations-to-converge would only compare two iteration caps. FISTA's
        ``O(1/k^2)`` against ISTA's ``O(1/k)`` shows up as a lower objective
        after the same number of steps.
        """
        space = EuclideanSpace(20)
        matrix = (
            np.diag(np.logspace(0, -2.5, 20))
            @ np.linalg.qr(rng.normal(size=(20, 20)))[0]
        )
        data = rng.normal(size=20)
        smooth, _ = least_squares(space, matrix, data)
        penalty = NormFunctional(space, weight=1e-3)
        start = space.random(rng=rng)

        budget = dict(max_iterations=300, gtol=0.0)
        plain = ProximalGradient(accelerated=False, **budget).minimise(
            smooth, start, nonsmooth=penalty
        )
        fast = ProximalGradient(accelerated=True, **budget).minimise(
            smooth, start, nonsmooth=penalty
        )
        assert fast.value < plain.value

    def test_a_ball_constraint_is_respected(self, rng):
        space = make_weighted_space()
        smooth = SquaredDistance(space, centre=space.random(rng=rng))
        constraint = BallIndicator(space, radius=0.2)
        result = ProximalGradient(max_iterations=2000, gtol=1e-14).minimise(
            smooth, space.random(rng=rng), nonsmooth=constraint
        )
        assert space.norm(result.minimiser) <= 0.2 * (1.0 + 1e-8)

    def test_it_works_with_no_nonsmooth_part(self, rng):
        space = make_weighted_space()
        centre = space.random(rng=rng)
        smooth = SquaredDistance(space, centre=centre)
        result = ProximalGradient(max_iterations=500, gtol=1e-14).minimise(
            smooth, space.random(rng=rng)
        )
        assert space.norm(space.subtract(result.minimiser, centre)) < 1e-8

    def test_a_nonsmooth_part_without_a_prox_is_refused(self, rng):
        space = EuclideanSpace(3)
        smooth = SquaredDistance(space)
        bad = Functional.from_callables(space, lambda x: float(abs(x).sum()))
        with pytest.raises(ValueError, match="proximal operator"):
            ProximalGradient().minimise(smooth, space.zero(), nonsmooth=bad)


class TestSubgradientDescent:
    def test_it_minimises_a_norm(self, rng):
        space = make_weighted_space()
        f = NormFunctional(space, weight=1.0)
        result = SubgradientDescent(
            step_size=0.5, rule="sqrt", max_iterations=3000
        ).minimise(f, space.random(rng=rng))
        assert result.value < 1e-2

    def test_the_polyak_rule_is_faster_when_the_optimum_is_known(self, rng):
        space = make_weighted_space()
        f = NormFunctional(space, weight=1.0)
        start = space.random(rng=rng)
        classical = SubgradientDescent(
            step_size=0.5, rule="sqrt", max_iterations=400
        ).minimise(f, start)
        polyak = SubgradientDescent(
            rule="polyak", target_value=0.0, max_iterations=400
        ).minimise(f, start)
        assert polyak.value < classical.value

    def test_polyak_needs_a_target(self):
        with pytest.raises(ValueError, match="target_value"):
            SubgradientDescent(rule="polyak")

    def test_an_unknown_rule_is_refused(self):
        with pytest.raises(ValueError, match="Unknown rule"):
            SubgradientDescent(rule="nonsense")

    def test_a_functional_without_a_subgradient_is_refused(self):
        space = EuclideanSpace(3)
        f = Functional.from_callables(space, lambda x: float(x @ x))
        with pytest.raises(ValueError, match="subgradient"):
            SubgradientDescent(step_size=0.1).minimise(f, space.zero())

    def test_a_smooth_functional_supplies_its_own_subgradient(self, rng):
        space = make_weighted_space()
        f = SquaredDistance(space, centre=space.random(rng=rng))
        assert f.has_subgradient
        assert (
            space.norm(
                space.subtract(f.subgradient(space.zero()), f.gradient(space.zero()))
            )
            < 1e-12
        )


class TestProximalPoint:
    def test_it_converges_on_a_squared_distance(self, rng):
        space = make_weighted_space()
        centre = space.random(rng=rng)
        f = SquaredDistance(space, centre=centre)
        result = ProximalPoint(step=1.0, max_iterations=500, gtol=1e-14).minimise(
            f, space.random(rng=rng)
        )
        assert space.norm(space.subtract(result.minimiser, centre)) < 1e-8

    def test_a_larger_step_converges_in_fewer_iterations(self, rng):
        space = make_weighted_space()
        f = SquaredDistance(space, centre=space.random(rng=rng))
        start = space.random(rng=rng)
        slow = ProximalPoint(step=0.1, max_iterations=2000, gtol=1e-12).minimise(
            f, start
        )
        fast = ProximalPoint(step=10.0, max_iterations=2000, gtol=1e-12).minimise(
            f, start
        )
        assert fast.iterations < slow.iterations

    def test_a_non_positive_step_is_refused(self):
        with pytest.raises(ValueError, match="positive"):
            ProximalPoint(step=0.0)

    def test_a_functional_without_a_prox_is_refused(self):
        space = EuclideanSpace(3)
        f = Functional.from_callables(space, lambda x: float(x @ x))
        with pytest.raises(ValueError, match="prox"):
            ProximalPoint().minimise(f, space.zero())


class TestCoordinateFreedom:
    """The proximal operators are norms and directions, so no basis is needed."""

    @pytest.fixture
    def opaque(self, rng):
        return OpaqueSpace(np.array([1.0, 4.0, 9.0, 0.25]))

    def test_norm_prox_without_components(self, opaque, rng):
        f = NormFunctional(opaque, weight=0.5)
        x = opaque.random(rng=rng)
        proximal = f.prox(x, 0.2)
        shrinkage = max(0.0, 1.0 - 0.2 * 0.5 / opaque.norm(x))
        assert (
            opaque.norm(opaque.subtract(proximal, opaque.scale(shrinkage, x))) < 1e-12
        )

    def test_ball_projection_without_components(self, opaque, rng):
        indicator = BallIndicator(opaque, radius=0.3)
        x = opaque.random(rng=rng)
        far = opaque.scale(50.0 / opaque.norm(x), x)
        assert opaque.norm(indicator.prox(far, 1.0)) == pytest.approx(0.3)

    def test_proximal_gradient_without_components(self, opaque, rng):
        centre = opaque.random(rng=rng)
        smooth = SquaredDistance(opaque, centre=centre)
        penalty = NormFunctional(opaque, weight=0.2)
        result = ProximalGradient(max_iterations=1000, gtol=1e-14).minimise(
            smooth, opaque.random(rng=rng), nonsmooth=penalty
        )
        expected = penalty.prox(centre, 1.0)
        assert opaque.norm(opaque.subtract(result.minimiser, expected)) < 1e-8

    def test_support_functions_without_components(self, opaque, rng):
        h = SupportFunction.of_ball(opaque, radius=2.0)
        y = opaque.random(rng=rng)
        assert h(y) == pytest.approx(2.0 * opaque.norm(y))
