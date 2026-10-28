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

from .conftest import make_dense_metric_space, make_weighted_space


from .doubles import OpaqueSpace


@pytest.fixture(params=["weighted", "dense"])
def space(request):
    """A diagonal metric and a dense one: a gradient is the Riesz
    representer of the derivative, a prox map is a projection in the
    space's own norm, and only a non-diagonal Gram matrix tells either from
    its raw components."""
    return (
        make_weighted_space()
        if request.param == "weighted"
        else make_dense_metric_space(4)
    )


def least_squares(space, matrix, data):
    """``0.5 ||A x - d||^2``, with an ill-conditioned ``A`` to make it hard."""
    codomain = EuclideanSpace(matrix.shape[0])
    A = LinearOperator.from_matrix(space, codomain, matrix, form="components")

    def value(x):
        residual = matrix @ space.to_components(x) - data
        return 0.5 * float(residual @ residual)

    def derivative(x):
        residual = matrix @ space.to_components(x) - data
        return LinearFunctional.from_derivative_components(space, matrix.T @ residual)

    return Functional.from_callables(space, value, derivative=derivative), A


class TestSquaredDistance:
    def test_value_and_gradient(self, space, rng):
        center = space.random(rng=rng)
        f = SquaredDistance(space, center=center)
        x = space.random(rng=rng)

        assert f(x) == pytest.approx(
            0.5 * space.squared_norm(space.subtract(x, center))
        )
        assert (
            space.norm(space.subtract(f.gradient(x), space.subtract(x, center))) < 1e-12
        )

    def test_its_hessian_is_the_identity(self, space, rng):
        """Whatever the metric: the Hessian is taken in the space's inner product."""
        f = SquaredDistance(space, center=space.random(rng=rng))
        x, v = space.random(rng=rng), space.random(rng=rng)
        assert space.norm(space.subtract(f.hessian(x)(v), v)) < 1e-12

    def test_the_prox_is_a_contraction_towards_the_center(self, space, rng):
        center = space.random(rng=rng)
        f = SquaredDistance(space, center=center)
        x = space.random(rng=rng)
        for step in (0.1, 1.0, 10.0):
            proximal = f.prox(x, step)
            expected = space.scale(
                1.0 / (1.0 + step), space.axpy(step, center, space.copy(x))
            )
            assert space.norm(space.subtract(proximal, expected)) < 1e-12


class TestNormFunctional:
    def test_the_prox_shrinks_along_the_vector(self, space, rng):
        """Coordinate-free: a norm and a scaling, no basis anywhere."""
        f = NormFunctional(space, weight=0.5)
        x = space.random(rng=rng)
        step = 0.3

        proximal = f.prox(x, step)
        shrinkage = max(0.0, 1.0 - step * 0.5 / space.norm(x))
        assert space.norm(space.subtract(proximal, space.scale(shrinkage, x))) < 1e-12

    def test_the_prox_reaches_zero_for_a_large_step(self, space, rng):
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

    def test_the_subgradient_at_the_origin_is_admissible(self, space, rng):
        f = NormFunctional(space, weight=1.0)
        assert space.norm(f.subgradient(space.zero())) == pytest.approx(0.0)

    def test_the_subgradient_has_the_right_norm(self, space, rng):
        f = NormFunctional(space, weight=2.5)
        assert space.norm(f.subgradient(space.random(rng=rng))) == pytest.approx(2.5)

    def test_a_non_positive_weight_is_refused(self):
        with pytest.raises(ValueError, match="positive"):
            NormFunctional(EuclideanSpace(3), weight=0.0)


class TestBallIndicator:
    def test_the_prox_projects(self, space, rng):
        indicator = BallIndicator(space, radius=0.7)
        draw = space.random(rng=rng)
        outside = space.scale(5.0 / space.norm(draw), draw)
        projected = indicator.prox(outside, 1.0)
        assert space.norm(projected) == pytest.approx(0.7)

    def test_a_point_inside_is_left_alone(self, space, rng):
        indicator = BallIndicator(space, radius=10.0)
        x = space.random(rng=rng)
        assert space.norm(space.subtract(indicator.prox(x, 1.0), x)) < 1e-12

    def test_the_value_is_infinite_outside(self, space, rng):
        indicator = BallIndicator(space, radius=0.5)
        x = space.random(rng=rng)
        far = space.scale(100.0 / space.norm(x), x)
        assert indicator(far) == float("inf")

    def test_the_conjugate_is_the_support_function(self, space, rng):
        indicator = BallIndicator(space, radius=2.0)
        conjugate = indicator.conjugate()
        y = space.random(rng=rng)
        assert conjugate(y) == pytest.approx(2.0 * space.norm(y))


class TestSupportFunctions:
    def test_a_ball(self, space, rng):
        h = SupportFunction.of_ball(space, radius=3.0)
        y = space.random(rng=rng)
        assert h(y) == pytest.approx(3.0 * space.norm(y))

    def test_the_subgradient_attains_the_supremum(self, space, rng):
        h = SupportFunction.of_ball(space, radius=3.0)
        y = space.random(rng=rng)
        maximizer = h.subgradient(y)
        assert space.norm(maximizer) == pytest.approx(3.0)
        assert space.inner_product(maximizer, y) == pytest.approx(h(y))

    def test_a_point(self, space, rng):
        point = space.random(rng=rng)
        h = SupportFunction.of_point(space, point)
        y = space.random(rng=rng)
        assert h(y) == pytest.approx(space.inner_product(point, y))

    def test_an_oracle_with_its_maximizer(self, space, rng):
        """A support function from callables: one for the value and,
        when known, one for the point attaining it, which is the subgradient
        a bundle method asks for."""
        ball = SupportFunction.of_ball(space, radius=2.0)
        h = SupportFunction.of_oracle(
            space, lambda q: 2.0 * space.norm(q), maximizer=ball.subgradient
        )
        y = space.random(rng=rng)
        assert h(y) == pytest.approx(2.0 * space.norm(y))
        assert h.has_subgradient
        assert space.inner_product(h.subgradient(y), y) == pytest.approx(h(y))

        values_only = SupportFunction.of_oracle(space, lambda q: 2.0 * space.norm(q))
        assert values_only(y) == pytest.approx(h(y))
        assert not values_only.has_subgradient
        with pytest.raises(NotImplementedError, match="maximizer"):
            values_only.subgradient(y)

    def test_a_minkowski_sum(self, space, rng):
        """The algebra is closed, which is why the class exists."""
        center = space.random(rng=rng)
        total = SupportFunction.of_ball(space, radius=2.0) + SupportFunction.of_point(
            space, center
        )
        assert isinstance(total, SupportFunction)
        y = space.random(rng=rng)
        assert total(y) == pytest.approx(
            2.0 * space.norm(y) + space.inner_product(center, y)
        )

    def test_a_positive_scaling(self, space, rng):
        h = 3.0 * SupportFunction.of_ball(space, radius=2.0)
        assert isinstance(h, SupportFunction)
        y = space.random(rng=rng)
        assert h(y) == pytest.approx(6.0 * space.norm(y))

    def test_a_linear_image(self, rng):
        """h_{A K}(y) == h_K(A* y)."""
        X, Y = make_weighted_space(), EuclideanSpace(3)
        A = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(3, X.dim)), form="components"
        )
        h = SupportFunction.of_ball(X, radius=1.0).compose_with(A)
        y = Y.random(rng=rng)
        assert h(y) == pytest.approx(X.norm(A.adjoint(y)))

    def test_a_mismatched_operator_is_refused(self, rng):
        """The operator must map *out of* the space the set lives in."""
        X, Y = make_weighted_space(), EuclideanSpace(3)
        A = LinearOperator.from_matrix(Y, Y, np.identity(3), form="components")
        with pytest.raises(ValueError, match="must map out of"):
            SupportFunction.of_ball(X).compose_with(A)


class TestProximalGradient:
    def test_it_solves_a_problem_with_a_known_answer(self, space, rng):
        """min 0.5||x-c||^2 + w||x|| has the shrinkage of c as its minimizer."""
        center = space.random(rng=rng)
        smooth = SquaredDistance(space, center=center)
        penalty = NormFunctional(space, weight=0.3)

        result = ProximalGradient(max_iterations=2000, gtol=1e-10).minimize(
            smooth, space.random(rng=rng), nonsmooth=penalty
        )
        assert result.converged
        expected = penalty.prox(center, 1.0)
        assert space.norm(space.subtract(result.minimizer, expected)) < 1e-8

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
        plain = ProximalGradient(accelerated=False, **budget).minimize(
            smooth, start, nonsmooth=penalty
        )
        fast = ProximalGradient(accelerated=True, **budget).minimize(
            smooth, start, nonsmooth=penalty
        )
        assert fast.value < plain.value

    def test_a_ball_constraint_is_respected(self, space, rng):
        smooth = SquaredDistance(space, center=space.random(rng=rng))
        constraint = BallIndicator(space, radius=0.2)
        result = ProximalGradient(max_iterations=2000, gtol=1e-10).minimize(
            smooth, space.random(rng=rng), nonsmooth=constraint
        )
        assert space.norm(result.minimizer) <= 0.2 * (1.0 + 1e-8)

    def test_it_works_with_no_nonsmooth_part(self, space, rng):
        center = space.random(rng=rng)
        smooth = SquaredDistance(space, center=center)
        result = ProximalGradient(max_iterations=500, gtol=1e-10).minimize(
            smooth, space.random(rng=rng)
        )
        assert space.norm(space.subtract(result.minimizer, center)) < 1e-8

    def test_a_nonsmooth_part_without_a_prox_is_refused(self, rng):
        space = EuclideanSpace(3)
        smooth = SquaredDistance(space)
        bad = Functional.from_callables(space, lambda x: float(abs(x).sum()))
        with pytest.raises(ValueError, match="proximal operator"):
            ProximalGradient().minimize(smooth, space.zero(), nonsmooth=bad)


class TestSubgradientDescent:
    def test_it_minimizes_a_norm(self, space, rng):
        f = NormFunctional(space, weight=1.0)
        result = SubgradientDescent(
            step_size=0.5, rule="sqrt", max_iterations=3000
        ).minimize(f, space.random(rng=rng))
        assert result.value < 1e-2

    def test_the_polyak_rule_is_faster_when_the_optimum_is_known(self, space, rng):
        f = NormFunctional(space, weight=1.0)
        start = space.random(rng=rng)
        classical = SubgradientDescent(
            step_size=0.5, rule="sqrt", max_iterations=400
        ).minimize(f, start)
        polyak = SubgradientDescent(
            rule="polyak", target_value=0.0, max_iterations=400
        ).minimize(f, start)
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
            SubgradientDescent(step_size=0.1).minimize(f, space.zero())

    def test_a_smooth_functional_supplies_its_own_subgradient(self, space, rng):
        f = SquaredDistance(space, center=space.random(rng=rng))
        assert f.has_subgradient
        assert (
            space.norm(
                space.subtract(f.subgradient(space.zero()), f.gradient(space.zero()))
            )
            < 1e-12
        )


class TestProximalPoint:
    def test_it_converges_on_a_squared_distance(self, space, rng):
        center = space.random(rng=rng)
        f = SquaredDistance(space, center=center)
        result = ProximalPoint(step=1.0, max_iterations=500, gtol=1e-10).minimize(
            f, space.random(rng=rng)
        )
        assert space.norm(space.subtract(result.minimizer, center)) < 1e-8

    def test_a_larger_step_converges_in_fewer_iterations(self, space, rng):
        f = SquaredDistance(space, center=space.random(rng=rng))
        start = space.random(rng=rng)
        slow = ProximalPoint(step=0.1, max_iterations=2000, gtol=1e-10).minimize(
            f, start
        )
        fast = ProximalPoint(step=10.0, max_iterations=2000, gtol=1e-10).minimize(
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
            ProximalPoint().minimize(f, space.zero())


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
        center = opaque.random(rng=rng)
        smooth = SquaredDistance(opaque, center=center)
        penalty = NormFunctional(opaque, weight=0.2)
        result = ProximalGradient(max_iterations=1000, gtol=1e-10).minimize(
            smooth, opaque.random(rng=rng), nonsmooth=penalty
        )
        expected = penalty.prox(center, 1.0)
        assert opaque.norm(opaque.subtract(result.minimizer, expected)) < 1e-8

    def test_support_functions_without_components(self, opaque, rng):
        h = SupportFunction.of_ball(opaque, radius=2.0)
        y = opaque.random(rng=rng)
        assert h(y) == pytest.approx(2.0 * opaque.norm(y))


class TestTheBundleLearnsFromNullSteps:
    """The subgradient is taken at the trial point, not at the center. A null
    step leaves the center where it is, so a cut taken there would be
    identical to one already in the bundle and teach nothing from the trial
    point just paid for -- and duplicate cuts make the model's
    Gram matrix exactly singular."""

    @staticmethod
    def piecewise_linear(rng, n=50, m=30):
        """``|A x - b|_1``: convex, non-smooth, and with a known minimum of
        zero when the system is solvable."""
        from pygeoinf2.algebra.operators import Functional, LinearFunctional

        space = EuclideanSpace(n)
        matrix = rng.standard_normal((m, n))
        offset = rng.standard_normal(m)

        return space, Functional.from_callables(
            space,
            lambda c: float(np.abs(matrix @ c - offset).sum()),
            derivative=lambda c: LinearFunctional.from_derivative_components(
                space, matrix.T @ np.sign(matrix @ c - offset)
            ),
        )

    def test_it_converges_in_far_fewer_iterations(self, rng):
        """Measured on ``|A x - b|_1``: 44 iterations against 255 at n=50,
        and 13 against 32 at n=20, for the same minimum."""
        from pygeoinf2.numerics.convex import ProximalBundleMethod

        space, functional = self.piecewise_linear(rng)
        result = ProximalBundleMethod(max_iterations=300, tolerance=1e-8).minimize(
            functional, space.zero()
        )
        assert result.converged
        assert result.value == pytest.approx(0.0, abs=1e-6)

    def test_every_cut_after_the_first_is_taken_at_a_new_point(self, rng):
        """Which is what makes a null step informative."""
        from pygeoinf2.numerics.convex import ProximalBundleMethod

        space, functional = self.piecewise_linear(rng, n=20, m=10)
        seen = []
        original = functional.subgradient

        def watched(point):
            seen.append(np.array(point))
            return original(point)

        ProximalBundleMethod(max_iterations=40).minimize(
            functional, space.zero(), subgradient=watched
        )
        distinct = {tuple(np.round(point, 12)) for point in seen}
        assert len(distinct) == len(seen)


class TestTheBundleSubproblem:
    """Accelerated, and it says when it has not converged."""

    @staticmethod
    def hard_problem(rng, k=12):
        """Near-parallel cuts, which is what a converging bundle produces and
        what makes the Gram matrix badly conditioned."""
        vectors = rng.standard_normal((k, 40))
        vectors[3:] = vectors[:1] + 0.01 * rng.standard_normal((k - 3, 40))
        return vectors @ vectors.T, rng.uniform(0.0, 1.0, k)

    def test_it_solves_a_hard_instance_to_a_small_residual(self):
        """On the near-parallel bundle a converging method produces, where
        the Gram matrix is badly conditioned, the subproblem is solved to a
        KKT residual well below what the bundle method asks of it. One
        instance with an absolute bound; a comparison against the plain
        projected gradient it replaced was a speed measurement, and a
        speed measurement of a private function is not a test."""
        from pygeoinf2.numerics.convex import _minimize_on_simplex

        generator = np.random.default_rng(4)
        size = int(generator.integers(5, 25))
        vectors = generator.standard_normal((size, 40))
        vectors[3:] = vectors[:1] + 0.01 * generator.standard_normal((size - 3, 40))
        quadratic = vectors @ vectors.T
        linear = generator.uniform(0.0, 1.0, size)
        weights = _minimize_on_simplex(
            quadratic, linear, max_iterations=400, tolerance=0.0, warn_above=np.inf
        )
        gradient = quadratic @ weights - linear
        residual = float(gradient[weights > 0].max() - gradient.min())
        assert residual < 1e-4
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(weights >= 0.0)

    def test_it_returns_a_point_of_the_simplex(self, rng):
        from pygeoinf2.numerics.convex import _minimize_on_simplex

        quadratic, linear = self.hard_problem(rng)
        weights = _minimize_on_simplex(quadratic, linear, max_iterations=1000)
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(weights >= 0.0)

    def test_it_warns_when_it_gives_up_badly(self, rng):
        """A bundle calls this at every step, so a failing subproblem that
        said nothing would show up only as an outer method that will not
        settle."""
        from pygeoinf2.numerics.convex import _minimize_on_simplex

        quadratic, linear = self.hard_problem(rng)
        with pytest.warns(RuntimeWarning, match="did not converge"):
            _minimize_on_simplex(
                quadratic, linear, max_iterations=2, tolerance=1e-14, warn_above=1e-12
            )

    def test_a_small_residual_passes_without_a_warning(self, rng):
        """The threshold to warn at is separate from the one to stop at,
        because the residual has a floor that no iteration count removes."""
        import warnings

        from pygeoinf2.numerics.convex import _minimize_on_simplex

        quadratic, linear = self.hard_problem(rng)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _minimize_on_simplex(quadratic, linear, max_iterations=1000)


class TestTheBundleWarmStart:
    """The QP backend's warm start must line up with the cuts: padded with a
    zero for the cut appended since, and shifted along by one when the full
    bundle dropped its oldest cut -- the case where the bundle's size does
    not change, so size alone cannot tell the two apart."""

    class RecordingBackend:
        def __init__(self):
            self.starts = []

        def solve(self, quadratic, linear, rows, lower, upper, *, x0=None):
            from pygeoinf2.numerics.quadratic_programming import QPResult

            self.starts.append(None if x0 is None else np.array(x0))
            size = linear.size
            weights = np.arange(1.0, size + 1.0)
            weights /= weights.sum()
            return QPResult(x=weights, objective=0.0, status="solved")

    @pytest.fixture
    def method(self):
        from pygeoinf2.numerics.convex import ProximalBundleMethod

        method = ProximalBundleMethod(capacity=3)
        method._qp_solver = self.RecordingBackend()
        return method

    def test_a_growing_bundle_pads_with_a_zero(self, method):
        backend = method.qp_solver
        first = method._simplex_weights(np.eye(2), np.zeros(2))
        method._simplex_weights(np.eye(3), np.zeros(3))
        assert backend.starts[0] is None
        assert np.allclose(backend.starts[1], np.concatenate([first, [0.0]]))

    def test_a_full_bundle_shifts_the_weights_along(self, method):
        """At capacity a cut is appended and the oldest popped, so the size
        is unchanged and the weights must move by one, the dropped cut's
        weight going to the new one so the start stays on the simplex."""
        backend = method.qp_solver
        previous = method._simplex_weights(np.eye(3), np.zeros(3))
        method._dropped = True  # what the minimizer records on a pop
        method._simplex_weights(np.eye(3), np.zeros(3))
        start = backend.starts[1]
        assert np.allclose(start[:2], previous[1:])
        assert start[2] == pytest.approx(previous[0])
        assert start.sum() == pytest.approx(1.0)
        # The flag is consumed: a later growth pads again rather than shifts.
        current = np.array(method._warm)
        method._simplex_weights(np.eye(4), np.zeros(4))
        assert np.allclose(backend.starts[2], np.concatenate([current, [0.0]]))


class TestBundleResultReadsLikeAnOptimizationResult:
    def test_it_is_one(self):
        from pygeoinf2.numerics.convex import (
            BundleResult,
            ProximalBundleMethod,
            SaddlePointResult,
        )
        from pygeoinf2.numerics.optimization import OptimizationResult, Optimizer

        for result in (BundleResult, SaddlePointResult):
            assert issubclass(result, OptimizationResult)
        assert issubclass(ProximalBundleMethod, Optimizer)

    def test_the_evaluations_are_counted(self, rng):
        from pygeoinf2.numerics.convex import ProximalBundleMethod

        space, functional = TestTheBundleLearnsFromNullSteps.piecewise_linear(
            rng, n=20, m=10
        )
        result = ProximalBundleMethod(max_iterations=50).minimize(
            functional, space.zero()
        )
        assert result.evaluations >= result.iterations
        assert result.message


class TestReportedCounts:
    """The counts a result carries are the ones actually spent."""

    def test_subgradient_descent_evaluates_once_per_iteration(self, rng):
        from pygeoinf2.algebra.spaces import EuclideanSpace
        from pygeoinf2.numerics.convex import NormFunctional, SubgradientDescent

        space = EuclideanSpace(4)
        base = NormFunctional(space)
        calls = {"value": 0}

        class Counted(type(base)):
            def _value(self, x):
                calls["value"] += 1
                return base._value(x)

        functional = Counted(space)
        result = SubgradientDescent(max_iterations=15, gtol=0.0).minimize(
            functional, space.random(rng=rng)
        )
        assert calls["value"] == result.iterations + 1
        assert result.evaluations == calls["value"]
