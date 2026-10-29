"""Optimization: correctness, coordinate freedom, and metric awareness."""

import numpy as np
import pytest

from pygeoinf2.algebra.operators import (
    Functional,
    LinearFunctional,
    LinearOperator,
    Operator,
)
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.numerics.line_search import (
    ArmijoLineSearch,
    StrongWolfeLineSearch,
)
from pygeoinf2.numerics.optimization import (
    LBFGS,
    NewtonCG,
    NonlinearCG,
    OptimizationResult,
    SteepestDescent,
    TrustRegionNewton,
    gauss_newton_hessian,
    truncated_cg,
)
from pygeoinf2.traits import Traits

from .conftest import make_dense_metric_space, make_weighted_space, spd_matrix
from .doubles import OpaqueSpace


def quadratic(space, matrix, offset):
    """``0.5 (c - offset)^T M (c - offset)``, with an exact derivative and Hessian."""
    symmetric = 0.5 * (matrix + matrix.T)

    def value(x):
        c = space.to_components(x) - offset
        return 0.5 * float(c @ symmetric @ c)

    def derivative(x):
        c = space.to_components(x) - offset
        return LinearFunctional.from_derivative_components(space, symmetric @ c)

    def hessian(x):
        return LinearOperator.self_adjoint(
            space,
            lambda v: space.from_components(
                space.solve_gram(symmetric @ space.to_components(v))
            ),
            traits=Traits.POSITIVE_DEFINITE,
        )

    return Functional.from_callables(
        space, value, derivative=derivative, hessian=hessian
    )


def rosenbrock(space):
    """The classic banana valley, on ``R^2``. Minimum one at ``(1, 1)``."""

    def value(x):
        return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)

    def derivative(x):
        g = np.array(
            [
                -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2),
                200.0 * (x[1] - x[0] ** 2),
            ]
        )
        return LinearFunctional.from_derivative_components(space, g)

    def hessian(x):
        matrix = np.array(
            [
                [2.0 - 400.0 * (x[1] - 3.0 * x[0] ** 2), -400.0 * x[0]],
                [-400.0 * x[0], 200.0],
            ]
        )
        return LinearOperator.from_matrix(
            space, space, matrix, traits=Traits.SELF_ADJOINT, form="components"
        )

    return Functional.from_callables(
        space, value, derivative=derivative, hessian=hessian
    )


FIRST_ORDER = [SteepestDescent, NonlinearCG, LBFGS]
SECOND_ORDER = [NewtonCG, TrustRegionNewton]


@pytest.fixture
def spd_problem(rng):
    space = EuclideanSpace(20)
    matrix = spd_matrix(rng, 20)
    offset = rng.normal(size=20)
    return space, quadratic(space, matrix, offset), offset


class TestQuadratics:
    @pytest.mark.parametrize("method", FIRST_ORDER + SECOND_ORDER)
    def test_the_minimum_is_found(self, method, spd_problem, rng):
        space, phi, offset = spd_problem
        result = method(max_iterations=3000).minimize(phi, space.random(rng=rng))
        assert isinstance(result, OptimizationResult)
        assert result.converged, result.message
        assert np.allclose(result.minimizer, offset, atol=1e-6)

    def test_newton_reports_why_each_inner_solve_stopped(self, spd_problem, rng):
        space, phi, offset = spd_problem
        result = NewtonCG(max_iterations=50).minimize(phi, space.random(rng=rng))
        assert result.converged
        assert len(result.inner_stops) == result.iterations
        assert set(result.inner_stops) <= {
            "converged",
            "negative curvature",
            "boundary",
            "iteration limit",
        }
        assert LBFGS().minimize(phi, space.random(rng=rng)).inner_stops == ()

    @pytest.mark.parametrize("method", FIRST_ORDER + SECOND_ORDER)
    def test_it_works_on_a_weighted_space(self, method, rng):
        """Where the metric is not the identity, and the gradient is not the array."""
        space = make_weighted_space()
        root = rng.normal(size=(space.dim, space.dim))
        matrix = root @ root.T + space.dim * np.identity(space.dim)
        offset = rng.normal(size=space.dim)
        phi = quadratic(space, matrix, offset)

        result = method(max_iterations=3000).minimize(phi, space.random(rng=rng))
        assert result.converged, result.message
        assert np.allclose(space.to_components(result.minimizer), offset, atol=1e-6)

    @pytest.mark.parametrize("method", FIRST_ORDER)
    def test_a_solved_quadratic_is_reported_as_solved(self, method, rng):
        """A first-order method must not report failure here.

        The gradient tolerance is unreachable for a line-search method in
        double precision, so on a well-conditioned quadratic L-BFGS stopped at
        |g| = 1.8e-7 and said "line search failed", and steepest descent spent
        2000 iterations and 5330 value evaluations to say the same. The
        tolerance is not the thing to loosen: a method that can no longer
        decrease the value has converged, and one whose line search fails at
        the rounding floor of the value has too.
        """
        space = EuclideanSpace(60)
        root = rng.normal(size=(60, 60))
        matrix = root @ root.T + 60.0 * np.identity(60)
        offset = rng.normal(size=60)
        phi = quadratic(space, matrix, offset)

        result = method(max_iterations=2000).minimize(phi, space.zero())

        assert result.converged, result.message
        assert result.evaluations < 200
        assert np.allclose(result.minimizer, offset, atol=1e-4)

    def test_a_flat_functional_stops_rather_than_grinding(self):
        """``ftol`` is relative to ``|f|`` with no absolute floor: a floor
        turns the test absolute wherever the minimum is near zero, and there a
        slow method in a valley takes steps below it while still far away —
        which stopped Rosenbrock two decimal places early."""
        space = EuclideanSpace(2)
        constant = Functional.from_callables(
            space, lambda x: 3.0, gradient=lambda x: np.zeros(2)
        )
        result = SteepestDescent(max_iterations=5000).minimize(
            constant, np.array([1.0, 1.0])
        )
        assert result.converged
        assert result.iterations < 5

    def test_newton_beats_steepest_descent(self, spd_problem, rng):
        space, phi, _ = spd_problem
        start = space.random(rng=rng)
        newton = NewtonCG().minimize(phi, start)
        steepest = SteepestDescent(max_iterations=3000).minimize(phi, start)
        assert newton.iterations < steepest.iterations

    def test_the_history_decreases(self, spd_problem, rng):
        space, phi, _ = spd_problem
        result = LBFGS().minimize(phi, space.random(rng=rng))
        assert np.all(np.diff(result.history) <= 1e-12)


class TestRosenbrock:
    """The banana valley: a genuine test of the direction rules.

    Steepest descent is excluded, and deliberately. It is famously unable to
    traverse this valley in a reasonable number of iterations -- the count runs
    to tens of thousands -- so including it would mean either a slow test or a
    tolerance loose enough to prove nothing. Its behavior is pinned separately
    below instead.
    """

    @pytest.mark.parametrize("method", [NonlinearCG, LBFGS] + SECOND_ORDER)
    def test_the_valley_is_traversed(self, method):
        space = EuclideanSpace(2)
        phi = rosenbrock(space)
        start = np.array([-1.2, 1.0])
        result = method(max_iterations=5000, gtol=1e-8).minimize(phi, start)
        assert np.allclose(result.minimizer, [1.0, 1.0], atol=1e-4), result

    def test_steepest_descent_makes_progress_but_does_not_finish(self):
        """Pinning the known behavior rather than pretending otherwise."""
        space = EuclideanSpace(2)
        phi = rosenbrock(space)
        start = np.array([-1.2, 1.0])
        result = SteepestDescent(max_iterations=2000).minimize(phi, start)
        assert result.value < 0.01 * phi(start)
        assert np.all(np.diff(result.history) <= 1e-12)

    def test_a_trust_region_copes_with_indefiniteness(self):
        """From a point where the Hessian really is indefinite.

        Not the usual start: at ``(-1.2, 1)`` the Rosenbrock Hessian is in fact
        positive definite, with eigenvalues near 24 and 1506. At ``(0, 1)`` it
        is genuinely indefinite, which is what this needs to exercise.
        """
        space = EuclideanSpace(2)
        phi = rosenbrock(space)
        start = np.array([0.0, 1.0])
        assert np.linalg.eigvalsh(phi.hessian(start).matrix()).min() < 0.0

        result = TrustRegionNewton(max_iterations=2000).minimize(phi, start)
        assert np.allclose(result.minimizer, [1.0, 1.0], atol=1e-4), result


class TestMetricAwareness:
    """The decisive difference from a component-space optimizer.

    Take ``phi(x) == 0.5 ||x - a||^2`` in the *space's* norm. Its gradient is
    ``x - a``, so the Hessian is the identity on the space and steepest descent
    with an exact step converges immediately, whatever the metric.

    In components the same function is ``0.5 (c - a)^T G (c - a)``, whose
    Hessian is the Gram matrix. A component-space method therefore sees a
    condition number equal to the spread of the metric values, and takes more
    iterations the worse the discretization is scaled. That is why the
    optimizers are written in the space rather than wrapped around SciPy.
    """

    @staticmethod
    def _space_metric_functional(space, center):
        def value(x):
            return 0.5 * space.squared_norm(space.subtract(x, center))

        def gradient(x):
            return space.subtract(x, center)

        return Functional.from_callables(space, value, gradient=gradient)

    @staticmethod
    def _component_metric_functional(space, center):
        """The same function, but with the gradient taken in the components.

        This is what a component-space optimizer effectively minimizes, and it
        is a genuinely different problem whenever the metric is not the
        identity.
        """
        offset = space.to_components(center)

        def value(x):
            c = space.to_components(x) - offset
            return 0.5 * float(c @ c)

        def gradient(x):
            return space.from_components(space.to_components(x) - offset)

        return Functional.from_callables(space, value, gradient=gradient)

    def test_the_metric_aware_problem_is_perfectly_conditioned(self, rng):
        space = make_weighted_space()
        center = space.random(rng=rng)
        phi = self._space_metric_functional(space, center)

        result = SteepestDescent(max_iterations=500).minimize(
            phi, space.random(rng=rng)
        )
        assert result.converged
        assert space.norm(space.subtract(result.minimizer, center)) < 1e-6
        # A Hessian of identity means very few iterations, whatever the metric.
        assert result.iterations <= 40

    def test_the_two_functionals_really_differ(self, rng):
        """Otherwise the previous test would be proving nothing."""
        space = make_weighted_space()
        center = space.random(rng=rng)
        x = space.random(rng=rng)
        space_metric = self._space_metric_functional(space, center)
        component = self._component_metric_functional(space, center)
        assert not np.isclose(space_metric(x), component(x))

    def test_convergence_does_not_degrade_with_the_metric(self, rng):
        """Refining a discretization should not change the iteration count."""
        from .conftest import WeightedSpace

        counts = []
        for spread in (1.0, 100.0, 10000.0):
            space = WeightedSpace(np.array([1.0, spread, spread**0.5, 1.0]))
            center = space.random(rng=rng)
            phi = self._space_metric_functional(space, center)
            result = SteepestDescent(max_iterations=2000).minimize(
                phi, space.random(rng=rng)
            )
            assert result.converged
            counts.append(result.iterations)
        assert max(counts) - min(counts) <= 5, counts


class TestLineSearches:
    @pytest.fixture
    def setup(self, spd_problem, rng):
        space, phi, _ = spd_problem
        x = space.random(rng=rng)
        model = phi.at(x)
        direction = space.negative(model.gradient)
        slope = space.inner_product(model.gradient, direction)
        return space, phi, x, model, direction, slope

    def test_armijo_gives_sufficient_decrease(self, setup):
        space, phi, x, model, direction, slope = setup
        search = ArmijoLineSearch()
        result = search(phi, x, direction, value=model.value, slope=slope)
        assert result.converged
        assert result.value <= model.value + 1e-4 * result.step * slope

    def test_strong_wolfe_satisfies_both_conditions(self, setup):
        space, phi, x, model, direction, slope = setup
        search = StrongWolfeLineSearch(decrease=1e-4, curvature=0.9)
        result = search(phi, x, direction, value=model.value, slope=slope)
        assert result.converged

        assert result.value <= model.value + 1e-4 * result.step * slope
        new_slope = space.inner_product(phi.at(result.point).gradient, direction)
        assert abs(new_slope) <= 0.9 * abs(slope)

    def test_the_zoom_interpolates(self, setup):
        """On a quadratic the interpolant is exact: a step that overshoots the
        minimizer along the line is followed by one evaluation at the
        minimizer itself, where the slope is zero and both conditions hold.
        Bisection would need several halvings to get there."""
        space, phi, x, model, direction, slope = setup
        # Put the line minimizer at 0.3 of the unit step, so the unit step
        # overshoots and the search must zoom. The curvature along the line is
        # exact by second differences on a quadratic.
        ahead = phi(space.axpy(1.0, direction, space.copy(x)))
        behind = phi(space.axpy(-1.0, direction, space.copy(x)))
        curvature = ahead - 2.0 * model.value + behind
        minimizer = -slope / curvature
        scaled = space.scale(minimizer / 0.3, direction)
        scaled_slope = space.inner_product(model.gradient, scaled)

        result = StrongWolfeLineSearch()(
            phi, x, scaled, value=model.value, slope=scaled_slope
        )
        assert result.converged
        assert result.step == pytest.approx(0.3, rel=1e-6)
        # Two: the overshoot and the minimizer. A third would be a wasted
        # re-evaluation of the bracket's low end on entering the zoom.
        assert result.evaluations == 2

    def test_every_evaluation_is_counted_and_none_is_wasted(self, setup):
        """The count the search reports is the number of times it called the
        functional, and entering the zoom does not re-evaluate the end of
        the bracket the caller already knew."""
        space, phi, x, model, direction, slope = setup
        calls = []
        counted = Functional.from_callables(
            space,
            lambda v: (calls.append("value"), phi(v))[1],
            gradient=lambda v: (calls.append("gradient"), phi.at(v).gradient)[1],
        )
        # A direction long enough that the unit step overshoots and zooms.
        long = space.scale(50.0, direction)
        result = StrongWolfeLineSearch()(
            counted, x, long, value=model.value, slope=50.0 * slope
        )
        assert result.converged
        assert result.evaluations == calls.count("value")

    def test_the_constants_must_be_ordered(self):
        with pytest.raises(ValueError, match="decrease < curvature"):
            StrongWolfeLineSearch(decrease=0.9, curvature=0.1)

    def test_a_failed_search_reports_it(self, setup):
        space, phi, x, model, direction, slope = setup
        # An ascent direction cannot satisfy Armijo.
        result = ArmijoLineSearch(max_backtracks=5)(
            phi, x, model.gradient, value=model.value, slope=-slope
        )
        assert not result.converged


class TestTruncatedCG:
    def test_it_solves_a_definite_system(self, rng):
        space = EuclideanSpace(10)
        root = rng.normal(size=(10, 10))
        matrix = root @ root.T + 10.0 * np.identity(10)
        H = LinearOperator.from_matrix(
            space, space, matrix, traits=Traits.POSITIVE_DEFINITE, form="components"
        )
        rhs = rng.normal(size=10)
        step, reason = truncated_cg(H, rhs, rtol=1e-12)
        assert reason == "converged"
        assert np.allclose(step, np.linalg.solve(matrix, rhs), atol=1e-8)

    def test_negative_curvature_stops_it(self, rng):
        """Where CGSolver would raise, because there it is a failure."""
        space = EuclideanSpace(6)
        matrix = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, -5.0])
        H = LinearOperator.from_matrix(
            space, space, matrix, traits=Traits.SELF_ADJOINT, form="components"
        )
        _, reason = truncated_cg(H, rng.normal(size=6), radius=1.0)
        assert reason in ("negative curvature", "boundary")

    def test_the_boundary_is_respected(self, rng):
        space = EuclideanSpace(10)
        root = rng.normal(size=(10, 10))
        H = LinearOperator.from_matrix(
            space,
            space,
            root @ root.T + 10.0 * np.identity(10),
            traits=Traits.POSITIVE_DEFINITE,
            form="components",
        )
        radius = 0.01
        step, reason = truncated_cg(H, rng.normal(size=10), radius=radius, rtol=1e-14)
        assert space.norm(step) <= radius * (1.0 + 1e-9)


class TestGaussNewton:
    def test_it_is_positive_semidefinite_by_construction(self, rng):
        """J* J, recognized by the palindrome rule with nothing asserted."""
        X, Y = EuclideanSpace(5), EuclideanSpace(8)
        matrix = rng.normal(size=(8, 5))

        F = Operator.from_callables(
            X,
            Y,
            lambda x: matrix @ x,
            derivative=lambda x: LinearOperator.from_matrix(
                X, Y, matrix, form="components"
            ),
        )
        H = gauss_newton_hessian(F, X.random(rng=rng))
        assert Traits.POSITIVE_SEMIDEFINITE & H.traits
        assert np.allclose(H.matrix(form="components"), matrix.T @ matrix)

    def test_a_weighting_is_applied(self, rng):
        X, Y = EuclideanSpace(5), EuclideanSpace(8)
        matrix = rng.normal(size=(8, 5))
        weight = np.diag(np.arange(1.0, 9.0))

        F = Operator.from_callables(
            X,
            Y,
            lambda x: matrix @ x,
            derivative=lambda x: LinearOperator.from_matrix(
                X, Y, matrix, form="components"
            ),
        )
        W = LinearOperator.from_matrix(
            Y, Y, weight, traits=Traits.POSITIVE_DEFINITE, form="components"
        )
        H = gauss_newton_hessian(F, X.random(rng=rng), weighting=W)
        assert Traits.POSITIVE_SEMIDEFINITE & H.traits
        assert np.allclose(H.matrix(form="components"), matrix.T @ weight @ matrix)

    def test_a_non_self_adjoint_weighting_is_refused(self, rng):
        X, Y = EuclideanSpace(5), EuclideanSpace(8)
        matrix = rng.normal(size=(8, 5))
        F = Operator.from_callables(
            X,
            Y,
            lambda x: matrix @ x,
            derivative=lambda x: LinearOperator.from_matrix(
                X, Y, matrix, form="components"
            ),
        )
        W = LinearOperator.from_matrix(Y, Y, rng.normal(size=(8, 8)), form="components")
        with pytest.raises(ValueError, match="self-adjoint"):
            gauss_newton_hessian(F, X.random(rng=rng), weighting=W)


class TestRequirements:
    def test_a_functional_without_a_derivative_is_refused(self):
        space = EuclideanSpace(3)
        phi = Functional.from_callables(space, lambda x: float(x @ x))
        with pytest.raises(ValueError, match="needs a functional with a derivative"):
            LBFGS().minimize(phi, space.zero())

    def test_newton_needs_a_hessian(self, rng):
        space = EuclideanSpace(3)
        phi = Functional.from_callables(
            space, lambda x: float(x @ x), gradient=lambda x: 2.0 * x
        )
        with pytest.raises(ValueError, match="needs a functional with a Hessian"):
            NewtonCG().minimize(phi, space.random(rng=rng))

    def test_the_message_suggests_an_alternative(self, rng):
        space = EuclideanSpace(3)
        phi = Functional.from_callables(
            space, lambda x: float(x @ x), gradient=lambda x: 2.0 * x
        )
        with pytest.raises(ValueError, match="LBFGS"):
            NewtonCG().minimize(phi, space.random(rng=rng))

    def test_lbfgs_memory_must_be_positive(self):
        with pytest.raises(ValueError, match="at least one"):
            LBFGS(memory=0)

    def test_an_unknown_cg_variant_is_refused(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            NonlinearCG(variant="nonsense")


class TestCoordinateFreedom:
    """Every method here runs on a space with no component map."""

    @pytest.fixture
    def opaque_problem(self, rng):
        space = OpaqueSpace(np.array([1.0, 4.0, 9.0, 0.25]))
        center = space.random(rng=rng)

        def value(x):
            return 0.5 * space.squared_norm(space.subtract(x, center))

        def gradient(x):
            return space.subtract(x, center)

        def hessian(x):
            return LinearOperator.self_adjoint(
                space, lambda v: v, traits=Traits.POSITIVE_DEFINITE
            )

        phi = Functional.from_callables(
            space, value, gradient=gradient, hessian=hessian
        )
        return space, phi, center

    @pytest.mark.parametrize("method", FIRST_ORDER + SECOND_ORDER)
    def test_it_minimizes_without_components(self, method, opaque_problem, rng):
        space, phi, center = opaque_problem
        result = method(max_iterations=500).minimize(phi, space.random(rng=rng))
        assert result.converged, result.message
        assert space.norm(space.subtract(result.minimizer, center)) < 1e-6

    def test_the_space_really_has_no_coordinates(self, opaque_problem):
        from pygeoinf2.algebra.spaces import CoordinateSpace

        space, _, _ = opaque_problem
        assert not isinstance(space, CoordinateSpace)


def dense_metric_space(n, rng):
    """Non-diagonal Gram, well conditioned: the metric is what is under test,
    not floating point."""
    return make_dense_metric_space(n)


def quadratic_on(space, rng):
    """``0.5 (x, M x) - (b, x)`` in components, with its own gradient."""
    matrix = rng.standard_normal((space.dim, space.dim))
    matrix = matrix @ matrix.T + np.eye(space.dim)
    offset = rng.standard_normal(space.dim)

    return (
        Functional.from_callables(
            space,
            lambda x: float(
                0.5 * space.to_components(x) @ (matrix @ space.to_components(x))
                - offset @ space.to_components(x)
            ),
            derivative=lambda x: LinearFunctional.from_derivative_components(
                space, matrix @ space.to_components(x) - offset
            ),
        ),
        np.linalg.solve(matrix, offset),
    )


class TestTheLineSearchHandsBackWhatItKnows:
    """A Wolfe search evaluates the gradient at each trial step to test the
    curvature condition, so on success it is holding the model the caller then
    went and recomputed -- one full evaluation per outer iteration."""

    def test_the_model_comes_back(self, rng):
        space = EuclideanSpace(10)
        functional, _ = quadratic_on(space, rng)
        point = space.random(rng=rng)
        model = functional.at(point)
        direction = space.negative(model.gradient)

        result = StrongWolfeLineSearch()(
            functional,
            point,
            direction,
            value=model.value,
            slope=space.inner_product(model.gradient, direction),
            initial_step=1.0,
        )
        assert result.converged
        assert result.model is not None
        assert result.model.value == pytest.approx(result.value)

    def test_a_backtracking_search_returns_none(self, rng):
        """It only ever needs values, so it has no gradient to hand back and
        must not pretend otherwise -- the caller evaluates as before."""
        space = EuclideanSpace(10)
        functional, _ = quadratic_on(space, rng)
        point = space.random(rng=rng)
        model = functional.at(point)
        direction = space.negative(model.gradient)

        result = ArmijoLineSearch()(
            functional,
            point,
            direction,
            value=model.value,
            slope=space.inner_product(model.gradient, direction),
            initial_step=1.0,
        )
        assert result.model is None

    def test_lbfgs_spends_fewer_evaluations_for_the_same_answer(self, rng):
        """Measured on a 40-dimensional quadratic: 58 evaluations against 94,
        for the same 42 iterations and the same gradient norm."""
        space = EuclideanSpace(40)
        functional, wanted = quadratic_on(space, rng)

        result = LBFGS(max_iterations=500).minimize(functional, space.zero())
        assert result.converged
        assert np.linalg.norm(result.minimizer - wanted) < 1e-4 * np.linalg.norm(wanted)
        # One evaluation per iteration went on recomputing a known model.
        assert result.evaluations < 2 * result.iterations


class TestSteepestDescentUsesAWolfeSearch:
    """It defaults to a strong Wolfe search; it inherited a backtracking one --
    which cannot take a larger step than it is offered, and a steepest-descent
    direction carries no natural scale."""

    def test_that_is_the_default(self):
        assert isinstance(SteepestDescent()._line_search, StrongWolfeLineSearch)

    def test_it_costs_fewer_evaluations(self, rng):
        """A Wolfe search accepts a step the Armijo backtracking would have
        shrunk, so it spends fewer evaluations on the same quadratic;
        measured once at about half, which is a trend and not a contract."""
        space = EuclideanSpace(20)
        basis, _ = np.linalg.qr(rng.standard_normal((20, 20)))
        matrix = basis @ np.diag(np.geomspace(1.0, 1e3, 20)) @ basis.T
        offset = rng.standard_normal(20)
        functional = Functional.from_callables(
            space,
            lambda c: float(0.5 * c @ (matrix @ c) - offset @ c),
            derivative=lambda c: LinearFunctional.from_derivative_components(
                space, matrix @ c - offset
            ),
        )

        counts = {}
        for label, search in [
            ("armijo", ArmijoLineSearch()),
            ("wolfe", StrongWolfeLineSearch()),
        ]:
            result = SteepestDescent(max_iterations=5000, line_search=search).minimize(
                functional, space.zero()
            )
            counts[label] = result.evaluations
        assert counts["wolfe"] < counts["armijo"]


class TestOptimisersOnADenseMetric:
    """A descent direction is the Riesz representer of the derivative, so a
    non-diagonal metric is the only thing that distinguishes it from the raw
    components."""

    @pytest.mark.parametrize("method", [LBFGS, NonlinearCG, SteepestDescent])
    def test_they_find_the_minimum(self, method, rng):
        space = dense_metric_space(15, rng)
        functional, wanted = quadratic_on(space, rng)

        result = method(max_iterations=5000).minimize(functional, space.zero())
        assert result.converged
        assert space.norm(
            space.subtract(result.minimizer, space.from_components(wanted))
        ) < 1e-4 * space.norm(space.from_components(wanted))


class TestArmijoReportsItsEvaluations:
    def test_a_stop_on_the_minimum_step_counts_what_was_spent(self, rng):
        """Along a direction of ascent no step passes, and the search stops
        when the step falls below ``min_step`` -- after far fewer than
        ``max_backtracks`` evaluations."""
        from pygeoinf2.algebra.spaces import EuclideanSpace
        from pygeoinf2.numerics.convex import SquaredDistance
        from pygeoinf2.numerics.line_search import ArmijoLineSearch

        space = EuclideanSpace(3)
        functional = SquaredDistance(space)
        point = np.ones(3)
        calls = {"n": 0}
        original = functional._value

        def counted(x):
            calls["n"] += 1
            return original(x)

        functional._value = counted
        search = ArmijoLineSearch(max_backtracks=200, min_step=1e-3)
        result = search(
            functional, point, np.ones(3), value=functional(point), slope=-1.0
        )
        assert not result.converged
        assert result.evaluations == calls["n"] - 1  # the value= argument's call
        assert result.evaluations < 200


class TestEvaluationFailureProtocol:
    """``EvaluationError``: a forward that cannot converge at a trial
    model raises it, the line searches treat the step as one that went too
    far, a trust region shrinks, and an optimizer that meets it with nowhere
    to retreat returns its state instead of crashing it away."""

    @staticmethod
    def walled(space, wall=-0.5, kind="raise"):
        """``||x||^2``, unevaluable where ``x_0 < wall``.

        The minimum sits at the origin, feasible; a full first step from a
        feasible start overshoots into the wall, so every search meets the
        failure before it converges.
        """
        from pygeoinf2.algebra.operators import EvaluationError

        def value(x):
            c = space.to_components(x)
            if c[0] < wall:
                if kind == "raise":
                    raise EvaluationError("the forward did not converge")
                return float("nan")
            return float(c @ c)

        def gradient(x):
            c = space.to_components(x)
            if c[0] < wall and kind == "raise":
                raise EvaluationError("the forward did not converge")
            return space.from_components(2.0 * c)

        return Functional.from_callables(space, value, gradient=gradient)

    @pytest.mark.parametrize("kind", ["raise", "nan"])
    @pytest.mark.parametrize(
        "search", [ArmijoLineSearch(), StrongWolfeLineSearch()], ids=["armijo", "wolfe"]
    )
    def test_a_search_backtracks_out_of_the_bad_region(self, search, kind):
        space = EuclideanSpace(2)
        functional = self.walled(space, kind=kind)
        start = space.from_components(np.array([0.9, 0.0]))
        result = LBFGS(line_search=search, max_iterations=50).minimize(
            functional, start
        )
        assert result.converged, result.message
        assert result.failure is None
        assert space.norm(result.minimizer) < 1e-6

    def test_a_failure_with_nowhere_to_retreat_returns_the_state(self):
        from pygeoinf2.algebra.operators import EvaluationError

        space = EuclideanSpace(2)
        start = space.from_components(np.array([0.9, 0.0]))

        def value(x):
            c = space.to_components(x)
            return float(c @ c)

        def gradient(x):
            c = space.to_components(x)
            if not np.allclose(c, space.to_components(start)):
                raise EvaluationError("the derivative solve diverged")
            return space.from_components(2.0 * c)

        functional = Functional.from_callables(space, value, gradient=gradient)
        result = LBFGS(line_search=ArmijoLineSearch(), max_iterations=50).minimize(
            functional, start
        )
        assert not result.converged
        assert result.failure == "the derivative solve diverged"
        assert "accepted point" in result.message
        # The last good state, not the point that failed.
        assert np.allclose(
            space.to_components(result.minimizer), space.to_components(start)
        )

    def test_a_trust_region_shrinks_around_the_bad_region(self):
        from pygeoinf2.algebra.operators import EvaluationError

        space = EuclideanSpace(2)
        wall = -0.5

        def value(x):
            c = space.to_components(x)
            if c[0] < wall:
                raise EvaluationError("the forward did not converge")
            return float((c @ c) ** 0.75)

        def derivative(x):
            c = space.to_components(x)
            if c[0] < wall:
                raise EvaluationError("the forward did not converge")
            n2 = float(c @ c)
            return LinearFunctional.from_derivative_components(
                space, 1.5 * n2 ** (-0.25) * c
            )

        def hessian(x):
            c = space.to_components(x)
            n2 = float(c @ c)
            return LinearOperator.self_adjoint(
                space,
                lambda v: space.from_components(
                    1.5 * n2 ** (-0.25) * space.to_components(v)
                    - 0.75 * n2 ** (-1.25) * float(c @ space.to_components(v)) * c
                ),
            )

        functional = Functional.from_callables(
            space, value, derivative=derivative, hessian=hessian
        )
        start = space.from_components(np.array([0.9, 0.0]))
        result = TrustRegionNewton(max_iterations=100).minimize(functional, start)
        assert isinstance(result, OptimizationResult)
        assert result.failure is None
        assert result.value < functional(start)
        assert space.to_components(result.minimizer)[0] >= wall


class TestTrustRegionReportsHowItStopped:
    """The result's count and message say what happened, not what the cap was."""

    def test_a_collapsed_region_is_not_an_iteration_limit(self):
        """Every trial fails, so the radius quarters until it is nothing. That
        was reported as "iteration limit reached" with ``max_iterations`` as
        the count, although the loop had stopped long before."""
        from pygeoinf2.algebra.operators import EvaluationError

        space = EuclideanSpace(2)
        start = space.from_components(np.array([0.9, 0.0]))

        def value(x):
            c = space.to_components(x)
            # Exactly the start and nowhere else: a tolerance here would let
            # a tiny late step through as an accepted, flat move.
            if not np.array_equal(c, space.to_components(start)):
                raise EvaluationError("the forward did not converge")
            return float(c @ c)

        def derivative(x):
            return LinearFunctional.from_derivative_components(
                space, 2.0 * space.to_components(x)
            )

        def hessian(x):
            return LinearOperator.identity(space) * 2.0

        functional = Functional.from_callables(
            space, value, derivative=derivative, hessian=hessian
        )
        result = TrustRegionNewton(max_iterations=100).minimize(functional, start)
        assert not result.converged
        assert "trust region collapsed" in result.message
        assert 0 < result.iterations < 100
        assert np.allclose(
            space.to_components(result.minimizer), space.to_components(start)
        )

    def test_the_value_tolerance_is_applied(self, rng):
        """``ftol`` is documented for every optimizer, the trust region
        included. A step that buys less than ``ftol |f|`` ends the run."""
        space = EuclideanSpace(4)
        offset = rng.normal(size=4)
        base = quadratic(space, np.identity(4), offset)
        lifted = Functional.from_callables(
            space,
            lambda x: base(x) + 100.0,
            derivative=base.derivative,
            hessian=base.hessian,
        )
        start = space.from_components(offset + 1e-3 * rng.normal(size=4))
        loose = TrustRegionNewton(ftol=1e-3).minimize(lifted, start)
        assert loose.converged
        assert loose.message == "value tolerance reached"
        strict = TrustRegionNewton(ftol=0.0).minimize(lifted, start)
        assert strict.converged
        assert strict.message == "gradient tolerance reached"


class TestAugmentedLagrangian:
    """The method of multipliers on a dense metric: a quadratic with a linear
    equality against the closed form, and a ball-constrained quadratic
    against the metric projection."""

    @staticmethod
    def squared_distance(space, center):
        """``|x - a|^2`` in the space's norm, with gradient and Hessian."""
        return Functional.from_callables(
            space,
            lambda x: space.squared_norm(space.subtract(x, center)),
            gradient=lambda x: space.scale(2.0, space.subtract(x, center)),
            hessian=lambda x: LinearOperator.self_adjoint(
                space, lambda v: space.scale(2.0, v), traits=Traits.POSITIVE_DEFINITE
            ),
        )

    @staticmethod
    def hyperplane(space, normal, level):
        """``(w, x) - level``, with gradient and a zero Hessian."""
        return Functional.from_callables(
            space,
            lambda x: space.inner_product(normal, x) - level,
            gradient=lambda x: normal,
            hessian=lambda x: LinearOperator.zero(space),
        )

    @pytest.mark.parametrize("inner", [LBFGS, NewtonCG])
    def test_a_quadratic_with_a_linear_equality(self, inner, rng):
        from pygeoinf2.numerics.optimization import (
            AugmentedLagrangian,
            AugmentedLagrangianResult,
        )

        from .conftest import make_dense_metric_space

        space = make_dense_metric_space(6)
        center = space.random(rng=rng)
        normal = space.random(rng=rng)
        objective = self.squared_distance(space, center)
        constraint = self.hyperplane(space, normal, 1.0)
        method = AugmentedLagrangian(
            inner(gtol=1e-10), equalities=[constraint], tolerance=1e-8, gtol=1e-8
        )
        result = method.minimize(objective, space.zero())

        # The metric projection of the center onto the plane, and the
        # multiplier from the stationarity condition 2 (x - a) + lam w == 0.
        excess = (space.inner_product(normal, center) - 1.0) / space.squared_norm(
            normal
        )
        expected = space.axpy(-excess, normal, space.copy(center))
        assert isinstance(result, AugmentedLagrangianResult)
        assert result.converged, result.message
        assert result.minimizer == pytest.approx(expected, abs=1e-6)
        assert result.multipliers[0] == pytest.approx(2.0 * excess, abs=1e-6)
        assert result.violation <= 1e-8
        assert result.inequality_multipliers.size == 0
        assert result.value == pytest.approx(objective(expected), abs=1e-9)
        assert result.outer_iterations >= 1

    def test_a_ball_constrained_quadratic_is_the_projection(self, rng):
        from pygeoinf2.numerics.optimization import AugmentedLagrangian

        from .conftest import make_dense_metric_space

        space = make_dense_metric_space(6)
        center = space.random(rng=rng)
        radius = 0.5 * space.norm(center)
        objective = self.squared_distance(space, center)
        ball = Functional.from_callables(
            space,
            lambda x: space.squared_norm(x) - radius**2,
            gradient=lambda x: space.scale(2.0, x),
        )
        method = AugmentedLagrangian(
            LBFGS(gtol=1e-10), inequalities=[ball], tolerance=1e-8, gtol=1e-8
        )
        result = method.minimize(objective, space.zero())

        expected = space.scale(radius / space.norm(center), center)
        assert result.converged, result.message
        assert result.minimizer == pytest.approx(expected, abs=1e-6)
        # 2 (x - a) + 2 mu x == 0 with x == r a / |a| gives mu == |a| / r - 1.
        assert result.inequality_multipliers[0] == pytest.approx(
            space.norm(center) / radius - 1.0, abs=1e-5
        )
        assert result.violation <= 1e-8

    def test_an_inactive_inequality_changes_nothing(self, rng):
        from pygeoinf2.numerics.optimization import AugmentedLagrangian

        from .conftest import make_dense_metric_space

        space = make_dense_metric_space(4)
        center = space.random(rng=rng)
        radius = 2.0 * space.norm(center)
        objective = self.squared_distance(space, center)
        ball = Functional.from_callables(
            space,
            lambda x: space.squared_norm(x) - radius**2,
            gradient=lambda x: space.scale(2.0, x),
        )
        result = AugmentedLagrangian(LBFGS(), inequalities=[ball]).minimize(
            objective, space.zero()
        )
        assert result.converged
        assert result.minimizer == pytest.approx(center, abs=1e-6)
        assert result.inequality_multipliers[0] == 0.0
        assert result.violation == 0.0

    def test_the_augmented_lagrangian_itself(self, rng):
        """Its value, gradient and Hessian at given multipliers."""
        from pygeoinf2.numerics.optimization import AugmentedLagrangian
        from pygeoinf2.testing import check_gradient

        from .conftest import make_dense_metric_space

        space = make_dense_metric_space(4)
        center = space.random(rng=rng)
        normal = space.random(rng=rng)
        objective = self.squared_distance(space, center)
        plane = self.hyperplane(space, normal, 0.3)
        ball = Functional.from_callables(
            space,
            lambda x: space.squared_norm(x) - 0.1,
            gradient=lambda x: space.scale(2.0, x),
            hessian=lambda x: LinearOperator.self_adjoint(
                space, lambda v: space.scale(2.0, v)
            ),
        )
        method = AugmentedLagrangian(LBFGS(), equalities=[plane], inequalities=[ball])
        augmented = method.augmented(
            objective, multipliers=[0.7], inequality_multipliers=[0.2], penalty=3.0
        )
        assert augmented.has_hessian
        x = space.random(rng=rng)
        c = plane(x)
        g = ball(x)
        expected = (
            objective(x)
            + 0.7 * c
            + 1.5 * c * c
            + (max(0.0, 0.2 + 3.0 * g) ** 2 - 0.04) / 6.0
        )
        assert augmented(x) == pytest.approx(expected)
        assert augmented.at(x).value == pytest.approx(expected)
        check_gradient(augmented, x, rng=rng)
        # The Hessian against a difference of gradients.
        direction = space.random(rng=rng)
        step = 1e-6
        differences = space.scale(
            1.0 / (2.0 * step),
            space.subtract(
                augmented.gradient(space.axpy(step, direction, space.copy(x))),
                augmented.gradient(space.axpy(-step, direction, space.copy(x))),
            ),
        )
        assert augmented.hessian(x)(direction) == pytest.approx(
            differences, rel=1e-5, abs=1e-6
        )

    def test_what_it_refuses(self, rng):
        from pygeoinf2.numerics.optimization import AugmentedLagrangian

        space = EuclideanSpace(3)
        other = EuclideanSpace(2)
        objective = self.squared_distance(space, space.zero())
        bare = Functional.from_callables(space, lambda x: 0.0)
        with pytest.raises(ValueError, match="derivative"):
            AugmentedLagrangian(LBFGS(), equalities=[bare])
        elsewhere = self.hyperplane(other, other.zero(), 0.0)
        with pytest.raises(ValueError, match="lives on"):
            AugmentedLagrangian(LBFGS(), equalities=[elsewhere]).minimize(
                objective, space.zero()
            )
        with pytest.raises(ValueError, match="penalty"):
            AugmentedLagrangian(LBFGS(), penalty=0.0)
        with pytest.raises(ValueError, match="growth"):
            AugmentedLagrangian(LBFGS(), penalty_growth=1.0)
        with pytest.raises(ValueError, match="reduction"):
            AugmentedLagrangian(LBFGS(), reduction=1.0)

    def test_with_tolerance_copies_the_method(self, spd_problem):
        space, functional, offset = spd_problem
        tight = LBFGS(gtol=1e-7, rtol=0.0, ftol=0.0)
        loose = tight.with_tolerance(gtol=1e-2)
        assert loose is not tight
        assert isinstance(loose, LBFGS)
        by_loose = loose.minimize(functional, space.zero())
        by_tight = tight.minimize(functional, space.zero())
        assert by_loose.gradient_norm <= 1e-2
        assert by_loose.iterations < by_tight.iterations
        assert by_tight.gradient_norm <= 1e-7


def _inverse_hessian(space, matrix):
    """The exact inverse of ``quadratic``'s Hessian, as a self-adjoint operator."""
    symmetric = 0.5 * (matrix + matrix.T)
    gram = space.gram_matrix()
    return LinearOperator.self_adjoint(
        space,
        lambda v: space.from_components(
            np.linalg.solve(symmetric, gram @ space.to_components(v))
        ),
        traits=Traits.POSITIVE_DEFINITE,
    )


class TestPreconditionedNewtonCG:
    """A preconditioner changes the inner solve's path, not its answer."""

    def test_the_minimizer_is_unchanged_on_a_quadratic(self, rng):
        space = make_dense_metric_space(6)
        matrix = rng.normal(size=(6, 6))
        matrix = matrix @ matrix.T + 6 * np.eye(6)
        offset = rng.normal(size=6)
        functional = quadratic(space, matrix, offset)
        start = space.random(rng=rng)
        plain = NewtonCG(max_iterations=20).minimize(functional, start)
        preconditioned = NewtonCG(
            max_iterations=20, preconditioner=_inverse_hessian(space, matrix)
        ).minimize(functional, start)
        assert plain.converged and preconditioned.converged
        assert (
            space.norm(space.subtract(plain.minimizer, preconditioned.minimizer)) < 1e-6
        )

    def test_the_exact_inverse_solves_in_one_step(self, rng):
        space = make_dense_metric_space(8)
        matrix = rng.normal(size=(8, 8))
        matrix = matrix @ matrix.T + 8 * np.eye(8)
        hessian = quadratic(space, matrix, np.zeros(8)).hessian(space.zero())
        inverse = _inverse_hessian(space, matrix)
        rhs = space.random(rng=rng)
        exact = inverse(rhs)
        step, reason = truncated_cg(
            hessian, rhs, rtol=1e-10, max_iterations=2, preconditioner=inverse
        )
        assert reason == "converged"
        assert space.norm(space.subtract(step, exact)) < 1e-8 * space.norm(exact)
        with pytest.raises(ValueError, match="trust region"):
            truncated_cg(hessian, rhs, radius=1.0, preconditioner=inverse)


class TestThePrecisionFloor:
    """A failed line search near the minimizer is convergence at the accuracy
    the arithmetic allows, judged against the curvature there.

    The decrease a step can make is about ``||g||^2 / (2 L)``, and the
    search is comparing noise once that falls below the rounding in the
    value, ``eps |f|``; so the floor on the gradient is
    ``sqrt(eps (1 + |f|) L)``, with a margin. The same quadratic scaled by
    ten thousand has a gradient floor a hundred times higher and must be
    judged converged just the same, and a floor read off ``|f|`` alone
    would call a gradient of one converged on an objective carrying a large
    constant.
    """

    @staticmethod
    def scaled_quadratic(space, rng, scale):
        """``scale (0.5 c^T M c - b^T c)``, whose minimum value is not zero,
        so that the rounding in the value is what stops a line search."""
        root = rng.standard_normal((space.dim, space.dim))
        matrix = root @ root.T + space.dim * np.identity(space.dim)
        b = rng.standard_normal(space.dim)
        offset = np.linalg.solve(matrix, b)

        def value(x):
            c = space.to_components(x)
            return scale * (0.5 * float(c @ matrix @ c) - float(b @ c))

        def gradient(x):
            c = space.to_components(x)
            return space.from_components(space.solve_gram(scale * (matrix @ c - b)))

        def hessian(x):
            return LinearOperator.self_adjoint(
                space,
                lambda v: space.from_components(
                    space.solve_gram(scale * (matrix @ space.to_components(v)))
                ),
                traits=Traits.POSITIVE_DEFINITE,
            )

        functional = Functional.from_callables(
            space, value, gradient=gradient, hessian=hessian
        )
        return functional, offset

    @pytest.mark.parametrize("scale", [1.0, 1e4])
    @pytest.mark.parametrize(
        "method", [LBFGS, NonlinearCG, SteepestDescent], ids=lambda m: m.__name__
    )
    def test_a_line_search_method_asked_for_zero_stops_converged(
        self, method, scale, rng
    ):
        space = make_dense_metric_space(6)
        functional, offset = self.scaled_quadratic(space, rng, scale)
        result = method(gtol=0.0, rtol=0.0, ftol=0.0, max_iterations=5000).minimize(
            functional, space.zero()
        )
        assert result.converged, result
        assert "double precision" in result.message
        assert space.to_components(result.minimizer) == pytest.approx(
            offset, rel=1e-6, abs=1e-6
        )

    def test_the_floor_follows_the_curvature_not_the_value(self):
        optimizer = LBFGS()
        # A gradient of 1e-3 on an objective near a million with unit
        # curvature is nowhere near the floor, which is about 6e-5 ...
        assert not optimizer._precision_limited(1e6, 1e-3, curvature=1.0)
        assert optimizer._precision_limited(1e6, 1e-5, curvature=1.0)
        # ... and a curvature of a million raises it a thousandfold.
        assert optimizer._precision_limited(1e6, 1e-2, curvature=1e6)
        # Without an estimate the curvature is taken as the value's scale.
        assert optimizer._precision_limited(1e6, 1e-2)
        assert not optimizer._precision_limited(1e6, 1e-1)

    def test_the_curvature_is_read_off_the_hessian_or_the_last_step(self, rng):
        space = make_dense_metric_space(5)
        functional, _ = self.scaled_quadratic(space, rng, 1.0)
        x = space.random(rng=rng)
        model = functional.at(x)
        gradient_norm = space.norm(model.gradient)
        rayleigh = LBFGS._curvature(space, model, model.gradient, gradient_norm)
        hessian = space.inner_product(model.gradient, model.hessian(model.gradient))
        assert rayleigh == pytest.approx(hessian / gradient_norm**2)
        # A functional without a Hessian falls back to the secant estimate,
        # which on a quadratic is the curvature along the last step.
        plain = Functional.from_callables(
            space, functional, derivative=lambda v: functional.at(v).derivative
        )
        previous = space.random(rng=rng)
        previous_model = plain.at(previous)
        secant = LBFGS._curvature(
            space,
            plain.at(x),
            plain.at(x).gradient,
            gradient_norm,
            previous_x=previous,
            previous_gradient=previous_model.gradient,
        )
        step = space.subtract(x, previous)
        along = space.inner_product(step, model.hessian(step)) / space.norm(step) ** 2
        assert secant > 0.0
        assert secant >= along * 0.99
        assert LBFGS._curvature(space, plain.at(x), plain.at(x).gradient, 1.0) is None
