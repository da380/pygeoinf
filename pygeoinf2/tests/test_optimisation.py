"""Optimisation: correctness, coordinate freedom, and metric awareness."""

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
from pygeoinf2.numerics.optimisation import (
    LBFGS,
    NewtonCG,
    NonlinearCG,
    OptimisationResult,
    SteepestDescent,
    TrustRegionNewton,
    gauss_newton_hessian,
    truncated_cg,
)
from pygeoinf2.traits import Traits

from .conftest import make_weighted_space
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
        return LinearOperator.from_component_matrix(
            space, space, matrix, traits=Traits.SELF_ADJOINT
        )

    return Functional.from_callables(
        space, value, derivative=derivative, hessian=hessian
    )


FIRST_ORDER = [SteepestDescent, NonlinearCG, LBFGS]
SECOND_ORDER = [NewtonCG, TrustRegionNewton]


@pytest.fixture
def spd_problem(rng):
    space = EuclideanSpace(20)
    root = rng.normal(size=(20, 20))
    matrix = root @ root.T + 20.0 * np.identity(20)
    offset = rng.normal(size=20)
    return space, quadratic(space, matrix, offset), offset


class TestQuadratics:
    @pytest.mark.parametrize("method", FIRST_ORDER + SECOND_ORDER)
    def test_the_minimum_is_found(self, method, spd_problem, rng):
        space, phi, offset = spd_problem
        result = method(max_iterations=3000).minimise(phi, space.random(rng=rng))
        assert isinstance(result, OptimisationResult)
        assert result.converged, result.message
        assert np.allclose(result.minimiser, offset, atol=1e-6)

    @pytest.mark.parametrize("method", FIRST_ORDER + SECOND_ORDER)
    def test_it_works_on_a_weighted_space(self, method, rng):
        """Where the metric is not the identity, and the gradient is not the array."""
        space = make_weighted_space()
        root = rng.normal(size=(space.dim, space.dim))
        matrix = root @ root.T + space.dim * np.identity(space.dim)
        offset = rng.normal(size=space.dim)
        phi = quadratic(space, matrix, offset)

        result = method(max_iterations=3000).minimise(phi, space.random(rng=rng))
        assert result.converged, result.message
        assert np.allclose(space.to_components(result.minimiser), offset, atol=1e-6)

    def test_newton_beats_steepest_descent(self, spd_problem, rng):
        space, phi, _ = spd_problem
        start = space.random(rng=rng)
        newton = NewtonCG().minimise(phi, start)
        steepest = SteepestDescent(max_iterations=3000).minimise(phi, start)
        assert newton.iterations < steepest.iterations

    def test_the_history_decreases(self, spd_problem, rng):
        space, phi, _ = spd_problem
        result = LBFGS().minimise(phi, space.random(rng=rng))
        assert np.all(np.diff(result.history) <= 1e-12)


class TestRosenbrock:
    """The banana valley: a genuine test of the direction rules.

    Steepest descent is excluded, and deliberately. It is famously unable to
    traverse this valley in a reasonable number of iterations -- the count runs
    to tens of thousands -- so including it would mean either a slow test or a
    tolerance loose enough to prove nothing. Its behaviour is pinned separately
    below instead.
    """

    @pytest.mark.parametrize("method", [NonlinearCG, LBFGS] + SECOND_ORDER)
    def test_the_valley_is_traversed(self, method):
        space = EuclideanSpace(2)
        phi = rosenbrock(space)
        start = np.array([-1.2, 1.0])
        result = method(max_iterations=5000, gtol=1e-8).minimise(phi, start)
        assert np.allclose(result.minimiser, [1.0, 1.0], atol=1e-4), result

    def test_steepest_descent_makes_progress_but_does_not_finish(self):
        """Pinning the known behaviour rather than pretending otherwise."""
        space = EuclideanSpace(2)
        phi = rosenbrock(space)
        start = np.array([-1.2, 1.0])
        result = SteepestDescent(max_iterations=2000).minimise(phi, start)
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

        result = TrustRegionNewton(max_iterations=2000).minimise(phi, start)
        assert np.allclose(result.minimiser, [1.0, 1.0], atol=1e-4), result


class TestMetricAwareness:
    """The decisive difference from a component-space optimiser.

    Take ``phi(x) == 0.5 ||x - a||^2`` in the *space's* norm. Its gradient is
    ``x - a``, so the Hessian is the identity on the space and steepest descent
    with an exact step converges immediately, whatever the metric.

    In components the same function is ``0.5 (c - a)^T G (c - a)``, whose
    Hessian is the Gram matrix. A component-space method therefore sees a
    condition number equal to the spread of the metric values, and takes more
    iterations the worse the discretisation is scaled. That is the conditioning
    half of DESIGN.md 5.6, and it is why this is worth writing rather than
    wrapping.
    """

    @staticmethod
    def _space_metric_functional(space, centre):
        def value(x):
            return 0.5 * space.squared_norm(space.subtract(x, centre))

        def gradient(x):
            return space.subtract(x, centre)

        return Functional.from_callables(space, value, gradient=gradient)

    @staticmethod
    def _component_metric_functional(space, centre):
        """The same function, but with the gradient taken in the components.

        This is what a component-space optimiser effectively minimises, and it
        is a genuinely different problem whenever the metric is not the
        identity.
        """
        offset = space.to_components(centre)

        def value(x):
            c = space.to_components(x) - offset
            return 0.5 * float(c @ c)

        def gradient(x):
            return space.from_components(space.to_components(x) - offset)

        return Functional.from_callables(space, value, gradient=gradient)

    def test_the_metric_aware_problem_is_perfectly_conditioned(self, rng):
        space = make_weighted_space()
        centre = space.random(rng=rng)
        phi = self._space_metric_functional(space, centre)

        result = SteepestDescent(max_iterations=500).minimise(
            phi, space.random(rng=rng)
        )
        assert result.converged
        assert space.norm(space.subtract(result.minimiser, centre)) < 1e-6
        # A Hessian of identity means very few iterations, whatever the metric.
        assert result.iterations <= 40

    def test_the_two_functionals_really_differ(self, rng):
        """Otherwise the previous test would be proving nothing."""
        space = make_weighted_space()
        centre = space.random(rng=rng)
        x = space.random(rng=rng)
        space_metric = self._space_metric_functional(space, centre)
        component = self._component_metric_functional(space, centre)
        assert not np.isclose(space_metric(x), component(x))

    def test_convergence_does_not_degrade_with_the_metric(self, rng):
        """Refining a discretisation should not change the iteration count."""
        from .conftest import WeightedSpace

        counts = []
        for spread in (1.0, 100.0, 10000.0):
            space = WeightedSpace(np.array([1.0, spread, spread**0.5, 1.0]))
            centre = space.random(rng=rng)
            phi = self._space_metric_functional(space, centre)
            result = SteepestDescent(max_iterations=2000).minimise(
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
        H = LinearOperator.from_component_matrix(
            space, space, matrix, traits=Traits.POSITIVE_DEFINITE
        )
        rhs = rng.normal(size=10)
        step, reason = truncated_cg(H, rhs, rtol=1e-12)
        assert reason == "converged"
        assert np.allclose(step, np.linalg.solve(matrix, rhs), atol=1e-8)

    def test_negative_curvature_stops_it(self, rng):
        """Where CGSolver would raise, because there it is a failure."""
        space = EuclideanSpace(6)
        matrix = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, -5.0])
        H = LinearOperator.from_component_matrix(
            space, space, matrix, traits=Traits.SELF_ADJOINT
        )
        _, reason = truncated_cg(H, rng.normal(size=6), radius=1.0)
        assert reason in ("negative curvature", "boundary")

    def test_the_boundary_is_respected(self, rng):
        space = EuclideanSpace(10)
        root = rng.normal(size=(10, 10))
        H = LinearOperator.from_component_matrix(
            space,
            space,
            root @ root.T + 10.0 * np.identity(10),
            traits=Traits.POSITIVE_DEFINITE,
        )
        radius = 0.01
        step, reason = truncated_cg(H, rng.normal(size=10), radius=radius, rtol=1e-14)
        assert space.norm(step) <= radius * (1.0 + 1e-9)


class TestGaussNewton:
    def test_it_is_positive_semidefinite_by_construction(self, rng):
        """J* J, recognised by the palindrome rule with nothing asserted."""
        X, Y = EuclideanSpace(5), EuclideanSpace(8)
        matrix = rng.normal(size=(8, 5))

        F = Operator.from_callables(
            X,
            Y,
            lambda x: matrix @ x,
            derivative=lambda x: LinearOperator.from_component_matrix(X, Y, matrix),
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
            derivative=lambda x: LinearOperator.from_component_matrix(X, Y, matrix),
        )
        W = LinearOperator.from_component_matrix(
            Y, Y, weight, traits=Traits.POSITIVE_DEFINITE
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
            derivative=lambda x: LinearOperator.from_component_matrix(X, Y, matrix),
        )
        W = LinearOperator.from_component_matrix(Y, Y, rng.normal(size=(8, 8)))
        with pytest.raises(ValueError, match="self-adjoint"):
            gauss_newton_hessian(F, X.random(rng=rng), weighting=W)


class TestRequirements:
    def test_a_functional_without_a_derivative_is_refused(self):
        space = EuclideanSpace(3)
        phi = Functional.from_callables(space, lambda x: float(x @ x))
        with pytest.raises(ValueError, match="needs a functional with a derivative"):
            LBFGS().minimise(phi, space.zero())

    def test_newton_needs_a_hessian(self, rng):
        space = EuclideanSpace(3)
        phi = Functional.from_callables(
            space, lambda x: float(x @ x), gradient=lambda x: 2.0 * x
        )
        with pytest.raises(ValueError, match="needs a functional with a Hessian"):
            NewtonCG().minimise(phi, space.random(rng=rng))

    def test_the_message_suggests_an_alternative(self, rng):
        space = EuclideanSpace(3)
        phi = Functional.from_callables(
            space, lambda x: float(x @ x), gradient=lambda x: 2.0 * x
        )
        with pytest.raises(ValueError, match="LBFGS"):
            NewtonCG().minimise(phi, space.random(rng=rng))

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
        centre = space.random(rng=rng)

        def value(x):
            return 0.5 * space.squared_norm(space.subtract(x, centre))

        def gradient(x):
            return space.subtract(x, centre)

        def hessian(x):
            return LinearOperator.self_adjoint(
                space, lambda v: v, traits=Traits.POSITIVE_DEFINITE
            )

        phi = Functional.from_callables(
            space, value, gradient=gradient, hessian=hessian
        )
        return space, phi, centre

    @pytest.mark.parametrize("method", FIRST_ORDER + SECOND_ORDER)
    def test_it_minimises_without_components(self, method, opaque_problem, rng):
        space, phi, centre = opaque_problem
        result = method(max_iterations=500).minimise(phi, space.random(rng=rng))
        assert result.converged, result.message
        assert space.norm(space.subtract(result.minimiser, centre)) < 1e-6

    def test_the_space_really_has_no_coordinates(self, opaque_problem):
        from pygeoinf2.algebra.spaces import CoordinateSpace

        space, _, _ = opaque_problem
        assert not isinstance(space, CoordinateSpace)
