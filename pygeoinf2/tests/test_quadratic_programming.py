"""The QP backends, and the proximal bundle method built on them."""

import numpy as np
import pytest

from pygeoinf2.algebra.operators import Functional, LinearFunctional, LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.numerics.convex import ProximalBundleMethod
from pygeoinf2.numerics.quadratic_programming import (
    ClarabelQPSolver,
    OSQPQPSolver,
    QPSolver,
    SciPyQPSolver,
    best_available_qp_solver,
)


def backends():
    """Every backend that is actually installed."""
    available = [SciPyQPSolver]
    for optional in (OSQPQPSolver, ClarabelQPSolver):
        try:
            optional()
        except ImportError:  # pragma: no cover - depends on the install
            continue
        available.append(optional)
    return available


@pytest.fixture
def program(rng):
    """A strictly convex QP with one equality and some one-sided bounds."""
    size, rows = 8, 5
    root = rng.standard_normal((size, size))
    quadratic = root @ root.T + np.eye(size)
    linear = rng.standard_normal(size)
    constraint = np.vstack([np.ones((1, size)), rng.standard_normal((rows - 1, size))])
    lower = np.concatenate([[1.0], np.full(rows - 1, -np.inf)])
    upper = np.concatenate([[1.0], rng.uniform(0.5, 2.0, rows - 1)])
    return quadratic, linear, constraint, lower, upper


class TestTheBackendsAgree:
    """Three solvers, one answer. They are interchangeable or they are not."""

    @pytest.mark.parametrize("backend", backends())
    def test_each_solves_it_feasibly(self, backend, program):
        quadratic, linear, constraint, lower, upper = program
        result = backend().solve(quadratic, linear, constraint, lower, upper)

        assert result.solved
        assert np.all(constraint @ result.x >= lower - 1e-6)
        assert np.all(constraint @ result.x <= upper + 1e-6)

    def test_they_reach_the_same_minimum(self, program):
        quadratic, linear, constraint, lower, upper = program
        objectives = [
            backend().solve(quadratic, linear, constraint, lower, upper).objective
            for backend in backends()
        ]
        for objective in objectives[1:]:
            # OSQP's own default accuracy is 1e-6 relative; the agreement
            # asked for sits above it.
            assert objective == pytest.approx(objectives[0], rel=1e-5, abs=1e-7)

    def test_they_all_satisfy_the_protocol(self):
        for backend in backends():
            assert isinstance(backend(), QPSolver)

    def test_the_best_available_is_one_of_them(self):
        assert isinstance(best_available_qp_solver(), QPSolver)

    @pytest.mark.parametrize("backend", backends())
    def test_a_warm_start_is_accepted(self, backend, program, rng):
        """Clarabel has no use for one and must still take it, or the three
        are not interchangeable."""
        quadratic, linear, constraint, lower, upper = program
        result = backend().solve(
            quadratic, linear, constraint, lower, upper, x0=rng.standard_normal(8)
        )
        assert result.solved


class TestBadProgrammesAreRefused:
    @pytest.mark.parametrize("backend", backends())
    def test_mismatched_shapes(self, backend):
        with pytest.raises(ValueError, match="must be"):
            backend().solve(np.eye(3), np.zeros(4), np.ones((1, 4)), [0.0], [1.0])

    @pytest.mark.parametrize("backend", backends())
    def test_crossed_bounds(self, backend):
        with pytest.raises(ValueError, match="lower bound"):
            backend().solve(np.eye(2), np.zeros(2), np.ones((1, 2)), [2.0], [1.0])


class TestTheProximalSubproblemBackend:
    """The simplex QP in the number of cuts, solved exactly when a backend is
    installed, and by the built-in projected gradient otherwise. Measured on a Backus dual over sixteen directions, Clarabel
    took 0.37 s against the built-in method's 10.8 s and matched the primal
    route to 6.5e-11 against 1.7e-8, with none of its nine warnings."""

    @staticmethod
    def near_parallel_bundle(rng, count, dimension):
        base = rng.standard_normal(dimension)
        gradients = np.array(
            [base + 1e-3 * rng.standard_normal(dimension) for _ in range(count)]
        )
        gram = gradients @ gradients.T
        errors = np.abs(rng.standard_normal(count)) * 1e-2
        return gram, errors

    def test_the_default_is_an_exact_backend_where_one_is_installed(self):
        pytest.importorskip("clarabel")
        method = ProximalBundleMethod()
        assert isinstance(method.qp_solver, ClarabelQPSolver)
        assert ProximalBundleMethod(qp_solver="builtin").qp_solver is None

    def test_the_backend_does_at_least_as_well_as_the_projected_gradient(self, rng):
        """On the badly conditioned bundle a converging method produces,
        which is where the projected gradient has its residual floor. The
        objective the backend reaches is no worse, and the weights are a
        point of the simplex."""
        pytest.importorskip("clarabel")
        gram, errors = self.near_parallel_bundle(rng, 30, 12)
        exact = ProximalBundleMethod(qp_solver=ClarabelQPSolver())
        builtin = ProximalBundleMethod(qp_solver="builtin")

        def objective(weights):
            return 0.5 * weights @ gram @ weights + errors @ weights

        approximate = builtin._simplex_weights(gram, errors)
        precise = exact._simplex_weights(gram, errors)
        assert precise.sum() == pytest.approx(1.0)
        assert np.all(precise >= 0.0)
        assert objective(precise) <= objective(approximate) + 1e-7 * (
            1.0 + abs(objective(approximate))
        )

    def test_both_routes_reach_the_same_minimum(self, rng):
        size, rows = 10, 6
        matrix = rng.standard_normal((rows, size))
        offset = rng.standard_normal(rows)
        space = EuclideanSpace(size)
        functional = Functional.from_callables(
            space,
            lambda c: float(np.abs(matrix @ c - offset).sum() + 0.1 * c @ c),
            derivative=lambda c: LinearFunctional.from_derivative_components(
                space, matrix.T @ np.sign(matrix @ c - offset) + 0.2 * c
            ),
        )
        exact = ProximalBundleMethod(tolerance=1e-8, max_iterations=300).minimize(
            functional, space.zero()
        )
        builtin = ProximalBundleMethod(
            tolerance=1e-8, max_iterations=300, qp_solver="builtin"
        ).minimize(functional, space.zero())
        assert exact.value == pytest.approx(builtin.value, abs=1e-5)


class TestTheBackendOrder:
    def test_clarabel_is_preferred_when_installed(self):
        pytest.importorskip("clarabel")
        assert isinstance(best_available_qp_solver(), ClarabelQPSolver)


class TestTheTwoRoutesMeet:
    """Chambolle-Pock maximizes over the feasible set; the bundle method
    minimizes the dual. Strong duality says they meet, and two unrelated
    algorithms agreeing is the strongest check either one gets."""

    @pytest.fixture
    def setting(self, rng):
        from pygeoinf2.geometry.convex import Ball
        from pygeoinf2.inference import BackusGilbertParker
        from pygeoinf2.inference.problem import LinearForwardProblem

        model = EuclideanSpace(12)
        data_space = EuclideanSpace(5)
        property_space = EuclideanSpace(2)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.standard_normal((5, 12)), form="components"
        )
        target = LinearOperator.from_matrix(
            model, property_space, rng.standard_normal((2, 12)), form="components"
        )
        truth = model.random(rng=rng)
        data = forward(truth)
        # Wide enough that the set is genuinely non-empty: an empty one has no
        # supremum for the two to agree on.
        prior = Ball(model, radius=2.0 * model.norm(truth))
        noise = Ball(data_space, radius=0.05)
        feasible = BackusGilbertParker(
            LinearForwardProblem(forward, error=noise), target, prior
        )(data)
        return feasible, property_space, prior, noise, forward, data

    @staticmethod
    def directions(space, count):
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        return [
            space.from_components(np.array([np.cos(angle), np.sin(angle)]))
            for angle in angles
        ]

    def test_the_splitting_and_the_dual_support_values_agree(self, setting):
        """To 1.7e-8 relative, measured over sixteen directions."""
        from pygeoinf2.geometry.feasible import Splitting

        feasible, space, *_ = setting
        directions = self.directions(space, 8)
        dual = feasible.support_values(directions, via="dual")
        tolerance = 1e-7
        splitting = Splitting(tolerance=tolerance, max_iterations=50_000)
        sweep = feasible.support_sweep(directions, via=splitting)
        assert sweep.values == pytest.approx(dual, rel=1e-5)
        for result in sweep.results:
            assert result.converged
            # The iterate is feasible only to the tolerance, so the inner
            # value may sit above the outer bound by that order.
            assert result.inner <= result.value + 10 * tolerance * (
                1 + abs(result.value)
            )

    def test_the_answer_is_actually_feasible(self, setting):
        """Which is the thing the residual is measuring, and the thing an
        aliasing bug in the over-relaxation silently broke: the iterate
        became the extrapolated point, and the method converged to
        something outside the set with a residual that would not go below
        0.9."""
        from pygeoinf2.numerics.convex import ChambollePockSolver

        feasible, space, prior, noise, forward, data = setting
        solver = ChambollePockSolver(
            prior, noise, forward, data, tolerance=1e-7, max_iterations=50_000
        )
        result = solver.solve(feasible.property_operator.adjoint(space.basis_vector(0)))
        assert result.converged
        assert result.residual < 1e-6
        model_space, data_space = forward.domain, forward.codomain
        assert (
            model_space.norm(
                model_space.subtract(prior.project(result.minimizer), result.minimizer)
            )
            < 1e-6
        )
        assert (
            data_space.norm(
                data_space.subtract(
                    data_space.add(forward(result.minimizer), result.discrepancy), data
                )
            )
            < 1e-6
        )

    def test_the_sets_must_live_where_the_operator_does(self, rng):
        from pygeoinf2.geometry.convex import Ball
        from pygeoinf2.numerics.convex import ChambollePockSolver

        model = EuclideanSpace(6)
        data_space = EuclideanSpace(3)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.standard_normal((3, 6)), form="components"
        )
        with pytest.raises(ValueError, match="model space"):
            ChambollePockSolver(
                Ball(data_space, radius=1.0),
                Ball(data_space, radius=1.0),
                forward,
                data_space.zero(),
            )
        with pytest.raises(ValueError, match="data space"):
            ChambollePockSolver(
                Ball(model, radius=1.0),
                Ball(model, radius=1.0),
                forward,
                data_space.zero(),
            )
