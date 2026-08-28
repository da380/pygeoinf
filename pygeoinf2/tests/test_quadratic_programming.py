"""The QP backends, and the level bundle method built on them."""

import numpy as np
import pytest

from pygeoinf2.algebra.operators import Functional, LinearFunctional, LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.numerics.convex import LevelBundleMethod, ProximalBundleMethod
from pygeoinf2.numerics.quadratic_programming import (
    ClarabelQPSolver,
    OSQPQPSolver,
    QPResult,
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
def programme(rng):
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
    def test_each_solves_it_feasibly(self, backend, programme):
        quadratic, linear, constraint, lower, upper = programme
        result = backend().solve(quadratic, linear, constraint, lower, upper)

        assert result.solved
        assert np.all(constraint @ result.x >= lower - 1e-6)
        assert np.all(constraint @ result.x <= upper + 1e-6)

    def test_they_reach_the_same_minimum(self, programme):
        quadratic, linear, constraint, lower, upper = programme
        objectives = [
            backend().solve(quadratic, linear, constraint, lower, upper).objective
            for backend in backends()
        ]
        for objective in objectives[1:]:
            assert objective == pytest.approx(objectives[0], rel=1e-6, abs=1e-8)

    def test_they_all_satisfy_the_protocol(self):
        for backend in backends():
            assert isinstance(backend(), QPSolver)

    def test_the_best_available_is_one_of_them(self):
        assert isinstance(best_available_qp_solver(), QPSolver)

    @pytest.mark.parametrize("backend", backends())
    def test_a_warm_start_is_accepted(self, backend, programme, rng):
        """Clarabel has no use for one and must still take it, or the three
        are not interchangeable."""
        quadratic, linear, constraint, lower, upper = programme
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


class TestTheLevelBundleBoundIsABound:
    """The reason this method exists: its gap is a statement about the
    minimum, where the proximal method's is a statement about its model."""

    @pytest.fixture
    def quadratic_problem(self, rng):
        size = 8
        root = rng.standard_normal((size, size))
        matrix = root @ root.T + np.eye(size)
        offset = rng.standard_normal(size)
        space = EuclideanSpace(size)
        functional = Functional.from_callables(
            space,
            lambda c: float(0.5 * c @ (matrix @ c) - offset @ c),
            derivative=lambda c: LinearFunctional.from_derivative_components(
                space, matrix @ c - offset
            ),
        )
        exact = float(-0.5 * offset @ np.linalg.solve(matrix, offset))
        return space, functional, exact

    @pytest.mark.parametrize("iterations", [15, 40, 200])
    def test_the_lower_bound_never_passes_the_minimum(
        self, quadratic_problem, iterations
    ):
        """At every stage, not just at the end -- an invalid bound part way
        through is still an invalid bound."""
        space, functional, exact = quadratic_problem
        result = LevelBundleMethod(
            tolerance=1e-14, iterations=iterations
        ).minimise(functional, space.zero())

        assert result.value - result.gap <= exact + 1e-9
        assert result.value >= exact - 1e-9

    def test_it_closes_on_the_answer(self, quadratic_problem):
        space, functional, exact = quadratic_problem
        # Relative to the value, which here is small: 1e-6 of 0.056 is already
        # 5.6e-8 absolute, and asking for 1e-8 of it would be asking the
        # bundle for ten significant figures.
        result = LevelBundleMethod(tolerance=1e-6).minimise(functional, space.zero())
        assert result.converged
        assert result.value == pytest.approx(exact, abs=1e-5)

    def test_it_agrees_with_the_proximal_method_on_a_nonsmooth_problem(self, rng):
        """Different methods, same minimum, or one of them is wrong."""
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
        level = LevelBundleMethod(tolerance=1e-6, iterations=300).minimise(
            functional, space.zero()
        )
        proximal = ProximalBundleMethod(tolerance=1e-6, iterations=300).minimise(
            functional, space.zero()
        )
        assert level.value == pytest.approx(proximal.value, abs=1e-4)

    def test_v1s_worked_example(self):
        """``x^2 + 2x``, minimum minus one at minus one -- v1's doctest."""
        space = EuclideanSpace(1)
        functional = Functional.from_callables(
            space,
            lambda c: float(c[0] ** 2 + 2 * c[0]),
            derivative=lambda c: LinearFunctional.from_derivative_components(
                space, np.array([2 * c[0] + 2.0])
            ),
        )
        result = LevelBundleMethod(tolerance=1e-8).minimise(
            functional, space.from_components(np.array([2.0]))
        )
        assert result.minimiser == pytest.approx([-1.0], abs=1e-3)
        assert result.value == pytest.approx(-1.0, abs=1e-6)

    def test_an_alpha_outside_the_unit_interval_is_refused(self):
        for bad in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError, match="alpha"):
                LevelBundleMethod(alpha=bad)

    def test_a_space_without_coordinates_is_refused(self):
        """The master problem is a QP in components, so unlike the proximal
        method this one needs a basis."""
        from .doubles import OpaqueSpace

        space = OpaqueSpace(6)
        functional = Functional.from_callables(
            space, lambda x: space.squared_norm(x), derivative=None
        )
        with pytest.raises(Exception):
            LevelBundleMethod().minimise(functional, space.zero())


class TestTheTwoRoutesMeet:
    """Chambolle-Pock maximises over the feasible set; the bundle method
    minimises the dual. Strong duality says they meet, and two unrelated
    algorithms agreeing is the strongest check either one gets."""

    @pytest.fixture
    def setting(self, rng):
        from pygeoinf2.geometry.convex import Ball
        from pygeoinf2.inference import DualFeasibleProperty
        from pygeoinf2.inference.problem import LinearForwardProblem

        model = EuclideanSpace(12)
        data_space = EuclideanSpace(5)
        target_space = EuclideanSpace(2)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.standard_normal((5, 12)), form="components"
        )
        target = LinearOperator.from_matrix(
            model, target_space, rng.standard_normal((2, 12)), form="components"
        )
        truth = model.random(rng=rng)
        data = forward(truth)
        # Wide enough that the set is genuinely non-empty: an empty one has no
        # supremum for the two to agree on.
        prior = Ball(model, radius=2.0 * model.norm(truth))
        estimator = DualFeasibleProperty(
            LinearForwardProblem(forward, error=Ball(data_space, radius=0.05)),
            target,
            prior,
        )
        return estimator, target_space, data

    @staticmethod
    def directions(space, count):
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        return [
            space.from_components(np.array([np.cos(angle), np.sin(angle)]))
            for angle in angles
        ]

    def test_the_primal_and_dual_support_values_agree(self, setting):
        """To 1.7e-8 relative, measured over sixteen directions."""
        estimator, space, data = setting
        directions = self.directions(space, 8)

        dual = estimator.support_values(directions, data)
        primal = estimator.support_values(
            directions, data, route="primal", tolerance=1e-9, iterations=50_000
        )
        assert primal == pytest.approx(dual, rel=1e-5)

    def test_the_primal_route_converges(self, setting):
        estimator, space, data = setting
        solver = estimator.primal_solver(data, tolerance=1e-9, iterations=50_000)
        result = solver.solve(estimator._target.adjoint(space.basis_vector(0)))

        assert result.converged
        assert result.residual < 1e-8

    def test_the_answer_is_actually_feasible(self, setting):
        """Which is the thing the residual is measuring, and the thing an
        aliasing bug in the over-relaxation silently broke: the iterate became
        the extrapolated point, and the method converged to something outside
        the set with a residual that would not go below 0.9."""
        estimator, space, data = setting
        solver = estimator.primal_solver(data, tolerance=1e-9, iterations=50_000)
        result = solver.solve(estimator._target.adjoint(space.basis_vector(0)))

        model_space = estimator._problem.model_space
        data_space = estimator.data_space
        assert model_space.norm(
            model_space.subtract(estimator._prior.project(result.model), result.model)
        ) < 1e-8
        assert data_space.norm(
            data_space.subtract(
                data_space.add(
                    estimator._problem.forward_operator(result.model),
                    result.discrepancy,
                ),
                data,
            )
        ) < 1e-7

    def test_an_unknown_route_is_refused(self, setting):
        estimator, space, data = setting
        with pytest.raises(ValueError, match="The route is"):
            estimator.support_values(self.directions(space, 2), data, route="sideways")

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


class TestTheKKTRouteAgreesWhereItApplies:
    """The third route: for balls and ellipsoids the KKT conditions give the
    answer in closed form, up to a two-variable root find on the multipliers,
    and the model space is never discretised."""

    @pytest.fixture
    def setting(self, rng):
        from pygeoinf2.geometry.convex import Ball

        model = EuclideanSpace(12)
        data_space = EuclideanSpace(5)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.standard_normal((5, 12)), form="components"
        )
        truth = model.random(rng=rng)
        return (
            forward,
            forward(truth),
            Ball(model, radius=2.0 * model.norm(truth)),
            Ball(data_space, radius=0.05),
        )

    def test_the_closed_form_is_the_dense_solve(self, setting, rng):
        """The Woodbury reduction, checked against forming and solving
        ``(lambda B + mu G* V G) m == r`` directly. This is the part that has
        to be exactly right; everything else is a root find on top of it."""
        from pygeoinf2.numerics.convex import PrimalKKTSolver

        forward, data, prior, noise = setting
        matrix = forward.matrix(form="components")
        solver = PrimalKKTSolver(prior, noise, forward, data)
        objective = forward.domain.random(rng=rng)

        for lam, mu in [(0.5, 0.01), (2.0, 1.0), (7.0, 50.0)]:
            reference = np.linalg.solve(
                lam * np.eye(12) + mu * matrix.T @ matrix,
                objective + mu * matrix.T @ data,
            )
            assert solver._model((lam, mu), objective) == pytest.approx(
                reference, abs=1e-10
            )

    def test_it_agrees_with_the_primal_route(self, setting, rng):
        """To 1e-10 on objectives where the data constraint has slack.
        Measured over six random directions."""
        from pygeoinf2.numerics.convex import ChambollePockSolver, PrimalKKTSolver

        forward, data, prior, noise = setting
        kkt = PrimalKKTSolver(prior, noise, forward, data)
        splitting = ChambollePockSolver(
            prior,
            noise,
            forward,
            data,
            iterations=100_000,
            tolerance=1e-11,
            rng=np.random.default_rng(1),
        )
        agreements = []
        for _ in range(6):
            objective = forward.domain.random(rng=rng)
            exact = kkt.solve(objective)
            iterated = splitting.solve(objective)
            assert exact.converged
            agreements.append(
                abs(exact.value - iterated.value) / abs(iterated.value)
            )
        # Most agree to machine precision; those where the second multiplier
        # saturates against a tight noise ball agree to about 1e-3, which is
        # the method's documented limit and not a failure of the port -- v1
        # returns the identical multiplier there.
        agreements = np.array(agreements)
        assert np.median(agreements) < 1e-8
        assert agreements.max() < 5e-3

    def test_a_slack_data_constraint_short_circuits(self, setting, rng):
        """When the prior's own support point already fits the data, that is
        the answer and the second multiplier is zero -- no root find at all."""
        from pygeoinf2.geometry.convex import Ball
        from pygeoinf2.numerics.convex import PrimalKKTSolver

        forward, data, prior, _ = setting
        roomy = Ball(forward.codomain, radius=1e6)
        solver = PrimalKKTSolver(prior, roomy, forward, data)

        objective = forward.domain.random(rng=rng)
        result = solver.solve(objective)
        assert result.multipliers[1] == 0.0
        assert result.value == pytest.approx(
            forward.domain.inner_product(
                objective, prior.support_maximiser(objective)
            )
        )

    def test_a_general_convex_set_is_refused(self, setting):
        """It is the *quadratic* constraint that makes the closed form
        possible, so anything else is turned away by name, pointing at the
        route that does handle it."""
        from pygeoinf2.geometry.convex import HalfSpace, Polytope
        from pygeoinf2.numerics.convex import PrimalKKTSolver

        forward, data, prior, noise = setting
        model = forward.domain
        box = Polytope(
            model,
            [
                HalfSpace(model, model.basis_vector(index), offset=1.0)
                for index in range(3)
            ],
            outer=True,
        )
        with pytest.raises(TypeError, match="ChambollePockSolver"):
            PrimalKKTSolver(box, noise, forward, data)

    def test_all_three_routes_agree(self, rng):
        """The whole point of having three."""
        from pygeoinf2.geometry.convex import Ball
        from pygeoinf2.inference import DualFeasibleProperty
        from pygeoinf2.inference.problem import LinearForwardProblem

        model = EuclideanSpace(12)
        data_space = EuclideanSpace(5)
        target_space = EuclideanSpace(2)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.standard_normal((5, 12)), form="components"
        )
        target = LinearOperator.from_matrix(
            model, target_space, rng.standard_normal((2, 12)), form="components"
        )
        truth = model.random(rng=rng)
        data = forward(truth)
        estimator = DualFeasibleProperty(
            LinearForwardProblem(forward, error=Ball(data_space, radius=0.05)),
            target,
            Ball(model, radius=2.0 * model.norm(truth)),
        )
        directions = [target_space.basis_vector(0), target_space.basis_vector(1)]

        dual = estimator.support_values(directions, data)
        primal = estimator.support_values(
            directions, data, route="primal", tolerance=1e-9, iterations=50_000
        )
        exact = estimator.support_values(directions, data, route="kkt")

        # The looser tolerance is the KKT route's own: where the noise ball is
        # tight its second multiplier saturates. See PrimalKKTSolver.
        assert primal == pytest.approx(dual, rel=1e-5)
        assert exact == pytest.approx(primal, rel=5e-3)
