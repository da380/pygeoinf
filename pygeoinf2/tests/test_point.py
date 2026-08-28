"""
The point estimators, the damped family, and the root find underneath them.

DESIGN.md §18.6 named one numerical kernel with four users — a damped solve
inside a monotone scalar root find — and the three things worth testing about
it are the three that are exact rather than matters of degree:

* the saturated cases, where no root exists and the *endpoint* is the answer.
  Getting the wrong end is how a discrepancy search returns the most
  structured model for data that support no structure;
* the identity between a Tikhonov normal operator and a Gaussian one with an
  isotropic prior, which is exact and so must hold to machine precision;
* the derivative of the discrepancy solution, against central differences and
  against its own adjoint.

See DESIGN.md section 24.
"""

import numpy as np
import pytest

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.geometry.subspaces import AffineSubspace
from pygeoinf2.inference import (
    ConstrainedLeastSquares,
    ConstrainedMinimumNorm,
    DiscrepancyPrinciple,
    LeastSquares,
    LinearForwardProblem,
    LinearGaussianInversion,
    MinimumNorm,
    TikhonovFamily,
    TikhonovNormalOperator,
)
from pygeoinf2.inference.preconditioners import NormalDiagonalPreconditioner
from pygeoinf2.numerics.preconditioners import JacobiPreconditioner
from pygeoinf2.numerics.root_find import (
    DampedSolves,
    Evaluation,
    monotone_root,
)
from pygeoinf2.numerics.solvers import CGSolver, CholeskySolver
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.traits import Traits

from .conftest import make_dense_metric_space, make_weighted_space


def constant(value):
    """A probe that ignores its multiplier, for the saturated cases."""
    return lambda multiplier, previous: Evaluation(value)


class TestMonotoneRoot:
    """One kernel, four users, and the endpoints that used to be wrong."""

    @pytest.mark.parametrize(
        "quantity, target, decreasing, expected",
        [
            (lambda t: 1.0 / t, 1.0, True, 1.0),
            (lambda t: 1.0 / t, 0.01, True, 100.0),
            (lambda t: t, 7.0, False, 7.0),
            (lambda t: t**2, 9.0, False, 3.0),
        ],
    )
    def test_it_finds_the_root_either_way_round(
        self, quantity, target, decreasing, expected
    ):
        result = monotone_root(
            lambda t, _: Evaluation(quantity(t)), target, decreasing=decreasing
        )
        assert result.converged
        assert result.argument == pytest.approx(expected, rel=1e-4)

    @pytest.mark.parametrize("decreasing", [True, False])
    def test_a_quantity_that_never_reaches_the_target_saturates_high(self, decreasing):
        """For an increasing misfit this is 'every damping fits', and the
        answer must be the largest — the smallest model. Returning the other
        end gives the most structured model for data supporting none."""
        value = 1e6 if decreasing else 0.0
        result = monotone_root(
            constant(value), 15.0, decreasing=decreasing, expansions=20
        )
        assert not result.converged
        assert result.exhausted == "high"
        assert result.argument > 1e10

    @pytest.mark.parametrize("decreasing", [True, False])
    def test_a_quantity_already_past_the_target_saturates_low(self, decreasing):
        value = 0.0 if decreasing else 1e6
        result = monotone_root(
            constant(value), 15.0, decreasing=decreasing, expansions=20
        )
        assert not result.converged
        assert result.exhausted == "low"
        assert result.argument < 1e-10

    def test_running_out_of_iterations_is_not_convergence(self):
        """The bracket tolerance is the claim; the iteration cap is not.

        Exhausting the loop reported ``converged=True`` regardless: with one
        iteration and zero tolerances it claimed a root while the bracket was
        still 6.8 wide, around an argument of 5.62 against a true root of
        3.333. A discrepancy sweep reading that flag would take a damping it
        had not actually found.
        """
        for iterations in (1, 2, 3):
            result = monotone_root(
                lambda t, _: Evaluation(1.0 / t),
                0.3,
                iterations=iterations,
                rtol=0.0,
                atol=0.0,
            )
            low, high = result.bracket
            assert high - low > 1.0
            assert not result.converged

        # With a reachable tolerance it still converges, and to the right root.
        found = monotone_root(lambda t, _: Evaluation(1.0 / t), 0.3, iterations=60)
        assert found.converged
        assert found.argument == pytest.approx(10.0 / 3.0, rel=1e-4)

    def test_the_previous_solution_reaches_the_next_probe(self):
        seen = []

        def evaluate(multiplier, previous):
            seen.append(previous)
            return Evaluation(1.0 / multiplier, solution=multiplier, iterations=3)

        result = monotone_root(evaluate, 1.0, iterations=4)
        assert seen[0] is None
        assert all(entry is not None for entry in seen[1:])
        assert result.inner_iterations == 3 * result.evaluations
        assert result.warm_started

    def test_warm_starting_can_be_turned_off(self):
        seen = []

        def evaluate(multiplier, previous):
            seen.append(previous)
            return Evaluation(1.0 / multiplier, solution=multiplier)

        result = monotone_root(evaluate, 1.0, iterations=4, warm_start=False)
        assert all(entry is None for entry in seen)
        assert not result.warm_started

    def test_bad_arguments_are_refused(self):
        with pytest.raises(ValueError, match="must be positive"):
            monotone_root(constant(1.0), 1.0, initial=0.0)
        with pytest.raises(ValueError, match="[Aa]t least one"):
            monotone_root(constant(1.0), 1.0, iterations=0)


class TestDampedSolves:
    """The family a sweep walks along, and what it keeps between steps."""

    @pytest.fixture
    def family(self, rng):
        space = EuclideanSpace(30)
        root = rng.normal(size=(30, 30))
        base = LinearOperator.from_derivative_matrix(
            space,
            space,
            root @ root.T,
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE,
        )
        return space, DampedSolves(
            base,
            LinearOperator.identity(space),
            CGSolver(rtol=1e-12, maxiter=5000).with_preconditioner(
                JacobiPreconditioner()
            ),
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )

    def test_it_solves_what_it_says_it_does(self, family, rng):
        space, solves = family
        for damping in (0.5, 5.0):
            vector = space.random(rng=rng)
            result = solves.solve(damping, vector)
            back = solves.operator(damping)(result.solution)
            assert space.norm(space.subtract(back, vector)) == pytest.approx(
                0.0, abs=1e-8 * space.norm(vector)
            )

    def test_a_nearby_multiplier_reuses_the_preconditioner(self, family, rng):
        """Otherwise it is rebuilt against every member of the family, which
        for an expensive preconditioner costs more than the solves it is
        accelerating."""
        space, solves = family
        built = []
        original = JacobiPreconditioner._invert

        def counting(self, operator):
            built.append(operator)
            return original(self, operator)

        JacobiPreconditioner._invert = counting
        try:
            vector = space.random(rng=rng)
            for damping in (1.0, 1.5, 2.0, 3.0):
                solves.solve(damping, vector)
            within = len(built)
            # Far enough away, and it must be rebuilt.
            solves.solve(1.0e6, vector)
            beyond = len(built)
        finally:
            JacobiPreconditioner._invert = original
        assert within == 1
        assert beyond == 2

    def test_a_direct_solver_reports_no_iterations(self, family, rng):
        space, _ = family
        root = rng.normal(size=(30, 30))
        solves = DampedSolves(
            LinearOperator.from_derivative_matrix(
                space, space, root @ root.T, traits=Traits.SELF_ADJOINT
            ),
            LinearOperator.identity(space),
            CholeskySolver(),
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )
        assert solves.solve(1.0, space.random(rng=rng)).iterations == 0


@pytest.fixture(params=["euclidean", "weighted"])
def setup(request, rng):
    """A forward problem, over- or under-determined, with a metric or without."""
    model = (
        EuclideanSpace(12) if request.param == "euclidean" else make_weighted_space()
    )
    data = EuclideanSpace(8)
    forward = LinearOperator.from_derivative_matrix(
        model, data, rng.normal(size=(data.dim, model.dim))
    )
    variances = rng.uniform(0.01, 0.04, data.dim)
    covariance = LinearOperator.from_derivative_matrix(
        data,
        data,
        np.diag(variances),
        traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
    )
    error = GaussianMeasure(
        data,
        covariance=covariance,
        covariance_factor=LinearOperator.from_derivative_matrix(
            data, data, np.diag(np.sqrt(variances))
        ),
        precision=CholeskySolver()(covariance),
    )
    return LinearForwardProblem(forward, error=error)


class TestTikhonovNormalOperator:
    """N(t), and the identity it is kept separate from."""

    def test_it_equals_the_assembly_it_stands_for(self, setup, rng):
        problem = setup
        forward = problem.forward_operator
        data = problem.data_space
        normal = TikhonovNormalOperator(
            forward, 2.0, error=problem.error_measure, formalism="data_space"
        )
        reference = forward @ forward.adjoint + 2.0 * problem.error_measure.covariance
        for _ in range(5):
            vector = data.random(rng=rng)
            assert data.norm(
                data.subtract(normal(vector), reference(vector))
            ) == pytest.approx(0.0, abs=1e-10 * data.norm(vector))

    @pytest.mark.parametrize("formalism", ["model_space", "data_space"])
    def test_tikhonov_is_the_gaussian_case_with_an_isotropic_prior(
        self, setup, formalism, rng
    ):
        """Exact, not approximate: the two factors of 1/t cancel between the
        gain and the operator. The classes are kept apart because one is a
        family and the other an assembly, not because the identity is doubtful
        — so the identity is asserted, at machine precision."""
        problem = setup
        model = problem.model_space
        damping = 3.0
        isotropic = GaussianMeasure(
            model,
            covariance=(1.0 / damping) * LinearOperator.identity(model),
            precision=damping * LinearOperator.identity(model),
        )
        bayesian = LinearGaussianInversion(
            problem, isotropic, formalism=formalism, solver=CholeskySolver()
        )
        tikhonov = LeastSquares(problem, damping=damping, formalism=formalism)
        for _ in range(5):
            observed = problem.data_space.random(rng=rng)
            expected = bayesian(observed).expectation
            assert model.norm(
                model.subtract(tikhonov(observed), expected)
            ) == pytest.approx(0.0, abs=1e-8 * model.norm(expected))

    def test_zero_damping_has_no_prior_reading(self, setup):
        problem = setup
        normal = TikhonovNormalOperator(
            problem.forward_operator, 0.0, error=problem.error_measure
        )
        with pytest.raises(ValueError, match="no prior at all"):
            normal.prior_covariance

    def test_a_surrogate_must_share_the_data_space(self, setup, rng):
        problem = setup
        normal = TikhonovNormalOperator(
            problem.forward_operator, 1.0, error=problem.error_measure
        )
        other = EuclideanSpace(problem.data_space.dim + 2)
        cheap = LinearOperator.from_derivative_matrix(
            problem.model_space,
            other,
            rng.normal(size=(other.dim, problem.model_space.dim)),
        )
        with pytest.raises(ValueError, match="share the data space"):
            normal.surrogate(forward=cheap)


class TestLeastSquares:
    """What it inverts, and the ways of asking for the same thing."""

    def test_the_two_formalisms_agree(self, setup, rng):
        """An exact identity, so it is tested with a solver tight enough to
        show it: at the default tolerance each side carries its own solver
        error and their difference says more about ``rtol`` than about the
        algebra."""
        problem = setup
        model = problem.model_space
        estimator = LeastSquares(problem, damping=0.5, solver=CGSolver(rtol=1e-13))
        other = estimator.with_formalism(
            "model_space" if estimator.formalism == "data_space" else "data_space"
        )
        for _ in range(5):
            observed = problem.data_space.random(rng=rng)
            expected = estimator(observed)
            assert model.norm(
                model.subtract(other(observed), expected)
            ) == pytest.approx(0.0, abs=1e-8 * model.norm(expected))

    def test_the_normal_operator_is_the_one_being_solved(self, setup, rng):
        """It is exposed so that a preconditioner can be built against it; the
        check is that it really is the system the estimate comes out of."""
        problem = setup
        estimator = LeastSquares(problem, damping=0.7, formalism="data_space")
        normal = estimator.normal_operator
        observed = problem.data_space.random(rng=rng)
        solution = estimator.inverse_normal_operator(
            estimator.right_hand_side(observed)
        )
        assert problem.model_space.norm(
            problem.model_space.subtract(
                normal.model_from(solution), estimator(observed)
            )
        ) == pytest.approx(0.0, abs=1e-8)

    def test_a_preconditioned_solve_gives_the_same_answer(self, setup, rng):
        problem = setup
        model = problem.model_space
        estimator = LeastSquares(problem, damping=0.5, formalism="data_space")
        observed = problem.data_space.random(rng=rng)
        reference = estimator(observed)
        solved = estimator.with_solver(
            CGSolver(rtol=1e-12).with_preconditioner(JacobiPreconditioner())
        )(observed)
        assert model.norm(model.subtract(solved, reference)) == pytest.approx(
            0.0, abs=1e-8 * model.norm(reference)
        )

    def test_a_damping_sweep_costs_no_reassembly(self, setup, rng):
        """The L-curve shape: the norm falls and the misfit rises, monotonely,
        which is the property every damping search relies on."""
        problem = setup
        model = problem.model_space
        observed = problem.data_space.random(rng=rng)
        family = LeastSquares(problem).family()
        right_hand_side = family.right_hand_side(observed)
        norms, misfits = [], []
        for damping in np.geomspace(1e-2, 1e3, 12):
            estimate = family.model_from(
                family.solve(damping, right_hand_side).solution
            )
            norms.append(model.norm(estimate))
            misfits.append(problem.chi_squared(estimate, observed))
        assert np.all(np.diff(norms) < 1e-9)
        assert np.all(np.diff(misfits) > -1e-9)

    def test_with_damping_matches_a_fresh_estimator(self, setup, rng):
        problem = setup
        model = problem.model_space
        observed = problem.data_space.random(rng=rng)
        moved = LeastSquares(problem, damping=0.1).with_damping(2.5)
        fresh = LeastSquares(problem, damping=2.5)
        assert model.norm(
            model.subtract(moved(observed), fresh(observed))
        ) == pytest.approx(0.0, abs=1e-10)

    def test_the_residual_callback_reports_something_falling(self, setup, rng):
        problem = setup
        observed = problem.data_space.random(rng=rng)
        estimator = LeastSquares(problem, damping=0.5, formalism="data_space")
        seen = []
        callback = estimator.residual_callback(observed, report=seen.append)
        estimator.with_solver(CGSolver(rtol=1e-12, callback=callback))(observed)
        assert len(seen) > 1


class TestDiscrepancyPrinciple:
    """A damping found rather than chosen, and the derivative that follows."""

    def test_it_hits_the_misfit_target(self, setup, rng):
        problem = setup
        truth = problem.model_space.random(rng=rng)
        observed = problem.synthetic_data(truth, rng=rng)
        found = MinimumNorm(problem).for_data(observed, level=0.95)
        target = problem.critical_chi_squared(level=0.95)
        assert problem.chi_squared(found(observed), observed) == pytest.approx(
            target, rel=1e-3
        )

    def test_data_that_anything_fits_give_the_smallest_model(self, setup, rng):
        """The saturated case, and the one that was wrong: when the data are
        consistent with noise the model should say so, not reproduce the
        noise. The largest damping is the answer, not the smallest."""
        problem = setup
        model = problem.model_space
        observed = problem.data_space.random(rng=rng)
        negligible = problem.data_space.scale(1e-8, observed)
        found = MinimumNorm(problem).for_data(negligible)
        assert model.norm(found(negligible)) < 1e-8 * model.norm(
            MinimumNorm(problem, damping=1e-6)(negligible)
        )

    @pytest.mark.parametrize(
        "build_data", [lambda: EuclideanSpace(3), make_dense_metric_space]
    )
    def test_data_that_nothing_fits_are_refused_by_both_routes(self, build_data):
        """The failure case, which is not saturation.

        The forward operator reaches only the first coordinate, so data with
        weight elsewhere cannot be fitted at any damping. ``for_data`` used to
        answer anyway — damping 1e-200, chi-squared 5e7 against a target of
        9.5 — while ``DiscrepancyPrinciple`` raised on the same input. They now
        share one search and so cannot disagree.
        """
        data_space = build_data()
        model_space = EuclideanSpace(1)
        forward = LinearOperator.from_component_matrix(
            model_space, data_space, np.array([[1.0], [0.0], [0.0]])
        )
        problem = LinearForwardProblem(
            forward,
            error=GaussianMeasure.from_standard_deviation(data_space, 1e-3),
        )
        unfittable = data_space.from_components(np.array([0.0, 4.0, 4.0]))

        with pytest.raises(ValueError, match="cannot be fitted"):
            MinimumNorm(problem, damping=1.0).for_data(unfittable)
        with pytest.raises(ValueError, match="cannot be fitted"):
            DiscrepancyPrinciple(problem)(unfittable)

    def test_a_structure_aware_preconditioner_works_inside_the_sweep(
        self, setup, rng
    ):
        """DESIGN's claim that every structure-aware preconditioner applies to
        the point estimators held everywhere except where the sweep was the
        point.

        ``DampedSolves`` built ``base + t * shift``, a plain sum, and a sum has
        no factors for a preconditioner to read: the same solver that worked at
        a fixed damping raised ``TypeError`` inside the search. It now asks the
        family for its member, which arrives as a ``TikhonovNormalOperator``.
        """
        problem = setup
        solver = CGSolver(rtol=1e-10).with_preconditioner(
            JacobiPreconditioner()
        )
        structure_aware = CGSolver(rtol=1e-10).with_preconditioner(
            NormalDiagonalPreconditioner()
        )
        observed = problem.synthetic_data(problem.model_space.random(rng=rng), rng=rng)

        family = TikhonovFamily(
            problem.forward_operator,
            error=problem.error_measure,
            solver=structure_aware,
        )
        assert isinstance(family.at(1.0), TikhonovNormalOperator)
        assert family.solve(1.0, family.right_hand_side(observed)).converged

        # And end to end, which is the test the review asked for.
        found = DiscrepancyPrinciple(problem, solver=structure_aware)(observed)
        plain = DiscrepancyPrinciple(problem, solver=solver)(observed)
        assert problem.model_space.norm(
            problem.model_space.subtract(found, plain)
        ) < 1e-6 * problem.model_space.norm(plain)

    def test_the_search_reports_what_it_cost(self, setup, rng):
        problem = setup
        observed = problem.data_space.random(rng=rng)
        result = MinimumNorm(
            problem, solver=CGSolver(rtol=1e-12, maxiter=5000)
        ).discrepancy_search(observed)
        assert result.evaluations > 1
        assert result.inner_iterations > 0
        assert result.warm_started

    @pytest.mark.slow
    def test_warm_starting_costs_fewer_iterations(self, rng):
        """On a system large enough that conjugate gradients cannot simply
        exhaust its Krylov space, which a small one can and does."""
        model, data = EuclideanSpace(120), EuclideanSpace(90)
        left, _ = np.linalg.qr(rng.normal(size=(90, 90)))
        right, _ = np.linalg.qr(rng.normal(size=(120, 120)))
        values = np.zeros((90, 120))
        np.fill_diagonal(values, np.geomspace(1.0, 1e-3, 90))
        forward = LinearOperator.from_derivative_matrix(
            model, data, left @ values @ right.T
        )
        problem = LinearForwardProblem(
            forward, error=GaussianMeasure.from_standard_deviation(data, 0.01)
        )
        observed = problem.synthetic_data(model.random(rng=rng), rng=rng)
        solver = CGSolver(rtol=1e-10, maxiter=20000)
        estimator = MinimumNorm(problem, solver=solver)
        warm = estimator.discrepancy_search(observed)

        family = estimator.family()
        right_hand_side = family.right_hand_side(observed)

        def evaluate(damping, previous):
            result = family.solve(damping, right_hand_side, x0=None)
            return Evaluation(
                problem.chi_squared(family.model_from(result.solution), observed),
                result.solution,
                result.iterations,
            )

        cold = monotone_root(
            evaluate,
            problem.critical_chi_squared(level=0.95),
            decreasing=False,
            warm_start=False,
        )
        assert warm.inner_iterations < cold.inner_iterations
        assert warm.argument == pytest.approx(cold.argument, rel=1e-6)

    def test_the_derivative_matches_central_differences(self, setup, rng):
        """The search's own tolerance is the floor on this: the map is only as
        differentiable as the damping is converged, so the search is tightened
        far below the finite-difference step before comparing."""
        problem = setup
        model, data = problem.model_space, problem.data_space
        observed = problem.synthetic_data(model.random(rng=rng), rng=rng)
        # Two tolerances bound this, not one: the damping search's, and the
        # inner solve's. A direct solver removes the second so the test is
        # about the derivative rather than about conjugate gradients.
        method = DiscrepancyPrinciple(
            problem, rtol=1e-14, iterations=200, solver=CholeskySolver()
        )
        derivative = method.derivative(observed)
        for _ in range(3):
            direction = data.random(rng=rng)
            direction = data.scale(1.0 / data.norm(direction), direction)
            step = 1e-6 * data.norm(observed)
            forward_value = method(data.add(observed, data.scale(step, direction)))
            backward = method(data.subtract(observed, data.scale(step, direction)))
            difference = model.scale(
                1.0 / (2.0 * step), model.subtract(forward_value, backward)
            )
            exact = derivative(direction)
            assert model.norm(model.subtract(difference, exact)) == pytest.approx(
                0.0, abs=1e-6 * model.norm(exact)
            )

    def test_the_adjoint_is_the_adjoint(self, setup, rng):
        """A right formula with a wrong adjoint is the more likely error, and
        the one that stays invisible until something upstream calls it."""
        problem = setup
        model, data = problem.model_space, problem.data_space
        observed = problem.synthetic_data(model.random(rng=rng), rng=rng)
        derivative = DiscrepancyPrinciple(
            problem, solver=CGSolver(rtol=1e-13)
        ).derivative(observed)
        for _ in range(10):
            left, right = data.random(rng=rng), model.random(rng=rng)
            assert model.inner_product(derivative(left), right) == pytest.approx(
                data.inner_product(left, derivative.adjoint(right)),
                abs=1e-9 * data.norm(left) * model.norm(right),
            )

    def test_it_needs_an_error_measure(self, setup):
        problem = LinearForwardProblem(setup.forward_operator)
        with pytest.raises(ValueError, match="data error measure"):
            DiscrepancyPrinciple(problem)


class TestConstrained:
    """Least squares and minimum norm inside an affine subspace."""

    @pytest.fixture
    def constraint(self, setup, rng):
        model = setup.model_space
        operator = LinearOperator.from_derivative_matrix(
            model, EuclideanSpace(2), rng.normal(size=(2, model.dim))
        )
        value = operator.codomain.random(rng=rng)
        return AffineSubspace.from_linear_equation(operator, value)

    def test_least_squares_respects_the_constraint(self, setup, constraint, rng):
        observed = setup.data_space.random(rng=rng)
        estimate = ConstrainedLeastSquares(setup, constraint, damping=0.5)(observed)
        assert constraint.contains(estimate)

    @staticmethod
    def fittable(problem, constraint, rng):
        """Data generated by a model *in* the subspace, so the discrepancy
        principle has a solution. Data the constraint cannot reproduce make it
        inapplicable, which is a different test."""
        inside = constraint.projector(problem.model_space.random(rng=rng))
        inside = problem.model_space.add(inside, constraint.translation)
        return problem.synthetic_data(inside, rng=rng)

    def test_minimum_norm_respects_the_constraint(self, setup, constraint, rng):
        observed = self.fittable(setup, constraint, rng)
        estimate = ConstrainedMinimumNorm(setup, constraint)(observed)
        assert constraint.contains(estimate)

    def test_the_constrained_derivative_matches_differences(
        self, setup, constraint, rng
    ):
        model, data = setup.model_space, setup.data_space
        observed = self.fittable(setup, constraint, rng)
        method = ConstrainedMinimumNorm(
            setup, constraint, rtol=1e-14, iterations=200, solver=CholeskySolver()
        )
        derivative = method.derivative(observed)
        direction = data.random(rng=rng)
        direction = data.scale(1.0 / data.norm(direction), direction)
        step = 1e-6 * data.norm(observed)
        difference = model.scale(
            1.0 / (2.0 * step),
            model.subtract(
                method(data.add(observed, data.scale(step, direction))),
                method(data.subtract(observed, data.scale(step, direction))),
            ),
        )
        exact = derivative(direction)
        assert model.norm(model.subtract(difference, exact)) == pytest.approx(
            0.0, abs=1e-5 * model.norm(exact)
        )

    def test_the_constraint_value_mapping_moves_the_answer(
        self, setup, constraint, rng
    ):
        """How much does insisting on this cost, and what changes if I insist
        on something else."""
        observed = self.fittable(setup, constraint, rng)
        method = ConstrainedMinimumNorm(setup, constraint)
        mapping = method.constraint_value_mapping(observed)
        value = constraint.constraint_value
        estimate = mapping(value)
        assert constraint.constraint_operator(estimate) == pytest.approx(
            value, abs=1e-6
        )
        # A *small* move. The mapping is a sensitivity, and a large one takes
        # the subspace somewhere the data cannot be fitted at all — at which
        # point the discrepancy principle has no solution and says so, which is
        # a different test.
        moved = constraint.constraint_operator.codomain.scale(1.02, value)
        assert constraint.constraint_operator(mapping(moved)) == pytest.approx(
            moved, abs=1e-6
        )

    def test_the_constraint_value_derivative_matches_differences(
        self, setup, constraint, rng
    ):
        model = setup.model_space
        observed = self.fittable(setup, constraint, rng)
        method = ConstrainedMinimumNorm(
            setup, constraint, rtol=1e-14, iterations=200, solver=CholeskySolver()
        )
        mapping = method.constraint_value_mapping(observed)
        space = constraint.constraint_operator.codomain
        value = constraint.constraint_value
        derivative = mapping.derivative(value)
        direction = space.basis_vector(0)
        step = 1e-5 * max(space.norm(value), 1.0)
        difference = model.scale(
            1.0 / (2.0 * step),
            model.subtract(
                mapping(space.add(value, space.scale(step, direction))),
                mapping(space.subtract(value, space.scale(step, direction))),
            ),
        )
        exact = derivative(direction)
        assert model.norm(model.subtract(difference, exact)) == pytest.approx(
            0.0, abs=1e-4 * model.norm(exact)
        )

    def test_a_geometric_subspace_has_no_constraint_value(self, setup, rng):
        model = setup.model_space
        projector = AffineSubspace.from_linear_equation(
            LinearOperator.from_derivative_matrix(
                model, EuclideanSpace(2), rng.normal(size=(2, model.dim))
            ),
            EuclideanSpace(2).random(rng=rng),
        ).projector
        geometric = AffineSubspace(projector, translation=model.random(rng=rng))
        method = ConstrainedMinimumNorm(setup, geometric)
        with pytest.raises(ValueError, match="defined geometrically"):
            method.constraint_value_mapping(setup.data_space.random(rng=rng))


class TestTikhonovFamilyAccessors:
    """The damping-independent pieces the family hands back."""

    @pytest.fixture
    def family(self, setup):
        return setup, TikhonovFamily(
            setup.forward_operator,
            error=setup.error_measure,
            formalism="model_space",
        )

    def test_the_weighted_adjoint_is_the_templates(self, family):
        """It returned ``self._weighted``, an attribute never set, so every
        call raised ``AttributeError``. It does not depend on the damping, so
        it comes from the template like the other invariant pieces."""
        problem, tikhonov = family
        weighted = tikhonov.weighted_adjoint()

        assert weighted.domain == problem.data_space
        assert weighted.codomain == problem.model_space

    def test_the_weighted_adjoint_builds_the_right_hand_side(self, family, rng):
        """Its documented job: turn a shifted data vector into the RHS."""
        problem, tikhonov = family
        data = problem.data_space.random(rng=rng)
        shifted = problem.data_space.subtract(
            data, problem.error_measure.expectation
        )

        built = tikhonov.weighted_adjoint()(shifted)
        stated = tikhonov.right_hand_side(data)
        assert problem.model_space.norm(
            problem.model_space.subtract(built, stated)
        ) == pytest.approx(0.0, abs=1e-12)
