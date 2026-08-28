"""Solvers: declared preconditions, coordinate freedom, and correctness."""

import numpy as np
import pytest

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.numerics import (
    BiCGStabSolver,
    CGSolver,
    CholeskySolver,
    ConvergenceError,
    EigenSolver,
    IdentityPreconditioner,
    InverseOperator,
    JacobiPreconditioner,
    LSQRSolver,
    LUSolver,
    MinResSolver,
)
from pygeoinf2.testing import check_operator, check_traits
from pygeoinf2.traits import Traits

from .conftest import DenseMetricSpace, make_weighted_space
from .doubles import NoCoordinatesError, StrictSpace

N = 12


def spd(rng, n=N):
    root = rng.normal(size=(n, n))
    return root @ root.T + n * np.identity(n)


@pytest.fixture
def spd_problem(rng):
    X = EuclideanSpace(N)
    matrix = spd(rng)
    A = LinearOperator.from_matrix(
        X, X, matrix, traits=Traits.POSITIVE_DEFINITE, form="components"
    )
    b = rng.normal(size=N)
    return A, b, np.linalg.solve(matrix, b)


SQUARE_SOLVERS = [
    CGSolver,
    MinResSolver,
    BiCGStabSolver,
    CholeskySolver,
    LUSolver,
    EigenSolver,
]


class TestCorrectness:
    @pytest.mark.parametrize("solver_class", SQUARE_SOLVERS)
    def test_solves_a_definite_system(self, solver_class, spd_problem, rng):
        A, b, exact = spd_problem
        result = solver_class()(A).solve(b)
        assert np.allclose(result.solution, exact, atol=1e-8)
        assert result.converged

    def test_minres_handles_an_indefinite_system(self, rng):
        """Where CG cannot go, and is not allowed to try."""
        X = EuclideanSpace(N)
        matrix = rng.normal(size=(N, N))
        matrix = matrix + matrix.T
        A = LinearOperator.from_matrix(
            X, X, matrix, traits=Traits.SELF_ADJOINT, form="components"
        )
        b = rng.normal(size=N)
        result = MinResSolver(rtol=1e-12)(A).solve(b)
        assert np.allclose(result.solution, np.linalg.solve(matrix, b), atol=1e-8)

    def test_bicgstab_handles_a_nonsymmetric_system(self, rng):
        X = EuclideanSpace(N)
        matrix = rng.normal(size=(N, N)) + N * np.identity(N)
        A = LinearOperator.from_matrix(X, X, matrix, form="components")
        b = rng.normal(size=N)
        result = BiCGStabSolver(rtol=1e-12)(A).solve(b)
        assert np.allclose(result.solution, np.linalg.solve(matrix, b), atol=1e-8)

    def test_solving_on_a_weighted_space(self, rng):
        """The metric must not leak into the answer."""
        X = make_weighted_space()
        n = X.dim
        matrix = spd(rng, n)
        A = LinearOperator.self_adjoint(
            X,
            lambda x: X.from_components(X.solve_gram(matrix @ X.to_components(x))),
            traits=Traits.POSITIVE_DEFINITE,
        )
        check_traits(A, rng=rng)
        b = X.random(rng=rng)
        solution = CGSolver(rtol=1e-12)(A).solve(b).solution
        assert np.allclose(X.to_components(A(solution)), X.to_components(b), atol=1e-8)


class TestDeclaredPreconditions:
    """v1 writes `assert operator.is_automorphism`, which -O deletes."""

    def test_cg_refuses_an_operator_that_has_not_earned_it(self, rng):
        X = EuclideanSpace(N)
        A = LinearOperator.from_matrix(
            X, X, spd(rng), form="components"
        )  # no traits claimed
        with pytest.raises(ValueError, match="POSITIVE_DEFINITE"):
            CGSolver()(A)

    def test_the_message_names_what_is_missing(self, rng):
        X = EuclideanSpace(N)
        A = LinearOperator.from_matrix(
            X, X, spd(rng), traits=Traits.SELF_ADJOINT, form="components"
        )
        with pytest.raises(ValueError, match="missing"):
            CGSolver()(A)

    def test_minres_accepts_mere_self_adjointness(self, rng):
        X = EuclideanSpace(N)
        A = LinearOperator.from_matrix(
            X, X, spd(rng), traits=Traits.SELF_ADJOINT, form="components"
        )
        assert isinstance(MinResSolver()(A), InverseOperator)

    def test_rectangular_systems_are_refused_with_a_pointer(self, rng):
        X, Y = EuclideanSpace(N), EuclideanSpace(5)
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(5, N)), form="components")
        with pytest.raises(ValueError, match="LeastSquaresSolver"):
            LUSolver()(A)

    def test_a_false_definiteness_claim_is_caught_during_the_solve(self, rng):
        """CG detects the lie itself, and says how to check the claim."""
        X = EuclideanSpace(N)
        matrix = rng.normal(size=(N, N))
        matrix = -(matrix @ matrix.T) - N * np.identity(N)
        liar = LinearOperator.from_matrix(
            X, X, matrix, traits=Traits.POSITIVE_DEFINITE, form="components"
        )
        with pytest.raises(ConvergenceError, match="check_traits"):
            CGSolver()(liar).solve(rng.normal(size=N))


class TestCoordinateFreedom:
    """The iterative solvers must never reach for a component map."""

    @pytest.fixture
    def strict_problem(self, rng):
        base = make_weighted_space()
        strict = StrictSpace(base)
        n = base.dim
        matrix = spd(rng, n)
        A = LinearOperator.self_adjoint(
            strict,
            lambda x: base.from_components(
                base.solve_gram(matrix @ base.to_components(x))
            ),
            traits=Traits.POSITIVE_DEFINITE,
        )
        return strict, A, strict.random(rng=rng)

    def test_cg_is_coordinate_free(self, strict_problem):
        strict, A, b = strict_problem
        result = CGSolver(rtol=1e-12)(A).solve(b)
        assert result.converged
        assert strict.norm(strict.subtract(A(result.solution), b)) < 1e-8

    def test_minres_is_coordinate_free(self, strict_problem):
        strict, A, b = strict_problem
        assert MinResSolver(rtol=1e-12)(A).solve(b).converged

    def test_bicgstab_is_coordinate_free(self, strict_problem):
        strict, A, b = strict_problem
        assert BiCGStabSolver(rtol=1e-12)(A).solve(b).converged

    def test_lsqr_is_coordinate_free(self, strict_problem):
        strict, A, b = strict_problem
        assert LSQRSolver(rtol=1e-12)(A).solve(b).converged

    def test_the_direct_solvers_do_need_coordinates(self, strict_problem):
        """The negative control: this split is real, not decorative."""
        _, A, _ = strict_problem
        with pytest.raises(NoCoordinatesError):
            CholeskySolver()(A)


class TestInverseOperator:
    def test_traits_propagate_through_inversion(self, spd_problem, rng):
        A, _, _ = spd_problem
        inverse = CholeskySolver()(A)
        assert Traits.POSITIVE_DEFINITE & inverse.traits
        check_traits(inverse, rng=rng)
        check_operator(inverse, rng=rng)

    def test_the_inverse_inverts(self, spd_problem, rng):
        A, _, _ = spd_problem
        inverse = CholeskySolver()(A)
        x = A.domain.random(rng=rng)
        assert np.allclose(inverse(A(x)), x, atol=1e-8)

    def test_the_adjoint_of_the_inverse_is_the_inverse_of_the_adjoint(self, rng):
        X = EuclideanSpace(N)
        matrix = rng.normal(size=(N, N)) + N * np.identity(N)
        A = LinearOperator.from_matrix(X, X, matrix, form="components")
        inverse = LUSolver()(A)
        check_operator(inverse, rng=rng)
        y = X.random(rng=rng)
        assert np.allclose(inverse.adjoint(y), np.linalg.solve(matrix.T, y), atol=1e-8)

    def test_it_composes_into_the_algebra(self, spd_problem, rng):
        A, b, exact = spd_problem
        inverse = CholeskySolver()(A)
        assert np.allclose((inverse @ A)(A.domain.random(rng=rng)) is not None, True)
        identity_like = inverse @ A
        x = A.domain.random(rng=rng)
        assert np.allclose(identity_like(x), x, atol=1e-8)

    def test_solvers_are_stateless(self, rng):
        """One solver, two operators, no interference."""
        X = EuclideanSpace(N)
        solver = CGSolver(rtol=1e-12)
        results = []
        for _ in range(2):
            matrix = spd(rng)
            A = LinearOperator.from_matrix(
                X, X, matrix, traits=Traits.POSITIVE_DEFINITE, form="components"
            )
            b = rng.normal(size=N)
            result = solver(A).solve(b)
            results.append((result, np.linalg.solve(matrix, b)))
        for result, exact in results:
            assert np.allclose(result.solution, exact, atol=1e-8)


class TestConvergenceReporting:
    def test_failure_raises_by_default(self, spd_problem):
        A, b, _ = spd_problem
        with pytest.raises(ConvergenceError, match="did not converge"):
            CGSolver(rtol=1e-16, maxiter=1)(A).solve(b)

    def test_failure_can_be_downgraded(self, spd_problem):
        A, b, _ = spd_problem
        with pytest.warns(RuntimeWarning, match="did not converge"):
            result = CGSolver(rtol=1e-16, maxiter=1, strict=False)(A).solve(b)
        assert not result.converged

    def test_diagnostics_come_back_with_the_answer(self, spd_problem):
        A, b, _ = spd_problem
        result = CGSolver()(A).solve(b)
        assert result.iterations > 0
        assert result.residual_norm >= 0.0
        assert "iterations" in repr(result)

    def test_a_warm_start_costs_fewer_iterations(self, spd_problem):
        A, b, exact = spd_problem
        inverse = CGSolver(rtol=1e-10)(A)
        cold = inverse.solve(b)
        warm = inverse.solve(b, x0=A.domain.from_components(exact * 0.999))
        assert warm.iterations < cold.iterations

    @pytest.mark.parametrize("solver_class", [CGSolver, MinResSolver, BiCGStabSolver])
    def test_every_iterative_solver_reports_its_history(
        self, solver_class, spd_problem
    ):
        """The class docstring promised a callback for all of them; MINRES,
        BiCGStab and LSQR never called one and came back with an empty
        history, so a non-convergence in those three left no trail."""
        A, b, _ = spd_problem
        seen = []
        result = solver_class(rtol=1e-10, callback=lambda i, r: seen.append((i, r)))(
            A
        ).solve(b)

        assert result.converged
        assert len(result.history) > 1
        assert len(seen) == len(result.history)
        assert [residual for _, residual in seen] == list(result.history)
        # The residual is being driven down, not merely recorded.
        assert result.history[-1] < result.history[0]

    def test_lsqr_reports_its_history(self, rng):
        A = LinearOperator.from_matrix(
            EuclideanSpace(12),
            EuclideanSpace(20),
            rng.normal(size=(20, 12)),
            form="components",
        )
        seen = []
        result = LSQRSolver(rtol=1e-12, callback=lambda i, r: seen.append(i))(A).solve(
            rng.normal(size=20)
        )
        assert len(result.history) > 1
        assert len(seen) == len(result.history)


class TestLSQRWarmStart:
    """``x0`` was accepted and silently dropped."""

    @pytest.fixture
    def least_squares(self, rng):
        domain, codomain = EuclideanSpace(20), EuclideanSpace(30)
        operator = LinearOperator.from_matrix(
            domain, codomain, rng.normal(size=(30, 20)), form="components"
        )
        return operator, rng.normal(size=30)

    def test_starting_from_the_answer_costs_one_iteration(self, least_squares):
        """It cost a full cold solve before, because ``solve_fn`` never passed
        ``x0`` down. Both halves are needed: the start has to reach the
        iteration, and the tolerances have to be relative to the data rather
        than to the shifted residual, which for a warm start is already ~0."""
        operator, data = least_squares
        inverse = LSQRSolver(rtol=1e-12)(operator)
        cold = inverse.solve(data)
        warm = inverse.solve(data, x0=cold.solution)

        assert cold.iterations > 5
        assert warm.iterations == 1
        assert (
            operator.domain.norm(operator.domain.subtract(warm.solution, cold.solution))
            < 1e-10
        )

    def test_a_damped_warm_start_is_refused(self, least_squares):
        """Shifting moves the penalty onto the correction, which minimises
        something else. v1 does it anyway; this says so instead."""
        operator, data = least_squares
        inverse = LSQRSolver(damping=0.5)(operator)
        with pytest.raises(ValueError, match="cannot be warm-started"):
            inverse.solve(data, x0=operator.domain.random())


class TestPreconditioning:
    def test_identity_preconditioner_changes_nothing(self, spd_problem):
        A, b, exact = spd_problem
        result = CGSolver(preconditioner=IdentityPreconditioner())(A).solve(b)
        assert np.allclose(result.solution, exact, atol=1e-8)

    def test_jacobi_helps_a_badly_scaled_system(self, rng):
        X = EuclideanSpace(N)
        scales = np.logspace(0, 5, N)
        matrix = np.diag(scales) + 0.01 * np.identity(N)
        A = LinearOperator.from_matrix(
            X, X, matrix, traits=Traits.POSITIVE_DEFINITE, form="components"
        )
        b = rng.normal(size=N)
        plain = CGSolver(rtol=1e-10)(A).solve(b)
        preconditioned = CGSolver(rtol=1e-10, preconditioner=JacobiPreconditioner())(
            A
        ).solve(b)
        assert preconditioned.iterations < plain.iterations
        assert np.allclose(
            preconditioned.solution, np.linalg.solve(matrix, b), atol=1e-6
        )

    def test_a_ready_made_operator_is_accepted(self, spd_problem, rng):
        A, b, exact = spd_problem
        approximate_inverse = CholeskySolver()(A)
        result = CGSolver(preconditioner=approximate_inverse)(A).solve(b)
        assert np.allclose(result.solution, exact, atol=1e-8)
        assert result.iterations <= 2

    def test_jacobi_needs_coordinates(self, rng):
        base = make_weighted_space()
        strict = StrictSpace(base)
        A = LinearOperator.self_adjoint(
            strict, lambda x: base.scale(2.0, x), traits=Traits.POSITIVE_DEFINITE
        )
        with pytest.raises(NoCoordinatesError):
            JacobiPreconditioner()(A)


class TestLeastSquares:
    @pytest.mark.parametrize("shape", [(7, 12), (12, 7), (9, 9)])
    def test_matches_dense_least_squares(self, shape, rng):
        m, n = shape
        X, Y = EuclideanSpace(n), EuclideanSpace(m)
        matrix = rng.normal(size=(m, n))
        b = rng.normal(size=m)
        A = LinearOperator.from_matrix(X, Y, matrix, form="components")
        result = LSQRSolver(rtol=1e-13)(A).solve(b)
        assert np.allclose(
            result.solution, np.linalg.lstsq(matrix, b, rcond=None)[0], atol=1e-8
        )

    def test_damping_solves_the_regularised_problem(self, rng):
        n, m, damping = 12, 7, 0.7
        X, Y = EuclideanSpace(n), EuclideanSpace(m)
        matrix = rng.normal(size=(m, n))
        b = rng.normal(size=m)
        A = LinearOperator.from_matrix(X, Y, matrix, form="components")
        result = LSQRSolver(damping=damping, rtol=1e-13)(A).solve(b)
        expected = np.linalg.solve(
            matrix.T @ matrix + damping**2 * np.identity(n), matrix.T @ b
        )
        assert np.allclose(result.solution, expected, atol=1e-8)

    def test_the_pseudo_inverse_claims_no_invertibility(self, rng):
        X, Y = EuclideanSpace(12), EuclideanSpace(7)
        A = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(7, 12)), form="components"
        )
        assert LSQRSolver()(A).traits == Traits.NONE

    def test_negative_damping_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            LSQRSolver(damping=-1.0)


class TestDirectInverseAdjoint:
    """The adjoint of a factorised inverse must reuse the factorisation."""

    def test_it_does_not_refactorise(self, rng):
        """``inv.adjoint`` inverted ``A*`` from scratch: a second matrix
        extraction and a second factorisation of what is, up to a transpose,
        the same matrix. And ``_adjoint_value`` went through a *second* cache,
        so both paths built their own.
        """
        space = EuclideanSpace(50)
        matrix = rng.normal(size=(50, 50)) + 50.0 * np.identity(50)
        applications = 0

        def value(x):
            nonlocal applications
            applications += 1
            return matrix @ x

        def adjoint(y):
            nonlocal applications
            applications += 1
            return matrix.T @ y

        operator = LinearOperator.from_callables(space, space, value, adjoint=adjoint)
        inverse = LUSolver()(operator)

        applications = 0
        probe = rng.normal(size=50)
        recovered = inverse.adjoint(probe)
        assert applications == 0
        # Second path, which used to build its own inverse.
        inverse.adjoint_inverse(probe)
        assert applications == 0

        # And it is the right operator: A* (A*)^-1 x == x.
        applications = 0
        assert operator.adjoint(recovered) == pytest.approx(probe)

    def test_it_is_right_on_a_weighted_space(self, rng):
        """The transposed solve carries the metric, and that is where a sign or
        a Gram in the wrong place would show."""
        space = make_weighted_space()
        size = space.dim
        matrix = rng.normal(size=(size, size)) + size * np.identity(size)
        operator = LinearOperator.from_matrix(space, space, matrix, form="components")
        inverse = LUSolver()(operator)

        probe = space.random(rng=rng)
        recovered = operator.adjoint(inverse.adjoint(probe))
        assert space.norm(space.subtract(recovered, probe)) < 1e-10 * space.norm(probe)


class TestProgressCallback:
    """The diagnostic an inversion otherwise discards."""

    def test_it_records_the_last_solve(self, spd_problem):
        from pygeoinf2.numerics.solvers import ProgressCallback

        A, b, _ = spd_problem
        progress = ProgressCallback()
        result = CGSolver(rtol=1e-10, callback=progress)(A).solve(b)

        assert progress.iterations == result.iterations
        assert progress.residual == pytest.approx(result.residual_norm)
        assert list(progress.residuals) == list(result.history)

    def test_it_resets_between_solves(self, spd_problem, rng):
        """An estimator solves more than once -- at construction and at every
        application -- so the counts have to belong to the last one rather than
        accumulating."""
        from pygeoinf2.numerics.solvers import ProgressCallback

        A, b, _ = spd_problem
        progress = ProgressCallback()
        inverse = CGSolver(rtol=1e-10, callback=progress)(A)
        inverse.solve(b)
        first = progress.iterations
        inverse.solve(A.domain.random(rng=rng))
        assert progress.iterations > 0
        assert len(progress.residuals) == progress.iterations + 1
        assert first > 0

    def test_it_reports_only_when_asked(self, spd_problem):
        """A library that writes to stdout uninvited is a nuisance inside a
        loop, so nothing is printed without a sink."""
        from pygeoinf2.numerics.solvers import ProgressCallback

        A, b, _ = spd_problem
        lines = []
        CGSolver(rtol=1e-10, callback=ProgressCallback(report=lines.append))(A).solve(b)
        assert lines
        assert lines[0].startswith("iteration 0:")

    def test_it_survives_an_estimator(self, rng):
        """The workflow it exists for: pyslfp prints the iteration count after
        every solve, and ``est(data)`` discards the SolveResult."""
        import pygeoinf2 as gi

        from pygeoinf2.numerics.solvers import ProgressCallback

        model, data_space = EuclideanSpace(30), EuclideanSpace(15)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.normal(size=(15, 30)), form="components"
        )
        problem = gi.LinearForwardProblem(
            forward, error=gi.GaussianMeasure.from_standard_deviation(data_space, 0.1)
        )
        prior = gi.GaussianMeasure.from_standard_deviation(model, 1.0)
        progress = ProgressCallback()
        estimator = gi.LinearGaussianInversion(
            problem, prior, solver=CGSolver(rtol=1e-10, callback=progress)
        )
        _, observed = problem.synthetic_model_and_data(prior, rng=rng)
        estimator(observed)
        assert progress.iterations > 0


class TestPreconditionedMinRes:
    """MINRES with a preconditioner, which used to raise at solve time.

    The class advertised preconditioning through its base and then refused it
    once a solve was under way, which is the worst place to find out.
    """

    @staticmethod
    def indefinite(space, rng, spread=300.0):
        """Self-adjoint with eigenvalues of both signs, which is the case
        MINRES exists for and CG cannot touch."""
        n = space.dim
        basis, _ = np.linalg.qr(rng.standard_normal((n, n)))
        eigenvalues = np.concatenate(
            [
                np.linspace(1.0, spread, n - n // 3),
                -np.linspace(1.0, spread / 4.0, n // 3),
            ]
        )
        matrix = basis @ np.diag(eigenvalues) @ basis.T
        return LinearOperator.from_matrix(
            space,
            space,
            0.5 * (matrix + matrix.T),
            form="galerkin",
            traits=Traits.SELF_ADJOINT,
        )

    def test_it_solves_an_indefinite_system(self, rng):
        space = EuclideanSpace(60)
        operator = self.indefinite(space, rng)
        wanted = space.random(rng=rng)

        result = (
            MinResSolver(rtol=1e-10, maxiter=3000)
            .with_preconditioner(JacobiPreconditioner())(operator)
            .solve(operator(wanted))
        )
        assert result.converged
        assert space.norm(
            space.subtract(result.solution, wanted)
        ) < 1e-8 * space.norm(wanted)

    def test_it_is_what_makes_a_dense_metric_solvable(self, rng):
        """The measurement that says the preconditioning is real rather than
        merely accepted: on a non-diagonal Gram matrix the plain method does
        not converge in 3000 iterations, at a relative error of 4.6e-2, and
        Jacobi gets there in 88."""
        space = DenseMetricSpace(
            (lambda root: root @ root.T)(
                np.tril(rng.standard_normal((40, 40))) + 2.0 * np.eye(40)
            )
        )
        operator = self.indefinite(space, rng)
        wanted = space.random(rng=rng)
        right_hand_side = operator(wanted)

        plain = MinResSolver(rtol=1e-10, maxiter=3000, strict=False)(operator).solve(
            right_hand_side
        )
        preconditioned = (
            MinResSolver(rtol=1e-10, maxiter=3000)
            .with_preconditioner(JacobiPreconditioner())(operator)
            .solve(right_hand_side)
        )

        assert not plain.converged
        assert preconditioned.converged
        assert preconditioned.iterations < plain.iterations / 10
        assert space.norm(
            space.subtract(preconditioned.solution, wanted)
        ) < 1e-6 * space.norm(wanted)

    def test_the_unpreconditioned_path_is_untouched(self, rng):
        """Passing no preconditioner must run the same recurrences it always
        did: with M the identity the M-norm is the norm."""
        space = EuclideanSpace(N)
        operator = LinearOperator.from_matrix(
            space, space, spd(rng), form="galerkin", traits=Traits.POSITIVE_DEFINITE
        )
        wanted = space.random(rng=rng)
        result = MinResSolver(rtol=1e-12)(operator).solve(operator(wanted))
        assert result.converged
        assert space.norm(space.subtract(result.solution, wanted)) < 1e-9

    def test_an_indefinite_preconditioner_is_refused(self, rng):
        """It runs in the inner product M induces, so an indefinite M has no
        inner product to offer. Said plainly, rather than as a NaN twenty
        iterations later."""
        space = EuclideanSpace(20)
        operator = self.indefinite(space, rng)
        flipped = LinearOperator.self_adjoint(
            space, lambda x: -x, traits=Traits.SELF_ADJOINT
        )

        solver = MinResSolver(maxiter=50).with_preconditioner(flipped)
        with pytest.raises(ValueError, match="positive-definite"):
            solver(operator).solve(space.random(rng=rng))


class TestPreconditionersStaySparse:
    """Neither of these may form the dense matrix.

    Both did, which defeats the point of them: a sparse preconditioner exists
    for the case where the dense matrix does not fit, and forming it to decide
    which entries to keep gives that away. v1 probed a matrix-free operator
    column by column for exactly this reason.
    """

    @staticmethod
    def banded(n, rng):
        """Applied matrix-free, so nothing dense exists unless someone makes
        it. Tridiagonal and diagonally dominant, hence positive definite."""
        diagonal = np.full(n, 4.0)
        off = rng.uniform(0.2, 0.5, n - 1)

        def apply(c):
            out = diagonal * c
            out[:-1] += off * c[1:]
            out[1:] += off * c[:-1]
            return out

        return LinearOperator.self_adjoint(
            EuclideanSpace(n), apply, traits=Traits.POSITIVE_DEFINITE
        )

    @staticmethod
    def build(which, n):
        from pygeoinf2.numerics.preconditioners import (
            BlockPreconditioner,
            ColumnThresholdedPreconditioner,
        )

        if which == "block":
            return BlockPreconditioner(
                [list(range(i, min(i + 20, n))) for i in range(0, n, 20)]
            )
        return ColumnThresholdedPreconditioner(0.1)

    @pytest.mark.parametrize("which", ["block", "thresholded"])
    def test_the_memory_grows_with_the_dimension_not_its_square(self, which, rng):
        """The claim that matters, and the one a fixed threshold cannot test:
        four times the dimension at a fixed number of entries per column must
        cost four times the memory, not sixteen. Measured at 500 and 2000, both
        come out at 4.0 and 3.8.

        Watching the allocations, not reading the code."""
        import tracemalloc

        peaks = []
        for n in (500, 2000):
            tracemalloc.start()
            self.build(which, n)(self.banded(n, rng))
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks.append(peak)

        growth = peaks[1] / peaks[0]
        assert growth < 8.0  # comfortably below the 16 a dense matrix would give
        assert peaks[1] < 2000 * 2000 * 8 / 4

    @pytest.mark.parametrize("which", ["block", "thresholded"])
    def test_it_never_forms_the_dense_matrix(self, which, rng, monkeypatch):
        """Named directly: the dense route is not merely avoided by accident,
        it is not taken. Both classes used to call it."""
        from pygeoinf2.algebra.operators import LinearOperator as Operator

        def refuse(*args, **kwargs):
            raise AssertionError("the dense matrix was formed")

        monkeypatch.setattr(Operator, "matrix", refuse)
        self.build(which, 200)(self.banded(200, rng))

    @pytest.mark.parametrize("which", ["block", "thresholded"])
    def test_and_it_still_preconditions(self, which, rng):
        """Sparse and useless would be no improvement."""
        n = 200
        operator = self.banded(n, rng)
        space = operator.domain
        wanted = space.random(rng=rng)
        preconditioner = self.build(which, n)

        result = (
            CGSolver(rtol=1e-10)
            .with_preconditioner(preconditioner)(operator)
            .solve(operator(wanted))
        )
        assert result.converged
        assert space.norm(space.subtract(result.solution, wanted)) < 1e-8


class TestJacobiCanEstimate:
    """v1 estimated the diagonal from 20 probes by default. v2 could only read
    it exactly, at one operator application per component -- which on a large
    matrix-free operator is the whole reason someone reached for a
    preconditioner in the first place."""

    @staticmethod
    def problem(rng, n=400):
        matrix = rng.standard_normal((n, n))
        matrix = matrix @ matrix.T / n + np.diag(rng.uniform(1.0, 20.0, n))
        space = EuclideanSpace(n)
        # Matrix-free, so nothing can read the diagonal off a stored array.
        operator = LinearOperator.self_adjoint(
            space, lambda c: matrix @ c, traits=Traits.POSITIVE_DEFINITE
        )
        return space, operator

    def test_an_estimate_preconditions_about_as_well(self, rng):
        """Measured: 20 probes builds in 0.6 ms against 12.1 exact, and costs
        14 CG iterations against 13. Unpreconditioned is 38."""
        space, operator = self.problem(rng)
        wanted = space.random(rng=rng)
        right_hand_side = operator(wanted)

        counts = {}
        for label, preconditioner in [
            ("exact", JacobiPreconditioner()),
            ("estimated", JacobiPreconditioner(samples=20, rng=np.random.default_rng(1))),
        ]:
            result = (
                CGSolver(rtol=1e-10, maxiter=2000)
                .with_preconditioner(preconditioner)(operator)
                .solve(right_hand_side)
            )
            assert result.converged
            counts[label] = result.iterations

        plain = CGSolver(rtol=1e-10, maxiter=2000)(operator).solve(right_hand_side)
        assert counts["estimated"] <= counts["exact"] + 5
        assert counts["estimated"] < plain.iterations / 2

    def test_it_costs_fewer_applications(self, rng):
        """The point of it. Exact is one application per component."""
        space, operator = self.problem(rng)
        counted = {"n": 0}
        original = operator._value

        def counting(x):
            counted["n"] += 1
            return original(x)

        object.__setattr__(operator, "_value", counting)

        JacobiPreconditioner(samples=20, rng=np.random.default_rng(1))(operator)
        assert counted["n"] <= 20

    def test_no_samples_at_all_is_refused(self):
        with pytest.raises(ValueError, match="At least one sample"):
            JacobiPreconditioner(samples=0)


class TestPreconditionerReplacement:
    """Returning the solver unchanged is not the same thing as attaching."""

    def test_replacing_one_is_refused(self, rng):
        space = EuclideanSpace(8)
        solver = CGSolver(preconditioner=LinearOperator.identity(space))
        with pytest.raises(ValueError, match="already has a preconditioner"):
            solver.with_preconditioner(JacobiPreconditioner())

    def test_attaching_to_a_bare_solver_still_works(self, rng):
        space = EuclideanSpace(8)
        attached = CGSolver().with_preconditioner(JacobiPreconditioner())
        assert isinstance(attached.preconditioner, JacobiPreconditioner)
        assert CGSolver().preconditioner is None


class TestSquareMeansTheSameSpace:
    """Matching dimensions are not enough: every iterative method here adds
    the iterate to the residual, so the two must be vectors of one space. Two
    spaces of equal dimension over the same vectors but different metrics
    would give a plausible wrong answer rather than an error."""

    def test_a_metric_mismatch_is_refused(self, rng):
        space = EuclideanSpace(4)
        other = make_weighted_space()
        assert space.dim == other.dim

        operator = LinearOperator.from_matrix(
            space, other, np.eye(4), form="components"
        )
        with pytest.raises(ValueError, match="not the same space"):
            CGSolver()(operator)


class TestWoodburyInnerDefault:
    """A strict inner solver aborts the outer solve, and inner precision does
    not show in the answer: a preconditioner is an approximation."""

    def test_the_default_inner_solver_is_loose_and_forgiving(self):
        from pygeoinf2.numerics.preconditioners import WoodburyPreconditioner

        space = EuclideanSpace(6)
        identity = LinearOperator.identity(space)
        preconditioner = WoodburyPreconditioner(identity, identity, identity)

        inner = preconditioner._inner_solver()
        assert not inner._strict
        assert inner._rtol >= 1e-4
