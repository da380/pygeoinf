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

from .conftest import make_weighted_space
from .doubles import NoCoordinatesError, StrictSpace

N = 12


def spd(rng, n=N):
    root = rng.normal(size=(n, n))
    return root @ root.T + n * np.identity(n)


@pytest.fixture
def spd_problem(rng):
    X = EuclideanSpace(N)
    matrix = spd(rng)
    A = LinearOperator.from_component_matrix(
        X, X, matrix, traits=Traits.POSITIVE_DEFINITE
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
        A = LinearOperator.from_component_matrix(
            X, X, matrix, traits=Traits.SELF_ADJOINT
        )
        b = rng.normal(size=N)
        result = MinResSolver(rtol=1e-12)(A).solve(b)
        assert np.allclose(result.solution, np.linalg.solve(matrix, b), atol=1e-8)

    def test_bicgstab_handles_a_nonsymmetric_system(self, rng):
        X = EuclideanSpace(N)
        matrix = rng.normal(size=(N, N)) + N * np.identity(N)
        A = LinearOperator.from_component_matrix(X, X, matrix)
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
        A = LinearOperator.from_component_matrix(X, X, spd(rng))  # no traits claimed
        with pytest.raises(ValueError, match="POSITIVE_DEFINITE"):
            CGSolver()(A)

    def test_the_message_names_what_is_missing(self, rng):
        X = EuclideanSpace(N)
        A = LinearOperator.from_component_matrix(
            X, X, spd(rng), traits=Traits.SELF_ADJOINT
        )
        with pytest.raises(ValueError, match="missing"):
            CGSolver()(A)

    def test_minres_accepts_mere_self_adjointness(self, rng):
        X = EuclideanSpace(N)
        A = LinearOperator.from_component_matrix(
            X, X, spd(rng), traits=Traits.SELF_ADJOINT
        )
        assert isinstance(MinResSolver()(A), InverseOperator)

    def test_rectangular_systems_are_refused_with_a_pointer(self, rng):
        X, Y = EuclideanSpace(N), EuclideanSpace(5)
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(5, N)))
        with pytest.raises(ValueError, match="LeastSquaresSolver"):
            LUSolver()(A)

    def test_a_false_definiteness_claim_is_caught_during_the_solve(self, rng):
        """CG detects the lie itself, and says how to check the claim."""
        X = EuclideanSpace(N)
        matrix = rng.normal(size=(N, N))
        matrix = -(matrix @ matrix.T) - N * np.identity(N)
        liar = LinearOperator.from_component_matrix(
            X, X, matrix, traits=Traits.POSITIVE_DEFINITE
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
        A = LinearOperator.from_component_matrix(X, X, matrix)
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
            A = LinearOperator.from_component_matrix(
                X, X, matrix, traits=Traits.POSITIVE_DEFINITE
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

    @pytest.mark.parametrize(
        "solver_class", [CGSolver, MinResSolver, BiCGStabSolver]
    )
    def test_every_iterative_solver_reports_its_history(
        self, solver_class, spd_problem
    ):
        """The class docstring promised a callback for all of them; MINRES,
        BiCGStab and LSQR never called one and came back with an empty
        history, so a non-convergence in those three left no trail."""
        A, b, _ = spd_problem
        seen = []
        result = solver_class(
            rtol=1e-10, callback=lambda i, r: seen.append((i, r))
        )(A).solve(b)

        assert result.converged
        assert len(result.history) > 1
        assert len(seen) == len(result.history)
        assert [residual for _, residual in seen] == list(result.history)
        # The residual is being driven down, not merely recorded.
        assert result.history[-1] < result.history[0]

    def test_lsqr_reports_its_history(self, rng):
        A = LinearOperator.from_component_matrix(
            EuclideanSpace(12), EuclideanSpace(20), rng.normal(size=(20, 12))
        )
        seen = []
        result = LSQRSolver(rtol=1e-12, callback=lambda i, r: seen.append(i))(
            A
        ).solve(rng.normal(size=20))
        assert len(result.history) > 1
        assert len(seen) == len(result.history)


class TestLSQRWarmStart:
    """``x0`` was accepted and silently dropped."""

    @pytest.fixture
    def least_squares(self, rng):
        domain, codomain = EuclideanSpace(20), EuclideanSpace(30)
        operator = LinearOperator.from_component_matrix(
            domain, codomain, rng.normal(size=(30, 20))
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
        assert operator.domain.norm(
            operator.domain.subtract(warm.solution, cold.solution)
        ) < 1e-10

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
        A = LinearOperator.from_component_matrix(
            X, X, matrix, traits=Traits.POSITIVE_DEFINITE
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
        A = LinearOperator.from_component_matrix(X, Y, matrix)
        result = LSQRSolver(rtol=1e-13)(A).solve(b)
        assert np.allclose(
            result.solution, np.linalg.lstsq(matrix, b, rcond=None)[0], atol=1e-8
        )

    def test_damping_solves_the_regularised_problem(self, rng):
        n, m, damping = 12, 7, 0.7
        X, Y = EuclideanSpace(n), EuclideanSpace(m)
        matrix = rng.normal(size=(m, n))
        b = rng.normal(size=m)
        A = LinearOperator.from_component_matrix(X, Y, matrix)
        result = LSQRSolver(damping=damping, rtol=1e-13)(A).solve(b)
        expected = np.linalg.solve(
            matrix.T @ matrix + damping**2 * np.identity(n), matrix.T @ b
        )
        assert np.allclose(result.solution, expected, atol=1e-8)

    def test_the_pseudo_inverse_claims_no_invertibility(self, rng):
        X, Y = EuclideanSpace(12), EuclideanSpace(7)
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(7, 12)))
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

        operator = LinearOperator.from_callables(
            space, space, value, adjoint=adjoint
        )
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
        operator = LinearOperator.from_matrix(
            space, space, matrix, form="components"
        )
        inverse = LUSolver()(operator)

        probe = space.random(rng=rng)
        recovered = operator.adjoint(inverse.adjoint(probe))
        assert space.norm(space.subtract(recovered, probe)) < 1e-10 * space.norm(probe)
