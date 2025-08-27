"""
Tests for the linear_solvers module.
"""
import pytest
import numpy as np
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.operators import LinearOperator, DiagonalLinearOperator
from pygeoinf.linear_solvers import (
    LUSolver,
    CholeskySolver,
    CGMatrixSolver,
    BICGMatrixSolver,
    BICGStabMatrixSolver,
    GMRESMatrixSolver,
    CGSolver,
)

# =============================================================================
# Fixtures for the Test Problem
# =============================================================================


@pytest.fixture
def space() -> EuclideanSpace:
    """Provides a simple 10D Euclidean space for the tests."""
    return EuclideanSpace(dim=10)


@pytest.fixture
def spd_operator(space: EuclideanSpace) -> LinearOperator:
    """
    Provides a well-conditioned, invertible, symmetric positive-definite
    linear operator for the tests.
    """
    matrix = np.random.randn(space.dim, space.dim)
    spd_matrix = matrix.T @ matrix + 0.1 * np.eye(space.dim)
    return LinearOperator.from_matrix(space, space, spd_matrix, galerkin=True)


@pytest.fixture
def non_symmetric_operator(space: EuclideanSpace) -> LinearOperator:
    """
    Provides a well-conditioned, invertible, non-symmetric operator.
    This fixture is constructed to ensure the matrix is not ill-conditioned,
    which can cause issues for sensitive iterative solvers.
    """
    # Create a random matrix
    matrix = np.random.randn(space.dim, space.dim)
    # Scale it to have a spectral radius of ~1
    matrix = matrix / np.linalg.norm(matrix, 2)
    # Add the identity to shift eigenvalues away from zero, ensuring it's
    # well-conditioned and invertible.
    well_conditioned_matrix = matrix + np.eye(space.dim)
    return LinearOperator.from_matrix(space, space, well_conditioned_matrix, galerkin=True)


@pytest.fixture
def action_defined_operator(space: EuclideanSpace, spd_operator) -> LinearOperator:
    """Provides a matrix-free operator defined only by its action."""
    # We can use the spd_operator's action and adjoint for this
    return LinearOperator(
        space, space, spd_operator, adjoint_mapping=spd_operator.adjoint
    )


@pytest.fixture
def x(space: EuclideanSpace) -> np.ndarray:
    """Provides a random vector from the space."""
    return space.random()


# =============================================================================
# Parametrized Tests for Different Solver Types
# =============================================================================

# Define a tolerance for the iterative solvers to aim for.
ITERATIVE_SOLVER_TOLERANCE = 1e-6
# Define a milder tolerance for the tests to check against.
TEST_TOLERANCE = 10 * ITERATIVE_SOLVER_TOLERANCE

# Direct solvers should be accurate to near machine precision.
direct_solvers = [
    LUSolver(galerkin=True),
    CholeskySolver(galerkin=True),
]

# Solvers that require or are optimized for symmetric positive-definite matrices
spd_iterative_solvers = [
    CGMatrixSolver(galerkin=True, rtol=ITERATIVE_SOLVER_TOLERANCE),
    CGSolver(rtol=ITERATIVE_SOLVER_TOLERANCE),
]

# Solvers that can handle general non-symmetric matrices
general_iterative_solvers = [
    BICGMatrixSolver(galerkin=True, rtol=ITERATIVE_SOLVER_TOLERANCE),
    BICGStabMatrixSolver(galerkin=True, rtol=ITERATIVE_SOLVER_TOLERANCE),
    GMRESMatrixSolver(galerkin=True, rtol=ITERATIVE_SOLVER_TOLERANCE),
]

preconditionable_solvers = [
    CGMatrixSolver(galerkin=True, rtol=ITERATIVE_SOLVER_TOLERANCE),
    CGSolver(rtol=ITERATIVE_SOLVER_TOLERANCE),
]


@pytest.mark.parametrize("solver", direct_solvers)
def test_direct_solvers(solver, spd_operator: LinearOperator, x: np.ndarray):
    """Tests direct solvers with a tight tolerance."""
    identity = spd_operator.domain.identity_operator()
    inverse_operator = solver(spd_operator)
    result_vector = (inverse_operator @ spd_operator)(x)
    expected_vector = identity(x)
    assert np.allclose(result_vector, expected_vector, atol=1e-12)


@pytest.mark.parametrize("solver", spd_iterative_solvers)
def test_spd_iterative_solvers(solver, spd_operator: LinearOperator, x: np.ndarray):
    """Tests iterative solvers for SPD operators with a relaxed tolerance."""
    identity = spd_operator.domain.identity_operator()
    inverse_operator = solver(spd_operator)
    result_vector = (inverse_operator @ spd_operator)(x)
    expected_vector = identity(x)
    assert np.allclose(
        result_vector,
        expected_vector,
        rtol=TEST_TOLERANCE,
        atol=TEST_TOLERANCE,
    )


@pytest.mark.parametrize("solver", general_iterative_solvers)
def test_general_iterative_solvers(
    solver, non_symmetric_operator: LinearOperator, x: np.ndarray
):
    """Tests general iterative solvers with a relaxed tolerance."""
    identity = non_symmetric_operator.domain.identity_operator()
    inverse_operator = solver(non_symmetric_operator)
    result_vector = (inverse_operator @ non_symmetric_operator)(x)
    expected_vector = identity(x)
    assert np.allclose(
        result_vector,
        expected_vector,
        rtol=TEST_TOLERANCE,
        atol=TEST_TOLERANCE,
    )


def test_action_defined_operator_solver(
    action_defined_operator: LinearOperator, x: np.ndarray
):
    """Tests a matrix-free solver with a matrix-free operator."""
    solver = CGSolver(rtol=ITERATIVE_SOLVER_TOLERANCE)
    identity = action_defined_operator.domain.identity_operator()
    inverse_operator = solver(action_defined_operator)
    result_vector = (inverse_operator @ action_defined_operator)(x)
    expected_vector = identity(x)
    assert np.allclose(
        result_vector,
        expected_vector,
        rtol=TEST_TOLERANCE,
        atol=TEST_TOLERANCE,
    )


@pytest.mark.parametrize("solver", preconditionable_solvers)
def test_preconditioned_solve(solver, spd_operator: LinearOperator, x: np.ndarray):
    """
    Tests iterative solvers with a preconditioner, checking both the
    forward and the adjoint solve paths.
    """
    space = spd_operator.domain
    
    # Create a simple Jacobi (diagonal) preconditioner
    diag_A = spd_operator.matrix(dense=True,galerkin=True).diagonal()
    preconditioner = DiagonalLinearOperator(space, space, 1.0 / diag_A, galerkin=True)

    # Get the inverse operator using the preconditioner
    inverse_op = solver(spd_operator, preconditioner=preconditioner)

    # --- Test 1: Primal solve: (A^-1 @ A) @ x = x ---
    result_primal = (inverse_op @ spd_operator)(x)
    assert np.allclose(result_primal, x, rtol=TEST_TOLERANCE, atol=TEST_TOLERANCE)

    # --- Test 2: Adjoint solve: ((A*)^-1 @ A*) @ x = x ---
    # Since spd_operator is self-adjoint, A* = A.
    # The inverse_op.adjoint property will call solve_adjoint_linear_system,
    # which is where the bug fix was.
    inverse_op_adj = inverse_op.adjoint
    result_adjoint = (inverse_op_adj @ spd_operator.adjoint)(x)
    assert np.allclose(result_adjoint, x, rtol=TEST_TOLERANCE, atol=TEST_TOLERANCE)