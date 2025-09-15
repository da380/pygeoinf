"""
Tests for the random_matrix module.

Since these methods are randomized and approximate, the tests focus on
verifying the mathematical properties of the outputs (e.g., orthonormality)
and ensuring that the approximation error is within a reasonable tolerance
compared to exact, deterministic methods.
"""

import pytest
import numpy as np
from scipy.linalg import svd, eigh
from scipy.sparse import spdiags

# MODIFIED: Import new functions to be tested
from pygeoinf.random_matrix import (
    fixed_rank_random_range,
    variable_rank_random_range,
    random_range,
    random_svd,
    random_eig,
    random_cholesky,
    random_diagonal,
)

# =============================================================================
# Fixtures for Test Matrices
# =============================================================================


@pytest.fixture
def rectangular_matrix() -> np.ndarray:
    """
    Provides a rectangular test matrix with a known, rapid singular value decay.
    This makes it a good candidate for low-rank approximation.
    """
    m, n = 100, 50
    U, _ = np.linalg.qr(np.random.randn(m, n))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    # Create singular values that decay exponentially
    s = 2.0 ** (-np.arange(n))
    S = np.diag(s)
    # Construct the matrix A = U @ S @ V.T
    A = U @ S @ V.T
    return A


@pytest.fixture
def symmetric_matrix() -> np.ndarray:
    """
    Provides a symmetric, positive-definite test matrix.
    """
    n = 50
    # Create a random matrix and make it symmetric positive-definite
    X = np.random.randn(n, n)
    A = X.T @ X + 1e-6 * np.eye(n)  # Add small identity to ensure full rank
    return A


@pytest.fixture
def symmetric_semidefinite_matrix() -> np.ndarray:
    """
    Provides a symmetric, positive SEMI-definite test matrix.
    This matrix is singular (rank-deficient) and will cause the standard
    cho_factor to fail, triggering the eigendecomposition fallback.
    """
    n = 50
    rank = 30  # Matrix will be n x n, but with rank < n
    X = np.random.randn(n, rank)
    A = X @ X.T  # A is now positive semi-definite and singular
    return A


# NEW: Fixture for diagonal estimation test
@pytest.fixture
def matrix_with_known_diagonal():
    """
    Provides a sparse square matrix with a clearly defined diagonal.
    """
    n_dim = 100
    # The true diagonal is just a simple sequence
    true_diag = np.arange(1, n_dim + 1, dtype=float)

    # Add some off-diagonal noise to make the problem non-trivial
    np.random.seed(42)
    num_off_diagonals = 3
    offsets = np.random.choice(
        np.arange(-n_dim + 1, n_dim), num_off_diagonals, replace=False
    )
    data = np.random.randn(num_off_diagonals, n_dim)

    # Combine the off-diagonal noise and the true diagonal
    all_data = np.vstack([data, true_diag])
    all_offsets = np.append(offsets, 0)

    matrix = spdiags(all_data, all_offsets, n_dim, n_dim, format="csr")

    return matrix, true_diag


# =============================================================================
# Tests for Randomized Range Finder
# =============================================================================


@pytest.mark.parametrize("parallel_flag", [False, True])
def test_fixed_rank_random_range_properties(parallel_flag):
    """
    Tests the properties of the output of fixed_rank_random_range.
    """
    m, n = 50, 30
    A = np.random.randn(m, n)
    rank = 10

    Q = fixed_rank_random_range(A, rank, parallel=parallel_flag)

    # 1. Test output shape
    assert Q.shape == (m, rank)

    # 2. Test for orthonormality of the columns
    # Q.T @ Q should be very close to the identity matrix.
    identity = np.eye(rank)
    assert np.allclose(Q.T @ Q, identity)


def test_fixed_rank_random_range_accuracy(rectangular_matrix):
    """
    Tests that the random range finder captures the action of the matrix.
    """
    A = rectangular_matrix
    rank = 10

    # Compute the approximate basis
    Q = fixed_rank_random_range(A, rank, power=2)

    # The error of the approximation is ||A - Q @ Q.T @ A||.
    # This should be close to the (rank+1)-th singular value.
    reconstruction_error = np.linalg.norm(A - Q @ (Q.T @ A), 2)

    # For comparison, compute the optimal error from a deterministic SVD
    _, s, _ = svd(A, full_matrices=False)
    optimal_error = s[rank]  # The first neglected singular value

    # The randomized method should be close to the optimal one.
    # We allow for a generous slack factor to prevent random failures.
    assert reconstruction_error < 10 * optimal_error


# NEW: Tests for the variable-rank range finder and its wrapper
@pytest.mark.parametrize("parallel_flag", [False, True])
def test_variable_rank_random_range_properties(rectangular_matrix, parallel_flag):
    """
    Tests the properties of the variable-rank range finder.
    """
    A = rectangular_matrix
    Q = variable_rank_random_range(A, 5, rtol=1e-3, parallel=parallel_flag)

    # 1. Test for orthonormality
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]))

    # 2. Test if the determined rank is reasonable
    # For the given rtol, the rank should be around 10 since s[10] ~ 1e-3
    assert 8 <= Q.shape[1] <= 15


def test_variable_rank_max_rank_warning(rectangular_matrix):
    """
    Tests that the variable-rank finder stops at max_rank and issues a warning.
    """
    A = rectangular_matrix
    # Set max_rank too low to meet the tolerance
    with pytest.warns(UserWarning, match="Tolerance .* not met"):
        Q = variable_rank_random_range(A, 5, max_rank=8, rtol=1e-5)

    # Check that the output rank is exactly max_rank
    assert Q.shape[1] == 8


def test_random_range_wrapper(rectangular_matrix):
    """
    Tests the `random_range` wrapper function.
    """
    A = rectangular_matrix

    # 1. Test 'fixed' method dispatch
    Q_fixed = random_range(A, 12, method="fixed")
    assert Q_fixed.shape == (A.shape[0], 12)

    # 2. Test 'variable' method dispatch
    Q_var = random_range(A, 5, method="variable", rtol=1e-2)
    assert Q_var.shape[0] == A.shape[0]
    assert Q_var.shape[1] > 5  # Should find more columns

    # 3. Test invalid method
    with pytest.raises(ValueError, match="Unknown method"):
        random_range(A, 10, method="invalid_method")


# =============================================================================
# Tests for Randomized Factorizations
# =============================================================================


def test_random_svd(rectangular_matrix):
    """
    Tests the randomized SVD factorization.
    """
    A = rectangular_matrix
    rank = 10

    Q = fixed_rank_random_range(A, rank, power=2)
    U, s, Vh = random_svd(A, Q)

    # 1. Test reconstruction error
    reconstruction = U @ np.diag(s) @ Vh
    reconstruction_error = np.linalg.norm(A - reconstruction)

    # Compare to the error of the optimal rank-k approximation from SVD
    U_true, s_true, Vh_true = svd(A, full_matrices=False)
    optimal_reconstruction = (
        U_true[:, :rank] @ np.diag(s_true[:rank]) @ Vh_true[:rank, :]
    )
    optimal_error = np.linalg.norm(A - optimal_reconstruction)

    assert reconstruction_error < 10 * optimal_error

    # 2. Test orthogonality of factors
    assert np.allclose(U.T @ U, np.eye(rank))
    assert np.allclose(Vh @ Vh.T, np.eye(rank))


def test_random_eig(symmetric_matrix):
    """
    Tests the randomized eigendecomposition for a symmetric matrix.
    """
    A = symmetric_matrix
    rank = 10

    Q = fixed_rank_random_range(A, rank, power=2)
    U, s = random_eig(A, Q)

    # 1. Test reconstruction error
    reconstruction = U @ np.diag(s) @ U.T
    reconstruction_error = np.linalg.norm(A - reconstruction)

    # Compare to optimal error from deterministic eigendecomposition
    s_true, U_true = np.linalg.eigh(A)
    # Eigenvalues are sorted ascending, so we take the largest ones
    s_true = s_true[::-1]
    U_true = U_true[:, ::-1]
    optimal_reconstruction = (
        U_true[:, :rank] @ np.diag(s_true[:rank]) @ U_true[:, :rank].T
    )
    optimal_error = np.linalg.norm(A - optimal_reconstruction)

    assert reconstruction_error < 10 * optimal_error

    # 2. Test orthogonality of eigenvectors
    assert np.allclose(U.T @ U, np.eye(rank))


def test_random_cholesky(symmetric_matrix):
    """
    Tests the randomized Cholesky factorization.
    """
    A = symmetric_matrix
    rank = 10

    Q = fixed_rank_random_range(A, rank, power=2)
    F = random_cholesky(A, Q)

    # 1. Test reconstruction error
    reconstruction = F @ F.T
    reconstruction_error = np.linalg.norm(A - reconstruction)

    # Compare to optimal error from eigendecomposition
    s_true, U_true = np.linalg.eigh(A)
    s_true = s_true[::-1]
    U_true = U_true[:, ::-1]
    optimal_reconstruction = (
        U_true[:, :rank] @ np.diag(s_true[:rank]) @ U_true[:, :rank].T
    )
    optimal_error = np.linalg.norm(A - optimal_reconstruction)

    # The Cholesky error can be larger, so we allow more slack.
    assert reconstruction_error < 20 * optimal_error


def test_random_cholesky_fallback_path(symmetric_semidefinite_matrix):
    """
    Tests the 'fallback path' (eigendecomposition) of random_cholesky.
    This test uses a semi-definite matrix which forces the `except` block
    to be executed, verifying its correctness.
    """
    A = symmetric_semidefinite_matrix
    rank = 25  # Choose a rank for the approximation

    Q = fixed_rank_random_range(A, rank, power=2)

    # This call should succeed by falling back to the eigendecomposition method
    F = random_cholesky(A, Q)

    # Check that the reconstruction error is reasonable
    reconstruction = F @ F.T
    reconstruction_error = np.linalg.norm(A - reconstruction)

    # Compare to the optimal error from a deterministic eigendecomposition
    s_true, U_true = eigh(A)
    s_true, U_true = s_true[::-1], U_true[:, ::-1]
    optimal_reconstruction = (
        U_true[:, :rank] @ np.diag(s_true[:rank]) @ U_true[:, :rank].T
    )
    optimal_error = np.linalg.norm(A - optimal_reconstruction)

    # The error should be close to the best possible low-rank approximation
    assert reconstruction_error < 1.5 * optimal_error + 1e-9


@pytest.mark.parametrize("parallel_flag", [False, True])
@pytest.mark.parametrize("use_rademacher", [True, False])
def test_random_diagonal_accuracy_and_properties(
    matrix_with_known_diagonal, parallel_flag, use_rademacher
):
    """
    Tests the accuracy and output properties of the random_diagonal function.
    """
    A, true_diag = matrix_with_known_diagonal
    n = A.shape[0]

    estimated_diag = random_diagonal(
        A,
        100,
        max_samples=500,  # More samples for a more stable test
        rtol=1e-3,
        use_rademacher=use_rademacher,
        parallel=parallel_flag,
    )

    # 1. Test output shape
    assert estimated_diag.shape == (n,)

    # 2. Test accuracy
    relative_error = np.linalg.norm(estimated_diag - true_diag) / np.linalg.norm(
        true_diag
    )

    # Assert that the relative error is reasonably small (e.g., < 5%)
    # This tolerance is heuristic and may need adjustment, but is a good starting point.
    assert relative_error < 0.05


def test_random_diagonal_max_samples_warning(matrix_with_known_diagonal):
    """
    Tests that random_diagonal issues a warning if max_samples is reached
    before the desired tolerance is met.
    """
    A, _ = matrix_with_known_diagonal

    with pytest.warns(UserWarning, match="Tolerance .* not met"):
        random_diagonal(
            A,
            10,
            max_samples=20,  # Set a low limit to force a stop
            rtol=1e-12,  # Set an impossible tolerance
        )
