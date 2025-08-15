"""
Tests for the random_matrix module.

Since these methods are randomized and approximate, the tests focus on
verifying the mathematical properties of the outputs (e.g., orthonormality)
and ensuring that the approximation error is within a reasonable tolerance
compared to exact, deterministic methods.
"""
import pytest
import numpy as np
from scipy.linalg import svd
from pygeoinf.random_matrix import (
    fixed_rank_random_range,
    random_svd,
    random_eig,
    random_cholesky,
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
    U = np.random.randn(m, n)
    V = np.random.randn(n, n)
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
    A = X.T @ X
    return A


# =============================================================================
# Tests for Randomized Range Finder
# =============================================================================

def test_fixed_rank_random_range_properties():
    """
    Tests the properties of the output of fixed_rank_random_range.
    """
    m, n = 50, 30
    A = np.random.randn(m, n)
    rank = 10

    Q = fixed_rank_random_range(A, rank)

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
    optimal_reconstruction = U_true[:, :rank] @ np.diag(s_true[:rank]) @ Vh_true[:rank, :]
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
    optimal_reconstruction = U_true[:, :rank] @ np.diag(s_true[:rank]) @ U_true[:, :rank].T
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
    optimal_reconstruction = U_true[:, :rank] @ np.diag(s_true[:rank]) @ U_true[:, :rank].T
    optimal_error = np.linalg.norm(A - optimal_reconstruction)

    # The Cholesky error can be larger, so we allow more slack.
    assert reconstruction_error < 20 * optimal_error
