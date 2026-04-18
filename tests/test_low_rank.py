"""
Tests for the low_rank module.

Since these methods are randomized and approximate, the tests focus on
verifying the mathematical properties of the outputs (e.g., orthonormality)
and ensuring that the approximation error is within a reasonable tolerance
compared to exact, deterministic methods.
"""

import pytest
import numpy as np
from scipy.linalg import svd, eigh
from scipy.sparse import spdiags

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.low_rank import (
    random_range,
    LowRankSVD,
    LowRankEig,
    LowRankCholesky,
    random_diagonal,
    deflated_diagonal,
)
from pygeoinf.gaussian_measure import GaussianMeasure

# =============================================================================
# Fixtures for Test Operators
# =============================================================================


@pytest.fixture
def rectangular_matrix() -> np.ndarray:
    """Provides a rectangular test matrix with a known, rapid singular value decay."""
    m, n = 100, 50
    U, _ = np.linalg.qr(np.random.randn(m, n))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    s = 2.0 ** (-np.arange(n))
    S = np.diag(s)
    return U @ S @ V.T


@pytest.fixture
def rectangular_operator(rectangular_matrix) -> LinearOperator:
    m, n = rectangular_matrix.shape
    return LinearOperator.from_matrix(
        EuclideanSpace(n), EuclideanSpace(m), rectangular_matrix, galerkin=False
    )


@pytest.fixture
def symmetric_matrix() -> np.ndarray:
    """Provides a symmetric, positive-definite test matrix."""
    n = 50
    X = np.random.randn(n, n)
    return X.T @ X + 1e-6 * np.eye(n)


@pytest.fixture
def symmetric_operator(symmetric_matrix) -> LinearOperator:
    n = symmetric_matrix.shape[0]
    return LinearOperator.from_matrix(
        EuclideanSpace(n), EuclideanSpace(n), symmetric_matrix, galerkin=True
    )


@pytest.fixture
def symmetric_semidefinite_operator() -> LinearOperator:
    """Provides a rank-deficient symmetric operator to trigger Cholesky fallback."""
    n, rank = 50, 30
    X = np.random.randn(n, rank)
    A = X @ X.T
    return LinearOperator.from_matrix(
        EuclideanSpace(n), EuclideanSpace(n), A, galerkin=True
    )


@pytest.fixture
def matrix_with_known_diagonal():
    """Provides a sparse square matrix with a clearly defined diagonal."""
    n_dim = 100
    true_diag = np.arange(1, n_dim + 1, dtype=float)
    np.random.seed(42)
    num_off_diagonals = 3
    offsets = np.random.choice(
        np.arange(-n_dim + 1, n_dim), num_off_diagonals, replace=False
    )
    data = np.random.randn(num_off_diagonals, n_dim)
    all_data = np.vstack([data, true_diag])
    all_offsets = np.append(offsets, 0)
    matrix = spdiags(all_data, all_offsets, n_dim, n_dim, format="csr")
    return matrix, true_diag


# =============================================================================
# Tests for Randomized Range Finder
# =============================================================================


@pytest.mark.parametrize("parallel_flag", [False, True])
def test_random_range_fixed_properties(rectangular_operator, parallel_flag):
    """Tests the properties of the Q operator returned by fixed random_range."""
    rank = 10
    Q_op = random_range(
        rectangular_operator, rank, method="fixed", parallel=parallel_flag
    )

    # 1. Test output domain/codomain shapes
    assert Q_op.domain.dim == rank
    assert Q_op.codomain.dim == rectangular_operator.codomain.dim

    # 2. Test for orthonormality: Q* Q should be the identity operator
    identity = np.eye(rank)
    qq_star = (Q_op.adjoint @ Q_op).matrix(dense=True)
    assert np.allclose(qq_star, identity)


def test_random_range_variable_properties(rectangular_operator):
    """Tests the properties of the Q operator returned by variable random_range."""
    Q_op = random_range(rectangular_operator, 5, method="variable", rtol=1e-3)
    k = Q_op.domain.dim

    # 1. Test for orthonormality
    assert np.allclose((Q_op.adjoint @ Q_op).matrix(dense=True), np.eye(k))

    # 2. Test if the determined rank is reasonable for the given decay
    assert 8 <= k <= 15


def test_random_range_accuracy(rectangular_operator, rectangular_matrix):
    """Tests that the random range finder captures the action of the operator."""
    rank = 10
    Q_op = random_range(rectangular_operator, rank, method="fixed", power=2)
    Q_mat = Q_op.matrix(dense=True)

    # Error ||A - Q Q* A||
    reconstruction_error = np.linalg.norm(
        rectangular_matrix - Q_mat @ (Q_mat.T @ rectangular_matrix), 2
    )

    _, s, _ = svd(rectangular_matrix, full_matrices=False)
    optimal_error = s[rank]

    assert reconstruction_error < 10 * optimal_error


# =============================================================================
# Tests for Randomized Factorizations
# =============================================================================


def test_low_rank_svd(rectangular_operator, rectangular_matrix):
    """Tests the LowRankSVD operator factorization."""
    rank = 10
    svd_op = LowRankSVD.from_randomized(
        rectangular_operator, rank, method="fixed", power=2
    )

    # 1. Test reconstruction error
    reconstruction = svd_op.matrix(dense=True)
    reconstruction_error = np.linalg.norm(rectangular_matrix - reconstruction)

    U_true, s_true, Vh_true = svd(rectangular_matrix, full_matrices=False)
    optimal_reconstruction = (
        U_true[:, :rank] @ np.diag(s_true[:rank]) @ Vh_true[:rank, :]
    )
    optimal_error = np.linalg.norm(rectangular_matrix - optimal_reconstruction)

    assert reconstruction_error < 10 * optimal_error

    # 2. Test orthogonality of factors via operator algebra
    U_adj_U = (svd_op.u_factor.adjoint @ svd_op.u_factor).matrix(dense=True)
    V_adj_V = (svd_op.v_factor.adjoint @ svd_op.v_factor).matrix(dense=True)
    assert np.allclose(U_adj_U, np.eye(rank))
    assert np.allclose(V_adj_V, np.eye(rank))


def test_low_rank_eig(symmetric_operator, symmetric_matrix):
    """Tests the LowRankEig operator factorization."""
    rank = 10
    eig_op = LowRankEig.from_randomized(
        symmetric_operator, rank, method="fixed", power=2
    )

    # 1. Test reconstruction error
    reconstruction = eig_op.matrix(dense=True)
    reconstruction_error = np.linalg.norm(symmetric_matrix - reconstruction)

    s_true, U_true = eigh(symmetric_matrix)
    s_true, U_true = s_true[::-1], U_true[:, ::-1]
    optimal_reconstruction = (
        U_true[:, :rank] @ np.diag(s_true[:rank]) @ U_true[:, :rank].T
    )
    optimal_error = np.linalg.norm(symmetric_matrix - optimal_reconstruction)

    assert reconstruction_error < 10 * optimal_error

    # 2. Test orthogonality of eigenvectors
    U_adj_U = (eig_op.u_factor.adjoint @ eig_op.u_factor).matrix(dense=True)
    assert np.allclose(U_adj_U, np.eye(rank))


def test_low_rank_cholesky(symmetric_operator, symmetric_matrix):
    """Tests the LowRankCholesky factorization."""
    rank = 10
    chol_op = LowRankCholesky.from_randomized(
        symmetric_operator, rank, method="fixed", power=2
    )

    reconstruction = chol_op.matrix(dense=True)
    reconstruction_error = np.linalg.norm(symmetric_matrix - reconstruction)

    s_true, U_true = eigh(symmetric_matrix)
    s_true, U_true = s_true[::-1], U_true[:, ::-1]
    optimal_reconstruction = (
        U_true[:, :rank] @ np.diag(s_true[:rank]) @ U_true[:, :rank].T
    )
    optimal_error = np.linalg.norm(symmetric_matrix - optimal_reconstruction)

    assert reconstruction_error < 20 * optimal_error


def test_low_rank_cholesky_fallback_path(symmetric_semidefinite_operator):
    """Tests the 'fallback path' (eigendecomposition) of LowRankCholesky."""
    rank = 25
    # Should succeed by falling back to the eigendecomposition method internally
    chol_op = LowRankCholesky.from_randomized(
        symmetric_semidefinite_operator, rank, method="fixed", power=2
    )

    reconstruction = chol_op.matrix(dense=True)
    A = symmetric_semidefinite_operator.matrix(dense=True)
    reconstruction_error = np.linalg.norm(A - reconstruction)

    s_true, U_true = eigh(A)
    s_true, U_true = s_true[::-1], U_true[:, ::-1]
    optimal_reconstruction = (
        U_true[:, :rank] @ np.diag(s_true[:rank]) @ U_true[:, :rank].T
    )
    optimal_error = np.linalg.norm(A - optimal_reconstruction)

    assert reconstruction_error < 1.5 * optimal_error + 1e-9


# =============================================================================
# Tests for Diagonal Approximations
# =============================================================================


@pytest.mark.parametrize("parallel_flag", [False, True])
@pytest.mark.parametrize("use_rademacher", [True, False])
def test_random_diagonal(matrix_with_known_diagonal, parallel_flag, use_rademacher):
    A, true_diag = matrix_with_known_diagonal
    n = A.shape[0]

    estimated_diag = random_diagonal(
        A,
        100,
        max_samples=500,
        rtol=1e-1,
        use_rademacher=use_rademacher,
        parallel=parallel_flag,
    )

    assert estimated_diag.shape == (n,)
    relative_error = np.linalg.norm(estimated_diag - true_diag) / np.linalg.norm(
        true_diag
    )
    assert relative_error < 0.2


class TestDeflatedDiagonal:
    """Tests for the standalone deflated_diagonal function."""

    def test_deflated_diagonal_pure_low_rank(self):
        """Tests exact extraction for a pure low-rank matrix."""
        domain = EuclideanSpace(6)
        A = np.random.randn(6, 2)
        B = np.random.randn(2, 6)
        M = A @ B
        op = LinearOperator.from_matrix(domain, domain, M, galerkin=False)

        diag_est = deflated_diagonal(op, 2, 0, galerkin=False)
        assert np.allclose(diag_est, np.diag(M), rtol=1e-7, atol=1e-7)

    def test_deflated_diagonal_pure_hutchinson(self):
        """Tests pure stochastic extraction."""
        domain = EuclideanSpace(5)
        M = np.random.randn(5, 5)
        op = LinearOperator.from_matrix(domain, domain, M, galerkin=True)

        diag_est = deflated_diagonal(
            op, 0, 5000, method="fixed", max_samples=5000, galerkin=True
        )
        assert np.allclose(diag_est, np.diag(M), rtol=0.1, atol=0.1)

    def test_deflated_diagonal_mixed_asymmetric(self):
        """Tests SVD deflation on a non-symmetric matrix."""
        domain = EuclideanSpace(6)
        M = np.random.randn(6, 6)
        op = LinearOperator.from_matrix(domain, domain, M, galerkin=False)

        diag_est = deflated_diagonal(
            op, 3, 2000, method="fixed", max_samples=2000, galerkin=False
        )
        assert np.allclose(diag_est, np.diag(M), rtol=0.1, atol=0.1)

    def test_deflated_diagonal_variable_convergence(self):
        """Tests progressive termination logic."""
        domain = EuclideanSpace(5)
        M = np.diag(np.random.rand(5) + 1.0)
        op = LinearOperator.from_matrix(domain, domain, M, galerkin=False)

        diag_est = deflated_diagonal(
            op, 1, 10, method="variable", max_samples=2000, rtol=1e-3, block_size=50
        )
        assert np.allclose(diag_est, np.diag(M), rtol=0.05, atol=0.05)


# =============================================================================
# Tests for Measure-Based (Abstract) Factorizations
# =============================================================================


@pytest.fixture
def prior_measure(symmetric_operator) -> GaussianMeasure:
    """Provides a GaussianMeasure with a non-trivial covariance structure."""
    n = symmetric_operator.domain.dim
    # Create a rapidly decaying diagonal covariance to bias the sampling
    cov_matrix = np.diag(np.exp(-np.arange(n) / 5.0))
    return GaussianMeasure.from_covariance_matrix(symmetric_operator.domain, cov_matrix)


def test_random_range_with_measure(rectangular_operator, prior_measure):
    """Tests the abstract range finder using structured samples from a measure."""
    rank = 10

    # Passing the measure forces the algorithm down the `_abstract_...` path
    Q_op = random_range(
        rectangular_operator, rank, measure=prior_measure, method="fixed", power=1
    )

    # 1. Test output shapes
    assert Q_op.domain.dim == rank
    assert Q_op.codomain.dim == rectangular_operator.codomain.dim

    # 2. Test orthonormality (Q* Q = I)
    # The Gram-Schmidt is done in the codomain, so Q should still be a strict isometry.
    identity = np.eye(rank)
    qq_star = (Q_op.adjoint @ Q_op).matrix(dense=True)
    assert np.allclose(qq_star, identity)


def test_low_rank_svd_with_measure(
    rectangular_operator, rectangular_matrix, prior_measure
):
    """Tests that LowRankSVD constructs correctly when given a prior measure."""
    rank = 10

    svd_op = LowRankSVD.from_randomized(
        rectangular_operator, rank, measure=prior_measure, method="fixed"
    )

    # 1. Verify it executed and produced the right shapes
    assert svd_op.rank == rank
    assert svd_op.singular_values.shape == (rank,)

    # 2. Test orthogonality of the resulting abstract factors
    U_adj_U = (svd_op.u_factor.adjoint @ svd_op.u_factor).matrix(dense=True)
    V_adj_V = (svd_op.v_factor.adjoint @ svd_op.v_factor).matrix(dense=True)
    assert np.allclose(U_adj_U, np.eye(rank))
    assert np.allclose(V_adj_V, np.eye(rank))


def test_low_rank_eig_with_measure(symmetric_operator, symmetric_matrix, prior_measure):
    """Tests LowRankEig execution via the abstract measure path."""
    rank = 10

    eig_op = LowRankEig.from_randomized(
        symmetric_operator, rank, measure=prior_measure, method="fixed"
    )

    assert eig_op.rank == rank
    U_adj_U = (eig_op.u_factor.adjoint @ eig_op.u_factor).matrix(dense=True)
    assert np.allclose(U_adj_U, np.eye(rank))


def test_abstract_variable_rank_termination(rectangular_operator, prior_measure):
    """Tests that the variable-rank measure-based algorithm terminates correctly."""

    # Use a generous tolerance to ensure it exits before hitting max_rank
    Q_op = random_range(
        rectangular_operator,
        2,
        measure=prior_measure,
        method="variable",
        rtol=1e-1,
        block_size=2,
    )

    k = Q_op.domain.dim
    assert 2 <= k < rectangular_operator.domain.dim
    assert np.allclose((Q_op.adjoint @ Q_op).matrix(dense=True), np.eye(k))
