"""
Implements randomized algorithms for low-rank matrix factorizations.

This module provides functions for computing approximate, low-rank matrix
factorizations (SVD, Cholesky, Eigendecomposition) using randomized methods.
These algorithms are particularly effective for large, high-dimensional matrices
where deterministic methods would be computationally prohibitive. They work by
finding a low-dimensional subspace that captures most of the "action" of the
matrix.

The implementations are based on the seminal work of Halko, Martinsson, and
Tropp, "Finding structure with randomness: Probabilistic algorithms for
constructing approximate matrix decompositions" (2011).
"""

from typing import Tuple, Union

import warnings

import numpy as np
from scipy.linalg import cho_factor, solve_triangular, eigh, svd, qr
from scipy.sparse.linalg import LinearOperator as ScipyLinOp

from .parallel import parallel_mat_mat

# A type for objects that act like matrices (numpy arrays or SciPy LinearOperators)
MatrixLike = Union[np.ndarray, ScipyLinOp]


def fixed_rank_random_range(
    matrix: MatrixLike,
    rank: int,
    power: int = 0,
    parallel: bool = False,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Computes an orthonormal basis for a fixed-rank approximation of a matrix's range.

    This randomized algorithm finds a low-dimensional subspace that captures
    most of the action of the matrix.

    Args:
        matrix: An (m, n) matrix or scipy.LinearOperator whose range is to be approximated.
        rank: The desired rank for the approximation.
        power: The number of power iterations to perform. Power iterations
            (multiplying by `A*A`) improves the accuracy of the approximation by
            amplifying the dominant singular values, but adds to the computational cost.
        parallel: Whether to use parallel matrix multiplication.
        n_jobs: Number of jobs for parallelism.

    Returns:
        An (m, rank) matrix with orthonormal columns whose span approximates
        the range of the input matrix.

    Notes:
        Based on Algorithm 4.4 in Halko et al. 2011.
    """
    m, n = matrix.shape
    random_matrix = np.random.randn(n, rank)

    if parallel:
        product_matrix = parallel_mat_mat(matrix, random_matrix, n_jobs)
    else:
        product_matrix = matrix @ random_matrix

    qr_factor, _ = qr(product_matrix, overwrite_a=True, mode="economic")

    for _ in range(power):
        if parallel:
            tilde_product_matrix = parallel_mat_mat(matrix.T, qr_factor, n_jobs)
        else:
            tilde_product_matrix = matrix.T @ qr_factor

        tilde_qr_factor, _ = qr(tilde_product_matrix, overwrite_a=True, mode="economic")

        if parallel:
            product_matrix = parallel_mat_mat(matrix, tilde_qr_factor, n_jobs)
        else:
            product_matrix = matrix @ tilde_qr_factor

        qr_factor, _ = qr(product_matrix, overwrite_a=True, mode="economic")

    return qr_factor


def variable_rank_random_range(
    matrix: MatrixLike,
    initial_rank: int,
    /,
    *,
    max_rank: int = None,
    power: int = 0,
    block_size: int = 10,
    rtol: float = 1e-4,
    parallel: bool = False,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Computes a variable-rank orthonormal basis using a progressive sampling algorithm.

    The algorithm starts with `initial_rank` samples, checks for convergence,
    and then progressively draws new blocks of random samples until the desired
    tolerance `rtol` is met or `max_rank` is reached.

    Args:
        matrix: The (m, n) matrix or LinearOperator.
        initial_rank: The number of vectors to sample initially.
        max_rank: A hard limit on the number of basis vectors. Defaults to min(m, n).
        power: Number of power iterations to improve accuracy on the initial sample.
        rtol: Relative tolerance for determining the output rank.
        block_size: The number of new vectors to sample in each iteration.
        parallel: Whether to use parallel matrix multiplication.
        n_jobs: Number of jobs for parallelism.

    Returns:
        An (m, k) matrix with orthonormal columns that approximates the matrix's
        range to the given tolerance.
    """
    m, n = matrix.shape
    if max_rank is None:
        max_rank = min(m, n)

    # Initial Sample
    random_matrix = np.random.randn(n, initial_rank)
    if parallel:
        ys = parallel_mat_mat(matrix, random_matrix, n_jobs)
    else:
        ys = matrix @ random_matrix

    # Power Iterations on initial sample for a better starting point
    for _ in range(power):
        ys, _ = qr(ys, mode="economic")
        if parallel:
            ys_tilde = parallel_mat_mat(matrix.T, ys, n_jobs)
            ys = parallel_mat_mat(matrix, ys_tilde, n_jobs)
        else:
            ys_tilde = matrix.T @ ys
            ys = matrix @ ys_tilde

    # Form the initial basis
    basis_vectors, _ = qr(ys, mode="economic")

    # Progressively sample and check for convergence
    converged = False

    # Dynamically estimate norm for tolerance calculation
    tol = None

    while basis_vectors.shape[1] < max_rank:
        # Generate a NEW block of random vectors for error checking
        test_vectors = np.random.randn(n, block_size)
        if parallel:
            y_test = parallel_mat_mat(matrix, test_vectors, n_jobs)
        else:
            y_test = matrix @ test_vectors

        # Estimate norm for tolerance on the first pass
        if tol is None:
            # Estimate spectral norm from the first block of test vectors.
            # A more stable estimate than from a single vector.
            norm_estimate = np.linalg.norm(y_test) / np.sqrt(block_size)
            tol = rtol * norm_estimate

        # Project test vectors onto current basis to find the residual
        residual = y_test - basis_vectors @ (basis_vectors.T @ y_test)
        error = np.linalg.norm(residual, ord=2)

        # Check for convergence
        if error < tol:
            converged = True
            break

        # If not converged, add the new information to the basis
        new_basis, _ = qr(residual, mode="economic")

        # Append new basis vectors, ensuring we don't exceed max_rank
        cols_to_add = min(new_basis.shape[1], max_rank - basis_vectors.shape[1])
        if cols_to_add <= 0:
            break

        basis_vectors = np.hstack([basis_vectors, new_basis[:, :cols_to_add]])

    if not converged and basis_vectors.shape[1] >= max_rank:
        warnings.warn(
            f"Tolerance {rtol} not met before reaching max_rank={max_rank}. "
            "Result may be inaccurate. Consider increasing `max_rank` or `power`.",
            UserWarning,
        )

    return basis_vectors


def random_range(
    matrix: MatrixLike,
    size_estimate: int,
    /,
    *,
    method: str = "variable",
    max_rank: int = None,
    power: int = 2,
    rtol: float = 1e-4,
    block_size: int = 10,
    parallel: bool = False,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    A unified wrapper for randomized range finding algorithms.

    Args:
        matrix: The (m, n) matrix or LinearOperator to analyze.
        size_estimate: For 'fixed' method, the exact target rank. For 'variable'
                       method, this is the initial rank to sample.
        method ({'variable', 'fixed'}): The algorithm to use.
            - 'variable': (Default) Progressively samples to find the rank needed
                          to meet tolerance `rtol`, stopping at `max_rank`.
            - 'fixed': Returns a basis with exactly `size_estimate` columns.
        max_rank: For 'variable' method, a hard limit on the rank. Ignored if
                  method='fixed'. Defaults to min(m, n).
        power: Number of power iterations to improve accuracy.
        rtol: Relative tolerance for the 'variable' method. Ignored if
              method='fixed'.
        block_size: Number of new vectors to sample per iteration in 'variable'
                    method. Ignored if method='fixed'.
        parallel: Whether to use parallel matrix multiplication.
        n_jobs: Number of jobs for parallelism.

    Returns:
        An (m, k) orthonormal matrix approximating the input matrix's range.

    Raises:
        ValueError: If an unknown method is specified.
    """
    if method == "variable":
        return variable_rank_random_range(
            matrix,
            size_estimate,
            max_rank=max_rank,
            power=power,
            block_size=block_size,
            rtol=rtol,
            parallel=parallel,
            n_jobs=n_jobs,
        )
    elif method == "fixed":
        if any([rtol != 1e-4, block_size != 10, max_rank is not None]):
            warnings.warn(
                "'rtol', 'block_size', and 'max_rank' are ignored when method='fixed'.",
                UserWarning,
            )
        return fixed_rank_random_range(
            matrix,
            rank=size_estimate,
            power=power,
            parallel=parallel,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from 'fixed' or 'variable'."
        )


def random_svd(
    matrix: MatrixLike, qr_factor: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes an approximate SVD from a low-rank range approximation.

    This function takes the original matrix and an orthonormal basis for its
    approximate range (the `qr_factor`) and projects the problem into a smaller
    subspace where a deterministic SVD is cheap to compute.

    Args:
        matrix: The original (m, n) matrix or LinearOperator.
        qr_factor: An (m, k) orthonormal basis for the approximate range,
            typically from a `random_range` function.

    Returns:
        A tuple `(U, S, Vh)` containing the approximate SVD factors, where S is
        a 1D array of singular values.

    Notes:
        Based on Algorithm 5.1 of Halko et al. 2011.
    """
    small_matrix = qr_factor.T @ matrix
    left_factor, diagonal_factor, right_factor_transposed = svd(
        small_matrix, full_matrices=False, overwrite_a=True
    )
    return (
        qr_factor @ left_factor,
        diagonal_factor,
        right_factor_transposed,
    )


def random_eig(
    matrix: MatrixLike, qr_factor: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes an approximate eigendecomposition for a symmetric matrix from a
    low-rank range approximation.

    Args:
        matrix (matrix-like): The original symmetric (n, n) matrix or
            LinearOperator.
        qr_factor (numpy.ndarray): An (n, k) orthonormal basis for the
            approximate range of the matrix.

    Returns:
        (numpy.ndarray, numpy.ndarray): A tuple (U, S) containing the
            approximate eigenvectors and eigenvalues, such that A ~= U @ S @ U.T.
            S is a 1D array of eigenvalues.

    Notes:
        Based on Algorithm 5.3 of Halko et al. 2011.
    """
    m, n = matrix.shape
    assert m == n
    small_matrix = qr_factor.T @ matrix @ qr_factor
    eigenvalues, eigenvectors = eigh(small_matrix, overwrite_a=True)
    return qr_factor @ eigenvectors, eigenvalues


def random_cholesky(
    matrix: MatrixLike, qr_factor: np.ndarray, *, rtol: float = 1e-12
) -> np.ndarray:
    """
    Computes a robust approximate Cholesky factorization using a fallback strategy.

    It first attempts a direct Cholesky factorization. If that fails, it falls
    back to a method based on eigendecomposition.

    Args:
        matrix (matrix-like): The original symmetric (n, n) matrix.
        qr_factor (numpy.ndarray): An (n, k) orthonormal basis for the
            approximate range of the matrix.
        rtol (float, optional): A relative tolerance used in the fallback path.
            Any eigenvalue `s` such that `s < rtol * max(eigenvalues)` will be
            treated as zero. Defaults to 1e-12.

    Returns:
        numpy.ndarray: The approximate Cholesky factor F, such that A ~= F @ F.T.
    """
    try:
        # --- Fast Path: Try direct Cholesky factorization ---
        small_matrix_1 = matrix @ qr_factor
        small_matrix_2 = qr_factor.T @ small_matrix_1

        factor, lower = cho_factor(small_matrix_2, overwrite_a=True)

        identity_operator = np.identity(factor.shape[0])
        inverse_factor = solve_triangular(
            factor, identity_operator, overwrite_b=True, lower=lower
        )
        return small_matrix_1 @ inverse_factor

    except np.linalg.LinAlgError:

        # --- Fallback Path: Eigendecomposition ---
        small_matrix = qr_factor.T @ (matrix @ qr_factor)
        eigenvalues, eigenvectors = eigh(small_matrix, overwrite_a=True)

        # Determine the threshold based on the largest eigenvalue.
        # eigh returns eigenvalues in ascending order.
        max_eigenvalue = eigenvalues[-1]

        if max_eigenvalue > 0:
            threshold = rtol * max_eigenvalue
        else:
            # If all eigenvalues are non-positive, all will be set to zero.
            threshold = 0

        # 2. Apply the threshold to create safe eigenvalues.
        safe_eigenvalues = eigenvalues.copy()
        safe_eigenvalues[eigenvalues < threshold] = 0.0

        y_matrix = matrix @ qr_factor
        temp_factor = y_matrix @ eigenvectors

        # Conditionally compute the inverse square root.
        sqrt_s = np.sqrt(safe_eigenvalues)
        sqrt_s_inv = np.where(sqrt_s > 0, np.reciprocal(sqrt_s), 0.0)

        cholesky_factor = temp_factor * sqrt_s_inv

        return cholesky_factor


def random_diagonal(
    matrix: MatrixLike,
    size_estimate: int,
    /,
    *,
    method: str = "variable",
    use_rademacher: bool = False,
    max_samples: int = None,
    rtol: float = 1e-2,
    block_size: int = 10,
    parallel: bool = False,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Computes an approximate diagonal of a square matrix using Hutchinson's method.

    This algorithm uses a progressive, iterative approach to estimate the diagonal.
    It starts with an initial number of samples and adds new blocks of random
    vectors until the estimate of the diagonal converges to a specified tolerance.

    Args:
        matrix: The (m, n) matrix or LinearOperator to analyze.
        size_estimate: For 'fixed' method, the exact target rank. For 'variable'
                       method, this is the initial rank to sample.
        method ({'variable', 'fixed'}): The algorithm to use.
            - 'variable': (Default) Progressively samples to find the rank needed
                          to meet tolerance `rtol`, stopping at `max_rank`.
            - 'fixed': Returns a basis with exactly `size_estimate` columns.
        use_rademacher: If true, draw components from [-1,1]. Default method draws
            normally distributed components.
        max_samples: For 'variable' method, a hard limit on the number of samples.
                     Ignored if method='fixed'. Defaults to dimension of matrix.
        rtol: Relative tolerance for the 'variable' method. Ignored if
              method='fixed'.
        block_size: Number of new vectors to sample per iteration in 'variable'
                    method. Ignored if method='fixed'.
        parallel: Whether to use parallel matrix multiplication.
        n_jobs: Number of jobs for parallelism.

    Returns:
        A 1D numpy array of size n containing the approximate diagonal of the matrix.
    """

    m, n = matrix.shape
    if m != n:
        raise ValueError("Input matrix must be square to estimate a diagonal.")

    if max_samples is None:
        max_samples = n

    num_samples = min(size_estimate, max_samples)
    if use_rademacher:
        z = np.random.choice([-1.0, 1.0], size=(n, num_samples))
    else:
        z = np.random.randn(n, num_samples)

    if parallel:
        az = parallel_mat_mat(matrix, z, n_jobs)
    else:
        az = matrix @ z

    diag_sum = np.sum(z * az, axis=1)
    diag_estimate = diag_sum / num_samples

    if method == "fixed":
        return diag_estimate

    if num_samples >= max_samples:
        return diag_estimate

    converged = False
    while num_samples < max_samples:
        old_diag_estimate = diag_estimate.copy()

        # Generate a NEW block of random vectors
        samples_to_add = min(block_size, max_samples - num_samples)
        if use_rademacher:
            z_new = np.random.choice([-1.0, 1.0], size=(n, samples_to_add))
        else:
            z_new = np.random.randn(n, samples_to_add)

        if parallel:
            az_new = parallel_mat_mat(matrix, z_new, n_jobs)
        else:
            az_new = matrix @ z_new

        new_diag_sum = np.sum(z_new * az_new, axis=1)

        # Update the running average
        total_samples = num_samples + samples_to_add
        diag_estimate = (diag_sum + new_diag_sum) / total_samples

        # Check for convergence
        norm_new_diag = np.linalg.norm(diag_estimate)
        if norm_new_diag > 0:
            error = np.linalg.norm(diag_estimate - old_diag_estimate) / norm_new_diag
            if error < rtol:
                converged = True
                break

        # Update sums and counts for next iteration
        diag_sum += new_diag_sum
        num_samples = total_samples

    if not converged and num_samples >= max_samples:
        warnings.warn(
            f"Tolerance {rtol} not met before reaching max_samples={max_samples}. "
            "Result may be inaccurate. Consider increasing `max_samples` or `rtol`.",
            UserWarning,
        )

    return diag_estimate
