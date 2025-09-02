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

import numpy as np
from scipy.linalg import (
    cho_factor,
    solve_triangular,
    eigh,
    svd,
    qr,
)
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
    matrix: MatrixLike, rank: int, /, *, power: int = 0, rtol: float = 1e-6
) -> np.ndarray:
    """
    Computes an orthonormal basis for a variable-rank approximation to the
    range of a matrix using a randomized method.

    The algorithm adaptively determines the rank required to meet a given
    error tolerance.

    Args:
        matrix (matrix-like): An (m, n) matrix or LinearOperator whose range
            is to be approximated.
        rank (int): The maximum rank for the approximation. The algorithm
            may return a basis with a smaller rank.
        power (int): Exponent for power iterations. Note: This parameter is
            reserved for future functionality and is currently unused.
        rtol (float): The relative tolerance for the approximation error, used
            to determine the output rank.

    Returns:
        numpy.ndarray: An (m, k) matrix with orthonormal columns, where k <= rank.
            Its span approximates the range of the input matrix.

    Notes:
        If the input matrix is a scipy LinearOperator, it must have the
        `matvec` method implemented.

        This method is based on Algorithm 4.5 in Halko et al. 2011.
    """

    m, n = matrix.shape

    random_vectors = [np.random.randn(n) for _ in range(rank)]
    ys = [matrix @ x for x in random_vectors]
    basis_vectors = []

    def projection(xs: list, y: np.ndarray) -> np.ndarray:
        ps = [np.dot(x, y) for x in xs]
        for p, x in zip(ps, xs):
            y -= p * x
        return y

    norm = max(np.linalg.norm(y) for y in ys)

    tol = rtol * norm / (10 * np.sqrt(2 / np.pi))
    error = 2 * tol
    j = -1
    while error > tol:
        j += 1

        ys[j] = projection(basis_vectors, ys[j])
        ys[j] /= np.linalg.norm(ys[j])
        basis_vectors.append(ys[j])

        y = matrix @ np.random.randn(n)
        y = projection(basis_vectors, y)
        ys.append(y)

        for i in range(j + 1, j + rank):
            p = np.dot(basis_vectors[j], ys[i])
            ys[i] -= p * basis_vectors[j]

        error = max(np.linalg.norm(ys[i]) for i in range(j + 1, j + rank + 1))

        if j > min(n, m):
            raise RuntimeError("Convergence has failed")

    qr_factor = np.column_stack(basis_vectors)

    return qr_factor


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


def random_cholesky(matrix: MatrixLike, qr_factor: np.ndarray) -> np.ndarray:
    """
    Computes an approximate Cholesky factorisation for a symmetric positive-
    definite matrix from a low-rank range approximation.

    Args:
        matrix (matrix-like): The original symmetric positive-definite (n, n)
            matrix or LinearOperator.
        qr_factor (numpy.ndarray): An (n, k) orthonormal basis for the
            approximate range of the matrix.

    Returns:
        numpy.ndarray: The approximate Cholesky factor F, such that A ~= F @ F.T.

    Notes:
        Based on Algorithm 5.5 of Halko et al. 2011.
    """
    small_matrix_1 = matrix @ qr_factor
    small_matrix_2 = qr_factor.T @ small_matrix_1
    factor, lower = cho_factor(small_matrix_2, overwrite_a=True)
    identity_operator = np.identity(factor.shape[0])
    inverse_factor = solve_triangular(
        factor, identity_operator, overwrite_b=True, lower=lower
    )
    return small_matrix_1 @ inverse_factor
