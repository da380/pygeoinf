"""
Module for random matrix factorisations. 
"""

import numpy as np
from scipy.linalg import (
    cho_factor,
    solve_triangular,
    eigh,
    svd,
    qr,
)


def fixed_rank_random_range(matrix, rank, power=0):
    """
    Forms the fixed-rank approximation to the range of a matrix using
    a random-matrix method.

    Args:
        matrix (matrix-like): (m,n)-matrix whose range is to be approximated.
        rank (int): The desired rank. Must be greater than 1.
        power (int): The exponent to use within the power iterations.

    Returns:
        matrix: A (m,rank)-matrix whose columns are orthonormal and
            whose span approximates the desired range.

    Notes:
        The input matrix can be a numpy array or a scipy LinearOperator. In the latter case,
        it requires the the matmat, and rmatmat methods have been implemented.

        This method is based on Algorithm 4.4 in Halko et. al. 2011
    """

    m, n = matrix.shape
    random_matrix = np.random.randn(n, rank)

    product_matrix = matrix @ random_matrix
    qr_factor, _ = qr(product_matrix, overwrite_a=True, mode="economic")

    for _ in range(power):
        tilde_product_matrix = matrix.T @ qr_factor
        tilde_qr_factor, _ = qr(tilde_product_matrix, overwrite_a=True, mode="economic")
        product_matrix = matrix @ tilde_qr_factor
        qr_factor, _ = qr(product_matrix, overwrite_a=True, mode="economic")

    return qr_factor


def variable_rank_random_range(matrix, rank, /, *, power=0, rtol=1e-6):
    """
    Forms the variable-rank approximation to the range of a matrix using
    a random-matrix method.

    Args:
        matrix (matrix-like): (m,n)-matrix whose range is to be approximated.

    Returns:
        matrix: A (m,rank)-matrix whose columns are orthonormal and
            whose span approximates the desired range.

    Notes:
        The input matrix can be a numpy array or a scipy LinearOperator. In the latter case,
        it requires the the matmat, and rmatmat methods have been implemented.

        This method is based on Algorithm 4.5 in Halko et. al. 2011
    """

    m, n = matrix.shape

    random_vectors = [np.random.randn(n) for _ in range(rank)]
    ys = [matrix @ x for x in random_vectors]
    basis_vectors = []

    def projection(xs, y):
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


def random_svd(matrix, qr_factor):
    """
    Given a matrix, A,  and a low-rank approximation to its range, Q,
    this function returns the approximate SVD factors, (U, S, Vh)
    such that A ~ U @ S @ VT where S is diagonal.

    Based on Algorithm 5.1 of Halko et al. 2011
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


def random_eig(matrix, qr_factor):
    """
    Given a symmetric matrix, A,  and a low-rank approximation to its range, Q,
    this function returns the approximate eigen-decomposition, (U, S)
    such that A ~ U @ S @ U.T where S is diagonal.

    Based on Algorithm 5.3 of Halko et al. 2011
    """
    m, n = matrix.shape
    assert m == n
    small_matrix = qr_factor.T @ matrix @ qr_factor
    eigenvalues, eigenvectors = eigh(small_matrix, overwrite_a=True)
    return qr_factor @ eigenvectors, eigenvalues


def random_cholesky(matrix, qr_factor):
    """
    Given a symmetric and positive-definite matrix, A,  along with a low-rank
    approximation to its range, Q, this function returns the approximate
    Cholesky factorisation A ~ F F*.

    Based on Algorithm 5.5 of Halko et al. 2011
    """
    small_matrix_1 = matrix @ qr_factor
    small_matrix_2 = qr_factor.T @ small_matrix_1
    factor, lower = cho_factor(small_matrix_2, overwrite_a=True)
    identity_operator = np.identity(factor.shape[0])
    inverse_factor = solve_triangular(
        factor, identity_operator, overwrite_b=True, lower=lower
    )
    return small_matrix_1 @ inverse_factor
