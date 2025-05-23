"""
Module for random matrix factorisations. 
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import (
    cho_factor,
    cho_solve,
    lu_factor,
    lu_solve,
    solve_triangular,
    eigh,
    svd,
    qr,
)
from scipy.stats import norm
from scipy.sparse.linalg import LinearOperator as ScipyLinOp
from scipy.sparse.linalg import gmres, bicgstab, cg, bicg
from scipy.sparse import diags


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
    random_matrix = np.random.rand(n, rank)

    product_matrix = matrix @ random_matrix
    qr_factor, _ = qr(product_matrix, overwrite_a=True, mode="economic")

    for _ in range(power):
        tilde_product_matrix = matrix.T @ qr_factor
        tilde_qr_factor, _ = qr(tilde_product_matrix, overwrite_a=True, mode="economic")
        product_matrix = matrix @ tilde_qr_factor
        qr_factor, _ = qr(product_matrix, overwrite_a=True, mode="economic")

    return qr_factor


def variable_rank_random_range(matrix, rtol, /, *, rank=None, power=0):
    """
    Forms the variable-rank approximation to the range of a matrix using
    a random-matrix method.

    Args:
        matrix (matrix-like): (m,n)-matrix whose range is to be approximated.
        rtol (float): The desired relative accuracy.
        rank (int): Starting rank for the decomposition. If none, then
            determined from the dimension of the matrix.
        power (int): The exponent to use within the power iterations.

    Returns:
        matrix: A (m,rank)-matrix whose columns are orthonormal and
            whose span approximates the desired range.

    Notes:
        The input matrix can be a numpy array or a scipy LinearOperator. In the latter case,
        it requires the the matmat, and rmatmat methods have been implemented.

        This method is based on Algorithm 4.5 in Halko et. al. 2011
    """
    raise NotImplementedError


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
