"""
This module contains classes related to linear algebra on vector spaces.
"""

import numpy as np
import scipy.linalg as sl
from scipy.sparse import diags
import warnings

from pygeoinf.linalg import LinearOperator, GaussianMeasure


def fixed_rank_basis(matrix, rank, power=0):
    """
    Forms the fixed-rank approximation to the range of a matrix using
    a random-matrix method.

    Args:
        A (matrix-like): (m,n)-matrix whose range is to be approximated.
        rank (int): The desired rank. Must be greater than 1.
        power (int): The exponent to use within the power iterations.

    Returns:
        Q (numpy-matrix): A (m,rank)-matrix whose columns are orthonormal and
            whose span approximate the desired range.

    Notes:
        The input matrix can be a numpy array or a scipy LinearOperator. In the latter case,
        it requires the the matmat, and rmatmat methods have been implemented.

        This method is based on Algorithm 4.4 in Halko et. al. 2011
    """

    assert rank > 1

    m, n = matrix.shape
    random_matrix = np.random.rand(n, rank)

    product_matrix = matrix @ random_matrix
    qr_factor, _ = sl.qr(product_matrix, overwrite_a=True, mode="economic")

    for _ in range(power):
        tilde_product_matrix = matrix.T @ qr_factor
        tilde_qr_factor, _ = sl.qr(
            tilde_product_matrix, overwrite_a=True, mode="economic"
        )
        product_matrix = matrix @ tilde_qr_factor
        qr_factor, _ = sl.qr(product_matrix, overwrite_a=True, mode="economic")

    return qr_factor


def svd(matrix, qr_factor):
    """
    Given a matrix, A,  and a low-rank approximation to its range, Q,
    this function returns the approximate SVD factors, (U, S, Vh)
    such that A ~ U @ S @ VT where S is diagonal.
    """
    small_matrix = qr_factor.T @ matrix
    left_factor, diagonal_factor, right_factor_transposed = sl.svd(
        small_matrix, full_matrices=False, overwrite_a=True
    )
    return (
        qr_factor @ left_factor,
        diags([diagonal_factor], [0]),
        right_factor_transposed,
    )


def eigh(matrix, qr_factor):
    """
    Given a symmetric matrix, A,  and a low-rank approximation to its range, Q,
    this function returns the approximate eigen-decomposition, (U, S)
    such that A ~ U @ S @ U.T where S is diagonal.
    """
    m, n = matrix.shape
    assert m == n
    small_matrix = qr_factor.T @ matrix @ qr_factor
    eigenvalues, eigenvectors = sl.eigh(small_matrix, overwrite_a=True)
    return qr_factor @ eigenvectors, diags([eigenvalues], [0])


class RandomSVDOperator(LinearOperator):
    """
    A LinearOperator formed using random SVD of another one.
    """

    def __init__(self, operator, rank, /, *, power=0, galerkin=False):
        assert operator.hilbert_operator or not galerkin
        assert operator.domain == operator.codomain
        self._galerkin = galerkin
        self._operator = operator
        matrix = operator.matrix(galerkin=galerkin)
        qr_factor = fixed_rank_basis(matrix, rank, power)
        self._left_factor, self._diagonal_factor, self._right_factor_transposed = svd(
            matrix, qr_factor
        )
        super().__init__(
            operator.domain,
            operator.codomain,
            self._mapping,
            dual_mapping=self._dual_mappping,
        )

    def _mapping(self, x):
        domain = self._operator.domain
        codomain = self._operator.codomain

        if self._galerkin:
            cx = domain.to_components(x)
            cyp = self._left_factor @ (
                self._diagonal_factor @ (self._right_factor_transposed @ cx)
            )
            yp = codomain.dual.from_components(cyp)
            return codomain.from_dual(yp)

        else:
            cx = domain.to_components(x)
            cy = self._left_factor @ (
                self._diagonal_factor @ (self._right_factor_transposed @ cx)
            )
            return codomain.from_components(cy)

    def _dual_mappping(self, yp):
        domain = self._operator.domain
        codomain = self._operator.codomain

        if self._galerkin:
            y = codomain.from_dual(yp)
            cy = codomain.to_components(y)
            cxp = self._right_factor_transposed.T @ (
                self._diagonal_factor @ (self._left_factor.T @ cy)
            )
            return domain.dual.from_components(cxp)

        else:
            cyp = codomain.dual.to_components(yp)
            cxp = self._right_factor_transposed.T @ (
                self._diagonal_factor @ (self._left_factor.T @ cyp)
            )
            return domain.dual.from_components(cxp)


class RandomEigenOperator(LinearOperator):
    """
    A self-adjoint LinearOperator formed using random eiegn-decomposition of another one.
    """

    def __init__(self, operator, rank, /, *, power=0):
        assert operator.hilbert_operator
        self._operator = operator
        matrix = operator.matrix(galerkin=True)
        qr_factor = fixed_rank_basis(matrix, rank, power)
        self._eigenvectors, self._eigenvalues = eigh(matrix, qr_factor)
        super().__init__(
            operator.domain,
            operator.codomain,
            self._mapping,
            adjoint_mapping=self._mapping,
        )

    def _mapping(self, x):
        domain = self._operator.domain
        codomain = self._operator.codomain
        cx = domain.to_components(x)
        cyp = self._eigenvectors @ (self._eigenvalues @ (self._eigenvectors.T @ cx))
        yp = codomain.dual.from_components(cyp)
        return codomain.from_dual(yp)
