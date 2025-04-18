"""
This contains classes and methods linked to random matrix decompositions.
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
from scipy.stats import norm, multivariate_normal
from scipy.sparse.linalg import LinearOperator as ScipyLinOp
from scipy.sparse.linalg import gmres, bicgstab, cg, bicg
from scipy.sparse import diags
import warnings

from pygeoinf.linalg import LinearOperator


def fixed_rank_basis(A, rank, power=0):
    """
    Forms the fixed-rank approximation to the range of a matrix using
    a random-matrix method.

    Args:
        A (matrix-like): (m,n)-matrix whose range is to be approximated.
        rank (int): The desired rank.
        power (int): The exponent to use within the power iterations.

    Returns:
        Q (numpy-matrix): A (m,rank)-matrix whose columns are orthonormal and
            whose span approximate the desired range.

    Notes:
        The input matrix can be a numpy array or a scipy LinearOperator. In the latter case,
        it requires the the matmat, and rmatmat methods have been implemented.

        This method is based on Algorithm 4.4 in Halko et. al. 2011
    """

    m, n = A.shape
    Omega = np.random.rand(n, rank)

    Y = A @ Omega
    Q, _ = qr(Y, overwrite_a=True, mode="economic")

    for _ in range(power):
        YT = A.T @ Q
        QT, _ = qr(YT, overwrite_a=True, mode="economic")
        Y = A @ QT
        Q, _ = qr(Y, overwrite_a=True, mode="economic")

    return Q


def low_rank_svd(A, Q):
    """
    Given a matrix A and a low-rank approximation to its range in Q,
    this function returns the approximate SVD factors, (U, S, Vh)
    such that A \approx U @ Simga @ VT
    """

    B = Q.T @ A
    U, S, Vh = svd(B, full_matrices=False, overwrite_a=True)
    return Q @ U, diags([S], [0]), Vh


class RandomSVDApproximation(LinearOperator):
    """
    Forms a fixed-rank approximation to a given operator using
    the random SVD method.
    """

    def __init__(self, operator, rank, /, *, galerkin=False, pow=0):
        """
        Args:
            operator (LinearOperator): The operator to approximate.
            rank (int): The rank for the approximation.
            galerkin (bool): If true use the Galerkin representation, otherwise
                use the standard one.
            pow (int): Exponent to use within the power iterations.
        """
        self._operator = operator
        self._rank = rank

        super().__init__(operator.domain, operator.codomain, lambda x: x)
