"""
A collection of helper functions for parallel computation using Joblib.

These functions are designed to be top-level to ensure they can be
"pickled" (serialized) and sent to worker processes by libraries like
multiprocessing or its wrapper, Joblib.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union
import numpy as np
from joblib import Parallel, delayed

if TYPE_CHECKING:
    from scipy.sparse.linalg import LinearOperator as ScipyLinOp

    MatrixLike = Union[np.ndarray, ScipyLinOp]


def parallel_mat_mat(A: "MatrixLike", B: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    """
    Computes the matrix product A @ B in parallel by applying A to each column of B.

    This is particularly useful when A is a LinearOperator whose action is
    computationally expensive.

    Args:
        A: The matrix or LinearOperator to apply.
        B: The matrix whose columns will be operated on.
        n_jobs: The number of CPU cores to use. -1 means all available.

    Returns:
        The result of the matrix product A @ B as a dense NumPy array.
    """
    columns = Parallel(n_jobs=n_jobs)(
        delayed(A.__matmul__)(B[:, i]) for i in range(B.shape[1])
    )
    return np.column_stack(columns)


def parallel_compute_dense_matrix_from_scipy_op(
    scipy_op: "ScipyLinOp", n_jobs: int = -1
) -> np.ndarray:
    """
    Computes the dense matrix representation of a scipy.LinearOperator in parallel.

    It builds the matrix column by column by applying the operator to each
    basis vector.

    Args:
        scipy_op: The SciPy LinearOperator wrapper for the matrix action.
        n_jobs: The number of CPU cores to use. -1 means all available.

    Returns:
        The dense matrix as a NumPy array.
    """
    codomain_dim, domain_dim = scipy_op.shape
    columns = Parallel(n_jobs=n_jobs)(
        delayed(_worker_compute_scipy_op_col)(scipy_op, j, domain_dim)
        for j in range(domain_dim)
    )
    return np.column_stack(columns)


def _worker_compute_scipy_op_col(
    scipy_op: "ScipyLinOp", j: int, domain_dim: int
) -> np.ndarray:
    """
    (Internal worker) Computes a single column of a SciPy LinearOperator's matrix.
    """
    cx = np.zeros(domain_dim)
    cx[j] = 1.0
    return scipy_op @ cx
