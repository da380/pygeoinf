"""
Implements randomized matrix-free algorithms for operators on Hilbert spaces.

This module provides fully abstract, matrix-free implementations of randomized
factorizations (SVD, Cholesky, Eigendecomposition). These algorithms operate
entirely via operator composition, adjoint mappings, and the intrinsic geometry
of the underlying `HilbertSpace` objects.

If a `GaussianMeasure` is provided, the algorithms draw structured samples to
respect mass matrices and continuous function space geometries. If no measure
is provided, they fall back to highly optimized, component-based matrix-free algorithms.
"""

from __future__ import annotations
from typing import List, Optional, Union
import warnings

import numpy as np
import scipy.linalg
from scipy.linalg import qr
from scipy.sparse.linalg import LinearOperator as ScipyLinOp

from .hilbert_space import Vector, EuclideanSpace
from .linear_operators import LinearOperator, DiagonalSparseMatrixLinearOperator
from .gaussian_measure import GaussianMeasure
from .parallel import parallel_mat_mat

# A type for objects that act like matrices
MatrixLike = Union[np.ndarray, ScipyLinOp]


class LowRankSVD(LinearOperator):
    """
    A LinearOperator representing the low-rank SVD: A ≈ U @ Sigma @ V*.

    This class encapsulates the components of a Singular Value Decomposition,
    allowing it to be used directly as a `LinearOperator` while providing
    access to the individual low-rank factors.
    """

    def __init__(
        self,
        u_op: LinearOperator,
        sigma_op: DiagonalSparseMatrixLinearOperator,
        v_op: LinearOperator,
    ):
        """
        Initializes the LowRankSVD operator.

        Args:
            u_op (LinearOperator): The left singular vectors operator (isometry).
            sigma_op (DiagonalSparseMatrixLinearOperator): The diagonal operator of singular values.
            v_op (LinearOperator): The right singular vectors operator (isometry).
        """
        self._u_op = u_op
        self._sigma_op = sigma_op
        self._v_op = v_op
        self._rank = sigma_op.domain.dim

        full_op = self.u_factor @ self.sigma_factor @ self.v_factor.adjoint

        super().__init__(
            full_op.domain,
            full_op.codomain,
            full_op,
            adjoint_mapping=full_op.adjoint,
        )

    @property
    def rank(self) -> int:
        """int: The rank of the approximation."""
        return self._rank

    @property
    def u_factor(self) -> LinearOperator:
        """LinearOperator: The left singular vectors (U)."""
        return self._u_op

    @property
    def sigma_factor(self) -> DiagonalSparseMatrixLinearOperator:
        """DiagonalSparseMatrixLinearOperator: The diagonal matrix of singular values (Sigma)."""
        return self._sigma_op

    @property
    def v_factor(self) -> LinearOperator:
        """LinearOperator: The right singular vectors (V)."""
        return self._v_op

    @property
    def singular_values(self) -> np.ndarray:
        """np.ndarray: A 1D array of the computed singular values."""
        return self._sigma_op.extract_diagonal()

    @classmethod
    def from_randomized(
        cls,
        operator: LinearOperator,
        size_estimate: int,
        *,
        measure: Optional[GaussianMeasure] = None,
        galerkin: bool = False,
        method: str = "variable",
        max_rank: Optional[int] = None,
        power: int = 2,
        rtol: float = 1e-4,
        block_size: int = 10,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> LowRankSVD:
        """
        Computes the SVD using a unified randomized range finder.

        Args:
            operator (LinearOperator): The operator to approximate.
            size_estimate (int): For 'fixed' method, the exact target rank. For 'variable'
                method, this is the initial rank to sample.
            measure (GaussianMeasure, optional): A prior measure used to draw test vectors.
                If provided, respects the domain's geometry. If None, falls back to a
                component-based SciPy LinearOperator representation.
            galerkin (bool): If True, computes the Galerkin representation when falling back to components.
            method (str): {'variable', 'fixed'}. The rank-determination algorithm to use.
            max_rank (int, optional): Hard limit on the rank for the 'variable' method.
            power (int): Number of power iterations to improve accuracy.
            rtol (float): Relative tolerance for the 'variable' method.
            block_size (int): Number of new vectors to sample per iteration in 'variable' method.
            parallel (bool): Whether to parallelize the matrix/operator evaluations.
            n_jobs (int): Number of cores to use if parallel=True (-1 for all).

        Returns:
            LowRankSVD: An instantiated operator containing the U, Sigma, and V factors.
        """
        Q_op = random_range(
            operator,
            size_estimate,
            measure=measure,
            galerkin=galerkin,
            method=method,
            max_rank=max_rank,
            power=power,
            rtol=rtol,
            block_size=block_size,
            parallel=parallel,
            n_jobs=n_jobs,
        )

        B = Q_op.adjoint @ operator
        M_op = B @ B.adjoint
        M_dense = M_op.matrix(dense=True)

        eigenvalues, U_tilde = scipy.linalg.eigh(M_dense)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        U_tilde = U_tilde[:, idx]
        sigmas = np.sqrt(np.maximum(eigenvalues, 0.0))

        k = len(sigmas)
        euclidean_k = EuclideanSpace(k)

        U_tilde_op = LinearOperator.from_matrix(euclidean_k, euclidean_k, U_tilde)
        u_op = Q_op @ U_tilde_op

        sigma_op = DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            euclidean_k, euclidean_k, sigmas
        )

        inv_sigmas = np.zeros_like(sigmas)
        valid = sigmas > 1e-12
        inv_sigmas[valid] = 1.0 / sigmas[valid]
        sigma_inv_op = DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            euclidean_k, euclidean_k, inv_sigmas
        )

        v_op = B.adjoint @ U_tilde_op @ sigma_inv_op

        return cls(u_op, sigma_op, v_op)


class LowRankEig(LinearOperator):
    """
    A LinearOperator representing the eigendecomposition: A ≈ U @ D @ U*.

    This class encapsulates the components of an Eigendecomposition for a
    self-adjoint operator, allowing it to act as a `LinearOperator` while
    exposing the eigenvectors and eigenvalues.
    """

    def __init__(
        self,
        u_op: LinearOperator,
        d_op: DiagonalSparseMatrixLinearOperator,
    ):
        """
        Initializes the LowRankEig operator.

        Args:
            u_op (LinearOperator): The eigenvectors operator (U).
            d_op (DiagonalSparseMatrixLinearOperator): The diagonal operator of eigenvalues (D).
        """
        self._u_op = u_op
        self._d_op = d_op
        self._rank = d_op.domain.dim

        full_op = self.u_factor @ self.d_factor @ self.u_factor.adjoint

        super().__init__(
            full_op.domain,
            full_op.codomain,
            full_op,
            adjoint_mapping=full_op.adjoint,
        )

    @property
    def rank(self) -> int:
        """int: The rank of the approximation."""
        return self._rank

    @property
    def u_factor(self) -> LinearOperator:
        """LinearOperator: The eigenvectors (U)."""
        return self._u_op

    @property
    def d_factor(self) -> DiagonalSparseMatrixLinearOperator:
        """DiagonalSparseMatrixLinearOperator: The diagonal matrix of eigenvalues (D)."""
        return self._d_op

    @property
    def eigenvalues(self) -> np.ndarray:
        """np.ndarray: A 1D array of the computed eigenvalues."""
        return self.d_factor.extract_diagonal()

    @classmethod
    def from_randomized(
        cls,
        operator: LinearOperator,
        size_estimate: int,
        *,
        measure: Optional[GaussianMeasure] = None,
        galerkin: bool = True,
        method: str = "variable",
        max_rank: Optional[int] = None,
        power: int = 2,
        rtol: float = 1e-4,
        block_size: int = 10,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> LowRankEig:
        """
        Computes the Eigendecomposition using a unified randomized range finder.

        Args:
            operator (LinearOperator): The self-adjoint operator to approximate.
            size_estimate (int): Target rank or initial block size.
            measure (GaussianMeasure, optional): Prior measure for drawing test vectors.
            galerkin (bool): Default True for Eig. Computes Galerkin representation on fallback.
            method (str): {'variable', 'fixed'}.
            max_rank (int, optional): Upper limit on rank for 'variable' method.
            power (int): Number of power iterations.
            rtol (float): Relative tolerance for 'variable' method.
            block_size (int): Samples per iteration.
            parallel (bool): Parallelize the sampling/multiplication.
            n_jobs (int): CPU cores to utilize.

        Returns:
            LowRankEig: An instantiated operator containing the U and D factors.

        Raises:
            ValueError: If the operator is not an automorphism (domain != codomain).
        """
        if operator.domain != operator.codomain:
            raise ValueError("Eigendecomposition requires an automorphism.")

        Q_op = random_range(
            operator,
            size_estimate,
            measure=measure,
            galerkin=galerkin,
            method=method,
            max_rank=max_rank,
            power=power,
            rtol=rtol,
            block_size=block_size,
            parallel=parallel,
            n_jobs=n_jobs,
        )

        B = Q_op.adjoint @ operator @ Q_op
        B_dense = B.matrix(dense=True)

        eigenvalues, U_tilde = scipy.linalg.eigh(B_dense)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        U_tilde = U_tilde[:, idx]

        k = len(eigenvalues)
        euclidean_k = EuclideanSpace(k)

        U_tilde_op = LinearOperator.from_matrix(euclidean_k, euclidean_k, U_tilde)
        u_op = Q_op @ U_tilde_op

        d_op = DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            euclidean_k, euclidean_k, eigenvalues
        )

        return cls(u_op, d_op)


class LowRankCholesky(LinearOperator):
    """
    A LinearOperator representing the Cholesky-like factorization: A ≈ L @ L*.

    This class provides a memory-efficient low-rank Cholesky decomposition
    of a positive semi-definite operator, highly useful for drawing samples
    from Gaussian measures.
    """

    def __init__(self, l_op: LinearOperator):
        """
        Initializes the LowRankCholesky operator.

        Args:
            l_op (LinearOperator): The Cholesky factor (L) such that A ≈ L @ L*.
        """
        self._l_op = l_op
        self._rank = l_op.domain.dim
        full_op = self.l_factor @ self.l_factor.adjoint

        super().__init__(
            full_op.domain,
            full_op.codomain,
            full_op,
            adjoint_mapping=full_op.adjoint,
        )

    @property
    def rank(self) -> int:
        """int: The rank of the approximation."""
        return self._rank

    @property
    def l_factor(self) -> LinearOperator:
        """LinearOperator: The Cholesky factor (L)."""
        return self._l_op

    @classmethod
    def from_randomized(
        cls,
        operator: LinearOperator,
        size_estimate: int,
        *,
        measure: Optional[GaussianMeasure] = None,
        galerkin: bool = True,
        method: str = "variable",
        max_rank: Optional[int] = None,
        power: int = 2,
        rtol: float = 1e-4,
        block_size: int = 10,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> LowRankCholesky:
        """
        Computes a robust approximate Cholesky factorization via randomized range finding.

        Attempts a direct dense Cholesky factorization on the projected core matrix.
        If it fails (due to numerical precision issues), it safely falls back to
        an Eigendecomposition-based square root.

        Args:
            operator (LinearOperator): Positive semi-definite operator to factorize.
            size_estimate (int): Target rank or initial block size.
            measure (GaussianMeasure, optional): Prior measure for drawing test vectors.
            galerkin (bool): Default True. Computes Galerkin representation on fallback.
            method (str): {'variable', 'fixed'}.
            max_rank (int, optional): Upper limit on rank for 'variable' method.
            power (int): Number of power iterations.
            rtol (float): Relative tolerance for 'variable' method.
            block_size (int): Samples per iteration.
            parallel (bool): Parallelize the sampling/multiplication.
            n_jobs (int): CPU cores to utilize.

        Returns:
            LowRankCholesky: An instantiated operator containing the L factor.

        Raises:
            ValueError: If the operator is not an automorphism.
        """
        if operator.domain != operator.codomain:
            raise ValueError("Cholesky requires an automorphism.")

        Q_op = random_range(
            operator,
            size_estimate,
            measure=measure,
            galerkin=galerkin,
            method=method,
            max_rank=max_rank,
            power=power,
            rtol=rtol,
            block_size=block_size,
            parallel=parallel,
            n_jobs=n_jobs,
        )

        B = Q_op.adjoint @ operator @ Q_op
        B_dense = B.matrix(dense=True)
        k = B_dense.shape[0]

        try:
            L = scipy.linalg.cholesky(B_dense, lower=True)
            L_inv = scipy.linalg.solve_triangular(L, np.eye(k), lower=True)
            transform_matrix = L_inv.T
        except scipy.linalg.LinAlgError:
            eigenvalues, U_tilde = scipy.linalg.eigh(B_dense)
            max_eig = eigenvalues[-1] if len(eigenvalues) > 0 else 0
            threshold = 1e-12 * max_eig if max_eig > 0 else 0
            safe_eigs = np.where(eigenvalues >= threshold, eigenvalues, 0.0)
            inv_sqrt_eigs = np.where(safe_eigs > 0, 1.0 / np.sqrt(safe_eigs), 0.0)
            transform_matrix = U_tilde @ np.diag(inv_sqrt_eigs)

        euclidean_k = EuclideanSpace(k)
        transform_op = LinearOperator.from_matrix(
            euclidean_k, euclidean_k, transform_matrix
        )

        l_op = operator @ Q_op @ transform_op

        return cls(l_op)


# =====================================================================
#                      Unified Range Finder
# =====================================================================


def random_range(
    operator: LinearOperator,
    size_estimate: int,
    /,
    *,
    measure: Optional[GaussianMeasure] = None,
    galerkin: bool = False,
    method: str = "variable",
    max_rank: Optional[int] = None,
    power: int = 2,
    rtol: float = 1e-4,
    block_size: int = 10,
    parallel: bool = False,
    n_jobs: int = -1,
) -> LinearOperator:
    """
    Unified random range finder acting as an architectural bridge.

    If a `GaussianMeasure` is provided, it draws abstract structured samples
    to respect Hilbert space geometries and mass matrices. If no measure is
    provided, it routes to high-performance component-based representations
    via SciPy `LinearOperator`s.

    Args:
        operator (LinearOperator): The linear operator whose range is to be approximated.
        size_estimate (int): Target rank ('fixed') or initial sample size ('variable').
        measure (GaussianMeasure, optional): Measure to draw test samples from.
        galerkin (bool): If True, uses the Galerkin representation for the component fallback.
        method (str): {'variable', 'fixed'}. Algorithm choice.
        max_rank (int, optional): Hard limit on rank for variable sampling.
        power (int): Number of power iterations to enhance singular value decay.
        rtol (float): Relative tolerance for convergence checking.
        block_size (int): Size of new sample batches.
        parallel (bool): Parallelize computations where applicable.
        n_jobs (int): CPU cores to use.

    Returns:
        LinearOperator: The isometry Q mapping from Euclidean(k) into the codomain.
    """

    # --- PATH A: Abstract, Measure-Based (Matrix-Free) ---
    if measure is not None:
        if method == "variable":
            q_basis = _abstract_variable_rank_random_range(
                operator,
                measure,
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
            q_basis = _abstract_fixed_rank_random_range(
                operator,
                measure,
                size_estimate,
                power=power,
                parallel=parallel,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from 'fixed' or 'variable'."
            )

        return LinearOperator.from_vectors(operator.codomain, q_basis).adjoint

    # --- PATH B: Component-Based Fallback ---
    else:
        matrix_repr = operator.matrix(
            dense=False, galerkin=galerkin, parallel=parallel, n_jobs=n_jobs
        )

        if method == "variable":
            q_matrix = _component_variable_rank_random_range(
                matrix_repr,
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
            q_matrix = _component_fixed_rank_random_range(
                matrix_repr,
                size_estimate,
                power=power,
                parallel=parallel,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from 'fixed' or 'variable'."
            )

        k = q_matrix.shape[1]
        euclidean_k = EuclideanSpace(k)

        return LinearOperator.from_matrix(
            euclidean_k, operator.codomain, q_matrix, galerkin=False
        )


# =====================================================================
#             Internal Abstract (Measure-Based) Routines
# =====================================================================


def _abstract_fixed_rank_random_range(
    operator: LinearOperator,
    measure: GaussianMeasure,
    rank: int,
    /,
    *,
    power: int = 0,
    parallel: bool = False,
    n_jobs: int = -1,
) -> List[Vector]:
    """
    Internal. Computes an abstract matrix-free fixed-rank range approximation.

    Draws exactly `rank` samples from the measure and applies Modified Gram-Schmidt.
    """
    if operator.domain != measure.domain:
        raise ValueError("The measure must be defined on the operator's domain.")

    omega_samples = measure.samples(rank, parallel=parallel, n_jobs=n_jobs)
    y_vectors = [operator(w) for w in omega_samples]
    q_basis = operator.codomain.gram_schmidt(y_vectors)

    for _ in range(power):
        z_vectors = [operator.adjoint(q) for q in q_basis]
        z_basis = operator.domain.gram_schmidt(z_vectors)
        y_vectors = [operator(z) for z in z_basis]
        q_basis = operator.codomain.gram_schmidt(y_vectors)

    return q_basis


def _abstract_variable_rank_random_range(
    operator: LinearOperator,
    measure: GaussianMeasure,
    initial_rank: int,
    /,
    *,
    max_rank: Optional[int] = None,
    power: int = 0,
    block_size: int = 10,
    rtol: float = 1e-4,
    parallel: bool = False,
    n_jobs: int = -1,
) -> List[Vector]:
    """
    Internal. Computes an abstract matrix-free variable-rank range approximation.

    Progressively samples blocks of random vectors from the measure until the
    residual projection falls below the defined relative tolerance.
    """
    if operator.domain != measure.domain:
        raise ValueError("The measure must be defined on the operator's domain.")

    if max_rank is None:
        max_rank = min(operator.domain.dim, operator.codomain.dim)

    q_basis = _abstract_fixed_rank_random_range(
        operator, measure, initial_rank, power=power, parallel=parallel, n_jobs=n_jobs
    )

    tol = None

    while len(q_basis) < max_rank:
        test_samples = measure.samples(block_size, parallel=parallel, n_jobs=n_jobs)
        y_test = [operator(w) for w in test_samples]

        if tol is None:
            norms_sq = [operator.codomain.squared_norm(y) for y in y_test]
            norm_estimate = np.sqrt(sum(norms_sq)) / np.sqrt(block_size)
            tol = rtol * norm_estimate

        residuals = []
        max_err = 0.0

        for y in y_test:
            y_proj = operator.codomain.zero
            for q in q_basis:
                proj = operator.codomain.inner_product(q, y)
                operator.codomain.axpy(proj, q, y_proj)

            res = operator.codomain.subtract(y, y_proj)
            err = operator.codomain.norm(res)
            max_err = max(max_err, err)

            if err > 1e-12:
                residuals.append(res)

        if max_err < tol:
            break
        if not residuals:
            break

        try:
            new_basis = operator.codomain.gram_schmidt(residuals)
            cols_to_add = min(len(new_basis), max_rank - len(q_basis))
            q_basis.extend(new_basis[:cols_to_add])
        except ValueError:
            break

    return q_basis


# =====================================================================
#             Internal Component-Based Routines
# =====================================================================


def _component_fixed_rank_random_range(
    matrix: MatrixLike,
    rank: int,
    *,
    power: int = 0,
    parallel: bool = False,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Internal. Computes a fixed-rank approximation of a matrix's range.

    Generates standard normal component arrays and uses SciPy QR decomposition
    for fast Euclidean subspace extraction.
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


def _component_variable_rank_random_range(
    matrix: MatrixLike,
    initial_rank: int,
    *,
    max_rank: Optional[int] = None,
    power: int = 0,
    block_size: int = 10,
    rtol: float = 1e-4,
    parallel: bool = False,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Internal. Computes a variable-rank orthonormal basis via progressive sampling.

    Applies standard Hutchinson-style block updates using component representations.
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

    # Power Iterations on initial sample
    for _ in range(power):
        ys, _ = qr(ys, mode="economic")
        if parallel:
            ys_tilde = parallel_mat_mat(matrix.T, ys, n_jobs)
            ys = parallel_mat_mat(matrix, ys_tilde, n_jobs)
        else:
            ys_tilde = matrix.T @ ys
            ys = matrix @ ys_tilde

    basis_vectors, _ = qr(ys, mode="economic")
    tol = None

    while basis_vectors.shape[1] < max_rank:
        test_vectors = np.random.randn(n, block_size)
        if parallel:
            y_test = parallel_mat_mat(matrix, test_vectors, n_jobs)
        else:
            y_test = matrix @ test_vectors

        if tol is None:
            norm_estimate = np.linalg.norm(y_test) / np.sqrt(block_size)
            tol = rtol * norm_estimate

        residual = y_test - basis_vectors @ (basis_vectors.T @ y_test)
        error = np.linalg.norm(residual, ord=2)

        if error < tol:
            break

        new_basis, _ = qr(residual, mode="economic")
        cols_to_add = min(new_basis.shape[1], max_rank - basis_vectors.shape[1])

        if cols_to_add <= 0:
            break

        basis_vectors = np.hstack([basis_vectors, new_basis[:, :cols_to_add]])

    return basis_vectors


# =====================================================================
#             Specialized Matrix-Based Diagonal Estimator
# =====================================================================


def random_diagonal(
    matrix: MatrixLike,
    size_estimate: int,
    /,
    *,
    method: str = "variable",
    use_rademacher: bool = False,
    max_samples: Optional[int] = None,
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

    Note: This is a specialized, component-based implementation relying on
    element-wise array multiplication.

    Args:
        matrix: The (n, n) matrix or LinearOperator to analyze.
        size_estimate: For 'fixed' method, the exact target rank. For 'variable'
                       method, this is the initial rank to sample.
        method ({'variable', 'fixed'}): The algorithm to use.
            - 'variable': Progressively samples to meet tolerance `rtol`.
            - 'fixed': Returns an estimate based on exactly `size_estimate` samples.
        use_rademacher: If true, draw components from [-1,1]. Default method draws
            normally distributed components.
        max_samples: For 'variable' method, a hard limit on the number of samples.
                     Ignored if method='fixed'. Defaults to dimension of matrix.
        rtol: Relative tolerance for the 'variable' method.
        block_size: Number of new vectors to sample per iteration in 'variable' method.
        parallel: Whether to use parallel matrix multiplication.
        n_jobs: Number of jobs for parallelism.

    Returns:
        np.ndarray: A 1D numpy array of size n containing the approximate diagonal.
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

    if method == "fixed" or num_samples >= max_samples:
        return diag_estimate

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
                break

        # Update sums and counts for next iteration
        diag_sum += new_diag_sum
        num_samples = total_samples

    return diag_estimate


def deflated_diagonal(
    operator: LinearOperator,
    rank: int,
    size_estimate: int,
    /,
    *,
    method: str = "variable",
    use_rademacher: bool = True,
    max_samples: Optional[int] = None,
    rtol: float = 1e-2,
    block_size: int = 10,
    galerkin: bool = False,
    parallel: bool = False,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Estimates the diagonal of a square operator's component matrix using SVD deflation.

    This combines a deterministic low-rank diagonal approximation (via SVD)
    with a stochastic estimate of the residual diagonal (via Hutchinson's).

    Args:
        operator: The square LinearOperator to analyze.
        rank: The rank of the deterministic SVD deflation.
        size_estimate: Initial number of samples for the stochastic residual.
        method: 'variable' or 'fixed' for the stochastic residual phase.
        use_rademacher: If True, uses [-1, 1] Rademacher noise for Hutchinson's.
        max_samples: Hard limit on residual samples.
        rtol: Relative tolerance for the stochastic residual phase.
        block_size: Samples added per iteration in the stochastic phase.
        galerkin: If True, computes the diagonal of the Galerkin matrix.
        parallel: Whether to compute operations in parallel.
        n_jobs: Number of CPU cores to utilize.

    Returns:
        np.ndarray: A 1D array representing the diagonal of the operator.
    """
    if operator.domain.dim != operator.codomain.dim:
        raise ValueError("Operator must be square to extract a diagonal.")

    if rank < 0 or size_estimate < 0:
        raise ValueError("Rank and size_estimate must be non-negative.")

    dim = operator.domain.dim
    total_diagonal = np.zeros(dim)

    # -------------------------------------------------------------
    # 1. Deterministic Low-Rank Diagonal via SVD
    # -------------------------------------------------------------
    if rank > 0:
        svd_op = LowRankSVD.from_randomized(
            operator,
            rank,
            method="fixed",
            galerkin=galerkin,
            parallel=parallel,
            n_jobs=n_jobs,
        )

        U_mat = svd_op.u_factor.matrix(dense=True, galerkin=galerkin)
        V_mat = svd_op.v_factor.matrix(dense=True, galerkin=galerkin)
        S_vec = svd_op.singular_values

        # Diag(U @ S @ V^T)_i = sum_k U_{ik} S_k V_{ik}
        total_diagonal += np.sum(U_mat * S_vec * V_mat, axis=1)
        residual_op = operator - svd_op
    else:
        residual_op = operator

    # -------------------------------------------------------------
    # 2. Stochastic Residual Diagonal
    # -------------------------------------------------------------
    if size_estimate > 0:
        scipy_residual_wrapper = residual_op.matrix(galerkin=galerkin)

        stochastic_diag = random_diagonal(
            scipy_residual_wrapper,
            size_estimate,
            method=method,
            use_rademacher=use_rademacher,
            max_samples=max_samples,
            rtol=rtol,
            block_size=block_size,
            parallel=parallel,
            n_jobs=n_jobs,
        )

        total_diagonal += stochastic_diag

    return total_diagonal
