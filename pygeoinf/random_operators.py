"""
Implements randomized matrix-free algorithms for operators on Hilbert spaces.

This module complements `random_matrix.py` by providing fully abstract,
matrix-free implementations of randomized factorizations (SVD, Cholesky,
Eigendecomposition). These algorithms operate entirely via operator composition,
adjoint mappings, and the intrinsic geometry (inner products and norms) of the
underlying `HilbertSpace` objects.

By drawing structured samples from a `GaussianMeasure`, these methods
inherently respect mass matrices and continuous function space geometries
without ever requiring the extraction of a dense operator matrix.
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import warnings

import numpy as np
import scipy.linalg

from .hilbert_space import Vector, EuclideanSpace
from .linear_operators import LinearOperator, DiagonalSparseMatrixLinearOperator
from .gaussian_measure import GaussianMeasure


class LowRankSVD(LinearOperator):
    """
    A LinearOperator representing the low-rank SVD: A ≈ U @ Sigma @ V*.

    This class behaves like a standard LinearOperator but stores its
    factors internally. It can be constructed directly from pre-computed
    factors or generated via the `from_randomized` class method.
    """

    def __init__(
        self,
        u_op: LinearOperator,
        sigma_op: DiagonalSparseMatrixLinearOperator,
        v_star_op: LinearOperator,
    ):
        """
        Initializes the SVD operator from its exact factors.

        Args:
            u_op: The left singular vector expansion operator (U).
            sigma_op: The diagonal singular value operator (Sigma).
            v_star_op: The right singular vector projection operator (V*).
        """
        self._u_op = u_op
        self._sigma_op = sigma_op
        self._v_star_op = v_star_op
        self._rank = sigma_op.domain.dim

        # Compose the operators using pygeoinf's standard algebra
        full_op = self._u_op @ self._sigma_op @ self._v_star_op

        super().__init__(
            full_op.domain,
            full_op.codomain,
            full_op,
            adjoint_mapping=full_op.adjoint,
        )

    @property
    def rank(self) -> int:
        """The rank of the factorization."""
        return self._rank

    @property
    def u_factor(self) -> LinearOperator:
        """The left singular vector operator (U)."""
        return self._u_op

    @property
    def singular_values(self) -> np.ndarray:
        """The singular values (diagonal of Sigma) as a NumPy array."""
        return self._sigma_op.extract_diagonal()

    @property
    def v_factor(self) -> LinearOperator:
        """The right singular vector operator (V)."""
        return self._v_star_op.adjoint

    @property
    def v_star_factor(self) -> LinearOperator:
        """The right singular vector adjoint operator (V*)."""
        return self._v_star_op

    @classmethod
    def from_randomized(
        cls,
        operator: LinearOperator,
        measure: GaussianMeasure,
        size_estimate: int,
        *,
        method: str = "variable",
        max_rank: Optional[int] = None,
        power: int = 2,
        rtol: float = 1e-4,
        block_size: int = 10,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> LowRankSVD:
        """
        Computes the SVD using an abstract randomized range finder.
        """
        q_basis = random_range(
            operator,
            measure,
            size_estimate,
            method=method,
            max_rank=max_rank,
            power=power,
            rtol=rtol,
            block_size=block_size,
            parallel=parallel,
            n_jobs=n_jobs,
        )

        k = len(q_basis)
        euclidean_k = EuclideanSpace(k)

        # Q* maps Codomain -> Euclidean(k)
        Q_star = LinearOperator.from_vectors(operator.codomain, q_basis)
        B = Q_star @ operator

        # Safely extract the Gram matrix to avoid metric distortions
        M_op = B @ B.adjoint
        M_dense = M_op.matrix(dense=True)

        eigenvalues, U_tilde = scipy.linalg.eigh(M_dense)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        U_tilde = U_tilde[:, idx]
        sigmas = np.sqrt(np.maximum(eigenvalues, 0.0))

        u_vectors = []
        for j in range(k):
            u_j = operator.codomain.zero
            for i in range(k):
                operator.codomain.axpy(U_tilde[i, j], q_basis[i], u_j)
            u_vectors.append(u_j)

        v_vectors = []
        for j in range(k):
            v_j_unscaled = B.adjoint(U_tilde[:, j])
            if sigmas[j] > 1e-12:
                v_j = operator.domain.multiply(1.0 / sigmas[j], v_j_unscaled)
            else:
                v_j = operator.domain.zero
            v_vectors.append(v_j)

        u_op = LinearOperator.from_vectors(operator.codomain, u_vectors).adjoint
        sigma_op = DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            euclidean_k, euclidean_k, sigmas
        )
        v_star_op = LinearOperator.from_vectors(operator.domain, v_vectors)

        return cls(u_op, sigma_op, v_star_op)


class LowRankEig(LinearOperator):
    """
    A LinearOperator representing the eigendecomposition: A ≈ U @ D @ U*.
    """

    def __init__(
        self,
        u_op: LinearOperator,
        d_op: DiagonalSparseMatrixLinearOperator,
    ):
        """
        Initializes the eigendecomposition from exact factors.

        Args:
            u_op: The eigenvector expansion operator (U).
            d_op: The diagonal eigenvalue operator (D).
        """
        self._u_op = u_op
        self._d_op = d_op
        self._rank = d_op.domain.dim

        # Compose A = U @ D @ U*
        full_op = self._u_op @ self._d_op @ self._u_op.adjoint

        super().__init__(
            full_op.domain,
            full_op.codomain,
            full_op,
            adjoint_mapping=full_op.adjoint,
        )

    @property
    def rank(self) -> int:
        """The rank of the factorization."""
        return self._rank

    @property
    def eigenvectors(self) -> LinearOperator:
        """The expansion operator for eigenvectors (U)."""
        return self._u_op

    @property
    def eigenvalues(self) -> np.ndarray:
        """The eigenvalues (diagonal of D) as a NumPy array."""
        return self._d_op.extract_diagonal()

    @classmethod
    def from_randomized(
        cls,
        operator: LinearOperator,
        measure: GaussianMeasure,
        size_estimate: int,
        *,
        method: str = "variable",
        max_rank: Optional[int] = None,
        power: int = 2,
        rtol: float = 1e-4,
        block_size: int = 10,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> LowRankEig:
        """
        Computes the eigendecomposition using an abstract randomized range finder.
        """
        if operator.domain != operator.codomain:
            raise ValueError("Eigendecomposition requires an automorphism.")

        q_basis = random_range(
            operator,
            measure,
            size_estimate,
            method=method,
            max_rank=max_rank,
            power=power,
            rtol=rtol,
            block_size=block_size,
            parallel=parallel,
            n_jobs=n_jobs,
        )

        k = len(q_basis)
        euclidean_k = EuclideanSpace(k)

        Q_op = LinearOperator.from_vectors(operator.domain, q_basis)
        B = Q_op @ operator @ Q_op.adjoint
        B_dense = B.matrix(dense=True)

        eigenvalues, U_tilde = scipy.linalg.eigh(B_dense)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        U_tilde = U_tilde[:, idx]

        u_vectors = []
        for j in range(k):
            u_j = operator.domain.zero
            for i in range(k):
                operator.domain.axpy(U_tilde[i, j], q_basis[i], u_j)
            u_vectors.append(u_j)

        u_op = LinearOperator.from_vectors(operator.domain, u_vectors).adjoint
        d_op = DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            euclidean_k, euclidean_k, eigenvalues
        )

        return cls(u_op, d_op)


class LowRankCholesky(LinearOperator):
    """
    A LinearOperator representing the Cholesky-like factorization: A ≈ F @ F*.
    """

    def __init__(self, f_op: LinearOperator):
        """
        Initializes the Cholesky factorization from its factor.

        Args:
            f_op: The factor F such that A ≈ F @ F*.
        """
        self._f_op = f_op
        self._rank = f_op.domain.dim

        # Compose A = F @ F*
        full_op = self._f_op @ self._f_op.adjoint

        super().__init__(
            full_op.domain,
            full_op.codomain,
            full_op,
            adjoint_mapping=full_op.adjoint,
        )

    @property
    def rank(self) -> int:
        """The rank of the factorization."""
        return self._rank

    @property
    def cholesky_factor(self) -> LinearOperator:
        """The factor F such that A ≈ F @ F*."""
        return self._f_op

    @classmethod
    def from_randomized(
        cls,
        operator: LinearOperator,
        measure: GaussianMeasure,
        size_estimate: int,
        *,
        method: str = "variable",
        max_rank: Optional[int] = None,
        power: int = 2,
        rtol: float = 1e-4,
        block_size: int = 10,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> LowRankCholesky:
        """
        Computes the Cholesky factorization using a robust Nyström approximation.
        """
        if operator.domain != operator.codomain:
            raise ValueError("Cholesky requires an automorphism.")

        q_basis = random_range(
            operator,
            measure,
            size_estimate,
            method=method,
            max_rank=max_rank,
            power=power,
            rtol=rtol,
            block_size=block_size,
            parallel=parallel,
            n_jobs=n_jobs,
        )

        k = len(q_basis)
        Q_op = LinearOperator.from_vectors(operator.domain, q_basis)

        y_vectors = [operator(q) for q in q_basis]

        B = Q_op @ operator @ Q_op.adjoint
        B_dense = B.matrix(dense=True)

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

        f_vectors = []
        for j in range(k):
            f_j = operator.domain.zero
            for i in range(k):
                operator.domain.axpy(transform_matrix[i, j], y_vectors[i], f_j)
            f_vectors.append(f_j)

        # f_vectors are functions in the domain.
        # from_vectors maps Domain -> Euclidean.
        # Therefore, the adjoint maps Euclidean -> Domain (the true F factor).
        f_op = LinearOperator.from_vectors(operator.domain, f_vectors).adjoint

        return cls(f_op)


def fixed_rank_random_range(
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
    Computes an orthonormal basis for a fixed-rank approximation of an operator's range.

    This algorithm draws structured samples from a specified prior measure and maps
    them through the operator. The resulting vectors are orthogonalized using the
    native geometry of the codomain, perfectly preserving mass-matrix weightings.

    Args:
        operator (LinearOperator): The operator whose range is to be approximated.
        measure (GaussianMeasure): A structured measure defined on the operator's domain
            used to draw test samples, avoiding mesh-dependent white noise.
        rank (int): The exact number of basis vectors to compute.
        power (int, optional): The number of power iterations to perform. Power iterations
            amplify the dominant singular values but require additional forward and
            adjoint operator evaluations. Defaults to 0.
        parallel (bool, optional): If True, draws samples from the measure in parallel.
            Defaults to False.
        n_jobs (int, optional): Number of jobs for parallelism. Defaults to -1 (all cores).

    Returns:
        List[Vector]: A list of `Vector` objects forming an orthonormal basis
        for the approximate range in the codomain.

    Raises:
        ValueError: If the measure is not defined on the operator's domain.
    """
    if operator.domain != measure.domain:
        raise ValueError("The measure must be defined on the operator's domain.")

    # 1. Sketching Step
    omega_samples = measure.samples(rank, parallel=parallel, n_jobs=n_jobs)

    # 2. Sampling Step
    y_vectors = [operator(w) for w in omega_samples]

    # 3. Orthogonalization Step (Respects codomain geometry)
    q_basis = operator.codomain.gram_schmidt(y_vectors)

    # 4. Power Iterations
    for _ in range(power):
        z_vectors = [operator.adjoint(q) for q in q_basis]
        z_basis = operator.domain.gram_schmidt(z_vectors)
        y_vectors = [operator(z) for z in z_basis]
        q_basis = operator.codomain.gram_schmidt(y_vectors)

    return q_basis


def variable_rank_random_range(
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
    Computes a variable-rank orthonormal basis using progressive abstract sampling.

    The algorithm starts with an initial block of samples, checks for convergence
    by measuring the residual norms in the abstract codomain, and progressively
    adds new samples until the desired relative tolerance is met.

    Args:
        operator (LinearOperator): The operator whose range is to be approximated.
        measure (GaussianMeasure): The structured prior measure for sample generation.
        initial_rank (int): The number of vectors to sample in the first pass.
        max_rank (int, optional): A hard limit on the number of basis vectors.
            Defaults to min(domain.dim, codomain.dim).
        power (int, optional): Number of power iterations to improve the initial sample.
            Defaults to 0.
        block_size (int, optional): The number of new vectors to sample in each
            subsequent iteration. Defaults to 10.
        rtol (float, optional): Relative tolerance for determining the output rank based
            on abstract residual norms. Defaults to 1e-4.
        parallel (bool, optional): If True, draws samples in parallel. Defaults to False.
        n_jobs (int, optional): Number of jobs for parallelism. Defaults to -1.

    Returns:
        List[Vector]: A list of `Vector` objects forming an orthonormal basis that
        approximates the operator's range to the given tolerance.
    """
    if operator.domain != measure.domain:
        raise ValueError("The measure must be defined on the operator's domain.")

    if max_rank is None:
        max_rank = min(operator.domain.dim, operator.codomain.dim)

    # Initial sample and power iterations
    q_basis = fixed_rank_random_range(
        operator, measure, initial_rank, power=power, parallel=parallel, n_jobs=n_jobs
    )

    tol = None

    while len(q_basis) < max_rank:
        # Generate new test vectors
        test_samples = measure.samples(block_size, parallel=parallel, n_jobs=n_jobs)
        y_test = [operator(w) for w in test_samples]

        if tol is None:
            # Estimate spectral norm from the test block
            norms_sq = [operator.codomain.squared_norm(y) for y in y_test]
            norm_estimate = np.sqrt(sum(norms_sq)) / np.sqrt(block_size)
            tol = rtol * norm_estimate

        residuals = []
        max_err = 0.0

        for y in y_test:
            # Abstract projection: y_proj = sum(<q_i, y> q_i)
            y_proj = operator.codomain.zero
            for q in q_basis:
                proj = operator.codomain.inner_product(q, y)
                operator.codomain.axpy(proj, q, y_proj)

            # Calculate residual and error
            res = operator.codomain.subtract(y, y_proj)
            err = operator.codomain.norm(res)
            max_err = max(max_err, err)

            # Keep only non-zero residuals for the next basis expansion
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
            # Gram-Schmidt caught a linear dependency among the residuals.
            break

    return q_basis


def random_range(
    operator: LinearOperator,
    measure: GaussianMeasure,
    size_estimate: int,
    /,
    *,
    method: str = "variable",
    max_rank: Optional[int] = None,
    power: int = 2,
    rtol: float = 1e-4,
    block_size: int = 10,
    parallel: bool = False,
    n_jobs: int = -1,
) -> List[Vector]:
    """
    A unified wrapper for abstract randomized range finding algorithms.

    Args:
        operator (LinearOperator): The operator to analyze.
        measure (GaussianMeasure): The structured measure defined on the domain.
        size_estimate (int): For 'fixed' method, the exact target rank. For 'variable'
            method, the initial rank to sample.
        method (str, optional): The algorithm to use. 'variable' progressively samples
            to meet `rtol`. 'fixed' returns exactly `size_estimate` vectors.
            Defaults to "variable".
        max_rank (int, optional): A hard limit on the rank for the 'variable' method.
            Defaults to None.
        power (int, optional): Number of power iterations. Defaults to 2.
        rtol (float, optional): Relative tolerance for the 'variable' method.
            Defaults to 1e-4.
        block_size (int, optional): Number of new vectors to sample per iteration
            in the 'variable' method. Defaults to 10.
        parallel (bool, optional): If True, draws samples in parallel. Defaults to False.
        n_jobs (int, optional): Number of jobs for parallelism. Defaults to -1.

    Returns:
        List[Vector]: An orthonormal basis for the approximate range.
    """
    if method == "variable":
        return variable_rank_random_range(
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
        return fixed_rank_random_range(
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
