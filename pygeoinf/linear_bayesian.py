"""
Implements the Bayesian framework for solving linear inverse problems.

This module treats the inverse problem from a statistical perspective, aiming to
determine the full posterior probability distribution of the unknown model
parameters, rather than a single best-fit solution.

Key Classes
-----------
- `LinearBayesianInversion`: Computes the posterior Gaussian measure `p(u|d)`
  for the model `u` given observed data `d`.
- `ConstrainedLinearBayesianInversion`: Solves the inverse problem subject to
  an affine constraint `u in A`.
"""

from __future__ import annotations
from typing import Optional, List

from joblib import Parallel, delayed
import numpy as np

import scipy.sparse as sps
import scipy.sparse.linalg as splinalg

from .inversion import LinearInversion
from .gaussian_measure import GaussianMeasure
from .forward_problem import LinearForwardProblem
from .linear_operators import LinearOperator, DiagonalSparseMatrixLinearOperator
from .linear_solvers import LinearSolver, IterativeLinearSolver
from .hilbert_space import Vector, EuclideanSpace


class LinearBayesianInversion(LinearInversion):
    """
    Solves a linear inverse problem using Bayesian methods.

    This class applies to problems of the form `d = A(u) + e`. It computes the
    full posterior probability distribution `p(u|d)`.
    """

    def __init__(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        /,
    ) -> None:
        """
        Args:
            forward_problem: The forward problem linking the model to the data.
            model_prior_measure: The prior Gaussian measure on the model space.
        """
        super().__init__(forward_problem)
        self._model_prior_measure: GaussianMeasure = model_prior_measure

    @property
    def model_prior_measure(self) -> GaussianMeasure:
        """The prior Gaussian measure on the model space."""
        return self._model_prior_measure

    @property
    def normal_operator(self) -> LinearOperator:
        """
        Returns the Bayesian Normal operator: N = A Q A* + R.
        """
        forward_operator = self.forward_problem.forward_operator
        model_prior_covariance = self.model_prior_measure.covariance

        if self.forward_problem.data_error_measure_set:
            return (
                forward_operator @ model_prior_covariance @ forward_operator.adjoint
                + self.forward_problem.data_error_measure.covariance
            )
        else:
            return forward_operator @ model_prior_covariance @ forward_operator.adjoint

    def kalman_operator(
        self,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[LinearOperator] = None,
    ) -> LinearOperator:
        """
        Returns the Kalman gain operator K = Q A* N^-1.
        """
        forward_operator = self.forward_problem.forward_operator
        model_prior_covariance = self.model_prior_measure.covariance
        normal_operator = self.normal_operator

        if isinstance(solver, IterativeLinearSolver):
            inverse_normal_operator = solver(
                normal_operator, preconditioner=preconditioner
            )
        else:
            inverse_normal_operator = solver(normal_operator)

        return (
            model_prior_covariance @ forward_operator.adjoint @ inverse_normal_operator
        )

    def model_posterior_measure(
        self,
        data: Vector,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[LinearOperator] = None,
    ) -> GaussianMeasure:
        """
        Returns the posterior Gaussian measure p(u|d).

        Args:
            data: The observed data vector.
            solver: A linear solver for inverting the normal operator.
            preconditioner: An optional preconditioner.
        """
        data_space = self.data_space
        model_space = self.model_space
        forward_operator = self.forward_problem.forward_operator
        model_prior_covariance = self.model_prior_measure.covariance

        # 1. Compute Kalman Gain
        kalman_gain = self.kalman_operator(solver, preconditioner=preconditioner)

        # 2. Compute Posterior Mean
        # Shift data: d - A(mu_u)
        if self.model_prior_measure.has_zero_expectation:
            shifted_data = data_space.copy(data)
        else:
            shifted_data = data_space.subtract(
                data, forward_operator(self.model_prior_measure.expectation)
            )

        # Shift for noise mean: d - A(mu_u) - mu_e
        if self.forward_problem.data_error_measure_set:
            if not self.forward_problem.data_error_measure.has_zero_expectation:
                error_expectation = self.forward_problem.data_error_measure.expectation
                shifted_data = data_space.subtract(shifted_data, error_expectation)

        mean_update = kalman_gain(shifted_data)

        if self.model_prior_measure.has_zero_expectation:
            expectation = mean_update
        else:
            expectation = model_space.add(
                self.model_prior_measure.expectation, mean_update
            )

        # 3. Compute Posterior Covariance (Implicitly)
        # C_post = C_u - K A C_u
        covariance = model_prior_covariance - (
            kalman_gain @ forward_operator @ model_prior_covariance
        )

        # 4. Set up Posterior Sampling
        # Logic: Can sample if prior is samplable AND (noise is absent OR samplable)
        can_sample_prior = self.model_prior_measure.sample_set
        can_sample_noise = (
            not self.forward_problem.data_error_measure_set
            or self.forward_problem.data_error_measure.sample_set
        )

        if can_sample_prior and can_sample_noise:

            def sample():
                # a. Sample Prior (u)
                model_sample = self.model_prior_measure.sample()

                # b. Calculate deterministic residual (v - Bu)
                prediction = forward_operator(model_sample)
                data_residual = data_space.subtract(data, prediction)

                # c. Subtract full noise sample (v - Bu - z)
                if self.forward_problem.data_error_measure_set:
                    noise_sample = self.forward_problem.data_error_measure.sample()
                    data_residual = data_space.subtract(data_residual, noise_sample)

                # d. Update with Kalman gain (u + K * residual)
                correction = kalman_gain(data_residual)
                return model_space.add(model_sample, correction)

            return GaussianMeasure(
                covariance=covariance, expectation=expectation, sample=sample
            )
        else:
            return GaussianMeasure(covariance=covariance, expectation=expectation)

    def diagonal_normal_preconditioner(
        self,
        /,
        *,
        blocks: Optional[List[List[int]]] = None,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> LinearOperator:
        """
        Constructs a diagonal preconditioner for the Bayesian normal operator
        (A Q A* + R) using an optimized formulation.

        This exploits the identity <v, A Q A* v> = <A* v, Q A* v>. If blocks
        of data indices are provided, it acts on the averaged basis vector
        for each block to compute a robust representative regional variance,
        requiring only one adjoint action of the forward operator per block.

        Args:
            blocks: An optional list of lists, where each sub-list contains indices
                of data points grouped together. Must perfectly partition the data space.
            parallel: If True, computes the adjoint actions in parallel.
            n_jobs: Number of parallel jobs to use. -1 means all available cores.

        Returns:
            A DiagonalSparseMatrixLinearOperator representing the inverse of
            the approximated normal operator.
        """

        data_space = self.data_space
        model_space = self.model_space
        A_adj = self.forward_problem.forward_operator.adjoint
        Q = self.model_prior_measure.covariance
        data_dim = data_space.dim

        if blocks is not None:
            flattened_indices = [idx for block in blocks for idx in block]

            if (
                len(flattened_indices) != data_dim
                or len(set(flattened_indices)) != data_dim
            ):
                raise ValueError(
                    f"The provided blocks must exactly partition the data space. "
                    f"Expected {data_dim} unique indices, but got {len(flattened_indices)} "
                    f"total indices with {len(set(flattened_indices))} unique."
                )

            if min(flattened_indices) < 0 or max(flattened_indices) >= data_dim:
                raise ValueError("Block indices are out of bounds for the data space.")

            blocks_to_compute = blocks
        else:
            blocks_to_compute = [[i] for i in range(data_dim)]

        def compute_block_aqa_diag(block: List[int]) -> float:
            """Worker function to compute the representative variance for a block."""
            n = len(block)

            # Form the averaged basis vector v = (1/n) * sum(e_i)
            c = np.zeros(data_dim)
            c[block] = 1.0 / n
            v = data_space.from_components(c)
            f = A_adj(v)
            Q_f = Q(f)
            return model_space.inner_product(f, Q_f)

        if parallel:
            computed_vals = Parallel(n_jobs=n_jobs)(
                delayed(compute_block_aqa_diag)(block) for block in blocks_to_compute
            )
        else:
            computed_vals = [
                compute_block_aqa_diag(block) for block in blocks_to_compute
            ]

        aqa_diag = np.zeros(data_dim)
        for block, val in zip(blocks_to_compute, computed_vals):
            aqa_diag[block] = val

        if self.forward_problem.data_error_measure_set:
            r_diag = (
                self.forward_problem.data_error_measure.covariance.extract_diagonal(
                    galerkin=True, parallel=parallel, n_jobs=n_jobs
                )
            )
            normal_diag = aqa_diag + r_diag
        else:
            normal_diag = aqa_diag

        approx_normal_op = DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            data_space, data_space, normal_diag, galerkin=True
        )

        return approx_normal_op.inverse

    def sparse_localized_preconditioner(
        self,
        interacting_blocks: list[list[int]],
        rank: int = 10,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> LinearOperator:
        """
        Builds a sparse preconditioner for the Bayesian normal equations using
        randomized Nystrom approximations on localized, potentially overlapping sub-blocks.

        Args:
            interacting_blocks: A list of lists, where each sub-list contains the
                                indices of data points that strongly couple to each other.
            rank: The rank of the randomized Nystrom approximation to use per block.
            parallel: If True, computes the sub-block approximations in parallel.
            n_jobs: Number of CPU cores to use if parallel=True (-1 uses all cores).
        """

        forward_op = self.forward_problem.forward_operator
        prior_cov = self.model_prior_measure.covariance
        data_space = forward_op.codomain

        noise_variance = (
            self.forward_problem.data_error_measure.covariance.extract_diagonal(
                parallel=parallel, n_jobs=n_jobs, galerkin=True
            )
            if self.forward_problem.data_error_measure_set
            else np.zeros(self.data_space.dim)
        )

        core_normal_op = forward_op @ prior_cov @ forward_op.adjoint

        data_dim = data_space.dim
        euclidean_full = EuclideanSpace(data_dim)
        to_components_op = data_space.coordinate_projection

        print(
            f"Building sparse preconditioner over {len(interacting_blocks)} blocks (Rank {rank})..."
        )

        # 2. Define the worker function for a single block
        def _process_block(indices: list[int]):
            block_size = len(indices)
            idx_array = np.array(indices)

            # Restrict to this specific block of coordinates
            restrict_coords = euclidean_full.subspace_projection(indices)
            P = restrict_coords @ to_components_op
            P_star = P.adjoint

            # Localized Normal Operator: H_k = P * (A Q A*) * P*
            H_local = P @ core_normal_op @ P_star

            # Randomized Nystrom Approximation
            actual_rank = min(rank, block_size)

            # Unpack correctly: (eigenvectors, eigenvalues)
            evecs_op, evals_op = H_local.random_eig(actual_rank, method="fixed")

            # 1. Extract the dense eigenvector matrix natively (shape: block_size x actual_rank)
            V = evecs_op.matrix(dense=True, galerkin=True)

            # 2. Extract the 1D array of eigenvalues
            # Extracting the dense matrix of the diagonal operator and grabbing its diagonal
            Lambda = evals_op.matrix(dense=True).diagonal()

            # 3. Reconstruct the dense sub-block matrix purely in NumPy
            # (V * Lambda) efficiently broadcasts the 1D eigenvalues across the columns of V
            M_local_array = (V * Lambda) @ V.T

            # 4. Extract global coordinates for the sparse COO format
            I_local, J_local = np.meshgrid(
                np.arange(block_size), np.arange(block_size), indexing="ij"
            )
            I_global = idx_array[I_local.flatten()]
            J_global = idx_array[J_local.flatten()]

            V_flattened = M_local_array.flatten()

            return I_global, J_global, V_flattened

        # 3. Execute the block computations (Sequential or Parallel)
        row_indices, col_indices, values = [], [], []

        if parallel:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_process_block)(indices) for indices in interacting_blocks
            )
            for I_glob, J_glob, V_flat in results:
                row_indices.extend(I_glob)
                col_indices.extend(J_glob)
                values.extend(V_flat)
        else:
            for indices in interacting_blocks:
                I_glob, J_glob, V_flat = _process_block(indices)
                row_indices.extend(I_glob)
                col_indices.extend(J_glob)
                values.extend(V_flat)

        # 4. Assemble the sparse normal matrix
        H_sparse = sps.coo_matrix(
            (values, (row_indices, col_indices)), shape=(data_dim, data_dim)
        )

        # 5. Add the data noise covariance (R) to the diagonal
        R_sparse = sps.diags(noise_variance)
        H_approx = (H_sparse + R_sparse).tocsc()

        # 6. Factorize using SuperLU
        print("Factorizing sparse matrix...")
        splu_solver = splinalg.splu(H_approx)

        # 7. Wrap the solver in a pygeoinf LinearOperator
        def apply_preconditioner(x):
            c = data_space.to_components(x)
            c_solved = splu_solver.solve(c)
            return data_space.from_components(c_solved)

        return LinearOperator(
            data_space,
            data_space,
            apply_preconditioner,
            adjoint_mapping=apply_preconditioner,
        )

    def surrogate_inversion(
        self,
        /,
        *,
        alternate_forward_operator: Optional[LinearOperator] = None,
        alternate_prior_measure: Optional[GaussianMeasure] = None,
        alternate_data_error_measure: Optional[GaussianMeasure] = None,
    ) -> LinearBayesianInversion:
        """
        Constructs a surrogate Bayesian inversion problem using simplified physics,
        priors, or data errors.

        This is primarily used to build robust, physics-based preconditioners.
        """

        A_tilde = alternate_forward_operator or self.forward_problem.forward_operator
        Q_tilde = alternate_prior_measure or self.model_prior_measure

        if alternate_data_error_measure is not None:
            R_tilde = alternate_data_error_measure
        elif self.forward_problem.data_error_measure_set:
            R_tilde = self.forward_problem.data_error_measure
        else:
            R_tilde = None

        if A_tilde.domain != Q_tilde.domain:
            raise ValueError(
                "The domain of the alternate forward operator must match "
                "the domain of the prior measure."
            )

        if A_tilde.codomain != self.data_space:
            raise ValueError(
                "The data space for the alternate forward operator must "
                "match that for the original problem"
            )

        if R_tilde.domain != self.data_space:
            raise ValueError(
                "The domain for the alternate error measure must "
                "match that for the original problem"
            )

        surrogate_forward_problem = LinearForwardProblem(
            A_tilde, data_error_measure=R_tilde
        )

        return LinearBayesianInversion(surrogate_forward_problem, Q_tilde)

    def surrogate_normal_preconditioner(
        self,
        solver: LinearSolver,
        /,
        *,
        alternate_forward_operator: Optional[LinearOperator] = None,
        alternate_prior_measure: Optional[GaussianMeasure] = None,
        alternate_data_error_measure: Optional[GaussianMeasure] = None,
    ) -> LinearOperator:
        """
        Builds a preconditioner by exactly inverting the normal operator of a
        simplified surrogate inverse problem.
        """
        # 1. Get the surrogate inverse problem
        surrogate_inv = self.surrogate_inversion(
            alternate_forward_operator=alternate_forward_operator,
            alternate_prior_measure=alternate_prior_measure,
            alternate_data_error_measure=alternate_data_error_measure,
        )

        # 2. Extract its normal operator and invert it using the provided solver
        return solver(surrogate_inv.normal_operator)

    def low_rank_surrogate(
        self,
        /,
        *,
        forward_rank: Optional[int] = None,
        prior_rank: Optional[int] = None,
        data_error_rank: Optional[int] = None,
        forward_kwargs: Optional[dict] = None,
        prior_kwargs: Optional[dict] = None,
        data_error_kwargs: Optional[dict] = None,
    ) -> LinearBayesianInversion:
        """
        Constructs a surrogate Bayesian inversion problem by replacing the exact
        physics and statistical measures with their low-rank approximations.

        This method is particularly useful for generating computationally cheap
        surrogate models to be used in constructing preconditioners for massive,
        ill-conditioned inverse problems (e.g., using spectral or banded methods).
        The low-rank approximations are computed using randomized algorithms.

        Args:
            forward_rank: The target rank for the randomized Singular Value
                Decomposition (SVD) of the forward operator. If None, the exact
                forward operator is retained.
            prior_rank: The target rank for the randomized eigendecomposition of
                the model prior measure. If None, the exact prior measure is retained.
            data_error_rank: The target rank for the randomized eigendecomposition
                of the data error measure. If None, the exact data error measure
                is retained.
            forward_kwargs: Additional keyword arguments passed directly to
                `LinearOperator.random_svd` (e.g., 'power', 'block_size', 'n_jobs').
            prior_kwargs: Additional keyword arguments passed directly to
                `GaussianMeasure.low_rank_approximation`.
            data_error_kwargs: Additional keyword arguments passed directly to
                `GaussianMeasure.low_rank_approximation`.

        Returns:
            LinearBayesianInversion: A new Bayesian inversion object configured
            with the specified low-rank surrogate components.

        Raises:
            ValueError: If `data_error_rank` is provided but the original forward
                problem does not have a data error measure explicitly set.
        """
        A_tilde = None
        Q_tilde = None
        R_tilde = None

        # 1. Low-rank forward operator via SVD
        if forward_rank is not None:
            f_kwargs = forward_kwargs or {}
            original_A = self.forward_problem.forward_operator
            L, D, R = original_A.random_svd(forward_rank, **f_kwargs)
            A_tilde = L @ D @ R

        # 2. Low-rank prior measure
        if prior_rank is not None:
            p_kwargs = prior_kwargs or {}
            Q_tilde = self.model_prior_measure.low_rank_approximation(
                prior_rank, **p_kwargs
            )

        # 3. Low-rank data error measure
        if data_error_rank is not None:
            if not self.forward_problem.data_error_measure_set:
                raise ValueError("Cannot approximate data error measure: none is set.")

            d_kwargs = data_error_kwargs or {}
            R_tilde = self.forward_problem.data_error_measure.low_rank_approximation(
                data_error_rank, **d_kwargs
            )

        # 4. Feed directly into the surrogate builder
        return self.surrogate_inversion(
            alternate_forward_operator=A_tilde,
            alternate_prior_measure=Q_tilde,
            alternate_data_error_measure=R_tilde,
        )
