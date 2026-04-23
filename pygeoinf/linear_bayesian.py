"""
Implements the Bayesian framework for solving linear inverse problems.

This module treats the inverse problem from a statistical perspective. Rather
than seeking a single deterministic "best-fit" solution, it aims to determine
the full posterior probability distribution of the unknown model parameters
given the observed data, prior knowledge, and noise statistics.

A core feature of this module is its dual algebraic formalism, allowing users to
optimize computational efficiency based on the problem geometry:

- **data_space**: Assembles the data-space normal operator (size M x M, where M is
  the data dimension).
  Normal Operator: `N = A Q A* + R`
  Kalman Gain:     `K = Q A* N^-1`
  Best suited for underdetermined problems (M << N).

- **model_space**: Assembles the model-space normal operator (size N x N, where N is
  the model dimension).
  Normal Operator: `N = Q^-1 + A* R^-1 A`
  Kalman Gain:     `K = N^-1 A* R^-1`
  Best suited for overdetermined problems (N << M).

Key Classes
-----------
- `LinearBayesianInversion`: Computes the posterior Gaussian measure `p(u|d)`
  for the model `u` given observed data `d`.
- `ConstrainedLinearBayesianInversion`: Solves the inverse problem subject to
  an affine constraint `u in A`.
"""

from __future__ import annotations
from typing import Optional, List, Literal, Union

from joblib import Parallel, delayed
import numpy as np

import scipy.sparse as sps
import scipy.sparse.linalg as splinalg

from .inversion import LinearInversion
from .gaussian_measure import GaussianMeasure
from .forward_problem import LinearForwardProblem
from .linear_operators import LinearOperator, DiagonalSparseMatrixLinearOperator
from .linear_solvers import (
    LinearSolver,
    IterativeLinearSolver,
    ResidualTrackingCallback,
)
from .hilbert_space import Vector, EuclideanSpace
from .affine_operators import AffineOperator
from .low_rank import LowRankSVD, LowRankEig


class LinearBayesianInversion(LinearInversion):
    """
    Solves a linear inverse problem using Bayesian methods.

    This class applies to problems of the form `d = A(u) + e`, where `u` is
    a Gaussian random variable representing the model prior, and `e` is a
    Gaussian random variable representing observation noise.

    It computes the exact posterior Gaussian measure `p(u|d)`, providing access
    to the posterior expectation, the posterior covariance operator, and an
    efficient exact-sampling mechanism using the randomize-then-optimize technique.
    """

    # =========================================================================
    # Initialization & Core Properties
    # =========================================================================

    def __init__(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        /,
        *,
        formalism: Literal["model_space", "data_space"] = "data_space",
    ) -> None:
        """
        Initializes the linear Bayesian inversion problem.

        Args:
            forward_problem: The forward problem linking the model to the data,
                containing the forward operator `A` and data error measure `R`.
            model_prior_measure: The prior Gaussian measure `Q` on the model space.
            formalism: The algebraic space in which the normal equations are
                assembled and solved. Must be 'model_space' or 'data_space'.
                Defaults to 'data_space'.

        Raises:
            ValueError: If an invalid formalism string is provided, or if the
                'model_space' formalism is selected but the necessary inverse
                covariance operators (precision operators) are not set.
        """
        super().__init__(forward_problem, formalism=formalism)
        self._model_prior_measure: GaussianMeasure = model_prior_measure

        if self.formalism == "model_space":
            if not self.model_prior_measure.inverse_covariance_set:
                raise ValueError(
                    "Prior inverse covariance must be set for model_space formalism."
                )
            if (
                self.forward_problem.data_error_measure_set
                and not self.forward_problem.data_error_measure.inverse_covariance_set
            ):
                raise ValueError(
                    "Data error inverse covariance must be set for model_space formalism."
                )

    def with_formalism(
        self, formalism: Literal["model_space", "data_space"]
    ) -> LinearBayesianInversion:
        """
        Returns a new instance of the Bayesian inversion using the specified formalism.

        Args:
            formalism: The algebraic space in which the normal equations should be
                assembled and solved. Must be 'model_space' or 'data_space'.

        Returns:
            A new LinearBayesianInversion instance sharing the exact same forward
            problem and prior measure, but configured to use the specified formalism.
        """
        return type(self)(
            self.forward_problem, self.model_prior_measure, formalism=formalism
        )

    @property
    def model_prior_measure(self) -> GaussianMeasure:
        """The prior Gaussian measure on the model space."""
        return self._model_prior_measure

    @property
    def data_prior_measure(self) -> GaussianMeasure:
        """
        The prior predictive distribution on the data space.
        This represents the expected distribution of data before observation.
        """
        return self.data_measure_from_model_measure(self.model_prior_measure)

    @property
    def joint_prior_measure(self) -> GaussianMeasure:
        """
        The joint prior distribution of both the model and the data.
        """
        return self.joint_measure(self.model_prior_measure)

    # =========================================================================
    # Core Algebra & Normal Equations
    # =========================================================================

    @property
    def normal_operator(self) -> LinearOperator:
        """
        Constructs the Bayesian Normal operator for the chosen formalism.

        For 'data_space': Returns `N = A Q A* + R`
        For 'model_space': Returns `N = Q^-1 + A* R^-1 A`

        Returns:
            A LinearOperator representing the normal equations matrix.
        """
        forward_operator = self.forward_problem.forward_operator

        if self.formalism == "data_space":
            model_prior_covariance = self.model_prior_measure.covariance
            if self.forward_problem.data_error_measure_set:
                return (
                    forward_operator @ model_prior_covariance @ forward_operator.adjoint
                    + self.forward_problem.data_error_measure.covariance
                )
            else:
                return (
                    forward_operator @ model_prior_covariance @ forward_operator.adjoint
                )

        else:  # model_space
            prior_inv_cov = self.model_prior_measure.inverse_covariance
            if self.forward_problem.data_error_measure_set:
                data_inv_cov = (
                    self.forward_problem.data_error_measure.inverse_covariance
                )
                return (
                    prior_inv_cov
                    + forward_operator.adjoint @ data_inv_cov @ forward_operator
                )
            else:
                return prior_inv_cov + forward_operator.adjoint @ forward_operator

    def get_normal_equations_rhs(self, data: Vector) -> Vector:
        """
        Computes the exact right-hand side vector (v) of the normal equations
        N * w = v for a given observed data vector, automatically accounting
        for non-zero prior and noise expectations.
        """
        forward_operator = self.forward_problem.forward_operator
        data_space = forward_operator.codomain

        # 1. Shift data by the prior model expectation: d - A(mu_u)
        if self.model_prior_measure.has_zero_expectation:
            shifted_data = data_space.copy(data)
        else:
            shifted_data = data_space.subtract(
                data, forward_operator(self.model_prior_measure.expectation)
            )

        # 2. Shift data by the noise expectation: d - A(mu_u) - mu_e
        if self.forward_problem.data_error_measure_set:
            if not self.forward_problem.data_error_measure.has_zero_expectation:
                error_expectation = self.forward_problem.data_error_measure.expectation
                shifted_data = data_space.subtract(shifted_data, error_expectation)

        # 3. Apply formalism-specific mappings
        if self.formalism == "data_space":
            # In data space, the RHS is exactly the shifted data residuals
            return shifted_data
        else:
            # In model space, the RHS is projected back via A* R^-1
            if self.forward_problem.data_error_measure_set:
                data_inv_cov = (
                    self.forward_problem.data_error_measure.inverse_covariance
                )
                return forward_operator.adjoint(data_inv_cov(shifted_data))
            else:
                return forward_operator.adjoint(shifted_data)

    def kalman_operator(
        self,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[LinearOperator] = None,
    ) -> LinearOperator:
        """
        Constructs the Kalman gain operator `K`.

        The Kalman gain maps data residuals to model space updates.

        For 'data_space': `K = Q A* (A Q A* + R)^-1`
        For 'model_space': `K = (Q^-1 + A* R^-1 A)^-1 A* R^-1`

        Args:
            solver: The LinearSolver used to invert the normal operator.
            preconditioner: Optional preconditioner for iterative solvers.

        Returns:
            A LinearOperator representing the Kalman gain.
        """
        forward_operator = self.forward_problem.forward_operator
        normal_operator = self.normal_operator

        if isinstance(solver, IterativeLinearSolver):
            inverse_normal_operator = solver(
                normal_operator, preconditioner=preconditioner
            )
        else:
            inverse_normal_operator = solver(normal_operator)

        if self.formalism == "data_space":
            model_prior_covariance = self.model_prior_measure.covariance
            return (
                model_prior_covariance
                @ forward_operator.adjoint
                @ inverse_normal_operator
            )
        else:  # model_space
            if self.forward_problem.data_error_measure_set:
                data_inv_cov = (
                    self.forward_problem.data_error_measure.inverse_covariance
                )
                return inverse_normal_operator @ forward_operator.adjoint @ data_inv_cov
            else:
                return inverse_normal_operator @ forward_operator.adjoint

    # =========================================================================
    # Posterior Extraction
    # =========================================================================

    def model_posterior_measure(
        self,
        data: Vector,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[LinearOperator] = None,
    ) -> GaussianMeasure:
        """
        Computes and returns the posterior Gaussian measure `p(u|d)`.

        This method applies the Kalman update equations to find the posterior
        expectation and covariance. If both the prior and data error measures
        have sampling enabled, it automatically constructs a randomize-then-optimize
        exact sampling function for the posterior.

        Args:
            data: The observed data vector.
            solver: A linear solver for inverting the normal operator.
            preconditioner: An optional preconditioner for iterative solvers.

        Returns:
            A GaussianMeasure representing the posterior distribution.
        """
        data_space = self.data_space
        model_space = self.model_space
        forward_operator = self.forward_problem.forward_operator
        model_prior_covariance = self.model_prior_measure.covariance

        # 1. Resolve Inverse Normal Operator & Kalman Gain
        normal_operator = self.normal_operator
        if isinstance(solver, IterativeLinearSolver):
            inverse_normal_operator = solver(
                normal_operator, preconditioner=preconditioner
            )
        else:
            inverse_normal_operator = solver(normal_operator)

        if self.formalism == "data_space":
            kalman_gain = (
                model_prior_covariance
                @ forward_operator.adjoint
                @ inverse_normal_operator
            )
            # C_post = C_u - K A C_u
            covariance = model_prior_covariance - (
                kalman_gain @ forward_operator @ model_prior_covariance
            )
        else:  # model_space
            if self.forward_problem.data_error_measure_set:
                data_inv_cov = (
                    self.forward_problem.data_error_measure.inverse_covariance
                )
                kalman_gain = (
                    inverse_normal_operator @ forward_operator.adjoint @ data_inv_cov
                )
            else:
                kalman_gain = inverse_normal_operator @ forward_operator.adjoint
            # Optimization: In model space, the inverted normal operator IS the posterior covariance
            covariance = inverse_normal_operator

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

        # 3. Set up Posterior Sampling (Randomize-then-Optimize)
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

    def posterior_expectation_operator(
        self,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[LinearOperator] = None,
    ) -> Union[LinearOperator, AffineOperator]:
        """
        Constructs the operator mapping observed data to the posterior expectation.

        The mapping evaluates F(d) = mu_u + K(d - A(mu_u) - mu_e).

        If the prior and data error measures both have a zero expectation, this
        mapping is purely linear and returns the Kalman gain operator directly.
        Otherwise, it regroups the terms into an AffineOperator:
        F(d) = K(d) + (mu_u - K(A(mu_u) + mu_e)).

        Args:
            solver: The LinearSolver used to invert the normal operator.
            preconditioner: Optional preconditioner for iterative solvers.

        Returns:
            A LinearOperator (if expectations are zero) or an AffineOperator.
        """
        kalman_gain = self.kalman_operator(solver, preconditioner=preconditioner)

        zero_prior_mean = self.model_prior_measure.has_zero_expectation
        zero_error_mean = (
            not self.forward_problem.data_error_measure_set
            or self.forward_problem.data_error_measure.has_zero_expectation
        )

        # Strictly linear case
        if zero_prior_mean and zero_error_mean:
            return kalman_gain

        data_space = self.data_space
        model_space = self.model_space
        A = self.forward_problem.forward_operator

        # 1. Compute the baseline shift in data space: A(mu_u) + mu_e
        if zero_prior_mean and not zero_error_mean:
            data_shift = self.forward_problem.data_error_measure.expectation
        elif not zero_prior_mean and zero_error_mean:
            data_shift = A(self.model_prior_measure.expectation)
        else:
            data_shift = data_space.add(
                A(self.model_prior_measure.expectation),
                self.forward_problem.data_error_measure.expectation,
            )

        # 2. Compute the translation vector in model space: mu_u - K(data_shift)
        k_shift = kalman_gain(data_shift)

        if zero_prior_mean:
            translation = model_space.negative(k_shift)
        else:
            translation = model_space.subtract(
                self.model_prior_measure.expectation, k_shift
            )

        return AffineOperator(linear_part=kalman_gain, translation=translation)

    def normal_residual_callback(
        self,
        data: Vector,
        /,
        *,
        message: str = "CG Iteration: {iter} | Normal Residual: {res:.3e}",
        print_progress: bool = True,
    ) -> ResidualTrackingCallback:
        """
        Generates a ResidualTrackingCallback pre-configured to track the
        convergence of the Bayesian normal equations for the given data vector.

        Args:
            data: The observed data vector.
            message: The formatting string for printing progress.
            print_progress: If True, prints the message to stdout at each iteration.

        Returns:
            A configured ResidualTrackingCallback ready to be passed to a solver.
        """
        rhs = self.get_normal_equations_rhs(data)

        return ResidualTrackingCallback(
            operator=self.normal_operator,
            y=rhs,
            print_progress=print_progress,
            message=message,
        )

    # =========================================================================
    # Preconditioners
    # =========================================================================

    def diagonal_normal_preconditioner(
        self,
        /,
        *,
        blocks: Optional[List[List[int]]] = None,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> LinearOperator:
        """
        Constructs a diagonal preconditioner specifically for the data-space
        Bayesian normal operator `(A Q A* + R)`.

        This exploits the identity `<v, A Q A* v> = <A* v, Q A* v>`. If blocks
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

        Raises:
            ValueError: If the inversion was initialized with `formalism='model_space'`,
                as this preconditioner is mathematically invalid for that normal operator.
        """
        if self.formalism != "data_space":
            raise ValueError(
                "This custom preconditioner is mathematically derived for the "
                "data-space normal operator (A Q A* + R) and cannot be used "
                "with the model-space formalism."
            )

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
            n = len(block)
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
        Builds a sparse preconditioner specifically for the data-space Bayesian
        normal equations using randomized Nystrom approximations on localized,
        potentially overlapping sub-blocks.

        Args:
            interacting_blocks: A list of lists, where each sub-list contains the
                indices of data points that strongly couple to each other.
            rank: The rank of the randomized Nystrom approximation to use per block.
            parallel: If True, computes the sub-block approximations in parallel.
            n_jobs: Number of CPU cores to use if parallel=True (-1 uses all cores).

        Returns:
            A LinearOperator representing the inverse of the sparse approximation.

        Raises:
            ValueError: If the inversion was initialized with `formalism='model_space'`,
                as this preconditioner is mathematically invalid for that normal operator.
        """
        if self.formalism != "data_space":
            raise ValueError(
                "This custom preconditioner is mathematically derived for the "
                "data-space normal operator (A Q A* + R) and cannot be used "
                "with the model-space formalism."
            )

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

        def _process_block(indices: list[int]):
            block_size = len(indices)
            idx_array = np.array(indices)

            restrict_coords = euclidean_full.subspace_projection(indices)
            P = restrict_coords @ to_components_op
            P_star = P.adjoint

            H_local = P @ core_normal_op @ P_star

            actual_rank = min(rank, block_size)
            eig_approx = LowRankEig.from_randomized(
                H_local, actual_rank, method="fixed"
            )

            V = eig_approx.u_factor.matrix(dense=True, galerkin=True)
            Lambda = eig_approx.eigenvalues
            M_local_array = (V * Lambda) @ V.T

            I_local, J_local = np.meshgrid(
                np.arange(block_size), np.arange(block_size), indexing="ij"
            )
            I_global = idx_array[I_local.flatten()]
            J_global = idx_array[J_local.flatten()]
            V_flattened = M_local_array.flatten()

            return I_global, J_global, V_flattened

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

        H_sparse = sps.coo_matrix(
            (values, (row_indices, col_indices)), shape=(data_dim, data_dim)
        )

        R_sparse = sps.diags(noise_variance)
        H_approx = (H_sparse + R_sparse).tocsc()

        print("Factorizing sparse matrix...")
        splu_solver = splinalg.splu(H_approx)

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

    def woodbury_data_preconditioner(
        self,
        solver: LinearSolver,
        /,
        *,
        prior_solver: Optional[LinearSolver] = None,
        noise_solver: Optional[LinearSolver] = None,
    ) -> LinearOperator:
        """
        Constructs a data-space preconditioner using the Woodbury matrix identity.

        Data Space Normal Operator: N_d = A Q A* + R
        Woodbury Identity: N_d^-1 = R^-1 - R^-1 A (Q^-1 + A* R^-1 A)^-1 A* R^-1

        Note:
            This method assumes the prior (Q) and noise (R) measures either already
            have well-defined inverse covariances, or are well-conditioned enough
            to be inverted by the provided solvers. If your prior is an unbounded
            operator on a function space, use `Q.with_regularized_inverse()`
            before passing it to the inversion.

        Args:
            solver: The LinearSolver used to invert the inner Woodbury operator (N_m).
            prior_solver: An optional solver used to explicitly invert the prior
                          covariance (Q) if its inverse is not already set.
                          Defaults to `solver` if not provided.
            noise_solver: An optional solver used to explicitly invert the data
                          error covariance (R) if its inverse is not already set.
                          Defaults to `solver` if not provided.

        Returns:
            A LinearOperator representing the Woodbury-approximated inverse.
        """
        A = self.forward_problem.forward_operator
        Q = self.model_prior_measure

        if not self.forward_problem.data_error_measure_set:
            raise ValueError(
                "Data error measure must be set for the Woodbury identity."
            )
        R = self.forward_problem.data_error_measure

        # 1. Invert the noise covariance R
        if R.inverse_covariance_set:
            R_inv = R.inverse_covariance
        else:
            r_solver = noise_solver if noise_solver is not None else solver
            R_inv = r_solver(R.covariance)

        # 2. Invert the prior covariance Q
        if Q.inverse_covariance_set:
            Q_inv = Q.inverse_covariance
        else:
            q_solver = prior_solver if prior_solver is not None else solver
            Q_inv = q_solver(Q.covariance)

        # 3. Assemble and invert the inner Woodbury operator (N_m)
        N_m = Q_inv + A.adjoint @ R_inv @ A
        N_m_inv = solver(N_m)

        # 4. Return the assembled preconditioner
        return R_inv - R_inv @ A @ N_m_inv @ A.adjoint @ R_inv

    # =========================================================================
    # Surrogates & Parameterization
    # =========================================================================

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

        This is primarily used to construct robust, computationally cheap
        surrogate models to use as preconditioners for the full, complex
        inverse problem.

        Args:
            alternate_forward_operator: An optional simplified forward operator.
            alternate_prior_measure: An optional simplified prior measure.
            alternate_data_error_measure: An optional simplified data error measure.

        Returns:
            A new LinearBayesianInversion instance representing the surrogate problem.
            The surrogate inherits the `formalism` of the parent problem.

        Raises:
            ValueError: If the alternative operators/measures exist in incompatible domains/codomains.
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

        # Inherit the formalism of the parent inversion
        return LinearBayesianInversion(
            surrogate_forward_problem, Q_tilde, formalism=self.formalism
        )

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

        Args:
            solver: The LinearSolver to use to exactly invert the surrogate normal operator.
            alternate_forward_operator: An optional simplified forward operator.
            alternate_prior_measure: An optional simplified prior measure.
            alternate_data_error_measure: An optional simplified data error measure.

        Returns:
            A LinearOperator representing the inverse of the surrogate normal equations.
        """
        surrogate_inv = self.surrogate_inversion(
            alternate_forward_operator=alternate_forward_operator,
            alternate_prior_measure=alternate_prior_measure,
            alternate_data_error_measure=alternate_data_error_measure,
        )
        return solver(surrogate_inv.normal_operator)

    def surrogate_woodbury_preconditioner(
        self,
        solver: LinearSolver,
        /,
        *,
        alternate_forward_operator: Optional[LinearOperator] = None,
        alternate_prior_measure: Optional[GaussianMeasure] = None,
        alternate_data_error_measure: Optional[GaussianMeasure] = None,
        prior_solver: Optional[LinearSolver] = None,
        noise_solver: Optional[LinearSolver] = None,
    ) -> LinearOperator:
        """
        Builds a data-space preconditioner by applying the Woodbury matrix identity
        to a simplified surrogate inverse problem.

        This method chains the construction of the surrogate model with the
        extraction of the Woodbury inverse in one step.

        Note:
            Ensure that any alternate measures provided have well-defined inverse
            covariances, or use `.with_regularized_inverse()` on them before
            passing them to this method.

        Args:
            solver: The LinearSolver used to invert the inner Woodbury operator.
            alternate_forward_operator: An optional simplified forward operator.
            alternate_prior_measure: An optional simplified prior measure.
            alternate_data_error_measure: An optional simplified data error measure.
            prior_solver: Optional solver for the prior covariance.
            noise_solver: Optional solver for the noise covariance.

        Returns:
            A LinearOperator representing the Woodbury-approximated inverse.
        """
        surrogate_inv = self.surrogate_inversion(
            alternate_forward_operator=alternate_forward_operator,
            alternate_prior_measure=alternate_prior_measure,
            alternate_data_error_measure=alternate_data_error_measure,
        )
        return surrogate_inv.woodbury_data_preconditioner(
            solver, prior_solver=prior_solver, noise_solver=noise_solver
        )

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

        This method generates computationally cheap surrogate models to be used in
        constructing preconditioners for massive, ill-conditioned inverse problems
        (e.g., using spectral or banded methods). The low-rank approximations are
        computed using randomized SVD and eigendecomposition algorithms.

        Args:
            forward_rank: Target rank for the randomized SVD of the forward operator.
            prior_rank: Target rank for the randomized eigendecomposition of the prior.
            data_error_rank: Target rank for the randomized eigendecomposition of the noise.
            forward_kwargs: Additional kwargs passed directly to `LinearOperator.random_svd`.
            prior_kwargs: Additional kwargs passed directly to `GaussianMeasure.low_rank_approximation`.
            data_error_kwargs: Additional kwargs passed directly to `GaussianMeasure.low_rank_approximation`.

        Returns:
            A LinearBayesianInversion representing the low-rank surrogate problem.
        """
        A_tilde = None
        Q_tilde = None
        R_tilde = None

        if forward_rank is not None:
            f_kwargs = forward_kwargs or {}
            original_A = self.forward_problem.forward_operator
            A_tilde = LowRankSVD.from_randomized(original_A, forward_rank, **f_kwargs)

        if prior_rank is not None:
            p_kwargs = prior_kwargs or {}
            Q_tilde = self.model_prior_measure.low_rank_approximation(
                prior_rank, **p_kwargs
            )

        if data_error_rank is not None:
            if not self.forward_problem.data_error_measure_set:
                raise ValueError("Cannot approximate data error measure: none is set.")

            d_kwargs = data_error_kwargs or {}
            R_tilde = self.forward_problem.data_error_measure.low_rank_approximation(
                data_error_rank, **d_kwargs
            )

        return self.surrogate_inversion(
            alternate_forward_operator=A_tilde,
            alternate_prior_measure=Q_tilde,
            alternate_data_error_measure=R_tilde,
        )

    def parameterized_inversion(
        self,
        parameterization: LinearOperator,
        /,
        *,
        parameter_prior: Optional[GaussianMeasure] = None,
        dense: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
        formalism: Optional[Literal["model_space", "data_space"]] = None,
    ) -> LinearBayesianInversion:
        """
        Constructs a parameterized surrogate of the Bayesian inversion.

        If the target formalism resolves to 'model_space' (which is typical for
        parameterized inversions), the parameter prior's covariance matrix will
        be automatically densified to explicitly compute the required precision
        (inverse covariance) operator.

        Args:
            parameterization: A LinearOperator mapping from the parameter
                space to the full model space.
            parameter_prior: An optional prior measure on the parameter space.
                If not provided, the original model prior is pulled back to the
                parameter space automatically.
            dense: If True, computes and stores operators as dense matrices.
            parallel: If True, computes the dense matrices in parallel.
            n_jobs: Number of CPU cores to use. -1 means all available.
            formalism: An optional override for the formalism of the new inversion.
                If None, inherits the formalism of the parent inversion.

        Returns:
            A new LinearBayesianInversion instance operating on the parameter space.
        """
        target_formalism = formalism or self.formalism

        if parameter_prior is None:
            parameter_prior = self.model_prior_measure.affine_mapping(
                operator=parameterization.adjoint
            )

        if parameter_prior.domain != parameterization.domain:
            raise ValueError(
                "The domain of the parameter prior must match the domain "
                "of the parameterization operator."
            )

        new_fp = self.forward_problem.parameterized_problem(
            parameterization, dense=dense, parallel=parallel, n_jobs=n_jobs
        )

        new_prior = parameter_prior
        if dense or target_formalism == "model_space":
            new_prior = new_prior.with_dense_covariance(
                parallel=parallel, n_jobs=n_jobs
            )

        return type(self)(new_fp, new_prior, formalism=target_formalism)
