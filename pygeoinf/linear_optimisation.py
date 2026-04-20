"""
Provides deterministic, optimization-based methods for solving linear inverse problems.

This module implements classical inversion techniques that seek a single "best-fit"
model by minimizing a specific cost functional. It leverages the abstract operator
algebra of the library, allowing inversions to be rigorously formulated in
Hilbert spaces and seamlessly applied to discrete representations.

A core feature of this module is its dual algebraic formalism, allowing users to
optimize computational efficiency based on the problem geometry:
- **Model Space Formulation**: Assembles and solves the standard normal equations
  (size N x N, where N is the model dimension). Best suited for overdetermined problems.
- **Data Space Formulation**: Assembles and solves the dual formulation (size M x M,
  where M is the data dimension) using the representer method. Highly efficient for
  underdetermined problems where data measurements are sparse compared to the model.

Key Classes
-----------
- `LinearLeastSquaresInversion`: Solves the inverse problem by minimizing a
  Tikhonov-regularized least-squares functional.
- `ConstrainedLinearLeastSquaresInversion`: Solves the regularized least-squares
  problem strictly within an affine subspace (e.g., enforcing exact boundary
  conditions or mean property values).
- `LinearMinimumNormInversion`: Finds the model with the smallest norm that fits
  the data to a statistically acceptable degree using the discrepancy principle.
  Provides exact analytical Fréchet derivatives of the discrepancy search.
- `ConstrainedLinearMinimumNormInversion`: Applies the discrepancy principle
  subject to an exact affine subspace constraint, resolving the non-linear
  mapping between constraint values and the resulting model.
"""

from __future__ import annotations
from typing import Optional, Union, Literal

from .nonlinear_operators import NonLinearOperator
from .affine_operators import AffineOperator
from .inversion import LinearInversion
from .forward_problem import LinearForwardProblem
from .linear_operators import LinearOperator
from .linear_solvers import LinearSolver, IterativeLinearSolver
from .hilbert_space import Vector
from .subspaces import AffineSubspace


class LinearLeastSquaresInversion(LinearInversion):
    """
    Solves a linear inverse problem using Tikhonov-regularized least-squares.

    This method finds the model `u` that minimizes the cost functional:
        J(u) = ||A(u) - d||^2_R + damping * ||u||^2

    where `A` is the forward operator, `d` is the observed data, `R` is the
    data covariance (if a data error measure is set), and `damping` is the
    Tikhonov regularization parameter.

    This class supports two formalisms for constructing the linear system:
    1. 'model_space': Solves the standard normal equations of size (N x N),
       where N is the model dimension. Best for overdetermined problems.
    2. 'data_space': Solves the dual formulation of size (M x M), where M
       is the data dimension. Best for highly underdetermined problems.
    """

    # =========================================================================
    # Initialization & State
    # =========================================================================

    def __init__(
        self,
        forward_problem: LinearForwardProblem,
        /,
        *,
        formalism: Literal["model_space", "data_space"] = "data_space",
    ) -> None:
        """
        Initializes the linear least-squares inversion.

        Args:
            forward_problem: The linear forward problem defining the mapping
                from model space to data space, along with the optional data
                error measure.
            formalism: The algebraic space in which the normal equations are
                assembled and solved. Must be either 'model_space' or 'data_space'.
                Defaults to 'data_space'.

        Raises:
            ValueError: If an invalid formalism string is provided.
        """
        super().__init__(forward_problem, formalism=formalism)

        if (
            self.forward_problem.data_error_measure_set
            and self.formalism == "model_space"
        ):
            self.assert_inverse_data_covariance()

    def with_formalism(
        self, formalism: Literal["model_space", "data_space"]
    ) -> LinearLeastSquaresInversion:
        """
        Returns a new instance of the inversion using the specified formalism.

        Args:
            formalism: The algebraic space in which the normal equations should be
                assembled and solved. Must be 'model_space' or 'data_space'.

        Returns:
            A new LinearLeastSquaresInversion instance with the updated formalism.
        """
        return type(self)(self.forward_problem, formalism=formalism)

    # =========================================================================
    # Normal Equations
    # =========================================================================

    def normal_operator(self, damping: float) -> LinearOperator:
        """
        Constructs the regularized normal operator for the chosen formalism.

        For 'model_space', this returns:  A* R^{-1} A + damping * I
        For 'data_space', this returns:   A A* + damping * R

        Args:
            damping: The non-negative Tikhonov regularization parameter.

        Returns:
            A LinearOperator representing the left-hand side of the normal equations.

        Raises:
            ValueError: If the damping parameter is negative.
        """
        if damping < 0:
            raise ValueError("Damping parameter must be non-negative.")

        forward_operator = self.forward_problem.forward_operator

        if self.formalism == "model_space":
            identity = self.forward_problem.model_space.identity_operator()
            if self.forward_problem.data_error_measure_set:
                inverse_data_covariance = (
                    self.forward_problem.data_error_measure.inverse_covariance
                )
                return (
                    forward_operator.adjoint
                    @ inverse_data_covariance
                    @ forward_operator
                    + damping * identity
                )
            else:
                return forward_operator.adjoint @ forward_operator + damping * identity

        else:  # data_space
            identity = self.forward_problem.data_space.identity_operator()
            if self.forward_problem.data_error_measure_set:
                data_covariance = self.forward_problem.data_error_measure.covariance
                return (
                    forward_operator @ forward_operator.adjoint
                    + damping * data_covariance
                )
            else:
                return forward_operator @ forward_operator.adjoint + damping * identity

    def normal_rhs(self, data: Vector) -> Vector:
        """
        Computes the right-hand side vector for the normal equations.

        Prior to construction, the data is shifted by the expected value of the
        data error measure (i.e., v - z_bar), if applicable.

        For 'model_space', this returns:  A* R^{-1} (v - z_bar)
        For 'data_space', this returns:   (v - z_bar)

        Args:
            data: The observed data vector in the data space.

        Returns:
            The right-hand side Vector for the chosen linear system.
        """
        forward_operator = self.forward_problem.forward_operator

        # Calculate the shifted data (v - z_bar)
        if (
            self.forward_problem.data_error_measure_set
            and not self.forward_problem.data_error_measure.has_zero_expectation
        ):
            shifted_data = self.forward_problem.data_space.subtract(
                data, self.forward_problem.data_error_measure.expectation
            )
        else:
            shifted_data = data

        if self.formalism == "model_space":
            if self.forward_problem.data_error_measure_set:
                inverse_data_covariance = (
                    self.forward_problem.data_error_measure.inverse_covariance
                )
                return (forward_operator.adjoint @ inverse_data_covariance)(
                    shifted_data
                )
            else:
                return forward_operator.adjoint(shifted_data)

        else:  # data_space
            # For data space, the RHS is just the shifted data
            # (the damping factor is accounted for analytically during operator assembly)
            return shifted_data

    def normal_residual_callback(
        self,
        damping: float,
        data: Vector,
        /,
        *,
        message: str = "Iteration: {iter} | Normal Residual: {res:.3e}",
        print_progress: bool = True,
    ):
        """
        Generates a ResidualTrackingCallback pre-configured to track the
        convergence of the least-squares normal equations for the given data vector.
        """
        from .linear_solvers import ResidualTrackingCallback

        rhs = self.normal_rhs(data)

        return ResidualTrackingCallback(
            operator=self.normal_operator(damping),
            y=rhs,
            print_progress=print_progress,
            message=message,
        )

    # =========================================================================
    # Inversion Operators
    # =========================================================================

    def least_squares_operator(
        self,
        damping: float,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[Union[LinearOperator, LinearSolver]] = None,
    ) -> Union[LinearOperator, AffineOperator]:
        """
        Constructs the full operator that maps observed data directly to the
        least-squares model solution.

        This method solves the internal normal equations and applies the necessary
        algebraic transformations (and affine shifts) to recover the model parameters
        from the data, seamlessly handling whichever formalism was selected during
        initialization.

        Args:
            damping: The Tikhonov regularization parameter.
            solver: The LinearSolver instance used to invert the normal operator.
            preconditioner: An optional LinearOperator, or a LinearSolver factory,
                used to precondition the normal equations. Only utilized if the
                provided solver is an IterativeLinearSolver.

        Returns:
            A LinearOperator (or AffineOperator, if a non-zero data expectation
            exists) that maps a vector from the data space to the optimal vector
            in the model space.

        Raises:
            TypeError: If the provided preconditioner is neither a LinearOperator
                nor a LinearSolver.
        """
        forward_operator = self.forward_problem.forward_operator
        normal_operator = self.normal_operator(damping)

        resolved_preconditioner = None
        if preconditioner is not None:
            if isinstance(preconditioner, LinearOperator):
                resolved_preconditioner = preconditioner
            elif isinstance(preconditioner, LinearSolver):
                resolved_preconditioner = preconditioner(normal_operator)
            else:
                raise TypeError(
                    "Preconditioner must be a LinearOperator or LinearSolver."
                )

        if isinstance(solver, IterativeLinearSolver):
            inverse_normal_operator = solver(
                normal_operator, preconditioner=resolved_preconditioner
            )
        else:
            inverse_normal_operator = solver(normal_operator)

        # Assemble the linear part mapping based on formalism
        if self.formalism == "model_space":
            if self.forward_problem.data_error_measure_set:
                inverse_data_covariance = (
                    self.forward_problem.data_error_measure.inverse_covariance
                )
                linear_part = (
                    inverse_normal_operator
                    @ forward_operator.adjoint
                    @ inverse_data_covariance
                )
            else:
                linear_part = inverse_normal_operator @ forward_operator.adjoint
        else:
            # Data space: u = A* @ (AA* + lambda R)^-1 @ (v - z_bar)
            linear_part = forward_operator.adjoint @ inverse_normal_operator

        # Apply the affine shift if there's a non-zero expectation, regardless of formalism
        if (
            self.forward_problem.data_error_measure_set
            and not self.forward_problem.data_error_measure.has_zero_expectation
        ):
            expected_data = self.forward_problem.data_error_measure.expectation
            translation = self.model_space.negative(linear_part(expected_data))
            return AffineOperator(linear_part, translation)
        else:
            return linear_part

    # =========================================================================
    # Preconditioners
    # =========================================================================

    def woodbury_data_preconditioner(
        self,
        damping: float,
        /,
        *,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> LinearOperator:
        """
        Constructs a data-space preconditioner using the Woodbury matrix identity.

        Data Space Normal Operator: N_d = A A* + damping * R
        Woodbury Identity: N_d^-1 = (1/damping) * [R^-1 - R^-1 A (A* R^-1 A + damping * I)^-1 A* R^-1]
        """
        if damping <= 0:
            raise ValueError(
                "Damping must be strictly positive for the Woodbury identity."
            )

        from .linear_solvers import LUSolver

        A = self.forward_problem.forward_operator

        if not self.forward_problem.data_error_measure_set:
            raise ValueError(
                "Data error measure must be set for the Woodbury identity."
            )
        R = self.forward_problem.data_error_measure

        # 1. Extract or compute R^-1
        if R.inverse_covariance_set:
            R_inv = R.inverse_covariance
        else:
            r_solver = LUSolver(galerkin=False, parallel=parallel, n_jobs=n_jobs)
            R_inv = r_solver(R.covariance)

        # 2. Form the model-space normal operator: N_m = A* R^-1 A + damping * I
        identity = self.model_space.identity_operator()
        N_m = A.adjoint @ R_inv @ A + damping * identity

        # 3. Exactly invert the small model-space operator
        nm_solver = LUSolver(galerkin=False, parallel=parallel, n_jobs=n_jobs)
        N_m_inv = nm_solver(N_m)

        # 4. Assemble the Woodbury identity
        term2 = R_inv @ A @ N_m_inv @ A.adjoint @ R_inv

        return (1.0 / damping) * (R_inv - term2)

    # =========================================================================
    # Surrogates & Parameterization
    # =========================================================================

    def surrogate_inversion(
        self,
        /,
        *,
        alternate_forward_operator: Optional[LinearOperator] = None,
        alternate_data_error_measure=None,  # Accepts GaussianMeasure
    ) -> LinearLeastSquaresInversion:
        """
        Constructs a surrogate least-squares inversion problem using simplified physics
        or data errors.
        """
        A_tilde = alternate_forward_operator or self.forward_problem.forward_operator

        if alternate_data_error_measure is not None:
            R_tilde = alternate_data_error_measure
        elif self.forward_problem.data_error_measure_set:
            R_tilde = self.forward_problem.data_error_measure
        else:
            R_tilde = None

        surrogate_forward_problem = LinearForwardProblem(
            A_tilde, data_error_measure=R_tilde
        )

        return LinearLeastSquaresInversion(
            surrogate_forward_problem, formalism=self.formalism
        )

    def surrogate_woodbury_preconditioner(
        self,
        damping: float,
        /,
        *,
        alternate_forward_operator: Optional[LinearOperator] = None,
        alternate_data_error_measure=None,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> LinearOperator:
        """
        Builds a data-space preconditioner by applying the Woodbury matrix identity
        to a simplified surrogate inverse problem.
        """
        surrogate_inv = self.surrogate_inversion(
            alternate_forward_operator=alternate_forward_operator,
            alternate_data_error_measure=alternate_data_error_measure,
        )
        return surrogate_inv.woodbury_data_preconditioner(
            damping, parallel=parallel, n_jobs=n_jobs
        )


class ConstrainedLinearLeastSquaresInversion(LinearInversion):
    """
    Solves a linear inverse problem subject to an affine subspace constraint.

    This method finds the model `u` that minimizes the Tikhonov-regularized
    least-squares functional while strictly confining the solution to an
    affine subspace (e.g., enforcing a specific average property value or
    boundary condition).

    Supports both 'model_space' and 'data_space' formalisms for the underlying
    unconstrained inversion.
    """

    # =========================================================================
    # Initialization & State
    # =========================================================================

    def __init__(
        self,
        forward_problem: LinearForwardProblem,
        constraint: AffineSubspace,
        /,
        *,
        formalism: Literal["model_space", "data_space"] = "data_space",
    ) -> None:
        """
        Initializes the constrained linear least-squares inversion.

        Args:
            forward_problem: The linear forward problem.
            constraint: The affine subspace defining the allowed model states.
            formalism: The algebraic space in which the normal equations are
                assembled and solved. Must be either 'model_space' or 'data_space'.
                Defaults to 'data_space'.

        Raises:
            ValueError: If an invalid formalism string is provided.
        """
        super().__init__(forward_problem, formalism=formalism)

        if (
            self.forward_problem.data_error_measure_set
            and self.formalism == "model_space"
        ):
            self.assert_inverse_data_covariance()

        self._constraint = constraint

        # Use the projection operator to seamlessly grab the base translation
        proj_op = constraint.projection_operator
        self._u_base = proj_op.translation_part

        # The reduced operator is just A @ P
        reduced_operator = forward_problem.forward_operator @ proj_op.linear_part

        self._reduced_forward_problem = LinearForwardProblem(
            reduced_operator,
            data_error_measure=(
                forward_problem.data_error_measure
                if forward_problem.data_error_measure_set
                else None
            ),
        )

        # Pass the formalism down to the underlying unconstrained solver
        self._unconstrained_inversion = LinearLeastSquaresInversion(
            self._reduced_forward_problem, formalism=formalism
        )

    def with_formalism(
        self, formalism: Literal["model_space", "data_space"]
    ) -> ConstrainedLinearLeastSquaresInversion:
        """
        Returns a new instance of the constrained inversion using the specified formalism.

        Args:
            formalism: The algebraic space in which the normal equations should be
                assembled and solved. Must be 'model_space' or 'data_space'.

        Returns:
            A new ConstrainedLinearLeastSquaresInversion instance with the updated formalism.
        """
        return type(self)(self.forward_problem, self._constraint, formalism=formalism)

    # =========================================================================
    # Inversion Operators & Callbacks
    # =========================================================================

    def least_squares_operator(
        self,
        damping: float,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[Union[LinearOperator, LinearSolver]] = None,
    ) -> AffineOperator:
        """
        Returns an operator that maps data to the constrained least-squares solution.
        """
        reduced_op = self._unconstrained_inversion.least_squares_operator(
            damping, solver, preconditioner=preconditioner
        )
        data_offset = self.forward_problem.forward_operator(self._u_base)
        domain = self.data_space
        codomain = self.model_space
        shift_in = AffineOperator(
            domain.identity_operator(), domain.negative(data_offset)
        )
        shift_out = AffineOperator(codomain.identity_operator(), self._u_base)
        return shift_out @ reduced_op @ shift_in

    def normal_residual_callback(
        self,
        damping: float,
        data: Vector,
        /,
        *,
        message: str = "Iteration: {iter} | Normal Residual: {res:.3e}",
        print_progress: bool = True,
    ):
        """
        Generates a ResidualTrackingCallback for the reduced, unconstrained
        normal equations.
        """
        # The unconstrained inversion is acting on shifted data due to the constraint
        data_offset = self.forward_problem.forward_operator(self._u_base)
        shifted_data = self.data_space.subtract(data, data_offset)

        return self._unconstrained_inversion.normal_residual_callback(
            damping, shifted_data, message=message, print_progress=print_progress
        )

    # =========================================================================
    # Surrogates & Parameterization
    # =========================================================================

    def parameterized_inversion(
        self,
        parameterization: LinearOperator,
        /,
        *,
        dense: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
        formalism: Optional[Literal["model_space", "data_space"]] = None,
    ) -> ConstrainedLinearLeastSquaresInversion:
        """
        Constructs a parameterized surrogate of the constrained least-squares inversion.

        Args:
            parameterization: A LinearOperator mapping from the parameter
                space to the full model space.
            dense: If True, computes and stores operators as dense matrices.
            parallel: If True, computes the dense matrices in parallel.
            n_jobs: Number of CPU cores to use. -1 means all available.
            formalism: An optional override for the formalism of the new inversion.
                If None, inherits the formalism of the parent inversion.

        Returns:
            A new ConstrainedLinearLeastSquaresInversion instance operating on
            the parameter space.
        """
        target_formalism = formalism or self.formalism

        if not self._constraint.has_explicit_equation:
            raise NotImplementedError(
                "Parameterized inversion for constrained problems is only "
                "supported for subspaces defined by an explicit linear equation."
            )

        # 1. Pull back the constraint equation: (B @ M) m = w
        B = self._constraint.constraint_operator
        w = self._constraint.constraint_value

        # Safety check: Number of constraints cannot exceed parameter space dimension
        if B.codomain.dim > parameterization.domain.dim:
            raise ValueError(
                f"The parameter space dimension ({parameterization.domain.dim}) is "
                f"smaller than the number of constraints ({B.codomain.dim})."
            )

        new_B = B @ parameterization
        if dense:
            new_B = new_B.with_dense_matrix(parallel=parallel, n_jobs=n_jobs)

        # 2. Build the surrogate constraint subspace in the parameter space
        from .subspaces import AffineSubspace

        new_S = AffineSubspace.from_linear_equation(
            new_B,
            w,
            solver=self._constraint.solver,
            preconditioner=self._constraint.preconditioner,
        )

        # 3. Build the parameterized forward problem
        new_fp = self.forward_problem.parameterized_problem(
            parameterization, dense=dense, parallel=parallel, n_jobs=n_jobs
        )

        return type(self)(new_fp, new_S, formalism=target_formalism)


class LinearMinimumNormInversion(LinearInversion):
    """
    Finds a regularized solution using the discrepancy principle.

    This method finds the model `u` with the smallest norm that fits the data
    to a statistically acceptable degree (determined by a target chi-squared
    value and significance level).

    This class supports two formalisms for constructing the linear systems:
    1. 'model_space': Solves the normal equations of size (N x N).
    2. 'data_space': Solves the dual formulation of size (M x M).
    """

    # =========================================================================
    # Initialization & State
    # =========================================================================

    def __init__(
        self,
        forward_problem: LinearForwardProblem,
        /,
        *,
        formalism: Literal["model_space", "data_space"] = "data_space",
    ) -> None:
        """
        Initializes the minimum norm inversion.

        Args:
            forward_problem: The linear forward problem.
            formalism: The algebraic space in which the normal equations are
                assembled and solved. Defaults to 'data_space'.
        """
        super().__init__(forward_problem, formalism=formalism)

        if (
            self.forward_problem.data_error_measure_set
            and self.formalism == "model_space"
        ):
            self.assert_inverse_data_covariance()

    def with_formalism(
        self, formalism: Literal["model_space", "data_space"]
    ) -> LinearMinimumNormInversion:
        """
        Returns a new instance of the inversion using the specified formalism.

        Args:
            formalism: The algebraic space in which the normal equations should be
                assembled and solved. Must be 'model_space' or 'data_space'.

        Returns:
            A new LinearMinimumNormInversion instance with the updated formalism.
        """
        return type(self)(self.forward_problem, formalism=formalism)

    # =========================================================================
    # Inversion Operators
    # =========================================================================

    def minimum_norm_operator(
        self,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[Union[LinearOperator, LinearSolver]] = None,
        significance_level: float = 0.95,
        minimum_damping: float = 0.0,
        maxiter: int = 100,
        rtol: float = 1.0e-6,
        atol: float = 0.0,
    ) -> Union[NonLinearOperator, LinearOperator]:
        """
        Maps data to the minimum-norm solution matching target chi-squared.

        The returned NonLinearOperator includes the exact analytical Fréchet
        derivative of the discrepancy search, complete with its adjoint mapping.
        """
        if self.forward_problem.data_error_measure_set:
            critical_value = self.forward_problem.critical_chi_squared(
                significance_level
            )
            lsq_inversion = LinearLeastSquaresInversion(
                self.forward_problem, formalism=self.formalism
            )

            def get_model_for_damping(
                damping: float, data: Vector, sys_sol0: Optional[Vector] = None
            ) -> tuple[Vector, float, Vector]:
                """
                Solves the normal equations. Returns the model, the chi-squared misfit,
                and the raw system solution (for iterative warm starts).
                """
                normal_operator = lsq_inversion.normal_operator(damping)
                normal_rhs = lsq_inversion.normal_rhs(data)

                res_precond = None
                if preconditioner is not None:
                    if isinstance(preconditioner, LinearOperator):
                        res_precond = preconditioner
                    else:
                        res_precond = preconditioner(normal_operator)

                if isinstance(solver, IterativeLinearSolver):
                    sys_sol = solver.solve_linear_system(
                        normal_operator, res_precond, normal_rhs, sys_sol0
                    )
                else:
                    inverse_normal_operator = solver(normal_operator)
                    sys_sol = inverse_normal_operator(normal_rhs)

                # Recover the model from the raw linear system solution
                if self.formalism == "model_space":
                    model = sys_sol
                else:
                    # In data space, sys_sol is the dual variable v'
                    model = self.forward_problem.forward_operator.adjoint(sys_sol)

                chi_squared = self.forward_problem.chi_squared(model, data)
                return model, chi_squared, sys_sol

            def _solve_discrepancy(data: Vector) -> tuple[Vector, float]:
                """Internal helper to find the model and optimal damping."""
                chi_squared = self.forward_problem.chi_squared_from_residual(data)

                if chi_squared <= critical_value:
                    return self.model_space.zero, 0.0

                damping = 1.0
                _, chi_squared, _ = get_model_for_damping(damping, data)
                damping_lower = damping if chi_squared <= critical_value else None
                damping_upper = damping if chi_squared > critical_value else None

                it = 0
                if damping_lower is None:
                    while chi_squared > critical_value and it < maxiter:
                        it += 1
                        damping /= 2.0
                        _, chi_squared, _ = get_model_for_damping(damping, data)
                        if damping < minimum_damping:
                            raise RuntimeError("Discrepancy principle failed.")
                    damping_lower = damping

                it = 0
                if damping_upper is None:
                    while chi_squared < critical_value and it < maxiter:
                        it += 1
                        damping *= 2.0
                        _, chi_squared, _ = get_model_for_damping(damping, data)
                    damping_upper = damping

                sys_sol0 = None
                for _ in range(maxiter):
                    damping = 0.5 * (damping_lower + damping_upper)

                    # Pass the previous linear system solution to warm-start the iterative solver
                    model, chi_squared, sys_sol = get_model_for_damping(
                        damping, data, sys_sol0
                    )

                    if chi_squared < critical_value:
                        damping_lower = damping
                    else:
                        damping_upper = damping

                    # Bug fix: use damping_upper as the scale, not
                    # (lower + upper). When lower→0 the old criterion
                    # collapses to rtol*upper, making the threshold
                    # proportionally tiny and causing maxiter exhaustion.
                    if damping_upper - damping_lower < atol + rtol * damping_upper:
                        return model, damping

                    sys_sol0 = sys_sol

                raise RuntimeError("Bracketing search failed to converge.")

            def mapping(data: Vector) -> Vector:
                model, _ = _solve_discrepancy(data)
                return model

            def derivative(data: Vector) -> LinearOperator:
                model, damping = _solve_discrepancy(data)

                if damping == 0.0:
                    return self.data_space.zero_operator(self.model_space)

                # 1. Standard Tikhonov linear part (L)
                lsq_op = lsq_inversion.least_squares_operator(
                    damping, solver, preconditioner=preconditioner
                )
                linear_part = getattr(lsq_op, "linear_part", lsq_op)

                # 2. Reconstruct H^{-1} for the denominator term
                normal_operator = lsq_inversion.normal_operator(damping)
                res_precond = None
                if preconditioner is not None:
                    if isinstance(preconditioner, LinearOperator):
                        res_precond = preconditioner
                    else:
                        res_precond = preconditioner(normal_operator)

                if isinstance(solver, IterativeLinearSolver):
                    inv_normal_op = solver(normal_operator, preconditioner=res_precond)
                else:
                    inv_normal_op = solver(normal_operator)

                # Push-through identity application for data_space formulation
                if self.formalism == "model_space":
                    h_inv_u = inv_normal_op(model)
                else:
                    A = self.forward_problem.forward_operator
                    Au = A(model)
                    Hd_inv_Au = inv_normal_op(Au)
                    A_star_Hd_inv_Au = A.adjoint(Hd_inv_Au)
                    h_inv_u = self.model_space.multiply(
                        1.0 / damping,
                        self.model_space.subtract(model, A_star_Hd_inv_Au),
                    )

                denominator = self.model_space.inner_product(model, h_inv_u)

                residual = self.data_space.subtract(
                    self.forward_problem.forward_operator(model), data
                )
                r_inv_residual = (
                    self.forward_problem.data_error_measure.inverse_covariance(residual)
                )

                l_star_u = linear_part.adjoint(model)
                scaled_r_inv_res = self.data_space.multiply(
                    1.0 / damping, r_inv_residual
                )
                w_unscaled = self.data_space.add(l_star_u, scaled_r_inv_res)
                w = self.data_space.multiply(1.0 / denominator, w_unscaled)

                def df_action(delta_data: Vector) -> Vector:
                    du_fixed = linear_part(delta_data)
                    delta_lambda = self.data_space.inner_product(w, delta_data)
                    return self.model_space.subtract(
                        du_fixed, self.model_space.multiply(delta_lambda, h_inv_u)
                    )

                def df_adjoint(delta_model: Vector) -> Vector:
                    l_star_dy = linear_part.adjoint(delta_model)
                    scalar = self.model_space.inner_product(h_inv_u, delta_model)
                    return self.data_space.subtract(
                        l_star_dy, self.data_space.multiply(scalar, w)
                    )

                return LinearOperator(
                    self.data_space,
                    self.model_space,
                    df_action,
                    adjoint_mapping=df_adjoint,
                )

            return NonLinearOperator(
                self.data_space, self.model_space, mapping, derivative=derivative
            )

        else:
            forward_operator = self.forward_problem.forward_operator
            normal_operator = forward_operator @ forward_operator.adjoint
            inverse_normal_operator = solver(normal_operator)
            return forward_operator.adjoint @ inverse_normal_operator


class ConstrainedLinearMinimumNormInversion(LinearInversion):
    """
    Finds the minimum-norm solution subject to an affine subspace constraint.

    This class solves the regularized inverse problem using the discrepancy
    principle while strictly confining the solution to an affine subspace.
    """

    # =========================================================================
    # Initialization & State
    # =========================================================================

    def __init__(
        self,
        forward_problem: LinearForwardProblem,
        constraint: AffineSubspace,
        /,
        *,
        formalism: Literal["model_space", "data_space"] = "data_space",
    ) -> None:
        """
        Initializes the constrained minimum norm inversion.

        Args:
            forward_problem: The original linear forward problem.
            constraint: The affine subspace defining the allowed model states.
            formalism: The algebraic space in which the normal equations are
                assembled and solved. Defaults to 'data_space'.
        """
        super().__init__(forward_problem, formalism=formalism)

        if (
            self.forward_problem.data_error_measure_set
            and self.formalism == "model_space"
        ):
            self.assert_inverse_data_covariance()

        self._constraint = constraint

        proj_op = constraint.projection_operator
        self._u_base = proj_op.translation_part

        reduced_operator = forward_problem.forward_operator @ proj_op.linear_part

        self._reduced_forward_problem = LinearForwardProblem(
            reduced_operator,
            data_error_measure=(
                forward_problem.data_error_measure
                if forward_problem.data_error_measure_set
                else None
            ),
        )
        self._unconstrained_inversion = LinearMinimumNormInversion(
            self._reduced_forward_problem, formalism=formalism
        )

    def with_formalism(
        self, formalism: Literal["model_space", "data_space"]
    ) -> ConstrainedLinearMinimumNormInversion:
        """
        Returns a new instance of the constrained inversion using the specified formalism.

        Args:
            formalism: The algebraic space in which the normal equations should be
                assembled and solved. Must be 'model_space' or 'data_space'.

        Returns:
            A new ConstrainedLinearMinimumNormInversion instance with the updated formalism.
        """
        return type(self)(self.forward_problem, self._constraint, formalism=formalism)

    # =========================================================================
    # Inversion Operators
    # =========================================================================

    def _get_reduced_operator(
        self,
        solver: LinearSolver,
        preconditioner: Optional[Union[LinearOperator, LinearSolver]] = None,
        significance_level: float = 0.95,
        minimum_damping: float = 0.0,
        maxiter: int = 100,
        rtol: float = 1.0e-6,
        atol: float = 0.0,
    ) -> NonLinearOperator:
        """
        Internal helper to construct the unconstrained reduced operator.
        """
        return self._unconstrained_inversion.minimum_norm_operator(
            solver,
            preconditioner=preconditioner,
            significance_level=significance_level,
            minimum_damping=minimum_damping,
            maxiter=maxiter,
            rtol=rtol,
            atol=atol,
        )

    def minimum_norm_operator(
        self,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[Union[LinearOperator, LinearSolver]] = None,
        significance_level: float = 0.95,
        minimum_damping: float = 0.0,
        maxiter: int = 100,
        rtol: float = 1.0e-6,
        atol: float = 0.0,
    ) -> NonLinearOperator:
        """
        Returns an operator that maps data to the constrained minimum-norm solution.
        The operator has its derivative set.
        """
        reduced_op = self._get_reduced_operator(
            solver,
            preconditioner,
            significance_level,
            minimum_damping,
            maxiter,
            rtol,
            atol,
        )

        data_offset = self.forward_problem.forward_operator(self._u_base)
        domain = self.data_space
        codomain = self.model_space

        def mapping(d: Vector) -> Vector:
            d_tilde = domain.subtract(d, data_offset)
            w = reduced_op(d_tilde)
            return codomain.add(self._u_base, w)

        def derivative(d: Vector) -> LinearOperator:
            d_tilde = domain.subtract(d, data_offset)
            return reduced_op.derivative(d_tilde)

        return NonLinearOperator(domain, codomain, mapping, derivative=derivative)

    def constraint_value_mapping(
        self,
        data: Vector,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[Union[LinearOperator, LinearSolver]] = None,
        significance_level: float = 0.95,
        minimum_damping: float = 0.0,
        maxiter: int = 100,
        rtol: float = 1.0e-6,
        atol: float = 0.0,
    ) -> NonLinearOperator:
        r"""
        Returns an operator mapping a constraint value 'w' to the corresponding
        constrained minimum norm solution 'u' for a strictly fixed dataset.
        The operator has its derivative set.
        """
        if not self._constraint.has_explicit_equation:
            raise ValueError(
                "Cannot compute mapping with respect to w: the constraint "
                "was defined geometrically without an explicit equation."
            )

        B_dagger = self._constraint.pseudo_inverse

        reduced_op = self._get_reduced_operator(
            solver,
            preconditioner,
            significance_level,
            minimum_damping,
            maxiter,
            rtol,
            atol,
        )

        A = self.forward_problem.forward_operator
        Id = self.model_space.identity_operator()

        domain = self._constraint.constraint_operator.codomain
        codomain = self.model_space

        def mapping(w: Vector) -> Vector:
            u_base = B_dagger(w)
            d_offset = A(u_base)
            d_tilde = self.data_space.subtract(data, d_offset)

            u_reduced = reduced_op(d_tilde)
            return self.model_space.add(u_base, u_reduced)

        def derivative(w: Vector) -> LinearOperator:
            u_base = B_dagger(w)
            d_offset = A(u_base)
            d_tilde = self.data_space.subtract(data, d_offset)

            D_unc = reduced_op.derivative(d_tilde)

            return (Id - D_unc @ A) @ B_dagger

        return NonLinearOperator(domain, codomain, mapping, derivative=derivative)

    # =========================================================================
    # Surrogates & Parameterization
    # =========================================================================

    def parameterized_inversion(
        self,
        parameterization: LinearOperator,
        /,
        *,
        dense: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
        formalism: Optional[Literal["model_space", "data_space"]] = None,
    ) -> ConstrainedLinearMinimumNormInversion:
        """
        Constructs a parameterized surrogate of the constrained minimum norm inversion.

        Args:
            parameterization: A LinearOperator mapping from the parameter
                space to the full model space.
            dense: If True, computes and stores operators as dense matrices.
            parallel: If True, computes the dense matrices in parallel.
            n_jobs: Number of CPU cores to use. -1 means all available.
            formalism: An optional override for the formalism of the new inversion.
                If None, inherits the formalism of the parent inversion.

        Returns:
            A new ConstrainedLinearMinimumNormInversion instance operating on
            the parameter space.
        """
        target_formalism = formalism or self.formalism

        if not self._constraint.has_explicit_equation:
            raise NotImplementedError(
                "Parameterized inversion for constrained problems is only "
                "supported for subspaces defined by an explicit linear equation."
            )

        B = self._constraint.constraint_operator
        w = self._constraint.constraint_value

        if B.codomain.dim > parameterization.domain.dim:
            raise ValueError(
                f"The parameter space dimension ({parameterization.domain.dim}) is "
                f"smaller than the number of constraints ({B.codomain.dim})."
            )

        new_B = B @ parameterization
        if dense:
            new_B = new_B.with_dense_matrix(parallel=parallel, n_jobs=n_jobs)

        from .subspaces import AffineSubspace

        new_S = AffineSubspace.from_linear_equation(
            new_B,
            w,
            solver=self._constraint.solver,
            preconditioner=self._constraint.preconditioner,
        )

        new_fp = self.forward_problem.parameterized_problem(
            parameterization, dense=dense, parallel=parallel, n_jobs=n_jobs
        )

        return type(self)(new_fp, new_S, formalism=target_formalism)
