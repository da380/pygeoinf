"""
Implements optimisation-based methods for solving linear inverse problems.

This module provides classical, deterministic approaches to inversion that seek
a single "best-fit" model. These methods are typically formulated as finding
the model `u` that minimizes a cost functional.

Key Classes
-----------
- `LinearLeastSquaresInversion`: Solves the inverse problem by minimizing a
  Tikhonov-regularized least-squares functional.
- `LinearMinimumNormInversion`: Finds the model with the smallest norm that
  fits the data to a statistically acceptable degree using the discrepancy
  principle.
- `ConstrainedLinearLeastSquaresInversion`: Solves a linear inverse problem
  subject to an affine subspace constraint.
"""

from __future__ import annotations
from typing import Optional, Union

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

    This method finds the model `u` that minimizes the functional:
    `J(u) = ||A(u) - d||² + α² * ||u||²`
    """

    def __init__(self, forward_problem: LinearForwardProblem, /) -> None:
        super().__init__(forward_problem)
        if self.forward_problem.data_error_measure_set:
            self.assert_inverse_data_covariance()

    def normal_operator(self, damping: float) -> LinearOperator:
        """Returns the Tikhonov-regularized normal operator (A*WA + αI)."""
        if damping < 0:
            raise ValueError("Damping parameter must be non-negative.")

        forward_operator = self.forward_problem.forward_operator
        identity = self.forward_problem.model_space.identity_operator()

        if self.forward_problem.data_error_measure_set:
            inverse_data_covariance = (
                self.forward_problem.data_error_measure.inverse_covariance
            )
            return (
                forward_operator.adjoint @ inverse_data_covariance @ forward_operator
                + damping * identity
            )
        else:
            return forward_operator.adjoint @ forward_operator + damping * identity

    def normal_rhs(self, data: Vector) -> Vector:
        """Returns the right hand side of the normal equations (A*W d)."""
        forward_operator = self.forward_problem.forward_operator

        if self.forward_problem.data_error_measure_set:
            inverse_data_covariance = (
                self.forward_problem.data_error_measure.inverse_covariance
            )
            shifted_data = self.forward_problem.data_space.subtract(
                data, self.forward_problem.data_error_measure.expectation
            )
            return (forward_operator.adjoint @ inverse_data_covariance)(shifted_data)
        else:
            return forward_operator.adjoint(data)

    def least_squares_operator(
        self,
        damping: float,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[Union[LinearOperator, LinearSolver]] = None,
    ) -> Union[LinearOperator, AffineOperator]:
        """
        Returns an operator that maps data to the least-squares solution.

        Args:
            damping: The Tikhonov damping parameter, alpha.
            solver: The linear solver for inverting the normal operator.
            preconditioner: Either a direct LinearOperator or a LinearSolver
                method (factory) used to generate the preconditioner.
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

        if self.forward_problem.data_error_measure_set:

            inverse_data_covariance = (
                self.forward_problem.data_error_measure.inverse_covariance
            )

            linear_part = (
                inverse_normal_operator
                @ forward_operator.adjoint
                @ inverse_data_covariance
            )

            expected_data = self.forward_problem.data_error_measure.expectation
            translation = self.model_space.negative(linear_part(expected_data))

            return AffineOperator(linear_part, translation)
        else:
            return inverse_normal_operator @ forward_operator.adjoint


class ConstrainedLinearLeastSquaresInversion(LinearInversion):
    """Solves a linear inverse problem subject to an affine subspace constraint."""

    def __init__(
        self, forward_problem: LinearForwardProblem, constraint: AffineSubspace
    ) -> None:
        super().__init__(forward_problem)
        self._constraint = constraint
        self._u_base = constraint.domain.subtract(
            constraint.translation, constraint.projector(constraint.translation)
        )

        reduced_operator = forward_problem.forward_operator @ constraint.projector
        self._reduced_forward_problem = LinearForwardProblem(
            reduced_operator,
            data_error_measure=(
                forward_problem.data_error_measure
                if forward_problem.data_error_measure_set
                else None
            ),
        )

        self._unconstrained_inversion = LinearLeastSquaresInversion(
            self._reduced_forward_problem
        )

    def least_squares_operator(
        self,
        damping: float,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[Union[LinearOperator, LinearSolver]] = None,
        **kwargs,
    ) -> AffineOperator:
        """Maps data to the constrained least-squares solution."""
        reduced_op = self._unconstrained_inversion.least_squares_operator(
            damping, solver, preconditioner=preconditioner, **kwargs
        )

        data_offset = self.forward_problem.forward_operator(self._u_base)
        domain = self.data_space
        codomain = self.model_space

        shift_in = AffineOperator(
            domain.identity_operator(), domain.negative(data_offset)
        )
        shift_out = AffineOperator(codomain.identity_operator(), self._u_base)

        return shift_out @ reduced_op @ shift_in


class LinearMinimumNormInversion(LinearInversion):
    """Finds a regularized solution using the discrepancy principle."""

    def __init__(self, forward_problem: LinearForwardProblem, /) -> None:
        super().__init__(forward_problem)
        if self.forward_problem.data_error_measure_set:
            self.assert_inverse_data_covariance()

    def minimum_norm_operator(
        self,
        solver: "LinearSolver",
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
            lsq_inversion = LinearLeastSquaresInversion(self.forward_problem)

            def get_model_for_damping(
                damping: float, data: Vector, model0: Optional[Vector] = None
            ) -> tuple[Vector, float]:
                normal_operator = lsq_inversion.normal_operator(damping)
                normal_rhs = lsq_inversion.normal_rhs(data)

                res_precond = None
                if preconditioner is not None:
                    if isinstance(preconditioner, LinearOperator):
                        res_precond = preconditioner
                    else:
                        res_precond = preconditioner(normal_operator)

                if isinstance(solver, IterativeLinearSolver):
                    model = solver.solve_linear_system(
                        normal_operator, res_precond, normal_rhs, model0
                    )
                else:
                    inverse_normal_operator = solver(normal_operator)
                    model = inverse_normal_operator(normal_rhs)

                chi_squared = self.forward_problem.chi_squared(model, data)
                return model, chi_squared

            def _solve_discrepancy(data: Vector) -> tuple[Vector, float]:
                """Internal helper to find the model and optimal damping."""
                chi_squared = self.forward_problem.chi_squared_from_residual(data)

                # If strictly inside the acceptable ball, zero model fits perfectly.
                if chi_squared <= critical_value:
                    return self.model_space.zero, 0.0

                damping = 1.0
                _, chi_squared = get_model_for_damping(damping, data)
                damping_lower = damping if chi_squared <= critical_value else None
                damping_upper = damping if chi_squared > critical_value else None

                it = 0
                if damping_lower is None:
                    while chi_squared > critical_value and it < maxiter:
                        it += 1
                        damping /= 2.0
                        _, chi_squared = get_model_for_damping(damping, data)
                        if damping < minimum_damping:
                            raise RuntimeError("Discrepancy principle failed.")
                    damping_lower = damping

                it = 0
                if damping_upper is None:
                    while chi_squared < critical_value and it < maxiter:
                        it += 1
                        damping *= 2.0
                        _, chi_squared = get_model_for_damping(damping, data)
                    damping_upper = damping

                model0 = None
                for _ in range(maxiter):
                    damping = 0.5 * (damping_lower + damping_upper)
                    model, chi_squared = get_model_for_damping(damping, data, model0)

                    if chi_squared < critical_value:
                        damping_lower = damping
                    else:
                        damping_upper = damping

                    if damping_upper - damping_lower < atol + rtol * (
                        damping_lower + damping_upper
                    ):
                        return model, damping
                    model0 = model

                raise RuntimeError("Bracketing search failed to converge.")

            def mapping(data: Vector) -> Vector:
                model, _ = _solve_discrepancy(data)
                return model

            def derivative(data: Vector) -> LinearOperator:
                model, damping = _solve_discrepancy(data)

                # If data is inside the chi-squared ball, the solution is constantly
                # the zero vector, making the derivative exactly zero.
                if damping == 0.0:
                    return self.model_space.zero_operator(self.data_space)

                # 1. Standard Tikhonov linear part (L)
                lsq_op = lsq_inversion.least_squares_operator(
                    damping, solver, preconditioner=preconditioner
                )
                linear_part = lsq_op.linear_part

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

                # 3. Pre-compute cached vectors to accelerate linear/adjoint actions
                h_inv_u = inv_normal_op(model)
                denominator = self.model_space.inner_product(model, h_inv_u)

                residual = self.data_space.subtract(
                    self.forward_problem.forward_operator(model), data
                )
                r_inv_residual = (
                    self.forward_problem.data_error_measure.inverse_covariance(residual)
                )

                # Pre-compute w = (1/denom) * (L* u + (1/lambda) R^{-1} r)
                l_star_u = linear_part.adjoint(model)
                scaled_r_inv_res = self.data_space.multiply(
                    1.0 / damping, r_inv_residual
                )
                w_unscaled = self.data_space.add(l_star_u, scaled_r_inv_res)
                w = self.data_space.multiply(1.0 / denominator, w_unscaled)

                # 4. Forward Fréchet derivative action
                def df_action(delta_data: Vector) -> Vector:
                    du_fixed = linear_part(delta_data)
                    delta_lambda = self.data_space.inner_product(w, delta_data)
                    return self.model_space.subtract(
                        du_fixed, self.model_space.multiply(delta_lambda, h_inv_u)
                    )

                # 5. Adjoint Fréchet derivative action
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
    """Finds min-norm solution subject to affine subspace constraint."""

    def __init__(
        self, forward_problem: LinearForwardProblem, constraint: AffineSubspace
    ) -> None:
        super().__init__(forward_problem)
        if self.forward_problem.data_error_measure_set:
            self.assert_inverse_data_covariance()
        self._constraint = constraint
        self._u_base = constraint.domain.subtract(
            constraint.translation, constraint.projector(constraint.translation)
        )

        reduced_operator = forward_problem.forward_operator @ constraint.projector
        self._reduced_forward_problem = LinearForwardProblem(
            reduced_operator,
            data_error_measure=(
                forward_problem.data_error_measure
                if forward_problem.data_error_measure_set
                else None
            ),
        )
        self._unconstrained_inversion = LinearMinimumNormInversion(
            self._reduced_forward_problem
        )

    def minimum_norm_operator(
        self,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[Union[LinearOperator, LinearSolver]] = None,
        **kwargs,
    ) -> NonLinearOperator:
        """Returns operator for constrained discrepancy principle inversion."""

        # Pass all kwargs directly to the underlying solver
        reduced_op = self._unconstrained_inversion.minimum_norm_operator(
            solver, preconditioner=preconditioner, **kwargs
        )

        data_offset = self.forward_problem.forward_operator(self._u_base)
        domain = self.data_space
        codomain = self.model_space

        def mapping(d: Vector) -> Vector:
            d_tilde = domain.subtract(d, data_offset)
            w = reduced_op(d_tilde)
            return codomain.add(self._u_base, w)

        def derivative(d: Vector) -> LinearOperator:
            # The derivative of u_base + F(d - d_offset) is just F'(d - d_offset)
            d_tilde = domain.subtract(d, data_offset)
            return reduced_op.derivative(d_tilde)

        return NonLinearOperator(
            domain,
            codomain,
            mapping,
            derivative=derivative,
        )

    def constraint_value_mapping(
        self,
        data: Vector,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[Union[LinearOperator, LinearSolver]] = None,
        **kwargs,
    ) -> NonLinearOperator:
        """
        Returns a NonLinearOperator mapping the constraint value 'w' to the
        constrained minimum norm solution 'u' for a fixed dataset 'd'.

        This operator encapsulates both the forward mapping u(w) and its
        analytical Fréchet derivative with respect to w. It provides the exact
        sensitivity field required to bound property regions in hypothesis
        testing (e.g., Backus-style inference).
        """
        if not self._constraint.has_explicit_equation:
            raise ValueError(
                "Cannot compute mapping with respect to w: the constraint "
                "was defined geometrically without an explicit equation."
            )

        # 1. Retrieve the constraint algebra stored in the AffineSubspace
        B = self._constraint.constraint_operator
        constraint_solver = self._constraint.solver

        # 2. Build the right pseudo-inverse: B_dagger = B* @ (B B*)^{-1}
        # This maps any constraint value w to its minimum-norm base vector in U.
        G = B @ B.adjoint
        if isinstance(constraint_solver, IterativeLinearSolver):
            G_inv = constraint_solver(G)
        else:
            G_inv = constraint_solver(G)

        B_dagger = B.adjoint @ G_inv

        # 3. Get the reduced unconstrained solver
        # The underlying reduced forward problem only depends on the tangent
        # projector, which is invariant to 'w'. So this operator is universally valid!
        reduced_op = self._unconstrained_inversion.minimum_norm_operator(
            solver, preconditioner=preconditioner, **kwargs
        )

        A = self.forward_problem.forward_operator
        Id = self.model_space.identity_operator()

        domain = B.codomain  # The constraint space (W)
        codomain = self.model_space  # The model space (U)

        def mapping(w: Vector) -> Vector:
            # u(w) = u_base(w) + F_u(d - A u_base(w))
            u_base = B_dagger(w)
            d_offset = A(u_base)
            d_tilde = self.data_space.subtract(data, d_offset)

            u_reduced = reduced_op(d_tilde)
            return self.model_space.add(u_base, u_reduced)

        def derivative(w: Vector) -> LinearOperator:
            # D_w = (I - D_unc @ A) @ B_dagger
            u_base = B_dagger(w)
            d_offset = A(u_base)
            d_tilde = self.data_space.subtract(data, d_offset)

            # Evaluate the Fréchet derivative of the discrepancy search at d_tilde
            D_unc = reduced_op.derivative(d_tilde)

            # Chain rule composition
            return (Id - D_unc @ A) @ B_dagger

        return NonLinearOperator(domain, codomain, mapping, derivative=derivative)
