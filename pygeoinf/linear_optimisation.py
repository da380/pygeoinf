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

    def __init__(self, forward_problem: "LinearForwardProblem", /) -> None:
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
        solver: "LinearSolver",
        /,
        *,
        preconditioner: Optional[Union[LinearOperator, LinearSolver]] = None,
    ) -> Union[NonLinearOperator, LinearOperator]:
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

        # Resolve the preconditioner if a method (LinearSolver) is provided
        resolved_preconditioner = None
        if preconditioner is not None:
            if isinstance(preconditioner, LinearOperator):
                resolved_preconditioner = preconditioner
            elif isinstance(preconditioner, LinearSolver):
                # Call the preconditioning method on the normal operator
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

            def mapping(data: Vector) -> Vector:
                shifted_data = self.forward_problem.data_space.subtract(
                    data, self.forward_problem.data_error_measure.expectation
                )
                return (
                    inverse_normal_operator
                    @ forward_operator.adjoint
                    @ inverse_data_covariance
                )(shifted_data)

            return NonLinearOperator(self.data_space, self.model_space, mapping)
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
    ) -> NonLinearOperator:
        """Maps data to the constrained least-squares solution."""
        reduced_op = self._unconstrained_inversion.least_squares_operator(
            damping, solver, preconditioner=preconditioner, **kwargs
        )

        data_offset = self.forward_problem.forward_operator(self._u_base)
        domain = self.data_space
        codomain = self.model_space

        def mapping(d: Vector) -> Vector:
            d_tilde = domain.subtract(d, data_offset)
            w = reduced_op(d_tilde)
            return codomain.add(self._u_base, w)

        return NonLinearOperator(domain, codomain, mapping)


class LinearMinimumNormInversion(LinearInversion):
    """Finds a regularized solution using the discrepancy principle."""

    def __init__(self, forward_problem: "LinearForwardProblem", /) -> None:
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

                # Resolve preconditioner for the specific trial damping alpha
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

            def mapping(data: Vector) -> Vector:
                # Bracketing search logic
                chi_squared = self.forward_problem.chi_squared_from_residual(data)
                if chi_squared <= critical_value:
                    return self.model_space.zero

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
                        return model
                    model0 = model

                raise RuntimeError("Bracketing search failed to converge.")

            return NonLinearOperator(self.data_space, self.model_space, mapping)
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

        return NonLinearOperator(domain, codomain, mapping)
