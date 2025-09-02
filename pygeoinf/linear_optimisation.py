"""
Implements optimisation-based methods for solving linear inverse problems.

This module provides classical, deterministic approaches to inversion that seek
a single "best-fit" model. These methods are typically formulated as finding
the model `u` that minimizes a cost functional.

The primary goal is to find a stable solution to an ill-posed problem by
incorporating regularization, which balances fitting the data with controlling
the complexity or norm of the solution.

Key Classes
-----------
- `LinearLeastSquaresInversion`: Solves the inverse problem by minimizing a
  Tikhonov-regularized least-squares functional.
- `LinearMinimumNormInversion`: Finds the model with the smallest norm that
  fits the data to a statistically acceptable degree using the discrepancy
  principle.
"""

from __future__ import annotations
from typing import Optional, Union

from .operators import Operator
from .inversion import Inversion


from .forward_problem import LinearForwardProblem
from .operators import LinearOperator
from .linear_solvers import LinearSolver, IterativeLinearSolver
from .hilbert_space import Vector


class LinearLeastSquaresInversion(Inversion):
    """
    Solves a linear inverse problem using Tikhonov-regularized least-squares.

    This method finds the model `u` that minimizes the functional:
    `J(u) = ||A(u) - d||² + α² * ||u||²`
    where `α` is the damping parameter. If a data error covariance is provided,
    the data misfit norm is appropriately weighted by the inverse covariance.
    """

    def __init__(self, forward_problem: "LinearForwardProblem", /) -> None:
        """
        Args:
            forward_problem: The forward problem. If it includes a data error
                measure, the measure's inverse covariance must be defined.
        """
        super().__init__(forward_problem)
        if self.forward_problem.data_error_measure_set:
            self.assert_inverse_data_covariance()

    def normal_operator(self, damping: float) -> LinearOperator:
        """
        Returns the Tikhonov-regularized normal operator.

        This operator, often written as `(A* @ W @ A + α*I)`, forms the left-hand
        side of the normal equations that must be solved to find the least-squares
        solution. `W` is the inverse data covariance (or identity).

        Args:
            damping: The Tikhonov damping parameter, `α`. Must be non-negative.

        Returns:
            The normal operator as a `LinearOperator`.
        """
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

    def least_squares_operator(
        self,
        damping: float,
        solver: "LinearSolver",
        /,
        *,
        preconditioner: Optional[LinearOperator] = None,
    ) -> Union[Operator, LinearOperator]:
        """
        Returns an operator that maps data to the least-squares solution.

        The returned operator `L` gives the solution `u = L(d)`. If the data has
        errors with a non-zero mean, `L` is a general non-linear `Operator`.
        Otherwise, it is a `LinearOperator`.

        Args:
            damping: The Tikhonov damping parameter, `alpha`.
            solver: The linear solver for inverting the normal operator.
            preconditioner: An optional preconditioner for iterative solvers.

        Returns:
            An operator that maps from the data space to the model space.
        """
        forward_operator = self.forward_problem.forward_operator
        normal_operator = self.normal_operator(damping)

        if isinstance(solver, IterativeLinearSolver):
            inverse_normal_operator = solver(
                normal_operator, preconditioner=preconditioner
            )
        else:
            inverse_normal_operator = solver(normal_operator)

        if self.forward_problem.data_error_measure_set:
            inverse_data_covariance = (
                self.forward_problem.data_error_measure.inverse_covariance
            )

            # This mapping is affine, not linear, if the error measure has a non-zero mean.
            def mapping(data: Vector) -> Vector:
                shifted_data = self.forward_problem.data_space.subtract(
                    data, self.forward_problem.data_error_measure.expectation
                )
                return (
                    inverse_normal_operator
                    @ forward_operator.adjoint
                    @ inverse_data_covariance
                )(shifted_data)

            return Operator(self.data_space, self.model_space, mapping)

        else:
            return inverse_normal_operator @ forward_operator.adjoint


class LinearMinimumNormInversion(Inversion):
    """
    Finds a regularized solution using the discrepancy principle.

    This method automatically selects a Tikhonov damping parameter `α` such that
    the resulting solution `u_α` fits the data to a statistically acceptable
    level. It finds the model with the smallest norm `||u||` that satisfies
    the target misfit, as determined by a chi-squared test.
    """

    def __init__(self, forward_problem: "LinearForwardProblem", /) -> None:
        """
        Args:
            forward_problem: The forward problem. Its data error measure and
                inverse covariance must be defined.
        """
        super().__init__(forward_problem)
        if self.forward_problem.data_error_measure_set:
            self.assert_inverse_data_covariance()

    def minimum_norm_operator(
        self,
        solver: "LinearSolver",
        /,
        *,
        preconditioner: Optional[LinearOperator] = None,
        significance_level: float = 0.95,
        minimum_damping: float = 0.0,
        maxiter: int = 100,
        rtol: float = 1.0e-6,
        atol: float = 0.0,
    ) -> Union[Operator, LinearOperator]:
        """
        Returns an operator that maps data to the minimum-norm solution.

        The method uses a bracketing search to finds the damping parameter `alpha`
        such that `chi_squared(u_alpha, d)` matches a critical value. The mapping
        is non-linear if data errors are present.

        Args:
            solver: A solver for the linear systems.
            preconditioner: An optional preconditioner for iterative solvers.
            significance_level: The target significance level for the
                chi-squared test (e.g., 0.95).
            minimum_damping: A floor for the damping parameter search.
            maxiter: Maximum iterations for the bracketing search.
            rtol: Relative tolerance for the damping parameter.
            atol: Absolute tolerance for the damping parameter.

        Returns:
            An operator that maps data to the minimum-norm model.
        """
        if self.forward_problem.data_error_measure_set:
            critical_value = self.forward_problem.critical_chi_squared(
                significance_level
            )
            lsq_inversion = LinearLeastSquaresInversion(self.forward_problem)

            def get_model_for_damping(
                damping: float, data: Vector, model0: Optional[Vector] = None
            ) -> tuple[Vector, float]:
                """Computes the LS model and its chi-squared for a given damping."""
                op = lsq_inversion.least_squares_operator(
                    damping, solver, preconditioner=preconditioner
                )
                model = op(data)
                chi_squared = self.forward_problem.chi_squared(model, data)
                return model, chi_squared

            def mapping(data: Vector) -> Vector:
                """The non-linear mapping from data to the minimum-norm model."""
                model = self.model_space.zero
                chi_squared = self.forward_problem.chi_squared(model, data)
                if chi_squared <= critical_value:
                    return model

                # Find upper and lower bounds for the optimal damping parameter
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
                            raise RuntimeError(
                                "Discrepancy principle has failed; critical value cannot be reached."
                            )
                    damping_lower = damping

                it = 0
                if damping_upper is None:
                    while chi_squared < critical_value and it < maxiter:
                        it += 1
                        damping *= 2.0
                        _, chi_squared = get_model_for_damping(damping, data)
                    damping_upper = damping

                if damping_lower is None or damping_upper is None:
                    raise RuntimeError(
                        "Failed to bracket the optimal damping parameter."
                    )

                # Bracket search for the optimal damping
                for _ in range(maxiter):
                    damping = 0.5 * (damping_lower + damping_upper)
                    model, chi_squared = get_model_for_damping(damping, data)

                    if chi_squared < critical_value:
                        damping_lower = damping
                    else:
                        damping_upper = damping

                    if damping_upper - damping_lower < atol + rtol * (
                        damping_lower + damping_upper
                    ):
                        return model

                raise RuntimeError("Bracketing search failed to converge.")

            return Operator(self.data_space, self.model_space, mapping)

        else:
            # For error-free data, compute the minimum-norm solution via A*(A*A)^-1
            forward_operator = self.forward_problem.forward_operator
            normal_operator = forward_operator @ forward_operator.adjoint
            inverse_normal_operator = solver(normal_operator)
            return forward_operator.adjoint @ inverse_normal_operator
