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

        The damping parameter :math:`\\alpha` is chosen by the discrepancy
        principle: find :math:`\\alpha^*` such that
        :math:`\\chi^2(c^\\dagger(\\alpha^*)) = \\chi^2_\\text{critical}`.

        The algorithm works in three stages:

        1. **Bracket lower bound** — starting from ``damping = 1.0`` halve
           repeatedly until :math:`\\chi^2 \\leq \\chi^2_\\text{critical}`.
        2. **Bracket upper bound** — if not already found, double repeatedly
           until :math:`\\chi^2 > \\chi^2_\\text{critical}`.
        3. **Bisect** — standard interval bisection on :math:`[\\alpha_\\text{lo},
           \\alpha_\\text{hi}]` until
           :math:`\\alpha_\\text{hi} - \\alpha_\\text{lo} <
           \\texttt{atol} + \\texttt{rtol}\\cdot\\alpha_\\text{hi}`.

        **Feasibility pre-condition**: the discrepancy principle has a solution
        only when the *chi-squared floor* — the minimum achievable
        :math:`\\chi^2` at any model — is strictly less than
        :math:`\\chi^2_\\text{critical}`.  The floor equals the noise power
        projected onto the null space of :math:`G^\\top`, which is
        :math:`n_\\text{rays} - \\operatorname{rank}G` in expectation.  If the
        forward-model discretisation error is large relative to the measurement
        noise the floor can exceed the critical value, making the problem
        infeasible.  Callers that want a robust outer loop should estimate the
        floor (one LU solve at tiny damping) and inflate the noise before
        calling this method.

        .. rubric:: Known bugs fixed (2026-03)

        **Bug 1 — silent false bracket (the critical bug)**:
        Previously the halving loop set ``damping_lower = damping``
        unconditionally after the ``while`` loop, even if the loop exited by
        exhausting ``maxiter`` iterations *without* chi-squared ever crossing
        the critical value.  This placed a meaningless tiny value (``1/2^100``)
        in ``damping_lower`` and started bisection on a completely unbracketed
        interval, always converging to ``damping ≈ 0`` while chi-squared
        remained above critical — ultimately hitting the
        ``'Bracketing search failed to converge'`` error.  Fixed by checking
        ``chi_squared <= critical_value`` **after** the loop and raising a
        descriptive ``RuntimeError`` immediately if infeasible.

        **Bug 2 — convergence criterion degenerates when lower ≈ 0**:
        The bisection stopping test was
        ``width < atol + rtol * (lower + upper)``.
        When ``lower`` approaches zero the scale collapses to
        ``rtol * upper``.  For a crossover at :math:`\\alpha^* \\approx 10^{-8}`
        this becomes ``1e-6 × 1e-8 = 1e-14``, but the interval width is
        ``~5e-9`` — the criterion is never satisfied and the loop terminates
        only by exhausting ``maxiter``.  Fixed by using
        ``atol + rtol * upper`` (scale on the larger endpoint), which correctly
        tests relative convergence regardless of how small ``lower`` becomes.
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
                    # Bug 1 fix: ensure the loop actually found a crossing before
                    # claiming we have a valid lower bracket.  Previously this
                    # assignment was unconditional, placing a meaningless tiny
                    # value in damping_lower and starting bisection on an
                    # unbracketed interval when chi-squared never fell below
                    # the critical value.
                    if chi_squared > critical_value:
                        raise RuntimeError(
                            "Discrepancy principle infeasible: the minimum "
                            "achievable chi-squared (chi2_floor) never falls "
                            "below chi2_critical even at vanishing damping. "
                            "This typically means the forward-model "
                            "discretisation error is large relative to the "
                            "measurement noise. Consider refining the spatial "
                            "grid, increasing noise_std, or inflating the "
                            "effective noise covariance before calling this "
                            "method."
                        )
                    damping_lower = damping

                it = 0
                if damping_upper is None:
                    while chi_squared < critical_value and it < maxiter:
                        it += 1
                        damping *= 2.0
                        _, chi_squared = get_model_for_damping(damping, data)
                    if chi_squared < critical_value:
                        raise RuntimeError(
                            "Discrepancy principle search failed: chi-squared "
                            "did not exceed critical_value even after doubling "
                            f"damping {maxiter} times."
                        )
                    damping_upper = damping

                model0 = None
                for _ in range(maxiter):
                    damping = 0.5 * (damping_lower + damping_upper)
                    model, chi_squared = get_model_for_damping(damping, data, model0)

                    if chi_squared < critical_value:
                        damping_lower = damping
                    else:
                        damping_upper = damping

                    # Bug 2 fix: use damping_upper as the scale, not
                    # (lower + upper).  The old criterion
                    # "atol + rtol*(lower+upper)" collapses to
                    # "rtol*upper" when lower→0, making the threshold
                    # proportionally tiny (e.g. 1e-6 × 1e-8 = 1e-14)
                    # while the interval width is still ~lower.  Using
                    # damping_upper as scale gives correct relative
                    # convergence regardless of how small lower becomes.
                    if damping_upper - damping_lower < atol + rtol * damping_upper:
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
