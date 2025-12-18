"""
Implements the Bayesian framework for solving linear inverse problems.

This module treats the inverse problem from a statistical perspective, aiming to
determine the full posterior probability distribution of the unknown model
parameters, rather than a single best-fit solution.

Key Classes
-----------
- `LinearBayesianInversion`: Computes the posterior Gaussian measure `p(u|d)`
  for the model `u` given observed data `d`.
- `LinearBayesianInference`: Extends the framework to compute the posterior
  distribution for a derived property of the model.
- `ConstrainedLinearBayesianInversion`: Solves the inverse problem subject to
  a hard affine constraint `u in A`, interpreting it as conditioning the prior.
"""

from __future__ import annotations
from typing import Optional

from .inversion import LinearInversion
from .gaussian_measure import GaussianMeasure
from .forward_problem import LinearForwardProblem
from .linear_operators import LinearOperator, NormalSumOperator
from .linear_solvers import LinearSolver, IterativeLinearSolver
from .hilbert_space import Vector
from .subspaces import AffineSubspace


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
        super().__init__(forward_problem)
        self._model_prior_measure: GaussianMeasure = model_prior_measure

    @property
    def model_prior_measure(self) -> GaussianMeasure:
        """The prior Gaussian measure on the model space."""
        return self._model_prior_measure

    @property
    def normal_operator(self) -> LinearOperator:
        """
        Returns the Bayesian Norm operator:

        N = A Q A* + R

        with A the forward operator (with A* its adjoint), Q the model
        prior covariance, and R the data error covariance. For error-free
        problems this operator is reduced to:

        N = A Q A*

        """
        forward_operator = self.forward_problem.forward_operator
        model_prior_covariance = self.model_prior_measure.covariance

        if self.forward_problem.data_error_measure_set:
            return (
                forward_operator @ model_prior_covariance @ forward_operator.adjoint
                + self.forward_problem.data_error_measure.covariance
            )
        else:
            return NormalSumOperator(forward_operator, model_prior_covariance)

    def kalman_operator(
        self,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[LinearOperator] = None,
    ):
        """
        Returns the Kalman gain operator for the problem:

        K = Q A* Ni

        where Q is the model prior covariance, A the forward operator
        (with adjoint A*), and Ni is the inverse of the normal operator.

        Args:
            solver: A linear solver for inverting the normal operator.
            preconditioner: An optional preconditioner for.

        Returns:
            A LinearOperator for the Kalman gain.
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
        Returns the posterior Gaussian measure for the model conditions on the data.

        Args:
            data: The observed data vector.
            solver: A linear solver for inverting the normal operator C_d.
            preconditioner: An optional preconditioner for C_d.
        """
        data_space = self.data_space
        model_space = self.model_space
        forward_operator = self.forward_problem.forward_operator
        model_prior_covariance = self.model_prior_measure.covariance

        kalman_gain = self.kalman_operator(solver, preconditioner=preconditioner)

        # u_bar_post = u_bar + K (v - A u_bar - v_bar)
        shifted_data = data_space.subtract(
            data, forward_operator(self.model_prior_measure.expectation)
        )
        if self.forward_problem.data_error_measure_set:
            shifted_data = data_space.subtract(
                shifted_data, self.forward_problem.data_error_measure.expectation
            )
        mean_update = kalman_gain(shifted_data)
        expectation = model_space.add(self.model_prior_measure.expectation, mean_update)

        # Q_post = Q - K A Q
        covariance = model_prior_covariance - (
            kalman_gain @ forward_operator @ model_prior_covariance
        )

        # Add in a sampling method if that is possible.
        can_sample_prior = self.model_prior_measure.sample_set
        can_sample_noise = (
            not self.forward_problem.data_error_measure_set
            or self.forward_problem.data_error_measure.sample_set
        )

        if can_sample_prior and can_sample_noise:

            if self.forward_problem.data_error_measure_set:
                error_expectation = self.forward_problem.data_error_measure.expectation

            def sample():
                model_sample = self.model_prior_measure.sample()
                prediction = forward_operator(model_sample)
                data_residual = data_space.subtract(data, prediction)

                if self.forward_problem.data_error_measure_set:
                    noise_raw = self.forward_problem.data_error_measure.sample()
                    epsilon = data_space.subtract(noise_raw, error_expectation)
                    data_space.axpy(1.0, epsilon, data_residual)

                correction = kalman_gain(data_residual)
                return model_space.add(model_sample, correction)

            return GaussianMeasure(
                covariance=covariance, expectation=expectation, sample=sample
            )
        else:
            return GaussianMeasure(covariance=covariance, expectation=expectation)


class ConstrainedLinearBayesianInversion(LinearInversion):
    """
    Solves a linear inverse problem using Bayesian methods subject to an
    affine subspace constraint `u in A`.

    This interprets the constraint as conditioning the prior on the subspace.
    The subspace must be defined by a linear equation B(u) = w.
    """

    def __init__(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        constraint: AffineSubspace,
        /,
        *,
        geometric: bool = False,
    ) -> None:
        """
        Args:
            forward_problem: The forward problem.
            model_prior_measure: The unconstrained prior Gaussian measure.
            constraint: The affine subspace A = {u | Bu = w}.
            geometric: If True, uses orthogonal projection to enforce the constraint.
                       If False (default), uses Bayesian conditioning.
        """
        super().__init__(forward_problem)
        self._unconstrained_prior = model_prior_measure
        self._constraint = constraint
        self._geometric = geometric

        if not constraint.has_constraint_equation:
            raise ValueError(
                "For Bayesian inversion, the subspace must be defined by a linear "
                "equation (constraint operator). Use AffineSubspace.from_linear_equation."
            )

    def conditioned_prior_measure(
        self,
        solver: LinearSolver,
        preconditioner: Optional[LinearOperator] = None,
    ) -> GaussianMeasure:
        """
        Computes the prior measure conditioned on the constraint B(u) = w.

        Args:
            solver: Linear solver used to invert the normal operator, BQB*.
            preconditioner: Optional preconditioner for the constraint solver.
        """

        if self._geometric:
            # --- Geometric Approach (Affine Mapping) ---
            # Map: u -> P u + v
            # P = I - B* (B B*)^-1 B
            # v = B* (B B*)^-1 w

            projector = self._constraint.projector
            translation = self._constraint.translation

            return self._unconstrained_prior.affine_mapping(
                operator=projector, translation=translation
            )

        else:
            # --- Bayesian Approach (Statistical Conditioning) ---
            # Treat the constraint as a noiseless observation: w = B(u)

            constraint_op = self._constraint.constraint_operator
            constraint_val = self._constraint.constraint_value

            constraint_problem = LinearForwardProblem(constraint_op)
            constraint_inversion = LinearBayesianInversion(
                constraint_problem, self._unconstrained_prior
            )

            return constraint_inversion.model_posterior_measure(
                constraint_val, solver, preconditioner=preconditioner
            )

    def model_posterior_measure(
        self,
        data: Vector,
        solver: LinearSolver,
        constraint_solver: LinearSolver,
        *,
        preconditioner: Optional[LinearOperator] = None,
        constraint_preconditioner: Optional[LinearOperator] = None,
    ) -> GaussianMeasure:
        """
        Returns the posterior Gaussian measure for the model given the constraint and the data.

        Args:
            data: Observed data vector.
            solver: Solver for the data update (inverts A C_cond A* + Ce).
            constraint_solver: Solver for the prior conditioning (inverts B C_prior B*).
            preconditioner: Preconditioner for the data update (acts on Data Space).
            constraint_preconditioner: Preconditioner for the constraint update (acts on Property Space).
        """
        # 1. Condition Prior (Uses constraint_solver and constraint_preconditioner)
        cond_prior = self.conditioned_prior_measure(
            constraint_solver, preconditioner=constraint_preconditioner
        )

        # 2. Solve Bayesian Inverse Problem (Uses solver and preconditioner)
        bayes_inv = LinearBayesianInversion(self.forward_problem, cond_prior)

        return bayes_inv.model_posterior_measure(
            data, solver, preconditioner=preconditioner
        )
