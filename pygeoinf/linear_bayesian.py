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
            return NormalSumOperator(forward_operator, model_prior_covariance)

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
        shifted_data = data_space.subtract(
            data, forward_operator(self.model_prior_measure.expectation)
        )

        # Shift for noise mean: d - A(mu_u) - mu_e
        if self.forward_problem.data_error_measure_set:
            error_expectation = self.forward_problem.data_error_measure.expectation
            shifted_data = data_space.subtract(shifted_data, error_expectation)
        else:
            error_expectation = data_space.zero

        mean_update = kalman_gain(shifted_data)
        expectation = model_space.add(self.model_prior_measure.expectation, mean_update)

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
                # a. Sample Prior
                model_sample = self.model_prior_measure.sample()

                # b. Calculate Residual
                prediction = forward_operator(model_sample)
                data_residual = data_space.subtract(data, prediction)

                # c. Perturb Residual
                if self.forward_problem.data_error_measure_set:
                    noise_raw = self.forward_problem.data_error_measure.sample()
                    epsilon = data_space.subtract(noise_raw, error_expectation)
                    data_space.axpy(1.0, epsilon, data_residual)

                # d. Update
                correction = kalman_gain(data_residual)
                return model_space.add(model_sample, correction)

            return GaussianMeasure(
                covariance=covariance, expectation=expectation, sample=sample
            )
        else:
            return GaussianMeasure(covariance=covariance, expectation=expectation)


class ConstrainedLinearBayesianInversion(LinearInversion):
    """
    Solves a linear inverse problem subject to an affine subspace constraint.

    This class enforces the constraint `u in A` using either:
    1. Bayesian Conditioning (Default): p(u | d, u in A).
       If A is defined geometrically (no explicit equation), an implicit
       operator (I-P) is used, which requires a robust solver in the subspace.
    2. Geometric Projection: Projects the unconstrained posterior onto A.
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
            constraint: The affine subspace A.
            geometric: If True, uses orthogonal projection (Euclidean metric).
                       If False (default), uses Bayesian conditioning.
        """
        super().__init__(forward_problem)
        self._unconstrained_prior = model_prior_measure
        self._constraint = constraint
        self._geometric = geometric

    def conditioned_prior_measure(self) -> GaussianMeasure:
        """
        Computes the prior measure conditioned on the constraint.
        """
        return self._constraint.condition_gaussian_measure(
            self._unconstrained_prior, geometric=self._geometric
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
        Returns the posterior Gaussian measure p(u | d, u in A).

        Args:
            data: Observed data vector.
            solver: Solver for the data update (inverts A C_cond A* + Ce).
            preconditioner: Preconditioner for the data update.

        Note: The solver for the constraint update is managed internally by
        the AffineSubspace object passed at initialization.
        """
        # 1. Condition Prior
        cond_prior = self.conditioned_prior_measure()

        # 2. Solve Bayesian Inverse Problem with the new prior
        bayes_inv = LinearBayesianInversion(self.forward_problem, cond_prior)

        return bayes_inv.model_posterior_measure(
            data, solver, preconditioner=preconditioner
        )
