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
from .hilbert_space import HilbertSpace, Vector
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
        Returns the covariance of the prior predictive distribution, `p(d)`.
        C_d = A @ C_u @ A* + C_e
        """
        forward_operator = self.forward_problem.forward_operator
        prior_model_covariance = self.model_prior_measure.covariance

        if self.forward_problem.data_error_measure_set:
            return (
                forward_operator @ prior_model_covariance @ forward_operator.adjoint
                + self.forward_problem.data_error_measure.covariance
            )
        else:
            return NormalSumOperator(forward_operator, prior_model_covariance)

    def model_posterior_measure(
        self,
        data: Vector,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[LinearOperator] = None,
    ) -> GaussianMeasure:
        """
        Returns the posterior Gaussian measure for the model, `p(u|d)`.

        Args:
            data: The observed data vector.
            solver: A linear solver for inverting the normal operator C_d.
            preconditioner: An optional preconditioner for C_d.
        """
        data_space = self.data_space
        model_space = self.model_space
        forward_operator = self.forward_problem.forward_operator
        prior_model_covariance = self.model_prior_measure.covariance
        normal_operator = self.normal_operator

        if isinstance(solver, IterativeLinearSolver):
            inverse_normal_operator = solver(
                normal_operator, preconditioner=preconditioner
            )
        else:
            inverse_normal_operator = solver(normal_operator)

        # Posterior Mean: mu_post = mu_prior + C_u A* C_d^-1 (d - A mu_prior)
        shifted_data = data_space.subtract(
            data, forward_operator(self.model_prior_measure.expectation)
        )
        if self.forward_problem.data_error_measure_set:
            shifted_data = data_space.subtract(
                shifted_data, self.forward_problem.data_error_measure.expectation
            )

        mean_update = (
            prior_model_covariance @ forward_operator.adjoint @ inverse_normal_operator
        )(shifted_data)
        expectation = model_space.add(self.model_prior_measure.expectation, mean_update)

        # Posterior Covariance: C_post = C_u - C_u A* C_d^-1 A C_u
        covariance = prior_model_covariance - (
            prior_model_covariance
            @ forward_operator.adjoint
            @ inverse_normal_operator
            @ forward_operator
            @ prior_model_covariance
        )

        return GaussianMeasure(covariance=covariance, expectation=expectation)


class LinearBayesianInference(LinearBayesianInversion):
    """
    Performs Bayesian inference on a derived property `p = B(u)`.
    """

    def __init__(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        property_operator: LinearOperator,
        /,
    ) -> None:
        super().__init__(forward_problem, model_prior_measure)
        if property_operator.domain != self.forward_problem.model_space:
            raise ValueError("Property operator domain must match the model space.")
        self._property_operator: LinearOperator = property_operator

    @property
    def property_space(self) -> HilbertSpace:
        return self._property_operator.codomain

    @property
    def property_operator(self) -> LinearOperator:
        return self._property_operator

    def property_posterior_measure(
        self,
        data: Vector,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[LinearOperator] = None,
    ) -> GaussianMeasure:
        """Returns the posterior measure on the property space, `p(p|d)`."""
        model_posterior = self.model_posterior_measure(
            data, solver, preconditioner=preconditioner
        )
        return model_posterior.affine_mapping(operator=self.property_operator)


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
    ) -> None:
        """
        Args:
            forward_problem: The forward problem.
            model_prior_measure: The unconstrained prior Gaussian measure.
            constraint: The affine subspace A = {u | Bu = w}.
        """
        super().__init__(forward_problem)
        self._unconstrained_prior = model_prior_measure
        self._constraint = constraint

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
            solver: Linear solver used to invert the prior covariance in the
                constraint space (B @ C_prior @ B*).
            preconditioner: Optional preconditioner for the constraint solver.
        """
        B = self._constraint.constraint_operator
        w = self._constraint.constraint_value
        prior_mean = self._unconstrained_prior.expectation
        prior_cov = self._unconstrained_prior.covariance

        # 1. Form the "Normal Operator" for the constraint update
        # S = B @ C @ B* (Acts on Property Space)
        S = B @ prior_cov @ B.adjoint

        # 2. Invert S
        if isinstance(solver, IterativeLinearSolver):
            S_inv = solver(S, preconditioner=preconditioner)
        else:
            S_inv = solver(S)

        # 3. Define Gain Operator K = C @ B* @ S^-1
        K_op = prior_cov @ B.adjoint @ S_inv

        # 4. Update Mean: m_new = m + K (w - B m)
        innovation = B.codomain.subtract(w, B(prior_mean))
        mean_update = K_op(innovation)
        new_mean = self.model_space.add(prior_mean, mean_update)

        # 5. Update Covariance: C_new = C - K B C
        correction_op = K_op @ B @ prior_cov
        new_cov = prior_cov - correction_op

        # 6. Define Sampling
        if self._unconstrained_prior.sample_set:

            def sample() -> Vector:
                u_prior = self._unconstrained_prior.sample()
                mismatch = B.codomain.subtract(B(u_prior), w)
                correction = K_op(mismatch)
                return self.model_space.subtract(u_prior, correction)

        else:
            sample = None

        return GaussianMeasure(covariance=new_cov, expectation=new_mean, sample=sample)

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
        Returns the posterior Gaussian measure p(u | d, u in A).

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
