"""
Implements the Bayesian framework for solving linear inverse problems.

This module treats the inverse problem from a statistical perspective, aiming to
determine the full posterior probability distribution of the unknown model
parameters, rather than a single best-fit solution.

It assumes that the prior knowledge about the model and the statistics of the
data errors can be described by Gaussian measures. For a linear forward problem,
the resulting posterior distribution for the model is also Gaussian, allowing
for an analytical solution.

Key Classes
-----------
- `LinearBayesianInversion`: Computes the posterior Gaussian measure `p(u|d)`
  for the model `u` given observed data `d`. This provides not only a mean
  estimate for the model but also its uncertainty (covariance).
- `LinearBayesianInference`: Extends the framework to compute the posterior
  distribution for a derived property of the model, `p(B(u)|d)`, where `B` is
  some linear operator.
"""

from __future__ import annotations
from typing import Optional

from .inversion import Inversion
from .gaussian_measure import GaussianMeasure


from .forward_problem import LinearForwardProblem
from .operators import LinearOperator
from .linear_solvers import LinearSolver, IterativeLinearSolver
from .hilbert_space import HilbertSpace, Vector


class LinearBayesianInversion(Inversion):
    """
    Solves a linear inverse problem using Bayesian methods.

    This class applies to problems of the form `d = A(u) + e`, where the prior
    knowledge of the model `u` and the statistics of the error `e` are described
    by Gaussian distributions. It computes the full posterior probability
    distribution `p(u|d)` for the model parameters given an observation `d`.
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
        Returns the covariance of the prior predictive distribution, `p(d)`.

        This operator, `C_d = A @ C_u @ A* + C_e`, represents the total
        expected covariance in the data space before any data is observed.
        Its inverse is central to calculating the posterior distribution and is
        often referred to as the Bayesian normal operator.
        """
        forward_operator = self.forward_problem.forward_operator
        prior_model_covariance = self.model_prior_measure.covariance

        if self.forward_problem.data_error_measure_set:
            return (
                forward_operator @ prior_model_covariance @ forward_operator.adjoint
                + self.forward_problem.data_error_measure.covariance
            )
        else:
            return forward_operator @ prior_model_covariance @ forward_operator.adjoint

    def data_prior_measure(self) -> GaussianMeasure:
        """
        Returns the prior predictive distribution on the data, `p(d)`.

        This measure describes the expected distribution of the data before any
        specific observation is made, combining the uncertainty from the prior
        model and the data errors.
        """
        if self.forward_problem.data_error_measure_set:
            # d = A(u) + e  =>  p(d) is convolution of p(A(u)) and p(e)
            return (
                self.model_prior_measure.affine_mapping(
                    operator=self.forward_problem.forward_operator
                )
                + self.forward_problem.data_error_measure
            )
        else:
            # d = A(u)  => p(d) is just the mapping of the model prior
            return self.model_prior_measure.affine_mapping(
                operator=self.forward_problem.forward_operator
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
        Returns the posterior Gaussian measure for the model, `p(u|d)`.

        This measure represents our updated state of knowledge about the model
        `u` after observing the data `d`. Its expectation is the most likely
        model, and its covariance quantifies the remaining uncertainty.

        Args:
            data: The observed data vector.
            solver: A linear solver for inverting the normal operator.
            preconditioner: An optional preconditioner for iterative solvers.

        Returns:
            The posterior `GaussianMeasure` on the model space.
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

        # Calculate posterior mean: mu_post = mu_u + C_u*A^T*C_d^-1*(d - A*mu_u - mu_e)
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

        # Calculate posterior covariance: C_post = C_u - C_u*A^T*C_d^-1*A*C_u
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
    Performs Bayesian inference on a derived property of the model.

    While `LinearBayesianInversion` solves for the model `u` itself, this class
    computes the posterior distribution for a property `p = B(u)`, where `B` is a
    linear operator acting on the model `u`. This is useful for uncertainty
    quantification of derived quantities (e.g., the average value of a field).
    """

    def __init__(
        self,
        forward_problem: LinearForwardProblem,
        model_prior_measure: GaussianMeasure,
        property_operator: LinearOperator,
        /,
    ) -> None:
        """
        Args:
            forward_problem: The forward problem linking the model to the data.
            model_prior_measure: The prior Gaussian measure on the model space.
            property_operator: The linear operator `B` that maps a model `u` to
                a property `p`.
        """
        super().__init__(forward_problem, model_prior_measure)
        if property_operator.domain != self.forward_problem.model_space:
            raise ValueError("Property operator domain must match the model space.")
        self._property_operator: LinearOperator = property_operator

    @property
    def property_space(self) -> HilbertSpace:
        """The Hilbert space in which the property `p` resides."""
        return self._property_operator.codomain

    @property
    def property_operator(self) -> LinearOperator:
        """The linear operator `B` that defines the property."""
        return self._property_operator

    def property_prior_measure(self) -> GaussianMeasure:
        """
        Returns the prior measure on the property space, `p(p)`.

        This is computed by propagating the model prior through the property
        operator.
        """
        return self.model_prior_measure.affine_mapping(operator=self.property_operator)

    def property_posterior_measure(
        self,
        data: Vector,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[LinearOperator] = None,
    ) -> GaussianMeasure:
        """
        Returns the posterior measure on the property space, `p(p|d)`.

        This is computed by first finding the posterior measure for the model,
        `p(u|d)`, and then propagating it through the property operator `B`.

        Args:
            data: The observed data vector.
            solver: A linear solver for the normal equations.
            preconditioner: An optional preconditioner for iterative solvers.

        Returns:
            The posterior `GaussianMeasure` on the property space.
        """
        # First, find the posterior distribution for the model u.
        model_posterior = self.model_posterior_measure(
            data, solver, preconditioner=preconditioner
        )
        # Then, map that distribution to the property space.
        return model_posterior.affine_mapping(operator=self.property_operator)
