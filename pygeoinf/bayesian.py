"""
Module for the Bayesian approaches to Bayesian inverse problems.
"""

from pygeoinf.linear_solvers import IterativeLinearSolver
from pygeoinf.gaussian_measure import GaussianMeasure


class LinearBayesianInversion:
    """
    Class for solving a linear inverse problem Bayesian methods assuming Gaussian priors and errors.
    """

    def __init__(self, forward_problem, model_prior_measure):
        """
        Args:
            forward_problem (LinearForwardProblem): The forward problem.
            model_prior_measure (GaussianMeasure): The prior measure on the data.
        """
        self._forward_problem = forward_problem
        self._model_prior_measure = model_prior_measure

    @property
    def forward_problem(self):
        """
        Return the forward problem.
        """
        return self._forward_problem

    @property
    def model_prior_measure(self):
        """Return the model prior measure."""
        return self._model_prior_measure

    @property
    def normal_operator(self):
        """
        Returns the data-space normal operator.
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

    def data_prior_measure(self):
        """
        Return the prior distribution on the data
        """
        if self.forward_problem.data_error_measure_set:
            return (
                self.model_prior_measure.affine_mapping(
                    operator=self.forward_problem.forward_operator
                )
                + self.forward_problem.data_error_measure
            )
        else:
            return self.model_prior_measure.affine_mapping(
                operator=self.forward_problem.forward_operator
            )

    def model_posterior_measure(self, data, solver, /, *, preconditioner=None):
        """
        Returns the posterior measure on the model space given the data.

        Args:
            data (data-space vector): The observed data.
            solver (LinearSolver): A linear solver for the normal equations.
            preconditioner (LinearSolver): A preconditioner for use in solving
                the normal equations.

        Returns:
            GaussianMeasure: The posterior measure.

        Notes:
            The posterior measure does not have a sampling method set. If required,
            this should be set directly afterwards.
        """

        forward_operator = self.forward_problem.forward_operator
        prior_model_covariance = self.model_prior_measure.covariance
        normal_operator = self.normal_operator

        if isinstance(solver, IterativeLinearSolver):
            inverse_normal_operator = solver(
                normal_operator, preconditioner=preconditioner
            )
        else:
            inverse_normal_operator = solver(normal_operator)

        expectation = (
            prior_model_covariance @ forward_operator.adjoint @ inverse_normal_operator
        )(data - forward_operator(self.model_prior_measure.expectation))
        covariance = (
            prior_model_covariance
            - prior_model_covariance
            @ forward_operator.adjoint
            @ inverse_normal_operator
            @ forward_operator
            @ prior_model_covariance
        )

        return GaussianMeasure(
            self.forward_problem.model_space, covariance, expectation=expectation
        )


class LinearBayesianInference(LinearBayesianInversion):
    """Class for solving Bayesian inference problems."""

    def __init__(self, forward_problem, model_prior_measure, property_operator):
        """
        Args:
            forward_problem (LinearForwardProblem): The forward problem.
            model_prior_measure (GaussianMeasure): The prior measure on the data.
            property_operator (LinearOperator): The property operator.

        """
        super().__init__(forward_problem, model_prior_measure)
        assert property_operator.domain == self.forward_problem.model_space
        self._property_operator = property_operator

    @property
    def property_space(self):
        """Return the property space."""
        return self._property_operator.codomain

    @property
    def property_operator(self):
        """Return the property operator."""
        return self._property_operator

    def property_prior_measure(self):
        """Return the prior measure on the property space."""
        return self.model_prior_measure.affine_mapping(operator=self.property_operator)

    def property_posterior_measure(self, data, solver, /, *, preconditioner=None):
        """
        Returns the posterior measure on the property space given the data.

        Args:
            data (data-space vector): The observed data.
            solver (LinearSolver): A linear solver for the normal equations.
            preconditioner (LinearSolver): A preconditioner for use in solving
                the normal equations.

        Returns:
            GaussianMeasure: The posterior measure.

        Notes:
            The posterior measure does not have a sampling method set.
        """
        pi = self.model_posterior_measure(
            data,
            solver,
            preconditioner=preconditioner,
        )
        return pi.affine_mapping(operator=self.property_operator)
