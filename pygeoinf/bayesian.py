"""
Module for the Bayesian approaches to Bayesian inverse problems.
"""

import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from pygeoinf.hilbert_space import LinearOperator, EuclideanSpace
from pygeoinf.linear_solvers import IterativeLinearSolver
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.inversion import Inversion


class LinearBayesianInversion(Inversion):
    """
    Class for solving a linear inverse problem Bayesian methods assuming Gaussian priors and errors.
    """

    def __init__(self, forward_problem, model_prior_measure):
        """
        Args:
            forward_problem (LinearForwardProblem): The forward problem.
            model_prior_measure (GaussianMeasure): The prior measure on the data.
        """
        super().__init__(forward_problem)
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

    def inverse_cholesky_factored_normal_operator(self):
        """
        Returns a Cholesky factorisation of the inverse normal operator. Calculations
        based on the dense matrix representation, and hence this is an expensive
        method if the data space is large.
        """

        normal_matrix = self.normal_operator.matrix(dense=True, galerkin=True)
        factor, lower = cho_factor(normal_matrix)
        identity_operator = np.identity(self.data_space_dim)
        inverse_factor = solve_triangular(
            factor, identity_operator, overwrite_b=True, lower=lower
        )
        domain = self.data_space
        codomain = EuclideanSpace(self.data_space_dim)
        return LinearOperator.from_matrix(domain, codomain, inverse_factor)

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

        return GaussianMeasure(covariance=covariance, expectation=expectation)

    def model_posterior_measure_using_random_factorisation(
        self, data, rank, /, *, power=0, method="fixed", rtol=1e-3
    ):
        """
        Returns the posterior model measure based on a low rank factorisation
        of the posteriori covariance. This method assumes that the prior model
        covariance is in a factored form.

        To use this method, the prior covariance must be provided in a factored form.
        """

        if not self.forward_problem.data_error_measure_set:
            raise NotImplementedError(
                "This method requires a prior data measure to be set."
            )

        if not self.model_prior_measure.covariance_factor_set:
            raise NotImplementedError(
                "This method requires the model prior covariance to be in factored form"
            )

        # Set up the necessary operators.
        forward_operator = self.forward_problem.forward_operator
        prior_covariance_factor = self.model_prior_measure.covariance_factor
        identity = prior_covariance_factor.domain.identity_operator()
        inverse_data_covariance = (
            self.forward_problem.data_error_measure.inverse_covariance
        )

        normal_operator = (
            prior_covariance_factor.adjoint
            @ forward_operator.adjoint
            @ inverse_data_covariance
            @ forward_operator
            @ prior_covariance_factor
            + identity
        )

        # Perform the random eigen-decomposition.
        eigenvectors, eigenvalues = normal_operator.random_eig(
            rank, power=power, method=method, rtol=rtol
        )
        inclusion = self.model_space.coordinate_inclusion
        inverse_eigenvectors = inclusion @ inclusion.adjoint @ eigenvectors
        inverse_normal_operator = (
            inverse_eigenvectors @ eigenvalues.inverse @ inverse_eigenvectors.adjoint
        )

        # Form the posterior covariance and its factor.
        posterior_covariance = (
            prior_covariance_factor
            @ inverse_normal_operator
            @ prior_covariance_factor.adjoint
        )

        posterior_covariance_factor = (
            prior_covariance_factor @ inverse_eigenvectors @ eigenvalues.inverse.sqrt
        )

        # Now the posterior expectation.
        prior_expectation = self.model_prior_measure.expectation
        expectation = self.model_space.add(
            prior_expectation,
            (posterior_covariance @ forward_operator.adjoint @ inverse_data_covariance)(
                self.data_space.subtract(data, forward_operator(prior_expectation))
            ),
        )

        # Return the measure.
        return GaussianMeasure(
            covariance_factor=posterior_covariance_factor, expectation=expectation
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
