import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
import pygeoinf.linalg as la
from pygeoinf.forward_problem import ForwardProblem


class BayesianInversion(ForwardProblem):
    """
    Class for solving a linear inverse problem Bayesian methods assuming Gaussian priors and errors.
    """

    def __init__(self, forward_operator, model_prior_measure, data_error_measure):
        """
        Args:
            forward_operator (LinearOperator): The forward operator for the problem.
            model_prior_measure (GaussianMeasure): The prior measure on the data.
            data_error_measure (GaussianMeasure): The error measure on the data.
        """
        super().__init__(forward_operator, data_error_measure)
        self._model_prior_measure = model_prior_measure

    @property
    def model_prior_measure(self):
        """Return the model prior measure."""
        return self._model_prior_measure

    def normal_operator(self):
        """
        Returns the data-space normal operator.
        """
        A = self.forward_operator
        Q = self.model_prior_measure.covariance
        R = self.data_error_measure.covariance
        return A @ Q @ A.adjoint + R

    def data_prior_measure(self):
        """
        Return the prior distribution on the data
        """
        return (
            self.model_prior_measure.affine_mapping(operator=self.forward_operator)
            + self.data_error_measure
        )

    def model_posterior_measure(
        self, data, /, *, solver=None, preconditioner=None, preconditioning_method=None
    ):
        """
        Returns the posterior measure on the model space given the data.

        Args:
            data (data-space vector): The observed data.
            solver (LinearSolver): A linear solver for the normal equations.
            preconditioner (LinearSolver): A preconditioner for use in solving
                the normal equations.
            preconditioning_method (PreconditioningMethod): A preconditioning
                method that constructs a preconditioner from the normal operator.

        Returns:
            GaussianMeasure: The posterior measure.

        Notes:
            The posterior measure does not have a sampling method set. If required,
            this should be set directly afterwards.
        """

        A = self.forward_operator
        Q = self.model_prior_measure.covariance
        N = self.normal_operator()

        if solver is None:
            _solver = la.CGMatrixSolver()
        else:
            _solver = solver

        if preconditioner is None:
            if preconditioning_method is None:
                _preconditioner = la.IdentityPreconditioner()(N)
            else:
                _preconditioner = preconditioning_method(N)
        else:
            _preconditioner = preconditioner

        if isinstance(_solver, la.IterativeLinearSolver):
            Ni = _solver(N, preconditioner=_preconditioner)
        else:
            Ni = _solver(N)

        expectation = (Q @ A.adjoint @ Ni)(
            data - A(self.model_prior_measure.expectation)
        )
        covariance = Q - Q @ A.adjoint @ Ni @ A @ Q

        return la.GaussianMeasure(self.model_space, covariance, expectation=expectation)


class BayesianInference(BayesianInversion):
    """Class for solving Bayesian inference problems."""

    def __init__(
        self,
        forward_operator,
        property_operator,
        model_prior_measure,
        data_error_measure,
    ):
        """
        Args:
            forward_operator (LinearOperator): The forward operator.
            property_operator (LinearOperator): The property operator.
            model_prior_measure (GaussianMeasure): The prior measure on the data.
            data_error_measure (GaussianMeasure): The error measure on the data.
        """
        super().__init__(forward_operator, model_prior_measure, data_error_measure)
        assert property_operator.domain == self.model_space
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

    def property_posterior_measure(
        self, data, /, *, solver=None, preconditioner=None, preconditioning_method=None
    ):
        """
        Returns the posterior measure on the property space given the data.

        Args:
            data (data-space vector): The observed data.
            solver (LinearSolver): A linear solver for the normal equations.
            preconditioner (LinearSolver): A preconditioner for use in solving
                the normal equations.
            preconditioning_method (PreconditioningMethod): A preconditioning
                method that constructs a preconditioner from the normal operator.

        Returns:
            GaussianMeasure: The posterior measure.

        Notes:
            The posterior measure does not have a sampling method set. If required,
            this should be set directly afterwards.
        """
        pi = self.model_posterior_measure(
            data,
            solver=solver,
            preconditioner=preconditioner,
            preconditioning_method=preconditioning_method,
        )
        return pi.affine_mapping(operator=self.property_operator)
