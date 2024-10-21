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

    def data_posterior_measure(self):
        """
        Return the prior distribution on the data
        """
        return self.model_prior_measure.affine_mapping(operator=self.forward_operator) + self.data_error_measure

    def model_posterior_measure(self, data, /, *,
                                solver=None,
                                preconditioner=None,
                                preconditioning_method=None,
                                factor_covariance=False):
        """
        Returns the posterior measure on the model space given the data.
        """

        A = self.forward_operator
        Q = self.model_prior_measure.covariance
        N = self.normal_operator()

        if solver is None:
            _solver = la.CGSolver()
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

        # Compute the posterior expectation.
        model0 = self.model_prior_measure.expectation
        data0 = A(model0)
        expectation = (Q @ A.adjoint @ Ni)(data - data0)

        if factor_covariance:

            """
            D, U = eigsh(N.matrix(), k=10, which="LM")

            Di = diags([1/D], [0])

            def factor_mapping(c):
                return (Q @ A.adjoint)(U @ Di @ U.T @ c)

            def adjoint_factor_mapping(x):
                return U @ Di @ U.T @ A(Q(x))

            factor = la.LinearOperator(la.EuclideanSpace(
                self.data_space.dim), self.model_space, factor_mapping, adjoint_mapping=adjoint_factor_mapping)



            return la.GaussianMeasure.from_factored_covariance(factor, expectation=expectation)

            """

            raise NotImplementedError()

        else:

            S = Q - Q @ A.adjoint @ Ni
            return la.GaussianMeasure(self.model_space, S, expectation=expectation)
