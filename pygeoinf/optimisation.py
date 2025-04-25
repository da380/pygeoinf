"""
Module for classes related to the solution of inverse problems via optimisation methods. 
"""

import numpy as np
from scipy.stats import chi2
import pygeoinf.hilbert as hs
from pygeoinf.forward_problem import ForwardProblem
from pygeoinf.utils import plot_colourline


class LeastSquaresInversion(ForwardProblem):
    """
    Class for the solution of regularised least-squares
    problems within a Hilbert space.
    """

    def __init__(self, forward_operator, data_error_measure):
        """
        Args:
            forward_operator (LinearOperator): The forward operator for the problem.
            data_error_measure (GaussianMeasure): The error measure on the data.
        """
        super().__init__(forward_operator, data_error_measure)

    @staticmethod
    def from_forward_problem(forward_problem):
        """
        Construct a least-squares problem from a forward problem.
        """
        return LeastSquaresInversion(
            forward_problem.forward_operator, forward_problem.data_error_measure
        )

    def normal_operator(self, damping):
        """
        Returns the least-squares normal operator.

        Args:
            damping (float): The norm damping parameter. Must be non-negative.

        Returns:
            LinearOperator: The normal operator.

        Raises:
            ValueError: If damping is not non-negative.
        """
        if damping < 0:
            raise ValueError("Damping parameter must be non-negative.")
        forward_operator = self.forward_operator
        inverse_data_covariance = self.data_error_measure.inverse_covariance
        identity = self.model_space.identity
        return (
            forward_operator.adjoint @ inverse_data_covariance @ forward_operator
            + damping * identity
        )

    def least_squares_operator(self, damping, /, *, solver=None, preconditioner=None):
        """
        Returns a linear operator that solves the least-squares problem.

        Args:
            damping (float): The norm damping parameter. Must be non-negative.
            solver (LinearSolver): Linear solver for the normal equations. If none
                is provided, matrix-free CG is used.
            preconditioner (LinearOperator): Preconditioner for use in
                solving the normal equations. The default is None.

        Returns:
            LinearOperator: Mapping from data space to least-squares solution.

        Raises:
            ValueError: If damping is not non-negative.
            ValueError: If solver is not a instance of LinearSolver.
        """

        forward_operator = self.forward_operator
        inverse_data_covariance = self.data_error_measure.inverse_covariance
        normal_operator = self.normal_operator(damping)

        if solver is None:
            _solver = hs.CGSolver()
        else:
            _solver = solver

        if isinstance(_solver, hs.IterativeLinearSolver):
            inverse_normal_operator = _solver(
                normal_operator, preconditioner=preconditioner
            )
        else:
            inverse_normal_operator = _solver(normal_operator)

        return (
            inverse_normal_operator @ forward_operator.adjoint @ inverse_data_covariance
        )

    def model_measure(
        self,
        damping,
        data,
        /,
        *,
        solver=None,
        preconditioner=None,
    ):
        """
        Returns the measure on the model space induced by the observed data under the least-squares solution.

        Args:
            damping (float): The norm damping parameter. Must be non-negative.
            data (data vector): Observed data
            solver (LinearSolver): Linear solver for solvint the normal equations.
                The default is conjugate-gradients.
            preconditioner (LinearOperator): Preconditioner for use in
                solving the normal equations. The default is the identity.

        Returns:
            GaussianMeasure: Measure on the model space induced by the
                least-squares solution for given data. Note that this measure only
                accounts for uncertainty due to the propagation of
                uncertainties within the data.

        Raises:
            ValueError: If damping is not non-negative.
            ValueError: If solver is not a instance of LinearSolver.
        """
        least_squares_operator = self.least_squares_operator(
            damping,
            solver=solver,
            preconditioner=preconditioner,
        )
        model = least_squares_operator(data)
        return self.data_error_measure.affine_mapping(
            operator=least_squares_operator, translation=model
        )

    def resolution_operator(self, damping, /, *, solver=None, preconditioner=None):
        """
        Returns the resolution operator for the least-squares problem.

        Args:
            damping (float): The norm damping parameter. Must be non-negative.
            solver (LinearSolver): Linear solver for solvint the normal equations.
                The default is conjugate-gradients.
            preconditioner (LinearOperator): Preconditioner for use in
                solving the normal equations. The default is the identity.

        Returns:
            LinearOperator: The resolution operator for the problem,
                with this operator mapping a known model into the
                result of its inversion in the absence of data errors.

        Raises:
            ValueError: If damping is not non-negative.
            ValueError: If solver is not a instance of LinearSolver.
        """
        forward_operator = self.forward_operator
        least_squares_operator = self.least_squares_operator(
            damping,
            solver=solver,
            preconditioner=preconditioner,
        )
        return least_squares_operator @ forward_operator
