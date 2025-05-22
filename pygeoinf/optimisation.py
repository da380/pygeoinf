"""
Module for classes related to the solution of inverse problems via optimisation methods. 
"""

from pygeoinf.linear_solvers import IterativeLinearSolver


class LinearLeastSquaresInversion:
    """
    Class for the solution of regularised least-squares
    problems within a Hilbert space.
    """

    def __init__(self, forward_problem):

        if not forward_problem.data_error_measure.inverse_covariance_set:
            raise ValueError(
                "data error measure does not have its inverse covariance set."
            )

        self._forward_problem = forward_problem

    @property
    def forward_problem(self):
        """
        Returns the forward problem.
        """
        return self._forward_problem

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
        forward_operator = self.forward_problem.forward_operator
        inverse_data_covariance = (
            self.forward_problem.data_error_measure.inverse_covariance
        )
        identity = self.forward_problem.model_space.identity_operator()
        return (
            forward_operator.adjoint @ inverse_data_covariance @ forward_operator
            + damping * identity
        )

    def least_squares_operator(self, damping, solver, /, *, preconditioner=None):
        """
        Returns a linear operator that solves the least-squares problem.

        Args:
            damping (float): The norm damping parameter. Must be non-negative.
            solver (LinearSolver): Linear solver for the normal equations.
            preconditioner (LinearOperator): Preconditioner for use in
                solving the normal equations. The default is None.

        Returns:
            LinearOperator: Mapping from data space to least-squares solution.

        Raises:
            ValueError: If damping is not non-negative.
            ValueError: If solver is not a instance of LinearSolver.
        """

        forward_operator = self.forward_problem.forward_operator
        inverse_data_covariance = (
            self.forward_problem.data_error_measure.inverse_covariance
        )
        normal_operator = self.normal_operator(damping)

        if isinstance(solver, IterativeLinearSolver):
            inverse_normal_operator = solver(
                normal_operator, preconditioner=preconditioner
            )
        else:
            inverse_normal_operator = solver(normal_operator)

        return (
            inverse_normal_operator @ forward_operator.adjoint @ inverse_data_covariance
        )

    def model_measure(
        self,
        damping,
        data,
        solver,
        /,
        *,
        preconditioner=None,
    ):
        """
        Returns the measure on the model space induced by the observed data under the least-squares solution.

        Args:
            damping (float): The norm damping parameter. Must be non-negative.
            data (data vector): Observed data
            solver (LinearSolver): Linear solver for solvint the normal equations.
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
            solver,
            preconditioner=preconditioner,
        )
        model = least_squares_operator(data)
        return self.forward_problem.data_error_measure.affine_mapping(
            operator=least_squares_operator, translation=model
        )

    def resolution_operator(self, damping, solver, /, *, preconditioner=None):
        """
        Returns the resolution operator for the least-squares problem.

        Args:
            damping (float): The norm damping parameter. Must be non-negative.
            solver (LinearSolver): Linear solver for solvint the normal equations.
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
        forward_operator = self.forward_problem.forward_operator
        least_squares_operator = self.least_squares_operator(
            damping,
            solver,
            preconditioner=preconditioner,
        )
        return least_squares_operator @ forward_operator
