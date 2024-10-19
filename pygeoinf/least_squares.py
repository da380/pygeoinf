import numpy as np
import pygeoinf.linalg as la
from pygeoinf.forward_problem import ForwardProblem


class LeastSquaresProblem(ForwardProblem):
    """
    Class for the solution of regularised least-squares
    problems within a Hilbert space.
    """

    def __init__(self, forward_operator, data_error_measure):
        """, /, *, solver=None
        Args:
            forward_operator (LinearOperator): The forward operator for the problem.
            data_error_measure (GaussianMeasure): The error measure on the data.
        """
        super().__init__(forward_operator, data_error_measure)
        self._solver = la.CGSolver()

    @staticmethod
    def from_forward_problem(forward_problem):
        """
        Construct a least-squares problem from a forward problem. 
        """
        return LeastSquaresProblem(forward_problem.forward_operator,
                                   forward_problem.data_error_measure)

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
        A = self.forward_operator
        Ci = self.data_error_measure.inverse_covariance
        I = self.model_space.identity()
        return A.adjoint @ Ci @ A + damping * I

    def least_squares_operator(self, damping, /, *, solver=None,  preconditioner=None, x0=None):
        """
        Returns a linear operator that solves the least-squares problem.

        Args:
            damping (float): The norm damping parameter. Must be non-negative. 
            solver (LinearSolver): Linear solver for solvint the normal equations.
                The default is conjugate-gradients.
            preconditioner (LinearOperator): Preconditioner for use in 
                solving the normal equations. The default is the identity.
            x0 (model-space vector): Initial guess within an iterative solution. 
                The default is zero.

        Returns:
            LinearOperator: Mapping from data space to least-squares solution. 

        Raises:
            ValueError: If damping is not non-negative. 
            ValueError: If solver is not a instance of LinearSolver. 
        """
        if solver is None:
            _solver = la.CGSolver()
        else:
            _solver = solver

        if x0 is None:
            _x0 = self.model_space.zero
        else:
            _x0 = x0

        A = self.forward_operator
        Ci = self.data_error_measure.inverse_covariance
        N = self.normal_operator(damping)

        if isinstance(_solver, la.IterativeLinearSolver):
            Ni = _solver(N, preconditioner=preconditioner, x0=_x0)
        elif isinstance(_solver, la.DirectLinearSolver):
            Ni = _solver(N)
        else:
            raise ValueError("input solver is of the wrong type.")

        return Ni @ A.adjoint @ Ci

    def model_measure(self, damping, /, *, solver=None,  preconditioner=None, x0=None):
        """
        Returns the measure on the model space induced by the least-squares solution. 

        Args:
            damping (float): The norm damping parameter. Must be non-negative. 
            solver (LinearSolver): Linear solver for solvint the normal equations.
                The default is conjugate-gradients.
            preconditioner (LinearOperator): Preconditioner for use in 
                solving the normal equations. The default is the identity.
            x0 (model-space vector): Initial guess within an iterative solution. 
                The default is zero.

        Returns:
            GaussianMeasure: Measure on the model space induced by the 
                least-squares solution. Note that this measure only 
                accounts for uncertainty due to the propagation of 
                uncertainties within the data. 

        Raises:
            ValueError: If damping is not non-negative. 
            ValueError: If solver is not a instance of LinearSolver.         
        """
        B = self.least_squares_operator(damping,
                                        solver=solver,
                                        preconditioner=preconditioner,
                                        x0=x0)
        return self.data_error_measure.affine_mapping(operator=B)

    def resolution_operator(self, damping, /, *, solver=None,  preconditioner=None, x0=None):
        """
        Returns the resolution operator for the least-squares problem. 

        Args:
            damping (float): The norm damping parameter. Must be non-negative. 
            solver (LinearSolver): Linear solver for solvint the normal equations.
                The default is conjugate-gradients.
            preconditioner (LinearOperator): Preconditioner for use in 
                solving the normal equations. The default is the identity.
            x0 (model-space vector): Initial guess within an iterative solution. 
                The default is zero.

        Returns:
            LinearOperator: The resolution operator for the problem, 
                with this operator mapping a known model into the 
                result of its inversion in the absence of data errors. 

        Raises:
            ValueError: If damping is not non-negative. 
            ValueError: If solver is not a instance of LinearSolver.         
        """
        A = self.forward_operator
        B = self.least_squares_operator(damping,
                                        solver=solver,
                                        preconditioner=preconditioner,
                                        x0=x0)

        return B @ A
