import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import brentq
import pygeoinf.linalg as la
from pygeoinf.forward_problem import ForwardProblem
from pygeoinf.least_squares import LeastSquaresInversion


class OccamInversion(ForwardProblem):
    """
    Class for performing the Occam inversions of Constable, Parker and Constable (1987)
    for a linear problem with Gaussian errors. 
    """

    def __init__(self, forward_operator, data_error_measure, /, *, rtol=1.e-5, damping_min=1.e-5):
        super().__init__(forward_operator, data_error_measure)
        self._least_squares_inversion = LeastSquaresInversion(
            forward_operator, data_error_measure)
        self._rtol = rtol
        self._damping_min = damping_min

    @property
    def least_squares_inversion(self):
        return self._least_squares_inversion

    def minimum_norm_operator(self, confidence_level, /, *, solver=None, preconditioner=None):
        """
        Returns the operator that maps data to the minimum norm solution for the
        given confidence level.

        Args:
            confidence_level (float): The confidence level used to define the data-space confidence set.
            solver (LinearSolver): Linear solver for the least-squares problems.
            preconditioner (LinearOperator): Preconditioner for the least-squares problems. Note that
                currently the same preconditioner is applied to all linear systems.

        Returns:
            Operator: A non-linear operator mapping the data to the minimum norm solution.
        """

        # Compute the critical value for chi-squared.
        critical_chi_squared = chi2.isf(
            1-confidence_level, self.data_space.dim)

        def mapping(data):

            # check to see if zero-model is the solution.
            model = self.model_space.zero
            chi_squared = self.chi_squared(model, data)
            if chi_squared <= critical_chi_squared:
                return model

            # Local function for solving the least-squares problem
            def compute_model_chi_squared(damping, initial_model=None):
                print("Solving least squares problem")
                B = self.least_squares_inversion.least_squares_operator(
                    damping, solver=solver, preconditioner=preconditioner, initial_model=initial_model)
                model = B(data)
                return model, self.chi_squared(model, data)

            # Bound the damping.
            damping = 1
            model, chi_squared = compute_model_chi_squared(damping)
            if chi_squared >= critical_chi_squared:
                damping_low = None
                damping_high = damping
            else:
                damping_low = damping
                damping_high = None

            if damping_low is None:
                damping_low = damping_high
                while (chi_squared > critical_chi_squared):
                    damping_low /= 2
                    model, chi_squared = compute_model_chi_squared(
                        damping_low, model)
                    if damping_low < self._damping_min:
                        break

            if damping_high is None:
                damping_high = damping_low
                while (chi_squared <= critical_chi_squared):
                    damping_high *= 2
                    model, chi_squared = compute_model_chi_squared(
                        damping_high, model)

            # Use bisection to determine the damping.
            damping = 0.5*(damping_high + damping_low)
            while (np.abs(damping-damping_low) > self._rtol * damping):
                model, chi_squared = compute_model_chi_squared(damping, model)
                if chi_squared >= critical_chi_squared:
                    damping_high = damping
                else:
                    damping_low = damping
                damping = 0.5*(damping_high + damping_low)

            return model

        return la.Operator(self.data_space, self.model_space, mapping)

    def resolution_operator(self, confidence_level, /, *, solver=None,  preconditioner=None):
        """
        Returns the resolution operator for the least-squares problem. 

        Args:
            confidence_level (float): The confidence level used to define the data-space confidence set.
            solver (LinearSolver): Linear solver for solvint the normal equations.
                The default is conjugate-gradients.
            preconditioner (LinearOperator): Preconditioner for use in 
                solving the normal equations. The default is the identity.            

        Returns:
            Operator: The resolution operator for the problem, 
                with this operator mapping a known model into the 
                result of its inversion in the absence of data errors. 

        Raises:            
            ValueError: If solver is not a instance of LinearSolver.         
        """
        A = self.forward_operator
        B = self.minimum_norm_operator(
            confidence_level, solver=solver, preconditioner=preconditioner)
        return la.Operator(self.model_space, self.model_space, lambda x: B(A(x)))
