import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import pygeoinf.linalg as la
from pygeoinf.forward_problem import ForwardProblem
from pygeoinf.least_squares import LeastSquaresInversion


class OccamInversion(LeastSquaresInversion):
    """
    Class for performing Occam inversions of Constable, Parker and Constable (1987).
    """

    def __init__(self, forward_operator, data_error_measure, confidence_level):
        super().__init__(forward_operator, data_error_measure)
        self._critical_chi_squared = chi2.isf(
            1-confidence_level, self.data_space.dim)

    def minimum_norm_operator(self, confidence_level, /, *, solver=None,  preconditioner=None):

        # Compute the critical chi-squared value.
        critical_chi_squared = chi2.isf(
            1-confidence_level, self.data_space.dim)

        def mapping(data):

            # Check if the zero-mapping is the solution.
            initial_model = self.model_space.zero
            chi_squared = self.chi_squared(initial_model, data)
            if chi_squared <= critical_chi_squared:
                return initial_model

    def _chi_squared_function(self, damping, data, solver, preconditioner, initial_model):
        B = self.least_squares_operator(
            damping, solver=solver, preconditioner=preconditioner, initial_model=initial_model)
        model = B(data)
        return self.chi_squared(model, data)
