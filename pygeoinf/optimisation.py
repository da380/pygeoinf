"""
Module for classes related to the solution of inverse problems via optimisation methods. 
"""

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import brentq
import pygeoinf.linalg as la
from pygeoinf.forward_problem import ForwardProblem
from pygeoinf.utils import plot_colourline


class LeastSquaresInversion(ForwardProblem):
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
        A = self.forward_operator
        Ci = self.data_error_measure.inverse_covariance
        I = self.model_space.identity()
        return A.adjoint @ Ci @ A + damping * I

    def least_squares_operator(
        self,
        damping,
        /,
        *,
        solver=None,
        preconditioner=None,
        preconditioning_method=None,
        initial_model=None,
    ):
        """
        Returns a linear operator that solves the least-squares problem.

        Args:
            damping (float): The norm damping parameter. Must be non-negative.
            solver (LinearSolver): Linear solver for solvint the normal equations.
                The default is conjugate-gradients.
            preconditioner (LinearOperator): Preconditioner for use in
                solving the normal equations. The default is None.
            preconditioning_method (PreconditioningMethod): If a preconditioner is not
                provided, this method is used to generate a preconditioner from the given
                operator. The default is the identity preconditioner.
            initial_model (model-space vector): Initial guess within an iterative solution.
                The default is zero.

        Returns:
            LinearOperator: Mapping from data space to least-squares solution.

        Raises:
            ValueError: If damping is not non-negative.
            ValueError: If solver is not a instance of LinearSolver.
        """

        # Set the operators.
        A = self.forward_operator
        Ci = self.data_error_measure.inverse_covariance
        N = self.normal_operator(damping)

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

        if initial_model is None:
            _initial_model = self.model_space.zero
        else:
            _initial_model = initial_model.copy()

        if isinstance(_solver, la.IterativeLinearSolver):
            Ni = _solver(N, preconditioner=_preconditioner, x0=_initial_model)
        elif isinstance(_solver, la.DirectLinearSolver):
            Ni = _solver(N)
        else:
            raise ValueError("input solver is of the wrong type.")

        return Ni @ A.adjoint @ Ci

    def model_error_measure(
        self,
        damping,
        /,
        *,
        solver=None,
        preconditioner=None,
        preconditioning_method=None,
        initial_model=None,
    ):
        """
        Returns the measure on the model space induced by the data errors under the least-squares solution.

        Args:
            damping (float): The norm damping parameter. Must be non-negative.
            solver (LinearSolver): Linear solver for solvint the normal equations.
                The default is conjugate-gradients.
            preconditioner (LinearOperator): Preconditioner for use in
                solving the normal equations. The default is the identity.
            preconditioning_method (PreconditioningMethod): If a preconditioner is not
                provided, this method is used to generate a preconditioner from the given
                operator. The default is the identity preconditioner.
            initial_model (model-space vector): Initial guess within an iterative solution.
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
        B = self.least_squares_operator(
            damping,
            solver=solver,
            preconditioner=preconditioner,
            initial_model=initial_model,
        )
        return self.data_error_measure.affine_mapping(operator=B)

    def model_measure(
        self,
        damping,
        data,
        /,
        *,
        solver=None,
        preconditioner=None,
        preconditioning_method=None,
        initial_model=None,
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
            initial_model (model-space vector): Initial guess within an iterative solution.
                The default is zero.

        Returns:
            GaussianMeasure: Measure on the model space induced by the
                least-squares solution for given data. Note that this measure only
                accounts for uncertainty due to the propagation of
                uncertainties within the data.

        Raises:
            ValueError: If damping is not non-negative.
            ValueError: If solver is not a instance of LinearSolver.
        """
        B = self.least_squares_operator(
            damping,
            solver=solver,
            preconditioner=preconditioner,
            preconditioning_method=preconditioning_method,
            initial_model=initial_model,
        )
        model = B(data)
        return self.data_error_measure.affine_mapping(operator=B, translation=model)

    def resolution_operator(
        self,
        damping,
        /,
        *,
        solver=None,
        preconditioner=None,
        preconditioning_method=None,
        initial_model=None,
    ):
        """
        Returns the resolution operator for the least-squares problem.

        Args:
            damping (float): The norm damping parameter. Must be non-negative.
            solver (LinearSolver): Linear solver for solvint the normal equations.
                The default is conjugate-gradients.
            preconditioner (LinearOperator): Preconditioner for use in
                solving the normal equations. The default is the identity.
            initial_model (model-space vector): Initial guess within an iterative solution.
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
        B = self.least_squares_operator(
            damping,
            solver=solver,
            preconditioner=preconditioner,
            preconditioning_method=preconditioning_method,
            initial_model=initial_model,
        )

        return B @ A

    def trade_off_curve(
        self,
        damping1,
        damping2,
        ndamping,
        data,
        /,
        *,
        solver=None,
        preconditioner=None,
        preconditioning_method=None,
        initial_model=None,
    ):
        """
        Plots a trade-off curve of squared model norm against chi-squared.

        Args:
            dampings (range): Damping values to plot.
            data (data-space vector): Observed data.
        """

        dampings = np.linspace(damping1, damping2, ndamping)

        squared_norms = []
        chi_squares = []

        if initial_model is None:
            model = self.model_space.zero
        else:
            model = initial_model.copy()

        for damping in dampings:
            B = self.least_squares_operator(
                damping,
                solver=solver,
                preconditioner=preconditioner,
                preconditioning_method=preconditioning_method,
                initial_model=model,
            )
            model = B(data)
            squared_norms.append(self.model_space.inner_product(model, model))
            chi_squares.append(self.chi_squared(model, data))

        plot_colourline(squared_norms, chi_squares, dampings)


class OccamInversion(ForwardProblem):
    """
    Class for performing the Occam inversions of Constable, Parker and Constable (1987)
    for a linear problem with Gaussian errors.
    """

    def __init__(
        self,
        forward_operator,
        data_error_measure,
        /,
        *,
        rtol=1.0e-5,
        damping_min=1.0e-5,
    ):
        super().__init__(forward_operator, data_error_measure)
        self._least_squares_inversion = LeastSquaresInversion(
            forward_operator, data_error_measure
        )
        self._rtol = rtol
        self._damping_min = damping_min

    def minimum_norm_operator(
        self,
        confidence_level,
        /,
        *,
        solver=None,
        preconditioner=None,
        preconditioning_method=None,
    ):
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
        critical_chi_squared = chi2.isf(1 - confidence_level, self.data_space.dim)

        def mapping(data):

            # check to see if zero-model is the solution.
            model = self.model_space.zero
            chi_squared = self.chi_squared(model, data)
            if chi_squared <= critical_chi_squared:
                return model

            # Local function for solving the least-squares problem
            def compute_model_chi_squared(damping, initial_model=None):
                B = self._least_squares_inversion.least_squares_operator(
                    damping,
                    solver=solver,
                    preconditioner=preconditioner,
                    preconditioning_method=preconditioning_method,
                    initial_model=initial_model,
                )
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
                while chi_squared > critical_chi_squared:
                    damping_low /= 2
                    model, chi_squared = compute_model_chi_squared(damping_low, model)
                    if damping_low < self._damping_min:
                        break

            if damping_high is None:
                damping_high = damping_low
                while chi_squared <= critical_chi_squared:
                    damping_high *= 2
                    model, chi_squared = compute_model_chi_squared(damping_high, model)

            # Use bisection to determine the damping.
            damping = 0.5 * (damping_high + damping_low)
            while np.abs(damping - damping_low) > self._rtol * damping:
                model, chi_squared = compute_model_chi_squared(damping, model)
                if chi_squared >= critical_chi_squared:
                    damping_high = damping
                else:
                    damping_low = damping
                damping = 0.5 * (damping_high + damping_low)

            return model

        return la.Operator(self.data_space, self.model_space, mapping)

    def resolution_operator(
        self,
        confidence_level,
        /,
        *,
        solver=None,
        preconditioner=None,
        preconditioning_method=None,
    ):
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
            confidence_level,
            solver=solver,
            preconditioner=preconditioner,
            preconditioning_method=preconditioning_method,
        )
        return la.Operator(self.model_space, self.model_space, lambda x: B(A(x)))
