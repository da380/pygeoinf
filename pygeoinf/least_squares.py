import numpy as np
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
        return LeastSquaresInversion(forward_problem.forward_operator,
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

    def least_squares_operator(self, damping, /, *,
                               solver=None,
                               preconditioner=None,
                               preconditioning_method=None,
                               initial_model=None):
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

    def model_error_measure(self, damping, /, *,
                            solver=None,
                            preconditioner=None,
                            preconditioning_method=None,
                            initial_model=None):
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
        B = self.least_squares_operator(damping,
                                        solver=solver,
                                        preconditioner=preconditioner,
                                        initial_model=initial_model)
        return self.data_error_measure.affine_mapping(operator=B)

    def model_measure(self, damping, data, /, *,
                      solver=None,  preconditioner=None,
                      preconditioning_method=None,
                      initial_model=None):
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
        B = self.least_squares_operator(damping,
                                        solver=solver,
                                        preconditioner=preconditioner,
                                        preconditioning_method=preconditioning_method,
                                        initial_model=initial_model)
        model = B(data)
        return self.data_error_measure.affine_mapping(operator=B, translation=model)

    def resolution_operator(self, damping, /, *, solver=None,
                            preconditioner=None,
                            preconditioning_method=None,
                            initial_model=None):
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
        B = self.least_squares_operator(damping,
                                        solver=solver,
                                        preconditioner=preconditioner,
                                        preconditioning_method=preconditioning_method,
                                        initial_model=initial_model)

        return B @ A

    def trade_off_curve(self, damping1, damping2, ndamping, data, /, *, solver=None,
                        preconditioner=None,
                        preconditioning_method=None,
                        initial_model=None):
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
                damping, solver=solver, preconditioner=preconditioner,
                preconditioning_method=preconditioning_method, initial_model=model)
            model = B(data)
            squared_norms.append(self.model_space.inner_product(model, model))
            chi_squares.append(self.chi_squared(model, data))

        plot_colourline(squared_norms, chi_squares, dampings)
