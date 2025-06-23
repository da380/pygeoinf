"""
Module for classes related to the solution of inverse problems via optimisation methods. 
"""

from scipy.stats import chi2
from pygeoinf.hilbert_space import Operator
from pygeoinf.inversion import Inversion
from pygeoinf.linear_solvers import IterativeLinearSolver


class LinearLeastSquaresInversion(Inversion):
    """
    Class for the solution of regularised linear least-squares problems within a Hilbert space.
    """

    def __init__(self, forward_problem):
        """
        Args:
            forward_problem (LinearForwardProblem): The forward problem. If the problem has
                a data errror measure, then this measure must have its inverse covariance
                operator set.
        """
        super().__init__(forward_problem)
        self.assert_inverse_data_covariance()

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
        identity = self.forward_problem.model_space.identity_operator()

        if self.forward_problem.data_error_measure_set:

            inverse_data_covariance = (
                self.forward_problem.data_error_measure.inverse_covariance
            )
            return (
                forward_operator.adjoint @ inverse_data_covariance @ forward_operator
                + damping * identity
            )
        else:
            return forward_operator.adjoint @ forward_operator + damping * identity

    def least_squares_operator(self, damping, solver, /, *, preconditioner=None):
        """
        Returns an operator that maps data to the least squares solution.

        Args:
            damping (float): The norm damping parameter. Must be non-negative.
            solver (LinearSolver): Linear solver for the normal equations.
            preconditioner (LinearOperator): Preconditioner for use in
                solving the normal equations. The default is None.

        Returns:
            Operator: Mapping from data space to least-squares solution.

        Raises:
            ValueError: If damping is not non-negative.
            ValueError: If solver is not a instance of LinearSolver.

        Notes. If the data is error-free the object returned is a LinearOperator.
        """

        forward_operator = self.forward_problem.forward_operator
        normal_operator = self.normal_operator(damping)

        if isinstance(solver, IterativeLinearSolver):
            inverse_normal_operator = solver(
                normal_operator, preconditioner=preconditioner
            )
        else:
            inverse_normal_operator = solver(normal_operator)

        if self.forward_problem.data_error_measure_set:
            inverse_data_covariance = (
                self.forward_problem.data_error_measure.inverse_covariance
            )

            def mapping(data):
                shifted_data = self.forward_problem.data_space.subtract(
                    data, self.forward_problem.data_error_measure.expectation
                )
                return (
                    inverse_normal_operator
                    @ forward_operator.adjoint
                    @ inverse_data_covariance
                )(shifted_data)

            return Operator(self.data_space, self.model_space, mapping)

        else:
            return inverse_normal_operator @ forward_operator.adjoint

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

        if not self.forward_problem.data_error_measure_set:
            raise NotImplementedError("Data error measure not set")

        least_squares_operator = self.least_squares_operator(
            damping,
            solver,
            preconditioner=preconditioner,
        )
        model = least_squares_operator(data)
        return self.forward_problem.data_error_measure.affine_mapping(
            operator=least_squares_operator, translation=model
        )


class LinearMinimumNormInversion(Inversion):
    """
    Class for the solution of linear minimum norm problems in a Hilbert space.
    """

    def __init__(self, forward_problem):
        """
        Args:
            forward_problem (LinearForwardProblem): The forward problem. If the problem has
                a data errror measure, then this measure must have its inverse covariance
                operator set.
        """
        super().__init__(forward_problem)
        self.assert_inverse_data_covariance()

    def minimum_norm_operator(
        self,
        solver,
        /,
        *,
        preconditioner=None,
        significance_level=0.05,
        minimum_damping=0,
        maxiter=100,
        rtol=1.0e-6,
        atol=0,
    ):
        """
        Return the operator that maps data to the minimum norm solution.
        In the case of error-free data this operator is linear, but once
        data errors are present it is a non-linear opearator.

        Args:
            solver (LinearSolver): Solver for solution of the necessary linear systems.
            preconditioner (LinearOperator): Preconditioner for use with iterative solvers.
            significance_level (float): Significance level used to set the crtical value
                for chi-squared. Must lie in the interval (0,1). Default is 0.05 which
                corresponds to 95% acceptance.
            minimum_damping (float): Minimum value of the damping when looking for a
                lower bound on the damping. Default is 0.
            maxiter (int): Maximum number of iterations within the bracketing method.
                Default is 100.
            rtol (float): Relative tolerance within the bracketing step. Default is 1e-6
            atyol (float): Absolute tolerance within the bracketing step. Default is 0.
        """

        if self.forward_problem.data_error_measure_set:

            critical_value = self.forward_problem.crtical_chi_squared(
                significance_level
            )
            least_squares_inversion = LinearLeastSquaresInversion(self.forward_problem)

            def least_squares_mapping(damping, data, model0=None):

                forward_operator = self.forward_problem.forward_operator
                normal_operator = least_squares_inversion.normal_operator(damping)
                inverse_data_covariance = (
                    self.forward_problem.data_error_measure.inverse_covariance
                )

                shifted_data = self.data_space.subtract(
                    data, self.forward_problem.data_error_measure.expectation
                )

                if isinstance(solver, IterativeLinearSolver):
                    model_in = (forward_operator.adjoint @ inverse_data_covariance)(
                        shifted_data
                    )
                    model = solver.solve_linear_system(
                        normal_operator,
                        preconditioner,
                        model_in,
                        x0=model0,
                    )

                else:
                    inverse_normal_operator = solver(normal_operator)
                    model = (
                        inverse_normal_operator
                        @ forward_operator.adjoint
                        @ inverse_data_covariance
                    )(shifted_data)

                chi_squared = self.forward_problem.chi_squared(model, data)
                return chi_squared, model

            def mapping(data):

                model = self.model_space.zero
                chi_squared = self.forward_problem.chi_squared(model, data)
                if chi_squared <= critical_value:
                    return model

                damping = 1
                chi_squared, _ = least_squares_mapping(damping, data)

                damping_lower = damping if chi_squared <= critical_value else None
                damping_upper = damping if chi_squared > critical_value else None

                if damping_lower is None:
                    it = 0
                    while chi_squared > critical_value:
                        it += 1
                        damping /= 2
                        chi_squared, _ = least_squares_mapping(damping, data)
                        if chi_squared < minimum_damping or it == maxiter:
                            raise RuntimeError("Crtical value cannot be obtained")
                    damping_lower = damping

                if damping_upper is None:
                    while chi_squared < critical_value:
                        damping *= 2
                        chi_squared, _ = least_squares_mapping(damping, data)
                    damping_upper = damping

                model0 = None
                for _ in range(maxiter):

                    damping = 0.5 * (damping_lower + damping_upper)
                    chi_squared, model = least_squares_mapping(damping, data, model0)
                    model0 = self.model_space.copy(model)

                    if chi_squared < critical_value:
                        damping_lower = damping
                    else:
                        damping_upper = damping

                    if damping_upper - damping_lower < atol + rtol * (
                        damping_lower + damping_upper
                    ):
                        return model

                raise RuntimeError("Bracketing has failed")

            return Operator(self.model_space, self.data_space, mapping)

        else:
            forward_operator = self.forward_problem.forward_operator
            normal_operator = forward_operator @ forward_operator.adjoint
            inverse_normal_operator = solver(normal_operator)
            return forward_operator.adjoint @ inverse_normal_operator
