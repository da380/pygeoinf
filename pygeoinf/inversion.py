"""
Provides the abstract base class for all inversion algorithms.

This module defines the `Inversion` class, which serves as a common
foundation for various methods that solve an inverse problem. Its primary role
is to maintain a reference to the `ForwardProblem` being solved, providing a
consistent interface and convenient access to the problem's core components like
the model space and data space.

It also includes helper methods to assert preconditions required by different
inversion techniques, such as the existence of a data error measure.
"""

from __future__ import annotations

from .forward_problem import LinearForwardProblem
from .hilbert_space import HilbertSpace


class Inversion:
    """
    An abstract base class for inversion methods.

    This class provides a common structure for different inversion algorithms
    (e.g., Bayesian, Least Squares). Its main purpose is to hold a reference
    to the forward problem being solved and provide convenient access to its
    properties. Subclasses should inherit from this class to implement a
    specific inversion technique.
    """

    def __init__(self, forward_problem: "LinearForwardProblem", /) -> None:
        """
        Initializes the Inversion class.

        Args:
            forward_problem: An instance of a forward problem that defines the
                relationship between model parameters and data.
        """
        self._forward_problem: "LinearForwardProblem" = forward_problem

    @property
    def forward_problem(self) -> "LinearForwardProblem":
        """The forward problem associated with this inversion."""
        return self._forward_problem

    @property
    def model_space(self) -> "HilbertSpace":
        """The model space (domain) of the forward problem."""
        return self.forward_problem.model_space

    @property
    def data_space(self) -> "HilbertSpace":
        """The data space (codomain) of the forward problem."""
        return self.forward_problem.data_space

    def assert_data_error_measure(self) -> None:
        """
        Checks if a data error measure is set in the forward problem.

        This is a precondition for statistical inversion methods.

        Raises:
            AttributeError: If no data error measure has been set.
        """
        if not self.forward_problem.data_error_measure_set:
            raise AttributeError(
                "A data error measure is required for this inversion method."
            )

    def assert_inverse_data_covariance(self) -> None:
        """
        Checks if the data error measure has an inverse covariance.

        This is a precondition for methods that require the data precision
        matrix (the inverse of the data error covariance).

        Raises:
            AttributeError: If no data error measure is set, or if the measure
                does not have an inverse covariance operator defined.
        """
        self.assert_data_error_measure()
        if not self.forward_problem.data_error_measure.inverse_covariance_set:
            raise AttributeError(
                "An inverse data covariance (precision) operator is required for this inversion method."
            )
