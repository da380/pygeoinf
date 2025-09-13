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


from .hilbert_space import HilbertSpace
from .nonlinear_operators import NonLinearOperator
from .linear_operators import LinearOperator
from .forward_problem import LinearForwardProblem, ForwardProblem


class Inversion:
    """
    A base class for inversion methods.

    This class provides a common structure for different inversion and inference algorithms
    (e.g., Bayesian, Least Squares). Its main purpose is to hold a reference
    to the forward problem being solved and provide convenient access to its
    properties. Subclasses should inherit from this class to implement a
    specific inversion algorithm.
    """

    def __init__(self, forward_problem: ForwardProblem, /) -> None:
        """
        Initializes the Inversion class.

        Args:
            forward_problem: An instance of a forward problem that defines the
                relationship between model parameters and data.
        """
        self._forward_problem: ForwardProblem = forward_problem

    @property
    def forward_problem(self) -> ForwardProblem:
        """The forward problem associated with this inversion."""
        return self._forward_problem

    @property
    def model_space(self) -> HilbertSpace:
        """The model space (domain) of the forward problem."""
        return self.forward_problem.model_space

    @property
    def data_space(self) -> HilbertSpace:
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


class LinearInversion(Inversion):
    """
    An abstract base class for linear inversion algorithms.
    """

    def __init__(self, forward_problem: LinearForwardProblem, /) -> None:
        """
        Initializes the LinearInversion class.

        Args:
            forward_problem: An instance of a linear forward problem.
        """
        if not isinstance(forward_problem, LinearForwardProblem):
            raise ValueError("Forward problem must be a LinearForwardProblem.")
        super().__init__(forward_problem)


class Inference(Inversion):
    """
    A base class for inference algorithms. These methods inherit common functionality from
    the inversion base class, but need not themselves derive from a specific inversion scheme.

    Within an inference problem, the aim is to estimate some property of the unknown model,
    and hence a property operator mapping from the model to a property space must be
    specified.
    """

    def __init__(
        self, forward_problem: ForwardProblem, property_operator: NonLinearOperator
    ) -> None:
        """
        Initializes the Inference class.

        Args:
            forward_problem: An instance of a forward problem that defines the
                relationship between model parameters and data.
            property_operator: A mapping takes elements of the model space to
                property vector of interest.

        Raises:
            ValueError: If the domain of the property operator is
                not equal to the model space.
        """

        super().__init__(forward_problem)

        if property_operator.domain != self.model_space:
            raise ValueError("Property operator incompatible with model space")

        self._property_operator = property_operator

    @property
    def property_operator(self) -> NonLinearOperator:
        """
        Returns the property operator.
        """
        return self._property_operator

    @property
    def property_space(self) -> HilbertSpace:
        """
        Returns the property space.
        """
        return self.property_operator.codomain


class LinearInference(Inference):
    """
    A base class for linear inference algorithms.
    """

    def __init__(
        self, forward_problem: LinearForwardProblem, property_operator: LinearOperator
    ) -> None:
        """
        Initializes the LinearInference class.

        Args:
            forward_problem: An instance of a linear forward problem that defines the
                relationship between model parameters and data.
            property_operator: A linear mapping takes elements of the model space to
                property vector of interest.

        Raises:
            ValueError: If the domain of the property operator is
                not equal to the model space.
        """

        if not isinstance(forward_problem, LinearForwardProblem):
            raise ValueError("Forward problem must be linear")

        if not isinstance(property_operator, LinearOperator):
            raise ValueError("Property mapping must be linear")

        super().__init__(forward_problem, property_operator)
