"""
Module defined the forward problem class. 
"""

from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.direct_sum import BlockLinearOperator


class ForwardProblem:
    """
    Class for forward problems. A class instance is defined by
    setting the forward operator and the data error measure.
    """

    def __init__(self, forward_operator, data_error_measure):
        """
        Args:
            forward_operator (LinearOperator): Mapping from the model to data space.
            data_error_measure (GaussianMeasure): Gaussian measure from which data errors
                are assumed to be drawn.
        """
        self._forward_operator = forward_operator
        self._data_error_measure = data_error_measure
        assert self.data_space == data_error_measure.domain

    @property
    def forward_operator(self):
        """The forward operator."""
        return self._forward_operator

    @property
    def data_error_measure(self):
        """The measure from which data errors are drawn."""
        return self._data_error_measure

    @property
    def model_space(self):
        """The model space."""
        return self.forward_operator.domain

    @property
    def data_space(self):
        """The data space."""
        return self.forward_operator.codomain


class LinearForwardProblem(ForwardProblem):
    """
    Class for linear forward problems. A class instance is defined by
    setting the forward operator and the data error measure.
    """

    @staticmethod
    def from_direct_sum(forward_problems):
        """
        Given a list of forward problems with a common model space, forms
        the joint forward problem.
        """

        forward_operator = BlockLinearOperator(
            [[forward_problem.forward_operator] for forward_problem in forward_problems]
        )

        data_error_measure = GaussianMeasure.from_direct_sum(
            [forward_problem.data_error_measure for forward_problem in forward_problems]
        )

        return LinearForwardProblem(
            forward_operator @ forward_operator.domain.subspace_inclusion(0),
            data_error_measure,
        )

    def data_measure(self, model):
        """Returns the data measure for a given model."""
        return self.data_error_measure.affine_mapping(
            translation=self.forward_operator(model)
        )

    def chi_squared(self, model, data):
        """
        Returns the chi-squared statistic for a given model and observed data.

        Raises:
            NotImplementedError -- If the inverse covariance for the data error measure
                has not been set.

        """
        if self.data_error_measure.inverse_covariance is None:
            raise NotImplementedError("Inverse covariance has not been implemented")

        difference = self.data_space.subtract(data, self.forward_operator(model))
        inverse_data_covariance = self.data_error_measure.inverse_covariance
        return self.data_space.inner_product(
            inverse_data_covariance(difference), difference
        )
