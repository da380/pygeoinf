"""
Module defined the forward problem class. 
"""

from pygeoinf.hilbert import LinearOperator


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

    def data_measure(self, model):
        """Returns the data measure for a given model."""
        return self.data_error_measure.affine_mapping(
            translation=self.forward_operator(model)
        )

    def chi_squared(self, model, data):
        """Returns the chi-squared statistic for a given model and observed data."""
        difference = data - self.forward_operator(model)
        inverse_data_covariance = self.data_error_measure.inverse_covariance
        return self.data_space.inner_product(
            inverse_data_covariance(difference), difference
        )
