"""
Module containing the base class for inversion methods
"""


class Inversion:
    """
    Base class for inversion methods.
    """

    def __init__(self, forward_problem):
        self._forward_problem = forward_problem

    @property
    def forward_problem(self):
        """
        Return the forward problem.
        """
        return self._forward_problem

    @property
    def model_space(self):
        """
        Return the model space.
        """
        return self.forward_problem.model_space

    @property
    def data_space(self):
        """
        Return the data space.
        """
        return self.forward_problem.data_space

    @property
    def model_space_dim(self):
        """
        Return model space dimension.
        """
        return self.model_space.dim

    @property
    def data_space_dim(self):
        """
        Return data space dimension.
        """
        return self.data_space.dim

    def assert_data_error_measure(self):
        """
        Raises a not implemented error if a data error
        measure is not present
        """
        if not self.forward_problem.data_error_measure_set:
            raise NotImplementedError(
                "Data error measure requried for the inversion method"
            )

    def assert_inverse_data_covariance(self):
        """
        Raises a not implemented error if a data error measure does not have an
        inverse covariance set.
        """
        if (
            self.forward_problem.data_error_measure_set
            and not self.forward_problem.data_error_measure.inverse_covariance_set
        ):
            raise NotImplementedError(
                "Inverse data covariance needed for the inversion method"
            )
