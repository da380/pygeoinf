"""
Module defined the forward problem class. 
"""

from scipy.stats import chi2
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.direct_sum import BlockLinearOperator


class ForwardProblem:
    """
    Class for forward problems. A class instance is defined by
    setting the forward operator and the data error measure.
    """

    def __init__(self, forward_operator, data_error_measure=None):
        """
        Args:
            forward_operator (LinearOperator): Mapping from the model to data space.
            data_error_measure (GaussianMeasure): Gaussian measure from which data errors
                are assumed to be drawn. Default is None, in which case the data is taken
                to be error free.
        """
        self._forward_operator = forward_operator
        self._data_error_measure = data_error_measure
        if self.data_error_measure_set:
            assert self.data_space == data_error_measure.domain

    @property
    def forward_operator(self):
        """The forward operator."""
        return self._forward_operator

    @property
    def data_error_measure_set(self):
        """
        Returns true is data error measure has been set.
        """
        return self._data_error_measure is not None

    @property
    def data_error_measure(self):
        """The measure from which data errors are drawn."""
        if not self.data_error_measure_set:
            raise NotImplementedError("Data error measure has not been set")
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

    Data error measures should be set either for not or for all of the
    forward problems.
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

        if all(
            forward_problem.data_error_measure_set
            for forward_problem in forward_problems
        ):
            data_error_measure = GaussianMeasure.from_direct_sum(
                [
                    forward_problem.data_error_measure
                    for forward_problem in forward_problems
                ]
            )
        else:
            data_error_measure = None

        return LinearForwardProblem(
            forward_operator @ forward_operator.domain.subspace_inclusion(0),
            data_error_measure,
        )

    def data_measure(self, model):
        """Returns the data measure for a given model."""
        if not self.data_error_measure_set:
            raise NotImplementedError("Data error measure has not been set")
        return self.data_error_measure.affine_mapping(
            translation=self.forward_operator(model)
        )

    def synthetic_data(self, model):
        """Returns synthetic data corresponding to a given model."""
        return self.data_measure(model).sample()

    def synthetic_model_and_data(self, mu):
        """
        Given a Gaussian measure on the model space, returns a random model and
        the corresponding synthetic data.
        """
        u = mu.sample()
        if self.data_error_measure_set:
            return u, self.data_measure(u).sample()
        else:
            return u, self.forward_operator(u)

    def crtical_chi_squared(self, significance_level):
        """
        Returns the crtical value of chi-squared for a given significance level.
        """
        return chi2.ppf(significance_level, self.data_space.dim)

    def chi_squared(self, model, data):
        """
        Returns the chi-squared statistic for a given model and observed data. The a
        data error measure has not been set, this returns the squared norm of the
        data residual.

        Raises:
            NotImplementedError -- If the inverse covariance for the data error measure
                has not been set.

        """
        if self.data_error_measure_set:

            if self.data_error_measure.inverse_covariance is None:
                raise NotImplementedError("Inverse covariance has not been implemented")

            difference = self.data_space.subtract(data, self.forward_operator(model))
            difference = self.data_space.subtract(
                difference, self.data_error_measure.expectation
            )
            inverse_data_covariance = self.data_error_measure.inverse_covariance
            return self.data_space.inner_product(
                inverse_data_covariance(difference), difference
            )

        else:

            difference = self.data_space.subtract(data, self.forward_operator(model))
            return self.data_space.squared_norm(difference)

    def chi_squared_test(self, significance_level, model, data):
        """
        Returns True if the model is compatible the given data at the
        specified significance level.
        """
        return self.chi_squared(model, data) < self.crtical_chi_squared(
            significance_level
        )
