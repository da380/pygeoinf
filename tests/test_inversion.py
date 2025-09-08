"""
Tests for the Inversion base class.
"""

import pytest
import numpy as np
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.forward_problem import LinearForwardProblem
from pygeoinf.inversion import Inversion


# =============================================================================
# Fixtures for Different Forward Problem Setups
# =============================================================================


@pytest.fixture
def forward_problem_no_error() -> LinearForwardProblem:
    """A forward problem with no data error measure."""
    model_space = EuclideanSpace(2)
    data_space = EuclideanSpace(2)
    op = LinearOperator.from_matrix(model_space, data_space, np.eye(2))
    return LinearForwardProblem(op)


@pytest.fixture
def forward_problem_with_error_no_inv_cov() -> LinearForwardProblem:
    """A forward problem with a data error measure that lacks an inverse covariance."""
    model_space = EuclideanSpace(2)
    data_space = EuclideanSpace(2)
    op = LinearOperator.from_matrix(model_space, data_space, np.eye(2))
    # Create a measure with a covariance factor, but no inverse factor
    cov_factor = LinearOperator.from_matrix(
        EuclideanSpace(2), data_space, 0.1 * np.eye(2)
    )
    error_measure = GaussianMeasure(covariance_factor=cov_factor)
    return LinearForwardProblem(op, data_error_measure=error_measure)


@pytest.fixture
def forward_problem_with_inv_cov() -> LinearForwardProblem:
    """A forward problem with a data error measure that has an inverse covariance."""
    model_space = EuclideanSpace(2)
    data_space = EuclideanSpace(2)
    op = LinearOperator.from_matrix(model_space, data_space, np.eye(2))
    error_measure = GaussianMeasure.from_standard_deviation(data_space, 0.1)
    return LinearForwardProblem(op, data_error_measure=error_measure)


# A simple concrete subclass for testing the abstract Inversion class
class ConcreteInversion(Inversion):
    pass


# =============================================================================
# Tests for the Inversion Class
# =============================================================================


class TestInversion:
    """A suite of tests for the Inversion base class."""

    def test_initialization(self, forward_problem_no_error: LinearForwardProblem):
        """Tests that the Inversion class can be instantiated."""
        inv = ConcreteInversion(forward_problem_no_error)
        assert inv is not None
        assert inv.forward_problem == forward_problem_no_error

    def test_assert_data_error_measure_raises(
        self, forward_problem_no_error: LinearForwardProblem
    ):
        """
        Tests that assert_data_error_measure raises an error when no
        measure is set.
        """
        inv = ConcreteInversion(forward_problem_no_error)
        with pytest.raises(AttributeError):
            inv.assert_data_error_measure()

    def test_assert_data_error_measure_passes(
        self, forward_problem_with_inv_cov: LinearForwardProblem
    ):
        """
        Tests that assert_data_error_measure passes when a measure is set.
        """
        inv = ConcreteInversion(forward_problem_with_inv_cov)
        try:
            inv.assert_data_error_measure()
        except AttributeError:
            pytest.fail("assert_data_error_measure raised an error unexpectedly.")

    def test_assert_inverse_data_covariance_raises_no_measure(
        self, forward_problem_no_error: LinearForwardProblem
    ):
        """
        Tests that assert_inverse_data_covariance raises an error when
        no data error measure is set at all.
        """
        inv = ConcreteInversion(forward_problem_no_error)
        with pytest.raises(AttributeError):
            inv.assert_inverse_data_covariance()

    def test_assert_inverse_data_covariance_raises_no_inv_cov(
        self, forward_problem_with_error_no_inv_cov: LinearForwardProblem
    ):
        """
        Tests that assert_inverse_data_covariance raises an error when the
        measure exists but has no inverse covariance.
        """
        inv = ConcreteInversion(forward_problem_with_error_no_inv_cov)
        with pytest.raises(AttributeError):
            inv.assert_inverse_data_covariance()

    def test_assert_inverse_data_covariance_passes(
        self, forward_problem_with_inv_cov: LinearForwardProblem
    ):
        """
        Tests that assert_inverse_data_covariance passes when an inverse
        covariance is available.
        """
        inv = ConcreteInversion(forward_problem_with_inv_cov)
        try:
            inv.assert_inverse_data_covariance()
        except AttributeError:
            pytest.fail("assert_inverse_data_covariance raised an error unexpectedly.")
