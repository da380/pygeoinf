"""
Tests for the forward_problem module.
"""

import pytest
import numpy as np
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.forward_problem import LinearForwardProblem

# =============================================================================
# Fixtures for the Test Problem
# =============================================================================


@pytest.fixture
def model_space() -> EuclideanSpace:
    """Provides a simple 3D model space."""
    return EuclideanSpace(dim=3)


@pytest.fixture
def data_space() -> EuclideanSpace:
    """Provides a simple 2D data space."""
    return EuclideanSpace(dim=2)


@pytest.fixture
def forward_operator(
    model_space: EuclideanSpace, data_space: EuclideanSpace
) -> LinearOperator:
    """Provides a simple linear operator for the forward problem."""
    matrix = np.random.randn(data_space.dim, model_space.dim)
    return LinearOperator.from_matrix(model_space, data_space, matrix)


@pytest.fixture
def data_error_measure(data_space: EuclideanSpace) -> GaussianMeasure:
    """Provides a Gaussian measure for the data errors."""
    # Use a non-zero mean to test that it's handled correctly
    mean = np.array([0.5, -0.5])
    std_devs = np.array([0.1, 0.2])
    return GaussianMeasure.from_standard_deviations(
        data_space, std_devs, expectation=mean
    )


@pytest.fixture
def forward_problem(
    forward_operator: LinearOperator, data_error_measure: GaussianMeasure
) -> LinearForwardProblem:
    """Provides a complete LinearForwardProblem instance."""
    return LinearForwardProblem(forward_operator, data_error_measure=data_error_measure)


# =============================================================================
# Tests for LinearForwardProblem
# =============================================================================


class TestLinearForwardProblem:
    """A suite of tests for the LinearForwardProblem class."""

    def test_initialization(self, forward_problem: LinearForwardProblem):
        """Tests that the forward problem is initialized correctly."""
        assert forward_problem is not None
        assert forward_problem.data_error_measure_set is True

    def test_data_measure(
        self, forward_problem: LinearForwardProblem, model_space: EuclideanSpace
    ):
        """Tests that the data_measure method correctly shifts the mean."""
        model = model_space.random()
        data_measure = forward_problem.data_measure_from_model(model)

        # The mean of the data measure should be A(u) + error_mean
        expected_mean = (
            forward_problem.forward_operator(model)
            + forward_problem.data_error_measure.expectation
        )

        assert np.allclose(data_measure.expectation, expected_mean)
        # The covariance should be unchanged
        assert np.allclose(
            data_measure.covariance.matrix(dense=True),
            forward_problem.data_error_measure.covariance.matrix(dense=True),
        )

    def test_synthetic_data_statistics(
        self, forward_problem: LinearForwardProblem, model_space: EuclideanSpace
    ):
        """
        A statistical test to verify the properties of synthetic data.
        """
        model = model_space.random()
        n_samples = 2000  # Use a large number of samples for statistical significance

        # Generate many synthetic data points
        samples = [forward_problem.synthetic_data(model) for _ in range(n_samples)]

        # 1. Test the sample mean
        expected_mean = (
            forward_problem.forward_operator(model)
            + forward_problem.data_error_measure.expectation
        )
        sample_mean = np.mean(samples, axis=0)
        # Check that the sample mean is close to the expected mean
        # The tolerance is based on the standard error of the mean
        assert np.allclose(sample_mean, expected_mean, atol=0.1)

        # 2. Test the sample covariance
        expected_covariance = forward_problem.data_error_measure.covariance.matrix(
            dense=True
        )
        sample_covariance = np.cov(np.array(samples).T)
        assert np.allclose(sample_covariance, expected_covariance, atol=0.01)

    def test_chi_squared(
        self,
        forward_problem: LinearForwardProblem,
        model_space: EuclideanSpace,
        data_space: EuclideanSpace,
    ):
        """Tests the chi_squared calculation for a simple case."""
        # Create a problem with identity data covariance for simplicity
        identity_error = GaussianMeasure.from_standard_deviation(data_space, 1.0)
        problem = LinearForwardProblem(
            forward_problem.forward_operator, data_error_measure=identity_error
        )

        model = model_space.random()
        data = data_space.random()

        # Calculate chi-squared
        chi2_val = problem.chi_squared(model, data)

        # For identity covariance, chi-squared should be the squared norm of the residual
        residual = data - problem.forward_operator(model)
        expected_chi2 = np.dot(residual, residual)

        assert np.isclose(chi2_val, expected_chi2)

    def test_from_direct_sum(
        self, forward_operator: LinearOperator, data_error_measure: GaussianMeasure
    ):
        """Tests the creation of a combined forward problem from a list."""
        # Create two identical forward problems
        fp1 = LinearForwardProblem(
            forward_operator, data_error_measure=data_error_measure
        )
        fp2 = LinearForwardProblem(
            forward_operator, data_error_measure=data_error_measure
        )

        # Combine them using from_direct_sum
        combined_fp = LinearForwardProblem.from_direct_sum([fp1, fp2])

        # 1. Check the model space
        assert combined_fp.model_space == fp1.model_space

        # 2. Check the data space dimension
        assert combined_fp.data_space.dim == 2 * fp1.data_space.dim

        # 3. Check that the combined data error measure has a block-diagonal covariance
        combined_cov = combined_fp.data_error_measure.covariance.matrix(dense=True)
        single_cov = fp1.data_error_measure.covariance.matrix(dense=True)

        d = fp1.data_space.dim
        # Top-left block should be the original covariance
        assert np.allclose(combined_cov[:d, :d], single_cov)
        # Bottom-right block should be the original covariance
        assert np.allclose(combined_cov[d:, d:], single_cov)
        # Off-diagonal blocks should be zero
        assert np.allclose(combined_cov[:d, d:], 0)
        assert np.allclose(combined_cov[d:, :d], 0)

    def test_data_measure_from_model_measure(
        self, forward_problem: LinearForwardProblem, model_space: EuclideanSpace
    ):
        """Tests the induced data measure from a model measure."""
        # Create a prior with a non-zero mean
        prior_mean = model_space.random()
        prior = GaussianMeasure.from_standard_deviation(
            model_space, 2.0, expectation=prior_mean
        )

        data_prior = forward_problem.data_measure_from_model_measure(prior)

        # 1. Check Expectation: A(mu_u) + mu_e
        A = forward_problem.forward_operator.matrix(dense=True)
        expected_mean = A @ prior_mean + forward_problem.data_error_measure.expectation
        assert np.allclose(data_prior.expectation, expected_mean)

        # 2. Check Covariance: A C_u A^T + C_e
        Cu = prior.covariance.matrix(dense=True)
        Ce = forward_problem.data_error_measure.covariance.matrix(dense=True)
        expected_cov = A @ Cu @ A.T + Ce

        assert np.allclose(data_prior.covariance.matrix(dense=True), expected_cov)

    def test_joint_measure(
        self, forward_problem: LinearForwardProblem, model_space: EuclideanSpace
    ):
        """Tests the construction of the joint (model, data) measure."""
        prior_mean = model_space.random()
        prior = GaussianMeasure.from_standard_deviation(
            model_space, 2.0, expectation=prior_mean
        )

        joint = forward_problem.joint_measure(prior)

        # 1. Check Expectation
        A = forward_problem.forward_operator.matrix(dense=True)
        expected_data_mean = (
            A @ prior_mean + forward_problem.data_error_measure.expectation
        )
        expected_joint_mean = np.concatenate([prior_mean, expected_data_mean])

        # Flatten joint.expectation to match the concatenated expected mean
        assert np.allclose(np.concatenate(joint.expectation), expected_joint_mean)

        # 2. Check Block Covariance Structure
        Cu = prior.covariance.matrix(dense=True)
        Ce = forward_problem.data_error_measure.covariance.matrix(dense=True)

        expected_joint_cov = np.block([[Cu, Cu @ A.T], [A @ Cu, A @ Cu @ A.T + Ce]])

        assert np.allclose(joint.covariance.matrix(dense=True), expected_joint_cov)

    def test_parameterized_problem(
        self,
        forward_problem: LinearForwardProblem,
        model_space: EuclideanSpace,
    ):
        """Tests the parameterized_problem method, including dense matrix freezing."""
        from pygeoinf.linear_operators import DenseMatrixLinearOperator

        # 1. Create a smaller parameter space and a parameterization operator
        param_space = EuclideanSpace(dim=2)
        param_matrix = np.random.randn(model_space.dim, param_space.dim)
        param_op = LinearOperator.from_matrix(param_space, model_space, param_matrix)

        # 2. Test standard parameterization (dense=False)
        param_fp = forward_problem.parameterized_problem(param_op)

        # The new model space should be the parameter space
        assert param_fp.model_space == param_space
        # The data space should remain unchanged
        assert param_fp.data_space == forward_problem.data_space

        # Verify the new forward operator matrix is exactly A @ M
        A_mat = forward_problem.forward_operator.matrix(dense=True)
        expected_matrix = A_mat @ param_matrix
        assert np.allclose(
            param_fp.forward_operator.matrix(dense=True), expected_matrix
        )

        # 3. Test dense parameterization (dense=True)
        dense_fp = forward_problem.parameterized_problem(param_op, dense=True)

        # Verify both the forward operator and the data error covariance were squashed
        # into memory-contiguous dense operators.
        assert isinstance(dense_fp.forward_operator, DenseMatrixLinearOperator)
        assert isinstance(
            dense_fp.data_error_measure.covariance, DenseMatrixLinearOperator
        )

        # Verify the domain mismatch safety check
        bad_param_op = LinearOperator.from_matrix(param_space, param_space, np.eye(2))
        with pytest.raises(ValueError, match="codomain of the parameterization"):
            forward_problem.parameterized_problem(bad_param_op)
