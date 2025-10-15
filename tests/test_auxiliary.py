"""
Tests for the auxiliary module.

This module contains comprehensive tests for the empirical_data_error_measure function,
which generates empirical data error measures based on samples from a model space measure.
The tests cover:

- Basic functionality and return types
- Behavior with different parameter settings (scale_factor, n_samples)
- Statistical properties and approximation quality
- Mean centering behavior using empirical sample means
- Edge cases and error conditions
- Integration with different Hilbert space implementations

The tests use parametrized fixtures to ensure compatibility across different
space implementations and use both deterministic and stochastic testing approaches
to validate the empirical statistical methods.
"""

import pytest
import numpy as np
from pygeoinf.hilbert_space import EuclideanSpace, HilbertSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.auxiliary import empirical_data_error_measure


# =============================================================================
# Parametrized Fixtures
# =============================================================================

# Define the different Hilbert space instances we want to test against.
space_implementations = [
    EuclideanSpace(dim=4),
    EuclideanSpace(dim=10),
]


@pytest.fixture(params=space_implementations)
def model_space(request) -> HilbertSpace:
    """Provides parametrized HilbertSpace instances for the model space."""
    return request.param


@pytest.fixture
def data_space() -> HilbertSpace:
    """Provides a data space instance for testing."""
    return EuclideanSpace(dim=6)


@pytest.fixture
def model_measure(model_space: HilbertSpace) -> GaussianMeasure:
    """Provides a GaussianMeasure instance for the model space."""
    # Use a non-zero mean and a simple covariance matrix
    mean = model_space.random()
    cov_matrix = np.diag(np.random.rand(model_space.dim) + 0.1)
    # Make it symmetric and positive definite
    cov_matrix = cov_matrix.T @ cov_matrix
    return GaussianMeasure.from_covariance_matrix(model_space, cov_matrix, expectation=mean)


@pytest.fixture
def forward_operator(model_space: HilbertSpace, data_space: HilbertSpace) -> LinearOperator:
    """Provides a forward operator mapping from model space to data space."""
    # Create a random matrix mapping from model space to data space
    matrix = np.random.randn(data_space.dim, model_space.dim)
    return LinearOperator.from_matrix(model_space, data_space, matrix, galerkin=False)


# =============================================================================
# Test Suite for empirical_data_error_measure
# =============================================================================

class TestEmpiricalDataErrorMeasure:
    """
    A suite of tests for the empirical_data_error_measure function.
    """

    def test_returns_gaussian_measure(self, model_measure: GaussianMeasure, 
                                    forward_operator: LinearOperator):
        """Test that the function returns a GaussianMeasure instance."""
        result = empirical_data_error_measure(model_measure, forward_operator)
        assert isinstance(result, GaussianMeasure)
        assert result.domain == forward_operator.codomain

    def test_basic_functionality(self, model_measure: GaussianMeasure,
                                            forward_operator: LinearOperator):
        """Test basic functionality."""
        n_samples = 20
        result = empirical_data_error_measure(
            model_measure, forward_operator, n_samples=n_samples
        )
        
        # Check that result is a valid measure
        assert isinstance(result, GaussianMeasure)
        assert result.domain == forward_operator.codomain
        
        # Check that covariance matrix is positive semidefinite
        cov_matrix = result.covariance.matrix(dense=True, galerkin=True)
        eigenvals = np.linalg.eigvals(cov_matrix)
        assert np.all(eigenvals >= -1e-10), "Covariance matrix should be positive semidefinite"

    def test_mean_centering_behavior(self, model_measure: GaussianMeasure,
                                   forward_operator: LinearOperator):
        """Test that samples are properly mean-centered using empirical sample mean."""
        n_samples = 50
        result = empirical_data_error_measure(
            model_measure, forward_operator, n_samples=n_samples
        )
        
        # Check that result is a valid measure
        assert isinstance(result, GaussianMeasure)
        assert result.domain == forward_operator.codomain
        
        # The resulting measure should have approximately zero mean since we subtract 
        # the empirical sample mean from each sample
        result_mean = result.expectation.data
        expected_zero_mean = np.zeros_like(result_mean)
        
        # Allow some tolerance due to finite sampling effects
        assert np.allclose(result_mean, expected_zero_mean, atol=0.1), \
            "Result should have approximately zero mean due to mean centering"

    def test_scale_factor_effect(self, model_measure: GaussianMeasure,
                               forward_operator: LinearOperator):
        """Test that scale_factor properly scales the covariance."""
        # Test using a deterministic approach by using the same samples
        n_samples = 20
        scale_factor = 2.5
        
        # Fix the random seed to get reproducible samples
        np.random.seed(42)
        result_unit = empirical_data_error_measure(
            model_measure, forward_operator, n_samples=n_samples, scale_factor=1.0
        )
        
        # Use the same seed to get the same samples, but with different scale factor
        np.random.seed(42)
        result_scaled = empirical_data_error_measure(
            model_measure, forward_operator, n_samples=n_samples, scale_factor=scale_factor
        )
        
        # Check that covariances are scaled by scale_factor^2
        cov_unit = result_unit.covariance.matrix(dense=True, galerkin=True)
        cov_scaled = result_scaled.covariance.matrix(dense=True, galerkin=True)
        
        expected_scaled_cov = (scale_factor ** 2) * cov_unit
        assert np.allclose(cov_scaled, expected_scaled_cov, rtol=1e-12), \
            "Scaled covariance should be scale_factor^2 times the original"

    def test_different_sample_counts(self, model_measure: GaussianMeasure,
                                   forward_operator: LinearOperator):
        """Test that different sample counts work and affect statistical estimates."""
        # Test with minimal samples
        result_small = empirical_data_error_measure(
            model_measure, forward_operator, n_samples=2
        )
        assert isinstance(result_small, GaussianMeasure)
        
        # Test with more samples
        result_large = empirical_data_error_measure(
            model_measure, forward_operator, n_samples=100
        )
        assert isinstance(result_large, GaussianMeasure)
        
        # Both should have the same domain
        assert result_small.domain == result_large.domain == forward_operator.codomain

    def test_empirical_vs_analytical_mean_behavior(self, model_measure: GaussianMeasure,
                                                  forward_operator: LinearOperator):
        """Test that empirical mean centering produces approximately zero-mean result."""
        n_samples = 100
        
        # Test that the function produces a valid measure with approximately zero mean
        result = empirical_data_error_measure(
            model_measure, forward_operator, n_samples=n_samples
        )
        
        # The result should be a valid GaussianMeasure
        assert isinstance(result, GaussianMeasure)
        assert result.domain == forward_operator.codomain
        
        # The result should have approximately zero mean due to empirical mean subtraction
        result_mean = result.expectation.data
        expected_zero_mean = np.zeros_like(result_mean)
        
        # Allow reasonable tolerance for finite sampling effects
        assert np.allclose(result_mean, expected_zero_mean, atol=0.2), \
            "Result should have approximately zero mean due to empirical mean subtraction"
        
        # Covariance should be positive semidefinite
        cov_matrix = result.covariance.matrix(dense=True, galerkin=True)
        eigenvals = np.linalg.eigvals(cov_matrix)
        assert np.all(eigenvals >= -1e-10), "Covariance matrix should be positive semidefinite"

    def test_consistent_results_with_fixed_seed(self, model_measure: GaussianMeasure,
                                              forward_operator: LinearOperator):
        """Test that results are consistent when using the same random seed."""
        # Set seeds and get results
        np.random.seed(42)
        result1 = empirical_data_error_measure(
            model_measure, forward_operator, n_samples=20
        )
        
        np.random.seed(42)
        result2 = empirical_data_error_measure(
            model_measure, forward_operator, n_samples=20
        )
        
        # Results should be identical
        cov1 = result1.covariance.matrix(dense=True, galerkin=True)
        cov2 = result2.covariance.matrix(dense=True, galerkin=True)
        assert np.allclose(cov1, cov2), "Results should be identical with same seed"

    def test_approximates_analytical_covariance(self, model_space: HilbertSpace,
                                              data_space: HilbertSpace):
        """Test that empirical covariance approximates analytical covariance with many samples."""
        # Create a simple case where we can compute the analytical result
        # Use identity covariance for the model measure with zero mean
        model_cov = np.eye(model_space.dim)
        model_measure = GaussianMeasure.from_covariance_matrix(
            model_space, model_cov, expectation=model_space.zero
        )
        
        # Use a simple forward operator
        np.random.seed(456)  # Different seed for forward operator matrix
        forward_matrix = np.random.randn(data_space.dim, model_space.dim)
        forward_operator = LinearOperator.from_matrix(
            model_space, data_space, forward_matrix, galerkin=False
        )
        
        # Analytical result: B @ C_model @ B^T where B is forward operator matrix
        # The empirical version should approximate this for large n_samples
        analytical_cov = forward_matrix @ model_cov @ forward_matrix.T
        
        # Empirical result with many samples
        np.random.seed(123)  # For reproducibility
        empirical_result = empirical_data_error_measure(
            model_measure, forward_operator, n_samples=2000
        )
        empirical_cov = empirical_result.covariance.matrix(dense=True, galerkin=True)
        
        # They should be approximately equal (allow for sampling variance and mean subtraction effects)
        assert np.allclose(empirical_cov, analytical_cov, rtol=0.3, atol=0.3), \
            "Empirical covariance should approximate analytical covariance with many samples"

    def test_edge_case_single_sample(self, model_measure: GaussianMeasure,
                                   forward_operator: LinearOperator):
        """Test edge case with a single sample."""
        result = empirical_data_error_measure(
            model_measure, forward_operator, n_samples=1
        )
        
        # Should still return a valid GaussianMeasure
        assert isinstance(result, GaussianMeasure)
        assert result.domain == forward_operator.codomain
        
        # With a single sample, covariance should be zero (no variance)
        cov_matrix = result.covariance.matrix(dense=True, galerkin=True)
        assert np.allclose(cov_matrix, 0, atol=1e-14), \
            "Single sample should give zero covariance"

    def test_different_domain_dimensions(self):
        """Test with different domain and codomain dimensions."""
        # Small model space, larger data space
        model_space = EuclideanSpace(dim=3)
        data_space = EuclideanSpace(dim=8)
        
        model_measure = GaussianMeasure.from_standard_deviations(
            model_space, np.ones(model_space.dim)
        )
        
        forward_matrix = np.random.randn(data_space.dim, model_space.dim)
        forward_operator = LinearOperator.from_matrix(
            model_space, data_space, forward_matrix, galerkin=False
        )
        
        result = empirical_data_error_measure(
            model_measure, forward_operator, n_samples=20
        )
        
        assert isinstance(result, GaussianMeasure)
        assert result.domain.dim == data_space.dim
        
        # Check covariance matrix has the right shape
        cov_matrix = result.covariance.matrix(dense=True, galerkin=True)
        assert cov_matrix.shape == (data_space.dim, data_space.dim)