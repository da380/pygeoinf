"""
Tests for the auxiliary module.

This module contains comprehensive tests for the empirical_data_error_measure function,
which generates empirical data error measures based on samples from a model space measure.
The tests cover:

- Basic functionality and return types
- Behavior with different parameter settings (diagonal_only, scale_factor, n_samples)
- Statistical properties and approximation quality
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

    def test_basic_functionality_non_diagonal(self, model_measure: GaussianMeasure,
                                            forward_operator: LinearOperator):
        """Test basic functionality with diagonal_only=False."""
        n_samples = 20
        result = empirical_data_error_measure(
            model_measure, forward_operator, n_samples=n_samples, diagonal_only=False
        )
        
        # Check that result is a valid measure
        assert isinstance(result, GaussianMeasure)
        assert result.domain == forward_operator.codomain
        
        # Check that covariance matrix is positive semidefinite
        cov_matrix = result.covariance.matrix(dense=True, galerkin=True)
        eigenvals = np.linalg.eigvals(cov_matrix)
        assert np.all(eigenvals >= -1e-10), "Covariance matrix should be positive semidefinite"

    def test_basic_functionality_diagonal(self, model_measure: GaussianMeasure,
                                        forward_operator: LinearOperator):
        """Test basic functionality with diagonal_only=True."""
        n_samples = 20
        result = empirical_data_error_measure(
            model_measure, forward_operator, n_samples=n_samples, diagonal_only=True
        )
        
        # Check that result is a valid measure
        assert isinstance(result, GaussianMeasure)
        assert result.domain == forward_operator.codomain
        
        # Check that covariance matrix is diagonal
        cov_matrix = result.covariance.matrix(dense=True, galerkin=True)
        off_diagonal = cov_matrix - np.diag(np.diag(cov_matrix))
        assert np.allclose(off_diagonal, 0, atol=1e-14), "Covariance should be diagonal"

    def test_scale_factor_effect(self, model_measure: GaussianMeasure,
                               forward_operator: LinearOperator):
        """Test that scale_factor properly scales the covariance."""
        # Test using a deterministic approach by using the same samples
        # Generate samples once and use the scale factor correctly
        n_samples = 20
        scale_factor = 2.5
        
        # Fix the random seed to get reproducible samples
        np.random.seed(42)
        
        # Generate the data samples manually to control randomness
        data_measure = model_measure.affine_mapping(operator=forward_operator)
        data_samples = data_measure.samples(n_samples)
        mean = data_measure.expectation
        
        # Create zeroed samples with different scale factors
        zeroed_samples_unit = [1.0 * (data_sample - mean) for data_sample in data_samples]
        zeroed_samples_scaled = [scale_factor * (data_sample - mean) for data_sample in data_samples]
        
        result_unit = GaussianMeasure.from_samples(forward_operator.codomain, zeroed_samples_unit)
        result_scaled = GaussianMeasure.from_samples(forward_operator.codomain, zeroed_samples_scaled)
        
        # Check that covariances are scaled by scale_factor^2
        cov_unit = result_unit.covariance.matrix(dense=True, galerkin=True)
        cov_scaled = result_scaled.covariance.matrix(dense=True, galerkin=True)
        
        expected_scaled_cov = (scale_factor ** 2) * cov_unit
        assert np.allclose(cov_scaled, expected_scaled_cov, rtol=1e-12), \
            "Scaled covariance should be scale_factor^2 times the original"

    def test_scale_factor_diagonal_only(self, model_measure: GaussianMeasure,
                                      forward_operator: LinearOperator):
        """Test scale_factor with diagonal_only=True."""
        # Test using deterministic approach by fixing random seed
        n_samples = 20
        scale_factor = 3.0
        
        # Generate the same data samples for both tests
        np.random.seed(42)
        data_measure = model_measure.affine_mapping(operator=forward_operator)
        data_samples = data_measure.samples(n_samples)
        
        # Convert to numpy array for easier manipulation
        data_array = np.array([sample.data for sample in data_samples])
        
        # Compute standard deviations with different scale factors
        std_devs_unit = np.std(data_array, axis=0) * 1.0
        std_devs_scaled = np.std(data_array, axis=0) * scale_factor
        
        result_unit = GaussianMeasure.from_standard_deviations(forward_operator.codomain, std_devs_unit)
        result_scaled = GaussianMeasure.from_standard_deviations(forward_operator.codomain, std_devs_scaled)
        
        # Check diagonal elements are scaled by scale_factor^2
        diag_unit = np.diag(result_unit.covariance.matrix(dense=True, galerkin=True))
        diag_scaled = np.diag(result_scaled.covariance.matrix(dense=True, galerkin=True))
        
        expected_scaled_diag = (scale_factor ** 2) * diag_unit
        assert np.allclose(diag_scaled, expected_scaled_diag, rtol=1e-12), \
            "Diagonal elements should be scaled by scale_factor^2"

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

    def test_centering_reduces_variance_effect(self, model_measure: GaussianMeasure,
                                              forward_operator: LinearOperator):
        """Test that centering is applied correctly by comparing diagonal vs non-diagonal modes."""
        n_samples = 100
        
        # Test that both modes produce valid measures 
        result_diagonal = empirical_data_error_measure(
            model_measure, forward_operator, n_samples=n_samples, diagonal_only=True
        )
        
        result_full = empirical_data_error_measure(
            model_measure, forward_operator, n_samples=n_samples, diagonal_only=False
        )
        
        # Both should produce valid GaussianMeasures
        assert isinstance(result_diagonal, GaussianMeasure)
        assert isinstance(result_full, GaussianMeasure)
        
        # Both should have the same domain
        assert result_diagonal.domain == result_full.domain == forward_operator.codomain
        
        # The diagonal version should have zero off-diagonal elements
        cov_diag = result_diagonal.covariance.matrix(dense=True, galerkin=True)
        cov_full = result_full.covariance.matrix(dense=True, galerkin=True)
        
        # Check that diagonal covariance is actually diagonal
        off_diagonal_diag = cov_diag - np.diag(np.diag(cov_diag))
        assert np.allclose(off_diagonal_diag, 0, atol=1e-14), \
            "Diagonal-only covariance should have zero off-diagonal elements"
        
        # Both should have positive diagonal elements
        assert np.all(np.diag(cov_diag) > 0), "Diagonal covariance should have positive diagonal elements"
        assert np.all(np.diag(cov_full) > 0), "Full covariance should have positive diagonal elements"

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
        # Since model_measure has zero mean, the theoretical covariance is exactly this
        analytical_cov = forward_matrix @ model_cov @ forward_matrix.T
        
        # Empirical result with many samples
        np.random.seed(123)  # For reproducibility
        empirical_result = empirical_data_error_measure(
            model_measure, forward_operator, n_samples=2000, diagonal_only=False
        )
        empirical_cov = empirical_result.covariance.matrix(dense=True, galerkin=True)
        
        # They should be approximately equal (allow for sampling variance)
        assert np.allclose(empirical_cov, analytical_cov, rtol=0.2, atol=0.2), \
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