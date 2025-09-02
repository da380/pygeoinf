"""
Tests for the gaussian_measure module.
"""

import pytest
import numpy as np
from pygeoinf.hilbert_space import EuclideanSpace, HilbertSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.gaussian_measure import GaussianMeasure

# =============================================================================
# Parametrized Fixtures
# =============================================================================

# Define the different Hilbert space instances we want to test against.
space_implementations = [
    EuclideanSpace(dim=4),
]


@pytest.fixture(params=space_implementations)
def space(request) -> HilbertSpace:
    """Provides parametrized HilbertSpace instances for the tests."""
    return request.param


@pytest.fixture
def measure(space: HilbertSpace) -> GaussianMeasure:
    """Provides a basic GaussianMeasure instance for the tests."""
    # Use a non-zero mean and a simple covariance matrix
    mean = space.random()
    cov_matrix = np.diag(np.random.rand(space.dim) + 0.1)
    # Make it symmetric
    cov_matrix = cov_matrix.T @ cov_matrix
    return GaussianMeasure.from_covariance_matrix(space, cov_matrix, expectation=mean)


# =============================================================================
# Unified Test Suite for GaussianMeasure
# =============================================================================


class TestGaussianMeasure:
    """
    A suite of tests for the GaussianMeasure class that runs against
    multiple different HilbertSpace implementations.
    """

    def test_initialization(self, measure: GaussianMeasure):
        """Tests that a GaussianMeasure object is created successfully."""
        assert measure is not None
        assert measure.covariance is not None
        assert measure.expectation is not None

    def test_sampling_statistics(self, measure: GaussianMeasure):
        """
        Statistically tests the sample mean and covariance.
        """
        if not measure.sample_set:
            pytest.skip("Sampling is not implemented for this measure.")

        n_samples = 5000  # A large number for statistical reliability
        samples = measure.samples(n_samples)

        # Convert samples to their component representations for numpy analysis
        sample_components = np.array([measure.domain.to_components(s) for s in samples])

        # 1. Test the sample mean
        sample_mean = np.mean(sample_components, axis=0)
        expected_mean_components = measure.domain.to_components(measure.expectation)
        # Use a tolerance based on the standard error of the mean
        assert np.allclose(sample_mean, expected_mean_components, atol=0.1)

        # 2. Test the sample covariance
        sample_covariance = np.cov(sample_components.T)
        expected_covariance = measure.covariance.matrix(dense=True, galerkin=True)
        assert np.allclose(sample_covariance, expected_covariance, atol=0.1)

    def test_affine_mapping(self, measure: GaussianMeasure):
        """
        Tests the affine_mapping method for transforming the measure.
        """
        # Create a simple linear operator for the transformation
        transform_matrix = np.random.randn(measure.domain.dim, measure.domain.dim)
        op = LinearOperator.from_matrix(
            measure.domain, measure.domain, transform_matrix, galerkin=True
        )
        # Create a simple translation vector
        translation = measure.domain.random()

        # Apply the affine mapping
        transformed_measure = measure.affine_mapping(
            operator=op, translation=translation
        )

        # 1. Test the transformed expectation: E[Ax + b] = A*E[x] + b
        expected_mean = op(measure.expectation) + translation
        assert np.allclose(
            measure.domain.to_components(transformed_measure.expectation),
            measure.domain.to_components(expected_mean),
        )

        # 2. Test the transformed covariance: Cov[Ax + b] = A*Cov[x]*A^T
        C = measure.covariance.matrix(dense=True, galerkin=True)
        A = transform_matrix
        expected_covariance = A @ C @ A.T
        actual_covariance = transformed_measure.covariance.matrix(
            dense=True, galerkin=True
        )
        assert np.allclose(actual_covariance, expected_covariance)

    def test_addition(self, measure: GaussianMeasure):
        """Tests the addition of two independent Gaussian measures."""
        # Create a second measure to add to the first one.
        std_devs = np.random.rand(measure.domain.dim) + 0.1
        measure2 = GaussianMeasure.from_standard_deviations(measure.domain, std_devs)

        sum_measure = measure + measure2

        # 1. Test the expectation
        expected_mean = measure.domain.add(measure.expectation, measure2.expectation)
        assert np.allclose(
            measure.domain.to_components(sum_measure.expectation),
            measure.domain.to_components(expected_mean),
        )

        # 2. Test the covariance
        expected_cov = measure.covariance.matrix(
            dense=True, galerkin=True
        ) + measure2.covariance.matrix(dense=True, galerkin=True)
        actual_cov = sum_measure.covariance.matrix(dense=True, galerkin=True)
        assert np.allclose(actual_cov, expected_cov)

    def test_from_standard_deviations(self, space: HilbertSpace):
        """Tests the from_standard_deviations factory method."""
        std_devs = np.random.rand(space.dim) + 0.1
        measure = GaussianMeasure.from_standard_deviations(space, std_devs)

        expected_cov = np.diag(std_devs**2)
        actual_cov = measure.covariance.matrix(dense=True, galerkin=True)

        assert np.allclose(actual_cov, expected_cov)
