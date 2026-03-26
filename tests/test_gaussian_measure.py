"""
Tests for the gaussian_measure module.
"""

import pytest
import numpy as np
from scipy.stats._multivariate import multivariate_normal_frozen
from pygeoinf.hilbert_space import EuclideanSpace, HilbertSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.affine_operators import AffineOperator

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
    # Make it symmetric and positive definite
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
        assert measure is not None
        assert measure.covariance is not None
        assert measure.expectation is not None

    def test_sampling_statistics(self, measure: GaussianMeasure):
        if not measure.sample_set:
            pytest.skip("Sampling is not implemented for this measure.")
        n_samples = 5000
        samples = measure.samples(n_samples)
        sample_components = np.array([measure.domain.to_components(s) for s in samples])
        sample_mean = np.mean(sample_components, axis=0)
        expected_mean_components = measure.domain.to_components(measure.expectation)
        assert np.allclose(sample_mean, expected_mean_components, atol=0.1)
        sample_covariance = np.cov(sample_components.T)
        expected_covariance = measure.covariance.matrix(dense=True, galerkin=True)
        assert np.allclose(sample_covariance, expected_covariance, atol=0.1)

    def test_affine_mapping(self, measure: GaussianMeasure):
        transform_matrix = np.random.randn(measure.domain.dim, measure.domain.dim)
        op = LinearOperator.from_matrix(
            measure.domain, measure.domain, transform_matrix, galerkin=True
        )
        translation = measure.domain.random()
        transformed_measure = measure.affine_mapping(
            operator=op, translation=translation
        )
        expected_mean = op(measure.expectation) + translation
        assert np.allclose(
            measure.domain.to_components(transformed_measure.expectation),
            measure.domain.to_components(expected_mean),
        )
        C = measure.covariance.matrix(dense=True, galerkin=True)
        A = transform_matrix
        expected_covariance = A @ C @ A.T
        actual_covariance = transformed_measure.covariance.matrix(
            dense=True, galerkin=True
        )
        assert np.allclose(actual_covariance, expected_covariance)

    def test_addition(self, measure: GaussianMeasure):
        std_devs = np.random.rand(measure.domain.dim) + 0.1
        measure2 = GaussianMeasure.from_standard_deviations(measure.domain, std_devs)
        sum_measure = measure + measure2
        expected_mean = measure.domain.add(measure.expectation, measure2.expectation)
        assert np.allclose(
            measure.domain.to_components(sum_measure.expectation),
            measure.domain.to_components(expected_mean),
        )
        expected_cov = measure.covariance.matrix(
            dense=True, galerkin=True
        ) + measure2.covariance.matrix(dense=True, galerkin=True)
        actual_cov = sum_measure.covariance.matrix(dense=True, galerkin=True)
        assert np.allclose(actual_cov, expected_cov)

    def test_from_standard_deviations(self, space: HilbertSpace):
        std_devs = np.random.rand(space.dim) + 0.1
        measure = GaussianMeasure.from_standard_deviations(space, std_devs)
        expected_cov = np.diag(std_devs**2)
        actual_cov = measure.covariance.matrix(dense=True, galerkin=True)
        assert np.allclose(actual_cov, expected_cov)

    def test_as_multivariate_normal_basic(self, measure: GaussianMeasure):
        mvn = measure.as_multivariate_normal()
        assert isinstance(mvn, multivariate_normal_frozen)
        assert np.allclose(mvn.mean, measure.domain.to_components(measure.expectation))
        assert np.allclose(
            mvn.cov, measure.covariance.matrix(dense=True, galerkin=True)
        )

    def test_as_multivariate_normal_parallel(self, measure: GaussianMeasure):
        mvn_serial = measure.as_multivariate_normal(parallel=False)
        mvn_parallel = measure.as_multivariate_normal(parallel=True, n_jobs=-1)
        assert np.allclose(mvn_serial.mean, mvn_parallel.mean)
        assert np.allclose(mvn_serial.cov, mvn_parallel.cov)

    def test_from_covariance_matrix_robustness(self, space: HilbertSpace):
        """
        Tests that the constructor correctly handles and warns for a matrix
        with small negative eigenvalues.
        """
        dim = space.dim
        # Deterministically create a matrix with a small negative eigenvalue
        U, _ = np.linalg.qr(np.random.randn(dim, dim))
        eigenvalues = np.random.rand(dim) + 0.1
        eigenvalues[-1] = -1e-12
        dirty_cov_matrix = U @ np.diag(eigenvalues) @ U.T

        with pytest.warns(UserWarning, match="Clipping them to zero"):
            measure = GaussianMeasure.from_covariance_matrix(space, dirty_cov_matrix)

        # Check that the resulting measure's covariance is now clean
        clean_matrix = measure.covariance.matrix(dense=True, galerkin=True)
        assert np.all(np.linalg.eigvalsh(clean_matrix) >= -1e-9)

    def test_as_multivariate_normal_fallback_robustness(self, space: HilbertSpace):
        """
        Tests the eigenvalue cleaning fallback in as_multivariate_normal
        by using a deterministically crafted numerically unstable matrix.
        """
        dim = space.dim
        # Deterministically create a matrix with a small negative eigenvalue
        U, _ = np.linalg.qr(np.random.randn(dim, dim))
        eigenvalues = np.random.rand(dim) + 0.1
        eigenvalues[-1] = -1e-8
        dirty_cov_matrix = U @ np.diag(eigenvalues) @ U.T

        # Manually create a LinearOperator to bypass the cleaning in the factory.
        cov_op = LinearOperator.from_matrix(
            space, space, dirty_cov_matrix, galerkin=True
        )

        # Initialize the measure directly, avoiding the cleaning factory.
        measure = GaussianMeasure(covariance=cov_op, expectation=space.zero)

        with pytest.warns(UserWarning, match="Setting negative eigenvalues to zero"):
            mvn = measure.as_multivariate_normal()

        assert isinstance(mvn, multivariate_normal_frozen)
        assert np.all(np.linalg.eigvalsh(mvn.cov) >= -1e-9)

    def test_samples_parallel_consistency(self, measure: GaussianMeasure):
        """
        Verifies that parallel sampling returns the same number of samples
        and consistent vector types as serial sampling.
        """
        n_samples = 10
        # Draw samples using the new parallel parameters
        samples_parallel = measure.samples(n_samples, parallel=True, n_jobs=2)

        assert len(samples_parallel) == n_samples
        for s in samples_parallel:
            assert measure.domain.is_element(s)

    def test_parallel_sampling_statistics(self, measure: GaussianMeasure):
        """
        Statistical test to ensure parallel sampling correctly represents
        the measure's mean and covariance.
        """
        if not measure.sample_set:
            pytest.skip("Sampling is not implemented for this measure.")

        n_samples = 5000
        # Use parallel execution for the statistical check
        samples = measure.samples(n_samples, parallel=True, n_jobs=-1)

        sample_components = np.array([measure.domain.to_components(s) for s in samples])

        # Verify Mean
        sample_mean = np.mean(sample_components, axis=0)
        expected_mean_components = measure.domain.to_components(measure.expectation)
        assert np.allclose(sample_mean, expected_mean_components, atol=0.1)

        # Verify Covariance
        sample_covariance = np.cov(sample_components.T)
        expected_covariance = measure.covariance.matrix(dense=True, galerkin=True)
        assert np.allclose(sample_covariance, expected_covariance, atol=0.1)

    def test_affine_mapping_with_affine_operator(self, measure: GaussianMeasure):
        """
        Tests that passing an AffineOperator yields the exact same
        transformation as passing the linear part and translation separately,
        and verifies that mixing the APIs raises an error.
        """
        # 1. Setup the transformation
        transform_matrix = np.random.randn(measure.domain.dim, measure.domain.dim)
        op = LinearOperator.from_matrix(
            measure.domain, measure.domain, transform_matrix, galerkin=True
        )
        translation = measure.domain.random()
        affine_op = AffineOperator(op, translation)

        # 2. Transform the measure using the new argument
        transformed_measure = measure.affine_mapping(affine_operator=affine_op)

        # 3. Verify Expectation: mu_y = A(mu) + b
        expected_mean = op(measure.expectation) + translation
        assert np.allclose(
            measure.domain.to_components(transformed_measure.expectation),
            measure.domain.to_components(expected_mean),
        )

        # 4. Verify Covariance: C_y = A @ C @ A.T
        C = measure.covariance.matrix(dense=True, galerkin=True)
        A = transform_matrix
        expected_covariance = A @ C @ A.T
        actual_covariance = transformed_measure.covariance.matrix(
            dense=True, galerkin=True
        )
        assert np.allclose(actual_covariance, expected_covariance)

        # 5. Verify the exclusivity check raises a ValueError
        error_msg = "Cannot provide `affine_operator` alongside"

        with pytest.raises(ValueError, match=error_msg):
            measure.affine_mapping(affine_operator=affine_op, operator=op)

        with pytest.raises(ValueError, match=error_msg):
            measure.affine_mapping(affine_operator=affine_op, translation=translation)

    def test_kl_divergence_self(self, measure: GaussianMeasure):
        """
        The KL divergence of a measure with itself must be exactly zero.
        """
        kl_div = measure.kl_divergence(measure)
        assert np.isclose(kl_div, 0.0, atol=1e-10)

    def test_kl_divergence_known_values(self, space: HilbertSpace):
        """
        Tests the KL divergence calculation against a known analytical result.
        P = N(0, I)
        Q = N(mu, sigma^2 * I)
        """
        k = space.dim

        # Measure P: Standard Normal N(0, I)
        mean_p = space.zero
        cov_p = np.eye(k)
        measure_p = GaussianMeasure.from_covariance_matrix(
            space, cov_p, expectation=mean_p
        )

        # Measure Q: N(1, 2 * I)
        sigma_sq = 2.0
        mean_q_components = np.ones(k)
        mean_q = space.from_components(mean_q_components)
        cov_q = sigma_sq * np.eye(k)
        measure_q = GaussianMeasure.from_covariance_matrix(
            space, cov_q, expectation=mean_q
        )

        # Analytical Calculation for D_KL(P || Q)
        # 1. Trace term: tr( (sigma^2 I)^-1 * I ) = k / sigma^2
        trace_term = k / sigma_sq
        # 2. Mahalanobis term: (0 - mu)^T (sigma^2 I)^-1 (0 - mu) = k / sigma^2
        mahalanobis_term = k / sigma_sq
        # 3. Log det term: ln(det(Q)/det(P)) = ln((sigma^2)^k / 1) = k * ln(sigma^2)
        log_det_term = k * np.log(sigma_sq)

        expected_kl = 0.5 * (trace_term + mahalanobis_term - k + log_det_term)

        actual_kl = measure_p.kl_divergence(measure_q)
        assert np.isclose(actual_kl, expected_kl, rtol=1e-7)

    def test_kl_divergence_domain_mismatch(self, measure: GaussianMeasure):
        """
        Verifies that computing KL divergence between measures on different
        domains raises a ValueError.
        """
        other_space = EuclideanSpace(dim=measure.domain.dim + 1)
        other_measure = GaussianMeasure.from_standard_deviation(other_space, 1.0)

        with pytest.raises(
            ValueError, match="Measures must be defined on the same domain"
        ):
            measure.kl_divergence(other_measure)
