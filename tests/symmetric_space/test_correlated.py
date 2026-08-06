"""
Tests for correlated tuples of invariant Gaussian measures on symmetric spaces.

Intended location: tests/symmetric_space/test_correlated_invariant_measure.py
"""

import pytest
import numpy as np

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.direct_sum import HilbertSpaceDirectSum
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.forward_problem import LinearForwardProblem
from pygeoinf.linear_bayesian import LinearBayesianInversion
from pygeoinf.linear_solvers import CholeskySolver

from pygeoinf.symmetric_space.circle import Lebesgue, Sobolev
from pygeoinf.symmetric_space import (
    InvariantLinearAutomorphism,
    InvariantGaussianMeasure,
    CorrelatedInvariantGaussianMeasure,
)


def correlation_profile(lam: float) -> float:
    """A scale-dependent correlation profile used throughout the tests."""
    return 0.9 * np.exp(-lam / 50.0)


def eigenvalues(space) -> np.ndarray:
    """Returns the Laplacian eigenvalues of a space as an array."""
    return np.array([space.laplacian_eigenvalue(k) for k in space.indices])


@pytest.fixture
def space() -> Lebesgue:
    """Provides a simple symmetric space for testing correlated measures."""
    return Lebesgue(8, radius=1.0)


@pytest.fixture
def sobolev_space() -> Sobolev:
    """Provides a space with non-trivial metric values."""
    return Sobolev(16, 2.0, 0.1)


@pytest.fixture
def measure(space: Lebesgue) -> CorrelatedInvariantGaussianMeasure:
    """A correlated pair of fields with distinct marginal spectra."""
    f1 = space.sobolev_kernel(2.0, 0.2)
    f2 = space.sobolev_kernel(3.0, 0.1)
    return space.correlated_invariant_gaussian_measure([f1, f2], correlation_profile)


class TestConstructionAndValidation:
    """Tests instantiation, the accepted correlation forms, and validation."""

    def test_instantiation_and_domain(self, space: Lebesgue, measure):
        assert isinstance(measure, CorrelatedInvariantGaussianMeasure)
        assert isinstance(measure, GaussianMeasure)
        assert measure.number_of_fields == 2
        assert measure.field_space == space
        assert measure.has_zero_expectation

        # The domain is the direct sum of two copies of the field space
        assert isinstance(measure.domain, HilbertSpaceDirectSum)
        assert measure.domain.number_of_subspaces == 2
        assert all(subspace == space for subspace in measure.domain.subspaces)

    def test_spectra_and_correlations(self, space: Lebesgue, measure):
        lam = eigenvalues(space)
        f1 = space.sobolev_kernel(2.0, 0.2)
        f2 = space.sobolev_kernel(3.0, 0.1)

        sigma = measure.spectral_cross_covariances
        assert sigma.shape == (space.dim, 2, 2)
        assert np.allclose(sigma, np.transpose(sigma, (0, 2, 1)))

        # Marginal spectra are unchanged, and the cross spectrum is
        # rho * sqrt(sigma_1 sigma_2)
        assert np.allclose(sigma[:, 0, 0], f1(lam))
        assert np.allclose(sigma[:, 1, 1], f2(lam))
        assert np.allclose(
            sigma[:, 0, 1], correlation_profile(lam) * np.sqrt(f1(lam) * f2(lam))
        )
        assert np.allclose(
            measure.spectral_correlations(0, 1), correlation_profile(lam)
        )

    def test_correlation_input_forms_agree(self, space: Lebesgue):
        """A scalar, a matrix, and a per-index array must all agree."""
        f = space.sobolev_kernel(2.0, 0.2)

        m_scalar = space.correlated_invariant_gaussian_measure(f, 0.5)
        m_matrix = space.correlated_invariant_gaussian_measure(
            f, np.array([[1.0, 0.5], [0.5, 1.0]])
        )
        m_array = CorrelatedInvariantGaussianMeasure.from_invariant_measures(
            [space.invariant_gaussian_measure(f)] * 2,
            np.full(space.dim, 0.5),
        )

        assert np.allclose(
            m_scalar.spectral_cross_covariances,
            m_matrix.spectral_cross_covariances,
        )
        assert np.allclose(
            m_scalar.spectral_cross_covariances,
            m_array.spectral_cross_covariances,
        )

    def test_from_function_with_matrix_values(self, space: Lebesgue):
        f1 = space.sobolev_kernel(2.0, 0.2)
        f2 = space.sobolev_kernel(3.0, 0.1)

        def cross_covariance_function(lam):
            rho = 0.5 * np.exp(-lam / 20.0)
            c = rho * np.sqrt(f1(lam) * f2(lam))
            return np.array([[f1(lam), c], [c, f2(lam)]])

        m = CorrelatedInvariantGaussianMeasure.from_function(
            space, cross_covariance_function
        )
        lam = eigenvalues(space)
        assert np.allclose(m.spectral_correlations(0, 1), 0.5 * np.exp(-lam / 20.0))

    def test_invalid_correlations_are_rejected(self, space: Lebesgue):
        f = space.sobolev_kernel(2.0, 0.2)

        with pytest.raises(ValueError):
            space.correlated_invariant_gaussian_measure(f, 1.2)

        with pytest.raises(ValueError):
            space.correlated_invariant_gaussian_measure(
                f, np.array([[0.9, 1.0], [1.0, 0.9]])
            )

        # Equicorrelation of n fields requires rho >= -1 / (n - 1)
        with pytest.raises(ValueError):
            space.correlated_invariant_gaussian_measure([f, f, f], -0.7)
        m3 = space.correlated_invariant_gaussian_measure([f, f, f], -0.4)
        assert m3.number_of_fields == 3

    def test_perfect_correlation_is_degenerate_but_valid(self, space: Lebesgue):
        f = space.sobolev_kernel(2.0, 0.2)
        m = space.correlated_invariant_gaussian_measure(f, 1.0)

        # The joint covariance is singular, so no precision is set
        assert m.inverse_covariance_set is False
        with pytest.raises(AttributeError):
            _ = m.inverse_covariance

        # Sampling still works, and returns identical fields
        u, v = m.sample()
        assert np.allclose(space.to_components(u), space.to_components(v))

    def test_expectation_passthrough(self, space: Lebesgue):
        f = space.sobolev_kernel(2.0, 0.2)
        expectation = [space.random(), space.random()]
        m = space.correlated_invariant_gaussian_measure(f, 0.3, expectation=expectation)

        assert not m.has_zero_expectation
        assert np.allclose(
            space.to_components(m.marginal(0).expectation),
            space.to_components(expectation[0]),
        )


class TestCovarianceStructure:
    """Tests the block operators attached to the joint measure."""

    def test_block_structure(self, space: Lebesgue, measure):
        block = measure.covariance.block(0, 1)
        assert isinstance(block, InvariantLinearAutomorphism)

        x = space.random()
        assert np.allclose(
            space.to_components(block(x)),
            space.to_components(measure.cross_covariance(0, 1)(x)),
        )

    def test_self_adjoint_and_positive(self, measure):
        domain = measure.domain
        x = domain.random()
        y = domain.random()

        Cx = measure.covariance(x)
        assert np.isclose(
            domain.inner_product(Cx, y),
            domain.inner_product(x, measure.covariance(y)),
        )
        assert domain.inner_product(Cx, x) >= 0

    def test_inverse_covariance_inverts_covariance(self, measure):
        domain = measure.domain
        x = domain.random()
        y = measure.inverse_covariance(measure.covariance(x))
        assert np.allclose(domain.to_components(y), domain.to_components(x))

    def test_marginals(self, measure):
        m0 = measure.marginal(0)
        assert isinstance(m0, InvariantGaussianMeasure)
        assert np.allclose(
            m0.spectral_variances, measure.spectral_cross_covariances[:, 0, 0]
        )
        _ = m0.sample()


class TestSampling:
    """Statistical checks on the joint sampler."""

    def test_sample_statistics(self, space: Lebesgue, measure):
        np.random.seed(42)
        lam = eigenvalues(space)
        metric = space.metric_values
        sigma = measure.spectral_cross_covariances

        samples = measure.samples(3000)
        u = np.array([space.to_components(s[0]) for s in samples])
        v = np.array([space.to_components(s[1]) for s in samples])

        # Component variances scale as sigma_ii(k) / m_k
        assert np.allclose(u.var(axis=0), sigma[:, 0, 0] / metric, rtol=0.2)
        assert np.allclose(v.var(axis=0), sigma[:, 1, 1] / metric, rtol=0.2)

        # Per-index correlations follow the prescribed profile
        covariance_uv = (u * v).mean(axis=0) - u.mean(axis=0) * v.mean(axis=0)
        correlations = covariance_uv / np.sqrt(u.var(axis=0) * v.var(axis=0))
        assert np.allclose(correlations, correlation_profile(lam), atol=0.08)

    def test_sample_statistics_with_metric_weighting(self, sobolev_space):
        """The same checks on a space with non-trivial metric values."""
        np.random.seed(42)
        space = sobolev_space
        f = space.sobolev_kernel(2.0, 0.2)
        m = space.correlated_invariant_gaussian_measure(f, correlation_profile)

        lam = eigenvalues(space)
        metric = space.metric_values
        sigma = m.spectral_cross_covariances

        samples = m.samples(2500)
        u = np.array([space.to_components(s[0]) for s in samples])
        v = np.array([space.to_components(s[1]) for s in samples])

        assert np.allclose(u.var(axis=0), sigma[:, 0, 0] / metric, rtol=0.2)
        covariance_uv = (u * v).mean(axis=0) - u.mean(axis=0) * v.mean(axis=0)
        correlations = covariance_uv / np.sqrt(u.var(axis=0) * v.var(axis=0))
        assert np.allclose(correlations, correlation_profile(lam), atol=0.08)


class TestKLDivergenceAndNorms:
    """Tests the spectral fast paths against dense reference computations."""

    def test_kl_divergence_fast_vs_dense(self, space: Lebesgue):
        f1 = space.sobolev_kernel(2.0, 0.2)
        f2 = space.sobolev_kernel(2.5, 0.15)

        measure_p = space.correlated_invariant_gaussian_measure(
            [f1, f2],
            lambda lam: 0.7 * np.exp(-lam / 30.0),
            expectation=[space.random(), space.random()],
        )
        measure_q = space.correlated_invariant_gaussian_measure([f1, f2], 0.3)

        # 1. Fast O(dim * n^3) spectral path
        kl_fast = measure_p.kl_divergence(measure_q)

        # 2. Force the dense path by downcasting Q to a base GaussianMeasure
        measure_q_base = GaussianMeasure(
            covariance=measure_q.covariance, expectation=measure_q.expectation
        )
        kl_slow = measure_p.kl_divergence(measure_q_base)

        assert np.isclose(kl_fast, kl_slow, rtol=1e-8)
        assert np.isclose(measure_p.kl_divergence(measure_p), 0.0, atol=1e-10)

    def test_norms_match_dense_reference(self, space: Lebesgue):
        m = space.correlated_invariant_gaussian_measure(
            space.sobolev_kernel(2.0, 0.2), 0.3
        )
        dense = m.covariance.matrix(dense=True)
        assert np.isclose(m.nuclear_norm(), np.trace(dense))
        assert np.isclose(m.hilbert_schmidt_norm(), np.linalg.norm(dense))


class TestAlgebra:
    """Tests the type preservation and scaling of measure arithmetic."""

    def test_algebra_type_preservation(self, space: Lebesgue):
        f = space.sobolev_kernel(2.0, 0.2)
        m1 = space.correlated_invariant_gaussian_measure(f, 0.5)
        m2 = space.correlated_invariant_gaussian_measure(f, -0.2)

        for m in (m1 + m2, m1 - m2, 2.0 * m1, m1 * 2.0, -m1, m1 / 2.0):
            assert isinstance(m, CorrelatedInvariantGaussianMeasure)

        # Scalars scale the spectra quadratically, sums add them
        assert np.allclose(
            (2.0 * m1).spectral_cross_covariances,
            4.0 * m1.spectral_cross_covariances,
        )
        assert np.allclose(
            (m1 + m2).spectral_cross_covariances,
            m1.spectral_cross_covariances + m2.spectral_cross_covariances,
        )

    def test_zero_expectation_type_preservation(self, space: Lebesgue):
        f = space.sobolev_kernel(2.0, 0.2)
        m = space.correlated_invariant_gaussian_measure(
            f, 0.5, expectation=[space.random(), space.random()]
        )

        assert not m.has_zero_expectation
        m_zeroed = m.zero_expectation()
        assert isinstance(m_zeroed, CorrelatedInvariantGaussianMeasure)
        assert m_zeroed.has_zero_expectation

    def test_rescale_norm_variance(self, space: Lebesgue):
        f = space.sobolev_kernel(2.0, 0.2)
        m = space.correlated_invariant_gaussian_measure(f, 0.5)
        m_scaled = m.rescale_norm_variance(2.0)

        assert isinstance(m_scaled, CorrelatedInvariantGaussianMeasure)
        assert np.isclose(m_scaled.nuclear_norm(), 4.0)

        # E[||x||^2] = std^2 = 4.0 on the direct sum
        np.random.seed(42)
        samples = m_scaled.samples(2000)
        mean_sq_norm = np.mean([m_scaled.domain.norm(s) ** 2 for s in samples])
        assert np.isclose(mean_sq_norm, 4.0, rtol=0.1)


class TestBayesianConditioning:
    """
    Tests that observations of one field update the others through the
    cross-covariance, using only existing library machinery on the joint
    measure.
    """

    def test_cross_field_updating(self, sobolev_space):
        np.random.seed(11)
        space = sobolev_space
        f = space.sobolev_kernel(2.0, 0.1)
        prior = space.correlated_invariant_gaussian_measure(f, correlation_profile)

        # Synthetic truth, observed through point values of the first field
        u_true, v_true = prior.sample()
        points = space.random_points(10)
        forward_operator = space.point_evaluation_operator(
            points
        ) @ prior.domain.subspace_projection(0)
        data_error_measure = GaussianMeasure.from_standard_deviation(
            EuclideanSpace(len(points)), 0.01
        )
        problem = LinearForwardProblem(
            forward_operator, data_error_measure=data_error_measure
        )
        data = problem.synthetic_data([u_true, v_true])

        posterior = LinearBayesianInversion(problem, prior).model_posterior_measure(
            data, CholeskySolver()
        )
        u_posterior, v_posterior = posterior.expectation

        def correlation(a, b):
            ca = space.to_components(a) * np.sqrt(space.metric_values)
            cb = space.to_components(b) * np.sqrt(space.metric_values)
            return ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb))

        # The observed field is recovered, and the unobserved field is
        # informed through the cross-covariance
        assert correlation(u_posterior, u_true) > 0.5
        assert correlation(v_posterior, v_true) > 0.3

        # Posterior sampling on the joint space works out of the box
        _ = posterior.sample()

        # Control: with an uncorrelated prior the unobserved field's
        # posterior mean must vanish identically
        prior_independent = space.correlated_invariant_gaussian_measure(f, 0.0)
        posterior_independent = LinearBayesianInversion(
            problem, prior_independent
        ).model_posterior_measure(data, CholeskySolver())
        _, v_independent = posterior_independent.expectation
        assert np.allclose(space.to_components(v_independent), 0.0, atol=1e-10)
