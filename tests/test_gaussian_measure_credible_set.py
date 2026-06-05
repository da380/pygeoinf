"""Tests for Gaussian credible subsets."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import chi2

from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.hilbert_space import (
    EuclideanSpace,
    HilbertSpace,
    MassWeightedHilbertSpace,
)
from pygeoinf.linear_forms import LinearForm
from pygeoinf.low_rank import LowRankEig
from pygeoinf.subsets import Ball, Ellipsoid


class ZeroDimensionalSpace(HilbertSpace):
    """Minimal basis-free-style Hilbert space for rank validation tests."""

    @property
    def dim(self) -> int:
        return 0

    def to_dual(self, x: np.ndarray) -> LinearForm:
        return LinearForm(self, components=np.array([]))

    def from_dual(self, xp: LinearForm) -> np.ndarray:
        return np.array([])

    def to_components(self, x: np.ndarray) -> np.ndarray:
        return x

    def from_components(self, c: np.ndarray) -> np.ndarray:
        return c

    def is_element(self, x) -> bool:
        return isinstance(x, np.ndarray) and x.shape == (0,)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ZeroDimensionalSpace)


def test_credible_set_returns_covariance_ellipsoid():
    """A full-rank Gaussian returns the chi-square Mahalanobis ellipsoid."""
    space = EuclideanSpace(2)
    mean = np.array([1.0, -2.0])
    covariance = np.diag([4.0, 9.0])
    probability = 0.8

    measure = GaussianMeasure.from_covariance_matrix(
        space, covariance, expectation=mean
    )

    credible_set = measure.credible_set(probability)

    radius = np.sqrt(chi2.ppf(probability, df=space.dim))
    assert isinstance(credible_set, Ellipsoid)
    assert_allclose(credible_set.center, mean, rtol=0.0, atol=0.0)
    assert_allclose(credible_set.radius, radius, rtol=1e-12, atol=1e-12)

    boundary_point = mean + np.array([2.0 * radius, 0.0])
    outside_point = mean + np.array([2.0 * radius * 1.01, 0.0])

    assert credible_set.is_element(boundary_point, rtol=1e-10)
    assert not credible_set.is_element(outside_point, rtol=1e-10)


def test_credible_set_support_function_uses_covariance_shape():
    """The returned ellipsoid exposes the covariance-shaped support function."""
    space = EuclideanSpace(2)
    mean = np.array([1.0, -2.0])
    covariance = np.array([[4.0, 1.5], [1.5, 2.25]])
    probability = 0.5

    measure = GaussianMeasure.from_covariance_matrix(
        space, covariance, expectation=mean
    )
    credible_set = measure.credible_set(probability)

    radius = np.sqrt(chi2.ppf(probability, df=space.dim))
    q = np.array([0.3, -1.2])
    expected_support = q @ mean + radius * np.sqrt(q @ covariance @ q)

    assert_allclose(
        credible_set.support_function(q), expected_support, rtol=1e-12, atol=1e-12
    )


def test_credible_set_can_return_cameron_martin_ball():
    """The same confidence region can be returned as a ball in CM geometry."""
    space = EuclideanSpace(2)
    mean = np.array([1.0, -2.0])
    covariance = np.diag([4.0, 9.0])
    probability = 0.9

    measure = GaussianMeasure.from_covariance_matrix(
        space, covariance, expectation=mean
    )
    credible_ball = measure.credible_set(probability, geometry="cameron_martin")

    radius = np.sqrt(chi2.ppf(probability, df=space.dim))
    assert isinstance(credible_ball, Ball)
    assert isinstance(credible_ball.domain, MassWeightedHilbertSpace)
    assert_allclose(credible_ball.radius, radius, rtol=1e-12, atol=1e-12)

    boundary_point = mean + np.array([2.0 * radius, 0.0])
    outside_point = mean + np.array([2.0 * radius * 1.01, 0.0])

    assert credible_ball.is_element(boundary_point, rtol=1e-10)
    assert not credible_ball.is_element(outside_point, rtol=1e-10)


def test_credible_set_accepts_custom_rank():
    """A custom rank changes the chi-square degrees of freedom."""
    space = EuclideanSpace(5)
    measure = GaussianMeasure.from_standard_deviation(space, 1.0)
    probability = 0.95

    credible_set = measure.credible_set(probability, rank=3)

    expected_radius = np.sqrt(chi2.ppf(probability, df=3))
    assert_allclose(credible_set.radius, expected_radius, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("probability", [-0.1, 0.0, 1.0, 1.1])
def test_credible_set_rejects_invalid_probability(probability):
    """Credible probabilities must lie strictly between zero and one."""
    space = EuclideanSpace(2)
    measure = GaussianMeasure.from_standard_deviation(space, 1.0)

    with pytest.raises(ValueError, match="Probability"):
        measure.credible_set(probability)


@pytest.mark.parametrize("rank", [0, -1, 1.5])
def test_credible_set_rejects_invalid_rank(rank):
    """The chi-square rank must be a positive integer."""
    space = EuclideanSpace(2)
    measure = GaussianMeasure.from_standard_deviation(space, 1.0)

    with pytest.raises(ValueError, match="Rank"):
        measure.credible_set(0.8, rank=rank)


def test_credible_set_requires_rank_for_zero_dimensional_domain():
    """Basis-free spaces must provide an effective chi-square rank."""
    space = ZeroDimensionalSpace()
    measure = GaussianMeasure(
        covariance=space.identity_operator(),
        inverse_covariance=space.identity_operator(),
    )

    with pytest.raises(ValueError, match="Rank must be provided"):
        measure.credible_set(0.8)


def test_credible_set_rejects_invalid_geometry():
    """Only ellipsoid and Cameron-Martin ball geometries are supported."""
    space = EuclideanSpace(2)
    measure = GaussianMeasure.from_standard_deviation(space, 1.0)

    with pytest.raises(ValueError, match="Geometry"):
        measure.credible_set(0.8, geometry="cube")


def test_credible_set_requires_precision_operator():
    """A Mahalanobis credible set requires an available inverse covariance."""
    space = EuclideanSpace(2)
    covariance = space.identity_operator()
    measure = GaussianMeasure(covariance=covariance)

    with pytest.raises(AttributeError, match="Inverse covariance"):
        measure.credible_set(0.8)


# ---------------------------------------------------------------------------
# Function-space modes (ambient_ball, weakened_ellipsoid)
# ---------------------------------------------------------------------------


def test_ambient_ball_equal_spectrum_matches_chi2():
    """ambient_ball with eigvals = (sigma^2)*ones matches sigma^2 * chi2_k."""
    space = EuclideanSpace(5)
    sigma2 = 4.0
    measure = GaussianMeasure.from_standard_deviation(space, np.sqrt(sigma2))
    eigvals = np.full(space.dim, sigma2)

    p = 0.9
    ball = measure.credible_set(p, geometry="ambient_ball", spectrum=eigvals)
    expected_radius = np.sqrt(sigma2 * chi2.ppf(p, df=space.dim))
    assert isinstance(ball, Ball)
    assert_allclose(ball.radius, expected_radius, rtol=1e-8)


def test_ambient_ball_anisotropic_matches_mc():
    """Spectral ambient_ball radius matches Monte Carlo over ||X - m||^2."""
    space = EuclideanSpace(3)
    cov_matrix = np.diag([4.0, 1.0, 0.25])
    measure = GaussianMeasure.from_covariance_matrix(space, cov_matrix)

    eigvals = np.diag(cov_matrix)
    p = 0.9
    ball_spectral = measure.credible_set(p, geometry="ambient_ball", spectrum=eigvals)

    # Monte Carlo: draw and quantile.
    samples = measure.samples(20_000)
    norms_sq = np.array([float(np.dot(x, x)) for x in samples])
    expected_r = float(np.sqrt(np.quantile(norms_sq, p)))

    assert_allclose(ball_spectral.radius, expected_r, rtol=3e-2)


def test_ambient_ball_sampling_path():
    """Sampling-based radius agrees with the spectral computation."""
    space = EuclideanSpace(3)
    cov_matrix = np.diag([4.0, 1.0, 0.25])
    measure = GaussianMeasure.from_covariance_matrix(space, cov_matrix)

    eigvals = np.diag(cov_matrix)
    p = 0.85
    ball_spectral = measure.credible_set(p, geometry="ambient_ball", spectrum=eigvals)
    ball_sampling = measure.credible_set(
        p,
        geometry="ambient_ball",
        radius_method="sampling",
        n_samples=20_000,
    )
    assert_allclose(ball_spectral.radius, ball_sampling.radius, rtol=5e-2)


def test_weakened_ellipsoid_requires_theta():
    space = EuclideanSpace(3)
    measure = GaussianMeasure.from_standard_deviation(space, 1.0)
    with pytest.raises(ValueError, match="theta is required"):
        measure.credible_set(
            0.8,
            geometry="weakened_ellipsoid",
            spectrum=np.ones(space.dim),
        )


@pytest.mark.parametrize("theta", [-0.1, 0.0, 1.0, 1.1])
def test_weakened_ellipsoid_rejects_invalid_theta(theta):
    space = EuclideanSpace(3)
    measure = GaussianMeasure.from_standard_deviation(space, 1.0)
    with pytest.raises(ValueError, match="theta"):
        measure.credible_set(
            0.8,
            geometry="weakened_ellipsoid",
            theta=theta,
            spectrum=np.ones(space.dim),
        )


def test_weakened_ellipsoid_returns_ellipsoid_on_finite_space():
    """For finite-rank Euclidean spaces, weakened ellipsoid is constructible."""
    space = EuclideanSpace(4)
    cov_matrix = np.diag([4.0, 2.0, 1.0, 0.5])
    measure = GaussianMeasure.from_covariance_matrix(space, cov_matrix)

    eig = LowRankEig.from_randomized(
        measure.covariance, space.dim, measure=measure, method="fixed"
    )
    p = 0.85
    theta = 0.5
    ell = measure.credible_set(
        p,
        geometry="weakened_ellipsoid",
        theta=theta,
        spectrum=eig,
    )
    assert isinstance(ell, Ellipsoid)
    assert ell.radius > 0


def test_weakened_ellipsoid_spectral_vs_lanczos():
    """Spectral and Lanczos backends agree on the radius and gauge action."""
    space = EuclideanSpace(4)
    rng = np.random.default_rng(13)
    A = rng.standard_normal((space.dim, space.dim))
    cov_matrix = A.T @ A + 0.5 * np.eye(space.dim)
    measure = GaussianMeasure.from_covariance_matrix(space, cov_matrix)

    # Use the same LowRankEig spectrum on both paths so the only difference
    # is how C^{-theta} is applied to vectors.
    eig = LowRankEig.from_randomized(
        measure.covariance, space.dim, measure=measure, method="fixed"
    )
    p = 0.85
    theta = 0.4

    ell_spectral = measure.credible_set(
        p,
        geometry="weakened_ellipsoid",
        theta=theta,
        spectrum=eig,
        fractional_apply="low_rank_eig",
    )
    ell_lanczos = measure.credible_set(
        p,
        geometry="weakened_ellipsoid",
        theta=theta,
        spectrum=eig.eigenvalues,
        fractional_apply="lanczos",
        # --- NEW DYNAMIC API ---
        lanczos_size_estimate=space.dim,
        lanczos_method="fixed",
    )
    # Same eigenvalues, same weights, same quantile method -> identical r_p.
    assert_allclose(ell_spectral.radius, ell_lanczos.radius, rtol=1e-8)

    # Gauge action on a test vector: at k = n Lanczos is exact, so both
    # operators should agree on every vector.
    test_v = rng.standard_normal(space.dim)
    qf_spectral = float(np.dot(test_v, ell_spectral.operator(test_v)))
    qf_lanczos = float(np.dot(test_v, ell_lanczos.operator(test_v)))
    assert_allclose(qf_spectral, qf_lanczos, rtol=1e-6, atol=1e-9)


def test_ambient_ball_no_spectrum_no_sampling_raises():
    """credible_set on non-sampling measure with no spectrum raises."""
    space = EuclideanSpace(3)
    covariance = space.identity_operator()
    measure = GaussianMeasure(covariance=covariance)  # no sampler
    with pytest.raises(ValueError, match="spectrum"):
        measure.credible_set(0.8, geometry="ambient_ball")


def test_ambient_ball_convenience_wrapper():
    space = EuclideanSpace(3)
    measure = GaussianMeasure.from_standard_deviation(space, 2.0)
    ball = measure.ambient_ball(0.9, spectrum=np.full(space.dim, 4.0))
    assert isinstance(ball, Ball)
    expected_radius = np.sqrt(4.0 * chi2.ppf(0.9, df=space.dim))
    assert_allclose(ball.radius, expected_radius, rtol=1e-8)


def test_weakened_ellipsoid_convenience_wrapper():
    space = EuclideanSpace(4)
    cov_matrix = np.diag([4.0, 2.0, 1.0, 0.5])
    measure = GaussianMeasure.from_covariance_matrix(space, cov_matrix)
    eig = LowRankEig.from_randomized(
        measure.covariance, space.dim, measure=measure, method="fixed"
    )
    ell = measure.weakened_ellipsoid(0.85, theta=0.5, spectrum=eig)
    assert isinstance(ell, Ellipsoid)


def test_credible_set_empirical_coverage():
    """Empirical mass inside the set should be near probability."""
    space = EuclideanSpace(5)
    cov_matrix = np.diag([4.0, 2.0, 1.0, 0.5, 0.25])
    measure = GaussianMeasure.from_covariance_matrix(space, cov_matrix)

    eigvals = np.diag(cov_matrix)
    p = 0.9
    ball = measure.credible_set(p, geometry="ambient_ball", spectrum=eigvals)

    n = 5000
    samples = measure.samples(n)
    inside = sum(1 for x in samples if ball.is_element(x, rtol=1e-12))
    p_hat = inside / n
    # Binomial 3-sigma tolerance.
    sigma = np.sqrt(p * (1 - p) / n)
    assert abs(p_hat - p) < 3 * sigma


def test_imhof_method_selectable():
    """quantile_method='ws' returns a slightly different radius than imhof."""
    space = EuclideanSpace(3)
    cov_matrix = np.diag([4.0, 1.0, 0.5])
    measure = GaussianMeasure.from_covariance_matrix(space, cov_matrix)

    eigvals = np.diag(cov_matrix)
    ball_imhof = measure.credible_set(
        0.9, geometry="ambient_ball", spectrum=eigvals, quantile_method="imhof"
    )
    ball_ws = measure.credible_set(
        0.9, geometry="ambient_ball", spectrum=eigvals, quantile_method="ws"
    )
    # WS approximation: within ~10% on this moderately anisotropic spectrum.
    assert abs(ball_imhof.radius - ball_ws.radius) / ball_imhof.radius < 0.1
