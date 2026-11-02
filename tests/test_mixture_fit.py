"""The Gaussian mixture fit, against mixtures whose parameters are known."""

import numpy as np
import pytest

from pygeoinf.numerics.mixture_fit import fit_gaussian_mixture


def two_clusters(rng, count=4000):
    first = rng.multivariate_normal([2.0, 0.0], [[0.3, 0.1], [0.1, 0.2]], int(0.75 * count))
    second = rng.multivariate_normal([-1.0, 1.0], [[0.1, 0.0], [0.0, 0.4]], count // 4)
    return np.vstack([first, second])


class TestTheFit:
    def test_it_recovers_a_known_mixture(self, rng):
        fit = fit_gaussian_mixture(two_clusters(rng), 2, rng=rng)
        order = np.argsort(fit.means[:, 0])
        assert fit.converged
        assert fit.weights[order] == pytest.approx([0.25, 0.75], abs=0.02)
        assert np.allclose(fit.means[order], [[-1.0, 1.0], [2.0, 0.0]], atol=0.05)
        assert np.allclose(fit.covariances[order[1]], [[0.3, 0.1], [0.1, 0.2]], atol=0.03)

    def test_one_component_is_the_weighted_moments(self, rng):
        points = rng.standard_normal((500, 3))
        weights = rng.random(500)
        fit = fit_gaussian_mixture(points, 1, weights=weights, rng=rng, ridge=0.0)
        normalized = weights / weights.sum()
        mean = normalized @ points
        assert fit.means[0] == pytest.approx(mean)
        deviation = points - mean
        assert fit.covariances[0] == pytest.approx((deviation * normalized[:, None]).T @ deviation)

    def test_a_weight_is_a_repetition(self, rng):
        points = rng.standard_normal((50, 2))
        doubled = fit_gaussian_mixture(np.vstack([points, points[:10]]), 1, rng=rng)
        weighted = fit_gaussian_mixture(
            points, 1, weights=np.r_[np.full(10, 2.0), np.ones(40)], rng=rng
        )
        assert doubled.means == pytest.approx(weighted.means)
        assert doubled.covariances == pytest.approx(weighted.covariances)

    def test_one_dimensional_points_are_accepted(self, rng):
        points = np.where(rng.random(3000) < 0.5, 1.0, -1.0) + 0.1 * rng.standard_normal(3000)
        fit = fit_gaussian_mixture(points, 2, rng=rng)
        assert np.sort(fit.means[:, 0]) == pytest.approx([-1.0, 1.0], abs=0.02)

    def test_the_ridge_keeps_a_collapsed_component_definite(self, rng):
        points = np.vstack([np.zeros((20, 2)), rng.standard_normal((200, 2))])
        fit = fit_gaussian_mixture(points, 2, rng=rng)
        for covariance in fit.covariances:
            assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


class TestWhatItRefuses:
    def test_more_components_than_points(self):
        with pytest.raises(ValueError, match="cannot fit"):
            fit_gaussian_mixture(np.zeros((2, 1)), 3)

    def test_no_components(self):
        with pytest.raises(ValueError, match="at least one"):
            fit_gaussian_mixture(np.zeros((5, 1)), 0)

    def test_weights_of_the_wrong_shape(self):
        with pytest.raises(ValueError, match="weight"):
            fit_gaussian_mixture(np.zeros((5, 1)), 1, weights=np.ones(4))
