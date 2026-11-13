"""
A Gaussian mixture fitted to weighted points, by expectation-maximization.

On plain arrays: points in a low-dimensional space, the coordinates of a
chart on a function space say, each with a weight. The number of components
is the caller's. The start is k-means++ seeding, and every covariance
carries a small ridge, relative to the points' own spread, so that a
component collapsing onto a few points stays positive definite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

__all__ = ["MixtureFit", "fit_gaussian_mixture"]


@dataclass(frozen=True)
class MixtureFit:
    """A fitted Gaussian mixture.

    Attributes:
        weights: the components' weights, summing to one.
        means: the components' means, shape ``(components, dimension)``.
        covariances: their covariances, shape
            ``(components, dimension, dimension)``.
        log_likelihood: the weighted mean log-likelihood of the points.
        iterations: the iterations taken.
        converged: whether the log-likelihood settled within the tolerance.
    """

    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    log_likelihood: float
    iterations: int
    converged: bool


def fit_gaussian_mixture(
    points: np.ndarray,
    components: int,
    /,
    *,
    weights: np.ndarray | None = None,
    rng: Generator | None = None,
    iterations: int = 500,
    tolerance: float = 1e-9,
    ridge: float = 1e-6,
) -> MixtureFit:
    """Fit ``components`` Gaussians to weighted points by EM.

    Args:
        points: shape ``(count, dimension)``, or ``(count,)`` in one
            dimension.
        components: how many Gaussians.
        weights: a non-negative weight per point; equal if omitted.
        rng: the generator for the seeding.
        iterations: the most EM iterations.
        tolerance: the change in mean log-likelihood that ends the
            iteration.
        ridge: added to every covariance, as a fraction of the points'
            mean variance.

    Returns:
        The fit.

    Raises:
        ValueError: for fewer points than components, a number of components
            below one, or weights of the wrong shape or all zero.
    """
    from ..algebra.spaces import resolve_rng

    x = np.asarray(points, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    count, dimension = x.shape
    if int(components) != components or components < 1:
        raise ValueError(f"A mixture has at least one component, got {components}.")
    if count < components:
        raise ValueError(f"{count} points cannot fit {components} components.")
    w = np.ones(count) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (count,) or np.any(w < 0.0) or not w.sum() > 0.0:
        raise ValueError("Expected one non-negative weight per point, not all zero.")
    w = w / w.sum()
    generator = resolve_rng(rng)

    centered = x - w @ x
    spread = float(np.sum(w @ centered**2)) / dimension
    floor = ridge * (spread if spread > 0.0 else 1.0) * np.identity(dimension)

    means = _seed(x, w, components, generator)
    distances = ((x[:, np.newaxis, :] - means[np.newaxis]) ** 2).sum(axis=2)
    shares = np.zeros((count, components))
    shares[np.arange(count), distances.argmin(axis=1)] = 1.0

    previous = -np.inf
    log_likelihood = -np.inf
    converged = False
    taken = 0
    for taken in range(1, iterations + 1):
        mass = w @ shares
        mass = np.maximum(mass, np.finfo(float).tiny)
        mixing = mass / mass.sum()
        means = (shares * w[:, np.newaxis]).T @ x / mass[:, np.newaxis]
        covariances = np.empty((components, dimension, dimension))
        for j in range(components):
            deviation = x - means[j]
            covariances[j] = (
                deviation * (w * shares[:, j])[:, np.newaxis]
            ).T @ deviation
            covariances[j] = covariances[j] / mass[j] + floor
        log_terms = np.log(mixing) + _log_normal(x, means, covariances)
        largest = log_terms.max(axis=1, keepdims=True)
        log_total = largest[:, 0] + np.log(np.exp(log_terms - largest).sum(axis=1))
        shares = np.exp(log_terms - log_total[:, np.newaxis])
        log_likelihood = float(w @ log_total)
        if abs(log_likelihood - previous) < tolerance * max(1.0, abs(log_likelihood)):
            converged = True
            break
        previous = log_likelihood
    return MixtureFit(mixing, means, covariances, log_likelihood, taken, converged)


def _seed(x: np.ndarray, w: np.ndarray, components: int, rng: Generator) -> np.ndarray:
    """k-means++ centers: each next one drawn in proportion to its weighted
    squared distance from the nearest already chosen."""
    chosen = [x[rng.choice(x.shape[0], p=w)]]
    for _ in range(components - 1):
        nearest = np.min([((x - center) ** 2).sum(axis=1) for center in chosen], axis=0)
        score = w * nearest
        total = score.sum()
        probabilities = score / total if total > 0.0 else w
        chosen.append(x[rng.choice(x.shape[0], p=probabilities)])
    return np.array(chosen)


def _log_normal(
    x: np.ndarray, means: np.ndarray, covariances: np.ndarray
) -> np.ndarray:
    """``log n(x_i; m_j, S_j)`` for every point and component."""
    count, dimension = x.shape
    result = np.empty((count, means.shape[0]))
    for j, (mean, covariance) in enumerate(zip(means, covariances)):
        factor = np.linalg.cholesky(covariance)
        deviation = np.linalg.solve(factor, (x - mean).T)
        result[:, j] = (
            -0.5 * np.sum(deviation**2, axis=0)
            - np.sum(np.log(np.diag(factor)))
            - 0.5 * dimension * np.log(2.0 * np.pi)
        )
    return result
