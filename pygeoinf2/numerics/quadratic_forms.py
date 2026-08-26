"""
The distribution of a quadratic form in Gaussian variables.

``sum_i w_i Z_i^2`` with independent standard normals is a *weighted* chi-square
and has no closed form. It comes up wherever a statement is made about the size
of a Gaussian vector in a metric other than its own: the Mahalanobis form is an
ordinary chi-square because the weights are all one, but the plain squared norm
of a field is not, and its weights are the covariance's eigenvalues.

Imhof's method is exact to the tolerance asked for — it inverts the
characteristic function numerically — and a moment-matched chi-square is the
cheap approximation to fall back on.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import chi2

__all__ = ["weighted_chi2_cdf", "weighted_chi2_quantile"]


def _validate(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("At least one weight is needed.")
    if np.any(values < 0.0):
        raise ValueError("The weights must be non-negative.")
    live = values[values > 0.0]
    if live.size == 0:
        raise ValueError("At least one weight must be positive.")
    return live


def _imhof(weights: np.ndarray, value: float, /, *, tolerance: float) -> float:
    """Imhof's inversion of the characteristic function."""

    def integrand(variable: float) -> float:
        theta = 0.5 * (np.sum(np.arctan(weights * variable)) - value * variable)
        modulus = np.prod((1.0 + (weights * variable) ** 2) ** 0.25)
        return np.sin(theta) / (variable * modulus)

    # Split at one. The integrand oscillates and decays like a power of the
    # variable, and a single call over the half-line spends its subdivisions
    # near the origin and gives up on the tail -- which is where a single
    # weight, the slowest-decaying case, keeps most of its mass.
    near, _ = quad(integrand, 0.0, 1.0, epsabs=tolerance, epsrel=tolerance, limit=400)
    far, _ = quad(integrand, 1.0, np.inf, epsabs=tolerance, epsrel=tolerance, limit=800)
    return float(np.clip(0.5 - (near + far) / np.pi, 0.0, 1.0))


def _matched(weights: np.ndarray, value: float) -> float:
    """A chi-square matched on its first two cumulants (Satterthwaite-Welch)."""
    first = float(np.sum(weights))
    second = float(np.sum(weights**2))
    scale = second / first
    degrees = first**2 / second
    return float(chi2.cdf(value / scale, degrees))


def weighted_chi2_cdf(
    weights: np.ndarray,
    value: float,
    /,
    *,
    method: str = "auto",
    tolerance: float = 1e-10,
    samples: int = 200_000,
    rng: Generator | None = None,
) -> float:
    """``P(sum_i w_i Z_i^2 <= value)`` for independent standard normals.

    Args:
        weights: the coefficients, non-negative and not all zero.
        value: the threshold.
        method: ``"imhof"`` inverts the characteristic function and is exact to
            ``tolerance``; ``"matched"`` fits a chi-square on two cumulants and
            is fast and rough; ``"monte_carlo"`` samples. ``"auto"`` takes
            Imhof, falling back to the matched form if the integral misbehaves.
        tolerance: for Imhof.
        samples: for Monte Carlo.
        rng: for Monte Carlo.
    """
    live = _validate(weights)
    if value <= 0.0:
        return 0.0
    # Equal weights are an ordinary chi-square, exactly -- and that is the
    # commonest case, since a Mahalanobis form has all its weights one. It is
    # also the case Imhof handles worst: with one or two terms the integrand
    # decays slowest and the quadrature is only good to about 1e-3 in the tail.
    if np.allclose(live, live[0]):
        return float(chi2.cdf(value / live[0], live.size))
    if method == "matched":
        return _matched(live, value)
    if method == "monte_carlo":
        generator = np.random.default_rng() if rng is None else rng
        draws = generator.standard_normal((samples, live.size))
        return float(np.mean(draws**2 @ live <= value))
    if method in ("imhof", "auto"):
        try:
            return _imhof(live, value, tolerance=tolerance)
        except Exception:  # pragma: no cover - quadrature failure
            if method == "imhof":
                raise
            return _matched(live, value)
    raise ValueError(f"Unknown method {method!r}.")


def weighted_chi2_quantile(
    weights: np.ndarray,
    probability: float,
    /,
    *,
    method: str = "auto",
    tolerance: float = 1e-10,
) -> float:
    """The ``probability`` quantile of ``sum_i w_i Z_i^2``.

    A root find on :func:`weighted_chi2_cdf`, bracketed from the moment-matched
    approximation — which is close enough to bracket and never close enough to
    trust on its own.
    """
    if not 0.0 < probability < 1.0:
        raise ValueError(f"A probability lies in (0, 1), got {probability}.")
    live = _validate(weights)
    if np.allclose(live, live[0]):
        return float(live[0] * chi2.ppf(probability, live.size))

    first = float(np.sum(live))
    second = float(np.sum(live**2))
    guess = (second / first) * float(chi2.ppf(probability, first**2 / second))

    low, high = 0.5 * guess, 2.0 * guess
    for _ in range(60):
        if weighted_chi2_cdf(live, low, method=method, tolerance=tolerance) <= (
            probability
        ):
            break
        low *= 0.5
    for _ in range(60):
        if weighted_chi2_cdf(live, high, method=method, tolerance=tolerance) >= (
            probability
        ):
            break
        high *= 2.0

    return float(
        brentq(
            lambda value: weighted_chi2_cdf(
                live, value, method=method, tolerance=tolerance
            )
            - probability,
            low,
            high,
            xtol=1e-12,
            rtol=1e-12,
        )
    )
