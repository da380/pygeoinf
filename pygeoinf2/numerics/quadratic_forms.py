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

from typing import Any

import numpy as np
from numpy.random import Generator

__all__ = ["weighted_chi2_cdf", "weighted_chi2_quantile"]


# ``scipy.stats`` and ``scipy.optimize`` are imported on first use, not at
# module scope. This module is reached by ``import pygeoinf2`` through the
# ``numerics`` package, so every session paid for them, and measured on this
# machine the two cost 0.26 s of a 0.48 s import -- more than half of it --
# for functions most sessions never call. ``scipy.integrate.quad`` was
# imported here too and never used: the Imhof integral is a vectorised
# trapezoid rule, for the reason ``_imhof`` gives.
#
# One other module still imports ``scipy.stats`` eagerly --
# ``inference/problem.py``, for a single ``chi2.ppf`` -- and until that one
# follows, this change saves nothing at all: measured 0.48 s either way, and
# 0.22 s once both are lazy.


def _chi2() -> "Any":
    """``scipy.stats.chi2``, imported on first use.

    Returns:
        The frozen-distribution factory.
    """
    from scipy.stats import chi2

    return chi2


def _brentq() -> "Any":
    """``scipy.optimize.brentq``, imported on first use.

    Returns:
        The bracketed root finder.
    """
    from scipy.optimize import brentq

    return brentq


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


def _imhof_integrand(grid: np.ndarray, weights: np.ndarray, value: float) -> np.ndarray:
    """Imhof's integrand on a whole grid at once.

    The denominator is formed in log space. It is a product over the weights,
    so a long anisotropic spectrum overflows a direct evaluation long before
    the result underflows.
    """
    scaled = weights[None, :] * grid[:, None]
    theta = 0.5 * (np.sum(np.arctan(scaled), axis=1) - value * grid)
    sine = np.sin(theta)
    log_modulus = 0.25 * np.sum(np.log1p(scaled * scaled), axis=1)
    magnitude = np.log(np.abs(sine) + 1e-300) - np.log(grid) - log_modulus
    return np.sign(sine) * np.exp(np.minimum(magnitude, 700.0))


def _imhof(weights: np.ndarray, value: float, /, *, tolerance: float) -> float:
    """Imhof's inversion of the characteristic function.

    A vectorised trapezoid rule, which is v1's, rather than adaptive
    quadrature on a scalar integrand. The integrand oscillates with a period
    set by *value* and decays like a power of the variable, and adaptive
    quadrature handles neither well: it spent its subdivisions near the origin,
    exhausted the 800 it was allowed, and warned. Measured at three weights --
    the slowest-decaying case -- that cost 106 ms against 3.7 ms here.

    The trapezoid works because both difficulties can be quantified rather than
    discovered. The amplitude is bounded by ``1 / (u rho(u))`` with
    ``rho(u) = prod_j (1 + w_j^2 u^2)^(1/4)``, so a truncation point follows
    from ``1 / (pi U rho(U)) < tolerance``; and ``rho`` is a product over every
    weight, so a long spectrum needs a far shorter range than a single weight
    would -- which is why the truncation is found by doubling rather than from
    the smallest weight alone. The step is set to at least 32 points per
    oscillation, then halved until the integral settles.
    """
    live = weights[weights > 0.0]
    if live.size == 0:
        return 1.0 if value >= 0.0 else 0.0

    # Far enough out that the remaining tail is below the tolerance, and at
    # least far enough to resolve the oscillation.
    target = np.log(1.0 / (np.pi * max(tolerance, 1e-300)))
    truncation = max(4.0 / float(np.max(live)), 1.0)
    for _ in range(80):
        log_modulus = 0.25 * float(np.sum(np.log1p((live * truncation) ** 2)))
        if np.log(max(truncation, 1e-300)) + log_modulus >= target:
            break
        truncation *= 2.0
    truncation = max(truncation, 64.0 * np.pi / max(value, 1e-6), 1.0)

    step = 2.0 * np.pi / max(value, 1e-6) / 32.0
    integral, previous = 0.0, None
    limit = 200_000
    # The u -> 0 limit of the integrand, which the grid below starts past.
    origin = 0.5 * (float(np.sum(live)) - value)
    for _ in range(6):
        count = min(int(truncation / step) + 1, limit)
        grid = np.linspace(step, count * step, count)
        values = _imhof_integrand(grid, weights, value)
        integral = step * (0.5 * (origin + values[-1]) + float(np.sum(values[:-1])))
        if previous is not None and abs(integral - previous) <= tolerance * max(
            abs(integral), 1e-12
        ):
            break
        previous = integral
        step *= 0.5
        if int(truncation / step) > limit:
            break

    return float(np.clip(0.5 - integral / np.pi, 0.0, 1.0))


def _matched(weights: np.ndarray, value: float) -> float:
    """A chi-square matched on its first two cumulants (Satterthwaite-Welch)."""
    first = float(np.sum(weights))
    second = float(np.sum(weights**2))
    scale = second / first
    degrees = first**2 / second
    return float(_chi2().cdf(value / scale, degrees))


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

    Returns:
        The probability.

    Raises:
        ValueError: for an unknown method, or weights that are negative or
            all zero -- a weighted sum of squares with no positive weight is
            not a distribution this can invert.
    """
    live = _validate(weights)
    if value <= 0.0:
        return 0.0
    # Equal weights are an ordinary chi-square, exactly -- and that is the
    # commonest case, since a Mahalanobis form has all its weights one. It is
    # also the case Imhof handles worst: with one or two terms the integrand
    # decays slowest and the quadrature is only good to about 1e-3 in the tail.
    if np.allclose(live, live[0]):
        return float(_chi2().cdf(value / live[0], live.size))
    if method == "matched":
        return _matched(live, value)
    if method == "monte_carlo":
        generator = np.random.default_rng() if rng is None else rng
        draws = generator.standard_normal((samples, live.size))
        return float(np.mean(draws**2 @ live <= value))
    if method in ("imhof", "auto"):
        try:
            return _imhof(live, value, tolerance=tolerance)
        except (ArithmeticError, ValueError):  # pragma: no cover - rare
            # Named rather than bare: an overflow or a degenerate weight set
            # is a reason to fall back on the moment-matched approximation,
            # and a KeyboardInterrupt or a bug in the integrand is not.
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
    samples: int = 100_000,
    rng: Generator | None = None,
) -> float:
    """The ``probability`` quantile of ``sum_i w_i Z_i^2``.

    A root find on :func:`weighted_chi2_cdf`, bracketed from the moment-matched
    approximation — which is close enough to bracket and never close enough to
    trust on its own.

    Args:
        weights: the coefficients ``w_i``.
        probability: the quantile wanted, in ``(0, 1)``.
        method: as for :func:`weighted_chi2_cdf`. ``"monte_carlo"`` is taken
            here as an *empirical* quantile of one sample rather than a root
            find on a noisy CDF, which would not converge: brentq on a
            function that returns a different value each call has no root to
            find. That is v1's behaviour too.
        tolerance: the accuracy asked of the CDF at each probe.
        samples: draws for ``method="monte_carlo"``.
        rng: the generator for those draws.

    Returns:
        The quantile.

    Raises:
        ValueError: for a probability outside ``(0, 1)``, or bad weights.
    """
    if not 0.0 < probability < 1.0:
        raise ValueError(f"A probability lies in (0, 1), got {probability}.")
    live = _validate(weights)
    if np.allclose(live, live[0]):
        return float(live[0] * _chi2().ppf(probability, live.size))

    if method == "monte_carlo":
        generator = np.random.default_rng() if rng is None else rng
        draws = generator.standard_normal((samples, live.size)) ** 2 @ live
        return float(np.quantile(draws, probability))

    first = float(np.sum(live))
    second = float(np.sum(live**2))
    guess = (second / first) * float(_chi2().ppf(probability, first**2 / second))

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
        _brentq()(
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
