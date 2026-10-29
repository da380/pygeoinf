"""
The distribution of a quadratic form in Gaussian variables.

``sum_i w_i Z_i^2`` with independent standard normals is a *weighted* chi-square
and has no closed form. It comes up wherever a statement is made about the size
of a Gaussian vector in a metric other than its own: the Mahalanobis form is an
ordinary chi-square because the weights are all one, but the plain squared norm
of a field is not, and its weights are the covariance's eigenvalues.

Imhof's method is exact to the tolerance asked for — it inverts the
characteristic function numerically — and is the default. The Lugannani-Rice
saddlepoint approximation is much faster and is available by name; a
moment-matched chi-square is the cheap approximation to fall back on. Equal
weights are an ordinary chi-square and are computed as one.

**The answer does not depend on the units of the weights.** The distribution
scales exactly with them -- ``sum (s w_i) Z_i^2 == s sum w_i Z_i^2`` -- so
both functions divide the weights and the threshold by the largest weight on
entry and multiply a quantile back on the way out. Every tolerance, floor
and cap below therefore acts on weights whose largest is one, and a
covariance in units that put its eigenvalues at ``1e-12`` gets the same
quantile, in its own units, as one whose eigenvalues are of order one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import Generator, default_rng

__all__ = ["weighted_chi2_cdf", "weighted_chi2_quantile"]

_METHODS = ("auto", "imhof", "saddlepoint", "matched", "monte_carlo")


# ``scipy.stats`` and ``scipy.optimize`` are imported on first use, not at
# module scope. This module is reached by ``import pygeoinf2`` through the
# ``numerics`` package, and the two are a large part of the package's import
# time, for functions most sessions never call. tests/test_code_practice.py
# checks that loading this module imports neither, and loads it on its own
# to do so -- which is why it resolves ``rng=None`` itself rather than
# through the package's helper.


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
    """The positive weights, refusing a set that defines no distribution.

    Raises:
        ValueError: for no weights, a negative one, or none positive.
    """
    values = np.asarray(weights, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("At least one weight is needed.")
    if np.any(values < 0.0):
        raise ValueError("The weights must be non-negative.")
    live = values[values > 0.0]
    if live.size == 0:
        raise ValueError("At least one weight must be positive.")
    return live


def _normalized(weights: np.ndarray) -> tuple[np.ndarray, float]:
    """The positive weights divided by the largest, and the largest.

    The weighted chi-square scales exactly with its weights, so working on
    weights whose largest is one loses nothing and makes every threshold in
    this module relative to the weights' own scale.

    Raises:
        ValueError: as :func:`_validate`.
    """
    live = _validate(weights)
    scale = float(np.max(live))
    return live / scale, scale


def _common_weight(live: np.ndarray, tolerance: float) -> float | None:
    """The common value of weights that are equal for the purpose, or ``None``.

    Weights whose spread is at most ``sqrt(tolerance)`` of the largest are
    taken as equal, at their mean. The distribution depends on the weights
    symmetrically, so moving them to their mean changes it only at second
    order in their spread: the ordinary chi-square at the mean is then exact
    to about ``tolerance``, with no quadrature at all.

    Args:
        live: positive weights.
        tolerance: the accuracy asked of the distribution function.

    Returns:
        The mean weight, or ``None`` when the weights differ by more.
    """
    largest = float(np.max(live))
    spread = largest - float(np.min(live))
    if spread <= np.sqrt(max(tolerance, 0.0)) * largest:
        return float(np.mean(live))
    return None


def _validate_method(method: str) -> None:
    """Refuse an unknown method before any shortcut can return without it.

    Raises:
        ValueError: for a method not in ``_METHODS``.
    """
    if method not in _METHODS:
        raise ValueError(
            f"Unknown method {method!r}; the methods are "
            + ", ".join(repr(name) for name in _METHODS)
            + "."
        )


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

    A vectorized trapezoid rule rather than adaptive quadrature on a scalar
    integrand. The integrand oscillates with a period set by *value* and
    decays like a power of the variable, and adaptive quadrature handles
    neither well: it spends its subdivisions near the origin and exhausts
    its budget.

    The trapezoid works because both difficulties can be quantified rather than
    discovered. The amplitude is bounded by ``1 / (u rho(u))`` with
    ``rho(u) = prod_j (1 + w_j^2 u^2)^(1/4)``, so a truncation point follows
    from ``1 / (pi U rho(U)) < tolerance``; and ``rho`` is a product over every
    weight, so a long spectrum needs a far shorter range than a single weight
    would -- which is why the truncation is found by doubling rather than from
    the smallest weight alone. The step is set to at least 32 points per
    oscillation, then halved until the integral settles.

    The weights and the threshold are divided by the largest weight first.
    The probability is unchanged by that, and the floors on the step and the
    truncation, the cap on the grid and the floor under the convergence
    test are then all relative to the weights' own scale.
    """
    live = weights[weights > 0.0]
    if live.size == 0:
        return 1.0 if value >= 0.0 else 0.0
    largest = float(np.max(live))
    live = live / largest
    value = value / largest

    # Far enough out that the remaining tail is below the tolerance, and at
    # least far enough to resolve the oscillation. The doubling starts at
    # four over the largest weight, which is four after the normalization.
    target = np.log(1.0 / (np.pi * max(tolerance, 1e-300)))
    truncation = 4.0
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
        values = _imhof_integrand(grid, live, value)
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


def _cumulant(order: int, s: float, weights: np.ndarray) -> float:
    """The cumulant generating function of the form and its derivatives.

    ``K(s) == -sum log(1 - 2 s w) / 2``, defined for ``s < 1 / (2 max w)``;
    ``K'`` is the mean of the tilted distribution and ``K''`` its variance.
    """
    factor = 1.0 - 2.0 * s * weights
    if order == 0:
        return float(-0.5 * np.sum(np.log(factor)))
    if order == 1:
        return float(np.sum(weights / factor))
    return float(np.sum(2.0 * weights**2 / factor**2))


def _saddlepoint(weights: np.ndarray, value: float) -> float:
    """The Lugannani-Rice approximation to ``P(sum w Z^2 <= value)``.

    Tilt the distribution to put its mean at the threshold -- the saddlepoint
    ``s`` solves ``K'(s) == value``, a scalar root find on a monotone
    function -- and read the tail off a normal with a first-order correction:
    ``Phi(w) + phi(w) (1 / w - 1 / u)`` with ``w == sign(s) sqrt(2 (s value
    - K(s)))`` and ``u == s sqrt(K''(s))``. A handful of sums per probe,
    against Imhof's oscillatory integral, and its relative error *falls* in
    the tails, where the quadrature's rises. At the mean the correction is
    ``0 / 0``, and the moment-matched value is returned there; the
    approximation is not exact on an ordinary chi-square either, which the
    caller handles before reaching here.
    """
    mean = float(np.sum(weights))
    if abs(value - mean) <= 1e-9 * mean:
        return _matched(weights, value)
    upper = 0.5 / float(np.max(weights))
    if value < mean:
        low, high = -1.0, -1e-12
        while _cumulant(1, low, weights) > value:
            low *= 10.0
            if low < -1e30:
                return 0.0
    else:
        low, high = 1e-12, upper * (1.0 - 1e-10)
    s = float(
        _brentq()(
            lambda t: _cumulant(1, t, weights) - value,
            low,
            high,
            xtol=1e-14,
            rtol=1e-12,
        )
    )
    inner = 2.0 * (s * value - _cumulant(0, s, weights))
    if inner <= 0.0:
        return _matched(weights, value)
    w = float(np.sign(s) * np.sqrt(inner))
    u = s * float(np.sqrt(_cumulant(2, s, weights)))
    from scipy.stats import norm

    return float(np.clip(norm.cdf(w) + norm.pdf(w) * (1.0 / w - 1.0 / u), 0.0, 1.0))


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

    The weights and *value* are divided by the largest weight before
    anything else, so the answer is the same in any units: the
    distribution scales exactly with its weights.

    Args:
        weights: the coefficients, non-negative and not all zero.
        value: the threshold, in the units of the weights.
        method: ``"imhof"`` inverts the characteristic function and is exact to
            ``tolerance``; ``"saddlepoint"`` is the Lugannani-Rice
            approximation, a few sums and a scalar root find, whose relative
            error falls in the tails and is around ``0.1 / nu`` in the body
            for ``nu`` the effective degrees of freedom ``(sum w)^2 / sum
            w^2``; ``"matched"`` fits a chi-square on two cumulants and is
            fast and rough; ``"monte_carlo"`` samples. ``"auto"`` takes
            Imhof, falling back to the matched form if the integral misbehaves.
        tolerance: the absolute accuracy asked of the probability, for
            Imhof. Weights within ``sqrt(tolerance)`` of each other, relative
            to the largest, are computed as an ordinary chi-square at their
            mean, which is exact to about this tolerance.
        samples: for Monte Carlo.
        rng: for Monte Carlo.

    Returns:
        The probability.

    Raises:
        ValueError: for an unknown method, or weights that are negative or
            all zero -- a weighted sum of squares with no positive weight is
            not a distribution this can invert.
    """
    _validate_method(method)
    live, scale = _normalized(weights)
    if value <= 0.0:
        return 0.0
    value = value / scale
    # Equal weights are an ordinary chi-square, exactly -- and that is the
    # commonest case, since a Mahalanobis form has all its weights one. It is
    # also the case Imhof handles worst: with a single term the integrand
    # decays slowest and the quadrature is only good to about 1e-7.
    common = _common_weight(live, tolerance)
    if common is not None:
        return float(_chi2().cdf(value / common, live.size))
    if method == "matched":
        return _matched(live, value)
    if method == "saddlepoint":
        return _saddlepoint(live, value)
    if method == "monte_carlo":
        generator = default_rng() if rng is None else rng
        draws = generator.standard_normal((samples, live.size))
        return float(np.mean(draws**2 @ live <= value))
    try:
        return _imhof(live, value, tolerance=tolerance)
    except (ArithmeticError, ValueError):  # pragma: no cover - rare
        # Named rather than bare: an overflow or a degenerate weight set
        # is a reason to fall back on the moment-matched approximation,
        # and a KeyboardInterrupt or a bug in the integrand is not.
        if method == "imhof":
            raise
        return _matched(live, value)


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

    The weights are divided by the largest before anything else and the
    quantile multiplied back, so the quantile is the same in any units: the
    root find's tolerances are relative to the weights' own scale.

    Args:
        weights: the coefficients ``w_i``.
        probability: the quantile wanted, in ``(0, 1)``.
        method: as for :func:`weighted_chi2_cdf`. ``"monte_carlo"`` is taken
            here as an *empirical* quantile of one sample rather than a root
            find on a noisy CDF, which would not converge: brentq on a
            function that returns a different value each call has no root to
            find.
        tolerance: the accuracy asked of the CDF at each probe, as for
            :func:`weighted_chi2_cdf`.
        samples: draws for ``method="monte_carlo"``.
        rng: the generator for those draws.

    Returns:
        The quantile, in the units of the weights.

    Raises:
        ValueError: for a probability outside ``(0, 1)``, an unknown method,
            or bad weights.
    """
    if not 0.0 < probability < 1.0:
        raise ValueError(f"A probability lies in (0, 1), got {probability}.")
    _validate_method(method)
    live, scale = _normalized(weights)
    return scale * _normalized_quantile(
        live,
        probability,
        method=method,
        tolerance=tolerance,
        samples=samples,
        rng=rng,
    )


def _normalized_quantile(
    live: np.ndarray,
    probability: float,
    /,
    *,
    method: str,
    tolerance: float,
    samples: int,
    rng: Generator | None,
) -> float:
    """The quantile for positive weights whose largest is one.

    Args:
        live: the weights, normalized.
        probability: the quantile wanted.
        method: as for :func:`weighted_chi2_quantile`.
        tolerance: likewise.
        samples: likewise.
        rng: likewise.

    Returns:
        The quantile, in the units of the normalized weights.
    """
    common = _common_weight(live, tolerance)
    if common is not None:
        return float(common * _chi2().ppf(probability, live.size))

    if method == "monte_carlo":
        generator = default_rng() if rng is None else rng
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
