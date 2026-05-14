"""
Weighted chi-square distribution: CDF and quantile.

This module implements numerical methods for the distribution of

    Q = sum_j w_j Z_j^2,    Z_j ~ iid N(0, 1),   w_j > 0,

which arises when calibrating Gaussian credible sets in function spaces.
With weights $w_j = \\lambda_j$ (eigenvalues of the covariance) the
distribution governs ambient norm balls; with $w_j = \\lambda_j^{1-\\theta}$
it governs weakened-covariance ellipsoids.

The mathematical background is in
``docs/agent-docs/theory/function-space-hardening.md``, section 4.

Public functions
----------------
weighted_chi2_cdf      : evaluate $P(Q \\le t)$.
weighted_chi2_quantile : invert the CDF for given probability $p$.

Both accept a ``method`` argument selecting between

* ``"imhof"`` (default): Imhof's numerical inversion of the
  characteristic function. Exact to quadrature tolerance.
* ``"ws"``: Welch--Satterthwaite moment-match to a scaled $\\chi^2_\\nu$.
  Closed form; exact when all weights are equal; degrades with anisotropy.
* ``"saddlepoint"``: Lugannani--Rice saddlepoint approximation. Excellent
  in deep tails.
* ``"mc"``: Monte Carlo empirical quantile / CDF. Unbiased reference.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipy.optimize
import scipy.stats


_VALID_METHODS = ("imhof", "ws", "saddlepoint", "mc")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def weighted_chi2_cdf(
    weights: np.ndarray,
    t: Union[float, np.ndarray],
    *,
    method: str = "imhof",
    rtol: float = 1e-8,
    n_samples: int = 100_000,
    rng: Optional[np.random.Generator] = None,
) -> Union[float, np.ndarray]:
    """Evaluate $P(Q \\le t)$ for $Q = \\sum_j w_j Z_j^2$.

    Args:
        weights: Non-negative weights $w_j$. Length-zero or all-zero arrays
            are treated as the degenerate point mass at 0.
        t: Threshold(s); scalar or array. Returns matching shape.
        method: One of {"imhof", "ws", "saddlepoint", "mc"}.
        rtol: Relative quadrature tolerance for Imhof's integral.
        n_samples: Sample count for Monte Carlo. Ignored by other methods.
        rng: Optional NumPy generator for Monte Carlo. Defaults to
            ``np.random.default_rng()``.

    Returns:
        $P(Q \\le t)$ as a float (if ``t`` is scalar) or numpy array.

    Raises:
        ValueError: For unknown method, negative weights, or NaN inputs.
    """
    w = _validate_weights(weights)
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose one of {_VALID_METHODS}."
        )

    scalar = np.isscalar(t)
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))

    if w.size == 0 or np.all(w == 0.0):
        # Q = 0 almost surely; F(t) = 1 if t >= 0 else 0.
        out = np.where(t_arr >= 0.0, 1.0, 0.0)
        return float(out[0]) if scalar else out

    if method == "imhof":
        out = np.array([_imhof_cdf(w, float(ti), rtol=rtol) for ti in t_arr])
    elif method == "ws":
        out = _ws_cdf(w, t_arr)
    elif method == "saddlepoint":
        out = np.array([_saddlepoint_cdf(w, float(ti)) for ti in t_arr])
    elif method == "mc":
        out = _mc_cdf(w, t_arr, n_samples=n_samples, rng=rng)
    else:  # pragma: no cover - guarded above
        raise AssertionError("unreachable")

    return float(out[0]) if scalar else out


def weighted_chi2_quantile(
    weights: np.ndarray,
    probability: float,
    *,
    method: str = "imhof",
    rtol: float = 1e-6,
    n_samples: int = 100_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Return $r$ such that $P(\\sum_j w_j Z_j^2 \\le r) = $ probability.

    Args:
        weights: Non-negative weights $w_j$.
        probability: Target probability, strictly between 0 and 1.
        method: One of {"imhof", "ws", "saddlepoint", "mc"}.
        rtol: Relative tolerance for root-finding (Imhof, saddlepoint).
        n_samples: Sample count for Monte Carlo.
        rng: Optional NumPy generator for Monte Carlo.

    Returns:
        The quantile $r$.

    Raises:
        ValueError: For invalid probability, unknown method, or negative weights.
    """
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between 0 and 1.")
    w = _validate_weights(weights)
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose one of {_VALID_METHODS}."
        )

    if w.size == 0 or np.all(w == 0.0):
        return 0.0

    if method == "ws":
        return float(_ws_quantile(w, probability))
    if method == "mc":
        return float(_mc_quantile(w, probability, n_samples=n_samples, rng=rng))

    # Closed-form fast path for equal positive weights.
    positive = w[w > 0.0]
    if positive.size > 0 and np.allclose(
        positive, positive[0], rtol=1e-12, atol=0.0
    ):
        return float(
            positive[0] * scipy.stats.chi2.ppf(probability, df=positive.size)
        )

    cdf_fn = (
        (lambda t: _imhof_cdf(w, t, rtol=rtol))
        if method == "imhof"
        else (lambda t: _saddlepoint_cdf(w, t))
    )
    return float(_invert_cdf(cdf_fn, w, probability, rtol=rtol))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float).ravel()
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite.")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative.")
    return w


# ---------------------------------------------------------------------------
# Imhof's method
# ---------------------------------------------------------------------------


def _imhof_integrand_vec(
    u: np.ndarray, weights: np.ndarray, t: float
) -> np.ndarray:
    """Vectorised Imhof integrand on a grid ``u`` of strictly positive points.

    Works in log-space for the denominator to avoid overflow when the
    spectrum is long and anisotropic.
    """
    wu = weights[None, :] * u[:, None]
    theta = 0.5 * (np.sum(np.arctan(wu), axis=1) - t * u)
    sin_theta = np.sin(theta)
    log_rho = 0.25 * np.sum(np.log1p(wu * wu), axis=1)
    log_abs = np.log(np.abs(sin_theta) + 1e-300) - np.log(u) - log_rho
    return np.sign(sin_theta) * np.exp(np.minimum(log_abs, 700.0))


def _imhof_cdf(weights: np.ndarray, t: float, *, rtol: float = 1e-8) -> float:
    """Imhof CDF $P(Q \\le t)$ via fixed-step trapezoidal quadrature.

    A vectorised trapezoidal rule on a truncated domain handles the
    oscillatory multi-scale integrands that arise from long anisotropic
    spectra (e.g. inverse-Laplacian eigenvalues $\\lambda_j \\propto 1/j^2$)
    far more reliably than ``scipy.integrate.quad``'s adaptive routine,
    which exhausts its subdivision budget on such cases.

    The step size $h$ is chosen to resolve the dominant oscillation period
    $2\\pi/t$ with $\\ge 32$ samples; the truncation $U$ is set so the
    integrand magnitude $|\\sin(\\theta)|/(u\\rho(u))$ is well below $rtol$
    at $u = U$, exploiting the fact that for $u \\gg 1/\\min(w_j)$ all
    $\\arctan(w_j u) \\to \\pi/2$ and the integrand decays as
    $1/u^{1+n/2}$.

    Refinement: the result of a step-$h$ trapezoidal pass is compared
    against a step-$h/2$ pass; the routine terminates when the relative
    Richardson-style change is below $rtol$ or the step budget is
    exhausted.
    """
    if t <= 0.0:
        return 0.0

    # Closed-form fast path for equal positive weights: Q = w * chi^2_n.
    positive = weights[weights > 0.0]
    if positive.size > 0 and np.allclose(
        positive, positive[0], rtol=1e-12, atol=0.0
    ):
        return float(scipy.stats.chi2.cdf(t / positive[0], df=positive.size))

    mean = float(np.sum(weights))

    # Truncation point.  We need U large enough that the integrand tail
    # integral is below rtol.  The amplitude bound is
    #
    #   |integrand(u)| <= 1 / (u * rho(u)),
    #   rho(u) = prod_j (1 + w_j^2 u^2)^{1/4}
    #
    # so the condition 1/(pi * U * rho(U)) < rtol is sufficient.
    #
    # Single-weight heuristic (16/w_min) massively overestimates U when N is
    # large: rho grows as a *product* over all weights, so U only needs to
    # reach the point where the product exceeds 1/(pi * rtol * U).  For an
    # N=50 decaying spectrum this is ~600 instead of ~400 000.  We find U by
    # doubling from an initial guess of 4/w_max until the condition is met.
    U_oscillation = 64.0 * np.pi / max(t, 1e-6)
    _target = np.log(1.0 / (np.pi * max(rtol, 1e-300)))
    _U = max(4.0 / float(np.max(positive)), 1.0)
    for _ in range(80):
        _log_rho = 0.25 * float(np.sum(np.log1p((positive * _U) ** 2)))
        if np.log(max(_U, 1e-300)) + _log_rho >= _target:
            break
        _U *= 2.0
    U = max(_U, U_oscillation, 1.0)

    # Step size: at least 32 points per oscillation period 2pi/t.
    h_base = 2.0 * np.pi / max(t, 1e-6) / 32.0
    # Iterate trapezoidal step halving for Richardson-style refinement.
    integral_prev: Optional[float] = None
    integral_curr = 0.0
    max_iters = 6
    n_steps_cap = 200_000
    h = h_base
    for _ in range(max_iters):
        n_steps = min(int(U / h) + 1, n_steps_cap)
        u_grid = np.linspace(h, n_steps * h, n_steps)
        vals = _imhof_integrand_vec(u_grid, weights, t)
        # u=0 contribution at the limit.
        f0 = 0.5 * (mean - t)
        integral_curr = h * (
            0.5 * (f0 + vals[-1]) + float(np.sum(vals[:-1]))
        )
        if integral_prev is not None:
            denom = max(abs(integral_curr), 1e-12)
            if abs(integral_curr - integral_prev) / denom < rtol:
                break
        integral_prev = integral_curr
        h *= 0.5
        if int(U / h) > n_steps_cap:
            break

    cdf = 0.5 - integral_curr / np.pi
    return float(np.clip(cdf, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Welch--Satterthwaite (moment matching)
# ---------------------------------------------------------------------------


def _ws_params(weights: np.ndarray) -> tuple[float, float]:
    s1 = float(np.sum(weights))
    s2 = float(np.sum(weights * weights))
    a = s2 / s1
    nu = s1 * s1 / s2
    return a, nu


def _ws_cdf(weights: np.ndarray, t: np.ndarray) -> np.ndarray:
    a, nu = _ws_params(weights)
    return scipy.stats.chi2.cdf(t / a, df=nu)


def _ws_quantile(weights: np.ndarray, probability: float) -> float:
    a, nu = _ws_params(weights)
    return a * float(scipy.stats.chi2.ppf(probability, df=nu))


# ---------------------------------------------------------------------------
# Lugannani--Rice saddlepoint
# ---------------------------------------------------------------------------


def _cgf(s: float, weights: np.ndarray) -> float:
    return -0.5 * float(np.sum(np.log1p(-2.0 * s * weights)))


def _cgf_prime(s: float, weights: np.ndarray) -> float:
    return float(np.sum(weights / (1.0 - 2.0 * s * weights)))


def _cgf_double_prime(s: float, weights: np.ndarray) -> float:
    denom = 1.0 - 2.0 * s * weights
    return 2.0 * float(np.sum((weights * weights) / (denom * denom)))


def _saddlepoint_cdf(weights: np.ndarray, t: float) -> float:
    if t <= 0.0:
        return 0.0
    mean = float(np.sum(weights))
    # Saddlepoint at the mean is exactly s_hat = 0 and the formula collapses;
    # use Welch-Satterthwaite to avoid a 0/0.
    if abs(t - mean) < 1e-12 * max(1.0, mean):
        return float(_ws_cdf(weights, np.array([t]))[0])

    w_max = float(np.max(weights))
    s_max = 0.5 / w_max  # s must satisfy s < 1/(2 max w_j) for convergence
    epsilon = 1e-12

    if t < mean:
        lo, hi = -1e8, -epsilon
    else:
        lo, hi = epsilon, s_max * (1.0 - 1e-10)

    def equation(s: float) -> float:
        return _cgf_prime(s, weights) - t

    # Expand the lower bound for very small t.
    if t < mean:
        while equation(lo) > 0.0:
            lo *= 10.0
            if lo < -1e30:
                return 0.0

    s_hat = scipy.optimize.brentq(equation, lo, hi, xtol=1e-14, rtol=1e-12)
    k_hat = _cgf(s_hat, weights)
    k_pp = _cgf_double_prime(s_hat, weights)

    inner = 2.0 * (s_hat * t - k_hat)
    if inner <= 0.0:
        return float(_ws_cdf(weights, np.array([t]))[0])

    w_hat = np.sign(s_hat) * np.sqrt(inner)
    u_hat = s_hat * np.sqrt(k_pp)

    cdf = scipy.stats.norm.cdf(w_hat) + scipy.stats.norm.pdf(w_hat) * (
        1.0 / w_hat - 1.0 / u_hat
    )
    return float(np.clip(cdf, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


def _draw_q_samples(
    weights: np.ndarray, n_samples: int, rng: Optional[np.random.Generator]
) -> np.ndarray:
    """Generate ``n_samples`` draws of $Q = \\sum_j w_j Z_j^2$.

    Chunked to keep peak memory at roughly 8 MB regardless of len(weights).
    """
    rng = np.random.default_rng() if rng is None else rng
    n = max(1, int(weights.size))
    target_floats = 1_000_000
    chunk = max(1, target_floats // n)
    samples = np.empty(n_samples, dtype=float)
    cursor = 0
    while cursor < n_samples:
        m = min(chunk, n_samples - cursor)
        z = rng.standard_normal((m, n))
        np.multiply(z, z, out=z)
        samples[cursor : cursor + m] = z @ weights
        cursor += m
    return samples


def _mc_cdf(
    weights: np.ndarray,
    t: np.ndarray,
    *,
    n_samples: int,
    rng: Optional[np.random.Generator],
) -> np.ndarray:
    samples = _draw_q_samples(weights, n_samples, rng)
    samples_sorted = np.sort(samples)
    # P(Q <= t) is the fraction of samples not exceeding t.
    return np.searchsorted(samples_sorted, t, side="right") / n_samples


def _mc_quantile(
    weights: np.ndarray,
    probability: float,
    *,
    n_samples: int,
    rng: Optional[np.random.Generator],
) -> float:
    samples = _draw_q_samples(weights, n_samples, rng)
    return float(np.quantile(samples, probability))


# ---------------------------------------------------------------------------
# CDF inversion
# ---------------------------------------------------------------------------


def _invert_cdf(
    cdf_func,
    weights: np.ndarray,
    probability: float,
    *,
    rtol: float = 1e-6,
) -> float:
    """Invert ``cdf_func`` at ``probability`` using a WS-seeded brentq bracket."""
    t_seed = max(_ws_quantile(weights, probability), 1e-12)

    lo, hi = 0.5 * t_seed, 2.0 * t_seed
    cdf_lo = cdf_func(lo)
    cdf_hi = cdf_func(hi)

    while cdf_lo > probability:
        lo *= 0.5
        if lo < 1e-300:
            return 0.0
        cdf_lo = cdf_func(lo)
    while cdf_hi < probability:
        hi *= 2.0
        if hi > 1e30:
            raise RuntimeError(
                "Unable to bracket the CDF for inversion; weights may be "
                "pathological."
            )
        cdf_hi = cdf_func(hi)

    return scipy.optimize.brentq(
        lambda t: cdf_func(t) - probability, lo, hi, rtol=rtol, xtol=rtol
    )
