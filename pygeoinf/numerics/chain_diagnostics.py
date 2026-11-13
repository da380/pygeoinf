"""
Diagnostics of Markov chains, on plain arrays of a scalar.

A chain's states are correlated, so ``n`` of them carry the information of
fewer independent draws. For a scalar ``f`` along the chain the integrated
autocorrelation time is ``tau == 1 + 2 sum_k rho(k)``, the effective sample
size is ``n / tau``, and the Monte Carlo standard error of the mean is
``sigma sqrt(tau / n)``. ``tau`` here is estimated by Geyer's initial
monotone sequence, combining the chains' autocovariances as Stan does, and
convergence across chains is judged by the split, rank-normalized
``R-hat`` of Vehtari, Gelman, Simpson, Carpenter and Burkner (2021), the
larger of its bulk and tail forms.

These are statements about the scalars measured, and only a lower bound on
how slowly the chain mixes as a whole: a slow direction that none of them
sees, a barrier between modes say, is invisible. On a function space the
scalars should be ones with a limit under refinement -- the potential, a
chart coordinate, a bounded property -- not a component of the
discretization.

Every function takes a one-dimensional array (one chain) or a
two-dimensional array of shape ``(chains, steps)``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "autocorrelation",
    "effective_sample_size",
    "integrated_autocorrelation_time",
    "split_rhat",
]


def _as_chains(values: np.ndarray, /) -> np.ndarray:
    """The draws as a ``(chains, steps)`` array.

    Raises:
        ValueError: for an array of more than two dimensions, or chains of
            fewer than four steps.
    """
    draws = np.asarray(values, dtype=float)
    if draws.ndim == 1:
        draws = draws[np.newaxis, :]
    if draws.ndim != 2:
        raise ValueError(
            f"Expected one chain or an array of shape (chains, steps); got shape "
            f"{draws.shape}."
        )
    if draws.shape[1] < 4:
        raise ValueError(f"A chain needs at least four steps, got {draws.shape[1]}.")
    return draws


def autocorrelation(values: np.ndarray, /) -> np.ndarray:
    """The autocorrelation of each chain at every lag, by FFT.

    Args:
        values: one chain, or an array of shape ``(chains, steps)``.

    Returns:
        An array of the input's shape, one at lag zero; a constant chain's
        autocorrelation is one at lag zero and zero beyond.

    Raises:
        ValueError: as for every function here, for an array of the wrong
            shape.
    """
    one = np.asarray(values).ndim == 1
    draws = _as_chains(values)
    steps = draws.shape[1]
    centered = draws - draws.mean(axis=1, keepdims=True)
    size = 1 << int(2 * steps - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=size, axis=1)
    covariance = np.fft.irfft(spectrum * np.conj(spectrum), n=size, axis=1)[:, :steps]
    variance = covariance[:, :1]
    result = np.zeros_like(covariance)
    nonzero = variance[:, 0] > 0.0
    result[nonzero] = covariance[nonzero] / variance[nonzero]
    result[~nonzero, 0] = 1.0
    return result[0] if one else result


def integrated_autocorrelation_time(values: np.ndarray, /) -> float:
    """``tau == 1 + 2 sum_k rho(k)``, by Geyer's initial monotone sequence.

    The chains' autocovariances are combined with the between-chain
    variance as in Stan, so a set of chains that disagree has a long
    autocorrelation time. The sum is over pairs ``rho(2k) + rho(2k+1)``,
    stopped at the first negative pair and made monotone, which bounds the
    estimate's variance without biasing it much.

    Args:
        values: one chain, or an array of shape ``(chains, steps)``.

    Returns:
        The autocorrelation time, in steps; at least ``1 / log10(total)``,
        as Stan floors it, so that an antithetic chain is not credited with
        unbounded efficiency.

    Raises:
        ValueError: for an array of the wrong shape.
    """
    draws = _as_chains(values)
    chains, steps = draws.shape
    variances = draws.var(axis=1, ddof=1)
    within = float(variances.mean())
    if within == 0.0:
        return 1.0
    between_ratio = float(draws.mean(axis=1).var(ddof=1)) if chains > 1 else 0.0
    pooled = (steps - 1) / steps * within + between_ratio
    rho_chains = autocorrelation(draws)
    covariance = rho_chains * (variances[:, np.newaxis] * (steps - 1) / steps)
    rho = 1.0 - (within - covariance.mean(axis=0)) / pooled
    rho[0] = 1.0
    pairs = rho[: steps - steps % 2].reshape(-1, 2).sum(axis=1)
    positive = np.argmax(pairs < 0.0) if np.any(pairs < 0.0) else pairs.size
    pairs = np.minimum.accumulate(pairs[:positive])
    tau = -1.0 + 2.0 * float(pairs.sum())
    return max(tau, 1.0 / np.log10(chains * steps))


def effective_sample_size(values: np.ndarray, /) -> float:
    """The number of independent draws the chains are worth, ``total / tau``.

    Args:
        values: one chain, or an array of shape ``(chains, steps)``.

    Returns:
        The effective sample size.

    Raises:
        ValueError: for an array of the wrong shape.
    """
    draws = _as_chains(values)
    return draws.size / integrated_autocorrelation_time(draws)


def _classic_rhat(draws: np.ndarray, /) -> float:
    """The potential scale reduction of ``(chains, steps)`` draws."""
    steps = draws.shape[1]
    within = float(draws.var(axis=1, ddof=1).mean())
    between = steps * float(draws.mean(axis=1).var(ddof=1))
    if within == 0.0:
        return 1.0 if between == 0.0 else float("inf")
    pooled = (steps - 1) / steps * within + between / steps
    return float(np.sqrt(pooled / within))


def _rank_normalized(draws: np.ndarray, /) -> np.ndarray:
    """The draws replaced by the normal scores of their pooled ranks."""
    from scipy.special import ndtri
    from scipy.stats import rankdata

    ranks = rankdata(draws, method="average").reshape(draws.shape)
    return ndtri((ranks - 0.375) / (draws.size + 0.25))


def split_rhat(values: np.ndarray, /) -> float:
    """The split, rank-normalized ``R-hat``: one when the chains agree.

    Each chain is split in two, so that a chain still drifting disagrees
    with itself; the draws are replaced by the normal scores of their ranks,
    so that heavy tails do not hide a disagreement; and the larger of the
    bulk value and the tail value (the same of the draws' distance from
    their median) is returned. Values above about 1.01 say the chains have
    not mixed.

    Args:
        values: one chain, or an array of shape ``(chains, steps)``.

    Returns:
        ``R-hat``.

    Raises:
        ValueError: for an array of the wrong shape.
    """
    draws = _as_chains(values)
    half = draws.shape[1] // 2
    split = np.concatenate([draws[:, :half], draws[:, -half:]], axis=0)
    bulk = _classic_rhat(_rank_normalized(split))
    folded = np.abs(split - np.median(split))
    tail = _classic_rhat(_rank_normalized(folded))
    return max(bulk, tail)
