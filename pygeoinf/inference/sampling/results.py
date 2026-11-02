"""
What a population of samplers returns: measures known through their draws.

Both results are absolutely continuous with respect to the prior, with the
density ``-Phi / T`` against it, and both hold the draws that represent them.
They differ in how. A :class:`SampledPosterior` comes from chains: its states
are exact in the long-run limit and correlated with one another. A
:class:`WeightedEnsemble` comes from tempering: its particles are weighted,
and its normalizer estimates the evidence.

A sampled posterior's states and a measure's draws have different
properties, so they are reached by different methods.
:meth:`SampledPosterior.sample` is a fresh draw, independent across calls;
:meth:`SampledPosterior.sample_thinned` reads the stored states, which costs
nothing and is correlated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from numpy.random import Generator

from ...algebra.operators import Functional
from ...algebra.spaces import resolve_rng
from ...numerics.chain_diagnostics import (
    effective_sample_size,
    integrated_autocorrelation_time,
    split_rhat,
)
from ...numerics.randomized import Estimate
from ...probability.absolutely_continuous import AbsolutelyContinuousMeasure
from ...probability.gaussian import GaussianMeasure

if TYPE_CHECKING:  # pragma: no cover
    from .corrections import Kernel

__all__ = ["Chain", "ChainDiagnostics", "SampledPosterior", "WeightedEnsemble"]


@dataclass(frozen=True, eq=False)
class Chain:
    """One chain's record, after its burn-in.

    Attributes:
        potential: the likelihood potential ``Phi`` at every step.
        accepted: whether each step's proposal was accepted.
        acceptance: each step's acceptance probability.
        records: the recorded scalars at every step, shape
            ``(steps, count)``, or ``None`` when nothing was recorded.
        positions: the stored states, one every ``thin`` steps.
        step_size: the time step the chain ran at after burn-in.
    """

    potential: np.ndarray
    accepted: np.ndarray
    acceptance: np.ndarray
    records: np.ndarray | None
    positions: tuple
    step_size: float

    @property
    def steps(self) -> int:
        """How many steps were kept."""
        return int(self.potential.size)


@dataclass(frozen=True)
class ChainDiagnostics:
    """The diagnostics of each recorded scalar across the chains.

    The scalars are the potential first, then the recorded ones in order.

    Attributes:
        names: what each scalar is.
        autocorrelation_time: the integrated autocorrelation time of each.
        effective_sample_size: the effective sample size of each.
        rhat: the split, rank-normalized ``R-hat`` of each.
    """

    names: tuple[str, ...]
    autocorrelation_time: np.ndarray
    effective_sample_size: np.ndarray
    rhat: np.ndarray

    @property
    def steps_per_independent_draw(self) -> int:
        """``ceil(2 max tau)``: a lower bound, a slow direction that no
        recorded scalar sees being invisible here."""
        return int(np.ceil(2.0 * float(np.max(self.autocorrelation_time))))


class SampledPosterior(AbsolutelyContinuousMeasure):
    """A posterior known through the chains that sampled it.

    The density against the prior is the target's, ``-Phi / T``, with an
    unknown normalizer. The chains' records give its diagnostics and the
    means of the recorded scalars with their Monte Carlo standard errors.

    Args:
        prior: the prior the density is taken against.
        relative_log_density: ``-Phi / T``.
        chains: the chains' records.
        kernel: the kernel the chains ran, at its starting step.
        record_names: names for the recorded scalars.
    """

    def __init__(
        self,
        prior: GaussianMeasure,
        relative_log_density: Functional,
        chains: Sequence[Chain],
        /,
        *,
        kernel: "Kernel",
        record_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(prior, relative_log_density)
        self._chains = tuple(chains)
        self._kernel = kernel
        count = 0
        if self._chains and self._chains[0].records is not None:
            count = self._chains[0].records.shape[1]
        if record_names is None:
            record_names = [f"record {k}" for k in range(count)]
        self._record_names = ("potential", *record_names)

    @property
    def chains(self) -> tuple[Chain, ...]:
        """The chains' records."""
        return self._chains

    @property
    def kernel(self) -> "Kernel":
        """The kernel the chains ran."""
        return self._kernel

    @property
    def acceptance_rate(self) -> float:
        """The mean acceptance probability over every kept step."""
        return float(np.mean([chain.acceptance.mean() for chain in self._chains]))

    def scalars(self) -> np.ndarray:
        """The potential and the recorded scalars, shape ``(count, chains, steps)``.

        Returns:
            The array.
        """
        columns = [np.stack([chain.potential for chain in self._chains])]
        if self._chains[0].records is not None:
            records = np.stack([chain.records for chain in self._chains])
            columns.extend(records[:, :, k] for k in range(records.shape[2]))
        return np.stack(columns)

    def diagnostics(self) -> ChainDiagnostics:
        """The autocorrelation time, effective sample size and ``R-hat`` of
        each scalar.

        Returns:
            The diagnostics.
        """
        values = self.scalars()
        return ChainDiagnostics(
            self._record_names,
            np.array([integrated_autocorrelation_time(v) for v in values]),
            np.array([effective_sample_size(v) for v in values]),
            np.array([split_rhat(v) for v in values]),
        )

    def scalar_means(self) -> list[Estimate]:
        """The mean of each scalar, with its Monte Carlo standard error.

        The error is ``sigma sqrt(tau / n)`` over all the chains' kept steps;
        the run length needed for a target error follows from it.

        Returns:
            One ``Estimate`` per scalar, the potential first.
        """
        values = self.scalars()
        estimates = []
        for v in values:
            total = v.size
            tau = integrated_autocorrelation_time(v)
            error = float(np.std(v, ddof=1)) * np.sqrt(tau / total)
            estimates.append(Estimate(float(np.mean(v)), error, total))
        return estimates

    @property
    def stored_positions(self) -> tuple:
        """Every stored state, chain after chain."""
        return tuple(x for chain in self._chains for x in chain.positions)

    def sample_thinned(self, *, rng: Generator | None = None) -> Any:
        """One of the stored states, chosen uniformly.

        No forward evaluation is made. The stored states are exact in the
        long-run limit but correlated with one another, by the chains'
        autocorrelation over the thinning interval; for a draw independent
        of the others, use :meth:`sample`.

        Args:
            rng: the generator for the choice.

        Returns:
            A stored model.

        Raises:
            ValueError: if no states were stored.
        """
        stored = self.stored_positions
        if not stored:
            raise ValueError("No states were stored; run the chains with thin= set.")
        return stored[int(resolve_rng(rng).integers(len(stored)))]

    def sample(self, *, rng: Generator | None = None) -> Any:
        """A fresh draw, independent of every other: not yet available.

        A draw is a short chain run from a reference fitted to this run, for
        a number of steps set from the diagnostics, returning its last
        state; the fitted reference is not built yet.

        Args:
            rng: the generator.

        Raises:
            NotImplementedError: always, until the fitted reference exists;
                :meth:`sample_thinned` reads the stored states meanwhile.
        """
        raise NotImplementedError(
            "A fresh draw is a short chain from a reference fitted to the run, "
            "which is not built yet; sample_thinned reads the stored states."
        )

    def __repr__(self) -> str:
        steps = sum(chain.steps for chain in self._chains)
        return f"SampledPosterior({len(self._chains)} chains, {steps} steps)"


class WeightedEnsemble(AbsolutelyContinuousMeasure):
    """Weighted particles, with the log evidence they estimate.

    The density against the prior is the target's, ``-Phi / T``, and the
    normalizer is the tempering path's estimate of ``log Z``, with no
    standard error: one run of a sequential Monte Carlo sampler does not
    estimate its own.

    Args:
        prior: the prior the density is taken against.
        relative_log_density: ``-Phi / T``.
        particles: the particles.
        log_weights: their unnormalized log weights.
        log_normalizer: the estimate of ``log Z``.
        temperatures: the inverse temperatures ``beta`` the path stopped at.
        effective_sizes: the effective sample size before each resampling.
    """

    def __init__(
        self,
        prior: GaussianMeasure,
        relative_log_density: Functional,
        particles: Sequence[Any],
        log_weights: np.ndarray,
        /,
        *,
        log_normalizer: Estimate,
        temperatures: np.ndarray,
        effective_sizes: np.ndarray,
    ) -> None:
        super().__init__(prior, relative_log_density, log_normalizer=log_normalizer)
        self._particles = tuple(particles)
        self._log_weights = np.asarray(log_weights, dtype=float)
        self._temperatures = np.asarray(temperatures, dtype=float)
        self._effective_sizes = np.asarray(effective_sizes, dtype=float)

    @property
    def particles(self) -> tuple:
        """The particles."""
        return self._particles

    @property
    def weights(self) -> np.ndarray:
        """The normalized weights."""
        shifted = np.exp(self._log_weights - self._log_weights.max())
        return shifted / shifted.sum()

    @property
    def effective_sample_size(self) -> float:
        """``1 / sum w^2`` of the normalized weights."""
        return float(1.0 / np.sum(self.weights**2))

    @property
    def temperatures(self) -> np.ndarray:
        """The inverse temperatures of the path, from the first above zero to one."""
        return self._temperatures.copy()

    @property
    def effective_sizes(self) -> np.ndarray:
        """The effective sample size at each stage, before resampling."""
        return self._effective_sizes.copy()

    def weighted_mean(self, function: Any, /) -> float:
        """The weighted mean of a scalar function of the particles.

        Args:
            function: a callable from a model to a number, a functional say.

        Returns:
            The mean.
        """
        values = np.array([float(function(x)) for x in self._particles])
        return float(self.weights @ values)

    def __repr__(self) -> str:
        return (
            f"WeightedEnsemble({len(self._particles)} particles, "
            f"{self._temperatures.size} stages)"
        )
