"""
The population layer: independent chains, or a tempered population.

Two geometries on measures, two ways of running a kernel:

* :class:`Chains` runs independent chains, each moving its probability by
  transport, the Wasserstein side. It returns a
  :class:`~pygeoinf.inference.sampling.results.SampledPosterior`.
* :class:`Tempering` moves a population of particles along the tempering
  path ``exp(-beta Phi)`` from the prior, reweighting at each stage (the
  Fisher-Rao side, and the one route to the evidence), resampling, and
  mutating with a kernel built for the new ``beta``. It returns a
  :class:`~pygeoinf.inference.sampling.results.WeightedEnsemble`.

Both seed per item: each chain, and each particle at each stage, gets its
own generator spawned from the caller's, so a result is a function of the
seed alone and does not change with ``n_jobs``. Both run their independent
items in worker processes through :func:`~pygeoinf.parallel.parallel_map`,
each chain serial within its worker; an operator that is itself parallel,
an MPI-split solve say, runs with ``n_jobs`` unset. The library does no I/O:
checkpointing a long run is the callback's business.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from numpy.random import Generator

from ...algebra.operators import LinearOperator
from ...algebra.spaces import CoordinateSpace, resolve_rng
from ...parallel import parallel_map
from ...probability.base import ProbabilityMeasure
from ...numerics.randomized import Estimate
from .corrections import Kernel, Step
from .results import Chain, SampledPosterior, WeightedEnsemble
from .target import State, Target

__all__ = ["Chains", "StepSizeAdaptation", "Tempering"]


class StepSizeAdaptation:
    """Robbins-Monro adaptation of ``log h`` towards a target acceptance.

    After each burn-in step, ``log h += rate (n + 1)^-decay (alpha - target)``
    with ``alpha`` the step's acceptance probability: a step that accepts
    too often grows, one that rejects too often shrinks, by amounts that
    diminish so that the step settles. It acts during burn-in only; the
    chain then runs at the step it reached, so the kept steps come from one
    fixed kernel.

    A reference that is the posterior accepts every step whatever its size,
    and the step then grows without settling; *maximum* stops it where a
    longer step gains nothing, a rotation of a quarter turn say.

    Args:
        acceptance: the acceptance probability to aim for, in ``(0, 1)``:
            about a quarter for pCN, 0.57 for MALA, 0.65 to 0.8 for HMC.
        rate: the gain of the first update.
        decay: how fast the gain falls, in ``(0.5, 1]``.
        maximum: the largest step the adaptation reaches.

    Raises:
        ValueError: for an acceptance outside ``(0, 1)``, a non-positive
            rate or maximum, or a decay outside ``(0.5, 1]``.
    """

    def __init__(
        self,
        *,
        acceptance: float,
        rate: float = 1.0,
        decay: float = 0.6,
        maximum: float = float("inf"),
    ) -> None:
        if not 0.0 < acceptance < 1.0:
            raise ValueError(f"A target acceptance lies in (0, 1), got {acceptance}.")
        if not rate > 0.0:
            raise ValueError(f"The rate is positive, got {rate}.")
        if not 0.5 < decay <= 1.0:
            raise ValueError(f"The decay lies in (0.5, 1], got {decay}.")
        if not maximum > 0.0:
            raise ValueError(f"The largest step is positive, got {maximum}.")
        self._acceptance = float(acceptance)
        self._rate = float(rate)
        self._decay = float(decay)
        self._maximum = float(maximum)

    @property
    def acceptance(self) -> float:
        """The acceptance probability aimed for."""
        return self._acceptance

    @property
    def maximum(self) -> float:
        """The largest step the adaptation reaches."""
        return self._maximum

    def update(self, step_size: float, iteration: int, acceptance: float, /) -> float:
        """The step after one burn-in step.

        Args:
            step_size: the current step.
            iteration: the burn-in step just taken, from zero.
            acceptance: its acceptance probability.

        Returns:
            The new step, at most :attr:`maximum`.
        """
        gain = self._rate / (iteration + 1.0) ** self._decay
        step = np.exp(np.log(step_size) + gain * (acceptance - self._acceptance))
        return float(min(step, self._maximum))

    def __repr__(self) -> str:
        return f"StepSizeAdaptation(acceptance={self._acceptance:g})"


class Chains:
    """Independent chains of one kernel.

    Args:
        count: how many chains.

    Raises:
        ValueError: for a count below one.
    """

    def __init__(self, count: int, /) -> None:
        if int(count) != count or count < 1:
            raise ValueError(f"A number of chains is a positive integer, got {count}.")
        self._count = int(count)

    @property
    def count(self) -> int:
        """How many chains."""
        return self._count

    def __call__(
        self,
        kernel: Kernel,
        /,
        *,
        start: Any,
        steps: int,
        burn_in: int = 0,
        adapt: StepSizeAdaptation | Sequence[StepSizeAdaptation | None] | None = None,
        thin: int | None = None,
        record: LinearOperator | None = None,
        record_names: list[str] | None = None,
        rng: Generator | None = None,
        n_jobs: int | None = None,
        callback: Callable[[int, int, Step], None] | None = None,
    ) -> SampledPosterior:
        """Run the chains.

        Every kept step records the likelihood potential, whether the
        proposal was accepted and its acceptance probability, and the
        scalars of *record*; the full position is stored every *thin* steps.

        Args:
            kernel: what each chain runs.
            start: where the chains start: a model, the same for every
                chain, or a measure on the model space each chain draws its
                start from, the prior or a reference say.
            steps: how many steps each chain keeps, after its burn-in.
            burn_in: how many steps each chain takes first and discards.
            adapt: how the step adapts during burn-in, towards its target
                acceptance; fixed if omitted. For a kernel of several blocks
                (:attr:`~pygeoinf.inference.sampling.corrections.Kernel.step_sizes`),
                one adaptation per block, each towards its own acceptance,
                ``None`` keeping a block's step fixed; one adaptation alone
                serves every block.
            thin: store the position every this many kept steps; nothing
                is stored if omitted, which leaves
                :meth:`~pygeoinf.inference.sampling.results.SampledPosterior.sample_thinned`
                nothing to read.
            record: a linear map from the model space to a space with
                coordinates, small, whose components are recorded at every
                step: chart coordinates, or the averages a property operator
                takes.
            record_names: names for the recorded components.
            callback: called as ``callback(chain, iteration, step)`` after
                every step, burn-in included (iterations below *burn_in*),
                where the chain runs: in a worker process when *n_jobs* asks
                for one, so it should write rather than accumulate.

        Returns:
            The sampled posterior.

        Raises:
            ValueError: for a negative step count or burn-in, a thinning
                interval below one, a start at which the potential cannot be
                evaluated, or a recording map that does not act on the model
                space.
            TypeError: if the recording map's codomain has no coordinates.
        """
        if steps < 1 or burn_in < 0:
            raise ValueError(
                "A chain keeps at least one step and burns in none or more."
            )
        if thin is not None and thin < 1:
            raise ValueError(f"A thinning interval is at least one, got {thin}.")
        target = kernel.target
        if record is not None:
            if record.domain != target.model_space:
                raise ValueError("The recording map must act on the model space.")
            if not isinstance(record.codomain, CoordinateSpace):
                raise TypeError(
                    "The recording map's codomain needs coordinates, which are "
                    "what is recorded."
                )
        streams = resolve_rng(rng).spawn(self._count)

        def run(item: tuple[int, Generator]) -> Chain:
            index, stream = item
            return _run_chain(
                kernel,
                index,
                stream,
                start=start,
                steps=steps,
                burn_in=burn_in,
                adapt=adapt,
                thin=thin,
                record=record,
                callback=callback,
            )

        chains = parallel_map(run, list(enumerate(streams)), n_jobs=n_jobs)
        return SampledPosterior(
            target.prior,
            target.log_density_against_prior(),
            chains,
            kernel=kernel,
            record=record,
            record_names=record_names,
        )

    def __repr__(self) -> str:
        return f"Chains({self._count})"


def _run_chain(
    kernel: Kernel,
    index: int,
    rng: Generator,
    /,
    *,
    start: Any,
    steps: int,
    burn_in: int,
    adapt: Any,
    thin: int | None,
    record: LinearOperator | None,
    callback: Callable[[int, int, Step], None] | None,
) -> Chain:
    """One chain, burned in and then recorded.

    Raises:
        ValueError: if the potential cannot be evaluated at the start.
    """
    target = kernel.target
    position = start.sample(rng=rng) if isinstance(start, ProbabilityMeasure) else start
    state = kernel.initial_state(position, rng=rng)
    if not np.isfinite(target.value(state)):
        raise ValueError(
            "The potential cannot be evaluated at the chain's start; start "
            "where the forward operator evaluates."
        )
    step_sizes = list(kernel.step_sizes)
    adapters = _adapters(adapt, len(step_sizes))
    for iteration in range(burn_in):
        step = kernel.step(state, rng=rng)
        state = step.state
        if adapters is not None:
            acceptances = step.info.acceptances or (step.info.acceptance,)
            for block, (adapter, acceptance) in enumerate(zip(adapters, acceptances)):
                if adapter is not None:
                    step_sizes[block] = adapter.update(
                        step_sizes[block], iteration, acceptance
                    )
            kernel = kernel.with_steps(step_sizes)
        if callback is not None:
            callback(index, iteration, step)

    potential = np.empty(steps)
    accepted = np.empty(steps, dtype=bool)
    acceptance = np.empty(steps)
    records = None
    space = None
    if record is not None:
        space = record.codomain
        records = np.empty((steps, space.dim))
    positions = []
    blocks = None
    if len(step_sizes) > 1:
        blocks = np.empty((steps, len(step_sizes)))
    for kept in range(steps):
        step = kernel.step(state, rng=rng)
        state = step.state
        potential[kept] = target.potential(state)
        accepted[kept] = step.info.accepted
        acceptance[kept] = step.info.acceptance
        if blocks is not None and step.info.acceptances is not None:
            blocks[kept] = step.info.acceptances
        if records is not None:
            records[kept] = space.to_components(record(state.position))
        if thin is not None and (kept + 1) % thin == 0:
            positions.append(state.position)
        if callback is not None:
            callback(index, burn_in + kept, step)
    return Chain(
        potential,
        accepted,
        acceptance,
        records,
        tuple(positions),
        step_sizes[0],
        step_sizes=tuple(step_sizes),
        block_acceptance=blocks,
    )


def _adapters(adapt: Any, count: int, /) -> list | None:
    """One adaptation per step of a kernel: the one given, for every step, or
    a sequence of them, ``None`` keeping a step fixed.

    Raises:
        ValueError: for a sequence of the wrong length.
    """
    if adapt is None:
        return None
    if isinstance(adapt, StepSizeAdaptation):
        return [adapt] * count
    adapters = list(adapt)
    if len(adapters) != count:
        raise ValueError(
            f"Expected {count} adaptations, one per block of the kernel; got {len(adapters)}."
        )
    return adapters


class Tempering:
    """A population moved along the tempering path from the prior.

    The path is ``d mu_beta / d mu_0 ~ exp(-beta Phi / T)`` for ``beta``
    from zero to one, each measure on it equivalent to the prior, so the
    method is defined on a function space and its number of stages is set
    by the information in the data, not by the mesh. At each stage the next
    ``beta`` is chosen so that the effective sample size of the reweighted
    particles is a fixed fraction of their number, which keeps the
    Fisher-Rao speed along the path constant; the particles are reweighted,
    resampled systematically, and each moved by the mutation kernel, rebuilt
    for the new ``beta``, for a few steps. The product of the stages' mean
    weights estimates the evidence.

    The mutation steps are what keep the particles diverse, and adapting
    them with the particles' covariance would put a measure of rank below
    the number of particles in the preconditioner, singular to the prior;
    so the kernel is the caller's, at a fixed step. Tempering from a
    reference other than the prior is not offered.

    Args:
        particles: how many particles.
        mutation: the kernel that moves each particle, built for the final
            target and rebuilt for each stage's.
        schedule: how ``beta`` advances; ``"ess"``, the one offered.
        ess_fraction: the effective sample size, as a fraction of the
            particles, that sets each step in ``beta``.
        mutation_steps: how many kernel steps each particle takes per
            stage.

    Raises:
        ValueError: for fewer than two particles, an unknown schedule, a
            fraction outside ``(0, 1)``, or a step count below one.
    """

    def __init__(
        self,
        particles: int,
        /,
        *,
        mutation: Kernel,
        schedule: str = "ess",
        ess_fraction: float = 0.5,
        mutation_steps: int = 1,
    ) -> None:
        if int(particles) != particles or particles < 2:
            raise ValueError(
                f"A population has at least two particles, got {particles}."
            )
        if schedule != "ess":
            raise ValueError(f"The schedule offered is 'ess', not {schedule!r}.")
        if not 0.0 < ess_fraction < 1.0:
            raise ValueError(f"The fraction lies in (0, 1), got {ess_fraction}.")
        if int(mutation_steps) != mutation_steps or mutation_steps < 1:
            raise ValueError(
                f"A particle takes at least one step, got {mutation_steps}."
            )
        self._particles = int(particles)
        self._mutation = mutation
        self._ess_fraction = float(ess_fraction)
        self._mutation_steps = int(mutation_steps)

    @property
    def particles(self) -> int:
        """How many particles."""
        return self._particles

    def __call__(
        self,
        target: Target,
        /,
        *,
        rng: Generator | None = None,
        n_jobs: int | None = None,
    ) -> WeightedEnsemble:
        """Move the population from the prior to the target.

        Args:
            target: the target at the end of the path, whose reference must
                be the prior.

        Returns:
            The weighted ensemble, with the estimate of ``log Z`` as its
            normalizer.

        Raises:
            ValueError: if the target's reference is not the prior, or
                follows the state.
        """
        if target.reference.rank != 0 or target.is_local:
            raise ValueError(
                "Tempering starts from the prior; a target whose reference is "
                "another Gaussian is not offered."
            )
        rng = resolve_rng(rng)
        count = self._particles
        temperature = target.temperature
        prior = target.prior
        positions = prior.samples(count, rng=rng.spawn(1)[0], n_jobs=n_jobs)
        potentials = np.array(
            parallel_map(
                lambda x: target.potential(State(x)), positions, n_jobs=n_jobs
            ),
            dtype=float,
        )
        beta = 0.0
        log_z = 0.0
        log_weights = np.zeros(count)
        temperatures = []
        effective = []
        while beta < 1.0:
            scaled = potentials / temperature
            delta = self._next_step(log_weights, scaled, 1.0 - beta)
            updated = log_weights - _times(delta, scaled)
            log_z += _logsumexp(updated) - _logsumexp(log_weights)
            beta = 1.0 if delta >= 1.0 - beta else beta + delta
            weights = np.exp(updated - updated.max())
            weights /= weights.sum()
            effective.append(1.0 / float(np.sum(weights**2)))
            temperatures.append(beta)
            chosen = _systematic(weights, rng)
            positions = [positions[i] for i in chosen]
            log_weights = np.zeros(count)
            kernel = self._mutation.with_target(
                target.with_temperature(temperature / beta)
            )
            streams = rng.spawn(count)
            moved = parallel_map(
                lambda item: _mutate(kernel, item[0], item[1], self._mutation_steps),
                list(zip(positions, streams)),
                n_jobs=n_jobs,
            )
            positions = [position for position, _ in moved]
            potentials = np.array([value for _, value in moved], dtype=float)
        return WeightedEnsemble(
            prior,
            target.log_density_against_prior(),
            positions,
            log_weights,
            log_normalizer=Estimate(float(log_z), float("nan"), count),
            temperatures=np.array(temperatures),
            effective_sizes=np.array(effective),
        )

    def _next_step(
        self, log_weights: np.ndarray, scaled: np.ndarray, remaining: float
    ) -> float:
        """The step in ``beta`` that brings the effective sample size to the
        set fraction, or the whole of what remains if it does not fall that
        far."""
        goal = self._ess_fraction * log_weights.size

        def ess(delta: float) -> float:
            updated = log_weights - _times(delta, scaled)
            if not np.any(np.isfinite(updated)):
                return 0.0
            weights = np.exp(updated - np.max(updated[np.isfinite(updated)]))
            return float(weights.sum() ** 2 / np.sum(weights**2))

        if ess(remaining) >= goal:
            return remaining
        low, high = 0.0, remaining
        for _ in range(60):
            middle = 0.5 * (low + high)
            if ess(middle) >= goal:
                low = middle
            else:
                high = middle
        return max(low, 1e-12 * remaining)

    def __repr__(self) -> str:
        return f"Tempering({self._particles}, mutation={self._mutation!r})"


def _mutate(
    kernel: Kernel, position: Any, rng: Generator, steps: int
) -> tuple[Any, float]:
    """A particle moved by a kernel, and its potential where it lands."""
    state = kernel.initial_state(position, rng=rng)
    for _ in range(steps):
        state = kernel.step(state, rng=rng).state
    return state.position, kernel.target.potential(state)


def _times(delta: float, values: np.ndarray, /) -> np.ndarray:
    """``delta * values`` with an infinite value kept infinite at ``delta == 0``."""
    return np.where(np.isinf(values), np.inf, delta * values)


def _logsumexp(values: np.ndarray, /) -> float:
    """``log sum exp(values)``, safe for large and infinite entries."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -np.inf
    top = float(finite.max())
    return top + float(np.log(np.sum(np.exp(finite - top))))


def _systematic(weights: np.ndarray, rng: Generator, /) -> np.ndarray:
    """Systematic resampling: the indices of ``len(weights)`` draws."""
    count = weights.size
    points = (rng.random() + np.arange(count)) / count
    indices = np.searchsorted(np.cumsum(weights), points)
    return np.minimum(indices, count - 1)
