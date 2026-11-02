"""
Splitting schemes: words over a dynamics' exact flows.

A word is stored in application order -- ``"OBABO"`` applies ``O`` first --
as a string, in which a letter met ``k`` times takes ``1/k`` of the step each
time, or as ``(letter, fraction)`` pairs. Each letter's fractions must sum to
one, so that over a step every piece of the dynamics runs for the step. Any
word is accepted: HMC's ``"OBAB"`` and MALA's ``"BO"`` are not palindromes.
What must be one is a Metropolized block, which a velocity flip has to
invert, and the correction checks that.

A scheme is not a sampler on its own: it is biased by its step, and only a
correction (:mod:`~pygeoinf.inference.sampling.corrections`) turns it into a
kernel a population can run. Unadjusted schemes are not offered.

Adjacent kicks need no merging for cost: the gradient one computes is cached
on the state, so the next reads it.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.random import Generator

from .dynamics import Dynamics
from .flows import Flow, GradientDrift, PriorOU
from .target import State

__all__ = ["SplittingScheme"]


class SplittingScheme:
    """A word over a dynamics' letters, applied with a time step.

    Args:
        dynamics: the dynamics whose letters the word is written in.
        word: the letters in application order: a string, each letter
            taking an equal share of the step for every time it appears, or
            a sequence of ``(letter, fraction)`` pairs.
        step: the time step ``h``. Each letter runs for its fraction of it,
            and each flow turns its time into its own angle.
        repeat: how many times the word is applied per call; an HMC block
            of ``L`` leapfrog steps is ``"BAB"`` repeated ``L`` times.

    Raises:
        ValueError: for an empty word, a letter the dynamics does not have,
            a fraction that is not positive, a letter whose fractions do not
            sum to one, a step that is not positive and finite, or a repeat
            below one.
    """

    def __init__(
        self,
        dynamics: Dynamics,
        word: str | Sequence[tuple[str, float]],
        /,
        *,
        step: float,
        repeat: int = 1,
    ) -> None:
        pairs = _parse(word)
        flows = dynamics.flows()
        unknown = sorted({letter for letter, _ in pairs} - set(flows))
        if unknown:
            raise ValueError(
                f"{type(dynamics).__name__} has the letters {sorted(flows)}, not "
                f"{unknown}."
            )
        totals: dict[str, float] = {}
        for letter, fraction in pairs:
            totals[letter] = totals.get(letter, 0.0) + fraction
        uneven = {letter: total for letter, total in totals.items() if abs(total - 1.0) > 1e-12}
        if uneven:
            raise ValueError(
                f"Each letter's fractions must sum to one over the word; got {uneven}."
            )
        if not 0.0 < step < float("inf"):
            raise ValueError(f"A step is positive and finite, got {step}.")
        if int(repeat) != repeat or repeat < 1:
            raise ValueError(f"A word is repeated a whole number of times, got {repeat}.")
        self._dynamics = dynamics
        self._pairs = pairs
        self._step = float(step)
        self._repeat = int(repeat)
        self._flows = [(flows[letter], fraction) for letter, fraction in pairs]

    # ----------------------------------------------------------------- #
    #                              Reading                              #
    # ----------------------------------------------------------------- #

    @property
    def dynamics(self) -> Dynamics:
        """The dynamics the word is written over."""
        return self._dynamics

    @property
    def word(self) -> tuple[tuple[str, float], ...]:
        """The letters with their fractions, in application order."""
        return tuple(self._pairs)

    @property
    def step(self) -> float:
        """The time step ``h``."""
        return self._step

    @property
    def repeat(self) -> int:
        """How many times the word is applied per call."""
        return self._repeat

    @property
    def is_palindrome(self) -> bool:
        """Whether the word reads the same backwards."""
        return self._pairs == self._pairs[::-1]

    @property
    def flows(self) -> tuple[tuple[Flow, float], ...]:
        """The flows with their fractions, in application order."""
        return tuple(self._flows)

    @property
    def is_deterministic(self) -> bool:
        """Whether every flow in the word is deterministic."""
        return all(flow.deterministic for flow, _ in self._flows)

    # ----------------------------------------------------------------- #
    #                             Applying                              #
    # ----------------------------------------------------------------- #

    def apply(
        self, state: State, /, *, rng: Generator | None = None
    ) -> tuple[State, float | None]:
        """Apply the word, ``repeat`` times.

        Stops early, with an energy change of ``+inf``, once a flow meets a
        position where the potential cannot be evaluated.

        Args:
            state: where to start.
            rng: the generator for the stochastic flows.

        Returns:
            The final state and the summed change in the reference's
            Gaussian energy, or ``None`` if a stochastic flow took part.
        """
        energy: float | None = 0.0
        for _ in range(self._repeat):
            for flow, fraction in self._flows:
                state, change = flow.apply(state, fraction * self._step, rng=rng)
                if change is None:
                    energy = None
                elif energy is not None:
                    energy += change
                if change == float("inf"):
                    return state, float("inf")
        return state, energy

    # ----------------------------------------------------------------- #
    #                           New schemes                             #
    # ----------------------------------------------------------------- #

    def block(self, word: str | Sequence[tuple[str, float]], /, *, repeat: int = 1) -> "SplittingScheme":
        """A sub-word on the same dynamics and step, for a correction.

        Args:
            word: the sub-word, read as for the constructor.
            repeat: how many times the block is applied.

        Returns:
            The block, itself a scheme.
        """
        return SplittingScheme(
            self._dynamics,
            word,
            step=self._step,
            repeat=repeat,
        )

    def with_step(self, step: float, /) -> "SplittingScheme":
        """The same word at another step.

        Args:
            step: the new time step.

        Returns:
            A new scheme; this one is unchanged.
        """
        return SplittingScheme(
            self._dynamics,
            self._pairs,
            step=step,
            repeat=self._repeat,
        )

    def with_dynamics(self, dynamics: Dynamics, /) -> "SplittingScheme":
        """The same word over another dynamics with the same letters.

        Args:
            dynamics: the new dynamics, a tempered copy of this one say.

        Returns:
            A new scheme.
        """
        return SplittingScheme(
            dynamics,
            self._pairs,
            step=self._step,
            repeat=self._repeat,
        )

    # ----------------------------------------------------------------- #
    #                        Transition densities                       #
    # ----------------------------------------------------------------- #

    @property
    def has_transition_density(self) -> bool:
        """Whether :meth:`log_transition_ratio` can be stated for this word.

        True for a word of prior Ornstein-Uhlenbeck steps alone, which is
        reversible with respect to the reference, and for a single drift
        followed by one such step, the overdamped MALA proposal.
        """
        return self._transition_kind() is not None

    def _transition_kind(self) -> str | None:
        """Which of the words with a known transition density this is."""
        kinds = [type(flow) for flow, _ in self._flows]
        if kinds and all(kind is PriorOU for kind in kinds):
            return "reversible"
        if self._repeat == 1 and kinds == [GradientDrift, PriorOU]:
            return "drift"
        return None

    def log_transition_ratio(self, state: State, proposal: State, /) -> float | None:
        """``log q(v, u) / q(u, v)`` against the reference's own kernel.

        For a proposal ``v`` from ``u``, the log ratio of the reverse and
        forward transition densities, each relative to the reference's
        Ornstein-Uhlenbeck kernel ``Q``, which is reversible with respect
        to the reference: the term a Metropolis-Hastings step adds to
        ``Phi_ref(u) - Phi_ref(v)``. Zero for a word of prior
        Ornstein-Uhlenbeck steps. For a drift for time ``s`` followed by a
        step of angle ``a``, the proposal is ``Q`` from the drifted point,
        a Cameron-Martin shift of ``Q`` from ``u`` by ``-c s K g``, with
        ``c, b == cos(a), sin(a)`` and ``g`` the gradient at ``u``, so

        .. code-block:: text

            log q(u, v) / Q(u, v) == -(c s / b^2) (g, v - m - c (u - m))
                                     - (c s)^2 / (2 b^2) (g, K g)

        finite on a function space, with no inverse covariance.

        Args:
            state: the current state.
            proposal: the proposed state.

        Returns:
            The log ratio, or ``None`` for a word whose transition density
            is not known.
        """
        kind = self._transition_kind()
        if kind is None:
            return None
        if kind == "reversible":
            return 0.0
        (_, drift_fraction), (ou, ou_fraction) = self._flows
        drift_time = drift_fraction * self._step
        angle = ou.angle(ou_fraction * self._step)
        c, b = float(np.cos(angle)), float(np.sin(angle))
        return self._drift_density(proposal, state, drift_time, c, b) - self._drift_density(
            state, proposal, drift_time, c, b
        )

    def _drift_density(
        self, origin: State, destination: State, drift_time: float, c: float, b: float
    ) -> float:
        """``log q(origin, destination) / Q(origin, destination)``."""
        target = self._dynamics.target
        space = target.model_space
        mean = target.component(origin).expectation
        gradient = target.gradient(origin)
        preconditioned = target.preconditioned_gradient(origin)
        centre = space.axpy(
            c, space.subtract(origin.position, mean), space.copy(mean)
        )
        offset = space.subtract(destination.position, centre)
        scale = c * drift_time
        return (
            -(scale / b**2) * space.inner_product(gradient, offset)
            - (scale**2 / (2.0 * b**2)) * space.inner_product(gradient, preconditioned)
        )

    def __repr__(self) -> str:
        repeat = f", repeat={self._repeat}" if self._repeat != 1 else ""
        return f"SplittingScheme({_spell(self._pairs)!r}, step={self._step:g}{repeat})"


def _parse(word: str | Sequence[tuple[str, float]], /) -> list[tuple[str, float]]:
    """A word as ``(letter, fraction)`` pairs.

    Raises:
        ValueError: for an empty word or a fraction that is not positive.
    """
    if isinstance(word, str):
        letters = [letter for letter in word if not letter.isspace()]
        if not letters:
            raise ValueError("A word needs at least one letter.")
        counts = {letter: letters.count(letter) for letter in set(letters)}
        return [(letter, 1.0 / counts[letter]) for letter in letters]
    pairs = [(str(letter), float(fraction)) for letter, fraction in word]
    if not pairs:
        raise ValueError("A word needs at least one letter.")
    bad = [pair for pair in pairs if not pair[1] > 0.0]
    if bad:
        raise ValueError(f"Every fraction of the step is positive; got {bad}.")
    return pairs


def _spell(pairs: Sequence[tuple[str, float]], /) -> str:
    """The letters of a word, for a message."""
    return "".join(letter for letter, _ in pairs)


def _locate(outer: Sequence[Any], inner: Sequence[Any], /) -> int | None:
    """Where ``inner`` first occurs contiguously in ``outer``, or ``None``."""
    width = len(inner)
    for start in range(len(outer) - width + 1):
        if list(outer[start : start + width]) == list(inner):
            return start
    return None
