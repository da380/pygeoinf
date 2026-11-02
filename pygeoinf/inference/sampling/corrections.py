"""
Metropolis corrections: what turns a splitting scheme into a kernel.

Every sampler here is corrected, so it targets the posterior exactly
whatever its step. There are two corrections, because which one applies
depends on the word and a single wrapper would hide it:

* :class:`MetropolizedBlock`, for a deterministic block that a velocity flip
  inverts, a palindrome of rotations and kicks such as ``"BAB"`` repeated.
  The proposal is the block followed by the flip, an involution, so the
  acceptance is ``exp(-dE)`` with ``dE`` the change in ``Phi_ref`` plus the
  change in the reference's Gaussian energy that the flows report. On a
  rejection the velocity is flipped, which is generalized HMC; with a full
  refresh that is immaterial. Stochastic letters sit outside the block and
  must preserve the target.
* :class:`MetropolisHastings`, for a stochastic proposal whose transition
  density is known relative to the reference's own kernel: pCN and its
  generalization over any reference, the independence sampler, and the
  overdamped MALA proposal.

A kernel's :meth:`Kernel.step` returns a :class:`Step`: the new state, a
log-weight increment (zero for a kernel that preserves its target, which
both of these do) and what the step measured.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator

from ...algebra.spaces import resolve_rng
from .dynamics import Dynamics
from .flows import TARGET, Flow
from .schemes import SplittingScheme, _locate
from .target import State, Target

__all__ = ["Kernel", "MetropolisHastings", "MetropolizedBlock", "Step", "StepInfo"]


@dataclass(frozen=True)
class StepInfo:
    """What one step of a kernel measured.

    Attributes:
        accepted: whether the proposal was accepted.
        acceptance: the acceptance probability, ``min(1, exp(log alpha))``.
        energy_error: ``-log alpha``, the error the correction removed:
            for a Metropolized block, the change in the shadow energy.
    """

    accepted: bool
    acceptance: float
    energy_error: float


@dataclass(frozen=True, eq=False)
class Step:
    """One step of a kernel.

    Attributes:
        state: the state after the step.
        log_weight: the log-weight increment the step carries: zero for a
            kernel that leaves its target invariant.
        info: what the step measured.
    """

    state: State
    log_weight: float
    info: StepInfo


class Kernel(ABC):
    """A Markov kernel that leaves its target invariant.

    Made by a correction from a splitting scheme; the population layer runs
    nothing else.
    """

    @property
    @abstractmethod
    def dynamics(self) -> Dynamics:
        """The dynamics the kernel's scheme is written over."""

    @property
    def target(self) -> Target:
        """The target the kernel leaves invariant."""
        return self.dynamics.target

    @property
    @abstractmethod
    def step_size(self) -> float:
        """The scheme's time step ``h``."""

    @abstractmethod
    def step(self, state: State, /, *, rng: Generator | None = None) -> Step:
        """One step from a state.

        Args:
            state: where to start.
            rng: the generator.

        Returns:
            The step.
        """

    @abstractmethod
    def with_step(self, step: float, /) -> "Kernel":
        """The same kernel at another time step.

        Args:
            step: the new step.

        Returns:
            A new kernel.
        """

    @abstractmethod
    def with_target(self, target: Target, /) -> "Kernel":
        """The same kernel on another target, a tempered copy say.

        Args:
            target: the new target.

        Returns:
            A new kernel.
        """

    def initial_state(self, position: Any, /, *, rng: Generator | None = None) -> State:
        """A state at a position, with a velocity drawn where there is one.

        Args:
            position: the model.
            rng: the generator for the velocity.

        Returns:
            The state.
        """
        return self.dynamics.initial_state(position, rng=rng)


def _accept(log_alpha: float, rng: Generator, /) -> tuple[bool, float]:
    """Whether to accept at ``log alpha``, and the acceptance probability."""
    if not np.isfinite(log_alpha):
        accepted = bool(log_alpha > 0.0)
        return accepted, 1.0 if accepted else 0.0
    acceptance = float(np.exp(min(0.0, log_alpha)))
    return bool(np.log(rng.random()) < log_alpha), acceptance


class MetropolizedBlock:
    """The correction of a deterministic, flip-reversible block.

    Args:
        block: a scheme of deterministic flows each reversible under a
            velocity flip, with a palindromic word; from
            :meth:`~pygeoinf.inference.sampling.schemes.SplittingScheme.block`.
        flip: reverse the velocity on a rejection. Needed for exactness
            whenever the velocity is kept between steps.

    Raises:
        ValueError: for a block with a stochastic or irreversible flow, a
            word that is not a palindrome, or a dynamics with no velocity.
    """

    def __init__(self, block: SplittingScheme, /, *, flip: bool = True) -> None:
        if not block.dynamics.has_velocity:
            raise ValueError(
                "A Metropolized block flips a velocity, and "
                f"{type(block.dynamics).__name__} has none; correct it with "
                "MetropolisHastings."
            )
        offending = [
            repr(flow)
            for flow, _ in block.flows
            if not (flow.deterministic and flow.reversible)
        ]
        if offending:
            raise ValueError(
                "A Metropolized block holds deterministic flows that a velocity "
                f"flip reverses; {', '.join(offending)} are not. Stochastic "
                "letters belong outside the block."
            )
        if not block.is_palindrome:
            raise ValueError(
                "A Metropolized block must be a palindrome, so that the flip "
                "turns it into its own inverse."
            )
        self._block = block
        self._flip = bool(flip)

    @property
    def block(self) -> SplittingScheme:
        """The deterministic block."""
        return self._block

    @property
    def flip(self) -> bool:
        """Whether a rejection reverses the velocity."""
        return self._flip

    def within(self, scheme: SplittingScheme, /) -> Kernel:
        """The kernel that runs a scheme with this block corrected.

        The block's word must appear in the scheme's word, once,
        contiguously and with the same fractions; the letters before and
        after it run uncorrected and must preserve the target, which a
        momentum refresh does. The block runs its own ``repeat`` times in
        that place.

        Args:
            scheme: the outer scheme, over the block's dynamics.

        Returns:
            The kernel.

        Raises:
            ValueError: if the scheme is over another dynamics or repeated,
                does not contain the block's word, has a letter outside the
                block that does not preserve the target, or lacks the label
                jump over a mixture reference.
        """
        if scheme.dynamics is not self._block.dynamics:
            raise ValueError("The block and the scheme must share one dynamics.")
        _require_label_jump(scheme)
        if scheme.repeat != 1:
            raise ValueError(
                "Repeat the block, not the outer word: the block's own repeat "
                "is the trajectory length."
            )
        if scheme.step != self._block.step:
            raise ValueError("The block and the scheme must share one step.")
        start = _locate(scheme.word, self._block.word)
        if start is None:
            raise ValueError(
                f"The block {self._block.word!r} does not appear in the scheme's "
                f"word {scheme.word!r}."
            )
        flows = scheme.flows
        stop = start + len(self._block.word)
        outside = list(flows[:start]) + list(flows[stop:])
        unfit = [repr(flow) for flow, _ in outside if TARGET not in flow.preserves]
        if unfit:
            raise ValueError(
                "The letters outside a Metropolized block run uncorrected, so "
                f"each must preserve the target; {', '.join(unfit)} does not."
            )
        return _BlockKernel(scheme, self._block, start, flip=self._flip)

    def __repr__(self) -> str:
        return f"MetropolizedBlock({self._block!r}, flip={self._flip})"


class _BlockKernel(Kernel):
    """A scheme whose block is Metropolized and whose other letters are not."""

    def __init__(
        self, scheme: SplittingScheme, block: SplittingScheme, start: int, /, *, flip: bool
    ) -> None:
        self._scheme = scheme
        self._block = block
        self._start = start
        self._flip = flip
        flows = scheme.flows
        stop = start + len(block.word)
        self._before: list[tuple[Flow, float]] = list(flows[:start])
        self._after: list[tuple[Flow, float]] = list(flows[stop:])

    @property
    def dynamics(self) -> Dynamics:
        """The dynamics the scheme is written over."""
        return self._scheme.dynamics

    @property
    def scheme(self) -> SplittingScheme:
        """The outer scheme."""
        return self._scheme

    @property
    def block(self) -> SplittingScheme:
        """The corrected block."""
        return self._block

    @property
    def step_size(self) -> float:
        """The scheme's time step."""
        return self._scheme.step

    def _run(
        self, flows: list[tuple[Flow, float]], state: State, rng: Generator
    ) -> State:
        """Apply uncorrected flows in order."""
        step = self._scheme.step
        for flow, fraction in flows:
            state, _ = flow.apply(state, fraction * step, rng=rng)
        return state

    def step(self, state: State, /, *, rng: Generator | None = None) -> Step:
        """One step: the letters before the block, the corrected block, the rest.

        Args:
            state: where to start.
            rng: the generator.

        Returns:
            The step, with a log-weight of zero.
        """
        rng = resolve_rng(rng)
        target = self.dynamics.target
        state = self._run(self._before, state, rng)
        proposal, energy = self._block.apply(state, rng=rng)
        if energy is None or energy == float("inf"):
            error = float("inf")
        else:
            error = target.value(proposal) - target.value(state) + energy
        if not np.isfinite(error):
            error = float("inf")
        accepted, acceptance = _accept(-error, rng)
        if accepted:
            state = proposal
        elif self._flip:
            state = self.dynamics.flip(state)
        state = self._run(self._after, state, rng)
        return Step(state, 0.0, StepInfo(accepted, acceptance, error))

    def with_step(self, step: float, /) -> Kernel:
        """The same kernel at another time step.

        Args:
            step: the new step.

        Returns:
            A new kernel.
        """
        scheme = self._scheme.with_step(step)
        block = self._block.with_step(step)
        return _BlockKernel(scheme, block, self._start, flip=self._flip)

    def with_target(self, target: Target, /) -> Kernel:
        """The same kernel on another target.

        Args:
            target: the new target.

        Returns:
            A new kernel.
        """
        dynamics = self.dynamics.with_target(target)
        scheme = self._scheme.with_dynamics(dynamics)
        block = self._block.with_dynamics(dynamics)
        return _BlockKernel(scheme, block, self._start, flip=self._flip)

    def __repr__(self) -> str:
        return f"MetropolizedBlock({self._block!r}).within({self._scheme!r})"


class MetropolisHastings(Kernel):
    """A stochastic proposal corrected through its transition density.

    The proposal is the scheme's word less any letters before and after it
    that preserve the target, such as a mixture's label jump, which run
    uncorrected. It is accepted with
    ``log alpha == Phi_ref(u) - Phi_ref(v) + log q(v, u) / q(u, v)``, the
    last term taken relative to the reference's own kernel so that it is
    finite on a function space.

    Args:
        scheme: a scheme whose proposal's transition density is known: a
            word of prior Ornstein-Uhlenbeck steps (pCN and its
            generalization), or a drift followed by one such step (the
            overdamped MALA proposal), with target-preserving letters
            around it.

    Raises:
        ValueError: for a proposal whose transition density is not known, a
            dynamics that carries a velocity, or a mixture reference and a
            word without its label jump.
    """

    def __init__(self, scheme: SplittingScheme, /) -> None:
        if scheme.dynamics.has_velocity:
            raise ValueError(
                "Metropolis-Hastings here corrects a proposal on the position; "
                f"{type(scheme.dynamics).__name__} carries a velocity, so "
                "correct its deterministic block with MetropolizedBlock."
            )
        _require_label_jump(scheme)
        flows = scheme.flows
        first = 0
        while first < len(flows) and TARGET in flows[first][0].preserves:
            first += 1
        last = len(flows)
        while last > first and TARGET in flows[last - 1][0].preserves:
            last -= 1
        if first == last:
            raise ValueError(f"{scheme!r} has no proposal for a correction to accept.")
        proposal = scheme.block(scheme.word[first:last])
        if not proposal.has_transition_density:
            raise ValueError(
                f"The transition density of {proposal!r} is not known, so no "
                "Metropolis-Hastings ratio can be formed for it."
            )
        self._scheme = scheme
        self._proposal = proposal
        self._before: list[tuple[Flow, float]] = list(flows[:first])
        self._after: list[tuple[Flow, float]] = list(flows[last:])

    @property
    def dynamics(self) -> Dynamics:
        """The dynamics the scheme is written over."""
        return self._scheme.dynamics

    @property
    def scheme(self) -> SplittingScheme:
        """The whole word."""
        return self._scheme

    @property
    def proposal(self) -> SplittingScheme:
        """The part of the word that is corrected."""
        return self._proposal

    @property
    def step_size(self) -> float:
        """The scheme's time step."""
        return self._scheme.step

    def _run(self, flows: list[tuple[Flow, float]], state: State, rng: Generator) -> State:
        """Apply uncorrected flows in order."""
        step = self._scheme.step
        for flow, fraction in flows:
            state, _ = flow.apply(state, fraction * step, rng=rng)
        return state

    def step(self, state: State, /, *, rng: Generator | None = None) -> Step:
        """The letters before the proposal, one proposal accepted or not,
        and the letters after.

        Args:
            state: where to start.
            rng: the generator.

        Returns:
            The step, with a log-weight of zero.
        """
        rng = resolve_rng(rng)
        target = self.dynamics.target
        state = self._run(self._before, state, rng)
        proposal, _ = self._proposal.apply(state, rng=rng)
        proposed = target.value(proposal)
        if proposed == float("inf"):
            error = float("inf")
        else:
            ratio = self._proposal.log_transition_ratio(state, proposal)
            error = proposed - target.value(state) - ratio
        accepted, acceptance = _accept(-error, rng)
        state = self._run(self._after, proposal if accepted else state, rng)
        return Step(state, 0.0, StepInfo(accepted, acceptance, error))

    def with_step(self, step: float, /) -> Kernel:
        """The same correction at another time step.

        Args:
            step: the new step.

        Returns:
            A new kernel.
        """
        return MetropolisHastings(self._scheme.with_step(step))

    def with_target(self, target: Target, /) -> Kernel:
        """The same correction on another target.

        Args:
            target: the new target.

        Returns:
            A new kernel.
        """
        return MetropolisHastings(
            self._scheme.with_dynamics(self.dynamics.with_target(target))
        )

    def __repr__(self) -> str:
        return f"MetropolisHastings({self._scheme!r})"


def _require_label_jump(scheme: SplittingScheme, /) -> None:
    """Refuse a word over a mixture reference that never changes component.

    Raises:
        ValueError: if the dynamics has a label jump and the word lacks it.
    """
    if "L" in scheme.dynamics.flows() and "L" not in {letter for letter, _ in scheme.word}:
        raise ValueError(
            "Over a mixture reference the word needs the label jump L, or a "
            "chain would never leave the component it started in."
        )
