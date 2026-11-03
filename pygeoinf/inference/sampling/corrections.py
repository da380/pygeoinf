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
* :class:`Sweep`, for a word whose every letter already leaves the target
  invariant, Gibbs blocks above all: it needs no correction and is applied
  as it stands.
* :class:`MetropolisHastings`, for a stochastic proposal whose transition
  density is known relative to the reference's own kernel: pCN and its
  generalization over any reference, the independence sampler, the
  overdamped MALA proposal, and the local MALA proposal, the prior
  Ornstein-Uhlenbeck step about a reference that follows the state, whose
  ratio compares the two references' pair laws.

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

__all__ = [
    "Kernel",
    "MetropolisHastings",
    "MetropolizedBlock",
    "Step",
    "StepInfo",
    "Sweep",
]


@dataclass(frozen=True)
class StepInfo:
    """What one step of a kernel measured.

    Attributes:
        accepted: whether the proposal was accepted.
        acceptance: the acceptance probability, ``min(1, exp(log alpha))``.
        energy_error: ``-log alpha``, the error the correction removed:
            for a Metropolized block, the change in the shadow energy.
        acceptances: for a kernel of several blocks, each block's mean
            acceptance probability, in the order of its ``step_sizes``;
            ``None`` for a kernel of one.
    """

    accepted: bool
    acceptance: float
    energy_error: float
    acceptances: tuple[float, ...] | None = None


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

    @property
    def step_sizes(self) -> tuple[float, ...]:
        """Every time step the kernel has, one per block; one for most."""
        return (self.step_size,)

    def with_steps(self, steps: Any, /) -> "Kernel":
        """The same kernel at other steps, one per entry of :attr:`step_sizes`.

        Args:
            steps: the new steps.

        Returns:
            A new kernel.

        Raises:
            ValueError: for the wrong number of steps.
        """
        steps = tuple(steps)
        if len(steps) != len(self.step_sizes):
            raise ValueError(
                f"Expected {len(self.step_sizes)} steps, one per block; got {len(steps)}."
            )
        return self.with_step(steps[0])

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
        self,
        scheme: SplittingScheme,
        block: SplittingScheme,
        start: int,
        /,
        *,
        flip: bool,
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
            dynamics that carries a velocity, a mixture reference and a
            word without its label jump, or a reference that follows the
            state and a word other than ``"O"``.
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
        local = scheme.dynamics.target.is_local
        if local and (
            len(scheme.flows) != 1 or scheme.repeat != 1 or proposal.word[0][0] != "O"
        ):
            raise ValueError(
                "Over a reference that follows the state the proposal is one "
                "prior Ornstein-Uhlenbeck step, the word 'O': each later letter "
                "would see another reference."
            )
        if not proposal.has_transition_density:
            raise ValueError(
                f"The transition density of {proposal!r} is not known, so no "
                "Metropolis-Hastings ratio can be formed for it."
            )
        self._local = local
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

    def _run(
        self, flows: list[tuple[Flow, float]], state: State, rng: Generator
    ) -> State:
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
        elif self._local:
            ratio = self._local_ratio(state, proposal)
            error = proposed - target.value(state) - ratio
        else:
            ratio = self._proposal.log_transition_ratio(state, proposal)
            error = proposed - target.value(state) - ratio
        accepted, acceptance = _accept(-error, rng)
        state = self._run(self._after, proposal if accepted else state, rng)
        return Step(state, 0.0, StepInfo(accepted, acceptance, error))

    def _local_ratio(self, state: State, proposal: State) -> float:
        """``G_B(u, v) - G_A(u, v)``: the pair laws' densities against the
        prior's, ``A`` the reference at the state and ``B`` at the proposal.

        The proposal is ``A``'s Ornstein-Uhlenbeck step, reversible with
        respect to ``A``, so ``pi(du) q(u, dv)`` has density
        ``exp(-Phi_ref,A(u) + G_A(u, v))`` against the prior's pair law, and
        the reverse move ``exp(-Phi_ref,B(v) + G_B(u, v))``; the ratio of the
        two is finite, both references being equivalent to the prior.
        """
        target = self.dynamics.target
        ((flow, fraction),) = self._proposal.flows
        correlation = float(np.cos(flow.angle(fraction * self._scheme.step)))
        u, v = state.position, proposal.position
        return target.component(proposal).pair_log_density(
            u, v, correlation
        ) - target.component(state).pair_log_density(u, v, correlation)

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
    if "L" in scheme.dynamics.flows() and "L" not in {
        letter for letter, _ in scheme.word
    }:
        raise ValueError(
            "Over a mixture reference the word needs the label jump L, or a "
            "chain would never leave the component it started in."
        )


class Sweep(Kernel):
    """A word of target-preserving letters, applied as it stands.

    Every letter already leaves the target invariant, so the word does too
    and nothing is corrected: the Gibbs blocks of a
    :class:`~pygeoinf.inference.sampling.dynamics.Blockwise` dynamics, each
    an exact kernel on its summand's conditional. A palindromic word, the
    symmetric scan, is reversible. Over a ``Blockwise`` dynamics the kernel
    has one step per block, and each step reports each block's acceptance.

    Args:
        scheme: a scheme every letter of which preserves the target.

    Raises:
        ValueError: if a letter does not preserve the target.
    """

    def __init__(self, scheme: SplittingScheme, /) -> None:
        unfit = [repr(flow) for flow, _ in scheme.flows if TARGET not in flow.preserves]
        if unfit:
            raise ValueError(
                "A sweep applies its letters uncorrected, so each must preserve "
                f"the target; {', '.join(unfit)} does not."
            )
        _require_label_jump(scheme)
        self._scheme = scheme

    @property
    def dynamics(self) -> Dynamics:
        """The dynamics the word is written over."""
        return self._scheme.dynamics

    @property
    def scheme(self) -> SplittingScheme:
        """The word."""
        return self._scheme

    @property
    def step_size(self) -> float:
        """The first block's step, or the scheme's; see :attr:`step_sizes`."""
        return self.step_sizes[0]

    @property
    def step_sizes(self) -> tuple[float, ...]:
        """Each block's step, in summand order; the scheme's step otherwise."""
        steps = getattr(self.dynamics, "block_steps", None)
        return tuple(steps) if steps is not None else (self._scheme.step,)

    def step(self, state: State, /, *, rng: Generator | None = None) -> Step:
        """Every letter in turn.

        Args:
            state: where to start.
            rng: the generator.

        Returns:
            The step, with a log-weight of zero, accepted when any block
            moved, with the blocks' mean acceptance and each block's.
        """
        rng = resolve_rng(rng)
        step = self._scheme.step
        totals: dict[int, list[float]] = {}
        moved = False
        for flow, fraction in self._scheme.flows:
            state, _ = flow.apply(state, fraction * step, rng=rng)
            index = getattr(flow, "index", None)
            if index is not None:
                totals.setdefault(index, []).append(
                    state.cache.get(("block acceptance", index), 1.0)
                )
                moved = moved or bool(state.cache.get(("block moved", index), False))
        if totals:
            acceptances = tuple(float(np.mean(totals[i])) for i in sorted(totals))
            return Step(
                state,
                0.0,
                StepInfo(moved, float(np.mean(acceptances)), 0.0, acceptances),
            )
        return Step(state, 0.0, StepInfo(True, 1.0, 0.0))

    def with_step(self, step: float, /) -> Kernel:
        """The same word with every block at one step.

        Args:
            step: the new step.

        Returns:
            A new kernel.
        """
        return self.with_steps([step] * len(self.step_sizes))

    def with_steps(self, steps: Any, /) -> Kernel:
        """The same word at other steps, one per block.

        Args:
            steps: the new steps, in summand order.

        Returns:
            A new kernel.

        Raises:
            ValueError: for the wrong number of steps.
        """
        steps = tuple(float(value) for value in steps)
        if len(steps) != len(self.step_sizes):
            raise ValueError(
                f"Expected {len(self.step_sizes)} steps, one per block; got {len(steps)}."
            )
        with_steps = getattr(self.dynamics, "with_steps", None)
        if with_steps is None:
            return Sweep(self._scheme.with_step(steps[0]))
        return Sweep(self._scheme.with_dynamics(with_steps(steps)))

    def with_target(self, target: Target, /) -> Kernel:
        """The same word on another target.

        Args:
            target: the new target.

        Returns:
            A new kernel.
        """
        return Sweep(self._scheme.with_dynamics(self.dynamics.with_target(target)))

    def __repr__(self) -> str:
        return f"Sweep({self._scheme!r})"
