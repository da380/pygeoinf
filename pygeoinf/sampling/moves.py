"""
Moves: how a chain steps, built for whatever target it is given.

A move fixes a dynamics, a word and a correction, and knows nothing of a
forward problem or a prior: given a
:class:`~pygeoinf.sampling.target.Target`, it builds the kernel that leaves
it invariant. What a move is run for, and how many times, is a population's
business (:mod:`.population`), and what is sampled, the posterior of a
forward problem and its data, is an estimator's
(:class:`~pygeoinf.inference.monte_carlo.MonteCarloEstimator`).

.. code-block:: text

    move                dynamics           word, block            correction
    PCN                 PreconditionedOU   "O"                    Metropolis-Hastings
    MALA                Overdamped         "BO"                   Metropolis-Hastings
    MALA, local         PreconditionedOU   "O", about the state's own reference
    HMC                 Hamiltonian        "OBAB", "BAB" x L      Metropolized block
    GeneralizedHMC      Underdamped        "OBAB", "BAB" x L      Metropolized block, flip
    Gibbs               Blockwise          a scan of the summands, each block its own move

So ``HMC(step=0.2, length=5).kernel(target)`` is

.. code-block:: python

    scheme = SplittingScheme(Hamiltonian(target), "OBAB", step=0.2)
    MetropolizedBlock(scheme.block("BAB", repeat=5)).within(scheme)

Every move is solved about the target's reference: the prior, or any
Gaussian equivalent to it, a Laplace Gaussian or one fitted to a run, which
moves the chain at the posterior's scale in the directions the data inform
and at the prior's elsewhere. Any reference leaves the chain exact; a poor
one only makes it slow. Over a
:class:`~pygeoinf.probability.reference.MixtureReference`, fitted to a
posterior with several modes, every word starts with the label jump ``L``,
which moves a chain between the mixture's components, and so between modes.

Each move carries its time step ``h``, where a run's burn-in starts, and
the acceptance it adapts the step towards during burn-in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

import numpy as np

from .corrections import Kernel, MetropolisHastings, MetropolizedBlock, Sweep
from .dynamics import Blockwise, Hamiltonian, Overdamped, PreconditionedOU, Underdamped
from .population import StepSizeAdaptation
from .schemes import SplittingScheme
from .target import Target

__all__ = ["GeneralizedHMC", "Gibbs", "HMC", "MALA", "Move", "PCN"]


class Move(ABC):
    """How a chain steps: a kernel for any target.

    A subclass says which kernel it builds on a target and how its step
    adapts during burn-in; :class:`~pygeoinf.sampling.population.Chains`
    and :class:`~pygeoinf.sampling.population.SMC` run it.
    """

    #: Whether the kernel differentiates the potential.
    uses_gradient: bool = False

    @property
    def is_local(self) -> bool:
        """Whether the move needs a reference that follows the state."""
        return False

    @abstractmethod
    def kernel(self, target: Target, /, *, step: float | None = None) -> Kernel:
        """The kernel on a target.

        Args:
            target: what the kernel leaves invariant.
            step: the time step; the move's if omitted.

        Returns:
            The kernel.
        """

    @abstractmethod
    def adaptation(
        self, target: Target, /
    ) -> StepSizeAdaptation | tuple[StepSizeAdaptation | None, ...] | None:
        """How the step adapts during burn-in on a target.

        Args:
            target: the target the kernel is built on.

        Returns:
            The adaptation, ``None`` for a fixed step, or one per block for
            :class:`Gibbs`.
        """


class _OneStep(Move):
    """A move with one time step, adapted towards one acceptance.

    Raises:
        ValueError: for a step that is not positive and finite, or an
            acceptance outside ``(0, 1)``.
    """

    def __init__(self, *, step: float, acceptance: float | None) -> None:
        if not 0.0 < step < float("inf"):
            raise ValueError(f"A step is positive and finite, got {step}.")
        if acceptance is not None and not 0.0 < acceptance < 1.0:
            raise ValueError(f"A target acceptance lies in (0, 1), got {acceptance}.")
        self._step = float(step)
        self._acceptance = acceptance

    @property
    def step(self) -> float:
        """The time step ``h``, where the burn-in's adaptation starts."""
        return self._step

    @property
    def acceptance(self) -> float | None:
        """The acceptance the step adapts towards, or ``None`` for a fixed
        step."""
        return self._acceptance

    @property
    def longest_step(self) -> float:
        """The step beyond which the adaptation gains nothing.

        For an Ornstein-Uhlenbeck step, ten: ``exp(-10)`` of the old point is
        left, and the proposal is as good as independent.
        """
        return 10.0

    def _warns(self, target: Target, /) -> bool:
        """Whether a step adapted to about :attr:`longest_step` is warned of."""
        return True

    def adaptation(self, target: Target, /) -> StepSizeAdaptation | None:
        """How the step adapts during burn-in, capped at :attr:`longest_step`.

        Args:
            target: the target the kernel is built on.

        Returns:
            The adaptation, or ``None`` for a fixed step.
        """
        if self._acceptance is None:
            return None
        return StepSizeAdaptation(
            acceptance=self._acceptance,
            maximum=self.longest_step,
            warns=self._warns(target),
        )

    def kernel(self, target: Target, /, *, step: float | None = None) -> Kernel:
        """The kernel on a target.

        Args:
            target: what the kernel leaves invariant.
            step: the time step; the move's if omitted.

        Returns:
            The kernel.
        """
        return self._build(target, self._step if step is None else float(step))

    @abstractmethod
    def _build(self, target: Target, step: float, /) -> Kernel:
        """The kernel on a target at a step."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(step={self._step:g})"


class PCN(_OneStep):
    """Preconditioned Crank-Nicolson: the reference's Ornstein-Uhlenbeck step.

    The proposal ``m + cos(theta) (u - m) + sin(theta) xi``, with ``xi`` a
    centered draw of the reference and ``cos(theta) == exp(-h)``, accepted on
    the residual potential alone. Needs no gradient. In the literature's
    parameter, ``beta == sqrt(1 - exp(-2 h))``.

    Args:
        step: the time step ``h``.
        acceptance: the acceptance to adapt towards during burn-in, about a
            quarter by default; ``None`` keeps the step fixed.

    Raises:
        ValueError: for a step that is not positive and finite, or an
            acceptance outside ``(0, 1)``.
    """

    def __init__(self, *, step: float, acceptance: float | None = 0.25) -> None:
        super().__init__(step=step, acceptance=acceptance)

    def _build(self, target: Target, step: float, /) -> Kernel:
        return MetropolisHastings(
            SplittingScheme(PreconditionedOU(target), _word(target, "O"), step=step)
        )


class MALA(_OneStep):
    """The preconditioned Metropolis-adjusted Langevin algorithm.

    A drift down the preconditioned gradient for a time ``h``, then the
    reference's Ornstein-Uhlenbeck step for ``h``, corrected by
    Metropolis-Hastings with the proposal's transition density, which on a
    function space is a Cameron-Martin shift and needs no inverse
    covariance.

    With ``local=True``, the local MALA: at every state the reference is
    the Laplace Gaussian there, ``K == (C^-1 + H(u))^-1`` centered on the
    Newton step from ``u``, which the target supplies
    (:attr:`~pygeoinf.sampling.target.Target.local`). That Gaussian is the
    posterior's quadratic model at ``u``, so the drift vanishes and the
    proposal is its Ornstein-Uhlenbeck step,
    ``v == m_u + cos(a) (u - m_u) + sin(a) xi`` with ``xi ~ N(0, K_u)``; the
    acceptance compares the pair laws of the references at ``u`` and ``v``,
    an ``r``-dimensional ratio, so the move is defined on a function space.
    Each step builds the reference at the proposal. The proposal follows the
    posterior's local shape, which pays where that shape changes across the
    posterior; a
    :class:`~pygeoinf.inference.monte_carlo.MonteCarloEstimator` builds the
    target for it from a Laplace approximation given as its reference.

    Args:
        step: the time step ``h``.
        acceptance: the acceptance to adapt towards during burn-in, 0.57 by
            default; ``None`` keeps the step fixed.
        local: whether the reference follows the state.

    Raises:
        ValueError: for a step that is not positive and finite, or an
            acceptance outside ``(0, 1)``.
    """

    uses_gradient = True

    def __init__(
        self,
        *,
        step: float,
        acceptance: float | None = 0.57,
        local: bool = False,
    ) -> None:
        super().__init__(step=step, acceptance=acceptance)
        self._local = bool(local)

    @property
    def is_local(self) -> bool:
        """Whether the reference follows the state."""
        return self._local

    def _build(self, target: Target, step: float, /) -> Kernel:
        if not self._local:
            return MetropolisHastings(
                SplittingScheme(Overdamped(target), _word(target, "BO"), step=step)
            )
        if not target.is_local:
            raise ValueError(
                "A local MALA moves about a reference that follows the state, "
                "and this target's does not: give the estimator a Laplace "
                "approximation as its reference."
            )
        return MetropolisHastings(
            SplittingScheme(PreconditionedOU(target), "O", step=step)
        )

    def __repr__(self) -> str:
        local = ", local=True" if self._local else ""
        return f"MALA(step={self._step:g}{local})"


class HMC(_OneStep):
    """Hamiltonian Monte Carlo, the reference's part solved by rotation.

    A full velocity refresh from ``N(0, K)``, then ``length`` leapfrog steps
    ``"BAB"`` in which the kick follows the residual potential and the
    rotation solves the reference exactly, Metropolized as a block. A
    trajectory lasts ``length * h``; at ``length * h`` near ``pi / 2`` the
    rotation alone carries a point to an independent one. Where the
    reference is close to the posterior, a Laplace Gaussian say, the
    rotation does nearly all the work, and a quarter turn gives nearly
    independent proposals that are nearly always accepted; a step adapted to
    it is then warned of only over the prior.

    Args:
        step: the time step ``h``.
        length: the leapfrog steps per trajectory.
        acceptance: the acceptance to adapt towards during burn-in, 0.65 by
            default; ``None`` keeps the step fixed.

    Raises:
        ValueError: for a length below one, a step that is not positive and
            finite, or an acceptance outside ``(0, 1)``.
    """

    uses_gradient = True

    def __init__(
        self, *, step: float, length: int, acceptance: float | None = 0.65
    ) -> None:
        if int(length) != length or length < 1:
            raise ValueError(
                f"A trajectory has at least one leapfrog step, got {length}."
            )
        super().__init__(step=step, acceptance=acceptance)
        self._length = int(length)

    @property
    def length(self) -> int:
        """The leapfrog steps per trajectory."""
        return self._length

    @property
    def longest_step(self) -> float:
        """A quarter turn over the trajectory, ``pi / (2 length)``."""
        return float(np.pi / (2 * self._length))

    def _warns(self, target: Target, /) -> bool:
        return _over_the_prior(target)

    def _build(self, target: Target, step: float, /) -> Kernel:
        scheme = SplittingScheme(Hamiltonian(target), _word(target, "OBAB"), step=step)
        return MetropolizedBlock(scheme.block("BAB", repeat=self._length)).within(
            scheme
        )

    def __repr__(self) -> str:
        return f"HMC(step={self._step:g}, length={self._length})"


class GeneralizedHMC(_OneStep):
    """HMC with a partial velocity refresh: underdamped Langevin, Metropolized.

    Generalized HMC, in Horowitz's (1991) name. Before each trajectory the
    velocity is rotated towards a fresh draw by the angle
    ``arccos(exp(-gamma h))``, rather than redrawn, and a rejected trajectory
    reverses the velocity, so the chain keeps its direction between
    trajectories. The refresh acts once per trajectory, for one step: with
    the default single leapfrog step the chain is a discretized underdamped
    Langevin dynamics, exact by its correction. Like every move here it is
    preconditioned by its reference; what sets it apart is the partial
    refresh. As a Gibbs block, the velocity is drawn afresh at each visit.

    Args:
        step: the time step ``h``.
        friction: ``gamma``, positive and finite.
        length: the leapfrog steps per trajectory.
        acceptance: the acceptance to adapt towards during burn-in, 0.8 by
            default; ``None`` keeps the step fixed.

    Raises:
        ValueError: for a friction that is not positive and finite, a length
            below one, a step that is not positive and finite, or an
            acceptance outside ``(0, 1)``.
    """

    uses_gradient = True

    def __init__(
        self,
        *,
        step: float,
        friction: float = 1.0,
        length: int = 1,
        acceptance: float | None = 0.8,
    ) -> None:
        if not 0.0 < friction < float("inf"):
            raise ValueError(f"A friction is positive and finite, got {friction}.")
        if int(length) != length or length < 1:
            raise ValueError(
                f"A trajectory has at least one leapfrog step, got {length}."
            )
        super().__init__(step=step, acceptance=acceptance)
        self._friction = float(friction)
        self._length = int(length)

    @property
    def friction(self) -> float:
        """The friction ``gamma``."""
        return self._friction

    @property
    def length(self) -> int:
        """The leapfrog steps per trajectory."""
        return self._length

    @property
    def longest_step(self) -> float:
        """A quarter turn over the trajectory, ``pi / (2 length)``."""
        return float(np.pi / (2 * self._length))

    def _warns(self, target: Target, /) -> bool:
        return _over_the_prior(target)

    def _build(self, target: Target, step: float, /) -> Kernel:
        dynamics = Underdamped(target, friction=self._friction)
        scheme = SplittingScheme(dynamics, _word(target, "OBAB"), step=step)
        return MetropolizedBlock(scheme.block("BAB", repeat=self._length)).within(
            scheme
        )

    def __repr__(self) -> str:
        return (
            f"GeneralizedHMC(step={self._step:g}, friction={self._friction:g}, "
            f"length={self._length})"
        )


class Gibbs(Move):
    """Gibbs sampling over the summands of a direct sum: each field its own
    move, against its conditional.

    For a model of several physical fields with different correlation
    structures, a level set beside elastic parameters say: each summand of
    the model space is a block, updated with the others held by its own
    move's kernel on its conditional target, whose reference is the prior's
    conditional law of that summand
    (:meth:`~pygeoinf.probability.gaussian.GaussianMeasure.conditional`), so
    every block is a function-space sampler and the sweep is exact. A
    systematic scan visits the blocks in summand order; a symmetric one
    there and back, which is reversible. Each block's step adapts towards
    its own move's acceptance during burn-in, and a run reports each block's
    acceptance (``block_acceptance_rates``). The target's reference must be
    the prior.

    Args:
        blocks: each summand's move, by position or label, covering every
            summand once when the kernel is built.
        order: ``"systematic"`` or ``"symmetric"``.
        repeat: each block's kernel steps at each visit, by position or
            label; one if omitted.

    Raises:
        TypeError: if a block is not a move with one step, a :class:`PCN`,
            :class:`MALA`, :class:`HMC` or :class:`GeneralizedHMC`.
        ValueError: if a block's move follows the state, whose reference
            cannot be the prior's conditional, the order is unknown, or a
            repeat is below one.
    """

    def __init__(
        self,
        blocks: Mapping[int | str, Move],
        *,
        order: str = "systematic",
        repeat: Mapping[int | str, int] | None = None,
    ) -> None:
        if order not in ("systematic", "symmetric"):
            raise ValueError(f"A scan is 'systematic' or 'symmetric', not {order!r}.")
        blocks = dict(blocks)
        for key, move in blocks.items():
            if not isinstance(move, _OneStep):
                raise TypeError(
                    "A block is a move with one step, PCN(step=...) say; got "
                    f"{type(move).__name__} for {key!r}."
                )
            if move.is_local:
                raise ValueError(
                    "A block's reference is the prior's conditional law, which "
                    "does not follow the state; use a move that is not local."
                )
        repeat = {} if repeat is None else dict(repeat)
        for key, count in repeat.items():
            if int(count) != count or count < 1:
                raise ValueError(
                    f"A block takes at least one step a visit, got {count} for {key!r}."
                )
        self._blocks = blocks
        self._order = order
        self._repeat = repeat
        self.uses_gradient = any(move.uses_gradient for move in blocks.values())

    @property
    def blocks(self) -> dict[int | str, Move]:
        """Each summand's move, as given."""
        return dict(self._blocks)

    @property
    def order(self) -> str:
        """The scan."""
        return self._order

    @property
    def repeat(self) -> dict[int | str, int]:
        """Each block's kernel steps at each visit, where given."""
        return dict(self._repeat)

    def kernel(self, target: Target, /, *, step: float | None = None) -> Kernel:
        """The sweep on a target over a direct sum.

        Args:
            target: what the sweep leaves invariant, its reference the prior.
            step: unused; each block has its own move's step.

        Returns:
            The kernel, whose steps are the blocks', in summand order.

        Raises:
            ValueError: as for
                :class:`~pygeoinf.sampling.dynamics.Blockwise`: a model space
                that is not a direct sum, a reference that is not the prior,
                or a summand with no block or two.
        """
        dynamics = Blockwise(target, self._blocks, repeat=self._repeat)
        return Sweep(
            SplittingScheme(dynamics, dynamics.scan(order=self._order), step=1.0)
        )

    def adaptation(self, target: Target, /) -> tuple[StepSizeAdaptation | None, ...]:
        """Each block's adaptation, in summand order.

        Args:
            target: the target over the direct sum.

        Returns:
            One adaptation per summand, ``None`` for a fixed step.
        """
        space = target.model_space
        indexed = {space.index(key): move for key, move in self._blocks.items()}
        return tuple(indexed[i].adaptation(target) for i in sorted(indexed))

    def __repr__(self) -> str:
        blocks = ", ".join(f"{key!r}: {move!r}" for key, move in self._blocks.items())
        return f"Gibbs({{{blocks}}}, order={self._order!r})"


def _word(target: Target, word: str, /) -> str:
    """A move's word, led by the label jump over a mixture reference."""
    return "L" + word if target.is_mixture else word


def _over_the_prior(target: Target, /) -> bool:
    """Whether the target's reference is the prior itself."""
    return not target.is_mixture and not target.is_local and target.reference.rank == 0
