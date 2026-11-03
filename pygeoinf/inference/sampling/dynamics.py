"""
Sampling dynamics: SDEs whose exact pieces a splitting scheme composes.

Each dynamics is written in its function-space form, with the reference
covariance ``K`` as its preconditioner and the reference mean ``m`` as the
centre of its linear force, so that the reference's part is solved exactly
and never split:

.. code-block:: text

    PreconditionedOU  du == -(u - m) dt + sqrt(2) K^1/2 dW                       reference
    Overdamped        du == -(u - m + K grad Phi_ref) dt + sqrt(2) K^1/2 dW      target
    Underdamped       du == v dt,
                      dv == -(u - m + K grad Phi_ref) dt - g v dt
                            + sqrt(2 g) K^1/2 dW                                 target x N(0, K)
    Hamiltonian       the same with the velocity refreshed in full between trajectories

A dynamics generates its own letters from its target, so the reference's
flow is never something a user supplies:

=================  =========================================================
dynamics           letters
=================  =========================================================
PreconditionedOU   ``O``: :class:`~.flows.PriorOU`
Overdamped         ``O``: :class:`~.flows.PriorOU`, ``B``: :class:`~.flows.GradientDrift`
Underdamped        ``A``: :class:`~.flows.PriorRotation`, ``B``: :class:`~.flows.LikelihoodKick`,
                   ``O``: :class:`~.flows.MomentumRefresh` at the friction
Hamiltonian        the same, ``O`` a full refresh
Blockwise          one letter per summand of a direct sum, named by its label
                   (or position): :class:`~.flows.GibbsBlock`
=================  =========================================================

Letters belong to their dynamics: ``O`` is a momentum refresh in one and the
prior's Ornstein-Uhlenbeck step in another. Over a reference that follows the state,
the Laplace reference at each position, only :class:`PreconditionedOU` is
offered: its one step about the local Gaussian is the local MALA proposal.
Over a mixture reference every
dynamics also has ``L``, :class:`~.flows.LabelJump`, the move between
components; a kernel over a mixture must use it, or a chain would never
leave the component it started in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from numpy.random import Generator

from ...algebra.direct_sum import DirectSum
from ...algebra.spaces import HilbertSpace
from ...probability.base import ProbabilityMeasure, ProductMeasure
from ...probability.gaussian import GaussianMeasure
from .flows import (
    Flow,
    GibbsBlock,
    GradientDrift,
    LabelJump,
    LikelihoodKick,
    MomentumRefresh,
    PriorOU,
    PriorRotation,
)
from .target import State, Target

__all__ = [
    "Blockwise",
    "Dynamics",
    "Hamiltonian",
    "Overdamped",
    "PreconditionedOU",
    "Underdamped",
]


class Dynamics(ABC):
    """A sampling SDE, given by the exact flows it splits into.

    Args:
        target: the residual potential and the reference.

    Raises:
        ValueError: if one of the dynamics' flows needs a gradient the
            target cannot supply, or the target's reference follows the
            state and the dynamics is not :class:`PreconditionedOU`.
    """

    #: Whether the state carries a velocity.
    has_velocity: bool = False

    def __init__(self, target: Target, /) -> None:
        if target.is_local and not isinstance(self, PreconditionedOU):
            raise ValueError(
                "A reference that follows the state is solved by the prior "
                "Ornstein-Uhlenbeck step alone, whose transition density is "
                f"known; {type(self).__name__} is not offered over one."
            )
        self._target = target
        self._flows = self._build_flows()
        if target.is_mixture:
            self._flows["L"] = LabelJump(target)
        if not target.has_gradient:
            needing = [
                letter for letter, flow in self._flows.items() if flow.uses_gradient
            ]
            if needing:
                raise ValueError(
                    f"{type(self).__name__} needs the potential's gradient (for "
                    f"{', '.join(needing)}), and the forward operator supplies no "
                    "derivative; use PreconditionedOU, whose only flow needs none."
                )

    @abstractmethod
    def _build_flows(self) -> dict[str, Flow]:
        """The letters and the flows they stand for."""

    @property
    def target(self) -> Target:
        """The residual potential and the reference."""
        return self._target

    @property
    def reference(self) -> Any:
        """The reference Gaussian the dynamics is built on."""
        return self._target.reference

    def flows(self) -> dict[str, Flow]:
        """The letters a splitting word is written in, and their flows.

        Returns:
            A new dictionary from letter to flow.
        """
        return dict(self._flows)

    @property
    def state_space(self) -> HilbertSpace:
        """Where the state lives: the model space, or position and velocity."""
        space = self._target.model_space
        if not self.has_velocity:
            return space
        return DirectSum([space, space], labels=("position", "velocity"))

    @property
    def invariant_measure(self) -> ProbabilityMeasure:
        """The measure the continuous dynamics leaves invariant.

        The target, against its reference, with an independent ``N(0, K)``
        velocity for the inertial dynamics.

        Raises:
            NotImplementedError: for an inertial dynamics over a mixture
                reference, whose velocity's law depends on the label.
        """
        measure = self._target.measure
        if not self.has_velocity:
            return measure
        if self._target.is_mixture:
            raise NotImplementedError(
                "Over a mixture reference the velocity's law depends on the "
                "component, so the invariant measure is not a product."
            )
        return ProductMeasure(
            [measure, _velocity_measure(self._target)],
            labels=("position", "velocity"),
        )

    def initial_state(self, position: Any, /, *, rng: Generator | None = None) -> State:
        """A state at a position, with a velocity drawn from ``N(0, K)`` if
        the dynamics has one.

        Over a mixture reference the state is first given a component, drawn
        from its share of the density at the position, ``P(j | u)``, and the
        velocity is drawn from that component's covariance.

        Args:
            position: the model.
            rng: the generator for the label and the velocity.

        Returns:
            The state.
        """
        from ...algebra.spaces import resolve_rng

        rng = resolve_rng(rng)
        label = None
        target = self._target
        if target.is_mixture:
            shares = target.reference.responsibilities(position)
            label = int(rng.choice(shares.size, p=shares))
        state = State(position, label=label)
        if not self.has_velocity:
            return state
        return state.with_velocity(target.component(state).fluctuation(rng=rng))

    def flip(self, state: State, /) -> State:
        """The state with its velocity reversed, its cache kept.

        Args:
            state: the state.

        Returns:
            The flipped state; the state itself for dynamics with no velocity.
        """
        if state.velocity is None:
            return state
        return state.with_velocity(self._target.model_space.negative(state.velocity))

    @abstractmethod
    def with_target(self, target: Target, /) -> "Dynamics":
        """The same dynamics on another target.

        Args:
            target: the new target.

        Returns:
            The dynamics.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._target!r})"


class PreconditionedOU(Dynamics):
    """The reference's own Ornstein-Uhlenbeck process, with no data force.

    Its invariant measure is the reference; the data enter only through a
    Metropolis step on the residual potential, which makes it pCN. The one
    dynamics that needs no gradient.
    """

    def _build_flows(self) -> dict[str, Flow]:
        return {"O": PriorOU(self._target)}

    @property
    def invariant_measure(self) -> ProbabilityMeasure:
        """The reference: the data are left to the correction.

        Raises:
            NotImplementedError: for a reference that follows the state,
                which no one measure is.
        """
        if self._target.is_local:
            raise NotImplementedError(
                "A reference that follows the state is not one measure."
            )
        return self._target.reference

    def with_target(self, target: Target, /) -> "PreconditionedOU":
        """The same dynamics on another target.

        Args:
            target: the new target.

        Returns:
            The dynamics.
        """
        return PreconditionedOU(target)


class Overdamped(Dynamics):
    """Preconditioned Langevin dynamics on the position alone.

    The reference's Ornstein-Uhlenbeck step and the data's drift; MALA is
    this dynamics' word ``"BO"`` under a Metropolis-Hastings correction.
    """

    def _build_flows(self) -> dict[str, Flow]:
        return {"O": PriorOU(self._target), "B": GradientDrift(self._target)}

    def with_target(self, target: Target, /) -> "Overdamped":
        """The same dynamics on another target.

        Args:
            target: the new target.

        Returns:
            The dynamics.
        """
        return Overdamped(target)


class Underdamped(Dynamics):
    """Preconditioned Langevin dynamics on position and velocity, with friction.

    Args:
        target: the residual potential and the reference.
        friction: ``gamma``, positive; the refresh angle over a time ``t`` is
            ``arccos(exp(-gamma t))``.

    Raises:
        ValueError: for a friction that is not positive and finite, or as
            for :class:`Dynamics`.
    """

    has_velocity = True

    def __init__(self, target: Target, /, *, friction: float) -> None:
        if not 0.0 < friction < float("inf"):
            raise ValueError(
                f"A friction is positive and finite, got {friction}; a full "
                "refresh between trajectories is the Hamiltonian dynamics."
            )
        self._friction = float(friction)
        super().__init__(target)

    @property
    def friction(self) -> float:
        """The friction ``gamma``."""
        return self._friction

    def _build_flows(self) -> dict[str, Flow]:
        return {
            "A": PriorRotation(self._target),
            "B": LikelihoodKick(self._target),
            "O": MomentumRefresh(self._target, friction=self._friction),
        }

    def with_target(self, target: Target, /) -> "Underdamped":
        """The same dynamics, at the same friction, on another target.

        Args:
            target: the new target.

        Returns:
            The dynamics.
        """
        return Underdamped(target, friction=self._friction)


class Hamiltonian(Dynamics):
    """Hamiltonian dynamics on position and velocity, refreshed in full.

    The frictionless limit with the velocity redrawn from ``N(0, K)``
    between trajectories: HMC is the word ``"OBAB"`` with the block
    ``"BAB"`` repeated and Metropolized.
    """

    has_velocity = True

    def _build_flows(self) -> dict[str, Flow]:
        return {
            "A": PriorRotation(self._target),
            "B": LikelihoodKick(self._target),
            "O": MomentumRefresh(self._target),
        }

    def with_target(self, target: Target, /) -> "Hamiltonian":
        """The same dynamics on another target.

        Args:
            target: the new target.

        Returns:
            The dynamics.
        """
        return Hamiltonian(target)


class Blockwise(Dynamics):
    """Gibbs sampling over the summands of a direct sum, one block at a time.

    Each summand is a letter, a :class:`~.flows.GibbsBlock` that updates it
    against its conditional target with its own sampler and step, the others
    held. Every letter preserves the target, so a word over them, a
    systematic scan or its palindrome, needs no correction and is applied
    by a :class:`~pygeoinf.inference.sampling.corrections.Sweep`. The
    letters are the summands' labels, or their positions where the direct
    sum has none.

    Args:
        target: the target, on a direct sum, its reference the prior.
        blocks: what samples each summand: a mapping from the summand (by
            position or label) to its recipe, covering every summand once.
        steps: each block's time step, by summand; the recipe's if omitted.

    Raises:
        ValueError: if the model space is not a direct sum, the reference is
            not the prior, a summand has no block or two, or as for
            :class:`Dynamics`.
    """

    def __init__(
        self,
        target: Target,
        blocks: Any,
        /,
        *,
        steps: Any = None,
    ) -> None:
        space = target.model_space
        if not isinstance(space, DirectSum):
            raise ValueError(
                f"Blocks are the summands of a direct sum; the model space is {space!r}."
            )
        if target.is_local or target.is_mixture or target.reference.rank != 0:
            raise ValueError(
                "A Gibbs block's reference is the prior's conditional law, so "
                "the target's reference must be the prior."
            )
        recipes: dict[int, Any] = {}
        for key, recipe in dict(blocks).items():
            index = space.index(key)
            if index in recipes:
                raise ValueError(f"Summand {key!r} has two blocks.")
            recipes[index] = recipe
        missing = [i for i in range(len(space.subspaces)) if i not in recipes]
        if missing:
            raise ValueError(
                f"Every summand needs a block, or it never moves; {missing} have none."
            )
        given = (
            {} if steps is None else {space.index(k): v for k, v in dict(steps).items()}
        )
        self._recipes = tuple(recipes[i] for i in range(len(space.subspaces)))
        self._steps = tuple(
            float(given.get(i, recipe.step)) for i, recipe in enumerate(self._recipes)
        )
        labels = space.labels
        self._letters = tuple(
            str(i) if labels is None else labels[i] for i in range(len(space.subspaces))
        )
        super().__init__(target)

    @property
    def letters(self) -> tuple[str, ...]:
        """The blocks' letters, in summand order."""
        return self._letters

    @property
    def recipes(self) -> tuple[Any, ...]:
        """The blocks' recipes, in summand order."""
        return self._recipes

    @property
    def block_steps(self) -> tuple[float, ...]:
        """The blocks' time steps, in summand order."""
        return self._steps

    def _build_flows(self) -> dict[str, Flow]:
        return {
            letter: GibbsBlock(self._target, index, recipe, step=step)
            for index, (letter, recipe, step) in enumerate(
                zip(self._letters, self._recipes, self._steps)
            )
        }

    def scan(self, *, order: str = "systematic") -> list[tuple[str, float]]:
        """The word that visits every block: in summand order, or, for
        ``"symmetric"``, there and back, a palindrome and so reversible.

        Args:
            order: ``"systematic"`` or ``"symmetric"``.

        Returns:
            The word, as ``(letter, fraction)`` pairs.

        Raises:
            ValueError: for another order.
        """
        letters = list(self._letters)
        if order == "systematic":
            return [(letter, 1.0) for letter in letters]
        if order == "symmetric":
            if len(letters) == 1:
                return [(letters[0], 1.0)]
            ends = [(letter, 0.5) for letter in letters[:-1]]
            return ends + [(letters[-1], 1.0)] + ends[::-1]
        raise ValueError(f"A scan is 'systematic' or 'symmetric', not {order!r}.")

    def with_steps(self, steps: Any, /) -> "Blockwise":
        """The same blocks at other steps.

        Args:
            steps: the blocks' steps, in summand order.

        Returns:
            The dynamics.
        """
        return Blockwise(
            self._target,
            dict(enumerate(self._recipes)),
            steps=dict(enumerate(steps)),
        )

    def with_target(self, target: Target, /) -> "Blockwise":
        """The same blocks, at the same steps, on another target.

        Args:
            target: the new target.

        Returns:
            The dynamics.
        """
        return Blockwise(
            target, dict(enumerate(self._recipes)), steps=dict(enumerate(self._steps))
        )


def _velocity_measure(target: Target, /) -> GaussianMeasure:
    """``N(0, K)`` on the model space, ``K`` the reference's covariance."""
    reference = target.reference
    return GaussianMeasure(
        target.model_space,
        covariance=reference.covariance,
        sample=lambda rng: reference.fluctuation(rng=rng),
        unsafe=True,
    )
