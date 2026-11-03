"""
The exactly solvable pieces of a sampling dynamics.

A dynamics is split into pieces each of which can be solved exactly for any
time; a splitting scheme composes them. Every piece here either preserves the
reference Gaussian ``N(m, K)`` exactly -- a rotation of position against
velocity, a momentum refresh, a prior Ornstein-Uhlenbeck step -- or is the
data force, which is all the step error comes from. That is what makes the
samplers built from them well defined on a function space: no step error is
committed on the reference's infinitely many modes.

Each flow declares what it touches, which measure it preserves exactly,
whether it is deterministic and reversible under a velocity flip, whether it
needs the gradient, and the angle a time step turns into. Its
:meth:`Flow.apply` returns the new state and the change it makes to the
reference's Gaussian energy ``(|u - m|_K^2 + |v|_K^2) / 2``: infinite at a
typical point, finite in its changes, and zero for every flow that preserves
the reference. A Metropolis correction of a deterministic block adds those
changes up; stochastic flows report none, never sitting inside one.

Over a mixture reference every flow solves the component the state is
labelled with, and :class:`LabelJump` changes the label.

Two converters take the literature's parameters for the prior
Ornstein-Uhlenbeck step to the time step ``h``, for readers coming from the
papers: pCN's ``beta`` and Crank-Nicolson's ``delta``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.random import Generator

from .target import State, Target, enter_block, leave_block

__all__ = [
    "Flow",
    "GibbsBlock",
    "GradientDrift",
    "KernelFlow",
    "LabelJump",
    "LikelihoodKick",
    "MomentumRefresh",
    "PriorOU",
    "PriorRotation",
    "crank_nicolson_delta",
    "pcn_beta",
    "time_from_crank_nicolson",
    "time_from_pcn_beta",
]

#: Names for what a flow can touch and preserve.
POSITION = "position"
VELOCITY = "velocity"
LABEL = "label"
REFERENCE = "reference"
TARGET = "target"


class Flow(ABC):
    """One exactly solvable piece of a dynamics, built on a target.

    Subclasses declare:

    * ``touches``: the parts of the state it changes, ``"position"``,
      ``"velocity"`` or both;
    * ``preserves``: the measures it leaves exactly invariant, from
      ``"reference"`` (the reference Gaussian, with an independent ``N(0,
      K)`` velocity where there is one) and ``"target"``;
    * ``deterministic`` and ``reversible``, the latter meaning that a
      velocity flip turns the flow into its inverse;
    * ``uses_gradient``.

    Args:
        target: what the flow is built on, which gives it the reference
            and, for the data force, the gradient.
    """

    touches: frozenset[str] = frozenset()
    preserves: frozenset[str] = frozenset()
    deterministic: bool = True
    reversible: bool = False
    uses_gradient: bool = False

    def __init__(self, target: Target, /) -> None:
        self._target = target

    @property
    def target(self) -> Target:
        """The target the flow is built on."""
        return self._target

    def angle(self, time: float, /) -> float:
        """The angle, or length, the flow applies for a time step.

        Each flow converts the dynamics' time to its own parameter, because
        the mapping differs from flow to flow.

        Args:
            time: the time step.

        Returns:
            The flow's own parameter for that time.
        """
        return float(time)

    @abstractmethod
    def apply(
        self, state: State, time: float, /, *, rng: Generator | None = None
    ) -> tuple[State, float | None]:
        """Solve the flow for a time.

        Args:
            state: where to start.
            time: how long to flow for.
            rng: the generator, for a stochastic flow.

        Returns:
            The new state and the change in the reference's Gaussian energy:
            a number for a deterministic flow, ``+inf`` where the potential
            could not be evaluated, ``None`` for a stochastic flow.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class PriorRotation(Flow):
    """The reference's own Hamiltonian flow: a rotation of ``(u - m, v)``.

    ``u' == m + cos(t) (u - m) + sin(t) v`` and
    ``v' == -sin(t) (u - m) + cos(t) v``. Two independent ``N(0, K)``
    fields rotated are two independent ``N(0, K)`` fields, so the reference
    with its velocity is preserved exactly, for any time and on any mesh.
    The angle is the time.
    """

    touches = frozenset({POSITION, VELOCITY})
    preserves = frozenset({REFERENCE})
    deterministic = True
    reversible = True

    def apply(
        self, state: State, time: float, /, *, rng: Generator | None = None
    ) -> tuple[State, float]:
        """Rotate by ``time``; the Gaussian energy does not change.

        Args:
            state: the state, with a velocity.
            time: the angle.
            rng: unused.

        Returns:
            The rotated state and an energy change of zero.
        """
        space = self._target.model_space
        mean = self._target.component(state).expectation
        offset = space.subtract(state.position, mean)
        c, s = np.cos(time), np.sin(time)
        position = space.axpy(
            float(s), state.velocity, space.axpy(float(c), offset, space.copy(mean))
        )
        velocity = space.axpy(float(-s), offset, space.scale(float(c), state.velocity))
        return state.moved(position, velocity), 0.0


class LikelihoodKick(Flow):
    """The data force on the velocity: ``v' == v - t K grad Phi_ref(u)``.

    The only piece that does not preserve the reference. It shifts the
    velocity by a Cameron-Martin vector, which changes the Gaussian energy
    by the finite amount ``-(t/2) DPhi_ref(u)[v + v']``: a derivative applied
    to vectors, with no inverse covariance anywhere. The kick length is the
    time.
    """

    touches = frozenset({VELOCITY})
    preserves = frozenset()
    deterministic = True
    reversible = True
    uses_gradient = True

    def apply(
        self, state: State, time: float, /, *, rng: Generator | None = None
    ) -> tuple[State, float]:
        """Kick for ``time``.

        Args:
            state: the state, with a velocity.
            time: the kick length.
            rng: unused.

        Returns:
            The kicked state, its cache kept, and the energy change; ``+inf``
            with the state unchanged where the potential cannot be
            evaluated.
        """
        target = self._target
        if target.value(state) == float("inf"):
            return state, float("inf")
        space = target.model_space
        gradient = target.gradient(state)
        velocity = space.axpy(
            -float(time),
            target.preconditioned_gradient(state),
            space.copy(state.velocity),
        )
        change = (
            -0.5
            * float(time)
            * (
                space.inner_product(gradient, state.velocity)
                + space.inner_product(gradient, velocity)
            )
        )
        return state.with_velocity(velocity), change


class MomentumRefresh(Flow):
    """Friction and noise on the velocity: an exact Ornstein-Uhlenbeck step.

    ``v' == cos(z) v + sin(z) xi`` with ``xi ~ N(0, K)`` and
    ``cos(z) == exp(-friction t)``; infinite friction is a full refresh,
    ``z == pi / 2``. It preserves the reference with its velocity and the
    target with its velocity, the velocity being independent of the
    position under both.

    Args:
        target: the target.
        friction: the friction ``gamma``, positive, or ``inf`` for a full
            refresh.

    Raises:
        ValueError: for a friction that is not positive.
    """

    touches = frozenset({VELOCITY})
    preserves = frozenset({REFERENCE, TARGET})
    deterministic = False
    reversible = False

    def __init__(self, target: Target, /, *, friction: float = float("inf")) -> None:
        if not friction > 0.0:
            raise ValueError(f"A friction is positive, got {friction}.")
        super().__init__(target)
        self._friction = float(friction)

    @property
    def friction(self) -> float:
        """The friction; ``inf`` for a full refresh."""
        return self._friction

    def angle(self, time: float, /) -> float:
        """``arccos(exp(-friction time))``, ``pi / 2`` for a full refresh.

        Args:
            time: the time step.

        Returns:
            The refresh angle.
        """
        if self._friction == float("inf"):
            return 0.5 * np.pi
        return float(np.arccos(np.exp(-self._friction * float(time))))

    def apply(
        self, state: State, time: float, /, *, rng: Generator | None = None
    ) -> tuple[State, None]:
        """Refresh the velocity for ``time``.

        Args:
            state: the state, with a velocity.
            time: the time step.
            rng: the generator for the fresh noise.

        Returns:
            The refreshed state, its cache kept, and no energy change.
        """
        space = self._target.model_space
        z = self.angle(time)
        noise = self._target.component(state).fluctuation(rng=rng)
        velocity = space.axpy(
            float(np.sin(z)), noise, space.scale(float(np.cos(z)), state.velocity)
        )
        return state.with_velocity(velocity), None

    def __repr__(self) -> str:
        return f"MomentumRefresh(friction={self._friction:g})"


class PriorOU(Flow):
    """The reference's Ornstein-Uhlenbeck step on the position: pCN's move.

    ``u' == m + cos(a) (u - m) + sin(a) xi`` with ``xi ~ N(0, K)`` and
    ``cos(a) == exp(-t)``, the exact solution of
    ``du == -(u - m) dt + sqrt(2) K^1/2 dW`` over time ``t``. It is
    reversible with respect to the reference, which it preserves exactly.
    pCN's ``beta`` is ``sin(a) == sqrt(1 - exp(-2 t))``.
    """

    touches = frozenset({POSITION})
    preserves = frozenset({REFERENCE})
    deterministic = False
    reversible = False

    def angle(self, time: float, /) -> float:
        """``arccos(exp(-time))``.

        Args:
            time: the time step.

        Returns:
            The rotation angle of the step.
        """
        return float(np.arccos(np.exp(-float(time))))

    def apply(
        self, state: State, time: float, /, *, rng: Generator | None = None
    ) -> tuple[State, None]:
        """Take the step.

        Args:
            state: the state.
            time: the time step.
            rng: the generator for the fresh noise.

        Returns:
            The new state, nothing cached, and no energy change.
        """
        target = self._target
        space = target.model_space
        component = target.component(state)
        mean = component.expectation
        a = self.angle(time)
        noise = component.fluctuation(rng=rng)
        offset = space.subtract(state.position, mean)
        position = space.axpy(
            float(np.sin(a)),
            noise,
            space.axpy(float(np.cos(a)), offset, space.copy(mean)),
        )
        return state.with_position(position), None


class GradientDrift(Flow):
    """The data force on the position: ``u' == u - t K grad Phi_ref(u)``.

    The overdamped dynamics' data piece. Deterministic but not reversible,
    so it is corrected by a Metropolis-Hastings step that knows the
    proposal's transition density, never inside a deterministic block.
    """

    touches = frozenset({POSITION})
    preserves = frozenset()
    deterministic = True
    reversible = False
    uses_gradient = True

    def apply(
        self, state: State, time: float, /, *, rng: Generator | None = None
    ) -> tuple[State, float | None]:
        """Drift for ``time``.

        Args:
            state: the state.
            time: the drift time.
            rng: unused.

        Returns:
            The new state and no energy change; ``+inf`` with the state
            unchanged where the potential cannot be evaluated.
        """
        target = self._target
        if target.value(state) == float("inf"):
            return state, float("inf")
        space = target.model_space
        position = space.axpy(
            -float(time),
            target.preconditioned_gradient(state),
            space.copy(state.position),
        )
        return state.with_position(position), None


class LabelJump(Flow):
    """A Metropolized change of component, over a mixture reference.

    A chain over a mixture reference follows one component at a time, and a
    Gibbs step on the label would leave it there when the components are
    well apart, since the label is then all but fixed by the position. This
    move proposes a new component ``k`` from the mixture's weights and
    carries the state to it by
    :meth:`~pygeoinf.inference.gaussian.MixtureReference.transport`, which
    maps component ``j`` onto ``k`` exactly, the velocity with it. The map's
    pushforward is exact and the label is proposed from the weights, so the
    acceptance is ``exp(Phi_ref(u) - Phi_ref(u'))``: where the mixture fits
    the posterior, a jump between modes is accepted as readily as the fit is
    good. It preserves the target, and so the reference when there is no
    likelihood; the time step plays no part.
    """

    touches = frozenset({POSITION, VELOCITY, LABEL})
    preserves = frozenset({REFERENCE, TARGET})
    deterministic = False
    reversible = False

    def __init__(self, target: Target, /) -> None:
        if not target.is_mixture:
            raise ValueError("A label jump needs a mixture reference.")
        super().__init__(target)

    def apply(
        self, state: State, time: float, /, *, rng: Generator | None = None
    ) -> tuple[State, None]:
        """Propose a component and carry the state to it, accepted or not.

        Args:
            state: the state, with a label.
            time: unused.
            rng: the generator for the proposal and the acceptance.

        Returns:
            The new state, or the old one on a rejection, and no energy
            change.
        """
        from ...algebra.spaces import resolve_rng

        rng = resolve_rng(rng)
        target = self._target
        mixture = target.reference
        source = state.label
        destination = int(rng.choice(len(mixture.weights), p=mixture.weights))
        if destination == source:
            return state, None
        position = mixture.transport(state.position, source, destination)
        velocity = state.velocity
        if velocity is not None:
            velocity = mixture.transport(velocity, source, destination, centered=True)
        proposal = State(position, velocity, label=destination)
        current = target.value(state)
        proposed = target.value(proposal)
        if proposed == float("inf"):
            return state, None
        if np.log(rng.random()) < current - proposed:
            return proposal, None
        return state, None


class KernelFlow(Flow):
    """A flow that runs a kernel: any update that leaves the target invariant.

    Stochastic and target-preserving, so it sits outside every correction,
    in a word that a :class:`~pygeoinf.inference.sampling.corrections.Sweep`
    applies as it stands. A subclass says which kernel runs and on what;
    :class:`GibbsBlock` is the first, a kernel on one summand's conditional
    target.
    """

    touches = frozenset({POSITION})
    preserves = frozenset({REFERENCE, TARGET})
    deterministic = False
    reversible = False


class GibbsBlock(KernelFlow):
    """One summand updated against its conditional, the others held: a
    Gibbs block.

    At each application the block's target is built from the state
    (:meth:`~pygeoinf.inference.sampling.target.Target.block`), its kernel is
    built from the recipe at the block's step and run for the recipe's
    number of steps, and the summand is written back. Every kernel leaves
    its target invariant, so the block leaves the conditional, and so the
    whole target, invariant; with no likelihood, the reference's conditional
    and so the reference. A velocity, for an inertial block sampler, is drawn
    afresh at every visit. The likelihood's value is carried in and out of
    the block, so a visit costs no forward evaluation beyond the kernel's
    own. The step's mean acceptance, and whether the summand moved, are left
    on the new state's cache for the sweep to read.

    Args:
        target: the whole target, on a direct sum.
        key: the summand, by position or by label.
        recipe: what samples the block: a
            :class:`~pygeoinf.inference.sampling.presets.BlockRecipe`, or any
            object with ``kernel(target, step)``, ``steps`` and
            ``uses_gradient``.
        step: the block kernel's time step.

    Raises:
        ValueError: for a step that is not positive and finite.
    """

    def __init__(
        self, target: Target, key: int | str, recipe: Any, /, *, step: float
    ) -> None:
        if not 0.0 < step < float("inf"):
            raise ValueError(f"A step is positive and finite, got {step}.")
        super().__init__(target)
        self._index = target.model_space.index(key)
        self._recipe = recipe
        self._step = float(step)
        self.uses_gradient = bool(recipe.uses_gradient)

    @property
    def index(self) -> int:
        """The summand's position."""
        return self._index

    @property
    def recipe(self) -> Any:
        """What samples the block."""
        return self._recipe

    @property
    def step(self) -> float:
        """The block kernel's time step."""
        return self._step

    def apply(
        self, state: State, time: float, /, *, rng: Generator | None = None
    ) -> tuple[State, None]:
        """Update the summand by its kernel, the others held.

        Args:
            state: the whole state.
            time: unused; the block has its own step.
            rng: the generator.

        Returns:
            The new state, and no energy change.
        """
        from ...algebra.spaces import resolve_rng

        rng = resolve_rng(rng)
        target = self._target
        block = target.block(self._index, state)
        kernel = self._recipe.kernel(block, self._step)
        summand = target.model_space.component(state.position, self._index)
        inner = kernel.initial_state(summand, rng=rng)
        enter_block(target, block, self._index, state, inner)
        acceptances = []
        moved = False
        for _ in range(int(self._recipe.steps)):
            step = kernel.step(inner, rng=rng)
            inner = step.state
            acceptances.append(step.info.acceptance)
            moved = moved or step.info.accepted
        result = leave_block(target, block, self._index, state, inner)
        result.cache[("block acceptance", self._index)] = float(np.mean(acceptances))
        result.cache[("block moved", self._index)] = moved
        return result, None

    def __repr__(self) -> str:
        return f"GibbsBlock({self._index}, step={self._step:g})"


def time_from_pcn_beta(beta: float, /) -> float:
    """The time step of a pCN move with the literature's ``beta``.

    pCN proposes ``sqrt(1 - beta^2) u + beta xi``; the prior
    Ornstein-Uhlenbeck step does so with ``beta == sqrt(1 - exp(-2 h))``.

    Args:
        beta: in ``(0, 1)``.

    Returns:
        ``h == -log(1 - beta^2) / 2``.

    Raises:
        ValueError: for a ``beta`` outside ``(0, 1)``.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError(f"pCN's beta lies in (0, 1), got {beta}.")
    return float(-0.5 * np.log1p(-beta * beta))


def pcn_beta(time: float, /) -> float:
    """pCN's ``beta`` for a time step, ``sqrt(1 - exp(-2 h))``.

    Args:
        time: the time step ``h``, positive.

    Returns:
        ``beta``.

    Raises:
        ValueError: for a time that is not positive.
    """
    if not time > 0.0:
        raise ValueError(f"A time step is positive, got {time}.")
    return float(np.sqrt(-np.expm1(-2.0 * time)))


def time_from_crank_nicolson(delta: float, /) -> float:
    """The time step of a Crank-Nicolson move with the literature's ``delta``.

    The Crank-Nicolson discretization of the prior's Ornstein-Uhlenbeck
    dynamics contracts by ``(2 - delta) / (2 + delta)``; the exact step does
    so by ``exp(-h)``.

    Args:
        delta: in ``(0, 2)``; ``delta == 2`` is the independence sampler,
            an infinite time.

    Returns:
        ``h == 2 artanh(delta / 2)``.

    Raises:
        ValueError: for a ``delta`` outside ``(0, 2)``.
    """
    if not 0.0 < delta < 2.0:
        raise ValueError(f"Crank-Nicolson's delta lies in (0, 2), got {delta}.")
    return float(2.0 * np.arctanh(0.5 * delta))


def crank_nicolson_delta(time: float, /) -> float:
    """Crank-Nicolson's ``delta`` for a time step, ``2 tanh(h / 2)``.

    Args:
        time: the time step ``h``, positive.

    Returns:
        ``delta``.

    Raises:
        ValueError: for a time that is not positive.
    """
    if not time > 0.0:
        raise ValueError(f"A time step is positive, got {time}.")
    return float(2.0 * np.tanh(0.5 * time))
