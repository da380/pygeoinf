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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from numpy.random import Generator

from .target import State, Target

__all__ = [
    "Flow",
    "GradientDrift",
    "LikelihoodKick",
    "MomentumRefresh",
    "PriorOU",
    "PriorRotation",
]

#: Names for what a flow can touch and preserve.
POSITION = "position"
VELOCITY = "velocity"
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
        mean = self._target.reference.expectation
        offset = space.subtract(state.position, mean)
        c, s = np.cos(time), np.sin(time)
        position = space.axpy(
            float(s), state.velocity, space.axpy(float(c), offset, space.copy(mean))
        )
        velocity = space.axpy(
            float(-s), offset, space.scale(float(c), state.velocity)
        )
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
        change = -0.5 * float(time) * (
            space.inner_product(gradient, state.velocity)
            + space.inner_product(gradient, velocity)
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
        noise = self._target.reference.fluctuation(rng=rng)
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
        mean = target.reference.expectation
        a = self.angle(time)
        noise = target.reference.fluctuation(rng=rng)
        offset = space.subtract(state.position, mean)
        position = space.axpy(
            float(np.sin(a)), noise, space.axpy(float(np.cos(a)), offset, space.copy(mean))
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

