"""
What a sampler samples: the residual potential against a reference, and the
state a chain carries.

A function-space sampler never sees the posterior's density against the
space's volume, which does not exist in the limit. It sees the posterior
against a reference Gaussian equivalent to the prior,

.. code-block:: text

    d mu / d nu (u) == exp(-Phi_ref(u)) / Z,    Phi_ref == Phi / T + log d nu / d mu_0

with ``Phi`` the likelihood potential of the forward problem and ``T`` a
temperature. The reference is solved exactly by the dynamics; ``Phi_ref`` is
all the integrator and the Metropolis step are left with. :class:`Target`
holds it and is the only caller of the forward operator.

:class:`State` is the chain's position and, for the inertial dynamics, its
velocity, with the evaluations at the position cached on the record itself:
a flow that moves the position makes a new record with nothing cached, one
that changes only the velocity keeps the cache, and a rejected proposal
restores the old record whole. So a gradient computed for one kick serves the
next, and a rejection costs nothing to undo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...algebra.linearization import QuadraticModel
from ...algebra.operators import EvaluationError, Functional, LinearFunctional
from ...probability.absolutely_continuous import AbsolutelyContinuousMeasure
from ...probability.gaussian import GaussianMeasure
from ..gaussian import GaussianReference, MixtureReference
from ..laplace import LaplaceApproximation

if TYPE_CHECKING:  # pragma: no cover
    from ...geometry.sets import Subset
    from ..problem import ForwardProblem

__all__ = ["State", "Target"]


@dataclass(frozen=True, eq=False)
class State:
    """A chain's state: a position, perhaps a velocity, and what is known there.

    Immutable in its vectors. The cache holds evaluations at the position
    and belongs to this record alone: :meth:`with_position` starts a new
    record with nothing cached, :meth:`with_velocity` keeps the cache, the
    position being unchanged.

    Attributes:
        position: the model.
        velocity: the velocity, or ``None`` for dynamics without one.
        label: the component of a mixture reference the chain is following,
            or ``None`` for a Gaussian reference.
    """

    position: Any
    velocity: Any = None
    cache: dict = field(default_factory=dict, repr=False)
    label: int | None = None

    def with_position(self, position: Any, /) -> "State":
        """A record at a new position, its velocity and label kept and
        nothing cached.

        Args:
            position: the new model.

        Returns:
            The new record.
        """
        return State(position, self.velocity, label=self.label)

    def with_velocity(self, velocity: Any, /) -> "State":
        """A record with a new velocity at the same position, cache kept.

        Args:
            velocity: the new velocity.

        Returns:
            The new record.
        """
        return State(self.position, velocity, self.cache, self.label)

    def with_label(self, label: int, /) -> "State":
        """A record following another component, at the same position and
        velocity, cache kept: what is cached per component is filed under
        its label.

        Args:
            label: the component.

        Returns:
            The new record.
        """
        return State(self.position, self.velocity, self.cache, label)

    def moved(self, position: Any, velocity: Any, /) -> "State":
        """A record at a new position and velocity, its label kept and
        nothing cached.

        Args:
            position: the new model.
            velocity: the new velocity.

        Returns:
            The new record.
        """
        return State(position, velocity, label=self.label)


class _Evaluation:
    """The potential and its gradient at one position, as far as computed."""

    __slots__ = ("value", "gradient")

    def __init__(self, value: float, gradient: Any) -> None:
        self.value = value
        self.gradient = gradient


class Target:
    """The residual potential ``Phi_ref == Phi / T + log d nu / d mu_0``.

    Built from a forward problem, the data and a reference. The likelihood
    potential is the problem's (:meth:`~pygeoinf.inference.problem.ForwardProblem.potential`),
    evaluated through one ``forward.at`` wherever a gradient is wanted, so
    that value and derivative cost one forward evaluation together; where
    only the value is wanted, the forward's value alone is computed. A
    forward that raises :class:`~pygeoinf.algebra.operators.EvaluationError`,
    or a position outside ``constraint``, has potential ``+inf``, so a
    proposal there is rejected rather than the run being lost.

    Evaluations are cached on the :class:`State`: the likelihood's at
    temperature one, shared by every tempered copy made by
    :meth:`with_temperature`, so that a tempering schedule re-reads rather
    than recomputes it; and the reference's density and the preconditioned
    gradient per target.

    Args:
        problem: the forward problem, with a Gaussian error measure.
        data: the observations.
        reference: a :class:`~pygeoinf.inference.gaussian.GaussianReference`
            or :class:`~pygeoinf.inference.gaussian.MixtureReference` on the
            model space, or a Gaussian prior, which is taken as its own
            reference; or a
            :class:`~pygeoinf.inference.laplace.LaplaceApproximation` on the
            same problem, for a reference that follows the state: the
            Laplace reference at each position, built on first use and
            cached on the state.
        temperature: divides the likelihood potential. One is the posterior.
        constraint: a subset of the model space outside which the potential
            is ``+inf``, checked before the forward is called.

    Raises:
        ValueError: if the reference does not live on the model space, a
            Laplace approximation is of another problem, or for a
            non-positive temperature.
        TypeError: if the reference is not a ``GaussianReference``, a
            ``MixtureReference``, a ``GaussianMeasure`` or a
            ``LaplaceApproximation``.
    """

    def __init__(
        self,
        problem: "ForwardProblem",
        data: Any,
        reference: (
            GaussianReference | MixtureReference | GaussianMeasure | LaplaceApproximation
        ),
        /,
        *,
        temperature: float = 1.0,
        constraint: "Subset | None" = None,
    ) -> None:
        local = None
        if isinstance(reference, LaplaceApproximation):
            if reference.estimator.forward_problem is not problem:
                raise ValueError(
                    "The Laplace approximation must be built on the target's "
                    "forward problem."
                )
            local = reference
            reference = reference.estimator.prior
        reference = _as_reference(reference)
        if reference.domain != problem.model_space:
            raise ValueError(
                f"The reference lives on {reference.domain!r}, not on the "
                f"model space {problem.model_space!r}."
            )
        potential = problem.potential(data)
        self._init(
            potential,
            reference,
            temperature=temperature,
            constraint=constraint,
            has_gradient=problem.forward_operator.has_derivative,
            source=_Source(potential),
            local=local,
            data=data,
        )

    def _init(
        self,
        potential: Functional,
        reference: GaussianReference | MixtureReference,
        /,
        *,
        temperature: float,
        constraint: "Subset | None",
        has_gradient: bool,
        source: "_Source",
        local: LaplaceApproximation | None = None,
        data: Any = None,
    ) -> None:
        if not temperature > 0.0:
            raise ValueError(f"A temperature is positive, got {temperature}.")
        self._local = local
        self._data = data
        self._potential = potential
        self._reference = reference
        self._temperature = float(temperature)
        self._constraint = constraint
        self._has_gradient = has_gradient
        self._source = source

    @classmethod
    def from_reference(
        cls, reference: GaussianReference | MixtureReference | GaussianMeasure, /
    ) -> "Target":
        """The target that is the reference itself, ``Phi_ref == 0``.

        Its likelihood potential is ``-log d nu / d mu_0``, which cancels the
        reference's density, so the target is ``nu``; for the prior as its
        own reference that is ``Phi == 0``.

        What a scheme is checked against: with nothing to correct, a word
        built from the reference's exact flows must leave the reference
        invariant at any step.

        Args:
            reference: the reference, or a prior taken as its own.

        Returns:
            The target.
        """
        reference = _as_reference(reference)
        space = reference.domain
        if reference.rank == 0:

            def linearize(x: Any) -> QuadraticModel:
                return QuadraticModel(
                    x, 0.0, LinearFunctional.from_representer(space, space.zero())
                )

            potential = Functional.from_callables(
                space, lambda x: 0.0, linearize=linearize
            )
        else:
            potential = -reference.relative_log_density
        target = cls.__new__(cls)
        target._init(
            potential,
            reference,
            temperature=1.0,
            constraint=None,
            has_gradient=True,
            source=_Source(potential),
        )
        return target

    # ----------------------------------------------------------------- #
    #                              Reading                              #
    # ----------------------------------------------------------------- #

    @property
    def reference(self) -> GaussianReference | MixtureReference:
        """The reference the dynamics solves exactly."""
        return self._reference

    @property
    def is_mixture(self) -> bool:
        """Whether the reference is a mixture, its states carrying a label."""
        return isinstance(self._reference, MixtureReference)

    @property
    def is_local(self) -> bool:
        """Whether the reference follows the state: the Laplace reference at
        each position, :attr:`reference` being the prior."""
        return self._local is not None

    @property
    def local(self) -> LaplaceApproximation | None:
        """The Laplace approximation a local reference is built from."""
        return self._local

    def component(self, state: State, /) -> GaussianReference:
        """The Gaussian reference the state's flows solve: the reference
        itself, the mixture's component the state is labelled with, or the
        local reference at the state's position.

        Args:
            state: the chain's state.

        Returns:
            The Gaussian reference.

        Raises:
            ValueError: for a mixture reference and a state with no label.
        """
        if self._local is not None:
            key = ("local", self._source)
            cached = state.cache.get(key)
            if cached is None:
                cached = self._local.reference(state.position, self._data)
                state.cache[key] = cached
            return cached
        if not isinstance(self._reference, MixtureReference):
            return self._reference
        if state.label is None:
            raise ValueError(
                "A state on a mixture reference carries the label of a "
                "component; start it with the dynamics' initial_state."
            )
        return self._reference.components[state.label]

    @property
    def prior(self) -> GaussianMeasure:
        """The prior the reference is equivalent to."""
        return self._reference.prior

    @property
    def model_space(self) -> Any:
        """Where the positions live."""
        return self._reference.domain

    @property
    def temperature(self) -> float:
        """What the likelihood potential is divided by."""
        return self._temperature

    @property
    def has_gradient(self) -> bool:
        """Whether the forward operator supplies a derivative."""
        return self._has_gradient

    @property
    def potential_functional(self) -> Functional:
        """The likelihood potential ``Phi`` at temperature one."""
        return self._potential

    def with_temperature(self, temperature: float, /) -> "Target":
        """The same target at another temperature, sharing its evaluations.

        Args:
            temperature: the new temperature.

        Returns:
            The tempered target. A state evaluated under one is not
            evaluated again under the other.
        """
        target = type(self).__new__(type(self))
        target._init(
            self._potential,
            self._reference,
            temperature=temperature,
            constraint=self._constraint,
            has_gradient=self._has_gradient,
            source=self._source,
            local=self._local,
            data=self._data,
        )
        return target

    def residual_functional(self) -> Functional:
        """``Phi_ref`` as a functional, for a measure to be built on it."""
        return (1.0 / self._temperature) * self._potential + (
            self._reference.relative_log_density
        )

    def log_density_against_prior(self) -> Functional:
        """``-Phi / T``: the target's log density against the prior.

        Returns:
            The functional, up to the log evidence.
        """
        return (-1.0 / self._temperature) * self._potential

    @property
    def measure(self) -> AbsolutelyContinuousMeasure:
        """The target as a measure: ``exp(-Phi_ref)`` against the reference."""
        return AbsolutelyContinuousMeasure(self._reference, -self.residual_functional())

    # ----------------------------------------------------------------- #
    #                            Evaluations                            #
    # ----------------------------------------------------------------- #

    def potential(self, state: State, /) -> float:
        """``Phi`` at the state's position, at temperature one.

        Args:
            state: the chain's state.

        Returns:
            The likelihood potential, ``+inf`` where the forward fails or the
            constraint excludes the position.
        """
        return self._likelihood(state, gradient=False).value

    def value(self, state: State, /) -> float:
        """``Phi_ref`` at the state's position.

        Args:
            state: the chain's state.

        Returns:
            The residual potential, ``+inf`` where the likelihood is.
        """
        phi = self._likelihood(state, gradient=False).value
        if phi == float("inf"):
            return phi
        return phi / self._temperature + self._density(state).value

    def gradient(self, state: State, /) -> Any:
        """The representer of ``Phi_ref``'s derivative at the position.

        Args:
            state: the chain's state.

        Returns:
            A vector of the model space.

        Raises:
            ValueError: if the forward operator has no derivative, or the
                potential is infinite at the position, where it has none.
        """
        if not self._has_gradient:
            raise ValueError(
                "The forward operator supplies no derivative, so this target "
                "has no gradient; use a sampler that needs none."
            )
        likelihood = self._likelihood(state, gradient=True)
        if likelihood.gradient is None:
            raise ValueError("The potential is infinite here and has no gradient.")
        space = self.model_space
        return space.axpy(
            1.0 / self._temperature,
            likelihood.gradient,
            space.copy(self._density(state).gradient),
        )

    def preconditioned_gradient(self, state: State, /) -> Any:
        """``K grad Phi_ref``, with ``K`` the covariance of the state's
        Gaussian reference (:meth:`component`).

        The data force of every gradient-based sampler, and the same vector
        whatever the space's metric: the metric that enters the gradient
        leaves through the covariance.

        Args:
            state: the chain's state.

        Returns:
            A vector of the model space.

        Raises:
            ValueError: as for :meth:`gradient`.
        """
        key = ("preconditioned", self, state.label)
        cached = state.cache.get(key)
        if cached is None:
            cached = self.component(state).covariance(self.gradient(state))
            state.cache[key] = cached
        return cached

    def _likelihood(self, state: State, /, *, gradient: bool) -> _Evaluation:
        """``Phi`` and, when asked, its gradient, cached at temperature one."""
        cache = state.cache
        entry = cache.get(self._source)
        if entry is not None and (not gradient or entry.gradient is not None):
            return entry
        if entry is not None and entry.value == float("inf"):
            return entry
        position = state.position
        if self._constraint is not None and not self._constraint.contains(position):
            entry = _Evaluation(float("inf"), None)
        else:
            try:
                if gradient:
                    model = self._potential.at(position)
                    entry = _Evaluation(float(model.value), model.gradient)
                else:
                    entry = _Evaluation(float(self._potential(position)), None)
            except EvaluationError:
                entry = _Evaluation(float("inf"), None)
        cache[self._source] = entry
        return entry

    def _density(self, state: State, /) -> _Evaluation:
        """``log d nu / d mu_0`` and its gradient, cached per reference: the
        local reference's own at its position, for a local reference."""
        local = self._local is not None
        key = ("local density", self._source) if local else ("reference", self._reference)
        entry = state.cache.get(key)
        if entry is None:
            reference = self.component(state) if local else self._reference
            if reference.rank == 0:
                space = self.model_space
                entry = _Evaluation(0.0, space.zero())
            else:
                model = reference.relative_log_density.at(state.position)
                entry = _Evaluation(float(model.value), model.gradient)
            state.cache[key] = entry
        return entry

    def __repr__(self) -> str:
        return f"Target(rank={self._reference.rank}, temperature={self._temperature:g})"


class _Source:
    """The identity a likelihood's cached evaluations are filed under."""

    __slots__ = ("potential",)

    def __init__(self, potential: Functional) -> None:
        self.potential = potential


def _as_reference(
    reference: GaussianReference | MixtureReference | GaussianMeasure,
) -> GaussianReference | MixtureReference:
    """A reference, taking a plain Gaussian as the prior and its own reference.

    Raises:
        TypeError: for anything that is not a Gaussian or a mixture reference.
    """
    if isinstance(reference, (GaussianReference, MixtureReference)):
        return reference
    if isinstance(reference, GaussianMeasure):
        return GaussianReference.from_prior(reference)
    raise TypeError(
        f"A target is built on a GaussianReference, a MixtureReference or a "
        f"Gaussian prior, not a "
        f"{type(reference).__name__}."
    )
