"""
Presets: the common samplers, assembled from the four layers.

Each preset fixes a dynamics, a word and a correction, and is a
:class:`~pygeoinf.inference.estimators.MeasureEstimator`: built on a forward
problem and a prior, and called on data, it returns the sampled posterior.

.. code-block:: text

    preset              dynamics           word, block            correction
    PCN                 PreconditionedOU   "O"                    Metropolis-Hastings
    MALA                Overdamped         "BO"                   Metropolis-Hastings
    HMC                 Hamiltonian        "OBAB", "BAB" x L      Metropolized block
    PreconditionedHMC   Underdamped        "OBAB", "BAB" x L      Metropolized block, flip
    GeometricMALA       as MALA, about a GaussianReference
    GeometricHMC        as HMC, about a GaussianReference
    SMC                 a chain preset's kernel, moved along the tempering path

So ``HMC(problem, prior, step=0.2, length=5)(data, steps=2000, rng=rng)`` is
the same sampler as

.. code-block:: python

    dynamics = Hamiltonian(Target(problem, data, prior))
    scheme = SplittingScheme(dynamics, "OBAB", step=0.2)
    kernel = MetropolizedBlock(scheme.block("BAB", repeat=5)).within(scheme)
    posterior = Chains(4)(kernel, start=prior, steps=2000, rng=rng)

with the step adapted towards the preset's acceptance during burn-in, and
:meth:`ChainSampler.kernel` returns that kernel for a given data set, to run
or to inspect. A geometric preset is its plain counterpart with a reference
other than the prior: the rotation and refresh then solve that reference
exactly, so the sampler moves at the posterior's scale in the directions the
data inform and at the prior's elsewhere. The reference is built from the
data, by :meth:`~pygeoinf.inference.gaussian.LinearGaussianInversion.reference`
or :meth:`~pygeoinf.inference.laplace.LaplaceApproximation.reference`, so a
geometric preset is meant to be called on the data its reference was built
from. Any reference leaves the chain exact; a poor one only makes it slow.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from numpy.random import Generator

from ...algebra.operators import LinearOperator
from ...algebra.spaces import HilbertSpace
from ...probability.gaussian import GaussianMeasure
from ..estimators import MeasureEstimator, _PushedMeasureEstimator
from ..gaussian import GaussianReference
from .corrections import Kernel, MetropolisHastings, MetropolizedBlock, Step
from .dynamics import Hamiltonian, Overdamped, PreconditionedOU, Underdamped
from .population import Chains, StepSizeAdaptation, Tempering
from .results import SampledPosterior, WeightedEnsemble
from .schemes import SplittingScheme
from .target import Target

if TYPE_CHECKING:  # pragma: no cover
    from ...geometry.sets import Subset
    from ..problem import ForwardProblem

__all__ = [
    "ChainSampler",
    "GeometricHMC",
    "GeometricMALA",
    "HMC",
    "MALA",
    "PCN",
    "PreconditionedHMC",
    "SMC",
]


class ChainSampler(MeasureEstimator):
    """A preset run as independent chains: data to a sampled posterior.

    A subclass names its kernel through :meth:`_kernel`; this class holds
    the problem, the prior, the reference and the step, builds the target
    for the data, and runs :class:`~pygeoinf.inference.sampling.population.Chains`.

    Args:
        problem: the forward problem, with a Gaussian error measure.
        prior: the Gaussian prior on the model space.
        step: the time step ``h``, where the burn-in's adaptation starts.
        reference: a :class:`~pygeoinf.inference.gaussian.GaussianReference`
            over the prior, which the dynamics solves exactly; the prior if
            omitted.
        acceptance: the acceptance probability the step is adapted towards
            during burn-in, or ``None`` to keep the step fixed.
        constraint: a subset of the model space outside which the
            posterior has no mass; a proposal outside it is rejected.

    Raises:
        TypeError: if the reference is not a ``GaussianReference``.
        ValueError: if the prior is not on the model space, the reference
            is not over the prior, the step is not positive and finite, the
            acceptance lies outside ``(0, 1)``, or the sampler needs a
            gradient the forward operator does not have.
    """

    #: Whether the kernel differentiates the forward operator.
    uses_gradient: bool = False

    def __init__(
        self,
        problem: "ForwardProblem",
        prior: GaussianMeasure,
        /,
        *,
        step: float,
        reference: GaussianReference | None = None,
        acceptance: float | None,
        constraint: "Subset | None" = None,
    ) -> None:
        if prior.domain != problem.model_space:
            raise ValueError("The prior is not defined on the model space.")
        if reference is None:
            reference = GaussianReference.from_prior(prior)
        elif not isinstance(reference, GaussianReference):
            raise TypeError(
                "A reference is a GaussianReference, which the estimators hand "
                f"out; got {type(reference).__name__}."
            )
        elif reference.prior is not prior:
            raise ValueError("The reference must be built over the sampler's prior.")
        if not 0.0 < step < float("inf"):
            raise ValueError(f"A step is positive and finite, got {step}.")
        if self.uses_gradient and not problem.forward_operator.has_derivative:
            raise ValueError(
                f"{type(self).__name__} follows the gradient, and the forward "
                "operator has no derivative; PCN needs none."
            )
        self._problem = problem
        self._prior = prior
        self._reference = reference
        self._step = float(step)
        self._adaptation = (
            None
            if acceptance is None
            else StepSizeAdaptation(acceptance=acceptance, maximum=self._longest_step())
        )
        self._constraint = constraint

    # ----------------------------------------------------------------- #
    #                             The pieces                            #
    # ----------------------------------------------------------------- #

    @property
    def forward_problem(self) -> "ForwardProblem":
        """The forward problem."""
        return self._problem

    @property
    def prior(self) -> GaussianMeasure:
        """The prior."""
        return self._prior

    @property
    def reference(self) -> GaussianReference:
        """The reference the dynamics solves exactly."""
        return self._reference

    @property
    def step(self) -> float:
        """The time step ``h`` the chains start at."""
        return self._step

    @property
    def adaptation(self) -> StepSizeAdaptation | None:
        """How the step adapts during burn-in, or ``None`` for a fixed step."""
        return self._adaptation

    @property
    def data_space(self) -> HilbertSpace:
        """Where the data live."""
        return self._problem.data_space

    @property
    def property_space(self) -> HilbertSpace:
        """The model space: the chains sample the model itself."""
        return self._problem.model_space

    def target(self, data: Any, /) -> Target:
        """The residual potential for the data, against the reference.

        Args:
            data: the observations.

        Returns:
            The target.
        """
        return Target(self._problem, data, self._reference, constraint=self._constraint)

    def kernel(self, data: Any, /) -> Kernel:
        """The kernel the chains run for the data, at the starting step.

        Args:
            data: the observations.

        Returns:
            The kernel.
        """
        return self._kernel(self.target(data))

    @abstractmethod
    def _kernel(self, target: Target, /) -> Kernel:
        """The preset's kernel on a target."""

    def _longest_step(self) -> float:
        """The step beyond which the adaptation gains nothing.

        For an Ornstein-Uhlenbeck step, ten: ``exp(-10)`` of the old point
        is left, and the proposal is as good as independent.
        """
        return 10.0

    # ----------------------------------------------------------------- #
    #                              Running                              #
    # ----------------------------------------------------------------- #

    def __call__(
        self,
        data: Any,
        /,
        *,
        steps: int,
        chains: int = 4,
        burn_in: int = 0,
        start: Any = None,
        thin: int | None = None,
        record: LinearOperator | None = None,
        record_names: list[str] | None = None,
        rng: Generator | None = None,
        n_jobs: int | None = None,
        callback: Callable[[int, int, Step], None] | None = None,
    ) -> SampledPosterior:
        """Sample the posterior for the data.

        Args:
            data: the observations.
            steps: how many steps each chain keeps, after its burn-in.
            chains: how many independent chains.
            burn_in: how many steps each chain takes first and discards,
                adapting its step towards the preset's acceptance.
            start: where the chains start: a model, or a measure each chain
                draws its start from; the reference if omitted.
            thin: store the position every this many kept steps, for
                :meth:`~pygeoinf.inference.sampling.results.SampledPosterior.sample_thinned`;
                nothing is stored if omitted.
            record: a small linear map from the model space to a space with
                coordinates, whose components are recorded at every step.
            record_names: names for the recorded components.
            callback: called as ``callback(chain, iteration, step)`` after
                every step, where the chain runs.

        Returns:
            The sampled posterior.

        Raises:
            ValueError: as for
                :class:`~pygeoinf.inference.sampling.population.Chains`.
        """
        return Chains(chains)(
            self.kernel(data),
            start=self._reference if start is None else start,
            steps=steps,
            burn_in=burn_in,
            adapt=self._adaptation,
            thin=thin,
            record=record,
            record_names=record_names,
            rng=rng,
            n_jobs=n_jobs,
            callback=callback,
        )

    def push_forward(self, operator: LinearOperator, /) -> MeasureEstimator:
        """The estimator of the law of ``T m`` rather than of ``m``.

        Each sampled posterior is pushed through ``T``. To follow a property
        along the chains, with its diagnostics, record it instead.

        Args:
            operator: the property map ``T``, from the model space.

        Returns:
            The measure estimator of the property.

        Raises:
            ValueError: if ``T`` does not act on the model space.
        """
        return _PushedMeasureEstimator(self, operator)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(step={self._step:g})"


class PCN(ChainSampler):
    """Preconditioned Crank-Nicolson: the reference's Ornstein-Uhlenbeck step.

    The proposal ``m + cos(theta) (u - m) + sin(theta) xi``, with ``xi`` a
    centred draw of the reference and ``cos(theta) == exp(-h)``, accepted on
    the residual potential alone. Needs no gradient. In the literature's
    parameter, ``beta == sqrt(1 - exp(-2 h))``.

    Args:
        problem: the forward problem, with a Gaussian error measure.
        prior: the Gaussian prior.
        step: the time step ``h``.
        reference: the reference; the prior if omitted.
        acceptance: the acceptance to adapt towards during burn-in, about a
            quarter by default; ``None`` keeps the step fixed.
        constraint: where the posterior has its mass.

    Raises:
        TypeError: as for :class:`ChainSampler`.
        ValueError: as for :class:`ChainSampler`.
    """

    def __init__(
        self,
        problem: "ForwardProblem",
        prior: GaussianMeasure,
        /,
        *,
        step: float,
        reference: GaussianReference | None = None,
        acceptance: float | None = 0.25,
        constraint: "Subset | None" = None,
    ) -> None:
        super().__init__(
            problem,
            prior,
            step=step,
            reference=reference,
            acceptance=acceptance,
            constraint=constraint,
        )

    def _kernel(self, target: Target, /) -> Kernel:
        return MetropolisHastings(SplittingScheme(PreconditionedOU(target), "O", step=self._step))


class MALA(ChainSampler):
    """The preconditioned Metropolis-adjusted Langevin algorithm.

    A drift down the preconditioned gradient for a time ``h``, then the
    reference's Ornstein-Uhlenbeck step for ``h``, corrected by
    Metropolis-Hastings with the proposal's transition density, which on a
    function space is a Cameron-Martin shift and needs no inverse
    covariance.

    Args:
        problem: the forward problem, with a Gaussian error measure and a
            differentiable forward operator.
        prior: the Gaussian prior.
        step: the time step ``h``.
        reference: the reference; the prior if omitted.
        acceptance: the acceptance to adapt towards during burn-in, 0.57 by
            default; ``None`` keeps the step fixed.
        constraint: where the posterior has its mass.

    Raises:
        TypeError: as for :class:`ChainSampler`.
        ValueError: as for :class:`ChainSampler`.
    """

    uses_gradient = True

    def __init__(
        self,
        problem: "ForwardProblem",
        prior: GaussianMeasure,
        /,
        *,
        step: float,
        reference: GaussianReference | None = None,
        acceptance: float | None = 0.57,
        constraint: "Subset | None" = None,
    ) -> None:
        super().__init__(
            problem,
            prior,
            step=step,
            reference=reference,
            acceptance=acceptance,
            constraint=constraint,
        )

    def _kernel(self, target: Target, /) -> Kernel:
        return MetropolisHastings(SplittingScheme(Overdamped(target), "BO", step=self._step))


class HMC(ChainSampler):
    """Hamiltonian Monte Carlo, the reference's part solved by rotation.

    A full velocity refresh from ``N(0, K)``, then ``length`` leapfrog steps
    ``"BAB"`` in which the kick follows the residual potential and the
    rotation solves the reference exactly, Metropolized as a block. A
    trajectory lasts ``length * h``; at ``length * h`` near ``pi / 2`` the
    rotation alone carries a point to an independent one.

    Args:
        problem: the forward problem, with a Gaussian error measure and a
            differentiable forward operator.
        prior: the Gaussian prior.
        step: the time step ``h``.
        length: the leapfrog steps per trajectory.
        reference: the reference; the prior if omitted.
        acceptance: the acceptance to adapt towards during burn-in, 0.65 by
            default; ``None`` keeps the step fixed.
        constraint: where the posterior has its mass.

    Raises:
        TypeError: as for :class:`ChainSampler`.
        ValueError: for a length below one, or as for
            :class:`ChainSampler`.
    """

    uses_gradient = True

    def __init__(
        self,
        problem: "ForwardProblem",
        prior: GaussianMeasure,
        /,
        *,
        step: float,
        length: int,
        reference: GaussianReference | None = None,
        acceptance: float | None = 0.65,
        constraint: "Subset | None" = None,
    ) -> None:
        if int(length) != length or length < 1:
            raise ValueError(f"A trajectory has at least one leapfrog step, got {length}.")
        self._length = int(length)
        super().__init__(
            problem,
            prior,
            step=step,
            reference=reference,
            acceptance=acceptance,
            constraint=constraint,
        )

    @property
    def length(self) -> int:
        """The leapfrog steps per trajectory."""
        return self._length

    def _longest_step(self) -> float:
        """A quarter turn over the trajectory, ``pi / (2 length)``."""
        return float(np.pi / (2 * self._length))

    def _kernel(self, target: Target, /) -> Kernel:
        scheme = SplittingScheme(Hamiltonian(target), "OBAB", step=self._step)
        return MetropolizedBlock(scheme.block("BAB", repeat=self._length)).within(scheme)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(step={self._step:g}, length={self._length})"


class PreconditionedHMC(ChainSampler):
    """HMC with a partial velocity refresh: underdamped Langevin, Metropolized.

    Before each trajectory the velocity is rotated towards a fresh draw by
    the angle ``arccos(exp(-gamma h))``, rather than redrawn, and a rejected
    trajectory reverses the velocity, so the chain keeps its direction
    between trajectories. The refresh acts once per trajectory, for one
    step: with the default single leapfrog step the chain is a discretised
    underdamped Langevin dynamics, exact by its correction.

    Args:
        problem: the forward problem, with a Gaussian error measure and a
            differentiable forward operator.
        prior: the Gaussian prior.
        step: the time step ``h``.
        friction: ``gamma``, positive and finite.
        length: the leapfrog steps per trajectory.
        reference: the reference; the prior if omitted.
        acceptance: the acceptance to adapt towards during burn-in, 0.8 by
            default; ``None`` keeps the step fixed.
        constraint: where the posterior has its mass.

    Raises:
        TypeError: as for :class:`ChainSampler`.
        ValueError: for a friction that is not positive and finite, a
            length below one, or as for :class:`ChainSampler`.
    """

    uses_gradient = True

    def __init__(
        self,
        problem: "ForwardProblem",
        prior: GaussianMeasure,
        /,
        *,
        step: float,
        friction: float = 1.0,
        length: int = 1,
        reference: GaussianReference | None = None,
        acceptance: float | None = 0.8,
        constraint: "Subset | None" = None,
    ) -> None:
        if not 0.0 < friction < float("inf"):
            raise ValueError(f"A friction is positive and finite, got {friction}.")
        if int(length) != length or length < 1:
            raise ValueError(f"A trajectory has at least one leapfrog step, got {length}.")
        self._friction = float(friction)
        self._length = int(length)
        super().__init__(
            problem,
            prior,
            step=step,
            reference=reference,
            acceptance=acceptance,
            constraint=constraint,
        )

    @property
    def friction(self) -> float:
        """The friction ``gamma``."""
        return self._friction

    @property
    def length(self) -> int:
        """The leapfrog steps per trajectory."""
        return self._length

    def _longest_step(self) -> float:
        """A quarter turn over the trajectory, ``pi / (2 length)``."""
        return float(np.pi / (2 * self._length))

    def _kernel(self, target: Target, /) -> Kernel:
        dynamics = Underdamped(target, friction=self._friction)
        scheme = SplittingScheme(dynamics, "OBAB", step=self._step)
        return MetropolizedBlock(scheme.block("BAB", repeat=self._length)).within(scheme)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(step={self._step:g}, friction={self._friction:g}, "
            f"length={self._length})"
        )


class GeometricMALA(MALA):
    """MALA about a Gaussian reference fitted to the posterior.

    :class:`MALA` with the reference given first: its prior is the
    reference's. The reference is built from the data, by an estimator's
    ``reference`` method, and the sampler is meant to be called on the same
    data.

    Args:
        problem: the forward problem, with a Gaussian error measure and a
            differentiable forward operator.
        reference: the reference, whose prior is the sampler's.
        step: the time step ``h``.
        acceptance: the acceptance to adapt towards during burn-in, 0.57 by
            default; ``None`` keeps the step fixed.
        constraint: where the posterior has its mass.

    Raises:
        TypeError: if the reference is not a ``GaussianReference``.
        ValueError: as for :class:`ChainSampler`.
    """

    def __init__(
        self,
        problem: "ForwardProblem",
        reference: GaussianReference,
        /,
        *,
        step: float,
        acceptance: float | None = 0.57,
        constraint: "Subset | None" = None,
    ) -> None:
        super().__init__(
            problem,
            _prior_of(reference),
            step=step,
            reference=reference,
            acceptance=acceptance,
            constraint=constraint,
        )


class GeometricHMC(HMC):
    """HMC about a Gaussian reference fitted to the posterior.

    :class:`HMC` with the reference given first: its prior is the
    reference's. Where the reference is close to the posterior, the rotation
    does nearly all the work and ``length * h`` near ``pi / 2`` gives nearly
    independent proposals that are nearly always accepted.

    Args:
        problem: the forward problem, with a Gaussian error measure and a
            differentiable forward operator.
        reference: the reference, whose prior is the sampler's.
        step: the time step ``h``.
        length: the leapfrog steps per trajectory.
        acceptance: the acceptance to adapt towards during burn-in, 0.65 by
            default; ``None`` keeps the step fixed.
        constraint: where the posterior has its mass.

    Raises:
        TypeError: if the reference is not a ``GaussianReference``.
        ValueError: for a length below one, or as for
            :class:`ChainSampler`.
    """

    def __init__(
        self,
        problem: "ForwardProblem",
        reference: GaussianReference,
        /,
        *,
        step: float,
        length: int,
        acceptance: float | None = 0.65,
        constraint: "Subset | None" = None,
    ) -> None:
        super().__init__(
            problem,
            _prior_of(reference),
            step=step,
            length=length,
            reference=reference,
            acceptance=acceptance,
            constraint=constraint,
        )


def _prior_of(reference: GaussianReference, /) -> GaussianMeasure:
    """The prior a reference is built over.

    Raises:
        TypeError: if the reference is not a ``GaussianReference``.
    """
    if not isinstance(reference, GaussianReference):
        raise TypeError(
            "A geometric sampler takes a GaussianReference, which the "
            f"estimators hand out; got {type(reference).__name__}."
        )
    return reference.prior


class SMC(MeasureEstimator):
    """Sequential Monte Carlo along the tempering path, with its evidence.

    Built on a chain preset whose kernel moves the particles: at each stage
    the particles are reweighted by ``exp(-delta_beta Phi)``, resampled, and
    moved by the preset's kernel, rebuilt for the stage's temperature, at
    the preset's step. The step is not adapted, so it should suit the
    posterior, where the path ends; the earlier stages, being wider, accept
    more often. The path starts from the prior, so the preset's reference
    must be the prior.

    Args:
        mutation: the chain preset whose kernel moves the particles.
        particles: how many particles.
        mutation_steps: how many kernel steps each particle takes per
            stage.
        ess_fraction: the effective sample size, as a fraction of the
            particles, that sets each step in ``beta``.

    Raises:
        TypeError: if the mutation is not a chain preset.
        ValueError: if the preset's reference is not its prior.
    """

    def __init__(
        self,
        mutation: ChainSampler,
        /,
        *,
        particles: int = 256,
        mutation_steps: int = 10,
        ess_fraction: float = 0.5,
    ) -> None:
        if not isinstance(mutation, ChainSampler):
            raise TypeError(
                f"The mutation is a chain preset, PCN say; got {type(mutation).__name__}."
            )
        if mutation.reference.rank != 0:
            raise ValueError(
                "Tempering starts from the prior, so the mutation's reference "
                "must be the prior."
            )
        self._mutation = mutation
        self._particles = particles
        self._mutation_steps = mutation_steps
        self._ess_fraction = ess_fraction

    @property
    def mutation(self) -> ChainSampler:
        """The chain preset that moves the particles."""
        return self._mutation

    @property
    def particles(self) -> int:
        """How many particles."""
        return int(self._particles)

    @property
    def data_space(self) -> HilbertSpace:
        """Where the data live."""
        return self._mutation.data_space

    @property
    def property_space(self) -> HilbertSpace:
        """The model space."""
        return self._mutation.property_space

    def __call__(
        self, data: Any, /, *, rng: Generator | None = None, n_jobs: int | None = None
    ) -> WeightedEnsemble:
        """The weighted particles for the data, with the log evidence.

        Args:
            data: the observations.

        Returns:
            The weighted ensemble, whose normalizer estimates ``log Z``.

        Raises:
            ValueError: for a population of fewer than two particles, a
                fraction outside ``(0, 1)`` or a step count below one, as
                for :class:`~pygeoinf.inference.sampling.population.Tempering`.
        """
        kernel = self._mutation.kernel(data)
        tempering = Tempering(
            self._particles,
            mutation=kernel,
            ess_fraction=self._ess_fraction,
            mutation_steps=self._mutation_steps,
        )
        return tempering(kernel.target, rng=rng, n_jobs=n_jobs)

    def push_forward(self, operator: LinearOperator, /) -> MeasureEstimator:
        """The estimator of the law of ``T m`` rather than of ``m``.

        Args:
            operator: the property map ``T``, from the model space.

        Returns:
            The measure estimator of the property.

        Raises:
            ValueError: if ``T`` does not act on the model space.
        """
        return _PushedMeasureEstimator(self, operator)

    def __repr__(self) -> str:
        return f"SMC({self._mutation!r}, particles={self._particles})"
