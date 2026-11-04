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

Over a :class:`~pygeoinf.inference.gaussian.MixtureReference`, fitted to a
posterior with several modes, every word starts with the label jump ``L``,
which moves a chain between the mixture's components, and so between modes.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np
from numpy.random import Generator

from ...algebra.direct_sum import DirectSum
from ...algebra.operators import LinearOperator, Operator
from ...algebra.spaces import HilbertSpace
from ...probability.gaussian import GaussianMeasure
from ..estimators import MeasureEstimator, _PushedMeasureEstimator
from ..gaussian import GaussianReference, MixtureReference
from ..laplace import LaplaceApproximation
from .corrections import Kernel, MetropolisHastings, MetropolizedBlock, Step, Sweep
from .dynamics import Blockwise, Hamiltonian, Overdamped, PreconditionedOU, Underdamped
from .population import Chains, StepSizeAdaptation, Tempering
from .results import SampledPosterior, WeightedEnsemble
from .schemes import SplittingScheme
from .target import Target

if TYPE_CHECKING:  # pragma: no cover
    from ...geometry.sets import Subset
    from ..problem import ForwardProblem

__all__ = [
    "BlockRecipe",
    "ChainSampler",
    "GeometricHMC",
    "GeometricMALA",
    "Gibbs",
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
            or :class:`~pygeoinf.inference.gaussian.MixtureReference` over the
            prior, which the dynamics solves exactly; the prior if omitted.
        acceptance: the acceptance probability the step is adapted towards
            during burn-in, or ``None`` to keep the step fixed.
        constraint: a subset of the model space outside which the
            posterior has no mass; a proposal outside it is rejected.

    Raises:
        TypeError: if the reference is not a ``GaussianReference`` or a
            ``MixtureReference``.
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
        reference: GaussianReference | MixtureReference | None = None,
        acceptance: float | None,
        constraint: "Subset | None" = None,
    ) -> None:
        if prior.domain != problem.model_space:
            raise ValueError("The prior is not defined on the model space.")
        if reference is None:
            reference = GaussianReference.from_prior(prior)
        elif not isinstance(reference, (GaussianReference, MixtureReference)):
            raise TypeError(
                "A reference is a GaussianReference or a MixtureReference, which "
                "the estimators and the sampled posteriors hand "
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
    def reference(self) -> GaussianReference | MixtureReference:
        """The reference the dynamics solves exactly."""
        return self._reference

    @property
    def step(self) -> float:
        """The time step ``h`` the chains start at."""
        return self._step

    @property
    def is_local(self) -> bool:
        """Whether the reference follows the state; see :class:`GeometricMALA`."""
        return False

    @property
    def adaptation(self) -> StepSizeAdaptation | tuple | None:
        """How the step adapts during burn-in, or ``None`` for a fixed step;
        one per block for :class:`Gibbs`."""
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

    @classmethod
    def _build(cls, target: Target, step: float, /, **options: Any) -> Kernel:
        """The preset's kernel on a target at a step, from its options alone:
        what a block recipe builds.

        Raises:
            NotImplementedError: for a preset that offers no recipe.
        """
        raise NotImplementedError(f"{cls.__name__} offers no block recipe.")

    @classmethod
    def _cap(cls, **options: Any) -> float:
        """The step beyond which the adaptation gains nothing.

        For an Ornstein-Uhlenbeck step, ten: ``exp(-10)`` of the old point
        is left, and the proposal is as good as independent.
        """
        return 10.0

    def _options(self) -> dict[str, Any]:
        """The preset's options beyond its step."""
        return {}

    def _longest_step(self) -> float:
        """The step beyond which the adaptation gains nothing."""
        return self._cap(**self._options())

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
        record: Operator | None = None,
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
            record: a small map from the model space to a space with
                coordinates, linear or not, whose components are recorded at
                every step.
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
        reference: GaussianReference | MixtureReference | None = None,
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

    @classmethod
    def recipe(
        cls, *, step: float, acceptance: float | None = 0.25, steps: int = 1
    ) -> "BlockRecipe":
        """pCN as one block of a :class:`Gibbs` sampler.

        Args:
            step: the block's time step.
            acceptance: the acceptance its step adapts towards during
                burn-in; ``None`` keeps it fixed.
            steps: the kernel's steps at each visit to the block.

        Returns:
            The recipe.
        """
        return BlockRecipe(cls, step=step, acceptance=acceptance, steps=steps)

    @classmethod
    def _build(cls, target: Target, step: float, /, **options: Any) -> Kernel:
        return MetropolisHastings(
            SplittingScheme(PreconditionedOU(target), _word(target, "O"), step=step)
        )

    def _kernel(self, target: Target, /) -> Kernel:
        return self._build(target, self._step)


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
        reference: GaussianReference | MixtureReference | None = None,
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

    @classmethod
    def recipe(
        cls, *, step: float, acceptance: float | None = 0.57, steps: int = 1
    ) -> "BlockRecipe":
        """MALA as one block of a :class:`Gibbs` sampler.

        Args:
            step: the block's time step.
            acceptance: the acceptance its step adapts towards during
                burn-in; ``None`` keeps it fixed.
            steps: the kernel's steps at each visit to the block.

        Returns:
            The recipe.
        """
        return BlockRecipe(cls, step=step, acceptance=acceptance, steps=steps)

    @classmethod
    def _build(cls, target: Target, step: float, /, **options: Any) -> Kernel:
        return MetropolisHastings(
            SplittingScheme(Overdamped(target), _word(target, "BO"), step=step)
        )

    def _kernel(self, target: Target, /) -> Kernel:
        return MALA._build(target, self._step)


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
        reference: GaussianReference | MixtureReference | None = None,
        acceptance: float | None = 0.65,
        constraint: "Subset | None" = None,
    ) -> None:
        if int(length) != length or length < 1:
            raise ValueError(
                f"A trajectory has at least one leapfrog step, got {length}."
            )
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

    @classmethod
    def recipe(
        cls,
        *,
        step: float,
        length: int,
        acceptance: float | None = 0.65,
        steps: int = 1,
    ) -> "BlockRecipe":
        """HMC as one block of a :class:`Gibbs` sampler.

        Args:
            step: the block's time step.
            length: the leapfrog steps per trajectory.
            acceptance: the acceptance its step adapts towards during
                burn-in; ``None`` keeps it fixed.
            steps: the kernel's trajectories at each visit to the block.

        Returns:
            The recipe.

        Raises:
            ValueError: for a length below one.
        """
        if int(length) != length or length < 1:
            raise ValueError(
                f"A trajectory has at least one leapfrog step, got {length}."
            )
        return BlockRecipe(
            cls, step=step, acceptance=acceptance, steps=steps, length=int(length)
        )

    @classmethod
    def _cap(cls, **options: Any) -> float:
        """A quarter turn over the trajectory, ``pi / (2 length)``."""
        return float(np.pi / (2 * options["length"]))

    def _options(self) -> dict[str, Any]:
        return {"length": self._length}

    @classmethod
    def _build(cls, target: Target, step: float, /, **options: Any) -> Kernel:
        scheme = SplittingScheme(Hamiltonian(target), _word(target, "OBAB"), step=step)
        return MetropolizedBlock(scheme.block("BAB", repeat=options["length"])).within(
            scheme
        )

    def _kernel(self, target: Target, /) -> Kernel:
        return HMC._build(target, self._step, **self._options())

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
        reference: GaussianReference | MixtureReference | None = None,
        acceptance: float | None = 0.8,
        constraint: "Subset | None" = None,
    ) -> None:
        if not 0.0 < friction < float("inf"):
            raise ValueError(f"A friction is positive and finite, got {friction}.")
        if int(length) != length or length < 1:
            raise ValueError(
                f"A trajectory has at least one leapfrog step, got {length}."
            )
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

    @classmethod
    def recipe(
        cls,
        *,
        step: float,
        friction: float = 1.0,
        length: int = 1,
        acceptance: float | None = 0.8,
        steps: int = 1,
    ) -> "BlockRecipe":
        """Partially refreshed HMC as one block of a :class:`Gibbs` sampler;
        the velocity is drawn afresh at each visit to the block.

        Args:
            step: the block's time step.
            friction: ``gamma``, positive and finite.
            length: the leapfrog steps per trajectory.
            acceptance: the acceptance its step adapts towards during
                burn-in; ``None`` keeps it fixed.
            steps: the kernel's trajectories at each visit to the block.

        Returns:
            The recipe.

        Raises:
            ValueError: for a friction that is not positive and finite, or a
                length below one.
        """
        if not 0.0 < friction < float("inf"):
            raise ValueError(f"A friction is positive and finite, got {friction}.")
        if int(length) != length or length < 1:
            raise ValueError(
                f"A trajectory has at least one leapfrog step, got {length}."
            )
        return BlockRecipe(
            cls,
            step=step,
            acceptance=acceptance,
            steps=steps,
            friction=float(friction),
            length=int(length),
        )

    @classmethod
    def _cap(cls, **options: Any) -> float:
        """A quarter turn over the trajectory, ``pi / (2 length)``."""
        return float(np.pi / (2 * options["length"]))

    def _options(self) -> dict[str, Any]:
        return {"friction": self._friction, "length": self._length}

    @classmethod
    def _build(cls, target: Target, step: float, /, **options: Any) -> Kernel:
        dynamics = Underdamped(target, friction=options["friction"])
        scheme = SplittingScheme(dynamics, _word(target, "OBAB"), step=step)
        return MetropolizedBlock(scheme.block("BAB", repeat=options["length"])).within(
            scheme
        )

    def _kernel(self, target: Target, /) -> Kernel:
        return PreconditionedHMC._build(target, self._step, **self._options())

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(step={self._step:g}, friction={self._friction:g}, "
            f"length={self._length})"
        )


class GeometricMALA(MALA):
    """MALA about a Gaussian fitted to the posterior, fixed or following the
    state.

    Given a reference, :class:`MALA` with that reference: its prior is the
    reference's. The reference is built from the data, by an estimator's
    ``reference`` method or fitted to a run, and the sampler is meant to be
    called on the same data.

    Given a :class:`~pygeoinf.inference.laplace.LaplaceApproximation`, the
    local MALA: at every state the reference is the Laplace Gaussian there,
    ``K == (C^-1 + H(u))^-1`` with ``H`` the Gauss-Newton or a supplied
    Hessian, centred on the Newton step from ``u``. That Gaussian is the
    posterior's quadratic model at ``u``, so the drift vanishes and the
    proposal is its Ornstein-Uhlenbeck step,
    ``v == m_u + cos(a) (u - m_u) + sin(a) xi`` with ``xi ~ N(0, K_u)``; the
    acceptance compares the pair laws of the references at ``u`` and ``v``,
    an ``r``-dimensional ratio, so the sampler is defined on a function
    space. Each step builds the reference at the proposal: with the full
    data chart, one adjoint and one prior-covariance application per datum.
    The proposal follows the posterior's local shape, which pays where that
    shape changes across the posterior.

    Args:
        problem: the forward problem, with a Gaussian error measure and a
            differentiable forward operator.
        reference: the reference, whose prior is the sampler's; or a Laplace
            approximation on *problem*, for the local MALA.
        step: the time step ``h``.
        acceptance: the acceptance to adapt towards during burn-in, 0.57 by
            default; ``None`` keeps the step fixed.
        constraint: where the posterior has its mass.

    Raises:
        TypeError: if the reference is not a ``GaussianReference``, a
            ``MixtureReference`` or a ``LaplaceApproximation``.
        ValueError: if a Laplace approximation is of another problem, or as
            for :class:`ChainSampler`.
    """

    @classmethod
    def recipe(cls, **options: Any) -> "BlockRecipe":
        """Not offered: a block's reference is the prior's conditional law.

        Args:
            **options: as for the plain preset.

        Raises:
            TypeError: always.
        """
        raise TypeError(
            f"{cls.__name__} takes its reference from the caller; a Gibbs block's "
            "reference is the prior's conditional, so use the plain preset's recipe."
        )

    def __init__(
        self,
        problem: "ForwardProblem",
        reference: GaussianReference | MixtureReference | LaplaceApproximation,
        /,
        *,
        step: float,
        acceptance: float | None = 0.57,
        constraint: "Subset | None" = None,
    ) -> None:
        self._local = None
        if isinstance(reference, LaplaceApproximation):
            if reference.estimator.forward_problem is not problem:
                raise ValueError(
                    "The Laplace approximation must be built on the sampler's "
                    "forward problem."
                )
            self._local = reference
            reference = GaussianReference.from_prior(reference.estimator.prior)
        super().__init__(
            problem,
            _prior_of(reference),
            step=step,
            reference=reference,
            acceptance=acceptance,
            constraint=constraint,
        )

    @property
    def is_local(self) -> bool:
        """Whether the reference follows the state."""
        return self._local is not None

    def target(self, data: Any, /) -> Target:
        """The residual potential for the data: against the fixed reference,
        or against the Laplace reference at each state.

        Args:
            data: the observations.

        Returns:
            The target.
        """
        if self._local is None:
            return super().target(data)
        return Target(self._problem, data, self._local, constraint=self._constraint)

    def _kernel(self, target: Target, /) -> Kernel:
        if target.is_local:
            return MetropolisHastings(
                SplittingScheme(PreconditionedOU(target), "O", step=self._step)
            )
        return super()._kernel(target)


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
        TypeError: if the reference is not a ``GaussianReference`` or a
            ``MixtureReference``.
        ValueError: for a length below one, or as for
            :class:`ChainSampler`.
    """

    @classmethod
    def recipe(cls, **options: Any) -> "BlockRecipe":
        """Not offered: a block's reference is the prior's conditional law.

        Args:
            **options: as for the plain preset.

        Raises:
            TypeError: always.
        """
        raise TypeError(
            f"{cls.__name__} takes its reference from the caller; a Gibbs block's "
            "reference is the prior's conditional, so use the plain preset's recipe."
        )

    def __init__(
        self,
        problem: "ForwardProblem",
        reference: GaussianReference | MixtureReference,
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


class BlockRecipe:
    """What samples one block of a :class:`Gibbs` sampler.

    A preset's kernel without its problem and prior, which a block takes
    from the whole: built by the presets' ``recipe`` methods,
    ``PCN.recipe(step=0.1)``, ``HMC.recipe(step=0.2, length=5)`` and the
    like, and turned into a kernel on each visit, on the block's conditional
    target.

    Args:
        sampler: the preset class whose kernel this is.
        step: the block's time step, where the burn-in's adaptation starts.
        acceptance: the acceptance its step adapts towards during burn-in,
            or ``None`` to keep it fixed.
        steps: the kernel's steps at each visit to the block.
        **options: the preset's options beyond its step, a trajectory's
            length say.

    Raises:
        ValueError: for a step that is not positive and finite, an
            acceptance outside ``(0, 1)``, or a step count below one.
    """

    def __init__(
        self,
        sampler: type,
        /,
        *,
        step: float,
        acceptance: float | None,
        steps: int = 1,
        **options: Any,
    ) -> None:
        if not 0.0 < step < float("inf"):
            raise ValueError(f"A step is positive and finite, got {step}.")
        if acceptance is not None and not 0.0 < acceptance < 1.0:
            raise ValueError(f"A target acceptance lies in (0, 1), got {acceptance}.")
        if int(steps) != steps or steps < 1:
            raise ValueError(f"A block takes at least one step per visit, got {steps}.")
        self._sampler = sampler
        self._step = float(step)
        self._acceptance = acceptance
        self._steps = int(steps)
        self._options = dict(options)

    @property
    def sampler(self) -> type:
        """The preset class whose kernel this is."""
        return self._sampler

    @property
    def step(self) -> float:
        """The block's starting time step."""
        return self._step

    @property
    def acceptance(self) -> float | None:
        """The acceptance the step adapts towards, or ``None``."""
        return self._acceptance

    @property
    def steps(self) -> int:
        """The kernel's steps at each visit."""
        return self._steps

    @property
    def uses_gradient(self) -> bool:
        """Whether the kernel differentiates the forward operator."""
        return bool(self._sampler.uses_gradient)

    @property
    def longest_step(self) -> float:
        """The step beyond which the adaptation gains nothing."""
        return self._sampler._cap(**self._options)

    def kernel(self, target: Target, step: float, /) -> Kernel:
        """The kernel on a block's target.

        Args:
            target: the block's conditional target.
            step: the time step.

        Returns:
            The kernel.
        """
        return self._sampler._build(target, step, **self._options)

    def adaptation(self) -> StepSizeAdaptation | None:
        """How the block's step adapts during burn-in, or ``None``.

        Returns:
            The adaptation, capped at :attr:`longest_step`.
        """
        if self._acceptance is None:
            return None
        return StepSizeAdaptation(
            acceptance=self._acceptance, maximum=self.longest_step
        )

    def __repr__(self) -> str:
        options = "".join(f", {key}={value!r}" for key, value in self._options.items())
        return f"{self._sampler.__name__}.recipe(step={self._step:g}{options})"


class Gibbs(ChainSampler):
    """Gibbs sampling over the summands of a direct sum: each field its own
    sampler, against its conditional.

    For a model of several physical fields with different correlation
    structures, a level set beside elastic parameters say: each summand of
    the model space is a block, updated with the others held by its own
    recipe's kernel on its conditional target, whose reference is the
    prior's conditional law of that summand
    (:meth:`~pygeoinf.probability.gaussian.GaussianMeasure.conditional`), so
    every block is a function-space sampler and the sweep is exact. A
    systematic scan visits the blocks in summand order; a symmetric one
    there and back, which is reversible. Each block's step adapts towards
    its own acceptance during burn-in, and the run reports each block's
    acceptance (``block_acceptance_rates``).

    Args:
        problem: the forward problem, on a direct sum.
        prior: the Gaussian prior on it, independent or correlated across
            the summands.
        blocks: each summand's recipe, by position or label, covering every
            summand once.
        order: ``"systematic"`` or ``"symmetric"``.
        constraint: where the posterior has its mass.

    Raises:
        TypeError: if a block is not a :class:`BlockRecipe`.
        ValueError: if the model space is not a direct sum, a summand has no
            block or two, the order is unknown, or as for
            :class:`ChainSampler`.
    """

    def __init__(
        self,
        problem: "ForwardProblem",
        prior: GaussianMeasure,
        /,
        *,
        blocks: Mapping[int | str, BlockRecipe],
        order: str = "systematic",
        constraint: "Subset | None" = None,
    ) -> None:
        space = problem.model_space
        if not isinstance(space, DirectSum):
            raise ValueError(
                f"Gibbs blocks are the summands of a direct sum; the model space is {space!r}."
            )
        if order not in ("systematic", "symmetric"):
            raise ValueError(f"A scan is 'systematic' or 'symmetric', not {order!r}.")
        indexed: dict[int, BlockRecipe] = {}
        for key, recipe in dict(blocks).items():
            if not isinstance(recipe, BlockRecipe):
                raise TypeError(
                    "A block is a recipe, PCN.recipe(step=...) say; got "
                    f"{type(recipe).__name__}."
                )
            index = space.index(key)
            if index in indexed:
                raise ValueError(f"Summand {key!r} has two blocks.")
            indexed[index] = recipe
        missing = [i for i in range(len(space.subspaces)) if i not in indexed]
        if missing:
            raise ValueError(
                f"Every summand needs a block, or it never moves; {missing} have none."
            )
        self._recipes = tuple(indexed[i] for i in range(len(space.subspaces)))
        self._order = order
        self.uses_gradient = any(recipe.uses_gradient for recipe in self._recipes)
        super().__init__(
            problem,
            prior,
            step=self._recipes[0].step,
            acceptance=None,
            constraint=constraint,
        )
        self._adaptation = tuple(recipe.adaptation() for recipe in self._recipes)

    @property
    def recipes(self) -> tuple[BlockRecipe, ...]:
        """The blocks' recipes, in summand order."""
        return self._recipes

    @property
    def steps(self) -> tuple[float, ...]:
        """The blocks' starting steps, in summand order."""
        return tuple(recipe.step for recipe in self._recipes)

    @property
    def order(self) -> str:
        """The scan."""
        return self._order

    def _kernel(self, target: Target, /) -> Kernel:
        dynamics = Blockwise(target, dict(enumerate(self._recipes)))
        return Sweep(
            SplittingScheme(dynamics, dynamics.scan(order=self._order), step=1.0)
        )

    def __repr__(self) -> str:
        blocks = ", ".join(repr(recipe) for recipe in self._recipes)
        return f"Gibbs([{blocks}], order={self._order!r})"


def _word(target: Target, word: str, /) -> str:
    """The preset's word, led by the label jump over a mixture reference."""
    return "L" + word if target.is_mixture else word


def _prior_of(reference: GaussianReference | MixtureReference, /) -> GaussianMeasure:
    """The prior a reference is built over.

    Raises:
        TypeError: if the reference is not a ``GaussianReference`` or a
            ``MixtureReference``.
    """
    if not isinstance(reference, (GaussianReference, MixtureReference)):
        raise TypeError(
            "A geometric sampler takes a GaussianReference or a MixtureReference, "
            f"which the estimators hand out; got {type(reference).__name__}."
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
        ValueError: if the preset's reference is not its prior, or follows
            the state.
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
        if mutation.reference.rank != 0 or mutation.is_local:
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
