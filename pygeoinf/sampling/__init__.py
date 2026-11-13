"""
Function-space Markov chain Monte Carlo, as discretised dynamics.

Every sampler here is defined on a function space, and so behaves the same
however finely the model is discretised. Its reference is a Gaussian
equivalent to the prior; the part of its dynamics that the reference defines
is solved exactly and never split; and every acceptance ratio, weight and
preconditioner it computes has a limit under refinement. That is why the
literature's "infinity" prefix is not needed: nothing here is the
finite-dimensional version.

The package is an engine, as :mod:`pygeoinf.numerics` is for the
deterministic estimators: it samples a :class:`Target`, the residual
potential against a reference, and knows nothing of forward problems.
:class:`~pygeoinf.inference.monte_carlo.MonteCarloEstimator` builds the
target for a problem's data and runs a sampler on it::

    sampler = Chains(HMC(step=0.2, length=5), steps=2000, burn_in=500, rng=rng, n_jobs=4)
    posterior = MonteCarloEstimator(problem, prior, sampler=sampler)(data)

A sampler is built in four layers:

1. a **dynamics** (:mod:`.dynamics`): an SDE whose invariant measure is the
   posterior, which generates its exactly solvable flows (:mod:`.flows`)
   from its :class:`Target`;
2. a **splitting scheme** (:mod:`.schemes`): a word over those flows,
   applied with a time step ``h``;
3. a **correction** (:mod:`.corrections`): a Metropolized deterministic
   block, or Metropolis-Hastings on a proposal whose transition density is
   known; only a correction makes a scheme into a :class:`Kernel`, so every
   sampler is exact whatever its step;
4. a **population** (:mod:`.population`): independent :class:`Chains`, or
   :class:`SMC` along the tempering path from the prior, returning a
   :class:`SampledPosterior` or a :class:`WeightedEnsemble`.

The moves (:mod:`.moves`) assemble the first three for the common samplers,
:class:`PCN`, :class:`MALA`, :class:`HMC`, :class:`GeneralizedHMC`, and
:class:`Gibbs` over the summands of a direct sum, and a population runs one::

    Chains(PCN(step=0.1), steps=5000, burn_in=1000, chains=4, rng=rng)
    SMC(PCN(step=0.1), particles=256, rng=rng)
    Chains(
        Gibbs({"level_set": PCN(step=0.1), "fields": HMC(step=0.2, length=5)},
              repeat={"level_set": 5}),
        steps=2000, burn_in=500, rng=rng,
    )

A reference fitted to the posterior moves the chains at its scale, given to
the estimator as a measure or as an estimator fitted to the data at each
call; a local MALA moves about the Laplace Gaussian at every state::

    laplace = LaplaceApproximation.from_problem(problem, prior, rank=50)
    MonteCarloEstimator(problem, prior, sampler=sampler, reference=laplace)
    MonteCarloEstimator(
        problem, prior, sampler=Chains(MALA(step=1.0, local=True), steps=2000),
        reference=laplace,
    )

A posterior with several modes is sampled over a mixture fitted to a first
run, whose components the chains jump between::

    mixture = posterior.fitted_reference(components=2, rng=rng)
    MonteCarloEstimator(problem, prior, sampler=sampler, reference=mixture)

The same samplers by hand, on a target built for the data::

    target = Target(problem.potential(data), prior)
    pcn = MetropolisHastings(SplittingScheme(PreconditionedOU(target), "O", step=0.1))

    dynamics = Hamiltonian(target)
    scheme = SplittingScheme(dynamics, "OBAB", step=0.2)
    hmc = MetropolizedBlock(scheme.block("BAB", repeat=5)).within(scheme)

    posterior = Chains(hmc, steps=2000, burn_in=500, rng=rng, n_jobs=4)(target)
"""

from .corrections import (
    Kernel,
    MetropolisHastings,
    MetropolizedBlock,
    Step,
    StepInfo,
    Sweep,
)
from .dynamics import (
    Blockwise,
    Dynamics,
    Hamiltonian,
    Overdamped,
    PreconditionedOU,
    Underdamped,
)
from .flows import (
    Flow,
    GibbsBlock,
    GradientDrift,
    KernelFlow,
    LabelJump,
    LikelihoodKick,
    MomentumRefresh,
    PriorOU,
    PriorRotation,
    crank_nicolson_delta,
    pcn_beta,
    time_from_crank_nicolson,
    time_from_pcn_beta,
)
from .moves import HMC, MALA, PCN, GeneralizedHMC, Gibbs, Move
from .population import SMC, Chains, Sampler, StepSizeAdaptation
from .results import Chain, ChainDiagnostics, SampledPosterior, WeightedEnsemble
from .schemes import SplittingScheme
from .target import State, Target

__all__ = [
    "Blockwise",
    "Chain",
    "ChainDiagnostics",
    "Chains",
    "Dynamics",
    "Flow",
    "GeneralizedHMC",
    "Gibbs",
    "GibbsBlock",
    "GradientDrift",
    "HMC",
    "Hamiltonian",
    "Kernel",
    "KernelFlow",
    "LabelJump",
    "LikelihoodKick",
    "MALA",
    "MetropolisHastings",
    "MetropolizedBlock",
    "Move",
    "MomentumRefresh",
    "Overdamped",
    "PCN",
    "PreconditionedOU",
    "PriorOU",
    "PriorRotation",
    "SMC",
    "SampledPosterior",
    "Sampler",
    "SplittingScheme",
    "State",
    "Step",
    "StepInfo",
    "StepSizeAdaptation",
    "Sweep",
    "Target",
    "Underdamped",
    "WeightedEnsemble",
    "crank_nicolson_delta",
    "pcn_beta",
    "time_from_crank_nicolson",
    "time_from_pcn_beta",
]
