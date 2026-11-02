"""
Function-space Markov chain Monte Carlo, as discretised dynamics.

Every sampler here is defined on a function space, and so behaves the same
however finely the model is discretised. Its reference is a Gaussian
equivalent to the prior; the part of its dynamics that the reference defines
is solved exactly and never split; and every acceptance ratio, weight and
preconditioner it computes has a limit under refinement. That is why the
literature's "infinity" prefix is not needed: nothing here is the
finite-dimensional version.

A sampler is built in four layers:

1. a **dynamics** (:mod:`.dynamics`): an SDE whose invariant measure is the
   posterior, which generates its exactly solvable flows (:mod:`.flows`)
   from its :class:`Target`, the residual potential against a reference;
2. a **splitting scheme** (:mod:`.schemes`): a word over those flows,
   applied with a time step ``h``;
3. a **correction** (:mod:`.corrections`): a Metropolized deterministic
   block, or Metropolis-Hastings on a proposal whose transition density is
   known; only a correction makes a scheme into a :class:`Kernel`, so every
   sampler is exact whatever its step;
4. a **population** (:mod:`.population`): independent :class:`Chains`, or
   :class:`Tempering` along the path from the prior, returning a
   :class:`SampledPosterior` or a :class:`WeightedEnsemble`.

The presets (:mod:`.presets`) assemble the common samplers and are measure
estimators, called on data::

    posterior = HMC(problem, prior, step=0.2, length=5)(
        data, steps=2000, burn_in=500, rng=rng, n_jobs=4
    )
    laplace = LaplaceApproximation.from_problem(problem, prior, gauss_newton=True)
    reference = laplace.reference(mode, data)
    refined = GeometricHMC(problem, reference, step=0.4, length=4)(data, steps=2000, rng=rng)
    ensemble = SMC(PCN(problem, prior, step=0.1), particles=256)(data, rng=rng)

The same samplers by hand, pCN and HMC::

    target = Target(problem, data, prior)
    pcn = MetropolisHastings(SplittingScheme(PreconditionedOU(target), "O", step=0.1))

    dynamics = Hamiltonian(target)
    scheme = SplittingScheme(dynamics, "OBAB", step=0.2)
    hmc = MetropolizedBlock(scheme.block("BAB", repeat=5)).within(scheme)

    posterior = Chains(4)(hmc, start=prior, steps=2000, burn_in=500, rng=rng, n_jobs=4)
"""

from .corrections import Kernel, MetropolisHastings, MetropolizedBlock, Step, StepInfo
from .dynamics import Dynamics, Hamiltonian, Overdamped, PreconditionedOU, Underdamped
from .flows import (
    Flow,
    GradientDrift,
    LikelihoodKick,
    MomentumRefresh,
    PriorOU,
    PriorRotation,
)
from .population import Chains, StepSizeAdaptation, Tempering
from .presets import (
    HMC,
    MALA,
    PCN,
    SMC,
    ChainSampler,
    GeometricHMC,
    GeometricMALA,
    PreconditionedHMC,
)
from .results import Chain, ChainDiagnostics, SampledPosterior, WeightedEnsemble
from .schemes import SplittingScheme
from .target import State, Target

__all__ = [
    "Chain",
    "ChainDiagnostics",
    "ChainSampler",
    "Chains",
    "Dynamics",
    "Flow",
    "GeometricHMC",
    "GeometricMALA",
    "GradientDrift",
    "HMC",
    "Hamiltonian",
    "Kernel",
    "LikelihoodKick",
    "MALA",
    "MetropolisHastings",
    "MetropolizedBlock",
    "MomentumRefresh",
    "Overdamped",
    "PCN",
    "PreconditionedHMC",
    "PreconditionedOU",
    "PriorOU",
    "PriorRotation",
    "SMC",
    "SampledPosterior",
    "SplittingScheme",
    "State",
    "Step",
    "StepInfo",
    "StepSizeAdaptation",
    "Target",
    "Tempering",
    "Underdamped",
    "WeightedEnsemble",
]
