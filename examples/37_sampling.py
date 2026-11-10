"""
37. Sampling: the presets on a posterior with two modes.

This example highlights the essential differences between the sampling
presets on a simple nonlinear problem whose posterior is known: the quadratic
scalar problem, ``d == m^2 + eta``, with ``eta`` Gaussian noise and a centered
Gaussian prior. The sign of ``m`` is invisible to the data, so the posterior
is a double well, symmetric with a mode at each of ``m == +-sqrt(d)``, more
or less, and a barrier between them at ``m == 0``. In one dimension it can be
computed by quadrature, so every sampler can be held to it.

Every preset is exact: run long enough, each gives the right answer. What
differs is how long "long enough" is, and here that means one question above
all, how often a chain crosses the barrier. Four chains are run for each, and
the diagnostics are what tell a reader whether to believe them.

The presets, each a measure estimator built on the problem and the prior and
called on the data:

* ``PCN`` proposes the current point rotated a little towards a fresh draw
  from the prior; no gradient.
* ``MALA`` drifts down the gradient before the same step.
* ``HMC`` runs a trajectory of leapfrog steps, refreshing the velocity in full
  between them.
* ``GeneralizedHMC`` refreshes the velocity only in part, so a chain keeps
  its direction.
* ``GeometricHMC`` is HMC about a Gaussian fitted to the posterior, here the
  Laplace approximation at one of the modes.
* ``GeometricMALA``, given the Laplace approximation itself rather than one
  of its Gaussians, is the local MALA: at every state it proposes from the
  Laplace Gaussian there, so its proposal follows the posterior's shape.
* ``SMC`` moves a population from the prior to the posterior through a
  sequence of tempered measures, and estimates the evidence on the way.

Then the fix for a posterior with several modes: a mixture of Gaussians
fitted to the pCN run, one per mode, as the reference of MALA and HMC, which
then jump between the modes; and fresh draws, short chains started from it.

Needs matplotlib.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import (
    HMC,
    MALA,
    PCN,
    SMC,
    EuclideanSpace,
    ForwardProblem,
    GaussianMeasure,
    GeneralizedHMC,
    GeometricHMC,
    GeometricMALA,
    LaplaceApproximation,
    LinearOperator,
    Operator,
    plotting,
)
from pygeoinf.numerics.chain_diagnostics import integrated_autocorrelation_time

rng = np.random.default_rng(37)

# ------------------------------------------------------------------ #
# The quadratic scalar problem                                       #
# ------------------------------------------------------------------ #

X = EuclideanSpace(1)
D = EuclideanSpace(1)


def value(m: np.ndarray) -> np.ndarray:
    """The prediction, ``m^2``."""
    return m**2


def derivative(m: np.ndarray) -> LinearOperator:
    """Its derivative, ``2 m``: zero at the barrier, where the data say nothing."""
    return LinearOperator.from_matrix(X, D, np.array([[2.0 * m[0]]]), form="components")


forward = Operator.from_callables(X, D, value, derivative=derivative)
prior = GaussianMeasure.from_standard_deviation(X, 1.0)
error = GaussianMeasure.from_standard_deviation(D, 0.3)
problem = ForwardProblem(forward, error=error)

# One observation. Any datum above sigma^2 / 2 gives two modes; this one puts
# them near +-1 with a barrier of about five nats between them, high enough to
# separate the samplers and low enough for some of them to cross it.
data = np.array([1.0])

# ------------------------------------------------------------------ #
# The posterior, by quadrature                                       #
# ------------------------------------------------------------------ #
#
# The posterior's density against the prior is exp(-Phi) / Z, with Phi the
# problem's own likelihood potential, so the evidence computed here is in the
# convention SMC reports.

potential = problem.potential(data)
grid = np.linspace(-2.0, 2.0, 4001)
spacing = grid[1] - grid[0]
phi = np.array([potential(np.array([m])) for m in grid])
prior_density = np.exp(-0.5 * grid**2) / np.sqrt(2.0 * np.pi)
unnormalized = np.exp(-phi) * prior_density
evidence = unnormalized.sum() * spacing
density = unnormalized / evidence
second_moment = float((grid**2 * density).sum() * spacing)
mean_potential = float((phi * density).sum() * spacing)

print(
    f"exact posterior: mean 0 by symmetry, E[m^2] = {second_moment:.4f}, "
    f"log Z = {np.log(evidence):.4f}"
)
# The barrier is the posterior's: its negative log density against the
# Lebesgue measure, Phi plus the prior's m^2 / 2, at m == 0 above its minimum.
negative_log_density = phi + 0.5 * grid**2
barrier = negative_log_density[np.argmin(np.abs(grid))] - negative_log_density.min()
print(f"  the posterior's barrier at m = 0 is {barrier:.1f} nats above its modes")
print(f"  the potential averages {mean_potential:.3f} under the posterior")
print()

# ------------------------------------------------------------------ #
# The chains                                                         #
# ------------------------------------------------------------------ #
#
# The Laplace approximation needs a mode: start the search on the positive
# side, and it finds that one. The reference it gives is a Gaussian on that
# mode alone, so GeometricHMC proposes well within it and is no help with the
# other. Its chain is exact all the same: a reference makes a sampler fast or
# slow, never wrong.
#
# The local MALA is given the estimator instead, and builds the Laplace
# Gaussian afresh at every state: at m the curvature is that of the data's
# 2m, so anywhere near a mode the proposal is narrow and centered on that mode.
# It would widen to the prior near the barrier, where the data say nothing
# about m, but a chain proposing so well within its mode never goes there.
#
# Its step is fixed rather than adapted. The proposal is a step of the Laplace
# Gaussian's own Ornstein-Uhlenbeck process, and where that Gaussian fits the
# posterior a long step is accepted nearly always: left to adapt, the step
# would climb to its cap and the run warn that its proposals are nearly
# independent of the state, which a reference too narrow in its tails would
# make dangerous. Here that is the point, so the step is set long, h == 3,
# where exp(-3) of the old point is left.

laplace = LaplaceApproximation.from_problem(problem, prior, gauss_newton=True)
mode = laplace.solve(data, start=np.array([0.5])).model
reference = laplace.reference(mode, data)

samplers = {
    "PCN": PCN(problem, prior, step=0.1),
    "MALA": MALA(problem, prior, step=0.05),
    "HMC": HMC(problem, prior, step=0.1, length=5),
    "GeneralizedHMC": GeneralizedHMC(problem, prior, step=0.1, friction=0.5),
    "GeometricHMC": GeometricHMC(problem, reference, step=0.3, length=3),
    "MALA, local": GeometricMALA(problem, laplace, step=3.0, acceptance=None),
}

# Record m itself at every step: a linear map to a small coordinate space, here
# the identity. On a function space this would be a few averages of the field.
record = LinearOperator.from_matrix(X, EuclideanSpace(1), np.eye(1), form="components")

# The run keeps only the steps after burn-in. To watch the chains leave their
# prior draws, the callback, called after every step burn-in included, keeps
# each chain's position and potential. With no workers asked for, it runs
# here, so it can simply write into an array.
chains, burn_in, steps = 4, 300, 1000


def watcher(positions: np.ndarray, potentials: np.ndarray):
    """A callback storing every step's position and potential."""

    def callback(chain: int, iteration: int, step) -> None:
        position = step.state.position
        positions[chain, iteration] = position[0]
        potentials[chain, iteration] = potential(position)

    return callback


results = {}
traces = {}
potential_traces = {}
posteriors = {}


def run(name: str, sampler) -> None:
    """Run a sampler's chains, keep what the figures need, and print its row."""
    traces[name] = np.empty((chains, burn_in + steps))
    potential_traces[name] = np.empty((chains, burn_in + steps))
    started = time.perf_counter()
    posterior = sampler(
        data,
        steps=steps,
        chains=chains,
        burn_in=burn_in,
        start=prior,
        record=record,
        record_names=["m"],
        rng=rng,
        callback=watcher(traces[name], potential_traces[name]),
    )
    elapsed = time.perf_counter() - started
    posteriors[name] = posterior
    m = posterior.scalars()[1]
    results[name] = m.ravel()
    # The error of E[m^2] needs the autocorrelation time of m^2 itself, which
    # does not see the sign: a chain stuck in one well has a long time for m
    # and a short one for m^2.
    squared = m**2
    error_of_mean = squared.std() * np.sqrt(
        integrated_autocorrelation_time(squared) / squared.size
    )
    crossings = int(np.sum(np.diff(np.sign(m), axis=1) != 0))
    rhat = posterior.diagnostics().rhat[1]
    step_size = np.mean([chain.step_size for chain in posterior.chains])
    print(
        f"{name:18s} {posterior.acceptance_rate:6.2f} {step_size:6.3f} "
        f"{squared.mean():7.4f}+-{error_of_mean:.4f} {np.mean(m > 0):6.2f} "
        f"{crossings:9d} {rhat:8.2f} {elapsed:7.1f}"
    )


header = (
    f"{'sampler':18s} {'accept':>6s} {'step':>6s} {'E[m^2]':>14s} {'m > 0':>6s} "
    f"{'crossings':>9s} {'R-hat(m)':>8s} {'seconds':>7s}"
)
print(header)
for name, sampler in samplers.items():
    run(name, sampler)

print()
print("E[m^2] does not see the sign, so every sampler gets it, within its own")
print("  error, from either mode. The fraction with m > 0 is the hard question:")
print("  it should be one half, and a chain that never crosses the barrier")
print("  answers it with the fraction of chains that started on each side.")
print("  R-hat of m is how a run reports that: well above one, the chains")
print("  disagree, and they are not to be believed on anything sign-dependent.")
print()
print("PCN crosses most, and that is no accident. Its proposal is a rotation")
print("  towards a fresh prior draw, so it can land in the other well in one")
print("  move, and with a prior as wide as the posterior a long step is still")
print("  accepted. MALA follows the gradient, which points into the well it is")
print("  in, and takes the short steps its drift allows. HMC's trajectories are")
print("  long enough to carry some chains over, and its partially refreshed")
print("  version keeps going the way it was going, which here means staying put.")
print("  GeometricHMC proposes about the positive mode alone. A gradient helps")
print("  a sampler move within a mode, quickly in many dimensions; it is no")
print("  help in getting from one mode to another.")
print()
print("The local MALA is the extreme of that: its proposal is the Laplace")
print("  Gaussian where it stands, accepted nearly always at a step long enough")
print("  to be all but independent, so within a mode it is the best sampler")
print("  here. It proposes about the mode it is in, and never crosses.")
print()

# ------------------------------------------------------------------ #
# A mixture fitted to the run                                        #
# ------------------------------------------------------------------ #
#
# The pCN run has seen both wells, so its record of m is a sample of the
# posterior to fit a reference to: a mixture of two Gaussians in the chart m,
# one per well, the prior everywhere else (here, nowhere else). Over a mixture
# reference a chain follows one component at a time and jumps between them
# by a Metropolized move that carries the state from one Gaussian onto the
# other exactly; where the mixture fits, the jump is accepted. The gradient
# samplers that could not cross the barrier now cross it at nearly every
# step. As for any reference, the fit decides how fast the chains mix, never
# what they sample.

mixture = posteriors["PCN"].fitted_reference(components=2, rng=rng)
print(
    "a mixture fitted to the pCN run: weights "
    f"{np.round(mixture.weights, 3)}, means "
    f"{np.round([c.expectation[0] for c in mixture.components], 3)}"
)
print(header)
run("MALA, mixture", GeometricMALA(problem, mixture, step=3.0, acceptance=None))
run("HMC, mixture", GeometricHMC(problem, mixture, step=0.3, length=3))

# And fresh draws: each a short chain of the pCN kernel started from the
# mixture, independent of the others and of the stored states.
draws = posteriors["PCN"].with_draw_reference(mixture).samples(200, rng=rng)
draws = np.array([x[0] for x in draws])
print(
    f"200 fresh draws from the pCN run, started from the mixture: E[m^2] = "
    f"{np.mean(draws**2):.4f}, m > 0 for {np.mean(draws > 0):.2f} of them"
)
print()

# ------------------------------------------------------------------ #
# Tempering                                                          #
# ------------------------------------------------------------------ #
#
# SMC starts from prior draws, which lie on both sides, and moves them to the
# posterior gradually, so the population arrives already split between the
# modes; the weights then settle how much each mode deserves. It also returns
# the evidence, which no chain can.
#
# Each stage raises the inverse temperature beta as far as keeps the
# effective sample size at a set fraction of the particles. One datum moves
# the prior only a little, so at the default half the path takes two stages;
# asking for nine tenths takes smaller steps, more of them, each cheaper to
# recover from, and draws a path worth looking at.

started = time.perf_counter()
ensemble = SMC(
    PCN(problem, prior, step=0.05), particles=500, mutation_steps=10, ess_fraction=0.9
)(data, rng=rng)
elapsed = time.perf_counter() - started
particles = np.array([x[0] for x in ensemble.particles])
weights = ensemble.weights
print(
    f"SMC: {ensemble.temperatures.size} stages, {elapsed:.1f} s, at beta = "
    f"{', '.join(f'{beta:.2f}' for beta in ensemble.temperatures)}"
)
print(
    f"  E[m^2] = {weights @ particles**2:.4f}, "
    f"m > 0 for {weights @ (particles > 0):.2f} of the weight"
)
print(f"  log Z = {ensemble.log_normalizer.value:.4f} against {np.log(evidence):.4f}")
print()

# ------------------------------------------------------------------ #
# The figures                                                        #
# ------------------------------------------------------------------ #
#
# First the answers: each sampler's kept draws against the exact posterior,
# with the prior they started from. In the SMC panel the path between them is
# drawn too: the tempered density, proportional to exp(-beta Phi) times the
# prior, at each beta the stages stopped at, fading from the prior into the
# posterior as beta rises to one.


def tempered(beta: float) -> np.ndarray:
    """The density at inverse temperature beta, normalized by quadrature."""
    values = np.exp(-beta * phi) * prior_density
    return values / (values.sum() * spacing)


path = plt.get_cmap("viridis")
panels = list(results.items()) + [("SMC", None)]
figure, axes = plt.subplots(3, 3, figsize=(13, 8.5), sharex=True, sharey=True)
for axis, (name, values) in zip(axes.ravel(), panels):
    if values is None:
        axis.hist(
            particles, bins=60, range=(-2, 2), weights=weights, density=True, alpha=0.5
        )
        for beta in ensemble.temperatures[:-1]:
            axis.plot(
                grid,
                tempered(beta),
                color=path(beta),
                linewidth=0.9,
                label=f"beta = {beta:.2f}",
            )
    else:
        axis.hist(values, bins=60, range=(-2, 2), density=True, alpha=0.5)
    axis.plot(
        grid, prior_density, color="0.5", linestyle="--", linewidth=1, label="prior"
    )
    axis.plot(grid, density, color="black", linewidth=1, label="posterior")
    axis.set_title(name)
axes[-1, -1].legend(fontsize=7, loc="upper center")
for axis in axes[-1]:
    axis.set_xlabel("m")
figure.suptitle("each sampler's draws against the exact posterior, and the prior")
figure.tight_layout()

# Then the paths: every chain from its prior draw, burn-in shaded. A chain
# that keeps to one side of the barrier is the R-hat of the table, drawn.

modes = grid[np.argmax(density * (grid > 0))]
figure, axes = plt.subplots(
    len(traces), 1, figsize=(10, 1.6 * len(traces)), sharex=True, sharey=True
)
for axis, (name, trace) in zip(axes, traces.items()):
    axis.axvspan(0, burn_in, color="0.92")
    for chain in trace:
        axis.plot(chain, linewidth=0.6)
    for level in (-modes, modes):
        axis.axhline(level, color="black", linestyle="--", linewidth=0.7)
    axis.set_ylabel("m", rotation=0)
    axis.set_title(name, loc="left", fontsize=10)
axes[-1].set_xlabel("step (burn-in shaded)")
figure.suptitle("each chain's path from its prior draw; the dashed lines are the modes")
figure.tight_layout()

# And the potential: Phi against its average under the posterior, on a log
# axis of steps so the first few can be seen. The prior draws start high; how
# fast each sampler comes down to the line is how fast it forgets where it
# started. Phi fluctuates from step to step even at equilibrium, so it is
# averaged over the chains and over bins of steps that widen as the axis
# compresses: the first bins are single steps, the last hundreds. Settling on
# the line says a chain has reached a mode, not that the modes are balanced:
# Phi does not see the sign of m, so it is blind to what the paths show.

edges = np.unique(np.geomspace(1, burn_in + steps + 1, 40).astype(int))
centers = np.sqrt(edges[:-1] * edges[1:])
figure, axis = plt.subplots(figsize=(8, 4.5))
for name, values in potential_traces.items():
    binned = [values[:, a - 1 : b - 1].mean() for a, b in zip(edges[:-1], edges[1:])]
    axis.plot(centers, binned, marker="o", markersize=3, linewidth=1, label=name)
axis.axhline(
    mean_potential,
    color="black",
    linestyle="--",
    linewidth=1,
    label="posterior average",
)
axis.axvline(burn_in, color="0.6", linewidth=0.8)
axis.set_xscale("log")
axis.set_yscale("log")
axis.set_xlabel("step (burn-in ends at the gray line)")
axis.set_ylabel("Phi, averaged over the chains")
axis.set_title(
    "convergence: the likelihood potential from the prior draws to the posterior"
)
axis.legend(fontsize=8)
figure.tight_layout()

print("three figures drawn")

plotting.show()
