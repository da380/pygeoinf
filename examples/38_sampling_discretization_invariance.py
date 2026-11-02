"""
38. Sampling: the acceptance rate does not depend on the mesh.

A field on a circle is represented on a grid of a few dozen points or a few
thousand, and its posterior is the same measure either way, the finer grid
only resolving it better. A sampler should behave the same way: with the same
step, the same acceptance rate at every resolution. Most finite-dimensional
samplers do not. A random walk in the field's grid values proposes a small
step in every one of its unknowns, and the more unknowns there are, the less
likely a proposal is to be accepted; refined far enough, it accepts nothing.
The reason is Feldman-Hajek: two Gaussians on a function space are either
equivalent or singular, and a proposal that perturbs every mode by a
comparable amount is singular to the prior in the limit.

Every sampler in this library is built to have a limit. Its reference is the
prior (or a Gaussian equivalent to it), the part of its dynamics the prior
defines is solved exactly rather than discretized, and so every acceptance
ratio it computes converges as the grid is refined. This example measures
that, on a nonlinear problem: the attenuation along arcs of the circle,
``d_i == exp(-integral of m over arc i)``, the function-space version of
example 31's paths.

Two passes over four grids, from 32 to 2048 points:

* at a fixed step, the acceptance rate of pCN, MALA and HMC, and of a plain
  random walk with the same proposal size as pCN, written out here since the
  library refuses to build one;
* with the step adapted during burn-in, the step each settles on.

Needs matplotlib.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import (
    ForwardProblem,
    GaussianMeasure,
    LinearOperator,
    MaximumAPosteriori,
    Operator,
    plotting,
)
from pygeoinf.inference.sampling import HMC, MALA, PCN, StepSizeAdaptation
from pygeoinf.numerics.chain_diagnostics import integrated_autocorrelation_time
from pygeoinf.symmetric_space import circle

rng = np.random.default_rng(38)

# ------------------------------------------------------------------ #
# The problem, at any resolution                                     #
# ------------------------------------------------------------------ #
#
# Six overlapping arcs, each 1.2 radians long. A datum is the attenuation
# exp(-(a_i, m)) along its arc, with (a_i, m) the path integral of the field:
# a bounded functional, so the forward map and the posterior have a limit as
# the grid is refined. The prior is a Sobolev Gaussian, the same measure at
# every resolution up to the modes the grid resolves.

arcs = [
    (np.array([a]), np.array([a + 1.2])) for a in np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
]
sigma = 0.05


def build(points: int) -> tuple:
    """The space, the prior and the forward problem on a grid of *points*."""
    space = circle.Sobolev(points, 1.0, 0.3)
    prior = space.sobolev_measure(2.0, 0.3)
    paths = space.path_integral_operator(arcs, dense=True)
    data_space = paths.codomain

    def value(m: np.ndarray) -> np.ndarray:
        """The attenuation along each arc."""
        return np.exp(-paths(m))

    def derivative(m: np.ndarray) -> LinearOperator:
        """``-diag(exp(-P m)) P``: the path integrals, weighted."""
        weights = LinearOperator.from_matrix(
            data_space, data_space, np.diag(-np.exp(-paths(m))), form="components"
        )
        return weights @ paths

    forward = Operator.from_callables(space, data_space, value, derivative=derivative)
    error = GaussianMeasure.from_standard_deviation(data_space, sigma)
    return space, prior, ForwardProblem(forward, error=error)


# One truth and one data set, made on the finest grid and shared by all: the
# question is what changes with the grid, so nothing else may.
levels = [32, 128, 512, 2048]
finest_space, finest_prior, finest_problem = build(levels[-1])
truth = finest_prior.sample(rng=rng)
data = finest_problem.forward_operator(truth) + sigma * rng.standard_normal(len(arcs))

# ------------------------------------------------------------------ #
# A random walk, for contrast                                        #
# ------------------------------------------------------------------ #
#
# m' == m + s xi with xi a prior draw: a proposal shaped like the prior, as
# large as pCN's at s == sqrt(1 - exp(-2 h)), and missing only pCN's
# contraction of m towards the prior mean. Its acceptance ratio needs the
# prior's density against the grid, the |m|^2_{C^-1} term, which exists on
# every grid and has no limit.


def random_walk(problem, prior, start, step, steps, generator, adapt=None):
    """Random-walk Metropolis; the acceptance probabilities and final step."""
    space = prior.domain
    potential = problem.potential(data)

    def log_density(m):
        return -potential(m) - 0.5 * space.inner_product(prior.precision(m), m)

    current, current_value = start, log_density(start)
    probabilities = np.empty(steps)
    for k in range(steps):
        proposal = space.axpy(step, prior.sample(rng=generator), space.copy(current))
        proposal_value = log_density(proposal)
        log_alpha = proposal_value - current_value
        probabilities[k] = np.exp(min(0.0, log_alpha))
        if np.log(generator.random()) < log_alpha:
            current, current_value = proposal, proposal_value
        if adapt is not None:
            step = adapt.update(step, k, probabilities[k])
    return probabilities, step


def mean_and_error(probabilities: np.ndarray) -> tuple[float, float]:
    """The mean acceptance probability and its Monte Carlo standard error."""
    tau = integrated_autocorrelation_time(probabilities)
    return float(probabilities.mean()), float(
        probabilities.std() * np.sqrt(tau / probabilities.size)
    )


# ------------------------------------------------------------------ #
# Pass one: a fixed step                                             #
# ------------------------------------------------------------------ #
#
# Each grid's chains start at its MAP model, so that what is measured is the
# acceptance at equilibrium rather than on the way there, and run 100 steps
# before the 300 that are kept. The steps are times (example 37); pCN's
# h == 0.005 is a proposal of relative size 0.1, the random walk's s.

steps = {"pCN": 0.005, "MALA": 0.005, "HMC": 0.08}
walk_step = np.sqrt(1.0 - np.exp(-2.0 * steps["pCN"]))
names = ["pCN", "MALA", "HMC", "random walk"]

acceptance = {name: [] for name in names}
posterior_means = {}
starts = {}
started = time.perf_counter()
print(
    f"acceptance at a fixed step; pCN h = {steps['pCN']}, MALA h = {steps['MALA']}, "
    f"HMC h = {steps['HMC']} x 3, random walk s = {walk_step:.3f}"
)
print(f"{'points':>6s}" + "".join(f"{name:>16s}" for name in names))
for points in levels:
    space, prior, problem = build(points)
    start = MaximumAPosteriori(problem, prior).solve(data).minimizer
    starts[points] = start
    samplers = {
        "pCN": PCN(problem, prior, step=steps["pCN"], acceptance=None),
        "MALA": MALA(problem, prior, step=steps["MALA"], acceptance=None),
        "HMC": HMC(problem, prior, step=steps["HMC"], length=3, acceptance=None),
    }
    row = f"{points:6d}"
    for name, sampler in samplers.items():
        posterior = sampler(
            data,
            steps=300,
            chains=2,
            burn_in=100,
            start=start,
            thin=10,
            rng=np.random.default_rng(1),
        )
        value, error = mean_and_error(
            np.stack([c.acceptance for c in posterior.chains])
        )
        acceptance[name].append((value, error))
        row += f"{value:10.3f}+-{error:.3f}"
        if name == "HMC":
            stored = posterior.stored_positions
            posterior_means[points] = sum(stored) / len(stored)
    probabilities, _ = random_walk(
        problem, prior, start, walk_step, 600, np.random.default_rng(2)
    )
    value, error = mean_and_error(probabilities)
    acceptance["random walk"].append((value, error))
    row += f"{value:10.3f}+-{error:.3f}"
    print(row)
print(f"  ({time.perf_counter() - started:.1f} s)")
print()
print("pCN, MALA and HMC accept at the same rate on every grid, to within the")
print("  Monte Carlo error. The random walk, whose proposals are as large as")
print("  pCN's, accepts less at each refinement: the same sampler, run on a")
print("  better resolved version of the same problem, stops moving.")
print()

# ------------------------------------------------------------------ #
# Pass two: the step adapted                                         #
# ------------------------------------------------------------------ #
#
# The same samplers with their step adapted during burn-in towards their usual
# acceptance. A sampler with a limit settles on one step whatever the grid; a
# random walk must shrink its step as the grid is refined to keep accepting,
# and with it how far each step moves. The random walk adapts for longer, as
# the step it needs is further from where it starts.

adapted = {name: [] for name in names}
started = time.perf_counter()
print("the step each settles on, adapted during burn-in")
print(f"{'points':>6s}" + "".join(f"{name:>16s}" for name in names))
for points in levels:
    space, prior, problem = build(points)
    start = starts[points]
    samplers = {
        "pCN": PCN(problem, prior, step=0.01),
        "MALA": MALA(problem, prior, step=0.01),
        "HMC": HMC(problem, prior, step=0.1, length=3),
    }
    row = f"{points:6d}"
    for name, sampler in samplers.items():
        posterior = sampler(
            data,
            steps=10,
            chains=2,
            burn_in=300,
            start=start,
            rng=np.random.default_rng(3),
        )
        settled = float(np.mean([chain.step_size for chain in posterior.chains]))
        adapted[name].append(settled)
        row += f"{settled:16.4f}"
    _, settled = random_walk(
        problem,
        prior,
        start,
        0.1,
        1500,
        np.random.default_rng(4),
        adapt=StepSizeAdaptation(acceptance=0.25),
    )
    adapted["random walk"].append(settled)
    row += f"{settled:16.4f}"
    print(row)
print(f"  ({time.perf_counter() - started:.1f} s)")
print()
print("the steps pCN, MALA and HMC adapt to are the same on every grid, up to")
print("  the noise of a short burn-in; the random walk's falls with the grid.")
print("  A step that has to shrink as the mesh is refined is a sampler that")
print("  needs ever more steps to explore the same posterior.")
print()

# ------------------------------------------------------------------ #
# The figure                                                         #
# ------------------------------------------------------------------ #

figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
markers = {"pCN": "o", "MALA": "s", "HMC": "^", "random walk": "x"}

axis = axes[0]
for name in names:
    values, errors = np.array(acceptance[name]).T
    axis.errorbar(
        levels, values, yerr=2 * errors, marker=markers[name], capsize=3, label=name
    )
axis.set_xscale("log")
axis.set_xticks(levels, [str(n) for n in levels])
axis.set_ylim(0.0, 1.0)
axis.set_xlabel("grid points")
axis.set_ylabel("acceptance rate")
axis.set_title("at a fixed step (two standard errors)")

axis = axes[1]
for name in names:
    axis.plot(levels, adapted[name], marker=markers[name], label=name)
axis.set_xscale("log")
axis.set_yscale("log")
axis.set_xticks(levels, [str(n) for n in levels])
axis.set_xlabel("grid points")
axis.set_ylabel("adapted step")
axis.set_title("the step adaptation settles on")

axis = axes[2]
shades = plt.get_cmap("viridis")
for k, (points, mean) in enumerate(posterior_means.items()):
    space = build(points)[0]
    axis.plot(
        space.grid_axes[0],
        mean,
        linewidth=1,
        color=shades(k / len(levels)),
        label=f"{points} points",
    )
axis.plot(
    finest_space.grid_axes[0],
    truth,
    color="black",
    linestyle="--",
    linewidth=1,
    label="truth",
)
bottom = min(
    float(np.min(truth)), *(float(np.min(m)) for m in posterior_means.values())
)
for k, (start_point, end_point) in enumerate(arcs):
    level = bottom - 0.3 - 0.12 * k
    first, last = start_point[0], end_point[0]
    pieces = (
        [(first, last)]
        if last <= 2 * np.pi
        else [(first, 2 * np.pi), (0.0, last - 2 * np.pi)]
    )
    for a, b in pieces:
        axis.plot([a, b], [level, level], color="0.5", linewidth=2)
axis.set_xlabel("position around the circle")
axis.set_ylabel("m")
top = max(float(np.max(truth)), *(float(np.max(m)) for m in posterior_means.values()))
axis.set_ylim(top=top + 0.9)
axis.set_title("HMC's posterior mean on each grid; the arcs below")
axis.legend(fontsize=7, loc="upper center", ncol=3)

handles, labels = axes[0].get_legend_handles_labels()
figure.legend(handles, labels, loc="lower center", ncol=len(names))
figure.suptitle("the samplers on four discretizations of one problem")
figure.tight_layout(rect=(0, 0.07, 1, 1))

print("one figure drawn")

plotting.show()
