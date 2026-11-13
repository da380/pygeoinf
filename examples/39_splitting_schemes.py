"""
39. Sampling: a splitting scheme written by hand.

The moves of example 37 are each a few lines over the sampling package's
lower layers: a dynamics, whose exact flows are its letters; a splitting
scheme, a word over those letters applied with a time step; and a correction,
which makes the scheme exact whatever its step. This example takes ``HMC``
apart into those layers, checks that the parts give the move's chains to the
last bit, and then changes the one thing a move fixes, the word.

``Hamiltonian`` dynamics has three letters, each solved exactly:

* ``A``, :class:`PriorRotation`: the reference's own dynamics, a rotation of
  position and velocity about the prior mean;
* ``B``, :class:`LikelihoodKick`: the data's force, a kick to the velocity
  along the preconditioned gradient of the likelihood potential;
* ``O``, :class:`MomentumRefresh`: a fresh velocity from ``N(0, K)``.

``HMC`` is the word ``"OBAB"`` with the block ``"BAB"``, the leapfrog step,
repeated and Metropolized. A word may be any other: here ``"ABA"``, the
leapfrog step with the roles of rotation and kick swapped, and a two-stage
step that kicks three times and rotates twice per step, with fractions of the
step chosen to make its error small. All three are second order: the error in
the energy over a trajectory of fixed length falls as ``h^2``. The library
takes positive fractions only, and every splitting of higher order needs a
negative one, so the schemes differ in their constant and their stability,
not their order. The example measures both, and then what they are worth to
the chains for the same number of gradients.

The problem is example 38's, a field on a circle seen through the attenuation
along arcs of it, with more arcs and noisier data.

Needs matplotlib.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import (
    HMC,
    Chains,
    ForwardProblem,
    GaussianMeasure,
    LinearOperator,
    MaximumAPosteriori,
    MonteCarloEstimator,
    Operator,
    plotting,
)
from pygeoinf.sampling import (
    Hamiltonian,
    MetropolizedBlock,
    SplittingScheme,
    Target,
)
from pygeoinf.symmetric_space import circle

rng = np.random.default_rng(39)

# ------------------------------------------------------------------ #
# The problem                                                        #
# ------------------------------------------------------------------ #
#
# Twelve arcs of a radian each, evenly spaced, so that every part of the
# circle is seen by two of them. A datum is exp(-(a_i, m)), with (a_i, m) the
# integral of the field along arc i. The forward map counts its derivatives,
# each of which is one gradient of the potential, the cost of a kick.

arcs = [
    (np.array([a]), np.array([a + 1.0])) for a in np.linspace(0.0, 2.0 * np.pi, 13)[:-1]
]
sigma = 0.1
space = circle.Sobolev(256, 1.0, 0.3)
prior = space.sobolev_measure(2.0, 0.3)
paths = space.path_integral_operator(arcs, dense=True)
data_space = paths.codomain
gradients = [0]


def value(m: np.ndarray) -> np.ndarray:
    """The attenuation along each arc."""
    return np.exp(-paths(m))


def derivative(m: np.ndarray) -> LinearOperator:
    """``-diag(exp(-P m)) P``, counted."""
    gradients[0] += 1
    weights = LinearOperator.from_matrix(
        data_space, data_space, np.diag(-np.exp(-paths(m))), form="components"
    )
    return weights @ paths


forward = Operator.from_callables(space, data_space, value, derivative=derivative)
error = GaussianMeasure.from_standard_deviation(data_space, sigma)
problem = ForwardProblem(forward, error=error)
truth = prior.sample(rng=rng)
data = value(truth) + sigma * rng.standard_normal(len(arcs))

# Every chain starts at the MAP model, so that what is measured is the
# schemes in equilibrium rather than on their way there. What each chain
# records is the twelve path integrals, the quantities the data constrain.
start = MaximumAPosteriori(problem, prior).solve(data).minimizer

# ------------------------------------------------------------------ #
# The move, and the same sampler by hand                             #
# ------------------------------------------------------------------ #
#
# The by-hand kernel is built in the package's order: the target, the
# residual potential against the reference, here the prior; the dynamics on
# it, which supplies the letters; the scheme, a word over them at a step;
# and the correction, which Metropolizes the block and runs the rest of the
# word, the refresh, as it stands. Chains then runs it as it runs the move,
# which the estimator builds on its own target for the data. With the step
# held fixed and the same generator, the two give the same chains exactly.

step, length = 0.1, 6
move = HMC(step=step, length=length, acceptance=None)
sampler = Chains(move, steps=200, chains=2, rng=np.random.default_rng(1))
estimated = MonteCarloEstimator(problem, prior, sampler=sampler, record=paths)(
    data, start=start
)

target = Target(problem.potential(data), prior)
dynamics = Hamiltonian(target)
scheme = SplittingScheme(dynamics, "OBAB", step=step)
kernel = MetropolizedBlock(scheme.block("BAB", repeat=length)).within(scheme)
by_hand = Chains(kernel, steps=200, chains=2, rng=np.random.default_rng(1))(
    target, start=start, record=paths
)

print(f"the Hamiltonian dynamics' letters: {dynamics.flows()}")
print(f"the move's kernel: {scheme!r} with the block {kernel!r}")
print(
    "the move and the kernel by hand give the same chains: "
    f"{np.array_equal(estimated.scalars(), by_hand.scalars())}"
)
print()

# What the correction will not take. A Metropolized block is inverted by a
# velocity flip only if it is a palindrome of deterministic flows: "BA" is
# not a palindrome, and a refresh inside the block is not deterministic.
for word in ("BA", "OBAB"):
    try:
        MetropolizedBlock(scheme.block(word))
    except ValueError as refusal:
        print(f"the block {word!r} is refused: {refusal}")
print()

# ------------------------------------------------------------------ #
# Three words                                                        #
# ------------------------------------------------------------------ #
#
# A word is a string, each letter taking an equal share of the step for every
# time it appears, or a sequence of (letter, fraction) pairs. The two-stage
# step is written in pairs: kicks of b, 1 - 2b and b, and two half rotations.
# Its weight b == 0.1932 minimizes the leading error term (McLachlan, 1995).
# It costs two gradients per step, where a leapfrog step costs one, the last
# kick of one step and the first of the next being at the same position and
# the gradient cached on the state; so it runs at twice the step for the same
# cost.

b = 0.1932
two_stage = [("B", b), ("A", 0.5), ("B", 1.0 - 2.0 * b), ("A", 0.5), ("B", b)]
words = {
    "BAB, as HMC": ("OBAB", "BAB", 1),
    "ABA": ("OABA", "ABA", 1),
    "two-stage": ([("O", 1.0)] + two_stage, two_stage, 2),
}
colors = dict(zip(words, ("#2a78d6", "#eb6834", "#1baf7a")))


def build(name: str, cost: int, duration: float):
    """The kernel of a word whose trajectories last *duration* and cost
    *cost* gradients."""
    word, block, stages = words[name]
    repeat = cost // stages
    scheme = SplittingScheme(dynamics, word, step=duration / repeat)
    return MetropolizedBlock(scheme.block(block, repeat=repeat)).within(scheme)


def run(name: str, cost: int, duration: float, chains: int, burn_in: int, steps: int):
    """Chains of one word; the sampled posterior and the gradients spent.

    The seed is the cost's, shared by the three words: at each cost they
    start from the same velocities and meet the same uniforms, so that what
    differs between their rows is the word.
    """
    gradients[0] = 0
    kernel = build(name, cost, duration)
    sampler = Chains(
        kernel,
        steps=steps,
        burn_in=burn_in,
        chains=chains,
        rng=np.random.default_rng(cost),
    )
    posterior = sampler(kernel.target, start=start, record=paths)
    return posterior, gradients[0]


# ------------------------------------------------------------------ #
# The integrators' convergence                                       #
# ------------------------------------------------------------------ #
#
# Trajectories of a fixed length in time, 1.2, made of ever more steps. The
# energy error is what the Metropolis step corrects, ``-log alpha``: the
# change in the residual potential plus the change in the reference's
# Gaussian energy, which the flows report as they run. A block is itself a
# scheme, and applying it to a state is one trajectory with no correction,
# so the error can be measured on the same states and velocities for every
# word and every step: eighty states from a run of HMC, each with a
# velocity from N(0, K). On a log-log plot against the gradients per
# trajectory a second-order scheme is a line of slope -2, once its step is
# short enough to be stable.

duration = 1.2
costs = [12, 16, 24, 32, 48, 96, 192]
sampler = Chains(
    HMC(step=0.05, length=24, acceptance=None),
    steps=200,
    chains=4,
    thin=10,
    rng=np.random.default_rng(2),
)
states = MonteCarloEstimator(problem, prior, sampler=sampler)(
    data, start=start
).stored_positions
velocities = np.random.default_rng(3)
states = [dynamics.initial_state(position, rng=velocities) for position in states]


def energy_error(name: str, cost: int) -> float:
    """The median energy error of a word's trajectories from the states."""
    _, block, stages = words[name]
    repeat = cost // stages
    trajectory = SplittingScheme(dynamics, block, step=duration / repeat, repeat=repeat)
    errors = []
    for state in states:
        end, change = trajectory.apply(state)
        errors.append(abs(target.value(end) - target.value(state) + change))
    return float(np.median(errors))


energy = {name: [] for name in words}
started = time.perf_counter()
print(
    f"the median energy error over a trajectory of length {duration}, from "
    f"{len(states)} states"
)
print(f"{'gradients':>9s}" + "".join(f"{name:>18s}" for name in words))
for cost in costs:
    row = f"{cost:9d}"
    for name in words:
        energy[name].append(energy_error(name, cost))
        row += f"{energy[name][-1]:18.2e}"
    print(row)
print(f"  ({time.perf_counter() - started:.1f} s)")
print()
halving = {name: energy[name][-2] / energy[name][-1] for name in words}
print(
    f"halving the step, from {costs[-2]} gradients to {costs[-1]}, divides the error by "
    + ", ".join(f"{ratio:.2f} ({name})" for name, ratio in halving.items())
    + ":"
)
print("  second order, all three, as a palindrome with positive fractions must be.")
print(
    "  At the same cost the two-stage step's error is "
    f"{energy['BAB, as HMC'][-1] / energy['two-stage'][-1]:.0f} times smaller "
    "than leapfrog's,"
)
closest = max(
    abs(aba / bab - 1.0)
    for aba, bab in zip(energy["ABA"], energy["BAB, as HMC"], strict=True)
)
print(f"  and ABA's is leapfrog's to within {closest:.0%} at every cost. With long")
print("  steps the order is reversed: at 16 gradients a trajectory leapfrog keeps")
print(
    f"  to an error of {energy['BAB, as HMC'][1]:.2f} costs the two-stage step "
    f"{energy['two-stage'][1]:.0f}. It takes the longer step per gradient, and"
)
print("  its stable range per gradient is the shorter.")
print()

# ------------------------------------------------------------------ #
# What the chains get for it                                         #
# ------------------------------------------------------------------ #
#
# Four chains of each word at three costs, and the effective sample size of
# the worst of the twelve path integrals per thousand gradients. Every
# scheme is exact, so the means must agree within their errors; what can
# differ is how many gradients each takes to get there.

chain_costs = [16, 24, 48]
mixing = {name: {"acceptance": [], "ess": []} for name in words}
started = time.perf_counter()
print("four chains of 300 steps after 50 of burn-in, at each cost")
print(
    f"{'gradients':>9s} {'word':18s} {'accept':>6s} {'ESS/1000':>8s} {'R-hat':>6s}"
    f" {'mean of arc 1':>16s}"
)
for cost in chain_costs:
    for name in words:
        posterior, spent = run(name, cost, duration, chains=4, burn_in=50, steps=300)
        diagnostics = posterior.diagnostics()
        ess = float(diagnostics.effective_sample_size[1:].min())
        arc = posterior.scalars()[1]
        error_of_mean = arc.std() * np.sqrt(
            diagnostics.autocorrelation_time[1] / arc.size
        )
        mixing[name]["acceptance"].append(posterior.acceptance_rate)
        mixing[name]["ess"].append(1000.0 * ess / spent)
        print(
            f"{cost:9d} {name:18s} {posterior.acceptance_rate:6.2f} "
            f"{mixing[name]['ess'][-1]:8.2f} {diagnostics.rhat[1:].max():6.2f} "
            f"{arc.mean():8.4f}+-{error_of_mean:.4f}"
        )
print(f"  ({time.perf_counter() - started:.1f} s)")
print()
lowest = min(
    acceptance
    for name in words
    for acceptance in mixing[name]["acceptance"][chain_costs.index(24) :]
)
print("At 16 gradients only the leapfrog words' trajectories are stable, and the")
print(
    "  two-stage chains barely move, accepting "
    f"{mixing['two-stage']['acceptance'][0]:.0%} of their trajectories; R-hat says"
)
print(f"  so. From 24 every word accepts at least {lowest:.0%} of them, and the")
print("  two-stage step's smaller error has nothing left to buy: its worst ESS")
print(
    f"  per thousand gradients is {mixing['two-stage']['ess'][1]:.1f} at 24 and "
    f"{mixing['two-stage']['ess'][2]:.1f} at 48, against leapfrog's "
    f"{mixing['BAB, as HMC']['ess'][1]:.1f} and"
)
print(
    f"  {mixing['BAB, as HMC']['ess'][2]:.1f}. The means agree within their "
    "errors wherever the"
)
print("  chains have mixed, since every word is corrected and so exact: the word")
print("  decides the cost of a sample, never its law. A smaller error constant")
print("  can pay where the energy error is what holds the acceptance down at a")
print("  cost the scheme is stable at, which needs more directions the data")
print("  constrain closely than these twelve arcs give.")
print()

# ------------------------------------------------------------------ #
# The figure                                                         #
# ------------------------------------------------------------------ #

figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))

axis = axes[0]
for name in words:
    axis.plot(costs, energy[name], "o-", color=colors[name], label=name)
# A guide of slope -2, set below the leapfrog line so that both show.
reference = 0.3 * energy["BAB, as HMC"][-1] * (costs[-1] / np.array(costs[3:])) ** 2
axis.plot(costs[3:], reference, color="0.6", linestyle="--", label="slope -2")
axis.set_xscale("log")
axis.set_yscale("log")
axis.set_xticks(costs, [str(cost) for cost in costs])
axis.xaxis.set_minor_locator(plt.NullLocator())
axis.set_xlabel("gradients per trajectory")
axis.set_ylabel("median energy error")
axis.set_title("the integrators' convergence", fontsize=10)
axis.legend(frameon=False, fontsize=8)

for axis, key, label, title in (
    (axes[1], "acceptance", "acceptance", "what the error costs the chains"),
    (
        axes[2],
        "ess",
        "worst ESS per 1000 gradients",
        "what a gradient buys",
    ),
):
    for name in words:
        axis.plot(chain_costs, mixing[name][key], "o-", color=colors[name], label=name)
    axis.set_xscale("log")
    axis.set_xticks(chain_costs, [str(cost) for cost in chain_costs])
    axis.minorticks_off()
    axis.set_xlabel("gradients per trajectory")
    axis.set_ylabel(label)
    axis.set_title(title, fontsize=10)
axes[1].set_ylim(-0.03, 1.03)
axes[1].legend(frameon=False, fontsize=8)
for axis in axes:
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
figure.suptitle(
    f"three splitting schemes for the same Hamiltonian dynamics, trajectories "
    f"of length {duration}"
)
figure.tight_layout()

print("one figure drawn")

plotting.show()
