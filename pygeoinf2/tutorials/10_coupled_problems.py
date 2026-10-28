# %% [markdown]
# # Tutorial 10: coupled problems
#
# Real problems rarely have one unknown. A field comes with calibration
# constants; two fields are measured together; a medium is made of layers
# that are smooth inside and jump between. The library's answer is the same
# each time: put the unknowns in one space — a **direct sum**, whose vectors
# are tuples — and build the operators and the prior on it from their parts.
# Nothing downstream changes: the inversion of tutorial 6 runs as it is.
#
# Three cases, in rising order of structure. The last needs `planetmodel`
# (`poetry install --extras planetmodel`).

# %%
import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import (
    BlockDiagonalLinearOperator,
    DirectSum,
    EuclideanSpace,
    GaussianMeasure,
    LinearForwardProblem,
    LinearGaussianInversion,
    LinearOperator,
    RowLinearOperator,
    plotting,
)
from pygeoinf2.probability import product
from pygeoinf2.symmetric_space.circle import Sobolev

rng = np.random.default_rng(10)

# %% [markdown]
# ## 1. A field and a number
#
# Tutorial 6's ring of thirty temperature sensors — but now every sensor
# shares an unknown calibration offset $b$. The data are $A m + b\,\mathbf 1$:
# the field's arc averages, plus the same offset on each. Two unknowns of
# different kinds, a function and a scalar, in one model space.

# %%
X = Sobolev(256, 1.0, 0.1)
B = EuclideanSpace(1)
model_space = DirectSum([X, B], labels=("temperature", "offset"))
print(model_space)

x = model_space.random(rng=rng)
print("a vector is a", type(x).__name__, "of", len(x), "parts")
print("reached by name:", model_space.component(x, "offset"))

# %% [markdown]
# The inner product of a direct sum is the sum of the parts' inner products,
# so the field keeps its $H^1$ metric and the offset its Euclidean one.
#
# The forward operator is a *row* of two operators, one per part:
# $(m, b) \mapsto A m + b\,\mathbf 1$. Its adjoint is a column, built for you.
# The prior is the independent *product* of a prior on each part: a smooth
# field about 15 degrees, and an offset of typical size 0.5.

# %%
sensors = np.sort(rng.uniform(0.0, 2.0 * np.pi, 30))
A = X.geodesic_ball_average_operator([np.array([s]) for s in sensors], 0.05)
D = A.codomain
ones = LinearOperator.from_matrix(B, D, np.ones((D.dim, 1)), form="components")
forward = RowLinearOperator([A, ones])

baseline = X.project_function(lambda t: 15.0 + 0.0 * t)
prior = product(
    [
        X.heat_measure(0.3, pointwise_std=1.0, expectation=baseline),
        GaussianMeasure.from_standard_deviation(B, 0.5),
    ],
    labels=("temperature", "offset"),
)
problem = LinearForwardProblem(
    forward, error=GaussianMeasure.from_standard_deviation(D, 0.05)
)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)
posterior = LinearGaussianInversion(problem, prior)(data)
print("the posterior is a", type(posterior).__name__, "on", posterior.domain)

# %% [markdown]
# A posterior on a direct sum answers questions about each part through its
# **marginals**. Summands are reached by position here: the posterior lives
# on the forward operator's domain, which the row operator built without the
# prior's labels.

# %%
offset = posterior.marginal(1)
sd = np.sqrt(offset.covariance.matrix(form="components")[0, 0])
print(
    f"offset: truth {truth[1][0]:+.3f}, posterior {offset.expectation[0]:+.3f} +/- {sd:.3f}"
)
print("the prior said +/- 0.5")

# %% [markdown]
# The data hardly narrowed the offset. The reason is a trade-off the joint
# posterior knows about: a uniform warming of the whole ring and an offset
# on every sensor produce identical data. Push the posterior forward to the
# pair (ring average, offset) — a block-diagonal operator, one block per
# part — and the correlation says so.

# %%
ring_average = X.geodesic_ball_average_operator([np.array([np.pi])], np.pi)
pair = posterior.push_forward(
    BlockDiagonalLinearOperator([ring_average, LinearOperator.identity(B)])
)
covariance = np.block(
    [
        [pair.cross_covariance(i, j).matrix(form="components") for j in (0, 1)]
        for i in (0, 1)
    ]
)
correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
print(f"posterior correlation of the ring average with the offset: {correlation:+.3f}")

# %% [markdown]
# Almost exactly $-1$: the data pin down the *sum* of the two and nothing
# else. Only the priors separate them. A joint model makes such trade-offs
# visible instead of hiding them in a field that silently absorbed a
# calibration error.
#
# ## 2. Two fields, one prior
#
# Next, temperature and salinity around the ring, correlated with each
# other — warm water tends to be salty. The joint prior lives on the direct
# sum of two copies of the space. On a symmetric space such a prior can be
# written scale by scale: here each field has the heat spectrum, calibrated
# to unit pointwise variance, and the two are correlated with coefficient
# 0.8 at every scale.

# %%
spectrum = X.heat_symbol(0.3)
spectrum = spectrum / X.pointwise_variance(spectrum)
joint = X.correlated_measure_from_correlations(
    [spectrum, spectrum],
    np.array([[1.0, 0.8], [0.8, 1.0]]),
    labels=("temperature", "salinity"),
)

# Only temperature is measured: a row with a zero in the salinity slot.
measure_temperature = RowLinearOperator([A, LinearOperator.zero(X, codomain=D)])
problem = LinearForwardProblem(
    measure_temperature, error=GaussianMeasure.from_standard_deviation(D, 0.05)
)
truth, data = problem.synthetic_model_and_data(joint, rng=rng)
posterior = LinearGaussianInversion(problem, joint)(data)

salinity = posterior.marginal(1)
grid = X.grid_axes[0]
spread = np.sqrt(X.pointwise_variance_at(salinity, grid[::4]))
print(
    f"salinity was never measured; its standard deviation went from 1 to {spread.min():.2f}-{spread.max():.2f}"
)
print(
    f"the most it could fall to with correlation 0.8 is sqrt(1 - 0.8^2) = {np.sqrt(1 - 0.64):.2f}"
)

# %% [markdown]
# Everything the data say about salinity they say through the prior's
# correlation, and no more: even a perfectly known temperature would leave
# salinity uncertain by $\sqrt{1 - \rho^2}$. The posterior got close to that
# limit where the sensors are dense.
#
# ## 3. Fields that jump: layers
#
# A layered medium is smooth inside each layer and discontinuous between
# them. Such a field is in no single smooth space, but it is in a direct sum
# of them, one per layer. `Layered` is that direct sum, with what a plain
# direct sum cannot know added: which layer a point is in, the two values a
# field has at an interface, the jump across it, integrals over the whole.
#
# The example is a slowness profile — the reciprocal of seismic speed —
# down to unit depth: slow sediment above 0.4, fast bedrock below, each with
# its own correlation length.

# %%
# The layered spaces are built on planetmodel's spectral elements, an optional
# extra, so they are imported only where they are used.
from pygeoinf2.interval import Layered  # noqa: E402

L = Layered(
    [0.0, 0.4, 1.0],
    [20, 30],  # spectral modes kept in each layer
    order=1.0,
    length_scale=[0.1, 0.15],
    labels=["sediment", "bedrock"],
)
print(L, "with an interface at", L.interfaces)

reference = L.project_function([lambda z: 0.6 + 0.2 * z, lambda z: 0.3 + 0.05 * z])
layered_prior = L.sobolev_measure(
    1.5, expectation=reference, pointwise_std=[0.05, 0.03]
)
jump = L.jump_operator(0)
print(f"the reference slowness jumps by {jump(reference)[0]:+.3f} at the interface")

# %% [markdown]
# The prior is a *product* over layers: knowing the sediment says nothing
# about the bedrock. That independence is what a discontinuity means.
#
# The data are of three kinds, and the space makes no difference between
# them: the vertical travel time through the whole column (the integral of
# slowness), the jump at the interface as a reflection sees it, and slowness
# logged in a shallow borehole at four depths.

# %%
travel_time = L.integral_functional()
depths = np.array([0.05, 0.15, 0.25, 0.35])
rows = np.vstack(
    [
        travel_time.matrix(form="galerkin"),
        jump.matrix(form="galerkin"),
        L.basis_matrix(depths),
    ]
)
forward = LinearOperator.from_matrix(
    L, EuclideanSpace(rows.shape[0]), rows, form="galerkin"
)
noise = GaussianMeasure.from_standard_deviations(
    forward.codomain, np.array([0.002, 0.02, 0.01, 0.01, 0.01, 0.01])
)
problem = LinearForwardProblem(forward, error=noise)
truth = layered_prior.sample(rng=rng)
data = problem.synthetic_data(truth, rng=rng)
posterior = LinearGaussianInversion(problem, layered_prior)(data)
print(
    f"jump: truth {jump(truth)[0]:+.3f}, posterior mean {jump(posterior.expectation)[0]:+.3f}"
)
print(
    f"travel time: truth {travel_time(truth):.4f}, posterior {travel_time(posterior.expectation):.4f}"
)

# %% [markdown]
# And the converse of independence: if the two layers were known to be
# *continuous* at the interface, that is the same prior **conditioned** on
# the jump being zero. Conditioning is exact and samplable; it is the way an
# exact linear constraint enters a Gaussian prior.

# %%
welded = layered_prior.condition(jump, np.zeros(1))
jumps = [jump(welded.sample(rng=rng))[0] for _ in range(100)]
print(
    f"conditioned on continuity: the largest jump in 100 draws is {max(map(abs, jumps)):.1e}"
)


# %%
def band(measure, lower, upper, count=50):
    """The mean and two standard deviations inside one layer."""
    depth = np.linspace(lower + 1e-3, upper - 1e-3, count)
    look = L.point_evaluation_operator(depth)
    variance = np.diag(
        (look @ measure.covariance @ look.adjoint).matrix(form="components")
    )
    return depth, look(measure.expectation), 2.0 * np.sqrt(variance)


fig, (left, right) = plt.subplots(
    1, 2, figsize=(9.0, 3.5), sharey=True, layout="constrained"
)
for draw in (layered_prior.sample(rng=rng) for _ in range(3)):
    plotting.plot(L, draw, ax=left, lw=0.8)
for draw in (welded.sample(rng=rng) for _ in range(2)):
    plotting.plot(L, draw, ax=left, lw=0.8, ls="--", color="black")
left.set_title("Prior draws: independent (solid), welded (dashed)", fontsize=9)
for lower, upper in L.layer_bounds:
    depth, mean, width = band(posterior, lower, upper)
    right.fill_between(depth, mean - width, mean + width, color="C0", alpha=0.3, lw=0.0)
    right.plot(depth, mean, color="C0")
plotting.plot(L, truth, ax=right, color="black", lw=0.8, label="truth")
right.plot(depths, data[2:], "o", color="C3", ms=4, label="borehole")
right.set_title("Posterior: travel time, a reflection, a borehole", fontsize=9)
right.legend(fontsize=8)
for ax in (left, right):
    ax.set_xlabel("depth")
left.set_ylabel("slowness")

# %% [markdown]
# The borehole pins the sediment; the travel time and the reflection
# constrain the bedrock only through their totals, so its band stays wide
# away from the interface. Example 33 is the same construction for a planet,
# with mass and moment of inertia as the integrals; tutorial 11 turns to the
# spaces each layer is built from.

# %%
plotting.show()
