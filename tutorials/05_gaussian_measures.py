# %% [markdown]
# # Tutorial 5: Gaussian measures on function spaces
#
# A prior is a probability measure on the space the unknown lives in. When
# that space discretises functions, the measure is a statement about which
# *functions* are plausible — how large, how smooth, how rough — and it
# should make the same statement however finely the functions are
# discretised. This tutorial is about building such measures, drawing from
# them, mapping them through operators, and one thing they cannot do: have a
# density in the usual sense.

# %%
import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import EuclideanSpace, GaussianMeasure, plotting
from pygeoinf.symmetric_space.circle import Lebesgue, Sobolev
from pygeoinf.testing import check_measure

rng = np.random.default_rng(5)

# %% [markdown]
# ## A measure is something you can draw from
#
# A Gaussian measure is fixed by its mean, a vector of the space, and its
# covariance, a self-adjoint positive operator on it. On a space of 256
# grid values the covariance would be a 256 × 256 matrix, but nothing here
# forms one: the measures below are *invariant* — they look the same at
# every point of the circle — and their covariance is diagonal in the
# Fourier basis, so applying it and drawing from it both cost one transform.
#
# The host space is $H^1$ on the circle, the space whose norm the
# inversions of later tutorials will measure things in. The measure is a
# separate object living on it.

# %%
X = Sobolev(256, 1.0, 0.1)
grid = X.grid_axes[0]

# %% [markdown]
# ## Four families
#
# A symmetric space offers four ways to write down an invariant Gaussian,
# each fixed by its spectrum — the variance it gives each Fourier mode:
#
# - `sobolev_measure(order, scale)` — variance $(1 + \ell^2 k^2)^{-s}$: the
#   *order* sets how smooth draws are, the *scale* where the spectrum turns
#   over;
# - `heat_measure(scale)` — variance $e^{-\ell^2 k^2}$: infinitely smooth
#   draws with correlation length $\ell$;
# - `power_measure(p)` — the power $p(k)$ per wavenumber, the way a
#   geophysicist describes a field;
# - `invariant_measure(variances)` — any spectrum at all, which the other
#   three call.
#
# Each takes `pointwise_std`, which rescales the spectrum so the field's
# standard deviation at a point is what you ask for. That is the number a
# modeller has an opinion about: nobody knows what spectral amplitude they
# want, everybody knows roughly how large the field is.

# %%
priors = {
    "Sobolev, order 1": X.sobolev_measure(1.0, 0.1, pointwise_std=1.0),
    "Sobolev, order 3": X.sobolev_measure(3.0, 0.1, pointwise_std=1.0),
    "heat, scale 0.3": X.heat_measure(0.3, pointwise_std=1.0),
}

fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.0), sharex=True, layout="constrained")
for ax, (name, prior) in zip(axes, priors.items()):
    for _ in range(3):
        plotting.plot(X, prior.sample(rng=rng), ax=ax, lw=1.0)
    ax.set_title(name, fontsize=10)
axes[-1].set_xlabel("angle")

# %% [markdown]
# Three measures, the same pointwise size, very different functions. The
# difference is in the correlation: how quickly the value at one point stops
# telling you about the value at another. For an invariant measure that is a
# function of distance alone, the *covariance function*.

# %%
distances = np.linspace(0.0, 1.5, 151)
fig, ax = plt.subplots(figsize=(7.0, 3.0), layout="constrained")
for name, prior in priors.items():
    ax.plot(distances, X.covariance_function(prior, distances), label=name)
ax.set_xlabel("distance")
ax.set_ylabel("covariance")
ax.legend()
ax.set_title("How far a point value reaches")

# %% [markdown]
# ## The space and the measure are different things
#
# The variances handed to `invariant_measure` — and so to `sobolev_measure`
# and `heat_measure` — are eigenvalues of the covariance *operator*, and an
# operator's eigenvalues as a covariance depend on the inner product it is a
# covariance in. So the same call on two host spaces makes two different
# random fields: on $H^1$ the draws are smoother than on $L^2$, because the
# host's own metric divides the spectrum.
#
# `power_measure` is the exception, and the reason it exists: its argument
# is the $L^2$ power, what you would measure from a draw, and it means the
# same field on any host. Compare the power of draws from the two
# constructions, averaged over forty draws, at a few wavenumbers.

# %%
L2 = Lebesgue(256)
wavenumbers = [2, 10, 40]
print(f"{'':20s}" + "".join(f"  k={k:<6d}" for k in wavenumbers))
for name, build in [
    ("heat", lambda S: S.heat_measure(0.05)),
    ("power", lambda S: S.power_measure(lambda k: np.exp(-0.0025 * k**2))),
]:
    for host, host_name in ((L2, "L2"), (X, "H1")):
        measure = build(host)
        draws = [measure.sample(rng=rng) for _ in range(40)]
        spectra = np.mean([host.power_spectrum(draw) for draw in draws], axis=0)
        label = f"{name} measure on {host_name}"
        print(f"{label:20s}" + "".join(f"  {spectra[k]:8.2e}" for k in wavenumbers))

# %% [markdown]
# The heat measure's high wavenumbers are weaker on $H^1$ than on $L^2$; the
# power measure's are the same on both, to sampling error. Neither is wrong.
# Use the operator-eigenvalue families when you think of the prior as a
# covariance operator on the space; use `power_measure` when you think of it
# as a spectrum.
#
# ## Smoothness that survives refinement
#
# A measure on a function space should not change its mind when the grid is
# refined. The pointwise standard deviation is the test. Below, a Sobolev
# measure with fixed amplitude, at two orders, on grids from 32 to 2048
# points, hosted on $L^2$ so that no metric smooths it: the field's typical
# size at a point either settles down or keeps growing. The library refuses
# to build the order-0.3 measure, for the reason the table shows;
# `unsafe=True` is the escape hatch for looking.

# %%
print(f"{'grid':>6} {'order 0.3':>10} {'order 1.5':>10}")
for n in (32, 128, 512, 2048):
    host = Lebesgue(n)
    stds = []
    for order in (0.3, 1.5):
        measure = host.sobolev_measure(order, 0.1, unsafe=True)
        draws = [measure.sample(rng=rng) for _ in range(20)]
        stds.append(np.std(np.concatenate(draws)))
    print(f"{n:6d} {stds[0]:10.2f} {stds[1]:10.2f}")

# %% [markdown]
# On a circle the variance at a point is the sum of the spectrum, and
# $\sum_k (1 + \ell^2 k^2)^{-s}$ converges only for $s > 1/2$. Below that the
# measure is not one on continuous functions — its draws have no values at
# points — and refining the grid exposes it. In $d$ dimensions the
# threshold is $s > d/2$.
#
# ## Push-forwards
#
# A linear operator maps a Gaussian measure to a Gaussian measure: the mean
# goes through the operator, the covariance becomes $T C T^*$. The usual use
# is a *property*: here, the averages of the field over two arcs of
# half-width 0.4, centred half a radian apart. The push-forward is a
# Gaussian on $\mathbb{R}^2$, computed exactly, without a sample.

# %%
prior = priors["heat, scale 0.3"]
T = X.geodesic_ball_average_operator([np.array([1.0]), np.array([1.5])], 0.4)
averages = T @ prior
print("the push-forward is a", type(averages).__name__, "on", averages.domain)
covariance = averages.covariance.matrix(form="components")
print("its covariance:\n", covariance.round(3))
correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
print(f"correlation of the two averages: {correlation:.2f}")

# A claim about a measure's moments is checkable by sampling it.
check_measure(averages, rng=rng, samples=4000, rtol=0.1)
print("sampled moments agree with the declared ones")

# %%
axes = plotting.plot_corner(
    averages,
    labels=["arc at 1.0", "arc at 1.5"],
    posterior_label="prior, pushed forward",
    rng=rng,
)
axes[0, 0].figure.suptitle("Two arc averages under the heat prior")

# %% [markdown]
# ## What a density means here, and why it is refused
#
# On $\mathbb{R}^n$ a Gaussian has a density, and its logarithm is
# $-\tfrac12 \|x - m\|^2_{C^{-1}}$ up to a constant: the squared
# *Mahalanobis* distance from the mean. On a finite-dimensional space the
# library computes it.

# %%
R3 = EuclideanSpace(3)
small = GaussianMeasure.from_standard_deviation(R3, 2.0)
print(f"log density on R^3: {small.log_density(np.array([1.0, -1.0, 0.5])):.4f}")

# %% [markdown]
# On a function space it refuses. Watch what the quantity behind it does
# under refinement, for a smooth function and for a typical draw from the
# measure itself. (`unsafe=True` overrides the refusal, which is what it is
# for: looking.)

# %%
print(f"{'grid':>6} {'smooth function':>16} {'a draw':>10}")
for n in (32, 128, 512, 2048):
    host = Sobolev(n, 1.0, 0.1)
    measure = host.sobolev_measure(1.5, 0.1)
    smooth = np.sin(host.grid_axes[0])
    draw = measure.sample(rng=rng)
    print(
        f"{n:6d} {measure.mahalanobis_squared(smooth, unsafe=True):16.3f}"
        f" {measure.mahalanobis_squared(draw, unsafe=True):10.1f}"
    )

try:
    measure.log_density(draw)
except ValueError as error:
    print("\nrefused:", str(error).split(".")[0] + ".")

# %% [markdown]
# For the smooth function the Mahalanobis distance converges; for the
# measure's own typical draw it grows like the number of grid points. In the
# limit a draw from a Gaussian measure on a function space is *infinitely*
# far from its mean in the measure's own metric — the draws lie outside the
# space of functions the density can see, the Cameron–Martin space. So the
# density of a draw depends on the grid, falls without bound as the grid is
# refined, and comparing two such numbers compares two discretisation
# artefacts.
#
# That is why the library refuses: every space that discretises functions
# says so through `is_function_space`, and the density family checks it.
# It is the same pattern as the refusal of point evaluation on a space too
# rough for it, and it is a feature — the number it would have returned is
# meaningless, and nothing else in the arrays would have said so.

# %%
print("X discretises functions:", X.is_function_space)
print("R^3 does not:          ", R3.is_function_space)

# %% [markdown]
# What *does* survive refinement is a density **relative to another
# measure**: the posterior relative to the prior, for instance, whose log is
# the data misfit and is finite for every draw. Tutorial 7 meets it, as the
# density a posterior carries. Sampling methods are built on that, and it is
# why they can be made to work in any dimension.

# %%
plotting.show()
