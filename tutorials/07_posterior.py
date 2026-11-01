# %% [markdown]
# # Tutorial 7: the posterior
#
# Tutorial 6 posed the forward problem and answered it with a point. Here
# the estimator adds a prior, a measure on the model space, and what comes
# back is a measure too: the posterior of tutorial 1's five lines, taken
# apart. This tutorial looks at what it carries — a mean and a covariance,
# draws, a density relative to the prior, and the evidence — and then asks
# it a question about a *property* of the unknown, with a credible set as
# the answer.

# %%
import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import (
    CholeskySolver,
    GaussianMeasure,
    LinearForwardProblem,
    LinearGaussianInversion,
    ModelSpace,
    Traits,
    plotting,
)
from pygeoinf.symmetric_space.circle import Sobolev

rng = np.random.default_rng(6)

# %% [markdown]
# ## The forward problem
#
# The unknown is a temperature around a ring. Thirty sensors each report the
# average temperature over a short arc — their footprint — with independent
# noise of standard deviation 0.1. The forward operator maps a temperature
# field to those thirty averages; the error measure is the distribution of
# the noise. Together they are a `LinearForwardProblem`.

# %%
X = Sobolev(256, 1.0, 0.1)
sensors = np.sort(rng.uniform(0.0, 2.0 * np.pi, 30))
A = X.geodesic_ball_average_operator([np.array([s]) for s in sensors], 0.05)
noise = GaussianMeasure.from_standard_deviation(A.codomain, 0.1)
problem = LinearForwardProblem(A, error=noise)
print(problem)

# A truth, drawn from a prior, and data from the truth.
truth_prior = X.heat_measure(0.3, pointwise_std=1.0)
truth, data = problem.synthetic_model_and_data(truth_prior, rng=rng)

# %% [markdown]
# ## The posterior
#
# Now a prior, and the estimator. `LinearGaussianInversion` is the
# closed-form Bayesian update for a linear problem with Gaussian prior and
# noise; called on data it returns the posterior.

# %%
prior = X.heat_measure(0.3, pointwise_std=1.0)
inversion = LinearGaussianInversion(problem, prior)
posterior = inversion(data)
print("the posterior is a", type(posterior).__name__)

# %% [markdown]
# Inside, the posterior comes from the normal equations, solved in the data
# space: $A Q A^* + R$, on 30 dimensions. Wrapping the solver as
# `ModelSpace(...)` solves them in the model space instead, $Q^{-1} + A^*
# R^{-1} A$ on 256. The posterior is the same; only the linear system
# changes. The model space suits a problem with few unknowns, or a
# surrogate. It needs the prior's precision, which for a smooth prior is
# huge at high wavenumbers, so on a function space the data space is the
# one to use; on 256 dimensions the model-space system is simply factorized.

# %%
sobolev = X.sobolev_measure(3.0, 0.3, pointwise_std=1.0)
in_data = LinearGaussianInversion(problem, sobolev)(data).expectation
in_model = LinearGaussianInversion(
    problem, sobolev, solver=ModelSpace(CholeskySolver())
)(data).expectation
agreement = X.norm(in_data - in_model) / X.norm(in_data)
print(f"the two agree to {agreement:.1e}")

# %% [markdown]
# A measure is something you can draw from, and a posterior draw is the
# best picture of what the data leave undetermined: it fits the data as well
# as the truth does, and is as rough as the prior says fields are. Each draw
# here costs one solve (the "randomise-then-optimise" construction), never a
# factorisation of the 256 × 256 posterior covariance.

# %%
mean = posterior.expectation
std = np.sqrt(X.pointwise_variance_at(posterior, X.grid_axes[0]))

_, ax = plotting.subplots(X, figsize=(8.0, 3.5))
plotting.plot_error_bounds(
    X, mean - 2.0 * std, mean + 2.0 * std, ax=ax, center=mean, label="mean, 2 sd"
)
for index in range(3):
    label = "posterior draws" if index == 0 else None
    plotting.plot(X, posterior.sample(rng=rng), ax=ax, color="C2", lw=0.7, label=label)
plotting.plot(X, truth, ax=ax, color="black", ls="--", lw=1.0, label="truth")
ax.plot(sensors, data, "o", color="C3", ms=4, label="sensor averages")
ax.set_xlabel("angle")
ax.legend(loc="upper right", fontsize=8)
ax.set_title("The posterior: its mean, its spread, and three draws")

# %% [markdown]
# ## Its density, relative to the prior
#
# Tutorial 5 refused a density on a function space, because a density
# against the grid's own volume measure depends on the grid. A density
# relative to *another measure* does not, and a posterior has one built in:
# against the prior it is $\exp(-\Phi)$, where $\Phi = \chi^2 / 2$ is the
# misfit. That is Bayes' theorem on a function space, and it is finite at
# every draw of the prior. The posterior carries the prior as its
# `reference` and $-\Phi$ as its `relative_log_density`.

# %%
print("its reference measure is the prior:", posterior.reference is prior)
log_density = posterior.relative_log_density
print(f"log density at the posterior mean: {log_density(mean):8.2f}")
print(f"log density at the truth:          {log_density(truth):8.2f}")
print(f"log density at a prior draw:       {log_density(prior.sample(rng=rng)):8.2f}")

# %% [markdown]
# Up to its normalising constant: the integral of $\exp(-\Phi)$ against the
# prior. That constant is the **evidence** for the model, $p(d)$ — how
# probable these data were before they were seen, under this prior and this
# noise model. It costs two log-determinants, computed only when asked for.
#
# The evidence is what compares models, which a posterior alone cannot do —
# a posterior is conditional on its prior being right. Here is the evidence
# for heat priors of six correlation lengths. The data were made with 0.3.

# %%
print(f"{'length':>7} {'log evidence':>13} {'misfit':>8} {'volume':>8}")
for length in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
    candidate = LinearGaussianInversion(
        problem, X.heat_measure(length, pointwise_std=1.0)
    )
    misfit, volume = candidate.evidence_terms(data)
    evidence = candidate.log_evidence(data)
    print(f"{length:7.2f} {evidence.value:13.2f} {misfit:8.2f} {volume.value:8.2f}")

# %% [markdown]
# The evidence peaks at the length scale the data were made with. Its two
# terms pull in opposite directions: the *misfit* grows as the prior gets
# smoother and can no longer bend to the data; the *volume* term penalises
# a prior so flexible that these data would have been unsurprising whatever
# they were. That penalty is Occam's razor, computed.
#
# ## A credible set for a property
#
# Usually the question is not "what is the field?" but something about it:
# the average temperature over two particular stretches of the ring, say.
# Push the posterior forward through the property operator — exactly, a
# Gaussian on $\mathbb{R}^2$ — and *harden* it into a set: the ellipse
# holding 95 per cent of its probability.
#
# An ellipsoid needs a precision, so the pushed covariance has to be
# invertible, and the library does not assume that: a push-forward keeps a
# definiteness claim only through an operator claiming to be onto (the
# trait `SURJECTIVE`, which two averages over different arcs earn), and
# the heat prior cannot make the claim in the first place, its spectrum
# underflowing to zero at high wavenumber. On $\mathbb{R}^2$ the honest
# route is the direct one: `CholeskySolver` factorizes the covariance, and
# a factorization that succeeds is the proof of definiteness.

# %%
T = X.geodesic_ball_average_operator(
    [np.array([1.0]), np.array([1.6])], 0.3
).with_traits(Traits.SURJECTIVE)
property_posterior = posterior.push_forward(T)
property_prior = prior.push_forward(T)
region = property_posterior.credible_set(level=0.95, solver=CholeskySolver())
before = property_prior.credible_set(level=0.95, solver=CholeskySolver())
print("the credible set is an", type(region).__name__, "on", region.domain)
print("the truth's averages lie inside it:", region.contains(T(truth)))

fig, ax = plt.subplots(figsize=(5.0, 4.5), layout="constrained")
plotting.plot_set(before, ax=ax, color="C7", alpha=0.25)
plotting.plot_set(region, ax=ax, color="C0", alpha=0.5)
ax.plot(*T(truth), "k*", ms=12, label="truth")
ax.set_xlabel("average over the arc at 1.0")
ax.set_ylabel("average over the arc at 1.6")
ax.set_title("95% credible sets: prior (grey) and posterior")
ax.legend()

# %% [markdown]
# Hardening is a named step, not something that happens quietly: the set
# keeps no memory of the distribution it came from, and on the function
# space itself it is refused (tutorial 5) — a credible set is meaningful
# for a handful of properties, not for the whole field.
#
# The posterior is the second kind of answer, a measure. Tutorial 8 gives
# the third, for when the prior and the noise are bounds rather than
# distributions and the answer is a set from the start.

# %%
plotting.show()
