# %% [markdown]
# # Tutorial 8: regularisation without priors
#
# Sometimes there is no prior distribution anyone would defend, and the
# question is simpler: which single model should be reported? This tutorial
# is about the deterministic answers — least squares, the Tikhonov family,
# the minimum-norm model and the discrepancy principle — what each assumes,
# what each costs, and where the Bayesian answer of tutorial 6 differs.
#
# The test case is the classic ill-posed problem: undoing a blur.

# %%
import numpy as np

from pygeoinf2 import (
    GaussianMeasure,
    LeastSquares,
    LinearForwardProblem,
    LinearGaussianInversion,
    MinimumNorm,
    plotting,
)
from pygeoinf2.symmetric_space.circle import Lebesgue, Sobolev

rng = np.random.default_rng(8)

# %% [markdown]
# ## The problem
#
# A signal on the circle — two bumps and a dip — is blurred by a heat
# kernel of width 0.15 and sampled at 64 equally spaced points with noise of
# standard deviation 0.01. Blurring kills fine detail exponentially fast,
# so undoing it amplifies noise exponentially fast: the textbook ill-posed
# problem.
#
# The unknown lives in $L^2$ to start with. Point evaluation is not defined
# on $L^2$ (tutorial 2), and the library refuses it there; but evaluation
# *after* a blur is perfectly well defined, because the blur makes the field
# smooth. The library cannot see that through the composition, so we tell
# it, with `unsafe=True`, knowing why.

# %%
X = Lebesgue(256)


def signal(t):
    first = np.exp(-(((t - 2.0) / 0.3) ** 2))
    second = 0.6 * np.exp(-(((t - 4.0) / 0.15) ** 2))
    return first + second - 0.4 * np.exp(-(((t - 5.2) / 0.5) ** 2))


truth = X.project_function(signal)
samples = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)


def forward_problem(space):
    """Blur, then sample: the same physics posed on a given model space."""
    blur = space.invariant_operator(lambda values: np.exp(-(0.15**2) * values))
    A = space.point_evaluation_operator(samples, unsafe=True) @ blur
    noise = GaussianMeasure.from_standard_deviation(A.codomain, 0.01)
    return LinearForwardProblem(A, error=noise)


problem = forward_problem(X)
data = problem.forward_operator(truth) + problem.error_measure.sample(rng=rng)
print(f"{problem.data_space.dim} data, model space of dimension {X.dim}")
print(f"misfit target at 95%: chi-squared below {problem.critical_chi_squared():.1f}")

# %% [markdown]
# ## Least squares, and why it is not enough
#
# Fitting the data as well as possible has no unique answer here — there are
# far more unknowns than data — and the natural tie-breaker is the smallest
# model that fits. Tikhonov's method makes that a single minimisation,
#
# $$\min_u\ \|A u - d\|^2_{R^{-1}} + t\, \|u\|^2 ,$$
#
# with the misfit weighted by the noise and a *damping* $t$ trading it
# against the size of the model. `LeastSquares(problem, damping=t)` is the
# estimator: a fixed affine map from data to model, so an operator, solved
# by a Krylov method on the 64-dimensional data-space normal operator.
# Watch the trade-off.

# %%
print(f"{'damping':>9} {'chi-squared':>12} {'error':>7}")
for damping in (1.0, 10.0, 100.0, 1000.0, 10000.0):
    model = LeastSquares(problem, damping=damping)(data)
    error = X.norm(model - truth) / X.norm(truth)
    print(f"{damping:9g} {problem.chi_squared(model, data):12.1f} {error:7.3f}")

# %% [markdown]
# With little damping the misfit is well *below* what the noise can explain:
# the model is fitting the noise, and its error is larger than the signal.
# With a lot, the misfit is far above it: the model is too small to explain
# the data. The error is least in between. But the error column needs the
# truth, which is exactly what is not known.
#
# ## The discrepancy principle
#
# The noise level *is* known, and it says how well a model should fit: a
# $\chi^2$ misfit about as large as the number of data, and below its 95 per
# cent point. The **discrepancy principle** takes the largest damping whose
# misfit is still that small — the smallest model the data cannot reject.
# The misfit grows monotonically with the damping, so finding it is a
# one-dimensional root search, one warm-started solve per probe.
#
# `MinimumNorm(problem)` is that estimator: data in, the smallest model
# that fits them out, with the confidence level as its one knob.

# %%
estimator = MinimumNorm(problem)
model = estimator(data)
search = estimator.discrepancy_search(data)
print(f"damping chosen: {search.argument:.0f}")
print(f"its misfit: {problem.chi_squared(model, data):.1f}")
print(f"its error:  {X.norm(model - truth) / X.norm(truth):.3f}")
print(
    f"cost: {search.evaluations} probes, {search.inner_iterations} Krylov iterations"
    " in all"
)

# %% [markdown]
# Close to the best damping in the table, without looking at the truth. The
# cost is a dozen or so solves, and they get cheaper as the search closes
# in, because each starts from the last.
#
# `MinimumNorm` is an operator from data to model, but *not* a linear one:
# two data vectors may need different dampings. It is differentiable, and
# it supplies its derivative, so it can sit inside a larger computation
# that needs one. For the affine estimator the data chose, ask
# `for_data`: it is the `LeastSquares` at the damping found, and applied
# to the same data it gives the same model.

# %%
chosen = estimator.for_data(data)
same = X.norm(chosen(data) - model) / X.norm(model)
print(f"LeastSquares at damping {chosen.damping:.0f} agrees to {same:.1e}")

# %% [markdown]
# ## The norm is the regulariser
#
# "Smallest" depends on the norm. In $L^2$ it means least energy; in a
# Sobolev space it means least energy *and* least roughness. Posing the same
# physics on $H^1$ and $H^2$ and letting the discrepancy principle choose
# the damping gives three different answers, each the smallest fitting model
# in its own sense. Errors are compared in $L^2$ so that they are comparable.

# %%
answers = {"L2": model}
for name, space in (("H1", Sobolev(256, 1.0, 0.2)), ("H2", Sobolev(256, 2.0, 0.2))):
    posed = forward_problem(space)
    answers[name] = MinimumNorm(posed)(data)
for name, answer in answers.items():
    print(f"{name}: error in L2 {X.norm(answer - truth) / X.norm(truth):.3f}")

# %%
_, ax = plotting.subplots(X, figsize=(8.0, 3.5))
plotting.plot(X, truth, ax=ax, color="black", lw=1.5, label="truth")
for name, answer in answers.items():
    plotting.plot(X, answer, ax=ax, lw=1.0, label=f"discrepancy principle, {name}")
ax.plot(samples, data, ".", color="C7", ms=4, label="blurred, sampled data")
ax.set_xlabel("angle")
ax.legend(fontsize=8)
ax.set_title("Deblurring: the smallest model that fits, in three norms")

# %% [markdown]
# All three round off the narrow bump at 4, which is sharper than anything
# the blurred data can resolve, and the smoother norms round it most: the
# norm is an assumption about what kind of function the truth is, and a
# wrong assumption costs accuracy. Choosing it is a modelling decision, and
# it should be made as one: on what is known about the truth, not on what
# makes the algebra convenient.
#
# ## An algebraic coincidence, not a hidden prior
#
# In finite dimensions Tikhonov's minimiser is also the posterior mean of a
# Gaussian problem whose prior covariance is $I / t$: the same linear
# system, solved for the same vector. The library refuses to build that
# prior on a function space, and the refusal is the lesson: the identity is
# not trace class on a function space, so a "measure" with that covariance
# has no draws in the space, and the truncation to 256 coefficients only
# hides it. Tikhonov damping is a norm, which is legitimate on any space;
# reading it as a prior is not. The agreement below is produced with the
# escape hatch a test uses, to show the algebra and nothing more.

# %%
damping = chosen.damping
try:
    GaussianMeasure.from_standard_deviation(X, 1.0 / np.sqrt(damping))
except TypeError as refusal:
    print(f"refused: {str(refusal)[:90]}...")
bayes = LinearGaussianInversion(
    problem,
    GaussianMeasure.from_standard_deviation(X, 1.0 / np.sqrt(damping), unsafe=True),
)(data).expectation
agreement = X.norm(bayes - model) / X.norm(model)
print(f"Tikhonov and the Gaussian posterior mean agree to {agreement:.1e}")

# %% [markdown]
# The coincidence is one of formulas, not of questions. Regularisation asks
# for the smallest model that fits the data; it names no probability and
# claims nothing about what is likely, and the number it returns is a
# complete answer to the question it asked. Reading the damping as a prior
# adds a claim that was never made — and here a strange one: a covariance
# proportional to the identity on $L^2$ is *white noise*, which tutorial 5
# showed is not a measure on functions at all. Its draws have no values,
# and refining the grid makes them rougher without limit. Nobody who
# regularises a deblurring problem believes that, and they need not: you
# do not have to be a Bayesian to regularise, and calling the method
# Bayesian would not make its answer any better. What the two share is a
# matrix. The library keeps them as different classes for that reason, and
# lets the caller say which question they are asking.
#
# What the deterministic estimators do *not* give is an uncertainty. For
# that, a prior you believe (tutorial 6), or a bound you believe
# (tutorial 7). And when part of what you know is exact — a total, a value
# at a point, a boundary condition — `ConstrainedLeastSquares` and
# `ConstrainedMinimumNorm` impose it exactly rather than as a penalty;
# example 18 shows the subspaces they are built on.

# %%
plotting.show()
