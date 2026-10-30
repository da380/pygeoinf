# %% [markdown]
# # Tutorial 9: nonlinear problems
#
# Everything so far has had a linear forward operator, and linearity does a
# great deal of work: the posterior is Gaussian, its mean is a linear
# function of the data, its covariance does not depend on the data at all.
# None of that survives a nonlinear forward map. What survives is the
# recipe — the log-posterior is still a misfit plus a prior term — and so
# the answer becomes an optimisation, followed by a Gaussian fitted where the
# optimisation ends.
#
# This tutorial builds a nonlinear operator, finds the most probable model,
# puts the Laplace approximation on it, and shows what happens when the
# forward model cannot be evaluated at a model the optimiser tries.

# %%
import numpy as np

from pygeoinf import (
    EvaluationError,
    ForwardProblem,
    GaussianMeasure,
    LaplaceApproximation,
    Linearization,
    MaximumAPosteriori,
    Operator,
    plotting,
)
from pygeoinf.numerics.optimization import NewtonCG
from pygeoinf.symmetric_space.circle import Sobolev
from pygeoinf.testing import check_derivative

rng = np.random.default_rng(9)

# %% [markdown]
# ## A nonlinear operator
#
# The unknown is the *logarithm* of a concentration around the ring, $m$, so
# the concentration $c = e^m$ is positive whatever $m$ is: a positivity
# constraint written into the parameterisation. Forty sensors report the
# concentration averaged over their footprints. The forward map is
# $F(m) = A\, e^m$, with $A$ the averaging operator — linear in $c$,
# nonlinear in $m$.
#
# Its derivative at $m$ is $h \mapsto A(e^m h)$: multiplication by the
# concentration, then averaging. A forward code usually produces the value
# and what it needs for the derivative in one run, and the library's
# protocol for that is `linearize`, which returns both together: `F.at(m)`
# calls it once. `F(m)` alone costs only the value, which is all a
# backtracking line search wants.
#
# One more thing a real forward does: fail. Here, a log-concentration above
# 20 stands for a model the code cannot run — an unphysical input, a solver
# that diverges. The protocol is to raise `EvaluationError`, which says
# "this model cannot be evaluated", as opposed to a bug.

# %%
X = Sobolev(256, 2.0, 0.2)
sensors = np.sort(rng.uniform(0.0, 2.0 * np.pi, 40))
A = X.geodesic_ball_average_operator([np.array([s]) for s in sensors], 0.05)

calls = {"value": 0, "linearize": 0, "failed": 0}


def concentration(m):
    if np.max(m) > 20.0:
        calls["failed"] += 1
        raise EvaluationError("the concentration is out of the code's range")
    return np.exp(m)


def value(m):
    calls["value"] += 1
    return A(concentration(m))


def linearize(m):
    calls["linearize"] += 1
    c = concentration(m)
    return Linearization(m, A(c), A @ X.multiplication_operator(c))


F = Operator.from_callables(X, A.codomain, value, linearize=linearize)

# %%
m = X.random(rng=rng)
local = F.at(m)
derivative = local.derivative
print(f"at(m): a value of shape {local.value.shape}, and a derivative")
print(f"  from {derivative.domain} to {derivative.codomain}")
report = check_derivative(F, m, rng=rng)
print("the derivative matches finite differences:", report)

# %% [markdown]
# ## The prior, and what it does to the optimiser
#
# The prior is a Gaussian measure on the model space, so its covariance
# must be trace class there: a Sobolev measure of order above half the
# dimension, here order 1 on the circle's $H^2$, calibrated as always by
# the pointwise standard deviation. Its draws are then functions of the
# space, smoother than $H^2$ asks. Equal variance in every mode would not
# do: that is the identity covariance, which is not trace class, so its
# "draws" are not in the space, and the library refuses it.
#
# Tutorial 3 showed that an optimiser converges at a rate set by the
# conditioning of the problem *in the space's own inner product*. For a MAP
# problem the curvature is the prior's precision plus the data's,
# $P + J^* R^{-1} J$, and the precision of a Sobolev prior is badly
# conditioned: the prior is confident about fine scales, so its precision
# grows without bound up the spectrum. The cure is a preconditioner, and
# the prior hands one over: its covariance is the inverse of the
# precision's part of the curvature, and leaves the optimiser only the
# data's spectrum to work through.

# %%
prior = X.sobolev_measure(1.0, 0.2, pointwise_std=0.8)
truth = prior.sample(rng=rng)
noise = GaussianMeasure.from_standard_deviation(A.codomain, 0.05)
problem = ForwardProblem(F, error=noise)
data = F(truth) + noise.sample(rng=rng)
print(
    f"concentrations in the truth: {np.exp(truth).min():.2f} to {np.exp(truth).max():.2f}"
)

# %% [markdown]
# `ForwardProblem` rather than `LinearForwardProblem`: the same object with
# the linearity claim dropped, so the methods that rely on it are not
# offered.
#
# ## The most probable model
#
# The posterior is still prior times likelihood, so its negative logarithm
# is still the misfit plus the prior's quadratic form. `MaximumAPosteriori`
# minimises it in the model space's inner product, starting from the prior
# mean. It takes L-BFGS by default, which needs only gradients; here it is
# handed Newton-CG with the prior's covariance as the preconditioner of its
# inner solve. The forward operator supplies its derivative but not its
# second derivative, so the Hessian is the Gauss-Newton one,
# $P + J^* R^{-1} J$, which `gauss_newton=True` asks for. It is a point
# estimator: called on the data it returns the mode, and `solve` returns
# the optimiser's own record of the search, which is worth reading before
# the mode is trusted.

# %%
for key in calls:
    calls[key] = 0
estimator = MaximumAPosteriori(
    problem,
    prior,
    gauss_newton=True,
    optimizer=NewtonCG(preconditioner=prior.covariance),
)
result = estimator.solve(data)
mode = result.minimizer
print(result)
print(
    f"forward runs: {calls['linearize'] + calls['value']},"
    f" of which {calls['failed']} could not be evaluated"
)
# Measured in L2, as in tutorial 6: the H2 norm weights the finest scales,
# which no data can see, and would report mostly what was never observable.
rms_error = np.sqrt(np.mean((mode - truth) ** 2))
print(f"rms error of the MAP log-concentration: {rms_error:.2f} (prior: 0.8)")

# %% [markdown]
# The preconditioned Newton steps never left the range the code can
# evaluate. L-BFGS, the default, has no curvature to go on at first: its
# first trial step along the gradient is sized from the value and the
# gradient alone, and the line search takes it from there.

# %%
for key in calls:
    calls[key] = 0
by_lbfgs = MaximumAPosteriori(problem, prior).solve(data)
print(by_lbfgs)
print(
    f"forward runs: {calls['linearize'] + calls['value']},"
    f" of which {calls['failed']} could not be evaluated"
)
agreement = X.norm(X.subtract(by_lbfgs.minimizer, mode)) / X.norm(mode)
print(f"the two modes agree to {agreement:.1e}")

# %% [markdown]
# It reaches the same mode, at many more forward runs, and on this problem
# none of its trial models fell out of range either. When one does — a
# step that lands where the code fails — the line search treats the
# `EvaluationError` as a step too far, backs off and tries a shorter one,
# and the search carries on. Had the failure come where there was nowhere
# to retreat to, the optimiser would have returned its last good state
# with the failure recorded in `result.failure`, rather than losing the run
# to an exception. Any *other* exception still propagates: a bug is not
# something to step around. Example 12 constructs both cases deliberately.

# %%
print("recorded failure:", by_lbfgs.failure)

# %% [markdown]
# The search is *mesh-independent*: refine the grid and the number of
# iterations stays put, because the problem the optimiser sees, in the
# space's inner product and with the prior's covariance as preconditioner,
# does not change. L-BFGS needs roughly the same count at every grid as
# well, the inner product doing that work; but it has no preconditioner to
# take, so it pays for the precision's conditioning in iterations, more than
# ten times Newton's here.

# %%
print(f"{'grid':>5} {'Newton-CG':>10} {'L-BFGS':>8}")
for n in (128, 256, 512):
    S = Sobolev(n, 2.0, 0.2)
    B = S.geodesic_ball_average_operator([np.array([s]) for s in sensors], 0.05)

    def linearize_on(m, S=S, B=B):
        c = np.exp(np.minimum(m, 20.0))
        return Linearization(m, B(c), B @ S.multiplication_operator(c))

    G = Operator.from_callables(
        S,
        B.codomain,
        lambda m, B=B: B(np.exp(np.minimum(m, 20.0))),
        linearize=linearize_on,
    )
    refined_prior = S.sobolev_measure(1.0, 0.2, pointwise_std=0.8)
    refined_problem = ForwardProblem(G, error=noise)
    newton = MaximumAPosteriori(
        refined_problem,
        refined_prior,
        gauss_newton=True,
        optimizer=NewtonCG(preconditioner=refined_prior.covariance),
    ).solve(data)
    lbfgs = MaximumAPosteriori(refined_problem, refined_prior).solve(data)
    print(f"{n:5d} {newton.iterations:10d} {lbfgs.iterations:8d}")

# %% [markdown]
# ## The Laplace approximation
#
# Expand the log-posterior to second order about the mode and you get a
# Gaussian: centred on the MAP model, with covariance the inverse of the
# normal operator of the problem *linearised at the mode*,
# $(P + J^* R^{-1} J)^{-1}$. It is the linear machinery of tutorial 7,
# evaluated at one point instead of assumed everywhere. `LaplaceApproximation`
# is a *measure* estimator built on the point estimator: called on the data
# it finds the mode and returns that Gaussian, and `at` places the Gaussian
# on a mode already found, which saves searching twice.
#
# Its covariance is applied by an iterative solve, so its variance at a point
# costs one solve. Here it is at 32 points around the ring.

# %%
laplace = LaplaceApproximation(estimator).at(mode, data)
where = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
spread = np.sqrt(X.pointwise_variance_at(laplace, where))
at_mode = X.point_evaluation_operator(where)(mode)
at_truth = X.point_evaluation_operator(where)(truth)
inside = np.abs(at_truth - at_mode) <= 2.0 * spread
print(
    f"truth within two Laplace standard deviations at {inside.mean():.0%} of the points"
)

# %%
_, ax = plotting.subplots(X, figsize=(8.0, 3.5))
plotting.plot(X, np.exp(truth), ax=ax, color="black", ls="--", lw=1.0, label="truth")
plotting.plot(X, np.exp(mode), ax=ax, color="C0", label="MAP model")
low, high = np.exp(at_mode - 2.0 * spread), np.exp(at_mode + 2.0 * spread)
ax.errorbar(
    where,
    np.exp(at_mode),
    yerr=[np.exp(at_mode) - low, high - np.exp(at_mode)],
    fmt="none",
    color="C0",
    alpha=0.6,
    label="Laplace, 2 sd",
)
ax.plot(sensors, np.zeros_like(sensors), "|", color="C3", ms=12, label="sensors")
ax.set_xlabel("angle")
ax.set_ylabel("concentration")
ax.legend(fontsize=8)
ax.set_title("The concentration, its most probable value, and the Laplace spread")

# %% [markdown]
# The error bars are asymmetric because they are symmetric in the
# *log*-concentration: the Gaussian lives on $m$, and $e^m$ bends it.
#
# What the Laplace approximation is, and is not:
#
# - it is centred on the **mode**, not the mean. For a linear problem the two
#   coincide; here they do not, and the mode is what an optimiser finds;
# - its covariance describes the posterior's curvature **at the mode** and
#   nothing about its shape elsewhere. Where the forward map bends a lot over
#   the posterior's width, the true posterior is not Gaussian and the
#   approximation misleads. Example 31 draws both on a problem small enough
#   to compare them exactly;
# - the curvature is the Gauss-Newton one, which drops the forward map's
#   second derivative. That term vanishes where the fit is perfect and is
#   small where it is good.
#
# One thing it carries is exact, though: like the linear posterior of
# tutorial 7, `laplace` is a density relative to the prior, and its
# `relative_log_density` is the *true* posterior's, $-\Phi$, not the
# Gaussian's. The Gaussian is the approximation; the density it carries is
# not. That pair — an exact target and a good Gaussian reference — is what a
# sampler needs to correct the approximation.

# %%
print("its reference is the prior:", laplace.reference is prior)
exact = laplace.relative_log_density(mode)
print(f"exact log density at the mode, relative to the prior: {exact:.2f}")
plotting.show()
