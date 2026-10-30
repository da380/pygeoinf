# %% [markdown]
# # Tutorial 4: solvers
#
# Almost all the cost of an inverse problem is solving linear systems, and
# the systems are too large to write down. Tutorial 1's posterior came from a
# solve with the *data-space normal operator*
#
# $$N = A\, Q\, A^* + R ,$$
#
# where $A$ is the forward operator, $Q$ the prior covariance and $R$ the
# noise covariance. With four hundred observations of a field on a
# $64 \times 64$ grid, $Q$ acts on a 4096-dimensional space and $N$ on a
# 400-dimensional one. Neither is ever formed here: every solve below is a
# Krylov iteration that only *applies* $N$, one application of $A$, $Q$ and
# $A^*$ per step.
#
# This tutorial covers what those iterations need, what they promise, how to
# make them faster, and how to read a number that was estimated rather than
# computed.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import (
    CGSolver,
    DiagonalLinearOperator,
    GaussianMeasure,
    LinearForwardProblem,
    LinearGaussianInversion,
    NormalDiagonalPreconditioner,
    NormalOperator,
    SpectralPreconditioner,
    plotting,
)
from pygeoinf.numerics import random_trace
from pygeoinf.numerics.solvers import ConvergenceError
from pygeoinf.symmetric_space.torus import Sobolev

rng = np.random.default_rng(4)

# %% [markdown]
# ## The problem
#
# A field on the torus, a prior with a correlation length of 0.2, and point
# values at 400 random places with noise of standard deviation 0.05.
# `NormalOperator` assembles $A Q A^* + R$ as an expression: an operator
# that applies its factors in turn and remembers what they were.

# %%
X = Sobolev((64, 64), 2.0, 0.1)
prior = X.heat_measure(0.2, pointwise_std=1.0)
points = rng.uniform(0.0, 2.0 * np.pi, size=(400, 2))
A = X.point_evaluation_operator(points)
noise = GaussianMeasure.from_standard_deviation(A.codomain, 0.05)
truth = prior.sample(rng=rng)
data = A(truth) + noise.sample(rng=rng)

N = NormalOperator(A, prior, error=noise, formalism="data_space")
print(f"N acts on a space of dimension {N.domain.dim}")
print("its traits:", N.traits)

# %% [markdown]
# Notice the traits. $A Q A^*$ is positive semidefinite by its shape, $R$ is
# positive definite, and so their sum is positive definite — derived, not
# asserted. Conjugate gradients *requires* positive definiteness, checks for
# it, and so accepts $N$ without anyone vouching for it.
#
# ## A solve
#
# A solver is a factory: applied to an operator it returns that operator's
# inverse, itself an operator. `solve` returns the answer together with its
# diagnostics. The tolerance is relative to $\|b\|$, measured in the space's
# own norm, and the default is $10^{-8}$.

# %%
inverse = CGSolver()(N)
result = inverse.solve(data)
print(result)
print(f"relative residual {result.residual_norm / N.domain.norm(data):.1e}")

# %% [markdown]
# A few hundred applications of $N$. Forming $N$ as a matrix would have cost
# 400, one per column, and then a factorisation; here the two are comparable,
# because $N$ is small. The difference is that the Krylov route scales with
# what the *spectrum* of $N$ demands rather than with its dimension, and it
# never needs the 4096 × 4096 prior covariance either.
#
# The inverse composes with other operators like any operator. The posterior
# mean of tutorial 1 is $m_0 + Q A^* N^{-1} (d - A m_0)$, and with a zero prior
# mean that is one line, which is all `LinearGaussianInversion` does for it.

# %%
mean = (prior.covariance @ A.adjoint @ inverse)(data)
print(
    f"relative error of the posterior mean: {X.norm(mean - truth) / X.norm(truth):.2f}"
)

# %% [markdown]
# ## Strict by default
#
# Now the same field observed by two instruments: the first 200 values
# reported in metres, the other 200 in millimetres. Nothing about the
# physics has changed — only the units — and the noise of the second
# instrument is 50 mm. But the diagonal of $N$ now spans six orders of
# magnitude, and that is what an unpreconditioned Krylov method cannot abide.

# %%
units = np.where(np.arange(400) < 200, 1.0, 1000.0)
D = A.codomain
mixed_A = DiagonalLinearOperator(D, units) @ A
mixed_noise = GaussianMeasure.from_standard_deviations(D, 0.05 * units)
mixed_data = mixed_A(truth) + mixed_noise.sample(rng=rng)
mixed_N = NormalOperator(mixed_A, prior, error=mixed_noise, formalism="data_space")

try:
    CGSolver()(mixed_N).solve(mixed_data)
except ConvergenceError as error:
    print("ConvergenceError:", error)

# %% [markdown]
# The solver stopped at its default cap of twice the dimension and *raised*.
# An unconverged solve is a wrong answer, and returning one quietly is how a
# wrong answer reaches a figure. If you want it anyway — to look at it, or
# because a rough answer is enough — say so with `strict=False`, and you get
# a warning and a result whose `converged` is false.

# %%
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    rough = CGSolver(strict=False)(mixed_N).solve(mixed_data)
print(rough, "|", caught[0].message)

# %% [markdown]
# ## Preconditioning
#
# A preconditioner is an approximate inverse that the iteration applies at
# every step; the better it approximates $N^{-1}$, the fewer steps. It
# changes how long the solve takes, never what it converges to. Each kind
# targets a particular defect, and the defect here is *scaling*, which a
# diagonal preconditioner removes. `NormalDiagonalPreconditioner` reads the
# diagonal of $A Q A^* + R$ from its factors, one adjoint and one covariance
# application per datum, without forming $N$.
#
# A preconditioner given as a solver is *deferred*: it is built against
# whatever operator the outer solver is asked to invert.

# %%
diagonal = CGSolver().with_preconditioner(NormalDiagonalPreconditioner())
scaled = diagonal(mixed_N).solve(mixed_data)
print("with the diagonal preconditioner:", scaled)

# %% [markdown]
# Back to the original iteration count: the units were the whole of the
# difficulty. Every solver records its residual history, which is the first
# thing to look at when a solve misbehaves.

# %%
fig, ax = plt.subplots(figsize=(7.0, 3.2), layout="constrained")
for run, label in ((rough, "no preconditioner"), (scaled, "diagonal preconditioner")):
    history = np.asarray(run.history)
    ax.semilogy(history / history[0], label=label)
ax.axhline(1e-8, color="gray", lw=0.8, ls=":")
ax.set_xlabel("iteration")
ax.set_ylabel("relative residual")
ax.set_title("Mixed units: the same solve, scaled and unscaled")
ax.legend()

# %% [markdown]
# A different defect calls for a different preconditioner. On the original,
# well-scaled problem the iteration count is set by the *spectrum* of $N$:
# a smooth prior seen at many points makes $A Q A^*$ nearly low-rank, with a
# few large eigenvalues and a long tail. `SpectralPreconditioner` finds the
# leading eigenpairs by a randomised method and inverts them exactly. It is
# no use for the scaling problem above, and it is the right tool here.

# %%
spectral = CGSolver().with_preconditioner(SpectralPreconditioner(rank=50, rng=rng))
faster = spectral(N).solve(data)
print(f"well scaled, plain              : {result.iterations} iterations")
print(f"well scaled, spectral, rank 50  : {faster.iterations} iterations")
same = N.domain.norm(faster.solution - result.solution) / N.domain.norm(result.solution)
print(f"the two answers agree to {same:.0e}")

# %% [markdown]
# It is not free. Building it applied $N$ to 240 vectors — sixty random
# probes, sharpened by one power iteration — against about 150 iterations
# saved per solve, so it pays from the second solve onwards. In inference the
# same operator is solved with many times: the estimate at the end of this
# tutorial takes twenty solves. Preconditioners that exploit the structure of
# the problem more deeply — a cheap *surrogate* problem on a coarser model space — are the
# subject of example 28.
#
# ## Estimates
#
# Some quantities are too expensive to compute exactly and are estimated
# instead: a trace, a diagonal, a log-determinant. Anything estimated comes
# back as an `Estimate`, a value with its standard error, because a number
# reported without its error cannot be interpreted.
#
# The trace of the prior covariance is the expected squared norm of a prior
# draw. For this prior it is known exactly, so the estimator can be watched
# converging to it. Hutchinson's estimator averages $(Q z, z)$ over random
# $z$, and its error falls like $1/\sqrt{n}$.

# %%
exact = prior.nuclear_norm()
print(f"exact trace of Q: {exact:.3f}")
for samples in (10, 100, 1000):
    estimate = random_trace(prior.covariance, samples=samples, rng=rng)
    print(f"  {samples:5d} probes: {estimate}")

# %% [markdown]
# A tenfold increase in the work buys a factor of about three in the error.
# Rather than guess a sample count, ask for a relative standard error and
# let the estimator add probes until it gets there.

# %%
adaptive = random_trace(prior.covariance, samples=20, rtol=0.01, rng=rng)
print("to one per cent:", adaptive)

# %% [markdown]
# That is where the error bar earns its keep: on an operator where every
# probe costs a solve. The posterior covariance is $Q - Q A^* N^{-1} A Q$, so
# each application is one preconditioned solve with $N$. Its trace is the
# expected squared error of the posterior mean, and the fraction of the prior
# trace it removes is how much the data taught us, in one number.

# %%
problem = LinearForwardProblem(A, error=noise)
posterior = LinearGaussianInversion(problem, prior, solver=spectral)(data)
remaining = random_trace(posterior.covariance, samples=20, rng=rng)
fraction = 1.0 - remaining.value / exact
print("trace of the posterior covariance:", remaining)
print(
    f"the data removed {fraction:.1%} +/- {remaining.standard_error / exact:.1%}"
    " of the prior's expected squared error"
)
plotting.show()
