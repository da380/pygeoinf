# %% [markdown]
# # Tutorial 1: a first inversion
#
# A smooth function on a circle, seen at fifteen points through noise, and
# recovered — together with an honest statement of how well. No theory yet:
# every idea used here gets a tutorial of its own later. This is the shape
# of the whole thing, so that the rest of the course has somewhere to land.
#
# The tutorials are Python scripts divided into cells. Run one top to bottom
# with `python 01_first_inversion.py`, or open it in Jupyter or VS Code,
# which read the `# %%` markers as notebook cells.

# %%
import numpy as np

from pygeoinf import (
    GaussianMeasure,
    LinearForwardProblem,
    LinearGaussianInversion,
    plotting,
)
from pygeoinf.symmetric_space.circle import Sobolev

rng = np.random.default_rng(7)

# %% [markdown]
# ## Where the unknown lives
#
# The unknown is a function on the unit circle. The computer holds it as its
# values at 128 equally spaced points, but the space it belongs to is a
# *function space*: the Sobolev space $H^2$, whose norm measures a function
# and its first two derivatives together. The two numbers after the grid size
# are that order and a length scale, the distance below which the norm
# starts to penalise wiggles.
#
# Why not just $\mathbb{R}^{128}$? Because the answer should not depend on how
# many points the computer happened to use. Tutorial 2 is about this.

# %%
X = Sobolev(128, 2.0, 0.2)
print(f"model space: {type(X).__name__} on the circle, {X.dim} grid values")

# %% [markdown]
# ## What we believe before looking
#
# A prior: a Gaussian measure on that space. This one says the function is
# smooth on a scale of about 0.3 radians and typically of size one. Its draws
# are functions, and drawing one is how the synthetic truth below is made.

# %%
prior = X.heat_measure(0.3, pointwise_std=1.0)

# %% [markdown]
# ## How we look
#
# Fifteen observation points at random around the circle, each value
# measured with independent noise of standard deviation 0.1. The *forward
# operator* maps a function to the fifteen numbers an instrument would
# record; together with the noise it makes the *forward problem*.

# %%
points = np.sort(rng.uniform(0.0, 2.0 * np.pi, 15))
forward = X.point_evaluation_operator(points)
noise = GaussianMeasure.from_standard_deviation(forward.codomain, 0.1)
problem = LinearForwardProblem(forward, error=noise)

# A truth drawn from the prior, and data drawn given the truth.
truth, data = problem.synthetic_model_and_data(prior, rng=rng)
print(f"{problem.data_space.dim} data, noise standard deviation 0.1")

# %% [markdown]
# ## The answer
#
# Bayes' theorem for a linear problem with Gaussian prior and noise has a
# closed form, and this is it. The answer is not a function but a
# *probability measure* on functions: the posterior.

# %%
posterior = LinearGaussianInversion(problem, prior)(data)

# %% [markdown]
# ## How good is it?
#
# The posterior's mean is the usual "best estimate". Its spread is what makes
# the answer honest: here, the standard deviation of the function's value at
# every grid point, computed exactly. Where there are data it should be close
# to the noise level; between data it grows back towards the prior's one.

# %%
grid = X.grid_axes[0]
std = np.sqrt(X.pointwise_variance_at(posterior, grid))
mean = posterior.expectation

inside = np.abs(truth - mean) <= 2.0 * std
print(f"posterior standard deviation: {std.min():.3f} to {std.max():.3f}")
print(f"truth within two standard deviations at {inside.mean():.0%} of the grid")
error = X.norm(X.subtract(mean, truth)) / X.norm(truth)
print(f"relative error of the posterior mean, in the H2 norm: {error:.2f}")

# %%
_, ax = plotting.subplots(X, figsize=(8.0, 3.5))
plotting.plot_error_bounds(
    X, mean - 2.0 * std, mean + 2.0 * std, ax=ax, center=mean, label="posterior, 2 sd"
)
plotting.plot(X, truth, ax=ax, color="black", lw=1.0, ls="--", label="truth")
# The axes are ordinary matplotlib axes, so anything else can go on them.
ax.errorbar(points, data, yerr=0.2, fmt="o", color="C3", ms=4, label="data, 2 sd")
ax.set_xlabel("angle")
ax.set_title("Fifteen noisy values and what they say")
ax.legend(loc="upper right")
plotting.show()

# %% [markdown]
# ## What just happened, and where it is explained
#
# - **A space that knows its own inner product** held the unknown
#   (tutorial 2), and the evaluation operator's adjoint — which the inversion
#   used without being asked — was taken in that inner product (tutorial 3).
# - **The posterior was computed by a matrix-free iterative solve** in the
#   fifteen-dimensional data space; no 128 × 128 matrix was ever formed
#   (tutorial 4).
# - **The prior and the posterior are Gaussian measures on a function
#   space** (tutorial 5). The forward problem with its error measure is the
#   object every inference method in the library starts from (tutorial 6),
#   and the posterior is one of the three kinds of answer it gives: a point
#   (tutorial 6), a measure (tutorial 7) or a set (tutorial 8).
