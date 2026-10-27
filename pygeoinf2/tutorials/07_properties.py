# %% [markdown]
# # Tutorial 7: properties, not models
#
# Thirty averages cannot determine a function. Whatever the inversion
# returns, most of the unknown is decided by the prior, not the data — the
# posterior draws of tutorial 6 agreed near the sensors and wandered
# between them. So the honest question is often not "what is the field?"
# but "what do the data say about *this* property of it?": an average over a
# region, a value at depth, a total. Some properties the data constrain well,
# some hardly at all, and the useful answer says which.
#
# This tutorial asks about two averages of tutorial 6's ring temperature,
# and gets three kinds of answer. Which kind is decided by what is assumed
# about the unknown:
#
# | assumed | answer | method |
# |---|---|---|
# | a probability measure | a probability measure | the Bayesian posterior, pushed forward (tutorial 6) |
# | a norm bound, used linearly | a point, with an error bar | Backus-Gilbert |
# | a norm bound, used exactly | a set | Backus-Gilbert-Parker |

# %%
import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import (
    Ball,
    BackusGilbert,
    BackusGilbertParker,
    CholeskySolver,
    GaussianMeasure,
    LinearForwardProblem,
    plotting,
)
from pygeoinf2.symmetric_space.circle import Lebesgue, Sobolev

rng = np.random.default_rng(6)

# %% [markdown]
# ## The problem, and the property
#
# Thirty sensors averaging over short arcs, as in tutorial 6. The property
# operator $T$ maps a field to two numbers: its averages over the arcs of
# half-width 0.3 centred at 1.0 and 1.6. The second sits where the sensors
# are sparse, which will matter.

# %%
X = Sobolev(256, 2.0, 0.5)
sensors = np.sort(rng.uniform(0.0, 2.0 * np.pi, 30))
A = X.geodesic_ball_average_operator([np.array([s]) for s in sensors], 0.05)
noise = GaussianMeasure.from_standard_deviation(A.codomain, 0.1)
problem = LinearForwardProblem(A, error=noise)
truth, data = problem.synthetic_model_and_data(
    X.heat_measure(0.3, pointwise_std=1.0), rng=rng
)

T = X.geodesic_ball_average_operator([np.array([1.0]), np.array([1.6])], 0.3)
P = T.codomain
print("the true averages:", T(truth).round(3))
near = [int(np.sum(np.abs(sensors - c) < 0.5)) for c in (1.0, 1.6)]
print("sensors within 0.5 of each arc's centre:", near)

# %% [markdown]
# ## A prior that is a bound
#
# Here the prior is not a distribution but a statement of the form "the
# field is no larger than this": $\|m\| \le M$ in the model space's norm.
# That makes the *choice of norm* the prior. In $H^2$ with a length scale of
# 0.5, a bounded norm rules out fields that are large, and also fields that
# wiggle on scales much below 0.5: the bound is a smoothness assumption as
# much as a size one. The radius here is half as large again as the truth's
# norm, which in practice would be an estimate from physics.
#
# The noise needs to become a bound too. The estimators *harden* the
# Gaussian error measure into the ball that carries 95 per cent of its
# probability, which is a named, lossy step: the ball remembers nothing of
# the distribution beyond its radius.

# %%
radius = 1.5 * X.norm(truth)
prior = Ball(X, radius=radius)

# What the bound alone says, before any data: the largest (T* q, m) over
# the ball is M ||T* q||, in each direction q.
prior_only = [radius * X.norm(T.adjoint(P.basis_vector(i))) for i in range(2)]
print("the prior bound alone allows +/-", np.round(prior_only, 2))

# %% [markdown]
# ## Backus-Gilbert: an estimate and an error bar
#
# The Backus-Gilbert estimate is a fixed linear combination of the data,
# $\hat p = X d$, with $X$ chosen to make the guaranteed error small. Its
# error has two parts, and the split is the point of the method:
#
# - **resolution**: $X A$ is not $T$. The estimate averages the field with
#   the *resolution kernels* of $X A$ rather than with the kernels you asked
#   for, and the difference is bounded by the prior;
# - **noise**: $X$ applied to the noise, bounded by the noise ball.
#
# More data improve the first; better data the second. A single error bar
# cannot tell you which you need.

# %%
estimator = BackusGilbert(problem, T, prior, solver=CholeskySolver())
certified = estimator(data)
estimate, resolution_part, noise_part = certified.error_bars()
for i in range(2):
    print(
        f"arc {i}: {estimate[i]:+.3f} +/- {resolution_part[i] + noise_part[i]:.3f}"
        f"  (resolution {resolution_part[i]:.3f}, noise {noise_part[i]:.3f})"
        f"   truth {T(truth)[i]:+.3f}"
    )

# %% [markdown]
# The resolution kernels are worth looking at. A property of the field
# that is an average, $\int K(x) m(x)\,dx$, is described by its kernel $K$;
# the target kernels are boxes over the two arcs, and the estimate's are
# whatever combinations of the sensors' footprints come closest. On the
# circle the Fourier coefficients of a kernel are the rows of the operator's
# component matrix, so reading them off is one line.

# %%
L2 = Lebesgue(256)
grid = L2.grid_axes[0]
target_rows = T.matrix(form="components")
achieved_rows = estimator.resolution.matrix(form="components")

fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.5), sharex=True, layout="constrained")
for i, ax in enumerate(axes):
    ax.plot(grid, L2.from_components(target_rows[i]), label="asked for")
    ax.plot(grid, L2.from_components(achieved_rows[i]), label="achieved")
    ax.plot(sensors, np.zeros_like(sensors), "|", color="C3", ms=10, label="sensors")
    ax.set_xlim(0.0, 3.5)
    ax.set_title(f"arc {i}", fontsize=10)
axes[0].legend(fontsize=8)
axes[-1].set_xlabel("angle")
fig.suptitle("Resolution kernels: what the estimate actually averages")

# %% [markdown]
# Where sensors crowd the arc the achieved kernel is a fair copy of the box;
# where they are sparse it can only approximate it with what it has, and
# the resolution error is correspondingly large. SOLA is this estimator
# with the trade-off between the two terms chosen by hand and no prior
# ball named: it reports the same kernels but cannot certify the
# resolution error, since there is no bound for it to be certified against.
#
# ## Backus-Gilbert-Parker: the set itself
#
# Backus-Gilbert's bound is valid, but it is the bound of one linear
# estimator, and nothing requires the truth's best description to be
# linear. Ask instead for the *exact* answer: the set of all values $T m$
# over all fields $m$ in the prior ball that fit the data to within the
# noise ball. It is convex, and it is what `BackusGilbertParker` returns —
# not a number, but an object that answers questions.

# %%
feasible = BackusGilbertParker(problem, T, prior)(data)
print(feasible)
print("is it empty?", feasible.is_empty())
for i in range(2):
    q = P.basis_vector(i)
    upper = feasible.support(q)
    lower = -feasible.support(P.scale(-1.0, q))
    inner, outer = feasible.support_bracket(q)
    print(
        f"arc {i}: between {lower:+.3f} and {upper:+.3f}   truth {T(truth)[i]:+.3f}"
        f"   (bracket width {outer - inner:.1e})"
    )

# %% [markdown]
# Each bound is a *support value*: the largest $(q, p)$ over the set in a
# direction $q$. The value reported is an outer bound, and it comes with a
# bracket: the inner value is attained by a model that lies in both sets,
# so the true support lies between the two, and the width says how far the
# algorithm got. Support values in many directions bound the set from
# outside, and the set can also be asked about single points directly —
# whether a value is consistent with the data and the prior at all — which
# bounds it from inside. The two characterisations together are what make
# it a usable answer.

# %%
print("the true averages are admissible:", feasible.contains(T(truth)))
print("so is the estimate:", feasible.contains(estimate))
print("(0, 3) is not:", feasible.contains(P.from_components(np.array([0.0, 3.0]))))

# %% [markdown]
# Both sets can also be summarised by one ellipsoid each. For the
# Backus-Gilbert set it is the smallest ellipsoid of a family containing
# the sum of its two parts; for the feasible set it is the image of the
# weighted constraint $\theta\,\text{prior} + (1 - \theta)\,\text{misfit}
# \le 1$, which every feasible model satisfies, at the weight that
# minimises the volume. Both are certified outer bounds, and the second is
# tighter because the data enter its constraint.

# %%
certified_ellipsoid = certified.outer_ellipsoid(criterion="volume")
feasible_ellipsoid = feasible.outer_ellipsoid(criterion="volume")
print(
    "ellipsoid half-widths along arc 0:",
    f"Backus-Gilbert {certified_ellipsoid.support_function()(P.basis_vector(0)) - estimate[0]:.3f},",
    f"feasible {feasible_ellipsoid.support_function()(P.basis_vector(0)) - P.to_components(feasible_ellipsoid.center)[0]:.3f}",
)

# %%
fig, ax = plt.subplots(figsize=(5.5, 5.0), layout="constrained")
plotting.plot_set(certified, ax=ax, color="C1", alpha=0.3)
plotting.plot_set(feasible_ellipsoid, ax=ax, color="C0", alpha=0.15)
plotting.plot_set(feasible, ax=ax, color="C0", alpha=0.6)
angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
hull = feasible.inner_hull(
    [P.from_components(np.array([np.cos(a), np.sin(a)])) for a in angles]
)
plotting.plot_set(
    hull, ax=ax, bounds=(ax.get_xlim(), ax.get_ylim()), color="C2", alpha=0.6
)
ax.plot(*T(truth), "k*", ms=12, label="truth")
ax.plot(*estimate, "o", color="C1", label="Backus-Gilbert estimate")
ax.set_xlabel("average over arc 0")
ax.set_ylabel("average over arc 1")
ax.set_title(
    "The feasible set (blue) inside its outer ellipsoid (pale blue) and\n"
    "the Backus-Gilbert set (orange), with its inner hull (green)"
)
ax.legend(loc="lower right", fontsize=8)

# %% [markdown]
# The feasible set is the exact answer to the question the bounds pose;
# Backus-Gilbert's set contains it, as it must, and costs less — one linear
# solve per property, where each support value of the feasible set is a
# small optimisation. Both are far inside what the prior bound alone allows
# (±2.8 on each axis), and both are much longer along arc 1, which the data
# barely see.
#
# ## Three answers to three questions
#
# Tutorial 6's credible set and this feasible set are not competing
# estimates of one thing. The credible set says where the property lies
# *with 95 per cent probability*, given a prior distribution you believe;
# the feasible set says where it lies *for certain*, given only a bound you
# believe. When the prior is a distribution the first is the stronger
# statement; when no one can defend a distribution, the second is the one
# that can be defended. The library keeps the three forms of answer — a
# point, a measure, a set — as different types, because confusing them is a
# common way to report an uncertainty that was never computed.
plotting.show()
