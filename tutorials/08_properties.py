# %% [markdown]
# # Tutorial 8: properties, not models
#
# Thirty averages cannot determine a function. Whatever the inversion
# returns, most of the unknown is decided by the prior, not the data — the
# posterior draws of tutorial 7 agreed near the sensors and wandered
# between them. So the honest question is often not "what is the field?"
# but "what do the data say about *this* property of it?": an average over a
# region, a value at depth, a total. Some properties the data constrain well,
# some hardly at all, and the useful answer says which.
#
# This tutorial asks about two averages of tutorial 7's ring temperature.
# Each of the three kinds of answer has a form for a property, and which
# kind comes back is decided by what is assumed about the unknown:
#
# | assumed | answer | method |
# |---|---|---|
# | nothing | a point, with its resolution and propagated error | Backus-Gilbert |
# | a probability measure | a probability measure | the Bayesian posterior, pushed forward (tutorial 7) |
# | a norm bound | a set | Backus-Gilbert-Parker |
#
# Tutorial 7 pushed its posterior forward through a property operator.
# This one does the other two, and spends most of its time on the third.

# %%
import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import (
    BackusGilbert,
    BackusGilbertParker,
    Ball,
    CholeskySolver,
    GaussianMeasure,
    LinearCertificate,
    LinearForwardProblem,
    plotting,
)
from pygeoinf.symmetric_space.circle import Lebesgue, Sobolev

rng = np.random.default_rng(6)

# %% [markdown]
# ## The problem, and the property
#
# Thirty sensors averaging over short arcs, as in tutorial 7. The property
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
# ## Backus-Gilbert: an estimate, and what it leaves out
#
# The Backus-Gilbert estimate is a fixed linear combination of the data,
# $\hat p = X d$, with $X$ chosen so that the *resolution* $X A$ comes as
# close to $T$ as the noise allows: $X = T A^* (A A^* + \alpha R)^{-1}$,
# with $R$ the noise covariance and $\alpha$ the trade-off. It is a point
# estimator, and like every point estimator here it carries its error
# budget as two operators rather than as a number:
#
# - **resolution**: $X A$ is not $T$. The estimate averages the field with
#   the *resolution kernels* of $X A$ rather than with the kernels you asked
#   for, and the difference $(T - XA)\,m$ depends on the unknown field;
# - **propagated error**: $X$ applied to the noise, whose covariance
#   $X R X^*$ is a statement about the noise alone.
#
# The second is a standard deviation you can quote. The first is not: it is
# an operator applied to a model nobody knows, and bounding it takes a
# statement about the model, which is what the prior bound is for.

# %%
estimator = BackusGilbert(problem, T, solver=CholeskySolver())
estimate = estimator(data)
propagated = estimator.propagated_covariance()
for i in range(2):
    q = P.basis_vector(i)
    print(
        f"arc {i}: {estimate[i]:+.3f}, propagated standard deviation"
        f" {np.sqrt(P.inner_product(propagated(q), q)):.3f}   truth {T(truth)[i]:+.3f}"
        f"   bias on the truth {estimator.bias(truth)[i]:+.3f}"
    )

# %% [markdown]
# Here the bias on the truth is small beside the propagated standard
# deviation. That is a fact about this truth, which nobody has: for another
# field the same estimator's bias is whatever the gap between the achieved
# and the target kernels makes of it, and it can dominate (tutorial 14
# meets such a case). That is why the propagated error is not "the
# uncertainty", and why the bias cannot be quoted. The resolution kernels
# say where it comes from. A
# property of the field that is an average, $\int K(x) m(x)\,dx$, is
# described by its kernel $K$; the target kernels are boxes over the two
# arcs, and the estimate's are whatever combinations of the sensors'
# footprints come closest. On the circle the Fourier coefficients of a
# kernel are the rows of the operator's component matrix, so reading them
# off is one line.

# %%
L2 = Lebesgue(256)
grid = L2.grid_axes[0]
target_rows = T.matrix(form="components")
achieved_rows = estimator.resolution().matrix(form="components")

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
# the unresolved part is correspondingly large. Backus and Gilbert's own
# treatment already reached for the missing ingredient, a bound on the
# norm of the field; with it the unresolved part has a largest value, and
# the estimate gets a certified width. That is the set of the next section
# seen through one linear estimator, and it is where we go now.
#
# ## Backus-Gilbert-Parker: the set itself
#
# Ask for the *exact* answer: the set of all values $T m$ over all fields
# $m$ in the prior ball that fit the data to within the noise ball. It is
# convex, and it is what `BackusGilbertParker` returns — not a number, but
# an object that answers questions.

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
# ## The linear certificate: Backus-Gilbert's estimate with its bound
#
# Every support value above was a small optimisation. There is a cheaper
# bound, and it is Backus-Gilbert's estimate completed by the prior: for
# *any* fixed weighting $X$ of the data, every feasible field satisfies
# $T m = X d + (T - XA)\,m - X e$, so its property lies in the sum of the
# estimate, the reach of $T - XA$ over the prior ball, and the reach of $X$
# over the noise ball. That is a valid outer bound whatever $X$ is — weak
# duality — and it is sharp only by luck. The feasible set offers it as the
# `linear_certificate` route, never chosen on its own, and splits each
# width into its two causes: more data narrow the resolution term, better
# data the noise term, and a single error bar could not say which is wanted.

# %%
route = LinearCertificate(operator=estimator.operator)
for i in range(2):
    q = P.basis_vector(i)
    center, resolution_part, noise_part = route.terms(feasible, q)
    print(
        f"arc {i}: {center:+.3f} +/- {resolution_part + noise_part:.3f}"
        f"  (resolution {resolution_part:.3f}, noise {noise_part:.3f})"
        f"   exact upper bound {feasible.support(q):+.3f}"
        f" against the certificate's {feasible.support(q, via=route):+.3f}"
    )

# %% [markdown]
# Both descriptions can also be summarised by one ellipsoid each. For the
# certificate it is the smallest ellipsoid of a family containing the sum
# of its two parts; for the feasible set it is the image of the weighted
# constraint $\theta\,\text{prior} + (1 - \theta)\,\text{misfit} \le 1$,
# which every feasible model satisfies, at the weight that minimises the
# volume. Both are certified outer bounds, and the second is tighter
# because the data enter its constraint.

# %%
certified_ellipsoid = feasible.outer_ellipsoid(criterion="volume", via=route)
feasible_ellipsoid = feasible.outer_ellipsoid(criterion="volume")
print(
    "ellipsoid half-widths along arc 0:",
    f"certificate {certified_ellipsoid.support_function()(P.basis_vector(0)) - estimate[0]:.3f},",
    f"feasible {feasible_ellipsoid.support_function()(P.basis_vector(0)) - P.to_components(feasible_ellipsoid.center)[0]:.3f}",
)

# %%
angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
around = [P.from_components(np.array([np.cos(a), np.sin(a)])) for a in angles]
fig, ax = plt.subplots(figsize=(5.5, 5.0), layout="constrained")
plotting.plot_set(certified_ellipsoid, ax=ax, color="C1", alpha=0.3)
plotting.plot_set(feasible_ellipsoid, ax=ax, color="C0", alpha=0.15)
# The set itself is drawn from its support function, one small optimisation
# per supporting line: forty-eight lines make a smooth outline, and they are
# independent, so four worker processes sweep them.
plotting.plot_set(feasible, ax=ax, directions=48, n_jobs=4, color="C0", alpha=0.6)
hull = feasible.inner_hull(around)
plotting.plot_set(
    hull, ax=ax, bounds=(ax.get_xlim(), ax.get_ylim()), color="C2", alpha=0.6
)
ax.plot(*T(truth), "k*", ms=12, label="truth")
ax.plot(*estimate, "o", color="C1", label="Backus-Gilbert estimate")
ax.set_xlabel("average over arc 0")
ax.set_ylabel("average over arc 1")
ax.set_title(
    "The feasible set (blue), its inner hull (green),\n"
    "its outer ellipsoid (pale) and the certificate's (orange)",
    fontsize=10,
)
ax.legend(loc="lower right", fontsize=8)

# %% [markdown]
# The feasible set is the exact answer to the question the bounds pose;
# the certificate's set contains it, as it must, and costs less — one
# linear solve per property, where each support value of the feasible set
# is a small optimisation. Both are far inside what the prior bound alone
# allows (±2.8 on each axis), and both are much longer along arc 1, which
# the data barely see.
#
# ## Three answers to three questions
#
# Tutorial 7's credible set and this feasible set are not competing
# estimates of one thing. The credible set says where the property lies
# *with 95 per cent probability*, given a prior distribution you believe;
# the feasible set says where it lies *for certain*, given only a bound you
# believe. When the prior is a distribution the first is the stronger
# statement; when no one can defend a distribution, the second is the one
# that can be defended. The Backus-Gilbert point sits between them as what
# either completes: an estimate with a resolution and a propagated error,
# which becomes an uncertainty only once something is assumed about the
# field. The library keeps the three forms of answer — a point, a measure,
# a set — as different types, because confusing them is a common way to
# report an uncertainty that was never computed.

# %%
plotting.show()
