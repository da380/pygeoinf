# %% [markdown]
# # Tutorial 14: one problem, three answers
#
# The course so far has produced three kinds of answer to an inverse
# problem: a point (tutorial 8), a measure (tutorial 6) and a set
# (tutorial 7). They were introduced on different problems, for different
# reasons, and it is easy to come away thinking of them as three methods
# that happen to be in one library. They are not. They are three answers
# to one question, and which one you get is decided before any data are
# seen, by what you are willing to say about the unknown.
#
# This tutorial puts the three side by side on a single problem, with a
# known truth so that each can be marked. The point comes with its error
# budget, and the budget is shown to be exactly that: a budget, not an
# uncertainty. The set and the measure are the two ways of completing it.
# A nonlinear coda repeats the pattern where the forward map bends.

# %%
import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import (
    BackusGilbert,
    BackusGilbertParker,
    CholeskySolver,
    ForwardProblem,
    GaussianMeasure,
    LinearForwardProblem,
    LinearGaussianInversion,
    MaximumAPosteriori,
    MinimumNorm,
    plotting,
)
from pygeoinf2.geometry import LinearCertificate
from pygeoinf2.inference import LaplaceApproximation, Occam
from pygeoinf2.numerics.optimization import NewtonCG
from pygeoinf2.symmetric_space import circle, plane

rng = np.random.default_rng(14)

# %% [markdown]
# ## The problem
#
# A field on a circle, observed at twenty points with Gaussian noise. The
# model space is $H^2$ with a length scale of 0.5, so that point values
# are defined; the prior is a heat measure on it with a correlation length
# of 0.3; and the truth is a draw from that prior, so that the measure
# answer is asked a fair question and the other two can be judged against
# it. The sensors are placed at random, which leaves gaps.

# %%
X = circle.Sobolev(256, 2.0, 0.5)
sensors = np.sort(rng.uniform(0.0, 2.0 * np.pi, 20))
A = X.point_evaluation_operator(sensors)
noise = GaussianMeasure.from_standard_deviation(A.codomain, 0.1)
problem = LinearForwardProblem(A, error=noise)
prior = X.heat_measure(0.3, pointwise_std=1.0)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)

# %% [markdown]
# The question is about three properties of the field, not the field
# itself: its averages over arcs of half-width 0.3 centred at 1.0, 2.6 and
# 4.4. The property operator $T$ maps a field to those three numbers, and
# every answer below is pushed through it. How many sensors fall near each
# arc will decide how well each is determined.

# %%
centers = (1.0, 2.6, 4.4)
T = X.geodesic_ball_average_operator([np.array([c]) for c in centers], 0.3)
P = T.codomain
true_property = T(truth)
print("the true averages:", true_property.round(3))
near = [int(np.sum(np.abs(sensors - c) < 0.5)) for c in centers]
print("sensors within 0.5 of each arc's centre:", near)

# %%
_, ax = plotting.subplots(X, figsize=(8.0, 3.0))
plotting.plot(X, truth, ax=ax, color="black", lw=1.2, label="truth")
ax.plot(sensors, data, "o", color="C3", ms=4, label="point observations")
for c in centers:
    ax.axvspan(c - 0.3, c + 0.3, color="C0", alpha=0.15)
ax.set_xlabel("angle")
ax.legend(fontsize=8, loc="upper right")
ax.set_title("The truth, the data, and the three arcs (shaded) asked about")

# %% [markdown]
# ## The point
#
# `MinimumNorm` is the smallest model that fits the data: the Tikhonov
# minimiser at the largest damping whose misfit still meets the
# discrepancy principle, with the confidence level its one knob. The
# damping is found from the data, so the estimator is not affine in them;
# but `for_data` returns the `LeastSquares` at the damping the data chose,
# which is affine, and that is the estimator that carries an error budget.
# Pushed through $T$ it estimates the three averages.

# %%
chosen = MinimumNorm(problem).for_data(data)
point = chosen.push_forward(T)
estimate = point(data)
print(f"damping chosen by the data: {chosen.damping:.1f}")

# %% [markdown]
# The budget is two operators. With $B$ the estimator, $A$ the forward
# operator and $e$ the noise, the estimate of $Tm$ from $d = Am + e$ is in
# error by
#
# $$B d - T m = (BA - T)\,m + B e .$$
#
# The second term is the noise seen through the estimator; its covariance,
# $B R B^*$, is `propagated_covariance`, and its square root is a standard
# deviation per property. The first term is the bias: the resolution
# $BA$ is not $T$, and the difference acts on the unknown model. Here the
# truth is known, so `bias(truth)` evaluates it.

# %%
propagated = point.propagated_covariance()
deviation = np.array(
    [
        np.sqrt(P.inner_product(propagated(P.basis_vector(i)), P.basis_vector(i)))
        for i in range(3)
    ]
)
bias = point.bias(truth)
print(f"{'arc':>4} {'estimate':>9} {'propagated sd':>14} {'bias':>8} {'truth':>8}")
for i, c in enumerate(centers):
    print(
        f"{c:4.1f} {estimate[i]:+9.3f} {deviation[i]:14.3f} {bias[i]:+8.3f}"
        f" {true_property[i]:+8.3f}"
    )
print(
    "bias larger than the propagated sd on",
    int(np.sum(np.abs(bias) > deviation)),
    "of the three arcs",
)

# %% [markdown]
# On every arc the bias exceeds the propagated standard deviation, by a
# factor of three on the arc at 2.6: an error bar built from the noise
# alone would be confidently wrong. Notice that the arc with the fewest
# sensors is not the one with the largest bias. The bias is $(BA - T)\,m$,
# and it depends on $m$ as much as on the sensors: where the truth happens
# to be flat, a blurred average of it is close to the true one, and where
# it bends, not. Nothing in the data can say how large it is, because
# $m$ is what nobody knows. Bounding it takes a statement about the model, and there are two
# honest ones: a set the model lies in, or a measure it is drawn from. Each
# gives a complete answer, and the point estimator is what either of them
# completes.
#
# ## The set
#
# The first statement is a norm bound: $\|m\| \le M$ in the model space's
# norm, which in $H^2$ bounds roughness as well as size. To ask the two
# completions from comparable knowledge, the bound here is the ball that
# carries 95 per cent of the prior measure, which is a lossy hardening of
# it; the noise is hardened the same way, into the ball carrying 95 per
# cent of the error measure. The truth, a draw from the prior, should
# satisfy the bound, and does.

# %%
ball = prior.ambient_ball(level=0.95)
print(
    f"the prior bound: ||m|| <= {ball.radius:.2f}; the truth's norm is {X.norm(truth):.2f}"
)
print("the truth satisfies it:", ball.contains(truth))

# %% [markdown]
# `BackusGilbertParker` returns the set of all values $Tm$ over the models
# in the ball that fit the data to within the noise ball. It is convex and
# it answers questions: a support value in any direction, with a bracket
# whose inner end is attained by a feasible model and whose outer end is
# certified. Asked in the directions $\pm q_i$ it gives a bracket for each
# average, and the truth lies within every one.

# %%
feasible = BackusGilbertParker(problem, T, ball)(data)
print(feasible)
bracket = np.zeros((3, 2))
print(f"{'arc':>4} {'from':>8} {'to':>8} {'truth':>8} {'certified gap':>14}")
for i, c in enumerate(centers):
    q = P.basis_vector(i)
    inner_hi, outer_hi = feasible.support_bracket(q)
    inner_lo, outer_lo = feasible.support_bracket(P.scale(-1.0, q))
    bracket[i] = (-outer_lo, outer_hi)
    gap = max(outer_hi - inner_hi, outer_lo - inner_lo)
    print(
        f"{c:4.1f} {bracket[i, 0]:+8.3f} {bracket[i, 1]:+8.3f} {true_property[i]:+8.3f}"
        f" {gap:14.1e}"
    )
print("the true averages are in the set:", feasible.contains(true_property))

# %% [markdown]
# Three brackets are a box around the set, and a box is a crude outer
# bound: support values in more directions give a polytope that follows
# its shape, and the contact points of those directions give an inner
# hull. The truth must lie in the polytope; whether it lies in the hull
# is a question the hull can only answer one way, since the hull is
# inside the set and need not reach the truth.

# %%
directions = []
for signs in np.array(np.meshgrid(*[(-1, 0, 1)] * 3)).reshape(3, -1).T:
    if np.any(signs):
        directions.append(P.from_components(signs / np.linalg.norm(signs)))
polytope = feasible.polytope(directions, n_jobs=4)
hull = feasible.inner_hull(directions)
print(
    f"{len(directions)} directions: a polytope with {len(polytope.half_spaces)} faces"
)
print("the truth is in the polytope:", polytope.contains(true_property))
print("and in the inner hull:", hull.contains(true_property))

# %% [markdown]
# Each support value above is a small optimisation. There is a cheaper
# bound, and it is the Backus-Gilbert estimate completed by the two balls:
# for any fixed weighting $X$ of the data, every feasible model satisfies
# $Tm = Xd + (T - XA)\,m - Xe$, so $Tm$ lies in the estimate plus the reach
# of $T - XA$ over the prior ball plus the reach of $X$ over the noise
# ball. That is valid whatever $X$ is, and it is sharp only by luck. The
# feasible set offers it as its `linear_certificate` route, which is never
# chosen on its own; here it is given the $X$ that `BackusGilbert` builds,
# and compared with the certified value.

# %%
averaging = BackusGilbert(problem, T, solver=CholeskySolver())
route = LinearCertificate(operator=averaging.operator)
print(f"{'arc':>4} {'certified':>10} {'certificate':>12} {'of which noise':>15}")
for i, c in enumerate(centers):
    q = P.basis_vector(i)
    _, resolution_part, noise_part = route.terms(feasible, q)
    print(
        f"{c:4.1f} {feasible.support(q):+10.3f} {feasible.support(q, via=route):+12.3f}"
        f" {noise_part:15.3f}"
    )

# %% [markdown]
# The certificate's bound contains the certified one, as weak duality
# says it must, and it splits its width into a resolution part and a noise
# part, which the exact value does not. The better an arc is covered, the
# closer the two come; on the arc at 1.0 the certificate gives away 0.4,
# because a fixed $X$ cannot adapt to the data as the exact optimisation
# does.
#
# ## The measure
#
# The second statement is the prior measure itself. `LinearGaussianInversion`
# returns the posterior, a Gaussian on the model space, and pushing it
# through $T$ gives the posterior on the three averages: a Gaussian on
# $\mathbb{R}^3$, with a standard deviation per average and a credible
# interval at any level. The truth lies within each interval.

# %%
posterior = LinearGaussianInversion(problem, prior)(data)
marginal = posterior.push_forward(T)
mean = marginal.expectation
posterior_sd = np.array(
    [
        np.sqrt(
            P.inner_product(marginal.covariance(P.basis_vector(i)), P.basis_vector(i))
        )
        for i in range(3)
    ]
)
interval = np.stack([mean - 1.96 * posterior_sd, mean + 1.96 * posterior_sd], axis=1)
print(f"{'arc':>4} {'mean':>8} {'sd':>7} {'95% from':>9} {'to':>8} {'truth':>8}")
for i, c in enumerate(centers):
    print(
        f"{c:4.1f} {mean[i]:+8.3f} {posterior_sd[i]:7.3f} {interval[i, 0]:+9.3f}"
        f" {interval[i, 1]:+8.3f} {true_property[i]:+8.3f}"
    )

# %% [markdown]
# ## Three answers, one plot per property

# %%
fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6), layout="constrained")
for i, (ax, c) in enumerate(zip(axes, centers)):
    ax.errorbar(
        0,
        estimate[i],
        yerr=2.0 * deviation[i],
        fmt="o",
        color="C1",
        capsize=4,
        label="point, 2 propagated sd",
    )
    ax.plot([1, 1], bracket[i], color="C0", lw=6, alpha=0.6, solid_capstyle="butt")
    ax.plot(
        1,
        0.5 * np.sum(bracket[i]),
        "_",
        color="C0",
        ms=14,
        label="set, certified bracket",
    )
    ax.errorbar(
        2,
        mean[i],
        yerr=1.96 * posterior_sd[i],
        fmt="s",
        color="C2",
        capsize=4,
        label="measure, 95% interval",
    )
    ax.axhline(true_property[i], color="black", ls="--", lw=1.0, label="truth")
    ax.set_xticks([0, 1, 2], ["point", "set", "measure"])
    ax.set_xlim(-0.6, 2.6)
    ax.set_title(f"arc at {c}, {near[i]} sensors near", fontsize=10)
axes[0].set_ylabel("average")
axes[0].legend(fontsize=7, loc="lower left")
fig.suptitle("The same question, answered three ways")

# %% [markdown]
# Read the three columns of each panel together. The point's bar is the
# noise alone: on the arc at 2.6 it misses the truth, because the bias is
# not in it, and on the other two it contains the truth only because the
# bias there happened to be small. The set's bracket and the measure's
# interval both contain the truth, and both are
# wider where the data are thin — but they are wide in different
# measures. The bracket is a guarantee: given the bound, the average is
# in it, full stop. The interval is a probability: given the prior, it
# holds the average with 95 per cent credence, and the prior is doing most
# of the work where the sensors are absent. The bracket is the wider by
# far, and that is not a defect of the algorithm but of what it was told:
# a norm bound admits the field that spends its whole budget on one arc,
# which the Gaussian prior calls vanishingly unlikely. More knowledge,
# narrower answer; the two answers differ because the two statements do.
# Neither is a corrected version of the point; each is a different
# answer, to a question that was posed differently.
#
# ## Coda: the same pattern where the forward map bends
#
# Ray tomography, as in tutorial 13 but small enough to run in seconds: a
# velocity $c_0 e^m$ on a square, sixteen crossing paths, travel times along
# the rays the model bends. The model space is $H^2$ on the square, the
# prior a Sobolev measure on it, and the truth a draw from the prior. The
# property is one number, the average of $m$ over a disc in the middle of
# the square, where the rays cross.

# %%
Y = plane.Sobolev((24, 24), 2.0, 0.25, bounds=[(0.0, 2.0), (0.0, 2.0)])
ray_prior = Y.sobolev_measure(2.0, 0.3, pointwise_std=0.25)
ray_truth = ray_prior.sample(rng=rng)
sources = [np.array([0.2, y]) for y in np.linspace(0.5, 1.5, 4)]
receivers = [np.array([1.8, y]) for y in np.linspace(0.5, 1.5, 4)]
paths = [(s, r) for s in sources for r in receivers]
F = Y.ray_travel_time_operator(paths, 1.0, count=12)
times = F(ray_truth)
ray_noise = GaussianMeasure.from_standard_deviation(
    F.codomain, 0.005 * float(np.mean(times))
)
ray_problem = ForwardProblem(F, error=ray_noise)
ray_data = F.codomain.add(times, ray_noise.sample(rng=rng))
S = Y.geodesic_ball_average_operator([np.array([1.0, 1.0])], 0.3)
q = S.codomain.basis_vector(0)
print(f"the true average over the disc: {S(ray_truth)[0]:+.4f}")

# %% [markdown]
# **A point.** `MaximumAPosteriori` minimises the negative log posterior
# and returns the mode. Its gradient is the representer of the derivative
# in the $H^2$ inner product and its Hessian a self-adjoint operator on
# $H^2$, the prior's precision plus the data term; the forward operator
# supplies its second derivative, so a Newton optimiser can be handed the
# exact Hessian. That Hessian is badly conditioned, because the prior's
# precision is confident about fine scales the data never see, and the
# cure is a preconditioner: the prior's covariance, which leaves the
# inner solve with only the data's spectrum to work through.
#
# `Occam` answers the question `MinimumNorm` asked in the linear case —
# the model closest to the reference whose misfit meets the level, here
# the smallest in the $H^2$ norm — but by a constrained minimisation
# rather than a damping search, since the misfit need no longer be
# monotone in the damping once the forward map bends. It is started from
# the mode, which already fits, and its tolerances are set to what a
# misfit computed through a ray tracer can meet. Both are points; the
# second is the more conservative, as the smallest model is.

# %%
map_estimator = MaximumAPosteriori(
    ray_problem,
    ray_prior,
    optimizer=NewtonCG(
        preconditioner=ray_prior.covariance, max_iterations=15, max_cg_iterations=40
    ),
)
search = map_estimator.solve(ray_data)
mode = search.minimizer
print(
    f"Newton: {search.iterations} iterations, gradient norm"
    f" {search.gradient_norm:.1e}, converged: {search.converged}"
)
occam = Occam(ray_problem, start=mode, gtol=1e-4, max_iterations=6).solve(ray_data)
print(f"{'':12s} {'average':>8} {'misfit':>8} {'norm':>7}")
for name, model in (("MAP mode", mode), ("Occam", occam.minimizer)):
    print(
        f"{name:12s} {S(model)[0]:+8.4f} {ray_problem.chi_squared(model, ray_data):8.1f}"
        f" {Y.norm(model):7.3f}"
    )
print(
    f"the misfit level is {ray_problem.critical_chi_squared():.1f};"
    f" Occam's constraint is active: {occam.inequality_multipliers[0] > 0.0}"
)

# %% [markdown]
# **A measure.** `LaplaceApproximation` is the measure estimator built on
# the MAP estimator: the Gaussian about the mode whose covariance is the
# inverse of the normal operator of the problem linearised there. Pushed
# through the property it gives a mean and a standard deviation for the
# average, as the linear posterior did; what it does not give is any
# guarantee that the posterior is Gaussian.

# %%
laplace = LaplaceApproximation(map_estimator).at(mode, ray_data).push_forward(S)
laplace_sd = np.sqrt(S.codomain.inner_product(laplace.covariance(q), q))
print(
    f"Laplace: {laplace.expectation[0]:+.4f} +/- {1.96 * laplace_sd:.4f} at 95%,"
    f" truth {S(ray_truth)[0]:+.4f}"
)

# %% [markdown]
# **A set.** The feasible set of a nonlinear problem is not convex, and
# the library does not pretend otherwise. What `BackusGilbertParker` does
# with a nonlinear problem is linearise it about a model, here the mode,
# and return the feasible property set of the linearised problem: exact
# for that problem, and a local statement about the real one, good to the
# extent that the linearisation holds across the constraint set. The
# constraint is the ball in $H^2$ carrying 95 per cent of the prior, as
# before; everything the set reports is in the property's own values.

# %%
ray_ball = ray_prior.ambient_ball(level=0.95)
linearized = BackusGilbertParker(ray_problem, S, ray_ball, model=mode)(ray_data)
lower = -linearized.support(S.codomain.scale(-1.0, q))
upper = linearized.support(q)
print(
    f"linearised set: the average lies between {lower:+.4f} and {upper:+.4f};"
    f" the mode's own is {linearized.linearization_property[0]:+.4f},"
    f" the truth's {S(ray_truth)[0]:+.4f}"
)
print("the truth is in the ball the set assumes:", ray_ball.contains(ray_truth))

# %% [markdown]
# The same three answers, from the same three descriptions of the
# uncertainty, in the same space, with the nonlinearity changing only how
# each is computed: a search instead of a solve for the point, a
# curvature at the mode instead of an exact covariance for the measure, a
# linearisation instead of the exact set.
#
# ## The pattern
#
# Data go in, and a point, a set or a measure comes out, and the kind of
# answer is not a choice of method but a consequence of how the
# uncertainty was described: no statement about the model gives a point
# with a budget it cannot close; a set the model lies in gives a set; a
# measure it is drawn from gives a measure. Every estimator in the library
# is one of the three, linear or not, and every one has `push_forward`,
# which takes the answer from the model to a property of it — applying $T$
# to a point, taking the image of a set, pushing a measure — so that the
# question "what do the data say about this?" is asked the same way
# whatever kind of answer it gets.

# %%
plotting.show()
