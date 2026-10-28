# %% [markdown]
# # Tutorial 13: nonlinear ray tomography on a function space
#
# Every inverse problem so far has had a linear forward map, or a nonlinear
# one simple enough to write its derivative by hand. Ray tomography is the
# real thing: the travel time of a wave between a source and a receiver is
# the integral of the slowness along the ray, and the ray itself bends
# with the velocity it travels through. The data depend on the model
# through the path the model chooses.
#
# This tutorial puts that problem on a function space and takes it all the
# way: the forward operator and the rays it traces, its linearisation and
# the sensitivity kernels that come from the adjoint, its Hessian action
# with the ray's own response inside it, and then the inverse problem
# solved by the nonlinear machinery of tutorial 9 — first with gradients,
# then with Newton steps built on that Hessian — against the straight-ray
# inversion that linear tomography would settle for. Nothing here needs
# more than the core library; it runs on the plane in seconds.

# %%
import numpy as np

from pygeoinf2 import (
    ForwardProblem,
    GaussianMeasure,
    LinearForwardProblem,
    LinearGaussianInversion,
    MaximumAPosteriori,
    plotting,
)
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.inference import LaplaceApproximation
from pygeoinf2.numerics.optimization import LBFGS, NewtonCG
from pygeoinf2.symmetric_space.plane import Sobolev

rng = np.random.default_rng(13)

# %% [markdown]
# ## The medium, and the rays through it
#
# The unknown is a field $m$ on a square, and the velocity is $c_0 e^{m}$:
# positive whatever $m$ is, and equal to the background $c_0$ where $m$
# vanishes. The model space is $H^2$ on the square, with a length scale a
# fifth of its side, and the truth is drawn from a Gaussian measure on it
# so that the inversion later asks a fair question. The anomalies are a
# few tenths in $\ln c$, which is enough to bend rays visibly.

# %%
X = Sobolev((40, 40), 2.0, 0.25, bounds=[(0.0, 2.0), (0.0, 2.0)])
c0 = 1.0
prior = X.sobolev_measure(2.0, 0.3, pointwise_std=0.25)
truth = prior.sample(rng=rng)
print(
    f"ln c in the truth: {X.grid_values(truth).min():+.2f} to {X.grid_values(truth).max():+.2f}"
)

# %% [markdown]
# Sources along the left and bottom edges, receivers along the right and
# top, every pair a path: a crossing geometry, as a tomographic survey has.
# The paths are kept away from the edges, where the periodic padding that
# gives the square its spectral basis takes over.

# %%
sources = [np.array([0.2, y]) for y in np.linspace(0.4, 1.6, 4)] + [
    np.array([x, 0.2]) for x in np.linspace(0.6, 1.4, 3)
]
receivers = [np.array([1.8, y]) for y in np.linspace(0.4, 1.6, 4)] + [
    np.array([x, 1.8]) for x in np.linspace(0.6, 1.4, 3)
]
paths = [(s, r) for s in sources for r in receivers]
print(f"{len(paths)} paths")

# %% [markdown]
# The forward operator is a method of the space, as the linear path
# operators are. It returns an `Operator`, the library's nonlinear kind:
# calling it gives the travel times, `at(m)` gives the travel times
# together with the derivative, and `second_derivative(m, dm)` the
# Hessian action. The ray between each pair is found by bending: the
# geodesic's nodes are moved across the path until the discrete travel
# time is stationary, every path at once, with the field evaluated at all
# nodes in one transform per step.

# %%
F = X.ray_travel_time_operator(paths, c0, count=16)
times = F(truth)
print(f"travel times from {times.min():.3f} to {times.max():.3f}")

# %% [markdown]
# Why bending matters can be seen by integrating the same slowness along
# the geodesics instead, which is what a straight-ray code would report.
# Fermat's principle says the true ray is never slower than that.

# %%
nodes = [X.geodesic_quadrature(s, r, count=60) for s, r in paths]
along_geodesics = np.array(
    [float(np.sum(w * np.exp(-X.evaluate(truth, pts)) / c0)) for pts, w in nodes]
)
print(
    f"bent rays are faster than the geodesics by up to "
    f"{np.max(along_geodesics - times) / np.mean(times):.1%} of a travel time,"
    f" never slower: {np.all(times <= along_geodesics * (1 + 1e-9))}"
)

# %%
# The box renderer puts a field's first coordinate on the vertical axis,
# as the point and path plotters do, so a ray drawn by hand goes on as
# (second coordinate, first coordinate).
rays = F.trace_rays(truth)
_, ax = plotting.subplots(X, figsize=(6.0, 5.5))
plotting.plot(X, truth, ax=ax, cmap="RdBu_r", symmetric=True, colorbar_label="ln c")
plotting.plot_paths(X, paths, ax=ax, color="0.5", linewidth=0.5, alpha=0.6)
for ray in rays:
    points = np.stack(ray)
    ax.plot(points[:, 1], points[:, 0], color="black", linewidth=0.7)
ax.set_title("The velocity anomaly, the geodesics (grey) and the rays (black)")

# %% [markdown]
# ## The linearisation, and what its adjoint draws
#
# Fermat's principle does more than make the ray fast: it makes the travel
# time stationary in the ray, so that to first order a change of the model
# changes the travel time only through the slowness along the ray that
# was already found. The derivative of the operator is therefore a path
# integral along the traced ray — the linear operator of tutorial 10 on a
# bent path — and its adjoint is the one that operator carries. The check
# is the same as for any operator: finite differences in a random
# direction.

# %%
linearized = F.at(truth)
direction = prior.sample(rng=rng)
step = 1.0e-4
differences = (
    F(X.add(truth, X.scale(step, direction)))
    - F(X.add(truth, X.scale(-step, direction)))
) / (2.0 * step)
predicted = linearized.derivative(direction)
print(
    "derivative against central differences: relative error "
    f"{np.max(np.abs(predicted - differences)) / np.max(np.abs(differences)):.1e}"
)

# %% [markdown]
# The adjoint applied to one datum's unit vector is a *function* on the
# square: the sensitivity kernel of that travel time, the Riesz
# representer in $H^2$ of the path integral along its ray. It is a smooth
# ridge along the ray rather than a line, because the space's inner
# product is what turns a line functional into a function, and $H^2$ does
# not contain lines.

# %%
which = 17
unit = np.zeros(len(paths))
unit[which] = 1.0
kernel = linearized.derivative.adjoint(unit)
kernel_straight = F.at(X.zero()).derivative.adjoint(unit)
figure, axes = plotting.subplots(X, columns=2, figsize=(11.0, 4.8))
for ax, field, title in (
    (axes[0], kernel_straight, "sensitivity of one travel time, straight ray"),
    (axes[1], kernel, "the same, along the ray through the anomaly"),
):
    plotting.plot(X, field, ax=ax, cmap="PuOr", symmetric=True, colorbar_label="kernel")
    s, r = paths[which]
    ax.plot([s[1], r[1]], [s[0], r[0]], color="0.4", linewidth=0.8)
    ax.set_title(title, fontsize=10)
ray = np.stack(rays[which])
axes[1].plot(ray[:, 1], ray[:, 0], color="black", linewidth=0.8)

# %% [markdown]
# ## The Hessian, with the ray's response in it
#
# Second order is where ray tomography stops being a path integral. The
# second derivative of a travel time has two parts: the curvature of the
# slowness in the model along the fixed ray, and the change of the
# derivative as the ray itself moves in response to the model. The
# operator supplies both, through the reduced Hessian of the discrete
# travel time at its stationary point — the same tridiagonal system the
# bending Newton step solves. Two checks that a Hessian can be put to: it
# must be the derivative of the derivative, and it must be symmetric.

# %%
second = prior.sample(rng=rng)
action = F.second_derivative(truth, direction)(second)
differences = (
    F.derivative(X.add(truth, X.scale(step, direction)))(second)
    - F.derivative(X.add(truth, X.scale(-step, direction)))(second)
) / (2.0 * step)
print(
    "Hessian against differences of the derivative: relative error "
    f"{np.max(np.abs(action - differences)) / np.max(np.abs(differences)):.1e}"
)
transposed = F.second_derivative(truth, second)(direction)
print(
    "symmetry in its two directions: relative error "
    f"{np.max(np.abs(action - transposed)) / np.max(np.abs(action)):.1e}"
)

# %% [markdown]
# How much the ray's response contributes can be read off by comparing
# with the curvature of the slowness alone, integrated along the fixed
# ray. For the pairs below it is not a correction but a term of the same
# size, which is the reason a Gauss-Newton Hessian — the derivative's
# square, no second derivative at all — is only an approximation here.

# %%
fixed_ray = np.zeros(len(paths))
for index, ray_points in enumerate(rays):
    points = np.stack(ray_points)
    # Trapezoid rule along the traced ray.
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    weights = np.zeros(len(points))
    weights[:-1] += 0.5 * lengths
    weights[1:] += 0.5 * lengths
    fixed_ray[index] = float(
        np.sum(
            weights
            * np.exp(-X.evaluate(truth, list(points)))
            / c0
            * X.evaluate(direction, list(points))
            * X.evaluate(second, list(points))
        )
    )
share = np.abs(action - fixed_ray) / np.abs(action)
print(f"the ray's response, as a share of the Hessian action: up to {share.max():.0%}")

# %% [markdown]
# ## The inverse problem
#
# Noisy travel times, a Gaussian prior on $m$, and the question of
# tutorial 6 and tutorial 9: what is the most probable field, and how sure
# is it? Three answers are compared.
#
# 1. **Linear tomography.** Straight rays: the operator linearised at
#    $m = 0$, which is the path-integral operator of tutorial 10 scaled by
#    $-1/c_0$, and the linear Gaussian inversion on it. One solve.
# 2. **The MAP field with gradients.** Tutorial 9's `MaximumAPosteriori`
#    with L-BFGS: every iteration traces the rays afresh and uses the
#    derivative's adjoint for the gradient.
# 3. **The MAP field with Newton steps.** The same estimator handed a
#    Newton optimizer: the forward operator supplies its second
#    derivative, so the objective carries the exact Hessian — the
#    Gauss-Newton term plus the operator's curvature weighted by the
#    residual — and each step solves it by conjugate gradients.
#
# One change of variables first. The prior's precision on $H^2$ is a
# fourth-order differential operator, so an optimizer working on $m$
# directly faces a Hessian whose eigenvalues span many decades and makes
# slow progress along the smooth directions that matter. Writing
# $m = L w$ with $C = L L^*$ the prior's covariance and $w$ white turns
# the prior term into $\|w\|^2$ and leaves the data term to supply all
# the structure: the posterior in $w$ is the identity plus something of
# low rank, which is what every method here likes. The prior carries its
# factor, and composing the operator with it is one line.

# %%
L = prior.covariance_factor
identity = LinearOperator.identity(X)
white = GaussianMeasure(X, covariance=identity, precision=identity)
G = F @ L
print("the composed operator keeps its second derivative:", G.has_second_derivative)

# %%
noise = GaussianMeasure.from_standard_deviation(
    F.codomain, 0.005 * float(np.mean(times))
)
data = F.codomain.add(times, noise.sample(rng=rng))
problem = ForwardProblem(F, error=noise)

# 1. Straight rays.
jacobian_at_zero = F.derivative(X.zero())
linear = LinearGaussianInversion(
    LinearForwardProblem(jacobian_at_zero, error=noise), prior
)
linear_model = linear(F.codomain.subtract(data, F(X.zero()))).expectation

# 2. Gradients, in the whitened variable. The estimator returns the mode;
#    `solve` returns the optimizer's record, which is what is compared here.
whitened = ForwardProblem(G, error=noise)
by_lbfgs = MaximumAPosteriori(
    whitened, white, optimizer=LBFGS(max_iterations=100)
).solve(data)

# 3. Newton, likewise.
newton = MaximumAPosteriori(
    whitened, white, optimizer=NewtonCG(max_iterations=15, max_cg_iterations=40)
)
by_newton = newton.solve(data)
lbfgs_model, newton_model = L(by_lbfgs.minimizer), L(by_newton.minimizer)


def error_of(model):
    return X.norm(X.subtract(model, truth)) / X.norm(truth)


print(f"{'':28s} {'relative error':>14s} {'iterations':>10s}")
print(f"{'straight rays, linear':28s} {error_of(linear_model):14.3f} {1:10d}")
print(
    f"{'MAP, L-BFGS':28s} {error_of(lbfgs_model):14.3f} " f"{by_lbfgs.iterations:10d}"
)
print(
    f"{'MAP, Newton':28s} {error_of(newton_model):14.3f} " f"{by_newton.iterations:10d}"
)
print(
    "the two MAP fields agree to "
    f"{X.norm(X.subtract(newton_model, lbfgs_model)) / X.norm(lbfgs_model):.1e}"
)

# %% [markdown]
# The straight-ray answer is not wrong so much as misplaced: it puts the
# anomalies where the geodesics cross, and the rays did not go there. The
# two nonlinear answers agree with each other, and Newton gets there in a
# fraction of the iterations because its model of the objective includes
# how the rays move.

# %%
figure, axes = plotting.subplots(X, columns=3, figsize=(15.0, 4.6))
limit = float(np.max(np.abs(X.grid_values(truth))))
for ax, field, title in (
    (axes[0], truth, "truth"),
    (axes[1], linear_model, "straight rays"),
    (axes[2], newton_model, "MAP, rays traced"),
):
    plotting.plot(
        X, field, ax=ax, cmap="RdBu_r", vmin=-limit, vmax=limit, colorbar_label="ln c"
    )
    ax.set_title(title)

# %% [markdown]
# ## How sure
#
# The Laplace approximation of tutorial 9 is a Gaussian about the mode
# whose covariance is the inverse of the linearised normal operator there;
# `LaplaceApproximation` wraps the MAP estimator and places it on the mode
# already found. It was found in $w$; pushed through $L$ it is a Gaussian on
# the field,
# and its pointwise standard deviation — one covariance application per
# point, each a solve with the linearised normal operator — says what the
# survey can and cannot see: the uncertainty is lowest where rays cross
# and rises towards the corners no ray reaches.

# %%
grid = [
    np.array([x, y]) for x in np.linspace(0.3, 1.7, 6) for y in np.linspace(0.3, 1.7, 6)
]
laplace = L @ LaplaceApproximation(newton).at(by_newton.minimizer, data)
spread = np.sqrt(X.pointwise_variance_at(laplace, grid))
prior_spread = np.sqrt(X.pointwise_variance_at(prior, grid[:1]))[0]
print(
    f"posterior standard deviation from {spread.min():.3f} to {spread.max():.3f},"
    f" against the prior's {prior_spread:.3f}"
)
_, ax = plotting.subplots(X, figsize=(6.0, 5.5))
plotting.plot_points(
    X, grid, data=spread, ax=ax, cmap="viridis", size=120.0, colorbar_label="sd of ln c"
)
for ray in rays:
    points = np.stack(ray)
    ax.plot(points[:, 1], points[:, 0], color="white", linewidth=0.3, alpha=0.6)
ax.set_title("Where the survey looks: the Laplace standard deviation")

# %% [markdown]
# ## What carried over, and what is new
#
# Everything the linear tutorials built is used unchanged: the space and
# its inner product, the prior as a measure, the adjoint as the thing that
# turns a datum into a function, the inversion as an estimator. What is
# new is the operator calculus the nonlinear map needs — `at`, the second
# derivative, the failure protocol when a trace does not converge — and
# the fact that the tracer lives in the same space the field does, so the
# ray, the kernel and the field are all functions of one geometry. Change
# `Sobolev` to the sphere's and the rays are great circles bent by the
# anomaly; nothing else in this tutorial changes.

# %%
plotting.show()
