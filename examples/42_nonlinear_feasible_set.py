"""
42. A set answer when the forward map is not linear.

Example 30 answered a linear problem with a set: the property values of
every model a prior bound allows and the data do not rule out. That set is
convex, and one optimization per direction describes it. Example 31 met a
nonlinear map and answered with a mode and a Gaussian about it.

This example asks for the set when the map is nonlinear. The feasible models
are then not a convex set, and may not even be in one piece, so three
answers are compared:

- the set of the problem **linearized** about a model that fits, which is
  convex and cheap, and is neither inside nor outside the real one;
- a **local search**, which exhibits a model, checked to lie in the prior
  and fit the data, and so bounds the set from inside and says nothing of
  what it did not reach;
- a **search over cells**, which brackets the set from both sides, given a
  bound on how far the map departs from its linearization.

The last needs something the others do not: a claim about the map's
curvature. Here the maps are simple enough for the claim to be proved, and
the bracket is certified. For a map known only through its code the claim
would be an estimate, and the result would say so.

Needs matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

from pygeoinf import (
    BackusGilbertParker,
    Ball,
    EuclideanSpace,
    EvaluationError,
    ForwardProblem,
    GaussianMeasure,
    LinearOperator,
    NonlinearBackusGilbertParker,
    Operator,
    plotting,
)
from pygeoinf.geometry import BranchAndPrune, ComponentwiseRemainder
from pygeoinf.symmetric_space import line

# ---------------------------------------------------------------------------
# Two parameters, so that everything can be drawn.
# ---------------------------------------------------------------------------
#
# Attenuation along three paths, as in example 31: each datum is
# exp(-(a_i, m)). The prior is a disc, the noise a ball, and the property is
# one linear combination of the two parameters.

rng = np.random.default_rng(3)
X, D = EuclideanSpace(2), EuclideanSpace(3)
paths = np.abs(rng.standard_normal((3, 2))) + 0.2


def attenuation(model: np.ndarray) -> np.ndarray:
    """The predicted data: attenuation along each path.

    A search may try a model far outside the prior, where the exponential
    overflows; saying so lets its line search back off (example 12).
    """
    exponents = -(paths @ model)
    if np.any(exponents > 50.0):
        raise EvaluationError("The attenuation overflows at this model.")
    return np.exp(exponents)


forward = Operator.from_callables(
    X,
    D,
    attenuation,
    derivative=lambda model: LinearOperator.from_matrix(
        X, D, -attenuation(model)[:, None] * paths, form="components"
    ),
)

truth = np.array([0.3, -0.2])
prior, noise = Ball(X, radius=1.0), Ball(D, radius=0.05)
error = rng.standard_normal(3)
data = forward(truth) + 0.02 * error / np.linalg.norm(error)

weights = np.array([1.0, 0.4])
T = LinearOperator.from_matrix(
    X, EuclideanSpace(1), weights[None, :], form="components"
)
up = np.array([1.0])

problem = ForwardProblem(forward, error=noise)
feasible = NonlinearBackusGilbertParker(problem, T, prior)(data)

# The linearized set, about a model that fits. Its outer value is a certified
# bound on the linearized problem, which is not the problem.
anchor = feasible.fitting_model()
linearized = BackusGilbertParker(problem, T, prior, model=anchor)(data)

# A local search: a model, checked by evaluating the map.
local = feasible.support_result(up)

# The search over cells. Each datum sees the model through one number,
# t = (a_i, m), and is exp(-t) of it, whose second derivative lies between
# exp(-t_high) and exp(-t_low) on an interval: that is the whole of the
# claim, and it is a theorem about this map.
remainder = ComponentwiseRemainder(
    D, list(paths), lambda low, high: (np.exp(-high), np.exp(-low))
)
bracket = feasible.support_result(up, via=BranchAndPrune(remainder))

print("two parameters: the largest value of the property over the feasible set")
print(f"  linearized about a fitting model   {linearized.support(up):.4f}")
print(f"  attained by a local search         {local.inner:.4f}")
print(
    f"  bracketed by the search over cells [{bracket.inner:.4f}, {bracket.value:.4f}]"
)
print(
    f"    {len(bracket.cells)} cells, certified: {bracket.certified},"
    f" refuted: {bracket.refuted}"
)
print()
print("the linearized value is below a value a feasible model attains. It is")
print("  a bound on the linearized problem and on nothing else.")
print()

# The true feasible set on a grid, which two parameters allow.
jacobian = -attenuation(anchor)[:, None] * paths


def on_grid(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, ...]:
    """A grid over a window of the plane, and where the two sets are on it."""
    mesh = np.meshgrid(
        np.linspace(low[0], high[0], 500), np.linspace(low[1], high[1], 500)
    )
    grid = np.stack(mesh, -1).reshape(-1, 2)
    in_prior = np.linalg.norm(grid, axis=1) <= prior.radius
    fits = np.linalg.norm(np.exp(-(grid @ paths.T)) - data, axis=1) <= noise.radius
    predicted = attenuation(anchor) + (grid - anchor) @ jacobian.T
    fits_linearized = np.linalg.norm(data - predicted, axis=1) <= noise.radius
    shape = mesh[0].shape
    return (
        mesh[0],
        mesh[1],
        (fits & in_prior).reshape(shape).astype(float),
        (fits_linearized & in_prior).reshape(shape).astype(float),
    )


def level_line(ax: plt.Axes, value: float, **style: object) -> None:
    """The line on which the property takes a value."""
    xs = np.array([-1.05, 1.05])
    ax.plot(xs, (value - weights[0] * xs) / weights[1], **style)


figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 6.0), layout="constrained")

# On the left, close in on the feasible models: a sliver, since three data
# pin two parameters, and curved, since the map is.
low_corner = np.minimum(anchor, local.extremal_model) - 0.06
high_corner = np.maximum(anchor, local.extremal_model) + 0.06
mesh_x, mesh_y, feasible_mask, linearized_mask = on_grid(low_corner, high_corner)
left.contourf(
    mesh_x, mesh_y, feasible_mask, levels=[0.5, 1.5], colors=["C0"], alpha=0.6
)
left.contour(
    mesh_x, mesh_y, linearized_mask, levels=[0.5], colors=["C3"], linewidths=1.2
)
level_line(left, linearized.support(up), color="C3", ls="--", lw=1.0)
level_line(left, local.inner, color="C0", ls="-", lw=1.0)
level_line(left, bracket.value, color="black", ls="--", lw=1.0)
left.plot(*anchor, "o", color="C3", ms=6)
left.plot(*local.extremal_model, "*", color="black", ms=14)
left.plot(*truth, "x", color="black", ms=8)
left.plot([], [], color="C0", lw=6, alpha=0.6, label="feasible models")
left.plot([], [], color="C3", lw=1.2, label="linearized about the dot")
left.plot([], [], color="C3", ls="--", label="linearized support")
left.plot([], [], color="C0", label="attained by the local search (star)")
left.plot([], [], color="black", ls="--", label="outer bound, by cells")
left.plot([], [], "x", color="black", label="the truth")
left.set_xlim(low_corner[0], high_corner[0])
left.set_ylim(low_corner[1], high_corner[1])
left.set_aspect("equal")
left.set_xlabel("parameter 0")
left.set_ylabel("parameter 1")
left.legend(loc="upper right", fontsize=8)
left.set_title("The feasible models, and the set linearization gives")

# On the right, the whole prior disc and the cells the search made of it:
# parallelograms, their sides along two of the paths.
mesh_x, mesh_y, feasible_mask, _ = on_grid(np.full(2, -1.05), np.full(2, 1.05))
right.contourf(
    mesh_x, mesh_y, feasible_mask, levels=[0.5, 1.5], colors=["C0"], alpha=0.6
)
disc = Circle((0.0, 0.0), prior.radius, facecolor="none", edgecolor="black", lw=1.0)
right.add_patch(disc)
shades = {"empty": "0.8", "pruned": "C2", "live": "C1"}
corners = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
for record in bracket.cells:
    if record.status == "split":
        continue
    outline = [record.cell.point(sign * record.cell.half_widths) for sign in corners]
    patch = Polygon(
        outline,
        facecolor=shades[record.status],
        edgecolor="black",
        lw=0.5,
        alpha=0.5,
    )
    right.add_patch(patch)
    patch.set_clip_path(disc)
for status, shade in shades.items():
    right.plot([], [], color=shade, lw=6, alpha=0.6, label=f"cell: {status}")
right.set_xlim(-1.05, 1.05)
right.set_ylim(-1.05, 1.05)
right.set_aspect("equal")
right.set_xlabel("parameter 0")
right.set_ylabel("parameter 1")
right.legend(loc="lower left", fontsize=8)
right.set_title("The cells: shown empty, pruned, or still standing")

figure, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
counts, outer, inner = (np.array(column) for column in zip(*bracket.history))
ax.step(counts, outer, where="post", color="black", label="outer value")
ax.step(counts, inner, where="post", color="C0", label="attained")
ax.axhline(linearized.support(up), color="C3", ls="--", lw=1.0, label="linearized")
ax.set_xlabel("cells examined")
ax.set_ylabel("support of the property")
ax.set_title("The bracket closing")
ax.legend()

# ---------------------------------------------------------------------------
# A field on an interval, seen through its values and its squares.
# ---------------------------------------------------------------------------
#
# Three data are point values of the field and three are point values of its
# square. A squared datum does not say which sign the field has there, so the
# feasible models can come in several parts, one for each choice of signs the
# prior's smoothness and the other data allow.

rng = np.random.default_rng(5)
H = line.Sobolev(64, 2.0, 0.1)
belief = H.sobolev_measure(2.0, 0.2, pointwise_std=1.0)
field = belief.sample(rng=rng)

value_points, square_points = [0.15, 0.55, 0.85], [0.3, 0.45, 0.7]
evaluation = H.point_evaluation_operator(value_points + square_points, route="dense")
S = evaluation.codomain
squared = np.arange(S.dim) >= len(value_points)


def observe(samples: np.ndarray) -> np.ndarray:
    """The values as they are, and the squares of the rest."""
    return np.where(squared, np.asarray(samples) ** 2, samples)


pointwise = Operator.from_callables(
    S,
    S,
    observe,
    derivative=lambda samples: LinearOperator.from_matrix(
        S,
        S,
        np.diag(np.where(squared, 2.0 * np.asarray(samples), 1.0)),
        form="components",
    ),
)
measurement = pointwise @ evaluation

noise_measure = GaussianMeasure.from_standard_deviation(S, 0.05)
observed = S.add(measurement(field), noise_measure.sample(rng=rng))

# The beliefs hardened into sets, as in example 30: the 95% balls.
bound = belief.ambient_ball(level=0.95)
where = 0.5
average = H.point_evaluation_operator([where])
estimator = NonlinearBackusGilbertParker(
    ForwardProblem(measurement, error=noise_measure), average, bound
)
answer = estimator(observed)
print(f"a field on an interval: dimension {H.dim}, prior ball of radius")
print(f"  {bound.radius:.3f}, noise ball of radius {estimator.noise.radius:.3f}")

# The claim, again a theorem: a value is linear in the field and has no
# remainder; a square has second derivative two, exactly.
rows = [evaluation.adjoint(S.basis_vector(i)) for i in range(S.dim)]
curvature = np.where(squared, 2.0, 0.0)
claim = ComponentwiseRemainder(S, rows, lambda low, high: (curvature, curvature))

start = answer.fitting_model()
flat = BackusGilbertParker(
    ForwardProblem(measurement, error=noise_measure), average, bound, model=start
)(observed)
searched = {sign: answer.support_result(sign * up) for sign in (1.0, -1.0)}
whole = {
    sign: answer.support_result(sign * up, via=BranchAndPrune(claim, rtol=3e-3))
    for sign in (1.0, -1.0)
}
alone = bound.support_function()(average.adjoint(up))

print(f"the value of the field at {where}, which is {average(field)[0]:+.3f} in truth:")
print(f"  the prior alone           [{-alone:+.3f}, {alone:+.3f}]")
print(
    f"  linearized                [{-flat.support(-up):+.3f}, {flat.support(up):+.3f}]"
)
print(
    f"  one local search each way [{-searched[-1.0].inner:+.3f},"
    f" {searched[1.0].inner:+.3f}]"
)
print(
    f"  the whole set, from inside [{-whole[-1.0].inner:+.3f},"
    f" {whole[1.0].inner:+.3f}]"
)
print(
    f"  the whole set, from outside [{-whole[-1.0].value:+.3f},"
    f" {whole[1.0].value:+.3f}]"
)
print(
    f"    {len(whole[1.0].cells)} and {len(whole[-1.0].cells)} cells;"
    f" certified: {whole[1.0].certified and whole[-1.0].certified}"
)
signs_at = H.point_evaluation_operator(square_points)
print()
print("the local search upward stopped in the part of the set it started in,")
print(
    "  where the field's signs at the squared points are"
    f" {np.sign(signs_at(searched[1.0].extremal_model)).astype(int)};"
)
print(
    "  the largest value is in another part, with signs"
    f" {np.sign(signs_at(whole[1.0].extremal_model)).astype(int)}."
)
print("  Nothing in the local search says so. The outer bound does: it is what")
print("  shows there is something beyond the model the search returned.")
print()

figure, ax = plotting.subplots(H, figsize=(9.0, 4.5), layout="constrained")
plotting.plot(H, field, ax=ax, color="black", lw=1.5, label="the truth")
plotting.plot(
    H,
    searched[1.0].extremal_model,
    ax=ax,
    color="C0",
    lw=1.2,
    label="largest value by a local search",
)
plotting.plot(
    H,
    whole[1.0].extremal_model,
    ax=ax,
    color="C1",
    lw=1.2,
    label="largest value over the whole set",
)
ax.plot(value_points, observed[~squared], "o", color="black", label="a value")
for sign in (1.0, -1.0):
    ax.plot(
        square_points,
        sign * np.sqrt(np.maximum(observed[squared], 0.0)),
        "s",
        color="C3",
        mfc="none",
        label="a square, either root" if sign > 0 else None,
    )
ax.axvline(where, color="0.6", lw=0.8, ls=":")
ax.set_xlim(0.0, 1.0)
ax.set_xlabel("position")
ax.set_title("Two models that fit, in different parts of the feasible set")
ax.legend(fontsize=8, loc="upper right")

figure, ax = plt.subplots(figsize=(7.5, 3.2), layout="constrained")
bars = [
    ("the prior alone", -alone, alone, "0.7"),
    ("linearized", -flat.support(-up), flat.support(up), "C3"),
    ("one local search each way", -searched[-1.0].inner, searched[1.0].inner, "C0"),
    ("the whole set, from inside", -whole[-1.0].inner, whole[1.0].inner, "C1"),
    ("the whole set, from outside", -whole[-1.0].value, whole[1.0].value, "black"),
]
for height, (label, low, high, color) in enumerate(reversed(bars)):
    ax.plot([low, high], [height, height], color=color, lw=6, solid_capstyle="butt")
ax.set_yticks(range(len(bars)), [bar[0] for bar in reversed(bars)])
ax.axvline(average(field)[0], color="black", lw=0.8, ls=":")
ax.set_xlabel(f"value of the field at {where}")
ax.set_title("What each answer says the value can be")

print("four figures drawn")

plotting.show()
