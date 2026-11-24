"""
43. Travel times through the exponential of a field: a certified set on a function space.

Example 42 bracketed a nonlinear feasible set by cutting the prior into
cells. There the forward map curved along a handful of directions, and the
cells could be thin in all of them. Here the unknown is a field on a
square, ``m = log(slowness)``, and a datum is a travel time along a straight
ray,

    d_i = integral over ray i of exp(m) ds,

so the map curves along every variation of the field on a ray: no finite
set of directions holds its curvature. Two things make a certified bound
possible all the same.

The first is where the remainder is bounded from. The model space is a
Sobolev space on which a point value is a bounded functional, so a model of
a given norm cannot be large anywhere, and the remainder of the exponential
along a ray is bounded by the cell's reach at each node of the quadrature:
through the cell's slabs along the directions they are cut in, and through
its radius across them.

The second is that the data, not halving, narrow the slabs. Along a
direction the rays sample well, the feasible models of a cell lie within a
range far smaller than the prior allows; cutting the slab back to that
range shrinks the remainder, which shrinks the range again. One cell,
tightened so, is the whole computation here.

Needs matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from pygeoinf import (
    BackusGilbertParker,
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
from pygeoinf.symmetric_space import plane

rng = np.random.default_rng(1)

# ---------------------------------------------------------------------------
# The problem: a field on a square, rays across it, travel times.
# ---------------------------------------------------------------------------

X = plane.Sobolev((16, 16), 3.0, 0.25, bounds=((0.0, 1.0), (0.0, 1.0)))
belief = X.sobolev_measure(3.0, 0.25, pointwise_std=0.05)
truth = belief.sample(rng=rng)


def on_the_edge() -> np.ndarray:
    """A point just inside one of the square's four sides."""
    along = rng.uniform(0.05, 0.95)
    return np.array(
        [(along, 0.02), (along, 0.98), (0.02, along), (0.98, along)][rng.integers(4)]
    )


paths = []
while len(paths) < 30:
    start, end = on_the_edge(), on_the_edge()
    if np.linalg.norm(start - end) > 0.5:
        paths.append((start, end))

# The quadrature written out: the field at the nodes of every ray, the
# exponential of each value, and the weights that sum a ray's nodes.
nodes, lengths, owner = [], [], []
for ray, (start, end) in enumerate(paths):
    points, weights = X.geodesic_quadrature(start, end, count=12)
    nodes += list(points)
    lengths += list(weights)
    owner += [ray] * len(points)
evaluation = X.point_evaluation_operator(nodes, route="dense")
N, D = evaluation.codomain, EuclideanSpace(len(paths))
weight_matrix = np.zeros((D.dim, N.dim))
weight_matrix[owner, np.arange(N.dim)] = lengths
integrate = LinearOperator.from_matrix(N, D, weight_matrix, form="components")


def exponential(values: np.ndarray) -> np.ndarray:
    """The slowness at the nodes; a search far from the prior is told so."""
    values = np.asarray(values)
    if np.any(values > 50.0):
        raise EvaluationError("The exponential overflows at this model.")
    return np.exp(values)


pointwise = Operator.from_callables(
    N,
    N,
    exponential,
    derivative=lambda values: LinearOperator.from_matrix(
        N, N, np.diag(np.exp(np.asarray(values))), form="components"
    ),
)
forward = integrate @ pointwise @ evaluation

clean = forward(truth)
signal = clean - weight_matrix.sum(axis=1)
noise = GaussianMeasure.from_standard_deviation(D, 0.1 * float(np.std(signal)))
data = clean + noise.sample(rng=rng)

# The property: the field at the middle of the square.
where = np.array([0.5, 0.5])
T = X.point_evaluation_operator([where])
up = np.array([1.0])

prior = belief.ambient_ball(level=0.95)
problem = ForwardProblem(forward, error=noise)
feasible = NonlinearBackusGilbertParker(problem, T, prior)(data)
print(f"a field of dimension {X.dim} seen along {len(paths)} rays;")
print(f"  prior ball of radius {prior.radius:.4f}, noise ball of radius")
print(f"  {feasible.noise.radius:.4f}; the travel times depart from the rays'")
print(f"  lengths by {np.std(signal):.4f} (root mean square)")
print()

# ---------------------------------------------------------------------------
# Three uncertified answers, then the certified one.
# ---------------------------------------------------------------------------

alone = prior.support_function()(T.adjoint(up))
anchor = feasible.fitting_model()
flat = BackusGilbertParker(problem, T, prior, model=anchor)(data)
searched = {sign: feasible.support_result(sign * up) for sign in (1.0, -1.0)}

# The claim. A node's value enters through the exponential, whose second
# derivative over an interval lies between its values at the ends; the
# weights carry the nodes' remainders to the rays. It is a theorem about
# this map, the quadrature being part of the map's definition.
claim = ComponentwiseRemainder(
    D, evaluation, lambda low, high: (np.exp(low), np.exp(high)), then=integrate
)

# The slab directions: the models the rays sample most, the leading
# eigenvectors of B* W B, with B the evaluation and W the weights.
sampled = evaluation.matrix(form="components")
coverage = sampled.T @ (np.array(lengths)[:, None] * sampled)
metric = LinearOperator.identity(X).matrix(form="galerkin")
values, vectors = eigh(coverage, metric)
leading = np.argsort(-values)[:4]
directions = [X.from_components(vectors[:, index]) for index in leading]

root = BranchAndPrune(claim, directions=directions).root(feasible)
tight = feasible.tightened_cell(root, claim, sweeps=2)
whole = {
    sign: feasible.cell_support_result(sign * up, root, claim) for sign in (1.0, -1.0)
}
bound = {
    sign: feasible.cell_support_result(sign * up, tight, claim) for sign in (1.0, -1.0)
}

print(f"the field at the middle of the square, {T(truth)[0]:+.4f} in truth:")
rows = [
    ("the prior alone", -alone, alone),
    ("linearized about a fitting model", -flat.support(-up), flat.support(up)),
    ("attained by local searches", -searched[-1.0].inner, searched[1.0].inner),
    ("certified, the prior as one cell", -whole[-1.0].value, whole[1.0].value),
    ("certified, the cell tightened", -bound[-1.0].value, bound[1.0].value),
]
for label, low, high in rows:
    print(f"  {label:34s} [{low:+.4f}, {high:+.4f}]")
print(
    f"    certified: {bound[1.0].certified and bound[-1.0].certified},"
    f" refuted: {bound[1.0].refuted or bound[-1.0].refuted}"
)
print()
before, after = claim.remainder_set(root).radius, claim.remainder_set(tight).radius
size = feasible.noise.radius
print(
    f"the remainder set's radius is {before / size:.1f} times the noise ball's over the"
)
print(f"  whole prior, and {after / size:.2f} times over the tightened cell: the")
print("  linearized problem's noise is grown by that much to hold every feasible")
print("  model. The linearized interval sits inside the certified one: here the")
print("  map is mild enough for linearization to be nearly right, and the")
print("  certified bound is what says how nearly.")
print()

# ---------------------------------------------------------------------------
# Three figures: the problem, the slabs, the intervals.
# ---------------------------------------------------------------------------

figure, ax = plotting.subplots(X, figsize=(5.5, 5.0), layout="constrained")
plotting.plot(X, truth, ax=ax, cmap="RdBu_r", symmetric=True, colorbar_label="m")
for start, end in paths:
    ax.plot([start[0], end[0]], [start[1], end[1]], color="black", lw=0.5, alpha=0.5)
ax.plot(*where, "*", color="black", ms=14)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_title("The field, the rays, and the point asked about")

figure, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
index = np.arange(len(directions))
ax.bar(index - 0.2, root.half_widths, width=0.4, color="0.7", label="the prior's reach")
ax.bar(index + 0.2, tight.half_widths, width=0.4, color="C0", label="after tightening")
ax.set_yscale("log")
ax.set_xticks(index, [f"{i + 1}" for i in index])
ax.set_xlabel("slab direction, by how well the rays sample it")
ax.set_ylabel("half-width of the slab")
ax.set_title("The data narrow the slabs; no cell is split")
ax.legend()

figure, ax = plt.subplots(figsize=(7.5, 3.2), layout="constrained")
colors = ["0.7", "C3", "C0", "0.3", "black"]
for height, ((label, low, high), color) in enumerate(
    zip(reversed(rows), reversed(colors))
):
    ax.plot([low, high], [height, height], color=color, lw=6, solid_capstyle="butt")
ax.set_yticks(range(len(rows)), [row[0] for row in reversed(rows)])
ax.axvline(T(truth)[0], color="black", lw=0.8, ls=":")
ax.set_xlabel("the field at the middle of the square")
ax.set_title("What each answer says the value can be")

print("three figures drawn")

plotting.show()
