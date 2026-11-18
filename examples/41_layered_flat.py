"""
41. Flat layers stacked in depth: a discontinuity, and continuity by
conditioning.

``strip.Layered`` and ``slab.Layered`` stack flat layers in depth, a strip or
a slab in each, all over one lateral period and grid. The field is smooth
within a layer, at that layer's own length scale, and free to jump at an
interface. Under a prior given layer by layer the layers are independent,
which is what a discontinuity means; the jump across an interface is an
operator, and the prior conditioned on its vanishing is continuous there.

Depth is positive downward, and *above* keeps its physical sense: at a point
on an interface ``side="above"`` is the shallower layer, and the jump is the
value above less the value below.

Here a soft sediment over a basement, the sediment's length scale the
shorter. Stations at the surface inform the sediment. Under the independent
prior they say nothing whatever of the basement; with the two layers welded
at their interface they reach into it. On a slab the interface is a plane,
and the jump is taken at ``(x, y)`` positions; a draw is cut by a vertical
plane, and mapped on either side of the interface.

Needs matplotlib.
"""

import matplotlib.pyplot as pyplot
import numpy as np

from pygeoinf import (
    CholeskySolver,
    GaussianMeasure,
    LinearForwardProblem,
    LinearGaussianInversion,
    plotting,
    slab,
    strip,
)

rng = np.random.default_rng(41)

WIDTH, INTERFACE, BASE = 4.0, 0.15, 1.0  # the period, and two depths

# ---------------------------------------------------------------------------
# A layered strip: (depth, x), a sediment over a basement. Every option is
# given once for both layers or, as a list, layer by layer; the lateral grid
# and period are one for both.
# ---------------------------------------------------------------------------

X = strip.Layered(
    [0.0, INTERFACE, BASE],
    41,
    lateral=WIDTH,
    order=2.0,
    length_scale=[0.08, 0.25],
    labels=["sediment", "basement"],
)
print(X)
for label, layer in zip(X.labels, X.layers):
    print(
        f"   {label}: depths {layer.top:g} to {layer.bottom:g}, length scale "
        f"{layer.length_scale:g}, {layer.dim} components"
    )

# Independent layers, each calibrated to its own standard deviation.
prior = X.sobolev_measure(2.0, pointwise_std=[0.5, 1.0])

# The jump across the interface at every point of the lateral grid. The layers
# share the grid, so the jump is a Fourier series with one term per grid point:
# made to vanish at all of them, it vanishes everywhere across. Its normal
# matrix is small, and a Cholesky solve of it makes the conditioned prior cheap
# to apply.
grid = X.layers[0].x
jump = X.jump_operator(0, positions=grid)
welded = prior.condition(jump, np.zeros(grid.size), solver=CholeskySolver())

anywhere = X.jump_operator(0, positions=rng.uniform(0.0, WIDTH, 200))
draws = {}
print()
print("the largest jump of a draw across the interface, at 200 positions:")
for name, measure in (("independent", prior), ("welded", welded)):
    draws[name] = measure.sample(rng=rng)
    print(f"   {name:12s} {np.abs(anywhere(draws[name])).max():.1e}")

# ---------------------------------------------------------------------------
# Surface stations, and how far down they reach under each prior.
# ---------------------------------------------------------------------------

stations = [(0.0, x) for x in np.sort(rng.uniform(0.0, WIDTH, 30))]
forward = X.point_evaluation_operator(stations)
noise = GaussianMeasure.from_standard_deviation(forward.codomain, 0.05)
problem = LinearForwardProblem(forward, error=noise)

across = np.linspace(0.0, WIDTH, 16, endpoint=False)
levels = [(0.0, None), (0.08, None), (INTERFACE, "above"), (INTERFACE, "below")]
levels += [(depth, None) for depth in (0.25, 0.4, 0.6, BASE)]
removed = {}
for name, measure in (("independent", prior), ("welded", welded)):
    _, data = problem.synthetic_model_and_data(measure, rng=rng)
    posterior = LinearGaussianInversion(problem, measure, solver=CholeskySolver())(data)
    removed[name] = []
    for depth, side in levels:
        # At a depth on the interface, the side says which layer is meant.
        look = X.point_evaluation_operator([(depth, x) for x in across], side=side)
        before = np.diag((look @ measure.covariance @ look.adjoint).matrix())
        after = np.diag((look @ posterior.covariance @ look.adjoint).matrix())
        removed[name].append(1.0 - after.sum() / before.sum())

print()
print("the share of the variance the surface data remove, level by level:")
print("                      independent   welded")
for (depth, side), lone, joined in zip(levels, *removed.values()):
    where = f"{depth:.2f}" + (f" ({side})" if side else "")
    print(f"   depth {where:13s}    {lone:5.2f}      {joined:5.2f}")
print("independent, the basement learns nothing; welded, the data reach into it")

# ---------------------------------------------------------------------------
# Depth sections of the two draws. plot draws a layered strip layer by layer,
# each layer from its own values and all on one color scale, so that a jump
# at the interface, which is marked, is drawn as one and not smeared.
# ---------------------------------------------------------------------------

limit = max(
    float(np.abs(layer.interior_values(part)).max())
    for draw in draws.values()
    for layer, part in zip(X.layers, draw)
)
figure, panels = pyplot.subplots(3, 1, figsize=(9.0, 8.0), layout="constrained")
for axis, (name, draw) in zip(panels, draws.items()):
    _, image = plotting.plot(
        X,
        draw,
        ax=axis,
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        colorbar=False,
        title=f"A draw of the {name} prior",
    )
figure.colorbar(image, ax=panels[:2], label="field")
depths = [depth for depth, _ in levels]
for name, values in removed.items():
    panels[2].plot(values, depths, marker="o", label=name)
panels[2].axhline(INTERFACE, color="k", linewidth=0.8, linestyle="--")
panels[2].set_ylim(BASE, 0.0)
panels[2].set_xlabel("share of the variance removed by the surface data")
panels[2].set_ylabel("depth")
panels[2].legend()
print()
print("two depth sections and a profile drawn")

# ---------------------------------------------------------------------------
# A layered slab: the interface is a plane, and the jump is taken at (x, y).
# ---------------------------------------------------------------------------

Y = slab.Layered(
    [0.0, INTERFACE, BASE],
    (15, 15),
    lateral=(3.0, 3.0),
    order=2.0,
    length_scale=[0.12, 0.25],
)
print()
print(Y)
box = Y.sobolev_measure(2.0, pointwise_std=[0.5, 1.0])
x, y = np.meshgrid(*Y.layers[0].lateral_axes, indexing="ij")
plane = Y.jump_operator(0, positions=np.column_stack([x.ravel(), y.ravel()]))
joined = box.condition(plane, np.zeros(x.size), solver=CholeskySolver())
scattered = Y.jump_operator(0, positions=rng.uniform(0.0, 3.0, (200, 2)))
boxed = {}
for name, measure in (("independent", box), ("welded", joined)):
    boxed[name] = measure.sample(rng=rng)
    print(
        f"   {name:12s} largest jump of a draw at 200 positions "
        f"{np.abs(scattered(boxed[name])).max():.1e}"
    )

# Each draw cut by a vertical plane, along x through the middle of the box,
# and the independent one's two lateral maps at the interface: a map at the
# depth of an interface says which side it means, above the shallower layer.
figure, panels = plotting.subplots(Y, rows=2, columns=2, figsize=(10.0, 7.0))
for axis, (name, draw) in zip(panels[0], boxed.items()):
    plotting.plot_section(
        Y, draw, y=1.5, ax=axis, symmetric=True, title=f"The {name} draw at y = 1.5"
    )
for axis, side in zip(panels[1], ("above", "below")):
    plotting.plot(
        Y,
        boxed["independent"],
        depth=INTERFACE,
        side=side,
        ax=axis,
        symmetric=True,
        title=f"The independent draw just {side} the interface",
    )
print("a layered slab drawn: two depth sections, and the two sides of its interface")
plotting.show()
