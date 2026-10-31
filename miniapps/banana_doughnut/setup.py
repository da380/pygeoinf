"""Part A: the rays, the Hessians across them and the paraxial kernels.

At each distance, the ray from the source to the receiver and the sums
``A1``, ``A2`` of the Hessians of the travel time from its two ends, by
dynamic ray tracing; then, for each band, the finite-frequency kernel of the
traveltime as the ``L2`` representer of the functional, drawn from the
paraxial formula ``K = -sqrt(A1 A2) G(u) / (2 pi c)`` on two sections: the
ray's own plane, and the plane across the ray at mid-path. The kernel is
zero on the ray, the hollow banana, and in the plane across it a set of
rings about the ray, the doughnut. Points outside the mantle annulus are
left blank: the functional drops them.
"""

from __future__ import annotations

import pathlib

import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

from src.common import parse, radius_contours, save, symmetric_mesh
from src.functionals import kernel
from src.planet import CORE_RADIUS, RADIUS, Path, trace
from src.problem import FULL, QUICK, Problem, Scale, build
from src.pulse import Band

#: The nodes along each ray in the drawings; the pixels across the drawing
#: in the ray's plane; the offsets across the ray in each direction in the
#: drawing across it; and how far from the ray both reach, in Fresnel
#: half-widths at mid-path.
DRAWN_NODES = 401
PIXELS = 600
DRAWN_OFFSETS = 241
REACH = 3.0
#: The label of the kernel's colorbar.
KERNEL_LABEL = "K, time per unit volume"


def _turned(points: np.ndarray, distance: float, /) -> tuple[np.ndarray, np.ndarray]:
    """Points of the ray plane turned so that the ray's mid-path direction
    is up: the source on the left, the receiver on the right."""
    angle = np.radians(90.0 - 0.5 * distance)
    x, y = points[..., 0], points[..., 1]
    return (
        np.cos(angle) * x - np.sin(angle) * y,
        np.sin(angle) * x + np.cos(angle) * y,
    )


def _unturned(x: np.ndarray, y: np.ndarray, distance: float, /) -> np.ndarray:
    """The inverse of :func:`_turned`: points of the ray plane, ``(..., 2)``."""
    angle = np.radians(90.0 - 0.5 * distance)
    return np.stack(
        [np.cos(angle) * x + np.sin(angle) * y, -np.sin(angle) * x + np.cos(angle) * y],
        axis=-1,
    )


def _arcs(ax, distance: float, /) -> None:
    """The surface and the core-mantle boundary in the ray's plane."""
    t = np.radians(np.linspace(-0.25 * distance, 1.25 * distance, 400))
    for radius in (CORE_RADIUS, RADIUS):
        arc = radius * np.column_stack([np.cos(t), np.sin(t)])
        ax.plot(*_turned(arc, distance), color="0.3", lw=0.8)


def in_plane_kernel(
    path: Path, band: Band, points: np.ndarray, /, *, scale: Scale
) -> tuple[np.ndarray, np.ndarray]:
    """The kernel at points of the ray's plane, ``(..., 2)``, and each
    point's arc length along the ray.

    Each point's nearest point of the ray is found from its nearest node,
    by the offset along the tangent there, which is the arc length of the
    foot of the normal through it, and the offset along the normal, which
    is ``q1``; the sums and the velocity are interpolated to that arc
    length, the sums through their reciprocals, which vanish at the ends
    like the arc length. Points whose foot lies beyond an end, or whose
    nearest node is an end, are NaN; beyond the largest detour time the
    quadrature keeps, the kernel is zero, as there.
    """
    dynamic = path.dynamic
    frame = dynamic.frame
    flat = points.reshape(-1, 2)
    _, nearest = cKDTree(frame.point[:, :2]).query(flat)
    offset = flat - frame.point[nearest, :2]
    arc = dynamic.arc_length[nearest] + np.einsum(
        "ij,ij->i", offset, frame.tangent[nearest, :2]
    )
    q1 = np.einsum("ij,ij->i", offset, frame.in_plane[nearest, :2])
    nodes = dynamic.arc_length
    reciprocal = 1.0 / dynamic.sums
    c = np.interp(arc, nodes, dynamic.velocity)
    with np.errstate(divide="ignore", invalid="ignore"):
        a1 = 1.0 / np.interp(arc, nodes, reciprocal[:, 0])
        a2 = 1.0 / np.interp(arc, nodes, reciprocal[:, 1])
        detour = 0.5 * a1 * q1**2
        values = (
            -np.sqrt(a1 * a2)
            * band.kernel_function(detour, frequencies=scale.frequency_nodes)
            / (2.0 * np.pi * c)
        )
    values[detour > band.cutoff(scale.cutoff)] = 0.0
    ends = (nearest == 0) | (nearest == len(nodes) - 1)
    values[ends | (arc <= 0.0) | (arc >= nodes[-1])] = np.nan
    shape = points.shape[:-1]
    return values.reshape(shape), arc.reshape(shape)


def ray_plane(ax, path: Path, band: Band, /, *, scale: Scale) -> None:
    """The kernel in the ray's plane on a regular grid, with the ray as a
    line; colored to the largest size it reaches away from the ends, the
    tenth of the ray at each, where it grows without bound."""
    width = path.fresnel_half_width(band.period)
    frame = path.dynamic.frame
    xs, ys = _turned(frame.point, path.distance)
    margin = REACH * width
    x = np.linspace(xs.min() - margin, xs.max() + margin, PIXELS)
    step = x[1] - x[0]
    y = np.arange(ys.min() - margin, RADIUS + step, step)
    gx, gy = np.meshgrid(x, y)
    points = _unturned(gx, gy, path.distance)
    values, arc = in_plane_kernel(path, band, points, scale=scale)
    radii = np.linalg.norm(points, axis=-1)
    values[(radii < CORE_RADIUS) | (radii > RADIUS)] = np.nan
    away = (arc > 0.1 * path.length) & (arc < 0.9 * path.length)
    limit = float(np.nanmax(np.abs(values[away])))
    symmetric_mesh(ax, x, y, values, limit=limit, label=KERNEL_LABEL)
    ax.plot(xs, ys, color="black", lw=0.8)
    _arcs(ax, path.distance)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_title(
        f"T0 = {band.period:g}, in the ray's plane: Fresnel half-width "
        f"{width:.3f} at mid-path",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def across(ax, path: Path, band: Band, /, *, scale: Scale) -> None:
    """The kernel in the plane across the ray at mid-path, on offsets
    ``q2`` out of the ray's plane and ``q1`` in it, with the ray at the
    center."""
    width = path.fresnel_half_width(band.period)
    middle = int(np.argmin(np.abs(path.dynamic.arc_length - 0.5 * path.length)))
    offsets = np.linspace(-REACH * width, REACH * width, DRAWN_OFFSETS)
    q2, q1 = np.meshgrid(offsets, offsets)
    values = kernel(
        path,
        band,
        q1[None],
        q2[None],
        nodes=[middle],
        frequencies=scale.frequency_nodes,
        cutoff=scale.cutoff,
    )[0]
    frame = path.dynamic.frame
    points = (
        frame.point[middle]
        + q1[..., None] * frame.in_plane[middle]
        + q2[..., None] * frame.out_of_plane[middle]
    )
    radii = np.linalg.norm(points, axis=-1)
    limit = float(np.abs(values).max())
    values[(radii < CORE_RADIUS) | (radii > RADIUS)] = np.nan
    symmetric_mesh(ax, q2, q1, values, limit=limit, label=KERNEL_LABEL)
    radius_contours(ax, q2, q1, radii)
    ax.plot(0.0, 0.0, "k+", ms=8)
    ax.set_xlabel("offset across the ray's plane, q2", fontsize=8)
    ax.set_ylabel("offset in it, q1 (up)", fontsize=8)
    ax.set_title(f"T0 = {band.period:g}, across the ray at mid-path", fontsize=9)


def kernels_figure(
    problem: Problem, index: int, /, *, outdir: pathlib.Path
) -> pathlib.Path:
    """The kernel of one ray in every band, on its two sections."""
    scale = problem.scale
    path = trace(
        problem.paths[index].distance,
        nodes=DRAWN_NODES,
        source_depth=scale.source_depth,
    )
    rows = len(problem.bands)
    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(11.0, 4.2 * rows),
        gridspec_kw={"width_ratios": [1.5, 1.0]},
        squeeze=False,
        layout="constrained",
    )
    for (left, right), band in zip(axes, problem.bands):
        across(right, path, band, scale=scale)
        ray_plane(left, path, band, scale=scale)
    name = f"kernels_{path.distance:g}deg.png"
    save(
        fig,
        outdir / name,
        title=f"Banana-doughnut, part A: the paraxial L2 kernel of the traveltime at "
        f"{path.distance:g} degrees,\nin the ray's plane (saturated near the ends) "
        "and across the ray at mid-path; blank outside the mantle",
    )
    return outdir / name


def rays_figure(problem: Problem, /, *, outdir: pathlib.Path) -> pathlib.Path:
    """The rays in the equatorial plane, and the sums of the Hessians along
    them against the straight chord's."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 5.0))
    t = np.linspace(0.0, 2.0 * np.pi, 400)
    for radius in (CORE_RADIUS, RADIUS):
        left.plot(radius * np.cos(t), radius * np.sin(t), color="0.3", lw=0.8)
    for k, path in enumerate(problem.paths):
        color = f"C{k}"
        point = path.dynamic.frame.point
        left.plot(point[:, 0], point[:, 1], color=color, label=f"{path.distance:g}°")
        arc = path.dynamic.arc_length
        length = path.length
        inner = slice(1, -1)
        chord = (
            path.dynamic.velocity[inner] * arc[inner] * (length - arc[inner]) / length
        )
        for which, style, name in ((0, "-", "in the plane"), (1, "--", "across it")):
            right.plot(
                arc[inner] / length,
                path.dynamic.sums[inner, which] * chord,
                style,
                color=color,
                label=f"{path.distance:g}°, {name}",
            )
    left.plot(1.0, 0.0, "*", color="C3", ms=12, label="source")
    left.set_aspect("equal")
    left.set_xticks([])
    left.set_yticks([])
    left.legend(fontsize=8, loc="lower left")
    left.set_title("The rays in the equatorial plane, the core below 0.55", fontsize=10)
    right.axhline(1.0, color="0.5", lw=0.8)
    right.set_xlabel("arc length from the source, l / L")
    right.set_ylabel("A_i c l (L - l) / L")
    right.legend(fontsize=7, ncol=2)
    right.set_title(
        "The sums of the Hessians, against a straight ray at constant speed (one)",
        fontsize=10,
    )
    path = outdir / "rays.png"
    save(
        fig,
        path,
        title="Banana-doughnut, part A: the rays, and the Hessians of the travel "
        "time across them from both ends",
    )
    return path


def run(scale: Scale, /, *, outdir: pathlib.Path) -> Problem:
    problem = build(scale=scale)
    rays_figure(problem, outdir=outdir)
    for index in range(len(problem.paths)):
        kernels_figure(problem, index, outdir=outdir)
    return problem


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
