"""Part F: the same planet in a ball, with its stations spread over the sphere.

The questions of B, C and D asked in three dimensions, at a toy size. A
dozen sources and three dozen receivers, drawn once from a fixed generator:
thirty receivers over a continent, eight on islands far from it, and the
sources spread evenly over the whole sphere. Every source-receiver pair
between 20 and 80 degrees apart is a ray. A ray lies in the plane through
its source, its receiver and the center, so the radial tracer of the disk
traces it unchanged, the angular distance being the great-circle distance,
and its nodes are placed along that great circle at the traced radii.

The model is a layered ball of degree 14, the truth a few bumps of the
slowness at different depths, latitudes and longitudes, one of them in the
core, and the three properties the means over a box of the mantle under the
continent, over a box of the same size at its antipode, and over the core.
The figures cut the ball three ways: by the equatorial plane; by the
meridian through the continent, which on the far side of the center passes
through its antipode; and by the meridian across them. Each section shows
the parts of the rays within a few degrees of its plane, and the shell at
mid-mantle depth shows where the rays run beneath it. A cut-away shows the
truth's mantle beside the posterior mean in the ball cut in half by the
plane of the meridian through the continent.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
from typing import Any, Sequence

import numpy as np

import matplotlib.pyplot as plt
from cartopy import crs
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from pygeoinf import GaussianMeasure, LinearForwardProblem, plotting
from pygeoinf.symmetric_space.sphere import Lebesgue as Sphere

from bayesian import exact_pointwise_std, posterior, property_posterior
from point import error_budget, minimum_norm
from sets import brackets_figure, support_brackets
from src.common import (
    FIXED,
    error_budget_panels,
    marginal_panels,
    parse,
    part_rng,
    save,
)
from src.problem import (
    CORE_RADIUS,
    MANTLE,
    QUICK,
    RADIUS,
    Problem,
    Scale,
    sobolev_prior,
    inner_truncation,
    model_space,
    path_integral_operator,
    prior_pointwise_std,
    property_operator,
)
from src.rays import Ray

# The network, drawn once from a generator of its own so that it is the same
# whatever seed the noise is drawn with: a continent of receivers about
# CONTINENT, islands at least ISLANDS degrees from its center, and sources
# spread evenly over the sphere.
NETWORK_SEED = 3
CONTINENT = (10.0, 30.0)
CONTINENT_RADIUS = 35.0
ISLANDS = 60.0
COUNTS = {"sources": 12, "continent": 30, "islands": 8}

#: The properties: the mean over a box of the mantle under the continent and
#: over one the same size at its antipode, ``(r1, r2, latitude1, latitude2,
#: longitude1, longitude2)`` in degrees, and over the core.
BOXES = (
    ("well covered", (0.75, 0.9, -5.0, 25.0, 15.0, 45.0)),
    ("poorly covered", (0.75, 0.9, -25.0, 5.0, -165.0, -135.0)),
)
PROPERTIES = ("well covered", "poorly covered", "core")
NAMES = (
    "mean over the box under the continent",
    "mean over the box at its antipode",
    "mean over the core",
)

#: The truth: bumps of the slowness perturbation, in percent, each an
#: amplitude, a center ``(radius, latitude, longitude)`` and a width; the
#: last is in the core.
MANTLE_BUMPS = (
    (2.4, (0.82, 15.0, 35.0), 0.15),
    (-2.4, (0.70, -20.0, -150.0), 0.175),
    (1.9, (0.92, 45.0, -100.0), 0.125),
    (-1.9, (0.78, -35.0, 95.0), 0.15),
)
CORE_BUMP = (1.6, (0.3, 20.0, 160.0), 0.19)

#: The sections, each named and given as ``plot_section`` takes its plane.
SECTIONS = (
    ("the equatorial plane", {"pole": (90.0, 0.0)}),
    ("the meridian of 30°", {"longitude": 30.0}),
    ("the meridian of 120°", {"longitude": 120.0}),
)
#: How far from a section's plane, in degrees from the center, the parts of
#: the rays drawn on it may lie.
SLAB = 6.0
#: The shell drawn as a map: mid-mantle, through the middle of the boxes.
MID_MANTLE = 0.825
#: The cut-away: the half of the ball east of the meridian of -150° and west
#: of the meridian of 30° removed at every latitude, so that the cut face is
#: the plane of the meridian through the continent: the box under it and its
#: bump on one side of the center, the box at its antipode and its bump on
#: the other. Seen from 25° north and 50° west of the middle of the half
#: removed, so that the half left shows beside the face; the surface and the
#: core plain, the mantle's field on the face alone.
CUTAWAY_LONGITUDES = (CONTINENT[1] - 180.0, CONTINENT[1])
CUTAWAY_VIEW = (25.0, CONTINENT[1] - 140.0)
SURFACE_COLOR = "0.85"
CORE_COLOR = "0.6"

#: Where the standard deviation is read exactly: the middle of each box, the
#: middle of the core, and the mantle under the stations near the surface.
PROBES = (
    ("well covered", (0.825, 10.0, 30.0)),
    ("poorly covered", (0.825, -10.0, -150.0)),
    ("core", (0.2, 0.0, 0.0)),
    ("under the stations", (0.95, 10.0, 30.0)),
)


def unit(latitude: Any, longitude: Any, /) -> np.ndarray:
    """Unit vectors of directions given in degrees."""
    b, a = np.radians(latitude), np.radians(longitude)
    return np.stack([np.cos(b) * np.cos(a), np.cos(b) * np.sin(a), np.sin(b)], -1)


def direction(vectors: np.ndarray, /) -> np.ndarray:
    """``(latitude, longitude)`` in degrees of vectors, as a ``(count, 2)``
    array."""
    vectors = np.atleast_2d(vectors)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.column_stack(
        [
            np.degrees(np.arcsin(np.clip(vectors[:, 2], -1.0, 1.0))),
            np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])),
        ]
    )


def apart(first: Any, second: Any, /) -> float:
    """The great-circle distance in degrees between two ``(latitude,
    longitude)`` pairs."""
    return float(np.degrees(np.arccos(np.clip(unit(*first) @ unit(*second), -1, 1))))


def stations() -> (
    tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]
):
    """The sources and the receivers, as ``(latitude, longitude)`` pairs.

    The receivers are uniform over the continent's cap and over the sphere
    beyond ``ISLANDS`` degrees from its center; the sources are a Fibonacci
    lattice, as even as a dozen points can be, turned at random.
    """
    rng = np.random.default_rng(NETWORK_SEED)

    def uniform(count: int, keep: Any) -> list[np.ndarray]:
        kept: list[np.ndarray] = []
        while len(kept) < count:
            z, phi = rng.uniform(-1.0, 1.0), rng.uniform(-np.pi, np.pi)
            point = np.array(
                [np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z]
            )
            if keep(apart(direction(point)[0], CONTINENT)):
                kept.append(point)
        return kept

    receivers = uniform(COUNTS["continent"], lambda d: d <= CONTINENT_RADIUS)
    receivers += uniform(COUNTS["islands"], lambda d: d > ISLANDS)
    count = COUNTS["sources"]
    i = np.arange(count) + 0.5
    z = 1.0 - 2.0 * i / count
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    lattice = np.column_stack(
        [np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z]
    )
    turn, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    sources = lattice @ turn.T

    def as_pairs(vectors: Any) -> tuple[tuple[float, float], ...]:
        return tuple((float(b), float(a)) for b, a in direction(np.asarray(vectors)))

    return as_pairs(sources), as_pairs(receivers)


SOURCES, RECEIVERS = stations()

#: The ball: the network above, degree 14, and a length scale wide enough
#: for the truncation; the physics, the noise and the distances the disk's.
BALL = dataclasses.replace(
    QUICK,
    name="ball",
    mmax=14,
    length_scale=0.2,
    sources=SOURCES,
    receivers=RECEIVERS,
    nodes=121,
    std_samples=128,
    n_jobs=None,
    bump_width=0.15,
    geometry="ball",
)
#: The ball's fields that are not options: its stations, points of the
#: sphere drawn once above, besides the disk's.
BALL_FIXED = FIXED | {"sources", "receivers"}


def network(
    sources: Sequence[tuple[float, float]],
    receivers: Sequence[tuple[float, float]],
    /,
    *,
    min_distance: float,
    max_distance: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Every source-receiver pair between ``min_distance`` and
    ``max_distance`` degrees apart."""
    return [
        (source, receiver)
        for source in sources
        for receiver in receivers
        if min_distance <= apart(source, receiver) <= max_distance
    ]


def trace(
    source: tuple[float, float], receiver: tuple[float, float], /, *, nodes: int
) -> Ray:
    """The ray from a source to a receiver, its nodes ``(radius, latitude,
    longitude)``.

    The radial tracer traces it in its own plane, from angle zero to the
    great-circle distance; each node keeps its radius and goes to the point
    of the great circle that far round from the source.
    """
    distance = apart(source, receiver)
    ray = MANTLE.trace(0.0, distance, nodes=nodes)
    start, end = unit(*source), unit(*receiver)
    reach, turned = np.radians(distance), np.radians(ray.nodes[:, 1])
    along = (
        np.sin(reach - turned)[:, None] * start + np.sin(turned)[:, None] * end
    ) / np.sin(reach)
    placed = np.column_stack([ray.nodes[:, 0], direction(along)])
    return Ray(placed, ray.weights, ray.travel_time)


def bump_field(space: Any, /) -> Any:
    """The truth: the mantle's bumps in the mantle and the core's in the core."""

    def bumps(entries: Sequence[tuple[float, tuple, float]]) -> Any:
        centers = [
            (amplitude, center[0] * unit(center[1], center[2]), width)
            for amplitude, center, width in entries
        ]

        def value(point: Sequence[float]) -> float:
            where = point[0] * unit(point[1], point[2])
            return float(
                sum(
                    amplitude * np.exp(-0.5 * np.sum((where - middle) ** 2) / width**2)
                    for amplitude, middle, width in centers
                )
            )

        return value

    return space.project_function([bumps([CORE_BUMP]), bumps(MANTLE_BUMPS)])


def build(*, scale: Scale = BALL, quiet: bool = False) -> Problem:
    """The network, the rays, the space, the prior, the truth and the data."""
    started = time.perf_counter()
    rng = np.random.default_rng([scale.seed, 0])
    say = (lambda *args: None) if quiet else print
    t0 = time.perf_counter()
    pairs = network(
        scale.sources,
        scale.receivers,
        min_distance=scale.min_distance,
        max_distance=scale.max_distance,
    )
    rays = [trace(source, receiver, nodes=scale.nodes) for source, receiver in pairs]
    deepest = min(float(ray.nodes[:, 0].min()) for ray in rays)
    say(
        f"rays: {len(scale.sources)} sources, {len(scale.receivers)} receivers "
        f"({COUNTS['continent']} over a continent of radius {CONTINENT_RADIUS:g} "
        f"degrees about {CONTINENT}, {COUNTS['islands']} islands), {len(rays)} "
        f"pairs between {scale.min_distance:g} and {scale.max_distance:g} degrees "
        f"apart, "
        f"{scale.nodes} nodes each, traced in {time.perf_counter() - t0:.1f} s; the "
        f"deepest turns at r = {deepest:.3f}"
    )
    space = model_space(scale)
    prior = sobolev_prior(space, order=scale.prior_order, std=scale.prior_std)
    core, mantle = space.layers
    say(
        f"space: the layered ball, core {core.dim} and mantle {mantle.dim} "
        f"coefficients (lmax {inner_truncation(core)} and "
        f"{inner_truncation(mantle)}), H^{scale.order:g} with length scale "
        f"{scale.length_scale}; prior: a Sobolev measure of order "
        f"{scale.prior_order:g} on each layer, pointwise std {scale.prior_std:g}%"
    )
    t0 = time.perf_counter()
    forward = path_integral_operator(scale, space, rays)
    say(
        f"forward: {len(rays)} rows from {len(rays) * scale.nodes} node evaluations "
        f"in {time.perf_counter() - t0:.1f} s"
    )
    noise = GaussianMeasure.from_standard_deviation(forward.codomain, scale.noise_std)
    problem = LinearForwardProblem(forward, error=noise)
    truth = bump_field(space)
    data = problem.synthetic_data(truth, rng=rng)
    say(
        f"truth: {len(MANTLE_BUMPS)} bumps in the mantle and one in the core, norm "
        f"{space.norm(truth):.2f}; travel-time perturbations of std "
        f"{np.std(forward(truth)):.3f} against noise of {scale.noise_std}; "
        f"chi-squared of "
        f"the truth {problem.chi_squared(truth, data):.1f} on {len(rays)} data"
    )
    properties = property_operator(scale, space, boxes=BOXES)
    true_properties = properties(truth)
    for name, value in zip(PROPERTIES, true_properties):
        say(f"  true mean perturbation, {name}: {value:+.3f}")
    say(f"setup took {time.perf_counter() - started:.1f} s")
    return Problem(
        scale,
        space,
        prior,
        pairs,
        rays,
        forward,
        problem,
        truth,
        data,
        properties,
        true_properties,
    )


# --------------------------------------------------------------------------- #
#                                  Drawing                                    #
# --------------------------------------------------------------------------- #


def plane_frame(plane: dict, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A section's unit normal and the two axes of its drawing.

    As ``plot_section`` draws it: the plane seen from its pole, the
    direction in it nearest north up the page; a meridian's own longitude on
    the right; the equatorial plane seen from above the north pole, longitude
    zero on the right and ninety at the top.
    """
    if "longitude" in plane:
        normal = unit(0.0, plane["longitude"] - 90.0)
    else:
        normal = unit(*plane["pole"])
    north = np.array([0.0, 0.0, 1.0])
    up = north - (north @ normal) * normal
    if np.linalg.norm(up) < 1e-9:
        return (
            normal,
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]) * np.sign(normal[2]),
        )
    up = up / np.linalg.norm(up)
    return normal, np.cross(up, normal), up


def in_plane(points: np.ndarray, plane: dict, /) -> tuple[np.ndarray, np.ndarray]:
    """Points ``(radius, latitude, longitude)`` where a section draws them, at
    their own radius over their foot in the plane, and how far off the plane
    each lies, in degrees from the center."""
    normal, right, up = plane_frame(plane)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    directions = unit(points[:, 1], points[:, 2])
    off = np.degrees(np.arcsin(np.clip(np.abs(directions @ normal), 0.0, 1.0)))
    feet = directions - np.outer(directions @ normal, normal)
    feet = feet / np.maximum(np.linalg.norm(feet, axis=1, keepdims=True), 1e-300)
    xy = points[:, :1] * np.column_stack([feet @ right, feet @ up])
    return xy, off


def draw_section_rays(
    ax: Any, rays: Sequence[Ray], plane: dict, /, **style: Any
) -> None:
    """The parts of the rays within :data:`SLAB` degrees of a section's
    plane, drawn in it."""
    segments = []
    for ray in rays:
        xy, off = in_plane(ray.nodes, plane)
        near = off <= SLAB
        # Each run of nodes near the plane is one piece of the ray.
        edges = np.flatnonzero(np.diff(np.concatenate([[0], near, [0]]).astype(int)))
        for begin, end in zip(edges[::2], edges[1::2]):
            if end - begin > 1:
                segments.append(xy[begin:end])
    ax.add_collection(LineCollection(segments, **style))


def draw_section_boxes(ax: Any, plane: dict, /) -> None:
    """The outline of each box where a section's plane cuts it: the arc of
    the plane's great circle inside the box's latitudes and longitudes,
    between the box's two radii."""
    _, right, up = plane_frame(plane)
    angle = np.radians(np.arange(0.0, 360.0, 0.1))
    located = direction(np.cos(angle)[:, None] * right + np.sin(angle)[:, None] * up)
    for _, (r1, r2, b1, b2, a1, a2) in BOXES:
        inside = (
            (located[:, 0] >= b1)
            & (located[:, 0] <= b2)
            & ((located[:, 1] - a1) % 360.0 <= a2 - a1)
        )
        if not inside.any():
            continue
        # One arc, which may run across the angle zero.
        span = np.where(
            angle[inside] > np.pi, angle[inside] - 2.0 * np.pi, angle[inside]
        )
        if not (inside[0] and inside[-1]):
            span = angle[inside]
        t = np.linspace(span.min(), span.max(), 60)
        radii = np.concatenate([np.full(60, r1), np.full(60, r2), [r1]])
        turns = np.concatenate([t, t[::-1], t[:1]])
        ax.plot(radii * np.cos(turns), radii * np.sin(turns), color="black", lw=1.0)


def section_panel(
    problem: Problem,
    field: Any,
    ax: Any,
    plane: dict,
    /,
    *,
    title: str,
    rays: bool = True,
    **options: Any,
) -> None:
    """One section of a field, with the parts of the rays near its plane, the
    stations on it and the boxes it cuts."""
    plotting.plot_section(problem.space, field, ax=ax, **plane, **options)
    if rays:
        draw_section_rays(
            ax, problem.rays, plane, colors="black", linewidths=0.5, alpha=0.6
        )
    for stations_, marker, color, size in (
        (problem.scale.receivers, "v", "C0", 16.0),
        (problem.scale.sources, "*", "C3", 60.0),
    ):
        plotting.plot_section_points(
            problem.space,
            [(1.06 * RADIUS, b, a) for b, a in stations_],
            ax=ax,
            tolerance=SLAB,
            marker=marker,
            size=size,
            color=color,
            clip_on=False,
            **plane,
        )
    draw_section_boxes(ax, plane)
    ax.set_title(title, fontsize=9)


def shell_panel(
    problem: Problem,
    field: Any,
    ax: Any,
    /,
    *,
    title: str,
    rays: bool = True,
    **options: Any,
) -> None:
    """The shell at mid-mantle depth as a map, with where the rays run
    beneath it, the stations and the boxes' footprints."""
    # Filled contours: the shell is sampled on the ball's own angular grid,
    # which at this degree is coarse, and cells of it would show as tiles.
    levels = np.linspace(options["vmin"], options["vmax"], 23)
    plotting.plot_shell(
        problem.space,
        field,
        radius=MID_MANTLE,
        ax=ax,
        gridlines=False,
        contour=True,
        levels=levels,
        extend="both",
        colorbar_kwargs={"ticks": MaxNLocator(5)},
        **options,
    )
    if rays:
        below = []
        for ray in problem.rays:
            deep = np.flatnonzero(ray.nodes[:, 0] < MID_MANTLE)
            if deep.size > 1:
                below.append((ray.nodes[deep[0], 1:], ray.nodes[deep[-1], 1:]))
        if below:
            plotting.plot_paths(
                Sphere(8), below, ax=ax, color="black", linewidth=0.6, alpha=0.6
            )
    for _, (_, _, b1, b2, a1, a2) in BOXES:
        ax.plot(
            [a1, a2, a2, a1, a1],
            [b1, b1, b2, b2, b1],
            color="black",
            linewidth=1.0,
            transform=crs.PlateCarree(),
        )
    ax.set_title(title, fontsize=9)


def figure_rows(rows: int, /, *, height: float) -> tuple[Any, list]:
    """A figure of rows of four panels, three sections and a map, each row a
    subfigure laid out on its own."""
    fig = plt.figure(figsize=(17.0, height * rows), layout="constrained")
    return fig, list(np.atleast_1d(fig.subfigures(rows, 1)))


def section_row(
    problem: Problem,
    row: Any,
    field: Any,
    /,
    *,
    label: str,
    rays: bool = True,
    **options: Any,
) -> None:
    """Three sections of a field and its shell at mid-mantle depth, in a row."""
    grid = row.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.55])
    for k, (name, plane) in enumerate(SECTIONS):
        ax = row.add_subplot(grid[0, k], aspect="equal")
        section_panel(
            problem,
            field,
            ax,
            plane,
            title=f"{label}, {name}",
            rays=rays,
            colorbar_label=None,
            **options,
        )
    ax = row.add_subplot(grid[0, 3], projection=crs.Robinson())
    shell_panel(
        problem,
        field,
        ax,
        title=f"{label}, the shell at r = {MID_MANTLE}",
        rays=rays,
        colorbar_label=None,
        **options,
    )


def cutaway_figure(problem: Problem, mean: Any, /, *, outdir: pathlib.Path) -> None:
    """The truth's mantle beside the posterior mean, on one color scale, in
    the ball cut in half by the plane of the meridian through the
    continent."""
    mantle = problem.space.layers[-1]
    limit = float(np.abs(mantle.interior_values(problem.truth[-1])).max())
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.2), layout="constrained")
    for ax, field, title in zip(
        axes, (problem.truth, mean), ("the truth", "the posterior mean")
    ):
        plotting.plot_cutaway(
            problem.space,
            field,
            ax=ax,
            longitudes=CUTAWAY_LONGITUDES,
            view=CUTAWAY_VIEW,
            inner=CORE_RADIUS,
            inner_color=CORE_COLOR,
            surface=SURFACE_COLOR,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        ax.set_title(title, fontsize=9)
    fig.colorbar(
        ScalarMappable(Normalize(-limit, limit), "RdBu_r"),
        ax=axes,
        shrink=0.7,
        label="slowness perturbation (%)",
    )
    save(
        fig,
        outdir / "ball_cutaway.png",
        title="Tomography, part F: the truth and the posterior mean, cut by the "
        f"meridian of {CONTINENT[1]:g}°",
    )


def setup_figure(problem: Problem, /, *, outdir: pathlib.Path) -> None:
    """The network on a map, and the truth in three sections and a shell."""
    fig = plt.figure(figsize=(17.0, 9.6), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.05, 1.0])
    left = top.add_subplot(1, 2, 1, projection=crs.Robinson())
    sphere = Sphere(8)
    plotting.plot_network(
        sphere,
        problem.pairs,
        ax=left,
        color="black",
        linewidth=0.5,
        alpha=0.35,
        source_kwargs={"color": "C3", "size": 90.0},
        receiver_kwargs={"marker": "v", "color": "C0", "size": 30.0},
    )
    left.set_global()
    # The traces of the three sections' planes, and the boxes' footprints.
    for (name, plane), color in zip(SECTIONS, ("C2", "C4", "C1")):
        if "pole" in plane:
            left.plot(
                [-180, 180],
                [0, 0],
                color=color,
                lw=1.4,
                transform=crs.PlateCarree(),
                label=name,
            )
        else:
            longitude = plane["longitude"]
            for meridian in (longitude, longitude - 180.0):
                left.plot(
                    [meridian, meridian],
                    [-90, 90],
                    color=color,
                    lw=1.4,
                    transform=crs.PlateCarree(),
                    label=name if meridian == longitude else None,
                )
    for _, (_, _, b1, b2, a1, a2) in BOXES:
        left.plot(
            [a1, a2, a2, a1, a1],
            [b1, b1, b2, b2, b1],
            color="black",
            lw=1.2,
            transform=crs.PlateCarree(),
        )
    left.legend(fontsize=7, loc="lower left", frameon=False)
    left.set_title(
        f"{len(problem.rays)} rays between {len(problem.scale.sources)} sources and "
        f"{len(problem.scale.receivers)} receivers, the sections' planes and the boxes",
        fontsize=9,
    )
    mantle = problem.space.layers[-1]
    limit = float(np.abs(mantle.interior_values(problem.truth[-1])).max())
    right = top.add_subplot(1, 2, 2, projection=crs.Robinson())
    shell_panel(
        problem,
        problem.truth,
        right,
        title=f"the truth on the shell at r = {MID_MANTLE}, and the rays beneath it",
        cmap="RdBu_r",
        symmetric=True,
        vmin=-limit,
        vmax=limit,
        colorbar_label="slowness perturbation (%)",
    )
    grid = bottom.add_gridspec(1, 3)
    for k, (name, plane) in enumerate(SECTIONS):
        ax = bottom.add_subplot(grid[0, k], aspect="equal")
        section_panel(
            problem,
            problem.truth,
            ax,
            plane,
            title=f"the truth, {name}",
            cmap="RdBu_r",
            symmetric=True,
            vmin=-limit,
            vmax=limit,
            colorbar_label="slowness perturbation (%)",
        )
    save(
        fig,
        outdir / "ball_setup.png",
        title="Tomography, part F: the ball's network, and the truth with the parts of "
        f"the rays within {SLAB:g} degrees of each section's plane",
    )


# --------------------------------------------------------------------------- #
#                                 The answers                                 #
# --------------------------------------------------------------------------- #


def point_part(problem: Problem, /, *, outdir: pathlib.Path) -> dict:
    """The minimum-norm estimate and its error budget on the three means."""
    forward = problem.problem
    t0 = time.perf_counter()
    estimator = minimum_norm(problem)
    search = estimator.discrepancy_search(problem.data)
    chosen = estimator.for_data(problem.data)
    smallest = problem.prior.covariance_factor(chosen(problem.data))
    budget = error_budget(problem, chosen)
    print(
        f"the point: minimum norm by the discrepancy principle, damping "
        f"{search.argument:.4g} after {search.evaluations} misfit evaluations, "
        f"{time.perf_counter() - t0:.2f} s; chi-squared "
        f"{forward.chi_squared(smallest, problem.data):.1f}, relative error over the "
        f"mantle {problem.relative_error(smallest):.3f}"
    )
    print(
        f"  {'mean over':<16s} {'estimate':>9s} {'propagated sd':>14s} {'bias':>8s} "
        f"{'truth':>8s}"
    )
    for name, e, s, b, t in zip(
        PROPERTIES,
        budget["estimate"],
        budget["propagated"],
        budget["bias"],
        problem.true_properties,
    ):
        print(f"  {name:<16s} {e:+9.3f} {s:14.3f} {b:+8.3f} {t:+8.3f}")

    mantle = problem.space.layers[-1]
    limit = float(np.abs(mantle.interior_values(problem.truth[-1])).max())
    fig, (upper, lower) = figure_rows(2, height=4.6)
    section_row(
        problem,
        upper,
        smallest,
        label="minimum norm",
        cmap="RdBu_r",
        symmetric=True,
        vmin=-limit,
        vmax=limit,
    )
    error_budget_panels(
        lower.subplots(1, len(NAMES)),
        NAMES,
        budget["estimate"],
        budget["propagated"],
        budget["bias"],
        problem.true_properties,
        ylabel="mean slowness perturbation (%)",
    )
    save(
        fig,
        outdir / "ball_point.png",
        title="Tomography, part F: the point answer in the ball. The minimum-norm "
        "model, and its estimate of each mean with its error budget",
    )
    return {"smallest": smallest, "damping": search.argument, **budget}


def measure_part(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The posterior: the standard deviation exactly at the probes and by
    sampling over the ball, the mean, and the three means, against the
    prior's."""
    space, scale = problem.space, problem.scale
    t0 = time.perf_counter()
    measure = posterior(problem)
    mean = measure.expectation
    places = [point for _, point in PROBES]
    exact = exact_pointwise_std(space, measure, places)
    before = exact_pointwise_std(space, problem.prior, places)
    std = measure.sample_pointwise_std(scale.std_samples, rng=rng, n_jobs=scale.n_jobs)
    properties = property_posterior(problem, measure)
    print(
        f"the measure: the posterior in {time.perf_counter() - t0:.1f} s with "
        f"{scale.std_samples} samples for the pointwise std; relative error over the "
        f"mantle {problem.relative_error(mean):.3f}, chi-squared "
        f"{problem.problem.chi_squared(mean, problem.data):.1f} on {len(problem.rays)} "
        "data"
    )
    for (name, point), value, was in zip(PROBES, exact, before):
        print(f"  pointwise std {name:<20s} at {point} {value:.3f} (prior {was:.3f})")
    print(
        f"  {'mean over':<16s} {'posterior':>9s} {'std':>7s} {'95% from':>9s} "
        f"{'to':>8s} {'truth':>8s} {'prior std':>10s}"
    )
    for name, m, s, t, p in zip(
        PROPERTIES,
        properties["center"],
        properties["deviation"],
        problem.true_properties,
        properties["prior_deviation"],
    ):
        print(
            f"  {name:<16s} {m:+9.3f} {s:7.3f} {m - 1.96 * s:+9.3f} "
            f"{m + 1.96 * s:+8.3f} {t:+8.3f} {p:10.3f}"
        )

    mantle = space.layers[-1]
    limit = float(np.abs(mantle.interior_values(problem.truth[-1])).max())
    fig, (first, second, third) = figure_rows(3, height=4.6)
    section_row(
        problem,
        first,
        mean,
        label="posterior mean",
        cmap="RdBu_r",
        symmetric=True,
        vmin=-limit,
        vmax=limit,
    )
    section_row(
        problem,
        second,
        std,
        label="posterior std",
        cmap="viridis",
        vmin=0.0,
        vmax=1.1 * scale.prior_std,
    )
    grid = third.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.55])
    ax = third.add_subplot(grid[0, 0], aspect="equal")
    section_panel(
        problem,
        prior_pointwise_std(space, order=scale.prior_order, std=scale.prior_std),
        ax,
        SECTIONS[0][1],
        title=f"prior std, {SECTIONS[0][0]}",
        rays=False,
        cmap="viridis",
        vmin=0.0,
        vmax=1.1 * scale.prior_std,
        colorbar_label=None,
    )
    marginal_panels(
        [third.add_subplot(grid[0, k]) for k in (1, 2, 3)],
        NAMES,
        properties["on_properties"],
        properties["prior_on_properties"],
        problem.true_properties,
        xlabel="mean slowness perturbation (%)",
    )
    save(
        fig,
        outdir / "ball_posterior.png",
        title="Tomography, part F: the Bayesian answer in the ball, prior against "
        "posterior. The mean and the standard deviation in three sections and at "
        "mid-mantle,\nthe prior's standard deviation, and each mean's marginal before "
        "and after: the standard deviation falls where the rays cross and nowhere else",
    )
    cutaway_figure(problem, mean, outdir=outdir)
    return {
        "posterior": measure,
        "mean": mean,
        "std": std,
        "exact_std": dict(zip((name for name, _ in PROBES), exact)),
        "prior_std": dict(zip((name for name, _ in PROBES), before)),
        **properties,
    }


def set_part(problem: Problem, /, *, outdir: pathlib.Path) -> dict:
    """The certified brackets on the three means, against the bounds of the
    prior's ball alone."""
    print("the set:")
    result = support_brackets(problem)
    brackets_figure(
        problem,
        result,
        names=NAMES,
        path=outdir / "ball_brackets.png",
        title="Tomography, part F: the set answer in the ball. For each mean, the "
        f"bound the prior's {problem.scale.level:.0%} ball admits alone\nand the "
        "certified bracket the data leave",
    )
    return result


def ball_answers(scale: Scale, /, *, outdir: pathlib.Path) -> dict:
    """The ball's setup, and its three answers with their figures."""
    started = time.perf_counter()
    print("\n=== F. The same planet in a ball ===")
    problem = build(scale=scale)
    setup_figure(problem, outdir=outdir)
    results = {"a": problem}
    results["b"] = point_part(problem, outdir=outdir)
    results["c"] = measure_part(problem, rng=part_rng(scale.seed, "f"), outdir=outdir)
    results["d"] = set_part(problem, outdir=outdir)
    print(f"F took {time.perf_counter() - started:.1f} s")
    return results


def run(scale: Scale = BALL, /, *, outdir: pathlib.Path) -> dict:
    return ball_answers(scale, outdir=outdir)


def main() -> None:
    # The ball is a toy at either scale; --quick only moves the figures.
    args = parse(__doc__, quick=BALL, full=BALL, fixed=BALL_FIXED)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
