"""Fields of the model space on the two sections the parts draw.

The ray's own plane is the equatorial plane, drawn turned so that the
ray's mid-path direction is up, the source on the left, as part A draws it;
the plane across the ray at its turning point, which is mid-path for a
source at the surface, is the meridian through that point, drawn with the
offset ``q2`` out of the ray's plane across and the offset ``q1`` in it up,
the ray at the center. The values come from the library's sections of the
ball, :func:`pygeoinf.plotting.ball.section_values`, which evaluate the
field itself round a great circle at the space's radii; nothing outside the
mantle annulus is drawn.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pygeoinf.plotting.ball import section_values

from .common import symmetric_mesh
from .correlations import turning_point
from .planet import CORE_RADIUS, RADIUS, Path

#: The sections, by name.
SECTIONS = ("across", "plane")


def turned(x: Any, y: Any, distance: float, /) -> tuple[Any, Any]:
    """Points of the ray's plane turned so that the ray's mid-path direction
    is up: the source on the left, the receiver on the right."""
    angle = np.radians(90.0 - 0.5 * distance)
    return (
        np.cos(angle) * x - np.sin(angle) * y,
        np.sin(angle) * x + np.cos(angle) * y,
    )


def _round_the_circle(values: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """The directions of a section's columns as angles in ``(-pi, pi]`` from
    its first, in increasing order, and the columns in that order."""
    count = values.shape[1]
    psi = 2.0 * np.pi * np.arange(count) / count
    psi = np.where(psi > np.pi, psi - 2.0 * np.pi, psi)
    order = np.argsort(psi)
    return psi[order], values[:, order]


def section(space: Any, field: np.ndarray, path: Path, which: str, /) -> tuple:
    """A field on one of the two sections, whole: the space's radii, the
    directions round the circle and the values, ``(radii, angles)``.

    Args:
        space: the model space.
        field: a vector of it.
        path: the ray.
        which: ``"across"`` or ``"plane"``.
    """
    if which == "plane":
        # The circle starts at longitude zero, the source's, and runs east.
        options = {"pole": (90.0, 0.0)}
    elif which == "across":
        options = {"longitude": turning_point(path)[2]}
    else:
        raise ValueError(f"A section is one of {SECTIONS}, not {which!r}.")
    values, radii, _, _ = section_values(space, field, **options)
    psi, values = _round_the_circle(values)
    return radii, psi, values


def drawing(
    whole: tuple, path: Path, which: str, /, *, reach: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A section's values near the ray, in the drawing's coordinates: the
    ray's plane turned, or the offsets ``(q2, q1)`` across the ray."""
    radii, psi, values = whole
    if which == "plane":
        span = reach / CORE_RADIUS
        keep = (psi >= -span) & (psi <= np.radians(path.distance) + span)
        psi, values = psi[keep], values[:, keep]
        x, y = turned(
            radii[:, None] * np.cos(psi), radii[:, None] * np.sin(psi), path.distance
        )
        return x, y, values
    keep = np.abs(psi) <= 1.2 * reach / CORE_RADIUS
    psi, values = psi[keep], values[:, keep]
    radius = turning_point(path)[0]
    # North is up the section, and along a ray that travels anticlockwise
    # round the north pole the frame's e2 points south.
    return -radii[:, None] * np.sin(psi), radii[:, None] * np.cos(psi) - radius, values


def near(x: np.ndarray, y: np.ndarray, path: Path, which: str, /, *, reach: float):
    """Which points of a drawing lie within its frame."""
    if which == "across":
        return (np.abs(x) <= reach) & (np.abs(y) <= reach)
    xs, ys = turned(
        path.dynamic.frame.point[:, 0], path.dynamic.frame.point[:, 1], path.distance
    )
    return (
        (x >= xs.min() - reach)
        & (x <= xs.max() + reach)
        & (y >= ys.min() - reach)
        & (y <= RADIUS)
    )


def largest(whole: tuple, path: Path, which: str, /, *, reach: float) -> float:
    """The largest size of a section's values within a drawing's frame."""
    x, y, values = drawing(whole, path, which, reach=reach)
    return float(np.abs(values[near(x, y, path, which, reach=reach)]).max())


def boundaries(ax: Any, path: Path, which: str, /) -> None:
    """The surface and the core-mantle boundary on a drawing."""
    if which == "plane":
        t = np.radians(np.linspace(-0.5 * path.distance, 1.5 * path.distance, 400))
        for radius in (CORE_RADIUS, RADIUS):
            ax.plot(
                *turned(radius * np.cos(t), radius * np.sin(t), path.distance),
                color="0.3",
                lw=0.8,
            )
        return
    center = turning_point(path)[0]
    t = np.linspace(-np.pi / 2, np.pi / 2, 400)
    for radius in (CORE_RADIUS, RADIUS):
        ax.plot(-radius * np.sin(t), radius * np.cos(t) - center, color="0.3", lw=0.8)


def frame(ax: Any, path: Path, which: str, /, *, reach: float) -> None:
    """A drawing's limits, the ray on it and the boundaries: across the
    ray a cross at the center with the offsets labeled, in the ray's plane
    the ray as a line and no ticks."""
    boundaries(ax, path, which)
    ax.set_aspect("equal")
    if which == "across":
        ax.plot(0.0, 0.0, "k+", ms=8)
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_xlabel("q2", fontsize=8)
        ax.set_ylabel("q1", fontsize=8)
        ax.tick_params(labelsize=7)
        return
    point = path.dynamic.frame.point
    xs, ys = turned(point[:, 0], point[:, 1], path.distance)
    ax.plot(xs, ys, color="black", lw=0.8)
    ax.set_xlim(xs.min() - reach, xs.max() + reach)
    ax.set_ylim(ys.min() - reach, RADIUS + 0.01)
    ax.set_xticks([])
    ax.set_yticks([])


def draw(
    ax: Any,
    whole: tuple,
    path: Path,
    which: str,
    /,
    *,
    reach: float,
    limit: float,
    label: str,
) -> None:
    """A section's values near the ray on a diverging scale, with the ray
    and the boundaries."""
    x, y, values = drawing(whole, path, which, reach=reach)
    symmetric_mesh(ax, x, y, values, limit=limit, label=label, shading="gouraud")
    frame(ax, path, which, reach=reach)


def shape(
    ax: Any,
    whole: tuple,
    path: Path,
    which: str,
    /,
    *,
    reach: float,
    levels: tuple[float, ...],
) -> None:
    """Thin contours of a section's values at fractions of its own extreme
    within the drawing's frame, whatever the color scale: its shape."""
    x, y, values = drawing(whole, path, which, reach=reach)
    inside = values[near(x, y, path, which, reach=reach)]
    extreme = float(inside[np.argmax(np.abs(inside))])
    ax.contour(
        x,
        y,
        values / extreme,
        levels=list(levels),
        colors="0.15",
        linewidths=0.5,
        linestyles="solid",
    )


def on_the_ray(space: Any, field: np.ndarray, path: Path, /) -> float:
    """A field's value on the ray at its turning point."""
    return float(space.evaluate(field, [turning_point(path)])[0])


def filled(
    space: Any, field: np.ndarray, whole: tuple, path: Path, /, *, reach: float
) -> float:
    """A field on the ray at its turning point over its value of largest size
    on the section across the ray within a reach of it: one when that is on
    the ray, the hole filled in, and zero for a field hollow on the ray."""
    x, y, values = drawing(whole, path, "across", reach=reach)
    inside = values[near(x, y, path, "across", reach=reach)]
    extreme = float(inside[np.argmax(np.abs(inside))])
    return on_the_ray(space, field, path) / extreme
