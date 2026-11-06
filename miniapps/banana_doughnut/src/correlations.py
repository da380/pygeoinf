"""What a prior's draws do: their correlation from a reference point.

A prior's covariance is an operator in the model space's own metric, so
its nominal length scales are not the lengths over which its draws
decorrelate; the space's own scale enters too. What a prior is, for the
comparison, is read off the covariance itself. The covariance applied to the
representer of evaluation at a point ``x0`` is the field ``c(x0, x)``, the
covariance of the model at ``x0`` with the model everywhere, and divided by
its value at ``x0`` it is the correlation of the field there with the field
at ``x0``: exactly so along the shell through ``x0``, where the prior's
variance is the same at every point, the prior commuting with rotations and
its calibration depending on radius alone; and along the radius to the
calibration's accuracy, which makes the pointwise standard deviation the
same everywhere to the truncation.

The reference point is a ray's turning point, the middle of the path for a
source at the surface. Along the shell the offsets run along the ray, in
arc length on the shell; along the radius they run down to the core-mantle
boundary and up to the surface. The distance at which the correlation first
falls to ``1/e`` is the prior's correlation distance in each direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .planet import CORE_RADIUS, RADIUS, Path

#: The level whose distance is read: the correlation ``1/e``.
LEVEL = float(np.exp(-1.0))
#: The arc lengths along the shell, and the steps along the radius, at which
#: the correlation is read.
SHELL_OFFSETS = np.linspace(0.0, 1.2, 481)
RADIAL_STEP = 0.0025


def turning_point(path: Path, /) -> tuple[float, float, float]:
    """A ray's deepest node, as a point ``(radius, latitude, longitude)``."""
    deepest = int(np.argmin(path.ray.nodes[:, 0]))
    radius, longitude = path.ray.nodes[deepest]
    return float(radius), 0.0, float(longitude)


def first_crossing(offsets: np.ndarray, values: np.ndarray, /) -> float:
    """The offset at which the values first fall below ``1/e``, by linear
    interpolation between the two readings either side, or NaN if they
    never do."""
    below = np.flatnonzero(values < LEVEL)
    if below.size == 0:
        return float("nan")
    k = int(below[0])
    return float(
        np.interp(LEVEL, [values[k], values[k - 1]], [offsets[k], offsets[k - 1]])
    )


@dataclass(frozen=True)
class Correlation:
    """A prior's correlation from a reference point, along the shell through
    it and along the radius.

    Attributes:
        point: the reference point, ``(radius, latitude, longitude)``.
        shell_offsets: arc lengths along the shell from the point.
        shell: the correlation at each.
        down_offsets: distances down the radius from the point, to the
            core-mantle boundary.
        down: the correlation at each.
        up_offsets: distances up the radius, to the surface.
        up: the correlation at each.
    """

    point: tuple[float, float, float]
    shell_offsets: np.ndarray
    shell: np.ndarray
    down_offsets: np.ndarray
    down: np.ndarray
    up_offsets: np.ndarray
    up: np.ndarray

    @property
    def shell_distance(self) -> float:
        """Where the correlation along the shell falls to ``1/e``."""
        return first_crossing(self.shell_offsets, self.shell)

    @property
    def down_distance(self) -> float:
        """Where the correlation down the radius falls to ``1/e``, or NaN if
        it does not above the core-mantle boundary."""
        return first_crossing(self.down_offsets, self.down)

    @property
    def up_distance(self) -> float:
        """Where the correlation up the radius falls to ``1/e``, or NaN if it
        does not below the surface."""
        return first_crossing(self.up_offsets, self.up)

    @property
    def radial_distance(self) -> float:
        """The correlation distance along the radius: the mean of the two
        directions where both reach ``1/e`` inside the mantle, the one that
        does where only one does, NaN where neither does."""
        reached = [d for d in (self.down_distance, self.up_distance) if np.isfinite(d)]
        return float(np.mean(reached)) if reached else float("nan")


def correlation(
    space: Any, prior: Any, point: tuple[float, float, float], /
) -> Correlation:
    """A prior's correlation from a point, along the shell through it in the
    direction of increasing longitude and along the radius both ways.

    One application of the covariance, to the representer of evaluation at
    the point, and one evaluation of the result at every offset.
    """
    radius, latitude, longitude = point
    dirac = space.point_evaluation_operator([point])
    field = prior.covariance(dirac.adjoint(dirac.codomain.basis_vector(0)))
    shell_points = [
        (radius, latitude, longitude + float(np.degrees(s / radius)))
        for s in SHELL_OFFSETS
    ]
    down_offsets = np.arange(0.0, radius - CORE_RADIUS, RADIAL_STEP)
    up_offsets = np.arange(0.0, RADIUS - radius, RADIAL_STEP)
    radial_points = [(radius - d, latitude, longitude) for d in down_offsets] + [
        (radius + d, latitude, longitude) for d in up_offsets
    ]
    values = space.evaluate(field, shell_points + radial_points)
    values = np.asarray(values, dtype=float) / values[0]
    shell = values[: len(shell_points)]
    down = values[len(shell_points) : len(shell_points) + len(down_offsets)]
    up = values[len(shell_points) + len(down_offsets) :]
    return Correlation(
        point, SHELL_OFFSETS.copy(), shell, down_offsets, down, up_offsets, up
    )


def describe(found: Correlation, /, *, unit: float | None = None) -> str:
    """The two correlation distances, in the radius's units and, with a
    unit, in it too, for titles."""

    def length(value: float) -> str:
        if not np.isfinite(value):
            return "beyond the mantle"
        text = f"{value:.3f}"
        return text if unit is None else f"{text} ({value / unit:.2f} wavelengths)"

    return (
        f"1/e at {length(found.shell_distance)} along the shell, "
        f"{length(found.radial_distance)} radially"
    )
