"""The planet and its rays: a mantle whose velocity rises linearly with depth.

A planet of radius one: a mantle on ``[0.55, 1]`` whose velocity rises
linearly with depth from 1 at the surface to 1.8 at its base, above a core
no ray traced here enters; dimensionless units throughout. The velocity is
one smooth piece, so a ray's distance falls monotonically with its ray
parameter, and no caustic lies between the ends of a ray short of the
antipode; the rays that turn in the mantle reach at most about 82 degrees.

The source is on the equator at longitude zero, at the surface or at a depth
below it, and each receiver is on the surface on the equator, at longitude
its distance east: every ray lies in the equatorial plane, whose
coordinates, ``x`` towards the source, ``y`` towards longitude 90 and ``z``
towards the north pole, are those of the tracer's ray plane. Along each ray
dynamic ray tracing gives the Hessians of the travel time across it and the
ray-centered frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rays import (
    DynamicRay,
    RadialProfile,
    Ray,
    linear_gradient,
    linear_gradient_derivatives,
)

# The structure.
RADIUS = 1.0
CORE_RADIUS = 0.55
SURFACE_VELOCITY = 1.0
BASE_VELOCITY = 1.8
GRADIENT = (BASE_VELOCITY - SURFACE_VELOCITY) / (RADIUS - CORE_RADIUS)
MANTLE = RadialProfile(
    linear_gradient(SURFACE_VELOCITY, GRADIENT, radius=RADIUS),
    radius=RADIUS,
    bottom=CORE_RADIUS,
    derivatives=linear_gradient_derivatives(SURFACE_VELOCITY, GRADIENT, radius=RADIUS),
)


@dataclass(frozen=True)
class Path:
    """A ray from the source to a receiver and dynamic ray tracing along it.

    Attributes:
        distance: the receiver's distance from the source, in degrees.
        ray: the ray, in the equatorial plane: nodes ``(radius, longitude)``
            at equal increments of arc length, the weights over its travel
            time, its ray parameter and each node's arc length.
        dynamic: the Hessians of the travel time across the ray at each
            node, their sums, the velocity and the ray-centered frame.
    """

    distance: float
    ray: Ray
    dynamic: DynamicRay

    @property
    def length(self) -> float:
        """The ray's length."""
        return float(self.ray.arc_length[-1])

    @property
    def travel_time(self) -> float:
        """The reference travel time."""
        return float(self.ray.travel_time)

    @property
    def turning_radius(self) -> float:
        """The deepest radius the ray reaches."""
        return float(self.ray.nodes[:, 0].min())

    def velocity_at(self, arc: float, /) -> float:
        """The velocity on the ray at an arc length from the source."""
        return float(np.interp(arc, self.dynamic.arc_length, self.dynamic.velocity))

    def fresnel_half_width(self, period: float, /) -> float:
        """The half-width of the first Fresnel zone at mid-path for a
        dominant period, ``sqrt(c T0 L) / 2`` with ``c`` the velocity there:
        where the detour time ``(A1 q1^2 + A2 q2^2) / 2`` reaches half the
        period across a ray whose sums are the straight chord's,
        ``A = 4 / (c L)`` at its middle."""
        c = self.velocity_at(0.5 * self.length)
        return 0.5 * float(np.sqrt(c * period * self.length))


def trace(distance: float, /, *, nodes: int, source_depth: float = 0.0) -> Path:
    """The ray from the source at a depth to a receiver at the surface a
    distance east, in degrees, and dynamic ray tracing along it.

    Raises:
        ValueError: if no ray turning in the mantle reaches that far, or one
            touches a caustic between its ends.
    """
    ray = MANTLE.trace(
        0.0,
        float(distance),
        nodes=int(nodes),
        source_radius=RADIUS - float(source_depth),
    )
    return Path(float(distance), ray, MANTLE.dynamic_ray(ray))


def period_for_width(path: Path, width: float, /) -> float:
    """The dominant period whose Fresnel half-width at mid-path is a given
    width, the inverse of :meth:`Path.fresnel_half_width`."""
    c = path.velocity_at(0.5 * path.length)
    return float((2.0 * width) ** 2 / (c * path.length))
