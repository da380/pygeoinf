"""A ray tracer in a radially layered planet: the user's own forward code.

Nothing here is library code. A ray in a medium whose velocity depends on
radius alone keeps its ray parameter ``p = r sin(i) / v(r)`` along its whole
length, ``i`` the angle from the vertical, and turns where ``r / v(r) = p``.
From the surface to that turning radius ``r_t`` and back, it sweeps the angle
and takes the time

.. code-block:: text

    Delta(p) = 2 int_{r_t}^{R} p v / (r sqrt(r^2 - p^2 v^2)) dr
    T(p)     = 2 int_{r_t}^{R} r / (v sqrt(r^2 - p^2 v^2)) dr

and its length is ``2 int r / sqrt(r^2 - p^2 v^2) dr``. The integrands are
singular at the turning point like an inverse square root; the substitution
``r = r_t + (R - r_t) s^2`` removes the singularity, after which each is an
analytic function of ``s`` on ``[0, 1]``, held here as a Chebyshev series
whose integral is exact to roundoff. The two-point ray is a root of
``Delta(p) = Delta`` on the branch of rays that turn above :attr:`bottom`,
where ``Delta`` falls monotonically from its grazing value to zero.

A traced ray is a list of nodes ``(radius, angle)``, the angle in degrees
anticlockwise from the ``x`` axis, at equal increments of arc length along
the curved ray, and the trapezoid rule's weights along the polyline through
them: the path integral of a field ``f`` is ``sum_k w_k f(node_k)``, accurate
to second order in the spacing. The same holds in a ball for a ray in the
plane of a great circle, which is why this module knows nothing about the
dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import numpy as np
from numpy.polynomial import Chebyshev
from scipy.optimize import brentq

#: Chebyshev degree of each integrand in ``s``; they are analytic, and this
#: holds the integrals to roundoff for every ray of the profiles used here.
_DEGREE = 96

#: How far inside the open interval of ray parameters the root find starts, in
#: units of its width: at either end an integral degenerates.
_MARGIN = 1e-9


class Ray(NamedTuple):
    """A traced ray: its nodes, the path-integral weights and its travel time.

    Attributes:
        nodes: ``(count, 2)`` array of ``(radius, angle)``, the angle in
            degrees anticlockwise from the ``x`` axis, from the source to the
            receiver at equal increments of arc length.
        weights: the trapezoid rule along the polyline through the nodes, so
            that ``weights @ f(nodes)`` is the path integral of ``f``.
        travel_time: the reference travel time, from the radial integral.
    """

    nodes: np.ndarray
    weights: np.ndarray
    travel_time: float


@dataclass(frozen=True)
class RadialProfile:
    """A velocity that depends on radius alone, and its rays.

    Attributes:
        velocity: ``v(r)``, vectorized, on ``[bottom, radius]``.
        radius: the surface, where sources and receivers are.
        bottom: the rays traced must turn above this radius; below it the
            velocity is not asked about.
    """

    velocity: Callable[[np.ndarray], np.ndarray]
    radius: float = 1.0
    bottom: float = 0.0
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ----------------------------------------------------------------- #
    #                          The ray parameter                        #
    # ----------------------------------------------------------------- #

    def eta(self, r: np.ndarray) -> np.ndarray:
        """``r / v(r)``, whose value at the turning radius is the ray parameter."""
        r = np.asarray(r, dtype=float)
        return r / self.velocity(r)

    @property
    def parameter_range(self) -> tuple[float, float]:
        """The ray parameters of the rays that turn between :attr:`bottom` and
        the surface: from the ray grazing the bottom to the one leaving the
        surface horizontally."""
        return float(self.eta(self.bottom)), float(self.eta(self.radius))

    def turning_radius(self, p: float, /) -> float:
        """Where ``r / v(r) = p``.

        Raises:
            ValueError: if the ray would turn below :attr:`bottom` or above the
                surface.
        """
        low, high = self.parameter_range
        if not low < p < high:
            raise ValueError(
                f"A ray parameter of {p:g} does not turn between {self.bottom:g} "
                f"and {self.radius:g}, where the parameters run from {low:g} to "
                f"{high:g}."
            )
        return float(
            brentq(lambda r: self.eta(r) - p, self.bottom, self.radius, xtol=1e-15)
        )

    def _integrals(self, p: float, /) -> dict[str, Chebyshev]:
        """The angle, the arc length and the time swept from the turning point
        out to ``r(s)``, each as a Chebyshev series in ``s`` on ``[0, 1]``, and
        ``r(s)`` itself."""
        key = float(p)
        if key in self._cache:
            return self._cache[key]
        turning = self.turning_radius(p)
        span = self.radius - turning
        if span <= _MARGIN * self.radius:
            # A ray that leaves the surface horizontally goes nowhere.
            zero = Chebyshev([0.0], domain=[0.0, 1.0])
            series = {"angle": zero, "length": zero, "time": zero}
            series["turning"] = turning
            series["radius"] = lambda s: turning + span * np.asarray(s) ** 2
            return series

        def radius(s: np.ndarray) -> np.ndarray:
            return turning + span * s**2

        def pieces(s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            r = radius(s)
            v = self.velocity(r)
            # (dr/ds) / sqrt(r^2 - p^2 v^2): finite at the turning point, where
            # both vanish like s. The Chebyshev points are interior, so s = 0
            # itself is never asked about.
            gap = (r - p * v) * (r + p * v)
            return 2.0 * span * s / np.sqrt(gap), r, v

        def angle(s: np.ndarray) -> np.ndarray:
            ratio, r, v = pieces(s)
            return p * v / r * ratio

        def length(s: np.ndarray) -> np.ndarray:
            ratio, r, _ = pieces(s)
            return r * ratio

        def time(s: np.ndarray) -> np.ndarray:
            ratio, r, v = pieces(s)
            return r / v * ratio

        series = {
            name: Chebyshev.interpolate(integrand, _DEGREE, domain=[0.0, 1.0]).integ(
                lbnd=0.0
            )
            for name, integrand in (
                ("angle", angle),
                ("length", length),
                ("time", time),
            )
        }
        series["turning"] = turning
        series["radius"] = radius
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = series
        return series

    def distance(self, p: float, /) -> float:
        """``Delta(p)``, the angular distance in radians from source to receiver."""
        return 2.0 * float(self._integrals(p)["angle"](1.0))

    def travel_time(self, p: float, /) -> float:
        """``T(p)``, the travel time from source to receiver."""
        return 2.0 * float(self._integrals(p)["time"](1.0))

    def path_length(self, p: float, /) -> float:
        """The length of the ray from source to receiver."""
        return 2.0 * float(self._integrals(p)["length"](1.0))

    @property
    def grazing_distance(self) -> float:
        """The angular distance in degrees of the ray that grazes the bottom:
        the farthest any ray reaches."""
        low, high = self.parameter_range
        return float(np.degrees(self.distance(low + _MARGIN * (high - low))))

    def ray_parameter(self, distance: float, /) -> float:
        """The two-point ray: the ``p`` with ``Delta(p)`` the given angular
        distance, in degrees.

        Raises:
            ValueError: if no ray that turns above the bottom reaches that far.
        """
        target = np.radians(float(distance))
        low, high = self.parameter_range
        width = high - low
        low, high = low + _MARGIN * width, high - _MARGIN * width
        reach = self.distance(low)
        if not 0.0 < target < reach:
            raise ValueError(
                f"No ray turning above {self.bottom:g} reaches {distance:g} degrees; "
                f"the farthest reaches {np.degrees(reach):.2f}."
            )
        return float(
            brentq(
                lambda p: self.distance(p) - target, low, high, xtol=1e-14, rtol=1e-14
            )
        )

    # ----------------------------------------------------------------- #
    #                            Tracing a ray                          #
    # ----------------------------------------------------------------- #

    def trace(
        self, source_angle: float, receiver_angle: float, /, *, nodes: int = 301
    ) -> Ray:
        """The ray from a source to a receiver on the surface.

        Args:
            source_angle: where the source is, in degrees anticlockwise from
                the ``x`` axis.
            receiver_angle: where the receiver is, likewise. The ray goes the
                shorter way round.
            nodes: how many nodes along the ray, the two ends included.

        Returns:
            The nodes, the path-integral weights and the travel time.

        Raises:
            ValueError: if the two are too far apart for a ray that turns above
                the bottom, or the same point.
        """
        turn = (float(receiver_angle) - float(source_angle) + 180.0) % 360.0 - 180.0
        distance = abs(turn)
        if distance == 0.0:
            raise ValueError("The source and the receiver are at the same place.")
        sign = 1.0 if turn > 0.0 else -1.0
        p = self.ray_parameter(distance)
        series = self._integrals(p)
        half_length = float(series["length"](1.0))
        half_angle = float(series["angle"](1.0))

        # Equal increments of arc length; each node's distance from the
        # turning point along the ray, and the ``s`` that reaches it.
        along = np.linspace(0.0, 2.0 * half_length, int(nodes))
        from_turn = np.abs(along - half_length)
        s = self._invert(series["length"], from_turn, half_length)
        swept = series["angle"](s)
        before = along < half_length
        angle = np.where(before, half_angle - swept, half_angle + swept)
        radii = series["radius"](s)
        radii[0] = radii[-1] = self.radius
        angles = float(source_angle) + sign * np.degrees(angle)
        points = np.column_stack([radii, angles])
        return Ray(points, polyline_weights(points), 2.0 * float(series["time"](1.0)))

    @staticmethod
    def _invert(length: Chebyshev, targets: np.ndarray, total: float) -> np.ndarray:
        """The ``s`` in ``[0, 1]`` at which the length swept from the turning
        point reaches each target: a monotone table for a start, then Newton
        on the series itself."""
        table = np.linspace(0.0, 1.0, 2049)
        s = np.interp(targets, length(table), table)
        slope = length.deriv()
        for _ in range(6):
            s = np.clip(s - (length(s) - targets) / slope(s), 0.0, 1.0)
        s[np.isclose(targets, total, rtol=0.0, atol=1e-15)] = 1.0
        return s


def polyline_weights(points: np.ndarray, /) -> np.ndarray:
    """The trapezoid rule along the polyline through points ``(radius,
    angle)``: half the straight segment either side of each node."""
    t = np.radians(points[:, 1])
    xy = np.column_stack([points[:, 0] * np.cos(t), points[:, 0] * np.sin(t)])
    segments = np.hypot(*np.diff(xy, axis=0).T)
    weights = np.zeros(len(points))
    weights[:-1] += 0.5 * segments
    weights[1:] += 0.5 * segments
    return weights


def linear_gradient(v0: float, gradient: float, /, *, radius: float = 1.0):
    """``v(r) = v0 + gradient (radius - r)``: speed rising linearly with depth."""

    def velocity(r: np.ndarray) -> np.ndarray:
        return v0 + gradient * (radius - np.asarray(r, dtype=float))

    return velocity


def flattened_linear_gradient(v0: float, gradient: float, /):
    """``v(r) = r (v0 - gradient ln r)`` on the unit disk: the image of the
    flat medium ``v(z) = v0 + gradient z`` under the flattening ``z = -ln r``,
    ``x = angle``, in which every ray is an arc of a circle.

    The map is conformal and takes the slowness ``1 / v(r)`` in the disk to
    ``r / v(r)`` in the flat medium, so travel times and ray parameters are
    the same in both.
    """

    def velocity(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        return r * (v0 - gradient * np.log(r))

    return velocity


@dataclass(frozen=True)
class CircularArcs:
    """The closed-form rays of the flat medium ``v(z) = v0 + gradient z``.

    A ray of horizontal slowness ``p`` is the arc of the circle of radius
    ``1 / (p g)`` centered a height ``v0 / g`` above the surface, which reaches
    a horizontal distance ``X = 2 sqrt(1 - p^2 v0^2) / (p g)`` in the time
    ``T = (2 / g) arccosh(1 / (p v0))``.
    """

    v0: float
    gradient: float

    def ray_parameter(self, distance: float, /) -> float:
        """The ``p`` that reaches a horizontal distance, the inverse of ``X(p)``."""
        return 1.0 / np.sqrt(self.v0**2 + (distance * self.gradient / 2.0) ** 2)

    def distance(self, p: float, /) -> float:
        return 2.0 * np.sqrt(1.0 - (p * self.v0) ** 2) / (p * self.gradient)

    def travel_time(self, p: float, /) -> float:
        return 2.0 / self.gradient * np.arccosh(1.0 / (p * self.v0))

    def turning_depth(self, p: float, /) -> float:
        return (1.0 / p - self.v0) / self.gradient

    def circle(self, p: float, distance: float, /) -> tuple[float, float, float]:
        """The ray's circle in the flat medium: its center ``(x, z)``, with
        the source at the origin and ``z`` the depth, and its radius."""
        return distance / 2.0, -self.v0 / self.gradient, 1.0 / (p * self.gradient)


# --------------------------------------------------------------------- #
#                Rays in a medium that is not radial                    #
# --------------------------------------------------------------------- #

#: A slowness field of the plane: ``(x, y) -> (u, du/dx, du/dy)``, vectorized.
type Slowness = Callable[
    [np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
]


class Shots(NamedTuple):
    """Rays shot from the surface and integrated until they emerge.

    Attributes:
        swept: the angle each swept from its source to where it emerged, in
            radians, positive towards its receiver.
        travel_time: the time each took.
        length: the length of each.
        path: ``(steps + 1, rays, 2)`` positions at every step of arc length.
        tangent: the unit tangents there.
        time: the time taken to reach each step.
        slowness: the slowness there.
        crossing: the step after which each ray emerges.
        fraction: how far into that step, from zero to one.
        step: the step of arc length.
    """

    swept: np.ndarray
    travel_time: np.ndarray
    length: np.ndarray
    path: np.ndarray
    tangent: np.ndarray
    time: np.ndarray
    slowness: np.ndarray
    crossing: np.ndarray
    fraction: np.ndarray
    step: float


def _hermite(
    tau: np.ndarray, a: np.ndarray, da: np.ndarray, b: np.ndarray, db: np.ndarray
) -> np.ndarray:
    """The cubic through ``a`` and ``b`` with slopes ``da`` and ``db`` (per
    unit of ``tau``), at ``tau`` in ``[0, 1]``; the trailing axes broadcast."""
    t2, t3 = tau * tau, tau * tau * tau
    h00, h10 = 2.0 * t3 - 3.0 * t2 + 1.0, t3 - 2.0 * t2 + tau
    h01, h11 = -2.0 * t3 + 3.0 * t2, t3 - t2
    return h00 * a + h10 * da + h01 * b + h11 * db


def shoot(
    slowness: Slowness,
    sources: np.ndarray,
    take_off: np.ndarray,
    sense: np.ndarray,
    /,
    *,
    radius: float = 1.0,
    bottom: float = 0.0,
    step: float = 2.5e-3,
    max_length: float = 4.0,
) -> Shots:
    """Rays from points of the surface, by the ray equations, until they emerge.

    The ray equations in the arc length ``s``, with ``p`` the slowness vector
    ``u t``, ``t`` the unit tangent: ``dx/ds = p / u``, ``dp/ds = grad u``,
    ``dT/ds = u``; integrated by the classical fourth-order Runge-Kutta rule
    at a fixed step, all the rays at once. Where each emerges is the root of
    ``|x| = radius`` on the cubic through the two steps either side.

    Args:
        slowness: the medium.
        sources: where the rays start, angles in degrees on the surface.
        take_off: the angle of each from the downward vertical, in radians.
        sense: ``+1`` to set off anticlockwise, ``-1`` clockwise.
        radius: the surface.
        bottom: no ray may go deeper; one that does is refused.
        step: the step of arc length.
        max_length: the longest a ray may be before it is refused.

    Returns:
        The shots.

    Raises:
        ValueError: if a ray goes below the bottom or has not emerged within
            the longest length allowed.
    """
    theta = np.radians(np.asarray(sources, dtype=float))
    take_off = np.asarray(take_off, dtype=float)
    sense = np.asarray(sense, dtype=float)
    radial = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    along = np.stack([-np.sin(theta), np.cos(theta)], axis=-1)
    position = radius * radial
    direction = (
        -np.cos(take_off)[:, None] * radial
        + (sense * np.sin(take_off))[:, None] * along
    )
    u, _, _ = slowness(position[:, 0], position[:, 1])
    momentum = u[:, None] * direction
    time = np.zeros(len(theta))

    def rates(x: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, ...]:
        u, ux, uy = slowness(x[:, 0], x[:, 1])
        return p / u[:, None], np.stack([ux, uy], axis=-1), u

    path, tangent, times, slownesses = [position], [direction], [time], [u]
    done = np.zeros(len(theta), dtype=bool)
    crossing = np.full(len(theta), -1)
    for index in range(int(np.ceil(max_length / step))):
        k1x, k1p, k1t = rates(position, momentum)
        k2x, k2p, k2t = rates(position + 0.5 * step * k1x, momentum + 0.5 * step * k1p)
        k3x, k3p, k3t = rates(position + 0.5 * step * k2x, momentum + 0.5 * step * k2p)
        k4x, k4p, k4t = rates(position + step * k3x, momentum + step * k3p)
        moved = position + step / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        pushed = momentum + step / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        later = time + step / 6.0 * (k1t + 2.0 * k2t + 2.0 * k3t + k4t)
        keep = done[:, None]
        position = np.where(keep, position, moved)
        momentum = np.where(keep, momentum, pushed)
        time = np.where(done, time, later)
        u, _, _ = slowness(position[:, 0], position[:, 1])
        path.append(position)
        tangent.append(momentum / u[:, None])
        times.append(time)
        slownesses.append(u)
        r = np.hypot(position[:, 0], position[:, 1])
        if np.any(~done & (r < bottom)):
            raise ValueError(f"A ray went below r = {bottom:g}.")
        emerged = ~done & (r > radius)
        crossing[emerged] = index
        done |= emerged
        if done.all():
            break
    else:
        raise ValueError(f"A ray had not emerged after a length of {max_length:g}.")

    path, tangent = np.array(path), np.array(tangent)
    times, slownesses = np.array(times), np.array(slownesses)
    rays = np.arange(len(theta))
    a, b = path[crossing, rays], path[crossing + 1, rays]
    da, db = step * tangent[crossing, rays], step * tangent[crossing + 1, rays]
    ra, rb = np.hypot(*a.T), np.hypot(*b.T)
    tau = (radius - ra) / (rb - ra)
    for _ in range(8):
        point = _hermite(tau[:, None], a, da, b, db)
        slope = _hermite_slope(tau[:, None], a, da, b, db)
        r = np.hypot(*point.T)
        tau = np.clip(tau - (r - radius) / (np.sum(point * slope, axis=1) / r), 0, 1)
    point = _hermite(tau[:, None], a, da, b, db)
    emerged_at = np.arctan2(point[:, 1], point[:, 0])
    swept = sense * ((emerged_at - theta + np.pi) % (2.0 * np.pi) - np.pi)
    travel = _hermite(
        tau,
        times[crossing, rays],
        step * slownesses[crossing, rays],
        times[crossing + 1, rays],
        step * slownesses[crossing + 1, rays],
    )
    return Shots(
        swept,
        travel,
        (crossing + tau) * step,
        path,
        tangent,
        times,
        slownesses,
        crossing,
        tau,
        step,
    )


def _hermite_slope(
    tau: np.ndarray, a: np.ndarray, da: np.ndarray, b: np.ndarray, db: np.ndarray
) -> np.ndarray:
    """The derivative in ``tau`` of :func:`_hermite`."""
    t2 = tau * tau
    return (
        (6.0 * t2 - 6.0 * tau) * a
        + (3.0 * t2 - 4.0 * tau + 1.0) * da
        + (-6.0 * t2 + 6.0 * tau) * b
        + (3.0 * t2 - 2.0 * tau) * db
    )


def _resampled(shots: Shots, index: int, count: int) -> np.ndarray:
    """``count`` points at equal arc length along one shot ray, ends
    included, each on the cubic through the steps either side."""
    along = np.linspace(0.0, shots.length[index], count) / shots.step
    k = np.minimum(np.floor(along).astype(int), shots.crossing[index])
    tau = (along - k)[:, None]
    h = shots.step
    a, b = shots.path[k, index], shots.path[k + 1, index]
    da, db = h * shots.tangent[k, index], h * shots.tangent[k + 1, index]
    return _hermite(tau, a, da, b, db)


def _bracketed_root(function: Callable[[float], float], start: float, /) -> float:
    """A root of a function of the take-off angle near ``start``: a bracket
    found by stepping out either side, doubling the step, then Brent's
    method.

    Raises:
        ValueError: if no change of sign is found within a radian or so.
    """
    value = function(start)
    if value == 0.0:
        return start
    for doubling in range(10):
        width = 2e-3 * 2.0**doubling
        for side in (-1.0, 1.0):
            other = float(np.clip(start + side * width, 1e-3, 0.5 * np.pi - 1e-3))
            if np.sign(function(other)) != np.sign(value):
                low, high = sorted((start, other))
                return float(brentq(function, low, high, xtol=1e-14, rtol=1e-14))
    raise ValueError(f"No take-off angle near {start:.4f} reaches the receiver.")


def simpson_weights(length: float, count: int, /) -> np.ndarray:
    """Simpson's rule on ``count`` equally spaced points over ``length``;
    ``count`` odd."""
    if count % 2 == 0:
        raise ValueError("Simpson's rule needs an odd number of points.")
    weights = np.full(count, 2.0)
    weights[1::2] = 4.0
    weights[0] = weights[-1] = 1.0
    return weights * length / (3.0 * (count - 1))


def bend(
    slowness: Slowness,
    pairs: list[tuple[float, float]],
    initial: np.ndarray,
    /,
    *,
    radius: float = 1.0,
    bottom: float = 0.0,
    step: float = 2.5e-3,
    nodes: int = 401,
    tolerance: float = 1e-13,
    max_iterations: int = 40,
    slope: np.ndarray | None = None,
) -> tuple[list[Ray], np.ndarray, np.ndarray]:
    """Two-point rays through a medium that is not radial, by shooting.

    The take-off angle of every ray is found together, by the secant method on
    the miss, the angle between where the ray emerges and its receiver,
    starting from the given angles and a second set: a Newton step from them
    when the slope of the miss is known, a thousandth of a radian off them
    when it is not. A ray the secant steps leave open is closed on its own,
    by bisection in a bracket about where they left it. A
    ray is then nodes at equal arc length along its path, on the cubic
    through the integrator's steps, with Simpson's weights in arc length, so
    that ``weights @ f(nodes)`` is the path integral of ``f`` along the ray
    to fourth order; and its travel time is the integrator's.

    Args:
        slowness: the medium.
        pairs: ``(source, receiver)`` angles in degrees, on the surface.
        initial: a take-off angle for each to start from, in radians: those
            of the same rays in a nearby medium.
        radius: the surface.
        bottom: no ray of the answer may go deeper; the search's trial rays
            may.
        step: the integrator's step of arc length.
        nodes: points along each ray, odd.
        tolerance: the largest miss allowed, in radians.
        max_iterations: secant steps before closing the open rays one by
            one.
        slope: the derivative of each miss with respect to its take-off
            angle, from a nearby medium, for the first step.

    Returns:
        The rays, their take-off angles, and the slopes of their misses
        there.

    Raises:
        ValueError: if a ray goes below the bottom, or no take-off angle near
            the one the search reached sends it to its receiver.
    """
    sources = np.array([source for source, _ in pairs], dtype=float)
    turns = np.array(
        [(receiver - source + 180.0) % 360.0 - 180.0 for source, receiver in pairs]
    )
    sense, target = np.sign(turns), np.radians(np.abs(turns))
    # The search's trial rays may go deeper than the answer is allowed to.
    options = dict(radius=radius, step=step)

    def miss(angles: np.ndarray, which: np.ndarray) -> np.ndarray:
        shots = shoot(slowness, sources[which], angles, sense[which], **options)
        return shots.swept - target[which]

    everyone = np.arange(len(pairs))
    previous = np.array(initial, dtype=float)
    previous_miss = miss(previous, everyone)
    if slope is None:
        current = previous + 1e-3
    else:
        current = previous - np.clip(previous_miss / slope, -0.05, 0.05)
        current = np.where(current == previous, previous + 1e-9, current)
    current_miss = miss(current, everyone)
    for _ in range(max_iterations):
        open_ = np.flatnonzero(np.abs(current_miss) > tolerance)
        if open_.size == 0:
            break
        difference = current_miss[open_] - previous_miss[open_]
        update = -current_miss[open_] * (current[open_] - previous[open_])
        update = np.clip(
            update / np.where(difference == 0.0, 1.0, difference), -0.05, 0.05
        )
        previous[open_], previous_miss[open_] = current[open_], current_miss[open_]
        current[open_] += update
        current_miss[open_] = miss(current[open_], open_)
    # A ray the secant steps could not close, near a fold of its miss, is
    # bracketed about its better angle and closed by bisection.
    for index in np.flatnonzero(np.abs(current_miss) > tolerance):
        if abs(previous_miss[index]) < abs(current_miss[index]):
            current[index] = previous[index]
        current[index] = _bracketed_root(
            lambda angle, index=index: float(
                miss(np.array([angle]), np.array([index]))[0]
            ),
            float(current[index]),
        )
        current_miss[index] = 0.0

    shots = shoot(slowness, sources, current, sense, bottom=bottom, **options)
    rays = []
    for index in everyone:
        xy = _resampled(shots, index, nodes)
        r = np.hypot(xy[:, 0], xy[:, 1])
        turned = np.degrees(
            (np.arctan2(xy[:, 1], xy[:, 0]) - np.radians(sources[index]) + np.pi)
            % (2.0 * np.pi)
            - np.pi
        )
        points = np.column_stack([r, sources[index] + turned])
        rays.append(
            Ray(
                points,
                simpson_weights(float(shots.length[index]), nodes),
                float(shots.travel_time[index]),
            )
        )
    gradient = (current_miss - previous_miss) / (current - previous)
    return rays, current, gradient
