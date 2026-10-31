"""Ray tracers in a planet: forward code that mini-apps share.

Nothing here is library code, and nothing here knows of the library. The
mini-apps that trace rays load this one module from beside them, as two
applications would load one external code: a tracer in a planet whose
velocity depends on radius alone, the closed form of a medium in which its
rays are circles, and a shooting tracer in a medium that is not radial.

A ray in a medium whose velocity depends on radius alone keeps its ray
parameter ``p = r sin(i) / v(r)`` along its whole length, ``i`` the angle
from the vertical, and turns where ``r / v(r) = p``. From that turning radius
``r_t`` out to a radius ``e`` it sweeps the angle and takes the time

.. code-block:: text

    Delta(p; e) = int_{r_t}^{e} p v / (r sqrt(r^2 - p^2 v^2)) dr
    T(p; e)     = int_{r_t}^{e} r / (v sqrt(r^2 - p^2 v^2)) dr

and its length is ``int r / sqrt(r^2 - p^2 v^2) dr``. A ray between a source
and a receiver, each at the surface or at any radius above the turning
point, is two such legs, one out to each end, summed.

The velocity may come in pieces, between given radii at which it or one of
its derivatives jumps: one function for each piece on its own closed
interval, so that the two sides of a discontinuity are each evaluated on
their own, and the integrals are taken piece by piece. The integrands are
singular at the turning point like an inverse square root. In the piece the
ray turns in, the substitution ``r = r_t + (b - r_t) s^2``, ``b`` the
piece's top, removes the singularity; in each piece above, from ``a`` to
``b``, ``s`` is linear in ``sqrt(r - r_t)``, from zero at ``a`` to one at
``b``, which keeps the integrand smooth in a piece that starts just above
the turning point, where it is nearly singular. After either, each integrand
is an analytic function of ``s`` on ``[0, 1]`` wherever the velocity is
analytic in its piece, held here as a Chebyshev series whose integral is
exact to roundoff.

The rays that turn in one piece form an interval of ``p`` on which
``Delta`` falls monotonically, from the ray that turns at the piece's bottom
to the one that turns at its top, or at the lower of the two ends if that
is lower. The two-point ray is a root of ``Delta(p) = Delta`` on the one such
interval that reaches the distance. A ray reflected at a discontinuity, its
``p`` in the jump of ``r / v`` there, is not traced, nor is one that leaves
a source at depth upwards and never turns.

A traced ray is a list of nodes ``(radius, angle)``, the angle in degrees
anticlockwise from the ``x`` axis, and the trapezoid rule's weights along the
polyline through them: the path integral of a field ``f`` is ``sum_k w_k
f(node_k)``, accurate to second order in the spacing. Where the ray crosses
the radius at which two pieces meet there is a node, and between the ends
and the crossings the nodes are at equal increments of arc length along the
curved ray, so that each stretch of the ray lies in one piece. A traced ray
also has the weights of the integral over its travel time, ``int f / v ds``:
the trapezoid rule of each stretch with the slowness of its own piece, so
that at a discontinuity, where the slowness jumps, each side has its own,
and the integral of a field smooth across it is still accurate to second
order. The same holds in a ball for a ray in the plane of a great circle,
which is why this module knows nothing about the dimension.

Along a traced ray, dynamic ray tracing, :meth:`RadialProfile.dynamic_ray`,
gives the Hessians of the travel time across the ray. A point near the ray
is ``xi + q1 e1 + q2 e2``, ``xi`` the point of the ray at arc length ``l``
from the source, ``e1`` the unit normal to the ray in its plane, on the
side of increasing radius, and ``e2`` the normal to the plane. The time
from the source to that point exceeds the time along the ray to ``xi`` by
``(M1' q1^2 + M2' q2^2) / 2`` to second order, the Hessian being diagonal
in a radial medium, and likewise from the point to the receiver with
``M1''`` and ``M2''``. Each ``M_i = P_i / Q_i``, with

.. code-block:: text

    dP_i/dl = -V_i Q_i / v^2 ,    dQ_i/dl = v P_i ,
    V_1 = v'' sin^2(i) + v' cos^2(i) / r ,    V_2 = v' / r ,

integrated along the ray from its end, where ``P_i = 1`` and ``Q_i = 0``:
``V_i`` is the second derivative of the velocity across the ray, in the
plane and out of it, ``v'`` and ``v''`` the first two radial derivatives,
the one place they are needed. Out of the plane there is a closed form, ``M2'
+ M2'' = p sin(Delta) / (r^2 sin(phi) sin(Delta - phi))``, ``phi`` the angle
swept from the source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Sequence

import numpy as np
from numpy.polynomial import Chebyshev
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

#: Chebyshev degree of each integrand in ``s``; they are analytic, and this
#: holds the integrals to roundoff for every ray of the profiles used here.
_DEGREE = 96

#: How far inside the open interval of ray parameters the root find starts, in
#: units of its width: at either end an integral degenerates.
_MARGIN = 1e-9

#: How near the turning point, in units of the surface radius, a series' node
#: may lie. Nearer, ``r - p v(r)`` is lost in the rounding of the turning
#: radius; a piece that ends within a few millionths of a radius of the
#: turning point has a series of fewer terms, whose nodes lie farther in.
_CLEARANCE = 1e-14

#: Where the Chebyshev series of a velocity is cut, for its derivatives: after
#: the last coefficient above this fraction of the largest. The coefficients
#: beyond have fallen to the rounding, which differentiation would amplify by
#: the fourth power of their degree.
_CHOP = 1e-14

#: Below this ``|cos(i)|``, near the turning point, the angle of incidence of
#: a node is taken from the slope of the radius along the ray rather than
#: from ``sin(i)``, whose complement loses the digits there.
_STEEP = 0.01


def _degree(reach: float, scale: float, /) -> int:
    """The degree of the series of a piece whose top is ``reach`` above the
    turning point: at most :data:`_DEGREE`, and with every node at least
    :data:`_CLEARANCE` times ``scale`` above the turning point, or ``-1`` if
    no series has. The node of the degree-``n`` series nearest ``s = 0`` is
    at ``s = sin^2(pi / (4 (n + 1)))``, at least ``reach s^2`` above."""
    ratio = (_CLEARANCE * scale / reach) ** 0.25 if reach > 0.0 else 1.0
    if ratio > np.sqrt(0.5):
        return -1
    return min(_DEGREE, int(np.pi / (4.0 * np.arcsin(ratio))) - 1)


class Ray(NamedTuple):
    """A traced ray: its nodes, the path-integral weights and its travel time.

    Attributes:
        nodes: ``(count, 2)`` array of ``(radius, angle)``, the angle in
            degrees anticlockwise from the ``x`` axis, from the source to the
            receiver; from a radial profile, at equal increments of arc
            length between the ends and the crossings of the radii at which
            its pieces meet, with a node at each crossing.
        weights: the trapezoid rule along the polyline through the nodes, so
            that ``weights @ f(nodes)`` is the path integral of ``f``.
        travel_time: the reference travel time, from the radial integral.
        time_weights: the trapezoid rule over the ray's travel time, so that
            ``time_weights @ f(nodes)`` is ``int f / v ds``, each stretch
            between crossings with the slowness of its own piece; ``None``
            from a tracer that does not give them.
        ray_parameter: ``p = r sin(i) / v``, in time per radian, constant
            along a ray of a radial profile; ``None`` from a tracer that does
            not give it.
        arc_length: each node's distance from the source along the ray;
            ``None`` from a tracer that does not give it.
    """

    nodes: np.ndarray
    weights: np.ndarray
    travel_time: float
    time_weights: np.ndarray | None = None
    ray_parameter: float | None = None
    arc_length: np.ndarray | None = None


class Frame(NamedTuple):
    """The ray-centered frame at each node of a ray, in the coordinates of
    the ray's plane: the node ``(radius, angle)`` is the point ``(radius
    cos(angle), radius sin(angle), 0)``, and the third coordinate is along
    the normal to the plane.

    With ``i`` the angle of the ray from the upward vertical, ``u`` the
    unit vector upward and ``a`` the unit vector along the circle through
    the node in the direction the ray travels, the tangent is ``k = u cos(i)
    + a sin(i)``, the normal in the plane is ``e1 = u sin(i) - a cos(i)``, on
    the side of increasing radius, and the normal to the plane is ``e2 = k x
    e1``, so that ``(e1, e2, k)`` is right-handed: ``e2`` is ``(0, 0, -1)``
    along a ray that travels anticlockwise and ``(0, 0, 1)`` along one that
    travels clockwise.

    Attributes:
        point: ``(count, 3)``: each node.
        tangent: ``(count, 3)``: ``k`` there.
        in_plane: ``(count, 3)``: ``e1`` there.
        out_of_plane: ``(count, 3)``: ``e2`` there.
    """

    point: np.ndarray
    tangent: np.ndarray
    in_plane: np.ndarray
    out_of_plane: np.ndarray

    def place(
        self, q1: np.ndarray | float, q2: np.ndarray | float = 0.0, /
    ) -> np.ndarray:
        """The points ``point + q1 in_plane + q2 out_of_plane``, in the
        plane's coordinates.

        Args:
            q1: the offsets along ``e1``: an array whose leading axis runs
                over the nodes, or a number for every node.
            q2: the offsets along ``e2``, likewise; the two broadcast.

        Returns:
            An array of the shape ``q1`` and ``q2`` broadcast to, the nodes
            along its leading axis, with a trailing axis of length three.
        """
        q1, q2 = np.broadcast_arrays(
            np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)
        )
        if q1.ndim == 0:
            q1, q2 = (np.full(len(self.point), float(q)) for q in (q1, q2))
        expand = (slice(None), *(None,) * (q1.ndim - 1))
        return (
            self.point[expand]
            + q1[..., None] * self.in_plane[expand]
            + q2[..., None] * self.out_of_plane[expand]
        )


class DynamicRay(NamedTuple):
    """The Hessians of the travel time across a ray of a radial profile, at
    each of its nodes, and the ray-centered frame there.

    At a node, a point ``q1`` along the frame's ``e1`` and ``q2`` along its
    ``e2`` is reached from the source in a time that exceeds the time along
    the ray to the node by ``(M1' q1^2 + M2' q2^2) / 2``, to second order,
    and the time from it to the receiver exceeds the time from the node by
    ``(M1'' q1^2 + M2'' q2^2) / 2``: the detour through it is longer than
    the ray by ``(A1 q1^2 + A2 q2^2) / 2``, ``A_i = M_i' + M_i''``. Each
    Hessian is ``M_i = P_i / Q_i``, with ``P_i`` and ``Q_i`` integrated along
    the ray from the end it is from, where ``P_i = 1`` and ``Q_i = 0``.

    Attributes:
        arc_length: ``(count,)``: each node's distance from the source along
            the ray.
        incidence: ``(count,)``: the angle of the ray from the upward
            vertical, in radians: more than a right angle before the turning
            point, less after.
        velocity: ``(count,)``: the velocity at each node.
        forward: ``(count, 2)``: ``M1'`` and ``M2'``, from the source;
            infinite at the source.
        backward: ``(count, 2)``: ``M1''`` and ``M2''``, from the receiver;
            infinite at the receiver.
        sums: ``(count, 2)``: ``A1`` and ``A2``, positive, infinite at both
            ends.
        forward_p: ``(count, 2)``: ``P1`` and ``P2`` from the source.
        forward_q: ``(count, 2)``: ``Q1`` and ``Q2`` from the source.
        backward_p: ``(count, 2)``: ``P1`` and ``P2`` from the receiver.
        backward_q: ``(count, 2)``: ``Q1`` and ``Q2`` from the receiver.
        frame: the ray-centered frame at each node.
    """

    arc_length: np.ndarray
    incidence: np.ndarray
    velocity: np.ndarray
    forward: np.ndarray
    backward: np.ndarray
    sums: np.ndarray
    forward_p: np.ndarray
    forward_q: np.ndarray
    backward_p: np.ndarray
    backward_q: np.ndarray
    frame: Frame


class _Piece(NamedTuple):
    """What a ray sweeps in one piece, from where it enters at ``s = 0`` out
    to the piece's top at ``s = 1``.

    Attributes:
        radius: ``r(s)``.
        position: ``s(r)``, its inverse.
        angle: the angle swept from ``s = 0`` to ``s``, a Chebyshev series in
            ``s`` on ``[0, 1]``.
        length: the arc length, likewise.
        time: the time, likewise.
    """

    radius: Callable[[np.ndarray], np.ndarray]
    position: Callable[[np.ndarray], np.ndarray]
    angle: Chebyshev
    length: Chebyshev
    time: Chebyshev


class _Integrals(NamedTuple):
    """What a ray sweeps from its turning point up through each piece in turn.

    Attributes:
        turning: the turning radius.
        tops: the top of each piece, from the one the ray turns in up.
        pieces: what it sweeps in each.
        below: for each of ``"angle"``, ``"length"`` and ``"time"``, what it
            sweeps from the turning point to where each piece begins.
    """

    turning: float
    tops: np.ndarray
    pieces: list[_Piece]
    below: dict[str, np.ndarray]


#: A velocity ``v(r)``, vectorized.
type Velocity = Callable[[np.ndarray], np.ndarray]


def _is_pair(value: object, /) -> bool:
    """Whether a value is a pair of functions."""
    return (
        isinstance(value, Sequence)
        and len(value) == 2
        and all(callable(item) for item in value)
    )


@dataclass(frozen=True)
class RadialProfile:
    """A velocity that depends on radius alone, and its rays.

    Attributes:
        velocity: ``v(r)``, vectorized, on ``[bottom, radius]``; or one such
            function for each piece, from the bottom up, each on its own
            closed piece, so that at a discontinuity each side is its own.
        radius: the surface: where sources and receivers are unless they are
            given a radius of their own, and the highest they may be.
        bottom: the rays traced must turn above this radius; below it the
            velocity is not asked about.
        discontinuities: the radii, increasing, strictly between the bottom
            and the surface, at which the velocity or one of its derivatives
            jumps: the integrals are taken piece by piece between them.
            Within each piece ``r / v(r)`` must increase with ``r``.
        derivatives: the first two radial derivatives of the velocity, for
            dynamic ray tracing: a pair of functions ``(v', v'')``,
            vectorized, for every piece, or one pair for each piece. Where
            none are given, those of the velocity's Chebyshev series on each
            piece are used, for an analytic velocity good to about ``1e-10``
            and ``1e-8`` of their largest sizes on the piece, the most near
            its ends.
    """

    velocity: Velocity | Sequence[Velocity]
    radius: float = 1.0
    bottom: float = 0.0
    discontinuities: tuple[float, ...] = ()
    derivatives: (
        tuple[Velocity, Velocity] | Sequence[tuple[Velocity, Velocity]] | None
    ) = None
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        radii = tuple(float(r) for r in self.discontinuities)
        if not all(
            low < high
            for low, high in zip((self.bottom, *radii), (*radii, self.radius))
        ):
            raise ValueError(
                "The discontinuities must increase strictly between the bottom, "
                f"{self.bottom:g}, and the surface, {self.radius:g}."
            )
        object.__setattr__(self, "discontinuities", radii)
        if not callable(self.velocity) and len(self.velocity) != len(radii) + 1:
            raise ValueError(
                f"{len(radii)} discontinuities make {len(radii) + 1} pieces, but "
                f"{len(self.velocity)} velocities are given."
            )
        derivatives = self.derivatives
        if derivatives is not None and not _is_pair(derivatives):
            if len(derivatives) != len(radii) + 1 or not all(
                _is_pair(pair) for pair in derivatives
            ):
                raise ValueError(
                    "The derivatives are a pair of functions, the first and the "
                    "second, or one such pair for each of the "
                    f"{len(radii) + 1} pieces."
                )

    # ----------------------------------------------------------------- #
    #                              The pieces                           #
    # ----------------------------------------------------------------- #

    @property
    def bounds(self) -> tuple[float, ...]:
        """The radii that bound the pieces, from the bottom to the surface."""
        return (float(self.bottom), *self.discontinuities, float(self.radius))

    def _holding(self, r: float, /) -> int:
        """The piece that holds a radius, counted from the bottom; at a
        discontinuity, the piece below it."""
        return int(np.searchsorted(self.discontinuities, r, side="left"))

    def _eta(self, piece: int, r: np.ndarray, /) -> np.ndarray:
        """``r / v(r)`` in one piece."""
        r = np.asarray(r, dtype=float)
        velocity = self.velocity if callable(self.velocity) else self.velocity[piece]
        return r / velocity(r)

    def _velocity(self, piece: int, /) -> Velocity:
        """The velocity in one piece."""
        return self.velocity if callable(self.velocity) else self.velocity[piece]

    def _derivatives(self, piece: int, /) -> tuple[Velocity, Velocity]:
        """The first two radial derivatives of the velocity in one piece: those
        given, or those of the velocity's Chebyshev series on the piece, cut
        where its coefficients fall to the rounding."""
        if self.derivatives is not None:
            if _is_pair(self.derivatives):
                return tuple(self.derivatives)
            return tuple(self.derivatives[piece])
        key = ("derivatives", piece)
        if key not in self._cache:
            bounds = self.bounds
            series = _chopped(
                Chebyshev.interpolate(
                    self._velocity(piece),
                    _DEGREE,
                    domain=[bounds[piece], bounds[piece + 1]],
                )
            )
            self._cache[key] = (series.deriv(1), series.deriv(2))
        return self._cache[key]

    # ----------------------------------------------------------------- #
    #                          The ray parameter                        #
    # ----------------------------------------------------------------- #

    def eta(self, r: np.ndarray, /) -> np.ndarray:
        """``r / v(r)``, whose value at the turning radius is the ray parameter;
        at a discontinuity, that of the piece below it."""
        r = np.asarray(r, dtype=float)
        if callable(self.velocity):
            return self._eta(0, r)
        pieces = np.searchsorted(self.discontinuities, r, side="left")
        values = np.empty_like(r)
        for piece in np.unique(pieces):
            values[pieces == piece] = self._eta(int(piece), r[pieces == piece])
        return values

    @property
    def parameter_range(self) -> tuple[float, float]:
        """The ray parameters of the rays that turn between :attr:`bottom` and
        the surface: from the ray grazing the bottom to the one leaving the
        surface horizontally."""
        return (
            float(self._eta(0, self.bottom)),
            float(self._eta(len(self.discontinuities), self.radius)),
        )

    def turning_radius(
        self,
        p: float,
        /,
        *,
        source_radius: float | None = None,
        receiver_radius: float | None = None,
    ) -> float:
        """Where ``r / v(r) = p``, for the ray between ends at the given radii,
        the surface by default.

        Raises:
            ValueError: if the ray would turn below :attr:`bottom` or above
                either end, or is reflected at a discontinuity.
        """
        ends = self._ends(source_radius, receiver_radius)
        _, turning = self._turning(p, max(ends))
        self._below(p, turning, ends)
        return turning

    def _turning(self, p: float, top: float, /) -> tuple[int, float]:
        """The piece a ray turns in and its turning radius, for a ray that
        reaches up to ``top``: the highest radius below ``top`` at which
        ``r / v(r) = p``, found piece by piece from the top down.

        Raises:
            ValueError: if the ray would turn below :attr:`bottom` or above
                ``top``, or is reflected at a discontinuity.
        """
        low, high = self.parameter_range
        if not low < p < high:
            raise ValueError(
                f"A ray parameter of {p:g} does not turn between {self.bottom:g} "
                f"and {self.radius:g}, where the parameters run from {low:g} to "
                f"{high:g}."
            )
        bounds = self.bounds
        piece = self._holding(top)
        if self._eta(piece, top) < p:
            raise ValueError(f"A ray parameter of {p:g} turns above r = {top:g}.")
        while self._eta(piece, bounds[piece]) > p:
            piece -= 1
            if self._eta(piece, bounds[piece + 1]) < p:
                raise ValueError(
                    f"A ray parameter of {p:g} is reflected at the discontinuity "
                    f"at r = {bounds[piece + 1]:g}, where r / v jumps past it; "
                    "reflected rays are not traced."
                )
        upper = min(bounds[piece + 1], top)
        turning = brentq(
            lambda r: self._eta(piece, r) - p, bounds[piece], upper, xtol=1e-15
        )
        return piece, float(turning)

    @staticmethod
    def _below(p: float, turning: float, ends: tuple[float, float], /) -> None:
        """Raise unless a ray turns below both its ends."""
        if turning > min(ends):
            raise ValueError(
                f"A ray parameter of {p:g} turns at r = {turning:g}, above the end "
                f"at r = {min(ends):g}; rays that do not turn are not traced."
            )

    def _ends(
        self, source_radius: float | None, receiver_radius: float | None, /
    ) -> tuple[float, float]:
        """The radii of a ray's two ends, the surface where none is given."""
        ends = tuple(
            float(self.radius) if end is None else float(end)
            for end in (source_radius, receiver_radius)
        )
        for end in ends:
            if not self.bottom < end <= self.radius:
                raise ValueError(
                    f"An end at r = {end:g} is not above the bottom, "
                    f"{self.bottom:g}, and at or below the surface, {self.radius:g}."
                )
        return ends

    def _swept(self, p: float, piece: int, turning: float, top: float, /) -> _Piece:
        """What a ray sweeps in one piece, from its turning point or the
        piece's bottom, whichever is higher, out to ``top``, the piece's top."""
        velocity = self.velocity if callable(self.velocity) else self.velocity[piece]
        bottom = self.bounds[piece]
        reach = top - turning
        degree = _degree(reach, self.radius)
        if bottom <= turning:
            # The piece the ray turns in: r = r_t + span s^2.
            span = reach

            def radius(s: np.ndarray) -> np.ndarray:
                return turning + span * np.asarray(s) ** 2

            def slope(s: np.ndarray) -> np.ndarray:
                return 2.0 * span * s

            def position(r: np.ndarray) -> np.ndarray:
                return np.sqrt((r - turning) / span)

            # A ray that turns that near the surface goes nowhere.
            nowhere = (
                piece == len(self.discontinuities) and span <= _MARGIN * self.radius
            )
        else:
            # A piece above: r = r_t + u^2, with u linear in s.
            low, high = np.sqrt(bottom - turning), np.sqrt(top - turning)

            def radius(s: np.ndarray) -> np.ndarray:
                return turning + (low + (high - low) * np.asarray(s)) ** 2

            def slope(s: np.ndarray) -> np.ndarray:
                return 2.0 * (high - low) * (low + (high - low) * s)

            def position(r: np.ndarray) -> np.ndarray:
                return (np.sqrt(r - turning) - low) / (high - low)

            nowhere = False
        if nowhere or degree < 0:
            # A piece whose top is too near the turning point for even one node,
            # within 4e-14 of a radius, is left out: the ray sweeps an angle of
            # the order of the square root of that in it, a few parts in ten
            # million of a radian, little more than the rounding of the turning
            # radius already moves it.
            zero = Chebyshev([0.0], domain=[0.0, 1.0])
            return _Piece(radius, position, zero, zero, zero)

        def parts(s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            r = radius(s)
            v = velocity(r)
            # (dr/ds) / sqrt(r^2 - p^2 v^2): finite at the turning point, where
            # both vanish like s. The Chebyshev points are interior, so s = 0
            # itself is never asked about.
            gap = (r - p * v) * (r + p * v)
            return slope(s) / np.sqrt(gap), r, v

        def angle(s: np.ndarray) -> np.ndarray:
            ratio, r, v = parts(s)
            return p * v / r * ratio

        def length(s: np.ndarray) -> np.ndarray:
            ratio, r, _ = parts(s)
            return r * ratio

        def time(s: np.ndarray) -> np.ndarray:
            ratio, r, v = parts(s)
            return r / v * ratio

        return _Piece(
            radius,
            position,
            *(
                Chebyshev.interpolate(integrand, degree, domain=[0.0, 1.0]).integ(
                    lbnd=0.0
                )
                for integrand in (angle, length, time)
            ),
        )

    def _integrals(self, p: float, top: float, /) -> _Integrals:
        """What a ray sweeps from its turning point up through each piece in
        turn, to the top of the one that holds ``top``, its higher end."""
        last = self._holding(top)
        key = (float(p), last)
        if key in self._cache:
            return self._cache[key]
        first, turning = self._turning(p, top)
        bounds = self.bounds
        pieces = [
            self._swept(p, piece, turning, bounds[piece + 1])
            for piece in range(first, last + 1)
        ]
        below = {
            name: np.cumsum(
                [0.0, *(float(getattr(piece, name)(1.0)) for piece in pieces[:-1])]
            )
            for name in ("angle", "length", "time")
        }
        integrals = _Integrals(
            turning, np.array(bounds[first + 1 : last + 2]), pieces, below
        )
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = integrals
        return integrals

    def _legs(self, p: float, ends: tuple[float, float], name: str, /) -> list[float]:
        """What the ray of parameter ``p`` sweeps of ``name`` from its turning
        point out to each of its ends."""
        integrals = self._integrals(p, max(ends))
        self._below(p, integrals.turning, ends)
        return [self._out_to(integrals, end, name) for end in ends]

    @staticmethod
    def _out_to(integrals: _Integrals, end: float, name: str, /) -> float:
        """What a ray sweeps of ``name`` from its turning point out to ``end``."""
        index = min(
            int(np.searchsorted(integrals.tops, end)), len(integrals.pieces) - 1
        )
        piece = integrals.pieces[index]
        s = 1.0 if end >= integrals.tops[index] else float(piece.position(end))
        return float(integrals.below[name][index] + getattr(piece, name)(s))

    def distance(
        self,
        p: float,
        /,
        *,
        source_radius: float | None = None,
        receiver_radius: float | None = None,
    ) -> float:
        """``Delta(p)``, the angular distance in radians from source to
        receiver, at the given radii, the surface by default."""
        ends = self._ends(source_radius, receiver_radius)
        to_source, to_receiver = self._legs(p, ends, "angle")
        return to_source + to_receiver

    def travel_time(
        self,
        p: float,
        /,
        *,
        source_radius: float | None = None,
        receiver_radius: float | None = None,
    ) -> float:
        """``T(p)``, the travel time from source to receiver."""
        ends = self._ends(source_radius, receiver_radius)
        to_source, to_receiver = self._legs(p, ends, "time")
        return to_source + to_receiver

    def path_length(
        self,
        p: float,
        /,
        *,
        source_radius: float | None = None,
        receiver_radius: float | None = None,
    ) -> float:
        """The length of the ray from source to receiver."""
        ends = self._ends(source_radius, receiver_radius)
        to_source, to_receiver = self._legs(p, ends, "length")
        return to_source + to_receiver

    @property
    def grazing_distance(self) -> float:
        """The angular distance in degrees, from surface to surface, of the ray
        that grazes the bottom: where ``Delta`` falls with ``p``, the farthest
        any ray reaches."""
        low, high = self.parameter_range
        return float(np.degrees(self.distance(low + _MARGIN * (high - low))))

    def _branches(
        self, ends: tuple[float, float], deepest: float, highest: float, /
    ) -> list[tuple[float, float]]:
        """The ray parameters of the rays that turn in each piece between two
        radii, below both ends, and reach up to both, as open intervals,
        inside by the margin, from the bottom up; a piece no ray turns in has
        none."""
        lower, upper = min(ends), max(ends)
        bounds = self.bounds
        top = self._holding(upper)
        branches = []
        for piece in range(self._holding(lower) + 1):
            start = max(bounds[piece], deepest)
            stop = min(bounds[piece + 1], highest, lower)
            if start >= stop:
                continue
            low = float(self._eta(piece, start))
            # Below the lower end and the highest radius allowed in its own
            # piece, and nowhere above beyond r / v at the bottom of a piece,
            # where it is least.
            high = min(
                [
                    float(self._eta(piece, stop)),
                    *(
                        float(self._eta(k, bounds[k]))
                        for k in range(piece + 1, top + 1)
                    ),
                ]
            )
            if low < high:
                width = high - low
                branches.append((low + _MARGIN * width, high - _MARGIN * width))
        return branches

    def ray_parameter(
        self,
        distance: float,
        /,
        *,
        source_radius: float | None = None,
        receiver_radius: float | None = None,
        turning_between: tuple[float, float] | None = None,
    ) -> float:
        """The two-point ray: the ``p`` with ``Delta(p)`` the given angular
        distance, in degrees, between ends at the given radii, the surface by
        default, among the rays that turn between two radii, the bottom and
        the lower end by default.

        Raises:
            ValueError: if no ray that turns there reaches that far, or more
                than one does.
        """
        target = np.radians(float(distance))
        ends = self._ends(source_radius, receiver_radius)
        if turning_between is None:
            deepest, highest = float(self.bottom), min(ends)
            where = f"above {self.bottom:g}"
        else:
            deepest, highest = (float(r) for r in turning_between)
            where = f"between {deepest:g} and {highest:g}"

        def reach(p: float) -> float:
            return self.distance(p, source_radius=ends[0], receiver_radius=ends[1])

        brackets, reaches = [], []
        for low, high in self._branches(ends, deepest, highest):
            near, far = reach(high), reach(low)
            reaches.append((near, far))
            if near < target < far:
                brackets.append((low, high))
        if len(brackets) == 1:
            low, high = brackets[0]
            return float(
                brentq(lambda p: reach(p) - target, low, high, xtol=1e-14, rtol=1e-14)
            )
        if brackets:
            raise ValueError(
                f"{len(brackets)} rays that turn in different pieces reach "
                f"{distance:g} degrees; a distance reached more than once is not "
                "traced unless the radii the ray must turn between choose one."
            )
        farthest = max((far for _, far in reaches), default=0.0)
        nearest = min((near for near, _ in reaches), default=0.0)
        if target >= farthest:
            raise ValueError(
                f"No ray turning {where} reaches {distance:g} degrees; "
                f"the farthest reaches {np.degrees(farthest):.2f}."
            )
        if target <= nearest:
            raise ValueError(
                f"Every ray turning {where} and below both ends reaches beyond "
                f"{distance:g} degrees, the nearest {np.degrees(nearest):.2f}; rays "
                "that do not turn are not traced."
            )
        raise ValueError(
            f"No ray that turns reaches {distance:g} degrees: the distance lies "
            "between the reaches of rays turning either side of a discontinuity, "
            "where the rays that reach it are reflected, and those are not traced."
        )

    # ----------------------------------------------------------------- #
    #                            Tracing a ray                          #
    # ----------------------------------------------------------------- #

    def trace(
        self,
        source_angle: float,
        receiver_angle: float,
        /,
        *,
        nodes: int = 301,
        source_radius: float | None = None,
        receiver_radius: float | None = None,
        turning_between: tuple[float, float] | None = None,
    ) -> Ray:
        """The ray from a source to a receiver.

        Args:
            source_angle: where the source is, in degrees anticlockwise from
                the ``x`` axis.
            receiver_angle: where the receiver is, likewise. The ray goes the
                shorter way round.
            nodes: how many nodes along the ray, the two ends and a node at
                each crossing of a radius at which two pieces meet included.
            source_radius: the source's radius, the surface by default.
            receiver_radius: the receiver's, likewise.
            turning_between: the radii between which the ray must turn, the
                bottom and the lower end by default.

        Returns:
            The nodes, the path-integral weights, the travel time, the
            weights of the integral over it, the ray parameter and each
            node's arc length from the source.

        Raises:
            ValueError: if no ray that turns there joins them, or more than
                one does, or they are at the same angle, or there are too few
                nodes for one at each end and each crossing.
        """
        turn = (float(receiver_angle) - float(source_angle) + 180.0) % 360.0 - 180.0
        distance = abs(turn)
        if distance == 0.0:
            raise ValueError("The source and the receiver are at the same angle.")
        sign = 1.0 if turn > 0.0 else -1.0
        ends = self._ends(source_radius, receiver_radius)
        p = self.ray_parameter(
            distance,
            source_radius=ends[0],
            receiver_radius=ends[1],
            turning_between=turning_between,
        )
        integrals = self._integrals(p, max(ends))
        length, swept, time = (
            [self._out_to(integrals, end, name) for end in ends]
            for name in ("length", "angle", "time")
        )

        # Where the ray crosses the bottom of each piece above the one it turns
        # in, in arc length from the source: on the source's leg before the
        # turning point and on the receiver's after. A crossing at an end is
        # that end.
        starts = integrals.below["length"]
        crossings, crossed = [], []
        for leg, side in ((0, -1.0), (1, 1.0)):
            for piece in range(1, len(starts)):
                if starts[piece] < length[leg]:
                    crossings.append(length[0] + side * starts[piece])
                    crossed.append(integrals.tops[piece - 1])
        order = np.argsort(crossings)
        total = length[0] + length[1]
        breaks = np.concatenate([[0.0], np.array(crossings)[order], [total]])

        # Equal increments of arc length from each break to the next; each
        # node's distance from the turning point along the ray, on the
        # source's leg before it and on the receiver's after.
        along, at = _spread(breaks, int(nodes))
        from_turn = np.abs(along - length[0])
        radii, from_turning = self._locate(integrals, from_turn)
        before = along < length[0]
        angle = np.where(before, swept[0] - from_turning, swept[0] + from_turning)
        radii[0], radii[-1] = ends
        radii[at[1:-1]] = np.array(crossed)[order]
        angle[0], angle[-1] = 0.0, swept[0] + swept[1]
        angles = float(source_angle) + sign * np.degrees(angle)
        points = np.column_stack([radii, angles])

        # Each stretch between breaks lies in one piece, the one its middle
        # is in, whose slowness it takes, up to and including its ends.
        first = self._holding(max(ends)) - len(integrals.pieces) + 1
        middles = np.abs(0.5 * (breaks[:-1] + breaks[1:]) - length[0])
        pieces = first + np.searchsorted(starts, middles, side="right") - 1
        time_weights = np.zeros(len(points))
        for begin, end, piece in zip(at[:-1], at[1:], pieces):
            velocity = (
                self.velocity if callable(self.velocity) else self.velocity[piece]
            )
            stretch = slice(begin, end + 1)
            time_weights[stretch] += polyline_weights(points[stretch]) / velocity(
                radii[stretch]
            )
        return Ray(
            points,
            polyline_weights(points),
            time[0] + time[1],
            time_weights,
            float(p),
            along,
        )

    def _locate(
        self, integrals: _Integrals, targets: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        """Where a ray is when it has come each target length from its turning
        point: the radius, and the angle swept from the turning point."""
        starts = integrals.below["length"]
        which = np.clip(
            np.searchsorted(starts, targets, side="right") - 1,
            0,
            len(integrals.pieces) - 1,
        )
        radii, swept = np.empty_like(targets), np.empty_like(targets)
        for index, piece in enumerate(integrals.pieces):
            mine = which == index
            if not mine.any():
                continue
            s = self._invert(
                piece.length, targets[mine] - starts[index], float(piece.length(1.0))
            )
            radii[mine] = piece.radius(s)
            swept[mine] = integrals.below["angle"][index] + piece.angle(s)
        return radii, swept

    @staticmethod
    def _invert(length: Chebyshev, targets: np.ndarray, total: float) -> np.ndarray:
        """The ``s`` in ``[0, 1]`` at which the length swept from the start of
        a piece reaches each target: a monotone table for a start, then
        Newton on the series itself."""
        table = np.linspace(0.0, 1.0, 2049)
        s = np.interp(targets, length(table), table)
        slope = length.deriv()
        for _ in range(6):
            s = np.clip(s - (length(s) - targets) / slope(s), 0.0, 1.0)
        s[np.isclose(targets, total, rtol=0.0, atol=1e-15)] = 1.0
        return s

    # ----------------------------------------------------------------- #
    #                         Dynamic ray tracing                       #
    # ----------------------------------------------------------------- #

    def dynamic_ray(
        self, ray: Ray, /, *, rtol: float = 1e-12, atol: float = 1e-14
    ) -> DynamicRay:
        """The Hessians of the travel time across a ray this profile traced,
        from the source and from the receiver, their sums, and the
        ray-centered frame, at every node.

        Along the ray, in its arc length ``l``, the Hessian from the source is
        ``M_i' = P_i / Q_i`` with

        .. code-block:: text

            dP_i/dl = -V_i Q_i / v^2 ,    dQ_i/dl = v P_i ,
            V_1 = v'' sin^2(i) + v' cos^2(i) / r ,    V_2 = v' / r ,

        ``P_i = 1`` and ``Q_i = 0`` at the source, ``i`` the angle of the ray
        from the upward vertical, ``sin(i) = p v / r``; the one from the
        receiver, ``M_i''``, is the same integration from the receiver back
        along the ray. The radius along the ray is the traced ray's, held
        between the ends and the crossings of the radii at which pieces meet
        as a Chebyshev series of ``r(l)^2`` in the arc length, which is
        analytic through the turning point, and the angle of incidence
        follows from it and the ray parameter. The system is integrated over
        each such stretch in turn by :func:`scipy.integrate.solve_ivp`'s
        DOP853, whose dense output gives the nodes. The velocity's
        derivatives are :attr:`derivatives`, or its series' where none are
        given.

        Args:
            ray: a ray this profile traced.
            rtol: the integrator's relative tolerance.
            atol: its absolute tolerance, on ``P_i`` and ``Q_i``, which start
                at one and zero.

        Returns:
            The Hessians and the frame at each node.

        Raises:
            ValueError: if the ray carries no ray parameter or arc lengths, if
                it crosses a radius at which the velocity jumps, or if either
                sum fails to be positive between the ends: a caustic lies
                between them, which adds a phase that is not carried here.
        """
        if ray.ray_parameter is None or ray.arc_length is None:
            raise ValueError(
                "Dynamic ray tracing needs a ray traced by a radial profile, "
                "which carries its ray parameter and its nodes' arc lengths."
            )
        p = float(ray.ray_parameter)
        along = np.asarray(ray.arc_length, dtype=float)
        radii = np.asarray(ray.nodes[:, 0], dtype=float)
        ends = (float(radii[0]), float(radii[-1]))
        integrals = self._integrals(p, max(ends))
        first = self._holding(max(ends)) - len(integrals.pieces) + 1
        length = [self._out_to(integrals, end, "length") for end in ends]

        # The stretches between the ends and the crossings, as the ray was
        # traced, each in one piece; across a jump in the velocity the
        # Hessians jump too, by conditions not applied here, but where only a
        # derivative jumps, P and Q do not.
        starts = integrals.below["length"]
        crossings = []
        for leg, side in ((0, -1.0), (1, 1.0)):
            for piece in range(1, len(starts)):
                if starts[piece] < length[leg]:
                    crossings.append(length[0] + side * starts[piece])
                    radius = float(integrals.tops[piece - 1])
                    below, above = (
                        float(self._velocity(first + k)(radius))
                        for k in (piece - 1, piece)
                    )
                    if not np.isclose(below, above, rtol=1e-12, atol=0.0):
                        raise ValueError(
                            f"The velocity jumps from {below:g} to {above:g} at "
                            f"r = {radius:g}, which the ray crosses; dynamic ray "
                            "tracing across a jump is not done."
                        )
        breaks = np.concatenate([[0.0], np.sort(crossings), [along[-1]]])
        middles = np.abs(0.5 * (breaks[:-1] + breaks[1:]) - length[0])
        pieces = first + np.searchsorted(starts, middles, side="right") - 1

        def squared(arc: np.ndarray) -> np.ndarray:
            r, _ = self._locate(integrals, np.abs(arc - length[0]))
            return r * r

        stretches = []
        for low, high, piece in zip(breaks[:-1], breaks[1:], pieces):
            if high <= low:
                continue
            series = _chopped(
                Chebyshev.interpolate(squared, _DEGREE, domain=[low, high])
            )
            mine = np.flatnonzero((along >= low) & (along <= high))
            stretches.append((low, high, int(piece), series, mine))

        options = dict(p=p, rtol=rtol, atol=atol)
        forward = self._propagate(stretches, along, 1.0, **options)
        backward = self._propagate(stretches[::-1], along, -1.0, **options)
        start = np.array([1.0, 1.0, 0.0, 0.0])
        forward[0], backward[-1] = start, start

        # Both Q_i stay positive between the ends, and both sums, unless a
        # caustic lies between them, where a Q_i and a sum change sign.
        with np.errstate(divide="ignore"):
            hessians = (
                forward[:, :2] / forward[:, 2:],
                backward[:, :2] / backward[:, 2:],
            )
        sums = hessians[0] + hessians[1]
        if not (
            np.all(forward[1:, 2:] > 0.0)
            and np.all(backward[:-1, 2:] > 0.0)
            and np.all(sums[1:-1] > 0.0)
        ):
            raise ValueError(
                f"The ray of parameter {p:g} from r = {ends[0]:g} to r = "
                f"{ends[1]:g} touches a caustic between its ends, where a sum of "
                "the Hessians from the two ends changes sign; the phase a caustic "
                "adds is not carried."
            )

        # The angle of incidence: sin(i) from the ray parameter, and cos(i)
        # from it too, negative before the turning point, except near the
        # turning point, where that loses the digits the slope of the radius
        # along the ray keeps.
        velocity, slope = np.empty_like(along), np.empty_like(along)
        for _, _, piece, series, mine in stretches:
            velocity[mine] = self._velocity(piece)(radii[mine])
            slope[mine] = series.deriv()(along[mine]) / (2.0 * radii[mine])
        sine = p * velocity / radii
        gap = (radii - p * velocity) * (radii + p * velocity)
        cosine = np.sqrt(np.maximum(gap, 0.0)) / radii
        cosine = np.where(along < length[0], -cosine, cosine)
        cosine = np.where(np.abs(cosine) < _STEEP, slope, cosine)
        norm = np.hypot(sine, cosine)
        sine, cosine = sine / norm, cosine / norm

        theta = np.radians(ray.nodes[:, 1])
        sense = 1.0 if ray.nodes[-1, 1] > ray.nodes[0, 1] else -1.0
        zero = np.zeros_like(theta)
        radial = np.column_stack([np.cos(theta), np.sin(theta), zero])
        around = sense * np.column_stack([-np.sin(theta), np.cos(theta), zero])
        frame = Frame(
            radii[:, None] * radial,
            cosine[:, None] * radial + sine[:, None] * around,
            sine[:, None] * radial - cosine[:, None] * around,
            np.tile([0.0, 0.0, -sense], (len(theta), 1)),
        )
        return DynamicRay(
            along,
            np.arctan2(sine, cosine),
            velocity,
            hessians[0],
            hessians[1],
            sums,
            forward[:, :2],
            forward[:, 2:],
            backward[:, :2],
            backward[:, 2:],
            frame,
        )

    def _propagate(
        self,
        stretches: list[tuple[float, float, int, Chebyshev, np.ndarray]],
        along: np.ndarray,
        sense: float,
        /,
        *,
        p: float,
        rtol: float,
        atol: float,
    ) -> np.ndarray:
        """``(P1, P2, Q1, Q2)`` at each node, from ``(1, 1, 0, 0)`` at one end
        of the ray, through the stretches in turn, each ``(from, to, piece,
        series of r^2 in l, its nodes)``: forward from the source when
        ``sense`` is one, back from the receiver when it is minus one."""
        state = np.array([1.0, 1.0, 0.0, 0.0])
        values = np.empty((len(along), 4))
        for low, high, piece, series, mine in stretches:
            velocity = self._velocity(piece)
            first, second = self._derivatives(piece)
            coefficients, domain = series.coef, series.domain
            scale = 2.0 / (domain[1] - domain[0])
            middle = 0.5 * (domain[0] + domain[1])

            def rates(
                arc,
                y,
                velocity=velocity,
                first=first,
                second=second,
                coefficients=coefficients,
                middle=middle,
                scale=scale,
            ):
                x = (arc - middle) * scale
                r = float(np.sqrt(np.polynomial.chebyshev.chebval(x, coefficients)))
                v = float(velocity(r))
                d1, d2 = float(first(r)), float(second(r))
                sin2 = (p * v / r) ** 2
                across = (d2 * sin2 + d1 * max(1.0 - sin2, 0.0) / r, d1 / r)
                return [
                    -sense * across[0] * y[2] / (v * v),
                    -sense * across[1] * y[3] / (v * v),
                    sense * v * y[0],
                    sense * v * y[1],
                ]

            span = (low, high) if sense > 0.0 else (high, low)
            solution = solve_ivp(
                rates,
                span,
                state,
                method="DOP853",
                rtol=rtol,
                atol=atol,
                dense_output=True,
            )
            if not solution.success:
                raise RuntimeError(f"Dynamic ray tracing failed: {solution.message}")
            values[mine] = solution.sol(along[mine]).T
            state = solution.y[:, -1]
        return values


def _chopped(series: Chebyshev, /) -> Chebyshev:
    """A Chebyshev series cut after its last coefficient above :data:`_CHOP`
    times its largest: beyond, the coefficients are rounding."""
    size = np.abs(series.coef)
    last = int(np.flatnonzero(size > _CHOP * size.max())[-1])
    return Chebyshev(series.coef[: last + 1], domain=series.domain)


def _spread(breaks: np.ndarray, count: int, /) -> tuple[np.ndarray, np.ndarray]:
    """``count`` points from the first break to the last, every break among
    them, at equal increments from each break to the next: the intervals are
    shared among the stretches in proportion to their lengths, at least one
    each. Returns the points and the index of each break among them.

    Raises:
        ValueError: if there are too few points for one at every break.
    """
    spans = np.diff(breaks)
    intervals = count - 1
    if intervals < len(spans):
        raise ValueError(
            f"A ray that crosses {len(spans) - 1} radii at which pieces meet needs "
            f"at least {len(spans) + 1} nodes, one at each end and each crossing; "
            f"{count} were asked for."
        )
    ideal = intervals * spans / spans.sum()
    share = np.maximum(np.floor(ideal).astype(int), 1)
    # What is left to the stretches furthest below their share; or, where the
    # one interval each took more than there are, back from those furthest
    # above theirs.
    while share.sum() < intervals:
        share[np.argmax(ideal - share)] += 1
    while share.sum() > intervals:
        share[np.argmax(np.where(share > 1, share - ideal, -np.inf))] -= 1
    points = np.concatenate(
        [
            *(
                np.linspace(low, high, n + 1)[:-1]
                for low, high, n in zip(breaks[:-1], breaks[1:], share)
            ),
            breaks[-1:],
        ]
    )
    return points, np.concatenate([[0], np.cumsum(share)])


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


def linear_gradient_derivatives(v0: float, gradient: float, /, *, radius: float = 1.0):
    """The first two radial derivatives of :func:`linear_gradient`, as a
    :class:`RadialProfile` takes them: ``-gradient`` and zero, whatever
    ``v0`` and ``radius``, which are taken so that the two are called
    alike."""

    def first(r: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(r, dtype=float), -gradient)

    def second(r: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(r, dtype=float))

    return first, second


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


def flattened_linear_gradient_derivatives(v0: float, gradient: float, /):
    """The first two radial derivatives of :func:`flattened_linear_gradient`
    with the same arguments, ``v0 - gradient (1 + ln r)`` and ``-gradient /
    r``, as a :class:`RadialProfile` takes them."""

    def first(r: np.ndarray) -> np.ndarray:
        return v0 - gradient * (1.0 + np.log(np.asarray(r, dtype=float)))

    def second(r: np.ndarray) -> np.ndarray:
        return -gradient / np.asarray(r, dtype=float)

    return first, second


@dataclass(frozen=True)
class CircularArcs:
    """The closed-form rays of the flat medium ``v(z) = v0 + gradient z``.

    A ray of horizontal slowness ``p`` is the arc of the circle of radius
    ``1 / (p g)`` centered a height ``v0 / g`` above the surface, which reaches
    a horizontal distance ``X = 2 sqrt(1 - p^2 v0^2) / (p g)`` in the time
    ``T = (2 / g) arccosh(1 / (p v0))``. From the circle's lowest point up to
    a depth ``z`` it takes the time ``arccosh(1 / (p v(z))) / g``.
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

    def from_depth(
        self, distance: float, depth: float, /
    ) -> tuple[float, tuple[float, float, float], float]:
        """The ray from a source at a depth to a receiver on the surface a
        horizontal distance away: its ``p``, its circle as :meth:`circle`
        gives it, with the source at ``x = 0``, and its travel time.

        The circle's center is the point a height ``v0 / g`` above the surface
        equidistant from the two ends. When it lies between them the ray
        turns between them, and its time is that of its two legs up from the
        circle's lowest point.
        """
        height = self.v0 / self.gradient
        center = (distance**2 + height**2 - (depth + height) ** 2) / (2.0 * distance)
        radius = float(np.hypot(center, depth + height))
        p = 1.0 / (radius * self.gradient)
        legs = [
            np.arccosh(1.0 / (p * (self.v0 + self.gradient * z))) for z in (depth, 0.0)
        ]
        return p, (center, -height, radius), float(sum(legs) / self.gradient)

    def between_depths(
        self, distance: float, source_depth: float, receiver_depth: float, /
    ) -> tuple[float, tuple[float, float, float], float]:
        """The ray between two points at depth a horizontal distance apart: its
        ``p``, its circle as :meth:`circle` gives it, with the source at ``x =
        0``, and its travel time.

        The circle's center ``x_c`` is the point a height ``v0 / g`` above the
        surface equidistant from the two ends, and along the circle, of
        radius ``R``, the time from its lowest point to the point at ``x`` is
        ``artanh((x - x_c) / R) / g``, negative before the lowest point and
        positive after. The time between the ends is the difference, whether
        the lowest point lies between them or not. The arguments broadcast;
        the distance must be positive.
        """
        height = self.v0 / self.gradient
        source, receiver = source_depth + height, receiver_depth + height
        center = (distance**2 + receiver**2 - source**2) / (2.0 * distance)
        radius = np.hypot(center, source)
        time = np.arctanh((distance - center) / radius) + np.arctanh(center / radius)
        p = 1.0 / (radius * self.gradient)
        return p, (center, -height, radius), time / self.gradient


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
