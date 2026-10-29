"""
Bounded domains, by embedding in a padded periodic box.

A Fourier basis is periodic, so representing a function on a bounded domain
means putting it inside a larger periodic one, with enough padding that the
wrap-around never reaches it.

The space is therefore the *same* space as its enclosing box — same components,
same metric, same operators — differing only in geometry: which physical point
a grid index corresponds to, where a random point comes from, and which part of
the grid is the domain proper rather than padding. Making it a subclass rather
than a wrapper says that plainly, and means every invariant operator and
measure carries over untouched.

The padding is worth sizing with care. A field lives on the whole periodic
box, so the two ends of the domain meet across the padding: a prior correlates
them unless the padding is wide compared with its correlation length, and too
little padding is the usual cause of a spurious correlation between the two
ends of an interval. :meth:`Box.project_function` rolls a sampled function off
to zero across the padding, and :meth:`Box.support_projection` zeroes it.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Hashable, Sequence

import numpy as np
from numpy.random import Generator

from ..algebra.operators import LinearOperator
from ..algebra.spaces import resolve_rng
from ..traits import Traits
from .base import PreparedPoints
from .fourier import PeriodicBox

__all__ = ["Box", "Interval", "Lebesgue", "Sobolev"]

#: Relative allowance on the boundary of the domain: a point this close
#: outside counts as on it, and one further out is refused.
_POSITION_RTOL = 1e-12


class Box(PeriodicBox):
    """A field on a bounded box, embedded in a padded periodic one."""

    def __init__(
        self,
        shape: Sequence[int],
        /,
        *,
        bounds: Sequence[tuple[float, float]],
        padding: Sequence[float] | float | None = None,
        order: float = 0.0,
        length_scale: float = 1.0,
    ) -> None:
        """
        Args:
            shape: grid points along each axis, spanning the *padded* domain.
            bounds: the ``(lower, upper)`` extent of the domain on each axis.
            padding: how much periodic padding to add on each side of each
                axis. A single number applies to every axis; the default is a
                tenth of each extent, which is enough for a correlation length
                well below the domain size and not enough for one comparable to
                it.
            order: the Sobolev order, as for :class:`PeriodicBox`.
            length_scale: the Sobolev length scale.
        """
        bounds = tuple((float(a), float(b)) for a, b in bounds)
        if len(bounds) != len(tuple(shape)):
            raise ValueError(f"Got {len(bounds)} bounds for {len(tuple(shape))} axes.")
        extents = tuple(b - a for a, b in bounds)
        if any(extent <= 0.0 for extent in extents):
            raise ValueError("Every bound must have its lower end first.")

        if padding is None:
            padding = tuple(0.1 * extent for extent in extents)
        elif np.isscalar(padding):
            padding = tuple(float(padding) for _ in extents)
        else:
            padding = tuple(float(p) for p in padding)
        if len(padding) != len(extents):
            raise ValueError(f"Got {len(padding)} paddings for {len(extents)} axes.")
        if any(p < 0.0 for p in padding):
            raise ValueError("Padding must be non-negative.")

        self._bounds = bounds
        self._padding = padding
        super().__init__(
            shape,
            lengths=tuple(e + 2.0 * p for e, p in zip(extents, padding)),
            order=order,
            length_scale=length_scale,
        )

    # ----------------------------------------------------------------- #
    #                              Geometry                             #
    # ----------------------------------------------------------------- #

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """The domain's extent on each axis, padding excluded."""
        return self._bounds

    @property
    def padding(self) -> tuple[float, ...]:
        """The periodic padding added on each side of each axis."""
        return self._padding

    @property
    def domain_volume(self) -> float:
        """The measure of the domain proper, excluding the padding."""
        return float(np.prod([b - a for a, b in self._bounds]))

    def _key(self) -> Hashable:
        return (
            self._shape,
            self._bounds,
            self._padding,
            self._order,
            # The length scale enters the metric only above order zero.
            self._length_scale if self._order != 0.0 else None,
        )

    def _coordinate_key(self) -> Hashable:
        """The grid and where its points are, which the metric does not touch.

        Tagged by geometry rather than by ``type(self)``, as the sphere's and
        the periodic box's are: ``Line``, ``Plane``, ``Lebesgue`` and
        ``Sobolev`` are thin subclasses over one grid and one point map. The
        bounds and padding stay in the key because they *are* the point map --
        a bounded box and its enclosing periodic one share a grid but not an
        idea of where a grid index sits.
        """
        return ("box", self._shape, self._bounds, self._padding)

    def __repr__(self) -> str:
        kind = "Lebesgue" if self._order == 0.0 else f"Sobolev(order={self._order})"
        return f"Box({self._shape}, bounds={self._bounds}, {kind})"

    def _domain_key(self) -> Hashable:
        """The bounds and the padding, which place the domain in the
        enclosing box: two boxes of one period on different bounds hold
        different fields under the same modes."""
        return ("box", self._bounds, self._padding)

    def _require_inside(self, positions: np.ndarray, /) -> None:
        """Refuse any row of *positions* that lies outside the domain.

        The padding is a numerical device, not part of the domain, and a
        point beyond it would be wrapped round the enclosing period onto
        another point altogether; either way the value read there is not the
        field's at the point asked for. A point within a relative ``1e-12``
        of the boundary counts as on it.

        Raises:
            ValueError: naming the domain and the first point outside it.
        """
        lower = np.array([a for a, _ in self._bounds])
        upper = np.array([b for _, b in self._bounds])
        slack = _POSITION_RTOL * (upper - lower)
        outside = np.any(
            (positions < lower - slack) | (positions > upper + slack), axis=1
        )
        if np.any(outside):
            domain = " x ".join(f"[{a:g}, {b:g}]" for a, b in self._bounds)
            point = ", ".join(f"{c:g}" for c in positions[outside][0])
            raise ValueError(
                f"Every point must lie in the domain, {domain}; one is at "
                f"({point})."
            )

    def _to_enclosing(self, point: Any, /, *, check: bool = True) -> np.ndarray:
        """Map a point of the domain into the enclosing periodic box.

        Raises:
            ValueError: if the point lies outside the domain, unless *check*
                is false, which only a quadrature reaching into the padding
                on purpose asks for.
        """
        position = np.atleast_1d(np.asarray(point, dtype=float))
        if position.shape != (self.spatial_dimension,):
            raise ValueError(
                f"A point needs {self.spatial_dimension} coordinates, got "
                f"{position.shape}."
            )
        if check:
            self._require_inside(position[None, :])
        lower = np.array([a for a, _ in self._bounds])
        return position - lower + np.asarray(self._padding)

    @cached_property
    def grid_axes(self) -> tuple[np.ndarray, ...]:
        """Sample coordinates along each axis, in the domain's own coordinates.

        These run from ``lower - padding`` to just short of ``upper + padding``,
        so the domain proper occupies the middle of each axis.
        """
        return tuple(
            (a - p) + np.arange(n) * length / n
            for (a, _), p, n, length in zip(
                self._bounds, self._padding, self._shape, self._lengths
            )
        )

    @property
    def reference_point(self) -> np.ndarray:
        """The grid point nearest the domain's lower corner, inside it.

        A grid point, as :meth:`pointwise_variance` needs, and a point of the
        domain, where evaluation is allowed.
        """
        corner = []
        for axis, (lower, upper) in zip(self.grid_axes, self._bounds):
            inside = axis[axis >= lower - _POSITION_RTOL * (upper - lower)]
            corner.append(float(np.min(inside)))
        return np.array(corner)

    def basis_at(self, point: Any, /) -> np.ndarray:
        """The basis functions at a point of the domain.

        Raises:
            ValueError: if the point lies outside the domain.
        """
        return super().basis_at(self._to_enclosing(point))

    def basis_matrix(
        self, points: Sequence[Any], /, *, n_jobs: int | None = None
    ) -> np.ndarray:
        """The basis at many points of the domain, one row per point.

        Args:
            points: points of the domain, or a set prepared by
                :meth:`prepare_points`, whose conversion is reused.
            n_jobs: workers for the points. Serial by default.

        Raises:
            ValueError: if a point lies outside the domain.
        """
        from functools import partial

        from ..parallel import parallel_map

        prepared = self.prepare_points(points)
        lengths = np.asarray(self._lengths, dtype=float)
        enclosed = np.column_stack(prepared.data) * lengths / (2.0 * np.pi)
        # The enclosing box's basis, by name rather than as ``super()``'s
        # bound method, which a worker process would rebind to this class.
        at = partial(PeriodicBox.basis_at, self)
        return np.stack(parallel_map(at, list(enclosed), n_jobs=n_jobs))

    def prepare_points(self, points: Sequence[Any], /) -> PreparedPoints:
        """The points as angles, through the same seam :meth:`basis_at` uses.

        The enclosing box's grid does not start where the domain does -- the
        padding sits in between -- so a point has to be moved into the
        enclosing coordinates before it becomes an angle. This is the map
        :meth:`basis_at` applies, so the non-uniform FFT route and the direct
        route evaluate at the same place.

        The move is a Python loop over the points, which is why doing it once
        per operator rather than once per application matters most here; an
        array of rows is moved in one step.

        Args:
            points: points of the domain, or an already prepared set.

        Returns:
            The prepared points, carrying one angle array per axis.

        Raises:
            ValueError: if a point lies outside the domain, which the
                spectral-element spaces refuse too: the padding is not part
                of the domain, and a point past it would wrap round.
        """
        return self._prepared(points, check=True)

    def _prepared(self, points: Sequence[Any], /, *, check: bool) -> PreparedPoints:
        """:meth:`prepare_points`, with the domain check optional for the
        quadratures that step into the padding on purpose."""
        if isinstance(points, PreparedPoints):
            return points
        lower = np.array([a for a, _ in self._bounds])
        shift = np.asarray(self._padding) - lower
        if isinstance(points, np.ndarray) and points.ndim == 2:
            # An array of rows moves into the enclosing box in one step.
            rows = np.asarray(points, dtype=float)
            if rows.shape[1] == self.spatial_dimension and check:
                self._require_inside(rows)
            inside = rows + shift
        else:
            points = tuple(points)
            inside = [self._to_enclosing(point, check=check) for point in points]
        return PreparedPoints(points, data=super().prepare_points(inside).data)

    def _prepared_nearby(self, positions: np.ndarray, /) -> PreparedPoints:
        """Positions a finite difference away from points of the domain,
        which may lie a step outside it."""
        return self._prepared(positions, check=False)

    def random_point(self, *, rng: Generator | None = None) -> np.ndarray:
        """A point drawn uniformly from the domain, never from the padding."""
        generator = resolve_rng(rng)
        return np.array(
            [generator.uniform(lower, upper) for lower, upper in self._bounds]
        )

    def _separation(self, start: Any, end: Any, /) -> np.ndarray:
        """``end - start``, straight, never round through the padding.

        A bounded domain is not periodic. It is *embedded* in a periodic one so
        that the spectral basis exists, and the padding is a numerical device,
        not part of the domain -- so the distance between two points is the
        straight-line one, and the periodic short cut through the padding is
        not a path at all.

        The distinction is not academic: with the default padding of a tenth of
        each extent the period is ``1.2`` times the domain, so two points near
        opposite ends are further apart than half a period and
        :class:`~pygeoinf2.symmetric_space.fourier.PeriodicBox` would take the
        wrap.
        """
        return self._as_point(end) - self._as_point(start)

    def _embedding(self, points: Sequence[Any], /) -> tuple[np.ndarray, Any]:
        """The coordinates, with nothing wrapping: see :meth:`_separation`."""
        return np.stack([self._as_point(point) for point in points]), None

    def _wrap(self, point: np.ndarray, /) -> np.ndarray:
        """Points left where they are: a bounded domain does not wrap, for the
        reason :meth:`_separation` gives."""
        return np.asarray(point, dtype=float)

    def geodesic_ball_quadrature(
        self, center: Any, radius: float, /, *, count: int
    ) -> tuple[list[Any], np.ndarray]:
        """Nodes and weights integrating over a ball inside the domain.

        The rule of :class:`~pygeoinf2.symmetric_space.fourier.PeriodicBox`,
        unwrapped. The ball must lie in the domain: past its boundary is the
        padding, which is not part of the domain, so an average reaching
        into it is not an average of the field.

        Args:
            center: the ball's center.
            radius: its radius, in the units of the domain.
            count: roughly how many nodes.

        Returns:
            ``(nodes, weights)``.

        Raises:
            ValueError: for a negative radius, fewer than one node, or a ball
                that leaves the domain.
            NotImplementedError: in three dimensions or more.
        """
        nodes, weights = super().geodesic_ball_quadrature(center, radius, count=count)
        try:
            self._require_inside(np.stack(nodes))
        except ValueError as error:
            raise ValueError(
                f"A ball of radius {radius:g} about "
                f"{tuple(float(c) for c in self._as_point(center))} leaves the "
                f"domain. {error}"
            ) from None
        return nodes, weights

    def with_order(
        self, order: float, /, *, length_scale: float | None = None
    ) -> "Box":
        """The same box, viewed with a different Sobolev order.

        The result is still a ``Box``, with this box's bounds and padding, not
        the enclosing periodic domain, which has the same components but not
        the same idea of where the boundary is.

        Args:
            order: the new Sobolev order.
            length_scale: the new length scale. Kept as it is if omitted,
                which is what makes this a change of *order* alone.

        Returns:
            The same domain and padding in the new metric, as
            :class:`Lebesgue` at order zero and :class:`Sobolev` otherwise.
        """
        return self._rebuilt(order=order, length_scale=length_scale)

    def _rebuilt(
        self,
        /,
        *,
        shape: Sequence[int] | None = None,
        order: float | None = None,
        length_scale: float | None = None,
    ) -> "Box":
        """The same bounded domain with some of its parameters changed.

        Overridden from
        :meth:`~pygeoinf2.symmetric_space.fourier.PeriodicBox._rebuilt` so that
        the result is a ``Box`` and keeps this box's bounds and padding -- the
        base class's version would hand back the enclosing periodic domain,
        which has the same components but not the same idea of where the
        boundary is -- and so that it is the :class:`Lebesgue` or
        :class:`Sobolev` subclass its order names.

        Args:
            shape: the new grid. Unchanged if omitted.
            order: the new Sobolev order. Unchanged if omitted.
            length_scale: the new Sobolev length scale. Unchanged if omitted.

        Returns:
            The space, as ``Lebesgue`` at order zero and ``Sobolev`` otherwise.
        """
        shape = self._shape if shape is None else tuple(int(n) for n in shape)
        order = self._order if order is None else float(order)
        scale = self._length_scale if length_scale is None else float(length_scale)
        if order == 0.0:
            return Lebesgue(shape, bounds=self._bounds, padding=self._padding)
        return Sobolev(shape, order, scale, bounds=self._bounds, padding=self._padding)

    # ----------------------------------------------------------------- #
    #                              Support                              #
    # ----------------------------------------------------------------- #

    @cached_property
    def interior_mask(self) -> np.ndarray:
        """A boolean grid array, true on the domain and false on the padding."""
        masks = [
            (axis >= lower - 1e-12) & (axis <= upper + 1e-12)
            for axis, (lower, upper) in zip(self.grid_axes, self._bounds)
        ]
        mask = masks[0]
        for extra in masks[1:]:
            mask = np.logical_and.outer(mask, extra)
        return mask.reshape(self._shape)

    @cached_property
    def _taper(self) -> np.ndarray:
        """A raised-cosine window: one on the domain, zero past the padding.

        The product of a per-axis window, each going smoothly from one at the
        boundary to zero at the edge of the padding.
        """
        windows = []
        for axis, (lower, upper), pad in zip(
            self.grid_axes, self._bounds, self._padding
        ):
            window = np.ones_like(axis)
            if pad > 0.0:
                below = axis < lower
                above = axis > upper
                # Distance into the padding, as a fraction of its width. The
                # grid is periodic, so a point past the far edge is also
                # "before" the near one; both are handled by measuring from
                # whichever boundary it fell off.
                left = np.clip((lower - axis[below]) / pad, 0.0, 1.0)
                right = np.clip((axis[above] - upper) / pad, 0.0, 1.0)
                window[below] = 0.5 * (1.0 + np.cos(np.pi * left))
                window[above] = 0.5 * (1.0 + np.cos(np.pi * right))
            windows.append(window)

        taper = windows[0]
        for extra in windows[1:]:
            taper = np.multiply.outer(taper, extra)
        return taper.reshape(self._shape)

    def project_function(
        self, function: Callable[[Any], float], /, *, taper: bool = True
    ) -> np.ndarray:
        """Sample a function on the grid, rolling it off across the padding.

        The function is never called outside the domain, since it need not be
        defined there. What fills the padding instead is the nearest boundary
        value, multiplied by a raised-cosine window that reaches zero at the
        far edge — so the sampled field is continuous, and periodic, without
        ``function`` ever being asked about a point it does not have.

        **Why not simply zero.** A hard cutoff puts a step into a field the
        space then represents by a truncated Fourier series, and a step rings.
        Near the boundary the error is then of order several per cent, even
        for a constant function, and it dies away only well inside the
        domain.

        Args:
            function: called with a point of the domain.
            taper: roll off across the padding. ``False`` gives the hard cutoff,
                which is right when the function genuinely vanishes at the
                boundary and there is nothing to ring.

        Returns:
            The sampled field.
        """
        mesh = np.meshgrid(*self.grid_axes, indexing="ij")
        points = np.stack([m.ravel() for m in mesh], axis=1)

        if not taper:
            mask = self.interior_mask.ravel()
            values = np.zeros(points.shape[0])
            for index in np.flatnonzero(mask):
                point = points[index]
                values[index] = float(function(point if point.size > 1 else point[0]))
            return values.reshape(self._shape)

        # Clamp each coordinate into the domain, so a padding point is given
        # the value of the nearest point of the domain and `function` is never
        # asked about anywhere else.
        lower = np.array([a for a, _ in self._bounds])
        upper = np.array([b for _, b in self._bounds])
        clamped = np.clip(points, lower, upper)
        values = np.array(
            [
                float(function(point if point.size > 1 else point[0]))
                for point in clamped
            ]
        )
        return values.reshape(self._shape) * self._taper

    def support_projection(self) -> LinearOperator:
        """Multiplication by the domain's indicator, as an operator.

        Self-adjoint and idempotent on a Lebesgue space, so an orthogonal
        projection onto fields supported in the domain — and the traits say so.

        On a **Sobolev** space it is neither: multiplying by a discontinuous
        mask does not commute with the metric. There it is built on this
        space's Lebesgue counterpart and lifted with
        :func:`~pygeoinf2.symmetric_space.base.lift_formal_adjoint`, which gives
        the right adjoint and claims nothing about symmetry, since the lifted
        operator is not self-adjoint in the Sobolev metric.
        """
        mask = self.interior_mask.astype(float)
        if self._order != 0.0:
            from .base import lift_formal_adjoint

            base = self.with_order(0.0)
            return lift_formal_adjoint(base.support_projection(), self)
        return LinearOperator.self_adjoint(
            self,
            lambda x: mask * x,
            traits=Traits.SELF_ADJOINT | Traits.IDEMPOTENT,
        )


def Interval(
    points: int,
    /,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    padding: float | None = None,
    order: float = 0.0,
    length_scale: float = 1.0,
) -> Box:
    """A one-dimensional bounded domain, embedded in a padded circle.

    A plain :class:`Box` of one axis. :class:`~pygeoinf2.symmetric_space.line.Line`
    builds the same geometry as a class of its own, with ``Lebesgue`` and
    ``Sobolev`` subclasses.

    Args:
        points: grid points spanning the padded interval.
        lower: the left endpoint.
        upper: the right endpoint.
        padding: periodic padding on each side. Defaults to a tenth of the
            interval's length.
        order: the Sobolev order.
        length_scale: the Sobolev length scale.
    """
    return Box(
        (points,),
        bounds=((lower, upper),),
        padding=None if padding is None else (padding,),
        order=order,
        length_scale=length_scale,
    )


class Lebesgue(Box):
    """The ``L2`` space on a bounded box."""

    def __init__(
        self,
        shape: Sequence[int],
        /,
        *,
        bounds: Sequence[tuple[float, float]],
        padding: Sequence[float] | float | None = None,
    ) -> None:
        """
        Args:
            shape: grid points along each axis, spanning the padded domain.
            bounds: the ``(lower, upper)`` extent on each axis.
            padding: periodic padding on each side of each axis.
        """
        super().__init__(shape, bounds=bounds, padding=padding, order=0.0)


class Sobolev(Box):
    """The Sobolev space ``H^order`` on a bounded box."""

    def __init__(
        self,
        shape: Sequence[int],
        order: float,
        length_scale: float,
        /,
        *,
        bounds: Sequence[tuple[float, float]],
        padding: Sequence[float] | float | None = None,
    ) -> None:
        """
        Args:
            shape: grid points along each axis, spanning the padded domain.
            order: the Sobolev order.
            length_scale: the length at which the Sobolev weight turns over.
            bounds: the ``(lower, upper)`` extent on each axis.
            padding: periodic padding on each side of each axis.
        """
        super().__init__(
            shape,
            bounds=bounds,
            padding=padding,
            order=order,
            length_scale=length_scale,
        )
