"""
Bounded domains, by embedding in a padded periodic box.

v1 builds the line from the circle and the plane from the torus in exactly this
way, and for the same reason: a Fourier basis is periodic, so representing a
function on a bounded domain means putting it inside a larger periodic one with
enough padding that the wrap-around never reaches it.

The space is therefore the *same* space as its enclosing box — same components,
same metric, same operators — differing only in geometry: which physical point
a grid index corresponds to, where a random point comes from, and which part of
the grid is the domain proper rather than padding. Making it a subclass rather
than a wrapper says that plainly, and means every invariant operator and
measure carries over untouched.

The support assumption is real and worth stating: a field is taken to vanish
outside the domain. Padding is what keeps that assumption from being violated
by periodic wrap-around, and too little of it is the usual cause of a spurious
correlation between the two ends of an interval.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Hashable, Sequence

import numpy as np
from numpy.random import Generator

from ..algebra.operators import LinearOperator
from ..traits import Traits
from .fourier import PeriodicBox

__all__ = ["Box", "Interval"]


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
            self._length_scale,
        )

    def __repr__(self) -> str:
        kind = "Lebesgue" if self._order == 0.0 else f"Sobolev(order={self._order})"
        return f"Box({self._shape}, bounds={self._bounds}, {kind})"

    def _to_enclosing(self, point: Any) -> np.ndarray:
        """Map a point of the domain into the enclosing periodic box."""
        position = np.atleast_1d(np.asarray(point, dtype=float))
        if position.shape != (self.spatial_dimension,):
            raise ValueError(
                f"A point needs {self.spatial_dimension} coordinates, got "
                f"{position.shape}."
            )
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

    def basis_at(self, point: Any, /) -> np.ndarray:
        """The basis functions at a point of the domain."""
        return super().basis_at(self._to_enclosing(point))

    def random_point(self, *, rng: Generator | None = None) -> np.ndarray:
        """A point drawn uniformly from the domain, never from the padding."""
        generator = np.random.default_rng() if rng is None else rng
        return np.array(
            [generator.uniform(lower, upper) for lower, upper in self._bounds]
        )

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

    def project_function(self, function: Callable[[Any], float], /) -> np.ndarray:
        """Sample a function on the grid, taking it to vanish on the padding.

        The function is never called outside the domain, since it need not be
        defined there — which is the practical reason the padding is filled
        with zeros rather than with periodic continuation.
        """
        mesh = np.meshgrid(*self.grid_axes, indexing="ij")
        points = np.stack([m.ravel() for m in mesh], axis=1)
        mask = self.interior_mask.ravel()
        values = np.zeros(points.shape[0])
        for index in np.flatnonzero(mask):
            point = points[index]
            values[index] = float(function(point if point.size > 1 else point[0]))
        return values.reshape(self._shape)

    def support_projection(self) -> LinearOperator:
        """Multiplication by the domain's indicator, as an operator.

        Self-adjoint and idempotent on a Lebesgue space, so an orthogonal
        projection onto fields supported in the domain — and the traits say so.

        On a **Sobolev** space it is neither: multiplying by a discontinuous
        mask does not commute with the metric. Lifting it there with
        :func:`~pygeoinf2.spaces.invariant.lift_formal_adjoint` gives the right
        adjoint and claims nothing about symmetry, which is the honest outcome
        and the same point as DESIGN.md 3.5.
        """
        mask = self.interior_mask.astype(float)
        traits = (
            Traits.SELF_ADJOINT | Traits.IDEMPOTENT
            if self._order == 0.0
            else Traits.NONE
        )
        if self._order != 0.0:
            raise ValueError(
                "A support projection is self-adjoint only on a Lebesgue "
                "space. Build it on this space's Lebesgue counterpart and lift "
                "it with lift_formal_adjoint, which will claim no symmetry."
            )
        return LinearOperator.self_adjoint(self, lambda x: mask * x, traits=traits)


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
    """A one-dimensional bounded domain: v1's ``line`` space.

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
