"""The circle: a one-dimensional periodic domain.

One of the geometries of DESIGN.md section 13, given its own module so that
the space a problem is posed on is named by its type. ``Circle`` is a
:class:`~pygeoinf2.symmetric_space.fourier.PeriodicBox` of one axis and nothing
else, and everything the box provides is available here unchanged.

**Points are arc lengths, not angles.** A point is a physical coordinate in
``[0, L)`` with ``L`` the circumference, as on every box. The default is the
unit circle, ``L == 2 pi``, where the coordinate *is* the angle and v1's
numbers are these; on a circle of radius ``r`` a v1 angle is ``r`` times
too small. ``geodesic_distance`` and the argument ``project_function`` hands
its callback are in these units (DESIGN §46).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .fourier import PeriodicBox

__all__ = ["Circle", "Lebesgue", "Sobolev"]


class Circle(PeriodicBox):
    """A field on a circle, expanded in a Fourier series."""

    def __init__(
        self,
        points: int,
        /,
        *,
        length: float | None = None,
        radius: float | None = None,
        order: float = 0.0,
        length_scale: float = 1.0,
    ) -> None:
        """
        Args:
            points: grid points around the circle.
            length: the circumference. ``2 pi``, the unit circle, if neither
                this nor *radius* is given.
            radius: the radius, as an alternative to the circumference;
                ``length == 2 pi radius``. v1's parameter.
            order: the Sobolev order. Zero gives the Lebesgue space.
            length_scale: the length at which the Sobolev weight turns over.

        Raises:
            ValueError: if both *length* and *radius* are given, or the radius
                is not positive.
        """
        super().__init__(
            (points,),
            lengths=(_period(length, radius),),
            order=order,
            length_scale=length_scale,
        )

    @property
    def radius(self) -> float:
        """The circumference over ``2 pi``."""
        return float(self.lengths[0]) / (2.0 * np.pi)

    def _rebuilt(
        self,
        /,
        *,
        shape: Sequence[int] | None = None,
        order: float | None = None,
        length_scale: float | None = None,
    ) -> "Circle":
        """The same circle with some of its parameters changed.

        Overridden so that ``with_order`` and ``with_shape`` give back a circle
        of the right D-3 subclass rather than a bare
        :class:`~pygeoinf2.symmetric_space.fourier.PeriodicBox`.

        Args:
            shape: the new grid, one axis. Unchanged if omitted.
            order: the new Sobolev order. Unchanged if omitted.
            length_scale: the new Sobolev length scale. Unchanged if omitted.

        Returns:
            The space, as ``Lebesgue`` at order zero and ``Sobolev`` otherwise.
        """
        shape = self._shape if shape is None else tuple(int(n) for n in shape)
        order = self._order if order is None else float(order)
        scale = self._length_scale if length_scale is None else float(length_scale)
        length = self._lengths[0]
        if order == 0.0:
            return Lebesgue(shape[0], length=length)
        return Sobolev(shape[0], order, scale, length=length)


class Lebesgue(Circle):
    """The ``L2`` space on a circle."""

    def __init__(
        self,
        points: int,
        /,
        *,
        length: float | None = None,
        radius: float | None = None,
    ) -> None:
        """
        Args:
            points: grid points around the circle.
            length: the circumference; ``2 pi`` by default.
            radius: the radius, as an alternative.

        Raises:
            ValueError: if both are given, or the radius is not positive.
        """
        super().__init__(points, length=length, radius=radius, order=0.0)


class Sobolev(Circle):
    """The Sobolev space ``H^order`` on a circle."""

    def __init__(
        self,
        points: int,
        order: float,
        length_scale: float,
        /,
        *,
        length: float | None = None,
        radius: float | None = None,
    ) -> None:
        """
        Args:
            points: grid points around the circle.
            order: the Sobolev order.
            length_scale: the length at which the Sobolev weight turns over.
            length: the circumference; ``2 pi`` by default.
            radius: the radius, as an alternative.

        Raises:
            ValueError: if both are given, or the radius is not positive.
        """
        super().__init__(
            points,
            length=length,
            radius=radius,
            order=order,
            length_scale=length_scale,
        )


def _period(length: float | None, radius: float | None, /) -> float:
    """The circumference from whichever of the two was given.

    Points on the circle are physical arc-length coordinates in ``[0, L)``,
    never angles; on the unit circle, the default, the two coincide, and on
    any other radius a v1 angle is this coordinate divided by the radius.
    """
    if length is not None and radius is not None:
        raise ValueError("Give the circumference or the radius, not both.")
    if radius is not None:
        if radius <= 0.0:
            raise ValueError(f"The radius must be positive, got {radius}.")
        return 2.0 * np.pi * float(radius)
    return 2.0 * np.pi if length is None else float(length)
