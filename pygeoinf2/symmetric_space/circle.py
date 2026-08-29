"""The circle: a one-dimensional periodic domain.

One of the geometries of DESIGN.md section 13, given its own module so that
the space a problem is posed on is named by its type. ``Circle`` is a
:class:`~pygeoinf2.symmetric_space.fourier.PeriodicBox` of one axis and nothing
else, and everything the box provides is available here unchanged.
"""

from __future__ import annotations

from typing import Sequence

from .fourier import PeriodicBox

__all__ = ["Circle", "Lebesgue", "Sobolev"]


class Circle(PeriodicBox):
    """A field on a circle, expanded in a Fourier series."""

    def __init__(
        self,
        points: int,
        /,
        *,
        length: float = 1.0,
        order: float = 0.0,
        length_scale: float = 1.0,
    ) -> None:
        """
        Args:
            points: grid points around the circle.
            length: the circumference.
            order: the Sobolev order. Zero gives the Lebesgue space.
            length_scale: the length at which the Sobolev weight turns over.
        """
        super().__init__(
            (points,),
            lengths=(length,),
            order=order,
            length_scale=length_scale,
        )

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

    def __init__(self, points: int, /, *, length: float = 1.0) -> None:
        """
        Args:
            points: grid points around the circle.
            length: the circumference.
        """
        super().__init__(points, length=length, order=0.0)


class Sobolev(Circle):
    """The Sobolev space ``H^order`` on a circle."""

    def __init__(
        self,
        points: int,
        order: float,
        length_scale: float,
        /,
        *,
        length: float = 1.0,
    ) -> None:
        """
        Args:
            points: grid points around the circle.
            order: the Sobolev order.
            length_scale: the length at which the Sobolev weight turns over.
            length: the circumference.
        """
        super().__init__(points, length=length, order=order, length_scale=length_scale)
