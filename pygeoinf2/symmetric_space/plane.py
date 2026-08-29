"""A bounded rectangle, embedded in a padded periodic domain.

v1's ``plane`` space, and the two-dimensional companion to
:mod:`~pygeoinf2.symmetric_space.line`. See that module on why a bounded domain
is a :class:`~pygeoinf2.symmetric_space.box.Box` rather than a symmetric space
in its own right, and on what the padding is for.
"""

from __future__ import annotations

from typing import Sequence

from .box import Box

__all__ = ["Plane", "Lebesgue", "Sobolev"]


class Plane(Box):
    """A field on a bounded rectangle."""

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
            shape: grid points along each of the two axes, spanning the
                *padded* domain.
            bounds: the ``(lower, upper)`` extent on each axis.
            padding: periodic padding on each side of each axis.
            order: the Sobolev order. Zero gives the Lebesgue space.
            length_scale: the length at which the Sobolev weight turns over.

        Raises:
            ValueError: if the shape is not two-dimensional.
        """
        shape = tuple(int(n) for n in shape)
        if len(shape) != 2:
            raise ValueError(f"A plane has two axes, got {len(shape)}.")
        super().__init__(
            shape,
            bounds=bounds,
            padding=padding,
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
    ) -> "Plane":
        """The same rectangle with some of its parameters changed.

        Overridden so that ``with_order`` and ``with_shape`` give back a plane
        of the right D-3 subclass rather than a bare
        :class:`~pygeoinf2.symmetric_space.box.Box`.

        Args:
            shape: the new grid, two axes. Unchanged if omitted.
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
        return Sobolev(
            shape, order, scale, bounds=self._bounds, padding=self._padding
        )


class Lebesgue(Plane):
    """The ``L2`` space on a rectangle."""

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
            shape: grid points along each axis.
            bounds: the extent on each axis.
            padding: periodic padding on each side of each axis.
        """
        super().__init__(shape, bounds=bounds, padding=padding, order=0.0)


class Sobolev(Plane):
    """The Sobolev space ``H^order`` on a rectangle."""

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
            shape: grid points along each axis.
            order: the Sobolev order.
            length_scale: the length at which the Sobolev weight turns over.
            bounds: the extent on each axis.
            padding: periodic padding on each side of each axis.
        """
        super().__init__(
            shape,
            bounds=bounds,
            padding=padding,
            order=order,
            length_scale=length_scale,
        )
