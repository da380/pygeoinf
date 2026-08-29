"""The two-dimensional torus: a doubly periodic domain.

One of the geometries of DESIGN.md section 13, given its own module so that
the space a problem is posed on is named by its type. ``Torus`` is a
:class:`~pygeoinf2.symmetric_space.fourier.PeriodicBox` of two axes.
"""

from __future__ import annotations

from typing import Sequence

from .fourier import PeriodicBox

__all__ = ["Torus", "Lebesgue", "Sobolev"]


class Torus(PeriodicBox):
    """A field on a two-dimensional torus, expanded in a Fourier series."""

    def __init__(
        self,
        shape: Sequence[int],
        /,
        *,
        lengths: Sequence[float] | None = None,
        order: float = 0.0,
        length_scale: float = 1.0,
    ) -> None:
        """
        Args:
            shape: grid points along each of the two axes.
            lengths: the period along each axis. Unit lengths by default.
            order: the Sobolev order. Zero gives the Lebesgue space.
            length_scale: the length at which the Sobolev weight turns over.

        Raises:
            ValueError: if the shape is not two-dimensional.
        """
        shape = tuple(int(n) for n in shape)
        if len(shape) != 2:
            raise ValueError(f"A torus has two axes, got {len(shape)}.")
        super().__init__(shape, lengths=lengths, order=order, length_scale=length_scale)

    def _rebuilt(
        self,
        /,
        *,
        shape: Sequence[int] | None = None,
        order: float | None = None,
        length_scale: float | None = None,
    ) -> "Torus":
        """The same torus with some of its parameters changed.

        Overridden so that ``with_order`` and ``with_shape`` give back a torus
        of the right D-3 subclass rather than a bare
        :class:`~pygeoinf2.symmetric_space.fourier.PeriodicBox`.

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
            return Lebesgue(shape, lengths=self._lengths)
        return Sobolev(shape, order, scale, lengths=self._lengths)


class Lebesgue(Torus):
    """The ``L2`` space on a torus."""

    def __init__(
        self, shape: Sequence[int], /, *, lengths: Sequence[float] | None = None
    ) -> None:
        """
        Args:
            shape: grid points along each of the two axes.
            lengths: the period along each axis.
        """
        super().__init__(shape, lengths=lengths, order=0.0)


class Sobolev(Torus):
    """The Sobolev space ``H^order`` on a torus."""

    def __init__(
        self,
        shape: Sequence[int],
        order: float,
        length_scale: float,
        /,
        *,
        lengths: Sequence[float] | None = None,
    ) -> None:
        """
        Args:
            shape: grid points along each of the two axes.
            order: the Sobolev order.
            length_scale: the length at which the Sobolev weight turns over.
            lengths: the period along each axis.
        """
        super().__init__(shape, lengths=lengths, order=order, length_scale=length_scale)
