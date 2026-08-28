"""A bounded interval, embedded in a padded periodic domain.

v1's ``line`` space. One of the geometries of DESIGN.md section 13, given its
own module so that the space a problem is posed on is named by its type.

An interval is not homogeneous, so it is not a symmetric space in its own
right; it is a :class:`~pygeoinf2.symmetric_space.box.Box` of one axis, which
embeds it in a periodic domain large enough that the periodicity does not reach
back into the region of interest. The padding is what buys that, and it is why
the constructors take one.
"""

from __future__ import annotations

from .box import Box

__all__ = ["Line", "Lebesgue", "Sobolev"]


class Line(Box):
    """A field on a bounded interval."""

    def __init__(
        self,
        points: int,
        /,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
        padding: float | None = None,
        order: float = 0.0,
        length_scale: float = 1.0,
    ) -> None:
        """
        Args:
            points: grid points spanning the *padded* interval.
            lower: the left endpoint.
            upper: the right endpoint.
            padding: periodic padding at each end. A tenth of the interval's
                length by default, which is enough for a correlation length
                well below it and not enough for one comparable to it.
            order: the Sobolev order. Zero gives the Lebesgue space.
            length_scale: the length at which the Sobolev weight turns over.
        """
        super().__init__(
            (points,),
            bounds=((lower, upper),),
            padding=None if padding is None else (padding,),
            order=order,
            length_scale=length_scale,
        )


class Lebesgue(Line):
    """The ``L2`` space on an interval."""

    def __init__(
        self,
        points: int,
        /,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
        padding: float | None = None,
    ) -> None:
        """
        Args:
            points: grid points spanning the padded interval.
            lower, upper: the endpoints.
            padding: periodic padding at each end.
        """
        super().__init__(points, lower=lower, upper=upper, padding=padding, order=0.0)


class Sobolev(Line):
    """The Sobolev space ``H^order`` on an interval."""

    def __init__(
        self,
        points: int,
        order: float,
        length_scale: float,
        /,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
        padding: float | None = None,
    ) -> None:
        """
        Args:
            points: grid points spanning the padded interval.
            order: the Sobolev order.
            length_scale: the length at which the Sobolev weight turns over.
            lower, upper: the endpoints.
            padding: periodic padding at each end.
        """
        super().__init__(
            points,
            lower=lower,
            upper=upper,
            padding=padding,
            order=order,
            length_scale=length_scale,
        )
