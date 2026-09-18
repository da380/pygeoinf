"""Functions of radius alone, in a ball or a shell: the measure is ``r^2 dr``.

A radial profile -- a density, a wave speed, a viscosity as a one-dimensional
reference model gives them -- is a field in three dimensions that happens not
to depend on direction, and this is the space of such fields: the part of
degree zero of :mod:`~pygeoinf2.sem1d.ball`, on its own and without
an angular grid. It is *not* the :mod:`~pygeoinf2.sem1d.interval`,
whose measure is ``dx``: there a prior's variance and every ``L2`` product
weigh each radius alike, and here they weigh it by the area of its sphere.

The inner product is the volume integral, ``4 pi int u v r^2 dr``, so a
profile has the norm it has as a field in the ball, and its components *are*
the ball's components of degree zero: a profile and the ball field it names
can be passed between the two spaces by copying them.

The operator is the radial part of ``A = 1 - div(L^2 grad)`` at degree zero,
``A u = u - r^-2 (r^2 L^2 u')'``. The mesh is padded outwards, and inwards as
far as the centre and no further: the centre is a regular point of the
operator, needing neither padding nor a condition, so a profile in a whole
ball is padded on the outside only. The eigenfunctions are even in ``r``
there -- for constant ``L`` they are ``sin(k r) / r`` -- which is what a field
regular at the centre has to be at degree zero (DECISIONS.md D-117), so every
draw of a prior is regular for nothing; a profile *handed in* should be even
in ``r`` too, and one linear in ``r`` is a cone.

**A point is a radius.** Its value is the field's on the whole sphere of that
radius, so away from the centre evaluation needs an order above one half, as
on a line. The centre itself is a point of three-dimensional space, and needs
three halves.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.random import Generator

from .axis import AxisSpace
from .base import Boundary, LengthScale

__all__ = ["Radial", "Lebesgue", "Sobolev"]


class Radial(AxisSpace):
    """Functions of radius on ``[inner_radius, radius]`` under ``r^2 dr``, in
    the leading eigenfunctions of ``A``."""

    _weight = "r2"
    _scale = float(np.sqrt(4.0 * np.pi))
    _geometry = "radial"

    def __init__(
        self,
        modes: int,
        /,
        *,
        radius: float = 1.0,
        inner_radius: float = 0.0,
        order: float = 0.0,
        length_scale: LengthScale = 1.0,
        padding: float | tuple[float, float] | None = None,
        boundary: Boundary = "robin",
        ngll: int = 5,
        element_length: float | None = None,
    ) -> None:
        """
        Args:
            modes: how many eigenfunctions of ``A`` on the padded mesh to
                keep, which is the dimension of the space.
            radius: the outer radius.
            inner_radius: the inner radius. Zero, the default, is a whole
                ball.
            order: the Sobolev order. Zero gives the Lebesgue space.
            length_scale: ``L`` in ``A``, a number or a function of radius. It
                is asked about the domain only; the padding takes its value
                at the nearer end.
            padding: how far the mesh extends past each end, one length or an
                ``(inwards, outwards)`` pair; inwards it stops at the centre.
                Two length scales by default; four under the natural condition.
            boundary: the condition at the ends of the mesh. ``"robin"``, the
                default, ends the mesh as if it went on, ``u' = -+ (0.7 / L +-
                1 / r) u`` with the curvature of the sphere at each end:
                measured, two length scales of padding under it do what four
                do under the natural condition (DECISIONS.md D-122). ``None``
                is the natural condition, and doubles the default padding. A
                number or a pair gives the Robin coefficients themselves. None
                is applied at the centre.
            ngll: Gauss-Lobatto-Legendre nodes per element.
            element_length: the longest an element may be. By default short
                enough that the mesh has two and a half nodes for every mode
                kept and an element is no longer than half the smallest
                length scale.

        Raises:
            ValueError: if the radii are out of order, a length scale is not
                positive, or the mesh cannot hold the modes asked for.
        """
        inner, outer = float(inner_radius), float(radius)
        if not 0.0 <= inner < outer:
            raise ValueError("The radii must satisfy 0 <= inner_radius < radius.")
        super().__init__(
            modes,
            inner,
            outer,
            order=order,
            length_scale=length_scale,
            padding=padding,
            boundary=boundary,
            ngll=ngll,
            element_length=element_length,
        )

    @property
    def radius(self) -> float:
        """The outer radius."""
        return self._bounds[1]

    @property
    def inner_radius(self) -> float:
        """The inner radius: zero for a whole ball."""
        return self._bounds[0]

    @property
    def spatial_dimension(self) -> int:
        """Three: a profile is a field in a ball."""
        return 3

    def point_evaluation_order(
        self, /, *, points: Sequence[float] | None = None
    ) -> float:
        """Three halves at the centre, one half anywhere else.

        A radius names a whole sphere, and the field along it is a function of
        one variable, so away from the centre a point has a value above order
        one half, as on a line. The centre is a point of space, and needs
        three halves. Measured: the norm of the Dirac's representer at the
        centre still grows with the modes at order 1.25 and has settled at
        1.75, and at ``r = 0.5`` has settled by 0.75 (DECISIONS.md D-121).

        Args:
            points: the radii in question. Any radius of the domain, the
                centre included if it is in it, when omitted.

        Returns:
            The order that must be exceeded.
        """
        if points is None:
            return 1.5 if self._bounds[0] == 0.0 else 0.5
        radii = np.asarray(points, dtype=float).reshape(-1)
        return 1.5 if np.any(radii <= 0.0) else 0.5

    @property
    def domain_volume(self) -> float:
        """The volume of the domain proper, excluding the padding."""
        inner, outer = self._bounds
        return 4.0 * np.pi * (outer**3 - inner**3) / 3.0

    def _class_for_order(self, order: float, /) -> type:
        return Lebesgue if order == 0.0 else Sobolev

    def random_point(self, *, rng: Generator | None = None) -> float:
        """A radius of the domain, drawn uniformly over its volume."""
        rng = np.random.default_rng() if rng is None else rng
        inner, outer = self._bounds
        return float(np.cbrt(rng.uniform(inner**3, outer**3)))

    def __repr__(self) -> str:
        inner, outer = self._bounds
        return (
            f"{type(self).__name__}(modes={self.dim}, radii=[{inner:g}, "
            f"{outer:g}], order={self._order:g})"
        )


class Lebesgue(Radial):
    """The ``L2`` space of radial profiles, under the volume measure."""

    def __init__(
        self,
        modes: int,
        /,
        *,
        radius: float = 1.0,
        inner_radius: float = 0.0,
        length_scale: LengthScale = 1.0,
        padding: float | tuple[float, float] | None = None,
        boundary: Boundary = "robin",
        ngll: int = 5,
        element_length: float | None = None,
    ) -> None:
        """
        Args:
            modes: eigenfunctions kept, the dimension of the space.
            radius, inner_radius: the radii; an inner radius of zero is a
                whole ball.
            length_scale: ``L`` in ``A``, which fixes the basis and the
                default padding though not this space's inner product.
            padding: the mesh's reach past each end.
            boundary: the condition at the ends of the mesh; see
                :class:`Radial`.
            ngll: nodes per element.
            element_length: the longest an element may be.
        """
        super().__init__(
            modes,
            radius=radius,
            inner_radius=inner_radius,
            order=0.0,
            length_scale=length_scale,
            padding=padding,
            boundary=boundary,
            ngll=ngll,
            element_length=element_length,
        )


class Sobolev(Radial):
    """The Sobolev space ``H^order`` of radial profiles, with the inner
    product ``(A^order u, v)``."""

    def __init__(
        self,
        modes: int,
        order: float,
        length_scale: LengthScale,
        /,
        *,
        radius: float = 1.0,
        inner_radius: float = 0.0,
        padding: float | tuple[float, float] | None = None,
        boundary: Boundary = "robin",
        ngll: int = 5,
        element_length: float | None = None,
    ) -> None:
        """
        Args:
            modes: eigenfunctions kept, the dimension of the space.
            order: the Sobolev order.
            length_scale: ``L`` in ``A``, a number or a function of radius:
                the length at which the Sobolev weight turns over.
            radius, inner_radius: the radii; an inner radius of zero is a
                whole ball.
            padding: the mesh's reach past each end.
            boundary: the condition at the ends of the mesh; see
                :class:`Radial`.
            ngll: nodes per element.
            element_length: the longest an element may be.
        """
        super().__init__(
            modes,
            radius=radius,
            inner_radius=inner_radius,
            order=order,
            length_scale=length_scale,
            padding=padding,
            boundary=boundary,
            ngll=ngll,
            element_length=element_length,
        )
