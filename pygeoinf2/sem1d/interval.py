"""A bounded interval of the line, in the eigenbasis of ``A = 1 - d/dx (L^2
d/dx)``.

**The measure is ``dx`` and the coordinate is a position on a line**, not a
radius: zero is a point like any other, the domain may straddle it, and the
mesh is padded at both ends. A function of radius in a ball or a shell, under
``r^2 dr``, is :mod:`~pygeoinf2.sem1d.radial`.

The counterpart of :mod:`~pygeoinf2.symmetric_space.line` for a length scale
``L(x)`` that varies, and for a mesh that is not uniform. The line embeds its
interval in a circle and diagonalizes the Laplacian there by an FFT; this one
embeds it in a longer interval, meshes that with Gauss-Lobatto-Legendre
spectral elements, and solves for the eigenfunctions of ``A`` on the mesh. The
padding plays the same part in both: ``A`` needs a boundary condition at each
end of its mesh, and whichever is chosen distorts a field within about a
correlation length of where it acts. Moving the ends away from ``[lower,
upper]`` moves the distortion out of the domain, and the Robin condition the
mesh ends with by default makes it smaller to begin with, so that half the
padding the natural condition wants is enough (DECISIONS.md D-122).

In the eigenbasis every power of ``A`` is diagonal, so the conveniences of a
symmetric space carry over with ``theta``, the eigenvalues of ``A``, in the
place of ``1 + L^2 k^2``: the ``H^s`` inner product is ``sum theta^s c^2``, a
Sobolev prior has covariance ``theta^-p``, and a draw is one synthesis. What
does *not* carry over is homogeneity. The pointwise variance of such a prior
depends on where it is asked for, which is why ``sobolev_measure`` calibrates
by a standard deviation *field* rather than by one number.

**The conventions are the package's.** A vector is the array of the field's
values at the nodes of the padded mesh, as a ``Line``'s is its values on the
padded grid; its components are the coefficients of the kept eigenfunctions;
a product is left on the nodes, untruncated (DECISIONS.md D-87); point
evaluation is refused below the order that admits it (D-11).

The numerics are ``planetmodel.randomfield``'s: ``padded_mesh``,
``RadialOperatorFamily`` under its plain measure, and the ``SpectralBasis`` of
degree zero, which is reachable as ``basis``.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from .axis import AxisSpace
from .base import Boundary, LengthScale

__all__ = ["Interval", "Lebesgue", "Sobolev"]


class Interval(AxisSpace):
    """Fields on ``[lower, upper]`` under ``dx``, in the leading
    eigenfunctions of ``A``."""

    _weight = "one"
    _scale = 1.0
    _geometry = "interval"

    def __init__(
        self,
        modes: int,
        /,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
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
            lower: the left endpoint.
            upper: the right endpoint.
            order: the Sobolev order. Zero gives the Lebesgue space.
            length_scale: ``L`` in ``A = 1 - d/dx (L^2 d/dx)``, a number or a
                function of position. It is asked about ``[lower, upper]``
                only; the padding takes its value at the nearer endpoint.
            padding: how far the mesh extends past each endpoint, one length
                or a ``(below, above)`` pair. Two length scales by default,
                those at the endpoints; four under the natural condition.
            boundary: the condition at the ends of the mesh. ``"robin"``, the
                default, ends the mesh as if it went on, ``u' = -+ 0.7 u / L``:
                the point of the padding is that the condition should not
                matter, and measured, two length scales under this one do
                what four do under the natural one (DECISIONS.md D-122).
                ``None`` is the natural condition, and doubles the default
                padding. A number or a pair gives the Robin coefficients
                themselves.
            ngll: Gauss-Lobatto-Legendre nodes per element.
            element_length: the longest an element may be. By default short
                enough that the mesh has two and a half nodes for every mode
                kept and an element is no longer than half the smallest
                length scale.

        Raises:
            ValueError: if the endpoints are out of order, a length scale is
                not positive, or the mesh cannot hold the modes asked for.
        """
        super().__init__(
            modes,
            lower,
            upper,
            order=order,
            length_scale=length_scale,
            padding=padding,
            boundary=boundary,
            ngll=ngll,
            element_length=element_length,
        )

    @property
    def spatial_dimension(self) -> int:
        """One."""
        return 1

    @property
    def domain_volume(self) -> float:
        """The length of the domain proper, excluding the padding."""
        return self._bounds[1] - self._bounds[0]

    def _class_for_order(self, order: float, /) -> type:
        return Lebesgue if order == 0.0 else Sobolev

    def random_point(self, *, rng: Generator | None = None) -> float:
        """A point of the domain, drawn uniformly."""
        rng = np.random.default_rng() if rng is None else rng
        return float(rng.uniform(*self._bounds))

    def __repr__(self) -> str:
        lower, upper = self._bounds
        return (
            f"{type(self).__name__}(modes={self.dim}, bounds=[{lower:g}, "
            f"{upper:g}], order={self._order:g})"
        )


class Lebesgue(Interval):
    """The ``L2`` space on an interval."""

    def __init__(
        self,
        modes: int,
        /,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
        length_scale: LengthScale = 1.0,
        padding: float | tuple[float, float] | None = None,
        boundary: Boundary = "robin",
        ngll: int = 5,
        element_length: float | None = None,
    ) -> None:
        """
        Args:
            modes: eigenfunctions kept, the dimension of the space.
            lower, upper: the endpoints.
            length_scale: ``L`` in ``A``, which fixes the basis and the
                default padding though not this space's inner product.
            padding: the mesh's reach past each endpoint.
            boundary: the condition at the ends of the mesh; see
                :class:`Interval`.
            ngll: nodes per element.
            element_length: the longest an element may be.
        """
        super().__init__(
            modes,
            lower=lower,
            upper=upper,
            order=0.0,
            length_scale=length_scale,
            padding=padding,
            boundary=boundary,
            ngll=ngll,
            element_length=element_length,
        )


class Sobolev(Interval):
    """The Sobolev space ``H^order`` on an interval, with the inner product
    ``(A^order u, v)``."""

    def __init__(
        self,
        modes: int,
        order: float,
        length_scale: LengthScale,
        /,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
        padding: float | tuple[float, float] | None = None,
        boundary: Boundary = "robin",
        ngll: int = 5,
        element_length: float | None = None,
    ) -> None:
        """
        Args:
            modes: eigenfunctions kept, the dimension of the space.
            order: the Sobolev order.
            length_scale: ``L`` in ``A``, a number or a function of position:
                the length at which the Sobolev weight turns over.
            lower, upper: the endpoints.
            padding: the mesh's reach past each endpoint.
            boundary: the condition at the ends of the mesh; see
                :class:`Interval`.
            ngll: nodes per element.
            element_length: the longest an element may be.
        """
        super().__init__(
            modes,
            lower=lower,
            upper=upper,
            order=order,
            length_scale=length_scale,
            padding=padding,
            boundary=boundary,
            ngll=ngll,
            element_length=element_length,
        )
