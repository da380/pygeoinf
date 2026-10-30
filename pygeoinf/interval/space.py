"""The interval space and its Lebesgue and Sobolev forms."""

from __future__ import annotations


from numpy.random import Generator

from ..algebra.spaces import resolve_rng
from ..spectral_element.axis import AxisSpace
from ..spectral_element.base import Boundary, LengthScale


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
                default, ends the mesh as if it went on, ``u' = -+ 0.7 u / L``,
                and with it two length scales of padding do what four do under
                the natural condition. ``None`` is the natural condition, and
                doubles the default padding. A number or a pair gives the Robin coefficients
                themselves.
            ngll: Gauss-Lobatto-Legendre nodes per element.
            element_length: the longest an element may be. By default short
                enough that the mesh has two and a half nodes for every mode
                kept and an element is no longer than half the smallest
                length scale.

        Raises:
            ValueError: if the endpoints are out of order, fewer than one mode
                is asked for, a length scale is not positive, or the mesh
                cannot hold the modes asked for.
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
        rng = resolve_rng(rng)
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
