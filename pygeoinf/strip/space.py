"""The strip space and its Lebesgue and Sobolev forms."""

from __future__ import annotations

from typing import Sequence

from ..spectral_element._flat import FlatSpace, LateralDomain
from ..spectral_element.base import Boundary, LengthScale


class Strip(FlatSpace):
    """Fields of ``(depth, x)`` on a flat layer over one lateral axis: see
    :class:`~pygeoinf.spectral_element._flat.FlatSpace` for the arguments."""

    _lateral_axes: int = 1
    _geometry: str = "strip"

    def _class_for_order(self, order: float, /) -> type:
        return Lebesgue if order == 0.0 else Sobolev


class Lebesgue(Strip):
    """The ``L2`` space on a strip."""

    def __init__(
        self,
        shape: int | Sequence[int],
        /,
        *,
        lateral: LateralDomain | None = None,
        top: float = 0.0,
        bottom: float = 1.0,
        length_scale: LengthScale = 1.0,
        depth_modes: int | None = None,
        max_eigenvalue: float | None = None,
        padding: float | tuple[float, float] | None = None,
        boundary: Boundary = "robin",
        ngll: int = 5,
        element_length: float | None = None,
    ) -> None:
        """
        Args:
            shape: the points of the lateral grid.
            lateral: the lateral box: its period from the origin, or a
                ``planetmodel.fourier.PaddedDomain`` for a bounded domain
                inside a wider period. A unit period if omitted.
            top, bottom: the depths of the top and the base of the layer,
                positive downward.
            length_scale: ``L`` in ``A``, which fixes the basis and the
                default padding though not this space's inner product.
            depth_modes, max_eigenvalue: the truncation in depth at each
                lateral wavenumber; isotropic when neither is given.
            padding: the reach of the mesh in depth past each end.
            boundary: the condition at the ends of the mesh in depth; see
                :class:`~pygeoinf.spectral_element._flat.FlatSpace`.
            ngll: nodes per element in depth.
            element_length: the longest an element in depth may be.
        """
        super().__init__(
            shape,
            lateral=lateral,
            top=top,
            bottom=bottom,
            order=0.0,
            length_scale=length_scale,
            depth_modes=depth_modes,
            max_eigenvalue=max_eigenvalue,
            padding=padding,
            boundary=boundary,
            ngll=ngll,
            element_length=element_length,
        )


class Sobolev(Strip):
    """The Sobolev space ``H^order`` on a strip, with the inner product
    ``(A^order u, v)``."""

    def __init__(
        self,
        shape: int | Sequence[int],
        order: float,
        length_scale: LengthScale,
        /,
        *,
        lateral: LateralDomain | None = None,
        top: float = 0.0,
        bottom: float = 1.0,
        depth_modes: int | None = None,
        max_eigenvalue: float | None = None,
        padding: float | tuple[float, float] | None = None,
        boundary: Boundary = "robin",
        ngll: int = 5,
        element_length: float | None = None,
    ) -> None:
        """
        Args:
            shape: the points of the lateral grid.
            order: the Sobolev order.
            length_scale: ``L`` in ``A``, a number or a function of depth:
                the length at which the Sobolev weight turns over.
            lateral: the lateral box: its period from the origin, or a
                ``planetmodel.fourier.PaddedDomain`` for a bounded domain
                inside a wider period. A unit period if omitted.
            top, bottom: the depths of the top and the base of the layer,
                positive downward.
            depth_modes, max_eigenvalue: the truncation in depth at each
                lateral wavenumber; isotropic when neither is given.
            padding: the reach of the mesh in depth past each end.
            boundary: the condition at the ends of the mesh in depth; see
                :class:`~pygeoinf.spectral_element._flat.FlatSpace`.
            ngll: nodes per element in depth.
            element_length: the longest an element in depth may be.
        """
        super().__init__(
            shape,
            lateral=lateral,
            top=top,
            bottom=bottom,
            order=order,
            length_scale=length_scale,
            depth_modes=depth_modes,
            max_eigenvalue=max_eigenvalue,
            padding=padding,
            boundary=boundary,
            ngll=ngll,
            element_length=element_length,
        )
