"""The layered slab: flat layers stacked in depth, a slab in each."""

from __future__ import annotations

from typing import Any, Sequence

from ..spectral_element._flat import FlatLayered, LateralDomain

from .space import Slab


class Layered(FlatLayered):
    """Piecewise-continuous fields of ``(depth, x, y)``: a :class:`Slab` in
    each of the flat layers between consecutive depths, all over one lateral
    box and one lateral grid.

    A direct sum of the layers, one summand per layer on its own padded mesh
    in depth with its own length scale and truncation, that knows its
    summands are adjacent in space: which layer a point is in, the field's
    two values at an interface and the jump between them, integrals over the
    whole, one function sampled into every layer, and priors given layer by
    layer, under which the layers are independent. It compares equal to the
    plain direct sum of its layers, which is what a block operator or a
    product measure lands on. Layers built by hand go in through
    :meth:`from_layers`, and must share the lateral grid and box.

    **Above is shallower.** Depth is positive downward, and the sides of an
    interface keep their physical sense: at a point on one, ``side="above"``
    is the layer of smaller depth, listed first, and ``side="below"`` the
    layer of greater depth. The jump across an interface, a horizontal plane,
    is the value above less the value below -- the shallower layer's less the
    deeper's, the sign a ball gives the same interface, outer less inner --
    and is taken at given lateral positions ``(x, y)``.
    """

    _layer_kind = Slab

    def __init__(
        self,
        boundaries: Sequence[float],
        shape: Sequence[int],
        /,
        *,
        lateral: LateralDomain | None = None,
        labels: Sequence[str] | None = None,
        **options: Any,
    ) -> None:
        """
        Args:
            boundaries: the depths bounding the layers, ascending: the top
                of the first layer, the interfaces in order of depth, and the
                base of the last. Depth is positive downward.
            shape: the points of the lateral grid along ``x`` and ``y``, the
                same for every layer: a pair, never read layer by layer.
            lateral: the lateral box, the same for every layer: its two
                periods from the origin, or a
                ``planetmodel.fourier.PaddedDomain`` for a bounded domain
                inside a wider period. A unit period on both axes if omitted.
            labels: names for the layers, so one can be reached by name.
            **options: the keyword arguments of :class:`Slab` -- ``order``,
                ``length_scale``, ``depth_modes``, ``max_eigenvalue``,
                ``padding``, ``boundary``, ``ngll``, ``element_length`` --
                each given once for all layers or, as a list, layer by layer.
                A tuple, such as a padding pair, is one value.

        Raises:
            ValueError: for fewer than two boundaries, a list of the wrong
                length, or what :class:`Slab` refuses.
        """
        super().__init__(boundaries, shape, lateral=lateral, labels=labels, **options)
