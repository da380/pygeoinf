"""The layered interval: several interval spaces end to end, one field per layer."""

from __future__ import annotations

from typing import Any, Sequence


from ..spectral_element import layered as _layered

from .space import Interval


class Layered(_layered.Layered):
    """Piecewise-continuous fields on the line: an :class:`Interval` between
    each pair of consecutive breakpoints.

    A direct sum of the layers, one summand per layer on its own padded mesh
    with its own length scale and truncation, that knows its summands are
    adjacent in space: which layer a point is in, the field's two values at
    an interface and the jump between them, integrals over the whole, one
    function sampled into every layer, and priors given layer by layer, under
    which the layers are independent. It compares equal to the plain direct
    sum of its layers, which is what a block operator or a product measure
    lands on. Layers built by hand go in through :meth:`from_layers`.
    """

    def __init__(
        self,
        breakpoints: Sequence[float],
        modes: int | list[int],
        /,
        *,
        labels: Sequence[str] | None = None,
        **options: Any,
    ) -> None:
        """
        Args:
            breakpoints: the ends of the layers, ascending.
            modes: eigenfunctions kept in each layer, for all or as a list.
            labels: names for the layers, so one can be reached by name.
            **options: the keyword arguments of :class:`Interval` --
                ``order``, ``length_scale``, ``padding``, ``boundary``, ``ngll``,
                ``element_length`` -- each given once for all layers or, as a
                list, layer by layer. A tuple, such as a padding pair, is
                one value.

        Raises:
            ValueError: for fewer than two breakpoints, or a list of the
                wrong length.
        """
        super().__init__(
            _layered.layers_between(
                Interval, breakpoints, modes, ("lower", "upper"), options
            ),
            labels=labels,
        )
