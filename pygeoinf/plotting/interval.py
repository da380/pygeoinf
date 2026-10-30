"""Functions of one coordinate: an interval of the line, or a radial profile.

Both are drawn as a line through the field's nodal values, as a box is, over
the domain alone unless ``padding=True`` asks for the padded mesh too. A
layered space is one line per layer in one color, which leaves a break at
every jump; a layered space of balls is passed on to the ball's shell.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..spectral_element import AxisSpace, Layered
from .base import plot, subplots

__all__: list[str] = []


@subplots.register
def _(space: AxisSpace, /, *, rows: int = 1, columns: int = 1, **kwargs: Any) -> Any:
    """Ordinary axes: a function of one coordinate needs no projection."""
    import matplotlib.pyplot as pyplot

    kwargs.setdefault("figsize", (5.0 * columns, 3.2 * rows))
    kwargs.setdefault("layout", "constrained")
    return pyplot.subplots(rows, columns, **kwargs)


@plot.register
def _(
    space: AxisSpace,
    field: np.ndarray,
    /,
    *,
    ax: Any = None,
    padding: bool = False,
    **kwargs: Any,
) -> Any:
    """Draw a field on an interval, or a radial profile, as a line through
    its nodal values.

    Args:
        space: the interval or the space of radial profiles.
        field: a vector of the space.
        ax: axes to draw on. A new figure is made if omitted.
        padding: draw the padded mesh too, shaded, and not the domain alone:
            worth seeing once, since it shows what the padding is for.
        **kwargs: passed to ``Axes.plot``.

    Returns:
        The ``(axes, line)`` pair.

    Raises:
        ValueError: if the field has the wrong shape.
    """
    if ax is None:
        _, ax = subplots(space)
    values = np.asarray(field, dtype=float)
    if values.shape != space.grid_shape:
        raise ValueError(
            f"A field of this space has shape {space.grid_shape}, got "
            f"{values.shape}."
        )
    lower, upper = space.bounds
    if padding:
        (line,) = ax.plot(space.nodes, values, **kwargs)
        for start, stop in ((space.nodes[0], lower), (upper, space.nodes[-1])):
            if stop > start:
                ax.axvspan(start, stop, color="0.5", alpha=0.15, linewidth=0.0)
        ax.set_xlim(space.nodes[0], space.nodes[-1])
    else:
        (line,) = ax.plot(space.interior_nodes, space.interior_values(values), **kwargs)
        ax.set_xlim(lower, upper)
    return ax, line


@subplots.register
def _(space: Layered, /, *, rows: int = 1, columns: int = 1, **kwargs: Any) -> Any:
    """The axes the layers want: ordinary ones here, a map for balls."""
    return subplots(space.layers[-1], rows=rows, columns=columns, **kwargs)


@plot.register
def _(space: Layered, field: Any, /, *, ax: Any = None, **kwargs: Any) -> Any:
    """Draw a piecewise-continuous field.

    Intervals and radial profiles are one line per layer in one color, which
    leaves a break at every jump. Balls are drawn as the ball is, by the shell
    at ``radius=``, the surface if omitted; the ball's ``plot_section`` and
    ``plot_profile`` take a layered space too.

    Args:
        space: the layered space.
        field: a vector of it, a tuple of layer fields.
        ax: axes to draw on. A new figure is made if omitted.
        **kwargs: those of the layers' own ``plot``.

    Returns:
        The ``(axes, mappable)`` pair, the mappable being the last layer's
        line for intervals and profiles.
    """
    if not issubclass(space.geometry, AxisSpace):
        from .ball import plot_shell

        return plot_shell(space, field, ax=ax, **kwargs)
    if ax is None:
        _, ax = subplots(space)
    line = None
    for layer, part in zip(space.layers, field):
        if line is not None:
            kwargs = {**kwargs, "color": line.get_color(), "label": "_nolegend_"}
        ax, line = plot(layer, part, ax=ax, **kwargs)
    ends = space.layer_bounds
    if not kwargs.get("padding", False):
        ax.set_xlim(ends[0][0], ends[-1][1])
    return ax, line
