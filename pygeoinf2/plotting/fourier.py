"""Fields on a periodic box or a bounded box: a line in 1D, an image in 2D."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..symmetric_space.fourier import PeriodicBox
from .base import colour_limits, plot, subplots

__all__: list[str] = []


@subplots.register
def _(space: PeriodicBox, /, *, rows: int = 1, columns: int = 1, **kwargs: Any) -> Any:
    """Ordinary axes: a box needs no projection."""
    import matplotlib.pyplot as pyplot

    kwargs.setdefault("figsize", (5.0 * columns, 3.2 * rows))
    kwargs.setdefault("layout", "constrained")
    return pyplot.subplots(rows, columns, **kwargs)


@plot.register
def _(
    space: PeriodicBox,
    field: np.ndarray,
    /,
    *,
    ax: Any = None,
    cmap: str = "viridis",
    symmetric: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    **kwargs: Any,
) -> Any:
    """Draw a field on a one- or two-dimensional box.

    Args:
        space: the box.
        field: a grid array of the space.
        ax: axes to draw on. A new figure is made if omitted.
        cmap: colour map, used in two dimensions.
        symmetric: put zero at the middle of the colour scale.
        vmin: lower colour limit.
        vmax: upper colour limit.
        colorbar: attach a colourbar, in two dimensions.
        colorbar_label: label for the colourbar.
        **kwargs: passed to ``plot`` or ``pcolormesh``.

    Returns:
        The ``(axes, mappable)`` pair. In one dimension the mappable is the
        line.
    """
    if space.spatial_dimension > 2:
        raise NotImplementedError(
            f"There is no renderer for a {space.spatial_dimension}-dimensional "
            "box. Take a slice or a projection first."
        )
    if ax is None:
        _, ax = subplots(space)

    values = np.asarray(field, dtype=float)
    if values.shape != space.shape:
        raise ValueError(
            f"A field on this box has shape {space.shape}, got {values.shape}."
        )

    if space.spatial_dimension == 1:
        (line,) = ax.plot(space.grid_axes[0], values, **kwargs)
        ax.set_xlim(space.grid_axes[0][0], space.grid_axes[0][-1])
        return ax, line

    low, high = colour_limits(values, vmin=vmin, vmax=vmax, symmetric=symmetric)
    first, second = space.grid_axes
    mappable = ax.pcolormesh(
        second,
        first,
        values,
        cmap=cmap,
        vmin=low,
        vmax=high,
        shading="auto",
        **kwargs,
    )
    ax.set_aspect("equal")
    if colorbar:
        bar = ax.figure.colorbar(mappable, ax=ax, shrink=0.85, pad=0.03)
        if colorbar_label is not None:
            bar.set_label(colorbar_label)
    return ax, mappable
