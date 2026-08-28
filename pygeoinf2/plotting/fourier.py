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


def plot_error_bounds(
    space: PeriodicBox,
    lower: Any,
    upper: Any,
    /,
    *,
    ax: Any = None,
    centre: Any = None,
    colour: str = "C0",
    alpha: float = 0.25,
    label: str | None = None,
    **kwargs: Any,
) -> Any:
    """Shade the band between two fields on a one-dimensional box.

    v1's ``plot_error_bounds``, and the natural way to draw what an inference
    actually produces: a bound above and below, not a single curve. A pair of
    lines says the same thing and reads as two estimates rather than as one
    with an uncertainty, which is the wrong impression to leave.

    The two bounds are *not* checked against each other. A band that crosses
    over is a real thing to want to see -- it is what an inconsistent bound
    looks like, and refusing to draw it would hide exactly the case worth
    looking at.

    Args:
        space: a one-dimensional box.
        lower: the lower bound, as a field.
        upper: the upper bound.
        ax: axes to draw on. A new figure is made if omitted.
        centre: an optional field to draw as a line through the band, usually
            the estimate the bounds belong to.
        colour: for the band and the centre line.
        alpha: the band's transparency.
        label: a legend entry for the band.
        **kwargs: passed to ``fill_between``.

    Returns:
        The ``(axes, band)`` pair, with the centre line left on the axes as
        ``.centre_line`` when one was drawn.

    Raises:
        ValueError: on a box of more than one dimension, or if a field has the
            wrong shape.
    """
    if space.spatial_dimension != 1:
        raise ValueError(
            f"Error bounds are drawn on a one-dimensional box; this one has "
            f"{space.spatial_dimension} dimensions. Take a slice first."
        )
    if ax is None:
        _, ax = subplots(space)

    axis = space.grid_axes[0]
    fields = []
    for name, given in (("lower", lower), ("upper", upper)):
        values = np.asarray(given, dtype=float)
        if values.shape != space.shape:
            raise ValueError(
                f"The {name} bound has shape {values.shape}, but a field on "
                f"this box has shape {space.shape}."
            )
        fields.append(values)

    band = ax.fill_between(
        axis, fields[0], fields[1], color=colour, alpha=alpha, label=label, **kwargs
    )
    if centre is not None:
        middle = np.asarray(centre, dtype=float)
        if middle.shape != space.shape:
            raise ValueError(
                f"The centre has shape {middle.shape}, but a field on this box "
                f"has shape {space.shape}."
            )
        (ax.centre_line,) = ax.plot(axis, middle, color=colour, lw=1.5)
    ax.set_xlim(axis[0], axis[-1])
    return ax, band
