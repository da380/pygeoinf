"""
Maps of fields on a sphere, through cartopy.

Needs ``cartopy``, which comes with the ``sphere`` extra. The import is
deferred to the call, so registering the renderer costs nothing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..symmetric_space.sphere import Sphere
from .base import colour_limits, plot, subplots

__all__: list[str] = []


def _require_cartopy() -> Any:
    """Import cartopy, with a message that says what to install."""
    try:
        import cartopy.crs as projections
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "Plotting a field on a sphere needs cartopy, which is an optional "
            "dependency. Install it with the 'sphere' extra."
        ) from error
    return projections


@subplots.register
def _(
    space: Sphere,
    /,
    *,
    rows: int = 1,
    columns: int = 1,
    projection: Any = None,
    **kwargs: Any,
) -> Any:
    """Axes carrying a map projection, defaulting to Robinson."""
    import matplotlib.pyplot as pyplot

    crs = _require_cartopy()
    chosen = crs.Robinson() if projection is None else projection
    kwargs.setdefault("figsize", (6.0 * columns, 3.2 * rows))
    kwargs.setdefault("layout", "constrained")
    return pyplot.subplots(rows, columns, subplot_kw={"projection": chosen}, **kwargs)


@plot.register
def _(
    space: Sphere,
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
    coasts: bool = False,
    gridlines: bool = False,
    **kwargs: Any,
) -> Any:
    """Draw a field on a sphere as a map.

    Args:
        space: the sphere.
        field: a grid array of the space.
        ax: axes to draw on. A new figure is made if omitted.
        cmap: colour map.
        symmetric: put zero at the middle of the colour scale. Use it for
            anything signed.
        vmin: lower colour limit; the field's minimum if omitted.
        vmax: upper colour limit; the field's maximum if omitted.
        colorbar: attach a colourbar. It is left on the returned mappable as
            ``.colorbar``, so it can be restyled afterwards.
        colorbar_label: label for the colourbar.
        coasts: draw coastlines.
        gridlines: draw a latitude and longitude graticule, left on the axes as
            ``.gridliner``.
        **kwargs: passed through to ``pcolormesh``.

    Returns:
        The ``(axes, mappable)`` pair.
    """
    crs = _require_cartopy()
    if ax is None:
        _, ax = subplots(space)

    values = np.asarray(field, dtype=float)
    if values.shape != space.grid_shape:
        raise ValueError(
            f"A field on this sphere has shape {space.grid_shape}, got "
            f"{values.shape}."
        )

    latitudes = 90.0 - np.degrees(space.colatitudes)
    longitudes = np.degrees(space.longitudes)
    # Close the seam: the grid stops one step short of 360 degrees, and without
    # the wrap a blank wedge appears down the dateline.
    longitudes = np.append(longitudes, 360.0)
    values = np.concatenate([values, values[:, :1]], axis=1)

    low, high = colour_limits(values, vmin=vmin, vmax=vmax, symmetric=symmetric)
    mappable = ax.pcolormesh(
        longitudes,
        latitudes,
        values,
        transform=crs.PlateCarree(),
        cmap=cmap,
        vmin=low,
        vmax=high,
        shading="auto",
        **kwargs,
    )
    ax.set_global()
    if coasts:
        ax.coastlines(linewidth=0.5)
    if gridlines:
        ax.gridliner = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    if colorbar:
        bar = ax.figure.colorbar(mappable, ax=ax, shrink=0.7, pad=0.03)
        if colorbar_label is not None:
            bar.set_label(colorbar_label)
    return ax, mappable
