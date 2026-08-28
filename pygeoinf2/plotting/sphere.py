"""
Maps of fields on a sphere, through cartopy.

Needs ``cartopy``, which comes with the ``sphere`` extra. The import is
deferred to the call, so registering the renderer costs nothing.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..symmetric_space.sphere import Sphere
from .base import colour_limits, plot, subplots

__all__ = ["plot_points", "plot_paths"]


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
    field: Any,
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
    borders: bool = False,
    rivers: bool = False,
    gridlines: bool = False,
    gridlines_kwargs: dict | None = None,
    colorbar_kwargs: dict | None = None,
    map_extent: Sequence[float] | None = None,
    contour: bool = False,
    contour_lines: bool = False,
    levels: Any = None,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Draw a field on a sphere as a map.

    Args:
        space: the sphere.
        field: a field of the space, as an ``SHGrid`` or a bare array of its
            grid values.
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
        borders: draw national borders.
        rivers: draw rivers.
        gridlines: draw a latitude and longitude graticule, left on the axes as
            ``.gridliner``.
        gridlines_kwargs: passed to ``gridlines``. ``lat_interval`` and
            ``lon_interval`` are translated into the locators cartopy wants,
            since those are what a caller actually has in mind.
        colorbar_kwargs: passed to ``figure.colorbar``, over the defaults.
        map_extent: ``(west, east, south, north)`` in degrees. Without it the
            map is global; with it ``set_global`` is skipped, which is the
            whole point -- calling both leaves the extent overridden.
        contour: draw filled contours rather than a pcolormesh.
        contour_lines: overlay contour lines. Can be combined with either.
        levels: the contour levels, or a count. Passed to ``contourf`` and
            ``contour``.
        title: a title for the axes.
        **kwargs: passed through to whichever of ``pcolormesh``, ``contourf``
            and ``contour`` is drawing.

    Returns:
        The ``(axes, mappable)`` pair.
    """
    crs = _require_cartopy()
    if ax is None:
        _, ax = subplots(space)

    values = space.grid_values(field)
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
    common = dict(transform=crs.PlateCarree(), cmap=cmap, vmin=low, vmax=high)

    if contour:
        mappable = ax.contourf(
            longitudes, latitudes, values, levels=levels, **common, **kwargs
        )
    else:
        mappable = ax.pcolormesh(
            longitudes, latitudes, values, shading="auto", **common, **kwargs
        )
    if contour_lines:
        # Left on the axes rather than returned, since the mappable a caller
        # wants for a colourbar is the filled one.
        ax.contour_set = ax.contour(
            longitudes,
            latitudes,
            values,
            levels=levels,
            transform=crs.PlateCarree(),
            colors="black",
            linewidths=0.5,
        )

    if map_extent is None:
        ax.set_global()
    else:
        # Not both: set_global would undo the extent that was asked for.
        ax.set_extent(list(map_extent), crs=crs.PlateCarree())

    if coasts:
        ax.coastlines(linewidth=0.5)
    if borders or rivers:
        import cartopy.feature as feature

        if borders:
            ax.add_feature(feature.BORDERS, linewidth=0.3)
        if rivers:
            ax.add_feature(feature.RIVERS, linewidth=0.3)
    if gridlines:
        ax.gridliner = ax.gridlines(**_gridline_options(gridlines_kwargs))
    if colorbar:
        options = dict(shrink=0.7, pad=0.03)
        options.update(colorbar_kwargs or {})
        bar = ax.figure.colorbar(mappable, ax=ax, **options)
        if colorbar_label is not None:
            bar.set_label(colorbar_label)
    if title is not None:
        ax.set_title(title)
    return ax, mappable


def _gridline_options(given: dict | None, /) -> dict:
    """Graticule options, with the intervals a caller thinks in.

    cartopy wants ``xlocs``/``ylocs`` as tick arrays; what anyone actually has
    in mind is "a line every 30 degrees". ``lat_interval`` and ``lon_interval``
    say that, and are translated here.
    """
    options = dict(draw_labels=True, linewidth=0.3, alpha=0.5)
    options.update(given or {})
    latitude = options.pop("lat_interval", None)
    longitude = options.pop("lon_interval", None)
    if latitude is not None:
        options["ylocs"] = np.arange(-90.0, 90.0 + latitude, latitude)
    if longitude is not None:
        options["xlocs"] = np.arange(-180.0, 180.0 + longitude, longitude)
    return options


def plot_points(
    space: Sphere,
    points: Any,
    /,
    *,
    ax: Any = None,
    marker: str = "^",
    size: float = 20.0,
    color: str = "black",
    **kwargs: Any,
) -> Any:
    """Scatter a set of points on a map.

    Args:
        space: the sphere.
        points: ``(latitude, longitude)`` pairs in degrees.
        ax: axes to draw on. A new map is made if omitted.
        marker: matplotlib marker.
        size: marker area.
        color: marker colour.
        **kwargs: passed to ``scatter``.

    Returns:
        The ``(axes, collection)`` pair.
    """
    crs = _require_cartopy()
    if ax is None:
        _, ax = subplots(space)
    positions = np.atleast_2d(np.asarray(list(points), dtype=float))
    collection = ax.scatter(
        positions[:, 1],
        positions[:, 0],
        transform=crs.PlateCarree(),
        marker=marker,
        s=size,
        c=color,
        **kwargs,
    )
    return ax, collection


def plot_paths(
    space: Sphere,
    paths: Any,
    /,
    *,
    ax: Any = None,
    count: int = 24,
    color: str = "black",
    linewidth: float = 0.4,
    alpha: float = 0.15,
    **kwargs: Any,
) -> Any:
    """Draw great-circle paths on a map.

    Each path is sampled along its geodesic rather than handed to cartopy as
    two endpoints, so it follows the great circle rather than a straight line
    in the projection — which for a global network is most of them.

    Args:
        space: the sphere.
        paths: ``(start, end)`` pairs of points.
        ax: axes to draw on.
        count: samples along each path.
        color: line colour.
        linewidth: line width.
        alpha: opacity, low by default because these overlap heavily.
        **kwargs: passed to ``plot``.

    Returns:
        The ``(axes, lines)`` pair.
    """
    crs = _require_cartopy()
    if ax is None:
        _, ax = subplots(space)

    lines = []
    for start, end in paths:
        nodes, _ = space.geodesic_quadrature(start, end, count=count)
        positions = np.atleast_2d(np.asarray(nodes, dtype=float))
        latitudes, longitudes = positions[:, 0], positions[:, 1]
        # A path crossing the dateline would otherwise be drawn straight across
        # the whole map; splitting it at the jump keeps each piece local.
        breaks = np.flatnonzero(np.abs(np.diff(longitudes)) > 180.0) + 1
        for piece in np.split(np.arange(longitudes.size), breaks):
            if piece.size < 2:
                continue
            (line,) = ax.plot(
                longitudes[piece],
                latitudes[piece],
                transform=crs.PlateCarree(),
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                **kwargs,
            )
            lines.append(line)
    return ax, lines
