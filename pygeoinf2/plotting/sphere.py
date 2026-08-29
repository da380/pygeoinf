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
    """Axes carrying a map projection.

    Args:
        space: the sphere whose fields will be drawn.
        rows: number of panel rows.
        columns: number of panel columns.
        projection: a cartopy projection. Defaults to ``PlateCarree``, which is
            v1's default: it is the projection in which the grid is stored, so
            it is the one that shows the data rather than an opinion about it.
            A downstream wrapper is free to prefer another -- pyslfp defaults
            to Robinson -- and that is its business.
        **kwargs: passed through to ``matplotlib.pyplot.subplots``.

    Returns:
        The ``(figure, axes)`` pair ``plt.subplots`` returns.
    """
    import matplotlib.pyplot as pyplot

    crs = _require_cartopy()
    chosen = crs.PlateCarree() if projection is None else projection
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
    cmap: str = "RdBu",
    symmetric: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool | None = None,
    colorbar_label: str | None = None,
    coasts: bool = False,
    borders: bool = False,
    rivers: bool = False,
    gridlines: bool = True,
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
        ax: axes to draw on. A new figure is made if omitted, on a
            ``PlateCarree`` projection.
        cmap: colour map. ``RdBu`` by default, as in v1: most fields drawn
            here are signed anomalies, and a diverging map is what those want.
        symmetric: put zero at the middle of the colour scale. Use it for
            anything signed. Off by default.
        vmin: lower colour limit; the field's minimum if omitted.
        vmax: upper colour limit; the field's maximum if omitted.
        colorbar: attach a colourbar. Off by default, as in v1 -- a bar takes
            room from the map, and a panel in a grid usually shares one --
            unless a *colorbar_label* is given, since asking for a label is
            asking for the bar it goes on. Pass ``False`` to override that. The
            bar is left on the returned mappable as ``.colorbar``, so it can be
            restyled afterwards.
        colorbar_label: label for the colourbar, which turns one on.
        coasts: draw coastlines. Off by default.
        borders: draw national borders. Off by default.
        rivers: draw rivers. Off by default.
        gridlines: draw a latitude and longitude graticule, left on the axes as
            ``.gridliner``. On by default, as in v1: a map without a graticule
            leaves the reader to guess where anything is.
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
    longitudes, values = _rolled_to_the_dateline(np.degrees(space.longitudes), values)

    low, high = colour_limits(values, vmin=vmin, vmax=vmax, symmetric=symmetric)
    common = dict(transform=crs.PlateCarree(), cmap=cmap, vmin=low, vmax=high)

    if contour or contour_lines:
        # Contours are drawn through points rather than over cells, so the seam
        # is closed there by repeating the first column at +180.
        closed_longitudes = np.append(longitudes, longitudes[0] + 360.0)
        closed_values = np.concatenate([values, values[:, :1]], axis=1)

    if contour:
        mappable = ax.contourf(
            closed_longitudes,
            latitudes,
            closed_values,
            levels=levels,
            **common,
            **kwargs,
        )
    else:
        edge_longitudes, edge_values = _cell_edges_across_the_dateline(
            longitudes, values
        )
        mappable = ax.pcolormesh(
            edge_longitudes,
            np.clip(_cell_edges(latitudes), -90.0, 90.0),
            edge_values,
            shading="flat",
            **common,
            **kwargs,
        )
    if contour_lines:
        # Left on the axes rather than returned, since the mappable a caller
        # wants for a colourbar is the filled one.
        ax.contour_set = ax.contour(
            closed_longitudes,
            latitudes,
            closed_values,
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
    if colorbar or (colorbar is None and colorbar_label is not None):
        options = dict(shrink=0.7, pad=0.03)
        options.update(colorbar_kwargs or {})
        bar = ax.figure.colorbar(mappable, ax=ax, **options)
        if colorbar_label is not None:
            bar.set_label(colorbar_label)
    if title is not None:
        ax.set_title(title)
    return ax, mappable


def _rolled_to_the_dateline(
    longitudes: np.ndarray, values: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """The same grid, with its columns rolled from ``[0, 360)`` to ``[-180, 180)``.

    The grid a sphere hands over starts at Greenwich; cartopy's projections are
    cut at the antimeridian. Handing them a mesh in ``[0, 360)`` puts the cut
    through the middle of the data, and every cell that straddles it sends
    cartopy down its per-polygon wrapping path -- 1028 cells at lmax 128, and
    with them 1.2 s of the 1.3 s a map used to cost. Rolling costs a copy.

    Args:
        longitudes: the grid longitudes in degrees, increasing over ``[0, 360)``.
        values: the grid values, longitude along the second axis.

    Returns:
        The rolled ``(longitudes, values)`` pair, longitudes increasing over
        ``[-180, 180)``.
    """
    crossing = int(np.searchsorted(longitudes, 180.0))
    rolled = np.concatenate([longitudes[crossing:] - 360.0, longitudes[:crossing]])
    return rolled, np.roll(values, -crossing, axis=1)


def _cell_edges(centres: np.ndarray, /) -> np.ndarray:
    """The edges of the cells centred on given points.

    What ``shading="auto"`` computes internally when it is handed as many
    values as coordinates, made explicit so that the longitude edges can be
    placed by hand at the antimeridian.

    Args:
        centres: the cell centres, monotonic.

    Returns:
        One more edge than there were centres.
    """
    centres = np.asarray(centres, dtype=float)
    middle = 0.5 * (centres[:-1] + centres[1:])
    return np.concatenate(
        [[2.0 * centres[0] - middle[0]], middle, [2.0 * centres[-1] - middle[-1]]]
    )


def _cell_edges_across_the_dateline(
    longitudes: np.ndarray, values: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Longitude cell edges that close the seam without straddling it.

    The first column's cell is centred on the antimeridian, so it lies half on
    each side of the map. Drawn as one cell it straddles the cut and costs the
    whole mesh its fast path; dropped, it leaves the blank wedge down the
    dateline that the wrap was there to close. So it is drawn as its two
    halves, one at each edge of the map, which is the same picture and stays on
    the fast path.

    Args:
        longitudes: the rolled grid longitudes, increasing over ``[-180, 180)``.
        values: the rolled grid values.

    Returns:
        The ``(edges, values)`` pair to hand ``pcolormesh`` with
        ``shading="flat"``: one more edge than there are columns of values, the
        first column repeated at the far edge.
    """
    inner = 0.5 * (longitudes[:-1] + longitudes[1:])
    seam = 0.5 * (longitudes[-1] + longitudes[0]) + 180.0
    edges = np.clip(
        np.concatenate([[-180.0], inner, [seam, 180.0]]), -180.0, 180.0
    )
    return edges, np.concatenate([values, values[:, :1]], axis=1)


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
    data: Any = None,
    ax: Any = None,
    marker: str = "^",
    size: float = 20.0,
    color: str = "black",
    cmap: str = "RdBu",
    symmetric: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool | None = None,
    colorbar_label: str | None = None,
    colorbar_kwargs: dict | None = None,
    **kwargs: Any,
) -> Any:
    """Scatter a set of points on a map, optionally coloured by a value.

    A scatter of stations is one thing; a scatter of *measurements* is the
    other, and it is the one an altimetry or gravity figure is made of. Passing
    the values as *data* colours the markers by them and gives them a bar to be
    read against, which is v1's ``plot_points(points, data=...)``. Without it
    every marker is the one flat *color*.

    Args:
        space: the sphere.
        points: ``(latitude, longitude)`` pairs in degrees.
        data: one value per point, to colour the markers by. Without it they
            are all *color*.
        ax: axes to draw on. A new map is made if omitted.
        marker: matplotlib marker.
        size: marker area.
        color: marker colour, used when there is no *data*.
        cmap: colour map for *data*. ``RdBu`` by default, as in v1.
        symmetric: put zero at the middle of the colour scale. Use it for
            anything signed. Off by default.
        vmin: lower colour limit; the data's minimum if omitted.
        vmax: upper colour limit; the data's maximum if omitted.
        colorbar: attach a colourbar, which needs *data* to mean anything. Off
            by default unless a *colorbar_label* is given; pass ``False`` to
            override that. Left on the returned collection as ``.colorbar``.
        colorbar_label: label for the colourbar, which turns one on.
        colorbar_kwargs: passed to ``figure.colorbar``, over the defaults.
        **kwargs: passed to ``scatter``.

    Returns:
        The ``(axes, collection)`` pair.

    Raises:
        ValueError: if *data* is given with a value per point missing or
            spare. Silently colouring the first few would be worse.
    """
    crs = _require_cartopy()
    if ax is None:
        _, ax = subplots(space)
    positions = np.atleast_2d(np.asarray(list(points), dtype=float))

    if data is None:
        colours: Any = color
    else:
        colours = np.asarray(data, dtype=float).ravel()
        if colours.size != positions.shape[0]:
            raise ValueError(
                f"There are {positions.shape[0]} points and {colours.size} "
                "values to colour them by."
            )
        low, high = colour_limits(colours, vmin=vmin, vmax=vmax, symmetric=symmetric)
        kwargs.setdefault("cmap", cmap)
        kwargs.setdefault("vmin", low)
        kwargs.setdefault("vmax", high)

    collection = ax.scatter(
        positions[:, 1],
        positions[:, 0],
        transform=crs.PlateCarree(),
        marker=marker,
        s=size,
        c=colours,
        **kwargs,
    )
    wanted = colorbar or (colorbar is None and colorbar_label is not None)
    if wanted and data is not None:
        options = dict(shrink=0.7, pad=0.03)
        options.update(colorbar_kwargs or {})
        bar = ax.figure.colorbar(collection, ax=ax, **options)
        if colorbar_label is not None:
            bar.set_label(colorbar_label)
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
