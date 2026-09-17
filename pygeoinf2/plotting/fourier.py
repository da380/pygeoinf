"""Fields on a periodic box or a bounded box: a line in 1D, an image in 2D."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..symmetric_space.fourier import PeriodicBox
from .base import color_limits, plot, plot_balls, plot_paths, plot_points, subplots

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
        cmap: color map, used in two dimensions.
        symmetric: put zero at the middle of the color scale.
        vmin: lower color limit.
        vmax: upper color limit.
        colorbar: attach a colorbar, in two dimensions.
        colorbar_label: label for the colorbar.
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

    low, high = color_limits(values, vmin=vmin, vmax=vmax, symmetric=symmetric)
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
    center: Any = None,
    color: str = "C0",
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
        center: an optional field to draw as a line through the band, usually
            the estimate the bounds belong to.
        color: for the band and the center line.
        alpha: the band's transparency.
        label: a legend entry for the band.
        **kwargs: passed to ``fill_between``.

    Returns:
        The ``(axes, band)`` pair, with the center line left on the axes as
        ``.center_line`` when one was drawn.

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
        axis, fields[0], fields[1], color=color, alpha=alpha, label=label, **kwargs
    )
    if center is not None:
        middle = np.asarray(center, dtype=float)
        if middle.shape != space.shape:
            raise ValueError(
                f"The center has shape {middle.shape}, but a field on this box "
                f"has shape {space.shape}."
            )
        (ax.center_line,) = ax.plot(axis, middle, color=color, lw=1.5)
    ax.set_xlim(axis[0], axis[-1])
    return ax, band


def _periodic(space: PeriodicBox) -> bool:
    """Whether the box wraps: a bounded box is a periodic one only inside."""
    from ..symmetric_space.box import Box

    return not isinstance(space, Box)


def _wrapped(space: PeriodicBox, positions: np.ndarray) -> np.ndarray:
    if not _periodic(space):
        return positions
    return positions % np.asarray(space.lengths, dtype=float)


def _pieces(space: PeriodicBox, positions: np.ndarray) -> list[np.ndarray]:
    """A polyline split where it wraps round an axis of a periodic box."""
    positions = _wrapped(space, np.atleast_2d(positions))
    if not _periodic(space) or positions.shape[0] < 2:
        return [positions]
    half = 0.5 * np.asarray(space.lengths, dtype=float)
    jumps = np.any(np.abs(np.diff(positions, axis=0)) > half, axis=1)
    breaks = np.flatnonzero(jumps) + 1
    return [
        positions[piece]
        for piece in np.split(np.arange(positions.shape[0]), breaks)
        if piece.size >= 2
    ]


def _xy(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Picture coordinates: axis 1 across and axis 0 up, as the field is drawn,
    and a line's own coordinate across in one dimension."""
    if positions.shape[1] == 1:
        return positions[:, 0], np.zeros(positions.shape[0])
    return positions[:, 1], positions[:, 0]


def _check_dimension(space: PeriodicBox, what: str) -> None:
    if space.spatial_dimension > 2:
        raise NotImplementedError(
            f"{what} are drawn on a one- or two-dimensional box, not a "
            f"{space.spatial_dimension}-dimensional one."
        )


@plot_points.register
def _(
    space: PeriodicBox,
    points: Any,
    /,
    *,
    data: Any = None,
    ax: Any = None,
    marker: str = "o",
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
    """Scatter points of a box, optionally colored by a value each.

    In two dimensions the points go where the field is drawn, axis 1 across
    and axis 0 up, reduced into the period on a periodic box. In one
    dimension they sit on the axis, or at their *data* value when one is
    given, which puts observations on the line the field is drawn on.

    Args:
        space: the box.
        points: the points, in the box's own coordinates.
        data: one value per point, to color the markers by (and, in one
            dimension, to place them at).
        ax: axes to draw on. A new figure is made if omitted.
        marker: matplotlib marker.
        size: marker area.
        color: marker color, used when there is no *data*.
        cmap: color map for *data*.
        symmetric: put zero at the middle of the color scale.
        vmin: lower color limit.
        vmax: upper color limit.
        colorbar: attach a colorbar, which needs *data*. On when a label is
            given unless ``False``.
        colorbar_label: label for the colorbar, which turns one on.
        colorbar_kwargs: passed to ``figure.colorbar``.
        **kwargs: passed to ``scatter``.

    Returns:
        The ``(axes, collection)`` pair.

    Raises:
        ValueError: if *data* does not have one value per point.
        NotImplementedError: on a box of more than two dimensions.
    """
    _check_dimension(space, "Points")
    if ax is None:
        _, ax = subplots(space)
    positions = _wrapped(
        space,
        np.atleast_2d(np.asarray([np.atleast_1d(p) for p in points], dtype=float)),
    )
    if data is None:
        colors: Any = color
    else:
        colors = np.asarray(data, dtype=float).ravel()
        if colors.size != positions.shape[0]:
            raise ValueError(
                f"There are {positions.shape[0]} points and {colors.size} "
                "values to color them by."
            )
        low, high = color_limits(colors, vmin=vmin, vmax=vmax, symmetric=symmetric)
        kwargs.setdefault("cmap", cmap)
        kwargs.setdefault("vmin", low)
        kwargs.setdefault("vmax", high)
    xs, ys = _xy(positions)
    if positions.shape[1] == 1 and data is not None:
        ys = colors
    collection = ax.scatter(xs, ys, marker=marker, s=size, c=colors, **kwargs)
    wanted = colorbar or (colorbar is None and colorbar_label is not None)
    if wanted and data is not None:
        options = dict(shrink=0.7, pad=0.03)
        options.update(colorbar_kwargs or {})
        bar = ax.figure.colorbar(collection, ax=ax, **options)
        if colorbar_label is not None:
            bar.set_label(colorbar_label)
    return ax, collection


@plot_paths.register
def _(
    space: PeriodicBox,
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
    """Draw the straight paths of a box as one line collection.

    On a periodic box a path takes the short way round, and is split where
    it wraps so that each piece is drawn where it lies, as the sphere's are
    split at the dateline. On a bounded box nothing wraps.

    Args:
        space: the box.
        paths: ``(start, end)`` pairs of points.
        ax: axes to draw on. A new figure is made if omitted.
        count: samples along each path, between its ends.
        color: line color.
        linewidth: line width.
        alpha: opacity, low by default because these overlap heavily.
        **kwargs: passed to ``LineCollection``.

    Returns:
        The ``(axes, collection)`` pair.

    Raises:
        NotImplementedError: on a box of more than two dimensions.
    """
    from matplotlib.collections import LineCollection

    _check_dimension(space, "Paths")
    if ax is None:
        _, ax = subplots(space)
    segments = []
    for start, end in paths:
        first = np.atleast_1d(np.asarray(start, dtype=float))
        last = np.atleast_1d(np.asarray(end, dtype=float))
        if _periodic(space):
            nodes, _ = space.geodesic_quadrature(first, last, count=count)
            positions = np.vstack([first, np.asarray(nodes, dtype=float), last])
        else:
            fractions = np.linspace(0.0, 1.0, count + 2)[:, None]
            positions = first[None, :] + fractions * (last - first)[None, :]
        for piece in _pieces(space, positions):
            segments.append(np.column_stack(_xy(piece)))
    collection = LineCollection(
        segments, colors=color, linewidths=linewidth, alpha=alpha, **kwargs
    )
    ax.add_collection(collection)
    ax.autoscale_view()
    return ax, collection


@plot_balls.register
def _(
    space: PeriodicBox,
    centers: Any,
    radius: float,
    /,
    *,
    ax: Any = None,
    count: int = 90,
    color: str = "black",
    linewidth: float = 0.8,
    **kwargs: Any,
) -> Any:
    """Outline geodesic balls of one radius on a box: discs, or intervals.

    In two dimensions each disc's rim goes on as a polyline, wrapped and
    split on a periodic box; in one dimension each ball is a span on the
    axis.

    Args:
        space: the box.
        centers: the centers, in the box's own coordinates.
        radius: the *physical* radius, common to all of them.
        ax: axes to draw on. A new figure is made if omitted.
        count: points around each rim.
        color: line color, or the span color in one dimension.
        linewidth: line width.
        **kwargs: passed to ``LineCollection``, or to ``axvspan``.

    Returns:
        The ``(axes, collection)`` pair; in one dimension the second is the
        list of spans.

    Raises:
        ValueError: for a non-positive radius.
        NotImplementedError: on a box of more than two dimensions.
    """
    from matplotlib.collections import LineCollection

    if radius <= 0.0:
        raise ValueError(f"The radius must be positive, got {radius}.")
    _check_dimension(space, "Balls")
    if ax is None:
        _, ax = subplots(space)
    if space.spatial_dimension == 1:
        kwargs.setdefault("alpha", 0.2)
        length = float(space.lengths[0])
        spans = []
        for center in centers:
            middle = float(np.atleast_1d(np.asarray(center, dtype=float))[0])
            low, high = middle - radius, middle + radius
            if _periodic(space) and (low < 0.0 or high > length):
                low, high = low % length, high % length
                spans.append(ax.axvspan(low, length, color=color, **kwargs))
                spans.append(ax.axvspan(0.0, high, color=color, **kwargs))
            else:
                spans.append(ax.axvspan(low, high, color=color, **kwargs))
        return ax, spans
    angles = np.linspace(0.0, 2.0 * np.pi, count + 1)
    rim = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    segments = []
    for center in centers:
        middle = np.asarray(center, dtype=float)[:2]
        for piece in _pieces(space, middle[None, :] + rim):
            segments.append(np.column_stack(_xy(piece)))
    collection = LineCollection(segments, colors=color, linewidths=linewidth, **kwargs)
    ax.add_collection(collection)
    ax.autoscale_view()
    return ax, collection
