"""Fields on a disk or an annulus, drawn whole in the plane.

A disk is two-dimensional and needs no cut: ``plot`` colors it in, on the
radial nodes of the domain -- or of the padded mesh, on request -- and on many
more angles than the grid has, with the domain's boundaries outlined. The
values round each ring are the Fourier modes evaluated at those angles times
their radial coefficient functions at the nodes: one small matrix product.

**What is drawn is the field with the vector's components**, since off the
grid that is the only field there is: a product left on the grid is drawn as
its projection onto the kept modes.

Points are ``(radius, angle)``, the angle in degrees anticlockwise from the
``x`` axis, and are drawn where they are in the plane. A path is a polyline
of such points, consecutive ones joined by a straight segment: a ``(start,
end)`` pair is a straight path, the geodesic of the plane, and a ray traced
through a medium is the list of its nodes.

A layered space of disks is drawn as one disk, each layer's part on one color
scale, with a circle at every interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..disk import Disk
from ..disk import Layered as LayeredDisk
from ..spectral_element._cylindrical import circular_harmonics
from .base import color_limits, plot, plot_paths, plot_points, subplots

__all__: list[str] = []


def _shells(space: Disk | LayeredDisk, /) -> list[Disk]:
    """The disks of a space: itself, or a layered space's layers."""
    return list(space.layers) if isinstance(space, LayeredDisk) else [space]


def _layers(space: Disk | LayeredDisk, field: Any, /) -> tuple[list[Disk], list[Any]]:
    """The disks a field lives in and its part in each."""
    if isinstance(space, LayeredDisk):
        return _shells(space), list(field)
    return [space], [field]


def _polar_values(
    space: Disk, field: np.ndarray, /, *, angles: int, padding: bool
) -> tuple[np.ndarray, np.ndarray]:
    """The field on the radial nodes and ``angles`` equally spaced angles from
    zero, as a ``(radii, angles)`` array, and the radii.

    Raises:
        ValueError: if the field has the wrong shape.
    """
    values = np.asarray(field, dtype=float)
    if values.shape != space.grid_shape:
        raise ValueError(
            f"A field on this disk has shape {space.grid_shape}, got {values.shape}."
        )
    functions = space.basis.synthesise(
        space.to_components(values), physical=not padding
    )
    psi = 2.0 * np.pi * np.arange(angles) / angles
    modes = circular_harmonics(space.mmax, psi)
    radii = space.radii if padding else space.interior_radii
    return np.einsum("smr,smp->rp", functions, modes), radii


def _xy(points: Any, /) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian coordinates of points ``(radius, angle)``, the angle in
    degrees."""
    located = np.asarray(points, dtype=float).reshape(-1, 2)
    t = np.radians(located[:, 1])
    return located[:, 0] * np.cos(t), located[:, 0] * np.sin(t)


def _outline(ax: Any, shells: list[Disk], /, *, color: str = "black") -> None:
    """Circles at every boundary of the domains, each once."""
    psi = np.linspace(0.0, 2.0 * np.pi, 361)
    edges = {shell.inner_radius for shell in shells} | {
        shell.radius for shell in shells
    }
    for radius in sorted(edges):
        if radius > 0.0:
            ax.plot(
                radius * np.cos(psi), radius * np.sin(psi), color=color, linewidth=0.6
            )


@subplots.register(Disk)
@subplots.register(LayeredDisk)
def _(
    space: Disk | LayeredDisk,
    /,
    *,
    rows: int = 1,
    columns: int = 1,
    **kwargs: Any,
) -> Any:
    """Ordinary axes with an equal aspect: a disk needs no projection.

    Args:
        space: the disk, or a layered space of them.
        rows: number of panel rows.
        columns: number of panel columns.
        **kwargs: passed through to ``matplotlib.pyplot.subplots``.

    Returns:
        The ``(figure, axes)`` pair.
    """
    import matplotlib.pyplot as pyplot

    kwargs.setdefault("figsize", (4.4 * columns, 4.0 * rows))
    kwargs.setdefault("layout", "constrained")
    kwargs.setdefault("subplot_kw", {"aspect": "equal"})
    return pyplot.subplots(rows, columns, **kwargs)


@plot.register(Disk)
@plot.register(LayeredDisk)
def _(
    space: Disk | LayeredDisk,
    field: Any,
    /,
    *,
    ax: Any = None,
    angles: int | None = None,
    padding: bool = False,
    cmap: str = "RdBu",
    symmetric: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    outline: bool = True,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Draw a field on a disk or an annulus, in the plane.

    Args:
        space: the disk or annulus, or a layered space of them.
        field: a vector of the space.
        ax: axes to draw on. A new figure is made if omitted.
        angles: how many angles round each ring to evaluate at. Four per
            order of the truncation by default, and at least 180.
        padding: draw the padded mesh too, beyond the outline of the domain.
        cmap: color map.
        symmetric: put zero at the middle of the color scale.
        vmin: lower color limit.
        vmax: upper color limit.
        colorbar: attach a colorbar.
        colorbar_label: label for the colorbar.
        outline: draw the boundaries of the domain.
        title: a title for the axes.
        **kwargs: passed to ``pcolormesh``.

    Returns:
        The ``(axes, mappable)`` pair.

    Raises:
        ValueError: if the field has the wrong shape, or ``padding`` is asked
            of a layered space.
    """
    shells, fields = _layers(space, field)
    if len(shells) > 1 and padding:
        raise ValueError(
            "The padding of one layer lies over its neighbours, and cannot be "
            "drawn beside them: draw that layer on its own."
        )
    count = (
        max(180, 4 * max(shell.mmax for shell in shells))
        if angles is None
        else int(angles)
    )
    rings = [
        _polar_values(shell, part, angles=count, padding=padding)
        for shell, part in zip(shells, fields)
    ]
    if ax is None:
        _, ax = subplots(shells[0])
    psi = 2.0 * np.pi * np.arange(count + 1) / count
    # One color scale for every layer, or a jump would not look like one.
    low, high = color_limits(
        np.concatenate([values.ravel() for values, _ in rings]),
        vmin=vmin,
        vmax=vmax,
        symmetric=symmetric,
    )
    for values, radii in rings:
        closed = np.concatenate([values, values[:, :1]], axis=1)
        mappable = ax.pcolormesh(
            radii[:, None] * np.cos(psi)[None, :],
            radii[:, None] * np.sin(psi)[None, :],
            closed,
            cmap=cmap,
            vmin=low,
            vmax=high,
            shading="gouraud",
            **kwargs,
        )
    if outline:
        _outline(ax, shells)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if colorbar:
        bar = ax.figure.colorbar(mappable, ax=ax, shrink=0.8, pad=0.03)
        if colorbar_label is not None:
            bar.set_label(colorbar_label)
    if title is not None:
        ax.set_title(title)
    return ax, mappable


@plot_points.register(Disk)
@plot_points.register(LayeredDisk)
def _(
    space: Disk | LayeredDisk,
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
    """Scatter points of a disk where they are in the plane, optionally
    colored by a value each.

    Args:
        space: the disk or annulus, or a layered space of them.
        points: ``(radius, angle)`` pairs, the angle in degrees anticlockwise
            from the ``x`` axis.
        data: one value per point, to color the markers by.
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
    """
    if ax is None:
        _, ax = subplots(space)
    xs, ys = _xy(points)
    if data is None:
        colors: Any = color
    else:
        colors = np.asarray(data, dtype=float).ravel()
        if colors.size != xs.size:
            raise ValueError(
                f"There are {xs.size} points and {colors.size} values to color "
                "them by."
            )
        low, high = color_limits(colors, vmin=vmin, vmax=vmax, symmetric=symmetric)
        kwargs.setdefault("cmap", cmap)
        kwargs.setdefault("vmin", low)
        kwargs.setdefault("vmax", high)
    collection = ax.scatter(xs, ys, marker=marker, s=size, c=colors, **kwargs)
    wanted = colorbar or (colorbar is None and colorbar_label is not None)
    if wanted and data is not None:
        options = dict(shrink=0.7, pad=0.03)
        options.update(colorbar_kwargs or {})
        bar = ax.figure.colorbar(collection, ax=ax, **options)
        if colorbar_label is not None:
            bar.set_label(colorbar_label)
    return ax, collection


@plot_paths.register(Disk)
@plot_paths.register(LayeredDisk)
def _(
    space: Disk | LayeredDisk,
    paths: Any,
    /,
    *,
    ax: Any = None,
    count: int = 0,
    color: str = "black",
    linewidth: float = 0.4,
    alpha: float = 0.15,
    outline: bool = False,
    **kwargs: Any,
) -> Any:
    """Draw paths in a disk as one line collection.

    A path is a polyline: a sequence of two or more points ``(radius,
    angle)``, consecutive ones joined by a straight segment. A ``(start,
    end)`` pair is the straight path between them, the plane's geodesic; a ray
    traced through a medium is the list of its nodes, in order.

    Args:
        space: the disk or annulus, or a layered space of them.
        paths: the paths, each a sequence of points.
        ax: axes to draw on. A new figure is made if omitted.
        count: points inserted along each segment, between its ends. Taken
            for the sake of the keywords every geometry's renderer answers
            to; a segment is straight, and they change nothing drawn.
        color: line color.
        linewidth: line width.
        alpha: opacity, low by default because these overlap heavily.
        outline: draw the boundaries of the domain too, for paths drawn on
            their own.
        **kwargs: passed to ``LineCollection``.

    Returns:
        The ``(axes, collection)`` pair.

    Raises:
        ValueError: if a path has fewer than two points.
    """
    from matplotlib.collections import LineCollection

    if ax is None:
        _, ax = subplots(space)
    fractions = np.linspace(0.0, 1.0, int(count) + 2)[:-1]
    segments = []
    for path in paths:
        xs, ys = _xy(path)
        if xs.size < 2:
            raise ValueError("A path needs two points at least.")
        nodes = np.column_stack([xs, ys])
        steps = nodes[1:] - nodes[:-1]
        filled = nodes[:-1, None, :] + fractions[None, :, None] * steps[:, None, :]
        segments.append(np.vstack([filled.reshape(-1, 2), nodes[-1:]]))
    collection = LineCollection(
        segments, colors=color, linewidths=linewidth, alpha=alpha, **kwargs
    )
    ax.add_collection(collection)
    if outline:
        _outline(ax, _shells(space))
    ax.autoscale_view()
    ax.set_aspect("equal")
    return ax, collection
