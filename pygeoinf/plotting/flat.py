"""Fields of a flat layer: a strip drawn whole, a slab cut to be seen.

A strip is a field of ``(depth, x)``, a vertical section already, and is drawn
whole: ``plot`` colors it in over ``x`` and depth, depth increasing downward,
so that the top of the layer is at the top of the picture and above on the
page is above in the layer, and ``plot_section`` draws the same. A slab is a
field of ``(depth, x, y)`` and has to be cut to be seen, as a ball has, and
there are three cuts: a **lateral map** at one depth, which is what ``plot``
draws there, the top of the slab unless ``depth=`` names another; a
**section** by a vertical plane, along ``x`` at one ``y`` or along ``y`` at
one ``x``, which ``plot_section`` draws as a strip is drawn; and a
**profile** beneath one lateral position, against depth, which
``plot_profile`` draws and a strip has too. The three dispatch on the space,
and this module registers their renderers for strips, slabs and layered
spaces of them.

**What is drawn.** A strip is drawn from the vector's values on its grid, the
nodes of the mesh in depth and the points of the lateral grid, as an interval
or a box is: a product left on the grid is drawn as it stands. The exception
is a bound of a bounded lateral domain that is not a point of the grid, where
the column drawn is evaluated, as below. A cut through a slab, and a profile,
lie off the grid in general, and are drawn from the space's own point
evaluation: the field with the vector's components, which off the grid is the
only field there is, at the grid's lateral points and nodes in depth that lie
in the cut, and at the bounds of a bounded lateral domain. For a vector in the
span of the kept modes, a draw of a prior or a posterior mean, the two agree
to rounding.

**Across.** Along a lateral axis whose domain is the whole period, the
picture spans the period: the first column of values is drawn again one
period on, where the field takes it again, as a disk's first angle closes its
ring. Along a bounded axis the picture is of the grid's points within the
bounds and of the bounds themselves, so that it fills the domain: where a
bound is not a point of the grid, a column at the bound is drawn from the
space's own point evaluation, the projection onto the kept modes, and the
rest of a strip's columns are its grid values. A bound that is a point of the
grid, to a relative ``1e-12`` of the domain's width, adds no column.

**Layers.** A layered space of strips or slabs is drawn layer by layer, each
layer from its own values and all on one color scale, so that a jump at an
interface is drawn as one: nothing is interpolated across it, and a line marks
every boundary of a layer. Above is shallower, as in the spaces themselves: a
lateral map at the depth of an interface takes ``side="above"`` for the
shallower layer's value or ``side="below"`` for the deeper's, as the space's
own evaluation does.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..slab import Layered as LayeredSlab
from ..slab import Slab
from ..spectral_element._flat import FlatLayered, FlatSpace
from ..spectral_element._points import _POSITION_RTOL
from ..strip import Layered as LayeredStrip
from ..strip import Strip
from .base import (
    color_limits,
    plot,
    plot_points,
    plot_profile,
    plot_section,
    subplots,
)

__all__: list[str] = []

#: The names of the lateral axes, in order.
_AXIS_NAMES = ("x", "y")


def _lateral_axes(space: FlatSpace | FlatLayered, /) -> int:
    """The number of lateral axes of a flat space or a layered space of them:
    one for a strip, two for a slab."""
    return space.spatial_dimension - 1


def _layers(space: Any, field: Any, /) -> tuple[list[FlatSpace], list[np.ndarray]]:
    """The flat layers a field lives in and its part in each, as arrays: one
    layer, or a layered space's layers.

    Raises:
        ValueError: if the field has the wrong number of parts or a part the
            wrong shape.
    """
    if isinstance(space, FlatLayered):
        layers, parts = list(space.layers), list(field)
        if len(parts) != len(layers):
            raise ValueError(
                f"A field of this layered space has {len(layers)} parts, one "
                f"per layer; got {len(parts)}."
            )
    else:
        layers, parts = [space], [field]
    checked = []
    for layer, part in zip(layers, parts):
        values = np.asarray(part, dtype=float)
        if values.shape != layer.grid_shape:
            raise ValueError(
                f"A field on this {layer._geometry} has shape {layer.grid_shape}, "
                f"got {values.shape}."
            )
        checked.append(values)
    return layers, checked


def _closed(layer: FlatSpace, axis: int, /, *, padding: bool) -> bool:
    """Whether the picture closes the period along a lateral axis: where the
    domain is the whole period, or the padding is drawn."""
    return padding or layer.lateral_domain.padding[axis] == 0.0


def _across(
    layer: FlatSpace, axis: int, /, *, padding: bool
) -> tuple[np.ndarray, np.ndarray]:
    """The coordinates drawn along a lateral axis, and which point of the
    grid along it each one is, ``-1`` for a coordinate that is none.

    The whole period, its first point again one period on, where the domain
    is the whole period or the padding is drawn. Along a bounded axis
    otherwise, the grid's points within the bounds, and before them and
    after them the bounds themselves where they are not points of the grid,
    to the tolerance the grid's mask of the domain takes: the coordinates at
    which the field is evaluated rather than read off the grid.
    """
    grid = layer.lateral_axes[axis]
    domain = layer.lateral_domain
    if _closed(layer, axis, padding=padding):
        index = np.append(np.arange(grid.size), 0)
        return np.append(grid, grid[0] + domain.lengths[axis]), index
    mask = layer.transform.mask
    others = tuple(a for a in range(mask.ndim) if a != axis)
    index = np.flatnonzero(mask.any(axis=others) if others else mask)
    coordinates = grid[index]
    low, high = (float(bound) for bound in domain.bounds[axis])
    slack = _POSITION_RTOL * (high - low)
    if not index.size or abs(coordinates[0] - low) > slack:
        coordinates, index = np.insert(coordinates, 0, low), np.insert(index, 0, -1)
    if abs(coordinates[-1] - high) > slack:
        coordinates, index = np.append(coordinates, high), np.append(index, -1)
    return coordinates, index


def _span(layer: FlatSpace, axis: int, /, *, padding: bool) -> tuple[float, float]:
    """What the axes span along a lateral axis: the period drawn, closed, or
    the bounds of a bounded domain."""
    if _closed(layer, axis, padding=padding):
        low = float(layer.lateral_axes[axis][0])
        return low, low + float(layer.lateral_domain.lengths[axis])
    low, high = layer.lateral_domain.bounds[axis]
    return float(low), float(high)


def _strip_image(
    layer: FlatSpace, values: np.ndarray, /, *, padding: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A strip's values as drawn, ``(depths, across)``, the depths and the
    coordinates across: at the nodes of the domain, or of the padded mesh.

    Every column is the strip's values on the grid but a column at a bound of
    a bounded domain that is not a point of the grid, which is evaluated by
    the layer's own point evaluation, from the layer's own values: at a depth
    that is the layer's top or base, the field on the layer's side of it.
    """
    if padding:
        rows = np.arange(layer.depths.size)
    else:
        rows = np.flatnonzero(layer.interior_mask.any(axis=1))
    depths = layer.depths[rows]
    across, index = _across(layer, 0, padding=padding)
    image = np.empty((depths.size, across.size))
    grid = index >= 0
    image[:, grid] = values[np.ix_(rows, index[grid])]
    if not grid.all():
        edges = across[~grid]
        points = np.column_stack(
            [np.repeat(depths, edges.size), np.tile(edges, depths.size)]
        )
        image[:, ~grid] = layer.evaluate(values, points).reshape(
            depths.size, edges.size
        )
    return image, depths, across


def _slab_section(
    layer: FlatSpace, values: np.ndarray, along: int, at: float, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A slab's field on a vertical plane, by its point evaluation: along the
    lateral axis ``along`` at the coordinate ``at`` of the other, at the
    nodes of the domain in depth and at the coordinates :func:`_across`
    draws, the bounds of a bounded domain among them. The values ``(depths,
    across)``, the depths and the coordinates across.

    Raises:
        ValueError: if ``at`` lies outside a bounded lateral domain.
    """
    depths = layer.interior_depths
    across, _ = _across(layer, along, padding=False)
    lateral = [np.full(depths.size * across.size, float(at))] * 2
    lateral[along] = np.tile(across, depths.size)
    points = np.column_stack([np.repeat(depths, across.size), *lateral])
    drawn = layer.evaluate(values, points).reshape(depths.size, across.size)
    return drawn, depths, across


def _outline(
    ax: Any, layers: list[FlatSpace], axis: int, /, *, color: str = "black"
) -> None:
    """Lines at every boundary of the layers' domains, each once: across at
    the depth of every top, interface and base, and down at the bounds of a
    bounded lateral domain along the axis drawn across."""
    edges = {layer.top for layer in layers} | {layer.bottom for layer in layers}
    for depth in sorted(edges):
        ax.axhline(depth, color=color, linewidth=0.6)
    if layers[0].lateral_domain.padding[axis] > 0.0:
        for bound in layers[0].lateral_domain.bounds[axis]:
            ax.axvline(bound, color=color, linewidth=0.6)


def _finish(
    ax: Any,
    mappable: Any,
    /,
    *,
    colorbar: bool,
    colorbar_label: str | None,
    title: str | None,
) -> None:
    """The colorbar and the title, where asked for.

    The colorbar is an inset of the axes, beside them and as tall as them:
    the axes keep the aspect of the layer, and a colorbar taken from the
    space allotted to them would be sized to that space and not to the
    picture.
    """
    if colorbar:
        bar = ax.figure.colorbar(mappable, cax=ax.inset_axes((1.03, 0.0, 0.03, 1.0)))
        if colorbar_label is not None:
            bar.set_label(colorbar_label)
    if title is not None:
        ax.set_title(title)


@subplots.register(FlatSpace)
@subplots.register(FlatLayered)
def _(
    space: FlatSpace | FlatLayered,
    /,
    *,
    rows: int = 1,
    columns: int = 1,
    **kwargs: Any,
) -> Any:
    """Ordinary axes: a flat layer needs no projection.

    Panels are wide for a strip, whose picture is a section, and square for
    a slab, whose ``plot`` is a lateral map.

    Args:
        space: the strip or slab, or a layered space of them.
        rows: number of panel rows.
        columns: number of panel columns.
        **kwargs: passed through to ``matplotlib.pyplot.subplots``.

    Returns:
        The ``(figure, axes)`` pair.
    """
    import matplotlib.pyplot as pyplot

    width, height = (6.0, 3.2) if _lateral_axes(space) == 1 else (4.4, 4.0)
    kwargs.setdefault("figsize", (width * columns, height * rows))
    kwargs.setdefault("layout", "constrained")
    return pyplot.subplots(rows, columns, **kwargs)


@plot.register(Strip)
@plot.register(LayeredStrip)
def _(
    space: Strip | LayeredStrip,
    field: Any,
    /,
    *,
    depth: float | None = None,
    side: str | None = None,
    **kwargs: Any,
) -> Any:
    """Draw a field of a strip, or of a layered strip, whole: an image over
    ``x`` and depth, which is its :func:`plot_section`, with the same
    keywords. :func:`plot_profile` draws it beneath one ``x``.

    Args:
        space: the strip, or a layered space of them.
        field: a vector of the space.
        depth: refused: it names a slab's map at one depth.
        side: refused: it names the side of an interface a slab's map is on.
        **kwargs: those of the strip's :func:`plot_section`.

    Returns:
        The ``(axes, mappable)`` pair.

    Raises:
        TypeError: if ``depth`` or ``side`` is given: a map at one depth is
            of a slab, and a strip's level at one depth is a line.
    """
    if depth is not None or side is not None:
        raise TypeError(
            "A map at one depth is of a slab: a strip has one lateral axis, and "
            "its level at one depth is a line. plot draws a strip whole."
        )
    return plot_section(space, field, **kwargs)


@plot.register(Slab)
@plot.register(LayeredSlab)
def _(
    space: Slab | LayeredSlab,
    field: Any,
    /,
    *,
    depth: float | None = None,
    side: str | None = None,
    ax: Any = None,
    cmap: str = "RdBu",
    symmetric: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Draw a field of a slab, or of a layered slab, on the horizontal plane
    at one depth, as a map over ``x`` and ``y``: the top, or ``depth=``.

    The field is evaluated at the lateral grid's points in the domain at that
    depth, and along a bounded axis at its bounds too, so that the map fills
    the domain, by the space's own point evaluation, and drawn with ``x``
    across and ``y`` up.
    :func:`plot_section` and :func:`plot_profile` are the other two cuts.

    Args:
        space: the slab, or a layered space of them.
        field: a vector of the space.
        depth: the depth of the plane, within the domain. The top if
            omitted.
        side: in a layered space, which of the two fields is meant at a
            depth that is an interface: ``"above"``, the shallower layer's,
            or ``"below"``, the deeper's.
        ax: axes to draw on. A new figure is made if omitted.
        cmap: color map.
        symmetric: put zero at the middle of the color scale.
        vmin: lower color limit.
        vmax: upper color limit.
        colorbar: attach a colorbar.
        colorbar_label: label for the colorbar.
        title: a title for the axes.
        **kwargs: passed to ``pcolormesh``.

    Returns:
        The ``(axes, mappable)`` pair.

    Raises:
        ValueError: if the field, or a part of it, has the wrong shape, or
            the depth lies outside the domain, in no layer, or on an
            interface with no side named.
    """
    layers, parts = _layers(space, field)
    first = layers[0]
    level = first.top if depth is None else float(depth)
    xs, _ = _across(first, 0, padding=False)
    ys, _ = _across(first, 1, padding=False)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    points = np.column_stack(
        [np.full(grid_x.size, level), grid_x.ravel(), grid_y.ravel()]
    )
    if isinstance(space, FlatLayered):
        values = space.evaluate(tuple(parts), points, side=side)
    else:
        values = space.evaluate(parts[0], points)
    values = values.reshape(grid_x.shape)
    if ax is None:
        _, ax = subplots(space)
    low, high = color_limits(values, vmin=vmin, vmax=vmax, symmetric=symmetric)
    mappable = ax.pcolormesh(
        grid_x,
        grid_y,
        values,
        cmap=cmap,
        vmin=low,
        vmax=high,
        shading="gouraud",
        **kwargs,
    )
    ax.set_xlim(*_span(first, 0, padding=False))
    ax.set_ylim(*_span(first, 1, padding=False))
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _finish(ax, mappable, colorbar=colorbar, colorbar_label=colorbar_label, title=title)
    return ax, mappable


@plot_section.register(Strip)
@plot_section.register(Slab)
@plot_section.register(LayeredStrip)
@plot_section.register(LayeredSlab)
def _(
    space: Strip | Slab | LayeredStrip | LayeredSlab,
    field: Any,
    /,
    *,
    x: float | None = None,
    y: float | None = None,
    ax: Any = None,
    padding: bool = False,
    aspect: str | float = "equal",
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
    """Draw a field of a strip, or a section of a slab by a vertical plane,
    as an image over the lateral coordinate and depth, depth increasing
    downward.

    A strip is drawn whole, from its values on the grid: what ``plot`` draws
    there. Across a bounded domain the picture fills the bounds: a column at
    a bound that is not a point of the grid is evaluated by the space's own
    point evaluation, the projection onto the kept modes, while the rest are
    the grid values. A slab is cut by a vertical plane, along ``x`` at one
    ``y`` or along ``y`` at one ``x``, and its field there evaluated at the
    grid's points along the cut, the bounds of a bounded domain too, and the
    nodes of the domain in depth. A layered space is drawn layer by layer on
    one color scale, each layer from its own values, an evaluated column
    too, so that a jump at an interface stays a jump.

    Args:
        space: the strip or slab, or a layered space of them.
        field: a vector of the space.
        x: in a slab, the section along ``y`` at this ``x``.
        y: in a slab, the section along ``x`` at this ``y``. With neither,
            the section along ``x`` through the middle of the domain in
            ``y``.
        ax: axes to draw on. A new figure is made if omitted.
        padding: draw a strip's padded mesh too, in depth and across a
            bounded lateral domain, beyond the outline of the domain.
        aspect: the axes' aspect, ``"equal"`` by default since both axes are
            lengths; ``"auto"``, or a number, the vertical exaggeration, for
            a layer much wider than it is deep.
        cmap: color map.
        symmetric: put zero at the middle of the color scale.
        vmin: lower color limit.
        vmax: upper color limit.
        colorbar: attach a colorbar.
        colorbar_label: label for the colorbar.
        outline: draw the boundaries of the domain: a line across at every
            top, interface and base, and lines down at the bounds of a
            bounded lateral domain.
        title: a title for the axes.
        **kwargs: passed to ``pcolormesh``.

    Returns:
        The ``(axes, mappable)`` pair, the mappable being the last layer's
        mesh.

    Raises:
        ValueError: if the field, or a part of it, has the wrong shape;
            ``x`` or ``y`` is given for a strip, or both for a slab; the cut
            lies outside a bounded lateral domain; or ``padding`` is asked of
            a slab, whose sections are evaluated and reach the domain alone,
            or of a layered space, whose layers' padding lies over their
            neighbours.
    """
    axes = _lateral_axes(space)
    layers, parts = _layers(space, field)
    if padding and len(layers) > 1:
        raise ValueError(
            "The padding of one layer lies over its neighbours, and cannot be "
            "drawn beside them: draw that layer on its own."
        )
    if axes == 1:
        if x is not None or y is not None:
            raise ValueError(
                "A strip is its own depth section: x and y name sections of a slab."
            )
        along = 0
        pieces = [
            _strip_image(layer, part, padding=padding)
            for layer, part in zip(layers, parts)
        ]
    else:
        if padding:
            raise ValueError(
                "A section of a slab is evaluated, and evaluation reaches the "
                "domain alone: its padding is drawn on a strip only."
            )
        if x is not None and y is not None:
            raise ValueError("Name the section by one of x and y.")
        if x is not None:
            along, at = 1, float(x)
        else:
            along = 0
            low, high = layers[0].lateral_domain.bounds[1]
            at = 0.5 * (low + high) if y is None else float(y)
        pieces = [
            _slab_section(layer, part, along, at) for layer, part in zip(layers, parts)
        ]
    if ax is None:
        _, ax = subplots(space)
    # One color scale for every layer, or a jump would not look like one.
    low, high = color_limits(
        np.concatenate([values.ravel() for values, _, _ in pieces]),
        vmin=vmin,
        vmax=vmax,
        symmetric=symmetric,
    )
    for values, depths, across in pieces:
        mappable = ax.pcolormesh(
            across,
            depths,
            values,
            cmap=cmap,
            vmin=low,
            vmax=high,
            shading="gouraud",
            **kwargs,
        )
    if outline:
        _outline(ax, layers, along)
    ax.set_xlim(*_span(layers[0], along, padding=padding))
    ax.set_ylim(float(pieces[-1][1][-1]), float(pieces[0][1][0]))
    ax.set_aspect(aspect)
    ax.set_xlabel(_AXIS_NAMES[along])
    ax.set_ylabel("depth")
    _finish(ax, mappable, colorbar=colorbar, colorbar_label=colorbar_label, title=title)
    return ax, mappable


@plot_profile.register(Strip)
@plot_profile.register(Slab)
@plot_profile.register(LayeredStrip)
@plot_profile.register(LayeredSlab)
def _(
    space: Strip | Slab | LayeredStrip | LayeredSlab,
    field: Any,
    /,
    *position: float,
    ax: Any = None,
    points: int = 200,
    **kwargs: Any,
) -> Any:
    """Draw a field of a strip or a slab beneath one lateral position,
    against depth, depth increasing downward.

    The field is evaluated by the space's own point evaluation at depths
    evenly spaced through each layer, both ends included, and drawn with its
    value across and depth down. A layered space is one line per layer in one
    color, which leaves a break at every jump.

    Args:
        space: the strip or slab, or a layered space of them.
        field: a vector of the space.
        *position: the lateral position: ``x`` in a strip, ``x, y`` in a
            slab.
        ax: ordinary axes to draw on. A new figure is made if omitted.
        points: how many depths of the domain to evaluate at, shared among
            the layers of a layered space by their thickness, at least eight
            each.
        **kwargs: passed to ``Axes.plot``.

    Returns:
        The ``(axes, line)`` pair, the line being the last layer's.

    Raises:
        ValueError: if the position has another number of coordinates than
            the space has lateral axes, lies outside a bounded lateral
            domain, or the field, or a part of it, has the wrong shape.
    """
    import matplotlib.pyplot as pyplot

    axes = _lateral_axes(space)
    if len(position) != axes:
        names = ", ".join(_AXIS_NAMES[:axes])
        raise ValueError(
            f"A profile of a {'strip' if axes == 1 else 'slab'} is beneath a "
            f"position {names}; got {len(position)} coordinates."
        )
    lateral = tuple(float(value) for value in position)
    layers, parts = _layers(space, field)
    if ax is None:
        _, ax = pyplot.subplots(figsize=(3.2, 5.0), layout="constrained")
    extent = layers[-1].bottom - layers[0].top
    line = None
    for layer, part in zip(layers, parts):
        # As many points as the layer's share of the whole, and a few at least.
        share = (layer.bottom - layer.top) / extent
        depths = np.linspace(
            layer.top, layer.bottom, max(8, int(round(share * int(points))))
        )
        values = layer.evaluate(part, [(depth, *lateral) for depth in depths])
        if line is not None:
            # One field, one color, one legend entry, and a break at the jump.
            kwargs = {**kwargs, "color": line.get_color(), "label": "_nolegend_"}
        (line,) = ax.plot(values, depths, **kwargs)
    ax.set_ylim(layers[-1].bottom, layers[0].top)
    ax.set_ylabel("depth")
    return ax, line


def _wrapped(layer: FlatSpace, located: np.ndarray, /) -> np.ndarray:
    """Lateral coordinates reduced into the period along every axis whose
    domain is the whole period; along a bounded axis, as they are."""
    domain = layer.lateral_domain
    out = located.copy()
    for axis, (origin, length, pad) in enumerate(
        zip(domain.origin, domain.lengths, domain.padding)
    ):
        if pad == 0.0:
            out[:, axis] = origin + np.mod(out[:, axis] - origin, length)
    return out


@plot_points.register(FlatSpace)
@plot_points.register(FlatLayered)
def _(
    space: FlatSpace | FlatLayered,
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
    """Scatter points of a strip or a slab where its picture puts them,
    optionally colored by a value each.

    In a strip a point ``(depth, x)`` goes where :func:`plot` and
    :func:`plot_section` draw it, ``x`` across and depth down, and the depth
    axis is turned downward if it is not already. In a slab a point
    ``(depth, x, y)`` goes on the lateral map :func:`plot` draws, at
    ``(x, y)``; its depth is set aside, as a map has nowhere to put it. Along an axis whose domain is the whole period, a
    coordinate is reduced into the period.

    Args:
        space: the strip or slab, or a layered space of them.
        points: the points, depth first: ``(depth, x)`` pairs in a strip,
            ``(depth, x, y)`` triples in a slab.
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
    axes = _lateral_axes(space)
    first = space.layers[0] if isinstance(space, FlatLayered) else space
    located = np.asarray(points, dtype=float).reshape(-1, 1 + axes)
    lateral = _wrapped(first, located[:, 1:])
    if ax is None:
        _, ax = subplots(space)
    if axes == 1:
        xs, ys = lateral[:, 0], located[:, 0]
    else:
        xs, ys = lateral[:, 0], lateral[:, 1]
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
    if axes == 1 and not ax.yaxis_inverted():
        ax.invert_yaxis()
    wanted = colorbar or (colorbar is None and colorbar_label is not None)
    if wanted and data is not None:
        options = dict(shrink=0.7, pad=0.03)
        options.update(colorbar_kwargs or {})
        bar = ax.figure.colorbar(collection, ax=ax, **options)
        if colorbar_label is not None:
            bar.set_label(colorbar_label)
    return ax, collection
