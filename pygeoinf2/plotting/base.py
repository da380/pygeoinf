"""
The dispatch layer: which renderer draws which space.

A space says how to *sample* itself; it does not say how to draw itself. So
rendering is a separate layer that dispatches on the space's type, and nothing
in ``symmetric_space`` imports matplotlib. That keeps the core usable from
anything headless, and it means a new space can be plotted by registering a
function rather than by growing a method.
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any

import numpy as np

__all__ = ["plot", "subplots", "color_limits", "show"]


def show() -> bool:
    """Show the figures, if the backend can.

    Under a non-interactive backend -- Agg, which is what the test suite runs
    the examples under -- this is a no-op, so an example can end with it
    unconditionally. ``pyplot.show()`` itself is not: on Agg it warns once per
    call that the canvas is non-interactive, which is six warnings in a suite
    run and no figures either way.

    Returns:
        Whether the figures were shown.
    """
    import matplotlib
    import matplotlib.pyplot as pyplot

    try:
        from matplotlib.backends import BackendFilter, backend_registry

        blind = backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE)
    except ImportError:  # pragma: no cover - matplotlib below 3.9
        from matplotlib import rcsetup

        blind = rcsetup.non_interactive_bk
    # Not "does the name end in agg": TkAgg and QtAgg do, and both have a
    # window. An unrecognized backend -- somebody's module:// -- is assumed to
    # be able to show, which is the way round that fails visibly.
    if matplotlib.get_backend().lower() in {name.lower() for name in blind}:
        return False
    pyplot.show()
    return True


@singledispatch
def subplots(space: Any, /, *, rows: int = 1, columns: int = 1, **kwargs: Any) -> Any:
    """A figure and axes with the projection this space needs.

    The counterpart of ``plt.subplots``, and it takes the same keywords, so a
    grid of panels works the way it does everywhere else. What it adds is the
    projection: a field on a sphere needs a map projection, and a field on an
    interval needs nothing at all.

    Args:
        space: the space whose fields will be drawn.
        rows: number of panel rows.
        columns: number of panel columns.
        **kwargs: passed through to ``matplotlib.pyplot.subplots``.

    Returns:
        The ``(figure, axes)`` pair ``plt.subplots`` returns.

    Raises:
        NotImplementedError: for a space with no registered renderer. The
            dispatch is by type, so a new space needs its own registration
            rather than inheriting one that would draw the wrong thing.
    """
    raise NotImplementedError(f"No renderer is registered for {type(space).__name__}.")


@singledispatch
def plot(space: Any, field: Any, /, **kwargs: Any) -> Any:
    """Draw a field of this space.

    Args:
        space: the space the field belongs to.
        field: the field to draw.
        **kwargs: renderer-specific; see the registered implementations.

    Returns:
        An ``(axes, mappable)`` pair, so the caller can set a title, restyle
        the colorbar, or add to the axes afterwards.

    Raises:
        NotImplementedError: for a space with no registered renderer.
    """
    raise NotImplementedError(f"No renderer is registered for {type(space).__name__}.")


def color_limits(
    values: np.ndarray,
    /,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
) -> tuple[float, float]:
    """Color limits for a field, optionally symmetric about zero.

    Symmetric limits are what a signed field almost always wants: with a
    diverging color map, limits that are not symmetric put the neutral color
    somewhere other than zero, and the eye reads the resulting picture as
    having a bias the data does not have.

    Args:
        values: the field's values.
        vmin: the lower limit. The data's minimum if omitted.
        vmax: the upper limit. The data's maximum if omitted.
        symmetric: widen the limits to be equal and opposite. An explicit
            *vmin* or *vmax* is respected, so this cannot silently override
            what a caller asked for.

    Returns:
        The ``(low, high)`` pair.
    """
    data = np.asarray(values, dtype=float)
    low = float(np.nanmin(data)) if vmin is None else vmin
    high = float(np.nanmax(data)) if vmax is None else vmax
    if symmetric:
        extent = max(abs(low), abs(high))
        return -extent, extent
    return low, high


@singledispatch
def plot_points(space: Any, points: Any, /, **kwargs: Any) -> Any:
    """Scatter points of a space, optionally colored by a value each.

    Dispatches on the space's type, as :func:`plot` does: a sphere draws on
    a map, a box on plain axes. See the registered implementations for the
    keywords; all take ``data=``, ``ax=``, ``marker=``, ``size=`` and
    ``color=``.

    Args:
        space: the space the points lie in.
        points: the points, in the space's own convention.
        **kwargs: renderer-specific.

    Returns:
        The ``(axes, collection)`` pair.

    Raises:
        NotImplementedError: for a space with no registered renderer.
    """
    raise NotImplementedError(f"No renderer is registered for {type(space).__name__}.")


@singledispatch
def plot_paths(space: Any, paths: Any, /, **kwargs: Any) -> Any:
    """Draw geodesic paths of a space as one line collection.

    Args:
        space: the space the paths lie in.
        paths: ``(start, end)`` pairs of points.
        **kwargs: renderer-specific; all take ``ax=``, ``count=``,
            ``color=``, ``linewidth=`` and ``alpha=``.

    Returns:
        The ``(axes, collection)`` pair.

    Raises:
        NotImplementedError: for a space with no registered renderer.
    """
    raise NotImplementedError(f"No renderer is registered for {type(space).__name__}.")


@singledispatch
def plot_balls(space: Any, centers: Any, radius: float, /, **kwargs: Any) -> Any:
    """Outline geodesic balls of one radius: the footprints of cap averages.

    The picture that goes with
    :meth:`~pygeoinf2.symmetric_space.base.SymmetricSpace.geodesic_ball_average_operator`:
    where each average was taken and how much it covers. Caps on a sphere,
    discs on a box, intervals on a line.

    Args:
        space: the space.
        centers: the ball centers, in the space's own convention.
        radius: the *physical* radius, common to all of them.
        **kwargs: renderer-specific; all take ``ax=``, ``count=``,
            ``color=`` and ``linewidth=``.

    Returns:
        The ``(axes, collection)`` pair.

    Raises:
        NotImplementedError: for a space with no registered renderer.
    """
    raise NotImplementedError(f"No renderer is registered for {type(space).__name__}.")


def plot_network(
    space: Any,
    paths: Any,
    /,
    *,
    ax: Any = None,
    sources: bool = True,
    receivers: bool = True,
    source_kwargs: dict | None = None,
    receiver_kwargs: dict | None = None,
    **kwargs: Any,
) -> tuple[Any, dict]:
    """The paths, with the sources and receivers that define them marked.

    v1's ``plot_geodesic_network``, on every geometry at once: the paths go
    on through :func:`plot_paths`, then the distinct start points as gold
    stars and the distinct end points as red circles through
    :func:`plot_points`, v1's styling. One path is a network of one, so
    there is no separate single-geodesic plotter.

    Args:
        space: the space the paths lie in.
        paths: ``(start, end)`` pairs of points.
        ax: axes to draw on. A new figure is made if omitted.
        sources: mark the distinct start points.
        receivers: mark the distinct end points.
        source_kwargs: styling for the sources, over the defaults.
        receiver_kwargs: styling for the receivers, over the defaults.
        **kwargs: passed to :func:`plot_paths`.

    Returns:
        The axes, and a dict of the artists under ``"paths"``, ``"sources"``
        and ``"receivers"``.
    """
    paths = list(paths)
    ax, collection = plot_paths(space, paths, ax=ax, **kwargs)
    artists: dict = {"paths": collection}
    if sources:
        style = dict(marker="*", color="gold", size=150.0, edgecolors="black", zorder=6)
        style.update(source_kwargs or {})
        _, artists["sources"] = plot_points(
            space, _distinct(p[0] for p in paths), ax=ax, **style
        )
    if receivers:
        style = dict(marker="o", color="red", size=50.0, edgecolors="white", zorder=6)
        style.update(receiver_kwargs or {})
        _, artists["receivers"] = plot_points(
            space, _distinct(p[1] for p in paths), ax=ax, **style
        )
    return ax, artists


def _distinct(points: Any) -> list[np.ndarray]:
    """The points with exact repeats removed, first occurrences kept in order."""
    seen: set = set()
    kept: list[np.ndarray] = []
    for point in points:
        array = np.atleast_1d(np.asarray(point, dtype=float))
        key = array.tobytes()
        if key not in seen:
            seen.add(key)
            kept.append(array)
    return kept
