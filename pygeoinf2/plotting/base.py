"""
The dispatch layer: which renderer draws which space.

A space says how to *sample* itself; it does not say how to draw itself. So
rendering is a separate layer that dispatches on the space's type, and nothing
in ``symmetric_space`` imports matplotlib. That keeps the core usable from
anything headless, and it means a new space can be plotted by registering a
function rather than by growing a method.

See DESIGN.md section 20.5, O8.
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any

import numpy as np

__all__ = ["plot", "subplots", "colour_limits"]


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
        the colourbar, or add to the axes afterwards.

    Raises:
        NotImplementedError: for a space with no registered renderer.
    """
    raise NotImplementedError(f"No renderer is registered for {type(space).__name__}.")


def colour_limits(
    values: np.ndarray,
    /,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
) -> tuple[float, float]:
    """Colour limits for a field, optionally symmetric about zero.

    Symmetric limits are what a signed field almost always wants: with a
    diverging colour map, limits that are not symmetric put the neutral colour
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
