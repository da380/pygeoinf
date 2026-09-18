"""
Drawing fields, as a layer rather than as methods on the spaces.

``plot`` and ``subplots`` dispatch on the space's type, so a space says how to
sample itself and this says how to draw it. Importing this module registers the
renderers; ``matplotlib`` and ``cartopy`` are imported only when a figure is
actually made.

    from pygeoinf2 import plotting

    ax, im = plotting.plot(space, field, symmetric=True, coasts=True)
    ax.set_title("Flexure")
    plotting.show()

A ball has to be cut to be seen: ``plot_shell`` (which is what ``plot`` does
there), ``plot_section`` by any plane through the centre, and
``plot_profile`` along a ray, with ``section_values`` for the numbers and
``plot_section_points`` for the stations in a section's plane. They are here
when ``planetmodel`` is installed.
"""

from .base import (
    color_limits,
    plot,
    plot_balls,
    plot_network,
    plot_paths,
    plot_points,
    show,
    subplots,
)
from .distributions import moments, plot_corner, plot_densities
from . import fourier as _fourier  # noqa: F401  (registers the box renderer)
from .fourier import plot_error_bounds
from .sets import plot_set
from . import sphere as _sphere  # noqa: F401  (registers the sphere renderers)

try:  # The spectral-element spaces need planetmodel, an optional extra.
    from .sem1d import (
        plot_profile,
        plot_section,
        plot_section_points,
        plot_shell,
        section_values,
    )
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "plot_error_bounds",
    "plot",
    "subplots",
    "color_limits",
    "show",
    "plot_points",
    "plot_paths",
    "plot_balls",
    "plot_network",
    "plot_set",
    "plot_densities",
    "plot_corner",
    "moments",
]
if "plot_section" in globals():
    __all__ += [
        "plot_shell",
        "plot_section",
        "plot_section_points",
        "plot_profile",
        "section_values",
    ]
