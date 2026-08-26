"""
Drawing fields, as a layer rather than as methods on the spaces.

``plot`` and ``subplots`` dispatch on the space's type, so a space says how to
sample itself and this says how to draw it. Importing this module registers the
renderers; ``matplotlib`` and ``cartopy`` are imported only when a figure is
actually made.

    from pygeoinf2 import plotting

    ax, im = plotting.plot(space, field, symmetric=True, coasts=True)
    ax.set_title("Flexure")

See DESIGN.md section 20.5, O8.
"""

from .base import colour_limits, plot, subplots
from . import fourier as _fourier  # noqa: F401  (registers the box renderer)
from . import sphere as _sphere  # noqa: F401  (registers the sphere renderer)

__all__ = ["plot", "subplots", "colour_limits"]
