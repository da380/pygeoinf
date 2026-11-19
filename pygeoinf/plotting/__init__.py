"""
Drawing fields, as a layer rather than as methods on the spaces.

A space says how to sample itself and this says how to draw it. Importing this
module registers the renderers; ``matplotlib`` and ``cartopy`` are imported
only when a figure is actually made.

    from pygeoinf import plotting

    ax, im = plotting.plot(space, field, symmetric=True, coasts=True)
    ax.set_title("Flexure")
    plotting.show()

**One vocabulary, the space deciding the picture.** ``plot``,
``plot_section``, ``plot_profile``, ``plot_points``, ``plot_paths``,
``plot_balls`` and ``subplots`` dispatch on the space's type, so one name
draws one kind of picture on every geometry that has it, and a space with no
renderer for it is refused by name:

* ``plot`` draws a field the way its space is seen: a map on a sphere, a
  line on an interval or a box of one dimension, an image on a box of two or
  a disk, the shell at the surface of a ball, a strip whole over ``x`` and
  depth, a slab's lateral map at its top or at ``depth=``.
* ``plot_section`` draws a field on a plane through a body: a ball's by a
  plane through its centre, a slab's by a vertical plane, and a strip, a
  section already, whole, as its ``plot`` does.
* ``plot_profile`` draws a field along one line: a ball's along a ray,
  against radius; a strip's or a slab's beneath one lateral position,
  against depth.

The keywords that name the cut are the space's own: a ball's section takes a
plane through the centre, by ``longitude=``, ``pole=`` or ``through=``, and
a slab's a vertical plane, by ``x=`` or ``y=``; a ball's profile is at
``latitude, longitude``, a strip's at ``x`` and a slab's at ``x, y``. The
keywords that style a section, ``ax=``, ``cmap=``, ``symmetric=`` and the
rest, are common to its renderers, and so are a profile's, ``ax=`` and
``points=``.

A ball has two more cuts of its own: ``plot_shell`` at any radius, which is
what ``plot`` draws at the surface, and ``plot_cutaway`` with a wedge cut
from it; ``section_values`` gives the numbers of a ball's section and
``plot_section_points`` the stations in its plane, and ``planet_model``
hands a field to ``planetmodel``'s own slices and pictures. A flat layer is
drawn with depth increasing downward. A layered space of balls, strips or
slabs is drawn layer by layer on one color scale, with a line at every
interface and a sharp jump across it; a map at the depth of an interface
takes ``side=``.
"""

from .base import (
    color_limits,
    plot,
    plot_balls,
    plot_network,
    plot_paths,
    plot_points,
    plot_profile,
    plot_section,
    show,
    subplots,
)
from .distributions import moments, plot_corner, plot_densities
from . import fourier as _fourier  # noqa: F401  (registers the box renderer)
from .fourier import plot_error_bounds
from .sets import plot_set
from . import sphere as _sphere  # noqa: F401  (registers the sphere renderers)

from . import interval as _interval  # noqa: F401  (registers the interval renderers)
from . import disk as _disk  # noqa: F401  (registers the disk renderers)
from . import flat as _flat  # noqa: F401  (registers the strip and slab renderers)
from .ball import (
    planet_model,
    plot_cutaway,
    plot_section_points,
    plot_shell,
    section_values,
)

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
    "plot_shell",
    "plot_section",
    "plot_section_points",
    "plot_profile",
    "plot_cutaway",
    "planet_model",
    "section_values",
]
