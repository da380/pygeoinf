"""Symmetric spaces: homogeneous domains whose basis diagonalises the Laplacian.

The circle, the torus and the N-dimensional periodic box are homogeneous under
a group action; intervals, rectangles and boxes are built by embedding into
them. What they share is a spectral basis in which the Laplace-Beltrami
operator is diagonal, and every convenience in this package follows from that
one fact.

**One submodule per geometry**, each exporting ``Lebesgue`` and ``Sobolev`` as
classes rather than factory functions, so that ``isinstance(x, Sobolev)``
answers what it looks like it answers and ``type(x).__name__`` names the
geometry a field lives on::

    from pygeoinf2.symmetric_space.sphere import Sobolev
    from pygeoinf2.symmetric_space.circle import Lebesgue

The available geometries are :mod:`~pygeoinf2.symmetric_space.sphere`,
:mod:`~pygeoinf2.symmetric_space.circle`,
:mod:`~pygeoinf2.symmetric_space.torus`,
:mod:`~pygeoinf2.symmetric_space.line`,
:mod:`~pygeoinf2.symmetric_space.plane` and
:mod:`~pygeoinf2.symmetric_space.box`. The names exported here without
qualification are the N-dimensional periodic box's, which is the general case
the periodic geometries specialise.

The Fourier-based spaces are always available. The sphere needs ``pyshtools``,
so its module is imported on demand rather than here.

See DESIGN.md sections 13 and 19.
"""

from . import box, circle, line, plane, torus
from .base import SymmetricSpace, lift_formal_adjoint
from .box import Box, Interval
from .fourier import Lebesgue, PeriodicBox, Sobolev

__all__ = [
    "Box",
    "Interval",
    "Lebesgue",
    "PeriodicBox",
    "Sobolev",
    "SymmetricSpace",
    "lift_formal_adjoint",
    "box",
    "circle",
    "line",
    "plane",
    "torus",
]
