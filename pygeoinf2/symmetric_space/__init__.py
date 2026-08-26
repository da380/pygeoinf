"""Symmetric spaces: homogeneous domains whose basis diagonalises the Laplacian.

The circle, the torus and the N-dimensional periodic box are homogeneous under
a group action; intervals and boxes are built by embedding into them. What they
share is a spectral basis in which the Laplace-Beltrami operator is diagonal,
and every convenience in this package follows from that one fact.

The Fourier-based spaces are always available. The sphere needs ``pyshtools``,
so it is imported on demand rather than here::

    from pygeoinf2.symmetric_space.sphere import Sobolev

See DESIGN.md sections 13 and 19.
"""

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
]
