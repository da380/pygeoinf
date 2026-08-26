"""Concrete Hilbert spaces of fields.

The Fourier-based spaces are always available. The sphere needs ``pyshtools``,
so it is imported on demand rather than here: ``from pygeoinf2.spaces.sphere
import Sobolev``.
"""

from .box import Box, Interval
from .fourier import Lebesgue, PeriodicBox, Sobolev
from .invariant import InvariantSpace, lift_formal_adjoint

__all__ = [
    "Box",
    "Interval",
    "InvariantSpace",
    "Lebesgue",
    "PeriodicBox",
    "Sobolev",
    "lift_formal_adjoint",
]
