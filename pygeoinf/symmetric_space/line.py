"""
Provides concrete implementations of function spaces on the Line (R¹).

This module uses the abstract framework from the symmetric space module to create
fully-featured `Lebesgue` (L²) and `Sobolev` (Hˢ) Hilbert spaces for functions
defined on a line.
"""

from __future__ import annotations

from typing import Callable, Tuple, Optional, Any, List
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, irfft
from scipy.sparse import diags


from matplotlib.figure import Figure
from matplotlib.axes import Axes

from pygeoinf.hilbert_space import (
    HilbertModule,
    MassWeightedHilbertModule,
)
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.linear_forms import LinearForm
from .symmetric_space import (
    AbstractInvariantLebesgueSpace,
    AbstractInvariantSobolevSpace,
)
from .circle import Lebesgue as CircleLebesgue


class LineHelper:
    pass


class Lebesgue(LineHelper, HilbertModule, AbstractInvariantLebesgueSpace):

    def __init__(self, kmax: int, a: float, b: float, delta: float):

        self._circle_space = CircleLebesgue(kmax)

    @property
    def spatial_dimension(self):
        """The dimension of the symetric space."""
        return 1

    @property
    def dim(self):
        """The dimension of the Hilbert space."""
        return self._circle_space.dim
