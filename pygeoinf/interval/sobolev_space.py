"""
Sobolev spaces on a segment/interval [a, b].
"""

from typing import TYPE_CHECKING
from pygeoinf import MassWeightedHilbertSpace
from .functions import Function
import numpy as np

# Import types for annotations but avoid runtime circular import
if TYPE_CHECKING:
    from pygeoinf.interval import Lebesgue, SpectralOperator


class Sobolev(MassWeightedHilbertSpace):
    def __init__(
        self,
        underlying_space: "Lebesgue",
        s: float,
        k: float,
        L: "SpectralOperator",
        /,
        *,
        dofs: int = 100
    ):

        # Attributes if Sobolev space
        self._s = s
        self._k = k
        self._L = L
        self._underlying_space = underlying_space
        self._dofs = dofs

        # Atributes borrowed from Lebesgue
        self._function_domain = underlying_space.function_domain

        # Import at runtime to avoid circular import
        from pygeoinf.interval.operators import (
            BesselSobolev, BesselSobolevInverse
        )

        # Create the mass operator M = (k^2 I + A)^s
        M_op = BesselSobolev(
            self._underlying_space,
            self._underlying_space,
            k=self._k,
            s=self._s,
            L=L,
            dofs=dofs
        )
        M_op_inv = BesselSobolevInverse(
            self._underlying_space,
            self._underlying_space,
            k=self._k,
            s=self._s,
            L=L,
            dofs=dofs
        )
        # Compute the inverse mass operator
        super().__init__(
            underlying_space,
            M_op,
            M_op_inv
        )

    @property
    def function_domain(self):
        return self._function_domain

    @property
    def zero(self):
        return Function(self, evaluate_callable=lambda x: np.zeros_like(x))
