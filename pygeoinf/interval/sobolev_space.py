"""
Sobolev spaces on a segment/interval [a, b].
"""

from typing import TYPE_CHECKING, Optional, Union
from pygeoinf import MassWeightedHilbertSpace
from .functions import Function
import numpy as np

# Import types for annotations but avoid runtime circular import
if TYPE_CHECKING:
    from pygeoinf.interval import Lebesgue, SpectralOperator, IntervalDomain


class Sobolev(MassWeightedHilbertSpace):
    def __init__(
        self,
        dim: int,
        function_domain: "IntervalDomain",
        s: float,
        k: float,
        L: "SpectralOperator",
        /, *,
        basis: Optional[Union[str, list]] = None,
    ):

        # Attributes if Sobolev space
        self._underlying_space = self._create_underlying_space(dim, function_domain, basis)
        self._s = s
        self._k = k
        self._L = L

        self._dofs = L._dofs

        # Atributes borrowed from Lebesgue
        self._function_domain = self._underlying_space.function_domain

        # Import at runtime to avoid circular import
        from pygeoinf.interval.operators import (
            BesselSobolev, BesselSobolevInverse
        )

        # Create the mass operator M = (k^2 I + A)^s
        M_op = BesselSobolev(
            self._underlying_space,
            self._underlying_space,
            k=self._k,
            s=2 * self._s,
            L=L,
            dofs=self._dofs
        )
        M_op_inv = BesselSobolevInverse(
            self._underlying_space,
            self._underlying_space,
            k=self._k,
            s=2 * self._s,
            L=L,
            dofs=self._dofs
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

    def to_dual(self, x: 'Function') -> 'LinearFormSobolev':
        from .linear_form_lebesgue import LinearFormSobolev
        if not isinstance(x, Function):
            raise TypeError("Expected Function for primal element")
        kernel = self._mass_operator(x)
        return LinearFormSobolev(self, kernel=kernel)

    def from_dual(self, xp: 'LinearFormSobolev') -> 'Function':
        x = self._inverse_mass_operator(xp.kernel)
        return x

    def _create_underlying_space(
        self,
        dim,
        function_domain,
        basis,
    ) -> 'Lebesgue':
        from .lebesgue_space import Lebesgue
        return Lebesgue(
            dim,
            function_domain,
            basis=basis
        )