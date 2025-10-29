"""
Sobolev spaces on a segment/interval [a, b].
"""

from typing import TYPE_CHECKING, Optional, Union, List
from pygeoinf import MassWeightedHilbertSpace, HilbertSpaceDirectSum
from .functions import Function
from .linear_form_lebesgue import LinearFormKernel

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
            self._underlying_space,
            M_op,
            M_op_inv
        )

    @property
    def function_domain(self):
        return self._function_domain

    @property
    def zero(self):
        return Function(self, evaluate_callable=lambda x: np.zeros_like(x))

    def to_dual(self, x: 'Function') -> 'LinearFormKernel':
        if not isinstance(x, Function):
            raise TypeError("Expected Function for primal element")
        kernel = self._mass_operator(x)
        return LinearFormKernel(self, kernel=kernel)

    def from_dual(self, xp: 'LinearFormKernel') -> 'Function':
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

    @property
    def mass_operator_factor(self):
        from .operators import BesselSobolev
        return BesselSobolev(
            self._underlying_space,
            self._underlying_space,
            k=self._k,
            s=self._s,
            L=self._L,
            dofs=self._dofs
        )

    @property
    def inverse_mass_operator_factor(self):
        from .operators import BesselSobolevInverse
        return BesselSobolevInverse(
            self._underlying_space,
            self._underlying_space,
            k=self._k,
            s=self._s,
            L=self._L,
            dofs=self._dofs
        )


class SobolevSpaceDirectSum(HilbertSpaceDirectSum):
    """
    Direct sum of Sobolev spaces using LinearFormKernel for basis-free operations.

    This class extends HilbertSpaceDirectSum to work with Sobolev spaces without
    requiring explicit basis functions. It uses LinearFormKernel which computes
    inner products via integration rather than component-based dot products.
    """

    def to_dual(self, xs: List[Function]) -> LinearFormKernel:
        """
        Maps a list of functions to a dual element using LinearFormKernel.

        For Sobolev spaces, this applies the mass operator to each component
        and wraps the result in a LinearFormKernel that uses integration
        instead of component-based operations.

        Args:
            xs: List of Function objects, one for each subspace

        Returns:
            LinearFormKernel that can evaluate inner products via integration
        """
        if len(xs) != self.number_of_subspaces:
            raise ValueError("Input list has incorrect number of vectors.")

        # Apply to_dual on each subspace (applies mass operator for Sobolev)
        # This returns a list of LinearFormKernel objects with mass-weighted kernels
        kernels = [space.to_dual(x).kernel for space, x in zip(self._spaces, xs)]

        return LinearFormKernel(self, kernel=kernels)

    def from_dual(self, xp: LinearFormKernel) -> List[Function]:
        """
        Maps a dual element back to a list of functions.

        Args:
            xp: LinearFormKernel containing kernel functions

        Returns:
            List of Function objects
        """
        # Handle both LinearFormKernel (specific) and generic LinearForm
        if isinstance(xp, LinearFormKernel):
            # The kernel is a list of mass-weighted functions
            # Apply from_dual to each to get back the original functions
            if isinstance(xp.kernel, list):
                return [space.from_dual(LinearFormKernel(space, kernel=k))
                        for space, k in zip(self._spaces, xp.kernel)]
            else:
                raise ValueError("Expected kernel to be a list for direct sum")
        else:
            # Delegate to base class for generic LinearForm objects
            return super().from_dual(xp)
