"""Mixed Dirichlet-Neumann and Neumann-Dirichlet function providers."""

import math
import numpy as np
from .base import IndexedFunctionProvider


class MixedDNFunctionProvider(IndexedFunctionProvider):
    """
    Mixed DN eigenfunctions for -d2/dx2 on (a,b):
      u(a)=0, u'(b)=0  ⇒  φ_k(x) = √(2/L) · sin( (k+1/2)π (x-a)/L ),  k≥0
    """
    def __init__(self, space):
        super().__init__(space)
        self._cache = {}

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        if index < 0:
            raise ValueError("index must be ≥ 0")
        if index not in self._cache:
            from ..functions import Function
            a, b = self.domain.a, self.domain.b
            L = b - a
            mu = (index + 0.5) * math.pi / L
            c = math.sqrt(2.0 / L)

            def phi(x):
                y = np.asarray(x) - a
                return c * np.sin(mu * y)

            func = Function(self.space, evaluate_callable=phi,
                            name=f"mixed_DN_k{index}")
            self._cache[index] = func
        return self._cache[index]


class MixedNDFunctionProvider(IndexedFunctionProvider):
    """
    Mixed ND eigenfunctions for -d2/dx2 on (a,b):
      u'(a)=0, u(b)=0  ⇒  φ_k(x) = √(2/L) · cos( (k+1/2)π (x-a)/L ),  k≥0
    """
    def __init__(self, space):
        super().__init__(space)
        self._cache = {}

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        if index < 0:
            raise ValueError("index must be ≥ 0")
        if index not in self._cache:
            from ..functions import Function
            a, b = self.domain.a, self.domain.b
            L = b - a
            mu = (index + 0.5) * math.pi / L
            c = math.sqrt(2.0 / L)

            def phi(x):
                y = np.asarray(x) - a
                return c * np.cos(mu * y)

            func = Function(self.space, evaluate_callable=phi,
                            name=f"mixed_ND_k{index}")
            self._cache[index] = func
        return self._cache[index]
