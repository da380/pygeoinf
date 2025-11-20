"""Sine function provider for Dirichlet boundary conditions."""

import math
import numpy as np
from .base import IndexedFunctionProvider


class SineFunctionProvider(IndexedFunctionProvider):
    """
    Provider for sine functions: sin(kπ(x-a)/L).

    These are the eigenfunctions for Dirichlet boundary conditions
    on the negative Laplacian operator.
    """

    def __init__(self, space):
        """Initialize the sine function provider."""
        super().__init__(space)
        self._cache = {}

    def get_function_by_index(self, index: int):
        """
        Get sine function with index k = index + 1.

        Args:
            index: Index of the sine function (0-based, maps to k=1,2,3,...)

        Returns:
            Function: Normalized sine function
        """
        if index not in self._cache:
            a, b = self.space.function_domain.a, self.space.function_domain.b
            length = b - a
            k = index + 1  # Sine functions start from k=1
            normalization = np.sqrt(2 / length)

            def sine_func(x):
                if isinstance(x, np.ndarray):
                    return normalization * np.sin(
                        k * np.pi * (x - a) / length
                    )
                else:
                    return normalization * math.sin(
                        k * np.pi * (x - a) / length
                    )

            from ..functions import Function
            func = Function(
                self.space,
                evaluate_callable=sine_func,
                name=f"sin({k}π(x-{a})/{length})"
            )
            self._cache[index] = func

        return self._cache[index]
