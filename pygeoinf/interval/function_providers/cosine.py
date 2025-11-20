"""Cosine function provider for Neumann boundary conditions."""

import math
import numpy as np
from .base import IndexedFunctionProvider


class CosineFunctionProvider(IndexedFunctionProvider):
    """
    Provider for cosine functions: cos(kπ(x-a)/L).

    These are the eigenfunctions for Neumann boundary conditions
    on the negative Laplacian operator (with constant mode for k=0).
    """

    def __init__(self, space, non_constant_only: bool = False):
        """
        Initialize the cosine function provider.

        Args:
            space: Lebesgue instance (contains domain information)
            non_constant_only: If True, skip constant mode (start at k=1)
        """
        super().__init__(space)
        self._cache = {}
        self.non_constant_only = non_constant_only

    def get_function_by_index(self, index: int):
        """
        Get cosine function or constant for index 0.

        Args:
            index: Index of the cosine function (0 = constant if not skipped)

        Returns:
            Function: Normalized cosine function or constant
        """
        # If non_constant_only is True, shift index by 1
        if self.non_constant_only:
            index += 1

        if index not in self._cache:
            a, b = self.space.function_domain.a, self.space.function_domain.b
            length = b - a

            if index == 0:
                # Constant mode for Neumann BC
                def constant_func(x):
                    return (np.ones_like(x) if isinstance(x, np.ndarray)
                            else 1.0) / np.sqrt(length)

                from ..functions import Function
                func = Function(
                    self.space,
                    evaluate_callable=constant_func,
                    name="1 (constant)"
                )
            else:
                # Cosine modes for k = index
                k = index
                normalization = np.sqrt(2 / length)

                def cosine_func(x):
                    if isinstance(x, np.ndarray):
                        return normalization * np.cos(
                            k * np.pi * (x - a) / length
                        )
                    else:
                        return normalization * math.cos(
                            k * np.pi * (x - a) / length
                        )

                from ..functions import Function
                func = Function(
                    self.space,
                    evaluate_callable=cosine_func,
                    name=f"cos({k}π(x-{a})/{length})"
                )

            self._cache[index] = func

        return self._cache[index]
