"""Fourier basis function provider."""

import numpy as np
from .base import IndexedFunctionProvider


class FourierFunctionProvider(IndexedFunctionProvider):
    """Provider for Fourier basis functions."""

    def __init__(self, space, non_constant_only: bool = False):
        """
        Initialize Fourier provider.

        Args:
            space: Lebesgue instance (contains domain information)
            non_constant_only: If True, skip constant function (index starts at 1)
        """
        super().__init__(space)
        self.non_constant_only = non_constant_only

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """
        Get Fourier basis function by index.

        Index 0: constant function
        Odd index (2k-1):  √(2/L) · cos(2πk(x−a)/L)
        Even index (2k):   √(2/L) · sin(2πk(x−a)/L)

        Args:
            index: Index of the Fourier function
            **kwargs: Additional keyword arguments (ignored)

        Returns:
            Function: Fourier basis function
        """
        from ..functions import Function

        a, b = self.domain.a, self.domain.b
        L = b - a

        if self.non_constant_only:
            index += 1

        if index == 0:
            # Constant function
            def const_func(x):
                return np.ones_like(x) / np.sqrt(L)

            return Function(
                self.space,
                evaluate_callable=const_func,
                name='fourier_const'
            )

        k = (index + 1) // 2

        if index % 2 == 1:  # Odd index: cosine
            def cosine_func(x):
                return np.sqrt(2/L) * np.cos(2 * k * np.pi * (x - a) / L)

            return Function(
                self.space,
                evaluate_callable=cosine_func,
                name=f'fourier_cos_{k}'
            )
        else:  # Even index: sine
            def sine_func(x):
                return np.sqrt(2/L) * np.sin(2 * k * np.pi * (x - a) / L)

            return Function(
                self.space,
                evaluate_callable=sine_func,
                name=f'fourier_sin_{k}'
            )
