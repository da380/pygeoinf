"""Discontinuous function provider."""

import numpy as np
from typing import Optional, Tuple
from .base import RandomFunctionProvider


class DiscontinuousFunctionProvider(RandomFunctionProvider):
    """Provider for functions with random discontinuities."""

    def __init__(self, space, random_state=None):
        """
        Initialize discontinuous function provider.

        Args:
            space: Lebesgue instance (contains domain information)
            random_state: Random seed for reproducibility
        """
        super().__init__(space, random_state)

    def sample_function(self,
                        n_discontinuities: Optional[int] = None,
                        jump_range: Tuple[float, float] = (-1, 1),
                        **kwargs) -> 'Function':
        """
        Sample a function with random discontinuities.

        Args:
            n_discontinuities: Number of discontinuities (random if None)
            jump_range: Range for jump sizes
        """
        from ..functions import Function

        a, b = self.domain.a, self.domain.b

        if n_discontinuities is None:
            n_discontinuities = self.rng.randint(1, 6)

        # Random discontinuity locations
        disc_locations = self.rng.uniform(a, b, n_discontinuities)
        disc_locations.sort()

        # Random jump sizes
        jumps = self.rng.uniform(
            jump_range[0], jump_range[1], n_discontinuities
        )

        def discontinuous_func(x):
            x_arr = np.asarray(x)
            result = np.zeros_like(x_arr, dtype=float)

            # Add jumps at discontinuities
            for loc, jump in zip(disc_locations, jumps):
                result[x_arr >= loc] += jump

            return result

        return Function(
            self.space,
            evaluate_callable=discontinuous_func,
            name=f'discontinuous_{n_discontinuities}'
        )
