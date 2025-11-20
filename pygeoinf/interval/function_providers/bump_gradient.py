"""Bump function gradient provider."""

import numpy as np
from typing import Dict, Any, Optional, List
from .base import ParametricFunctionProvider, IndexedFunctionProvider
from .bump import BumpFunctionProvider


class BumpFunctionGradientProvider(ParametricFunctionProvider,
                                   IndexedFunctionProvider):
    """
    Provider for analytical gradients of smooth bump functions.

    This class provides the exact analytical derivatives of bump functions
    from BumpFunctionProvider. It leverages the BumpFunctionProvider for
    the base bump function computation and applies the analytical derivative
    formula:

    f'(x) = f(x) * k * (-2t)/(t²-1)² * (2/width)

    where f(x) is the bump function and t = 2(x-center)/width.
    """

    def __init__(self, space, default_width: float = 0.2,
                 default_k: float = 1.0,
                 centers: Optional[List[float]] = None):
        """
        Initialize the bump function gradient provider.

        Args:
            space: Function space for the gradients
            default_width: Default width for bump functions
            default_k: Default shape parameter k
            centers: Optional list of predetermined centers
        """
        super().__init__(space)
        self.default_width = default_width
        self.default_k = default_k
        self.centers = centers
        self._domain = space.function_domain
        self._cache = {}

        # Create a BumpFunctionProvider to reuse its functionality
        # Convert centers to numpy array if provided
        centers_array = None
        if centers is not None:
            centers_array = np.array(centers)

        self._bump_provider = BumpFunctionProvider(
            space, default_width=default_width,
            centers=centers_array, default_k=default_k
        )

    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default parameters for bump function gradient.

        Returns:
            Dictionary with default center, width, and k parameters
        """
        # Use the middle of the domain as default center
        a, b = self._domain.a, self._domain.b
        default_center = (a + b) / 2.0

        return {
            'center': default_center,
            'width': self.default_width,
            'k': self.default_k
        }

    def get_function_by_parameters(self, parameters: Dict[str, Any],
                                   **kwargs) -> 'Function':
        """
        Get the analytical gradient of a bump function with parameters.

        Args:
            parameters: Dictionary containing:
                - 'center': Center of the bump function
                - 'width': Width of the compact support
                - 'k': Shape parameter (optional, defaults to default_k)
        """
        from ..functions import Function

        center = parameters['center']
        width = parameters['width']
        k = parameters.get('k', self.default_k)

        # Get the corresponding bump function
        bump_func = self._bump_provider.get_function_by_parameters(parameters)

        # Calculate support interval [a, b] (same as bump function)
        a_support = center - width / 2
        b_support = center + width / 2

        def bump_gradient_func(x):
            """
            Analytical gradient using the bump function and derivative factor.

            The gradient is: f'(x) = f(x) * derivative_factor(x)
            where derivative_factor(x) = k * (-2t)/(t²-1)² * (2/width)
            """
            x_arr = np.asarray(x)

            # Get the bump function values
            bump_values = bump_func(x_arr)

            # Transform coordinates: t = 2 * (x - center) / width
            t = 2.0 * (x_arr - center) / width

            # Initialize result array
            result = np.zeros_like(x_arr, dtype=float)

            # Interior condition: |t| < 1 (where bump function is non-zero)
            interior_mask = np.abs(t) < 1.0

            if np.any(interior_mask):
                t_interior = t[interior_mask]

                if np.isclose(k, 0.0):
                    # If k is zero, gradient is zero everywhere
                    result[interior_mask] = 0.0
                else:
                    # Compute the derivative factor
                    # d/dt[exp(k*t²/(t²-1))] = exp(...) * k * (-2t)/(t²-1)²
                    # d/dx = d/dt * dt/dx = d/dt * (2/width)
                    denominator = t_interior**2 - 1.0
                    derivative_factor = (k * (-2 * t_interior) /
                                         (denominator**2) * (2.0 / width))

                    # Apply the product rule: f'(x) = f(x) * factor
                    # Ensure bump_values is array-like for consistent indexing
                    bump_vals_array = np.asarray(bump_values)
                    if bump_vals_array.ndim == 0:
                        # If bump_values is scalar, broadcast to match shape
                        bump_vals_interior = np.full(len(t_interior),
                                                     float(bump_vals_array))
                    else:
                        bump_vals_interior = bump_vals_array[interior_mask]

                    result[interior_mask] = (bump_vals_interior *
                                             derivative_factor)

            return result

        return Function(
            self.space,
            evaluate_callable=bump_gradient_func,
            name=(f'bump_gradient_center_{center:.3f}_width_{width:.3f}'
                  f'_k_{k:.3f}'),
            support=(a_support, b_support)
        )

    def get_function_by_index(self, index: int, k: Optional[float] = None,
                              **kwargs) -> 'Function':
        """
        Get bump function gradient by index.

        Args:
            index: Index of the bump function (must be >= 0)
            k: Shape parameter (optional, defaults to default_k)
        """
        if index < 0:
            raise ValueError(f"Index must be non-negative, got {index}")

        # Use provided k or default
        if k is None:
            k = self.default_k

        # Create cache key that includes k parameter
        cache_key = (index, k)

        if cache_key not in self._cache:
            a, b = self._domain.a, self._domain.b
            domain_length = b - a

            if self.centers is not None:
                # Use predetermined centers
                if index >= len(self.centers):
                    raise IndexError(
                        f"Index {index} exceeds number of centers "
                        f"{len(self.centers)}"
                    )
                center = self.centers[index]
                width = self.default_width
            else:
                # Distribute centers across domain
                n_bumps_estimate = max(
                    1, int(domain_length / self.default_width)
                )
                if index >= n_bumps_estimate:
                    # For indices beyond estimate, space them further
                    extra_spacing = (index - n_bumps_estimate + 1) * 0.1
                    center = a + (index * self.default_width) + extra_spacing
                else:
                    center = (a + (index + 0.5) * domain_length /
                              n_bumps_estimate)
                width = self.default_width

            # Ensure the bump is within domain bounds
            if center < a or center > b:
                raise ValueError(
                    f"Bump center {center} is outside domain [{a}, {b}]"
                )

            # Create the gradient function
            gradient_func = self.get_function_by_parameters({
                'center': center,
                'width': width,
                'k': k
            })

            self._cache[cache_key] = gradient_func

        return self._cache[cache_key]

    @property
    def num_functions(self):
        """Return estimated number of functions."""
        if self.centers is not None:
            return len(self.centers)
        else:
            a, b = self._domain.a, self._domain.b
            domain_length = b - a
            return max(1, int(domain_length / self.default_width))
