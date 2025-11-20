"""
Bump function provider for smooth, compactly supported functions.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from scipy import integrate

from .base import ParametricFunctionProvider, IndexedFunctionProvider
from pygeoinf.interval.functions import Function


class BumpFunctionProvider(ParametricFunctionProvider,
                           IndexedFunctionProvider):
    """
    Provider for smooth bump functions with compact support.

    Bump functions are infinitely differentiable (C∞) functions that are
    zero outside a finite interval and positive inside. They use the
    generalized mathematical form: exp(k*t/(t²-1)) where t is a scaled
    coordinate and k is a shape parameter.

    The shape parameter k controls the concentration of the function:
    - k = 1: Standard bump function
    - k > 1: More concentrated near the center
    - k < 1: More spread out (but k > 0 required for convergence)

    All bump functions are automatically normalized so that their integral
    over their compact support equals 1 (unimodular property). The
    normalization constants are computed numerically and cached for
    efficiency.
    """

    def __init__(self, space, default_width: float = 0.2,
                 centers: Optional[np.ndarray] = None, default_k: float = 1.0):
        """
        Initialize bump function provider.

        Args:
            space: Lebesgue instance (contains domain information)
            default_width: Default width for indexed access (as fraction of
                          domain)
            centers: Optional array of centers for indexed access. If provided,
                    get_function_by_index will use these centers directly.
                    If None, centers will be automatically distributed.
            default_k: Parameter controlling the shape - higher k values make
                      the function more concentrated near the center.
        """
        super().__init__(space)
        self.default_width = default_width
        self.centers = np.asarray(centers) if centers is not None else None
        self.default_k = default_k
        self._cache = {}
        # Cache for normalization constants: {k_value: normalization_constant}
        self._normalization_cache = {}

    def _get_normalization_constant(self, k: float, width: float) -> float:
        """
        Get or compute the normalization constant for given k and width.

        The normalization is computed so that the integral over the compact
        support equals 1. For the generalized bump function
        exp(k * t / (t² - 1)) where t = 2(x-center)/width, the normalization
        depends on both k and width.

        Args:
            k: Shape parameter (higher values make function more concentrated)
            width: Width of the compact support

        Returns:
            float: Normalization constant
        """
        if k not in self._normalization_cache:
            # Compute normalization constant numerically for this k value

            def unnormalized_bump(t):
                """Unnormalized bump function on [-1, 1]."""
                if abs(t) >= 1.0:
                    return 0.0
                return np.exp(k * t**2 / (t**2 - 1))

            # Integrate the unnormalized function over [-1, 1]
            integral, _ = integrate.quad(unnormalized_bump, -1, 1)

            # Store the base normalization constant (for width = 2)
            self._normalization_cache[k] = integral

        # The full normalization constant includes the width scaling
        # Since our support is [center - width/2, center + width/2],
        # we need to scale by width/2 to account for coordinate
        # transformation
        return self._normalization_cache[k] * (width / 2)

    def get_function_by_parameters(self, parameters: Dict[str, Any],
                                   **kwargs) -> Function:
        """
        Get a normalized bump function with specific parameters.

        The bump function is normalized so that its integral over its
        compact support equals 1 (unimodular).

        Args:
            parameters: Dictionary containing:
                - 'center': Center of the bump function
                - 'width': Width of the compact support
                - 'k': Shape parameter (optional, defaults to default_k)
        """
        center = parameters['center']
        width = parameters['width']
        k = parameters.get('k', self.default_k)

        # Calculate support interval [a, b]
        a_support = center - width / 2
        b_support = center + width / 2

        # Get or compute normalization constant for this k value
        normalization_constant = self._get_normalization_constant(k, width)

        def normalized_bump_func(x):
            """
            Normalized generalized bump function: exp(k*t/(t²-1)) where t
            is scaled.

            Uses the form exp(k*t/(t²-1)) defined on [-1,1], but transformed
            to have custom center and width.

            The function is defined as:
            - exp(k*t/(t²-1)) for t ∈ (-1,1) where t = 2(x-center)/width
            - 0 for t = ±1 (boundaries)
            - 0 for |t| > 1 (outside support)
            """
            x_arr = np.asarray(x)

            # Transform coordinates: map [a_support, b_support] to [-1, 1]
            # t = 2 * (x - center) / width
            t = 2.0 * (x_arr - center) / width

            # Handle the interior and boundary cases
            result = np.zeros_like(x_arr, dtype=float)

            # Interior condition: |t| < 1 (strictly inside [-1, 1])
            interior_mask = np.abs(t) < 1.0

            # Only compute exponential for interior points
            if np.any(interior_mask):
                t_interior = t[interior_mask]
                if np.isclose(k, 0.0):
                    # If k is zero, the function is constant 1 in the interior
                    result[interior_mask] = 1.0
                else:
                    # Use exp(k*t/(t²-1)) form
                    denominator = t_interior**2 - 1.0
                    result[interior_mask] = np.exp(
                        k * t_interior**2 / denominator
                    )

            # Boundary points (|t| = 1) and exterior points remain zero
            return result / normalization_constant

        return Function(
            self.space,
            evaluate_callable=normalized_bump_func,
            name=(f'bump_normalized_center_{center:.3f}_width_{width:.3f}'
                  f'_k_{k:.3f}'),
            support=(a_support, b_support)  # Use Function's compact support
        )

    def get_function_by_index(self, index: int, k: Optional[float] = None,
                              **kwargs) -> Function:
        """
        Get bump function by index with predetermined centers and widths.

        If centers were provided during initialization, uses those centers
        directly. Otherwise, distributes centers across the domain using
        the default width.

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
            a, b = self.domain.a, self.domain.b
            domain_length = b - a

            if self.centers is not None:
                # Use provided centers
                if index >= len(self.centers):
                    raise IndexError(
                        f"Index {index} out of range for provided centers "
                        f"(length {len(self.centers)})"
                    )
                center = self.centers[index]

                # Validate that the center is within the domain
                if not (a <= center <= b):
                    raise ValueError(
                        f"Center {center} at index {index} is outside "
                        f"domain [{a}, {b}]"
                    )
            else:
                # Distribute centers across the domain (original behavior)
                # Use a pattern that avoids boundary issues
                n_divisions = index + 2  # At least 2 divisions
                center_positions = np.linspace(a + 0.1 * domain_length,
                                               b - 0.1 * domain_length,
                                               n_divisions)
                center = center_positions[index % len(center_positions)]

            # Use default width, but ensure it doesn't exceed domain bounds
            width = min(self.default_width,
                        2 * min(center - a, b - center))

            parameters = {'center': center, 'width': width, 'k': k}
            func = self.get_function_by_parameters(parameters)
            func.name = (f'bump_{index}_center_{center:.3f}_width_{width:.3f}'
                         f'_k_{k:.3f}')

            self._cache[cache_key] = func

        return self._cache[cache_key]

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for bump functions."""
        a, b = self.domain.a, self.domain.b
        domain_length = b - a

        return {
            'center': (a + b) / 2,  # Center of domain
            'width': self.default_width * domain_length,
            'k': self.default_k  # Include default k parameter
        }

    def get_function(self, parameters: Optional[Dict[str, Any]] = None,
                     **kwargs) -> Function:
        """Get bump function with given or default parameters."""
        if parameters is None:
            parameters = self.get_default_parameters()
        return self.get_function_by_parameters(parameters, **kwargs)

    def get_n_functions(self) -> Optional[int]:
        """
        Get the number of available functions.

        Returns:
            int: Number of functions if centers were provided, None otherwise
        """
        return len(self.centers) if self.centers is not None else None

    def get_centers(self) -> Optional[np.ndarray]:
        """
        Get the array of centers used by this provider.

        Returns:
            np.ndarray: Array of centers if provided, None otherwise
        """
        return self.centers.copy() if self.centers is not None else None

    def clear_normalization_cache(self):
        """
        Clear the normalization constant cache.

        This may be useful if you want to free memory or if there are
        numerical precision concerns with cached values.
        """
        self._normalization_cache.clear()

    def get_cached_k_values(self) -> List[float]:
        """
        Get the list of k values for which normalization constants
        have been computed and cached.

        Returns:
            List[float]: List of cached k values
        """
        return list(self._normalization_cache.keys())
