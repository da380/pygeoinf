"""Box-car (rectangular/step) function provider."""

import numpy as np
from typing import Dict, Any, Optional, List
from .base import ParametricFunctionProvider, IndexedFunctionProvider


class BoxCarFunctionProvider(ParametricFunctionProvider,
                             IndexedFunctionProvider):
    """
    Provider for box-car (rectangular/step) functions.

    Box-car functions are piecewise constant functions that are zero outside
    a finite interval and have a constant value inside. They are useful for
    modeling step functions, rectangular windows, and discontinuous phenomena.

    The functions have the form:
    f(x) = height for x ∈ [center - width/2, center + width/2]
    f(x) = 0 elsewhere

    By default, functions are normalized so that their integral equals 1.
    """

    def __init__(self, space, default_width: float = 0.2,
                 centers: Optional[np.ndarray] = None,
                 default_height: float = 1.0,
                 normalize: bool = True):
        """
        Initialize box-car function provider.

        Args:
            space: Lebesgue instance (contains domain information)
            default_width: Default width for indexed access (as fraction of
                          domain)
            centers: Optional array of centers for indexed access. If provided,
                    get_function_by_index will use these centers directly.
                    If None, centers will be automatically distributed.
            default_height: Default height of the box-car function
            normalize: If True, normalize so integral equals 1
        """
        super().__init__(space)
        self.default_width = default_width
        self.centers = np.asarray(centers) if centers is not None else None
        self.default_height = default_height
        self.normalize = normalize
        self._cache = {}

    def get_function_by_parameters(self, parameters: Dict[str, Any],
                                   **kwargs) -> 'Function':
        """
        Get a box-car function with specific parameters.

        Args:
            parameters: Dictionary containing:
                - 'center': Center of the box-car function
                - 'width': Width of the box-car function
                - 'height': Height of the box-car function (optional)
                - 'normalize': Whether to normalize (optional)
        """
        from ..functions import Function

        center = parameters['center']
        width = parameters['width']
        height = parameters.get('height', self.default_height)
        normalize = parameters.get('normalize', self.normalize)

        # Calculate support interval [a, b]
        a_support = center - width / 2
        b_support = center + width / 2

        # Calculate normalization factor if needed
        if normalize:
            # For normalized box-car: integral = height * width = 1
            # So height = 1 / width
            actual_height = 1.0 / width
        else:
            actual_height = height

        def boxcar_func(x):
            """
            Box-car function: constant value inside support, zero outside.
            """
            x_arr = np.asarray(x)
            result = np.zeros_like(x_arr, dtype=float)

            # Box-car is non-zero in [center - width/2, center + width/2]
            mask = (x_arr >= a_support) & (x_arr <= b_support)
            result[mask] = actual_height

            return result

        name_suffix = "_normalized" if normalize else f"_height_{height:.3f}"
        return Function(
            self.space,
            evaluate_callable=boxcar_func,
            name=(f'boxcar_center_{center:.3f}_width_{width:.3f}'
                  f'{name_suffix}'),
            support=(a_support, b_support)
        )

    def get_function_by_index(self, index: int,
                              height: Optional[float] = None,
                              normalize: Optional[bool] = None,
                              **kwargs) -> 'Function':
        """
        Get box-car function by index with predetermined centers and widths.

        If centers were provided during initialization, uses those centers
        directly. Otherwise, distributes centers across the domain using
        the default width.

        Args:
            index: Index of the box-car function (must be >= 0)
            height: Height of the box-car (optional, defaults to
                   default_height)
            normalize: Whether to normalize (optional, defaults to
                      self.normalize)
        """
        if index < 0:
            raise ValueError(f"Index must be non-negative, got {index}")

        # Use provided parameters or defaults
        if height is None:
            height = self.default_height
        if normalize is None:
            normalize = self.normalize

        # Create cache key that includes height and normalize parameters
        cache_key = (index, height, normalize)

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
                # Distribute centers across the domain
                # Use a pattern that avoids boundary issues
                n_divisions = index + 2  # At least 2 divisions
                center_positions = np.linspace(a + 0.1 * domain_length,
                                               b - 0.1 * domain_length,
                                               n_divisions)
                center = center_positions[index % len(center_positions)]

            # Use default width, but ensure it doesn't exceed domain bounds
            width = min(self.default_width * domain_length,
                        2 * min(center - a, b - center))

            parameters = {
                'center': center,
                'width': width,
                'height': height,
                'normalize': normalize
            }
            func = self.get_function_by_parameters(parameters)

            # Update function name to include index
            if normalize:
                name_suffix = "_normalized"
            else:
                name_suffix = f"_height_{height:.3f}"
            func.name = (f'boxcar_{index}_center_{center:.3f}_'
                         f'width_{width:.3f}{name_suffix}')

            self._cache[cache_key] = func

        return self._cache[cache_key]

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for box-car functions."""
        a, b = self.domain.a, self.domain.b
        domain_length = b - a

        return {
            'center': (a + b) / 2,  # Center of domain
            'width': self.default_width * domain_length,
            'height': self.default_height,
            'normalize': self.normalize
        }

    def get_function(self, parameters: Optional[Dict[str, Any]] = None,
                     **kwargs) -> 'Function':
        """Get box-car function with given or default parameters."""
        if parameters is None:
            parameters = self.get_default_parameters()
        return self.get_function_by_parameters(parameters, **kwargs)

    def get_n_functions(self) -> Optional[int]:
        """
        Get the number of available functions.

        Returns:
            int: Number of functions if centers were provided, None otherwise
        """
        if self.centers is not None:
            return len(self.centers)
        return None
