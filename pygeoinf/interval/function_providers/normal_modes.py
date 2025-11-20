"""
Normal modes provider for generating Gaussian-modulated trigonometric functions.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from .base import (
    RandomFunctionProvider,
    ParametricFunctionProvider,
    IndexedFunctionProvider
)
from pygeoinf.interval.functions import Function


class NormalModesProvider(RandomFunctionProvider, ParametricFunctionProvider,
                          IndexedFunctionProvider):
    """
    Provider for random trigonometric functions modulated by Gaussians.

    Generates functions as linear combinations of sine functions with random
    frequencies, each modulated by a Gaussian envelope with random center
    and width.

    When a random_state is provided, this provider can generate a deterministic
    sequence of functions that can be accessed by index, making it function as
    a RandomFunctionProvider, ParametricFunctionProvider, and
    IndexedFunctionProvider simultaneously.
    """

    def __init__(self, space, random_state: Optional[int] = None,
                 n_modes_range: Tuple[int, int] = (3, 8),
                 coeff_range: Tuple[float, float] = (-2.0, 2.0),
                 freq_range: Tuple[float, float] = (0.5, 10.0),
                 gaussian_width_percent_range: Tuple[float, float] = (10.0,
                                                                      50.0)):
        """
        Initialize with space and random parameters.

        Args:
            space: Lebesgue instance (contains domain information)
            random_state: Random seed for reproducibility
            n_modes_range: (min, max) number of sine functions to combine
            coeff_range: (min, max) range for linear combination coefficients
            freq_range: (min, max) range for sine function frequencies
            gaussian_width_percent_range: (min, max) percentage of interval
                                        length for Gaussian width
        """
        super().__init__(space, random_state)
        self.n_modes_range = n_modes_range
        self.coeff_range = coeff_range
        self.freq_range = freq_range
        self.gaussian_width_percent_range = gaussian_width_percent_range

        # Store the original seed for deterministic indexed access
        self._original_seed = random_state
        if random_state is not None:
            self._base_seed = random_state
        else:
            # Use a default base seed if none provided
            self._base_seed = 12345

    def sample_function(self, **kwargs) -> Function:
        """
        Sample a random trigonometric function modulated by Gaussians.

        Returns:
            Function: Random function combining multiple sine functions with
                     Gaussian envelopes
        """
        a, b = self.domain.a, self.domain.b
        interval_length = b - a

        # Random number of modes
        n_modes = self.rng.randint(self.n_modes_range[0],
                                   self.n_modes_range[1] + 1)

        # Generate random parameters for each mode
        coefficients = self.rng.uniform(self.coeff_range[0],
                                        self.coeff_range[1], n_modes)
        frequencies = self.rng.uniform(self.freq_range[0],
                                       self.freq_range[1], n_modes)

        # Single random Gaussian center (anywhere in the interval)
        gaussian_center = self.rng.uniform(a, b)

        # Single random Gaussian width (as percentage of interval length)
        width_percentage = self.rng.uniform(
            self.gaussian_width_percent_range[0],
            self.gaussian_width_percent_range[1]
        )
        gaussian_width = (width_percentage / 100.0) * interval_length

        def combined_func(x):
            x_arr = np.asarray(x)

            # First compute the linear combination of sine functions
            trig_combination = np.zeros_like(x_arr, dtype=float)
            for i in range(n_modes):
                # Sine function with random frequency
                sine_part = np.sin(2 * np.pi * frequencies[i] *
                                   (x_arr - a) / interval_length)
                # Add to linear combination with random coefficient
                trig_combination += coefficients[i] * sine_part

            # Then modulate the entire combination with a single Gaussian
            gaussian_envelope = np.exp(-0.5 * ((x_arr - gaussian_center) /
                                               gaussian_width)**2)

            result = trig_combination * gaussian_envelope
            return result

        return Function(
            self.space,
            evaluate_callable=combined_func,
            name=f'gaussian_modulated_trig_{n_modes}_modes'
        )

    def get_function_by_index(self, index: int, **kwargs) -> Function:
        """
        Get a deterministic function by index.

        This uses the provider's original random state to generate a
        reproducible sequence of functions, where each index corresponds to a
        unique function determined by the initial random_state. The indexed
        functions are completely independent of any calls to sample_function().

        Args:
            index: Index of the function to generate (must be >= 0)

        Returns:
            Function: Deterministically generated function for this index
        """
        if index < 0:
            raise ValueError(f"Index must be non-negative, got {index}")

        # Create a deterministic RNG state for this specific index
        # Use a hash-like approach to ensure good distribution
        base_seed = self._base_seed

        # Create index-specific seed by combining base seed with index
        index_seed = (base_seed + index * 1000003) % (2**31 - 1)
        index_rng = np.random.RandomState(index_seed)

        # Create a temporary provider with the index-specific RNG
        # We need to temporarily replace our RNG for the generation
        a, b = self.domain.a, self.domain.b
        interval_length = b - a

        # Use the index-specific RNG for all random choices
        n_modes = index_rng.randint(self.n_modes_range[0],
                                    self.n_modes_range[1] + 1)

        # Generate random parameters for each mode
        coefficients = index_rng.uniform(self.coeff_range[0],
                                         self.coeff_range[1], n_modes)
        frequencies = index_rng.uniform(self.freq_range[0],
                                        self.freq_range[1], n_modes)

        # Single random Gaussian center (anywhere in the interval)
        gaussian_center = index_rng.uniform(a, b)

        # Single random Gaussian width (as percentage of interval length)
        width_percentage = index_rng.uniform(
            self.gaussian_width_percent_range[0],
            self.gaussian_width_percent_range[1]
        )
        gaussian_width = (width_percentage / 100.0) * interval_length

        def combined_func(x):
            x_arr = np.asarray(x)

            # First compute the linear combination of sine functions
            trig_combination = np.zeros_like(x_arr, dtype=float)
            for i in range(n_modes):
                # Sine function with random frequency
                sine_part = np.sin(2 * np.pi * frequencies[i] *
                                   (x_arr - a) / interval_length)
                # Add to linear combination with random coefficient
                trig_combination += coefficients[i] * sine_part

            # Then modulate the entire combination with a single Gaussian
            gaussian_envelope = np.exp(-0.5 * ((x_arr - gaussian_center) /
                                               gaussian_width)**2)

            result = trig_combination * gaussian_envelope
            return result

        function = Function(
            self.space,
            evaluate_callable=combined_func,
            name=f'normal_mode_{index}_gaussian_modulated_trig_{n_modes}_modes'
        )

        return function

    def get_indexed_functions(self, n_functions: int,
                              **kwargs) -> List[Function]:
        """
        Get a deterministic sequence of n functions.

        This is a convenience method for generating multiple indexed functions
        at once, which is useful for SOLA operators and other applications
        that need a fixed set of projection functions.

        Args:
            n_functions: Number of functions to generate

        Returns:
            List[Function]: List of n deterministically generated functions
        """
        return [self.get_function_by_index(i, **kwargs)
                for i in range(n_functions)]

    def get_function_by_parameters(self, parameters: Dict[str, Any],
                                   **kwargs) -> Function:
        """
        Get a function with specific parameters.

        Args:
            parameters: Dictionary containing 'coefficients', 'frequencies',
                       'gaussian_center', and 'gaussian_width'
        """
        a, b = self.domain.a, self.domain.b
        interval_length = b - a

        coefficients = parameters['coefficients']
        frequencies = parameters['frequencies']
        gaussian_center = parameters['gaussian_center']
        gaussian_width = parameters['gaussian_width']

        n_modes = len(coefficients)

        def combined_func(x):
            x_arr = np.asarray(x)

            # First compute the linear combination of sine functions
            trig_combination = np.zeros_like(x_arr, dtype=float)
            for i in range(n_modes):
                # Sine function with specified frequency
                sine_part = np.sin(2 * np.pi * frequencies[i] *
                                   (x_arr - a) / interval_length)
                # Add to linear combination with specified coefficient
                trig_combination += coefficients[i] * sine_part

            # Then modulate the entire combination with the single Gaussian
            gaussian_envelope = np.exp(-0.5 * ((x_arr - gaussian_center) /
                                               gaussian_width)**2)

            result = trig_combination * gaussian_envelope
            return result

        return Function(
            self.space,
            evaluate_callable=combined_func,
            name=f'parametric_gaussian_modulated_trig_{n_modes}_modes'
        )

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for this family."""
        a, b = self.domain.a, self.domain.b
        interval_length = b - a

        # Use middle values from ranges for defaults
        n_modes = (self.n_modes_range[0] + self.n_modes_range[1]) // 2

        return {
            'coefficients': np.ones(n_modes),
            'frequencies': np.linspace(self.freq_range[0],
                                       self.freq_range[1], n_modes),
            'gaussian_center': (a + b) / 2,  # Center of interval
            'gaussian_width': 0.3 * interval_length  # 30% of interval
        }
