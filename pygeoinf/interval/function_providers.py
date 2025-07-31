"""
Generalized function providers for creating families of functions.

This module provides a flexible architecture for generating functions from
various families (splines, discontinuous functions, wavelets, etc.) that can be
used independently or as building blocks for basis providers.
"""

import numpy as np
import math
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .l2_functions import Function


class FunctionProvider(ABC):
    """
    Abstract base class for function providers.

    Function providers create Function objects from various families
    with a composable, lazy approach. All providers require explicit
    space specification from the user.
    """

    def __init__(self, space):
        """
        Initialize provider.

        Args:
            space: L2Space instance (contains domain information)
        """
        if space is None:
            raise ValueError(
                f"Space must be provided to {self.__class__.__name__}. "
                "Space cannot be None."
            )
        self.space = space

    @property
    def domain(self):
        """Get the domain from the space."""
        return self.space.function_domain


class IndexedFunctionProvider(FunctionProvider):
    """
    Provider for functions that can be accessed by index.

    Useful for basis functions, orthogonal families, etc.
    """

    @abstractmethod
    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """Get function by index."""
        pass

    def get_functions(self, indices: List[int], **kwargs) -> List['Function']:
        """Get multiple functions by indices."""
        return [self.get_function_by_index(i, **kwargs) for i in indices]


class ParametricFunctionProvider(FunctionProvider):
    """
    Provider for parametric function families.

    Functions are defined by a parameter dictionary.
    """

    @abstractmethod
    def get_function_by_parameters(self, parameters: Dict[str, Any],
                                   **kwargs) -> 'Function':
        """Get a function with specific parameters."""
        pass

    def get_function(self, parameters: Optional[Dict[str, Any]] = None,
                     **kwargs) -> 'Function':
        """Get function with given or default parameters."""
        if parameters is None:
            parameters = self.get_default_parameters()
        return self.get_function_by_parameters(parameters, **kwargs)

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for this family."""
        pass


class RandomFunctionProvider(FunctionProvider):
    """
    Provider for randomly generated functions from a family.
    """

    def __init__(self, space, random_state: Optional[int] = None):
        """Initialize with optional random seed."""
        super().__init__(space)
        self.rng = np.random.RandomState(random_state)

    @abstractmethod
    def sample_function(self, **kwargs) -> 'Function':
        """Sample a random function from the family."""
        pass

    def get_function(self, **kwargs) -> 'Function':
        """Default implementation delegates to sample_function."""
        return self.sample_function(**kwargs)


# Concrete implementations for common function families

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
            space: L2Space instance (contains domain information)
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

    def sample_function(self, **kwargs) -> 'Function':
        """
        Sample a random trigonometric function modulated by Gaussians.

        Returns:
            Function: Random function combining multiple sine functions with
                     Gaussian envelopes
        """
        from .l2_functions import Function

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

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
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
        from .l2_functions import Function

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
                              **kwargs) -> List['Function']:
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
                                   **kwargs) -> 'Function':
        """
        Get a function with specific parameters.

        Args:
            parameters: Dictionary containing 'coefficients', 'frequencies',
                       'gaussian_center', and 'gaussian_width'
        """
        from .l2_functions import Function

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


class FourierFunctionProvider(IndexedFunctionProvider):
    """Provider for Fourier basis functions."""

    def __init__(self, space):
        """
        Initialize Fourier provider.

        Args:
            space: L2Space instance (contains domain information)
        """
        super().__init__(space)

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """
        Get Fourier basis function by index.

        Index 0: constant function
        Index 2k-1: cos(kπx/(b-a))
        Index 2k: sin(kπx/(b-a))
        """
        from .l2_functions import Function

        a, b = self.domain.a, self.domain.b
        L = b - a

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


class SplineFunctionProvider(IndexedFunctionProvider,
                             ParametricFunctionProvider):
    """Provider for spline functions."""

    def __init__(self, space):
        """
        Initialize spline provider.

        Args:
            space: L2Space instance (contains domain information)
        """
        super().__init__(space)

    def get_function_by_index(self, index: int, degree: int = 3,
                              n_knots: int = 10, **kwargs) -> 'Function':
        """Get B-spline basis function by index."""
        from .l2_functions import Function
        from scipy.interpolate import BSpline

        a, b = self.domain.a, self.domain.b

        # Create knot vector
        internal_knots = np.linspace(a, b, n_knots + 2)[1:-1]
        knots = np.concatenate([
            [a] * (degree + 1),
            internal_knots,
            [b] * (degree + 1)
        ])

        # Create coefficient vector (1 at index, 0 elsewhere)
        n_coeffs = len(knots) - degree - 1
        coeffs = np.zeros(n_coeffs)
        coeffs[index % n_coeffs] = 1.0

        spline = BSpline(knots, coeffs, degree)

        def spline_func(x):
            return spline(x)

        return Function(
            self.space,
            evaluate_callable=spline_func,
            name=f'spline_{index}'
        )

    def get_function_by_parameters(self, parameters: Dict[str, Any],
                                   **kwargs) -> 'Function':
        """Get spline with specific parameters."""
        from .l2_functions import Function
        from scipy.interpolate import BSpline

        a, b = self.domain.a, self.domain.b
        degree = parameters.get('degree', 3)
        knots = parameters.get('knots', np.linspace(a, b, 10))
        coeffs = parameters.get('coeffs', np.ones(len(knots) - degree - 1))

        spline = BSpline(knots, coeffs, degree)

        def spline_func(x):
            return spline(x)

        return Function(
            self.space,
            evaluate_callable=spline_func,
            name=f'spline_deg{degree}'
        )

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default spline parameters."""
        a, b = self.domain.a, self.domain.b
        return {
            'degree': 3,
            'knots': np.linspace(a, b, 10),
            'coeffs': np.ones(6)  # Compatible with degree 3 and 10 knots
        }


class BumpFunctionProvider(ParametricFunctionProvider,
                           IndexedFunctionProvider):
    """
    Provider for smooth bump functions with compact support.

    Bump functions are infinitely differentiable (C∞) functions that are
    zero outside a finite interval and positive inside. They use the standard
    mathematical form: exp(-1/((x-a)(b-x))) inside (a,b), zero elsewhere.
    """

    def __init__(self, space, default_width: float = 0.2):
        """
        Initialize bump function provider.

        Args:
            space: L2Space instance (contains domain information)
            default_width: Default width for indexed access (as fraction of
                          domain)
        """
        super().__init__(space)
        self.default_width = default_width
        self._cache = {}

    def get_function_by_parameters(self, parameters: Dict[str, Any],
                                   **kwargs) -> 'Function':
        """
        Get a bump function with specific parameters.

        Args:
            parameters: Dictionary containing:
                - 'center': Center of the bump function
                - 'width': Width of the compact support
        """
        from .l2_functions import Function

        center = parameters['center']
        width = parameters['width']

        # Calculate support interval [a, b]
        a_support = center - width / 2
        b_support = center + width / 2

        def bump_func(x):
            """
            Modified bump function: exp(1/(t²-1)) where t is scaled coordinate.

            Uses the form exp(1/(t²-1)) defined on [-1,1], but transformed
            to have custom center and width. This gives larger, more practical
            values compared to the standard exp(-1/((x-a)(b-x))) form.

            The function is defined as:
            - exp(1/(t²-1)) for t ∈ (-1,1) where t = 2(x-center)/width
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
                # Use exp(1/(t²-1)) form - note the positive sign
                denominator = t_interior**2 - 1.0
                result[interior_mask] = np.exp(1.0 / denominator)

            # Boundary points (|t| = 1) and exterior points remain zero
            return result

        return Function(
            self.space,
            evaluate_callable=bump_func,
            name=f'bump_center_{center:.3f}_width_{width:.3f}',
            support=(a_support, b_support)  # Use Function's compact support
        )

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """
        Get bump function by index with predetermined centers and widths.

        The index determines the center position distributed across the domain,
        using the default width.

        Args:
            index: Index of the bump function (must be >= 0)
        """
        if index < 0:
            raise ValueError(f"Index must be non-negative, got {index}")

        if index not in self._cache:
            a, b = self.domain.a, self.domain.b
            domain_length = b - a

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

            parameters = {'center': center, 'width': width}
            func = self.get_function_by_parameters(parameters)
            func.name = f'bump_{index}_center_{center:.3f}_width_{width:.3f}'

            self._cache[index] = func

        return self._cache[index]

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for bump functions."""
        a, b = self.domain.a, self.domain.b
        domain_length = b - a

        return {
            'center': (a + b) / 2,  # Center of domain
            'width': self.default_width * domain_length
        }

    def get_function(self, parameters: Optional[Dict[str, Any]] = None,
                     **kwargs) -> 'Function':
        """Get bump function with given or default parameters."""
        if parameters is None:
            parameters = self.get_default_parameters()
        return self.get_function_by_parameters(parameters, **kwargs)


class DiscontinuousFunctionProvider(RandomFunctionProvider):
    """Provider for functions with random discontinuities."""

    def __init__(self, space, random_state=None):
        """
        Initialize discontinuous function provider.

        Args:
            space: L2Space instance (contains domain information)
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
        from .l2_functions import Function

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


class WaveletFunctionProvider(IndexedFunctionProvider):
    """Provider for wavelet basis functions."""

    def __init__(self, space, wavelet_type: str = 'haar'):
        """
        Initialize wavelet provider.

        Args:
            space: L2Space instance (contains domain information)
            wavelet_type: Type of wavelet ('haar', etc.)
        """
        super().__init__(space)
        self.wavelet_type = wavelet_type

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """Get wavelet function by index."""

        if self.wavelet_type == 'haar':
            return self._get_haar_wavelet(index)
        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

    def _get_haar_wavelet(self, index: int) -> 'Function':
        """Get Haar wavelet by index."""
        from .l2_functions import Function

        a, b = self.domain.a, self.domain.b

        if index == 0:
            # Scaling function (constant)
            def scaling_func(x):
                return np.ones_like(x) / np.sqrt(b - a)

            return Function(
                self.space,
                evaluate_callable=scaling_func,
                name='haar_scaling'
            )

        # Decode index to get level and position
        level = int(np.log2(index)) + 1
        index_in_level = index - (2**(level-1) - 1)

        def haar_func(x):
            x_arr = np.asarray(x)
            result = np.zeros_like(x_arr, dtype=float)

            # Normalize x to [0, 1]
            x_norm = (x_arr - a) / (b - a)

            # Calculate shift and scale for this wavelet
            scale = 2**level
            shift = index_in_level / scale

            # Haar wavelet: +1 on first half, -1 on second half of support
            mask1 = ((x_norm >= shift) &
                     (x_norm < shift + 0.5/scale))
            mask2 = ((x_norm >= shift + 0.5/scale) &
                     (x_norm < shift + 1.0/scale))

            result[mask1] = np.sqrt(scale)
            result[mask2] = -np.sqrt(scale)

            return result

        return Function(
            self.space,
            evaluate_callable=haar_func,
            name=f'haar_L{level}_I{index}'
        )


# Adapter classes to bridge to the current basis/spectrum provider system

class FunctionProviderAdapter:
    """
    Adapter to use FunctionProviders in the current basis system.

    This bridges the gap between the new flexible function providers
    and the existing basis/spectrum provider interfaces.
    """

    def __init__(self, function_provider: FunctionProvider):
        """
        Initialize adapter.

        Args:
            function_provider: The function provider to adapt
        """
        self.function_provider = function_provider
        self._cache = {}

    @property
    def space(self):
        """Get the space from the function provider."""
        return self.function_provider.space

    def get_basis_function(self, index: int):
        """Get basis function by index (for indexed providers)."""
        if index not in self._cache:
            if isinstance(self.function_provider, IndexedFunctionProvider):
                func = self.function_provider.get_function_by_index(index)
                # Ensure the function belongs to our space
                func.space = self.space
                self._cache[index] = func
            else:
                raise ValueError(
                    "Provider must be IndexedFunctionProvider for basis use"
                )
        return self._cache[index]


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
        """Get sine function with index k = index + 1."""
        if index not in self._cache:
            a, b = self.space.function_domain.a, self.space.function_domain.b
            length = b - a
            k = index + 1  # Sine functions start from k=1

            def sine_func(x):
                if isinstance(x, np.ndarray):
                    return np.sin(2 * k * np.pi * (x - a) / length)
                else:
                    return math.sin(2 * k * math.pi * (x - a) / length)

            from .l2_functions import Function
            func = Function(
                self.space,
                evaluate_callable=sine_func,
                name=f"sin({k}π(x-{a})/{length})"
            )
            self._cache[index] = func

        return self._cache[index]


class CosineFunctionProvider(IndexedFunctionProvider):
    """
    Provider for cosine functions: cos(kπ(x-a)/L).

    These are the eigenfunctions for Neumann boundary conditions
    on the negative Laplacian operator (with constant mode for k=0).
    """

    def __init__(self, space):
        """Initialize the cosine function provider."""
        super().__init__(space)
        self._cache = {}

    def get_function_by_index(self, index: int):
        """Get cosine function or constant for index 0."""
        if index not in self._cache:
            a, b = self.space.function_domain.a, self.space.function_domain.b
            length = b - a

            if index == 0:
                # Constant mode for Neumann BC
                def constant_func(x):
                    return (np.ones_like(x) if isinstance(x, np.ndarray)
                            else 1.0)

                from .l2_functions import Function
                func = Function(
                    self.space,
                    evaluate_callable=constant_func,
                    name="1 (constant)"
                )
            else:
                # Cosine modes for k = index
                k = index

                def cosine_func(x):
                    if isinstance(x, np.ndarray):
                        return np.cos(k * np.pi * (x - a) / length)
                    else:
                        return math.cos(k * math.pi * (x - a) / length)

                from .l2_functions import Function
                func = Function(
                    self.space,
                    evaluate_callable=cosine_func,
                    name=f"cos({k}π(x-{a})/{length})"
                )

            self._cache[index] = func

        return self._cache[index]


class HatFunctionProvider(IndexedFunctionProvider):
    """
    Provider for hat functions (piecewise linear basis functions).

    Hat functions are continuous, piecewise linear functions that form
    a basis for finite element methods. Each function is 1 at one node
    and 0 at all other nodes.

    For homogeneous hat functions, the boundary nodes are omitted,
    satisfying homogeneous Dirichlet boundary conditions.
    """

    def __init__(self, space, homogeneous=False, n_nodes=None):
        """
        Initialize the hat function provider.

        Args:
            space: L2Space instance (contains domain information)
            homogeneous: If True, omit boundary nodes (homogeneous Dirichlet)
            n_nodes: Number of nodes. If None, uses space.dim + boundary
                     adjustment
        """
        super().__init__(space)
        self._cache = {}
        self.homogeneous = homogeneous

        # Determine number of nodes
        if n_nodes is None:
            if homogeneous:
                # For homogeneous: space.dim interior nodes + 2 boundary nodes
                self.n_nodes = self.space.dim + 2
            else:
                # For non-homogeneous: space.dim total nodes
                # (including boundary)
                self.n_nodes = self.space.dim
        else:
            self.n_nodes = n_nodes

        # Create node coordinates
        a, b = self.space.function_domain.a, self.space.function_domain.b
        self.nodes = np.linspace(a, b, self.n_nodes)
        self.h = (b - a) / (self.n_nodes - 1)  # Node spacing

    def get_function_by_index(self, index: int):
        """
        Get hat function for given index.

        Args:
            index: Index of the hat function

        Returns:
            Function: Hat function that is 1 at node[effective_index] and 0
                      elsewhere
        """
        if index not in self._cache:
            # Determine which node this function corresponds to
            if self.homogeneous:
                if not (0 <= index < self.space.dim):
                    raise IndexError(
                        f"Index {index} out of range [0, {self.space.dim})"
                    )
                # Skip first boundary node: effective_index = index + 1
                effective_index = index + 1
                node_position = self.nodes[effective_index]
            else:
                if not (0 <= index < self.n_nodes):
                    raise IndexError(
                        f"Index {index} out of range [0, {self.n_nodes})"
                    )
                effective_index = index
                node_position = self.nodes[effective_index]

            def hat_func(x):
                """Piecewise linear hat function."""
                x = np.asarray(x)
                result = np.zeros_like(x, dtype=float)

                # Hat function is non-zero only in [x_{i-1}, x_{i+1}]
                left_node = effective_index - 1
                right_node = effective_index + 1

                if left_node >= 0:
                    # Left piece: linear from 0 to 1
                    left_x = self.nodes[left_node]
                    mask_left = (x >= left_x) & (x <= node_position)
                    if np.any(mask_left):
                        result[mask_left] = (x[mask_left] - left_x) / self.h

                if right_node < self.n_nodes:
                    # Right piece: linear from 1 to 0
                    right_x = self.nodes[right_node]
                    mask_right = (x >= node_position) & (x <= right_x)
                    if np.any(mask_right):
                        result[mask_right] = (right_x - x[mask_right]) / self.h

                # Handle the case where x is exactly at the node
                mask_exact = np.isclose(
                    x, node_position, rtol=1e-14, atol=1e-14
                )
                result[mask_exact] = 1.0

                return result

            from .l2_functions import Function

            # Create function name
            if self.homogeneous:
                name = f"hat_hom_{index}(x={node_position:.3f})"
            else:
                name = f"hat_{index}(x={node_position:.3f})"

            func = Function(
                self.space,
                evaluate_callable=hat_func,
                name=name
            )

            self._cache[index] = func

        return self._cache[index]

    def get_nodes(self):
        """
        Get the node coordinates.

        Returns:
            np.ndarray: Array of node coordinates
        """
        return self.nodes.copy()

    def get_active_nodes(self):
        """
        Get the coordinates of nodes corresponding to basis functions.

        For homogeneous hat functions, this excludes boundary nodes.

        Returns:
            np.ndarray: Array of active node coordinates
        """
        if self.homogeneous:
            return self.nodes[1:-1].copy()  # Exclude boundary nodes
        else:
            return self.nodes.copy()  # All nodes are active
