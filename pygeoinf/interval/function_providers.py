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
    def get_function_by_parameters(self, parameters: Dict[str, Any], **kwargs) -> 'Function':
        """Get a function with specific parameters."""
        pass

    def get_function(self, parameters: Optional[Dict[str, Any]] = None, **kwargs) -> 'Function':
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


class SplineFunctionProvider(IndexedFunctionProvider, ParametricFunctionProvider):
    """Provider for spline functions."""

    def __init__(self, space):
        """
        Initialize spline provider.

        Args:
            space: L2Space instance (contains domain information)
        """
        super().__init__(space)

    def get_function_by_index(self, index: int, degree: int = 3, n_knots: int = 10, **kwargs) -> 'Function':
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

    def get_function_by_parameters(self, parameters: Dict[str, Any], **kwargs) -> 'Function':
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
        jumps = self.rng.uniform(jump_range[0], jump_range[1], n_discontinuities)

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
            n_nodes: Number of nodes. If None, uses space.dim + boundary adjustment
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
                # For non-homogeneous: space.dim total nodes (including boundary)
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
            Function: Hat function that is 1 at node[effective_index] and 0 elsewhere
        """
        if index not in self._cache:
            # Determine which node this function corresponds to
            if self.homogeneous:
                if not (0 <= index < self.space.dim):
                    raise IndexError(f"Index {index} out of range [0, {self.space.dim})")
                # Skip first boundary node: effective_index = index + 1
                effective_index = index + 1
                node_position = self.nodes[effective_index]
            else:
                if not (0 <= index < self.n_nodes):
                    raise IndexError(f"Index {index} out of range [0, {self.n_nodes})")
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
                mask_exact = np.isclose(x, node_position, rtol=1e-14, atol=1e-14)
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
