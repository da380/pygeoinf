"""
Abstract base classes and implementations for basis and spectrum providers.

This module defines the interfaces that basis and spectrum providers must
implement to work with L2Space and SobolevSpace classes. It also provides
concrete implementations for common basis types.
"""

import numpy as np
import math
from abc import ABC, abstractmethod


class BasisProvider(ABC):
    """
    Abstract base class for basis function providers.

    This defines the minimum interface that any basis provider must implement
    to work with L2Space and function spaces. Users can inherit from this
    class to create custom basis providers.
    """

    def __init__(self, space, **kwargs):
        """
        Initialize the basis provider.

        Args:
            space: The function space that owns this provider
            **kwargs: Additional parameters specific to the implementation
        """
        self.space = space

    @abstractmethod
    def get_basis_function(self, index: int):
        """
        Get basis function for given index.

        Args:
            index: Index of the basis function (0 to dim-1)

        Returns:
            Function: The basis function at the given index

        Raises:
            IndexError: If index is out of range [0, space.dim)
        """
        pass

    def __getitem__(self, index: int):
        """Allow indexing syntax: provider[i]."""
        return self.get_basis_function(index)

    def __len__(self):
        """Return the dimension of the space."""
        return self.space.dim

    def __iter__(self):
        """Allow iteration over basis functions."""
        for i in range(self.space.dim):
            yield self.get_basis_function(i)

    def get_all_basis_functions(self):
        """
        Return all basis functions as a list.

        Returns:
            list: List of all Function basis functions
        """
        return [self.get_basis_function(i) for i in range(self.space.dim)]


class SpectrumProvider(BasisProvider):
    """
    Abstract base class for spectrum providers.

    Extends BasisProvider to include eigenvalue information needed for
    spectral inner products in Sobolev spaces. Users should inherit from
    this class when creating custom spectrum providers.
    """

    @abstractmethod
    def get_eigenvalue(self, index: int):
        """
        Get eigenvalue for given basis function index.

        Args:
            index: Index of the eigenfunction (0 to dim-1)

        Returns:
            float: Eigenvalue corresponding to the eigenfunction

        Raises:
            IndexError: If index is out of range [0, space.dim)
        """
        pass

    def get_eigenfunction(self, index: int):
        """
        Get eigenfunction for given index.

        This is the same as get_basis_function but emphasizes that
        these are eigenfunctions of some operator.
        """
        return self.get_basis_function(index)

    def get_all_eigenvalues(self):
        """
        Return all eigenvalues as an array.

        Returns:
            np.ndarray: Array of all eigenvalues
        """
        return np.array([
            self.get_eigenvalue(i) for i in range(self.space.dim)
        ])


class LazyBasisProvider(BasisProvider):
    """
    Lazy provider for basis functions.

    Creates basis functions on demand and caches them to avoid
    memory issues with high-dimensional spaces. This is the base
    implementation that provides only basis functions, no eigenvalue
    information.
    """

    def __init__(self, space, basis_type: str):
        """
        Initialize the lazy basis provider.

        Args:
            space: Space that owns this provider (L2Space or Sobolev)
            basis_type: Type of basis functions
                ('fourier', 'hat', 'hat_homogeneous')
        """
        super().__init__(space)
        self.basis_type = basis_type
        self._cache = {}

    def get_basis_function(self, index: int):
        """
        Get basis function for given index.

        Args:
            index: Index of the basis function (0 to dim-1)

        Returns:
            Function for that index
        """
        if not (0 <= index < self.space.dim):
            raise IndexError(
                f"Basis index {index} out of range [0, {self.space.dim})"
            )

        if index not in self._cache:
            self._cache[index] = self._create_basis_function(index)
        return self._cache[index]

    def _create_basis_function(self, index: int):
        """Create a single basis function for the given index."""
        if self.basis_type == 'fourier':
            return self._create_fourier_basis_function(index)
        elif self.basis_type == 'hat':
            return self._create_full_hat_basis_function(index)
        elif self.basis_type == 'hat_homogeneous':
            return self._create_homogeneous_hat_basis_function(index)
        else:
            raise ValueError(f"Unsupported basis type: {self.basis_type}")

    def _create_fourier_basis_function(self, index: int):
        """Create a single Fourier basis function.

        For L2Space, we create the full Fourier basis (periodic case)
        without boundary condition considerations.
        """
        from .l2_functions import Function

        domain = self.space.function_domain
        length = domain.b - domain.a

        # Full Fourier basis (periodic case)
        normalization_factor = math.sqrt(2 / length)

        if index == 0:
            # Constant term
            def constant_func(x):
                return (normalization_factor *
                        np.ones_like(x) / np.sqrt(2))
            return Function(
                self.space,
                evaluate_callable=constant_func,
                name='constant'
            )
        else:
            # For index > 0, we alternate between cosine and sine
            k = (index + 1) // 2  # Frequency index
            freq = 2 * k * math.pi / length

            if index % 2 == 1:  # Odd indices are cosine
                def cosine_func(x):
                    return (normalization_factor *
                            np.cos(freq * (x - domain.a)))
                return Function(
                    self.space,
                    evaluate_callable=cosine_func,
                    name=f'cos_{k}'
                )
            else:  # Even indices > 0 are sine
                def sine_func(x):
                    return (normalization_factor *
                            np.sin(freq * (x - domain.a)))
                return Function(
                    self.space,
                    evaluate_callable=sine_func,
                    name=f'sin_{k}'
                )

    def _create_homogeneous_hat_basis_function(self, index: int):
        """Create a single homogeneous hat (piecewise linear) basis function.

        These are interior hat functions that vanish at the boundaries,
        suitable for homogeneous Dirichlet boundary conditions.
        """
        from .l2_functions import Function

        domain = self.space._function_domain

        # Create uniform mesh for hat functions
        # For dim basis functions, we need dim+2 nodes (including boundaries)
        nodes = np.linspace(domain.a, domain.b, self.space.dim + 2)
        element_size = nodes[1] - nodes[0]

        # Hat function φᵢ has support on [nodes[i], nodes[i+2]]
        # and has value 1 at nodes[i+1]
        node_index = index + 1  # Interior node index
        support = (nodes[index], nodes[index + 2])

        def hat_func(x):
            x_array = np.asarray(x)
            is_scalar = x_array.ndim == 0
            if is_scalar:
                x_array = x_array.reshape(1)

            result = np.zeros_like(x_array, dtype=float)

            # Left element: increasing from 0 to 1
            in_left_element = (
                (x_array >= support[0]) &
                (x_array <= nodes[node_index])
            )
            if np.any(in_left_element):
                x_left = x_array[in_left_element]
                result[in_left_element] = (
                    (x_left - support[0]) / element_size
                )

            # Right element: decreasing from 1 to 0
            in_right_element = (
                (x_array > nodes[node_index]) &
                (x_array <= support[1])
            )
            if np.any(in_right_element):
                x_right = x_array[in_right_element]
                result[in_right_element] = (
                    (support[1] - x_right) / element_size
                )

            return result.item() if is_scalar else result

        return Function(
            self.space,
            evaluate_callable=hat_func,
            name=f'φ_{index}',
            support=support
        )

    def _create_full_hat_basis_function(self, index: int):
        """Create a single full hat (piecewise linear) basis function.

        These include boundary functions and interior functions,
        forming a complete basis without boundary conditions.
        """
        from .l2_functions import Function

        domain = self.space._function_domain

        # For full hat functions, we have dim nodes from a to b
        nodes = np.linspace(domain.a, domain.b, self.space.dim)

        if self.space.dim == 1:
            # Special case: single node (constant function)
            def constant_hat_func(x):
                x_array = np.asarray(x)
                return np.ones_like(x_array, dtype=float)

            return Function(
                self.space,
                evaluate_callable=constant_hat_func,
                name=f'φ_{index}',
                support=(domain.a, domain.b)
            )

        element_size = nodes[1] - nodes[0]
        node_x = nodes[index]

        # Determine support based on position
        if index == 0:
            # Left boundary: half-hat from left boundary to first interior
            support = (domain.a, nodes[1])

            def left_boundary_hat_func(x):
                x_array = np.asarray(x)
                is_scalar = x_array.ndim == 0
                if is_scalar:
                    x_array = x_array.reshape(1)

                result = np.zeros_like(x_array, dtype=float)

                # Decreasing from 1 to 0
                in_support = (x_array >= support[0]) & (x_array <= support[1])
                if np.any(in_support):
                    x_support = x_array[in_support]
                    result[in_support] = (nodes[1] - x_support) / element_size

                return result.item() if is_scalar else result

            hat_func = left_boundary_hat_func

        elif index == self.space.dim - 1:
            # Right boundary: half-hat from last interior to right boundary
            support = (nodes[-2], domain.b)

            def right_boundary_hat_func(x):
                x_array = np.asarray(x)
                is_scalar = x_array.ndim == 0
                if is_scalar:
                    x_array = x_array.reshape(1)

                result = np.zeros_like(x_array, dtype=float)

                # Increasing from 0 to 1
                in_support = (x_array >= support[0]) & (x_array <= support[1])
                if np.any(in_support):
                    x_support = x_array[in_support]
                    result[in_support] = (x_support - nodes[-2]) / element_size

                return result.item() if is_scalar else result

            hat_func = right_boundary_hat_func

        else:
            # Interior: full triangle hat
            support = (nodes[index-1], nodes[index+1])

            def interior_hat_func(x):
                x_array = np.asarray(x)
                is_scalar = x_array.ndim == 0
                if is_scalar:
                    x_array = x_array.reshape(1)

                result = np.zeros_like(x_array, dtype=float)

                # Left element: increasing from 0 to 1
                in_left_element = (
                    (x_array >= support[0]) &
                    (x_array <= node_x)
                )
                if np.any(in_left_element):
                    x_left = x_array[in_left_element]
                    result[in_left_element] = (
                        (x_left - support[0]) / element_size
                    )

                # Right element: decreasing from 1 to 0
                in_right_element = (
                    (x_array > node_x) &
                    (x_array <= support[1])
                )
                if np.any(in_right_element):
                    x_right = x_array[in_right_element]
                    result[in_right_element] = (
                        (support[1] - x_right) / element_size
                    )

                return result.item() if is_scalar else result

            hat_func = interior_hat_func

        return Function(
            self.space,
            evaluate_callable=hat_func,
            name=f'φ_{index}',
            support=support
        )


class LazySpectrumProvider(SpectrumProvider, LazyBasisProvider):
    """
    Lazy provider for eigenfunctions and eigenvalues (spectral information).

    Extends LazyBasisProvider to include eigenvalue information needed
    for spectral inner products in Sobolev spaces.
    """

    def __init__(self, space, basis_type: str, operator=None):
        """
        Initialize the lazy spectrum provider.

        Args:
            space: Space that owns this provider
            basis_type: Type of basis functions (must be eigenfunctions)
            operator: Optional operator whose spectrum defines eigenvalues
        """
        LazyBasisProvider.__init__(self, space, basis_type)
        self.operator = operator
        self._eigenvalue_cache = {}

    def get_eigenvalue(self, index: int):
        """
        Get eigenvalue for given basis function index.

        Args:
            index: Index of the eigenfunction (0 to dim-1)

        Returns:
            float: Eigenvalue corresponding to the eigenfunction
        """
        if not (0 <= index < self.space.dim):
            raise IndexError(
                f"Eigenvalue index {index} out of range [0, {self.space.dim})"
            )

        if index not in self._eigenvalue_cache:
            self._eigenvalue_cache[index] = self._compute_eigenvalue(index)
        return self._eigenvalue_cache[index]

    def _compute_eigenvalue(self, index: int):
        """
        Compute eigenvalue for given index.

        Uses the cached computation from _compute_all_eigenvalues()
        to ensure consistency and efficiency.
        """
        # Compute all eigenvalues and cache them
        if not hasattr(self, '_all_eigenvalues'):
            self._all_eigenvalues = self._compute_all_eigenvalues()
        return self._all_eigenvalues[index]

    def _compute_all_eigenvalues(self):
        """Compute all eigenvalues at once."""
        # This will be implemented differently for different spaces
        # For now, implement Fourier basis with periodic boundary conditions
        if self.basis_type == 'fourier':
            return self._compute_fourier_eigenvalues()
        else:
            # For non-Fourier bases, return zeros as placeholder
            return np.zeros(self.space.dim)

    def _compute_fourier_eigenvalues(self):
        """
        Compute eigenvalues for Fourier basis with periodic boundary
        conditions.

        For the negative Laplacian operator -Δ with periodic BCs on [a,b]:
        - Eigenfunction φ₀ = 1 (constant) → eigenvalue λ₀ = 0
        - Eigenfunctions φ₂ₖ₋₁ = cos(2πkx/L), φ₂ₖ = sin(2πkx/L)
          → eigenvalue λₖ = (2πk/L)² for both cos and sin modes

        where L = b - a is the domain length.
        """
        domain = self.space.function_domain
        length = domain.b - domain.a
        dim = self.space.dim

        eigenvalues = np.zeros(dim)

        for index in range(dim):
            if index == 0:
                # Constant term has eigenvalue 0
                eigenvalues[index] = 0.0
            else:
                # For index > 0, we alternate between cosine and sine
                k = (index + 1) // 2  # Frequency index
                # Both cos and sin modes have the same eigenvalue
                eigenval = (2 * k * math.pi / length) ** 2
                eigenvalues[index] = eigenval

        return eigenvalues


class CustomSpectrumProvider(SpectrumProvider):
    """
    Custom spectrum provider that wraps a basis provider and stores
    eigenvalues.

    This is useful when users provide basis_callables + eigenvalues to
    SobolevSpace for spectral inner products.
    """

    def __init__(self, basis_provider: BasisProvider, eigenvalues: np.ndarray):
        """
        Initialize with a basis provider and eigenvalues.

        Args:
            basis_provider: Provider for the basis functions
            eigenvalues: Array of eigenvalues corresponding to basis functions
        """
        super().__init__(basis_provider.space)
        self.basis_provider = basis_provider
        self.eigenvalues = np.asarray(eigenvalues)

        if len(self.eigenvalues) != self.space.dim:
            raise ValueError(
                f"eigenvalues length ({len(self.eigenvalues)}) "
                f"must match space dimension ({self.space.dim})"
            )

    def get_basis_function(self, index: int):
        """Get basis function from the wrapped provider."""
        return self.basis_provider.get_basis_function(index)

    def get_eigenvalue(self, index: int):
        """Get eigenvalue for given index."""
        if not (0 <= index < self.space.dim):
            raise IndexError(
                f"Eigenvalue index {index} out of range [0, {self.space.dim})"
            )
        return self.eigenvalues[index]


# Backward compatibility aliases
LazyL2BasisProvider = LazyBasisProvider
