"""
Abstract base classes and implementations for basis and spectrum providers.

This module defines the interfaces that basis and spectrum providers must
implement to work with L2Space and SobolevSpace classes. The design follows
a composable pattern:

- BasisProvider: Wraps any IndexedFunctionProvider and adds space functionality
- SpectrumProvider: Extends BasisProvider with eigenvalue information
- EigenvalueProvider: Provides eigenvalues for spectrum computations

All providers are lazy by nature - they compute and cache functions/eigenvalues
on demand to handle high-dimensional spaces efficiently.
"""

import numpy as np
import math
from abc import ABC, abstractmethod
from typing import Optional
from .function_providers import IndexedFunctionProvider


class EigenvalueProvider(ABC):
    """
    Abstract base class for eigenvalue providers.

    Eigenvalue providers compute eigenvalues for corresponding function
    providers, typically for specific differential operators.
    """

    @abstractmethod
    def get_eigenvalue(self, index: int) -> float:
        """
        Get eigenvalue for given index.

        Args:
            index: Index of the eigenfunction

        Returns:
            float: Eigenvalue at the given index
        """
        pass

    def get_eigenvalues(self, n: int) -> np.ndarray:
        """
        Get first n eigenvalues as array.

        Args:
            n: Number of eigenvalues to compute

        Returns:
            np.ndarray: Array of eigenvalues
        """
        return np.array([self.get_eigenvalue(i) for i in range(n)])


class BasisProvider:
    """
    Basis provider that wraps any IndexedFunctionProvider.

    This class adds space-specific functionality on top of function providers:
    - Ensures the number of functions matches space dimension
    - Provides space-aware caching and validation
    - Implements convenient access patterns (indexing, iteration)

    All computation is lazy with caching for efficiency.
    """

    def __init__(self, space, function_provider: IndexedFunctionProvider,
                 orthonormal: bool = False, basis_type: Optional[str] = None):
        """
        Initialize basis provider with a function provider.

        Args:
            space: The function space that owns this provider
            function_provider: IndexedFunctionProvider for the basis functions
            orthonormal: True if the basis functions are orthonormal with
                        respect to the L² inner product
        """
        self.space = space
        self.function_provider = function_provider
        self.orthonormal = orthonormal
        self._cache = {}
        self.type = basis_type

        # Ensure function provider knows about the space
        if hasattr(function_provider, 'space'):
            function_provider.space = space

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
        if not (0 <= index < self.space.dim):
            raise IndexError(
                f"Basis index {index} out of range [0, {self.space.dim})"
            )

        if index not in self._cache:
            func = self.function_provider.get_function_by_index(index)
            # Ensure function belongs to our space
            if hasattr(func, 'space'):
                func.space = self.space
            self._cache[index] = func

        return self._cache[index]

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
        Return n basis functions as a list.

        Args:
            n: Number of basis functions to return (default: space.dim)

        Returns:
            list: List of Function basis functions
        """
        return [self.get_basis_function(i) for i in range(self.space.dim)]


class SpectrumProvider(BasisProvider):
    """
    Spectrum provider that extends BasisProvider with eigenvalue information.

    Combines a function provider (for eigenfunctions) with an eigenvalue
    provider (for corresponding eigenvalues). This is used for spectral
    inner products in Sobolev spaces.

    All computation is lazy with caching for efficiency.
    """

    def __init__(self, space, function_provider: IndexedFunctionProvider,
                 eigenvalue_provider: EigenvalueProvider,
                 orthonormal: bool = False, basis_type: Optional[str] = None):
        """
        Initialize spectrum provider.

        Args:
            space: The function space
            function_provider: IndexedFunctionProvider for eigenfunctions
            eigenvalue_provider: EigenvalueProvider for eigenvalues
            orthonormal: True if the basis functions are orthonormal with
                        respect to the L² inner product
            basis_type: String identifier for the type of basis functions
        """
        super().__init__(space, function_provider, orthonormal, basis_type)
        self.eigenvalue_provider = eigenvalue_provider
        self._eigenvalue_cache = {}

    def get_eigenvalue(self, index: int) -> float:
        """
        Get eigenvalue for given basis function index.

        Args:
            index: Index of the eigenfunction (0 to dim-1)

        Returns:
            float: Eigenvalue corresponding to the eigenfunction

        Raises:
            IndexError: If index is out of range [0, space.dim)
        """
        if not (0 <= index < self.space.dim):
            raise IndexError(
                f"Eigenvalue index {index} out of range [0, {self.space.dim})"
            )

        if index not in self._eigenvalue_cache:
            self._eigenvalue_cache[index] = (
                self.eigenvalue_provider.get_eigenvalue(index)
            )
        return self._eigenvalue_cache[index]

    def get_eigenfunction(self, index: int):
        """
        Get eigenfunction for given index.

        This is the same as get_basis_function but emphasizes that
        these are eigenfunctions of some operator.
        """
        return self.get_basis_function(index)

    def get_all_eigenvalues(self, n=None):
        """
        Return n eigenvalues as an array.

        Args:
            n: Number of eigenvalues to return. If None, returns space.dim
               eigenvalues for space-based computations.

        Returns:
            np.ndarray: Array of eigenvalues
        """
        if n is None:
            n = self.space.dim

        return np.array([
            self.get_eigenvalue(i) for i in range(n)
        ])


# Concrete eigenvalue providers for common operators

class FourierEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider for Fourier eigenfunctions of the negative Laplacian.

    Computes eigenvalues λₖ = (2πk/L)² where L is the domain length.
    """

    def __init__(self, domain_length: float):
        """
        Initialize Fourier eigenvalue provider.

        Args:
            domain_length: Length of the domain (b - a)
        """
        self.domain_length = domain_length

    def get_eigenvalue(self, index: int) -> float:
        """Compute eigenvalue for the negative Laplacian operator."""
        if index == 0:
            # Constant term has eigenvalue 0
            return 0.0
        else:
            # For index > 0, we alternate between cosine and sine
            k = (index + 1) // 2  # Frequency index
            # Both cos and sin modes have the same eigenvalue
            return (2 * k * math.pi / self.domain_length) ** 2


class ZeroEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider that returns zero for all indices.

    Useful for basis functions that aren't eigenfunctions of a specific
    operator, or when eigenvalue information isn't needed.
    """

    def get_eigenvalue(self, index: int) -> float:
        """Return zero eigenvalue."""
        return 0.0


class CustomEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider with user-specified eigenvalues.

    Allows users to provide their own eigenvalue arrays.
    """

    def __init__(self, eigenvalues: np.ndarray):
        """
        Initialize with eigenvalue array.

        Args:
            eigenvalues: Array of eigenvalues
        """
        self.eigenvalues = np.asarray(eigenvalues)

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue from the stored array."""
        if not (0 <= index < len(self.eigenvalues)):
            raise IndexError(
                f"Eigenvalue index {index} out of range "
                f"[0, {len(self.eigenvalues)})"
            )
        return self.eigenvalues[index]


# Factory functions for creating providers

def create_basis_provider(space, basis_type: str):
    """
    Create a BasisProvider with the appropriate function provider for the
    given basis type.

    Args:
        space: The L2Space instance
        basis_type: Type of basis ('fourier', 'hat', 'hat_homogeneous', 'sine')

    Returns:
        BasisProvider: Provider configured for the specified basis type
    """
    from .function_providers import (FourierFunctionProvider,
                                     HatFunctionProvider,
                                     SineFunctionProvider,
                                     CosineFunctionProvider)

    if basis_type == 'fourier':
        func_provider = FourierFunctionProvider(space)
        return BasisProvider(space, func_provider, orthonormal=True,
                             basis_type='fourier')
    elif basis_type == 'hat':
        # Non-homogeneous hat functions (include boundary nodes)
        func_provider = HatFunctionProvider(space, homogeneous=False)
        return BasisProvider(space, func_provider, orthonormal=False,
                             basis_type='hat')
    elif basis_type == 'hat_homogeneous':
        # Homogeneous hat functions (exclude boundary nodes)
        func_provider = HatFunctionProvider(space, homogeneous=True)
        return BasisProvider(space, func_provider, orthonormal=False,
                             basis_type='hat_homogeneous')
    elif basis_type == 'sine':
        # Sine functions for Dirichlet boundary conditions
        func_provider = SineFunctionProvider(space)
        return BasisProvider(space, func_provider, orthonormal=True,
                             basis_type='sine')
    elif basis_type == 'cosine':
        # Cosine functions for Neumann boundary conditions
        func_provider = CosineFunctionProvider(space)
        return BasisProvider(space, func_provider, orthonormal=True,
                             basis_type='cosine')
    else:
        raise ValueError(f"Unsupported basis type: {basis_type}")


def create_spectrum_provider(space, function_provider, eigenvalue_provider,
                             orthonormal: bool = False,
                             basis_type: Optional[str] = None):
    """
    Create a SpectrumProvider with the given components.

    Args:
        space: The L2Space instance
        function_provider: IndexedFunctionProvider for the functions
        eigenvalue_provider: EigenvalueProvider for the eigenvalues
        orthonormal: True if the basis functions are orthonormal with
                    respect to the L² inner product
        basis_type: String identifier for the type of basis functions

    Returns:
        SpectrumProvider: Configured spectrum provider
    """
    return SpectrumProvider(space, function_provider, eigenvalue_provider,
                            orthonormal, basis_type)


class LaplacianEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider for the negative Laplacian operator.

    Computes eigenvalues λₖ based on boundary conditions and domain.
    For the inverse Laplacian, eigenvalues are 1/λₖ.
    """

    def __init__(
        self,
        function_domain,
        boundary_conditions,
        inverse=False,
        alpha=1.0
    ):
        """
        Initialize the eigenvalue provider.

        Args:
            function_domain: Interval domain
            boundary_conditions: Boundary conditions ('dirichlet', 'neumann',
                                 'periodic')
            inverse: If True, compute eigenvalues of (-Δ)⁻¹, else of -Δ
            alpha: Scaling factor for the eigenvalues (default: 1.0)
        """
        self.function_domain = function_domain
        self.boundary_conditions = boundary_conditions
        self.inverse = inverse
        self.alpha = alpha
        self._eigenvalue_cache = {}

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue for given index."""
        if index not in self._eigenvalue_cache:
            self._eigenvalue_cache[index] = self._compute_eigenvalue(index)
        return self._eigenvalue_cache[index]

    def _compute_eigenvalue(self, index: int) -> float:
        """Compute eigenvalue based on boundary conditions."""
        length = self.function_domain.b - self.function_domain.a

        if self.boundary_conditions.type == 'dirichlet':
            # λₖ = (kπ/L)² where k = index + 1
            k = index + 1
            eigenval = self.alpha * (k * math.pi / length)**2

        elif self.boundary_conditions.type == 'neumann':
            if index == 0:
                # Constant mode has eigenvalue 0 for -Δ
                eigenval = 0.0
            else:
                # λₖ = (kπ/L)² where k = index
                k = index
                eigenval = self.alpha * (k * math.pi / length)**2

        elif self.boundary_conditions.type == 'periodic':
            if index == 0:
                # Constant mode has eigenvalue 0 for -Δ
                eigenval = 0.0
            else:
                # Both cos and sin modes: λₖ = (2πk/L)²
                k = (index + 1) // 2  # Frequency index
                eigenval = self.alpha * (2 * k * math.pi / length)**2

        else:
            raise ValueError(f"Unsupported boundary condition: "
                             f"{self.boundary_conditions.type}")

        # Return inverse if requested
        if self.inverse:
            if eigenval == 0.0:
                return float('inf')  # 1/0 = ∞
            else:
                return 1.0 / eigenval
        else:
            return eigenval


def create_laplacian_spectrum_provider(space, boundary_conditions,
                                       inverse=False):
    """
    Factory function to create spectrum provider for Laplacian eigenfunctions.

    Args:
        space: Function space
        boundary_conditions: BoundaryConditions object
        inverse: If True, create provider for (-Δ)^(-1), else for -Δ

    Returns:
        SpectrumProvider: Configured spectrum provider for Laplacian
    """
    # Import here to avoid circular imports
    from .function_providers import (
        SineFunctionProvider, CosineFunctionProvider, FourierFunctionProvider
    )

    # Choose appropriate function provider based on boundary conditions
    if boundary_conditions.type == 'dirichlet':
        function_provider = SineFunctionProvider(space)
        orthonormal = True  # Sine functions are normalized
        basis_type = 'sine_dirichlet'
    elif boundary_conditions.type == 'neumann':
        function_provider = CosineFunctionProvider(space)
        orthonormal = True  # Cosine functions are normalized
        basis_type = 'cosine_neumann'
    elif boundary_conditions.type == 'periodic':
        function_provider = FourierFunctionProvider(space)
        orthonormal = True  # Fourier functions are orthonormal
        basis_type = 'fourier_periodic'
    else:
        raise ValueError(f"Unsupported boundary condition: "
                         f"{boundary_conditions.type}")

    # Create eigenvalue provider
    eigenvalue_provider = LaplacianEigenvalueProvider(
        space.function_domain, boundary_conditions, inverse=inverse
    )

    return SpectrumProvider(space, function_provider, eigenvalue_provider,
                            orthonormal, basis_type)
