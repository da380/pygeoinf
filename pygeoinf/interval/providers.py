from __future__ import annotations

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
from typing import Optional, Union, TYPE_CHECKING

from .utils.robin_utils import RobinRootFinder

if TYPE_CHECKING:
    from .lebesgue_space import Lebesgue
    from .sobolev_space import Sobolev


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
        Get array of eigenvalues up to index n.

        Args:
            n: Number of eigenvalues to return (default: all computed so far)

        Returns:
            np.ndarray: Array of eigenvalues
        """

        return np.array([self.get_eigenvalue(i) for i in range(n)])


class BasisProvider(ABC):
    """
    Basis provider that wraps any IndexedFunctionProvider.

    This class adds space-specific functionality on top of function providers:
    - Ensures the number of functions matches space dimension
    - Provides space-aware caching and validation
    - Implements convenient access patterns (indexing, iteration)

    All computation is lazy with caching for efficiency.
    """

    def __init__(self, space,
                 orthonormal: bool = False,
                 basis_type: Optional[str] = None):
        """
        Initialize basis provider with a function provider.

        Args:
            space: The function space that owns this provider
            function_provider: IndexedFunctionProvider for the basis functions
            orthonormal: True if the basis functions are orthonormal with
                        respect to the L² inner product
        """
        self.space = space
        self.orthonormal = orthonormal
        self.type = basis_type

    @abstractmethod
    def get_basis_function(self, index: int):
        """
        Get basis function for given index.
        """
        pass


class CustomBasisProvider(BasisProvider):
    """
    Basis provider with user-specified basis functions.

    Allows users to provide their own basis function arrays.
    """
    from pygeoinf.interval.function_providers import IndexedFunctionProvider

    def __init__(self, space, functions_provider: IndexedFunctionProvider,
                 orthonormal: bool = False,
                 basis_type: Optional[str] = None):
        """
        Initialize with function array.

        Args:
            space: The function space that owns this provider
            functions: Array of basis functions
            orthonormal: True if the basis functions are orthonormal with
                        respect to the L² inner product
            basis_type: String identifier for the type of basis functions
        """
        super().__init__(space, orthonormal, basis_type)
        self.functions_provider = functions_provider

    def get_basis_function(self, index: int):
        """Get basis function from the stored array."""
        if not (0 <= index < self.space.dim):
            raise IndexError(
                f"Function index {index} out of range "
                f"[0, {self.space.dim})"
            )
        return self.functions_provider.get_function_by_index(index)


class SpectrumProvider(ABC):
    """
    Spectrum provider that extends BasisProvider with eigenvalue information.

    Combines a function provider (for eigenfunctions) with an eigenvalue
    provider (for corresponding eigenvalues). This is used for spectral
    inner products in Sobolev spaces.

    All computation is lazy with caching for efficiency.
    """

    def __init__(self, space,
                 orthonormal: bool = False,
                 basis_type: Optional[str] = None):
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
        self.space = space
        self.orthonormal = orthonormal
        self.type = basis_type

    @abstractmethod
    def get_eigenvalue(self, index: int) -> float:
        """
        Get eigenvalue for given basis function index.
        """
        pass

    @abstractmethod
    def get_eigenfunction(self, index: int):
        """
        Get eigenfunction for given index.

        This is the same as get_basis_function but emphasizes that
        these are eigenfunctions of some operator.
        """
        pass


class CustomSpectrumProvider(SpectrumProvider):
    """
    Spectrum provider with user-specified basis functions and eigenvalues.

    Allows users to provide their own basis function and eigenvalue arrays.
    """
    from pygeoinf.interval.function_providers import IndexedFunctionProvider

    def __init__(
        self,
        space,
        functions_provider: IndexedFunctionProvider,
        eigenvalue_provider: EigenvalueProvider,
        orthonormal: bool = False,
        basis_type: Optional[str] = None
    ):
        """
        Initialize with function and eigenvalue arrays.

        Args:
            space: The function space
            functions_provider: IndexedFunctionProvider for eigenfunctions
            eigenvalue_provider: EigenvalueProvider for eigenvalues
            orthonormal: True if the basis functions are orthonormal with
                        respect to the L² inner product
            basis_type: String identifier for the type of basis functions
        """
        super().__init__(space, orthonormal, basis_type)
        self.functions_provider = functions_provider
        self.eigenvalue_provider = eigenvalue_provider

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue from the stored array."""
        if not (0 <= index < self.space.dim):
            raise IndexError(
                f"Eigenvalue index {index} out of range "
                f"[0, {self.space.dim})"
            )
        return self.eigenvalue_provider.get_eigenvalue(index)

    def get_eigenfunction(self, index: int):
        """Get eigenfunction from the stored array."""
        if not (0 <= index < self.space.dim):
            raise IndexError(
                f"Function index {index} out of range "
                f"[0, {self.space.dim})"
            )
        return self.functions_provider.get_function_by_index(index)

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


class SineEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider for sine eigenfunctions of the negative Laplacian.

    Computes eigenvalues λₖ = (kπ/L)² where L is the domain length.
    """

    def __init__(self, domain_length: float):
        """
        Initialize sine eigenvalue provider.

        Args:
            domain_length: Length of the domain (b - a)
        """
        self.domain_length = domain_length

    def get_eigenvalue(self, index: int) -> float:
        """Compute eigenvalue for the Laplacian operator."""
        # Sine functions start from k=1
        k = index + 1
        return (k * math.pi / self.domain_length) ** 2


class CosineEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider for cosine eigenfunctions of the negative Laplacian.

    Computes eigenvalues λₖ = (kπ/L)² where L is the domain length.
    The first eigenvalue (k=0) is zero corresponding to the constant mode.
    """

    def __init__(self, domain_length: float):
        """
        Initialize cosine eigenvalue provider.

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
            # For index > 0, k = index
            k = index
            return (k * math.pi / self.domain_length) ** 2


class ZeroEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider that returns zero for all indices.

    Useful for basis functions that aren't eigenfunctions of a specific
    operator, or when eigenvalue information isn't needed.
    """

    def get_eigenvalue(self, index: int) -> float:
        """Return zero eigenvalue."""
        return 0.0


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
        self._function_domain = function_domain
        self._boundary_conditions = boundary_conditions
        self._inverse = inverse
        self._alpha = alpha
        self._eigenvalue_cache = {}
        # robin cache (μ_k roots)
        self._robin_mu = []   # type: list[float]

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue for given index."""
        if index not in self._eigenvalue_cache:
            self._eigenvalue_cache[index] = self._compute_eigenvalue(index)
        return self._eigenvalue_cache[index]

    def _compute_eigenvalue(self, index: int) -> float:
        """Compute eigenvalue based on boundary conditions."""
        length = self._function_domain.b - self._function_domain.a

        if self._boundary_conditions.type == 'dirichlet':
            sine_provider = SineEigenvalueProvider(length)
            eigenval = sine_provider.get_eigenvalue(index)

        elif self._boundary_conditions.type == 'neumann':
            if self._inverse:
                index += 1  # jump over the first index to mimick restriction to mean-zero subspace
            cosine_provider = CosineEigenvalueProvider(length)
            eigenval = cosine_provider.get_eigenvalue(index)

        elif self._boundary_conditions.type == 'periodic':
            if self._inverse:
                index += 1  # jump over the first index to mimick restriction
            fourier_provider = FourierEigenvalueProvider(length)
            eigenval = fourier_provider.get_eigenvalue(index)

        elif self._boundary_conditions.type == 'mixed_dirichlet_neumann':
            # Eigenvalues: λₖ = ((k+1/2)π/L)² for k=0,1,2,...
            eigenval = (((index + 0.5) * np.pi) / length)**2

        elif self._boundary_conditions.type == 'mixed_neumann_dirichlet':
            # Eigenvalues: λₖ = ((k+1/2)π/L)² for k=0,1,2,...
            eigenval = (((index + 0.5) * np.pi) / length)**2

        # ---- general separated Robin
        elif self._boundary_conditions.type == 'robin':
            mu = self._robin_mu_at(index)   # compute & cache μ_k
            eigenval = mu * mu
        else:
            raise ValueError(f"Unknown boundary condition type: {self._boundary_conditions.type}")

        # Apply alpha scaling and inverse if needed
        if self._inverse:
            return 1.0 / (eigenval * self._alpha)
        else:
            return eigenval * self._alpha

    # ---------- ROBIN root finding  ----------
    def _robin_mu_at(self, k: int) -> float:
        # ensure we have μ_0,...,μ_k
        while len(self._robin_mu) <= k:
            # Pass the target index k (not the current length) to get the right bracket
            self._append_next_robin_root(k)
        return self._robin_mu[k]

    def _append_next_robin_root(self, target_index: int):
        a, b = self._function_domain.a, self._function_domain.b
        L = b - a
        alpha_0 = float(self._boundary_conditions.get_parameter('left_alpha'))
        beta_0 = float(self._boundary_conditions.get_parameter('left_beta'))
        alpha_L = float(self._boundary_conditions.get_parameter('right_alpha'))
        beta_L = float(self._boundary_conditions.get_parameter('right_beta'))

        # Use shared RobinRootFinder utility
        mu = RobinRootFinder.compute_robin_eigenvalue(
            target_index, alpha_0, beta_0, alpha_L, beta_L, L,
            tol=1e-12, maxit=100
        )
        self._robin_mu.append(mu)

    # Note: _bisect method removed - now using RobinRootFinder.bisect


class CustomEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider with user-specified eigenvalues.

    Allows users to provide their own eigenvalue arrays.
    """

    def __init__(self, eigenvalues: Union[np.ndarray, list]):
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


class LaplacianSpectrumProvider(SpectrumProvider):
    """
    Spectrum provider for Laplacian eigenfunctions.

    Combines a function provider (for eigenfunctions) with an eigenvalue
    provider (for corresponding eigenvalues of -Δ or (-Δ)⁻¹).
    """

    def __init__(
        self,
        space: "Union[Lebesgue, Sobolev]",
        boundary_conditions,
        alpha: float = 1,
        inverse: bool = False,
    ):
        """
        Initialize Laplacian spectrum provider.

        Args:
            space: The L2Space instance
            function_provider: IndexedFunctionProvider for eigenfunctions
            eigenvalue_provider: EigenvalueProvider for eigenvalues
            orthonormal: True if the basis functions are orthonormal with
                        respect to the L² inner product
            basis_type: String identifier for the type of basis functions
        """
        self._boundary_conditions = boundary_conditions
        self._inverse = inverse
        super().__init__(space, orthonormal=True, basis_type='')

        self._eigenvalue_provider = LaplacianEigenvalueProvider(
            space.function_domain,
            boundary_conditions,
            inverse, alpha,
        )
        self._function_provider = self._choose_basis()

    def get_eigenvalue(self, index: int) -> float:
        return self._eigenvalue_provider.get_eigenvalue(index)

    def get_eigenfunction(self, index: int):
        if self._boundary_conditions.type == 'dirichlet':
            return self._function_provider.get_function_by_index(index)
        elif self._boundary_conditions.type == 'neumann':
            if self._inverse:
                index += 1
            return self._function_provider.get_function_by_index(index)
        elif self._boundary_conditions.type == 'periodic':
            if self._inverse:
                index += 1
            return self._function_provider.get_function_by_index(index)
        elif self._boundary_conditions.type in ['mixed_dirichlet_neumann',
                                                'mixed_neumann_dirichlet',
                                                'robin']:
            return self._function_provider.get_function_by_index(index)

    def _choose_basis(self):
        # Import here to avoid circular imports
        from .function_providers import (
            SineFunctionProvider,
            CosineFunctionProvider,
            FourierFunctionProvider,
            MixedDNFunctionProvider,
            MixedNDFunctionProvider,
            RobinFunctionProvider
        )

        # Choose appropriate function provider based on boundary conditions
        if self._boundary_conditions.type == 'dirichlet':
            function_provider = SineFunctionProvider(self.space)
        elif self._boundary_conditions.type == 'neumann':
            function_provider = CosineFunctionProvider(self.space)
        elif self._boundary_conditions.type == 'periodic':
            function_provider = FourierFunctionProvider(self.space)
        elif self._boundary_conditions.type == 'mixed_dirichlet_neumann':
            function_provider = MixedDNFunctionProvider(self.space)
        elif self._boundary_conditions.type == 'mixed_neumann_dirichlet':
            function_provider = MixedNDFunctionProvider(self.space)
        elif self._boundary_conditions.type == 'robin':
            function_provider = RobinFunctionProvider(
                self.space,
                self._boundary_conditions
            )
        else:
            raise ValueError(f"Unsupported boundary condition: "
                             f"{self._boundary_conditions.type}")

        return function_provider
