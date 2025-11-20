"""Bessel-Sobolev operators for interval domains."""

from typing import Optional, Literal, List

import numpy as np

from pygeoinf.linear_operators import LinearOperator

from .base import SpectralOperator
from ..lebesgue_space import Lebesgue
from ..functions import Function
from ..configs import IntegrationConfig
from ..fast_spectral_integration import (
    fast_spectral_coefficients,
    create_uniform_samples,
)
from .spectral_helpers import (
    build_eigenfunction_expansion,
    compute_spectral_coefficients_fast,
    compute_spectral_coefficients_slow,
    validate_eigenvalue,
)


class BesselSobolev(LinearOperator):
    """
    Fast Bessel potential operator using fast transforms for coefficient computation.

    This is a drop-in replacement for BesselSobolev that uses fast transforms
    (DST/DCT/DFT) instead of numerical integration when the underlying spectral
    operator L uses Laplacian eigenfunctions with homogeneous boundary conditions.

    Supports:
    - Dirichlet BC: Uses DST (Discrete Sine Transform)
    - Neumann BC: Uses DCT (Discrete Cosine Transform)
    - Periodic BC: Uses DFT (Discrete Fourier Transform)

    For non-Laplacian operators, falls back to the original slow method.
    """

    def __init__(
        self, domain: Lebesgue, codomain: Lebesgue, k: float, s: float,
                 L: SpectralOperator, dofs: Optional[int] = None,
                 n_samples: int = 1024, use_fast_transforms: bool = True,
                 integration_config: IntegrationConfig = IntegrationConfig(method='simpson', n_points=100)):
        """
        Initialize the fast Bessel potential operator.

        Args:
            domain: Lebesgue space (input)
            codomain: Lebesgue space (output)
            k: Bessel parameter k²
            s: Sobolev order s
            L: Spectral operator (should be Laplacian for fast transforms)
            dofs: Number of degrees of freedom
            n_samples: Number of samples for fast transform (should be >= dofs)
            use_fast_transforms: If False, fall back to slow numerical integration
            integration_config: Integration configuration
        """
        self._domain = domain
        self._codomain = codomain
        self._L = L
        self._k = k
        self._s = s
        self._dofs = dofs if dofs is not None else domain.dim
        self._n_samples = max(n_samples, self._dofs)  # Ensure enough samples
        self._use_fast_transforms = use_fast_transforms

        # Store integration config
        self.integration = integration_config

        # Detect if we can use fast transforms
        self._boundary_condition = self._detect_boundary_condition()
        self._can_use_fast_transforms = (
            self._use_fast_transforms and
            self._boundary_condition is not None
        )

        super().__init__(domain, codomain, self._apply)

    def _detect_boundary_condition(self) -> Optional[Literal[
        'dirichlet', 'neumann', 'periodic',
        'mixed_dirichlet_neumann', 'mixed_neumann_dirichlet'
    ]]:
        """
        Detect boundary condition type from the spectral operator.

        Returns:
            Boundary condition type if detected, None if unknown/unsupported
        """
        # Try to access the underlying spectrum provider
        if hasattr(self._L, '_spectrum_provider'):
            provider = self._L._spectrum_provider
            if hasattr(provider, 'type'):
                if provider.type == 'sine_dirichlet':
                    return 'dirichlet'
                elif provider.type == 'cosine_neumann':
                    return 'neumann'
                elif provider.type == 'fourier_periodic':
                    return 'periodic'

        # Try to access boundary conditions directly
        if hasattr(self._L, '_boundary_conditions'):
            bc = self._L._boundary_conditions
            if hasattr(bc, 'type'):
                # Support all fast-transform-capable BCs including mixed
                if bc.type in ['dirichlet', 'neumann', 'periodic',
                               'mixed_dirichlet_neumann',
                               'mixed_neumann_dirichlet']:
                    return bc.type

        return None

    def _apply(self, f: Function) -> Function:
        """Apply the Bessel potential operator to a function."""
        if self._can_use_fast_transforms:
            return self._apply_fast(f)
        else:
            return self._apply_slow(f)

    def _apply_fast(self, f: Function) -> Function:

        """Apply using fast transforms."""
        # Get domain information
        domain_interval = self._domain.function_domain
        domain_tuple = (domain_interval.a, domain_interval.b)
        domain_length = domain_interval.b - domain_interval.a

        # Create uniform samples of the input function
        f_samples = create_uniform_samples(
            f, domain_tuple, self._n_samples, self._boundary_condition
        )

        # Compute all spectral coefficients at once using fast transforms
        coefficients = fast_spectral_coefficients(
            f_samples, self._boundary_condition, domain_length, self._dofs
        )

        # Use spectral helpers for eigenfunction expansion with Bessel scaling
        terms = compute_spectral_coefficients_fast(
            self._L, f, coefficients,
            scale_func=lambda i, ev: (self._k**2 + ev)**(self._s / 2)
        )
        return build_eigenfunction_expansion(
            terms, self._domain, self._codomain
        )

    def _apply_slow(self, f: Function) -> Function:
        """Apply using slow numerical integration (fallback)."""
        terms = compute_spectral_coefficients_slow(
            self._L, f, self._dofs,
            self.integration.method,
            self.integration.n_points,
            scale_func=lambda i, ev: (self._k**2 + ev)**(self._s / 2)
        )
        return build_eigenfunction_expansion(
            terms, self._domain, self._codomain
        )

    def get_eigenfunction(self, index: int) -> Function:
        """Get the eigenfunction at a specific index."""
        return self._L.get_eigenfunction(index)

    def get_eigenvalue(self, index: int) -> float:
        """Get the eigenvalue at a specific index."""
        eigval = self._L.get_eigenvalue(index)
        if eigval is None:
            raise ValueError(f"Eigenvalue not available for index {index}")
        if eigval < 0:
            raise ValueError(f"Negative eigenvalue {eigval} at index {index}")
        return (self._k**2 + eigval)**(self._s / 2)


class BesselSobolevInverse(LinearOperator):
    """
    Fast inverse Bessel potential operator using fast transforms.

    This is a drop-in replacement for BesselSobolevInverse that uses fast transforms
    instead of numerical integration for coefficient computation.
    """

    def __init__(self, domain: Lebesgue, codomain: Lebesgue, k: float, s: float,
                 L: SpectralOperator, dofs: Optional[int] = None,
                 n_samples: int = 1024, use_fast_transforms: bool = True):
        """
        Initialize the fast inverse Bessel potential operator.

        Args:
            domain: Lebesgue space (input)
            codomain: Lebesgue space (output)
            k: Bessel parameter k²
            s: Sobolev order s
            L: Spectral operator (should be Laplacian for fast transforms)
            dofs: Number of degrees of freedom
            n_samples: Number of samples for fast transform
            use_fast_transforms: If False, fall back to slow method
        """
        self._domain = domain
        self._codomain = codomain
        self._L = L
        self._k = k
        self._s = s
        self._dofs = dofs
        self._n_samples = max(n_samples, self._dofs)
        self._use_fast_transforms = use_fast_transforms

        # Detect boundary condition
        self._boundary_condition = self._detect_boundary_condition()
        self._can_use_fast_transforms = (
            self._use_fast_transforms and
            self._boundary_condition is not None
        )

        super().__init__(domain, codomain, self._apply)

    def _detect_boundary_condition(self) -> Optional[Literal[
        'dirichlet', 'neumann', 'periodic',
        'mixed_dirichlet_neumann', 'mixed_neumann_dirichlet'
    ]]:
        """Detect boundary condition type from the spectral operator."""
        # Same logic as FastBesselSobolev
        if hasattr(self._L, '_spectrum_provider'):
            provider = self._L._spectrum_provider
            if hasattr(provider, 'type'):
                if provider.type == 'sine_dirichlet':
                    return 'dirichlet'
                elif provider.type == 'cosine_neumann':
                    return 'neumann'
                elif provider.type == 'fourier_periodic':
                    return 'periodic'

        if hasattr(self._L, '_boundary_conditions'):
            bc = self._L._boundary_conditions
            if hasattr(bc, 'type'):
                # Support all fast-transform-capable BCs including mixed
                if bc.type in ['dirichlet', 'neumann', 'periodic',
                               'mixed_dirichlet_neumann',
                               'mixed_neumann_dirichlet']:
                    return bc.type

        return None

    def _apply(self, f: Function) -> Function:
        """Apply the inverse Bessel potential operator to a function."""
        if self._can_use_fast_transforms:
            return self._apply_fast(f)
        else:
            return self._apply_slow(f)

    def _apply_fast(self, f: Function) -> Function:
        """Apply using fast transforms."""
        # Get domain information
        domain_interval = self._domain.function_domain
        domain_tuple = (domain_interval.a, domain_interval.b)
        domain_length = domain_interval.b - domain_interval.a

        # Create uniform samples of the input function
        f_samples = create_uniform_samples(
            f, domain_tuple, self._n_samples, self._boundary_condition
        )

        # Compute all spectral coefficients at once
        coefficients = fast_spectral_coefficients(
            f_samples, self._boundary_condition, domain_length, self._dofs
        )

        # Use spectral helpers for eigenfunction expansion with inverse Bessel scaling
        terms = compute_spectral_coefficients_fast(
            self._L, f, coefficients,
            scale_func=lambda i, ev: (self._k**2 + ev)**(-self._s / 2)
        )
        return build_eigenfunction_expansion(
            terms, self._domain, self._codomain
        )

    def _apply_slow(self, f: Function) -> Function:
        """Apply using slow numerical integration (fallback)."""
        terms = compute_spectral_coefficients_slow(
            self._L, f, self._dofs,
            'simpson',
            10000,
            scale_func=lambda i, ev: (self._k**2 + ev)**(-self._s / 2)
        )
        return build_eigenfunction_expansion(
            terms, self._domain, self._codomain
        )

    def get_eigenfunction(self, index: int) -> Function:
        """Get the eigenfunction at a specific index."""
        return self._L.get_eigenfunction(index)

    def get_eigenvalue(self, index: int) -> float:
        """Get the eigenvalue at a specific index."""
        eigval = self._L.get_eigenvalue(index)
        if eigval is None:
            raise ValueError(f"Eigenvalue not available for index {index}")
        if eigval < 0:
            raise ValueError(f"Negative eigenvalue {eigval} at index {index}")
        return (self._k**2 + eigval)**(-self._s / 2)

    def get_eigenvalues(self, indices: List[int]) -> List[float]:
        """Get multiple eigenvalues at specified indices."""
        return [self.get_eigenvalue(i) for i in indices]
