"""
L² spaces on interval domains.

This module provides L² Hilbert spaces on intervals as the foundation
for more specialized function spaces like Sobolev spaces.
"""

import numpy as np
from typing import Optional

from pygeoinf.hilbert_space import HilbertSpace


from pygeoinf.operators import LinearOperator
from pygeoinf.forms import LinearForm
from pygeoinf.interval.functions import Function
from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval.providers import BasisProvider


class L2Space(HilbertSpace):
    """
    L² Hilbert space on an interval [a,b] with inner product
    ⟨u,v⟩ = ∫_a^b u(x)v(x) dx.

    This class provides the foundation for Sobolev spaces and manages:
    - L² inner product and norm via integration
    - Basis function creation and management (Fourier, etc.)
    - Function evaluation and coefficient transformations
    - Domain operations on intervals

    This serves as the base class for SobolevSpace.
    """

    # Default integration settings (can be overridden per-instance)
    default_integration_method: str = 'simpson'
    default_integration_npoints: int = 1000

    def __init__(
        self,
        dim: int,
        function_domain: IntervalDomain,
        /,
        *,
        basis_type: Optional[str] = None,
        basis_callables: Optional[list] = None,
        basis_provider: Optional[BasisProvider] = None,
        integration_method: Optional[str] = None,
        integration_npoints: Optional[int] = None,
    ):
        """
        Args:
            dim (int): Dimension of the space.
            function_domain (IntervalDomain): Domain object with optional
                boundary conditions.
            basis_type (str, optional): Type of basis functions to auto-gen
                ('fourier', 'hat', 'hat_homogeneous'). Creates a lazy provider.
            basis_callables (list, optional): List of callable functions that
                will be converted to L2Function basis functions on this space.
                Solves the circular dependency problem.
            basis_provider (BasisProvider, optional): Custom basis provider
                for basis functions.

        Note:
            Exactly one of basis_type, basis_callables, or basis_provider
            must be provided. If none are provided, defaults to 'fourier'.

            The three options correspond to:
            1. basis_type: Auto-generate standard basis (Fourier, hat, etc.)
            2. basis_callables: User-provided callable functions
            3. basis_provider: Custom lazy provider implementation
        """

        self._dim = dim
        self._function_domain = function_domain
        self._gaussian_measure = None

        # Make sure the space has a basis
        self._validate_basis_options(
            basis_type, basis_callables, basis_provider
        )

        # Initialize Gram matrix as None - computed lazily when needed
        self._gram_matrix = None

        # Integration settings (use class defaults if not provided)
        self.integration_method = (
            integration_method
            if integration_method is not None
            else self.default_integration_method
        )
        self.integration_npoints = (
            integration_npoints
            if integration_npoints is not None
            else self.default_integration_npoints
        )

        # Initialize the parent HilbertSpace with L² inner product
        super().__init__(
            dim,
            self._to_components,
            self._from_components,
            self.inner_product,
            self._default_to_dual,
            self._default_from_dual,
            copy=self._copy,
        )

    @property
    def dim(self):
        """Return the dimension of the space."""
        return self._dim

    @property
    def function_domain(self):
        """Return the IntervalDomain object for this space."""
        return self._function_domain

    @property
    def zero(self):
        """
        Create the zero function in this L2 space.

        Returns:
            Function: The zero function in this space
        """
        return Function(self, evaluate_callable=lambda x: np.zeros_like(x))

    @property
    def basis_functions(self):
        """Property to access basis functions with consistent interface."""
        if self._basis_functions is not None:
            return self._basis_functions
        else:
            # Use the lazy provider to get all basis functions as a list
            # This ensures consistent interface - always returns a list
            return self._basis_provider.get_all_basis_functions()

    def get_basis_function(self, index: int):
        """Get basis function by index, works with both lazy and explicit."""
        if index < 0 or index >= self.dim:
            raise IndexError(
                f"Basis index {index} out of range [0, {self.dim})"
            )
        if self._basis_functions is not None:
            return self._basis_functions[index]
        elif self._basis_provider is not None:
            return self._basis_provider.get_basis_function(index)
        else:
            raise RuntimeError(
                "Neither explicit nor lazy basis functions available"
            )

    @property
    def basis_provider(self):
        """Return the lazy basis provider for this space."""
        if self._basis_provider is not None:
            return self._basis_provider
        else:
            raise RuntimeError("No basis provider available for this space")

    @property
    def gram_matrix(self):
        """The Gram matrix of basis functions."""
        # Compute the Gram matrix if not already done
        if self._gram_matrix is None:
            self._compute_gram_matrix()
        return self._gram_matrix

    @property
    def basis_type(self):
        """The type of basis functions used."""
        return self._basis_type

    @property
    def provider_basis_type(self):
        """The detailed basis type from the provider."""
        if self._basis_provider is not None:
            return getattr(self._basis_provider, 'type', None)
        return None

    @property
    def is_orthonormal_basis(self):
        """Check if the basis is orthonormal."""
        if self._basis_provider is not None:
            return getattr(self._basis_provider, 'orthonormal', False)
        return False

    def inner_product(self, u, v, method: Optional[str] = None,
                      n_points: Optional[int] = None):
        """
        L² inner product: ⟨u,v⟩_L² = ∫_a^b u(x)v(x) dx

        Args:
            u, v: Functions in this L² space
            method (str): Numerical integration method ('simpson', 'trapezoid')
            n_points (int, optional): Number of points for numerical
                integration. If None, automatically selects sufficient points
                based on space dimension to resolve all basis functions
                (especially important for high-frequency Fourier modes).

        Returns:
            float: L² inner product

        For L² functions, we compute ⟨u,v⟩_L² = ∫_a^b u(x)v(x) dx through
        numerical integration, not pointwise evaluation (which is not
        mathematically well-defined for general L² functions).

        Note: For large dimensions (N >> 500), adaptive integration point
        selection is crucial to avoid numerical errors from under-sampling
        high-frequency basis functions.
        """
        # Prefer instance-level settings when method/n_points are omitted
        if method is None:
            method = self.integration_method

        if n_points is None:
            # Use the instance default if available
            if getattr(self, 'integration_npoints', None) is not None:
                n_points = self.integration_npoints
            else:
                # Fallback: adaptive selection based on basis type
                if self.provider_basis_type == 'fourier':
                    n_points = max(1000, min(50000, 4 * self.dim))
                else:
                    n_points = max(1000, min(20000, 2 * self.dim))

        # For L² functions, we need to be careful about pointwise operations
        # In practice, we work with smooth approximations
        product = u * v
        return product.integrate(method=method, n_points=n_points)

    def project(self, f: Function):
        """
        Project a function onto this L2 space.

        Args:
            f: Function to project (callable or L2Function)

        Returns:
            L2Function: The projection of f onto this space
        """

        # Compute coefficients via L2 inner products
        coeffs = self._to_components(f)
        return self._from_components(coeffs)

    def _compute_gram_matrix(self):
        """
        Compute the Gram matrix of the basis functions using L2 inner products.
        """
        if self.is_orthonormal_basis:
            # If orthonormal, return identity matrix
            self._gram_matrix = np.eye(self.dim)
        else:
            n = self.dim
            self._gram_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(i, n):  # Only compute upper triangle
                    # Get basis functions - works with both lazy and explicit
                    basis_i = self.get_basis_function(i)
                    basis_j = self.get_basis_function(j)

                    inner_prod = self.inner_product(basis_i, basis_j)
                    self._gram_matrix[i, j] = inner_prod
                    self._gram_matrix[j, i] = inner_prod  # Symmetric matrix

    def _to_components(self, u):
        """
        Convert a function to coefficients using inner products with basis
        functions.
        """
        # Compute right-hand side: b_i = <u, φ_i>_L²
        rhs = np.zeros(self.dim)
        for k in range(self.dim):
            basis_func = self.get_basis_function(k)
            rhs[k] = self.inner_product(u, basis_func)

        # For orthonormal basis, coefficients are just the inner products
        if self.is_orthonormal_basis:
            return rhs

        # For non-orthonormal basis, solve the linear system: G * c = rhs
        gram = self.gram_matrix
        if gram is None:
            raise ValueError("Gram matrix not computed")
        coeffs = np.linalg.solve(gram, rhs)
        return coeffs

    def _from_components(self, coeff):
        """
        Convert coefficients to a function using linear combination of
        basis functions.
        """
        coeff = np.asarray(coeff)
        if len(coeff) != self.dim:
            raise ValueError(f"Coefficients must have length {self.dim}")

        # Create Function directly with coefficients
        return Function(self, coefficients=coeff)

    # Default dual space mappings
    def _default_to_dual(self, u: Function):
        """Default mapping to dual space using Gram matrix."""
        return LinearForm(self, mapping=lambda v: self.inner_product(u, v))

    def _default_from_dual(self, up: LinearForm):
        """Default mapping from dual space using inverse Gram matrix."""
        dual_components = np.zeros(self.dim)
        for i in range(self.dim):
            basis_func = self.get_basis_function(i)
            dual_components[i] = up(basis_func)

        # For orthonormal basis, components are just the dual components
        if self.is_orthonormal_basis:
            components = dual_components
        else:
            # For non-orthonormal basis, solve with Gram matrix
            gram = self.gram_matrix
            if gram is None:
                raise ValueError("Gram matrix not computed")
            components = np.linalg.solve(gram, dual_components)

        return Function(
            self,
            coefficients=components,
        )

    def _validate_basis_options(
        self, basis_type, basis_callables, basis_provider
    ):
        # Count non-None options
        options = [basis_type, basis_callables, basis_provider]
        non_none = [opt for opt in options if opt is not None]

        if len(non_none) == 0:
            basis_type = 'fourier'
        elif len(non_none) > 1:
            raise ValueError(
                "Specify only one of basis_type, basis_callables, or "
                "basis_provider."
            )

        if basis_callables is not None:
            if len(basis_callables) != self.dim:
                raise ValueError(
                    f"basis_callables length ({len(basis_callables)}) "
                    f"must match dimension ({self.dim})"
                )
            self._basis_type = 'custom'
            self._basis_functions = [
                Function(self, evaluate_callable=cb)
                for cb in basis_callables
            ]
            self._basis_provider = None
            return

        if basis_provider is not None:
            self._basis_type = 'custom_provider'
            self._basis_functions = None
            self._basis_provider = basis_provider
            return

        # If here, use basis_type (either provided or defaulted)
        self._basis_type = basis_type
        self._basis_functions = None
        self._basis_provider = None
        if basis_type in ['fourier', 'hat', 'hat_homogeneous', 'sine']:
            from .providers import create_basis_provider
            self._basis_provider = create_basis_provider(self, basis_type)

    def _copy(self, x):
        """Custom copy implementation for Functions."""
        return Function(
            self,
            coefficients=self.to_components(x).copy(),
            name=getattr(x, 'name', None)
        )

    def __eq__(self, other):
        if not isinstance(other, L2Space):
            return False
        # Compare dimension and domain
        if self.function_domain != other.function_domain:
            return False

        return True
        # Compare basis type (or provider/callables if custom)

    # ========================================================================
    # Gaussian Measure Methods
    # ========================================================================

    @property
    def gaussian_measure(self):
        """
        Property to access the Gaussian measure for this L2 space.
        This is a convenience method that allows using the same interface
        for both KL and SPDE methods.
        """
        if self._gaussian_measure is None:
            raise ValueError("Gaussian measure not created yet. "
                             "Use create_gaussian_measure() first.")
        return self._gaussian_measure

    def gaussian_measure_kl(self, covariance: LinearOperator,
                            expectation: Optional[Function] = None,
                            kl_expansion: int = 100):
        """
        Create a Gaussian measure using Karhunen-Loève expansion.

        For a Gaussian measure μ = N(m, C), samples are generated as:
        X = m + Σⱼ₌₁^N √λⱼ ξⱼ φⱼ

        where {(λⱼ, φⱼ)} are eigenpairs of the covariance operator C,
        and ξⱼ ~ N(0,1) are i.i.d. standard normal variables.

        Args:
            covariance (LinearOperator): Covariance operator that should have
                a spectrum provider available. If the operator has a
                spectrum_provider attribute, it will be used for KL expansion.
                Otherwise, raises an error.
            expectation (Function, optional): Mean function. Defaults to zero.
            kl_expansion (int): Number of modes for KL expansion (default 100).

        Returns:
            GaussianMeasure: Gaussian measure with spectral covariance

        Example:
            >>> # Using LaplacianInverseOperator with spectrum
            >>> lap_inv = LaplacianInverseOperator(l2_space, bc)
            >>> measure = l2_space.gaussian_measure_kl(lap_inv)
            >>> sample = measure.sample()
        """
        from pygeoinf.gaussian_measure import GaussianMeasure

        # Check if the operator has a spectrum provider
        if not hasattr(covariance, 'spectrum_provider'):
            raise ValueError(
                f"Covariance operator {type(covariance).__name__} does not "
                f"have a spectrum_provider attribute. KL expansion requires "
                f"analytical spectrum information."
            )

        def sample_gaussian_kl():
            """
            Sample from Gaussian measure using Karhunen-Loève expansion.
            """
            # Ensure kl_expansion doesn't exceed space dimension
            effective_expansion = min(kl_expansion, self.dim)

            # Get eigenvalues using the spectrum provider
            eigenvalues = covariance.get_all_eigenvalues(n=effective_expansion)
            eigenvalues = np.array(eigenvalues)
            sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 1e-12))

            # Generate i.i.d. standard normal variables ξⱼ
            xi = np.random.randn(len(eigenvalues))

            # Compute coefficients: cⱼ = √λⱼ ξⱼ
            coefficients = sqrt_eigenvalues * xi

            # Create sample as linear combination: Σⱼ cⱼ φⱼ
            for j in range(len(eigenvalues)):
                basis_func = covariance.get_eigenfunction(j)
                if j == 0:
                    sample = coefficients[j] * basis_func
                else:
                    sample = sample + coefficients[j] * basis_func

            # Add expectation (mean)
            if expectation is not None:
                sample = sample + expectation

            return sample

        self._gaussian_measure = GaussianMeasure(
            covariance=covariance,
            expectation=expectation,
            sample=sample_gaussian_kl
        )

    def create_gaussian_measure(self, method='kl',
                                **kwargs):
        """
        Unified interface for creating Gaussian measures with different
        methods.

        Args:
            method (str): Sampling method - 'kl' for Karhunen-Loève,
                         'spde' for SPDE/precision operator
            kl_expansion (int): Number of modes for KL expansion (default 100)
            **kwargs: Method-specific arguments

        For method='kl':
            covariance (LinearOperator): Covariance operator that should have
                a spectrum provider available (e.g., LaplacianInverseOperator)
            expectation (Function, optional): Mean function
            kl_expansion (int): Number of modes for KL expansion (default 100).

        For method='spde':
            precision_operator (LinearOperator): Precision/covariance operator
            expectation (Function, optional): Mean function

        Returns:
            GaussianMeasure: Configured Gaussian measure

        Example:
            >>> # Karhunen-Loève method
            >>> lap_inv = LaplacianInverseOperator(l2_space, bc)
            >>> measure1 = l2_space.create_gaussian_measure(
            ...     method='kl',
            ...     covariance=lap_inv
            ... )
            >>>
            >>> # SPDE method
            >>> measure2 = l2_space.create_gaussian_measure(
            ...     method='spde',
            ...     precision_operator=laplacian_inv
            ... )
        """
        if method == 'kl':
            self.gaussian_measure_kl(**kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'kl' or 'spde'.")
