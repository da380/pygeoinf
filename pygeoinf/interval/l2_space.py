"""
L² spaces on interval domains.

This module provides L² Hilbert spaces on intervals as the foundation
for more specialized function spaces like Sobolev spaces.
"""

import numpy as np

from pygeoinf.hilbert_space import HilbertSpace
from pygeoinf.hilbert_space import LinearForm
from pygeoinf.interval.l2_functions import L2Function
from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval.providers import LazyBasisProvider


# Keep backward compatibility alias
LazyL2BasisProvider = LazyBasisProvider


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

    def __init__(
        self,
        dim: int,
        function_domain: IntervalDomain,
        /,
        *,
        basis_type: str = None,
        basis_callables: list = None,
        basis_provider: LazyBasisProvider = None,
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
            basis_provider (LazyBasisProvider, optional): Custom lazy provider
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

        # Validate that exactly one basis option is provided
        basis_options = [basis_type, basis_callables, basis_provider]
        non_none_count = sum(1 for opt in basis_options if opt is not None)

        if non_none_count == 0:
            # Default to fourier if nothing specified
            basis_type = 'fourier'
        elif non_none_count > 1:
            raise ValueError(
                "Exactly one of basis_type, basis_callables, or "
                "basis_provider must be provided, but multiple were given"
            )

        # Handle the three basis options
        if basis_callables is not None:
            if len(basis_callables) != dim:
                raise ValueError(
                    f"basis_callables length ({len(basis_callables)}) "
                    f"must match dimension ({dim})"
                )
            self._basis_type = 'custom'
            # Convert callables to L2Function objects after space is created
            self._pending_callables = basis_callables
            self._basis_functions = None  # Will be created after init
            self._basis_provider = None
        elif basis_provider is not None:
            self._basis_type = 'custom_provider'
            self._basis_functions = None
            self._basis_provider = basis_provider
            self._pending_callables = None
        else:
            # basis_type is specified (or defaulted to 'fourier')
            self._basis_type = basis_type
            self._basis_functions = None
            self._basis_provider = None  # Will be created below
            self._pending_callables = None

        # Create basis provider for standard basis types
        if basis_type in ['fourier', 'hat', 'hat_homogeneous']:
            self._basis_provider = LazyBasisProvider(self, basis_type)

        # Initialize Gram matrix as None - computed lazily when needed
        self._gram_matrix = None

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

        # Now that the space is fully initialized, convert pending callables
        # to L2Function objects (this solves the circular dependency)
        if hasattr(self, '_pending_callables') and self._pending_callables:
            from .l2_functions import L2Function
            self._basis_functions = []
            for i, callable_func in enumerate(self._pending_callables):
                l2_func = L2Function(self, evaluate_callable=callable_func)
                self._basis_functions.append(l2_func)
            # Clear the pending callables
            self._pending_callables = None

    @property
    def dim(self):
        """Return the dimension of the space."""
        return self._dim

    @property
    def function_domain(self):
        """Return the IntervalDomain object for this space."""
        return self._function_domain

    def get_basis_function(self, index: int):
        """Get basis function by index, works with both lazy and explicit."""
        if self._basis_functions is not None:
            return self._basis_functions[index]
        elif self._basis_provider is not None:
            return self._basis_provider.get_basis_function(index)
        else:
            raise RuntimeError(
                "Neither explicit nor lazy basis functions available"
            )

    @property
    def basis_functions(self):
        """Property to access basis functions with consistent interface."""
        if self._basis_functions is not None:
            return self._basis_functions
        else:
            # Use the lazy provider to get all basis functions as a list
            # This ensures consistent interface - always returns a list
            return self._basis_provider.get_all_basis_functions()

    def basis_function(self, i):
        """Return the ith basis function directly."""
        if i < 0 or i >= self.dim:
            raise IndexError(f"Basis index {i} out of range [0, {self.dim})")

        return self.get_basis_function(i)

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
        if self._gram_matrix is None:
            self._compute_gram_matrix()
        return self._gram_matrix

    @property
    def basis_type(self):
        """The type of basis functions used."""
        return self._basis_type

    def inner_product(self, u, v):
        """
        L² inner product: ⟨u,v⟩_L² = ∫_a^b u(x)v(x) dx

        Args:
            u, v: Functions in this L² space

        Returns:
            float: L² inner product

        For L² functions, we compute ⟨u,v⟩_L² = ∫_a^b u(x)v(x) dx through
        numerical integration, not pointwise evaluation (which is not
        mathematically well-defined for general L² functions).
        """
        # For L² functions, we need to be careful about pointwise operations
        # In practice, we work with smooth approximations
        product = u * v
        return product.integrate()

    def _compute_gram_matrix(self):
        """
        Compute the Gram matrix of the basis functions using L2 inner products.
        """
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

    def project(self, f):
        """
        Project a function onto this L2 space.

        Args:
            f: Function to project (callable or L2Function)

        Returns:
            L2Function: The projection of f onto this space
        """
        if callable(f):
            # Create L2Function from callable
            func = L2Function(self, evaluate_callable=f)
        else:
            func = f

        # Compute coefficients via L2 inner products
        coeffs = self._to_components(func)
        return self._from_components(coeffs)

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

        # Solve the linear system: G * c = rhs
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

        # Create L2Function directly with coefficients
        return L2Function(self, coefficients=coeff)

    # Default dual space mappings
    def _default_to_dual(self, u: L2Function):
        """Default mapping to dual space using Gram matrix."""
        return LinearForm(self, mapping=lambda v: self.inner_product(u, v))

    def _default_from_dual(self, up: LinearForm):
        """Default mapping from dual space using inverse Gram matrix."""
        dual_components = np.zeros(self.dim)
        for i in range(self.dim):
            basis_func = self.get_basis_function(i)
            dual_components[i] = up(basis_func)

        gram = self.gram_matrix
        if gram is None:
            raise ValueError("Gram matrix not computed")
        components = np.linalg.solve(gram, dual_components)
        return L2Function(
            self,
            coefficients=components,
        )

    def _copy(self, x):
        """Custom copy implementation for L2Functions."""
        return L2Function(
            self,
            coefficients=self.to_components(x).copy(),
            name=getattr(x, 'name', None)
        )
