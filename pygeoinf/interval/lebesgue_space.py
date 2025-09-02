"""
Lebesgue spaces on interval domains.

This module provides L² and more general Lebesgue spaces on intervals as proper
Hilbert spaces that inherit from the main pygeoinf HilbertSpace abstract base class.
This is a refactoring of the previous l2_space.py to properly integrate with the
pygeoinf ecosystem.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Any, TYPE_CHECKING, Union

from pygeoinf.hilbert_space import HilbertSpace

if TYPE_CHECKING:
    from pygeoinf.interval.functions import Function
    from pygeoinf.interval.interval_domain import IntervalDomain
    from pygeoinf.interval.providers import BasisProvider
else:
    # Import for runtime use
    from pygeoinf.interval.functions import Function


class Lebesgue(HilbertSpace):
    """
    Lebesgue space L² on an interval [a,b] with inner product
    ⟨u,v⟩ = ∫_a^b u(x)v(x) dx.

    This class properly inherits from the pygeoinf HilbertSpace abstract base class,
    using Function objects as the Vector type. It provides:
    - L² inner product and norm via integration
    - Basis function management (Fourier, hat functions, etc.)
    - Function evaluation and coefficient transformations
    - Proper dual space relationships via Riesz representation
    - Full integration with pygeoinf operators and linear forms

    The mathematical foundation is the Lebesgue space L²([a,b]) with the
    standard inner product defined by integration.
    """

    def __init__(
        self,
        dim: int,
        function_domain: "IntervalDomain",
        /,
        *,
        basis: Optional[Union[str, list]] = None,
    ):
        """
        Initialize a Lebesgue space L²([a,b]).

        Args:
            dim: Dimension of the finite-dimensional approximation space
            function_domain: IntervalDomain object specifying [a,b] and boundary conditions
            basis: Basis specification, can be:
                - str: 'fourier', 'hat', 'sine', etc. (auto-generated but not fully implemented)
                - str: 'none' (creates baseless space for temporary use)
                - list: [func1, func2, ...] (custom callable functions)
                - None: 'none' basis (default - create baseless space)

        Examples:
            >>> # Create baseless space (typical workflow)
            >>> space = Lebesgue(dim=50, function_domain=domain)
            >>> space = Lebesgue(dim=50, function_domain=domain, basis='none')

            >>> # Custom functions (immediate setup)
            >>> space = Lebesgue(dim=3, function_domain=domain,
            ...                  basis=[lambda x: 1, lambda x: x, lambda x: x**2])

            >>> # Standard basis types (when implemented)
            >>> space = Lebesgue(dim=50, function_domain=domain, basis='fourier')
        """
        self._dim = dim
        self._function_domain = function_domain

        # Integration settings (internal variables with defaults)
        self._integration_method = 'simpson'
        self._integration_npoints = 1000

        # Initialize basis (simplified single-parameter approach)
        self._initialize_basis(basis or 'none')

        # Cached computations
        self._metric = None
        self._inverse_metric = None

    # ================================================================
    # Abstract methods that MUST be implemented for HilbertSpace
    # ================================================================

    @property
    def dim(self) -> int:
        """The finite dimension of the space."""
        return self._dim

    @property
    def metric(self) -> np.ndarray:
        """The metric tensor (Gram matrix) G[i,j] = ⟨φᵢ,φⱼ⟩_L² of the basis functions."""
        if self._metric is None:
            self._compute_metric()
        return self._metric

    @property
    def inverse_metric(self) -> np.ndarray:
        """The inverse metric tensor."""
        if self._inverse_metric is None:
            metric = self.metric  # This triggers computation if needed
            self._inverse_metric = np.linalg.inv(metric)
        return self._inverse_metric

    def to_dual(self, x: "Function") -> Any:
        """
        Maps a function to its canonical dual vector (a linear functional).

        This implements the Riesz representation using the metric tensor:
        for u ∈ L², the dual element is computed as G * c_u where G is the
        metric (Gram matrix) and c_u are the coefficients of u.

        Args:
            x: A Function in this Lebesgue space

        Returns:
            LinearForm representing the dual element
        """
        from pygeoinf.linear_forms import LinearForm

        # Get coefficient representation
        components = self.to_components(x)

        # Apply metric tensor (Gram matrix) to get dual components
        dual_components = self.metric @ components

        # Create LinearForm directly from dual components
        return LinearForm(self, components=dual_components)

    def from_dual(self, xp: Any) -> "Function":
        """
        Maps a dual vector back to its representative in the primal space.

        This is the inverse Riesz map: given a linear functional, find the
        unique function u such that the functional is v ↦ ⟨u,v⟩_L².

        Args:
            xp: A LinearForm from the dual space

        Returns:
            Function representing the primal element
        """
        from pygeoinf.linear_forms import LinearForm

        if not isinstance(xp, LinearForm):
            raise TypeError("Expected LinearForm for dual element")

        # Get dual components from LinearForm
        dual_components = xp.components

        # Apply inverse metric to get primal components
        primal_components = self.inverse_metric @ dual_components

        return self.from_components(primal_components)

    def to_components(self, x: "Function") -> np.ndarray:
        """
        Maps a function to its representation as basis coefficients.

        For a function u = Σᵢ cᵢφᵢ, this returns the coefficient vector [c₁, c₂, ..., cₙ].

        This method handles two cases:
        1. If x already has coefficients (is in this space), return them
        2. If x is defined by a callable, project using L² integration

        Args:
            x: A Function in this space

        Returns:
            NumPy array of basis coefficients
        """
        # Case 1: Function already has coefficients in this space
        if (hasattr(x, 'coefficients') and x.coefficients is not None
                and hasattr(x, 'space') and x.space is self):
            return x.coefficients.copy()

        # Case 2: Project onto basis using continuous L² inner product
        # For non-orthonormal basis, solve the linear system: G c = b
        # where G[i,j] = ⟨φᵢ, φⱼ⟩_L² and b[i] = ⟨x, φᵢ⟩_L²

        # Compute right-hand side: b[i] = ⟨x, φᵢ⟩_L²
        rhs = np.zeros(self.dim)
        for i in range(self.dim):
            basis_func = self.get_basis_function(i)
            rhs[i] = self._continuous_l2_inner_product(x, basis_func)

        # Get the metric tensor (Gram matrix)
        metric = self.metric

        # Solve G c = b for the coefficients c
        coeffs = np.linalg.solve(metric, rhs)

        return coeffs

    def from_components(self, c: np.ndarray) -> "Function":
        """
        Maps basis coefficients back to a function in the space.

        Given coefficients [c₁, c₂, ..., cₙ], constructs u = Σᵢ cᵢφᵢ.

        This is the fundamental operation: coefficients ARE the representation
        of functions in this finite-dimensional space.

        Args:
            c: NumPy array of basis coefficients

        Returns:
            Function constructed from the coefficients
        """
        c = np.asarray(c)
        if len(c) != self.dim:
            raise ValueError(f"Coefficients must have length {self.dim}")

        # Create Function directly with coefficients - this is fundamental!
        return Function(self, coefficients=c.copy())

    def __eq__(self, other: object) -> bool:
        """
        Checks equality between two Lebesgue spaces.

        Two Lebesgue spaces are equal if they have the same dimension,
        function domain, and basis configuration.

        Args:
            other: Another object to compare with

        Returns:
            True if the spaces are mathematically equivalent
        """
        if not isinstance(other, Lebesgue):
            return False

        # Must have same dimension
        if self.dim != other.dim:
            return False

        # Must have same function domain
        if self.function_domain != other.function_domain:
            return False

        # TODO: Could also compare basis types/providers for stricter equality
        # For now, same dimension + same domain = equal spaces
        return True

    # ================================================================
    # Overridden methods
    # ================================================================

    def multiply(self, a: float, x: "Function") -> "Function":
        """
        Compute scalar multiplication a*x.

        This overrides the base class to ensure coefficient consistency.
        """
        if hasattr(x, 'coefficients') and x.coefficients is not None:
            # Direct coefficient scaling for functions with coefficients
            new_coefficients = a * x.coefficients
            return Function(self, coefficients=new_coefficients.copy())
        else:
            # Fall back to default behavior for functions without coefficients
            return a * x

    def add(self, x: "Function", y: "Function") -> "Function":
        """
        Compute vector addition x + y.

        This overrides the base class to ensure coefficient consistency.
        """
        if (hasattr(x, 'coefficients') and x.coefficients is not None and
            hasattr(y, 'coefficients') and y.coefficients is not None):
            # Direct coefficient addition for functions with coefficients
            new_coefficients = x.coefficients + y.coefficients
            return Function(self, coefficients=new_coefficients.copy())
        else:
            # Fall back to default behavior
            return x + y

    def ax(self, a: float, x: "Function") -> None:
        """
        Performs in-place scaling x := a*x.

        Since Function objects don't support in-place operations,
        we modify the coefficients directly.
        """
        if hasattr(x, 'coefficients') and x.coefficients is not None:
            x.coefficients *= a
        else:
            raise ValueError("Cannot perform in-place operation on function without coefficients")

    def axpy(self, a: float, x: "Function", y: "Function") -> None:
        """
        Performs in-place operation y := y + a*x.

        Since Function objects don't support in-place operations,
        we modify the coefficients directly.
        """
        if (hasattr(y, 'coefficients') and y.coefficients is not None and
            hasattr(x, 'coefficients') and x.coefficients is not None):
            y.coefficients += a * x.coefficients
        else:
            raise ValueError("Cannot perform in-place operation on functions without coefficients")

    # ================================================================
    # Additional methods specific to function spaces
    # ================================================================

    @property
    def function_domain(self) -> "IntervalDomain":
        """The interval domain [a,b] for this space."""
        return self._function_domain

    # ================================================================
    # Additional methods for numerical integration
    # ================================================================

    @property
    def integration_method(self) -> str:
        """The numerical integration method used for inner products."""
        return self._integration_method

    @integration_method.setter
    def integration_method(self, method: str) -> None:
        """Set the numerical integration method."""
        if not isinstance(method, str):
            raise TypeError("Integration method must be a string")

        # Optional: Add validation for supported methods
        supported_methods = {'simpson', 'trapz', 'gauss', 'adaptive'}
        if method not in supported_methods:
            raise ValueError(f"Unsupported integration method '{method}'. "
                             f"Supported methods: {supported_methods}")

        self._integration_method = method
        # Clear cached matrices when integration settings change
        self._metric = None
        self._inverse_metric = None

    @property
    def integration_npoints(self) -> int:
        """The number of points used for numerical integration."""
        return self._integration_npoints

    @integration_npoints.setter
    def integration_npoints(self, npoints: int) -> None:
        """Set the number of integration points."""
        if not isinstance(npoints, int):
            raise TypeError("Number of integration points must be an integer")
        if npoints <= 0:
            raise ValueError("Number of integration points must be positive")

        self._integration_npoints = npoints
        # Clear cached matrices when integration settings change
        self._metric = None
        self._inverse_metric = None

    # ================================================================
    # Additional methods for basis function management
    # ================================================================
    def _initialize_basis(self, basis):
        """
        Initialize basis from simplified single parameter.

        Args:
            basis: str or list of callables
        """
        if isinstance(basis, str):
            if basis == 'none':
                # Create a baseless space - useful for temporary spaces when creating BasisProviders
                self._basis_type = 'none'
                self._basis_functions = None
                self.basis_provider = None
                self._use_basis_provider = False
            else:
                # String-based standard basis types using provider factory
                from pygeoinf.interval.providers import create_basis_provider
                try:
                    basis_provider = create_basis_provider(self, basis)
                    self._basis_type = basis
                    self._basis_functions = None
                    self.basis_provider = basis_provider
                    self._use_basis_provider = True
                except ValueError as e:
                    raise ValueError(f"Unsupported basis type '{basis}': {e}")

        elif isinstance(basis, list):
            # List of callables/functions provided directly - transform to Functions
            if len(basis) != self._dim:
                raise ValueError(
                    f"Number of basis functions ({len(basis)}) "
                    f"must match dimension ({self._dim})"
                )
            self._basis_type = 'direct_functions'
            # Transform callables to Function instances immediately
            self._basis_functions = [
                self._create_function_from_callable(callable_func)
                for callable_func in basis
            ]
            self.basis_provider = None
            self._use_basis_provider = False

        else:
            # Invalid input type
            raise TypeError(
                f"basis must be a string or list of callables, got {type(basis)}"
            )

    def _create_function_from_callable(self, callable_func):
        """Create Function from callable."""
        return Function(self, evaluate_callable=callable_func)

    def set_basis_provider(self, basis_provider: "BasisProvider"):
        """
        Set a BasisProvider after space creation.

        This is useful when creating a temporary baseless space to construct
        a BasisProvider that requires the space as an argument.

        Args:
            basis_provider: The BasisProvider to use for this space
        """
        self.basis_provider = basis_provider
        self._use_basis_provider = True
        self._basis_type = getattr(basis_provider, 'type', 'custom_provider')
        self._basis_functions = None
        # Clear cached computations since basis changed
        self._metric = None
        self._inverse_metric = None

    @property
    def basis_functions(self) -> list["Function"]:
        """List of basis functions for this space."""
        if self._basis_type == 'none':
            raise RuntimeError("No basis functions available - space is baseless")

        if self._use_basis_provider:
            # Use BasisProvider for lazy evaluation (set via set_basis_provider)
            if self.basis_provider is None:
                raise RuntimeError("No basis provider available")
            if hasattr(self.basis_provider, 'get_basis_function'):
                return [self.basis_provider.get_basis_function(i)
                        for i in range(self.dim)]
            else:
                raise RuntimeError("BasisProvider missing get_basis_function")
        else:
            # Direct access to stored Function instances (from callables list)
            if self._basis_functions is None:
                raise RuntimeError("No basis functions available")
            return self._basis_functions

    def get_basis_function(self, index: int) -> "Function":
        """Get the i-th basis function."""
        if self._basis_type == 'none':
            raise RuntimeError("No basis functions available - space is baseless")

        if self._use_basis_provider:
            # Use BasisProvider
            if self.basis_provider is None:
                raise RuntimeError("No basis provider available")
            if hasattr(self.basis_provider, 'get_basis_function'):
                return self.basis_provider.get_basis_function(index)
            else:
                raise RuntimeError("BasisProvider missing get_basis_function")
        else:
            # Direct access to stored Function instances
            if not (0 <= index < self.dim):
                raise IndexError(f"Index {index} out of range [0, {self.dim})")

            if self._basis_functions is None:
                raise RuntimeError("No basis functions available")
            return self._basis_functions[index]

    # ================================================================
    # Function space operations
    # ================================================================

    def project(self, f: "Function") -> "Function":
        """
        Project a function onto this finite-dimensional space.

        The L² projection of f onto this space is the function in this space
        that minimizes ||f - g||_{L²} over all g in the space.

        This is computed by:
        1. Computing coefficients via to_components (which projects onto basis)
        2. Reconstructing the function via from_components

        Args:
            f: Function to project (can be from different space or callable)

        Returns:
            The L² projection of f onto this space
        """
        # Use to_components to project onto basis, then reconstruct
        coefficients = self.to_components(f)
        return self.from_components(coefficients)

    # ================================================================
    # Private helper methods
    # ================================================================

    def _continuous_l2_inner_product(self, u: "Function", v: "Function") -> float:
        """
        Compute the continuous L² inner product via numerical integration.

        This is different from the discrete inner product of the finite-dimensional
        space. It's used for projecting external functions onto the basis.

        ⟨u, v⟩_L² = ∫_a^b u(x) v(x) dx

        Args:
            u, v: Functions to integrate

        Returns:
            The continuous L² inner product
        """
        # Use Function's built-in multiplication and integration!
        product = u * v  # Function.__mul__ handles this properly
        return product.integrate(
            method=self.integration_method,
            n_points=self.integration_npoints
        )

    def _compute_metric(self):
        """
        Compute and cache the metric tensor (Gram matrix) using continuous L² inner products.

        G[i,j] = ⟨φᵢ, φⱼ⟩_L² = ∫_a^b φᵢ(x) φⱼ(x) dx

        This is computed once and cached for efficiency.
        """
        if self._metric is not None:
            return

        # Check if we have basis functions available
        if self._basis_type == 'none':
            raise RuntimeError(
                "Cannot compute metric: no basis functions available. "
                "Set a basis or BasisProvider first."
            )

        n = self.dim
        self._metric = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):  # Only compute upper triangle
                basis_i = self.get_basis_function(i)
                basis_j = self.get_basis_function(j)

                # Use continuous L² inner product for metric computation
                inner_prod = self._continuous_l2_inner_product(basis_i, basis_j)
                self._metric[i, j] = inner_prod
                self._metric[j, i] = inner_prod  # Symmetric
