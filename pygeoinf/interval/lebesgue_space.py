"""
Lebesgue spaces on interval domains.

This module provides L² and more general Lebesgue spaces on intervals as proper
Hilbert spaces that inherit from the main pygeoinf HilbertSpace abstract base class.
This is a refactoring of the previous l2_space.py to properly integrate with the
pygeoinf ecosystem.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Optional, Any, TYPE_CHECKING, Union, List, Callable

from pygeoinf import HilbertSpace, HilbertSpaceDirectSum, LinearForm
from pygeoinf.interval.linear_form_kernel import LinearFormKernel
from pygeoinf.interval.configs import (
    IntegrationConfig,
    LebesgueIntegrationConfig,
    ParallelConfig,
    LebesgueParallelConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    This class properly inherits from the pygeoinf HilbertSpace abstract base
    class, using Function objects as the Vector type. It provides:
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
        weight: Optional[Callable] = None,
        integration_config: Optional[Union[
            IntegrationConfig,
            LebesgueIntegrationConfig
        ]] = None,
        parallel_config: Optional[Union[
            ParallelConfig,
            LebesgueParallelConfig
        ]] = None,
    ):
        """
        Initialize a Lebesgue space L²([a,b]).

        Args:
            dim: Dimension of the finite-dimensional approximation space
            function_domain: IntervalDomain object specifying [a,b] and
                boundary conditions
            basis: Basis specification, can be:
                - str: 'fourier', 'hat', 'sine', etc.
                  (auto-generated but not fully implemented)
                - str: 'none'
                  (creates baseless space for temporary use)
                - list: [func1, func2, ...]
                  (custom callable functions)
                - None: 'none' basis
                  (default - create baseless space)
            weight: Optional weight function for weighted L² space
            integration_config: Hierarchical integration configuration.
                Can be IntegrationConfig (same for all subsystems) or
                LebesgueIntegrationConfig (different for inner_product,
                dual, and general operations). If None, uses old parameters.
            parallel_config: Hierarchical parallelization configuration.
                Can be ParallelConfig (same for all subsystems) or
                LebesgueParallelConfig (different for inner_product, dual,
                and general operations). If None, defaults to serial.

        Examples:
            >>> # Simple - just works with good defaults
            >>> space = Lebesgue(dim=50, function_domain=domain)
            >>> space = Lebesgue(dim=50, function_domain=domain, basis='sine')

            >>> # Old API still works (backward compatible)
            >>> space = Lebesgue(
            ...     dim=50,
            ...     function_domain=domain,
            ...     basis='sine',
            ...     integration_npoints=5000
            ... )

            >>> # Modify after construction (old API)
            >>> space.integration_npoints = 10000

            >>> # New hierarchical API - different settings per subsystem
            >>> space = Lebesgue(dim=50, function_domain=domain, basis='sine')
            >>> space.integration.inner_product.n_points = 20000  # Gram
            >>> space.parallel.dual.enabled = True  # Parallel dual ops
            >>> space.parallel.dual.n_jobs = 8

            >>> # Use preset configurations
            >>> from pygeoinf.interval.integration_config import (
            ...     LebesgueIntegrationConfig,
            ...     LebesgueParallelConfig
            ... )
            >>> int_cfg = LebesgueIntegrationConfig.high_accuracy_galerkin()
            >>> par_cfg = LebesgueParallelConfig.parallel_dual()
            >>> space = Lebesgue(
            ...     dim=100,
            ...     function_domain=domain,
            ...     basis='sine',
            ...     integration_config=int_cfg,
            ...     parallel_config=par_cfg
            ... )

            >>> # Adaptive config based on dimension
            >>> config = LebesgueIntegrationConfig.adaptive_spectral(dim=100)
            >>> space = Lebesgue(
            ...     dim=100,
            ...     function_domain=domain,
            ...     basis='sine',
            ...     integration_config=config
            ... )
        """
        self._dim = dim
        self._function_domain = function_domain
        self._weight = weight

        # Integration configuration (hierarchical)
        if integration_config is None:
            # No config provided, use defaults
            self.integration = LebesgueIntegrationConfig()
        elif isinstance(integration_config, IntegrationConfig):
            # Single config: use for all subsystems
            self.integration = LebesgueIntegrationConfig.from_single(
                integration_config
            )
        else:
            # Hierarchical config: use as-is
            self.integration = integration_config

        # Parallelization configuration
        if parallel_config is None:
            # Default: no parallelization
            self.parallel = LebesgueParallelConfig()
        else:
            # New API: use provided config
            if isinstance(parallel_config, ParallelConfig):
                # Single config: use for all subsystems
                self.parallel = LebesgueParallelConfig.from_single(
                    parallel_config
                )
            else:
                # Hierarchical config: use as-is
                self.parallel = parallel_config

        # Initialize basis (simplified single-parameter approach)
        self._initialize_basis(basis or 'none')

        # Cached computations
        self._metric = None
        self._inverse_metric_chol = None

    # ================================================================
    # Abstract methods that MUST be implemented for HilbertSpace
    # ================================================================

    @property
    def dim(self) -> int:
        """The finite dimension of the space."""
        return self._dim

    # ================================================================
    # Backward-compatible properties for old integration API
    # ================================================================

    @property
    def integration_method(self) -> str:
        """
        Get integration method (backward compatible).

        Returns the method from general config. To set different methods
        for different subsystems, use the hierarchical config:
            space.integration.inner_product.method = 'simpson'
            space.integration.dual.method = 'trapz'
        """
        return self.integration.general.method

    @integration_method.setter
    def integration_method(self, value: str):
        """
        Set integration method for all subsystems (backward compatible).

        This sets the method for inner_product, dual, and general configs.
        For fine-grained control, modify configs directly:
            space.integration.inner_product.method = 'simpson'
        """
        for cfg in [self.integration.inner_product,
                    self.integration.dual,
                    self.integration.general]:
            cfg.method = value  # type: ignore

    @property
    def integration_npoints(self) -> int:
        """Get integration points (backward compatible)."""
        return self.integration.general.n_points

    @integration_npoints.setter
    def integration_npoints(self, value: int):
        """
        Set integration points for all subsystems (backward compatible).
        """
        for cfg in [self.integration.inner_product,
                    self.integration.dual,
                    self.integration.general]:
            cfg.n_points = value

    # ================================================================
    # Core HilbertSpace abstract methods
    # ================================================================

    @property
    def metric(self) -> np.ndarray:
        self._require_basis()

        if self._metric is None:
            self._compute_metric()
            # Symmetrize hard to dampen quadrature asymmetry
            self._metric = 0.5 * (self._metric + self._metric.T)
        return self._metric

    def _compute_inverse_metric_chol(self) -> np.ndarray:
        if self._inverse_metric_chol is None:
            G = self.metric
            try:
                self._inverse_metric_chol = np.linalg.cholesky(G)
            except np.linalg.LinAlgError:
                # Minimal Tikhonov regularization if noise makes G semidefinite
                eps = 1e-12 * max(1.0, np.linalg.norm(G, ord=2))
                self._inverse_metric_chol = np.linalg.cholesky(G + eps * np.eye(G.shape[0]))
        return self._inverse_metric_chol

    @property
    def inverse_metric_chol(self) -> np.ndarray:
        if self._inverse_metric_chol is None:
            self._compute_inverse_metric_chol()
        return self._inverse_metric_chol

    def _spd_solve(self, b: np.ndarray) -> np.ndarray:
        L = self.inverse_metric_chol
        y = np.linalg.solve(L, b)
        return np.linalg.solve(L.T, y)

    def to_dual(self, x: "Function") -> Any:
        """
        Maps a function to its canonical dual vector (a linear functional).

        This implements the Riesz representation using the metric tensor:
        for u ∈ L², the dual element is computed as G * c_u where G is the
        metric (Gram matrix) and c_u are the coefficients of u.

        Uses self.integration.dual and self.parallel.dual configs for
        the LinearFormKernel, allowing different integration and
        parallelization settings for dual operations than for inner products.

        Args:
            x: A Function in this Lebesgue space

        Returns:
            LinearForm representing the dual element
        """
        # Use dual configs for LinearFormKernel
        int_cfg = self.integration.dual
        par_cfg = self.parallel.dual
        return LinearFormKernel(
            self,
            kernel=x,
            integration_config=int_cfg,
            parallel_config=par_cfg,
        )

    def from_dual(self, xp: Union[LinearFormKernel, LinearForm]) -> "Function":
        """
        Maps a dual vector back to its representative in the primal space.

        This is the inverse Riesz map: given a linear functional, find the
        unique function u such that the functional is v ↦ ⟨u,v⟩_L².

        Args:
            xp: A LinearForm from the dual space

        Returns:
            Function representing the primal element
        """
        if isinstance(xp, LinearFormKernel) and xp.kernel is not None:
            return xp.kernel
        else:
            primal_components = self._spd_solve(xp.components)
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
        self._require_basis()
        # Compute right-hand side: b[i] = ⟨x, φᵢ⟩_L²
        rhs = np.zeros(self.dim)
        for i in range(self.dim):
            basis_func = self.get_basis_function(i)
            rhs[i] = self._continuous_l2_inner_product(x, basis_func)

        # Solve G c = b for the coefficients c
        coeffs = self._spd_solve(rhs)

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

    @property
    def zero(self):
        return Function(self, evaluate_callable=lambda x: np.zeros_like(x))
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

    def axpy(self, a: float, x: "Function", y: "Function") -> "Function":
        """
        Performs operation y := y + a*x and returns the result.

        For functions with coefficients, modifies coefficients in place.
        For baseless functions, returns a new function (cannot truly be in-place).

        Note: For baseless functions, the caller must capture the return value
        since true in-place modification is not possible for immutable callables.
        """
        if (hasattr(y, 'coefficients') and y.coefficients is not None and
            hasattr(x, 'coefficients') and x.coefficients is not None):
            y.coefficients += a * x.coefficients
            return y
        else:
            # For baseless functions, return a new function
            return self.add(y, self.multiply(a, x))

    # ================================================================
    # Additional methods specific to function spaces
    # ================================================================

    @property
    def function_domain(self) -> "IntervalDomain":
        """The interval domain [a,b] for this space."""
        return self._function_domain

    def restrict_to_subinterval(
        self,
        subdomain: "IntervalDomain",
    ) -> "Lebesgue":
        """
        Create a new Lebesgue space restricted to a subinterval.

        This method creates a new Lebesgue space defined on a subinterval
        of the current function domain. The new space has the same dimension
        and inherits the integration settings, but has a smaller domain.
        This is useful for modeling discontinuities by breaking a space into
        pieces.

        Args:
            subdomain: An IntervalDomain representing the subinterval.
                Must be contained within the current function_domain.

        Returns:
            A new Lebesgue space defined on the subinterval.

        Raises:
            ValueError: If subdomain is not contained in function_domain.

        Example:
            >>> domain = IntervalDomain(0, 1)
            >>> space = Lebesgue(100, domain)
            >>> subdomain = IntervalDomain(0, 0.5)
            >>> restricted_space = space.restrict_to_subinterval(subdomain)
        """
        # Validate that subdomain is within current domain
        if not (self.function_domain.a <= subdomain.a and
                subdomain.b <= self.function_domain.b):
            raise ValueError(
                f"Subdomain {subdomain} must be contained in "
                f"function domain {self.function_domain}"
            )

        # Create new space with same dimension but restricted domain
        # Start with a baseless space
        restricted_space = Lebesgue(
            self.dim,
            subdomain,
            basis='none',
            weight=self._weight
        )

        # Copy integration and parallel settings from parent space
        # Use deepcopy to preserve hierarchical per-subsystem settings
        import copy as _copy
        if hasattr(self, 'integration'):
            restricted_space.integration = _copy.deepcopy(self.integration)
        if hasattr(self, 'parallel'):
            restricted_space.parallel = _copy.deepcopy(self.parallel)

        # If the current space has a basis, we need to handle it
        # For now, we create a baseless space - the user can set a basis
        # later if needed. A more sophisticated approach would be to
        # restrict basis functions to the subinterval, but this requires
        # more complex logic.

        return restricted_space

    @classmethod
    def with_discontinuities(
        cls,
        dim: int,
        function_domain: "IntervalDomain",
        discontinuity_points: list,
        /,
        *,
        basis: Optional[Union[str, list]] = None,
        weight: Optional[Callable] = None,
        dim_per_subspace: Optional[list] = None,
        basis_per_subspace: Optional[list] = None,
        integration_config = None,
        parallel_config = None,
    ) -> LebesgueSpaceDirectSum:
        """
        Create a LebesgueSpaceDirectSum with discontinuities.

        This factory method creates a direct sum of Lebesgue spaces, where
        each component space is defined on a subinterval separated by
        discontinuity points. This allows modeling of functions with jump
        discontinuities.

        The total dimension `dim` is distributed across the subspaces.
        By default, dimension is allocated proportionally to subinterval
        lengths, but you can specify custom dimensions.

        Args:
            dim: Total dimension across all subspaces
            function_domain: The full interval domain
            discontinuity_points: List of points where discontinuities occur
            basis: Basis type for ALL subspaces (string like 'fourier',
                'sine', etc., or 'none'). Ignored if basis_per_subspace
                is provided. If basis is a list of functions, raises an
                error (use basis_per_subspace instead).
            weight: Weight function for inner product
            dim_per_subspace: Optional list specifying dimension of each
                subspace. If None, dimensions are allocated proportionally
                to subinterval lengths. Must sum to `dim`.
            basis_per_subspace: Optional list of basis specifications, one
                for each subspace. Each element can be a string (basis type)
                or 'none'. If provided, overrides the `basis` parameter.
                Must have length equal to number of subspaces.
            integration_config: Integration configuration for all subspaces.
                If None, uses default configuration.
            parallel_config: Parallel configuration for all subspaces.
                If None, uses default configuration.

        Returns:
            LebesgueSpaceDirectSum representing the space with discontinuities

        Raises:
            ValueError: If basis is a list (not supported for discontinuous
                spaces), or if basis_per_subspace has wrong length.

        Example:
            >>> # Create space with discontinuity at x=0.5
            >>> domain = IntervalDomain(0, 1)
            >>> space = Lebesgue.with_discontinuities(
            ...     100, domain, [0.5]
            ... )
            >>> # Creates direct sum: [Lebesgue(50, [0,0.5]),
            >>> #                       Lebesgue(50, [0.5,1])]

            >>> # With custom dimensions per subspace
            >>> space = Lebesgue.with_discontinuities(
            ...     100, domain, [0.5], dim_per_subspace=[30, 70]
            ... )

            >>> # With same basis type for all subspaces
            >>> space = Lebesgue.with_discontinuities(
            ...     100, domain, [0.5], basis='fourier'
            ... )

            >>> # With different basis for each subspace
            >>> space = Lebesgue.with_discontinuities(
            ...     100, domain, [0.5],
            ...     basis_per_subspace=['fourier', 'sine']
            ... )

            >>> # With no basis initially (set manually later)
            >>> space = Lebesgue.with_discontinuities(
            ...     100, domain, [0.5], basis='none'
            ... )
            >>> # Later: set basis on individual subspaces
            >>> from pygeoinf.interval.providers import CustomBasisProvider
            >>> space.subspace(0).set_basis_provider(my_provider_0)
            >>> space.subspace(1).set_basis_provider(my_provider_1)

        Notes:
            - If basis is a list of functions, this raises an error because
              function restriction to subdomains is not implemented. Use
              basis_per_subspace to specify basis for each subdomain, or
              use 'none' and set basis providers manually afterward.
            - String-based bases ('fourier', 'sine', etc.) are automatically
              created for each subdomain with appropriate domain.
        """
        # Validate basis parameter
        if isinstance(basis, list):
            raise ValueError(
                "Providing a list of basis functions is not supported for "
                "discontinuous spaces. Use basis_per_subspace to specify "
                "basis type for each subspace, or use basis='none' and set "
                "basis providers manually after creation."
            )

        # Split domain at discontinuities
        subdomains = function_domain.split_at_discontinuities(
            discontinuity_points
        )
        n_subspaces = len(subdomains)

        # Determine dimensions for each subspace
        if dim_per_subspace is None:
            # Allocate proportionally to subdomain lengths
            lengths = [sd.length for sd in subdomains]
            total_length = sum(lengths)

            # Start with proportional allocation (floored)
            dims = [int(dim * length / total_length)
                    for length in lengths]

            # Distribute remaining dimensions
            remainder = dim - sum(dims)
            # Add to largest subdomains first
            length_indices = sorted(range(n_subspaces),
                                    key=lambda i: lengths[i],
                                    reverse=True)
            for i in range(remainder):
                dims[length_indices[i % n_subspaces]] += 1
        else:
            # Use provided dimensions
            if len(dim_per_subspace) != n_subspaces:
                raise ValueError(
                    f"dim_per_subspace must have length {n_subspaces}, "
                    f"got {len(dim_per_subspace)}"
                )
            if sum(dim_per_subspace) != dim:
                raise ValueError(
                    f"dim_per_subspace must sum to {dim}, "
                    f"got {sum(dim_per_subspace)}"
                )
            dims = list(dim_per_subspace)

        # Determine basis for each subspace
        if basis_per_subspace is not None:
            # Use provided basis for each subspace
            if len(basis_per_subspace) != n_subspaces:
                raise ValueError(
                    f"basis_per_subspace must have length {n_subspaces}, "
                    f"got {len(basis_per_subspace)}"
                )
            bases = list(basis_per_subspace)
        else:
            # Use same basis for all subspaces
            bases = [basis] * n_subspaces

        # Create subspaces
        subspaces = [
            cls(d, subdomain, basis=b, weight=weight,
                integration_config=integration_config,
                parallel_config=parallel_config)
            for d, subdomain, b in zip(dims, subdomains, bases)
        ]

        # Return direct sum
        return LebesgueSpaceDirectSum(subspaces)

    # ================================================================
    # NOTE: Legacy _integration_method/_integration_npoints properties
    # were removed to avoid duplication with the hierarchical
    # `self.integration` configuration. Use the `integration` config
    # object (and `self.integration_npoints`/`self.integration_method`
    # backward-compatible properties earlier in the class) instead.

    # ================================================================
    # Additional methods for basis function management
    # ================================================================
    def _require_basis(self):
        if self._basis_type == 'none':
            raise RuntimeError(
                "This operation requires a basis; set a BasisProvider "
                "or direct basis first."
            )

    def _initialize_basis(self, basis):
        """
        Initialize basis from simplified single parameter.

        Args:
            basis: str or list of callables
        """
        if isinstance(basis, str):
            if basis == 'none':
                # Create a baseless space - useful for temporary spaces when
                # creating BasisProviders
                self._basis_type = 'none'
                self._basis_functions = None
                self.basis_provider = None
                self._use_basis_provider = False
            else:
                # String-based standard basis types using provider factory
                try:
                    basis_provider = create_basis_provider(self, basis)
                    self._basis_type = basis
                    self._basis_functions = None
                    self.basis_provider = basis_provider
                    self._use_basis_provider = True
                except ValueError as e:
                    raise ValueError(f"Unsupported basis type '{basis}': {e}")

        elif isinstance(basis, list):
            # List of callables/functions provided directly - transform to
            # Functions
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
                "basis must be a string or list of callables, got "
                f"{type(basis)}"
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
            raise RuntimeError(
                "No basis functions available - space is baseless"
            )

        if self._use_basis_provider:
            # Use BasisProvider for lazy evaluation (set via
            # set_basis_provider)
            if self.basis_provider is None:
                raise RuntimeError("No basis provider available")
            if hasattr(self.basis_provider, 'get_basis_function'):
                return [f for i in range(self.dim)
                        if (f := self.basis_provider.get_basis_function(i)) is not None]
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

    def _continuous_l2_inner_product(
        self, u: "Function", v: "Function"
    ) -> float:
        """
        Compute the continuous L² inner product via numerical integration.

        This is different from the discrete inner product of the
        finite-dimensional
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
            n_points=self.integration_npoints,
            weight=self._weight
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

    def _clear_metric_caches(self):
        self._metric = None
        self._chol = None


def create_basis_provider(space: Lebesgue, basis_type: str) -> "BasisProvider":
    """
    Factory function to create a BasisProvider for a given Lebesgue space.

    Args:
        space: The Lebesgue space to create the basis for
        basis_type: Type of basis ('fourier', 'hat', etc.)

    Returns:
        An instance of BasisProvider for the specified basis type
    """
    from pygeoinf.interval.function_providers import (
        FourierFunctionProvider,
        HatFunctionProvider,
        SineFunctionProvider,
        CosineFunctionProvider,
        MixedDNFunctionProvider,
        MixedNDFunctionProvider
    )
    from pygeoinf.interval.providers import CustomBasisProvider

    if basis_type == 'fourier':
        return CustomBasisProvider(
            space,
            functions_provider=FourierFunctionProvider(space),
            orthonormal=True,
            basis_type='fourier'
        )
    elif basis_type == 'fourier_non_constant':
        return CustomBasisProvider(
            space,
            functions_provider=FourierFunctionProvider(
                space,
                non_constant_only=True
            ),
            orthonormal=True,
            basis_type='fourier_non_constant'
        )
    elif basis_type == 'hat':
        return CustomBasisProvider(
            space,
            functions_provider=HatFunctionProvider(space),
            orthonormal=False,
            basis_type='hat'
        )
    elif basis_type == 'sine':
        return CustomBasisProvider(
            space,
            functions_provider=SineFunctionProvider(space),
            orthonormal=True,
            basis_type='sine'
        )
    elif basis_type == 'cosine':
        return CustomBasisProvider(
            space,
            functions_provider=CosineFunctionProvider(space),
            orthonormal=True,
            basis_type='cosine'
        )
    elif basis_type == 'cosine_non_constant':
        return CustomBasisProvider(
            space,
            functions_provider=CosineFunctionProvider(
                space,
                non_constant_only=True
            ),
            orthonormal=True,
            basis_type='cosine_non_constant'
        )
    elif basis_type == 'DN':
        return CustomBasisProvider(
            space,
            functions_provider=MixedDNFunctionProvider(space),
            orthonormal=True,
            basis_type='DN'
        )
    elif basis_type == 'ND':
        return CustomBasisProvider(
            space,
            functions_provider=MixedNDFunctionProvider(space),
            orthonormal=True,
            basis_type='ND'
        )
    elif basis_type == 'radial_dirichlet':
        from pygeoinf.interval.function_providers import RadialLaplacianDirichletProvider
        return CustomBasisProvider(
            space,
            functions_provider=RadialLaplacianDirichletProvider(space),
            orthonormal=True,
            basis_type='radial_dirichlet'
        )
    elif basis_type == 'radial_neumann':
        from pygeoinf.interval.function_providers import RadialLaplacianNeumannProvider
        return CustomBasisProvider(
            space,
            functions_provider=RadialLaplacianNeumannProvider(space),
            orthonormal=True,
            basis_type='radial_neumann'
        )
    elif basis_type == 'radial_DD':
        from pygeoinf.interval.function_providers import RadialLaplacianDDProvider
        return CustomBasisProvider(
            space,
            functions_provider=RadialLaplacianDDProvider(space),
            orthonormal=True,
            basis_type='radial_DD'
        )
    elif basis_type == 'radial_DN':
        from pygeoinf.interval.function_providers import RadialLaplacianDNProvider
        return CustomBasisProvider(
            space,
            functions_provider=RadialLaplacianDNProvider(space),
            orthonormal=True,
            basis_type='radial_DN'
        )
    elif basis_type == 'radial_ND':
        from pygeoinf.interval.function_providers import RadialLaplacianNDProvider
        return CustomBasisProvider(
            space,
            functions_provider=RadialLaplacianNDProvider(space),
            orthonormal=True,
            basis_type='radial_ND'
        )
    elif basis_type == 'radial_NN':
        from pygeoinf.interval.function_providers import RadialLaplacianNNProvider
        return CustomBasisProvider(
            space,
            functions_provider=RadialLaplacianNNProvider(space),
            orthonormal=True,
            basis_type='radial_NN'
        )
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")


# =============================================================================
# Partitioned Lebesgue Space for Known Regions
# =============================================================================


class KnownRegion:
    """
    Specification of a region where the model is known (fixed).

    A known region is a sub-interval of the full domain where the model
    value is fixed and not part of the inference. For example, shear wave
    velocity (Vs) in the Earth's outer core is known to be zero.

    Attributes:
        interval: The IntervalDomain where the model is known.
        value: A Function instance representing the known value on this
            interval. The function must be defined on a Lebesgue space
            with the same function_domain as the interval.

    Example:
        >>> from pygeoinf.interval import IntervalDomain, Lebesgue, Function
        >>> # Define outer core region (Vs = 0)
        >>> outer_core_domain = IntervalDomain(1217.5, 3480.0)  # radii in km
        >>> outer_core_space = Lebesgue(10, outer_core_domain, basis='none')
        >>> zero_function = Function(
        ...     outer_core_space,
        ...     evaluate_callable=lambda r: 0.0
        ... )
        >>> outer_core = KnownRegion(outer_core_domain, zero_function)
    """

    def __init__(
        self,
        interval: "IntervalDomain",
        value: "Function",
    ):
        """
        Initialize a KnownRegion.

        Args:
            interval: The IntervalDomain where the model is known.
            value: A Function instance representing the known value.
                Must be defined on a space with the same function_domain
                as the interval.

        Raises:
            TypeError: If value is not a Function instance.
            ValueError: If the function's domain doesn't match the interval.
        """
        # Import here to avoid circular imports
        from pygeoinf.interval.functions import Function as FunctionClass

        if not isinstance(value, FunctionClass):
            raise TypeError(
                f"value must be a Function instance, "
                f"got {type(value).__name__}. "
                "Create a Function with evaluate_callable for known values."
            )

        # Check that function's domain matches the interval
        func_domain = value.space.function_domain
        if not (func_domain.a == interval.a and func_domain.b == interval.b):
            raise ValueError(
                f"Function's domain {func_domain} does not match "
                f"the known region interval {interval}"
            )

        self.interval = interval
        self.value = value

    @classmethod
    def zero(
        cls,
        interval: "IntervalDomain",
        dim: int = 10,
        **lebesgue_kwargs
    ) -> "KnownRegion":
        """
        Convenience factory for creating a known region with zero value.

        Args:
            interval: The IntervalDomain where the model is zero.
            dim: Dimension of the temporary Lebesgue space (only used
                for creating the Function; doesn't affect computations).
            **lebesgue_kwargs: Additional arguments passed to Lebesgue.

        Returns:
            A KnownRegion with value = 0 everywhere on the interval.

        Example:
            >>> outer_core_domain = IntervalDomain(1217.5, 3480.0)
            >>> outer_core = KnownRegion.zero(outer_core_domain)
        """
        from pygeoinf.interval.functions import Function as FunctionClass

        # Create a minimal Lebesgue space for the zero function
        space = Lebesgue(dim, interval, basis='none', **lebesgue_kwargs)

        def zero_callable(x):
            return np.zeros_like(np.asarray(x), dtype=float)

        zero_func = FunctionClass(
            space,
            evaluate_callable=zero_callable,
            name="zero"
        )
        return cls(interval, zero_func)

    @classmethod
    def constant(
        cls,
        interval: "IntervalDomain",
        constant_value: float,
        dim: int = 10,
        **lebesgue_kwargs
    ) -> "KnownRegion":
        """
        Convenience factory for creating a known region with constant value.

        Args:
            interval: The IntervalDomain where the model is constant.
            constant_value: The constant value of the model.
            dim: Dimension of the temporary Lebesgue space.
            **lebesgue_kwargs: Additional arguments passed to Lebesgue.

        Returns:
            A KnownRegion with a constant value on the interval.
        """
        from pygeoinf.interval.functions import Function as FunctionClass

        space = Lebesgue(dim, interval, basis='none', **lebesgue_kwargs)
        const_func = FunctionClass(
            space,
            evaluate_callable=lambda x: np.full_like(
                np.asarray(x), constant_value, dtype=float
            ),
            name=f"constant_{constant_value}"
        )
        return cls(interval, const_func)

    def __repr__(self) -> str:
        return f"KnownRegion(interval={self.interval}, value={self.value})"


class PartitionedLebesgueSpace:
    """
    A Lebesgue space partitioned into known and unknown regions.

    This class handles the common case in geophysical inversions where the
    model is known (fixed) on some sub-intervals and unknown on others.
    For example, shear wave velocity (Vs) is known to be zero in the
    Earth's outer core, so only the inner core and mantle regions need
    to be inferred.

    The class creates a direct sum of Lebesgue spaces for the unknown
    regions, which serves as the model space for inference. It also
    provides methods to:
    - Extend a model from unknown regions to the full domain
    - Restrict functions (like sensitivity kernels) to unknown regions
    - Create forward operators that correctly handle the partition

    Attributes:
        full_domain: The complete IntervalDomain.
        known_regions: List of KnownRegion specifications.
        unknown_intervals: List of IntervalDomain for unknown regions.
        unknown_spaces: List of Lebesgue spaces for unknown regions.
        model_space: HilbertSpaceDirectSum of unknown spaces (use this
            as the model space for inference).

    Example:
        >>> # Earth model: Vs is zero in outer core
        >>> full_domain = IntervalDomain(0, 6371)  # radius in km
        >>>
        >>> # Define known region (outer core, Vs = 0)
        >>> outer_core_interval = IntervalDomain(1217.5, 3480.0)
        >>> outer_core = KnownRegion.zero(outer_core_interval)
        >>>
        >>> # Create partitioned space
        >>> partitioned = PartitionedLebesgueSpace(
        ...     full_domain=full_domain,
        ...     known_regions=[outer_core],
        ...     dims=[50, 50],  # dimensions for [inner_core, mantle]
        ...     basis='cosine'
        ... )
        >>>
        >>> # Use model_space for inference
        >>> M_vs = partitioned.model_space  # DirectSum
    """

    def __init__(
        self,
        full_domain: "IntervalDomain",
        known_regions: List[KnownRegion],
        dims: List[int],
        *,
        basis: Optional[Union[str, list]] = None,
        bases: Optional[List[Optional[Union[str, list]]]] = None,
        integration_config: Optional[Union[
            IntegrationConfig,
            LebesgueIntegrationConfig
        ]] = None,
        parallel_config: Optional[Union[
            ParallelConfig,
            LebesgueParallelConfig
        ]] = None,
        weight: Optional[Callable] = None,
    ):
        """
        Initialize a PartitionedLebesgueSpace.

        Args:
            full_domain: The complete IntervalDomain.
            known_regions: List of KnownRegion specifications. Must be
                non-overlapping and contained within full_domain.
            dims: List of dimensions for each unknown region. The number
                of elements must match the number of unknown regions
                (which is len(known_regions) + 1 if known regions don't
                touch boundaries, or fewer if they do).
            basis: Basis type for ALL unknown region spaces. Use 'none',
                'fourier', 'sine', 'cosine', etc.
            bases: Optional list of basis types, one per unknown region.
                Overrides `basis` if provided.
            integration_config: Integration configuration for all spaces.
            parallel_config: Parallel configuration for all spaces.
            weight: Weight function for inner product.

        Raises:
            ValueError: If known regions overlap, are outside full_domain,
                or if dims doesn't match the number of unknown regions.
        """
        self.full_domain = full_domain
        self.known_regions = sorted(
            known_regions, key=lambda kr: kr.interval.a
        )

        # Validate known regions
        self._validate_known_regions()

        # Compute unknown intervals
        self.unknown_intervals = self._compute_unknown_intervals()

        # Validate dims
        n_unknown = len(self.unknown_intervals)
        if len(dims) != n_unknown:
            raise ValueError(
                f"dims must have {n_unknown} elements "
                f"(one per unknown region), got {len(dims)}. "
                f"Unknown regions: {self.unknown_intervals}"
            )

        # Determine basis for each unknown region
        if bases is not None:
            if len(bases) != n_unknown:
                raise ValueError(
                    f"bases must have {n_unknown} elements, got {len(bases)}"
                )
            region_bases = bases
        else:
            region_bases = [basis] * n_unknown

        # Create Lebesgue spaces for unknown regions
        self.unknown_spaces: List[Lebesgue] = []
        for i, (interval, dim, region_basis) in enumerate(
            zip(self.unknown_intervals, dims, region_bases)
        ):
            space = Lebesgue(
                dim,
                interval,
                basis=region_basis,
                weight=weight,
                integration_config=integration_config,
                parallel_config=parallel_config,
            )
            self.unknown_spaces.append(space)

        # Create the model space as direct sum of unknown spaces
        self.model_space = LebesgueSpaceDirectSum(self.unknown_spaces)

    def _validate_known_regions(self) -> None:
        """Validate known regions are non-overlapping and within domain."""
        for i, kr in enumerate(self.known_regions):
            # Check within full domain
            if kr.interval.a < self.full_domain.a:
                raise ValueError(
                    f"Known region {i} starts at {kr.interval.a}, which is "
                    f"before full_domain start {self.full_domain.a}"
                )
            if kr.interval.b > self.full_domain.b:
                raise ValueError(
                    f"Known region {i} ends at {kr.interval.b}, which is "
                    f"after full_domain end {self.full_domain.b}"
                )

            # Check non-overlapping with previous region
            if i > 0:
                prev_kr = self.known_regions[i - 1]
                if kr.interval.a < prev_kr.interval.b:
                    raise ValueError(
                        f"Known regions {i-1} and {i} overlap: "
                        f"{prev_kr.interval} and {kr.interval}"
                    )

    def _compute_unknown_intervals(self) -> List["IntervalDomain"]:
        """
        Compute the unknown intervals by subtracting known regions.

        Returns a list of IntervalDomain for each unknown region,
        ordered from low to high.
        """
        from pygeoinf.interval.interval_domain import IntervalDomain

        if not self.known_regions:
            # No known regions, entire domain is unknown
            return [self.full_domain]

        unknown = []
        current_start = self.full_domain.a

        for kr in self.known_regions:
            # Add unknown region before this known region (if any)
            if kr.interval.a > current_start:
                unknown.append(IntervalDomain(
                    current_start,
                    kr.interval.a,
                    boundary_type=self.full_domain.boundary_type
                ))
            current_start = kr.interval.b

        # Add unknown region after last known region (if any)
        if current_start < self.full_domain.b:
            unknown.append(IntervalDomain(
                current_start,
                self.full_domain.b,
                boundary_type=self.full_domain.boundary_type
            ))

        return unknown

    @property
    def n_unknown_regions(self) -> int:
        """Number of unknown regions."""
        return len(self.unknown_intervals)

    @property
    def n_known_regions(self) -> int:
        """Number of known regions."""
        return len(self.known_regions)

    def get_unknown_space(self, index: int) -> Lebesgue:
        """Get the Lebesgue space for the i-th unknown region."""
        return self.unknown_spaces[index]

    def get_known_region(self, index: int) -> KnownRegion:
        """Get the i-th known region."""
        return self.known_regions[index]

    def extend_to_full_domain(
        self,
        unknown_model: List["Function"],
        name: Optional[str] = None
    ) -> "Function":
        """
        Extend a model from unknown regions to the full domain.

        Takes a list of Functions (one per unknown region) and creates
        a single Function on the full domain that:
        - Evaluates to the given functions on unknown regions
        - Evaluates to the known values on known regions

        Args:
            unknown_model: List of Functions, one per unknown region.
                Must have length equal to n_unknown_regions.
            name: Optional name for the resulting function.

        Returns:
            A Function on the full domain.

        Raises:
            ValueError: If unknown_model has wrong length or types.
        """
        from pygeoinf.interval.functions import Function as FunctionClass

        if len(unknown_model) != self.n_unknown_regions:
            raise ValueError(
                f"unknown_model must have {self.n_unknown_regions} elements, "
                f"got {len(unknown_model)}"
            )

        # Validate types
        for i, func in enumerate(unknown_model):
            if not isinstance(func, FunctionClass):
                raise ValueError(
                    f"unknown_model[{i}] must be a Function, "
                    f"got {type(func).__name__}"
                )

        # Create a temporary space on the full domain for the extended function
        # Use min dimension across unknown spaces
        min_dim = min(space.dim for space in self.unknown_spaces)
        full_space = Lebesgue(min_dim, self.full_domain, basis='none')

        # Build the callable that evaluates correctly on each region
        def extended_callable(x):
            x_arr = np.atleast_1d(np.asarray(x))
            result = np.zeros_like(x_arr, dtype=float)

            # Evaluate on unknown regions
            for interval, func in zip(self.unknown_intervals, unknown_model):
                mask = (x_arr >= interval.a) & (x_arr <= interval.b)
                if np.any(mask):
                    result[mask] = func.evaluate(x_arr[mask], check_domain=False)

            # Evaluate on known regions
            for kr in self.known_regions:
                mask = (x_arr >= kr.interval.a) & (x_arr <= kr.interval.b)
                if np.any(mask):
                    result[mask] = kr.value.evaluate(x_arr[mask], check_domain=False)

            # Return scalar if input was scalar
            if np.ndim(x) == 0:
                return float(result[0])
            return result

        return FunctionClass(
            full_space,
            evaluate_callable=extended_callable,
            name=name or "extended_model"
        )

    def restrict_function(
        self,
        full_function: "Function",
        region_index: int
    ) -> "Function":
        """
        Restrict a function from the full domain to an unknown region.

        This is useful for restricting sensitivity kernels (defined on
        the full domain) to individual unknown regions for creating
        forward operators.

        Args:
            full_function: A Function defined on the full domain.
            region_index: Index of the unknown region (0-indexed).

        Returns:
            A Function on the specified unknown region's space.

        Raises:
            IndexError: If region_index is out of range.
            ValueError: If full_function cannot be restricted.
        """
        if region_index < 0 or region_index >= self.n_unknown_regions:
            raise IndexError(
                f"region_index {region_index} out of range "
                f"[0, {self.n_unknown_regions})"
            )

        target_space = self.unknown_spaces[region_index]
        return full_function.restrict(target_space)

    def create_restricted_kernel_provider(
        self,
        base_provider,
        region_index: int
    ):
        """
        Create a kernel provider that restricts kernels to an unknown region.

        This wraps a base kernel provider (like SensitivityKernelProvider)
        to automatically restrict each kernel to the specified unknown region.

        Args:
            base_provider: An IndexedFunctionProvider that provides kernels
                on the full domain.
            region_index: Index of the unknown region.

        Returns:
            A RestrictedKernelProvider for the specified region.
        """
        return RestrictedKernelProvider(
            base_provider,
            self.unknown_spaces[region_index]
        )

    def __repr__(self) -> str:
        return (
            f"PartitionedLebesgueSpace(\n"
            f"  full_domain={self.full_domain},\n"
            f"  n_unknown_regions={self.n_unknown_regions},\n"
            f"  unknown_intervals={self.unknown_intervals},\n"
            f"  n_known_regions={self.n_known_regions},\n"
            f"  known_intervals={[kr.interval for kr in self.known_regions]}\n"
            f")"
        )


from .function_providers.base import IndexedFunctionProvider


class RestrictedKernelProvider(IndexedFunctionProvider):
    """
    A kernel provider that restricts kernels to a sub-interval.

    This wraps a base kernel provider (like SensitivityKernelProvider)
    and restricts each kernel function to a specified Lebesgue space
    defined on a sub-interval.

    This is used when building forward operators for partitioned model
    spaces, where the forward operator for each unknown region only
    integrates the kernel over that region.

    Attributes:
        base_provider: The underlying kernel provider.
        restricted_space: The Lebesgue space to restrict to.

    Example:
        >>> # Create restricted provider for inner core
        >>> vs_kernel_full = SensitivityKernelProvider(
        ...     full_space, catalog, kernel_type='vs'
        ... )
        >>> vs_kernel_inner_core = RestrictedKernelProvider(
        ...     vs_kernel_full, inner_core_space
        ... )
        >>> # Each kernel is now restricted to inner core domain
        >>> kernel_0 = vs_kernel_inner_core.get_function_by_index(0)
    """

    def __init__(
        self,
        base_provider,
        restricted_space: Lebesgue
    ):
        """
        Initialize a RestrictedKernelProvider.

        Args:
            base_provider: An IndexedFunctionProvider that provides kernels.
                Must have get_function_by_index(i) and __len__ methods.
            restricted_space: The Lebesgue space to restrict kernels to.
                Its function_domain must be a subset of the base provider's
                space domain.
        """
        self.base_provider = base_provider
        self.restricted_space = restricted_space
        self.space = restricted_space  # For compatibility with IndexedFunctionProvider

    def get_function_by_index(self, index: int) -> "Function":
        """
        Get the restricted kernel function for the i-th index.

        Args:
            index: The index of the kernel to retrieve.

        Returns:
            A Function on the restricted_space.
        """
        # Get the full kernel
        full_kernel = self.base_provider.get_function_by_index(index)
        # Restrict to sub-interval
        return full_kernel.restrict(self.restricted_space)

    def __len__(self) -> int:
        """Return the number of available kernels."""
        return len(self.base_provider)

    def __repr__(self) -> str:
        return (
            f"RestrictedKernelProvider(\n"
            f"  base_provider={self.base_provider},\n"
            f"  restricted_domain={self.restricted_space.function_domain}\n"
            f")"
        )


class LebesgueSpaceDirectSum(HilbertSpaceDirectSum):
    def to_dual(self, xs: List[Function]) -> LinearFormKernel:
        if len(xs) != self.number_of_subspaces:
            raise ValueError("Input list has incorrect number of vectors.")
        # Use default config for direct sum
        # Get integration config from first subspace
        from .configs import IntegrationConfig, ParallelConfig
        subspace = self.subspace(0)
        if hasattr(subspace, 'integration'):
            int_cfg = subspace.integration.dual  # type: ignore
            par_cfg = subspace.parallel.dual  # type: ignore
        else:
            # Default config if subspace doesn't have config
            int_cfg = IntegrationConfig(method='trapz', n_points=1000)
            par_cfg = ParallelConfig(enabled=False, n_jobs=-1)
        return LinearFormKernel(
            self,
            kernel=xs,
            integration_config=int_cfg,
            parallel_config=par_cfg,
        )

    def from_dual(self, xp) -> List[Function]:
        # Handle both LinearFormKernel (specific) and generic LinearForm
        if isinstance(xp, LinearFormKernel):
            return xp.kernel
        else:
            # Delegate to base class for generic LinearForm objects
            return super().from_dual(xp)
