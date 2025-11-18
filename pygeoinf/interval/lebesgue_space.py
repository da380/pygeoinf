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

        # Integration configuration (hierarchical with backward compatibility)
        if integration_config is None:
            # No config provided, use old parameters for backward compatibility
            self.integration = LebesgueIntegrationConfig()
            # Set all subsystems from old params
            for cfg in [self.integration.inner_product,
                        self.integration.dual,
                        self.integration.general]:
                cfg.method = integration_method  # type: ignore
                cfg.n_points = integration_npoints
        else:
            # New API: use provided config
            if isinstance(integration_config, IntegrationConfig):
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
            cls(d, subdomain, basis=b, weight=weight)
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
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")


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
