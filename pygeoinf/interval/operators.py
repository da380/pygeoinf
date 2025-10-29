"""
Differential operators on interval domains.

This module provides differential operators for interval domains with various
solution methods:

- LaplacianOperator: -Δ with spectral/finite difference methods
- LaplacianInverseOperator: (-Δ)^(-1) with FEM solvers
- GradientOperator: ∇ with finite difference and automatic differentiation

The operators implement proper function space mappings and support
boundary conditions through the underlying function spaces.
"""

import logging
from abc import ABC, abstractmethod
from typing import (
    Optional,
    Union,
    Literal,
    List,
    Callable,
    Any,
    TYPE_CHECKING,
)

import numpy as np

# Project imports
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator

# Local/relative imports
from .lebesgue_space import Lebesgue
from .sobolev_space import Sobolev
from .boundary_conditions import BoundaryConditions
from .functions import Function

# FEM import only needed for LaplacianInverseOperator
from pygeoinf.interval.fem_solvers import GeneralFEMSolver
from pygeoinf.interval.function_providers import IndexedFunctionProvider
from pygeoinf.interval.linear_form_lebesgue import LinearFormKernel
from pygeoinf.interval.providers import LaplacianSpectrumProvider
from .fast_spectral_integration import (
    fast_spectral_coefficients,
    create_uniform_samples,
)

if TYPE_CHECKING:
    from pygeoinf import LinearForm
    from pygeoinf.hilbert_space import Vector

logger = logging.getLogger(__name__)


class SpectralOperator(LinearOperator, ABC):
    """
    Abstract base class for spectral operators on interval domains.

    Provides common functionality for eigenvalue/eigenfunction access
    and error handling.
    """

    def __init__(
        self,
        domain,
        codomain,
        mapping
    ):
        super().__init__(domain, codomain, mapping)

    @abstractmethod
    def get_eigenvalue(self, index: int) -> float:
        """Get the eigenvalue at a specific index."""
        pass

    @abstractmethod
    def get_eigenfunction(self, index: int) -> Function:
        """Get the eigenfunction at a specific index."""
        pass

    @abstractmethod
    def _apply(self, f: Function) -> Function:
        """Apply the operator to a function."""
        pass


class Gradient(LinearOperator):
    """
    The gradient operator (d/dx) on interval domains.

    In 1D, the gradient is simply the first derivative.

    - 'finite_difference': Numerical differentiation using finite differences
    """

    def __init__(
        self,
        domain: "Sobolev",
        /,
        *,
        fd_order: int = 2,
        fd_step: Optional[float] = None,
        boundary_treatment: str = 'one_sided'
    ):
        """
        Initialize the gradient operator.

        Args:
            domain: Function space (Lebesgue or SobolevSpace)
            fd_order: Order of finite difference stencil (2, 4, 6)
                for FD method
            fd_step: Step size for finite differences (auto-computed if None)
            boundary_treatment: How to handle boundaries for FD
                ('one_sided', 'extrapolate')
        """
        self._domain = domain

        # Create codomain with s-1 regularity
        self._codomain = domain
        self._fd_order = fd_order
        self._fd_step = fd_step
        self._boundary_treatment = boundary_treatment
        # logger
        self._log = logging.getLogger(__name__)

        # No boundary conditions needed for gradient (unlike Laplacian)
        super().__init__(domain, domain, self._apply)

        # Initialize method-specific components
        self._setup_finite_difference()

    def _setup_finite_difference(self):
        """Setup finite difference method."""
        # Determine step size if not provided
        if self._fd_step is None:
            a, b = self._domain.function_domain.a, \
                   self._domain.function_domain.b
            # Use a fraction of the domain size
            self._fd_step = (b - a) / 1000

        self._log.debug(
            "GradientOperator (finite difference, order %s) initialized with "
            "step size %.2e",
            self._fd_order,
            self._fd_step,
        )

    # ------------------------------------------------------------------
    # Small helpers for finite-difference evaluation
    # ------------------------------------------------------------------
    def _fd_stencil(self, order: int, location: str = "center"):
        """Return (offsets, coeffs) for first-derivative finite-difference.

        offsets are integer multiples of h; coeffs are the weights to be
        applied and divided by h when computing df/dx.
        """
        if order == 2:
            if location == "center":
                return np.array([-1, 1]), np.array([-0.5, 0.5])
            if location == "forward":
                # forward 2nd-order: (-3/2, 2, -1/2)
                return np.array([0, 1, 2]), np.array([-1.5, 2.0, -0.5])
            if location == "backward":
                return np.array([-2, -1, 0]), np.array([0.5, -2.0, 1.5])
        if order == 4:
            if location == "center":
                # coefficients for central 4th-order: 1/12, -2/3, 2/3,
                # -1/12
                offsets = np.array([-2, -1, 1, 2])
                coeffs = np.array([1/12, -2/3, 2/3, -1/12])
                return offsets, coeffs
            # For one-sided 4th-order coefficients, fall back to 2nd-order
            if location in ("forward", "backward"):
                return self._fd_stencil(2, location)
        raise ValueError(
            f"Unsupported fd_order={order} or location={location}"
        )

    def _safe_eval(self, func: Function, x: np.ndarray) -> np.ndarray:
        """Evaluate Function `func` on array x safely.

        Tries to call func(x) directly; if that fails (user's Function only
        accepts scalars), falls back to list comprehension.
        """
        try:
            return np.asarray(func(x))
        except Exception:
            # Fall back to Python loop
            return np.asarray([func(float(xi)) for xi in x])

    def _apply(self, f: Function) -> Function:
        """Apply gradient using finite difference method (vectorized).

        This implementation evaluates the user's function on shifted arrays
        (when possible) which greatly reduces Python-level loops and is much
        faster for array inputs. It falls back to scalar evaluation if
        necessary.
        """

        def gradient_func(x):
            scalar_input = np.isscalar(x)
            x_arr = np.asarray([x]) if scalar_input else np.asarray(x)

            h = self._fd_step
            a = self._domain.function_domain.a
            b = self._domain.function_domain.b

            # Prepare output container
            y = np.empty_like(x_arr, dtype=float)

            # Masks for boundary/interior
            left_mask = x_arr <= a + h
            right_mask = x_arr >= b - h
            interior_mask = ~(left_mask | right_mask)

            # Interior points: central stencil
            if np.any(interior_mask):
                xi = x_arr[interior_mask]
                offs, coeffs = self._fd_stencil(self._fd_order, "center")
                shifts = (xi[None, :] + offs[:, None] * h)
                vals = self._safe_eval(f, shifts)
                # vals.shape == (len(offs), n_points)
                y[interior_mask] = (coeffs[:, None] * vals).sum(axis=0) / h

            # Left boundary: forward one-sided
            if np.any(left_mask):
                xi = x_arr[left_mask]
                offs, coeffs = self._fd_stencil(self._fd_order, "forward")
                shifts = (xi[None, :] + offs[:, None] * h)
                vals = self._safe_eval(f, shifts)
                y[left_mask] = (coeffs[:, None] * vals).sum(axis=0) / h

            # Right boundary: backward one-sided
            if np.any(right_mask):
                xi = x_arr[right_mask]
                offs, coeffs = self._fd_stencil(self._fd_order, "backward")
                shifts = (xi[None, :] + offs[:, None] * h)
                vals = self._safe_eval(f, shifts)
                y[right_mask] = (coeffs[:, None] * vals).sum(axis=0) / h

            return float(y[0]) if scalar_input else y

        return Function(
            self.codomain,
            evaluate_callable=gradient_func,
            name=f"∇({getattr(f, 'name', 'f')})",
        )


class Laplacian(SpectralOperator):
    """
    The Laplacian operator (-Δ) on interval domains.

    This operator computes -d²u/dx² with specified boundary conditions.
    Multiple discretization methods are available:

    - 'spectral': Uses analytical eigendecomposition with
                  Fourier/sine/cosine basis
    - 'finite_difference': Centered finite differences

    The operator maps between function spaces based on the method:
    - Spectral: L² → L² (preserves regularity via eigenfunction expansion)
    - FD: H^s → H^(s-2) (requires domain with sufficient regularity)
    """

    def __init__(
        self,
        domain: Union[Lebesgue, Sobolev],
        boundary_conditions: BoundaryConditions,
        alpha: float = 1.0,
        /,
        *,
        method: Literal['spectral', 'fd'] = 'spectral',
        dofs: Optional[int] = None,
        fd_order: int = 2,
        n_samples: int = 512,
        integration_method: Literal['trapz', 'simpson'] = 'simpson',
        npoints: int = 1000
    ):
        """
        Initialize the negative Laplacian operator.

        Args:
            domain: Function space (Lebesgue)
            boundary_conditions: Boundary conditions for the operator
            method: Discretization method ('spectral', 'finite_difference')
            dofs: Number of degrees of freedom for FD or spectral methods
                  (default: domain.dim)
            fd_order: Order of finite difference stencil (2, 4, 6)
            n_samples: Number of samples for fast spectral transforms
            integration_method: Method for numerical integration fallback
                ('trapz', 'simpson')
            npoints: Number of points for numerical integration fallback
        """
        self._domain = domain
        self._boundary_conditions = boundary_conditions
        self._alpha = alpha
        self._dofs = dofs if dofs is not None else domain.dim
        self._fd_order = fd_order
        self._method = method
        self._n_samples = max(n_samples, self._dofs)  # Ensure enough samples
        self._integration_method = integration_method
        self._npoints = npoints

        super().__init__(domain, domain, self._apply)

        # Initialize method-specific components
        self._spectrum_provider = LaplacianSpectrumProvider(
            domain,
            boundary_conditions,
            alpha
        )

        # Check if fast transforms are available for this BC type
        self._can_use_fast_transforms = boundary_conditions.type in [
            'dirichlet', 'neumann', 'periodic',
            'mixed_dirichlet_neumann', 'mixed_neumann_dirichlet'
        ]

        if self._method == 'fd':
            self._setup_finite_difference()

    def get_eigenvalue(self, index: int) -> float:
        """Get the eigenvalue at a specific index."""
        return self._alpha * self._spectrum_provider.get_eigenvalue(index)

    def get_eigenfunction(self, index: int) -> Function:
        """Get the eigenfunction at a specific index."""
        return self._spectrum_provider.get_eigenfunction(index)

    def _setup_finite_difference(self):
        """Setup finite difference discretization."""
        # Create finite difference grid
        a, b = self._domain.function_domain.a, self._domain.function_domain.b
        self._x_grid = np.linspace(a, b, self._dofs)
        self._dx = (b - a) / (self._dofs - 1)

        # Create finite difference matrix
        self._fd_matrix = self._create_fd_matrix()

        # use module logger if available
        logger = logging.getLogger(__name__)
        logger.info(
            "LaplacianOperator (finite difference, order %s) "
            "initialized with %s grid points",
            self._fd_order,
            self._dofs,
        )

    def _create_fd_matrix(self):
        """Create finite difference matrix for the negative Laplacian."""
        n = self._dofs
        dx2 = self._dx**2

        def _second_derivative_stencil(order: int, location: str = "center"):
            """Return offsets and coefficients for second-derivative FD.

            Offsets are integer multiples of h; coeffs should be applied and
            divided by h**2 when computing d2/dx2.
            """
            if order == 2:
                if location == "center":
                    return np.array([-1, 0, 1]), np.array([1.0, -2.0, 1.0])
                if location == "near_left":
                    # one-sided second derivative (3-point) at left boundary
                    return np.array([0, 1, 2]), np.array([1.0, -2.0, 1.0])
                if location == "near_right":
                    return np.array([-2, -1, 0]), np.array([1.0, -2.0, 1.0])
            if order == 4:
                if location == "center":
                    # 5-point fourth-order accurate second derivative
                    offsets = np.array([-2, -1, 0, 1, 2])
                    coeffs = np.array([1.0, -16.0, 30.0, -16.0, 1.0]) / 12.0
                    return offsets, coeffs
                # near boundaries, fall back to second-order
                if location in ("near_left", "near_right"):
                    return _second_derivative_stencil(2, location)
            raise ValueError(
                "Unsupported fd_order or location for second derivative"
            )

        if self._fd_order == 2:
            matrix = np.zeros((n, n))

            # Fill interior with central stencil
            offsets, coeffs = _second_derivative_stencil(2, "center")
            for i in range(1, n-1):
                for off, c in zip(offsets, coeffs):
                    # Negative Laplacian: -d2/dx2
                    matrix[i, i + off] = -c / dx2

            # Fill near-boundary rows using one-sided stencils
            # left boundary (i=0)
            offsets, coeffs = _second_derivative_stencil(2, "near_left")
            for off, c in zip(offsets, coeffs):
                matrix[0, 0 + off] = -c / dx2

            # right boundary (i=n-1)
            offsets, coeffs = _second_derivative_stencil(2, "near_right")
            for off, c in zip(offsets, coeffs):
                matrix[n-1, n-1 + off] = -c / dx2

        elif self._fd_order == 4:
            matrix = np.zeros((n, n))

            # Interior with 4th-order central stencil
            offsets, coeffs = _second_derivative_stencil(4, "center")
            for i in range(2, n-2):
                for off, c in zip(offsets, coeffs):
                    matrix[i, i + off] = -c / dx2

            # Near-boundary points use lower-order stencils. Use the
            # appropriate left- and right-sided stencils so column indices
            # stay in bounds (avoid writing to column index == n).
            offsets_left, coeffs_left = _second_derivative_stencil(2, "near_left")
            offsets_right, coeffs_right = _second_derivative_stencil(2, "near_right")

            for off, c in zip(offsets_left, coeffs_left):
                matrix[1, 1 + off] = -c / dx2

            for off, c in zip(offsets_right, coeffs_right):
                matrix[n-2, n-2 + off] = -c / dx2

        else:
            raise ValueError(
                f"Finite difference order {self._fd_order} not implemented"
            )

        return matrix

    def _apply(self, f: Function) -> Function:
        """Apply the negative Laplacian operator to function f."""
        if self._method == 'spectral':
            return self._apply_spectral(f)
        elif self._method == 'fd':
            return self._apply_finite_difference(f)
        else:
            raise ValueError(f"Unknown method '{self._method}'")

    def _apply_spectral(self, f: Function) -> Function:
        """Apply Laplacian using spectral method (eigenfunction expansion)."""
        if self._can_use_fast_transforms:
            return self._apply_spectral_fast(f)
        else:
            return self._apply_spectral_slow(f)

    def _apply_spectral_fast(self, f: Function) -> Function:
        """Apply Laplacian using fast transforms (Dirichlet/Neumann/Periodic/Mixed)."""
        # Project f onto eigenfunctions and apply eigenvalues
        # Get domain information
        domain_interval = self._domain.function_domain
        domain_tuple = (domain_interval.a, domain_interval.b)
        domain_length = domain_interval.b - domain_interval.a

        # Create uniform samples of the input function
        f_samples = create_uniform_samples(
            f, domain_tuple, self._n_samples,
            self._boundary_conditions.type
        )

        # Compute all spectral coefficients at once using fast transforms
        coefficients = fast_spectral_coefficients(
            f_samples, self._boundary_conditions.type, domain_length,
            self._dofs
        )

        # Collect terms to avoid deep recursion from repeated +=
        terms = []
        for i in range(self._dofs):
            # Get the basis functions
            eigval = self.get_eigenvalue(i)

            # Compute coefficient via inner product
            coeff = coefficients[i] * (eigval)

            if abs(coeff) > 1e-14:
                eigenfunc = self.get_eigenfunction(i)
                terms.append((coeff, eigenfunc))

        # Create single callable that evaluates all terms
        if not terms:
            return self._domain.zero

        def f_new_eval(x):
            result = (np.zeros_like(x, dtype=float)
                      if isinstance(x, np.ndarray) else 0.0)
            for coeff, eigfunc in terms:
                result += coeff * eigfunc(x)
            return result

        return Function(self._codomain, evaluate_callable=f_new_eval)

    def _apply_spectral_slow(self, f: Function) -> Function:
        """Apply Laplacian using numerical integration (Robin/Mixed BCs)."""
        # Collect terms to avoid deep recursion from repeated +=
        terms = []
        for i in range(self._dofs):
            eigval = self.get_eigenvalue(i)
            eigfunc = self.get_eigenfunction(i)

            # Compute coefficient via numerical integration
            coeff = (f * eigfunc).integrate(
                method=self._integration_method,
                n_points=self._npoints
            )
            scaled_coeff = coeff * eigval

            if abs(scaled_coeff) > 1e-14:
                terms.append((scaled_coeff, eigfunc))

        # Create single callable that evaluates all terms
        if not terms:
            return self._domain.zero

        def f_new_eval(x):
            result = (np.zeros_like(x, dtype=float)
                      if isinstance(x, np.ndarray) else 0.0)
            for coeff, eigfunc in terms:
                result += coeff * eigfunc(x)
            return result

        return Function(self._codomain, evaluate_callable=f_new_eval)

    def _apply_finite_difference(self, f: Function) -> Function:
        """Apply Laplacian using finite difference method."""
        # Evaluate function on grid
        f_values = f(self._x_grid)

        # Apply finite difference matrix (this is where we apply the negative 1!!)
        laplacian_values = self._fd_matrix @ f_values

        # Create result function by interpolation
        def laplacian_func(x):
            return np.interp(x, self._x_grid, laplacian_values)

        return Function(self.codomain, evaluate_callable=laplacian_func)


class InverseLaplacian(SpectralOperator):
    """
    Inverse Laplacian operator that acts as a covariance operator.

    This operator solves -Δu = f with homogeneous boundary conditions,
    providing a self-adjoint operator suitable for Gaussian measures.

    Uses native Python FEM implementation (no external dependencies).
    """

    def __init__(
        self,
        domain: Union[Lebesgue, Sobolev],
        boundary_conditions: BoundaryConditions,
        alpha: float = 1.0,
        /,
        *,
        method: Literal['fem', 'spectral'],
        dofs: int = 100,
        fem_type: str = "hat",
        n_samples: int = 512,
        integration_method: Literal['trapz', 'simpson'] = 'simpson',
        npoints: int = 1000
    ):
        """
        Initialize the Laplacian inverse operator.

        The operator maps within the function space:
        (-Δ)⁻¹: L² → L² or H^s → H^s with specified boundary conditions

        Args:
            domain: The function space (Lebesgue or Sobolev)
            boundary_conditions: Boundary conditions for the operator
            alpha: Scaling factor for the operator (default: 1.0)
            dofs: Number of degrees of freedom (default: 100)
            fem_type: FEM implementation to use (deprecated):
                - "hat": Uses hat function basis (default)
                - "general": Uses domain's basis functions
                Note: Both options now use GeneralFEMSolver internally.
            n_samples: Number of samples for fast spectral transforms
            integration_method: Method for numerical integration fallback
                ('trapz', 'simpson')
            npoints: Number of points for numerical integration fallback
        """
        # Check that domain is a Lebesgue or Sobolev space
        if not isinstance(domain, (Lebesgue, Sobolev)):
            raise TypeError(
                f"domain must be a Lebesgue or Sobolev space, got {type(domain)}"
            )

        self._domain = domain
        self._boundary_conditions = boundary_conditions
        self._alpha = alpha
        self._method = method
        self._dofs = dofs if dofs is not None else domain.dim
        self._fem_type = fem_type
        self._n_samples = max(n_samples, self._dofs)
        self._integration_method = integration_method
        self._npoints = npoints

        # Validate fem_type
        if fem_type not in ["hat", "general"]:
            raise ValueError(
                f"fem_type must be 'hat' or 'general', got '{fem_type}'"
            )

        self._codomain = domain  # FOR TESTING ONLY

        super().__init__(domain, domain, self._apply)

        # Initialize spectrum provider for lazy eigenvalue computation
        self._spectrum_provider = LaplacianSpectrumProvider(
            domain,
            boundary_conditions,
            alpha,
            inverse=True
        )

        # Check if fast transforms are available for this BC type
        self._can_use_fast_transforms = boundary_conditions.type in [
            'dirichlet',
            'neumann',
            'periodic',
            'mixed_dirichlet_neumann', 'mixed_neumann_dirichlet'
        ]

        if method == 'fem':
            self._initialize_fem_solver()

        # logger
        self._log = logging.getLogger(__name__)
        self._log.info(
            "InverseLaplacian initialized: dofs=%s, fem_type=%s, alpha=%s",
            self._dofs,
            self._fem_type,
            self._alpha,
        )

    def _initialize_fem_solver(self):
        # Create and setup FEM solver
        self._fem_solver = GeneralFEMSolver(
            function_domain=self._domain._function_domain,
            dofs=self._dofs,
            operator_domain=self._domain,
            boundary_conditions=self._boundary_conditions
        )

    def _apply(self, f: Function) -> Function:
        """Apply the inverse Laplacian operator to function f."""
        if self._method == 'fem':
            return self._apply_fem(f)
        elif self._method == 'spectral':
            return self._apply_spectral(f)

    def _apply_spectral(self, f: Function) -> Function:
        """Apply inverse Laplacian using spectral method."""
        if self._can_use_fast_transforms:
            return self._apply_spectral_fast(f)
        else:
            return self._apply_spectral_slow(f)

    def _apply_spectral_fast(self, f: Function) -> Function:
        """Apply inverse Laplacian using fast transforms."""
        # Project f onto eigenfunctions and apply inverse eigenvalues
        # Get domain information
        domain_interval = self._domain.function_domain
        domain_tuple = (domain_interval.a, domain_interval.b)
        domain_length = domain_interval.b - domain_interval.a

        # Create uniform samples of the input function
        # Use self._n_samples (not self._dofs) to get accurate integration
        if self._boundary_conditions.type == 'neumann' or \
           self._boundary_conditions.type == 'periodic':
            f_samples = create_uniform_samples(
                f, domain_tuple, self._n_samples + 1,
                self._boundary_conditions.type
            )
            # Compute all spectral coefficients at once
            coefficients = fast_spectral_coefficients(
                f_samples, self._boundary_conditions.type,
                domain_length, self._dofs + 1  # Only need self._dofs + 1 coefficients
            )
        else:
            f_samples = create_uniform_samples(
                f, domain_tuple, self._n_samples,
                self._boundary_conditions.type
            )
            # Compute all spectral coefficients at once
            coefficients = fast_spectral_coefficients(
                f_samples, self._boundary_conditions.type,
                domain_length, self._dofs  # Only need self._dofs coefficients
            )
        if self._boundary_conditions.type == 'neumann' or \
           self._boundary_conditions.type == 'periodic':
            # For Neumann and periodic, skip zero eigenvalue
            coefficients = coefficients[1:]

        # Collect terms to avoid deep recursion from repeated +=
        terms = []
        for i in range(self._dofs):
            eigval = self.get_eigenvalue(i)

            if abs(eigval) < 1e-14:
                continue  # Skip zero eigenvalue

            # Compute coefficient via inner product
            coeff = coefficients[i] * eigval

            if abs(coeff) > 1e-14:
                eigenfunc = self.get_eigenfunction(i)
                terms.append((coeff, eigenfunc))

        # Create single callable that evaluates all terms
        if not terms:
            return self._domain.zero

        def f_new_eval(x):
            result = (np.zeros_like(x, dtype=float)
                      if isinstance(x, np.ndarray) else 0.0)
            for coeff, eigfunc in terms:
                result += coeff * eigfunc(x)
            return result

        return Function(self._codomain, evaluate_callable=f_new_eval)

    def _apply_spectral_slow(self, f: Function) -> Function:
        """Apply inverse Laplacian using numerical integration."""
        # Collect terms to avoid deep recursion from repeated +=
        terms = []
        for i in range(self._dofs):
            eigval = self.get_eigenvalue(i)

            if abs(eigval) < 1e-14:
                continue  # Skip zero eigenvalue

            eigenfunc = self.get_eigenfunction(i)

            # Compute coefficient via numerical integration
            coeff = (f * eigenfunc).integrate(
                method=self._integration_method,
                n_points=self._npoints
            )
            scaled_coeff = coeff * eigval

            if abs(scaled_coeff) > 1e-14:
                terms.append((scaled_coeff, eigenfunc))

        # Create single callable that evaluates all terms
        if not terms:
            return self._domain.zero

        def f_new_eval(x):
            result = (np.zeros_like(x, dtype=float)
                      if isinstance(x, np.ndarray) else 0.0)
            for coeff, eigfunc in terms:
                result += coeff * eigfunc(x)
            return result

        return Function(self._codomain, evaluate_callable=f_new_eval)

    def _apply_fem(self, f: Function) -> Function:
        f = (1/self._alpha) * f  # Scale input by alpha

        # Solve PDE using GeneralFEMSolver
        # Both hat and general types now use GeneralFEMSolver
        solution_coeffs = self._fem_solver.solve_poisson(f)
        return self._fem_solver.solution_to_function(solution_coeffs)

    @property
    def boundary_conditions(self) -> BoundaryConditions:
        """Get the boundary conditions object."""
        return self._boundary_conditions

    @property
    def fem_type(self) -> str:
        """Get the FEM implementation type."""
        return self._fem_type

    @property
    def dofs(self) -> int:
        """Get the mesh resolution."""
        return self._dofs

    @property
    def spectrum_provider(self):
        """Get the spectrum provider for eigenvalue computations."""
        return self._spectrum_provider

    def get_eigenvalue(self, index: int) -> float:
        return self._spectrum_provider.get_eigenvalue(index)

    def get_eigenvalues(self, indices: List[int]) -> List[float]:
        return [
            self._spectrum_provider.get_eigenvalue(i)
            for i in indices
        ]

    def get_eigenfunction(self, index: int) -> Function:
        return self._spectrum_provider.get_eigenfunction(index)



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
                 integration_method: Literal['trapz', 'simpson'] = 'simpson',
                 npoints: int = 100):
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
        """
        self._domain = domain
        self._codomain = codomain
        self._L = L
        self._k = k
        self._s = s
        self._dofs = dofs if dofs is not None else domain.dim
        self._n_samples = max(n_samples, self._dofs)  # Ensure enough samples
        self._use_fast_transforms = use_fast_transforms
        self._integration_method = integration_method
        self._npoints = npoints

        # Detect if we can use fast transforms
        self._boundary_condition = self._detect_boundary_condition()
        self._can_use_fast_transforms = (
            self._use_fast_transforms and
            self._boundary_condition is not None
        )

        if self._can_use_fast_transforms:
            logger.info(f"FastBesselSobolev: Using fast {self._boundary_condition} transforms")
        else:
            logger.info("FastBesselSobolev: Using slow numerical integration (fallback)")

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

        # Apply Bessel scaling to coefficients
        # Collect terms to avoid deep recursion from repeated +=
        terms = []
        for i in range(self._dofs):
            eigval = self._L.get_eigenvalue(i)
            if eigval is None:
                raise ValueError(f"Eigenvalue not available for index {i}")
            if eigval < 0:
                raise ValueError(f"Negative eigenvalue {eigval} at index {i}")

            # Bessel scaling: (k² + λᵢ)^s
            scale = (self._k**2 + eigval)**(self._s / 2)
            coeff = coefficients[i] * scale

            if abs(coeff) > 1e-14:  # Skip negligible coefficients
                eigfunc = self._L.get_eigenfunction(i)
                terms.append((coeff, eigfunc))

        # Create single callable that evaluates all terms
        if not terms:
            return self._domain.zero

        def f_new(x):
            result = np.zeros_like(x, dtype=float) if isinstance(x, np.ndarray) else 0.0
            for coeff, eigfunc in terms:
                result += coeff * eigfunc(x)
            return result

        return Function(self._codomain, evaluate_callable=f_new)

    def _apply_slow(self, f: Function) -> Function:
        """Apply using slow numerical integration (fallback)."""
        # Collect terms to avoid deep recursion from repeated +=
        terms = []
        for i in range(self._dofs):
            eigval = self._L.get_eigenvalue(i)
            if eigval is None:
                raise ValueError(f"Eigenvalue not available for index {i}")
            if eigval < 0:
                raise ValueError(f"Negative eigenvalue {eigval} at index {i}")

            eigfunc = self._L.get_eigenfunction(i)
            if eigfunc is None:
                raise ValueError(f"Eigenfunction not available for index {i}")

            # Slow numerical integration
            coeff = (f * eigfunc).integrate(method=self._integration_method, n_points=self._npoints)
            scale = (self._k**2 + eigval)**(self._s / 2)

            scaled_coeff = scale * coeff
            if abs(scaled_coeff) > 1e-14:
                terms.append((scaled_coeff, eigfunc))

        # Create single callable that evaluates all terms
        if not terms:
            return self._domain.zero

        def f_new(x):
            result = np.zeros_like(x, dtype=float) if isinstance(x, np.ndarray) else 0.0
            for coeff, eigfunc in terms:
                result += coeff * eigfunc(x)
            return result

        return Function(self._codomain, evaluate_callable=f_new)

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

        if self._can_use_fast_transforms:
            logger.info(f"FastBesselSobolevInverse: Using fast {self._boundary_condition} transforms")
        else:
            logger.info("FastBesselSobolevInverse: Using slow numerical integration (fallback)")

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

        # Apply inverse Bessel scaling to coefficients
        terms = []
        for i in range(self._dofs):
            eigval = self._L.get_eigenvalue(i)
            if eigval is None:
                raise ValueError(f"Eigenvalue not available for index {i}")
            if eigval < 0:
                raise ValueError(f"Negative eigenvalue {eigval} at index {i}")

            # Inverse Bessel scaling: (k² + λᵢ)^(-s)
            scale = (self._k**2 + eigval)**(-self._s / 2)
            coeff = coefficients[i] * scale

            if abs(coeff) > 1e-14:  # Skip negligible coefficients
                eigfunc = self._L.get_eigenfunction(i)

                terms.append((coeff, eigfunc))

        def f_new(x):
            result = np.zeros_like(x, dtype=float) if isinstance(x, np.ndarray) else 0.0
            for coeff, eigfunc in terms:
                result += coeff * eigfunc(x)
            return result

        return Function(self._codomain, evaluate_callable=f_new)

    def _apply_slow(self, f: Function) -> Function:
        """Apply using slow numerical integration (fallback)."""
        # Collect terms to avoid deep recursion from repeated +=
        terms = []
        for i in range(self._dofs):
            eigval = self._L.get_eigenvalue(i)
            if eigval is None:
                raise ValueError(f"Eigenvalue not available for index {i}")
            if eigval < 0:
                raise ValueError(f"Negative eigenvalue {eigval} at index {i}")

            eigfunc = self._L.get_eigenfunction(i)
            if eigfunc is None:
                raise ValueError(f"Eigenfunction not available for index {i}")

            # Slow numerical integration
            coeff = (f * eigfunc).integrate(method='simpson', n_points=10000)
            scale = (self._k**2 + eigval)**(-self._s / 2)

            scaled_coeff = scale * coeff
            if abs(scaled_coeff) > 1e-14:
                terms.append((scaled_coeff, eigfunc))

        # Create single callable that evaluates all terms
        if not terms:
            return self._domain.zero

        def f_new(x):
            result = np.zeros_like(x, dtype=float) if isinstance(x, np.ndarray) else 0.0
            for coeff, eigfunc in terms:
                result += coeff * eigfunc(x)
            return result

        return Function(self._codomain, evaluate_callable=f_new)

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


class SOLAOperator(LinearOperator):
    """
    SOLA operator that applies kernel functions to input functions via
    integration.

    This operator takes a function from a Lebesgue space and computes integrals
    against a set of kernel functions, resulting in a vector in the specified
    Euclidean space.

    The operator maps: Lebesgue -> EuclideanSpace

    For each kernel function k_i, it computes: ∫ f(x) * k_i(x) dx

    The kernel functions can be provided in three ways:
    1. Via a FunctionProvider (original functionality)
    2. Via a list of Function objects
    3. Via a list of callables (automatically converted to Function objects)

    Examples:
        # Using a function provider
        >>> provider = NormalModesProvider(lebesgue_space)
        >>> sola_op = SOLAOperator(lebesgue_space, euclidean_space,
        ...                        function_provider=provider)

        # Using direct callables
        >>> kernels = [lambda x: np.sin(x), lambda x: np.cos(x)]
        >>> sola_op = SOLAOperator(lebesgue_space, euclidean_space,
        ...                        functions=kernels)

        # Using Function objects
        >>> func1 = Function(lebesgue_space, evaluate_callable=lambda x: x**2)
        >>> func2 = Function(lebesgue_space, evaluate_callable=lambda x: x**3)
        >>> sola_op = SOLAOperator(lebesgue_space, euclidean_space,
        ...                        functions=[func1, func2])
    """

    def __init__(
        self,
        domain: Sobolev,
        codomain: EuclideanSpace,
        kernels: Optional[
            Union[
                IndexedFunctionProvider,
                List[Union[Function, Callable]]
            ]
        ] = None,
        cache_kernels: bool = False,
        integration_method: Optional[
            Literal['simpson', 'trapezoid']
        ] = 'simpson',
        n_points: Optional[int] = 1000
    ):
        """
        Initialize the SOLA operator.

        Args:
            domain: Lebesgue instance (the function space)
            codomain: EuclideanSpace instance that defines the output dimension
            function_provider: Provider for generating kernels.
                              If None and functions is None, creates a default
                              NormalModesProvider.
            functions: List of Function instances or callables to use as
                      kernels. If provided, takes precedence over
                      function_provider. Callables will be converted to
                      Function instances.
            random_state: Random seed for reproducible function generation
            cache_functions: If True, cache kernels after first
                           access for faster repeated operations
            integration_method: Method for numerical integration
                ('simpson', 'trapezoid')
            n_points: Number of points for numerical integration
        """
        self._domain = domain
        self._codomain = codomain
        self.N_d = codomain.dim
        self._kernels_provider = None
        self.cache_kernels = cache_kernels
        self._kernels_cache = {} if cache_kernels else None

        self._integration_method = integration_method
        self._npoints = n_points

        self._initialize_kernels(kernels)

        super().__init__(
            domain,
            codomain,
            self._mapping,
            dual_mapping=self._dual_mapping
        )

    # Define the mapping function
    def _mapping(self, f: 'Function') -> np.ndarray:
        """Apply kernel functions to input function via integration."""
        return self._apply_kernels(f)

    def _dual_mapping(self, yp: 'LinearForm') -> 'LinearFormKernel':
        """Reconstruct function from data using kernel functions."""
        kernel = self._reconstruct_function(yp.components)
        return LinearFormKernel(self.domain, kernel=kernel)

    def _initialize_kernels(
        self,
        kernels: Optional[
            Union[
                IndexedFunctionProvider,
                List[Union[Function, Callable]]
            ]
        ] = None
    ):
        if isinstance(kernels, list):
            if len(kernels) != self.N_d:
                raise ValueError(
                    f"Number of kernels ({len(kernels)}) must match "
                    f"codomain dimension ({self.N_d})"
                )
            if isinstance(kernels[0], Function):
                # Directly use provided Function instances
                self._kernels = kernels
            elif isinstance(kernels[0], Callable):
                # Convert callables to Function instances
                self._kernels = [
                    Function(
                        self._domain.function_domain,
                        evaluate_callable=func
                    )
                    for func in kernels
                ]
        elif isinstance(kernels, IndexedFunctionProvider):
            self._kernels_provider = kernels
            self._kernels = None

    def get_kernel(self, index: int):
        """
        Lazily get the i-th kernel with optional caching.

        Args:
            index: Index of the kernel to retrieve

        Returns:
            Function: The i-th kernel
        """
        # If kernels are directly provided, return from list
        if self._kernels is not None:
            return self._kernels[index]

        # Otherwise use the provider to get the kernel
        assert self._kernels_provider is not None  # For type checker
        return self._kernels_provider.get_function_by_index(index)

    def _apply_kernels(self, func):
        """
        Apply the kernel functions to a function by integrating their product.

        For each kernel k_i, computes ∫ func(x) * k_i(x) dx

        Args:
            func: Function from the domain space

        Returns:
            numpy.ndarray: Vector of data in R^{N_d}
        """
        data = np.zeros(self.N_d)

        for i in range(self.N_d):
            # Lazily get the i-th kernel
            kernel = self.get_kernel(i)
            # Compute integral of product: ∫ func(x) * kernel(x) dx
            # Avoid creating intermediate Function to prevent deep recursion
            def product_callable(x):
                return func.evaluate(x) * kernel.evaluate(x)

            product_func = Function(self.domain, evaluate_callable=product_callable)
            data[i] = product_func.integrate(
                method=self._integration_method,
                n_points=self._npoints
            )

        return data

    def _reconstruct_function(self, data):
        """
        Reconstruct a function from data using lazy evaluation.

        Args:
            data: numpy.ndarray of data in R^{N_d}

        Returns:
            Function: Reconstructed function in the domain space
        """
        # Collect non-zero terms to avoid deep recursion
        terms = []
        for i, coeff in enumerate(data):
            if abs(coeff) > 1e-14:  # Avoid numerical noise
                kernel = self.get_kernel(i)
                terms.append((coeff, kernel))

        # Create a single callable that evaluates all terms
        if not terms:
            return self.domain.zero

        def evaluate_sum(x):
            result = np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
            for coeff, kernel in terms:
                result = result + coeff * kernel.evaluate(x)
            return result

        return Function(self.domain, evaluate_callable=evaluate_sum)

    def get_kernels(self):
        """
        Get the list of kernels used by this operator.
        Note: This materializes all functions and may be expensive.

        Returns:
            list: List of kernels used for projection
        """
        return [self.get_kernel(i) for i in range(self.N_d)]

    def compute_gram_matrix(self):
        """
        Compute the Gram matrix of the kernels using function integration.

        For kernels k_i, k_j, computes ∫ k_i(x) * k_j(x) dx

        Returns:
            numpy.ndarray: N_d x N_d matrix of integrals between kernels
        """
        gram = np.zeros((self.N_d, self.N_d))

        for i in range(self.N_d):
            kernel_i = self.get_kernel(i)
            for j in range(self.N_d):
                kernel_j = self.get_kernel(j)
                # Compute integral: ∫ k_i(x) * k_j(x) dx
                # Avoid creating intermediate Function to prevent deep recursion

                def product_callable(x):
                    return kernel_i.evaluate(x) * kernel_j.evaluate(x)

                product_func = Function(
                    self.domain, evaluate_callable=product_callable
                )
                gram[i, j] = product_func.integrate(
                    method=self._integration_method,
                    n_points=self._npoints
                )

        return gram

    def clear_cache(self):
        """Clear the function cache if caching is enabled."""
        if self.cache_functions and self._function_cache is not None:
            self._function_cache.clear()

    def get_cache_info(self):
        """
        Get information about the function cache.

        Returns:
            dict: Cache statistics including size and hit rate
        """
        if not self.cache_functions:
            return {"caching_enabled": False}

        assert self._function_cache is not None  # For type checker
        return {
            "caching_enabled": True,
            "cached_functions": len(self._function_cache),
            "total_functions": self.N_d,
            "cache_coverage": len(self._function_cache) / self.N_d
        }

    def __str__(self):
        """String representation of the SOLA operator."""
        if self._use_direct_functions:
            return (f"SOLAOperator: {self.domain} -> {self.codomain}\n"
                    f"  Uses {self.N_d} direct function kernels\n"
                    f"  Domain dimension: {self.domain.dim}\n"
                    f"  Codomain dimension: {self.codomain.dim}")
        else:
            provider_type = type(self.function_provider).__name__
            return (f"SOLAOperator: {self.domain} -> {self.codomain}\n"
                    f"  Uses {self.N_d} kernels "
                    f"from {provider_type}\n"
                    f"  Domain dimension: {self.domain.dim}\n"
                    f"  Codomain dimension: {self.codomain.dim}")
