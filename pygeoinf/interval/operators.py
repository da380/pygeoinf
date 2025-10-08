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

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union, Literal, List, Callable

from pygeoinf.hilbert_space import EuclideanSpace
from .lebesgue_space import Lebesgue
from .sobolev_space import Sobolev
from .boundary_conditions import BoundaryConditions
from pygeoinf.linear_operators import LinearOperator
from .functions import Function
from .providers import create_laplacian_spectrum_provider
# FEM import only needed for LaplacianInverseOperator
from pygeoinf.interval.fem_solvers import GeneralFEMSolver
from pygeoinf.interval.function_providers import (
    IndexedFunctionProvider,
)

class SpectralOperator(LinearOperator, ABC):
    """
    Abstract base class for spectral operators on interval domains.

    Provides common functionality for eigenvalue/eigenfunction access
    and error handling.
    """

    def __init__(self, domain, codomain, spectrum_provider: IndexedFunctionProvider):
        self._spectrum_provider = spectrum_provider
        super().__init__(domain, codomain, self._apply)

    @abstractmethod
    def _apply(self, f: Function) -> Function:
        """Apply the operator to a function."""
        pass

    def get_eigenvalue(self, index: int) -> float:
        """Get the eigenvalue at a specific index."""
        return self._spectrum_provider.get_eigenvalue(index)

    def get_eigenfunction(self, index: int) -> Function:
        """Get the eigenfunction at a specific index."""
        return self._spectrum_provider.get_eigenfunction(index)


class DifferentialOperator(LinearOperator, ABC):
    """
    Abstract base class for differential operators on interval domains.

    Provides common functionality for discretization method selection
    and error handling.
    """

    def __init__(self, domain, codomain, boundary_conditions):
        self.boundary_conditions = boundary_conditions
        super().__init__(domain, codomain, self._apply)

    @abstractmethod
    def _apply(self, f: Function) -> Function:
        """Apply the operator to a function."""
        pass


class GradientOperator(DifferentialOperator):
    """
    The gradient operator (d/dx) on interval domains.

    In 1D, the gradient is simply the first derivative. This operator
    provides multiple methods for computing derivatives:

    - 'finite_difference': Numerical differentiation using finite differences
    - 'automatic': Automatic differentiation using JAX (if available)

    The operator maps between function spaces:
    - H^s → H^(s-1) (reduces regularity by one order)
    """

    def __init__(
        self,
        domain: Union[Lebesgue, Sobolev],
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
            method: Differentiation method
                ('finite_difference', 'automatic')
            fd_order: Order of finite difference stencil (2, 4, 6)
                for FD method
            fd_step: Step size for finite differences (auto-computed if None)
            boundary_treatment: How to handle boundaries for FD
                ('one_sided', 'extrapolate')
        """
        self._domain = domain
        self._codomain = domain
        self._fd_order = fd_order
        self._fd_step = fd_step
        self._boundary_treatment = boundary_treatment
        # logger
        self._log = logging.getLogger(__name__)

        # No boundary conditions needed for gradient (unlike Laplacian)
        super().__init__(domain, domain, None)

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


class Laplacian(DifferentialOperator):
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
        domain: Lebesgue,
        boundary_conditions: BoundaryConditions,
        /,
        *,
        method: Literal['spectral', 'finite_difference'] = 'spectral',
        dofs: Optional[int] = None,
        fd_order: int = 2
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
        """
        self._domain = domain
        self._dofs = dofs if dofs is not None else domain.dim
        self._fd_order = fd_order
        self._method = method

        super().__init__(domain, domain, boundary_conditions)

        # Initialize method-specific components
        self._spectrum_provider = create_laplacian_spectrum_provider(
            self._domain, self.boundary_conditions, inverse=False
        )
        if self._method == 'finite_difference':
            self._setup_finite_difference()

    def get_eigenvalue(self, index: int) -> float:
        """Get the eigenvalue at a specific index."""
        return self._spectrum_provider.get_eigenvalue(index)

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
        elif self._method == 'finite_difference':
            return self._apply_finite_difference(f)
        else:
            raise ValueError(f"Unknown method '{self._method}'")

    def _apply_spectral(self, f: Function) -> Function:
        """Apply Laplacian using spectral method (eigenfunction expansion)."""
        # Project f onto eigenfunctions and apply eigenvalues
        f_new = self._domain.zero
        for i in range(self._dofs):
            # Get the basis functions
            basis_func = self.get_eigenfunction(i)
            eigenvalue = self.get_eigenvalue(i)
            # Compute coefficient via inner product
            coeff = (
                (basis_func * f).integrate(method='simpson', n_points=10000)
            )
            coeff *= eigenvalue  # Scale by eigenvalue

            f_new += coeff * basis_func

        return f_new

    def _apply_finite_difference(self, f: Function) -> Function:
        """Apply Laplacian using finite difference method."""
        # Evaluate function on grid
        f_values = f(self._x_grid)

        # Apply finite difference matrix (this is where we apply the negative 1!!)
        laplacian_values = -self._fd_matrix @ f_values

        # Create result function by interpolation
        def laplacian_func(x):
            return np.interp(x, self._x_grid, laplacian_values)

        return Function(self.codomain, evaluate_callable=laplacian_func)


class InverseLaplacian(LinearOperator):
    """
    Inverse Laplacian operator that acts as a covariance operator.

    This operator solves -Δu = f with homogeneous boundary conditions,
    providing a self-adjoint operator suitable for Gaussian measures.

    Uses native Python FEM implementation (no external dependencies).
    """

    def __init__(
        self,
        domain: Lebesgue,
        boundary_conditions: BoundaryConditions,
        /,
        *,
        alpha: float = 1.0,
        dofs: int = 100,
        fem_type: str = "hat"
    ):
        """
        Initialize the Laplacian inverse operator.

        The operator maps from L² space to Sobolev space:
        (-Δ)⁻¹: L² → H^s with specified boundary conditions

        Args:
            domain: The L2 space (domain of the operator)
            boundary_conditions: Boundary conditions for the operator
            alpha: Scaling factor for the operator (default: 1.0)
            dofs: Number of degrees of freedom (default: 100)
            fem_type: FEM implementation to use (deprecated):
                - "hat": Uses hat function basis (default)
                - "general": Uses domain's basis functions
                Note: Both options now use GeneralFEMSolver internally.
        """
        # Check that domain is a Lebesgue space
        if not isinstance(domain, Lebesgue):
            raise TypeError(
                f"domain must be a Lebesgue space, got {type(domain)}"
            )

        self._domain = domain
        self._dofs = dofs
        self._boundary_conditions = boundary_conditions
        self._fem_type = fem_type

        # Validate fem_type
        if fem_type not in ["hat", "general"]:
            raise ValueError(
                f"fem_type must be 'hat' or 'general', got '{fem_type}'"
            )

        self._codomain = domain  # FOR TESTING ONLY

        # Get function domain from domain (Lebesgue and SobolevSpace)
        self._function_domain = domain.function_domain

        # Store scaling factor
        if not isinstance(alpha, (int, float)):
            raise TypeError(f"alpha must be a number, got {type(alpha)}")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self._alpha = alpha

        # Initialize spectrum provider for lazy eigenvalue computation
        self._spectrum_provider = create_laplacian_spectrum_provider(
            domain, self._boundary_conditions, inverse=True
        )

        self._initialize_fem_solver()

        # logger
        self._log = logging.getLogger(__name__)
        self._log.info(
            "InverseLaplacian initialized: dofs=%s, fem_type=%s, alpha=%s",
            self._dofs,
            self._fem_type,
            self._alpha,
        )

        # Initialize LinearOperator with L2 domain and Sobolev codomain
        # The operator maps (-Δ)⁻¹: L² → H^s
        super().__init__(
            self._domain, self._codomain,
            self._solve_laplacian,
            adjoint_mapping=None  # Adjoint maps H^s → L²
        )

    def _initialize_fem_solver(self):
        # Create and setup FEM solver
        self._fem_solver = GeneralFEMSolver(
            function_domain=self._function_domain,
            dofs=self._dofs,
            operator_domain=self._domain,
            boundary_conditions=self._boundary_conditions
        )

    def _solve_laplacian(self, f: Function) -> Function:
        """
        Apply the inverse Laplacian operator to function f.

        Args:
            f: L2Function or SobolevFunction to apply operator to

        Returns:
            L2Function representing (-Δ)⁻¹f
        """
        f = self._alpha * f  # Scale input by alpha

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
        """
        Get eigenvalue of the inverse Laplacian operator.

        Args:
            index: Index of the eigenvalue (0 to dim-1)

        Returns:
            float: Eigenvalue λₖ of (-Δ)^(-1)
        """
        return self._alpha * self._spectrum_provider.get_eigenvalue(index)

    def get_eigenfunction(self, index: int) -> Function:
        """
        Get eigenfunction of the inverse Laplacian operator.

        Args:
            index: Index of the eigenfunction (0 to dim-1)

        Returns:
            Function: Eigenfunction φₖ of (-Δ)^(-1)
        """
        return self._spectrum_provider.get_basis_function(index)

    from typing import Optional

    def get_all_eigenvalues(self, n: Optional[int] = None) -> np.ndarray:
        """
        Get all eigenvalues of the inverse Laplacian operator.
        Args:
            n: Number of eigenvalues to compute (default: space.dim)
        Returns:
            np.ndarray: Array of all eigenvalues
        """
        if n is None:
            n = self._domain.dim
        return self._alpha * self._spectrum_provider.get_all_eigenvalues(n)

    def get_solver_info(self) -> dict:
        """Get information about the current solver."""
        # Some FEM solvers may not implement get_coordinates; guard access
        try:
            coords = self._fem_solver.get_coordinates()
        except Exception:
            coords = None

        return {
            'fem_type': self._fem_type,
            'boundary_conditions': self._boundary_conditions,
            'dofs': self._dofs,
            'fem_coordinates': coords,
            'spectrum_available': True,
            'eigenvalue_range': (
                self.get_eigenvalue(0) if self._domain.dim > 0 else None,
                (self.get_eigenvalue(self._domain.dim - 1)
                 if self._domain.dim > 0 else None)
            )
        }


class BesselSobolev(LinearOperator):
    """
    Bessel potential operator that maps between Sobolev and Lebesgue spaces.

    This operator implements the Bessel potential of order s, which acts as
    a smoothing operator. It maps functions from a Sobolev space H^s to a
    Lebesgue space L², effectively reducing the regularity of the function.

    The operator is defined via its action on the Fourier coefficients of
    the input function, scaling them by (k^2I - \Delta)^(s).

    The operator maps: Sobolev -> Lebesgue

    Examples:
        # Create a Sobolev space H^1 on [0, 1] with 100 basis functions
        >>> from pygeoinf.interval import Sobolev, BoundaryConditions
        >>> sobolev_space = Sobolev(1, (0, 1), 100, BoundaryConditions('dirichlet'))

        # Create a Lebesgue space L2 on [0, 1] with 100 basis functions
        >>> from pygeoinf.interval import Lebesgue
        >>> lebesgue_space = Lebesgue((0, 1), 100)

        # Create the Bessel potential operator of order 1
        >>> bessel_op = BeselSobolev(sobolev_space, lebesgue_space)

        # Apply the operator to a function in H^1
        >>> from pygeoinf.interval import Function
        >>> f = Function(sobolev_space, evaluate_callable=lambda x: x*(1-x))
        >>> g = bessel_op(f)
    """

    def __init__(self, domain: Lebesgue, codomain: Lebesgue, k: float, s: float, L: SpectralOperator, dofs: Optional[int] = None):
        """
        Initialize the Bessel potential operator.

        Args:
            domain: Sobolev instance (the function space H^s)
            codomain: Lebesgue instance (the function space L²)
        """

        self._domain = domain
        self._codomain = codomain
        self._L = L
        self._k = k
        self._s = s
        self._dofs = dofs if dofs is not None else domain.dim

        super().__init__(domain, codomain, self._apply)


    def _apply(self, f: Function) -> Function:
        """
        Apply the Bessel potential operator to a function.

        Args:
            f: Function from the domain space

        Returns:
            Function: Resulting function in the codomain space
        """
        for i in range(self._dofs):
            eigval = self._L.get_eigenvalue(i)
            if eigval is None:
                raise ValueError("Eigenvalue not available for index {i}")
            if eigval < 0:
                raise ValueError(f"Negative eigenvalue {eigval} at index {i}")
            eigfunc = self._L.get_eigenfunction(i)
            if eigfunc is None:
                raise ValueError("Eigenfunction not available for index {i}")
            coeff = (f * eigfunc).integrate(method='simpson', n_points=10000)
            scale = (self._k**2 + eigval)**(self._s)
            if i == 0:
                f_new = scale * coeff * eigfunc # Initialize
            else:
                f_new += scale * coeff * eigfunc

        return f_new

    def get_eigenfunction(self, index: int) -> Function:
        """Get the eigenfunction at a specific index."""
        return self._L.get_eigenfunction(index)

    def get_eigenvalue(self, index: int) -> float:
        """Get the eigenvalue at a specific index."""
        eigval = self._L.get_eigenvalue(index)
        if eigval is None:
            raise ValueError("Eigenvalue not available for index {index}")
        if eigval < 0:
            raise ValueError(f"Negative eigenvalue {eigval} at index {index}")
        return (self._k**2 + eigval)**(self._s)

class BesselSobolevInverse(LinearOperator):
    """
    Inverse Bessel potential operator that maps between Lebesgue and Sobolev spaces.

    This operator implements the inverse of the Bessel potential of order s,
    which acts as a smoothing operator. It maps functions from a Lebesgue space
    L² to a Sobolev space H^s, effectively increasing the regularity of the function.

    The operator is defined via its action on the Fourier coefficients of
    the input function, scaling them by (k^2I - \Delta)^(-s).

    The operator maps: Lebesgue -> Sobolev

    Examples:
        # Create a Lebesgue space L2 on [0, 1] with 100 basis functions
        >>> from pygeoinf.interval import Lebesgue
        >>> lebesgue_space = Lebesgue((0, 1), 100)

        # Create a Sobolev space H^1 on [0, 1] with 100 basis functions
        >>> from pygeoinf.interval import Sobolev, BoundaryConditions
        >>> sobolev_space = Sobolev(1, (0, 1), 100, BoundaryConditions('dirichlet'))
        # Create the inverse Bessel potential operator of order 1
        >>> bessel_inv_op = BesselSobolevInverse(lebesgue_space, sobolev_space)
        # Apply the operator to a function in L2
        >>> from pygeoinf.interval import Function
        >>> f = Function(lebesgue_space, evaluate_callable=lambda x: np.sin(np.pi*x))
        >>> g = bessel_inv_op(f)
    """

    def __init__(self, domain: Lebesgue, codomain: Lebesgue, k: float, s: float, L: SpectralOperator, dofs: Optional[int] = None):
        """
        Initialize the inverse Bessel potential operator.

        Args:
            domain: Lebesgue instance (the function space L²)
            codomain: Sobolev instance (the function space H^s)
        """

        self._domain = domain
        self._codomain = codomain
        self._L = L
        self._k = k
        self._s = s
        self._dofs = dofs if dofs is not None else domain.dim

        super().__init__(domain, codomain, self._apply)

    def _apply(self, f: Function) -> Function:
        """
        Apply the inverse Bessel potential operator to a function.

        Args:
            f: Function from the domain space

        Returns:
            Function: Resulting function in the codomain space
        """
        for i in range(self._dofs):
            eigval = self._L.get_eigenvalue(i)
            if eigval is None:
                raise ValueError("Eigenvalue not available for index {i}")
            if eigval < 0:
                raise ValueError(f"Negative eigenvalue {eigval} at index {i}")
            eigfunc = self._L.get_eigenfunction(i)
            if eigfunc is None:
                raise ValueError("Eigenfunction not available for index {i}")
            coeff = (f * eigfunc).integrate(method='simpson', n_points=10000)
            scale = (self._k**2 + eigval)**(-self._s)
            if i == 0:
                f_new = scale * coeff * eigfunc # Initialize
            else:
                f_new += scale * coeff * eigfunc

        return f_new

    def get_eigenfunction(self, index: int) -> Function:
        """Get the eigenfunction at a specific index."""
        return self._L.get_eigenfunction(index)

    def get_eigenvalue(self, index: int) -> float:
        """Get the eigenvalue at a specific index."""
        eigval = self._L.get_eigenvalue(index)
        if eigval is None:
            raise ValueError("Eigenvalue not available for index {index}")
        if eigval < 0:
            raise ValueError(f"Negative eigenvalue {eigval} at index {index}")
        return (self._k**2 + eigval)**(-self._s)


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
        domain: Lebesgue,
        codomain: EuclideanSpace,
        kernels: Optional[
            Union[
                IndexedFunctionProvider,
                List[Union[Function, Callable]]
            ]
        ] = None,
        cache_kernels: bool = False
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

        self._initialize_kernels(kernels)

        # Define the mapping function
        def mapping(func):
            """Apply kernel functions to input function via integration."""
            return self._apply_kernels(func)

        # Define the adjoint mapping
        def adjoint_mapping(data):
            """Reconstruct function from data."""
            return self._reconstruct_function(data)

        super().__init__(
            domain,
            codomain,
            mapping,
            adjoint_mapping=adjoint_mapping
        )

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
                    Function(self._domain.function_domain, evaluate_callable=func)
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
            data[i] = (func * kernel).integrate()

        return data

    def _reconstruct_function(self, data):
        """
        Reconstruct a function from data using lazy evaluation.

        Args:
            data: numpy.ndarray of data in R^{N_d}

        Returns:
            Function: Reconstructed function in the domain space
        """
        # Start with zero function
        reconstructed = self.domain.zero

        # Add weighted kernels
        for i, coeff in enumerate(data):
            if abs(coeff) > 1e-14:  # Avoid numerical noise
                # Lazily get the i-th kernel
                proj_func = self.get_kernel(i)
                reconstructed += coeff * proj_func

        return reconstructed

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
                gram[i, j] = (kernel_i * kernel_j).integrate()

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
