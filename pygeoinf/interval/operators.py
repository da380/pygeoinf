"""
Differential Operators for Interval Domains

This module provides implementations of common differential operators
on interval domains, with multiple discretization methods available.

Operators included:
- LaplacianOperator: The negative Laplacian operator (-Δ)
- LaplacianInverseOperator: The inverse negative Laplacian operator ((-Δ)^(-1))

Available discretization methods:
- 'spectral': Spectral methods using Fourier/sine/cosine basis
- 'finite_difference': Finite difference discretization
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Literal
import warnings

from .l2_space import L2Space
from .sobolev_space import Sobolev
from .boundary_conditions import BoundaryConditions
from pygeoinf.hilbert_space import LinearOperator
from .l2_functions import Function
from .providers import create_laplacian_spectrum_provider
# FEM import only needed for LaplacianInverseOperator
from pygeoinf.interval.fem_solvers import create_fem_solver


class DifferentialOperator(LinearOperator, ABC):
    """
    Abstract base class for differential operators on interval domains.

    Provides common functionality for discretization method selection
    and error handling.
    """

    def __init__(self, domain, codomain, boundary_conditions,
                 method='spectral'):
        self.boundary_conditions = boundary_conditions
        self.method = method
        self._validate_method()
        super().__init__(domain, codomain, self._apply)

    def _validate_method(self):
        """Validate the chosen discretization method."""
        valid_methods = ['spectral', 'finite_difference']
        if self.method not in valid_methods:
            raise ValueError(f"Unknown method '{self.method}'. "
                             f"Valid methods: {valid_methods}")

    @abstractmethod
    def _apply(self, f: Function) -> Function:
        """Apply the operator to a function."""
        pass


class LaplacianOperator(DifferentialOperator):
    """
    The negative Laplacian operator (-Δ) on interval domains.

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
        domain: Union[L2Space, Sobolev],
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
            domain: Function space (L2Space or SobolevSpace)
            boundary_conditions: Boundary conditions for the operator
            method: Discretization method ('spectral', 'finite_difference')
            dofs: Number of degrees of freedom for FD methods
                  (default: domain.dim)
            fd_order: Order of finite difference stencil (2, 4, 6)
        """
        self._domain = domain
        self._dofs = dofs or domain.dim
        self._fd_order = fd_order

        # Determine codomain based on method
        if method == 'spectral':
            # Spectral methods preserve the space (with proper basis)
            codomain = domain
        else:
            # FD/FE methods: Laplacian maps H^s → H^(s-2)
            # For Laplacian, domain should be at least H^2
            if isinstance(domain, Sobolev):
                # Ensure domain has sufficient regularity for Laplacian
                if domain.order < 2.0:
                    warnings.warn(
                        f"Laplacian requires domain with Sobolev order ≥ 2, "
                        f"got {domain.order}. Consider using H² space."
                    )

                # Create codomain with two orders less regularity
                target_order = domain.order - 2.0
                # Get basis type if available, default to fourier
                if hasattr(domain, '_basis_type'):
                    basis_type = domain._basis_type
                else:
                    basis_type = 'fourier'
                codomain = Sobolev(
                    domain.dim, domain.function_domain,
                    target_order, domain._inner_product_type,
                    basis_type=basis_type,
                    boundary_conditions=boundary_conditions
                )
            else:
                # For L2 domain (H⁰), Laplacian maps to H⁻²
                # Create H⁻² space
                codomain = Sobolev(
                    domain.dim, domain.function_domain,
                    -2.0, 'spectral',  # Default to spectral for negative order
                    basis_type='fourier',
                    boundary_conditions=boundary_conditions
                )

        super().__init__(domain, codomain, boundary_conditions, method)

        # Initialize method-specific components
        self._setup_method()

    def _setup_method(self):
        """Setup method-specific components."""
        if self.method == 'spectral':
            self._setup_spectral()
        elif self.method == 'finite_difference':
            self._setup_finite_difference()

    def _setup_spectral(self):
        """Setup spectral method using eigenfunction expansion."""
        # Create spectrum provider for eigenvalues/eigenfunctions
        self._spectrum_provider = create_laplacian_spectrum_provider(
            self._domain, self.boundary_conditions, inverse=False
        )

        # For spectral methods, we use the analytical eigendecomposition
        print(f"LaplacianOperator (spectral) initialized with "
              f"{self.boundary_conditions} boundary conditions")

    def _setup_finite_difference(self):
        """Setup finite difference discretization."""
        # Create finite difference grid
        a, b = self._domain.function_domain.a, self._domain.function_domain.b
        self._x_grid = np.linspace(a, b, self._dofs)
        self._dx = (b - a) / (self._dofs - 1)

        # Create finite difference matrix
        self._fd_matrix = self._create_fd_matrix()

        print(f"LaplacianOperator (finite difference, order {self._fd_order}) "
              f"initialized with {self._dofs} grid points")

    def _create_fd_matrix(self):
        """Create finite difference matrix for the negative Laplacian."""
        n = self._dofs
        dx2 = self._dx**2

        if self._fd_order == 2:
            # Second-order centered differences: [-1, 2, -1] / dx²
            matrix = np.zeros((n, n))

            # Interior points
            for i in range(1, n-1):
                matrix[i, i-1] = -1.0 / dx2
                matrix[i, i] = 2.0 / dx2
                matrix[i, i+1] = -1.0 / dx2

            # Apply boundary conditions
            self._apply_fd_boundary_conditions(matrix)

        elif self._fd_order == 4:
            # Fourth-order centered differences: [1, -8, 12, -8, 1] / 12dx²
            matrix = np.zeros((n, n))

            # Interior points (need at least 2 points from boundary)
            for i in range(2, n-2):
                matrix[i, i-2] = 1.0 / (12 * dx2)
                matrix[i, i-1] = -8.0 / (12 * dx2)
                matrix[i, i] = 12.0 / (12 * dx2)
                matrix[i, i+1] = -8.0 / (12 * dx2)
                matrix[i, i+2] = 1.0 / (12 * dx2)

            # Near-boundary points use second-order
            for i in [1, n-2]:
                matrix[i, i-1] = -1.0 / dx2
                matrix[i, i] = 2.0 / dx2
                matrix[i, i+1] = -1.0 / dx2

            self._apply_fd_boundary_conditions(matrix)

        else:
            raise ValueError(
                f"Finite difference order {self._fd_order} not implemented"
            )

        return matrix

    def _apply_fd_boundary_conditions(self, matrix):
        """Apply boundary conditions to finite difference matrix."""
        if self.boundary_conditions.type == 'dirichlet':
            # Dirichlet: u = 0 at boundaries
            matrix[0, :] = 0
            matrix[0, 0] = 1
            matrix[-1, :] = 0
            matrix[-1, -1] = 1

        elif self.boundary_conditions.type == 'neumann':
            # Neumann: du/dx = 0 at boundaries (second-order)
            # Left boundary: u_0 = u_1
            matrix[0, :] = 0
            matrix[0, 0] = -1
            matrix[0, 1] = 1

            # Right boundary: u_n = u_{n-1}
            matrix[-1, :] = 0
            matrix[-1, -1] = -1
            matrix[-1, -2] = 1

        elif self.boundary_conditions.type == 'periodic':
            # Periodic: handle wraparound
            # This is more complex and requires special treatment
            warnings.warn(
                "Periodic boundary conditions for FD not fully implemented"
            )

        else:
            raise ValueError(
                f"Boundary condition {self.boundary_conditions.type} "
                f"not supported for finite differences"
            )

    def _apply(self, f: Function) -> Function:
        """Apply the negative Laplacian operator to function f."""
        if self.method == 'spectral':
            return self._apply_spectral(f)
        elif self.method == 'finite_difference':
            return self._apply_finite_difference(f)
        else:
            raise ValueError(f"Unknown method '{self.method}'")

    def _apply_spectral(self, f: Function) -> Function:
        """Apply Laplacian using spectral method (eigenfunction expansion)."""
        # Project f onto eigenfunctions and apply eigenvalues
        if not hasattr(self._domain, 'to_components'):
            raise ValueError(
                "Spectral method requires domain with "
                "coefficient representation"
            )

        # Get coefficients of f in the eigenfunction basis
        coeffs = self._domain.to_components(f)

        # Apply eigenvalues (multiply by λᵢ for -Δ)
        laplacian_coeffs = np.zeros_like(coeffs)
        for i in range(len(coeffs)):
            eigenvalue = self._spectrum_provider.get_eigenvalue(i)
            laplacian_coeffs[i] = eigenvalue * coeffs[i]

        # Reconstruct function in the codomain
        return self.codomain.from_components(laplacian_coeffs)

    def _apply_finite_difference(self, f: Function) -> Function:
        """Apply Laplacian using finite difference method."""
        # Evaluate function on grid
        f_values = f(self._x_grid)

        # Apply finite difference matrix
        laplacian_values = self._fd_matrix @ f_values

        # Create result function by interpolation
        def laplacian_func(x):
            return np.interp(x, self._x_grid, laplacian_values)

        return Function(self.codomain, evaluate_callable=laplacian_func)


class LaplacianInverseOperator(LinearOperator):
    """
    Inverse Laplacian operator that acts as a covariance operator.

    This operator solves -Δu = f with homogeneous boundary conditions,
    providing a self-adjoint operator suitable for Gaussian measures.

    Supports multiple FEM backends:
    - 'dolfinx': Uses DOLFINx library (requires installation)
    - 'native': Pure Python implementation (no external dependencies)
    """

    def __init__(
        self,
        domain: L2Space,
        boundary_conditions: BoundaryConditions,
        /,
        *,
        alpha: float = 1.0,
        dofs: int = 100,
        solver_type: str = "auto"
    ):
        """
        Initialize the Laplacian inverse operator.

        The operator maps from L² space to Sobolev space:
        (-Δ)⁻¹: L² → H^s with specified boundary conditions

        Args:
            domain: The L2 space (domain of the operator)
            boundary_conditions: Boundary conditions for the operator
            sobolev_order: Order of the Sobolev space (codomain), default 1.0
            dofs: Number of degrees of freedom (default: 100)
            solver_type: 'dolfinx', 'native', or 'auto'
        """
        # Check that domain is an L2 space
        if not isinstance(domain, L2Space):
            raise TypeError(
                f"domain must be an L2 space, got {type(domain)}"
            )

        self._domain = domain
        self._dofs = dofs
        self._boundary_conditions = boundary_conditions

        # Create the Sobolev space as codomain
        # IN TESTING NOW!!!!
        """ self._codomain = Sobolev(
            domain.dim,
            domain.function_domain,
            2.0,
            'spectral',
            basis_type='fourier',  # Use Fourier basis for spectral approach
            boundary_conditions=boundary_conditions
        ) """
        self._codomain = domain # FOR TESTING ONLY

        # Get function domain from domain (L2Space and SobolevSpace)
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

        # Choose solver type (NOW ONLY NATIVE WORKS!)
        if solver_type == "auto":
            # Try DOLFINx first, fall back to native
            try:
                from .fem_solvers import DOLFINX_AVAILABLE
                if DOLFINX_AVAILABLE:
                    # Try to create DOLFINx solver to test compatibility
                    try:
                        test_bc = BoundaryConditions('dirichlet')
                        test_solver = create_fem_solver(
                            "dolfinx", self._function_domain, 10, test_bc
                        )
                        test_solver.setup()
                        self._solver_type = "dolfinx"
                    except Exception:
                        # DOLFINx available but has compatibility issues
                        self._solver_type = "native"
                        print("DOLFINx found but has compatibility issues, "
                              "using native solver")
                else:
                    self._solver_type = "native"
            except ImportError:
                self._solver_type = "native"
        else:
            self._solver_type = solver_type

        # Create and setup FEM solver
        self._fem_solver = create_fem_solver(
            self._solver_type,
            self._function_domain,
            dofs,
            self._boundary_conditions
        )
        self._fem_solver.setup()

        print(f"LaplacianInverseOperator initialized with {self._solver_type} "
              f"solver, {self._boundary_conditions} BCs")

        # Initialize LinearOperator with L2 domain and Sobolev codomain
        # The operator maps (-Δ)⁻¹: L² → H^s
        super().__init__(
            domain, self._codomain,
            self._solve_laplacian,
            adjoint_mapping=None  # Adjoint maps H^s → L²
        )

    def _solve_laplacian(self, f: Function) -> Function:
        """
        Apply the inverse Laplacian operator to function f.

        Args:
            f: L2Function or SobolevFunction to apply operator to

        Returns:
            L2Function representing (-Δ)⁻¹f
        """
        f = (1 / self._alpha) * f  # Scale input by alpha

        # Solve PDE using FEM solver with L2Function
        solution_values = self._fem_solver.solve_poisson(f)

        # Convert back to hilbert space function
        return self._fem_to_hilbert_function(solution_values)

    def _fem_to_hilbert_function(self, fem_solution):
        """Convert FEM solution to hilbert space function."""

        # Create interpolation function
        def evaluate_func(x):
            return self._fem_solver.interpolate_solution(fem_solution, x)

        # Create Function with interpolated solution
        return Function(
            self._domain,
            evaluate_callable=evaluate_func,
            name=f"Laplacian⁻¹ solution ({self._solver_type})"
        )

    @property
    def solver_type(self) -> str:
        """Get the current solver type."""
        return self._solver_type

    @property
    def boundary_conditions(self) -> BoundaryConditions:
        """Get the boundary conditions object."""
        return self._boundary_conditions

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
        return self._spectrum_provider.get_eigenvalue(index)

    def get_eigenfunction(self, index: int) -> Function:
        """
        Get eigenfunction of the inverse Laplacian operator.

        Args:
            index: Index of the eigenfunction (0 to dim-1)

        Returns:
            Function: Eigenfunction φₖ of (-Δ)^(-1)
        """
        return self._spectrum_provider.get_basis_function(index)

    def get_all_eigenvalues(self, n: int | None = None) -> np.ndarray:
        """
        Get all eigenvalues of the inverse Laplacian operator.
        Args:
            n: Number of eigenvalues to compute (default: space.dim)
        Returns:
            np.ndarray: Array of all eigenvalues
        """
        if n is None:
            n = self._domain.dim
        return self._spectrum_provider.get_all_eigenvalues(n)

    def get_solver_info(self) -> dict:
        """Get information about the current solver."""
        return {
            'solver_type': self._solver_type,
            'boundary_conditions': self._boundary_conditions,
            'dofs': self._dofs,
            'fem_coordinates': self._fem_solver.get_coordinates(),
            'spectrum_available': True,
            'eigenvalue_range': (
                self.get_eigenvalue(0) if self._domain.dim > 0 else None,
                (self.get_eigenvalue(self._domain.dim - 1)
                 if self._domain.dim > 0 else None)
            )
        }

    def change_solver(self, new_solver_type: str):
        """
        Change the FEM solver backend.

        Args:
            new_solver_type: 'dolfinx' or 'native'

        Note:
            This does not affect the spectrum computation, which is
            analytical and independent of the FEM solver.
        """
        if new_solver_type == self._solver_type:
            print(f"Already using {new_solver_type} solver")
            return

        # Create new solver
        self._solver_type = new_solver_type
        self._fem_solver = create_fem_solver(
            self._solver_type,
            self._function_domain,
            self._dofs,
            self._boundary_conditions
        )
        self._fem_solver.setup()

        print(f"Switched to {self._solver_type} solver")
