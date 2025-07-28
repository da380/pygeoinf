"""
Laplacian Inverse Operator with Multiple FEM Backend Support

This module provides the LaplacianInverseOperator that can use different
FEM solvers as backends (DOLFINx or native Python implementation).
"""

import numpy as np
import math
from .fem_solvers import create_fem_solver
from .l2_space import L2Space
from .sobolev_space import Sobolev
from .boundary_conditions import BoundaryConditions
from pygeoinf.hilbert_space import LinearOperator
from .l2_functions import Function
from .providers import SpectrumProvider


class LaplacianInverseSpectrumProvider(SpectrumProvider):
    """
    Spectrum provider for the inverse Laplacian operator.

    Computes eigenvalues and eigenfunctions of (-Δ)^(-1) based on the
    boundary conditions and domain. Uses lazy evaluation to compute
    spectral information only when needed.
    """

    def __init__(self, space, boundary_conditions):
        """
        Initialize the spectrum provider.

        Args:
            space: Hilbert space (domain for the operator)
            boundary_conditions: Boundary conditions determining the spectrum
        """
        super().__init__(space)
        self.space = space
        self.function_domain = space.function_domain
        self.boundary_conditions = boundary_conditions
        self._eigenvalue_cache = {}
        self._eigenfunction_cache = {}
        self._all_eigenvalues = None

    def get_eigenvalue(self, index: int):
        """
        Get eigenvalue of (-Δ)^(-1) for given index.

        For Dirichlet boundary conditions on [a,b]:
        λₖ = 1 / (kπ/(b-a))² = (b-a)²/(kπ)²

        For Neumann boundary conditions:
        λ₀ = ∞ (constant mode), λₖ = (b-a)²/(kπ)² for k ≥ 1

        For periodic boundary conditions:
        λ₀ = ∞ (constant mode), λₖ = (b-a)²/(2πk)² for k ≥ 1
        """
        if not (0 <= index < self.space.dim):
            raise IndexError(
                f"Eigenvalue index {index} out of range [0, {self.space.dim})"
            )

        if index not in self._eigenvalue_cache:
            self._eigenvalue_cache[index] = self._compute_eigenvalue(index)
        return self._eigenvalue_cache[index]

    def _compute_eigenvalue(self, index: int):
        """Compute single eigenvalue."""
        length = self.function_domain.b - self.function_domain.a

        if self.boundary_conditions.type == 'dirichlet':
            # For Dirichlet: λₖ = L²/(kπ)² where k = index + 1
            k = index + 1
            return (length / (k * math.pi))**2

        elif self.boundary_conditions.type == 'neumann':
            if index == 0:
                # Constant mode for Neumann has infinite eigenvalue
                # In practice, we use a very large value or handle separately
                return float('inf')
            else:
                # For k ≥ 1: λₖ = L²/(kπ)²
                k = index
                return (length / (k * math.pi))**2

        elif self.boundary_conditions.type == 'periodic':
            if index == 0:
                # Constant mode for periodic has infinite eigenvalue
                return float('inf')
            else:
                # For periodic: both cos and sin modes have λₖ = L²/(2πk)²
                k = (index + 1) // 2  # Frequency index
                return (length / (2 * k * math.pi))**2

        else:
            raise ValueError(f"Unsupported boundary condition type: "
                             f"{self.boundary_conditions.type}")

    def get_basis_function(self, index: int):
        """
        Get eigenfunction of (-Δ)^(-1) for given index.

        These are the same as the eigenfunctions of -Δ, since
        (-Δ)^(-1) and -Δ have the same eigenfunctions.
        """
        if not (0 <= index < self.space.dim):
            raise IndexError(
                f"Eigenfunction index {index} out of range [0, {self.space.dim})"
            )

        if index not in self._eigenfunction_cache:
            self._eigenfunction_cache[index] = self._compute_eigenfunction(index)
        return self._eigenfunction_cache[index]

    def _compute_eigenfunction(self, index: int):
        """Compute eigenfunction based on boundary conditions."""
        a, b = self.function_domain.a, self.function_domain.b
        length = b - a

        if self.boundary_conditions.type == 'dirichlet':
            # sin(kπ(x-a)/L) for k = index + 1
            k = index + 1

            def eigenfunction(x):
                if isinstance(x, np.ndarray):
                    return np.sin(k * np.pi * (x - a) / length)
                else:
                    return math.sin(k * math.pi * (x - a) / length)

            return Function(
                self.space,
                evaluate_callable=eigenfunction,
                name=f"sin({k}π(x-{a})/{length})"
            )

        elif self.boundary_conditions.type == 'neumann':
            if index == 0:
                # Constant mode
                def constant_func(x):
                    return np.ones_like(x) if isinstance(x, np.ndarray) else 1.0

                return Function(
                    self.space,
                    evaluate_callable=constant_func,
                    name="1 (constant)"
                )
            else:
                # cos(kπ(x-a)/L) for k = index
                k = index

                def eigenfunction(x):
                    if isinstance(x, np.ndarray):
                        return np.cos(k * np.pi * (x - a) / length)
                    else:
                        return math.cos(k * math.pi * (x - a) / length)

                return Function(
                    self.space,
                    evaluate_callable=eigenfunction,
                    name=f"cos({k}π(x-{a})/{length})"
                )

        elif self.boundary_conditions.type == 'periodic':
            if index == 0:
                # Constant mode
                def constant_func(x):
                    return np.ones_like(x) if isinstance(x, np.ndarray) else 1.0

                return Function(
                    self.space,
                    evaluate_callable=constant_func,
                    name="1 (constant)"
                )
            else:
                # Alternate between cos and sin modes
                k = (index + 1) // 2  # Frequency index
                is_sin = (index % 2 == 0)  # Even indices are sin, odd are cos

                if is_sin:
                    def eigenfunction(x):
                        if isinstance(x, np.ndarray):
                            return np.sin(2 * k * np.pi * (x - a) / length)
                        else:
                            return math.sin(2 * k * math.pi * (x - a) / length)

                    return Function(
                        self.space,
                        evaluate_callable=eigenfunction,
                        name=f"sin(2π{k}(x-{a})/{length})"
                    )
                else:
                    def eigenfunction(x):
                        if isinstance(x, np.ndarray):
                            return np.cos(2 * k * np.pi * (x - a) / length)
                        else:
                            return math.cos(2 * k * math.pi * (x - a) / length)

                    return Function(
                        self.space,
                        evaluate_callable=eigenfunction,
                        name=f"cos(2π{k}(x-{a})/{length})"
                    )
        else:
            raise ValueError(f"Unsupported boundary condition type: "
                             f"{self.boundary_conditions.type}")

    def get_all_eigenvalues(self):
        """Return all eigenvalues as an array."""
        if self._all_eigenvalues is None:
            self._all_eigenvalues = np.array([
                self.get_eigenvalue(i) for i in range(self.space.dim)
            ])
        return self._all_eigenvalues

    def clear_cache(self):
        """Clear all cached computations."""
        self._eigenvalue_cache.clear()
        self._eigenfunction_cache.clear()
        self._all_eigenvalues = None


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
        self._codomain = Sobolev(
            domain.dim,
            domain.function_domain,
            2.0,
            'spectral',
            basis_type='fourier',  # Use Fourier basis for spectral approach
            boundary_conditions=boundary_conditions
        )

        # Get function domain from domain (L2Space and SobolevSpace)
        self._function_domain = domain.function_domain
        self._interval = (self._function_domain.a, self._function_domain.b)

        # Initialize spectrum provider for lazy eigenvalue computation
        self._spectrum_provider = LaplacianInverseSpectrumProvider(
            domain, self._boundary_conditions
        )

        # Choose solver type
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
    def spectrum_provider(self) -> LaplacianInverseSpectrumProvider:
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

    def get_all_eigenvalues(self) -> np.ndarray:
        """
        Get all eigenvalues of the inverse Laplacian operator.

        Returns:
            np.ndarray: Array of all eigenvalues
        """
        return self._spectrum_provider.get_all_eigenvalues()

    def clear_spectrum_cache(self):
        """Clear cached spectrum computations."""
        self._spectrum_provider.clear_cache()

    def get_solver_info(self) -> dict:
        """Get information about the current solver."""
        return {
            'solver_type': self._solver_type,
            'boundary_conditions': self._boundary_conditions,
            'dofs': self._dofs,
            'interval': self._interval,
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
