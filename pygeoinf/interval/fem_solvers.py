"""
FEM Solvers for Laplacian Inverse Operator

This module provides Python FEM solvers for the LaplacianInverseOperator.
Two implementations are available:

1. GeneralFEMSolver: Works with any basis functions from L2Space
2. FEMSolver: Optimized for hat basis functions (legacy implementation)

Both solvers have no external dependencies.
"""

import numpy as np

from .interval_domain import IntervalDomain
from .boundary_conditions import BoundaryConditions
from .functions import Function


class GeneralFEMSolver:
    """
    General FEM solver that works with any basis functions from L2Space.

    This implementation follows the general finite element method described
    in the mathematical formulation, where basis functions {φᵢ} can be any
    suitable functions that span the finite-dimensional subspace Vₕ ⊂ H₀¹(a,b).

    The solver extracts basis functions from the L2Space basis provider and
    assembles the stiffness matrix and load vector using numerical integration:

    [K]ᵢⱼ = ∫ φ'ᵢ(x) φ'ⱼ(x) dx
    [F]ᵢ = ∫ f(x) φᵢ(x) dx
    """

    def __init__(self, l2_space, boundary_conditions: BoundaryConditions):
        """
        Initialize general FEM solver with L2Space basis functions.

        Args:
            l2_space: L2Space object providing the basis functions
            boundary_conditions: BoundaryConditions object (must be Dirichlet)
        """
        # Import here to avoid circular imports
        from .l2_space import L2Space

        if not isinstance(l2_space, L2Space):
            raise TypeError(
                f"l2_space must be L2Space object, got {type(l2_space)}"
            )

        if not isinstance(boundary_conditions, BoundaryConditions):
            raise TypeError(
                f"boundary_conditions must be BoundaryConditions object, "
                f"got {type(boundary_conditions)}"
            )

        # Currently only support Dirichlet boundary conditions
        if boundary_conditions.type != 'dirichlet':
            raise ValueError(
                f"GeneralFEMSolver only supports 'dirichlet' boundary "
                f"conditions, got '{boundary_conditions.type}'"
            )

        # Check that the L2Space basis is compatible with Dirichlet BCs
        # For Dirichlet BCs, basis functions should vanish at boundaries
        if hasattr(l2_space, '_basis_type'):
            basis_type = l2_space._basis_type
            if basis_type not in ['hat_homogeneous', 'fourier_sin']:
                print(f"Warning: L2Space basis type '{basis_type}' may not "
                      f"satisfy homogeneous Dirichlet boundary conditions. "
                      f"Consider using 'hat_homogeneous' or 'fourier_sin'.")
        else:
            print("Warning: L2Space basis type unknown. Ensure basis "
                  "functions vanish at boundaries for Dirichlet BCs.")

        self.l2_space = l2_space
        self.boundary_conditions = boundary_conditions
        self.function_domain = l2_space.function_domain
        self.dofs = l2_space.dim

        # Integration parameters for numerical assembly
        self.integration_method = 'simpson'
        self.n_integration_points = 1000

        # Storage for assembled matrices
        self._stiffness_matrix = None
        self._mass_matrix = None
        self._is_setup = False

    @property
    def bc_type(self) -> str:
        """Get the boundary condition type for backward compatibility."""
        return self.boundary_conditions.type

    def setup(self):
        """Set up the FEM solver by pre-assembling matrices."""
        print(f"Setting up GeneralFEMSolver with {self.dofs} basis functions "
              f"from {type(self.l2_space._basis_provider).__name__}")

        # Pre-assemble the stiffness matrix (does not depend on RHS)
        self._stiffness_matrix = self._assemble_stiffness_matrix()

        # Optionally pre-assemble mass matrix (useful for some algorithms)
        self._mass_matrix = self.l2_space.gram_matrix

        self._is_setup = True

    def _assemble_stiffness_matrix(self) -> np.ndarray:
        """
        Assemble the stiffness matrix K where [K]ᵢⱼ = ∫ φ'ᵢ(x) φ'ⱼ(x) dx.

        This uses numerical differentiation and integration to compute
        the entries for arbitrary basis functions.
        """
        n = self.dofs
        K = np.zeros((n, n))

        # Integration domain
        a, b = self.function_domain.a, self.function_domain.b
        x_quad = np.linspace(a, b, self.n_integration_points)
        dx = (b - a) / (self.n_integration_points - 1)

        # Get basis functions and compute their derivatives numerically
        basis_functions = []
        basis_derivatives = []

        for i in range(n):
            phi_i = self.l2_space._basis_provider.get_basis_function(i)
            basis_functions.append(phi_i)

            # Compute derivative using central differences
            phi_i_values = phi_i(x_quad)
            phi_i_prime = np.gradient(phi_i_values, dx)
            basis_derivatives.append(phi_i_prime)

        # Assemble stiffness matrix using numerical integration
        for i in range(n):
            for j in range(i, n):  # Exploit symmetry
                # Integrate φ'ᵢ * φ'ⱼ using Simpson's rule
                integrand = basis_derivatives[i] * basis_derivatives[j]
                integral = np.trapz(integrand, x_quad)

                K[i, j] = integral
                if i != j:
                    K[j, i] = integral  # Symmetric matrix

        return K

    def _assemble_load_vector(self, rhs_function: Function) -> np.ndarray:
        """
        Assemble the load vector F where [F]ᵢ = ∫ f(x) φᵢ(x) dx.

        This leverages the Function integration capabilities.
        """
        f_vec = np.zeros(self.dofs)

        for i in range(self.dofs):
            # Get basis function as Function
            phi_i = self.l2_space._basis_provider.get_basis_function(i)

            # Compute ∫ f(x) φᵢ(x) dx using Function operations
            integrand = rhs_function * phi_i
            f_vec[i] = integrand.integrate(method=self.integration_method)

        return f_vec

    def solve_poisson(self, rhs_function: Function) -> np.ndarray:
        """
        Solve the Poisson equation -d²u/dx² = f with Dirichlet BCs.

        Args:
            rhs_function: Right-hand side function f(x)

        Returns:
            Coefficients of the solution in the basis {φᵢ}
        """
        if not self._is_setup:
            raise RuntimeError("Must call setup() first")

        if not isinstance(rhs_function, Function):
            raise TypeError(
                f"rhs_function must be Function, got {type(rhs_function)}"
            )

        # Assemble load vector
        f_vec = self._assemble_load_vector(rhs_function)

        # Solve linear system K * u = f
        u_coeffs = np.linalg.solve(self._stiffness_matrix, f_vec)

        return u_coeffs

    def solution_to_function(self, coefficients: np.ndarray) -> Function:
        """
        Convert solution coefficients to a Function object.

        Args:
            coefficients: Solution coefficients in the basis

        Returns:
            Function representing u(x) = Σ coeffᵢ φᵢ(x)
        """
        if len(coefficients) != self.dofs:
            raise ValueError(
                f"Expected {self.dofs} coefficients, got {len(coefficients)}"
            )

        # Create solution function as linear combination of basis functions
        def solution_func(x):
            result = np.zeros_like(x)
            for i, coeff in enumerate(coefficients):
                if abs(coeff) > 1e-14:  # Skip negligible coefficients
                    phi_i = self.l2_space._basis_provider.get_basis_function(i)
                    result += coeff * phi_i(x)
            return result

        return Function(
            self.l2_space,
            evaluate_callable=solution_func,
            name="FEM solution (general)"
        )

    def get_basis_function(self, dof_index: int) -> Function:
        """Get the i-th basis function."""
        if dof_index >= self.dofs:
            raise ValueError(f"dof_index {dof_index} >= {self.dofs}")
        return self.l2_space._basis_provider.get_basis_function(dof_index)

    def get_basis_functions(self) -> list:
        """Get all basis functions."""
        return [self.l2_space._basis_provider.get_basis_function(i)
                for i in range(self.dofs)]

    def get_coordinates(self) -> np.ndarray:
        """Get evaluation points (not meaningful for general basis)."""
        # For general basis functions, we don't have a fixed mesh
        # Return evaluation points spanning the domain
        return np.linspace(
            self.function_domain.a,
            self.function_domain.b,
            self.n_integration_points
        )

    def interpolate_solution(self, coefficients: np.ndarray,
                             eval_points: np.ndarray) -> np.ndarray:
        """
        Evaluate the solution at arbitrary points.

        Args:
            coefficients: Solution coefficients
            eval_points: Points where to evaluate the solution

        Returns:
            Solution values at eval_points
        """
        solution_func = self.solution_to_function(coefficients)
        return solution_func(eval_points)

    def get_stiffness_matrix(self) -> np.ndarray:
        """Get the assembled stiffness matrix."""
        if not self._is_setup:
            raise RuntimeError("Must call setup() first")
        return self._stiffness_matrix.copy()

    def get_mass_matrix(self) -> np.ndarray:
        """Get the mass matrix (Gram matrix of basis functions)."""
        if not self._is_setup:
            raise RuntimeError("Must call setup() first")
        return self._mass_matrix.copy()


class FEMSolver:
    """FEM solver - supports Dirichlet BCs."""

    def __init__(self, function_domain: IntervalDomain,
                 dofs: int, boundary_conditions: BoundaryConditions):
        # Validate that function_domain is an IntervalDomain object
        if not isinstance(function_domain, IntervalDomain):
            raise TypeError(
                f"function_domain must be IntervalDomain object, "
                f"got {type(function_domain)}"
            )

        # Validate that boundary_conditions is a BoundaryConditions object
        if not isinstance(boundary_conditions, BoundaryConditions):
            raise TypeError(
                f"boundary_conditions must be BoundaryConditions object, "
                f"got {type(boundary_conditions)}"
            )

        # Validate that solver can handle this boundary condition type
        if boundary_conditions.type != 'dirichlet':
            raise ValueError(
                f"FEMSolver only supports 'dirichlet' boundary "
                f"conditions, got '{boundary_conditions.type}'"
            )

        self.function_domain = function_domain
        self.dofs = dofs
        self.boundary_conditions = boundary_conditions
        self._nodes = None
        self._dofs = dofs
        self._l2_space = None  # L2Space with hat basis functions

    @property
    def bc_type(self) -> str:
        """Get the boundary condition type as string for backward
        compatibility."""
        return self.boundary_conditions.type

    def setup(self):
        """Set up FEM mesh and structures using L2Space hat basis."""
        # Create uniform 1D mesh
        self._nodes = np.linspace(
            self.function_domain.a, self.function_domain.b, self._dofs + 2
        )

        self._h = self._nodes[1] - self._nodes[0]  # Element size

        # Create L2Space with hat basis functions for FEM
        from .l2_space import L2Space  # Import here to avoid circular imports
        # Use homogeneous hat basis for Dirichlet boundary conditions
        # Number of basis functions = number of interior nodes
        self._l2_space = L2Space(
            self._dofs,  # Interior nodes only (exclude boundaries)
            self.function_domain,  # IntervalDomain object
            basis_type='hat_homogeneous'
        )

    def _assemble_stiffness_matrix(self) -> np.ndarray:
        """Assemble stiffness matrix using L2Space basis functions."""
        if self._l2_space is None:
            raise RuntimeError("Must call setup() first")

        # Get interior basis functions (exclude boundary)
        n_interior = self._dofs  # Number of interior nodes
        K = np.zeros((n_interior, n_interior))

        # Assemble using Function operations
        for i in range(n_interior):
            for j in range(i, n_interior):  # Symmetric matrix
                # Compute ∫ φ'ᵢ φ'ⱼ dx using finite differences
                # For hat functions, this gives the standard FEM stiffness
                if i == j:
                    # Diagonal term
                    K[i, j] = 2.0 / self._h
                elif abs(i - j) == 1:
                    # Adjacent terms
                    K[i, j] = -1.0 / self._h
                    K[j, i] = -1.0 / self._h
                # All other terms are zero due to compact support

        return K

    def _assemble_mass_matrix(self) -> np.ndarray:
        """Assemble mass matrix using L2Space gram matrix."""
        if self._l2_space is None:
            raise RuntimeError("Must call setup() first")

        # The mass matrix is exactly the gram matrix of the L2Space
        return self._l2_space.gram_matrix

    def _assemble_load_vector(self, rhs_function: Function) -> np.ndarray:
        """
        Assemble the load vector for the FEM system.

        This method leverages Function multiplication and integration
        with automatic compact support handling.

        Args:
            rhs_function: Function to use as right-hand side

        Returns:
            Load vector for interior nodes only
        """
        if self._l2_space is None:
            raise RuntimeError("Must call setup() first")

        if not isinstance(rhs_function, Function):
            raise TypeError(
                f"rhs_function must be Function, got {type(rhs_function)}"
            )

        f_vec = np.zeros(self._dofs)

        # For each interior node, compute ∫ f(x) φᵢ(x) dx using Function
        for i in range(self._dofs):
            # Get basis function as Function
            phi_i = self._l2_space._basis_provider.get_basis_function(i)

            # Multiply f * φᵢ using Function multiplication
            # This automatically handles compact support intersection
            integrand = rhs_function * phi_i

            # Integrate using Function integration
            f_vec[i] = integrand.integrate(method='simpson')

        return f_vec

    def solve_poisson(self, rhs_function: Function) -> np.ndarray:
        """Solve Poisson equation with Dirichlet BCs using L2Space basis."""
        # Assemble system using L2Space operations
        K = self._assemble_stiffness_matrix()
        f = self._assemble_load_vector(rhs_function)

        # Solve interior system (boundary conditions already incorporated)
        u_interior = np.linalg.solve(K, f)

        # Reconstruct full solution including boundary values (u=0)
        u_full = np.zeros(len(self._nodes))
        u_full[1:-1] = u_interior  # Interior values
        # Boundary values remain 0 (Dirichlet BC)

        return u_full

    def get_basis_function(self, dof_index: int) -> Function:
        """
        Get basis function for given dof as a Function.

        Args:
            dof_index: Degree of freedom (0 to num_interior_dofs-1)

        Returns:
            Function (hat basis function with compact support)
        """
        if self._l2_space is None:
            raise RuntimeError("Must call setup() first")
        return self._l2_space._basis_provider.get_basis_function(dof_index)

    def get_basis_functions(self) -> list:
        """Get all interior basis functions as Functions."""
        if self._l2_space is None:
            raise RuntimeError("Must call setup() first")
        return [self._l2_space._basis_provider.get_basis_function(i)
                for i in range(self._dofs)]

    def get_coordinates(self) -> np.ndarray:
        """Get mesh node coordinates."""
        if self._nodes is None:
            raise RuntimeError("Must call setup() first")
        return self._nodes.copy()

    def interpolate_solution(self, solution_values: np.ndarray,
                             eval_points: np.ndarray) -> np.ndarray:
        """Linear interpolation of solution."""
        if self._nodes is None:
            raise RuntimeError("Must call setup() first")
        return np.interp(eval_points, self._nodes, solution_values)
