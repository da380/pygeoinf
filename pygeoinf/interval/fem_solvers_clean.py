"""
FEM Solvers for Laplacian Inverse Operator

This module provides Python FEM solvers for the LaplacianInverseOperator.
GeneralFEMSolver works with any basis functions from L2Space and automatically
optimizes for hat functions when detected.

No external dependencies required.
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
    assembles the stiffness matrix and load vector. For the stiffness matrix:

    [K]ᵢⱼ = ∫ φ'ᵢ(x) φ'ⱼ(x) dx
    [F]ᵢ = ∫ f(x) φᵢ(x) dx

    **Analytical Optimization**: For hat functions with homogeneous Dirichlet
    boundary conditions, the solver automatically detects this configuration
    and uses an analytical formula for the stiffness matrix, which is much
    faster than numerical integration.

    **Numerical Method**: For arbitrary basis functions, the solver uses the
    GradientOperator for computing derivatives and Function.integrate() for
    integration, providing a robust and mathematically consistent approach.
    """

    def __init__(self, l2_space, boundary_conditions: BoundaryConditions):
        """
        Initialize general FEM solver with L2Space basis functions.

        Args:
            l2_space: L2Space object providing the basis functions
            boundary_conditions: BoundaryConditions object (must be Dirichlet)
        """
        # Validate boundary conditions
        if not isinstance(boundary_conditions, BoundaryConditions):
            raise TypeError(
                f"boundary_conditions must be BoundaryConditions object, "
                f"got {type(boundary_conditions)}"
            )

        if boundary_conditions.type != 'dirichlet':
            raise ValueError(
                f"GeneralFEMSolver only supports 'dirichlet' boundary "
                f"conditions, got '{boundary_conditions.type}'"
            )

        # Check that the L2Space basis is compatible with Dirichlet BCs
        # For Dirichlet BCs, basis functions should vanish at boundaries
        if hasattr(l2_space, '_basis_type'):
            basis_type = l2_space._basis_type
            if basis_type not in ['hat_homogeneous', 'sine']:
                print(f"Warning: L2Space basis type '{basis_type}' may not "
                      f"satisfy homogeneous Dirichlet boundary conditions. "
                      f"Consider using 'hat_homogeneous' or 'sine'.")
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

        # Check if analytical computation is possible
        self._can_use_analytical = self._check_analytical_computation()

        if self._can_use_analytical:
            print("Using analytical stiffness matrix computation for "
                  "hat functions")
            self._stiffness_matrix = (
                self._assemble_stiffness_matrix_analytical()
            )
        else:
            print("Using numerical stiffness matrix computation")
            self._stiffness_matrix = (
                self._assemble_stiffness_matrix_numerical()
            )

        self._is_setup = True

    def _check_analytical_computation(self) -> bool:
        """
        Check if analytical stiffness matrix computation is possible.

        Returns True if:
        1. Basis is hat functions with homogeneous Dirichlet BCs
        2. Boundary conditions are homogeneous Dirichlet
        """
        # Check boundary conditions are homogeneous Dirichlet
        if (self.boundary_conditions.type != 'dirichlet' or
            self.boundary_conditions.left != 0.0 or
            self.boundary_conditions.right != 0.0):
            return False

        # Check if the basis provider is specifically for hat functions
        # by examining its type name
        provider_name = type(self.l2_space._basis_provider).__name__
        if 'Hat' in provider_name and 'Homogeneous' in provider_name:
            return True

        return False

    def _assemble_stiffness_matrix_analytical(self) -> np.ndarray:
        """
        Assemble stiffness matrix analytically for hat functions.

        For homogeneous hat functions on uniform grid with Dirichlet BCs,
        the stiffness matrix has the analytical form:
        - Diagonal: 2/h
        - Off-diagonal (adjacent): -1/h
        - All other entries: 0
        """
        n = self.dofs
        h = (self.function_domain.b - self.function_domain.a) / (n + 1)

        # Create tridiagonal matrix
        K = np.zeros((n, n))

        for i in range(n):
            # Diagonal terms
            K[i, i] = 2.0 / h

            # Off-diagonal terms (adjacent only)
            if i > 0:
                K[i, i-1] = -1.0 / h
            if i < n - 1:
                K[i, i+1] = -1.0 / h

        return K

    def _assemble_stiffness_matrix_numerical(self) -> np.ndarray:
        """
        Assemble stiffness matrix numerically using basis function derivatives.

        For general basis functions, computes:
        [K]ᵢⱼ = ∫ φ'ᵢ(x) φ'ⱼ(x) dx

        Uses GradientOperator for derivatives and Function.integrate()
        for integration.
        """
        from .operators import GradientOperator  # Import here to avoid circular imports

        n = self.dofs
        K = np.zeros((n, n))

        # Create gradient operator for computing derivatives
        gradient_op = GradientOperator(
            self.l2_space,
            method='finite_difference'
        )

        # Compute derivatives of all basis functions
        basis_derivatives = []
        for i in range(n):
            phi_i = self.l2_space._basis_provider.get_basis_function(i)
            phi_i_prime = gradient_op._apply(phi_i)
            basis_derivatives.append(phi_i_prime)

        # Assemble stiffness matrix using Function integration
        for i in range(n):
            for j in range(i, n):  # Exploit symmetry
                # Compute ∫ φ'ᵢ * φ'ⱼ dx using Function multiplication
                # and integration
                integrand = basis_derivatives[i] * basis_derivatives[j]
                integral = integrand.integrate(method=self.integration_method)

                K[i, j] = integral
                if i != j:
                    K[j, i] = integral  # Symmetric matrix

        return K

    def _assemble_load_vector(self, rhs_function: Function) -> np.ndarray:
        """
        Assemble the load vector for the FEM system.

        Computes: [F]ᵢ = ∫ f(x) φᵢ(x) dx

        Uses Function multiplication and integration for robust computation.

        Args:
            rhs_function: Function representing the right-hand side

        Returns:
            Load vector for the FEM system
        """
        if not isinstance(rhs_function, Function):
            raise TypeError(
                f"rhs_function must be Function, got {type(rhs_function)}"
            )

        f_vec = np.zeros(self.dofs)

        for i in range(self.dofs):
            # Get basis function
            phi_i = self.l2_space._basis_provider.get_basis_function(i)

            # Compute ∫ f(x) φᵢ(x) dx using Function operations
            integrand = rhs_function * phi_i
            f_vec[i] = integrand.integrate(method=self.integration_method)

        return f_vec

    def solve_poisson(self, rhs_function: Function) -> np.ndarray:
        """
        Solve the Poisson equation: -Δu = f with Dirichlet boundary conditions.

        Args:
            rhs_function: Function representing the right-hand side f

        Returns:
            Coefficient vector for the solution in the basis functions
        """
        if not self._is_setup:
            raise RuntimeError("Must call setup() first")

        # Assemble load vector
        f_vec = self._assemble_load_vector(rhs_function)

        # Solve linear system K * u = f
        solution_coeffs = np.linalg.solve(self._stiffness_matrix, f_vec)

        return solution_coeffs

    def solution_to_function(self, coefficients: np.ndarray) -> Function:
        """
        Convert solution coefficients to a Function.

        Args:
            coefficients: Solution coefficients in basis function expansion

        Returns:
            Function representing the solution
        """
        if len(coefficients) != self.dofs:
            raise ValueError(
                f"coefficients must have length {self.dofs}, "
                f"got {len(coefficients)}"
            )

        # Create solution function as linear combination of basis functions
        def solution_func(eval_points):
            # Handle scalar input
            scalar_input = np.isscalar(eval_points)
            if scalar_input:
                eval_points = np.array([eval_points])

            result = np.zeros_like(eval_points, dtype=float)

            for i, coeff in enumerate(coefficients):
                if abs(coeff) > 1e-14:  # Skip negligible coefficients
                    phi_i = self.l2_space._basis_provider.get_basis_function(i)
                    result += coeff * phi_i(eval_points)

            return result[0] if scalar_input else result

        return Function(
            self.l2_space,
            evaluate_callable=solution_func,
            name="FEM solution"
        )

    def get_basis_function(self, dof_index: int) -> Function:
        """Get a basis function as a Function object."""
        if dof_index < 0 or dof_index >= self.dofs:
            raise IndexError(f"dof_index must be in [0, {self.dofs})")
        return self.l2_space._basis_provider.get_basis_function(dof_index)

    def get_basis_functions(self) -> list:
        """Get all basis functions as Functions."""
        return [self.get_basis_function(i) for i in range(self.dofs)]

    def get_coordinates(self) -> np.ndarray:
        """Get mesh node coordinates (for hat function basis)."""
        if hasattr(self.l2_space._basis_provider, '_domain'):
            # For hat functions, return node coordinates
            domain = self.l2_space._basis_provider._domain
            return np.linspace(domain.a, domain.b, self.dofs + 2)
        else:
            # For general basis, return evaluation grid
            return np.linspace(
                self.function_domain.a,
                self.function_domain.b,
                self.dofs + 2
            )

    def interpolate_solution(self, solution_coeffs: np.ndarray,
                           eval_points: np.ndarray) -> np.ndarray:
        """
        Evaluate solution at given points using basis function expansion.

        Args:
            solution_coeffs: Solution coefficients
            eval_points: Points where to evaluate the solution

        Returns:
            Solution values at eval_points
        """
        solution_func = self.solution_to_function(solution_coeffs)
        return solution_func(eval_points)

    def get_stiffness_matrix(self) -> np.ndarray:
        """Get the assembled stiffness matrix."""
        if not self._is_setup:
            raise RuntimeError("Must call setup() first")
        return self._stiffness_matrix.copy()
