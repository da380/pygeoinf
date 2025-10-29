"""
FEM Solver for Laplacian Inverse Operator

This module provides a Python FEM solver for the LaplacianInverseOperator.
GeneralFEMSolver works with any basis functions from Lebesgue and automatically
optimizes for hat functions when detected.

No external dependencies required.
"""

import numpy as np
import logging

from .boundary_conditions import BoundaryConditions
from .functions import Function
from .lebesgue_space import Lebesgue
from .interval_domain import IntervalDomain
from .function_providers import HatFunctionProvider


class GeneralFEMSolver:
    """
    General FEM solver that works with any basis functions from Lebesgue.

    This implementation follows the general finite element method described
    in the mathematical formulation, where basis functions {φᵢ} can be any
    suitable functions that span the finite-dimensional subspace Vₕ ⊂ H₀¹(a,b).

    The solver extracts basis functions from the Lebesgue basis provider and
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

    def __init__(self, function_domain: IntervalDomain,
                 dofs: int, operator_domain: Lebesgue,
                 boundary_conditions: BoundaryConditions):
        """
        Initialize general FEM solver with Lebesgue basis functions.

        Args:
            fem_space: Lebesgue object providing the basis functions
            boundary_conditions: BoundaryConditions object (must be Dirichlet)
        """
        self._operator_domain = operator_domain
        self._function_domain = function_domain
        self._dofs = dofs
        self._boundary_conditions = boundary_conditions

        # Configure hat provider according to boundary conditions.
        # Homogeneous Dirichlet: ghost nodes, basis vanishes at boundaries
        # Mixed/Neumann/Robin/Periodic: non-homogeneous hat functions
        if (self._boundary_conditions.type == 'dirichlet' and
                self._boundary_conditions.is_homogeneous):
            homogeneous_flag = True
            n_nodes = self._dofs + 2
        else:
            # All other BCs use non-homogeneous hat functions
            homogeneous_flag = False
            n_nodes = self._dofs

        self._provider = HatFunctionProvider(
            self._operator_domain, homogeneous=homogeneous_flag,
            n_nodes=n_nodes)

        # Integration parameters for numerical assembly
        self.integration_method = 'simpson'
        self.n_integration_points = 1000

        # Storage for assembled matrices
        self._stiffness_matrix = self._assemble_stiffness_matrix()
        self._is_setup = True

        # logger
        self._log = logging.getLogger(__name__)
        self._log.info(
            "GeneralFEMSolver initialized: dofs=%s, domain=[%s,%s]",
            self._dofs,
            self._function_domain.a,
            self._function_domain.b,
        )

    def _assemble_stiffness_matrix(self) -> np.ndarray:
        """
        Assemble stiffness matrix analytically for hat functions.

        For homogeneous hat functions on uniform grid with Dirichlet BCs,
        the stiffness matrix has the analytical form:
        - Diagonal: 2/h
        - Off-diagonal (adjacent): -1/h
        - All other entries: 0
        """
        if (self._boundary_conditions.type == 'dirichlet' and
                self._boundary_conditions.is_homogeneous):
            n = self._dofs
            h = (self._function_domain.b - self._function_domain.a) / (n + 1)

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
        elif (self._boundary_conditions.type == 'neumann' and
              self._boundary_conditions.is_homogeneous):
            n = self._dofs
            h = (self._function_domain.b - self._function_domain.a) / (n - 1)

            # Create tridiagonal matrix
            K0 = np.zeros((n, n))

            for i in range(1, n - 1):
                # Diagonal terms
                K0[i, i] = 2.0 / h

                # Off-diagonal terms (adjacent only)
                K0[i, i-1] = -1.0 / h
                K0[i, i+1] = -1.0 / h
            K0[0, 0] = 1.0 / h
            K0[0, 1] = -1.0 / h
            K0[-1, -1] = 1.0 / h
            K0[-1, -2] = -1.0 / h
            w = np.ones(n) * h
            w[0] = h / 2
            w[-1] = h / 2

            K = np.zeros((n + 1, n + 1))
            K[:n, :n] = K0
            K[:n, -1] = w
            K[-1, :n] = w
        elif self._boundary_conditions.type == 'periodic':
            n = self._dofs
            h = (self._function_domain.b - self._function_domain.a) / n

            # Circulant stiffness (wrap-around indices)
            K0 = np.zeros((n, n))
            for i in range(n):
                K0[i, i] = 2.0 / h
                K0[i, (i - 1) % n] = -1.0 / h
                K0[i, (i + 1) % n] = -1.0 / h
            # Enforce mean-zero (remove constant nullspace) via Lagrange
            # multiplier
            w = np.ones(n) * h  # uniform periodic trapezoid weights

            K = np.zeros((n + 1, n + 1))
            K[:n, :n] = K0
            K[:n, -1] = w
            K[-1, :n] = w
        elif (self._boundary_conditions.type == 'mixed_dirichlet_neumann' and
              self._boundary_conditions.is_homogeneous):
            # Mixed BC: Dirichlet at left (u(a) = 0),
            # Neumann at right (u'(b) = 0)
            # Nodes: 0 at a (fixed at 0), 1...n-1 are free
            # We solve for n-1 unknowns (nodes 1 to n-1)
            # since node 0 is fixed by Dirichlet BC
            n = self._dofs
            h = (self._function_domain.b - self._function_domain.a) / (n - 1)

            # Build (n-1) x (n-1) matrix for free nodes
            K = np.zeros((n - 1, n - 1))

            # First free node (index 0 in K, node 1 in global)
            # Stiffness from element [0,1] (node 0 is fixed at 0)
            # and element [1,2]
            K[0, 0] = 2.0 / h
            if n > 2:
                K[0, 1] = -1.0 / h

            # Interior free nodes
            for i in range(1, n - 2):
                K[i, i] = 2.0 / h
                K[i, i-1] = -1.0 / h
                K[i, i+1] = -1.0 / h

            # Last free node (Neumann boundary at right)
            if n > 2:
                K[n-2, n-2] = 1.0 / h
                K[n-2, n-3] = -1.0 / h
        elif (self._boundary_conditions.type == 'mixed_neumann_dirichlet' and
              self._boundary_conditions.is_homogeneous):
            # Mixed BC: Neumann at left (u'(a) = 0),
            # Dirichlet at right (u(b) = 0)
            # Nodes: 0...n-2 are free, n-1 at b (fixed at 0)
            # We solve for n-1 unknowns (nodes 0 to n-2)
            n = self._dofs
            h = (self._function_domain.b - self._function_domain.a) / (n - 1)

            # Build (n-1) x (n-1) matrix for free nodes
            K = np.zeros((n - 1, n - 1))

            # First free node (Neumann boundary at left)
            K[0, 0] = 1.0 / h
            if n > 2:
                K[0, 1] = -1.0 / h

            # Interior free nodes
            for i in range(1, n - 2):
                K[i, i] = 2.0 / h
                K[i, i-1] = -1.0 / h
                K[i, i+1] = -1.0 / h

            # Last free node (index n-2 in K, node n-2 in global)
            # Stiffness from element [n-3,n-2] and element [n-2,n-1]
            # (node n-1 is fixed at 0)
            if n > 2:
                K[n-2, n-2] = 2.0 / h
                K[n-2, n-3] = -1.0 / h
        elif (self._boundary_conditions.type == 'robin' and
              self._boundary_conditions.is_homogeneous):
            # Robin BC: α*u + β*u' = 0 at boundaries
            n = self._dofs
            h = (self._function_domain.b - self._function_domain.a) / (n - 1)

            # Get Robin coefficients
            alpha_left = self._boundary_conditions.get_parameter(
                'left_alpha', 1.0
            )
            beta_left = self._boundary_conditions.get_parameter(
                'left_beta', 1.0
            )
            alpha_right = self._boundary_conditions.get_parameter(
                'right_alpha', 1.0
            )
            beta_right = self._boundary_conditions.get_parameter(
                'right_beta', 1.0
            )

            # Ensure values are not None
            if alpha_left is None:
                alpha_left = 1.0
            if beta_left is None:
                beta_left = 1.0
            if alpha_right is None:
                alpha_right = 1.0
            if beta_right is None:
                beta_right = 1.0

            # Build tridiagonal matrix
            K = np.zeros((n, n))

            # Left boundary node (i=0)
            # Robin: alpha_left*u(a) + beta_left*u'(a) = 0
            # Discretize u'(a) ≈ (u_1 - u_0)/h
            # Stiffness contribution from weak form
            if abs(float(beta_left)) > 1e-14:
                K[0, 0] = 1.0 / h + float(alpha_left) / float(beta_left)
            else:
                K[0, 0] = 1.0 / h
            K[0, 1] = -1.0 / h

            # Interior nodes (1 to n-2)
            for i in range(1, n - 1):
                K[i, i] = 2.0 / h
                K[i, i-1] = -1.0 / h
                K[i, i+1] = -1.0 / h

            # Right boundary node (i=n-1)
            # Robin: alpha_right*u(b) + beta_right*u'(b) = 0
            # Discretize u'(b) ≈ (u_{n-1} - u_{n-2})/h
            if abs(float(beta_right)) > 1e-14:
                K[n-1, n-1] = 1.0 / h + float(alpha_right) / float(beta_right)
            else:
                K[n-1, n-1] = 1.0 / h
            K[n-1, n-2] = -1.0 / h
        else:
            raise NotImplementedError(
                f"Boundary condition type '{self._boundary_conditions.type}' "
                f"with homogeneous={self._boundary_conditions.is_homogeneous} "
                f"is not yet implemented in FEM solver"
            )

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

        f_vec = np.zeros(self._dofs)
        for i in range(self._dofs):
            phi_i = self._provider.get_function_by_index(i)
            f_phi_i = rhs_function * phi_i
            f_vec[i] = f_phi_i.integrate(
                method=self.integration_method,
                n_points=self.n_integration_points
            )

        if (self._boundary_conditions.type == 'dirichlet' and
                self._boundary_conditions.is_homogeneous):
            return f_vec
        elif (self._boundary_conditions.type == 'neumann' and
              self._boundary_conditions.is_homogeneous):
            return np.append(f_vec, 0)
        elif self._boundary_conditions.type == 'periodic':
            return np.append(f_vec, 0)
        elif (self._boundary_conditions.type == 'mixed_dirichlet_neumann' and
              self._boundary_conditions.is_homogeneous):
            # Return load vector for free nodes only (exclude first node)
            return f_vec[1:]
        elif (self._boundary_conditions.type == 'mixed_neumann_dirichlet' and
              self._boundary_conditions.is_homogeneous):
            # Return load vector for free nodes only (exclude last node)
            return f_vec[:-1]
        elif (self._boundary_conditions.type == 'robin' and
              self._boundary_conditions.is_homogeneous):
            return f_vec
        else:
            raise NotImplementedError(
                f"Load vector assembly for BC type "
                f"'{self._boundary_conditions.type}' "
                f"with homogeneous={self._boundary_conditions.is_homogeneous} "
                f"is not yet implemented"
            )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def get_coordinates(self) -> np.ndarray:
        """Return the coordinates of active basis nodes used by the
        FEM provider.

        Falls back to uniform interior nodes if provider does not expose
        active node coordinates.
        """
        try:
            return self._provider.get_active_nodes()
        except Exception:
            # Fallback: return uniform interior nodes
            a, b = self._function_domain.a, self._function_domain.b
            return np.linspace(a, b, self._dofs)

    @property
    def stiffness_matrix(self) -> np.ndarray:
        """Expose assembled stiffness matrix."""
        return self._stiffness_matrix

    def solve_poisson(self, rhs_function: Function) -> np.ndarray:
        """
        Solve the Poisson equation: -Δu = f with Dirichlet boundary conditions.

        Args:
            rhs_function: Function representing the right-hand side f

        Returns:
            Coefficient vector for the solution in the basis functions
        """

        # Assemble load vector
        f_vec = self._assemble_load_vector(rhs_function)

        # Solve linear system K * u = f
        solution_coeffs = np.linalg.solve(self._stiffness_matrix, f_vec)

        if (self._boundary_conditions.type == 'dirichlet' and
                self._boundary_conditions.is_homogeneous):
            return solution_coeffs
        elif (self._boundary_conditions.type == 'neumann' and
              self._boundary_conditions.is_homogeneous):
            return solution_coeffs[0:-1]
        elif self._boundary_conditions.type == 'periodic':
            return solution_coeffs[0:-1]
        elif (self._boundary_conditions.type == 'mixed_dirichlet_neumann' and
              self._boundary_conditions.is_homogeneous):
            # Prepend zero for the fixed Dirichlet node at left
            return np.concatenate([[0.0], solution_coeffs])
        elif (self._boundary_conditions.type == 'mixed_neumann_dirichlet' and
              self._boundary_conditions.is_homogeneous):
            # Append zero for the fixed Dirichlet node at right
            return np.concatenate([solution_coeffs, [0.0]])
        elif (self._boundary_conditions.type == 'robin' and
              self._boundary_conditions.is_homogeneous):
            return solution_coeffs
        else:
            raise NotImplementedError(
                f"Solution extraction for BC type "
                f"'{self._boundary_conditions.type}' "
                f"with homogeneous={self._boundary_conditions.is_homogeneous} "
                f"is not yet implemented"
            )

    def solution_to_function(self, coefficients: np.ndarray) -> Function:
        """
        Convert solution coefficients to a Function.

        Args:
            coefficients: Solution coefficients in basis function expansion

        Returns:
            Function representing the solution
        """
        # Create solution function as linear combination of basis functions
        def solution_func(eval_points):
            # Handle scalar input
            scalar_input = np.isscalar(eval_points)
            if scalar_input:
                eval_points = np.array([eval_points])

            result = np.zeros_like(eval_points, dtype=float)

            for i, coeff in enumerate(coefficients):
                if abs(coeff) > 1e-14:  # Skip negligible coefficients
                    phi_i = self._provider.get_function_by_index(i)
                    result += coeff * phi_i(eval_points)

            return result[0] if scalar_input else result

        return Function(
            self._operator_domain,
            evaluate_callable=solution_func,
            name="FEM solution"
        )
