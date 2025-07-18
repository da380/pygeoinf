"""
Inverse Elliptic Operators for Pygeoinf

This module provides inverse elliptic operators that extend pygeoinf's LinearOperator
class and integrate with DOLFINx for finite element computations.

Classes:
    InverseEllipticOperator: Main inverse elliptic operator class

Author: Adrian-Mag
Date: July 2025
"""

import numpy as np
from typing import Optional, Union, Dict, Any

# Pygeoinf imports
from ..linalg import LinearOperator
from .sobolev_functions import SobolevFunction

# DOLFINx imports (optional)
try:
    import ufl
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    from petsc4py.PETSc import ScalarType
    DOLFINX_AVAILABLE = True
except ImportError:
    DOLFINX_AVAILABLE = False

# Bridge import (optional)
try:
    from .dolfinx_bridge import DOLFINxSobolevBridge
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False


class InverseEllipticOperator(LinearOperator):
    """
    Inverse elliptic operator extending pygeoinf's LinearOperator.
    Maps from H^s to H^{s+2} using DOLFINx for PDE solving.

    This operator solves elliptic PDEs of the form:
        L[u] = f
    where L is an elliptic differential operator (e.g., -Δ + αI).

    The inverse operator L^{-1} maps from the space of right-hand sides
    to the space of solutions, providing covariance operators for
    Gaussian measures in Bayesian inference.
    """

    def __init__(self, domain_sobolev_space, codomain_sobolev_space,
                 dolfinx_domain, dolfinx_function_space,
                 pde_type='laplacian', **pde_params):
        """
        Args:
            domain_sobolev_space: pygeoinf Sobolev space (RHS of PDE)
            codomain_sobolev_space: pygeoinf Sobolev space (solution)
            dolfinx_domain: DOLFINx mesh
            dolfinx_function_space: DOLFINx function space
            pde_type: Type of PDE ('laplacian', 'reaction_diffusion')
            **pde_params: Additional PDE parameters (e.g., alpha for reaction)

        Raises:
            ImportError: If DOLFINx is not available
        """

        if not DOLFINX_AVAILABLE:
            raise ImportError("DOLFINx is required for InverseEllipticOperator. "
                            "Install DOLFINx using conda-forge or build from source.")

        if not BRIDGE_AVAILABLE:
            raise ImportError("DOLFINxSobolevBridge is required but not available.")

        # Initialize parent LinearOperator
        super().__init__(domain_sobolev_space, codomain_sobolev_space, self._mapping)

        # Store DOLFINx components
        self.dolfinx_domain = dolfinx_domain
        self.dolfinx_function_space = dolfinx_function_space
        self.pde_type = pde_type
        self.pde_params = pde_params

        # Create bridges for domain and codomain
        self.domain_bridge = DOLFINxSobolevBridge(
            domain_sobolev_space, dolfinx_domain, dolfinx_function_space
        )
        self.codomain_bridge = DOLFINxSobolevBridge(
            codomain_sobolev_space, dolfinx_domain, dolfinx_function_space
        )

        # Setup PDE variational forms
        self._setup_pde_forms()
        self._setup_boundary_conditions()

    def _setup_pde_forms(self):
        """Setup the variational forms for the PDE."""
        u = ufl.TrialFunction(self.dolfinx_function_space)
        v = ufl.TestFunction(self.dolfinx_function_space)

        if self.pde_type == 'laplacian':
            # -Δu = f
            self.bilinear_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        elif self.pde_type == 'reaction_diffusion':
            # -Δu + αu = f
            alpha = self.pde_params.get('alpha', 1.0)
            self.bilinear_form = (
                ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx +
                fem.Constant(self.dolfinx_domain, alpha) * u * v * ufl.dx
            )
        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")

    def _setup_boundary_conditions(self):
        """Setup boundary conditions based on domain Sobolev space."""
        # Get boundary condition type from domain space
        bc_type = 'dirichlet'  # Default
        if hasattr(self.domain, 'boundary_conditions') and self.domain.boundary_conditions:
            bc_type = self.domain.boundary_conditions.get('type', 'dirichlet')

        if bc_type == 'dirichlet':
            # Homogeneous Dirichlet: u = 0 at boundaries
            facets = mesh.locate_entities_boundary(
                self.dolfinx_domain, 0,
                lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
            )
            dofs = fem.locate_dofs_topological(
                V=self.dolfinx_function_space, entity_dim=0, entities=facets
            )
            self.boundary_conditions = [
                fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=self.dolfinx_function_space)
            ]
        else:
            # No boundary conditions for other types (natural BCs)
            self.boundary_conditions = []

    def _mapping(self, rhs_sobolev_function):
        """
        The core mapping: solve the PDE to get the solution.
        """
        # Convert RHS to DOLFINx
        if isinstance(rhs_sobolev_function, np.ndarray):
            # Input is coefficient vector
            rhs_dolfinx = self.domain_bridge.coefficients_to_dolfinx(rhs_sobolev_function)
        else:
            # Input is SobolevFunction
            rhs_dolfinx = self.domain_bridge.sobolev_to_dolfinx(rhs_sobolev_function)

        # Create linear form
        v = ufl.TestFunction(self.dolfinx_function_space)
        linear_form = rhs_dolfinx * v * ufl.dx

        # Solve the PDE
        problem = LinearProblem(
            self.bilinear_form, linear_form,
            bcs=self.boundary_conditions,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        solution_dolfinx = problem.solve()

        # Convert solution back to Sobolev space
        return self.codomain_bridge.dolfinx_to_sobolev(solution_dolfinx)

    def apply_to_coefficients(self, rhs_coefficients):
        """Apply operator to coefficient vector, return coefficient vector."""
        # Convert to DOLFINx
        rhs_dolfinx = self.domain_bridge.coefficients_to_dolfinx(rhs_coefficients)

        # Solve PDE
        v = ufl.TestFunction(self.dolfinx_function_space)
        linear_form = rhs_dolfinx * v * ufl.dx

        problem = LinearProblem(
            self.bilinear_form, linear_form,
            bcs=self.boundary_conditions,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        solution_dolfinx = problem.solve()

        # Convert back to coefficients
        return self.codomain_bridge.dolfinx_to_coefficients(solution_dolfinx)