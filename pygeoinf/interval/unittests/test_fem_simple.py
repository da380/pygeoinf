"""Simple test to verify FEM solver works for new boundary conditions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
from pygeoinf.interval import (
    Lebesgue, IntervalDomain, BoundaryConditions,
    InverseLaplacian, Function
)

# Create domain
domain = IntervalDomain(0.0, 1.0)
space = Lebesgue(128, domain, basis='sine')

print("Testing FEM implementations for new boundary conditions")
print("="*70)

# Test 1: Mixed Dirichlet-Neumann
print("\n1. Mixed Dirichlet-Neumann (u(0)=0, u'(1)=0)")
bc_dn = BoundaryConditions.mixed_dirichlet_neumann(0.0, 0.0)
inv_lap_dn = InverseLaplacian(space, bc_dn, method='fem', dofs=128)

# Simple test: constant RHS
f_const = Function(space, evaluate_callable=lambda x: np.ones_like(x))
u_dn = inv_lap_dn(f_const)
print(f"   u(0) = {u_dn(0.0):.6e} (should be ~0)")
print(f"   u(0.5) = {u_dn(0.5):.6e}")
print(f"   u(1) = {u_dn(1.0):.6e}")

# Test 2: Mixed Neumann-Dirichlet
print("\n2. Mixed Neumann-Dirichlet (u'(0)=0, u(1)=0)")
bc_nd = BoundaryConditions.mixed_neumann_dirichlet(0.0, 0.0)
inv_lap_nd = InverseLaplacian(space, bc_nd, method='fem', dofs=128)

u_nd = inv_lap_nd(f_const)
print(f"   u(0) = {u_nd(0.0):.6e}")
print(f"   u(0.5) = {u_nd(0.5):.6e}")
print(f"   u(1) = {u_nd(1.0):.6e} (should be ~0)")

# Test 3: Robin
print("\n3. Robin (αu + βu' = 0 at boundaries, α=β=1)")
bc_robin = BoundaryConditions('robin',
                              left_alpha=1.0, left_beta=1.0, left_value=0.0,
                              right_alpha=1.0, right_beta=1.0, right_value=0.0)
inv_lap_robin = InverseLaplacian(space, bc_robin, method='fem', dofs=128)

u_robin = inv_lap_robin(f_const)
print(f"   u(0) = {u_robin(0.0):.6e}")
print(f"   u(0.5) = {u_robin(0.5):.6e}")
print(f"   u(1) = {u_robin(1.0):.6e}")

print("\n" + "="*70)
print("All FEM solvers created and ran successfully!")
print("(Numerical accuracy needs further tuning, but no errors occurred)")
