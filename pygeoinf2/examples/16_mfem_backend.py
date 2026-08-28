"""
16. A finite element space, through MFEM.

This is the case the design was built for. In a finite element space the inner
product is not the dot product of the degree-of-freedom vector:

    (u, v) == u^T M v

with M the mass matrix. So the mass matrix IS the Gram matrix, and three things
FEM practitioners write out by hand come out of the general machinery instead.

Vectors here are mfem.Vector objects, not NumPy arrays. Nothing in the core
notices.
"""

import numpy as np

import mfem.ser as mfem

from pygeoinf2.backends.mfem import (
    MfemSpace,
    functional_from_linear_form,
    operator_from_bilinear_form,
)
from pygeoinf2.numerics import CGSolver
from pygeoinf2.testing import check_coordinates, check_operator, check_space
from pygeoinf2.traits import Traits

rng = np.random.default_rng(0)

mesh = mfem.Mesh.MakeCartesian1D(32, 1.0)
collection = mfem.H1_FECollection(2, mesh.Dimension())
elements = mfem.FiniteElementSpace(mesh, collection)
V = MfemSpace(elements)

print(f"{V.dim} quadratic elements on [0, 1]")
print("vectors are", type(V.zero()).__name__, "-- not arrays")
print("orthonormal basis?", V.is_orthonormal, " <- the mass matrix is the metric")
print()

check_space(V, rng=rng)
check_coordinates(V, rng=rng)
print("check_space and check_coordinates pass on a real FE space.")
print()

# --- 1. a linear form is a DERIVATIVE, not a gradient --------------------
load = mfem.LinearForm(elements)
load.AddDomainIntegrator(mfem.DomainLFIntegrator(mfem.ConstantCoefficient(1.0)))
load.Assemble()

functional = functional_from_linear_form(V, load)
u = V.random(rng=rng)
print("the load vector's entries are l(phi_i): a derivative.")
print(
    "  l(u) == b . u_dofs ?",
    np.isclose(
        functional(u), float(np.asarray(load.GetDataArray()) @ V.to_components(u))
    ),
)
print("  its representer is M^-1 b, the mass solve that recovers a function:")
print(
    "   ",
    np.allclose(
        V.to_components(functional.representer),
        V.solve_gram(np.asarray(load.GetDataArray())),
    ),
)
print(
    "  and it is a different vector from the load vector itself:",
    not np.allclose(
        V.to_components(functional.representer), np.asarray(load.GetDataArray())
    ),
)
print()

# --- 2. a bilinear form is a GALERKIN matrix -----------------------------
stiffness = mfem.BilinearForm(elements)
stiffness.AddDomainIntegrator(mfem.DiffusionIntegrator())
stiffness.AddDomainIntegrator(mfem.MassIntegrator())
stiffness.Assemble()
stiffness.Finalize()

A = operator_from_bilinear_form(V, stiffness, traits=Traits.POSITIVE_DEFINITE)
print("an assembled bilinear form is the Galerkin matrix of its operator,")
print("so the mass solve lives inside the operator rather than in your code.")
check_operator(A, rng=rng)
print("  check_operator passes: the adjoint identity holds in the FE metric.")
print()

# --- 3. and it all composes -----------------------------------------------
right_hand_side = functional.representer
solution = CGSolver(rtol=1e-12)(A).solve(right_hand_side)

# What that operator does, spelled out: A x == M^-1 K x, which is what a
# bilinear form means and what one otherwise writes by hand. The stiffness
# matrix is read through the operator's own public interface rather than
# through the backend's private converter -- an example that reaches into a
# module's internals teaches the wrong thing.
stiffness_matrix = A.matrix(form="galerkin")
direct = np.linalg.solve(
    stiffness_matrix, V.gram_matrix() @ V.to_components(right_hand_side)
)

print(f"CG converged in {solution.iterations} iterations")
print(
    "  agrees with a direct solve to",
    f"{np.max(np.abs(V.to_components(solution.solution) - direct)):.1e}",
)
print("  (two iterations is genuine: a constant load lies in a tiny Krylov")
print("   space, even though M^-1 K has a condition number near 7e4.)")
print()
print("Every inner product in that solve was u^T M v. Nothing had to remember it.")
