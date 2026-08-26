"""
17. PETSc vectors, and the adjoint that is not a transpose.

PETSc vectors are opaque, possibly distributed objects. Nothing in the core
notices, because arithmetic goes through the space.

The substance of this example is the second half. On a plain R^n a matrix's
adjoint is its transpose, so ``multTranspose`` is right and a derivative and a
gradient coincide. Put a mass matrix on the space -- the finite element
situation -- and the adjoint becomes M^-1 A^T M. Reaching for multTranspose
there is the mistake of example 5, in the setting where it is most tempting,
because PETSc offers the transpose and does not offer the adjoint.

Needs petsc4py, which has no binary wheel and builds PETSc from source, so it
is skipped unless that has been installed:

    poetry run pip install petsc petsc4py
"""

import numpy as np

from petsc4py import PETSc

from pygeoinf2.backends.petsc import (
    PetscSpace,
    PetscWeightedSpace,
    operator_from_matrix,
)
from pygeoinf2.numerics import CGSolver
from pygeoinf2.testing import check_coordinates, check_operator, check_space
from pygeoinf2.traits import Traits

rng = np.random.default_rng(0)
n = 12


def petsc_matrix(array):
    """A dense PETSc.Mat from a NumPy array."""
    matrix = PETSc.Mat().createDense([n, n], array=np.ascontiguousarray(array))
    matrix.assemble()
    return matrix


# --- an unweighted space: vectors are opaque, and that is all -------------
X = PetscSpace(n)
print("vectors are", type(X.zero()).__name__, "-- not arrays")
check_space(X, rng=rng)
check_coordinates(X, rng=rng)
print("check_space and check_coordinates pass over PETSc vectors.")
print()

root = rng.normal(size=(n, n))
spd = root @ root.T + n * np.identity(n)
A = operator_from_matrix(X, petsc_matrix(spd), traits=Traits.POSITIVE_DEFINITE)
check_operator(A, rng=rng)

b = X.random(rng=rng)
result = CGSolver(rtol=1e-12)(A).solve(b)
print(f"CG over PETSc vectors converged in {result.iterations} iterations,")
print("  residual", f"{X.norm(X.subtract(A(result.solution), b)) / X.norm(b):.2e}")
print()

# --- now give the space a mass matrix ------------------------------------
mass_root = rng.normal(size=(n, n))
mass = mass_root @ mass_root.T + n * np.identity(n)
V = PetscWeightedSpace(petsc_matrix(mass))

B = operator_from_matrix(V, petsc_matrix(spd))
check_operator(B, rng=rng)
print("on a weighted space the adjoint identity still holds,")
print("  because the adjoint is M^-1 A^T M and not A^T:")

y = V.random(rng=rng)
adjoint = V.to_components(B.adjoint(y))
transpose = np.linalg.solve(mass, spd.T @ mass @ V.to_components(y))
naive = spd.T @ V.to_components(y)

print("   B.adjoint(y) == M^-1 A^T M y ?", np.allclose(adjoint, transpose))
print("   B.adjoint(y) == A^T y        ?", np.allclose(adjoint, naive))
print()
print("PETSc will happily give you the second. It is not the adjoint,")
print("and on an unweighted space the two agree -- which is why the")
print("mistake survives until the metric stops being the identity.")
