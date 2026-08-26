"""
7. Solvers declare what they need, and the iterative ones need no coordinates.

CG, MINRES, BiCGStab and LSQR are written against the inner product and axpy
alone, so they run on a space with no component map at all. The direct solvers
say that they need one.

And a solver states its preconditions as traits, so an operator that has not
earned them is refused before any work is done.
"""

import numpy as np

from pygeoinf2 import LinearOperator
from pygeoinf2.numerics import CGSolver, CholeskySolver, MinResSolver
from pygeoinf2.spaces import Sobolev

rng = np.random.default_rng(0)
X = Sobolev((32,), 2.0, 0.3)

# A positive-definite operator on the Sobolev space, built from its spectrum.
A = X.invariant_operator(lambda values: 1.0 + values)
print("A traits:", A.traits)
print()

b = X.random(rng=rng)
result = CGSolver(rtol=1e-12)(A).solve(b)
print(
    f"CG converged in {result.iterations} iterations, residual {result.residual_norm:.2e}"
)
print("A x == b ?", X.norm(X.subtract(A(result.solution), b)) < 1e-8 * X.norm(b))
print()
print("Every inner product in that solve carried the Sobolev metric.")
print("Nothing touched a component array.")
print()

# Preconditions, checked.
unclaimed = LinearOperator.from_callables(X, X, A, adjoint=A)
try:
    CGSolver()(unclaimed)
except ValueError as error:
    print("CG refuses an operator with no claim:")
    print("  ", str(error).split(". ")[0])
print()

# MINRES asks for less, and gets it.
print("MINRES requires:", MinResSolver.requires)
print("CG requires    :", CGSolver.requires)
print("Cholesky needs coordinates:", CholeskySolver.requires_coordinates)
print()

# Diagnostics come back with the answer; the solver itself is stateless.
solver = CGSolver(rtol=1e-10)
first = solver(A).solve(b)
second = solver(A).solve(X.random(rng=rng))
print(
    f"one solver, two problems: {first.iterations} and {second.iterations} iterations"
)
