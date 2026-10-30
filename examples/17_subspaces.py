"""
17. Subspaces, and the projections onto them.

A subspace is a convex set whose projection happens to be linear, so it carries
the whole convex interface -- including an indicator, which is how a *linear
constraint* enters a proximal method.

The projector is the object doing the work, and it carries its structure:
self-adjoint and idempotent, which closes to positive semidefinite. Its
complement is another projector rather than a generic difference, which would
have forgotten that it is idempotent.
"""

import numpy as np

from pygeoinf import (
    AffineSubspace,
    EuclideanSpace,
    LinearOperator,
    LinearSubspace,
    OrthogonalProjector,
)
from pygeoinf.numerics.convex import ProximalGradient, SquaredDistance
from pygeoinf.symmetric_space import circle
from pygeoinf.testing import check_operator, check_projection, check_traits

rng = np.random.default_rng(0)
X = circle.Sobolev(16, 2.0, 0.3)
Y = EuclideanSpace(4)
# The checks below draw their probes from a smooth measure on the space rather
# than from white noise, which is rough in every mode at once.
probes = X.sobolev_measure(2.0, 0.3)

# --- a projector from a basis: Gram-Schmidt and outer products -----------
vectors = [X.random(rng=rng) for _ in range(5)]
P = OrthogonalProjector.from_basis(X, vectors)
print("projector traits:", P.traits, " <- semidefinite by closure")
print(" ", check_operator(P, rng=rng, measure=probes))
print(" ", check_traits(P, rng=rng, measure=probes))

complement = P.complement()
print("its complement is a", type(complement).__name__, "not a generic sum")

x = X.random(rng=rng)
residual = X.norm(X.subtract(X.add(P(x), complement(x)), x))
print(f"and the pair sums to the identity: residual {residual:.1e}")
print(
    "the two pieces are orthogonal    :",
    abs(X.inner_product(P(x), complement(x))) < 1e-10,
)

print()

# --- the kernel of an operator, coordinate-free --------------------------
A = LinearOperator.from_matrix(X, Y, rng.normal(size=(4, X.dim)), form="components")
kernel = LinearSubspace.from_kernel(A)
print("kernel of A:")
print("  P x = x - A* (A A*)^-1 A x, and A A* is recognized as")
print("  semidefinite from its form, so CG is admissible with")
print("  nothing claimed.")
print("  A P x == 0 ?", np.max(np.abs(A(kernel.project(X.random(rng=rng))))) < 1e-8)
print(f"  dimension {kernel.dimension()} == {X.dim} - 4")
print(" ", check_projection(kernel, rng=rng, measure=probes))
print()

# --- an affine subspace is the solution set of a linear equation ---------
data = Y.random(rng=rng)
solutions = AffineSubspace.from_linear_equation(A, data)
point = solutions.project(X.random(rng=rng))
print("solution set of A x == b:")
print(
    "  a projected point satisfies the equation:",
    np.allclose(A(point), data, atol=1e-8),
)
print(
    "  its translation is the minimum-norm solution:",
    X.norm(solutions.translation) <= X.norm(point) + 1e-8,
)
print("  its tangent space is the kernel:", type(solutions.tangent).__name__)
print()

# --- and so a linear constraint is just another indicator ----------------
result = ProximalGradient(max_iterations=2000, rtol=1e-14).minimize(
    SquaredDistance(X, center=X.random(rng=rng)),
    X.random(rng=rng),
    nonsmooth=solutions.indicator(),
)
print("min ||x - c||^2/2 subject to A x == b")
print(f"  converged in {result.iterations} iterations")
print("  and the constraint holds:", np.allclose(A(result.minimizer), data, atol=1e-7))
print()
print("Nothing above needed a matrix, a basis, or a mass matrix written out.")
print("The one thing that does need coordinates is dimension(), being a trace.")
