"""
8. f(A) without a matrix.

Lanczos gives ``f(A) x`` from the operator's action alone, so a covariance
square root or a log-determinant is available on a space with no basis. An
operator that happens to be diagonal takes a faster path -- exactly, on its
eigenvalues -- and the two agree.
"""

import numpy as np
import scipy.linalg as sla

from pygeoinf2 import EuclideanSpace, LinearOperator, Traits
from pygeoinf2.numerics import (
    apply_operator_function,
    operator_log,
    operator_quadratic_form,
    operator_sqrt,
)
from pygeoinf2.symmetric_space import Sobolev

rng = np.random.default_rng(0)

# A dense operator: Lanczos, matrix-free.
X = EuclideanSpace(20)
root = rng.normal(size=(20, 20))
matrix = root @ root.T + 20.0 * np.identity(20)
A = LinearOperator.from_component_matrix(X, X, matrix, traits=Traits.POSITIVE_DEFINITE)

x = X.random(rng=rng)
print(
    "sqrt(A) x vs scipy.sqrtm :",
    np.max(
        np.abs(
            apply_operator_function(A, np.sqrt, x, max_iterations=20)
            - np.real(sla.sqrtm(matrix)) @ x
        )
    ),
)
print(
    "log(A) x  vs scipy.logm  :",
    np.max(
        np.abs(
            apply_operator_function(A, np.log, x, max_iterations=20)
            - np.real(sla.logm(matrix)) @ x
        )
    ),
)
print()

# (x, f(A) x) by Gauss quadrature on the Lanczos spectrum -- the kernel of
# stochastic Lanczos quadrature, and so of log-determinant estimation.
quadratic = operator_quadratic_form(A, np.log, x, max_iterations=20)
print("(x, log(A) x) :", round(quadratic, 6))
print("  exact       :", round(float(x @ np.real(sla.logm(matrix)) @ x), 6))
print()

# f(A) as an operator, carrying the right claim.
print("operator_sqrt(A).traits:", operator_sqrt(A).traits)
print("operator_log(A).traits :", operator_log(A).traits, " <- a log is indefinite")
print()

# On a space whose operators are diagonal, the calculus is exact and free.
H = Sobolev((32,), 2.0, 0.3)
smoother = H.invariant_operator(lambda values: np.exp(-0.01 * values))
root = smoother.sqrt
print("a diagonal operator takes the fast path:", type(root).__name__)
y = H.random(rng=rng)
print(
    "root(root(y)) == smoother(y) ?",
    H.norm(H.subtract(root(root(y)), smoother(y))) < 1e-12,
)
