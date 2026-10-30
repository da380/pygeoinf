# %% [markdown]
# # Tutorial 2: spaces
#
# Everything in this library happens in a Hilbert space: a vector space with
# an inner product. That sounds like the most abstract possible starting
# point, and it is the most practical one, because the inner product is
# exactly the thing that a discretised function and an array of numbers do
# *not* share. This tutorial is about that difference.
#
# Four ideas:
#
# 1. vectors are whatever the backend hands you, and the space does the
#    arithmetic;
# 2. the inner product is the space's, and on a discretised function it is
#    not the dot product of the arrays;
# 3. coordinates are a capability some spaces have, not a requirement;
# 4. there are no dual spaces: a linear functional is represented by a
#    vector of the space itself.

# %%
import numpy as np

from pygeoinf import (
    CholeskySolver,
    CoordinateSpace,
    EuclideanSpace,
    LinearFunctional,
    LinearOperator,
    MassWeightedSpace,
    Traits,
)

rng = np.random.default_rng(2)

# %% [markdown]
# ## 1. The space does the arithmetic
#
# A vector of `EuclideanSpace(4)` is a plain NumPy array. Nothing wraps it.
# But you add, scale and pair vectors *through the space*, never by
# operators on the vectors themselves. That is what lets a PETSc vector or
# an MFEM grid function be a vector here without being adapted: the space
# knows how to do arithmetic on them, and the vector need not.

# %%
E = EuclideanSpace(4)
x, y = E.random(rng=rng), E.random(rng=rng)
print("a vector is a", type(x).__name__, "of shape", x.shape)
print("x + y  :", E.add(x, y).round(3))
print("(x, y) :", round(E.inner_product(x, y), 4))
print("||x||  :", round(E.norm(x), 4))

# The in-place operations a Krylov method wants update their target where
# the backend allows it and *return* the result. Always use the return value:
# for an immutable vector type there is nothing to update in place.
z = E.axpy(2.0, x, E.copy(y))
print("axpy(2, x, y) == y + 2x:", np.allclose(z, y + 2.0 * x))

# %% [markdown]
# ## 2. A space of functions, and its inner product
#
# Now a space whose vectors *mean* something. Take the functions on $[0, 1]$
# that are piecewise linear between $n$ equally spaced nodes, and hold each as
# its $n$ nodal values. The natural inner product is the integral,
#
# $$(f, g) = \int_0^1 f(x)\, g(x)\, dx = c_f^{\mathsf T} M\, c_g ,$$
#
# where $c_f$ are the nodal values and $M$ is the *mass matrix* of the
# finite-element method: tridiagonal, and not the identity. The library
# builds such a space from a base space and a mass operator.


# %%
def piecewise_linear(n):
    """Piecewise-linear functions on [0, 1] with n nodes, under the L2 product."""
    h = 1.0 / (n - 1)
    mass = (h / 6.0) * (4.0 * np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1))
    mass[0, 0] = mass[-1, -1] = h / 3.0
    base = EuclideanSpace(n)
    operator = LinearOperator.from_matrix(base, base, mass, form="components")
    return MassWeightedSpace(
        base,
        operator.with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE),
        # The mass matrix is small here, so it is factorised once, densely,
        # rather than solved with iteratively every time a representer is
        # wanted. A real backend would factorise it sparsely (tutorial 12).
        mass_solver=CholeskySolver(),
    )


X = piecewise_linear(11)
nodes = np.linspace(0.0, 1.0, X.dim)
print("the Gram matrix of X is the mass matrix:")
print((X.gram_matrix()[:4, :4] * 60).round(2), "... (times 60)")

# %% [markdown]
# The point of all this shows up under refinement. Take $f(x) = \sin \pi x$,
# whose $L^2$ norm is $1/\sqrt 2 \approx 0.7071$, and measure it two ways on
# three grids: by the space's norm, and by the Euclidean length of the array
# of nodal values.

# %%
print(f"{'nodes':>6} {'space norm':>11} {'array length':>13}")
for n in (11, 101, 1001):
    space = piecewise_linear(n)
    f = np.sin(np.pi * np.linspace(0.0, 1.0, n))
    print(f"{n:6d} {space.norm(f):11.4f} {np.linalg.norm(f):13.4f}")

# %% [markdown]
# The space's norm converges to the right answer. The array's length grows
# like $\sqrt n$, and means nothing. Every algorithm in this library measures
# distances, steps, residuals and convergence in the space's norm, which is
# why its behaviour does not drift when the grid is refined. An algorithm
# that worked with the arrays would be solving a different problem on every
# grid.

# %% [markdown]
# ## 3. Coordinates are a capability
#
# `X` has *components*, the nodal values, and a Gram matrix relating their
# dot product to the inner product. Spaces with a basis say so by being a
# `CoordinateSpace`, and methods that need a matrix check for it. But a
# Hilbert space does not need a basis: an inner product and vector
# arithmetic are enough, and tutorial 12 builds one with no coordinates at
# all, where a finite-element package does every computation itself.

# %%
print("X has coordinates:", isinstance(X, CoordinateSpace))
print("its basis is orthonormal:", X.is_orthonormal)
c = X.to_components(np.sin(np.pi * nodes))
print("components are the nodal values here:", np.allclose(c, np.sin(np.pi * nodes)))

# %% [markdown]
# ## 4. Functionals, and why there are no dual spaces
#
# A *linear functional* maps a vector to a number linearly: the integral
# $I(f) = \int_0^1 f\,dx$, or the value at a point. Written in components it
# is a row vector $g$ with $I(f) = g \cdot c_f$. For the integral, $g = M
# \mathbf{1}$: the column sums of the mass matrix.
#
# The library builds a functional from that row vector, and then offers two
# readings of it:
#
# - its **derivative components**, $g$ itself — what an adjoint code
#   returns, tied to the basis;
# - its **representer**, the one vector $r$ of the space with
#   $I(f) = (r, f)$ for every $f$ — the Riesz representer, and the
#   *gradient* of the functional.
#
# They are related by $c_r = M^{-1} g$, and they are different vectors.

# %%
row = X.gram_matrix() @ np.ones(X.dim)
integral = LinearFunctional.from_derivative_components(X, row)
f = np.sin(np.pi * nodes)
print(f"I(sin pi x) = {integral(f):.4f}   (exact 2/pi = {2 / np.pi:.4f})")
print("derivative components:", (row * 10).round(3), "(times 10)")
print("representer          :", integral.representer.round(3))

# %% [markdown]
# The derivative components are half as large at the ends, an artefact of
# the basis. The representer is the constant function one, which is the
# honest answer: $\int f\,dx = (1, f)$.
#
# The representer is `integral.adjoint(1.0)`. A functional is an operator
# into the real numbers, and its adjoint maps a number back into the space;
# applied to one, it returns the representer. So every functional lives in
# the space it acts on, through its adjoint, and the library has no separate
# dual space and no dual vectors to keep track of.

# %%
print("representer == adjoint(1):", np.allclose(integral.adjoint(1.0), np.ones(11)))

# %% [markdown]
# The distinction matters most where the representer misbehaves. Evaluation
# at the midpoint has derivative components $e_j$, a one in the middle node.
# Its representer is $M^{-1} e_j$, and its norm is the norm of the
# functional. Watch it under refinement.

# %%
print(f"{'nodes':>6} {'norm of evaluation at 1/2':>27}")
for n in (11, 101, 1001):
    space = piecewise_linear(n)
    spike = np.zeros(n)
    spike[n // 2] = 1.0
    evaluation = LinearFunctional.from_derivative_components(space, spike)
    print(f"{n:6d} {space.norm(evaluation.representer):27.2f}")

# %% [markdown]
# It grows without bound, like $1/\sqrt h$. Point evaluation is not a
# continuous functional on $L^2$: a square-integrable function has no value
# at a point. On every grid the computation runs and returns a number, and
# nothing in the arrays warns you. The norm of the representer does.
#
# This is why the first tutorial used the Sobolev space $H^2$ rather than
# $L^2$: there, point values exist, and the representer of evaluation
# converges. Tutorial 5 returns to it, and to the library's habit of refusing
# such a computation on a space that does not admit it.
#
# ## Spaces are values
#
# Two spaces built the same way are equal, and spaces can be dictionary
# keys. Equality is what lets the library check that an operator's domain is
# the space a vector came from, which catches a whole class of mistakes at
# the point they are made.

# %%
print("EuclideanSpace(4) == EuclideanSpace(4):", E == EuclideanSpace(4))
print("usable as a dict key:", {E: "yes"}[EuclideanSpace(4)])
