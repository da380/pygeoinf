# %% [markdown]
# # Tutorial 3: operators
#
# A linear operator maps one space to another, and comes with an *adjoint*
# going back. The adjoint is where the inner products of both spaces enter,
# and so it is where a discretisation most often goes quietly wrong. This
# tutorial goes slowly, on a space whose inner product is not the dot
# product, because on a space where it is the dot product every mistake
# below is invisible.
#
# Four lessons:
#
# 1. the adjoint is not the transpose;
# 2. an operator you write yourself is checked, not trusted;
# 3. a derivative is not a gradient, and optimising with the wrong one costs
#    iterations that grow with the resolution;
# 4. a matrix says which representation it is in: components, or Galerkin.

# %%
import matplotlib.pyplot as plt
import numpy as np

from pygeoinf import (
    CholeskySolver,
    EuclideanSpace,
    Functional,
    LinearFunctional,
    LinearOperator,
    MassWeightedSpace,
    Traits,
    plotting,
)
from pygeoinf.numerics import LBFGS
from pygeoinf.symmetric_space.circle import Sobolev
from pygeoinf.testing import check_gradient, check_operator

rng = np.random.default_rng(3)

# %% [markdown]
# The space from tutorial 2: piecewise-linear functions on $[0, 1]$, held as
# nodal values, with the integral as inner product and the finite-element
# mass matrix as Gram matrix.


# %%
def piecewise_linear(n):
    """Piecewise-linear functions on [0, 1] with n nodes, and their mass matrix."""
    h = 1.0 / (n - 1)
    mass = (h / 6.0) * (4.0 * np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1))
    mass[0, 0] = mass[-1, -1] = h / 3.0
    base = EuclideanSpace(n)
    operator = LinearOperator.from_matrix(base, base, mass, form="components")
    space = MassWeightedSpace(
        base,
        operator.with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE),
        # The mass matrix is small here, so it is factorised once, densely,
        # rather than solved with iteratively every time a representer is
        # wanted. A real backend would factorise it sparsely (tutorial 12).
        mass_solver=CholeskySolver(),
    )
    return space, mass


X, M = piecewise_linear(21)
h = 1.0 / (X.dim - 1)
nodes = np.linspace(0.0, 1.0, X.dim)

# %% [markdown]
# ## 1. An operator, and its adjoint
#
# An instrument that reports the average of the function over each quarter
# of the interval: four numbers. For a piecewise-linear function the
# average over a quarter is exact in the nodal values, a trapezoidal sum,
# so the operator is a $4 \times 21$ matrix $B$ acting on components. The
# library is told that this is a *component* matrix, $c_{Af} = B\, c_f$.

# %%
Y = EuclideanSpace(4)
B = np.zeros((4, X.dim))
for k in range(4):
    inside = slice(5 * k, 5 * k + 6)  # the six nodes of quarter k
    B[k, inside] = h
    B[k, 5 * k] = B[k, 5 * k + 5] = h / 2.0
B /= 0.25  # an average, not an integral

A = LinearOperator.from_matrix(X, Y, B, form="components")
f = np.sin(np.pi * nodes)
print("quarter averages of sin(pi x):", A(f).round(4))

# %% [markdown]
# The adjoint $A^*$ is *defined* by
#
# $$(A f, y)_Y = (f, A^* y)_X \quad \text{for every } f, y .$$
#
# Both inner products appear. $Y$ is Euclidean; $X$ has the mass matrix. In
# components that makes $A^* = M^{-1} B^{\mathsf T}$, and the library derives
# it from the declaration above. The transpose $B^{\mathsf T}$ alone is a
# different vector, and fails the identity.

# %%
y = Y.random(rng=rng)
transpose = B.T @ y
print(f"(A f, y)_Y           = {Y.inner_product(A(f), y):+.6f}")
print(f"(f, A* y)_X          = {X.inner_product(f, A.adjoint(y)):+.6f}")
print(f"(f, B^T y)_X         = {X.inner_product(f, transpose):+.6f}   <- wrong")
print("A* y == M^-1 B^T y  :", np.allclose(A.adjoint(y), np.linalg.solve(M, transpose)))

# %% [markdown]
# What does $A^*$ *look like*? Apply it to the first unit vector: the result
# is the function that represents "the average over the first quarter".
# It should be four on the first quarter and zero elsewhere, a box of unit
# area. The transpose gives the trapezoid weights instead: numbers of size
# $h$, which shrink as the grid is refined, and are not a function at all.

# %%
e0 = Y.basis_vector(0)
fig, ax = plt.subplots(figsize=(7.0, 3.0), layout="constrained")
ax.plot(nodes, A.adjoint(e0), "o-", label=r"$A^* e_0$, the representer")
ax.plot(nodes, B.T @ e0, "s--", label=r"$B^T e_0$, the transpose")
ax.set_xlabel("x")
ax.legend()
ax.set_title("The adjoint is a function; the transpose is a list of weights")

# %% [markdown]
# The representer rings at the edge of the box: it is the best
# piecewise-linear approximation, in the mean-square sense, to a
# discontinuous function, and such approximations overshoot. Refine the grid
# and the ringing is confined to an ever narrower strip around the jump.
# Refine the grid and the transpose converges to zero.
#
# ## 2. Operators you write yourself
#
# A forward operator in practice is a code, not a matrix: a solver for a
# differential equation, a ray tracer. You give the library its action and
# its adjoint as functions. The trap is the adjoint. A numerical adjoint
# method — the one that differentiates a code — returns the *derivative
# components*, $B^{\mathsf T} y$ here: the derivative of $(A f, y)_Y$ with
# respect to the nodal values. The adjoint wants the representer.
#
# `from_derivative_callables` takes the derivative components and applies
# $M^{-1}$ itself. `from_callables` takes a true adjoint. Hand the second
# the first's output and `check_operator` catches it.

# %%
right = LinearOperator.from_derivative_callables(
    X, Y, lambda f: B @ f, lambda y: B.T @ y
)
wrong = LinearOperator.from_callables(X, Y, lambda f: B @ f, adjoint=lambda y: B.T @ y)

report = check_operator(right, rng=rng)
print("the right construction passes:", report)
try:
    check_operator(wrong, rng=rng)
except AssertionError as failure:
    print("the wrong one fails:", str(failure).splitlines()[0][:72], "...")

# %% [markdown]
# On a Euclidean space the two constructions agree and the check passes for
# both. That is why the error survives in practice: the test problem every
# code is first written against has an orthonormal basis.
#
# ## 3. A derivative is not a gradient
#
# The same split for a functional. Take the data misfit
# $J(f) = \tfrac12 \|A f - d\|^2$. Differentiating a code gives the derivative
# components $B^{\mathsf T}(A f - d)$. The gradient — the direction of
# steepest ascent *in the space's own norm* — is the representer,
# $A^*(A f - d)$. `Functional.from_callables` takes either, by name; the
# documented route is the derivative, and `check_gradient` catches the
# derivative passed as a gradient.

# %%
d = np.array([0.4, 0.9, 0.9, 0.4])


def misfit(f):
    r = B @ f - d
    return 0.5 * float(r @ r)


J = Functional.from_callables(
    X,
    misfit,
    derivative=lambda f: LinearFunctional.from_derivative_components(
        X, B.T @ (B @ f - d)
    ),
)
mistaken = Functional.from_callables(X, misfit, gradient=lambda f: B.T @ (B @ f - d))

check_gradient(J, X.random(rng=rng), rng=rng)
print("check_gradient passes when the derivative is given as a derivative")
try:
    check_gradient(mistaken, X.random(rng=rng), rng=rng)
except AssertionError:
    print("and fails when it is passed off as the gradient")
print(
    "J's gradient is A*(Af - d):",
    np.allclose(J.gradient(f), A.adjoint(A(f) - d)),
)

# %% [markdown]
# Why it matters: an optimiser steps along the gradient. Step along the
# derivative components instead and you are steepest-descending in the
# Euclidean norm of the arrays, a norm tutorial 2 showed means nothing — and
# the conditioning of the problem changes with the grid.
#
# The cleanest place to see it is a space whose Gram matrix is far from the
# identity: $H^2$ on the circle, where the metric weights a Fourier mode of
# wavenumber $k$ by $(1 + \ell^2 k^2)^2$. Fit twelve point values with a
# little smoothing, $J(f) = \tfrac12 |A f - d|^2 + \tfrac{\alpha}{2}
# \|f\|^2_{H^2}$, by L-BFGS, twice: once posed on the space, and once posed
# on the arrays of components, which is what handing the same numbers to a
# component-based optimiser does.

# %%
points = np.sort(rng.uniform(0.0, 2.0 * np.pi, 12))
values = np.sin(points) + 0.3 * np.cos(3.0 * points)
alpha = 1e-2

print(f"{'grid':>5} {'in the space':>13} {'on the arrays':>14}")
for n in (32, 64, 128, 256):
    S = Sobolev(n, 2.0, 0.2)
    evaluate = S.point_evaluation_operator(points, route="dense")
    rows = evaluate.matrix(form="components")
    metric = S.metric_values  # the diagonal Gram matrix of H^2

    def value(f, S=S, evaluate=evaluate):
        r = evaluate(f) - values
        return 0.5 * float(r @ r) + 0.5 * alpha * S.squared_norm(f)

    def derivative(f, S=S, rows=rows, metric=metric):
        c = S.to_components(f)
        g = rows.T @ (rows @ c - values) + alpha * metric * c
        return LinearFunctional.from_derivative_components(S, g)

    on_space = Functional.from_callables(S, value, derivative=derivative)

    # The same objective, as a function of the component array.
    C = EuclideanSpace(n)

    def array_value(c, rows=rows, metric=metric):
        r = rows @ c - values
        return 0.5 * float(r @ r) + 0.5 * alpha * float(c @ (metric * c))

    on_arrays = Functional.from_callables(
        C,
        array_value,
        gradient=lambda c, rows=rows, metric=metric: rows.T @ (rows @ c - values)
        + alpha * metric * c,
    )
    first = LBFGS(max_iterations=10_000).minimize(on_space, S.zero())
    second = LBFGS(max_iterations=10_000).minimize(on_arrays, C.zero())
    print(f"{n:5d} {first.iterations:13d} {second.iterations:14d}")

# %% [markdown]
# The same minimum, found both ways. In the space the iteration count
# hardly moves as the grid is refined; on the arrays it grows without limit,
# because each refinement adds Fourier modes whose metric weight is larger
# and the arrays' conditioning gets worse. Working in the space's inner
# product is a preconditioner you get for nothing. L-BFGS here is written
# against the inner product, which is the whole of the trick.
#
# ## 4. Components or Galerkin: which matrix?
#
# Back on the piecewise-linear space. The operator $-\,d^2/dx^2$ is usually
# discretised through its weak form, $(L f, g) = \int f' g'\,dx$, which a
# finite-element code assembles as the *stiffness matrix* $K$. Its entries
# are inner products, $K_{ij} = (L \phi_j, \phi_i)$ — a matrix in
# **Galerkin** form. The **component** matrix, $c_{Lf} = L_c\, c_f$, is
# $M^{-1} K$.
#
# They differ by a factor of the Gram matrix, and the difference is not
# cosmetic. $L$ is self-adjoint, and it is the Galerkin matrix that is
# symmetric: symmetry of the component matrix is self-adjointness only on an
# orthonormal basis. So `from_matrix` requires you to say which you have.

# %%
K = (2.0 * np.eye(X.dim) - np.eye(X.dim, k=1) - np.eye(X.dim, k=-1)) / h
K[0, 0] = K[-1, -1] = 1.0 / h

L = LinearOperator.from_matrix(X, X, K, form="galerkin")
galerkin = L.matrix(form="galerkin")
components = L.matrix(form="components")
print("Galerkin matrix symmetric  :", np.allclose(galerkin, galerkin.T))
print("component matrix symmetric :", np.allclose(components, components.T))
g = X.random(rng=rng)
print(
    "L is self-adjoint in X     :",
    np.isclose(X.inner_product(L(f), g), X.inner_product(f, L(g))),
)

# Declared with the wrong form, K is a different operator, and not a
# self-adjoint one.
mislabelled = LinearOperator.from_matrix(X, X, K, form="components")
print(
    "mislabelled is self-adjoint:",
    np.isclose(X.inner_product(mislabelled(f), g), X.inner_product(f, mislabelled(g))),
)

# %% [markdown]
# The rule of thumb: a matrix read off a discretisation as "new values from
# old values" is in components; a matrix assembled from a bilinear form, or
# made of rows of derivative components, is Galerkin. Direct solvers that
# need symmetry ask the operator for its Galerkin matrix themselves, so once
# the form is declared correctly nothing further has to be remembered.
#
# ## Operators compose, and adjoints come along
#
# Operators add, scale and compose with `+`, `*` and `@`. The adjoint of a
# composition is the composition of the adjoints in reverse order, built
# automatically. Structure is recorded as *traits*: $A^* A$ is self-adjoint
# and positive semidefinite by construction, which is what lets the
# conjugate-gradient solver of tutorial 4 accept it without anyone claiming
# anything.
#
# A matrix proves nothing about itself, so `L` above carries no traits, and
# a sum keeps a trait only when every term has it. Say what you know when
# building an operator — `testing.check_traits` verifies the claim — and it
# survives the algebra.

# %%
print("traits of A* A          :", (A.adjoint @ A).traits)
print("traits of A* A + 0.1 L  :", (A.adjoint @ A + 0.1 * L).traits)
stiffness = LinearOperator.from_matrix(
    X, X, K, form="galerkin", traits=Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE
)
print("with L's traits declared:", (A.adjoint @ A + 0.1 * stiffness).traits)
composed = A @ L
print(
    "(A L)* == L* A*      :",
    np.allclose(composed.adjoint(y), L.adjoint(A.adjoint(y))),
)
plotting.show()
