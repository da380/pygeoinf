# %% [markdown]
# # Tutorial 12: writing a backend
#
# Every space so far came with the library. This tutorial is for the case
# where the vectors belong to somebody else: a finite-element package, a
# distributed array library, a spectral code. The design was built for this
# — a vector is never wrapped, the space does its arithmetic — so the work
# is to tell the library how, and then to *check* what you told it, because
# a backend that is subtly wrong produces plausible numbers everywhere
# downstream.
#
# The plan: build a small backend by hand, run the ladder of checks on it,
# break it three ways and watch each break get caught, validate an adjoint
# that is only right to discretisation level, and finish with where the
# finite element case is worked through in full.

# %%
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

from pygeoinf import CGSolver, CoordinateSpace, GaussianMeasure, LinearOperator, Traits
from pygeoinf.testing import (
    check_coordinates,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)

rng = np.random.default_rng(12)

# %% [markdown]
# ## The contract
#
# A backend with a basis supplies a `CoordinateSpace`. Six things are
# required:
#
# | | |
# |---|---|
# | `dim` | the dimension |
# | `_key` | what makes two spaces equal: the parameters they were built from |
# | `zero`, `copy`, `axpy`, `scale_inplace` | the vector algebra, on the backend's own vectors |
# | `to_components`, `from_components` | the coordinate map, both ways |
# | `apply_gram`, `solve_gram` | the metric: $G c$ and $G^{-1} c$ on components |
#
# The inner product, the norm, matrices and everything above them follow.
# So does white noise, which every check and every randomized method draws,
# but by default from a dense factorisation of the Gram matrix: right at
# this size, and not at a million unknowns. A backend at scale overrides one
# more member:
#
# | | |
# |---|---|
# | `white_noise_components` | $c = L^{-\mathsf T} \xi$, from a sparse factor $G = L L^{\mathsf T}$; the default is dense |
#
# A coordinate-free backend supplies a plain `HilbertSpace` with an
# `inner_product` instead of the coordinate rows, as the finite element
# space at the end does in one of its two forms, and its own `white_noise`.
#
# Our vectors will be instances of a class that stands for a foreign
# library's: it holds nodal values of a piecewise-linear function on
# $[0, 1]$ and knows no arithmetic at all. The metric is the mass matrix,
# sparse, factorised once; white noise comes from its banded Cholesky
# factor.


# %%
class NodalField:
    """A vector from somebody else's library: it holds values, nothing more."""

    def __init__(self, values):
        self.values = np.array(values, dtype=float)


class PiecewiseLinear(CoordinateSpace):
    """Piecewise-linear functions on [0, 1], with the L2 inner product."""

    def __init__(self, nodes):
        self._nodes = nodes
        h = 1.0 / (nodes - 1)
        diagonal = np.full(nodes, 4.0)
        diagonal[[0, -1]] = 2.0
        off = np.ones(nodes - 1)
        self._mass = sparse.diags([off, diagonal, off], [-1, 0, 1], format="csc") * (
            h / 6.0
        )
        self._factor = sparse_linalg.splu(self._mass)
        # The banded Cholesky factor of the same matrix, G == U^T U.
        self._upper = linalg.cholesky_banded(
            np.vstack([np.r_[0.0, off], diagonal]) * (h / 6.0)
        )

    @property
    def dim(self):
        return self._nodes

    def _key(self):
        return self._nodes

    def zero(self):
        return NodalField(np.zeros(self._nodes))

    def copy(self, x):
        return NodalField(x.values)

    def axpy(self, a, x, y):
        y.values += a * x.values
        return y

    def scale_inplace(self, a, x):
        x.values *= a
        return x

    def to_components(self, x):
        return x.values.copy()

    def from_components(self, c):
        return NodalField(c)

    def apply_gram(self, c):
        return self._mass @ c

    def solve_gram(self, c):
        return self._factor.solve(np.asarray(c, dtype=float))

    def white_noise_components(self, *, rng=None):
        # c = U^-1 xi has covariance U^-1 U^-T == G^-1: the components of
        # white noise on the space, at the cost of a banded solve.
        xi = np.random.default_rng(rng).standard_normal(self._nodes)
        return linalg.solve_banded((0, 1), self._upper, xi)


V = PiecewiseLinear(41)
u = V.random(rng=rng)
print(
    "a vector is a",
    type(u).__name__,
    "and the norm of a random one is",
    round(V.norm(u), 4),
)

# %% [markdown]
# Two rules of the vector algebra deserve emphasis, because both are easy
# to get wrong and neither shows up until much later:
#
# - `axpy` and `scale_inplace` update their target where the backend can and
#   **return** the result; callers always use the return value, so an
#   immutable vector type is fine;
# - `copy` returns an **independent** vector. A Krylov solver keeps several
#   and updates them in place; if two of them are secretly one object, the
#   iteration is arithmetic on garbage.
#
# ## The ladder
#
# Then check it, in this order, stopping at the first failure — each rung
# assumes the ones below it:

# %%
print(check_space(V, rng=rng, rebuild=lambda: PiecewiseLinear(41)))
print(check_coordinates(V, rng=rng))
print(check_white_noise(V, rng=rng))

# %% [markdown]
# - `check_space` tests the vector-space and inner-product axioms on random
#   vectors, and — given `rebuild` — that a second, independently built space
#   compares equal to the first.
# - `check_coordinates` tests that the coordinate maps are inverse to each
#   other and that the Gram pair is consistent with the inner product.
# - `check_white_noise` draws white noise and tests that its covariance is
#   the identity *on the space* — which in components is $G^{-1}$, the
#   mistake every hand-written sampler makes once.
#
# Each returns a `CheckReport`: the worst relative defect it measured for
# each identity. A failure raises instead, so a report in hand means a pass,
# and the numbers say by how much. The white-noise check is statistical, so
# its defect is sampling error, about 1/√(number of samples).
#
# Operators are checked next: `check_operator` for linearity and the adjoint
# identity, `check_traits` for every trait the operator claims. The
# stiffness-plus-mass operator, declared positive definite:

# %%
nodes = V.dim
h = 1.0 / (nodes - 1)
stiffness = (2.0 * np.eye(nodes) - np.eye(nodes, k=1) - np.eye(nodes, k=-1)) / h
stiffness[0, 0] = stiffness[-1, -1] = 1.0 / h
operator = LinearOperator.from_matrix(
    V,
    V,
    stiffness + V.gram_matrix(),
    form="galerkin",
    traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
)
print(check_operator(operator, rng=rng))
print(check_traits(operator, rng=rng))
solve = CGSolver(rtol=1e-10)(operator).solve(V.random(rng=rng))
print("and conjugate gradients accepts it:", solve)

# %% [markdown]
# ## Three ways to break it
#
# A check is only worth something if it fails when it should. Three bugs
# that real backends have had:


# %%
class Aliased(PiecewiseLinear):
    """copy hands back the same object."""

    def copy(self, x):
        return x


class Lumped(PiecewiseLinear):
    """solve_gram uses the lumped (diagonal) mass, apply_gram the consistent one."""

    def solve_gram(self, c):
        return np.asarray(c) / np.asarray(self._mass.sum(axis=1)).ravel()


class ByIdentity(PiecewiseLinear):
    """Equality by object identity, so two equal spaces are not equal."""

    def _key(self):
        return id(self)


for check, broken, options in (
    (check_space, Aliased(41), {}),
    (check_coordinates, Lumped(41), {}),
    (check_space, ByIdentity(41), {"rebuild": lambda: ByIdentity(41)}),
):
    try:
        check(broken, rng=rng, **options)
        print(f"{type(broken).__name__}: passed, which it should not have")
    except AssertionError as failure:
        print(
            f"{type(broken).__name__:10s} caught by {check.__name__}: {str(failure)[:70]}..."
        )

# %% [markdown]
# The lumped mass is the instructive one: a lumped mass matrix is a common
# and legitimate *approximation*, but $G$ and $G^{-1}$ must be the inverse
# of one another, or every representer, adjoint and gradient in the library
# is off by the difference. The identity-equality bug looks harmless until an
# operator built on one copy of a space refuses vectors from another.
#
# ## An adjoint that is only right in the limit
#
# Sometimes the adjoint a backend can supply is the *continuous* one,
# discretised separately from the operator: the adjoint of a PDE solve
# derived on paper and then discretised, say. It satisfies the adjoint
# identity only to discretisation error, so `check_operator` at its default
# tolerance fails — correctly. The right test is a loose tolerance and the
# measured defect *falling* under refinement.
#
# The cumulative integral $f \mapsto \int_0^x f$ has adjoint
# $g \mapsto \int_x^1 g$ in $L^2$. Discretise both with the trapezoidal rule
# and check the pair on four meshes.
#
# The probes matter here. The space's `random` vector is white noise, rough
# in every mode at once, and a discretisation error is largest exactly on
# the modes a mesh barely resolves, so a defect measured on white noise
# swings from probe to probe. Probes drawn from a smooth measure on the
# space, passed as `measure=`, are the kind of field the operator will
# meet. One is built from the backend's own pieces: covariance
# $(1 - \ell^2 d^2/dx^2)^{-2}$, whose factor is the inverse of
# $1 - \ell^2 d^2/dx^2$, a sparse solve.


# %%
def cumulative_integral(nodes):
    space = PiecewiseLinear(nodes)
    h = 1.0 / (nodes - 1)

    def forward(f):
        v = f.values
        return NodalField(
            np.concatenate([[0.0], np.cumsum(0.5 * h * (v[1:] + v[:-1]))])
        )

    def adjoint(g):
        v = g.values[::-1]
        tail = np.cumsum(0.5 * h * (v[1:] + v[:-1]))[::-1]
        return NodalField(np.concatenate([tail, [0.0]]))

    return LinearOperator.from_callables(space, space, forward, adjoint=adjoint)


def smooth_measure(space, length=0.1):
    """The measure of covariance (1 - l^2 d^2/dx^2)^-2 on the space."""
    nodes = space.dim
    main = np.full(nodes, 2.0)
    main[[0, -1]] = 1.0
    off = -np.ones(nodes - 1)
    stiffness = sparse.diags([off, main, off], [-1, 0, 1], format="csc") * (nodes - 1)
    solve = sparse_linalg.splu((space._mass + length**2 * stiffness).tocsc()).solve
    root = LinearOperator.self_adjoint(
        space,
        lambda f: NodalField(solve(space._mass @ f.values)),
        traits=Traits.POSITIVE_DEFINITE,
    )
    return GaussianMeasure(space, covariance_factor=root)


adjoint_identity = "the adjoint satisfies (A x, y) == (x, A* y)"
print(f"{'nodes':>6} {'adjoint defect':>15}")
for count in (41, 161, 641, 2561):
    operator = cumulative_integral(count)
    report = check_operator(
        operator, rng=rng, rtol=0.1, measure=smooth_measure(operator.domain), trials=12
    )
    print(f"{count:6d} {report.defects[adjoint_identity]:15.1e}")

# %% [markdown]
# The defect falls under refinement, by orders of magnitude: the adjoint is
# consistent, and the report
# records by how much at each resolution. That number has a second use. A
# solver asked for a tolerance below it is asking for noise: no iteration
# can make an operator and its adjoint agree better than they do.
#
# ## Finite elements, the worked case
#
# A real finite element space is exactly this construction, and the MFEM
# mini-app, `miniapps/mfem/`, works it through with MFEM's Python
# bindings in two forms. `MfemSpace` reads MFEM's mass matrix as the Gram
# matrix of a coordinate space, as above. `MfemHilbertSpace` supplies no
# coordinates at all: every inner product, mass solve and form application
# is MFEM's own, which is the arrangement a distributed space needs. The
# ladder runs on both, and an inverse problem runs on each. The app also
# wraps MFEM the other way round, as a solve inside an operator on the
# library's own periodic box, with the adjoint written from the transposes
# and the Gram factors, and the ladder proving it.
#
# What MFEM taught, which applies to any backend over memory the library
# does not own:
#
# - **Who owns the buffer.** Foreign vectors can alias memory their library
#   will free. Copy on the way in unless ownership is explicit.
# - **Keep the solver alive.** A wrapped solver may hold no reference to what
#   it needs, and the foreign side will free it; hold a reference yourself.
# - **The mass matrix is the Gram matrix.** That is the whole reason a finite
#   element space fits without adaptation.
# - **Never densify to slice.** Take rows and columns from the sparse
#   structure.
# - **Objects live and die together.** An operator returned by a form may
#   refer back to the form; dropping the form while the operator lives is a
#   crash, not an error.
#
# And the general rule the ladder enforces: a backend is not finished when
# it runs, but when it passes `check_space`, `check_coordinates`,
# `check_white_noise`, `check_operator` and `check_traits` — and, where an
# adjoint is continuous, when its measured defect falls under refinement.
