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
# that is only right to discretisation level, and finish with MFEM. The last
# section needs MFEM (`poetry install --extras mfem`).

# %%
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

from pygeoinf2 import CGSolver, CoordinateSpace, LinearOperator, Traits
from pygeoinf2.testing import (
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
# The inner product, the norm, white noise, matrices and everything above
# them follow. A coordinate-free backend supplies a plain `HilbertSpace`
# with an `inner_product` instead of the last two rows, as the MFEM backend
# at the end does.
#
# Our vectors will be instances of a class that stands for a foreign
# library's: it holds nodal values of a piecewise-linear function on
# $[0, 1]$ and knows no arithmetic at all. The metric is the mass matrix,
# sparse, factorised once.


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
# and check the pair on four meshes:


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


adjoint_identity = "the adjoint satisfies (A x, y) == (x, A* y)"
print(f"{'nodes':>6} {'adjoint defect':>15}")
for count in (11, 41, 161, 641):
    report = check_operator(cumulative_integral(count), rng=rng, rtol=0.1)
    print(f"{count:6d} {report.defects[adjoint_identity]:15.1e}")

# %% [markdown]
# The defect falls steadily: the adjoint is consistent, and the report
# records by how much at each resolution. That number has a second use. A
# solver asked for a tolerance below it is asking for noise: no iteration
# can make an operator and its adjoint agree better than they do.
#
# ## MFEM, the worked case
#
# The library's MFEM backend is exactly this construction for real finite
# element spaces, in two forms. `backends.mfem.MfemSpace` reads MFEM's mass
# matrix as the Gram matrix of a coordinate space, as above.
# `backends.mfem_hilbert.MfemHilbertSpace` goes further and supplies no
# coordinates at all: every inner product, mass solve and form application
# is MFEM's own, which is the arrangement a distributed space needs. The
# ladder runs on both.

# %%
import mfem.ser as mfem  # noqa: E402

from pygeoinf2.backends.mfem import MfemSpace  # noqa: E402
from pygeoinf2.backends.mfem_hilbert import MfemHilbertSpace  # noqa: E402

mesh = mfem.Mesh.MakeCartesian2D(8, 8, mfem.Element.QUADRILATERAL)
elements = mfem.FiniteElementSpace(mesh, mfem.H1_FECollection(2, mesh.Dimension()))
with_coordinates = MfemSpace(elements)
without = MfemHilbertSpace(elements)
print(with_coordinates, "| vectors are", type(with_coordinates.zero()).__name__)
print(check_space(with_coordinates, rng=rng))
print(check_coordinates(with_coordinates, rng=rng))
print(check_space(without, rng=rng))
print("the coordinate-free one has coordinates:", isinstance(without, CoordinateSpace))

# %% [markdown]
# Examples 16, 27 and 29 build inversions on both. What MFEM taught, which
# applies to any backend over memory the library does not own:
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
