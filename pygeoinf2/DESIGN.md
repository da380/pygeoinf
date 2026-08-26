# pygeoinf 2.0 — design of the algebraic core

Status: design agreed, not yet implemented.
Scope of this document: the base classes only — `HilbertSpace`, `Operator` /
`LinearOperator`, `LinearSolver`, `ProbabilityMeasure` / `GaussianMeasure`, and
the trait system that ties them together. Numerics, inversion, geometry and the
concrete function spaces follow from these and are out of scope here.

---

## 1. Decisions taken

| # | Question | Decision |
|---|----------|----------|
| 1 | Primal/dual | **Riesz-identify everywhere.** No dual spaces, no `to_dual`/`from_dual`, no `LinearForm`. Operators have `.adjoint` only. |
| 2 | Vectors | **Raw backend objects**; all arithmetic mediated by the space. No wrapper class. |
| 3 | Structure | **Traits (flags) with algebraic propagation**, not subclasses. |
| 4 | Nonlinear | **Explicit `Linearisation` object** returned by `F.at(x)`, so value and derivative share work. Derivatives and second derivatives are *optional* and carried by the operator. |
| 5 | Algebra | **Structured expression nodes**, not closures. |
| 6 | Functionals | **Thin `Functional` subclass** over a canonical `Reals` space; forms are subsumed, not removed. The **derivative is primitive, the gradient is derived** through the adjoint — see §5.6. |
| 7 | Concrete spaces | **Adapter first**: wrap v1 spaces as v2 spaces, rewrite natively later. |
| 8 | Packaging | **`pygeoinf2/`**, a sibling top-level package, relative imports throughout, renamed to `pygeoinf/` at release. |

### 1.1 The consequence of decision 1 that shapes everything else

Dropping duals does not drop the mass matrix; it **relocates it into the
coordinate layer**. For a coordinatised space with Gram (mass) matrix `G`:

```
inner_product(x, y) = c_x^T G c_y
(A x)_c             = A_c c_x                     "component" matrix
adjoint component matrix = G_X^{-1} A_c^T G_Y     NOT A_c^T
```

So `A_c^T` is the matrix of the *dual* map, which we no longer name. Three
things follow, and they are improvements rather than concessions:

1. `gram` and `gram_solver` belong to `CoordinateSpace`, not to `HilbertSpace`.
   A coordinate-free space (PETSc, MFEM) has no Gram matrix and is no longer
   obliged to invent one — in v1 every space must implement `to_dual`, which is
   meaningless without a basis.
2. **Self-adjointness is visible as matrix symmetry only in the Galerkin
   representation.** So the matrix representation is chosen *from the
   operator's traits*, not by a `galerkin=True` flag hand-threaded through
   `matrix()`, `CholeskySolver`, `EigenSolver`, `CGSolver` and every
   preconditioner, as it is today.
3. The Riesz map has not disappeared, it is expressed through the adjoint:
   for a linear functional `f: X -> Reals`, `f.adjoint(1.0)` **is** the Riesz
   representer. This is why `Functional.gradient` returns a vector in `X`
   with no extra machinery.

**What is lost:** you can no longer hold a functional in "load vector" form
without a mass solve. Mitigated by exposing `gram` / `gram_solver` explicitly,
so a user who wants that does it deliberately and visibly.

---

## 2. Code practice

Three house rules, enforced by `tests/test_code_practice.py` rather than
recorded here and forgotten:

1. **Docstrings on every public class and function.** For an override, the
   docstring says what is specific to the override rather than repeating the
   base — "the summands' components, concatenated", not "returns the
   components".
2. **Type hints on every parameter and return.** Including private methods:
   the subclass contract (`_value`, `_adjoint_value`, `_solve`) is where a
   reader most needs to know what is expected.
3. **Optional arguments are keyword-only.** An optional positional argument is
   a compatibility hazard: once callers pass it positionally its position is
   part of the API, and nothing can be inserted before it. So `random(*, rng)`,
   `zero(domain, *, codomain)`, `DirectSum(spaces, *, labels)`.

The test parametrises over every module, so a failure names the file, line and
symbol.

### 2.1 Scope

The focus is the **algebraic structures and numerical methods**, coordinate-free
wherever possible, with SciPy-backed coordinate implementations available as an
option rather than as the foundation. The inversion layer is deliberately out of
scope for now.

Within numerics, the areas that matter:

| area | v1 state | intent |
|---|---|---|
| Randomised range finding, low-rank factorisation | partly coordinate-free already | retain that, and finish the job |
| Functional calculus (`f(A)` via Lanczos) | matrix-free | port and generalise (§5.7) |
| Nonlinear optimisation | a thin SciPy wrapper that conflates gradients and derivatives | rewrite as bespoke coordinate-free methods, as the linear solvers were |
| Convex optimisation | entangled with the inversion layer | leave for now |

### 2.2 Package layout

```
pygeoinf2/
  __init__.py            curated public API
  traits.py              Traits flag, closure and propagation rules
  algebra/
    spaces.py            HilbertSpace, CoordinateSpace, EuclideanSpace, Reals
    direct_sum.py        DirectSum of spaces, block operators
    operators.py         Operator, LinearOperator, Functional, AffineOperator
    nodes.py             _Sum, _Scaled, _Composition, _Adjoint, _Inverse, ...
    linearisation.py     Linearisation
  numerics/
    solvers.py           LinearSolver and implementations
    preconditioners.py
  probability/
    base.py              ProbabilityMeasure
    gaussian.py          GaussianMeasure
  compat.py              adapter presenting a v1 space as a v2 space
  testing.py             axiom and trait checks (out of the production classes)
```

Everything imports relatively (`from ..traits import Traits`) so that the
release-time rename of `pygeoinf2/` to `pygeoinf/` touches no import statement.

---

## 3. Spaces

### 3.1 `HilbertSpace` — the coordinate-free core

```python
class HilbertSpace[V](ABC):
    """A real Hilbert space whose vectors are opaque objects of type V."""

    # ---- identity ------------------------------------------------------
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def _key(self) -> Hashable:
        """Structural identity. Two spaces are equal iff their keys are."""

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash((type(self), self._key()))

    # ---- required vector operations ------------------------------------
    @abstractmethod
    def zero(self) -> V: ...                          # a METHOD: it allocates
    @abstractmethod
    def copy(self, x: V) -> V: ...
    @abstractmethod
    def inner_product(self, x: V, y: V) -> float: ...
    @abstractmethod
    def axpy(self, a: float, x: V, y: V) -> V: ...    # y <- y + a*x, in place
    @abstractmethod
    def scale_inplace(self, a: float, x: V) -> V: ... # x <- a*x

    # ---- derived (final, overridable only for performance) -------------
    def add(self, x, y) -> V:        return self.axpy(1.0, x, self.copy(y))
    def subtract(self, x, y) -> V:   return self.axpy(-1.0, y, self.copy(x))
    def scale(self, a, x) -> V:      return self.scale_inplace(a, self.copy(x))
    def negative(self, x) -> V:      return self.scale(-1.0, x)
    def norm(self, x) -> float:      return sqrt(self.inner_product(x, x))
    def squared_norm(self, x) -> float
    def gram_schmidt(self, vectors) -> list[V]
    def mean(self, vectors) -> V

    # ---- randomness (explicit generator, no global state) --------------
    def random(self, rng: Generator | None = None) -> V:
        """An arbitrary random vector. For testing. NOT white noise."""
    def white_noise(self, rng: Generator | None = None) -> V:
        """A sample with covariance equal to the identity ON THIS SPACE."""

    # ---- convenience ----------------------------------------------------
    def identity(self) -> LinearOperator[V, V]
    def zero_operator(self, codomain=None) -> LinearOperator
```

Notes on specific changes from v1, each with a reason:

- **`zero()` is a method, not a property.** In v1 `space.zero` allocates a fresh
  vector on every access while looking free; it is read in loops.
- **`__hash__` is defined.** v1 declares `__eq__` abstract and never defines
  `__hash__`, so *every* `HilbertSpace` in the library is unhashable
  (confirmed: `{gi.EuclideanSpace(3): 1}` raises `TypeError`). Nothing can be
  memoised per space, and no `set[HilbertSpace]` can exist.
- **`random` and `white_noise` are separated.** In v1 `HilbertSpace.random`
  returns `from_components(randn(dim))`, which has covariance `G`, not the
  identity, whenever the component basis is not orthonormal. See §9.
- **`to_dual` / `from_dual` / `dual` are gone** (decision 1).
- **`to_components` / `from_components` / `basis_vector` are gone from the
  base** and move to `CoordinateSpace`.
- **`is_element` is gone.** With raw backend vectors it cannot be implemented
  honestly — v1's `isinstance(x, type(self.zero))` cannot distinguish two
  distinct 3-dimensional spaces both backed by `np.ndarray`. Replaced by
  `pygeoinf2.testing` checks used in tests, not in production paths.

### 3.2 `CoordinateSpace` — the optional capability

```python
class CoordinateSpace[V](HilbertSpace[V], ABC):
    """A Hilbert space with a distinguished finite basis."""

    @abstractmethod
    def to_components(self, x: V) -> np.ndarray: ...
    @abstractmethod
    def from_components(self, c: np.ndarray) -> V: ...

    @property
    def gram(self) -> LinearOperator:
        """The metric/mass operator on R^n. Identity by default."""

    @property
    def gram_solver(self) -> LinearSolver: ...

    # supplied for free from the above
    def inner_product(self, x, y) -> float          # c_x^T G c_y
    def basis_vector(self, i: int) -> V
    def white_noise(self, rng) -> V                 # components ~ N(0, G^{-1})
    def zero(self) -> V
```

Specialisations, in increasing generality:

| class | Gram | supplies |
|---|---|---|
| `OrthonormalSpace` | `I` | fastest path; `inner_product = c_x . c_y` |
| `DiagonalMetricSpace` | `diag(g)` | v1's `OrthogonalHilbertSpace` |
| `CoordinateSpace` | any `LinearOperator` | v1's `MassWeightedHilbertSpace` |

`EuclideanSpace(n)` is an `OrthonormalSpace` with `V = np.ndarray`.
`Reals` is a singleton `OrthonormalSpace` with `V = float`, `dim == 1`,
`inner_product(x, y) = x * y`. Using `float` rather than `np.ndarray` of
length 1 means a functional evaluates to a plain number.

Numerical methods that need coordinates declare it and validate at
construction time:

```python
def require_coordinates(*spaces) -> None:
    for s in spaces:
        if not isinstance(s, CoordinateSpace):
            raise TypeError(f"{type(s).__name__} provides no coordinate map; "
                            f"this method requires one.")
```

The payoff: the **native iterative solvers, the operator algebra, the traits
system, and Gaussian sampling from a factor all work with no coordinates at
all**, so they run unchanged against a PETSc- or MFEM-backed space. Only the
direct solvers, dense/sparse matrix representations, and randomised linear
algebra require the coordinate capability.

### 3.3 Direct sums, and why there is no tensor product

**Direct sums are in.** `DirectSum([X, Y, ...])` is a `HilbertSpace` whose
vectors are tuples of the component vectors and whose inner product is the sum
of the component inner products. It needs nothing from the backends beyond what
they already provide, it is a `CoordinateSpace` exactly when every summand is,
and its Gram matrix is block diagonal. The block operator classes
(`BlockOperator`, `RowOperator`, `ColumnOperator`, `BlockDiagonalOperator`)
carry over from v1 essentially unchanged, now with trait propagation: a block
diagonal operator is self-adjoint iff every block is, and PSD iff every block
is.

**Tensor products of spaces are out.** There is no general concrete
representation of `X (x) Y` that works across arbitrary backends: it needs a
basis on both factors and a rule for combining them, which is precisely what a
coordinate-free space does not have. Nothing in the library currently requires
one. If a specific case is ever needed (a separable space on a product domain,
say), it is better built as a concrete space that happens to be a tensor
product than as a general construction.

Two things that *sound* like tensor products and are staying, to avoid
confusion:

- `LinearOperator.from_tensor_product(u, v)` — the **outer product** operator
  `x |-> (v, x) u`, a rank-one map. This is an operator construction, not a
  space construction, and it is the useful practical slice of tensor-product
  structure.
- Low-rank operators (`LowRankSVD` and friends) — sums of outer products, which
  is where that structure earns its keep numerically.

### 3.4 Complex data, without a complex core

The core stays **real**. Complex-valued data is supported by *realification*,
which costs nothing and changes no other class.

A complex Hilbert space `H` with Hermitian inner product `<.,.>`, viewed with

```
(x, y)_R = Re <x, y>
```

is an ordinary real Hilbert space of twice the dimension, with the same norm.
Crucially `Re<Ax, y> = Re<x, A*y>`, so **adjoints agree**: a `C`-linear
operator's Hermitian adjoint *is* its real adjoint on the realification.
Hermitian becomes `SELF_ADJOINT`, positive-definite stays positive-definite,
and every trait, solver, measure and algebra rule in this document applies
unchanged.

```python
class ComplexSpace[V](CoordinateSpace[V]):
    """Vectors are complex arrays; the space is real of dimension 2n."""
    def inner_product(self, x, y) -> float:
        return float(np.real(np.vdot(x, self._gram_apply(y))))
    def to_components(self, x) -> np.ndarray:      # length 2n, real
        return np.concatenate([x.real, x.imag])
```

This covers the case that actually arises: **frequency-domain waveform
inversion**, where the model space is real, the data space is complex, and the
objective is real. That needs a complex codomain, not a complex core.

**Why not a genuinely complex core.** The space and operator layers generalise
cleanly — sesquilinear inner product, Hermitian adjoint, a convention choice
about which argument is conjugate-linear. The *functional* layer does not. A
real-valued objective on a complex space is not holomorphic, so its
differential calculus is Wirtinger calculus, and that assumption would spread
through `Functional`, `QuadraticModel`, every optimiser and every convex
object. Worse, a real-valued `C`-linear functional does not exist except the
zero one, so `LinearFunctional[X, Reals]` would have to become
`LinearFunctional[X, Complexes]` and the neat statement of §5.6 would need
restating.

**What realification gives up**: genuinely complex-*linear* structure —
complex eigenvalues and eigenvectors, and Krylov methods run in complex
arithmetic rather than on the doubled real system. Neither is needed for
inference against complex data. If it is ever wanted, a `ComplexHilbertSpace`
branch can be added later without disturbing anything here, precisely because
the realification route touches no other class.

### 3.5 Mass-weighted spaces and formal adjoints

**Correction to an earlier draft.** The concept map previously said
`from_formal_adjoint` was "removed — the Gram matrix lives in the space, so no
lifting step is needed". That is wrong, and it conflated two different objects:

| object | layer | what it relates |
|---|---|---|
| `CoordinateSpace.gram` | coordinate | the inner product to the **component basis** |
| mass operator | core | one inner product to **another inner product on the same vectors** |

Only the first is automatic. The second is a genuine construction, and the
lifting step it needs does not disappear.

```python
class MassWeightedSpace[V](HilbertSpace[V]):
    """(x, y)_V = (M x, y)_base, with M self-adjoint positive-definite on base."""
    base: HilbertSpace[V]
    mass: LinearOperator          # SELF_ADJOINT | POSITIVE_DEFINITE on base
    mass_solver: LinearSolver

    def inner_product(self, x, y):
        return self.base.inner_product(self.mass(x), y)
```

Note this needs **no coordinates** — only `base.inner_product` and `M` — so it
belongs in the core space layer and works against a PETSc-backed base. When the
base *is* coordinatised, the two compose: `MassWeightedSpace(base, M).gram ==
base.gram @ M_c`. v1's spaces are exactly this chain: component-Euclidean, then
L2 with a diagonal Gram, then Sobolev via an invariant mass operator on L2.

#### Lifting an operator from the base space

The practical point: it is usually far easier to derive the action of the
**formal (L2) adjoint** than the adjoint with respect to a weighted inner
product. That workflow is preserved. With `(x,y)_V = (Mx, y)_U`, requiring
`(Ax, y)_{V_Y} = (x, A^{*V} y)_{V_X}` gives

```
A^{*V} = M_X^{-1} A^{*U} M_Y
```

which is the v1 formula unchanged, and

```python
A = LinearOperator.from_formal_adjoint(V_X, V_Y, A_l2)
```

stays the way to say it. The name is kept: it is accurate, and it is used
throughout the concrete spaces (`sphere.py`, `plane.py`, `torus.py`,
`circle.py`, `symmetric_space.py`) so continuity is worth more than a
marginally better name.

Two things improve:

1. **`M^{-1}` is derived, not supplied.** v1 requires the user to hand
   `inverse_mass_operator` to the space constructor. v2's `MassWeightedSpace`
   carries a `mass_solver`, so the inverse is exact when the mass operator is
   spectrally diagonal (the usual case) and iterative otherwise. One fewer
   thing to get wrong, and the construction becomes usable when `M` has no
   closed-form inverse.
2. **Direct sums stop being a special case.** v1's `from_formal_adjoint`
   contains a recursive `get_properties` helper that walks
   `HilbertSpaceDirectSum` and assembles block-diagonal mass operators. In v2 a
   direct sum of mass-weighted spaces simply *is* a mass-weighted space with
   block-diagonal mass, so the recursion is a property of `DirectSum` and the
   lifting code has no knowledge of it.

#### One name not worth inheriting

There is nothing wrong with a lifted operator failing to be self-adjoint on the
weighted space — that is simply what the mathematics gives. A formally
self-adjoint `A` is self-adjoint with respect to `(.,.)_V` only when it
commutes with the mass operator:

```
(Ax, y)_V = (x, Ay)_V  for all x, y   <=>   M A = A^{*U} M   <=>   M A = A M
```

which for a variable-coefficient operator on a Sobolev space it generally does
not. Measured on `circle.Sobolev(32, 2.0, 0.2)` with a rank-2 operator that is
*exactly* formally self-adjoint on L2: `(Ax, y)_V = -131989` against
`(x, Ay)_V = +162009`. The lifted `.adjoint` is correct throughout — verified
in the same run — so the operator is perfectly usable. It is just not
symmetric, and there is no reason it should be.

The only thing to change is a name. v1's `from_formally_self_adjoint` is a
one-line alias for `from_formal_adjoint(domain, domain, operator)` whose name
and docstring ("promotes it to a truly self-adjoint operator") describe
something it does not do. In v1 that is a documentation defect and nothing
more, since no caller relies on the claim. In v2 it would acquire teeth,
because the constructor is precisely where a trait gets attached, and a name
reading "self-adjoint" invites attaching `SELF_ADJOINT` to an operator that is
not.

Resolved by having one constructor rather than two, with the codomain
defaulting to the domain:

```python
@classmethod
def from_formal_adjoint(cls, domain, codomain=None, operator=..., *, traits=NONE):
    """Lift an operator from the base spaces. Claims no traits by default."""
```

The case where self-adjointness genuinely does survive — `A` and `M` both
diagonal in the same spectral basis, hence commuting — is picked up by the
specialisation protocol of §5.4 without anyone asserting anything. Any other
claim is the caller's, and `testing.check_traits` will catch it if it is
false.

---

## 4. Traits

Mathematical properties are traits; *representational* structure
(dense, sparse, diagonal, low-rank) stays a class, because it carries data and
extra API. That split is the rule.

```python
class Traits(Flag):
    NONE                  = 0
    SELF_ADJOINT          = auto()
    POSITIVE_SEMIDEFINITE = auto()   # implies SELF_ADJOINT (real convention)
    POSITIVE_DEFINITE     = auto()   # implies PSD and INVERTIBLE
    INVERTIBLE            = auto()
    ISOMETRY              = auto()   # A* A = I
    UNITARY               = auto()   # implies ISOMETRY and INVERTIBLE
    IDEMPOTENT            = auto()   # A @ A = A
```

`Traits.close()` adds implied traits, so no inconsistent state is
representable: `PD -> PSD | INVERTIBLE | SELF_ADJOINT`,
`UNITARY -> ISOMETRY | INVERTIBLE`, `IDEMPOTENT & SELF_ADJOINT -> PSD`.

### 4.1 Propagation rules

| expression | resulting traits |
|---|---|
| `A + B` | `A.traits & B.traits` for `SELF_ADJOINT`, `PSD`; `PD` if either is `PD` and the other `PSD` |
| `a * A`, `a > 0` | `SELF_ADJOINT`, `PSD`, `PD`, `INVERTIBLE` preserved |
| `a * A`, `a < 0` | `SELF_ADJOINT`, `INVERTIBLE` preserved; definiteness dropped |
| `A.adjoint` | `SELF_ADJOINT`, `PSD`, `PD`, `INVERTIBLE`, `UNITARY` preserved; `ISOMETRY` **not** (the adjoint of an isometry is a co-isometry) |
| `A @ B` | `ISOMETRY` if both; `UNITARY` if both; `INVERTIBLE` if both and square |
| `A.inverse` | `SELF_ADJOINT`, `PD`, `UNITARY`, `INVERTIBLE` preserved; bare `PSD` **not** |

Plus one general structural rule that subsumes the important special cases.
Flatten a composition to its factor list `[f_1, ..., f_n]`. Since
`(f_1 ... f_n)* = f_n* ... f_1*`, the composition is self-adjoint iff the list
is **adjoint-palindromic**: `f_i.adjoint == f_{n+1-i}` for all `i`. It is
additionally PSD if `n` is even, or if `n` is odd and the middle factor is PSD.
This single rule gives:

```python
L @ L.adjoint            # n=2, palindromic          -> SELF_ADJOINT | PSD
A @ C @ A.adjoint        # n=3, C self-adjoint PSD   -> SELF_ADJOINT | PSD
```

which is exactly the covariance-pushforward pattern, recognised structurally
rather than asserted by hand. It requires that `A.adjoint` be **memoised**, so
that `A.adjoint is A.adjoint`, and that `_Adjoint` compare structurally.

### 4.2 Traits are claims, not proofs

A user constructing `LinearOperator.self_adjoint(...)` is asserting something
the library cannot verify. `pygeoinf2.testing.check_traits(op, rng)` verifies
every claimed trait numerically and belongs in test suites, not in the hot path.

---

## 5. Operators

### 5.1 `Operator` — the nonlinear base

```python
class Operator[X, Y](ABC):
    domain: HilbertSpace[X]
    codomain: HilbertSpace[Y]

    # ---- the two evaluation paths --------------------------------------
    def __call__(self, x: X) -> Y:
        """Value only. The cheap path — a line search calls this and nothing else."""
        return self._value(x)

    def at(self, x: X) -> Linearisation[X, Y]:
        """Value AND derivative, sharing work where the operator can."""
        return self._linearise(x)

    def derivative(self, x: X) -> LinearOperator[X, Y]:
        return self.at(x).derivative

    def second_derivative(self, x: X, dx: X) -> LinearOperator[X, Y]:
        """F''(x)[dx, .], the second derivative curried on its first slot."""

    @property
    def has_derivative(self) -> bool: ...
    @property
    def has_second_derivative(self) -> bool: ...

    # ---- subclass contract ---------------------------------------------
    @abstractmethod
    def _value(self, x: X) -> Y: ...

    def _derivative(self, x: X) -> LinearOperator[X, Y]:
        raise NotImplementedError

    def _second_derivative(self, x: X, dx: X) -> LinearOperator[X, Y]:
        raise NotImplementedError

    def _linearise(self, x: X) -> Linearisation[X, Y]:
        """Override when one backend call yields both."""
        return Linearisation(x, self._value(x), self._derivative(x))

    # ---- algebra -------------------------------------------------------
    __add__ __sub__ __neg__ __mul__ __rmul__ __truediv__ __matmul__

    @classmethod
    def from_callables(cls, domain, codomain, value, *, derivative=None,
                       second_derivative=None, linearise=None) -> Operator
```

Three changes of substance:

- **Subclassing is the primary way to define an operator**, with
  `from_callables` for the quick path. v1 only ever injects callables, which is
  a C++ function-pointer idiom: it makes `MatrixLinearOperator` contort itself,
  and every traceback inside an operator names an anonymous closure.
- **`at()` is separate from `__call__`** precisely so the value-only path stays
  cheap. A line search performs many value-only evaluations and must not be
  charged for Jacobians it will discard.
- **Second derivatives are supported but optional**, in the curried form
  `F''(x)[dx, .]`. See §5.2.

```python
class MyPDEOperator(Operator):
    def _value(self, m):
        return observe(self._solve(m))

    def _linearise(self, m):
        u = self._solve(m)                     # ONE solve
        return Linearisation(m, observe(u), JacobianAt(u))
```

### 5.2 `Linearisation` and `QuadraticModel`

```python
@dataclass(frozen=True)
class Linearisation[X, Y]:
    point: X
    value: Y
    derivative: LinearOperator[X, Y]

    def as_affine(self) -> AffineOperator[X, Y]:
        """x |-> value + derivative(x - point)"""
```

#### Second derivatives

A second derivative of `F: X -> Y` is a symmetric bilinear map `X x X -> Y`.
**Curried on its first slot it lands back in `LinearOperator`**, which the
library already handles in full:

```
B(dx) := F''(x)[dx, .]  :  X -> Y        a LinearOperator
```

so there is nothing exotic to represent, and it is supported. What is *not*
available is a matrix representation — that would be an `n_y x n_x x n_x`
object nobody forms — but second derivatives are used matrix-free anyway.

It propagates through the algebra like everything else. For a composition,
differentiating the chain rule gives
`(F o G)''(x)[d1, d2] = F''(Gx)[G'd1, G'd2] + F'(Gx)[G''(x)[d1, d2]]`, i.e. in
curried form

```python
B_FG(d) = B_F(G_prime @ d) @ G_prime  +  F_prime @ B_G(d)
```

and it is available exactly when both factors have one.

What it buys is the exact Newton Hessian of a composed functional. For
`phi = psi(F(m))`:

```
phi''[d1, d2] = psi''[F'd1, F'd2] + <psi'(F m), F''(m)[d1, d2]>
```

so the Hessian operator is

```python
H @ dm = Fp.adjoint(H_psi(Fp(dm)))     # Gauss-Newton term
       + B(dm).adjoint(grad_psi)       # second-order-adjoint term
```

with `B(dm) = F.second_derivative(m, dm)` and `grad_psi` the gradient of `psi`
at `F(m)`. This is the standard second-order-adjoint / Newton-CG construction,
and supplying `B` is real work for the operator author — hence optional, in
exactly the way `derivative` is optional.

#### `QuadraticModel`

Scalar-valued maps get a second-order local model, because a Hessian *does*
have a natural representation as a self-adjoint `LinearOperator[X, X]`, and it
is what a Newton or trust-region step consumes:

```python
@dataclass(frozen=True)
class QuadraticModel[X](Linearisation[X, float]):
    point: X
    value: float
    derivative: LinearFunctional[X]              # PRIMITIVE — see §5.6
    hessian: LinearOperator[X, X] | None = None  # SELF_ADJOINT when present

    @cached_property
    def gradient(self) -> X:
        """The Riesz representer. Derived, via a Gram solve. See §5.6."""
        return self.derivative.adjoint(1.0)
```

One evaluation, everything the optimiser needs, shared work:

```python
model = phi.at(m)          # one call
f     = model.value
g     = model.gradient     # metric applied here, and only here
H     = model.hessian      # None if the functional does not carry one
```

Compare v1, where `NonLinearForm.__matmul__` builds a composed gradient that
calls `other(x)` and `other.derivative(x)` as two separate evaluations
(nonlinear_forms.py:331-336) — two PDE solves where one would do. That is the
concrete cost `at()` removes.

### 5.3 `LinearOperator`

```python
class LinearOperator[X, Y](Operator[X, Y]):
    traits: Traits

    @property
    def adjoint(self) -> LinearOperator[Y, X]:   # memoised; self if SELF_ADJOINT
        ...

    def _linearise(self, x) -> Linearisation:
        return Linearisation(x, self._value(x), self)   # derivative is self

    # ---- coordinate layer: requires CoordinateSpace on both sides ------
    def matrix(self, *, form: Literal["auto", "components", "galerkin"] = "auto",
               dense: bool = False, ...) -> np.ndarray | ScipyLinOp:
        """form="auto" selects "galerkin" when SELF_ADJOINT is claimed, so the
        returned matrix is symmetric and can be handed to a symmetric solver."""

    def diagonal(self, ...) -> np.ndarray

    # ---- factories -----------------------------------------------------
    @classmethod
    def from_callables(cls, domain, codomain, value, *, adjoint=None, traits=NONE)
    @classmethod
    def self_adjoint(cls, domain, value, *, traits=NONE)
    @classmethod
    def from_component_matrix(cls, domain, codomain, M, *, traits=NONE):
        """c_{Ax} = M c_x."""
    @classmethod
    def from_derivative_matrix(cls, domain, codomain, M, *, traits=NONE):
        """Row i of M holds the derivative components of the i-th output
        functional: M = G_Y A_c. The adjoint then applies G_X^{-1}
        automatically, which is what makes A* return representers."""
    @classmethod
    def from_tensor_product(cls, u, v)      # outer product x |-> (v, x) u
```

**Two matrix representations, precisely defined.** For `A: X -> Y` with Gram
matrices `G_X`, `G_Y`:

| form | matrix | characterisation |
|---|---|---|
| `"components"` | `A_c`, where `c_{Ax} = A_c c_x` | the change-of-coordinates matrix |
| `"galerkin"` | `G_Y A_c` | the matrix of the bilinear form `(A y, x)_Y`; symmetric iff `A` is self-adjoint |

The name "galerkin" is kept because it is the established term in the weak/FEM
setting, even though there is no dual space left to motivate it.

**`form="auto"` applies to extraction, never to construction.** `A.matrix()`
can pick its representation from `A.traits`, because the operator already knows
what it claims to be. `from_matrix` cannot: the caller is asserting which
representation their array is in, and no trait implies it — v1
(gaussian_measure.py:267-270) builds a rectangular covariance factor `L` with
`galerkin=True` and its inverse `Li` with `galerkin=False` in adjacent lines,
which is correct and not inferable. Hence the two explicitly named
constructors above rather than a `form=` argument on construction. See §12.2.

### 5.4 Expression nodes

Private, small, one job each:

```
_Identity  _Zero  _Scaled  _Sum  _Composition  _Adjoint  _Inverse
```

Each computes `traits` from its children by the rules of §4.1, defines
`adjoint` structurally (`_Composition([A, B]).adjoint == _Composition([B.adjoint, A.adjoint])`),
and gives a `repr` that names real objects. Simplifications are limited to the
obviously safe and locally decidable — `A.adjoint.adjoint is A`,
`1.0 * A is A`, flattening nested sums and compositions. **No simplification
engine**; that way lies a CAS.

#### The specialisation protocol

A generic node is the *fallback*, not the only outcome. Some operator families
are closed under the algebra and must stay in their class, or an expensive
representation and its functional calculus are silently thrown away. The
motivating case is `InvariantLinearAutomorphism` — diagonal in the spectral
basis — where the sum of two invariant operators is invariant with summed
eigenvalues, and the composition is invariant with multiplied eigenvalues.

Before building a node, the algebra asks both operands, in order:

```python
def __add__(self, other):
    for result in (self._combine_add(other), other._combine_radd(self)):
        if result is not None:
            return result
    return _Sum.of(self, other)          # generic fallback

# default on LinearOperator
def _combine_add(self, other) -> LinearOperator | None:
    return None
```

`InvariantLinearAutomorphism` then implements `_combine_add` and
`_combine_compose` alone, rather than overriding every dunder, and
`OrthogonalProjector.complement` returns a projector rather than a `_Sum` that
has forgotten it is idempotent.

This also fixes an asymmetry in v1. `LinearOperator.__add__` never returns
`NotImplemented` for another `LinearOperator`, so the right operand's
specialisation is never consulted: `invariant + invariant` preserves structure
(the subclass overrides `__add__`), but `generic + invariant` silently degrades
to an anonymous closure. Trying both operands makes the result independent of
the order of the arguments, which for a commutative operation it must be.

### 5.5 `Functional` and `AffineOperator`

Forms are **not removed, they are subsumed**: a form is a map into `Reals`, so
it is an `Operator` (or a `LinearOperator`) like any other, and inherits the
whole algebra rather than duplicating it. What v1 calls `NonLinearForm` is a
`Functional`; what it calls `LinearForm` is a `LinearFunctional`. The gain is
that `A.adjoint`, trait propagation, composition and `at()` are written once.

```python
class Functional[X](Operator[X, float]):
    codomain = Reals

    @property
    def has_derivative(self) -> bool: ...
    @property
    def has_hessian(self) -> bool: ...

    def at(self, x: X) -> QuadraticModel[X]: ...   # value + derivative + hessian
    def derivative(self, x: X) -> LinearFunctional[X]: ...  # PRIMITIVE
    def gradient(self, x: X) -> X: ...                      # DERIVED, via §5.6
    def hessian(self, x: X) -> LinearOperator[X, X]: ...    # SELF_ADJOINT

    # ---- hooks for the convex layer ------------------------------------
    def subgradient(self, x) -> X: ...
    def prox(self, x, t: float) -> X: ...
    def conjugate(self) -> Functional: ...

class LinearFunctional[X](LinearOperator[X, float], Functional[X]):
    @classmethod
    def from_derivative_components(cls, domain, g: np.ndarray) -> Self:
        """<f, x> = g . c_x. This is what an adjoint solve returns."""
    @classmethod
    def from_representer(cls, domain, v: X) -> Self:
        """<f, x> = (v, x)_X. This is a gradient you already hold."""

    @property
    def representer(self) -> X:
        return self.adjoint(1.0)
    # derivative is itself; hessian is the zero operator
```

Note there is deliberately **no bare `components=` constructor**. v1's
`LinearForm(domain, components=g)` does not say in its name which of the two
conventions `g` follows; you have to read `_mapping_impl` to find out. The two
named constructors above make the choice conscious and unmistakable, which
matters for the reason set out in §5.6.

#### What composition can and cannot propagate

Derivatives propagate through the algebra by the chain rule. Hessians do not,
and the design says so explicitly rather than quietly returning something
wrong.

| expression | derivative | hessian |
|---|---|---|
| `a*phi`, `phi + psi` | yes | yes |
| `phi @ A`, `A` linear | yes | yes: `A.adjoint @ H @ A` (palindromic, so PSD is preserved) |
| `phi @ F`, `F` nonlinear | yes: `F'(x)* d phi(F(x))` | yes **iff** `F.has_second_derivative` |

The last row is the one to understand. For a misfit `phi = psi(F(m))` the exact
Hessian is `J* H_psi J + sum_i r_i F_i''(m)`; the second term needs `F''`,
which an operator carries only if its author supplied it. So:

```python
H = (psi @ F).hessian(m)            # exact, iff F.has_second_derivative
gauss_newton_hessian(psi, F, m)     # J* H_psi J — the approximation, named
```

When `F` has no second derivative, `(psi @ F).has_hessian` is `False` and the
Gauss-Newton term is available only under a name that says what it is. v1
behaves correctly here already (`NonLinearForm.__matmul__` passes no `hessian`,
so `has_hessian` is `False`) but silently, and offers no route to the exact
Hessian at all. Naming the approximation makes the choice visible at the call
site, because Gauss-Newton versus Newton is a modelling decision, not an
implementation detail.

```python
class AffineOperator[X, Y](Operator[X, Y]):
    linear_part: LinearOperator[X, Y]
    translation: Y
    # _linearise returns a constant derivative
```

`AffineOperator` also fixes v1's `type(other).__name__ == "AffineOperator"`
string type-check in `LinearOperator.__add__`/`__sub__`/`__matmul__`, replaced
by ordinary `NotImplemented` / `__radd__` dispatch.

### 5.6 Derivatives, gradients, and the adjoint-method trap

This is a primary motivation for the library, so it gets stated explicitly
rather than left implicit in the types.

**The trap.** A standard numerical adjoint method returns an array

```
g_i = dJ / dm_i
```

the components of the **derivative** with respect to the chosen coefficient
basis. It is a covector: it eats a model perturbation and returns a number.
Common practice in the field is to hand that array straight to an optimiser and
step `m <- m - alpha * g`, treating it as an element of the model space. It is
not one. The **gradient** — the Riesz representer, the direction of steepest
ascent in the model metric — has components

```
c_grad = G^{-1} g
```

where `G` is the Gram (mass) matrix. Skipping `G^{-1}` gives a search direction
that is not steepest in any metric the modeller chose; it is an artefact of the
discretisation, and it changes when the mesh changes. On an orthonormal basis
`G = I` and the two coincide, which is why the error survives: it is invisible
in exactly the toy problems people test on.

**How the design handles it.** Riesz-identifying the spaces does not erase the
distinction; it relocates it from *two kinds of object* to *two ways of reading
one operator*. For a functional `f: X -> Reals`, `Reals` has unit Gram, so

| reading | what it is | metric applied? |
|---|---|---|
| `f.matrix()` | the derivative, the row vector `g` | no |
| `f.adjoint(1.0)` | the gradient, the Riesz representer | **yes** |

The adjoint is where the metric enters, and it is the *only* place it enters.
That is why §5.5 makes the derivative primitive and the gradient a derived,
cached property: the derivative is what the adjoint code produces, and the
gradient is what it costs a Gram solve to obtain. The cost is exactly the cost
of the correction, and it vanishes when `G = I`.

Three concrete requirements follow, and they are binding on the implementation:

1. **Named constructors, no ambiguous `components=`.**
   `LinearFunctional.from_derivative_components(X, g)` versus
   `from_representer(X, v)`. Naming is the mechanism; a user cannot supply one
   while meaning the other without writing the wrong word.
2. **`Functional` is built from a derivative, not a gradient.** A `gradient=`
   entry point may exist for people who genuinely hold one, but the derivative
   is the documented and default route, because it is what adjoint codes emit.
3. **`testing.check_gradient(phi, x, rng)`** finite-differences `phi` along
   random directions and compares against `(grad, dx)_X`. If a derivative array
   has been supplied as a gradient, this fails by exactly a factor of `G` — so
   the classic error becomes a test failure rather than a slow, mesh-dependent
   convergence mystery.

```python
# what an adjoint solve gives you
g = my_adjoint_solve(m)                              # ndarray, dJ/dm_i

# the right thing to build
d = LinearFunctional.from_derivative_components(X, g)
grad = d.adjoint(1.0)                                # G^{-1} g, the gradient

# and the wrong thing is now hard to write by accident
grad_wrong = X.from_components(g)                    # a deliberate act
```

#### The coordinate axiom that makes this work

`CoordinateSpace` must guarantee

```
<f, x>  ==  f.matrix() . X.to_components(x)
```

for every functional `f` and vector `x`. This is not decoration: v1's scipy
line-search bridge depends on it, and says so
(nonlinear_optimisation.py:170-176) — the derivative components are passed to
`scipy.optimize.line_search` precisely *because* SciPy's Euclidean
`dot(gc, pc)` then evaluates the correct directional derivative. Hand it the
gradient components instead and the Wolfe conditions are tested against
`c_g . c_p` rather than `c_g^T G c_p`, i.e. against the wrong slope.
`testing.check_coordinates` asserts this identity.

#### One nuance, so the claim is not overstated

Handing a *consistent* (value, derivative-components) pair to a purely
coordinate-space optimiser such as `scipy.optimize.minimize` is **correct**,
not a bug. It is a reparameterisation: the same minimiser, reached by a
different path. What it is not is metric-aware — the path and the conditioning
depend on the discretisation. The error is *mixing* the two conventions, and
the trap is using `g` as a direction in the model space.

#### The same thing at operator level

An observation operator is a stack of functionals, and the distinction is
identical. In v1 this is what `galerkin=True` means on `from_matrix`
(linear_operators.py:1069-1079): the matrix maps domain components to codomain
*dual* components, so the adjoint applies `G^{-1}` and returns representers.
`point_evaluation_operator` (symmetric_space.py:2455) relies on exactly that —
its rows are the Dirac derivative components, and `galerkin=True` is what makes
`A*` produce Dirac *representers* rather than raw component arrays.

In v2 that becomes a named constructor rather than a boolean, mirroring the
functional case:

```python
LinearOperator.from_derivative_matrix(X, Y, M)   # rows are derivative components
LinearOperator.from_component_matrix(X, Y, M)    # c_y = M c_x
```

The same argument applies verbatim to `Operator.derivative` and its adjoint:
`F'(m)*` maps a covector on the data space to a covector on the model space,
and it is the space's metric, carried in `adjoint`, that makes the result a
model-space object rather than an array of partial derivatives.

#### This is already v1 practice

Every `LinearForm` in the concrete spaces is built by this idiom, deliberately
and correctly — build from derivative components, then map to the representer:

| v1 | derivative | representer |
|---|---|---|
| Dirac | `dirac(point)` | `dirac_representation(point)` |
| geodesic integral | `geodesic_integral(...)` | `geodesic_integral_representation(...)` |
| geodesic ball integral | `geodesic_ball_integral(...)` | via `from_dual` |

v2 renames the two halves rather than changing the mathematics:
`from_derivative_components(...)` and `.adjoint(1.0)`.

#### A bonus: functionals stop being dense by construction

v1's `LinearForm(domain, mapping=...)` computes **all `dim` components eagerly
in `__init__`** (linear_forms.py:212-245), looping over the whole basis. Every
functional is therefore dense-by-construction and costs `n` evaluations to
build, even one that is trivial to apply. Because a v2 functional is just an
operator, it is matrix-free by default and `.matrix()` is computed only when
something actually asks for it.

### 5.7 Functional calculus

Not in the first draft, and it should have been: v1 uses it heavily and the
core has to accommodate it.

- `InvariantLinearAutomorphism` carries a complete calculus — `inverse`,
  `sqrt`, `exp`, `log`, `apply_function`, `__pow__`, `__abs__` — evaluated
  directly on its eigenvalues.
- `LanczosOperatorFunction` (functional_calculus.py) provides matrix-free
  `f(A)` for a general self-adjoint operator via Lanczos, and is used for
  covariance square roots, log-determinants and stochastic trace estimation in
  `linear_bayesian.py`.

So `f(A)` is a core capability, and it is exactly the kind of thing traits
should gate:

```python
def operator_function(A: LinearOperator, f, *, method="auto") -> LinearOperator:
    """f(A) for self-adjoint A. Requires SELF_ADJOINT; sqrt and log require PSD."""
```

Dispatch by structure, not by hand: an operator that is diagonal in some basis
evaluates `f` on its eigenvalues; anything else self-adjoint goes to Lanczos.
This is the specialisation protocol of §5.4 applied to a unary operation, and
it is why "diagonal in a basis" is a *class* (it carries eigenvalues) while
"self-adjoint" is a *trait* (it carries no data).

Trait propagation for `f(A)`: `SELF_ADJOINT` always; `PSD` when `f >= 0` on the
spectrum; `POSITIVE_DEFINITE` when `f > 0`. Since the library cannot inspect
`f`, these are declared by the caller for the named cases (`sqrt`, `exp`, and
`inverse` of a PD operator all preserve positive-definiteness) and left empty
otherwise.

---

## 6. `LinearSolver`

```python
class LinearSolver(ABC):
    requires: ClassVar[Traits] = Traits.NONE
    requires_coordinates: ClassVar[bool] = False

    def __call__(self, operator: LinearOperator) -> LinearOperator:
        """Return the inverse AS AN OPERATOR. Validates first."""
        self._validate(operator)
        return self._invert(operator)

    def _validate(self, operator) -> None:
        # square; traits superset of `requires`; coordinates if required
        ...
```

Returning the inverse as a `LinearOperator` is the one piece of the v1 solver
API that is straightforwardly right, and it is kept. What changes:

- **Solvers are stateless.** v1 stores `self._iterations` on the solver, so a
  solver object cannot be shared or reused concurrently and the count belongs
  to whichever solve ran last. Iteration counts and residual histories move
  onto the returned `_Inverse` operator (`.last_solve`) or onto explicit
  callback objects.
- **Preconditions are declared and checked**, e.g. `CGSolver.requires =
  SELF_ADJOINT | POSITIVE_DEFINITE`, instead of `assert
  operator.is_automorphism` (which vanishes under `python -O`).
- **The iterative solvers are coordinate-free.** CG, MINRES, BiCGStab and LSQR
  are implemented directly against `inner_product` / `axpy`, so they run
  against a PETSc space with no component map. Only `LUSolver`,
  `CholeskySolver`, `EigenSolver` and the scipy-matrix wrappers set
  `requires_coordinates = True`.
- **Non-square is a different operation.** `LeastSquaresSolver` is a sibling
  ABC returning a pseudo-inverse, rather than `LSQRSolver` sitting under an
  `IterativeLinearSolver` base whose `__call__` asserts `is_automorphism`.
- Result traits propagate: the inverse of an SPD operator is SPD.

---

## 7. Measures

```python
class ProbabilityMeasure[X](ABC):
    domain: HilbertSpace[X]

    @abstractmethod
    def sample(self, rng: Generator | None = None) -> X: ...
    def samples(self, n: int, rng=None) -> list[X]: ...

    @property
    def expectation(self) -> X | None: ...
    @property
    def covariance(self) -> LinearOperator[X, X] | None: ...   # SELF_ADJOINT|PSD

    # optional, for the nonlinear groundwork
    def log_density(self, x: X) -> float: ...          # up to an additive constant
    def grad_log_density(self, x: X) -> X: ...         # a VECTOR, via Riesz

    def push_forward(self, op: Operator[X, Z]) -> ProbabilityMeasure[Z]: ...
    def __rmatmul__(self, op):  return self.push_forward(op)   # A @ mu
```

`grad_log_density` returning a vector in `X` rather than a functional is a
direct dividend of decision 1, and it is what MALA/HMC and any gradient-based
posterior exploration need.

`push_forward` is closed for `(Gaussian, affine)` and returns a
`PushForwardMeasure` otherwise — which can still be *sampled* (draw from `mu`,
apply `F`) even with no closed-form density. That is the minimum groundwork for
nonlinear inference.

```python
class GaussianMeasure[X](ProbabilityMeasure[X]):
    def __init__(self, domain, *, expectation=None,
                 covariance=None,          # SELF_ADJOINT | PSD, validated
                 covariance_factor=None,   # L with C = L @ L.adjoint
                 precision=None,
                 precision_factor=None): ...
```

`covariance` is validated against its claimed traits at construction rather
than trusted. `A @ mu` for affine `A` stays Gaussian, with the pushed-forward
covariance `A @ C @ A.adjoint` recognised as PSD by the palindrome rule of
§4.1 with no special-casing.

**RNG**: every sampling entry point takes `rng: Generator | None`, defaulting
to a module-level `default_rng()` that can be seeded. v1 calls
`np.random.randn` against the legacy global state, so no result involving
sampling is reproducible.

**Measures need the specialisation protocol of §5.4 too.**
`InvariantGaussianMeasure` and `CorrelatedInvariantGaussianMeasure` each
override `affine_mapping`, `__add__`, `__mul__`, `kl_divergence`,
`nuclear_norm` and `hilbert_schmidt_norm` so that the result stays spectral —
because a spectral measure that degrades to a generic one loses the closed-form
KL divergence and norms and falls back to randomised estimation. So
`ProbabilityMeasure` gets the same `_combine_*` hooks, and for the same reason:
structure is expensive to acquire and must not be dropped by an arithmetic
operation.

`CorrelatedInvariantGaussianMeasure` also lives on a direct sum of field spaces
with spectral cross-covariances, which is the main consumer of `DirectSum` and
the reason it is in scope (§3.3).

---

## 8. `pygeoinf2.testing`

v1 mixes the axiom checks into the production classes:
`class HilbertSpace(ABC, HilbertSpaceAxiomChecks)`,
`class LinearOperator(NonLinearOperator, LinearOperatorAxiomChecks)`. The
checks are good and worth keeping — they belong in a testing module, not in
the MRO of every space and operator.

```python
check_space(space, rng, n=10)      # vector-space axioms, inner product, in-place ops
check_coordinates(space, rng)      # round trip, Gram consistency, white noise covariance
check_operator(op, rng)            # linearity, adjoint identity, derivative by finite differences
check_traits(op, rng)              # every CLAIMED trait, verified numerically
check_gradient(phi, x, rng)        # THE adjoint-method trap check — see §5.6
check_hessian(phi, x, rng)         # second derivative by finite differences
check_measure(mu, rng, n)          # sample moments against expectation/covariance
```

`check_traits` is the safety net for the fact that traits are user assertions.
`check_gradient` is the safety net for §5.6: it finite-differences the
functional and compares against `(grad, dx)_X`, so a derivative array supplied
as a gradient fails by a factor of `G` instead of quietly producing
mesh-dependent convergence.

---

## 9. v1 defects this design removes by construction

**Unhashable spaces.** v1 declares `__eq__` abstract on `HilbertSpace` and no
subclass defines `__hash__`, so Python sets `__hash__ = None` throughout.
Confirmed: `{pygeoinf.EuclideanSpace(3): 1}` raises `TypeError: unhashable
type`. Fixed by deriving both from `_key()`.

**White noise is not white on non-orthonormal spaces.** `HilbertSpace.random`
returns `from_components(randn(dim))`. If `x = Phi c` with `c ~ N(0, I)` then
`E[(x,u)(x,v)] = c_u^T G^2 c_v`, whereas the identity covariance requires
`c_u^T G c_v`. True white noise needs `c ~ N(0, G^{-1})`. Measured on a
`Sobolev` space on the circle (`Sobolev(16, 2.0, 0.2)`, `u` the 3rd basis
vector, 40k samples):

```
E[(x,u)^2] = 13.74        (u,u) = 3.70        ratio 3.71 ~= (u,u)
```

i.e. the covariance is `G`, not `I`. This propagates into `white_noise_measure`
(whose docstring claims the opposite), and from there into `random_range`,
`LowRankSVD`, `random_trace`, `random_diagonal` and the low-rank Gaussian
approximations, on every mass-weighted space — which is all checksthe workhorse
Sobolev spaces on the sphere, circle, torus and plane.
`SymmetricGaussianMeasure` compensates separately with a
`1/sqrt(metric_values)` factor, so the fully symmetric path is unaffected; the
generic randomised-algebra path is not.

*This is a live v1 bug, independent of the refactor, and worth fixing in 1.x.*

In 2.0 `white_noise` is defined on `CoordinateSpace` as `c ~ N(0, G^{-1})`
using `gram_solver`, and `check_coordinates` verifies the sample covariance
against the inner product.

*Update after reading the call sites:* this is worse than it first appeared.
`random_range` documents a "Geometric Safety Guard" (low_rank.py:515-527) that
*deliberately* routes mass-weighted codomains through `white_noise_measure`,
on the stated grounds that Euclidean QR "cannot guarantee geometric
orthogonality" there. So the fallback that exists specifically to respect the
mass matrix is defeated on precisely the spaces it exists for.

**Generic mass-weighted space equality is identity-based.**
`MassWeightedHilbertSpace.__eq__` (hilbert_space.py:837) compares
`self.mass_operator == other.mass_operator`, but `LinearOperator` defines no
`__eq__`, so that comparison is object identity. Two mathematically identical
mass-weighted spaces constructed separately therefore compare **unequal**, and
any subsequent operator construction raises a spurious "Domain mismatch".
Verified:

```
MassWeightedHilbertSpace(L, M, M)  ==  MassWeightedHilbertSpace(L, M, M)         -> True
MassWeightedHilbertSpace(L, M, M)  ==  MassWeightedHilbertSpace(L, I(), I())     -> False
```

with `M` and `I()` both the identity on `L`. The concrete Sobolev spaces escape
this only because they override `__eq__` with `(underlying_space, order,
scale)`. The implication for §3.1: **a space whose identity depends on an
operator must supply its own `_key()`**, since operator equality is and should
remain identity-based. `testing.check_space` asserts that a space equals — and
hashes equal to — an independently constructed copy of itself.

---

## 10. Concept map, v1 to v2

| v1 | v2 |
|---|---|
| `HilbertSpace.to_dual` / `from_dual` / `.dual` | *removed* (decision 1) |
| `DualHilbertSpace` | *removed* |
| `LinearForm` | `LinearFunctional = LinearOperator[X, Reals]`, `.representer = adjoint(1.0)` — subsumed, not removed |
| `NonLinearForm` | `Functional`; `.at(x)` returns a `QuadraticModel` |
| `LinearForm(domain, components=g)` | `LinearFunctional.from_derivative_components` / `.from_representer` — convention named, not inferred |
| `form.gradient` supplied directly | derivative supplied; gradient derived via `adjoint` (§5.6) |
| `LinearOperator.dual`, `dual_mapping` | *removed*; `.adjoint` only |
| `LinearOperator.self_dual` | *removed* |
| `matrix(galerkin=True)` | `matrix(form="galerkin")`, `form="auto"` from traits |
| `MassWeightedHilbertSpace` | `MassWeightedSpace(base, mass, mass_solver)` — coordinate-free (§3.5) |
| `OrthogonalHilbertSpace` | `DiagonalMetricSpace` |
| `OrthonormalHilbertSpace` | `OrthonormalSpace` |
| `from_formal_adjoint` | **kept** (§3.5); `M^{-1}` now derived from `mass_solver`, direct-sum recursion handled by `DirectSum` |
| `from_formally_self_adjoint` | folded into `from_formal_adjoint` as a default codomain (§3.5); the name promised a self-adjointness it never provided |
| `NonLinearOperator` | `Operator` |
| `space.zero` (property) | `space.zero()` (method) |
| `space.ax` / `space.axpy` | `scale_inplace` / `axpy` |
| `space.multiply(a, x)` | `space.scale(a, x)` |
| `space.is_element` | *removed*; `pygeoinf2.testing` |
| `checks/` mixins in the MRO | `pygeoinf2.testing` functions |
| `IterativeLinearSolver.iterations` | `inverse.last_solve.iterations` |
| `HilbertSpace.random` | `random(rng)` **and** `white_noise(rng)`, separated |

---

## 11. Milestones

| # | Deliverable | Acceptance |
|---|---|---|
| M0 | **done** — `traits.py`, `algebra/spaces.py`, `testing.py` | met; 67 tests, see §11.1 |
| M1 | **done** — `operators.py`, `nodes.py`, `linearisation.py`, `Functional`, `AffineOperator` | met; 135 tests, see §11.2 |
| M2 | **done** — `compat.py` adapter for v1 spaces | met; 161 tests, see §11.4 |
| M3 | **done** — `numerics/solvers.py`, `numerics/preconditioners.py` | met; 204 tests, see §11.5 |
| M4 | **done** — `probability/` | met; 242 tests, see §11.5 |
| M5 | Linear inversion rebuilt on the new core | Parity harness: v1 and v2 side by side on the existing test problems |

The packaging decision is what makes M5 cheap — `import pygeoinf as gi` and
`import pygeoinf2 as gi2` in the same process, same problem, compared
numerically.

### 11.1 M0 as built

```
pygeoinf2/
  __init__.py          curated exports
  traits.py            Traits, close(), and the propagation rules of §4.1
  algebra/spaces.py    HilbertSpace, CoordinateSpace, DiagonalMetricSpace,
                       OrthonormalSpace, EuclideanSpace, Reals, ArrayVectorMixin
  testing.py           check_space, check_coordinates, check_representer,
                       check_white_noise
  tests/               67 tests
```

Acceptance met: `check_space` and `check_coordinates` pass on `EuclideanSpace`,
`Reals`, a diagonal-metric space and a full-Gram space. The v1 suite is
unaffected (971 passed, 1 xfailed).

Seven deviations from the design above, all deliberate:

1. **The Gram is not a `LinearOperator`.** It is exposed as `apply_gram` /
   `solve_gram` on component arrays, plus `gram_matrix()` for those that want
   it explicitly. This avoids M0 depending on M1, and it is more honest: the
   Gram acts on components, which is precisely why it lives in the coordinate
   layer.
2. **`white_noise_components(rng)` replaces the speculative
   `gram_factor_apply` / `gram_factor_solve`.** The correct draw is
   `c = L^-T xi` with `G = L L^T`, not `L^-1 xi`. The two coincide for the
   symmetric factors of the diagonal and orthonormal cases, so a method named
   "factor solve" would have concealed the distinction until the first
   non-symmetric factor. One unambiguous primitive instead.
3. **The in-place contract is explicit.** `axpy` and `scale_inplace` return
   their result, and for an immutable backend such as `Reals` that result is a
   new object. Callers must use the return value. Documented on
   `HilbertSpace`, and `Reals` is in the test suite specifically to keep the
   contract honest.
4. **`ArrayVectorMixin` was added.** It supplies `copy`, `axpy` and
   `scale_inplace` for array-backed spaces, so a concrete coordinate space
   needs only `dim`, `_key`, `to_components` and `from_components`. Without it
   the seven abstract methods were a real burden for the common case.
5. **`CoordinateSpace.representer(derivative_components)` was added.** The
   §5.6 operation is expressible in pure coordinate terms, so it does not have
   to wait for `LinearFunctional`, and `check_representer` can verify the
   pairing axiom at M0 rather than M1.
6. **`random` is a capability, not an abstract method.** It raises
   `NotImplementedError` with a message naming what needs it;
   `CoordinateSpace` supplies it. A coordinate-free space is not obliged to
   provide randomness it may not have.
7. **Two closure rules were added to §4.1**: `PSD & INVERTIBLE -> PD` and
   `ISOMETRY & INVERTIBLE -> UNITARY`. Both are needed to keep `close()`
   idempotent and to make `inverse_traits` behave.

Deferred to M1 because they need `LinearOperator`: `MassWeightedSpace` (§3.5),
`HilbertSpace.identity()` and `zero_operator()`, and the structural
adjoint-palindrome detection. The *trait* half of the palindrome rule is
already present as `congruence_traits` and `gramian_traits`, and is tested
against the Bayesian normal operator `A Q A* + R`.

### 11.2 M1 as built

```
pygeoinf2/algebra/
  linearisation.py   Linearisation, QuadraticModel
  operators.py       Operator, LinearOperator, Functional, LinearFunctional,
                     AffineOperator, require_coordinates
  nodes.py           _Identity, _Zero, _Adjoint, _Scaled, _Sum, _Composition
                     and their nonlinear counterparts
pygeoinf2/tests/
  doubles.py         Opaque, OpaqueSpace, StrictSpace
```

Acceptance met: `check_operator` and `check_traits` pass, and
`A @ C @ A.adjoint` — built in two steps, so the pattern only exists once the
composition is complete — reports `SELF_ADJOINT | POSITIVE_SEMIDEFINITE`.
`A @ Q @ A.adjoint + R` reports `POSITIVE_DEFINITE`. 135 tests.

**One finding worth recording.** Memoising `adjoint` is not enough on its own.
A composed operator's adjoint is built as a *new* node
(`(A B)* == B* A*`), so `(A @ B).adjoint.adjoint` was a different object from
`A @ B` — and since the palindrome rule compares factors by identity, it
silently stopped firing for any operator that had been through the algebra.
`_Sum._make_adjoint` and `_Composition._make_adjoint` therefore close the loop
explicitly, caching `self` as the new node's adjoint. Caught by
`check_operator`, which asserts the involution.

Deviations:

1. **`Functional.__init__` takes an optional codomain.** `LinearFunctional`
   inherits from both `LinearOperator` and `Functional`, so
   `LinearOperator.__init__`'s cooperative `super()` call lands on
   `Functional.__init__`. The argument is accepted, validated to be `Reals`,
   and otherwise ignored.
2. **`matrix()` is dense only.** The matrix-free SciPy wrapper is deferred to
   M3, where the solvers are what actually want it.
3. **`with_traits`** was added, for attaching a claim to an operator built
   elsewhere without rebuilding it.
4. `_Inverse` is deferred to M3: it comes from a solver, not from the algebra.

### 11.3 M2 as built

`pygeoinf2/compat.py`, plus `tests/test_compat.py`. Circle and sphere Sobolev
spaces, point evaluation, Dirac functionals and invariant covariances all run
against the v2 core, with results checked against v1 where v1 is right and
against the mathematics where it is not. 161 tests.

**This module is scaffolding with an expiry date.** The end state is that v1 is
gone and every concrete space is written natively against the v2 core — the
adapter exists so that the core is exercised on real problems while it is still
cheap to change, not so that v1 survives inside v2. Nothing else in `pygeoinf2`
imports it, and nothing should come to depend on it. When the concrete spaces
are rewritten, `compat.py` and `tests/test_compat.py` are deleted outright.

The adapter is short, because the two designs agree on more than they differ:

| v1 | v2 |
|---|---|
| `to_components` | `to_components` |
| `to_dual`, then its components | `apply_gram` |
| `from_dual` | `solve_gram` |
| `inner_product` | delegated, not rederived |
| `operator.adjoint` | `adjoint` |
| `LinearForm.components` | `from_derivative_components` |
| `space.random` | `random`, but **not** `white_noise` |

**The mass matrix was already there under another name.** v1's duality pairing
is `<xp, x> == dot(dual.to_components(xp), to_components(x))` and its inner
product is `(x, y) == <to_dual(x), y>`, so the components of `to_dual(x)` are
by definition `G c_x`. No introspection of the space is needed to recover the
Gram matrix; `to_dual` *is* the Gram apply. That single observation is most of
the adapter.

Three things that needed care:

1. **v1 spaces are unhashable**, so they cannot be a `_key` directly. A
   `_V1SpaceKey` wrapper compares exactly, by delegating to v1's `__eq__`, and
   hashes coarsely on the dimension. That satisfies the hash contract — equal
   objects hash equally, and equal spaces necessarily share a dimension —
   while distinct spaces of the same dimension merely collide, which costs
   nothing when a program holds a handful of spaces.
2. **`white_noise` is deliberately not delegated.** Delegating would import the
   defect of §9. `test_v1_white_noise_is_not` pins the failure quantitatively:
   on a circle Sobolev space, v1 gives `E[(x,u)^2] == (u,u)^2` where the
   identity covariance requires `(u,u)`. The adapted space passes
   `check_white_noise`.
3. **Forming the Gram matrix is not an option on a real space** — a Sobolev
   space on the sphere at degree 128 has `dim` above 16000, so an `O(dim^2)`
   build is out. The adapter instead *probes* for diagonality: it computes
   `G 1`, then checks `G s == (G 1) * s` for two random `s`. If `G` is diagonal
   this holds identically; if it is not, the probes would have to lie in the
   kernel of `G - diag(G 1)`, which for random vectors has probability zero.
   Two Gram applications instead of `dim`. The strategy is overridable with
   `gram="diagonal" | "dense"`.

The end-to-end test computes a posterior mean two ways — assembled with the v2
algebra, and assembled with v1's — and agrees to `1e-10`. The v2 side never
mentions a Galerkin flag: `normal.matrix()` picks its representation from the
traits, and the result is symmetric positive definite.

### 11.4 M3 as built

```
pygeoinf2/numerics/
  solvers.py          LinearSolver, InverseOperator, SolveResult,
                      CG, MINRES, BiCGStab (coordinate-free),
                      LU, Cholesky, Eigen (coordinate-requiring),
                      LeastSquaresSolver + LSQR
  preconditioners.py  Identity, Jacobi
```

Acceptance met: coordinate-free CG on an adapted circle Sobolev space agrees
with v1's own `CGSolver` to `1e-8`, and with a dense solve to `1e-9`. 204 tests.

**The coordinate split is real, and tested as such.** `StrictSpace` raises if a
component map is touched, and CG, MINRES, BiCGStab and LSQR all solve against
it; a negative control confirms `CholeskySolver` *does* trip the guard. So the
claim that the iterative solvers run against a PETSc-backed space is now
mechanically checked rather than asserted.

**Preconditions are declared and validated, and both halves matter.**
`CGSolver.requires` is `POSITIVE_DEFINITE`, so an operator that has not earned
the trait is refused at `__call__` with a message naming what is missing.
Separately, CG detects a *false* claim during the solve — a non-positive
curvature direction — and its error points at `testing.check_traits`. v1 has
neither check: it writes `assert operator.is_automorphism`, which says nothing
about symmetry or definiteness and is deleted by `python -O`.

**Solvers are stateless.** Diagnostics come back from
`InverseOperator.solve(y)` as a `SolveResult`, so one solver instance can serve
many operators. v1 keeps `iterations` on the solver, where it belongs to
whichever solve ran last.

Two bugs found while writing this, both caught by testing against dense linear
algebra rather than by inspection:

1. **The default iteration cap was `dim`.** Krylov methods terminate within
   `dim` steps in exact arithmetic, but rounding routinely costs an extra step,
   so MINRES failed on an indefinite system that it had in fact solved. The
   default is now `max(2 dim, 20)`.
2. **LSQR discarded a sign.** `np.hypot(rho_bar, 0)` returns `|rho_bar|`, but
   the recurrence carries `rho_bar` signed when there is no damping. The
   iteration converged confidently to the wrong point — error `7e-1` while
   reporting success. Damping now takes its own branch, and the undamped path
   uses the signed value. Verified against `numpy.linalg.lstsq` for
   over-determined, under-determined and square systems, and against the
   normal equations for the damped case.

Deviations from §6: `InverseOperator` does not expose `last_solve`. Recording
the most recent solve on the operator would have made "stateless" only half
true; `solve()` returns the diagnostics with the answer instead.

Still open for M3's original brief: a matrix-free SciPy wrapper for `matrix()`,
which nothing yet needs, and the **petsc4py** question of §11.4. The doubles
have carried the coordinate-free claim further than expected — every Krylov
method is verified against a space that refuses coordinates — so a real backend
is now about ergonomics and distributed vectors rather than about correctness.

### 11.5 M4 as built

```
pygeoinf2/probability/
  base.py      ProbabilityMeasure, PushForwardMeasure
  gaussian.py  GaussianMeasure
```

Acceptance met: sampled moments agree with the declared covariance, the
pushforward under a real point-evaluation operator is verified, and v1's own
`InvariantGaussianMeasure` and v2's measure agree on the same second moments
to sampling tolerance. 242 tests.

**The base class asks for very little**: a way to draw a sample. An
expectation, a covariance, a log density and its gradient are all optional. A
measure that can only be sampled is still a measure, and that is what nonlinear
inference actually presents.

**`grad_log_density` returns a vector.** For a Gaussian it is `-P (x - m)`,
and no Riesz map appears anywhere — the precision maps the space to itself, so
its output already *is* a vector. That is section 5.6 in its most agreeable
form, and it is what MALA and HMC step along. A finite-difference test checks
it against the log density, which would catch a metric applied in the wrong
place.

**The covariance carries its structure for free.** Building from a factor gives
`L L*`, which the palindrome rule recognises as self-adjoint and positive
semidefinite — positive *definite* when `L` is invertible — with nothing
claimed. The pushforward `A C A*` is likewise recognised. A covariance passed
in directly must claim those traits, and the error names `check_traits` as the
remedy; that is a cheap structural check, with numerical verification left to
the testing module where it belongs.

**Sampling is where the white-noise correction pays.** A draw is
`m + L xi` with `xi` white noise **on the factor's own domain**. For the
isotropic construction that domain is the space itself, so the noise must be
white with respect to the space's inner product. `from_standard_deviation` on a
mass-weighted space is exactly where v1 produces `sigma^2 G` instead of
`sigma^2 I`.

One thing the first draft of §7 missed: **derived measures must stay
samplable**. The sum of two Gaussians has a covariance but no factor, and the
pushforward under an operator has a factor only if the base did — so both
initially came out unsamplable. Both now carry a composed sampler instead
(draw from each and add; draw and map), which is what v1 does and is obviously
right in hindsight. Found by `check_measure`, not by reading the code.

A tolerance bug in `check_measure` was worth fixing carefully: an off-diagonal
covariance entry that is zero in expectation was being judged against an
absolute floor of 1, while the diagonal entries were of order 36, so ordinary
sampling noise failed the check. Tolerances are now scaled by
`sqrt(C_ii C_jj)`, which is the actual standard error of the estimator.

### 11.6 Direct sums, built out of order

Built ahead of M5, because the joint structure the nonlinear work needs lives
here rather than in the inversion layer. 282 tests.

```
pygeoinf2/algebra/direct_sum.py   DirectSum, BlockOperator/BlockLinearOperator,
                                  Column/Row/BlockDiagonal and their linear forms
pygeoinf2/probability/base.py     ProductMeasure, product()
```

**The joint model is already the nonlinear bridge.** v1 builds the joint law of
model and data (forward_problem.py:209) not by assembling a covariance but as a
pushforward of a product measure through a block operator:

```
mu    = model_measure (x) noise_measure       on X (+) Y
op    = [[I, 0], [A, I]]
joint = op @ mu                               the law of (m, A m + e)
```

Putting a nonlinear `F` where `A` is gives the law of `(m, F(m) + e)`, which
`op @ mu` returns as a `PushForwardMeasure` — samplable, no closed density,
which is prior predictive sampling. And `op.derivative(x)` is
`[[I, 0], [F'(m), I]]`, so the linearised joint model comes from the same
object rather than from a separate construction. That is why block operators
are nonlinear by default here, with the linear case dispatched to as a
specialisation.

**One gap in v1 closed.** The product measure existed only for Gaussians
(`GaussianMeasure.from_direct_sum`), but the independent product of *any*
samplable measures is samplable — which is precisely the case a non-Gaussian
prior presents. `product()` collapses to a Gaussian when every factor is one
and returns a `ProductMeasure` otherwise.

Decisions taken:

- **Vectors are tuples**, with optional labels on the space. The container is
  fixed-length and should not be restructured; the components stay mutable so
  `axpy` still updates in place. Labels give named access to the `(model,
  data)` split without the vector wrapper rejected everywhere else.
- **Labels are not part of a space's identity.** Two sums over the same
  summands are the same space whatever their parts are called. Making labels
  structural put a block operator — which cannot know what its user chose to
  call things — on a space that compared unequal to the one its own vectors
  came from. Found immediately, by `product()` failing to construct.
- **`DirectSum` dispatches on construction** to a coordinate-providing subclass
  when every summand provides coordinates, so `isinstance(X, CoordinateSpace)`
  stays a reliable answer. A sum containing one coordinate-free summand is
  itself coordinate-free, which is the honest answer and the one
  `require_coordinates` depends on.

**The M1 lesson recurred, in a new place.** `projection(i)` rebuilt its
operator on every call, so `P.adjoint is inclusion(i)` never held — and since
the palindrome rule compares factors by identity, `P @ C @ P.adjoint` was not
recognised as a congruence. Projections are now memoised per index. Anything
that participates in the algebra and can be *rebuilt* needs this; it is the
third time it has come up.

### 11.7 Numerics as built

The three areas of §2.1, ported and improved. 449 tests.

```
pygeoinf2/algebra/diagonal.py             DiagonalLinearOperator
pygeoinf2/numerics/functional_calculus.py Lanczos, f(A), quadratic forms
pygeoinf2/numerics/randomised.py          range finding, low rank, estimators
pygeoinf2/numerics/line_search.py         Armijo, strong Wolfe
pygeoinf2/numerics/optimisation.py        descent methods, Newton, trust region
```

**Functional calculus** was already fully coordinate-free in v1 — zero
component uses in 509 lines — so the port added trait gating and the §5.7
dispatch rather than restructuring. `DiagonalLinearOperator` is the v2 home for
`InvariantLinearAutomorphism`: eigenvalues stored, algebra closed through the
specialisation protocol, calculus exact. It claims self-adjointness only when
the space's metric is also diagonal, since `diag(d)` commutes with a general
Gram matrix only if that matrix is diagonal too — hence
`CoordinateSpace.has_diagonal_metric`. A test asserts the diagonal and Lanczos
paths agree, so the dispatch is an optimisation and not a different answer.

**Randomised linear algebra** kept v1's coordinate-free structure and fixed the
distribution feeding it. `random_svd` avoids forming `Q* A` by assembling the
`k x k` Gram matrix `C_ij == (A* q_i, A* q_j)` from inner products alone. Only
`random_diagonal` needs components. Factors are built with
`LinearOperator.from_vectors`, which claims `ISOMETRY` for an orthonormal
family — so `U D U*` is recognised as PSD by the palindrome rule, and a
`LowRankCholesky` factor drops straight into `GaussianMeasure`.

**Optimisation** is written rather than wrapped, and §5.6 is why. Verified in
v1: the `jac` handed to SciPy is `G^-1 dJ/dc` — the ratio to the true
derivative is exactly the Gram diagonal — while the `hess` is the Galerkin
matrix and so correct. The two are in different conventions in one call, and a
Newton-CG step therefore solves `H p == -G^-1 g`. Here a gradient is a vector,
a direction is a vector, and the slope is their inner product: there is no
array to put in the wrong convention.

`truncated_cg` is Steihaug's method, and its relationship to `CGSolver` is
worth stating: that solver *refuses* an indefinite operator and raises on
negative curvature, which is right for a linear system and wrong for an
optimiser, where negative curvature says where to move rather than that
something failed.

Three things the tests caught that reading would not have:

1. **The steepest-descent initial step was `1/||g||` with a backtracking-only
   search.** Armijo can never take a *larger* step than it is offered, so the
   method crawled — destroying precisely the conditioning advantage that
   working in the metric is supposed to buy. Found by a test asserting the
   iteration count is stable as the metric spread runs over four orders of
   magnitude; it read `[3, 7, 311]`. The default is now a strong Wolfe search
   with the slope-ratio heuristic.
2. **`white_noise` on a `CoordinateSpace` is legitimately a coordinate
   operation**, since `G^(-1/2)` is a statement about a basis. So `StrictSpace`
   cannot test the randomised methods; `OpaqueSpace` can, and is stronger — it
   is not a `CoordinateSpace` at all, so a component map does not merely raise,
   it does not exist.
3. A Rosenbrock test asserted the Hessian is indefinite at `(-1.2, 1)`, where
   it is in fact positive definite with eigenvalues near 24 and 1506.

### 11.8 Convex analysis and methods

The review asked for in §2.1, followed by the part of it that belongs here.

**What is entangled with inversion, and stays there.** Measured by references
to forward problems, model and data spaces:

| class | inversion references |
|---|---|
| `ChambollePockSolver` | 31 |
| `PrimalKKTSolver` | 9 |
| `SmoothedDualMaster` | 4 |
| `SubgradientDescent`, `Bundle`, `ProximalBundleMethod`, `LevelBundleMethod`, `SmoothedLBFGSSolver` | **0** |
| all of `convex_analysis.py` | **0** |

So the support-function layer and the generic non-smooth solvers are core
numerics; the KKT, Chambolle-Pock and support-value machinery is not, and is
left where it is.

**On coordinates.** The observation that the practical convex problems are
finite-dimensional and canonically Euclidean is right, and it splits cleanly:

- The *proximal operators* that matter are coordinate-free anyway, because
  they are written with a norm and a direction rather than with components —
  `prox` of a norm is `max(0, 1 - t w/||x||) x`, and projection onto a ball is
  `min(1, r/||x||) x`. Both are metric-aware for free and mean the same thing
  under refinement.
- The *subproblem* a bundle method solves over its cut coefficients lives in
  `R^k` for a handful of cuts, has no metric of its own, and is exactly where a
  SciPy- or OSQP-backed quadratic programme behind a protocol is the right
  shape. Coordinates are not a constraint there.

**Brought across so far** (`numerics/convex.py`): `SquaredDistance`,
`NormFunctional`, `BallIndicator`, the `SupportFunction` family with its closed
algebra (Minkowski sum, positive scaling, linear image), and three methods —
`SubgradientDescent`, `ProximalGradient` with FISTA acceleration, and
`ProximalPoint`. `Functional` gained the convex interface the design promised
but the implementation lacked: `subgradient` (defaulting to the gradient, which
is correct for a convex differentiable function), `prox`, and `conjugate`.

v1's `SubgradientDescent` uses a *constant* step and says in its own docstring
that convergence is not guaranteed — which is true, and means it is not a
usable method. The port supplies the rules that do converge, including Polyak's
when the optimal value is known.

Two bugs the tests found:

1. **`compose_with` had its domains backwards.** For `K` in `X` and
   `A: X -> Y`, the image `A K` lives in `Y` and `h_{AK}(y) == h_K(A* y)`, so
   the result is a functional on the *codomain* and the operator must map *out
   of* the set's space. The implementation required the opposite and built the
   result on the wrong space.
2. **The proximal-gradient step ran away.** Doubling it each iteration looks
   like sensible adaptivity, but as the gradient vanishes the
   sufficient-decrease test degenerates and accepts any step; the step reached
   `1e12` and the proximal operator then slammed the iterate onto the minimiser
   of the non-smooth part alone, so the objective *rose* by seven orders of
   magnitude. Beck and Teboulle keep the step monotonically non-increasing for
   exactly this reason, which is what it now does — with an initial step from a
   two-point Lipschitz estimate, since a step that only decreases had better
   start somewhere sensible.

**Still to bring**, in rough order of value: the bundle machinery
(`Cut`, `Bundle`, `ProximalBundleMethod`, `LevelBundleMethod`) together with
the `QPSolver` protocol and its SciPy, OSQP and Clarabel backends; and the
convex *sets* of `subsets.py`, which pair with the support functions.

### 11.9 Testing the abstract framework

Every NumPy-backed test space has vectors that *are* their own components, so
code reaching for array arithmetic or for a coordinate map works by accident.
Three doubles remove the accident:

- **`Opaque`** vectors support no arithmetic at all — no `+`, no `*`, no
  `.copy()` — so anything not routed through the space raises `TypeError`.
- **`OpaqueSpace`** is a `HilbertSpace` and deliberately *not* a
  `CoordinateSpace`, with an inner product that is not the component dot
  product, so code that substitutes one for the other gives wrong answers
  rather than right ones by luck.
- **`StrictSpace`** wraps a coordinate space and raises if the coordinate map
  is touched. This is what turns "coordinate-free" from a claim into an
  assertion: `test_the_algebra_is_coordinate_free` fails loudly if any path
  through the algebra reaches for components. A negative control asserts that
  `matrix()` *does* trip the guard, so the test is not vacuous.

**On petsc4py.** Not a dependency, and not yet a test dependency. A real
backend tests one implementation; the doubles test the contract, and can be
adversarial in ways a real backend cannot. What they cannot show is whether the
API is ergonomic against a genuinely distributed backend, or whether anything
assumes vectors are cheap to copy or local. That gap bites at **M3**, where
coordinate-free Krylov is the claim that needs a real backend — so revisit it
there, as an optional extra (`petsc = ["petsc4py"]`) behind `skipif`, never a
hard dependency.

---

## 12. Findings from reading the v1 concrete spaces and inversion layer

A detailed pass over `symmetric_space/` (~9.5k lines) and the inversion stack
(`forward_problem`, `inversion`, `linear_bayesian`, `linear_optimisation`,
`low_rank`, `preconditioners`, `subspaces`, `convex_*`). Recorded here because
several items changed the design above rather than merely confirming it.

### 12.1 What was confirmed

**The inversion layer is already adjoint-only.** `forward_problem.py`,
`inversion.py`, `linear_bayesian.py`, `linear_optimisation.py`, `low_rank.py`
and `preconditioners.py` contain **zero** occurrences of `.dual`, `to_dual`,
`from_dual` or `LinearForm`. `LinearBayesianInversion.normal_operator` is
written as

```python
A @ Q @ A.adjoint + R              # data-space formalism
Q.inverse + A.adjoint @ R.inverse @ A   # model-space formalism
```

Decision 1 is therefore not a leap — it ratifies how half the library is
already written. Duals survive in v1 only in the core plumbing being replaced
(`hilbert_space`, `linear_operators`, `linear_forms`, `direct_sum`) and in the
functional/representer idiom of §5.6.

**The palindrome rule earns its keep immediately.** `A @ Q @ A.adjoint + R` is
exactly the pattern of §4.1: recognised as `SELF_ADJOINT | PSD`, and
`POSITIVE_DEFINITE` once `R` is. So `CGSolver`'s precondition on the single
most important operator in the library becomes checkable instead of assumed.

**Preconditioners fit the solver protocol unchanged.** Every method in
`preconditioners.py` is already a `LinearSolver` subclass returning an
approximate inverse operator. Nothing to redesign.

**The Gram-in-the-coordinate-layer design fits the concrete spaces.**
`SymmetricHilbertSpace` is an `OrthogonalHilbertSpace` whose metric values are
`laplacian_eigenvector_squared_norm` — a diagonal Gram in the spectral basis.
`SymmetricSobolevSpace` is mass-weighted with an *invariant* (spectrally
diagonal) mass operator. So `gram` and `gram_solver` are both diagonal and
cheap on every workhorse space.

**The trait set matches what exists.** `OrthogonalProjector` is
`SELF_ADJOINT | IDEMPOTENT | PSD`; covariance factors give
`SELF_ADJOINT | PSD`; invariant automorphisms are invertible and self-adjoint.
No trait in §4 is unused, and none was found missing.

### 12.2 What changed in the design

1. **`form="auto"` is for extraction only** (§5.3). `from_matrix` must be told
   which representation its array is in, because no trait implies it —
   `gaussian_measure.py:267-270` builds a covariance factor with
   `galerkin=True` and its inverse with `galerkin=False` on adjacent lines,
   both correctly. Replaced by two named constructors.
2. **A specialisation protocol was added** (§5.4). `InvariantLinearAutomorphism`
   overrides `__add__`, `__sub__` and `__matmul__` to stay in its class, and
   `OrthogonalProjector.complement` must stay a projector. Generic nodes would
   discard both, along with the functional calculus that makes the invariant
   class worth having.
3. **Functional calculus was added** (§5.7). It was missing entirely from the
   first draft, and `linear_bayesian.py` depends on it for covariance square
   roots, log-determinants and stochastic trace estimation.
4. **Measures got the same closure hooks** (§7), for the same reason: a
   spectral measure that degrades to a generic one loses its closed-form KL
   divergence and norms.
5. **§5.6 gained the operator-level case**, the `CoordinateSpace` pairing
   axiom, and a nuance that stops the claim being overstated.
6. **Two more v1 defects** were found and recorded (§9): mass-weighted space
   equality, and the fact that `random_range`'s geometric safety guard is
   defeated by the white-noise bug on exactly the spaces it protects.
7. **`from_formal_adjoint` was wrongly marked for removal and is restored**
   (§3.5), along with `MassWeightedSpace` as a coordinate-free core-layer
   construction. It is used by every concrete space and is the natural way to
   work: derive the formal L2 adjoint, then lift. Its companion
   `from_formally_self_adjoint` is folded into it as a default codomain, since
   its name promises a self-adjointness that the mathematics does not provide
   and that nothing needs.

### 12.3 Not in scope, but noted

- `with_degree` / `degree_transfer_operator` and `with_order` /
  `order_inclusion_operator` build operators between different discretisations
  of the same space family. Nothing in the core obstructs this; it belongs in
  the concrete-space layer, and it is a natural route to multilevel methods
  later.
- `with_formalism("model_space" | "data_space")` runs through the whole
  inversion layer. It is a choice of which of two equivalent normal systems to
  assemble, not a duality question, and needs nothing from the core beyond an
  algebra that is efficient in both directions.
- `AbstractSymmetricLebesgueSpace` supplies pointwise multiplication
  (`HilbertModuleMixin`), used by `spatial_multiplication_operator`,
  `flexural_operator` and `gradient_dot_product`. This is the `HilbertModule`
  capability of §3.1 and carries over unchanged. Note that a pointwise
  multiplication operator is self-adjoint but *not* spectrally diagonal, so it
  is a good test that the trait system and the specialisation protocol stay
  independent of each other.

---

## 13. Symmetric spaces: the port plan

v1's `symmetric_space/` is about 9.5k lines across six files. The port is much
smaller than that number suggests, for two reasons.

### 13.1 Most of the machinery is already general

| v1 | v2 | status |
|---|---|---|
| `SymmetricHilbertSpace` (orthogonal metric) | `DiagonalMetricSpace` | built |
| `InvariantLinearAutomorphism` | `DiagonalLinearOperator` | built |
| its functional calculus | `apply_function`, `sqrt`, `log`, ... | built |
| its closed algebra | the specialisation protocol | built |
| `InvariantGaussianMeasure` | `GaussianMeasure` with a diagonal covariance | built |
| its `1/sqrt(metric_values)` sampling factor | falls out of `white_noise` | built |
| `MassWeightedHilbertModule` for Sobolev | *not needed* — see §13.2 | — |

That last row and the one above it are the interesting ones.
`InvariantGaussianMeasure._kl_sample` hand-codes a
`sqrt(spectral_variances / metric_values)` correction. In v2 the covariance is a
`DiagonalLinearOperator`, its square root is exact, and sampling draws
`white_noise` on a `DiagonalMetricSpace` — which is `xi / sqrt(g)` by
construction. The correction is *derived* rather than written down, which is why
the generic `white_noise_measure` path in v1 could be wrong while this one was
right.

### 13.2 Four spaces become one

v1 implements the circle (1D, `rfft`), the torus (2D, `rfft2`), the line (1D,
built on the circle) and the plane (2D, built on the torus) — roughly 4.1k
lines. All four are the same construction at different dimensions, so v2
implements an **N-dimensional periodic box** via `rfftn` and gets 1D, 2D and 3D
from it. The 3D case is then free rather than a fifth file.

Two simplifications follow:

- **The Lebesgue space gets an orthonormal spectral basis**, so its Gram matrix
  is the identity and v1's factor-of-two bookkeeping
  (`laplacian_eigenvector_squared_norm` returning 1 or 2) disappears. The
  invariant to hold, and to test, is Parseval: `||x||^2 == sum(c^2)`.
- **The Sobolev space is not a mass-weighted space.** It is the *same*
  coordinate map with a different diagonal metric, `g_k == (1 + s^2 lambda_k)^order`.
  So both variants are `DiagonalMetricSpace` and no `MassWeightedSpace` is
  needed for them at all.

What is still needed from §3.5 is the **formal-adjoint lift**: define an
operator on L2, where its adjoint is easy, and reuse it on the Sobolev space.
For two diagonal metrics over a shared coordinate map that is cheap,

```
A*_V == (G_VX^-1 G_UX) . A*_U . (G_UY^-1 G_VY)
```

which is §3.5's `M_X^-1 A*_U M_Y` with `M == G_U^-1 G_V` read off the diagonals.

### 13.3 Stages

| stage | contents |
|---|---|
| **S1** | `spaces/invariant.py`: the shared abstraction — a coordinate space whose basis diagonalises the Laplacian, with invariant operators, invariant measures, Sobolev symbols, Dirac functionals and the formal-adjoint lift |
| **S2** | `spaces/fourier.py`: `PeriodicBox` in any dimension via `rfftn`, giving circle, torus and the 3D box |
| **S3** | non-periodic domains by embedding in a larger box, which is how v1 builds the line and the plane |
| **S4** | the sphere, behind the existing `pyshtools` extra; it is the one space that is genuinely its own implementation |

Out of scope for the core: plotting, which is v1's `plot.py` and the
`matplotlib` imports in every space file.

### 13.4 As built

All four stages are done. 630 tests.

```
pygeoinf2/symmetric_space/base.py     SymmetricSpace, lift_formal_adjoint
pygeoinf2/symmetric_space/fourier.py  PeriodicBox in any dimension, Lebesgue, Sobolev
pygeoinf2/symmetric_space/box.py      Box, Interval -- bounded domains by embedding
pygeoinf2/symmetric_space/sphere.py   Sphere, behind the pyshtools extra
```

**S3, bounded domains**, is a subclass of the periodic box rather than a
wrapper, because that is what it is: the same components, the same metric, the
same operators, differing only in which physical point a grid index means and
where a random point comes from. The support assumption — a field vanishes
outside the domain — is made real by `project_function`, which never calls the
function outside the domain, since it need not be defined there.

`support_projection` is a small illustration of the whole design. Multiplying
by the domain's indicator is self-adjoint and idempotent on a **Lebesgue**
space, so it is an orthogonal projector and its traits say so. On a Sobolev
space it is neither, because a discontinuous mask does not commute with the
metric — so it is refused there, with the error pointing at
`lift_formal_adjoint`, which gives the correct adjoint and claims no symmetry.

**S4, the sphere**, is the one space that is genuinely its own implementation:
a harmonic transform is not an FFT. But everything downstream of the transform
is shared, so what is written is the transform, the spectrum and the geometry
and nothing else — a little over 300 lines against v1's 2215, most of the
difference being plotting and geometry helpers that are not core numerics.

Its conventions are pinned by test rather than assumed: orthonormal harmonics
with the Condon-Shortley phase, a Driscoll-Healy grid at `sampling=2`, and a
point as a `(colatitude, longitude)` pair in radians. The test that pins them
is the same one used for the Fourier spaces — `sum_i c_i phi_i(p) == f(p)` at
grid points — which fails on any normalisation or phase error.

Two details worth recording:

- **The radius belongs in the basis, not the coefficients.** Components are
  scaled by the radius so that the Lebesgue basis is orthonormal on *that*
  sphere. Otherwise every norm, prior and inner product would silently be the
  unit sphere's.
- **`random_point` is uniform in `cos(colatitude)`**, not in colatitude.
  Sampling the angle uniformly crowds the poles, which is the classic way to
  bias a set of station locations, and a test checks the variance against the
  1/3 of a uniform distribution.

### 13.5 Where the packing is delicate

The one genuinely fiddly part is packing `rfftn` output into real components
in more than one dimension. A mode and its conjugate must be counted once, and
which modes are self-conjugate depends on the parity of every axis length.
The design here is to enumerate the conjugate orbits once at construction —
fixed points contribute one real component, pairs contribute a real and an
imaginary one — and to *test* the result rather than trust the derivation:
round-trip, Parseval, and orthonormality of the explicit basis functions.

## 14. Examples

`pygeoinf2/examples/` holds fifteen short scripts, one idea each, meant to be
read in order. They are run by the test suite, so an example that has stopped
working is a failing test rather than something discovered later by a reader.

The sequence: spaces, coordinates, operators and adjoints, traits, **the
derivative-and-gradient distinction**, nonlinear operators and `at()`, solvers,
functional calculus, randomised methods, measures, direct sums and the joint
model, optimisation, convex methods, concrete fields, a worked inverse problem,
the two foreign backends, and the geometry of sets and subspaces.

Number 5 is the one that matters. The distinction between a derivative and a
gradient is why most of the rest of the design looks as it does, and the
example shows all three faces of it: the two readings of one functional, the
fact that they coincide on an orthonormal basis — which is why the mistake
survives — and `check_gradient` failing on the wrong one.

Number 15 is the argument for the whole thing: a complete Bayesian inversion
with no Galerkin flag, no mass matrix written out and no conversion between
derivatives and gradients anywhere in it. Each of those is handled where it
belongs, so none of them has to be remembered at the call site.

The examples are excluded from the code-practice checks of §2. They are
teaching material, and annotating a three-line helper written to illustrate one
idea makes the illustration worse; their test is that they run.

## 15. Foreign backends

`pygeoinf2/backends/` adapts vector and matrix types the library does not own.
MFEM is an optional extra (`mfem`) and is skipped when absent. It settles the
question §11.8 leaves open: `OpaqueSpace` shows the core does not *need*
components, but only a real backend shows it works when the vectors belong to
someone else.

### 15.1 MFEM, which is the case the design was built for

In a finite element space the inner product is `(u, v) == u^T M v`. So the mass
matrix **is** the Gram matrix of §3.2, and three things a practitioner
otherwise writes out by hand fall out of the general machinery:

| MFEM object | what it is here |
|---|---|
| an assembled `BilinearForm` | the **Galerkin matrix** of its operator: `a(u,v) == u^T K v` and `(Au, v) == (Au)^T M v` give `K == M A_c`, so it goes straight into `from_derivative_matrix` |
| an assembled `LinearForm` | a **derivative**, not a gradient: entries `l(phi_i)`, whose representer is `M^-1 b` |
| a mass solve | `solve_gram`, the only place the inverse metric appears |

Verified against a direct solve: `A x == M^-1 K x` to `1e-13`, and CG on the FE
operator matches `np.linalg.solve` to `7e-13`. Vectors are `mfem.Vector`
objects rather than arrays, and nothing in the core notices.

**One real hazard found.** `GetDataArray` returns a NumPy view into memory MFEM
owns, and that view does not keep its owner alive. In
`to_components(from_components(c))` the temporary vector is collected and the
view is left pointing at freed memory — giving plausible wrong numbers rather
than an error. `to_components` therefore returns a **copy**, and a test pins
it. Any backend that owns its own memory needs the same care.

### 15.2 PETSc, withdrawn for now

A `petsc` backend was written and exercised against a source build of PETSc
3.25.4: `PetscSpace` over `PETSc.Vec`, plus a `PetscWeightedSpace` carrying a
mass matrix. It made §5.6's point in the setting where the mistake is most
tempting, since PETSc offers `multTranspose` — the transpose — and does not
offer the adjoint, which on a weighted space is `M^-1 A^T M`. The two agree
whenever the metric is trivial, which is why the substitution survives.

**It has been removed**, along with the extra and the example, because the way
it was installed was wrong rather than because the adapter was. The `petsc`
PyPI package builds its own PETSc, and its own MPI, into the virtual
environment: a slow source build that duplicates whatever the machine already
has and pins the library to a copy nobody else uses. The right approach is to
build `petsc4py` against an existing PETSc installation — the usual case being
a PETSc configured with `--download-mpich`, so that PETSc builds the MPI it
uses rather than taking a distribution package. That is a packaging question,
not a design one, and the backend comes back once it is answered.

What the exercise established still stands: the core ran unmodified over
opaque, possibly distributed vectors, `check_space`, `check_coordinates` and
`check_operator` all passed over `PETSc.Vec` objects, CG converged over them to
a residual of `2e-16`, and the adjoint on a weighted space was confirmed to be
`M^-1 A^T M` and confirmed **not** to be `A^T`.

## 16. Subsets and subspaces

`pygeoinf2/geometry/` replaces v1's `subsets.py` (1713 lines) and
`subspaces.py` (799). The modernisation is one idea and two corrections.

### 16.1 Three views of one object

A convex set, its indicator functional and its support function are the same
thing seen from three directions. v1 keeps them in three places — the sets in
`subsets`, the support functions in `convex_analysis`, and nothing at all tying
an indicator to a proximal method. Here:

```python
subset.contains(x)            # the predicate
subset.project(x)             # the nearest point
subset.indicator()            # a Functional whose prox IS project
subset.support_function()     # the convex-analysis view
```

So a hard constraint needs no new machinery: `ProximalGradient(...).minimise(f,
x0, nonsmooth=subset.indicator())` works for a ball, a half-space, a hyperplane
or an affine subspace alike, and the conjugate of the indicator is the support
function.

### 16.2 `project` means the metric projection

The nearest point of the set, which leaves a point already inside where it is.
That makes it idempotent, and idempotence is what a proximal method relies on.

v1's `HalfSpace.project` instead maps onto the bounding hyperplane whichever
side the point is on — its docstring says so, "independent of inequality type".
That is a legitimate map but it is *not* the convex projection: it is not
idempotent and it moves feasible points. Using it in a proximal iteration would
walk perfectly good iterates onto the boundary and hold them there.
`testing.check_projection` pins all three properties: lands in the set,
idempotent, and nearest.

### 16.3 The trace is not `sum (P e_i, e_i)`

Writing `LinearSubspace.dimension` I reached for the projector's trace as
`sum_i (P e_i, e_i)` and got 169 for a subspace of dimension 12. That sum is
the trace only on an **orthonormal** basis; on a weighted space it is the
Galerkin diagonal and means nothing. The trace is the sum of the *component*
matrix's diagonal, `sum_i [A_c]_ii`.

This is §5.6 in another costume, and worth recording precisely because I made
it while writing the module whose subject is that distinction. A test now
checks the dimension on both a weighted and an unweighted space, where the
wrong formula agrees on one and not the other.

### 16.4 What needs what

Coordinate-free: the whole set algebra, every closed-form projection (ball,
half-space, hyperplane), projectors from a basis, and — less obviously —
projection onto the kernel of an operator, since `P x == x - A* (A A*)^-1 A x`
and `A A*` is recognised as positive semidefinite by the palindrome rule, so
conjugate gradients is admissible with nothing claimed.

Not coordinate-free: `dimension`, which is a trace and so needs a basis.

Not closed-form at all: **the projection onto a general ellipsoid**, which
requires solving a secular equation in a scalar with a linear solve per
evaluation. `Ellipsoid.project` raises and says so, rather than offering an
approximation under the name of an exact operation. Its `contains` and
`support_function` are both exact and coordinate-free.

## 17. Open questions

1. ~~**`DirectSum` vector type.**~~ **Settled**: tuples, with optional labels
   on the space, and labels excluded from the space's identity. See §11.6.

2. **In-place API surface.** `axpy` and `scale_inplace` are the minimum Krylov
   needs. Whether to expose more (`dot_into`, `copy_into`) should be driven by
   a real PETSc backend, not guessed now.

    Response: Agreed. But this should be easy to address as needed. 

3. **Where `gauss_newton_hessian` lives.** A free function in the optimisation
   layer, or a method on `Functional` that returns `None` when it does not
   apply. Leaning free function, so that the base class carries no API it
   cannot generally honour.

    Response: I think free function

4. **Whether `Functional` should accept a `gradient=` entry point at all.**
   Allowing it is convenient for users who genuinely hold a gradient; forbidding
   it makes the §5.6 trap unwritable. Currently written as "allowed but not the
   documented route".

    Response: I think the option is good to have. 

5. **Realification conventions.** Component ordering (`[re, im]` stacked versus
   interleaved) and the factor-of-2 in complex white noise. Cosmetic, but should
   be fixed once and documented rather than discovered twice.

   Response: stacked would be my guess, but not a strong feeling on what is optimal. 

Settled since the first draft: the base operator class is `Operator`
(`NonLinearOperator` retained as an alias); forms are subsumed rather than
removed; **second derivatives are supported for general operators**, curried and
optional; **the derivative is primitive and the gradient derived** (§5.6);
direct sums are in and tensor products of spaces are out; complex data is
handled by realification rather than a complex core (§3.4).



## 18 Inverse problems. 

Here are some rough notes on how the inverse problem section will be redone. It is all based 
on the following methods for classification along a number of axes. First, there is a distinction 
between inverse and inference problems. In an inverse problem you output a model or an object (subset, distribution)
on the model space. In an inference problem there is an associated property operator to a finite-dimensional space 
of quantities of interest (the property space), and the aim is to output a property vector or a subset or distribution 
on that space. Next, we have different types of prior information. There could be none, in which case you map data 
to a point in either the model space or property space (least squares and backus-gilbert would be examples of this). 
You might have prior information as a distribution on the model space, in which case you map to a distribution 
on the model or property spaces (this is Bayesian, of course). Finally, you might have a constraint set on the 
model space as prior, and then you map to a constraint set on the model or property space, with the latter being the 
norm here; these are the backus-type methods that we wrote about later. Added to this you then have the usual 
linear non-linear split. And you have specialisations for different types of prior (e.g., Gaussian priors, convex constraint 
sets). Currently its really just the linear cases with either Gaussian or convex priors that are implemented, but we want a general 
structure in place. If you look, there are already quite a few methods, while there are low hanging fruit on the linear side -- simple 
Backus-Gilbert type estimators that aim to reconsruct a property vector without prior constrainta -- this is the so called "SOLA" method, but note
that we will not reuse this term as bakcus 1970 wrote about it first and the SOLA papers all miss this (and parker developed backus' ideas too 
which they also ignore). This is hopefully a starting point for planning. Some good names will be helpful -- the current ones on the 
convex analysis side need work in particular. 

Oh, and another point, I still want there to be an explicitly named symmetric_space sub-package. These spaces all have a common structure and they are distinct from other spaces we might later implement. 

### 18.1 The classification is a type signature

The axes above are not documentation. They fix the domain and codomain of every
method in the layer, and that is the whole interface.

There are **three** axes, and the third is easy to miss. Following the notation
of the two papers in this directory — `G` there is `A` here, `B` there is `T`:

- the **data relation** is point-valued (`d == A(m)`, exact), measure-valued
  (`d == A(m) + e` with `e` distributed), or set-valued (`d - A(m) in S_eta`
  for a convex noise set);
- the **prior** is absent, a measure `mu_0` on `M`, or a convex set `S_M` in `M`;
- the **target** is the model space, or a property space `P` reached by `T`.

The first two together give the frameworks. Measure/measure is Bayesian.
Set/set is the geometric case both papers develop. The mixed cells are hybrids
and are out of scope, as BGP's Table 1 also says.

Given a data relation and a prior, the target fixes what comes out:

| | **no prior** | **measure prior** | **set prior** |
|---|---|---|---|
| **target = model** `M` | `D -> M` | `D -> Measure(M)` | `D -> Set(M)` |
| **target = property** `P` | `D -> P` | `D -> Measure(P)` | `D -> Set(P)` |

Two consequences carry the design.

**The answer kind matches the prior kind.** No prior gives a point, a
distribution gives a distribution, a constraint set gives a constraint set.
That is the rule that says these six exist and stops a seventh being invented.

**The property row is the model row pushed forward through `T`.** Push-forward
of a point is `T m`; of a measure it is `affine_mapping`; of a convex set it is
the support-function image, which `geometry` already computes. An inverse
problem is an inference problem with `T == identity`, so `target` defaults to
the identity and there is one code path, not two.

So there are four algorithms, not six:

| cell | status | note |
|---|---|---|
| `(none, M)` | v1 has it | Tikhonov and minimum-norm, `linear_optimisation.py` |
| `(measure, M)` | v1 has it | the posterior, `linear_bayesian.py` |
| `(set, M)` | nearly free | prior set intersected with the preimage of the noise set |
| `(measure, P)` | free | push the posterior through `T` |
| `(none, P)` | **empty in v1** | see §18.3 |
| `(set, P)` | v1 has it | `convex_optimisation.py`; four routes, §18.3 |

**Converting between the axes is possible but never canonical.** A Gaussian
data error hardens into an ellipsoid at a chosen chi-squared level; a measure
prior hardens into a ball. The reverse restores detail the set never carried.
Backus (1988) argued a measure prior is strictly the richer object, so the
passage set -> measure *adds* an assumption and measure -> set *discards*
information. Both directions get a named method that says which it is doing —
`harden(level=...)` and nothing at all in the other direction without an
explicit choice — rather than an implicit conversion inside a constructor.

### 18.2 Why two cells are computed in the dual, and what that buys

`(none, P)` and `(set, P)` are still push-forwards mathematically, but they are
not computed that way, for different reasons.

With **no prior**, the model-space answer is ill-posed: `D -> M` needs a damping
nobody can justify. A well-chosen property need not — that is the whole Backus
and Gilbert motivation, and it is why the empty cell is worth filling rather
than deriving.

With a **set prior**, the model-space answer is well defined but lives in an
infinite-dimensional space. A convex subset of `M` can be *represented* and can
answer `contains`; it cannot be explored. Its image in a finite-dimensional `P`
can be, one support direction at a time.

The payoff is a test rather than an assertion. **Where two routes to the same
set both exist, they must agree.** §18.3 gives four of them, overlapping in
pairs, and every overlap is a parity test. Every real defect found in this
refactor was caught by an independent computation of this kind, and this is the
layer where the temptation to skip that is strongest.

v1 already states the central identity without naming it.
`DualMasterCostFunction.__doc__` gives

```
h_U(q) = inf_{lambda in D} { (lambda, d~)_D + sigma_B(T* q - G* lambda) + sigma_V(-lambda) }
```

which is BGP eq. (28): the support function of an image, computed by duality.
Naming it as such is most of the modernisation.

### 18.3 The set-theoretic case: one set, four routes

The feasible model set and its image are

```
S_M = S_M^0  intersect  A^-1(d~ - S_eta)
S_P = T(S_M)
```

both convex when `S_M^0` and `S_eta` are (BGP eqs. 4-5). `S_P` is characterised
by its support function `h(q) = sup { (q, p) : p in S_P }`, and a closed convex
set is determined by it (Rockafellar 13.1-13.2). Four routes compute it, in
increasing order of cost and generality. **They are the same set**, which is
what makes them testable against each other.

**(a) Closed form — no data error, ball prior.** Al-Attar (2021) eq. (2.84).
With `m~ == A*(A A*)^-1 d~` the minimum-norm model and `p~ == T m~`,

```
S_P = { p : ((T P_ker(A) T*)^-1 (p - p~), p - p~) <= r^2 - ||m~||^2 }
```

An **ellipsoid**, centred at `p~`, shaped by `T P_ker(A) T*`, and costing
`dim(P) + 1` minimum-norm solves: eq. (2.88) gives the shape column by column
as `T(u_j - u~_j)` with `u_j == T* g_j`. Every piece of this already exists —
`A A*` is recognised positive-semidefinite by the palindrome rule,
`OrthogonalProjector.onto_kernel` is example 18's subject, and `Ellipsoid`
carries the closed-form support function `(p~, q) + sqrt((r^2 - ||m~||^2)(Sq, q))`
that eq. (2.89) derives by Lagrange multipliers. **This is the low-hanging
fruit**, and it is nearly free.

Two ellipsoids of the same family bracket it and are worth returning together:
the prior alone, `((T T*)^-1 p, p) <= r^2` (eq. 2.85), and the Backus-Gilbert
one below, which is the same shape with `r^2` in place of `r^2 - ||m~||^2`.

**(b) Linear certificate — any `X`, valid always.** BGP §2.5. Weak duality
means *every* `lambda` gives a valid bound, so restricting to a linear family
`lambda(q) == L q`, `X == L*`, costs sharpness and never validity. The dual cost
collapses to the support function of a **Minkowski sum**,

```
X d~  +  (T - X A) S_M^0  +  (-X) S_eta
```

and for norm balls to `(q, X d~) + M ||(T - X A)* q|| + D ||L q||`: a resolution
term and a noise term, added. Choosing `X` by the quadratic surrogate
`M^2 ||T - X A||_HS^2 + D^2 ||C_D X||_HS^2` gives

```
X = T A* (A A* + alpha C_D)^-1,    alpha = D^2 / M^2
```

which is the Backus-Gilbert estimator — a Tikhonov-regularised least-norm
solution pushed through `T`. Al-Attar (2021) eq. (2.106) is the `alpha -> 0`
case, and its eq. (2.109) is the resulting ellipsoid. That paper also explains
why it is looser than (a): the optimality criterion `tr[(T - XA)(T - XA)*]`
never mentions the data, so the norm budget the data has already consumed is
not credited back.

This is the honest form of an error bar on a Backus-Gilbert point estimate. The
estimate is `X d~`; the uncertainty is a set, and it separates into what the
data cannot resolve and what the noise contributes.

**(c) Primal — ball constraints, exact.** BGP §2.6. The support value is a
concave maximisation over the intersection of two balls; multipliers `s, t`
attached to the prior and data constraints give the stationarity condition

```
(s I + t A* A) m = T* q + t A* d~
```

which is a **damped least-squares solve** with damping `gamma == s/t`. The
multipliers are fixed by `||m*|| == M` and `||d~ - A m*|| == D`; both residuals
are monotone in their own multiplier, so nested bisection converges. If the
prior support point `M T* q / ||T* q||` already fits the data, the data
constraint is slack and `h(q) == M ||T* q||` in closed form. Unlike the dual,
this route returns the **extremal model** attaining the bound.

**(d) Dual — general convex sets.** BGP §2.4, the formula above. Needs only the
support functions of `S_M^0` and `S_eta`, which `geometry` supplies, and a
nonsmooth convex minimisation over `lambda in D` — the bundle methods. This is
the general route and the expensive one.

The overlaps are the tests. (a) and (c) agree when the noise set is trivial;
(c) and (d) agree for norm balls; (b) bounds all of them from outside, always,
by construction.

### 18.4 Inner and outer: never report one alone

Support-function evaluation in directions `q_1, ..., q_N` gives a **certified
outer** polyhedron (BGP eq. 11):

```
P_N = intersect_j { p : -h(-q_j) <= (q_j, p) <= h(q_j) }
```

Each direction can only tighten it; the coordinate directions alone give an
axis-aligned box, which contains `S_P` but throws away every correlation
between properties — and the correlations are the reason for working in a
multi-dimensional `P` at all.

Feasibility testing (§18.5) at sampled points and taking the convex hull gives
an **inner** approximation. It is an underestimate and, for a curved boundary,
always a strict one.

The two sandwich `S_P`, and the API should make it impossible to mistake one
for the other. An inner hull returned on its own reads as the answer while
being a guaranteed undercount, which is exactly BGP's Figure 4. So a set
estimator returns an object that knows which side it is on, and the gap between
them is the accuracy statement.

### 18.5 Testing a property value for admissibility

Al-Attar (2021) §2.3 and §3.3. Worth implementing in its own right: it is the
membership oracle behind the inner approximation, and it answers "is this
value possible?" without computing the whole set.

**Error-free.** Parker's joint data-property map `C == (A, T) : M -> D (+) P`.
A value `p` is admissible exactly when the minimum-norm model reproducing both
the data and that property stays inside the prior ball:

```
min { ||m|| : A m == d~, T m == p }  <=  r
```

which is `||C^+(d~ (+) p)|| <= r` (eq. 2.46-2.47). `C C*` acts on `D (+) P`,
so this is Parker's square system of size `dim(D) + dim(P)`, it is
positive-semidefinite by the palindrome rule, and `Column` already builds `C`.
The paper notes `p -> ||C^+(d~ (+) p)||` is convex and continuous, so the
admissible set is convex and bounded — which is the primal proof of what §18.3
computes by duality.

**With data errors.** Reduce to the subspace: `m~ == T*(T T*)^-1 p`,
`r'^2 == r^2 - ||m~||^2` — already a failure if negative — and then the same
problem inside `ker T`, replacing `A*` by `P_ker(T) A*`. Feasibility is decided
by a scalar root find over the multiplier `eta`, where **Lemma 3.1** proves
`eta -> l(d~ - A m_eta)` is non-increasing. If its limit stays above the
confidence level there is no root, and that is a *constructive proof of
incompatibility* rather than a failure to converge.

`l` there is any strictly convex negative log-likelihood, not only the Gaussian
one, so this generalises past the chi-squared case for free.

### 18.6 One numerical kernel

Routes (a), (c), the inclusion test, and the discrepancy principle already in
v1's `LinearMinimumNormInversion` are all the same computation:

> **a damped least-squares solve inside a monotone scalar root find.**

Monotonicity is proved in each case — Lemma 3.1 for the likelihood multiplier,
BGP §2.6 for the two ball multipliers — so bisection is guaranteed, and
non-existence of a root is itself the answer to a feasibility question. One
primitive, four users, and each of them already has a solver: CG on
`A* A + gamma I`, or the data-space form when `dim(D)` is smaller.

That primitive belongs in `numerics`, not in the inference layer.

### 18.7 The interface

`ForwardProblem` stays what its name says: the observation model, `A` and the
data uncertainty — a measure or a convex set, per §18.1. The prior and the
target are arguments to the **estimator**, because the prior is what selects the
method. One problem can then be attacked several ways without being rebuilt,
which is the usual workflow.

```python
problem = LinearForwardProblem(A, error=noise)

point = MinimumNorm(problem, solver=CG())         # D -> M
post  = Bayesian(problem, prior, solver=CG())     # D -> Measure(M)
band  = BackusInference(problem, ball, target=T)  # D -> Set(P)

mu = post(data)
```

The estimator **is** the mapping. Three abstract kinds, distinguished only by
what `__call__` returns:

```python
class Estimator(ABC):
    data_space: HilbertSpace
    target_space: HilbertSpace
    def __call__(self, data: Vector) -> Any: ...
    def push_forward(self, T: LinearOperator) -> Estimator: ...

class PointEstimator(Estimator):    # -> Vector in target_space
class MeasureEstimator(Estimator):  # -> ProbabilityMeasure on target_space
class SetEstimator(Estimator):      # -> Subset of target_space
```

`push_forward` is implemented once per kind, and that single method is the
second consequence of §18.1 made executable.

**A linear point estimator is an `AffineOperator`.** v1 already returns one from
`least_squares_operator`, so `LinearPointEstimator` subclasses `AffineOperator`
and joins the existing algebra. It carries what the method actually produces:

```python
estimator.operator      # X  : D -> P      the estimator itself
estimator.resolution    # XA : M -> P      the averaging kernel
estimator.uncertainty() # a Set or a Measure, per the prior kind
```

The resolution operator is the *output* of a Backus-Gilbert method, not a
diagnostic bolted on afterwards, which is the argument for a typed estimator
over a bare `LinearOperator`.

**A linear Gaussian estimator has a data-independent covariance.** Only the mean
moves, and affinely, so the object is a pair and the push-forward is one line:

```python
class GaussianEstimator(MeasureEstimator):
    mean_map: AffineOperator      # D -> X
    covariance: LinearOperator    # on X

    def __call__(self, data):
        return GaussianMeasure(self.covariance, self.mean_map(data))

    def push_forward(self, T):
        return GaussianEstimator(T @ self.mean_map, T @ self.covariance @ T.adjoint)
```

That is cell `(measure, P)` complete. It is worth a direct path anyway, since
`T C T*` on a small `P` is cheaper than forming the posterior covariance on `M`.

Randomize-then-optimise sampling depends on the data, so the sampler is
constructed inside `__call__` rather than stored.

**The three kinds are connected.** A linear point estimator `X` induces a
measure-valued one — mean `X d`, covariance `X R X*` — which induces a
set-valued one through `credible_set`. With a ball prior it induces a
set-valued one directly, by §18.3(b). Those bridges are methods, not duplicated
implementations.

### 18.8 Names

The Backus family keeps the names of the people who did the work first.
`BackusGilbert` for the no-prior property estimator, `BackusInference` for the
constraint-set one, crediting Backus (1970) and Parker (1977) rather than the
later SOLA literature, which reached the same construction without the
citation. `BackusInference` takes a `method` argument selecting among the four
routes of §18.3, defaulting to the cheapest one its arguments admit.

The algorithm names — bundle methods, Chambolle-Pock, the KKT solver — retreat
into `numerics.convex` as solver strategies, where they describe how rather than
what.

Everything else keeps its standard name: least squares, Tikhonov, minimum norm,
Bayesian.

### 18.9 What leaves the inversion layer

`LinearBayesianInversion` has 25 methods. Four of them are inversion.

| v1 | goes to |
|---|---|
| `diagonal_normal_preconditioner`, `sparse_localized_preconditioner`, `woodbury_data_preconditioner`, `woodbury_model_preconditioner` and their four surrogate variants | `numerics.preconditioners` |
| `_trace_log_slq`, `estimate_log_determinant` | `numerics.functional_calculus` — already ported |
| `low_rank_surrogate` | `numerics.randomised` — already ported |
| `parameterized_inversion`, `data_reduced_inversion` | already just forward to `ForwardProblem`; drop the forwarders |
| `log_evidence`, `mahalanobis_evidence_term` | stay, as a functional on the data rather than a method returning a float |

The same applies to `linear_optimisation.py`, where the `woodbury_*` and
`surrogate_*` methods repeat on each of four classes.

### 18.10 The `formalism` flag

`formalism="model_space" | "data_space"` selects between assembling `Q^-1 +
A* R^-1 A` on `M` and `A Q A* + R` on `D`. This is a **computational** choice:
both assemble the same mapping, and which is cheaper depends only on
`dim(M)` against `dim(D)`.

So it belongs on the construction of the estimator, not on the problem, and it
defaults to whichever is smaller when both dimensions are known — falling back
to an explicit choice when they are not, as for an MFEM-backed model space. The
two paths must be tested to produce the same operator;
`tests/test_linear_optimisation.py:120-128` already does exactly this, so the
harness is inherited rather than written.

The same reduction appears in BGP §2.6: the Woodbury identity turns the
model-space damped solve into a `dim(D)` square system that can be assembled
and factored once and reused across directions and multiplier values. That is
the same flag, reached from the set-theoretic side, and it is the reason the
bisection of §18.6 is affordable at all.

### 18.11 Predicates become geometry

`chi_squared_test(model, data)` asks whether the data lies in an ellipsoid
around `A(model)`. `test_data_compatibility(data)` asks whether the feasible set
is non-empty. Both are `geometry` questions wearing a boolean, and in v2 the
object comes first:

```python
problem.consistency_set(model, level=0.95)   # an Ellipsoid in D
feasible(data).is_empty()                    # on the set estimator
```

The booleans stay as conveniences on top. Note that `consistency_set` is also
exactly the hardening of §18.1 — the data error measure at a chi-squared level —
so a Gaussian problem and a set problem meet here rather than in two code paths.

### 18.12 What `geometry` still needs

Three constructors, all prerequisites rather than afterthoughts:

- **`ConvexSet.from_support_function`** — a set defined by an oracle, since
  route (d) evaluates one bundle solve per direction.
- **Minkowski sums** — route (b) returns `X d~ + (T - XA) S_M^0 + (-X) S_eta`,
  and support functions add. v1 has `MinkowskiSumSupportFunction`; v2 has
  nothing.
- **`Polytope`**, as an intersection of half-spaces with a recorded inner/outer
  status, so §18.4's sandwich has a type rather than a convention.

The linear image of a convex set already works, and it is the operation
everything here is built from.

### 18.13 Stages

| | |
|---|---|
| 5.1 | `problem.py`: port `ForwardProblem` onto the v2 core — measure *or* set data uncertainty, direct sums, parameterisation, data reduction, synthetic data, consistency sets |
| 5.2 | `estimators.py`: the three kinds, `push_forward`, `LinearPointEstimator`, `GaussianEstimator` |
| 5.3 | `point.py`: least squares and minimum norm, with the formalism parity test; the damped-solve-inside-a-root-find primitive into `numerics` |
| 5.4 | `bayesian.py`: the posterior, and the direct `(measure, P)` path |
| 5.5 | `geometry`: `from_support_function`, Minkowski sums, `Polytope` |
| 5.6 | `BackusGilbert`: the point estimate `X d~`, the resolution operator, and the route-(b) uncertainty set — the empty cell, and the cheapest real result in the layer |
| 5.7 | `BackusInference` route (a): the closed-form ellipsoid, `dim(P) + 1` solves, with the prior-only and Backus-Gilbert ellipsoids that bracket it |
| 5.8 | the inclusion test of §18.5, error-free then with errors, and the inner hull it supports |
| 5.9 | `BackusInference` routes (c) and (d): primal bisection, then the dual with bundle methods; the parity tests between all four |
| 5.10 | move the preconditioners and evidence machinery out; delete the forwarders |

Stages 5.6 to 5.8 are worth doing before 5.9. They are closed-form or
near-closed-form, they need no convex solver, and they produce the reference
values that make the bundle-method route testable rather than merely plausible.

The two papers in this directory are the specification for 5.5 onwards, and
both cite this package as their implementation. Their worked examples —
spherical-cap averages under a `H^{3/2}(S^2)` Sobolev prior, and a two-field
flexure-gravity problem — are within reach of the existing `symmetric_space`
sphere and are the natural end-to-end tests.

The nonlinear generalisation is deliberately not staged here. The point of
making every method a mapping with a declared domain and codomain is that the
nonlinear versions occupy the same table with the same signatures — a nonlinear
`PointEstimator` is a general `Operator` rather than an `AffineOperator`, and a
nonlinear `MeasureEstimator` no longer has a data-independent covariance. Those
are the two places the linear specialisation is used, and they are the two
places to look when the time comes.

## 19. The symmetric-space package

**Done.** `pygeoinf2/spaces/` is now `pygeoinf2/symmetric_space/`, `invariant.py`
is `base.py`, and `InvariantSpace` is `SymmetricSpace`, so the package and its
base class say the same thing.

Everything in it qualifies: the circle, the torus and the N-dimensional
periodic box are homogeneous, and `box.py`'s intervals and boxes are built by
embedding into them — which is how v1 arranges `line.py` and `plane.py` for the
same reason. These spaces share a structure that no later space will, and the
name now says so.

**Invariance stays the word for operators.** A `DiagonalLinearOperator` on such
a space is invariant under the group action; the space is symmetric. Using both
words for both things is what made the current naming read oddly.

`spaces` is now free for the cases that are not homogeneous — a finite element
space, an arbitrary mesh — which is where the MFEM backend already points.
