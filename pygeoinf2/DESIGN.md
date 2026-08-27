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
| M5 | The inference layer, on the new core | Parity harness: v1 and v2 side by side on the existing test problems; and the four routes of §18.3 agreeing where they overlap |
| M6 | The observation layer: the operators the worked examples need | `work/sphere_dli_example.py` and `work/tomo.py` reproduced on v2 |

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

Deferred, not dropped: plotting, which is v1's `plot.py` (2164 lines) and the
`matplotlib` imports scattered through every space file. It leaves the space
classes for now so that they carry no rendering, and comes back as its own
layer — see §20.5, O8. **The target is everything v1 does, improved.**

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

**They also need a property operator to exist.** 5.6 and 5.7 are untestable on
anything but a toy until §20.5's O3 or O4 supplies one, and the end-to-end
examples need O2, O3 and O6 as well. M6 is independent of the rest of M5 and
should run alongside it, not after.

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

## 20. What the worked examples need

`work/` and `tutorials/` are the real API surface — what the library is
actually asked to do, as opposed to what its classes offer. Skimming them turns
up one missing layer, one missing constructor, and a confirmation.

`work/sphere_dli_example.py` is the reference implementation of BGP §3.1: a
Sobolev space on the sphere, IRIS stations and USGS events, great-circle path
integrals, spherical-cap averages as the property operator, ball support
functions for the prior and the data-confidence set, and `PrimalKKTSolver` per
direction — route (c) of §18.3. It solves only for `±e_k`, so it computes the
axis-aligned box of §18.4 rather than the polyhedron. `work/tomo.py`,
`work/flexure.py` and `work/dynamic_topography.py` are the same machinery on
one, one and two fields respectively.

### 20.1 The observation layer

None of these examples is limited by the algebra. They are limited by the
*observation operators*, and those live on the concrete space. What v1's sphere
carries beyond the space itself:

| v1 | what it is | v2 |
|---|---|---|
| `point_evaluation_operator` | field to point values | built, dense only |
| `path_average_operator` | geodesic path integrals — the tomography forward map | **missing** |
| `spherical_cap_average`, `geodesic_ball_quadrature`, `spherical_cap_integral` | cap averages — BGP's property operator `T` | **missing** |
| `to_coefficient_operator`, `from_coefficient_operator` | spherical-harmonic coefficients — Al-Attar (2021)'s property operator | **missing** |
| `with_degree`, `degree_transfer_operator` | multi-resolution surrogates; `work/evidence.py` builds a problem at several degrees | only `with_order` |
| `iris_stations`, `random_earthquakes`, `domain_mask`, `random_domain_points` | real acquisition geometry and land/ocean masks, from `pygeoinf/data/*.csv` | **missing** |
| `geodesic_distance`, `pairs_within_distance`, `invariant_covariance_function` | what the localised and sparse preconditioners need | **missing** |
| `point_value_scaled_heat_kernel_gaussian_measure(scale, std=)` | a prior calibrated by *pointwise* standard deviation | `amplitude=` only |

That last one is a one-line derivation rather than a feature: for a homogeneous
field the pointwise variance is constant, so calibrating `std` is a single
scalar read off the spectrum. It is worth having because it is the
parameterisation every example actually uses — nobody knows what amplitude they
want, and everybody knows what pointwise standard deviation they want.

The rest is a real body of work, and it is what stands between the plan and the
two papers' examples running on v2. It should be staged explicitly rather than
assumed, and it is largely independent of §18, so it can proceed in parallel.

### 20.2 The matrix-free adjoint has no guard

Every one of these operators comes in a dense and a matrix-free variant, and
`work/point_evaluation.py` and `work/path_average.py` exist to check that the
two agree and to time them. The check they run is the adjoint identity
`(Au, y) == (u, A*y)_H`, which is `check_operator`.

v1 gets this right, and the way it does so is instructive.
`point_evaluation_operator`'s hand-written `adjoint_mapping` accumulates
`y[i] * laplacian_eigenvectors_at_point(points[i])` — **derivative
components** — and only then applies `from_dual`. The metric enters once, at
the end, exactly as §5.6 requires. But that correctness rests on the author
remembering `from_dual`, across eighty lines with two parallel branches, in
every such operator.

In v2 the dense path is protected: `from_derivative_matrix` takes the rows and
derives the adjoint. **The matrix-free path is not.** `from_callables` takes an
`adjoint`, which is the gradient-valued map, so a matrix-free observation
operator is written in precisely the place where the derivative/gradient
confusion is easiest to make and nothing catches it.

The missing constructor is the matrix-free counterpart:

```python
LinearOperator.from_derivative_callables(
    domain, codomain, value, derivative_components
)
```

where `derivative_components` returns the components of `y` pulled back as a
functional, and the framework applies `representer` once. This closes the last
place in the library where a user must apply the inverse metric by hand.

### 20.3 The `v2/` skeleton already named this

The repository's `v2/` directory is an empty tree of directory names — no code,
zero lines throughout. It is a sketch, and it independently arrives at the same
layout:

```
v2/inversion/{common,forward,point,probabilistic,set}
v2/geometry/{convex,sets,shapes,subspaces}
v2/geometry/shapes/{ellipsoid,polyhedra}.py
v2/symmetric_space/
```

`point`, `probabilistic`, `set` are §18.1's three answer kinds; `polyhedra` is
§18.4's outer bound; `symmetric_space` is §19. Those names are better than the
ones in §18.13 and should be adopted: `inference/{point,probabilistic,set}.py`.

### 20.4 The tutorials

Ten numbered tutorials plus two demos. Structurally they follow the same
sequence as `pygeoinf2/examples/`, which is reassuring rather than informative,
with two exceptions worth noting.

`gaussian_measure_to_sets_demo.ipynb` is the measure-to-set hardening of §18.1,
already a user-facing operation with a credible set at its centre. That
confirms the conversion deserves a name rather than being buried in a
constructor.

`tutorial10` builds the same tomography problem across several geometries to
show what changes and what does not — which is the coordinate-free claim, tested
by variation rather than by assertion. Worth reproducing once the observation
layer exists.

### 20.5 Stages for the observation layer

Independent of §18 and worth running alongside it. O1 is algebra and gates
everything else; O2 to O4 are what a worked example cannot start without.

| | |
|---|---|
| **O1** | `LinearOperator.from_derivative_callables`: the matrix-free counterpart of `from_derivative_matrix`, per §20.2. With a **negative control** — the same operator built through `from_callables` with a metric-free adjoint must fail `check_operator` — and a matrix-free `point_evaluation_operator` built through it |
| **O2** | Geodesics on the sphere: `geodesic_distance`, great-circle quadrature, and `path_average_operator` in dense and matrix-free form. This is the tomography forward map |
| **O3** | Cap averages: the exact spherical-harmonic form and a quadrature form, each checked against the other. This is BGP's property operator `T` |
| **O4** | Coefficient operators: `to_coefficient_operator` and `from_coefficient_operator`, which are Al-Attar (2021)'s property operator. On a spectral space this is a selection of components, so it is `from_derivative_matrix` on a sparse selection and nearly free |
| **O5** | Resolution: `with_degree` and `degree_transfer_operator`, prolongation and restriction as an adjoint pair. Surrogates need it, and so does BGP's convergence-under-refinement claim |
| **O6** | Acquisition geometry: the IRIS station and USGS event tables, `domain_mask`, `random_domain_points`. Data files, so a packaging question as much as a code one |
| **O7** | The parameterisations the examples actually use: pointwise-variance calibration on the invariant measures (§20.1), and `geodesic_distance`-based localisation for the sparse preconditioner |
| **O8** | Plotting, as its own layer rather than methods on the spaces: v1's `plot.py`, `create_map_figure`, `plot`, `plot_points`, `plot_geodesic_network`. A space should say how to sample itself, not how to draw itself, so the natural shape is a renderer that dispatches on the space type — which also keeps `matplotlib` and `cartopy` out of the import path of anything headless |

**On O1's signature.** The caller supplies the action and the *derivative
components* of the pulled-back functional:

```python
LinearOperator.from_derivative_callables(
    domain, codomain, value, derivative_components, *, traits=Traits.NONE
)
```

where `derivative_components(y)` returns `d(A x, y)_Y / d c_x` — for a
Euclidean codomain, exactly the `y[i] * basis_at(points[i])` sum v1
accumulates — and the framework applies `domain.representer` once. Only the
**domain** needs coordinates, which is the right way round: the model space is
where matrix-freeness matters, and it is the space whose metric is not the
identity.

A fully coordinate-free variant would take a callable returning a
`LinearFunctional` instead of components. Nothing needs it yet, and it costs a
functional object per adjoint application, so it waits for a caller.

**On O3 and O4 being the same shape.** Both are property operators into a small
Euclidean space, both are rows of derivative components, and both are therefore
`from_derivative_matrix` with `dim(P)` rows. The difference is only how a row is
computed — a cap integral against each basis function, or a single one — which
is why O4 is nearly free once O3 exists and why neither needs the matrix-free
path. It is the *forward* operators, O2 in particular, that need O1.

### 20.6 As built

O1 to O7 are done; O8 (plotting) remains. 783 tests.

```
algebra/operators.py   from_derivative_callables, matrix(by=...), assembled(),
                       LinearFunctional.derivative_components
symmetric_space/base.py  evaluate, accumulate, point_evaluation_operator,
                       geodesic_distance/quadrature, geodesic_ball_quadrature,
                       path_average_operator, geodesic_ball_average_operator,
                       reference_point, pointwise_variance, pointwise_std=
symmetric_space/sphere.py  the spherical implementations of all of the above,
                       plus spherical_cap_integral/average, coefficient_operator,
                       with_degree, degree_transfer_operator, stations,
                       earthquakes, domain_mask, pairs_within_distance
symmetric_space/fourier.py  evaluate and accumulate by non-uniform FFT
data/                  gsn_stations.csv, usgs_event_cache.csv
```

**Scattered evaluation on a periodic box goes through finufft**, as v1's torus
does. A type-2 transform evaluates and a type-1 transform accumulates, and the
second is *literally* the adjoint of the first — so `evaluate` and `accumulate`
are one algorithm run both ways rather than two implementations to keep in
step. Measured at 655x the direct sum on a 256x256 grid with 3000 points, and
the gap grows with both. finufft handles up to three dimensions; above that,
and when it is not installed, the direct sum still answers.

**No operator has a `matrix_free` flag.** v1 carries one on
`point_evaluation_operator` and `path_average_operator`, each with its own
parallel branches and its own hand-written adjoint. v2 builds matrix-free
always and adds `assembled()` to the algebra, so the choice is made once, in
one place, by whoever knows whether the matrix fits.

For that to be the right default, `matrix()` had to stop filling column by
column. An observation operator is short and wide — a few hundred data from
thousands of components — and filling it by columns costs one *forward*
application per component. Filling it by rows costs one adjoint application per
datum, and row `i` of the Galerkin matrix is `G_X c_{A* e_i}`. `matrix(by=...)`
takes the smaller side by default, and the two directions are tested to agree.

**No operator writes down an adjoint either.** Each is
`from_derivative_callables` or a composition of things that are, so the inverse
metric is applied once by the framework. Two consequences worth recording:

- `path_average_operator` is `W E` — point evaluation at the pooled quadrature
  nodes, then a sparse weight matrix between two Euclidean spaces. All the
  metric lives in `E`; `W`'s adjoint really is its transpose. v1 writes about
  120 lines of adjoint for this operator, and it is right, but its correctness
  rests on remembering `from_dual` in the middle of them.
- `degree_transfer_operator`'s adjoint is the *other* transfer only when the
  two spaces carry the same metric on their shared degrees. Derived, it stays
  right when they do not. That identity is tested rather than assumed.

### 20.7 Four findings

**The chosen quantity was calibrated wrongly, and only on a Sobolev space.**
`pointwise_variance` is `sum_k s_k phi_k(p)^2 / g_k`. The metric appears because
the spectral variances are the covariance *operator's* eigenvalues while a
sample's components carry white noise's `1/sqrt(g)`. Written without it, the
calibration was out by a factor of 2.9 on `H^2` — and exactly right on every
Lebesgue space, which is where it would have been tested first. The measured
sample standard deviation is what caught it, and there is now a negative
control pinning that the naive expression differs.

**`arccos(u . v)` is the wrong great-circle formula.** The cosine is flat near
zero separation, so the arccosine loses about half its digits exactly where a
localisation radius needs them: at `1e-6` radians the relative error is `4e-5`,
and a point against itself does not come out as zero. `pairs_within_distance`
was dropping points from their own neighbourhood. `geodesic_distance` now uses
`atan2(|u x v|, u . v)`, which is accurate at both ends, and the pairwise
version uses the chord form, which is exactly zero on the diagonal.

**`basis_at` was quadratic in the truncation.** It rebuilt pyshtools' packed
Legendre index array on every call — a Python loop of length `dim` — making one
call cost 21 ms at `lmax == 48`. Caching it made that 0.094 ms, a factor of 223,
and turned the adjoint of a realistic tomography operator from minutes into a
second. Nothing about the mathematics changed; the loop was simply in the wrong
place.

**The fast path was missed on the periodic box.** v1 evaluates on the torus
and plane through `finufft`, and the first version of this layer used the
direct sum there — correct, and 655 times slower than it needed to be. Caught
by review rather than by a test, because a slow right answer passes every test
a fast one does. The padding it needs is the subtle part: a Nyquist wavenumber
is `+n/2`, outside the mode range finufft indexes, and folding it to `-n/2`
flips its phase on that axis alone. That is wrong at every point *off* the
grid and right at every point on it, so a grid-only test would have passed.
Widening the spectrum by two along each axis removes the case entirely, and
there is a test on the modes that would move without it.

**The Condon-Shortley constant was misnamed.** pyshtools spells "leave the
phase out" as `csphase=1`, and the constant was called `_CONDON_SHORTLEY` with
a docstring saying the phase was included. The *value* was consistent
everywhere, so nothing was wrong, but the next person to add a call would have
read the name and passed the other one. Now `_NO_CONDON_SHORTLEY`.

### 20.8 Still to come from v1

Superseded by **`V1_CATALOGUE.md`**, which inventories all of v1 — 44 modules,
145 classes, 919 public methods, 84 functions — and gives each one a status:
Ported, Subsumed, Planned, Dropped, or **Open**, the last meaning there is no
recommendation and it needs a decision. It exists because the `finufft` path
of §20.7 was lost without anyone deciding to lose it, and a slow right answer
passes every test a fast one does. The failure mode to guard against is not a
decision to drop something; it is an absence nobody made.

Fifty-three rows are Open. Five of them block work that already exists:
`flexural_operator`, `inverse_flexural_operator`,
`spatial_multiplication_operator`, `vector_multiply` and `vector_sqrt`, without
which `work/flexure.py` and `work/dynamic_topography.py` cannot be reproduced
on v2. `vector_multiply` is the module structure of v1's `HilbertModuleMixin`,
and v2 has no home for it.

The summary that was here:

**The target is everything v1 does, with improvements — not a subset.** What is
listed here is deferred, not dropped.

| v1 | status |
|---|---|
| `plot.py`, `create_map_figure`, `plot_points`, `plot_geodesic_network` | **O8.** Its own layer, dispatching on the space type, rather than rendering methods on the space classes. Deferred so the spaces stay free of `matplotlib` and `cartopy` in the meantime, not because plotting is out of scope |
| `sample_power_measure`, `invariant_covariance_function` | reachable from `invariant_measure` and the diagonal calculus; port when something needs them |
| `datasets.py`'s live FDSN and USGS downloads | the cached tables ship with the package. A live fetch belongs behind an explicit call, never on the path a test takes |
| `parallel=`, `n_jobs=` on every operator | **replaced rather than deferred.** The joblib branches doubled the code and each carried its own copy of the adjoint. finufft and pyshtools thread internally, and parallelism over an operator belongs around it, not inside every one of them |
| `lazy_quadrature=` | **replaced.** An escape hatch for memory the `W E` factorisation does not use |

## 21. Plan of action

From the marked-up `V1_CATALOGUE.md`. Every Open row is now settled, and this
is the order the answers imply.

### 21.1 The target is the worked examples

Not "empty the catalogue". **Reproduce `work/flexure.py`, `work/tomo.py`,
`work/dynamic_topography.py` and `work/sphere_dli_example.py` on v2, end to
end, producing the same figures.** Four real problems, each pulling in a
different part of what is missing, and each arriving with a test that is a
result rather than an assertion.

The ordering falls out of what each one needs, and the first surprise is how
little of it is optional:

| script | needs | inversion |
|---|---|---|
| `flexure.py` | field algebra, the flexural operator, plotting | **none** — CG and a forward model |
| `tomo.py` | fast point evaluation on the sphere, acquisition helpers | M5 stages 5.1, 5.2, 5.4 |
| `dynamic_topography.py` | correlated invariant measures, resolution transfer | M5 as above |
| `sphere_dli_example.py` | the convex machinery | M5 stage 5.9 |

So **`flexure.py` is reachable now** and the other three need the Bayesian
layer. M5 is not the last thing after all; it arrives second.

### 21.2 Work packages

| | contents |
|---|---|
| **F** field algebra | `HilbertModule` as a *capability* alongside `CoordinateSpace` — `multiply`, `sqrt`, `multiplication_operator`. Then `flexural_operator` and its inverse. Small, and it unblocks flexure |
| **S** symmetric-space operators | the sphere point-evaluation speedup; `derivative_operator`, `order_inclusion_operator`, `spectral_projection_operator`, `l2_products_operator`, `gaussian_curvature`, `degree_multiplicity`, `estimate_truncation_degree`, complex coefficient accessors, `cluster_points`, `random_source_receiver_paths` |
| **P** probability | `kl_divergence` with its O(N) spectral path, `nuclear_norm`, `hilbert_schmidt_norm`, `directional_*`, `rescale_directional_variance`, `two_point_covariance`, `with_regularized_inverse`, `with_sparse_approximation`, `deflated_*`, correlated invariant measures, norm-scaled calibration, `sample_power_measure`, `invariant_covariance_function` |
| **A** algebra | matrix-element access, a public sparse operator, `coordinate_inclusion`/`projection`, `extract_diagonal(s)` |
| **N** numerics | GMRES, flexible CG, solver callbacks with richer results, constrained optimisation by projection |
| **G** geometry | ball and ellipsoid *surfaces*, and `AffineSubspace`'s eight remaining methods |
| **O8** plotting | the renderer layer |
| **X** examples and data | the live IRIS and USGS refresh commands, off the import path |

### 21.3 The order

**Phase 1 — `flexure.py` runs.** F in full, `gaussian_curvature` from S, and
enough of O8 to draw a scalar field on a sphere. Nothing else. It is the only
target that needs no inversion, so it is the one that proves the space layer
on its own.

**Phase 2 — M5 stages 5.1 to 5.4, and `tomo.py` runs.** The forward problem,
the estimator kinds, the point estimators and the posterior. Alongside it, the
sphere point-evaluation speedup and the acquisition helpers from S. This is the
largest phase and the one the whole refactor was for.

**Phase 3 — `dynamic_topography.py` runs.** Correlated invariant measures from
P, resolution transfer and the remaining operators from S, and the rest of O8:
maps, point sets, geodesic networks, error bounds.

**Phase 4 — the rest of P, A, N, G.** The catalogue's remaining rows, none of
which blocks an example, so they are done where they fit rather than in a
block.

**Phase 5 — M5 stages 5.5 to 5.10, and `sphere_dli_example.py` runs.** The
geometry constructors, the four routes to the feasible property set, and the
convex solvers. The end of the inference layer.

**Later.** `dynamical_system.py` as a common interface for sequential problems,
and sequential data assimilation on top of it.

### 21.4 Verification, not porting

Four things to check rather than build, from "so long as the functionality is
the same":

1. **`DiagonalLinearOperator` against `InvariantLinearAutomorphism`**, method
   by method, construction included. `invariant_operator` covers
   `from_function`; there is nothing for `from_index_function`, which is what
   a degree-band operator wants.
2. **`GaussianMeasure`'s invariant optimisations.** v1's `kl_divergence` has an
   `O(N)` path when both measures are invariant on the same domain. v2 has no
   `kl_divergence` at all, so both it and its fast path are owed.
3. **`deflated_pointwise_variance`**, which you suspect never worked. Establish
   what it should give against a dense computation before porting it.
4. **`distance_localized_preconditioner`**, which never performed as hoped.
   Measure it against the alternatives before deciding it is worth having.

### 21.5 Decisions taken

- **`from_derivative_matrix` keeps its name.** The alternatives — Galerkin
  matrix, linear forms, functional rows — all name the object accurately and
  none of them names the *convention*, which is the thing the design exists to
  protect. The entries are derivatives, not gradients, and the name should keep
  saying so.
- **Pointwise multiplication is a capability, not a base-class assumption.**
  `HilbertModule` sits alongside `CoordinateSpace`: a space whose vectors are
  functions declares it, and code that needs it asks. An MFEM space can opt in;
  nothing in the core assumes fields multiply.
- **Constrained optimisation is projection onto a convex set**, reusing
  `ConvexSet.project`. That covers a convex constraint with a non-convex
  objective, which is the case that comes up. An augmented Lagrangian, for
  constraints that cannot be projected onto cheaply, waits until M5's convex
  machinery exists and can be the substrate.
- **Parallelism stays outside operators.** Recorded with your caveat: the
  action of a single operator will need parallelising for large problems, and
  when it does it belongs in the operator's own implementation rather than as a
  flag threaded through every constructor.

### 21.6 Phase 1 as built

`work/flexure.py` runs on v2, as `examples/20_flexure.py`. 836 tests.

```
algebra/spaces.py       HilbertModule, require_module
symmetric_space/base.py truncate, multiply, sqrt, multiplication_operator,
                        gaussian_curvature, gradient_dot_product,
                        flexural_operator, inverse_flexural_operator
numerics/solvers.py     IterativeSolver.with_preconditioner
plotting/               base.py dispatch, sphere.py, fourier.py
```

**Pointwise multiplication is a capability.** `HilbertModule` sits beside
`CoordinateSpace`: a space whose vectors are functions declares it, and
`require_module` names it when something needs it. Nothing in the core assumes
fields multiply, so an MFEM space can opt in and a space of abstract
coefficients is not asked to pretend.

**Multiplication by a field is not self-adjoint on a Sobolev space.** It is
self-adjoint for the `L2` inner product, and a space that weights its modes does
not have that inner product. So `multiplication_operator` builds the operator
where the claim is true and lifts it through `lift_formal_adjoint`, and the
lifted operator claims nothing — §3.5 again, in the place it is easiest to get
wrong, since the *action* is identical either way.

**The flexure operator needed the Bochner identity**, which produces
`tr(Hess D_eff Hess w) + 2 K grad D_eff . grad w` as a unit from three calls to
`gradient_dot_product`. No Hessian and no tangent frame is ever formed. That
matters beyond flexure: it is the general route to a second-order
variable-coefficient operator on a symmetric space.

### 21.7 Three findings

**v1's `gradient_dot_product` has the wrong sign.** With the positive
Laplacian this package and v1 both use, `grad f . grad g == (f L g + g L f -
L(f g)) / 2`; v1 computes the negative of that. Verified against
`grad sin . grad cos` on a circle, where the ratio to the analytic answer is
exactly `-1`.

It propagates into `flexural_operator`, and the reason it survived is worth
recording: **every use of it is inside a term proportional to the gradient of a
coefficient, and those vanish identically when the coefficient is constant.**
So the constant-coefficient case — the one with a closed-form spectral symbol
to check against — cannot see it. Measured against the exact one-dimensional
beam operator `(D w'')'' + rho w` with a varying `D`, the corrected version is
right to `6e-9` relative and v1's sign is wrong by 36%. With a constant `D`,
both agree to `4.5e-4`.

**Self-adjointness does not pin the curvature term.** The first check tried was
whether `2 K grad D_eff . grad w` is constrained by the operator being
symmetric. It is not: dropping, doubling and negating `K` all leave the
operator self-adjoint to machine precision. The term is live — it is 0.6% of
the operator — so `check_operator` passing said nothing about it. What does pin
it is a closed form: a degree-one harmonic on the unit sphere is the
restriction of a linear function, so `Hess f == -f g_ab` and therefore
`tr(Hess f Hess g) == 2 f g` exactly. The Bochner block matches that to
`7e-11`, and is 50% to 100% wrong with the curvature coefficient dropped,
doubled or negated.

**A product on a sphere has no canonical grid representative.** The
Driscoll-Healy grid is oversampled — eight grid points per dimension of the
space — so `from_components(to_components(x))` is a projection rather than the
identity, and many grid arrays share a set of components. A pointwise product
leaves the space, so its raw grid array is one of those non-canonical
representatives, and the formal-adjoint lift, which round-trips through
components, disagreed with a direct application by 8%. `multiply` now
truncates, so the product depends only on its factors. `truncate` is the
identity on a periodic box, where `rfftn` gives one component per grid point,
and is overridden there to skip the two transforms.

### 21.8 Next

Phase 2: M5 stages 5.1 to 5.4 and the sphere point-evaluation speedup, with
`work/tomo.py` as the acceptance test.

### 21.9 Phase 2 as built

`work/tomo.py` runs on v2, as `examples/21_tomography.py`. 882 tests.

```
inference/problem.py     ForwardProblem, LinearForwardProblem
inference/estimators.py  Estimator and its three kinds, LinearPointEstimator,
                         GaussianEstimator
inference/point.py       LeastSquares, MinimumNorm, choose_formalism
inference/bayesian.py    Bayesian
probability/gaussian.py  credible_set
geometry/convex.py       Ball.translate, Ellipsoid.translate
symmetric_space/         basis_matrix, dense= on the observation operators
plotting/sphere.py       plot_points, plot_paths
```

**The estimator is the mapping**, as §18.7 said it should be. `Bayesian(problem,
prior)` *is* the map from data to posterior; `LeastSquares(problem)` *is* an
`AffineOperator` and joins the algebra. `push_forward` is implemented once per
answer kind, and cell `(measure, P)` is the one line the design promised.

**The formalism chooses itself.** `choose_formalism` takes the smaller of the
two spaces, and the tests pin that both assemble the same mapping — mean and
covariance agreeing to machine precision, and both agreeing with a dense
reference.

### 21.10 Three findings

**A dense reference is easy to write wrongly, in exactly the documented way.**
The first reference for the posterior used `Q A_c^T (A_c Q A_c^T + R)^-1 d`,
which disagreed. In components the adjoint is `G^-1 A_c^T`, not `A_c^T`, so the
right expression carries the inverse metric — §5.6, met while writing the test
rather than the code. The reference in `test_inference.py` says so, because the
mistake is the natural one.

**A supplied sampler must be centred.** `GaussianMeasure.sample` adds the
expectation to whatever a `sample` callable returns, so a randomise-then-
optimise draw written in the obvious way lands at *twice* the posterior mean.
Caught by comparing three thousand draws against the mean.

Writing it centred turned out to say something better. The centred draw is

```
(u - mu) + K(-A(u - mu) - (e - mu_e))
```

which **never mentions the data**. The posterior *fluctuation* is
data-independent — the same statement as the covariance being
data-independent, and the reason this estimator is a pair. So the sampler is
built once, not per call.

**A tall operator wants assembling by its rows, not by `assembled()`.**
`matrix(by="rows")` costs one adjoint application per datum, and for a
tomography operator each of those touches every quadrature node — quadratic in
the data. But the rows are known in closed form: they are `basis_matrix`. So
the observation operators take `dense=True`, which is not the `matrix_free`
flag §20.6 argued against, because it duplicates no adjoint — the same
derivative components are used either way. Measured: building 60 paths at
`lmax == 64` takes 153 ms, after which applications are 1.6 ms against 66 ms.

Separately, batching the sphere's `basis_at` gave 2x on the adjoint. The
azimuthal factor depends only on the order `m`, of which there are `lmax + 1`
rather than `dim`, so computing `cos(m phi)` once per order and gathering
replaced tens of millions of trigonometric evaluations with a few hundred
thousand.

### 21.11 Next

Phase 3: correlated invariant measures, the remaining symmetric-space
operators, and the rest of the plotting layer, with
`work/dynamic_topography.py` as the acceptance test.

### 21.12 The sphere can use a non-uniform FFT, and should

Prompted by the question of whether the `finufft` route that made the periodic
box fast can help the sphere. It can, by a lot, and the measurements are worth
recording before the work is done.

**The method** is the double Fourier sphere. A band-limited function on the
sphere extends to a *trigonometric polynomial* on the torus, by

```
g(theta, phi) = f(theta, phi)          theta in [0, pi]
g(2pi - theta, phi) = f(theta, phi + pi)
```

so its two-dimensional Fourier coefficients are exact, obtained by one FFT of a
grid twice as tall. Evaluation at scattered points is then a type-2 NUFFT, and
no associated Legendre function is touched at all.

**Verified.** A prototype agrees with the current route to a relative error of
`2.5e-13` at `lmax` of 32, 64 and 128, which is what confirms the extension is
exact rather than an approximation.

**Measured**, at `lmax == 128` (`dim == 16641`):

| points | current | NUFFT at `eps=1e-8` |
|---|---|---|
| 2 000 | 0.65 s | 26 ms |
| 20 000 | 7.1 s | 23 ms |
| 100 000 | 35.6 s | 27 ms |

The current cost is `O(n dim)`; the NUFFT's is `O(dim log dim + n)`, so the gap
grows without limit. A realistic tomography adjoint goes from tens of seconds to
tens of milliseconds. One wrinkle: the coefficient step took 255 ms in the
prototype, which is far more than an FFT of a 512x512 grid should cost and is
the first thing to look at.

**The adjoint is the work.** The chain is `T == F D B` — synthesis onto the
grid, the double-sphere fold, the FFT — and its transpose needs `B^H`, which is

```
B^H g == (4 pi / R) SHExpandDH(g / a_j)
```

with `a_j` the Driscoll-Healy quadrature weights. That is derivable rather than
calibrated, which is the requirement; but `DHaj` is not exposed by the
installed pyshtools, so the weights must be derived and pinned against an
explicit basis sum first. Two further pieces, `F^H == ifft2` and the transpose
of the fold, are easy and separately testable.

**One row is missing from the fold.** The Driscoll-Healy grid samples
colatitude on `[0, pi)`, so the south pole — needed for the doubled grid's
middle row — is not a sample. It is one `basis_at` call, which is exact and
cheap, but it is the kind of detail that would give a plausible wrong answer if
missed.

Staged as **S-NUFFT**, ahead of the rest of package S: it changes what is
feasible rather than making an existing thing tidier. **Built; see §21.15.**

### 21.13 Phase 3 as built

`work/dynamic_topography.py` runs on v2, as `examples/22_coupled_fields.py`.

```
symmetric_space/base.py  spectral_operator, degrees, degree_multiplicity,
                         spectral_projection_operator, order_inclusion_operator,
                         l2_products_operator, estimate_truncation_degree
probability/base.py      directional_covariance, directional_variance,
                         two_point_covariance
probability/gaussian.py  kl_divergence, nuclear_norm, hilbert_schmidt_norm
```

**The coupled prior turned out not to need correlated measures.**
`work/dynamic_topography.py` builds its two-field prior with `from_direct_sum`
of two *independent* measures, so it is a product measure, which v2 already
had. `CorrelatedInvariantGaussianMeasure` stays Planned but is not on the
critical path, and the example shows why: the coupling that matters is
generated by the *data*, not asserted by the prior.

**`spectral_operator` replaces `from_index_function`** and takes an array
rather than a callable on indices, so the caller writes
`space.spectral_operator(f(space.degrees))`. That is what `degrees` is for: the
index a component sits at, as opposed to the eigenvalue it carries. The two
differ exactly when a symbol is not a function of the Laplacian — a band
projection, or the `1/(2l+1)` of a gravity kernel.

### 21.14 Two findings

**A trace is the component matrix's, not the Galerkin matrix's.** The first
`nuclear_norm` and `hilbert_schmidt_norm` read `matrix(form="galerkin")`, which
is `G C_c` — and `tr(G C_c)` is simply a different number from `tr(C_c)`. The
operator trace is basis-independent and comes from the component matrix. On a
Lebesgue space the two agree, so this is another one that is invisible exactly
where it would first be tested; the test now asserts that the two matrices have
*different* traces before checking which one the norm uses.

The Kullback-Leibler divergence is the opposite case and worth stating beside
it: there, every metric factor cancels. `tr(Q^-1 P)` and `log det Q - log det
P` are the same in either representation, and the metric survives only in the
quadratic term `(Q^-1 d, d)`, which is an inner product rather than a trace.

**A second observable can decouple two fields entirely.** The example expected
the density anomaly and the basal traction to trade off against each other,
since the prior makes them independent and the data see them together. Measured
as a Frobenius ratio of the posterior covariance's blocks, the coupling is
`0.0000`. The reason is structural: the topography channel sees the flexure,
and so the traction, on its own — which leaves the geoid constraining the
density alone. Dropping the topography restores the coupling to `0.0836`. The
example now demonstrates that rather than asserting the opposite, and it is the
better result: it says what the second observable was worth.

### 21.16 Phase 4 as built

The catalogue's remaining algebra, numerics, probability and geometry — the
rows that block nothing, which is why they came last. 993 tests.

```
algebra/operators.py    sparse matrices in the matrix constructors, diagonals()
algebra/spaces.py       coordinate_projection, coordinate_inclusion
numerics/solvers.py     GMRESSolver, FlexibleCGSolver, callbacks and history
probability/gaussian.py with_regularized_inverse, with_sparse_approximation,
                        rescale_directional_variance
symmetric_space/        norm_std=, power_measure, covariance_function,
                        derivative_operator, cluster_points,
                        source_receiver_paths
geometry/convex.py      BallSurface, EllipsoidSurface
geometry/subspaces.py   from_tangent_basis, from_complement_basis,
                        from_hyperplanes, to_hyperplanes, pseudo_inverse,
                        projection_operator, boundary, with_translation,
                        with_constraint_value, the remembered equation
```

**Sparse matrices needed no new constructor.** A `scipy.sparse` matrix supports
`@` and `.T`, which is all `from_component_matrix` and `from_derivative_matrix`
ever use, so the only change was to stop calling `np.asarray` on it. v1 has two
classes for this; v2 has one branch in one helper.

**A subspace now remembers the equation it was built from**, and says so. One
built from a basis knows its tangent space but not which particular `A x == b`
a caller had in mind, so `constraint_operator` raises rather than inventing one
with the same solution set. `to_hyperplanes` is the honest alternative: *an*
equation, from an arbitrary orthonormal basis of the complement.

### 21.17 Three findings

**GMRES was silently restarting every step.** The Givens rotation zeroes the
subdiagonal entry it acts on — that is the point of it — and I then used that
same entry as the breakdown test and as the next basis vector's normalisation.
So the test `== 0.0` was always true, the inner loop broke after one column,
and the algorithm was GMRES(1) wearing GMRES(30)'s name. It converged, slowly
and geometrically, which is exactly how a correct restarted GMRES on a hard
problem also looks. What caught it was not the residual but the *count*: a
twelve-dimensional system must be solved exactly in twelve steps, and it was
taking twenty-four to reach `4e-5`. Now it is `5e-16` in twelve.

**Thresholding a covariance usually leaves a covariance, which is why it needs
checking.** Dropping small entries from a positive semidefinite matrix
preserves positivity often enough that a spot check would pass. It is not a
theorem: `with_sparse_approximation` verifies and refuses, and the test carries
a specific four-by-four covariance that goes to a smallest eigenvalue of
`-0.41` at a tenth of its largest entry.

**An exact diagonal costs one application per column; a banded one need not.**
v1 extracts diagonals with `dim` applications of the operator. For an operator
that really is banded, probing with vectors that are one on a whole residue
class of columns gets every requested diagonal in one application per *offset
span*, independent of dimension. That is exact only for a banded operator and
sums in the out-of-band entries otherwise — so it is a named option rather than
the default, and there is a negative control showing it give a different answer
on a full operator.

### 21.18 What is left

Four rows, all in package P, and two of them are one thing:

- **`CorrelatedInvariantGaussianMeasure`** and `correlated_invariant_gaussian_measure`. Not needed by any worked example — `work/dynamic_topography.py` uses a *product* measure — so the design question it raises is unforced: your note says the reason to keep it is the extended Karhunen-Loeve expansion it makes samplable, which is a statement about sampling rather than about covariance, and worth settling before writing it.
- **`deflated_diagonal`**, and `deflated_pointwise_variance` with it. Your note says you are not sure it ever worked properly. That makes it a verification task before it is a porting one: establish what it should give against a dense computation, then decide.
- Plus the two live download commands in package X, and `IterativePreconditioningMethod`, which belongs with M5's other preconditioners.

### 21.19 Correlated invariant measures, and the deflated diagonal

The two rows §21.18 left open. Both are built.

**`correlated_measure`** puts several fields on one domain and correlates them
*scale by scale*: at each mode the coefficients come from one small covariance
matrix rather than independently, so the correlation between the fields is a
function of scale rather than a single number multiplying two marginals. That
is the whole difference from a product measure, and it is what a coupled
physical prior needs.

**Sampling is an extended Karhunen-Loeve expansion, and it costs no code.** The
covariance *factor* is the block operator carrying the symmetric square roots
``L(k)``, so one draw of white noise on the direct sum — correlated mode by
mode by the factor — is a sample. The ``1/sqrt(g)`` a non-trivial metric
demands rides on the white noise rather than being written out, which is §13.1
one field wider. v1 hand-codes the expansion and the metric correction; here
neither appears.

`correlated_measure_from_correlations` is the parameterisation anyone has an
opinion about: each field's own spectrum, and how strongly they are correlated,
either once or once per scale.

**`deflated_diagonal`** removes the leading eigenpairs before sampling the
rest. The Bekas-Kokiopoulou-Saad estimator's variance is set by the size of the
*whole* operator, not by the size of what it is failing to resolve, so every
covariance with a decaying spectrum is estimated badly for a reason unrelated
to its tail. Measured on a Sobolev space with a spectrum decaying by `0.6` per
mode, the relative error falls from `0.36` undeflated to `0.0004` at full rank
— a factor of a thousand for the same number of probes. It does work.

One thing it needed care with: the exact contribution of the low-rank part is
``sum_i lambda_i (G c_i)_j^2`` for the Galerkin diagonal and
``sum_i lambda_i c_{ij} (G c_i)_j`` for the component one — one metric or two.
The first version had it wrong both ways and passed on a Euclidean space, where
they coincide. It is tested on a weighted one.

`pointwise_variance_at` is the general counterpart of `pointwise_variance`:
exact at one covariance application per point, or deflated when there are many.
The invariant case has a closed form, which is what it is checked against.

## 22. Phase 5: the set-theoretic layer

### 22.1 As built so far

Stages 5.5 to 5.8, and route (c) of 5.9. 1031 tests.

```
geometry/convex.py     ConvexSet.from_support_function, Minkowski sums,
                       translation, Polytope
inference/backus.py    BackusGilbert, BackusInference, FeasibleProperty
```

**A set given only by its support function** is the shape §18.3(d) forces, so
it is a first-class object. What it can do is bounded by what a support
function determines: it gives a certified outer `Polytope` and a one-sided
certificate of non-membership, and it refuses `contains` and `project` rather
than approximating them. `Polytope` carries whether it is an inner or an outer
bound, because BGP's Figure 4 is about exactly the mistake of reporting the
first as the answer, and the two cannot be intersected with each other.

**Three routes to the feasible property set are now here, and they agree.**

- `BackusInference` — route (a), the closed-form ellipsoid for error-free data
  and a ball prior. Its `prior_only()` is the bracket the data are meant to
  improve on.
- `BackusGilbert` — route (b), the linear certificate. Its `error_bars` split
  the bound into a resolution term and a noise term, because more data narrows
  the second and better coverage the first, and one number cannot say which is
  wanted.
- `FeasibleProperty` — route (c), the primal bisection, which returns the
  **extremal model** attaining each bound as well as the bound.

### 22.2 The parity tests, which are the point

- Route (c) against route (a), as the noise ball shrinks: the disagreement
  *tracks the noise radius*, `5.7e-4` at a radius of `1e-3` and `5.6e-7` at
  `1e-6`. That is what says the difference is the problem rather than the
  method.
- Route (b) bounds both, in every direction tested. Validity is free by weak
  duality; only sharpness is lost.
- The **inclusion test** against the closed-form ellipsoid: 400 candidate
  values, 400 agreements. One is a minimum-norm solve on Parker's joint map
  ``C == (A, T)``; the other is a projection and an eigen-shape. They have no
  code in common.
- Every model sampled *from* the feasible set lands inside the reported set.

### 22.3 Four numerical findings, all in route (c)

The primal bisection is the most delicate thing in the package so far, and it
was wrong four times before it was right. Each failure is worth recording
because each looked like success.

**The bracket must widen at both ends.** Widening only upwards leaves the
search converging to whatever the lower end happened to be — a wrong answer
that has converged, with a residual that looks fine.

**`(s, t)` is the wrong parameterisation.** With the multiplier `s` itself, the
model's norm is a ratio of two large numbers as the data weight grows and the
whole expression cancels. BGP writes it with `gamma == s / t`, and that is not
a stylistic choice: with `gamma`, the ``1/t`` multiplies the *small* term and
everything stays bounded.

**The misfit has an exact form with no cancellation at all.** Woodbury gives
``m* == (1/gamma)(w' - A* z)``, but ``A m* == (gamma I + A A*)^-1 A w'``
identically — the difference cancels analytically. So the residual is stable at
every damping, and only the norm needs care.

**And the norm needs its kernel part computed directly.** Taking it as *the
whole minus the range part* subtracts two quantities of order one to get one of
order ``1e-16``. At a data weight of ``1e8`` that returned `2.8` for a model
whose norm was `0.85` — and the bisection then converged, confidently, on
nothing. Since ``A* d`` lies entirely in the range of ``A*``, the kernel part
of ``w'`` comes only from ``(1/t) T* q``, and can be had at its own scale.

**The honest limit.** With all four fixed, route (c) tracks route (a) down to a
noise radius of about ``1e-7`` and degrades below it — a double-precision limit
of a Woodbury-based bisection, and not a problem, since the noise-free case is
exactly what route (a) is for.

### 22.4 The DLI example

`work/sphere_dli_example.py` runs on v2, as `examples/23_feasible_set.py`:
`H^{3/2}` on the sphere, 186 real source-receiver paths, four spherical caps as
the property, a norm ball for the prior and another for the noise. Both routes
(b) and (c) run, so the certificate can be seen bounding the exact answer — by
12% on average here.

The example's own conclusion is the one worth having, and it is measured rather
than asserted. The certificate's width splits into a resolution term and a
noise term; resolution accounts for **95%** of it, so the limit is coverage and
not measurement quality. Counting ray samples near each cap confirms the
mechanism exactly:

| cap | interval width | ray samples within 0.3 rad |
|---|---|---|
| 0 | 12.20 | **0** |
| 1 | 5.75 | 203 |
| 2 | 10.71 | 34 |
| 3 | 6.13 | 51 |

The widest interval is the cap the rays miss entirely, and the ordering is
monotone in coverage. A method that reported a number with an error bar would
have reported a number for cap 0 too.

### 22.5 Preconditioners

`SpectralPreconditioner`, `BandedPreconditioner` and `BlockPreconditioner`,
beside the identity and Jacobi that were already there. Each is a different
guess about *why* an operator is ill-conditioned, and each is only worth
anything when its guess is right.

Measured on a positive-definite operator with a spectrum decaying by `0.9` per
mode: 125 conjugate-gradient iterations plain, 89 with Jacobi, 62 with rank-20
spectral, **22** with rank-50. On a genuinely banded operator: 41 plain,
**4** banded.

**And one that makes things worse, recorded rather than guarded against.** A
tridiagonal preconditioner applied to a *dense* operator took the same solve
from 125 iterations to outright failure. Nothing in the library can detect that
the operator lacks the structure being assumed, which is why the bandwidth is a
required argument with no default — asking for it is the only place the caller
is made to state the assumption.

That is also why `BandedPreconditioner` extracts its diagonals with the *exact*
probe by default. The fast probe of §21.17 sums out-of-band entries into the
band, which is harmless when the operator is banded — the two agree, and there
is a test — and compounds the damage when it is not.

### 22.6 Route (d), the general one

`DualFeasibleProperty`, on a `ProximalBundleMethod`. This is BGP eq. (28) —

```
h(q) == inf over lambda of  (lambda, d) + h_prior(T* q - A* lambda) + h_noise(-lambda)
```

— and it is exactly what v1's `DualMasterCostFunction` writes down without
naming it as the support function of an image.

**It asks for nothing but a support function and a maximiser.** Routes (a),
(b) and (c) all need norm balls and say so by name; this one accepts an
ellipsoid, a box, an intersection — anything convex that can exhibit the point
attaining its own support. That is the whole reason for its expense, and the
tests demonstrate it: an anisotropic prior, which the other three refuse, runs
here, and a ball *written as* an ellipsoid gives the same numbers as the ball.

**Against route (c): agreement to nine figures.** A nested bisection over two
Lagrange multipliers against a nonsmooth convex minimisation in the data space,
with no code in common. That is the parity §18.2 was written to make possible.

**An unbounded dual is an answer, not a failure**, and a dangerous one to
return: with no model both inside the prior and fitting the data, the primal
supremum is over an empty set and the dual falls away without limit — to
`-2e8` in the case that first showed it up. A number that size is a perfectly
plausible-looking support value, so it is raised with the reason named instead.

### 22.7 Two findings in the bundle method

**A bundle method must carry the linearisation error.** A cut taken at ``x_i``
bounds ``f`` from below everywhere, but its offset *at the current centre* is
``f(x_i) + (g_i, c - x_i)``, not ``f(x_i)``. Dropping the second term makes
every cut look tight, the predicted decrease collapses, and the method reports
convergence immediately — at `0.65` for a problem whose minimum is `0.23`, with
a gap of `1e-11` to say how sure it was. Each cut now stores the point it was
taken at.

**And the subproblem does not want a general solver.** The dual of the model is
a quadratic program on the unit simplex in the number of cuts — a few dozen
variables. Handing it to `scipy.optimize.minimize` cost **50 seconds** per
minimisation, essentially all of it in setting up problems that small.
Projected gradient with a closed-form simplex projection does the same job in
**0.6 seconds**, an eighty-fold speed-up, and the whole test module went from
189 seconds to 26.

### 22.8 Hardening a measure into a set, properly

`weighted_chi2_cdf` and `weighted_chi2_quantile`, and the two measure methods
that need them.

`sum_i w_i Z_i^2` has no closed form. It appears wherever something is said
about the size of a Gaussian vector in a metric other than its own: the
Mahalanobis form is an ordinary chi-square because its weights are all one, but
the plain squared norm of a field is not, and its weights are the covariance's
eigenvalues. Imhof's inversion of the characteristic function is exact to the
tolerance asked for — matching `chi2.ppf` to `1e-12` at seven terms — and
degrades where the integrand decays slowest, so the equal-weight case, which is
both the commonest and the worst for the quadrature, is short-circuited to the
exact chi-square.

`ambient_ball` is the second hardening of §18.1: the smallest ball about the
mean carrying a given probability, in the *space's* norm rather than the
distribution's. It is the bridge from a Gaussian belief to the norm bound a
set-theoretic prior wants, so it is the conversion that actually gets used.

**Neither hardening contains the other in general**, which is what "not
canonical" means made concrete. Both carry the same probability and they are
different regions; which way any containment runs depends on how anisotropic
the measure is. A test asserts only that they differ, because the stronger
claim — that each reaches somewhere the other does not — is true for a mild
anisotropy and false for a strong one, and asserting it was the first thing
tried.

**And `credible_set` was wrong.** On a weighted space it covered **46%** of its
nominal 90%. The precision was built by inverting the covariance's Galerkin
matrix, but the Galerkin matrix of `C^-1` is `G C_gal^-1 G`, not `C_gal^-1` —
inverting the Galerkin matrix gives the component matrix of something else
entirely. The two coincide on an orthonormal basis, which is where it had been
tested. `as_multivariate_normal` had the same error in the other direction: the
covariance *of the components* is `G^-1 C_gal G^-1`, and the operator's
component matrix is not symmetric on a weighted space, so scipy refused it
outright rather than accepting it and being wrong.

`condition` completes the pair: the Bayesian update as a statement about a
measure, needing no forward problem. It agrees exactly with `Bayesian` on the
same data, which is two routes to one answer again.

### 22.9 Evidence, and an exact constraint

`Bayesian.log_evidence` and `ConstrainedLeastSquares`, which close M5's point
and measure estimators.

The evidence is returned as two terms as well as one number, because they
answer different questions: the Mahalanobis term says whether the data are
surprising under this model, and the log-determinant term penalises a model
flexible enough that they could not have been. It matches `scipy.stats`'
multivariate normal density exactly, on an orthonormal space and on a weighted
one — the second only after subtracting the metric's own determinant, which is
the same `G` bookkeeping as §22.8.

`ConstrainedLeastSquares` handles a constraint that is *exact* rather than
probable — a boundary condition, a fixed total mass, a known mean — which a
prior cannot express and a penalty only approximates. Writing the subspace as
``t + range(P)`` and substituting ``u == t + P w`` turns it into the
unconstrained problem for ``A P`` and ``d - A t``, so it stays affine in the
data and stays an operator.

### 22.10 What is left

Nineteen rows, and they fall into three groups.

**Alternative convex solvers** — `QPSolver` with its SciPy, OSQP and Clarabel
backends, `SmoothedDualMaster`, `SmoothedLBFGSSolver`, `ChambollePockSolver`.
These solve the problem `ProximalBundleMethod` already solves, by other means.
Porting them is worthwhile for speed on particular shapes, and is not
worthwhile merely for completeness — the question is which of them earns its
place, and that is a judgement about workloads rather than about code.

**The rest of the preconditioners** — `ColumnThresholded` and the localised one
from `linear_bayesian.py`. Mechanical, and each needs a problem of the right
shape to show it is worth anything, which §22.5's measurements argue is the
only honest way to add one. The Woodbury one is done: §22.12.

**Plotting and the tail** — `SubspaceSlicePlotter`, `plot_slice`, the corner
plots, `config.py`, the live download commands, and the `dynamical_system`
family, which was marked *later* deliberately.

### 22.11 Set inclusion with errors, as a constrained optimisation

Route (a) as first written did only the error-free case: the ellipsoid of
Al-Attar (2021) eq. (2.84), and an `inclusion_norm` built on Parker's joint map
`C = (A, T)` answering `min { ||m|| : A m = d, T m = p }`. The paper also does
the case *with* errors, and §3.3 is the reduction: replace the data constraint
`A m = d` by `||d - A m|| <= D` and the problem becomes, in eq. (3.28), the
same one restricted to `ker T`.

The reduction is worth writing out, because it is what makes the computation
stable. Split `m = m~ + u` with `m~ = T*(TT*)^-1 p` the smallest model having
the proposed property and `u ∈ ker T`. The two are orthogonal, so the norms
separate: `||m||^2 = ||m~||^2 + ||u||^2`, and `T m = p` holds for every `u`.
What is left is a pure discrepancy problem — the smallest `u` in the kernel
bringing the residual `r = d - A m~` inside the noise ball — and with
`P` the orthogonal projector onto `ker T` and `K = A P A*` its data-space
form, the solution at damping `γ` is `u = P A* z` with `z = (γI + K)^-1 r`.
Then

```
misfit(γ) = ||r - A u|| = γ ||z||        ||u||^2 = (K z, z)
```

both monotone in `γ` and both free of cancellation, so the bisection on the
misfit is a bisection on a quantity computed at its own scale. This is the same
lesson as §22.3's kernel norm, where computing a small quantity as
*whole minus range* returned 2.8 for a model of norm 0.85.

Two consequences are worth naming. The answer is `inf` when the part of `r`
outside the range of `A P` exceeds the noise radius — no model with that
property fits the data however large it is allowed to be. That is a *proof* of
inadmissibility, the constructive half of the paper's Lemma 3.1, not a failure
to converge, and the tests assert it as such.

And it is the complement of the support function, exactly as the sandwich of
§18.4 needs. A support function bounds the set from outside, one direction at a
time; the inclusion test decides membership exactly, one point at a time, and
is the only one of the two that can produce an *inner* bound — `inner_hull`
takes the convex hull of the admitted candidates. The two are computed by
completely different routes and must agree about every point they both have an
opinion on, which is the parity test: of 120 probes, none admitted lay outside
the outer polytope and none the support function certified outside was
admitted. Against route (a) with the noise ball shrunk, the difference tracks
the radius — 1.46e-2 at 1e-2, 1.24e-4 at 1e-4, 1.02e-6 at 1e-6.

**A singular normal operator, found by the new fixture.** The error-free
`inclusion_norm` solved `C C* x = t` with conjugate gradients and a
`POSITIVE_DEFINITE` claim, while its own docstring said positive
*semi*definite. Both are right: `C C*` acts on `D ⊕ P`, and once
`dim(D) + dim(P)` exceeds `dim(M)` it is genuinely singular, because a model
cannot generically match more numbers than it has. Every existing fixture had
`dim(D) + dim(P) <= dim(M)`, so it had never been seen; the with-errors tests
introduced a 3 + 2 against 4 and CG broke down at iteration 6.

The fix is the same reasoning as the reduction above rather than a wider
tolerance: go through the spectrum, and the unreachable part of the joint
target is by definition the part in the kernel. If it is non-zero the answer is
`inf`. So `inclusion_norm` became total, and for the same reason.

That needed a metric-correct eigendecomposition, `_self_adjoint_spectrum`. The
*Galerkin* matrix `G N_c` is the symmetric one (§5.6), not the component
matrix, so the eigenproblem is the generalised `M v = λ G v`, whose vectors are
orthonormal in the space's own inner product. The previous code symmetrised the
component matrix by hand, which is right on a Euclidean space and wrong
anywhere else — the same class of error as the `credible_set` bug in §21.11,
and again invisible until a weighted space was used. The tests now run the
whole construction on a weighted model space for that reason.

The property pseudo-inverse `T*(TT*)^-1` is factored, not iterated. The
property space is small — that is what makes it a property space (§18.1) — so
CG runs out of Krylov space before it runs out of tolerance and reports the
round-off as a non-positive curvature direction. A `2 x 2` system does not want
an iterative solver.

### 22.12 The Woodbury preconditioner

Marked in §22.10 as mechanical; it is, but it was asked for by name as the one
that has been useful in practice, and it is. The two normal operators

```
N_m = Q^-1 + A* R^-1 A     N_d = A Q A* + R
```

each have an inverse written in terms of a solve in the *other* space:

```
N_m^-1 = Q - Q A* (R + A Q A*)^-1 A Q
N_d^-1 = R^-1 - R^-1 A (Q^-1 + A* R^-1 A)^-1 A* R^-1
```

which is the same trade as `choose_formalism` (§18.6) — solve wherever the
dimension is smaller — but used as a *preconditioner* rather than as the solve
itself, which is what makes it useful when neither space is small enough to
settle the matter outright.

Two things turn an identity into a preconditioner: the inner solve is cheap, or
the pieces are surrogates — a smoother forward operator, a stationary prior
standing in for a non-stationary one, a diagonal noise covariance standing in
for a correlated one. v1 exposed the surrogate case as
`surrogate_woodbury_data_preconditioner`; here it needs no separate entry point,
because the pieces are constructor arguments and passing cheap ones is the
whole of it. Correctness never depends on how close they are, only the
iteration count does.

`WoodburyPreconditioner` is a `LinearSolver` like the others, and works out
which of the two identities to use from the space the operator it is handed
acts on. The model form needs only *applications* of `Q` and `R`, never their
inverses, which is why it survives a prior whose inverse is unbounded — a
Sobolev covariance. The data form needs `Q^-1` and `R^-1`, so it wants
covariances given in inverse form, and takes them directly when they are known.

**How it is tested.** Woodbury is an identity, so the decisive test is not that
it helps but that it is *exactly right* when nothing is approximated: with
exact inner solves, `wood.model_form()(N_m x)` must return `x`. It does, to
1e-14, on a Euclidean space, a weighted model space and a weighted data space —
the metric variants included because the whole construction is written in
Hilbert adjoints and a components-for-Galerkin slip would show up there and
nowhere else. An iteration-count improvement would not have revealed such a
slip at all; it would just have preconditioned slightly worse.

The practical claim is measured separately. Against `N_m` of condition 1.5e4
with an exponentially decaying prior, a deliberately crude surrogate — that
decay replaced by a ten-step staircase, and cruder noise — took CG from 262
iterations to 23, with the two answers agreeing to 6e-12.

One caveat is documented on the class rather than guarded: with an inexact
inner solve the preconditioner is no longer exactly symmetric, and ordinary CG
relies on a symmetric preconditioner. `FlexibleCGSolver` (§22.5) exists for
precisely that, and is the answer when the inner solver is itself iterative.

While adding it, the package `__init__` was found to export only two of the
five preconditioners. All six are exported now.

## 23. The inversion classes: solvers, preconditioners and names

Adding `WoodburyPreconditioner` in §22.12 built the preconditioner and left no
way for an inversion to use it, which is most of what made it useful. Looking
at the inversion classes properly showed the gap was wider than that one class,
and in three distinct places.

### 23.1 What had been lost

**The normal operator was a local variable.** In `Bayesian.__init__`, and in
`LeastSquares`, `MinimumNorm` and `ConstrainedLeastSquares`, it was assembled,
inverted and discarded. v1 exposed it as a property. Nothing could be built
against it and nothing could be measured about it — its condition number, which
is the number that says whether a solve will be hard, was unobtainable.

**Exposing it would not have been enough.** v1's `diagonal_normal_preconditioner`
uses `<v, A Q A* v> == <A* v, Q A* v>`, and `sparse_localized_preconditioner`
takes sub-blocks of `A Q A*` before the noise is added. Both need `A` and `Q`
*apart*. Woodbury needs all three. Assembling `N` into a single operator
destroys exactly the structure they run on — which is the real reason they were
methods on the inversion class in v1, and not an accident of organisation. It
is the only place that still held the parts.

**The surrogate case had no expression at all.** And reading `work/tomo.py`
showed that it needs something I had not allowed for: the surrogate lives on a
*different model space*, a sphere of a sixth the degree with its own prior and
its own path-average operator. Only the data space is shared. That is precisely
why `A Q A* + R` is the formalism that survives the substitution — it acts on
the data space whatever the model space is — and why the model-space form has no
surrogate story.

### 23.2 The normal operator carries its factors

So `NormalOperator` is a `LinearOperator` that remembers what it was assembled
from: `formalism`, `forward`, `prior`, `error`, and the covariances and
precisions taken off them. It behaves as the assembled operator everywhere an
operator is wanted.

Generic preconditioners — Jacobi, spectral, banded, block — see a
`LinearOperator` and are unchanged; v2's `with_preconditioner` already hands
them the operator being inverted, which is better than v1, where the caller had
to fetch `normal_operator` and build one by hand. Structure-aware
preconditioners take a `NormalOperator` and read the factors off it.

The gain over v1 is that preconditioners stay free-standing. In v1 each was a
method on `LinearBayesianInversion`, so it could not be used with any other
inversion, and every new inversion class would need the whole family added to
it again. Here they are ordinary `LinearSolver` objects, and the inversion
exposes one property instead of eight methods.

`NormalOperator` also owns the formalism-dependent algebra that was inline in
`Bayesian`: `gain`, `posterior_covariance`, `right_hand_side`,
`weighted_adjoint`. That is the right place for it, because the formalism *is*
the normal operator, and it makes `LinearGaussianInversion` thin.

The surrogate is a method on it. It returns a normal operator rather than a
whole inversion — v1 returned a `LinearBayesianInversion` — because the normal
operator is the only part of a surrogate problem that is ever used. v1's four
`surrogate_woodbury_*` entry points collapse into passing cheap arguments.

### 23.3 The name

`Bayesian` claimed a paradigm and delivered one closed form. The class computes
the conjugate Gaussian update: the forward operator must be linear, the prior
must be Gaussian, and the error measure must be Gaussian. Nothing else is
covered, and the name said none of it. Its `prior` argument was typed
`ProbabilityMeasure` while the code read `.covariance` and `.precision` off it,
so a non-Gaussian measure failed somewhere inside rather than at the door.

It is now `LinearGaussianInversion`, keeping v1's `Inversion` suffix, and both
the prior and the error measure are checked to be Gaussian at construction with
an error that says why.

### 23.4 The two structure-aware preconditioners

`NormalDiagonalPreconditioner` computes the Galerkin diagonal of `A Q A* + R`
through `<v, A Q A* v> == <A* v, Q A* v>` — one adjoint application and one
prior application per entry, never an application of the assembled operator and
never the forward operator at all. Given blocks it uses each block's normalised
indicator as the probe, so a block shares one adjoint application.

`LocalisedPreconditioner` keeps the sub-blocks of `A Q A*` that couple, each by
a randomised Nystrom decomposition at a fixed rank, assembles them sparsely,
adds the noise diagonal and factorises once with a sparse LU. Blocks may
overlap, unlike the diagonal one, because this approximates the operator rather
than partitioning it.

**How they are tested.** Three kinds of statement, chosen because most of what
can be said about a preconditioner is a matter of degree and these are not:

* *An identity that must hold exactly.* Woodbury with exact inner solves is the
  inverse, and `from_normal` must not change that. A single full-rank block
  covering every index reproduces `N^-1` when `R` is diagonal.
* *Two routes to one number.* The cheap diagonal identity against
  `JacobiPreconditioner` on the same operator, which forms the Galerkin matrix
  and reads its diagonal. They agree to 1.9e-16.
* *An answer that must not depend on the preconditioner.* The posterior mean,
  through all four, agreeing to 1e-13.

Every one runs on a weighted data space as well as a Euclidean one. A diagonal
is a statement about a basis, and the Galerkin form is the one in which a
self-adjoint operator is symmetric; on a Euclidean space the two coincide and
the distinction hides. It is the same class of error as §21.11's `credible_set`
and §22.11's eigendecomposition, and this is the third time it has come up.

**A silent approximation, found by testing an exactness claim.** The localised
preconditioner at full rank on one block was 39% off, and the comment in the
code said the noise "is never approximated". Both were wrong in the same place:
only `A Q A*` is treated block-wise and the error covariance contributes its
diagonal, so `R`'s off-diagonal is dropped. With a diagonal `R` the same
construction is exact to 1e-14 on both space types, which is what confirmed the
diagnosis. The behaviour is v1's and is right — `R` is diagonal in nearly every
real problem — but it is now documented as an approximation and there is a test
asserting that it *stops* being exact when `R` is correlated, so the
documentation cannot quietly become false.

### 23.5 What example 24 measures

The tomography problem of example 21 at degree 64: 4225 model dimensions, 704
paths, and per-station noise spanning two orders of magnitude, which is what
makes the normal operator badly scaled rather than merely large. Condition
number 3.5e4.

```
no preconditioner                   1057 iterations
diagonal, 16 receiver blocks         518 iterations
diagonal, exact                      609 iterations
Woodbury from the surrogate           10 iterations
```

all agreeing on the answer to 1e-10. The surrogate is a sphere of degree 10 —
121 dimensions against 4225, thirty-five times smaller — with its own prior,
damped to give it the precision the Woodbury data form needs.

The third line is worth more than the fourth. The *blocked* diagonal beats the
*exact* one, at a forty-fourth of the cost, because the blocks are the receivers
and the noise is constant per receiver: the block structure is the operator's
structure, and approximating the rest is not a loss. A preconditioner that
matches the problem beats a more accurate one that does not — which is the
argument for keeping the factors around, since without them the block structure
is not expressible at all.

An earlier version of the example used uniform noise, giving condition number
354, and there both diagonal preconditioners made the solve *worse* (90
iterations to 101 and 114). That is the §22.5 lesson again and it was left in
the design record rather than tuned away: a preconditioner is worth what it
measures on the problem at hand, and nothing in general.

### 23.6 The distance preconditioner, and why it disappointed

The catalogue's note against `distance_localized_preconditioner` read *"This
was never as good as I hoped -- so check the implementation -- but should be
useful"*. It is worth having: when the forward operator is point evaluation and
the prior is invariant, the two-point covariance depends on a pair of points
only through the distance between them, so

```
(A Q A*)_ij == k(d(p_i, p_j))
```

and the whole matrix can be written down from a table of distances with **no**
applications of the forward operator, its adjoint, or the prior covariance. It
is the only preconditioner here whose cost does not scale with the model space
at all.

The implementation is right — with no truncation it reproduces the Galerkin
matrix of `A Q A*` to 3.5e-11 — and the disappointment is one line: `apply_taper`
defaulted to `False`.

**Truncating a covariance matrix does not preserve positive definiteness.**
Dropping entries beyond a radius is not a positive operation, and conjugate
gradients needs a positive definite preconditioner — an indefinite one does not
slow it down, it breaks the recurrence it relies on. On 150 points of a degree
48 sphere with a heat-kernel prior, against a true spectrum of `[7.9e-5, 7.4]`:

```
radius   pairs   untapered min eig   tapered min eig
0.10       210        -6.59e-01          +1.44e-01
0.20       394        -9.86e-01          +4.03e-02
0.40       986        -5.32e-01          +1.14e-02
```

Not marginally indefinite: eigenvalues comparable in magnitude to the
operator's largest, and negative, at every radius. The Gaspari-Cohn taper is
what fixes it — multiplying by a compactly supported positive definite function
before cutting is a Schur product, which does preserve definiteness — and it
is what `apply_taper` switched on. So the default was the broken one.

In v2 it defaults to `True`, and the test asserts the sign of the smallest
eigenvalue both ways, so the reason is recorded as a fact about the code rather
than as a remark.

Two further things the measurements say, both worth keeping:

**Even tapered, it is a weak preconditioner here** — 374 iterations to 257,
where the surrogate Woodbury on a comparable problem gives a hundredfold. It
earns its place on cost, not on strength: nothing else in the library builds a
preconditioner without touching the model space.

**The `max_distance == 0` case is provably useless as v1 used it.** An
invariant prior has one pointwise variance, so with uniform noise the diagonal
is a constant and the preconditioner is a multiple of the identity — which
cannot change what conjugate gradients does. Measured: 374 iterations
preconditioned, 374 unpreconditioned, exactly. v1 had a special case for it. It
is worth something only when the noise varies between data, and the docstring
now says so.

### 23.7 What this leaves

The catalogue's preconditioner and surrogate rows are now closed. `low_rank_surrogate`
is real rather than a pointer at `numerics.randomised`, and needed
`GaussianMeasure.low_rank_approximation`, which returns a measure with a
covariance *factor* and therefore no precision — so a low-rank measure can
stand in for a prior in the data-space formalism and not in the model-space
one, which is the sort of thing the formalism check now says out loud.

`parameterised` and `data_reduced` existed on `LinearForwardProblem` and are
lifted onto the inversion, as in v1. They are not preconditioners: they give a
different and generally worse answer, and the point is that it may be the only
one that fits.

Still open from §22.10: the alternative convex solvers, and the plotting tail
with the `dynamical_system` family.

### 23.8 The last preconditioner, and the same bug a third time

`ColumnThresholdedPreconditioner` keeps a column's entries when they are large
relative to that column's diagonal. It is what to use when the operator is
sparse in a basis but not *banded* in it — a covariance over scattered points,
where what couples to what is geometry rather than index distance.

It has the defect of §23.6 in a different disguise. Dropping entries column by
column does not produce a symmetric matrix: entry `(i, j)` can pass its own
column's test while `(j, i)` fails its own. So the result is asymmetric, and
conjugate gradients needs a symmetric preconditioner. v1 did not symmetrise.

The fix is to threshold the *pattern* rather than the values — keep a position
when either column wants it — and then read the values off the Galerkin matrix,
which is symmetric to begin with. Nothing that was asked for is dropped, and
the result is symmetric by construction. The test checks self-adjointness
directly rather than trusting the argument.

That is three appearances of one shape:

* the untapered distance truncation (§23.6), indefinite;
* column-wise thresholding, asymmetric;
* and, from §22.12, an inexact inner Woodbury solve, also asymmetric.

Each is an approximation made independently in each column, row or iteration,
of an object whose whole meaning is a relation *between* them. It is cheap to
prevent in every case and silent in every case, and the third one is documented
rather than fixed only because there the fix is the caller's — use
`FlexibleCGSolver`, which is what it is for.

Two boundary cases pin the implementation, both exactly rather than by degree:
a zero threshold keeps everything, so the preconditioner is the exact inverse;
and a cap of one entry per column is exactly `JacobiPreconditioner`, which is
the only sane reading of "keep the largest, and the diagonal". Both run on a
weighted space as well as a Euclidean one.

With this, every preconditioner row in the catalogue is closed.

## 24. The point estimators, and the kernel they all share

The point estimators were the thinnest thing in the library: 292 lines against
v1's 1330. Some of that is genuine compression — v1 repeats the surrogate and
preconditioner family across four classes — but most of it was loss, and one
piece of it was a bug.

### 24.1 The bug

`MinimumNorm.for_data` returned the wrong end of its bracket. The discrepancy
principle wants the *largest* damping whose misfit still reaches the threshold;
when no damping is large enough to miss it — when the data are consistent with
noise and every model fits — the answer is therefore the largest damping and
the smallest model. It returned the smallest damping, and its docstring stated
the correct reasoning immediately before doing so.

```
data that anything fits:
   for_data chose damping 1.0e-12  ->  model norm 3.1e-06
   largest damping that fits: 1.0e+12  ->  model norm 4.5e-15
```

Nine orders of magnitude, and in the worst possible direction: for data
supporting no structure it returned the *most* structured answer available.
v1 handles this correctly and directly — `if chi_squared <= critical: return
zero`.

The fix is not a corrected branch but the primitive below, which reports which
end it ran out at and returns that end's value. The correct behaviour is then
the default rather than a case someone has to remember.

### 24.2 One kernel, four users — finally

§18.6 already said it:

> a damped least-squares solve inside a monotone scalar root find. One
> primitive, four users... That primitive belongs in `numerics`, not in the
> inference layer.

It was listed in stage 5.3 and never built. What existed instead was two
ad-hoc copies — `backus.py`'s `_bisect` and an inline loop in `for_data` — and
`InverseOperator.solve` had taken an `x0` all along that **no caller in the
package passed**.

`numerics/root_find.py` now holds `monotone_root`, and `backus.py`'s `_bisect`
delegates to it. All thirty-five Backus tests pass unchanged, parity between
routes (a), (c) and (d) included, which is what makes the retrofit safe to
claim.

Three things belong to the primitive rather than to any of its users.

**Bracketing at both ends**, which §22.3 already learned the hard way.

**Saturation as an answer.** Failing to bracket is not an error: the
non-existence of a root is the answer to a feasibility question. It is reported
with the endpoint reached, and getting that endpoint right is §24.1.

**Warm starting.** Consecutive multipliers in a bisection converge on each
other, so each solve is a correction to the last. On a 300-dimensional
ill-conditioned system with 24 probes:

```
cold          6504 inner iterations   0.54 s
warm-started  4892 inner iterations   0.36 s     (identical damping)
```

A second saving is separate and larger where it applies: a preconditioner
supplied as a `LinearSolver` is otherwise rebuilt against *every member of the
family*, which for a Woodbury surrogate with its own inner factorisation costs
more than the solves it accelerates. `DampedSolves` builds it once and rebuilds
only when the multiplier has moved by more than a set factor. A preconditioner
is an approximation, so reuse costs accuracy, not correctness — the threshold
is where that stops being a good trade.

The cost is reported rather than hidden. Without the iteration count there is
no way to tell a warm start that is working from one that silently is not,
which is the failure mode of an optimisation nobody can see.

### 24.3 Tikhonov as a family, not an assembly

Tikhonov least squares **is** the Gaussian case with an isotropic prior, and
exactly so: `Q^-1 == t I` in the model space, `Q == (1/t) I` in the data space,
where the two factors of `1/t` cancel between the gain and the operator. The
test asserts it at machine precision rather than leaving it as a remark.

They are still separate classes. Not because the identity is doubtful, but
because a `NormalOperator` is one assembly of one problem and `N(t)` is a
*family* whose whole purpose is to be walked along — by a discrepancy search,
by an L-curve. An object whose point is the sweep should say so in its type,
and warm starting has nowhere to live on a single assembly. Reading `t` as a
prior variance is also a claim about what regularisation means, and a damping
does not have to make it.

`TikhonovNormalOperator` carries its factors, so every structure-aware
preconditioner of §23 applies to the point estimators — through the shared base
of §25.1, which is what makes that true rather than merely plausible. The estimators
gained what §23 gave the Gaussian one: `normal_operator`, `right_hand_side`,
`with_solver`, `with_formalism`, `with_damping`, `surrogate`, `parameterised`,
`data_reduced`, and `residual_callback` for v1's progress tracking.

### 24.4 The derivative of a damping found from the data

v1's `minimum_norm_operator` returns a non-linear operator with an exact
analytic Fréchet derivative *and its adjoint*. v2 had replaced it with "fix a
damping, hand back a linear estimator", which has the wrong derivative with
respect to the data and cannot sit in a differentiable chain.

The formula was derived here independently rather than ported on trust.
Differentiating `H(t) u == A* R^-1 d` and `chi^2(u, d) == target` together
gives `du/dd == L - h (x) dt/dd` with `h == H^-1 u`, and

```
dt/dd == (L* A* R^-1 r - R^-1 r) / (R^-1 r, A h)
```

which is not v1's expression. The normal equations themselves supply the
missing step: `A* R^-1 A u + t u == A* R^-1 d` gives **`A* R^-1 r == -t u`**,
and with it both numerator and denominator come out as `-t` times v1's, so the
factors cancel and the two agree. v1's formula is correct.

`DiscrepancyPrinciple` is an `Operator`, not a `LinearPointEstimator`, because
the map genuinely is not affine — two data vectors needing different dampings
are related by no fixed matrix. The correction for the damping moving is
rank one, which is why the derivative costs one extra solve rather than a new
problem.

**Verified two ways, and the first attempt looked like a failure.** Central
differences disagreed by 2.5e-2 — until the damping search was tightened. The
map is only as differentiable as its damping is converged, and at the default
`rtol` of 1e-6 with a finite-difference step of 1e-6 the difference quotient is
all noise:

```
search rtol 1e-06, FD step 1e-06  ->  2.48e-02
search rtol 1e-14, FD step 1e-06  ->  5.57e-10
search rtol 1e-14, FD step 1e-04  ->  7.43e-10
search rtol 1e-14, FD step 1e-03  ->  6.98e-08
```

The last line rising is truncation error behaving as it should, which is how
one can tell the middle rows are the real agreement. The adjoint dot-product
test — the check that catches a right formula with a wrong adjoint, which is
the more likely error and the one that stays invisible until something upstream
calls it — passes at 7e-17 independently.

**A saturated search has no such term.** When no damping brings the misfit to
its target, the damping in force is pinned by the end of its range rather than
chosen by the data, so it does not move when the data do and the rank-one
correction is not merely unnecessary but wrong. This was found by the
finite-difference test on a constrained problem with a two-dimensional subspace
and eight data, which cannot fit them at any damping. The derivative there is
the fixed-damping estimator alone.

Reaching that case also broke the primitive: the bracket walked down to a
damping of `1e-200`, where the normal operator is numerically singular and the
factorisation fails. That is not a bug either — it is the edge of the usable
range — so a breakdown during bracketing now ends the walk and reports the last
multiplier that worked, which is exactly the saturated answer.

### 24.5 The constrained pair

`ConstrainedMinimumNorm` had no v2 counterpart, and `constraint_value_mapping`
— how the answer moves when the constraint value does, at fixed data — is the
question a constrained inversion invites.

Porting it produced one error worth recording, because it is the kind that
looks right. The mapping adds the unconstrained solution to a point of the
subspace; using the *unconstrained* method there walks straight off the
constraint, and only the **reduced** method — the one built on `A P` — has
answers in the tangent space. v1 does this correctly and the reason is not
obvious from its code, since its "unconstrained inversion" attribute is already
the reduced one. The test that caught it simply asks whether `B u == w` still
holds; it did not, by 130%.

## 25. A pass over the new code

Reading §§23–24 back with an eye for reach-throughs and misplaced imports found
one false claim, two private-member accesses, one contract weakened by
accident, and a wasted solve per search.

### 25.1 A claim that was not true

§24.3 said every structure-aware preconditioner of §23 applies to the point
estimators unchanged. It did not. `_require_normal` checked
`isinstance(operator, NormalOperator)`, and `TikhonovNormalOperator` is not
one — so `NormalDiagonalPreconditioner` and `LocalisedPreconditioner` refused
the point estimators outright, with a message telling the caller to do the
thing they had just done.

This is what comes of two classes agreeing by construction rather than by
declaration. The contract is now written down as `FactoredNormalOperator`, an
abstract base declaring `formalism`, `forward`, `prior_covariance` and
`error_covariance`, from which the spaces follow. Both normal operators inherit
it and the preconditioners check for it.

Structural typing — a `Protocol` — was the other candidate and is worse here:
`runtime_checkable` resolves attributes by `hasattr`, which *calls* the
property, and `TikhonovNormalOperator.prior_covariance` deliberately raises at
zero damping. The check would have evaluated the thing it was only supposed to
look for.

`WoodburyPreconditioner.from_normal` still duck-types, and deliberately: it
lives in `numerics`, which must not import `inference`. Its error message names
the three attributes it wants.

### 25.2 Two reach-throughs

`DampedSolves` read `solver._preconditioner` and wrote it on a copy — the copy
of a private field of another object, which is the worst kind. The solver now
exposes `preconditioner` and `resolved_for(operator)`, the second returning a
clone with a deferred preconditioner already built, or itself when there is
nothing to resolve. The copying belongs to the class that owns the field.

`Bayesian.evidence_terms` called `prior_data._weighted_squared`, reaching for
the private method because the public `mahalanobis_squared` raises without a
precision — and a prior predictive covariance `A Q A* + R` has none, being
assembled rather than given with an inverse.

The first fix was to give the public method the private one's dense fallback.
That broke a test asserting the opposite, and the test was right: *no precision
means no Mahalanobis distance* is a deliberate contract, so a cubic cost is
never incurred by something that looks like a quadratic form. Weakening a
guard to reach a caller is the wrong direction.

The real fix was in the caller. `evidence_terms` forms the Galerkin matrix
already, for the log-determinant of its volume term — so it takes the misfit
from the same assembly, preferring the precision when one exists. One matrix
now serves both terms where it previously served one and a private method
computed the other.

### 25.3 A wasted solve in every search

`monotone_root` widens the bracket in both directions from a common starting
multiplier, and each walk probed that multiplier itself. A probe is a *solve*.
Every search since the primitive was written had paid one for nothing, and the
`evaluations` count was reporting it. The start is now probed once and handed
to both walks.

### 25.4 Imports and types

Function-level imports of `scipy.sparse`, `random_eig`, `random_svd`,
`random_cholesky` and `CGSolver` were hoisted, having been written where they
were used rather than where they belonged; `numerics`, `algebra` and `geometry`
import neither `inference` nor `probability`, so there was never a cycle to
avoid. What remains at function level in `probability/gaussian.py` and
`inference/backus.py` is older code and left alone.

`Any` was standing in for `AffineSubspace`, `GaussianMeasure`, `SymmetricSpace`
and `Traits` in signatures where the real type was available — the symmetric
space behind `TYPE_CHECKING`, since it is a heavy import for a hint.

Left as it is, and worth a decision rather than a drift: the top-level
`pygeoinf2/__init__` exports algebra, geometry, symmetric spaces, probability
and traits, but not `inference` or `numerics`, so an inversion is reached as
`from pygeoinf2.inference import ...`. v1 exported everything flat.

## 26. Evidence without assembling anything

The evidence calculation was dense throughout: `evidence_terms` formed the
Galerkin matrix of `A Q A* + R`, took its `slogdet`, and solved against it with
`np.linalg.solve`. That confines model comparison to problems small enough to
assemble, which is not where model comparison is interesting. v1 was
matrix-free on both halves and v2 had lost it.

### 26.1 The two halves come apart differently

**The misfit** needs no new machinery, only the solver the estimator already
has. `<v, N_d^-1 v>` is one solve of the normal equations, read off as an inner
product — so it inherits whatever preconditioner was supplied, for free. In the
model-space formalism the data-space inverse is avoided outright by Woodbury:

```
<v, N_d^-1 v> == <v, R^-1 v> - <A* R^-1 v, N_m^-1 A* R^-1 v>
```

which is the point of that formalism, since the data space is the large one
there.

**The log-determinant** is the part that looks as though it needs the matrix.
It does not: `log det A == tr(log A)`, `log(A) z` is a Lanczos iteration on the
Krylov space of `z`, and Hutchinson's estimator turns a handful of those into
the trace. `numerics.functional_calculus.log_determinant` does both routes
behind one signature and returns an `Estimate`, so the dense route reports a
standard error of zero and a caller can treat them uniformly while still seeing
which it got.

**Which determinant** is the usual question in this library, and has the usual
answer: the *component* matrix's, since `det(G A_c) == det G det A_c` and only
the first factor out is a property of the operator. The dense route subtracts
the metric's own determinant; the stochastic route needs no correction at all,
because `random_trace` probes with white noise *on the space* and so estimates
`tr A_c` already. The two agreeing on a weighted and a dense-metric space is
what pins that, and neither route can be adjusted to match the other.

### 26.2 Sylvester, so the model space stays in the model space

Estimating `log|N_d|` by Lanczos still applies `N_d`, which in the model-space
formalism is the operator that formalism exists to avoid. Sylvester's identity
removes it:

```
|A Q A* + R| == |Q| |R| |Q^-1 + A* R^-1 A|
```

taking `X == R^-1 A` and `Y == Q A*` in `det(I + XY) == det(I + YX)`. So the
data-space operator is never formed even to take its determinant. The cost is
two further log-determinants, of `Q` and `R`, which are usually the cheap ones:
a prior with a known spectrum, a diagonal noise covariance. Their errors add in
quadrature, which the returned `Estimate` reports.

### 26.3 What was checked

Four routes to one number — two formalisms times two determinant routes —
against a dense `scipy.stats.multivariate_normal` reference built
independently, with the `sqrt(det G)` that turns a density in components into a
density on the space:

```
                     dense reference  -18.538149
  data_space   dense  -18.538149      stochastic  -18.416331   (logdet +/- 0.163)
  model_space  dense  -18.538149      stochastic  -18.510830   (logdet +/- 0.113)
```

The dense routes agree with the reference and with each other to every digit
printed, which is Sylvester's identity and the metric handling together. The
stochastic routes agree within a standard error.

The stochastic tests are written in units of the estimator's own standard
error — four of them — rather than in a fixed tolerance. A Hutchinson estimate
converges as `1/sqrt(n)`, so `4 sigma` is a statement about the estimator and
`1e-6` would be a statement about the seed.

And the misfit through a preconditioned conjugate-gradient solve equals the
misfit through a Cholesky factorisation to eight figures, which is the whole of
the matrix-free claim on that half.

## 27. Two defaults that were wrong

Two of my choices deviated from v1 without cause, and both were defaults —
which is the worst place for an unwarranted choice, because a default is what
happens when nobody is paying attention.

### 27.1 Direct solvers

`LinearGaussianInversion`, `LeastSquares`, `MinimumNorm`, `DiscrepancyPrinciple`
and `TikhonovFamily` all defaulted to `CholeskySolver`. A Cholesky
factorisation forms the matrix: `O(n^2)` memory and `O(n^3)` time, and one
application per column of the operator merely to assemble it. That is the right
tool for a small problem and unusable for anything else, and choosing it by
default says the library expects small problems.

**Matrix-free is preferable unless the use is obviously restricted to small
problems, and so iterative solvers, not direct ones.** The defaults are now
`CGSolver`. A direct solver remains a keyword argument away, and is still the
right choice where it is: `BackusInference._property_pseudo_inverse` factorises
its property-space normal operator because that space is a handful of rows
by construction (§22.11), and a test that wants to isolate a derivative from
solver noise should say `solver=CholeskySolver()` and mean it.

The change is not free, and both of its costs are worth recording.

**It found a singular system a direct solver had been hiding.** An undamped
`ConstrainedLeastSquares` assembles `(A P)(A P)*` on the data space, whose rank
is at most the subspace's dimension — so it is singular whenever the subspace is
smaller than the data space, which is the ordinary case. Measured on the
existing fixture: eigenvalues `[0, 0.070, 0.481, 7.610]`, condition `6.2e17`.
Cholesky returned an answer. It satisfied the constraint, because the projector
enforces that structurally, and it was the solution of a singular system in
every other respect. Conjugate gradients refuses, and its message says the
operator claims positive definiteness and does not have it — which is exactly
what happened.

**It broke an example, correctly.** Example 22 couples a density in kg/m^3 with
a traction in Pa, so the normal operator's eigenvalues span `9e12` to `7e18`:
badly *scaled* rather than badly conditioned — the condition number is only
`7.5e5` and every eigenvalue is positive. Unpreconditioned conjugate gradients
loses orthogonality on that and reports a residual of `nan`. A diagonal
preconditioner is the entire fix, because a scaling problem is what a diagonal
preconditioner is for, and the example now says so. That is a better example
than one that quietly factorised its way past the issue.

### 27.2 "Whichever space is smaller"

The formalism defaulted to `"auto"`, meaning whichever of the model and data
spaces has fewer dimensions. v1 defaults to `"data_space"` everywhere, and v1
is right.

**Data spaces can be large; model spaces are usually larger still.** So the
data space is where the normal equations belong unless there is a reason
otherwise, and the model-space formalism is kept for when there is one — an
overdetermined problem whose precision is cheap, or a surrogate (§23.2), where
it is genuinely the smaller side.

Comparing dimensions was the wrong test in two further ways. It says nothing
about whether the model-space route is *available*: that needs `Q^-1`, and a
function-space prior often has none — a Sobolev covariance is singular in
practice long before it is in theory. And it reads a discretisation's size as
though it were the problem's, so refining a grid could silently change which
algebra runs.

`"auto"` remains, and is now something to opt into rather than the thing that
happens by default.

### 27.3 The discrepancy principle has no answer sometimes

Making the solves iterative also exposed that `DiscrepancyPrinciple` was
returning garbage in a case where it should refuse. When no damping is small
enough to bring the misfit to its target — a two-dimensional subspace against
eight data, say — the bracket walks down until the normal operator is
numerically singular, and the "solution" there is the solution of a singular
system. Its norm was `2.4e5` where the derivative test expected order one.

v1 raises, and that is right: the principle has no solution, and the
least-damped model is not a fallback but a different thing entirely. It now
raises with the misfit it reached and the target it was aiming at, and suggests
the three things that actually help — a lower level, a wider model, or a chosen
damping.

Note that the *other* saturated case is not like this. "Every damping fits" is
a real answer — the data support no structure and the smallest model says so
(§24.1) — and it still returns one. The two ends of the bracket mean different
things, and only one of them is an answer.

### 27.4 A diagonal should not cost a matrix

Four preconditioners read a diagonal as `np.diag(operator.matrix(...))`, which
allocates `dim^2` to keep `dim` numbers. `diagonals(offsets=(0,))` costs the
same applications and `O(dim)` memory. On a data space of any size the
difference is the whole question of whether the preconditioner can be built at
all — and these are preconditioners, so they exist for problems where it is.

### 27.5 How many times will you use the inverse?

Switching the defaults took example 22 from thirty seconds to **525**. The
posterior mean was not the problem: that is one solve either way. The coupling
diagnostic was, because it *forms* three posterior-covariance blocks, which is
some three thousand applications of the inverse normal operator — three
thousand independent Krylov runs where a factorisation would have done one
decomposition and three thousand triangular solves.

So the criterion is not the one the change was made under. "Iterative unless
the problem is obviously small" is right for *choosing a default*, because a
default cannot know how large the problem will be. But at a call site the
question is sharper:

> How many times will this inverse be applied?

Once or a few times — a posterior mean, a misfit, a damping search — and the
iterative solver wins outright, since it never assembles and its cost is
proportional to what it is asked for. Thousands of times, to *form* something
dense, and the factorisation amortises and the Krylov runs do not. That is the
same trade `OperatorFunction` already documents about `f(A)`, arriving from the
other direction.

Example 22 now says so: the inversion is iterative, the diagnostic that builds
matrices calls `with_solver(CholeskySolver())`, and the comment explains that
the choice is about the number of applications rather than the size of the
problem. 85 seconds, and the same numbers.

Worth noticing for its own sake: a dense diagnostic on an iterative inversion
is exactly what `with_solver` is for, and it would not have been expressible
before §23 gave the estimators one.

## 28. Handing an inversion its solver

The solver is a constructor argument, and the good preconditioners are built
*from* the operator being inverted — which does not exist until the inversion
is constructed. That looks circular, and the question was whether it needs a
two-phase construction, or a mutable solver set after the fact.

It needs neither, and the reason is worth stating before the machinery:
**the normal operator does not depend on the solver.** A solver only inverts
it. So the two phases already exist, and the only question is how to reach the
first one conveniently.

Measured, on what actually works today:

```
1. Generic preconditioner, deferred -- needs nothing built first:
   Jacobi                              OK
   Spectral(rank=10)                   OK
   NormalDiagonal (structure-aware)    OK
2. WoodburyPreconditioner(A, Q, R) directly                OK
3. NormalOperator(...) built by hand, then from_normal     OK
4. Build an estimator, read .normal_operator, with_solver  OK  (0.59 ms)
```

### 28.1 Three routes, and which case each is for

**A preconditioner that is itself a `LinearSolver` is already deferred.**
`with_preconditioner` applies it to the operator at solve time, so it is handed
the normal operator without anyone arranging it. That covers every generic
preconditioner and every structure-aware one that reads its factors off the
operator it is given — which is most of them, `NormalDiagonalPreconditioner`
and `LocalisedPreconditioner` included. No sequencing problem exists in this
case, and it is the common case.

**The operator can be built alone.** `NormalOperator(forward, prior, error=...)`
takes no solver, so anything can be built against it before an inversion
exists. The cost is repeating the formalism, which is a chance to get out of
step with the inversion that follows.

**A factory closes the remaining gap.** A preconditioner built from *other*
factors — a surrogate on a coarser space, the tomography case of §23.1 —
cannot be derived from the operator, so it genuinely needs the operator first.
The `solver` argument now also accepts a callable taking the assembled normal
operator and returning the solver:

```python
inversion = LinearGaussianInversion(
    problem,
    prior,
    solver=lambda normal: CGSolver().with_preconditioner(
        WoodburyPreconditioner.from_normal(
            normal.surrogate(forward=coarse_operator, prior=coarse_prior)
        )
    ),
)
```

One expression, no throwaway object, and the preconditioner is built from the
very operator it will precondition. `with_solver` takes the same, so a solver
can still be chosen after looking at the operator.

### 28.2 Why not a mutable solver

A `set_solver` would have been the obvious answer and is the wrong one here.
Everything else in this library returns a new object rather than mutating —
`with_traits`, `with_formalism`, `with_damping`, `with_preconditioner` — and an
inversion whose solver can change underneath it is an inversion whose
`covariance` and `gain`, both already handed out as operators, quietly refer to
a solve that is no longer the one being performed. The estimator is an operator
(§18.7); operators here do not change.

The two-phase construction was the other candidate. It is what route 4 above
already is, it costs 0.59 ms, and it needs no new API — so it is documented as
an idiom rather than built as a mechanism.

### 28.3 What the factory is checked against

That it receives the operator the inversion actually uses — `seen[0] is
estimator.normal_operator`, not an equal one — and that all three routes land
on the same answer as a direct factorisation to 1e-13. A factory returning
something that is not a solver, and a solver that is neither, are refused where
they are given rather than several calls deeper.

## 29. The KL divergence, and what was already wrong in it

The last dense-only calculation. It is often used on low-dimensional spaces,
where dense is right, so the point is flexibility rather than a change of
default.

```
2 D(P||Q) == tr(C_q^-1 C_p) + (m_q - m_p, C_q^-1 (m_q - m_p))
             - dim + log det C_q - log det C_p
```

Three routes, differing only in how the trace and the two determinants are
reached: **spectral** when both covariances are diagonal in the space's own
basis, which is `O(dim)` and exact and covers every invariant measure on a
symmetric space; **dense**, which forms both matrices; and **stochastic**,
which forms nothing — Hutchinson for the trace, stochastic Lanczos for the
determinants (§26.1). `"auto"` takes them in that order, falling to stochastic
above a dimension it will not assemble at.

`kl_divergence` still returns a float. `kl_divergence_estimate` returns an
`Estimate`, because a stochastic answer without its error is uninterpretable
and the exact routes can report an error of zero rather than being a different
shape.

### 29.1 Cruft found on the way in

The spectral branch computed its quadratic term twice. The first computation
was dead — overwritten on the next line — and the replacement carried a term
reading

```python
float(other.mahalanobis_squared(other.expectation) * 0.0) + ...
```

which is identically zero twice over: the Mahalanobis distance of a measure's
own mean is zero, and it was then multiplied by zero. What remained was a call
to `other.precision(shift)`, which requires a precision the branch never said
it needed. Replaced by `other._weighted_squared(shift)`, which is the same
quantity, is what the dense branch already used, and falls back when there is
no precision.

### 29.2 The metric, again, and a test that was wrong before the code was

Checked against a reference written directly from the definition, on a
Euclidean space and a weighted one:

```
euclidean  reference 2.10245 | dense 2.10245 (0.0e+00) | stochastic 2.09800 +/- 0.167 (0.0 sigma)
weighted   reference 5.77116 | dense 5.77116 (8.9e-16) | stochastic 5.80805 +/- 0.049 (0.7 sigma)
```

and the spectral route equal to the dense one to `0.0e+00` on diagonal
covariances.

The first version of that reference disagreed by a factor of five, and the code
was right both times. Two mistakes, both mine and both instructive.

The reference dropped the metric from the quadratic term. `tr(C_q^-1 C_p)` and
the *difference* of the log-determinants are both metric-free — `G` cancels in
each, in the second because it is a difference — but the quadratic is an inner
product on the space and carries `G`. Two of the three terms not needing it is
exactly what makes the third easy to forget.

And the measures were not valid. Building a covariance with
`from_component_matrix` and a symmetric matrix does not give a self-adjoint
operator on a space whose metric is not the identity: self-adjointness wants
`G C_c` symmetric, and a symmetric `C_c` with a non-constant `G` is not. The
stochastic route reported `nan` and a `log` of a negative eigenvalue — which
was the operator telling the truth about itself. `from_derivative_matrix` with a
symmetric matrix is the construction that gives a symmetric Galerkin matrix,
and it is the one used everywhere else for this reason.

## 30. Drawing a measure

v1's `plot.py` is 2164 lines and none of it had been ported. What v2 had was
*field* plotting — which in v1 lived in `symmetric_space/sphere.py`, not in
`plot.py` — so `plot(space, field)`, `subplots`, `plot_points`, `plot_paths`.
The distribution and slice plotting was the O8 row, still Planned.

This is the first half: `plot_densities` and `plot_corner`, from v1's
`plot_1d_distributions` (213 lines) and `plot_corner_distributions` (377).
`SubspaceSlicePlotter` (1467) and `plot_slice` follow separately, being the
larger and more independent piece.

### 30.1 What they take

Both accept a Gaussian, drawn exactly from its mean and covariance, **or any
measure that can be sampled**, drawn from draws by histogram and kernel
density. v1 took a Gaussian and refused everything else.

The addition is not decoration. `push_forward` through a *non-linear* property
map produces a `PushForwardMeasure` — the largest of four cap averages, the
spread between them — which has no covariance however Gaussian the model
posterior is, and whose marginals are visibly skewed. A Gaussian summary of it
would throw away the thing worth looking at. The same applies to a
randomise-then-optimise posterior (§18.7), which has a sampler and nothing else.

The two routes are checked against each other on *one measure*: a linear
push-forward is Gaussian, so both apply, and wrapping the operator as a general
one hides that and forces the sampling branch. They agree to 0.17%, which is
what sixty thousand draws are worth.

### 30.2 The metric, a fifth time

The covariance that a marginal is about is the covariance **of the
components**, `G^-1 C_gal G^-1` — not the covariance operator's component
matrix, which is a different thing by 75% on the weighted space in the test
suite. `as_multivariate_normal` already computed it, so this asks for that
rather than doing it again.

**And it was wrong.** The expression was

```python
np.stack([solve_gram(row) for row in solve_gram(galerkin).T])
```

`solve_gram` takes a *vector*. Handed a matrix it broadcasts, and what it does
then depends on the space: a diagonal metric divides row-wise and the double
application happens to land on `M_ij / (g_i g_j)`, which is right; a dense
metric performs a genuine solve and the same expression produces `M G^-1 G^-1`,
which is 8% wrong and **not even symmetric** — scipy rejected it outright on
one of the two spaces I tried.

The test that should have caught it compared against 5000 samples with a
tolerance of 35%, and the fixture held a Euclidean space and a weighted one —
both diagonal. It could not have failed. It now asserts the formula exactly, on
a dense-metric space as well, with the sampling check kept as independent
corroboration rather than as the test.

That is the fifth metric bug in this library of exactly one kind: an expression
that is right when `G` is diagonal and wrong when it is not. §21.11's
`credible_set`, §22.11's eigendecomposition, §23.4's preconditioner diagonals,
§29.2's KL reference, and now this. The lesson has been learned once per
occurrence and is worth stating as a rule: **a fixture of Euclidean and
weighted spaces cannot tell a metric-correct expression from a metric-naive
one.** Only a non-diagonal Gram matrix can.

### 30.3 Two things worth keeping from v1

**Contours that open until the truth is inside one.** v1 computes the 2-D
Mahalanobis distance to the true value over every pair and draws enough sigma
contours to enclose the furthest. Without it a truth outside every contour
looks the same whether it missed by two sigma or by nine, and the picture reads
as a worse fit than it is — or a better one. Kept, and the same idea widens the
axis limits.

**Priors on their own y-axis.** A prior is usually far wider than the posterior
it is being compared with, so sharing an axis makes the posterior a spike. The
comparison worth seeing is of shape and position, not of height.

### 30.4 A regression I had documented as a design fact

The example first said, in a comment, that the property posterior "is a
covariance with no factor and so cannot be drawn from". That was true of the
code and false of the design, and writing it down as though it were a fact
about the mathematics is the worse of the two mistakes.

v1 attaches a randomise-then-optimise sampler when it builds the posterior
measure, and v2 does too — `_draw_fluctuation` is v1's algorithm written
centred, the fluctuation rather than the draw, with the measure adding the
mean. But it was attached in an override of `LinearGaussianInversion.__call__`,
and `push_forward` builds a plain `GaussianEstimator` from the mean map and the
covariance — so the property posterior lost it. A draw of `T m` is `T` applied
to a draw of `m`; nothing was in the way except where the sampler was kept.

It is now carried *by the estimator*, passed to `GaussianEstimator.__init__`
rather than attached on the way out, and `push_forward` maps it through. A
sampler that only exists on the way out is a sampler `push_forward` silently
drops.

Worth noting what this costs: randomise-then-optimise is **one solve per
draw** — a prior sample, a noise sample, one application of the Kalman gain.
Four thousand draws are four thousand solves, which is §27.5's criterion
arriving from a third direction, and why example 25 factorises.

While checking against v1: `.gain` reproduces v1's `kalman_operator` exactly in
both formalisms, and `.mean_map` is v1's `posterior_expectation_operator`.
v1's `data_prior_measure` and `joint_prior_measure` had no counterpart on the
inversion — they existed on the forward problem — and are now `.data_prior` and
`.joint_prior` as well.

### 30.5 One small addition to the measures

`can_sample` was a `GaussianMeasure` property, so asking any other measure the
question raised `AttributeError` — and a `PushForwardMeasure` answered nothing.
It is now on `ProbabilityMeasure` and returns True there, since `sample` is
abstract on the base and every concrete measure implements it. `GaussianMeasure`
overrides it: a covariance without a factor defines a measure that genuinely
cannot be sampled.

And so do the measures built *on* others. `PushForwardMeasure`,
`_IndependentSum` and `ProductMeasure` delegate to what they wrap, because a
default of True is a promise none of them can make on its own behalf — the
first version of this said True unconditionally, claimed a
`PushForwardMeasure` of an unsamplable Gaussian could be drawn from, and then
raised when asked to do it.

## 31. Gaussian mixtures

A parameterised Gaussian, ``theta -> N(m(theta), C(theta))``, coupled with a
measure on the parameter. The parameter space is low-dimensional in practice
and often finite, and that finiteness is what makes everything closed form.

What it buys is **multimodality**, which a single Gaussian cannot express at
all. "Either the structure is smooth or it is rough, and I do not know which"
is a two-component mixture; a single Gaussian can only say "somewhere between",
which is a statement about a model neither scenario considers likely.

### 31.1 The measure

`GaussianMixture` holds components and weights. Sampling is exact — choose a
component by its weight, draw from it — which is what makes a mixture easy to
sample even where it is awkward to write down. The expectation is the weighted
mean; the covariance is the law of total covariance,

```
C = sum_k w_k C_k  +  sum_k w_k (m_k - mbar) (m_k - mbar)*
```

whose second term is the spread *between* components and is where the
multimodality lives. It has rank at most `K - 1`, so it is built as a low-rank
factor rather than assembled. On the two-component example its leading entry is
an order of magnitude larger than either component's own variance, and dropping
it would turn a mixture into a blur without any error being raised — so the
test asserts both the sampled covariance and that inequality.

The density is a log-sum-exp rather than a sum of exponentials, because the
whole point of a mixture is that one component may be many orders of magnitude
more likely than another at a given point. `marginal_probabilities` answers
"which component did this come from", which is how a mixture does
classification.

Two constructors match the two ways a parameter measure arrives.
`from_family(build, parameters, weights)` takes a finite support directly.
`from_parameter_samples(build, measure, count)` discretises a continuous one by
sampling — a Monte Carlo approximation to an integral, named as one, and
affordable exactly because the parameter space is small.

### 31.2 The inversion, which needed almost nothing new

Under a linear Gaussian likelihood the posterior of a mixture prior is again a
mixture:

```
posterior  sum_k w'_k N(m_k^post, C_k^post)
w'_k       proportional to  w_k p(d | k)
```

Each component is updated by the usual Kalman formulas — one
`LinearGaussianInversion` per component, inheriting everything §23 to §28 gave
them, preconditioning and solver factories included — and the weights are
rescored by each component's *evidence*, which §26 already computes
matrix-free. The whole class is a loop and a softmax.

`LinearGaussianMixtureInversion` is deliberately **not** a `GaussianEstimator`.
That class is a pair of a moving mean and a fixed covariance, and the whole
point here is that the weights depend on the data, so the *shape* of the
posterior does too. A mixture posterior can change which mode it prefers when
the data change; a data-independent covariance is precisely what cannot.

`push_forward` leaves the weights alone. They are decided by the data through
the evidence, and a property map is applied afterwards.

### 31.3 What was checked

Almost everything, against a reference written independently in plain numpy
from the definitions — component posteriors by the Kalman formulas, weights by
each component's marginal likelihood through `scipy.stats.multivariate_normal`.
Agreement is to machine precision rather than to a tolerance: weights to
4.7e-20, means to 1e-15, covariances to 1.7e-16, and the mixture's own evidence
exact to every digit printed. On a weighted model space as well as a Euclidean
one, with the reference reading component covariances as `G^-1 C_gal G^-1`
(§30.2).

The law of total covariance is the one part with no closed-form reference, so
it is checked against three hundred thousand draws, and marked slow.

### 31.4 An example that had to be retuned to be honest

Example 26 first used two scenarios the data could separate outright, and the
posterior weights came out `(0.000, 1.000)` — while the closing text claimed a
visible second mode. A mixture that collapses to one component is a mixture
doing nothing a single Gaussian could not, so the scenarios are now two that
predict nearly the same datum by different means, and the weights come out
`(0.639, 0.361)` with the modes well separated.

That retuning exposed something better than the original point. The truth lies
in the *less* favoured component: scenario 0 has the tighter prior, so it earns
more evidence for the fit it achieves. That is Occam's razor arithmetic
behaving correctly, and it is the argument for keeping both components rather
than choosing — the mixture mean sits between the two lobes, at a point neither
considers likely, which is why the mean of a multimodal posterior is the wrong
thing to quote.

## 32. Slow tests

The suite had reached nine minutes, most of it in a handful of tests whose cost
*is* the point: an iteration count on a badly-scaled problem, a Monte Carlo
estimate checked against its own standard error, a posterior covariance formed
block by block. Shrinking those would make them demonstrate something else.

So they carry a `slow` marker, and the default run excludes it:

```
pytest pygeoinf2            the fast suite       2m57s
pytest pygeoinf2 -m slow    only the expensive   
pytest pygeoinf2 -m ""      everything           9m09s
```

Marked at parametrisation for the examples, rather than skipped inside the
test — a skip written against the `-m` string cannot be selected *by* `-m`, and
the first attempt got it backwards anyway, since `"not slow".endswith("slow")`
is true and nothing was skipped at all.

## 33. Boundary conditions, and an inverse problem on a finite element space

Example 16 showed that a finite element space *is* a Hilbert space once the
mass matrix is the metric. It did nothing with that: one dimension, no boundary
condition, no inverse problem. This is the elaborated version.

### 33.1 A boundary condition is a subspace

The backend had no support for essential conditions at all, which is the gap
that had to close first. The functions vanishing on part of the boundary form a
Hilbert space in their own right, and building *that* is the whole of what a
homogeneous Dirichlet condition does:

* the Gram matrix is the mass matrix's free-free block;
* a bilinear form restricted to the same block is the Galerkin matrix of the
  operator on the subspace;
* a load vector restricted to the free entries is the functional's derivative
  components there.

So `MfemSpace(elements, essential_dofs=...)` and `essential_dofs_of(elements,
attributes=...)`, and everything else in the library applies unchanged —
`check_space`, `check_coordinates`, `check_operator` and `check_traits` all
pass on the constrained space with nothing special done for it.

Vectors stay the length MFEM expects and simply hold zero on the constrained
degrees of freedom. That costs a little memory and buys the thing that matters:
a vector of this space goes straight into a `GridFunction` and can be drawn,
which a vector of free values alone could not. The boundary condition is then
not enforced anywhere — there is nowhere for a non-zero boundary value to be
*stored*, because it is not a coordinate of the space. The example's recovered
source vanishes on the boundary to `0.0e+00` exactly.

The sharpest statement of why this matters is a trait. The pure Laplacian is
**singular** on the unconstrained space — constants are in its kernel — and
positive definite on the constrained one. `check_traits` verifies that rather
than taking it on trust, and the test asserts the smallest eigenvalue on both
sides of the restriction.

### 33.2 The inverse problem

Steady-state diffusion on the unit square with `u = 0` on the boundary, and the
inverse question: given sixteen sensors, where was the heat put in? The unknown
is the source, a function on the same space as the solution.

Three things it exercises that example 16 did not.

**The forward operator is a PDE solve**, `sensors @ inverse_of_the_stiffness`,
and is never assembled. Applying it solves the PDE; applying its adjoint solves
it again, and the adjoint is derived rather than written down.

**Each sensor is a linear form**, so its natural output is a derivative and the
mass solve that turns one into a function stays inside the operator. The rows
of the observation operator are load vectors, handed to
`from_derivative_matrix` — which is §5.6 in the setting where it is most often
got wrong.

**The prior is a differential operator.** With `S` the operator of
`a(u,v) = uv + l^2 grad u . grad v`, the covariance is `sigma^2 S^-1`: smoothing,
self-adjoint, positive definite, and with its *precision* known in closed form,
which almost no covariance is. A factor for sampling comes from a Lanczos
inverse square root, so nothing is ever factorised densely.

The relative error of the recovered field is 0.87, and the example says plainly
that this is not a failure: sixteen numbers cannot determine a field with 361
degrees of freedom. What they determine is a *property* — the total source over
a central square — which comes out at `1.428 +/- 0.136` against a truth of
`1.517`, seven tenths of a standard deviation.

### 33.3 What the order refinement found

The same property, from the same data, at three polynomial orders:

```
order 1:    81 free dofs   +1.4804 +/- 0.0782
order 2:   361 free dofs   +1.4278 +/- 0.1364
order 3:   841 free dofs   +1.4259 +/- 0.1402
```

Orders 2 and 3 agree, which is the claim: the answer is a statement about the
problem rather than about the mesh, and it *can* be, because the mass matrix is
the metric — a norm, an inner product and a covariance mean the same thing on
both spaces, so the prior is the same prior rather than a rescaled one.

Order 1 is the interesting row, and it was not what the example was written to
show. It differs, and its error bar is **smaller** than the finer spaces'. That
is not better information: a coarse space cannot represent the sources it is
therefore certain do not exist, so the discretisation is acting as a prior
nobody wrote down. The first draft of the text claimed all three agreed; the
numbers did not, and the honest reading is the better lesson — this whole
arrangement exists to make an implicit prior visible, and here it is visible.
