# A vector backend in C++: how it uses the library

The [README](README.md) says what the app asks and what it finds. This
document walks through how it is built on `pygeoinf2`: a vector type from
C++ made into a Hilbert space the library can only call, the operators and
the prior written on it, the solvers and the three answers run on it, and
what the library asked of the backend while doing so. It assumes the
tutorials under `pygeoinf2/tutorials/`, cited by number; tutorial 12 is the
contract and the ladder of checks this app applies. Code is quoted from the
app's own files.

## 1. The spaces

The problem is posed on a space on which its forward operator is bounded,
and the choice is a modeling decision, stated here as one. The data are
values at points, and in one dimension a value at a point is a bounded
functional on a Sobolev space of order above one half: `H^1` is the choice,
with a length `l` that weighs the derivative against the value. The prior
is a Gaussian measure on that space, whose covariance must be trace class
there (section 3); it does not move the problem to a smoother space.

**The backend contract.** `CppSpace` in `src/space.py` is a plain
`HilbertSpace`, not a `CoordinateSpace`: it has no `to_components`, no
`from_components`, no Gram matrix the library can read (tutorial 2:
coordinates are a capability, and this space does not have it). It
supplies the members `HilbertSpace` leaves abstract, `dim`, `_key`,
`zero`, `copy`, `inner_product`, `axpy` and `scale_inplace`, and the two
that every check and every randomized method draws from, `random` and
`white_noise`. Each is one call into C++: `zero` constructs a
`cppvec.Vector`, `copy` is the vector's own copy, `axpy` and
`scale_inplace` mutate in C++ and return the vector, as the library's
in-place contract asks. `_key` is `("CppSpace", n_elements, length_scale)`,
so that two spaces built alike compare equal and an `H^1` and an `L^2` on
one mesh do not.

**The inner product and the Riesz map, in C++.** The mesh's mass and
stiffness matrices are assembled from element matrices as triplets handed
to `cppvec.SparseMatrix`, and the Gram matrix is their combination:

```python
self._gram = _matrix(self._n_elements, 1.0, self._length_scale**2)
```

The inner product is the vector's weighted dot product with it,

```python
return x.dot_weighted(y, self._gram)
```

and the Riesz map, `G^-1`, is the C++ banded Cholesky factor of `G`, built
once with the space:

```python
def riesz(self, b: Any) -> Any:
    """``G^-1 b``: the representer of the functional ``x -> b^T x``."""
    self.calls["riesz"] += 1
    return self._gram_cholesky.solve(b)
```

White noise on the space has the identity as its covariance in the space's
inner product, which on nodal values is `G^-1`; with `G = R R^T` it is
`R^-T z` for standard normal `z`, the transposed triangular solve,

```python
return self._gram_cholesky.solve_factor_transpose(self._standard_normal(rng))
```

and the draw `z` is made in C++ from one integer drawn from the NumPy
generator: the values never exist on the Python side.

At length zero the Gram matrix is the mass matrix and the space is `L^2`.
Part A builds it, and only to show that a value at a point is not bounded
there: the norm of the evaluation functional is the norm of its
representer, `space.norm(evaluate.adjoint(1))`, and the README's table has
it settling at `1 / sqrt(2 l)` in `H^1` and growing as the root of the
number of elements in `L^2`.

The data space and the property space are the codomains of the operators
below, `EuclideanSpace(12)` and `EuclideanSpace(2)`, where the library does
work in components, which is the right place for it.

## 2. The operators and what the checks verified

Both the forward operator and the property operator are rows of
functionals `x -> c_j^T x` on the nodal values: the two nodes of the element
holding a point, weighted by linear interpolation, and the exact integral of
each hat function over a subinterval, divided by its length. Both are
`_functional_operator` in `src/space.py`, built with `from_callables`
(tutorial 3):

```python
def value(x: Any) -> np.ndarray:
    space.calls[f"{name}.value"] += 1
    return data_space.from_components(np.array([c.dot(x) for c in coefficients]))

def adjoint(y: np.ndarray) -> Any:
    space.calls[f"{name}.adjoint"] += 1
    load = cppvec.Vector(space.n_elements + 1)
    for weight, c in zip(data_space.to_components(y), coefficients):
        load.axpy(float(weight), c)
    return space.riesz(load)
```

The coefficients `c_j` are derivative components, not vectors of the
space: the representer of `x -> c^T x` under the inner product `x^T G y` is
`G^-1 c`, so the adjoint is one Riesz map however many functionals. An
adjoint that returned the combination of the `c_j` itself, or took the mass
solve `M^-1` of an `L^2` space, would be wrong by a metric, and
`check_operator` says so.

**What was checked.** Part A (`ladder.py`) runs the ladder of tutorial 12,
every probe drawn from the prior with `measure=prior`: white noise on a
function space is rough in every mode and not the kind of field the
operators meet.

```python
"space": check_space(space, rng=rng, measure=prior, rebuild=...),
"white noise": check_white_noise(space, rng=rng, measure=prior),
"point evaluation": check_operator(problem.evaluate, rng=rng, measure=prior),
"local averages": check_operator(problem.averages, rng=rng, measure=prior),
```

At the full scale the axioms hold to `1.2e-13`, white noise is white to
`0.008` of a statistical tolerance of `0.06`, and the two operators' adjoints
hold to `1.8e-13` and `3.5e-13`. What failures would have meant:
`check_white_noise`, a draw applying `R` rather than `R^-T`, covariance `G`
instead of `G^-1`; `check_operator`, an adjoint missing the Riesz map;
`check_space`'s `rebuild`, a `_key` that forgot the length, so that two
spaces built alike compared unequal.

## 3. The prior and the error measure

The prior is the Matérn-type field `σ^2 (I + ℓ^2 Δ)^-2` with natural ends,
its nodal values the law of `A u = σ M^(1/2) z` with `A = M + ℓ^2 K` the
Helmholtz matrix of the prior's length: covariance `σ^2 A^-1 M A^-1` on the
nodal values. That is a statement about the values and holds whatever the
space; as an operator on the space, whose inner product is `x^T G y`, the
covariance is

```
Q = σ^2 A^-1 M A^-1 G,
```

and `G Q` is symmetric, so `Q` is self-adjoint on the space. Its Galerkin
form depends on the inner product, and that is why the measure is built on
the space rather than borrowed from an `L^2` formula: on `H^1` the
covariance is of order one, its eigenvalues falling as the inverse square of
the mode number, so their sum converges and `Q` is trace class there, which
is what makes the measure one on the space (tutorial 5: a measure of order
above half the dimension, its draws functions of the space).

`matern_measure` in `src/space.py` gives the measure four operators, each a
few banded solves and sparse products in C++ with nothing formed:

```python
return GaussianMeasure(
    space,
    covariance=LinearOperator.self_adjoint(space, covariance, traits=_DEFINITE),
    covariance_factor=LinearOperator.from_callables(
        space, space, factor, adjoint=factor_adjoint
    ),
    precision=LinearOperator.self_adjoint(space, precision, traits=_DEFINITE),
    precision_factor=LinearOperator.from_callables(
        space, space, whiten, adjoint=whiten_adjoint
    ),
)
```

- the covariance, `Q x = σ^2 A^-1 M A^-1 G x`:
  `solver.solve(mass.apply(solver.solve(gram.apply(x))))`;
- the factor, `L = σ A^-1 R_M R_G^T` with `M = R_M R_M^T` and
  `G = R_G R_G^T` the C++ Cholesky factors, a map of the space to itself
  with `L L* = Q`, applied as `solver.solve(mass_factor.apply_factor(
  gram_factor.apply_factor_transpose(w)))`; its adjoint in the space's
  inner product, `L* = G^-1 L^T G`, simplifies to `σ R_G^-T R_M^T A^-1 G`,
  a transposed solve with `G`'s factor where a Riesz map might be expected;
- the precision, `Q^-1 = σ^-2 G^-1 A M^-1 A`;
- the precision factor, `L^-1 = σ^-1 R_G^-T R_M^-1 A`, with
  `Q^-1 = (L^-1)* L^-1` as the library's convention asks: the whitening of
  a model, `w = L^-1 u`, which the point answer's bias needs.

`σ` is set by one Helmholtz solve so that the pointwise standard deviation at
the middle of the interval is the scale's, `c^T A^-1 M A^-1 c` for the
nodal weights `c` of the value there; the natural ends double the variance
within a length or so of either end.

**What was checked.** `check_operator` on all four, `check_traits` on the
covariance and the precision, both claimed self-adjoint and positive
definite, and `check_measure` on 20 000 samples, which compares their
second moments with the declared covariance (tutorial 5). The four hold
their adjoints to between `4e-13` and `1.3e-11` at the full scale; the
precision returns a draw through the covariance to `3.5e-10` and the
precision factor through the factor to `1.4e-12`, printed beside the
ladder, since a wrong precision passes `check_operator` on its own.

The noise is `GaussianMeasure.from_standard_deviation(codomain, 0.1)` on the
Euclidean data space, where an isotropic measure is what it should be. On
the model space it would be the identity covariance, white noise in the
space's metric, which no measure on a function space can have, and the app
never builds it there.

## 4. The solvers

There are twelve data. Every answer comes down to the data-space normal
operator `A C A* + t R` (tutorial 4), with `C` the prior's covariance for
the point and the measure and the shape of the prior's ellipsoid for the
set, and the app passes `SpectralFamilySolver()` to all three estimators: it
assembles the base `A C A*` once, a column per datum, whitens it by `R` and
diagonalizes it, after which any member of the family, at any multiplier,
is two small matrix products. The discrepancy search visits a dozen
multipliers, the pointwise standard deviation pushes a thousand columns
through the posterior covariance, and the set's primal walk visits a family
member at every step, all for the twelve columns of the assembly. It is
dense in the number of data and nowhere else; on the model space nothing is
formed.

The library has no other dense route on this space: `matrix()` on an
operator of it, or anything that would assemble one, needs coordinates the
space does not have. Two places in the set answer pay for that instead with
inner products. The radius of the prior's ellipsoid is a quantile over the
covariance's spectrum, which on a coordinate-free space is the library's
randomized eigendecomposition, `random_eig`, by Gram-Schmidt in the space's
inner product; and the ellipsoid's fractional powers of the
covariance are applied by the Lanczos calculus, each application its own
reorthogonalized Krylov run. Section 6 counts both.

## 5. The estimators

**The point** (part B, `point.py`). Tutorial 8's minimum-norm model under
the discrepancy principle, with size measured in the prior's norm: the
problem is parameterized by the prior's factor, `u = L w`, so that `|w|` is
the prior's norm of `u`:

```python
factor = problem.prior.covariance_factor
whiten = problem.prior.precision_factor
estimator = MinimumNorm(
    forward.parameterized(factor),
    level=scale.level,
    solver=SpectralFamilySolver(),
)
search = estimator.discrepancy_search(problem.data)
chosen = estimator.for_data(problem.data)
change = chosen(problem.data)
smallest = factor(change)
```

The space's own `H^1` norm is the other honest choice; the prior's is taken
so that the point answer damps towards what the Bayesian answer assumes,
with the posterior mean at a damping of one. The model is then the prior's
and the data's alone: the factor's domain is the space, but `|L^-1 u|` does
not depend on the space's metric, which the README's override of the
space's length shows to every printed digit. `for_data` returns the affine
estimator at the damping found, which carries tutorial 14's error budget,
pushed through the factor to the averages:

```python
pushed = chosen.push_forward(problem.averages @ factor)
estimate = np.asarray(pushed(problem.data))
propagated = np.sqrt(pushed.propagated_covariance().diagonals()[0])
bias = np.asarray(pushed.bias(whiten(problem.truth)))
```

**The measure** (part C, `bayesian.py`). `LinearGaussianInversion(forward,
prior, solver=SpectralFamilySolver())(data)` is the posterior (tutorial 6);
`posterior.expectation` its mean and `posterior.samples(3, rng=rng)` three
draws, each a `cppvec.Vector`. The space has no pointwise product, so
`sample_pointwise_std` is not available on it; the pointwise standard
deviation is read exactly instead, by pushing the measure through
evaluation at every node and taking the diagonal of that covariance,

```python
at_nodes = point_evaluation_operator(space, space.nodes())
return np.sqrt(measure.affine_map(at_nodes).covariance.diagonals()[0])
```

for the prior and for the posterior. `posterior.affine_map(problem.averages)`
is the Gaussian on the two averages, drawn against the prior's by
`plotting.plot_densities(posterior, prior=prior, ...)`.

**The set** (part D, `sets.py`). Tutorial 7's feasible property set, the
prior hardened to an ellipsoid of its own shape and the noise to its 95%
ball:

```python
return prior.weakened_ellipsoid(
    level=level,
    power=power,
    method="spectral",
    rank=prior.domain.dim,
    quantile_method="matched",
    rng=rng,
)
```

`{u : (Q^-p u, u) <= r^2}`, with `r` the radius that carries 95% of the
prior, a quantile of `sum lambda^(1-p) Z^2` over the covariance's
eigenvalues. At `p = 1` it would be the Mahalanobis ellipsoid, whose radius
grows with the mesh and which the library refuses on a function space; at
`p = 0` the ball in the space's norm, which forgets the covariance's shape.
In between the radius has a limit when the sum converges: the eigenvalues
on `H^1` fall as `k^-2`, so the weights as `k^-2(1-p)`, and the sum
converges for `p < 1/2`, which is `1 - d / 2s` with `s = 1` the
covariance's order against the space's metric and `d = 1`. The scale's
power is half of that, `1/4`. Part D measures it: the bound the ellipsoid
alone admits on the first average settles at `p = 1/4` and keeps growing at
`p = 1/2`. The quantile is the moment-matched chi-square, from two traces
of the spectrum; and the rank is the full dimension, because weights that
decay as `k^-3/2` leave a tail no truncated spectrum can drop: at 1000
elements the leading 50 eigenvalues give a squared radius of 6.00, the
leading 100 give 6.27, and all of them 6.69.

```python
feasible = BackusGilbertParker(
    forward,
    problem.averages,
    ellipsoid,
    noise=noise_ball,
    solver=SpectralFamilySolver(),
)(problem.data)
sweep = feasible.support_sweep(directions)
```

An ellipsoid and a ball make the set quadratic, and its support is computed
by the primal walk over the data-space family. Each result carries a
`value`, the certified outer bound, and an `inner` value attained by a
feasible model; four directions, plus and minus each average, give the
brackets, and `feasible.polytope(directions).contains(true_averages)` the
check. What the ellipsoid admits with no data is its support function along
each property's representer, `prior_alone` in `src/common.py`.

## 6. What the tally shows

`CppSpace` counts every call of its members in `space.calls`, and the
operators built on it count their callables under their own names
(`point_evaluation.value`, `prior.covariance`, ...); each part clears the
tally when it starts and prints it when it is done, and the README has the
full-scale table. Read from it:

- **The ladder** is statistical: 40 000 white-noise draws for
  `check_white_noise` and `check_measure`, each a C++ fill and a triangular
  solve, and 120 000 inner products to compare their moments.
- **The point answer** is the assembly: 42 applications of the forward
  operator, its adjoint and the factor each, the twelve columns of
  `A L L* A*` and what the error budget pushes through.
- **The measure** is the pointwise standard deviation: three thousand
  evaluations, Riesz maps and covariance applications for 1001 nodes, prior
  and posterior; the posterior itself costs a dozen.
- **The set** is five million inner products and as many `axpy`s, three and
  a half million of them the eigendecomposition to the full dimension and
  the rest the Lanczos runs of the ellipsoid's powers. That is the price of
  a coordinate-free space where the library's routes are spectral; a
  backend with a basis would form a matrix here instead.

What the library never asked for: a component, a Gram matrix, a basis
vector, a dense matrix of an operator on the model space, a pointwise
product. `cppvec.Vector` has none of the faces those would need, and no
arithmetic operators, so any such reach would have raised `TypeError`
rather than passed; the test asserts that the tally holds nothing but the
contract.

The lessons of the binding, for a backend of your own:

- **The vector owns its memory.** A `std::vector<double>` inside a nanobind
  class is owned by the Python object, and `copy()` returns a new object by
  value. A type that wraps a pointer must deep-copy in `copy`, and the space
  must say who frees what.
- **In-place methods return.** The library always uses the return value of
  `axpy` and `scale_inplace`, so a C++ method that returns nothing is
  wrapped to return its vector, one line in the space.
- **The metric lives in C++.** `dot_weighted` takes the Gram matrix as an
  argument, so one vector type serves any inner product; the Riesz map and
  the white-noise draw are methods of the C++ factor of that matrix.
- **Only a seed crosses.** `random` and `white_noise` pass the C++
  generator one integer; a distributed vector would take a rank-dependent
  seed.
- **The closures own the factors.** The Helmholtz factor of the prior is
  held by the closures of the measure's operators, not by any C++ object,
  and lives as long as they do.
- **`to_list` is for the figures.** It, `from_list` for building
  functionals, and the error against the truth taken by the vectors'
  own weighted dot product are the app's, outside the tally.

## 7. What a reader would change

- **The vector type**: a class with copy, `axpy`, scale, a weighted dot
  product and a seeded fill is the whole surface; `CppSpace` is the pattern
  for making it a `HilbertSpace`, then the ladder of tutorial 12 before
  anything else, with `measure=` a prior on the space.
- **The space**: the Gram matrix in `CppSpace.__init__`. Pick it so that
  the forward operator is bounded: the length of `H^1` here, a higher order
  for a derivative at a point. The `_key` must carry whatever the metric is
  built from.
- **The forward operator**: rows of derivative components go through
  `_functional_operator` unchanged, their adjoint one Riesz map; an
  operator into the space would be `from_callables` with its adjoint taken
  through the Riesz map on the other side, and checked the same way.
- **The prior**: the covariance as an operator on the space, which
  carries the space's Gram matrix, with a factor whose adjoint is taken in
  the space's inner product, and its order against the space's metric
  above half the dimension. Measure the point answer by the factor if it
  is to damp as the Bayesian one does, and keep the set's power below
  `1 - d / 2s` with `s` that order.
- **The solver**: `SpectralFamilySolver` while the data are few; a Krylov
  solver with a preconditioner when they are many, as the point-data app
  shows.
