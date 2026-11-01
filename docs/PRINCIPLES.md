# pygeoinf: principles and structure

How the library is built, the conventions it relies on, and what a new
backend or feature has to respect. It describes the package as it is; the
history is in git, and the reasons for individual numerical choices are in
the docstrings.

---

## 1. The one idea

Every space is a Hilbert space that knows its own inner product, and nothing
above it works in components unless the space says it can. Most of the rest
follows from that:

- A **derivative** is a linear functional and a **gradient** is its Riesz
  representer. They are different objects and the library keeps them apart.
  An adjoint code's array `dJ/dm_i` enters as a derivative, and the gradient
  costs one Gram solve.
- A **Galerkin matrix** `G A_c` and a **component matrix** `A_c` are two
  representations of one operator. Anything that takes a matrix asks which.
- An **adjoint** is taken in the space's metric, `G_X⁻¹ A_cᵀ G_Y`, and so is
  not a transpose. On a space with a non-diagonal Gram matrix the two differ.

Vectors are whatever objects a backend provides: arrays, `SHGrid`s, tuples
on a direct sum, MFEM vectors, a C++ type. The same estimator runs on a
Euclidean space, a field on a sphere or a ball, or a finite element space.

## 2. Design rules

### The mathematics

1. **Coordinates are a capability.** `HilbertSpace` needs only `dim`,
   `_key`, `zero`, `copy`, `inner_product`, `axpy` and `scale_inplace`.
   `CoordinateSpace` adds the component map and the Gram actions, and
   `HilbertModule` adds pointwise multiplication. Code that needs more says
   so (`require_coordinates`, `require_module`). On a coordinate space,
   internal arithmetic may run on component arrays, with the metric
   entering through `apply_gram`.
2. **The derivative is primitive.** `Functional.gradient(x)` is the
   derivative's adjoint applied to one. A matrix is always given with its
   form (`from_matrix(..., form=)`), and a functional with whether it is
   derivative components or a representer. There are no dual spaces; the
   adjoint is the only map.
3. **Traits are claims.** Self-adjointness, definiteness and the rest are
   asserted by whoever builds an operator, propagated through sums,
   scalings, adjoints and compositions by fixed rules, and verified only in
   tests (`check_traits`). Structure that carries data, such as diagonal,
   low-rank or sparse, is a class.
4. **Function spaces are taken seriously.**
   - A Gaussian measure on a function space has a trace-class covariance, so
     there is never an identity covariance.
   - Point evaluation needs a Sobolev order that admits it.
   - A density against a discretization's volume measure is refused, since
     it does not survive refinement; the robust object is a density
     relative to a reference measure.

   Each refusal has `unsafe=True` for the case that means it.
5. **The space comes from the problem.** The forward model fixes the model
   space and its inner product. A prior is a measure on that space, with
   its own parameters, and does not redefine the space.
6. **An inverse problem is answered with a point, a measure or a set**,
   according to the uncertainty supplied: none, a prior measure, or prior
   and noise sets. Each answer comes from an estimator that maps data to a
   statement about a property of the model. A set answer reports outer
   bounds with a certified bracket.

   Point estimators are provided as tools, used within the methods that
   carry uncertainty. A point estimator reports its error budget
   (resolution, propagated covariance, bias) and never presents it as a
   posterior. The library offers nothing on choosing an optimal
   regularization parameter.

### Cost and honesty

7. **Matrix-free by default.** A dense matrix is formed when asked for by
   name, or when it is the cheaper route counted in operator applications.
   `matrix()` is free where a matrix is held and costs one application per
   column otherwise.
8. **Costs and approximations are visible.**
   - An operator built from its action refuses an adjoint unless
     `with_probed_adjoint()` asks for one to be assembled.
   - A stochastic quantity comes back as an `Estimate` with its standard
     error.
   - A Gauss-Newton Hessian is asked for by name.
9. **Strict by default.** A solve that does not converge raises
   `ConvergenceError` (`strict=False` downgrades it to a warning), and a
   non-finite residual always raises. A refusal names the alternative.
10. **Tolerances are relative** to a scale the problem supplies, so that a
    result does not depend on the units.
11. **Configuration returns new objects.** Solvers are stateless and return
    their diagnostics from `solve`. `with_solver`, `with_traits` and the
    other `with_*` methods return new objects, so a handed-out covariance or
    Kalman operator keeps referring to the solve that produced it.
12. **Parallelism goes around operators, not inside them:** `n_jobs=` on
    loops of independent, expensive applications, all through
    `parallel_map`.

### Structure and practice

13. **Layering runs one way.** `algebra`, `numerics` and `geometry` never
    import `probability` or `inference`.
14. **Optional dependencies are extras**, and the core works without them.
15. **Names.** Use American spelling in identifiers, and one name per
    concept: `max_iterations` is a budget going in and `iterations` a count
    coming out; `rng`, `n_jobs` and `solver` mean the same thing everywhere.
16. **House rules, enforced by `tests/test_code_practice.py`:**
    - every public class and function has a docstring;
    - every parameter and return is annotated;
    - every optional argument is keyword-only;
    - nothing in the code cites a planning document or development history.

    A docstring is self-contained: what is done, what its terms mean and,
    where a caller needs it to choose well, why.

## 3. The package map

| module | what lives there |
|---|---|
| `algebra` | `HilbertSpace`, `CoordinateSpace`, `HilbertModule`, `MassWeightedSpace`, `DirectSum`; `Operator`, `LinearOperator`, `Functional`, the expression nodes, `DiagonalLinearOperator`, the block operators, `Linearization` |
| `traits` | the trait flags and their propagation rules |
| `numerics` | linear solvers and preconditioners, the normal operators and the Tikhonov family, randomized linear algebra, the operator calculus and log-determinants, optimization and line searches, convex methods, quadratic programming, root finding, weighted chi-square distributions |
| `geometry` | convex sets, level and sublevel sets, subspaces and projectors, support functions, the feasible property set and its algorithms |
| `probability` | `ProbabilityMeasure`, `GaussianMeasure`, `GaussianMixture`, `AbsolutelyContinuousMeasure`; products, push-forwards, conditioning, credible sets |
| `inference` | forward problems and their likelihood potential; point, measure and set estimators, linear and nonlinear |
| `symmetric_space` | spaces in the eigenbasis of the Laplacian: the sphere; the periodic box (circle, torus); the bounded box (line, plane); scattered evaluation, gradients at points, the ray travel-time operator |
| `spectral_element`, `interval`, `ball`, `disk` | spaces in the eigenbasis of `1 − div(L² grad)` on a padded spectral-element mesh, for a varying length scale: an interval (and radial profiles), a ball or shell, a disk or annulus, each with a `Layered` form; `spectral_element` is the shared core over `planetmodel` |
| `plotting` | figures of fields, points, paths, sets and distributions on every geometry |
| `testing` | the `check_*` functions, from the space axioms to affine operators, each returning a `CheckReport` |
| `datasets` | shipped station and earthquake tables and their explicit downloads |
| `parallel` | `parallel_map` and `resolve_jobs` |

Subpackages and the common workflow names are importable from the top level.
Spaces are reached by geometry (`symmetric_space.sphere.Sobolev`,
`interval.Sobolev`), and each geometry exports its own `Lebesgue` and
`Sobolev`. `testing` and `datasets` are imported explicitly. The
spectral-element packages need the `planetmodel` extra.

## 4. Conventions that bite

- **Points.** On a sphere they are `(latitude, longitude)` in degrees. In a
  disk they are `(radius, angle)` with the angle in degrees anticlockwise
  from the `x` axis. Lengths are physical; a circle or torus is unit by
  default, so its coordinate is the angle.
- **Vectors.** A sphere's are `pyshtools.SHGrid` objects, a box's are arrays,
  and a spectral-element space's are values on the padded grid with the
  eigen-coefficients as components.
- **Products stay on the grid**, untruncated, and `truncate` is public. An
  operator reads a grid array through its components.
- **Spherical harmonics** are orthonormal on the sphere of the given radius,
  without the Condon-Shortley phase.
- **`power_measure(p)` takes the `L²` spectrum** on every space;
  `invariant_measure` takes eigenvalues in the space's own metric.
- **A function given to a ball must be regular at its centre**: its
  degree-`l` part goes like `r^l`, so a radial profile is even.
- **An inference result carries its reference measure.** A Bayesian update
  returns a `GaussianPosterior` that also knows the prior, the potential and
  the evidence. Linear algebra on it gives a plain Gaussian.
- **A forward model that cannot evaluate raises `EvaluationError`.** Line
  searches back off, and an optimizer returns its last good state with
  `failure` set.
- **Solver defaults** are `rtol=1e-8`, `strict=True`, and conjugate
  gradients unless a solver is passed. Normal equations are solved in the
  data space; `solver=ModelSpace(...)` solves them in the model space, for a
  finite-dimensional problem or a surrogate.

## 5. Writing a backend

A backend supplies a `CoordinateSpace`: `dim`, `_key`, `to_components`,
`from_components`, `axpy`, `scale_inplace`, `copy`, `apply_gram` and
`solve_gram` unless the basis is orthonormal, and `random` and
`white_noise` (white in the space's inner product). `ArrayVectorMixin`
supplies the vector algebra for array-backed vectors. A coordinate-free
backend supplies a `HilbertSpace`, or a base space and a mass operator
through `MassWeightedSpace`. What it cannot be asked for refuses by name.

Check it in this order, stopping at the first failure:

```python
check_space(V, rng=rng, rebuild=lambda: build_again())
check_coordinates(V, rng=rng)
check_white_noise(V, rng=rng)
check_operator(A, rng=rng, measure=prior)
check_traits(A, rng=rng)
```

Each check takes the `rtol` it enforces and reports the worst relative
defect it measured. On a function space, probe with a smooth measure. An
adjoint consistent only to discretization level passes at the `rtol` it can
meet, and its defect should fall under refinement.

Wrapping foreign memory: know who owns each buffer, and keep wrapped solvers
alive from the pygeoinf object that owns them. A finite element space fits
directly, because its mass matrix is the Gram matrix. `miniapps/mfem/` and
`miniapps/cpp_backend/` work through both cases.

## 6. Testing

- **The metric rule.** A metric-sensitive test runs on a non-diagonal Gram
  matrix (`make_dense_metric_space`). The common bug is right on a diagonal
  metric and wrong otherwise.
- **Two routes to one number**, with a negative control for each protected
  convention. An iteration count alone proves nothing.
- **The coordinate-free claim is tested** with adversarial doubles
  (`tests/doubles.py`) and over vectors with no array behaviour
  (`tests/test_no_array_backend.py`).
- **Tolerances sit above the arithmetic.** Statistical checks compare
  against their own standard error, and nothing asserts on exact iteration
  counts.
- **Expensive tests carry `slow`**, and the default run excludes them. The
  suite caps BLAS threads (`tests/conftest.py`), and plotting runs with
  `MPLBACKEND=Agg`.
