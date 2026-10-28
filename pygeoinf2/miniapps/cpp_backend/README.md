# A vector backend in C++

## The problem

A random field `u` on `[0, 1]` is observed at twelve points under Gaussian
noise of standard deviation 0.1. The field is piecewise linear on a mesh of
200 elements, its prior is Matérn-like with covariance `σ² (I + ℓ² Δ)⁻²` for
`ℓ = 0.08`, and three questions are asked of the data:

- the posterior (`LinearGaussianInversion`): its mean, its pointwise standard
  deviation, and a few samples;
- the smallest model that fits the data (`MinimumNorm` under the discrepancy
  principle at the 95% level);
- the range of values the average of `u` over `[0.2, 0.35]` and over
  `[0.6, 0.85]` can take, given the data and a norm bound
  (`BackusGilbertParker` with the prior's ambient ball and the noise's
  credible ball), as support brackets.

None of that is new. What is new is that every vector in the problem is a
`cppvec.Vector`: a `std::vector<double>` behind a nanobind class with no
buffer protocol, no `__array__`, no `__getitem__` and no arithmetic
operators. The library cannot read a component of it, so if the run
completes, the library deferred every vector calculation to the backend.
That is the claim this mini-app proves.

## Build and run

```
cd pygeoinf2/miniapps/cpp_backend
python build.py          # cmake + nanobind, Release with -O2, module copied in place
python build.py --check  # only says whether the module imports
MPLBACKEND=Agg python app.py
```

The build needs `cmake`, a C++17 compiler and `nanobind` (a development
dependency of the package); `build.py` configures the `CMakeLists.txt` in a
scratch directory with `-DPython_EXECUTABLE` set to the running interpreter
and `-Dnanobind_DIR` to `nanobind.cmake_dir()`, builds, and copies
`cppvec.abi3.so` (`.pyd` on Windows) next to the sources. It takes about
seven seconds. The app runs the check ladder, the three estimators and
writes `cpp_backend.png` next to the sources, in under a second, printing
the checks' reports and the numbers.

The binding is nanobind rather than pybind11 because nanobind is the
successor to pybind11 by the same author, with a fraction of the per-call
overhead, the compile time and the binary size, and because it targets the
stable ABI: the module is built once as `cppvec.abi3` and serves every
CPython from 3.12 on. A backend is called tens of thousands of times in a
run like this one, so the per-call cost is the number that matters.

## The files

| file | what it is |
|---|---|
| `cppvec.cpp` | `Vector` (size, copy, axpy, scale, dot, weighted dot, a seeded normal fill, `to_list`/`from_list`), `SparseMatrix` (CSR from triplets; apply, apply transposed) and `CholeskyFactor` (banded; `solve`, `apply_factor`, `solve_factor`, `solve_factor_transpose`, log-determinant). About 300 lines. |
| `CMakeLists.txt` | `find_package(nanobind)` and `nanobind_add_module(cppvec STABLE_ABI NB_STATIC ...)`. |
| `build.py` | drives `cmake` for the interpreter running it and copies the module in place; `--check` reports whether it imports. |
| `space.py` | `CppSpace(HilbertSpace)`: the mass matrix is the inner product; `point_evaluation_operator`, `local_average_operator`, `laplacian_operator`, `matern_measure`, each from callables with its adjoint through the mass solve. |
| `app.py` | the problem above, with the check ladder in front and a tally of what the backend was asked. |

## The point

`CppSpace` implements exactly the members `HilbertSpace` leaves abstract —
`dim`, `_key`, `zero`, `copy`, `inner_product`, `axpy`, `scale_inplace` — plus
`random` and `white_noise`, which the checks and the samplers need. The
inner product is `xᵀ M y` with the mass matrix `M` applied by the C++ sparse
matrix; the mass solve, which is the Riesz map of this space, is the C++
banded Cholesky; white noise is `L⁻ᵀ z` with `M = L Lᵀ` and `z` a standard
normal draw made in C++ from a seed. The operators built on top are
`LinearOperator.from_callables` with hand-written adjoints: for a functional
`x ↦ cᵀx` the adjoint is one mass solve, `y ↦ M⁻¹ Σ yⱼ cⱼ`; for the Laplacian
`M⁻¹ K`, which is self-adjoint in the mass inner product; for the prior,
`(I + ℓ²Δ)⁻¹ = A⁻¹ M` with `A = M + ℓ² K` factorized once in C++, the
covariance being its square.

The space tallies every member call. On the run as shipped, the library
asked for:

| member | during the checks | the inversion | the minimum-norm fit | the feasible set |
|---|---|---|---|---|
| `inner_product` | 60 353 | 2 | 2 | 10 034 |
| `copy` | 314 | 413 | 0 | 11 248 |
| `axpy` | 177 | 212 | 0 | 10 079 |
| `scale_inplace` | 143 | 201 | 0 | 1 169 |
| `zero` | 22 | 3 541 | 318 | 1 136 |
| `random` | 20 135 | 3 | 0 | 10 000 |
| `white_noise` | 20 103 | 3 | 0 | 10 000 |
| `dim`, `_key` | 7, 3 | 0, 0 | 0, 0 | 5, 0 |

and of the operators' callables: the point evaluations' action and adjoint
(the forward operator), the local averages' action and adjoint (the property
operator), the Laplacian's action (checked, then unused: the prior holds its
own factorization), the prior's `smooth` (once per factor application, twice
per covariance application) and the mass solve behind every adjoint. The
20 000 `random`/`white_noise` calls are `check_white_noise`; the 10 000 in
the feasible set are the prior's `ambient_ball`, which on a space without
coordinates is taken by sampling. The 3 500 forward-and-adjoint pairs of
the inversion are the posterior covariance's diagonal, probed at 201 nodes
through the data-space formalism.

What the library never asked for: `to_components`, `from_components`,
`apply_gram`, `solve_gram`, `basis_vector`, a Gram matrix, a dense matrix of
any operator on the model space, `multiply` or `sqrt` (the space is not a
`HilbertModule`, so the pointwise standard deviation is read off the
covariance of the posterior pushed through evaluation at the nodes, not from
`sample_pointwise_std`), and no arithmetic on a vector outside the space:
`cppvec.Vector` has no operators, so any such reach would have raised
`TypeError`. The data space and the property space are Euclidean, and the
library does work in components there — the 12-by-12 data-space normal
operator, the feasible set's two-dimensional property space — which is the
right place for it.

Two things the figure shows that are worth knowing. The minimum-norm
estimate is spiky: the norm is the `L²` norm, in which the representer of
point evaluation is nearly a delta, so the smallest model fitting twelve
point values is twelve bumps. That is the correct answer to the question
asked, and the reason the posterior with its `H²`-like prior looks like a
function. And the feasible intervals of the local averages are wide, about
`±6`, because the prior's ambient ball is a crude hardening of an anisotropic
measure (its radius is 2.85 in `L²` while the truth's norm is 1.52) and a
norm bound alone lets a short-interval average be large; the posterior's
two-standard-deviation interval is twenty times narrower because it keeps
the prior's shape. Both are statements about the problem, not the backend,
and both are what the coordinate backends give on the same problem.

## Lessons for a nanobind backend

- **The vector owns its memory and the library owns nothing.** A
  `std::vector<double>` inside a nanobind class is owned by the Python
  object; `copy()` returns a new object by value and there is no aliasing to
  worry about, unlike MFEM's `Vector`, which can wrap memory it does not
  own. If the C++ type ever wraps a pointer, `copy` must deep-copy and the
  space must say who frees what.
- **`axpy` and `scale_inplace` mutate and return.** The library always uses
  the return value, so an in-place C++ method wrapped to return `None` must
  be wrapped again in the space to return the vector; `CppSpace.axpy` does
  that in one line.
- **The inner product lives in C++, and so does the Riesz map.** `dot_weighted`
  takes the Gram matrix as an argument so that one vector type serves any
  metric; the space supplies the mass matrix. The mass solve is a method of
  the C++ factor, and every adjoint in `space.py` is one call to it.
- **White noise is `L⁻ᵀ z`, not `L z`.** With `M = L Lᵀ`, applying the factor
  gives covariance `M` on the nodal values and identity covariance on the
  space needs `M⁻¹`, so the draw is a transposed triangular solve.
  `check_white_noise` catches the other choice; that is what it is for.
- **Only a seed crosses the boundary.** `random` and `white_noise` take a
  NumPy generator and pass it one integer; the draw is made by the C++
  generator and the values never exist in Python. A backend over a
  distributed vector would do the same with a rank-dependent seed.
- **Keep the factor alive on the Python side.** The Helmholtz Cholesky in
  `matern_measure` is held by the closures of the measure's operators, not
  by any C++ object; nanobind keeps it alive as long as the operator does.
  A factor that holds a raw pointer to its matrix (this one copies the band)
  would need `nb::keep_alive`.
- **`to_list` exists for the figure and for building functionals, nowhere
  else.** The check that the library never used it is that it is not in the
  tally.
- **Use the ladder's `measure=`.** Standard normal nodal values are not a
  plausible field; the checks here draw their probes from the prior, which
  is what makes the reported defects meaningful for the problem at hand.
- **nanobind's stricter conversions cost nothing here.** It refuses the
  implicit conversions pybind11 allowed (a float where an int is declared, a
  list where a vector is not declared), which is what one wants of an opaque
  type; NumPy scalars still pass where the space hands one over (`np.float64`
  to `scale`, `np.int64` to `fill_random`), so nothing in `space.py` had to
  change in the port. Only the binding syntax did: `nb::class_`, `nb::init`,
  `nb::arg`, `def_prop_ro`, `NB_MODULE`, and the STL casters included one
  header at a time (`nanobind/stl/vector.h`, `nanobind/stl/string.h`).
