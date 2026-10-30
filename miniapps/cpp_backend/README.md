# A vector backend in C++

How the app is built on the library is walked through in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## The problem

A field `u` on `[0, 1]` is observed at twelve points, drawn uniformly over
`[0.03, 0.97]`, under Gaussian noise of standard deviation 0.1. The field is
piecewise linear on a uniform mesh, a draw from the prior, so that every
answer can be marked against the truth, and two questions are asked of the
data: the average of `u` over `[0.2, 0.35]` and over `[0.6, 0.85]`.

Every vector in the problem is a `cppvec.Vector`: a `std::vector<double>`
behind a nanobind class with no buffer protocol, no `__array__`, no
`__getitem__` and no arithmetic operators. The library cannot read a
component of it, so when a run completes, the library has deferred every
vector calculation to the backend, and the space counts what it was asked.

**The space is `H^1`.** The inner product is

```
(x, y) = x^T (M + l^2 K) y,       l = 0.08,
```

with `M` the mass matrix and `K` the stiffness matrix: the value and the
derivative, the derivative weighed by a length. The data are values at
points, and a value at a point is a bounded functional on a Sobolev space of
order above one half in one dimension, so on `H^1` and not on `L^2`. Part A
shows it: the norm of the value at the middle settles at 2.5000, which is
`1 / sqrt(2 l)`, the norm of its continuous representer, the Green's
function of `1 - l^2 d^2/dx^2` with natural ends, while in `L^2`
(the same space at `l = 0`) it grows as the root of the number of elements,
from 20.8 at 250 elements to 83.2 at 4000. The forward operator is an
operator on `H^1`; on `L^2` it is not one.

**The prior is a measure on that space.** Its covariance is the Matérn-type
`σ^2 (I + ℓ^2 Δ)^-2` with natural ends and `ℓ = 0.08`, which is of order two
against `L^2` and so of order one against `H^1`'s own metric: its eigenvalues
in `H^1` fall as the inverse square of the mode number, their sum converges,
and the covariance is trace class on the space, its draws functions of it
with nearly half a derivative more than `H^1` asks. `σ` is set so that the
pointwise standard deviation at the middle is 1, rising to `sqrt 2` at the
natural ends.

## The parts

- **A, the backend and its check ladder** (`ladder.py`): the norm of a
  value at a point under refinement in `H^1` and in `L^2`; then
  `check_space`, `check_white_noise`, `check_operator` on the point
  evaluations, the local averages, and the prior's covariance, factor,
  precision and precision factor, `check_traits` on the covariance and the
  precision, and `check_measure` on the prior's samples, every probe drawn
  from the prior. Its figure: the truth, prior draws and the data, and the
  representer of the value at the middle on two meshes, in `H^1` and in
  `L^2`.
- **B, the point answer** (`point.py`): the minimum-norm model under the
  discrepancy principle, smallest in the prior's norm, and its estimate of
  each average with its error budget: the noise it propagates and its bias.
  Its figure: the model against the truth and the data, and each average's
  estimate with two propagated standard deviations, the value it would take
  on exact data (the truth plus the bias) and the truth.
- **C, the Bayesian answer** (`bayesian.py`): the posterior mean, the
  pointwise standard deviation at every node, samples, and the posterior on
  the two averages. Its figure sets the prior against the posterior: mean,
  two standard deviations and samples before and after, and each average's
  marginal before and after, with the truth.
- **D, the set answer** (`sets.py`): the feasible set of the two averages
  with the noise hardened to its 95% ball and the prior to two constraint
  sets it calibrates: its 95% ball in the space's own `H^1` norm, and its
  95% ball in the `L^2` norm, an ellipsoid of `H^1` given by the operator
  `G^-1 M` that represents the `L^2` inner product there, checked on the
  space before it is used. For each, certified support brackets, whether
  the true averages lie in the polytope they bound, and the bound the ball
  alone admits under refinement. Its figure sets each bracket against the
  bound its ball admits alone, before any datum.

Each part prints the tally of what the backend was asked during it, member
by member and operator by operator.

Two scales. `--quick` is 100 elements and runs every part in about 2
seconds; it is what the test suite runs. The full scale is 1000 elements and
runs every part in about 35 seconds. Both are seeded; the points are drawn
before the truth, so they are the same at both scales and the truth is not.

## How to build and run

```
cd miniapps/cpp_backend
python build.py                          # cmake + nanobind, module copied into src/
python build.py --check                  # only says whether the module imports
MPLBACKEND=Agg python run_all.py         # every part, full scale
MPLBACKEND=Agg python run_all.py --quick # every part, small scale
MPLBACKEND=Agg python sets.py            # one part; likewise ladder.py,
                                         # point.py, bayesian.py
```

The build needs `cmake`, a C++17 compiler and `nanobind` (a development
dependency of the package), and takes about seven seconds; `build.py`
configures `cpp/CMakeLists.txt` in a scratch directory for the interpreter
running it and copies `cppvec.abi3.so` (`.pyd` on Windows) into `src/`. The
module is built once against the stable ABI and serves every CPython from
3.12 on.

A script runs from any directory, by its path. Every script takes the same
options: `--quick` or `--full` (the default) picks a preset scale, every
field of the scale is then an option that overrides it, and `--outdir` says
where the figures go. `python run_all.py --help`, in short:

```
  --quick | --full      the quick scale, or the full one (the default)
  --outdir OUTDIR       where the figures go (default: results/,
                        results/quick/ at the quick scale)
the scale, each default the full preset's:
  --seed int            the seed of the points, the truth and the noise (2026)
  --n-elements int      the elements of the uniform mesh of the interval (1000)
  --n-points int        the number of observation points (12)
  --noise-std float     the standard deviation of the noise on each datum (0.1)
  --space-length float  the length l of the space's inner product, above zero
                        for a value at a point to be bounded (0.08)
  --prior-length float  the correlation length of the prior (0.08)
  --prior-std float     the prior's pointwise standard deviation at the middle (1.0)
  --level float         the credibility of the discrepancy level and the sets (0.95)
```

Two overrides at the quick scale, against its own numbers:

- **The space's length**, `--space-length 0.04` or `0`: parts B and C print
  the same digits, because the minimum-norm model in the prior's norm and
  the posterior are the prior's and the data's, whatever the space; the
  truth is the same draw too. What changes is part A's representer, whose
  norm at the middle settles at 3.5 at length 0.04 and does not settle at
  length zero, and part D's `H^1` ball, a ball in the space's metric: the
  bracket on the first average under it is `[-0.326, +2.623]` at 0.08,
  `[-1.229, +3.014]` at 0.04 and `[-3.754, +3.908]` at length zero. The
  `L^2` ball is the same set at every length, and its bracket is
  `[-3.754, +3.908]` at all three: at length zero the space is `L^2`, and
  the two balls are one.
- **A noisier survey**, `--noise-std 0.3`: the same points and truth with
  three times the noise. The posterior standard deviation within a quarter
  of the prior's length of a point rises from 0.20 to 0.32 at most, and
  the first average's from 0.147 to 0.231; the discrepancy principle
  chooses a damping of 9.4 rather than 21.3, and the minimum-norm
  estimate's two propagated standard deviations grow from 0.10 to 0.24;
  the certified brackets under the `H^1` ball widen from 2.9 to 3.8 and
  from 2.4 to 3.2, and still hold the truth.

A part run on its own builds the problem from the seed, so it gives the
same numbers as in the full sequence. Each part writes its own figure,
titled with the app and the part and showing one kind of answer, into
`results/` (`results/quick/` at the quick scale), which git ignores. The
test is `tests/test_miniapp_cpp_backend.py`: the space, its
operators, the prior and the three answers on a small mesh in under a
second, and, marked `slow`, every part at the quick scale through its
script's entry point.

| file | what it is |
|---|---|
| `cpp/cppvec.cpp` | `Vector` (size, copy, axpy, scale, dot, weighted dot, a seeded normal fill, `to_list`/`from_list`), `SparseMatrix` (CSR from triplets; apply, apply transposed) and `CholeskyFactor` (banded; the solve, the factor and its transpose, their inverses, the log-determinant) |
| `cpp/CMakeLists.txt` | `find_package(nanobind)` and `nanobind_add_module(cppvec STABLE_ABI NB_STATIC ...)` |
| `build.py` | drives `cmake` on `cpp/` for the interpreter running it and copies the module into `src/`; `--check` reports whether it imports |
| `src/space.py` | `CppSpace`, `H^1` over `cppvec.Vector`; `point_evaluation_operator`, `local_average_operator`, `l2_form` (the `L^2` inner product as an operator on the space, and its inverse) and `matern_measure`, each from callables with its adjoint through the Riesz map |
| `src/problem.py` | the scale with its two presets, the intervals, `build(scale)`: space, prior, points, truth, data and the two operators |
| `src/common.py` | the command line built from the scale, the tally's report, the figure panels |
| `ladder.py`, `point.py`, `bayesian.py`, `sets.py` | parts A to D, each a script with a `main()` and a `run` the others and the test call |
| `run_all.py` | the parts in order |
| `results/` | the figures a run writes, `results/quick/` at the quick scale |

## The numbers at the full scale

Machine: AMD Ryzen AI 7 PRO 350, Python 3.12. Seed 2026, 1000 elements.

**A.** The space passes `check_space` to 1.2e-13 and its white noise
`check_white_noise` to 0.008 (a statistical check at 0.06); the point
evaluations and the local averages hold their adjoints to 1.8e-13 and
3.5e-13; the prior's covariance, factor, precision and precision factor to
1.1e-12, 1.3e-11, 4.0e-13 and 3.3e-12; the covariance and the precision are
self-adjoint and positive definite to 1.5e-13 and 2.3e-12; and 20 000
samples have the declared covariance. The precision returns a draw through
the covariance to 3.5e-10, and the precision factor through the factor to
1.4e-12.

**B, C and D**, on the two averages:

| average | truth | B: minimum norm, 2 propagated sd | bias | C: posterior mean, 95% interval | D: bracket, `H^1` ball | the `H^1` ball alone | D: bracket, `L^2` ball | the `L^2` ball alone |
|---|---|---|---|---|---|---|---|---|
| over `[0.2, 0.35]` | +1.638 | +1.416 ± 0.096 | -0.201 | +1.614, [+1.324, +1.903] | [-0.373, +3.059] | ±3.721 | [-4.059, +4.090] | ±4.102 |
| over `[0.6, 0.85]` | -0.033 | -0.043 ± 0.096 | +0.006 | -0.052, [-0.281, +0.177] | [-1.488, +1.437] | ±3.247 | [-3.151, +3.149] | ±3.179 |

- **B** (0.3 s): a damping of 30.9 meets the discrepancy principle,
  chi-squared 21.03 against the critical 21.03 after 12 misfit evaluations;
  the model's prior norm is 1.66 against the truth's 31, about the root of
  the 1001 nodes, as a draw's must be: a Gaussian measure's draws do not
  lie in its Cameron-Martin space, and their norm there grows with the
  mesh. The propagated error on the first average, 0.048, says nothing of
  the bias there, -0.20, which only the truth reveals: at a damping of 31,
  where the posterior mean's is one, the model trades fit for size and
  passes below the data inside the interval.
- **C** (2.5 s): the posterior mean fits the data to a chi-squared of 1.11
  on 12; the pointwise standard deviation is at most 0.23 within a quarter
  of the prior's length of a point, against the prior's 1, and 1.28 at the
  left end, where there are no data; the first average's standard
  deviation falls from the prior's 0.928 to 0.148, the second's from 0.857
  to 0.117.
- **D** (30 s): the noise hardened to its ball of radius 0.459, and the
  prior to its 95% ball in `H^1`, radius 1.944, in which the truth lies at
  0.42 of the boundary's gauge `(||u|| / r)^2`, and to its 95% ball in
  `L^2`, `u^T M u <= r^2` with `r = 1.590`, in which it lies at 0.27. Under
  either the brackets hold the truth, each to a gap below 1.2e-3, and the
  polytope of the four half-spaces contains the true averages. The form
  `G^-1 M` and its inverse hold their adjoints in the space's inner product
  to 3.1e-11 and 7.6e-13, and their claims to be self-adjoint and positive
  definite to 9.4e-14 and 1.3e-12.

  The two balls bound different things. Alone, the `L^2` ball admits more
  than the `H^1` ball on the narrower average, ±4.10 against ±3.72, and
  less on the wider one, ±3.18 against ±3.25: the `L^2` norm does not
  charge a field for its roughness, so it lets more of its budget into the
  fine scales a short average weighs, while its smaller radius binds the
  smoother one harder. With the data the difference is stark. Under the
  `H^1` ball the twelve values narrow the bounds to under half, brackets
  3.4 and 2.9 wide against 7.4 and 6.5, still wider than the posterior's
  intervals of 0.58 and 0.46, as a set bound is. Under the `L^2` ball they
  barely narrow them, brackets 8.15 and 6.30 wide against 8.20 and 6.36: a
  value at a point is not a bounded functional on `L^2`, so a model can
  meet the data with spikes a few elements wide at the points, at an `L^2`
  cost that vanishes with the mesh, and keep the rest of its budget for
  the averages. At 100 elements the `L^2` brackets are 94% and 90% of the
  bounds alone, at 1000 elements 99%. It is part A's lesson again, in the
  set answer: the data are posed on `H^1`, and a constraint that does not
  see the derivative cannot use them.

  Both bounds alone settle under refinement: on the first average 3.709,
  3.717 and 3.721 in `H^1` and 4.094, 4.098 and 4.102 in `L^2`, at 250,
  500 and 1000 elements, a draw's squared norm having a finite expectation
  in either. The two spectra for the radii take 12 and 13 s, the four
  directions under each a fraction of a second, and the refinement 5 s.

**The tally**, the calls the backend answered during each part:

| member | A, the ladder | B, the point | C, the measure | D, the set |
|---|---|---|---|---|
| `inner_product` | 120 619 | 2 | 0 | 7 020 856 |
| `axpy` | 218 | 1 | 1 013 | 7 004 696 |
| `copy`, `zero` | 395, 16 | 1, 3 | 2 016, 8 | 6 672, 2 005 |
| `scale_inplace` | 183 | 0 | 1 003 | 6 490 |
| `random`, `white_noise` | 23, 40 136 | 0, 0 | 0, 6 | 12, 2 116 |
| `dim`, `_key` | 11, 3 | 0, 0 | 0, 0 | 22, 0 |
| `riesz`, the Riesz map | 10 | 43 | 3 025 | 319 |
| point evaluation, value and adjoint | 20, 5 | 42, 41 | 3 025, 3 021 | 103, 291 |
| local averages, value and adjoint | 20, 5 | 6, 2 | 6, 4 | 8, 28 |
| prior covariance | 64 | 0 | 2 022 | 4 036 |
| prior factor, value and adjoint | 20 160, 8 | 48, 43 | 1 009, 1 003 | 4 083, 4 033 |
| prior precision; precision factor, value and adjoint | 52; 24, 8 | 0; 2, 0 | 0; 0, 0 | 0; 0, 0 |
| the `L^2` form `G^-1 M`, and its inverse | 0, 0 | 0, 0 | 0, 0 | 4 090, 204 |

The 40 000 draws of A are `check_white_noise`'s and `check_measure`'s. C's
3000 evaluations and Riesz maps are the pointwise standard deviation: the
prior and the posterior pushed through evaluation at the 1001 nodes, which
is a point evaluation operator too. D's seven million inner products are
the price of a space without coordinates: the two spectra for the radii,
the covariance's and that of `L* B L` with `B` the `L^2` form, each to the
full dimension by Gram-Schmidt, three and a half million apiece. What the
library never asked for, at any
scale: a component, a Gram matrix, a dense matrix of an operator on the
model space, a basis vector of it, a pointwise product.
