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
  with the prior hardened to its 95% ellipsoid and the noise to its 95%
  ball, as certified support brackets; whether the true averages lie in the
  polytope they bound; and the bound the ellipsoid alone admits under
  refinement. Its figure sets each bracket against the bound the prior's
  ellipsoid admits alone, before any datum.

Each part prints the tally of what the backend was asked during it, member
by member and operator by operator.

Two scales. `--quick` is 100 elements and runs every part in about 2.5
seconds; it is what the test suite runs. The full scale is 1000 elements and
runs every part in about 42 seconds. Both are seeded; the points are drawn
before the truth, so they are the same at both scales and the truth is not.

## How to build and run

```
cd pygeoinf2/miniapps/cpp_backend
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
  --set-power float     the power of the prior's covariance that shapes its
                        ellipsoid, below one half (0.25)
```

Two overrides at the quick scale, against its own numbers:

- **The space's length**, `--space-length 0.04` or `0`: parts B and C print
  the same digits, because the minimum-norm model in the prior's norm and
  the posterior are the prior's and the data's, whatever the space; the
  truth is the same draw too. What changes is part A's representer, whose
  norm at the middle settles at 3.5 at length 0.04 and does not settle at
  length zero, and part D, whose ellipsoid is the covariance to a power in
  the space's metric: the bracket on the first average is `[+0.005,
  +2.505]` at 0.08, `[-0.362, +2.657]` at 0.04 and `[-1.855, +3.150]` in
  `L^2`.
- **A noisier survey**, `--noise-std 0.3`: the same points and truth with
  three times the noise. The posterior standard deviation within a quarter
  of the prior's length of a point rises from 0.20 to 0.32 at most, and
  the first average's from 0.147 to 0.231; the discrepancy principle
  chooses a damping of 9.4 rather than 21.3, and the minimum-norm
  estimate's two propagated standard deviations grow from 0.10 to 0.24;
  the certified brackets widen from 2.5 to 3.4 and from 2.1 to 3.0, and
  still hold the truth.

A part run on its own builds the problem from the seed, so it gives the
same numbers as in the full sequence. Each part writes its own figure,
titled with the app and the part and showing one kind of answer, into
`results/` (`results/quick/` at the quick scale), which git ignores. The
test is `pygeoinf2/tests/test_miniapp_cpp_backend.py`: the space, its
operators, the prior and the three answers on a small mesh in under a
second, and, marked `slow`, every part at the quick scale through its
script's entry point.

| file | what it is |
|---|---|
| `cpp/cppvec.cpp` | `Vector` (size, copy, axpy, scale, dot, weighted dot, a seeded normal fill, `to_list`/`from_list`), `SparseMatrix` (CSR from triplets; apply, apply transposed) and `CholeskyFactor` (banded; the solve, the factor and its transpose, their inverses, the log-determinant) |
| `cpp/CMakeLists.txt` | `find_package(nanobind)` and `nanobind_add_module(cppvec STABLE_ABI NB_STATIC ...)` |
| `build.py` | drives `cmake` on `cpp/` for the interpreter running it and copies the module into `src/`; `--check` reports whether it imports |
| `src/space.py` | `CppSpace`, `H^1` over `cppvec.Vector`; `point_evaluation_operator`, `local_average_operator` and `matern_measure`, each from callables with its adjoint through the Riesz map |
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

| average | truth | B: minimum norm, 2 propagated sd | bias | C: posterior mean, 95% interval | D: certified bracket | the ellipsoid alone |
|---|---|---|---|---|---|---|
| over `[0.2, 0.35]` | +1.638 | +1.416 ± 0.096 | -0.201 | +1.614, [+1.324, +1.903] | [+0.007, +2.955] | ±4.099 |
| over `[0.6, 0.85]` | -0.033 | -0.043 ± 0.096 | +0.006 | -0.052, [-0.281, +0.177] | [-1.287, +1.218] | ±3.641 |

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
- **D** (37 s): the prior hardened to `(Q^-1/4 u, u) <= r^2`, in which the
  truth lies at 0.56 of the boundary's gauge, and the noise to its ball of
  radius 0.459. The brackets hold the truth, each to a gap below 1e-4, and
  the polytope of the four half-spaces contains the true averages. A set
  bound admits more than the Gaussian posterior: the brackets are 2.9 and
  2.5 wide against intervals of 0.58 and 0.46, and the data narrow the
  ellipsoid's bound alone by a factor of nearly three. That bound settles under
  refinement at the scale's power, 4.026, 4.069 and 4.099 at 250, 500 and
  1000 elements, its increments falling by the root of two, and at power
  one half, where the radius has no limit, it keeps growing, 5.02, 5.28 and
  5.53. The covariance's spectrum for the radius takes 11 s and the four
  directions another 11.

**The tally**, the calls the backend answered during each part:

| member | A, the ladder | B, the point | C, the measure | D, the set |
|---|---|---|---|---|
| `inner_product` | 120 592 | 3 | 0 | 5 263 712 |
| `axpy` | 209 | 1 | 1 013 | 5 278 784 |
| `copy`, `zero` | 386, 16 | 1, 3 | 2 016, 8 | 3 299, 1 165 |
| `scale_inplace` | 183 | 0 | 1 003 | 26 424 |
| `random`, `white_noise` | 23, 40 136 | 0, 0 | 0, 6 | 0, 1 001 |
| `dim`, `_key` | 11, 3 | 0, 0 | 0, 0 | 171, 0 |
| `riesz`, the Riesz map | 10 | 44 | 3 025 | 156 |
| point evaluation, value and adjoint | 20, 5 | 42, 42 | 3 025, 3 021 | 52, 144 |
| local averages, value and adjoint | 20, 5 | 6, 2 | 6, 4 | 4, 12 |
| prior covariance | 58 | 0 | 2 022 | 27 365 |
| prior factor, value and adjoint | 20 157, 5 | 48, 44 | 1 009, 1 003 | 0, 0 |
| prior precision; precision factor, value and adjoint | 46; 21, 5 | 0; 2, 0 | 0; 0, 0 | 0; 0, 0 |

The 40 000 draws of A are `check_white_noise`'s and `check_measure`'s. C's
3000 evaluations and Riesz maps are the pointwise standard deviation: the
prior and the posterior pushed through evaluation at the 1001 nodes, which
is a point evaluation operator too. D's five million inner
products are the price of a space without coordinates: the covariance's
spectrum to the full dimension by Gram-Schmidt, three and a half million of
them, and the Lanczos iterations of the ellipsoid's fractional powers, each
reorthogonalized, for the rest. What the library never asked for, at any
scale: a component, a Gram matrix, a dense matrix of an operator on the
model space, a basis vector of it, a pointwise product.
