# Finite elements through MFEM, two ways

How the app is built on the library is walked through in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## What it is

[MFEM](https://mfem.org) is a finite element library; PyMFEM is its Python
binding. This app wraps it for `pygeoinf` in the two ways a foreign library
can meet the library, and runs an inverse problem through each.

**Form 1: an MFEM function space as a `HilbertSpace`.** In a finite element
space the inner product is `(u, v) == u^T M v` with `M` the mass matrix, so
the mass matrix is the Gram matrix, an assembled bilinear form is a Galerkin
matrix and an assembled linear form is a derivative, whose gradient is a mass
solve. The vectors are MFEM's own `mfem.Vector` objects. Two arrangements:

- `src/matrix_space.py`, the matrix form: `MfemSpace` is a `CoordinateSpace`
  whose components are the degree-of-freedom values and whose Gram matrix is
  the mass matrix read out of MFEM into SciPy and factorized there once.
  Forms become operators through their matrices, and the library does the
  linear algebra. Right for a serial problem of modest size.
- `src/hilbert_space.py`, the Hilbert-space form: `MfemHilbertSpace` is a plain
  `HilbertSpace` (the library's `MassWeightedSpace`) with no coordinates.
  The inner product is MFEM's mass operator, the mass solve MFEM's conjugate
  gradients, forms become operators through `FormSystemMatrix`, partially
  assembled or not, and white noise is MFEM's own integrator. The library
  composes and never sees a matrix; nothing asks whether a vector is local or
  distributed, which is what an MPI space would need (not run here).

**Form 2: an operator between the library's own spaces, through MFEM.**
`src/box_operator.py`: the space is the library's periodic box, a Sobolev space
of grid arrays, and MFEM is only inside the action of an operator on it, the
screened Poisson solve `(1 - s^2 Laplacian) u = f`. The field is integrated
against MFEM's basis functions at MFEM's quadrature points (the weak form's
load, the field read there by the box's non-uniform transform), MFEM solves
on a periodic mesh of the same box, and the finite element solution is read
back at the box's grid. The adjoint is written from the pieces: the transpose
of each transfer, with the box's inverse Gram factor on the left and its Gram
factor on the right, around MFEM's own solve again, the system being
symmetric. The library never sees an MFEM space, an MFEM vector or a finite
element, and the check ladder proves the adjoint.

The parts build up from the first form to the second:

- **A, the Hilbert-space form** (`hilbert_form.py`): `MfemHilbertSpace` on
  partially assembled operators over the unit square with the source zero on
  the boundary; its axioms, a refused `matrix()`, the inner product against
  the matrix form's, the true adjoint `M^-1 K^T` of a non-symmetric
  convection form, the Laplacian and MFEM's inverse of it; then the heat
  source behind sixteen sensors under a Matérn prior built from MFEM's
  solves and white noise, and the same data through the matrix form. Its
  figure: the truth, the posterior mean and its difference from the matrix
  form's, and the total source's marginal before and after.
- **B, the matrix form** (`matrix_form.py`): `MfemSpace` on an interval; its
  axioms and coordinate map, a linear form's load vector against its
  representer, a bilinear form as the Galerkin matrix of its operator, and
  the library's conjugate gradients on it against a direct solve.
- **C, an inverse problem on the matrix form** (`matrix_inverse.py`): steady
  diffusion on the unit square, `u = 0` on the boundary as a subspace on
  which the Laplacian is positive definite, the forward operator the sensors
  composed with MFEM's solve; the posterior on the total source over the
  central square, and the same data at polynomial orders one to three,
  drawn as the prior's and the posterior's 95% intervals at each order.
- **D, the box operator and its check ladder** (`box_ladder.py`): the
  operator's three steps and their composition each checked; the plain
  transpose without the Gram factors refused; the operator against the exact
  Fourier multiplier it discretizes under mesh refinement; the adjoint
  defect against MFEM's tolerance.
- **E, an inverse problem on the box operator** (`box_inverse.py`): the
  source `f` behind noisy readings of `u` at scattered sensors, none inside a
  disc of radius 0.18 about (0.7, 0.3). The source lies in the box's `H^2`,
  on which a value at a point exists, and the prior is a Sobolev measure of
  order 2 on it, a measure of extra regularity whose covariance is trace
  class there; the minimum-norm estimate under the discrepancy principle,
  smallest in the prior's norm, the posterior, and the feasible set's
  certified brackets under the prior's 95% ball, on two disc averages
  of `f`, one among the sensors and one in the gap. A figure for each: the
  estimate with its error budget, the posterior against the prior (the
  pointwise standard deviation before and after, each average's marginal
  before and after), and each bracket against the bound the prior's
  ball admits alone.

Two scales. `--quick` runs every part in about 10 seconds and is what the
test suite runs: the unit square in 6 by 6 elements, the box on a 32 by 32
grid with 24 by 24 quadratic elements, length scale 0.1, screening length
0.05 and 150 sensors. The full scale runs them in about 75 seconds: the
unit square in 10 by 10 elements, the box on a 64 by 64 grid with 48 by 48
quadratic elements (9216 degrees of freedom), length scale 0.05, screening
length 0.025 and 600 sensors. Both are seeded.

## How to run

PyMFEM is the package's `mfem` extra, whose wheels are built for Linux only:

```
poetry install --extras mfem
cd miniapps/mfem
MPLBACKEND=Agg python run_all.py            # every part, full scale
MPLBACKEND=Agg python run_all.py --quick    # every part, small scale
MPLBACKEND=Agg python box_ladder.py         # one part; likewise hilbert_form.py,
                                            # matrix_form.py, matrix_inverse.py,
                                            # box_inverse.py
```

A script runs from any directory, by its path. Every script takes the same
options: `--quick` or `--full` (the default) picks a preset scale, every
field of the scale is then an option that overrides it, and `--outdir` says
where the figures go. `python run_all.py --help`, in short:

```
  --quick | --full      the quick scale, or the full one (the default)
  --outdir OUTDIR       where the figures go (default: results/,
                        results/quick/ at the quick scale)
the scale, each default the full preset's:
  --seed int            the seed of every part's draws (2026)
  --divisions, --line-elements, --orders
                        parts A to C's meshes and part C's orders (10, 32, 1 2 3)
  --conductivity float  part C's conductivity (1.0)
  --source-strength, --smoothness, --correlation-length
                        parts A and C's Matern prior (4.0, 1.0, 0.12)
  --sensor-width, --noise-std
                        parts A and C's sensor windows and noise (0.06, 0.0002)
  --box-grid, --box-elements, --refinements
                        the box's grid and MFEM's meshes on it (64, 48, 24 48 96)
  --box-order, --length-scale
                        the box's Sobolev order and length scale (2.0, 0.05)
  --box-prior-order, --box-prior-std
                        part E's prior on the box, its order above one (2.0, 1.0)
  --screening-length    the box solve's screening length (0.025)
  --sensors, --box-noise-std, --gap-radius, --region-radius
                        part E's survey and questions (600, 0.05, 0.18, 0.08)
  --level float         part E's discrepancy level and sets (0.95)
  --std-samples int     part E's posterior samples (48)
```

Two overrides of part E at the quick scale, against its own numbers
(`box_inverse.py --quick ...`):

- **A noisier survey**, `--box-noise-std 0.2`: four times the noise on the
  same sensors and truth. The posterior standard deviation at the center
  of the disc among the sensors rises from 0.09 to 0.17 and the 95%
  interval on its average widens from 0.23 to 0.53; the minimum-norm
  estimate's two propagated standard deviations there grow from 0.06 to
  0.15, and its certified bracket widens from 2.4 to 3.9. In the gap,
  where the data say less, the standard deviation moves from 0.24 to 0.34.
- **A wider gap**, `--gap-radius 0.3`: the sensors avoid a disc of radius
  0.3 rather than 0.18. They are drawn before the truth, so this is also a
  different draw of the truth. The standard deviation at the gap's center
  rises from 0.24 to 0.52, half way to the prior's 1, and the certified
  bracket on the gap's average widens from 5.9 to 6.5, against the 10.3 the
  prior's ball admits alone; the minimum-norm estimate's bias there
  grows from -0.21 to -0.60.

A part run on its own gives the numbers it gives in the sequence. Each
part's figures are titled with the app and the part, one kind of answer to a
figure, and written into `results/` (`results/quick/` at the quick scale),
which git ignores; on the box's maps the first coordinate runs up the page,
as the library draws a box. The test is
`tests/test_miniapp_mfem.py`, marked `slow` and skipped without
PyMFEM: it tests the three modules class by class, builds the quick scale
through the same command line, and runs every part at it through its
script's entry point.

| file | what it is |
|---|---|
| `src/matrix_space.py` | `MfemSpace`, `essential_dofs_of`, and forms as operators, functionals, solvers, white noise and Matérn measures, through MFEM's matrices |
| `src/hilbert_space.py` | `MfemDofSpace`, `MfemHilbertSpace`, and the same constructions with every computation MFEM's |
| `src/box_operator.py` | `ScreenedPoisson`: the load, the solve and the evaluation as operators, their composition, and the continuum it discretizes |
| `src/common.py` | the command line built from the scale, the scale with its two presets, the figures, the box's space and prior, the sensors of the unit-square problem |
| `hilbert_form.py`, `matrix_form.py`, `matrix_inverse.py`, `box_ladder.py`, `box_inverse.py` | parts A to E, each a script with a `main()` and a `run` the others and the test call |
| `run_all.py` | the parts in order |
| `results/` | the figures a run writes, `results/quick/` at the quick scale |

## The numbers at the full scale

Machine: AMD Ryzen AI 7 PRO 350, Python 3.12, PyMFEM 4.8.0.1. Seed 2026.

**A and C.** The Hilbert-space form passes `check_space` to 3.1e-15, its
inner product agrees with the matrix form's to every printed digit, the
convection form's adjoint passes `check_operator` to 4.5e-14, and MFEM's
inverse of the Laplacian returns its input to 8.0e-11. On the same data the
two forms' posterior means differ by 1.2e-9 relative and their estimates of
the total source over the central square by 1.4e-11: two arrangements, one
answer. In C, sixteen sensors leave the field itself undetermined (relative
error 0.47 on 361 free degrees of freedom) and fix the total source:

| order | free dofs | posterior | 1 sd |
|---|---|---|---|
| 1 | 81 | -0.0425 | 0.0130 |
| 2 | 361 | -0.0397 | 0.0233 |
| 3 | 841 | -0.0396 | 0.0232 |

against a truth of -0.0406. Orders two and three agree; order one's bar is
the narrowest, and that is overconfidence, not information: a coarse space
cannot represent the sources it is therefore sure do not exist.

**B.** The interval's 65 quadratic degrees of freedom pass the axioms and the
coordinate checks to 6.1e-15; the load vector of a window function peaks at
0.021 where its representer peaks at 1.00, the difference a mass solve
makes; the Galerkin operator's adjoint holds to 1.3e-14; conjugate gradients
on it agree with a direct solve to 2.0e-14 in 99 iterations.

**D, the ladder.** The box is `H^2` on a 64 by 64 grid, its Gram weights
running from 1 to 4.1e4. Each MFEM solve takes 47 iterations at the default
tolerance of 1e-14. The adjoint defects `check_operator` measured, with probe
fields drawn from E's prior:

| piece | adjoint defect |
|---|---|
| the load, box to MFEM's loads | 3.6e-15 |
| MFEM's solve | 2.0e-14 (and symmetric to 8.5e-14) |
| the evaluation, MFEM's coefficients to the box | 2.1e-15 |
| the whole operator | 8.5e-10 |
| the plain transpose, Gram factors left out | 0.15, refused |

The two transfers are exact transposes and hold to rounding; the solve and
the whole hold to the solve's tolerance, which the whole amplifies through
the codomain's Gram weights, because the check's codomain probe is white
noise in the box's components. That amplification is what the tolerance
sweep shows:

| MFEM tolerance | 1e-6 | 1e-9 | 1e-12 | 1e-14 |
|---|---|---|---|---|
| adjoint defect | 3.5e-2 | 6.5e-5 | 9.8e-8 | 1.9e-10 |
| iterations a solve | 19 | 29 | 40 | 46 |

Under refinement the operator approaches the exact multiplier
`1 / (1 + s^2 |k|^2)`, and the adjoint the continuum promises (the solve is
self-adjoint before discretization) holds only as well as the discretization
does, while the true adjoint holds at every mesh:

| elements a side | dofs | against the multiplier, grid | in `H^2` | true adjoint | operator as its own adjoint |
|---|---|---|---|---|---|
| 24 | 2304 | 2.3e-3 | 9.9e-2 | 1.6e-12 | 1.5 |
| 48 | 9216 | 3.1e-4 | 7.8e-3 | 2.8e-9 | 0.45 |
| 96 | 36864 | 1.8e-6 | 3.9e-5 | 2.9e-10 | 1.8e-3 |
| 64, aligned with the grid | 16384 | 1.2e-5 | 1.9e-4 | 1.7e-8 | 1.5e-11 |

The last row is why no other mesh here lines up with the box's grid. When the
grid spacing is a whole number of element widths, the mesh is invariant under
the grid's translations and reflections, the discrete operator commutes with
them, and it is a Fourier multiplier and self-adjoint to rounding whatever
the discretization error: the shortcut `A* == A` would pass the check there
and be wrong everywhere else. The continuum's adjoint only comes right once
the mesh resolves the grid's own wavenumbers, at three nodes per grid
spacing; the defect it shows on coarser meshes is the white-noise probe
reaching modes the elements cannot carry.

**E, three answers.** 600 sensors, noise 0.05; the truth's chi-squared is
631 on 600 data.

| average of `f` | truth | minimum norm, 2 propagated sd | bias | posterior, 95% interval | certified bracket |
|---|---|---|---|---|---|
| among the sensors | -0.150 | -0.142 ± 0.037 | +0.011 | -0.137, [-0.184, -0.089] | [-0.93, +0.67] |
| in the gap | +0.233 | +0.395 ± 0.142 | +0.124 | +0.481, [-0.461, +1.422] | [-5.10, +5.40] |

- The minimum-norm estimate, smallest in the prior's norm: a damping of
  7.1 meets the discrepancy principle (chi-squared 658 against the critical
  658), 1226 solves, 18 s. Its two propagated standard deviations in the
  gap, 0.142, say nothing of its bias there, which only the truth reveals.
- The posterior: pointwise standard deviation 0.07 at the center of the
  observed disc and 0.65 in the gap, against the prior's 1; the posterior
  mean fits the data to a chi-squared of 436. 1203 MFEM solves for the mean
  and 97 more for 48 samples, 17 s.
- The feasible set under the prior's 95% ball, `||f|| <= r` in `H^2`'s
  own norm with `r = 1.94`, in which the truth lies at 0.67 of the
  boundary's gauge `(||f|| / r)^2`, and the noise's ball: 1370 solves,
  22 s; the true averages lie in the polytope of the four half-spaces. A
  set bound admits far more than the Gaussian posterior, and the gap's
  bracket is six and a half times the observed disc's: the ball alone
  admits ±7.5 for either average, which the data narrow to a ninth among
  the sensors and to seven tenths in the gap.

## Choices

- **The data space is assembled.** Every answer in E solves in the data
  space. Plain conjugate gradients there take more iterations than there
  are data, each a forward and an adjoint MFEM solve: 217 iterations and 435
  solves for the posterior mean on the quick scale's 150 data, and as many
  again for every sample and every damping. The library's
  `SpectralFamilySolver` assembles the data-space operator once, a column
  per datum (300 solves there), and diagonalizes it, after which a sample or
  a new damping costs two small matrix products. It is passed explicitly, as
  it must be: it is dense in the number of data.
- **The solves are tight.** The default tolerance of MFEM's conjugate
  gradients is a relative residual of 1e-14, which MFEM reaches here in
  under fifty iterations: the adjoint holds only as well as the solve, and
  every method above it inherits the difference.
- **Serial.** MFEM's solver keeps its work vectors between calls, so the
  operator serializes its solves with a lock, and nothing in the app asks
  for worker threads: they would queue at the lock.
- **The prior is a measure on the space, and sets the norm.** In E the
  source lies in `H^2` and the prior is a Sobolev measure of order 2 on
  it, whose covariance is trace class there. The minimum-norm estimate
  measures size in the prior's norm, through the problem parameterized by
  the prior's covariance factor, so that it damps towards what the
  posterior assumes. The set hardens the prior to the ball of `H^2`'s own
  norm that carries 95% of it, the classical norm bound with the prior
  calibrating its radius: the radius has a limit however fine the grid,
  because a draw's squared norm has a finite expectation, the trace of
  the covariance, where the Mahalanobis ellipsoid's radius grows with the
  grid. The ball is a convenience, one constraint set among those the
  physics might suggest; any other convex set with a support function
  serves the same estimator.
- **The solve's claims are checked on probes.** It claims to be
  self-adjoint and positive definite, and `check_traits` tests both on
  probe vectors; the invertibility that positive definiteness implies is
  taken on trust, since verifying it would form the whole matrix, one MFEM
  solve per degree of freedom.
- **What PyMFEM 4.8.0.1 needs worked around**, all in `src/matrix_space.py`
  and `src/hilbert_space.py` and each explained where it is done: reading an
  unfinalized matrix's CSR arrays segfaults, so it is refused; the CSR
  constructor of `mfem.SparseMatrix` double-frees, so matrices are built
  entry by entry; `FormSystemMatrix` takes ownership of the form's matrix, so
  the matrix form never calls it and the Hilbert-space form keeps operator,
  handle and form together; a solver holds raw pointers to its operator and
  smoother, so the object that owns the solver holds them too; the
  white-noise integrator's constructor raises `NameError`, so it is built
  through SWIG directly and kept alive through the assembly; partial
  assembly of a matrix coefficient returns garbage, so the Hilbert-space
  Matérn field uses a scalar coefficient when it is isotropic and refuses
  otherwise; and MFEM's AAA rational approximation is not exposed, so only
  integer Matérn exponents are offered.
