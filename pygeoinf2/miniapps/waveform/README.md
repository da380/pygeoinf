# Acoustic waveform inversion on a C++ solver

## The problem

A vertical section of rock 2 km wide, its top a free surface, its velocity
increasing linearly with depth from 1.8 km/s at a rate of 0.6 per second,
with three Gaussian anomalies in it: a fast one of +0.25 km/s at 250 m depth,
a slow one of -0.20 km/s at 320 m and a smaller fast one of +0.20 km/s at
460 m. Three shots are fired 20 m below the surface and recorded by a line of
receivers at the same depth for about a second. The unknown is
`m = 1 / c²`, the slowness squared, in s²/km², and the data are the pressure
at every receiver at every time step, with Gaussian noise of 3% of the data's
root-mean-square added sample by sample. The background is known; the
anomalies are what the data are asked for.

The wave equation `m ∂²u/∂t² = ∇²u + f(t) δ(x − x_s)` is solved by a
finite-difference scheme written here in C++: second order in time, fourth
order in space, a Ricker wavelet, a Cerjan sponge 20 cells wide on the sides
and the bottom. The C++ also provides the scheme's linearization, its exact
discrete adjoint and the second-order adjoint, and they are wrapped as one
library `Operator` with a first and a second derivative. On it the library's
nonlinear estimators run unchanged: the MAP model by Newton-CG on the
Gauss-Newton and on the exact Hessian, the Laplace Gaussian at the mode, and
Occam's inversion. Built with MPI, the same solver splits its grid across
ranks, and the operator and the estimators above it do not change.

The parts:

- **A, the solver and its proofs** (`proofs.py`): the stencil's explicit
  transpose, the fused kernels against the unfused ones, dot tests, central
  differences at a ladder of steps, and the library's check ladder, on the
  64 by 32 proof grid and again on the grid of the scale asked for; a
  figure of wavefield snapshots and a shot gather.
- **B, the data** (`synthetic.py`): the truth's data with noise, the misfit
  of the truth and of the background, and the misfit's gradient at the
  background.
- **C, the MAP model** (`newton.py`): Newton-CG from the background with the
  Gauss-Newton Hessian; then, from where that stopped, with the exact
  Hessian; and a short run with the exact Hessian from the background.
- **D, the uncertainty** (`laplace.py`): the Laplace Gaussian at the mode,
  pushed onto the mean of `m` over four boxes, one over each anomaly and one
  over plain background, against the prior's.
- **D, continued** (`occam.py`): Occam's model at the 95% level, against the
  MAP mode and the truth.
- **E, the grid split across MPI ranks** (`mpi_run.py`, under `mpiexec`): the
  distributed operator against the serial one, the check ladder through it,
  the time of each kind of application, and part C's Gauss-Newton MAP.

Two scales. `--quick` is the section on a 100 by 50 grid of 20 m cells with
a 2.5 Hz wavelet and 400 steps of 3.6 ms; every part runs in about 40 s in
all, and it is what the test runs. The full scale is 200 by 80 cells of
10 m (the section 0.8 km deep), a 5 Hz wavelet and 600 steps of 1.8 ms; each
part runs in under two minutes. In both, the shortest wavelength that
matters, at two and a half times the peak frequency in the slowest rock,
spans at least fourteen cells. The proof grid is a 0.64 by 0.32 km sliver,
mostly sponge, with 300 steps, two shots and twelve receivers.

## Build and run

```
cd pygeoinf2/miniapps/waveform
python build.py                              # cmake + nanobind, -O3, module copied in place
MPLBACKEND=Agg python run_all.py --quick     # parts A to D, small scale
MPLBACKEND=Agg python proofs.py              # one part at the full scale; likewise
                                             # synthetic.py, newton.py, laplace.py, occam.py
MPICXX=/path/to/mpicxx python build.py --mpi              # also acoustic_mpi
mpiexec -n 3 python mpi_run.py --quick                    # part E, that MPI's mpiexec
```

The build is the C++ backend mini-app's: `find_package(nanobind)`,
`nanobind_add_module(acoustic STABLE_ABI NB_STATIC acoustic.cpp)`, `build.py`
configuring the `CMakeLists.txt` in a scratch directory for the running
interpreter and copying `acoustic.abi3.so` next to the sources, in about
seven seconds. No dependency beyond nanobind and NumPy. `--mpi` turns on the
CMake option `WAVEFORM_MPI`, which builds the same source a second time as
`acoustic_mpi`, linked to the MPI that `FindMPI` finds, or to the one whose
compiler wrapper `MPICXX` names. No `mpi4py` is needed: the module does its
own communication.

Every script takes `--quick`, `--outdir` and `--seed`. A part run alone
rebuilds the problem from the seed. Part C saves its modes to
`modes_<scale>_<seed>.npz` beside the figures, and the two scripts of part D
read the polished mode from there, or run part C's Newton iterations first
when it is missing; at the full scale run the scripts one at a time, C before
D. Figures are written next to the sources (`quick/` at the quick scale). The
test is `pygeoinf2/tests/test_miniapp_waveform.py`, marked `slow`: the proofs
on the proof grid and parts B to D at the quick scale, in about 50 s, and part
E on two ranks in a copy of the app built with MPI, in about 25 s, which
skips when no MPI compiler wrapper builds it (set `MPICXX` to choose one).

| file | what it is |
|---|---|
| `acoustic.cpp` | the solver: `Model` (grid, spacing, step, taper, stencil, receivers) with `forward`, `born` (the linearized run), `adjoint`, `adjoint_born` (the second adjoint), `correlate`, the fused `gradient`, `curvature` and `second_born`, `laplacian` with its transpose, and `snapshots`; fields held in C++ behind integer handles; with `WAVEFORM_MPI`, the split of the columns, the halo exchange and the commands rank 0 sends the others |
| `CMakeLists.txt`, `build.py` | the build, as in the C++ backend mini-app, with the MPI option |
| `survey.py` | the scales, the survey, the true and background models, the boxes, the model space and the prior |
| `wave_operator.py` | `WaveformOperator`: the forward map as an `Operator` with both derivatives, its field cache and its threads; the unfused compositions the fused kernels are checked against |
| `inversion.py` | `GaussNewtonMAP`, `GaussNewtonOccam`, a Newton-CG that logs its gradient norms, the saved mode |
| `common.py` | the command line, the stopwatch, the section plots |
| `proofs.py`, `synthetic.py`, `newton.py`, `laplace.py`, `occam.py` | parts A to D, each a script with a `main()` and a `run()` the others and the test call |
| `run_all.py` | parts A to D in order |
| `mpi_run.py` | part E, run by every rank under `mpiexec` |

## The scheme, the adjoints and what the checks found

From rest, for `n = 0 .. N−1`,

```
u^{n+1} = τ ⊙ (2 u^n − τ ⊙ u^{n−1} + (Δt² / m) ⊙ (L u^n + s^n))
```

with `L` the fourth-order Laplacian, `τ` the Cerjan taper
`exp(−(0.015 (20 − d))²)` at `d` cells from an edge and `s^n` the source.
Before dividing by `m` the step is the equation
`(m / Δt²)(u^{n+1}/τ − 2u^n + τ u^{n−1}) − L u^n = s^n`, so the run is a
system `A(m) U = s`, lower block triangular in time and affine in `m`, and
everything follows from it: the derivative along `δm` is one run with the
source `−δm ⊙ a^n`, `a^n = (u^{n+1}/τ − 2u^n + τ u^{n−1})/Δt²` the
acceleration of the stored field; the adjoint is one run backward,

```
λ^{k−1} = τ ⊙ (2 λ^k − τ ⊙ λ^{k+1} + (Δt² / m) ⊙ (Lᵀ λ^k + Sᵀ r^k)),   k = N .. 1,
```

with the grid gradient `−Σ_n a^n ⊙ λ^n`; and the adjoint of the second
derivative `F''(m)[δm, ·]` applied to `w` is

```
−Σ_n δa^n ⊙ λ_w^n + Σ_n a^n ⊙ μ_w^n,    Aᵀ Λ_w = Sᵀ w,   Aᵀ M_w = ∂A[δm]ᵀ Λ_w,
```

`δa` the acceleration of the linearized field along `δm` and the source of
`M_w` at step `k` being `δm ⊙ (λ^{k−1}/τ − 2λ^k + τ λ^{k+1})/Δt²`, each taper
factor on the other side of the step from where the forward has it.

Three things to get right, and how each was settled:

- **The sign of the second adjoint's last term**, by the dot test. With
  `Aᵀ M_w = ∂A[δm]ᵀ Λ_w` the term is `+(∂A[·] U)ᵀ M_w`, not `−`: on the
  proof grid the pairing `(F''(m)[δm, x], w)` was −0.1128, the adjoint with
  the minus sign gave +0.0632, and with the plus sign −0.1128 to fifteen
  digits.
- **The index alignment of the adjoint**, by the dot test of `F'(m)`. `λ^n`
  is the multiplier of the equation of step `n`, which produces `u^{n+1}`; it
  pairs with `a^n`, which contains `u^{n+1}`, and the residual of the data of
  `u^k` is injected at backward step `k`. The stored field is
  `U[n] = u^{n+1}`.
- **The sponge**, by measuring what it sends back; the checks pass with any
  taper whose transpose is taken consistently. Tapering only the new level,
  `u^{n+1} = τ ⊙ v`, damps the field but not its rate of change: on the
  full grid with a homogeneous 2 km/s model, the waves that come back from
  that sponge carry 40% of the data's rms, against a reference run on a grid
  three times the size. Cerjan's own sponge, which tapers both levels as
  above, sends back 5%, and its rate of 0.015 is the best of those tried
  from 0.008 to 0.04. The two-level taper is what the code does.

The Laplacian is not symmetric: its row 0, the free surface, is zero, while
rows 1 and 2 reach row 0 of the field, and above the surface the stencil
reads the image `u(−z) = −u(z)`. Its transpose is written out twice, as the
gather the steps use and as the literal scatter of the forward's terms, and
the two are compared. `L u` and `Lᵀ u` differ by 6% of `|L u|` on the proof
grid, which is how much an adjoint that assumed symmetry would be wrong by.

### The measured defects

The linearization point is a draw from the prior and so are the directions;
the model space is the Sobolev space `H²` on the box, whose metric runs from
1 to 3.9 × 10⁶. The dot tests are taken against `|A x| |y|`, the scale
rounding in the two sums is proportional to; the library's checks floor
their scale at one.

| identity | proof grid | full grid |
|---|---|---|
| column stencil `L` against its literal sum | 1.3e-16 | 1.4e-16 |
| column stencil `Lᵀ` against the literal scatter | 1.4e-16 | 1.4e-16 |
| fused gradient against `adjoint` + `correlate` | 1.2e-15 | 1.5e-15 |
| fused curvature against `adjoint` + `adjoint_born` + `correlate` | 2.8e-15 | 2.7e-15 |
| fused `second_born` against two `born` runs | 3.5e-15 | 7.4e-15 |
| `(F'(m) x, y) − (x, F'(m)* y)`, five pairs | 1.6e-15 | 6.0e-16 |
| `(F''(m)[d, x], y) − (x, F''(m)[d, ·]* y)`, five pairs | 3.4e-15 | 6.3e-16 |
| `check_operator(F'(m))` | 6.9e-15 | 1.9e-14 |
| `check_operator(F''(m)[d, ·])` | 1.6e-15 | 1.3e-14 |
| `check_derivative(F)`, step 1e-6 | 2.0e-9 | 3.5e-9 |
| `check_second_derivative(F)`, step 1e-5 | 1.5e-9 | 8.1e-10 |
| `check_gradient` of the MAP objective | 9.4e-9 | 3.1e-8 |
| `check_traits` of the exact Hessian (its symmetry) | 1.3e-13 | 1.1e-12 |

Central differences of `F` against `F'(m) d`, and of `F'` against
`F''(m)[d, e]`, relative errors, on the full grid:

| step | `F` against `F'` | `F'` against `F''` |
|---|---|---|
| 1e-1 | 1.7e-3 | 8.4e-3 |
| 1e-2 | 1.7e-5 | 1.3e-4 |
| 1e-3 | 1.7e-7 | 1.3e-6 |
| 1e-4 | 1.8e-9 | 1.3e-8 |
| 1e-5 | 2.3e-9 | 1.1e-8 |

A factor of a hundred per decade of step until rounding takes over: the
signature of a derivative that is exactly the derivative of the computed
map. The ladder takes 1 s on the proof grid and 32 s on the full one.

## The parts at the full scale

Machine: AMD Ryzen AI 7 PRO 350 (16 threads), shared during these runs with
other jobs, so the seconds are indicative. Three shots run on three threads.
One shot on the true model takes 0.05 s forward, 0.06 to 0.08 s linearized,
0.04 s for the gradient's backward pass and 0.07 s for the curvature's.

**B.** 97 200 data; noise standard deviation 3.7 × 10⁻⁴. The truth's
chi-squared is 97 872 (critical at 95%: 97 926) and the background's 128 485:
the anomalies are 1.7% of the data's rms and thirty thousand units of
chi-squared.

**C.** Newton-CG, preconditioned by the prior covariance, inner conjugate
gradients capped at 40 with the default forcing of 0.1, stopped when an
iteration lowers the objective by less than 10⁻⁵ of it (about one unit of
chi-squared):

| run | iterations | objective | gradient norm | chi-squared | relative error | applications `J` / `J*` / curvature | seconds |
|---|---|---|---|---|---|---|---|
| Gauss-Newton from the background | 7 | 97 536.57 | 444 | 97 445 | 0.270 | 212 / 220 / 0 | 37.6 |
| then the exact Hessian, forcing 0.01, inner cap 60 | 2 | 97 536.44 | 16.6 | 97 442 | 0.273 | 120 / 123 / 120 | 37.7 |
| exact Hessian from the background, capped at 3 | 3 | 97 864.60 | 2.2 × 10⁵ | 97 830 | 0.225 | 29 / 33 / 29 | 10.7 |

The relative error is `|m − truth| / |m0 − truth|` over the cells outside the
sponge: one at the background. The gradient norm starts at 2.8 × 10⁶.
Gauss-Newton gets the mode; the exact Hessian then takes its gradient down
27-fold in two steps, held back by the inner cap, and at the quick scale, with
inner solves to 1%, from 51 to 0.004. From the background the exact Hessian is
slower: the residual is large there, its curvature term makes the Hessian
indefinite, and truncated conjugate gradients stop at the first direction of
negative curvature with a step no better than steepest descent's. That is
measured, not inferred: at the background the inner solve stops on negative
curvature at both scales, and at the Gauss-Newton mode it does not. The mode
fits the data better than the truth does, as a mode should, and its relative
error of 0.27 is the noise and the prior's smoothing, not the solver: the
Gauss-Newton and the exact modes differ by 0.13 in an objective of 97 536.

**D.** The Laplace covariance on the four box means. A randomized
eigendecomposition of the normal operator at the mode, rank 100, 440
applications in 31 s; then four conjugate-gradient solves preconditioned by
it, of 37 to 43 iterations, 159 applications in 35 s, residuals reduced to
10⁻³. Preconditioned by the prior alone, the same four solves reach the cap of
120 iterations each with their residuals reduced only to between 0.014 and
0.036, in 93 s.

| box | truth | prior, 2 sd | Laplace at the mode, 2 sd | sd over prior sd |
|---|---|---|---|---|
| fast, shallow | 0.2388 | 0.2640 ± 0.0309 | 0.2389 ± 0.0016 | 0.052 |
| slow | 0.2761 | 0.2524 ± 0.0288 | 0.2760 ± 0.0017 | 0.059 |
| fast, deep | 0.2138 | 0.2322 ± 0.0326 | 0.2132 ± 0.0027 | 0.082 |
| background | 0.2301 | 0.2302 ± 0.0318 | 0.2294 ± 0.0033 | 0.103 |

Every truth lies within two posterior standard deviations; the deeper box
and the one beside the shots are the less well resolved. Each standard
deviation had settled to three digits by three quarters of its solve, which
matters because a conjugate-gradient solve stopped early underestimates a
variance. The posterior correlations between the boxes run from −0.19 to
+0.39.

**D, continued.** Occam at 95%: five outer iterations of the augmented
Lagrangian, fifteen Newton iterations on the Gauss-Newton Hessian, 43 s; the
constraint met to 7 × 10⁻⁶ of the critical value, its multiplier 0.090.

| model | chi-squared | distance from the background (`H²`) | relative error |
|---|---|---|---|
| MAP mode | 97 442 | 0.0687 | 0.273 |
| Occam | 97 927 | 0.0344 | 0.289 |
| truth | 97 872 | 0.0453 | 0 |

Occam's model is half the MAP's size and smaller than the truth, as the
least structure the data allow should be; the MAP mode, fitting to the
noise's mean rather than its 95% level, is larger than the truth.

At the quick scale (2.5 Hz, half the resolution) the same parts give a MAP
relative error of 0.26 and Occam's 0.41, all four truths within two
posterior standard deviations, and standard deviations from 9% to 22% of the
prior's.

## E. The grid split across ranks

Built with `WAVEFORM_MPI`, each rank owns a contiguous block of the grid's
columns and its part of every field, and every step of every kernel
exchanges two columns of halo with its neighbours, which is how far the
stencil reaches. Rank 0 runs the library: each call it makes sends a command
to the other ranks, waiting in `serve()`, scatters the model by columns,
broadcasts the traces, and gathers the receiver data (each receiver belongs
to one rank, so a sum gathers them) and the gradients back. Every kernel is
written once, for the block a rank owns; the serial build is the same code
with one block. The Python above it, the operator, the estimators and the
parts, is the serial code: `synthetic.setup(..., backend=acoustic_mpi)` is the
only change, and the shots then run one after another, each on every rank.

Measured on the same machine with MPICH 5.0.0 (the one PETSc built; the
system's Open MPI has no development headers installed):

- **The answers are the serial ones, bit for bit.** `F(m)`, `F'(m) d`,
  `F'(m)* r`, `F''(m)[d, e]` and `F''(m)[d, ·]* r` at a prior draw agree
  with the serial operator's to 0.0, on the proof grid, the quick grid and
  the full grid on three ranks and on the full grid on six: each cell sees
  the same arithmetic in the same order, and the sum that gathers the data
  adds zeros. The library's check ladder through the distributed operator
  passes with defects of the size part A measures. Part C's Gauss-Newton
  MAP, run across three or six ranks, is the mode part C saved serially to
  the last bit: 7 iterations, objective 97 536.57.
- **The memory divides, the time does not.** A field is 26 MB on each of
  three ranks and 13 MB on each of six, against 77 MB whole. One application,
  in seconds, three shots:

  | | forward | `F'` | `F'*` | `F''*` | Gauss-Newton MAP |
  |---|---|---|---|---|---|
  | serial, the shots on three threads | 0.06 | 0.05 | 0.05 | 0.18 | 37.6 |
  | three ranks | 0.10 | 0.08 | 0.07 | 0.24 | 58.0 |
  | six ranks | 0.08 | 0.09 | 0.07 | 0.26 | 56.5 |

  At 200 by 80 cells a rank's block takes microseconds a step, and with two
  halo exchanges a step and the shots run one after another, splitting the
  grid costs more than it saves; three threads over the three shots, which
  share nothing, are the faster way to use three cores. The decomposition is
  for a grid that one process cannot hold.

The `--quick` run of part E on three ranks takes 9 s; the full one 62 s.

## Choices

- **The model space and the prior.** The library's `symmetric_space.fourier`
  box, `Sobolev((nx, nz), 2, 0.1)` with the section's lengths, whose
  vectors are the solver's grid arrays. The prior on `m − m0` has the
  identity covariance on that space, scaled to a pointwise standard deviation
  of 0.02 s²/km² (about 8% of `m`), with the background as its expectation:
  its norm is the model space's, so Occam's distance is the prior's and the
  prior covariance is a preconditioner that costs nothing. The box is
  periodic, so the prior correlates opposite edges a correlation length
  apart; the edges are sponge. A wider prior draws models fast enough at the
  bottom of the section to break the scheme's stability limit.
- **The adjoints are derivative components.** The kernels return the grid
  gradient; `LinearOperator.from_derivative_callables` applies the inverse
  metric. On the box, components are orthonormal in `L²` and a grid sum is
  `cells / area` times an integral, so the derivative components of
  `x ↦ g · x` are `(cells / area)` times the components of `g`.
- **Fused kernels.** The operator uses `gradient`, `curvature` and
  `second_born`, which correlate on the fly and store three steps of the
  adjoint fields instead of all of them; `adjoint`, `adjoint_born`,
  `correlate` and `born` are the unfused pieces, and the two are checked to
  agree to rounding.
- **Storage and memory.** Each shot's forward field is stored whole as
  `float64`, in C++, behind a handle the operator caches and releases: 77 MB
  at the full scale, three of them cached for the last model asked about.
  The curvature also needs the linearized field of its shot, stored for its
  backward pass; it runs three shots at a time at the full scale, so the
  stored wavefields peak at about 460 MB. Checkpointing is not needed at
  these sizes; `snapshots(..., stride)` keeps every `stride`-th step of a
  run, which is what the snapshot figure uses and where checkpointing would
  start. The full grid is 80 cells deep rather than 100 so that this fits.
  Split across ranks, each rank holds its block of every field.
- **Threads.** The kernels release the interpreter lock; the operator runs
  its shots on threads (`n_jobs`, `curvature_jobs`), and the randomized
  eigendecomposition in part D runs two probes at a time on threads through
  a `joblib.parallel_config(backend="threading")` context. The operator holds
  the C++ solver and cannot be sent to a process.
- **Stability.** A model with a non-positive value, or fast enough anywhere
  to come within 2% of the stability limit `c Δt / h = √(3/8)`, raises
  `EvaluationError`, from which the library's line searches retreat.
- **Gauss-Newton.** `MaximumAPosteriori` and `Occam` carry a Hessian only when
  the forward operator has a second derivative, and then the exact one. The
  Gauss-Newton runs use the same operator without its second derivative,
  through `GaussNewtonMAP` and `GaussNewtonOccam`, which attach
  `2 (P + J* R⁻¹ J)` and `2 J* R⁻¹ J / χ²_crit` to the objectives; the
  symmetry of both is in the test.
- **Ranks without `mpi4py`.** The MPI module initializes MPI itself and does
  its own communication, so a run needs nothing beyond the MPI the module was
  built against. Rank 0 checks every argument before it sends a command, and
  a failure in the driver aborts the run rather than leave the other ranks
  waiting. Both modules can be imported into one process, as part E does to
  compare them: their C++ classes live in different namespaces.
- **Sources and receivers** sit two cells below the free surface, where the
  pressure is not held at zero.
- **The boxes** average `m`, the slowness squared, which is linear in the
  model.
- **Inverse crime.** The data are made by the operator that inverts them, on
  the same grid. Part C's relative errors are therefore a statement about the
  noise and the prior, not about modeling error.

## What the app shows

- An adjoint method written as the transpose of the discrete scheme is
  exact: the dot tests hold to 10⁻¹⁵ and the differences fall as the square
  of the step, for the first derivative and the second. A sign in the
  second-order adjoint is caught by its dot test or by the symmetry of the
  exact Hessian built on it; the differences test only its forward action.
- One operator, every nonlinear estimator: the MAP, the Laplace Gaussian and
  Occam's inversion are the library's classes, unchanged, on a forward map
  that is a C++ time-stepper. Each linearization costs one forward run per
  shot and is cached; a Gauss-Newton Hessian application costs one
  linearized and one adjoint run per shot, an exact one a further linearized
  run and a backward pass carrying two adjoint fields.
- The kernels written once for a block of columns serve one process and
  many: the distributed operator is the serial one to the last bit, and the
  library cannot tell them apart.
- The exact Hessian is worth having at the mode and not before it: from the
  Gauss-Newton mode it converges quadratically when the inner solves are
  allowed to, and from the background it is indefinite and slower than
  Gauss-Newton.
- At this signal-to-noise the data inform hundreds of directions, and every
  solve with the normal operator feels it. A randomized eigendecomposition of
  that operator, once, turns four solves of hundreds of iterations into four
  of forty.
