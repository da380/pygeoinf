# Acoustic waveform inversion on a C++ solver

How the app is built on the library is walked through in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## The problem

A vertical section of rock 2 km wide and 0.8 km deep, its top a free
surface, its velocity increasing linearly with depth from 1.8 km/s at a rate
of 0.6 per second, with three Gaussian anomalies in it: a fast one of
+0.25 km/s at 250 m depth, a slow one of -0.20 km/s at 320 m and a smaller
fast one of +0.20 km/s at 460 m. Three shots are fired 20 m below the surface
and recorded by a line of receivers across the section at the same depth for
about a second. The unknown is
`m = 1 / c²`, the slowness squared, in s²/km², and the data are the pressure
at every receiver at every time step, with Gaussian noise of 3% of the data's
root-mean-square added sample by sample. The background is known; the
anomalies are what the data are asked for.

The wave equation `m ∂²u/∂t² = ∇²u + f(t) δ(x − x_s)` is solved by a
finite-difference scheme written here in C++: second order in time, fourth
order in space, a Ricker wavelet, on a grid bigger than the section with a
Cerjan sponge 24 cells wide beyond it on the sides and the bottom. The C++
also provides the scheme's linearization, its exact
discrete adjoint and the second-order adjoint, and they are wrapped as one
library `Operator` with a first and a second derivative. On it the library's
nonlinear estimators run unchanged: the MAP model by Newton-CG on the
Gauss-Newton and on the exact Hessian, Occam's inversion, and the Laplace
Gaussian at the mode. Built with MPI, the same solver splits its grid across
ranks, and the operator and the estimators above it do not change.

The parts:

- **A, the solver and its proofs** (`proofs.py`): the stencil's explicit
  transpose, the fused kernels against the unfused ones, dot tests, central
  differences at a ladder of steps, and the library's check ladder, on the
  64 by 32 proof grid and again on the grid of the scale asked for; what the
  sponge sends back, against a grid too big to reflect; a figure of
  wavefield snapshots on the whole grid, the sponge marked, and a shot
  gather.
- **B, the data** (`synthetic.py`): the truth's data with noise, the misfit
  of the truth and of the background, and the misfit's gradient at the
  background.
- **C, the MAP model** (`newton.py`): Newton-CG from the background with the
  Gauss-Newton Hessian; then, from where that stopped, with the exact
  Hessian; and a short run with the exact Hessian from the background. Its
  figure: the three models against the truth, and how the runs converged.
- **D, Occam's inversion** (`occam.py`): Occam's model at the 95% level,
  against the MAP mode and the truth, as its figure shows them.
- **E, the uncertainty** (`laplace.py`): the Laplace Gaussian at the mode,
  pushed onto the mean of `m` over four boxes, one over each anomaly and one
  over plain background, against the prior's; its figure sets each box
  mean's two standard deviations after against before, with the truth.
- **F, the grid split across MPI ranks** (`mpi_run.py`, under `mpiexec`): the
  distributed operator against the serial one, the check ladder through it,
  the time of each kind of application, and part C's Gauss-Newton MAP.

Two scales of the same section. `--quick` puts it on 100 by 40 cells of
20 m, in a grid of 148 by 64 with the sponge, with 50 receivers, a 2.5 Hz
wavelet and 400 steps of 3.6 ms; parts A to E run in about 70 s in all, and
it is what the test runs. The full scale is 200 by 80 cells of 10 m, in a
grid of 248 by 104, with 67 receivers, a 5 Hz wavelet and 600 steps of
1.8 ms; each part runs in under two minutes. In both, the shortest wavelength
that matters, at two and a half times the peak frequency in the slowest
rock, spans at least fourteen cells. Below the section the velocity stops
increasing: it levels off within about 0.2 km, below 2.4 km/s, so that the
rock in the sponge stays slow enough for the time step under the prior's
draws. The
proof grid is 64 by 32 cells of 10 m, a section of 24 by 12 inside a sponge of
its own, 20 cells at a rate of 0.015, with 300 steps, two shots and twelve
receivers.

## Build and run

```
cd miniapps/waveform
python cpp/build.py                          # cmake + nanobind, -O3, module copied into src/
MPLBACKEND=Agg python run_all.py --quick     # parts A to E, small scale
MPLBACKEND=Agg python proofs.py              # one part at the full scale; likewise
                                             # synthetic.py, newton.py, occam.py, laplace.py
MPICXX=/path/to/mpicxx python cpp/build.py --mpi          # also acoustic_mpi
mpiexec -n 3 python mpi_run.py --quick                    # part F, that MPI's mpiexec
```

The build is the C++ backend mini-app's: `find_package(nanobind)`,
`nanobind_add_module(acoustic STABLE_ABI NB_STATIC acoustic.cpp)`,
`cpp/build.py` configuring the `CMakeLists.txt` in a scratch directory for
the running interpreter and copying `acoustic.abi3.so` into `src/`, where
`src/survey.py` imports it, in about seven seconds. No dependency beyond
nanobind and NumPy. `--mpi` turns on the CMake option `WAVEFORM_MPI`, which
builds the same source a second time as `acoustic_mpi`, linked to the MPI
that `FindMPI` finds, or to the one whose compiler wrapper `MPICXX` names,
and puts it in `src/` too, where `mpi_run.py` imports it. No `mpi4py` is
needed: the module does its own communication.

A script runs from any directory, by its path, and so does the build.
Every script takes the same options: `--quick` or `--full` (the default)
picks a preset scale, every field of the scale is then an option that
overrides it, and `--outdir` says where the figures go. The proof grid of
part A keeps its own preset. `python run_all.py --help`, in short:

```
  --quick | --full      the quick scale, or the full one (the default)
  --outdir OUTDIR       where the figures go (default: results/,
                        results/quick/ at the quick scale)
the scale, each default the full preset's:
  --nx, --nz, --h, --dt, --steps
                        the section's cells, the cell size and the time steps
                        (200, 80, 0.01 km, 0.0018 s, 600)
  --peak-frequency float
                        the Ricker wavelet's peak frequency, Hz (5.0)
  --shots float [float ...]
                        the shots' positions across the section, km (0.5 1.0 1.5)
  --receiver-spacing int
                        the cells between receivers (3)
  --seed int            the seed of the noise and every part's draws (2026)
  --order, --length-scale, --measure-order, --prior-std
                        the model space and the prior (2.0, 0.1 km, 0.0, 0.02)
  --noise-fraction float
                        the noise, as a fraction of the data's rms (0.03)
  --level float         the discrepancy level's credibility (0.95)
  --sponge, --sponge-rate
                        the sponge's width and taper (24, 0.008)
  --newton-iterations, --cg-iterations, --occam-iterations, --laplace-rank, ...
                        how hard each solver may work
  --probe-jobs, --n-jobs, --curvature-jobs
                        the threads
```

Two overrides at the quick scale, against its own numbers:

- **Noisier data**, `run_all.py --quick --noise-fraction 0.1`: the noise at
  a tenth of the data's rms rather than three hundredths, on the same
  truth. The background's chi-squared falls from 81 916 to 62 577 on 60 000
  data, so the anomalies barely stand above the noise; the MAP model's
  relative error rises from 0.60 to 1.05, further from the truth than the
  background is, Occam's from 0.37 to 0.57, and the Laplace standard
  deviation of the slow box's mean more than doubles, from 0.08 to 0.18 of
  the prior's, the truth lying more than two of them from the Laplace
  mean.
- **A single shot**, `run_all.py --quick --shots 1.0`: one shot in the
  middle of the section rather than three, 20 000 data. The MAP model's
  relative error rises from 0.60 to 0.90, and the Laplace standard
  deviations rise from 0.06 to 0.12 of the prior's on the shallow fast
  box and from 0.13 to 0.23 on the deep one.

A part run alone rebuilds the problem from the seed. Part C saves its modes
to `modes_<scale>_<seed>.npz` beside the figures, with a digest of the scale
added when an option departs from the preset, and parts D and E read the
polished mode from there, or run part C's Newton iterations first when it is
missing; at the full scale run the scripts one at a time, C before D and E.
Each part's figure is titled with the app and the part, and written into
`results/` (`results/quick/` at the quick scale), which git ignores, with
the saved modes and part F's `mpi_<scale>.json`. The test is
`tests/test_miniapp_waveform.py`, marked `slow`: the command line
built from the scale, the proofs on the proof grid, what the sponge sends
back at the quick scale, and parts B to E at the quick scale built through
the same command line, in about a minute, and part F on two ranks in a copy
of the app built with MPI, in about 20 s, which skips when no MPI compiler
wrapper builds it (set `MPICXX` to choose one).

| file | what it is |
|---|---|
| `cpp/acoustic.cpp` | the solver: `Model` (grid, spacing, step, taper, stencil, receivers) with `forward`, `born` (the linearized run), `adjoint`, `adjoint_born` (the second adjoint), `correlate`, the fused `gradient`, `curvature` and `second_born`, `laplacian` with its transpose, and `snapshots`; fields held in C++ behind integer handles; with `WAVEFORM_MPI`, the split of the columns, the halo exchange and the commands rank 0 sends the others |
| `cpp/CMakeLists.txt`, `cpp/build.py` | the build, as in the C++ backend mini-app, with the MPI option; the modules land in `src/` |
| `src/survey.py` | the scale with its three presets, the grid around the section, the survey, the true and background models, the boxes, the model space and the prior |
| `src/wave_operator.py` | `WaveformOperator`: the forward map as an `Operator` with both derivatives, its field cache and its threads; the unfused compositions the fused kernels are checked against |
| `src/inversion.py` | a Newton-CG that logs its gradient norms, the saved mode |
| `src/common.py` | the command line built from the scale, the stopwatch, the section plots |
| `proofs.py`, `synthetic.py`, `newton.py`, `occam.py`, `laplace.py` | parts A to E, each a script with a `main()` and a `run()` the others and the test call; `proofs.reflection`, what the sponge sends back |
| `run_all.py` | parts A to E in order |
| `mpi_run.py` | part F, run by every rank under `mpiexec` |
| `results/` | what a run writes: the figures, the saved modes and part F's numbers, `results/quick/` at the quick scale |

## The scheme, the adjoints and what the checks found

From rest, for `n = 0 .. N−1`,

```
u^{n+1} = τ ⊙ (2 u^n − τ ⊙ u^{n−1} + (Δt² / m) ⊙ (L u^n + s^n))
```

with `L` the fourth-order Laplacian, `τ` the Cerjan taper, one inside and
`exp(−(0.008 (24 − d))²)` at `d < 24` cells from the left, right or bottom
edge of the grid, and `s^n` the source.
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

Two conventions the expressions above depend on. `λ^n` is the multiplier of
the equation of step `n`, which produces `u^{n+1}`; it pairs with `a^n`,
which contains `u^{n+1}`, and the residual of the data of `u^k` is injected
at backward step `k`; the stored field is `U[n] = u^{n+1}`. The taper is
Cerjan's, applied to both time levels as above, so that it damps the field
and its rate of change together; its width and rate, and what it sends
back, are below. The dot tests of `F'(m)` and `F''(m)[δm, ·]` and the
differences test cover the first two; the sponge is judged by what it
sends back, since the checks pass with any taper whose transpose is taken
consistently.

The Laplacian is not symmetric: its row 0, the free surface, is zero, while
rows 1 and 2 reach row 0 of the field, and above the surface the stencil
reads the image `u(−z) = −u(z)`. Its transpose is written out twice, as the
gather the steps use and as the literal scatter of the forward's terms, and
the two are compared. `L u` and `Lᵀ u` differ by 6% of `|L u|` on the proof
grid, which is how much an adjoint that assumed symmetry would be wrong by.

### The box and the sponge

The grid is bigger than the section: the sponge lies beyond it, 24 cells on
the left, the right and below, so that no cell of the section is damped.
The model space and the prior are over the whole grid; the shots, the
receivers, the boxes of part E and the relative errors are within the
section. Every figure shows the section alone but the snapshots, which show
the whole grid with the section outlined and the sponge hatched.

What the sponge sends back is measured by `proofs.reflection`: every shot
over the true model, on the scale's grid and on one padded by 200 cells of
the same medium on each absorbing side, with the same sources and receivers.
Nothing comes back from the padded grid's edges within the record (300 cells
of padding give the same reference to the last bit), so the difference of
the two runs is the reflected field. It is measured twice: its rms over the
data of every shot against the data's rms, and its peak over the section at
the four snapshot times against the peak of the reference field there, for
the worst shot. The test asks that at the quick scale they be below 1% and
3%.

A sweep at the quick scale, data and section in percent:

| width (cells) | rate: data / section |
|---|---|
| 20 | 0.0080: 1.07 / 9.6; 0.0100: 1.18 / 7.2; 0.0162: 3.6 / 8.5 |
| 24 | 0.0065: 0.57 / 1.9; 0.0070: 0.52 / 1.5; 0.0080: 0.70 / 2.0; 0.0090: 1.08 / 3.3; 0.0135: 2.8 / 7.0 |
| 30 | 0.0050: 0.31 / 1.3; 0.0065: 0.60 / 1.9; 0.0108: 1.9 / 5.5 |
| 40 | 0.0030: 0.10 / 0.35; 0.0081: 1.0 / 3.7 |
| 60 | 0.0020: 0.04 / 0.16; 0.0054: 0.40 / 1.5 |

The last rate of each width is the one that tapers to 0.9 at the grid's
edge; the best of those tried are well below it, tapering to 0.97 to 0.99
there, a gentler taper reflecting less from its own gradient. Wider is
better, but at the
quick scale a sponge of 40 cells or more is 0.8 km deep, and its edge's echo
hardly returns within the record, so the sweep understates what the wide
ones send back. At the full
scale 30 cells at 0.0065 send back about 0.45% / 2% and 40 cells at 0.005
0.22% / 0.64%, but a field of 600 steps is then 137 or 161 MB against 124 MB
with 24 cells, and part C holds four fields at once (see "Storage and
memory"): 24 cells is the widest sponge whose fields fit in 500 MB.

The sponge is 24 cells at a rate of 0.008, a taper of 0.964 at the edge.
What it sends back:

| | data | section |
|---|---|---|
| quick scale | 0.70% | 1.96% |
| full scale | 0.57% | 3.82% |

In the shot gather the reflected arrivals peak at 1.9% of the gather's clip
and do not show; in the snapshot at 0.87 s at the full scale a faint upgoing
arc from the bottom does.

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
| fused gradient against `adjoint` + `correlate` | 1.2e-15 | 1.7e-15 |
| fused curvature against `adjoint` + `adjoint_born` + `correlate` | 1.9e-15 | 1.9e-15 |
| fused `second_born` against two `born` runs | 7.4e-15 | 3.6e-14 |
| `(F'(m) x, y) − (x, F'(m)* y)`, five pairs | 6.9e-17 | 1.1e-16 |
| `(F''(m)[d, x], y) − (x, F''(m)[d, ·]* y)`, five pairs | 7.7e-16 | 1.2e-15 |
| `check_operator(F'(m))` | 3.3e-14 | 1.1e-12 |
| `check_operator(F''(m)[d, ·])` | 7.1e-13 | 2.6e-13 |
| `check_operator` of the box means | | 3.8e-14 |
| `check_derivative(F)`, step 1e-6 | 9.7e-10 | 2.6e-9 |
| `check_second_derivative(F)`, step 1e-5 | 6.9e-8 | 6.4e-8 |
| `check_gradient` of the MAP objective | 1.3e-7 | 7.1e-6 |
| `check_traits` of the exact Hessian (its symmetry) | 5.8e-15 | 6.4e-15 |

The boxes do not fit on the proof grid.

Central differences of `F` against `F'(m) d`, and of `F'` against
`F''(m)[d, e]`, relative errors, on the full grid:

| step | `F` against `F'` | `F'` against `F''` |
|---|---|---|
| 1e-1 | 6.8e-4 | 4.3e-3 |
| 1e-2 | 6.9e-6 | 4.8e-5 |
| 1e-3 | 6.9e-8 | 4.8e-7 |
| 1e-4 | 7.9e-10 | 5.7e-9 |
| 1e-5 | 3.8e-9 | 2.7e-8 |

A factor of a hundred per decade of step until rounding takes over: the
signature of a derivative that is exactly the derivative of the computed
map. The ladder takes 1 s on the proof grid and 50 s on the full one; part A
at the full scale, the reflection and the figure with it, 53 s.

## The parts at the full scale

Machine: AMD Ryzen AI 7 PRO 350 (16 threads), shared during these runs with
other jobs, so the seconds are indicative. Three shots run on three threads,
but the curvature's one at a time (see "Storage and memory"). One shot on
the true model takes 0.06 s forward, 0.07 s linearized, 0.05 s for the
gradient's backward pass and 0.11 s for the curvature's.

**B.** 120 600 data; noise standard deviation 3.3 × 10⁻⁴. The truth's
chi-squared is 121 297 (critical at 95%: 121 409) and the background's
155 065: the anomalies are 1.6% of the data's rms and thirty-four thousand
units of chi-squared.

**C.** Newton-CG, preconditioned by the prior covariance, inner conjugate
gradients capped at 40 with the default forcing of 0.1, stopped when an
iteration lowers the objective by less than 10⁻⁵ of it (about one unit of
chi-squared):

| run | iterations | objective | gradient norm | chi-squared | relative error | kernel runs per shot: linearized / adjoint / curvature | seconds |
|---|---|---|---|---|---|---|---|
| Gauss-Newton from the background | 5 | 121 850.34 | 1 220 | 121 193 | 0.643 | 159 / 165 / 0 | 31.7 |
| then the exact Hessian, forcing 0.01, inner cap 60, one step | 1 | 121 850.30 | 50.6 | 121 194 | 0.646 | 120 / 62 / 60 | 43.8 |
| exact Hessian from the background, capped at 2 | 2 | 142 638.23 | 2.8 × 10⁶ | 142 293 | 3.30 | 54 / 30 / 27 | 20.5 |

Part C took 97 s. The relative error is
`|m − truth| / |m0 − truth|` over the section: one at the background. The
gradient norm starts at 3.0 × 10⁶. Gauss-Newton gets the mode; one exact
Newton step from there, its inner solve stopped by the cap of 60, takes the
gradient down by a factor of 24, and at the quick scale, where the inner
solves run to 1%, two steps take it from 185 to 0.017. At the full scale an
exact Hessian application costs about 0.8 s, two linearized runs, an adjoint
run and a curvature pass for each shot with the curvature's shots one at a
time, which is why part C stops at one exact step from the mode and two
from the background there: to stay under two minutes. From the background
the exact Hessian is slower: the residual is large there, its curvature term
makes the Hessian indefinite, and truncated conjugate gradients stop at the
first direction of negative curvature with a step no better than steepest
descent's, two of which leave a model more than three times further from
the truth than the background, a fast smear under the anomalies. That is
measured, not inferred: part C prints why each inner solve stopped, and
from the background the first ones stop on negative curvature at both
scales, the first five at the quick one, while at the Gauss-Newton mode
none does. At the quick scale the exact Hessian from the background comes,
in its ten iterations, to within 2.7 of the mode's objective of 60 849.

The mode fits the data better than the truth does, as a mode should, and the
Gauss-Newton and the exact modes differ by 0.04 in an objective of 121 850.
Their relative errors of 0.64 and 0.65 are the noise's and the prior's, not
the solver's, and much of them lies where the survey sees little. At the
quick scale over half the mode's squared miss lies within 0.4 km of the
section's sides and two fifths in its bottom 0.2 km (the two overlap at the
corners), and over the middle 1.2 km of the top 0.6 km its relative error is
0.32. On data without noise the quick Gauss-Newton mode's relative error is
0.36, the prior's pull alone, and the structure outside that middle falls
from an rms of 0.0054 s²/km², a quarter of the prior's standard deviation,
to 0.0022. At the full scale 37% of the mode's squared miss lies outside the
middle 1.6 km of the top 0.6 km, and its relative error over that middle is
0.51.

**D.** Occam at 95%: five outer iterations of the augmented
Lagrangian, twenty-one Newton iterations on the Gauss-Newton Hessian, 44 s; the
constraint met to 3 × 10⁻⁶ of the critical value, its multiplier 0.072.
Occam's norm is the space's `H²`, the MAP's the prior's.

| model | chi-squared | distance from the background (`H²`) | relative error |
|---|---|---|---|
| MAP mode | 121 194 | 0.0406 | 0.646 |
| Occam | 121 409 | 0.0314 | 0.566 |
| truth | 121 297 | 0.0453 | 0 |

Occam's model is the smallest of the three in `H²`, as the least structure
the data allow in that norm should be; the MAP mode, fitting to the
noise's mean rather than its 95% level but measured by a prior that
penalizes short scales harder than `H²` does, is smaller than the truth
too. Over the section Occam's relative error is the smaller: it carries
less of the noise's structure along the sides and the bottom.

**E.** The Laplace covariance on the four box means. A randomized
eigendecomposition of the normal operator at the mode seen through the
prior's covariance factor, rank 100, 440 applications in 50 s; then four
conjugate-gradient solves preconditioned by it, of 16 or 17 iterations, 67
applications in 13 s, residuals reduced to 10⁻³; part E took 64 s.
Preconditioned by the prior covariance alone, the same four solves reach
the cap of 120 iterations each with their residuals reduced only to between
10⁻³ and 9 × 10⁻³, in 92 s.

| box | truth | prior, 2 sd | Laplace at the mode, 2 sd | sd over prior sd |
|---|---|---|---|---|
| fast, shallow | 0.2388 | 0.2640 ± 0.0371 | 0.2356 ± 0.0023 | 0.062 |
| slow | 0.2761 | 0.2524 ± 0.0360 | 0.2689 ± 0.0033 | 0.093 |
| fast, deep | 0.2138 | 0.2322 ± 0.0379 | 0.2095 ± 0.0047 | 0.125 |
| background | 0.2301 | 0.2302 ± 0.0375 | 0.2253 ± 0.0065 | 0.173 |

The deep and the background boxes' truths lie 1.8 and 1.5 posterior
standard deviations from the Laplace mean, the two shallow ones' 2.8 and
4.4: the mode underestimates both shallow anomalies, and the Laplace
Gaussian, centered on the mode, does not move it. The Gauss-Newton and the
exact modes give the same box means to a twentieth of a standard deviation,
so it is the noise and the prior's pull, not the mode's convergence; the
quick scale's noise-free mode, in error by 0.36, is that pull alone. The
two deeper boxes are the less well resolved, the background box the least.
Each standard deviation had settled to three digits by half of its solve,
which matters because a conjugate-gradient solve stopped early
underestimates a variance. The posterior correlations between the boxes run
from −0.25 to +0.35.

At the quick scale (2.5 Hz, half the resolution) the same parts give a MAP
relative error of 0.60 over the section (0.32 over its middle) and Occam's
0.37, all four truths within two posterior standard deviations, and
standard deviations from 6% to 19% of the prior's.

## F. The grid split across ranks

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
  the last bit: 5 iterations, objective 121 850.34.
- **The memory divides, the time does not, but for the curvature.** A field
  is 41 MB on each of three ranks and 21 MB on each of six, against 124 MB
  whole. One application, in seconds, three shots:

  | | forward | `F'` | `F'*` | `F''*` | Gauss-Newton MAP |
  |---|---|---|---|---|---|
  | serial, the shots on three threads, the curvature's one at a time | 0.07 | 0.06 | 0.06 | 0.52 | 31.7 |
  | three ranks | 0.08 | 0.07 | 0.07 | 0.25 | 42.8 |
  | six ranks | 0.06 | 0.07 | 0.07 | 0.23 | 40.8 |

  At 248 by 104 cells a rank's block takes microseconds a step, and with two
  halo exchanges a step and the shots run one after another, splitting the
  grid costs more than it saves; three threads over the three shots, which
  share nothing, are the faster way to use three cores. The second adjoint
  is the exception: to hold the memory to four fields the serial operator
  runs its shots one at a time, where the ranks share each shot's work. The
  decomposition is for a grid that one process cannot hold.

The `--quick` run of part F on three ranks takes 10 s; the full one 47 s on
three ranks and 46 s on six.

## Choices

- **The model space and the prior.** The library's `symmetric_space.fourier`
  box, `Sobolev((nx, nz), 2, 0.1)` over the whole grid, sponge and all,
  whose vectors are the solver's grid arrays. The prior on `m − m0` is a
  Sobolev measure of order 2 on that space, a measure of extra regularity
  whose covariance is trace class on `H²`, calibrated to a pointwise
  standard deviation of 0.02 s²/km² (about 8% of `m`), with the background
  as its expectation. The MAP model is penalized in the prior's norm and
  its Newton steps preconditioned by the prior covariance; Occam's distance
  is the space's own `H²` norm, whose Hessian is the identity and needs no
  preconditioner. The box is periodic, so the prior correlates opposite
  edges a correlation length apart; the edges are sponge. A wider prior
  draws models fast enough at the bottom of the grid to break the scheme's
  stability limit.
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
  `float64`, in C++, behind a handle the operator caches and releases:
  124 MB at the full scale, three of them cached for the last model asked
  about. The curvature also needs the linearized field of its shot, stored
  for its backward pass; at the full scale it runs one shot at a time
  (`curvature_jobs=1`), so the stored wavefields peak at four, 495 MB, and
  the unfused compositions part A checks against hold no more. Checkpointing
  is not needed at these sizes; `snapshots(..., stride)` keeps every
  `stride`-th step of a run, which is what the snapshot figure uses and where
  checkpointing would start. The sponge is as wide as this allows. Split
  across ranks, each rank holds its block of every field.
- **Threads.** The kernels release the interpreter lock; the operator runs
  its shots on threads (`n_jobs`, `curvature_jobs`), and the randomized
  eigendecomposition in part E runs two probes at a time on threads through
  a `joblib.parallel_config(backend="threading")` context. The operator holds
  the C++ solver and cannot be sent to a process.
- **Stability.** A model with a non-positive value, or fast enough anywhere
  to come within 2% of the stability limit `c Δt / h = √(3/8)`, raises
  `EvaluationError`, from which the library's line searches retreat.
- **Gauss-Newton.** `MaximumAPosteriori` and `Occam` carry the exact Hessian
  when the forward operator has a second derivative; asked for it
  (`gauss_newton=True`) they carry the Gauss-Newton Hessian instead,
  `2 (P + J* R⁻¹ J)` and `2 J* R⁻¹ J / χ²_crit`, which is what the
  Gauss-Newton runs use on the operator without its second derivative; the
  symmetry of both is in the test.
- **Ranks without `mpi4py`.** The MPI module initializes MPI itself and does
  its own communication, so a run needs nothing beyond the MPI the module was
  built against. Rank 0 checks every argument before it sends a command, and
  a failure in the driver aborts the run rather than leave the other ranks
  waiting. Both modules can be imported into one process, as part F does to
  compare them: their C++ classes live in different namespaces.
- **Sources and receivers** sit two cells below the free surface, where the
  pressure is not held at zero, the receivers across the whole section.
- **The boxes** average `m`, the slowness squared, which is linear in the
  model.
- **Inverse crime.** The data are made by the operator that inverts them, on
  the same grid. Part C's relative errors are therefore a statement about the
  noise and the prior, not about modeling error; and an echo from the grid's
  edges, modeled too, would inform the inversion, which is why the sponge
  keeps it well below the noise.

## What the app shows

- An adjoint method written as the transpose of the discrete scheme is
  exact: the dot tests hold to 10⁻¹⁵ and the differences fall as the square
  of the step, for the first derivative and the second; the differences
  test only the forward action, the dot tests the adjoints.
- One operator, every nonlinear estimator: the MAP, Occam's inversion and
  the Laplace Gaussian are the library's classes, unchanged, on a forward map
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
  solve with the normal operator feels it, beside the prior precision's
  spectrum of many decades. A randomized eigendecomposition of that operator
  seen through the prior's covariance factor, once, turns four solves that
  reach a cap of 120 iterations into four of 17.
- An absorbing boundary is measured against a grid too big to reflect, not
  judged by eye: the sponge here sends back under 1% of the data's rms, well
  below the noise, so that the answers are the survey's own and not the
  grid's edges'.
