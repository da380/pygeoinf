# Code notes

A working list, to be turned into issues or acted on and then deleted.
Items are removed as they are fixed. Line numbers are approximate after
later edits.

## 1. ducc0 0.42 and planetmodel: resolved, release pending

ducc0 0.42 broke planetmodel 1.4.0's `real_harmonics`, and with it every
spectral-element space. planetmodel 1.4.1 fixes it. The `planetmodel` extra
now requires `planetmodel>=1.4.1`, and the lock holds planetmodel 1.4.1 and
ducc0 0.42.0. The fast suite passes on the lock and under the newest versions
the pins admit. What is left is the next release (2.1), so that
`pip install pygeoinf[planetmodel]` no longer admits the broken pair.
`.github/workflows/newest.yml` now catches this kind of break each week.

## 2. Defects

- `ProximalBundleMethod` keeps per-run state on `self`
  (`numerics/convex.py`, the warm start and the cut cache), so one instance
  is not reentrant across threads.

## 3. Open design items

Each item below is still open as checked against the code.

**Feasible sets**

- `Cylinder`, a quadratic set with an explicit kernel basis, and with it the
  dual's affine constraint for degenerate priors. Neither exists.
  Semidefinite priors still go through `_covariance_of`
  (`geometry/feasible.py:2059, :2070, :444-454`).
- `Splitting`'s inner value is attained only to its feasibility tolerance,
  so that end of the bracket is not certified (`feasible.py:3097-3101,
  :3162`). It is documented only.
- The data-space S-procedure refuses where the data outnumber a slack prior
  (`feasible.py:2549-2552`). The model-space form is the default.
- Polytopes: there is no way to append a half-space, outer or inner.
- `support_sweep` with processes merges results but loses the workers'
  application counters (`feasible.py:3925-3929`).
- There is no rule on which cached pieces of a feasible structure may cross
  a process boundary. Today every cache is dropped (`feasible.py:296-303`).

**Preconditioners and solvers**

- The Woodbury preconditioner reads a Tikhonov member two ways: `from_normal`
  takes `prior_covariance` off it (`numerics/preconditioners.py:1129`), and
  `_invert` divides by `scale` (:1293-1300, :1348). A preconditioner built
  with `C = I` and handed a member at damping 10 inverts the member at
  damping 1 and divides by 10. One reading should win.
- Parallel randomized estimates oversubscribe (`random_eig(n_jobs=8)`
  through `apply_block` sat at 580% CPU in the parent with the workers idle).
  This has not been investigated.
- The QP backends keep Clarabel's absolute tolerances at tiny scales
  (`numerics/quadratic_programming.py:356, :446-448`).
- Renames not taken: LSQR's squared `damping` (`solvers.py:2036`), and the
  coexisting tolerance spellings `rtol`, `tolerance`, `gtol`, `ftol`,
  `relative_tolerance` and `absolute_tolerance`.
- `TrustRegionNewton` stays unpreconditioned, since its region would have to
  be measured in the preconditioner's norm, and its forcing is fixed where
  `NewtonCG` takes `forcing=`.
- `SpectralPreconditioner` takes no `factor=`. The waveform app builds that
  by hand (`miniapps/waveform/laplace.py:82-91`).
- Jacobi requires self-adjointness (`preconditioners.py:117`). Whether it
  should accept a nonsymmetric operator is undecided.

**Measures and estimators**

- `dense_limit` on `kl_divergence` (4000) and `ambient_ball` (1024) keys on
  dimension, not application count (`probability/gaussian.py:1607, :2326`).
  `log_determinant` already does the right comparison.
- A product measure's `L L*` is not folded block by block
  (`gaussian.py:747-758`). On a layered spectral-element space a covariance
  application costs two transform round trips, about 2.5 times the folded
  cost.
- A spectral-element measure calibrated by `pointwise_std=` has no precision
  (`spectral_element/base.py:1179-1184`). A symmetric space's invariant
  measure has a diagonal precision but no `precision_factor`
  (`symmetric_space/base.py:935-941`).
- `sample_pointwise_variance` and `sample_pointwise_std` could take
  `points=`, to sample the variance at chosen points directly. The
  docstring now says why reading the grid field off the grid smooths it.
- A mixture's one-dimensional marginals are closed form, but `moments` and
  `plot_densities` sample them (`plotting/distributions.py:71, :86, :387`).
  `LinearGaussianMixtureInversion` factorizes the same normal operator once
  per component under a shared covariance (`inference/mixture.py:106-111`).
- `covariance=` and `reference=` on `LeastSquares` and `MinimumNorm`, so
  that the prior-norm whitening the apps build by hand becomes a knob
  (`inference/point.py:102-110, :525-537`).
- Gates on `as_multivariate_normal` and `kl_divergence` on a function space
  exist in docstrings only.
- `random_diagonal` returns an array, not an `Estimate`
  (`numerics/randomized.py:1024`). An array-valued `Estimate` would be
  needed.
- `Functional.gradient` is `at(x).gradient` (`algebra/operators.py:2266`).
  For a composed functional the derivative route can be the cheaper one.
- Layered correlation (`Layered.sobolev_measure(correlation=)`) was withdrawn
  because the spectral bases' eigenvector signs were arbitrary. planetmodel
  1.4 now normalizes them, so this is unblocked but not built
  (`gaussian.py:687`).

**Spaces**

- `import pygeoinf` imports planetmodel whenever it is installed
  (`pygeoinf/__init__.py:287-288`, `plotting/__init__.py:40-43`). A module
  `__getattr__` with lazy renderer registration would defer it.
- There is no domain quadrature on the spectral-element spaces. The
  tomography app reaches into planetmodel for one
  (`miniapps/tomography/src/problem.py:455-459`).
- The length scale is not carried through `Lebesgue`.
  `sphere.Sobolev(8, 2, 0.1).with_order(0).with_order(2)` comes back with
  length scale 1.0.
- Radial pointwise calibration is accepted below the order that gives the
  centre a value: `radial.Sobolev(16, 1.0, 0.1).sobolev_measure(1.0,
  pointwise_std=1.0)` runs.
- There is no `heat_measure` on the spectral-element base class
  (`spectral_element/base.py:1063` has only `spectral_measure`).
- `jump_operator` assembles a dense matrix (`spectral_element/layered.py:617-626`).
- `_block_variance` stays local until planetmodel's `pointwise_variance_of`
  takes `physical=False` (`spectral_element/base.py:1029`).

**Checks and results**

- `check_operator(measure=)` probes the codomain with white noise, not a
  measure (`testing.py:766`).
- One `atol` default across the checks: some are 1e-12 (`testing.py:134,
  :207, :393`) and others 0.0 (:690, :859, :1021 and on).
- `OptimizationResult.history` holds values only. Per-iteration gradient
  norms would spare the waveform app its wrapper
  (`numerics/optimization.py:64-75`).

**Plotting**

- There is no public way to draw a path on a ball's section: the section
  frame is private (`plotting/ball.py:220`), and `plot_section_points`
  places markers only.

**Anticipated, not started**

- The data space on the nonlinear route: a Newton step and the Laplace
  covariance solve in the model space today. If the data space is wanted
  there, it is chosen through `solver=` as on the linear route, with
  `with_probed_derivative()` mirroring `with_probed_adjoint()` for problems
  with few data.
- Function-space sampling: presets `PCN`, `MALA`, `HMC`, `SMC` and results
  `SampledPosterior`, `WeightedEnsemble`, with the low-rank posterior as a
  reference.
- Process-safety declarations for operators, checkpointing in the
  optimizers, resource-owning linearizations, and a tensor-product space as
  a backend.
- The paper-scale feasible-set benchmarks (flexure–gravity, nested
  tomography), last seen in git at `8c53839` under `benchmarks/`. The dual
  did not converge on flexure–gravity at 1 500 bundle iterations.

## 4. Mini-apps

- The point-data app runs no checks on its operators
  (`miniapps/point_data/IMPLEMENTATION.md:103`). The planet-density app
  checks its linear rows only in its test. Each app should run the check
  ladder on its own operator in part A.
- Rows of derivative components into a Euclidean data space are declared
  `form="components"` in the planet-density app and `form="galerkin"` in the
  tomography, MFEM and banana-doughnut apps. Both are right, since the
  codomain's Gram is the identity, but one convention, or a named
  constructor, would spare the explanation.
