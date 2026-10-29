# Review: the examples and tutorials

Layer: `pygeoinf2/examples/` (32 numbered scripts), `pygeoinf2/tutorials/`
(14 scripts and their notebooks), and, for contradictions only, the
mini-apps' `README.md` and `IMPLEMENTATION.md`. Read with a user's eye
against the aim "briefly illustrate the core functionality of the
library". Every example and every tutorial was run (timings below, two
BLAS threads, several runs in parallel); every numeric claim in the prose
of tutorials 4, 6, 7, 8, 10, 14 and examples 21 to 28 was checked against
the printed output. The scripts named `tools/*.py` below are in
`/tmp/claude-1000/-home-david-dev-pygeoinf/4bec2589-8482-4f4c-843c-b27f331a8905/scratchpad/tools/` (session scratch, may not survive; each reproduction quotes the code that matters); nothing under `pygeoinf2/` was changed but this file.

## The coverage map

First appearance of each name the top-level `pygeoinf2/__init__.py`
exports (by AST: an import or an attribute access; "returned" or
"printed" where the reader meets the type only as a result). 51 of the
122 names (111 classes and functions, 11 subpackages) appear in no example
and no tutorial; most are base classes, the solver and preconditioner zoo
and the convex internals, but the inference names among them are core
(see Should 1 and 2).

| public name | first example | first tutorial |
|---|---|---|
| `algebra` | 5 | 13 |
| `geometry` | 17 | 7 |
| `inference` | 21 | 4 |
| `numerics` | 7 | 3 |
| `plotting` | 20 | 1 |
| `probability` | 11 | 10 |
| `symmetric_space` | 2 | 1 |
| `BackusGilbert` | 23 | 7 |
| `BackusGilbertParker` | 23 | 7 |
| `FeasiblePropertySet` | 23 (returned) | 7 (printed) |
| `ConstrainedLeastSquares` | — | — (named in 8) |
| `ConstrainedMinimumNorm` | — | — (named in 8) |
| `ComposedPointEstimator` | — | — |
| `ConstrainedMinimization` | — | — |
| `ForwardProblem` | 28 | 9 |
| `LeastSquares` | — | 8 |
| `LinearForwardProblem` | 21 | 1 |
| `LaplaceApproximation` | 28 | 9 |
| `LaplaceResult` | — | — |
| `LinearizedFeasiblePropertySet` | — | 14 (returned, `model=`) |
| `GaussianPosterior` | 21 (returned) | 6 (printed) |
| `LowRankGaussianPosterior` | — | — |
| `LinearGaussianInversion` | 21 | 1 |
| `MaximumAPosteriori` | 28 | 9 |
| `Minimization` | — | — |
| `NonlinearLeastSquares` | — | — |
| `Occam` | — | 14 |
| `LinearGaussianMixtureInversion` | 26 | — |
| `MinimumNorm` | — | 8 |
| `NormalOperator` | 24 (returned) | 4 |
| `TikhonovFamily` | — | — |
| `TikhonovNormalOperator` | — | — |
| `BandedPreconditioner` | — | — |
| `BiCGStabSolver` | — | — |
| `CGSolver` | 7 | 4 |
| `CholeskySolver` | 7 | 2 |
| `EigenSolver` | — | — |
| `FlexibleCGSolver` | — | — |
| `GMRESSolver` | — | — |
| `IdentityPreconditioner` | — | — |
| `JacobiPreconditioner` | 22 | — |
| `LocalizedPreconditioner` | — | — |
| `LSQRSolver` | — | — |
| `LUSolver` | — | — |
| `MinResSolver` | 7 | — |
| `NormalDiagonalPreconditioner` | 24 | 4 |
| `ProgressCallback` | — | — |
| `SolutionTrackingCallback` | — | — |
| `SolveResult` | 7 (returned) | 4 (printed) |
| `SolveStep` | — | — |
| `SpectralPreconditioner` | — | 4 |
| `WoodburyPreconditioner` | 24 | — |
| `plot` | 20 | 1 |
| `plot_corner` | 25 | 5 |
| `plot_densities` | 25 | — |
| `plot_points` | 21 | 13 |
| `subplots` | 20 | 1 |
| `ArrayVectorMixin` | — | — |
| `CoordinateSpace` | 2 | 2 |
| `DiagonalMetricSpace` | — | — |
| `EuclideanSpace` | 1 | 2 |
| `HilbertModule` | — | — |
| `HilbertSpace` | — | — (named in 12; built only in the cpp_backend and mfem apps) |
| `MassWeightedSpace` | — | 2 |
| `OrthonormalSpace` | — | — |
| `Reals` | — | — |
| `require_module` | — | — |
| `AffineOperator` | — | — |
| `EvaluationError` | 35 | 9 |
| `Functional` | 5 | 3 |
| `LinearFunctional` | 5 | 2 |
| `LinearOperator` | 3 | 2 |
| `MatrixLinearOperator` | — | — |
| `Operator` | 6 | 9 |
| `require_coordinates` | — | — |
| `DiagonalLinearOperator` | 8 (type printed) | 4 |
| `BlockDiagonalLinearOperator` | 22 | 10 |
| `BlockDiagonalOperator` | — | — |
| `BlockLinearOperator` | 22 | — |
| `BlockOperator` | 11 | — |
| `ColumnLinearOperator` | 22 | — |
| `ColumnOperator` | — | — |
| `DirectSum` | 11 | 10 |
| `RowLinearOperator` | — | 10 |
| `RowOperator` | — | — |
| `Linearization` | 6 | 9 |
| `QuadraticModel` | — | — |
| `AffineSubspace` | 18 | — |
| `Ball` | 17 | 7 |
| `BallSurface` | — | — |
| `ConvexSet` | — | — |
| `Ellipsoid` | 34 | 6 (returned by `credible_set`) |
| `EllipsoidSurface` | — | — |
| `EmptySet` | — | — |
| `HalfSpace` | 17 | — |
| `Hyperplane` | 17 | — |
| `LinearCertificate` | 23 | 7 |
| `LinearSubspace` | 18 | — |
| `OrthogonalProjector` | 18 | — |
| `Polytope` | 23 (returned) | 7 (returned by `inner_hull`) |
| `enclosing_ellipsoid` | — | — |
| `sum_of_ellipsoids` | — | — |
| `Subset` | — | — |
| `SublevelSet` | — | — |
| `LevelSet` | — | — |
| `UniversalSet` | 17 | — |
| `Box` | — | — |
| `Interval` | 14 | — |
| `Lebesgue` | 2 | 5 |
| `PeriodicBox` | — | — |
| `Sobolev` | 2 | 1 |
| `AbsolutelyContinuousMeasure` | — | — |
| `GaussianMeasure` | 10 | 1 |
| `GaussianMixture` | 26 | — |
| `ProbabilityMeasure` | — | — |
| `ProductMeasure` | 11 (returned by `product`) | 10 (returned) |
| `PushForwardMeasure` | 11 (type printed) | — |
| `product` | 11 | 10 |
| `Traits` | 4 | 2 |
| `spectral_element` | — | — |
| `interval` | 30 | 10 |
| `ball` | 31 | 11 |
| `disk` | — | — |

The recent additions and the patterns the mini-apps rely on, which are
methods or keywords rather than top-level names:

| feature | first example | first tutorial |
|---|---|---|
| `LowRankGaussianPosterior.from_linear_problem`, `modes`, `pointwise_variance_reduction` | — | — |
| `LaplaceApproximation(rank=)` | — | — |
| `GaussianMeasure.from_product(correlation=)` | — | — (10 §2 does the constant-correlation case with `correlated_measure_from_correlations`) |
| `Layered.sobolev_measure(correlation=)` | — | — |
| `GaussianMeasure.ambient_ball(level=)` | — | 14 |
| `ambient_ball(operator=)` (an `Ellipsoid` in another norm) | — | — |
| `CheckReport` returned by a check | — (every example discards it) | 3, 12 |
| `measure=` on a `check_*` | — | — |
| `LinearForwardProblem.parameterized(prior.covariance_factor)` (point answers in the prior's norm; all five mini-apps) | — | — |
| `credible_set` | — | 6 |
| `relative_log_density`, `log_evidence` | 26 (evidence) | 6 |
| `propagated_covariance`, `bias`, `resolution` | — | 7 |
| `n_jobs=` | — | 14 (`polytope`) |
| `ray_travel_time_operator` | — | 13 |
| `disk` (space, `Layered`) | — | — |
| a `HilbertSpace` with no coordinates | — | — |

## Must fix before 2.0

### Tutorials 9 and 13 teach the identity covariance on a function space
`tutorials/09_nonlinear.py:109`, `:182`; `tutorials/13_ray_tomography.py:257-258` — Tutorial 9 builds its prior as "white noise on the model space", `X.invariant_measure(np.ones(X.dim), pointwise_std=0.8)` on `H^2` of the circle, draws its truth from it, and argues for it as the prior "that suits the optimiser". Tutorial 13 builds `GaussianMeasure(X, covariance=identity, precision=identity)` on `H^2` of the plane as the latent prior of `m = L w` and runs MAP and Laplace on it. Both are the cardinal sin of D-190: the covariance is not trace class on the space, the draws are not in it, and the pointwise calibration only hides that. D-168 accepted tutorial 13's whitening as "exact for MAP and Laplace"; I think D-190 supersedes that ("never", in its words), and D-168 itself supplied the replacement, `NewtonCG(preconditioner=prior.covariance)`, which tutorial 14's coda already uses on the same ray problem. The optimizer argument of tutorial 9 does not survive measurement either: with a trace-class prior of order 1 and `MaximumAPosteriori(..., gauss_newton=True, optimizer=NewtonCG(preconditioner=prior.covariance))` the MAP search takes 10 iterations at 128, 256 and 512 grid points (L-BFGS 188, 190, 196), where the white prior's L-BFGS takes 285 to 301. The mini-app walkthroughs send readers to exactly these two tutorials for the nonlinear machinery (`miniapps/waveform/IMPLEMENTATION.md:10`), and those walkthroughs say the library refuses such priors. Verified by running.
Reproduction: `X = circle.Sobolev(n, 2.0, 0.2); mu = X.invariant_measure(np.ones(X.dim), pointwise_std=0.8)` gives a mean `H^2` norm of a draw of 8.06, 16.1, 32.5 at `n` = 128, 512, 2048 and a trace of 65.5, 262, 1049 (`tools/white.py`); the Newton-CG counts are from `tools/t9.py`.
Proposed fix: tutorial 9 takes `X.sobolev_measure(1.0, 0.2, pointwise_std=0.8)` (or order 2) and the preconditioned Gauss-Newton Newton-CG, and replaces its "mesh-independent" table with that one; tutorial 13 drops the whitening and runs MAP and Laplace in the model space with `preconditioner=prior.covariance`, as tutorial 14's coda does.

### Three constructors still accept an identity covariance on a function space
`probability/gaussian.py:376` (`from_standard_deviations`), `symmetric_space/base.py:832` (`invariant_measure`), the `GaussianMeasure` constructor — D-195 (2) made `from_standard_deviation` refuse on `is_function_space`; its plural sibling, which "generalizes" it and whose own error message tells the caller to "start from the isotropic `from_standard_deviation`", has no such check, nor does the flat-spectrum `invariant_measure` that tutorial 9 uses, nor the raw constructor given `LinearOperator.identity(X)` that tutorial 13 uses. Example 10 tells the reader "the identity is not [trace class], and the library refuses it", which is true of one door in four. This overlaps the probability layer; it is reported here because two tutorials rest on it. Verified by running.
Reproduction (`tools/backdoor.py`), on `X = circle.Sobolev(128, 2.0, 0.2)`: `from_standard_deviation(X, 1.0)` refused (TypeError); `from_standard_deviations(X, np.ones(X.dim))` accepted; `X.invariant_measure(np.ones(X.dim))` accepted; `GaussianMeasure(X, covariance=LinearOperator.identity(X), precision=...)` accepted; `X.sobolev_measure(0.4, 0.2)` refused.
Proposed fix: `from_standard_deviations` refuses on `is_function_space` with `unsafe=True`, as its sibling does; the raw constructor refuses the one form it can recognize, an identity or scaled-identity covariance on a function space; `invariant_measure` cannot decide trace class from a finite spectrum, so its docstring says so and points to `sobolev_measure`.

## Should fix

### The recent additions have no example, and two have no tutorial either
`examples/README.md` (the sequence), `CURRENT_STATE.md` §10 item 3 — The low-rank posterior (`LowRankGaussianPosterior`, `LaplaceApproximation(rank=)`, D-198) and the correlated product (`from_product(correlation=)`, `Layered.sobolev_measure(correlation=)`, D-196) appear in no example and no tutorial, though §10 assigns an example of each to this review. `ambient_ball` appears once (tutorial 14, `level=` only); `operator=` nowhere outside the C++ app. The parameterized pattern every mini-app uses for its point answer (size in the prior's norm, least squares at damping one being the posterior mean) appears nowhere, though four of the five walkthroughs cite "tutorial 8" for it. And the design's one idea, a space with no coordinates, is asserted in examples 2 and 7 ("Nothing touched a component array") but never shown: every space in every example and tutorial is a `CoordinateSpace`. Tutorial 10 §2 builds a constant correlation of 0.8 between two fields with `correlated_measure_from_correlations`, the case D-196 made `from_product(correlation=)` for. Verified by grep and by running drafts of each proposed example (below), all well under a few seconds.
Reproduction: `tools/drafts.py`, `tools/lowrank.py`, `tools/laplace_rank.py`, `tools/param.py`. On a circle with 200 point data, the adaptive `from_linear_problem` picks rank 25 in 0.05 s and matches `LinearGaussianInversion`'s mean to 1e-10; `LaplaceApproximation(est, rank=40)` matches the full Laplace pointwise variance to 5e-10 in 0.12 s against 1.44 s; `from_product([T, S], correlation=[[1, .8], [.8, 1]])` gives an empirical point correlation of 0.808 and an unobserved field's posterior standard deviation of 0.60, the `sqrt(1 - 0.8^2)` floor; `ambient_ball(level=0.95)` has radius 4.1887, 4.1889, 4.1889 at 128, 512, 2048 points and `operator=` (the `L^2` form on `H^2`) returns an `Ellipsoid`; `L(LeastSquares(problem.parameterized(L), damping=1.0)(data))` equals the posterior mean to 5e-10.
Proposed fix: four new examples, one idea each (numbers as in the reordering below): **no_coordinates**, a `HilbertSpace` whose vectors are plain Python lists: CG and a Gaussian run on it, `CholeskySolver` and `matrix()` refuse by name; **correlated_product**, two fields tied through their latent noise: marginals unchanged, the cross-covariance `R_ij L_i L_j*`, an unmeasured field informed only through `R`; **point_estimates**, `LeastSquares`, `MinimumNorm`'s discrepancy search and `BackusGilbert` with `resolution`, `propagated_covariance` and `bias`, and the problem parameterized by the prior's factor so that size is the prior's norm; **low_rank_posterior**, the eigenvalues of `L* A* R^-1 A L`, the low-rank posterior at ranks either side of the eigenvalue of order one against the full one, `pointwise_variance_reduction`, a draw at the cost of a prior draw. Then `ambient_ball(level=)` and `ambient_ball(operator=)` go into the cut-down feasible-set example, `LaplaceApproximation(rank=)` into the nonlinear one, `Layered.sobolev_measure(correlation=)` into the layered planet, and tutorial 10 §2 switches to `from_product(correlation=)` with one sentence on when the scale-dependent `correlated_measure` is wanted instead.

### The reference sequence never shows a point estimator, and its worked example bypasses the estimators
`examples/15_worked_example.py:44-71` — Example 15, "all of it, on a small inverse problem", forms `A Q A* + R` by hand, solves it with `CGSolver`, and writes the posterior covariance as `Q - K A Q` with a hand-built Kalman operator; `LinearForwardProblem` and `LinearGaussianInversion`, the one call every later example uses, first appear at 21. `LeastSquares`, `MinimumNorm`, `ConstrainedLeastSquares` and `ConstrainedMinimumNorm` appear in no example, and `BackusGilbert` only as the centre of a linear certificate in 23, its error budget never asked for. So a reader of the examples alone meets the measure answer and the set answer, never the point answer, which D-184 puts first. Verified by grep.
Reproduction: `grep -l "LeastSquares\|MinimumNorm\|propagated_covariance" pygeoinf2/examples/*.py` prints nothing.
Proposed fix: rewrite 15 on `LinearForwardProblem` (chi-squared test, synthetic data) and `LinearGaussianInversion`, keeping the hand-built formula as a two-line cross-check of what the estimator does inside; add the point-estimates example above; one sentence in it for the constrained forms.

### The order of the examples, and their numbering
`examples/README.md`, `tests/test_examples.py` — The numbers skip 16, 27 and 29 (the MFEM examples that became mini-app 7). Threads are split: the proximal machinery is 13 and again 17 and 18, with fields and the worked example between; 13 and 17 both minimize `SquaredDistance` under a ball indicator with `ProximalGradient` (`BallIndicator(X, radius=0.2)` against `Ball(X, radius=0.25).indicator()`), the same computation twice; evaluation failure (35) sits 23 places from optimization (12) and 7 from the nonlinear MAP (28); the 3-D sets (34) sit after the planet spaces, far from 17 and 23; the answers come measure (21), set (23), never point. Read only.
Reproduction: the README's table.
Proposed fix: renumber without gaps in the brief's order; one possible sequence (source number in brackets, **new** marked):

| # | name | content |
|---|---|---|
| 1-6 | spaces .. nonlinear | (1-6) unchanged |
| 7 | no_coordinates | **new**, see above |
| 8-11 | solvers, functional_calculus, randomized, optimization | (7, 8, 9, 12) |
| 12 | evaluation_failure | (35) |
| 13 | measures | (10) |
| 14 | joint_law | (11, direct sums) |
| 15 | correlated_product | **new** |
| 16 | sets | (17, with 13's shrinkage and support-function algebra folded in) |
| 17 | subspaces | (18) |
| 18 | sets_in_three_dimensions | (34) |
| 19-21 | fields, observation, flexure | (14, 19, 20) |
| 22 | forward_problem | (15, rewritten) |
| 23 | point_estimates | **new** |
| 24 | tomography | (21) |
| 25 | low_rank_posterior | **new** |
| 26-29 | distributions, coupled_fields, preconditioning, mixture | (25, 22, 24, 26) |
| 30 | feasible_set | (23, cut to one idea, with `ambient_ball`) |
| 31 | nonlinear_map | (28, plus `rank=`) |
| 32-35 | varying_length_scale, mantle, density_profile, layered_planet | (30-33) |

The cost is the 22 "example N" citations in the examples, tutorials and `CURRENT_STATE.md`.

### The checks are run without their report and without a measure
`examples/03_operators.py:39`, `04_traits.py:47`, `05_derivative_and_gradient.py:74`, `06_nonlinear.py:54`, `17_sets.py:58`, `18_subspaces.py:30`, `19_observation.py:50`; `tutorials/13_ray_tomography.py:130-142`, `:182-197` — Every example discards the `CheckReport` its check returns and prints "passed"; no example or tutorial passes `measure=`, so every check on a function space probes with white noise, which the mini-app walkthroughs call "rough in every mode and not the kind of field the operator will meet" (`miniapps/cpp_backend/IMPLEMENTATION.md:111-118`, `point_data/IMPLEMENTATION.md:114`). Tutorial 13 writes its derivative and Hessian checks as hand-rolled central differences although `check_derivative` and `check_second_derivative` exist and take the prior. Verified by grep and by running the library checks on tutorial 14's ray operator.
Reproduction: `check_derivative(F, truth, measure=prior, step=1e-4)` gives `CheckReport(check_derivative, worst=3.56e-07)` in 0.4 s and `check_second_derivative(...)` `worst=4.12e-06` in 0.5 s (`tools/raycheck.py`).
Proposed fix: print the report in 3 and 6 (one line each) and pass `measure=` a Sobolev measure in 3 and 19; tutorial 13 calls the two checks with `measure=prior`.

### Tutorial 7's prose says the opposite of what it prints
`tutorials/07_properties.py:122` — "The bias on the truth is larger than the propagated standard deviation, which is the usual situation and the reason the propagated error is not 'the uncertainty'." The cell above prints bias 0.007 and 0.0003 against propagated standard deviations of 0.050 and 0.093; the error on arc 0 is a 2.2-sigma noise term. `bias` is right (it equals `resolution(m) - T m` exactly); the narrative no longer matches `BackusGilbert` as D-165 defined it. Tutorial 14 makes the same point correctly with `MinimumNorm`. Verified by running.
Reproduction: `tools/t7bias.py`: `bias() [0.00719 0.00032]`, `propagated sd [0.0496 0.0933]`, noise term `[-0.109 0.033]`.
Proposed fix: rewrite the paragraph around what is printed (the resolution kernels, and that the bias is small for this truth and large for another, which is why it cannot be quoted), or show a truth for which the bias dominates, and leave the "larger than" lesson to tutorial 14.

### Tutorial 13 reports an unconverged L-BFGS run as an answer, and cites a tutorial that is not there
`tutorials/13_ray_tomography.py:280`, `:311`, `:124`, `:234` — L-BFGS stops at its cap of 100 iterations with a gradient norm of 70.5; the table lists its relative error beside Newton's and the text says "The two nonlinear answers agree with each other" just after the cell has printed that they differ by 8.6e-02. Lines 124 and 234 call the derivative "the linear operator of tutorial 10" and "the path-integral operator of tutorial 10"; tutorial 10 is coupled problems, and no tutorial has a path operator (examples 19 and 21 do). Verified by running.
Reproduction: `tools/t13.py`: `L-BFGS: False 100 iteration limit reached grad 70.53`, `Newton: True 13 value tolerance reached grad 9.95e-07`.
Proposed fix: with the preconditioned Newton-CG of the first Must finding the L-BFGS row can go, or the text can say plainly that it did not converge; point the two references at example 19.

### Example 22's reason for its preconditioner is no longer true
`examples/22_coupled_fields.py:133-141` — "Unpreconditioned conjugate gradients loses orthogonality on that and reports a residual of nan. Jacobi is the whole fix." Plain CG converges on the example's normal operator in 117 iterations to `rtol=1e-10`; Jacobi takes 29; the solutions agree to 3e-11. Verified by running.
Reproduction: `tools/ex22.py`: `plain 117 True ... jacobi 29 True`, `solutions agree to 2.99e-11`.
Proposed fix: say what is true: the units make the operator badly scaled and Jacobi cuts the iterations fourfold.

### Example 12's metric demonstration is degenerate and mislabeled
`examples/12_optimization.py:62-77` — It minimizes `||x - a||^2 / 2` in the space's norm, whose Hessian is the identity, so steepest descent takes 1 iteration at every metric, which is printed; the claim that "in components ... the count grows with it" is asserted and not shown; and "the metric spread over four decades" prints condition numbers of 4.2e3, 4.1e11 and 4.1e19 (the length scale spans four decades, the metric sixteen). Tutorial 3 makes the same point properly (40 against 175 iterations at 32 points). Verified by running.
Reproduction: the example's output, `metric condition number 409601280001.0 -> 1 iterations`.
Proposed fix: run the same objective on the component array beside it, as tutorial 3 does, and say what spans four decades.

### Example 23 is six ideas and largely tutorial 7 again
`examples/23_feasible_set.py` — 266 lines: support values with brackets, the Backus-Gilbert estimate, the linear certificate and its split, a ray-coverage count, an axis box and a 12-direction polytope, an inner hull, the S-procedure ellipsoid, an extremal model and three figures; tutorial 7 makes the same calls in the same order. The prior ball's radius is a hand-picked 3.0, not `ambient_ball`, and the bound the prior ball alone admits is never printed, so the reader cannot see what the data bought (D-184's convention for the apps' set figures, and tutorial 7's `prior_only` line). Read and run.
Reproduction: output, `cap 0: [-6.39994, +6.41118] truth +0.28238`, with nothing to compare the width against.
Proposed fix: cut 23 to the set answer on the sphere: `prior.ambient_ball(level=0.95)` and `ambient_ball(operator=...)` as two constraint sets, the brackets against the prior-alone bound; leave certificate, hull and ellipsoids to tutorial 7.

### Tutorial 7 spends 81 of its 93 seconds drawing one set, and the hull is drawn in the wrong colour
`tutorials/07_properties.py:245`, `:248`; `examples/23_feasible_set.py:254`; `plotting/sets.py:48`, `:349` — `plot_set(feasible)` takes the support route with its default `directions=180`, 180 serial support optimizations, and `plot_set` has no `n_jobs=` though the set has `support_sweep`. `plot_set(polytope, color=...)` on the level route passes `color` to `contourf`, which ignores it with a `UserWarning` that both scripts print; the inner hull comes out in viridis teal while the titles call it green. These are plotting defects, seen through this layer. Verified by running.
Reproduction: `tools/t7.py`: `feasible set 81.2` s, `inner hull 6.0` s; `tools/t7b.py`: `warnings: ["The following kwargs were not used by contour: 'color'"]`, facecolor `(0.128, 0.567, 0.551)`.
Proposed fix: the tutorial passes `directions=36`; `plot_set` takes `n_jobs=` and sweeps through `support_sweep`; the level route maps `color` to `colors=`. Tutorial 7 could then leave the slow set.

### History and planning talk in two tutorials
`tutorials/02_spaces.py:168`, `tutorials/09_nonlinear.py:254-255` — "(Version 1 did, and converting between the two was a place mistakes lived.)" and "which is where the library's sampling layer will start". The rule is no v1 and no planning talk anywhere; `test_code_practice` exempts the tutorials. Verified by grep (no other hits in examples or tutorials).
Reproduction: `grep -n "Version 1\|will start" pygeoinf2/tutorials/*.py`.
Proposed fix: delete the parenthesis; end tutorial 9 at "what a sampler needs to correct the approximation".

### Tutorial 12 says white noise follows from the contract; the default is a dense factorization
`tutorials/12_writing_a_backend.py:36-48`; `algebra/spaces.py:412-429` — The tutorial lists six required members and says "the inner product, the norm, white noise, matrices and everything above them follow". For a `CoordinateSpace` the default `white_noise_components` "forms the Gram matrix densely and factorizes it once" (its docstring): `dim` applications of `apply_gram` and a dense Cholesky, which a backend author following the tutorial at a million unknowns meets on the first draw, and which every check and randomized method makes. `CURRENT_STATE.md` §6 lists `random` and `white_noise` as part of what a backend supplies (D-169); the two documents disagree. Read only.
Reproduction: the docstring at `algebra/spaces.py:419`.
Proposed fix: the contract table gains a row, "`white_noise_components`, at scale: `L^-T xi` from a sparse factor of the Gram matrix; the default is dense", and §6 and the tutorial say the same.

### The low-rank posterior's mean is far off when the rank drops informative directions
`inference/gaussian.py:965-1015` — `from_linear_problem` computes the mean as `m_0 + L (I - V D V*) L* A* R^-1 r`; in a dropped direction with eigenvalue `lambda` the correction is not shrunk by `1/(1 + lambda)`, so the mean errs by a factor of about the first dropped eigenvalue. The class docstring's "always between the prior and the full posterior" is true of the covariance only, and nothing warns about the mean. The adaptive default chooses well; an explicit small `rank=`, which an example would naturally show, does not. Verified by running; it bears on how the proposed example must be written.
Reproduction: `tools/lowrank.py` (200 point data on a circle, heat prior): rank 4, next eigenvalue 1648, mean off by 7.5e+02 of its norm; rank 8, next 37.5, 9.9e-01; rank 12, next 0.12, 2.2e-04; adaptive rank 25, 1e-10.
Proposed fix: the docstring of `from_linear_problem` says the mean is accurate once the dropped eigenvalues are small against one; the example prints the eigenvalues and picks the rank by them.

### Examples build by hand what a constructor provides
`examples/24_preconditioning.py:57-69`, `examples/26_mixture.py:45-62` — Example 24 builds diagonal noise from two `from_matrix` calls with traits, 13 lines, which is `GaussianMeasure.from_standard_deviations(D, quality)` (used in 32 and 33); example 26's `scenario` helper, 18 lines with a Cholesky precision, is `GaussianMeasure.from_standard_deviation(X, np.sqrt(spread), expectation=...)` on its Euclidean space. A reader copies the long way. Read only.
Reproduction: the two constructors' signatures, `probability/gaussian.py:312`, `:376`.
Proposed fix: use the constructors.

### The Colab notebooks cannot run, and the rename must regenerate them
`tutorials/make_notebooks.py:37-50`, `tutorials/README.md` — Every badge links to `github.com/da380/pygeoinf/blob/main/pygeoinf2/tutorials/notebooks/...` and every notebook's first cell installs `pygeoinf @ main` and imports `pygeoinf2`; `main` has neither the notebooks nor `pygeoinf2`, and after the rename the path and the import both change. Verified with `git cat-file`.
Reproduction: `git cat-file -e main:pygeoinf2/__init__.py` fails; so does `origin/main:pygeoinf2/tutorials/notebooks/01_first_inversion.ipynb`.
Proposed fix: put "rewrite the badge paths and the import name in `make_notebooks.py`, regenerate, and open one notebook on Colab" into the release mechanics of §10.

### The mini-app walkthroughs teach patterns the tutorials do not show, and spell one of them three ways
`miniapps/point_data/IMPLEMENTATION.md:160`, `:272-281`; `tomography/IMPLEMENTATION.md:286-305`; `cpp_backend/IMPLEMENTATION.md:228-260`; `mfem/IMPLEMENTATION.md:304-310`; `planet_density/IMPLEMENTATION.md:275-290` — All five give their point answer through `problem.parameterized(prior.covariance_factor)`, and four cite "tutorial 8" for the deterministic answers "with size measured in the prior's norm"; tutorial 8 measures size only in `L^2`, `H^1` and `H^2`. To take the truth into `w` for its bias they spell the inverse three ways: `prior.precision_factor` (planet density, C++), `factor.inverse` (point data), `factor.adjoint @ prior.precision` (MFEM, tomography); §7's finding (5) is one cause. The point-data walkthrough cites tutorial 7 for the ambient ball's radius; tutorial 7 picks `1.5 * ||truth||` by hand and `ambient_ball` is in 14. The apps that run the check ladder probe with `measure=prior`, and every app with a set part hardens with `ambient_ball`; the examples do neither (above). Read only.
Reproduction: `grep -n "parameterized\|whiten =" pygeoinf2/miniapps/*/IMPLEMENTATION.md pygeoinf2/miniapps/*/point.py`.
Proposed fix: the point-estimates example shows the parameterized pattern and one spelling of the inverse; the walkthroughs cite it; `point_data/IMPLEMENTATION.md:160` cites tutorial 14.

## Could fix

### Top-level names imported from four depths
`examples/21_tomography.py:20-21`, `examples/28_nonlinear_map.py:29-33`, and 20 other files — The package exports the workflow "flat", and the tutorials mostly import it so; examples 21 to 33 import the same names from `pygeoinf2.inference`, `pygeoinf2.probability.gaussian`, `pygeoinf2.algebra.operators`, and 28 `ForwardProblem` from `pygeoinf2.inference.problem`, beside `import pygeoinf2 as gi`. Read only (AST scan, `tools/imports.py`).
Reproduction: `from pygeoinf2.probability.gaussian import GaussianMeasure` in 21, 22, 24, 25, 30-33; `from pygeoinf2 import GaussianMeasure` in 15 and every tutorial.
Proposed fix: `from pygeoinf2 import ...` for exported names, subpackage paths for the rest.

### Two spellings of the circle
`examples/02_coordinates.py:17` and examples 3-18; `tutorials/01_first_inversion.py:22` and every tutorial — The examples write `symmetric_space.Sobolev((16,), 2.0, 0.3)` (the periodic box with a shape tuple), the tutorials `symmetric_space.circle.Sobolev(16, 2.0, 0.3)`. Both are right; a reader sees two APIs for one space before being told they are one. Read only.
Proposed fix: the named geometries in the examples too, with the box once in "fields" as what they are.

### Example 20 is peripheral, and example 22's diagnostic is most of its cost
`examples/20_flexure.py`, `examples/22_coupled_fields.py:163-204` — 20 has no inference in it and illustrates one sphere operator, a variable-coefficient flexure solve (the mini-app plan dropped flexure as "contrived"); it exists mainly as 22's prerequisite. 22's coupling diagnostic forms three covariance blocks through about three thousand solves, 30 of its 32 s, to print 0.0000 and 0.0836. Run.
Proposed fix: fold 20's operator into 22 and drop 20, or keep both and cut the diagnostic to one block.

### No example of the disk, and none of parallelism
`disk` (D-175), `n_jobs=` — The disk, the two-dimensional ball, is in no example or tutorial; `n_jobs=` appears once, on a polytope in tutorial 14. Read only.
Proposed fix: a short disk example beside the mantle (a field on an annulus, sections and a profile), and `n_jobs=` on one sweep in the feasible-set example.

### The examples README drifts from what it describes
`examples/README.md` — "Short scripts, one new idea each" is untrue for 22 to 28 (150 to 266 lines); the optional-dependency table, which says it is "the one `tests/test_examples.py` enforces", omits pyshtools for 31, which the test lists. `tests/test_examples.py` marks 24 slow (8.8 s here) and not 23 or 25 (7.7, 7.8 s). Run and read.
Proposed fix: regenerate the table from the test's `OPTIONAL`; revisit `SLOW` after the cuts above.

### CURRENT_STATE's counts of the examples and tutorials disagree with each other and with the tree
`CURRENT_STATE.md:12`, `:59-61`, `:506` — "the 35 examples and 13 tutorials", then "Thirty-two examples ... run as a slow test" and "Fourteen tutorials ... run in the fast suite"; there are 32 examples, of which 2 are slow, and 14 tutorials, of which 3 are slow; §10 still says the mini-apps "are new and not started", and `EXAMPLES_PLAN.md` still plans "a CheckReport walk ... (a coda to 16 and 29)", two examples that are gone. Read only.
Proposed fix: correct the counts; strike the stale plan items.

### Small wording against the output or the principles
`examples/14_fields.py:42-43`, `examples/21_tomography.py:76`, `examples/28_nonlinear_map.py:181`, `tutorials/12_writing_a_backend.py:269` — 14 labels draws of order-1 and order-4 measures on `H^2` "an H1 sample" and "an H4 sample" (their draws have 2.5 and 5.5 derivatives; the order is the covariance's against the space, D-190's language); 21 says the normal equations are "assembled" two lines after "applied, never assembled"; 28 says "Every linear method in this library returns a mean", which `LeastSquares`, `MinimumNorm` and `BackusGilbert` do not; 12 says the adjoint defect "falls steadily" over 3.1e-2, 3.4e-2, 7.6e-3, 3.1e-3, white-noise probes making it noisy (probes from a smooth measure, as the checks finding above asks, would likely make it clean). Run and read.
Proposed fix: "draws of an order-1 measure"; "formed in the data space"; "every Bayesian linear method"; "falls under refinement".

### The tutorials' answers come measure, set, point
`tutorials/06_*.py`, `07_*.py`, `08_*.py`, `14_one_problem_three_answers.py` — The course meets the measure (6), the set (7), then the point (8), not the order of treatment the mini-apps follow (point, measure, set, D-184); tutorial 14 uses point, set, measure in its linear part and its figure columns, and point, measure, set in its coda. The backend tutorial (12) also interrupts the inference arc 9, 13, 14. A judgment, not a defect.
Proposed fix: at least make tutorial 14 point, measure, set throughout; if the course is renumbered (6 point answers from today's 8, 7 the posterior, 8 the set; backend last), the cost is the 67 tutorial-number citations in the tutorials and examples and 70 in the mini-app walkthroughs.

### Unshown exports, for the `__init__` review
`pygeoinf2/__init__.py` — Names exported flat that no example or tutorial uses: the inference classes `ComposedPointEstimator`, `ConstrainedMinimization`, `Minimization`, `NonlinearLeastSquares`, `TikhonovFamily`, `TikhonovNormalOperator`, `LaplaceResult`; six solvers and three preconditioners (`BiCGStabSolver`, `EigenSolver`, `FlexibleCGSolver`, `GMRESSolver`, `LSQRSolver`, `LUSolver`, `BandedPreconditioner`, `IdentityPreconditioner`, `LocalizedPreconditioner`), the callbacks and `SolveStep`; and the bases `ArrayVectorMixin`, `DiagonalMetricSpace`, `OrthonormalSpace`, `HilbertModule`, `Reals`, `ConvexSet`, `Subset`, `ProbabilityMeasure`, `AbsolutelyContinuousMeasure`, `Box`, `PeriodicBox`. Not every export needs an example; the inference ones are the gap (Should 1 and 2), and the rest is a question of what belongs at the top level. Read only.

## What is sound

- **Everything runs.** All 32 examples and all 14 tutorials ran to completion: examples in 0.6 to 3 s except 23, 25, 24 (7.7, 7.8, 8.8 s) and 22 (32 s); tutorials in 1 to 10 s except 14 (33 s), 13 (51 s) and 7 (93 s). The notebooks are current and carry no outputs, and the README-listing tests pass.
- **The examples' priors are posed right.** Every Gaussian on a function space in the examples is trace class on its space: Sobolev orders 1 on a line (10, 14, 30), 2 on the sphere and circle (15, 19, 20), 1.25 for 22's traction (above 1 on the sphere), 2 in the mantle (31), 1.5 on the radial and layered spaces (32, 33), heat measures elsewhere; identity measures appear only on Euclidean spaces (11, 26, 28). The refusals of D-131 and D-190 are taught as features in example 10 and tutorials 5 and 8, and the `unsafe=` uses (examples 10, 19; tutorials 5, 8) each say why.
- **No decision numbers, history or planning talk** anywhere in the examples; in the tutorials only the two lines above.
- **The numbers in the prose match the output** in tutorials 4 (417 iterations, 265 with the spectral preconditioner, the 1% trace), 5 (the refinement tables), 6 (the evidence peaking at the generating length scale), 8 (the damping table and discrepancy choice), 10 (the 0.60 correlation floor), 14 (every bias, bracket and interval statement), and examples 9, 21, 24 (1051 to 10 iterations, blocked 531 against exact 604), 25, 26 and 28.
- **Tutorials 2, 3, 12 and 14 are the strongest material**: 2 and 3 teach the metric on a non-diagonal mass matrix and show the derivative-against-gradient cost under refinement (40 against 175 L-BFGS iterations at 32 points); 12 runs the ladder, breaks it three ways and reads the `CheckReport`; 14's linear part is the inference pattern made concrete, and its coda already shows the preconditioned Newton-CG the first Must finding asks of 9 and 13.
- **The recent features work from a user's eye.** Drafts of the proposed examples ran in under two seconds each: the adaptive low-rank posterior exact to 1e-10, `LaplaceApproximation(rank=)` matching the full Laplace to 5e-10 twelve times faster, the correlated product reproducing its correlation and its posterior floor, `ambient_ball`'s radius converging under refinement and its `operator=` form returning an `Ellipsoid`, the parameterized least squares equal to the posterior mean, and `check_derivative` and `check_second_derivative` with `measure=` on the ray operator.

## Fixed

Fix package F, 2026-10-29. The examples are renumbered gap-free in the
proposed order, with a disk example inserted between the interval and the
mantle (so the varying-structure group is 32 interval, 33 disk, 34 mantle,
35 density profile, 36 layered planet); the tutorial course is reordered
point, measure, set. Every example and tutorial was run after the change;
the numbers the prose quotes were re-measured against the output. The tests
covering every item are `tests/test_examples.py` (each example runs and
prints, fast and slow; README listing) and `tests/test_tutorials.py` (each
tutorial runs; README listing; notebooks current), with the two new tests
named where they apply.

### Must fix

- **Tutorials 9 and 13 teach the identity covariance; three constructors
  accept it.** Closed by package A (the identity gate on every route, both
  tutorials re-posed on trace-class priors with the prior covariance as the
  Newton-CG preconditioner). Here: tutorial 13's prose re-measured after its
  checks changed the random stream (Newton now 14 iterations, "a dozen or
  so"; the ray's response "as large as the rest or larger").

### Should fix

- **The recent additions have no example.** Four new examples, one idea
  each, all under two seconds: `examples/07_no_coordinates.py` (a
  `HilbertSpace` whose vectors are Python lists; `check_operator` and
  `check_traits` reports, CG against the continuous solution, a Gaussian
  through a factor from `R^30` sampled and pushed forward, `CholeskySolver`
  and `matrix()` refused by name), `examples/15_correlated_product.py`
  (marginals unchanged to 4e-14, cross-covariance `0.8 L_1 L_2*` to 4e-14,
  the pointwise correlation, an unmeasured salinity at the
  `0.5 sqrt(1 - 0.8^2)` floor, the independent product leaving it at the
  prior), `examples/23_point_estimates.py` (below),
  `examples/25_low_rank_posterior.py` (the eigenvalues of `L* A* R^-1 A L`,
  ranks 15 and 19 either side of the eigenvalue of order one and the
  adaptive rank against the full posterior, `pointwise_variance_reduction`
  against the full pointwise variance, a draw at zero forward applications
  against 92). `ambient_ball(level=)` and `ambient_ball(operator=, inverse=)`
  (the `L^2` form on `H^1.5`) are the two constraint sets of the cut
  feasible-set example (30); `LaplaceApproximation(rank=)` is in the
  nonlinear one (31), ranks 2 and 1 with the eigenvalues, the dropped one
  shown to be far above one. Tutorial 10 §2 uses
  `from_product(correlation=)`, with one sentence on `correlated_measure`
  for a scale-dependent correlation (same output, the 0.60 floor).
  `Layered.sobolev_measure(correlation=)`: skipped, see below.
- **No point estimator; the worked example bypasses the estimators.**
  `examples/22_forward_problem.py` rewritten on `LinearForwardProblem` (the
  chi-squared test) and `LinearGaussianInversion`, the hand-built
  `Q A* (A Q A* + R)^-1 d` kept as a two-line cross-check (7e-10).
  `examples/23_point_estimates.py` opens with decision 5's framing and shows
  each estimator once, on the problem parameterized by the prior's factor:
  `LeastSquares` at damping one is the posterior mean (7e-9);
  `MinimumNorm`'s discrepancy search (damping, probes, Krylov iterations,
  misfit at the critical value, size in the prior's norm against the
  posterior mean's); `BackusGilbert(covariance=prior.covariance)` with
  `resolution()`, `propagated_covariance()` and `bias(truth)`, completed by
  the prior into the posterior of `T m` (mean 1e-8, covariance 4e-7); one
  sentence for the constrained forms. `covariance=` on the point estimators
  was not built by package D, so the parameterization is the route.
- **The order and numbering.** Renamed with `git mv`; 13 (`convex`) merged
  into 16 (`sets`: the shrinkage by FISTA and the support-function algebra
  folded in, the duplicate ball-constrained solve kept once). Citations
  followed through the examples, the tutorials, `examples/README.md`, the
  mini-app walkthroughs (`cpp_backend`, `mfem`, `planet_density`,
  `point_data`, `tomography` `IMPLEMENTATION.md`), `tests/test_examples.py`
  (`OPTIONAL`, `NEEDS_COASTLINES`, `SLOW`, the discovery count),
  `tests/test_normal.py` and `tests/test_plotting.py` docstrings, and
  `CURRENT_STATE.md` (the counts, "Example 34" now 18, "tutorial 6" now 7,
  "tutorial 10's tomography" now 13).
- **The checks run without their report and without a measure.** Every
  check in the examples now prints its `CheckReport` (3, 4, 5, 6, 7, 13 with
  each defect, 16, 17, 20); on function spaces the probes come from a
  Sobolev measure through `measure=` (3, 5, 16, 17), in 20 from the prior,
  built before the checks for that. Tutorial 13 calls
  `check_derivative(F, truth, measure=prior, step=1e-4)` and
  `check_second_derivative(...)` in place of its hand-rolled differences
  (8e-8 and 5e-7), keeping only the symmetry comparison by hand.
- **Tutorial 7's bias claim** (now `tutorials/08_properties.py`). The
  paragraph says what prints: the bias is small for this truth, which is a
  fact about a truth nobody has, can dominate for another (tutorial 14
  meets one), and so cannot be quoted.
- **Tutorial 13: the L-BFGS agreement and the tutorial-10 citation.** The
  prose already said L-BFGS stops unconverged (package A); the "linear
  operator of tutorial 10" is now "the path-average operator of example
  20"; the numbers re-measured (above).
- **Example 22's CG claim** (now 27). Plain CG converges in about 120
  iterations and Jacobi in about 30 (measured 117 and 29, agreeing to
  7e-11); the comment says so.
- **Example 12's metric demonstration** (now 11). The same `||x - a||^2 / 2`
  by steepest descent in the space and as a function of the component
  array, on `H^1` of a circle at length scales 0.01, 0.1, 1: 1, 1, 1
  iterations against 5, 34, 1561, the metric's condition number 1.0, 3.6,
  257 printed beside them.
- **Example 23 is six ideas** (now 30). Cut to the set answer on the sphere:
  a Gaussian prior hardened by `ambient_ball(level=0.95)` and by
  `ambient_ball(operator=L2 form, inverse=...)`, the feasible intervals of
  four caps with brackets beside what each bound alone allows, the eight
  directions swept by `support_sweep(n_jobs=4)`, two figures; certificate,
  hull and ellipsoids left to tutorial 8. 3.6 s.
- **Tutorial 7 spends 81 s drawing one set** (now 8). `plot_set(feasible,
  directions=48, n_jobs=4)`; the whole tutorial 15 s, so it leaves `SLOW`.
  The hull is now green (the plotting fix), and the figure's title is
  shortened to fit.
- **History and planning talk.** Tutorial 2's "Version 1" parenthesis and
  tutorial 9's "sampling layer will start" are gone.
- **Tutorial 12's contract and white noise.** The contract says white noise
  follows by default from a dense factorization of the Gram matrix and adds
  the row a backend at scale overrides, `white_noise_components` from a
  sparse factor; `PiecewiseLinear` now does so from its banded Cholesky
  factor (the draws are identical to the default's, the factor being
  unique). From package E: the refinement study probes with a smooth
  measure built from the backend's own pieces (covariance
  `(1 - l^2 d^2/dx^2)^-2`, its factor one sparse solve), 12 trials, on 41
  to 2561 nodes; on nine seeds the defect falls monotonically, where on
  white noise it does not; the prose says "falls under refinement, by
  orders of magnitude".
- **The low-rank mean.** Fixed in the library by package D; example 25
  prints the eigenvalues and picks the ranks by them, and example 31 says
  that a rank is chosen by the eigenvalues.
- **Examples build by hand what a constructor provides.** 28 uses
  `from_standard_deviations(D, quality)` (iteration counts 1045, 523, 599
  against 1051, 531, 604: the diagonal now carries its own factor), 29
  `from_standard_deviation(X, sqrt(spread), expectation=...)`.
- **The Colab notebooks.** `tutorials/make_notebooks.py` gains `BRANCH =
  "<release-branch>"`, used by the install cell; the README's badges open
  `blob/<release-branch>/...`; notebooks regenerated, without outputs. New
  test: `test_tutorials.py::test_the_badges_open_the_generated_notebooks`
  (each badge opens its own notebook on `BRANCH`; fails before, the badges
  pointing at `main`).
- **The walkthroughs.** Their tutorial citations follow the reorder; the
  four that cite a tutorial for size in the prior's norm now also cite
  example 23, and `point_data/IMPLEMENTATION.md`'s ambient-ball radius cites
  `ambient_ball` in tutorial 14 and example 30.

### Could fix

- **Imports from four depths.** Every exported name in the examples and
  tutorials is imported from `pygeoinf2`; subpackage paths only for the
  rest (`pygeoinf2.numerics.optimization`, the spaces, `testing`).
- **Two spellings of the circle.** The examples use
  `symmetric_space.circle`; 19 (`fields`) shows the box in any dimension
  and that `circle.Sobolev(32, ...)` holds the box's vectors.
- **Example 20's place.** Kept at 21 (`flexure`) in the "function spaces
  meet data" group, as the table has it; 27 cites it.
- **No disk example, none of parallelism.** `examples/33_disk.py`: an
  annulus with a radius-dependent length scale, surface stations, the
  low-rank posterior, the standard deviation shell by shell (exact, no
  solve), three whole-annulus panels including the variance removed;
  1.4 s. `n_jobs=` in example 30's sweep and tutorial 8's `plot_set`.
- **The examples README.** Rewritten for the sequence in groups, the "one
  idea each" claim scoped to the early ones, the optional-dependency table
  from `OPTIONAL`. New test:
  `test_examples.py::test_the_readme_lists_the_optional_dependencies`
  (fails before: the old table omitted pyshtools for the mantle).
- **CURRENT_STATE's counts.** 36 examples, 14 tutorials, two of each slow.
- **Small wording.** "draws of an order-1 measure" (19), "formed in the
  data space" (24), "every Bayesian linear method" (31), "falls under
  refinement" (tutorial 12).
- **The tutorials' answers come measure, set, point.** The course is now 6
  `regularization` (the forward problem, its chi-squared test moved here
  from the posterior tutorial, the deterministic answers with decision 5's
  framing, and least squares in a trace-class prior's norm as that prior's
  posterior mean, 5.5e-8), 7 `posterior`, 8 `properties` (its table point,
  measure, set); tutorial 14 is point, measure, set in its sections, its
  figure columns and its closing.

## Skipped

- **`Layered.sobolev_measure(correlation=)` in the layered planet.** The
  correlation ties the layers' modes by index, and the modes' signs are not
  aligned across layers: on `Layered([0, 0.19, 0.55, 1], 20, order=1.0,
  length_scale=[0.08, 0.15, 0.2])` with `sobolev_measure(1.5,
  pointwise_std=[0.6, 0.5, 0.25], correlation=` all pairs 0.8`)`, the
  correlations of the three layers' masses (`integral_functional` on each
  summand) are +0.78 (inner and outer core), -0.77 (outer core and mantle),
  -0.70 (inner core and mantle), and the jump at the core-mantle boundary
  varies more (0.66) than under independence (0.56). The docstring's "makes
  the layers' structure alike in scale and sign" is not what it does; an
  example would teach the opposite. A library defect, for the spectral
  element layer.
- **Cutting example 27's coupling diagnostic.** Restricting the three
  covariance blocks to low-degree coefficients (`lmax` 4, 8, 16) is cheap
  but tells a different story (coupling 0.73, 0.85, 2e-6 with both data
  channels), so it is not the same measurement made cheaper; what the
  diagnostic should measure is a decision. The example stays slow.
- **Backend last in the tutorial course.** The reviewer's renumbering also
  moved tutorial 12 to the end; decision 5 settles point, measure, set
  only, so 9 to 14 keep their numbers, and with them most of the
  walkthroughs' citations.
- **The walkthroughs' three spellings of the inverse factor.** Content of
  the apps, not a citation; example 23 needs no inverse (it evaluates the
  bias with `BackusGilbert(covariance=prior.covariance)` on the model).
- **Unshown exports.** Not every export needs an example; the inference
  ones now appear (examples 22, 23, 25, 30, 31). The rest is a question
  for `__init__`.
- **Outside this package's files, reported instead of edited:**
  `numerics/normal.py:585` cites "example 24" for the preconditioning
  example, now 28; `GaussianMeasure.ambient_ball`'s docstring still says
  `harden_error` calls it (deleted by package D); `check_operator`'s
  `measure=` draws only the domain probes, the adjoint identity's `y`
  being `codomain.random()` white noise even when the codomain is the
  domain, which is why tutorial 12 needs twelve trials for a clean
  refinement; `CURRENT_STATE.md` §7's tutorial-12 entry is done, §10's
  "an example of its own and one of the correlated product belong to the
  examples review" is done (25 and 15), "mini-apps ... are new and not
  started" is stale, and §6 could say that white noise follows by default
  from a dense factorization; the release mechanics need "fill
  `make_notebooks.BRANCH` and the badges (one test ties them), change the
  import name, regenerate, open one notebook on Colab";
  `EXAMPLES_PLAN.md`'s stale items.
