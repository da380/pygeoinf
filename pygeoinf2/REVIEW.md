# pygeoinf 2.0 refactor — review

Date: 2026-08-27. Reviewed: `pygeoinf2/` at commit `5d4e0e4` against `pygeoinf/` (v1), `pygeoinf2/DESIGN.md`, `pygeoinf2/V1_CATALOGUE.md`, and the downstream usage in `/home/david/dev/pyslfp` (package and `Heathcote2026/` scripts).

This file is the synthesis. The detail — method-by-method tables, measurements, `file:line` references and per-area Must/Should/Consider lists — is in `pygeoinf2/review/`:

| appendix | file |
|---|---|
| A | `review/algebra.md` — spaces, operators, nodes, direct sums, traits, testing |
| S | `review/solvers_preconditioners.md` — Krylov/direct solvers, all preconditioners, `DampedSolves` |
| N | `review/numerics.md` — functional calculus, randomised methods, optimisation, convex methods, root finding, weighted χ² |
| P | `review/probability.md` — measures, Gaussians, mixtures, the measure constructors on symmetric spaces |
| I | `review/inference.md` — forward problems, estimators, point/Gaussian/Backus inversions, structure-aware preconditioners |
| Y | `review/symmetric_space.md` — sphere, periodic box, bounded box, observation operators, field algebra |
| G | `review/geometry_plotting.md` — sets, convex sets, subspaces, plotting |
| K | `review/packaging_tests_docs.md` — public API, MFEM backend, tests, examples, Sphinx, packaging |
| U | `review/pyslfp_usage.md` — every pygeoinf API pyslfp touches, its v2 equivalent, and the port blockers |

Everything below marked **[verified]** was executed against the code, not inferred from reading. The fast v2 suite passes as of this review (`pytest pygeoinf2`: 1316 passed, 25 slow deselected, 82 s).

---

## 0. Verdict in brief

**The design is right and most of the core is right.** Riesz identification everywhere, traits with algebraic propagation, coordinate-free Krylov methods verified against a space that refuses coordinates, `at()` sharing one evaluation between value and derivative, the derivative/gradient distinction made structural, `NormalOperator` carrying its factors so preconditioners are free-standing, solver factories, the `Estimator` hierarchy with a one-line `push_forward`, `DirectSum` with labels, the NUFFT paths, the MFEM backend's treatment of the mass matrix as the Gram matrix. Several genuine v1 defects are fixed (white noise on weighted spaces, unhashable spaces, `gradient_dot_product` sign, `HalfSpace.project`, `credible_set` on weighted spaces, LSQR sign, GMRES restart). The metric bookkeeping in the places the design document worried about most (`from_derivative_matrix` adjoints, `matrix()` forms, `white_noise`, `as_multivariate_normal`, evidence log-determinant, Woodbury) is correct on a non-diagonal Gram matrix **[verified]**.

**But the port is materially less complete than DESIGN.md and V1_CATALOGUE.md say**, and the losses cluster exactly where you predicted: features dropped without a decision, and performance paths rewritten into slower ones. The catalogue's "Ported" status is unreliable — every area review found rows marked Ported whose functionality is absent, sphere-only, or semantically different. Concretely, the five findings that matter most:

1. **pyslfp cannot be ported to v2 as it stands** (appendix U). Its central idiom — `LinearOperator.from_formal_adjoint(domain, codomain, l2_op)` with a *direct-sum* codomain of Sobolev spaces and a `EuclideanSpace` domain — has no v2 equivalent: `lift_formal_adjoint` accepts only a single `SymmetricSpace` on each side. There is no `MassWeightedSpace` and no general formal-adjoint lift, although DESIGN §3.5 specifies both. Three further blockers: sphere vectors are bare arrays on a grid twice v1's size, not `SHGrid`; conditioned Gaussians (`GaussianMeasure.condition`) cannot be sampled, and pyslfp conditions every prior; the point convention changed from `(lat, lon)` degrees to `(colat, lon)` radians with no public converter.

2. **Structure is silently discarded by wrapping**, and one mechanism causes several visible failures. `with_traits` returns a `_RetraitedOperator` that forgets its class (A §3), so: a `DiagonalLinearOperator` loses its exact log-determinant and every log-evidence of a diagonal prior goes through 100 Hutchinson probes × Lanczos instead of `sum(log λ)` **[verified, N §2.3]**; a `NormalOperator` stops being a `FactoredNormalOperator` and structure-aware preconditioners refuse it **[verified, S §5.1]**; `DampedSolves` builds plain sums, so the same preconditioners are refused inside every discrepancy-principle sweep **[verified, I §2]**. In the measure layer, `_rebuild` drops the precision under `scale`, `add`, `translate`, `affine_map` and `from_product` **[verified, P §1.3]**, so the O(dim) spectral KL route silently becomes a dense O(n³) solve and the Woodbury data form loses `Q⁻¹`.

3. **Matrix-free defaults with dense fallbacks reached silently.** `from_component_matrix`/`from_derivative_matrix` return closures with no matrix accessor, so `matrix()`, `diagonals()` and every direct solver re-extract a stored matrix by `dim` applications **[verified, A §2]**; `_Composition.__init__` constructs adjoints eagerly, which for a direct-solver inverse means a second factorisation at `@` time **[verified, A §2.1]**; `MfemSpace.restrict` densifies the FE matrix (DESIGN §33.4 claims the opposite) **[K §2]**; `BlockPreconditioner`, `ColumnThresholdedPreconditioner`, `with_sparse_approximation`, `ambient_ball`, `nuclear_norm`, `condition`, `BackusGilbert._harden` all form full matrices where v1 was matrix-free or sparse.

4. **The symmetric-space port is sphere-centric and slower in places** (Y). `geodesic_distance`, `path_average_operator`, `covariance_function`, `with_degree`, `cluster_points`, `pairs_within_distance`, `random_points` exist only on `Sphere`; the box classes raise. The formal-adjoint lift costs four spherical-harmonic transforms per forward application where v1 cost none **[verified, my own count: 2 `SHExpandDH` + 2 `MakeGridDH` at lmax 128, 72 ms vs 0 ms]**. The sphere NUFFT forward path is 2–8× slower than v1's Fortran evaluation at 2000 points because finufft defaults to all cores — 20 ms single-threaded **[Y §2]**. `Sphere._quadrature` bootstraps DH weights with `2(lmax+1)` full transforms (10 s at lmax 256) although `pyshtools.utils.DHaj` returns them exactly. `Box.project_function` dropped v1's raised-cosine taper and documents the resulting Gibbs ringing as a "support assumption". The Sobolev-order guard on point evaluation is gone.

5. **Verified wrong answers**, all fixable in a few lines each: `GaussianMixture.log_density`/`marginal_probabilities` omit the per-component `log det` (responsibilities 0.47/0.53 where the truth is 0.90/0.10) [P §5.3]; `from_covariance_matrix(form="components")` is wrong on a non-diagonal Gram [P §5.1]; `MinimumNorm.for_data` returns damping 1e-31 with χ² ≈ 1e39 on data no model fits, where `DiscrepancyPrinciple` correctly raises [I §5.4]; `LinearForwardProblem.from_direct_sum` silently drops noise if any member has none [I §5.3]; `monotone_root` reports `converged=True` after exhausting its iterations [S §5.2]; `Polytope.project` is not a projection, so its indicator's prox is wrong [G §5.1]; the optimisers' default `gtol` is unreachable in double precision, so `LBFGS`/`NonlinearCG` report failure on a well-conditioned quadratic and `SteepestDescent` burns 5000 evaluations [N §5.4]; stochastic `kl_divergence` returns −88.6 ± 21.7 for KL(μ‖μ) [P §5.4]; `BackusGilbert._harden` uses a metric-wrong variance and crashes on the error-free path (`Ball(radius=0)`) [I §5.1, G §5.2]; `BallSurface.contains` breaks any set operation by a keyword rename [G §5.3]; the model-space formalism with `error=None` solves a different problem from the data-space one (inherited from v1) [I §5.2].

**Shape of the fix.** Not a rewrite. The layered architecture and the base classes stay. The work is (i) a short list of correctness fixes; (ii) three structural repairs that each close several gaps at once — a class-preserving `with_traits`, a `_rebuild` that carries precision, and a `MatrixLinearOperator` that stores what it was built from; (iii) a general `from_formal_adjoint`/`MassWeightedSpace`; (iv) restoring the box-space geometry, the coefficient accessors and the plotting options; (v) applying the non-diagonal-Gram test rule to the layers that never see one; (vi) a documentation pass that adds arguments, returns and exceptions to the 79% of parameterised public functions that lack them. §10 orders this into phases.

---

## 1. Method

- DESIGN.md (5162 lines) and V1_CATALOGUE.md were read in full first, then v1 and v2 side by side, area by area. Each area review read the v1 modules before the v2 ones, and checked the catalogue's status column against the code rather than trusting it.
- Claims about performance and correctness were tested with throwaway scripts (transform counts by monkey-patching pyshtools, operator-application counters, timings, dense-Gram spaces built from `compat.AdaptedSpace` over a v1 `MassWeightedHilbertSpace`, and direct numerical comparison with v1). Where a claim could not be tested it is stated as a reading.
- pyslfp was read in full (package and `Heathcote2026/`) to produce the inventory in appendix U; nothing in it was changed.
- No code in either package was modified. This review adds `pygeoinf2/REVIEW.md` and `pygeoinf2/review/*.md` only.

---

## 2. The target: what pyslfp actually uses

Appendix U tabulates ~110 distinct pygeoinf touch-points. The Heathcote2026 scripts are a good specification of "the type of functionality we need to retain" because they exercise the whole chain at scale: `Sobolev(256, …)` load spaces, a direct-sum response space `[Sob, Sob, Sob, R²]`, a matrix-free fingerprint operator built on L² and lifted, `point_evaluation_operator` on 10⁴–10⁵ altimetry points, `to_coefficient_operator` for GRACE, priors from `point_value_scaled_sobolev_kernel_gaussian_measure`, correlated priors from `CorrelatedInvariantGaussianMeasure.from_invariant_measures(…, corr_fn)`, priors masked by `affine_mapping` and **conditioned** on a mass-conservation constraint through `AffineSubspace.from_linear_equation(...).condition_gaussian_measure`, `LinearBayesianInversion` with a surrogate Woodbury preconditioner blended with `α R⁻¹`, `CGSolver(callback=ProgressCallback())`, property push-forward with `with_dense_covariance(parallel=…)`, `kl_divergence`, `kalman_operator` for resolution kernels, `samples(n, parallel=…)` for pointwise std, `plot_corner_distributions(…, title=…)`, and `sphere.plot(SHGrid, colorbar_kwargs=…, gridlines_kwargs=…)`.

What v2 handles well for this workflow: the inference core (`LinearForwardProblem`, `from_direct_sum`, `LinearGaussianInversion`, `gain`, `surrogate` + `WoodburyPreconditioner.from_normal`, `push_forward` carrying the RTO sampler, `kl_divergence`, directional statistics), the matrix-free observation operators, and the measure constructors' spectral conventions (so pyslfp's noise-amplitude solver stays valid).

What blocks the port, in order (U §c):

| # | blocker | where |
|---|---|---|
| 1 | no formal-adjoint lift for direct-sum / Euclidean / general spaces | A Must-5, Y Should-13 |
| 2 | sphere vectors are `(n, 2n)` arrays, not `SHGrid`; grid is 2× v1's default | Y §1.1, §11 Q1 |
| 3 | `condition` (and no `condition_gaussian_measure`) gives an unsamplable measure | P Should-8, G Should-4 |
| 4 | `(colat, lon)` radians vs `(lat, lon)` degrees, no public converter | Y Must-10, §11 Q2 |
| 5 | no parallelism at `samples`, `with_dense_covariance`, dense assembly | §11 Q6 |
| 6 | solve diagnostics (iterations) not surfaced from `est(data)`; no `ProgressCallback` | S Must-4, I Should-13 |
| 7 | `from_product` and `+` drop precision → Woodbury data form and `chi_squared` degrade | P Must-3 |
| 8 | `check_operator` cannot take a smooth test measure (the SLE iteration needs one) | A Consider-19 |
| 9 | plotting keyword losses (`title`, `colorbar_kwargs`, `map_extent`, `contour`, `data=` on points) | G Must-7, Should-7 |
| 10 | naming traps: `scale`/`length_scale`, `multiply` now pointwise, `zero()`, `heat_measure(time)` = v1 `scale²`, factory functions instead of classes | §7 |

---

## 3. Cross-cutting findings

### 3.1 "Ported" in the catalogue is not evidence

Every area review found catalogue rows marked Ported or Subsumed that are not (A §1 items 1–8; S §1.1–1.3; N §1.4–1.5; P §1.1, §1.5–1.6; I §1.2–1.4; Y §1.1; G §1.1–1.4). Some are outright absent (`LevelSet`, `from_coefficient_operator`, `sample_power_measure`, `from_standard_deviations`, `PrimalKKTSolver`, saddlepoint χ²), some exist only on the sphere, some changed meaning (`from_vectors` is the adjoint of what it was; `path_average_operator` returns an average not an integral; `heat_measure(time)` vs `scale`; `estimate_truncation_degree` inverted), and at least thirteen rows carry Ported in the status column and "Not ported" in the text. The catalogue was the instrument meant to prevent silent loss; it needs a mechanical check. Recommendation: a parity test that constructs every concrete v2 space and calls the union of v1's public method names on each (Y Consider-30), and a one-pass reconciliation of the catalogue against `grep`.

### 3.2 Structure lost by wrapping

Three mechanisms, each responsible for several visible problems:

- **`with_traits` → `_RetraitedOperator`** (`algebra/operators.py:386–391, 920–931`). Loses `DiagonalLinearOperator` (fast paths, `eigenvalues`, `_combine_*`), `FactoredNormalOperator` (preconditioners), `InverseOperator`. Both in-package log-determinant call sites wrap a diagonal covariance in `.with_traits(definite)` first, so the exact path is never taken. Fix once: `with_traits` returns a shallow copy of the same class with `_traits` replaced (A Must-4, N Must-2, S Must-2).
- **`GaussianMeasure._rebuild`** has no `precision`/`precision_factor` parameters (`probability/gaussian.py:913–931`), so no operation can keep them. Fix once: carry them, with the obvious rules (P Must-3).
- **`DampedSolves.operator()`** builds `base + t*shift` (`numerics/root_find.py:296–299`) rather than asking the family for `at(t)`. Fix once: take an `assemble` callable (S Must-2, I Must-2).

The general lesson for the implementing model: any operation that returns a *new* object from a *structured* one must return the same class or a class that preserves its capabilities; otherwise the specialisation protocol of DESIGN §5.4 is defeated at the next step.

### 3.3 Eager work in constructors and at composition

`_Composition.__init__` calls `.adjoint` on the first half of its factors to test the palindrome rule (`algebra/nodes.py:240, 251`); for an `InverseOperator` from a direct solver, `.adjoint` means a second matrix extraction and factorisation **[verified]** — an `@` costing O(n³). `_block_traits` does the same for every block. `Sphere._quadrature`, `_packing`, `_legendre_indices` are per-instance caches rebuilt by every `with_order(0.0)`, which `multiplication_operator`, `flexural_operator`, `l2_products_operator` and `derivative_operator` all call (0.2 s at lmax 256 before any work; 10 s if the quadrature is needed). The palindrome check must read existing adjoint links only (A Must-2); the sphere caches should be keyed on `lmax` at module level (Y Should-12).

### 3.4 Dense fallbacks reached silently

The design says matrix-free is the default. A user cannot tell from a call whether it will assemble. Places that do, without saying so in the docstring (all with the v1 behaviour noted in the appendices): `matrix()`/`diagonals()`/`assembled()` on a matrix-built operator (A §2.2); `DirectSolver` on any operator (A §2.2); `MfemSpace.restrict` (K §2); `BlockPreconditioner`, `ColumnThresholdedPreconditioner` (S §1.2); `credible_set`, `_weighted_squared`, `condition`, `with_sparse_approximation`, `ambient_ball`, `nuclear_norm`, `hilbert_schmidt_norm` (P §2.3); the spectral KL quadratic term (P §2.4); `log_determinant(method="auto")` below dim 512 (I §1.2); `BackusGilbert._harden`, `BackusInference.inclusion_norm` (I §2); `moments()` for plotting (G §2); `pairs_within_distance` (Y §2). Recommendation: every such method states its cost in the docstring's first paragraph, and the ones v1 did matrix-free get their matrix-free route back (listed per appendix).

### 3.5 The metric-bug rule is applied in 6 of 30 test files

DESIGN §30.2 records the same class of bug five times and states the rule: only a non-diagonal Gram matrix can tell a metric-correct expression from a metric-naive one. `conftest.py` provides `make_dense_metric_space()`. It is used in `test_spaces`, `test_operators`, `test_functional_calculus`, `test_probability`, `test_plotting`, `test_phase_four` — and never in the solver, preconditioner, inference, KL, randomised, direct-sum, geometry, optimisation or convex tests (K §3). Every `SymmetricSpace` is a `DiagonalMetricSpace`, so sphere tests do not count. Two of the metric bugs found here (`from_covariance_matrix(form="components")`, `InvariantDistancePreconditioner`) are in untested-with-dense-Gram code. K Must-4 lists the fixtures to parametrise.

### 3.6 Options and diagnostics dropped

Beyond `parallel=`/`n_jobs=` (an agreed decision, with your caveat), the port dropped things that were not decided: solver callbacks in MINRES/BiCGStab/LSQR and the iterate-carrying callback (S §1.1); `x0` in LSQR; MINRES preconditioning; the stochastic Jacobi option; ILU in the banded preconditioner; adaptive stopping in `random_trace`/`random_diagonal`/`deflated_diagonal`; fixed-size Lanczos, `atol`, `check_interval`, Kahan reorthogonalisation and the coefficient-space convergence test; the convergence test in `operator_quadratic_form`; `n_clusters` in `cluster_points`; `include_names` on stations; `rank=`/`open_set=` on `credible_set`; `title`/legend on the distribution plots; contour modes and `map_extent` on field plots; `full=` on box plots; `plot_error_bounds`; `from_sobolev_parameters`' automatic `lmax`. Each is listed with a line reference in its appendix.

### 3.7 Defaults changed without record

`IterativeSolver` `rtol` 1e-5 → 1e-10 with `strict=True` (150 vs 305 iterations on the same system **[S §1.1]**, and the cause of DESIGN §27.1's "broke an example"); `Optimiser` `gtol=1e-8` unreachable; `apply_operator_function(rtol=1e-10, max_iterations=50)` effectively "always 50 applications" per `f(A)x`, so every Gaussian sample through `operator_sqrt` costs 50 covariance applications; `random_range(power=1)` vs v1's 2; `earthquakes(minimum_magnitude=0.0)` vs 5.0; `path_average_operator(count=20, normalise=True)` vs a length-scale heuristic and an integral; `Sphere(sampling=2)` vs 1; `plot(cmap="viridis", colorbar=True, gridlines=False)` vs `RdBu`, off, on; `heat_measure(time)` vs `scale`; `SteepestDescent` inherits Armijo although DESIGN §11.7 says strong Wolfe. Each default should either be restored or written down with the measured reason (§11 Q8, Q9, Q12).

### 3.8 Documentation: why is excellent, what is thin

Module and class docstrings explain motivation better than most libraries. But over 857 public functions, 54% are one-liners; of 547 with parameters, 21% have an Args section, 2% Returns, **0 Raises** (K §5). `algebra/spaces.py`, `algebra/direct_sum.py`, `probability/base.py`, `traits.py` are at 0% Args. Units and conventions (radians/degrees, angular/physical radius, `time` vs `scale`, `(colat, lon)`) are the commonest omissions in the symmetric-space layer. Several docstrings are false: `assembled()` ("the metric still enters exactly once, inside the adjoint"), `multiplication_operator` ("no self-adjointness is claimed" — no adjoint is supplied at all), `IterativeSolver.__init__` (`maxiter` default; callback claim), `onto_kernel` ("nothing claimed" but claims PD), `Polytope.project` ("the nearest point"), `LocalisedPreconditioner` (overlap "adds sensibly"), `GaussianMixture.log_density` (shared constant), `FeasibleProperty` (`solver` use), `MinimumNorm.for_data`. `test_code_practice.py` checks presence only; K Should-12 proposes checking content.

---

## 4. Feature register (retained / extended / lost) by area

Summary only; the appendices have the full tables.

| area | retained & extended | lost or not equivalent (headline items) |
|---|---|---|
| **Algebra** (A) | traits, memoised involutive adjoint, palindrome rule, `at()`, second derivatives, correct `white_noise`, `Reals`, `DirectSum` labels, sparse input, `from_derivative_callables`, `diagonals(probe="banded")`, hashable spaces | functional algebra not closed (`phi+psi`, `phi@F` lose `gradient`/`hessian` → optimisers fail on composed functionals); `LinearFunctional` not closed and unbuildable from a mapping; `AffineOperator @ AffineOperator` degrades; no matrix-backed operator class (O(1) matrix/diagonal access gone); no `MassWeightedSpace`/general `from_formal_adjoint`; `from_vectors` transposed; weighted `from_tensor_product`; SciPy `LinearOperator` view; block `matrix()` fast paths; `EuclideanSpace.subspace_projection`; test-vector measures in `check_*` |
| **Solvers** (S) | coordinate-free CG/MINRES/BiCGStab/LSQR verified on a strict space, GMRES and FlexibleCG added, `SolveResult`, declared preconditions, `with_preconditioner`, factories, Woodbury unified, structure-aware preconditioners free-standing | `FlexibleCGSolver`/`GMRESSolver` not exported; MINRES preconditioning; LSQR `x0`; callbacks/history in 3 of 6 solvers; stochastic Jacobi; ILU; `ExactBlock` now dense and non-overlapping; `ColumnThresholded` dense; O(1) `diagonals()` overrides; LU adjoint reuse; `rtol` default |
| **Numerics** (N) | native optimisers (SD, NCG, LBFGS, NewtonCG, TR-Newton), Steihaug CG, Armijo/Wolfe, `Estimate` with error bars, named `operator_*` helpers, `monotone_root`/`DampedSolves`, FISTA, Polyak subgradient, proximal bundle with linearisation errors, `from_support_function` | Lanczos: fixed mode, `atol`, coefficient-space test, Kahan reorth, quadratic-form stopping; `random_range` QR path and incremental orthogonalisation; adaptive stopping in estimators; `LevelBundleMethod`, `QPSolver` backends, `PrimalKKTSolver`, Chambolle–Pock, smoothed dual master, `solve_support_values` warm start, fused value+subgradient; saddlepoint χ²; vectorised `t`; MC quantile |
| **Probability** (P) | explicit `rng`, factor-based sampling, push-forward as nodes, `from_samples`, `low_rank_approximation`, `condition`, `log_density`/`grad_log_density`, mixtures, products of arbitrary measures, KL with three routes, `pointwise_std=`/`norm_std=` calibration, `correlated_measure` by extended KL | precision dropped by every operation; `from_standard_deviations`; `from_covariance_matrix` PSD-singular & precision; `from_product` precision; `credible_set` `rank`/`open_set`/`cameron_martin`; `weakened_ellipsoid`; stochastic norms; matrix-free `with_sparse_approximation` with LU precision; `sample_pointwise_variance`; `-mu`; `directional_statistics`; correlated-measure accessors and exact KL/norms; `covariance_function` off the sphere; `sample_power_measure`; `heat_measure` parameter |
| **Inference** (I) | `ForwardProblem` with measure *or* set error, `consistency_set`, estimator kinds, `NormalOperator` with factors and `surrogate`, matrix-free evidence with Sylvester, `TikhonovFamily`, `DiscrepancyPrinciple` with exact derivative, constrained pair, four Backus routes with parity tests, mixture inversion | `for_data` divergent duplicate; `from_direct_sum` noise drop; `parameterised`/`data_reduced` on constrained classes; `dense=` options; `normal_residual_callback` on the Gaussian inversion; `with_solver`/`with_formalism` on three classes; `test_data_compatibility`/`is_empty`; fused dual oracle; `inclusion_norm` caching; top-level exports; nonlinear MAP/Laplace (absent in both versions) |
| **Symmetric spaces** (Y) | one N-d periodic box (3-D free), Sobolev as a metric, vectorised packing, NUFFT on box and sphere (adjoint exact), cap integrals exact, `degree_transfer_operator` adjoint derived, `pointwise_variance` exact, flexure via Bochner, real station/event tables | box geometry (`geodesic_*`, `walk_from`, `random_points`, `path_average_operator`, `covariance_function`, `with_degree`, `degree_transfer_operator`, `cluster_points`, `pairs_within_distance` all sphere-only); `to/from_coefficients`; `from_coefficient_operator`; Sobolev-order guard; `Box.project_function` taper; `n_clusters`; `from_sobolev_parameters`; `SHGrid` vectors; `sampling=1`; `(lat, lon)` degrees; `include_names`; `random_domain_points`; `plot_geodesic` |
| **Geometry / plotting** (G) | `ConvexSet` as predicate+projection+indicator+support, Minkowski sums, oracle sets, `Polytope` with inner/outer status, correct `HalfSpace.project`, `dimension` by component trace, `subplots` replacing `create_map_figure`, `plot_paths` with dateline splitting, `plot_densities`/`plot_corner` accepting sampled measures | `Intersection` of convex sets not convex; `LevelSet`/`SublevelSet`; `HalfSpace.support_function`; `distance_to`; `boundary` links; `condition_gaussian_measure` on subspaces; equation dropped by `from_kernel`/`with_translation`; fused `value_and_support_point`; field-plot options (contour, extent, rivers/borders, colorbar/gridline kwargs, projection); `plot_points` `data=`; box `plot_points`/`plot_paths`; `plot_error_bounds`; `full=`; `title`, legend, σ-axis, colourbar in distribution plots; `SubspaceSlicePlotter`/`plot_slice` |
| **Packaging / tests / docs** (K) | acyclic 8-package layout, `test_code_practice`, doubles (`Opaque`, `StrictSpace`), examples run by tests, slow markers, 0.32 s import | `inference`/`plotting` not reachable from the top level; ~15 public names unexported; `pygeoinf2` not in the wheel; dense-Gram rule in 6/30 test files; Sphinx documents v1 only; v1 tutorials have no v2 counterpart; no least-squares or nonlinear example; `v2/` empty skeleton; catalogue stale |

---

## 5. Performance register

Items where v2 does measurably more work than v1, or than it should, ordered by likely impact on the large-problem use case. Measurements are in the appendices.

| # | what | measured | appendix |
|---|---|---|---|
| 1 | `lift_formal_adjoint` round-trips through components: 4 SH transforms per forward apply (v1: 0); `multiplication_operator` 6 vs 0, `flexural_operator` 58 vs 26–30 | 72 ms/apply at lmax 128 (this review); counts at lmax 32 (Y §2) | Y Should-13 |
| 2 | finufft `nthreads` default → sphere forward evaluation 173–794 ms at 2000 points vs 20 ms single-threaded; v1 93 ms | Y §2 | Y Must-3 |
| 3 | `Sphere._quadrature` bootstrap: 10 s at lmax 256, minutes at 512; `DHaj` gives it exactly | Y §2 | Y Must-4 |
| 4 | sphere caches rebuilt per `with_order` (0.2 s at lmax 256 per operator construction) | Y §2 | Y Should-12 |
| 5 | grid `sampling=2`: every field and pointwise op 2× v1's default | Y §2 | §11 Q1 |
| 6 | `_Composition` eagerly builds adjoints → second factorisation at `@` for direct inverses | A §2.1 | A Must-2 |
| 7 | matrix-built operators re-extracted by `dim` applications for `matrix()`, `diagonals()`, every direct solve; `assembled()` applies a `solve_gram` per forward apply | n=1500: 0.1–0.2 s each, O(n³) growth (A §2) | A Must-3 |
| 8 | `MfemSpace.restrict` densifies (80 GB at 1e5 dofs); `_to_mfem` Python loop per nonzero | K §2 | K Must-1 |
| 9 | diagonal log-det goes stochastic (100 probes × 40 Lanczos) because of `with_traits`; `log_determinant` never consults `DiagonalLinearOperator.log_determinant` | ±1.3 on 129.5 (N §2.3) | N Must-2 |
| 10 | `apply_operator_function` rebuilds the full-space result every Lanczos step (O(k²) axpys) and `operator_quadratic_form` never stops early | N §2.1 | N Must-3 |
| 11 | `random_range` adaptive mode re-orthonormalises the whole basis every block: 1.17M inner products for a 380-vector basis; no QR path on coordinate spaces | N §2.2 | N Must-4 |
| 12 | line-search methods re-evaluate value+gradient after the Wolfe search; `_zoom` is bisection; default `gtol` unreachable → LBFGS 75 gradient calls for 19 iterations, SD 5330 value calls | N §2.5 | N Must-1, Must-5 |
| 13 | `diagonals()` has no O(1) override, so a diagonal `R` costs `dim` matvecs inside every structure-aware preconditioner and Jacobi | S §2.5 | S Should-9 |
| 14 | `BlockPreconditioner`/`ColumnThresholded` form the full matrix (v1: column probing, sparse) | S §1.2 | S Must-7 |
| 15 | LU inverse adjoint re-extracts and re-factorises; two adjoint caches on `InverseOperator` | S §2.4 | S Should-12 |
| 16 | precision dropped → spectral KL quadratic term dense O(n³); Woodbury data form inverts `Q` by CG | P §2.4 | P Must-3, Must-4 |
| 17 | stochastic KL: ~100 solves + 8000 covariance applications, 45 s, wrong sign | P §5.4 | P Must-7 |
| 18 | `pairs_within_distance` O(n²) memory (576 MB at n=3000; v1 7 MB via KD-tree) | Y §2 | Y Must-7 |
| 19 | `power_measure` O(dim²) count comprehension (7 s at lmax 511) | Y §2 | Y Must-8 |
| 20 | `BackusInference.inclusion_norm` dense eigendecomposition per call, uncached | I §2 | I Should-10 |
| 21 | mixture inversion recomputes data-independent log-dets per call | I §2 | I Should-9 |
| 22 | `from_linear_equation` builds `(A A*)⁻¹` three times | G §2 | G Should-5 |
| 23 | `WoodburyPreconditioner` applies `Q` three times per outer iteration; inner CG at `rtol=1e-10, strict=True` | S §2.6 | S Should-14, Consider-20 |
| 24 | `HilbertSpace.add/subtract/scale` default to copy+axpy (two allocations) | A §2.5 | A Should-8 |
| 25 | weighted χ² Imhof via scalar `quad`: 25× slower than v1's vectorised trapezoid at small n | N §1.5 | N Should-16 |

Where v2 is faster: memoised adjoint (v1 allocated per access), functionals matrix-free (v1 `LinearForm(mapping=)` did `dim` evaluations in `__init__`), NUFFT adjoints on both box and sphere (30 ms vs 1.1 s at 20 000 points), `matrix(by=)` from the cheaper side, warm-started discrepancy sweeps with preconditioner reuse, `basis_at` batching, vectorised Fourier packing, Sylvester evidence, the bundle subproblem dualised.

---

## 6. Correctness findings (all verified by execution)

| # | defect | fix | appendix |
|---|---|---|---|
| 1 | `GaussianMixture.log_density`/`marginal_probabilities` omit `−½ log det C_k` | add `log_normalising_constant()` to `GaussianMeasure` and use it | P Must-2 |
| 2 | `from_covariance_matrix(form="components")` gives `sym(C_c G)` not `G C_c` on non-diagonal Gram | build column-wise with `apply_gram` | P Must-1 |
| 3 | `MinimumNorm.for_data` returns a singular-system solution where `DiscrepancyPrinciple` raises; leaks `LinAlgError` with a direct solver | delegate to `DiscrepancyPrinciple._resolve` | I Must-1 |
| 4 | `LinearForwardProblem.from_direct_sum` sets `error=None` if any member lacks one (noise silently dropped) | raise, or build zero-covariance members | I Must-3 |
| 5 | `monotone_root` returns `converged=True` after exhausting iterations | return `False` unless the bracket tolerance is met | S Must-3 |
| 6 | `Polytope.project` is cyclic projection → a feasible point, not the nearest; its prox is wrong | Dykstra, or raise and rename | G Must-1 |
| 7 | optimiser termination: `gtol` unreachable; no value/step criterion; Wolfe failure reported as non-convergence at `|g| = 1e-6` | `ftol`, precision-loss outcome, reachable defaults | N Must-1 |
| 8 | stochastic `kl_divergence` returns −88.6 ± 21.7 for KL(μ‖μ) (dim 578) | precondition, more iterations, refuse when SE > value, exact block-spectral route | P Must-7 |
| 9 | `BackusGilbert._harden`: metric-wrong mean variance, dense matrix for one scalar, `Ball(radius=0)` raises on the error-free path | use `ambient_ball(level=)`; allow radius 0 | I Must-5, G Must-2 |
| 10 | `BallSurface`/`EllipsoidSurface.contains(*, tolerance)` vs abstract `rtol` → `TypeError` in any `Intersection` | rename | G Must-3 |
| 11 | model-space formalism with `error=None` solves `Q⁻¹ + A*A` (R = I) while data-space solves with R = 0; posteriors differ (inherited from v1) | forbid or document | I Must-6 |
| 12 | `TikhonovFamily.weighted_adjoint` references an attribute that is never set → `AttributeError` | delete or delegate | I Must-4 |
| 13 | `InvariantDistancePreconditioner` correct only on an orthonormal data space; mixes a Galerkin noise diagonal into a component-form kernel | assemble `G K G + diag(R_gal)` or require orthonormal | S Should-10 |
| 14 | `LocalisedPreconditioner` double-counts overlapping blocks (COO duplicates sum); docstring says this is desirable | divide by multiplicity or forbid overlap | S Should-11 |
| 15 | `plot_densities` fixed 2000-point grid aliases a posterior 1000× narrower than its prior (the case the twin axis exists for) | grid from the narrowest σ | G Must-4 |
| 16 | `plot_corner(fill=True)` fills Mahalanobis distance in the Gaussian branch and density in the sampled branch | fill density in both | G Must-5 |
| 17 | `onto_kernel` claims `POSITIVE_DEFINITE` for `A A*` (false when `A` is rank-deficient; docstring says the opposite) | claim PSD | G Should-5 |
| 18 | `DiagonalLinearOperator.sqrt/log/pow` gate on traits never deduced on a non-diagonal metric, so refuse valid operators there; `operator_function`'s diagonal branch skips the self-adjoint check | gate on eigenvalues | A Must-4, N Should-17 |
| 19 | `GaussianMeasure(...)` accepts mutually inconsistent covariance/factor/precision; `check_measure` never tests precision·covariance | check in `testing` | P Should-15 |
| 20 | `lift_formal_adjoint`-built operators and `dirac` accept `order ≤ dim/2` and return grid-scale noise (v1 refused) | restore the guard | Y Must-2 |
| 21 | `Box.project_function` zero-fills the padding without taper → Gibbs ringing for non-vanishing boundary values | restore raised-cosine taper | Y Must-9 |
| 22 | `pointwise_variance` on boxes evaluates at the origin, where the Nyquist cosines make the sum a maximum, not the constant claimed | average over the grid | Y Consider-28 |
| 23 | `Sphere.geodesic_quadrature` uses `arccos` right after the docstring explaining why `atan2` is needed | `atan2` | Y Should-19 |
| 24 | `with_translation` drops the constraint equation; `from_kernel` never records `(A, 0)` | record | G Should-2 |
| 25 | `Linearisation`/`QuadraticModel` are `frozen=True, eq=True` dataclasses with array fields: `==` raises, `hash` raises | `eq=False` | A Should-11 |
| 26 | identity-based palindrome recognition is not thread-safe | lock or document | A Consider-18 |

Confirmed correct on a non-diagonal Gram (no action): the `from_*_matrix` adjoints and `matrix()` forms, `diagonals()`, `coordinate_inclusion/projection`, `white_noise`, `DirectSum` gram handling, all Krylov solvers and PCG pairing, Cholesky/Eigen on the Galerkin form, Woodbury identities, Jacobi/Banded/NormalDiagonal/Localised diagonals, `as_multivariate_normal`, `credible_set`, `from_samples`, dense KL, evidence log-determinant, `_self_adjoint_spectrum`, projections onto ball/half-space/hyperplane/affine subspace, `dimension`, `plot_corner` marginals, `pointwise_variance` on the sphere, cap integrals with radius ≠ 1, `degree_transfer_operator` adjoint, the DFS NUFFT adjoint, and v2's `gradient_dot_product` sign (v1's is wrong).

---

## 7. API and naming — comments and recommendations

These are judgement calls; §11 asks for your decision where it is yours to make.

- **`from_derivative_matrix`.** You said the name is not the clearest. The same object is called "galerkin" on extraction and "derivative" on construction, and `assembled()`'s docstring uses both in one sentence. Recommendation (A §3): one constructor `from_matrix(domain, codomain, M, *, form: Literal["components", "galerkin"], traits)` with `form` **required**. DESIGN §5.3's objection was to `form="auto"` on construction; a required choice keeps the convention explicit, round-trips exactly with `matrix(form=)`, and leaves one thing to document. Keep `LinearFunctional.from_derivative_components`, where "derivative" is exact. **Decided (D-4): adopt this; the docstring must make the convention unmistakable.**
- **`multiply` / `scale`.** v1 `space.multiply(a, x)` was scalar multiplication; v2 `multiply(x, y)` is the pointwise product and `scale(a, x)` the scalar one. On a `HilbertModule`, `space.multiply(2.0, x)` does not error — it runs a pointwise product through the transforms (A §1.15). Either rename the pointwise product (`pointwise`, `hadamard`) or make `multiply` reject a scalar first argument, and add the guard to `test_code_practice`.
- **`scale` / `length_scale`.** `Sobolev(lmax, order, scale)` became `Sobolev(lmax, order, length_scale)` because `scale` is the vector method. Fine, but it breaks every v1 call; say so in the migration notes.
- **Point convention.** `(colatitude, longitude)` radians is consistent and pinned by test, but every existing script and pyslfp use `(lat, lon)` degrees, `spherical_cap_integral` takes an angular radius while `geodesic_ball_average_operator` takes a physical one, and there is no public converter (Y §3). **Decided (D-2): degrees, `(lat, lon)`**, converted privately at the boundary; every radius argument says which it is.
- **`Lebesgue`/`Sobolev` as factory functions.** Preserves the call shape but `isinstance(X, Sobolev)` fails and `type(X).__name__` says nothing about the metric; pyslfp's `sl/utils.py` does exactly that check (U §A). **Decided (D-3): per-geometry submodules exporting thin subclasses.**
- **Sphere vectors.** v2 uses bare `(n, 2n)` arrays at `sampling=2`. **Decided (D-1): `SHGrid` vectors, `sampling=1` by default and user-selectable** — vectors being general objects is a point of the library, and it halves field memory against the current default.
- **Top-level namespace.** `import pygeoinf2 as gi` reaches neither `inference` nor `plotting`; `numerics` only by accident (K §1). v1 was flat and pyslfp uses `inf.LinearForwardProblem`, `inf.CGSolver`, `inf.plot_corner_distributions`. Recommendation: import the subpackages in `__init__` and re-export the dozen workflow names. Preconditioners are split between `numerics` and `inference` by which factors they read — an implementation criterion the user does not care about; put them in one place.
- **Parameter names.** Iteration cap is `maxiter` / `max_iterations` / `iterations` and tolerance `rtol` / `tolerance` depending on module (K §1). Pick one pair.
- **Too many ways to hand over a preconditioner** (S §3): constructor `preconditioner=`, `with_preconditioner`, `resolved_for`, `resolve_solver`, `with_solver`, `WoodburyPreconditioner(...)` vs `.from_normal(...)`. Each has a reason, but `with_preconditioner` silently returns `self` unchanged when one is already set — a `with_X` that ignores its argument. Raise instead.
- **Estimator names** (I §3): `for_data` is opaque; `DiscrepancyPrinciple` names a rule rather than an estimator; `BackusInference` now means only route (a) and DESIGN §18.8's `method=` selector was not built; `FeasibleProperty` vs `DualFeasibleProperty` say how, not what. `MinimumNorm` adds nothing to `LeastSquares` but a divergent duplicate of the discrepancy search.
- **`heat_measure(time)`** is v1's `heat_kernel_gaussian_measure(scale)` with `time = scale²`, undocumented (P §1.6). **Decided (D-9): parameterise by the length scale.**
- **`from_vectors`** now builds the adjoint of what v1 built under the same name (A §1.6). Rename (`from_span`) or document loudly.
- **`push_forward` / `affine_map` / `translate` / `__rmatmul__`** — four spellings of one idea; acceptable, but only `push_forward` accepts an `AffineOperator`.
- **`path_average_operator`** returns an average where v1 returned an integral and the docstring said so; `count=20` replaces a length-scale heuristic (Y §1.1). **Decided (D-12): v1's action under an honest name (`path_integral_operator`), with the heuristic and an optional weight.**
- **Vector aliasing.** `_Identity._value` returns its input and `DirectSum.projection` returns the component object; in-place `axpy` on an operator's output can mutate its input (A §5.4). Same as v1; document the rule or copy.
- **Factory argument order** varies (`from_component_matrix(domain, codomain, M)`, `from_vectors(codomain, vectors)`, `from_tensor_product(u, v, *, domain, codomain)`, `DiagonalLinearOperator(domain, values)`). Spaces first everywhere.

Good practice worth keeping as it is: keyword-only optionals (enforced), `rng` everywhere, immutable `with_*` returning new objects, `NotImplemented` dispatch, typed `ConvergenceError`, actionable error messages naming the remedy, lazy optional imports, the doubles (`Opaque`, `StrictSpace`) and the parity tests between independent routes in the inference layer.

---

## 8. Code quality — recurring patterns

- **Reach-through into private members** across modules: `__dict__["_adjoint_cache"]` written from seven places in three modules; `FeasibleProperty` calling `BackusGilbert._harden`; `InvariantDistancePreconditioner` using `normal.prior` (not on the ABC it checks for); `random_range` calling `codomain._orthogonalise_against`; `from_linear_equation` poking `subspace._equation` after construction; tests reaching `_quadrature`, `_double`, `_packing`. Provide the small public hooks (`_link_adjoint`, `FactoredNormalOperator.prior`, `orthogonalise_against`) and stop.
- **Duplication**: CG/FlexibleCG share ~40 lines; the seven preconditioners each reimplement `apply_gram → solve → from_components`; `with_solver`/`with_formalism` written three times and missing on three classes; `choose_formalism` defined twice; `path_average_operator`/`geodesic_ball_average_operator` are the same 45 lines; the four linear block classes re-validate their nonlinear parents' inputs; result-construction tails repeated across six optimisers; `Ball.support_maximiser` ≡ `_BallSupport._maximiser`.
- **Dead code**: `_ConvexResult`, `random_eig`'s empty loop, `Box.support_projection`'s unreachable branch, `pointwise_variance_at`'s unused operator on the exact path, `onto_kernel`'s `_ = codomain`, `TikhonovFamily.weighted_adjoint`, `**kwargs` on `parameterised`/`data_reduced` that go nowhere.
- **Type hints** are enforced for presence; the generics are used inconsistently (`DirectSum[V]` unused, block classes unparameterised, `with_traits -> Self` wrong, `solver: Any`, vectors `Any` throughout geometry and numerics), and closures are unchecked.
- **Imports**: function-local imports of `Traits`/`LinearOperator` in `probability/gaussian.py` where the module already imports them; `testing.py` eagerly imports geometry and probability.
- **RNG**: DESIGN §7 promises a seedable module default; none exists, and `mixture._resolve_rng` creates a fresh generator per call while `spaces._resolve_rng` shares one.

---

## 9. The design documents

DESIGN.md is a valuable engineering journal — the findings sections (§20.7, §21.7, §22.3, §23.6, §27, §30.2) are the best record of *why* the code is as it is. But as a description of the code it is stale in ways that will mislead the next model:

- Status line: "design agreed, not yet implemented".
- §3.1–3.2 (`identity()`/`zero_operator()` on the space, `gram`/`gram_solver` properties), §3.5 (`MassWeightedSpace`, `from_formal_adjoint`), §5.3 (`matrix(dense=False)`), §5.4 (`_Inverse` node), §5.5 (Hessian propagation table), §7 (seedable RNG), §11.7 (steepest descent uses strong Wolfe), §13.4 (Condon–Shortley included — code excludes it), §14 ("two foreign backends"), §18.7 (sampler built in `__call__`), §18.8 (`method=` on `BackusInference`), §18.11 (`is_empty`), §21.12 (`DHaj` not exposed — it is), §24.3/§25.1 (structure-aware preconditioners on all point estimators), §33.4 (sparsity survives `restrict`) — none match the code.

Recommendation: split it. Keep DESIGN.md as the journal (rename `DESIGN_LOG.md` if you like), and write a short **current-state** document (the `§2.2` package map, the space/operator/solver/measure contracts as they are, conventions and units, the dense-fallback list, the backend recipe). Every module docstring currently cites "DESIGN.md section N", which will not exist for a released package. V1_CATALOGUE.md should be reconciled against `grep` once and then either kept mechanically checked or retired into the migration notes.

---

## 10. Work plan for the implementing model

Ordered so that each phase leaves the suite green and the earlier phases make the later ones cheaper. Item codes refer to the appendix Must/Should/Consider lists; **D-n** refers to the decisions in §11, which override any appendix ranking they conflict with. Working rules for every item:

1. **Read the v1 implementation before writing** (the house rule). Where the appendix cites v1 lines, those are the reference.
2. **Every fix gets a test on `make_dense_metric_space()`** in addition to Euclidean and weighted, unless the code cannot touch a metric.
3. **Verify performance claims by counting** (operator applications, transforms, factorisations) with a `CallCounter`, not by reading; add the count as a test where a regression would be silent.
4. **Do not change a default without writing the measured reason** into the docstring and the current-state document (D-10).
5. **Update the catalogue row** for every item that changes what exists.
6. **Do not cut anything without a recorded reason** (D-13). If a v1 feature is in the way, keep it and open a question.

### Phase 0 — correctness (small, independent, do first)
§6 items 1–17, i.e. P Must-1, P Must-2, I Must-1, I Must-3, S Must-3, G Must-1, N Must-1, P Must-7 (at least: stop selecting the stochastic route under `"auto"` until it is fixed), I Must-5 + G Must-2, G Must-3, I Must-6, I Must-4, S Should-10, S Should-11, G Must-4, G Must-5, G Should-5. Also S Must-5 (LSQR `x0`), S Must-4 (callbacks in all solvers), I Must-7 (dead `**kwargs`), and the solver-default change D-8 (`rtol=1e-8`, `strict=True`, recorded).

### Phase 1 — the three structural repairs
- A Must-4: `with_traits` preserves the class (shallow copy). Then N Must-2 (`log_determinant` exact for diagonal; drop the `.with_traits(definite)` wrapping at the two call sites) and S Must-2 / I Must-2 (`DampedSolves` takes `assemble`; `_require_normal` accepts retraited operators). Test: `TikhonovFamily(..., solver=CGSolver().with_preconditioner(NormalDiagonalPreconditioner())).solve(1.0, b)`; `log_determinant(D.with_traits(PD))` is exact.
- P Must-3 / Must-4 / Must-5: `_rebuild` carries precision; scale/translate/affine/add/product rules; spectral KL quadratic term from eigenvalues; precision-only measures fail with a message. Then P Must-6 (`from_standard_deviations`), P Should-8 (`condition` gets a sampler and a solver hook — this is pyslfp blocker 3).
- A Must-3 + A Must-2: `MatrixLinearOperator` (dense or sparse, stored form, O(1) `matrix`/`diagonals`/`assembled`, `DirectSolver` reads it); `_Composition`/`_block_traits` read existing adjoint links only. Then S Should-9 (O(1) `diagonals` hook), S Should-12 (LU adjoint reuse, one adjoint cache), K Must-1 (MFEM `restrict` sparse, `_to_mfem` from CSR). Rename the matrix constructors per D-4 while touching them.

### Phase 2 — the sphere as pyslfp needs it, and the formal-adjoint lift
- **D-1, sphere vectors are `SHGrid`.** `Sphere` vectors become `pyshtools.SHGrid` objects; `sampling` is a constructor option (default 1, part of `_key`); `copy`/`axpy`/`scale_inplace` act on `.data` (v1 `sphere.py` is the reference); `to_components`/`from_components`, `multiply`/`truncate`, `basis_matrix`, the NUFFT paths, `plotting.plot` and `project_function` all take/return `SHGrid`. `ArrayVectorMixin` no longer applies to `Sphere`; keep it for the boxes. Test with `check_space`/`check_coordinates` and `test_sphere_transform` on both samplings.
- **D-2, points are `(latitude, longitude)` in degrees** everywhere on the sphere: `point_evaluation_operator`, `dirac`, `geodesic_*`, `walk_from`, `random_point(s)`, `stations`, `earthquakes`, `spherical_cap_*`, `plot_points`, `plot_paths`, `two_point_covariance`. Convert to colatitude/radians once, privately, at the boundary; keep a documented `Sphere.to_colatitude_radians` for anyone who needs it. Every radius argument says angular-degrees or physical.
- **D-3, explicit geometries.** Provide `symmetric_space.sphere`, `.circle`, `.line`, `.torus`, `.plane` (and `.box`) submodules each exporting `Lebesgue` and `Sobolev` as thin **subclasses** (so `isinstance` works and `type(X).__name__` says what it is), built on `Sphere`/`PeriodicBox`/`Box`. Keep `scale` as the Sobolev parameter name if the vector method can be renamed without breaking the Krylov code paths (`scale` → `rescale` or similar); otherwise keep `length_scale` and document.
- A Must-5: `MassWeightedSpace(base, mass, mass_solver)` on any `HilbertSpace`, and `LinearOperator.from_formal_adjoint(domain, codomain, operator, *, traits)` handling `DirectSum` recursively and identity-metric factors; `lift_formal_adjoint` becomes its diagonal fast path and stops round-tripping through components when the coordinate map is shared (Y Should-13). Test against pyslfp's shape: `[Sob, Sob, Sob, R²]` codomain, `R²` domain, and count transforms (forward must be 0 on shared grids).
- **D-11**: restore the Sobolev-order guard as a hard error with `unsafe=True` (Y Must-2).
- **D-12**: `path_integral_operator` (v1's action, honestly named) with the length-scale node heuristic; `path_average_operator` as the normalised variant; optional `weight=` callable along the path, with a docstring noting that a non-constant background is a composition with a multiplication operator (ray tracing is out of scope).
- **D-9**: `heat_measure(length_scale)` with symbol `exp(-length_scale² λ)` (v1's meaning); document the relation to diffusion time.
- **D-5**: top-level namespace — `from . import inference, numerics, plotting, geometry, symmetric_space` in `__init__` plus re-exports of the workflow names (K Must-2); preconditioners in one place.
- A Must-1: close the functional algebra (`_FunctionalSum/_Scaled/_Composition`, `LinearFunctional` closure and `from_callables`), per DESIGN §5.5. Test: `(phi + psi).at(x).gradient`, `(psi @ F).hessian(x)`, LBFGS on a composed misfit.
- I Should-13 / S Consider-24: surface `SolveResult` from `est(data)` (e.g. `est.solve(data)`) and a stock progress callback; iterate-carrying callback.
- **D-6, parallel hooks.** One `pygeoinf2/parallel.py` (joblib behind an optional extra, `n_jobs`/executor argument, serial fallback) and an `n_jobs` keyword at the embarrassingly parallel loops *around* operators: `ProbabilityMeasure.samples`, `LinearOperator.matrix()`/`assembled()`/`diagonals()` (hence every direct solver's assembly and `with_dense_covariance`'s successor), `random_range`/`random_trace`/`random_diagonal` probes, `pointwise_variance_at`. Not inside operator actions. Note finufft/pyshtools thread internally — set `nthreads=1` inside an outer parallel loop (Y Must-3).
- P Should-9/10/11, G Should-4, A Consider-19: matrix-free norms/balls/sparse approximation, `AffineSubspace.condition`, test measures in `check_operator`.

### Phase 3 — symmetric-space parity and performance
Y Must-1 (box geometry primitives — make them abstract or implement on `PeriodicBox`), Y Must-3 (finufft threads + `eps`/`nthreads` pass-through; re-measure crossovers), Y Must-4 (`DHaj`), Y Must-5 (`with_degree` on boxes), Y Must-6 (`to/from_coefficients` returning `SHCoeffs`/complex arrays, `from_coefficient_operator`, `power_spectrum`), Y Must-7 (KD-tree pairs), Y Must-8 (`bincount`), Y Must-9 (taper), Y Should-12 (caches keyed on `lmax`), Y Should-15/16/17 (`n_clusters`, correlated accessors, `truncation_degree_for`), Y Consider-30 (the cross-space parity test — do this one first, since it is what would have caught the box regressions).

### Phase 4 — solvers, numerics, preconditioners, convex
S Must-1 (export FlexibleCG/GMRES), S Must-6 (MINRES preconditioning), S Must-7 (sparse block/column-thresholded), S Should-8 (stochastic Jacobi), S Should-13/14 (`with_preconditioner` raises; Woodbury inner default), N Must-3/4/5 (Lanczos convergence in coefficient space and quadratic-form stopping; `random_range` incremental orthogonalisation + QR path; line search returns the gradient), N Should-8/9/10/11/12/15/16, N Should-17, I Should-9/10/11/12/14/15/16.

**D-13, convex solvers come back.** Port `LevelBundleMethod` (with its LP global bound), the `QPSolver` protocol and `SciPyQPSolver`/`OSQPQPSolver`/`ClarabelQPSolver`/`best_available_qp_solver`, `PrimalKKTSolver`, `ChambollePockSolver`/`solve_primal_feasibility`, `SmoothedDualMaster`/`SmoothedLBFGSSolver`, and `solve_support_values` with its warm start across directions, into `numerics.convex` as solver strategies (DESIGN §18.8's intent), selectable from `DualFeasibleProperty`/`FeasibleProperty` by a `method=` argument. This code was written by Mag; his view on the API is to be sought before changing anything beyond the port itself, and nothing in it is to be cut without his agreement. Restore the fused `value_and_subgradient` oracle (G Should-6) at the same time.

### Phase 5 — geometry, plotting, examples
G Should-1/2/3/7/8/10 (convex intersections, subspace equations, ellipsoid projection, field-plot options and dispatch, error bounds, tolerance semantics), G Must-7 (`title`, legend, kwarg table), K Should-9/10 (least-squares and nonlinear examples; assertions instead of prose), G Consider-1 (`plot_slice`). Plotting takes `SHGrid` on the sphere (D-1) and degrees (D-2).

**D-7, nonlinear inference.** In scope for 2.0: `MaximumAPosteriori(problem, prior, optimiser=...)` minimising `chi²(m, d) + prior.mahalanobis(m)` via `Operator.at`/`gauss_newton_hessian`, returning a Laplace `GaussianEstimator`-like object whose covariance is the inverse of `NormalOperator(F.derivative(m_map), prior, error, formalism="model_space")` (I Consider-19). Design it so a later MCMC layer (pCN / Stuart-style function-space samplers) plugs in: it needs only `prior.sample`, `log_density`/`grad_log_density` on the posterior, and the forward problem — all of which exist. Add an example.

### Phase 6 — tests, docs, packaging, design documents
K Must-3/5/6/7/8 (exports, wheel, `v2/` deletion, example fixes), K Must-4 (dense-Gram fixtures in the eight test files listed), K Should-12/13/14/15/16/17/18 (docstring-content check, rule-3 violations, slow markers, backend recipe, Sphinx for v2, catalogue reconciliation, missing tests), the Args/Returns/Raises pass per appendix §4 (start with `algebra/spaces.py`, `algebra/operators.py`, `algebra/direct_sum.py`, `probability/base.py`, `inference/gaussian.py`, `symmetric_space/*` units), the false docstrings in §3.8, and **D-10**: DESIGN.md stays as the journal (`DESIGN_LOG.md`), and a short current-state document is written (package map, contracts, conventions and units, the dense-fallback list, the backend recipe, the decisions of §11).

Deferred by agreement, not by omission: `dynamical_system`, sequential assimilation, PETSc (packaging question), `parallel=` inside operator actions (D-6 puts it around them), full MCMC sampling (D-7: seeds only).

---

## 11. Decisions taken (2026-08-27)

Your answers to the review's questions, restated as decisions with their consequences. These override any appendix ranking they conflict with.

| # | decision | consequences for the code |
|---|---|---|
| **D-1** | **Sphere vectors are `pyshtools.SHGrid` objects, not bare arrays.** Vectors may be general objects; that is a point of the library. Grid `sampling` defaults to **1** (v1's default) and is a user option. | `Sphere` drops `ArrayVectorMixin`; `copy`/`axpy`/`scale_inplace` on `.data`; `sampling` in the constructor and in `_key`; every method that produces or consumes a field (`to/from_components`, `multiply`, `truncate`, `project_function`, `basis_matrix`, NUFFT `evaluate`/`accumulate`, `plotting.plot`, `HilbertModule` ops) takes `SHGrid`. Boxes stay arrays. Halves field memory and pointwise cost against the current `sampling=2`. |
| **D-2** | **Sphere points are `(latitude, longitude)` in degrees**, as pyshtools uses. | Every point-taking method on `Sphere` and in `plotting/sphere.py` converts at its boundary; radius arguments say angular-degrees or physical; the data loaders stop converting privately; `Sphere.to_colatitude_radians` kept public for internal users. |
| **D-3** | **Explicit geometries are wanted**, ideally as `symmetric_space.line.Sobolev`-style submodules. | Submodules `sphere`, `circle`, `line`, `torus`, `plane`, `box` each exporting `Lebesgue`/`Sobolev` as thin subclasses of `Sphere`/`PeriodicBox`/`Box`, so `isinstance` works and the class name names the geometry. `Sobolev(lmax, order, scale)` call shape preserved where possible. |
| **D-4** | **Rename `from_derivative_matrix`.** `from_matrix(domain, codomain, M, *, form=...)` with `form` required is acceptable; the essential thing is that the docs make the convention unmistakable. | Rename both constructors into one; keep `LinearFunctional.from_derivative_components`; the docstring states the two representations with the formulas of DESIGN §5.3. |
| **D-5** | **Both**: subpackages importable from the top level and the workflow names re-exported flat. | K Must-2; preconditioners in one package. |
| **D-6** | **Parallelism around operators is wanted** — generating many samples must not require a bespoke loop each time. Inside operator actions it stays out. | A `parallel` helper (joblib optional extra, serial fallback) and `n_jobs` at `samples`, `matrix`/`assembled`/`diagonals`, randomised probes, `pointwise_variance_at`. Document the finufft/pyshtools thread interaction. |
| **D-7** | **Nonlinear MAP/Laplace is in scope for 2.0**; full function-space MCMC (Stuart-style) later, so its seeds should be laid now. | Phase 5 `MaximumAPosteriori`; keep `log_density`/`grad_log_density`/prior samplers as the interface a sampler will consume. |
| **D-8** | **Solver default `rtol=1e-8`, `strict=True`**, recorded with the measured reason. | S Should-15; docstring on `IterativeSolver`. |
| **D-9** | **`heat_measure` is parameterised by a length scale**, not a time. | `heat_measure(length_scale)` with `exp(-length_scale² λ)` (v1's meaning); document the diffusion-time relation. |
| **D-10** | **DESIGN.md stays as the journal; a short current-state document is added.** | Phase 6; module docstrings cite the current-state document, not DESIGN.md sections. |
| **D-11** | **Sobolev-order guard as in v1**: hard error, with an `unsafe=True` escape (useful pedagogically). | Y Must-2. |
| **D-12** | **`path_average_operator`: v1's action (the integral) was right, its name wrong.** Rename; optionally allow a weighting along the path; non-constant backgrounds by composition (ray tracing out of scope). | `path_integral_operator` (integral, length-scale node heuristic, optional `weight=`), `path_average_operator` (normalised). |
| **D-13** | **The convex solvers come back** (`LevelBundleMethod`, QP protocol and backends, `PrimalKKTSolver`, Chambolle–Pock, smoothed dual master, `solve_support_values`). This code is Mag's; his views are crucial and nothing is cut without good reason. | Phase 4; `method=` selection on the Backus routes; consult Mag before API changes; correct the catalogue rows (currently "Ported"/"Planned") to say they are being restored. |
