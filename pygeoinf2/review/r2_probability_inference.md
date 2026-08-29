# Re-review: probability + inference (PI) — 2026-08-29

Scope: `pygeoinf2/probability/{base,gaussian,mixture}.py`, `pygeoinf2/inference/*` except `preconditioners.py`; appendices P and I; REVIEW.md §10/§11.
Tests: `pytest test_probability test_normal test_mixture test_inference test_point test_backus test_laplace test_nonlinear test_phase_four -q` → **528 passed, 10 deselected (slow), 37 s**.
Scripts (all in `work/review_r2/PI/`): `count_ops.py` (operator/solve counting wrapper), `metric_checks.py` (closed forms on `make_dense_metric_space(5)`), `profile_ex21.py`, `ex21_solve.py`, `ex21_evidence.py`, `misc_checks.py`, `misc2.py`, `idreuse.py`, `backus_sweep.py`, `isotropic.py`, `ambient_scaling.py`. All runs `OMP_NUM_THREADS=1`.

## 1. Review items — status

| item | status | verified by | note |
|---|---|---|---|
| P Must-1 `from_covariance_matrix(form="components")` | done | gaussian.py:370-375; `metric_checks` #2 (3.6e-15 on dense G) | |
| P Must-2 mixture densities | done | mixture.py:323-330, gaussian.py:1330-1375; `metric_checks` #9 vs scipy (3e-16); test_mixture `..._match_an_independent_reference` | |
| P Must-3 precision through the algebra | partial | gaussian.py:1423-1453 (`_rebuild`), 1600-1643 (scale), 1533-1565 (translate/diagonal map), 1587-1596 (diag+diag), 308-315 (product); TestPrecisionSurvivesTheAlgebra | not carried: `correlated_measure` (symmetric_space/base.py:823-828 builds none), non-diagonal sums |
| P Must-4 spectral KL quadratic | done | gaussian.py:729-733; `metric_checks` #4 (closed form on dense G and on weighted space, 0 / 5e-15) | |
| P Must-5 precision-only measures | done | gaussian.py:540-565 `_require_covariance`; TestPrecisionOnlyMeasures | |
| P Must-6 `from_standard_deviations` | done differently | gaussian.py:162-211 | operator `diag(σ²)` on the space, not v1's component-basis reading; documented |
| P Must-7 stochastic KL | partial | gaussian.py:701-716 refuses under `"auto"`; `misc2.py`: KL(μ‖μ)=**368.8±34.8** and **−350.9±33.3** (Sobolev(16), dim 289, 50 probes) | route itself unfixed (no preconditioning, `with_traits(definite)` kept, error bar wrong); no block-spectral route for correlated measures |
| P Should-8 `condition` sampler + solver hook | done | gaussian.py:1107-1199; `metric_checks` #5 (mean 1e-13, cov 9e-13, exact-constraint 2e-12, 20k-draw cov 1.5e-2) | |
| P Should-9 matrix-free norms / ball radius / `rank=` | not done | gaussian.py:494-526 (dense component matrix), 1066-1067 (dense `eigvals`) | docstring 488-489 promises `"stochastic"`: false, see §2 B2 |
| P Should-10 sparse approximation purpose | not done | gaussian.py:956-975 dense matrix, dense `eigvalsh`, no precision, no sampler | |
| P Should-11 generic `sample_pointwise_variance` | not done | grep: only `SymmetricSpace.pointwise_variance_at` (base.py:969) | |
| P Should-12 PSD-singular `from_covariance_matrix` + precision | not done | gaussian.py:380 `np.linalg.cholesky`, no precision attached | v1 gaussian_measure.py:215-278 (eigh + clip) accepted PSD |
| P Should-13 `walk_from` on boxes | done | fourier.py:844 | area Y |
| P Should-14 `sample_power_measure` | done differently | symmetric_space/base.py:334 `power_spectrum(x)` | per-sample spectra are a comprehension over it |
| P Should-15 `check_measure` checks precision | not done | testing.py:734-…: no `precision` reference (grep) | |
| P Should-16 `__neg__`, `directional_statistics`, correlation forms, `marginal`/`cross_covariance` | partial | gaussian.py:1212, 1255 done; `__neg__`/`directional_statistics` absent (grep); base.py:864-870 accepts only `(n,n)`/`(dim,n,n)` | |
| P Should-17 `heat_measure` param; dense-cost docs | partial | base.py:688 `heat_measure(length_scale)` (D-9) | `with_sparse_approximation`, `ambient_ball`, norms still do not state their dense cost |
| P Should-18 seeding API | not done | spaces.py:40 `_DEFAULT_RNG`, no `seed` (grep) | |
| P Consider-19 Args/Raises | done | test_code_practice contract | |
| P Consider-21 dead operator in `pointwise_variance_at` | done | base.py:1010-1013 | |
| P Consider-22 rename `marginal_probabilities`, `samples=`→`probes=` | not done | mixture.py:358; gaussian.py:579 | |
| P Consider-23 `n_jobs` on `samples` | done (D-6) | base.py:76-127; `pointwise_variance_at` has it too (base.py:977) | |
| P Consider-24 catalogue rows | partial | V1_CATALOGUE.md:258 still "as `Bayesian`" | |
| I Must-1 `for_data` | done | point.py:475-513 shared `_searched_damping`; test_point `..._refused_by_both_routes` | |
| I Must-2 `DampedSolves` builds Tikhonov members | done | root_find.py:329-341, tikhonov.py:807-817; test_point `..._preconditioner_works_inside_the_sweep` | |
| I Must-3 `from_direct_sum` noise | done | problem.py:302-318; test_inference `..._exact_one_is_refused` | |
| I Must-4 `TikhonovFamily.weighted_adjoint` | done | tikhonov.py:848-856; TestTikhonovFamilyAccessors | |
| I Must-5 `_harden` → `ambient_ball` | done | backus.py:91-126; TestHardeningTheError | but O(n³) dense, §3 O7 |
| I Must-6 no-error model-space formalism | done | normal.py:213-226; test_normal `..._has_no_model_space_form` | |
| I Must-7 dead `**kwargs` | partial | gaussian.py:327-377 clean; point.py:827-867, 1015-1030 still forward `**kwargs` to `LinearForwardProblem.parameterised`, which takes none → `TypeError` (`misc_checks.py`) | §2 B4 |
| I Should-8 top-level exports | done | `pygeoinf2.LinearGaussianInversion` (`misc_checks.py`) | |
| I Should-9 log-det memo | done | gaussian.py:494-524; `count_ops`: second `log_evidence` = 1 solve, 0 assembly | |
| I Should-10 joint-spectrum cache | done | backus.py:398-412 `cached_property` | |
| I Should-11 `_linearise` once; derivative solves | done | point.py:634-646; `misc2`: derivative and adjoint 1 solve each | |
| I Should-12 constrained `parameterised`/`data_reduced` | done | point.py:827-932, 1015-1044; TestConstrainedEstimatorsCanBeReduced | |
| I Should-13 `with_solver`/`with_formalism`/`residual_callback` | partial | grep: absent on `DiscrepancyPrinciple`, `ConstrainedMinimumNorm`, `LinearGaussianInversion.residual_callback` | |
| I Should-14 joint-inversion test | done | TestJointInversionEndToEnd | |
| I Should-15 typed errors for set-valued error | done | problem.py:96-136 | |
| I Should-16 `FeasibleProperty` docs / `is_feasible` | partial | `is_feasible` backus.py:352, 767, 1440; `solver:` doc backus.py:537 still "how to invert the damped normal operator" — it is used only in `_reduced` (824); orthonormal requirement stated only in `_data_gram` (582-587) | |
| I Should-17 DESIGN/catalogue drift | partial | V1_CATALOGUE.md:258 | |
| I Should-18 docstring pass | done | test_code_practice | |
| I Consider-19 / D-7 MAP-Laplace | done | laplace.py; test_laplace | |
| I Consider-20 collapse `MinimumNorm` | not done | point.py:334 | shares the search now; harmless |
| I Consider-21 fused dual oracle | done differently | backus.py:1062-1072 one-entry memo keyed on `id()` | **bug**, §2 B1 |
| I Consider-22 `propagated_covariance` property | not done | estimators.py:634 | |
| I Consider-23 rename `for_data`; one `choose_formalism` | not done | point.py:48 and normal.py:50 both exported | |
| I Consider-24 typing of `from_product` callers | not done | problem.py:328, 347 | |
| D-6 parallel hooks in this area | done | base.py:76, symmetric_space/base.py:977, backus.py:1260 | |
| D-13 Backus routes | done | `backus_sweep.py`, 16 directions: kkt 0.022 s, smoothed 0.015 s, primal 0.028 s, dual 8.6 s, all agreeing to 1e-6 | dual cost is the bundle subproblem (42 A applications/direction), Mag's code |
| Metric rule (this round) | pass | `metric_checks.py`: `as_multivariate_normal`, `from_covariance_matrix`, dense/spectral KL, `condition`, `credible_set`, norms, normalising constant, mixture, `ambient_ball` coverage — all closed-form on dense G (≤1e-12) | |
| K Must-4 dense fixtures in this area | partial | `make_dense_metric_space` uses: test_probability 5, test_inference 2, test_point 2, test_mixture/test_normal/test_backus/test_laplace **0** | |

## 2. Bugs and regressions found now

**Verified**

- **B1. `DualFeasibleProperty.dual_cost` memo is keyed on `id(certificate)`** (backus.py:1064-1072). Once a certificate array is freed, the next array may get the same `id`, and `gradient(new)` returns the *previous* certificate's residual/negation. Reproduction `idreuse.py`: evaluate `cost(λ1)`, `del λ1`, allocate `λ2` with the same id → wrong gradient in **145/200 trials**. Inside `support()` the bundle keeps references to its points so it is probably latent there; any other optimiser passed as `method=`, or a caller doing a line search, is exposed. Fix (one line): keep the certificate object in the cache (`cache["certificate"] = certificate`; compare with `is`) — the reference keeps the id alive.
- **B2. `hilbert_schmidt_norm`/`nuclear_norm(method="stochastic")` silently form the dense component matrix** (gaussian.py:494-502, 516-526) while the docstring (488-489) says `"stochastic"` estimates the trace. `metric_checks.py` NOTE line. v1 had `random_trace` (gaussian_measure.py:1536-1647).
- **B3. The stochastic KL route is still wrong, and its error bar is wrong** (gaussian.py:754-793). KL(μ‖μ) on Sobolev(16), dim 289, 50 probes: `368.8 ± 34.8` and, with the reference's precision available, `−350.9 ± 33.3` (`misc2.py`). Opt-in only now, but P Must-7's fix items (precondition the CG, larger `max_iterations`, drop `with_traits(definite)`, refuse when the error exceeds the value) were not applied.
- **B4. `ConstrainedLeastSquares.parameterised(P, **kwargs)` and `ConstrainedMinimumNorm.parameterised`** (point.py:862, 1025) forward `**kwargs` to `LinearForwardProblem.parameterised`, which accepts none: `cls.parameterised(P, dense=True)` → `TypeError` (`misc_checks.py`). Dead parameter, I Must-7 leftover.
- **B5. `from_covariance_matrix` still refuses PSD-singular input and attaches no precision** (gaussian.py:380). v1 (gaussian_measure.py:215-278) took `eigh`, clipped, and attached the inverse factor. Regression carried over, not recorded as a decision.
- **B6. `FeasibleProperty` `solver` docstring is still false** (backus.py:537 vs use at 824). Documentation only.

**Unverified suspicions**

- U1. `condition(A, value)` with `noise=None` and dependent rows of `A` hands a singular `A C A*` (claimed `POSITIVE_DEFINITE`, gaussian.py:1159) to CG; no test covers it.
- U2. `_combine_add` (gaussian.py:1571) returns `None` when either side lacks a covariance, so `P + P` for precision-only measures becomes `_IndependentSum` with no density — consistent with the P Must-5 "fail clearly" rule only if that is documented.

## 3. Optimisations, ranked (gain × confidence)

1. **The normal operator round-trips through the grid on a spectral space.** `forward @ prior.covariance @ forward.adjoint` (normal.py:238) applies `A*` (`from_components`, operators.py:1439-1443), `Q` (`from_components(λ·to_components(x))`, diagonal.py), then `A` (`to_components`, operators.py:1432): **4 spherical-harmonic transforms per application** where a component-space `A_c (λ ⊙ G⁻¹ A_cᵀ v) + σ²v` needs none. Measured on example 21 (lmax 48, dim 2401, 960 data; `ex21_solve.py`, interleaved, 5 rounds): 2.65–2.89 ms vs 1.16–1.40 ms per application (**2.3×**); one CG solve 0.21–0.23 s vs 0.09–0.10 s (**2.4×**, 85 iterations, 340 transforms), solutions agreeing to 1e-10. Every solve, sample and covariance action in the tomography workflow pays this (≈0.9 s of the 3.8 s example, the rest being operator assembly, area Y). Fix: either a fast path in `NormalOperator.__init__`/`_value` when `forward` is a `MatrixLinearOperator` and `Q` a `DiagonalLinearOperator` on a diagonal-metric space (local, low risk), or a `_Composition` rule fusing adjacent component-native factors (algebra area, general). Risk: low — same arithmetic, verified to 4e-15.

2. **`dense_limit=512` sends the evidence to the wrong route at realistic data sizes.** On example 21 (dim 960, `ex21_evidence.py`): dense log det exact in **2.78 s**; stochastic default (100 probes) **−7270 ± 30** in **4.84 s** against the exact −7315; `sample_rtol=1e-3` reaches ±16 after 400 probes in 23 s. A 30-nat error bar makes a model comparison meaningless, and the dense route is also faster until dim ≈ `samples × max_iterations` ≈ 4000 applications. (functional_calculus.py:515 default; gaussian.py:431-524 inherits it.) Fix: raise the default to ~4000 for `"auto"`, or pick by estimated applications; leave memory as the guard. Gain: 1.7× and exactness at 512 < dim ≲ 4000. Confidence high.

3. **Evidence with a direct solver assembles `N` twice.** `CholeskySolver` extracts `N.matrix()` at construction (solvers.py:397); `normal_log_determinant` then calls `log_determinant(self._normal)` (gaussian.py:501), whose dense route extracts it again (functional_calculus.py:598). Measured (`ex21_evidence.py`): construction 2.63 s, `log_evidence` **4.33 s** — where `2 Σ log diag L − log det G` is free from the factorisation. `count_ops.py` shows the same: `log_evidence` after Cholesky = 6 fresh N applications on a 6-dim data space. Fix: let a direct-solver `InverseOperator` expose `log_determinant` (or its stored matrix) and read it in `normal_log_determinant`. Gain: the whole second assembly (~50% of a Cholesky-solved evidence calculation). Confidence high; risk low.

4. **Two solves per datum where one suffices.** The mean is `gain(d) + translation` (gaussian.py:119-124) and the misfit solves `N⁻¹(d − shift)` (gaussian.py:416-418), so `est(data)` + `log_evidence(data)` = 2 solves; `LinearGaussianMixtureInversion.__call__` = **2K solves** (`count_ops`: `solve=4` for K = 2: mixture.py:1071-1072 means, then `weights` → `mahalanobis`), and `_PushedMixture.__call__` 2K again. v1 computed `m₀ + K(d − shift)` (linear_bayesian.py:353-362), which shares the solve. Fix: one private `_residual_solve(data)` returning `N⁻¹(d − shift)`, used by the mean (`m₀ + Q A* w`) and the misfit (`(d − shift, w)`), memoised on the last residual (key on content, not `id`, see B1), and used by the mixture explicitly. Gain: 2× on mixture inversion and on posterior+evidence loops; each solve on example 21 is 0.14–0.22 s. Confidence high.

5. **`estimator.push_forward(T)(data)` re-solves; `posterior.push_forward(T)` does not.** `count_ops`: the estimator route costs `solve=1` per property posterior; `misc_checks`: the measure route costs 0 solves and gives the identical measure (mean 4e-16, covariance 0, still samplable). Example 21 line 101 uses the expensive form (0.136 s, 15% of its inference time). Fix: use `posterior.push_forward(caps)` in the example/docs, or memoise as in item 4. Confidence high.

6. **`DampedSolves` with a direct solver re-extracts and refactorises at every multiplier** (root_find.py:404-406 → solvers.py:394-398). `DiscrepancyPrinciple(problem, solver=Cholesky)(data)` on a 6-datum problem: **23 factorisations, 162 forward + 162 adjoint applications** (`misc2.py`) where `base.matrix()` and `shift.matrix()` are fixed and 6 + 6 applications would do (23×). v1 did the same (linear_optimisation.py:871-876) — not a regression. Fix: cache the two matrices in `DampedSolves` when the solver is a `DirectSolver` and factorise `B + tS`. Confidence high; risk low.

7. **`ambient_ball` is O(n³) even for a known-diagonal covariance** (gaussian.py:1066-1067: `matrix(form="components")` then dense `eigvals`). `ambient_scaling.py` on `from_standard_deviation` (a `DiagonalLinearOperator`): 0.06 / 0.18 / **1.15 s** at dim 2000 / 4000 / 8000; 800 MB at 10⁴; impossible at pyslfp's 10⁵ observations. `harden_error` (backus.py:126) hits this on every Backus route with a Gaussian error. Fix: `_diagonal_eigenvalues()` fast path (v1's `_kl` case), then a randomised spectrum (v1 `LowRankEig`, gaussian_measure.py:748-798) or a sampling radius. Gain: O(n³) → O(n) in the common case. Confidence high.

8. **Three covariance applications per posterior-covariance action where two suffice.** `Q − K A Q` (normal.py:475) and `C − cross inverse cross*` (gaussian.py:1170) apply `Q`/`C` to the input twice (`count_ops`: `Q=3` per action, 36 per 12-column matrix). Rewrite as `(I − K A) Q` and `(I − cross inverse A) C`. Gain: 33% of `Q` applications — irrelevant for a diagonal prior, material for MFEM/correlated priors. Same in v1 (linear_bayesian.py:338). Confidence high; risk trivial.

9. **Dense KL assembles the reference covariance twice** (gaussian.py:747 `other._weighted_squared` → 538 `_symmetric_matrix()` again): `C1=24` vs `C0=12` applications at dim 12 (`misc2.py`). Pass `theirs_matrix` into the quadratic. Gain: 1/3 of the dense route's assembly. Confidence high.

10. **`FeasibleProperty._state` recomputes `vectors.T @ forward_w`** (backus.py:647-648) at each of 60 inner bisection steps though it depends only on `weight`: 16 directions cost 0.97 s with only 88 operator applications (`backus_sweep.py`). Hoist per weight (~2× on that route). Low priority — `route="kkt"` answers the same question in 0.022 s.

Not proposed (under 10% or already minimal): posterior sample (1 prior draw + 1 noise draw + 1 A + 1 solve + 1 A* + 1 Q — minimum; `count_ops`), gain (1 solve), mixture sampling (categorical + one draw), `condition` sampler (minimum), `as_multivariate_normal`'s second gram-solve pass.

## 4. Open questions for the user

1. Item 4 changes what `LinearGaussianInversion.__call__` computes internally (mean as `m₀ + K(d − shift)`) and needs either a content-keyed one-entry memo or an explicit combined entry point (`posterior_and_evidence(data)`). Which is preferred? The memo is invisible; the entry point is honest about the shared solve.
2. Item 2: raise `dense_limit` globally in `log_determinant` (affects every caller) or only in `normal_log_determinant`?
3. B3: fix the stochastic KL route (preconditioned CG, honest error bar) or delete it? It is opt-in, but it still returns confidently wrong numbers.
4. Item 7: for non-diagonal covariances, restore v1's randomised spectrum (`LowRankEig`) or its sampling radius (`radius_method="sampling"`)? Both existed; neither survived.
5. Item 1: fuse in `NormalOperator` (local, covers the Gaussian and Tikhonov normal operators) or in `_Composition` (general, algebra area, touches every product of component-native operators)?
