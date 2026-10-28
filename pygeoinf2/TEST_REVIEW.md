# pygeoinf2 test-suite review

2026-10-28. Scope: `pygeoinf2/tests/` on branch `refactor`. Nothing outside this file was changed.

**Line numbers** refer to commit 795b80a. During the review another session edited five test files in the working tree:

- `test_laplace.py` and `test_no_array_backend.py`: both macOS exemplars are already being fixed.
- `test_assorted.py`: +1 line at 24, +100 lines at 1446.
- `test_known_structure.py`: +24 lines from 123 on.
- `test_miniapp_oceans.py` was renamed `test_miniapp_point_data.py` and is now module-wide `slow`.

**Method.** I read `conftest.py`, `doubles.py`, `test_laplace`, `test_no_array_backend`, `test_optimization` and `test_minimization` in full. Six read-only reviewers read the other 50 files in full. I verified every High and Medium item in §B by reading the cited lines, and ran the vacuous counter in §E. Everything else is the reviewers' citation and was not re-checked.

**Runs.** These were on a shared Linux machine, without xdist. Another session was running the point-data mini-app with loky workers at the same time (load average 12–15), so absolute times are inflated; the ranking is what matters.

- Fast suite: **3352 passed, 40 deselected, in 485 s.**
- `-m slow`: **40 passed in 132 s.**
- **Coverage: not obtained.** `--cov=pygeoinf2` hit the 15-minute cap twice: on the fast suite at 99%, and without tutorials and examples at 71%. I deleted the partial `.coverage.*` files the first run left in the repo root. §D uses a scripted scan of every module's `__all__` against the test sources instead.

## A. Summary

### Suite shape

- 56 test files plus `conftest.py` and `doubles.py`, about 36.5k lines, 3392 collected items.
- `conftest.py` provides a diagonal (`WeightedSpace`) and a dense-Gram (`DenseMetricSpace`) coordinate space.
- `doubles.py` provides three coordinate-free doubles. `ListSpace`, the only one with a non-diagonal Gram, is used in one file (`test_no_array_backend.py`).

### CI never runs the slow tests

- `.github/workflows/ci.yml:67` runs `poetry run pytest --cov=pygeoinf --cov=pygeoinf2`, and `pyproject.toml:133` (`addopts = "-m 'not slow'"`) applies.
- So **none of the 40 slow tests has ever run on Windows or macOS.** Those are the parity and Monte Carlo checks the pyproject comment calls "the parity checks the design record leans on".
- The 16–28 minutes per platform in CI is the *fast* suite under coverage tracing. Locally that took over 15 minutes, against 8 without `--cov`.

### Where the runtime goes

- 12 fast-suite tests take 302 s of the 485 s (62%): tutorials 187 s (39%), ray-operator parity 86 s (18%), and three memo tests that call `gc.collect()` in a loop, 44 s (9%).
- **The `slow` marker is misallocated.** The whole slow set runs in 132 s, and 15 of its 40 tests take under 1 s each, while the default run carries tests of 87 s, 48 s, 31 s and 27 s.

### 15 slowest tests (fast and slow runs combined)

| s | marker | test |
|---|---|---|
| 87.2 | fast | `test_tutorials.py::test_the_tutorial_runs[07_properties]` |
| 47.7 | fast | `test_tutorials.py::test_the_tutorial_runs[13_ray_tomography]` |
| 34.0 | slow | `test_examples.py::test_the_example_runs[22_coupled_fields]` |
| 31.0 | fast | `test_rays.py::TestInAField::test_shooting_agrees_with_bending[plane]` |
| 26.7 | fast | `test_backus.py::TestTheDualCostMemo::test_a_recycled_address_gives_no_stale_gradient` |
| 26.7 | fast | `test_tutorials.py::test_the_tutorial_runs[14_one_problem_three_answers]` |
| 24.7 | slow | `test_miniapp_point_data.py::test_the_setup_is_over_the_sea` (setup) |
| 17.4 | fast | `test_rays.py::TestInAField::test_shooting_agrees_with_bending[torus]` |
| 16.2 | fast | `test_rays.py::TestTheShootingHessian::test_it_matches_differences_of_the_derivative_and_is_symmetric[sphere]` |
| 13.6 | fast | same, `[plane]` |
| 10.0 | slow | `test_examples.py::test_the_example_runs[24_preconditioning]` |
| 9.7 | fast | `test_normal.py::TestTheSharedSolve::test_the_memo_is_keyed_on_the_contents_not_the_address[model_space]` |
| 9.1 | fast | `test_tutorials.py::test_the_tutorial_runs[04_solvers]` |
| 8.8 | fast | `test_plotting.py::TestSampledDensity::test_the_panel_agrees_with_the_exact_estimate` |
| 8.1 | fast | `test_tutorials.py::test_the_tutorial_runs[06_linear_forward_problem]` |

### Overall judgement

The suite is strong where this library most needs it:

- Algorithms are mostly checked against independent computations: dense NumPy closed forms, SciPy SLSQP, `chi2`, `brentq`.
- The adversarial doubles turn "coordinate-free" and "metric-correct" into assertions.
- Monte Carlo is seeded, almost always at 5 standard errors or more.
- All 13 `check_*` functions have negative controls, and refusals are tested by message.
- There are no wall-clock or speedup assertions.

Its weaknesses are five:

1. Iterative methods asked for tolerances at or below what double precision allows, with `converged` then asserted: `gtol` 1e-11 to 1e-14; strict CG at `rtol` 1e-12 or 1e-13 under the default cap `max(2·dim, 20)`; Splitting, Dual and Chambolle–Pock at 1e-9 or 1e-10. **This is the macOS failure class, and about 40 sites share it.**
2. A few assertions about behaviour that rounding produces: an asserted *non*-convergence, exact iteration-count equality across code paths, and a C++ RNG whose output is implementation-defined.
3. Cost in the wrong place (§A, §C).
4. The metric rule is honoured unevenly. `test_normal`, `test_convex` and `test_mixture` run only on Euclidean or diagonal spaces, and so does most of `test_feasible` and `test_backus`.
5. About 165 lines of v1, history and planning narrative in docstrings and test names, which nothing enforces against.

## B. Brittle tests, ranked by likelihood of a cross-platform failure

### High

**B1. `test_laplace.py::TestAgainstTheRecordedAnswers` (:570-626).**
- *Asserts:* `iterations == 319` and `== 9`; recorded values at rel 1e-9 and 1e-8.
- *Status:* fails on macOS. The working tree now drops the counts and loosens to rel 1e-5/1e-6.
- *Replace:* also assert stationarity independently, `objective(data).gradient(minimizer)` ≤ 10·gtol. Keep the recorded numbers only as a loose regression guard. The docstring at :565-568 ("before it was rebuilt") is history.

**B2. Optimizers asked for a gradient below the rounding floor, with `converged` asserted.**
- *Mechanism (verified):* a failed line search counts as success only if ‖g‖ ≤ √eps·(1+|f|), about 1.5e-8·(1+|f|) (`numerics/optimization.py:204-219`). So with `gtol` = 1e-11, `converged` depends on where the last failed search stops relative to a heuristic floor. That is the macOS `SteepestDescent` failure (‖g‖ = 5.5e-8).
- *Sites:*
  - `test_no_array_backend.py:785-797`: four optimizers are still at 1e-11 after today's edit; :808 has an inner NewtonCG at 1e-12.
  - `test_minimization.py:73` (with `converged` at :104); also :208, :228, :237-240, :334/:348, :384/:389.
  - `test_optimization.py:934/:945` and `:968/:973`.
  - `test_convex.py:244-247` (gtol 1e-14 plus `converged`); :283, :292, :411 use the same gtol without the assert.
  - `test_laplace.py:112`, `:206`; `test_geometry.py:554`, `:866`.
- *Replace:* use the default gtol 1e-8 (1e-9 for Newton on a quadratic). Assert the minimizer against the closed form at gtol/λ_min. Assert `converged` only where gtol > √eps·(1+|f|).

**B3. `test_solvers.py:928-956` `test_it_is_what_makes_a_dense_metric_solvable`.**
- *Asserts:* `not plain.converged` after 3000 MINRES iterations on a 40-dimensional system, and `pre.iterations < plain.iterations / 10`.
- *Why fragile:* in exact arithmetic MINRES finishes in at most 40 steps, so the failure is loss of orthogonality, which rounding produces. The Gram is unscaled `tril(N(0,1)) + 2I`, the construction `conftest.py:134-140` warns about.
- *Replace:* `make_dense_metric_space(40)`; assert that the preconditioned solve converges with error < 1e-6. If the contrast matters, run plain with `strict=False` and a cap of about 200, and assert `pre.iterations <= plain.iterations`.

**B4. `test_miniapp_cpp_backend.py:188-202` `test_feasible_set_brackets_the_truth` (and :186).**
- *Why fragile:*
  - The truth, the data and the ambient-ball draws come from `std::normal_distribution` (`miniapps/cpp_backend/cppvec.cpp:70-71`). Its output is implementation-defined, so libstdc++, libc++ (macOS) and MSVC differ for one `mt19937_64` seed.
  - Bracketing the truth needs truth and noise inside their 95% sets, so roughly 1 realization in 10 fails, and the seed pins the realization only on Linux.
  - The build runs in CI on all three OSes (`nanobind` is a dev dependency; cmake is on the runners).
- *Replace:* draw in NumPy and hand the values to the backend, or Box–Muller on `mt19937_64`.

**B5. `test_probability.py:1317-1335` `test_workers_get_one_thread_each`.**
- *Asserts:* workers see `OMP_NUM_THREADS == OPENBLAS_NUM_THREADS == "1"`.
- *Why fragile:* `parallel.py:40-42` says an exported variable wins, so a runner that exports either fails the test. Two neighbours pin joblib and CPython internals: `"MainThread"` at :1355-1360 and `"DummyProcess"` at :1363-1379.
- *Replace:* `monkeypatch.delenv` both variables first, and assert the contract rather than class names.

### Medium

**B6. Strict CG at rtol 1e-12/1e-13 under the default cap `max(2·dim, 20)`** (`solvers.py:1193`). Non-convergence raises.
- *Sites:*
  - `test_point.py:480` (`CGSolver(rtol=1e-13)` for an exact identity); `test_point.py:789` (inside a damping search, where a ConvergenceError propagates out of Brent); `test_point.py:514`, `:554`.
  - `test_inference.py:678`, `:890`.
  - `test_flexure.py:524`, `:538` (1e-13, only so that `check_operator` passes).
  - `test_solvers.py:80`, `:88`, `:444-459` (the plain comparator is a strict CG at κ ≈ 1e5 with cap 24).
- *Replace:* `CholeskySolver` for exact identities; otherwise rtol 1e-10 with comparisons at 1e-6, or `max_iterations=200`.

**B7. Exact iteration-count equality between two arithmetic paths.**
- *Sites:* `test_known_structure.py:495-503` (×6 solvers, rtol 1e-12) and `test_assorted.py:211` (CG vs flexible CG).
- *Why fragile:* this holds only if the paths are bitwise identical.
- *Replace:* |Δ| ≤ 1 plus a comparison of the solutions.

**B8. Splitting, Dual and Chambolle–Pock at 1e-9/1e-10 with `converged` asserted.**
- *Sites:*
  - `test_feasible.py:849-856`: tol 1e-9 and 50,000 iterations. The step sizes come from a power iteration on `random(rng=None)` (`feasible.py:2527`, `convex.py:1589`), so the run is not reproducible even on one machine.
  - `test_feasible.py:774` (Dual's default gap is 1e-10); `test_backus.py:1506-1510` (bundle at 1e-10).
  - `test_quadratic_programming.py:217-246`: Chambolle–Pock at 1e-9, then an absolute residual < 1e-8; `:202-215` (slow) has the same pattern.
- *Replace:* default tolerances, a fixed `rng`, and assert that the brackets overlap (`a.inner ≤ b.value + δ`, `b.inner ≤ a.value + δ`) instead of `converged`.

**B9. Bounds that depend on the QP backend.**
- *Sites:*
  - `test_quadratic_programming.py:117-134`: Clarabel ≤ projected gradient + 1e-12, while Clarabel's own gap is 1e-8.
  - `test_quadratic_programming.py:55-62`: OSQP at rel 1e-6, which equals its default eps.
  - `test_convex.py:459`: `iterations < 120`, which depends on which QP solver is installed.
  - `test_convex.py:491-533`: acceleration must win on at least 85% of 30 random problems. It measured 28 wins against 26 needed, with residuals at the rounding floor; it is also a speed test of a private function.
- *Replace:* a slack of 1e-7·|f|; drop the iteration bound; reduce the statistic to one instance with a residual bound.

**B10. Parallel vs serial at rel 1e-8** (`test_feasible.py:714`, `test_parallel_knobs.py:212`).
- *Why fragile:* Primal warm-starts Brent per thread id (`feasible.py:1843-1849`), so the result depends on scheduling, and its own accuracy is O(1e-8).
- *Replace:* rel 1e-6, or assert each value lies inside the serial bracket.

**B11. `test_assorted.py:901-927`.**
- *Asserts:* a diverging preconditioned CG warns "did not converge".
- *Why fragile:* warning versus raising (a negative (r, Pr), `solvers.py:1347-1365`) depends on rounding along the divergent run.
- *Replace:* assert the structural fact (the probed band differs from the exact band, or is indefinite).

**B12. `test_probability.py:183-200`.**
- *Why fragile:* the singular covariance is factored by `eigh`, whose eigenvector signs depend on the LAPACK build, so one seed gives a different realization per platform. The bound is about 3 SE at 4000 draws.
- *Replace:* 6·√(2/n)·max|C|, or `check_measure`.

**B13. Two routes compared at 1e-11/1e-12 on quantities that can cancel.**
- *Sites:*
  - `test_sphere_transform.py:111-123` (an inner product of random fields at rel 1e-12), `:147-149`, `:199-210`, `:350-359`.
  - `test_flexure.py:122`; `test_ball.py:297`, `:305`.
  - `test_spectral_element_plotting.py:125`, `:219`, and `:133-135` (`round(·, 9)`, then `unique`, then `array_equal`).
  - `test_functional_calculus.py:886-905` (eigenvectors from two LAPACK drivers); `test_solvers.py:1497-1511` (generalized eig vs LU, κ uncontrolled at damping 0).
  - `test_mixture.py:296`; `test_mfem_hilbert.py:146` (Linux only).
- *Replace:* absolute tolerances scaled by ‖x‖‖y‖ or max|values| at 1e-10.

**B14. Network access.** `test_sphere_transform.py:457-478` calls `_land_test("110m")`, so cartopy downloads Natural Earth on a fresh runner. `test_examples.py:62-78` guards against exactly this. `test_miniapp_point_data.py:76` is the same (slow only). *Fix:* move `_coastlines_are_cached` into conftest.

**B15. Other.** `test_space_parity.py:654-664` treats finufft's ℓ2 `eps` as a pointwise bound. `test_plotting.py:361-396` samples rendered pixel luminance. `test_plotting.py:538` reads the private `fig._suptitle`.

### Low

Deterministic, but tied to today's numbers:

- `test_point.py:151` (`calls <= 14`, measured 10–13); `test_solvers.py:423` (`warm.iterations == 1`).
- `test_point.py:715` (slow; warm < cold iterations, and rel 1e-6 against a bracket of about 2e-6).
- `test_optimization.py:641-665` (Wolfe < 0.75 × Armijo evaluations over 5000 iterations).
- `test_tutorials.py:203-221` (notebook bytes against the installed nbformat).
- `test_compat.py:98`, `:452` (`np.random.seed` leaks global state), `:277`, `:285` (unseeded).
- `test_radial.py:115-116` (eigenvector signs); `test_space_parity.py:916-922` (`== 354` via libm `pow`).

These look exact but are fine: `test_optimization.py:353`, `test_point.py:193`, `test_convex.py:666-667`, `test_geometry.py:910-924`, and the call counts in `test_nonlinear.py`.

### Skip hygiene

CI installs `--all-extras`, so these bite only on partial installs.

- `test_parallel_knobs.py:235` and `test_escape_hatches.py:150` call `importorskip("finufft")` at module level after earlier classes; their sphere tests do not importorskip pyshtools.
- The planetmodel gate, copied into 5 files (`test_ball.py:18`, `test_interval.py:12`, `test_radial.py:12`, `test_layered.py:11`, `test_spectral_element_plotting.py:16`), checks a 1.2-era attribute, while the pin is ≥1.3.
- `test_spectral_element_plotting.py:19` gates the map tests on pyshtools, but they need cartopy.
- The C++ build decodes compiler output with `text=True`, i.e. in the locale codec (cp1252 on Windows).

## C. Tests that go too far

### C1. Cost centres in the default run

| cost | where | what to do |
|---|---|---|
| 187 s | whole tutorials (`test_tutorials.py:56`), 07/13/14/04/06 the heaviest; each asserts only that something printed | a quick-scale switch, or 07/13/14 into the file's `SLOW` set plus a scheduled job |
| 86 s | `test_rays.py:128-142` (plane 48×48, torus 48×40, sphere lmax 24; 48 segments vs 384 RK steps) and `:238-256` | the only shooting-route checks, so keep them; halve the grids and counts |
| 44 s | `gc.collect()` inside loops: `test_backus.py:1908-1922` (200×), `test_normal.py` memo test (50× ×2) | 5–10 iterations |
| 16 s | an exact KDE on 20000 and 8000 draws over 160×160 grids (`test_plotting.py:429-463`, `:465-499`) | 1000–2000 draws on a 60-point grid |
| per session | the cmake build of the C++ mini-app (`test_miniapp_cpp_backend.py:51-79`) | cache by source hash |

### C2. The slow set

It never runs in CI, and 15 of its 40 tests take under 1 s:

- `test_point.py:715` (0.03 s), `test_quadratic_programming.py:202` (0.19 s)
- `test_backus.py:1689`, `:1733`, `:1834`, `:1845`
- `test_functional_calculus.py:629`, `:649`
- `test_spectral.py:330`, `:359`
- `test_probability.py:1270`, `test_plotting.py:260`

Shrink these and unmark them. Keep examples 22 and 24, the mini-app and the sampler Monte Carlo slow.

### C3. Monte Carlo larger than its tolerance needs

All of these are seeded.

- `test_mixture.py:54` (slow): 300,000 draws at 16–36 SE. Use the exact law of total covariance plus 20k draws.
- `test_probability.py`: `:1181` (slow; 60k × 4 at about 14 SE); `SAMPLES = 30000` at `:20` (11 calls at about 10 SE); `:925` (40k).
- `test_inference.py:756` (slow; 40k). It earns its place, since the sampler is approximate by construction; 10k still gives about 4 SE.
- `test_plotting.py:228` (slow): 60k × 3 metrics, although the metric does not matter to the sampled route.
- `test_spaces.py:148-185`: about 240k white-noise draws. The negative controls spend 20k draws to detect a 4–9× error.
- `test_backends.py`: `:442` (20k at about 20 SE); `:417` (20k MFEM assemblies, its docstring says "Slow" but it is unmarked); `:480-554` (slow). The Matérn variance and correlation there can be computed from the covariance operator, deterministically.
- `test_interval.py:61`, `:406`; `test_radial.py:48`, `:260`: the default 20k draws, each a synthesis.
- `test_assorted.py:1051-1193` (`TestHardening`): about 50k draws; `:1182-1193` duplicates `:1163`.

### C4. Performance assertions rather than correctness

- `test_solvers.py:1027`: `tracemalloc` growth at n = 2000. It duplicates the refusal at :1049, and calls `start()` without try/finally.
- `test_optimization.py:641-665`; `test_convex.py:491-533`; `test_point.py:715`.
- `test_solvers.py:1085-1137` (dense n = 400).
- `test_normal.py:527-688`: Sobolev(32), 150 data, CG up to 4000 iterations. It is the only test of `InvariantDistancePreconditioner`; lmax 16 with 60 points would do.
- `test_functional_calculus.py:687` (dims 300 and 1000) and `:957` (a dense slogdet at 600 to test a constant).

### C5. Duplicates

- **Feasible-set parity is repeated across two files.** Keep `test_feasible` for the algorithms and `test_backus` for the estimator layer.
  - primal vs dual on balls: `test_feasible.py:490`, `:764`; `test_backus.py:1546`
  - quartic ≈ ball: `test_feasible.py:783`, `:878`; `test_backus.py:1000`, `:1407`, `:1445`
  - emptiness: `test_feasible.py:123`, `:583`, `:810`; `test_backus.py:156`, `:1241`, `:1587`
  - sweep vs singly: `test_feasible.py:700`; `test_backus.py:896`, `:1845`
- `test_backus.py:1845` duplicates `:1834`: `estimator(data)` builds a fresh set each call (`backus.py:394-400`). `:1733` follows from `:1689`.
- `test_spectral.py:359` duplicates `:330`; `test_functional_calculus.py:649` duplicates `:629` and `:963`.
- `test_space_parity.py` is 55 tests × 6 geometries. `:244`, `:456`, `:507`, `:677` and `:904` are geometry-independent; `:993-1012` parametrizes 11 names one by one.
- Further pairs: `test_interval.py:473` = `:305`; `test_observation.py:453` = `:683`; `test_assorted.py:634-648` vs `:1889-1898`.

### C6. What earns its cost

- Examples 22 and 24.
- The shooting-route parity, made smaller.
- `test_quadratic_programming.py:202`, `test_backus.py:1689`, `test_spectral.py:330`, `test_functional_calculus.py:629`, `test_inference.py:756`.
- The point-data mini-app.
- All of `test_no_array_backend.py`: cheap, and the strongest metric and coordinate-freedom check in the suite.

## D. Coverage gaps

### D1. Public names never mentioned in any test

A scripted check of each module's `__all__` against the test sources:

- `numerics`: `FactoredNormalOperator`, `BoundFamily`, `DirectSolver`, `LineSearch`, `LineSearchResult`, `RootResult`, `Formalism`, `misfit_search`
- `inference`: `Estimator`, `GaussianEstimator`, `SetEstimator`
- `geometry`: `Algorithm`, `EllipsoidResult`, `SweepResult`, `Complement`
- `spectral_element`: `AxisSpace`, `SpectralElementSpace`, `Boundary`, `LengthScale`
- `compat`: `AdaptedSpace`, `AdaptedOperator`
- `algebra`: `resolve_rng`, `OrthonormalSpace`, `REORTHOGONALIZATION_THRESHOLD`

Most are bases or result types. Those with behaviour of their own and no direct test:

- `FactoredNormalOperator`, which the structure-aware preconditioners require (`preconditioners.py:1072`, `:1416`).
- `misfit_search`, reached only through `MinimumNorm` (`point.py:351`).
- `AxisSpace.components_of` and `vectors_from` (`axis.py:214-226`), batched paths never compared with the per-vector ones.

The reviewers found these methods never referenced:

- `LinearOperator.from_tensor_product`, `adjoints_are_linked`
- `HilbertSpace.orthogonalize_against`; `CoordinateSpace.orthonormalize_columns`; `BlockOperator.blocks`
- `PeriodicBox.with_shape`, `Sphere.domain_mask`
- `GaussianMeasure.has_zero_expectation`, `GaussianMixture.with_weights`
- `ConvexSet.intersect`, `Polytope.excesses`, `slacks`, `linear_certificate_operator`
- `mfem.restrict_vector`, `MfemDofSpace.constrain`

### D2. Behaviours

- **`strict=`.** Raising, and the `strict=False` downgrade, are tested only for CG and LSQR (`test_solvers.py:311-336`), not for MinRes, GMRES, BiCGStab or FlexibleCG. No test checks that an estimator surfaces a strict solver's ConvergenceError.
- **Trait refusals.** Only CG and LU are tested. Cholesky, Eigen, MinRes and FCG (`solvers.py:727`, `:739`, `:1439`, `:1639`) are not. GMRES and FCG are absent from `SQUARE_SOLVERS` (`test_solvers.py:53`); GMRES meets an exact solve only in `test_no_array_backend.py:429-434`.
- **`unsafe=`.** Refusal and proceed are tested for the density family (`test_probability.py:1579-1627`), path integrals (`test_observation.py`, `TestPathGuard`) and point evaluation on the spectral-element spaces. They are not pinned for `kl_divergence`, `ambient_ball`, or `log_evidence` on a function-space model (`problem.py:194-199` passes `unsafe=True` internally, unchecked).
- **`check_*`.** All 13 are used, with negative controls (e.g. `test_operators.py:221`, `:1118`; `test_nonlinear.py:83`, `:422`; `test_probability.py:299`; `test_geometry.py:235`; `test_spaces.py:158`). `measure=` on function spaces appears only at `test_operators.py:1276-1345` and `:1430-1454`. Every other function-space check uses white probes (e.g. `test_flexure.py:46`, `:306`, `:529`; `test_assorted.py:518`, `:1846`).
- **Feasible-set certificates.** Brackets are compared algorithm-to-algorithm, except `test_no_array_backend.py:620-664` (primal vs SLSQP). **No test checks that a non-converged Primal, Dual or Splitting run still returns a valid bracket.** That is the certified-outer-bound guarantee. `test_feasible.py:693-696` compares a loose and a tight run on value and gap only.
- **Rays.**
  - The derivative's adjoint is never checked (no `check_operator` on `operator.at(m).derivative`), although Gauss–Newton and MAP use it.
  - The torus shooting Hessian and the circle Hessian are missing.
  - The refusals are untested: unknown method (`rays.py:1012`) and empty paths (`:1136`).
- **Posterior push-forward.** Degrading to a plain Gaussian is checked only through `translate` (`test_absolutely_continuous.py:138-146`).
- **Convex and QP.** Untested:
  - SubgradientDescent's `"inverse"` and `"constant"` rules
  - ProximalGradient with a fixed step, and its backtracking failure
  - Chambolle–Pock's refusals and its non-convergence
  - infeasible QPs, and the `best_available_qp_solver` fallback
- **Spectral element.** The order > 0 inner product is checked against planetmodel only on Interval (`test_interval.py:118-125`). `spectral_operator`, `spectral_measure` and `sqrt` are untested on these spaces.
- **Compat.** It is tested only on a diagonal Gram (`test_compat.py:82`). `gram="dense"` and its RuntimeWarning (`compat.py:218-237`) are untested.

### D3. Metric-sensitive code tested only on Euclidean or diagonal spaces

- **`test_normal.py`, the whole file.** The `setup` fixture (:58-71) drives the normal operator in both formalisms, NormalDiagonal vs Jacobi ("agree or the metric handling is wrong"), Localized, Woodbury and the damped families. The module docstring (:14-17) argues that weighted is enough, and `:1093` claims a non-diagonal metric while using `make_weighted_space`.
- **`test_convex.py`, the whole file.**
  - SquaredDistance gradient and Hessian (:43-61, "whatever the metric")
  - the prox maps (:76-153)
  - the SupportFunction adjoint image (:218-226)
  - all the optimizers (:236-382)
- **`test_feasible.py` and `test_backus.py`.**
  - The `setting` fixture (`test_feasible.py:51`, duplicated at `test_backus.py:41`) has a WeightedSpace model and drives every algorithm-parity test.
  - **The data space is Euclidean in every test of both files**, so R⁻¹ and the dual's metric gradient never meet a metric.
  - Only `test_backus.py:340`, `:541` and `:651` are dense.
- **`test_mixture.py`, the whole file.**
- **`test_point.py:288`, the `setup` fixture.** It drives Tikhonov, LeastSquares, the MinimumNorm derivative (:755) and adjoint (:783), and the family-aware preconditioners. `test_no_array_backend.py:512-575` partly offsets it on `ListSpace`.
- **`test_nonlinear.py`.** `TestFunctional` checks gradient vs derivative, the rule's own example. `TestLinearFunctional`, `TestNonlinearOperator` and `TestAffineOperator` are also Euclidean.
- **Probability and inference.**
  - `test_probability.py`: `TestDensities`, `TestAlgebra`, `TestPushForward`, `TestConditioning`
  - `test_inference.py`: `TestEvidenceWithoutAssembling`, `TestPosteriorSampling`
- **Randomized, functional calculus and spectral.**
  - `test_randomized.py`: `TestRangeFinding`, `TestFactorizations`
  - `test_functional_calculus.py`: `TestLanczos`, `TestOperatorFunction`
  - `test_spectral.py:696`, whose docstring says non-diagonal.
- **`test_geometry.py`.** `TestThreeViewsOfOneSet` (:515-560), `TestBoundaries` (:696-748), `TestEllipsoid` (:563-693; the docstring at :612 claims non-diagonal), and the intersection projection (:1299).
- **Operators, direct sums, compat and assorted.**
  - `test_direct_sum.py:172-456`
  - `test_operators.py:442-605` (the domain is always weighted) and `:1144-1197` (gradient == 2x on Euclidean)
  - `test_coordinate_free.py`, the whole file: OpaqueSpace and `StrictSpace(make_weighted_space())` are diagonal
  - the compat adapters
  - `test_assorted.py`: Woodbury (:1278-1294), column-thresholded (:1534-1541), subspaces (:412-497)
- **Already covered on a non-diagonal Gram.** `test_no_array_backend.py:419-441` checks CG, MINRES, GMRES and LSQR against exact solutions on `ListSpace`, and `:785` runs all five optimizers there.
- **Diagonal by construction.** Symmetric-space and spectral-element spaces cannot have a dense Gram, so those files are not breaches; the MFEM backends satisfy the rule.

### D4. Rules violated

- **Decision citations: none.** `test_code_practice.py:405-447` enforces this over every non-`.md` file, tests included.
- **v1, history and planning narrative is not enforced and is widespread.** About 107 lines outside `test_compat` mention v1 or v2, across 25 files, plus about 58 lines with "used to", "no longer", "the review", "the audit", "the port", "renamed".
  - **Test and class names:**
    - `test_spaces.py:150`, `:178`, `:153` (`V1StyleSpace`)
    - `test_space_parity.py:916` (`test_it_gives_v1s_answer`)
    - `test_direct_sum.py:402`; `test_fourier_spaces.py:443`; `test_plotting.py:635`
    - `test_probability.py:293` (`V1StyleMeasure`)
    - `test_normal.py:1093` (`…it_always_was`); `test_known_structure.py:528` (`…keeps_the_old_route`); `test_functional_calculus.py:957`
    - `test_compat.py:86` (`TestWhiteNoiseIsFixed`)
  - **Planning references:**
    - `test_inference.py:799-801` ("The review asked"); `test_optimization.py:669` ("The review found"); `test_parallel_knobs.py:1-8` ("The audit")
    - `test_probability.py:691-699` ("Bug 3.9")
    - `test_operators.py:147` and `test_compat.py:4`, `:307`, `:407` ("M1"–"M4 acceptance")
    - `test_inference.py:66` and `test_assorted.py:84` ("section 5.6")
    - `test_spaces.py:543` ("David: …"); `test_laplace.py:565-568`
  - **Shared file:** `conftest.py:30-33`, `:76-79`, `:142-143`.
  - **Dead shim:** `test_minimization.py:29-32`, a try/except for "the renamed class" `DiscrepancyPrinciple`, which no longer exists in the package (verified).
  - **Stale path:** `test_miniapp_point_data.py:4` still says `miniapps/oceans`.

## E. Quality notes

### Vacuous counter bounds

- `test_randomized.py:345-372` bounds `EuclideanSpace.inner_product` calls by 60,000. **On an equivalent problem (dim 600, rank 120) I measured 0 calls**, because `random_range` takes the component route on Euclidean (`randomized.py:331`).
- `test_functional_calculus.py:774-798` has the same flaw (Lanczos runs on components).
- `test_assorted.py:121-126` and `:888-899` pass `probe="banded"` to `from_matrix` operators, which ignore it (`operators.py:1130-1133`).
- *Fix:* run on `StrictSpace` or `ListSpace` and also assert the count is > 0.

### Trivially true, tautological or unasserted

- `test_solvers.py:288` (`allclose(x is not None, True)`).
- `test_normal.py:490` and `test_probability.py:472` (`norm >= 0`).
- `test_mixture.py:322` (log Σ ≤ max).
- `test_operators.py:997-1003` (no assertion).
- `test_bounded_and_sphere.py:157-165` (a call compared with itself, `box.py:391-392`).
- `test_interval.py:81`, `test_backus.py:112`, `:758`, `test_geometry.py:748`.

### Wrong construction

- `test_operators.py:130-134`: `OpaqueSpace(4)` builds a 1-dimensional space, under a `raises(TypeError)` with no `match=`.
- `test_operators.py:54-62` declares `SELF_ADJOINT` falsely on a weighted space.
- `test_convex.py:129-131` and `test_geometry.py:62`, `:484`, `:533` rescale by the norm of a different draw.

### Names that don't match what is checked

`test_spectral.py:684` (`and_nests`), `:353`; `test_interval.py:409`; `test_probability.py:1474`; `test_point.py:520`; `test_feasible.py:824`; `test_assorted.py:1734`; `test_normal.py:171`, `:447`; `test_quadratic_programming.py:204` (sixteen directions vs eight).

### Helpers to centralise in conftest or doubles

- **SPD builders** in at least 10 files with 3 signatures: `test_solvers.py:37`, `test_functional_calculus.py:37`, `test_probability.py:23`, `test_escape_hatches.py:31`, `test_parallel_knobs.py:42`, `test_known_structure.py:50`, `test_operators.py:23`, `test_assorted.py:39`, `test_normal.py:53`, `test_inference.py:612`, `test_no_array_backend.py:106`.
- **Inline dense metrics** that duplicate `make_dense_metric_space`: `test_optimization.py:549`, `test_randomized.py:317`, `test_solvers.py:933`, `test_normal.py:681`, `test_functional_calculus.py:987`, `test_assorted.py:1710`.
- **Feasible-set fixtures** `setting`, `directions`, `quartic_ball`: `test_feasible.py:51`, `:28`, `:742` vs `test_backus.py:41`, `:61`, `:69`.
- **Skip guards:** the planetmodel gate (×5) and the coastline check.
- **Cross-file import:** `test_minimization.py:441` imports `ray_problem` from `test_laplace`.
- **Counters:** about 10 hand-rolled, while `doubles.CallCounter` is used only inside `doubles.py`.
- **Fixture shadowing:** `test_spectral_element_plotting.py:26-29` redefines the autouse `_close_figures`.
- **Runners:** `test_examples.py` and `test_tutorials.py` are near-identical.

### Hand-rolled monkeypatching (use the fixture)

- `test_point.py:254`, `:1145`, `:1211`; `test_feasible.py:314`; `test_direct_sum.py:517`.
- `test_operators.py:777`, `:880`, `:971`, `:1247`.
- `test_randomized.py:353` (patches `EuclideanSpace.inner_product` class-wide), `:622`.
- `test_functional_calculus.py:706`, `:787`, `:822`.

### Leaked global state

- `test_miniapp_point_data.py:46-50` registers plain module names (`problem`, `sets`, `point`) in `sys.modules`.
- `test_miniapp_cpp_backend.py:73` keeps its `sys.path` entry, and `:205` depends on calls made by earlier tests.

### Dead code

`test_minimization.py:29-32`; `test_assorted.py:1207-1211` (`if False else`); `test_spaces.py:310`, `:328`; `test_operators.py:773-774`; many redundant local re-imports.

### Split `test_assorted.py`

It is 1898 lines and growing:

- solvers: GMRES, FlexibleCG, diagnostics
- a new `test_preconditioners.py`: preconditioners, Woodbury, column-thresholded, sparse approximation
- operators: sparse, coordinate, diagonals
- probability: adjustments, hardening, conditioning, quadratic forms
- geometry: subspaces, surfaces
- randomized: deflated diagonal

## F. Recommended actions

| # | action | effort |
|---|---|---|
| 1 | Fix the tolerance-below-the-floor family: B2 (about 25 optimizer sites), B6 (about 12 strict-CG sites), B8 (default tolerances, a fixed `rng`, bracket overlap instead of `converged`). | 3 h |
| 2 | Remove assertions about rounding-produced behaviour: B3, B7, B9 (`test_convex.py:459`, `:491-533`), B11. | 1 h |
| 3 | Make the C++ backend's draws platform-independent (B4). | 30 min + rebuild |
| 4 | `delenv` the thread variables and drop the joblib class-name asserts (B5). Move the coastline guard into conftest (B14). | 30 min |
| 5 | Add a scheduled CI job running `-m slow` on all three OSes. Rebalance markers: unmark the 15 sub-second slow tests after shrinking them; tutorials 07/13/14 to a quick scale or slow; halve the ray grids; cut the `gc.collect` loops to 10 or fewer; shrink the KDE tests. Expected default-run saving is about 200 of 485 s. | 2 h |
| 6 | Scale the two-route comparisons (B13, about 15 sites) and the parallel-vs-serial ones (B10). | 45 min |
| 7 | Shrink the C3 Monte Carlo; make the MFEM Matérn and mixture checks deterministic; delete the C4 performance tests and the C5 duplicates. | 2–3 h |
| 8 | Metric rule: add a dense-metric parameter, and a non-Euclidean data space, to the fixtures of `test_normal`, `test_convex`, `test_feasible`/`test_backus`, `test_mixture`, `test_point.py:288` and `test_nonlinear::TestFunctional`. Budget separately for any bugs this finds. | 3–4 h |
| 9 | Fill the D2 gaps: strict and refusal tests for the remaining solvers; bracket validity on non-converged runs; the ray derivative's adjoint; `FactoredNormalOperator`; `unsafe=` on `kl_divergence`/`ambient_ball`/`log_evidence`; `measure=` on function-space checks. | 3 h |
| 10 | Fix the vacuous, trivial, misnamed and wrongly constructed tests in §E. | 1 h |
| 11 | Centralise the helpers and switch to the `monkeypatch` fixture. | 2 h |
| 12 | Strip the v1, history and planning narrative and rename the v1-named tests (§D4). Optionally make `test_code_practice` flag "v1", "used to", "the review" and "the audit" in tests. | 2–3 h |
| 13 | Split `test_assorted.py`. | 1.5 h |
