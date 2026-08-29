# pygeoinf2 — second review (2026-08-29)

The first review (`REVIEW.md`, 2026-08-27) was answered by 63 commits. This
is the check on that work, with a lens the user asked for this time:
**optimisation**. The parallelism question was taken separately
(`review/parallel.md`, appendix R) and is not repeated here.

Five area appendices carry the detail — method-by-method status tables,
reproductions, and measurements with the scripts that produced them:

| appendix | file | scripts |
|---|---|---|
| R2-A | `review/r2_algebra_solvers.md` — spaces, operators, nodes, direct sums, solvers, preconditioners | `work/review_r2/AS/` |
| R2-N | `review/r2_numerics.md` — randomised methods, Lanczos, optimisation, convex methods, QP, root finding; `review/r2_numerics_recipes.md` — verified prototype code for the component path, the dualised level master, the bundle subproblem, `monotone_root`, and what 3.1 needs | `work/review_r2/N/` |
| R2-P | `review/r2_probability_inference.md` — measures, Gaussians, mixtures, estimators, Backus | `work/review_r2/PI/` |
| R2-Y | `review/r2_symmetric_space_geometry_plotting.md` — sphere, boxes, observation operators, sets, plotting | `work/review_r2/YG/` |
| R2-K | `review/r2_packaging_tests_docs.md` — exports, suite, examples, MFEM, catalogue, pyslfp | `work/review_r2/K/` |

Each was produced by a separate reviewer working from the same brief
(`work/review_r2/BRIEF.md`): verify against the code, never the commit
message; apply the metric rule; profile at realistic sizes. All timings were
taken with `OMP_NUM_THREADS=1` on a 16-thread laptop that throttles under
load and with five reviewers profiling at once, so **ratios are reliable,
absolute numbers are ±30 %**. The findings I quote below I have checked to the
extent of reading the cited code; the numbers are the reviewers'.

> **Status (2026-08-29, branch `refactor-r2`):** the fixture (3.16) and §4.1 a–c
> are on `refactor` (`26baa84`, `06d651c`, with `b692230`, `4bf99f5` for the
> small numerics items and `monotone_root`). On `refactor-r2`, one commit each:
> `review/parallel.md` R1–R6 (`542de76`); §4.4 with `apply_block`, D-6's
> remaining `n_jobs` hooks, 3.5 and 3.14's `_rebuild` half (`6620637`); §4.1 e
> generalised as component-native products and sums (`63ddc8a`); §4.1 d, the
> Krylov loops on a `ComponentView` (`849d7c0`); 3.2 and §4.2.2, the flexure
> operator fused and its inverse taken in L² (`cb70beb`); §4.2.9 with P Must-3's
> precision (`8511651`). Each commit message carries its before/after numbers.
> Still open from this list: the rest of §3 (one-to-five-line bugs), §4.2
> except 4.2.2/4.2.9, §4.3, §4.5, and the questions in §5.

## 0. Verdict

1. **The first review's plan was largely executed, and executed well.** The
   full default suite is 2023 passed / 0 failed in 87 s. Every Must item in
   the symmetric-space, numerics (bar one) and packaging areas is done; all
   four pyslfp port blockers are gone and verified by running pyslfp's
   shapes; the metric rule holds on every closed-form check run this round
   (`as_multivariate_normal`, KL, conditioning, credible sets, forms,
   diagonals, lifts, LU adjoints, Jacobi — all ≤1e-12 on a dense Gram).
   The formal-adjoint lift, which the first review flagged at four transforms
   per application, now costs zero on the forward and is *faster* than v1 on
   the adjoint.
2. **One structural cost dominates everything else, and four reviewers found
   it independently: an inner product on a spectral space transforms both
   arguments.** `CoordinateSpace.inner_product` and
   `DiagonalMetricSpace.inner_product` (`algebra/spaces.py:409, 772`) call
   `to_components` on `x` and on `y` — a spherical-harmonic analysis each,
   1.4 / 11 / 73 ms at lmax 64 / 128 / 256. Every routine that
   orthogonalises — `orthonormal_basis`, `gram_schmidt`, Lanczos, `random_range`
   and so `random_eig`, `SpectralPreconditioner`, `low_rank_surrogate`,
   `from_vectors.adjoint`, the bundle Gram — does O(k²) of them on vectors
   that never change. `random_range(rank=50)` at lmax 128 spends 17 916
   transforms and 54 s where ~300 transforms are needed; `random_eig(rank=50)`
   on a 3000-dim dense-metric space spends 40.7 of 42 s in Gram applications.
   Prototypes doing the same arithmetic on component arrays give **3–4× on
   Lanczos, 17–25× on orthogonalisation, 23–30× on `random_eig`**, identical
   to 1e-14. This is the one design decision this review needs from you
   (§5, Q1).
3. **A second theme: the sphere pays transforms it does not owe.** `plot` is
   6× *slower* than v1 (a 0–360° seam sends Cartopy down a per-polygon path;
   71× available), the flexural operator costs 50 transforms where v1 cost 24
   and six would do, an invariant-measure draw costs 3 where v1 cost 1, exact
   cap averages and `covariance_function` have closed forms 40–150× cheaper
   than the representer route they take, and the dense path operator
   multiplies a densified sparse matrix (90×). None of these is hard.
4. **Bugs: 16 verified, two serious.** `DualFeasibleProperty` reports a
   support value 2× the truth as converged on a tight-noise over-determined
   problem, and the other routes disagree with each other there (Mag's code,
   D-13 — needs his view); and `inverse_flexural_operator` on a Sobolev space
   with varying coefficients asserts a false self-adjointness and CG either
   fails or silently returns a 1e-4 residual (inherited from v1; v2 now
   claims the trait). The rest are one-to-five-line fixes (§3).
5. **The metric-rule fixture is broken above dim ≈ 300.**
   `make_dense_metric_space(dim)` has condition number 8e3 at 60, 5e10 at 200
   and is not positive definite at 500. Every dense-metric test at realistic
   size is testing roundoff. Scaling the off-diagonal by `1/√dim` fixes it
   (cond ≈ 3 at 3000). This should be the first commit of the next round,
   since everything in §4 needs to be tested on it.

## 1. Status of the first review's plan

Counts are of the appendix Must/Should/Consider items plus the §10 phase
items, as tabulated in each R2 appendix.

| area | done | partial / done differently | not done | notable gaps |
|---|---|---|---|---|
| A + S | 22 | 5 | 20 | A Must-4 second half (Diagonal calculus on a non-diagonal metric; `_rebuild` drops traits); `assembled()` still Galerkin (3.4× dearer CG on a dense metric); no `matrix()`/`diagonals()` fast paths on nodes; S Should-12 done *wrong* (§3.5) |
| N | 19 | 5 | 8 | N Must-4 component/QR path (= §0.2); `breakdown_tol`; callbacks; `test_convex` has no non-diagonal-metric test |
| P + I | 25 | 11 | 9 | P Must-3 precision on `correlated_measure`; P Must-7 stochastic KL still wrong; P Should-9/10/11/12 dense-route regressions untouched; I Must-7 leftover (`**kwargs` → `TypeError`) |
| Y + G | 41 | 3 | 12 | all Y Must done; G S5/S6, `plot_points` dispatch, `Ball/Ellipsoid.boundary`, three false catalogue rows |
| K + U | 20 | 9 | 9 | Sphinx still documents v1 only; export holes; no least-squares example; U 8 (`check_operator` with test measures), U 9 (`plot_points(data=)`) |

Decisions D-1 … D-13: all implemented as decided, with two slips against
their stated reasons — `with_order` returns a bare `Sphere`, so D-3's
"`isinstance` works" fails on every derived space; and D-6's `n_jobs` reached
about half its list (`review/parallel.md` R7).

## 2. Where the time goes — a map

Reading the five profiles together, the package has four cost regimes, and
the optimisations in §4 group by them:

1. **Transforms spent on vector algebra rather than on operators.** Inner
   products, norms, orthogonalisation, `from_vectors`, block operators
   starting from `zero()`, `sample` synthesising white noise only to analyse
   it again. On the sphere this is 60–95 % of every Krylov and randomised
   routine. Fix once, in `CoordinateSpace` (§4.1).
2. **Sphere routines that take the general route where a closed form exists**
   — cap averages, covariance functions, pointwise variance of a diagonal
   covariance, the south-pole value in `evaluate`, the flexural operator's
   `laplacian(multiply(·))` pairs (§4.2).
3. **Inference bookkeeping** — the same solve done twice, a matrix assembled
   twice, a factorisation thrown away, a default (`dense_limit=512`) that
   routes a 960-dim evidence to a ±30-nat estimate that is also slower than
   the exact answer (§4.3).
4. **Missing fast paths on the operator algebra** — `(A+B).matrix()`,
   `A.adjoint.matrix()`, `Cholesky(M + tI)`, `diagonals` of a sum: 1000×
   between the general route and reading the stored matrix (§4.4).

The Euclidean core is not the problem: CG on a 2000-dim dense operator is at
parity with SciPy (16.0 vs 16.1 ms), node overhead is 0.1–0.2 µs per factor,
direct-sum overhead ≈ 2 µs per block per operation.

## 3. Bugs and regressions, ranked

Verified by the reviewers with reproductions in the appendices; the file
lines I have read myself.

| # | what | where | severity | fix size | v1? |
|---|---|---|---|---|---|
| 3.1 | `ProximalBundleMethod` reports `converged` at **2× the true support value** on a tight-noise over-determined problem (300 model, 2000 data); `DualFeasibleProperty` returns it. `kkt` gives a value *below* a feasible one, `primal` stops unconverged at its cap without flagging, `smoothed` 0.32 vs truth 0.18. All five routes agree only when the noise set is loose — which is the only case the tests and the CURRENT_STATE "1.7e-8 agreement" table cover. | `numerics/convex.py:866-876`, `inference/backus.py:1026`; R2-N B1 | **high** — wrong answer, no warning | design; Mag's | n/a |
| 3.2 | `inverse_flexural_operator` **DONE** (`cb70beb`). on a Sobolev space with varying rigidity lifts the L2 operator and claims `POSITIVE_DEFINITE`; it is not self-adjoint in `H^s` (`(Fx,y)=238`, `(x,Fy)=−1.2`). CG raises on the sphere and **silently returns a 1e-4 residual on the circle**. Only Lebesgue spaces are tested; examples use L2 or constant rigidity. | `symmetric_space/base.py:1756`; R2-Y 2.1 | **high** | one line (lift the L2 *inverse*), verified | yes, same defect |
| 3.3 | `DualFeasibleProperty.dual_cost` memo keyed on `id(certificate)`: a freed-and-reallocated array returns the previous certificate's residual — wrong gradient in 145/200 trials. Latent inside the bundle (which holds references), live for any other `method=`. | `inference/backus.py:1064-1072`; R2-P B1 | medium (latent) | one line (hold the reference) | no |
| 3.4 | `Sphere.walk_from` past the antipode returns latitude −250°; direct and NUFFT evaluation then disagree by O(1). `to_colatitude_radians` documents a range check it never performs. | `sphere.py:965-978, 1003-1018`; R2-Y 2.2/2.4 | medium | small | — |
| 3.5 | `LUSolver` factorises **twice** **DONE** (`6620637`). at construction (`_factorise` and `_factorise_transposed` each call `lu_factor`): O(n³) doubled, 91 vs 47 ms at 1500. Introduced by the fix for S Should-12; the test counts applications, not factorisations. | `numerics/solvers.py:398, 409, 461`; R2-A bug 1 | medium (regression) | trivial | v1 factorised once |
| 3.6 | `from_formal_adjoint` over a `MassWeightedSpace` raises when the base is an equal-but-distinct space (`shares_vectors_with` uses `is`); `EuclideanSpace(n)` is minted freely. | `algebra/spaces.py:683`; R2-A bug 2 | medium | one token (`==`) | — |
| 3.7 | `with_order`/`with_degree` return a bare `Sphere`: `isinstance(x.with_order(0), Lebesgue)` is false, defeating D-3's stated reason and every such check in pyslfp's `sl/utils.py`. | `sphere.py:1573-1594`; R2-K bug 1 | medium (pyslfp) | small | — |
| 3.8 | `covariance_function` raises on every Lebesgue space via the D-11 guard; v1's closed form worked on L2. The function is a property of the measure's spectrum, not of the space's order (see §4.2.4). | `symmetric_space/base.py:956-967`; R2-Y 2.3 | medium (regression) | with §4.2.4 | v1 worked |
| 3.9 | Stochastic KL still returns 369 ± 35 and −351 ± 33 for KL(μ‖μ) at dim 289 — opt-in now, but confidently wrong. P Must-7's fix list was not applied. | `probability/gaussian.py:754-793`; R2-P B3 | medium | fix or delete | — |
| 3.10 | `hilbert_schmidt_norm`/`nuclear_norm(method="stochastic")` silently form the dense matrix; docstring promises `random_trace`. | `gaussian.py:494-526`; R2-P B2 | low | small | v1 had it |
| 3.11 | `ConstrainedLeastSquares/MinimumNorm.parameterised(P, **kwargs)` forward to a method taking none → `TypeError`. | `inference/point.py:862, 1025`; R2-P B4 | low | trivial | — |
| 3.12 | `from_covariance_matrix` refuses PSD-singular input and attaches no precision; v1 took `eigh`, clipped, attached the inverse factor. Not recorded as a decision. | `gaussian.py:380`; R2-P B5 | low (regression) | small | v1 accepted |
| 3.13 | `BandedPreconditioner`/`BlockPreconditioner` claim `SELF_ADJOINT` on any operator, including `form="components"` of a non-self-adjoint one. | `numerics/preconditioners.py:249, 402`; R2-A bug 5 | low | small | — |
| 3.14 | `DiagonalLinearOperator` on a non-diagonal metric refuses **DONE** (`6620637`, the `_rebuild`/traits half; `sqrt`/`log_determinant` gating still open). `sqrt`/`log_determinant`; `_rebuild` drops caller traits (`(2·D_sa).traits == NONE`). A Must-4, still open. | `algebra/diagonal.py:126-128, 199-219`; R2-A bug 4 | low | small | — |
| 3.15 | `LevelBundleMethod`: misreports `iterations` **DONE** (`b692230`, the `iterations` misreport only; the rest is Mag's). after an early break; fails on iteration 1 with OSQP where Clarabel solves it; does not converge where the proximal method does and is 10–60× slower; its master problem is dense in the **data dimension** (401 MB per call at 5000 data), not "in the number of cuts" as CURRENT_STATE:99 says. | `convex.py:1319-1369, 1443-1456`; R2-N B2/B3/B6, O4 | medium; Mag's | see §4.5 | v1 also dense |
| 3.16 | `make_dense_metric_space(dim)` not PD **DONE** (`26baa84`). above dim ≈ 300 (cond 5.7e13 at 300). | `tests/conftest.py:83-103`; R2-A note, R2-N B4 | **high for the test strategy** | scale off-diagonal by `1/√dim` | — |

Smaller, all verified: `from_grid_values` aliases the caller's array
(`sphere.py:370`); box `pointwise_variance` "same everywhere" is 2 % off on
even axes; `plot_points` cannot colour by data (`c=` collides); `Functional.__init__`
still has a positional-optional argument; `Linearisation`/`QuadraticModel`
`hash()` raises; export holes (`MassWeightedSpace`, `HilbertModule`,
`weighted_chi2_*`, `resolve_solver`, `FactoredNormalOperator`, mfem's
`matern_measure`); false docstrings — Jacobi's "sums of matrix-backed operators
give up their diagonal for free" (measured: 2000 applications), `assembled()`'s
"metric enters once", `_to_mfem`'s reason for its loop (the CSR constructor is
bound; it double-frees), `FeasibleProperty.solver`; false catalogue rows
(`deflated_pointwise_variance`, `LevelSet`, `Cut`/`Bundle`, `best_available_qp_solver`
"Planned", `from_vectors`, `MassWeightedHilbertSpace`); Sphinx still builds
v1 only; import time grew 0.32 → 0.43 s from eager `scipy.stats`/`integrate`.

## 4. Optimisations, consolidated and ranked

Gain × confidence, deduplicated across appendices. "Measured" means a
prototype ran and agreed with the current code to the stated precision.

### 4.1 Component-space arithmetic on `CoordinateSpace` (the big one)

| item | now | prototype | where | ref |
|---|---|---|---|---|
| a. `orthonormal_basis`/`gram_schmidt`/`_orthogonalise_against` in components **DONE** (`06d651c`). (Cholesky-QR or MGS on `(dim,k)` arrays with `apply_gram`) | 50 vectors at lmax 128: 2650 transforms, 7.4 s; `random_range(50)` 54 s | 100 transforms, 0.31 s (24×); `random_range` ≈ 2 s (25×); `random_eig(50)` on dense-metric 3000: 42 → 1.5 s | `algebra/spaces.py:215-262`; consumers `numerics/randomised.py:95-98, 151-184, 388, 449` | R2-A 2, R2-N O1, R2-K 1 |
| b. Lanczos basis kept in components **DONE** (`06d651c`)., one analysis per step | `apply_operator_function` lmax 64: 1047 transforms, 60 the operator's | 152 transforms, 3.1–3.7× | `numerics/functional_calculus.py:127-142` | R2-N O1 |
| c. `from_vectors` caches components **DONE** (`06d651c`). when the codomain has them | 40 analyses per adjoint at rank 20 (`SpectralPreconditioner.apply` 145 ms) | 1 transform each way (3.6 ms) | `algebra/operators.py:1041-1042` | R2-A 3 |
| d. `norm` transforms once; CG reuses **DONE** (`849d7c0`). ‖r‖² when unpreconditioned; Krylov loop in coefficients on a `CoordinateSpace` | CG on the sphere: 7.4 analyses + 1.1 syntheses per iteration, 6.4 of them vector algebra | 7 → 4 transforms/it (43 %); full coefficient-space loop: diagonal operator 237 → 0.4 ms, grid operator 22–30 %, model-space normal 1.3× now and ≈10× once point evaluation is components-native | `spaces.py:772`, `solvers.py:750-756` | R2-A 1, R2-K 6 |
| e. `NormalOperator` fused in components **DONE** (`63ddc8a`, generalised to every product and sum of component-native operators). when `A` is matrix-backed and `Q` diagonal: `A_c(λ ⊙ G⁻¹A_cᵀv) + σ²v` | 4 transforms per application (ex21 lmax 48: 2.7 ms) | 0 transforms (1.2 ms, 2.3×); a CG solve 0.22 → 0.09 s | `inference/normal.py:238` | R2-P 1 |
| f. `apply_block(vectors)` hook **DONE** (`6620637`)., overridden by matrix/diagonal/`from_vectors`/sums/compositions — also where D-6's `n_jobs` for `random_range` belongs | `random_eig(50)` Euclidean 3000: 240 matvecs, 0.72 s | 4 GEMMs, 0.16 s (4.6×) | `randomised.py:80, 95, 98, 386` | R2-N O2 |
| g. Lebesgue inner products by DH quadrature: 0 transforms | 2 per inner product | 0 | `spaces.py:772`, sphere | R2-K 6 |
| h. Block operators skip `_Zero` and start from a copy, not `zero()` (a synthesis on the sphere) | `[[I,0],[I,I]]` on two spheres: 3 syntheses per application | 0 | `algebra/direct_sum.py:416-424, 537-542, 585-590` | R2-A 8 |

Risk is low throughout: the metric enters only through `apply_gram`, the
coordinate-free code stays as the fallback for `OpaqueSpace`, and the results
were checked against the current code on three space types. The precondition
is 3.16, so that the tests can run on a dense Gram at these sizes.

### 4.2 Sphere routines with a cheaper exact route

| item | now | prototype | ref |
|---|---|---|---|
| 1. `plot`: roll the mesh to [−180, 180) with flat cell edges | 1481 ms at lmax 128 (v1 232 ms) | 20 ms (71×), image identical | R2-Y 3.1 |
| 2. `flexural_operator` summing grid terms **DONE** (`cb70beb`). per spectral multiplier | 50/54 transforms fwd/adj (v1 24/28), 166 ms | ~6 transforms, est. ~25 ms (~8×); every PCG iteration of the inverse shrinks with it | R2-Y 3.2 |
| 3. Invariant-measure `sample` via components: `from_components(√(s/g)·z)` | 1 analysis + 2 syntheses (10.3 ms) | 1 synthesis (3.6 ms, 2.8×); `_rebuild` already carries `sample=` | R2-Y 3.3 |
| 4. `covariance_function` and `pointwise_variance_at` for a `DiagonalLinearOperator` covariance by Legendre closed form / `basis_matrix` | 23–123 ms per 50 distances; 17–116 ms per point | 0.5–1.2 ms; 0.15 ms per point (40–150×); also fixes 3.8 | R2-Y 3.6 |
| 5. Exact cap averages by closed form `2πR² I_l(cos α)·basis_matrix` | 15 ms per centre (`from_cap` rotation + an unread representer) | 0.2 ms (70×), 7.6e-13 | R2-Y 3.4 |
| 6. Dense path/ball operators: keep the weight matrix sparse; vectorise node generation, cache `leggauss` | 2000 paths: 1.8–3.3 s nodes, 1.59 s densified product | 0.30 s nodes (11×), 0.018 s product (90×) | R2-Y 3.5, R2-K 3 |
| 7. `point_evaluation_operator` converts points once, not per application | sphere 14.5 of 37 ms at 10⁵ points; Box 91 % | 1.4× sphere, 2× torus, 10× Box | R2-Y 3.7 |
| 8. `evaluate`: south-pole value from row means, not a full analysis | 35 % of a forward at lmax 256 | 0.05 ms | R2-Y 3.8 |
| 9. Correlated (block-diagonal) measures **DONE** (`8511651`).: transform each field once | 4 + 6 transforms per application, floor 2 + 2; sample 4 + 8 vs 0 + 2 | 2.5× / 6× (pyslfp's `(Dyn, Rho)` prior) | R2-K 5 |
| 10. `plot_paths` as one `LineCollection`; `plot_corner` density by binned histogram + Gaussian filter | 1294 `ax.plot` calls, 3.9 s in ex21; KDE 8.9 of 12.2 s in ex26 | est. 10–20×; 500× at 6 % contour error | R2-K 2, 7 |

Where v2 already wins, for the record: NUFFT point evaluation is 100–170×
faster than v1 at 10⁵ points; path-operator application 20–100×; the lift
adjoint 26 vs 37 ms.

### 4.3 Inference bookkeeping

| item | now | after | ref |
|---|---|---|---|
| 1. `dense_limit` 512 → ~4000 (or pick by estimated applications) | ex21 evidence, dim 960: stochastic −7270 ± 30 in 4.8 s | exact −7315 in 2.8 s | R2-P 2 |
| 2. Direct-solver `InverseOperator` exposes `log_determinant` from its factorisation | `log_evidence` after `CholeskySolver`: second full assembly, 4.3 s | free | R2-P 3 |
| 3. One residual solve shared by the mean and the misfit (v1 did `m₀ + K(d − shift)`) | 2 solves per datum; mixture inversion 2K | 1; K | R2-P 4 |
| 4. `estimator.push_forward(T)(data)` → use `posterior.push_forward(T)` (identical measure, 0 solves); fix example 21 | 1 solve per property | 0 | R2-P 5 |
| 5. `DampedSolves` with a direct solver: cache `B.matrix()`, `S.matrix()`, factorise `B + tS` | 23 factorisations, 162 + 162 applications per discrepancy sweep on a 6-datum problem | 1 + 6 + 6 | R2-P 6 |
| 6. `ambient_ball` O(n) for a diagonal covariance; randomised spectrum or sampling radius otherwise (both were in v1) | O(n³): 1.15 s at 8000, impossible at pyslfp's 10⁵ — and `harden_error` hits it on every Backus route | O(n) | R2-P 7 |
| 7. Posterior covariance as `(I − KA)Q`, not `Q − KAQ` | 3 `Q` applications per action | 2 | R2-P 8 |
| 8. `monotone_root`: Brent **DONE** (`4bf99f5`). in log t after the decade bracketing | 23–28 solves per root | 10–13 | R2-N O5 |

### 4.4 Fast paths on the operator algebra

| item | now | after | ref |
|---|---|---|---|
| 1. `_known_matrix(form)` hook **DONE** (`6620637`). on `MatrixLinearOperator`, `Diagonal`, `_Identity/_Zero`, `_Scaled`, `_Sum`, `_Composition`, `_Adjoint`, direct-solver inverses, blocks | `(A+B).matrix()` 3.5 s vs 3 ms; `A.adjoint.matrix()` 23 s vs 0.8 s on a dense metric; `Cholesky(M + tI)` 1.6 s (25 s dense metric) vs 78–88 ms | read | R2-A 4 |
| 2. `_known_diagonals` alongside it **DONE** (`6620637`, Jacobi default unchanged, docstring corrected).; Jacobi reads it | Jacobi on `M + 0.5I`: 2000 applications, 1.9 s | 0.6 ms; on `A*A` the composition cannot be read → restore `samples=20` there | R2-A 7 |
| 3. Vectorised `matrix()` post-processing **DONE** (`6620637`). (`apply_gram_matrix`/`solve_gram_matrix`) | 43–79 % of the call on a dense Gram | one product/solve | R2-A 5 |
| 4. `assembled()` stores the components form | CG 320 vs 95 ms on a dense metric (a `solve_gram` per application) | — | R2-A Must-3 row |
| 5. `MatrixLinearOperator.diagonals` via `np.diagonal` **DONE** (`6620637`).; `DiagonalLinearOperator.matrix()` override | 6.6 ms / 2·dim transforms | 0.07 ms / 0 | R2-A 10 |

### 4.5 Convex methods (Mag's code — proposals, not instructions)

| item | now | after | ref |
|---|---|---|---|
| 1. Bundle subproblem: OSQP with polishing at `eps` 1e-6…1e-8 instead of FISTA (a warm start was tried and gives nothing — the cap is hit either way on near-parallel cuts) | 93 % of a `support_values` run; FISTA 13.8 ms per call at KKT residual 2.6e-5 on bundles of condition 3e13 | 0.9 ms per call at residual 4e-16; ≈10× on the subproblem, 2–4× on the run (measured 3.86 → 2.70 s at capacity 40) | R2-N O3, recipes §3 |
| 2. Level master in its k-variable dual (the Gram is already kept) | dense `(n+1)²`: 401 MB per call at 5000 data; LP 0.58 s/iteration | 2 ms | R2-N O4 |
| 3. `PrimalKKTSolver._kernel`: one generalised eigendecomposition instead of an n×n solve per residual evaluation | O(n³) × 20–50 per direction | O(n²) | R2-N unverified |

### 4.6 Small and certain

Lazy `scipy.stats`/`integrate` imports (0.43 → 0.25 s import); `LUSolver`
once (3.5, **DONE** `6620637`); Lanczos convergence check every k steps instead of an
`eigh_tridiagonal` per step (30 % of stochastic log-det); `_angles` cached;
`SubgradientDescent` evaluating `f` twice per iteration (**DONE** `b692230`).

## 5. Questions that decide the work

1. **May the numerics work in components on a `CoordinateSpace`?** *(Answered yes on 2026-08-29; implemented, and the principle is recorded in CURRENT_STATE.md.)* §4.1 is
   3–30× and every item in it assumes yes: `orthonormal_basis`, Lanczos,
   `from_vectors`, the Krylov loop and the randomised routines doing their
   arithmetic on `(dim, k)` arrays whenever `to_components` exists, with the
   coordinate-free code as the fallback. `random_diagonal` already does this.
   *Recommendation: yes; state the principle as "coordinate-free when it must
   be" in DESIGN.md.*

   Response: Accept recomendation. 


2. **What should `DualFeasibleProperty` do given 3.1?** Cross-check routes,
   run the primal to a feasibility tolerance rather than an iteration cap,
   or refuse a dual value without a certified gap? And 4.5 as a whole. This
   is Mag's API (D-13) — his call, with the reproduction in R2-N to hand.

   Response: Leave this point open for later consideration. 


3. **`multiply` truncation** (DESIGN 21.7) costs 2 transforms per product and
   is why the flexure count doubled. Keep the semantics and fuse inside
   `flexural_operator` (4.2.2), or let `multiply` return an untruncated grid?


   Response: I think it is fine to work on an untruncated grid. It should be 
   the user's responsibily to pick a discretisation suited to the problem. 

4. **Is D-11 a guard on the space or on the operation?** `covariance_function`
   and `pointwise_variance` converge whenever the measure's spectrum decays,
   whatever the space's order; 3.8 and 4.2.4 assume the guard is on the
   operation.

   Response: I'm not entirely sure about this one. My sense is that you should not 
   be looking at point values unless you have chosen your space such that point 
   values are going to exist. Now, this can be done implicitly through the prior, 
   but I think that that is probably not the correct thing. 


5. `dense_limit`: raise globally in `log_determinant` or only for the
   evidence?


   Response: I would be consistent unless there is a strong argument otherwise. 

6. Shared solve (4.3.3): an invisible one-entry memo (content-keyed, not
   `id`) or an explicit `posterior_and_evidence(data)`?

   Response: As I understand this point, the first option is quicker, so that that 
   unless there is an obvious cost. but expand if I am missin the point. 

7. `with_order` returning the D-3 subclass — yes? (It changes nothing
   numerically and is what D-3 promised.)

   Response: yes, it's just a convenience method that is sometimes of use. 


8. Sphere `plot` defaults flipped from v1 (`RdBu`→`viridis`, colorbar on,
   gridlines off, PlateCarree→Robinson) with no recorded reason: keep or
   restore?

   Response: Yes, restore. Note that pyslfp defaults to robinson via it's wrapper, but that is its buisness. 

9. Stochastic KL: fix (preconditioned CG, honest error bar) or delete?

   Response: Fix if possible. 

10. `ambient_ball` for non-diagonal covariances: v1's randomised spectrum or
    its sampling radius?

   Response: Not a strong view, again, this is Mag's stuff. Keep close to v1. 

11. Sphinx: point RTD at `pygeoinf2` now or at the rename?

   Response: Leave this for the moment. 

12. Commit the `plt.show()` example edits as they are, or behind a backend
    guard so the suite stays warning-free?

    Response: This was just me looking at the results. Put behind a guard. 

## 6. Suggested order

1. Fix the fixture (3.16) — everything after is tested on it. **DONE.**
2. The one-line bugs: 3.2, 3.3, 3.5, 3.6, 3.7, 3.11, 3.13, 3.14; the
   docstring and catalogue falsities; export holes.
3. §4.1 a–d and 4.2.3 (the component-space work), behind Q1. **DONE except 4.2.3.**
4. §4.2.1, 4.2.5, 4.2.6, 4.2.7 and 4.3 (local, low risk, large).
5. §4.4 (the hooks) and 4.1 e–f. **DONE** (4.4.4, `assembled()`'s form, left as it was pending Q3-style decision).
6. §4.2.2 flexure, 4.2.9 correlated measures (new operator class; dense-Gram
   adjoint tests). **DONE.**
7. §4.5 and 3.1/3.15 with Mag.
8. `review/parallel.md` R1–R7, which several of the above create the hooks
   for (`apply_block`, `DirectSolver(n_jobs=)`). **DONE** except R7's `with_dense_covariance(n_jobs=)`.

## 7. Not done

No cluster or MPI measurements. The reviewers covered their areas' Must/Should
lists completely and the Consider lists by sampling; each reports what it
did not measure. Timings are from a throttling laptop under contention —
re-measure any number you intend to quote. Nothing in the package was
modified; this round adds `REVIEW2.md`, `review/r2_*.md`, the appendix-R row
in `REVIEW.md`, and the scripts under `work/review_r2/`.
