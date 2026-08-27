# Review: PROBABILITY layer (v1 `GaussianMeasure` / invariant measures → `pygeoinf2.probability` + `SymmetricSpace` measure constructors)

> **Note (2026-08-27):** the decisions recorded in `pygeoinf2/REVIEW.md` §11 (D-1 … D-13) override the Must/Should/Consider ranking below wherever they conflict — in particular D-1 (sphere vectors are `SHGrid`, `sampling=1` default), D-2 (points in `(lat, lon)` degrees), D-3 (per-geometry submodules with `Lebesgue`/`Sobolev` subclasses), D-4 (`from_matrix(..., form=)`), D-6 (parallel hooks around operators), D-12 (path *integral* operator), D-13 (convex solvers restored).

All claims below were checked against the code; the ones marked **[verified]** were additionally confirmed by running small scripts (`Sphere(16, order=2)`, dim 289; a non-diagonal-Gram space built by wrapping a v1 `MassWeightedHilbertSpace` through `pygeoinf2.compat.AdaptedSpace`; a 1-D mixture).

Headline: the core is sound (sampling is factor + white noise, push-forward composes nodes, the diagonal closure holds, the metric bookkeeping in `credible_set`/`as_multivariate_normal`/`_weighted_squared`/`from_samples`/norms is right on a non-diagonal Gram), but (a) **precision is dropped by every algebraic operation, including `translate`**, which silently turns the "O(dim) spectral" KL route into a dense O(n³) solve; (b) **`GaussianMixture.log_density`/`marginal_probabilities` are wrong** whenever components have different covariances; (c) **`from_covariance_matrix(form="components")` is wrong on a non-diagonal Gram**; (d) several matrix-free v1 routes (stochastic norms, matrix-free sparse approximation with LU precision, spectral/sampling credible radii, exact correlated-measure KL/norms, Monte-Carlo pointwise variance) have been replaced by dense ones or dropped, contrary to the catalogue.

---

## 1. Functionality retained / extended / lost

### 1.1 `GaussianMeasure` method by method (v1 `pygeoinf/gaussian_measure.py` → v2 `pygeoinf2/probability/gaussian.py`, `base.py`)

| v1 (line) | v2 (line) | Status | Notes |
|---|---|---|---|
| `__init__` (75) | `__init__` (40) | changed | `domain` now explicit; `precision`/`precision_factor` renamed; trait validation added (79–96). No consistency check between the five inputs (see §3). |
| `from_standard_deviation` (144) | (116) | retained | White-noise-correct on weighted spaces (improvement). |
| `from_standard_deviations` (173) | — | **lost** | Catalogue says "scalar or an array"; **[verified]** passing an array raises `ValueError: truth value of an array…` at line 127. |
| `from_covariance_matrix` (215) | (220) | regressed | v1: `eigh` with clipping, PSD-singular allowed, attaches inverse factor. v2: `np.linalg.cholesky` (245) → **[verified]** `LinAlgError` on `diag(1,1,1,0)`; no precision attached; `form="components"` is metric-wrong (§5.1). `form=` is an extension. |
| `from_samples` (280) | (142) | retained | Coordinate-free, correct factor/adjoint **[verified on non-diagonal Gram]**. Requires n ≥ 2 (v1 allowed 1). |
| `from_direct_sum` (328) | `from_product` (174) | regressed | Block-diagonal precision no longer propagated (v1 328–352 did). Covariance comes back as a `_Composition` of the block factor, not a `BlockDiagonalLinearOperator` **[verified]**. |
| `covariance`, `expectation`, `has_zero_expectation`, `domain` | same | retained | |
| `inverse_covariance(_set)`, `covariance_factor(_set)`, `inverse_covariance_factor(_set)` | `precision`, `covariance_factor`, `precision_factor` returning `None` | retained | `_set` booleans replaced by `is None` tests. |
| `sample`, `samples`, `sample_expectation` (443–500) | base.py 69–85, gaussian.py 305 | retained | `parallel=`/`n_jobs=` dropped everywhere. `rng` explicit (improvement). |
| `sample_pointwise_variance/_std` (502, 543) | — | **lost** | Catalogue: "the sampled version is still the general answer" — it is not ported. `SymmetricSpace.pointwise_variance_at` (base.py 601) is point-list based and `SymmetricSpace`-only; no field-valued Monte-Carlo estimate on a generic `HilbertModule`. |
| `deflated_pointwise_variance/_std` (560, 689) | `pointwise_variance_at(rank=, samples=)` (601) | changed | Only on `SymmetricSpace`, returns array at user points, no `_std`. |
| `credible_set` (737) | (705) | regressed | Only the χ² Mahalanobis ellipsoid. Lost: `rank=` (chi-square dof), `open_set=`, `cameron_martin` geometry, `weakened_ellipsoid`, spectral/sampling radius, `rng`. Dense O(n³) precision fallback when no precision (731–746). |
| `ambient_ball` (867) | (755) | regressed | Radius from a **dense** `np.linalg.eigvals` of the full component matrix (780). v1 had randomized `LowRankEig` spectrum or Monte-Carlo radius, both matrix-free. |
| `weakened_ellipsoid` (872) | — | **lost** | |
| `with_dense_covariance` (878) | `covariance.assembled()` | subsumed | but `assembled()` gives no factor/precision; v1's version gave both. |
| `with_regularized_inverse` (906) | (624) | changed | No `preconditioner=`; covariance left unregularised by design (documented). Crashes with `TypeError` if the measure is precision-only (648). |
| `with_sparse_approximation` (967) | (661) | regressed | v1: matrix-free column probing + `deflated_diagonal`, correlation-thresholded, **`splu` sparse precision**, parallel. v2: forms the full dense matrix (680), dense `eigvalsh` (682), returns **no precision and no sampler**. The reason for the method (localised preconditioners need the sparse inverse) is gone. |
| `affine_mapping` (1101) | `affine_map`/`push_forward`/`translate`/`__rmatmul__` (base.py 176–229) | changed | Lost: `inverse_solver=` KKT precision construction (v1 1178–1207); translation-only path no longer preserves precision/precision-factor (v1 1131–1146). |
| `as_multivariate_normal` (1244) | (784) | extended | Any `CoordinateSpace`, metric-correct **[verified]**. |
| `low_rank_approximation` (1289) | (583) | retained | Via `random_cholesky`; rank/oversampling/power/rtol reachable through `**kwargs` (undocumented). |
| `two_point_covariance` (1340) | base.py 126 | retained | Not generalised to direct sums (your catalogue note). |
| `directional_statistics` (1362) | — | lost (minor) | `directional_covariance`/`_variance` ported (base.py 119–124). |
| `zero_expectation` (1405) | `translate(-m)` | subsumed | but drops precision (§1.3). |
| `rescale_directional_variance` (1436) | (563) | retained | |
| `kl_divergence` (1460) | (381) + `kl_divergence_estimate` (408) | extended | Spectral/dense/stochastic; see §2.4, §5.3–5.4. |
| `nuclear_norm`, `hilbert_schmidt_norm` (1536, 1590) | (350, 340) | regressed | Stochastic `random_trace` route gone; non-diagonal case forms the full component matrix (347, 361). `method=` accepts only "auto"/"diagonal" meaningfully. |
| `__neg__` (1649) | — | **lost** | **[verified]** `-mu` → `TypeError`. |
| `__mul__`, `__rmul__`, `__truediv__`, `__add__`, `__sub__` | base.py 259–289 | retained | |
| — | `condition` (814) | new | Not samplable, no precision (§1.2). |
| — | `mahalanobis_squared`, `log_density`, `grad_log_density` | new | |
| — | `PushForwardMeasure`, `ProductMeasure`, `product`, `GaussianMixture` | new | |

### 1.2 Sampling paths **[verified]**

| Derived measure | Samplable? | Cost per sample |
|---|---|---|
| `invariant_measure` | yes | 1 white-noise draw + 1 diagonal multiply (same as v1 `_kl_sample`) |
| `alpha * mu` | yes | factor scaled (`_combine_scale`, 991) |
| `mu + nu` | yes | two draws + add (`_combine_add`, 968); factor dropped. Two diagonal measures could keep a diagonal factor `sqrt(a+b)` (v1 `InvariantGaussianMeasure.__add__` did, one draw) |
| `mu.translate(v)`, `A @ mu` (factor present) | yes | `A(L ξ)` (`_combine_affine`, 933) |
| `A @ mu` (no factor, sampler) | yes | sampler + `A` (955–962) |
| nonlinear `F @ mu` | yes | `PushForwardMeasure` (base.py 313) |
| `low_rank_approximation` | yes | k axpys |
| `from_product` / `ProductMeasure` | yes | block factor or per-factor draws |
| `correlated_measure` | yes | n white-noise draws + n² diagonal multiplies, O(n²·dim) — equal to v1's einsum |
| `GaussianMixture` | yes | categorical + one component draw |
| `condition(...)` | **no** | `can_sample=False`, no factor, no precision (856–862). A Matheron/perturbed-observation sampler (`x + K(d − A x − e)`) is available for free from the prior's sampler and should be attached. |
| `with_sparse_approximation` | **no** | |
| `with_regularized_inverse` | yes | keeps the original sampler (655) |

### 1.3 `covariance_factor` / `precision` / `precision_factor` propagation **[verified]**

| Operation | covariance type | factor | precision |
|---|---|---|---|
| `mu` (invariant) | Diagonal | Diagonal | **yes** |
| `2*mu`, `mu/2` | Diagonal | Diagonal | **dropped** |
| `mu+mu`, `mu-mu` | Diagonal | None | **dropped** |
| `mu.translate(v)` | Diagonal | Diagonal | **dropped** |
| `mu.affine_map(D)`, `D @ mu` (D diagonal) | Diagonal | Diagonal | **dropped** |
| `rescale_directional_variance` | Diagonal | Diagonal | **dropped** |
| `from_product` | `_Composition` | BlockDiagonal | dropped |
| `correlated_measure` | Block (retraited) | Block | **never built** (v1 built `inv(Σ(k))` blocks, symmetric_space.py 822–836) |
| `condition` | node | None | none |

`_rebuild` (gaussian.py 913–931) has no `precision`/`precision_factor` parameters, so no subclass can fix this either. v1's base class also dropped the inverse under `*`/`+`, but kept it under translation and in every `InvariantGaussianMeasure` overload (symmetric_space.py 411–705). Consequence: `mu.translate(v).log_density(x)` raises `NotImplementedError` while `mu.log_density` works; `zero_expectation` (now `translate`) loses the density; and the spectral KL route silently goes dense (§2.4).

### 1.4 Invariant-measure closure

Confirmed **[verified]**: a `DiagonalLinearOperator` covariance stays `DiagonalLinearOperator` under scale, add, subtract, translate, and push-forward through a `DiagonalLinearOperator` (`diagonal.py` 126–147; `_Identity._combine_compose` in nodes.py 63 drops the identity). `_diagonal_eigenvalues()` (gaussian.py 333) is therefore non-None after all of these. Good. The losses versus v1 `InvariantGaussianMeasure`:

- `spectral_variances` → use `mu.covariance.eigenvalues` (fine).
- `rescale_norm_variance` → only `norm_std=` at construction (base.py 338–345); cannot re-scale an existing measure without reconstructing.
- `affine_mapping(inverse_solver=…)` with the "water-filling" spectral preconditioner (symmetric_space.py 470–490) gone.
- `-mu` gone.
- exact KL/norms survive **only while the covariance object is literally a `DiagonalLinearOperator`**; a diagonal measure pushed through anything non-diagonal (e.g. the coordinate inclusion into a different-order space, `order_inclusion_operator`) loses them, as expected.

### 1.5 Correlated measures (`correlated_measure`, `correlated_measure_from_correlations`, base.py 422–553)

Retained: construction from `(dim,n,n)` cross-covariances with symmetry/PSD checks; sampling by the extended KL expansion via a block factor (cheap, correct). Lost relative to `CorrelatedInvariantGaussianMeasure` (symmetric_space.py 720–1433):
- precision (never built);
- exact O(dim·n³) `kl_divergence` (v1 ~1130) → v2 falls to dense (≤512) or stochastic: **[verified]** `cm.kl_divergence_estimate(cm)` on dim 578 took 45 s and returned **−88.6 ± 21.7** for a true value of 0;
- exact `nuclear_norm`/`hilbert_schmidt_norm` (v1 sums of Σ(k) eigenvalues) → v2 forms the dense component matrix (0.3 s at dim 578, O(n²) memory);
- `marginal(i)`, `cross_covariance(i,j)`, `spectral_correlations(i,j)`, `rescale_norm_variance`, `from_function`/`from_index_function`, `from_invariant_measures`;
- the correlation argument forms: v1 `_correlation_array` (1339) accepted scalar, `(dim,)`, `(n,n)`, `(dim,n,n)` and a callable of the Laplacian eigenvalue; v2 accepts only `(n,n)`/`(dim,n,n)` (base.py 537–544);
- `correlated_invariant_gaussian_measure(f, rho)` single-function-pair convenience (symmetric_space.py 1785).

### 1.6 Space-level constructors (v1 symmetric_space.py 1714–1930, 2481–2560; sphere.py 321 → v2 base.py 267–637)

| v1 | v2 | Notes |
|---|---|---|
| `invariant_gaussian_measure(f)` | `invariant_measure(f\|array, expectation=, pointwise_std=, norm_std=)` | good; `pointwise_std`/`norm_std` mutually exclusive (333) |
| `heat_kernel_gaussian_measure(scale)` = exp(−scale²Δ) | `heat_measure(time)` = exp(−time·Δ) | **silent parameter change** (scale² → time); the docstring (400–412) does not say so |
| `sobolev_kernel_gaussian_measure(order, scale)` | `sobolev_measure(order, scale, amplitude=)` | fine |
| `norm_scaled_*` (three) | `norm_std=` | fine; uses `sum(variances)` = trace, same as v1 |
| `point_value_scaled_*` (three) | `pointwise_std=` | improvement: exact `pointwise_variance` (267) instead of a random point; formula `Σ s_k φ_k(p)²/g_k` checked analytically, agrees with v1's `(C u, u)` |
| `invariant_covariance_function(spectral_variances)` (closed-form, all five v1 spaces) | `covariance_function(measure, distances)` (588) | needs `walk_from`, implemented **only on `Sphere`** (sphere.py 571); `PeriodicBox` → `NotImplementedError` |
| `sample_power_measure(measure, n, lmin, lmax)` — power spectra **of samples** (sphere.py 321) | `power_measure(power)` — a measure **from** a per-degree power (555) | different functions. The v1 diagnostic is **not ported**; the catalogue row ("Sampling from a prescribed power spectrum") misdescribes v1. `power_measure` itself is a sensible addition. |
| `correlated_invariant_gaussian_measure` | `correlated_measure*` | partial, §1.5 |

---

## 2. Algorithmic performance

1. **Sampling**: one factor application + one white-noise draw (gaussian.py 305–322); no Lanczos square root anywhere. Good. `_combine_add` doubles the cost for sums of diagonal measures where one draw would do (§1.2).
2. **Push-forward covariance** composes nodes (`operator @ C @ operator.adjoint`, 946–950) and is recognised PSD by the palindrome rule; nothing assembled **[verified: `_Composition`]**.
3. **Precision is never formed densely by default**, but dense fallbacks exist and are reached silently: `credible_set` without precision (731–746: Galerkin matrix + `solve`, O(n²) memory, O(n³)); `_weighted_squared` without precision (372–373, same); `condition` (833: `np.linalg.inv` of the data-space normal matrix, plus m covariance applications to form it — acceptable for small data but no solver hook); `with_sparse_approximation` (680–682, full dense matrix + `eigvalsh`); `ambient_ball` (780, dense `eigvals`); `nuclear_norm`/`hilbert_schmidt_norm` on any non-diagonal covariance (347, 361: full component matrix, n applications, O(n²) memory). v1 had Hutchinson/`random_trace` for both norms (1536–1647) and the low-rank/sampling radius for balls. Given "matrix-free is the default assumption", these are regressions for large problems.
4. **`kl_divergence` route selection** (476–482): spectral if both covariances are `DiagonalLinearOperator`; else dense if `CoordinateSpace` and dim ≤ 512; else stochastic. Costs: spectral O(n) **except** the quadratic term (490) which calls `other._weighted_squared` → dense O(n³) whenever `other` has no precision — which after §1.3 is any measure that has been scaled, added, translated or pushed **[verified: one dense matrix formation for `mu.kl_divergence(2*mu)`]**. The eigenvalues `theirs` are already in hand; the quadratic should be `Σ g_k c_k² / λ_k`. Stochastic: `samples=100` Hutchinson probes each needing a CG solve against `other.covariance` (525–531) plus two SLQ log-determinants (100 probes × ≤40 Lanczos steps each) → ~100 solves + ~8000 covariance applications; 45 s and wrong sign at dim 578 **[verified]**. No preconditioner hook for the CG; `with_traits(definite)` (525, 537, 541) asserts positive-definiteness on covariances that may be numerically singular.
5. **`pointwise_variance`** (base.py 267): O(n), exact, invariant only. `pointwise_variance_at` exact route: one covariance application per point; deflated route: `deflated_diagonal(E C E*)` (rank + samples applications). Line 623 builds `operator` even on the exact path (dead in that branch).
6. **`correlated_measure` sampling**: O(n²·dim), same as v1's `_kl_sample` (1412). Good.
7. **Mixture**: sample = categorical + one draw; `covariance` application = K component applications + K inner products + K axpys (231–262, low-rank between-term built as a factor, sensible); `log_density` = K precision applications.
8. **`low_rank_approximation`**: `random_eig` → `random_range` (randomised.py 103) with rank/oversampling/power/adaptive block mode — functionally equivalent to v1's `LowRankCholesky.from_randomized`; `random_cholesky` (457–463) re-applies the eigen-factor to k basis vectors, a small avoidable cost. `parallel=` gone.
9. `samples(parallel=…)` (joblib) is gone from every entry point; `deflated_pointwise_variance`'s adaptive `rtol`/`block_size` stopping rule is gone (`deflated_diagonal` takes a fixed `samples`).

---

## 3. Code practice / quality

- **Constructor** (`gaussian.py` 40–105): accepts any combination of `covariance`, `covariance_factor`, `precision`, `precision_factor`, `sample`. Coherent as a *description*, but nothing checks that the pieces describe the same measure. **[verified]** `GaussianMeasure(X, covariance=σ=1, covariance_factor=σ=3, precision=σ=7)` is accepted: samples have std 3, `directional_variance` says 1, `mahalanobis` uses 1/49. `testing.check_measure` (testing.py 645–713) catches the covariance/sampler mismatch but **never checks precision·covariance ≈ I** or factor vs covariance. A `sample=` that ignores `rng` is also accepted.
- **Precision-only measures are a landmine**: `covariance is None` is a legal state (85–89) but **[verified]** `2*P`, `P.translate`, `P.credible_set`, `P.nuclear_norm`, `P.affine_map`, `P.with_regularized_inverse`, `P.directional_variance`, `P.kl_divergence(dense)` all die with `TypeError: unsupported operand … NoneType` / `AttributeError` rather than a message; `P + P` returns a `_IndependentSum` of two Gaussians (base.py 269, because `_combine_add` returns None at 974).
- **`_combine_*` hooks**: the protocol is clean and mirrors the operator algebra. `_combine_add` is asymmetric (only `self`'s hook can fire for a `GaussianMeasure + GaussianMixture`, and the mixture defines no `_combine_add`, so a Gaussian plus a mixture is a `_IndependentSum` with a correct covariance but no density — acceptable, but undocumented). `_rebuild` lacks precision parameters (913).
- **Naming**: `affine_map` (linear + optional translation), `push_forward` (any operator, dispatches to `affine_map`), `translate`, `__rmatmul__` — four spellings of one idea; fine, but `push_forward` on a `LinearOperator` is the only one that accepts an `AffineOperator`. `marginal_probabilities` (mixture.py 334) computes component *responsibilities*, not marginals. `kl_divergence(samples=…)` reuses the word `samples` for probe count while `samples()` draws vectors. `heat_measure(time)` vs v1 `scale`.
- **Reach-through into privates**: `other._weighted_squared` (490, 506), `other._diagonal_eigenvalues()` (471), `other._covariance`/`_precision` (517–527), `self._sample_fn` (655), `mu2._symmetric_matrix()` — all within the class, tolerable; but `_weighted_squared` is a public-behaviour switch (dense O(n³) vs O(n)) hidden behind a private name.
- **Duplication**: `from ..traits import Traits as _Traits` re-imported inside four methods (629, 670, 825, 1013) although `Traits` is imported at module top (28); `LinearOperator` re-imported locally (628, 667, 736, 1013); `_resolve_rng` reimplemented in mixture.py (46) instead of the shared one in `algebra/spaces.py` (47). `_combine_affine`'s translation branch duplicates `PushForwardMeasure` logic.
- **Error handling**: mostly good `ValueError`s with remedies. Gaps: `LinAlgError` from `cholesky` (245) uncaught; spectral KL clips a singular `self` to `1e-300` (492) and returns a large finite number where +∞ is the answer; `with_sparse_approximation` `form=` unvalidated until the builder choice (693).
- **Type hints**: `solver: Any` (386, 413, 626), `**kwargs: Any` pass-throughs (391, 418, 588), `-> Any` for `ambient_ball`/`as_multivariate_normal` (755, 784), `translation: object | None` (base.py 181, 242) where `X | None` is meant. `ProbabilityMeasure.__mul__` (base.py 278) tests `isinstance(alpha, (int, float))` while the operator layer also admits `np.floating/np.integer` (operators.py 229).
- **RNG**: every sampling entry point takes `rng` — good. The module default is `algebra/spaces.py:39 _DEFAULT_RNG = default_rng()` with **no public seeding API** (DESIGN §7 says "can be seeded"; only monkeypatching `spaces._DEFAULT_RNG` works). `mixture.py:46` and `testing.py:667` create a *fresh* `default_rng()` per call, so `rng=None` means "shared stream" in spaces but "new unseeded stream" in mixtures — inconsistent; reproducibility with `rng=None` is impossible either way.

---

## 4. Documentation gaps (concrete)

Scan of public methods for missing `Args:`/`Raises:` and for cost/assumption notes:

- `probability/base.py`: `sample` 69, `samples` 76 (raises on n<0, undocumented), `sample_expectation` 82, `two_point_covariance` 126 (raises `TypeError`), `log_density` 146, `grad_log_density` 152, `affine_map` 176, `push_forward` 197, `translate` 223, `PushForwardMeasure.sample` 313, `ProductMeasure.sample` 398, `factor` 418 — no `Args:` sections. `directional_covariance` 119 does not say it requires `covariance is not None`.
- `probability/gaussian.py`: `from_standard_deviation` 116 (Args/Raises), `from_samples` 142 (Raises), `from_product` 174 (Args/Raises; does not say precision is dropped), `from_covariance_matrix` 220 (Args/Raises; does not say PD is required or that no precision is attached), `sample` 305, `hilbert_schmidt_norm` 340 / `nuclear_norm` 350 (`method` values undocumented; **O(n²) memory / n applications cost unstated**), `kl_divergence` 381 (no Args — points to `_estimate`), `kl_divergence_estimate` 408 (Raises; **does not say the spectral route goes dense without a precision**), `rescale_directional_variance` 563 (Raises; does not say precision is lost), `low_rank_approximation` 583 (Args; `kwargs` undocumented), `with_regularized_inverse` 624 (Args; `solver` protocol undocumented), `with_sparse_approximation` 661 (Args; **dense O(n²)/O(n³) cost unstated**; result unsamplable, no precision — unstated), `credible_set` 705 (Raises; dense fallback cost only in a comment), `ambient_ball` 755 (Args; dense eigendecomposition unstated), `condition` 814 (Args; **unsamplable result unstated**; dense inverse of the data-space normal matrix unstated), `mahalanobis_squared` 864, `log_density` 882 (does not say the constant dropped is component-dependent), `grad_log_density` 886.
- `probability/mixture.py`: `log_density` 283 contains a **false statement** ("the same additive constant … which is shared"); `marginal_probabilities` 334 (Raises); `sample` 207; `from_parameter_samples` 128 (Raises).
- `symmetric_space/base.py`: `invariant_measure` 309 (no `Args:`, Raises), `sobolev_measure` 375 / `heat_measure` 400 (no `Args:`; `time` vs v1 `scale` unmentioned; `amplitude` unexplained), `correlated_measure` 422 (Raises), `correlated_measure_from_correlations` 515 (Raises), `power_measure` 555 (Raises), `pointwise_variance_at` 601 (no `Args:`), `covariance_function` 588 (does not say it needs `walk_from`, Sphere-only), `reference_point` 256 (Raises).
- `V1_CATALOGUE.md` rows that are wrong: `from_standard_deviations` ("scalar or an array"), `sample_pointwise_variance` ("sampled version still the general answer" — not ported), `nuclear_norm`/`hilbert_schmidt_norm` ("`random_trace` gives the first stochastically" — no stochastic route), `directional_statistics` (not ported), `sample_power_measure` (misdescribed and not ported), `with_sparse_approximation` (LU precision not ported), `ambient_ball, weakened_ellipsoid` ("Ported (ambient_ball)" ok, but `credible_set`'s `rank`/`open_set`/`cameron_martin` losses unrecorded). DESIGN §7's "module-level `default_rng()` that can be seeded" is not implemented.

---

## 5. Correctness concerns

### 5.1 Metric-sensitive, **wrong** on a non-diagonal Gram
- **`from_covariance_matrix(form="components")`** (gaussian.py 241): `domain.apply_gram(matrix.T).T` yields `sym(C_c G)`, not `G C_c`. **[verified]** on a 4-D space with a dense Gram: max error 0.40 against `G C_c`, 1.8e-15 against `sym(C_c G)`. Correct only when `G` is diagonal (where the broadcasting coincidence saves it). Fix: `np.column_stack([domain.apply_gram(col) for col in matrix.T])`, then symmetrise.

### 5.2 Metric-sensitive, **verified correct** on a non-diagonal Gram
`from_covariance_matrix(form="galerkin")` (4e-16), `as_multivariate_normal` (`G⁻¹ C_gal G⁻¹`, 1e-16), `credible_set` (90 % → 0.901 coverage, dense fallback), `_weighted_squared` dense fallback (matches `cᵀ G S⁻¹ G c`), `ambient_ball` (0.897), `nuclear_norm`/`hilbert_schmidt_norm` (match operator eigenvalues), `from_samples` (1.8 % at 20k draws), dense `kl_divergence` (matches reference). `pointwise_variance` formula checked analytically. White noise in factors: `from_samples` factor domain is `EuclideanSpace(n)` (standard normal, correct); `from_standard_deviation`, `invariant_measure`, `correlated_measure` draw on the space / `DirectSum` (metric-correct via `white_noise_components`).

### 5.3 **`GaussianMixture.log_density` / `marginal_probabilities` are wrong** (mixture.py 283–352)
`GaussianMeasure.log_density = −½ Mahalanobis` (882) omits `−½ log det C_k`, which differs between components. **[verified]** 1-D mixture ½N(0,1)+½N(0,100) at x=0.5: true responsibilities (0.898, 0.102), v2 gives (0.469, 0.531); log-density difference between x=0 and x=3 is 0.73 vs true 2.33. Every use of the mixture density (classification, MCMC on a mixture, evidence-based reweighting if it ever routes through here) is affected unless all components share a covariance. Needs a per-component normalising constant — exact for diagonal covariances (`Σ log λ`), dense ≤ limit, `log_determinant(method="stochastic")` otherwise.

### 5.4 KL divergence
- Spectral route quadratic term (490) → dense O(n³) without precision (§2.4).
- Stochastic route: **[verified]** KL(cm‖cm) = −88.6 ± 21.7 (true 0), so the error bar is also wrong (SLQ on a badly conditioned Sobolev spectrum with `max_iterations=40`, and `with_traits(definite)` asserting PD). Not usable as shipped for the covariances this library produces. v1's randomized route needed an explicit precision and applied `log` through `operator_function_quadratic_form` per probe — also fragile, but v1 additionally had exact O(N)/O(dim·n³) routes for both invariant classes.
- `np.clip(mine, 1e-300)` (492) hides a singular `self`.

### 5.5 Other
- Constructor accepts inconsistent inputs; `check_measure` does not test precision (§3).
- `from_covariance_matrix` rejects PSD-singular matrices (245) and attaches no precision.
- `condition` (814): metric handling is right (component-form inverse applied between `to_components`/`from_components`), but the posterior is unsamplable and precision-less.
- `credible_set` requires the precision to *claim* `POSITIVE_DEFINITE`; a precision built as `precision_factor.adjoint @ precision_factor` gets PD only through the palindrome/invertibility rule — fine for `from_standard_deviation`, not for a user-supplied factor without traits.
- `covariance_function` uses `dirac(point).representer` on the measure's own space; on an order-0 space that is the truncated delta — same as v1, acceptable.

---

## Recommendations

### Must
1. **Fix `from_covariance_matrix(form="components")`** (gaussian.py 241): build the Galerkin matrix as `G @ C_c` column-wise via `apply_gram`; add a test on a non-diagonal-Gram space (use `compat.AdaptedSpace` over a v1 `MassWeightedHilbertSpace` with a dense SPD mass matrix, as in this review).
2. **Fix mixture densities** (mixture.py 283–352): give `GaussianMeasure` a `log_normalising_constant()` (diagonal: `−½Σ log λ_k`; dense ≤ `dense_limit`; else `log_determinant(method="stochastic")`) and add it in `GaussianMixture.log_density`/`marginal_probabilities`; test against `scipy.stats.norm` with unequal variances.
3. **Propagate precision through the algebra** (gaussian.py 913–1000): add `precision`/`precision_factor` to `_rebuild`; `_combine_scale` → `precision/α²`, `precision_factor/α`; `_combine_affine` with the identity operator (translation) → keep both; `_combine_affine` with a diagonal `D` on a diagonal precision → `D⁻¹ P D⁻¹` when `D` is invertible; `_combine_add` of two diagonal covariances → diagonal factor `sqrt(a+b)` and diagonal precision; `from_product` → block-diagonal precision when every factor has one; `correlated_measure` → block precision from `inv(Σ(k))` when PD. Test: `mu.translate(v).log_density`, `(2*mu).precision`, `(mu+mu).covariance_factor` diagonal.
4. **Spectral KL quadratic term** (gaussian.py 490): compute `Σ g_k c_k²/λ_k` from `theirs` and the shift's components instead of `other._weighted_squared`; add a test asserting `_symmetric_matrix` is never called on the spectral route.
5. **Make precision-only measures either work or fail clearly**: every method that touches `self._covariance` (`_combine_scale`, `_combine_affine`, `credible_set`, `nuclear_norm`, `hilbert_schmidt_norm`, `with_regularized_inverse`, `directional_covariance`, `condition`, `low_rank_approximation`, dense KL) must raise a `ValueError("this measure has only a precision …")` — or, for `credible_set`/`log_density`, use the precision directly.
6. **Restore `from_standard_deviations`** (array of per-component std) or make `from_standard_deviation` accept an array as the catalogue claims (build a `DiagonalLinearOperator` factor and precision; note this is component-wise, so document that it is a covariance in the *component* basis, i.e. operator `diag(σ²)` only on an orthonormal space).
7. **Stochastic KL**: either fix (precondition the CG with `other.precision` when present, raise `max_iterations`, refuse when `Estimate.standard_error` exceeds the value, and stop asserting `POSITIVE_DEFINITE` on the covariances) or do not select it under `"auto"`; add an exact block-spectral route for `correlated_measure` (O(dim·n³), v1 symmetric_space.py ~1130) and for `from_product` of diagonal measures. Test `kl(cm, cm) == 0`.

### Should
8. Give `condition` a sampler (`x_prior + K(d − A x_prior − e)`, reusing the prior's and noise's samplers) and a solver hook instead of `np.linalg.inv` (gaussian.py 833); document dimensions/cost.
9. Restore matrix-free routes: `nuclear_norm`/`hilbert_schmidt_norm` `method="stochastic"` via `random_trace` (and `random_trace(C @ C)`), with `Estimate` variants; `ambient_ball` radius from a randomised spectrum (`random_eig`) or from Monte-Carlo samples (`radius_method="sampling"`) instead of dense `eigvals`; `credible_set` `rank=` and `open_set=`.
10. Restore `with_sparse_approximation`'s purpose: matrix-free column probing (or at least a `deflated_diagonal`-based correlation threshold) and a sparse-LU precision returned on the measure; return a sampler when the input had one.
11. Restore a generic Monte-Carlo pointwise variance/std field on any `HilbertModule` (`sample_pointwise_variance(n, rng)`), and a `deflated` variant returning a field, not only `pointwise_variance_at(points)`.
12. `from_covariance_matrix`: accept PSD-singular input (eigh + clip, as v1 215–278) and attach a `precision_factor` (pseudo-inverse) when the matrix is PD; catch `LinAlgError` with a message.
13. Implement `walk_from` on `PeriodicBox` (or provide a closed-form `covariance_function` from the spectrum as v1 did on circle/torus/line/plane) so `covariance_function` is not Sphere-only.
14. Port the v1 `sample_power_measure` diagnostic (power spectrum per degree of samples) — the catalogue row has it confused with `power_measure`; rename the catalogue row.
15. `testing.check_measure`: verify `precision(covariance(u)) ≈ u` and `factor @ factor.adjoint` against `covariance` on the probe directions; verify the sampler honours `rng` (same seed → same draw).
16. Add `__neg__`, `directional_statistics`, the scalar/callable/`(dim,)` correlation forms in `correlated_measure_from_correlations`, and `marginal(i)`/`cross_covariance(i,j)` accessors for correlated measures.
17. Document (or rename) `heat_measure(time)` vs v1 `scale` (time = scale²); document every dense fallback (`credible_set`, `_weighted_squared`, `with_sparse_approximation`, `ambient_ball`, norms, `condition`) in the docstrings with its O(·) cost; fix the false sentence in `GaussianMixture.log_density`.
18. RNG: expose `pygeoinf2.seed(n)` (reseeding `spaces._DEFAULT_RNG`) and route `mixture._resolve_rng` and `testing` through the shared `_resolve_rng`; DESIGN §7 currently promises this.

### Consider
19. Fill in the `Args:`/`Raises:` gaps listed in §4 (file:line given).
20. Tighten types: `solver: Callable[[LinearOperator], LinearOperator] | None`, `translation: X | None`, return types for `ambient_ball`/`as_multivariate_normal`; hoist the repeated local imports of `Traits`/`LinearOperator` in gaussian.py (628–629, 667–670, 736, 825, 1013).
21. Drop the dead `operator` construction on the exact path of `pointwise_variance_at` (base.py 623); make `deflated_diagonal` accept an `rtol`/`block_size` adaptive stopping rule like v1.
22. Rename `marginal_probabilities` → `responsibilities` (or `component_posteriors`); rename the `samples=` keyword in `kl_divergence*` to `probes=`.
23. Optional `parallel=`/`n_jobs=` on `samples()` (joblib was used in v1 for expensive samplers such as MFEM solves).
24. Correct the V1_CATALOGUE rows listed in §4 so the catalogue stops claiming ports that did not happen.
