# Function-Space Hardening of Gaussian Measures

> **Operational plan.** Extends `pygeoinf.gaussian_measure.GaussianMeasure.credible_set` to handle two probabilistically valid hardenings of Gaussian priors in genuinely infinite-dimensional / function-space settings: the **ambient norm ball** and the **weakened covariance ellipsoid**. The math behind these constructions is documented in [`../theory/function-space-hardening.md`](../theory/function-space-hardening.md); this file is the implementation tracker.

---

## Context

The current `credible_set(probability, *, geometry, rank, open_set)` in [`pygeoinf/gaussian_measure.py`](../../../pygeoinf/gaussian_measure.py) lines 428–522 implements exactly one calibration: the chi-square law $\chi^2_k(p)$ with $k = \text{rank or domain.dim}$. This is correct for full-rank finite-dim Gaussians and for fixed rank-$k$ KL truncations, but it is the *wrong* calibration in genuinely infinite-dim function-space settings.

The theory note [`../theory/hardening.md`](../theory/hardening.md) identifies two valid replacements:

1. **Ambient norm balls** $\|m - m_0\|_H \le r_p$ (theory §4): radius solves $\mathbb{P}\left(\sum_j \lambda_j Z_j^2 \le r_p^2\right) = p$.
2. **Weakened covariance ellipsoids** $\|C^{-\theta/2}(m - m_0)\|_H \le r_p$, $0 < \theta < 1$ (theory §6): radius solves $\mathbb{P}\left(\sum_j \lambda_j^{1-\theta} Z_j^2 \le r_p^2\right) = p$.

The Cameron–Martin ball ($\theta = 1$) has Gaussian measure zero in infinite-dim (theory §5.2). The current equal-weights chi-square calibration ignores the eigenvalue spectrum of $C$ entirely. This plan closes that gap.

**Scope (per plan negotiation).** Two modes only: ambient ball + weakened ellipsoid. Posterior property-space hardening (theory §10) and noise-set hardening (theory §9) are out of scope.

**Calibration paths (per plan negotiation).** Both modes ship with two paths:
- **Spectral**: user supplies eigenvalues, or we compute them via `LowRankEig.from_randomized`. Weighted-chi-square quantile gives $r_p$.
- **Non-spectral**: for ambient balls this is Monte Carlo over $\|X^{(i)} - m_0\|^2$; for weakened ellipsoids the primary non-spectral path is **Lanczos matrix functions** applied to $C^{-\theta/2}$, with **randomized `LowRankEig`** as a robust fallback.

**Test backend.** `intervalinf` basis-free Hilbert spaces (`Lebesgue(dim=0, basis=None)`) with `InverseLaplacian` / `BesselSobolevInverse` covariances and `KLSampler`.

---

## Goals & non-goals

### Goals
1. Construct probability-calibrated `Ball` and `Ellipsoid` subsets for Gaussian measures on function spaces.
2. Reuse existing `Subset` / `Ellipsoid` / `Ball` infrastructure ([`pygeoinf/subsets.py`](../../../pygeoinf/subsets.py) lines 884–1152). No new subset class unless strictly necessary.
3. Two independent calibration paths (spectral, non-spectral) that can be cross-validated in tests.
4. Backward compatibility: existing callers of `credible_set(p)` and `credible_set(p, geometry="cameron_martin")` keep current behavior bit-for-bit.
5. End-to-end testability on `intervalinf` basis-free spaces.

### Non-goals
- Posterior property-space hardening, likelihood/noise-set hardening (theory §§9–10).
- New convex-analysis primitives beyond what's needed for the new ellipsoid metric.
- Optimisation-layer integration (BG dual cost, DLI feasible set) — exists already and consumes the new sets unchanged.

---

## API design

### Extended `credible_set()` signature

Replaces [`pygeoinf/gaussian_measure.py:428-522`](../../../pygeoinf/gaussian_measure.py) with a strict superset signature. All existing keyword combinations remain valid.

```python
def credible_set(
    self,
    probability: float,
    /,
    *,
    # Existing kwargs — semantics unchanged when new kwargs are absent.
    geometry: str = "ellipsoid",
    rank: Optional[int] = None,
    open_set: bool = False,
    # New kwargs — invoked only by spectrum-aware geometries.
    theta: Optional[float] = None,
    spectrum: Optional[
        Union[np.ndarray, "LowRankEig", Callable[[int], np.ndarray]]
    ] = None,
    spectrum_size: Optional[int] = None,
    radius_method: str = "auto",         # {"auto", "spectral", "sampling"}
    quantile_method: str = "imhof",      # {"imhof", "ws", "saddlepoint", "mc"}
    fractional_apply: str = "auto",      # {"auto", "lanczos", "low_rank_eig"}
    n_samples: int = 10_000,
    n_lanczos: int = 50,
    spectrum_low_rank_kwargs: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
) -> Union["Ball", "Ellipsoid"]:
```

Accepted `geometry` values:

| Value | Behavior |
|---|---|
| `"ellipsoid"` / `"mahalanobis"` / `"domain"` | Unchanged: $\chi^2_k(p)$ Mahalanobis ellipsoid. |
| `"cameron_martin"` / `"cm"` / `"ball"` / `"norm_ball"` | Unchanged: same set as a ball in `MassWeightedHilbertSpace`. |
| `"ambient_ball"` / `"ambient"` | **NEW.** $\|m - m_0\|_H \le r_p$. |
| `"weakened_ellipsoid"` / `"fractional"` | **NEW.** $\|C^{-\theta/2}(m - m_0)\|_H \le r_p$. Requires `theta ∈ (0, 1)`. |

### Convenience wrappers

For discoverability:

```python
def ambient_ball(self, probability, /, **kwargs) -> Ball: ...
def weakened_ellipsoid(self, probability, /, *, theta, **kwargs) -> Ellipsoid: ...
```

Both delegate to `credible_set`.

### Spectrum resolution

`spectrum` accepts four input forms, resolved by an internal `_resolve_spectrum(...)` helper:

1. **`np.ndarray` of length $N$** — explicit eigenvalues; highest priority.
2. **`LowRankEig` instance** ([`pygeoinf/low_rank.py:194`](../../../pygeoinf/low_rank.py)) — `.eigenvalues` and `.u_factor` used directly.
3. **`Callable[[int], np.ndarray]`** — `spectrum(k)` returns first $k$ eigenvalues; combined with `spectrum_size`. Natural fit for analytic spectra (e.g., `InverseLaplacian.get_eigenvalue(i)`).
4. **`None`** — fallback to `LowRankEig.from_randomized(self.covariance, spectrum_size, measure=self, **spectrum_low_rank_kwargs)`. Requires `spectrum_size`.

### `radius_method = "auto"` policy
- Spectrum provided → `"spectral"`.
- No spectrum, but `self.sample_set is True` → `"sampling"`.
- Neither → raise `ValueError` with actionable message.

### `fractional_apply = "auto"` policy (weakened ellipsoid only)
- Spectrum available → use `SpectralFractionalOperator` (cheap).
- No spectrum + `"auto"` → `"lanczos"`.
- `"lanczos"` always uses Lanczos matrix-function quadrature.
- `"low_rank_eig"` always runs `LowRankEig.from_randomized` internally — used as fallback when Lanczos breakdown is detected.

---

## Module layout

| New file | Purpose |
|---|---|
| [`pygeoinf/quadratic_form_quantile.py`](../../../pygeoinf/quadratic_form_quantile.py) | Weighted-chi-square CDF / quantile (Imhof / WS / saddlepoint / MC). |
| [`pygeoinf/spectral_operator.py`](../../../pygeoinf/spectral_operator.py) | `SpectralFractionalOperator` — matrix-free $f(C)$ from `LowRankEig` or eigenpairs. |
| [`pygeoinf/matrix_function.py`](../../../pygeoinf/matrix_function.py) | Lanczos-based `apply_matrix_function(op, v, func, k)`. |

| Existing file modified | Change |
|---|---|
| [`pygeoinf/gaussian_measure.py`](../../../pygeoinf/gaussian_measure.py) | Extend `credible_set`; add `_resolve_spectrum`, `_build_gauge_squared`, `_sampling_radius`, `_spectral_radius`, `ambient_ball`, `weakened_ellipsoid`. |
| [`tests/test_gaussian_measure_credible_set.py`](../../../tests/test_gaussian_measure_credible_set.py) | Add coverage for new modes (finite-dim sanity). |
| [`tests/test_quadratic_form_quantile.py`](../../../tests/test_quadratic_form_quantile.py) | NEW — unit tests for Imhof/WS/MC. |
| [`tests/test_spectral_operator.py`](../../../tests/test_spectral_operator.py) | NEW — unit tests for fractional operators. |
| [`tests/test_matrix_function.py`](../../../tests/test_matrix_function.py) | NEW — Lanczos matrix-function convergence tests. |
| `intervalinf/tests/spaces/test_lebesgue_hardening.py` | NEW — basis-free function-space hardening tests. |

The existing `Ellipsoid` and `Ball` classes are reused as-is. No new subset class is introduced.

---

## Implementation phases

Each phase is one commit. Phases 1–3 are independent leaves; Phase 4 wires them into `credible_set`; Phase 5 exercises the basis-free `intervalinf` test path; Phase 6 polishes.

### Phase 0 — Theory and plan documents ✓
- Create [`../theory/function-space-hardening.md`](../theory/function-space-hardening.md).
- Create this plan.
- Commit: `docs(hardening): add function-space hardening plan and theory companion`.

### Phase 1 — Weighted chi-square quantile module
- New module: [`pygeoinf/quadratic_form_quantile.py`](../../../pygeoinf/quadratic_form_quantile.py).
- Public API:
  ```python
  def weighted_chi2_cdf(weights, t, *, method="imhof", **kwargs) -> float | np.ndarray
  def weighted_chi2_quantile(weights, probability, *, method="imhof", rtol=1e-6, **kwargs) -> float
  ```
- Methods (priority order):
  1. **Imhof** (default) — numerical inversion via `scipy.integrate.quad`; CDF root-found with `scipy.optimize.brentq`.
  2. **Welch–Satterthwaite** — closed-form moment-match to scaled $\chi^2_\nu$.
  3. **Monte Carlo** — empirical quantile over $\sum_j w_j Z_j^2$ draws.
  4. **Saddlepoint (Lugannani–Rice)** — closed-form, accurate in tails.
- Math derivations in theory companion §4.
- Tests in [`tests/test_quadratic_form_quantile.py`](../../../tests/test_quadratic_form_quantile.py):
  - Equal weights of length $k$ → matches `scipy.stats.chi2.ppf(p, k)` to `1e-8`.
  - Two-weight `(a, b)` → Imhof vs MC ($N = 10^5$) within 3σ.
  - 1000 decaying weights → Imhof and WS bracket each other.

### Phase 2 — Spectral fractional operator
- New module: [`pygeoinf/spectral_operator.py`](../../../pygeoinf/spectral_operator.py).
- `class SpectralFractionalOperator(LinearOperator)`:
  - Constructed from `(u_factor, eigenvalues, func)`; represents $A = U f(\Lambda) U^*$.
  - Implements `__matmul__`, `.adjoint`, `.matrix()` (lazy).
  - Factories: `.from_low_rank_eig(eig, power)`, `.from_callable(u_factor, eigenvalues, func)`.
- Tests in [`tests/test_spectral_operator.py`](../../../tests/test_spectral_operator.py):
  - Round-trip $C^\theta \cdot C^{-\theta} \approx I$ on the eigenspace.
  - Adjointness for symmetric $f$.
  - Match against `DiagonalSparseMatrixLinearOperator ** theta` ([`pygeoinf/linear_operators.py:1600`](../../../pygeoinf/linear_operators.py)) on diagonal cases.

### Phase 3 — Lanczos matrix-function module + sampling-radius helper
- New module: [`pygeoinf/matrix_function.py`](../../../pygeoinf/matrix_function.py).
- Public API:
  ```python
  def lanczos_tridiagonalize(op, v, k, *, reorth="full") -> tuple[np.ndarray, np.ndarray]  # (V, T)
  def apply_matrix_function(op, v, func, k, *, reorth="full") -> Vector
  ```
- Implementation: classical Lanczos with full reorthogonalisation; $f(C) v \approx \|v\|_H V_k f(T_k) e_1$ via `np.linalg.eigh(T_k)`.
- Math derivations in theory companion §6.
- Add private `GaussianMeasure._sampling_radius(probability, gauge_squared_callable, *, n_samples, parallel, n_jobs, rng) -> float`.
- Tests in [`tests/test_matrix_function.py`](../../../tests/test_matrix_function.py):
  - On `EuclideanSpace(20)` with random SPD $C$, verify `apply_matrix_function(C, v, λ→λ**-0.5, k=20)` matches `scipy.linalg.fractional_matrix_power` to `1e-10`.
  - Convergence rate: error vs $k$ on $20\times 20$ SPD with smooth spectrum — verify geometric decay.
  - Full vs no reorthogonalisation parity on $k=30$ of $50$-dim problem.

### Phase 4 — Wire spectrum-aware paths into `credible_set`
- Replace the body of `credible_set` in [`pygeoinf/gaussian_measure.py:428`](../../../pygeoinf/gaussian_measure.py).
- Add helpers:
  - `_resolve_spectrum(spectrum, spectrum_size, low_rank_kwargs, rng) -> (eigenvalues, u_factor_or_None)`.
  - `_spectral_radius(geometry, theta, eigenvalues, probability, quantile_method, rtol, rng) -> r_p`.
  - `_build_gauge_squared(geometry, theta, spectrum_info, fractional_apply, n_lanczos) -> Callable[[Vector], float]`.
- Build returned set:
  - `"ambient_ball"` → `Ball(self.domain, self.expectation, r_p, open_set=open_set)`.
  - `"weakened_ellipsoid"`, spectrum available → `Ellipsoid(..., operator=SpectralFractionalOperator(eig, λ→λ^{-θ}), inverse_operator=..., inverse_sqrt_operator=..., open_set=open_set)`.
  - `"weakened_ellipsoid"`, no spectrum + `fractional_apply="lanczos"` → `Ellipsoid` carrying a `LanczosMatrixFunctionOperator` (matvec wraps `apply_matrix_function`). Slow but correct.
- Tests in [`tests/test_gaussian_measure_credible_set.py`](../../../tests/test_gaussian_measure_credible_set.py):
  - `theta` and `geometry="weakened_ellipsoid"` mutually required → `ValueError` otherwise.
  - Equal-weight degenerate case: `spectrum = np.ones(k)`, `theta = 0` matches $\chi^2_k(p)$.
  - Empirical coverage (5000 samples) within binomial tolerance of $p$ on `EuclideanSpace(5)` with anisotropic $C$.

### Phase 5 — `intervalinf` basis-free tests
- New file: `intervalinf/tests/spaces/test_lebesgue_hardening.py`.
- Setup:
  - `space = Lebesgue(dim=0, domain=[0,1], basis=None)`.
  - `cov_op = InverseLaplacian(space)` (analytic eigenvalues $\lambda_j = 1/(j\pi)^2$).
  - `sampler = KLSampler(cov_op, n_modes=200)`.
  - `measure = GaussianMeasure(covariance=cov_op, sample=sampler.sample, ...)`.
- Tests:
  - Ambient ball, spectral path with truncation $N \in \{50, 200, 1000\}$ — verify Cauchy convergence.
  - Ambient ball, sampling path with `n_samples = 5000` — agrees with spectral $r_p$ within statistical tolerance.
  - Weakened ellipsoid ($\theta = 0.5$), Lanczos vs spectral path — agree to within `5e-3` relative.
  - Sobolev example with `BesselSobolevInverse` covariance.

### Phase 6 — Polish, examples, living-reference update
- Update [`pygeoinf/docs/agent-docs/references/living/`](../references/living/) to document the new modes.
- Add demo `pygeoinf/work/function_space_hardening_demo.py` reproducing theory figures for $\theta \in \{0.2, 0.5, 0.8\}$.
- Benchmark Imhof vs WS on `InverseLaplacian` spectra of size $\{50, 500, 5000\}$.
- Move plan to `completed-plans/`.

---

## Test strategy (detailed)

### Phase 1: weighted-chi-square unit tests

| Test | Setup | Assertion |
|---|---|---|
| `test_equal_weights_match_chi2` | `weights = np.ones(k)` for $k \in \{1, 5, 50\}$ | `weighted_chi2_quantile(weights, p) == chi2.ppf(p, k)` within `1e-8` |
| `test_imhof_vs_mc` | `weights = [4, 9]` | Imhof vs MC ($N = 10^5$) within 3σ |
| `test_imhof_vs_ws_asymptotic` | `weights = np.full(1000, 1.0)` | Imhof and WS within `1e-4` |
| `test_anisotropic_heavy_tail` | `weights = [100, 1, 1, …, 1]` | Imhof and WS bracket each other; MC inside |
| `test_saddlepoint_deep_tail` | exponential decay, $p = 0.999$ | Saddlepoint vs Imhof within `1e-3` |

### Phase 2: spectral fractional operator
- Round-trip $C^\theta \cdot C^{-\theta}$ on eigenspace.
- Match `DiagonalSparseMatrixLinearOperator ** theta` on diagonal cases.
- Adjointness for symmetric $f$.

### Phase 3: Lanczos
- Match `scipy.linalg.fractional_matrix_power(C, -0.5) @ v` on dense SPD $5\times 5$ to `1e-10`.
- Convergence: error vs $k$ on $20\times 20$ SPD with smooth spectrum — geometric decay.
- Reorthogonalisation parity on $k=30$ of $50$-dim.

### Phase 4: finite-dim sanity for `credible_set`
- Ambient ball cross-check vs MC over $\|X - m\|^2$.
- $\theta = 0$ weakened ellipsoid equals ambient ball.
- `spectrum = np.ones(k)`, $\theta = 1$ equals $\chi^2_k$.
- Coverage test: 5000 samples within binomial tolerance.

### Phase 5: basis-free `intervalinf` integration

| Test | Asserts |
|---|---|
| `test_ambient_ball_spectral_convergence` | $r_p(N)$ Cauchy in $N \in \{50, 200, 1000\}$ |
| `test_ambient_ball_spectral_vs_sampling` | Spectral $r_p$ vs sampling $r_p$ within statistical tolerance |
| `test_weakened_ellipsoid_lanczos_vs_spectral` | Lanczos ($n_{\text{lanczos}}=50$) vs spectral ($N=200$) within `5e-3` rel. error, $\theta=0.5$ |
| `test_sobolev_weakened_ellipsoid` | Same on `Sobolev(dim=0, s=1, k=1)` with `BesselSobolevInverse` |
| `test_no_spectrum_no_sampling_raises` | `credible_set("ambient_ball", spectrum=None)` on non-sampling measure → `ValueError` |
| `test_cm_warning` | $\theta = 0.99$ with $N=20$ emits borderline-trace `UserWarning` |

### Statistical coverage
For $(d, \theta, p) \in \{5, 20\} \times \{0.0, 0.3, 0.5, 0.7\} \times \{0.5, 0.9, 0.99\}$ on `EuclideanSpace(d)`, draw 5000 samples, assert $|\hat p - p| \le 3 \sqrt{p(1-p)/5000}$.

---

## Risks and open items

### Trace-class violation for $\theta \in (0, 1)$
For some spectra (Laplacian $\lambda_j \propto j^{-2}$), $\sum \lambda_j^{1-\theta}$ diverges for $\theta \ge 1/2$. We emit `UserWarning` and proceed with truncated radius.

### Lanczos numerical robustness
Loss of orthogonality is the dominant failure mode. Default `reorth="full"`.

### `Ellipsoid` with non-materialisable operator
When `fractional_apply="lanczos"` and `spectrum` is `None`, the returned `Ellipsoid` carries a `LanczosMatrixFunctionOperator`. Each `__matmul__` is $O(k \cdot \text{cost}(C\cdot v))$. Correct but slow. Document in docstring.

### `is_element` on basis-free spaces
`Ellipsoid.is_element(m)` requires evaluating $\langle A(m-c), m-c\rangle$. Need to confirm `intervalinf.Lebesgue.is_element` accepts general `Function` objects.

### Choice of quantile-method default
Imhof default. For $N > 10^4$, evaluate switching to WS in Phase 6 benchmark.

### Eigenvalue ordering
`LowRankEig.from_randomized` returns descending. For analytic spectra (`InverseLaplacian.get_eigenvalue(i)`), index ordering is also descending. Document; tests cross-check.

### Backward compatibility
The replacement signature keeps `(probability, /)` positional and all old kwargs. Searched repo: no callers use positional kwargs beyond `probability`. Safe.

---

## Verification

After Phase 5:

```bash
conda activate inferences3
cd /home/adrian/PhD/Inferences/pygeoinf && python -m pytest tests/test_quadratic_form_quantile.py tests/test_spectral_operator.py tests/test_matrix_function.py tests/test_gaussian_measure_credible_set.py -v
cd /home/adrian/PhD/Inferences/intervalinf && python -m pytest tests/spaces/test_lebesgue_hardening.py -v
ruff check pygeoinf/pygeoinf/quadratic_form_quantile.py pygeoinf/pygeoinf/spectral_operator.py pygeoinf/pygeoinf/matrix_function.py pygeoinf/pygeoinf/gaussian_measure.py
```

Expected: all tests pass; ruff clean.

Manual smoke test (`pygeoinf/work/function_space_hardening_demo.py`):
- Construct $\mu = \mathcal{N}(0, \text{InverseLaplacian})$ on `Lebesgue(dim=0, [0,1])`.
- Build `μ.ambient_ball(0.9)` (spectral) — print $r_p$.
- Build `μ.ambient_ball(0.9, radius_method="sampling", n_samples=5000)` — print $r_p$.
- Build `μ.weakened_ellipsoid(0.9, theta=0.5)` two ways (spectral / Lanczos).
- Verify all four radii agree within 1%.

---

## Commit cadence

Each phase ships as one commit. Following [`CLAUDE.md`](../../../../CLAUDE.md) commit convention:

```
feat(hardening): <subject>

- specific changes

Plan: pygeoinf/docs/agent-docs/active-plans/function-space-hardening-plan.md
Phase: N of 6
Related: <phase-complete file when applicable>
```

Final commit moves this plan to `completed-plans/` and adds a phase-complete summary.
