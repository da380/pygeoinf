# Convex Analysis Reference

## Scope

This reference covers [pygeoinf/convex_analysis.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/convex_analysis.py), the convex-analysis public surface in [pygeoinf/__init__.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/__init__.py), the slice-plotting helpers in [pygeoinf/plot.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/plot.py), the Gaussian credible-subset bridge in [pygeoinf/gaussian_measure.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/gaussian_measure.py), the weighted-chi-square backend in [pygeoinf/quadratic_form_quantile.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/quadratic_form_quantile.py), the finite-dimensional tutorial notebook [tutorials/gaussian_measure_to_sets_demo.ipynb](/home/adrian/PhD/Inferences/pygeoinf/tutorials/gaussian_measure_to_sets_demo.ipynb), and their direct tests in [tests/test_support_function_constructors.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_support_function_constructors.py), [tests/test_support_function_algebra.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_support_function_algebra.py), [tests/test_gaussian_measure_credible_set.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_gaussian_measure_credible_set.py), [tests/test_quadratic_form_quantile.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_quadratic_form_quantile.py), [tests/test_plot.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_plot.py), [tests/test_subsets.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_subsets.py), [tests/test_halfspaces.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_halfspaces.py), and the basis-free end-to-end coverage in [intervalinf/tests/spaces/test_lebesgue_hardening.py](/home/adrian/PhD/Inferences/intervalinf/tests/spaces/test_lebesgue_hardening.py).

The module introduces support-function primitives for closed convex sets in a Hilbert space, plus algebraic combinators for linear images, Minkowski sums, and nonnegative scaling.

## Core Types

### `SupportFunction`

- Base abstraction for support functions of closed convex sets in a Hilbert space.
- Inherits from `NonLinearForm` and exposes a primal-space view via `primal_domain`.
- Main extension points:
  - `_mapping(q)` for evaluation.
  - `support_point(q)` for an optional maximizer / subgradient representative.
  - `value_and_support_point(q)` for fused value/support-point computation.
- Convenience constructors:
  - `SupportFunction.callable(...)`
  - `SupportFunction.point(...)`
- Algebra helpers:
  - `image(operator)`
  - `translate(point)`
  - `scale(alpha)`
  - `__add__` for Minkowski-sum composition
  - `__mul__` / `__rmul__` for nonnegative scalar scaling

### Concrete support-function implementations

- `BallSupportFunction`
  - Support function of a closed ball with center and radius.
  - Provides fused value/support-point computation that reuses a single norm evaluation.
- `EllipsoidSupportFunction`
  - Support function of an ellipsoid defined by a center, radius, and SPD shape operator.
  - Evaluation uses `A^{-1/2}` when supplied, or falls back to `A^{-1}` via `sqrt(<q, A^{-1}q>)`.
  - Support-point recovery requires `A^{-1}`.
  - Handles small negative quadratic forms by clamping numerical noise to zero.
- `CallableSupportFunction`
  - Wraps user-supplied evaluation and optional support-point callbacks.
- `PointSupportFunction`
  - Support function of a singleton set `{p}`.
  - Always returns the fixed point as the support point.

### Half-space support form

- `HalfSpaceSupportFunction`
  - Extended-real-valued support function for a half-space with either `<=` or `>=` orientation.
  - Finite only in directions parallel to the half-space normal with the allowed sign.
  - Returns `float("inf")` for unbounded directions.
  - Can optionally return the minimum-norm boundary point as a canonical support point.
  - Inherits directly from `NonLinearForm`, not from `SupportFunction`, because the implementation focuses on the extended-real half-space form rather than the finite-valued support-function algebra used by the other classes.

## Algebraic Combinators

- `LinearImageSupportFunction`
  - Represents the support function of `A(C)` via `h_C(A^* q)`.
  - The returned object lives on `operator.codomain`.
  - If the base support point is available, maps it through `A`.
- `MinkowskiSumSupportFunction`
  - Represents `h_C + h_D` on a shared primal domain.
  - Returns a support point only when both operands provide one.
- `ScaledSupportFunction`
  - Represents `alpha * h_C` for `alpha >= 0`.
  - `alpha == 0` collapses to the singleton `{0}` and returns the zero vector as support point.

## Visualisation Surface

- `SubspaceSlicePlotter`
  - Plots 1D, 2D, and 3D affine slices of subsets in Euclidean spaces.
  - Uses exact polyhedral slicing for `PolyhedralSet` inputs and sampled membership masks for other subsets.
  - Supports default bounds parsing, subspace embedding, and dimension-specific renderers.
- `plot_slice`
  - Functional entry point that wraps `SubspaceSlicePlotter` and returns `(figure, axes, payload)`.
  - Serves as the implementation target for `Subset.plot()` delegation tests.

## Gaussian Credible Subsets

- `GaussianMeasure.credible_set(probability, /, *, geometry="ellipsoid", rank=None, open_set=False, theta=None, spectrum=None, spectrum_size=None, radius_method="auto", quantile_method="auto", quantile_tol=1e-2, fractional_apply="auto", n_samples=10000, n_lanczos=50, spectrum_low_rank_kwargs=None, rng=None)`
  - Bridges a Gaussian probability to a calibrated convex subset in the measure domain.
  - **Finite-dimensional / chi-square calibration** (default, `geometry in {"ellipsoid", "cameron_martin", "ball"}`):
    - Converts the probability into a chi-square radius `sqrt(chi2.ppf(probability, rank))`.
    - Domains with no positive finite `dim` (for example basis-free function spaces) must pass an explicit effective `rank`.
    - `geometry="ellipsoid"` returns a `subsets.Ellipsoid` whose shape operator is the inverse covariance.
    - `geometry="cameron_martin"`/`"ball"` returns the equivalent `subsets.Ball` in a `MassWeightedHilbertSpace` whose mass operator is the inverse covariance.
  - **Function-space ambient ball** (`geometry="ambient_ball"`/`"ambient"`):
    - Returns a `subsets.Ball` in the ambient measure domain calibrated against `||X - m||^2 ~ sum_j lambda_j Z_j^2` (weighted chi-square).
    - `radius_method="spectral"` uses `weighted_chi2_quantile` over a supplied spectrum; `radius_method="sampling"` draws `n_samples` Monte Carlo radii.
    - `radius_method="auto"` selects the spectral path only when an explicit non-`None` spectrum object/array/callable is supplied; otherwise it falls back to sampling when the measure supports `sample()`, else raises `ValueError`.
    - `spectrum` may be an `np.ndarray`, a `LowRankEig`, a callable `f(k) -> first-k eigenvalues`, or `None` together with `spectrum_size` to trigger `LowRankEig.from_randomized(...)` on the covariance operator.
    - `quantile_method="auto"` chooses between saddlepoint and Imhof using an effective-spectrum heuristic; `quantile_tol` is the desired relative quantile accuracy for that auto-selection and is ignored for explicit backends.
    - Convenience wrapper: `GaussianMeasure.ambient_ball(probability, **kwargs)`.
  - **Weakened ellipsoid** (`geometry="weakened_ellipsoid"`/`"fractional"`):
    - Returns an `Ellipsoid` with shape operator $C^{-\theta}$ for $\theta \in (0, 1)$, interpolating between the ambient ball limit ($\theta \downarrow 0$) and the Cameron-Martin boundary ($\theta \uparrow 1$).
    - Radius is calibrated against the weakened weighted chi-square $\sum_j \lambda_j^{1-\theta} Z_j^2$.
    - `fractional_apply="auto"` prefers the low-rank eig backend when a `LowRankEig` is already available or when `spectrum=None` is resolved through randomized eigendecomposition; otherwise it falls back to the matrix-free Lanczos backend.
    - `fractional_apply="lanczos"` builds the gauge operator via matrix-free Krylov (see `apply_matrix_function`); `"low_rank_eig"` materialises a `LowRankEig` factorisation.
    - Emits `UserWarning` when the truncated trace $\sum_j \lambda_j^{1-\theta}$ is near the Cameron–Martin boundary.
    - Convenience wrapper: `GaussianMeasure.weakened_ellipsoid(probability, *, theta, **kwargs)`.
  - Backend modules: [pygeoinf/quadratic_form_quantile.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/quadratic_form_quantile.py) (auto/Imhof/Wood-Satterthwaite/saddlepoint/MC quantiles), [pygeoinf/spectral_operator.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/spectral_operator.py) (`SpectralFractionalOperator`), [pygeoinf/matrix_function.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/matrix_function.py) (Lanczos `apply_matrix_function`).

## Tutorial Notebook

- [tutorials/gaussian_measure_to_sets_demo.ipynb](/home/adrian/PhD/Inferences/pygeoinf/tutorials/gaussian_measure_to_sets_demo.ipynb)
  - Beginner-oriented finite-dimensional demo: builds a 2D `GaussianMeasure`, constructs a probability-calibrated `Ellipsoid` and equivalent Cameron-Martin `Ball`, plots the returned subset directly via `.plot()`, estimates their probabilities by Monte Carlo, compares against the exact chi-squared calibration, and ends with an optional affine-pushforward section.
  - Focuses only on sets induced directly by the Gaussian measure itself and keeps the main workflow on the `pygeoinf` API surface rather than manual geometry reconstruction.
- [work/post_merge_demo.ipynb](/home/adrian/PhD/Inferences/pygeoinf/work/post_merge_demo.ipynb)
  - Developer-facing post-merge showcase notebook for PR #132 additions. Section B now uses a periodic line `Lebesgue` space with a heat-kernel prior and explicit `spectral_variances`, giving a continuum-faithful function-space example while keeping `ambient_ball` and `weakened_ellipsoid` on the fast spectral calibration path.

## Public Exports

- [pygeoinf/__init__.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/__init__.py)
  - Re-exports the convex-analysis classes, `SubgradientDescent`, `ProximalBundleMethod`, `DualMasterCostFunction`, `SubspaceSlicePlotter`, and `plot_slice` from the package root.
  - Must not import from `symmetric_space` for the convex-analysis port.

## Packaging Notes

- [pyproject.toml](/home/adrian/PhD/Inferences/pygeoinf/pyproject.toml)
  - Adds optional extras for `osqp`, `clarabel`, and `plotly`.
  - Keeps the dev-group `Cartopy` constraint aligned with the `sphere` extra so
    `poetry install --with dev --all-extras` resolves cleanly in CI.
  - Caps `pyqt6` below 6.10 because the current package index does not provide
    installable `PyQt6-Qt6` artifacts for the newer 6.10+/6.11 line used by Poetry.
  - Should not reference excluded paths such as `symmetric_space_new`.

## File Map

- [pygeoinf/convex_analysis.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/convex_analysis.py)
  - All support-function classes and algebraic combinators.
- [pygeoinf/gaussian_measure.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/gaussian_measure.py)
  - `credible_set()` bridge from Gaussian measures to `Ellipsoid`/Cameron-Martin `Ball` subsets, plus function-space `ambient_ball` and `weakened_ellipsoid` modes.
- [pygeoinf/quadratic_form_quantile.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/quadratic_form_quantile.py)
  - Weighted chi-square CDF/quantile (`weighted_chi2_cdf`, `weighted_chi2_quantile`) via auto, Imhof, Wood-Satterthwaite, saddlepoint, and Monte Carlo backends.
  - `_auto_select_method(weights, tol)` uses an effective-dimension heuristic `nu_eff = (sum w)^2 / sum w^2` to pick saddlepoint when the requested tolerance is loose enough and Imhof otherwise.
  - `_imhof_cdf` uses adaptive truncation: the integration upper-bound `U` is found by binary search on `log(U) + 0.25·Σlog(1+w_j²U²) ≥ log(1/(π·rtol))`. For N=50 decaying-spectrum weights this gives U~640 vs the old single-weight heuristic 16/w_min~400 000 — ~625× smaller grid and ~100× faster.
- [pygeoinf/spectral_operator.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/spectral_operator.py)
  - `SpectralFractionalOperator` and `fractional_operators_from_eig`; finite-rank fractional power of a covariance.
- [pygeoinf/matrix_function.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/matrix_function.py)
  - Lanczos tridiagonalisation and `apply_matrix_function(op, v, func, k, reorth="full")` for matrix-free $f(C) v$.
- [tests/test_quadratic_form_quantile.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_quadratic_form_quantile.py)
  - Weighted chi-square quantile/CDF tests across auto/Imhof/WS/saddlepoint/MC, including auto-selector coverage for isotropic vs anisotropic spectra and tolerance-driven backend switching.
- [tests/test_spectral_operator.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_spectral_operator.py)
  - `SpectralFractionalOperator` round-trips, adjoint, diagonal parity.
- [tests/test_matrix_function.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_matrix_function.py)
  - Lanczos accuracy, convergence, and reorthogonalisation parity.
- [intervalinf/tests/spaces/test_lebesgue_hardening.py](/home/adrian/PhD/Inferences/intervalinf/tests/spaces/test_lebesgue_hardening.py)
  - End-to-end basis-free integration tests on `Lebesgue` with `InverseLaplacian` and `BesselSobolevInverse` covariances.
- [tutorials/gaussian_measure_to_sets_demo.ipynb](/home/adrian/PhD/Inferences/pygeoinf/tutorials/gaussian_measure_to_sets_demo.ipynb)
  - Runnable notebook demonstration of the finite-dimensional Gaussian-to-credible-sets workflow, now centered on the simplest beginner path: `GaussianMeasure.from_covariance_matrix(...)`, `credible_set(...)`, `samples(...)`, and subset `.plot()`.
- [work/function_space_hardening_demo.py](/home/adrian/PhD/Inferences/pygeoinf/work/function_space_hardening_demo.py)
  - Hardening exploration / benchmarking script for ambient-ball and weakened-ellipsoid radii, including `benchmark_all_methods()` for comparing Imhof, Wood-Satterthwaite, saddlepoint, and Monte Carlo calibration across spectrum families.
- [tests/test_gaussian_measure_credible_set.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_gaussian_measure_credible_set.py)
  - Chi-square radius, ellipsoid membership/support, Cameron-Martin ball, rank override, and validation tests for `GaussianMeasure.credible_set()`.
- [tests/test_support_function_constructors.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_support_function_constructors.py)
  - Constructor, evaluation, support-point, and error-path coverage for the base and concrete support-function types.
- [tests/test_support_function_algebra.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_support_function_algebra.py)
  - Algebraic composition coverage for linear images, translation, scaling, addition, operator overloading, and support-point propagation.
- [tests/test_halfspaces.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_halfspaces.py)
  - Half-space integration coverage that now exercises the live `pygeoinf.convex_analysis` module instead of skipping import-guarded tests.
- [pygeoinf/plot.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/plot.py)
  - Distribution plots plus `SubspaceSlicePlotter` and `plot_slice` for subset visualisation.
- [tests/test_plot.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_plot.py)
  - Plot-module coverage, including the new subspace-slice plotting helpers.
- [tests/test_subsets.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_subsets.py)
  - `Subset.plot()` entry-point tests that verify delegation to `plot_slice` and default-subspace behaviour.
- [pygeoinf/__init__.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/__init__.py)
  - Package-level public export surface for convex-analysis and plotting symbols.
- [pyproject.toml](/home/adrian/PhD/Inferences/pygeoinf/pyproject.toml)
  - Optional dependency groups and Ruff path ignores.

## Test Expectations

- Support-function constructor and algebra tests add 102 passing tests combined.
- Half-space tests should run without skip guards once [pygeoinf/convex_analysis.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/convex_analysis.py) is present.
- Function-space hardening adds 98 new tests in pygeoinf (Phases 1–4) + 6 end-to-end tests in intervalinf (Phase 5). pygeoinf suite runs in ~10s; intervalinf hardening suite in ~11s.
- The weighted-chi-square auto-selection follow-up adds 16 tests in [tests/test_quadratic_form_quantile.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_quadratic_form_quantile.py), bringing that module to 58 tests and the last validated pygeoinf suite count to 925 passing tests.