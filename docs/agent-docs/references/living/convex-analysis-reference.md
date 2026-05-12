# Convex Analysis Reference

## Scope

This reference covers [pygeoinf/convex_analysis.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/convex_analysis.py), the convex-analysis public surface in [pygeoinf/__init__.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/__init__.py), the slice-plotting helpers in [pygeoinf/plot.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/plot.py), the Gaussian credible-subset bridge in [pygeoinf/gaussian_measure.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/gaussian_measure.py), the finite-dimensional tutorial notebook [tutorials/gaussian_measure_to_sets_demo.ipynb](/home/adrian/PhD/Inferences/pygeoinf/tutorials/gaussian_measure_to_sets_demo.ipynb), and their direct tests in [tests/test_support_function_constructors.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_support_function_constructors.py), [tests/test_support_function_algebra.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_support_function_algebra.py), [tests/test_gaussian_measure_credible_set.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_gaussian_measure_credible_set.py), [tests/test_plot.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_plot.py), [tests/test_subsets.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_subsets.py), and [tests/test_halfspaces.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_halfspaces.py).

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

- `GaussianMeasure.credible_set(probability, geometry="ellipsoid", rank=None, open_set=False)`
  - Converts a finite-dimensional/truncated Gaussian probability into a chi-square radius `sqrt(chi2.ppf(probability, rank))`.
  - Domains with no positive finite `dim` (for example basis-free function spaces) must pass an explicit effective `rank`.
  - `geometry="ellipsoid"` returns a `subsets.Ellipsoid` in the measure domain with shape operator equal to the inverse covariance and inverse operator equal to the covariance.
  - `geometry="cameron_martin"`/`"ball"` returns the equivalent `subsets.Ball` in a `MassWeightedHilbertSpace` whose mass operator is the inverse covariance.
  - Intended bridge from Bayesian Gaussian priors to set-theoretic DLI constraints; exact for finite-dimensional full-rank measures and interpreted as finite-rank/truncated in infinite-dimensional theory.

## Tutorial Notebook

- [tutorials/gaussian_measure_to_sets_demo.ipynb](/home/adrian/PhD/Inferences/pygeoinf/tutorials/gaussian_measure_to_sets_demo.ipynb)
  - Beginner-oriented finite-dimensional demo: builds a 2D `GaussianMeasure`, constructs a probability-calibrated `Ellipsoid` and equivalent Cameron-Martin `Ball`, plots the returned subset directly via `.plot()`, estimates their probabilities by Monte Carlo, compares against the exact chi-squared calibration, and ends with an optional affine-pushforward section.
  - Focuses only on sets induced directly by the Gaussian measure itself and keeps the main workflow on the `pygeoinf` API surface rather than manual geometry reconstruction.

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
  - `credible_set()` bridge from Gaussian measures to `Ellipsoid`/Cameron-Martin `Ball` subsets.
- [tutorials/gaussian_measure_to_sets_demo.ipynb](/home/adrian/PhD/Inferences/pygeoinf/tutorials/gaussian_measure_to_sets_demo.ipynb)
  - Runnable notebook demonstration of the finite-dimensional Gaussian-to-credible-sets workflow, now centered on the simplest beginner path: `GaussianMeasure.from_covariance_matrix(...)`, `credible_set(...)`, `samples(...)`, and subset `.plot()`.
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
- Full-suite baseline after the Phase 5 visualisation/public-surface port: 624 passed, 1 xfailed, 6 warnings.