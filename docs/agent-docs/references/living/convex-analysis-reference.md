# Convex Analysis Reference

## Scope

This reference covers [pygeoinf/convex_analysis.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/convex_analysis.py) and its direct tests in [tests/test_support_function_constructors.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_support_function_constructors.py) and [tests/test_support_function_algebra.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_support_function_algebra.py).

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
  - Evaluation requires `A^{-1/2}`; support-point recovery requires `A^{-1}`.
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

## File Map

- [pygeoinf/convex_analysis.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/convex_analysis.py)
  - All support-function classes and algebraic combinators.
- [tests/test_support_function_constructors.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_support_function_constructors.py)
  - Constructor, evaluation, support-point, and error-path coverage for the base and concrete support-function types.
- [tests/test_support_function_algebra.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_support_function_algebra.py)
  - Algebraic composition coverage for linear images, translation, scaling, addition, operator overloading, and support-point propagation.
- [tests/test_halfspaces.py](/home/adrian/PhD/Inferences/pygeoinf/tests/test_halfspaces.py)
  - Half-space integration coverage that now exercises the live `pygeoinf.convex_analysis` module instead of skipping import-guarded tests.

## Test Expectations

- Support-function constructor and algebra tests add 102 passing tests combined.
- Half-space tests should run without skip guards once [pygeoinf/convex_analysis.py](/home/adrian/PhD/Inferences/pygeoinf/pygeoinf/convex_analysis.py) is present.
- Full-suite baseline after the Phase 2 port: 532 passed, 1 xfailed, 1 warning.