# Geodesic Integration Reference

## Bottom line

The new geodesic-ball and spherical-cap functionality is **not redundant** with the pre-existing operator primitives, even though some of the resulting linear functionals can be expressed in terms of those primitives after the right weighting fields are already known.

The key distinction is:

- **Old primitives** such as `l2_products_operator(...)` and `LinearOperator.from_vectors(...)` evaluate pairings once the user already has the correct weighting vectors.
- **New geodesic-ball / spherical-cap methods** construct those weighting vectors or linear forms directly from geometry, provide exact sphere-specific implementations when available, handle normalization and physical area scaling, and expose a public API for region integrals rather than only path integrals.

Reducibility to lower-level primitives is therefore true in a mathematical sense, but false as an argument that the public functionality was already present.

## What the older API could already do

### `l2_products_operator(weighting_functions)`

For a space `X`, this builds the operator

$$
u \mapsto \big(\langle u, w_i \rangle_{L^2}\big)_i.
$$

This is a low-level batching primitive. It does **not** know how to build a cap indicator, a normalized cap average, or a geodesic-ball weight from geometric inputs.

### `geodesic_integral(p1, p2)`

This builds a **path** functional

$$
u \mapsto \int_{\gamma(p_1,p_2)} u\,ds,
$$

implemented as a weighted sum of Dirac evaluations along a geodesic quadrature rule.

This is about 1-D geodesic segments, not 2-D spherical caps or higher-dimensional geodesic balls.

### `path_average_operator(paths)`

This batches many `geodesic_integral(...)` forms into one operator. Again, it is a path API, not a region API.

## Why the new functionality is still needed

### 1. Weight construction is the missing piece

For a spherical cap `C`, the desired functionals are

$$
I_C(u) = \int_C u\,dS,
\qquad
A_C(u) = \frac{1}{|C|}\int_C u\,dS.
$$

Yes, these can be written as

$$
I_C(u) = \langle u, \mathbf{1}_C \rangle_{L^2},
\qquad
A_C(u) = \left\langle u, \frac{1}{|C|}\mathbf{1}_C \right\rangle_{L^2}.
$$

But that only helps if the user already has the coefficient vector for the cap indicator `1_C` in the truncated basis. Constructing that vector from `(center, radius)` is precisely the geometric work that the new API performs.

Without the new methods, the user would need to manually:

- derive or approximate the cap indicator coefficients,
- decide how to normalize them,
- convert between angular radius and physical geodesic radius,
- handle sphere radius scaling,
- and then finally call `l2_products_operator(...)`.

That is not “already available” in any meaningful public-API sense.

### 2. Path integrals and region integrals are different objects

`geodesic_integral(...)` is for line integrals along a 1-D path. `geodesic_ball_integral(...)` is for volume or area integrals over a geodesic ball. One cannot obtain the latter from the former without introducing extra geometry-specific machinery.

So even if the implementation patterns look similar, the functionality is not the same:

- path integrals consume **two endpoints** and a 1-D quadrature rule,
- geodesic-ball integrals consume a **center and radius** and a region quadrature rule or exact spectral formula.

### 3. The sphere implementation provides an exact spectral cap constructor

On `sphere.Lebesgue`, `spherical_cap_integral(...)` and `spherical_cap_average(...)` compute the truncated cap coefficients exactly using `pyshtools.SHCoeffs.from_cap`.

That matters because the alternative “old API only” route would require the user to recreate the same coefficient construction manually. The exact spectral constructor is therefore not redundant; it is the missing geometry-aware piece that makes the `L^2` pairing representation actually usable.

### 4. The Sobolev-domain API is wired correctly for the user

On Sobolev spaces, `l2_products_operator(...)` is not the raw underlying `LinearOperator.from_vectors(...)`. It is lifted from the underlying Lebesgue operator through `LinearOperator.from_formal_adjoint(...)` so that:

- the **forward action** still computes the intended `L^2` pairing,
- the **adjoint** is correct for the Sobolev geometry.

Likewise, `SymmetricSobolevSpace.geodesic_ball_integral(...)` delegates to the underlying Lebesgue implementation and then wraps the resulting components back on the Sobolev domain. This saves the caller from handling the dual/primal lifting themselves.

### 5. Cap / ball integrals avoid overloading the path machinery

`geodesic_integral(...)` is built from Dirac evaluations along a curve, which is appropriate for ray or path physics. `geodesic_ball_integral(...)` is a region integral built from a volume form. Keeping both public APIs makes the intent clear and avoids forcing users to express region averaging in terms of unrelated point-evaluation machinery.

### 6. The new API defines a reusable abstraction boundary

The additions are not just “one more way” to get a number. They introduce reusable extension points:

- `geodesic_ball_quadrature(...)` as a geometry hook for region quadrature,
- `geodesic_ball_integral(...)` / `geodesic_ball_average(...)` as generic region-functional constructors,
- `spherical_cap_integral(...)` / `spherical_cap_average(...)` as exact sphere-specific overrides.

This is a coherent public abstraction that did not exist before.

## What each new function does

### `SymmetricHilbertSpace.geodesic_ball_quadrature(center, radius, n_points)`

- Low-level geometry hook for region quadrature.
- Returns quadrature points and weights for a geodesic ball.
- Default implementation raises `NotImplementedError`; geometries add their own rule when possible.
- Analogous to the pre-existing `geodesic_quadrature(...)`, but for regions rather than paths.

### `SymmetricHilbertSpace.geodesic_ball_integral(center, radius, n_points=None, normalize=False)`

- High-level generic constructor for a geodesic-ball linear form.
- Represents either
  $$
  u \mapsto \int_{B(center,radius)} u\,dV
  $$
  or, with `normalize=True`, the average over that ball.
- Uses `geodesic_ball_quadrature(...)` when an exact override is not available.

### `SymmetricHilbertSpace.geodesic_ball_average(center, radius, n_points=None)`

- Convenience wrapper for the normalized version of `geodesic_ball_integral(...)`.
- Makes the “average over a geodesic ball” concept explicit in the public API.

### `SymmetricSobolevSpace.geodesic_ball_integral(...)`

- Sobolev-domain override.
- Delegates region-functional construction to the underlying Lebesgue space when possible.
- Rewraps the resulting coefficients as a `LinearForm` on the Sobolev domain.
- This is what makes the cap/ball API available directly on Sobolev spaces without requiring the caller to reason about formal adjoints or domain lifting.

### `sphere.Lebesgue.geodesic_ball_quadrature(center, radius, n_points)`

- Deterministic quadrature rule for a spherical cap.
- Uses Gauss-Legendre nodes in the polar variable and uniform azimuthal rings.
- Includes the physical area element in the weights so the sum of weights equals the cap area.
- Provides a geometry-aware fallback when an exact spectral formula is not requested.

### `sphere.Lebesgue.spherical_cap_integral(center, angular_radius, normalize=False)`

- Exact truncated spectral constructor for a spherical-cap integral.
- Computes the spherical-harmonic coefficients of the cap indicator using `pyshtools.SHCoeffs.from_cap`.
- Handles both raw integrals and normalized averages.
- Handles physical sphere-radius scaling correctly.

### `sphere.Lebesgue.spherical_cap_average(center, angular_radius)`

- Convenience wrapper for the normalized cap-average functional.
- Public API for the common “average over a spherical cap” use-case.

### `sphere.Lebesgue.geodesic_ball_integral(center, radius, n_points=None, normalize=False)`

- Sphere-specific override connecting physical geodesic balls to spherical caps.
- Converts physical radius to angular radius.
- Uses the exact cap formula when `n_points` is omitted.
- Falls back to geodesic-ball quadrature when `n_points` is provided.

## Relationship to `l2_products_operator(...)`

There is a useful equivalence:

- If the exact cap weight `w_C` is already available as a space vector, then
  `space.l2_products_operator([w_C])` reproduces the cap integral or average.
- On Sobolev spaces, the lifted `l2_products_operator(...)` preserves the same forward `L^2` pairing while fixing the adjoint for the Sobolev geometry.

So the right interpretation is:

- `l2_products_operator(...)` is the **evaluation / batching primitive**,
- cap / ball methods are the **geometry-aware constructors** of the corresponding weights or linear forms.

That means the new API is not mathematically independent of the old primitives, but it is still essential as a public, reusable, geometry-driven layer.

## Recommended interpretation for developers

- Keep `geodesic_integral(...)` and `path_average_operator(...)` as the 1-D path API.
- Keep `geodesic_ball_integral(...)` / `geodesic_ball_average(...)` as the generic region API.
- Keep `sphere.Lebesgue.spherical_cap_integral(...)` / `spherical_cap_average(...)` as the exact sphere-specific constructors.
- Use `l2_products_operator(...)` when batching several already-known weighting fields, including cap weights once they have been constructed.

In short: the new functionality should be viewed as **geometry-aware construction of region functionals**, not as a duplicate of the older pairing primitives.