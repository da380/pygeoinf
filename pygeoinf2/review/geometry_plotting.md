# Review: geometry (sets, convex sets, subspaces) and plotting — v1 → v2

> **Note (2026-08-27):** the decisions recorded in `pygeoinf2/REVIEW.md` §11 (D-1 … D-13) override the Must/Should/Consider ranking below wherever they conflict — in particular D-1 (sphere vectors are `SHGrid`, `sampling=1` default), D-2 (points in `(lat, lon)` degrees), D-3 (per-geometry submodules with `Lebesgue`/`Sobolev` subclasses), D-4 (`from_matrix(..., form=)`), D-6 (parallel hooks around operators), D-12 (path *integral* operator), D-13 (convex solvers restored).

Scope read: v1 `pygeoinf/subsets.py`, `subspaces.py`, set parts of `convex_analysis.py`, `plot.py`, `config.py`, the `plot*`/`create_map_figure` functions in `symmetric_space/{sphere,plane,torus,circle,line}.py`; v2 `pygeoinf2/geometry/*`, `pygeoinf2/plotting/*`, the callers in `inference/backus.py`, `inference/point.py`, `probability/gaussian.py`, the tests, and pyslfp's usage. Numerical claims below marked *(verified)* were checked with a throwaway script.

---

## 1. Functionality: retained / extended / lost

### 1.1 Sets (`geometry/sets.py`)

| v1 (`subsets.py`) | v2 | Status |
|---|---|---|
| `Subset` :35 — `domain`, `is_empty`, `is_element(rtol)`, `boundary` (abstract), `complement`, `intersect`, `union`, `plot` | `Subset` sets.py:29 — `domain`, `contains(rtol)`, `__contains__`, `complement()`, `intersect`, `union`, `&`/`|`/`~` | **Renamed** `is_element→contains`. **Lost**: `is_empty` :74, `boundary` :99 (dropped from the base; now only `HalfSpace.boundary` property and `AffineSubspace.boundary()` method exist), `plot` :179. **Added** operators. |
| `EmptySet` :256, `UniversalSet` :278, `Complement` :296 | sets.py:92, :104, :116 | Retained. |
| `Intersection` :333, `Union` :390 | sets.py:144, :179 | Retained, flattening added. **Lost**: the v1 `intersect` fast path :158 that returned a `ConvexIntersection` when every part is convex. In v2 an intersection of two `ConvexSet`s is a plain `Subset`: no `project`, no `indicator`, so it cannot be used as a constraint in `ProximalGradient`. The catalogue row "ConvexIntersection — Subsumed: `Intersection` of convex sets, which reports itself convex" is **false**; there is no convexity branch anywhere in sets.py. |
| `SublevelSet` :432, `LevelSet` :496 | — | **Lost.** Catalogue says "Ported … arrives with M5 stage 5.8"; `grep -rn LevelSet pygeoinf2` finds nothing. |
| `ConvexSubset.check` :683 (randomised convexity check) | — | Lost (minor). |

### 1.2 Convex sets (`geometry/convex.py`)

| v1 | v2 | Status |
|---|---|---|
| `ConvexSubset` :536 (`support_function` property, `directional_bound`, `closure`/open sets, `support_fn` cache) | `ConvexSet` convex.py:54 — `project` (abstract), `contains` default via projection :66, `support_function()` :76, `support_maximiser` :82, `indicator()` :93, `from_support_function` :102, `__add__` (Minkowski) :131, `translate` :144 | **Extended**: indicator/prox tie-in, Minkowski sums, oracle sets. **Lost**: open/closed distinction (acceptable), `_support_fn` caching (`support_function()` allocates a new object per call). |
| `Ball` :1131 (open by default, `radius>=0`) | `Ball` :419 (`radius>0` enforced :435) | Retained + `project`. **Latent bug**: `inference/backus.py:171` does `Ball(problem.data_space, radius=0.0)` on the error-free path of `_harden`; this raises `ValueError` *(verified)*. |
| `Ellipsoid` :968 (centre, radius, operator `A`, optional `inverse_operator`, `inverse_sqrt_operator`; `normalized`, `boundary`, `directional_bound`) | `Ellipsoid` :613 (precision, centre, optional covariance; `mahalanobis_squared`, `support_maximiser`, `support_function`, `translate`; `project` raises :697) | Radius folded into precision (fine). **Lost**: `boundary` → `EllipsoidSurface` link, `inverse_sqrt_operator` route for `h(q)` (v1 convex_analysis.py:345-364 used `A^{-1/2}` when only that is available). **On `project` raising**: v1 had *no* projection onto ellipsoids either (v1 `ConvexSubset` has no `project`; only `HyperPlane`, `HalfSpace`, `AffineSubspace` did), so this is parity, not regression. But the catalogue note ("worth keeping with an eye to constrained optimisation") means an ellipsoid projection *will* be wanted; it is a 1-D secular equation in λ with one `(I + λP)` solve per Newton step — see recommendation S3. |
| `NormalisedEllipsoid` :1071 | — | Subsumed; user scales precision by `1/r²` and covariance by `r²` themselves. |
| `Sphere` :1198 / `EllipsoidSurface` :1093 | `BallSurface` :750 (contains, project, `sample`), `EllipsoidSurface` :827 (contains only) | Retained. **Bug**: both declare `contains(x, *, tolerance=…)` :785, :864 instead of the abstract `rtol`. `Intersection([Ball, BallSurface]).contains(x)` raises `TypeError` *(verified)*. `EllipsoidSurface` has no `project`/`sample`. |
| `HyperPlane` :1222 (`normal_vector`, `offset`, `normal_norm`, `distance_to`, `project`, `dimension`) | `Hyperplane` :499 (`normal`, `offset`, `contains`, `project`) | Retained. Lost `distance_to`, `dimension`. No support function (v1 didn't have one either). |
| `HalfSpace` :1354 (`inequality_type` "<=" / ">=", `distance_to`, `project`-onto-boundary :1483, `support_function` :1547 via `HalfSpaceSupportFunction`) | `HalfSpace` :554 ("<=" only; `boundary` property; metric `project` :595) | **`project` semantics changed and the change is right**: v2 is the metric projection (idempotent, identity on the set) *(verified on a dense-metric space)*; v1's mapped every point to the boundary hyperplane, which is not a projection. Catalogue row "HalfSpaceSupportFunction — Ported: `HalfSpace`" is **false**: v2 `HalfSpace` has no `support_function` (inherits the `NotImplementedError` at :76). Lost `">="` (negate the normal — fine but undocumented), `distance_to`. |
| `PolyhedralSet` :1578 | `Polytope` :300 with `outer: bool` (required), `__and__` :369, `contains` :343, `project` :350 (cyclic projections) | Extended with inner/outer status (§18.4). **`project` is not a projection** — see §5. |
| `SupportFunction.value_and_support_point` (convex_analysis.py:80, fused for ellipsoid :394) | — | **Lost**: v2 `_EllipsoidSupport._value` :735 and `_maximiser` :741 each apply the covariance; backus.py:932-946 calls both per bundle iteration → 2 covariance applications where v1 needed 1. |
| `CallableSupportFunction`, `MinkowskiSumSupportFunction`, `LinearImageSupportFunction`, `ScaledSupportFunction` | `_OracleSupport` :254, `_MinkowskiSupport`, `_ImageSupport`, `_ScaledSupport` (numerics/convex.py:335-390) | Retained. `_MinkowskiSum` :265 and `_Translated` :149 do not implement `support_maximiser` (only the support functions do, via `subgradient`). |

### 1.3 Subspaces (`geometry/subspaces.py`)

| v1 (`subspaces.py`) | v2 | Status |
|---|---|---|
| `OrthogonalProjector` :26 (`complement` property :49, `from_basis(orthonormalize=True)` :63) | :39 (`complement()` :92, `basis()` :78, `from_basis(orthonormal=False)` :108, `onto_kernel(solver)` :144) | Retained + traits (`SELF_ADJOINT|IDEMPOTENT` → PSD by closure, traits.py:57) + `basis()` (replaces v1 `get_tangent_basis` :183). |
| `AffineSubspace.__init__(projector, translation, constraint_operator, constraint_value, solver, preconditioner)` :109 | `__init__(projector, *, translation)` :185; equation and solver only recorded by `from_linear_equation` :270-271 | `solver`/`preconditioner` are now attributes of the solver object; a preconditioner goes in via `CGSolver(preconditioner=…)` (numerics/solvers.py:522). Acceptable. |
| `translation`, `projector`, `tangent_space`, `get_tangent_basis` | `translation`, `projector`, `tangent` :217, `projector.basis()` | Retained. |
| `has_explicit_equation` :216, `constraint_operator` :221, `constraint_value` :233 | :363, :374, :385 | Retained but v1 fell back to `(I-P)`, `(I-P)x0` when no equation; v2 raises `AttributeError`. Defensible (§21.16) but `LinearSubspace.from_kernel` :483 **does not record** `(A, 0)` even though it obviously has one, so `K.constraint_operator` raises. |
| `pseudo_inverse` :246, `projection_operator` :266, `boundary` :279 | :405, :417, :431 | Retained (**catalogue Part 2 is stale**: it still says "Not ported"). `boundary` semantics changed: v1 returned `EmptySet`, v2 returns `self`. |
| `from_linear_equation` :290, `from_tangent_basis` :335, `from_complement_basis` :374, `from_hyperplanes` :437, `to_hyperplanes` :504 | :250, :277, :297, :317, :343 | Retained (catalogue stale again). `from_complement_basis` no longer records an explicit constraint operator (v1 did :420-434), so conditioning on it needs `to_hyperplanes` first. `from_tangent_basis` lost the `orthonormalize` flag. |
| `with_translation` :590 (kept the equation, recomputed `w`) | :391 | **Regression**: drops the equation and solver; `has_explicit_equation` becomes False after a translation. |
| `with_constraint_value` :613 | :395 | Retained. |
| `is_element` :570 | inherited `ConvexSet.contains` :66 | Retained. |
| `condition_gaussian_measure(prior, geometric=False)` :634 (Bayesian via `LinearBayesianInversion` with the stored solver/preconditioner, or geometric via `affine_mapping(projector)`) | `GaussianMeasure.condition(operator, value, noise=None)` gaussian.py:814 | **Moved, and weakened**: `condition` does `np.linalg.inv(normal.matrix(form="components"))` (:836) — dense in the constraint codomain, no solver/preconditioner argument. pyslfp (Heathcote2026/grace_utils.py:274-277) does `LinearSubspace.from_kernel(op).condition_gaussian_measure(prior)`; the v2 equivalent is `prior.condition(op, op.codomain.zero())` — usable because that codomain has 3 coefficients, but the subspace object no longer participates and there is no matrix-free route. The geometric variant is `prior.push_forward(subspace.projection_operator())` if `push_forward` accepts an `AffineOperator` (not verified). |
| `LinearSubspace` :706 — `complement` :720, `from_kernel(operator, solver, preconditioner)` :732, `from_basis` :757, `from_complement_basis` :781 | :444 — `complement()` :454, `from_kernel(operator, /, *, solver)` :483, `from_basis` :469 | Retained. pyslfp's `LinearSubspace.from_kernel(constraint_operator)` (positional only) is source-compatible. `from_complement_basis` on `LinearSubspace` dropped (use `AffineSubspace.from_complement_basis`). |
| `dimension` | :227 | **New in v2.** The catalogue row "v1's `dimension` used the trace formula … returned 169" is a **misattribution**: v1 has no `AffineSubspace.dimension` (only `HyperPlane.dimension`, subsets.py:1342); DESIGN §16.3 says the 169 was the v2 author's own draft. |

### 1.4 Plotting

**Field plotting (`plotting/base.py`, `sphere.py`, `fourier.py`)**

| v1 | v2 | Status |
|---|---|---|
| `sphere.create_map_figure(figsize, projection, **kwargs)` :1910 | `plotting.subplots(space, *, rows=1, columns=1, projection=None, **kwargs)` sphere.py:33-50; box variant fourier.py:16 | **The requested `subplots`-like overload exists and does multi-panel** (examples/20_flexure.py:86, 21_tomography.py:114). Differences to note: keywords are `rows`/`columns` not plt's `nrows`/`ncols`; default projection changed PlateCarree→Robinson; dispatches on the space, so there is no space-free way to make a map figure (pyslfp keeps its own `subplots` at pyslfp/plot.py:20-43). |
| `sphere.plot(u: SHGrid, *, ax, projection, contour, cmap="RdBu", coasts, rivers, borders, map_extent, gridlines=True, gridlines_kwargs, symmetric: bool|float, contour_lines, contour_lines_kwargs, num_levels, colorbar=False, colorbar_kwargs, **kw)` :1939 | `plot(space: Sphere, field: ndarray, *, ax, cmap="viridis", symmetric: bool, vmin, vmax, colorbar=True, colorbar_label, coasts, gridlines=False, **kw)` sphere.py:53 | **Lost**: `contour`/`contourf` mode, `contour_lines`, `levels`/`num_levels`, `map_extent` (v2 always `set_global()` :118), `rivers`, `borders`, `projection` when `ax is None`, `gridlines_kwargs` with `lat_interval`/`lon_interval` and cartopy formatters (v1 :2000-2016), `colorbar_kwargs`, `symmetric` as a float scale factor, tuple-`ax` unwrapping (`_unwrap_axes` :1926). Defaults flipped (cmap, colorbar, gridlines, projection). Input is an ndarray of `grid_shape` instead of an `SHGrid` (pyslfp passes `SHGrid`, pyslfp/plot.py:66). **Gained**: `vmin`/`vmax`, seam closure :104-107, shape validation. |
| `sphere.plot_points(points=(lat,lon) deg, *, data=, cmap, color, s, marker, symmetric, colorbar, coasts, gridlines, map_extent…)` :2037 | `plot_points(space, points=(colat,lon) rad, *, ax, marker, size, color, **kw)` sphere.py:133 | **Convention change** (degrees lat/lon → radians colat/lon) that will silently misplace pyslfp's tide gauges. **Lost**: `data=` colouring with colormap/colorbar/symmetric, coasts/gridlines/extent. |
| `plot_geodesic(p1, p2)` :2133 | — | Lost. |
| `plot_geodesic_network(paths, plot_sources, plot_receivers, source_kwargs, receiver_kwargs)` :2156 | `plot_paths(space, paths, *, count, color, linewidth, alpha)` :174 | Retained minus source/receiver markers (v1 :2178-2214). Dateline splitting is an improvement. |
| `plane.plot(space, u, *, contour, …, full)` :757, `torus.plot` :1229 | `plot(space: PeriodicBox|Box, field)` fourier.py:26 | 1-D line and 2-D `pcolormesh` retained. **Lost**: `contour`/`contour_lines`/`num_levels`, `full` (show the tapered padding, plane.py:841-851), `colorbar_kwargs`. |
| `plane/torus.plot_points`, `plot_geodesic`, `plot_geodesic_network` :861-1010, :1298-1450 | — | **Lost**: `plot_points`/`plot_paths` are Sphere-only free functions, not dispatched. |
| `circle/line.plot_error_bounds(space, u, u_bound)` :787/:773 | — | **Lost** entirely. |
| `line.plot(…, full=False)` :741 | fourier 1-D branch :77-79 | `full` lost. |

**Distribution plotting (`plotting/distributions.py`)**

| v1 (`plot.py`) | v2 | Status |
|---|---|---|
| `plot_1d_distributions(posterior_measures, *, prior_measures, true_value, show_true_value_in_legend, ax, xlabel, title, prior_labels, posterior_labels, width_scaling=6, legend_position, fill_density, **kw)` :28 | `plot_densities(posterior, *, prior, truth, index, ax, labels, prior_labels, width=6, fill, samples, rng, xlabel)` :131 | Renamed kwargs: `true_value→truth`, `posterior_labels→labels`, `width_scaling→width`, `fill_density→fill`. **Lost**: `title`, `legend_position`, `show_true_value_in_legend`, acceptance of scipy frozen distributions (:106-113), **adaptive grid resolution** (:143-152, see §5). **Gained**: `index`, sampled measures. pyslfp calls (joint_inversion.py:105-112, :749-756; grace_inversion.py:320-327) pass `title=`, `true_value=`, `posterior_labels=` → `TypeError` under v2. |
| `plot_corner_distributions(posterior_measure, *, prior_measure, true_values, show_true_value_in_legend, labels, title, figsize, colormap, contour_color, parallel, n_jobs, width_scaling=3.75, legend_position, fill_density, num_sigmas=3)` :1788 | `plot_corner(posterior, *, prior, truth, labels, axes, figsize, sigmas=3, width=3.75, fill, colormap, colour, samples, rng)` :296 | Renamed: `prior_measure→prior`, `true_values→truth`, `contour_color→colour`, `num_sigmas→sigmas`, `fill_density→fill`. **Lost**: `title` (pyslfp passes `title=""` in every call: joint_inversion.py:862-890, :1243; grace_inversion.py:407 → `TypeError`), the **legend** (v1 :2118-2140 put "Posterior Mean"/"True Value" in the empty top-right panel; v2 draws none), `show_true_value_in_legend`, `legend_position`, 1-D case (v2 raises :331), the colourbar for `fill`, and the prior as a **secondary x-axis in σ_prior units** (v1 :1915-1935) — replaced by a dotted prior density on a tickless twin axis. `parallel`/`n_jobs` dropped (fine). **Gained**: `axes=` passing, sampled measures with mass-matched contour levels :418-436, `moments()` helper :41. |
| `SubspaceSlicePlotter` :241, `plot_slice` :1708, `Subset.plot` :179 (exact polyhedral slice via `HalfspaceIntersection` + Chebyshev LP :1438-1560; exact quadratic slice via pulled-back metric :1037-1123; sampled fallback; plotly 3-D backend :791) | — | **Not ported** (catalogue: Planned). |
| `config.py` | — | Catalogue says "v1's plotting backend switch". **False**: v1 `config.py` is `DATADIR`/`CACHEDIR` (dataset paths, `PYGEOINF_CACHE_DIR`); the backend switch is `SubspaceSlicePlotter._resolve_backend` (plot.py:791). v2 has neither. |

---

## 2. Algorithmic performance

- **`OrthogonalProjector.from_basis`** (subspaces.py:108-141): orthonormalises once at construction; each application is `k` inner products + `k` axpys. Same as v1's tensor-product operator. Good.
- **`onto_kernel`** (:144-175): one `(A A*)^{-1}` solve per application, CG by default (`rtol=1e-12`, no preconditioner unless the caller builds `CGSolver(preconditioner=…)`). v1 defaulted to a dense Cholesky of `A A*` (factor once, cheap per application, O(m³) once). v2's default is the right one for large `m`, but there is no way to *cache* a factorisation except by passing a direct solver. Fine.
- **`from_linear_equation`** (:250-274) builds `(A A*)` + solver **twice** — once at :260 for the translation and once inside `onto_kernel` at :167-168 — and `pseudo_inverse` (:414-415) a third time. With a direct solver that is two or three factorisations of the same operator; v1 computed `G_inv` once and reused it (subspaces.py:315-322). Same three lines duplicated in three places.
- **`dimension`** (:227-247): `dim` applications of `P`; for a kernel projector that is `dim` CG solves. Documented as a diagnostic. For large spaces a Hutchinson estimate (`E[z·Pz]` in components) is O(10–50) applications — worth offering.
- **`basis()`** (:78-90) / **`to_hyperplanes`** (:343-360): `dim` projector applications then orthonormalisation of `dim` vectors, O(dim·rank) inner products. `to_hyperplanes` orthonormalises `dim` vectors to extract `codim` — same as v1's `get_tangent_basis`.
- **`Polytope.contains`** (:343): `m` inner products. **`Polytope.project`** (:350): up to `200·m` half-space projections — and gives the wrong point (§5).
- **`Intersection`**: no `project` at all (not a `ConvexSet`), so no alternating-projection or Dykstra option exists. v1 had none either, but v1's `ConvexIntersection` gave a combined max-functional with gradient/Hessian (subsets.py:740-780) usable by smooth constrained solvers.
- **`Ellipsoid.contains`**: one precision application; when the precision is a solver-wrapped inverse (backus.py:346, :424) that is one linear solve per membership test — unavoidable, matrix-free, fine.
- **Support value + maximiser**: `_EllipsoidSupport._value` (:735) and `_maximiser` (:741) each apply `C`; the bundle loop in backus.py:932-946 wants both per direction. v1's `value_and_support_point` (convex_analysis.py:394-444) fused them. 2× the dominant cost in route (d).
- **`support_function()`** allocates a new functional per call (:483, :718); v1 cached it. Harmless unless called in a loop — backus.py:932 calls it once, fine.
- **Plotting**: sphere `plot` computes lat/lon arrays and one `pcolormesh` per call — same as v1, no resampling, fine. `plot_paths` samples `count=24` nodes per path via `geodesic_quadrature` — fine. `moments()` on a Gaussian (distributions.py:63-65) builds the dense Galerkin matrix through `as_multivariate_normal` (gaussian.py:806): `dim` covariance applications + `dim²` Gram solves — fine on a property space, but `plot_densities(index=i)` on a large-space measure pays the whole dense cost for one variance (v1 did the same, plot.py:104). `moments()` on a sampled measure takes `samples=20000` sequential draws (:76-78); with a randomise-then-optimise posterior each draw is a solve (DESIGN §30.4), so the default is 20 000 solves with no warning.

---

## 3. Code practice and quality

**Hierarchy coherence**
- `AffineSubspace` is a `ConvexSet` (good), but `Intersection` of convex sets is not (sets.py:144) — the algebra is not closed under the one operation constraints are built from.
- `BallSurface.contains(*, tolerance)` / `EllipsoidSurface.contains(*, tolerance)` (convex.py:785, :864) violate the abstract signature `Subset.contains(*, rtol)` (sets.py:45) → runtime `TypeError` inside `Intersection`/`Union`/`Complement` *(verified)*.
- `Polytope.project(x, *, iterations=200)` (:350) adds a keyword the abstract `project(x, /)` (:58) does not have; `_SetIndicator.prox` (:407) calls it positionally, so `iterations` is unreachable from the proximal path.
- `boundary` is a property on `HalfSpace` (:585) and a method on `AffineSubspace` (:431); absent on `Ball`, `Ellipsoid`, `Hyperplane`, `Polytope`, and the base.
- `LinearSubspace.complement()` (:454) shadows `Subset.complement()` with a different meaning (orthogonal vs set complement). Documented, and v1 did the same, but `~K` (`__invert__`, sets.py:77) now returns the orthogonal complement of a subspace — surprising.
- `_OracleSet.contains` and `_MinkowskiSum.contains` raise `NotImplementedError` (:208, :289), so `x in set` raises; `Subset.__contains__` does not document that.

**Duplication**
- `Ball.support_maximiser` (:442) ≡ `_BallSupport._maximiser` (numerics/convex.py:313); `Ellipsoid.support_maximiser` (:656) ≡ `_EllipsoidSupport._maximiser` (:741). One should delegate to the other.
- `Ball.indicator` (:489) returns `numerics.convex.BallIndicator`, while the generic `_SetIndicator(Ball)` would be equivalent — two indicator classes for one set.
- `(A @ A.adjoint).with_traits(PD)` + `(solver or CGSolver(rtol=1e-12))(normal)` appears at subspaces.py:167-168, :260-261, :414-415.
- pyslfp/plot.py:20-43 `subplots` duplicates v2's `subplots` — can be deleted once pyslfp moves.

**Reach-through / hygiene**
- `from_linear_equation` pokes `subspace._equation`, `subspace._solver` after construction (:270-271) instead of constructor kwargs; `from_kernel` (:483-487) then forgets to do the same, which is why the kernel subspace loses its equation.
- `dimension()` re-imports `require_coordinates` locally (:238) though it is imported at :29.
- `onto_kernel` keeps `_ = codomain  # named for the docstring's sake` (:174) — dead code.
- `Polytope.contains` (:343-348): `plane.contains(...) if hasattr(plane, "contains") else False` — a guard for a type error the constructor should have raised; `__init__` (:310-331) never checks that `half_spaces` are `HalfSpace`s.
- `AffineSubspace.__init__` (:185-204) no longer checks `translation ∈ domain` (v1 did, subspaces.py:135-137).
- sphere.py:122 `ax.gridliner = …` sets an ad-hoc attribute on a matplotlib object (inherited from v1).

**Type hints**: geometry uses `Any` for every vector (53 occurrences in convex.py, 13 in subspaces.py) while the rest of v2 is generic (`LinearOperator[X, Y]`, `GaussianMeasure[X]`). `Subset` should be `Subset[X]`; `from_hyperplanes(hyperplanes: Sequence[Any])` (:317), `Polytope.half_spaces -> tuple` (:334), `projection_operator() -> Any` (:417) should name their types.

**Optional dependencies**: handled well — matplotlib, cartopy and scipy.stats are imported inside the drawing functions (sphere.py:20-30, :43; distributions.py:113, :118, :167, :322); `plotting/__init__.py` imports `symmetric_space.sphere`, which defers pyshtools to construction (sphere.py:57). `_require_cartopy` gives an install hint.

**Dispatch and extensibility**: `plot`/`subplots` are `functools.singledispatch` on the space type (base.py:24, :45), so a pyslfp space that subclasses `Sphere` dispatches by MRO with no registration (as `Box` does via `PeriodicBox`, test_plotting.py:47), and a genuinely new space registers with `@plotting.plot.register(MySpace)`. That is the right design. Two gaps: `plot_points`/`plot_paths` are plain Sphere-only functions, not dispatched, so planes/boxes have no point or path plotting; and nothing handles a `DirectSum` (draw each block).

---

## 4. Documentation gaps

Most public docstrings state purpose; the recurring gaps are missing `Args`/`Returns`/`Raises` and undocumented defaults.

- sets.py:59-78 `complement`, `intersect`, `union`, `__and__`, `__or__`, `__invert__`: no `Args`/`Returns`; operators have no docstring. `__contains__` (:55) should say it may raise for oracle/Minkowski sets.
- convex.py:76 `support_function`, :82 `support_maximiser`: no `Returns`/`Raises` sections; :93 `indicator` no `Returns`. :144 `translate` no `Args`. :131 `__add__` documents the maths but not the `ValueError`.
- convex.py:102-130 `from_support_function`: does not say that `contains`/`project` on the result raise, or that `outside`/`polytope` exist (they are only on the private `_OracleSet`).
- convex.py:310-331 `Polytope.__init__`: does not say elements must be `HalfSpace`; :350 `project` omits `iterations` from `Args` and does not state it is *not* the metric projection in the summary line ("The nearest point, by cyclic projection" is wrong).
- convex.py:626-654 `Ellipsoid.__init__`: `precision` must *claim* `SELF_ADJOINT|POSITIVE_DEFINITE` traits — stated only in the error text. :697 `project` suggests "a proximal method on the indicator's smooth surrogate", which does not exist in the codebase.
- convex.py:785, :864 `contains(*, tolerance)`: the keyword differs from the base and is not flagged.
- subspaces.py:48-70 `OrthogonalProjector.__init__`: `mapping: Any` untyped; :78 `basis()` no `Returns`; :144 `onto_kernel` docstring says "nothing claimed" but the code claims `POSITIVE_DEFINITE` (:167) — docstring and code disagree.
- subspaces.py:250-274 `from_linear_equation`: no `Args` (solver default CG `rtol=1e-12` unstated); :277 `from_tangent_basis`, :297 `from_complement_basis`, :317 `from_hyperplanes`: no `Args`/`Raises`; :343 `to_hyperplanes` no `Returns`; :391 `with_translation` does not say the equation is dropped; :405 `pseudo_inverse` does not state the solver used; :431 `boundary` returns `self` even for the whole space (wrong there).
- subspaces.py:483 `from_kernel`: one line; no `Args`, no note that the equation is not recorded.
- base.py:60 `colour_limits`: no `Args`/`Returns`.
- sphere.py:33 `subplots`: no `Args` (projection default Robinson, figsize heuristic); :133 `plot_points` says radians/colatitude — good, but must be flagged as a change from v1.
- distributions.py:296 `plot_corner`: `colormap`/`colour` say "for the filled and unfilled cases" but the filled case plots Mahalanobis distance, not density (§5); no `Raises` for the `size<2` and truth-length errors. :131 `plot_densities`: returns a tuple when priors are given — documented; `index` default 0 fine.
- Module docstrings sets.py, convex.py, subspaces.py are good.

---

## 5. Correctness concerns

**Metric-correct (checked):**
- `HalfSpace`/`Hyperplane`/`Ball`/`AffineSubspace` projections use only `inner_product`, `norm`, `axpy` — idempotent and in-set on the dense-metric fixture *(verified)*.
- `dimension()` uses the component trace (:244) — returns 2 for a rank-1 constraint on the dense-metric space *(verified)*; tested on Sobolev + Euclidean (test_geometry.py:284-292).
- `from_hyperplanes` builds the constraint as `from_vectors(...).adjoint` (:337-340), whose adjoint is `x ↦ [(n_i, x)]` — correct.
- `Ellipsoid.contains` = `(P(x−c), x−c)` in the space's inner product — correct given `P` is an operator; `credible_set` (gaussian.py:705-735) scales the operator covariance/precision by the χ² threshold — correct.
- `plot_corner`/`plot_densities` use `as_multivariate_normal` → `G⁻¹ C_gal G⁻¹` (gaussian.py:801-811), tested on the dense-metric space (test_plotting.py:145-170). Correct.
- backus.py:820-834 builds the inner `Polytope` from a `ConvexHull` of *component* arrays and wraps each normal with `space.representer` — consistent, since `(rep(n), x) = n·components(x)`.

**Defects:**
1. **`Polytope.project` is not the projection** (convex.py:350-367). Cyclic projection onto half-spaces converges to *a* feasible point. For `{x ≤ 0} ∩ {x+y ≤ 0}` and `x = (1, 0.5)` it returns `(−0.25, 0.25)` (dist² 1.625) while the nearest point is `(0, 0)` (dist² 1.25) *(verified)*. `indicator().prox` (:407) hands this to `ProximalGradient`, so a polytope constraint gives a wrong fixed point — precisely the "approximation under the name of an exact operation" DESIGN §16.4 says the module refuses for the ellipsoid. Dykstra's algorithm (same loop, with correction vectors) gives the true projection.
2. **`Ball(radius=0.0)` at backus.py:171** raises `ValueError` (convex.py:435) — the error-free branch of `BackusInference._harden` cannot run.
3. **`BallSurface`/`EllipsoidSurface` `contains(*, tolerance)`** breaks any set operation *(verified)*.
4. **`onto_kernel` claims `POSITIVE_DEFINITE` for `A A*`** (:167) — false for a rank-deficient `A`; the docstring says the opposite. CG still converged on a consistent RHS in my test, but the claim would let a Cholesky-type solver be selected on a singular operator, and a preconditioned CG with `rtol=1e-12` can stall on the null-space component.
5. **`plot_corner(fill=True)`** (distributions.py:405-443): in the Gaussian branch `field` is the Mahalanobis *distance* (:411-417), in the sampled branch it is the KDE *density* (:427-435); `contourf(field, cmap=colormap)` therefore paints the Gaussian case darkest *far from the mean* and the sampled case darkest *at* the mean. v1 filled density (plot.py:2011). All pyslfp calls pass `fill_density=False`, so they would not see it, but it is wrong.
6. **`plot_densities` grid resolution** (distributions.py:210): a fixed 2000 points over the union of all ±6σ windows. v1 (plot.py:143-152) scaled the grid to ≥25 points per σ of the narrowest peak, up to 10 000. With a prior 1000× wider than a posterior — the case DESIGN §30.3 says the twin axis exists for, and the case pyslfp works around with a zoom helper (joint_inversion.py:78-121) — the spacing is ~6 σ_post, so the posterior curve is aliased or missing. This regression is in the feature the port explicitly kept.
7. **`with_translation`** (:391) silently drops the constraint equation; `with_constraint_value` on the result then raises.
8. **`HalfSpace.contains`** (:589-593) scales tolerance by `max(|offset|, 1)`, independent of `‖x‖‖n‖`; `Hyperplane.contains` (:536-541) includes `‖x‖‖n‖`. Same predicate family, different tolerance semantics; the half-space one is not scale-invariant in `x`.
9. **`Intersection` of `ConvexSet`s is not a `ConvexSet`**, contrary to the catalogue; `(ball & half_space).indicator()` is an `AttributeError`.
10. **Geometry tests use only diagonal metrics** (test_geometry.py fixtures: `EuclideanSpace`, `Sobolev((16,),…)`, `OpaqueSpace` with a diagonal weight, doubles.py:74-80). No `make_dense_metric_space` — the memory rule for metric-sensitive code is not applied in this package (test_plotting.py:145-156 does apply it).
11. `BallSurface.project` and `sample` are metric-correct (white noise, `norm`); `EllipsoidSurface` has neither, so "used for sampling on a shell" only holds for the ball.

---

## Recommendations

### Must
- **M1** Replace `Polytope.project` (convex.py:350-367) with Dykstra's algorithm, or make `project` raise like `Ellipsoid.project` and expose the feasibility routine under a different name (`feasible_point`). Add `Polytope` to the `check_projection` tests. Until then `Polytope.indicator()` is a wrong prox.
- **M2** Fix backus.py:171: either allow `Ball(radius=0.0)` (a point; `project` returns the centre, `support_function` is the point support) or return `SupportFunction.of_point`-backed set on the error-free path. Add a test for `BackusInference` with `has_error=False`.
- **M3** Rename `tolerance→rtol` in `BallSurface.contains` (convex.py:785) and `EllipsoidSurface.contains` (:864); add a test that puts a surface inside an `Intersection`.
- **M4** `plot_densities` (distributions.py:210): size the grid from the narrowest σ — `n = clip(25 · span / σ_min, 2000, 10000)`, or evaluate each curve on its own window and plot with the shared limits. Add a test with σ_prior/σ_post = 1000 asserting the posterior peak height is within 1% of `1/(σ√2π)`.
- **M5** `plot_corner(fill=True)` (distributions.py:440-443): fill the *density* in both branches (`exp(−field²/2)` in the Gaussian branch), with levels matched to the sigma contours; restore v1's colourbar for the filled case.
- **M6** Correct the catalogue rows that are false: `ConvexIntersection` (not subsumed), `LevelSet`/`SublevelSet` (not ported), `HalfSpaceSupportFunction` (not ported), `config.py` (not a plotting switch; it is `DATADIR`/`CACHEDIR`), the `dimension` "169" attribution (v1 has no such method), and the stale "Not ported" rows in Part 2 for `from_tangent_basis`…`with_constraint_value`.
- **M7** For pyslfp: `plot_corner` and `plot_densities` need a `title` keyword (every pyslfp call passes it) and `plot_corner` needs its legend back (v1 plot.py:2118-2140). Document the kwarg renames (`prior_measure→prior`, `true_values→truth`, `posterior_labels→labels`, `fill_density→fill`, `num_sigmas→sigmas`, `width_scaling→width`, `contour_color→colour`) in one table.

### Should
- **S1** Make an `Intersection` whose parts are all `ConvexSet`s a `ConvexSet` (e.g. `ConvexIntersection(ConvexSet)` returned from `Subset.intersect`/`__and__`), with `project` by Dykstra over the parts and `support_function` raising or returning the `min_i h_i` **upper bound** under an honest name (v1 subsets.py:809). Restore `Ball.boundary`/`Ellipsoid.boundary` → the surfaces.
- **S2** Record the equation in `LinearSubspace.from_kernel` (subspaces.py:483) — it is `(A, 0)` — and make `with_translation` keep the equation with `value = A t` (as v1 subspaces.py:590-611). Move `_equation`/`_solver` into `AffineSubspace.__init__` keywords instead of post-hoc attribute writes.
- **S3** Provide `Ellipsoid.project(x, *, solver=None, tol=…)` by Newton on the secular equation `φ(λ) = (P y_λ, y_λ) − 1`, `y_λ = (I + λP)⁻¹(x − c)`, one solve per step; keep the raising behaviour only when no solver can be built. This is what the "constrained optimisation" note asks for.
- **S4** Give `GaussianMeasure.condition` (gaussian.py:814) a `solver=` argument and use it instead of `np.linalg.inv(normal.matrix(...))` at :836; add `AffineSubspace.condition(measure, *, solver=None)` (Bayesian) and document `measure.push_forward(subspace.projection_operator())` as the geometric variant, so pyslfp's `from_kernel(...).condition_gaussian_measure(prior)` has a one-line replacement.
- **S5** Build `(A A*)⁻¹` once in `from_linear_equation` and share it with `onto_kernel` and `pseudo_inverse` (subspaces.py:167, :260, :414); mark `A A*` `POSITIVE_SEMIDEFINITE` (it already is by the palindrome rule) rather than `POSITIVE_DEFINITE`, and fix the `onto_kernel` docstring to match the code.
- **S6** Add a fused `value_and_maximiser(y)` on `SupportFunction` (default: two calls) and override in `_EllipsoidSupport`; use it in backus.py:932-946. Make `Ball.support_maximiser`/`Ellipsoid.support_maximiser` delegate to the support function's `_maximiser` to remove the duplicate formulas.
- **S7** Sphere `plot`: restore `map_extent` (skip `set_global()` when given), `contour=`/`contour_lines=`/`levels=`, `gridlines_kwargs` with `lat_interval`/`lon_interval`, `colorbar_kwargs`, `rivers`/`borders`, and accept a `projection=` when `ax is None`. Either keep v1's `RdBu`/`colorbar=False` defaults or state the change in DESIGN. Make `plot_points` and `plot_paths` `singledispatch` on the space so boxes get them, and either accept `(lat, lon)` degrees or add a `units=` argument — the radians/colatitude convention will silently misplace pyslfp's gauges.
- **S8** Restore `plot_error_bounds` (a `fill_between` on the 1-D box renderer, fourier.py:77) and the `full=` padding view for `Box`.
- **S9** Add `make_dense_metric_space()` to the `X` fixtures in test_geometry.py for the projection, subspace and `dimension` tests, per the metric rule.
- **S10** Unify `contains` tolerance: `HalfSpace.contains` should scale by `max(|offset|, ‖n‖‖x‖, 1)` like `Hyperplane.contains`. Make `boundary` a method everywhere (or a property everywhere) and define it on the base as raising `NotImplementedError`.

### Consider
- **C1** Port `SubspaceSlicePlotter`/`plot_slice` as `plotting.plot_slice(subset, subspace, …)`: the exact quadratic path needs only `OrthogonalProjector.basis()`, `Ellipsoid.precision/centre`; the exact polyhedral path needs `HalfSpace.normal/offset` (with `Polytope`); keep the sampled fallback with the 3-D grid warning (v1 plot.py:341-360). Make the plotly backend a keyword, not a config module.
- **C2** `plot_corner`: offer the v1 secondary x-axis in σ_prior units as an option alongside the twin-axis prior density; support `size == 1` by delegating to `plot_densities`.
- **C3** `moments()`: warn (or cap) when `samples` draws each cost a solve; for `plot_densities(index=i)` on a Gaussian compute the single component variance `e_iᵀ G⁻¹ C_gal G⁻¹ e_i` instead of the dense matrix.
- **C4** `dimension()`: add `method="exact"|"hutchinson"` for large spaces.
- **C5** Parametrise `Subset[X]`/`ConvexSet[X]` and drop the `Any`s; type `Polytope.half_spaces` and `from_hyperplanes`.
- **C6** Register a `DirectSum` renderer that draws each block on a panel of `subplots(space, rows=…)`.
- **C7** Re-add `HalfSpace.support_function` (finite only along `+normal`; v1 convex_analysis.py:446-605) so a `Polytope` can at least return `min_i` as a labelled upper bound, and `Hyperplane.distance_to`/`HalfSpace.distance_to` (signed distance), which are one-liners on the existing `_residual`.
- **C8** Cache `support_function()` objects on `Ball`/`Ellipsoid` as v1 did.
