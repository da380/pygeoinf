# Re-review YG — symmetric spaces, geometry, plotting (2026-08-29)

Tests: `test_sphere_transform, test_bounded_and_sphere, test_fourier_spaces, test_space_parity, test_observation, test_flexure, test_geometry, test_plotting` — **597 passed, 6 deselected (slow), 9.9 s**. Scripts in `work/review_r2/YG/` (`prof1…7_*.py`, logs beside them). All timings `OMP_NUM_THREADS=1`, interleaved medians, machine shared with other agents (absolute numbers ±30%; ratios are what I report).

## 1. Review items — status

| item | status | verified by | note |
|---|---|---|---|
| Y Must-1 box geodesics | done | `fourier.py:548-616,784-863`, `box.py:173-192`; `test_space_parity` runs all six geometries | `Box._separation` is straight-line, not periodic — correct |
| Y Must-2 / D-11 order guard | done | `base.py:1260-1297`, used at `:1321,:1411`; path/ball ops pass `unsafe=True` with a reason | guard also blocks `covariance_function` on L2 (§2.3) |
| Y Must-3 finufft threads, `eps`/`nthreads` pass-through | done | `sphere.py:811-862`, `base.py:1358-1440`; `test_space_parity` "passes them down"; direct route 15–70× slower than NUFFT at n=2000 (prof2) so threshold 200 is right | docstring's "1 thread faster at every size" is false at 10⁵ points on a 512² torus: 4 threads 26 ms vs 41 ms (prof5, noisy) |
| Y Must-4 `DHaj` | done | `sphere.py:667-715`; first `_quadrature` at lmax 256 is 297 ms then cached (prof1) | |
| Y Must-5 `with_degree`/transfer on boxes | done | `fourier.py:698-773` | |
| Y Must-6 coefficient access | done | `sphere.py:514-581`, `fourier.py:618-665`, `base.py:247-364` | |
| Y Must-7 KD-tree pairs | done | `base.py:1058-1118` | |
| Y Must-8 `bincount` | done | `base.py:953` | |
| Y Must-9 taper | done | `box.py:235-317` | |
| Y Must-10 degree conversion | done | `sphere.py:984-1038` | docstring promises a `ValueError` for |lat|>90 that is never raised (§2.4) |
| Y Should-11 abstract primitives | done | `base.py` `@abstractmethod` on all; `test_space_parity::test_each_geometric_primitive_is_required` | |
| Y Should-12 caches keyed on lmax | done | `sphere.py:79-117,667-715`; `with_order` 0.5 ms at lmax 256 (prof1) | |
| Y Should-13 lift round trip | done | `operators.py:150-172,1150-1180`; counts: multiplication fwd 2 / adj 6, inclusion fwd 0 / adj 6 (v1: 0/4, 2/8) | adjoint still 4 structural transforms (§3.5) |
| Y Should-14 / D-12 path integral + heuristic | done | `base.py:1842-1963`; parity test "path integral of one is its length" | |
| Y Should-15/16/17/18/19/20/21 | done | `base.py:1136-1199,877-911`; `sphere.py:443-512,1131-1136`; `fourier.py:449-451`; `box.py:319-344`; `gaussian.py:1212,1255` | |
| Y Should-22/23 docs, DESIGN fixes | done | `DESIGN.md:20,1978,3259` | |
| Y Consider-24 `sampling` | done (D-1) | `sphere.py:160` | |
| Y Consider-25 vectorised `project_function` | not done | `sphere.py:924-940` loop; 55 ms at lmax 128 | minor |
| Y Consider-26 derivative codomain `order-1` | not done | `fourier.py:879-938` codomain is `self` | design choice, documented |
| Y Consider-28 box `pointwise_variance` at the origin | not done | ran: `(6,4)` box, unit spectrum: origin 24.000, random point 23.564 (2% high); odd axes agree | `base.py:532` "provably the same everywhere" still false on even axes |
| Y Consider-29 `_read_table` cache | not done | 1–2 ms per call | irrelevant |
| Y Consider-30 parity test | done | `tests/test_space_parity.py` | |
| D-1 `SHGrid` vectors, `sampling=1` | done | `sphere.py:333-384` | `from_grid_values(copy=False)` aliases (§2.5) |
| D-2 `(lat, lon)` degrees | done | `sphere.py:17-32`, every point method converts at its boundary; plotting takes degrees | |
| D-3 submodules with `Lebesgue`/`Sobolev` subclasses | done | `circle.py, line.py, plane.py, torus.py, box.py, sphere.py:1645-1702`; `_coordinate_key` tagged by geometry not class (`sphere.py:277-284`) | |
| D-9 `heat_measure(length_scale)` | done | `base.py:478-502,688-723` | |
| G M1 `Polytope.project` Dykstra | done | `convex.py:386-417,478-509` | |
| G M2 `Ball(radius=0)` | done | ran: constructs; `backus.py:118` | |
| G M3 `contains(rtol)` on surfaces | done | `convex.py:1057,1159` | |
| G M4 density grid resolution | done differently | `distributions.py:128-160` fine grid per curve, unioned | |
| G M5 `plot_corner(fill=True)` density | done | `distributions.py:548-575` | |
| G M6 catalogue rows | not done | `V1_CATALOGUE.md:197` (`HalfSpaceSupportFunction` "Ported"), `:238` (`LevelSet` "Ported" — no such class anywhere in `pygeoinf2/`), `:344` (`config.py` = plotting switch) all still false | |
| G M7 `title`, legend, rename table | done | `distributions.py:8-24,203,387-388,605-620` | |
| G S1 `ConvexIntersection`, `support_bound` | done; `Ball/Ellipsoid.boundary` not restored | `convex.py:1180-1300`; only `HalfSpace.boundary` exists (`:753`) | |
| G S2 kernel equation kept, `with_translation` keeps it | done | `subspaces.py:447-458,561-581` | |
| G S3 `Ellipsoid.project` Newton | done | `convex.py:906-975` | |
| G S4 `condition(solver=)` | done | ran: signature has `solver` | |
| G S5 one `(A A*)⁻¹` | not done | `subspaces.py:180,284,482` still three builds | |
| G S6 fused `value_and_maximiser` | not done | no such method in `geometry/convex.py`, `numerics/convex.py`, `backus.py` | |
| G S7 sphere plot options; `plot_points/paths` dispatched | done / not done | `plotting/sphere.py:52-83` has extent, contours, gridline intervals, borders, rivers, title; `plot_points` is a plain `Sphere`-only function (`:212`), no box version; defaults still `viridis`/`colorbar=True`/`gridlines=False` (flipped from v1, undocumented) | |
| G S8 `plot_error_bounds`; `full=` | done / not done | `plotting/fourier.py:103-173`; no padding view | |
| G S9 dense metric in geometry tests | partial | `test_geometry.py:445` one parametrised case | |
| G S10 `HalfSpace.contains` scaling | done | `convex.py:757-781` | |
| G C1 `plot_slice`, C2–C8 | not done | REVIEW §10 records C1 open; `dimension()` still `dim` applications (`subspaces.py:255`); no `support_function` cache | |
| REVIEW §10 Phase-3 note "flexural 50 fwd / 54 adj" | confirmed | prof1 counts: 50 / 54 (v1 24 / 28) | see §3.2 |

## 2. Bugs and regressions found now

**2.1 `inverse_flexural_operator` on a Sobolev space with varying coefficients is wrong and fails** *(verified: prof7)*. `base.py:1756-1758` lifts the L2 operator and claims `POSITIVE_DEFINITE`, which closes to `SELF_ADJOINT` (`traits.py:54`); the lifted operator is `G⁻¹AG`-adjoint, not self-adjoint in `H^s`. Sphere lmax 24, `H²`, `D = 2 + heat sample`: `(Fx,y)=238.35`, `(x,Fy)=−1.16`; `check_traits` fails; CG raises `ConvergenceError` after 1250 iterations (residual 230). Circle 64 `H²`: CG "converges" to relative residual **1.2e-4** at `rtol=1e-8` — a silent wrong answer where the operator is nearly symmetric. v1 has the same defect (`CG numerical breakdown`), so inherited, but v2 now *asserts* the false trait. Only Lebesgue spaces are tested (`test_flexure.py:308-318`); examples 20/22 use L2 or a constant rigidity. Fix (verified): `lift_formal_adjoint(base.inverse_flexural_operator(...), self)` — residual 1.1e-7, adjoint identity exact.

**2.2 `Sphere.walk_from` past the antipode returns latitudes below −90°** *(verified)*. `sphere.py:965-978`: `walk_from([20, 40], [1.5π])` → `[-250., 40.]`. The direct route then evaluates `f(2π−θ, φ)` and the NUFFT route `f(2π−θ, φ+π)`: on a non-zonal field they differ by 1.02 on a field of max 0.47 (agreement 1.6e-11 before the pole). `covariance_function` from the north pole hides it (zonal field). Fix: reflect through the pole (`θ → 2π−θ, φ → φ+π`) or clip distances to `πR`. Flagged in the appendix (§3 "dead code / errors"); not fixed.

**2.3 `covariance_function` raises on every Lebesgue space** *(verified: prof3)*. `base.py:956-967` → `two_point_covariance` → `dirac` → D-11 guard. v1's `invariant_covariance_function` (`pygeoinf/symmetric_space/sphere.py:730`) worked on L2 by the addition theorem. The function is a property of the measure's spectrum (`Σ_k s_k φ_k(p)φ_k(q)/g_k`, convergent whenever the spectrum decays), not of the space's order; the closed form needs no representer. Same fix as §3.6.

**2.4 `to_colatitude_radians` docstring promises a `ValueError` for latitudes outside `[-90, 90]` that the code never raises** *(verified: `sphere.py:1003-1005` vs `:1007-1018`; `[-126, 0]` → `[[3.77, 0]]`)*. That check would have caught 2.2.

**2.5 `from_grid_values` aliases the caller's array** *(verified)*. `sphere.py:370` `SHGrid.from_array(array, copy=False)`; `axpy` (`:376-379`) then mutates the caller's array. v1's `from_array` copied by default. In-place API only (`add` copies), but `_as_field(ndarray)` at `base.py:1590` wraps a user array this way.

**2.6 Box `pointwise_variance` "same everywhere" claim** — see Consider-28 row: 2% high on a `(6,4)` grid, `O(1/n)` per even axis; `pointwise_std` calibration inherits the bias.

Unverified suspicion: `_path_nodes` uses `self.length_scale` (`base.py:1854`), which on a Lebesgue space is the constructor default `1.0` and means nothing there, so path operators built with `unsafe`/on L2 get an arbitrary node count.

## 3. Optimisations, ranked by gain × confidence

**3.1 Sphere `plot`: 20–70× (verified prototype).** `plotting/sphere.py:117-135` closes the seam by appending 360° and passes a 0–360 mesh with `shading="auto"`; every cell straddling ±180° (1028 at lmax 128) sends cartopy into `_wrap_quadmesh` → `pcolor` → per-polygon `_attach_lines_to_boundary` (5.2 of 5.4 s in the profile). Measured: v2 1481 ms vs v1 232 ms at lmax 128 (2951 vs 494 at 256), i.e. v2 is 6× slower than v1. Rolling the columns to `[-180, 180)` and passing explicit flat cell edges clipped to `[-90, 90]` gives 20 ms (Robinson, lmax 128; 71×), 55 ms at 256, 11 ms PlateCarree, 33 ms Mollweide; rendered image identical (0.00% of pixels differ by >30/255). Risk: none beyond a half-cell shift already inherent in `shading="auto"`. Applies to every map pyslfp draws.

**3.2 `flexural_operator` in components: ~8× per application (estimated from counts).** prof1: 50 transforms forward, 54 adjoint (v1 24/28); 166/179 ms at lmax 128 vs v1 96/113 ms. The cause is `laplacian(multiply(·))` pairs (`base.py:1572-1574,1648-1672`): each `multiply` truncates (2 transforms) and each `laplacian` re-analyses what was just synthesised. Analysis is linear, so all grid terms sharing a spectral multiplier can be summed on the grid and analysed once: `A(w)`, `S(λc)`, `S(λ²c)`, one analysis of the plain terms, one of the terms inside an outer `L`, one final synthesis — 6 transforms. Estimated 166 → ~25 ms at lmax 128, and every preconditioned-CG iteration of `inverse_flexural_operator` (prof7: 996+807 transforms per solve at lmax 24) shrinks with it. Risk: moderate; the L2 self-adjointness test and the constant-coefficient closed form already exist to catch errors. Phase-3 note called this "a different and larger" job; it is ~40 lines.

**3.3 Invariant-measure sampling: 2.8× per draw (verified prototype).** prof3: `sobolev_measure.sample` = 1 analysis + 2 syntheses (white noise synthesised at `spaces.py:447`, then `DiagonalLinearOperator._value` `diagonal.py:100-102`); v1 `_kl_sample` is one synthesis. 10.3 ms vs 3.6 ms (lmax 128), 68 vs 24 ms (256). Fix: in `invariant_measure` (`base.py:635-649`) pass `sample=lambda rng: self.from_components(sqrt(variances / metric_values) * standard_normal)`; `_rebuild` carries `sample=` (`gaussian.py:1423-1452`) so scale/translate keep it. Risk: none (`check_white_noise`-style test exists). Every prior sample and `samples(n)` benefits.

**3.4 Exact cap averages: 70× construction (verified prototype).** `sphere.py:1344-1351` calls `SHCoeffs.from_cap` (a rotation, 8.5 ms/centre at lmax 128) then `from_derivative_components` (`operators.py:1935-1961`) synthesises a representer nobody reads (1 `MakeGridDH`), and `.derivative_components` on that functional re-analyses it (`operators.py:1869`): 15 ms/centre, 1.44 s per 100 centres. Closed form `2πR² I_l(cos α) · basis_matrix(centres)` with `I_l = (P_{l-1}−P_{l+1})/(2l+1)` agrees to 7.6e-13 and costs 20 ms per 100. Also make `from_derivative_components` lazy (store `g`, synthesise on `.representer`) — it saves a synthesis per `dirac` too.

**3.5 Path-operator construction: ~11× (verified prototype).** `_path_operator` (`base.py:1982-2004`) for 2000 paths / 64k nodes: 1.8–3.3 s, of which `_to_point` per node (`sphere.py:1063-1075`, 64k calls) and `leggauss` per path (`:1147`, 0.8 s for 2000 calls of the same few counts) are ~2.5 s. Vectorised node generation (one `arctan2`/`arccos` over the node array, `lru_cache` on `leggauss`) reproduces nodes to 4e-13° and weights exactly: 304 ms vs 3343 ms in the same run. Box `geodesic_quadrature` (`fourier.py:611`) and `geodesic_ball_quadrature` (`:1204-1214` per-node `_to_point`) have the same shape.

**3.6 `covariance_function` and `pointwise_variance_at` for diagonal covariances: 40–150× (verified prototype).** prof3: `covariance_function` 23 ms (lmax 128) / 123 ms (256) for 50 distances via representer + covariance + NUFFT; the Legendre closed form (`legval` over per-degree `s_l/g_l`) is 0.5 / 1.2 ms and agrees to 1.1e-14 — and does not need the Dirac, fixing §2.3. `pointwise_variance_at` exact route (`base.py:1016-1025`) is 5 transforms per point (17 ms/pt at lmax 128, 116 ms/pt at 256); for a `DiagonalLinearOperator` covariance `Σ_k s_k φ_k(p)²/g_k` via `basis_matrix` is 0.15 ms/pt (agrees to all printed digits). Dispatch on `isinstance(measure.covariance, DiagonalLinearOperator)`; the general path stays for posteriors.

**3.7 Cache the point conversion in `point_evaluation_operator`: 1.4× sphere, 2× torus, 10× Box (verified).** `_angles` runs per application: sphere 14.5 of 37 ms at 10⁵ points (prof2); torus 32 of 61 ms (prof5); `Box._angles` (`box.py:154-164`) loops `_to_enclosing` per point: 230 of 254 ms (91%). `point_evaluation_operator` (`base.py:1418-1424`) should convert once and hand arrays to `evaluate`/`accumulate`; `path_integral_operator` inherits it (64k nodes per apply). Risk: none.

**3.8 Sphere `evaluate`: drop the analysis done only for the south pole: 1.3× at lmax 256 (verified prototype).** `_double` (`sphere.py:775`) runs `to_components(x)` (23 ms at lmax 256, 35% of a 67 ms forward; 3 of 33 ms at 128) to get `f(π)`. `f(π) = K · row_means(grid)` with `K_j = 2π a_j Σ_l Ȳ_l0(θ_j) Ȳ_l0(π)` precomputable per lmax (calibrated once as `_quadrature` is): agrees to 1e-14, 0.05 ms.

**3.9 `lift_formal_adjoint` adjoint: 6 → 3 transforms for `multiplication_operator` (estimated).** `operators.py:1163-1172` synthesises the reweighted vector, the base adjoint (`multiply`) analyses it, truncates, synthesises, and the lift analyses again. A components-in/components-out hook on operators built by `LinearOperator.self_adjoint`/`DiagonalLinearOperator` would cut the adjoint to synth–product–analysis(–synth). Adjoint 19.7 vs v1 13.4 ms at lmax 128. Lower priority than 3.2, which subsumes the flexure case.

Not worth doing (<10% or opt-in): `np.add.at` vs assignment in the box spectrum (0.4 ms); `eps=1e-8` halves the box NUFFT (3.2 vs 6.6 ms at 10⁴) but is a user choice; `dense=True` breaks even with NUFFT only at n≈2000 (prof2); `project_function` 55 ms.

## 4. Open questions for the user

1. **Truncating `multiply`** (DESIGN 21.7) costs 2 transforms per product and is what makes v2's flexure 2× v1's transform count. 3.2 keeps the public semantics and fuses internally; alternatively `multiply` could return an untruncated grid and rely on the fact that every consumer that leaves the grid analyses anyway. Which?
2. Should `covariance_function`/`pointwise_variance(_at)` be *allowed* on a Lebesgue space (the measure's spectrum decides convergence), i.e. is D-11 a guard on the space or on the operation? 3.6 assumes the former.
3. The sphere `plot` defaults flipped from v1 (`RdBu`→`viridis`, `colorbar=False`→`True`, `gridlines=True`→`False`, PlateCarree→Robinson) without a recorded reason (D-10 rule). Keep or restore?
