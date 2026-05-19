## Plan Complete: Exact Ball and Ellipsoid Plotters

**Status:** All 4 phases complete. 25 tests passing in `tests/test_plot_exact_quadratic_slices.py`. Exact quadratic path implemented in `pygeoinf/plot.py` for both `Ball` and `Ellipsoid` in 1D, 2D, and 3D slices.

---

## Plan: Exact Ball and Ellipsoid Plotters

Add exact slice rendering for `Ball` and `Ellipsoid` subsets in `pygeoinf` so they no longer fall back to rasterized membership sampling on 1D, 2D, and 3D affine slices. The implementation will extend the existing `SubspaceSlicePlotter` fast-path architecture in `pygeoinf/plot.py` by pulling quadratic sets back to slice coordinates, rendering the resulting interval / ellipse / ellipsoid directly, and preserving sampled fallback only for degenerate or unsupported cases.

**Phases: 4**

1. **Phase 1: Ball exact slices in 1D and 2D**
    - **Objective:** Add an exact fast path for `Ball` slices on lines and planes, avoiding `grid_size^n` membership sampling.
    - **Files/Functions to Modify/Create:**
      - `pygeoinf/pygeoinf/plot.py` — extend dispatch in `SubspaceSlicePlotter.plot()`, add helpers for quadratic pullback on a slice, add exact 1D and 2D ball rendering.
      - `pygeoinf/tests/test_plot_exact_quadratic_slices.py` — new focused test file for exact quadratic-set rendering.
    - **Tests to Write:**
      - `test_ball_1d_exact_slice_returns_interval_payload`
      - `test_ball_2d_exact_slice_returns_ellipse_boundary_points`
      - `test_ball_exact_path_bypasses_membership_sampling`
      - `test_ball_exact_slice_respects_plot_bounds`
    - **Steps:**
      1. Write the four tests against the current plotting API and run the new test file to confirm the missing fast path fails.
      2. Implement a helper that expresses a ball slice in local slice coordinates as a quadratic inequality and reduces 1D slices to clipped intervals.
      3. Implement exact 2D ellipse rendering from the pulled-back quadratic form and dispatch balls to that renderer.
      4. Run the new test file again until it passes.

2. **Phase 2: Ellipsoid exact slices in 1D and 2D**
    - **Objective:** Generalize the exact quadratic renderer from balls to ellipsoids using the ellipsoid shape operator in slice coordinates.
    - **Files/Functions to Modify/Create:**
      - `pygeoinf/pygeoinf/plot.py` — extend the quadratic pullback helper to `Ellipsoid`, add exact 1D / 2D ellipsoid rendering.
      - `pygeoinf/tests/test_plot_exact_quadratic_slices.py` — add ellipsoid-specific tests.
    - **Tests to Write:**
      - `test_ellipsoid_1d_exact_slice_matches_quadratic_roots`
      - `test_ellipsoid_2d_exact_slice_matches_membership_on_boundary`
      - `test_ellipsoid_exact_path_bypasses_membership_sampling`
      - `test_ellipsoid_degenerate_plane_slice_falls_back_or_raises_consistently`
    - **Steps:**
      1. Add the four ellipsoid tests and run the focused test file to observe failures.
      2. Implement the exact pullback of `⟨A(x-c), x-c⟩ <= r^2` onto local coordinates and reuse the phase-1 renderer where possible.
      3. Handle numerically degenerate 2D slices explicitly, with the behavior fixed by tests.
      4. Re-run the focused test file until all ball and ellipsoid tests pass.

3. **Phase 3: Exact 3D slices and backend integration**
    - **Objective:** Add exact 3D rendering for ball and ellipsoid slices, matching the current Matplotlib and Plotly backend split used by `PolyhedralSet`.
    - **Files/Functions to Modify/Create:**
      - `pygeoinf/pygeoinf/plot.py` — add exact 3D quadratic rendering helpers for Matplotlib and Plotly, integrate them into the existing backend resolution flow.
      - `pygeoinf/tests/test_plot_exact_quadratic_slices.py` — add exact 3D and backend behavior tests.
    - **Tests to Write:**
      - `test_ball_3d_exact_slice_returns_surface_payload`
      - `test_ellipsoid_3d_exact_slice_returns_surface_payload`
      - `test_exact_quadratic_slice_uses_plotly_backend_when_requested`
      - `test_exact_quadratic_3d_path_avoids_sampling_warning`
    - **Steps:**
      1. Add the four 3D/backend tests and run the focused test file to confirm the gap.
      2. Implement exact 3D rendering by parameterizing the unit sphere and mapping it through the pulled-back quadratic slice geometry.
      3. Wire the new exact path into both backends without changing the public `plot_slice()` signature.
      4. Re-run the focused test file until it passes.

4. **Phase 4: Regression coverage, docs, and notebook touch-up**
    - **Objective:** Validate the new fast path against existing plotting behavior, document the new dispatch rules, and update the Gaussian tutorial to use a sensible default now that exact ellipsoid plotting exists.
    - **Files/Functions to Modify/Create:**
      - `pygeoinf/pygeoinf/plot.py` — docstring updates for exact quadratic rendering.
      - `pygeoinf/tutorials/gaussian_measure_to_sets_demo.ipynb` — reduce or remove the oversized `grid_size` emphasis in the plotting cell.
      - `pygeoinf/docs/agent-docs/references/living/*-reference.md` — update if living references exist on this branch.
    - **Tests to Write:**
      - `test_plot_slice_keeps_polyhedral_exact_path_unchanged`
      - `test_non_quadratic_subsets_still_use_sampled_path`
    - **Steps:**
      1. Add the two regression tests and run the focused plotting test file to confirm the baseline behavior.
      2. Update docstrings and the tutorial notebook only after the exact path is in place.
      3. Run the focused plotting test file, then the relevant `pygeoinf` test subset, then the full `pygeoinf` test suite.
      4. Update living references if any are present for this package on the current branch.

**Design Decisions (resolved)**
1. **2D payload:** return boundary points as `np.ndarray` of shape `(N, 2)`; filled polygon is optional, exposed via a `fill=True` parameter on the internal renderer and propagated through `Subset.plot()` in Phase 4.
2. **Degenerate slices:** fail explicitly with a `ValueError` (no sampled fallback for degenerate quadratic slices); degenerate means `rho_sq < 0` (slice is empty) or `rho_sq == 0` (a single point, not a renderable region).
3. **Scope:** narrow to `Ball` and `Ellipsoid` only; the quadratic helper will be a private method, not a general API surface.