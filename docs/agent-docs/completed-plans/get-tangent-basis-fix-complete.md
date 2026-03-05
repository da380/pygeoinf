## Plan Complete: Fix `get_tangent_basis()` for Non-Axis-Aligned Subspaces

The bug in `AffineSubspace.get_tangent_basis()` — which wrongly reported dimension 3 for any diagonal 1D subspace — has been fixed with a two-phase tolerant Gram-Schmidt algorithm. Five regression tests covering axis-aligned, diagonal, and oblique cases were added. The demo notebook was updated to use a genuine diagonal direction, removing the workaround caveat.

**Phases Completed:** 1 of 1
1. ✅ Phase 1: Fix, test, and update notebook

**All Files Created/Modified:**
- `pygeoinf/pygeoinf/subspaces.py` — `get_tangent_basis()` replaced with tolerant GS
- `pygeoinf/tests/test_subspaces.py` — five new regression tests added
- `pygeoinf/convex_analysis_demos/subspace_slice_plotter_demo.ipynb` — cell 5 updated to use diagonal direction, workaround comment removed

**Key Functions/Classes Added:**
- `AffineSubspace.get_tangent_basis()` (rewritten) — two-phase tolerant Gram-Schmidt using `domain.basis_vector`, `domain.copy`, `domain.axpy`, `domain.ax`, `domain.norm`, `domain.inner_product`

**Test Coverage:**
- `test_get_tangent_basis_axis_aligned_1d` — axis-aligned line → dim 1 ✅
- `test_get_tangent_basis_axis_aligned_2d` — xy-plane → dim 2 ✅
- `test_get_tangent_basis_diagonal_1d` — diagonal (1,1,1)/√3 → dim 1 (the regression) ✅
- `test_get_tangent_basis_diagonal_2d` — oblique plane → dim 2, orthonormal basis ✅
- `test_get_tangent_basis_oblique_1d_with_translation` — translation does not affect dimension ✅
- Total: 9 tests in `test_subspaces.py`, all passing ✅

**Recommendations for Next Steps:**
- Consider adding a `tolerance` parameter to `get_tangent_basis()` if higher-dimensional spaces with near-degenerate projectors are anticipated.
- The notebook could add an example demonstrating `SubspaceSlicePlotter` with a truly oblique 2D cutting plane (not just a translated coordinate plane) as a showcase of the fix.
