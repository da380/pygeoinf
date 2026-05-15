## Plan Complete: Port Convex Analysis to port_convex_analysis Branch

Ported the convex-analysis stack from the source branch into pygeoinf across six phases, covering the foundational space/subset changes, support-function surface, optimisation layer, Backus-Gilbert integration, plotting helpers, public exports, packaging, and the final reviewer guide. The archived validation state for the plan is 624 passing tests with 1 expected xfail, with all major convex-analysis classes importable from the package surface.

**Phases Completed:** 6 of 6
1. ✅ Phase 1: Port Foundational Changes
2. ✅ Phase 2: Port Support-Function Core
3. ✅ Phase 3: Port Optimisation Layer
4. ✅ Phase 4: Port Problem Integration
5. ✅ Phase 5: Port Visualisation & Public Surface
6. ✅ Phase 6: Reviewer Guide & Final Validation

**All Files Created/Modified:**
- `CONVEX_ANALYSIS_REVIEWER_GUIDE.md`
- `pygeoinf/pygeoinf/hilbert_space.py`
- `pygeoinf/pygeoinf/nonlinear_forms.py`
- `pygeoinf/pygeoinf/direct_sum.py`
- `pygeoinf/pygeoinf/linear_optimisation.py`
- `pygeoinf/pygeoinf/subsets.py`
- `pygeoinf/pygeoinf/convex_analysis.py`
- `pygeoinf/pygeoinf/convex_optimisation.py`
- `pygeoinf/pygeoinf/backus_gilbert.py`
- `pygeoinf/pygeoinf/subspaces.py`
- `pygeoinf/pygeoinf/plot.py`
- `pygeoinf/pygeoinf/__init__.py`
- `pygeoinf/pyproject.toml`
- `pygeoinf/tests/test_direct_sum.py`
- `pygeoinf/tests/test_halfspaces.py`
- `pygeoinf/tests/test_mass_weighted.py`
- `pygeoinf/tests/test_support_function_constructors.py`
- `pygeoinf/tests/test_support_function_algebra.py`
- `pygeoinf/tests/test_subgradient.py`
- `pygeoinf/tests/test_proximal_bundle.py`
- `pygeoinf/tests/test_level_bundle.py`
- `pygeoinf/tests/test_smoothed_lbfgs.py`
- `pygeoinf/tests/test_chambolle_pock.py`
- `pygeoinf/tests/test_bundle_core.py`
- `pygeoinf/tests/test_qp_backends.py`
- `pygeoinf/tests/test_solve_support_values.py`
- `pygeoinf/tests/test_dual_master_cost.py`
- `pygeoinf/tests/test_subspaces.py`
- `pygeoinf/tests/test_plot.py`
- `pygeoinf/tests/test_subsets.py`

**Key Functions/Classes Added:**
- `HyperPlane`, `HalfSpace`, `PolyhedralSet`
- `SupportFunction` and its concrete/algebraic subclasses
- `SubgradientDescent`, `ProximalBundleMethod`, `LevelBundleMethod`, `SmoothedLBFGSMethod`, `ChambolleChockMethod`
- `QPBackend`, `OSQPBackend`, `ClarabelBackend`, `solve_support_function_values`
- `DualMasterCostFunction`
- `AffineSubspace.get_tangent_basis()`, `AffineSubspace.from_hyperplanes()`, `AffineSubspace.to_hyperplanes()`
- `SubspaceSlicePlotter`, `plot_slice()`

**Test Coverage:**
- Total tests written: 236 new tests
- All tests passing: ✅

**Recommendations for Next Steps:**
- Use `CONVEX_ANALYSIS_REVIEWER_GUIDE.md` as the entry point for future review or regression work on the convex-analysis surface.