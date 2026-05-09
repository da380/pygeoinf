## Plan: Port Convex Analysis to port_convex_analysis Branch

Port all convex-analysis features from the `convex_analysis` branch onto `port_convex_analysis` in logical dependency order, excluding agentic files and development playground material. Each phase includes tests and a full test-suite run. A developer-facing reviewer guide is produced at the end.

**Decisions:**
- Reviewer guide location: repository root (`CONVEX_ANALYSIS_REVIEWER_GUIDE.md`)
- Packaging: include only optional extras needed by the CA stack (osqp, clarabel, plotly)
- README: keep minimal; reviewer guide provides detail

**Exclusions (not ported):**
- `docs/agent-docs/` (entire directory)
- `AGENTS.md`
- `pygeoinf/testing_sets/` (notebooks, PNGs, mp4, scripts)
- `symmetric_space_new/`, `rough_work/`
- Branch-local markdown files (`main-vs-convex-analysis-diff.md`, `diff.md`, `convex-analysis-branch-changes.md`)

**Phases: 6**

1. **Phase 1: Port Foundational Changes**
    - **Objective:** Bring foundational module improvements that the CA stack depends on.
    - **Files/Functions to Modify/Create:**
      - `pygeoinf/hilbert_space.py` — `axpy()` return type (Vector), `distance()` method on HilbertSpace
      - `pygeoinf/nonlinear_forms.py` — subgradient infrastructure (`SubgradientResult`, `has_subgradient`, `subgradient()` on NonlinearForm)
      - `pygeoinf/subsets.py` — `HyperPlane`, `HalfSpace`, `PolyhedralSet` classes
      - `pygeoinf/direct_sum.py` — `zero`, `inner_product`, `random` method completions on DirectSumSpace
      - `pygeoinf/linear_optimisation.py` — Bug 2 bisection convergence fix
    - **Tests to Write:** Port corresponding test changes from CA branch for these modules (existing test files updated)
    - **Steps:**
      1. Switch to `port_convex_analysis` branch
      2. For each file, diff CA vs port_convex_analysis and apply changes
      3. Port updated/new test assertions for foundation modules
      4. Run full test suite — all pre-existing tests must still pass
      5. Verify no regressions

2. **Phase 2: Port Support-Function Core (convex_analysis.py)**
    - **Objective:** Add the entire `convex_analysis.py` module — 9 SupportFunction classes plus algebraic operations.
    - **Files/Functions to Modify/Create:**
      - `pygeoinf/convex_analysis.py` (new, ~913 lines) — `SupportFunction`, `IndicatorSupportFunction`, `ZeroSupportFunction`, `NormSupportFunction`, `QuadraticSupportFunction`, `LinearSupportFunction`, `AffineImageSupportFunction`, `InfimalConvolutionSupportFunction`, `SupremumSupportFunction`
    - **Tests to Write:**
      - `tests/test_support_function_constructors.py`
      - `tests/test_support_function_algebra.py`
    - **Steps:**
      1. Copy `convex_analysis.py` from CA branch
      2. Copy associated test files
      3. Run support-function tests — verify all pass
      4. Run full test suite — no regressions

3. **Phase 3: Port Optimisation Layer (convex_optimisation.py + tests)**
    - **Objective:** Add the convex-optimisation solvers and QP infrastructure.
    - **Files/Functions to Modify/Create:**
      - `pygeoinf/convex_optimisation.py` (new, ~2205 lines) — `SubgradientMethod`, `ProximalBundleMethod`, `LevelBundleMethod`, `SmoothedLBFGSMethod`, `ChambolleChockMethod`, plus `QPBackend`, `OSQPBackend`, `ClarabelBackend`, `solve_support_function_values`
    - **Tests to Write:**
      - `tests/test_subgradient.py`
      - `tests/test_proximal_bundle.py`
      - `tests/test_level_bundle.py`
      - `tests/test_smoothed_lbfgs.py`
      - `tests/test_chambolle_pock.py`
      - `tests/test_bundle_core.py`
      - `tests/test_qp_backends.py`
      - `tests/test_solve_support_values.py`
    - **Steps:**
      1. Copy `convex_optimisation.py` from CA branch
      2. Copy all 8 test files
      3. Run optimisation tests — verify all pass
      4. Run full test suite — no regressions

4. **Phase 4: Port Problem Integration (backus_gilbert, subspaces)**
    - **Objective:** Port the higher-level integration changes that use the CA stack.
    - **Files/Functions to Modify/Create:**
      - `pygeoinf/backus_gilbert.py` — `DualMasterCostFunction` class (replaces HyperEllipsoid approach)
      - `pygeoinf/subspaces.py` — `get_tangent_basis`, `from_hyperplanes`, `to_hyperplanes` static/class methods
    - **Tests to Write:**
      - `tests/test_dual_master_cost.py`
      - `tests/test_halfspaces.py`
    - **Steps:**
      1. Diff and apply changes to `backus_gilbert.py` and `subspaces.py`
      2. Copy new test files
      3. Run integration tests — verify all pass
      4. Run full test suite — no regressions

5. **Phase 5: Port Visualisation & Public Surface**
    - **Objective:** Port plotting additions and update `__init__.py` exports + packaging.
    - **Files/Functions to Modify/Create:**
      - `pygeoinf/plot.py` — `SubspaceSlicePlotter`, `plot_slice()` method
      - `pygeoinf/__init__.py` — export all new CA classes
      - `pyproject.toml` — add optional extras for osqp, clarabel, plotly
    - **Tests to Write:** Port any plot-related test updates
    - **Steps:**
      1. Diff and apply changes to `plot.py`, `__init__.py`, `pyproject.toml`
      2. Run full test suite — no regressions
      3. Verify imports work: `from pygeoinf import SupportFunction, SubgradientMethod` etc.

6. **Phase 6: Reviewer Guide & Final Validation**
    - **Objective:** Write the developer-facing reviewer guide and complete full validation.
    - **Files/Functions to Modify/Create:**
      - `CONVEX_ANALYSIS_REVIEWER_GUIDE.md` (new, at repo root)
    - **Tests to Write:** None (documentation phase)
    - **Steps:**
      1. Write comprehensive reviewer guide covering:
         - Overview: what convex analysis adds and why
         - Architecture: dependency diagram showing layer ordering
         - Per-component sections: purpose, dependencies, dependents, key classes/functions
         - Test coverage summary
         - Optional dependencies (osqp, clarabel) and how to install
      2. Run full test suite one final time
      3. Verify test count matches expectations

**Open Questions:** None (all resolved)
