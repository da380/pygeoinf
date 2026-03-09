## Plan: Support-Function Algebra and Notebook Refactor

Add reusable support-function composition to pygeoinf so admissible-region constructions can be expressed declaratively rather than with handwritten support oracles. The implementation will expose both explicit wrapper classes and convenience methods/operators, include nonnegative scaling, and then refactor the intervalinf DLI notebook to use the new algebra directly.

**Phases 4**
1. **Phase 1: Core Constructors and Public Entry Points**
    - **Objective:** Introduce first-class support-function wrappers for user-defined support maps and singleton sets, and expose them through both concrete classes and convenience constructors on `SupportFunction`.
    - **Files/Functions to Modify/Create:** `pygeoinf/pygeoinf/convex_analysis.py`; tests under `pygeoinf/tests/` covering callable and point supports.
    - **Tests to Write:** callable support value evaluation; callable support with optional `support_point`; point support value evaluation; point support fixed support-point semantics; `subgradient` fallback through `support_point`.
    - **Steps:**
        1. Write failing tests for callable support functions and singleton-set support functions.
        2. Implement `CallableSupportFunction` with optional support-point callback and strict Hilbert-space consistency checks.
        3. Implement `PointSupportFunction` for $h(q)=\langle q, p\rangle$ and expose it as a singleton-set constructor.
        4. Add convenience constructors such as `SupportFunction.callable(...)` and `SupportFunction.point(...)`.
        5. Run the targeted tests and confirm the red-green cycle is closed before moving on.

2. **Phase 2: Algebraic Composition and Scaling**
    - **Objective:** Add composable support-function algebra for translation, linear-image pullback, Minkowski sums, and nonnegative scaling, with both explicit classes and convenience methods/operators.
    - **Files/Functions to Modify/Create:** `pygeoinf/pygeoinf/convex_analysis.py`; tests under `pygeoinf/tests/` covering algebraic composition.
    - **Tests to Write:** linear-image adjoint pullback identity; translation by a point; Minkowski-sum value identity; positive-scalar scaling; zero-scaling behavior; rejection of negative scaling; space/domain mismatch checks.
    - **Steps:**
        1. Write failing tests for each algebraic operation independently.
        2. Implement `LinearImageSupportFunction`, `MinkowskiSumSupportFunction`, and `ScaledSupportFunction`.
        3. Add convenience methods `image(...)`, `translate(...)`, and `scale(...)` on `SupportFunction`.
        4. Add operators `__add__`, `__mul__`, and `__rmul__` so declarative notebook expressions read naturally.
        5. Re-run the targeted tests and confirm the composed values match the dual identities.

3. **Phase 3: Support-Point Propagation and Edge Cases**
    - **Objective:** Make the new algebra robust by propagating support points and subgradients through compositions wherever that is mathematically justified, while handling unbounded or unavailable cases safely.
    - **Files/Functions to Modify/Create:** `pygeoinf/pygeoinf/convex_analysis.py`; tests under `pygeoinf/tests/` covering support-point propagation and extended-real-valued behavior.
    - **Tests to Write:** translated support-point propagation; Minkowski-sum support-point addition; scaled support-point propagation; linear-image support-point propagation where representable; `None` propagation when support points are unavailable; `+inf` behavior for unbounded directions.
    - **Steps:**
        1. Write failing tests for support-point propagation and failure cases.
        2. Implement `support_point(...)` on the new wrappers with conservative semantics.
        3. Ensure inherited `subgradient(...)` behavior remains correct through the new composition classes.
        4. Tighten docstrings to state precisely when a support point is returned and when `None` is expected.
        5. Re-run the targeted tests and resolve any semantic gaps before notebook refactoring.

4. **Phase 4: Declarative Notebook Refactor**
    - **Objective:** Replace the handwritten admissible-region support oracle in the reduced DLI notebook with direct composition of support-function objects built from the new pygeoinf algebra.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/old_demos/pli_demos/bg_with_errors_minkowski.ipynb`; any supporting imports required by the new API.
    - **Tests to Write:** notebook execution checks for the admissible-region cells; containment checks for the true property and estimator point; consistency checks for the polyhedral approximation after refactor.
    - **Steps:**
        1. Refactor the notebook cells so the admissible-region support is expressed as a point term plus transformed support objects rather than a handwritten function.
        2. Keep the refactor operator-first, using composed pygeoinf abstractions only.
        3. Re-run the affected notebook cells in order and confirm the admissible region is still constructed successfully.
        4. Update the explanatory markdown so the notebook describes the algebraic composition rather than the old manual support oracle.
        5. Stop after review with a phase summary and commit message before any further work.

**Open Questions**
1. None at present; the approved decisions are to expose both classes and convenience methods, include nonnegative scaling, and keep the notebook refactor within this same plan.