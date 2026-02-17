---
type: test-status
package: pygeoinf
python_version: "3.11-3.12 (project requires >=3.11)"
pytest_command: "pytest pygeoinf/tests --junitxml=reports/junit.xml -q"
timestamp: "2026-02-17T00:00:00Z"
last_run_by: "Atlas (orchestration agent)"
last_updated: "2026-02-17 (Phase 7 tests completed)"
total_tests: 28
failing_count: 13
critical_blocked_count: 10
optional_failing: 3
passing_tests: 15
new_tests_added: "test_halfspaces.py (35 tests, all passing)"
failing_tests:
  - id: "test_auxiliary.py::*"
    file: "pygeoinf/tests/test_auxiliary.py"
    line: 22
    function: "module import"
    short_message: "TypeError: 'type' object is not subscriptable"
    source_file: "pygeoinf/pygeoinf/visualization.py"
    source_line: 758
    priority: "P1"
    category: "import-error"
  - id: "test_direct_sum.py::*"
    file: "pygeoinf/tests/test_direct_sum.py"
    line: 10
    function: "module import"
    short_message: "TypeError: 'type' object is not subscriptable"
    source_file: "pygeoinf/pygeoinf/visualization.py"
    source_line: 758
    priority: "P1"
    category: "import-error"
  - id: "test_linear_forms.py::*"
    file: "pygeoinf/tests/test_linear_forms.py"
    line: 10
    function: "module import"
    short_message: "TypeError: 'type' object is not subscriptable"
    source_file: "pygeoinf/pygeoinf/visualization.py"
    source_line: 758
    priority: "P1"
    category: "import-error"
  - id: "test_linear_operators.py::*"
    file: "pygeoinf/tests/test_linear_operators.py"
    line: 12
    function: "module import"
    short_message: "TypeError: 'type' object is not subscriptable"
    source_file: "pygeoinf/pygeoinf/visualization.py"
    source_line: 758
    priority: "P1"
    category: "import-error"
  - id: "test_nonlinear_forms.py::*"
    file: "pygeoinf/tests/test_nonlinear_forms.py"
    line: 10
    function: "module import"
    short_message: "TypeError: 'type' object is not subscriptable"
    source_file: "pygeoinf/pygeoinf/visualization.py"
    source_line: 758
    priority: "P1"
    category: "import-error"
  - id: "test_normal_sum_operator.py::*"
    file: "pygeoinf/tests/test_normal_sum_operator.py"
    line: 8
    function: "module import"
    short_message: "TypeError: 'type' object is not subscriptable"
    source_file: "pygeoinf/pygeoinf/visualization.py"
    source_line: 758
    priority: "P1"
    category: "import-error"
  - id: "test_parallelism.py::*"
    file: "pygeoinf/tests/test_parallelism.py"
    line: 10
    function: "module import"
    short_message: "TypeError: 'type' object is not subscriptable"
    source_file: "pygeoinf/pygeoinf/visualization.py"
    source_line: 758
    priority: "P1"
    category: "import-error"
  - id: "test_plot.py::*"
    file: "pygeoinf/tests/test_plot.py"
    line: 13
    function: "module import"
    short_message: "TypeError: 'type' object is not subscriptable"
    source_file: "pygeoinf/pygeoinf/visualization.py"
    source_line: 758
    priority: "P1"
    category: "import-error"
  - id: "test_circle_lebesgue.py::*"
    file: "pygeoinf/tests/symmetric_space/test_circle_lebesgue.py"
    line: 8
    function: "module import"
    short_message: "TypeError: 'type' object is not subscriptable"
    source_file: "pygeoinf/pygeoinf/visualization.py"
    source_line: 758
    priority: "P1"
    category: "import-error"
  - id: "test_circle_sobolev.py::*"
    file: "pygeoinf/tests/symmetric_space/test_circle_sobolev.py"
    line: 8
    function: "module import"
    short_message: "TypeError: 'type' object is not subscriptable"
    source_file: "pygeoinf/pygeoinf/visualization.py"
    source_line: 758
    priority: "P1"
    category: "import-error"
  - id: "test_sh_tools.py::*"
    file: "pygeoinf/tests/symmetric_space/test_sh_tools.py"
    line: 7
    function: "module import"
    short_message: "ModuleNotFoundError: No module named 'pyshtools'"
    source_file: null
    source_line: null
    priority: "P3"
    category: "missing-optional-dependency"
  - id: "test_sphere_lebesgue.py::*"
    file: "pygeoinf/tests/symmetric_space/test_sphere_lebesgue.py"
    line: 7
    function: "module import"
    short_message: "ModuleNotFoundError: No module named 'pyshtools'"
    source_file: null
    source_line: null
    priority: "P3"
    category: "missing-optional-dependency"
  - id: "test_sphere_sobolev.py::*"
    file: "pygeoinf/tests/symmetric_space/test_sphere_sobolev.py"
    line: 7
    function: "module import"
    short_message: "ModuleNotFoundError: No module named 'pyshtools'"
    source_file: null
    source_line: null
    priority: "P3"
    category: "missing-optional-dependency"
---

# pygeoinf — Testing Status Report

**Last Updated**: 2026-02-17 (Phase 7 tests completed)
**Orchestrated by**: Atlas (with Explorer, Oracle, Code-Review subagents)
**Package**: pygeoinf v1.4.2

---

## Executive Summary

- **Total Tests**: 28 test files
- **Passing**: 15 test files (54%)
- **Failing**: 13 tests (46%)
  - **P1 (Critical)**: 10 tests blocked by single import error
  - **P3 (Optional)**: 3 tests missing optional dependency

**Recent Addition**: `test_halfspaces.py` (35 tests for HyperPlane, HalfSpace, PolyhedralSet) — **✅ All passing**

**Critical Issue**: Type annotation syntax error in `visualization.py` blocks 10 core tests. **Fix time: ~2 minutes**.

---

## New Tests Added (Phase 7)

### test_halfspaces.py ✅
**Status**: All 35 tests passing (100%)
**Location**: [tests/test_halfspaces.py](../tests/test_halfspaces.py)
**Implementation**: [pygeoinf/subsets.py](../pygeoinf/subsets.py#L1144-L1634), [pygeoinf/convex_analysis.py](../pygeoinf/convex_analysis.py#L194-L350)

**Test Coverage:**
- **TestHyperPlane** (12 tests): initialization, membership, projection, distance, boundary properties
- **TestHalfSpace** (13 tests): initialization, membership, support functions, inequality types, boundaries
- **TestPolyhedralSet** (7 tests): initialization, intersection semantics, box/simplex constraints
- **TestNumericalRobustness** (3 tests): parallel directions, large offsets, small normal vectors

**Classes Tested:**
- `HyperPlane`: {x | ⟨a,x⟩ = b} hyperplane representation
- `HalfSpace`: {x | ⟨a,x⟩ ≤ b} or {x | ⟨a,x⟩ ≥ b} with support functions
- `PolyhedralSet`: Intersection of half-spaces (convex polytopes)
- `HalfSpaceSupportFunction`: Convex analysis support function implementation

---

## Test Organization

### Test-to-Source Mapping
All test files follow the convention: `test_<module>.py` → `<module>.py`

**Test Categories** *(conceptual groupings, some tests span multiple categories)*:
- Core Infrastructure: 3 files
- Linear Algebra: 5 files
- Nonlinear Algebra: 2 files
- Inversion Methods: 6 files
- Geometry: 2 files
- Visualization: 1 file
- Symmetric Spaces: 6 files (3 for sphere, 3 for circle)
- Integration: 1 file (parallelism)
- Stub/Future: 4 files (not yet implemented)

---

## Failing Tests

### Category 1: Type Annotation Error (P1 - CRITICAL)

**Impact**: 10 tests blocked at import time

**Root Cause**: Inconsistent type annotation syntax in `pygeoinf/visualization.py`

**Location**:
- File: [pygeoinf/pygeoinf/visualization.py](../pygeoinf/pygeoinf/visualization.py)
- Lines: 758, 833

**Code Excerpt** (Line 758):
```python
def _polyhedral_inequalities_in_params(self, bounds: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert parameter bounds into polyhedral inequality constraints.
    ...
```
