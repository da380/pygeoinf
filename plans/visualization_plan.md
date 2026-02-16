# Visualization Plan — Phase 8 (pygeoinf)

**Location:** `pygeoinf/plans/visualization_plan.md`

**Purpose:** Detailed roadmap for Phase 8 visualization implementation. Consolidates existing `SubspaceSlicePlotter` implementation and provides TDD-ready tasks for Atlas/Sisyphus orchestration.

---

## Progress Summary

**Overall Progress:** 1/9 tasks complete (~11%)

| Task | Status | Owner | Progress | Notes |
|------|--------|-------|----------|-------|
| V1: Core module setup | ⏸️ NOT STARTED | Sisyphus | 0/3 | Move SubspaceSlicePlotter, expose plot() function |
| V2: Subset.plot() API | ⏸️ NOT STARTED | Sisyphus | 0/2 | Add entrypoint to Subset base class |
| V3: Plotly 3D backend | ⏸️ NOT STARTED | Sisyphus | 0/4 | Interactive rendering; optional for MVP |
| V4: Demo notebook | ⏸️ NOT STARTED | Sisyphus | 0/1 | Ball/Ellipsoid/HalfSpace slices + HTML export |
| V5: Polish & docs | ⏸️ NOT STARTED | Sisyphus | 0/2 | README section, agent usage guide |
| **Foundation** | 🟨 IN PROGRESS | - | 1/1 | `SubspaceSlicePlotter` exists in `plot.py` ✅ |

---

## Current State (Foundation) ✅

**What exists:**
- ✅ `SubspaceSlicePlotter` class in `pygeoinf/plot.py` (fully functional)
- ✅ Membership-oracle slice plotting (1D/2D/3D)
- ✅ 1D: bar plots with pixel-based height conversion
- ✅ 2D: filled contours and line contours
- ✅ 3D: voxel and surface plots (matplotlib backend)
- ✅ Parameter validation (7 checks)
- ✅ Flexible instantiation with sensible defaults
- ✅ Used in test harness: `test_polyhedra.py`

**What's NOT done:**
- ❌ No official `pygeoinf/visualization.py` module
- ❌ No `Subset.plot()` entrypoint
- ❌ No Plotly backend for interactive 3D
- ❌ No demo notebook
- ❌ No formal unit tests

**Next step:** Tasks V1 → V2 → V3 → V4 → V5

---

## Phase 8 Goals

1. **Consolidate** SubspaceSlicePlotter into official module with clean API
2. **Expose** user-facing `Subset.plot()` method for all convex sets
3. **Implement** Plotly backend for interactive 3D rendering
4. **Demonstrate** with end-to-end demo notebook (Ball, Ellipsoid, HalfSpace, PolyhedralSet)
5. **Polish** with tests, docs, and agent integration guides

---

## Detailed Task Breakdown

### Task V1: Core Module Setup ⏸️

**Status:** NOT STARTED (0/3 sub-tasks)
**Owner:** Sisyphus
**Depends on:** None (can start immediately)
**Files:** `pygeoinf/visualization.py` (create), `tests/test_visualization.py` (update)

#### Goal
Move `SubspaceSlicePlotter` from `pygeoinf/plot.py` into official `pygeoinf/visualization.py` module and expose clean public API.

#### Sub-Tasks

**V1.1** — Create `pygeoinf/visualization.py` module
- [ ] Move `SubspaceSlicePlotter` class from `plot.py`
- [ ] Move all imports (matplotlib, numpy, typing, HilbertSpace, etc.)
- [ ] Add module docstring explaining visualization architecture
- [ ] Export public API: `__all__ = ['SubspaceSlicePlotter', 'plot_slice']`
- [ ] Keep `plot.py` temporarily (dual location for backward compatibility or deprecation warning)

**V1.2** — Add `plot_slice()` convenience function
```python
def plot_slice(
    subset: ConvexSubset,
    on_subspace: AffineSubspace,
    backend: str = 'matplotlib',
    method: str = 'membership',
    resolution: int = 100,
    **kwargs
) -> plt.Figure:
    """
    Plot a convex subset restricted to an affine subspace.

    Args:
        subset: ConvexSubset to visualize
        on_subspace: 1D, 2D, or 3D affine subspace (plotting surface)
        backend: 'matplotlib' (2D) or 'plotly' (3D, requires plotly)
        method: 'membership' (oracle-based) or 'support_function' (fallback)
        resolution: grid resolution for plotting
        **kwargs: passed to backend-specific plotter

    Returns:
        matplotlib.figure.Figure or plotly.graph_objs.Figure
    """
    plotter = SubspaceSlicePlotter(subset, on_subspace, **kwargs)
    return plotter.plot()
```

**V1.3** — Add smoke tests for V1
- [ ] Test: `test_visualization_import()` — verify `pygeoinf.visualization` module loads
- [ ] Test: `test_plot_slice_ball_2d()` — plot Ball slice on 2D subspace, verify Figure returned
- [ ] Test: `test_plot_slice_ball_3d()` — plot Ball slice on 3D subspace, verify Figure returned (matplotlib backend)

#### Acceptance Criteria
✅ `pygeoinf/visualization.py` exists and exports `SubspaceSlicePlotter`, `plot_slice`
✅ `plot_slice()` works for 2D and 3D Ball subsets without errors
✅ Smoke tests pass without GUI (use matplotlib Agg backend)

---

### Task V2: Subset.plot() API ⏸️

**Status:** NOT STARTED (0/2 sub-tasks)
**Owner:** Sisyphus
**Depends on:** V1 ✅
**Files:** `pygeoinf/subsets.py`

#### Goal
Add a `plot()` method to the `Subset` base class so users call `subset.plot(on_subspace)` instead of `plot_slice(subset, on_subspace)`.

#### Sub-Tasks

**V2.1** — Add `plot()` method to `Subset` base class
```python
class Subset(ABC):
    # ... existing code ...

    def plot(
        self,
        on_subspace: Optional['AffineSubspace'] = None,
        backend: str = 'auto',
        method: str = 'membership',
        **kwargs
    ) -> 'Figure':
        """
        Plot this subset restricted to an affine subspace.

        Args:
            on_subspace: 1D, 2D, or 3D affine subspace for plotting.
                        If None, auto-generates from first N basis vectors.
            backend: 'matplotlib' (2D), 'plotly' (3D), or 'auto' (infer from dim)
            method: 'membership' (oracle) or 'support_function' (fallback)
            **kwargs: passed to visualization backend

        Returns:
            Figure object (matplotlib or plotly)

        Examples:
            >>> ball = Ball(euclidean_3d, center=origin, radius=1.0)
            >>> slice_2d = AffineSubspace(...)  # 2D subspace
            >>> fig = ball.plot(on_subspace=slice_2d)
            >>> fig.show()  # or plt.show() for matplotlib
        """
        from . import visualization

        if on_subspace is None:
            on_subspace = self._default_plot_subspace()

        return visualization.plot_slice(
            self, on_subspace, backend=backend, method=method, **kwargs
        )

    def _default_plot_subspace(self) -> 'AffineSubspace':
        """Generate default 2D or 3D plotting subspace from first N basis vectors."""
        # Implementation: orthonormalize first 2-3 basis vectors
        # Return AffineSubspace with center at origin
        pass
```

**V2.2** — Add corresponding tests
- [ ] Test: `test_subset_plot_callable()` — verify `subset.plot` is callable
- [ ] Test: `test_subset_plot_returns_figure()` — verify returns Figure-like object
- [ ] Test: `test_subset_plot_default_subspace()` — verify auto-subspace generation works
- [ ] Test: `test_subset_plot_custom_backend()` — verify backend parameter respected

#### Acceptance Criteria
✅ `Subset.plot(on_subspace)` method exists and is callable
✅ Returns matplotlib Figure for 2D slices, handles 3D subspaces
✅ Tests pass; API is intuitive and consistent with pygeoinf patterns

---

### Task V3: Plotly 3D Backend ⏸️

**Status:** NOT STARTED (0/4 sub-tasks)
**Owner:** Sisyphus
**Depends on:** V1 + V2
**Files:** `pygeoinf/visualization.py` (extend), `pyproject.toml` (update deps)

#### Goal
Add interactive 3D rendering via Plotly (GPU-accelerated WebGL, smooth rotation/zoom/pan, HTML export).

#### Sub-Tasks

**V3.1** — Update dependencies
- [ ] Add `plotly` to optional extras in `pyproject.toml` or create `requirements-visualization.txt`
  ```toml
  [project.optional-dependencies]
  visualization = ["plotly>=5.0.0"]
  ```
- [ ] Or (alternative): list plotly as a soft dependency with graceful fallback

**V3.2** — Implement Plotly backend in `visualization.py`
```python
def _plot_slice_plotly_3d(
    subset: ConvexSubset,
    on_subspace: AffineSubspace,
    resolution: int = 50,
    title: Optional[str] = None,
    **kwargs
) -> 'pgo.Figure':
    """
    Render 3D subset slice using Plotly WebGL backend.

    Args:
        subset: ConvexSubset to visualize
        on_subspace: 3D affine subspace
        resolution: grid resolution (50×50 by default for fast render)
        title: figure title
        **kwargs: passed to plotly Figure creation

    Returns:
        plotly.graph_objs.Figure (interactive)
    """
    try:
        import plotly.graph_objs as pgo
    except ImportError:
        raise ImportError(
            "plotly not installed. Install via:\n"
            "  pip install plotly\n"
            "or:\n"
            "  pip install pygeoinf[visualization]"
        )

    # 1. Create membership oracle grid in subspace
    # 2. Evaluate subset.is_element() on grid
    # 3. Extract surface mesh (boundary points)
    # 4. Create Plotly surface plot
    # 5. Add lighting, camera presets for interactive exploration

    fig = pgo.Figure()
    # ... implementation details ...
    return fig
```

**V3.3** — Integrate Plotly backend into `plot_slice()` and `SubspaceSlicePlotter`
- [ ] Extend `SubspaceSlicePlotter.plot()` to detect 3D and offer Plotly option
- [ ] Auto-select backend: matplotlib for 2D, Plotly for 3D (if available)
- [ ] Add `backend` parameter to override (e.g., `backend='plotly'` force 3D as Plotly)

**V3.4** — Tests for Plotly backend
- [ ] Test: `test_plotly_backend_available()` — check Plotly import and fallback
- [ ] Test: `test_plot_3d_plotly_ball()` — render Ball on 3D subspace with Plotly
- [ ] Test: `test_plotly_returns_figure()` — verify returns `plotly.graph_objs.Figure`
- [ ] Test: `test_plotly_html_export()` — verify `.to_html()` works for saving

#### Acceptance Criteria
✅ Plotly backend renders 3D slices interactively
✅ Graceful fallback if plotly not installed
✅ HTML export works (`fig.to_html()`)
✅ Tests pass; 3D rotation/zoom smooth on typical hardware

---

### Task V4: Demo Notebook ⏸️

**Status:** NOT STARTED (0/1 sub-task)
**Owner:** Sisyphus
**Depends on:** V1 + V2 + V3
**File:** `pygeoinf/testing_sets/visualization_demo.ipynb`

#### Goal
Create end-to-end demonstration notebook showing visualization workflows for all major convex sets.

#### Sub-Task V4.1: Build demo notebook
- [ ] **Section 1: Setup**
  - Import pygeoinf, numpy, create sample spaces (D, M, P)

- [ ] **Section 2: Ball visualization**
  - Create `Ball` in 3D
  - Plot on 2D subspace (matplotlib)
  - Plot on 3D subspace (Plotly)
  - Interactive rotate/zoom demo

- [ ] **Section 3: Ellipsoid visualization**
  - Create `Ellipsoid` with tensor (or identity scaling)
  - Compare to Ball visually
  - Show how geometry changes with ellipse axes

- [ ] **Section 4: HalfSpace and polyhedral sets**
  - Display `HalfSpace` slice
  - Display `PolyhedralSet` (intersection of 3-4 HalfSpaces)
  - Show bounded polytope geometries

- [ ] **Section 5: Dual master application**
  - Define `DualMasterCostFunction`
  - Plot admissible property set U as directional bounds
  - Visualize constraint from different property directions

- [ ] **Section 6: Saving and exporting**
  - Save matplotlib figure as PNG/PDF
  - Export Plotly 3D as interactive HTML (shareable, no Jupyter needed)
  - Example: `fig.to_html('my_3d_model.html')`

#### Acceptance Criteria
✅ Notebook runs end-to-end without errors
✅ Generates 2D plots (matplotlib) and 3D interactive plots (Plotly)
✅ HTML export demo works; file is shareable standalone
✅ Demonstrates all major convex set types (Ball, Ellipsoid, HalfSpace, PolyhedralSet)

---

### Task V5: Polish & Documentation ⏸️

**Status:** NOT STARTED (0/2 sub-tasks)
**Owner:** Sisyphus
**Depends on:** V1–V4 all complete
**Files:** `pygeoinf/README.md`, `pygeoinf/plans/visualization_plan.md`, or separate `VISUALIZATION_GUIDE.md`

#### Sub-Tasks

**V5.1** — API documentation & quick-start
- [ ] Add section to `pygeoinf/README.md` or create `docs/VISUALIZATION.md`:
  ```markdown
  ## Visualization

  ### Quick Start

  ```python
  from pygeoinf import Ball, EuclideanSpace, AffineSubspace

  # Create a 3D ball
  M = EuclideanSpace(3)
  ball = Ball(M, center=M.zero, radius=1.0)

  # Create a 2D plotting subspace
  subspace_2d = AffineSubspace.from_tangent_basis(
      M, basis_vectors=[M.basis_vector(0), M.basis_vector(1)]
  )

  # Plot
  fig = ball.plot(on_subspace=subspace_2d)
  fig.show()
  ```

  ### Backends

  | Backend | Dimensions | Style |
  |---------|------------|-------|
  | matplotlib | 2D (1D/3D supported) | Static, publication-ready |
  | plotly | 2D, 3D | Interactive, WebGL, HTML export |
  ```

**V5.2** — Agent integration guide
- [ ] Document in this file how agents should use visualization:
  ```markdown
  ## For Agents: Using Visualization

  ### Explorer Agent
  - Can discover plotting utilities via `grep_search` for "plot"
  - Should identify `SubspaceSlicePlotter` and `Subset.plot()` locations

  ### Oracle Agent
  - Can research backend tradeoffs (matplotlib vs Plotly)
  - Should propose caching strategies for repeated slices

  ### Sisyphus Agent
  - Implement tasks V1→V5 in order
  - Each task has specific test names and acceptance criteria
  - Use TDD: write failing test, implement minimal code, refactor

  ### Atlas Agent
  - Orchestrate Sisyphus through V1→V5 tasks
  - After each task completion: run tests, commit, move to next
  ```

#### Acceptance Criteria
✅ README section exists with examples
✅ Agent integration guide clear and usable
✅ Users can get started with visualization in <5 minutes

---

## Design Decisions

### 1. Why separate `visualization.py` from `plot.py`?
**Rationale:** `plot.py` is currently a thin wrapper; `visualization.py` will be the public, stable API. Keeps concerns separated and allows future plotting backends (e.g., D3.js, Three.js) to be added alongside matplotlib/Plotly.

### 2. Why Optional Plotly dependency?
**Rationale:** Not everyone needs 3D interactive rendering. Making Plotly optional reduces install size and keeps dependencies lean for CLI-only users. Graceful fallback (error message with install instructions) improves UX.

### 3. Why membership oracle (representation A)?
**Rationale:** Most general and works for any set with `is_element()`. Trade-off: lower resolution/coarser rendering vs. universal applicability. Cleaner API than supporting multiple representations simultaneously.

### 4. Why `Subset.plot()` instead of standalone function?
**Rationale:** Aligns with OOP patterns; users naturally write `subset.plot(...)` like `.visualize()` in other libraries. Reduces cognitive load. Allows composition with other Subset methods.

### 5. Why 2D matplotlib + 3D Plotly split?
**Rationale:** Matplotlib excels at 2D; matplotlib's 3D is limited (slow, poor interactivity). Plotly WebGL is fast, interactive, exportable. Users expect 3D interactivity; 2D is fine static.

---

## File Organization

| File | Status | Responsibility |
|------|--------|-----------------|
| `pygeoinf/visualization.py` | 📝 TO CREATE | Public API: `SubspaceSlicePlotter`, `plot_slice()`, backend integration |
| `pygeoinf/plot.py` | ✅ EXISTS | Current home of `SubspaceSlicePlotter` (temporary; migrate to visualization.py) |
| `pygeoinf/subsets.py` | 🔄 TO MODIFY | Add `plot()` method to `Subset` base class |
| `pygeoinf/README.md` | 🔄 TO MODIFY | Add visualization quick-start section |
| `tests/test_visualization.py` | 📝 TO CREATE/UPDATE | Smoke tests + unit tests for all tasks |
| `pygeoinf/testing_sets/visualization_demo.ipynb` | 📝 TO CREATE | End-to-end demo notebook |
| `pyproject.toml` | 🔄 TO MODIFY | Add plotly to optional deps |

---

## Run Instructions (Local Testing)

### Single task test
```bash
cd pygeoinf
PYTHONPATH=. pytest -q tests/test_visualization.py::test_visualization_import
```

### All visualization tests
```bash
cd pygeoinf
PYTHONPATH=. pytest -q tests/test_visualization.py
```

### Run demo notebook (after V4 complete)
```bash
cd pygeoinf
jupyter notebook testing_sets/visualization_demo.ipynb
```

---

## Commit Message Template

```
Phase 8.V{task}: {title}

What changed: {files modified}
Why: Complete Phase 8 visualization tasks (see plans/visualization_plan.md)
Tests: pytest -q tests/test_visualization.py::test_*
Status: {acceptance criteria status}
```

Example:
```
Phase 8.V1: Core visualization module setup

What changed:
- Created pygeoinf/visualization.py (moved SubspaceSlicePlotter from plot.py)
- Added plot_slice() convenience function
- Updated __all__ exports

Why: Consolidate visualization into official module (Phase 8.V1)
Tests: pytest -q tests/test_visualization.py (all pass)
Status: V1 acceptance criteria met ✅
```

---

## Next Actions

**Immediate Priority:**

1. **V1 (Core setup)** — Move code, expose API, write smoke tests
   ```bash
   echo "Create pygeoinf/visualization.py and populate"
   # Expected time: 1–2 hours
   ```

2. **V2 (Subset.plot())** — Add method to base class, integrate
   ```bash
   echo "Add plot() to Subset base class in subsets.py"
   # Expected time: 30 min
   ```

3. **V3 (Plotly backend)** — Interactive 3D rendering
   ```bash
   echo "Extend visualization.py with Plotly backend"
   # Expected time: 1–2 hours
   ```

4. **V4 (Demo)** — Create notebook showcasing all features
   ```bash
   echo "jupyter notebook testing_sets/visualization_demo.ipynb"
   # Expected time: 1 hour
   ```

5. **V5 (Polish)** — Docs, README, agent guide
   ```bash
   echo "Update README.md and agent integration guide"
   # Expected time: 30 min
   ```

**Recommended Start:**
```bash
conda activate inferences3
code pygeoinf/visualization.py
# Begin V1 implementation with TDD
```

---

## Agent Usage Guide

### For Prometheus (Research Agent)
- Research Matplotlib 3D limitations vs Plotly WebGL performance
- Propose caching strategy for repeated slices with same subspace
- Estimate optimization opportunities for large resolution grids

### For Explorer (Exploration Agent)
- Locate all plotting code currently in codebase
- Find existing tests for visualization
- Identify where `plot.py` is imported/used

### For Sisyphus (Implementation Agent)
- Implement V1→V5 sequentially using TDD
- Each task has explicit test names and acceptance criteria
- Create PR after each task completion

### For Atlas (Conductor Agent)
- Orchestrate Sisyphus through all 5 tasks
- Verify acceptance criteria after each task
- Run full test suite before moving to next task
- Commit and document progress
