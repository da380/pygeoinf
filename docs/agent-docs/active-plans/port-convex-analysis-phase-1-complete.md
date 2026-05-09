## Phase 1 Complete: Port Foundational Changes

Ported foundational module improvements from `convex_analysis` branch to `port_convex_analysis`. Fixed 3 pre-existing bugs found during code review (PolyhedralSet.boundary missing @property, DirectSumSpace.axpy missing return, HyperPlane.dimension wrong value). All 424 tests pass, 6 skipped (pending Phase 2 convex_analysis.py).

**Files created/changed:**
- `pygeoinf/hilbert_space.py` — `axpy()` returns Vector, `distance()` method, improved docstrings
- `pygeoinf/nonlinear_forms.py` — subgradient infrastructure (has_subgradient, subgradient(), algebraic propagation)
- `pygeoinf/direct_sum.py` — `zero`, `inner_product`, `random` for basis-free spaces, axpy return type fix
- `pygeoinf/linear_optimisation.py` — Bug 2 bisection convergence fix
- `pygeoinf/subsets.py` — HyperPlane, HalfSpace, PolyhedralSet classes, ConvexSubset extensions
- `tests/test_direct_sum.py` — 7 new basis-free direct sum tests
- `tests/test_halfspaces.py` — NEW, 35 tests (29 passing, 6 skipped)
- `tests/test_mass_weighted.py` — trivial formatting

**Functions created/changed:**
- `HilbertSpace.axpy()` — now returns Vector
- `HilbertSpace.distance()` — NEW
- `HilbertSpaceDirectSum.zero` — NEW property
- `HilbertSpaceDirectSum.inner_product()` — NEW override
- `HilbertSpaceDirectSum.random()` — NEW override
- `HilbertSpaceDirectSum.axpy()` — now returns List[Any]
- `NonLinearForm.__init__()` — subgradient parameter added
- `NonLinearForm.has_subgradient` — NEW property
- `NonLinearForm.subgradient()` — NEW method
- `ConvexSubset` — abstract `support_function`, `directional_bound`, `closure()`, `_warn_if_open()`
- `HyperPlane` — NEW class
- `HalfSpace` — NEW class
- `PolyhedralSet` — NEW class

**Tests created/changed:**
- `tests/test_direct_sum.py::TestBasisFreeDirectSum` — 5 new tests
- `tests/test_direct_sum.py::TestDirectSumProperties` — 2 new tests
- `tests/test_halfspaces.py` — 35 new tests (6 skipped pending convex_analysis.py)

**Review Status:** APPROVED after 3 fixes (PolyhedralSet.boundary @property, DirectSumSpace.axpy return, HyperPlane.dimension)

**Git Commit Message:**
```
feat(foundation): port foundational changes for convex analysis stack

- HilbertSpace.axpy() returns Vector; add distance() method
- NonLinearForm: subgradient infrastructure (has_subgradient, subgradient())
- DirectSumSpace: zero, inner_product, random for basis-free spaces
- linear_optimisation: Bug 2 bisection convergence fix
- subsets: HyperPlane, HalfSpace, PolyhedralSet; ConvexSubset abstracts
- Fix 3 bugs: PolyhedralSet.boundary @property, DirectSum.axpy return, HyperPlane.dimension

Plan: pygeoinf/docs/agent-docs/active-plans/port-convex-analysis-plan.md
Phase: 1 of 6
Related: pygeoinf/docs/agent-docs/active-plans/port-convex-analysis-phase-1-complete.md
```
