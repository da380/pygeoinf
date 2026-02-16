---
handoff:
  - label: Start implementation with Atlas
    agent: Atlas
    prompt: Implement the plan
---

# Dual Master — Implementation Plan (Atlas)

Location: `pygeoinf/plans/dual_master_implementation.md`

Purpose
- Concrete, phase-by-phase TDD plan meant for Atlas orchestration and Sisyphus implementation.

Phases

- id: 2
  title: ConvexSubset support_function refactor
  owner: Sisyphus
  status: complete
  tests: []
  files:
    - pygeoinf/subsets.py
  acceptance: "`Ball` and `Ellipsoid` expose cached `support_function` property."

- id: 3
  title: DualMasterCostFunction
  owner: Sisyphus
  status: complete
  tests:
    - tests/test_dual_master_cost.py
  files:
    - pygeoinf/backus_gilbert.py
  acceptance: "Cost function evaluates and exposes subgradient oracle for simple examples."

- id: 4.2
  title: Subgradient step size rules
  owner: Sisyphus
  status: in-progress
  tests:
    - tests/test_subgradient.py
  files:
    - pygeoinf/convex_optimisation.py
  acceptance: "`SubgradientDescent` supports 'constant','diminishing','polyak','adaptive' rules and unit tests validate behavior on toy problems."

- id: 4.3
  title: Integration helpers for DualMasterCostFunction
  owner: Sisyphus
  status: todo
  tests:
    - tests/test_dual_master_cost.py
  files:
    - pygeoinf/backus_gilbert.py
  acceptance: "`solve_subgradient` and `solve_for_support_value` convenience methods added with warnings and diagnostics."

- id: 5
  title: Integration & Testing (demo notebook)
  owner: Sisyphus
  status: todo
  tests:
    - tests/test_dual_linear_inversion.py
  files:
    - pygeoinf/testing_sets/dual_master_demo.ipynb
  acceptance: "Demo notebook runs end-to-end on toy problem; integration tests added."

- id: 7
  title: Planes & Half-Spaces
  owner: Sisyphus
  status: in-progress
  tests:
    - tests/test_halfspaces.py
  files:
    - pygeoinf/subsets.py
    - pygeoinf/convex_analysis.py
  acceptance: "`HalfSpace` and `PolyhedralSet` support functions implemented; basic unit tests pass."

- id: 8
  title: Visualization API and demos
  owner: Sisyphus
  status: todo
  tests:
    - tests/test_visualization.py
  files:
    - pygeoinf/visualization.py
    - pygeoinf/testing_sets/visualization_demo.ipynb
  acceptance: "Plotting backends return figure objects; demo notebook renders example slices."

Notes & Run Instructions
- Each phase is TDD: tests first (fail), minimal code to pass, refactor.
- To run a single test file:
```
cd pygeoinf
PYTHONPATH=. pytest -q tests/test_dual_master_cost.py
```

Per-phase commit message template
"""
Phase {id}: {title}

What changed: files modified listed above.
Why: follow plan in `plans/dual_master_implementation.md`.
Tests: run `pytest -q tests/{matching_test}.py` and ensure failures were fixed by code changes.
"""
