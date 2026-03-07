## Phase 1 Complete: Remove HyperEllipsoid

Removed the unused `HyperEllipsoid` class from the public package surface and updated the maintained docs that still described it. Validation confirmed there are no remaining active code or maintained-doc references to `HyperEllipsoid`, and the quadratic-set role is already covered by `subsets.Ellipsoid`.

**Files created/changed:**
- `pygeoinf/pygeoinf/backus_gilbert.py`
- `pygeoinf/pygeoinf/__init__.py`
- `pygeoinf/docs/theory_papers_index.md`
- `pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md`

**Functions created/changed:**
- `DualMasterCostFunction` module context in `backus_gilbert.py`
- top-level package exports in `__init__.py`

**Tests created/changed:**
- None

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
refactor(backus-gilbert): remove unused HyperEllipsoid

- Delete dead HyperEllipsoid class from backus_gilbert
- Remove obsolete top-level export from pygeoinf
- Update maintained docs to reference Ellipsoid instead

Plan: pygeoinf/docs/agent-docs/active-plans/hyperellipsoid-removal-and-redundancy-audit-plan.md
Phase: 1 of 3
Related: pygeoinf/docs/agent-docs/completed-plans/hyperellipsoid-removal-and-redundancy-audit-phase-1-complete.md