# Completed Plans Archive

This folder contains finalized plan documents and phase-by-phase completion summaries from finished projects.

## Organization

Files follow the naming convention:
- `<project>-plan-complete.md` — Top-level summary of all phases completed
- `<project>-phase-<N>-complete.md` — Phase-specific completion summary with review status and git commit info

## How to Use This Archive

### For Code Review
When investigating why/how a particular feature was built, find the relevant plan and read:
1. The main `-complete.md` to understand overall scope
2. Phase completion files to see incremental decisions and test coverage

### For Agents Learning Patterns
Agents can study completed plans to:
- Learn similar problem structures (e.g., "mathematical documentation + LaTeX + tests")
- Understand the typical pattern: research → planning → TDD → review → commit
- Identify reusable patterns for similar future work

### For Git History
Each phase-complete file includes a git commit message, making it easy to `git log --grep=<pattern>` to find related commits.

## Navigation by Project

### Bundle Methods & Optimization
- `bundle-methods-full-plan-complete.md` + 7 phase files — Full implementation of bundle methods, QP backends, and smoothing

### Technical Manual (LaTeX Documentation)
- `linear-operators-chapter-complete.md` + 7 phase files — Linear operators chapter (Riesz-first, dual vs adjoint)
- `technical-manual-boxes-reintro-complete.md` + phase file — Reintroduce boxes with notation-to-API alignment
- `riesz-first-manual-section-phase-1-complete.md` — Riesz-first design section
- `linear-forms-section-phase-1-complete.md` — Linear forms chapter rewrite
- `coefficient-space-notation-and-metric-phase-1-complete.md` — Coefficient space notation clarity

### Intervalinf Features
- `intervalinf-function-support-metadata-complete.md` + 2 phase files — Function.support propagation through operations
- `example-3-notebook-cleanup-phase-1-complete.md` — Clean discretization example notebook

### Bug Fixes & Maintenance
- `get-tangent-basis-fix-complete.md` — Fixed non-axis-aligned subspace dimension bug

---

**See also:** [Agent Docs Index](../index.md)
