# Agent References & Research Materials

This folder contains exploration reports, research summaries, and foundational documents that agents use to gather context **before** writing plans.

## Contents

### Exploration Reports
- `exploration-report.md` — High-level codebase structure and module overview
- `paper-demos-exploration-report.md` — Analysis of demo notebooks and examples
- `theory-txt-exploration-report.md` — Mapping of existing theory documentation

### Research & Design Proposals
- `theory-integration-proposal.md` — Proposal for theory-aware agent system design
- `theory-integration-implementation-summary.md` — Completed integration of theory validators
- `pygeoinf-analysis.md` — Deep dive into pygeoinf package structure
- `testing-status.md` — Overview of test coverage and test patterns

### Historical Research Plans
- `bundle-methods-optimizer-plan.md` — Original (Prometheus) research plan, superseded by full bundle-methods implementation
- `dual_master_implementation.md` — DualMasterCostFunction research (implemented as part of bundle methods)
- `dual_master_prometheus.md` — Prometheus research precursor to implementation
- `dual-master-fast-convex-optimisation-report.md` — Fast convex optimization report

## How Agents Use This Folder

1. **Explorer-subagent**: Reads exploration reports to understand codebase structure
2. **Oracle-subagent**: Consults research proposals and design docs to contextualize problems
3. **Theory-Validator-subagent**: References theory integration materials for validation rules
4. **Conductor (Atlas)**: Directs subagents to relevant research docs before planning

## Lifecycle

- **Creation**: Generated during Explorer, Oracle, or Prometheus phases
- **Use**: Consulted during planning and implementation
- **Archival**: Older research docs (>6 months) without active references can be moved to a `legacy/` subfolder or deleted

---

**See also:** [Agent Docs Index](../index.md)
