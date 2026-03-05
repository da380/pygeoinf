# References

This folder contains **living reference documents** that agents actively use, plus **legacy archives** of superseded plans and historical research.

## Structure

### `living/`
**Active, maintained architecture and API references.**

- `pygeoinf-reference.md` — Current package architecture, class hierarchies, APIs
  - **Agents must read this FIRST before exploring source files**
  - Updated: immediately after code changes (by Sisyphus-subagent)
  - Lifecycle: never becomes stale (updated every phase)

### `legacy/`
**Archived artifacts: superseded plans and research reports.**

Each file is marked with a deprecation notice showing:
- Date archived
- Reason (superseded by newer plan, or historical research)
- Link to current resource (if applicable)

#### `legacy/superseded-plans/`
Plans that have been superseded, replaced, or abandoned.

- `bundle-methods-optimizer-plan.md` — Prometheus research draft, superseded by full bundle-methods plan
- `dual_master_implementation.md` — DualMasterCostFunction research, implemented in bundle methods
- `dual_master_prometheus.md` — Prometheus research precursor

#### `legacy/research-reports/`
Exploration and research reports from Oracle/Explorer phases.

- `exploration-report.md`, `paper-demos-exploration-report.md`, `theory-txt-exploration-report.md`
- `dual-master-fast-convex-optimisation-report.md`
- `pygeoinf-analysis.md`, `testing-status.md`
- `theory-integration-proposal.md`, `theory-integration-implementation-summary.md`

---

## For Agents

### Reading Strategy
1. **Always start with:** `living/pygeoinf-reference.md`
2. **Then check:** `../active-plans/*-plan.md` for current work
3. **Consult legacy only if:** Debugging past design decisions or understanding evolution

### Updating Living References
After implementing features:
1. Open `living/pygeoinf-reference.md`
2. Update affected sections (new classes, changed signatures, new files, patterns)
3. Add timestamp: `<!-- Last updated: YYYY-MM-DD by Sisyphus -->`
4. Commit with plan reference

### Archiving Artifacts
When a plan completes or research is superseded:
1. Move to `legacy/<category>/`
2. Add deprecation notice:
   ```markdown
   > ⚠️ **ARCHIVED** (YYYY-MM-DD)
   > Superseded by [link] / No longer maintained
   > Historical reference only.
   ```

---

**See also:** [Agent Docs Index](../index.md)
