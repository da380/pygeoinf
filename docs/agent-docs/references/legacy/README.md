# Legacy Archives

This folder contains superseded plans and research reports kept for historical reference.

## When to Consult Legacy Materials

- **Debugging historical decisions**: Understanding why certain design choices were made
- **Learning from past analysis**: Previous exploration reports may provide insights for similar problems
- **Understanding evolution**: Tracing how the package design changed over time

## When NOT to Consult

- **Active development**: Use `../living/` and `../active-plans/` instead
- **Implementation decisions**: Reference documents are authoritative, not legacy artifacts
- **Creating new plans**: Build on current understanding, not archived proposals

## Deprecation Notice Format

Every legacy file begins with:

```markdown
> ⚠️ **ARCHIVED** (YYYY-MM-DD)
> Reason: [superseded by X / historical research / plan abandoned]
> See: [link to current resource if applicable]
> Kept for historical reference only.
```

Example:
```markdown
> ⚠️ **ARCHIVED** (2026-03-05)
> This plan was superseded by bundle-methods-full-plan.md (Phase 1-7 covered the full scope)
> See: docs/agent-docs/active-plans/bundle-methods-full-plan.md
> Kept for historical reference only.
```

## Organization

### `superseded-plans/`
Implementation plans that have been replaced or abandoned.

### `research-reports/`
Exploration and analysis reports from earlier phases.

---

**Do not add new items here.** Move them only when a plan is complete or explicitly archived.
