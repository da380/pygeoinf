# Agent Documentation

This folder contains planning documents, research summaries, and implementation guides intended for **agent-driven development** in pygeoinf. User-facing documentation lives in the parent `docs/` folder.

## Structure

### `completed-plans/`
Archived plans and phase summaries from finished projects. These are read-only references showing what was accomplished and how.

- **Purpose:** Historical record and knowledge base for agent learning
- **Examples:** Bundle methods implementation, linear operators chapter, function support metadata
- **Retention:** Keep indefinitely; useful for code review and understanding past decisions

### `active-plans/`
Plans currently under development or awaiting implementation. These move here when close to launch, then graduate to `completed-plans/` upon finish.

- **Purpose:** Real-time tracking of in-progress work
- **Status:** Linked from main `plans/` folder; managed by Conductor agents
- **Lifecycle:** Planned → Active → Completed

### `references/`
Exploration reports, research summaries, and **living reference documents**.

- **Purpose:**
  - Intermediate findings used to inform planning
  - **Living architecture references** (`*-reference.md`) — describe package structure, class hierarchies, APIs
- **Examples:** Oracle research reports, exploration summaries, theory integration proposals, `pygeoinf-reference.md`
- **Retention:** Keep indefinitely; references especially valuable for agent learning
- **Agent usage:** Agents **must read `*-reference.md` first** before exploring source files

### `theory/`
Theory-to-code mappings and mathematical validation materials used by Theory-Validator-subagent.

- **Purpose:** Ensure mathematical correctness of implementations
- **Contents:** Theory integration summaries, validator docs, theory papers index
- **Audience:** Theory-Validator-subagent, developers needing mathematical context

---

## How Agents Use This Folder

1. **Conductor (Atlas)**: Orchestrates planning → implementation → review cycle; writes plans to `active-plans/`.
2. **Oracle**: Researches context; findings inform plan drafting.
3. **Sisyphus**: Implements phase-by-phase; updates phase-complete files.
4. **Code-Review**: Validates implementations; approves phase completion.
5. **Theory-Validator**: Checks mathematical correctness; consults `theory/` folder for validation rules.

## Navigation

- **For users**: See `../` for user documentation and examples.
- **For agents**: Start with the most recent `completed-plans/` entry to trace decisions from similar prior work.
- **For maintainers**: Periodically archive old `references/` that are no longer active.

---

**Last Updated:** 2026-03-05
