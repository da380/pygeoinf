# Windsurf/Cascade Instructions for pygeoinf

Historical Copilot-oriented workspace instructions are archived at `../AGENTS_copilot.md`.

## Package Role

`pygeoinf` is the core geophysical inference library. It contains Hilbert spaces, linear and nonlinear operators, convex analysis, optimisation methods, Backus-Gilbert methods, and visualization utilities.

## Orientation

Before reading individual source files, look for and read every living reference:

```text
pygeoinf/docs/agent-docs/references/living/*-reference.md
```

If no living references are present on a shared branch, proceed with targeted source exploration and do not treat the absence as stale by itself.

## Plan Directory

Use:

```text
pygeoinf/docs/agent-docs/
```

Expected subdirectories are `active-plans/`, `completed-plans/`, `references/`, and `theory/` where present.

## Mathematical Code Rules

- Preserve numerical correctness over speed, while still preferring vectorized numpy/scipy operations.
- For operators, verify adjoint identities, composition order, domain/codomain consistency, and edge cases.
- For convex analysis, verify convexity, positive homogeneity, subgradient conditions, and infinite/domain behavior.
- Use Google-style docstrings with LaTeX math and theory references for mathematical APIs.
- Document array shapes in docstrings.
- Use `np.testing.assert_allclose` with explicit tolerances in tests.

## Common Source Areas

- `pygeoinf/hilbert_space.py`: Hilbert-space abstractions.
- `pygeoinf/linear_operators.py`: Linear operators and adjoints.
- `pygeoinf/nonlinear_operators.py`: Nonlinear maps and forms.
- `pygeoinf/convex_optimisation.py`: Convex optimisation methods.
- `pygeoinf/plot.py` and `pygeoinf/visualization.py`: Plotting and visualization.

## After Changes

Update all affected living reference files under `pygeoinf/docs/agent-docs/references/living/` when they exist.
