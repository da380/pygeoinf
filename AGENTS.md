# Agent Configuration for pygeoinf

## Plan Directory
`plans/`

## Package Context
**pygeoinf** is an abstract Hilbert space framework for geophysical inverse problems and Bayesian inference.

### Key Features
- Abstract mathematical structure: spaces, operators, measures
- Numerical implementations: matrix backends, iterative solvers
- Inversion algorithms: Bayesian, least-squares, Backus-Gilbert
- Convex analysis: support functions, convex sets, optimization

### Python Requirements
- Python ≥ 3.11
- Dependencies: numpy, scipy, matplotlib, joblib

## Theory Documents
Mathematical foundations for Deterministic Linear Inversion (DLI):
- **`theory/theory.txt`** (2672 lines) - Main LaTeX document "DLI as Convex Analysis problems"
- **`docs/theory_map.md`** (758 lines) - Theory-to-code mapping reference
- **`docs/theory_papers_index.md`** (400 lines) - Index of 18 reference papers
- **`theory/`** directory - 18 PDF research papers on inverse problems

Key theory concepts:
- Model space M (Banach/Hilbert), Data space D, Property space P
- Forward operator G: M → D, Property operator T: M → P
- Support functions σ_S(q) for convex sets
- Dual master equation: h_U(q) = inf_λ {⟨λ, d̃⟩ + σ_B(T*q - G*λ) + σ_V(-λ)}

## Related Packages
**intervalinf** at `../intervalinf/`
- Provides concrete implementations of HilbertSpace interface
- Lebesgue (L²) and Sobolev spaces on 1D intervals
- Differential operators (Laplacian, gradient)
- Spectral methods with fast transforms
- **pygeoinf depends on intervalinf for concrete spaces**

## Current Development Status
Working on: Dual Master implementation (~50% complete, Phase 4/8)
- Phase 1-3: ✅ Architecture, support functions, cost function
- Phase 4: 🟨 Bundle methods optimizer (in progress)
- Phase 7: ✅ Planes & half-spaces (35 tests passing)
- Phase 8: 🟨 Visualization (SubspaceSlicePlotter complete)

Known issues:
- 13 test failures due to type hint compatibility (Python 3.10 vs 3.11 requirement)
- Need to add `from __future__ import annotations` to visualization.py:1
