## Phase 6 Complete: Polish, living references, demo, archival

Updated living references to document all new modules and the Imhof
adaptive truncation fix; verified the existing demo script runs cleanly;
archived all plan and phase-complete files to `completed-plans/`.

**Files created/changed:**
- `pygeoinf/docs/agent-docs/references/living/convex-analysis-reference.md` (updated)
  - Added `_imhof_cdf` adaptive truncation note under `quadratic_form_quantile.py`
  - Updated test expectations to reflect final counts (98 pygeoinf + 6 intervalinf)
- `pygeoinf/work/function_space_hardening_demo.py` (verified — already complete)
  - Theory figure: radius vs confidence for theta ∈ {0.0, 0.2, 0.5, 0.8}
  - Backend benchmark: Imhof vs WS on N ∈ {50, 500, 5000}

**Performance fix (committed separately as Phase 5.5):**
- `pygeoinf/pygeoinf/quadratic_form_quantile.py` — `_imhof_cdf` adaptive U
  - Old: `U = 16/w_min_pos` → 400 000 for N=50 InverseLaplacian
  - New: binary search on `log(U) + 0.25·Σlog(1+w_j²U²) ≥ log(1/(π·rtol))`
  - N=50: U ≈ 640 (625× smaller), credible_set: 6.5 s → 10 ms
  - All 98 pygeoinf tests: 41 s → 10 s; intervalinf Phase 5 suite: 70 s → 11 s
- `intervalinf/tests/spaces/test_lebesgue_hardening.py` — n_samples 5000→500
  - basis-free L² inner_product costs ~13 ms/call; 500 samples ≈ 1.3 % MC noise,
    well within the 15 % tolerance

**Tests created/changed:** (no new tests; existing 98 + 6 remain green)

**Review Status:** APPROVED — all tests pass, demo runs, living references accurate

**Git Commit Message:**
```
docs(hardening): Phase 6 — living refs, demo, archival

- Update convex-analysis-reference.md with adaptive-U Imhof note and
  final test counts
- Verify work/function_space_hardening_demo.py runs post-fix
- Archive all phase-complete and plan files to completed-plans/

Plan: pygeoinf/docs/agent-docs/active-plans/function-space-hardening-plan.md
Phase: 6 of 6
Related: pygeoinf/docs/agent-docs/active-plans/function-space-hardening-phase-6-complete.md
```
