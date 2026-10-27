# Benchmarks

Scripts run by hand, outside the test suite, whose measurements set the
library's defaults; the figures they produce are recorded in
`pygeoinf2/DECISIONS.md` (its appendix, and the entry each one belongs to).

| script | what it measures | recorded in |
|---|---|---|
| `stage0_unimodality.py` | whether the one-multiplier reduction of the feasible set's primal route is unimodal in `γ`, over twelve problem families; evaluation counts warm and cold | D-153 |
| `stage0_checks.py` | which formalism keeps its digits at tight noise (extended precision), and the overdetermined case where the prior is slack | D-153 |

Run from the repository root: `poetry run python benchmarks/<script>.py [seed] [case]`.
