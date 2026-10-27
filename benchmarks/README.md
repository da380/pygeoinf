# Benchmarks

> Exploratory, 2026-10-27 (D-155): the three feasible-set problem scripts
> were run at reduced size, found three defects since fixed, and are kept
> here for a later return; nothing in the test suite runs them.

Scripts run by hand, outside the test suite, whose measurements set the
library's defaults; the figures they produce are recorded in
`pygeoinf2/DECISIONS.md` (its appendix, and the entry each one belongs to).

| script | what it measures | recorded in |
|---|---|---|
| `stage0_unimodality.py` | whether the one-multiplier reduction of the feasible set's primal route is unimodal in `γ`, over twelve problem families; evaluation counts warm and cold | D-153 |
| `stage0_checks.py` | which formalism keeps its digits at tight noise (extended precision), and the overdetermined case where the prior is slack | D-153 |
| `point_data_sphere.py` | Al-Attar (2021) §2.5.2 and §3.3.2: 250 point values on the sphere, `H²` with length scale 0.25, a ball prior and a ball noise set, coefficients of degree ≤ 1 and 1–3 as the properties; the quadratic tier's algorithms, brackets and cost, extents against brackets, one worker against several; the exact-data closed form and the noise-to-zero sweep behind `--limit` (D-154) | D-152, D-154 |
| `flexure_gravity.py` | Mag et al. §3.2: the two-field chain of example 22 on an `L²` amplitude ellipsoid and a chi-squared noise set, paired cap averages; the algorithms and their brackets at each truncation, the S-procedure weight, the contact-point hull, convergence under refinement | D-152 |
| `operator_timings.py` | the point-evaluation and path-average operators on the sphere, the torus and the plane, matrix-free against dense: agreement, the adjoint identity, construction and application times (the three v1 harnesses of `work/`, ported) | — |
| `phase_delay.py` | Mag et al. §3.1: nested acquisition from the catalogues, `N_d` from 10² to 10⁴, a ball prior, twelve directions; the spectral solver against conjugate gradients plain, deflated, and preconditioned by a surrogate's Woodbury inverse, in applications and wall time | D-152 |

Run from the repository root: `poetry run python benchmarks/<script>.py`;
the three problem scripts take `--help`. `_common.py` holds what they
share: an application counter around the forward operator, timing, and
the markdown tables. Their `build` and `measure` functions run at any
size, which is how they were checked.
