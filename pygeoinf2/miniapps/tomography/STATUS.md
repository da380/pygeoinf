# Status of MIXTURE_BRIEF.md (stopped 2026-10-29 for a shutdown)

Every file is syntactically whole; `ruff check pygeoinf2/` and `ruff format`
pass. Nothing committed. Keep MIXTURE_BRIEF.md until the remaining items are
done, then delete it and this file.

## Done

- `mixture.py`, part E, written and run at both scales (quick 1.4 s, full
  about 5 s). Five candidates at full scale, three at quick; the truth at
  candidate (0.80, 330), amplitude 3%, plume width 0.06 / 0.12 (the
  `PLUME_WIDTH` dict), background half a draw (`BACKGROUND = 0.5`), sweep at
  noise x0.1 to x100 (seven levels, 32 / 8 redraws), plus the "confusion"
  weights (each candidate's own noise-free travel times). Four figures,
  `mixture_*.png`, titled "Tomography, part E: ...".
- `point.bump` takes `extension=` (default "fit", unchanged for part B);
  part E uses "constant" because a fit of the 0.06 bump leaks into the
  padding at full scale.
- `common.py`: `mixture_marginal`, `mixture_density`, `grouped_components`,
  `mixture_marginal_panels` (exact mixture densities with weighted
  components).
- Test: `mixture` in NAMES, `results["e"]` in the quick fixture,
  `test_the_mixture_finds_the_plume`,
  `test_the_mixture_posterior_on_the_properties`, mixture figures in
  `test_the_figures_were_written`. Only the mixture tests have been run
  (`-k "mixture or figures_were_written"`: 4 passed).
- README: part E bullet in the parts list, how-to-run, test paragraph, file
  table, the **E** answers bullet, the whole "A plume among candidates"
  section (full-scale numbers), the "What the app shows" bullet for E.
- Letter changes, code: `ball.py` is part F (docstring, four titles, banner,
  "F took", `part_rng(seed, "f")`); `bent.py` is part G (docstring, "parts A
  to F", `default_rng([seed, ord("g")])`, banner, title, "G took",
  `part_rng(seed, "g")`); `run_all.py` runs E = mixture, F = ball,
  G = bent, docstring updated.
- Letter changes, README: parts list (F ball, G bent), "part G's derivative
  check", file table "parts A to G", answers bullet **F** for the ball,
  "Part G at its full size", "operator of parts A to F".
- Ball rerun at full after the key change: every printed number unchanged
  (only the sampled-std figure moves).
- `bent.py` FLOOR lowered from CORE_RADIUS + 0.03 to + 0.02 (comment says
  why): with the key "g" the derivative check's draw 1 + step 20 turns a
  ray at r = 0.5793 and the old floor 0.58 refused it, so part G failed at
  full size. With 0.57 it runs (49 s). Quick bent passes with key "g".

## Half done: README numbers after the letter changes

1. "Two scales" paragraph (README around line 113) still says "parts A to
   D", "The ball, part E", "Part F has its own two sizes", "about 80 s".
   Measured: quick A to E 5.9 s; full A to E 13.8 s (setup 2.4, B 1.2,
   C 2.3, D 2.9, E 4.6, with OMP_NUM_THREADS=4 and other agents running);
   ball F 13 to 15 s; G 26 s quick, 49 s full; run_all --quick 46 s.
2. "Bent rays" section: every number is still from the old key. New full
   size numbers (key "g"): truth measures 44.1; derivative size 0.28;
   table (reference, draw 1, draw 2): step 20: 5.8e-4 4.1e-4 6.5e-4;
   10: 1.5e-4 1.1e-4 1.5e-4; 5: 4.2e-5 7.2e-5 6.6e-5; 1: 1.5e-5 9.1e-5
   1.6e-4; 0.1: 1.4e-5 1.3e-4 1.9e-4. Floor 1.4e-5 at the reference,
   1e-4 to 2e-4 at the draws. Quick integrator check (step 0.005 vs
   0.0025, smallest FD step): reference 9.6e-7 -> 2.6e-7, draw 1
   3.0e-5 -> 4.5e-6, draw 2 1.3e-5 -> 8.2e-6 (not cleanly second order;
   reword). Bending: std 1.43; bent less linear 0.163 rms, 0.514 at most;
   rays move median 0.0162, at most 0.0610; deepest points up to 0.0537.
   Answers (49 s, 36 of them the derivative check): linear mean chi-squared
   1.1 on 32 as it believes, 392.9 along its own bent rays; mode 9
   iterations, 10 tracings, 9.0 s, chi-squared 0.6; differs from the linear
   mean by 16%, in the upper mantle under the stations; well-covered box
   mean +0.591 (linear) -> +0.500 (mode), truth +0.608 (moves away);
   poorly covered -1.186 -> -1.213, truth +0.049; error over the mantle
   0.779 -> 0.796, under the stations 0.565 -> 0.556. Add a sentence on the
   floor (0.57, two hundredths above the core; the check's deepest ray
   turns at 0.58).
3. "What the app shows", last bullet (bent): "a few times the noise",
   "changing the model by only a few percent" are now wrong (up to ten
   times the noise, 16%, not closer to the truth over the box).

## Remaining

- Items 1 to 3 above.
- The tomography row in `pygeoinf2/miniapps/README.md`: add the mixture part.
- Run the app's slow test in full:
  `MPLBACKEND=Agg poetry run pytest pygeoinf2/tests/test_miniapp_tomography.py -m "" -q -p no:cacheprovider`
  then the whole fast suite once.
- Delete MIXTURE_BRIEF.md and this file; report (weights across the sweep
  at full scale are in the README's sweep table).

## For the report

- Full-scale weights at the app's noise: 1.6e-5, 0.018, 0.923 (truth),
  0.026, 0.032. Average weights across the sweep: in the README table;
  effective number 1.23, 1.25, 1.48, 2.41, 3.83, 4.77, 4.99 from x0.1 to
  x100. Below the app's noise the weights saturate (the shared covariance,
  0.49 a datum, dominates the noise); candidates 4 and 5 stay confusable
  (0.70/0.24 and 0.23/0.68, still 0.23 at x0.1).
- Choices: sweep extended to x100 (to x10 the weights never reach equal);
  32 redraws averaged; background half a draw (with a whole draw the truth
  wins in 9 of 10 seeds at full, 7 of 10 at quick); box marginals show the
  hypothesis only as a shoulder (plume shifts its component by 1.15 prior
  sd).
- Library notes, no edits: (a) `plot_densities`/`moments` sample a
  `GaussianMixture` (20000 draws a panel; 2 to 4 ms a draw here, 40 to 80 s
  a panel) though its marginals are closed form; (b) `project_function`
  with the default "fit" leaks into the padding for the 0.06 bump at full
  scale (warning; H^2 norm 677 against 0.5 with "constant"), which part B's
  bumps also trigger at full scale; (c) `LinearGaussianMixtureInversion`
  factorizes the identical normal operator once per component when the
  covariance is shared (harmless here, 0.1 s for five).
