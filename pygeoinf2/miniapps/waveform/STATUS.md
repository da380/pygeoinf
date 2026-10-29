# Status of BOUNDARY_BRIEF.md (stopped 2026-10-29 for a shutdown)

Nothing committed. `acoustic.cpp` is unchanged (the sponge is a parameter),
so the built extension is current. `ruff check pygeoinf2/` passes and the
app's files are formatted. The slow test has NOT been run since the changes
and will fail as it stands (see "remains").

## Done

**Step 1, the reference.** `proofs.reflection(scale, pad=200)`: every shot
over the true model on the scale's grid and on one padded by `pad` cells of
the same medium on each absorbing side (`Survey(scale, pad=...)`); returns
`data` (rms of the reflected data over the data's rms, all shots), `interior`
(peak of the reflected field over the section at the four snapshot times,
over the reference field's peak there, worst shot), `times`, `reflected`,
`reference`. Part A's `run` prints it. Padding 200 and 300 cells give the
same reference to the last bit, so nothing returns within the run.

**Before** (old layout, quick grid, 20 cells at 0.015 inside the grid):
reflected data 4.2% of the data's rms; interior peak 8.2% of the direct
wave's (8.2/7.2/8.2% per shot).

**Step 2, the bigger box** (in `survey.py`). `Scale.nx, nz` now count the
section's cells: QUICK 100 x 40 at 20 m, FULL 200 x 80 at 10 m (both
2 km x 0.8 km); `Scale.sponge`, `Scale.sponge_rate` (defaults `SPONGE`,
`SPONGE_RATE`). The grid adds `margin = pad + sponge` cells left, right and
below: QUICK 148 x 64, FULL 248 x 104. `survey.x` runs from the section's
left edge (negative in the left sponge), `survey.window` slices the section,
`survey.section_lengths`, `survey.lengths` is the grid's (model space).
Receivers every `receiver_spacing` across the whole section (QUICK 50, FULL
67); sources at the same km positions. `interior()` = the section less the
surface row. The model space and prior are over the whole grid as before.
Background velocity: linear to `SECTION_DEPTH` = 0.8 km, then levels off
as `0.8 + L tanh((z - 0.8)/L)`, L = 0.2 km (max 2.4 km/s), because a
continued gradient in a deep sponge breaks the stability limit for prior
draws (at quick, 40 cells would reach 2.76 km/s, a 2-sigma margin).
PROOF keeps a 64 x 32 grid: section 24 x 12, its own sponge 20 / 0.015,
shots at 0.06 and 0.18 km. FULL `curvature_jobs=1`: 3 cached forward fields
+ 1 linearized = 4 x 124 MB = 495 MB at most. `wave_operator.unfused_*` now
run one shot at a time from their own forward field after `clear()`, so the
proofs also hold at most four fields; `proofs.own_defects` re-fetches the
cache after each unfused call.

**Step 3, the sweep** (quick, 400 steps; data % / interior %):
- width 20: rate 0.0080 1.07/9.6, 0.0100 1.18/7.2, 0.0162 (edge taper 0.9) 3.6/8.5
- width 24: 0.0065 0.57/1.9, 0.0070 0.52/1.5, 0.0080 0.70/2.0, 0.0090 1.08/3.3, 0.0135 (0.9) 2.8/7.0
- width 30: 0.0050 0.31/1.3, 0.0065 0.60/1.9, 0.0108 (0.9) 1.9/5.5
- width 40: 0.0030 0.10/0.35, 0.0081 (0.9) 1.0/3.7
- width 60: 0.0020 0.04/0.16, 0.0054 (0.9) 0.40/1.5
The best rate is well below the "0.9 at the edge" one (edge taper ~0.96-0.97).
At quick, 40+ cells (0.8 km) are so deep that the edge echo hardly returns
within the 1.44 s record, which is why weak tapers look so good there.
Full grid, 600 steps: width 24 at 0.008 gives 0.57/3.8 (gather: reflected
peak 1.9% of the 99th-percentile clip, invisible); width 30 at 0.0065 about
0.45/2; width 40 at 0.005 0.22/0.64 but 161 MB a field (645 MB for four).
Width 24 is the widest whose four 600-step fields fit 500 MB.
**Chosen: `SPONGE = 24`, `SPONGE_RATE = 0.008`** (edge taper 0.964).
After: quick 0.70% data / 1.96% interior; full 0.57% / 3.82%. The full
snapshot at 0.87 s still shows a faint upgoing arc from the bottom.

**Step 4, figures (code done):** `common.section` shows the section only;
`section(..., whole=True)` the whole grid; `common.sponge_outline` now draws
the section's outline and hatches the sponge (whole-grid plots only). Only
the snapshot figure uses `whole=True`; the outlines were removed from the
data, MAP and Occam figures. Titles already "Waveform, part X: ...".
Regenerated: `quick/*.png` (all five), full `snapshots.png`, full
`data.png`. Part B's setup now prints the grid, the sponge and the memory.

## Measured with the new layout

- Part A full: 52.5 s, all identities at rounding (dot tests 7e-16,
  check_operator 3.8e-14, check_traits 1.3e-13, check_gradient 2.4e-7).
  Part A quick: 14 s.
- Quick B-E: C 37 s, D 10 s, E 19 s. Relative errors over the new section
  are larger: MAP 0.54-0.56, Occam 0.37 (over the old 1.2 x 0.6 km
  interior the MAP's is 0.25, as before: 79% of the error now lies at the
  section's edges and bottom, which the old sponge covered). Laplace sd
  ratios 0.084, 0.122, 0.204, 0.283; all truths inside.
- Full C is over two minutes as configured: Gauss-Newton 39 s (6
  iterations, error 0.665), polish (2 iterations, inner cap 60) ~90 s,
  exact from the background (3 iterations) 21 s. The exact Hessian costs
  ~0.8 s an application with the curvature's shots serial. Polish with
  cap 30: 50 s, |grad| 1200 -> 571 -> 661 (a poor demonstration).

## Half done / remains, in order

1. **Full part C under two minutes**: decide the cut (options: one polish
   iteration at inner cap 60, about 45 s; or drop the exact run from the
   background at full; or exact_iterations=2). Set it in `FULL` in
   `survey.py`, rerun `newton.py` (full) with a 118 s cap.
2. **`modes_full_2026.npz` is stale** (old 200 x 80 grid): parts D and E at
   full will fail to use it until part C reruns and rewrites it. Then run
   `occam.py` and `laplace.py` at full (each capped at two minutes; check
   their times, the grid is 1.6x bigger) and `map.png`, `occam.png`,
   `laplace.png` at full are regenerated.
3. **Test** `pygeoinf2/tests/test_miniapp_waveform.py`: add a test of
   `proofs.reflection(QUICK)`: `data < 0.01`, `interior < 0.03`; the
   MAP-error bounds `< 0.4` will now fail (0.54-0.56): loosen to about 0.7
   or measure over the section explicitly; `std < 0.3 * prior_std` passes
   only narrowly (0.283), loosen to 0.4. Then run the slow test, then the
   whole fast suite once.
4. **README**: the problem and scales paragraphs (section 2 x 0.8 km at
   both scales, grids 148 x 64 and 248 x 104, receivers across the section,
   the velocity leveling below 0.8 km, the proof grid's own sponge), the
   scheme section (box, sponge 24 at 0.008, reflection before 4.2% / 8.2%
   and after 0.70% / 1.96% quick, 0.57% / 3.82% full, the sweep table),
   the defects table (rerun numbers above, full column from the part A
   run), every full-scale number of B-E, "Storage and memory" (4 fields,
   495 MB, curvature_jobs=1), "relative error over the section", and the
   F section's timings, which were measured on the old grid (no MPICXX
   here, so part F was not rerun; the MPI code path needs no change).
5. Delete `BOUNDARY_BRIEF.md` and this file when all of it is done.
