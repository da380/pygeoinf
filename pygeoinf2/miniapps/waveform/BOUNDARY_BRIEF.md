# Brief: the absorbing boundary of the waveform mini-app (2026-10-29)

The snapshots (`snapshots.png`) show the trouble: at 0.87 s a wavefront
crosses the whole width going up, the reflection of the direct wave from
the bottom of the grid, and the shot gather carries the later arrivals it
produces. The maintainer's decision: the computational box becomes bigger
than the physical section, so that the sponge lies outside the section;
every figure but the snapshot one shows the physical domain only; then the
sponge is tuned by measurement. Delete this file when done; the README
carries the measurements.

## Read first

`README.md`, `survey.py` (the scales, `SPONGE`, `SPONGE_RATE`, the
interior mask, the receivers), `acoustic.cpp` (the taper and its place in
every kernel), `common.py` (the section plots), `synthetic.py`,
`proofs.py`, and the test `pygeoinf2/tests/test_miniapp_waveform.py`.

## The steps

1. **A reference for the reflection.** On the quick grid, the same shot on
   a grid padded by at least 200 cells on every absorbing side (so that
   nothing comes back within the run), the same source and receivers; the
   reflected field is the difference, at the receivers and in the
   interior at the snapshot times. Report its rms over the data relative
   to the data's rms, and its peak in the interior relative to the direct
   wave's. This is the number everything below is judged by; keep the
   helper as a function in `survey.py` or `proofs.py` and run it in the
   test at the quick scale.
2. **A bigger box.** Enlarge the computational grid so that the physical
   section (2 km wide, the depth the README states) is the interior and
   the sponge lies entirely outside it: the model space stays the periodic
   box over the whole grid (the prior as now), the sources and receivers
   are placed within the physical section, the boxes of part D stay in
   it, and the interior mask is the physical section. Rebalance the full
   scale so the stored fields stay under 500 MB (fewer cached fields,
   coarser snapshots, a shorter record if needed; say what you did).
3. **Tune the sponge.** With the reference of step 1, sweep the sponge
   width (20, 30, 40, 60 cells) and the Cerjan rate (a few values around
   the one that gives a taper near 0.9 at the outer edge for each width)
   on the quick grid; confirm the chosen pair on the full grid. The
   adjoints already carry the taper, so this is a parameter change; the
   check ladder (part A) must still pass to rounding. Target: the
   reflected rms below 1 % of the data's rms and no reflected arrival
   visible in the gather. If the sponge cannot reach that at a width that
   fits the memory budget, say so with the numbers and stop; a one-way
   boundary or a perfectly matched layer is a separate decision.
4. **Figures.** Every figure shows the physical domain only (models, the
   MAP, the Laplace standard deviation, Occam, the data gathers' receivers
   within it), except the snapshot figure, which shows the whole grid with
   the physical section outlined and the sponge marked. Titles on every
   figure as the other apps now have them: the app, the part and what the
   figure shows.
5. The README's scheme section states the box, the sponge's width and
   rate, and the measured reflection before and after; the test checks the
   reflection at the quick scale against the stated bound.

## House rules

As before: no decision numbers, planning or history talk in code or docs;
American spelling; ruff clean; `MPLBACKEND=Agg`; quick scale first; no
run over two minutes (cap or shorten and say so); the app's slow test
(`-m ""`) and the whole fast suite once at the end; touch only
`pygeoinf2/miniapps/waveform/` and its test; rebuild the extension with
`build.py` after any change to `acoustic.cpp`; do not commit. Report: the
reflection numbers before and after, the box and sponge chosen, the
memory at the full scale, the figures changed, anything undone.
