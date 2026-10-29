# Brief: every mini-app's set answer hardens with the ambient ball

Working directory `/home/david/dev/pygeoinf`, package `pygeoinf2`, run with
`poetry run`. Read `pygeoinf2/DECISIONS.md` entries D-191 to D-194 first (D-194
is the decision this brief carries out), then `GaussianMeasure.ambient_ball`
in `pygeoinf2/probability/gaussian.py` as it now stands, then each app's set
part and its tests.

## The decision

`GaussianMeasure.weakened_ellipsoid` is gone from the library. The convenience
hardening of a Gaussian prior into a constraint set is `prior.ambient_ball(
level=)`: the ball of the space's own norm with the radius at which a draw lies
inside with the given probability. Its radius converges under refinement for
any trace-class prior, it is the classical Backus–Gilbert–Parker norm bound
with the prior used only to calibrate the radius, and it has no knob but the
level. The same method takes `operator=B` (a bounded, self-adjoint, positive
definite operator on the space, with `inverse=` for the ellipsoid's
covariance and so its support function) and then returns the ball of the inner
product `B` defines, an `Ellipsoid`: a constraint set of another shape, still
convergent because `B` is bounded.

## What to change, app by app

Six apps call `weakened_ellipsoid` in their set parts: `point_data/sets.py`,
`tomography/sets.py`, `mfem/box_inverse.py`, `waveform` (check), `planet_density/src/answers.py`,
`cpp_backend/sets.py`. In each:

- Replace the ellipsoid with `prior.ambient_ball(level=scale.level)`. Where a
  coarse surrogate prior's ellipsoid preconditioned the family (point_data),
  the surrogate becomes the coarse prior's ball; the family swept is then
  `A A* r^2 + t R`, and the Woodbury or spectral solver takes the ball's
  `covariance`/`precision` as before.
- Remove `set_power` from the `Scale` dataclass, the CLI and the README's
  option table. Remove `quantile_method="matched"` only if the ball's own
  `"auto"` choice is affordable at full scale; the ball's diagonal route is
  exact and free on the symmetric spaces, and `"auto"` picks the matched
  quantile for long spectra by itself.
- Re-measure every number the README quotes for the set part at the full
  scale, and the command-line override examples that touch it. Quick scale
  first; no single command over two minutes (point_data part E at full ran
  about 4.5 minutes with the ellipsoid; measure direction by direction if it
  is still over two, as before, and say so in your report).
- Update the README's set section and the IMPLEMENTATION.md walkthrough: the
  set is the prior's ambient ball, why it converges where the credible set
  does not, and that it is a convenience, one constraint set among those a
  user can build. No mention of the removed method, no before/after, no
  decision numbers.
- The app's test asserts follow (the ball's `radius`, the brackets' new
  values where they are asserted).

**One app shows the operator form**: `cpp_backend`. Its space is `H^1` with
the Gram `G = M + l^2 K`. Add to its set part a second constraint set, the
`L^2` ball calibrated by the prior: `operator=` the mass inner product seen
from `H^1`, `x -> G^-1 M x`, with `inverse=` `x -> M^-1 G x`, both through the
backend's Cholesky factors (`cppvec` has the mass factor; add a `solve` for it
if only the Gram one exists). Everything still goes through `cppvec.Vector`.
Show both sets' brackets in the README (the `L^2` ball binds less where the
prior is rough, more where it is smooth; say what the numbers show), keep the
tally, and walk through the construction in IMPLEMENTATION.md: why the
`L^2` form is a bounded operator on `H^1`, how the adjoint through the Riesz
map makes it self-adjoint there, and that the library checks the traits.
Add a test.

Also look at `pygeoinf2/tutorials/14_*.py` and `pygeoinf2/examples/` for any
remaining `weakened_ellipsoid` call (there should be none) and leave the
tutorial's ambient-ball usage as it is.

## Rules

- `MPLBACKEND=Agg`; never show a figure; quick scale first.
- `poetry run ruff check pygeoinf2/` and `ruff format` clean.
- American spelling. No decision numbers, history, planning or v1 talk in
  code, docstrings, tests, README or IMPLEMENTATION.md.
- No library edits beyond the one C++ addition inside the app's `cpp/`. If the
  library gets in the way, record the exact reproduction in your report and
  work around it in the app.
- Run each app's slow test: `MPLBACKEND=Agg poetry run pytest
  pygeoinf2/tests/test_miniapp_<app>.py -m "" -q -p no:cacheprovider`.
- Delete this brief when done. Do not commit.

## Report

Per app: the set numbers before and after at full scale, timings, the test
counts; the cpp_backend operator-form numbers for both sets; anything in the
library that got in the way, with reproductions; anything you could not do.
