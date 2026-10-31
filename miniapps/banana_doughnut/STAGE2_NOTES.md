# Stage 2: finished (2026-10-31)

Stage 2 is complete and nothing in it is left to resume. These notes keep
the measured numbers stage 3 and the README will want; delete them once
those carry what is needed. Nothing committed.

## What changed on resuming

- The model space is fixed: `model_space(scale)` is
  `ball.Sobolev(lmax, 2, 0.1, inner_radius=0.55)`, `Scale.length_scale`
  0.1 at both scales; no prior rebuilds it.
- Priors: `sobolev_prior(space, radial, lateral, *, order, std)` is
  `space.sobolev_measure(order, length_scale=, lateral_length_scale=,
  pointwise_std=)`. `Problem.priors` is a dict keyed by `(radial,
  lateral)`, every one of `Scale.prior_scales` (the isotropic
  `prior_lengths`, then `prior_pairs`), built in `build`;
  `Problem.prior(length, lateral=None)` returns one or makes and keeps a
  new one on the same space. `raw_pointwise_std` reports the uncalibrated
  profile from `sobolev_pointwise_variance`.
- Scale: `prior_lengths` quick (0.08, 0.32), full (0.04, 0.08, 0.16,
  0.32); `prior_pairs` quick ((0.05, 0.2),), full ((0.05, 0.2), (0.05,
  0.5)); `prior_std` 0.02; `lmax` 16 / 96; bands quick (0.024,), full
  (0.006, 0.024, 0.096).
- `src/common.py`: a tuple field whose items are tuples reads each item as
  one word with commas, `--prior-pairs 0.05,0.2 0.05,0.5`; `--help` shows
  them the same way. `symmetric_mesh` puts the colorbar in an inset of the
  drawing's own height; the kernels figure uses the constrained layout.
- `Problem.ray_operator` and `finite_frequency_operator` lost `space=`.
- `src/functionals.py`: the weight dropped is reported at each boundary
  apart, below the core-mantle boundary (physics) and above the surface
  (the paraxial approximation); `build` prints both per ray and band.
- `tests/test_miniapp_banana_doughnut.py`: 23 tests, slow set, about 33 s.

## Calibration

Pointwise (`pointwise_std=0.02`). The uncalibrated standard deviation is a
function of radius only and varies across the annulus by up to 15% at
degree 96 (0.951 to 1.094 of its mean for (0.05, 0.5); 0.952 to 1.032 for
0.32; within 0.4% for 0.04 and 0.08), so amplitude calibration would leave
up to about 8% variation. Calibrated, the standard deviation at points
through `E C E*` is 0.02 to 1e-3 at degree 96 and to 4% at degree 16 (the
projection onto the kept modes). Nothing in stages 2 or 3 needs a
precision: representers `C l`, push-forwards `B C B*`, draws and the
space's own dual norms all go through the factor.

## Timings at lmax 96 (dim 116299, radial grid 81, OMP 4 threads)

- Space 0.07 s; six priors 0.14 s; a draw 0.04 s; eight applications of a
  prior's factor adjoint 0.75 s.
- Point evaluation: 1e4 points 0.18 s, 1e5 points apply 1.7-1.8 s,
  adjoint 2.0 s, every time (the harmonics are recomputed per chunk).
- One finite-frequency functional (45 degrees, T0 0.006, 201,286 points):
  apply 2.3-3.2 s, adjoint 3.3 s. The ray functional: 0.02 s.
- `check_operator` on that functional: 214 s (some 80 applications); run
  it at the quick scale only. The ray functional's: 2 s.
- The quick space for comparison: the same functional 0.14 s; degree 48
  0.8 s.
- `setup.py` at the full scale: 6-7 s wall.

## Checks (quick scale unless said; the tests hold these)

1. Gaussian defect -2.476e-8 in every band; 32 and 64 nodes agree to
   1.8e-14 (the tail cut off sets it, not the count).
2. Constant, exact and unclipped: FF / ray - 1 equals the defect to 6e-16.
   Through the space: FF / ray - 1 + clipped = 8.6e-6 at degree 16 (the
   projected constant's departure), -2.48e-8 at degree 96 in every band
   (the defect itself). Radial `cos(2 pi (1 - r) / 0.9)` exact and
   unclipped, (FF - ray) / T: 6.5e-6, 1.6e-6, 4.2e-7, 1.2e-7 for T0
   0.024, 0.012, 0.006, 0.003.
3. A draw from the smoothest prior through the space, |FF - ray| over the
   ray delay's prior std, T0 0.024 / 0.012 / 0.006 / 0.003: median over 20
   draws 1.4e-3, 3.7e-4, 3.1e-4, 4.3e-6, worst 5.6e-3 ... 1.6e-5. Full
   scale, 45 degrees, T0 0.024 / 0.006 / 0.003 / 0.0015: prior 0.32 9.2e-4,
   1.3e-4, 6.9e-5, 1.7e-5; prior 0.04 0.038, 0.005, 0.001, below 5e-4;
   prior (0.05, 0.5) 8.5e-3, 5.2e-4, 1.1e-4, 2.6e-5.
4. `check_operator`, `measure=` each quick prior, rtol 1e-12: worst
   1.5e-13 (finite frequency, composed, under (0.05, 0.2)); at degree 96
   4.5e-14 (finite frequency) and 1.0e-14 (ray). Composed and assembled
   matrices identical.
5. Weight dropped, signed (absolute), reported apart at the two
   boundaries (`Quadrature.surface_clipped`, `core_clipped`, each with
   `_absolute`; `clipped`, `clipped_absolute` stay as the totals). Full
   scale, distances 30 / 45 / 60 / 75:
   - above the surface: T0 0.006: -0.19% (0.99%), -0.013% (0.14%),
     +0.001% (0.00%), +0.000% (0.00%); T0 0.024: +0.21% (6.7%), -0.099% (1.2%),
     -0.099% (0.28%), +0.080% (0.05%); T0 0.096: -4.3% (24%), +0.48%
     (10.5%), -1.4% (3.4%), -0.19% (0.92%).
   - below the core-mantle boundary: T0 0.006: 0, 0, 0, +0.11% (0.85%);
     T0 0.024: 0, 0, +0.027% (0.01%), -0.94% (4.4%); T0 0.096: 0, +0.032%
     (0.02%), -0.98% (2.1%), +3.9% (8.8%).
   The 75-degree ray turns 0.054 above the core, so its loss is nearly all
   below the core-mantle boundary, where the functional is zero by physics.
   The signed fraction moves with the nodes along the ray (quick 45
   degrees T0 0.024, surface: -0.31%, +0.037%, -0.088% at 101, 201, 401
   nodes); the absolute one is steady (1.05%, 1.23%, 1.35%). Quick 45
   degrees at T0 0.096: surface 10.3%, core 0.021% absolute.
6. The full quadrature (203,778 points, 201,286 kept at 45 degrees, T0
   0.006) on the quick space applies in 0.14 s.

Priors at the quick scale: `check_measure` (4000 draws, about 6 s each)
sample covariance worst 0.045 over three seeds; `check_traits` on the
covariance 3e-15.

## For the owner

- On smooth priors the weight dropped at the boundaries, first order in
  the band and uneven with the nodes, sets the FF - ray gap rather than
  the paraxial physics: at 45 degrees, T0 0.024, the ray and
  finite-frequency delays correlate at 0.99998 or more under priors of
  0.08 and longer (degree 48). Stage 3's relative standard deviations for
  long priors will mostly measure it: the surface's (the approximation)
  at the short distances, the core's (physics) at 75 degrees.
- The space's padding is sized to its own length scale, 0.2 each side,
  shorter than the longest prior lengths. Padding 0.64 changes the ray
  delay's prior std by 0.5% for prior 0.32 (degree 48) and the correlation
  of the two delays by less than 1e-6, and doubles the dimension; the
  default is kept.
