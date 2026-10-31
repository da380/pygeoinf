# Brief: the banana-doughnut mini-app

## Status: restarted 2026-10-31 on paraxial kernels

The first design (October 2026) took finite-frequency kernels from yspec,
sampled on the ball's grid, and failed on the integrable singularities of a
point source's kernel at the source and the receiver: sampled on a fixed
Gauss-Legendre grid and paired by its quadrature, the kernels aliased, and a
radially uniform perturbation came out 10 to 20 per cent wrong at every
affordable degree. That tree is withdrawn (DECISIONS D-212) and archived by
the owner at `/home/david/dev/banana_doughnut-pinned-2026-10-31.tar.gz`;
extract it to a scratch directory when a module of it is wanted, never into
the repository. What carries over is listed under "What is established".

This design computes the finite-frequency kernel itself, in the paraxial
approximation of Dahlen, Hung and Nolet (2000), *Fréchet kernels for
finite-frequency traveltimes, I. Theory*, Geophys. J. Int. 141, 157–174,
and never represents it on a grid: the functional is a quadrature of our own
in ray-centred coordinates composed with point evaluation of the model, the
same form as the ray functional. In the right coordinates the singularities
are not there at all. Equation numbers below are the paper's; everything
the work needs from it is transcribed in the appendix at the end of this
brief, checked for dimensions and against the constant-velocity chord, so
the paper itself is not required.

## What the app shows

A traveltime is a linear functional of the velocity perturbation. What
seismology calls its *kernel* is the `L2` Riesz representer of that
functional, `δT = ∫ K δln v dV`, and in `L2` ray theory and finite-frequency
theory look incomparable: the ray kernel is a delta on a curve, the
finite-frequency (banana-doughnut) kernel a hollow banana with no
sensitivity on the ray itself.

Both are functionals on a model space, though, and the comparison belongs
there. On the mantle in `H^s`, with `s > 3/2` so that point values exist,
both functionals are bounded and both have smooth representers, their
Sobolev gradients. Under a Gaussian prior with covariance `C`, what a datum
depends on is `C ℓ`, the representer in the prior's Cameron–Martin space,
which is also the covariance between the field and the traveltime. The
relative standard deviation of `(ℓ_FF − ℓ_ray)(m)` under the prior is one
number per ray, band and prior. For priors whose correlation length exceeds
the Fresnel width it should be small, though the `L2` kernels look nothing
alike, and it should grow as the prior roughens. The app makes that visible
and measures it: a point about choosing the function space, made with the
physics of finite-frequency traveltimes and no inversion.

The reference for the function-space side is Francis Syvret's Cambridge PhD
thesis, *Theoretical and computational studies in seismic tomography*
(https://www.repository.cam.ac.uk/handle/1810/339758): Sobolev model spaces,
Sobolev gradients as a post-processing of the derivative, and 2-D and 3-D ray
kernels shown in that setting. New here: finite-frequency kernels set beside
ray kernels in the same space, and the quantitative comparison through prior
push-forwards over a range of rays and bands.

The app makes a point; it is not an application. The planet is simple on
purpose, so that nothing in the ray theory distracts from the point.

## The theory, as the app uses it

**Ray theory.** For a phase travelling at speed `c(r)` along the ray traced
in the reference planet, `δT = −∫ m / c dl`, `m = δln c`, equation (115).
The tracer gives the nodes and the weights; the functional is point
evaluation at the nodes times the weights, as the tomography app has it.

**Finite-frequency theory, paraxial.** The cross-correlation traveltime of a
pulse with source-time power spectrum `|ṁ(ω)|²` is, to first order,
`δT = ∫ K m d³x` with, equations (108) and (109),

```
K = −(1 / 2πc) √|det(M' + M'')| · ∫ ω³ |ṁ|² sin Φ dω / ∫ ω² |ṁ|² dω ,
Φ = ½ ω qᵀ (M' + M'') q − [sig(M' + M'') − 2] π/4 ,
```

in ray-centred coordinates `x = ξ(l) + q1 q̂1 + q2 q̂2`: `ξ(l)` the point of
the central ray at arc length `l` from the source, `q̂1` the unit normal in
the ray plane, `q̂2` the normal to that plane, equation (A3). `M'(l)` and
`M''(l)` are the `2×2` Hessians of the traveltime from the source and from
the receiver at `ξ(l)`, diagonal in a spherically symmetric planet,
equation (A10), so only the two sums `A_i(l) = M_i'(l) + M_i''(l)` enter.
The volume element is `dl dq1 dq2`. Where no caustic lies between the ends,
`A1, A2 > 0` and the signature term vanishes; the app asserts this along
every ray and refuses a ray that fails it, rather than carrying the Maslov
phase.

**The scaling that removes the singularities.** Put `q_i = η_i / √A_i(l)`.
Then `√|det(M' + M'')| dq1 dq2 = dη1 dη2` exactly, `Φ = ½ ω |η|²`, and the
kernel's whole dependence on `l` sits in the map from `η` to `q` and in the
prefactor `1/c`, taken at `ξ(l)`, which section 6.5 says is immaterial.
With `u = |η|² / 2` and the angle `θ` of `η`,

```
δT = −∫₀ᴸ dl / (2π c(l)) ∫₀²π dθ ∫₀^∞ du  G(u)  m(ξ(l) + q(u, θ, l)) ,
q1 = √(2u / A1(l)) cos θ ,   q2 = √(2u / A2(l)) sin θ ,
G(u) = ∫ ω³ |ṁ|² sin(ωu) dω / ∫ ω² |ṁ|² dω ,   ∫₀^∞ G(u) du = 1 .
```

The last identity is the Gaussian integral (113); with it, a model constant
across the banana gives `−∫ m / c dl`, the ray functional, which is the
reduction to ray theory of section 7 and the exact check that the quadrature
must pass. Near either end `A_i → 1/(c l)` (equations (102) and (103)), so
the physical points `q` close in on the ray as `√(c l)` while the integrand
in `(l, u, θ)` stays bounded: nothing singular is ever evaluated. This is
the whole reason the design works where the sampled kernels did not.

**Dynamic ray tracing**, the one new ingredient. Along the traced ray, with
arc length as the variable, the forward Hessian is `M' = P Q⁻¹` with,
equations (A26), (A27), (A14) and (A35),

```
dP_i/dl = −c⁻² V_i Q_i ,   dQ_i/dl = c P_i ,   P_i(0) = 1 ,   Q_i(0) = 0 ,
V_1 = c̈ sin² i + ċ cos² i / r ,   V_2 = ċ / r ,
```

`ċ`, `c̈` the first two radial derivatives of `c`, `i` the angle of the ray
from the upward vertical (`sin i = p c / r`; `i > π/2` on the downgoing
leg), and `M_i' → 1 / (c_s l)` as `l → 0`, consistent with (102). The
backward Hessian `M''` is the same integration from the receiver along the
reversed ray. This is a linear, non-singular system; integrate it with
`scipy.integrate.solve_ivp` at tight tolerances, or with the tracer's own
Chebyshev machinery if that is natural, from the kinematic ray's `r(l)`,
`i(l)`. Checks that the paper supplies:

- the out-of-plane sum in closed form, equation (A51):
  `A2 = p sin Δ / (r² sin φ sin(Δ − φ))`, `φ` the angle swept from the
  source and `p = r sin i / c` the ray parameter in time per radian, as the
  tracer's `p` is; and (A50), `M2' = sin(i + φ) / (c r sin φ)`, for the
  forward one alone. A constant velocity checks the reading: at the chord's
  midpoint both give `2 / (c sin(Δ/2))`;
- a constant velocity: the ray is a chord of length `L` and
  `A1 = A2 = L / (c l (L − l))`;
- the flattened linear gradient (below): the in-plane sum against second
  differences in `q1` of the exact detour time `T_xs + T_xr − T_rs`, the
  one check that does not go through the paper's own equations;
- the in-plane conserved quantity (A53), transcribed in the appendix,
  constant along every ray to the integrator's tolerance.

Nothing in the repository does dynamic ray tracing today. `shoot` and `bend`
in the shared tracer are kinematic, for the tomography app's bent rays.

**Boundaries.** The two boundaries of the mantle are different in kind.
The outer core has no three-dimensional structure, so the functional is
zero there by physics: quadrature points below the core-mantle boundary
are dropped, and the model space is the mantle annulus alone, on which
every representer and every figure lives; nothing is ever computed or
drawn in the core. The free surface is where the paraxial kernel is
blind: it knows nothing of the surface, so the points of the doughnut
that fall above it, half of it near a surface receiver, would be lost,
up to a quarter of the weight at 30 degrees in the widest band, and on
smooth priors that loss, not the physics, would set the gap between the
two delays. The free surface is taken by images instead: for a receiver
on the surface the back-propagated field includes the surface reflection,
and for a locally flat surface the reflected leg's travel time from a
scatterer equals the direct leg's from the scatterer's mirror image, so
the kernel inside the mantle carries what the mirrored point would have;
a quadrature point above the surface is folded back to the same direction
at radius `2a − r`, keeping its weight, which for an SH-type phase (free
surface reflection coefficient plus one) adds with the same sign and
restores the Gaussian identity. `surface="fold"` is the default and
`"clip"` keeps the plain dropping, a field of the scale, so that both can
be shown. Report the fraction of the weight folded (or clipped) above the
surface and the fraction dropped below the core-mantle boundary per ray,
separately, and state the fold in the README in one sentence as the
free surface by images; it does not touch the point the app makes.

## The design

- **The planet** is the tomography app's: radius one, mantle on
  `[0.55, 1]` with `c(r) = 1 + 1.778 (1 − r)` (`linear_gradient` in the
  shared tracer, the constants in `miniapps/tomography/src/problem.py`), a
  fluid core below, dimensionless units throughout. One smooth piece: no
  discontinuities, no jump conditions, no branches; `Δ(p)` is monotone and
  no caustic lies between the ends of any ray short of the antipode. Rays
  turning above the core reach at most 82 degrees: keep distances between
  about 20 and 75 degrees.
- **The flattened linear gradient** (`flattened_linear_gradient`,
  `CircularArcs`) is the test planet. Its rays are circles and its two-point
  travel time is closed form, and because any two points of a sphere lie in
  a plane through the centre, that closed form gives `T_xs` for every point
  `x` in three dimensions, not only in the ray plane. `CircularArcs` has the
  time from a point at depth to the surface; add the time between two points
  at depth, with the turning point between them or not, checked against the
  tracer.
- **One phase**, S-like, no conversions, measured on the speed it travels
  at. Source at the surface (a shallow source is a field of the scale, in
  case a figure wants one), receivers at the surface.
- **The pulse**: `|ṁ(ω)|²` a Gaussian in `ω` centred on `ω0 = 2π / T0`
  with standard deviation `ω0 / 4`, so that `G(u)` decays like
  `exp(−σ² u² / 2)` and a few Fresnel zones hold everything. **Bands**:
  three dominant periods `T0`, chosen so that the Fresnel half-width at
  mid-path, `½ √(c T0 L)` with `c` at mid-path, is about 0.04, 0.08 and
  0.16 of the radius; the quick scale keeps the middle one. State the
  periods in the README with the travel times they are fractions of.
- **The quadrature**: `G(u)` tabulated once per band by Gauss–Legendre in
  `ω`; in `u` Gauss–Legendre on `[0, u_max]` with `u_max` where
  `σ² u² / 2` reaches about 30, with enough nodes for the oscillation at
  `ω0` (about five periods of it: 64 nodes is plenty, measure the Gaussian
  identity to confirm); the trapezoid rule in `θ`; Simpson or the trapezoid
  rule in `l` at the tracer's nodes. The counts are fields of the scale.
  Then the functional is one row: point evaluation at every surviving point
  of every `(l, u, θ)`, times the product of the three weights and
  `−G(u) / (2π c(l))`.
- **The model space**: `m = δln c` on the mantle annulus,
  `pygeoinf.ball.Sobolev(lmax, 2, 0.02, inner_radius=0.55)`: `H^2`, where
  a value at a point exists, with one fixed length scale of 0.02 in its
  inner product, which is the norm the Sobolev gradients and the dual
  norms are measured in. The space is the problem's and never changes
  with the prior. The scale is short on purpose: a prior's covariance is
  an operator in this metric, so the draws' `L2` spectrum carries the
  space's symbol as a factor, and the space's scale is a floor below which
  no prior can roughen its draws; at 0.02 that floor sits below every
  Fresnel half-width and every prior of interest. `lmax` is a field of the
  scale, 160 at the full scale (resolving about 0.02, so that the space's
  own scale and the shortest priors are above the truncation; 128 if the
  timings demand it, stated), 16 at the quick scale. Both
  functionals are built on it exactly as the tomography app builds its
  path integrals: point evaluation composed with a matrix of weights, the
  adjoint the library's.
- **The ray functional** has the same form with the ray's own nodes and
  weights over the travel time (`Ray.time_weights`), and is also the
  `T0 → 0` limit of the finite-frequency one: at the smallest period the
  two must approach each other on a smooth model, and they must agree to
  quadrature accuracy on a model constant across every banana.
- **Priors**: Sobolev-type measures of order 2 on that one space, each
  with its own length scales, through the library's
  `sobolev_measure(order, length_scale=, lateral_length_scale=)` on the
  spectral-element spaces, which builds the covariance
  `a² (1 − div(L_r² ∇) − L_h² Δ_h)^(−2)` in the space's own basis (block
  by degree). Never rebuild the space to change a prior. The isotropic
  lengths run from well below to well above the Fresnel width, about 0.02
  to 0.5 of the radius; and anisotropic pairs, a radial length shorter
  than the lateral one as in the mantle (say 0.03 radial with 0.2 and 0.5
  lateral), are part of the sweep. **A prior's scale is never read off
  its covariance's nominal length** (David, 2026-11-01): the metric makes
  that reading unreliable. Each prior is characterized by what its draws
  do: its correlation function from a reference point along a radius and
  along a shell, computed exactly from the covariance, its distance to a
  correlation of `1/e` in each direction, and a representative draw on
  the two sections, all shown in a figure of the priors; and the
  comparison's axes use the measured correlation distance, not the
  nominal length. **The reference length is the wavelength** (David,
  2026-11-01): each band's `λ = c T0` at mid-path; every length is quoted
  in units of the radius and in wavelengths, and part C's primary axis is
  the prior's measured correlation distance over the band's wavelength,
  with each band's Fresnel half-width marked on it in the same units, so
  that the three bands' curves line up where the physics is the same. The pointwise standard deviation of
  `δln c` is a few per cent, as in global tomographic models: 0.02 by
  default, by the library's own `pointwise_std=` calibration, which makes
  it 0.02 at every point (everything is linear, so it only scales the
  answers; nothing here needs a precision, so the pointwise rescaling
  costs nothing). Every length and the standard deviation are fields of
  the scale.
- **The `L2` pictures** are drawn from the formula, not from the space: the
  paraxial kernel evaluated on a fine grid of a section through the ray
  plane and of a section across the ray at mid-path, with the ray drawn as
  a line. The Sobolev gradients come from the space, the representer of
  each functional through the space's Riesz map, drawn on the same two
  sections. The across-path view is the point: the doughnut's hole filling
  in.
- **No new dependency.** NumPy, SciPy, matplotlib and the library's own
  extras (`planetmodel` for the ball) are all the app uses.

## How the work runs

Four stages, in order, each ending with its checks passing. At the end of
every stage, report: what was built, the checks run and their numbers, the
test results, timings at each scale, any library finding, and anything
decided that this brief did not settle. Press on from one stage to the next;
stop only for something serious that needs the owner's decision.

## House rules

- **Never commit**, stage or push. The repository's owner commits by hand.
- **Library changes are not part of this work.** If `pygeoinf/` lacks
  something the app needs, or something in it is wrong, stop and report it:
  what is missing, a minimal reproduction, and what you would add. Do not
  work around it in the app and do not edit `pygeoinf/`.
- **Layout and conventions** are those of the other apps (read
  `miniapps/README.md` and the tomography app): modules in a package
  `src/`, one short executable script per part, `run_all.py`, `README.md`
  and `IMPLEMENTATION.md` at the top, output in `results/` (`results/quick/`
  at the quick scale), never a monolithic file, no `pyproject` of the app's
  own. Every script takes `--quick` or `--full` (the default) and every
  field of the app's scale as an option of its own name; `--help` lists
  them with the preset's values. Follow `miniapps/tomography/src/common.py`
  for the command line.
- **README**: the problem, the results and what they mean, with figures. No
  development narrative, no before and after. **IMPLEMENTATION.md**: how
  the library is used, nothing else.
- **No decision numbers, planning or history** in code, docstrings,
  comments or tests; docstrings are self-contained. American spelling in
  code and docstrings, as in the other apps.
- **Figures**: titled, self-explanatory, saved, never shown. Run everything
  with `MPLBACKEND=Agg`. Headings are short (David, 2026-11-01): one
  one-line suptitle naming the figure, panel titles of a few words with
  the one parameter that varies, short colorbar and axis labels; every
  other number goes to the printed report and the results table. The
  ray-kernel panels show just the ray, a line or a cross, nothing else.
- **Threads**: give every loop over independent rays or bands an `n_jobs=`;
  set `OMP_NUM_THREADS` for runs by hand; one heavy job at a time.
- **Diagnose small before large**: time one functional at the quick scale
  before anything bigger, then one capped middle size, then the full scale.
  Never repeat large runs to find a problem.
- **Function spaces**: the space comes from the problem; a prior is a
  measure on it. Never an identity covariance on a function space: every
  prior is a Sobolev measure of order above `d/2`.
- **Tests**: tolerances sit above what the arithmetic produces; measure the
  spread before choosing one. Checks of operators take a realistic
  `measure=` (a prior), not white noise.

## What is established

### The shared tracer (`miniapps/raytracer/raytracer.py`)

`RadialProfile(velocity, radius, bottom, discontinuities=)`: ray parameter,
turning radius, `Δ(p)`, `T(p)` by Chebyshev series in `s` from the turning
point, piece by piece; `trace(source_angle, receiver_angle, nodes=,
source_radius=, receiver_radius=)` gives a `Ray`: nodes `(radius, angle)`
at equal arc length, path weights, the travel time and `time_weights`.
`linear_gradient`, `flattened_linear_gradient` and `CircularArcs` are there;
so are `shoot` and `bend`, kinematic. Tests in
`tests/test_miniapp_raytracer.py` (fast, NumPy and SciPy only). The
tomography app's `src/rays.py` loads the module with a temporary
`sys.path` insert and re-exports it; the new app's `src/rays.py` does the
same (the archive has it).

### The ball

`pygeoinf.ball.Ball(inner_radius=0.55)` is the mantle annulus;
`point_evaluation_operator`, `sobolev_measure`, the Riesz maps and the
plotting of shells, sections and profiles (`pygeoinf/plotting/ball.py`)
exist. The tomography app's `ball.py` and `src/problem.py` (`path_weights`,
`lift`, `path_integral_operator`, `composed_operator`) show a ray in a ball
through point evaluation at its nodes, and how a node `(radius, angle)` in
the ray plane becomes `(radius, latitude, longitude)` on the great circle
through the ends.

### The archive

From the withdrawn tree, worth taking after reading: `src/rays.py` (the
re-export), `src/common.py` (the command line and figure helpers),
`functionals.py`'s `unit`, `direction` and `lift`, and the shape of
`problem.py`'s `Scale`, `Problem` and `build(scale=)`. Everything that
touched yspec (`earth.py`, `survey.py`, `phases.py`, the deck, the kernel
cache) is gone for good.

## Stage 1: dynamic ray tracing in the shared tracer

**Done (2026-10-31).** `RadialProfile(..., derivatives=)` takes the
velocity's first two radial derivatives (one pair, or one per piece) and
differentiates a Chebyshev series of the velocity where none are given;
`Ray` carries `ray_parameter` and `arc_length`;
`RadialProfile.dynamic_ray(ray)` returns a `DynamicRay` with per-node
`forward`, `backward` and `sums` (`(count, 2)`, `M1`, `M2` and `A1`, `A2`,
infinite at the ends), the `P` and `Q` of both integrations, `incidence`,
`velocity` and a `Frame` whose `place(q1, q2)` gives `ξ + q1 e1 + q2 e2`
in the ray plane's coordinates (node `(r, θ)` at `(r cos θ, r sin θ, 0)`,
`e2 = (0, 0, −1)` along an anticlockwise ray). A ray crossing a jump in
the velocity, or with a caustic between its ends, is refused.
`CircularArcs.between_depths(distance, source_depth, receiver_depth)` and
`linear_gradient_derivatives`, `flattened_linear_gradient_derivatives`
exist. Checks: (A51) and (A53) to about `1e-11`, the chord to `1e-13`, the
in-plane sum against the exact detour time to `1.4e-9`, forward against
backward to `3e-12`; 47 fast tests added, the tomography tests unchanged.

Add to `miniapps/raytracer/raytracer.py`, beside the kinematic tracer and
knowing nothing of the library:

1. Along a traced `Ray` of a `RadialProfile`, the forward and backward
   Hessians `M'(l)`, `M''(l)` and their sums `A1(l)`, `A2(l)` at the ray's
   nodes, by the arc-length system above, with the ray's `r`, `i`, `c`,
   `ċ`, `c̈` along it. The profile needs the velocity's first two radial
   derivatives: take them as optional callables beside the velocity, and
   fall back to differentiating the Chebyshev representation the profile
   already builds if that is accurate to the same order; say which.
   Refuse a ray along which either sum changes sign.
2. The ray-centred frame at every node: `ξ`, `q̂1`, `q̂2` as vectors in the
   ray plane and its normal, so that a point `ξ + q1 q̂1 + q2 q̂2` is one
   call away, in the plane's coordinates; the app lifts to the ball.
3. `CircularArcs`: the time between two points at depth, as above.

Tests, in `tests/test_miniapp_raytracer.py`, fast: (A51) at every node of
several rays; (A50) for `M2'`; the chord in a constant velocity; the
flattened profile's in-plane sum against second differences of the exact
detour time at several points along the ray (measure the finite-difference
error's floor before choosing the tolerance); forward from the receiver
equals backward from the source. The tomography app's existing tests pass
unchanged.

## Stage 2: the functionals on the ball

**Done 2026-10-31** (its numbers are now in the README and IMPLEMENTATION.md): the package,
`setup.py` at both scales and `tests/test_miniapp_banana_doughnut.py`
(22 tests, 30 s). At `lmax` 96 a finite-frequency functional has 201 286
points and applies in 2.3 to 3.3 s; the space, its six priors and a draw
cost nothing beside it; a library operator check at the full scale takes
214 s and belongs at the quick scale only. Established for stage 3: on
smooth priors the weight lost above the surface, not the physics, sets
the gap between the two delays (at 45 degrees in the middle band the two
correlate at 0.99998 or better for priors of 0.08 and longer), so the
relative standard deviation of the difference for long priors largely
measures the clipping; say so where it is shown. The space's padding of
0.2 is shorter than the longest priors; padding of 0.64 moved the ray
delay's prior standard deviation by 0.5 per cent for the 0.32 prior and
the correlation of the two delays by under `1e-6` while doubling the
dimension, so the default stays. The quick scale's priors are below what
`lmax` 16 resolves and are smoke-test values only. Three corrections from
the owner's review were applied on resumption:
the model space is fixed as above and the priors carry their own lengths
through the library's new `length_scale=` and `lateral_length_scale=`
options (the agent had rebuilt the space with each prior length:
`model_space(scale, length_scale=)` and `Problem.prior_space` go, the
prior lengths become a list in the scale, and anisotropic pairs join it);
`lmax` 96 at the full scale with the narrowest band kept; the prior's
standard deviation 0.02. The clipping stays as it is, the two fractions
reported separately (below the core-mantle boundary, which is physics,
and above the surface, which is the approximation noted in the README);
the second shrinks with the narrower bands.

The app's `src/`: the planet (from the tomography constants), the bands
and `G(u)`, the quadrature in `(l, u, θ)` with its clipping, both
functionals as operators on the annulus in `H^2`, the priors, `Scale`
with quick and full presets, `Problem` and `build(scale=)`. Then part A,
`setup.py`: the rays at the chosen distances, the Hessian sums along them,
and the `L2` kernels drawn from the formula on the two sections for each
band, the ray as a line.

Checks, which become the app's tests: the Gaussian identity on the
quadrature per band (report the defect); the two functionals equal on a
model constant across every banana (a radial model of long wavelength,
and the constant one); the finite-frequency functional approaching the
ray's as `T0` falls, on a smooth random draw from a prior; each functional
through `pygeoinf.testing`'s operator checks with a prior as the measure;
the fraction of weight clipped per ray, reported. Time the full-scale
functional once at the quick scale and once at a capped middle size before
the full scale.

## Stage 3: the comparison

**First half done 2026-11-01**: the fold (`surface="fold"|"clip"`, the
Gaussian identity and the constant model exact with the fold on; the
smooth-draw gap in the quick band seventeen times smaller; the gap at 45
degrees in the middle band unchanged and physics), part B
(`gradients.py`) and part C (`agreement.py`, all four distances in 41 s
at `lmax` 96, with a ratio axis). Found on the way: with the space at
0.1 no prior decorrelated below 0.19, every prior smoother than every
Fresnel half-width, so the comparison never reached the regime where ray
theory breaks down; hence the space's scale of 0.02, the measured
correlation distances and the figure of the priors above, and the sweep
redone at `lmax` 160.

**Redone 2026-11-01 at the space's scale of 0.02**: `lmax` 160 and
padding 0.32 at the full scale (padding is a field of the scale: the
default of two space scales moved the longest prior's delay standard
deviation by 6 per cent; 0.32 against 0.64 by 1.2), dimension 659 316,
one finite-frequency adjoint 10 s, a push-forward 10 s, 2.1 GB. A new
part, `priors.py` with `src/correlations.py`, measures and draws each
prior; `src/sections.py` draws the sections for parts B and priors. The
measured `1/e` distances along the shell run from 0.058 (nominal 0.02) to
0.37 (nominal 0.32), the anisotropic pairs 0.07 radially; in wavelengths
the sweep reaches 7 to 47 in the narrow band, 1.8 to 12 in the middle and
0.46 to 2.9 in the widest. The breakdown shows: in the widest band under
the shortest prior the correlation of the two delays is 0.70 at 45
degrees (0.63 and 0.59 at 60 and 75) and the slope 0.54; above a few
wavelengths both are within a per cent of one. On the wavelength axis the
three bands' spread curves nearly collapse; on the Fresnel axis the slope
curves do, so both figures are kept. Part B's 2-by-2 figures: the `H^2`
gradient's hole persists in the widest band (fill 0.52) and fills under
the priors (0.87 and above, the pairs 0.74 to 0.92). Full-scale times:
part A 9 s, priors 40 s, part B 4 min, part C 8 min; the slow set 35
tests in 108 s. The narrow band is resolution-sensitive (`1 − ρ` under
the shortest prior 1.8e-3 at `lmax` 160 against 4.5e-4 at 96): say so in
the README.

Part B, `gradients.py`: one figure per band and prior, a 2-by-2 grid on
one section (David, 2026-11-01): top row the `L2` picture, the ray kernel
as the ray drawn as a line beside the banana-doughnut kernel from the
formula; bottom row the Sobolev equivalents under that prior, the two
functionals' Cameron-Martin representers `C ℓ`, ray's and
finite-frequency's, on one shared color scale per row. Two sections, the
across-the-ray one at mid-path (the figure that carries the point) and
the ray plane; the set made for one fixed band and prior first, then
repeated for every band and prior of the scale, each its own file, titles
stating the band, its Fresnel half-width, the prior and its measured
`1/e` distances. The `H^2` gradient in the space's own metric may be
shown the same way as the case of no prior. With the space's scale below
the Fresnel widths the hole persists in the `H^2` gradient of the wider
bands and fills under the priors: say so in the captions.

Part C, `agreement.py`, the centerpiece: for one source-receiver
geometry, one band and one prior, the operator `m ↦ (δT_ray, δT_FF)`,
two rows on the one space, and the prior's push-forward through it, a
Gaussian on `R²`. The figure is that Gaussian's density with the
diagonal `δT_FF = δT_ray` drawn; the numbers are the two standard
deviations, their correlation, the regression slope of the
finite-frequency delay on the ray delay, and the conditional standard
deviation of the finite-frequency delay given the ray delay, the part of
the physics ray theory cannot predict, relative to the unconditional one.
Across bands and prior lengths, isotropic and anisotropic, these numbers
against the prior's lengths with the Fresnel width marked are the
quantitative story; the relative standard deviation of `(ℓ_FF − ℓ_ray)(m)`
is a reading of the same covariance and the deterministic counterpart,
the relative dual norm of the difference in the space's own dual norm,
goes beside it.

Part D, `pushforward.py`: over a fan of distances at each band, the
push-forwards of a smooth and of a rough prior as Gaussians on `R^N`,
their KL divergence, and a scatter of ray against finite-frequency delays
for prior draws.

Figures per method and per band, titled; no head-to-head summary figure
beyond the curves of part C.

## Stage 4: finishing

**Done 2026-11-01**, with part D (`pushforward.py`: the fan's two laws per
band and prior, their KL divergence both ways, the scatter of draws; the
divergence grows ten to a hundred times a band and falls with the prior's
distance) and the cut-away through the library's planetmodel adapter
(`plotting.plot_cutaway`, D-215) in part B. `run_all.py`, README,
IMPLEMENTATION.md, the row and the `raytracer/` paragraph in
`miniapps/README.md`, 42 slow tests in about two minutes. Full scale 18
minutes end to end (A 8 s, priors 36 s, B 3.7 min, C 7.3 min, D 5.9 min),
9.7 GB at peak; quick 11 s. `STAGE2_NOTES.md` deleted. The app is recorded
as D-216.

README (the problem, the figures and what they mean, the planet and the
periods stated, the boundary clipping in one sentence),
`IMPLEMENTATION.md`, `run_all.py`, the quick scale in about a minute and
the full scale in about ten, `tests/test_miniapp_banana_doughnut.py` in
the slow set with `pytest.importorskip("planetmodel")` in the pattern of
the tomography test, and the app's row in `miniapps/README.md`, whose
paragraph on `raytracer/` should then name both apps.

**Stretch, last**: a cut-away of the mantle, a wedge removed, the Sobolev
gradient on the cut faces and an inner shell with the ray through it, in
matplotlib's 3-D axes or plotly (the `plot3d` extra) written to an HTML
file beside a static image. Never open a window or a browser.

## Appendix: the paper's equations, transcribed

Dahlen, Hung and Nolet (2000), Geophys. J. Int. 141, 157–174. Everything
below is what the work needs from it; the numbering is the paper's. The
paper's sections 6 (paraxial approximation), 7 (reduction to ray theory)
and Appendix A (spherical earth kernels) are the source. Dimensions were
checked, and (A50), (A51) and (A53) against the constant-velocity chord;
the limit (A15) is stated as (102) implies it.

**Coordinates and conventions (Appendix A1, equations (A1) to (A5)).**
The ray plane is the equator; the source is at `(r_s, θ = π/2, φ = 0)`,
the receiver at `(a, π/2, Δ)`, `a` the planet's radius, `φ` the longitude
swept from the source. The angle of incidence `i` is measured from the
upward vertical `r̂`, `0 ≤ i ≤ π`: `π/2 ≤ i ≤ π` on a downgoing leg,
`0 ≤ i ≤ π/2` on an upgoing one, `i = π/2` at the turning point. The unit
wave vector is `k̂ = r̂ cos i + φ̂ sin i` (A2). The ray-centred directions
are `q̂1 = q̂_SV = r̂ sin i − φ̂ cos i`, in the ray plane and normal to the
ray, and `q̂2 = q̂_SH = θ̂`, normal to the plane (A3); `(q̂1, q̂2, k̂)` is
right-handed. The ray parameter `p = r sin i / c` (A4) is constant along
the ray, in time per radian. The kinematic ray equations in arc length `l`
are `dr/dl = cos i`, `dφ/dl = r⁻¹ sin i`, `di/dl = p r⁻¹ (ċ − c/r)` (A5),
`ċ = dc/dr`.

**Ray-centred coordinates (section 6.2, (87), (88)).** A scatterer is
`x = ξ + q`, `ξ` the nearest point of the central ray, `k̂ · q = 0`,
`q = q1 q̂1 + q2 q̂2`, so `x = (q1, q2, l)` with `l` the arc length of `ξ`
from the source, `0 ≤ l ≤ L`. The volume element in these coordinates is
`(1 + q_i ∂_i c / c) dq1 dq2 dl` (section 7, after (110)), taken as
`dq1 dq2 dl` in the paraxial approximation.

**The traveltime difference (section 6.3, (91)).** With `M' = M_ξs` and
`M'' = M_ξr` the `2×2` Hessians `∂²T/∂q_i ∂q_j` of the traveltime from the
source and from the receiver at `ξ`,

```
T_xs + T_xr − T_rs = ½ qᵀ (M' + M'') q .                                (91)
```

**Properties of the Hessians (section 6.4).** The forward Hessian obeys
the Riccati equation `dM'/dl + c M'² + c⁻¹ V = 0` (92) and the backward one
`dM''/dl − c M''² − c⁻¹ V = 0` (93), `V` the matrix `∂²c/∂q_i ∂q_j` of the
speed's transverse second derivatives at `ξ`. The spreading factors obey
`dR'/dl = ½ c (tr M') R'`, `dR''/dl = −½ c (tr M'') R''` (96), (97). The
limits at the ends are

```
R_ξs → l ,        det M_ξs → c_s⁻² l⁻²           as l → 0 ,           (102)
R_ξr → L − l ,    det M_ξr → c_r⁻² (L − l)⁻²     as l → L ,           (103)
```

and the ratio of spreading factors is `R_rs / (c_r R_ξs R_ξr) =
√|det(M' + M'')|` (104), valid across any boundary. The Maslov indices
satisfy `M_ξs + M_ξr − M_rs = ½ [sig(M' + M'') − 2]` (105), `sig` the
number of positive minus the number of negative eigenvalues: zero
whenever `M' + M''` is positive definite.

**The paraxial kernel (section 6.5, (107) to (109)).** For a traveltime
measured by cross-correlation of a pulse whose source-time power spectrum
is `|ṁ(ω)|²`, the perturbation from a relative speed perturbation `δc/c`
of the propagating wave type alone is `δT = ∫ K (δc/c) d³x` (107) with

```
K = −(1 / 2πc) √|det(M' + M'')| · ∫₀^∞ ω³ |ṁ(ω)|² sin Φ dω / ∫₀^∞ ω² |ṁ(ω)|² dω ,   (108)
Φ = ½ ω qᵀ (M' + M'') q − [sig(M' + M'') − 2] π/4 .                              (109)
```

The leading `c` may be taken at the scatterer `x` or at `ξ`; the paper
takes the scatterer's and says the choice is immaterial. The kernel of a
differential traveltime `δ(T_B − T_A)` is `K_B − K_A` (110). The paraxial
kernel is reliable within the first few Fresnel zones, where
`0 ≤ ω̄ (T' + T'' − T) ≤ π` with `ω̄` the dominant frequency (83); beyond
them destructive interference across the band makes it negligible.

**Reduction to ray theory (section 7, (111) to (115)).** Integrating
(107) over `q1, q2` at fixed `l` with the Gaussian identity

```
∫∫ sin Φ dq1 dq2 = 2π / (ω √|det(M' + M'')|)                             (113)
```

turns the bracketed frequency ratio into one and leaves

```
δT = −∫₀ᴸ c(0, 0, l)⁻² δc(0, 0, l) dl ,                                  (114)
```

which is ray theory's `δT_ray = −∫ c⁻² δc dl` (115), the first-order shift
along the unperturbed ray by Fermat's principle.

**Dynamic ray tracing in a spherically symmetric planet (Appendix A2 to
A6).** Both Hessians are diagonal in `(q̂1, q̂2)`, `M' = diag(M1', M2')`,
`M'' = diag(M1'', M2'')` (A10), and `T' + T'' − T = ½ (M1' + M1'') q1² +
½ (M2' + M2'') q2²` (A11). The forward Hessian obeys

```
dM/dl + c M² + c⁻² V = 0 ,                                              (A13)
V = diag( c̈ sin² i + r⁻¹ ċ cos² i ,  r⁻¹ ċ ) ,                           (A14)
M → (c_s l)⁻¹ I  as  l → 0 ,                                            (A15)
```

`c̈ = d²c/dr²`; (A15) is the point-source start, as (102) says. (The
paper prints the coefficient of `V` as `c⁻²` in (A13) and as `c⁻¹` in
(92); dimensions, and (A30) below, which is (A13) with (A14) rewritten in
longitude, settle it as `c⁻²`.) The divergent start is handled by
`M = P Q⁻¹` (A26), with

```
dP/dl = −c⁻² V Q ,     dQ/dl = c P ,                                    (A27)
P = diag(P1, P2) ,     Q = diag(Q1, Q2) ,     M_i = P_i / Q_i ,       (A28), (A29)
P1(0) = P2(0) = 1 ,    Q1(0) = Q2(0) = 0     at the source.            (A35)
```

In longitude, by `dl/dφ = r² / (p c)`, the same functions obey

```
dP1/dφ = −p c⁻¹ (c̈ + r⁻¹ ċ cot² i) Q1 ,    dQ1/dφ = p⁻¹ r² P1 ,          (A30)
dP2/dφ = −p⁻¹ r c⁻³ ċ Q2 ,                  dQ2/dφ = p⁻¹ r² P2 .          (A31)
```

Jump conditions (A16) to (A24) and (A32) to (A34) apply only at
discontinuities of the planet, which this app has none of; (A54) says the
in-plane sum jumps so that `cos² i (M1' + M1'')` is continuous, and (A52)
that the out-of-plane sum is continuous, should a layered planet ever be
wanted. The backward Hessian `M''` is the same integration from the
receiver along the reversed ray, `r_s → r_r`, `φ → Δ − φ`, `i → π − i`
(Appendix A4 obtains it instead from the forward propagator's symplectic
inverse; either is fine).

**Closed forms (Appendix A6).** Out of the ray plane,

```
P2 = p⁻¹ c⁻¹ r_s sin(i + φ) ,    Q2 = p⁻¹ r r_s sin φ ,                  (A49)
M2' = sin(i + φ) / (c r sin φ) ,                                         (A50)
M2' + M2'' = p sin Δ / (r² sin φ sin(Δ − φ)) ,                           (A51)
```

(A51) continuous along the whole ray and symmetric between source and
receiver only when `r_s = r_r`; `M2` diverges at `φ = π`, the antipodal
caustic, which is why distances stay well short of it. In the ray plane
there is no closed form, but the combination

```
r cos i P1 + (p² r⁻¹ ċ − c⁻¹) Q1 = r_s cos i_s                           (A53)
```

is constant along the ray (`i_s` the take-off angle at the source), and is
a check on the integration together with (A49) to (A51). The chord of a
constant velocity checks the reading of all of them: with `L` the chord's
length, `M1' = M2' = 1 / (c l)`, `M1'' = M2'' = 1 / (c (L − l))`, so
`M1' + M1'' = M2' + M2'' = L / (c l (L − l))`, and at the midpoint of a
surface-to-surface chord of angular length `Δ` all of these equal
`2 / (c sin(Δ/2))`, as (A51) gives there.

**Spreading factors (Appendix A7, (A55) to (A58)).** Not needed for the
kernel, but a by-product: `R = c_s⁻¹ √|Q1(Δ) Q2(Δ)|`, `R' = c_s⁻¹
√|Q1(φ) Q2(φ)|`, and `R / (c_r R' R'') = √|det(M' + M'')|` (A58).
