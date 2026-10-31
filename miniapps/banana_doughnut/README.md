# Ray and finite-frequency traveltimes on one model space

How the app is built on the library is walked through in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## The problem

A traveltime is a linear functional of the velocity perturbation `m = δln
c`, and what seismology calls its kernel is that functional's `L2` Riesz
representer, `δT = ∫ K m dV`. In `L2` the two theories of the traveltime
look incomparable: ray theory's kernel is a delta on the ray, and
finite-frequency theory's is a hollow banana with no sensitivity on the
ray itself, a doughnut in cross-section.

Both are functionals on a model space, though, and the comparison belongs
there. On the mantle in `H^2`, where a value at a point exists, both are
bounded and both have smooth representers, their Sobolev gradients. Under
a Gaussian prior with covariance `C`, what a delay depends on is `C ℓ`, the
functional's representer in the prior's Cameron-Martin space and the
covariance of the field with the delay; and the prior pushed forward
through the two functionals is the joint law of the two delays before any
datum. How far ray theory predicts the finite-frequency delay is then a
number for each ray, band and prior, and it is decided by the prior: the
two delays agree to a per cent where the prior's correlation distance is a
few wavelengths or more, though the `L2` kernels look nothing alike, and
part ways where it is below one. No inversion is done; the point is about
choosing the function space, made with the physics of finite-frequency
traveltimes.

The function-space side follows Francis Syvret's Cambridge PhD thesis,
*Theoretical and computational studies in seismic tomography*
(https://www.repository.cam.ac.uk/handle/1810/339758): Sobolev model
spaces, and Sobolev gradients of ray kernels. The finite-frequency kernels
set beside them in the same space, and the comparison through prior
push-forwards over rays, bands and priors, are new here.

**The planet** has radius one: a mantle on `[0.55, 1]` whose velocity
rises linearly with depth, `c(r) = 1 + 1.778 (1 - r)`, from 1 at the
surface to 1.8 at its base, over a fluid core; dimensionless units
throughout. It is one smooth piece, so no ray that turns in the mantle has
a caustic between its ends, and none reaches beyond 82.2 degrees. The
source is at the surface on the equator at longitude zero, the receivers
on the surface east of it, and every ray lies in the equatorial plane:

| distance | 30° | 45° | 60° | 75° |
|---|---|---|---|---|
| ray length | 0.532 | 0.801 | 1.058 | 1.292 |
| travel time | 0.484 | 0.669 | 0.812 | 0.920 |
| deepest radius | 0.913 | 0.822 | 0.716 | 0.604 |

**The bands.** The pulse's power spectrum is a Gaussian in angular
frequency about `ω0 = 2π / T0` with standard deviation `ω0 / 4`. Three
dominant periods, `T0 = 0.006`, `0.024` and `0.096`, are chosen so that the
first Fresnel zone's half-width at mid-path, `½ √(c T0 L)`, is about 0.04,
0.08 and 0.16 of the radius at 45 degrees. They are 0.9, 3.6 and 14 per
cent of that ray's travel time of 0.669; 1.2, 5.0 and 20 per cent of the
30-degree ray's 0.484, and 0.65, 2.6 and 10 per cent of the 75-degree
ray's 0.920. At mid-path of the 45-degree ray the wavelengths `λ = c T0`
are 0.0079, 0.032 and 0.126, and the Fresnel half-widths 0.040, 0.080 and
0.159, which is 5.0, 2.5 and 1.3 wavelengths.

## The two theories

**Ray theory.** The delay is `δT = -∫ m / c dl` along the ray traced in
the reference planet (equation (115) of Dahlen, Hung and Nolet, 2000). The
shared tracer gives 201 nodes at equal arc length and their weights over
the travel time, and the functional is the model's values at the nodes
times the weights.

**Finite-frequency theory**, in the paraxial approximation of Dahlen, Hung
and Nolet (2000), *Fréchet kernels for finite-frequency traveltimes, I.
Theory*, Geophys. J. Int. 141, 157-174. At a point `ξ(l) + q1 e1 + q2 e2`
off the ray, `l` the arc length along it and `q1`, `q2` the offsets in the
ray's plane and across it, the kernel of a cross-correlation traveltime is
`K = -√(A1 A2) G(u) / (2π c)`. Here `u = (A1 q1² + A2 q2²) / 2` is the time
by which the path through the point exceeds the ray, `A1` and `A2` the sums
of the Hessians of the travel time from the two ends, which dynamic ray
tracing along the ray gives, and `G(u)` the band's average of `ω³ sin(ω u)`
over the spectrum. `G(0) = 0`, so the kernel vanishes on the ray. Putting
`q1 = √(2u / A1) cos θ` and `q2 = √(2u / A2) sin θ` turns `√(A1 A2) dq1 dq2`
into `du dθ` exactly, and the functional becomes a quadrature in `(l, u,
θ)` whose integrand is bounded everywhere, the ends of the ray included,
where the kernel is singular: Gauss-Legendre in `u`, 32 nodes out to where
the spectrum's envelope has fallen by `e^-30`, the trapezoid rule in `θ`, 32
nodes, and the ray's own weights in `l`. Composed with point evaluation, it
has the ray functional's form, with about two hundred thousand points
rather than 201. Across the ray `∫ G du = 1`, the Gaussian identity, which
the quadrature meets to `-2.5e-8` in every band, the cut-off tail and not
the node count setting it; so on a model constant across every banana the
two functionals agree, which is the reduction of finite-frequency theory to
ray theory.

The free surface is taken by images, a quadrature point above it folded
back to its mirror image below, which for an SH-type phase adds with the
same sign, and the core, which has no three-dimensional structure, is the
region where the functional is zero, where nothing is computed or drawn.
The fold carries up to a quarter of the weight (in absolute value) at 30
degrees in the widest band; below the core-mantle boundary the widest band
loses up to 8.8 per cent at 75 degrees.

## The model space and the priors

The model is `δln c` on the mantle annulus in `H^2` with a length scale of
0.02, truncated at degree 160, which resolves about 0.02 at the surface:
659 316 coefficients. The space is the problem's, chosen so that both
functionals are bounded; its norm is the one the Sobolev gradients and the
dual norms are measured in, and no prior changes it.

A prior is a Sobolev measure of order 2 on that space with length scales
of its own, along a radius and across one: isotropic, nominally 0.02,
0.04, 0.08, 0.16 and 0.32, and two anisotropic pairs, 0.03 radially with
0.2 and with 0.5 laterally, shorter radially as in the mantle. Each is
calibrated to a pointwise standard deviation of 0.02, two per cent, at
every point. A prior's covariance is an operator in the space's metric, so
the space's own symbol enters its draws and its nominal lengths are not
theirs. Each prior is therefore placed by what its draws do: its
correlation from the 45-degree ray's turning point, computed from the
covariance, falls to `1/e` at

| prior | 0.02 | 0.04 | 0.08 | 0.16 | 0.32 | 0.03 × 0.2 | 0.03 × 0.5 |
|---|---|---|---|---|---|---|---|
| along the shell | 0.058 | 0.084 | 0.130 | 0.214 | 0.368 | 0.307 | 0.821 |
| along the radius | 0.058 | 0.084 | 0.130 | 0.214 | beyond the mantle | 0.069 | 0.070 |
| in wavelengths, `T0 = 0.006` | 7.4 | 10.7 | 16.5 | 27.1 | 46.6 | 8.7 | 8.9 |
| in wavelengths, `T0 = 0.024` | 1.84 | 2.67 | 4.11 | 6.78 | 11.7 | 2.17 | 2.22 |
| in wavelengths, `T0 = 0.096` | 0.46 | 0.67 | 1.03 | 1.70 | 2.91 | 0.54 | 0.55 |

the isotropic priors placed by the distance along the shell and the
anisotropic ones by the shorter, radial one, as part C places them. The
sweep reaches from well above every Fresnel half-width to well inside the
widest band's.

`priors_45deg.png`: each prior's correlation along the shell and down and
up the radius, with each band's Fresnel half-width marked, and one draw
across the ray at its turning point and in the ray's plane. The shortest
prior decorrelates between the narrow and the middle band's Fresnel
half-widths; the longest still correlates at about 0.7 across the widest
band's; the anisotropic priors' draws are layered, short along the radius
and long across it.

## What each part shows

**A, the rays and the kernels** (`setup.py`). `rays.png`: the four rays,
and the sums `A1`, `A2` along each over the straight chord's, which they
meet at the ends and exceed by up to 40 per cent at mid-path. `kernels_{30,45,60,75}deg.png`: the `L2` kernel from
the formula in each band, in the ray's plane and across the ray at
mid-path: the hollow banana, zero on the ray and largest about six tenths
of a Fresnel half-width off it, and across the ray its rings, the
doughnut.

**B, the kernels and their Sobolev equivalents** (`gradients.py`). One
figure per band, prior and section, at 45 degrees: the two `L2` kernels
above, ray theory's a line or a cross, nothing else, and finite-frequency
theory's the banana; below, the two functionals' representers `C ℓ` under
the prior, on one color scale. Across the ray at mid-path, the figure that
carries the point, the doughnut's hole survives in one representer only:
the finite-frequency functional's `H^2` gradient in the widest band, whose
value on the ray is half its extreme across it (0.52). The space's length
scale of 0.02 sits far inside that band's Fresnel half-width of 0.159, and
in the space's own metric the gradient keeps the kernel's shape. Under every isotropic prior the
finite-frequency `C ℓ` is largest on the ray, at 0.87 of its extreme in the
widest band under the shortest prior and 0.98 or more otherwise; under the
anisotropic priors both representers peak a little off the ray (0.74 to
0.92), drawn sideways by the long lateral correlation. The two `C ℓ` differ
visibly only in the widest band under the shortest priors, where the
finite-frequency one is broader and weaker.

The files are `gradients_{across,plane}_45deg_T{period}_{prior}.png`,
`noprior` for the `H^2` gradients: `gradients_across_45deg_T0.024_prior0.08.png`
is the middle band under the middle prior, and
`gradients_across_45deg_T0.096_noprior.png` the widest band's gradients
with the hole.

`gradients_cutaway_75deg_T0.024_prior0.08.png` shows the same two `C ℓ`
for the deepest ray, at 75 degrees, in the whole mantle, with a wedge cut
from it north of the equator from 52.5 degrees west of the source to
mid-path at 37.5 degrees and the core a plain sphere standing in the cut.
The wedge's second face, seen nearly square-on, is the section across the
ray at mid-path, where it turns at 0.60, just above the core: the
doughnut's section, filled under the prior, darkest on the ray and
reaching down to the core. The floor, seen at a grazing angle, is the ray's
own plane from the source to mid-path, with that half of the ray drawn on
it. Under that prior the two representers are one shape.

**C, the agreement of the two delays** (`agreement.py`), the centerpiece.
For one ray, one band and one prior, the prior pushed forward through the
two functionals is a Gaussian on `R^2`, the joint law of the ray delay and
the finite-frequency delay. Its numbers: the two standard deviations, the
correlation `ρ`, the regression slope of the finite-frequency delay on the
ray delay, and the conditional standard deviation of the finite-frequency
delay given the ray delay over its own, `√(1 - ρ²)`, the share of the
physics ray theory cannot predict. At 45 degrees, each cell giving the
prior's correlation distance in the band's wavelengths, then `ρ`, the slope
and `√(1 - ρ²)`:

| prior | `T0 = 0.006` | `T0 = 0.024` | `T0 = 0.096` |
|---|---|---|---|
| 0.02 | 7.4 λ: 0.9982, 0.988, 0.060 | 1.8 λ: 0.9568, 0.889, 0.291 | 0.46 λ: 0.6995, 0.538, 0.715 |
| 0.04 | 11 λ: 1 - 3.6e-4, 0.996, 0.027 | 2.7 λ: 0.9888, 0.960, 0.149 | 0.67 λ: 0.8739, 0.759, 0.486 |
| 0.08 | 16 λ: 1 - 7.7e-5, 0.999, 0.012 | 4.1 λ: 0.9973, 0.988, 0.073 | 1.0 λ: 0.9615, 0.911, 0.275 |
| 0.16 | 27 λ: 1 - 2.0e-5, 1.000, 0.006 | 6.8 λ: 1 - 7.0e-4, 0.996, 0.037 | 1.7 λ: 0.9891, 0.973, 0.147 |
| 0.32 | 47 λ: 1 - 6.3e-6, 1.000, 0.004 | 12 λ: 1 - 2.2e-4, 0.999, 0.021 | 2.9 λ: 0.9964, 0.992, 0.084 |
| 0.03 × 0.2 | 8.7 λ: 1 - 1.9e-4, 0.998, 0.020 | 2.2 λ: 0.9946, 0.983, 0.104 | 0.54 λ: 0.9395, 0.906, 0.343 |
| 0.03 × 0.5 | 8.9 λ: 1 - 1.3e-4, 0.999, 0.016 | 2.2 λ: 0.9963, 0.989, 0.085 | 0.55 λ: 0.9574, 0.946, 0.289 |

**Ray theory breaks down where the prior's correlation distance falls
below about a wavelength.** In the widest band, whose period is 14 per cent
of the travel time and whose Fresnel half-width at mid-path is 0.159, under
the shortest prior, whose draws decorrelate within 0.058 (0.46
wavelengths), the two delays at 45 degrees correlate at only 0.70: the
regression slope of the finite-frequency delay on the ray delay is 0.54,
the finite-frequency delay's standard deviation is 0.77 of the ray
delay's, and 72 per cent of it is what the ray delay cannot predict. The
longer rays fare worse, the correlation being 0.80, 0.70, 0.63 and 0.59 at
30, 45, 60 and 75 degrees. As the prior lengthens the two delays converge:
where the prior's distance is four wavelengths or more, the correlation is
above 0.997 and the slope above 0.988 in every band at 45 degrees, and at
tens of wavelengths, as in the narrow band, `1 - ρ` is of order `1e-5`,
while the `L2` kernels look nothing alike.

`agreement_{30,45,60,75}deg.png` draw the joint law's density at one, two
and three standard deviations, with the diagonal and the line of
regression, in every band under the shortest prior and under every prior
in the widest band. `agreement_curves_*deg.png` set the numbers against
the prior's correlation distance in wavelengths, each band's Fresnel
half-width marked: the three bands' curves of `1 - ρ` and of the two
standard-deviation ratios continue one another over five decades, a
narrower band a few times above a wider one where they meet, so that what
decides is how many wavelengths the prior's correlation spans.
`agreement_fresnel_*deg.png` put the same numbers against the distance in
Fresnel half-widths, on which the slopes of the three bands fall on one
curve.

The anisotropic priors, short along the radius and long across it, agree
better than isotropic priors of the same radial distance: in the widest
band, at 0.54 and 0.55 wavelengths radially, their correlations are 0.94
and 0.96, against 0.70 and 0.87 for the isotropic priors at 0.46 and 0.67.
The doughnut averages across the ray in both directions, and across it
these priors are smooth.

In the space's own norm, which no prior enters, the two functionals are far
apart: the dual norm of their difference is 0.27, 0.68 and 0.94 of the ray
functional's at 45 degrees in the three bands. The space, whose length
scale of 0.02 lies inside every Fresnel zone, sees the whole difference
between the kernels; a delay sees only what the prior's draws hold on the
scale of the wavelength.

The narrow band is resolution-sensitive. The difference between its two
functionals lives on the scale of its Fresnel half-width, 0.040 at 45
degrees, two of degree 160's shortest resolved lengths, and its numbers
rise with the truncation: `1 - ρ` at 45 degrees under the shortest prior
is `7.0e-5` at degree 64, `4.5e-4` at 96 and `1.8e-3` at 160, and under the
longest `6.7e-7` at 64 against `6.3e-6` at 160. The widest band's are
steady over the same range, `0.28` and `0.30` under the shortest prior and
`3.6e-3` under the longest at both, and the middle band's lie between. The
narrow band's numbers are therefore indications, not converged values.

**D, the two theories' laws over the distances** (`pushforward.py`). Each
theory's law of the delays of the four rays together, a Gaussian on `R^4`,
is the prior pushed forward through that theory's four functionals
stacked. The two laws are compared
by their Kullback-Leibler divergence, in nats, both ways, for the smoothest
prior, the roughest and an anisotropic one (`pushforward.csv`;
`KL(FF || ray)` first):

| prior | `T0 = 0.006` | `T0 = 0.024` | `T0 = 0.096` |
|---|---|---|---|
| 0.32 | 1.2e-5, 1.2e-5 | 5.7e-4, 5.8e-4 | 0.025, 0.030 |
| 0.03 × 0.2 | 7.1e-5, 7.2e-5 | 6.2e-3, 6.7e-3 | 0.19, 0.33 |
| 0.02 | 1.3e-3, 1.4e-3 | 0.046, 0.055 | 0.41, 0.69 |

The divergence grows with the band, by a factor of ten to a hundred a
band, and falls as the prior lengthens. It compares the two laws, not the
two delays of one model, and it is smaller than part C's numbers might
suggest: in the widest band under the roughest prior, where the ray delay
accounts for barely half the variance of the finite-frequency delay at 45
degrees and a third of it at 75, it is
0.41 nats over four delays, so that noise-free delays obeying
finite-frequency theory would favor it over ray theory by a likelihood
ratio of about 1.5, on the logarithmic average. Both laws are centered
Gaussians, and there they differ mostly in the finite-frequency delays'
smaller spread, 0.70 to 0.78 of the ray delays' standard deviation, which
is also why `KL(ray || FF)`, scoring the wider law by the narrower, is the
larger. Under the smoothest prior in the narrow band the two laws differ
by `1.2e-5` nats, the same for any purpose.

`pushforward_prior{0.32,0.02,0.03x0.2}.png` scatter the two delays of each
ray for a hundred draws of the model from the prior, each through both
theories' functionals, a panel a band and a color a distance, with the
diagonal: on it under the smoothest prior in every band, and in the widest
band under the roughest a cloud whose slope is about half the diagonal's.
The draws' correlation of the two delays at each distance matches the
push-forward's to within sampling error, at most 2.1 standard errors by
Fisher's transform over the 36 cases.

## How to run

```
cd miniapps/banana_doughnut
MPLBACKEND=Agg python run_all.py            # every part, full scale
MPLBACKEND=Agg python run_all.py --quick    # every part, small scale
MPLBACKEND=Agg python agreement.py          # one part; likewise setup.py,
                                            # priors.py, gradients.py,
                                            # pushforward.py
```

A script runs from any directory, by its path, and takes the same options:
`--quick` or `--full` (the default) picks a preset scale, every field of
the scale is then an option that overrides it (`--lmax`, `--periods`,
`--distances`, `--prior-lengths`, `--prior-pairs 0.03,0.2 0.03,0.5`,
`--prior-std`, `--surface fold|clip`, `--draws`, `--n-jobs`, ...), and
`--outdir` says where the figures go; `--help` lists them with the
preset's values. `run_all.py` builds the problem once and hands it to
every part, so that the functionals' rows one part forms serve the rest.
Run with `OMP_NUM_THREADS` set and one heavy job at a time.

Two scales. `--quick` is one ray at 45 degrees in the middle band on a
space of degree 16 (835 coefficients), with smoke-test priors long enough
for that degree, and runs every part in about 11 s; it is what the test
suite runs. The full scale is the four rays, three bands and seven priors
above on the space of degree 160, and runs in about 18 minutes: part A 8 s,
the priors 36 s, part B 3.7 minutes (the four gradients and their 28
images under the priors 1.5 minutes, their sections 0.6 and the 49 figures
1.6), part C 7.3 minutes (the other rays' rows and 84 push-forwards) and
part D 5.9 minutes (three push-forwards through all sixteen functionals and
300 draws), with four threads (`--n-jobs 4`, `OMP_NUM_THREADS=4`), at a
peak of 9.7 GB. Machine: AMD
Ryzen AI 7 PRO 350 (16 cpus), Python 3.12, numpy 2.4. All draws are
seeded.

Each part writes its figures and tables into `results/` (`results/quick/`
at the quick scale), which git ignores. The test is
`tests/test_miniapp_banana_doughnut.py`, marked `slow`: the quadrature
against the Gaussian identity, the two functionals on constant and radial
models and on smooth draws as the period falls, with the surface folded
and clipped, both through the library's operator check under a prior, the
weight at each boundary, the priors through the measure checks and their
correlations read off the covariance, the Sobolev gradients against the
functionals, the push-forwards against draws, each theory's law as a
marginal of the joint one, the divergence as the band and the prior
change, the cut-away's view of the ray, and every part and `run_all.py`
through their entry points at the quick scale.

| file | what it is |
|---|---|
| `src/rays.py` | the ray tracer, loaded from the mini-apps' shared `raytracer/`: rays in a radial planet, and dynamic ray tracing along them, the Hessians of the travel time across a ray and the ray-centered frame |
| `src/planet.py` | the planet, the rays at each distance with their Hessians, the wavelength and Fresnel half-width at mid-path |
| `src/pulse.py` | the spectrum and its bands, `G(u)` and the quadrature in the detour time |
| `src/functionals.py` | the two functionals: the ray's nodes and weights, the paraxial quadrature with the fold at the surface and the drop at the core, the kernel from the formula, and their composition with point evaluation |
| `src/correlations.py` | a prior's correlation from a point along a shell and a radius, read off its covariance, and its `1/e` distances |
| `src/problem.py` | the scale with its two presets, the space, the priors, the rows of the functionals and the operators stacked from them, `build(scale=)` |
| `src/sections.py` | fields of the space on the two sections, across the ray and in its plane |
| `src/common.py` | the command line built from the scale, where a run writes, the figures' helpers |
| `setup.py`, `priors.py`, `gradients.py`, `agreement.py`, `pushforward.py` | part A, the priors, and parts B, C and D, each a script with a `main()` and a `run()` the others and the test call |
| `run_all.py` | the parts in order on one problem |
| `results/` | the figures and tables a run writes, `results/quick/` at the quick scale |
