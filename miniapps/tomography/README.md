# Ray tomography in a disk, then a ball, then with bent rays

How the app is built on the library is walked through in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## The problem

Travel times of rays through a radially layered planet, and what they say
about its slowness. The forward code is the user's own: a ray tracer
(`src/rays.py`, loading the mini-apps' shared `raytracer/`) that knows
nothing of the library, wrapped as a path-integral operator on the
library's layered disk through point evaluation at the ray nodes, with the
adjoint the library's. The geometry is Earth-like and the data synthetic:
the slowness perturbation is a draw from the prior, so that every answer
below can be marked against the truth.

The planet has radius one: a mantle on `[0.55, 1]` whose velocity rises
linearly with depth, `v(r) = 1 + 1.778 (1 - r)`, from 1 at the surface to
1.8 at its base, over a core of uniform velocity 1.2. The core is slow, so a
ray that turns in the mantle never enters it, and the ray that grazes it
emerges 82.2 degrees from its source: beyond is the shadow. Only mantle rays
are used: every source-receiver pair between 20 and 80 degrees apart, each
path once where a source sits at a station. The network is lopsided on
purpose: a continent of stations over the upper half of the planet, three
island stations below, and a dozen sources round it, so that the mantle under
the continent is crossed in every direction and the mantle opposite it by a
handful of rays at most.

The tracer works for any velocity that depends on radius alone. A ray keeps
its parameter `p = r sin(i) / v(r)` and turns where `r / v(r) = p`; its
angular distance, travel time and length are integrals from the turning
radius to the surface whose inverse-square-root singularity the substitution
`r = r_t + (1 - r_t) s^2` removes, after which each is an analytic function
of `s`, held as a Chebyshev series and integrated to roundoff. The two-point
ray is a root of `Delta(p) = Delta` on the branch of mantle rays, where
`Delta` falls monotonically from 82.2 degrees to zero. A traced ray is a few
hundred nodes at equal increments of arc length and the trapezoid weights
along the polyline through them.

The model is the slowness perturbation, in percent of the surface slowness,
on the layered disk `pygeoinf.disk.Layered([0, 0.55, 1])`: the core and the
mantle as two layers, each the Sobolev space `H^2` with a length scale of
0.1, on which a value at a ray's node exists. The prior is a Gaussian
measure of extra regularity on that space, a Sobolev measure of order 2 on
each layer, whose covariance is trace class on `H^2` and whose draws have
nearly one derivative more than the space asks; its amplitude is set so
that its pointwise standard deviation, uniform to within two per cent,
averages one percent. The layers are independent under it, which is what a
discontinuity at the core's edge means. A datum is the travel-time
perturbation `int ds dl` along a reference ray, in hundredths of a time unit,
with Gaussian noise of standard deviation 0.05. Three questions are asked:
the mean perturbation over a box of the mantle under the continent (well
covered), over a box of the same size opposite it (poorly covered), and over
the core, which no ray enters.

The same questions are answered every way the library offers:

- **A, the setup** (`setup.py`): the structure, the rays and the problem,
  and a figure of the rays over the reference velocity and of the truth with
  the two boxes.
- **B, the point answers** (`point.py`): least squares at the damping a
  Bayesian would guess and the minimum-norm model under the discrepancy
  principle, both measuring size in the prior's norm; its resolution of
  a unit bump in each box, and its error budget on the three means. Its
  figure: the two models, the two bumps as the estimate sees them, and each
  mean's estimate with two propagated standard deviations, the value it
  would take on exact data (the truth plus the bias) and the truth.
- **C, the Bayesian answer** (`bayesian.py`): the posterior mean, the
  pointwise standard deviation (exactly at four points, by sampling over the
  whole disk), a sample, and the posterior on the three means with their
  95% intervals. Its figure sets the prior against the posterior: the
  pointwise standard deviation before and after on one color scale, the
  posterior mean beside the truth, and each mean's marginal before and
  after.
- **D, the set answer** (`sets.py`): the feasible set of the three means
  under the prior's 95% ball and the noise's, as certified brackets in the
  six coordinate directions, and the truth against the polytope they
  bound. Its figure sets each bracket against the bound the prior's ball
  admits alone, before any datum.
- **E, a plume among candidates** (`mixture.py`): a discrete question
  inside the continuous problem. A plume rises at one of five angles, a
  conduit from the core-mantle boundary to the surface, each angle a
  component of a mixture prior; the posterior mixture's weights say which,
  and the three means are answered as mixtures. Its figures: the candidate
  conduits under the rays and the truth; the weights against
  the prior's, against the noise level, and as each candidate's own travel
  times would set them; the three means before and after, each drawn as the
  mixture's density over its weighted components; and the winning
  component's mean with the truth.
- **F, the ball** (`ball.py`): the same planet as a layered ball of degree
  14, its stations spread over the sphere: a dozen sources as even as a
  dozen points can be, thirty receivers over a continent and eight islands
  far from it, and every pair between 20 and 80 degrees apart a ray, 174 in
  all. A ray lies in the plane of its ends and the center, so the disk's
  tracer traces it unchanged and its nodes go along the great circle at the
  traced radii. The truth is five bumps at different depths, latitudes and
  longitudes, one of them in the core; the properties are the means over a
  box under the continent, over a box at its antipode and over the core.
  The three answers again, each with its figure: the minimum-norm model and
  its error budget, the posterior against the prior, and the brackets
  against the prior's ball alone. The figures cut the ball by the
  equatorial plane
  and by the meridians of 30 and 120 degrees, the first through the
  continent and its antipode, the second across them, each section with the
  parts of the rays within six degrees of its plane and the stations on it,
  and by the shell at radius 0.825 through the middle of the boxes, with the
  rays where they run beneath it; a fourth figure is the network on a map
  and the truth. A fifth sets the truth's mantle beside the posterior mean
  on one color scale, the ball cut in half by the plane of the meridian of
  30 degrees and the core a plain sphere standing in the cut, so that one
  face holds both boxes: the bump under the continent comes back nearly
  whole and the one at its antipode smeared and weakened.
- **G, bent rays** (`bent.py`): the forward problem as it is rather than as
  linearized, on a few dozen rays in the disk under a prior five times
  wider: the travel time through the perturbed slowness, along the ray
  re-traced in it by shooting. Its derivative against finite differences,
  how far the rays and the times move, the MAP model by Newton-CG, and the
  MAP model against the linear posterior mean computed from the same data on
  the reference rays.

The data are a few hundred and the model coefficients a few thousand, so the
data-space normal operator `A Q A* + R`, and every damped `A Q A* + t R` and
`A A* r^2 + t R` the point and set answers sweep, is a few hundred square:
every solve is a Cholesky factorization of it, and there is no
preconditioner story here.

Two scales. `--quick` is angular order 16, a length scale of 0.2 and 25 rays,
and runs parts A to E in about five seconds; it is what the test suite runs.
The full scale is angular order 96 in the mantle and 53 in the core (4688
coefficients), 275 rays of 301 nodes each and the length scale above, and
runs parts A to E in about 24 s. The ball, part F, is the same toy at both
and takes about 7 s. Part G has its own two sizes, 19 rays at angular order
16 and 32 rays at order 32, and takes about 20 s and 40 s. A full run of
every part takes about 70 s, a quick one about 35 s. All are seeded.

## How to run

```
cd miniapps/tomography
MPLBACKEND=Agg python run_all.py            # every part, full scale
MPLBACKEND=Agg python run_all.py --quick    # every part, small scale
MPLBACKEND=Agg python point.py              # one part; likewise setup.py,
                                            # bayesian.py, sets.py,
                                            # mixture.py, ball.py, bent.py
```

A script runs from any directory, by its path. Every script takes the same
options: `--quick` or `--full` (the default) picks a preset scale, every
field of the scale is then an option that overrides it, and `--outdir` says
where the figures go. The ball, `ball.py`, has one preset at both, and its
stations, points of the sphere, are not options; the bent rays, `bent.py`,
have their own two presets, with the prior five times wider and the rays
kept within 70 degrees. `run_all.py` takes the disk's options and applies
those given to the ball's and the bent rays' presets as well, the ball's
stations excepted. `python run_all.py --help`, in short:

```
  --quick | --full      the quick scale, or the full one (the default)
  --outdir OUTDIR       where the figures go (default: results/,
                        results/quick/ at the quick scale)
the scale, each default the full preset's:
  --seed int            the seed of the truth and the noise (2026)
  --mmax int            the mantle's angular truncation (96)
  --order float         the Sobolev order of each layer, above one (2.0)
  --length-scale float  the length scale of the space and the prior (0.1)
  --prior-order float   the prior's Sobolev order on each layer, above one (2.0)
  --prior-std float     the prior's pointwise std, in % of the slowness (1.0)
  --noise-std float     the noise on each travel time, in hundredths (0.05)
  --level float         the discrepancy level's and the sets' credibility (0.95)
  --sources float [float ...]
                        the sources' angles, in degrees (-60.0 ... 300.0)
  --receivers float [float ...]
                        the receivers' angles, in degrees (0.0 2.5 ... 320.0)
  --min-distance float, --max-distance float
                        the shortest and longest rays kept, in degrees (20, 80)
  --nodes int           the nodes along each ray (301)
  --plume-width, --plume-peak, --background, --redraws
                        part E's plume and its noise sweep
  --std-samples, --bump-width, --n-jobs
                        the sampler's and the figures' knobs
```

Two overrides at the quick scale, against its own numbers:

- **A noisier survey**, `run_all.py --quick --noise-std 0.2`: four times
  the noise on the same truth and rays. The posterior standard deviation
  under the stations rises from 0.18 to 0.34 and the 95% interval on the
  well-covered box's mean widens from ±0.21 to ±0.42; the bracket on it
  widens from 3.7 to 5.4. The mixture's weight on the true candidate
  barely moves, from 1.00 to 0.98. In the bent rays of part G the
  bending, 0.056 rms, is now below the noise: the linear posterior mean
  fits the bent data to a chi-squared of 7.6 against the 7.1 it believes,
  where at the app's noise it was 8.3 against 1.5.
- **A sparser network**, `run_all.py --quick --receivers 0 50 100 150 270`:
  five receivers rather than twelve, 12 rays rather than 25 and 9 in part
  G. The posterior standard deviation in the well-covered box rises from
  0.15 to 0.23, the 95% interval on its mean widens from ±0.21 to ±0.37,
  its bracket from 3.7 to 5.8 wide, and the mixture's weight on the true
  candidate falls from 1.00 to 0.94.

A part run on its own rebuilds the problem from the seed, so it gives the
same numbers as in the full sequence. Each part writes its own figures,
titled with the app and the part and showing one kind of answer apiece,
into `results/` (`results/quick/` at the quick scale), which git ignores.
The test is `tests/test_miniapp_tomography.py`, marked `slow`: the
command line built from the scale, the tracer against the closed form, the
operator against the composition it stands for and the library's adjoint
check, in the disk and in the ball, the ball's rays on the great circles of
their ends and drawn in the library's section frame, the shooting tracer
against the radial one, part E's conduit flat along its axis and Gaussian
across it, and every part through its script's entry point at
the quick scale built through the same command line, the mixture's weights
and part G's derivative check among them.

| file | what it is |
|---|---|
| `src/rays.py` | the ray tracer, loaded from the mini-apps' shared `raytracer/`: in a radial medium, ray parameter, turning radius, `Delta(p)`, `T(p)`, the two-point ray, the nodes and weights; the circular rays of a flattened linear gradient, its closed form; and the shooting tracer for a medium that is not radial, `shoot` and the two-point `bend` |
| `src/problem.py` | the structure, the disk's network, the scale with its two presets, the forward operator, the means over boxes in a disk or a ball, the Sobolev prior and its exact pointwise standard deviation, `build(scale=)` for the disk |
| `src/common.py` | the command line built from the scale, the disk's panels, the panels of an error budget, of marginals before and after, Gaussian or mixtures', and of brackets against the prior's ball alone |
| `setup.py`, `point.py`, `bayesian.py`, `sets.py`, `mixture.py`, `ball.py`, `bent.py` | parts A to G, each a script with a `main()` and a function the others and the test call; `mixture.py` also holds the conduit, the candidates, the mixture prior and the noise sweep; `ball.py` also holds the network on the sphere, the rays placed on their great circles, the bump truth, the sections and shells and the cut-away; `bent.py` the perturbed slowness, the bent forward operator and the Gauss-Newton MAP |
| `run_all.py` | the parts in order |
| `results/` | the figures a run writes, `results/quick/` at the quick scale |

## The ray tracer

The flattening `z = -ln r`, `x = angle` maps the disk conformally onto a
half-plane and takes the slowness `1 / v(r)` to `r / v(r)`, so the disk whose
velocity is `v(r) = r (1 - b ln r)` is the flat medium `v(z) = 1 + b z`, in
which every ray is an arc of a circle and the two-point problem has a closed
form. With `b` chosen for the app's contrast, `v(0.55) = 1.8`, the tracer's
general radial machinery against that closed form, from 5 to 150 degrees:
the ray parameter to 1.5e-11, the travel time to 9.4e-11, the turning radius
to 7.0e-12, and every node on the ray's circle to 1.1e-11. The path weights
integrate the slowness along a 90-degree ray to 9.4e-5 with 101 nodes, 1.0e-5
with 301 and 1.2e-6 with 901: second order. Applied to the reference
slowness, the forward operator returns the reference travel times to 3e-5.

## The answers at the full scale

The forward operator's rows are assembled ray by ray in 2.2 s, from 82 775
node evaluations; the library's adjoint check, with probes drawn from the
prior, holds the adjoint identity to 1.8e-15. The truth's travel-time perturbations
have a standard deviation of 0.56 against noise of 0.05. Machine: AMD Ryzen
AI 7 PRO 350 (16 cpus), Python 3.12, numpy 2.4.

| mean over | truth | B: minimum norm, 2 propagated sd | C: posterior mean, 95% interval | D: certified bracket |
|---|---|---|---|---|
| well-covered box | +1.019 | +0.979 ± 0.052 | +0.984, [+0.904, +1.063] | [-0.23, +2.21] |
| poorly covered box | -0.983 | -0.967 ± 0.102 | -0.958, [-1.498, -0.418] | [-5.67, +3.82] |
| core | -0.334 | 0.000 ± 0.000 | 0.000, [-0.971, +0.971] | [-3.58, +3.58] |

- **B** (2 s): least squares at a damping of one in the prior's norm is the
  posterior mean of C, and fits to a chi-squared of 190 against a critical
  315; the discrepancy principle chooses a damping of 9.1 in 6 misfit
  evaluations, a factorization each, and its model is smaller in the
  prior's norm than the truth, 6.7 against 69. A unit bump of width 0.06 is
  seen at 0.43 of its height in the well-covered box and at 0.12 in the
  poorly covered one, smeared along the few rays that pass. In the core the
  estimate is zero, the noise it propagates is zero, and its whole error is
  bias, which nothing in the point answer reveals.
- **C** (5 s): one factorization of the 275-square normal operator
  gives the posterior mean in a fiftieth of a second; it fits the data
  to a chi-squared of 190 on 275.
  The pointwise standard deviation, from the prior's 1.0, is 0.11 in the
  mantle under the stations, 0.12 in the middle of the well-covered box, 0.45
  in the middle of the poorly covered one and 1.001 in the core, exactly; 256
  posterior samples, 3.1 s, give the map to 4%. The core's mean has the
  prior's own posterior, center zero and standard deviation 0.496.
- **D** (7 s): the prior's 95% ball is `||m|| <= r` in the space's own
  `H^2` norm with `r = 4.59`, which the truth lies in at 0.98 of the
  boundary's gauge `(||m|| / r)^2`; the six supports take 7 s. A set bound
  admits far more than the Gaussian posterior: the brackets are many times
  the posterior intervals, widest where the data say least. The ball alone
  admits ±8.74 for a box's mean and ±4.03 for the core's, the radius times
  the norm of each mean's representer; the data narrow the well-covered
  box's to a seventh of that, the poorly covered one's to a half, and the
  core's hardly at all, to ±3.58, which is what the ball has left once the
  mantle fits the data: the smallest fitting model plus as much of the
  core mean's representer as the ball holds, the model the support
  search's inner end attains. The truth lies in every bracket and in the
  polytope they bound.
- **E** (7 s): the plume is found at its candidate, with a posterior weight
  of 0.990 against the prior's 0.2; see
  [a plume among candidates](#a-plume-among-candidates) below.
- **F** (7 s): in the ball, 174 rays of 121 nodes on a layered ball of 984
  coefficients, 240 in the core at degree 8 and 744 in the mantle at degree
  14; the truth's travel-time perturbations have a standard deviation of
  0.29 against noise of 0.05. The prior's 95% ball has the radius 9.20,
  which the truth lies in at 0.55 of the boundary's gauge.

  | mean over | truth | minimum norm, 2 propagated sd | posterior mean, 95% interval | certified bracket | the ball alone |
  |---|---|---|---|---|---|
  | the box under the continent | +1.145 | +1.026 ± 0.042 | +1.118, [+1.042, +1.194] | [+0.08, +2.29] | ±12.1 |
  | the box at its antipode | -1.019 | -0.627 ± 0.096 | -0.720, [-1.032, -0.408] | [-6.42, +5.21] | ±12.1 |
  | the core | +0.198 | 0.000 ± 0.000 | 0.000, [-1.270, +1.270] | [-6.40, +6.40] | ±6.47 |

  The point estimate's bias at the antipode, +0.36, is seven times the
  noise it propagates there, and nothing in its budget shows it. The
  pointwise standard deviation is 0.07 in the middle of the box under the
  continent and 0.11 under the stations, 0.25 at the antipode's box and the
  prior's 1.00 in the core. The sections show where that comes from: the
  standard deviation falls along the rays that run near each plane and
  nowhere else, so that the meridian of 30 degrees, through the continent
  and its antipode, is dark on one side and pale on the other, and the
  meridian across them pale throughout but for the few rays that pass. The
  truth lies in every bracket and in the polytope they bound.

## A plume among candidates

Part E puts a discrete question inside the continuous problem: a plume rises
through the mantle at one of a handful of angles, and which one? The plume
is an idealized conduit, a slowness perturbation `a exp(-d^2 / (2 w^2))`
with `d` the distance from the point to a radial segment that runs from the
core-mantle boundary to the surface: Gaussian across the segment, of width
`w = 0.06` (0.12 at the quick scale), flat along it, and nothing in the
core. Its peak `a` is 3%, three times the prior's pointwise standard
deviation. The candidates, of which the quick scale keeps the first, third
and fifth:

| candidate | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| angle | 75° | 120° | 330° | 200° | 270° |
| where | under the continent, through the well-covered box | under the continent | at its edge | opposite it | opposite it, through the poorly covered box |
| rays within the conduit's width | 105 | 74 | 31 | 10 | 6 |

Each candidate is a component of the prior, a `GaussianMixture`: the app's
prior translated to the conduit there, so that every component has the
app's covariance, and the weights equal. The truth is the conduit at
candidate 3, under moderate coverage, over half a draw from the shared
covariance, and the data come from it with the app's noise. Under the
linear forward problem the posterior is again a mixture
(`LinearGaussianMixtureInversion`): each component updated by its own
linear Gaussian inversion, with the app's Cholesky solver in the data
space, and the weights rescored by the components' evidences.

**What the weights are.** A posterior weight is the prior weight times the
component's evidence `p(d | k)`, the density of the data under that
component, normalized: the posterior odds of two candidates are their prior
odds times the ratio of their evidences. The covariance being shared, the
evidences' volume halves, `log det (A C A* + R)`, are equal, and the ratio
is `exp(-(M_k - M_j) / 2)`, with `M_k` the misfit of the data to candidate
`k`'s travel times in the metric of `A C A* + R`, the covariance of the
data under any one component. A single Gaussian posterior could not say
this. Its mean is one field and its covariance does not depend on the data;
given the mixture's own mean and covariance as a prior, it would treat the
plume's amplitude at each candidate as a continuous unknown, free to be
fractional or negative at all five at once, and return a field, not a
place. The weights are the probabilities of a discrete unknown, the plume's
place, which only a prior that has one can return, and their odds move with
the data, as the sweep below shows, which a covariance fixed in advance
cannot.

**The answer** (the whole part in 7 s; a factorization of the 275-square
normal operator for each component). A conduit is crossed by every ray
that passes over its angle, and its travel-time perturbations have a
standard deviation of 0.184, nearly four times the noise of 0.05; the
truth's have 0.318. But a candidate must stand out from the prior's own
variability too, which the shared covariance puts at 0.64 a datum on the
same rays.

| candidate | 1 | 2 | 3, the truth | 4 | 5 |
|---|---|---|---|---|---|
| misfit `M_k`, on 275 data | 273.7 | 255.0 | 210.2 | 220.2 | 221.6 |
| log evidence less the best | -31.75 | -22.41 | 0 | -5.00 | -5.71 |
| posterior weight, from 0.2 | 1.6e-14 | 1.8e-10 | 0.990 | 6.7e-3 | 3.3e-3 |

A plume under the continent is ruled out, the conduit at 120 degrees at
odds of five billion to one and the one at 75 at far more; where few rays
pass, the data can say only that the plume is probably not there, and
candidates 4 and 5 keep a hundredth between them. The winning component's
mean has the conduit in its place at its full height, and around it the
part of the background the rays resolve: it fits the data to a chi-squared
of 188 on 275, and its error over the mantle, 0.34 of the truth's norm, is
the background they leave. On the three means the mixtures are drawn as
their densities. The prior on a box's mean is four components at zero and,
for the box's own candidate, one at +1.05, 1.2 of the mean's prior standard
deviation away: a mean over a box 30 degrees wide averages a conduit 0.06
wide that crosses it down to that, and the hypothesis shows as a shoulder,
not a second mode. After the data the well-covered box's mean is
-0.612 ± 0.041 against the truth's -0.618 and the poorly covered box's
-0.22 ± 0.28 against -0.35, the plume component in it at three
thousandths; the core's is its prior, exactly.

**The noise sweep** (4 s): the data redrawn 32 times at each of seven noise
levels, from a tenth of the app's to a hundred times it, the components
factorized again at each. The average weights, and the effective number of
candidates, the exponential of the weights' entropy:

| noise, times the app's | 1 | 2 | 3 | 4 | 5 | effective number |
|---|---|---|---|---|---|---|
| 0.1 | 0.000 | 0.000 | 0.989 | 0.004 | 0.007 | 1.07 |
| 0.3 | 0.000 | 0.000 | 0.978 | 0.009 | 0.013 | 1.13 |
| 1 | 0.000 | 0.000 | 0.962 | 0.014 | 0.023 | 1.20 |
| 3 | 0.001 | 0.000 | 0.968 | 0.011 | 0.019 | 1.18 |
| 10 | 0.038 | 0.032 | 0.813 | 0.048 | 0.069 | 2.08 |
| 30 | 0.072 | 0.172 | 0.487 | 0.121 | 0.148 | 3.98 |
| 100 | 0.134 | 0.195 | 0.269 | 0.200 | 0.202 | 4.89 |

The weights sharpen from the prior's equal ones at a hundred times the
noise as it falls, and stop at three times the app's, short of one: from
there down to a tenth of the app's noise the true candidate keeps 0.96 to
0.99 on average, and candidates 4 and 5 the rest. The misfits are measured
in the metric of `A C A* + R`, and once `R` is small beside `A C A*` it is
the prior's own variability, not the noise, that a plume must stand out
from: the background beneath the truth, which every redraw keeps, is that
variability, and it lends the two candidates opposite the continent a
hundredth or more however small the noise. Nor does the noise decide which
candidates can be told apart. The weights that each conduit's own travel
times would earn, noise-free and with no background, at the app's noise:

| the plume at | weight on 1 | on 2 | on 3 | on 4 | on 5 |
|---|---|---|---|---|---|
| 1 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.983 | 0.006 | 0.011 |
| 4 | 0.000 | 0.000 | 0.006 | 0.927 | 0.067 |
| 5 | 0.000 | 0.000 | 0.010 | 0.067 | 0.923 |

These are the weights at the mean of the log evidences: for data drawn
about candidate `j`, the difference `M_k - M_j` averages the squared
distance between the two candidates' travel times in the same metric, which
is what the noise-free data give. Under the continent each conduit names
itself. Candidates 4 and 5, with ten rays and six within their width, give
each other a fifteenth of their weight, and at a tenth of the noise still
0.055: the coverage leaves them the hardest to tell apart, and less noise
helps only so far. Even the true conduit, crossed by 31 rays, gives a
sixtieth of its weight to those two: the rays that cross a conduit cross
the prior's variability along it too, and the metric that decides weighs
the one against the other.

The truth's background is half a draw from the shared covariance. With a
whole draw the truth would be an exact draw from its component, and a
background that happens to resemble another candidate's conduit could win.
Over ten seeds at the full scale it does in two, candidate 4 taking 0.92
and 0.91 of the weight and the true candidate 0.03 and 0.05; in the other
eight the true candidate's weight runs from 0.59 to 1.00, and at the quick
scale from 0.97 to 1.00 in all ten. Opposite the continent a conduit is
seen by the handful of rays that cross it there, and along those rays a
whole draw of the prior can look like one. Under the continent, where the
rays cross in every direction, none does: candidates 1 and 2 never take
more than 1e-7 of the weight.

## Bent rays

Part G at its full size: 32 rays between 20 and 70 degrees on a layered disk
of 708 coefficients (angular order 32 in the mantle, length scale 0.15), a
prior of pointwise standard deviation 5% of the surface slowness and the
same noise, 0.05. The slowness is the reference `1 / v(r)` plus the model's
perturbation, its radial functions interpolated by cubic splines on 361
radii and its angular modes exact; a ray is the ray equations integrated by
fourth-order Runge-Kutta at a step of 0.005 in arc length, and the two-point
ray is a secant search on the take-off angles of all the rays together. A
datum is the travel time less the reference ray's, so the forward is zero at
the reference model, where the shooting reproduces the radial tracer's times
to 7e-10 and its nodes to 2e-10. By Fermat's principle the derivative at a
model is the path integral along that model's rays: the operator of parts A
to F built on them, with Simpson's weights along 401 nodes a ray.

No ray may turn below 0.57, two hundredths above the core. The core is the
limit: the medium the rays are shot through has none, being the mantle
continued below its base, and the derivative would read the core's layer
there; the two hundredths are a margin. The deepest reference ray turns at
0.64, the truth's at 0.61, and the deepest of all the rays the part traces,
one in the derivative check's largest step, at 0.59 (0.588 at the quick
size).

**The derivative** against central differences, in one direction drawn from
the prior and scaled to unit norm in `H^2` (the truth measures 22.5), at the
reference model and at two draws. The library's defect is the residual over
the larger of one and the directional derivative's size, here 1.1, so it is
the residual relative to the derivative:

| step | reference | draw 1 | draw 2 |
|---|---|---|---|
| 20 | 1.9e-3 | 1.6e-3 | 1.6e-3 |
| 10 | 4.7e-4 | 4.1e-4 | 4.0e-4 |
| 5 | 1.2e-4 | 1.0e-4 | 1.0e-4 |
| 1 | 5.7e-6 | 4.6e-5 | 3.1e-5 |
| 0.1 | 4.3e-6 | 2.5e-4 | 1.9e-4 |

The truncation falls as the square of the step, close to a fourfold drop at
each halving, down to a floor of 4e-6 at the reference model and 3e-5 to
5e-5 at the draws. At the draws the smallest step rises above that floor
again, to 2e-4: an error of the computed map that does not shrink with the
step, divided by a step that does. The floor is the integrator's: at the
quick size, halving the integrator's step lowers it at every point, by
factors of three to four (from 6.4e-7 to 1.7e-7 at the reference model). The integrated time is a
sum of the slowness at the Runge-Kutta stages, which lie off the path, and
its derivative differs from the path integral along the path by about that
much.

**The bending** in the truth: the travel-time perturbations have a standard
deviation of 2.56; along the bent rays they are shorter than the linear
prediction on every ray, as Fermat's principle requires, by 0.063 rms and by
0.200 at most, about the noise of 0.05 and four times it. The rays move by
a median of 0.008 and at most 0.031, and their deepest points by up to
0.031.

**The answers** (40 s, 31 of them the derivative check): the linear posterior
mean, on the reference rays, fits the data to a chi-squared of 2.3 on 32 as
it believes, and to 23.0 along its own bent rays. The MAP model, by Newton-CG
on the Gauss-Newton Hessian `2 (P + J* R^-1 J)` from the prior's mean with
the prior covariance as the preconditioner, takes 6 iterations, 7 ray
tracings and 5 s; the operator's second derivative is never needed, the
Gauss-Newton Hessian being the derivative's own normal operator. It fits the
data to 2.3 along its own rays, and differs from the linear mean by 3.4% of
it, in the upper mantle under the stations. It comes barely closer to the
truth. The mean over the well-covered box moves from +0.254 to +0.240,
against the truth's -0.208, and over the poorly covered box from -1.743 to
-1.561, against -0.901: both moves are small beside the linear posterior's
standard deviations of those means, 0.51 and 2.41. The error over the
mantle, 0.521 of the truth's norm, and under the stations, 0.311, become
0.515 and 0.300: they are what no ray resolves, and fitting the data with
the right physics reduces them by a few per cent.

## What the app shows

- A forward code the library knows nothing about becomes an operator with a
  correct adjoint through two of its pieces: point evaluation at the nodes,
  and a matrix of weights. The operator here is the product of the two,
  formed ray by ray, and the test checks it against the composition written
  as it reads.
- Coverage is the whole story. Where rays cross in every direction, all three
  answers agree and are tight; where a handful pass in one direction, the
  resolution smears along them, the point estimate's propagated error stays
  as small as it was under the stations, and the posterior widens.
  Where no ray goes, the core, the point estimate is zero with no error bar
  and the posterior is exactly the prior: only the measure and the set say
  how little is known.
- A layered space is the natural model of a discontinuity the data never
  cross: the core's posterior is its prior by construction, not by
  approximation.
- With a few hundred data and thousands of coefficients every solve is
  direct, and the methods that sweep a damped family pay a factorization per
  member and nothing else. The loops that could run in parallel are run
  serially: worker processes cost more than they save at this size, 256
  posterior samples taking 3.0 s serially and 5.5 s on eight processes.
- The prior is a Sobolev measure of order 2 on each layer, and the point
  answers measure size by its norm, so that least squares at a damping of
  one is the posterior mean. The set answer hardens it to the ball of the
  space's own norm that carries 95% of it: the classical norm bound, the
  prior only calibrating the radius. A draw's squared norm has the
  covariance's trace as its expectation, finite because the covariance is
  trace class, so the radius has a limit however fine the truncation; the
  Mahalanobis ellipsoid of the covariance does not, its radius growing with
  the coefficient count. The ball is a convenience, one constraint set
  among many: a bound the physics suggests, on another norm or a region,
  goes to the same estimator.
- A discrete question needs a prior that holds it. A mixture over the
  places a plume might rise returns, in closed form and in seconds, the
  odds of each place, which no single Gaussian can; the sweep of the noise
  shows what limits them: below three times the app's noise the prior's
  own variability, and for two candidates the rays barely separate, the
  coverage.
- The ball asks the same questions of a network spread over a sphere, with
  the same tracer: a radial medium makes every ray planar, so placing the
  traced nodes on the great circle of their ends is all the third dimension
  costs. What changes is what can be seen: coverage is no longer a sector of
  a disk but a patch of the sphere and a cone beneath it, and only sections
  in several planes and a shell together show where the data reach.
- A nonlinear forward needs no more of the library than a linear one: the
  value is the user's code, and the derivative is the linear operator of the
  other parts rebuilt on the current rays. With a perturbation of 5% the
  bending moves the travel times by about the noise on average and four
  times it at most; the linear answer is then inconsistent with its own
  physics, a chi-squared of 23 on 32 data along its own rays where it
  believes 2, and the MAP model repairs that, changing the model by 3.4%
  where the rays bend. Consistency is not accuracy: the MAP model comes
  barely closer to the truth, its error being, like the linear answer's,
  what the rays do not resolve.
