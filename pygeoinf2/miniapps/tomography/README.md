# Ray tomography in a disk, then a ball, then with bent rays

## The problem

Travel times of rays through a radially layered planet, and what they say
about its slowness. The forward code is the user's own: a ray tracer
(`rays.py`) that knows nothing of the library, wrapped as a path-integral
operator on the library's layered disk through point evaluation at the ray
nodes, with the adjoint the library's. The geometry is Earth-like and the
data synthetic: the slowness perturbation is a draw from the prior, so that
every answer below can be marked against the truth.

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
on the layered disk `pygeoinf2.disk.Layered([0, 0.55, 1])`: the core and the
mantle as two layers, each the Sobolev space `H^2` with a length scale of
0.1. The prior's covariance is the identity on each layer, scaled to a
pointwise standard deviation of one percent, so that the space's own norm is
the prior's and the point and set answers damp towards the norm the Bayesian
one assumes; the layers are independent under it, which is what a
discontinuity at the core's edge means. A datum is the travel-time
perturbation `int ds dl` along a reference ray, in hundredths of a time unit,
with Gaussian noise of standard deviation 0.05. Three questions are asked:
the mean perturbation over a box of the mantle under the continent (well
covered), over a box of the same size opposite it (poorly covered), and over
the core, which no ray enters.

The same questions are answered every way the library offers:

- **A, the setup** (`setup.py`): the structure, the rays and the problem,
  and a figure of the rays over the reference velocity and of the truth with
  the ray nodes and the two boxes.
- **B, the point answers** (`point.py`): least squares at a guessed damping,
  the minimum-norm model under the discrepancy principle, its resolution of
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
  under the prior's 95% ball and the noise's credible ball, as certified
  brackets in the six coordinate directions, and the truth against the
  polytope they bound. Its figure sets each bracket against the bound the
  prior's ball admits alone, before any datum.
- **E, a plume among candidates** (`mixture.py`): a discrete question
  inside the continuous problem. A plume sits at one of five places in the
  mantle, each a component of a mixture prior; the posterior mixture's
  weights say where, and the three means are answered as mixtures. Its
  figures: the candidates under the rays and the truth; the weights against
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
  against the ball alone. The figures cut the ball by the equatorial plane
  and by the meridians of 30 and 120 degrees, the first through the
  continent and its antipode, the second across them, each section with the
  parts of the rays within six degrees of its plane and the stations on it,
  and by the shell at radius 0.825 through the middle of the boxes, with the
  rays where they run beneath it; a fourth figure is the network on a map
  and the truth.
- **G, bent rays** (`bent.py`): the forward problem as it is rather than as
  linearized, on a few dozen rays in the disk under a prior three times
  wider: the travel time through the perturbed slowness, along the ray
  re-traced in it by shooting. Its derivative against finite differences,
  how far the rays and the times move, the mode by Newton-CG, and the mode
  against the linear posterior mean computed from the same data on the
  reference rays.

The data are a few hundred and the model coefficients a few thousand, so the
data-space normal operator `A Q A* + R`, and every damped `A C A* + t R` the
point and set answers sweep, is a few hundred square: every solve is a
Cholesky factorization of it, and there is no preconditioner story here.

Two scales. `--quick` is angular order 16, a length scale of 0.2 and 25 rays,
and runs parts A to D in about five seconds; it is what the test suite runs.
The full scale is angular order 96 in the mantle and 53 in the core (4688
coefficients), 275 rays of 301 nodes each and the length scale above, and
runs parts A to D in about half a minute. The ball, part E, is the same toy
at both and takes about 15 s. Part F has its own two sizes, 19 rays at
angular order 16 and 32 rays at order 32, and takes 25 s and 45 s. A full
run of every part takes about 80 s. All are seeded.

## How to run

```
cd pygeoinf2/miniapps/tomography
MPLBACKEND=Agg python run_all.py            # every part, full scale
MPLBACKEND=Agg python run_all.py --quick    # every part, small scale
MPLBACKEND=Agg python point.py              # one part; likewise setup.py,
                                            # bayesian.py, sets.py,
                                            # mixture.py, ball.py, bent.py
```

Every script takes `--quick`, `--outdir` and `--seed`. A part run on its own
rebuilds the problem from the seed, so it gives the same numbers as in the
full sequence. Each part writes its own figures, titled with the app and the
part and showing one kind of answer apiece, next to the sources (`quick/` at
the quick scale). The test is `pygeoinf2/tests/test_miniapp_tomography.py`,
marked `slow`: the tracer against the closed form, the operator against the
composition it stands for and the library's adjoint check, in the disk and
in the ball, the ball's rays on the great circles of their ends and drawn in
the library's section frame, the shooting tracer against the radial one,
and every part through its script's entry point, the mixture's weights and
part G's derivative check among them.

| file | what it is |
|---|---|
| `rays.py` | the ray tracer in a radial medium: ray parameter, turning radius, `Delta(p)`, `T(p)`, the two-point ray, the nodes and weights; the circular rays of a flattened linear gradient, its closed form; and the shooting tracer for a medium that is not radial, `shoot` and the two-point `bend` |
| `problem.py` | the structure, the disk's network, the scales, the forward operator, the means over boxes in a disk or a ball, the prior and its exact pointwise standard deviation, `build(scale=)` for the disk |
| `common.py` | the command line, the disk's panels, the panels of an error budget, of marginals before and after, Gaussian or mixtures', and of brackets against the ball alone |
| `setup.py`, `point.py`, `bayesian.py`, `sets.py`, `mixture.py`, `ball.py`, `bent.py` | parts A to G, each a script with a `main()` and a function the others and the test call; `mixture.py` also holds the candidates, the mixture prior and the noise sweep; `ball.py` also holds the network on the sphere, the rays placed on their great circles, the bump truth and the sections and shells; `bent.py` the perturbed slowness, the bent forward operator and the Gauss-Newton MAP |
| `run_all.py` | the parts in order |

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

The forward operator's rows are assembled ray by ray in 2.7 s, from 82 775
node evaluations; the library's adjoint check finds a defect of 1.4e-13. The
truth's travel-time perturbations have a standard deviation of 0.46 against
noise of 0.05. Machine: AMD Ryzen AI 7 PRO 350 (16 cpus), Python 3.12,
numpy 2.4.

| mean over | truth | B: minimum norm, 2 propagated sd | C: posterior mean, 95% interval | D: certified bracket |
|---|---|---|---|---|
| well-covered box | +0.810 | +0.756 ± 0.064 | +0.740, [+0.590, +0.891] | [-4.20, +5.66] |
| poorly covered box | -0.801 | -0.814 ± 0.086 | -0.803, [-1.560, -0.046] | [-27.8, +26.1] |
| core | -0.182 | 0.000 ± 0.000 | 0.000, [-0.613, +0.613] | [-21.6, +21.6] |

- **B** (2 s): least squares at a guessed damping of 1 fits to a chi-squared
  of 48 against a critical 315, six times better than the data warrant; the
  discrepancy principle chooses a damping of 54 in 9 misfit evaluations, a
  factorization each. A unit bump of width 0.06 is seen at 0.63 of its height
  in the well-covered box and at 0.12 in the poorly covered one, smeared
  along the few rays that pass. In the core the estimate is zero, the noise
  it propagates is zero, and its whole error is bias, which nothing in the
  point answer reveals.
- **C** (3 to 4 s): one factorization of the 275-square normal operator
  gives the posterior mean in a fifth of a second at most; it fits the data
  to a chi-squared of 128 on 275.
  The pointwise standard deviation, from the prior's 1.0, is 0.31 in the
  mantle under the stations, 0.36 in the middle of the well-covered box, 0.78
  in the middle of the poorly covered one and 1.000 in the core, exactly; 256
  posterior samples, 2.4 s, give the map to 4%. The core's mean has the
  prior's own posterior, center zero and standard deviation 0.313.
- **D** (8 to 15 s): the prior's 95% ball has radius 24.8 in `H^2`, the
  truth's norm 24.5; the six supports take 7 to 14 s from run to run. A norm
  bound admits far more than the Gaussian prior: the brackets are tens of
  times the posterior intervals, widest where the data say least. The ball
  alone admits ±47.2 for a box's mean and ±21.8 for the core's; the data
  narrow the well-covered box's to a tenth of that, the poorly covered one's
  to a little over a half, and the core's hardly at all, to ±21.6, which is
  what the ball has left once the mantle fits the data: the smallest fitting
  model, norm 3.1, plus as much of the core mean's representer as the ball
  holds, the model the support search's inner end attains. The truth lies
  in every bracket and in the polytope they bound.
- **E** (5 s): the plume is found at its candidate, with a posterior weight
  of 0.92 against the prior's 0.2; see
  [a plume among candidates](#a-plume-among-candidates) below.
- **F** (15 s): in the ball, 174 rays of 121 nodes on a layered ball of 984
  coefficients, 240 in the core at degree 8 and 744 in the mantle at degree
  14; the truth's travel-time perturbations have a standard deviation of
  0.29 against noise of 0.05.

  | mean over | truth | minimum norm, 2 propagated sd | posterior mean, 95% interval | certified bracket | the ball alone |
  |---|---|---|---|---|---|
  | the box under the continent | +1.145 | +1.002 ± 0.042 | +1.169, [+1.054, +1.283] | [-0.37, +2.79] | ±24.0 |
  | the box at its antipode | -1.019 | -0.468 ± 0.066 | -0.652, [-1.302, -0.001] | [-11.8, +10.5] | ±24.0 |
  | the core | +0.198 | 0.000 ± 0.000 | 0.000, [-0.762, +0.762] | [-12.79, +12.79] | ±12.8 |

  The point estimate's bias at the antipode, +0.53, is sixteen times the
  noise it propagates there, and nothing in its budget shows it. The
  pointwise standard deviation is 0.12 in the middle of the box under the
  continent and 0.16 under the stations, 0.59 at the antipode's box and the
  prior's 1.00 in the core. The sections show where that comes from: the
  standard deviation falls along the rays that run near each plane and
  nowhere else, so that the meridian of 30 degrees, through the continent
  and its antipode, is dark on one side and pale on the other, and the
  meridian across them pale throughout but for the few rays that pass. The
  truth lies in every bracket and in the polytope they bound.

## A plume among candidates

Part E puts a discrete question inside the continuous problem: a plume sits
at one of a handful of places in the mantle, and which one? The plume is
part B's bump, of width 0.06 (0.12 at the quick scale), with a peak of 3%,
three times the prior's pointwise standard deviation. The candidates, of
which the quick scale keeps the first, third and fifth:

| candidate | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| radius, angle | 0.85, 75° | 0.70, 120° | 0.80, 330° | 0.75, 200° | 0.85, 270° |
| where | under the continent, in the well-covered box | under the continent, deep | at its edge | opposite it | opposite it, in the poorly covered box |
| rays within the plume's width | 22 | 20 | 10 | 7 | 2 |

Each candidate is a component of the prior, a `GaussianMixture`: the app's
prior translated to the plume there, so that every component has the app's
covariance, and the weights equal. The truth is the plume at candidate 3,
under moderate coverage, over half a draw from the shared covariance, and
the data come from it with the app's noise. Under the linear forward
problem the posterior is again a mixture (`LinearGaussianMixtureInversion`):
each component updated by its own linear Gaussian inversion, with the app's
Cholesky solver in the data space, and the weights rescored by the
components' evidences.

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

**The answer** (the whole part in 5 s; a factorization of the 275-square
normal operator for each component). The plume's travel-time perturbations have a standard
deviation of 0.078 and the truth's 0.236, against noise of 0.05 and the
shared covariance's 0.49: the plume is a small part of the signal.

| candidate | 1 | 2 | 3, the truth | 4 | 5 |
|---|---|---|---|---|---|
| misfit `M_k`, on 275 data | 183.1 | 169.0 | 161.2 | 168.3 | 167.9 |
| log evidence less the best | -10.95 | -3.92 | 0 | -3.57 | -3.35 |
| posterior weight, from 0.2 | 1.6e-5 | 0.018 | 0.923 | 0.026 | 0.032 |

A plume under the continent is ruled out, the shallow candidate at odds of
57 000 to one; where few rays pass, the data can say only that the plume is
probably not there, and candidates 4 and 5 keep 3% each. The winning
component's mean has the plume in its place at its full height, and over it
the part of the background the rays resolve: it fits the data to a
chi-squared of 111 on 275, and its error over the mantle, 0.63 of the
truth's norm, is the background they leave. On the three means the
mixtures are drawn as their densities. The prior on a box's mean is four
components at zero and, for the box's own candidate, one at +0.78, 1.15 of
the mean's prior standard deviation away: a mean over a box 30 degrees wide
averages a plume 0.06 wide down to that, and the hypothesis shows as a
shoulder, not a second mode. After the data the well-covered box's mean is
-0.538 ± 0.077 against the truth's -0.550 and the poorly covered box's
-0.12 ± 0.39 against -0.34, the plume component in it at 3%; the core's is
its prior, exactly.

**The noise sweep** (3 s): the data redrawn 32 times at each of seven noise
levels, from a tenth of the app's to a hundred times it, the components
factorized again at each. The average weights, and the effective number of
candidates, the exponential of the weights' entropy:

| noise, times the app's | 1 | 2 | 3 | 4 | 5 | effective number |
|---|---|---|---|---|---|---|
| 0.1 | 0.000 | 0.005 | 0.960 | 0.016 | 0.019 | 1.23 |
| 0.3 | 0.000 | 0.006 | 0.955 | 0.018 | 0.021 | 1.25 |
| 1 | 0.000 | 0.017 | 0.911 | 0.032 | 0.040 | 1.48 |
| 3 | 0.002 | 0.076 | 0.733 | 0.079 | 0.110 | 2.41 |
| 10 | 0.036 | 0.187 | 0.484 | 0.123 | 0.169 | 3.83 |
| 30 | 0.117 | 0.222 | 0.300 | 0.162 | 0.199 | 4.77 |
| 100 | 0.179 | 0.206 | 0.223 | 0.196 | 0.197 | 4.99 |

The weights sharpen from the prior's equal ones at a hundred times the
noise to one candidate as it falls, and below the app's level they hardly
move: the misfits are measured in the metric of `A C A* + R`, and once `R`
is small beside `A C A*` it is the prior's own variability, not the noise,
that a plume must stand out from. Nor does the noise decide which
candidates can be told apart. The weights that each plume's own travel
times would earn, noise-free and with no background, at the app's noise:

| the plume at | weight on 1 | on 2 | on 3 | on 4 | on 5 |
|---|---|---|---|---|---|
| 1 | 0.997 | 0.000 | 0.001 | 0.001 | 0.002 |
| 2 | 0.000 | 0.970 | 0.005 | 0.009 | 0.017 |
| 3 | 0.000 | 0.004 | 0.835 | 0.066 | 0.094 |
| 4 | 0.001 | 0.006 | 0.055 | 0.699 | 0.239 |
| 5 | 0.001 | 0.012 | 0.076 | 0.232 | 0.679 |

These are the weights at the mean of the log evidences: for data drawn
about candidate `j`, the difference `M_k - M_j` averages the squared
distance between the two candidates' travel times in the same metric, which
is what the noise-free data give. Under the continent each plume all but
names itself. Candidates 4 and 5, under seven rays and two, give each other
a quarter of their weight, and at a tenth of the noise still 0.23: the
coverage leaves them hard to tell apart, and less noise does not help.

The truth's background is half a draw from the shared covariance. With a
whole draw the truth would be an exact draw from its component, and a
background that happens to resemble another candidate's plume would
sometimes win: over ten seeds the true candidate's weight at the app's
noise then runs from 0.16 to 1.00 at the full scale, above a half in nine,
and from 0.20 to 0.99 at the quick scale, above a half in seven.

## Bent rays

Part G at its full size: 32 rays between 20 and 70 degrees on a layered disk
of 708 coefficients (angular order 32 in the mantle, length scale 0.15), a
prior of pointwise standard deviation 3% of the surface slowness and the
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

**The derivative** against central differences, in one direction drawn from
the prior and scaled to unit norm in `H^2` (the truth measures 42), at the
reference model and at two draws. The library's defect is the residual over
the larger of one and the directional derivative's size, here 0.28, so it is
the residual itself:

| step | reference | draw 1 | draw 2 |
|---|---|---|---|
| 20 | 5.0e-4 | 5.9e-4 | 3.9e-4 |
| 10 | 1.2e-4 | 1.7e-4 | 1.0e-4 |
| 5 | 3.1e-5 | 8.2e-5 | 4.6e-5 |
| 1 | 1.6e-5 | 1.3e-4 | 1.1e-4 |
| 0.1 | 1.8e-5 | 1.6e-4 | 1.2e-4 |

The truncation falls as the square of the step, a fourfold drop at each
halving, down to a floor of 2e-5 at the reference model and 1e-4 at the
draws. The floor is the integrator's and close to second order in its step:
at the quick size it falls from 1.6e-5 to 4.5e-6 when the step is halved.
The integrated time is a sum of the slowness at the Runge-Kutta stages,
which lie off the path by the square of the step, and its derivative differs
from the path integral along the path by about that much.

**The bending** in the truth: the travel-time perturbations have a standard
deviation of 0.86; along the bent rays they are shorter than the linear
prediction on every ray, as Fermat's principle requires, by 0.124 rms and by
0.45 at most, against noise of 0.05. The rays move by a median of 0.007 and
at most 0.058, and their deepest points by up to 0.056.

**The answers** (45 s, 36 of them the derivative check): the linear posterior
mean, on the reference rays, fits the data to a chi-squared of 0.6 on 32 as
it believes, and 21.8 along its own bent rays. The mode, by Newton-CG on the
Gauss-Newton Hessian `2 (P + J* R^-1 J)` from the prior's mean with the prior
covariance as the preconditioner, takes 7 iterations, 8 ray tracings and 6 s;
the operator's second derivative is never needed, the Gauss-Newton Hessian
being the derivative's own normal operator. It fits the data to 0.8 along its
own rays, and differs from the linear mean by 6% of it, in the upper mantle
where the rays bend most. The mean over the well-covered box moves from
-1.43 to -1.26, towards the truth's -1.14; the error over the mantle, 0.75 of
the truth's norm, and under the stations, 0.54, hardly move, being what no
ray resolves.

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
  posterior samples taking 3.5 s serially and 6.0 s on eight processes.
- A discrete question needs a prior that holds it. A mixture over the
  places a plume might be returns, in closed form and in seconds, the odds
  of each place, which no single Gaussian can; the sweep of the noise shows
  what limits them: below the app's noise the prior's own variability, and
  for two candidates the rays barely separate, the coverage.
- The ball asks the same questions of a network spread over a sphere, with
  the same tracer: a radial medium makes every ray planar, so placing the
  traced nodes on the great circle of their ends is all the third dimension
  costs. What changes is what can be seen: coverage is no longer a sector of
  a disk but a patch of the sphere and a cone beneath it, and only sections
  in several planes and a shell together show where the data reach.
- A nonlinear forward needs no more of the library than a linear one: the
  value is the user's code, and the derivative is the linear operator of the
  other parts rebuilt on the current rays. With a perturbation of 3% the
  bending moves the travel times by a few times the noise; the linear answer
  is then inconsistent with its own physics, and the mode repairs that while
  changing the model by only a few percent.
