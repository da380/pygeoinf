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
- **B, the Bayesian answer** (`bayesian.py`): the posterior mean, the
  pointwise standard deviation (exactly at four points, by sampling over the
  whole disk), two samples, and the posterior on the three means with their
  95% intervals.
- **C, the point answers** (`point.py`): least squares at a guessed damping,
  the minimum-norm model under the discrepancy principle, its resolution of
  a unit bump in each box, and its error budget on the three means.
- **D, the set answer** (`sets.py`): the feasible set of the three means
  under the prior's 95% ball and the noise's credible ball, as certified
  brackets in the six coordinate directions, the truth against the polytope
  they bound, and the three kinds of answer on one plot.
- **E, the ball** (`ball.py`): the same planet as a layered ball of degree
  12, the rays the quick network's in its equatorial plane, traced by the
  same tracer with the angle read as a longitude, and B to D run on it
  unchanged; the figures are sections through that plane.
- **F, bent rays** (`bent.py`): the forward problem as it is rather than as
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
and runs every part in about ten seconds; it is what the test suite runs.
The full scale is angular order 96 in the mantle and 53 in the core (4688
coefficients), 275 rays of 301 nodes each and the length scale above, and
runs parts A to E in about half a minute. The ball, part E, is the same toy
at both. Part F has its own two sizes, 19 rays at angular order 16 and 32
rays at order 32, and takes 25 s and 45 s. A full run of every part takes
about 75 s. All are seeded.

## How to run

```
cd pygeoinf2/miniapps/tomography
MPLBACKEND=Agg python run_all.py            # every part, full scale
MPLBACKEND=Agg python run_all.py --quick    # every part, small scale
MPLBACKEND=Agg python bayesian.py           # one part; likewise setup.py,
                                            # point.py, sets.py, ball.py,
                                            # bent.py
```

Every script takes `--quick`, `--outdir` and `--seed`. A part run on its own
rebuilds the problem from the seed, so it gives the same numbers as in the
full sequence; part D, which compares against B and C, recomputes what it
needs. Figures are written next to the sources (`quick/` at the quick scale).
The test is `pygeoinf2/tests/test_miniapp_tomography.py`, marked `slow`: the
tracer against the closed form, the operator against the composition it
stands for and the library's adjoint check, the shooting tracer against the
radial one, and the quick scale of every part through its script's entry
point, part F's derivative check among them.

| file | what it is |
|---|---|
| `rays.py` | the ray tracer in a radial medium: ray parameter, turning radius, `Delta(p)`, `T(p)`, the two-point ray, the nodes and weights; the circular rays of a flattened linear gradient, its closed form; and the shooting tracer for a medium that is not radial, `shoot` and the two-point `bend` |
| `problem.py` | the structure, the network, the scales, the forward operator, the three means, the prior, `build(scale=)` for the disk and the ball |
| `common.py` | the command line, the figure panels for a disk or a ball's section |
| `setup.py`, `bayesian.py`, `point.py`, `sets.py`, `ball.py`, `bent.py` | parts A to F, each a script with a `main()` and a function the others and the test call; `bent.py` also holds the perturbed slowness, the bent forward operator and the Gauss-Newton MAP |
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

| mean over | truth | B: posterior mean, 95% interval | C: minimum norm, 2 propagated sd | D: certified bracket |
|---|---|---|---|---|
| well-covered box | +0.810 | +0.740, [+0.590, +0.891] | +0.756 ± 0.064 | [-4.20, +5.66] |
| poorly covered box | -0.801 | -0.803, [-1.560, -0.046] | -0.814 ± 0.086 | [-27.8, +26.1] |
| core | -0.182 | 0.000, [-0.613, +0.613] | 0.000 ± 0.000 | [-21.6, +21.6] |

- **B** (3 to 4 s): one factorization of the 275-square normal operator
  gives the posterior mean in a fifth of a second at most; it fits the data
  to a chi-squared of 128 on 275.
  The pointwise standard deviation, from the prior's 1.0, is 0.31 in the
  mantle under the stations, 0.36 in the middle of the well-covered box, 0.78
  in the middle of the poorly covered one and 1.000 in the core, exactly; 256
  posterior samples, 2.4 s, give the map to 4%. The core's mean has the
  prior's own posterior, center zero and standard deviation 0.313.
- **C** (2 s): least squares at a guessed damping of 1 fits to a chi-squared
  of 48 against a critical 315, six times better than the data warrant; the
  discrepancy principle chooses a damping of 54 in 9 misfit evaluations, a
  factorization each. A unit bump of width 0.06 is seen at 0.63 of its height
  in the well-covered box and at 0.12 in the poorly covered one, smeared
  along the few rays that pass. In the core the estimate is zero, the noise
  it propagates is zero, and its whole error is bias, which nothing in the
  point answer reveals.
- **D** (10 to 15 s): the prior's 95% ball has radius 24.8 in `H^2`, the
  truth's norm 24.5; the six supports take 8 to 14 s from run to run. A norm bound admits far more than the
  Gaussian prior: the brackets are tens of times the posterior intervals,
  widest where the data say least. The core's bracket is what the ball has
  left once the mantle fits the data: the smallest fitting model, norm 3.1,
  plus the core mean's representer filled out to the ball, a model built and
  checked against both constraints that attains the outer value to 1e-9. The
  truth lies in every bracket and in the polytope they bound.
- **E** (7 s): in the ball, 40 rays in its equatorial plane on a layered ball
  of 764 coefficients. The structure reads the same: the standard deviation
  0.24 in the well-covered box, 0.60 in the poorly covered one and the
  prior's in the core; the bump seen at 0.61 and at 0.38; the truth in every
  bracket. The mantle off the plane of the rays is the prior's, which is why
  the posterior mean's error over the whole mantle is 0.89 of the truth's
  norm, against 0.52 in the disk.

## Bent rays

Part F at its full size: 32 rays between 20 and 70 degrees on a layered disk
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
to E built on them, with Simpson's weights along 401 nodes a ray.

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
  posterior widens, the resolution smears along them, and the point
  estimate's propagated error stays as small as it was under the stations.
  Where no ray goes, the core, the posterior is exactly the prior and the
  point estimate is zero with no error bar: only the measure and the set say
  how little is known.
- A layered space is the natural model of a discontinuity the data never
  cross: the core's posterior is its prior by construction, not by
  approximation.
- With a few hundred data and thousands of coefficients every solve is
  direct, and the methods that sweep a damped family pay a factorization per
  member and nothing else. The loops that could run in parallel are run
  serially: worker processes cost more than they save at this size, 256
  posterior samples taking 3.5 s serially and 6.0 s on eight processes.
- The ball runs the same parts with the same tracer: only the space, the
  points the nodes are lifted to and the figures change.
- A nonlinear forward needs no more of the library than a linear one: the
  value is the user's code, and the derivative is the linear operator of the
  other parts rebuilt on the current rays. With a perturbation of 3% the
  bending moves the travel times by a few times the noise; the linear answer
  is then inconsistent with its own physics, and the mode repairs that while
  changing the model by only a few percent.
