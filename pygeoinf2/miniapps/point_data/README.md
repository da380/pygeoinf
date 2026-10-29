# Point data on the sphere

How the app is built on the library is walked through in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## The problem

A field on the unit sphere, observed only where there is water. The geometry
is real: the ocean is the complement of the Natural Earth 110m land polygons,
read through cartopy and tested with shapely, and the observation points are
a uniform draw over the sphere's area kept where they did not fall on land.
The field is synthetic: a draw from the prior, so that every answer below can
be marked against the truth.

The problem is posed on the Sobolev space `H^2` of the sphere with a length
scale of 0.06 (about 380 km): the data are values at points, which exist on
a Sobolev space of order above one, and `H^2` is the choice made within
that. The prior is a Gaussian measure of extra regularity on that space, a
Sobolev measure of order 2 with the same length scale, calibrated to a
pointwise standard deviation of 1: its covariance is trace class on `H^2`,
as a measure's on a function space must be, and its draws have nearly one
derivative more than the space asks. Their spectrum has run out, to one
part in a million, by degree 70. The truth is observed at 20 000 ocean
points with Gaussian noise of standard deviation 0.1. Three
questions are asked of the data: the average of the field over caps of
radius 0.3 (about 1900 km) centered on the North Pacific, the South
Atlantic and Africa, the third being asked precisely because there are no
data under it.

The same problem is answered every way the library offers, in parts that
build up from the simplest:

- **A, the setup** (`truth.py`): the space, the prior, the mask, the
  observations and the forward problem, and a map of the truth with the
  points.
- **B, the linear algebra** (`solve.py`): the one solve behind every
  answer, conjugate gradients on the data-space normal operator
  `A Q A* + R`, whose damped relatives C and E sweep and which D inverts
  once. Plain and capped, which does not finish at this scale, against the
  Woodbury form of a surrogate on a coarse sphere, which does; iteration
  counts, application counts and wall times in a table. This part is about
  the linear algebra, not the answer; the answers are C to E.
- **C, the point answers** (`point.py`): least squares at the damping a
  Bayesian would guess and the minimum-norm model under the discrepancy
  principle, both measuring size in the prior's norm; the latter's
  error budget on the averages, propagated noise and bias against the truth,
  and its resolution of a localized bump under the Pacific and under Africa.
  Its figure: the two models, the two bumps as the estimate sees them, and
  each average's estimate with two propagated standard deviations, the value
  it would take on exact data (the truth plus the bias) and the truth.
- **D, the Bayesian answer** (`bayesian.py`): the posterior mean, the
  pointwise standard deviation, samples, and the posterior on the three
  averages with their 95% intervals. Its figure sets the prior against the
  posterior: the pointwise standard deviation before and after on one color
  scale, the posterior mean and a sample, and each average's marginal before
  and after, with the truth.
- **E, the set answer** (`sets.py`): the feasible set of the averages under
  the prior's 95% ball and the noise's, as certified support brackets, and
  whether the truth lies in the polytope they bound. Its figure sets each
  bracket against the bound the prior's ball admits alone, before any
  datum, with the truth.
- **F, an extension** (`nonstationary.py`): the prior pushed through a
  multiplication operator so that its amplitude is three times larger in the
  southern hemisphere than in the northern, and D rerun under it, with D's
  figure of prior against posterior.

Two scales. `--quick` is degree 32, 400 points and a length scale of 0.15,
and runs every part in about 15 seconds; it is what the test suite runs. The
full scale is degree 128 (16 641 coefficients on a 258 by 258 grid), 20 000
points and the length scale above, and runs every part in about six
minutes, most of them the set answer and the non-stationary extension. Both
are seeded.

## How to run

```
cd pygeoinf2/miniapps/point_data
MPLBACKEND=Agg python run_all.py            # every part, full scale
MPLBACKEND=Agg python run_all.py --quick    # every part, small scale
MPLBACKEND=Agg python solve.py              # one part; likewise truth.py,
                                            # point.py, bayesian.py, sets.py,
                                            # nonstationary.py
```

A script runs from any directory, by its path. Every script takes the same
options: `--quick` or `--full` (the default) picks a preset scale, every
field of the scale is then an option that overrides it, and `--outdir` says
where the figures go. `python run_all.py --help`, in short:

```
  --quick | --full      the quick scale, or the full one (the default)
  --outdir OUTDIR       where the figures go (default: results/,
                        results/quick/ at the quick scale)
the scale, each default the full preset's:
  --seed int            the seed of the truth, the points and the noise (2026)
  --lmax int            the truncation degree of the model space (128)
  --order float         the Sobolev order of the model space, above one (2.0)
  --length-scale float  the length scale of the space and the prior (0.06)
  --prior-order float   the prior's Sobolev order on the space, above one (2.0)
  --prior-std float     the prior's pointwise standard deviation (1.0)
  --n-data int          the number of ocean observation points (20000)
  --noise-std float     the standard deviation of the noise on each datum (0.1)
  --cap-radius float    the radius of the caps averaged over (0.3)
  --level float         the credibility of the discrepancy level and the sets (0.95)
  --surrogate-degree, --cg-cap, --std-samples, --bump-width, --n-jobs
                        the solver's and the sampler's knobs
```

Two overrides at the quick scale, against its own numbers:

- **A noisier survey**, `run_all.py --quick --noise-std 0.5`: the same
  truth and points with five times the noise. The posterior standard
  deviation under the sea rises from 0.12 to 0.28 (area-weighted mean),
  the North Pacific average's from 0.030 to 0.126, and the minimum-norm
  model sees a unit bump under the North Pacific at 0.72 of its height
  rather than 0.96, and one under Africa at 0.30 rather than 0.71. The
  certified brackets widen with the noise's ball, from 2.2 to 5.2 wide on
  the North Pacific and from 8.5 to 12.1 on Africa, and still hold the
  truth.
- **A sparser network**, `run_all.py --quick --n-data 100`: a quarter of
  the points. The points are drawn before the truth, so this is also a
  different draw of the truth. Under the sea the posterior standard
  deviation rises from 0.12 to 0.29, central Asia's from 0.74 to 0.87, and
  the South Atlantic bracket is 5.6 wide rather than 3.5.

A part run on its own rebuilds the problem from the seed, so it gives the
same numbers as in the full sequence; part F, which compares against D,
recomputes what it needs. Each part writes its own figure, titled with the
app and the part and showing one kind of answer, into `results/`
(`results/quick/` at the quick scale), which git ignores. The test is
`pygeoinf2/tests/test_miniapp_point_data.py`, marked `slow`, which builds the
quick scale through the same command line and runs every part through its
script's entry point.

| file | what it is |
|---|---|
| `src/ocean_mask.py` | the land polygons, the vectorized land test, the ocean points, the grid mask, a geodesic bump |
| `src/problem.py` | the scale with its two presets, the regions, `build(scale=)`: space, prior, truth, observations, forward problem, the cap-average operator |
| `src/solvers.py` | the prior on a coarse sphere and the Woodbury surrogate built from it: family-aware for the damped families of C and E, and for the Bayesian operator |
| `src/common.py` | the command line built from the scale, the application counter, the map panels |
| `truth.py`, `solve.py`, `point.py`, `bayesian.py`, `sets.py`, `nonstationary.py` | parts A to F, each a script with a `main()` and a function the others and the test call |
| `run_all.py` | the parts in order |
| `results/` | the figures a run writes, `results/quick/` at the quick scale |

## B. The linear algebra at the full scale

Degree 128, 20 000 ocean points, length scale 0.06, noise 0.1; the data-space
normal operator `A Q A* + R` acts on 20 000 dimensions and is never formed.
Every solve is serial, so the application counts and the wall times compare.
Machine: AMD Ryzen AI 7 PRO 350 (16 cpus, 61 GB), Python 3.12, numpy 2.4,
pyshtools 4.14, finufft for the point evaluations.

| solver | iterations | converged | build `A`/`A*` | solve `A`/`A*` | build s | solve s |
|---|---|---|---|---|---|---|
| plain CG, capped at 500 | 500 | no | 0/0 | 500/501 | 0.0 | 15.0 |
| Woodbury surrogate, degree 48 | 16 | yes | 0/0 | 16/17 | 1.2 | 0.9 |

The capped plain solve is 1.3e-3 away from the preconditioned answer in
relative norm, and it was not going to arrive: the data resolve the field to
about degree 55 (the prior's spectrum against noise of 0.1 at 20 000
points), some three thousand modes, and that is the number of distinct
eigenvalues plain conjugate gradients has to work through. The surrogate is
the same problem on a sphere of degree 48, which carries nearly all of those
modes, and the Woodbury identity inverts it exactly through one Cholesky
factorization of `Q_c^-1 + A_c* R^-1 A_c` on the 2401-dimensional coarse
space: the build column is that factorization, after which each application
of the preconditioner is a pair of coarse matrix products. The surrogate's
forward operator is a dense 20 000 by 2401 matrix, so none of its
applications are counted; its cost is in the build.

Its damped relatives are what C and E sweep with the same surrogate along
the family (the kernel read once, the factorization redone at each
multiplier): `A Q A* + t R` itself in C, which measures size by the prior,
and `A A* r^2 + t R` in E, with `r` the radius of the prior's ball. The
operator itself is inverted once more for the posterior mean in D and once
per posterior sample.

## The answers at the full scale

The same three averages, by the three kinds of answer; the truth is what
the field was drawn with.

| region | truth | C: minimum norm, 2 propagated sd | D: posterior mean, 95% interval | E: certified bracket |
|---|---|---|---|---|
| North Pacific | -0.562 | -0.562 ± 0.008 | -0.563, [-0.571, -0.555] | [-1.006, -0.118] |
| South Atlantic | +0.845 | +0.842 ± 0.008 | +0.844, [+0.836, +0.852] | [+0.409, +1.277] |
| Africa | +0.593 | +0.035 ± 0.070 | -0.038, [-0.742, +0.667] | [-5.51, +5.46] |

- **C** (18 s): least squares at a damping of one in the prior's norm is the
  posterior mean of D, and fits to a chi-squared of 16 666 against a
  critical 20 330; the discrepancy principle chooses a damping of 6.6 in 6
  misfit evaluations and 48 CG iterations in all, and its model is
  smaller in the prior's norm than the truth, 37 against 130.
  Its propagated error on Africa's average, 0.035, says nothing about the
  bias there, -0.52, which only the truth reveals: the estimate sees a
  unit bump under the Pacific at 0.99 of its height and one under Africa
  at 0.17.
- **D** (43 s): the posterior mean fits the data to a chi-squared of 16 666
  on 20 000; the pointwise standard deviation is 0.04 under the oceans and
  0.87 to 0.99 under the continents, where it returns to the prior's 1.0;
  the two ocean averages are pinned to ±0.004 and Africa's is as wide as
  the prior's.
- **E** (115 s, the six directions in six worker processes): with the
  prior hardened to its 95% ball, `||m|| <= r` in the space's own `H^2`
  norm with `r = 6.33`, which the truth lies in at 0.92 of the boundary's
  gauge `(||m|| / r)^2`, and the noise to its 95% ball, the certified
  brackets hold the truth in every region, each to a gap below 4e-4, and
  the polytope of the six half-spaces contains the true averages. A set bound admits far more than the Gaussian posterior: the
  ball alone admits ±10.1 for each average, the data narrow the ocean
  averages to brackets 0.89 and 0.87 wide, and Africa's to ±5.5, what is
  left of the ball once the ocean data are fitted. The six directions
  take about 3700 forward and 3900 adjoint applications in all, in some
  35 solves apiece, and 39 to 52 s each when run one at a time; each
  worker builds the family's factorizations for its own direction. The
  ball's family `A A* r^2` decays only as the space's own kernel does,
  more slowly than the prior's, so the coarse surrogate catches less of
  it: a solve takes from a few iterations to two dozen, and the one at the
  walk's smallest multiplier 70 to 130.
- **F** (about 160 s, 100 of them the 64 samples): under a prior 1.5 times
  larger in the south and 0.5 times in the north, the ocean averages do
  not move, Africa's interval shrinks with its prior, and the standard
  deviation under central Asia halves with the prior's amplitude there;
  the Woodbury surrogate still uses the stationary prior and the solves
  take 52 applications instead of 18.

## What the app shows

- One operator, every method: the damped relatives of the data-space
  normal operator of B are what C and E sweep, and the operator itself is
  what D inverts once and once per sample. The same surrogate on a sphere
  of degree 48 serves them all through the Woodbury identity: 16
  iterations for the Bayesian solve, eight or so per member of C's family,
  up to two dozen for E's and more at the end of its walk.
- At this scale nothing finishes without that preconditioner, and no
  spectral deflation of a workable rank would do: the data resolve some
  three thousand modes.
- The three answers agree where the data speak and say different things
  where they do not. Over Africa the point estimate's error budget is
  small and wrong, the posterior interval is the prior's, and the set
  answer is a little over half of what the prior's ball admits alone.
- The model space is `H^2`, the prior a measure of order 2 on it, and the
  point answers measure size by the prior's own norm, so that they damp
  towards what the Bayesian answer assumes and the one surrogate serves
  all three. The set answer hardens the prior to the ball of the space's
  own norm that carries 95% of it, the classical norm bound with the prior
  calibrating its radius: a draw's squared norm has the covariance's trace
  as its expectation, finite on `H^2`, so the radius has a limit, and the
  ball's bound on Africa's average alone is ±9.86 at degree 64, ±10.07 at
  128 and ±10.13 at 256. The Mahalanobis ellipsoid of the covariance has a
  radius that grows with the truncation, as the square root of the
  coefficient count, and alone would bound the same average by ±35 at
  degree 64 and ±69 at 128. The ball is a convenience, one constraint set
  among many: a bound the physics suggests, on another norm or a region,
  goes to the same estimator.
- The data outnumber the model coefficients, which makes the damped
  family singular at a multiplier of zero; the library's searches and the
  primal walk the multiplier only down to a floor on such a family, set
  here at a ten-thousandth of the family's scale; the six walks end above
  it, which their gaps below 4e-4 show.
