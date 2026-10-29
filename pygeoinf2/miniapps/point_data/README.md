# Point data on the sphere

## The problem

A field on the unit sphere, observed only where there is water. The geometry
is real: the ocean is the complement of the Natural Earth 110m land polygons,
read through cartopy and tested with shapely, and the observation points are
a uniform draw over the sphere's area kept where they did not fall on land.
The field is synthetic: a draw from the prior, so that every answer below can
be marked against the truth.

The model space is the Sobolev space `H^2` on the sphere with a length scale
of 0.06 (about 380 km), and the prior is a Sobolev measure on it of order 2
with the same length scale and a pointwise standard deviation of 1, so that
its draws are smooth to the order of `H^4` and their spectrum has run out, to
one part in a million, by degree 160. The truth is observed at 20 000 ocean
points with Gaussian noise of standard deviation 0.1. Three questions are
asked of the data: the average of the field over caps of radius 0.3 (about
1900 km) centered on the North Pacific, the South Atlantic and Africa, the
third being asked precisely because there are no data under it.

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
- **C, the point answers** (`point.py`): least squares at a guessed damping
  and the minimum-norm model under the discrepancy principle; the latter's
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
  the prior's 95% ball and the noise's credible ball, as certified support
  brackets, and whether the truth lies in the polytope they bound. Its
  figure sets each bracket against the bound the prior's ball admits alone,
  before any datum, with the truth.
- **F, an extension** (`nonstationary.py`): the prior pushed through a
  multiplication operator so that its amplitude is three times larger in the
  southern hemisphere than in the northern, and D rerun under it, with D's
  figure of prior against posterior.

Two scales. `--quick` is degree 32, 400 points and a length scale of 0.15,
and runs every part in about 20 seconds; it is what the test suite runs. The
full scale is degree 128 (16 641 coefficients on a 258 by 258 grid), 20 000
points and the length scale above, and runs every part in about seven
minutes. Both are seeded.

## How to run

```
cd pygeoinf2/miniapps/point_data
MPLBACKEND=Agg python run_all.py            # every part, full scale
MPLBACKEND=Agg python run_all.py --quick    # every part, small scale
MPLBACKEND=Agg python solve.py              # one part; likewise truth.py,
                                            # point.py, bayesian.py, sets.py,
                                            # nonstationary.py
```

Every script takes `--quick`, `--outdir` and `--seed`. A part run on its own
rebuilds the problem from the seed, so it gives the same numbers as in the
full sequence; part F, which compares against D, recomputes what it needs.
Each part writes its own figure, titled with the app and the part and
showing one kind of answer, next to the sources (`quick/` at the quick
scale). The test is `pygeoinf2/tests/test_miniapp_point_data.py`, marked `slow`,
which runs the quick scale of every part through its script's entry point.

| file | what it is |
|---|---|
| `ocean_mask.py` | the land polygons, the vectorized land test, the ocean points, the grid mask, a geodesic bump |
| `problem.py` | the two scales, the regions, `build(scale=)`: space, prior, truth, observations, forward problem, the cap-average operator |
| `solvers.py` | the Woodbury surrogate on a coarse sphere: family-aware for the damped families of C and E, and for the Bayesian operator |
| `common.py` | the command line, the application counter, the map panels |
| `truth.py`, `solve.py`, `point.py`, `bayesian.py`, `sets.py`, `nonstationary.py` | parts A to F, each a script with a `main()` and a function the others and the test call |
| `run_all.py` | the parts in order |

## B. The linear algebra at the full scale

Degree 128, 20 000 ocean points, length scale 0.06, noise 0.1; the data-space
normal operator `A Q A* + R` acts on 20 000 dimensions and is never formed.
Every solve is serial, so the application counts and the wall times compare.
Machine: AMD Ryzen AI 7 PRO 350 (16 cpus, 61 GB), Python 3.12, numpy 2.4,
pyshtools 4.14, finufft for the point evaluations.

| solver | iterations | converged | build `A`/`A*` | solve `A`/`A*` | build s | solve s |
|---|---|---|---|---|---|---|
| plain CG, capped at 500 | 500 | no | 0/0 | 500/501 | 0.0 | 14.6 |
| Woodbury surrogate, degree 48 | 16 | yes | 0/0 | 16/17 | 1.1 | 0.9 |

The capped plain solve is 1.8e-3 away from the preconditioned answer in
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

Its damped relatives `A A* + t R` and `A C A* + t R` are what C and E
sweep, with the same surrogate along the family (the kernel read once, the
factorization redone at each multiplier), and the operator itself is
inverted once more for the posterior mean in D and once per posterior
sample. At degree 256 with a length scale of 0.04 and the same noise (the
first attempt at the full scale) the resolved modes numbered about twelve
thousand, a degree-32 surrogate still left 131 iterations, and a posterior
sample cost 26 s; that is why the full scale is degree 128.

## The answers at the full scale

The same three averages, by the three kinds of answer; the truth is what
the field was drawn with.

| region | truth | C: minimum norm, 2 propagated sd | D: posterior mean, 95% interval | E: certified bracket |
|---|---|---|---|---|
| North Pacific | -0.562 | -0.562 ± 0.008 | -0.563, [-0.571, -0.555] | [-0.956, -0.168] |
| South Atlantic | +0.845 | +0.842 ± 0.008 | +0.844, [+0.836, +0.852] | [+0.446, +1.242] |
| Africa | +0.593 | +0.035 ± 0.070 | -0.038, [-0.742, +0.667] | [-46.4, +46.5] |

- **C** (65 s): least squares at a guessed damping of 10 is three times
  better fitted than the data warrant; the discrepancy principle chooses a
  damping of 48.8 in 7 misfit evaluations and 87 CG iterations in all.
  Its propagated error on Africa's average, 0.035, says nothing about the
  bias there, -0.52, which only the truth reveals: the estimate sees a
  unit bump under the Pacific at 0.99 of its height and one under Africa
  at 0.17.
- **D** (36 s): the posterior mean fits the data to a chi-squared of 16 666
  on 20 000; the pointwise standard deviation is 0.04 under the oceans and
  0.87 to 0.99 under the continents, where it returns to the prior's 1.0;
  the two ocean averages are pinned to ±0.004 and Africa's is as wide as
  the prior's.
- **E** (189 s): with the prior hardened to its 95% ball, radius 47.9 in
  `H^4`, and the noise to its 95% ball, the certified brackets hold the
  truth in every region and the polytope of the six half-spaces contains
  the true averages. A norm bound admits far more than the Gaussian prior:
  the ball alone admits ±69 for each average, the data narrow the ocean
  averages to brackets 0.8 wide, and Africa's only to ±46, what is left of
  the ball once the ocean data are fitted. The six
  directions took 2662 forward and 2818 adjoint applications in eight
  threads, 189 s against about 25 s for one direction alone: the threads
  bought nothing here, because a solve is a chain of short calls that
  hold the interpreter lock between them.
- **F** (128 s): under a prior 1.5 times larger in the south and 0.5
  times in the north, the ocean averages do not move, Africa's interval
  shrinks with its prior, and the standard deviation under central Asia
  halves with the prior's amplitude there; the Woodbury surrogate still
  uses the stationary prior and the solves take 52 applications instead
  of 18.

## What the app shows

- One operator, every method: the damped relatives of the data-space
  normal operator of B are what C and E sweep, and the operator itself is
  what D inverts once and once per sample. The same surrogate on a sphere
  of degree 48 serves them all through the Woodbury identity: 16
  iterations for the Bayesian solve, ten or so per member of a damped
  family.
- At this scale nothing finishes without that preconditioner, and no
  spectral deflation of a workable rank would do: the data resolve some
  three thousand modes.
- The three answers agree where the data speak and say different things
  where they do not. Over Africa the point estimate's error budget is
  small and wrong, the posterior interval is the prior's, and the set
  answer is two thirds of what the norm bound admits alone.
- The model space is `H^4` with the identity as the prior covariance, so
  that the point and set methods damp towards the same norm the Bayesian
  prior uses; under an `H^2` norm their families resolve far beyond the
  surrogate and the sweeps crawl.
- The data outnumber the model coefficients, which makes the damped
  family singular at a multiplier of zero; the library's searches and the
  primal walk the multiplier only down to a floor on such a family, and
  that is what makes E a three-minute part rather than a stalled one.
