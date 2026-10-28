# Point data over the oceans

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
1900 km) centred on the North Pacific, the South Atlantic and Africa, the
third being asked precisely because there are no data under it.

The same problem is answered every way the library offers, in parts that
build up from the simplest:

- **A, the setup** (`truth.py`): the space, the prior, the mask, the
  observations and the forward problem, and a map of the truth with the
  points.
- **B, the solve** (`solve.py`): the posterior mean by conjugate gradients on
  the data-space normal operator `A Q A* + R`, plain and capped, with a
  randomized spectral deflation, and with the Woodbury form of a surrogate on
  a coarse sphere; iteration counts, application counts and wall times in a
  table.
- **C, the Bayesian answer** (`bayesian.py`): the posterior mean, the
  pointwise standard deviation, samples, and the posterior on the three
  averages with their 95% intervals.
- **D, the point answers** (`point.py`): least squares at a guessed damping
  and the minimum-norm model under the discrepancy principle; the latter's
  error budget on the averages, propagated noise and bias against the truth,
  and its resolution of a localized bump under the Pacific and under Africa.
- **E, the set answer** (`sets.py`): the feasible set of the averages under
  the prior's 95% ball and the noise's credible ball, as certified support
  brackets, whether the truth lies in the polytope they bound, and the
  brackets against the intervals of C and the bars of D on one plot.
- **F, an extension** (`nonstationary.py`): the prior pushed through a
  multiplication operator so that its amplitude is three times larger in the
  southern hemisphere than in the northern, and C rerun under it.

Two scales. `--quick` is degree 32, 400 points and a length scale of 0.15,
and runs every part in about 20 seconds; it is what the test suite runs. The
full scale is degree 128 (16 641 coefficients on a 258 by 258 grid), 20 000
points and the length scale above, and is still being timed part by part (see the status at the end). Both are seeded.

## How to run

```
cd pygeoinf2/miniapps/oceans
MPLBACKEND=Agg python run_all.py            # every part, full scale
MPLBACKEND=Agg python run_all.py --quick    # every part, small scale
MPLBACKEND=Agg python solve.py              # one part; likewise truth.py,
                                            # bayesian.py, point.py, sets.py,
                                            # nonstationary.py
```

Every script takes `--quick`, `--outdir` and `--seed`. A part run on its own
rebuilds the problem from the seed, so it gives the same numbers as in the
full sequence; parts E and F, which compare against C and D, recompute what
they need. Figures are written next to the sources (`quick/` at the quick
scale). The test is `pygeoinf2/tests/test_miniapp_oceans.py`, marked `slow`,
which runs the quick scale of every part through its script's entry point.

| file | what it is |
|---|---|
| `ocean_mask.py` | the land polygons, the vectorized land test, the ocean points, the grid mask, a geodesic bump |
| `problem.py` | the two scales, the regions, `build(scale=)`: space, prior, truth, observations, forward problem, the cap-average operator |
| `solvers.py` | the two preconditioned solvers: the Woodbury surrogate for the Bayesian operator, the spectral deflation for the damped families |
| `common.py` | the command line, the application counter, the map panels |
| `truth.py`, `solve.py`, `bayesian.py`, `point.py`, `sets.py`, `nonstationary.py` | parts A to F, each a script with a `main()` and a function the others and the test call |
| `run_all.py` | the parts in order |

## B. The solve at the full scale

Degree 128, 20 000 ocean points, length scale 0.06, noise 0.1; the data-space
normal operator `A Q A* + R` acts on 20 000 dimensions and is never formed.
Every solve is serial, so the application counts and the wall times compare.
Machine: AMD Ryzen AI 7 PRO 350 (16 cpus, 61 GB), Python 3.12, numpy 2.4,
pyshtools 4.14, finufft for the point evaluations.

| solver | iterations | converged | build `A`/`A*` | solve `A`/`A*` | build s | solve s | rel. error |
|---|---|---|---|---|---|---|---|
| plain CG, capped at 500 | 500 | no | 1/0 | 1002/1001 | 0.0 | 19.8 | 0.409 |
| spectral deflation, rank 200 | 325 | yes | 841/840 | 652/651 | 19.3 | 11.8 | 0.409 |
| Woodbury surrogate, degree 48 | 16 | yes | 1/0 | 34/33 | 0.0 | 9.3 | 0.409 |

The capped plain solve is 1.3e-3 away from the Woodbury answer in relative
norm; the deflated one agrees with it to 1.7e-6 (its tolerance). The
Woodbury line's "build" is zero because its inner factorization, a Cholesky
of `Q_c^-1 + A_c* R^-1 A_c` on the 2401-dimensional coarse space, is deferred
to the first solve, so its 9.3 s is the factorization and the 16 iterations
together; the factorization is assembled column by column, 2401 applications
of the coarse operator, which is most of that time. The table was taken from
a run in which each solve was done twice (the estimator's own and a repeat
for the diagnostics); the script now does it once, and the wall times should
be re-measured.

What the table says: the data resolve the field to about degree 55 here
(the prior's spectrum against noise of 0.1 at 20 000 points), some three
thousand modes, and that is the number every preconditioner is up against. A
deflation of rank 200 covers degrees to 14 and takes the iteration count from
over 500 to 325; the surrogate on a sphere of degree 48 covers nearly all of
them and takes it to 16. At degree 256 with a length scale of 0.04 and the
same noise (the first attempt at the full scale) the resolved modes numbered
about twelve thousand, a degree-32 surrogate still left 131 iterations, and a
posterior sample cost 26 s; that is why the full scale is degree 128.

## Status

Done: the structure above, every part at the quick scale (about 20 s for all
six, checked by the test), and part B at the full scale (the table). Not yet
measured at the full scale: parts C to F, so the figures next to the sources
and the regional-average table are still to be produced, and the lessons
section is still to be written.
