# Point data on the sphere: how it uses the library

The [README](README.md) says what the app asks and what it finds. This
document walks through how it is built on `pygeoinf2`: which space, which
operators, which measures, which solvers and which estimators, and why each
was chosen at this scale. It assumes the tutorials under
`pygeoinf2/tutorials/`, cited by number, and does not teach again what they
teach. Code is quoted from the app's own files.

The app has no forward code of its own, so what it shows is the other half
of an application: a space and a prior chosen together, one linear solve
made to finish at 20 000 data, and the three answers run on it unchanged.

## 1. The spaces

The problem is posed on a space appropriate to the forward model and the
answers, and the choice is a modeling decision, stated here as one: `H^2`
on the sphere, on which point evaluation, the forward operator, is bounded
and a value at a point exists (any order above one would give that, so
boundedness narrows the choice without making it). The prior is a Gaussian
measure of extra regularity on that space, a Sobolev measure of order 2,
whose covariance is trace class on `H^2`; it does not move the problem to a
smoother space, and nothing in the inference is done relative to `L^2`. The
point answers measure size in the prior's own norm, the other honest choice
being the space's: see section 5.

The unknown is a field on the unit sphere, held in
`pygeoinf2.symmetric_space.sphere.Sobolev`, whose vectors are
`pyshtools.SHGrid` objects and whose points are `(latitude, longitude)` in
degrees. From `src/problem.py`:

```python
space = Sobolev(scale.lmax, scale.order, scale.length_scale)
prior = space.sobolev_measure(
    scale.prior_order, scale.length_scale, pointwise_std=scale.prior_std
)
```

with the scale's `order` 2 and `prior_order` 2, both command-line options
like every field of the scale. Three things decide them.

- **Point evaluation.** The data are values at points, and a value at a point
  exists only on a Sobolev space of order above half the dimension, one on
  the sphere (tutorials 2 and 5). Any order above one would do for that;
  2 is the choice.
- **A measure on the space.** The covariance `(1 + L^2 lambda)^-s` in the
  space's own metric is trace class there, so that the draws are functions
  of the space, exactly when the prior's order `s` exceeds half the
  dimension, one on the sphere; at or below that the library refuses the
  measure (tutorial 5), and the `--prior-order` option says so. Order 2 is
  the safe default: the draws have nearly `s - 1 = 1` derivative more than
  the space asks.
- **The length scale** is a physical length on the unit sphere (0.06 is
  about 380 km), shared by the space's norm and the prior.

`Sobolev.truncation_degree_for(order + prior_order, length_scale,
rtol=1e-6)` reports the degree that holds the draws' spectrum to a
tolerance, 70 at the full scale, and `build` prints it; the scale's `lmax`,
128, is a choice of cost, set beside it.

The data space is the point evaluation operator's codomain, a
`EuclideanSpace` with a dimension per ocean point; the property space is
`geodesic_ball_average_operator(centers, scale.cap_radius)`'s, three
numbers.

A second sphere appears inside the preconditioner only:
`space.with_degree(scale.surrogate_degree)` is the same order and length
scale at a third of the degree (`Problem.surrogate`), and `coarse_prior` in
`src/solvers.py` puts the same measure on it. It shares nothing with the
model space but the data space, which is what section 4 needs.

Errors against the truth are taken in the area-weighted `L2` norm
(`Problem.relative_error`): the `H^2` norm weighs the degrees no datum
resolves more (tutorial 9 makes the same remark).

## 2. The forward operator

The forward operator is the library's own:
`space.point_evaluation_operator(points)`, matrix-free, its adjoint the
library's. The ocean mask (`src/ocean_mask.py`, cartopy and shapely) only chooses
the points; it is library-adjacent code that produces the geometry of the
data and never touches an operator.

What the app does wrap is a counter around it, `Counted` in `src/common.py`:

```python
self.operator = LinearOperator.from_callables(
    operator.domain, operator.codomain, self._value, adjoint=self._adjoint
)
```

Each callable counts and delegates. The adjoint handed over is the inner
operator's own adjoint, a vector of the model space, so `from_callables` is
the right constructor; a code that returned derivative components would go
through `from_derivative_callables` instead, which applies the inverse metric
itself (tutorial 3). The counts are the application columns of the README's
tables.

The metric enters only in the adjoint: the representer of evaluation in
`H^2`, the inverse Gram applied to the basis values at the points. The Gram
is diagonal here and spans many orders of magnitude, so an adjoint taken as
the transpose would be wrong by far more than rounding.

**What was checked.** The app runs no `testing.check_*`: every operator in it
is the library's, checked by the library's own suite, and the counter passes
values through unchanged. A reader who replaces point evaluation with a code
of their own should run

```python
check_operator(A, rng=rng, measure=prior)
```

before anything else. `measure=` draws the probe vectors from the prior
rather than from the space's white noise, which on a sphere is rough in every
mode at once and not the kind of field the operator will meet. A failure of
the adjoint identity on this space would almost always be the confusion of
tutorial 3: derivative components handed over as an adjoint.

## 3. The priors and the error measures

The prior is the measure of section 1, with expectation zero; the truth
and the data are drawn through the forward problem:

```python
noise = GaussianMeasure.from_standard_deviation(
    counted.operator.codomain, scale.noise_std
)
problem = LinearForwardProblem(counted.operator, error=noise)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)
```

`pointwise_std=` calibrates the measure to the number a modeler has an
opinion about, the field's standard deviation at a point (tutorial 5); on a
homogeneous space it is the same everywhere.

**A non-stationary prior** (part F, `nonstationary.py`) is the stationary one
pushed through a multiplication operator:

```python
problem.prior.affine_map(space.multiplication_operator(amplitude))
```

The push-forward of tutorial 5, with an amplitude field that is 1.5 in the
south and 0.5 in the north. Its covariance is `M Q M*` with `M` the
multiplication, its pointwise standard deviation the amplitude times the
stationary one, and nothing about it is formed.

**Hardening to sets** (part E, `sets.py`, in `hardened`):

```python
ellipsoid = problem.prior.weakened_ellipsoid(
    level=level, power=power, quantile_method="matched"
)
noise_ball = forward.error_measure.ambient_ball(level=level)
```

The noise is hardened to its 95% ball, which for an isotropic error on a
Euclidean data space is its credible ellipsoid too. The prior is hardened
to an ellipsoid of its own shape, `{m : (Q^-p m, m) <= r^2}` with `r` the
radius that carries 95% of the prior, a quantile of `sum lambda^(1-p) Z^2`
over the covariance's spectrum (tutorial 7). Which power is a matter of the
space being one of functions:

- At `p = 1` it is the Mahalanobis ellipsoid, `prior.credible_set`, and the
  library refuses it on a function space: its radius is a chi-squared
  quantile on as many degrees of freedom as there are coefficients, so it
  grows with the truncation, and the Cameron-Martin space it bounds holds
  none of the prior's draws in the limit. Its bound on a cap average alone
  is ±35 at degree 64 and ±69 at degree 128.
- At `p = 0` it is the ball in the space's norm, `prior.ambient_ball`, which
  survives refinement but forgets the covariance's shape.
- Between them the radius has a limit exactly when `sum lambda^(1-p)`
  converges, which for an order-`s` measure in `d` dimensions is
  `p < 1 - d / 2s`, one half on the sphere at order 2. The scale's
  `set_power`, one quarter, is half way: the ellipsoid bounds a norm half
  way between the space's and the smoothness the draws have. Its bound on a
  cap average alone is ±12.0 at degree 64 and ±12.9 at 128.

The covariance is diagonal in the harmonic basis, so the ellipsoid's
precision and covariance are exact diagonal operators and its support
function costs nothing. The radius is a weighted chi-square quantile over
thousands of comparable weights, which the moment-matched chi-square gives
to a few parts in a thousand; `quantile_method="matched"` says so, where
Imhof's integral, the default here, would take most of a minute at the full
scale. A set bound is a lossy step, as tutorial 14 says: it admits the field
that spends its whole budget on one cap, which the Gaussian calls unlikely;
that is why the set answer over Africa is ±7.6 where the posterior
interval is ±0.7. The two answers are not to be reconciled; they answer
different statements of what is known.

No measure is checked here; `testing.check_measure` (tutorial 5), which
samples a measure against its claimed moments, is the one for a prior built
by hand.

## 4. The solvers and the preconditioner

Every answer comes down to one operator, the data-space normal operator

```
N = A Q A* + R        on 20 000 dimensions, never formed
```

which `LinearGaussianInversion(...).normal_operator` assembles from its
factors (tutorial 4), and its damped relatives `A C A* + t R`, which the
point and set estimators sweep: with `C = Q` in the point answers, which
measure size by the prior, and with `C` the shape of the prior's ellipsoid
in the set answer. The data space is the library's default
formalism: the model-space form would need `Q^-1` and gives the same answer.

Part B (`solve.py`) sets plain conjugate gradients, capped and with
`strict=False` so that the unfinished solve is returned and shown (tutorial
4), against the preconditioner every other part uses: the Woodbury form of a
surrogate on the coarse sphere. From `src/solvers.py`:

```python
surrogate = normal.surrogate(forward=coarse_forward, prior=coarse_prior)
return CGSolver().with_preconditioner(
    WoodburyPreconditioner.from_normal(surrogate, solver=CholeskySolver())
)
```

`NormalOperator.surrogate` replaces `A` and `Q` by their versions on the
coarse sphere, sharing only the data space, which is why the data-space
formalism survives the substitution. `WoodburyPreconditioner.from_normal`
reads the factors off it and inverts it exactly through the Woodbury
identity, the inner solve one Cholesky factorization of
`Q_c^-1 + A_c* R^-1 A_c` on the coarse space. The coarse forward operator is
`coarse.point_evaluation_operator(points, dense=True)`, a dense matrix that
is cheap to apply and to multiply through. The function returns a factory
rather than a solver because the surrogate is read off whichever normal
operator the inversion builds. The surrogate's accuracy decides the
iteration count, never the answer.

Why not tutorial 4's `SpectralPreconditioner`: the data resolve some three
thousand modes, and a deflation of that rank costs more than it saves; a
coarse sphere of degree 48 carries nearly all of them for one factorization
in 2401 dimensions. Hence the README's 16 iterations against a capped 500,
with no forward application in the build: the coarse operator is a matrix.

Along a damped family the same preconditioner is built from its pieces
(`woodbury_family_solver` in `src/solvers.py`), so that it knows the
family:

```python
WoodburyPreconditioner(
    coarse_forward, prior_covariance, noise_covariance,
    prior_inverse=prior_precision, noise_inverse=noise_precision,
    solver=CholeskySolver(),
)
```

Handed the family by the estimator, it reads the kernel `A_c* R^-1 A_c` once
and refactors only the coarse inner operator at each multiplier. The prior
pieces are the coarse prior's covariance and precision for the point
answers' family, and the coarse prior's own ellipsoid of the same power for
the set answer's, its radius close enough to the fine one's for a
surrogate. The ellipsoid's shape decays more slowly than the prior's, so the
coarse sphere catches less of it: a member at a small multiplier takes tens
of iterations rather than the Bayesian solve's 16, and the set answer is
half the full run. Part F keeps the stationary surrogate under the
non-stationary prior, as a surrogate may: the solves take 52 applications
instead of 18, and give the same answer.

The posterior samples of D and F and the support directions of E take
`n_jobs=`; in E the first direction is solved serially, so that the family's
factorizations exist before the threads share them.

## 5. The estimators

**The point** (part C, `point.py`). Tutorial 8's two deterministic answers,
with size measured in the prior's norm. The problem is parameterized by the
prior's covariance factor, `m = L w`, so that `|w|` is the prior's norm of
`m` (`|m|` in the Cameron-Martin space):

```python
factor = problem.prior.covariance_factor
whitened = forward.parameterized(factor)
guess = factor(LeastSquares(whitened, damping=GUESSED_DAMPING, solver=solver)(data))
estimator = MinimumNorm(whitened, level=scale.level, solver=solver, rtol=1e-2)
search = estimator.discrepancy_search(data)
chosen = estimator.for_data(data)
smallest = factor(chosen(data))
```

The space's own `H^2` norm is the other honest choice. The prior's is taken
because the point answers then damp towards what the Bayesian answer
assumes: their family `A Q A* + t R` is the Bayesian operator's own, the one
surrogate serves both, and least squares at a damping of one is the
posterior mean exactly, which is the guess of `GUESSED_DAMPING`.
`MinimumNorm` needs the error measure, which sets the misfit level, and no
prior: the factor only says which norm. `discrepancy_search` reports the
damping, the misfit evaluations and the iterations behind them; `for_data`
returns the affine `LeastSquares` at that damping, which carries the error
budget of tutorial 14, pushed through the factor to the averages:

```python
pushed = chosen.push_forward(problem.averages @ factor)
propagated = np.sqrt(pushed.propagated_covariance().diagonals()[0])
bias = pushed.bias(whiten(problem.truth))
```

with `whiten` the factor's inverse, exact because the factor is diagonal in
the harmonic basis. The propagated covariance is the noise seen through the
estimator; the bias is `(BA - T) m`, which only the truth reveals.
`chosen.push_forward(factor).resolution(whiten(bump))` applies `L B A L` to a
unit bump's `w`, which is what the estimate makes of the bump, under the sea
and under Africa. Figure `point.png` draws each estimate with two
propagated standard deviations beside the truth plus the bias.

**The measure** (part D, `bayesian.py`). `LinearGaussianInversion` with the
Woodbury solver returns a `GaussianPosterior` (tutorial 6). The app reads
from it:

- `posterior.expectation`, one solve;
- `space.pointwise_variance_at(posterior, places)`, exact at four places,
  one solve each;
- `posterior.sample_pointwise_std(samples, rng=rng, n_jobs=n_jobs)`, the map,
  one solve per draw, with a relative standard error of about
  `1 / sqrt(2 samples)`;
- `posterior.affine_map(problem.averages)`, a Gaussian on the three
  averages, exact, whose covariance's diagonal gives the 95% intervals.

Figure `posterior.png` sets each quantity against the prior's: the two
standard-deviation maps on one color scale, and each average's marginal
before and after by `plotting.plot_densities(posterior, prior=prior, ...)`.

**The set** (part E, `sets.py`). Tutorial 7's feasible property set under
the prior's ellipsoid and the noise's ball:

```python
feasible = BackusGilbertParker(
    forward, problem.averages, ellipsoid, noise=noise_ball, solver=solver
)(problem.data)
sweep = feasible.support_sweep(directions, via=PRIMAL, n_jobs=scale.n_jobs)
```

An ellipsoid or a ball each side makes the set quadratic, and its support is
computed by the primal walk over the damped family;
`via=Primal(tolerance=1e-3, floor=1e-4)` names that algorithm with a looser
tolerance than its default and the kernel part of each direction taken at a
ten-thousandth of the family's scale rather than a millionth. The data
outnumber the coefficients at the full scale, so the walk also stops at that
floor; every walk here ends above it, as the brackets' gaps below `1e-4`
say (a walk the floor stopped leaves a wide one), and the solve at the floor
is the one most iterations go to. Each result carries a
`value`, the certified outer bound, and an `inner` value attained by a
feasible model; their difference is the bracket's gap. Six directions, plus
and minus each average, give a box, and `feasible.polytope(directions)` the
polytope whose membership of the true averages is the check. What the ellipsoid
admits with no data is its support function along each property's
representer, `ellipsoid.support_function()(averages.adjoint(q))`. Figure
`brackets.png` draws each bracket against that bound on a logarithmic axis.

## 6. What a reader would change

- **The forward operator**, in `build` in `src/problem.py`: put your operator where
  `space.point_evaluation_operator(points)` is. A hand-written one is
  `from_callables` with a true adjoint, or `from_derivative_callables` with
  the derivative components an adjoint code returns, followed by
  `check_operator` and, if it claims traits, `check_traits`, with
  `measure=prior`.
- **The space and the prior**, the two lines quoted in section 1: a space on
  which the forward operator is bounded, and a measure on it of order above
  half the dimension. Measure the point answers by the prior's factor if
  they are to damp in the norm the Bayesian one uses, and keep the set's
  power below `1 - d / 2s` if its ellipsoid is to survive refinement. To try
  another order, length scale, power or noise level without editing
  anything, pass the scale's fields on the command line (`--order`,
  `--prior-order`, `--length-scale`, `--set-power`, `--noise-std`; the
  README lists them).
- **The surrogate**, in `Problem.surrogate` in `src/problem.py`: a cheaper version of your
  forward operator on a coarser space that shares the data space. Dense if
  it fits; the closer it is, the fewer iterations, and nothing else changes.
- **The data and the questions**: the points and the noise measure, and the
  property operator in place of `geodesic_ball_average_operator`.
