# Acoustic waveform inversion: how it uses the library

The [README](README.md) says what the survey sees and what each estimator
makes of it. This document walks through how the app is built on
`pygeoinf2`: a C++ time-stepper, its linearization and its first and second
adjoints, wrapped as one `Operator` with both derivatives, proved by the
check ladder, and handed to the library's nonlinear estimators unchanged. It
is about the wrapping, not the finite-difference scheme, which the README
sets out. It assumes the tutorials under `pygeoinf2/tutorials/`, cited by
number; tutorials 9 and 13 are the nonlinear machinery on toys, and
tutorial 12 is the ladder. Code is quoted from the app's own files.

## 1. The spaces

The problem is posed on a space appropriate to the forward model and the
answers, and the choice is a modeling decision, stated here as one: `H^2` on
the periodic box laid over the grid. No datum is a value of `m` at a point,
so point evaluation does not set the order; what the forward model needs is
a slowness bounded away from zero near the background, which in two
dimensions any order above one guarantees, and that narrows the choice
without making it. Order two is chosen, and its norm is the one Occam's
model is smallest in. The prior is a measure of extra regularity on that
space, a Sobolev measure of order two (section 3): it does not move the
problem to a smoother space, and nothing is done relative to `L^2`.

The unknown is `m = 1 / c²`, the slowness squared, on the solver's grid. The
model space is the library's periodic box laid over the whole grid, the
sponge included, so that its vectors are the solver's own `(nx, nz)` arrays
and nothing is converted between the two. From `model_space` in
`src/survey.py`:

```python
scale = survey.scale
return Sobolev(survey.shape, scale.order, scale.length_scale, lengths=survey.lengths)
```

from `pygeoinf2.symmetric_space.fourier`, with the scale's `order` 2 and
`length_scale` 0.1 km, both command-line options like every field of the
scale. The space's metric runs from 1 to `3.9e6` on the full grid, which is
also what makes the adjoint checks below able to fail: on a space whose
Gram matrix is the identity a transpose passes for an adjoint.

The box is periodic, so the prior correlates opposite edges of the grid a
correlation length apart; the edges are sponge, and the shots, the
receivers, the boxes asked about and the errors reported all lie within the
section.

The data space is `EuclideanSpace(survey.data_size)`, every receiver at
every step of every shot, flattened: 120 600 numbers at the full scale. The
property space is `EuclideanSpace(4)`, the mean of `m` over four boxes.

## 2. The forward code wrapped as an operator

The forward code is `cpp/acoustic.cpp`, bound with nanobind as
`acoustic.Model`.
Its kernels run a shot forward, run the linearized field along a direction
(`born`), run the adjoint backward and correlate it into a grid gradient in
one pass (`gradient`), and do the same for the second-order adjoint
(`curvature`, `second_born`). Fields stay in C++ behind integer handles, and
the kernels release the interpreter lock. `WaveformOperator`, in
`src/wave_operator.py`, turns them into the library's objects:

```python
self.operator = Operator.from_callables(
    space, self.data_space, self.value,
    derivative=self.derivative, second_derivative=self.second_derivative,
)
self.first_order = Operator.from_callables(
    space, self.data_space, self.value, derivative=self.derivative
)
```

`first_order` is the same map without its second derivative, sharing the
cache; on it a Newton method gets the Gauss-Newton Hessian.

**The derivative** at `m` is a linear operator whose action is one
linearized run per shot and whose adjoint is one backward run per shot. An
adjoint code returns a grid gradient, the derivative of `(F'(m) x, r)` with
respect to the grid values: derivative components, not a vector of the
space (tutorial 3). So the derivative is built from those:

```python
return LinearOperator.from_derivative_callables(
    self.space, self.data_space, value, derivative_components
)
```

with `derivative_components` returning
`self._scale * self.space.to_components(np.sum(parts, axis=0))`. The box's
components are orthonormal in `L2` and a grid sum is `cells / area` times an
integral, so the derivative components of `x -> g . x` are `cells / area`
times the components of `g`; the library then applies the inverse metric.
That one line is where the metric enters, and the forward action never sees
it. **The second derivative** `second_derivative(m, dm)` returns
`F''(m)[dm, .]` built the same way: action `second_born`, adjoint
`curvature`. The box means of part E are `from_derivative_callables` too.

**What else the wrapper does**, each a library protocol:

- `admissible(m)` raises `EvaluationError` for a model with a non-positive
  value or fast enough to break the scheme's stability limit, from which the
  line searches retreat (tutorial 9);
- the forward fields of the last model are cached by a hash of `m`, since
  the value, the derivative and the second derivative at one model all read
  them;
- the shots run on threads (`n_jobs`), the kernels releasing the lock, and
  the curvature's on `curvature_jobs` threads to bound the fields stored.

Across MPI ranks (part F) the only change is the module, passed as
`backend=acoustic_mpi` to `synthetic.setup`. The operator, the checks and
the estimators are the serial ones, and the answers agree bit for bit.

**What was checked** (`proofs.py`, on the proof grid and the scale's grid).
The app's own identities first: the stencil's transpose against its literal
scatter, the fused kernels against the unfused compositions in
`WaveformOperator.unfused_*`, dot tests of `F'(m)` and `F''(m)[d, .]` taken
against `|A x| |y|`, and central differences at a ladder of steps. Then the
library's ladder, `proofs.library_ladder`, as a test suite runs it, every
probe drawn from the zero-mean prior `experiment.probes` with `measure=`:

```python
"check_operator(F'(m))": check_operator(operator.derivative(point), rng=rng, measure=probes),
"check_derivative(F)": check_derivative(operator, point, rng=rng, measure=probes),
"check_second_derivative(F)": check_second_derivative(operator, point, rng=rng, measure=probes),
```

with `check_operator` of `F''(m)[d, .]` and of the box means, and two on the
objective a Newton method minimizes:
`check_gradient(MaximumAPosteriori(problem, prior).objective(data), ...)` and
`check_traits(objective.hessian(point), ..., rtol=1e-8)`, the exact
Hessian's claimed symmetry. The README tabulates the defects: a few
`1e-14` or less for the adjoint identities, a few `1e-9` for the differences,
which
fall a hundredfold per decade of step until rounding.

What a failure of each would have meant. `check_operator(F'(m))`: the
adjoint's time alignment wrong, or the grid gradient handed over without its
scale or as a vector rather than derivative components.
`check_derivative`: the linearized run not the derivative of the forward
one, a taper on the wrong side of a step. `check_second_derivative`:
`second_born` wrong. `check_operator(F''(m)[d, .])`: the second adjoint
wrong, a sign in its last term, which the differences of `F'` cannot see
since they test only the forward action. `check_traits` of the exact
Hessian: the same, seen from the objective, since the Hessian is symmetric
only if the curvature term's adjoint is right. The test also checks the
Gauss-Newton Hessians of `MaximumAPosteriori` and of Occam's constraint for
symmetry.

## 3. The priors and the error measures

The prior is a Sobolev measure of order two on the model space about the
known background, and the probes of the check ladder are the same measure
about zero:

```python
probes = space.sobolev_measure(
    scale.prior_order, scale.length_scale, pointwise_std=scale.prior_std
)
prior = space.sobolev_measure(
    scale.prior_order,
    scale.length_scale,
    pointwise_std=scale.prior_std,
    expectation=background,
)
```

with the scale's `prior_order` 2 and `prior_std` 0.02 s²/km². Its covariance
is `(1 + L^2 lambda)^-2` in the space's own metric, scaled to the pointwise
standard deviation: trace class on `H^2`, because the order exceeds half the
dimension, one, so that its draws are functions of the space with nearly
one derivative more than it asks (tutorial 5). The library refuses an order
at or below one, and the `--prior-order` option says so. The MAP model's
penalty is the prior's norm; Occam's distance is the space's own `H^2`
norm, a different one now, which is what sets the two answers apart beyond
the level each fits to. A wider prior would draw models fast enough to
break the stability limit.

The noise is `GaussianMeasure.from_standard_deviation(wave.data_space,
noise_std)`, 3% of the data's rms; the data are the truth's plus a draw from
it. Two forward problems share it, one on `wave.operator` and one on
`wave.first_order`:

```python
problem = ForwardProblem(wave.operator, error=noise)
first_order = ForwardProblem(wave.first_order, error=noise)
```

No measure is hardened: this app answers with points and a measure only.

## 4. The solvers

The MAP model and Occam's model are minimizations, and the minimizer is the
library's `NewtonCG`, set up by `newton` in `src/inversion.py`:

```python
return LoggedNewtonCG(
    max_iterations=..., max_cg_iterations=...,
    preconditioner=experiment.prior.covariance,
    forcing=forcing, rtol=0.0, gtol=0.0, ftol=ftol,
)
```

`LoggedNewtonCG` is `NewtonCG` with a record of the value and gradient norm
at each linearization, made by wrapping the objective in
`Functional.from_callables(space, functional, linearize=..., hessian=...)`:
a subclass that adds nothing to the algorithm. The inner conjugate gradients
are preconditioned by the prior covariance, so that they see `I + Q J* R^-1
J` rather than the prior precision's spectrum of many decades, and capped at
40; with the exact
Hessian they stop at the first direction of negative curvature, and
`result.inner_stops` says why each one stopped, which is how the README
knows the exact Hessian is indefinite at the background and not at the
mode. Kernel runs are counted per shot by `Stopwatch` (`src/common.py`) over
`wave.calls`: the columns of the README's table. A Gauss-Newton Hessian
application costs one linearized and one adjoint run per shot; an exact one
adds a linearized run and a backward pass carrying two adjoint fields.

Occam's Newton-CG has no preconditioner: its objective's quadratic term is
the space's own norm, whose Hessian is the identity, so the inner solves
already see `I` plus the data's term. The prior covariance there would
precondition the wrong operator: at the quick scale Occam then ran out its
ten outer iterations, 55 Newton iterations against 17, without converging.

The Laplace covariance on the box means (part E) is four solves with the
normal operator at the mode, `P + J* R^-1 J`, whose prior precision spans
many decades. Seen through the prior's covariance factor `L` it is
`I + L* J* R^-1 J L`, the identity but for the directions the data inform,
and a randomized eigendecomposition of that operator, tutorial 4's
`SpectralPreconditioner` with the identity's damping of one, finds them; `L`
on either side takes the inverse back:

```python
def solver(normal):
    whitened = (factor.adjoint @ normal @ factor).with_traits(Traits.POSITIVE_DEFINITE)
    deflated = SpectralPreconditioner(rank=scale.laplace_rank, damping=1.0, ...)(whitened)
    return CGSolver(..., preconditioner=factor @ deflated @ factor.adjoint, ...)
```

`LaplaceApproximation` takes the factory and hands it the normal operator.
A rank of 100 takes 440 applications to build and brings the solves to 16
or 17 iterations each, where the prior covariance alone leaves them at the
cap of 120 with their residuals reduced to between 1e-3 and 1e-2; a
spectral preconditioner of the normal operator itself would find the prior
precision's largest eigenvalues, not the data's. Its probes run on threads inside
`joblib.parallel_config(backend="threading")`, because the operator holds
the C++ solver and cannot be sent to a process. `strict=False` returns a
capped solve with a warning rather than raising, and the `callback` records
each iterate's box means, which is how the app knows each standard deviation
had settled: a conjugate-gradient solve stopped early underestimates a
variance.

## 5. The estimators

**The point, by the MAP model** (part C, `newton.py`), three runs of
`MaximumAPosteriori`, each `.solve(data, start=...)`:

```python
MaximumAPosteriori(state.first_order, prior, optimizer=optimizer, gauss_newton=True)
MaximumAPosteriori(state.problem, prior, optimizer=optimizer)
```

The first, on the operator without its second derivative, minimizes with the
Gauss-Newton Hessian `2 (P + J* R^-1 J)`. The second, on the full operator,
carries the exact Hessian, which adds the second derivative weighted by the
residual; it starts from where the first stopped, and is run once more from
the background to show what the exact Hessian costs there. The estimator
needs the forward problem with its error measure, and the prior for the
precision term; the exact Hessian also needs the operator's second
derivative. Its result carries the minimizer, the iterations, the stopping
message and the inner stops. The modes are saved for parts D and E, and
figure `map.png` draws the three models and how each converged.

**The point, by Occam** (part D, `occam.py`):

```python
estimator = Occam(
    state.first_order, gauss_newton=True, level=scale.level,
    reference=experiment.background, optimizer=NewtonCG(...), start=mode, ...
)
```

The model closest to the background in the model space's norm whose
chi-squared meets its 95% critical value, `estimator.target`, by an
augmented Lagrangian over Newton-CG solves started from the MAP mode. It
needs the forward problem's error measure for the level and a reference,
and no prior: its norm is the space's `H^2`, not the prior's. Its result
carries the outer iterations, the constraint's violation and its
multiplier. Figure `occam.png` sets it against the MAP mode and the truth.

**The measure, by the Laplace approximation** (part E, `laplace.py`):

```python
laplace = LaplaceApproximation(
    MaximumAPosteriori(state.first_order, prior, gauss_newton=True),
    solver=solver,
).at(mode, state.data)
on_boxes = laplace.affine_map(boxes)
```

`at` places the Gaussian on the saved mode without searching again
(tutorial 9); its covariance is the inverse of the Gauss-Newton normal
operator there, each application a solve. Pushed through the box means it is
a Gaussian on four numbers whose covariance matrix costs four solves; the
prior's push-forward beside it costs nothing. Figure `laplace.png` draws
each mean's two standard deviations before and after, with the truth.

**No set.** A nonlinear forward's set answer in the library is the feasible
set of the problem linearized about a model, `BackusGilbertParker(...,
model=mode)` (tutorial 14); this app does not run it.

## 6. What a reader would change

- **The kernels**: your solver's forward run, its linearized run, and its
  adjoint returning a grid gradient, with the second-order pair if a Newton
  method should see the exact Hessian. Bind them as you like; the library
  needs only callables.
- **The wrapping**: `WaveformOperator`, with `from_derivative_callables`
  and the scale that turns your grid gradient into derivative components of
  your space; a cache keyed on the model; `EvaluationError` for a model the
  scheme cannot run.
- **The space and the prior**: a box over your grid, so that the vectors are
  your arrays, and a Sobolev measure on it of order above half the
  dimension; its covariance preconditions the MAP's inner solves, and its
  factor whitens the normal operator for the Laplace solves.
- **The proofs**: `proofs.library_ladder` as it stands, with probes from
  your prior, before any estimator is run; and the survey, the noise and the
  properties in `src/survey.py`, the size and the physics in its `Scale`,
  whose every field is a command-line option.
