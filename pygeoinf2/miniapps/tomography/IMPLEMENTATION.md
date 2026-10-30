# Ray tomography: how it uses the library

The [README](README.md) says what the rays see and what each answer makes of
it. This document walks through how the app is built on `pygeoinf2`: a ray
tracer that knows nothing of the library turned into an operator, the spaces
it acts on, the priors, the solves and every kind of answer. It assumes the
tutorials under `pygeoinf2/tutorials/`, cited by number; tutorial 13 is ray
tomography on a toy plane with the library's own tracer. Code is quoted from
the app's own files.

## 1. The spaces

The problem is posed on a space appropriate to the forward model and the
answers, and the choice is a modeling decision, stated here as one: the
layered disk, or ball, in `H^2`. The forward operator, point evaluation at
the ray nodes followed by quadrature weights, is bounded wherever a value at
a point exists, above order one in the disk and three halves in the ball,
which narrows the choice without making it; `H^2` is chosen within that.
The prior is a Gaussian measure of extra regularity on that space, a
Sobolev measure of order 2 on each layer, whose covariance is trace class
on `H^2`; it does not move the problem to a smoother space, and nothing is
done relative to `L^2`. The point answers measure size in the prior's own
norm and the set answer hardens the prior to the ball of the space's norm
that carries 95% of it (sections 3 and 5).

The unknown is the slowness perturbation, in percent of the surface
slowness, in a planet of radius one with a core no ray enters. It is
discontinuous at the core's edge, so the space is a direct sum of two layers
(tutorial 10), each in the eigenbasis of `1 - div(L² grad)` on a padded
spectral-element mesh (tutorial 11). From `model_space` in `src/problem.py`:

```python
kind = ball.Layered if scale.is_ball else disk.Layered
return kind(
    [0.0, CORE_RADIUS, RADIUS],
    [inner, scale.mmax],
    order=scale.order,
    length_scale=scale.length_scale,
    labels=["core", "mantle"],
)
```

`pygeoinf2.disk.Layered` holds Fourier modes in angle times radial
eigenfunctions under `r dr`, its points `(radius, angle)` with the angle in
degrees; `pygeoinf2.ball.Layered` holds spherical harmonics times radial
eigenfunctions under `r² dr`, its points `(radius, latitude, longitude)`.
One function builds either, and everything downstream serves both.

- **The order**, the scale's `order`, 2 (a command-line option like every
  field of the scale). The forward operator is point evaluation at
  the ray nodes, and a point value needs order above half the dimension: one
  in the disk, three halves in the ball (tutorials 2 and 5). `H^2` clears
  both.
- **The truncation** at the full scale: angular order 96 in the mantle and,
  in proportion to its radius, 53 in the core; a length scale of 0.1.
- **The two layers.** A layer the data never reach is independent of the
  one they do, and the core's posterior comes out exactly its prior: by
  construction, not by approximation.

The data space is `EuclideanSpace(len(rays))`, in hundredths of a time
unit; the property space is `EuclideanSpace(3)`, three means.

## 2. The forward code wrapped as an operator

The forward code is `src/rays.py`, a tracer in a radial medium written without
the library. For a source and a receiver it returns a `Ray`: nodes at equal
arc length along the curved ray, and the trapezoid weights along the
polyline through them, so that the path integral of a field `f` is
`weights @ f(nodes)`. That sentence is the operator: point evaluation at
every node, then a matrix of weights. `composed_operator` in `src/problem.py` writes it
exactly so:

```python
evaluation = space.point_evaluation_operator(lift(scale, nodes))
weights = LinearOperator.from_matrix(
    evaluation.codomain, EuclideanSpace(len(rays)), path_weights(rays),
    form="components",
)
return weights @ evaluation
```

The composition's adjoint is the library's, `evaluation.adjoint @
weights.adjoint`, and the metric enters through the adjoint of point
evaluation: the representer of a value at a point in `H^2`. The app writes
no adjoint at all. `lift` turns the tracer's nodes into the space's points,
`(radius, angle)` or `(radius, latitude, longitude)`.

A dense evaluation of every basis function at every node would be
gigabytes at the full scale, where `route="auto"` evaluates locally
instead; the composition then applies in a fifth of a second, against a
millisecond for the matrix of 275 rows that `path_integral_operator` forms a
ray at a time, and assembling the data-space normal operator applies the
forward operator once per datum. So the operator the app uses is the rows:

```python
for index, ray in enumerate(rays):
    rows[index] = ray.weights @ space.basis_matrix(lift(scale, ray.nodes))
return LinearOperator.from_matrix(
    space, EuclideanSpace(len(rays)), rows, form="galerkin"
)
```

Each row is the derivative components of one path integral (tutorial 2).
Into a Euclidean space the Galerkin and the component matrix are the same
matrix, the codomain's Gram matrix being the identity, so either
declaration is right; what would be wrong is a row of representers declared
as derivative components, the confusion of tutorial 3. The property
operator is built the same way: `box_average_row` integrates the basis over
a box by Gauss-Legendre under `r dr dangle` or `r² cos(latitude)`, and
`core_average_row` takes the core layer's `integral_functional()`.

**What was checked**, in `pygeoinf2/tests/test_miniapp_tomography.py`:

- the tracer against the closed form of a flattened linear gradient, and
  its weights converging at second order (README, "The ray tracer");
- the row-by-row operator against the composition written as it reads, to
  `1e-12`, in the disk and in the ball;
- `check_operator(forward, rng=rng, measure=prior, rtol=1e-12)`, a defect of
  `1.8e-15` at the full scale: the adjoint identity in the layered metric,
  with probes drawn from the prior rather than from white noise;
- the operator on the reference slowness against the tracer's own travel
  times, and on the constant one against the path lengths.

The first two fail on a wrong node, angle or unit in `lift`; the third on
rows that are not derivative components; the last on weights that do not
integrate along the right paths.

**In the ball** (part F) a ray lies in the plane of its ends and the
center, so `ball.trace` runs the disk's tracer and places each node on the
great circle at its traced radius; nothing else in the wrapping changes.

**Bent rays** (part G, `bent.py`) make the forward nonlinear: the travel time
through the perturbed slowness, along the ray re-traced in it by shooting
(`bend` in `src/rays.py`). By Fermat's principle the travel time is stationary in the
path, so its derivative at a model is the path integral along that model's
own rays: the linear operator above, rebuilt on the current rays. The
library's protocol for a code that produces the value and what the
derivative needs in one run is `linearize` (tutorial 9):

```python
self.operator = Operator.from_callables(
    space, EuclideanSpace(len(pairs)),
    lambda model: self.trace(model)[0],
    linearize=self._linearize,
)
```

`_linearize` returns a `Linearization` of the model, the values and
`path_integral_operator` on the traced rays. `BentForward.trace` remembers
the last model, since an optimizer asks for the value, the derivative and
the Gauss-Newton Hessian at one point in turn, and raises `EvaluationError`
when a ray cannot be traced, from which the line searches retreat. No second
derivative is supplied; the estimator asks for the Gauss-Newton Hessian.

Its check is `check_derivative` at a ladder of steps, `rtol=1.0` so that the
defect is read rather than judged:

```python
check_derivative(problem.forward.operator, point, rng=np.random.default_rng(seed),
                 measure=prior, step=step, trials=1, rtol=1.0)
```

The defect falls as the square of the step, which says the derivative is
right, down to a floor of `4e-6` to `2.5e-4`, which says how far it is from
the derivative of the computed map. The floor is the integrator's: the path
integral is the derivative of the exact travel time, the Runge-Kutta time a
discretization of it. As with the continuous adjoint of tutorial 12, a
derivative consistent to discretization level is checked by its fall, not
by one tolerance.

## 3. The priors and the error measures

The prior is a Sobolev measure of order 2 on each layer, the layers
independent: the covariance `a^2 A^-2` of each layer's own operator `A`,
trace class on `H^2` because the order exceeds half the dimension (one in
the disk, three halves in the ball), so that its draws are functions of the
space with nearly `2 - d/2` derivatives more than it asks. It is the layered
space's own `Layered.sobolev_measure`, which refuses an order at or below
half the dimension, built in `sobolev_prior` in `src/problem.py`:

```python
return space.sobolev_measure(
    order, amplitude=layer_amplitudes(space, order=order, std=std)
)
```

**The calibration** is by an amplitude per layer, `layer_amplitudes`, the
one that brings the layer's mean pointwise standard deviation over its
interior to one percent; the result is uniform to within two per cent.
Calibrating with `pointwise_std=` instead would rescale the measure from
place to place, `S C S*` with `S` a multiplication, exactly uniform but
neither diagonal nor with a precision, so that the prior's norm would go
through a solver and its spectrum through dense matrices or draws, where
here they are diagonal and free. The measure is the product of the layers'
measures, put by the library on the layered space itself, so that its draws
and its posteriors are vectors of that space, with the pointwise product a
pointwise standard deviation needs. Because the covariance is diagonal in
each layer's basis, the prior's pointwise standard deviation is exact and
cheap, `prior_pointwise_std`. The
truth and the data come from `problem.synthetic_model_and_data(prior,
rng=rng)` with Gaussian noise of 0.05; part G uses the same construction at
five times the width, so that the bending of its smooth draws matters.

**A mixture prior** (part E, `mixture.py`) puts a discrete question in the
continuous problem: a plume rising at one of five angles.

```python
GaussianMixture(
    [problem.prior.translate(plume(problem, axis)) for axis in angles]
)
```

Each component is the app's prior translated to the plume at a candidate
angle, all with the same covariance, the weights equal. The plume is a
conduit, `exp(-d^2 / (2 w^2))` scaled to its peak, with `d` the distance
from the point to the radial segment at the candidate's angle between the
core-mantle boundary and the surface (`distance_to_axis`): Gaussian across
the segment and flat along it. The layered space takes it as a function of
position, one piece a layer, nothing in the core:

```python
space.project_function([lambda point: 0.0, value], extension="constant")
```

`extension="constant"` keeps the function's values on the mantle's grid and
continues them at their end values across the padding. The conduit runs at
full height to both edges of the mantle, and the default, the best fit over
the domain in the kept modes, would leak into the padding at the full
scale, with values there in the thousands; the library warns when it does.

**Hardening to sets** (part D, `sets.py`):
`prior.ambient_ball(level=0.95)` and the noise's `ambient_ball`, which for
an isotropic error on a Euclidean data space is its credible ellipsoid too.
The prior's set is the ball `{m : ||m|| <= r}` in the space's own `H^2`
norm, with `r` the radius at which a draw from the prior lies inside with
probability 0.95, a quantile of `sum lambda_i Z_i^2` over the covariance's
spectrum: the classical Backus-Gilbert-Parker norm bound, the prior used
only to calibrate it. The expectation of that sum is the covariance's
trace, finite because the covariance is trace class on `H^2`, so the
radius has a limit however fine the truncation, in the disk and in the
ball alike. The Mahalanobis ellipsoid, `prior.credible_set`, has none: its
radius is a chi-squared quantile on as many degrees of freedom as there are
coefficients, which is why the library refuses it on a function space. The
covariance being diagonal, the radius is an exact quantile read off its
diagonal with no applications, and the ball's support function is the
radius times the norm of the representer. The ball forgets the covariance's
shape: it admits the model that spends its whole budget on the core's
mean, which is why the core's bracket is the ball's own `±4.03` less what
the mantle must spend to fit the data. The test checks that bracket's
inner end attains its outer one: the smallest fitting model plus as much
of the core mean's representer as the ball still holds. The ball is a
convenience, one constraint set among many; a bound the physics suggests,
on another norm, a region or one layer more than the other, is a convex
set the same estimator takes.

The test checks the sampled pointwise standard deviation against the exact
one at four points: the measure check this app needs.

## 4. The solvers

A few hundred data and a few thousand coefficients: the data-space normal
operator `A Q A* + R` is 275 square at the full scale, and every damped
`A C A* + t R` the point and set answers sweep is the same size. Every
estimator is handed `solver=CholeskySolver()`, which assembles the operator,
one application per datum, and factorizes it. The posterior mean costs one
factorization, a fifth of a second; the discrepancy search one per damping
it tries, nine in all; the mixture one per component and per noise level.
There is no preconditioner, because there is no iteration.

The posterior samples, the support directions and the mixture's components
take `n_jobs=` and run serially here: at this size workers cost more than
they save (README).

Part G's MAP model is the only iterative solve: `NewtonCG` on the
Gauss-Newton Hessian, preconditioned by the prior covariance so that the
inner solves see only the data's spectrum (tutorial 14); 6 iterations and 7
ray tracings at the full size.

## 5. The estimators

**The point** (parts B and F, `point.py`). Tutorial 6's estimators with the
error budget of tutorial 14, with size measured in the prior's norm
(example 23): the problem is parameterized by the prior's covariance
factor, `m = L w`, so that `|w|` is the prior's norm of `m`:

```python
whitened = problem.problem.parameterized(problem.prior.covariance_factor)
estimator = MinimumNorm(whitened, level=level, solver=CholeskySolver(), rtol=rtol)
chosen = estimator.for_data(problem.data)
pushed = chosen.push_forward(problem.properties @ problem.prior.covariance_factor)
```

The space's own `H^2` norm is the other honest choice. The prior's is taken
because the point answers then damp towards what the Bayesian answer
assumes: least squares at a damping of one is the posterior mean exactly,
which is the guess of `GUESSED_DAMPING`, and the discrepancy principle's
model is the smallest in the norm the prior calls typical. `pushed(data)` is
the estimate of the three means, `pushed.propagated_covariance()` the noise
it carries, and `pushed.bias(whiten(problem, truth))` the error only the
truth reveals, with `whiten` the factor's inverse, `L* Q^-1`, exact through
the prior's precision. `chosen.push_forward(factor).resolution()(...)` of a
bump's `w` is what the estimate makes of a unit bump in each box. In the
core the estimate is zero and so is its propagated noise: its whole error
is bias, which nothing in the point answer shows. Figure `point.png` draws
the budget per mean.

**The measure** (parts C and F, `bayesian.py`).
`LinearGaussianInversion(problem, prior, solver=CholeskySolver())(data)` is
the posterior. The exact pointwise standard deviation at a few points is
its `affine_map` through `space.point_evaluation_operator(points)`; the map
is `sample_pointwise_std(samples, rng=rng)`; the means' posterior is
`affine_map(problem.properties)`. Figure `posterior.png` sets the prior's
map beside the posterior's, and each mean's marginal before and after.

**The mixture** (part E). `LinearGaussianMixtureInversion(forward, prior,
solver=CholeskySolver(), n_jobs=scale.n_jobs)` updates each component by its
own linear Gaussian inversion, on the scale's workers, and rescores the
weights by the components' evidences:

- `estimator.weights(data)`, the posterior weights, which are evidence
  ratios times the prior's;
- `estimator.log_evidence_terms(data)`, each component's log evidence;
- `estimator(data)`, the posterior `GaussianMixture`, and
  `estimator.push_forward(properties)(data)`, the mixture on the means;
- `estimator.inversions`, each component's `LinearGaussianInversion`, whose
  `mahalanobis(data)` is the misfit `M_k` of the README.

The noise sweep rebuilds the estimator at each noise level from
`LinearForwardProblem(problem.forward, error=noise)`. Figures
`mixture_*.png` draw the candidate conduits, the weights, the marginals as
the mixtures' own densities (`mixture_marginal` in `src/common.py`, exact)
and the winner's mean.

**The set** (parts D and F, `sets.py`). `BackusGilbertParker(forward,
properties, ball, noise=noise_ball, solver=CholeskySolver())(data)`, a
`support_sweep` in the six coordinate directions, each result a certified
`value` with an attained `inner`, and `feasible.polytope(directions)` for
the truth's membership. Figure `brackets.png` draws each bracket against
the ball's own support along the property's representer.

**Bent rays** (part G). `MaximumAPosteriori(problem, prior,
optimizer=optimizer, gauss_newton=True).solve(data)` from the prior's mean;
its result carries the iterations, the message and the objective's history.
It is compared with `LinearGaussianInversion` on the reference rays;
`ForwardProblem.chi_squared` along each model's own rays shows the linear
answer inconsistent with its physics. Figure `bent.png` draws both.

## 6. What a reader would change

- **The tracer**: replace `src/rays.py` by any code that returns, per ray,
  nodes and quadrature weights; `lift` maps its nodes to the space's
  points. Check it against a closed form and the operator against the
  composition, then `check_operator` with `measure=prior`.
- **The space**: `model_space` in `src/problem.py`, its boundaries,
  truncations and length scale; a disk or a ball by one flag. The order,
  the truncation, the length scale, the prior's width, the noise and the
  network are the scale's, and each is a command-line option.
- **The prior and the properties**: `sobolev_prior` is a measure of order
  above half the dimension on each layer, calibrated by an amplitude so that
  it stays diagonal; the order, the width and the set's level are options
  (`--prior-order`, `--prior-std`, `--level`), and the prior's set is one
  line in `sets.py` to replace by any convex set with a support function.
  `box_average_row`
  and `core_average_row` are any linear functionals given by derivative
  components.
- **A nonlinear forward**: a trace that raises `EvaluationError` when it
  fails, and a `linearize` returning the linear operator rebuilt on the
  current rays, as `BentForward` does; check its derivative at a ladder of
  steps.
