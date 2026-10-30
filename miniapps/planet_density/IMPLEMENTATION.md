# A planet's density: how it uses the library

The [README](README.md) says what the app asks of two numbers and what they
answer. This document walks through how it is built on `pygeoinf`, and why.
It assumes the tutorials under `tutorials/`, cited by number;
tutorials 10 and 11 build the same kind of planet on a toy. Code is quoted
from the app's own files.

## 1. The spaces

The problem is posed on a space appropriate to the forward model and the
answers, and the choice is a modeling decision, stated here as one: PREM's
layers, each `H^2` under `r² dr`. The mass and the moment of inertia are
bounded on it, as they would be on far rougher spaces, so boundedness
narrows the choice without making it; a value at a point exists in every
layer, the center's included, which the bands of the figures need; and its
norm is the one Occam's first model of part D is measured in (the point
answers of parts B and C are smallest in the prior's precision, a second
norm, chosen as such, and the set answers harden the prior to the ball of
the space's norm that carries 95% of it). The priors are measures of extra regularity on that
space, Sobolev measures of order two whose covariance is trace class on
it; they do not move the problem to a smoother space, and nothing is done
relative to `L^2`.

The unknown is a density profile, a function of radius that is smooth inside
each of PREM's layers and may jump between them. That is
`pygeoinf.interval.radial.Layered`: a direct sum of radial Sobolev spaces,
one per layer, each under the volume measure `r² dr` and each held by the
leading eigenfunctions of `1 - d/dr (L² d/dr)` on a padded spectral-element
mesh (tutorial 11). From `src/problem.py`:

```python
space = Layered(
    radii,
    list(scale.basis_functions),
    order=scale.order,
    length_scale=scales,
    labels=names,
)
```

with `radii` the layers' boundaries, read off PREM by `layering()` in
`src/reference.py`, and `scales` the scale's `length_scale` of 200 km, the
crust's excepted, both converted to Earth radii, and the scale's `order` 2.
Every field of the scale is a command-line option, a length given in
kilometers.

**The units** are radius in Earth radii, density in g/cm³, and the data in
`10^24 kg` and `10^37 kg m²`. The inner product is a volume integral, so
the unit of radius enters every norm through its cube; on a planet of
radius one the norms are of order one, as the densities and the data are
(PREM's norm on the space is 20.3), and the library's absolute floors, its
solvers' tolerances among them, meet numbers of the size they assume.
Lengths given in kilometers, at the command line and in the figures, are
converted where they meet the space.

What each choice buys:

- **The layers.** Every interface of PREM is an interface of the space, so a
  profile may jump exactly where PREM's may and nowhere else. The layers
  thinner than 20 km are merged into one crust, too thin otherwise to carry a
  few basis functions each.
- **The order.** A point value needs order above one half away from the
  center and above three halves at it (tutorial 11,
  `point_evaluation_order`). `H^2` clears both, and the bands of the figures
  are the covariance pushed through point evaluation.
- **The measure `r² dr`.** A shell near the surface holds far more mass than
  one near the center, and the inner product knows it: two integrals can pin
  the mantle and leave the inner core almost as free as the prior had it.
- **The crust's length scale**, a quarter of the crust's thickness. The mesh
  is padded at each end by a length scale or two (tutorial 11), and the
  short length scale keeps the padding no wider than the crust.
- **The basis functions per layer**, `Scale.basis_functions`: 104 in all at
  the full scale, most in the core and the lower mantle.

The data space is `EuclideanSpace(2)`, mass and moment in the units above;
the property space is `EuclideanSpace(3)`.

The reference profile is PREM projected into the space layer by layer:

```python
layer.project_function(
    lambda r, prem=prem: prem(R_SURFACE * r),
    extension="constant" if name == "crust" else "fit",
)
```

with `prem` PREM's density in the layer as a function of the radius in km,
read at `R_SURFACE r` for the space's `r`.

`"fit"` is the best fit over the layer's domain in its basis functions,
right for a smooth profile; `"constant"` samples at the nodes and continues
the end values across the padding, which keeps the crust's two internal
jumps out of it. `space.truncate` settles the tuple into the span.

Part D adds the two boundary displacements as unknowns:

```python
model_space = DirectSum([space, EuclideanSpace(2)], labels=("density", "radii"))
```

A direct sum's inner product is the sum of its parts' (tutorial 10), so the
unit of the displacements, the scale's `radius_scale`, 50 km, against the
space's unit of radius decides how a kilometer of boundary weighs against
density in the model space's norm (section 5).

## 2. The forward code wrapped as an operator

The forward code is a quadrature. Mass and moment of inertia are volume
integrals of the density, linear in it, and `volume_integral` computes the
derivative components of `rho -> int weight(r) rho dV`: five-point
Gauss-Legendre between adjacent mesh nodes of each layer, where every basis
function is a polynomial, against the basis values from
`space.basis_matrix(radii)`. The two rows become the operator:

```python
forward = LinearOperator.from_matrix(
    space, data_space, np.vstack([mass_row, moment_row]), form="components"
)
```

The rows are derivative components, the `g` with `I(rho) = g . c_rho` of
tutorial 2. As a matrix of an operator into a Euclidean space they are both
the component matrix and the Galerkin matrix, since the codomain's Gram
matrix is the identity; the tomography app declares the same kind of rows
`form="galerkin"` and means the same operator. The metric enters in the
adjoint, which the library forms as the inverse Gram of the layered space
applied to the rows' transpose: the representer of the mass functional is a
function, not a list of quadrature weights (tutorial 3).

The property operator is built the same way: the volume row restricted to
`space.columns(name)` over the volume for the two means, and the row of
`space.jump_operator(index)` for the jump.

**What was checked.** The test puts `volume_integral(space, np.ones_like)`
against the library's `space.integral_functional().derivative_components` to
`1e-10`, and the mass and moment of a uniform density against their closed
forms: what a hand-written quadrature row needs. A `check_operator` on a
`from_matrix` operator would only confirm the library's own algebra.

**Part D, a nonlinear forward** (`src/radii.py`). Moving a boundary out by
`delta` turns a shell from the material above it into the material below, so
to first order the data gain a term bilinear in the density's jump and the
displacement. The app writes it as an `Operator` with both derivatives:

```python
operator = Operator.from_callables(
    model_space, data_space, value,
    derivative=derivative, second_derivative=second_derivative,
)
```

`derivative(x)` returns `F'(x)` and `second_derivative(x, direction)`
returns `F''(x)[direction, .]`, each a matrix on the direct sum's components
(the density's, then the two displacements) passed to
`LinearOperator.from_matrix(..., form="components")`. The operator is
bilinear, so the second derivative does not depend on `x`.

`check` in `src/radii.py` runs three checks at three draws from the prior, each drawing
its probe directions from a measure passed as `measure=`:

```python
testing.check_operator(operator.derivative(point), rng=rng, measure=probes)
testing.check_derivative(operator, point, rng=rng, measure=probes, step=1.0)
testing.check_second_derivative(operator, point, rng=rng, measure=probes, step=1.0)
```

The probes are the prior less its expectation,
`prior.translate(model_space.negative(prior.expectation))`. PREM has a norm
of 20 on the space, where a centered draw's density part has a norm of
about a half and each displacement a standard deviation of one, so a
normalized draw from the prior itself would point along PREM every time,
with the displacements a small part of it. Centered, the two parts are of
the same order, and each direction moves the density and both boundaries.
The differences take a step of one: the operator is bilinear, central
differences are exact for it, and the only error is rounding, which a
large step keeps small. The checks judge each defect relative to the size
of the derivative itself, so a derivative the difference cannot resolve
shows up as a defect rather than hiding behind the operator's values. The
worst defect at the full scale is `1.9e-13`. A failure of the first
would mean the matrix declared in the wrong form on a space whose Gram
matrix is not the identity, and every gradient of part D wrong with it; of
the second, a derivative that is not the value's, a sign of the jump or the
displacement's unit missing from one of them; of the third, the cross term,
which couples one direction's jump with the other's displacement, assigned
the wrong way round.

## 3. The priors and the error measures

Both priors are Sobolev measures about the reference, built by one function:

```python
space.sobolev_measure(
    order, expectation=reference, pointwise_std=std
).with_dense_covariance()
```

On a layered space the measure is a product over the layers, independent
between them, which is what a discontinuity at every interface means
(tutorial 10). The order, the scale's `measure_order` 2, is above the
floor of one half the library enforces: a radial profile's spectrum grows
as on a line, so the covariance is trace class on `H^2` above that, and
the draws have nearly one and a half derivatives more than the space asks. `pointwise_std` is calibrated in each layer separately, since
a space whose structure varies has no single raw variance (tutorial 11): 0.1
g/cm³ for the tight prior of part B, 3 for the wide one of part C.
`with_dense_covariance()` assembles the covariance once as a 104 by 104
matrix and factorizes it, which gives the measure the covariance factor and
the precision factor the point answer and Occam's prior norm use.

The noise is independent, 0.1% of each datum,
`GaussianMeasure.from_standard_deviations(data_space, scale.relative_error * clean)`.

Part D's prior is the product of part B's on the density and a standard
normal on the displacements, a unit of 50 km being one standard deviation:

```python
GaussianMeasure.from_product(
    [problem.prior(scale.tight_std),
     GaussianMeasure.from_standard_deviation(EuclideanSpace(2), 1.0)],
    labels=("density", "radii"),
)
```

**Hardening to sets** (`set_answer` in `src/answers.py`):
`prior.ambient_ball(level=level)` and the error measure's
`credible_set(level=level)`, its own Mahalanobis ellipse on the two data:
the errors are independent but of two sizes, so a ball would not have their
shape, and on a Euclidean data space the credible set is what a Gaussian
error's 95% region is. The prior's set is the ball `{m : ||m - m0|| <= r}`
in the space's own norm about the prior's expectation, PREM, with `r` the
radius at which a draw from the prior lies inside with probability 0.95, a
quantile of `sum lambda_i Z_i^2` over the covariance's spectrum, here from
a dense generalized eigenproblem of the 104 by 104 covariance against the
Gram matrix. It is the classical Backus-Gilbert-Parker norm bound with the
prior used only to calibrate it. The expectation of that sum is the
covariance's trace, finite for a trace-class covariance, so the radius has
a limit however many basis functions there are. The Mahalanobis ellipsoid,
`prior.credible_set`, has none: its radius is a chi-squared quantile on as
many degrees of freedom as there are basis functions, which is why the
library refuses it on a function space. The ball forgets the covariance's
shape, and is a weaker statement than the Gaussian: its brackets are wider
than the posterior's intervals wherever the data leave a direction free.
It is a convenience, one constraint set among many; bounds from mineral
physics on each layer, or a norm that weights the core more than the
mantle, are convex sets the same estimator takes. No measure is checked;
`testing.check_measure` (tutorial 5) is the check for a prior built by
hand.

## 4. The solvers

There is no solver story, and that is the lesson of a small data space.
The data-space normal operator `A Q A* + R` of every linear answer is two by
two, and the estimators' default, conjugate gradients, needs at most two
iterations on it; the point answer's damping search is 12 such solves. The
model space has 104 dimensions, small enough to assemble the prior dense
and factorize any model-space operator.

Part D's solves are in the model space:

- the MAP model by `NewtonCG`, its inner conjugate gradients preconditioned
  by the prior covariance, which leaves them only the data's spectrum
  (tutorial 14);
- the Laplace covariance, the inverse of `P + J* R^-1 J` on 106 dimensions,
  by `CholeskySolver()`, passed to `LaplaceApproximation`;
- Occam's model by the same Newton-CG inside an augmented Lagrangian.

## 5. The estimators

`three_answers` in `src/answers.py` runs the point, the measure and the set, in that
order, under the prior parts B and C hand it.

**The point** (`point_answer`). The smallest change to PREM that
fits the data, measured in the prior's precision. The change is written
`L w`, with `L` the prior's covariance factor, and `MinimumNorm` is run on
the problem parameterized by `L`, against the data less the reference's
prediction:

```python
estimator = MinimumNorm(problem.problem.parameterized(factor), level=level)
search = estimator.discrepancy_search(residual)
chosen = estimator.for_data(residual)
```

`LinearForwardProblem.parameterized(L)` is the problem `A L` on `L`'s domain
with the same error. The smallest `w` is the smallest change in the prior's
Mahalanobis norm, so the estimator needs the error measure and the prior's
shape, not its size: the wide prior's point answer is the tight one's. The
error budget comes from `chosen.push_forward(properties @ factor)`, its
`propagated_covariance()` and its `bias(departure)`, with the truth's
departure in `w` read off by `prior.precision_factor`. When the reference
already fits the data the search reports `exhausted == "high"` and the
smallest change is none, as another seed can make it (README). Figures
`*_point.png` draw the estimate, its propagated noise and the truth plus the
bias.

**The measure** (`measure_answer`). `LinearGaussianInversion` gives
the posterior; `posterior.push_forward(properties)` the Gaussian on the
three properties, exactly; `inversion.log_evidence(data)` the evidence of
tutorial 7, an `Estimate` whose `value` is exact here, the data space being
assembled, and whose difference between the tight and the wide prior is
part C's Bayes factor. The bands on the profile are the covariance pushed through
point evaluation at radii inside each layer, never on an interface
(`pointwise_std` in `src/common.py`). Figures `*_measure.png` set the bands of prior and
posterior side by side and the property marginals in a corner plot,
`plotting.plot_corner(after, prior=before, truth=...)`.

**The set** (`set_answer`). `BackusGilbertParker(problem, properties,
ball, noise=noise)(data)` and `support_sweep` over the six coordinate
directions: each result's `value` is a certified outer bound, its `inner`
attained by a feasible model. Figures `*_set.png` draw the brackets against
the ball's own support along each property's representer.

**Part D** (`boundaries.py`) answers with the nonlinear estimators:

```python
estimator = MaximumAPosteriori(
    forward, prior, optimizer=newton(preconditioner=prior.covariance),
    gauss_newton=True,
)
laplace = LaplaceApproximation(estimator, solver=CholeskySolver())
gaussian = laplace.at(map_model, data).push_forward(readout)
```

`MaximumAPosteriori` needs the forward problem and the prior; with
`gauss_newton=True` it carries the Gauss-Newton Hessian `2 (P + J* R^-1 J)`
in place of the exact one it would build from the second derivative.
`LaplaceApproximation.at` places the Gaussian on the mode already found
(tutorial 9), and `readout` takes it to the three properties and the two
displacements in km.

`Occam` is run twice from the same reference to the same level, which is
where the direct sum's norm becomes the answer:

```python
to_model = AffineOperator(prior.covariance_factor, reference)
ForwardProblem(forward.forward_operator @ to_model, error=forward.error)
```

The first run is on the problem itself, smallest in the model space's norm;
the second on the problem composed with `m = m0 + L w`, smallest in `w`,
which is the prior's norm. The results are mapped back to the model space
and their changes from PREM measured both ways. In the model space's norm a
displacement of 50 km has norm one, where a density change with the same
effect on the mass has norm 0.079 (README), so Occam leaves the boundaries
where PREM has them and changes the density; in the prior's norm both parts
are measured in their own standard deviations, and it lands near the MAP
model. Nothing in the library chooses between them: a direct sum's norm is
the caller's choice, and Occam reports it. Figure `boundaries.png` draws
the three models.

## 6. What a reader would change

- **The layers**: `layering()` in `src/reference.py` and the scale's
  `basis_functions` (`--basis-functions` on the command line), with a
  length scale per layer; a thin layer wants a short one.
- **The reference**: one function per layer, projected with `"fit"` where it
  is smooth and `"constant"` where it holds jumps of its own.
- **The data**: one row per datum from `volume_integral(space, weight)`, or
  any other derivative components of a linear functional of the profile,
  stacked into `from_matrix`; check each row against
  `integral_functional()` or a closed form, and choose units in which the
  data and the space's norms are of order one: the radius in the planet's
  own.
- **Further unknowns**: a `DirectSum` and an `Operator.from_callables` with
  its derivatives, as in `boundary_operator` in `src/radii.py`, put through
  its `check` before use. Their unit is part of the model space's norm,
  and with it of every minimum-norm answer.
