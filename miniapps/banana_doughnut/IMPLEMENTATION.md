# Ray and finite-frequency traveltimes: how they use the library

The [README](README.md) says what is compared and what the comparison
shows. This document walks through how the app is built on `pygeoinf`: the
model space, the priors with scales of their own and their calibration,
the two functionals as point evaluation composed with weights, the stacked
operators and the push-forwards, the representers, and the pictures drawn
through `planetmodel`. It assumes the tutorials under `tutorials/`, cited
by number. Code is quoted from the app's own files.

## 1. The space

The model is `m = δln c` on the mantle annulus, `[0.55, 1]`, and the space
is chosen for the functionals: both are point evaluation followed by
weights, bounded wherever a value at a point exists, so the order must
exceed three halves in three dimensions (tutorials 2 and 5); `H^2` clears
it. The space is the problem's and never changes: no prior rebuilds it.
From `model_space` in `src/problem.py`:

```python
return ball.Sobolev(
    scale.lmax,
    scale.order,
    scale.length_scale,
    radius=RADIUS,
    inner_radius=CORE_RADIUS,
    padding=scale.padding,
)
```

`pygeoinf.ball.Sobolev` holds spherical harmonics to degree `lmax` times
the radial eigenfunctions of `1 - div(L^2 grad)` on a spectral-element
mesh of the radius (tutorial 11), its inner product `(A^2 m, m)` with `A =
1 - L^2 Laplacian`. With `inner_radius` it is an annulus: the core is not
part of the domain, so nothing is ever computed there. Its points are
`(radius, latitude, longitude)` in degrees.

- **The length scale**, 0.02, is the one the Sobolev gradients and the
  space's own dual norms are measured in. It is short on purpose: a
  prior's covariance is an operator in this metric, so the space's scale
  is the shortest over which a draw can decorrelate, and it sits below
  every Fresnel half-width and every prior of interest.
- **The truncation**, degree 160 at the full scale, resolves about `pi /
  160 = 0.02` at the surface; 659 316 coefficients.
- **The padding** of the radial mesh beyond each end of the annulus is a
  field of the scale, 0.32. A prior much longer than the padding feels
  the mesh's ends: with the space's default, two of its length scales,
  the longest prior's delay standard deviation is 6 per cent off, and at
  0.32 it is within 1.2 per cent of its value at 0.64.

The delays live in `EuclideanSpace(k)`, `k` the functionals stacked.

## 2. The priors and their calibration

A prior is a Sobolev measure of order 2 on that one space with length
scales of its own, along a radius and across one, which the space builds
in its own basis, block by degree: the covariance `a^2 (1 - div(L_r^2
grad) - L_h^2 Laplacian_h)^-2` as an operator in the space's metric,
trace class because the order exceeds half the dimension (tutorial 5).
From `sobolev_prior` in `src/problem.py`:

```python
return space.sobolev_measure(
    float(order),
    length_scale=float(radial),
    lateral_length_scale=float(lateral),
    pointwise_std=float(std),
)
```

`pointwise_std=` is the calibration: the uncalibrated pointwise variance
is a function of radius alone (`space.sobolev_pointwise_variance`, which
`raw_pointwise_std` reads to report how far from uniform it was), and the
measure is rescaled by the radial profile that brings it to `std^2`
everywhere. At degree 160 that profile is uniform to 0.2 per cent for the
isotropic priors up to 0.16, and varies from 0.98 to 1.01 of its average
for 0.32 and from 0.97 to 1.07 for the pair 0.03 × 0.5, which an amplitude
alone would leave. That rescaling commutes with rotations, so correlations along
a shell are untouched. The calibrated measure has a factor and a
covariance but no cheap precision, which nothing here needs: representers
`C g`, push-forwards `B C B*`, draws and the space's dual norms all go
through the factor.

**What a prior's draws do** is read off its covariance, not off its
nominal lengths, because the metric enters the draws too. The covariance
applied to the representer of evaluation at a point `x0` is the field
`c(x0, x)`, and divided by its value at `x0` it is the correlation from
there. From `correlation` in `src/correlations.py`:

```python
dirac = space.point_evaluation_operator([point])
field = prior.covariance(dirac.adjoint(dirac.codomain.basis_vector(0)))
...
values = space.evaluate(field, shell_points + radial_points)
```

one application of the covariance and one evaluation at every offset;
`Correlation.shell_distance` and `radial_distance` are where it first
falls to `1/e`. `Problem.prior(radial, lateral)` makes any other prior on
the same space and keeps it, and `Problem.correlation(key, index)` keeps
each prior's correlation from each ray's turning point.

The checks, in `tests/test_miniapp_banana_doughnut.py`: each prior through
`check_measure` (draws against the covariance) and `check_traits` on its
covariance, with the prior itself as the probe measure; the pointwise
standard deviation `E C E*` at sixty points against the scale's.

## 3. The functionals: point evaluation composed with weights

Both functionals are the same construction: point evaluation of the model
at a set of points, then a row of weights. For ray theory the points are
the tracer's nodes and the weights minus its weights over the travel time,
`dT = -int m / c dl`; for finite-frequency theory the points are the
quadrature's in ray-centered coordinates, about two hundred thousand of
them, and the weights its products of the detour-time, angle and arc-length
weights with `-G(u) / (2 pi c)`. `composed_operator` in
`src/functionals.py` writes the composition as it reads, one sparse row a
functional:

```python
matrix = csr_matrix((data, (rows, columns)), shape=(len(pieces), sum(counts)))
evaluation = space.point_evaluation_operator(points)
weights = LinearOperator.from_matrix(
    evaluation.codomain, EuclideanSpace(len(pieces)), matrix, form="components"
)
return weights @ evaluation
```

The adjoint is the library's, `evaluation.adjoint @ weights.adjoint`, and
the metric enters through the adjoint of point evaluation, the
representer of a value at a point in `H^2`; the app writes no adjoint. A
point is `lift`ed from the tracer's ray plane, the equatorial plane, to
`(radius, latitude, longitude)`. A finite-frequency point above the free
surface is moved to its mirror image before it is lifted, and one below
the core-mantle boundary is dropped with its weight.

**The rows.** Applying the composition evaluates the model at every
point, and its adjoint sums the representers of evaluation at every point:
about ten seconds for one finite-frequency functional at the full scale.
The app applies the adjoint once per functional and keeps the result, the
functional's derivative components (tutorial 3, section 4), from which
every later application is a dot product. From `Problem.rows` in `src/problem.py`:

```python
operator = functionals.composed_operator(self.space, [self.piece(*key)])
return operator.matrix(form="galerkin")[0]
```

`matrix(form="galerkin")` of an operator onto `R^1` is one application of
the adjoint. The rows are formed in threads (`n_jobs=`) and kept on the
problem, so that each part reuses the rows another formed. An operator of
several functionals is then

```python
LinearOperator.from_matrix(
    self.space, EuclideanSpace(len(keys)), self.rows(keys), form="galerkin"
)
```

`ray_operator`, `finite_frequency_operator(band)`, `pair_operator(index,
band)` (the two functionals of one ray in one band, onto `R^2`) and
`stacked_operator()` (every functional of the scale, ray theory's of
every ray, then finite-frequency theory's of every ray band by band) are
all this, with `assembled=False` giving the composition instead. Into a
Euclidean space the Galerkin and the component matrix are the same, the
codomain's Gram matrix being the identity.

**What was checked**: `check_operator(operator, measure=prior,
rtol=1e-12)` on both functionals, composed and assembled, with probes
drawn from a prior rather than white noise (worst defect 1.5e-13); the
composed and assembled matrices against each other; on a model constant
across every banana the finite-frequency functional against the ray's
times the quadrature's Gaussian identity; and the full scale's largest
quadrature through the operator check on the quick space.

## 4. The representers

A functional's representer in the space's own inner product, its Sobolev
gradient `g` with `(g, m) = l(m)` for every model, is the Riesz map of its
row, the derivative components (tutorial 3, section 3, and tutorial 13 for
a ray's). From `Problem.representer`:

```python
return self.space.representer(self.rows([(index, band)])[0])
```

Under a prior it is `C g`, the representer in the prior's Cameron-Martin
space and the covariance of the field with the delay: `prior.covariance(g)`
in part B, one application of the covariance a functional and prior. The
delay's standard deviation is `sqrt((C g, g))`, `space.inner_product`.
The space's own dual norm of `l_FF - l_ray` against `l_ray`'s is the norm
of the difference of the two gradients, `space.norm(space.subtract(...))`
in `dual_norm_ratio`. The test checks the gradient's inner product with a
draw against the functional applied to it.

## 5. The push-forwards

**Part C.** A prior pushed forward through the pair operator of one ray in
one band is a Gaussian on `R^2`, the joint law of the two delays. From
`pushed_covariance` in `agreement.py`:

```python
pushed = problem.prior(*key).push_forward(problem.pair_operator(index, band))
return np.asarray(pushed.covariance.matrix(form="components"))
```

The covariance's two columns are two applications of the prior's factor
and its adjoint through the two rows. The standard deviations, the
correlation, the regression slope and the conditional standard deviation
are read off that 2-by-2 matrix. The push-forwards of every ray, band and
prior run in threads.

**Part D.** Each theory's law of the delays of every ray in one band is
the prior pushed forward through that theory's stacked operator, a
Gaussian on `R^N`. All of them are marginals of one push-forward, through
every functional at once, whose covariance is assembled once a prior. From
`joint_law` and `laws` in `pushforward.py`:

```python
pushed = problem.prior(*key).push_forward(problem.stacked_operator())
joint = pushed.with_dense_covariance()
ray = joint.push_forward(projection(total, 0, count))
finite = joint.push_forward(projection(total, count * (1 + band), count))
```

`with_dense_covariance` assembles the `N (1 + B)`-square covariance, one
application of the factor and its adjoint a functional, and factorizes it;
`projection` is a coordinate projection of `R^(N (1 + B))`, a
`LinearOperator.from_matrix` in component form, so that each theory's law
is the push-forward of the joint law through it, which is the push-forward
of the prior through the composition, that theory's stacked operator (the
test checks it against `prior.push_forward(problem.ray_operator())` and
the finite-frequency one to 1e-16). Where the two theories' delays nearly
coincide, in the narrow band, the joint covariance is singular to
rounding, and the factorization clips the rounding's negative
eigenvalues, which the library warns of; `joint_law` filters that one
warning. The divergences are the library's, on Gaussians on `R^N`, where
the dense route is exact:

```python
finite.kl_divergence(ray), ray.kl_divergence(finite)
```

The draws go through both operators at once: a draw of the model from the
prior, each from its own generator spawned from the seed so that they do
not depend on the threads, through the stacked operator.

```python
streams = np.random.SeedSequence(seed).spawn(int(count))
delays = thread_map(
    lambda stream: stack(prior.sample(rng=np.random.default_rng(stream))),
    streams,
    n_jobs=n_jobs,
)
```

At degree 160 a draw is three spectral transforms and its delays a fourth,
about 0.85 s a draw with four threads, some four of part D's six minutes
for its 300 draws; the push-forward's covariance is sixteen applications of
the factor and its adjoint a prior. The test checks the draws' sample
correlation against the push-forward's by Fisher's transform.

## 6. The pictures

**The `L2` kernels** of parts A and B are not the space's: they are the
paraxial formula evaluated on a grid of each section (`kernel` in
`src/functionals.py`), with the ray drawn as a line or a cross.

**Sections** of fields of the space, the draws of the priors part and the
Sobolev equivalents of part B, come from the library's sections of the
ball, which evaluate the field round a great circle at the space's own
radii. From `section` in `src/sections.py`:

```python
values, radii, _, _ = section_values(space, field, **options)
```

with `pole=(90, 0)` for the ray's own plane, the equator, and
`longitude=` the turning point's for the meridian across the ray there;
the app turns and crops them and draws them with `pcolormesh`.

**The cut-away** of part B is `pygeoinf.plotting.plot_cutaway`, which
hands the field to `planetmodel` as a model, `plotting.planet_model(space,
field)`, and draws `planetmodel`'s cut-away of it, in degrees and
latitudes. From `cutaway_figure` in `gradients.py`:

```python
plotting.plot_cutaway(
    space, image, ax=ax, longitudes=longitudes, latitudes=latitudes,
    view=view, inner=CORE_RADIUS, n=CUTAWAY_PIXELS,
    inner_color=CORE_COLOR, surface=SURFACE_COLOR,
    cmap="RdBu_r", vmin=-limit, vmax=limit,
)
```

The ray is the scale's deepest, the one to the largest distance
(`deepest`), whose two representers alone are formed for the picture. The
wedge is north of the equator, `latitudes=(0, 90)`, from a quarter of the
way round west of the ray's turning point to it, under `planetmodel`'s
default view (`view=None`), which shows the wedge's second face nearly
square-on: that face is the section across the ray at mid-path, and the
floor of the cut the ray's own plane from the source to mid-path. The
sphere left standing at the inner radius is the core, a plain color, and
so is the surface, the field being on the faces of the cut alone. To draw
the ray on the floor the app asks `planetmodel` for the geometry of the
same picture: its cut-away of the same wedge and view through
`plotting.planet_model`, two pixels across, whose `project` puts points
where the picture does and whose `visible` says which it shows
(`cutaway_geometry`). The ray's nodes, projected and kept where seen, are
drawn over the picture; the test checks that the nodes from the source to
mid-path are seen and the rest, inside the body, are not.

## 7. What a reader would change

- **The planet and the phase**: `src/planet.py` holds the profile and its
  derivatives; any radial profile without discontinuities that the shared
  tracer traces, and whose rays have no caustic between their ends, gives
  the dynamic ray tracing the kernel needs.
- **The bands, the distances and the quadrature**: fields of the scale and
  options of every script (`--periods`, `--distances`, `--detour-nodes`,
  `--angle-nodes`, `--nodes`); `--surface clip` drops the points above the
  free surface instead of folding them.
- **The space**: `model_space`, its truncation, order, length scale and
  padding, each an option; the functionals are bounded on any order above
  three halves.
- **The priors**: `--prior-lengths` and `--prior-pairs`, any lengths on
  the same space, `--prior-order` above three halves and `--prior-std`;
  each is placed by its measured correlation distance whatever its
  nominal lengths. `--draws` sets part D's draws.
