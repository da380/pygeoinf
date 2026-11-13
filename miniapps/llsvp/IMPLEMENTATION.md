# The LLSVPs: how the app uses the library

The [README](README.md) says what the app asks and what it finds. This document
walks through how it is built on `pygeoinf`: a nested direct sum, a coupled prior, a
forward map with a sharp indicator in it, and the sampling package run on them.
It assumes the tutorials under `tutorials/` and the sampling package's module
docstring. Code is quoted from the app's own files.

## 1. The spaces

Every field is on `pygeoinf.symmetric_space.sphere.Sobolev`, `H^2` of the unit sphere
with a length scale of 0.3, about 1100 km at the layer: a value at a point exists,
which the indicator's evaluation at quadrature nodes needs, and the vectors are
`pyshtools.SHGrid`, so a recorded scalar reads the grid without a transform. The
model space is a direct sum, and in `src/prior.py` it comes straight from the priors:

```python
GaussianMeasure.from_product(
    [thermal_prior(scale, space), level_set_prior(scale, space), chemistry_prior(scale)],
    labels=SUMMANDS,
)
```

The first factor is itself a product, so the first summand is itself a `DirectSum`, of
the three thermal fields; the third is `EuclideanSpace(3)`. Part D wants the same measure
cut differently, the thermal fields and the contrasts as one summand, and that is
another nesting of the same factors (`grouped_prior`), with `ungrouped` reading a model
of one as the triple the forward map takes. A correlation ties factors on one latent
space only, so the thermal fields are correlated inside their own sum and cannot be
top-level summands beside a Euclidean one.

The data space is one `EuclideanSpace`, the kinds stacked (`Forward.slices`), and the
noise `GaussianMeasure.from_standard_deviations` on it, one level per kind, set from
the rms of that kind's clean data (`synthetic.noise_levels`).

## 2. The coupled prior

```python
factors = [
    space.sobolev_measure(scale.prior_order, scale.length_scale, pointwise_std=std)
    for std in scale.thermal_std
]
return GaussianMeasure.from_product(
    factors, labels=FIELDS, correlation=correlation_matrix(scale)
)
```

Each field keeps its own pointwise standard deviation and draws through its own
factor, and the latent white noises are mixed by the symmetric root of the correlation
matrix, mode by mode. The marginals are the factors unchanged. `prior.conditional(name,
given)` reads the closed form of that structure, a Schur complement of the correlation
matrix, which part A checks against the dense Schur complement of the whole covariance at
degree 8. The formula holds of the covariance operator's matrix as it stands: its
blocks are the components' covariance times the one field's metric, and the inverse
in the Schur complement cancels the metric.

## 3. The forward map

`T` is assembled once as a dense matrix on the components, by field (`Forward._assemble`).
The arcs and caps come from the sphere's own operators, `path_average_operator(arcs,
count=, dense=True)` and `geodesic_ball_average_operator(centers, radius, dense=True)`,
and the modes' rows pick components by `component_of(degree, order)`, each scaled by
that mode's weight on that field. `LinearOperator.from_matrix(thermal_space, data_space,
matrix, form="components")` takes it from there, the adjoint included, with the metric
in the right place, which `check_operator` is there to catch.

`G(phi)` is the same functionals of the indicator, and it cannot use those operators,
since an indicator is not a vector of the space: band-limited, it would ring. So
`Forward._prepare_indicator` takes the quadrature nodes and weights of the same arcs
and caps (`geodesic_quadrature`, `geodesic_ball_quadrature`, pooled by `pooled` into
nodes and a sparse matrix of weights). `response` evaluates `phi` at the nodes,
thresholds and averages. Part A checks that these quadratures give the operators'
answers on a smooth field: to rounding on the arcs, whose nodes are the operator's
own, and to 7e-6 on the caps, whose operator is exact where the quadrature is not.

The modes' coefficients of the indicator are integrated on a fine grid of one degree at
every truncation, 128: `phi` is lifted by `degree_transfer_operator`, thresholded on that
grid and analysed. The coarse grid will not do: a sharp edge falls between its rows.

The last `G(phi)` is kept against the grid's bytes, since the linear block holds `phi`
through its whole trajectory and each gradient asks for it. A worker holds its own.

## 4. The derivative, zero in the level set

The indicator's derivative in `phi` is zero wherever it exists, so the operator's
derivative is

```python
def adjoint(residual: np.ndarray) -> tuple:
    return (self.thermal.adjoint(residual), self.space.zero(), gain.T @ residual)
```

which part A proves in three ways: `check_operator` on it, its action against the map's
difference along `(t, c)`, where the map is linear so that the two are equal to rounding,
and the same with a direction in `phi` added, which changes nothing. It is not the
derivative of the map as a whole. It is what a `Gibbs` block needs, since a block's
gradient is the summand's part of the whole gradient: `HMC` on `(t, c)` follows it,
and `PCN` on `phi` never asks. A sampler that followed it across the level set would
be blind to the provinces' edges, which is why no such block is built.

## 5. Sampling a linear problem, and marking it

`linear.pose` builds `LinearGaussianInversion(problem, thermal, solver=CGSolver(rtol=))`:
one solve per posterior mean, one per column of the recorded map's covariance, and one
for a draw of the posterior by randomize-then-optimize. The chains start from
`start=posterior`, a measure each chain draws from: stationary from the first step, so
the burn-in only adapts the step.

```python
reference = inversion.reference(data, rank=rank, rng=rng)
sampler = Chains(HMC(step=scale.ghmc_step, length=scale.ghmc_length), steps=..., ...)
MonteCarloEstimator(problem, prior, sampler=sampler, reference=reference, record=...)
```

Every sampler of the part is that estimator with another move: `PCN`, `MALA`, `HMC`
and `GeneralizedHMC` about the prior, and `HMC` again about the reference. The
reference's cost is why that last run has its own count of steps. It pairs a state
with each chart vector by the space's inner product, and on the sphere each pairing runs
two spherical-harmonic analyses per field, so its cost grows with the reference's rank.

A sampler's `evals` column counts forward evaluations, to compare samplers on equal
terms, and a row's `|z|` means nothing above R-hat 1.1: a chain's own error bar on a
mean is only as good as its autocorrelation estimate. On the same grounds the table has
the likelihood potential's τ beside the regional means': it is what sees a slow direction
the means do not (README, part C).

The ladder (`linear.ladder`) rebuilds `Forward` and the prior at each degree on one
network, one data set and one noise, and copies the random walk from example 38: a
proposal shaped like the prior, whose ratio needs the prior's density against the grid,
and which the library does not offer.

## 6. Gibbs

```python
move = Gibbs(
    {
        "linear": HMC(step=scale.linear_step, length=scale.linear_length),
        "level_set": PCN(step=scale.level_step),
    },
    order=order,
    repeat={"level_set": level_steps},
)
MonteCarloEstimator(state.grouped_problem, state.grouped, sampler=Chains(move, ...), ...)
```

A block is a summand of the top-level sum, however nested. Each visit builds the
block's target against `prior.conditional`, runs the block's move for its `repeat`
steps and writes the summand back; each block's step adapts to its own acceptance during burn-in, and
`posterior.block_acceptance_rates` reports it. The nesting here is what the measurements
ask for: three blocks are one `gibbs.three_blocks` away on the ungrouped sum, and they
creep. `run_chains` picks the recording map to suit the sum it is on.

`Chains(..., n_jobs=)` runs the chains in worker processes: the sphere's objects, the
operators' closures and the stored `SHGrid`s all pickle, and joblib hands the workers the
parent's `sys.path`, so they import `src` by its plain name. A chain's records and stored
states come back when it ends, and the library's rule of one thread to a worker applies.

The start is a measure too: `prior.affine_map(spread * identity, translation=start)`,
the law of `start + spread X` with `X` a prior draw, a Gaussian on the same space. It gives
each chain its own start, close to the guess, so R-hat there tests agreement within that
guess's basin and no more. The chains from the prior are the test of that.

## 7. Recording nonlinear functions

`record=` takes any `Operator` into a space with coordinates. `src/record.py` builds one
with `Operator.from_callables` and no derivative:

```python
Operator.from_callables(
    domain, EuclideanSpace(len(NAMES)), lambda model: summary(space, read(model))
)
```

`summary` reads the grids directly: the contrasts, the provinces' area, their number by
`ndimage.label` joined across the date line, and the jump of each field across their
edges, whose ratio is one scalar among them. Being recorded, each gets the chains'
diagnostics, `scalar_means` and `scalars()` free. The province probability is an average
over stored states instead (`thin=`), being a map. A `fitted_reference` needs a linear
chart, so over this record it asks for `chart=`.

## 8. What a reader would change

- `--noise-s`, `--noise-p`, `--noise-modes`: the data's strength, the one number the
  sampler's speed turns on; `--weak-noise` for part C's other problem.
- `--contrasts`, `--province-width`, `--province-level`: the truth; contrasts of zero ask
  whether the app can tell a thermal Earth.
- `--level-steps`, `--gibbs-steps`, `--gibbs-chains`: how long part D runs.
- `--thermal-correlation`: the scaling the prior holds to.
