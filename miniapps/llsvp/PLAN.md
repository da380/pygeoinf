# LLSVPs: thermal or chemical? A sampling mini-app (plan)

A plan, not an app: what the mini-app would ask, how it is built, what each
part shows of the sampling package, and what the library still lacks for it.
Nothing below is built. The facts marked *checked* were confirmed against
the `sampling` branch on 2026-11-03.

## 1. The question

The two large low-shear-velocity provinces at the base of the mantle, under
Africa and the Pacific, are slow in S, less slow in P, and possibly dense. If
they are hot mantle, their anomalies in Vs, Vp and density follow the
thermal scaling of mineral physics, Vs about twice as sensitive as Vp and
density following both. If they are chemically distinct piles, there is an
extra contrast inside them that breaks the scaling: a ratio
`dlnVs / dlnVp` well above the thermal value, and density anti-correlated
with Vs. The app asks the data for that extra contrast, for where the
provinces are, and whether the data need chemistry at all.

Every answer is a posterior quantity that is nonlinear in the model: the
probability that a point lies inside a province, the sign of a density
contrast, a ratio of regional means, an evidence ratio. Sampling is how they
are reached, not an alternative to a linear answer.

## 2. The problem

### Geometry

The D″ layer, the lowermost ~300 km of the mantle, treated as a thin shell:
every field lives on one `Sphere` (lmax by scale) at the layer's mid-radius,
3630 km. The data are averages through the layer, so its radial structure is
folded into their weights. A 3-D layered ball (`ball.Layered`, as the
tomography app uses) is the extension, not the start (§8).

### The unknowns: a nested direct sum

```
model space = DirectSum(
    thermal   = DirectSum(vs, vp, rho),   # three fields on one Sphere
    level_set = Sphere,                   # the provinces are {phi > 0}
    chemistry = EuclideanSpace(3),        # (c_s, c_p, c_rho), the contrasts inside
)
```

*Checked:* `from_product` nests, giving exactly this space with its labels;
a prior draw takes 13 ms at lmax 32; `prior.conditional(key, x)` is in closed
form for each of the three top-level summands, and within the thermal triple
for one field given the other two (0.6 ms a draw).

### The prior

- **thermal:** `GaussianMeasure.from_product([vs, vp, rho], correlation=R)`,
  each a Sobolev measure on the sphere with its own pointwise standard
  deviation (about 1%, 0.5% and 0.3% for the logarithmic anomalies) and one
  length scale. `R` is the thermal scaling as a correlation, high (0.9 to
  0.95 between Vs and Vp): the fields share their latent noise mode by mode,
  which is exactly the case `correlation=` is built for.
- **level_set:** a Sobolev measure on the same sphere, its length scale
  setting how many provinces a draw holds (zero to three at the default).
- **chemistry:** Gaussian on the three contrasts, centred on zero, wide
  (2%, 1%, 1%).

So the anomaly fields seen by the data are

```
dlnVs = vs  + c_s   H(phi)
dlnVp = vp  + c_p   H(phi)
dlnrho = rho + c_rho H(phi)
```

with `H` the indicator of `phi > 0`.

### The data: two kinds, seeing the fields differently

- **Travel-time residuals.** Sdiff and Pdiff as averages along great-circle
  arcs on the shell (`path_integral_operator` / `path_average_operator` on
  the sphere, *checked* present), and ScS−S and PcP−P as averages over small
  caps at the bounce points (`geodesic_ball_average_operator`). These see Vs
  and Vp, not density. Events along plate boundaries, stations on the
  continents through the land mask the point-data app uses, so the Pacific
  is crossed far less than Africa.
- **Normal-mode structure coefficients.** For a handful of modes sensitive
  to D″, the even-degree coefficients `c_st`, s = 2, 4, 6, each a weighted
  sum of the three fields' spherical-harmonic coefficients of that degree
  and order with weights `(K_s, K_p, K_rho)` per mode (`component_of`).
  The only data that see density, so they carry the thermal-or-chemical
  question; and the LLSVPs are mostly degree 2, where these data live.

The forward map is linear in the thermal fields and the contrasts, and
nonlinear only through `H(phi)`. The indicator is evaluated where the data
need it, at the path and cap quadrature nodes and on the analysis grid for
the low-degree coefficients, never as a band-limited field (which would ring).

### The truth and the noise

A deterministic level set dominated by degree 2 that puts provinces under
Africa and the Pacific, chemical contrasts of (−2%, −0.5%, +1%), dense piles,
and a thermal draw from the prior. Noise: a fixed fraction of the rms of each
data type, chosen so that the travel times pin the provinces' edges where
coverage is good and the mode data constrain density to a factor of two.

### Scales

| preset | lmax | arcs | caps | modes | use |
|---|---|---|---|---|---|
| `proof` | 12 | 40 | 20 | 2 | the check ladder on the forward map, the conditionals against brute force |
| `quick` | 24 | 600 | 300 | 6 | the test, minutes |
| `full` | 48 | 3000 | 1500 | 10 | a few minutes per part |

## 3. The parts

Each part is a script with `run()` and `main()`, as in the other apps.

**A. The forward map and its proofs** (`proofs.py`). The check ladder on the
linear part, the indicator's evaluation against a brute-force grid, and the
conditionals of the coupled prior against `condition` by brute force at the
proof scale. Figure: the network and the mode kernels.

**B. The prior and the data** (`synthetic.py`). Prior draws of the three
coupled fields, side by side, and of the level set; the truth; the data.
Shows `from_product(correlation=)` and a nested direct sum. Figure: draws
and the truth on maps.

**C. The thermal-only problem, exactly** (`linear.py`). With no chemistry
the problem is linear and Gaussian, so the posterior is known exactly. PCN,
MALA, HMC, PreconditionedHMC and GeometricHMC run on it and each is checked
against the exact posterior on regional means. The same runs at lmax 12, 24
and 48 show acceptance and step flat under refinement. Shows: every preset,
the diagnostics, discretisation invariance. Figure: autocorrelations, and
acceptance against lmax.

**D. Gibbs over the fields** (`gibbs.py`). The full problem by
`Gibbs(problem, prior, blocks={"thermal": HMC.recipe(...), "level_set":
PCN.recipe(..., steps=k), "chemistry": MALA.recipe(...)})`: per-block step
adaptation and acceptance (`block_acceptance_rates`), and systematic against
symmetric scan. The level set is the hard block: pCN on a level set slows as
the data inform its edges, and the part says how many pCN steps per visit
pay. Figure: traces per block, block acceptances, the province probability.

**E. What coupling costs** (`coupling.py`). The thermal fields sampled as one
block, then as three: with R near 0.95 the three-block Gibbs crawls along
the correlation, and the part measures it, ESS per forward evaluation
against R from 0 to 0.95. This needs grouped blocks or a block-sparse
correlation (gap 2 below).

**F. The answers** (`answers.py`). From the part-D posterior: the map of
`P(x inside a province)`, with the truth's outline; the posterior of
`c_rho` and of its sign (dense or not); `dlnVs / dlnVp` inside the
provinces against outside, against the thermal value; the number of
connected provinces. Every one is nonlinear in the model, and the Laplace
approximation gives none of them. Figure: the probability map, and the
marginals against the truth.

**G. Does the data need chemistry?** (`evidence.py`). SMC with the Gibbs
kernel as its mutation, once on the thermal-only model and once on the full
one, and the log evidence of each from the `WeightedEnsemble`. A sweep of
the noise shows where the data stop telling the models apart. This needs a
standard error on the evidence (gap 5).

**H. (stretch) Several modes.** The model has a near-symmetry that makes
the posterior bimodal by construction: flipping `phi` and the contrasts'
signs and shifting the thermal fields' mean gives nearly the same anomalies
(`c H(-phi) == c - c H(phi)`), broken only by the prior. Either break it (a
constraint `c_s < 0`, `constraint=`) or show it: tempering crosses between
the modes where a chain does not. A fitted mixture reference would need
per-block references (gap 6).

**Run times.** A forward evaluation is a matrix-vector product per data type
plus one sphere synthesis for the indicator: a few milliseconds at lmax 24.
A Gibbs sweep of three blocks costs a few such evaluations and gradients, so
10 000 sweeps fit in about a minute, and the forward is pure numpy and
pyshtools, so chains and SMC particles can run in worker processes
(`Chains(n_jobs=)`), unlike the waveform app's C++ operator. To check: that
the sphere's objects pickle and that pyshtools is safe there (gap 7).

## 4. Layout

```
miniapps/llsvp/
  README.md, IMPLEMENTATION.md, run_all.py
  proofs.py  synthetic.py  linear.py  gibbs.py  coupling.py  answers.py  evidence.py
  src/
    shell.py      # the Scale presets, the sphere, the network, the mode kernels, the truth
    prior.py      # the nested coupled prior, labels, the thermal correlation
    forward.py    # the anomaly fields, the indicator at the nodes, the two data operators
    common.py     # the command line from the Scale, the stopwatch, the map plots
  results/        # git-ignored; results/quick/ at the quick scale
```

The same conventions as the other apps: `--quick` / `--full`, every field of
the scale an option, a part rebuilding the problem from the seed when run
alone, and a slow test that runs the quick scale.

## 5. What the library has, and what the app would ask for

### Already there (*checked*)

- the nested `DirectSum` with labels, and a coupled prior on it through
  `from_product(correlation=)`, with closed-form `conditional`s;
- `Gibbs` with `PCN`, `MALA`, `HMC` and `PreconditionedHMC` recipes, the
  geometric presets refusing a recipe with a clear message;
- a block's target keeping the target's temperature, and `Sweep.with_target`,
  so a Gibbs kernel can in principle be SMC's mutation (not yet run);
- path and cap averages on the sphere, spherical-harmonic components by
  degree and order, and point evaluation at prepared points.

### Gaps it would expose, most important first

1. **Hyperparameter blocks.** The scientific heart of the question is the
   scaling itself: inferring the thermal correlation `R`, or the ratio of the
   fields' standard deviations, rather than fixing it. Likewise the level
   set's length scale and the noise levels. `Gibbs` does not offer
   hyperparameter blocks; this app is the natural first user, and the
   design (a block whose update changes the prior the other blocks'
   conditionals are taken from, and the function-space care that needs,
   since priors with different `R` are mutually singular) is a design
   question to settle with David.
2. **Grouping blocks, or a block-sparse correlation.** `from_product`
   correlates factors on one latent space only, so the three thermal fields
   can be correlated only inside their own nested sum, and Gibbs blocks are
   the top-level summands. Part E needs the three fields as separate blocks
   while correlated: either `blocks={("vs", "vp", "rho"): ...}` over a flat
   sum, or `correlation=` allowing zeros between summands of different
   latent spaces.
3. **Differentiable in some summands only.** The forward map has no
   derivative in `phi` (the indicator), but the thermal and chemistry blocks
   need gradients. A block's potential is the whole potential composed with
   the insertion, so its gradient asks for the whole derivative. The options:
   a derivative declared zero in `phi` (true almost everywhere, but a quiet
   lie), a smoothed indicator (a modelling choice to defend), or a library
   notion of a partial derivative per summand, which a block would use.
4. **Nonlinear records and streaming statistics.** `record=` takes a linear
   map, but the answers here are nonlinear (an indicator, a ratio, a count).
   Storing states instead costs about 40 KB each at lmax 24 for five fields,
   400 MB for 10 000 states. A recorded nonlinear function, or running
   pointwise means of a function of the state, would avoid that; this fits
   the planned change to how chains are stored.
5. **The evidence's standard error.** Part G compares two log evidences, and
   tempering reports no error for them (already an open item); nor the
   per-stage history that shows where the estimate's variance comes from.
6. **Per-block references.** Gibbs blocks take the conditional prior as
   reference, so neither a geometric block (the thermal block about its
   Laplace Gaussian would mix in a step) nor a mixture reference (part H)
   fits inside a sweep. Already listed as not offered.
7. **Process safety of the sphere.** Chains and particles in worker
   processes need the sphere space and its prepared points to pickle and
   pyshtools to be safe in a worker; to be checked, and the app is the test
   of it.
8. **A block's cost.** Every block step evaluates the whole forward map,
   though a level-set step changes only the indicator's term and a thermal
   step only the linear term. Cheap here; for an expensive forward map a
   block would want the others' contribution held. Not needed for this app,
   worth noting for later ones.

## 6. Order of work

1. `src/` and parts A and B: the problem, its proofs, the coupled prior.
2. Part C: the exact linear benchmark, which tests the samplers before the
   hard problem does.
3. Part D with the zero-derivative or smoothed choice for gap 3, and part F
   from its states (gap 4 worked around by thinning).
4. Gaps 5 and 2 in the library, then parts G and E.
5. Gap 1, hyperparameters, then the version of part F that infers the
   scaling; part H if the near-symmetry is kept.

## 7. Decisions to make first

- The thin shell, or a layered ball from the start.
- A sharp or a smoothed indicator (gap 3), and whether the near-symmetry of
  part H is broken by a constraint or kept as a demonstration.
- A synthetic network on a realistic geometry, as the other apps use, or a
  real catalogue's events and stations; for the mode coefficients,
  synthetic, or published values with their own errors.
- Which quantities are *the* answers, so that the README can lead with them:
  my suggestion is the province probability map, the sign of `c_rho`, and
  the evidence ratio.

## 8. Extensions

- The layered ball: the same fields on `ball.Layered` through D″ and above,
  with the travel times as true ray integrals (`symmetric_space/rays.py`
  bends rays on symmetric spaces).
- CMB topography as a fourth field on the same sphere, correlated or not,
  seen by PcP and PKP: the classic trade-off with D″ velocity.
- Real data: Sdiff and Pdiff residuals and published structure
  coefficients, with their published uncertainties.
