# LLSVPs: thermal or chemical? A sampling mini-app (plan)

What the mini-app asks, how it is built, what each part shows of the sampling
package, and what the library still lacks for it. **Parts A to D and F are built**, on
a thin shell with a sharp indicator and a synthetic network; their numbers are in the
[README](README.md) and their construction in [IMPLEMENTATION.md](IMPLEMENTATION.md).
Parts E, G and H and the gaps below are what remains. Facts marked *checked* were
confirmed against the `sampling` branch, and the measurements are at the quick scale
unless they say otherwise.

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
  continents through the land mask the point-data app uses. *Checked:* about
  6% of the arcs enter a cap of radius 0.3 about Africa and 4% about the Pacific.
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
and a thermal draw from the prior. Noise: 20% of the rms of the S data, 30% of the P data and 20% of the modes'.

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

**C. The thermal-only problem, exactly** (`linear.py`). Built. With no chemistry
the problem is linear and Gaussian, so the posterior is known exactly. PCN, MALA, HMC,
PreconditionedHMC and GeometricHMC run on it, on the data as given and on a weak
problem, the same noise draw times 15, where the prior-reference samplers mix and the
check has power; and on a ladder of degrees, at fixed steps and against a random walk.

**D. Gibbs over the fields** (`gibbs.py`). Built. Two blocks, not the three first
planned: HMC on the thermal fields and the contrasts as one nested summand, and pCN on
the level set, since apart the first two creep along a valley they trade off in
(*checked:* `c_s` at −1.29, τ 717 after 800 sweeps of three blocks, against −1.9 with
two). Compared in squared jumps per forward evaluation: pCN steps per visit, a symmetric
scan, three blocks, and a start from the prior.

**E. What coupling costs** (`coupling.py`). The thermal fields sampled as one
block, then as three: with R near 0.95 the three-block Gibbs crawls along the
correlation, and the part measures it, against R from 0 to 0.95. Part A already
measures how little room there is: each of `vs` and `vp` has 9% of its marginal's
variance given the other two. This needs grouped blocks or a block-sparse correlation
(gap 2 below).

**F. The answers** (`answers.py`). Built. From the part-D posterior: the map of
`P(x inside a province)`, with the truth's outline; the posterior of `c_rho` and of
its sign; the jump in `dlnVs` across the provinces' edges over the jump in `dlnVp`,
against the thermal value and what thermal fields alone make of the data; and the number
of connected provinces. The jump is used, not a ratio inside against one outside, since
a contrast offsets a field inside without changing its slope there.

**G. Does the data need chemistry?** (`evidence.py`). SMC with the Gibbs
kernel as its mutation, once on the thermal-only model and once on the full
one, and the log evidence of each from the `WeightedEnsemble`. A sweep of
the noise shows where the data stop telling the models apart. This needs a
standard error on the evidence (gap 5).

**H. (stretch) Several modes.** Two kinds are now measured. An exact near-symmetry,
`c H(-phi) == c - c H(phi)` (*checked* to 8e-15), which only the prior breaks. And
arrangements of the provinces that chains hold on to: part D's chains from one guess
split into two groups, 40 apart in potential, one with a lobe over the North Atlantic,
and chains from the prior stay higher still. Tempering is the natural test. A fitted
mixture reference would need per-block references (gap 6).

## 4. Layout

```
miniapps/llsvp/
  README.md, IMPLEMENTATION.md, PLAN.md, run_all.py
  proofs.py  synthetic.py  linear.py  gibbs.py  answers.py     # built
  coupling.py  evidence.py                                     # to come
  src/
    shell.py      # the Scale presets, the network, the mode kernels
    prior.py      # the coupled prior, its two groupings, the true level set
    forward.py    # T, G(phi), the map and its derivative
    record.py     # what a chain records
    common.py     # the command line from the Scale, the stopwatch, the maps
  results/        # git-ignored; results/quick/ at the quick scale
```

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
3. **Differentiable in some summands only.** Worked around, and enough for now: the
   forward map's derivative is declared zero in `phi`. A library notion of a partial
   derivative per summand would stop a sampler being able to follow it across the level
   set by accident.
4. **Nonlinear records.** Done: `record=` takes any `Operator`, and a fitted reference
   asks for `chart=` over a nonlinear one. Streaming pointwise statistics remain: the
   province map still averages stored states, 40 KB each.
5. **The evidence's standard error.** Part G compares two log evidences, and
   tempering reports no error for them (already an open item); nor the
   per-stage history that shows where the estimate's variance comes from.
6. **Per-block references.** Gibbs blocks take the conditional prior as
   reference, so neither a geometric block (the thermal block about its
   Laplace Gaussian would mix in a step) nor a mixture reference (part H)
   fits inside a sweep. Already listed as not offered.
7. **Process safety of the sphere.** *Checked:* chains in worker processes work on
   the sphere, with the operators' closures and the stored `SHGrid`s, and run 2.3 times
   faster on two workers.
8. **A block's cost.** Every block step evaluates the whole forward map, though a
   level-set step changes only the indicator's term and a linear step only the linear
   term. `Forward` caches `G(phi)` itself. Cheap here; for an expensive forward map a
   block would want the others' contribution held.
9. **The chart's pairings.** A `GaussianReference` pairs a state with each chart vector
   by the space's inner product, in `coordinates` and `_apply_covariance`, which on the
   sphere is two analyses per field per vector: 0.42 s a step at rank 50, nine tenths of
   it there, and 3.7 s to build the reference. A chart's dual components computed once
   would make each a single analysis and a matrix product. This is why part C counts
   `GeometricHMC` in steps of its own.
10. **A joint Gaussian block.** Given `phi` the linear part's conditional is linear and
    Gaussian, known exactly, and part D samples it by HMC with τ in the hundreds. Per-block
    references (gap 6) on it, or an exact-draw block, should take most of that away and
    leave pCN's level set as the slow part. Probably the largest gain available to part D.
11. **The trajectory's resonance.** HMC's adaptation stops at `L h == pi / 2`, where a
    direction of frequency near 4 makes a whole turn: on the weak problem the potential's
    τ is 16 or 143 by seed, against 6.4 at `h == 0.15`, and the regional means do not see
    it. Jittering the trajectory's length or step would remove it.
12. **Threads on the sphere.** `ducc0` fixes its pool at import to one thread per core,
    and at these sizes one thread is several times faster: 0.10 ms an analysis against
    0.59. The workers are told one by the library; the main process is not. The app sets
    `OMP_NUM_THREADS` in `src/__init__.py`, and the library could default the sphere's
    transforms to one thread, as it does its NUFFTs.

## 6. Order of work

1. Done: `src/` and parts A to D and F.
2. Gaps 5 and 2 in the library, then parts G and E.
3. Gap 10 or 6, for a linear block that does not crawl, then a longer part D, to see
   whether its two groups of chains become one.
4. Gap 1, hyperparameters, then the version of part F that infers the scaling; part H.

## 7. Decisions

Made: a thin shell, a sharp indicator with a derivative zero in it, and a synthetic
network. Open, for the later stages: whether the near-symmetry of part H is broken by a
constraint or kept as a demonstration; synthetic mode coefficients, or published values
with their own errors; and which quantities the README leads with once part G exists.

## 8. Extensions

- The layered ball: the same fields on `ball.Layered` through D″ and above,
  with the travel times as true ray integrals (`symmetric_space/rays.py`
  bends rays on symmetric spaces).
- CMB topography as a fourth field on the same sphere, correlated or not,
  seen by PcP and PKP: the classic trade-off with D″ velocity.
- Real data: Sdiff and Pdiff residuals and published structure
  coefficients, with their published uncertainties.
