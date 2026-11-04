# The large low-shear-velocity provinces: thermal or chemical?

How the app is built on the library is walked through in
[IMPLEMENTATION.md](IMPLEMENTATION.md). What is still to come is in
[PLAN.md](PLAN.md): parts E and G, which wait on the library.

## The problem

The two large low-shear-velocity provinces at the base of the mantle, under
Africa and the Pacific, are slow in S, less slow in P, and possibly dense. If they
are hot mantle, their anomalies follow the thermal scaling of mineral physics. If
they are chemically distinct piles, there is an extra contrast inside them that
breaks it. This app is a sampler's: every answer it asks for is nonlinear in the
model, and the posterior has a sharp edge in it.

The D″ layer, the lowermost 300 km of the mantle, is treated as one sphere, and
a datum as an average through it. The model is three things on that sphere:

- **Three coupled thermal fields**, `dlnVs`, `dlnVp` and `dlnrho`, in percent: a
  Sobolev measure each, of pointwise standard deviation 1, 0.5 and 0.3, sharing
  their latent noise mode by mode through `from_product(correlation=)`, with
  correlations 0.95, 0.8 and 0.8, the thermal scaling.
- **A level set** `phi`, a Sobolev measure of unit pointwise standard deviation.
  The provinces are `{phi > 0}`, as many and as shaped as its zero contour says.
- **Three contrasts** `c`, independent Gaussians of standard deviation 2, 1 and 1
  percent, added to the three fields inside the provinces.

So `dlnVs == vs + c_s H(phi)` and likewise for the others, with `H` a sharp
indicator. The data are `d == T t + G(phi) c`, linear in the thermal fields `t` and
the contrasts and nonlinear only in `phi`:

- **Travel times.** Sdiff and Pdiff as averages along great-circle arcs, the part
  of the path between an event and a station that remains when 40° of descent is
  taken off each end; ScS and PcP as averages over small caps at the bounce points.
  These see `dlnVs` and `dlnVp`, and not density.
- **Normal modes.** The structure coefficients of degrees 2, 4 and 6 of a handful
  of modes, each a weighted sum of the three fields' coefficients. The weights are
  illustrative, spanning mostly shear to mostly density, and are not the kernels of
  any named mode. These are the only data that see density.

Everything is synthetic on a realistic geometry: events along hand-placed plate
boundaries, stations on the Natural Earth land north of Antarctica. The truth is a
thermal draw from the prior, a level set with one province under Africa and one
under the Pacific, 27% of the sphere between them, and contrasts of (−2, −0.5, +1)
percent: slow and dense. The noise is 20% of the rms of the S data, 30% of the P
data and 20% of the modes'.

The parts:

- **A, the forward map and its proofs** (`proofs.py`): the library's check ladder
  on `T`, on the derivative and on the recorded property map; the indicator against
  what can be done by hand on a cap about the pole; the two identities the sharp
  map obeys; and the coupled prior's closed-form conditionals against the dense
  Schur complement. Its figure: the modes' weights, and how much of each S datum
  the provinces fill.
- **B, the prior and the data** (`synthetic.py`): the coupled fields' correlation
  against what was asked, the level set's draws, the truth, and each kind of
  data's budget of thermal signal, chemical signal and noise. Its figures: prior
  draws, and the truth with what the network sees of it.
- **C, the thermal-only problem** (`linear.py`): with no chemistry the problem is
  linear and Gaussian and its posterior known exactly, so that every sampler is
  marked against it; then a ladder of truncation degrees. Its figure: what a
  forward evaluation buys, the check, and the ladder.
- **D, Gibbs over the fields** (`gibbs.py`): the whole problem, HMC on the linear
  part and pCN on the level set, against three blocks, other scans and steps per
  visit, and chains from the prior. Its figure: the chains, and those comparisons.
- **F, the answers** (`answers.py`): from part D's chains, where the provinces are,
  whether they are dense, what the shear jump over the compressional is, and how
  many there are. Its figure: the province probability and the marginals, by chain.

Two scales of the same problem. `--quick` is degree 24, 600 arcs, 300 caps and 6
modes, 1062 data and 2503 unknowns; every part runs in about 13 minutes on 4 workers,
of which part C takes about 4 and part D about 9. The full scale is degree 48, 3000
arcs, 1500 caps and 10 modes, 4770 data and 9607 unknowns; every part runs in about 82
minutes on 4 workers, of which part C takes 20 and part D 60. Its numbers are
[below](#the-full-scale), and its part D has not converged.

## How to run

```
cd miniapps/llsvp
python run_all.py --quick     # every part built, small scale
python proofs.py              # one part at the full scale; likewise synthetic.py,
                              # linear.py, gibbs.py; answers.py runs part D first
```

Every script takes the same options: `--quick` or `--full` (the default) picks a
preset scale, every field of the scale is then an option that overrides it
(`--noise-s`, `--budget`, `--gibbs-steps`, ...), and `--outdir` says where the
figures go. A part run alone rebuilds the problem from the seed. Figures go to
`results/` (`results/quick/` at the quick scale), which git ignores. The test is
`tests/test_miniapp_llsvp.py`, marked `slow`, about 5 minutes, on shorter chains.

Chains run in worker processes (`n_jobs`). Importing `src` sets `OMP_NUM_THREADS`
to one unless it is already set, since the sphere's transforms decide their threads
when first imported and at these sizes one is several times faster than one per
core: on the quick scale in one process, a `GeometricHMC` step takes 0.42 s against
2.49 s and the reference it needs 3.7 s against 24 s.

| file | what it is |
|---|---|
| `src/shell.py` | the scale with its three presets, the events, the stations, what their waves see on the shell, the modes' weights |
| `src/prior.py` | the sphere, the coupled thermal prior, the level set's and the contrasts', the two groupings of the model space, the true level set |
| `src/forward.py` | `Forward`: `T` as a dense matrix, `G(phi)` by quadrature at the nodes and on a fine grid, the map and its derivative |
| `src/record.py` | what a chain records, nonlinear functions of its state read off the grid |
| `src/common.py` | the command line built from the scale, the stopwatch, the maps |
| `proofs.py`, `synthetic.py`, `linear.py`, `gibbs.py`, `answers.py` | parts A to D and F, each a script with a `main()` and a `run()` the others and the test call |
| `run_all.py` | the parts in order |

## A. The proofs

`T` and the derivative pass the check ladder with defects of 3e-15 at the
quick scale, and the map is linear in its contrasts to 9e-16. On a cap about the
pole, the share of a meridional arc is right to 2.4e-4 and the zonal coefficients of
degrees 2, 4 and 6 to 4e-3 on the fine grid of degree 128. On the sphere's own grid at
degree 24 the same coefficients were 25% off at degree 4, a sharp edge falling between
its rows, which is why the fine one is there. The data of `(t + c, -phi, -c)` and `(t,
phi, c)` differ by 6e-15: the model has an exact near-symmetry, `c H(-phi) == c - c
H(phi)`, broken only by the prior. Each coupled field's conditional matches the dense
Schur complement to 2e-14, and its variance given the other two is 9% of its marginal's
for `vs` and `vp` and 34% for `rho`: a Gibbs sampler that took the three in turn would
have little room to move in.

## B. The data at the quick scale

| kind | count | rms | thermal | chemical | noise |
|---|---|---|---|---|---|
| Sdiff arcs | 360 | 1.33 | 0.89 | 0.85 | 0.28 |
| ScS caps | 180 | 1.46 | 0.87 | 0.92 | 0.28 |
| Pdiff arcs | 240 | 0.43 | 0.37 | 0.20 | 0.13 |
| PcP caps | 120 | 0.44 | 0.36 | 0.18 | 0.13 |
| modes | 162 | 0.41 | 0.26 | 0.30 | 0.08 |

The truth's chi-squared is 1073 on 1062 data; its thermal part alone, the chemistry
left out, 9238. The draws' correlation matches the 0.95 and 0.8 asked for to 0.01.
About 6% of the arcs enter a cap of radius 0.3 about Africa and 4% about the Pacific,
at both scales: the Pacific is crossed less, not rarely.

## C. Every sampler against the exact posterior

On the data as given the curvature through the prior's factor has eigenvalues from
3600 down to 1 by about the 80th. On a weak problem, the same noise draw times 15, it
runs from 16 to 0.7 by the 10th. The chains start from exact posterior draws, two at the
quick scale, and each prior-reference sampler gets 2000 forward evaluations per chain.
Quick scale:

| problem | sampler | step | acceptance | τ, regional means | τ, potential | smallest ESS | R-hat | ESS per 1000 | worst offset, sd | s |
|---|---|---|---|---|---|---|---|---|---|---|
| weak | pCN | 0.051 | 0.25 | 609 | 67 | 5 | 1.20 | 1.3 | 0.32 | 8 |
| | MALA | 0.086 | 0.57 | 34 | 16 | 94 | 1.03 | 24 | 0.20 | 20 |
| | HMC | 0.196 | 0.95 | 1.9 | 35 | 215 | 1.03 | 54 | 0.22 | 18 |
| | PreconditionedHMC | 0.299 | 0.81 | 15 | 6.6 | 212 | 1.03 | 53 | 0.15 | 20 |
| | GeometricHMC, rank 10 | 0.393 | 0.97 | 1.3 | 1.1 | 128 | 1.01 | 159 | 0.09 | 12 |
| as given | pCN | 2.2e-4 | 0.25 | 3156 | 291 | 1.0 | 2.60 | 0.25 | 0.83 | 8 |
| | MALA | 3.9e-4 | 0.58 | 3007 | 509 | 1.1 | 2.53 | 0.27 | 0.55 | 22 |
| | HMC | 0.028 | 0.61 | 259 | 42 | 1.5 | 1.70 | 0.39 | 0.74 | 19 |
| | PreconditionedHMC | 0.023 | 0.73 | 2082 | 260 | 1.5 | 1.99 | 0.38 | 0.96 | 21 |
| | GeometricHMC, rank 50 | 0.375 | 0.80 | 8.1 | 24 | 20 | 1.06 | 25 | 0.19 | 42 |

`GeometricHMC` gets 100 steps per chain of 4 leapfrog steps, which is 400 evaluations.
It is the only sampler that is not crawling on the data as given, because its step
does not depend on the eigenvalues it has taken into its reference, where the others'
fall with the leading one. Over the same data the exact posterior mean's
chi-squared is 1253, 4.2 of its standard deviations too many: the misfit of thermal
fields made to hold the chemistry.

Longer chains on the weak problem, 8 of them at 20 000 evaluations each, reduce the
worst offsets to 0.013 for MALA, 0.017 for PreconditionedHMC and 0.031 for GeometricHMC
(at 3600), with standard deviations right to 2%. pCN was not run for as long. HMC's
worst offset there is 0.041, at 4.0 of its own Monte Carlo errors, and its smallest
standard deviation ratio 0.958, which is the next paragraph's doing.

HMC's chains are exact too, but its burn-in shows why a single scalar's τ can mislead.
The adaptation pushes its step to the cap at which a trajectory is a quarter turn,
`L h == pi / 2`, and the leading direction on the weak problem has frequency
`sqrt(1 + 16.1) == 4.14`, so that a trajectory turns it through 1.03 full periods and
brings it almost back. Its potential's τ is then 16 or 143 between seeds, where the
regional means, which barely see that direction, report 3.3, and their error bars are
too small. At `h == 0.15`, where the same trajectory turns it through 0.79 periods, the
potential's τ is 6.3 and 6.5. Both match the exact posterior to 0.02 sd over 20 000
steps. Hence the column for the potential, and a resonance worth jittering away.

The ladder, on the weak problem, at each sampler's step from the table and the same
data at each degree (HMC with three leapfrog steps), and two chains of 100 steps:

| degree | unknowns | pCN | MALA | HMC | random walk of pCN's size |
|---|---|---|---|---|---|
| 12 | 507 | 0.20 ± 0.02 | 0.61 ± 0.02 | 0.93 ± 0.01 | 0.000 |
| 24 | 1875 | 0.30 ± 0.03 | 0.57 ± 0.03 | 0.94 ± 0.01 | 0.000 |

The steps the burn-in settles on are flat: 0.042 and 0.045, 0.095 and 0.096, 0.46
and 0.45. The difference in pCN's acceptance at the quick scale is within what chains
so far short of their τ can say; see the full-scale ladder below.

## D. Gibbs over the fields

Two blocks: HMC, 16 leapfrog steps, on the thermal fields and the contrasts together
as one summand of a nested direct sum, and pCN, 5 steps per visit, on the level set.
The derivative of the forward map is zero in `phi`, and only the first block follows
it. The chains start from a guess the data make, the slowest quarter of the
thermal-only posterior mean as the provinces and the linear part's exact conditional
mean given them: potential 626, the truth's 536, 25% of the sphere in two provinces, and
contrasts of (−0.53, −0.08, +0.88), the thermal fields holding most of the shear.

At the quick scale, four chains of 200 + 800 sweeps in 248 s, steps of 0.017 and 8e-4,
acceptances 0.73 and 0.13, mean potential 565. The contrasts reach the truth's size
within the burn-in. The chains agree on the density contrast and on the ratio of the
jumps, and not on what depends on exactly where the provinces' edges lie:

| scalar | τ | ESS | R-hat |
|---|---|---|---|
| `c_s` | 1295 | 2.5 | 1.86 |
| `c_p` | 1162 | 2.8 | 1.71 |
| `c_rho` | 44 | 73 | 1.04 |
| share of the sphere | 1247 | 2.6 | 1.86 |
| shear jump over compressional | 97 | 33 | 1.07 |

Against the same forward evaluations per chain, in 100 sweeps from the same start
(far shorter than these τ, so squared jumps and not effective sample sizes), with the
potential and `c_s` over the last 50:

| run | evaluations a sweep | squared jump of `c_s` per 1000 | of the share | potential | `c_s` | acceptance |
|---|---|---|---|---|---|---|
| 1 pCN step per visit | 17 | 0.51 | 2.0e-4 | 613 | −1.30 | 0.68, 0.08 |
| 5 | 21 | 0.26 | 9.0e-4 | 596 | −1.31 | 0.73, 0.08 |
| 20 | 36 | 0.16 | 5.5e-4 | 577 | −1.68 | 0.59, 0.06 |
| symmetric scan, 5 | 42 | 0.21 | 1.7e-4 | 560 | −1.89 | 0.67, 0.07 |
| three blocks | 18 | 0.06 | 2.7e-3 | 594 | −0.91 | 0.59, 0.18, 0.62 |
| two blocks, from the prior | 21 | 0.16 | 1.0e-3 | 629 | +0.36 | 0.66, 0.05 |

Moving the thermal fields and the contrasts apart costs a factor of 4 in how far the
contrast moves. Over 200 + 600 sweeps run separately, three blocks left `c_s` at −1.29
with τ 717 where two had reached −1.90. Given the thermal fields a contrast is pinned,
its step adapting to 2e-4, and it can only follow them along the valley they trade off
in. Chains from the prior stay well above the guess's in potential, 629 over these
sweeps: they are holding another arrangement of provinces.

## F. The answers

From the main run's 3200 records and 800 stored states, quick scale:

| scalar | truth | mean ± sd | by chain |
|---|---|---|---|
| `c_s` | −2.00 | −1.77 ± 0.25 | −2.00, −1.98, −1.51, −1.60 |
| `c_p` | −0.50 | −0.39 ± 0.11 | −0.49, −0.48, −0.33, −0.28 |
| `c_rho` | 1.00 | 0.87 ± 0.08 | 0.86, 0.87, 0.87, 0.87 |
| share | 0.266 | 0.281 ± 0.015 | 0.27, 0.27, 0.29, 0.30 |
| jump in `dlnVs` | −1.90 | −1.94 ± 0.06 | −1.93, −1.93, −1.94, −1.95 |
| jump in `dlnVp` | −0.51 | −0.49 ± 0.03 | −0.49, −0.49, −0.50, −0.49 |
| jump in `dlnrho` | 0.94 | 0.82 ± 0.06 | 0.85, 0.87, 0.79, 0.78 |
| shear jump over compressional | 3.75 | 3.96 ± 0.20 | 3.99, 3.95, 3.87, 4.02 |

- **The provinces are dense:** `P(c_rho > 0) == 1.000`, on every chain, with R-hat
  1.04, the one answer the chains fully agree on.
- **Where they are:** a point inside the true provinces is inside the sampled ones with
  mean probability 0.95, a point outside with 0.04. All the states hold two provinces.
- **The chains have not finished:** they fall into two groups. Chains 1 and 2 have a
  mean potential of 544 and 545, against the truth's 536, and find `c_s == -2`. Chains 3
  and 4 have 593 and 576, hold a lobe of province over the North Atlantic and Europe, at
  probability 0.5 on the map, and take −1.5 and −1.6. At this difference of potential
  the first group should carry nearly all of the weight, and a longer run should find
  so; `c_s`, `c_p` and the share are not yet answers.
- **The ratio does not settle it alone:** the shear jump over the compressional is 3.96
  ± 0.20 against the thermal scaling's 1.90, but thermal fields alone, across the true
  edges, make 3.56 of the same data, which fix the jumps directly. What tells them
  apart is the misfit of part C and, once built, the evidence of part G.

## The full scale

Part A holds as at the quick scale: defects of 1e-14 on `T` and 8e-15 on the derivative,
and 1e-14 between the data of `(t + c, -phi, -c)` and `(t, phi, c)`. In part B the truth's
chi-squared is 4882 on 4770 data, and its thermal part alone 36308. The truth's thermal
draw is not the same field at the two scales, so neither are the numbers that read it,
such as its ratio of jumps.

Part C, on 4 chains of 1000 forward evaluations each, `GeometricHMC` with 40 steps of 4
leapfrog steps. The curvature runs from 1.5e4 to 4.7 by the 80th eigenvalue on the data
as given, and from 66 to 2.6 by the 10th on the weak problem. The exact posterior mean's
chi-squared on the data as given is 5506, 7.5 of its standard deviations too many.

| problem | sampler | step | acceptance | τ, regional means | τ, potential | smallest ESS | R-hat | ESS per 1000 | worst offset, sd |
|---|---|---|---|---|---|---|---|---|---|
| weak | pCN | 0.012 | 0.27 | 1047 | 948 | 3.1 | 1.77 | 0.76 | 0.36 |
| | MALA | 0.021 | 0.57 | 127 | 82 | 25 | 1.12 | 6.3 | 0.39 |
| | HMC | 0.195 | 0.72 | 8.7 | 8.3 | 46 | 1.10 | 11 | 0.15 |
| | PreconditionedHMC | 0.154 | 0.79 | 54 | 19 | 59 | 1.06 | 15 | 0.13 |
| | GeometricHMC, rank 20 | 0.393 | 0.95 | 1.4 | 0.9 | 115 | 1.02 | 144 | 0.12 |
| as given | pCN | 1.3e-4 | 0.09 | 1595 | 1457 | 2.0 | 3.18 | 0.50 | 0.97 |
| | MALA | 9.1e-5 | 0.59 | 1554 | 1406 | 2.1 | 2.64 | 0.51 | 0.93 |
| | HMC | 0.014 | 0.67 | 189 | 120 | 2.1 | 3.19 | 0.53 | 0.40 |
| | PreconditionedHMC | 0.011 | 0.78 | 1480 | 311 | 2.2 | 3.11 | 0.54 | 1.07 |
| | GeometricHMC, rank 80 | 0.388 | 0.82 | 2.2 | 1.2 | 72 | 1.03 | 89 | 0.20 |

The ladder, on the weak problem, at each sampler's step from the table and the same data
at each degree (HMC with three leapfrog steps), with a random walk of size 0.154:

| degree | unknowns | pCN | MALA | HMC | random walk of pCN's size |
|---|---|---|---|---|---|
| 12 | 507 | 0.26 ± 0.02 | 0.59 ± 0.03 | 0.72 ± 0.02 | 0.000 |
| 24 | 1875 | 0.28 ± 0.03 | 0.56 ± 0.03 | 0.76 ± 0.02 | 0.000 |
| 48 | 7203 | 0.24 ± 0.02 | 0.53 ± 0.03 | 0.74 ± 0.02 | 0.000 |

The steps the burn-in settles on are flat over the same degrees: 0.010, 0.0087 and 0.011
for pCN, 0.024, 0.025 and 0.022 for MALA, 0.222, 0.223 and 0.220 for HMC.

Part D's main run is the quick scale's, four chains of 200 + 800 sweeps, and takes 28
minutes, with steps of 0.009 and 4.5e-4 and acceptances 0.62 and 0.07. It has not
converged. The guess starts at potential 2745, and the chains hold 2582, 2580, 2624 and
2625 on average, all at least 139 above the truth's 2441, so that what the quick scale's
chains agree on is not settled here:

| scalar | truth | mean ± sd | R-hat | by chain |
|---|---|---|---|---|
| `c_s` | −2.00 | −1.54 ± 0.24 | 2.05 | −1.65, −1.79, −1.36, −1.38 |
| `c_p` | −0.50 | −0.41 ± 0.09 | 1.76 | −0.41, −0.52, −0.36, −0.36 |
| `c_rho` | 1.00 | 0.84 ± 0.13 | 1.69 | 0.96, 0.89, 0.71, 0.79 |
| share | 0.265 | 0.282 ± 0.055 | 3.65 | 0.26, 0.20, 0.34, 0.32 |
| shear jump over compressional | 4.75 | 3.95 ± 0.32 | 2.24 | 4.32, 4.00, 3.52, 3.94 |

- **What holds:** `P(c_rho > 0) == 1.000` on every chain, every state holds two provinces,
  and a point inside the true provinces is inside the sampled ones with mean probability
  0.89, a point outside with 0.06. The density contrast's sign is an answer. Its size is
  not, with R-hat 1.69 where the quick scale has 1.04, and neither is the ratio.
- **The ratio again does not settle it alone:** thermal fields alone, across the true
  edges, make 4.54 of the same data, between the chains' values and the truth's.
- **The comparisons, in 100 sweeps from the guess, with the same caution:**

| run | evaluations a sweep | squared jump of `c_s` per 1000 | of the share | potential | `c_s` | acceptance |
|---|---|---|---|---|---|---|
| 1 pCN step per visit | 17 | 0.102 | 8.9e-4 | 2745 | −0.81 | 0.63, 0.15 |
| 5 | 21 | 0.095 | 3.4e-4 | 2545 | −1.54 | 0.67, 0.07 |
| 20 | 36 | 0.056 | 1.6e-4 | 2563 | −1.60 | 0.73, 0.03 |
| symmetric scan, 5 | 42 | 0.044 | 1.5e-4 | 2583 | −1.52 | 0.66, 0.04 |
| three blocks | 18 | 0.018 | 1.4e-3 | 2655 | −0.84 | 0.71, 0.08, 0.56 |
| two blocks, from the prior | 21 | 0.082 | 7.7e-4 | 2788 | 0.00 | 0.58, 0.04 |

Three blocks move the contrast 5 times less far than two, as at the quick scale, by the
same margin and the same `c_s` left behind. The rows on the level set's visits are not
as clean: the share's jump falls steadily with steps per visit here and does not at
the quick scale.

That the 5-step row sits below the main run in potential, after a tenth of its sweeps, is
not a benefit of the shorter burn-in. Run separately with the main run's settings and
another seed, four chains after 200 sweeps of burn-in hold 2474, 2501, 2512 and 2734 over
the last 50 of 100: the gap between chains of one run is wider than the gap between those
two runs, and some chains get close to the truth's potential while others do not.
The level set's step does matter, and not as the adaptation's target suggests. Held
fixed at 0.002, 0.001, 5e-4 and 2.5e-4 over 150 sweeps, its acceptance is 0.09, 0.17, 0.21
and 0.40, and the smaller steps end at the higher potentials, the largest step's chains
holding 2512 to 2601 and the smallest's 2627 to 2765. Acceptance also falls as a chain
descends, to 0.05 at a step of 6e-4 in that adapted run against 0.21 at 5e-4 from a
fixed step higher up, and the adaptation's gain has decayed by then, so the block runs
below its target of a quarter.

## What the app shows

- A prior can couple function-valued fields mode by mode and still give a sampler its
  conditionals in closed form, on a nested direct sum.
- Function-space samplers do not notice refinement: a random walk of pCN's size accepts
  nothing at any degree, where they keep their acceptance and their step from 507
  unknowns to 7203.
- Being exact is not being fast. Every sampler here is exact, and on the data as given
  only the one whose reference holds the leading directions mixes in this many
  evaluations. An exact posterior is what lets that be said with numbers.
- How a model is cut into blocks decides what can move. The cut the fields suggest,
  the thermal fields and the contrasts apart, walks a valley the other cut does not see.
- A sharp level set is sampled, with a derivative zero in it, by a block that
  needs none, and it is the slowest part of the run, in chains R-hat duly reports as
  unfinished. What they already agree on is an answer, and what they do not is not.
  Refinement leaves the steps alone and not this: the same number of sweeps that nearly
  settles the density contrast at degree 24 settles only its sign at degree 48.
