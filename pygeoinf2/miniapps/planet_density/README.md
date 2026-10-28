# A planet's density

## The problem

The radial density of the Earth from two numbers, its mass and its moment of
inertia. The data are real, `M = 5.972e24 kg` and `I = 8.034e37 kg m^2`,
with errors of 0.1% of each that are stated here rather than measured. The
structure is real too: the reference density is integrated in the app from
PREM's seismic velocities, and PREM's own density is the yardstick every
answer is read against. PREM is a model fitted to far more than these two
numbers; it is not the truth, and nothing here is marked against one.

The problem is deliberately underdetermined. The model space is three layers
of radial profiles, inner core, outer core and mantle, with boundaries at
PREM's 1221.5 and 3480 km. Each layer is the Sobolev space `H^2` with a
length scale of 200 km under the volume measure `4 pi r^2 dr`, so a profile
is smooth within its layer and may jump at the two boundaries. The mantle
runs to the surface: the crust and the ocean are mantle here. Both data are
linear in the density,

```
M = int rho dV,        I = (2/3) int rho r^2 dV,
```

so the forward operator is a two-row matrix on 88 coefficients at the full
scale. Every answer is asked the same three questions: the mean density of
the core (both core layers together), the mean density of the mantle, and
the jump in density at the core-mantle boundary, core less mantle. The mass
is exactly the volume-weighted sum of the first two; nothing in the data
touches the third directly.

Units: radii in km and densities in g/cm^3 throughout; the data are carried
in units of `10^24 kg` and `10^37 kg m^2`, so that both are of order one.
The datum for `I` is the polar moment; PREM, which is spherical, has
`8.020e37 kg m^2`, 1.7 standard deviations below it, and a mass 0.2 above.

**The reference** (`reference.py`) is the Adams-Williamson profile,
`d rho / dr = -rho g / Phi` with `Phi = Vp^2 - 4 Vs^2 / 3` from PREM's `vpv`
and `vsv`, integrated from the surface down with PREM's density jumps added
by hand at the core-mantle boundary (4.337 g/cm^3) and the inner-core
boundary (0.597 g/cm^3). The gravity is that of the profile's own mass,
`g = G M(r) / r^2`, and the integration is repeated until the profile stops
changing, 21 times, so that `M` vanishes at the center and nothing is
singular there. Integrating the pair `(rho, M)` down from the known total
mass instead leaves `4.3e23 kg` below the inner-core boundary, where PREM
has `1.0e23`, and the gravity of that mass drives the density to nothing
before the center. The surface density is the top of PREM's lid, 3.381
g/cm^3, and the velocities above the Moho are held at the lid's.

The reference agrees with PREM down to the transition zone and parts from it
there, because the integration cannot know the phase changes at 410 and 660
km that make PREM's mantle denser: it is 3.7% light above 660 km, 14% light
at the base of the mantle, and the core, hung from that base by PREM's jump,
8 to 10% light. Its mass is 9.0% short of the datum and its moment of
inertia 7.7%: 90 and 77 standard deviations. Every part has to deal with
that, and each kind of answer deals with it in its own way.

## The parts

- **A, the setup** (`setup.py`): the space, the references, the data, and a
  figure of the reference, PREM's density and PREM's velocities.
- **B, an effectively non-informative prior** (`uninformative.py`): a
  Sobolev prior about the reference with a pointwise standard deviation of 3
  g/cm^3 in every layer. The posterior on the three properties, and the
  feasible set under the prior's 95% ball as brackets.
- **C, a prior tight about the reference** (`tight.py`): the same with 0.1
  g/cm^3. The posterior, the minimum-norm change to the reference with its
  error budget, the evidence against B's prior, and the feasible set, all
  beside B's answers.
- **D, a mixture** (`mixture.py`): two references, the one above and one
  with a 6% density jump at 660 km (5711 km radius) put into the
  integration, with equal prior weights; the posterior weights at B's and at
  C's prior strength.
- **E, the boundaries** (`boundaries.py`): the two boundary radii as further
  unknowns, a nonlinear problem; the mode, the Laplace Gaussian about it,
  and Occam's smallest model.

Both priors are layered Sobolev measures of order 2 with the space's length
scale, calibrated to the stated pointwise standard deviation in every layer
and independent between layers; each is assembled once as a dense matrix,
which on 88 dimensions is cheap and gives it the precision the MAP
estimator needs.

Two scales. `--quick` keeps 12 modes in each layer; the full scale keeps 16,
24 and 48. The posterior answers agree between the two to the third decimal;
the brackets, whose ball grows as modes are added, widen by about 10%. Both
run every part in a few seconds, 3.7 s and 5.3 s on an AMD Ryzen AI 7 PRO
350 with Python 3.12, numpy 2.4 and planetmodel 1.3.2.

## How to run

```
cd pygeoinf2/miniapps/planet_density
MPLBACKEND=Agg python run_all.py            # every part, full scale
MPLBACKEND=Agg python run_all.py --quick    # every part, small scale
MPLBACKEND=Agg python tight.py              # one part; likewise setup.py,
                                            # uninformative.py, mixture.py,
                                            # boundaries.py
```

Every script takes `--quick`, `--outdir` and `--seed`; the seed only draws
the points at which part E checks its derivatives. A part run on its own
rebuilds the problem, and C, which compares with B, recomputes what it
needs. Figures are written next to the sources (`quick/` at the quick
scale). The test is `pygeoinf2/tests/test_miniapp_planet_density.py`,
marked `slow`, which runs the quick scale of every part through its
script's entry point.

| file | what it is |
|---|---|
| `reference.py` | PREM through planetmodel, the self-consistent Adams-Williamson integration, PREM's mass, moment and properties by quadrature |
| `problem.py` | the two scales, the layered space, the forward and property operators, the priors, `build(scale=)` |
| `common.py` | the command line, profiles with their bands, the property panels |
| `setup.py`, `uninformative.py`, `tight.py`, `mixture.py`, `boundaries.py` | parts A to E, each a script with a `main()` and a `run()` the others and the test call |
| `run_all.py` | the parts in order |

## The answers at the full scale

The three properties in g/cm^3. PREM's are 10.987, 4.449 and 4.337; the
reference's 10.017, 4.044 and 4.337.

| | core mean | mantle mean | CMB jump |
|---|---|---|---|
| B: prior (std 3) | 10.02 ± 2.07 | 4.04 ± 1.90 | 4.34 ± 4.24 |
| B: posterior | 10.66 ± 1.67 | 4.512 ± 0.324 | 4.23 ± 4.23 |
| B: bracket | [1.44, 20.66] | [2.57, 6.30] | [-24.3, 33.0] |
| C: prior (std 0.1) | 10.017 ± 0.069 | 4.044 ± 0.063 | 4.337 ± 0.141 |
| C: posterior | 10.547 ± 0.058 | 4.501 ± 0.011 | 4.237 ± 0.141 |
| C: minimum-norm change, propagated sd | 10.971 ± 0.030 | 4.436 ± 0.004 | 4.327 ± 0.002 |
| C: its bias, were PREM the truth | +0.021 | -0.019 | -0.005 |
| C: bracket | empty | empty | empty |
| E: Laplace about the mode | 10.477 ± 0.059 | 4.436 ± 0.015 | 4.402 ± 0.141 |

- **B** (2.2 s): the posterior mean fits both data. The data cut the
  mantle mean's standard deviation from 1.90 to 0.32 and the core's only
  from 2.07 to 1.67, and they leave the jump alone. The core and mantle
  means are correlated at -0.9998: their volume-weighted sum, the Earth's
  mean density, is known to 0.0055 g/cm^3, which is the mass's 0.1% and
  nothing more. Along that line the moment of inertia does little. It says
  how the mass is spread with radius, and a mantle denser at depth does that
  as well as a heavier core, so the core mean narrows only from 2.07 to
  1.67. A two-dimensional constraint through an 88-dimensional space. The brackets, under a ball of radius `7.1e6` in the
  space's norm, are wider again, as a norm bound is: it admits the profile
  that spends its whole budget on one property, which the Gaussian calls
  vanishingly unlikely. PREM is inside every interval and every bracket.
- **C** (2.0 s): the data and the prior disagree. Over a whole mantle a
  pointwise 0.1 g/cm^3 averages down to 0.06 in the mean, and the reference
  is 0.4 short of PREM's. The posterior moves the core and mantle means 7.7
  and 7.2 prior standard deviations from the reference towards the data and
  still misses them, by 4.9 standard deviations in the mass and 2.9 in the
  moment; its narrow intervals miss PREM by 7.6 and 4.5 of its own
  deviations. The evidence falls by 102 nats against B's prior. The smallest
  change to the reference that meets the discrepancy level (`chi^2 = 5.99`)
  lands within 0.02 of PREM's means. The data fix their volume-weighted sum
  and little more, as B shows; how close the rest comes is the shape the
  norm prefers. Its propagated noise is smaller than its bias at PREM in the
  mantle mean and the jump, and the budget cannot see the bias. The feasible set under the tight prior's 95%
  ball is empty: the nearest model that fits the data lies 2.65 radii out.
- **D** (0.4 s): the jump at 660 km adds 4.7% to the mass and 4.2% to the
  moment, since everything below it is heavier.

  | prior std | component | log evidence | data from its predictive (Mahalanobis) | posterior weight |
  |---|---|---|---|---|
  | 3 | no jump | -2.41 | 0.5 | 0.473 |
  | 3 | jump at 660 km | -2.30 | 0.3 | 0.527 |
  | 0.1 | no jump | -104.88 | 14.8 | 5e-36 |
  | 0.1 | jump at 660 km | -23.61 | 7.5 | 1.000 |

  Under the loose prior the data cannot tell the references apart. Under the
  tight one they give all the weight to the reference with the jump, because
  it is the heavier, and both components are still rejected against B's
  prior. Two numbers can say which reference carries the mass and the moment
  the Earth has; they cannot say where in the mantle the mass sits. In the
  mantle the two components' posterior means come within 0.14 g/cm^3 of
  each other, from references 0.39 apart below 660 km, and each keeps its
  prior's shape at 660 km; in the core, which the tight prior holds nearer
  each reference, they stay up to 0.6 apart.
- **E** (0.3 s): at three draws from the prior the derivative and the second
  derivative agree with central differences, and the derivative's adjoint
  with its defining identity, to `4.6e-9`. The mode takes 5 Newton
  iterations, preconditioned by the prior covariance,
  and fits the data to `chi^2 = 20.8`, against a critical 5.99: the tight
  density prior still resists. It moves the core-mantle boundary out by 115
  km, 5.8 prior standard deviations, and the inner-core boundary by 2 km.
  The Laplace standard deviations of the two shifts, 20.0 and 18.0 km, are
  close to the prior's 20: the data barely see the boundaries. The CMB moves
  out because turning mantle into core adds mass that the tight density
  prior will not give; it is the reference's mass deficit, not information
  about the boundary.
  Occam meets the discrepancy level in 15 outer iterations, by moving the
  inner-core boundary 17 908 km in and the core-mantle boundary 1 160 km out.

## What the app shows

- What two numbers say through a function space: the mass pins the Earth's
  mean density, the moment of inertia says a little about how the mass is
  spread with radius, and nothing pins the jump at the core-mantle boundary.
  The posterior says it as a thin ellipse, the feasible set as brackets that
  are wider again.
- A tight prior about a reference the data reject is caught by every kind of
  answer, each in its own currency: a posterior that moves many deviations
  and still misses the data, an evidence far below the loose prior's, a
  point estimate whose bias is invisible to its own error budget, and an
  empty feasible set.
- The evidence weighs references by the mass and moment they predict, and
  no more: under a shared prior covariance a component's evidence depends on
  its mean only through the data it predicts. A jump at 660 km gets the
  weight because it adds mass; any reference predicting the same two numbers
  would have scored the same, wherever it carried them.
- **The boundary model is first order and the mesh does not move.** A shift
  `delta_k`, positive outwards, turns a shell of thickness `delta_k` from the
  material above the boundary into the material below it, so the mass gains
  `-4 pi r_k^2 [rho]_k delta_k` and the moment `-(8 pi / 3) r_k^4 [rho]_k
  delta_k`, with `[rho]_k` the value above less the value below and `r_k`
  the mesh radius. The terms it drops are of relative size about
  `delta_k / r_k`, 3% for the mode's 115 km.
- **Occam's answer is the norm's.** Occam asks for the model nearest the
  reference in the model space's own norm, which on `(density, radii)` adds
  the squared kilometers of the shifts to the squared `H^2` norm of the
  density change, a volume integral in km^3. On that scale a kilometer is
  almost free, so Occam explains the data with the boundaries, by distances
  that leave the first-order model, and the planet, far behind. Nothing in
  the model space says what a kilometer of boundary is worth against a
  density; the prior does, with its 20 km, and that is the difference
  between the mode and Occam's model here.
- Profiles are put into the space by fitting the kept modes on each layer's
  domain, which is accurate there: the reference's projected mass and moment
  agree with the integration's to `1e-6`. With 48 modes in the mantle, a
  profile that jumps inside the layer, as PREM's density and the second
  reference do, is fitted with large values in the mesh's padding outside
  the domain. Every operator here reads the domain alone and every norm is
  taken of a change from the reference, so no answer depends on them; a
  norm of a projected profile itself would.
