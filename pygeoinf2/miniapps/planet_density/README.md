# A planet's density

## The problem

The radial density of the Earth from two numbers, its mass and its moment of
inertia. The model space is PREM's own layering. Every boundary between two
of PREM's layers in planetmodel is a boundary of the space. The three layers
thinner than 20 km, the lower crust (9.4 km), the upper crust (12 km) and the
ocean (3 km), are too thin to carry a few basis functions each. They are
merged into one crust, which therefore holds PREM's jumps at 6356 and
6368 km. That leaves eleven layers: the inner core, the outer core, the
lowermost mantle, the lower mantle, the layer above it to 670 km depth, the
transition zone's three layers, the low-velocity zone, the lid and the
crust. PREM's density jumps at six of the ten boundaries; at 3630, 5600,
5771 and 6291 km only its gradient changes.

Each layer is the Sobolev space `H^2` under the volume measure `4 pi r^2 dr`,
with a length scale of 200 km; the crust's is a quarter of its thickness
(see below). A profile is smooth within each layer and may jump between
them. Each layer is held by the leading eigenfunctions of its Sobolev
operator on a padded mesh. Its basis functions number 16 in the inner core,
24 in the outer core and 32 in the lower mantle, and 3 to 5 in each thin
layer: 104 in all at the full scale. Both data are linear in the density,

```
M = int rho dV,        I = (2/3) int rho r^2 dV,
```

so the forward operator is a two-row matrix, assembled by Gauss-Legendre
quadrature, exact on the basis. Every answer is asked the same three
questions:

- the mean density of the core, both core layers together;
- the mean density of the mantle, everything above the core-mantle boundary,
  crust included;
- the jump in density at the core-mantle boundary, core less mantle.

**The reference is PREM in the space.** Within every layer but the crust,
PREM's density is smooth, and there it is fitted over the layer's domain by
the layer's basis functions. At the full scale this holds PREM to within
`4.4e-4` g/cm^3, and to within `1e-4` across the transition zone. The crust
holds two of PREM's jumps, which a fit would chase into the mesh's padding.
So there PREM is sampled at the nodes, the padding given the value at the
nearer end, and the result settled into the span. The crust's length scale
is a quarter of its thickness, so that the padding the sample is continued
into is no wider than the crust. Its three basis functions cannot follow the
jumps from 2.9 to 2.6 and 1.02 g/cm^3: the crust misses PREM by up to 0.84
g/cm^3 and its mass by 0.1%. The reference's mass and moment of inertia
agree with PREM's own to `1.2e-5`, and its three properties, 10.987, 4.449
and 4.337, are PREM's.

**The truth is the reference plus a draw from the tight prior** of part B, a
Sobolev measure about the reference with a pointwise standard deviation of
0.1 g/cm^3 in every layer, independent between the layers. The data are the
truth's mass and moment of inertia, with Gaussian errors of 0.1% of each
drawn from the error measure. They are carried in units of `10^24 kg` and
`10^37 kg m^2`, so that both are of order one.

PREM is then a reference that is wrong by a known amount. The tight prior
is honest about how much, and part C's wide prior is deliberately loose.
The truth's mass is 1.25% below PREM's and its moment 1.20%. Its properties
are 10.919, 4.379 and 4.248. The data fit the truth to a chi-squared of 3.09
and reject the reference at 265.8. Every answer can be marked against the
truth.

Units: radii in km and densities in g/cm^3 throughout.

## The parts

- **A, the setup** (`setup.py`): the space, the reference, the truth, the
  data, the property operator. Its figure shows PREM, the reference and the
  truth, with the layers shaded, over the whole radius and over the upper
  mantle and the crust.
- **B, a prior tight about PREM** (`tight.py`): the prior the truth was
  drawn from, and the three answers in order.
  - The **point**: the smallest change to PREM that fits the data at the
    discrepancy level, measured in the prior's precision. It is computed
    by `MinimumNorm` on the problem parameterized by the prior's covariance
    factor. Its error budget on the three properties is the noise it
    propagates and its bias at the truth.
  - The **measure**: the posterior against the prior, as bands on the
    profile and marginals on the three properties, and the log evidence.
  - The **set**: the prior hardened to its 95% ball in the model space's
    norm, and the noise to its 95% ball. Their certified brackets on the
    three properties are drawn against the bound the prior ball admits
    alone.
- **C, a deliberately wide bound** (`wide.py`): a ball in the model space's
  norm whose radius is the 95% radius of the same measure with a pointwise
  standard deviation of 3 g/cm^3, and that measure as the Gaussian. The same
  three answers follow, in the same order.
- **D, the boundaries** (`boundaries.py`): the inner-core and core-mantle
  boundary radii as further unknowns, a weakly nonlinear problem on
  `DirectSum([space, EuclideanSpace(2)])`.
  - **The forward model.** Moving a boundary out by `delta_k` turns a shell
    of thickness `delta_k` from the material above it into the material
    below it. To first order the mass gains `-4 pi r_k^2 [rho]_k delta_k`
    and the moment `-(8 pi / 3) r_k^4 [rho]_k delta_k`. Here `[rho]_k` is
    the jump, the value above less the value below, and `r_k` the mesh
    radius; the mesh does not move.
  - **The checks.** The operator is bilinear in the density and the
    displacements. Its derivative, the derivative's adjoint and its second
    derivative are checked against finite differences at draws from the
    prior.
  - **The truth and the data.** The truth is part A's with the core-mantle
    boundary 30 km out and the inner-core boundary 20 km in, and the data
    are its own through this operator.
  - **The prior.** B's on the density, independent of a standard normal on
    the displacements.
  - **The estimates.** The MAP model, by Newton-CG on the Gauss-Newton
    Hessian preconditioned by the prior covariance, and the Laplace
    Gaussian about it. Then Occam's model at the 95% level twice, from the
    same reference, PREM at its own radii, in two norms. The first is the
    model space's own. The second is the prior's Mahalanobis norm: the
    model is written `m0 + L w`, with `L` the prior's covariance factor, and
    Occam asks for the smallest `w`, as the point answers of parts B and C
    do.
  - **The displacement unit.** The displacements are carried in units of
    `RADIUS_SCALE`, 50 km. A unit is then one standard deviation of the
    displacements' prior, as 0.1 g/cm^3 is of the density's, and that is
    the statement the MAP model and the Laplace Gaussian rest on.
  - **The two norms.** The model space's norm adds the squared
    displacements, in units of 50 km, to the squared `H^2` norm of the
    density change, a volume integral in km^3. There a displacement costs
    next to nothing:
    - one unit at the core-mantle boundary changes the mass by 0.55%, as
      the smallest density change of norm `3.9e4` would;
    - the tight prior's 95% ball has radius `3.5e5`;
    - in kilometers it would be cheaper again.

    The prior's norm measures both parts in the prior's own standard
    deviations, so a unit of displacement costs what a unit of density
    departure does.

In parts B and C every figure is one method's, titled with the part and what
it shows, each with the truth marked:

- the point answer with its error budget;
- the measure answer as the prior against the posterior;
- the set answer as the prior ball's bound against the bracket the data
  leave.

Part D's one figure shows the density and the displacements of the MAP
model and of Occam's model in the two norms, side by side.

Two scales. `--quick` keeps 12 basis functions in each core layer and the
lower mantle and 3 in each thin layer, 60 in all; the full scale keeps 104.
They run every part in about 5 and 6 seconds on an AMD Ryzen AI 7 PRO 350
with Python 3.12, numpy 2.4 and planetmodel 1.3.2.

## How to run

```
cd pygeoinf2/miniapps/planet_density
MPLBACKEND=Agg python run_all.py            # every part, full scale
MPLBACKEND=Agg python run_all.py --quick    # every part, small scale
MPLBACKEND=Agg python tight.py              # one part; likewise setup.py,
                                            # wide.py, boundaries.py
```

Every script takes `--quick`, `--outdir` and `--seed`, the seed 2027 by
default. The seed draws the truth's departure from PREM and the data's
noise, the points at which part D checks its derivatives, and part D's
noise. The numbers below are one realization. With another seed, the
truth's departure can leave the mass and the moment of inertia nearly where
PREM's are, and PREM may then fit the data, in which case the point answer
is PREM itself. A part run on its own rebuilds the problem from the seed and
gives the same numbers. Figures are written next to the sources (`quick/` at
the quick scale). The test is `pygeoinf2/tests/test_miniapp_planet_density.py`,
marked `slow`, which runs the quick scale of every part through its script's
entry point.

| file | what it is |
|---|---|
| `reference.py` | PREM through planetmodel: the layering, the density layer by layer, and PREM's own mass, moment and properties by quadrature |
| `problem.py` | the two scales, the layered space, the reference and the truth, the forward and property operators, the data, the priors, `build(scale=)` |
| `answers.py` | the point, measure and set answers under one prior, each with its figure |
| `common.py` | the command line, profiles and their bands, the titles |
| `radii.py` | part D's forward operator, the density and the two boundary displacements to the mass and the moment, with its derivatives and their check |
| `setup.py`, `tight.py`, `wide.py`, `boundaries.py` | parts A to D, each a script with a `main()` and a `run()` the others and the test call |
| `run_all.py` | the parts in order |

## The answers at the full scale

The three properties in g/cm^3. The truth's are 10.919, 4.379 and 4.248;
PREM's, the reference's, are 10.987, 4.449 and 4.337.

| | core mean | mantle mean | CMB jump |
|---|---|---|---|
| B point: estimate, propagated sd | 10.970 ± 0.003 | 4.394 ± 0.003 | 4.326 ± 0.002 |
| B point: bias at the truth | +0.046 | +0.011 | +0.075 |
| B measure: prior | 10.987 ± 0.069 | 4.449 ± 0.046 | 4.337 ± 0.140 |
| B measure: posterior | 10.953 ± 0.056 | 4.386 ± 0.011 | 4.316 ± 0.138 |
| B set: the prior ball alone | [10.229, 11.745] | [4.172, 4.726] | [2.932, 5.742] |
| B set: with the data | [10.429, 11.436] | [4.301, 4.478] | [2.949, 5.696] |
| C point | as B's | as B's | as B's |
| C measure: prior | 10.99 ± 2.07 | 4.45 ± 1.37 | 4.34 ± 4.20 |
| C measure: posterior | 10.95 ± 1.51 | 4.386 ± 0.294 | 4.31 ± 4.11 |
| C set: the prior ball alone | [-11.76, 33.73] | [-3.87, 12.76] | [-37.8, 46.5] |
| C set: with the data | [-1.62, 23.49] | [1.96, 6.82] | [-37.5, 46.1] |
| D: Laplace about the MAP model | 10.973 ± 0.064 | 4.405 ± 0.018 | 4.330 ± 0.139 |

The brackets are certified to gaps of `4.1e-4` or less under the tight
prior and `1.3e-2` or less under the wide one. The truth is inside every
posterior interval and every bracket, and inside both priors' balls.

**B** (about 3 s).

- **The point.** The damping search takes 12 evaluations and lands at a
  chi-squared of 5.99, the critical value, from 265.8 at PREM. The
  smallest change to PREM moves the three properties by -0.017, -0.055 and
  -0.011, each towards the truth: most of the way for the mantle mean
  (0.055 of 0.070), a quarter of it for the core mean and an eighth for the
  jump.
  - The estimate misses the truth by +0.051, +0.015 and +0.079. Almost all
    of that is the bias at the truth, +0.046, +0.011 and +0.075; the noise
    it propagates has a standard deviation of 0.002 to 0.003.
  - Its error budget accounts for the error, and its propagated noise, the
    part a reader of the estimate would quote, is the smallest part of it. The rest is
    the part of the truth's departure that the smallest change does not
    make.
- **The measure.** The posterior mean fits the data to a chi-squared of
  0.07. The data narrow the mantle mean from 0.046 to 0.011, the core mean
  only from 0.069 to 0.056, and leave the jump alone; the two means are
  correlated at -0.85 through the mass. Pointwise, the posterior band is
  narrowed most in the lower mantle, to about two thirds of the prior's,
  and hardly in the thin layers. The log evidence is 3.94.
- **The set.** The prior's 95% ball, of radius `3.47e5` in the model space's
  norm, holds the truth. With the data the brackets on the two means narrow
  from 1.52 to 1.01 and from 0.55 to 0.18 g/cm^3, and the jump's barely
  moves.

**C** (about 2.5 s).

- **The point** is B's, since the minimum-norm change sees the prior's
  shape and not its size.
- **The measure.** Two data cut a slice through a 104-dimensional space.
  - The mass fixes the volume-weighted sum of the two means, so the
    posterior on that pair is a thin ellipse along the line of constant
    mass, correlated at -0.9998.
  - The moment of inertia says how the mass is spread with radius rather
    than how it splits, and shortens the ellipse only a little: the core
    mean's standard deviation falls from 2.07 to 1.51.
  - The mantle mean, which carries most of the volume, falls from 1.37 to
    0.29, and the jump is untouched.
  - The data prefer the tight prior, from which the truth was drawn, by
    5.54 nats: both fit, and the wide prior spreads itself over profiles
    the data rule out.
- **The set.** Under a ball of radius `1.04e7` the brackets are wider again
  than the posterior's intervals, as a norm bound is: it admits the profile
  that spends its whole budget on one property, which the Gaussian calls
  vanishingly unlikely.

**D** (under a second).

- **The checks.** The derivatives agree with central differences, and the
  adjoint with its identity, to `3.7e-9` at three prior draws.
- **The data.** They reject PREM at its own radii (chi-squared 229.4) and
  the truth's density at PREM's radii (6.2), against a critical 5.99, and
  fit the truth (1.76). The truth's density departure moves the data far
  more than its displacements do.
- **The MAP model** takes 6 Newton iterations and fits the data to a
  chi-squared of 0.01. The Laplace Gaussian about it:

  | | prior | Laplace about the MAP model | truth |
  |---|---|---|---|
  | ICB displacement (km) | 0 ± 50 | -0.4 ± 50.0 | -20 |
  | CMB displacement (km) | 0 ± 50 | -30.0 ± 30.8 | +30 |

  - The data say the Earth is lighter than PREM. The prior allows two ways
    to make it so, lighter material, or a core-mantle boundary moved in so
    that mantle replaces core. The MAP model takes some of each: a mantle
    0.044 g/cm^3 lighter and the boundary 30 km in.
  - The truth took the first way further and moved the boundary out. The
    two data cannot tell these apart; the Laplace standard deviation of the
    CMB's displacement, 31 km against the prior's 50, says so.
  - The truth lies 1.95 of those standard deviations from the MAP model, at
    the edge of its 95% interval; the ICB's displacement stays at the
    prior's spread.
- **Occam, in two norms.** Both runs start from PREM at its own radii and
  meet the same level, a chi-squared of 5.99, in 11 and 8 outer iterations.
  The sizes below are of the change from PREM, measured both ways.

  | | the model space's norm | the prior's norm |
  |---|---|---|
  | ICB displacement (km) | +1547 | -0.3 |
  | CMB displacement (km) | -132 | -24.0 |
  | core mean | 10.987 | 10.976 |
  | mantle mean | 4.449 | 4.411 |
  | CMB jump | 4.337 | 4.331 |
  | the density change, model space's norm | 0.056 | `5.4e4` |
  | the whole change, model space's norm | 31.1 | `5.4e4` |
  | the whole change, prior's norm | 31.1 | 0.98 |

  - Each answer is the smaller change in its own norm, and the larger, by
    far, in the other's. Both fit the data equally well.
  - **In the model space's norm** a displacement is by far the cheapest
    way to change the mass. Occam leaves the density as PREM's and moves
    the inner-core boundary out by 1547 km and the core-mantle boundary in
    by 132 km. That puts the inner-core boundary at 2769 km, more than
    twice its radius and far beyond where the first-order model holds.
  - **In the prior's norm**, where a unit of displacement costs what a unit
    of density departure does, Occam makes the mantle 0.038 g/cm^3 lighter
    and moves the core-mantle boundary 24 km in. That is close to the MAP
    model, which weighs the same two costs.
  - The two answers' displacements differ by more than 100 km at both
    boundaries, more than three Laplace standard deviations.
  - "Simple" depends on the norm. On a direct sum of spaces, the relative
    scaling of the summands' norms decides what simple means: here, whether
    a kilometer of boundary is cheap or dear against a g/cm^3 of density.
    The data do not fix the norm; both answers fit them to the same level.
    So an Occam answer is a statement about the norm chosen as much as
    about the Earth.

## What the app shows

- **What two numbers say through a function space.** The mass pins the
  Earth's mean density, the moment of inertia says a little about how the
  mass is spread with radius, and nothing pins the jump at the core-mantle
  boundary. The posterior says it as a thin ellipse along the line of
  constant mass. The feasible set says it as brackets that are wider again.
  Both say it more plainly the less the prior says.
- **The point answer moves only as far as the data demand.** The smallest
  change from a reference that is wrong by a known amount meets the
  discrepancy level and stops. Its error is then mostly a bias the
  propagated noise does not show. Its answer does not depend on the prior's
  width, only on its shape.
- **The prior's width.** Under the prior the truth was drawn from, the
  measure and the set hold the truth: the posterior's 95% intervals are
  0.04 wide on the mantle mean and 0.22 on the core's. Under a deliberately
  wide one they widen in the directions the data do not see, and keep the
  volume-weighted mean the mass fixes. The evidence prefers the honest
  prior.
- **What "simple" means.** With the boundaries free, the data trade a
  boundary's displacement against the density on either side of it. The
  MAP model splits the difference as the prior weighs the two. Occam's
  model depends on the norm it is asked to be small in:
  - in the model space's norm, where the summands' scales make a boundary
    almost free, it moves the boundaries alone and further than the model
    can follow;
  - in the prior's norm it lands near the MAP model.

  Both fit the data to the same level. The norm on a direct sum is a choice
  the data do not make, and an Occam answer reports that choice as much as
  the Earth.
