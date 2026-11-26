# Parameters of a dynamical system

A double pendulum is released from rest at known angles. Its two masses and
two lengths are unknown. Some of its angles are read at irregular times, with
noise, and its angular rates are never seen. The problem is to say what the
readings tell about the masses and lengths.

The app is about the shape of such a problem, and the pendulum stands in for
any system of a few variables whose rule depends on a few parameters:

    dx/dt = f(x, p) + G w(t),

with `x` the state, `p` the parameters and `w` a forcing that may be absent,
known or unknown. The forward operator takes the unknowns to the readings and
is a composition of four operators, each with one job:

| operator | from | to |
|---|---|---|
| parameter map | free coordinates | parameters that are positive whatever the coordinates |
| solution operator | parameters, and a forcing | the state at the nodes of a mesh in time |
| lift | that table | a trajectory, a function of time for each component of the state |
| observation operator | a trajectory | what an instrument reports of it |

The solution operator knows nothing of the pendulum or of function spaces. It
takes a rule, integrates it with an adaptive integrator used as it comes, and
differentiates the differential equation: the linearized equation for the
derivative, the adjoint equation for its adjoint, and the sensitivity
equations for the whole Jacobian at once while the parameters are few. The
second derivative has the same pair one order up. The rule's own derivatives
are taken by the complex step, so a new system is one function.

Three instruments are provided, chosen with `--instrument`:

| instrument | what it reports |
|---|---|
| `angles` | chosen components of the state at irregular times, each with times of its own |
| `averages` | their means over windows of time |
| `bob` | the position of the lower mass, which is not linear in the angles and depends on the lengths |

## Running it

    python run_all.py --quick     # four minutes
    python run_all.py             # the full scale, most of half an hour
    python ladder.py              # one part alone
    python ladder.py --help       # every field of the scale is an option

Figures and numbers go to `results/`, or `results/quick/`.

## Part A: the forward operator and its checks (`ladder.py`)

At the full scale: six seconds of motion on a mesh of 481 nodes, thirty
readings of each angle.

| check | result |
|---|---|
| the trajectory between the nodes, against the integrator's own | 1e-5 rad in the angles |
| operators between tables, functions and readings: adjoint identity | 1e-13 or better |
| derivative against finite differences | 1e-7 |
| adjoint against derivative, by the sensitivities | 5e-15 |
| adjoint against derivative, by the adjoint equation | 2e-8 |
| adjoint against derivative, with an unknown forcing of 1098 components | 8e-9 |
| second derivative against differences of the first | 5e-7 |
| its adjoint against it | 2e-9 |

The adjoint equation's answer is as good as the integrator's tolerance, with
a forcing or without. The transpose of the sensitivities is exact.

The second figure of the part is the reason the pendulum was chosen. From the
release the data come from, the angles' sensitivity to the parameters grows
about five-fold between one second and six. From a livelier release it grows
nearly three hundred-fold, to a thousand radians per unit parameter: the
motion is chaotic, two nearby parameter values part company in a few seconds,
and a longer record is harder to fit, not easier.

Two more things the rule decides. The masses enter it through their ratio
alone, so the readings say nothing of the masses' common size. And a forcing
given at the nodes of a mesh is only as smooth as its interpolant: an
integrator of high order that strides across several nodes at once meets its
tolerance erratically, and the solution stops being a smooth function of the
forcing. With a forcing the integrator's step is capped at the nodes'
spacing.

## Part B: a point answer (`point.py`)

The unknowns are the logarithms of the four parameters, under a Gaussian
prior of standard deviation 0.3 about a pendulum whose masses and lengths
are all one. The most probable model is found by a Newton method from the
prior's expectation.

| | truth | estimate | noise carried | bias |
|---|---|---|---|---|
| `m1` | 1.0000 | 1.1166 | 0.10% | +11.80% |
| `m2` | 0.8000 | 0.8956 | 0.10% | +11.80% |
| `l1` | 1.0000 | 1.0003 | 0.14% | 0.00% |
| `l2` | 0.7000 | 0.7008 | 0.21% | 0.00% |

The exact Hessian, through the second-order adjoint, and the Gauss-Newton
one reach the same model, in 6 and 8 iterations and 23 and 2 seconds. The
lengths are found to a part in a thousand and the masses' ratio to a quarter
of a per cent. The masses' common size is not found at all: no noise reaches
it, because the readings do not depend on it, and the estimate of it is the
prior's, which is where the masses' bias of twelve per cent comes from. A
point estimate's error is the noise it carries and its bias together, and
the second is invisible without the truth.

The figure's last panel is the misfit along the straight line from the
prior's expectation to the truth. From the release the data come from it has
one valley. From the livelier release it has 32, the one at the truth too
narrow to see, and the same search on that record ends in another of them
with a chi-squared in the millions.

## Part C: a measure answer (`bayesian.py`)

The posterior, by the Laplace approximation and by sampling. Standard
deviations of the logarithms, which are relative errors:

| | `m1` | `m2` | `l1` | `l2` | masses' size | masses' ratio |
|---|---|---|---|---|---|---|
| prior | 30% | 30% | 30% | 30% | 30% | 30% |
| Laplace | 21.2% | 21.2% | 0.14% | 0.21% | 30.0% | 0.14% |
| sampled | 21.0% | 21.0% | 0.14% | 0.21% | 29.7% | 0.14% |

Along the masses' common size the posterior is the prior, to every figure
shown: the measure answer says what the point answer could only show as a
bias. Each mass alone keeps most of its prior spread for that reason, while
their ratio is as well known as the lengths.

The chains are preconditioned Crank-Nicolson steps about the Laplace
Gaussian, which need no derivative: 2400 kept steps, acceptance 0.99, R-hat
1.00, about 1700 effective draws. They start from draws of that Gaussian.
The posterior is a hundred times narrower than the prior in three directions
of four, and chains started from draws of the prior did not find it.

Each figure also shows a component the instrument never read, the first
angular rate, under thirty draws from the posterior: the readings of the
angles fix it.

## Part D: a pendulum that was pushed (`forcing.py`)

The readings now come from a pendulum under a forcing nobody recorded, an
angular acceleration of each arm drawn from a Gaussian prior with standard
deviation 0.3 rad/s^2 and a correlation time of 0.3 s.

| | `l1` | `l2` | chi-squared, 60 readings |
|---|---|---|---|
| forcing ignored: posterior sd | 0.15% | 0.20% | 1406 |
| forcing ignored: truth away by | 43 sd | 25 sd | |
| forcing unknown: posterior sd | 2.3% | 1.2% | 35 |
| forcing unknown: truth away by | 0.1 sd | 0.6 sd | |

With the forcing ignored the model of parts B and C cannot fit the readings,
and reports the lengths to a fifth of a per cent with the truth tens of
standard deviations away. With the forcing unknown the model is the pair of
the parameters and the forcing, 1102 unknowns; the search for the most
probable pair takes 15 Newton iterations, each gradient one adjoint
integration, and the lengths' posterior is ten times wider and holds the
truth.

The forcing itself is recovered where the readings constrain it: the mode's
error is half the forcing's own size, and the truth lies within two
posterior standard deviations of it at 97% of the nodes. At the end of the
record, where a push has had no time to show in the readings, the mode
returns to zero and the posterior is the prior.

The joint Gaussian is the prior with as many directions corrected as there
are readings, which is exact for the Gauss-Newton curvature and needs no
solve for its covariance.

This part is the expensive one, twenty-one minutes at the full scale against
under three at the quick one: six minutes for the search and the Gaussian,
every correction a tangent and an adjoint integration, and fifteen for the
chains. `--table 16` reads the rule's Jacobians along the state from a table
instead of from the rule, which makes the linearized integrations three
times cheaper and leaves the adjoint agreeing with the derivative to 1e-6
where the rule itself gives 1e-8.

The chains about the joint Gaussian are converged at the quick scale
(acceptance 0.86, R-hat below 1.04, the same spreads as the Gaussian's) and
not at the full one: with a step fixed at 0.5 their acceptance is 0.12 and
R-hat reaches 1.13 after 2400 steps. Their spreads for the lengths, 2.0% and
1.1%, are near the Gaussian's, and are not to be relied on until they are
run for longer or with a shorter step.
