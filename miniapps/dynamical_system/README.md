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
| observation operator | a trajectory | chosen components at chosen times |

The solution operator knows nothing of the pendulum or of function spaces. It
takes a rule, integrates it with an adaptive integrator used as it comes, and
differentiates the differential equation: the linearized equation for the
derivative, the adjoint equation for its adjoint, and the sensitivity
equations for the whole Jacobian at once while the parameters are few. The
rule's own derivatives are taken by the complex step, so a new system is one
function.

## Running it

    python run_all.py --quick     # seconds
    python run_all.py             # the full scale
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
