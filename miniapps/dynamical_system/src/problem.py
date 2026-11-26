"""The inverse problem every part shares: a double pendulum seen through a few angles.

A double pendulum is released from rest at known angles. Its masses and
lengths are unknown. Some of its angles are read at irregular times, with
noise; its angular rates are never seen. The forward operator takes the
unknowns to those readings and is a composition of four operators, each
with one job:

1. the **parameter map**, from free coordinates to masses and lengths that
   are positive whatever the coordinates;
2. the **solution operator**, from the masses and lengths, and a forcing if
   there is one, to the state at the nodes of a mesh in time;
3. the **lift** of that table to a trajectory, a function of time for each
   component of the state;
4. the **observation operator**, which reads chosen components of a
   trajectory at chosen times.

With an unknown forcing the model is the pair of the free coordinates and
the forcing, an angular acceleration of each arm as a function of time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pygeoinf import EuclideanSpace, GaussianMeasure, LinearOperator, Operator, interval
from pygeoinf.algebra.direct_sum import BlockDiagonalOperator
from pygeoinf.algebra.tensor_product import TensorProduct

from .parameters import ParameterMap
from .pendulum import double_pendulum
from .spaces import from_table, grid_of, observation_operator, time_space, to_table
from .system import DynamicalSystem
from .trajectory import Solver, TimeGrid, solution_operator


@dataclass(frozen=True)
class Scale:
    """The sizes of a run.

    Attributes:
        name: the preset's name.
        seed: the seed of every random draw.
        duration: how long the pendulum is watched, in seconds.
        release: the two angles it is released from, at rest, in radians.
        lively_release: a higher release, for showing how sensitivity grows.
        true_parameters: the masses and lengths behind the data, m1 m2 l1 l2.
        element_length: the longest spectral element of the mesh in time, in seconds.
        order: the Sobolev order of a function of time.
        time_scale: the time scale of its inner product, in seconds.
        rtol: the integrator's relative tolerance.
        atol: the integrator's absolute tolerance.
        observed: which components of the state are read: 0 and 1 are the angles, 2 and 3 their rates.
        n_times: how many times each is read at.
        noise_std: the standard deviation of a reading's error, in radians.
        prior_std: the prior standard deviation of the logarithm of each parameter.
        forcing_std: the prior standard deviation of the forcing at a time, in radians per second squared.
        forcing_time: the prior correlation time of the forcing, in seconds.
        forcing_order: the order of the forcing's prior covariance.
    """

    name: str
    seed: int
    duration: float
    release: tuple[float, float]
    lively_release: tuple[float, float]
    true_parameters: tuple[float, float, float, float]
    element_length: float
    order: float
    time_scale: float
    rtol: float
    atol: float
    observed: tuple[int, ...]
    n_times: int
    noise_std: float
    prior_std: float
    forcing_std: float
    forcing_time: float
    forcing_order: float


QUICK = Scale(
    name="quick",
    seed=2026,
    duration=3.0,
    release=(1.0, -0.5),
    lively_release=(2.0, 2.5),
    true_parameters=(1.0, 0.8, 1.0, 0.7),
    element_length=0.1,
    order=1.0,
    time_scale=0.2,
    rtol=1e-8,
    atol=1e-10,
    observed=(0, 1),
    n_times=15,
    noise_std=0.02,
    prior_std=0.3,
    forcing_std=0.3,
    forcing_time=0.3,
    forcing_order=2.0,
)
FULL = Scale(
    name="full",
    seed=2026,
    duration=6.0,
    release=(1.0, -0.5),
    lively_release=(2.0, 2.5),
    true_parameters=(1.0, 0.8, 1.0, 0.7),
    element_length=0.05,
    order=1.0,
    time_scale=0.2,
    rtol=1e-9,
    atol=1e-11,
    observed=(0, 1),
    n_times=30,
    noise_std=0.02,
    prior_std=0.3,
    forcing_std=0.3,
    forcing_time=0.3,
    forcing_order=2.0,
)


@dataclass
class Problem:
    """Everything the parts share."""

    scale: Scale
    system: DynamicalSystem
    parameter_map: ParameterMap
    space: interval.Sobolev
    grid: TimeGrid
    solver: Solver
    forced_solver: Solver
    trajectory: TensorProduct
    forcing_space: TensorProduct
    lift: LinearOperator
    observe: LinearOperator
    times: list[np.ndarray]
    forward: Operator
    forward_by_adjoint: Operator
    forced_forward: Operator
    noise: GaussianMeasure
    forcing_prior: GaussianMeasure
    truth: np.ndarray
    clean: Any
    data: Any

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The parameters' names."""
        return self.system.parameter_names

    def solver_for(self, release: tuple[float, float], /) -> Solver:
        """A solver from another release, on the same grid and to the same tolerances."""
        scale = self.scale
        return Solver(
            self.system,
            np.array([*release, 0.0, 0.0]),
            self.grid,
            rtol=scale.rtol,
            atol=scale.atol,
        )


def build(scale: Scale, /) -> Problem:
    """The spaces, the operators and one set of synthetic data."""
    rng = np.random.default_rng(scale.seed)
    system = double_pendulum()
    parameter_map = ParameterMap(["positive"] * system.parameter_dim)
    space = time_space(
        scale.duration, scale.element_length, scale.order, scale.time_scale
    )
    grid = grid_of(space)
    start = np.array([*scale.release, 0.0, 0.0])
    solver = Solver(system, start, grid, rtol=scale.rtol, atol=scale.atol)
    forced_solver = Solver(
        system,
        start,
        grid,
        rtol=scale.rtol,
        atol=scale.atol,
        max_step=float(np.median(np.diff(grid.nodes))),
    )

    trajectory = TensorProduct([EuclideanSpace(system.state_dim), space])
    forcing_space = TensorProduct([EuclideanSpace(system.forcing_dim), space])
    times = [
        np.sort(rng.uniform(0.02 * scale.duration, scale.duration, scale.n_times))
        for _ in scale.observed
    ]
    observe = observation_operator(trajectory, list(zip(scale.observed, times)))

    by_sensitivity = solution_operator(solver, route="sensitivity")
    by_adjoint = solution_operator(solver, route="adjoint")
    forced = solution_operator(forced_solver, forced=True)
    lift = by_sensitivity.codomain.lift(from_table(space), axis=1)
    coordinates = parameter_map.operator()
    forward = observe @ lift @ by_sensitivity @ coordinates
    forward_by_adjoint = observe @ lift @ by_adjoint @ coordinates
    front = BlockDiagonalOperator(
        [coordinates, forcing_space.lift(to_table(space), axis=1)],
        codomain=forced.domain,
    )
    forced_forward = observe @ lift @ forced @ front

    noise = GaussianMeasure.from_standard_deviation(observe.codomain, scale.noise_std)
    forcing_prior = GaussianMeasure.from_tensor_product(
        [
            GaussianMeasure.from_standard_deviation(forcing_space.factors[0], 1.0),
            space.sobolev_measure(
                scale.forcing_order,
                length_scale=scale.forcing_time,
                pointwise_std=scale.forcing_std,
            ),
        ],
        domain=forcing_space,
    )
    truth = parameter_map.coordinates(np.array(scale.true_parameters))
    clean = forward(truth)
    data = observe.codomain.add(clean, noise.sample(rng=rng))
    return Problem(
        scale=scale,
        system=system,
        parameter_map=parameter_map,
        space=space,
        grid=grid,
        solver=solver,
        forced_solver=forced_solver,
        trajectory=trajectory,
        forcing_space=forcing_space,
        lift=lift,
        observe=observe,
        times=times,
        forward=forward,
        forward_by_adjoint=forward_by_adjoint,
        forced_forward=forced_forward,
        noise=noise,
        forcing_prior=forcing_prior,
        truth=truth,
        clean=clean,
        data=data,
    )
