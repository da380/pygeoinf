"""The sampler a part runs, chosen on the command line.

Every move here samples the same posterior exactly; they differ in what a
step costs and in how far it goes. Each moves about a Gaussian reference,
the Laplace approximation, which is what lets a step be long.

``pcn``
    Preconditioned Crank-Nicolson about the Laplace Gaussian at the mode.
    A step needs the misfit alone: one integration of the system.
``mala``
    The same with a drift down the misfit's gradient: one integration and
    one adjoint a step.
``hmc``
    Hamiltonian trajectories about the same Gaussian, a gradient for each
    leapfrog step.
``local``
    A step of the Laplace Gaussian *at the present state*, rebuilt at
    every proposal, so the reference follows the chain to where the
    posterior's shape is not the one it has at the mode. A rebuild takes an
    adjoint for each reading, which makes a step far the dearest of the
    four.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pygeoinf import (
    ForwardProblem,
    GaussianMeasure,
    LaplaceApproximation,
    MonteCarloEstimator,
    Operator,
)
from pygeoinf.sampling import HMC, MALA, PCN, Chains

MOVES = ("pcn", "mala", "hmc", "local")
# The step of each move where the scale gives none; a part gives its own
# for "pcn", which depends on how well the Gaussian fits.
STEPS = {"mala": 1.0, "hmc": 0.3, "local": 1.0}


def move_for(scale: Any, pcn_step: float, /) -> tuple[Any, float]:
    """The move a scale asks for, and the step it starts with.

    Args:
        scale: the scale, read for ``sampler``, ``sampler_step``,
            ``sampler_length`` and ``sampler_adapt``.
        pcn_step: the step of ``"pcn"`` where the scale gives none.

    Returns:
        The move and its step. With ``sampler_adapt`` the step adapts
        during burn-in towards the move's own target acceptance; without,
        it is fixed.

    Raises:
        ValueError: for a sampler that is none of :data:`MOVES`.
    """
    name = scale.sampler
    if name not in MOVES:
        raise ValueError(f"The sampler is one of {', '.join(MOVES)}; got {name!r}.")
    step = scale.sampler_step
    if step is None:
        step = pcn_step if name == "pcn" else STEPS[name]
    fixed = {} if scale.sampler_adapt else {"acceptance": None}
    if name == "pcn":
        return PCN(step=step, **fixed), step
    if name == "hmc":
        return HMC(step=step, length=scale.sampler_length, **fixed), step
    return MALA(step=step, local=name == "local", **fixed), step


def sample(
    problem: ForwardProblem,
    prior: GaussianMeasure,
    data: Any,
    scale: Any,
    /,
    *,
    at_mode: LaplaceApproximation,
    start: GaussianMeasure,
    record: Operator,
    record_names: Sequence[str],
    pcn_step: float,
    rng: np.random.Generator,
) -> tuple[Any, str]:
    """Run the scale's sampler on a problem's posterior.

    Args:
        problem: the forward problem.
        prior: the prior on its model space.
        data: the readings.
        scale: the scale, read for the sampler and the chains' sizes.
        at_mode: the Laplace approximation whose Gaussian at the mode the
            moves other than ``"local"`` move about.
        start: the measure each chain draws its start from: the Laplace
            Gaussian, since a draw of the prior is too far away to start.
        record: the map whose components are kept at every step.
        record_names: their names.
        pcn_step: the step of ``"pcn"`` where the scale gives none.
        rng: the generator.

    Returns:
        The sampled posterior, and a line saying which move ran.
    """
    move, step = move_for(scale, pcn_step)
    if scale.sampler == "local":
        # The Gaussian is rebuilt at every state from the Gauss-Newton
        # curvature there, through the readings: an adjoint for each.
        reference: Any = LaplaceApproximation.from_problem(
            problem, prior, gauss_newton=True
        )
    else:
        reference = at_mode.reference(data)
    sampler = Chains(
        move, steps=scale.steps, burn_in=scale.burn_in, chains=scale.chains, rng=rng
    )
    estimator = MonteCarloEstimator(
        problem,
        prior,
        sampler=sampler,
        reference=reference,
        record=record,
        record_names=list(record_names),
    )
    posterior = estimator(data, start=start)
    ended = float(np.mean([chain.step_size for chain in posterior.chains]))
    how = "adapted to" if scale.sampler_adapt else "fixed at"
    return posterior, f"{scale.sampler}, step {how} {ended:.3g} from {step:.3g}"
