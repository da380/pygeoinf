"""The two preconditioned solvers the parts share.

The Woodbury surrogate for the Bayesian normal operator, where the prior's
smoothness makes a coarse sphere a good stand-in; the spectral deflation for
the damped families of the point and set answers, where it is paid once and,
the noise being white, is exact on the resolved modes at every damping.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pygeoinf2 import CGSolver, SpectralPreconditioner
from pygeoinf2.numerics.preconditioners import WoodburyPreconditioner
from pygeoinf2.numerics.solvers import CholeskySolver

from problem import ORDER, PRIOR_STD, Problem


def woodbury_solver(problem: Problem) -> Any:
    """Conjugate gradients preconditioned by the Woodbury form of a coarse surrogate.

    A factory: handed the normal operator at solve time, it replaces the
    forward operator and the prior by their versions on the coarse sphere and
    inverts the surrogate exactly through the Woodbury identity, the inner
    factorization being on the coarse space.
    """
    coarse, coarse_forward = problem.surrogate()
    coarse_prior = coarse.sobolev_measure(
        ORDER, problem.scale.length_scale, pointwise_std=PRIOR_STD
    )

    def factory(normal: Any) -> Any:
        surrogate = normal.surrogate(forward=coarse_forward, prior=coarse_prior)
        return CGSolver().with_preconditioner(
            WoodburyPreconditioner.from_normal(surrogate, solver=CholeskySolver())
        )

    return factory


def spectral_solver(
    problem: Problem, /, *, rng: np.random.Generator, n_jobs: int | None = None
) -> Any:
    """Conjugate gradients preconditioned by a randomized deflation."""
    return CGSolver(rtol=1e-6).with_preconditioner(
        SpectralPreconditioner(rank=problem.scale.spectral_rank, rng=rng, n_jobs=n_jobs)
    )
