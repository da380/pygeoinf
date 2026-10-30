"""The preconditioned solver the parts share: the Woodbury form of a surrogate.

The prior is smooth, so a sphere of a third the degree carries nearly every
mode the data resolve, and the Woodbury identity inverts the surrogate's
normal operator exactly through one Cholesky factorization on the coarse
space. For the Bayesian operator ``A Q A* + R`` the surrogate is read off the
operator itself; for the damped families ``A C A* + t R`` that the point and
set answers sweep, the pieces are given by hand, the kernel ``A_c* R^-1 A_c``
is read once and the factorization is redone at each multiplier.
"""

from __future__ import annotations

from typing import Any

from pygeoinf import CGSolver, GaussianMeasure, LinearOperator
from pygeoinf.numerics.preconditioners import WoodburyPreconditioner
from pygeoinf.numerics.solvers import CholeskySolver, LinearSolver

from .problem import Problem


def coarse_prior(problem: Problem, /) -> GaussianMeasure:
    """The prior's version on the coarse sphere: the same order, length scale
    and pointwise standard deviation."""
    coarse, _ = problem.surrogate()
    scale = problem.scale
    return coarse.sobolev_measure(
        scale.prior_order, scale.length_scale, pointwise_std=scale.prior_std
    )


def woodbury_solver(problem: Problem) -> Any:
    """Conjugate gradients preconditioned by the Woodbury form of a coarse surrogate.

    A factory: handed the normal operator at solve time, it replaces the
    forward operator and the prior by their versions on the coarse sphere and
    inverts the surrogate exactly through the Woodbury identity, the inner
    factorization being on the coarse space.
    """
    _, coarse_forward = problem.surrogate()
    prior = coarse_prior(problem)

    def factory(normal: Any) -> Any:
        surrogate = normal.surrogate(forward=coarse_forward, prior=prior)
        return CGSolver().with_preconditioner(
            WoodburyPreconditioner.from_normal(surrogate, solver=CholeskySolver())
        )

    return factory


def woodbury_family_solver(
    problem: Problem,
    /,
    *,
    prior_covariance: LinearOperator,
    prior_precision: LinearOperator,
    noise_covariance: LinearOperator,
    noise_precision: LinearOperator,
) -> LinearSolver:
    """Conjugate gradients with the Woodbury surrogate along a damped family.

    The preconditioner is family-aware: the estimators that sweep
    ``A C A* + t R`` hand it the family once and ask it for each multiplier.
    The prior pieces are on the coarse sphere (``C`` the coarse prior's
    covariance for the least-squares and minimum-norm families, the shape of
    the coarse prior's ball for the set answer); the noise pieces are on the
    shared data space.
    """
    _, coarse_forward = problem.surrogate()
    return CGSolver().with_preconditioner(
        WoodburyPreconditioner(
            coarse_forward,
            prior_covariance,
            noise_covariance,
            prior_inverse=prior_precision,
            noise_inverse=noise_precision,
            solver=CholeskySolver(),
        )
    )
