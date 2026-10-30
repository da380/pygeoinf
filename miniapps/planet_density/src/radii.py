"""The boundary radii as unknowns: the forward operator of part D.

The unknown is the density together with ``s``, the outward displacements of
the inner-core boundary and the core-mantle boundary from their places in
the mesh, in units of the scale's ``radius_scale``, 50 km by default. The
mesh does not move. Moving a boundary out by ``delta_k`` turns a shell of
thickness ``delta_k`` from the material above it into the material below
it, so to first order

    M(rho, s) = int rho dV - sum_k 4 pi r_k^2 [rho]_k delta_k,
    I(rho, s) = (2/3) int rho r^2 dV - sum_k (8 pi / 3) r_k^4 [rho]_k delta_k,

with ``delta_k = h s_k``, ``h`` the ``radius_scale`` in the space's unit of
radius, the Earth's (50 / 6371 by default), ``[rho]_k`` the jump at
interface ``k``, the value above less the value below, and ``r_k`` the mesh
radius, in Earth radii too. That is bilinear: the derivative in ``rho`` is
the linear part plus the jumps weighted by the displacements, the
derivative in ``s_k`` is the number ``-4 pi r_k^2 [rho]_k h``, and the
second derivative couples the jump of one direction with the displacement
of the other and is otherwise zero. :func:`check` puts all three against
finite differences.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pygeoinf import (
    DirectSum,
    EuclideanSpace,
    GaussianMeasure,
    LinearOperator,
    Operator,
    testing,
)

from .problem import TO_MASS, TO_MOMENT, Problem
from .reference import R_CMB, R_ICB, R_SURFACE


def boundary_operator(problem: Problem, /) -> tuple[DirectSum, Operator]:
    """The model space ``(density, displacements)`` and the bilinear forward
    operator on it."""
    space = problem.space
    model_space = DirectSum([space, EuclideanSpace(2)], labels=("density", "radii"))
    data_space = problem.forward.codomain
    linear = problem.forward.matrix(form="components")
    jumps = problem.jumps
    radii = np.array([R_ICB, R_CMB]) / R_SURFACE
    # The unit of displacement, the scale's radius_scale, given in km and
    # converted to Earth radii. A unit is one standard deviation of the
    # displacements' prior, as 0.1 g/cm^3 is of the density's.
    unit = problem.scale.radius_scale / R_SURFACE
    # Data per unit of jump times unit of displacement: rows mass and moment,
    # columns the two interfaces.
    shells = unit * np.vstack(
        [TO_MASS * 4.0 * np.pi * radii**2, TO_MOMENT * 8.0 * np.pi / 3.0 * radii**4]
    )
    n = space.dim

    def parts(x: Any) -> tuple[np.ndarray, np.ndarray]:
        c = model_space.to_components(x)
        return c[:n], c[n:]

    def value(x: Any) -> np.ndarray:
        c, s = parts(x)
        return linear @ c - shells @ (s * (jumps @ c))

    def matrix(c: np.ndarray, s: np.ndarray, constant: bool) -> np.ndarray:
        density_block = -shells @ (s[:, None] * jumps)
        if constant:
            density_block = density_block + linear
        return np.hstack([density_block, -shells * (jumps @ c)[None, :]])

    def derivative(x: Any) -> LinearOperator:
        c, s = parts(x)
        return LinearOperator.from_matrix(
            model_space, data_space, matrix(c, s, True), form="components"
        )

    def second_derivative(x: Any, direction: Any) -> LinearOperator:
        dc, ds = parts(direction)
        return LinearOperator.from_matrix(
            model_space, data_space, matrix(dc, ds, False), form="components"
        )

    operator = Operator.from_callables(
        model_space,
        data_space,
        value,
        derivative=derivative,
        second_derivative=second_derivative,
    )
    return model_space, operator


def check(
    operator: Operator,
    prior: GaussianMeasure,
    rng: np.random.Generator,
    /,
    *,
    points: int,
) -> float:
    """The derivatives against finite differences at a few draws from the
    prior, and the derivative's adjoint; the worst defect seen.

    The probe directions are draws from the prior less its expectation. The
    expectation, the reference model, has a norm of about 20 on the space,
    where the density part of a centered draw has a norm of about a half and
    each displacement a standard deviation of one: a normalized draw from the
    prior itself would point along the reference model every time, with the
    displacements a small part of it. Centered, the two parts are of the same
    order, and each direction moves the density and both boundaries, in
    proportions that vary from draw to draw.

    The differences take a step of one. The operator is bilinear, so central
    differences are exact for it and the only error is rounding, which is
    least at a large step.

    Args:
        operator: the forward operator.
        prior: what the points are drawn from.
        rng: the generator.
        points: how many draws to check at.
    """
    model_space = operator.domain
    probes = prior.translate(model_space.negative(prior.expectation))
    worst = 0.0
    for _ in range(points):
        point = prior.sample(rng=rng)
        reports = (
            testing.check_operator(operator.derivative(point), rng=rng, measure=probes),
            testing.check_derivative(
                operator, point, rng=rng, measure=probes, step=1.0
            ),
            testing.check_second_derivative(
                operator, point, rng=rng, measure=probes, step=1.0
            ),
        )
        worst = max(worst, *(max(r.defects.values()) for r in reports))
    return worst
