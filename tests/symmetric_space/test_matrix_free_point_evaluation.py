"""
Tests for the matrix-free point evaluation operators on the circle, line,
torus and plane Sobolev spaces.

The matrix-free operators must reproduce the dense operators built by
`point_evaluation_operator` to machine precision, in both their forward
and adjoint actions.
"""

import numpy as np
import pytest

from pygeoinf.symmetric_space.circle import Sobolev as CircleSobolev
from pygeoinf.symmetric_space.line import Sobolev as LineSobolev
from pygeoinf.symmetric_space.plane import Sobolev as PlaneSobolev
from pygeoinf.symmetric_space.torus import Sobolev as TorusSobolev

KMAX = 8
ORDER = 2.0
SCALE = 0.1
N_POINTS = 7
SEED = 20260716
RTOL = 1e-10


def _circle_case(rng):
    space = CircleSobolev(KMAX, ORDER, SCALE)
    points = list(rng.uniform(0.0, 2.0 * np.pi, N_POINTS))
    return space, points


def _line_case(rng):
    space = LineSobolev(KMAX, ORDER, SCALE, a=-1.0, b=2.0)
    points = list(rng.uniform(-1.0, 2.0, N_POINTS))
    return space, points


def _torus_case(rng):
    space = TorusSobolev(KMAX, ORDER, SCALE)
    points = [tuple(p) for p in rng.uniform(0.0, 2.0 * np.pi, (N_POINTS, 2))]
    return space, points


def _plane_case(rng):
    space = PlaneSobolev(KMAX, ORDER, SCALE, ax=-1.0, bx=2.0, ay=0.5, by=3.0)
    xs = rng.uniform(-1.0, 2.0, N_POINTS)
    ys = rng.uniform(0.5, 3.0, N_POINTS)
    return space, list(zip(xs, ys))


CASES = {
    "circle": _circle_case,
    "line": _line_case,
    "torus": _torus_case,
    "plane": _plane_case,
}


@pytest.fixture(params=CASES.keys())
def case(request):
    rng = np.random.default_rng(SEED)
    space, points = CASES[request.param](rng)
    return space, points


def _relative_error(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


class TestMatrixFreePointEvaluation:
    def test_forward_matches_dense(self, case):
        space, points = case
        op_dense = space.point_evaluation_operator(points)
        op_free = space.point_evaluation_operator(points, matrix_free=True)

        assert op_free.domain == op_dense.domain
        assert op_free.codomain == op_dense.codomain

        rng = np.random.default_rng(SEED + 1)
        for _ in range(5):
            u = space.from_components(rng.standard_normal(space.dim))
            assert _relative_error(op_free(u), op_dense(u)) < RTOL

    def test_adjoint_matches_dense(self, case):
        space, points = case
        op_dense = space.point_evaluation_operator(points)
        op_free = space.point_evaluation_operator(points, matrix_free=True)

        rng = np.random.default_rng(SEED + 2)
        for _ in range(5):
            y = rng.standard_normal(len(points))
            u_free = op_free.adjoint(y)
            u_dense = op_dense.adjoint(y)
            assert (
                _relative_error(
                    space.to_components(u_free), space.to_components(u_dense)
                )
                < RTOL
            )

    def test_operator_axioms(self, case):
        space, points = case
        op_free = space.point_evaluation_operator(points, matrix_free=True)
        op_free.check(n_checks=3)
