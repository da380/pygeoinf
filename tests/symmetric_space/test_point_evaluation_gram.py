"""
Tests for the Toeplitz point-evaluation Gram operators on the circle, line,
torus and plane Sobolev spaces.

The Gram operator A* diag(w) A must reproduce the composition of the dense
point-evaluation operator with its adjoint to machine precision, for scalar
and per-point weights alike, and must be self-adjoint.
"""

import numpy as np
import pytest

from pygeoinf.symmetric_space.circle import Sobolev as CircleSobolev
from pygeoinf.symmetric_space.line import Sobolev as LineSobolev
from pygeoinf.symmetric_space.plane import Sobolev as PlaneSobolev
from pygeoinf.symmetric_space.torus import Sobolev as TorusSobolev

ORDER = 2.0
SCALE = 0.1
N_POINTS = 7
SEED = 20260717
RTOL = 1e-10


def _circle_case(rng, kmax):
    space = CircleSobolev(kmax, ORDER, SCALE)
    points = list(rng.uniform(0.0, 2.0 * np.pi, N_POINTS))
    return space, points


def _line_case(rng, kmax):
    space = LineSobolev(kmax, ORDER, SCALE, a=-1.0, b=2.0)
    points = list(rng.uniform(-1.0, 2.0, N_POINTS))
    return space, points


def _torus_case(rng, kmax):
    space = TorusSobolev(kmax, ORDER, SCALE)
    points = [tuple(p) for p in rng.uniform(0.0, 2.0 * np.pi, (N_POINTS, 2))]
    return space, points


def _plane_case(rng, kmax):
    space = PlaneSobolev(kmax, ORDER, SCALE, ax=-1.0, bx=2.0, ay=0.5, by=3.0)
    xs = rng.uniform(-1.0, 2.0, N_POINTS)
    ys = rng.uniform(0.5, 3.0, N_POINTS)
    return space, list(zip(xs, ys))


CASES = {
    "circle": _circle_case,
    "line": _line_case,
    "torus": _torus_case,
    "plane": _plane_case,
}


@pytest.fixture(params=[7, 8], ids=["kmax_odd", "kmax_even"])
def kmax(request):
    return request.param


@pytest.fixture(params=CASES.keys())
def case(request, kmax):
    rng = np.random.default_rng(SEED)
    space, points = CASES[request.param](rng, kmax)
    return space, points, rng


def _relative_error(space, actual, expected):
    a = space.to_components(actual)
    e = space.to_components(expected)
    return np.linalg.norm(a - e) / np.linalg.norm(e)


class TestPointEvaluationGramOperator:

    def test_matches_dense_composition_with_vector_weights(self, case):
        """A* diag(w) A must match composing the dense operator with its
        adjoint, for heteroscedastic per-point weights."""
        space, points, rng = case
        weights = rng.uniform(0.5, 2.0, N_POINTS)

        dense = space.point_evaluation_operator(points)
        gram = space.point_evaluation_gram_operator(points, weights)

        for _ in range(3):
            x = space.random()
            expected = dense.adjoint(weights * dense(x))
            assert _relative_error(space, gram(x), expected) < RTOL

    def test_scalar_weight_equals_uniform_vector(self, case):
        """A scalar weight must act as a uniform per-point weight vector."""
        space, points, _ = case

        gram_scalar = space.point_evaluation_gram_operator(points, 0.7)
        gram_vector = space.point_evaluation_gram_operator(
            points, np.full(N_POINTS, 0.7)
        )

        x = space.random()
        assert _relative_error(space, gram_scalar(x), gram_vector(x)) < RTOL

    def test_self_adjoint(self, case):
        """<G x, y> must equal <x, G y>."""
        space, points, rng = case
        weights = rng.uniform(0.5, 2.0, N_POINTS)
        gram = space.point_evaluation_gram_operator(points, weights)

        assert gram.domain == space
        assert gram.codomain == space

        x = space.random()
        y = space.random()
        lhs = space.inner_product(gram(x), y)
        rhs = space.inner_product(x, gram(y))
        assert np.isclose(lhs, rhs, rtol=RTOL)

    def test_adjoint_action_matches(self, case):
        """The operator's declared adjoint must act identically to itself."""
        space, points, rng = case
        weights = rng.uniform(0.5, 2.0, N_POINTS)
        gram = space.point_evaluation_gram_operator(points, weights)

        x = space.random()
        assert _relative_error(space, gram.adjoint(x), gram(x)) < RTOL

    def test_wrong_weight_length_raises(self, case):
        space, points, _ = case
        with pytest.raises(ValueError):
            space.point_evaluation_gram_operator(points, np.ones(N_POINTS + 1))
