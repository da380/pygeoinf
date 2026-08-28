"""The QP backends, and the level bundle method built on them."""

import numpy as np
import pytest

from pygeoinf2.algebra.operators import Functional, LinearFunctional
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.numerics.convex import LevelBundleMethod, ProximalBundleMethod
from pygeoinf2.numerics.quadratic_programming import (
    ClarabelQPSolver,
    OSQPQPSolver,
    QPResult,
    QPSolver,
    SciPyQPSolver,
    best_available_qp_solver,
)


def backends():
    """Every backend that is actually installed."""
    available = [SciPyQPSolver]
    for optional in (OSQPQPSolver, ClarabelQPSolver):
        try:
            optional()
        except ImportError:  # pragma: no cover - depends on the install
            continue
        available.append(optional)
    return available


@pytest.fixture
def programme(rng):
    """A strictly convex QP with one equality and some one-sided bounds."""
    size, rows = 8, 5
    root = rng.standard_normal((size, size))
    quadratic = root @ root.T + np.eye(size)
    linear = rng.standard_normal(size)
    constraint = np.vstack([np.ones((1, size)), rng.standard_normal((rows - 1, size))])
    lower = np.concatenate([[1.0], np.full(rows - 1, -np.inf)])
    upper = np.concatenate([[1.0], rng.uniform(0.5, 2.0, rows - 1)])
    return quadratic, linear, constraint, lower, upper


class TestTheBackendsAgree:
    """Three solvers, one answer. They are interchangeable or they are not."""

    @pytest.mark.parametrize("backend", backends())
    def test_each_solves_it_feasibly(self, backend, programme):
        quadratic, linear, constraint, lower, upper = programme
        result = backend().solve(quadratic, linear, constraint, lower, upper)

        assert result.solved
        assert np.all(constraint @ result.x >= lower - 1e-6)
        assert np.all(constraint @ result.x <= upper + 1e-6)

    def test_they_reach_the_same_minimum(self, programme):
        quadratic, linear, constraint, lower, upper = programme
        objectives = [
            backend().solve(quadratic, linear, constraint, lower, upper).objective
            for backend in backends()
        ]
        for objective in objectives[1:]:
            assert objective == pytest.approx(objectives[0], rel=1e-6, abs=1e-8)

    def test_they_all_satisfy_the_protocol(self):
        for backend in backends():
            assert isinstance(backend(), QPSolver)

    def test_the_best_available_is_one_of_them(self):
        assert isinstance(best_available_qp_solver(), QPSolver)

    @pytest.mark.parametrize("backend", backends())
    def test_a_warm_start_is_accepted(self, backend, programme, rng):
        """Clarabel has no use for one and must still take it, or the three
        are not interchangeable."""
        quadratic, linear, constraint, lower, upper = programme
        result = backend().solve(
            quadratic, linear, constraint, lower, upper, x0=rng.standard_normal(8)
        )
        assert result.solved


class TestBadProgrammesAreRefused:
    @pytest.mark.parametrize("backend", backends())
    def test_mismatched_shapes(self, backend):
        with pytest.raises(ValueError, match="must be"):
            backend().solve(np.eye(3), np.zeros(4), np.ones((1, 4)), [0.0], [1.0])

    @pytest.mark.parametrize("backend", backends())
    def test_crossed_bounds(self, backend):
        with pytest.raises(ValueError, match="lower bound"):
            backend().solve(np.eye(2), np.zeros(2), np.ones((1, 2)), [2.0], [1.0])


class TestTheLevelBundleBoundIsABound:
    """The reason this method exists: its gap is a statement about the
    minimum, where the proximal method's is a statement about its model."""

    @pytest.fixture
    def quadratic_problem(self, rng):
        size = 8
        root = rng.standard_normal((size, size))
        matrix = root @ root.T + np.eye(size)
        offset = rng.standard_normal(size)
        space = EuclideanSpace(size)
        functional = Functional.from_callables(
            space,
            lambda c: float(0.5 * c @ (matrix @ c) - offset @ c),
            derivative=lambda c: LinearFunctional.from_derivative_components(
                space, matrix @ c - offset
            ),
        )
        exact = float(-0.5 * offset @ np.linalg.solve(matrix, offset))
        return space, functional, exact

    @pytest.mark.parametrize("iterations", [15, 40, 200])
    def test_the_lower_bound_never_passes_the_minimum(
        self, quadratic_problem, iterations
    ):
        """At every stage, not just at the end -- an invalid bound part way
        through is still an invalid bound."""
        space, functional, exact = quadratic_problem
        result = LevelBundleMethod(
            tolerance=1e-14, iterations=iterations
        ).minimise(functional, space.zero())

        assert result.value - result.gap <= exact + 1e-9
        assert result.value >= exact - 1e-9

    def test_it_closes_on_the_answer(self, quadratic_problem):
        space, functional, exact = quadratic_problem
        # Relative to the value, which here is small: 1e-6 of 0.056 is already
        # 5.6e-8 absolute, and asking for 1e-8 of it would be asking the
        # bundle for ten significant figures.
        result = LevelBundleMethod(tolerance=1e-6).minimise(functional, space.zero())
        assert result.converged
        assert result.value == pytest.approx(exact, abs=1e-5)

    def test_it_agrees_with_the_proximal_method_on_a_nonsmooth_problem(self, rng):
        """Different methods, same minimum, or one of them is wrong."""
        size, rows = 10, 6
        matrix = rng.standard_normal((rows, size))
        offset = rng.standard_normal(rows)
        space = EuclideanSpace(size)
        functional = Functional.from_callables(
            space,
            lambda c: float(np.abs(matrix @ c - offset).sum() + 0.1 * c @ c),
            derivative=lambda c: LinearFunctional.from_derivative_components(
                space, matrix.T @ np.sign(matrix @ c - offset) + 0.2 * c
            ),
        )
        level = LevelBundleMethod(tolerance=1e-6, iterations=300).minimise(
            functional, space.zero()
        )
        proximal = ProximalBundleMethod(tolerance=1e-6, iterations=300).minimise(
            functional, space.zero()
        )
        assert level.value == pytest.approx(proximal.value, abs=1e-4)

    def test_v1s_worked_example(self):
        """``x^2 + 2x``, minimum minus one at minus one -- v1's doctest."""
        space = EuclideanSpace(1)
        functional = Functional.from_callables(
            space,
            lambda c: float(c[0] ** 2 + 2 * c[0]),
            derivative=lambda c: LinearFunctional.from_derivative_components(
                space, np.array([2 * c[0] + 2.0])
            ),
        )
        result = LevelBundleMethod(tolerance=1e-8).minimise(
            functional, space.from_components(np.array([2.0]))
        )
        assert result.minimiser == pytest.approx([-1.0], abs=1e-3)
        assert result.value == pytest.approx(-1.0, abs=1e-6)

    def test_an_alpha_outside_the_unit_interval_is_refused(self):
        for bad in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError, match="alpha"):
                LevelBundleMethod(alpha=bad)

    def test_a_space_without_coordinates_is_refused(self):
        """The master problem is a QP in components, so unlike the proximal
        method this one needs a basis."""
        from .doubles import OpaqueSpace

        space = OpaqueSpace(6)
        functional = Functional.from_callables(
            space, lambda x: space.squared_norm(x), derivative=None
        )
        with pytest.raises(Exception):
            LevelBundleMethod().minimise(functional, space.zero())
