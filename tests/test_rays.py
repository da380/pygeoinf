"""
The nonlinear ray operator: travel times along traced rays.

On every geometry the constant background's rays are the geodesics, so the
travel time at ``m == 0`` is the geodesic's length over the reference
velocity to rounding; the derivative is checked against finite differences
of the value; bending and shooting are checked against each other; and a
traced ray takes no longer than the geodesic through the same field, which
is Fermat's principle in the form a test can read.
"""

import numpy as np
import pytest

from pygeoinf.algebra.operators import EvaluationError
from pygeoinf.symmetric_space.circle import Sobolev as CircleSobolev
from pygeoinf.symmetric_space.plane import Sobolev as PlaneSobolev
from pygeoinf.symmetric_space.torus import Sobolev as TorusSobolev

C0 = 2.0


def plane():
    return PlaneSobolev((48, 48), 2.0, 0.15, bounds=[(0.0, 2.0), (0.0, 2.0)])


def torus():
    return TorusSobolev((48, 40), 2.0, 0.15, lengths=(2.0, 1.6))


def sphere():
    pytest.importorskip("pyshtools")
    pytest.importorskip("ducc0")
    from pygeoinf.symmetric_space.sphere import Sobolev

    return Sobolev(24, 2.0, 0.3, radius=1.5)


def paths_on(space, rng, count=6):
    if hasattr(space, "bounds"):
        # Inside the plane, clear of the padding at its edges.
        return [
            (
                np.array([0.3 + 0.5 * rng.uniform(), 0.3 + 0.5 * rng.uniform()]),
                np.array([1.2 + 0.5 * rng.uniform(), 1.2 + 0.5 * rng.uniform()]),
            )
            for _ in range(count)
        ]
    return [
        (space.random_point(rng=rng), space.random_point(rng=rng)) for _ in range(count)
    ]


def mild_field(space, rng, size=0.15):
    raw = space.heat_measure(0.3 * space.length_scale).sample(rng=rng)
    values = (
        space.grid_values(raw) if hasattr(space, "grid_values") else np.asarray(raw)
    )
    return space.scale(size / np.max(np.abs(values)), raw)


@pytest.mark.parametrize(
    "make", [plane, torus, sphere], ids=["plane", "torus", "sphere"]
)
class TestOnTheConstantBackground:
    def test_the_travel_time_is_the_geodesic_length_over_the_speed(self, make, rng):
        space = make()
        paths = paths_on(space, rng)
        operator = space.ray_travel_time_operator(paths, C0, count=12)
        times = operator(space.zero())
        lengths = np.array([space.geodesic_distance(a, b) for a, b in paths])
        assert np.allclose(times, lengths / C0, rtol=1e-12, atol=1e-14)

    def test_the_derivative_is_the_linear_path_integral_scaled(self, make, rng):
        space = make()
        paths = paths_on(space, rng)
        operator = space.ray_travel_time_operator(
            paths, C0, count=12, nodes_per_segment=6
        )
        linear = space.path_integral_operator(paths, count=60)
        perturbation = mild_field(space, rng)
        derivative = operator.at(space.zero()).derivative
        assert np.allclose(
            derivative(perturbation), -linear(perturbation) / C0, rtol=1e-5, atol=1e-9
        )


@pytest.mark.parametrize(
    "make", [plane, torus, sphere], ids=["plane", "torus", "sphere"]
)
class TestInAField:
    def test_the_derivative_matches_finite_differences(self, make, rng):
        space = make()
        paths = paths_on(space, rng, count=4)
        operator = space.ray_travel_time_operator(paths, C0, count=10)
        model = mild_field(space, rng)
        direction = mild_field(space, rng, size=0.05)
        linearized = operator.at(model)
        assert np.allclose(linearized.value, operator(model))
        step = 1.0e-4
        forward = operator(space.add(model, space.scale(step, direction)))
        backward = operator(space.add(model, space.scale(-step, direction)))
        differences = (forward - backward) / (2.0 * step)
        assert np.allclose(
            linearized.derivative(direction), differences, rtol=1e-5, atol=1e-10
        )

    def test_the_ray_is_no_slower_than_the_geodesic(self, make, rng):
        space = make()
        paths = paths_on(space, rng, count=5)
        model = mild_field(space, rng, size=0.3)
        bent = space.ray_travel_time_operator(paths, C0, count=12)(model)
        # The geodesic's travel time through the same field: the linear
        # operator of the slowness, which is the ray operator with no bending.
        straight = space.ray_travel_time_operator(paths, C0, count=12, max_iterations=0)
        with pytest.raises(EvaluationError):
            straight(model)
        # Through the slowness field itself: integrate exp(-m)/c0 along the geodesics.
        nodes = [space.geodesic_quadrature(a, b, count=60) for a, b in paths]
        geodesic_times = np.array(
            [
                float(np.sum(w * np.exp(-space.evaluate(model, pts)) / C0))
                for pts, w in nodes
            ]
        )
        assert np.all(bent <= geodesic_times * (1.0 + 1e-10))

    def test_shooting_agrees_with_bending(self, make, rng):
        space = make()
        paths = paths_on(space, rng, count=2)
        model = mild_field(space, rng, size=0.2)
        # Two discretizations of one ray, each second order in its step:
        # a polyline of 32 segments against 192 Runge-Kutta steps, so the
        # two agree to a few parts in a hundred thousand.
        bent = space.ray_travel_time_operator(paths, C0, count=32)
        shot = space.ray_travel_time_operator(
            paths, C0, count=32, nodes_per_segment=6, method="shooting"
        )
        assert np.allclose(shot(model), bent(model), rtol=1e-4)
        rays = bent.trace_rays(model)
        assert len(rays) == len(paths)


class TestOnOneDimension:
    def test_a_ray_is_the_interval(self, rng):
        space = CircleSobolev(64, 2.0, 0.2)
        paths = [(0.3, 1.9), (2.0, 5.5)]
        model = mild_field(space, rng, size=0.4)
        operator = space.ray_travel_time_operator(paths, C0, count=20)
        expected = np.array(
            [
                float(np.sum(w * np.exp(-space.evaluate(model, pts)) / C0))
                for pts, w in (
                    space.geodesic_quadrature(a, b, count=80) for a, b in paths
                )
            ]
        )
        assert np.allclose(operator(model), expected, rtol=1e-8)
        direction = mild_field(space, rng, size=0.1)
        step = 1.0e-5
        differences = (
            operator(space.add(model, space.scale(step, direction)))
            - operator(space.add(model, space.scale(-step, direction)))
        ) / (2.0 * step)
        assert np.allclose(
            operator.at(model).derivative(direction), differences, rtol=1e-6
        )


@pytest.mark.parametrize(
    "make", [plane, torus, sphere], ids=["plane", "torus", "sphere"]
)
class TestTheHessianAction:
    """``F''(m)[δm₁, δm₂]``: against differences of the derivative, and symmetric."""

    def test_it_matches_differences_of_the_derivative(self, make, rng):
        space = make()
        paths = paths_on(space, rng, count=3)
        operator = space.ray_travel_time_operator(paths, C0, count=16)
        assert operator.has_second_derivative
        model = mild_field(space, rng, size=0.2)
        first = mild_field(space, rng, size=0.1)
        second = mild_field(space, rng, size=0.1)
        action = operator.second_derivative(model, first)(second)
        step = 1.0e-3
        up = operator.derivative(space.add(model, space.scale(step, first)))(second)
        down = operator.derivative(space.add(model, space.scale(-step, first)))(second)
        differences = (up - down) / (2.0 * step)
        assert np.allclose(action, differences, rtol=2e-4, atol=1e-9)

    def test_it_is_symmetric_in_its_two_directions(self, make, rng):
        space = make()
        paths = paths_on(space, rng, count=3)
        operator = space.ray_travel_time_operator(paths, C0, count=16)
        model = mild_field(space, rng, size=0.2)
        first = mild_field(space, rng, size=0.1)
        second = mild_field(space, rng, size=0.1)
        one = operator.second_derivative(model, first)(second)
        other = operator.second_derivative(model, second)(first)
        assert np.allclose(one, other, rtol=1e-5, atol=1e-10)

    def test_the_ray_response_matters(self, make, rng):
        """Without the ray's response the Hessian would be the explicit term alone."""
        space = make()
        paths = paths_on(space, rng, count=3)
        operator = space.ray_travel_time_operator(paths, C0, count=16)
        model = mild_field(space, rng, size=0.2)
        first = mild_field(space, rng, size=0.1)
        second = mild_field(space, rng, size=0.1)
        full = operator.second_derivative(model, first)(second)
        rays = operator.trace_rays(model)
        explicit = np.array(
            [
                float(
                    np.sum(
                        w
                        * np.exp(-space.evaluate(model, pts))
                        / C0
                        * space.evaluate(first, pts)
                        * space.evaluate(second, pts)
                    )
                )
                for (pts, w) in (
                    (space.geodesic_quadrature(a, b, count=40)) for a, b in paths
                )
            ]
        )
        del rays
        # The explicit term along the geodesic is only a stand-in for the
        # one along the ray; the point is that the two are not the same
        # number, since the ray's response is of the same order.
        assert not np.allclose(full, explicit, rtol=1e-3)


@pytest.mark.parametrize("make", [plane, sphere], ids=["plane", "sphere"])
class TestTheShootingHessian:
    """The shooting route's Hessian action, from the tangent linear system."""

    def test_it_matches_differences_of_the_derivative_and_is_symmetric(self, make, rng):
        space = make()
        paths = paths_on(space, rng, count=2)
        operator = space.ray_travel_time_operator(
            paths, C0, count=24, nodes_per_segment=6, method="shooting", tolerance=1e-10
        )
        model = mild_field(space, rng, size=0.2)
        first = mild_field(space, rng, size=0.1)
        second = mild_field(space, rng, size=0.1)
        action = operator.second_derivative(model, first)(second)
        step = 1.0e-2
        up = operator.derivative(space.add(model, space.scale(step, first)))(second)
        down = operator.derivative(space.add(model, space.scale(-step, first)))(second)
        differences = (up - down) / (2.0 * step)
        assert np.allclose(action, differences, rtol=1e-3, atol=1e-9)
        # The two responses are integrated separately, so symmetry holds to
        # the tangent linear system's accuracy rather than to rounding.
        other = operator.second_derivative(model, second)(first)
        assert np.allclose(action, other, rtol=5e-3, atol=1e-9)


def test_a_non_finite_model_is_an_evaluation_failure():
    """NaN in the model must raise, not reach the tracer."""
    import numpy as np
    import pytest

    from pygeoinf.algebra.operators import EvaluationError
    from pygeoinf.symmetric_space import plane

    space = plane.Sobolev((8, 8), 2.0, 0.3, bounds=((0.0, 1.0), (0.0, 1.0)))
    operator = space.ray_travel_time_operator(
        [((0.1, 0.1), (0.9, 0.9))], 1.0, nodes_per_segment=4
    )
    bad = space.from_components(np.full(space.dim, np.nan))
    with pytest.raises(EvaluationError, match="non-finite"):
        operator(bad)


@pytest.mark.parametrize("make", [plane, sphere], ids=["plane", "sphere"])
class TestTheDerivativeIsAnOperator:
    """The derivative at a model is a linear operator with the library's own
    adjoint, which is what Gauss-Newton and the MAP estimator apply; the
    check ladder holds it to the adjoint identity on random pairs."""

    def test_its_adjoint_passes_the_check(self, make, rng):
        from pygeoinf.testing import check_operator

        space = make()
        paths = paths_on(space, rng, count=3)
        operator = space.ray_travel_time_operator(paths, C0, count=12)
        model = mild_field(space, rng, size=0.2)
        report = check_operator(operator.derivative(model), rng=rng)
        assert report.worst < 1e-8


class TestRefusals:
    def test_an_unknown_method_is_refused_by_name(self, rng):
        """The rays are traced when the operator is first applied, so that is
        where the method is read and an unknown one refused."""
        space = sphere()
        operator = space.ray_travel_time_operator(
            paths_on(space, rng), C0, method="newton"
        )
        with pytest.raises(ValueError, match="'bending' or 'shooting'"):
            operator(space.zero())

    def test_no_paths_is_refused(self):
        space = sphere()
        with pytest.raises(ValueError, match="At least one path"):
            space.ray_travel_time_operator([], C0)

    def test_a_space_too_rough_for_a_path_integral_is_refused(self):
        """The derivative is a path integral, which is not bounded at or
        below order ``(d - 1) / 2``."""
        from pygeoinf.symmetric_space.torus import Lebesgue

        space = Lebesgue((16, 16))
        paths = [((1.0, 1.0), (4.0, 2.0))]
        with pytest.raises(ValueError, match="above 0.5"):
            space.ray_travel_time_operator(paths, C0)
        operator = space.ray_travel_time_operator(paths, C0, unsafe=True)
        assert np.isfinite(operator(space.zero())).all()


class TestAModelUpdatedInPlace:
    def test_the_linearization_follows_the_values_not_the_object(self, rng):
        """``axpy`` changes a vector in place; the rays traced for the old
        values must not be returned for the new ones."""
        space = PlaneSobolev((24, 24), 2.0, 0.25, bounds=[(0.0, 2.0), (0.0, 2.0)])
        paths = [((0.2, 0.3), (1.8, 1.5)), ((0.3, 1.7), (1.6, 0.2))]
        operator = space.ray_travel_time_operator(paths, 1.0, count=12)
        model = space.zero()
        operator.at(model)
        bump = space.project_function(
            lambda p: 0.3 * np.exp(-((p[0] - 1.0) ** 2 + (p[1] - 1.0) ** 2) / 0.1)
        )
        space.axpy(1.0, bump, model)
        updated = operator.at(model)
        fresh = operator.at(space.copy(model))
        assert np.allclose(updated.value, operator(model), rtol=1e-10)
        assert np.allclose(updated.value, fresh.value, rtol=1e-10)
        assert np.allclose(updated.derivative(bump), fresh.derivative(bump), rtol=1e-10)
