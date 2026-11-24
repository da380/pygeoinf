"""The feasible property set of a nonlinear forward problem: fixtures and what they establish.

Small problems whose true feasible set is known by other means, so that a
bound on it can be judged: a two-parameter attenuation map whose support is
exact for one datum and found by a zooming search for several, a quadratic
map whose curvature is known in closed form, and point observations of a
field and of its square on an interval.
"""

from __future__ import annotations

import numpy as np
import pytest

from pygeoinf import (
    BackusGilbertParker,
    EuclideanSpace,
    ForwardProblem,
    LinearOperator,
    Operator,
)
from pygeoinf.algebra.operators import EvaluationError, LinearFunctional
from pygeoinf.geometry.convex import Ball, HalfSpace, Polytope
from pygeoinf.geometry.feasible import FeasiblePropertySet, FeasibleStructure
from pygeoinf.geometry.nonlinear import LocalSearch, NonlinearFeasiblePropertySet
from pygeoinf.symmetric_space import line
from pygeoinf.testing import check_derivative, check_second_derivative

from .conftest import make_dense_metric_space

# --------------------------------------------------------------------- #
#                               Fixtures                                 #
# --------------------------------------------------------------------- #


def attenuation(model, paths):
    """``F_i(m) = exp(−(a_i, m))`` for path vectors ``a_i`` of the model space.

    Written through the model space's inner product alone, so that it is the
    same map on any metric. Its second derivative in every data component is
    positive semi-definite, which makes the linearization's remainder
    one-signed.
    """
    data = EuclideanSpace(len(paths))

    def weights(vector):
        return np.array([model.inner_product(path, vector) for path in paths])

    def combine(coefficients):
        out = model.zero()
        for coefficient, path in zip(coefficients, paths):
            out = model.axpy(coefficient, path, out)
        return out

    def value(m):
        exponents = -weights(m)
        if np.any(exponents > 50.0):
            raise EvaluationError("The attenuation overflows at this model.")
        return np.exp(exponents)

    def derivative(m):
        values = value(m)
        return LinearOperator.from_callables(
            model,
            data,
            lambda v: -values * weights(v),
            adjoint=lambda y: combine(-values * y),
        )

    def second_derivative(m, dx):
        scaled = value(m) * weights(dx)
        return LinearOperator.from_callables(
            model,
            data,
            lambda v: scaled * weights(v),
            adjoint=lambda y: combine(scaled * y),
        )

    return Operator.from_callables(
        model, data, value, derivative=derivative, second_derivative=second_derivative
    )


def quadratic_map(model, data, rng):
    """``F(m) = sum_i [(a_i, m) + ½ (m, H_i m)] e_i`` over the data space's basis ``e_i``.

    Returns the operator and the symmetric Galerkin matrices ``S_i`` of the
    ``H_i``, so that ``(u, H_i v) == c_u · S_i c_v`` in components.
    """
    linear = rng.normal(size=(data.dim, model.dim))
    forms = [rng.normal(size=(model.dim, model.dim)) for _ in range(data.dim)]
    forms = [0.5 * (form + form.T) for form in forms]

    def value(m):
        c = model.to_components(m)
        return data.from_components(
            linear @ c + 0.5 * np.array([c @ form @ c for form in forms])
        )

    def derivative(m):
        c = model.to_components(m)
        return LinearOperator.from_matrix(
            model,
            data,
            linear + np.stack([form @ c for form in forms]),
            form="components",
        )

    def second_derivative(m, dx):
        c = model.to_components(dx)
        return LinearOperator.from_matrix(
            model, data, np.stack([form @ c for form in forms]), form="components"
        )

    operator = Operator.from_callables(
        model, data, value, derivative=derivative, second_derivative=second_derivative
    )
    return operator, forms


def values_and_squares(space, linear_points, squared_points):
    """Point values of a field at some points and of its square at others.

    ``F(u) = (u(x_i))_i ⊕ (u(y_j)²)_j``: a linear evaluation followed by a
    componentwise map, so the curvature lives in the directions of the
    evaluation functionals at the squared points and nowhere else.
    """
    evaluation = space.point_evaluation_operator(
        list(linear_points) + list(squared_points), route="dense"
    )
    samples = evaluation.codomain
    first = len(linear_points)

    def value(y):
        out = np.array(y, dtype=float)
        out[first:] = out[first:] ** 2
        return out

    def scaling(weights):
        return LinearOperator.from_matrix(
            samples, samples, np.diag(weights), form="components"
        )

    def derivative(y):
        weights = np.ones(samples.dim)
        weights[first:] = 2.0 * np.asarray(y)[first:]
        return scaling(weights)

    def second_derivative(y, dy):
        weights = np.zeros(samples.dim)
        weights[first:] = 2.0 * np.asarray(dy)[first:]
        return scaling(weights)

    pointwise = Operator.from_callables(
        samples,
        samples,
        value,
        derivative=derivative,
        second_derivative=second_derivative,
    )
    return pointwise @ evaluation


# The two-parameter problem: Euclidean, with the support of the true feasible
# set known in the direction ``COST``.

COST = np.array([1.0, 0.4])
TRUE_MODEL = np.array([0.3, -0.2])
CASES = [
    # data, prior radius, noise radius
    (1, 0.5, 0.02),
    (1, 1.0, 0.02),
    (1, 2.0, 0.02),
    (1, 1.0, 0.20),
    (3, 0.5, 0.05),
    (3, 1.0, 0.05),
]


def path_weights(count):
    """The path weights of the two-parameter problem, one row per datum."""
    return np.abs(np.random.default_rng(3).standard_normal((count, 2))) + 0.2


def two_parameter_problem(count):
    """The attenuation map on the plane, its path matrix and the noise-free data."""
    model = EuclideanSpace(2)
    weights = path_weights(count)
    forward = attenuation(model, list(weights))
    return model, forward, weights, forward(TRUE_MODEL)


def strip_support(path, datum, radius, sigma):
    """The support for one datum, in closed form.

    ``|exp(−a·m) − d| <= σ`` is the strip ``lo <= a·m <= hi``, and the
    largest ``c·m`` over a strip cut by a disc is at the disc's own
    maximizer when the strip holds it, and otherwise where an edge of the
    strip meets the circle.
    """
    low = -np.log(datum + sigma)
    high = -np.log(datum - sigma) if datum > sigma else np.inf
    top = radius * COST / np.linalg.norm(COST)
    if low <= path @ top <= high:
        return float(COST @ top)
    length = np.linalg.norm(path)
    along = path / length
    across = np.array([-along[1], along[0]])
    best = -np.inf
    for level in (low, high):
        offset = level / length
        if abs(offset) <= radius:
            reach = np.sqrt(radius**2 - offset**2)
            for sign in (-1.0, 1.0):
                best = max(best, COST @ (offset * along + sign * reach * across))
    return float(best)


def searched_support(weights, data, radius, sigma, *, resolution=1e-7):
    """The largest ``c·m`` over feasible points of ever finer grids, and the model.

    The value is attained by a model that is in the prior disc and fits the
    data, so it never exceeds the true support; it falls short by at most
    the last grid's spacing times ``|c|`` provided the first grid found the
    right part of the set.
    """

    def feasible(points):
        fits = np.linalg.norm(np.exp(-(points @ weights.T)) - data, axis=1) <= sigma
        return fits & (np.linalg.norm(points, axis=1) <= radius)

    def best_of(points):
        inside = points[feasible(points)]
        return inside[np.argmax(inside @ COST)]

    axis = np.linspace(-radius, radius, 801)
    spacing = axis[1] - axis[0]
    grid = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
    best = best_of(grid)
    while spacing > resolution:
        local = np.linspace(-2.0 * spacing, 2.0 * spacing, 81)
        spacing = local[1] - local[0]
        grid = best + np.stack(np.meshgrid(local, local), -1).reshape(-1, 2)
        best = best_of(grid)
    return float(COST @ best), best


def linearized_support(count, radius, sigma):
    """The support of the set linearized about the origin, with its bracket."""
    model, forward, _, data = two_parameter_problem(count)
    property_operator = LinearOperator.from_matrix(
        model, EuclideanSpace(1), COST[None, :], form="components"
    )
    feasible = BackusGilbertParker(
        ForwardProblem(forward),
        property_operator,
        Ball(model, radius=radius),
        noise=Ball(forward.codomain, radius=sigma),
        model=model.zero(),
    )(data)
    return feasible.support_result(np.array([1.0]))


# --------------------------------------------------------------------- #
#                                Tests                                   #
# --------------------------------------------------------------------- #


class TestTheFixtures:
    def test_the_attenuation_map_on_a_dense_metric(self, rng):
        model = make_dense_metric_space(4)
        forward = attenuation(model, [model.random(rng=rng) for _ in range(3)])
        point = model.scale(0.3, model.random(rng=rng))
        check_derivative(forward, point, rng=rng)
        check_second_derivative(forward, point, rng=rng)

    def test_the_quadratic_map_on_dense_metrics(self, rng):
        model, data = make_dense_metric_space(4), make_dense_metric_space(3)
        forward, _ = quadratic_map(model, data, rng)
        point = model.random(rng=rng)
        check_derivative(forward, point, rng=rng)
        check_second_derivative(forward, point, rng=rng)

    def test_values_and_squares_on_an_interval(self, rng):
        space = line.Sobolev(64, 2.0, 0.1)
        forward = values_and_squares(space, [0.2, 0.7], [0.4, 0.9])
        measure = space.sobolev_measure(2.0, 0.2)
        point = measure.sample(rng=rng)
        assert forward.has_second_derivative
        check_derivative(forward, point, rng=rng, measure=measure)
        check_second_derivative(forward, point, rng=rng, measure=measure, rtol=1e-3)


class TestTheCurvatureOfACertificate:
    """``(λ, F(m))`` has Hessian ``v -> F''(m)[v, ·]* λ``, in the two spaces' metrics."""

    def setting(self, rng):
        model, data = make_dense_metric_space(4), make_dense_metric_space(3)
        forward, forms = quadratic_map(model, data, rng)
        certificate = data.random(rng=rng)
        return model, data, forward, forms, certificate

    def exact_hessian(self, model, data, forms, certificate):
        # (λ, F(m)) = sum_i (λ, e_i) f_i(m), so the weights are inner
        # products with the basis, not the certificate's components.
        weights = [
            data.inner_product(certificate, data.basis_vector(i))
            for i in range(data.dim)
        ]
        matrix = sum(weight * form for weight, form in zip(weights, forms))
        return LinearOperator.from_matrix(model, model, matrix, form="galerkin")

    def test_it_is_the_adjoint_of_the_curried_second_derivative(self, rng):
        model, data, forward, forms, certificate = self.setting(rng)
        hessian = self.exact_hessian(model, data, forms, certificate)
        point, probe = model.random(rng=rng), model.random(rng=rng)
        found = forward.second_derivative(point, probe).adjoint(certificate)
        assert model.norm(model.subtract(found, hessian(probe))) <= 1e-12 * model.norm(
            found
        )

    def test_the_composed_functional_agrees(self, rng):
        model, data, forward, forms, certificate = self.setting(rng)
        functional = LinearFunctional.from_representer(data, certificate) @ forward
        point, probe = model.random(rng=rng), model.random(rng=rng)
        composed = functional.hessian(point)(probe)
        found = forward.second_derivative(point, probe).adjoint(certificate)
        assert model.norm(model.subtract(found, composed)) <= 1e-12 * model.norm(found)

    def test_it_is_symmetric_in_the_model_metric(self, rng):
        model, data, forward, _, certificate = self.setting(rng)
        point, u, v = (model.random(rng=rng) for _ in range(3))
        left = model.inner_product(
            u, forward.second_derivative(point, v).adjoint(certificate)
        )
        right = model.inner_product(
            forward.second_derivative(point, u).adjoint(certificate), v
        )
        assert left == pytest.approx(right, rel=1e-12)

    def test_the_components_of_the_certificate_are_the_wrong_weights(self, rng):
        """The control: on a dense data metric the two readings differ."""
        model, data, forward, forms, certificate = self.setting(rng)
        wrong = LinearOperator.from_matrix(
            model,
            model,
            sum(
                weight * form
                for weight, form in zip(data.to_components(certificate), forms)
            ),
            form="galerkin",
        )
        point, probe = model.random(rng=rng), model.random(rng=rng)
        found = forward.second_derivative(point, probe).adjoint(certificate)
        assert model.norm(model.subtract(found, wrong(probe))) > 1e-2 * model.norm(
            found
        )


class TestTheKnownSupport:
    @pytest.mark.parametrize(
        "radius, sigma", [case[1:] for case in CASES if case[0] == 1]
    )
    def test_the_search_finds_the_closed_form(self, radius, sigma):
        """Two routes to the support for one datum: the strip, and the zooming search."""
        _, _, weights, data = two_parameter_problem(1)
        exact = strip_support(weights[0], data[0], radius, sigma)
        searched, model = searched_support(weights, data, radius, sigma)
        assert searched <= exact + 1e-12
        assert searched == pytest.approx(exact, abs=1e-6)
        assert np.linalg.norm(model) <= radius
        assert np.linalg.norm(np.exp(-(weights @ model)) - data) <= sigma


class TestLinearizationIsNotABound:
    @pytest.mark.parametrize("count, radius, sigma", CASES)
    def test_a_feasible_model_lies_beyond_the_linearized_support(
        self, count, radius, sigma
    ):
        """The linearized set's outer value is exceeded by a model that fits.

        The outer value is a bound on the linearized problem's set and is
        certified as that; a model in the prior that fits the data through
        the real map and has a larger property shows it bounds nothing of
        the real problem.
        """
        _, _, weights, data = two_parameter_problem(count)
        attained, model = searched_support(weights, data, radius, sigma)
        assert np.linalg.norm(model) <= radius
        assert np.linalg.norm(np.exp(-(weights @ model)) - data) <= sigma
        linearized = linearized_support(count, radius, sigma)
        assert linearized.converged
        assert linearized.value < attained - 1e-3

    def test_a_linear_map_is_bounded_by_its_own_linearization(self):
        """The control: with no curvature the same comparison goes the other way."""
        model = EuclideanSpace(2)
        weights = path_weights(3)
        forward = LinearOperator.from_matrix(
            model, EuclideanSpace(3), -weights, form="components"
        )
        data = forward(TRUE_MODEL)
        radius, sigma = 1.0, 0.05
        property_operator = LinearOperator.from_matrix(
            model, EuclideanSpace(1), COST[None, :], form="components"
        )
        feasible = BackusGilbertParker(
            ForwardProblem(forward),
            property_operator,
            Ball(model, radius=radius),
            noise=Ball(forward.codomain, radius=sigma),
            model=model.zero(),
        )(data)
        rng = np.random.default_rng(0)
        value = feasible.support(np.array([1.0]))
        for _ in range(2000):
            candidate = rng.uniform(-radius, radius, 2)
            fits = np.linalg.norm(forward(candidate) - data) <= sigma
            if fits and np.linalg.norm(candidate) <= radius:
                assert COST @ candidate <= value + 1e-9


# --------------------------------------------------------------------- #
#                      The set probed from inside                        #
# --------------------------------------------------------------------- #


def two_parameter_set(count, radius, sigma):
    """The nonlinear feasible set of the two-parameter problem, for the property ``c·m``."""
    model, forward, _, data = two_parameter_problem(count)
    property_operator = LinearOperator.from_matrix(
        model, EuclideanSpace(1), COST[None, :], form="components"
    )
    return NonlinearFeasiblePropertySet(
        forward,
        property_operator,
        Ball(model, radius=radius),
        data,
        noise=Ball(forward.codomain, radius=sigma),
    )


def known_support(count, radius, sigma):
    """The support in closed form for one datum, and by the search for several."""
    _, _, weights, data = two_parameter_problem(count)
    if count == 1:
        return strip_support(weights[0], data[0], radius, sigma)
    return searched_support(weights, data, radius, sigma)[0]


def dense_metric_setting(rng):
    """The attenuation map on a dense metric, with sets, data and a scalar property."""
    model = make_dense_metric_space(4)
    forward = attenuation(
        model, [model.scale(0.5, model.random(rng=rng)) for _ in range(3)]
    )
    truth = model.scale(0.4 / 2.0, model.random(rng=rng))
    data = forward(truth)
    representer = model.random(rng=rng)
    property_operator = LinearOperator.from_callables(
        model,
        EuclideanSpace(1),
        lambda m: np.array([model.inner_product(representer, m)]),
        adjoint=lambda y: model.scale(float(y[0]), representer),
    )
    radius = 2.0 * model.norm(truth)
    feasible = NonlinearFeasiblePropertySet(
        forward,
        property_operator,
        Ball(model, radius=radius),
        data,
        noise=Ball(forward.codomain, radius=0.05),
    )
    return model, forward, feasible, radius


class TestTheLocalSupport:
    @pytest.mark.parametrize("count, radius, sigma", CASES)
    def test_it_reaches_the_known_support_from_inside(self, count, radius, sigma):
        """Two routes to one number: the search's checked model, and the oracle."""
        feasible = two_parameter_set(count, radius, sigma)
        truth = known_support(count, radius, sigma)
        result = feasible.support_result(np.array([1.0]))
        assert result.converged
        assert result.algorithm == "local"
        assert result.value == float("inf")
        assert feasible.is_feasible(result.extremal_model)
        assert result.inner == pytest.approx(COST @ result.extremal_model, abs=1e-14)
        # Attained, so never beyond the truth; the search's margin and the
        # oracle's resolution are the only slack.
        assert result.inner <= truth + 1e-6
        assert result.inner >= truth - 1e-4

    @pytest.mark.parametrize("count, radius, sigma", CASES)
    def test_it_lies_beyond_the_linearized_support(self, count, radius, sigma):
        feasible = two_parameter_set(count, radius, sigma)
        result = feasible.support_result(np.array([1.0]))
        assert result.inner > linearized_support(count, radius, sigma).value + 1e-3

    def test_a_feasible_start_is_never_lost(self):
        """With no budget to search, the answer is still the best checked model."""
        feasible = two_parameter_set(3, 1.0, 0.05)
        idle = LocalSearch(max_iterations=1)
        result = feasible.support_result(np.array([1.0]), start=TRUE_MODEL, via=idle)
        assert feasible.is_feasible(result.extremal_model)
        assert result.inner >= COST @ TRUE_MODEL - 1e-14

    def test_the_fitting_model_is_feasible_and_closest_found(self):
        feasible = two_parameter_set(3, 1.0, 0.05)
        result = feasible.fitting_result()
        assert result.converged
        assert feasible.is_feasible(result.fitting_model)
        # The true model fits exactly, so the closest fitting model is no
        # farther from the center than it is.
        assert result.level <= TRUE_MODEL @ TRUE_MODEL + 1e-9
        assert result.lower_bound == float("-inf")

    def test_several_starts_agree_on_a_connected_set(self, rng):
        feasible = two_parameter_set(3, 1.0, 0.05)
        starts = [rng.uniform(-0.5, 0.5, 2) for _ in range(4)]
        results = feasible.support_results(np.array([1.0]), starts)
        inners = [
            result.inner for result in results if result.extremal_model is not None
        ]
        assert len(inners) >= 3
        assert max(inners) - min(inners) <= 1e-5


class TestTheLocalCertificate:
    """At a local maximum ``T*q == F'(m)* λ`` plus a normal to the prior, in the model metric."""

    def test_stationarity_on_a_dense_metric(self, rng):
        model, forward, feasible, radius = dense_metric_setting(rng)
        direction = np.array([1.0])
        result = feasible.support_result(direction)
        assert result.converged
        extremal = result.extremal_model
        pulled = feasible.property_operator.adjoint(direction)
        through_data = forward.derivative(extremal).adjoint(result.certificate)
        # The prior's scaled level function is |m|^2 / r^2, with gradient
        # 2 m / r^2 and the first multiplier.
        normal = model.scale(2.0 * result.multipliers[0] / radius**2, extremal)
        defect = model.subtract(model.subtract(pulled, through_data), normal)
        assert model.norm(defect) <= 1e-5 * model.norm(pulled)
        assert all(multiplier >= 0.0 for multiplier in result.multipliers)

    def test_the_opposite_sign_is_not_stationary(self, rng):
        """The control: the certificate's sign is part of the convention."""
        model, forward, feasible, radius = dense_metric_setting(rng)
        direction = np.array([1.0])
        result = feasible.support_result(direction)
        assert result.multipliers[1] > 0.0
        extremal = result.extremal_model
        pulled = feasible.property_operator.adjoint(direction)
        flipped = forward.derivative(extremal).adjoint(
            forward.codomain.negative(result.certificate)
        )
        normal = model.scale(2.0 * result.multipliers[0] / radius**2, extremal)
        defect = model.subtract(model.subtract(pulled, flipped), normal)
        assert model.norm(defect) > 1e-2 * model.norm(pulled)


class TestALinearMapIsRecovered:
    def test_the_local_support_closes_the_linear_bracket(self, rng):
        """With no curvature the search from inside meets the certified outer value."""
        model, data_space = make_dense_metric_space(5), make_dense_metric_space(3)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.normal(size=(3, 5)), form="components"
        )
        property_operator = LinearOperator.from_matrix(
            model, EuclideanSpace(2), rng.normal(size=(2, 5)), form="components"
        )
        truth = model.random(rng=rng)
        data = forward(truth)
        prior = Ball(model, radius=1.5 * model.norm(truth))
        noise = Ball(data_space, radius=0.1 * data_space.norm(data))
        linear = FeasiblePropertySet.from_structure(
            FeasibleStructure(
                forward, property_operator, prior, noise=noise, solver=None
            ),
            data,
        )
        nonlinear = NonlinearFeasiblePropertySet(
            forward, property_operator, prior, data, noise=noise
        )
        for angle in (0.0, 1.0, 2.5, 4.0):
            direction = np.array([np.cos(angle), np.sin(angle)])
            outer = linear.support(direction)
            result = nonlinear.support_result(direction)
            assert nonlinear.is_feasible(result.extremal_model)
            assert result.inner <= outer + 1e-9 * abs(outer)
            assert result.inner == pytest.approx(outer, rel=1e-4)


class TestLocalMembership:
    def test_a_value_between_two_attained_ones_is_proved_inside(self):
        feasible = two_parameter_set(3, 1.0, 0.05)
        upper = feasible.support_result(np.array([1.0])).inner
        lower = -feasible.support_result(np.array([-1.0])).inner
        value = np.array([0.5 * (lower + upper)])
        result = feasible.inclusion_result(value)
        assert result.level <= feasible.level
        assert feasible.is_feasible(result.fitting_model)
        assert COST @ result.fitting_model == pytest.approx(value[0], abs=1e-13)

    def test_a_value_beyond_the_exact_support_is_not_proved(self):
        """The control: the closed form says no model has this property."""
        feasible = two_parameter_set(1, 1.0, 0.2)
        beyond = known_support(1, 1.0, 0.2) + 0.01
        result = feasible.inclusion_result(np.array([beyond]))
        assert result.level > feasible.level
        assert result.lower_bound == float("-inf")

    def test_the_level_of_a_fully_observed_model_is_its_own(self):
        """With the property the model itself, the only model with a value is the value."""
        model, forward, _, data = two_parameter_problem(3)
        feasible = NonlinearFeasiblePropertySet(
            forward,
            LinearOperator.identity(model),
            Ball(model, radius=1.0),
            data,
            noise=Ball(forward.codomain, radius=0.05),
        )
        result = feasible.inclusion_result(TRUE_MODEL)
        assert result.level == pytest.approx(TRUE_MODEL @ TRUE_MODEL, rel=1e-12)
        assert np.allclose(result.fitting_model, TRUE_MODEL, atol=1e-13)

    def test_membership_is_not_decided_without_a_model(self):
        feasible = two_parameter_set(3, 1.0, 0.05)
        assert not feasible.has_membership
        with pytest.raises(NotImplementedError, match="inclusion_result"):
            feasible.contains(np.array([0.2]))


class TestRefusals:
    def test_exact_data_are_refused_by_name(self):
        model, forward, _, data = two_parameter_problem(3)
        with pytest.raises(ValueError, match="small ball"):
            NonlinearFeasiblePropertySet(
                forward,
                LinearOperator.identity(model),
                Ball(model, radius=1.0),
                data,
                noise=Ball(forward.codomain, radius=0.0),
            )

    def test_a_prior_without_a_smooth_level_function_is_refused(self):
        model, forward, _, data = two_parameter_problem(3)
        box = Polytope(
            model,
            [HalfSpace(model, model.basis_vector(i), offset=1.0) for i in range(2)],
            outer=True,
        )
        with pytest.raises(TypeError, match="level function"):
            NonlinearFeasiblePropertySet(
                forward,
                LinearOperator.identity(model),
                box,
                data,
                noise=Ball(forward.codomain, radius=0.05),
            )

    def test_a_margin_within_the_tolerance_is_refused(self):
        with pytest.raises(ValueError, match="margin"):
            LocalSearch(tolerance=1e-6, margin=1e-6)


class TestASetInSeveralParts:
    """Values and squares on an interval: the sign at a squared point splits the set."""

    def setting(self):
        space = line.Sobolev(64, 2.0, 0.1)
        truth = space.sobolev_measure(2.0, 0.2, pointwise_std=1.0).sample(
            rng=np.random.default_rng(5)
        )
        linear_points, squared_points = [0.15, 0.55, 0.85], [0.3, 0.45, 0.7, 0.95]
        forward = values_and_squares(space, linear_points, squared_points)
        data = forward(truth)
        feasible = NonlinearFeasiblePropertySet(
            forward,
            space.point_evaluation_operator([0.5]),
            Ball(space, radius=1.5 * space.norm(truth)),
            data,
            noise=Ball(forward.codomain, radius=0.2),
        )
        evaluation = space.point_evaluation_operator(
            linear_points + squared_points, route="dense"
        )
        squares = space.point_evaluation_operator(squared_points)
        return space, feasible, data, evaluation, squares

    def interpolant(self, evaluation, data, signs):
        """The smallest field with the linear data and these signs of the squared ones' roots."""
        gram = np.array([evaluation(evaluation.adjoint(e)) for e in np.eye(7)])
        target = np.concatenate([data[:3], np.array(signs) * np.sqrt(data[3:])])
        return evaluation.adjoint(np.linalg.solve(gram, target))

    def test_a_second_start_reaches_beyond_the_first(self):
        space, feasible, data, evaluation, squares = self.setting()
        direction = np.array([1.0])
        first = feasible.support_result(direction)
        other = feasible.support_result(
            direction, start=self.interpolant(evaluation, data, (-1, 1, -1, -1))
        )
        for result in (first, other):
            assert feasible.is_feasible(result.extremal_model)
        assert tuple(np.sign(squares(first.extremal_model))) == (-1, -1, -1, -1)
        assert tuple(np.sign(squares(other.extremal_model))) == (-1, 1, -1, -1)
        # The start at the prior's center finds one part of the set; the
        # other part holds values of the property the first does not.
        assert other.inner > first.inner + 0.1
        # And the set linearized about a model of the first part is below both.
        linearized = feasible.linearized(first.extremal_model).support(direction)
        assert linearized < other.inner - 0.1
