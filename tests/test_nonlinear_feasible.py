"""The feasible property set of a nonlinear forward problem: fixtures and what they establish.

Small problems whose true feasible set is known by other means, so that a
bound on it can be judged: a two-parameter attenuation map whose support is
exact for one datum and found by a zooming search for several, a quadratic
map whose curvature is known in closed form, and point observations of a
field and of its square on an interval.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from pygeoinf import (
    BackusGilbertParker,
    EuclideanSpace,
    ForwardProblem,
    GaussianMeasure,
    LinearOperator,
    Operator,
)
from pygeoinf.algebra.operators import EvaluationError, LinearFunctional
from pygeoinf.geometry.convex import Ball, HalfSpace, Polytope
from pygeoinf.geometry.feasible import FeasiblePropertySet, FeasibleStructure
from pygeoinf.inference import NonlinearBackusGilbertParker
from pygeoinf.geometry.nonlinear import (
    BranchAndPrune,
    Cell,
    ComponentwiseRemainder,
    LocalSearch,
    NonlinearFeasiblePropertySet,
    NormRemainder,
    SampledRemainder,
)
from pygeoinf.symmetric_space import line, plane
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


# --------------------------------------------------------------------- #
#                         The set on a cell                              #
# --------------------------------------------------------------------- #


def attenuation_remainder(forward, paths):
    """The certified one-sided remainder of the attenuation map: ``g'' = exp(−t)``."""
    return ComponentwiseRemainder(
        forward.codomain, list(paths), lambda low, high: (np.exp(-high), np.exp(-low))
    )


def attenuation_norm_remainder(forward, weights, cell):
    """A certified norm bound over a cell's ball: ``|(sup exp(−a_i·m) |a_i|^2)_i|``."""
    norms = np.linalg.norm(weights, axis=1)
    largest = np.exp(-(weights @ cell.center) + cell.radius * norms)
    return NormRemainder(forward.codomain, float(np.linalg.norm(largest * norms**2)))


def square_cell(feasible, center, side):
    """A square of the plane about a model, as two slabs, with its circumscribed radius."""
    half = 0.5 * side
    return feasible.cell(
        center,
        directions=[np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        half_widths=[half, half],
        radius=half * np.sqrt(2.0),
    )


class TestTheCell:
    def setting(self, rng):
        space = make_dense_metric_space(5)
        directions = [space.random(rng=rng) for _ in range(2)]
        cell = Cell(
            space,
            space.random(rng=rng),
            directions=directions,
            half_widths=[0.3, 0.1],
            radius=2.0,
        )
        return space, cell

    def models(self, space, cell, rng, count=400):
        """Models of the cell: random offsets within the slabs, random steps across them."""
        found = []
        while len(found) < count:
            inside = cell.point(rng.uniform(-1.0, 1.0, 2) * cell.half_widths)
            across = space.random(rng=rng)
            weights = np.linalg.solve(
                np.array(
                    [
                        [space.inner_product(a, b) for b in cell.directions]
                        for a in cell.directions
                    ]
                ),
                np.array([space.inner_product(u, across) for u in cell.directions]),
            )
            for weight, direction in zip(weights, cell.directions):
                across = space.axpy(-float(weight), direction, across)
            model = space.axpy(rng.uniform(0.0, 1.0), across, inside)
            if cell.contains(model):
                found.append(model)
        return found

    def test_offsets_and_points_are_inverse(self, rng):
        space, cell = self.setting(rng)
        offsets = np.array([0.12, -0.07])
        assert np.allclose(cell.offsets(cell.point(offsets)), offsets, atol=1e-13)
        assert cell.contains(cell.point(offsets))
        assert not cell.contains(cell.point([0.31, 0.0]))

    def test_the_half_width_of_a_slab_direction_is_exact(self, rng):
        space, cell = self.setting(rng)
        for direction, half_width in zip(cell.directions, cell.half_widths):
            assert cell.half_width(direction) == pytest.approx(half_width, rel=1e-12)
            assert cell.half_width(space.scale(-3.0, direction)) == pytest.approx(
                3.0 * half_width, rel=1e-12
            )

    def test_the_half_width_bounds_every_model_of_the_cell(self, rng):
        space, cell = self.setting(rng)
        models = self.models(space, cell, rng)
        mixed = space.add(cell.directions[0], space.scale(2.0, cell.directions[1]))
        for vector in (mixed, space.random(rng=rng), space.random(rng=rng)):
            reach = max(
                abs(space.inner_product(vector, space.subtract(model, cell.center)))
                for model in models
            )
            assert reach <= cell.half_width(vector) * (1.0 + 1e-12)
        # Inside the slabs' span the bound owes nothing to the radius.
        assert cell.half_width(mixed) == pytest.approx(0.3 + 2.0 * 0.1, rel=1e-12)

    def test_dependent_directions_are_refused(self, rng):
        space = make_dense_metric_space(5)
        direction = space.random(rng=rng)
        with pytest.raises(ValueError, match="independent"):
            Cell(
                space,
                space.zero(),
                directions=[direction, space.scale(2.0, direction)],
                half_widths=[1.0, 1.0],
            )

    def test_a_ball_prior_bounds_the_radius(self):
        feasible = two_parameter_set(3, 1.0, 0.05)
        cell = feasible.cell(np.array([0.3, 0.4]))
        assert cell.radius == pytest.approx(1.5)
        assert feasible.cell(np.array([0.3, 0.4]), radius=0.2).radius == 0.2


class TestRemainderBounds:
    def sampled_remainders(self, forward, cell, models):
        linearization = forward.at(cell.center)
        space, data_space = forward.domain, forward.codomain
        out = []
        for model in models:
            step = linearization.derivative(space.subtract(model, cell.center))
            out.append(
                data_space.subtract(
                    data_space.subtract(forward(model), linearization.value), step
                )
            )
        return out

    def dense_setting(self, rng):
        model = make_dense_metric_space(4)
        paths = [model.scale(0.5, model.random(rng=rng)) for _ in range(3)]
        forward = attenuation(model, paths)
        cell = Cell(
            model,
            model.scale(0.2, model.random(rng=rng)),
            directions=paths,
            half_widths=[0.3, 0.2, 0.4],
        )
        models = [
            cell.point(rng.uniform(-1.0, 1.0, 3) * cell.half_widths) for _ in range(200)
        ]
        return model, paths, forward, cell, models

    def test_the_componentwise_set_holds_the_remainder(self, rng):
        _, paths, forward, cell, models = self.dense_setting(rng)
        claimed = attenuation_remainder(forward, paths).remainder_set(cell)
        assert all(
            claimed.contains(remainder)
            for remainder in self.sampled_remainders(forward, cell, models)
        )
        # One-sided: the attenuation's second derivative is positive.
        assert np.all(claimed.lower == 0.0) and np.all(claimed.upper > 0.0)

    def test_an_understated_curvature_does_not(self, rng):
        """The control: a fifth of the true second derivative is caught by the same models."""
        _, paths, forward, cell, models = self.dense_setting(rng)
        understated = ComponentwiseRemainder(
            forward.codomain,
            paths,
            lambda low, high: (0.2 * np.exp(-high), 0.2 * np.exp(-low)),
        ).remainder_set(cell)
        assert not all(
            understated.contains(remainder)
            for remainder in self.sampled_remainders(forward, cell, models)
        )

    def test_the_norm_bound_holds_the_remainder(self, rng):
        feasible = two_parameter_set(3, 1.0, 0.05)
        _, forward, weights, _ = two_parameter_problem(3)
        cell = square_cell(feasible, np.array([0.2, -0.1]), 0.5)
        claimed = attenuation_norm_remainder(forward, weights, cell).remainder_set(cell)
        models = [
            cell.point(rng.uniform(-1.0, 1.0, 2) * cell.half_widths) for _ in range(200)
        ]
        assert all(
            claimed.contains(remainder)
            for remainder in self.sampled_remainders(forward, cell, models)
        )

    def test_the_square_s_remainder_is_exact_at_a_slab_s_end(self):
        """For ``u^2`` the remainder is the squared step, so the box's corner is attained."""
        space = line.Sobolev(64, 2.0, 0.1)
        linear_points, squared_points = [0.2, 0.7], [0.4, 0.9]
        forward = values_and_squares(space, linear_points, squared_points)
        evaluation = space.point_evaluation_operator(
            linear_points + squared_points, route="dense"
        )
        rows = [evaluation.adjoint(e) for e in np.eye(4)]
        curvature = np.array([0.0, 0.0, 2.0, 2.0])
        remainder = ComponentwiseRemainder(
            forward.codomain, rows, lambda low, high: (curvature, curvature)
        )
        center = space.sobolev_measure(2.0, 0.2).sample(rng=np.random.default_rng(1))
        cell = Cell(
            space, center, directions=rows[2:], half_widths=[0.3, 0.1], radius=5.0
        )
        claimed = remainder.remainder_set(cell)
        assert np.allclose(claimed.lower, 0.0)
        assert np.allclose(claimed.upper, [0.0, 0.0, 0.09, 0.01], atol=1e-14)
        corner = self.sampled_remainders(forward, cell, [cell.point([0.3, -0.1])])[0]
        assert np.allclose(corner, [0.0, 0.0, 0.09, 0.01], atol=1e-10)

    def test_a_sampled_remainder_is_an_estimate_and_says_so(self, rng):
        feasible = two_parameter_set(3, 1.0, 0.05)
        _, forward, weights, _ = two_parameter_problem(3)
        cell = square_cell(feasible, np.array([0.2, -0.1]), 0.5)
        sampled = SampledRemainder(forward, samples=16, rng=rng)
        assert not sampled.certified
        assert attenuation_remainder(forward, weights).certified
        estimate = sampled.remainder_set(cell)
        corners = [
            cell.point(np.array(signs) * cell.half_widths)
            for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1))
        ]
        assert all(
            estimate.contains(remainder)
            for remainder in self.sampled_remainders(forward, cell, corners)
        )
        with pytest.raises(ValueError, match="slabs"):
            sampled.remainder_set(feasible.cell(np.array([0.2, -0.1])))


class TestTheSandwichOnACell:
    DIRECTION = np.array([1.0])

    @pytest.mark.parametrize("count, radius, sigma", [(1, 1.0, 0.2), (3, 1.0, 0.05)])
    def test_the_known_support_lies_between_the_two_ends(self, count, radius, sigma):
        """Inner, truth, outer, in that order, and closer as the cell shrinks."""
        feasible = two_parameter_set(count, radius, sigma)
        _, forward, weights, _ = two_parameter_problem(count)
        truth = known_support(count, radius, sigma)
        center = feasible.support_result(self.DIRECTION).extremal_model
        box = attenuation_remainder(forward, weights)
        previous = float("inf")
        for side in (0.5, 0.125, 0.03125):
            cell = square_cell(feasible, center, side)
            ball = attenuation_norm_remainder(forward, weights, cell)
            by_box = feasible.cell_support_result(self.DIRECTION, cell, box)
            by_ball = feasible.cell_support_result(self.DIRECTION, cell, ball)
            for result in (by_box, by_ball):
                assert result.converged and result.certified and not result.refuted
                assert feasible.is_feasible(result.extremal_model)
                assert result.inner <= truth + 1e-6
                assert result.value >= truth - 1e-6
            # The one-sided box is inside the ball of the norm bound.
            assert by_box.value <= by_ball.value + 1e-6
            assert by_box.value <= previous + 1e-9
            previous = by_box.value
        assert previous - truth <= 2e-3

    def test_a_linear_map_needs_no_remainder(self, rng):
        """Two routes: the cell's dual with slab rows, and the linear set itself."""
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
        nothing = NormRemainder(data_space, 0.0)
        # Slabs wider than the prior cut nothing away.
        slack = [model.random(rng=rng) for _ in range(2)]
        wide = [4.0 * prior.radius * model.norm(u) for u in slack]
        whole = nonlinear.cell(truth, directions=slack, half_widths=wide)
        for angle in (0.3, 2.0, 4.4):
            direction = np.array([np.cos(angle), np.sin(angle)])
            exact = linear.support(direction)
            for cell in (nonlinear.cell(truth), whole):
                result = nonlinear.cell_support_result(direction, cell, nothing)
                assert result.value == pytest.approx(exact, rel=1e-5)
                assert result.inner == pytest.approx(exact, rel=1e-4)
                assert not result.refuted

    def test_slabs_cut_the_linear_set(self, rng):
        """The control for the slab rows: a thin slab lowers the support to what it allows."""
        model = EuclideanSpace(3)
        forward = LinearOperator.from_matrix(
            model, EuclideanSpace(1), np.array([[1.0, 0.0, 0.0]]), form="components"
        )
        property_operator = LinearOperator.from_matrix(
            model, EuclideanSpace(1), np.array([[0.0, 1.0, 0.0]]), form="components"
        )
        nonlinear = NonlinearFeasiblePropertySet(
            forward,
            property_operator,
            Ball(model, radius=1.0),
            np.array([0.0]),
            noise=Ball(EuclideanSpace(1), radius=0.1),
        )
        cell = nonlinear.cell(
            model.zero(), directions=[np.array([0.0, 1.0, 0.0])], half_widths=[0.25]
        )
        result = nonlinear.cell_support_result(
            np.array([1.0]), cell, NormRemainder(EuclideanSpace(1), 0.0)
        )
        assert result.value == pytest.approx(0.25, rel=1e-6)

    def test_a_cell_away_from_every_fitting_model_is_shown_empty(self):
        feasible = two_parameter_set(3, 1.0, 0.05)
        _, forward, weights, data = two_parameter_problem(3)
        cell = square_cell(feasible, np.array([-0.5, 0.5]), 0.2)
        result = feasible.cell_support_result(
            self.DIRECTION, cell, attenuation_remainder(forward, weights)
        )
        assert result.value == float("-inf") and result.inner == float("-inf")
        assert result.certified
        # The other route: no point of a fine grid on the cell fits.
        axis = np.linspace(-0.1, 0.1, 201)
        grid = cell.center + np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
        misfits = np.linalg.norm(np.exp(-(grid @ weights.T)) - data, axis=1)
        assert misfits.min() > 0.05

    def test_an_understated_remainder_is_refuted_by_a_feasible_model(self):
        """The control: with no remainder allowed, the "outer" value falls below an attained one."""
        feasible = two_parameter_set(3, 1.0, 0.05)
        _, forward, weights, _ = two_parameter_problem(3)
        cell = square_cell(feasible, np.zeros(2), 1.0)
        nothing = ComponentwiseRemainder(
            forward.codomain,
            list(weights),
            lambda low, high: (np.zeros(3), np.zeros(3)),
            certified=False,
        )
        result = feasible.cell_support_result(self.DIRECTION, cell, nothing)
        assert feasible.is_feasible(result.extremal_model)
        assert result.value < result.inner - 1e-2
        assert result.refuted and not result.certified
        honest = feasible.cell_support_result(
            self.DIRECTION, cell, attenuation_remainder(forward, weights)
        )
        assert honest.value >= honest.inner and not honest.refuted

    def test_the_inner_set_is_made_of_feasible_models(self):
        feasible = two_parameter_set(1, 1.0, 0.2)
        _, forward, weights, _ = two_parameter_problem(1)
        center = feasible.support_result(self.DIRECTION).extremal_model
        remainder = attenuation_remainder(forward, weights)
        # Too large a cell leaves nothing of the noise ball.
        assert (
            feasible.inner_set(square_cell(feasible, np.zeros(2), 2.0), remainder)
            is None
        )
        inner = feasible.inner_set(square_cell(feasible, center, 0.0625), remainder)
        outer = feasible.outer_set(square_cell(feasible, center, 0.0625), remainder)
        truth = known_support(1, 1.0, 0.2)
        result = inner.support_result(self.DIRECTION)
        assert feasible.is_feasible(result.extremal_model)
        assert result.inner <= truth + 1e-9
        assert outer.support(self.DIRECTION) >= truth - 1e-9

    def test_a_noise_set_that_is_not_a_ball_is_refused_by_name(self):
        from pygeoinf.geometry.convex import Ellipsoid

        model, forward, weights, data = two_parameter_problem(3)
        data_space = forward.codomain
        noise = Ellipsoid(data_space, LinearOperator.identity(data_space) * 400.0)
        feasible = NonlinearFeasiblePropertySet(
            forward,
            LinearOperator.identity(model),
            Ball(model, radius=1.0),
            data,
            noise=noise,
        )
        with pytest.raises(TypeError, match="ball"):
            feasible.outer_set(
                feasible.cell(model.zero()), NormRemainder(data_space, 1.0)
            )


class TestACoverOfASetInSeveralParts:
    """Values and squares: one slab cell per sign pattern at the squared points covers the set."""

    def cover(self):
        _, feasible, data, evaluation, _ = TestASetInSeveralParts().setting()
        sigma = feasible.noise.radius
        rows = [evaluation.adjoint(e) for e in np.eye(7)]
        gram = np.array([evaluation(row) for row in rows])
        curvature = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0])
        remainder = ComponentwiseRemainder(
            feasible.data_space, rows, lambda low, high: (curvature, curvature)
        )
        # |u(y)^2 − d| <= sigma puts |u(y)| between two roots, whatever its sign.
        high = np.sqrt(data[3:] + sigma)
        low = np.sqrt(np.maximum(data[3:] - sigma, 0.0))

        def cell(signs):
            target = np.concatenate([data[:3], np.array(signs) * 0.5 * (high + low)])
            center = evaluation.adjoint(np.linalg.solve(gram, target))
            return feasible.cell(
                center, directions=rows[3:], half_widths=0.5 * (high - low)
            )

        return feasible, remainder, cell

    def test_a_part_is_bracketed_and_an_impossible_pattern_is_shown_empty(self):
        feasible, remainder, cell = self.cover()
        direction = np.array([1.0])
        results = {
            signs: feasible.cell_support_result(direction, cell(signs), remainder)
            for signs in ((-1, -1, -1, -1), (-1, 1, -1, -1), (1, 1, 1, 1))
        }
        for signs in ((-1, -1, -1, -1), (-1, 1, -1, -1)):
            result = results[signs]
            assert result.certified and not result.refuted and result.converged
            assert feasible.is_feasible(result.extremal_model)
            assert 0.0 <= result.value - result.inner <= 0.05
        assert results[(-1, 1, -1, -1)].inner > results[(-1, -1, -1, -1)].value
        assert results[(1, 1, 1, 1)].value == float("-inf")

    @pytest.mark.slow
    def test_the_sixteen_cells_bracket_the_whole_set(self):
        """Every feasible model is in one of the cells, so the largest outer value bounds the set."""
        import itertools

        feasible, remainder, cell = self.cover()
        direction = np.array([1.0])
        results = [
            feasible.cell_support_result(direction, cell(signs), remainder)
            for signs in itertools.product((-1, 1), repeat=4)
        ]
        occupied = [result for result in results if result.value > float("-inf")]
        assert len(occupied) == 4
        assert all(result.inner > float("-inf") for result in occupied)
        outer = max(result.value for result in occupied)
        inner = max(result.inner for result in occupied)
        assert inner == pytest.approx(0.2335, abs=1e-3)
        assert 0.0 <= outer - inner <= 0.03


# --------------------------------------------------------------------- #
#                        The search over cells                           #
# --------------------------------------------------------------------- #


class TestBranchAndPrune:
    DIRECTION = np.array([1.0])

    @pytest.mark.parametrize(
        "count, radius, sigma", [(1, 1.0, 0.02), (1, 1.0, 0.2), (3, 0.5, 0.05)]
    )
    def test_the_bracket_holds_the_known_support_and_closes(self, count, radius, sigma):
        feasible = two_parameter_set(count, radius, sigma)
        _, forward, weights, _ = two_parameter_problem(count)
        truth = known_support(count, radius, sigma)
        method = BranchAndPrune(attenuation_remainder(forward, weights), rtol=1e-2)
        result = feasible.support_result(self.DIRECTION, via=method)
        assert result.algorithm == "branch"
        assert result.converged and result.certified and not result.refuted
        assert feasible.is_feasible(result.extremal_model)
        assert result.inner <= truth + 1e-6 <= result.value + 2e-6
        # The tolerance is relative to the prior's width along the direction.
        width = 2.0 * radius * np.linalg.norm(COST)
        assert result.value - result.inner <= 1e-2 * width
        assert {record.status for record in result.cells} <= {
            "empty",
            "pruned",
            "split",
            "live",
        }
        assert all(
            record.value == float("-inf")
            for record in result.cells
            if record.status == "empty"
        )
        outer = [upper for _, upper, _ in result.history]
        inner = [lower for _, _, lower in result.history]
        assert all(b <= a + 1e-12 for a, b in zip(outer, outer[1:]))
        assert all(b >= a for a, b in zip(inner, inner[1:]))

    def test_the_root_cell_is_cut_along_independent_rows(self):
        """Three path vectors in the plane: two of them span it."""
        feasible = two_parameter_set(3, 1.0, 0.05)
        _, forward, weights, _ = two_parameter_problem(3)
        root = BranchAndPrune(attenuation_remainder(forward, weights)).root(feasible)
        assert len(root.directions) == 2
        norms = sorted(np.linalg.norm(weights, axis=1))
        # As wide as the unit ball's reach along each chosen row.
        assert all(
            any(width == pytest.approx(norm) for norm in norms)
            for width in root.half_widths
        )

    def test_a_budget_that_runs_out_still_brackets(self):
        feasible = two_parameter_set(3, 1.0, 0.05)
        _, forward, weights, _ = two_parameter_problem(3)
        truth = known_support(3, 1.0, 0.05)
        method = BranchAndPrune(
            attenuation_remainder(forward, weights), rtol=1e-6, max_cells=5
        )
        result = feasible.support_result(self.DIRECTION, via=method)
        assert not result.converged
        assert "budget" in result.message
        assert len(result.cells) <= 5
        assert result.inner <= truth + 1e-6 <= result.value + 2e-6
        assert result.live

    def test_a_linear_map_is_answered_by_one_cell(self, rng):
        """Two routes: the search over cells, and the linear set."""
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
        method = BranchAndPrune(
            NormRemainder(data_space, 0.0),
            directions=[model.random(rng=rng) for _ in range(2)],
        )
        # A remainder that depends on no slab leaves the root cell without any.
        assert not method.root(nonlinear).directions
        direction = np.array([0.6, -0.8])
        result = nonlinear.support_result(direction, via=method)
        assert len(result.cells) == 1 and result.converged and not result.refuted
        assert result.value == pytest.approx(linear.support(direction), rel=1e-5)
        assert result.inner == pytest.approx(result.value, rel=1e-3)

    def test_a_false_claim_is_refuted_by_the_model_it_misses(self):
        """The control: no remainder allowed, so the search is the linearized set, and wrong."""
        feasible = two_parameter_set(3, 1.0, 0.05)
        _, forward, weights, _ = two_parameter_problem(3)
        nothing = ComponentwiseRemainder(
            forward.codomain,
            list(weights),
            lambda low, high: (np.zeros(3), np.zeros(3)),
            certified=False,
        )
        result = feasible.support_result(self.DIRECTION, via=BranchAndPrune(nothing))
        assert result.value < result.inner - 1e-2
        assert result.refuted and not result.certified
        assert "contradicts" in result.message

    def test_directions_are_needed_from_somewhere(self):
        with pytest.raises(ValueError, match="directions"):
            BranchAndPrune(NormRemainder(EuclideanSpace(3), 1.0))

    def setting(self):
        _, feasible, _, evaluation, squares = TestASetInSeveralParts().setting()
        rows = [evaluation.adjoint(e) for e in np.eye(7)]
        curvature = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0])
        remainder = ComponentwiseRemainder(
            feasible.data_space, rows, lambda low, high: (curvature, curvature)
        )
        return feasible, remainder, squares

    def test_the_search_finds_a_part_the_first_local_search_missed(self):
        """Values and squares: a few splits reach the part that holds the support."""
        feasible, remainder, squares = self.setting()
        first = feasible.support_result(self.DIRECTION)
        method = BranchAndPrune(remainder, max_cells=13)
        # The root cell keeps a slab for each squared datum and none for a linear one.
        assert len(method.root(feasible).directions) == 4
        result = feasible.support_result(self.DIRECTION, via=method)
        assert not result.converged and not result.refuted
        assert result.inner > first.inner + 0.1
        assert result.value >= result.inner
        assert tuple(np.sign(squares(result.extremal_model))) == (-1, 1, -1, -1)

    @pytest.mark.slow
    def test_the_search_brackets_a_set_in_several_parts(self):
        feasible, remainder, _ = self.setting()
        result = feasible.support_result(
            self.DIRECTION, via=BranchAndPrune(remainder, max_cells=200)
        )
        assert result.converged and result.certified and not result.refuted
        assert result.inner == pytest.approx(0.2335, abs=1e-3)
        assert 0.0 <= result.value - result.inner <= 5e-3


class TestMembershipByCells:
    """A value proved inside by a model, or outside by a cover that excludes it."""

    def setting(self):
        feasible = two_parameter_set(1, 1.0, 0.2)
        _, forward, weights, _ = two_parameter_problem(1)
        method = BranchAndPrune(attenuation_remainder(forward, weights))
        return feasible, method, known_support(1, 1.0, 0.2)

    def test_a_value_inside_is_proved_by_a_model(self):
        feasible, method, truth = self.setting()
        value = np.array([truth - 0.05])
        result = feasible.inclusion_result(value, via=method)
        assert result.converged and result.algorithm == "branch"
        assert result.level <= feasible.level
        assert feasible.is_feasible(result.fitting_model)
        assert COST @ result.fitting_model == pytest.approx(value[0], abs=1e-13)
        assert result.lower_bound == float("-inf")

    @pytest.mark.parametrize("beyond", [0.02, 0.3])
    def test_a_value_beyond_the_exact_support_is_proved_outside(self, beyond):
        """Two routes: the closed form says no model has it, and so does the cover."""
        feasible, method, truth = self.setting()
        result = feasible.inclusion_result(np.array([truth + beyond]), via=method)
        assert result.converged and result.certified
        assert result.fitting_model is None
        assert result.lower_bound == float("inf")
        statuses = {record.status for record in result.cells}
        assert statuses <= {"empty", "split"}

    def test_a_budget_that_runs_out_decides_nothing(self):
        feasible, _, truth = self.setting()
        _, forward, weights, _ = two_parameter_problem(1)
        method = BranchAndPrune(attenuation_remainder(forward, weights), max_cells=3)
        result = feasible.inclusion_result(np.array([truth + 1e-3]), via=method)
        assert not result.converged
        assert result.fitting_model is None
        assert result.lower_bound == float("-inf") and result.level == float("inf")
        assert "budget" in result.message

    def test_an_exclusion_carries_its_claim_s_standing(self):
        feasible, _, truth = self.setting()
        _, forward, weights, _ = two_parameter_problem(1)
        estimated = ComponentwiseRemainder(
            forward.codomain,
            list(weights),
            lambda low, high: (np.exp(-high), np.exp(-low)),
            certified=False,
        )
        result = feasible.inclusion_result(
            np.array([truth + 0.3]), via=BranchAndPrune(estimated)
        )
        assert result.lower_bound == float("inf") and not result.certified


# --------------------------------------------------------------------- #
#             A function of a field under line integrals                 #
# --------------------------------------------------------------------- #


@functools.lru_cache
def ray_problem(*, rays=10, nodes=8, std=0.05, seed=4):
    """Travel times ``int exp(m) ds`` along straight rays across a square.

    The quadrature written out: evaluation at the nodes, the exponential of
    each value, and the weights. Returns the space, the three operators,
    the composed map and the prior measure.
    """
    rng = np.random.default_rng(seed)
    space = plane.Sobolev((12, 12), 3.0, 0.3, bounds=((0.0, 1.0), (0.0, 1.0)))
    belief = space.sobolev_measure(3.0, 0.3, pointwise_std=std)
    points, weights, owner = [], [], []
    for ray in range(rays):
        start = np.array([rng.uniform(0.1, 0.9), 0.05])
        end = np.array([rng.uniform(0.1, 0.9), 0.95])
        if ray % 2:
            start, end = start[::-1].copy(), end[::-1].copy()
        along, lengths = space.geodesic_quadrature(start, end, count=nodes)
        points += list(along)
        weights += list(lengths)
        owner += [ray] * nodes
    evaluation = space.point_evaluation_operator(points, route="dense")
    values, times = evaluation.codomain, EuclideanSpace(rays)
    matrix = np.zeros((rays, len(points)))
    matrix[owner, np.arange(len(points))] = weights
    integrate = LinearOperator.from_matrix(values, times, matrix, form="components")

    def diagonal(entries):
        return LinearOperator.from_matrix(
            values, values, np.diag(entries), form="components"
        )

    def exponential(samples):
        samples = np.asarray(samples)
        if np.any(samples > 50.0):
            raise EvaluationError("The exponential overflows at this model.")
        return np.exp(samples)

    pointwise = Operator.from_callables(
        values,
        values,
        exponential,
        derivative=lambda samples: diagonal(np.exp(np.asarray(samples))),
        second_derivative=lambda samples, step: diagonal(
            np.exp(np.asarray(samples)) * np.asarray(step)
        ),
    )
    forward = integrate @ pointwise @ evaluation
    return space, evaluation, integrate, forward, belief


def exponential_curvature(low, high):
    """The second derivative of the exponential over an interval: between its ends' values."""
    return np.exp(low), np.exp(high)


def covering_directions(space, evaluation, integrate, count):
    """The model directions the rays sample most: leading eigenvectors of ``B* W B``."""
    from scipy.linalg import eigh

    sampled = evaluation.matrix(form="components")
    lengths = integrate.matrix(form="components").sum(axis=0)
    gram = LinearOperator.identity(space).matrix(form="galerkin")
    values, vectors = eigh(sampled.T @ (lengths[:, None] * sampled), gram)
    leading = np.argsort(-values)[:count]
    return [space.from_components(vectors[:, index]) for index in leading]


class TestTheHalfWidthOfASlabCellInABall:
    """``sup |(v, m − center)|`` over slabs cut by a ball, for orthonormal slab directions."""

    def setting(self, rng):
        space = EuclideanSpace(5)
        frame = np.linalg.qr(rng.normal(size=(5, 5)))[0]
        directions = [frame[:, 0], frame[:, 1], frame[:, 2]]
        cell = Cell(
            space,
            rng.normal(size=5),
            directions=directions,
            half_widths=[0.2, 0.9, 3.0],
            radius=1.0,
        )
        return space, cell

    def supremum(self, cell, vector):
        """The same number by a constrained maximization over the model."""
        from scipy.optimize import minimize

        constraints = [
            {"type": "ineq", "fun": lambda d: cell.radius**2 - d @ d},
        ]
        for direction, half_width in zip(cell.directions, cell.half_widths):
            constraints.append(
                {"type": "ineq", "fun": lambda d, u=direction, w=half_width: w - u @ d}
            )
            constraints.append(
                {"type": "ineq", "fun": lambda d, u=direction, w=half_width: w + u @ d}
            )
        best = 0.0
        for start in (0.1 * vector, -0.1 * vector, np.zeros(5)):
            found = minimize(
                lambda d: -(vector @ d),
                start,
                constraints=constraints,
                method="SLSQP",
                options={"ftol": 1e-14, "maxiter": 500},
            )
            best = max(best, -found.fun)
        return best

    def test_it_is_the_supremum(self, rng):
        """Two routes to one number: the closed iteration, and an optimizer."""
        space, cell = self.setting(rng)
        for _ in range(6):
            vector = rng.normal(size=5)
            assert cell.half_width(vector) == pytest.approx(
                self.supremum(cell, vector), rel=1e-6
            )

    def test_it_is_below_both_cruder_bounds(self, rng):
        space, cell = self.setting(rng)
        vector = rng.normal(size=5)
        along = np.array([u @ vector for u in cell.directions])
        rest = np.sqrt(vector @ vector - along @ along)
        split = np.abs(along) @ cell.half_widths + rest * cell.radius
        whole = np.linalg.norm(vector) * cell.radius
        assert cell.half_width(vector) <= min(split, whole) * (1.0 + 1e-12)
        # A slab wider than the ball is no constraint, and the bound knows it.
        wide = Cell(
            space,
            cell.center,
            directions=cell.directions,
            half_widths=[5, 5, 5],
            radius=1.0,
        )
        assert wide.half_width(vector) == pytest.approx(whole, rel=1e-10)

    def test_several_vectors_at_once_agree_with_one_at_a_time(self, rng):
        space, cell = self.setting(rng)
        vectors = rng.normal(size=(7, 5))
        pairings = np.array([[u @ v for u in cell.directions] for v in vectors])
        together = cell.bounds_along(pairings, np.linalg.norm(vectors, axis=1))
        apart = [cell.half_width(v) for v in vectors]
        assert np.allclose(together, apart, rtol=1e-12)


class TestTheRemainderOfAFunctionUnderLineIntegrals:
    def setting(self, rng):
        space, evaluation, integrate, forward, belief = ray_problem()
        remainder = ComponentwiseRemainder(
            forward.codomain, evaluation, exponential_curvature, then=integrate
        )
        directions = covering_directions(space, evaluation, integrate, 4)
        prior = belief.ambient_ball(level=0.95)
        return (
            space,
            evaluation,
            integrate,
            forward,
            belief,
            remainder,
            directions,
            prior,
        )

    def test_the_claimed_set_holds_the_remainder(self, rng):
        space, _, _, forward, belief, remainder, directions, prior = self.setting(rng)
        center = belief.sample(rng=rng)
        cell = Cell(
            space,
            center,
            directions=directions,
            half_widths=[0.02] * 4,
            radius=prior.radius,
        )
        claimed = remainder.remainder_set(cell)
        linearization = forward.at(center)
        inside = 0
        for _ in range(200):
            step = belief.sample(rng=rng)
            step = space.scale(rng.uniform(0.0, prior.radius) / space.norm(step), step)
            model = space.add(center, step)
            if not cell.contains(model):
                continue
            inside += 1
            departure = (
                forward(model) - linearization.value - linearization.derivative(step)
            )
            assert claimed.contains(departure, rtol=1e-9)
            # One-sided: the exponential's second derivative is positive.
            assert np.all(departure >= -1e-14)
        assert inside >= 5
        assert np.all(claimed.lower == 0.0) and np.all(claimed.upper > 0.0)

    def test_the_functionals_as_vectors_give_the_same_set(self, rng):
        space, evaluation, integrate, forward, belief, remainder, directions, prior = (
            self.setting(rng)
        )
        nodes = evaluation.codomain
        rows = [evaluation.adjoint(nodes.basis_vector(q)) for q in range(nodes.dim)]
        listed = ComponentwiseRemainder(
            forward.codomain, rows, exponential_curvature, then=integrate
        )
        cell = Cell(
            space,
            belief.sample(rng=rng),
            directions=directions,
            half_widths=[0.02, 0.05, 0.01, 0.2],
            radius=prior.radius,
        )
        first, second = remainder.remainder_set(cell), listed.remainder_set(cell)
        assert np.allclose(first.upper, second.upper, rtol=1e-9)
        assert np.allclose(first.lower, second.lower, atol=1e-15)

    def test_narrower_slabs_give_a_smaller_set_down_to_a_floor(self, rng):
        """What the slabs do not span is bounded through the radius, and stays."""
        space, _, _, _, belief, remainder, directions, prior = self.setting(rng)
        center = belief.sample(rng=rng)
        sizes = []
        for width in (
            prior.radius,
            0.1 * prior.radius,
            1e-3 * prior.radius,
            1e-6 * prior.radius,
        ):
            cell = Cell(
                space,
                center,
                directions=directions,
                half_widths=[width] * 4,
                radius=prior.radius,
            )
            sizes.append(remainder.remainder_set(cell).radius)
        assert sizes[0] > sizes[1] > sizes[2] >= sizes[3] > 0.0
        assert sizes[2] == pytest.approx(sizes[3], rel=5e-2)

    def test_refusals(self, rng):
        space, evaluation, integrate, forward, _, _, _, _ = self.setting(rng)
        with pytest.raises(ValueError, match="functionals"):
            ComponentwiseRemainder(forward.codomain, evaluation, exponential_curvature)
        weighted = LinearOperator.from_matrix(
            space,
            make_dense_metric_space(3),
            np.ones((3, space.dim)),
            form="components",
        )
        with pytest.raises(TypeError, match="Euclidean"):
            ComponentwiseRemainder(forward.codomain, weighted, exponential_curvature)


class TestTighteningACell:
    def test_no_feasible_model_is_lost_and_the_cell_is_narrower(self):
        feasible = two_parameter_set(3, 1.0, 0.05)
        _, forward, weights, data = two_parameter_problem(3)
        remainder = attenuation_remainder(forward, weights)
        root = BranchAndPrune(remainder).root(feasible)
        tightened = feasible.tightened_cell(root, remainder)
        assert np.all(tightened.half_widths <= root.half_widths * (1.0 + 1e-12))
        assert tightened.half_widths.max() < 0.5 * root.half_widths.max()
        axis = np.linspace(-1.0, 1.0, 401)
        grid = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
        fits = np.linalg.norm(np.exp(-(grid @ weights.T)) - data, axis=1) <= 0.05
        fits &= np.linalg.norm(grid, axis=1) <= 1.0
        assert fits.sum() > 50
        assert all(tightened.contains(model, rtol=1e-6) for model in grid[fits])

    def test_a_cell_with_nothing_feasible_in_it_comes_back_as_none(self):
        feasible = two_parameter_set(3, 1.0, 0.05)
        _, forward, weights, _ = two_parameter_problem(3)
        cell = square_cell(feasible, np.array([-0.5, 0.5]), 0.2)
        assert (
            feasible.tightened_cell(cell, attenuation_remainder(forward, weights))
            is None
        )

    def test_for_a_linear_map_the_slab_is_the_feasible_set_s_extent(self, rng):
        """Two routes: the tightened slab, and the linear set's support along the direction."""
        model, data_space = make_dense_metric_space(5), make_dense_metric_space(3)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.normal(size=(3, 5)), form="components"
        )
        direction = model.random(rng=rng)
        along = LinearOperator.from_callables(
            model,
            EuclideanSpace(1),
            lambda m: np.array([model.inner_product(direction, m)]),
            adjoint=lambda y: model.scale(float(y[0]), direction),
        )
        truth = model.random(rng=rng)
        data = forward(truth)
        prior = Ball(model, radius=1.5 * model.norm(truth))
        noise = Ball(data_space, radius=0.1 * data_space.norm(data))
        linear = FeasiblePropertySet.from_structure(
            FeasibleStructure(forward, along, prior, noise=noise, solver=None), data
        )
        high, low = linear.support(np.array([1.0])), -linear.support(np.array([-1.0]))
        nonlinear = NonlinearFeasiblePropertySet(
            forward, along, prior, data, noise=noise
        )
        wide = nonlinear.cell(
            model.zero(),
            directions=[direction],
            half_widths=[prior.radius * model.norm(direction)],
        )
        tightened = nonlinear.tightened_cell(wide, NormRemainder(data_space, 0.0))
        middle = model.inner_product(direction, tightened.center)
        assert middle + tightened.half_widths[0] == pytest.approx(high, rel=1e-4)
        assert middle - tightened.half_widths[0] == pytest.approx(low, rel=1e-4)

    def test_the_search_needs_fewer_cells_with_it(self):
        feasible = two_parameter_set(3, 1.0, 0.05)
        _, forward, weights, _ = two_parameter_problem(3)
        truth = known_support(3, 1.0, 0.05)
        remainder = attenuation_remainder(forward, weights)
        direction = np.array([1.0])
        plain = feasible.support_result(
            direction, via=BranchAndPrune(remainder, rtol=1e-2)
        )
        tight = feasible.support_result(
            direction, via=BranchAndPrune(remainder, rtol=1e-2, tighten=2)
        )
        for result in (plain, tight):
            assert result.converged and not result.refuted
            assert result.inner <= truth + 1e-6 <= result.value + 2e-6
        assert len(tight.cells) < len(plain.cells)


class TestAFieldUnderLineIntegrals:
    """Travel times through ``exp`` of a field: a certified bracket from one tightened cell."""

    def test_one_tightened_cell_brackets_the_value_at_a_point(self, rng):
        space, evaluation, integrate, forward, belief = ray_problem()
        truth = belief.sample(rng=rng)
        clean = forward(truth)
        error = GaussianMeasure.from_standard_deviation(
            forward.codomain,
            0.1 * np.std(clean - integrate(np.ones(evaluation.codomain.dim))),
        )
        data = clean + error.sample(rng=rng)
        prior = belief.ambient_ball(level=0.95)
        property_operator = space.point_evaluation_operator([np.array([0.5, 0.5])])
        feasible = NonlinearBackusGilbertParker(
            ForwardProblem(forward, error=error), property_operator, prior
        )(data)
        remainder = ComponentwiseRemainder(
            forward.codomain, evaluation, exponential_curvature, then=integrate
        )
        directions = covering_directions(space, evaluation, integrate, 4)
        method = BranchAndPrune(remainder, directions=directions)
        root = method.root(feasible)
        assert len(root.directions) == 4
        cell = feasible.tightened_cell(root, remainder, sweeps=2)
        direction = np.array([1.0])
        whole = feasible.cell_support_result(direction, root, remainder)
        tight = feasible.cell_support_result(direction, cell, remainder)
        alone = prior.support_function()(property_operator.adjoint(direction))
        for result in (whole, tight):
            assert result.certified and not result.refuted
            assert feasible.is_feasible(result.extremal_model)
            assert result.inner <= result.value <= alone
        # The data pin the directions the rays sample, and the bound follows.
        assert cell.half_widths.min() < 0.2 * root.half_widths.min()
        assert (
            remainder.remainder_set(cell).radius
            < 0.5 * remainder.remainder_set(root).radius
        )
        assert tight.value < whole.value
        assert tight.value - tight.inner < 0.5 * (whole.value - whole.inner)


class TestASampledRemainderAcrossTheSlabs:
    """An estimate that also moves across the slabs, checked against a proved bound."""

    def setting(self, rng):
        space, evaluation, integrate, forward, belief = ray_problem()
        prior = belief.ambient_ball(level=0.95)
        directions = covering_directions(space, evaluation, integrate, 4)
        center = belief.sample(rng=rng)
        cell = Cell(
            space,
            center,
            directions=directions,
            half_widths=[0.01] * 4,
            radius=prior.radius,
        )
        proved = ComponentwiseRemainder(
            forward.codomain, evaluation, exponential_curvature, then=integrate
        )
        return space, forward, belief, prior, cell, proved

    def test_it_lies_under_the_proved_bound_and_sees_more_than_the_slabs(self, rng):
        space, forward, belief, prior, cell, proved = self.setting(rng)
        along = SampledRemainder(forward, samples=16, safety=1.0, rng=rng)
        across = SampledRemainder(
            forward,
            samples=16,
            safety=1.0,
            rng=rng,
            across=lambda: belief.sample(rng=rng),
        )
        assert not across.certified
        bound = proved.remainder_set(cell)
        seen_along, seen_across = along.remainder_set(cell), across.remainder_set(cell)
        # Two routes: what evaluations find, under what the theorem allows.
        assert np.all(seen_across.upper <= bound.upper * (1.0 + 1e-9))
        assert np.all(seen_across.lower >= bound.lower - 1e-15)
        # The cell is thin along its slabs and as wide as its radius across.
        assert seen_across.radius > 2.0 * seen_along.radius

    def test_as_a_ball_it_holds_every_remainder_it_saw(self, rng):
        space, forward, belief, prior, cell, _ = self.setting(rng)
        draws = [belief.sample(rng=rng) for _ in range(8)]
        supply = iter(draws)
        sampled = SampledRemainder(
            forward,
            samples=8,
            safety=1.0,
            rng=rng,
            across=lambda: next(supply),
            shape="ball",
        )
        estimate = sampled.remainder_set(cell)
        assert isinstance(estimate, Ball) and estimate.radius > 0.0
        box = SampledRemainder(forward, samples=8, safety=1.0, rng=rng).remainder_set(
            cell
        )
        # Along the slabs alone the remainders are smaller than with a step across.
        assert estimate.radius >= 0.5 * np.linalg.norm(box.upper)
        with pytest.raises(ValueError, match="box"):
            SampledRemainder(forward, shape="sphere")

    def test_it_is_a_fixed_function_of_the_cell(self, rng):
        space, forward, belief, prior, cell, _ = self.setting(rng)
        sampled = SampledRemainder(
            forward, samples=8, rng=rng, across=lambda: belief.sample(rng=rng)
        )
        first, second = sampled.remainder_set(cell), sampled.remainder_set(cell)
        assert np.array_equal(first.upper, second.upper)
        assert np.array_equal(first.lower, second.lower)

    def test_the_whole_prior_can_be_sampled_with_no_slabs(self, rng):
        space, forward, belief, prior, cell, proved = self.setting(rng)
        sampled = SampledRemainder(
            forward,
            samples=8,
            safety=1.0,
            rng=rng,
            across=lambda: belief.sample(rng=rng),
            within=prior,
        )
        whole = Cell(space, prior.center, radius=prior.radius)
        estimate = sampled.remainder_set(whole)
        assert estimate.radius > 0.0
        assert np.all(
            estimate.upper <= proved.remainder_set(whole).upper * (1.0 + 1e-9)
        )
        unbounded = Cell(space, prior.center)
        with pytest.raises(ValueError, match="finite radius"):
            sampled.remainder_set(unbounded)

    def test_models_the_operator_refuses_are_passed_over(self, rng):
        model, _, weights, _ = two_parameter_problem(3)
        inner = attenuation(model, list(weights))

        def guarded(m):
            if np.linalg.norm(m) > 0.5:
                raise EvaluationError("outside the operator's domain")
            return inner(m)

        forward = Operator.from_callables(
            model, inner.codomain, guarded, derivative=inner.derivative
        )
        cell = Cell(
            model,
            np.zeros(2),
            directions=[np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            half_widths=[0.45, 0.45],
        )
        # The corners are beyond the domain and the ends of the slabs within.
        estimate = SampledRemainder(forward, samples=8, rng=rng).remainder_set(cell)
        assert estimate.radius > 0.0


class TestRepairingAContradictedClaim:
    """A sampled claim a checked model contradicts takes the model in, and the bound is solved again."""

    DIRECTION = np.array([1.0])

    def remainder_at(self, forward, cell, model):
        linearization = forward.at(cell.center)
        space, data_space = forward.domain, forward.codomain
        step = linearization.derivative(space.subtract(model, cell.center))
        return data_space.subtract(
            data_space.subtract(forward(model), linearization.value), step
        )

    def thin_estimate(self, forward, prior, *, safety=1.0):
        """Sampled across no slabs and along one line only: what the operator does elsewhere is unseen."""
        return SampledRemainder(
            forward,
            samples=4,
            safety=safety,
            rng=np.random.default_rng(0),
            across=lambda: np.array([1.0, -1.0]),
            within=prior,
        )

    def test_an_estimate_takes_a_model_in_for_the_cells_that_hold_it(self, rng):
        model = make_dense_metric_space(4)
        paths = [model.scale(0.5, model.random(rng=rng)) for _ in range(3)]
        forward = attenuation(model, paths)
        center = model.scale(0.2, model.random(rng=rng))
        cell = Cell(
            model, center, directions=paths[:2], half_widths=[0.05, 0.05], radius=3.0
        )
        # Within the slabs and far across them, where no sample went.
        across = model.random(rng=rng)
        step = model.subtract(
            across,
            model.subtract(cell.point(cell.offsets(model.add(center, across))), center),
        )
        outlier = model.add(center, model.scale(2.5 / model.norm(step), step))
        assert cell.contains(outlier)
        sampled = SampledRemainder(forward, samples=8, safety=1.0, rng=rng)
        departure = self.remainder_at(forward, cell, outlier)
        before = sampled.remainder_set(cell)
        assert not before.contains(departure)
        learned = sampled.including(outlier)
        assert learned is not sampled and not learned.certified
        assert len(learned.kept) == 1 and sampled.kept == ()
        assert learned.remainder_set(cell).contains(departure)
        # The estimate it was made from is as it was.
        again = sampled.remainder_set(cell)
        assert np.array_equal(again.upper, before.upper)
        # A cell that does not hold the model is estimated as before.
        elsewhere = Cell(
            model, center, directions=paths[:2], half_widths=[0.05, 0.05], radius=1.0
        )
        assert not elsewhere.contains(outlier)
        assert np.array_equal(
            learned.remainder_set(elsewhere).upper,
            sampled.remainder_set(elsewhere).upper,
        )

    def test_a_proved_bound_takes_nothing_in(self):
        _, forward, weights, _ = two_parameter_problem(3)
        assert attenuation_remainder(forward, weights).including(np.zeros(2)) is None
        assert NormRemainder(forward.codomain, 1.0).including(np.zeros(2)) is None

    def test_a_cell_s_contradicted_estimate_is_repaired(self):
        """The control is the same solve with no repairs: an "outer" value below an attained one."""
        feasible = two_parameter_set(3, 0.5, 0.05)
        _, forward, weights, _ = two_parameter_problem(3)
        truth = known_support(3, 0.5, 0.05)
        cell = feasible.cell(np.zeros(2))
        thin = self.thin_estimate(forward, feasible.prior)
        left = feasible.cell_support_result(self.DIRECTION, cell, thin, max_repairs=0)
        assert left.refuted and left.repairs == 0 and left.remainder is thin
        assert left.value < left.inner - 5e-4
        result = feasible.cell_support_result(self.DIRECTION, cell, thin)
        assert not result.refuted and not result.certified
        assert result.repairs >= 1 and "enlarged" in result.message
        assert feasible.is_feasible(result.extremal_model)
        assert result.inner == pytest.approx(truth, abs=1e-5)
        assert result.value >= truth - 1e-6
        # The proved bound on the same cell lies above the repaired estimate.
        proved = feasible.cell_support_result(
            self.DIRECTION, cell, attenuation_remainder(forward, weights)
        )
        assert result.value <= proved.value
        # The claim it ends on holds the remainder of every model it took in.
        claim = result.remainder
        assert claim is not thin and thin.kept == ()
        claimed = claim.remainder_set(cell)
        assert len(claim.kept) >= result.repairs
        assert all(
            claimed.contains(self.remainder_at(forward, cell, model))
            for model in claim.kept
        )

    def test_repairs_that_run_out_leave_the_result_refuted(self):
        feasible = two_parameter_set(3, 0.5, 0.05)
        _, forward, _, _ = two_parameter_problem(3)
        cell = feasible.cell(np.zeros(2))
        thin = self.thin_estimate(forward, feasible.prior)
        whole = feasible.cell_support_result(self.DIRECTION, cell, thin)
        assert whole.repairs >= 2
        result = feasible.cell_support_result(self.DIRECTION, cell, thin, max_repairs=1)
        assert result.refuted and result.repairs == 1
        assert "outside the claimed set" in result.message

    def test_a_claim_that_cannot_be_enlarged_is_left_refuted(self):
        feasible = two_parameter_set(3, 1.0, 0.05)
        _, forward, weights, _ = two_parameter_problem(3)
        nothing = ComponentwiseRemainder(
            forward.codomain,
            list(weights),
            lambda low, high: (np.zeros(3), np.zeros(3)),
            certified=False,
        )
        cell = square_cell(feasible, np.zeros(2), 1.0)
        on_a_cell = feasible.cell_support_result(self.DIRECTION, cell, nothing)
        searched = feasible.support_result(self.DIRECTION, via=BranchAndPrune(nothing))
        for result in (on_a_cell, searched):
            assert result.refuted and result.repairs == 0
            assert result.remainder is nothing

    def test_the_search_over_cells_repairs_and_searches_again(self):
        """Slabs along one path only, sampled along them: the other paths' curvature is unseen."""
        feasible = two_parameter_set(3, 0.5, 0.05)
        _, forward, weights, _ = two_parameter_problem(3)
        truth = known_support(3, 0.5, 0.05)
        thin = SampledRemainder(
            forward, samples=4, safety=1.0, rng=np.random.default_rng(0)
        )

        def search(max_repairs):
            method = BranchAndPrune(
                thin, directions=[weights[0]], rtol=1e-2, max_repairs=max_repairs
            )
            return feasible.support_result(self.DIRECTION, via=method)

        left = search(0)
        assert left.refuted and left.repairs == 0 and "contradicts" in left.message
        result = search(4)
        assert not result.refuted and not result.certified and result.converged
        assert result.repairs >= 1 and "enlarged" in result.message
        assert feasible.is_feasible(result.extremal_model)
        assert result.inner == pytest.approx(truth, abs=1e-5)
        assert result.value >= truth - 1e-6
        assert len(result.remainder.kept) >= result.repairs and thin.kept == ()
        # The work of every search is counted, not the last one's alone.
        assert result.evaluations > left.evaluations
        assert result.searches > left.searches
        # A proved bound is searched once, and is the other route to the support.
        proved = feasible.support_result(
            self.DIRECTION,
            via=BranchAndPrune(attenuation_remainder(forward, weights), rtol=1e-2),
        )
        assert proved.repairs == 0 and proved.certified
        assert proved.inner == pytest.approx(result.inner, abs=1e-5)


class TestWhereTheOperatorCannotBeEvaluated:
    def guarded_set(self):
        model, _, weights, data = two_parameter_problem(3)
        inner = attenuation(model, list(weights))

        def guarded(m):
            if m[0] < -0.2:
                raise EvaluationError("outside the operator's domain")
            return inner(m)

        forward = Operator.from_callables(
            model, inner.codomain, guarded, derivative=inner.derivative
        )
        property_operator = LinearOperator.from_matrix(
            model, EuclideanSpace(1), COST[None, :], form="components"
        )
        return NonlinearFeasiblePropertySet(
            forward,
            property_operator,
            Ball(model, radius=1.0),
            data,
            noise=Ball(inner.codomain, radius=0.05),
        )

    def test_a_model_that_cannot_be_evaluated_fits_nothing(self):
        feasible = self.guarded_set()
        assert not feasible.fits(np.array([-0.5, 0.0]))
        assert not feasible.is_feasible(np.array([-0.5, 0.0]))
        assert feasible.is_feasible(TRUE_MODEL)

    def test_a_search_that_cannot_begin_says_so(self):
        feasible = self.guarded_set()
        result = feasible.support_result(np.array([1.0]), start=np.array([-0.5, 0.0]))
        assert result.extremal_model is None and result.inner == float("-inf")
        assert not result.converged
        assert "could not be evaluated" in result.message

    def test_a_search_kept_in_a_cell_falls_back_on_its_center(self):
        feasible = self.guarded_set()
        cell = square_cell(feasible, TRUE_MODEL, 0.3)
        result = LocalSearch().support(
            feasible, np.array([1.0]), np.array([-0.5, 0.0]), cell=cell
        )
        assert feasible.is_feasible(result.extremal_model)
        assert cell.contains(result.extremal_model)


class TestTheCellsLinearizedOperatorAsAMatrix:
    def test_the_matrix_and_the_operator_give_the_same_bound(self, rng):
        """Two routes: the derivative assembled row by row, and applied as it is."""
        model, forward, feasible, radius = dense_metric_setting(rng)
        paths_free = NonlinearFeasiblePropertySet(
            forward,
            feasible.property_operator,
            feasible.prior,
            feasible.data,
            noise=feasible.noise,
            dense_limit=0,
        )
        directions = [model.random(rng=rng) for _ in range(2)]
        direction = np.array([1.0])
        values = []
        for candidate in (feasible, paths_free):
            cell = candidate.cell(
                candidate.fitting_model(), directions=directions, half_widths=[0.2, 0.1]
            )
            remainder = NormRemainder(forward.codomain, 0.5)
            values.append(candidate.outer_set(cell, remainder).support(direction))
        assert values[0] == pytest.approx(values[1], rel=1e-6)

    def test_a_cell_s_slabs_default_to_the_prior_s_reach(self):
        feasible = two_parameter_set(3, 1.0, 0.05)
        cell = feasible.cell(
            np.array([0.3, 0.4]),
            directions=[np.array([1.0, 0.0]), np.array([0.0, 2.0])],
        )
        # The unit ball reaches 1 + 0.3 one way along the first, and
        # 2 (1 + 0.4) along the second, from this center.
        assert np.allclose(cell.half_widths, [1.3, 2.8])


# --------------------------------------------------------------------- #
#                            The estimator                               #
# --------------------------------------------------------------------- #


class TestTheEstimator:
    def pieces(self):
        model, forward, _, data = two_parameter_problem(3)
        property_operator = LinearOperator.from_matrix(
            model, EuclideanSpace(1), COST[None, :], form="components"
        )
        return model, forward, data, property_operator

    def test_it_returns_the_set_the_geometry_layer_builds(self):
        model, forward, data, property_operator = self.pieces()
        noise = Ball(forward.codomain, radius=0.05)
        estimator = NonlinearBackusGilbertParker(
            ForwardProblem(forward, error=noise),
            property_operator,
            Ball(model, radius=1.0),
        )
        feasible = estimator(data)
        assert isinstance(feasible, NonlinearFeasiblePropertySet)
        assert estimator.noise is noise and feasible.noise is noise
        direct = two_parameter_set(3, 1.0, 0.05).support_result(np.array([1.0]))
        assert feasible.support_result(np.array([1.0])).inner == pytest.approx(
            direct.inner, abs=1e-12
        )

    def test_an_error_measure_is_hardened_to_its_ambient_ball(self):
        model, forward, _, property_operator = self.pieces()
        error = GaussianMeasure.from_standard_deviation(forward.codomain, 0.02)
        estimator = NonlinearBackusGilbertParker(
            ForwardProblem(forward, error=error),
            property_operator,
            Ball(model, radius=1.0),
            level=0.9,
        )
        assert isinstance(estimator.noise, Ball)
        assert estimator.noise.radius == pytest.approx(
            error.ambient_ball(level=0.9).radius
        )

    def test_a_problem_without_error_is_refused_by_name(self):
        model, forward, _, property_operator = self.pieces()
        with pytest.raises(ValueError, match="noise="):
            NonlinearBackusGilbertParker(
                ForwardProblem(forward), property_operator, Ball(model, radius=1.0)
            )

    def test_a_set_on_the_wrong_space_is_refused_at_construction(self):
        model, forward, _, property_operator = self.pieces()
        with pytest.raises(ValueError, match="wrong space"):
            NonlinearBackusGilbertParker(
                ForwardProblem(forward, error=Ball(forward.codomain, radius=0.05)),
                property_operator,
                Ball(forward.codomain, radius=1.0),
            )

    def test_a_further_property_is_the_composed_one(self):
        model, forward, data, property_operator = self.pieces()
        estimator = NonlinearBackusGilbertParker(
            ForwardProblem(forward, error=Ball(forward.codomain, radius=0.05)),
            property_operator,
            Ball(model, radius=1.0),
        )
        doubled = estimator.push_forward(
            LinearOperator.identity(estimator.property_space) * 2.0
        )
        direction = np.array([1.0])
        assert doubled(data).support_result(direction).inner == pytest.approx(
            2.0 * estimator(data).support_result(direction).inner, rel=1e-6
        )
