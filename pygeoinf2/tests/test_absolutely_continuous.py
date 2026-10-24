"""Measures that know their density relative to a reference (D-132)."""

import numpy as np
import pytest

import pygeoinf2 as gi
from pygeoinf2.algebra.operators import Functional, LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.inference.gaussian import GaussianPosterior, LinearGaussianInversion
from pygeoinf2.inference.laplace import MaximumAPosteriori
from pygeoinf2.inference.problem import ForwardProblem, LinearForwardProblem
from pygeoinf2.probability import AbsolutelyContinuousMeasure
from pygeoinf2.testing import check_gradient

from .conftest import make_dense_metric_space


@pytest.fixture
def euclidean_setting(rng):
    """A linear Gaussian problem small enough to verify against numpy."""
    model = EuclideanSpace(4)
    data_space = EuclideanSpace(6)
    matrix = rng.standard_normal((6, 4))
    forward = LinearOperator.from_matrix(model, data_space, matrix, form="components")
    prior = gi.GaussianMeasure.from_standard_deviation(model, 2.0).translate(
        model.from_components(rng.standard_normal(4))
    )
    error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.5)
    problem = LinearForwardProblem(forward, error=error)
    data = data_space.add(forward(prior.sample(rng=rng)), error.sample(rng=rng))
    return model, data_space, matrix, forward, prior, error, problem, data


class TestTheInterface:
    def test_a_density_on_the_wrong_space_is_refused(self, rng):
        space = EuclideanSpace(3)
        other = EuclideanSpace(2)
        reference = gi.GaussianMeasure.from_standard_deviation(space, 1.0)
        density = Functional.from_callables(other, lambda x: 0.0)
        with pytest.raises(ValueError, match="domain"):
            AbsolutelyContinuousMeasure(reference, density)

    def test_it_cannot_be_sampled_and_says_why(self, rng):
        space = EuclideanSpace(3)
        reference = gi.GaussianMeasure.from_standard_deviation(space, 1.0)
        density = Functional.from_callables(space, lambda x: 0.0)
        measure = AbsolutelyContinuousMeasure(reference, density)
        assert measure.reference is reference
        assert measure.log_normalizer is None
        with pytest.raises(NotImplementedError, match="sampler"):
            measure.sample(rng=rng)

    def test_relative_to_needs_the_shared_reference(self, rng):
        space = EuclideanSpace(3)
        reference = gi.GaussianMeasure.from_standard_deviation(space, 1.0)
        other_reference = gi.GaussianMeasure.from_standard_deviation(space, 2.0)
        f = Functional.from_callables(space, lambda x: 1.0)
        g = Functional.from_callables(space, lambda x: 2.0)
        one = AbsolutelyContinuousMeasure(reference, f)
        two = AbsolutelyContinuousMeasure(reference, g)
        x = space.random(rng=rng)
        assert one.relative_to(two)(x) == pytest.approx(-1.0)
        stranger = AbsolutelyContinuousMeasure(other_reference, g)
        with pytest.raises(ValueError, match="same reference"):
            one.relative_to(stranger)


class TestTheLinearPosterior:
    def test_it_is_both_things_and_carries_its_prior(self, euclidean_setting):
        *_, prior, _, problem, data = euclidean_setting
        posterior = LinearGaussianInversion(problem, prior)(data)
        assert isinstance(posterior, gi.GaussianMeasure)
        assert isinstance(posterior, AbsolutelyContinuousMeasure)
        assert posterior.reference is prior

    def test_the_relative_density_is_minus_the_potential(self, euclidean_setting, rng):
        model, *_, prior, _, problem, data = euclidean_setting
        posterior = LinearGaussianInversion(problem, prior)(data)
        for _ in range(3):
            x = prior.sample(rng=rng)
            assert posterior.relative_log_density(x) == pytest.approx(
                -0.5 * problem.chi_squared(x, data)
            )

    def test_the_normalizer_against_numpy(self, euclidean_setting):
        """``c == (log|R| - log|S| - <v, S^-1 v>) / 2`` with ``S = A Q A* + R``,
        verified against a dense computation that shares no code with it."""
        model, data_space, matrix, forward, prior, error, problem, data = (
            euclidean_setting
        )
        posterior = LinearGaussianInversion(problem, prior)(data)
        normalizer = posterior.log_normalizer

        S = 4.0 * matrix @ matrix.T + 0.25 * np.identity(6)
        residual = data_space.to_components(data) - matrix @ model.to_components(
            prior.expectation
        )
        expected = 0.5 * (
            6 * np.log(0.25)
            - np.linalg.slogdet(S)[1]
            - float(residual @ np.linalg.solve(S, residual))
        )
        assert normalizer.value == pytest.approx(expected, rel=1e-8)

    def test_the_density_integrates_to_one_on_a_dense_metric(self, rng):
        """``E_prior[exp(f - c)] == 1``: the Monte Carlo statement of what a
        normalizer is, on a space whose Gram matrix is not diagonal."""
        model = make_dense_metric_space(3)
        data_space = EuclideanSpace(4)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.standard_normal((4, 3)), form="components"
        )
        prior = gi.GaussianMeasure.from_standard_deviation(model, 1.0)
        # Weak data, so the weights have low variance and 4000 draws suffice.
        error = gi.GaussianMeasure.from_standard_deviation(data_space, 4.0)
        problem = LinearForwardProblem(forward, error=error)
        data = data_space.add(forward(prior.sample(rng=rng)), error.sample(rng=rng))
        posterior = LinearGaussianInversion(problem, prior)(data)

        c = posterior.log_normalizer.value
        f = posterior.relative_log_density
        weights = [np.exp(f(prior.sample(rng=rng)) - c) for _ in range(4000)]
        assert np.mean(weights) == pytest.approx(1.0, rel=0.05)

    def test_the_density_gradient_is_metric_correct(self, rng):
        model = make_dense_metric_space(4)
        data_space = EuclideanSpace(5)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.standard_normal((5, 4)), form="components"
        )
        prior = gi.GaussianMeasure.from_standard_deviation(model, 1.0)
        error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.5)
        problem = LinearForwardProblem(forward, error=error)
        data = data_space.random(rng=rng)
        posterior = LinearGaussianInversion(problem, prior)(data)
        check_gradient(posterior.relative_log_density, prior.sample(rng=rng), rng=rng)

    def test_the_algebra_degrades_to_a_plain_gaussian(self, euclidean_setting, rng):
        """D-73's worry, answered: a push-forward or a translation rebuilds
        through ``GaussianMeasure._rebuild``, so the density is dropped rather
        than silently wrong."""
        model, *_, prior, _, problem, data = euclidean_setting
        posterior = LinearGaussianInversion(problem, prior)(data)
        moved = posterior.translate(model.random(rng=rng))
        assert isinstance(moved, gi.GaussianMeasure)
        assert not isinstance(moved, AbsolutelyContinuousMeasure)

    def test_two_posteriors_compare_through_the_prior(self, euclidean_setting, rng):
        model, data_space, *_, prior, _, problem, _ = euclidean_setting
        d1, d2 = data_space.random(rng=rng), data_space.random(rng=rng)
        estimator = LinearGaussianInversion(problem, prior)
        one, two = estimator(d1), estimator(d2)
        ratio = one.relative_to(two)
        x = prior.sample(rng=rng)
        assert ratio(x) == pytest.approx(
            -0.5 * (problem.chi_squared(x, d1) - problem.chi_squared(x, d2))
        )


class TestThePotential:
    def test_value_and_temperature(self, euclidean_setting, rng):
        model, *_, prior, _, problem, data = euclidean_setting
        potential = problem.potential(data)
        tempered = problem.potential(data, temperature=2.0)
        x = prior.sample(rng=rng)
        assert potential(x) == pytest.approx(0.5 * problem.chi_squared(x, data))
        assert tempered(x) == pytest.approx(0.5 * potential(x))
        with pytest.raises(ValueError, match="temperature"):
            problem.potential(data, temperature=0.0)

    def test_the_gradient_on_a_dense_metric(self, rng):
        model = make_dense_metric_space(4)
        data_space = EuclideanSpace(5)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.standard_normal((5, 4)), form="components"
        )
        error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.5)
        problem = LinearForwardProblem(forward, error=error)
        potential = problem.potential(data_space.random(rng=rng))
        check_gradient(potential, model.random(rng=rng), rng=rng)

    def test_one_forward_evaluation_serves_both(self, rng):
        model = EuclideanSpace(3)
        data_space = EuclideanSpace(5)
        matrix = rng.standard_normal((5, 3))
        counts = {"n": 0}

        def value(c):
            counts["n"] += 1
            return matrix @ c

        from pygeoinf2.algebra.operators import Operator

        forward = Operator.from_callables(
            model,
            data_space,
            value,
            derivative=lambda c: LinearOperator.from_matrix(
                model, data_space, matrix, form="components"
            ),
        )
        error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.5)
        problem = ForwardProblem(forward, error=error)
        potential = problem.potential(data_space.random(rng=rng))
        counts["n"] = 0
        quadratic = potential.at(model.random(rng=rng))
        quadratic.gradient  # noqa: B018
        assert counts["n"] == 1


class TestTheLaplacePosterior:
    def test_the_gaussian_is_approximate_and_the_density_exact(self, rng):
        model = EuclideanSpace(3)
        data_space = EuclideanSpace(8)
        matrix = rng.standard_normal((8, 3))

        from pygeoinf2.algebra.operators import Operator

        forward = Operator.from_callables(
            model,
            data_space,
            lambda c: np.exp(matrix @ c),
            derivative=lambda c: LinearOperator.from_matrix(
                model,
                data_space,
                np.diag(np.exp(matrix @ c)) @ matrix,
                form="components",
            ),
        )
        prior = gi.GaussianMeasure.from_standard_deviation(model, 0.4)
        error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.1)
        problem = ForwardProblem(forward, error=error)
        data = data_space.add(forward(prior.sample(rng=rng)), error.sample(rng=rng))
        result = MaximumAPosteriori(problem, prior)(data)

        measure = result.measure
        assert isinstance(measure, GaussianPosterior)
        assert measure.reference is prior
        assert measure.log_normalizer is None
        # Scaled down: the exponential forward overflows on a wild draw, and
        # a comparison of two infinities proves nothing.
        x = model.scale(0.3, prior.sample(rng=rng))
        assert measure.relative_log_density(x) == pytest.approx(
            -0.5 * problem.chi_squared(x, data)
        )
