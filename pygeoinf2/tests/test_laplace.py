"""Nonlinear inference: the MAP model and the Gaussian about it (D-7)."""

import numpy as np
import pytest

import pygeoinf2 as gi
from pygeoinf2.algebra.operators import LinearOperator, Operator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.inference.gaussian import LinearGaussianInversion
from pygeoinf2.inference.laplace import MaximumAPosteriori
from pygeoinf2.inference.problem import ForwardProblem, LinearForwardProblem


@pytest.fixture
def linear_setting(rng):
    """A linear problem, where the Laplace answer must be the exact one."""
    model = EuclideanSpace(6)
    data_space = EuclideanSpace(10)
    forward = LinearOperator.from_matrix(
        model, data_space, rng.standard_normal((10, 6)), form="components"
    )
    prior = gi.GaussianMeasure.from_standard_deviation(model, 2.0)
    error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.3)
    truth = prior.sample(rng=rng)
    data = data_space.add(forward(truth), error.sample(rng=rng))
    return forward, prior, error, data


@pytest.fixture
def nonlinear_setting(rng):
    """``exp(B m)``: genuinely nonlinear, with an exact derivative."""
    model = EuclideanSpace(4)
    data_space = EuclideanSpace(12)
    matrix = rng.standard_normal((12, 4))

    def value(components):
        return np.exp(matrix @ components)

    def derivative(components):
        return LinearOperator.from_matrix(
            model,
            data_space,
            np.diag(np.exp(matrix @ components)) @ matrix,
            form="components",
        )

    forward = Operator.from_callables(model, data_space, value, derivative=derivative)
    prior = gi.GaussianMeasure.from_standard_deviation(model, 0.4)
    error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.05)
    truth = prior.sample(rng=rng)
    data = data_space.add(forward(truth), error.sample(rng=rng))
    return forward, prior, error, truth, data


class TestItIsExactWhenTheProblemIsLinear:
    """The Laplace approximation is exact for a linear operator, so this is the
    check that the construction is the right one and not merely plausible."""

    def test_the_mode_is_the_posterior_mean(self, linear_setting):
        forward, prior, error, data = linear_setting
        exact = LinearGaussianInversion(
            LinearForwardProblem(forward, error=error), prior
        )(data)
        found = MaximumAPosteriori(ForwardProblem(forward, error=error), prior)(data)

        assert found.converged
        assert found.model == pytest.approx(exact.expectation, abs=1e-6)

    def test_the_covariance_is_the_posterior_covariance(self, linear_setting, rng):
        forward, prior, error, data = linear_setting
        space = forward.domain
        exact = LinearGaussianInversion(
            LinearForwardProblem(forward, error=error), prior
        )(data)
        found = MaximumAPosteriori(ForwardProblem(forward, error=error), prior)(data)

        for _ in range(3):
            vector = space.random(rng=rng)
            assert found.measure.covariance(vector) == pytest.approx(
                exact.covariance(vector), abs=1e-10
            )


class TestOnANonlinearProblem:
    def test_it_finds_a_stationary_point(self, nonlinear_setting):
        """Which is what a mode is. Checked on the objective's own gradient,
        not on the optimiser's report of itself."""
        forward, prior, error, truth, data = nonlinear_setting
        estimator = MaximumAPosteriori(ForwardProblem(forward, error=error), prior)
        found = estimator(data)

        assert found.converged
        gradient = estimator.objective(data).gradient(found.model)
        assert forward.domain.norm(gradient) < 1e-4

    def test_it_recovers_the_truth(self, nonlinear_setting):
        forward, prior, error, truth, data = nonlinear_setting
        found = MaximumAPosteriori(ForwardProblem(forward, error=error), prior)(data)
        space = forward.domain
        assert space.norm(space.subtract(found.model, truth)) < 0.2 * space.norm(truth)

    def test_the_covariance_is_positive_definite(self, nonlinear_setting):
        """A curvature that is not would mean the point found is not a
        minimum, and the Gaussian on it would be meaningless."""
        forward, prior, error, truth, data = nonlinear_setting
        found = MaximumAPosteriori(ForwardProblem(forward, error=error), prior)(data)

        matrix = found.measure.covariance.matrix(form="components")
        assert np.all(np.linalg.eigvalsh(0.5 * (matrix + matrix.T)) > 0.0)

    def test_the_objective_is_twice_the_negative_log_posterior(
        self, nonlinear_setting, rng
    ):
        """The factor that goes missing if it is written at the call site."""
        forward, prior, error, truth, data = nonlinear_setting
        estimator = MaximumAPosteriori(ForwardProblem(forward, error=error), prior)
        objective = estimator.objective(data)
        density = estimator.log_posterior(data)

        for _ in range(3):
            model = forward.domain.random(rng=rng)
            assert density(model) == pytest.approx(-0.5 * objective(model))

    def test_the_gradient_is_the_gradient(self, nonlinear_setting, rng):
        """Against a central difference: the derivative is where the
        nonlinearity enters, and the only place it does."""
        forward, prior, error, truth, data = nonlinear_setting
        space = forward.domain
        objective = MaximumAPosteriori(
            ForwardProblem(forward, error=error), prior
        ).objective(data)

        model = space.scale(0.3, space.random(rng=rng))
        gradient = objective.gradient(model)
        step = 1e-6
        for index in range(space.dim):
            basis = space.basis_vector(index)
            forward_value = objective(space.axpy(step, basis, space.copy(model)))
            backward = objective(space.axpy(-step, basis, space.copy(model)))
            assert (forward_value - backward) / (2.0 * step) == pytest.approx(
                space.inner_product(gradient, basis), rel=1e-4, abs=1e-6
            )

    def test_the_gaussian_can_be_placed_anywhere(self, nonlinear_setting, rng):
        """``at`` exists for when the mode is already known -- from a previous
        run, or to see what the approximation looks like elsewhere."""
        forward, prior, error, truth, data = nonlinear_setting
        estimator = MaximumAPosteriori(ForwardProblem(forward, error=error), prior)
        found = estimator(data)

        elsewhere = estimator.at(prior.expectation, data)
        assert elsewhere.expectation == pytest.approx(prior.expectation)
        # A different point, so a different curvature.
        vector = forward.domain.random(rng=rng)
        assert not np.allclose(
            elsewhere.covariance(vector), found.measure.covariance(vector)
        )


class TestWhatItRefuses:
    def test_a_prior_on_the_wrong_space(self, linear_setting):
        forward, _, error, _ = linear_setting
        wrong = gi.GaussianMeasure.from_standard_deviation(EuclideanSpace(3), 1.0)
        with pytest.raises(ValueError, match="model space"):
            MaximumAPosteriori(ForwardProblem(forward, error=error), wrong)

    def test_no_error_measure(self, linear_setting):
        """Without one there is nothing to weigh the prior against, and the
        mode is wherever the data are matched exactly."""
        forward, prior, _, _ = linear_setting
        with pytest.raises(ValueError, match="error measure"):
            MaximumAPosteriori(ForwardProblem(forward), prior)

    def test_a_set_valued_error(self, linear_setting):
        from pygeoinf2.geometry.convex import Ball

        forward, prior, _, _ = linear_setting
        problem = ForwardProblem(forward, error=Ball(forward.codomain, radius=1.0))
        with pytest.raises(TypeError, match="backus"):
            MaximumAPosteriori(problem, prior)


class TestTheSamplerHooksAreThere:
    """D-7 asks for this to be built so a function-space MCMC layer plugs in.
    That needs three things and no more: prior draws, the posterior's log
    density, and the forward problem."""

    def test_all_three(self, nonlinear_setting, rng):
        forward, prior, error, truth, data = nonlinear_setting
        estimator = MaximumAPosteriori(ForwardProblem(forward, error=error), prior)

        assert prior.can_sample
        assert prior.sample(rng=rng) is not None

        density = estimator.log_posterior(data)
        model = forward.domain.random(rng=rng)
        assert np.isfinite(density(model))
        assert forward.domain.norm(density.gradient(model)) > 0.0

    def test_the_density_peaks_at_the_mode(self, nonlinear_setting, rng):
        forward, prior, error, truth, data = nonlinear_setting
        estimator = MaximumAPosteriori(ForwardProblem(forward, error=error), prior)
        found = estimator(data)
        density = estimator.log_posterior(data)

        at_mode = density(found.model)
        for _ in range(5):
            nearby = forward.domain.axpy(
                0.05, forward.domain.random(rng=rng), forward.domain.copy(found.model)
            )
            assert density(nearby) <= at_mode
