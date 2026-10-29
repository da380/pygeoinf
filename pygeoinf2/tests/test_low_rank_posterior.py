"""The low-rank posterior: the prior corrected in the directions the data
inform, against the full posterior it approximates."""

import numpy as np
import pytest

from pygeoinf2.algebra.operators import Operator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.inference import (
    LaplaceApproximation,
    LinearForwardProblem,
    LinearGaussianInversion,
    LowRankGaussianPosterior,
)
from pygeoinf2.probability import GaussianMeasure
from pygeoinf2.symmetric_space.circle import Sobolev
from pygeoinf2.testing import check_measure


def linear_setting(rng, n_data=6):
    space = Sobolev(32, 2.0, 0.2)
    prior = space.sobolev_measure(2.0, 0.2)
    points = [(float(t),) for t in np.linspace(0.3, 5.9, n_data)]
    forward = space.point_evaluation_operator(points)
    noise = GaussianMeasure.from_standard_deviation(forward.codomain, 0.05)
    problem = LinearForwardProblem(forward, error=noise)
    truth = prior.sample(rng=rng)
    data = forward.codomain.add(forward(truth), noise.sample(rng=rng))
    return space, prior, problem, data


class TestAgainstTheFullPosterior:
    def test_at_full_rank_it_is_the_posterior(self, rng):
        space, prior, problem, data = linear_setting(rng)
        full = LinearGaussianInversion(problem, prior)(data)
        low = LowRankGaussianPosterior.from_linear_problem(
            problem, prior, data, rank=6, rng=rng
        )
        assert low.rank == 6
        assert space.norm(space.subtract(low.expectation, full.expectation)) < 1e-8 * (
            space.norm(full.expectation)
        )
        x = space.random(rng=rng)
        assert space.norm(
            space.subtract(low.covariance(x), full.covariance(x))
        ) < 1e-8 * space.norm(full.covariance(x))
        assert low.log_normalizer.value == pytest.approx(
            full.log_normalizer.value, rel=1e-8
        )

    def test_below_full_rank_the_mean_omits_the_dropped_update(self, rng):
        """At rank 2 of 6 the dropped eigenvalues are in the hundreds, and an
        update left unshrunk there would be hundreds of times the posterior
        mean; kept in the retained directions only, it is the posterior's
        update projected in the prior's Cameron-Martin norm."""
        space, prior, problem, data = linear_setting(rng)
        full = LinearGaussianInversion(problem, prior)(data)
        low = LowRankGaussianPosterior.from_linear_problem(
            problem, prior, data, rank=2, rng=rng
        )
        assert low.eigenvalues.min() > 100.0
        assert space.norm(low.expectation) < 2.0 * space.norm(full.expectation)

        def prior_norm(model):
            return space.inner_product(prior.precision(model), model)

        assert prior_norm(low.expectation) <= prior_norm(full.expectation) * (
            1.0 + 1e-8
        )

    def test_a_noise_expectation_is_removed(self, rng):
        space, prior, problem, data = linear_setting(rng)
        noise = problem.error_measure
        offset = noise.domain.from_components(np.linspace(-0.2, 0.3, noise.domain.dim))
        biased = LinearForwardProblem(
            problem.forward_operator,
            error=GaussianMeasure.from_standard_deviation(
                noise.domain, 0.05, expectation=offset
            ),
        )
        shifted = noise.domain.add(data, offset)
        full = LinearGaussianInversion(biased, prior)(shifted)
        low = LowRankGaussianPosterior.from_linear_problem(
            biased, prior, shifted, rank=6, rng=rng
        )
        assert space.norm(space.subtract(low.expectation, full.expectation)) < 1e-8 * (
            space.norm(full.expectation)
        )
        assert low.log_normalizer.value == pytest.approx(
            full.log_normalizer.value, rel=1e-8
        )

    def test_the_modes_do_not_depend_on_the_job_count(self, rng):
        space, prior, problem, data = linear_setting(rng)
        low = LowRankGaussianPosterior.from_linear_problem(
            problem, prior, data, rank=3, rng=rng
        )
        for one, other in zip(low.modes(), low.modes(n_jobs=2)):
            assert space.norm(space.subtract(one, other)) < 1e-12 * space.norm(one)

    def test_below_full_rank_it_sits_between_prior_and_posterior(self, rng):
        space, prior, problem, data = linear_setting(rng)
        full = LinearGaussianInversion(problem, prior)(data)
        low = LowRankGaussianPosterior.from_linear_problem(
            problem, prior, data, rank=3, rng=rng
        )
        assert low.rank == 3
        for _ in range(5):
            x = space.random(rng=rng)
            quadratic = lambda m: space.inner_product(m.covariance(x), x)  # noqa: E731
            assert quadratic(full) <= quadratic(low) * (1 + 1e-10)
            assert quadratic(low) <= quadratic(prior) * (1 + 1e-10)
        assert np.all(np.diff(low.eigenvalues) <= 1e-12)

    def test_it_is_a_measure_with_both_factors(self, rng):
        space, prior, problem, data = linear_setting(rng)
        low = LowRankGaussianPosterior.from_linear_problem(
            problem, prior, data, rank=4, rng=rng
        )
        check_measure(low, rng=rng)
        x = space.random(rng=rng)
        back = low.precision(low.covariance(x))
        assert space.norm(space.subtract(back, x)) < 1e-8 * space.norm(x)
        draws = low.samples(3, rng=rng)
        assert len(draws) == 3

    def test_the_pointwise_variance_reduction_is_the_modes_squared(self, rng):
        space, prior, problem, data = linear_setting(rng)
        low = LowRankGaussianPosterior.from_linear_problem(
            problem, prior, data, rank=4, rng=rng
        )
        reduction = low.pointwise_variance_reduction()
        shrink = low.eigenvalues / (1.0 + low.eigenvalues)
        expected = space.zero()
        for weight, mode in zip(shrink, low.modes()):
            expected = space.axpy(float(weight), space.multiply(mode, mode), expected)
        assert space.norm(space.subtract(reduction, expected)) < 1e-12 * space.norm(
            expected
        )
        # Sampled pointwise variances agree with the closed form to Monte
        # Carlo accuracy.
        sampled_prior = prior.sample_pointwise_variance(400, rng=rng)
        sampled_low = low.sample_pointwise_variance(400, rng=rng)
        difference = space.subtract(sampled_prior, sampled_low)
        assert space.norm(space.subtract(difference, reduction)) < 0.35 * space.norm(
            reduction
        )


class TestTheLaplaceRoute:
    def test_at_full_rank_it_is_the_laplace_gaussian(self, rng):
        X, Y = EuclideanSpace(3), EuclideanSpace(4)
        matrix = rng.normal(size=(4, 3))
        forward = Operator.from_callables(
            X,
            Y,
            lambda c: matrix @ np.tanh(c),
            derivative=lambda c: __import__("pygeoinf2").LinearOperator.from_matrix(
                X, Y, matrix * (1.0 - np.tanh(c) ** 2)[None, :], form="components"
            ),
        )
        from pygeoinf2.inference import ForwardProblem

        noise = GaussianMeasure.from_standard_deviation(Y, 0.1)
        problem = ForwardProblem(forward, error=noise)
        prior = GaussianMeasure.from_standard_deviation(X, 1.0)
        data = Y.add(forward(prior.sample(rng=rng)), noise.sample(rng=rng))
        full = LaplaceApproximation.from_problem(problem, prior)
        low = LaplaceApproximation.from_problem(problem, prior, rank=3, rng=rng)
        mode = full.solve(data).model
        exact = full.at(mode, data)
        approx = low.at(mode, data)
        assert isinstance(approx, LowRankGaussianPosterior)
        x = X.random(rng=rng)
        assert approx.covariance(x) == pytest.approx(exact.covariance(x), rel=1e-6)
        assert approx.precision(x) == pytest.approx(exact.precision(x), rel=1e-6)
        assert approx.log_normalizer is not None


class TestWhatIsRefused:
    def test_a_prior_without_a_factor_is_refused(self, rng):
        from pygeoinf2.algebra.operators import LinearOperator
        from pygeoinf2.traits import Traits

        X = EuclideanSpace(3)
        covariance = LinearOperator.self_adjoint(
            X, lambda v: 2.0 * v, traits=Traits.POSITIVE_DEFINITE
        )
        prior = GaussianMeasure(X, covariance=covariance)
        forward = LinearOperator.identity(X)
        problem = LinearForwardProblem(
            forward, error=GaussianMeasure.from_standard_deviation(X, 1.0)
        )
        with pytest.raises(ValueError, match="covariance factor"):
            LowRankGaussianPosterior.from_linear_problem(
                problem, prior, X.random(rng=rng), rank=2
            )
