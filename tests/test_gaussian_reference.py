"""The Gaussian reference: the prior with a finite chart's law replaced.

Its moments and its density must describe one measure, and the references
the estimators hand out must be the Gaussians they stand for. Everything
runs on a dense metric, where a component standing in for a vector, or a
representer for a derivative, would show.
"""

import numpy as np
import pytest

import pygeoinf as gi
from pygeoinf.algebra.operators import Functional, LinearOperator, Operator
from pygeoinf.algebra.spaces import EuclideanSpace
from pygeoinf.inference import (
    ForwardProblem,
    GaussianReference,
    LaplaceApproximation,
    LinearForwardProblem,
    LinearGaussianInversion,
    MixtureReference,
)
from pygeoinf.inference.minimization import _misfit_plus_quadratic
from pygeoinf.testing import check_gradient, check_measure

from .conftest import make_dense_metric_space


def covariance_matrix(measure, space):
    """The covariance applied to every basis vector, as columns of components."""
    return np.array(
        [measure.covariance(space.basis_vector(i)) for i in range(space.dim)]
    )


@pytest.fixture
def linear(rng):
    space = make_dense_metric_space(5)
    data_space = EuclideanSpace(3)
    forward = LinearOperator.from_matrix(
        space, data_space, rng.standard_normal((3, 5)), form="components"
    )
    prior = gi.GaussianMeasure.from_standard_deviation(space, 1.5).translate(
        space.from_components(rng.standard_normal(5))
    )
    error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.3).translate(
        np.array([0.1, -0.2, 0.05])
    )
    problem = LinearForwardProblem(forward, error=error)
    data = rng.standard_normal(3)
    return space, prior, problem, data


@pytest.fixture
def nonlinear(rng):
    """``exp(B m)`` on a dense metric: the curvature moves with m, and the
    exact Hessian differs from the Gauss-Newton one."""
    space = make_dense_metric_space(4)
    data_space = EuclideanSpace(3)
    matrix = 0.5 * rng.standard_normal((3, 4))

    def value(components):
        return np.exp(matrix @ components)

    def derivative(components):
        return LinearOperator.from_matrix(
            space,
            data_space,
            np.diag(np.exp(matrix @ components)) @ matrix,
            form="components",
        )

    def second_derivative(components, direction):
        return LinearOperator.from_matrix(
            space,
            data_space,
            np.diag(np.exp(matrix @ components) * (matrix @ direction)) @ matrix,
            form="components",
        )

    forward = Operator.from_callables(
        space,
        data_space,
        value,
        derivative=derivative,
        second_derivative=second_derivative,
    )
    prior = gi.GaussianMeasure.from_standard_deviation(space, 0.5)
    error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.1)
    data = forward(prior.sample(rng=rng)) + error.sample(rng=rng)
    return space, prior, ForwardProblem(forward, error=error), data


class TestThePriorAsItsOwnReference:
    def test_it_has_the_prior_moments_and_no_density(self, linear, rng):
        space, prior, _, _ = linear
        reference = GaussianReference.from_prior(prior)
        x = prior.sample(rng=rng)
        assert reference.rank == 0
        assert reference.prior is prior
        assert np.allclose(reference.expectation, prior.expectation)
        assert np.allclose(
            covariance_matrix(reference, space), covariance_matrix(prior, space)
        )
        assert reference.relative_log_density(x) == 0.0
        assert reference.log_normalizer.value == 0.0

    def test_it_draws_as_the_prior(self, linear, rng):
        _, prior, _, _ = linear
        check_measure(GaussianReference.from_prior(prior), rng=rng)


class TestTheLinearReference:
    def test_at_full_rank_it_is_the_posterior(self, linear):
        space, prior, problem, data = linear
        inversion = LinearGaussianInversion(problem, prior)
        posterior = inversion(data)
        reference = inversion.reference(data)
        assert reference.rank == 3
        assert np.allclose(reference.expectation, posterior.expectation, atol=1e-12)
        assert np.allclose(
            covariance_matrix(reference, space),
            covariance_matrix(posterior, space),
            atol=1e-12,
        )

    def test_its_density_is_the_posteriors_up_to_the_normalizer(self, linear, rng):
        """``-Phi`` is the posterior's own density, so differences agree."""
        _, prior, problem, data = linear
        inversion = LinearGaussianInversion(problem, prior)
        posterior = inversion(data)
        density = inversion.reference(data).relative_log_density
        u, v = prior.sample(rng=rng), prior.sample(rng=rng)
        assert density(u) - density(v) == pytest.approx(
            posterior.relative_log_density(u) - posterior.relative_log_density(v),
            rel=1e-10,
        )

    def test_its_density_is_normalized(self, linear, rng):
        """``E_prior exp(f) == 1``, to the Monte Carlo error."""
        _, prior, problem, data = linear
        density = (
            LinearGaussianInversion(problem, prior).reference(data).relative_log_density
        )
        weights = np.array(
            [np.exp(density(prior.sample(rng=rng))) for _ in range(20000)]
        )
        error = weights.std() / np.sqrt(weights.size)
        assert abs(weights.mean() - 1.0) < 5 * error

    def test_its_density_has_the_right_gradient(self, linear, rng):
        _, prior, problem, data = linear
        density = (
            LinearGaussianInversion(problem, prior).reference(data).relative_log_density
        )
        check_gradient(density, prior.sample(rng=rng), rng=rng)

    def test_it_draws_from_its_own_moments(self, linear, rng):
        _, prior, problem, data = linear
        reference = LinearGaussianInversion(problem, prior).reference(data)
        check_measure(reference, rng=rng)

    def test_at_the_data_count_the_low_rank_route_is_the_posterior(self, linear, rng):
        space, prior, problem, data = linear
        inversion = LinearGaussianInversion(problem, prior)
        posterior = inversion(data)
        reference = inversion.reference(data, rank=3, rng=rng)
        assert np.allclose(reference.expectation, posterior.expectation, atol=1e-10)
        assert np.allclose(
            covariance_matrix(reference, space),
            covariance_matrix(posterior, space),
            atol=1e-10,
        )

    def test_below_it_the_reference_is_the_prior_outside_the_chart(self, linear, rng):
        space, prior, problem, data = linear
        reference = LinearGaussianInversion(problem, prior).reference(
            data, rank=1, rng=rng
        )
        assert reference.rank == 1
        (w,) = reference.chart
        x = space.random(rng=rng)
        outside = space.axpy(
            -space.inner_product(w, prior.covariance(x)), w, space.copy(x)
        )
        assert np.allclose(reference.covariance(outside), prior.covariance(outside))


class TestTheLaplaceReference:
    def test_on_a_linear_problem_it_is_the_posterior(self, linear):
        space, prior, problem, data = linear
        posterior = LinearGaussianInversion(problem, prior)(data)
        laplace = LaplaceApproximation.from_problem(problem, prior, gauss_newton=True)
        reference = laplace.reference(posterior.expectation, data)
        assert np.allclose(reference.expectation, posterior.expectation, atol=1e-10)
        assert np.allclose(
            covariance_matrix(reference, space),
            covariance_matrix(posterior, space),
            atol=1e-10,
        )

    def test_its_covariance_is_the_gauss_newton_one_anywhere(self, nonlinear, rng):
        space, prior, problem, data = nonlinear
        laplace = LaplaceApproximation.from_problem(problem, prior, gauss_newton=True)
        model = prior.sample(rng=rng)
        reference = laplace.reference(model, data)
        approximation = laplace.at(model, data)
        assert np.allclose(
            covariance_matrix(reference, space),
            covariance_matrix(approximation, space),
            atol=1e-10,
        )

    def test_at_the_mode_its_mean_is_the_mode(self, nonlinear):
        space, prior, problem, data = nonlinear
        laplace = LaplaceApproximation.from_problem(problem, prior, gauss_newton=True)
        mode = laplace.solve(data).model
        reference = laplace.reference(mode, data)
        assert space.norm(space.subtract(reference.expectation, mode)) < 1e-5

    def test_the_low_rank_route_at_the_data_count_matches_the_full_one(
        self, nonlinear, rng
    ):
        space, prior, problem, data = nonlinear
        model = prior.sample(rng=rng)
        full = LaplaceApproximation.from_problem(problem, prior, gauss_newton=True)
        low = LaplaceApproximation.from_problem(
            problem, prior, gauss_newton=True, rank=3, rng=rng
        )
        a, b = full.reference(model, data), low.reference(model, data)
        assert np.allclose(a.expectation, b.expectation, atol=1e-8)
        assert np.allclose(
            covariance_matrix(a, space), covariance_matrix(b, space), atol=1e-8
        )

    def test_the_full_rank_exact_hessian_is_refused(self, nonlinear, rng):
        _, prior, problem, data = nonlinear
        laplace = LaplaceApproximation.from_problem(problem, prior)
        with pytest.raises(ValueError, match="gauss_newton"):
            laplace.reference(prior.sample(rng=rng), data)

    def test_the_low_rank_exact_hessian_is_a_reference(self, nonlinear, rng):
        _, prior, problem, data = nonlinear
        laplace = LaplaceApproximation.from_problem(problem, prior, rank=2, rng=rng)
        reference = laplace.reference(laplace.solve(data).model, data)
        assert 1 <= reference.rank <= 3
        check_measure(reference, rng=rng)


class TestTheChart:
    def test_dependent_directions_are_dropped(self, linear, rng):
        space, prior, _, _ = linear
        w = space.random(rng=rng)
        reference = GaussianReference(
            prior, [w, space.scale(2.0, w)], np.zeros(2), 0.5 * np.identity(2)
        )
        assert reference.rank == 1

    def test_the_chart_is_whitened_under_the_prior(self, linear, rng):
        space, prior, _, _ = linear
        vectors = [space.random(rng=rng) for _ in range(2)]
        reference = GaussianReference(prior, vectors, np.zeros(2), 0.3 * np.identity(2))
        gram = np.array(
            [
                [space.inner_product(a, prior.covariance(b)) for b in reference.chart]
                for a in reference.chart
            ]
        )
        assert np.allclose(gram, np.identity(2))

    def test_the_coordinates_are_the_charts_law(self, linear, rng):
        """Under the reference the whitened coordinates are ``N(a, S)``."""
        space, prior, _, _ = linear
        reference = GaussianReference(
            prior, [space.random(rng=rng)], np.array([0.0]), np.array([[1.0]])
        )
        a, s = reference.chart_mean, reference.chart_covariance
        reference = GaussianReference(
            prior, list(reference.chart), np.array([0.7]), np.array([[0.2]])
        )
        coordinates = np.array(
            [reference.coordinates(reference.sample(rng=rng))[0] for _ in range(4000)]
        )
        assert a.shape == (1,) and s.shape == (1, 1)
        assert coordinates.mean() == pytest.approx(0.7, abs=5 * np.sqrt(0.2 / 4000))
        assert coordinates.var() == pytest.approx(0.2, rel=0.1)

    def test_a_singular_chart_covariance_is_refused(self, linear, rng):
        space, prior, _, _ = linear
        with pytest.raises(ValueError, match="positive definite"):
            GaussianReference(
                prior,
                [space.random(rng=rng), space.random(rng=rng)],
                np.zeros(2),
                np.diag([1.0, 0.0]),
            )

    def test_a_mismatched_chart_is_refused(self, linear, rng):
        space, prior, _, _ = linear
        with pytest.raises(ValueError, match="shape"):
            GaussianReference(
                prior, [space.random(rng=rng)], np.zeros(2), np.identity(1)
            )

    def test_a_prior_that_is_not_gaussian_is_refused(self, linear):
        space, _, _, _ = linear
        with pytest.raises(TypeError):
            GaussianReference("prior", [], np.zeros(0), np.zeros((0, 0)))


@pytest.fixture
def mixture(linear, rng):
    space, prior, _, _ = linear
    chart = [space.random(rng=rng) for _ in range(2)]
    return MixtureReference(
        prior,
        chart,
        np.array([0.3, 0.7]),
        np.array([[1.0, 0.0], [-1.0, 0.5]]),
        np.array([np.diag([0.1, 0.2]), np.diag([0.3, 0.05])]),
    )


class TestTheMixtureReference:
    def test_its_density_is_normalized(self, mixture, rng):
        prior = mixture.prior
        weights = np.array(
            [
                np.exp(mixture.relative_log_density(prior.sample(rng=rng)))
                for _ in range(20000)
            ]
        )
        assert abs(weights.mean() - 1.0) < 5 * weights.std() / np.sqrt(weights.size)

    def test_its_density_has_the_right_gradient(self, mixture, rng):
        check_gradient(mixture.relative_log_density, mixture.sample(rng=rng), rng=rng)

    def test_it_draws_from_its_own_moments(self, mixture, rng):
        check_measure(mixture, rng=rng)

    def test_its_components_share_one_chart(self, mixture):
        for component in mixture.components:
            assert component.chart == mixture.chart

    def test_transport_carries_one_component_onto_another(self, mixture, rng):
        end = mixture.components[1]
        carried = np.array(
            [
                mixture.coordinates(
                    mixture.transport(mixture.components[0].sample(rng=rng), 0, 1)
                )
                for _ in range(4000)
            ]
        )
        assert carried.mean(axis=0) == pytest.approx(end.chart_mean, abs=0.02)
        assert np.cov(carried.T) == pytest.approx(end.chart_covariance, abs=0.02)

    def test_transport_there_and_back_is_the_identity(self, mixture, rng):
        space = mixture.domain
        x = mixture.sample(rng=rng)
        back = mixture.transport(mixture.transport(x, 0, 1), 1, 0)
        assert space.norm(space.subtract(back, x)) < 1e-12
        v = space.random(rng=rng)
        back = mixture.transport(
            mixture.transport(v, 0, 1, centered=True), 1, 0, centered=True
        )
        assert space.norm(space.subtract(back, v)) < 1e-12

    def test_the_responsibilities_pick_the_nearer_component(self, mixture, rng):
        x = mixture.components[1].expectation
        shares = mixture.responsibilities(x)
        assert shares.sum() == pytest.approx(1.0)
        assert shares[1] > 0.99

    def test_mismatched_parameters_are_refused(self, linear, rng):
        space, prior, _, _ = linear
        with pytest.raises(ValueError, match="shape"):
            MixtureReference(
                prior,
                [space.random(rng=rng)],
                np.ones(2),
                np.zeros((2, 2)),
                np.ones((2, 1, 1)),
            )


def components_covariance(measure, space):
    """``C_c`` with ``C x == C_c G x``: the covariance in components."""
    return covariance_matrix(measure, space).T @ np.linalg.inv(space.gram_matrix())


class TestTheNewtonMean:
    """The Laplace reference's mean is the Newton step from the model."""

    def test_on_a_linear_problem_it_is_the_posterior_from_anywhere(self, linear, rng):
        space, prior, problem, data = linear
        posterior = LinearGaussianInversion(problem, prior)(data)
        laplace = LaplaceApproximation.from_problem(problem, prior, gauss_newton=True)
        reference = laplace.reference(prior.sample(rng=rng), data)
        assert np.allclose(reference.expectation, posterior.expectation, atol=1e-10)

    @pytest.mark.parametrize("rank", [None, 3])
    def test_it_is_the_explicit_newton_step(self, nonlinear, rng, rank):
        """``u - K (grad Phi + C^-1 (u - m_0))``, formed here with the prior's
        precision, which the reference itself never uses."""
        space, prior, problem, data = nonlinear
        laplace = LaplaceApproximation.from_problem(
            problem, prior, gauss_newton=True, rank=rank, rng=rng
        )
        model = prior.sample(rng=rng)
        reference = laplace.reference(model, data)
        gram = space.gram_matrix()
        covariance = components_covariance(reference, space)
        prior_covariance = components_covariance(prior, space)
        gradient = problem.potential(data).gradient(model)
        offset = model - prior.expectation
        newton = (
            model
            - covariance @ (gram @ gradient)
            - covariance @ np.linalg.solve(prior_covariance, offset)
        )
        assert np.allclose(reference.expectation, newton, atol=1e-10)

    def test_the_low_rank_reference_is_a_fixed_function_of_the_model(self, nonlinear, rng):
        space, prior, problem, data = nonlinear
        laplace = LaplaceApproximation.from_problem(
            problem, prior, gauss_newton=True, rank=2, rng=rng
        )
        model = prior.sample(rng=rng)
        first, second = laplace.reference(model, data), laplace.reference(model, data)
        assert np.array_equal(first.expectation, second.expectation)
        assert np.array_equal(first.chart_covariance, second.chart_covariance)


def with_hessian(problem):
    """The likelihood potential carrying its exact Hessian, as a user with
    second-order adjoints would supply it."""

    def build(data):
        potential = problem.potential(data)
        misfit = _misfit_plus_quadratic(problem, data)
        return Functional.from_callables(
            potential.domain,
            potential,
            gradient=potential.gradient,
            hessian=lambda m: misfit.hessian(m) * 0.5,
        )

    return build


class TestASuppliedHessian:
    def test_it_is_the_exact_hessian_s_reference(self, nonlinear, rng):
        space, prior, problem, data = nonlinear
        mode = LaplaceApproximation.from_problem(problem, prior).solve(data).model
        exact = LaplaceApproximation.from_problem(
            problem, prior, rank=3, rng=np.random.default_rng(1)
        )
        supplied = LaplaceApproximation.from_problem(
            problem, prior, gauss_newton=True, rank=3, hessian=with_hessian(problem),
            rng=np.random.default_rng(1),
        )
        a, b = exact.reference(mode, data), supplied.reference(mode, data)
        assert np.allclose(a.expectation, b.expectation, atol=1e-12)
        assert np.allclose(covariance_matrix(a, space), covariance_matrix(b, space), atol=1e-12)

    def test_it_drives_the_laplace_gaussian_too(self, nonlinear):
        space, prior, problem, data = nonlinear
        mode = LaplaceApproximation.from_problem(problem, prior).solve(data).model
        exact = LaplaceApproximation.from_problem(problem, prior).at(mode, data)
        supplied = LaplaceApproximation.from_problem(
            problem, prior, gauss_newton=True, hessian=with_hessian(problem)
        ).at(mode, data)
        assert np.allclose(
            covariance_matrix(exact, space), covariance_matrix(supplied, space), atol=1e-10
        )

    def test_a_negative_eigenvalue_is_refused(self, nonlinear, rng):
        space, prior, problem, data = nonlinear

        def negative(data):
            return Functional.from_callables(
                space,
                lambda m: 0.0,
                gradient=lambda m: space.zero(),
                hessian=lambda m: LinearOperator.from_matrix(
                    space, space, -np.eye(space.dim), form="components"
                ),
            )

        laplace = LaplaceApproximation.from_problem(
            problem, prior, gauss_newton=True, rank=3, hessian=negative, rng=rng
        )
        with pytest.raises(ValueError, match="negative eigenvalue"):
            laplace.reference(prior.sample(rng=rng), data)

    def test_the_full_rank_reference_is_refused(self, nonlinear, rng):
        _, prior, problem, data = nonlinear
        laplace = LaplaceApproximation.from_problem(
            problem, prior, gauss_newton=True, hessian=with_hessian(problem)
        )
        with pytest.raises(ValueError, match="rank"):
            laplace.reference(prior.sample(rng=rng), data)

    def test_a_potential_without_a_hessian_is_refused(self, nonlinear, rng):
        _, prior, problem, data = nonlinear
        laplace = LaplaceApproximation.from_problem(
            problem, prior, gauss_newton=True, rank=3, hessian=problem.potential, rng=rng
        )
        with pytest.raises(TypeError, match="Hessian"):
            laplace.reference(prior.sample(rng=rng), data)


class TestThePairLaw:
    def test_it_is_the_ratio_of_two_correlated_gaussians(self, nonlinear, rng):
        """Against the ``2 n``-dimensional densities, in components."""
        from scipy.stats import multivariate_normal

        space, prior, problem, data = nonlinear
        laplace = LaplaceApproximation.from_problem(problem, prior, gauss_newton=True)
        reference = laplace.reference(prior.sample(rng=rng), data)
        c = 0.6

        def pair(mean, covariance):
            joint = np.block([[covariance, c * covariance], [c * covariance, covariance]])
            return multivariate_normal(np.r_[mean, mean], joint)

        u, v = prior.sample(rng=rng), prior.sample(rng=rng)
        direct = pair(
            reference.expectation, components_covariance(reference, space)
        ).logpdf(np.r_[u, v]) - pair(
            prior.expectation, components_covariance(prior, space)
        ).logpdf(np.r_[u, v])
        assert reference.pair_log_density(u, v, c) == pytest.approx(direct, rel=1e-10)

    def test_the_prior_s_is_zero(self, linear, rng):
        _, prior, _, _ = linear
        reference = GaussianReference.from_prior(prior)
        assert reference.pair_log_density(prior.sample(rng=rng), prior.sample(rng=rng), 0.3) == 0.0
