"""
Gaussian mixtures, and the posterior of a mixture prior.

The mixture posterior is *exact*, so almost everything here is checked against
a reference written independently in plain numpy from the definitions —
component means and covariances by the Kalman formulas, weights by each
component's marginal likelihood. Agreement to machine precision is the claim,
not agreement to a tolerance.

The one thing that cannot be checked that way is the law of total covariance,
which is checked against sampling instead: it is the term a single Gaussian
does not have, and getting it wrong is how a mixture silently becomes a blur.

See DESIGN.md section 31.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.inference import (
    LinearForwardProblem,
    LinearGaussianMixtureInversion,
)
from pygeoinf2.numerics.solvers import CholeskySolver
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.probability.mixture import GaussianMixture
from pygeoinf2.traits import Traits

from .conftest import make_weighted_space


def isotropic(space, mean, deviation):
    return GaussianMeasure.from_standard_deviation(
        space, deviation, expectation=space.from_components(np.asarray(mean, float))
    )


class TestGaussianMixture:
    @pytest.fixture
    def mixture(self):
        space = EuclideanSpace(2)
        return space, GaussianMixture(
            [isotropic(space, [-2.0, 0.0], 0.3), isotropic(space, [3.0, 1.0], 0.5)],
            weights=[0.3, 0.7],
        )

    def test_the_expectation_is_the_weighted_mean(self, mixture):
        space, mix = mixture
        assert space.to_components(mix.expectation) == pytest.approx(
            [0.3 * -2.0 + 0.7 * 3.0, 0.7 * 1.0]
        )

    @pytest.mark.slow
    def test_the_covariance_is_the_law_of_total_covariance(self, mixture, rng):
        """``E[C | k] + Cov(m | k)``. The second term is the spread *between*
        components — here it is an order of magnitude larger than either
        component's own, which is what a mixture is for and what dropping it
        would quietly discard."""
        space, mix = mixture
        stated = mix.covariance.matrix(form="components")
        draws = np.stack(
            [space.to_components(mix.sample(rng=rng)) for _ in range(300000)]
        )
        assert np.abs(np.cov(draws.T) - stated).max() < 0.03 * np.abs(stated).max()
        within = 0.3 * 0.09 + 0.7 * 0.25
        assert stated[0, 0] > 10.0 * within

    def test_sampling_picks_components_in_proportion(self, mixture, rng):
        space, mix = mixture
        draws = np.stack(
            [space.to_components(mix.sample(rng=rng)) for _ in range(4000)]
        )
        # The two modes are far apart relative to their widths, so which
        # component a draw came from is unambiguous.
        assert np.mean(draws[:, 0] > 0.5) == pytest.approx(0.7, abs=0.03)

    def test_the_density_is_the_weighted_sum(self, mixture):
        space, mix = mixture
        point = space.from_components(np.array([0.4, 0.2]))
        expected = np.log(
            0.3
            * np.exp(
                mix.components[0].log_density(point)
                + mix.components[0].log_normalising_constant()
            )
            + 0.7
            * np.exp(
                mix.components[1].log_density(point)
                + mix.components[1].log_normalising_constant()
            )
        )
        assert mix.log_density(point) == pytest.approx(expected)

    def test_the_density_matches_an_independent_reference(self, mixture):
        """Against scipy, with components of *unequal* width.

        Each component's ``-1/2 log det C`` differs, so it does not cancel out
        of the sum. Omitting it makes the broad component as tall at its centre
        as the narrow one; sampling never notices, because sampling does not
        consult the density.
        """
        space, mix = mixture
        components = np.array([0.4, 0.2])
        reference = np.log(
            0.3 * multivariate_normal(mean=[-2.0, 0.0], cov=0.3**2).pdf(components)
            + 0.7 * multivariate_normal(mean=[3.0, 1.0], cov=0.5**2).pdf(components)
        )
        assert mix.log_density(space.from_components(components)) == pytest.approx(
            reference
        )

    def test_it_says_which_component_a_point_came_from(self, mixture):
        space, mix = mixture
        assert mix.marginal_probabilities(
            mix.components[0].expectation
        ) == pytest.approx([1.0, 0.0], abs=1e-6)

    def test_the_responsibilities_match_an_independent_reference(self, mixture):
        """The per-component constant does not cancel in the softmax."""
        space, mix = mixture
        components = np.array([0.4, 0.2])
        weighted = np.array(
            [
                0.3 * multivariate_normal(mean=[-2.0, 0.0], cov=0.3**2).pdf(components),
                0.7 * multivariate_normal(mean=[3.0, 1.0], cov=0.5**2).pdf(components),
            ]
        )
        assert mix.marginal_probabilities(
            space.from_components(components)
        ) == pytest.approx(weighted / weighted.sum())

    def test_an_affine_map_maps_every_component(self, mixture, rng):
        space, mix = mixture
        target = EuclideanSpace(1)
        operator = LinearOperator.from_matrix(
            space, target, np.array([[1.0, -1.0]]), form="galerkin"
        )
        mapped = mix.push_forward(operator)
        assert isinstance(mapped, GaussianMixture)
        assert np.allclose(mapped.weights, mix.weights)
        assert target.to_components(mapped.expectation) == pytest.approx(
            target.to_components(operator(mix.expectation))
        )

    def test_from_family_builds_the_support(self):
        space = EuclideanSpace(2)
        mix = GaussianMixture.from_family(
            lambda scale: isotropic(space, [0.0, 0.0], scale),
            [0.1, 1.0, 3.0],
            weights=[0.2, 0.5, 0.3],
        )
        assert len(mix) == 3
        assert mix.weights == pytest.approx([0.2, 0.5, 0.3])

    def test_a_continuous_parameter_is_discretised_by_sampling(self):
        space = EuclideanSpace(2)
        parameters = GaussianMeasure.from_standard_deviation(EuclideanSpace(1), 1.0)
        mix = GaussianMixture.from_parameter_samples(
            lambda theta: isotropic(space, [float(theta[0]), 0.0], 0.5),
            parameters,
            count=16,
            rng=np.random.default_rng(1),
        )
        assert len(mix) == 16
        assert mix.weights == pytest.approx(np.full(16, 1 / 16))

    def test_what_cannot_be_mixed_is_refused(self, mixture):
        space, mix = mixture
        with pytest.raises(ValueError, match="at least one component"):
            GaussianMixture([])
        with pytest.raises(ValueError, match="must share a space"):
            GaussianMixture(
                [mix.components[0], isotropic(EuclideanSpace(3), [0, 0, 0], 1.0)]
            )
        with pytest.raises(ValueError, match="weights for"):
            GaussianMixture(mix.components, weights=[1.0])
        with pytest.raises(ValueError, match="non-negative"):
            GaussianMixture(mix.components, weights=[-1.0, 2.0])


class TestMixtureInversion:
    """The posterior of a mixture prior, against an independent reference."""

    @pytest.fixture(params=["euclidean", "weighted"])
    def setup(self, request, rng):
        model = (
            EuclideanSpace(3) if request.param == "euclidean" else make_weighted_space()
        )
        data = EuclideanSpace(2)
        matrix = rng.normal(size=(2, model.dim))
        forward = LinearOperator.from_matrix(model, data, matrix, form="galerkin")
        variances = np.array([0.04, 0.09])
        chol = CholeskySolver()
        covariance = LinearOperator.from_matrix(
            data,
            data,
            np.diag(variances),
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
            form="galerkin",
        )
        noise = GaussianMeasure(
            data,
            covariance=covariance,
            precision=chol(covariance),
            covariance_factor=LinearOperator.from_matrix(
                data, data, np.diag(np.sqrt(variances)), form="galerkin"
            ),
        )
        problem = LinearForwardProblem(forward, error=noise)

        def component(mean, spread):
            galerkin = np.diag(np.full(model.dim, spread))
            operator = LinearOperator.from_matrix(
                model,
                model,
                galerkin,
                traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
                form="galerkin",
            )
            return GaussianMeasure(
                model,
                covariance=operator,
                precision=chol(operator),
                expectation=model.from_components(np.asarray(mean, float)),
            )

        first = np.full(model.dim, -1.0)
        second = np.full(model.dim, 2.0)
        prior = GaussianMixture(
            [component(first, 0.2), component(second, 1.0)], weights=[0.4, 0.6]
        )
        return problem, prior, model, data

    @staticmethod
    def reference(problem, prior, data_vector, model, data):
        """The mixture posterior, written out in components.

        Component covariances are read off the operators rather than
        reconstructed, so the metric is whatever the space says it is and this
        stays a reference for the *mixture* algebra rather than a second
        implementation of the Hilbert-space bookkeeping.
        """
        forward = problem.forward_operator.matrix(form="components")
        gram = model.gram_matrix()
        noise_cov = problem.error_measure.covariance.matrix(form="components")
        observed = data.to_components(data_vector)
        noise_inverse = np.linalg.inv(noise_cov)

        means, covariances, logs = [], [], []
        for component, weight in zip(prior.components, prior.weights):
            # Covariances of the *components*, which is G^-1 C_gal G^-1.
            galerkin = component.covariance.matrix(form="galerkin")
            covariance = np.linalg.solve(gram, np.linalg.solve(gram, galerkin).T)
            mean = model.to_components(component.expectation)
            inverse = np.linalg.inv(covariance)
            posterior = np.linalg.inv(inverse + forward.T @ noise_inverse @ forward)
            means.append(
                posterior @ (inverse @ mean + forward.T @ noise_inverse @ observed)
            )
            covariances.append(posterior)
            logs.append(
                np.log(weight)
                + multivariate_normal(
                    mean=forward @ mean,
                    cov=forward @ covariance @ forward.T + noise_cov,
                ).logpdf(observed)
            )
        logs = np.array(logs)
        weights = np.exp(logs - logs.max())
        return means, covariances, weights / weights.sum(), logs

    def test_it_matches_the_reference_exactly(self, setup, rng):
        problem, prior, model, data = setup
        truth = model.from_components(np.full(model.dim, 2.1))
        observed = problem.synthetic_data(truth, rng=rng)
        posterior = LinearGaussianMixtureInversion(
            problem, prior, solver=CholeskySolver()
        )(observed)

        means, covariances, weights, _ = self.reference(
            problem, prior, observed, model, data
        )
        assert posterior.weights == pytest.approx(weights, abs=1e-12)
        for index, component in enumerate(posterior.components):
            galerkin = component.covariance.matrix(form="galerkin")
            gram = model.gram_matrix()
            got = np.linalg.solve(gram, np.linalg.solve(gram, galerkin).T)
            assert model.to_components(component.expectation) == pytest.approx(
                means[index], abs=1e-9
            )
            assert got == pytest.approx(covariances[index], abs=1e-9)

    def test_the_data_choose_a_mode(self, setup, rng):
        """The reason a mixture is worth having: a single Gaussian posterior
        cannot change which mode it prefers, and this can."""
        problem, prior, model, _ = setup
        inversion = LinearGaussianMixtureInversion(
            problem, prior, solver=CholeskySolver()
        )
        near_first = problem.synthetic_data(
            model.from_components(np.full(model.dim, -1.0)), rng=rng
        )
        near_second = problem.synthetic_data(
            model.from_components(np.full(model.dim, 2.0)), rng=rng
        )
        assert inversion.weights(near_first)[0] > inversion.weights(near_second)[0]
        assert inversion(near_second).weights[1] > 0.5

    def test_the_mixture_evidence_is_not_the_best_component(self, setup, rng):
        """A mixture that hedges pays for the components that were wrong, and
        that is where a model comparison sees the cost of hedging."""
        problem, prior, model, _ = setup
        inversion = LinearGaussianMixtureInversion(
            problem, prior, solver=CholeskySolver()
        )
        observed = problem.synthetic_data(
            model.from_components(np.full(model.dim, 2.0)), rng=rng
        )
        terms = inversion.log_evidence_terms(observed)
        assert inversion.log_evidence(observed) < terms.max()

    def test_a_property_posterior_is_a_mixture_with_the_same_weights(self, setup, rng):
        """The weights are decided by the data through the evidence; a property
        map is applied afterwards and cannot change them."""
        problem, prior, model, _ = setup
        target = EuclideanSpace(1)
        operator = LinearOperator.from_matrix(
            model, target, np.ones((1, model.dim)), form="galerkin"
        )
        inversion = LinearGaussianMixtureInversion(
            problem, prior, solver=CholeskySolver()
        )
        observed = problem.synthetic_data(
            model.from_components(np.full(model.dim, 2.0)), rng=rng
        )
        pushed = inversion.push_forward(operator)(observed)
        assert isinstance(pushed, GaussianMixture)
        assert pushed.weights == pytest.approx(inversion.weights(observed))
        assert target.to_components(pushed.expectation) == pytest.approx(
            target.to_components(operator(inversion(observed).expectation)), abs=1e-9
        )

    def test_a_single_gaussian_prior_is_refused(self, setup):
        problem, prior, _, _ = setup
        with pytest.raises(TypeError, match="must be a GaussianMixture"):
            LinearGaussianMixtureInversion(problem, prior.components[0])
