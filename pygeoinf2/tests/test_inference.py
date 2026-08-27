"""
The inference layer: forward problems and the estimators built on them.

Every estimator here assembles the same mapping two ways, in the model space
and in the data space, so the first thing each test does is check that the two
agree. Where a dense reference is available it is used, and the reference
carries the ``G^-1`` that turns a transpose into an adjoint — getting that
wrong is the one way to write a plausible reference that disagrees.

See DESIGN.md section 18.
"""

import numpy as np
import pytest

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.geometry.convex import Ball
from pygeoinf2.inference import (
    LinearGaussianInversion,
    ForwardProblem,
    LeastSquares,
    LinearForwardProblem,
    LinearPointEstimator,
    MeasureEstimator,
    MinimumNorm,
    PointEstimator,
    choose_formalism,
)
from pygeoinf2.probability.gaussian import GaussianMeasure

from .conftest import make_weighted_space


@pytest.fixture
def problem(rng):
    model = make_weighted_space()
    data = EuclideanSpace(4)
    operator = LinearOperator.from_derivative_matrix(
        model, data, rng.normal(size=(data.dim, model.dim))
    )
    error = GaussianMeasure.from_standard_deviation(data, 0.1)
    return LinearForwardProblem(operator, error=error)


@pytest.fixture
def prior(problem):
    return GaussianMeasure.from_standard_deviation(problem.model_space, 2.0)


def dense_posterior(problem, prior, data):
    """The posterior in components, by dense linear algebra.

    ``A*`` in components is ``G^-1 A_c^T`` when the codomain is Euclidean. A
    reference that writes ``A_c^T`` instead is the mistake of section 5.6, and
    it disagrees.
    """
    model, data_space = problem.model_space, problem.data_space
    forward = problem.forward_operator.matrix(form="components")
    covariance = prior.covariance.matrix(form="components")
    noise = problem.error_measure.covariance.matrix(form="components")
    adjoint = np.linalg.inv(model.gram_matrix()) @ forward.T
    normal = forward @ covariance @ adjoint + noise
    components = data_space.to_components(data)
    mean = covariance @ adjoint @ np.linalg.solve(normal, components)
    posterior = covariance - covariance @ adjoint @ np.linalg.solve(
        normal, forward @ covariance
    )
    return mean, posterior


class TestForwardProblem:
    def test_it_reports_its_spaces(self, problem):
        assert problem.model_space is problem.forward_operator.domain
        assert problem.data_space is problem.forward_operator.codomain
        assert problem.has_error

    def test_an_error_on_the_wrong_space_is_refused(self, problem):
        wrong = GaussianMeasure.from_standard_deviation(EuclideanSpace(7), 1.0)
        with pytest.raises(ValueError, match="not on the data space"):
            LinearForwardProblem(problem.forward_operator, error=wrong)

    def test_a_nonlinear_operator_is_refused_by_the_linear_class(self, problem):
        from pygeoinf2.algebra.operators import Operator

        nonlinear = Operator.from_callables(
            problem.model_space, problem.data_space, lambda x: problem.data_space.zero()
        )
        with pytest.raises(TypeError, match="LinearOperator"):
            LinearForwardProblem(nonlinear)
        ForwardProblem(nonlinear)  # the general class accepts it

    def test_a_set_may_stand_in_for_a_measure(self, problem):
        ball = Ball(problem.data_space, radius=0.3)
        with_set = LinearForwardProblem(problem.forward_operator, error=ball)
        assert with_set.error_set is ball
        with pytest.raises(AttributeError, match="not a measure"):
            with_set.error_measure

    def test_synthetic_data_carries_the_error(self, problem, prior, rng):
        model = prior.sample(rng=rng)
        exact = problem.forward_operator(model)
        noisy = problem.synthetic_data(model, rng=rng)
        assert not np.allclose(exact, noisy)
        assert np.allclose(exact, noisy, atol=1.0)

    def test_the_joint_measure_gives_consistent_pairs(self, problem, prior, rng):
        """Drawn together, not predicted then perturbed."""
        misfits = [
            problem.chi_squared(*problem.synthetic_model_and_data(prior, rng=rng))
            for _ in range(400)
        ]
        # the misfit of a truly consistent pair is chi-squared on dim(D)
        assert np.mean(misfits) == pytest.approx(problem.data_space.dim, rel=0.2)

    def test_the_chi_squared_test_agrees_with_the_consistency_set(
        self, problem, prior, rng
    ):
        model, data = problem.synthetic_model_and_data(prior, rng=rng)
        region = problem.consistency_set(model, level=0.95)
        assert region.contains(data) == problem.chi_squared_test(
            model, data, level=0.95
        )

    def test_a_direct_sum_joins_two_problems(self, problem, prior, rng):
        joint = LinearForwardProblem.from_direct_sum([problem, problem])
        assert joint.model_space == problem.model_space
        assert joint.data_space.dim == 2 * problem.data_space.dim
        model = prior.sample(rng=rng)
        first, second = joint.forward_operator(model)
        assert np.allclose(first, second)

    def test_a_parameterisation_restricts_the_model_space(self, problem, rng):
        small = EuclideanSpace(2)
        parameterisation = LinearOperator.from_derivative_matrix(
            small, problem.model_space, rng.normal(size=(problem.model_space.dim, 2))
        )
        reduced = problem.parameterised(parameterisation)
        assert reduced.model_space == small
        assert reduced.data_space == problem.data_space

    def test_data_reduction_pushes_the_error_forward(self, problem, rng):
        reduction = LinearOperator.from_derivative_matrix(
            problem.data_space, EuclideanSpace(2), rng.normal(size=(2, 4))
        )
        reduced = problem.data_reduced(reduction)
        assert reduced.data_space.dim == 2
        assert reduced.has_error


class TestFormalism:
    def test_auto_takes_the_smaller_space(self, problem):
        assert choose_formalism(problem) in ("model_space", "data_space")
        smaller = (
            "model_space"
            if problem.model_space.dim <= problem.data_space.dim
            else "data_space"
        )
        assert choose_formalism(problem) == smaller

    def test_an_unknown_formalism_is_refused(self, problem):
        with pytest.raises(ValueError, match="'auto'"):
            choose_formalism(problem, formalism="spectral")


class TestBayesian:
    def test_the_two_formalisms_agree(self, problem, prior, rng):
        data = problem.synthetic_data(prior.sample(rng=rng), rng=rng)
        results = {
            name: LinearGaussianInversion(problem, prior, formalism=name)(data)
            for name in ("model_space", "data_space")
        }
        model = problem.model_space
        assert np.allclose(
            model.to_components(results["model_space"].expectation),
            model.to_components(results["data_space"].expectation),
        )
        assert np.allclose(
            results["model_space"].covariance.matrix(form="components"),
            results["data_space"].covariance.matrix(form="components"),
        )

    def test_it_matches_a_dense_reference(self, problem, prior, rng):
        data = problem.synthetic_data(prior.sample(rng=rng), rng=rng)
        posterior = LinearGaussianInversion(problem, prior)(data)
        mean, covariance = dense_posterior(problem, prior, data)
        assert np.allclose(
            problem.model_space.to_components(posterior.expectation), mean
        )
        assert np.allclose(posterior.covariance.matrix(form="components"), covariance)

    def test_the_covariance_does_not_depend_on_the_data(self, problem, prior, rng):
        estimator = LinearGaussianInversion(problem, prior)
        first = estimator(problem.data_space.random(rng=rng))
        second = estimator(problem.data_space.random(rng=rng))
        assert first.covariance is second.covariance

    def test_it_is_a_measure_estimator(self, problem, prior):
        assert isinstance(LinearGaussianInversion(problem, prior), MeasureEstimator)

    def test_pushing_forward_agrees_with_pushing_the_answer(self, problem, prior, rng):
        """``(measure, P)`` is free, and the direct path must give the same."""
        model = problem.model_space
        target = LinearOperator.from_derivative_matrix(
            model, EuclideanSpace(2), rng.normal(size=(2, model.dim))
        )
        data = problem.synthetic_data(prior.sample(rng=rng), rng=rng)
        estimator = LinearGaussianInversion(problem, prior)
        direct = estimator.push_forward(target)(data)
        indirect = estimator(data).push_forward(target)
        assert np.allclose(direct.expectation, indirect.expectation)
        assert np.allclose(
            direct.covariance.matrix(form="components"),
            indirect.covariance.matrix(form="components"),
        )

    def test_the_posterior_fluctuation_does_not_depend_on_the_data(
        self, problem, prior, rng
    ):
        """The same statement as the covariance being data-independent.

        A randomise-then-optimise draw written about the mean never touches
        the data vector, so two estimators differing only in their data give
        the same fluctuation from the same seed.
        """
        from numpy.random import default_rng

        estimator = LinearGaussianInversion(problem, prior)
        first = estimator._centred_sample(default_rng(11))
        second = estimator._centred_sample(default_rng(11))
        assert np.allclose(first, second)

    def test_the_sampler_is_not_double_counting_the_mean(self, problem, prior, rng):
        """The negative control for a supplied ``sample`` callable.

        ``GaussianMeasure`` adds the expectation to whatever the callable
        returns, so a sampler that already includes it lands at twice the mean.
        """
        estimator = LinearGaussianInversion(problem, prior)
        model = problem.model_space
        centred = [estimator._centred_sample(rng) for _ in range(2000)]
        assert np.allclose(model.to_components(model.mean(centred)), 0.0, atol=0.2)

    def test_the_posterior_can_be_sampled(self, problem, prior, rng):
        """Randomise-then-optimise, whose mean must be the posterior mean."""
        data = problem.synthetic_data(prior.sample(rng=rng), rng=rng)
        posterior = LinearGaussianInversion(problem, prior)(data)
        assert posterior.can_sample
        draws = [posterior.sample(rng=rng) for _ in range(3000)]
        mean = problem.model_space.mean(draws)
        assert np.allclose(
            problem.model_space.to_components(mean),
            problem.model_space.to_components(posterior.expectation),
            atol=0.15,
        )

    def test_a_prior_on_the_wrong_space_is_refused(self, problem):
        wrong = GaussianMeasure.from_standard_deviation(EuclideanSpace(9), 1.0)
        with pytest.raises(ValueError, match="model space"):
            LinearGaussianInversion(problem, wrong)

    def test_a_nonzero_prior_mean_shifts_the_answer(self, problem, rng):
        model = problem.model_space
        shifted = GaussianMeasure.from_standard_deviation(
            model, 2.0, expectation=model.random(rng=rng)
        )
        centred = GaussianMeasure.from_standard_deviation(model, 2.0)
        data = problem.data_space.random(rng=rng)
        assert not np.allclose(
            model.to_components(
                LinearGaussianInversion(problem, shifted)(data).expectation
            ),
            model.to_components(
                LinearGaussianInversion(problem, centred)(data).expectation
            ),
        )


class TestPointEstimators:
    def test_least_squares_agrees_across_formalisms(self, problem, prior, rng):
        data = problem.synthetic_data(prior.sample(rng=rng), rng=rng)
        results = [
            LeastSquares(problem, damping=1e-2, formalism=name)(data)
            for name in ("model_space", "data_space")
        ]
        model = problem.model_space
        assert np.allclose(
            model.to_components(results[0]), model.to_components(results[1])
        )

    def test_it_is_a_point_estimator_and_an_operator(self, problem):
        estimator = LeastSquares(problem, damping=1e-2)
        assert isinstance(estimator, PointEstimator)
        assert isinstance(estimator, LinearPointEstimator)
        assert estimator.data_space == problem.data_space
        assert estimator.target_space == problem.model_space

    def test_the_resolution_operator_is_available(self, problem):
        estimator = LeastSquares(problem, damping=1e-2)
        resolution = estimator.resolution
        assert resolution.domain == problem.model_space
        assert resolution.codomain == problem.model_space

    def test_the_data_error_propagates(self, problem):
        estimator = LeastSquares(problem, damping=1e-2)
        covariance = estimator.propagated_covariance()
        assert covariance.domain == problem.model_space
        matrix = covariance.matrix(form="galerkin")
        assert np.linalg.eigvalsh(0.5 * (matrix + matrix.T)).min() > -1e-12

    def test_negative_damping_is_refused(self, problem):
        with pytest.raises(ValueError, match="non-negative"):
            LeastSquares(problem, damping=-1.0)

    def test_the_discrepancy_principle_hits_its_target(self, problem, prior, rng):
        data = problem.synthetic_data(prior.sample(rng=rng), rng=rng)
        estimator = MinimumNorm(problem).for_data(data, level=0.95)
        target = problem.critical_chi_squared(level=0.95)
        assert problem.chi_squared(estimator(data), data) == pytest.approx(
            target, rel=1e-3
        )

    def test_more_damping_gives_a_smaller_model(self, problem, prior, rng):
        data = problem.synthetic_data(prior.sample(rng=rng), rng=rng)
        model = problem.model_space
        norms = [
            model.norm(LeastSquares(problem, damping=damping)(data))
            for damping in (1e-4, 1e-1, 1e2)
        ]
        assert norms[0] > norms[1] > norms[2]

    def test_a_point_estimator_induces_a_measure(self, problem, rng):
        estimator = LeastSquares(problem, damping=1e-2)
        induced = estimator.as_measure()
        data = problem.data_space.random(rng=rng)
        assert np.allclose(induced(data).expectation, estimator(data))


class TestEvidence:
    def test_it_matches_scipy(self, problem, prior, rng):
        """``log p(d)`` is a multivariate normal density on the data space,
        and the reference is the one scipy computes."""
        from scipy.stats import multivariate_normal

        estimator = LinearGaussianInversion(problem, prior)
        data = problem.data_space.random(rng=rng)
        covariance = problem.data_measure_from_model_measure(prior).covariance
        reference = multivariate_normal(
            mean=problem.data_space.to_components(
                problem.forward_operator(prior.expectation)
            ),
            cov=covariance.matrix(form="components"),
        ).logpdf(problem.data_space.to_components(data))
        assert estimator.log_evidence(data) == pytest.approx(reference)

    def test_the_two_terms_answer_different_questions(self, problem, prior, rng):
        """Misfit and volume, separately: whether the data are surprising, and
        whether the model was flexible enough that they could not have been."""
        estimator = LinearGaussianInversion(problem, prior)
        data = problem.data_space.random(rng=rng)
        mahalanobis, volume = estimator.evidence_terms(data)
        assert mahalanobis > 0.0
        assert np.isfinite(volume)

    def test_a_wilder_prior_is_penalised_on_data_it_did_not_need(self, problem, rng):
        """The whole point of an evidence: fitting is not the same as
        explaining."""
        model = problem.model_space
        tight = GaussianMeasure.from_standard_deviation(model, 0.5)
        loose = GaussianMeasure.from_standard_deviation(model, 50.0)
        small = problem.synthetic_data(model.scale(0.1, model.random(rng=rng)), rng=rng)
        assert LinearGaussianInversion(problem, tight).log_evidence(
            small
        ) > LinearGaussianInversion(problem, loose).log_evidence(small)


class TestConstrainedLeastSquares:
    @pytest.fixture
    def constrained(self, problem, rng):
        from pygeoinf2.geometry.subspaces import AffineSubspace

        model = problem.model_space
        constraint = LinearOperator.from_derivative_matrix(
            model, EuclideanSpace(1), rng.normal(size=(1, model.dim))
        )
        subspace = AffineSubspace.from_linear_equation(constraint, np.array([2.0]))
        return constraint, subspace

    def test_the_answer_satisfies_the_constraint(self, problem, constrained, rng):
        from pygeoinf2.inference import ConstrainedLeastSquares

        constraint, subspace = constrained
        data = problem.data_space.random(rng=rng)
        answer = ConstrainedLeastSquares(problem, subspace, damping=1e-3)(data)
        assert np.allclose(constraint(answer), [2.0])

    def test_the_unconstrained_answer_does_not(self, problem, constrained, rng):
        """Which is what makes the constrained one a different estimator and
        not a tidier way of writing the same one."""
        from pygeoinf2.inference import ConstrainedLeastSquares

        constraint, subspace = constrained
        data = problem.data_space.random(rng=rng)
        assert not np.allclose(
            constraint(LeastSquares(problem, damping=1e-3)(data)), [2.0]
        )
        ConstrainedLeastSquares(problem, subspace, damping=1e-3)(data)

    def test_it_is_optimal_within_the_subspace(self, problem, constrained, rng):
        from pygeoinf2.inference import ConstrainedLeastSquares

        constraint, subspace = constrained
        model, data_space = problem.model_space, problem.data_space
        data = data_space.random(rng=rng)
        answer = ConstrainedLeastSquares(problem, subspace, damping=1e-3)(data)

        def objective(x):
            return data_space.squared_norm(
                data_space.subtract(problem.forward_operator(x), data)
            ) + 1e-3 * model.squared_norm(x)

        best = objective(answer)
        for size in (0.01, 0.1, 0.5):
            for _ in range(4):
                nudged = model.add(
                    answer,
                    model.scale(size, subspace.projector(model.random(rng=rng))),
                )
                assert objective(nudged) > best

    def test_it_is_still_an_operator(self, problem, constrained):
        from pygeoinf2.inference import ConstrainedLeastSquares

        _, subspace = constrained
        estimator = ConstrainedLeastSquares(problem, subspace)
        assert isinstance(estimator, LinearPointEstimator)
        assert estimator.subspace is subspace

    def test_a_subspace_of_the_wrong_space_is_refused(self, problem, rng):
        from pygeoinf2.geometry.subspaces import AffineSubspace, OrthogonalProjector
        from pygeoinf2.inference import ConstrainedLeastSquares

        other = EuclideanSpace(3)
        subspace = AffineSubspace(
            OrthogonalProjector.from_basis(other, [other.random(rng=rng)])
        )
        with pytest.raises(ValueError, match="model space"):
            ConstrainedLeastSquares(problem, subspace)
