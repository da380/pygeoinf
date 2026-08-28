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
from pygeoinf2.numerics.solvers import CholeskySolver
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.traits import Traits

from .conftest import make_dense_metric_space, make_weighted_space


@pytest.fixture(
    params=[make_weighted_space, make_dense_metric_space],
    ids=["weighted", "dense-metric"],
)
def model_space(request):
    """The metric rule: only a *non-diagonal* Gram matrix distinguishes
    metric-correct inference from inference that happens to agree with the
    components. A weighted space does not."""
    return request.param()


@pytest.fixture
def problem(rng, model_space):
    model = model_space
    data = EuclideanSpace(4)
    operator = LinearOperator.from_matrix(
        model, data, rng.normal(size=(data.dim, model.dim)), form="galerkin"
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
        """And asking for it as a measure is a *type* error, not a missing
        attribute: bounded errors are a different kind of problem, not a
        problem missing a field, and the message says where they are
        handled."""
        ball = Ball(problem.data_space, radius=0.3)
        with_set = LinearForwardProblem(problem.forward_operator, error=ball)
        assert with_set.error_set is ball
        with pytest.raises(TypeError, match="backus"):
            with_set.error_measure

    def test_a_measure_is_not_a_set_either(self, problem):
        with pytest.raises(TypeError, match="gaussian"):
            problem.error_set

    def test_no_error_at_all_is_still_a_missing_attribute(self, problem):
        """Which it is: nothing was supplied, so there is nothing to be the
        wrong type."""
        bare = LinearForwardProblem(problem.forward_operator)
        with pytest.raises(AttributeError):
            bare.error_measure
        with pytest.raises(AttributeError):
            bare.error_set

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

    def test_joining_a_noisy_problem_to_an_exact_one_is_refused(self, problem):
        """The joint problem used to come out with ``error=None``: a noisy data
        set joined to an exact one silently lost its noise, and every inversion
        built on the result then treated it as exact."""
        exact = LinearForwardProblem(problem.forward_operator)
        assert problem.has_error and not exact.has_error

        with pytest.raises(ValueError, match="or none may"):
            LinearForwardProblem.from_direct_sum([problem, exact])
        with pytest.raises(ValueError, match="or none may"):
            LinearForwardProblem.from_direct_sum([exact, problem])

        # All-or-nothing still works, both ways round.
        assert LinearForwardProblem.from_direct_sum([problem, problem]).has_error
        assert not LinearForwardProblem.from_direct_sum([exact, exact]).has_error

    def test_a_parameterisation_restricts_the_model_space(self, problem, rng):
        small = EuclideanSpace(2)
        parameterisation = LinearOperator.from_matrix(
            small,
            problem.model_space,
            rng.normal(size=(problem.model_space.dim, 2)),
            form="galerkin",
        )
        reduced = problem.parameterised(parameterisation)
        assert reduced.model_space == small
        assert reduced.data_space == problem.data_space

    def test_data_reduction_pushes_the_error_forward(self, problem, rng):
        reduction = LinearOperator.from_matrix(
            problem.data_space,
            EuclideanSpace(2),
            rng.normal(size=(2, 4)),
            form="galerkin",
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
        target = LinearOperator.from_matrix(
            model, EuclideanSpace(2), rng.normal(size=(2, model.dim)), form="galerkin"
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
        constraint = LinearOperator.from_matrix(
            model, EuclideanSpace(1), rng.normal(size=(1, model.dim)), form="galerkin"
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
        estimator = ConstrainedLeastSquares(problem, subspace, damping=1e-3)
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


class TestEvidenceWithoutAssembling:
    """The evidence for a problem too large to form a matrix for.

    A log-determinant is the one part of the calculation that looks as though
    it needs the matrix, and the reason a dense-only evidence is confined to
    problems where model comparison is not the interesting question. Both
    halves are matrix-free here, and both are checked against the dense route
    they must reproduce.
    """

    @pytest.fixture
    def gaussian(self, rng):
        from pygeoinf2.numerics.solvers import CholeskySolver

        model, data = make_weighted_space(), EuclideanSpace(9)
        forward = LinearOperator.from_matrix(
            model, data, rng.normal(size=(data.dim, model.dim)), form="galerkin"
        )

        def positive(space, scale=1.0):
            root = rng.normal(size=(space.dim, space.dim))
            return LinearOperator.from_matrix(
                space,
                space,
                scale * (root @ root.T + space.dim * np.identity(space.dim)),
                traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
                form="galerkin",
            )

        chol = CholeskySolver()
        prior_covariance = positive(model)
        error_covariance = positive(data, 0.2)
        prior = GaussianMeasure(
            model, covariance=prior_covariance, precision=chol(prior_covariance)
        )
        error = GaussianMeasure(
            data, covariance=error_covariance, precision=chol(error_covariance)
        )
        return LinearForwardProblem(forward, error=error), prior

    def test_sylvesters_identity_gives_the_same_evidence(self, gaussian, rng):
        """The model-space formalism never forms ``A Q A* + R``, even to take
        its determinant: ``|A Q A* + R| == |Q| |R| |Q^-1 + A* R^-1 A|``. Two
        different operators on two different spaces, and one answer."""
        problem, prior = gaussian
        data = problem.data_space.random(rng=rng)
        values = {
            name: LinearGaussianInversion(problem, prior, formalism=name).log_evidence(
                data, method="dense"
            )
            for name in ("data_space", "model_space")
        }
        assert values["model_space"] == pytest.approx(values["data_space"], rel=1e-9)

    @pytest.mark.parametrize("formalism", ["data_space", "model_space"])
    def test_the_stochastic_route_agrees_with_the_dense_one(self, gaussian, formalism):
        problem, prior = gaussian
        estimator = LinearGaussianInversion(problem, prior, formalism=formalism)
        exact = estimator.normal_log_determinant(method="dense")
        estimate = estimator.normal_log_determinant(
            method="stochastic",
            samples=4000,
            rng=np.random.default_rng(3),
            max_iterations=60,
            rtol=1e-10,
        )
        assert estimate.standard_error > 0.0
        assert abs(estimate.value - exact.value) < 4.0 * estimate.standard_error

    def test_the_misfit_is_matrix_free(self, gaussian, rng):
        """It goes through whatever solver the estimator was given, so an
        iterative preconditioned solve must land in the same place as a
        factorisation — that is the whole claim."""
        from pygeoinf2.numerics.preconditioners import JacobiPreconditioner
        from pygeoinf2.numerics.solvers import CGSolver, CholeskySolver

        problem, prior = gaussian
        data = problem.data_space.random(rng=rng)
        direct = LinearGaussianInversion(
            problem, prior, formalism="data_space", solver=CholeskySolver()
        ).mahalanobis(data)
        iterative = LinearGaussianInversion(
            problem,
            prior,
            formalism="data_space",
            solver=CGSolver(rtol=1e-12).with_preconditioner(JacobiPreconditioner()),
        ).mahalanobis(data)
        assert iterative == pytest.approx(direct, rel=1e-8)

    def test_the_two_formalisms_agree_on_the_misfit(self, gaussian, rng):
        """The model-space route reaches it by Woodbury, avoiding the
        data-space inverse entirely."""
        problem, prior = gaussian
        data = problem.data_space.random(rng=rng)
        values = [
            LinearGaussianInversion(problem, prior, formalism=name).mahalanobis(data)
            for name in ("data_space", "model_space")
        ]
        assert values[0] == pytest.approx(values[1], rel=1e-8)

    def test_evidence_needs_a_data_error_measure(self, gaussian, rng):
        problem, prior = gaussian
        without = LinearForwardProblem(problem.forward_operator)
        # Data space: an error-free problem has no model-space assembly at all.
        estimator = LinearGaussianInversion(without, prior, formalism="data_space")
        with pytest.raises(ValueError, match="data error measure"):
            estimator.mahalanobis(problem.data_space.random(rng=rng))
        with pytest.raises(ValueError, match="data error measure"):
            estimator.normal_log_determinant()


class TestPosteriorSampling:
    """Randomise-then-optimise, and that it survives a push-forward.

    v1 attaches the sampler when it builds the posterior measure. v2 did the
    same, in an override of ``__call__`` — which ``push_forward`` then threw
    away, so the *property* posterior could not be drawn from even though the
    model posterior could and the draw is one operator application away. The
    sampler is now carried by the estimator, which is what makes it travel.
    """

    @pytest.fixture
    def setup(self, rng):
        from pygeoinf2.numerics.functional_calculus import operator_sqrt

        model, data = make_weighted_space(), EuclideanSpace(6)
        forward = LinearOperator.from_matrix(
            model, data, rng.normal(size=(6, model.dim)), form="galerkin"
        )
        root = rng.normal(size=(model.dim, model.dim))
        covariance = LinearOperator.from_matrix(
            model,
            model,
            root @ root.T + model.dim * np.identity(model.dim),
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
            form="galerkin",
        )
        prior = GaussianMeasure(
            model,
            covariance=covariance,
            covariance_factor=operator_sqrt(covariance),
        )
        problem = LinearForwardProblem(
            forward, error=GaussianMeasure.from_standard_deviation(data, 0.05)
        )
        return problem, prior

    def test_the_property_posterior_can_be_sampled(self, setup, rng):
        problem, prior = setup
        target = EuclideanSpace(2)
        operator = LinearOperator.from_matrix(
            problem.model_space,
            target,
            rng.normal(size=(2, problem.model_space.dim)),
            form="galerkin",
        )
        estimator = LinearGaussianInversion(
            problem, prior, solver=CholeskySolver()
        ).push_forward(operator)
        assert estimator.can_sample
        posterior = estimator(problem.data_space.random(rng=rng))
        assert posterior.can_sample

    @pytest.mark.slow
    def test_the_draws_have_the_posterior_covariance(self, setup, rng):
        """The check that the sampler is the *right* sampler and not merely
        present: randomise-then-optimise never forms a factor of the posterior
        covariance, so nothing about the draw makes this true by construction.
        """
        problem, prior = setup
        target = EuclideanSpace(2)
        operator = LinearOperator.from_matrix(
            problem.model_space,
            target,
            rng.normal(size=(2, problem.model_space.dim)),
            form="galerkin",
        )
        estimator = LinearGaussianInversion(
            problem, prior, solver=CholeskySolver()
        ).push_forward(operator)
        posterior = estimator(problem.data_space.random(rng=rng))
        draws = np.array(
            [target.to_components(posterior.sample(rng=rng)) for _ in range(40000)]
        )
        stated = posterior.as_multivariate_normal()
        scale = np.abs(stated.cov).max()
        assert np.abs(draws.mean(axis=0) - stated.mean).max() < 0.05 * np.sqrt(scale)
        assert np.abs(np.cov(draws.T) - stated.cov).max() < 0.05 * scale

    def test_a_prior_that_cannot_be_sampled_gives_a_posterior_that_cannot(self, setup):
        """The sampler needs a draw of the prior and a draw of the noise. A
        covariance with no factor supplies neither, and the estimator says so
        rather than failing at the point of use."""
        problem, prior = setup
        unsamplable = GaussianMeasure(problem.model_space, covariance=prior.covariance)
        assert not unsamplable.can_sample
        estimator = LinearGaussianInversion(
            problem, unsamplable, solver=CholeskySolver()
        )
        assert not estimator.can_sample
        assert not estimator.push_forward(
            LinearOperator.identity(problem.model_space)
        ).can_sample


class TestJointInversionEndToEnd:
    """Two problems, different noise, joined and inverted -- against a dense
    reference. The review asked for exactly this: the pieces were each tested
    and the combination was not."""

    @pytest.fixture
    def parts(self, rng):
        import pygeoinf2 as gi

        model = EuclideanSpace(24)
        first_data = EuclideanSpace(10)
        second_data = EuclideanSpace(6)

        first = LinearOperator.from_matrix(
            model, first_data, rng.standard_normal((10, 24)), form="components"
        )
        second = LinearOperator.from_matrix(
            model, second_data, rng.standard_normal((6, 24)), form="components"
        )
        # Deliberately different noise levels: a joint inversion that ignored
        # the weighting would still look plausible on equal ones.
        problems = [
            LinearForwardProblem(
                first, error=gi.GaussianMeasure.from_standard_deviation(first_data, 0.05)
            ),
            LinearForwardProblem(
                second,
                error=gi.GaussianMeasure.from_standard_deviation(second_data, 0.50),
            ),
        ]
        prior = gi.GaussianMeasure.from_standard_deviation(model, 1.0)
        truth = prior.sample(rng=rng)
        data = (first(truth), second(truth))
        return problems, prior, truth, data

    @staticmethod
    def dense_reference(problems, prior, data):
        """``(Q^-1 + A* R^-1 A)^-1 A* R^-1 d``, assembled in components."""
        forward = np.vstack(
            [
                problem.forward_operator.matrix(form="components")
                for problem in problems
            ]
        )
        noise = np.diag(
            np.concatenate(
                [
                    np.full(
                        problem.data_space.dim,
                        problem.error_measure.covariance.eigenvalues[0],
                    )
                    for problem in problems
                ]
            )
        )
        covariance = np.diag(prior.covariance.eigenvalues)
        stacked = np.concatenate([np.asarray(part) for part in data])
        gain = covariance @ forward.T @ np.linalg.inv(
            forward @ covariance @ forward.T + noise
        )
        return gain @ stacked

    @pytest.mark.parametrize("formalism", ["data_space", "model_space"])
    def test_both_formalisms_match_a_dense_reference(self, parts, formalism):
        problems, prior, truth, data = parts
        joint = LinearForwardProblem.from_direct_sum(problems)
        inversion = LinearGaussianInversion(joint, prior, formalism=formalism)

        estimate = inversion(joint.data_space.from_components(
            np.concatenate([np.asarray(part) for part in data])
        ))
        reference = self.dense_reference(problems, prior, data)
        assert np.asarray(estimate.expectation) == pytest.approx(reference, rel=1e-6)

    def test_a_blocked_preconditioner_gets_the_same_answer(self, parts, rng):
        """On the direct-sum data space, where the blocks are the two
        instruments."""
        from pygeoinf2.inference.preconditioners import NormalDiagonalPreconditioner
        from pygeoinf2.numerics.solvers import CGSolver

        problems, prior, truth, data = parts
        joint = LinearForwardProblem.from_direct_sum(problems)
        stacked = joint.data_space.from_components(
            np.concatenate([np.asarray(part) for part in data])
        )

        plain = LinearGaussianInversion(joint, prior)(stacked)
        preconditioned = LinearGaussianInversion(
            joint,
            prior,
            solver=CGSolver(rtol=1e-12).with_preconditioner(
                NormalDiagonalPreconditioner()
            ),
        )(stacked)

        assert np.asarray(preconditioned.expectation) == pytest.approx(
            np.asarray(plain.expectation), rel=1e-6
        )

    def test_the_evidence_and_a_push_forward_come_through(self, parts):
        problems, prior, truth, data = parts
        joint = LinearForwardProblem.from_direct_sum(problems)
        inversion = LinearGaussianInversion(joint, prior)
        stacked = joint.data_space.from_components(
            np.concatenate([np.asarray(part) for part in data])
        )

        assert np.isfinite(inversion.log_evidence(stacked))

        summary = LinearOperator.from_matrix(
            joint.model_space,
            EuclideanSpace(2),
            np.ones((2, 24)) / 24.0,
            form="components",
        )
        pushed = inversion.push_forward(summary)(stacked)
        assert np.asarray(pushed.expectation).shape == (2,)


class TestConstrainedEstimatorsCanBeReduced:
    """Both used to raise. The constraint is pulled back with the problem:
    ``B u == w`` becomes ``(B M) p == w``, which is v1's construction."""

    @pytest.fixture
    def setting(self, rng):
        import pygeoinf2 as gi
        from pygeoinf2.geometry.subspaces import AffineSubspace

        model = EuclideanSpace(20)
        data_space = EuclideanSpace(8)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.standard_normal((8, 20)), form="components"
        )
        constraint = LinearOperator.from_matrix(
            model, EuclideanSpace(2), rng.standard_normal((2, 20)), form="components"
        )
        values = np.array([0.3, -0.2])
        subspace = AffineSubspace.from_linear_equation(constraint, values)
        problem = LinearForwardProblem(
            forward,
            error=gi.GaussianMeasure.from_standard_deviation(data_space, 0.05),
        )
        parameterisation = LinearOperator.from_matrix(
            EuclideanSpace(6), model, rng.standard_normal((20, 6)), form="components"
        )
        data = forward(model.random(rng=rng))
        return problem, subspace, constraint, values, parameterisation, data

    def test_the_constraint_still_holds_in_the_parameter_space(self, setting):
        from pygeoinf2.inference.point import ConstrainedLeastSquares

        problem, subspace, constraint, values, parameterisation, data = setting
        estimator = ConstrainedLeastSquares(problem, subspace, damping=1e-3)

        reduced = estimator.parameterised(parameterisation)
        parameters = reduced(data)
        assert (constraint @ parameterisation)(parameters) == pytest.approx(
            values, abs=1e-8
        )

    def test_the_minimum_norm_route_too(self, setting, rng):
        """With data the reduced problem can actually fit -- generated from a
        model inside the parameterisation's range, since a discrepancy search
        on data outside it has no root to find and says so."""
        from pygeoinf2.inference.point import ConstrainedMinimumNorm

        problem, subspace, constraint, values, _, _ = setting
        # Enough parameters to fit: 8 data and 2 constraints, so 6 would leave
        # only 4 free and the discrepancy target would be unreachable -- which
        # is a fact about the reduced problem, not about the pull-back.
        parameterisation = LinearOperator.from_matrix(
            EuclideanSpace(14),
            problem.model_space,
            rng.standard_normal((20, 14)),
            form="components",
        )
        reachable = parameterisation(parameterisation.domain.random(rng=rng))
        data = problem.forward_operator(subspace.project(reachable))

        method = ConstrainedMinimumNorm(problem, subspace)
        parameters = method.parameterised(parameterisation)(data)
        assert (constraint @ parameterisation)(parameters) == pytest.approx(
            values, abs=1e-6
        )

    def test_a_subspace_without_an_equation_is_still_refused(self, setting, rng):
        """A basis fixes the solution set but not which equation defines it,
        and inventing one would be a different equation."""
        from pygeoinf2.geometry.subspaces import AffineSubspace
        from pygeoinf2.inference.point import ConstrainedLeastSquares

        from pygeoinf2.geometry.subspaces import OrthogonalProjector

        problem, _, _, _, parameterisation, _ = setting
        model = problem.model_space
        basis = [model.basis_vector(index) for index in range(3)]
        from_basis = AffineSubspace(OrthogonalProjector.from_basis(model, basis))

        estimator = ConstrainedLeastSquares(problem, from_basis, damping=1e-3)
        with pytest.raises(NotImplementedError, match="built from a basis"):
            estimator.parameterised(parameterisation)

    def test_too_few_parameters_for_the_constraints_is_refused(self, setting, rng):
        from pygeoinf2.inference.point import ConstrainedLeastSquares

        problem, subspace, _, _, _, _ = setting
        tiny = LinearOperator.from_matrix(
            EuclideanSpace(1),
            problem.model_space,
            rng.standard_normal((20, 1)),
            form="components",
        )
        estimator = ConstrainedLeastSquares(problem, subspace, damping=1e-3)
        with pytest.raises(ValueError, match="cannot carry"):
            estimator.parameterised(tiny)

    def test_a_data_reduction_leaves_the_constraint_alone(self, setting, rng):
        """It lives in the model space, so there is nothing to pull back."""
        from pygeoinf2.inference.point import ConstrainedLeastSquares

        problem, subspace, constraint, values, _, data = setting
        estimator = ConstrainedLeastSquares(problem, subspace, damping=1e-3)

        # Combine the eight observations into four.
        reduction = LinearOperator.from_matrix(
            problem.data_space,
            EuclideanSpace(4),
            np.repeat(np.eye(4), 2, axis=1) / 2.0,
            form="components",
        )
        reduced = estimator.data_reduced(reduction)
        assert reduced.subspace is subspace


class TestFeasibilityIsAskable:
    """All three noisy routes reported an empty feasible set only by raising,
    which is right when a set was asked for and unhelpful when the question was
    whether one exists. v1 had ``test_data_compatibility``."""

    @pytest.fixture
    def setting(self, rng):
        model = EuclideanSpace(12)
        data_space = EuclideanSpace(5)
        target_space = EuclideanSpace(1)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.standard_normal((5, 12)), form="components"
        )
        target = LinearOperator.from_matrix(
            model, target_space, rng.standard_normal((1, 12)), form="components"
        )
        return model, forward, target, forward(model.random(rng=rng))

    def test_a_tight_prior_is_infeasible_and_a_roomy_one_is_not(self, setting):
        from pygeoinf2.inference import (
            BackusInference,
            DualFeasibleProperty,
            FeasibleProperty,
        )

        model, forward, target, data = setting
        exact = LinearForwardProblem(forward)
        noisy = LinearForwardProblem(forward, error=Ball(forward.codomain, radius=1e-6))

        assert not BackusInference(exact, target, Ball(model, radius=1e-3)).is_feasible(data)
        assert BackusInference(exact, target, Ball(model, radius=100.0)).is_feasible(data)

        for route in (FeasibleProperty, DualFeasibleProperty):
            assert not route(noisy, target, Ball(model, radius=1e-3)).is_feasible(data)
            assert route(noisy, target, Ball(model, radius=100.0)).is_feasible(data)

    def test_the_predicate_agrees_with_the_exception(self, setting):
        from pygeoinf2.inference import BackusInference

        model, forward, target, data = setting
        estimator = BackusInference(
            LinearForwardProblem(forward), target, Ball(model, radius=1e-3)
        )
        assert not estimator.is_feasible(data)
        with pytest.raises(ValueError, match="No model"):
            estimator(data)
