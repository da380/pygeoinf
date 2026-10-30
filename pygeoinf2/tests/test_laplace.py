"""Nonlinear inference: the MAP estimator and the Laplace approximation on it."""

import numpy as np
import pytest

import pygeoinf2 as gi
from pygeoinf2.algebra.operators import LinearOperator, Operator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.inference.estimators import MeasureEstimator, PointEstimator
from pygeoinf2.inference.gaussian import GaussianPosterior, LinearGaussianInversion
from pygeoinf2.inference.laplace import (
    LaplaceApproximation,
    LaplaceResult,
    MaximumAPosteriori,
)
from pygeoinf2.inference.minimization import Minimization, NonlinearLeastSquares
from pygeoinf2.inference.problem import ForwardProblem, LinearForwardProblem
from pygeoinf2.numerics.optimization import LBFGS, NewtonCG, OptimizationResult
from pygeoinf2.testing import check_measure

from .conftest import make_dense_metric_space


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
        estimator = MaximumAPosteriori(ForwardProblem(forward, error=error), prior)
        found = estimator.solve(data)

        assert found.converged
        assert found.minimizer == pytest.approx(exact.expectation, abs=1e-6)
        assert estimator(data) == pytest.approx(found.minimizer)

    def test_the_covariance_is_the_posterior_covariance(self, linear_setting, rng):
        forward, prior, error, data = linear_setting
        space = forward.domain
        exact = LinearGaussianInversion(
            LinearForwardProblem(forward, error=error), prior
        )(data)
        found = LaplaceApproximation.from_problem(
            ForwardProblem(forward, error=error), prior
        )(data)

        for _ in range(3):
            vector = space.random(rng=rng)
            assert found.covariance(vector) == pytest.approx(
                exact.covariance(vector), abs=1e-10
            )

    def test_the_derivative_is_the_linear_estimator_on_a_dense_metric(self, rng):
        """``(P + J* R^-1 J)^-1 J* R^-1`` at the mode is the Kalman gain of the
        linear problem, so the MAP estimator's derivative with respect to
        the data is the linear posterior's expectation operator."""
        model = make_dense_metric_space(5)
        data_space = EuclideanSpace(8)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.standard_normal((8, 5)), form="components"
        )
        prior = gi.GaussianMeasure.from_standard_deviation(model, 1.5)
        error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.4)
        data = data_space.random(rng=rng)
        linear = LinearGaussianInversion(
            LinearForwardProblem(forward, error=error), prior
        )
        estimator = MaximumAPosteriori(
            ForwardProblem(forward, error=error), prior, optimizer=LBFGS(gtol=1e-10)
        )
        assert estimator.has_derivative

        derivative = estimator.derivative(data)
        gain = linear.expectation_operator.linear_part
        perturbation = data_space.random(rng=rng)
        assert derivative(perturbation) == pytest.approx(gain(perturbation), abs=1e-7)
        direction = model.random(rng=rng)
        assert derivative.adjoint(direction) == pytest.approx(
            gain.adjoint(direction), abs=1e-7
        )


class TestOnANonlinearProblem:
    def test_it_finds_a_stationary_point(self, nonlinear_setting):
        """Which is what a mode is. Checked on the objective's own gradient,
        not on the optimizer's report of itself."""
        forward, prior, error, truth, data = nonlinear_setting
        estimator = MaximumAPosteriori(ForwardProblem(forward, error=error), prior)
        found = estimator.solve(data)

        assert found.converged
        objective = estimator.objective(data)
        gradient = objective.gradient(found.minimizer)
        at_start = objective.gradient(forward.domain.zero())
        # Judged against the gradient where the search began, as the
        # optimizer judges itself: the objective's units are the data's.
        assert forward.domain.norm(gradient) <= 1e-7 * forward.domain.norm(at_start)

    def test_it_is_a_point_estimator(self, nonlinear_setting):
        forward, prior, error, truth, data = nonlinear_setting
        estimator = MaximumAPosteriori(ForwardProblem(forward, error=error), prior)
        assert isinstance(estimator, PointEstimator)
        assert isinstance(estimator, Minimization)
        assert isinstance(estimator, Operator)
        assert estimator.data_space == forward.codomain
        assert estimator.property_space == forward.domain
        assert estimator.starting_point(data) == pytest.approx(prior.expectation)

    def test_it_recovers_the_truth(self, nonlinear_setting):
        forward, prior, error, truth, data = nonlinear_setting
        found = MaximumAPosteriori(ForwardProblem(forward, error=error), prior)(data)
        space = forward.domain
        assert space.norm(space.subtract(found, truth)) < 0.2 * space.norm(truth)

    def test_the_covariance_is_positive_definite(self, nonlinear_setting):
        """A curvature that is not would mean the point found is not a
        minimum, and the Gaussian on it would be meaningless."""
        forward, prior, error, truth, data = nonlinear_setting
        found = LaplaceApproximation.from_problem(
            ForwardProblem(forward, error=error), prior
        )(data)

        matrix = found.covariance.matrix(form="components")
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

    def test_the_derivative_with_respect_to_the_data(self, nonlinear_setting, rng):
        """The implicit-function derivative at the mode against a central
        difference of the estimate itself. The Hessian is the Gauss-Newton
        one here, so the agreement is to the size of the dropped curvature
        term, which the tight noise keeps small."""
        forward, prior, error, truth, data = nonlinear_setting
        space, data_space = forward.domain, forward.codomain
        estimator = MaximumAPosteriori(
            ForwardProblem(forward, error=error), prior, optimizer=LBFGS(gtol=1e-10)
        )
        linearized = estimator.at(data)
        direction = data_space.scale(0.1, data_space.random(rng=rng))
        step = 1e-3
        ahead = estimator(data_space.axpy(step, direction, data_space.copy(data)))
        behind = estimator(data_space.axpy(-step, direction, data_space.copy(data)))
        differences = space.scale(1.0 / (2.0 * step), space.subtract(ahead, behind))
        predicted = linearized.derivative(direction)
        assert space.norm(space.subtract(predicted, differences)) < 2e-2 * space.norm(
            differences
        )

    def test_the_objective_evaluates_the_forward_once_at_a_point(self, rng):
        """``objective.at(m)`` runs the forward once, not once for the value
        and again for the derivative. With a forward that takes minutes, the
        double evaluation was the whole cost of a line search."""
        model = EuclideanSpace(4)
        data_space = EuclideanSpace(12)
        matrix = rng.standard_normal((12, 4))
        counts = {"value": 0}

        def value(components):
            counts["value"] += 1
            return np.exp(matrix @ components)

        def derivative(components):
            return LinearOperator.from_matrix(
                model,
                data_space,
                np.diag(np.exp(matrix @ components)) @ matrix,
                form="components",
            )

        forward = Operator.from_callables(
            model, data_space, value, derivative=derivative
        )
        prior = gi.GaussianMeasure.from_standard_deviation(model, 0.4)
        error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.05)
        data = data_space.random(rng=rng)
        objective = MaximumAPosteriori(
            ForwardProblem(forward, error=error), prior
        ).objective(data)

        point = model.random(rng=rng)
        counts["value"] = 0
        quadratic = objective.at(point)
        quadratic.gradient  # noqa: B018 - force the derivative's use
        assert counts["value"] == 1
        counts["value"] = 0
        objective(point)
        assert counts["value"] == 1

    def test_the_gaussian_can_be_placed_anywhere(self, nonlinear_setting, rng):
        """``at`` exists for when the mode is already known -- from a previous
        run, or to see what the approximation looks like elsewhere."""
        forward, prior, error, truth, data = nonlinear_setting
        laplace = LaplaceApproximation.from_problem(
            ForwardProblem(forward, error=error), prior
        )
        found = laplace(data)

        elsewhere = laplace.at(prior.expectation, data)
        assert elsewhere.expectation == pytest.approx(prior.expectation)
        # A different point, so a different curvature.
        vector = forward.domain.random(rng=rng)
        assert not np.allclose(elsewhere.covariance(vector), found.covariance(vector))


class TestTheLaplaceApproximation:
    """A measure estimator wrapping the point estimator and a solver."""

    def test_it_wraps_a_map_estimator(self, nonlinear_setting):
        forward, prior, error, truth, data = nonlinear_setting
        estimator = MaximumAPosteriori(ForwardProblem(forward, error=error), prior)
        laplace = LaplaceApproximation(estimator)
        assert isinstance(laplace, MeasureEstimator)
        assert laplace.estimator is estimator
        assert laplace.data_space == forward.codomain
        assert laplace.property_space == forward.domain

    def test_from_problem_builds_the_same_thing(self, nonlinear_setting, rng):
        forward, prior, error, truth, data = nonlinear_setting
        problem = ForwardProblem(forward, error=error)
        direct = LaplaceApproximation(MaximumAPosteriori(problem, prior))(data)
        built = LaplaceApproximation.from_problem(problem, prior)(data)
        assert built.expectation == pytest.approx(direct.expectation)
        vector = forward.domain.random(rng=rng)
        assert built.covariance(vector) == pytest.approx(direct.covariance(vector))

    def test_it_refuses_anything_else(self, nonlinear_setting):
        with pytest.raises(TypeError, match="MaximumAPosteriori"):
            LaplaceApproximation(LBFGS())

    def test_the_record_is_one_call_away(self, nonlinear_setting):
        forward, prior, error, truth, data = nonlinear_setting
        estimator = MaximumAPosteriori(ForwardProblem(forward, error=error), prior)
        result = LaplaceApproximation(estimator).solve(data)
        assert isinstance(result, LaplaceResult)
        assert result.converged
        assert isinstance(result.optimization, OptimizationResult)
        assert result.model == pytest.approx(result.optimization.minimizer)
        assert isinstance(result.measure, GaussianPosterior)
        assert result.measure.expectation == pytest.approx(result.model)
        assert result.model == pytest.approx(estimator(data))

    def test_the_gaussian_is_a_posterior_relative_to_the_prior(self, nonlinear_setting):
        forward, prior, error, truth, data = nonlinear_setting
        problem = ForwardProblem(forward, error=error)
        measure = LaplaceApproximation.from_problem(problem, prior)(data)
        assert isinstance(measure, GaussianPosterior)
        assert measure.reference is prior
        assert np.isfinite(measure.log_normalizer.value)

    def test_a_start_can_be_given(self, nonlinear_setting, rng):
        forward, prior, error, truth, data = nonlinear_setting
        laplace = LaplaceApproximation.from_problem(
            ForwardProblem(forward, error=error), prior
        )
        from_prior = laplace.solve(data)
        elsewhere = laplace.solve(
            data, start=forward.domain.scale(0.3, forward.domain.random(rng=rng))
        )
        assert elsewhere.converged
        assert elsewhere.model == pytest.approx(from_prior.model, abs=1e-4)

    def test_push_forward_conjugates(self, nonlinear_setting, rng):
        """The property's Gaussian has mean ``T m`` and covariance ``T C T*``."""
        forward, prior, error, truth, data = nonlinear_setting
        space = forward.domain
        target = EuclideanSpace(2)
        operator = LinearOperator.from_matrix(
            space, target, rng.standard_normal((2, 4)), form="components"
        )
        laplace = LaplaceApproximation.from_problem(
            ForwardProblem(forward, error=error), prior
        )
        pushed = laplace.push_forward(operator)
        assert isinstance(pushed, MeasureEstimator)
        assert pushed.data_space == forward.codomain
        assert pushed.property_space == target

        on_model = laplace(data)
        on_property = pushed(data)
        assert on_property.expectation == pytest.approx(operator(on_model.expectation))
        vector = target.random(rng=rng)
        assert on_property.covariance(vector) == pytest.approx(
            operator(on_model.covariance(operator.adjoint(vector)))
        )


class TestWithTheExactHessian:
    """A forward operator that supplies its second derivative gets the exact
    Hessian in the objective, and a Newton optimizer can use it."""

    @pytest.fixture
    def curved(self, nonlinear_setting, rng):
        forward, prior, error, truth, data = nonlinear_setting
        model, data_space = forward.domain, forward.codomain
        matrix = None
        # Recover the matrix from the derivative at zero: exp(0) == 1.
        matrix = forward.derivative(model.zero()).matrix(form="components")

        def value(components):
            return np.exp(matrix @ components)

        def derivative(components):
            return LinearOperator.from_matrix(
                model,
                data_space,
                np.diag(np.exp(matrix @ components)) @ matrix,
                form="components",
            )

        def second_derivative(components, direction):
            # F''(m)[dm, .] == diag(exp(B m) (B dm)) B
            return LinearOperator.from_matrix(
                model,
                data_space,
                np.diag(np.exp(matrix @ components) * (matrix @ direction)) @ matrix,
                form="components",
            )

        operator = Operator.from_callables(
            model,
            data_space,
            value,
            derivative=derivative,
            second_derivative=second_derivative,
        )
        return operator, prior, error, truth, data

    def test_the_hessian_is_the_derivative_of_the_gradient(self, curved, rng):
        operator, prior, error, truth, data = curved
        objective = MaximumAPosteriori(
            ForwardProblem(operator, error=error), prior
        ).objective(data)
        assert objective.has_hessian
        space = operator.domain
        point = space.random(rng=rng)
        direction = space.random(rng=rng)
        action = objective.hessian(point)(direction)
        step = 1.0e-5
        differences = space.scale(
            1.0 / (2.0 * step),
            space.subtract(
                objective.gradient(space.add(point, space.scale(step, direction))),
                objective.gradient(space.add(point, space.scale(-step, direction))),
            ),
        )
        assert np.allclose(action, differences, rtol=1e-5, atol=1e-8)

    def test_the_estimator_uses_the_exact_hessian_for_its_derivative(self, curved, rng):
        operator, prior, error, truth, data = curved
        estimator = MaximumAPosteriori(ForwardProblem(operator, error=error), prior)
        mode = estimator(data)
        exact = estimator.objective(data).hessian(mode)
        direction = operator.domain.random(rng=rng)
        assert estimator.hessian(data, mode)(direction) == pytest.approx(
            exact(direction)
        )

    def test_newton_finds_the_mode_in_fewer_iterations(self, curved):
        operator, prior, error, truth, data = curved
        problem = ForwardProblem(operator, error=error)
        by_lbfgs = MaximumAPosteriori(
            problem, prior, optimizer=LBFGS(max_iterations=500)
        ).solve(data)
        by_newton = MaximumAPosteriori(
            problem, prior, optimizer=NewtonCG(max_iterations=50)
        ).solve(data)
        assert by_newton.converged
        space = operator.domain
        assert space.norm(
            space.subtract(by_newton.minimizer, by_lbfgs.minimizer)
        ) < 1e-4 * max(space.norm(by_lbfgs.minimizer), 1.0)
        assert by_newton.iterations < by_lbfgs.iterations

    def test_without_a_second_derivative_there_is_no_hessian(self, nonlinear_setting):
        forward, prior, error, truth, data = nonlinear_setting
        objective = MaximumAPosteriori(
            ForwardProblem(forward, error=error), prior
        ).objective(data)
        assert not objective.has_hessian


class TestTheCurvatureFollowsTheEstimator:
    """The Laplace Gaussian's precision is half the Hessian the MAP objective
    carries: exact by default when the forward operator supplies its second
    derivative, Gauss-Newton when asked, as for the MAP's own derivative.
    The forward is ``A m + c (A m)^2``, with an exact second derivative, on
    a model space with a dense Gram matrix."""

    @pytest.fixture
    def quadratic(self):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        from .conftest import spd_matrix

        rng = np.random.default_rng(3)
        model = make_dense_metric_space(3)
        data_space = EuclideanSpace(10)
        matrix = LinearOperator.from_matrix(
            model, data_space, rng.normal(size=(10, 3)), form="components"
        )
        bend = 0.8
        forward = Operator.from_callables(
            model,
            data_space,
            lambda m: matrix(m) + bend * matrix(m) ** 2,
            derivative=lambda m: (
                DiagonalLinearOperator(data_space, 1.0 + 2.0 * bend * matrix(m))
                @ matrix
            ),
            second_derivative=lambda m, dm: (
                DiagonalLinearOperator(data_space, 2.0 * bend * matrix(dm)) @ matrix
            ),
        )
        error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.3)
        prior = gi.GaussianMeasure.from_covariance_matrix(
            model, 0.3 * spd_matrix(rng, 3), form="galerkin"
        )
        problem = ForwardProblem(forward, error=error)
        data = problem.synthetic_data(prior.sample(rng=rng), rng=rng)
        return problem, prior, data

    @staticmethod
    def inverses(problem, prior, data, mode):
        """The inverses of half the exact Hessian and of the Gauss-Newton
        normal operator at the mode, as component matrices."""
        exact = MaximumAPosteriori(problem, prior).objective(data).hessian(mode)
        normal = MaximumAPosteriori(problem, prior).linearized_normal(mode)
        return (
            np.linalg.inv(0.5 * exact.matrix(form="components")),
            np.linalg.inv(normal.matrix(form="components")),
        )

    @pytest.mark.parametrize("gauss_newton", [False, True])
    def test_the_covariance_inverts_the_estimators_hessian(
        self, quadratic, gauss_newton
    ):
        problem, prior, data = quadratic
        laplace = LaplaceApproximation.from_problem(
            problem,
            prior,
            optimizer=NewtonCG(max_iterations=100, preconditioner=prior.covariance),
            solver=gi.CholeskySolver(),
            gauss_newton=gauss_newton,
        )
        assert laplace.estimator._gauss_newton is gauss_newton
        result = laplace.solve(data)
        assert result.converged
        exact, normal = self.inverses(problem, prior, data, result.model)
        expected = normal if gauss_newton else exact
        other = exact if gauss_newton else normal
        covariance = result.measure.covariance.matrix(form="components")
        assert np.linalg.norm(covariance - expected) < 1e-10 * np.linalg.norm(expected)
        # The two curvatures differ here, so the test can tell them apart.
        assert np.linalg.norm(covariance - other) > 1e-2 * np.linalg.norm(other)

    @pytest.mark.parametrize("gauss_newton", [False, True])
    def test_the_low_rank_route_takes_the_same_curvature(self, quadratic, gauss_newton):
        problem, prior, data = quadratic
        mode = MaximumAPosteriori(
            problem,
            prior,
            optimizer=NewtonCG(max_iterations=100, preconditioner=prior.covariance),
        )(data)
        low = LaplaceApproximation.from_problem(
            problem,
            prior,
            gauss_newton=gauss_newton,
            rank=3,
            rng=np.random.default_rng(0),
        ).at(mode, data)
        exact, normal = self.inverses(problem, prior, data, mode)
        expected = normal if gauss_newton else exact
        covariance = low.covariance.matrix(form="components")
        assert np.linalg.norm(covariance - expected) < 1e-8 * np.linalg.norm(expected)

    @pytest.mark.parametrize("gauss_newton", [False, True])
    def test_the_evidence_is_the_laplace_estimate(self, quadratic, gauss_newton):
        """``-J(m) - log det(Q K) / 2``, with ``K`` the curvature, against a
        dense reference; the full and the low-rank routes agree on it."""
        problem, prior, data = quadratic
        laplace = LaplaceApproximation.from_problem(
            problem,
            prior,
            optimizer=NewtonCG(max_iterations=100, preconditioner=prior.covariance),
            solver=gi.CholeskySolver(),
            gauss_newton=gauss_newton,
        )
        result = laplace.solve(data)
        mode, measure = result.model, result.measure
        space = problem.model_space
        offset = space.subtract(mode, prior.expectation)
        penalized = problem.potential(data)(mode) + 0.5 * space.inner_product(
            prior.precision(offset), offset
        )
        exact, normal = self.inverses(problem, prior, data, mode)
        curvature = np.linalg.inv(normal if gauss_newton else exact)
        volume = np.linalg.slogdet(
            prior.covariance.matrix(form="components") @ curvature
        )[1]
        expected = -penalized - 0.5 * volume
        assert measure.log_normalizer.value == pytest.approx(expected, rel=1e-10)
        low = LaplaceApproximation.from_problem(
            problem,
            prior,
            gauss_newton=gauss_newton,
            rank=3,
            rng=np.random.default_rng(0),
        ).at(mode, data)
        assert low.log_normalizer.value == pytest.approx(expected, rel=1e-8)

    def test_the_gauss_newton_gaussian_samples_and_the_exact_one_does_not(
        self, quadratic
    ):
        problem, prior, data = quadratic
        built = {
            gauss_newton: LaplaceApproximation.from_problem(
                problem,
                prior,
                optimizer=NewtonCG(max_iterations=100, preconditioner=prior.covariance),
                gauss_newton=gauss_newton,
            )(data)
            for gauss_newton in (False, True)
        }
        assert not built[False].can_sample
        assert built[True].can_sample


class TestTheFullGaussianSamplesAndCarriesTheEvidence:
    """The full Laplace Gaussian draws by randomize-then-optimize and carries
    the Laplace evidence, as the low-rank one does; on a linear problem both
    are the linear posterior's."""

    def test_on_a_linear_problem_the_evidence_is_the_posteriors(self, linear_setting):
        forward, prior, error, data = linear_setting
        exact = LinearGaussianInversion(
            LinearForwardProblem(forward, error=error), prior
        )(data)
        found = LaplaceApproximation.from_problem(
            ForwardProblem(forward, error=error), prior, solver=gi.CholeskySolver()
        )(data)
        assert found.log_normalizer.value == pytest.approx(
            exact.log_normalizer.value, rel=1e-8
        )

    def test_the_draws_have_the_laplace_covariance(self, nonlinear_setting, rng):
        forward, prior, error, truth, data = nonlinear_setting
        measure = LaplaceApproximation.from_problem(
            ForwardProblem(forward, error=error), prior, solver=gi.CholeskySolver()
        )(data)
        assert measure.can_sample
        check_measure(measure, rng=rng)


class TestWithTheGaussNewtonHessian:
    """Asking for the Gauss-Newton Hessian equips the objective with it on a
    forward operator that supplies only its derivative."""

    def test_newton_runs_on_a_derivative_only_operator(self, nonlinear_setting):
        forward, prior, error, truth, data = nonlinear_setting
        problem = ForwardProblem(forward, error=error)
        estimator = MaximumAPosteriori(
            problem, prior, optimizer=NewtonCG(max_iterations=50), gauss_newton=True
        )
        assert estimator.objective(data).has_hessian
        by_newton = estimator.solve(data)
        by_lbfgs = MaximumAPosteriori(
            problem, prior, optimizer=LBFGS(max_iterations=500)
        ).solve(data)
        assert by_newton.converged
        space = forward.domain
        assert space.norm(
            space.subtract(by_newton.minimizer, by_lbfgs.minimizer)
        ) < 1e-4 * max(space.norm(by_lbfgs.minimizer), 1.0)

    def test_it_is_the_prior_term_plus_the_normal_operator(
        self, nonlinear_setting, rng
    ):
        forward, prior, error, truth, data = nonlinear_setting
        estimator = MaximumAPosteriori(
            ForwardProblem(forward, error=error), prior, gauss_newton=True
        )
        space = forward.domain
        point, direction = space.random(rng=rng), space.random(rng=rng)
        jacobian = forward.derivative(point)
        expected = space.add(
            space.scale(2.0, prior.precision(direction)),
            space.scale(2.0, jacobian.adjoint(error.precision(jacobian(direction)))),
        )
        assert np.allclose(
            estimator.objective(data).hessian(point)(direction), expected
        )
        assert np.allclose(estimator.hessian(data, point)(direction), expected)

    def test_least_squares_takes_the_same_choice(self, nonlinear_setting):
        forward, prior, error, truth, data = nonlinear_setting
        problem = ForwardProblem(forward, error=error)
        assert (
            not NonlinearLeastSquares(problem, damping=1.0).objective(data).has_hessian
        )
        estimator = NonlinearLeastSquares(
            problem,
            damping=1.0,
            optimizer=NewtonCG(max_iterations=50),
            gauss_newton=True,
        )
        assert estimator.objective(data).has_hessian
        assert estimator.solve(data).converged

    def test_occam_takes_the_same_choice(self, nonlinear_setting):
        from pygeoinf2.inference.minimization import Occam

        forward, prior, error, truth, data = nonlinear_setting
        problem = ForwardProblem(forward, error=error)
        assert not Occam(problem).misfit_constraint(data).has_hessian
        assert Occam(problem, gauss_newton=True).misfit_constraint(data).has_hessian


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
    """This is built so that a function-space MCMC layer can plug in.
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
        mode = estimator(data)
        density = estimator.log_posterior(data)

        at_mode = density(mode)
        for _ in range(5):
            nearby = forward.domain.axpy(
                0.05, forward.domain.random(rng=rng), forward.domain.copy(mode)
            )
            assert density(nearby) <= at_mode


def tutorial_nine_problem():
    """Tutorial 9's problem: the log-concentration around a ring, averaged
    over forty sensor footprints, an order-one Sobolev prior on ``H^2``."""
    from pygeoinf2 import Linearization
    from pygeoinf2.symmetric_space.circle import Sobolev

    rng = np.random.default_rng(9)
    space = Sobolev(256, 2.0, 0.2)
    sensors = np.sort(rng.uniform(0.0, 2.0 * np.pi, 40))
    averaging = space.geodesic_ball_average_operator(
        [np.array([s]) for s in sensors], 0.05
    )

    def value(m):
        return averaging(np.exp(np.minimum(m, 20.0)))

    def linearize(m):
        c = np.exp(np.minimum(m, 20.0))
        return Linearization(
            m, averaging(c), averaging @ space.multiplication_operator(c)
        )

    forward = Operator.from_callables(
        space, averaging.codomain, value, linearize=linearize
    )
    prior = space.sobolev_measure(1.0, 0.2, pointwise_std=0.8)
    truth = prior.sample(rng=rng)
    noise = gi.GaussianMeasure.from_standard_deviation(averaging.codomain, 0.05)
    problem = ForwardProblem(forward, error=noise)
    data = forward(truth) + noise.sample(rng=rng)
    return space, problem, prior, data


def ray_problem():
    """A small instance of tutorial 13's problem: travel times along four
    bent rays through a field on the square, under a Sobolev prior."""
    from pygeoinf2.symmetric_space.plane import Sobolev

    rng = np.random.default_rng(13)
    space = Sobolev((16, 16), 2.0, 0.25, bounds=[(0.0, 2.0), (0.0, 2.0)])
    prior = space.sobolev_measure(2.0, 0.3, pointwise_std=0.25)
    truth = prior.sample(rng=rng)
    sources = [np.array([0.2, y]) for y in np.linspace(0.5, 1.5, 2)]
    receivers = [np.array([1.8, y]) for y in np.linspace(0.5, 1.5, 2)]
    paths = [(s, r) for s in sources for r in receivers]
    forward = space.ray_travel_time_operator(paths, 1.0, count=8)
    times = forward(truth)
    noise = gi.GaussianMeasure.from_standard_deviation(
        forward.codomain, 0.005 * float(np.mean(times))
    )
    data = forward.codomain.add(times, noise.sample(rng=rng))
    return space, ForwardProblem(forward, error=noise), prior, data


class TestAgainstTheRecordedAnswers:
    """The mode and the Gaussian on two problems, against recorded numbers:
    a regression check on the answer, to the optimizer's tolerance. Not to
    rounding, and not in the iteration count: a search takes a different
    number of steps on another platform's BLAS, and the recorded numbers
    are a check on the answer, not on the path to it. The Gaussian is the
    Gauss-Newton one in both, the curvature a forward operator without a
    second derivative has anyway."""

    def test_tutorial_nine(self):
        space, problem, prior, data = tutorial_nine_problem()
        rng = np.random.default_rng(1)
        probes = [prior.sample(rng=rng) for _ in range(2)]
        estimator = MaximumAPosteriori(
            problem,
            prior,
            gauss_newton=True,
            optimizer=NewtonCG(preconditioner=prior.covariance),
        )
        result = estimator.solve(data)
        assert result.converged
        assert result.value == pytest.approx(31.961649761807358, rel=1e-6)
        assert space.norm(result.minimizer) == pytest.approx(
            2.260978466139511, rel=1e-5
        )
        assert problem.chi_squared(result.minimizer, data) == pytest.approx(
            17.298941511833846, rel=1e-5
        )
        measure = LaplaceApproximation(estimator).at(result.minimizer, data)
        for probe, quadratic, mean in zip(
            probes,
            (0.22121036118396292, 0.3975650083516245),
            (0.31839897217461366, 0.21824242523329418),
        ):
            assert space.inner_product(measure.covariance(probe), probe) == (
                pytest.approx(quadratic, rel=1e-6)
            )
            assert space.inner_product(result.minimizer, probe) == pytest.approx(
                mean, rel=1e-5
            )

    def test_the_ray_problem(self):
        """The exact Hessian drives the search, the forward operator
        supplying its second derivative, and the prior's covariance
        preconditions it; the Gaussian is placed on the mode found."""
        space, problem, prior, data = ray_problem()
        rng = np.random.default_rng(2)
        probes = [space.random(rng=rng) for _ in range(2)]
        estimator = MaximumAPosteriori(
            problem,
            prior,
            optimizer=NewtonCG(
                preconditioner=prior.covariance, max_iterations=15, max_cg_iterations=40
            ),
        )
        result = estimator.solve(data)
        assert result.converged
        assert result.value == pytest.approx(1.3629738932651345, rel=1e-6)
        assert space.norm(result.minimizer) == pytest.approx(
            0.4366038043367584, rel=1e-5
        )
        assert problem.chi_squared(result.minimizer, data) == pytest.approx(
            0.0020768703102813906, rel=1e-6
        )
        gauss_newton = MaximumAPosteriori(problem, prior, gauss_newton=True)
        measure = LaplaceApproximation(gauss_newton).at(result.minimizer, data)
        for probe, quadratic, mean in zip(
            probes,
            (22.0260801461811, 22.69915458174353),
            (0.36827123855287286, 0.6138104952900771),
        ):
            assert space.inner_product(measure.covariance(probe), probe) == (
                pytest.approx(quadratic, rel=1e-6)
            )
            assert space.inner_product(result.minimizer, probe) == pytest.approx(
                mean, rel=1e-5
            )
