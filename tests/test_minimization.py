"""Point estimators that minimize: the generic class and its special cases."""

import numpy as np
import pytest

import pygeoinf as gi
import pygeoinf.testing  # noqa: F401 - gi.testing below
from pygeoinf.algebra.operators import Functional, LinearOperator, Operator
from pygeoinf.algebra.spaces import EuclideanSpace
from pygeoinf.inference.estimators import PointEstimator
from pygeoinf.inference.laplace import MaximumAPosteriori
from pygeoinf.inference.minimization import (
    ConstrainedMinimization,
    Minimization,
    NonlinearLeastSquares,
    Occam,
)
from pygeoinf.inference.point import LeastSquares
from pygeoinf.inference.problem import ForwardProblem, LinearForwardProblem
from pygeoinf.numerics.optimization import (
    LBFGS,
    AugmentedLagrangianResult,
    NewtonCG,
    OptimizationResult,
)
from pygeoinf.traits import Traits

from .conftest import make_dense_metric_space

from pygeoinf.inference.point import MinimumNorm as _MinimumNorm


def damped_fit(space, data_space, forward, damping):
    """``d -> argmin_m |A m - d|^2 / 2 + t |m|^2 / 2`` as a generic minimization,
    with the Hessian ``A* A + t I`` and the mixed derivative ``-A*``."""

    def objective(data):
        def value(m):
            residual = data_space.subtract(forward(m), data)
            return 0.5 * data_space.squared_norm(residual) + 0.5 * damping * (
                space.squared_norm(m)
            )

        def gradient(m):
            residual = data_space.subtract(forward(m), data)
            return space.axpy(damping, m, forward.adjoint(residual))

        def hessian(m):
            return LinearOperator.self_adjoint(
                space,
                lambda v: space.axpy(damping, v, forward.adjoint(forward(v))),
                traits=Traits.POSITIVE_DEFINITE,
            )

        return Functional.from_callables(
            space, value, gradient=gradient, hessian=hessian
        )

    def mixed(data, model):
        return LinearOperator.from_callables(
            data_space,
            space,
            lambda dd: space.negative(forward.adjoint(dd)),
            adjoint=lambda dm: data_space.negative(forward(dm)),
        )

    return Minimization(
        data_space,
        space,
        objective,
        optimizer=LBFGS(gtol=1e-10),
        mixed_derivative=mixed,
    )


class TestMinimization:
    @pytest.fixture
    def setting(self, rng):
        space = make_dense_metric_space(5)
        data_space = EuclideanSpace(7)
        forward = LinearOperator.from_matrix(
            space, data_space, rng.standard_normal((7, 5)), form="components"
        )
        return space, data_space, forward, damped_fit(space, data_space, forward, 0.3)

    def test_it_is_a_point_estimator_and_an_operator(self, setting):
        space, data_space, forward, estimator = setting
        assert isinstance(estimator, PointEstimator)
        assert isinstance(estimator, Operator)
        assert estimator.data_space == data_space
        assert estimator.property_space == space
        assert estimator.model_space == space
        assert estimator.has_derivative

    def test_it_finds_the_damped_least_squares_model(self, setting, rng):
        space, data_space, forward, estimator = setting
        data = data_space.random(rng=rng)
        reference = LeastSquares(LinearForwardProblem(forward), damping=0.3)(data)
        assert estimator(data) == pytest.approx(reference, abs=1e-5)
        record = estimator.solve(data)
        assert isinstance(record, OptimizationResult)
        assert record.converged
        assert record.minimizer == pytest.approx(estimator(data))

    def test_the_start_is_the_zero_a_vector_or_a_callable(self, setting, rng):
        space, data_space, forward, estimator = setting
        data = data_space.random(rng=rng)
        assert estimator.starting_point(data) == pytest.approx(space.zero())
        fixed = space.random(rng=rng)
        assert Minimization(
            data_space, space, estimator.objective, start=fixed
        ).starting_point(data) == pytest.approx(fixed)
        assert Minimization(
            data_space, space, estimator.objective, start=forward.adjoint
        ).starting_point(data) == pytest.approx(forward.adjoint(data))
        assert estimator(data, start=fixed) == pytest.approx(estimator(data), abs=1e-6)

    def test_the_derivative_is_the_implicit_function_one(self, setting, rng):
        """``-H^-1 M`` is the linear estimator's own operator here, checked
        on a dense metric where a derivative and a gradient differ."""
        space, data_space, forward, estimator = setting
        data = data_space.random(rng=rng)
        reference = LeastSquares(LinearForwardProblem(forward), damping=0.3).operator
        derivative = estimator.derivative(data)
        perturbation = data_space.random(rng=rng)
        assert derivative(perturbation) == pytest.approx(
            reference(perturbation), abs=1e-8
        )
        direction = space.random(rng=rng)
        assert derivative.adjoint(direction) == pytest.approx(
            reference.adjoint(direction), abs=1e-8
        )
        linearized = estimator.at(data)
        assert linearized.value == pytest.approx(estimator(data), abs=1e-7)
        assert linearized.derivative(perturbation) == pytest.approx(
            reference(perturbation), abs=1e-8
        )

    def test_without_a_mixed_derivative_there_is_none(self, setting, rng):
        space, data_space, forward, estimator = setting
        bare = Minimization(data_space, space, estimator.objective)
        assert not bare.has_derivative
        with pytest.raises(NotImplementedError, match="mixed derivative"):
            bare.derivative(data_space.random(rng=rng))

    def test_without_a_hessian_there_is_none_either(self, setting, rng):
        space, data_space, forward, estimator = setting

        def objective(data):
            full = estimator.objective(data)
            return Functional.from_callables(space, full, gradient=full.gradient)

        bare = Minimization(
            data_space,
            space,
            objective,
            mixed_derivative=estimator.mixed_derivative,
        )
        assert bare.has_derivative
        with pytest.raises(NotImplementedError, match="Hessian"):
            bare.derivative(data_space.random(rng=rng))

    def test_push_forward_composes_the_property(self, setting, rng):
        space, data_space, forward, estimator = setting
        target = EuclideanSpace(2)
        operator = LinearOperator.from_matrix(
            space, target, rng.standard_normal((2, 5)), form="components"
        )
        pushed = estimator.push_forward(operator)
        assert isinstance(pushed, PointEstimator)
        assert pushed.data_space == data_space
        assert pushed.property_space == target
        data = data_space.random(rng=rng)
        assert pushed(data) == pytest.approx(operator(estimator(data)))
        perturbation = data_space.random(rng=rng)
        assert pushed.at(data).derivative(perturbation) == pytest.approx(
            operator(estimator.derivative(data)(perturbation)), abs=1e-8
        )
        further = pushed.push_forward(LinearOperator.identity(target) * 2.0)
        assert further(data) == pytest.approx(2.0 * operator(estimator(data)))

    def test_push_forward_is_the_public_composition(self, setting, rng):
        """The composed estimator every point estimator pushes forward to,
        with the inner estimator and the property operator reachable, a
        ``start=`` reaching the search, and the domain checked."""
        from pygeoinf.inference import ComposedPointEstimator

        space, data_space, forward, estimator = setting
        target = EuclideanSpace(2)
        operator = LinearOperator.from_matrix(
            space, target, rng.standard_normal((2, 5)), form="components"
        )
        pushed = estimator.push_forward(operator)
        assert isinstance(pushed, ComposedPointEstimator)
        assert pushed.estimator is estimator
        assert pushed.property_operator is operator
        data = data_space.random(rng=rng)
        start = space.random(rng=rng)
        assert pushed(data, start=start) == pytest.approx(
            operator(estimator(data, start=start))
        )
        with pytest.raises(ValueError, match="property space"):
            estimator.push_forward(LinearOperator.identity(target))


class TestNonlinearLeastSquares:
    @pytest.fixture
    def linear(self, rng):
        space = make_dense_metric_space(5)
        data_space = EuclideanSpace(8)
        forward = LinearOperator.from_matrix(
            space, data_space, rng.standard_normal((8, 5)), form="components"
        )
        error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.3)
        truth = space.random(rng=rng)
        data = data_space.add(forward(truth), error.sample(rng=rng))
        return space, data_space, forward, error, data

    def test_parity_with_least_squares(self, linear, rng):
        """On a linear problem it is the linear estimator at the same
        damping: the model, the derivative and its adjoint."""
        space, data_space, forward, error, data = linear
        reference = LeastSquares(
            LinearForwardProblem(forward, error=error), damping=0.7
        )
        estimator = NonlinearLeastSquares(
            ForwardProblem(forward, error=error),
            damping=0.7,
            optimizer=LBFGS(gtol=1e-10),
        )
        assert estimator.damping == 0.7
        assert estimator.forward_problem.forward_operator is forward
        assert estimator(data) == pytest.approx(reference(data), abs=1e-7)

        derivative = estimator.derivative(data)
        perturbation = data_space.random(rng=rng)
        assert derivative(perturbation) == pytest.approx(
            reference.operator(perturbation), abs=1e-8
        )
        direction = space.random(rng=rng)
        assert derivative.adjoint(direction) == pytest.approx(
            reference.operator.adjoint(direction), abs=1e-8
        )

    def test_parity_without_an_error_measure(self, linear, rng):
        space, data_space, forward, error, data = linear
        reference = LeastSquares(LinearForwardProblem(forward), damping=0.2)
        estimator = NonlinearLeastSquares(
            ForwardProblem(forward), damping=0.2, optimizer=LBFGS(gtol=1e-10)
        )
        assert estimator(data) == pytest.approx(reference(data), abs=1e-7)

    def test_the_damping_pulls_towards_the_reference(self, linear, rng):
        space, data_space, forward, error, data = linear
        reference_model = space.random(rng=rng)
        problem = ForwardProblem(forward, error=error)
        loose = NonlinearLeastSquares(
            problem, damping=0.1, reference=reference_model, optimizer=LBFGS(gtol=1e-10)
        )
        tight = NonlinearLeastSquares(
            problem, damping=1e4, reference=reference_model, optimizer=LBFGS(gtol=1e-10)
        )
        assert tight.reference is reference_model
        assert tight.starting_point(data) is reference_model
        assert space.norm(space.subtract(tight(data), reference_model)) < space.norm(
            space.subtract(loose(data), reference_model)
        )

    def test_the_objective_is_the_misfit_plus_the_damping(self, linear, rng):
        space, data_space, forward, error, data = linear
        problem = ForwardProblem(forward, error=error)
        estimator = NonlinearLeastSquares(problem, damping=0.5)
        objective = estimator.objective(data)
        model = space.random(rng=rng)
        assert objective(model) == pytest.approx(
            problem.chi_squared(model, data) + 0.5 * space.squared_norm(model)
        )
        assert objective(model) == pytest.approx(
            2.0 * problem.potential(data)(model) + 0.5 * space.squared_norm(model)
        )
        gi.testing.check_gradient(objective, model, rng=rng)

    def test_on_a_nonlinear_operator_it_reaches_a_stationary_point(self, rng):
        space = EuclideanSpace(3)
        data_space = EuclideanSpace(9)
        matrix = rng.standard_normal((9, 3))
        forward = Operator.from_callables(
            space,
            data_space,
            lambda c: np.exp(matrix @ c),
            derivative=lambda c: LinearOperator.from_matrix(
                space,
                data_space,
                np.diag(np.exp(matrix @ c)) @ matrix,
                form="components",
            ),
        )
        error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.1)
        truth = space.scale(0.3, space.random(rng=rng))
        data = data_space.add(forward(truth), error.sample(rng=rng))
        estimator = NonlinearLeastSquares(
            ForwardProblem(forward, error=error),
            damping=0.1,
            optimizer=LBFGS(gtol=1e-5),
        )
        result = estimator.solve(data)
        assert result.converged
        gradient = estimator.objective(data).gradient(result.minimizer)
        assert space.norm(gradient) < 1e-5
        assert estimator.has_derivative
        # Gauss-Newton Hessian without a second derivative: positive definite.
        hessian = estimator.hessian(data, result.minimizer)
        direction = space.random(rng=rng)
        assert space.inner_product(hessian(direction), direction) > 0.0

    def test_refusals(self, linear):
        space, data_space, forward, error, data = linear
        with pytest.raises(ValueError, match="non-negative"):
            NonlinearLeastSquares(ForwardProblem(forward), damping=-1.0)
        from pygeoinf.geometry.convex import Ball

        with pytest.raises(TypeError, match="measure"):
            NonlinearLeastSquares(
                ForwardProblem(forward, error=Ball(data_space, radius=1.0))
            )


class TestConstrainedMinimization:
    def test_a_quadratic_with_a_data_dependent_equality(self, rng):
        """Minimize ``|m - d|^2 / 2`` subject to ``(w, m) == (w, d) / 2``: the
        answer is the metric projection of ``d`` onto the plane."""
        space = make_dense_metric_space(5)
        w = space.random(rng=rng)

        def objective(data):
            return Functional.from_callables(
                space,
                lambda m: 0.5 * space.squared_norm(space.subtract(m, data)),
                gradient=lambda m: space.subtract(m, data),
            )

        def plane(data):
            level = 0.5 * space.inner_product(w, data)
            return Functional.from_callables(
                space,
                lambda m: space.inner_product(w, m) - level,
                gradient=lambda m: w,
            )

        estimator = ConstrainedMinimization(
            space,
            space,
            objective,
            equalities=[plane],
            optimizer=LBFGS(gtol=1e-10),
            tolerance=1e-8,
            gtol=1e-8,
        )
        assert isinstance(estimator, Minimization)
        assert not estimator.has_derivative
        data = space.random(rng=rng)
        expected = space.axpy(
            -0.5 * space.inner_product(w, data) / space.squared_norm(w),
            w,
            space.copy(data),
        )
        result = estimator.solve(data)
        assert isinstance(result, AugmentedLagrangianResult)
        assert result.converged
        assert result.minimizer == pytest.approx(expected, abs=1e-5)
        assert estimator(data) == pytest.approx(expected, abs=1e-5)
        assert result.multipliers[0] == pytest.approx(
            0.5 * space.inner_product(w, data) / space.squared_norm(w), abs=1e-5
        )
        assert len(estimator.equalities(data)) == 1
        assert estimator.inequalities(data) == ()
        assert estimator.method(data).equalities[0](expected) == pytest.approx(
            0.0, abs=1e-12
        )


class TestOccam:
    @pytest.fixture
    def linear(self, rng):
        space = make_dense_metric_space(5)
        data_space = EuclideanSpace(8)
        forward = LinearOperator.from_matrix(
            space, data_space, rng.standard_normal((8, 5)), form="components"
        )
        error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.3)
        truth = space.random(rng=rng)
        data = data_space.add(forward(truth), error.sample(rng=rng))
        return space, data_space, forward, error, data

    def test_parity_with_minimum_norm_on_a_linear_problem(self, linear):
        """The smallest model whose misfit meets the level: the discrepancy
        principle finds it by a damping search, this by a constrained
        minimization, and they must agree."""
        space, data_space, forward, error, data = linear
        linear_problem = LinearForwardProblem(forward, error=error)
        reference = _MinimumNorm(linear_problem, level=0.95)(data)
        estimator = Occam(
            ForwardProblem(forward, error=error),
            level=0.95,
            optimizer=LBFGS(gtol=1e-10),
            tolerance=1e-8,
            gtol=1e-8,
        )
        result = estimator.solve(data)
        assert result.converged
        assert space.norm(space.subtract(result.minimizer, reference)) < 1e-4 * (
            space.norm(reference)
        )
        assert linear_problem.chi_squared(result.minimizer, data) == pytest.approx(
            estimator.target, rel=1e-6
        )
        assert result.inequality_multipliers[0] > 0.0

    def test_a_reference_that_already_fits_is_the_answer(self, linear):
        space, data_space, forward, error, data = linear
        problem = ForwardProblem(forward, error=error)
        fitting = LeastSquares(
            LinearForwardProblem(forward, error=error), formalism="model_space"
        )(data)
        assert problem.chi_squared(fitting, data) < problem.critical_chi_squared()
        estimator = Occam(problem, reference=fitting)
        result = estimator.solve(data)
        assert result.converged
        assert result.minimizer == pytest.approx(fitting, abs=1e-8)
        assert result.inequality_multipliers[0] == 0.0

    def test_the_constraint_and_the_objective(self, linear, rng):
        space, data_space, forward, error, data = linear
        problem = ForwardProblem(forward, error=error)
        reference = space.random(rng=rng)
        estimator = Occam(problem, level=0.9, reference=reference)
        assert estimator.level == 0.9
        assert estimator.reference is reference
        assert estimator.forward_problem is problem
        assert estimator.target == pytest.approx(
            problem.critical_chi_squared(level=0.9)
        )
        model = space.random(rng=rng)
        assert estimator.objective(data)(model) == pytest.approx(
            space.squared_norm(space.subtract(model, reference))
        )
        constraint = estimator.misfit_constraint(data)
        assert constraint(model) == pytest.approx(
            problem.chi_squared(model, data) / estimator.target - 1.0
        )
        gi.testing.check_gradient(constraint, model, rng=rng)
        gi.testing.check_gradient(estimator.objective(data), model, rng=rng)

    def test_refusals(self, linear):
        space, data_space, forward, error, data = linear
        with pytest.raises(ValueError, match="error measure"):
            Occam(ForwardProblem(forward))

    def test_on_the_ray_problem(self):
        """The constraint is active at the solution and the model is smaller
        than the MAP model, which fits the data better than it need."""
        from .test_laplace import ray_problem

        space, problem, prior, data = ray_problem()
        mode = MaximumAPosteriori(
            problem,
            prior,
            optimizer=NewtonCG(
                preconditioner=prior.covariance, max_iterations=15, max_cg_iterations=40
            ),
        )(data)
        # The ray operator supplies its second derivative, so the objective
        # and the constraint both carry Hessians and the inner solves can be
        # Newton steps.
        estimator = Occam(
            problem,
            optimizer=NewtonCG(max_iterations=30, max_cg_iterations=40),
            tolerance=1e-4,
            gtol=1e-6,
        )
        result = estimator.solve(data)
        assert result.converged
        assert problem.chi_squared(result.minimizer, data) == pytest.approx(
            estimator.target, rel=1e-3
        )
        assert result.inequality_multipliers[0] > 0.0
        assert space.norm(result.minimizer) < space.norm(mode)
        assert problem.chi_squared(mode, data) < estimator.target
