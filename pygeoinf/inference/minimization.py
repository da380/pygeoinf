"""
Point estimators that minimize: an objective built from the data, a start,
and an optimizer.

The generic class, :class:`Minimization`, is a mapping from data to the
model that minimizes an objective the data define, and it is an operator:
where the objective's Hessian and its mixed second derivative with respect
to the data are available, the implicit function theorem gives the exact
derivative of the estimate with respect to the data at the solution, and a
Bayesian inversion can be built on top of the estimator by the chain rule.

Two special cases are built here. :class:`NonlinearLeastSquares` minimizes
the weighted misfit of a forward problem plus Tikhonov damping about a
reference model, and agrees with the linear least-squares estimator whenever
the forward operator happens to be linear. :class:`ConstrainedMinimization`
minimizes under equality and inequality constraints by an augmented
Lagrangian, and :class:`Occam` is its first special case: the model closest
to a reference whose misfit is no worse than a chosen level. The maximum a
posteriori estimator lives with its Laplace approximation in
:mod:`pygeoinf.inference.laplace` and is the third special case.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ..algebra.linearization import Linearization, QuadraticModel
from ..algebra.operators import (
    Functional,
    LinearFunctional,
    LinearOperator,
    Operator,
)
from ..algebra.spaces import HilbertSpace
from ..numerics.optimization import (
    LBFGS,
    AugmentedLagrangian,
    AugmentedLagrangianResult,
    OptimizationResult,
    Optimizer,
)
from ..numerics.solvers import LinearSolver, resolve_solver
from ..traits import Traits
from .estimators import ComposedPointEstimator, PointEstimator
from .problem import ForwardProblem

__all__ = [
    "ConstrainedMinimization",
    "Minimization",
    "NonlinearLeastSquares",
    "Occam",
]


class Minimization(PointEstimator, Operator):
    """The model minimizing an objective built from the data.

    ``d -> argmin_m f_d(m)``, with the objective ``f_d`` supplied as a
    callable of the data returning a functional on the model space, the
    search run by an optimizer from a starting model, and the record of the
    search one call away in :meth:`solve`.

    The estimator is an operator on the data. Its derivative comes from the
    implicit function theorem: at a solution ``m`` the gradient of the
    objective vanishes, so to first order

    .. code-block:: text

        H(m) dm == -M(d, m) dd

    with ``H`` the objective's Hessian and ``M`` the derivative of its
    gradient with respect to the data. The derivative is available when the
    objective carries a Hessian and a ``mixed_derivative`` is supplied, and
    costs one solve with the Hessian per application; otherwise the
    estimator has no derivative and says so.
    """

    def __init__(
        self,
        data_space: HilbertSpace,
        model_space: HilbertSpace,
        objective: Callable[[Any], Functional],
        /,
        *,
        optimizer: Optimizer | None = None,
        start: Any = None,
        mixed_derivative: Callable[[Any, Any], LinearOperator] | None = None,
        solver: LinearSolver | None = None,
    ) -> None:
        """
        Args:
            data_space: where the data live.
            model_space: where the models live.
            objective: the data to the functional on the model space to
                minimize. It needs a derivative; it needs a Hessian only for
                a Newton optimizer or for the estimator's derivative.
            optimizer: how to minimize. L-BFGS with 500 iterations by
                default, which needs only gradients.
            start: where the search begins: a model, a callable of the data
                returning one, or ``None`` for the zero model.
            mixed_derivative: ``(data, model) -> M``, the derivative of the
                objective's gradient with respect to the data at a point, a
                linear operator from the data space to the model space.
                Supplying it is what gives the estimator a derivative.
            solver: how to invert the Hessian for the derivative: a
                :class:`~pygeoinf.numerics.solvers.LinearSolver`, or a
                callable receiving the Hessian and returning one. Conjugate
                gradients if omitted.
        """
        super().__init__(data_space, model_space)
        self._objective = objective
        self._optimizer = optimizer or LBFGS(max_iterations=500)
        self._start = start
        self._mixed_derivative = mixed_derivative
        self._solver = solver

    # ----------------------------------------------------------------- #
    #                            Structure                               #
    # ----------------------------------------------------------------- #

    @property
    def data_space(self) -> HilbertSpace:
        """Where the data live."""
        return self.domain

    @property
    def property_space(self) -> HilbertSpace:
        """The model space: this estimates the model itself."""
        return self.codomain

    @property
    def model_space(self) -> HilbertSpace:
        """Where the models live."""
        return self.codomain

    @property
    def optimizer(self) -> Optimizer:
        """The method that runs the search."""
        return self._optimizer

    @property
    def has_derivative(self) -> bool:
        """Whether the estimate can be differentiated with respect to the data.

        True when a mixed derivative was supplied. The objective must then
        also carry a Hessian, which is checked when the derivative is asked
        for.
        """
        return self._mixed_derivative is not None

    # ----------------------------------------------------------------- #
    #                         The problem solved                         #
    # ----------------------------------------------------------------- #

    def objective(self, data: Any, /) -> Functional:
        """The functional on the model space minimized for this data.

        Args:
            data: the observations.

        Returns:
            The objective, with whatever derivatives the builder supplies.
        """
        return self._objective(data)

    def starting_point(self, data: Any, /) -> Any:
        """Where the search begins for this data, unless a start is passed.

        Args:
            data: the observations.

        Returns:
            The model given at construction, the one its callable returns
            for this data, or the zero model.
        """
        if self._start is None:
            return self.model_space.zero()
        if callable(self._start):
            return self._start(data)
        return self._start

    def hessian(self, data: Any, model: Any, /) -> LinearOperator:
        """The objective's Hessian at a model, for the derivative.

        Args:
            data: the observations.
            model: where to take it, normally the solution.

        Returns:
            A self-adjoint operator on the model space.

        Raises:
            NotImplementedError: if the objective carries no Hessian.
        """
        objective = self.objective(data)
        if not objective.has_hessian:
            raise NotImplementedError(
                "The objective carries no Hessian, so the estimate cannot be "
                "differentiated with respect to the data."
            )
        return objective.hessian(model)

    def mixed_derivative(self, data: Any, model: Any, /) -> LinearOperator:
        """The derivative of the objective's gradient with respect to the data.

        Args:
            data: the observations.
            model: where to take it.

        Returns:
            A linear operator from the data space to the model space.

        Raises:
            NotImplementedError: if none was supplied.
        """
        if self._mixed_derivative is None:
            raise NotImplementedError(
                f"{type(self).__name__} was built without a mixed derivative, "
                "so it carries no derivative with respect to the data."
            )
        return self._mixed_derivative(data, model)

    # ----------------------------------------------------------------- #
    #                            The estimate                            #
    # ----------------------------------------------------------------- #

    def solve(self, data: Any, /, *, start: Any = None) -> OptimizationResult:
        """Run the search and return its record.

        Args:
            data: the observations.
            start: where to begin; :meth:`starting_point` if omitted.

        Returns:
            The optimizer's result: the minimizer, and whether and how the
            search stopped. A search that did not converge still returns its
            last iterate, because a stalled search says something and
            raising would say less; read ``converged`` before trusting it.
        """
        origin = self.starting_point(data) if start is None else start
        return self._optimizer.minimize(self.objective(data), origin)

    def __call__(self, data: Any, /, *, start: Any = None) -> Any:
        """The minimizer, given data.

        Args:
            data: the observations.
            start: where to begin; :meth:`starting_point` if omitted.

        Returns:
            The model the search ended at.
        """
        return self.solve(data, start=start).minimizer

    def _value(self, data: Any) -> Any:
        return self.solve(data).minimizer

    def derivative_at(self, data: Any, model: Any, /) -> LinearOperator:
        """The derivative with respect to the data, at a solution in hand.

        ``-H^-1 M`` by the implicit function theorem, with the Hessian
        inverted by the estimator's solver. The Hessian is taken to be
        positive definite, which it is at a strict minimizer; at any other
        model the formula does not apply.

        Args:
            data: the observations.
            model: the minimizer for them.

        Returns:
            A linear operator from the data space to the model space, with
            its adjoint.

        Raises:
            NotImplementedError: if the objective carries no Hessian or no
                mixed derivative was supplied.
        """
        space = self.model_space
        data_space = self.data_space
        hessian = self.hessian(data, model)
        mixed = self.mixed_derivative(data, model)
        definite = LinearOperator.self_adjoint(
            space, hessian, traits=Traits.POSITIVE_DEFINITE
        )
        inverse = resolve_solver(self._solver, definite)(definite)

        def value(perturbation: Any) -> Any:
            return space.negative(inverse(mixed(perturbation)))

        def adjoint(direction: Any) -> Any:
            return data_space.negative(mixed.adjoint(inverse(direction)))

        return LinearOperator.from_callables(data_space, space, value, adjoint=adjoint)

    def _derivative(self, data: Any) -> LinearOperator:
        return self.derivative_at(data, self._value(data))

    def _linearize(self, data: Any) -> Linearization:
        model = self._value(data)
        return Linearization(data, model, self.derivative_at(data, model))

    def push_forward(self, operator: LinearOperator, /) -> ComposedPointEstimator:
        """The estimator of ``T m`` rather than of ``m``.

        The property operator is composed with this estimator, so the
        derivative, where there is one, follows by the chain rule, and a
        ``start=`` passed to the composed estimator's call reaches the
        search.

        Args:
            operator: the property map ``T``, from the model space.

        Returns:
            The point estimator of the property.

        Raises:
            ValueError: if ``T`` does not act on the model space.
        """
        return ComposedPointEstimator(self, operator)


# --------------------------------------------------------------------- #
#                  Misfit objectives shared by the subclasses            #
# --------------------------------------------------------------------- #


def _misfit_plus_quadratic(
    problem: ForwardProblem,
    data: Any,
    /,
    *,
    center: Any = None,
    precision: LinearOperator | None = None,
    weight: float = 1.0,
    gauss_newton: bool = False,
) -> Functional:
    """``chi^2(m, d) + weight (m - c, P (m - c))``, with its derivatives.

    The misfit is the problem's: the squared Mahalanobis distance of the
    residual under the error measure, or the squared norm when the data are
    exact. The gradient is ``-2 J* R^-1 r + 2 weight P (m - c)``, with ``J``
    the forward operator's derivative at the point and ``r`` the residual
    less the error's expectation; one linearization of the forward operator
    serves the value and the gradient together. The exact Hessian,
    ``2 weight P + 2 J* R^-1 J - 2 F''(m)[., .]* (R^-1 r)``, is supplied
    whenever the forward operator can give its second derivative. With
    ``gauss_newton`` the Hessian carried is ``2 weight P + 2 J* R^-1 J``
    instead, the curvature term dropped by choice, whether or not the
    operator has a second derivative: the Gauss-Newton approximation, which
    a Newton-type optimizer can run on a forward operator that supplies
    only its derivative.

    With ``center`` omitted there is no quadratic term; with ``precision``
    omitted ``P`` is the identity.
    """
    space = problem.model_space
    data_space = problem.data_space
    forward = problem.forward_operator
    error = problem.error_measure if problem.has_error else None
    quadratic = center is not None and weight != 0.0
    carries_hessian = gauss_newton or forward.has_second_derivative

    def apply_precision(vector: Any) -> Any:
        return vector if error is None else error.precision(vector)

    def weighted_residual(residual: Any) -> Any:
        if error is None:
            return residual
        return error.precision(data_space.subtract(residual, error.expectation))

    def quadratic_term(model: Any) -> tuple[Any, float]:
        """``P (m - c)`` and ``weight (m - c, P (m - c))``."""
        offset = space.subtract(model, center)
        image = offset if precision is None else precision(offset)
        return image, weight * float(space.inner_product(image, offset))

    def value(model: Any) -> float:
        total = problem.chi_squared(model, data)
        if quadratic:
            total += quadratic_term(model)[1]
        return total

    def hessian_at(
        model: Any, jacobian: LinearOperator, weighted: Any
    ) -> LinearOperator:
        def apply(direction: Any) -> Any:
            out = space.scale(
                2.0, jacobian.adjoint(apply_precision(jacobian(direction)))
            )
            if quadratic:
                image = direction if precision is None else precision(direction)
                out = space.axpy(2.0 * weight, image, out)
            if gauss_newton:
                return out
            curvature = forward.second_derivative(model, direction).adjoint(weighted)
            return space.axpy(-2.0, curvature, out)

        return LinearOperator.self_adjoint(space, apply)

    def linearize(model: Any) -> QuadraticModel:
        linearization = forward.at(model)
        residual = data_space.subtract(data, linearization.value)
        weighted = weighted_residual(residual)
        total = problem.chi_squared_from_residual(residual)
        gradient = space.scale(-2.0, linearization.derivative.adjoint(weighted))
        if quadratic:
            image, term = quadratic_term(model)
            total += term
            gradient = space.axpy(2.0 * weight, image, gradient)
        return QuadraticModel(
            model,
            total,
            LinearFunctional.from_representer(space, gradient),
            hessian=(
                hessian_at(model, linearization.derivative, weighted)
                if carries_hessian
                else None
            ),
        )

    def hessian(model: Any) -> LinearOperator:
        return linearize(model).hessian

    return Functional.from_callables(
        space,
        value,
        linearize=linearize,
        hessian=hessian if carries_hessian else None,
    )


def _gauss_newton(
    problem: ForwardProblem,
    model: Any,
    /,
    *,
    precision: LinearOperator | None = None,
    weight: float = 1.0,
) -> LinearOperator:
    """``2 weight P + 2 J* R^-1 J``: the Gauss-Newton Hessian of the objective.

    The exact Hessian less the term in the forward operator's second
    derivative, which vanishes at a perfect fit. Positive definite whenever
    the quadratic term is present.
    """
    space = problem.model_space
    forward = problem.forward_operator
    error = problem.error_measure if problem.has_error else None
    jacobian = forward.derivative(model)

    def apply(direction: Any) -> Any:
        image = jacobian(direction)
        if error is not None:
            image = error.precision(image)
        out = space.scale(2.0, jacobian.adjoint(image))
        if weight != 0.0:
            quadratic = direction if precision is None else precision(direction)
            out = space.axpy(2.0 * weight, quadratic, out)
        return out

    traits = Traits.POSITIVE_DEFINITE if weight > 0.0 else Traits.NONE
    return LinearOperator.self_adjoint(space, apply, traits=traits)


def _mixed_derivative(problem: ForwardProblem, model: Any, /) -> LinearOperator:
    """``-2 J* R^-1``: how the misfit's gradient moves with the data."""
    space = problem.model_space
    data_space = problem.data_space
    error = problem.error_measure if problem.has_error else None
    jacobian = problem.forward_operator.derivative(model)

    def value(perturbation: Any) -> Any:
        weighted = perturbation if error is None else error.precision(perturbation)
        return space.scale(-2.0, jacobian.adjoint(weighted))

    def adjoint(direction: Any) -> Any:
        image = jacobian(direction)
        if error is not None:
            image = error.precision(image)
        return data_space.scale(-2.0, image)

    return LinearOperator.from_callables(data_space, space, value, adjoint=adjoint)


# --------------------------------------------------------------------- #
#                        Nonlinear least squares                         #
# --------------------------------------------------------------------- #


class NonlinearLeastSquares(Minimization):
    """Damped least squares for a nonlinear forward operator.

    Minimizes ``chi^2(m, d) + t |m - m_ref|^2``: the weighted misfit of the
    forward problem, which is twice its likelihood potential, plus Tikhonov
    damping about a reference model in the model space's own norm. With a
    linear forward operator this is the linear least-squares estimator at
    the same damping, found by iteration instead of by a solve; its reason
    to exist is the nonlinear case, where there is no solve to do.

    The derivative with respect to the data is the implicit-function one,
    ``(t I + J* R^-1 J)^-1 J* R^-1`` at the solution when the forward
    operator has no second derivative, and the same with the operator's
    curvature in the Hessian when it has. The derivative needs ``t > 0`` or
    a forward operator whose linearization is injective; otherwise the
    Hessian is singular and the solve fails.
    """

    def __init__(
        self,
        problem: ForwardProblem,
        /,
        *,
        damping: float = 0.0,
        reference: Any = None,
        optimizer: Optimizer | None = None,
        solver: LinearSolver | None = None,
        gauss_newton: bool = False,
    ) -> None:
        """
        Args:
            problem: the forward problem. Its operator needs a derivative;
                its error measure, if any, must be Gaussian.
            damping: the Tikhonov parameter ``t``, non-negative.
            reference: the model the damping pulls towards, and the default
                start of the search; the zero model if omitted.
            optimizer: how to find the minimizer. L-BFGS with 500 iterations
                by default; a Newton method can be used when the forward
                operator supplies its second derivative, or with
                ``gauss_newton``.
            solver: how to invert the Hessian for the derivative with
                respect to the data. Conjugate gradients if omitted.
            gauss_newton: carry the Gauss-Newton Hessian,
                ``2 t I + 2 J* R^-1 J``, in the objective whether or not the
                forward operator has a second derivative, dropping the
                operator's curvature by choice.

        Raises:
            ValueError: for a negative damping, or if the reference model
                does not live on the model space.
            TypeError: if the problem's data uncertainty is a set rather than
                a measure.
        """
        if damping < 0.0:
            raise ValueError(f"The damping is non-negative, got {damping}.")
        if problem.has_error:
            problem.error_measure  # noqa: B018 - raises for a set-valued error
        space = problem.model_space
        self._problem = problem
        self._damping = float(damping)
        self._reference = space.zero() if reference is None else reference
        self._gauss_newton = bool(gauss_newton)
        super().__init__(
            problem.data_space,
            space,
            self.objective,
            optimizer=optimizer,
            start=self._reference,
            mixed_derivative=lambda data, model: _mixed_derivative(problem, model),
            solver=solver,
        )

    @property
    def forward_problem(self) -> ForwardProblem:
        """The forward problem."""
        return self._problem

    @property
    def damping(self) -> float:
        """The Tikhonov parameter."""
        return self._damping

    @property
    def reference(self) -> Any:
        """The model the damping pulls towards."""
        return self._reference

    def objective(self, data: Any, /) -> Functional:
        """``chi^2(m, d) + t |m - m_ref|^2`` for this data.

        Carries the gradient, and the exact Hessian when the forward
        operator supplies its second derivative, or the Gauss-Newton one
        when the estimator was asked for it.

        Args:
            data: the observations.

        Returns:
            A functional on the model space.
        """
        return _misfit_plus_quadratic(
            self._problem,
            data,
            center=self._reference if self._damping > 0.0 else None,
            weight=self._damping,
            gauss_newton=self._gauss_newton,
        )

    def hessian(self, data: Any, model: Any, /) -> LinearOperator:
        """The Hessian at a model: exact, or Gauss-Newton.

        The exact Hessian of :meth:`objective` when the forward operator
        supplies its second derivative; otherwise ``2 t I + 2 J* R^-1 J``,
        which drops the operator's curvature weighted by the residual, a
        term that vanishes at a perfect fit.

        Args:
            data: the observations.
            model: where to take it.

        Returns:
            A self-adjoint operator on the model space.
        """
        objective = self.objective(data)
        if objective.has_hessian:
            return objective.hessian(model)
        return _gauss_newton(self._problem, model, weight=self._damping)


# --------------------------------------------------------------------- #
#                        Constrained minimization                        #
# --------------------------------------------------------------------- #


class ConstrainedMinimization(Minimization):
    """The model minimizing an objective under constraints the data define.

    ``argmin_m f_d(m)`` subject to ``c_i(m; d) == 0`` and ``g_j(m; d) <= 0``,
    each constraint a callable of the data returning a functional on the
    model space, solved by the method of multipliers
    (:class:`~pygeoinf.numerics.optimization.AugmentedLagrangian`) over the
    unconstrained optimizer given. :meth:`solve` returns the augmented
    Lagrangian's result, with the multipliers and the final violation.

    No derivative with respect to the data is offered: it would need the
    optimality conditions differentiated through the active set, which this
    does not do.
    """

    def __init__(
        self,
        data_space: HilbertSpace,
        model_space: HilbertSpace,
        objective: Callable[[Any], Functional],
        /,
        *,
        equalities: Sequence[Callable[[Any], Functional]] = (),
        inequalities: Sequence[Callable[[Any], Functional]] = (),
        optimizer: Optimizer | None = None,
        start: Any = None,
        penalty: float = 10.0,
        penalty_growth: float = 10.0,
        reduction: float = 0.25,
        tolerance: float = 1e-6,
        rtol: float = 1e-8,
        gtol: float = 0.0,
        max_iterations: int = 30,
    ) -> None:
        """
        Args:
            data_space: where the data live.
            model_space: where the models live.
            objective: the data to the functional to minimize.
            equalities: callables of the data returning the functionals
                constrained to vanish.
            inequalities: callables of the data returning the functionals
                constrained to be at most zero.
            optimizer: the unconstrained method for the inner solves.
                L-BFGS with 500 iterations by default.
            start: where the search begins: a model, a callable of the data,
                or ``None`` for the zero model. It need not be feasible.
            penalty: the augmented Lagrangian's initial penalty.
            penalty_growth: the factor it grows by when the violation does
                not fall enough between outer iterations.
            reduction: the factor the violation must fall by for the penalty
                to stay put.
            tolerance: the constraint violation at which the search may
                stop, in the constraints' units.
            rtol: the gradient tolerance of the augmented Lagrangian the
                final inner solve must meet, relative to its gradient at
                the start.
            gtol: an absolute floor under the inner gradient tolerances, in
                the model space's norm; zero by default.
            max_iterations: the budget of outer iterations; the inner
                budget is the optimizer's own.
        """
        super().__init__(
            data_space, model_space, objective, optimizer=optimizer, start=start
        )
        self._equalities = tuple(equalities)
        self._inequalities = tuple(inequalities)
        self._penalty = penalty
        self._penalty_growth = penalty_growth
        self._reduction = reduction
        self._tolerance = tolerance
        self._rtol = rtol
        self._gtol = gtol
        self._max_iterations = max_iterations

    @property
    def has_derivative(self) -> bool:
        """False: a constrained minimizer is not differentiated here."""
        return False

    def equalities(self, data: Any, /) -> tuple[Functional, ...]:
        """The equality constraints for this data.

        Args:
            data: the observations.

        Returns:
            The functionals constrained to vanish.
        """
        return tuple(build(data) for build in self._equalities)

    def inequalities(self, data: Any, /) -> tuple[Functional, ...]:
        """The inequality constraints for this data.

        Args:
            data: the observations.

        Returns:
            The functionals constrained to be at most zero.
        """
        return tuple(build(data) for build in self._inequalities)

    def method(self, data: Any, /) -> AugmentedLagrangian:
        """The augmented Lagrangian that solves the problem for this data.

        Args:
            data: the observations.

        Returns:
            The constrained optimizer, carrying this data's constraints.
        """
        return AugmentedLagrangian(
            self._optimizer,
            equalities=self.equalities(data),
            inequalities=self.inequalities(data),
            penalty=self._penalty,
            penalty_growth=self._penalty_growth,
            reduction=self._reduction,
            tolerance=self._tolerance,
            rtol=self._rtol,
            gtol=self._gtol,
            max_iterations=self._max_iterations,
        )

    def solve(self, data: Any, /, *, start: Any = None) -> AugmentedLagrangianResult:
        """Run the constrained search and return its record.

        Args:
            data: the observations.
            start: where to begin; :meth:`starting_point` if omitted.

        Returns:
            The augmented Lagrangian's result: the minimizer, the
            multipliers, the final constraint violation, and whether the
            search converged.
        """
        origin = self.starting_point(data) if start is None else start
        return self.method(data).minimize(self.objective(data), origin)


class Occam(ConstrainedMinimization):
    """The model closest to a reference whose misfit meets a level.

    ``argmin_m |m - m_0|^2`` subject to ``chi^2(m, d) <= chi^2_crit``, the
    critical value of the chi-squared distribution on as many degrees of
    freedom as there are data at the chosen confidence level. The principle
    of parsimony as an inverse method: among the models the data do not
    reject, the one with the least structure, measured in the model space's
    own norm from the reference.

    For a linear forward operator this is the minimum-norm model of the
    discrepancy principle, which finds the same point by searching the
    Tikhonov damping for the one whose fit sits at the level. That search
    relies on the misfit of the damped minimizer being monotone in the
    damping, which it is for a linear operator and need not be once the
    operator bends; so this solves the constrained problem directly, by an
    augmented Lagrangian, with the damping sweep offered nowhere.

    When the reference model already fits the data to the level, it is the
    answer and the constraint is inactive; the result's inequality
    multiplier is zero in that case and the misfit's Lagrange multiplier
    otherwise.
    """

    def __init__(
        self,
        problem: ForwardProblem,
        /,
        *,
        level: float = 0.95,
        reference: Any = None,
        optimizer: Optimizer | None = None,
        start: Any = None,
        penalty: float = 10.0,
        penalty_growth: float = 10.0,
        reduction: float = 0.25,
        tolerance: float = 1e-4,
        rtol: float = 1e-8,
        gtol: float = 0.0,
        max_iterations: int = 30,
        gauss_newton: bool = False,
    ) -> None:
        """
        Args:
            problem: the forward problem. It must carry a Gaussian error
                measure, which sets the misfit's threshold.
            level: the confidence level the misfit is held to.
            reference: ``m_0``, the model the answer is kept close to; the
                zero model if omitted.
            optimizer: the unconstrained method for the inner solves.
                L-BFGS with 500 iterations by default.
            start: where the search begins; the reference model if omitted.
            penalty: the augmented Lagrangian's initial penalty.
            penalty_growth: the factor it grows by when the violation does
                not fall enough between outer iterations.
            reduction: the factor the violation must fall by for the penalty
                to stay put.
            tolerance: the constraint violation at which the search may
                stop, relative to the critical ``chi^2``: the constraint is
                ``chi^2 / chi^2_crit - 1 <= 0``, so an absolute tolerance in
                misfit units would be unreachable for a forward operator
                evaluated iteratively, whose misfit is itself noisy at its
                solver's tolerance.
            rtol: the gradient tolerance of the augmented Lagrangian the
                final inner solve must meet, relative to its gradient at
                the start.
            gtol: an absolute floor under the inner gradient tolerances, in
                the model space's norm; zero by default.
            max_iterations: the budget of outer iterations.
            gauss_newton: carry the Gauss-Newton Hessian of the misfit,
                ``2 J* R^-1 J``, in the constraint whether or not the
                forward operator has a second derivative, dropping the
                operator's curvature by choice; it lets a Newton-type inner
                optimizer run on an operator that supplies only its
                derivative.

        Raises:
            ValueError: if the problem has no error measure, there being
                then no level to hold the misfit to.
            TypeError: if its data uncertainty is a set rather than a
                measure.
        """
        if not problem.has_error:
            raise ValueError(
                "Occam's inversion needs a data error measure: the misfit "
                "level it holds the model to is a statement about the noise."
            )
        problem.error_measure  # noqa: B018 - raises for a set-valued error
        space = problem.model_space
        self._problem = problem
        self._level = level
        self._reference = space.zero() if reference is None else reference
        self._gauss_newton = bool(gauss_newton)
        super().__init__(
            problem.data_space,
            space,
            self.objective,
            inequalities=[self.misfit_constraint],
            optimizer=optimizer,
            start=self._reference if start is None else start,
            penalty=penalty,
            penalty_growth=penalty_growth,
            reduction=reduction,
            tolerance=tolerance,
            rtol=rtol,
            gtol=gtol,
            max_iterations=max_iterations,
        )

    @property
    def forward_problem(self) -> ForwardProblem:
        """The forward problem."""
        return self._problem

    @property
    def level(self) -> float:
        """The confidence level the misfit is held to."""
        return self._level

    @property
    def reference(self) -> Any:
        """The model the answer is kept close to."""
        return self._reference

    @property
    def target(self) -> float:
        """The chi-squared threshold the misfit is held to."""
        return self._problem.critical_chi_squared(level=self._level)

    def objective(self, data: Any, /) -> Functional:
        """``|m - m_0|^2``, with its gradient and its Hessian.

        Args:
            data: the observations, which the objective does not depend on.

        Returns:
            The squared distance from the reference, in the model space's
            norm.
        """
        space = self.model_space
        reference = self._reference

        def value(model: Any) -> float:
            return float(space.squared_norm(space.subtract(model, reference)))

        def gradient(model: Any) -> Any:
            return space.scale(2.0, space.subtract(model, reference))

        def hessian(model: Any) -> LinearOperator:
            return LinearOperator.self_adjoint(
                space,
                lambda direction: space.scale(2.0, direction),
                traits=Traits.POSITIVE_DEFINITE,
            )

        return Functional.from_callables(
            space, value, gradient=gradient, hessian=hessian
        )

    def misfit_constraint(self, data: Any, /) -> Functional:
        """``chi^2(m, d) / chi^2_crit - 1``, the constraint held at or below zero.

        Args:
            data: the observations.

        Returns:
            A functional on the model space with a gradient, and the exact
            Hessian when the forward operator supplies its second derivative,
            or the Gauss-Newton one when the estimator was asked for it.
        """
        misfit = _misfit_plus_quadratic(
            self._problem, data, gauss_newton=self._gauss_newton
        )
        target = self.target
        space = self.model_space

        scale = 1.0 / target

        def value(model: Any) -> float:
            return misfit(model) * scale - 1.0

        def linearize(model: Any) -> QuadraticModel:
            local = misfit.at(model)
            return QuadraticModel(
                model,
                local.value * scale - 1.0,
                local.derivative * scale,
                None if local.hessian is None else local.hessian * scale,
            )

        return Functional.from_callables(
            space,
            value,
            linearize=linearize,
            hessian=(lambda m: misfit.hessian(m) * scale)
            if misfit.has_hessian
            else None,
        )
