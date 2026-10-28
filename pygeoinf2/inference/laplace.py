"""Nonlinear inference: the MAP model, and the Gaussian that sits on it.

The rest of :mod:`pygeoinf2.inference` assumes the forward operator is linear,
which makes the posterior Gaussian and its expectation an affine function of
the data. When the operator is nonlinear, neither holds, and the standard
first thing to do is the two-step this module implements:

1. **Find the mode.** Minimize ``chi^2(m, d) + (m - m0, P (m - m0))``, which is
   twice the negative log posterior. That is an ordinary optimization, which
   :class:`MaximumAPosteriori` runs with the optimizers in
   :mod:`pygeoinf2.numerics.optimization`, and it is a point estimator: data
   in, the mode out.
2. **Put a Gaussian on it.** Expand the same functional to second order about
   the minimizer. The curvature is the normal operator of the *linearized*
   problem, so the posterior covariance is its inverse -- which is exactly the
   linear answer, evaluated at the mode rather than assumed everywhere.
   :class:`LaplaceApproximation` does this, and it is a measure estimator:
   data in, the Gaussian out.

That is the Laplace approximation. It is *an approximation*, and its quality is
the question the module cannot answer for you: it is exact when the operator is
linear, good when the posterior is unimodal and nearly quadratic near its peak,
and silently wrong when it is neither. The Gaussian carries the exact posterior
density relative to the prior beside its own moments, which is what a sampler
needs to check.

**What a sampler needs.** A function-space MCMC method -- preconditioned
Crank-Nicolson, and the Stuart-school samplers generally -- needs three
things: draws from the prior, the log density of the posterior up to a
constant, and the forward problem. The prior supplies the first,
:meth:`MaximumAPosteriori.log_posterior` is the second, and the Gaussian from
:class:`LaplaceApproximation` carries the posterior's exact density relative
to the prior beside its Gaussian approximation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..algebra.operators import Functional, LinearOperator
from ..algebra.spaces import HilbertSpace
from ..numerics.normal import NormalOperator
from ..numerics.optimization import OptimizationResult, Optimizer
from ..numerics.solvers import LinearSolver
from ..probability.gaussian import GaussianMeasure
from .estimators import MeasureEstimator
from .minimization import Minimization, _gauss_newton, _misfit_plus_quadratic
from .minimization import _mixed_derivative as _misfit_mixed_derivative
from .problem import ForwardProblem

if TYPE_CHECKING:  # pragma: no cover
    from ..probability.base import ProbabilityMeasure
    from .gaussian import GaussianPosterior

__all__ = ["MaximumAPosteriori", "LaplaceApproximation", "LaplaceResult"]


class MaximumAPosteriori(Minimization):
    """The MAP model of a nonlinear problem: the posterior's mode.

    A point estimator. The objective is twice the negative log posterior,
    ``chi^2(m, d) + (m - m0, P (m - m0))``; the search starts from the
    prior's expectation, the honest starting guess, and is run by the
    optimizer given. For a linear forward operator the mode is the posterior
    mean and the two linear-Gaussian routes agree; for a nonlinear one the
    mode is what an optimizer finds, and it is a mode, not a mean.

    The estimator carries its derivative with respect to the data, by the
    implicit function theorem at the mode: ``(P + J* R^-1 J)^-1 J* R^-1``
    when the forward operator has no second derivative, the Kalman gain of
    the linearized problem, and the same with the operator's curvature in
    the Hessian when it has one. The Gaussian about the mode is
    :class:`LaplaceApproximation`'s business.

    Args:
        problem: the forward problem. Its operator may be nonlinear; it needs
            to supply a derivative, which :meth:`~pygeoinf2.algebra.operators.Operator.at`
            provides.
        prior: a Gaussian prior on the model space.
        optimizer: how to find the mode. L-BFGS with 500 iterations by
            default, which needs only gradients. A Newton-type optimizer can
            be passed when the forward operator supplies its second
            derivative: the objective then carries the exact Hessian, the
            Gauss-Newton term plus the operator's curvature weighted by the
            residual; or with ``gauss_newton``, which needs only the
            derivative. Give such an optimizer the prior covariance as its
            preconditioner (``NewtonCG(preconditioner=prior.covariance)``):
            the Hessian is ``P + J* R^-1 J`` in the model space's own inner
            product, and without it the inner solve has to resolve the
            prior precision's spectrum, which on a Sobolev space spans many
            decades.
        solver: how to invert the Hessian at the mode for the derivative
            with respect to the data: a
            :class:`~pygeoinf2.numerics.solvers.LinearSolver`, or a callable
            receiving the Hessian and returning one. Conjugate gradients if
            omitted.
        gauss_newton: carry the Gauss-Newton Hessian ``2 P + 2 J* R^-1 J``
            in the objective whether or not the forward operator has a
            second derivative, dropping the operator's curvature by choice.
            It is then also the Hessian behind the derivative with respect
            to the data and behind the Laplace approximation.

    Raises:
        ValueError: if the prior does not live on the model space, if the
            problem has no error measure, or if the prior has no precision --
            the objective is built from ``P``, and a prior that cannot supply
            one cannot state its own contribution to it.
        TypeError: if the problem's data uncertainty is a set rather than a
            measure.
    """

    def __init__(
        self,
        problem: ForwardProblem,
        prior: GaussianMeasure,
        /,
        *,
        optimizer: Optimizer | None = None,
        solver: LinearSolver | None = None,
        gauss_newton: bool = False,
    ) -> None:
        self._gauss_newton = bool(gauss_newton)
        if prior.domain != problem.model_space:
            raise ValueError(
                f"The prior lives on {prior.domain!r}, not on the model space "
                f"{problem.model_space!r}."
            )
        if not problem.has_error:
            raise ValueError(
                "A MAP estimate needs a data error measure: without one there "
                "is no misfit to weigh against the prior, and the mode is "
                "wherever the data are matched exactly."
            )
        problem.error_measure  # noqa: B018 - raises with the right message
        if prior.precision is None:
            raise ValueError(
                "The prior needs a precision. The objective is "
                "chi^2 + (m - m0, P (m - m0)), and a prior that cannot supply "
                "P cannot state its own half of it. Build it with a precision, "
                "or from standard deviations."
            )

        self._problem = problem
        self._prior = prior
        super().__init__(
            problem.data_space,
            problem.model_space,
            self.objective,
            optimizer=optimizer,
            start=prior.expectation,
            mixed_derivative=lambda data, model: _misfit_mixed_derivative(
                problem, model
            ),
            solver=solver,
        )

    @property
    def forward_problem(self) -> ForwardProblem:
        """The forward problem."""
        return self._problem

    @property
    def prior(self) -> GaussianMeasure:
        """The prior on the model space."""
        return self._prior

    def objective(self, data: Any, /) -> Functional:
        """``chi^2(m, d) + (m - m0, P (m - m0))``: twice the negative log posterior.

        The misfit comes from the problem and the prior term from the prior.
        The gradient is ``-2 J* R^-1 r + 2 P (m - m0)``, with ``J`` the
        derivative of the forward operator at the point and ``r`` the residual
        less the error's expectation; ``J`` is where the nonlinearity enters,
        and the only place it does. Each linearization costs one forward
        evaluation, serving the value and the gradient together. The exact
        Hessian, with the operator's curvature weighted by the residual, is
        carried whenever the forward operator can supply its second
        derivative, or the Gauss-Newton Hessian when the estimator was asked
        for it.

        Args:
            data: the observations.

        Returns:
            A functional on the model space, with a gradient.
        """
        return _misfit_plus_quadratic(
            self._problem,
            data,
            center=self._prior.expectation,
            precision=self._prior.precision,
            gauss_newton=self._gauss_newton,
        )

    def log_posterior(self, data: Any, /) -> Functional:
        """The log posterior density, up to an additive constant.

        Minus half :meth:`objective`: the sign and scale a sampler wants,
        where an optimizer wants the objective itself.

        Args:
            data: the observations.

        Returns:
            A functional on the model space, with a gradient.
        """
        return -0.5 * self.objective(data)

    def linearized_normal(self, model: Any, /) -> NormalOperator:
        """The normal operator of the problem linearized at a model.

        ``P + J* R^-1 J`` in the model-space formalism, with ``J`` the
        derivative there. This is half the Gauss-Newton approximation to the
        Hessian of :meth:`objective`, which drops the term in the operator's
        *second* derivative. That term vanishes at a perfect fit and is small
        near a good one, and a forward operator is rarely asked to supply it.

        Args:
            model: where to linearize.

        Returns:
            The normal operator, which carries its own factors.
        """
        return NormalOperator(
            self._problem.forward_operator.at(model).derivative,
            self._prior,
            error=self._problem.error_measure,
            formalism="model_space",
        )

    def hessian(self, data: Any, model: Any, /) -> LinearOperator:
        """The Hessian of :meth:`objective` at a model: exact, or Gauss-Newton.

        The exact Hessian when the forward operator supplies its second
        derivative; otherwise twice :meth:`linearized_normal`, which drops
        the operator's curvature weighted by the residual.

        Args:
            data: the observations.
            model: where to take it.

        Returns:
            A self-adjoint operator on the model space.
        """
        objective = self.objective(data)
        if objective.has_hessian:
            return objective.hessian(model)
        return _gauss_newton(self._problem, model, precision=self._prior.precision)


class LaplaceResult:
    """The mode, the Gaussian about it, and what the search cost."""

    def __init__(
        self,
        model: Any,
        measure: GaussianMeasure,
        optimization: OptimizationResult,
        /,
    ) -> None:
        self._model = model
        self._measure = measure
        self._optimization = optimization

    @property
    def model(self) -> Any:
        """The maximum a posteriori model: the mode, not the mean.

        For a nonlinear problem these differ, and it is the mode that an
        optimizer finds. Saying which is which matters because every linear
        method here returns a mean, and the two are quietly interchangeable
        only in the linear case.
        """
        return self._model

    @property
    def measure(self) -> GaussianMeasure:
        """The Laplace approximation: a Gaussian centered on the mode.

        Its covariance is the inverse of the linearized normal operator, and
        so describes the posterior's curvature *at the mode* and nothing about
        its shape elsewhere.
        """
        return self._measure

    @property
    def optimization(self) -> OptimizationResult:
        """The optimizer's own result.

        Worth looking at rather than a formality: a MAP model from a search
        that did not converge is not the mode, and the Gaussian built on it is
        a curvature at the wrong point.
        """
        return self._optimization

    @property
    def converged(self) -> bool:
        """Whether the search reached the mode."""
        return bool(self._optimization.converged)

    def __repr__(self) -> str:
        return (
            f"LaplaceResult(converged={self.converged}, "
            f"iterations={self._optimization.iterations})"
        )


class LaplaceApproximation(MeasureEstimator):
    """The Gaussian about the mode: a measure estimator on a MAP estimator.

    Wraps a :class:`MaximumAPosteriori` and the solver that inverts its
    linearized normal operator. Called on data it finds the mode and returns
    the Gaussian centered there whose covariance is the inverse of
    ``P + J* R^-1 J`` at the mode -- the linear posterior's covariance,
    evaluated at one point instead of assumed everywhere. That is what the
    Laplace approximation is: a point estimate with the curvature at it, and
    wrapping the point estimator says so.

    The result is a :class:`~pygeoinf2.inference.gaussian.GaussianPosterior`
    whose two readings deliberately differ: the Gaussian moments are the
    *approximation*, the curvature at the mode, while
    ``relative_log_density`` is the **exact** posterior's log density
    relative to the prior, ``-Phi`` with ``Phi`` the likelihood potential of
    :meth:`~pygeoinf2.inference.problem.ForwardProblem.potential``. A sampler
    takes the pair whole: exact target, Gaussian approximation. The
    normalizer is ``None``; a Laplace estimate of the evidence is a separate
    claim this does not make.
    """

    def __init__(
        self,
        estimator: MaximumAPosteriori,
        /,
        *,
        solver: LinearSolver | None = None,
    ) -> None:
        """
        Args:
            estimator: the MAP estimator supplying the mode and the
                linearized normal operator.
            solver: how to invert the linearized normal operator for the
                covariance: a :class:`~pygeoinf2.numerics.solvers.LinearSolver`,
                or a callable receiving the normal operator and returning
                one. Conjugate gradients if omitted.

        Raises:
            TypeError: if the estimator is not a :class:`MaximumAPosteriori`.
        """
        if not isinstance(estimator, MaximumAPosteriori):
            raise TypeError(
                "A Laplace approximation is built on a MaximumAPosteriori "
                f"estimator, not a {type(estimator).__name__}."
            )
        self._estimator = estimator
        self._solver = solver

    @classmethod
    def from_problem(
        cls,
        problem: ForwardProblem,
        prior: GaussianMeasure,
        /,
        *,
        optimizer: Optimizer | None = None,
        solver: LinearSolver | None = None,
    ) -> "LaplaceApproximation":
        """The approximation for a problem and a prior, building the MAP inside.

        Args:
            problem: the forward problem, with a Gaussian error measure.
            prior: a Gaussian prior on the model space, with a precision.
            optimizer: how the mode is found; see :class:`MaximumAPosteriori`.
            solver: how the linearized normal operator is inverted, for the
                covariance and for the MAP estimator's own derivative.

        Returns:
            The estimator.
        """
        return cls(
            MaximumAPosteriori(problem, prior, optimizer=optimizer, solver=solver),
            solver=solver,
        )

    @property
    def estimator(self) -> MaximumAPosteriori:
        """The MAP estimator this is built on."""
        return self._estimator

    @property
    def data_space(self) -> HilbertSpace:
        """Where the data live."""
        return self._estimator.data_space

    @property
    def property_space(self) -> HilbertSpace:
        """The model space: the Gaussian is on the model itself."""
        return self._estimator.model_space

    @property
    def model_space(self) -> HilbertSpace:
        """Where the models live."""
        return self._estimator.model_space

    def at(self, model: Any, data: Any, /) -> "GaussianPosterior":
        """The Laplace Gaussian about a *given* model, without searching.

        For when the mode is already known -- from a previous run, from a
        different method, or because the question is what the approximation
        looks like somewhere else. :meth:`__call__` is this composed with the
        search.

        Args:
            model: where to center it.
            data: the observations.

        Returns:
            A Gaussian with that mean and the linearized posterior covariance,
            carrying the exact relative density.
        """
        from ..inference.gaussian import GaussianPosterior
        from ..numerics.solvers import resolve_solver

        estimator = self._estimator
        normal = estimator.linearized_normal(model)
        # In the model-space formalism the posterior covariance *is* N^-1, the
        # construction LinearGaussianInversion makes, here evaluated at this
        # model. The precision is N itself, which is worth keeping: it is
        # what a sampler and a Mahalanobis distance want, and it is exact
        # where the inverse is a solve.
        inverse = resolve_solver(self._solver, normal)(normal)
        return GaussianPosterior(
            self.model_space,
            covariance=inverse,
            precision=normal,
            expectation=model,
            reference=estimator.prior,
            relative_log_density=-estimator.forward_problem.potential(data),
        )

    def solve(self, data: Any, /, *, start: Any = None) -> LaplaceResult:
        """Find the mode, put a Gaussian on it, and keep the search's record.

        Args:
            data: the observations.
            start: where the search begins. The prior's expectation by
                default.

        Returns:
            The mode, the Gaussian, and the optimizer's own result. The last
            is worth reading: a search that did not converge gives a curvature
            at the wrong point, and this does not raise on one, because a
            stalled search still says something and refusing to return it
            would say less.
        """
        outcome = self._estimator.solve(data, start=start)
        return LaplaceResult(
            outcome.minimizer, self.at(outcome.minimizer, data), outcome
        )

    def __call__(self, data: Any, /, *, start: Any = None) -> "GaussianPosterior":
        """The Gaussian about the mode, given data.

        Args:
            data: the observations.
            start: where the search begins. The prior's expectation by
                default.

        Returns:
            The Laplace Gaussian, carrying the exact relative density.
        """
        return self.solve(data, start=start).measure

    def push_forward(self, operator: LinearOperator, /) -> MeasureEstimator:
        """The estimator of the Gaussian on ``T m`` rather than on ``m``.

        The mode goes through ``T`` and the covariance is conjugated,
        ``T C T*``, as any Gaussian is pushed forward.

        Args:
            operator: the property map ``T``, from the model space.

        Returns:
            The measure estimator of the property.
        """
        return _PushedMeasureEstimator(self, operator)

    def __repr__(self) -> str:
        return f"LaplaceApproximation({self.data_space!r} -> {self.model_space!r})"


class _PushedMeasureEstimator(MeasureEstimator):
    """A measure estimator whose every answer is pushed through an operator."""

    def __init__(self, base: MeasureEstimator, operator: LinearOperator) -> None:
        if operator.domain != base.property_space:
            raise ValueError(
                f"The property operator acts on {operator.domain!r}, not on "
                f"the estimator's space {base.property_space!r}."
            )
        self._base = base
        self._operator = operator

    @property
    def data_space(self) -> HilbertSpace:
        """Where the data live."""
        return self._base.data_space

    @property
    def property_space(self) -> HilbertSpace:
        """The property operator's codomain."""
        return self._operator.codomain

    def __call__(self, data: Any, /, *, start: Any = None) -> "ProbabilityMeasure":
        return self._base(data, start=start).push_forward(self._operator)

    def push_forward(self, operator: LinearOperator, /) -> MeasureEstimator:
        """The property of the property: the operators compose."""
        return _PushedMeasureEstimator(self._base, operator @ self._operator)
