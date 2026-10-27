"""Nonlinear inference: the MAP model, and the Gaussian that sits on it.

The rest of :mod:`pygeoinf2.inference` assumes the forward operator is linear,
which makes the posterior Gaussian and its expectation an affine function of
the data. When the operator is nonlinear, neither holds, and the standard
first thing to do is the two-step this module implements:

1. **Find the mode.** Minimize ``chi^2(m, d) + (m - m0, P (m - m0))``, which is
   twice the negative log posterior. That is an ordinary optimization, and the
   optimizers in :mod:`pygeoinf2.numerics.optimization` do it.
2. **Put a Gaussian on it.** Expand the same functional to second order about
   the minimizer. The curvature is the normal operator of the *linearized*
   problem, so the posterior covariance is its inverse -- which is exactly the
   linear answer, evaluated at the mode rather than assumed everywhere.

That is the Laplace approximation. It is *an approximation*, and its quality is
the question the module cannot answer for you: it is exact when the operator is
linear, good when the posterior is unimodal and nearly quadratic near its peak,
and silently wrong when it is neither. :meth:`MaximumAPosteriori.at` hands back
the machinery a sampler would need to check -- see below.

**What a sampler needs.** A function-space MCMC method -- preconditioned
Crank-Nicolson, and the Stuart-school samplers generally -- needs three
things: draws from the prior, the log density of the posterior up to a
constant, and the forward problem. The prior supplies the first,
:meth:`MaximumAPosteriori.log_posterior` is the second, and the result of
:meth:`MaximumAPosteriori.at` carries the posterior's exact density relative
to the prior beside its Gaussian approximation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..algebra.linearization import QuadraticModel
from ..algebra.operators import Functional, LinearFunctional
from ..algebra.spaces import HilbertSpace
from ..numerics.optimization import LBFGS, Optimizer
from ..numerics.solvers import LinearSolver
from ..probability.gaussian import GaussianMeasure
from ..numerics.normal import NormalOperator
from .problem import ForwardProblem

if TYPE_CHECKING:  # pragma: no cover
    from .gaussian import GaussianPosterior

__all__ = ["MaximumAPosteriori", "LaplaceResult"]


class LaplaceResult:
    """The mode, the Gaussian about it, and what the search cost."""

    def __init__(
        self,
        model: Any,
        measure: GaussianMeasure,
        optimization: Any,
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
    def optimization(self) -> Any:
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


class MaximumAPosteriori:
    """The MAP model of a nonlinear problem, with a Gaussian about it.

    Args:
        problem: the forward problem. Its operator may be nonlinear; it needs
            to supply a derivative, which :meth:`~pygeoinf2.algebra.operators.Operator.at`
            provides.
        prior: a Gaussian prior on the model space.
        optimizer: how to find the mode. L-BFGS with 500 iterations by
            default, which needs only gradients -- the exact Hessian of a
            nonlinear misfit carries a second-derivative term that a forward
            operator is rarely asked to supply. Pass a Newton-type optimizer
            if it can.
        solver: how to invert the linearized normal operator for the
            covariance: a :class:`~pygeoinf2.numerics.solvers.LinearSolver`,
            or a callable receiving the normal operator and returning one.
            Conjugate gradients if omitted.

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
    ) -> None:
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
        self._optimizer = optimizer or LBFGS(max_iterations=500)
        self._solver = solver

    @property
    def model_space(self) -> HilbertSpace:
        """Where the models live."""
        return self._problem.model_space

    @property
    def data_space(self) -> HilbertSpace:
        """Where the data live."""
        return self._problem.data_space

    def objective(self, data: Any, /) -> Functional:
        """``chi^2(m, d) + (m - m0, P (m - m0))``: twice the negative log posterior.

        The misfit comes from the problem and the prior term from the prior.
        The gradient is ``-2 J* R^-1 r + 2 P (m - m0)``, with ``J`` the
        derivative of the forward operator at the point and ``r`` the residual
        less the error's expectation; ``J`` is where the nonlinearity enters,
        and the only place it does. Each linearization costs one forward
        evaluation, serving the value and the gradient together.

        Args:
            data: the observations.

        Returns:
            A functional on the model space, with a gradient.
        """
        space = self.model_space
        data_space = self.data_space
        forward = self._problem.forward_operator
        error = self._problem.error_measure
        precision = self._prior.precision
        expectation = self._prior.expectation

        def prior_term(model: Any) -> tuple[Any, float]:
            offset = space.subtract(model, expectation)
            return offset, float(space.inner_product(precision(offset), offset))

        def value(model: Any) -> float:
            return self._problem.chi_squared(model, data) + prior_term(model)[1]

        def linearize(model: Any) -> QuadraticModel:
            # One forward evaluation serves both halves: the misfit is read
            # from the linearization's value and the derivative from its
            # operator, so an optimizer's ``at()`` evaluates the forward
            # operator once, not twice.
            linearization = forward.at(model)
            residual = data_space.subtract(data, linearization.value)
            offset, offset_term = prior_term(model)
            weighted = error.precision(data_space.subtract(residual, error.expectation))
            gradient = space.axpy(
                2.0,
                precision(offset),
                space.scale(-2.0, linearization.derivative.adjoint(weighted)),
            )
            return QuadraticModel(
                model,
                self._problem.chi_squared_from_residual(residual) + offset_term,
                LinearFunctional.from_representer(space, gradient),
            )

        return Functional.from_callables(space, value, linearize=linearize)

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

    def at(self, model: Any, data: Any, /) -> "GaussianPosterior":
        """The Laplace Gaussian about a *given* model, without searching.

        For when the mode is already known -- from a previous run, from a
        different method, or because the question is what the approximation
        looks like somewhere else. :meth:`__call__` is this composed with the
        search.

        The result is a :class:`~pygeoinf2.inference.gaussian.GaussianPosterior`
        whose two readings deliberately differ: the Gaussian moments are the
        *approximation* -- the curvature at this model -- while
        ``relative_log_density`` is the **exact** posterior's log density
        relative to the prior, ``-Phi`` with ``Phi`` the likelihood potential
        of :meth:`~pygeoinf2.inference.problem.ForwardProblem.potential`,
        computed in the data space. A sampler takes the pair whole: exact
        target, Gaussian approximation. The normalizer is ``None``; a Laplace
        estimate of the evidence is a separate claim this does not make.

        Args:
            model: where to center it.
            data: the observations.

        Returns:
            A Gaussian with that mean and the linearized posterior covariance,
            carrying the exact relative density.
        """
        from ..inference.gaussian import GaussianPosterior
        from ..numerics.solvers import resolve_solver

        normal = self.linearized_normal(model)
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
            reference=self._prior,
            relative_log_density=-self._problem.potential(data),
        )

    def __call__(self, data: Any, /, *, start: Any = None) -> LaplaceResult:
        """Find the mode, and put a Gaussian on it.

        Args:
            data: the observations.
            start: where the search begins. The prior's mean by default, which
                is the honest starting guess -- it is what is believed before
                the data are seen.

        Returns:
            The mode, the Gaussian, and the optimizer's own result. The last
            is worth reading: a search that did not converge gives a curvature
            at the wrong point, and this does not raise on one, because a
            stalled search still says something and refusing to return it
            would say less.
        """
        origin = self._prior.expectation if start is None else start
        outcome = self._optimizer.minimize(self.objective(data), origin)
        return LaplaceResult(
            outcome.minimizer, self.at(outcome.minimizer, data), outcome
        )

    def __repr__(self) -> str:
        return f"MaximumAPosteriori({self.model_space!r} -> {self.data_space!r})"
