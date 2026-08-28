"""Nonlinear inference: the MAP model, and the Gaussian that sits on it.

The one substantive capability absent from both versions (**D-7**). Everything
else here assumes the forward operator is linear, which makes the posterior
Gaussian and its mean a linear function of the data. When it is not, neither
holds -- and the standard first thing to do is the two-step this module
implements:

1. **Find the mode.** Minimise ``chi^2(m, d) + (m - m0, P (m - m0))``, which is
   twice the negative log posterior. That is an ordinary optimisation, and the
   optimisers in :mod:`pygeoinf2.numerics.optimisation` do it.
2. **Put a Gaussian on it.** Expand the same functional to second order about
   the minimiser. The curvature is the normal operator of the *linearised*
   problem, so the posterior covariance is its inverse -- which is exactly the
   linear answer, evaluated at the mode rather than assumed everywhere.

That is the Laplace approximation. It is *an approximation*, and its quality is
the question the module cannot answer for you: it is exact when the operator is
linear, good when the posterior is unimodal and nearly quadratic near its peak,
and silently wrong when it is neither. :meth:`MaximumAPosteriori.at` hands back
the machinery a sampler would need to check -- see below.

**Built so a sampler can replace step 2.** A function-space MCMC method --
preconditioned Crank-Nicolson, and the Stuart-school samplers generally -- needs
three things: draws from the prior, the log density of the posterior up to a
constant, and the forward problem. All three are here, and
:meth:`MaximumAPosteriori.log_posterior` is the second, so adding such a sampler
later means adding the sampler and nothing else.
"""

from __future__ import annotations

from typing import Any

from ..algebra.operators import Functional, LinearFunctional
from ..algebra.spaces import HilbertSpace
from ..numerics.optimisation import LBFGS, Optimiser
from ..numerics.solvers import LinearSolver
from ..probability.gaussian import GaussianMeasure
from .normal import NormalOperator
from .problem import ForwardProblem

__all__ = ["MaximumAPosteriori", "LaplaceResult"]


class LaplaceResult:
    """The mode, the Gaussian about it, and what the search cost."""

    def __init__(
        self,
        model: Any,
        measure: GaussianMeasure,
        optimisation: Any,
        /,
    ) -> None:
        self._model = model
        self._measure = measure
        self._optimisation = optimisation

    @property
    def model(self) -> Any:
        """The maximum a posteriori model: the mode, not the mean.

        For a nonlinear problem these differ, and it is the mode that an
        optimiser finds. Saying which is which matters because every linear
        method here returns a mean, and the two are quietly interchangeable
        only in the linear case.
        """
        return self._model

    @property
    def measure(self) -> GaussianMeasure:
        """The Laplace approximation: a Gaussian centred on the mode.

        Its covariance is the inverse of the linearised normal operator, and
        so describes the posterior's curvature *at the mode* and nothing about
        its shape elsewhere.
        """
        return self._measure

    @property
    def optimisation(self) -> Any:
        """The optimiser's own result.

        Worth looking at rather than a formality: a MAP model from a search
        that did not converge is not the mode, and the Gaussian built on it is
        a curvature at the wrong point.
        """
        return self._optimisation

    @property
    def converged(self) -> bool:
        """Whether the search reached the mode."""
        return bool(self._optimisation.converged)

    def __repr__(self) -> str:
        return (
            f"LaplaceResult(converged={self.converged}, "
            f"iterations={self._optimisation.iterations})"
        )


class MaximumAPosteriori:
    """The MAP model of a nonlinear problem, with a Gaussian about it.

    Args:
        problem: the forward problem. Its operator may be nonlinear; it needs
            to supply a derivative, which :meth:`~pygeoinf2.algebra.operators.Operator.at`
            provides.
        prior: a Gaussian prior on the model space.
        optimiser: how to find the mode. L-BFGS by default, which needs only
            gradients -- the exact Hessian of a nonlinear misfit carries a
            second-derivative term that a forward operator is rarely asked to
            supply. Pass a Newton method if it can.
        solver: how to invert the linearised normal operator for the
            covariance. Left to :class:`~pygeoinf2.inference.gaussian.LinearGaussianInversion`'s
            own default if omitted.

    Raises:
        ValueError: if the prior does not live on the model space, if the
            problem has no error measure, or if the prior has no precision --
            the objective is built from ``P``, and a prior that cannot supply
            one cannot state its own contribution to it.
    """

    def __init__(
        self,
        problem: ForwardProblem,
        prior: GaussianMeasure,
        /,
        *,
        optimiser: Optimiser | None = None,
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
        self._optimiser = optimiser or LBFGS(max_iterations=500)
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

        Both halves are already available -- the first from the problem, the
        second from the prior -- so this assembles rather than derives. Its
        gradient is ``-2 J* R^-1 r + 2 P (m - m0)`` with ``J`` the derivative
        at the point, which is where the nonlinearity enters and the only
        place it does.

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

        def value(model: Any) -> float:
            offset = space.subtract(model, expectation)
            return self._problem.chi_squared(model, data) + float(
                space.inner_product(precision(offset), offset)
            )

        def derivative(model: Any) -> LinearFunctional:
            linearisation = forward.at(model)
            residual = data_space.subtract(data, linearisation.value)
            weighted = error.precision(
                data_space.subtract(residual, error.expectation)
            )
            offset = space.subtract(model, expectation)
            gradient = space.axpy(
                2.0,
                precision(offset),
                space.scale(-2.0, linearisation.derivative.adjoint(weighted)),
            )
            return LinearFunctional.from_representer(space, gradient)

        return Functional.from_callables(space, value, derivative=derivative)

    def log_posterior(self, data: Any, /) -> Functional:
        """The log posterior density, up to an additive constant.

        Minus half :meth:`objective`. Offered separately because it is what a
        sampler wants and the optimiser wants the other sign -- and because
        writing ``-0.5 *`` at the call site is exactly the kind of factor that
        goes missing.

        Args:
            data: the observations.

        Returns:
            A functional on the model space, with a gradient.
        """
        return -0.5 * self.objective(data)

    def linearised_normal(self, model: Any, /) -> NormalOperator:
        """The normal operator of the problem linearised at a model.

        ``P + J* R^-1 J`` in the model-space formalism, with ``J`` the
        derivative there. This is the curvature of :meth:`objective` up to a
        factor of two -- up to the Gauss-Newton approximation, which drops the
        term in the operator's *second* derivative. That term vanishes at a
        perfect fit and is small near a good one, and a forward operator is
        rarely asked to supply it.

        Args:
            model: where to linearise.

        Returns:
            The normal operator, which carries its own factors.
        """
        return NormalOperator(
            self._problem.forward_operator.at(model).derivative,
            self._prior,
            error=self._problem.error_measure,
            formalism="model_space",
        )

    def at(self, model: Any, data: Any, /) -> GaussianMeasure:
        """The Laplace Gaussian about a *given* model, without searching.

        For when the mode is already known -- from a previous run, from a
        different method, or because the question is what the approximation
        looks like somewhere else. :meth:`__call__` is this composed with the
        search.

        Args:
            model: where to centre it.
            data: the observations.

        Returns:
            A Gaussian with that mean and the linearised posterior covariance.
        """
        from ..numerics.solvers import resolve_solver

        normal = self.linearised_normal(model)
        # In the model-space formalism the posterior covariance *is* N^-1, so
        # this is the same construction LinearGaussianInversion makes -- and
        # the same one, evaluated at the mode rather than assumed everywhere.
        # The precision is N itself, which is worth keeping: it is what a
        # sampler and a Mahalanobis distance want, and it is exact where the
        # inverse is a solve.
        inverse = resolve_solver(self._solver, normal)(normal)
        return GaussianMeasure(
            self.model_space,
            covariance=inverse,
            precision=normal,
            expectation=model,
        )

    def __call__(self, data: Any, /, *, start: Any = None) -> LaplaceResult:
        """Find the mode, and put a Gaussian on it.

        Args:
            data: the observations.
            start: where the search begins. The prior's mean by default, which
                is the honest starting guess -- it is what is believed before
                the data are seen.

        Returns:
            The mode, the Gaussian, and the optimiser's own result. The last
            is worth reading: a search that did not converge gives a curvature
            at the wrong point, and this does not raise on one, because a
            stalled search still says something and refusing to return it
            would say less.
        """
        origin = self._prior.expectation if start is None else start
        outcome = self._optimiser.minimise(self.objective(data), origin)
        return LaplaceResult(
            outcome.minimiser, self.at(outcome.minimiser, data), outcome
        )

    def __repr__(self) -> str:
        return f"MaximumAPosteriori({self.model_space!r} -> {self.data_space!r})"
