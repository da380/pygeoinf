"""Nonlinear inference: the MAP model, and the Gaussian that sits on it.

The rest of :mod:`pygeoinf.inference` assumes the forward operator is linear,
which makes the posterior Gaussian and its expectation an affine function of
the data. When the operator is nonlinear, neither holds, and the standard
first thing to do is the two-step this module implements:

1. **Find the mode.** Minimize ``chi^2(m, d) + (m - m0, P (m - m0))``, which is
   twice the negative log posterior. That is an ordinary optimization, which
   :class:`MaximumAPosteriori` runs with the optimizers in
   :mod:`pygeoinf.numerics.optimization`, and it is a point estimator: data
   in, the mode out.
2. **Put a Gaussian on it.** Expand the same functional to second order about
   the minimizer. The curvature is half its Hessian there: the normal
   operator of the *linearized* problem when the Hessian is Gauss-Newton,
   which makes the covariance exactly the linear answer evaluated at the
   mode rather than assumed everywhere, and that plus the forward operator's
   curvature when the exact Hessian is carried.
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

from numpy.random import Generator

from typing import TYPE_CHECKING, Any

from ..algebra.operators import Functional, LinearOperator
from ..algebra.spaces import HilbertSpace
from ..numerics.normal import NormalOperator
from ..numerics.optimization import OptimizationResult, Optimizer
from ..numerics.solvers import LinearSolver
from ..probability.gaussian import GaussianMeasure
from ..traits import Traits
from .estimators import MeasureEstimator, _PushedMeasureEstimator
from .minimization import Minimization, _gauss_newton, _misfit_plus_quadratic
from .minimization import _mixed_derivative as _misfit_mixed_derivative
from .problem import ForwardProblem

if TYPE_CHECKING:  # pragma: no cover
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
            to supply a derivative, which :meth:`~pygeoinf.algebra.operators.Operator.at`
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
            :class:`~pygeoinf.numerics.solvers.LinearSolver`, or a callable
            receiving the Hessian and returning one. Conjugate gradients if
            omitted.
        gauss_newton: carry the Gauss-Newton Hessian ``2 P + 2 J* R^-1 J``
            in the objective whether or not the forward operator has a
            second derivative, dropping the operator's curvature by choice.
            Whichever Hessian the objective carries is also the one behind
            the derivative with respect to the data and behind the Laplace
            approximation.

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
                "as a Sobolev or heat measure of the space carries."
            )

        self._problem = problem
        self._prior = prior
        # The prior's domain rather than the forward operator's: the two
        # compare equal, and the prior's carries the labels of its summands,
        # which a block operator's domain does not.
        super().__init__(
            problem.data_space,
            prior.domain,
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
        near a good one. It is the curvature of the Laplace approximation
        whenever the Hessian is Gauss-Newton: with ``gauss_newton``, or when
        the forward operator has no second derivative.

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


def _exact_hessian(estimator: MaximumAPosteriori) -> bool:
    """Whether the estimator's objective carries an exact Hessian that differs
    from the Gauss-Newton one: a forward operator with a curvature of its
    own, and Gauss-Newton not asked for. A linear operator's curvature is
    zero, and its exact Hessian is the Gauss-Newton one."""
    forward = estimator.forward_problem.forward_operator
    return (
        not estimator._gauss_newton
        and forward.has_second_derivative
        and not isinstance(forward, LinearOperator)
    )


def _linearized_sampler(normal: NormalOperator, gain: LinearOperator, /) -> Any:
    """Randomize-then-optimize for the linearized problem: a callable taking a
    generator and returning one draw of the fluctuation about the mode.

    A prior draw ``x`` and a noise draw ``e``, both centered, give
    ``x + K (-J x - e)`` with ``K`` the linearized gain, whose covariance is
    the inverse of ``P + J* R^-1 J``; nothing is factorized.
    """
    prior, error, forward = normal.prior, normal.error, normal.forward
    model_space, data_space = forward.domain, forward.codomain

    def draw(rng: Generator | None, /) -> Any:
        drawn = model_space.subtract(prior.sample(rng=rng), prior.expectation)
        noise = data_space.subtract(error.sample(rng=rng), error.expectation)
        residual = data_space.negative(data_space.add(forward(drawn), noise))
        return model_space.add(drawn, gain(residual))

    return draw


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

        Its covariance is the inverse of the curvature at the mode, half the
        Hessian of the MAP objective there, and so describes the posterior's
        shape *at the mode* and nothing about it elsewhere.
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
    curvature. Called on data it finds the mode and returns the Gaussian
    centered there whose precision is the curvature at the mode, half the
    Hessian of the MAP objective: the exact Hessian when the estimator's
    objective carries it, the forward operator's curvature weighted by the
    residual included, and otherwise the Gauss-Newton ``P + J* R^-1 J``, the
    linear posterior's precision evaluated at one point instead of assumed
    everywhere. Which one follows the estimator's ``gauss_newton``, as its
    derivative with respect to the data does. That is what the Laplace
    approximation is: a point estimate with the curvature at it, and
    wrapping the point estimator says so.

    The result is a :class:`~pygeoinf.inference.gaussian.GaussianPosterior`
    whose two readings deliberately differ: the Gaussian moments are the
    *approximation*, the curvature at the mode, while
    ``relative_log_density`` is the **exact** posterior's log density
    relative to the prior, ``-Phi`` with ``Phi`` the likelihood potential of
    :meth:`~pygeoinf.inference.problem.ForwardProblem.potential`. A sampler
    takes the pair whole: exact target, Gaussian approximation. The
    normalizer is the Laplace estimate of the evidence,

    .. code-block:: text

        log Z == -J(m) - log det(I + H) / 2

    with ``J(m)`` the penalized misfit at the mode, ``Phi`` plus half the
    squared prior norm about the prior's expectation, and ``H`` the misfit's
    curvature seen through the prior covariance ``Q``, computed on first use
    by :func:`~pygeoinf.numerics.functional_calculus.log_determinant`. For
    the Gauss-Newton curvature that is done in the data space, as
    ``log |F' Q F'* + R| - log |R|`` with ``F'`` the forward operator's
    derivative at the mode; for the exact Hessian it is
    ``log |K| + log |Q|``, with ``K`` the curvature. The Gauss-Newton
    Gaussian samples by randomize-then-optimize, as the linear posterior
    does, with the linearized gain through the solver; the exact-Hessian one
    has no such route and does not sample.

    With ``rank=`` the result is a
    :class:`~pygeoinf.inference.gaussian.LowRankGaussianPosterior` built
    from the leading eigenpairs of the misfit's curvature seen through the
    prior's covariance factor. For the exact Hessian that curvature need not
    be positive semi-definite: its negative eigenvalues are clipped at zero,
    so a direction in which the forward operator's curvature would widen the
    prior keeps the prior instead.
    """

    def __init__(
        self,
        estimator: MaximumAPosteriori,
        /,
        *,
        solver: LinearSolver | None = None,
        rank: int | None = None,
        rng: Generator | None = None,
        n_jobs: int | None = None,
    ) -> None:
        """
        Args:
            estimator: the MAP estimator supplying the mode and the
                curvature.
            solver: how to invert the curvature for the covariance: a
                :class:`~pygeoinf.numerics.solvers.LinearSolver`, or a
                callable receiving the curvature and returning one.
                Conjugate gradients if omitted.
            rank: keep only the leading directions the data inform, and
                return a
                :class:`~pygeoinf.inference.gaussian.LowRankGaussianPosterior`
                in place of the full Gaussian: the prior's factors with
                rank-``rank`` corrections, no solve behind the covariance,
                and the Laplace evidence in closed form. The full Gaussian,
                through the solver, if omitted.
            rng: the generator for the randomized eigendecomposition behind
                *rank*.
            n_jobs: workers for it.

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
        self._rank = rank
        self._rng = rng
        self._n_jobs = n_jobs

    @classmethod
    def from_problem(
        cls,
        problem: ForwardProblem,
        prior: GaussianMeasure,
        /,
        *,
        optimizer: Optimizer | None = None,
        solver: LinearSolver | None = None,
        gauss_newton: bool = False,
        rank: int | None = None,
        rng: Generator | None = None,
        n_jobs: int | None = None,
    ) -> "LaplaceApproximation":
        """The approximation for a problem and a prior, building the MAP inside.

        Args:
            problem: the forward problem, with a Gaussian error measure.
            prior: a Gaussian prior on the model space, with a precision.
            optimizer: how the mode is found; see :class:`MaximumAPosteriori`.
            solver: how the curvature is inverted, for the covariance and for
                the MAP estimator's own derivative.
            gauss_newton: carry the Gauss-Newton Hessian whether or not the
                forward operator has a second derivative; see
                :class:`MaximumAPosteriori`. It is then the curvature of the
                approximation too.
            rank: the low-rank form; see the constructor.
            rng: the generator for its eigendecomposition.
            n_jobs: workers for it.

        Returns:
            The estimator.
        """
        return cls(
            MaximumAPosteriori(
                problem,
                prior,
                optimizer=optimizer,
                solver=solver,
                gauss_newton=gauss_newton,
            ),
            solver=solver,
            rank=rank,
            rng=rng,
            n_jobs=n_jobs,
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
        search. The curvature, half the MAP objective's Hessian, is taken to
        be positive definite, which it is at a strict minimizer and need not
        be elsewhere when the Hessian is exact.

        Args:
            model: where to center it.
            data: the observations.

        Returns:
            A Gaussian with that mean and the inverse of the curvature there
            as its covariance, carrying the exact relative density and the
            Laplace evidence as its normalizer; the low-rank form when a rank
            was given.

        Raises:
            ValueError: on the low-rank route, if the error measure has no
                precision or the prior no covariance factor.
        """
        from ..numerics.solvers import resolve_solver
        from .gaussian import GaussianPosterior

        estimator = self._estimator
        problem = estimator.forward_problem
        prior = estimator.prior
        exact = _exact_hessian(estimator)
        relative_log_density = -problem.potential(data)
        if self._rank is not None:
            return self._low_rank(model, data, exact, relative_log_density)

        sample = None
        if exact:
            curvature = (estimator.hessian(data, model) * 0.5).with_traits(
                Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
            )
            inverse = resolve_solver(self._solver, curvature)(curvature)
        else:
            # In the model-space formalism the posterior covariance *is*
            # N^-1, the construction LinearGaussianInversion makes, here
            # evaluated at this model. The precision is N itself, which is
            # worth keeping: it is what a Mahalanobis distance wants, and it
            # is exact where the inverse is a solve.
            curvature = estimator.linearized_normal(model)
            inverse = resolve_solver(self._solver, curvature)(curvature)
            if prior.can_sample and problem.error_measure.can_sample:
                sample = _linearized_sampler(
                    curvature, curvature.kalman_operator(inverse)
                )
        return GaussianPosterior(
            self.model_space,
            covariance=inverse,
            precision=curvature,
            expectation=model,
            sample=sample,
            reference=prior,
            relative_log_density=relative_log_density,
            log_normalizer=self._log_normalizer(
                model, relative_log_density, curvature, exact
            ),
        )

    def _low_rank(
        self, model: Any, data: Any, exact: bool, relative_log_density: Functional, /
    ) -> "GaussianPosterior":
        """The low-rank route of :meth:`at`."""
        from .gaussian import LowRankGaussianPosterior, _information_eigenpairs

        estimator = self._estimator
        problem = estimator.forward_problem
        if exact:
            # Half the misfit's own Hessian: the Gauss-Newton term and the
            # forward operator's curvature weighted by the residual, built
            # without the prior's precision rather than by subtracting it.
            misfit = _misfit_plus_quadratic(problem, data)
            curvature = misfit.hessian(model) * 0.5
        else:
            error_precision = problem.error_measure.precision
            if error_precision is None:
                raise ValueError("The error measure needs a precision.")
            jacobian = problem.forward_operator.derivative(model)
            curvature = jacobian.adjoint @ error_precision @ jacobian
        directions, eigenvalues = _information_eigenpairs(
            curvature,
            estimator.prior,
            rank=self._rank,
            rng=self._rng,
            n_jobs=self._n_jobs,
            semidefinite=not exact,
        )
        return LowRankGaussianPosterior(
            estimator.prior,
            model,
            eigenvalues,
            directions,
            relative_log_density=relative_log_density,
        )

    def _log_normalizer(
        self,
        model: Any,
        relative_log_density: Functional,
        curvature: LinearOperator,
        exact: bool,
        /,
    ) -> Any:
        """The Laplace evidence ``-J(m) - log det(I + H) / 2``, lazily."""
        estimator = self._estimator
        problem = estimator.forward_problem
        prior = estimator.prior
        space = self.model_space

        def compute() -> Any:
            from ..numerics.functional_calculus import log_determinant
            from ..probability.absolutely_continuous import combine_log_normalizer

            offset = space.subtract(model, prior.expectation)
            penalized = -float(relative_log_density(model)) + 0.5 * float(
                space.inner_product(prior.precision(offset), offset)
            )
            definite = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
            error = problem.error_measure
            if not exact and prior.covariance is not None:
                # det(I + H) == |J Q J* + R| / |R|: data-space quantities,
                # as for the linear posterior's evidence.
                data_normal = NormalOperator(
                    curvature.forward, prior, error=error, formalism="data_space"
                )
                volume = [
                    (-0.5, log_determinant(data_normal)),
                    (0.5, log_determinant(error.covariance.with_traits(definite))),
                ]
            else:
                # det(I + H) == |K| |Q|, Sylvester's identity in the model
                # space, with K the curvature.
                volume = [(-0.5, log_determinant(curvature))]
                if prior.covariance is not None:
                    covariance = prior.covariance.with_traits(definite)
                    volume.append((-0.5, log_determinant(covariance)))
                else:
                    precision = prior.precision.with_traits(definite)
                    volume.append((0.5, log_determinant(precision)))
            return combine_log_normalizer([(-1.0, penalized), *volume])

        return compute

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
            The measure estimator of the property. Its call takes the same
            ``start=`` as this one's.

        Raises:
            ValueError: if ``T`` does not act on the model space.
        """
        return _PushedMeasureEstimator(self, operator)

    def __repr__(self) -> str:
        return f"LaplaceApproximation({self.data_space!r} -> {self.model_space!r})"
