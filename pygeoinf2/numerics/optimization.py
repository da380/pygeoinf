"""
Unconstrained optimization, coordinate-free.

Written against the space's own inner product rather than wrapped around
SciPy. A gradient is a vector, a direction is a vector, and the slope along a
direction is their inner product, so there is no component array to put in
the wrong convention. A component-space optimizer can be handed gradient
components where it expects derivative components, beside a Hessian in
Galerkin form; a Newton step then solves ``H p = -G^-1 g`` instead of
``H p = -g``. Here that mistake cannot be made.

Two further consequences of working in the space rather than in components:

- **Convergence is mesh-independent.** Progress is judged on ``||grad f||`` in
  the space's own norm, which means the same thing under refinement, unlike the
  Euclidean norm of a component array.
- **Quasi-Newton methods become metric-aware for free.** The L-BFGS two-loop
  recursion is built entirely from inner products, so working in the space's
  metric preconditions it by the inverse Gram matrix at no cost. A component
  version is blind to the metric, and conditioned accordingly.
"""

from __future__ import annotations

import copy
import dataclasses
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from ..algebra.linearization import QuadraticModel
from ..algebra.operators import (
    EvaluationError,
    Functional,
    LinearFunctional,
    LinearOperator,
    Operator,
)
from ..traits import Traits
from .line_search import (
    ArmijoLineSearch,
    LineSearch,
    StrongWolfeLineSearch,
)

__all__ = [
    "truncated_cg",
    "OptimizationResult",
    "Optimizer",
    "SteepestDescent",
    "NonlinearCG",
    "LBFGS",
    "NewtonCG",
    "TrustRegionNewton",
    "AugmentedLagrangian",
    "AugmentedLagrangianResult",
    "gauss_newton_hessian",
]


@dataclass(frozen=True)
class OptimizationResult:
    """Where an optimizer stopped, and why."""

    minimizer: Any
    value: float
    gradient_norm: float
    iterations: int
    evaluations: int
    converged: bool
    message: str
    history: list[float] = field(default_factory=list)
    failure: str | None = None
    """What an :class:`~pygeoinf2.algebra.operators.EvaluationError` said,
    when one ended the run. The rest of the result is still the last
    good state -- ``minimizer`` was evaluated successfully -- so a run of
    hours is returned, not lost. ``None`` on every other outcome."""
    inner_stops: tuple[str, ...] = ()
    """Why each inner solve stopped, for the Newton methods that run
    truncated conjugate gradients inside every iteration: ``"converged"``,
    ``"negative curvature"``, ``"boundary"`` or ``"iteration limit"``, one
    entry per outer iteration. A run that keeps meeting negative curvature
    is being steered by the exits, not by the Hessian, which the final
    message does not say. Empty for the other optimizers."""

    def __repr__(self) -> str:
        return (
            f"OptimizationResult(value={self.value:.6g}, "
            f"|grad|={self.gradient_norm:.3g}, iterations={self.iterations}, "
            f"converged={self.converged}: {self.message})"
        )


_PRECISION_MARGIN = 4.0
"""How far above the bare rounding floor a failed line search still counts
as having converged; see :meth:`Optimizer._precision_limited`."""


class Optimizer(ABC):
    """Minimizes a functional on a Hilbert space."""

    def __init__(
        self,
        /,
        *,
        gtol: float = 1e-8,
        rtol: float = 1e-10,
        ftol: float = 1e-12,
        max_iterations: int = 500,
        line_search: LineSearch | None = None,
    ) -> None:
        """
        Args:
            gtol: absolute tolerance on ``||grad f||`` in the space's norm.
            rtol: tolerance relative to the initial gradient norm. The test is
                ``||g|| <= max(gtol, rtol * ||g0||)``.

                The default is tighter than a line-search method can always
                reach in double precision. Such a method then finishes in one
                of two other ways, both reported as converged: the value
                stops decreasing (*ftol*), or no step is found with the
                gradient already as small as the arithmetic allows (the
                message says "no further decrease is representable").
            ftol: stop when one iteration decreases the value by no more than
                ``ftol * |f|``. The criterion that says "no further progress",
                which a gradient test alone cannot: near a minimizer the
                gradient is small everywhere the value is flat.

                Relative to ``|f|`` with no absolute floor, deliberately: where
                the minimum value is near zero a floor would make the test
                absolute, and a slow method in a narrow valley -- the
                Rosenbrock function is the standard example -- would be
                stopped far from the minimizer. Set it to zero to rely on the
                gradient test alone. Every method applies it to its accepted
                steps; a trust-region step that was rejected made no
                progress to measure and is not tested.
            max_iterations: iteration cap.
            line_search: the step rule. Each method supplies a suitable default.
        """
        self._gtol = gtol
        self._rtol = rtol
        self._ftol = ftol
        self._max_iterations = max_iterations
        self._line_search = line_search or self._default_line_search()

    def _default_line_search(self) -> LineSearch:
        """The line search this method wants when none is supplied."""
        return ArmijoLineSearch()

    def minimize(self, functional: Functional, x0: Any, /) -> OptimizationResult:
        """Minimize ``functional`` starting from ``x0``.

        Args:
            functional: what to minimize. What it must supply -- a gradient, a
                Hessian, a subgradient, a proximal operator -- depends on the
                method; each says so.
            x0: where to start.

        Returns:
            The optimization result, whose ``converged`` and ``message`` say
            how it stopped.

        Raises:
            ValueError: if the functional cannot supply what the method needs.
        """
        if not functional.has_derivative:
            raise ValueError(
                f"{type(self).__name__} needs a functional with a derivative. "
                f"Supply one to Functional.from_callables."
            )
        return self._minimize(functional, x0)

    def with_tolerance(
        self,
        /,
        *,
        gtol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
    ) -> "Optimizer":
        """The same method with some of its stopping tolerances changed.

        What an outer loop wants when it drives this method as an inner
        solve whose accuracy tightens as the outer iteration proceeds: the
        method, its memory and its line search are kept, and only the
        tolerances named here move.

        Args:
            gtol: the absolute gradient tolerance, if it is to change.
            rtol: the relative one, if it is to change.
            ftol: the value tolerance, if it is to change.

        Returns:
            A copy of this optimizer with the new tolerances.
        """
        clone = copy.copy(self)
        if gtol is not None:
            clone._gtol = gtol
        if rtol is not None:
            clone._rtol = rtol
        if ftol is not None:
            clone._ftol = ftol
        return clone

    @abstractmethod
    def _minimize(self, functional: Functional, x0: Any) -> OptimizationResult:
        """Run the method, having passed validation."""

    # -- shared machinery ----------------------------------------------

    def _tolerance(self, initial_gradient_norm: float) -> float:
        return max(self._gtol, self._rtol * initial_gradient_norm)

    def _precision_limited(
        self, value: float, gradient_norm: float, *, curvature: float | None = None
    ) -> bool:
        """Whether the gradient is as small as double precision allows here.

        A line search fails when no step along a descent direction improves the
        value. Near a minimizer that is not a defect: the decrease a step can
        make, about ``||g||^2 / (2 L)`` for a curvature ``L`` along the
        direction, eventually falls below the rounding error in the value
        itself, which is ``eps * |f|`` rather than ``eps``. Past that point
        the search is comparing noise, so a failure with the gradient already
        at that scale is success at the accuracy the arithmetic allows, and
        saying "line search failed" instead reports a wrong answer about a
        right one.

        The floor is therefore ``sqrt(eps (1 + |f|) L)``, times a margin of
        four for the search's own inefficiency: on a quadratic the methods
        here stop between one and four times the bare floor, and a verdict
        that turned on which side of it the last failed step landed would
        change with the platform's arithmetic. Without a curvature estimate
        ``L`` is taken as ``1 + |f|``, which is right when the objective and
        its curvature share one scale and loose when the objective carries a
        large constant.

        Args:
            value: ``f`` at the point.
            gradient_norm: ``||g||`` there, in the space's norm.
            curvature: an estimate of ``L``; see :meth:`_curvature`.
        """
        if curvature is None or not np.isfinite(curvature) or curvature <= 0.0:
            curvature = 1.0 + abs(value)
        eps = float(np.finfo(float).eps)
        floor = _PRECISION_MARGIN * float(np.sqrt(eps * (1.0 + abs(value)) * curvature))
        return gradient_norm <= floor

    @staticmethod
    def _curvature(
        space: Any,
        model: Any,
        gradient: Any,
        gradient_norm: float,
        *,
        previous_x: Any = None,
        previous_gradient: Any = None,
    ) -> float | None:
        """An estimate of the curvature where a line search has just failed.

        The Rayleigh quotient of the Hessian along the gradient when the
        functional has one, since that is the curvature a steepest-descent
        step meets; otherwise the secant estimate from the last accepted
        step, ``||g - g_prev|| / ||x - x_prev||``; ``None`` on the first
        iteration of a functional without a Hessian.
        """
        if gradient_norm > 0.0 and getattr(model, "has_hessian", False):
            quotient = space.inner_product(gradient, model.hessian(gradient))
            return float(quotient) / gradient_norm**2
        if previous_x is None or previous_gradient is None:
            return None
        length = space.norm(space.subtract(model.point, previous_x))
        if length == 0.0:
            return None
        return float(space.norm(space.subtract(gradient, previous_gradient)) / length)

    def _finish_search_failure(
        self,
        x: Any,
        value: float,
        gradient_norm: float,
        iteration: int,
        evaluations: int,
        history: list[float],
        *,
        curvature: float | None = None,
    ) -> OptimizationResult:
        """The outcome when no step improved the value."""
        limited = self._precision_limited(value, gradient_norm, curvature=curvature)
        return OptimizationResult(
            x,
            value,
            gradient_norm,
            iteration,
            evaluations,
            limited,
            (
                "no further decrease is representable in double precision"
                if limited
                else "line search failed to find a suitable step"
            ),
            history,
        )

    def _finish_evaluation_failure(
        self,
        x: Any,
        value: float,
        gradient_norm: float,
        iteration: int,
        evaluations: int,
        history: list[float],
        error: Exception,
    ) -> OptimizationResult:
        """The outcome when an accepted point could not be evaluated.

        The line searches and the trust region absorb an
        :class:`~pygeoinf2.algebra.operators.EvaluationError` at a *trial*
        point by retreating; this is for the one place there is nowhere to
        retreat to. The result carries the last good state, so nothing is
        lost but the step that failed.
        """
        return OptimizationResult(
            x,
            value,
            gradient_norm,
            iteration,
            evaluations,
            False,
            "the functional could not be evaluated at an accepted point",
            history,
            failure=str(error),
        )


class _DescentMethod(Optimizer):
    """A method that repeatedly picks a direction and searches along it.

    The loop is identical across steepest descent, nonlinear CG and L-BFGS;
    only the direction rule differs, so only that is left to subclasses.
    """

    def _minimize(self, functional: Functional, x0: Any) -> OptimizationResult:
        space = functional.domain
        x = space.copy(x0)
        model = functional.at(x)
        gradient = model.gradient
        gradient_norm = space.norm(gradient)
        tolerance = self._tolerance(gradient_norm)

        history = [model.value]
        evaluations = 1
        state = self._initial_state(space)
        previous: tuple[float, float] | None = None
        previous_x: Any = None
        previous_gradient: Any = None

        for iteration in range(1, self._max_iterations + 1):
            if gradient_norm <= tolerance:
                return OptimizationResult(
                    x,
                    model.value,
                    gradient_norm,
                    iteration - 1,
                    evaluations,
                    True,
                    "gradient tolerance reached",
                    history,
                )

            direction, state = self._direction(space, gradient, state)
            slope = space.inner_product(gradient, direction)
            if slope >= 0.0:
                # A non-descent direction means the accumulated curvature
                # information has gone bad; steepest descent always works.
                direction = space.negative(gradient)
                slope = -(gradient_norm**2)
                state = self._initial_state(space)

            search = self._line_search(
                functional,
                x,
                direction,
                value=model.value,
                slope=slope,
                initial_step=self._initial_step(previous, slope),
            )
            evaluations += search.evaluations
            if not search.converged:
                return self._finish_search_failure(
                    x,
                    model.value,
                    gradient_norm,
                    iteration,
                    evaluations,
                    history,
                    curvature=self._curvature(
                        space,
                        model,
                        gradient,
                        gradient_norm,
                        previous_x=previous_x,
                        previous_gradient=previous_gradient,
                    ),
                )

            previous = (search.step, slope)
            step_vector = space.scale(search.step, direction)
            new_x = search.point
            # A Wolfe search hands back the model it already built to test the
            # curvature condition. Only a backtracking one, which never needs
            # a gradient, leaves us to evaluate here -- where the derivative
            # can fail at a point whose value did not.
            new_model = search.model
            if new_model is None:
                try:
                    new_model = functional.at(new_x)
                except EvaluationError as error:
                    return self._finish_evaluation_failure(
                        x,
                        model.value,
                        gradient_norm,
                        iteration,
                        evaluations,
                        history,
                        error,
                    )
                evaluations += 1
            new_gradient = new_model.gradient

            state = self._update(
                space,
                state,
                step_vector,
                space.subtract(new_gradient, gradient),
            )
            previous_value = model.value
            previous_x, previous_gradient = x, gradient
            x, model, gradient = new_x, new_model, new_gradient
            gradient_norm = space.norm(gradient)
            history.append(model.value)

            decrease = previous_value - model.value
            if 0.0 <= decrease <= self._ftol * abs(model.value):
                # Flat: the step was accepted but bought nothing. Continuing
                # would only spend evaluations to confirm it.
                return OptimizationResult(
                    x,
                    model.value,
                    gradient_norm,
                    iteration,
                    evaluations,
                    True,
                    "value tolerance reached",
                    history,
                )

        return OptimizationResult(
            x,
            model.value,
            gradient_norm,
            self._max_iterations,
            evaluations,
            gradient_norm <= tolerance,
            "gradient tolerance reached"
            if gradient_norm <= tolerance
            else "iteration limit reached",
            history,
        )

    def _initial_state(self, space: Any) -> Any:
        """Whatever the direction rule needs to carry between iterations."""
        return None

    def _initial_step(
        self, previous: tuple[float, float] | None, slope: float
    ) -> float:
        """The first step the line search should try.

        A unit step by default, which is what a Newton or quasi-Newton method
        wants: its direction already carries the scale, and trying anything
        else throws that information away.
        """
        return 1.0

    @abstractmethod
    def _direction(self, space: Any, gradient: Any, state: Any) -> tuple[Any, Any]:
        """The search direction, and the updated state."""

    def _update(self, space: Any, state: Any, step: Any, gradient_change: Any) -> Any:
        """Fold the completed step into the state."""
        return state


def _slope_ratio_step(previous: tuple[float, float] | None, slope: float) -> float:
    """Nocedal and Wright's initial-step heuristic for a scale-free direction."""
    if previous is None or slope == 0.0:
        return 1.0
    previous_step, previous_slope = previous
    return float(np.clip(previous_step * previous_slope / slope, 1e-12, 1e8))


class SteepestDescent(_DescentMethod):
    """The negative gradient direction.

    Note this is the gradient in the *space's* metric, so the direction is
    steepest with respect to the inner product the modeler chose rather than
    with respect to an arbitrary coordinate basis, and it is why this converges
    at a mesh-independent rate where a component version does not.
    """

    def _direction(self, space: Any, gradient: Any, state: Any) -> tuple[Any, Any]:
        return space.negative(gradient), state

    def _default_line_search(self) -> LineSearch:
        """A strong Wolfe search.

        A backtracking search cannot take a *larger* step than it is offered,
        and a steepest-descent direction carries no natural scale, so the step
        it is offered is a guess.
        """
        return StrongWolfeLineSearch()

    def _initial_step(
        self, previous: tuple[float, float] | None, slope: float
    ) -> float:
        """The slope-ratio heuristic.

        A steepest-descent direction carries no natural scale, so the previous
        step is the only information available: match the first-order change
        along the new direction to the change achieved along the last one.
        """
        return _slope_ratio_step(previous, slope)


class NonlinearCG(_DescentMethod):
    """Nonlinear conjugate gradients.

    Polak-Ribiere with the standard non-negative restart, which is what makes
    it globally convergent under a Wolfe line search; Fletcher-Reeves is
    available for comparison.
    """

    def __init__(
        self,
        /,
        *,
        variant: Literal["polak-ribiere", "fletcher-reeves"] = "polak-ribiere",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            variant: the formula for the conjugacy parameter.
            **kwargs: passed to :class:`Optimizer`.
        """
        if variant not in ("polak-ribiere", "fletcher-reeves"):
            raise ValueError(f"Unknown variant {variant!r}.")
        self._variant = variant
        super().__init__(**kwargs)

    def _default_line_search(self) -> LineSearch:
        # Nonlinear CG needs a tighter curvature condition than quasi-Newton.
        return StrongWolfeLineSearch(curvature=0.1)

    def _initial_state(self, space: Any) -> Any:
        return None

    def _initial_step(
        self, previous: tuple[float, float] | None, slope: float
    ) -> float:
        """As for steepest descent: a CG direction carries no natural scale."""
        return _slope_ratio_step(previous, slope)

    def _direction(self, space: Any, gradient: Any, state: Any) -> tuple[Any, Any]:
        if state is None:
            direction = space.negative(gradient)
            return direction, (gradient, direction)

        previous_gradient, previous_direction = state
        squared = space.squared_norm(previous_gradient)
        if squared == 0.0:
            direction = space.negative(gradient)
            return direction, (gradient, direction)

        if self._variant == "fletcher-reeves":
            beta = space.squared_norm(gradient) / squared
        else:
            change = space.subtract(gradient, previous_gradient)
            beta = max(0.0, space.inner_product(gradient, change) / squared)

        direction = space.axpy(beta, previous_direction, space.negative(gradient))
        return direction, (gradient, direction)


class LBFGS(_DescentMethod):
    """Limited-memory BFGS.

    The two-loop recursion is built from inner products and ``axpy`` alone, so
    it runs on a space with no component map — and, more importantly, the
    implicit inverse Hessian it accumulates is expressed in the space's metric.
    A component-space L-BFGS builds curvature information in whatever basis the
    discretization happened to supply, which is the ill-conditioning that makes
    the same problem converge differently under refinement.
    """

    def __init__(self, /, *, memory: int = 10, **kwargs: Any) -> None:
        """
        Args:
            memory: how many correction pairs to keep. Five to twenty is usual;
                more helps a badly scaled problem and costs one inner product
                per pair per iteration.
            **kwargs: passed to :class:`Optimizer`.
        """
        if memory < 1:
            raise ValueError("memory must be at least one.")
        self._memory = memory
        super().__init__(**kwargs)

    def _default_line_search(self) -> LineSearch:
        # The curvature condition is what keeps (y, s) positive, and so keeps
        # the implicit Hessian positive definite.
        return StrongWolfeLineSearch(curvature=0.9)

    def _initial_state(self, space: Any) -> Any:
        return deque(maxlen=self._memory)

    def _direction(self, space: Any, gradient: Any, state: Any) -> tuple[Any, Any]:
        pairs = state
        if not pairs:
            return space.negative(gradient), pairs

        q = space.copy(gradient)
        alphas = []
        for step, change, rho in reversed(pairs):
            alpha = rho * space.inner_product(step, q)
            q = space.axpy(-alpha, change, q)
            alphas.append(alpha)

        # The initial inverse-Hessian scaling, in the space's metric.
        last_step, last_change, _ = pairs[-1]
        scaling = space.inner_product(last_step, last_change) / space.squared_norm(
            last_change
        )
        r = space.scale_inplace(scaling, q)

        for (step, change, rho), alpha in zip(pairs, reversed(alphas)):
            beta = rho * space.inner_product(change, r)
            r = space.axpy(alpha - beta, step, r)

        return space.negative(r), pairs

    def _update(self, space: Any, state: Any, step: Any, gradient_change: Any) -> Any:
        curvature = space.inner_product(step, gradient_change)
        # Reject a pair that would make the implicit Hessian indefinite. This
        # is the standard safeguard, and a Wolfe line search should make it
        # unnecessary; it fires when the line search was inexact.
        if curvature > 1e-12 * space.norm(step) * space.norm(gradient_change):
            state.append((step, gradient_change, 1.0 / curvature))
        return state


class NewtonCG(Optimizer):
    """Newton's method with the step computed by truncated CG.

    The Newton system ``H p == -g`` is solved with conjugate gradients, stopped
    early on negative curvature. That is what makes the method usable when the
    Hessian is indefinite, which it generally is far from a minimum.

    The relationship to :class:`~pygeoinf2.numerics.solvers.CGSolver` is worth
    stating: that solver *refuses* an indefinite operator and raises when it
    meets negative curvature, which is right for solving a linear system and
    wrong here, where negative curvature is information about the direction to
    move in rather than a failure.
    """

    def __init__(
        self,
        /,
        *,
        forcing: float = 0.1,
        max_cg_iterations: int | None = None,
        preconditioner: LinearOperator | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            forcing: the inexact-Newton forcing term. The inner solve stops
                once the residual falls below ``min(forcing, sqrt(||g||))``
                times the gradient norm, so early iterations are cheap and
                later ones accurate.
            max_cg_iterations: cap on the inner iteration. ``None`` means the
                larger of the dimension and ten.
            preconditioner: an approximation to the inverse Hessian for the
                inner solve, self-adjoint and positive on the domain. For a
                posterior's Hessian the prior covariance is the natural
                choice: it leaves the inner solve with the data's spectrum
                only, where the Sobolev precision alone spans many decades.
            **kwargs: passed to :class:`Optimizer`.
        """
        self._forcing = forcing
        self._max_cg_iterations = max_cg_iterations
        self._preconditioner = preconditioner
        super().__init__(**kwargs)

    def minimize(self, functional: Functional, x0: Any, /) -> OptimizationResult:
        """Minimize, requiring a Hessian as well as a gradient.

        Args:
            functional: what to minimize, which must supply both.
            x0: where to start.

        Returns:
            The optimization result.

        Raises:
            ValueError: if the functional carries no Hessian.
        """
        if not functional.has_hessian:
            raise ValueError(
                "NewtonCG needs a functional with a Hessian. Use LBFGS when "
                "only a gradient is available, or supply a Gauss-Newton "
                "approximation with gauss_newton_hessian."
            )
        return super().minimize(functional, x0)

    def _minimize(self, functional: Functional, x0: Any) -> OptimizationResult:
        stops: list[str] = []
        result = self._run(functional, x0, stops)
        return dataclasses.replace(result, inner_stops=tuple(stops))

    def _run(
        self, functional: Functional, x0: Any, stops: list[str]
    ) -> OptimizationResult:
        space = functional.domain
        x = space.copy(x0)
        model = functional.at(x)
        gradient_norm = space.norm(model.gradient)
        tolerance = self._tolerance(gradient_norm)
        history = [model.value]
        evaluations = 1

        for iteration in range(1, self._max_iterations + 1):
            if gradient_norm <= tolerance:
                return OptimizationResult(
                    x,
                    model.value,
                    gradient_norm,
                    iteration - 1,
                    evaluations,
                    True,
                    "gradient tolerance reached",
                    history,
                )

            direction, reason = truncated_cg(
                model.hessian,
                space.negative(model.gradient),
                rtol=min(self._forcing, np.sqrt(gradient_norm)),
                max_iterations=self._max_cg_iterations,
                preconditioner=self._preconditioner,
            )
            stops.append(reason)
            slope = space.inner_product(model.gradient, direction)
            if slope >= 0.0:
                direction = space.negative(model.gradient)
                slope = -(gradient_norm**2)

            search = self._line_search(
                functional,
                x,
                direction,
                value=model.value,
                slope=slope,
                initial_step=1.0,
            )
            evaluations += search.evaluations
            if not search.converged:
                return self._finish_search_failure(
                    x,
                    model.value,
                    gradient_norm,
                    iteration,
                    evaluations,
                    history,
                    curvature=self._curvature(
                        space, model, model.gradient, gradient_norm
                    ),
                )

            previous_value = model.value
            previous_x = x
            x = search.point
            try:
                model = functional.at(x)
            except EvaluationError as error:
                return self._finish_evaluation_failure(
                    previous_x,
                    previous_value,
                    gradient_norm,
                    iteration,
                    evaluations,
                    history,
                    error,
                )
            evaluations += 1
            gradient_norm = space.norm(model.gradient)
            history.append(model.value)

            decrease = previous_value - model.value
            if 0.0 <= decrease <= self._ftol * abs(model.value):
                return OptimizationResult(
                    x,
                    model.value,
                    gradient_norm,
                    iteration,
                    evaluations,
                    True,
                    "value tolerance reached",
                    history,
                )

        return OptimizationResult(
            x,
            model.value,
            gradient_norm,
            self._max_iterations,
            evaluations,
            gradient_norm <= tolerance,
            "gradient tolerance reached"
            if gradient_norm <= tolerance
            else "iteration limit reached",
            history,
        )


class TrustRegionNewton(Optimizer):
    """Newton's method with a trust region, solved by Steihaug-CG.

    Preferred to a line search when the Hessian is badly indefinite: the region
    bounds the step directly rather than relying on a direction being a descent
    direction at all.
    """

    def __init__(
        self,
        /,
        *,
        initial_radius: float | None = None,
        max_radius: float | None = None,
        acceptance: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            initial_radius: the starting trust-region radius, a length in the
                space's norm. Unset, it is the length of the first Cauchy
                step, ``||g||^3 / (g, H g)``, the minimizer of the quadratic
                model along the gradient, which carries the problem's own
                units; where the curvature along the gradient is not
                positive, ``||x0||``, or one at the origin.
            max_radius: cap on the radius. Unset, a million times the
                initial radius.
            acceptance: reject a step whose actual-to-predicted reduction ratio
                falls below this.
            **kwargs: passed to :class:`Optimizer`, except ``line_search``:
                a trust region chooses its steps itself.

        Raises:
            TypeError: if a line search is passed.
        """
        if "line_search" in kwargs:
            raise TypeError(
                "TrustRegionNewton takes no line search: the trust region "
                "sets the length of every step. Use NewtonCG for a line-search "
                "Newton method."
            )
        self._initial_radius = initial_radius
        self._max_radius = max_radius
        self._acceptance = acceptance
        super().__init__(**kwargs)

    def minimize(self, functional: Functional, x0: Any, /) -> OptimizationResult:
        """Minimize, requiring a Hessian as well as a gradient.

        Args:
            functional: what to minimize, which must supply both.
            x0: where to start.

        Returns:
            The optimization result.

        Raises:
            ValueError: if the functional carries no Hessian.
        """
        if not functional.has_hessian:
            raise ValueError("TrustRegionNewton needs a functional with a Hessian.")
        return super().minimize(functional, x0)

    @staticmethod
    def _cauchy_length(space: Any, model: Any, gradient_norm: float, x0: Any) -> float:
        """The length of the minimizer of the quadratic model along ``-g``.

        ``||g||^3 / (g, H g)``: one Hessian application, and a length in the
        problem's own units. Where the curvature along the gradient is not
        positive the model has no minimizer along it, and the size of the
        start stands in, or one at the origin.
        """
        if gradient_norm > 0.0:
            curvature = space.inner_product(
                model.gradient, model.hessian(model.gradient)
            )
            if curvature > 0.0 and np.isfinite(curvature):
                return float(gradient_norm**3 / curvature)
        size = space.norm(x0)
        return float(size) if size > 0.0 else 1.0

    def _minimize(self, functional: Functional, x0: Any) -> OptimizationResult:
        stops: list[str] = []
        result = self._run(functional, x0, stops)
        return dataclasses.replace(result, inner_stops=tuple(stops))

    def _run(
        self, functional: Functional, x0: Any, stops: list[str]
    ) -> OptimizationResult:
        space = functional.domain
        x = space.copy(x0)
        model = functional.at(x)
        gradient_norm = space.norm(model.gradient)
        tolerance = self._tolerance(gradient_norm)
        radius = self._initial_radius
        if radius is None:
            radius = self._cauchy_length(space, model, gradient_norm, x)
        max_radius = 1e6 * radius if self._max_radius is None else self._max_radius
        # The region has collapsed once it is below rounding against the
        # problem's lengths: its starting radius and the iterate's own size.
        start = radius
        history = [model.value]
        evaluations = 1

        def collapsing() -> bool:
            return radius < 1e-14 * max(start, space.norm(x))

        def collapsed(iteration: int) -> OptimizationResult:
            # The region shrank below what a step can represent, so no
            # further progress is possible here. Converged exactly when the
            # gradient is as small as the arithmetic allows, as a failed
            # line search is judged; otherwise an honest stop short.
            limited = gradient_norm <= tolerance or self._precision_limited(
                model.value,
                gradient_norm,
                curvature=self._curvature(space, model, model.gradient, gradient_norm),
            )
            return OptimizationResult(
                x,
                model.value,
                gradient_norm,
                iteration,
                evaluations,
                limited,
                (
                    "no further decrease is representable in double precision"
                    if limited
                    else "the trust region collapsed before the gradient "
                    "tolerance was reached"
                ),
                history,
            )

        for iteration in range(1, self._max_iterations + 1):
            if gradient_norm <= tolerance:
                return OptimizationResult(
                    x,
                    model.value,
                    gradient_norm,
                    iteration - 1,
                    evaluations,
                    True,
                    "gradient tolerance reached",
                    history,
                )

            step, reason = truncated_cg(
                model.hessian,
                space.negative(model.gradient),
                rtol=min(0.1, np.sqrt(gradient_norm)),
                radius=radius,
            )
            stops.append(reason)
            predicted = -(
                space.inner_product(model.gradient, step)
                + 0.5 * space.inner_product(model.hessian(step), step)
            )
            if predicted <= 0.0:
                radius *= 0.25
                if collapsing():
                    return collapsed(iteration)
                continue

            trial = space.axpy(1.0, step, space.copy(x))
            try:
                trial_value = functional(trial)
            except EvaluationError:
                # An infeasible trial is a trust region's ordinary business:
                # the model was wrong there, so shrink and try again.
                evaluations += 1
                radius *= 0.25
                if collapsing():
                    return collapsed(iteration)
                continue
            evaluations += 1
            if not np.isfinite(trial_value):
                radius *= 0.25
                if collapsing():
                    return collapsed(iteration)
                continue
            ratio = (model.value - trial_value) / predicted

            if ratio < 0.25:
                radius *= 0.25
            elif ratio > 0.75 and abs(space.norm(step) - radius) < 1e-8 * radius:
                radius = min(2.0 * radius, max_radius)

            if ratio > self._acceptance:
                try:
                    accepted = functional.at(trial)
                except EvaluationError as error:
                    return self._finish_evaluation_failure(
                        x,
                        model.value,
                        gradient_norm,
                        iteration,
                        evaluations,
                        history,
                        error,
                    )
                previous_value = model.value
                x, model = trial, accepted
                evaluations += 1
                gradient_norm = space.norm(model.gradient)
                history.append(model.value)

                decrease = previous_value - model.value
                if 0.0 <= decrease <= self._ftol * abs(model.value):
                    # Flat: the step was accepted but bought nothing, as the
                    # line-search methods judge it.
                    return OptimizationResult(
                        x,
                        model.value,
                        gradient_norm,
                        iteration,
                        evaluations,
                        True,
                        "value tolerance reached",
                        history,
                    )

            if collapsing():
                return collapsed(iteration)

        return OptimizationResult(
            x,
            model.value,
            gradient_norm,
            self._max_iterations,
            evaluations,
            gradient_norm <= tolerance,
            "gradient tolerance reached"
            if gradient_norm <= tolerance
            else "iteration limit reached",
            history,
        )


def truncated_cg(
    hessian: LinearOperator,
    rhs: Any,
    /,
    *,
    rtol: float = 0.1,
    max_iterations: int | None = None,
    radius: float | None = None,
    preconditioner: LinearOperator | None = None,
) -> tuple[Any, str]:
    """Steihaug's truncated conjugate gradients.

    Solves ``H p == rhs`` approximately, stopping on any of three conditions:
    the residual falls below ``rtol``; a direction of negative curvature is
    met; or the iterate reaches the trust-region boundary. With a trust region,
    the last two return the step extended to the boundary along the current
    direction, which is what makes the step useful rather than an error.
    Without one, negative curvature returns the step taken so far, or ``rhs``
    itself if no step has been taken.

    Coordinate-free, like every other Krylov method here.

    Args:
        hessian: the (approximate) Hessian ``H``, self-adjoint but not
            necessarily definite -- indefiniteness is a case this handles
            rather than refuses.
        rhs: the right-hand side, usually the negated gradient.
        rtol: stop when the residual falls to this fraction of its initial
            norm. Loose by default: an inexact Newton step is the point, and
            solving accurately far from the optimum is wasted work.
        max_iterations: the cap. The larger of the dimension and ten, if
            omitted.
        radius: the trust-region radius. Unbounded if omitted, which makes
            this ordinary CG with an early stop.
        preconditioner: an approximation to ``H^-1``, self-adjoint and
            positive, applied to the residual at every iteration; the
            prior covariance is the natural one for a posterior's Hessian.
            Not combined with a trust region, since the region would then
            have to be measured in the preconditioner's norm.

    Returns:
        The step and a word saying which condition stopped it.

    Raises:
        ValueError: if both a radius and a preconditioner are given.
    """
    if radius is not None and preconditioner is not None:
        raise ValueError(
            "A trust region and a preconditioner are not combined here: the "
            "region would have to be measured in the preconditioner's norm."
        )
    space = hessian.domain
    limit = max_iterations if max_iterations is not None else max(space.dim, 10)

    def preconditioned(r: Any) -> Any:
        return r if preconditioner is None else preconditioner(r)

    step = space.zero()
    residual = space.copy(rhs)
    residual_norm = space.norm(residual)
    target = rtol * residual_norm

    if residual_norm == 0.0:
        return step, "converged"

    z = preconditioned(residual)
    direction = space.copy(z)
    rz = space.inner_product(residual, z)

    for _ in range(limit):
        curvature_vector = hessian(direction)
        curvature = space.inner_product(direction, curvature_vector)

        if curvature <= 0.0:
            if radius is None:
                # No region to stop at: fall back on the steepest direction.
                return (rhs if space.norm(step) == 0.0 else step), "negative curvature"
            return _to_boundary(space, step, direction, radius), "negative curvature"

        alpha = rz / curvature
        trial = space.axpy(alpha, direction, space.copy(step))

        if radius is not None and space.norm(trial) >= radius:
            return _to_boundary(space, step, direction, radius), "boundary"

        step = trial
        residual = space.axpy(-alpha, curvature_vector, residual)
        residual_norm = space.norm(residual)
        if residual_norm <= target:
            return step, "converged"

        z = preconditioned(residual)
        rz_new = space.inner_product(residual, z)
        beta = rz_new / rz
        rz = rz_new
        direction = space.axpy(1.0, z, space.scale_inplace(beta, direction))

    return step, "iteration limit"


def _to_boundary(space: Any, step: Any, direction: Any, radius: float) -> Any:
    """Extend a step to the trust-region boundary along a direction.

    Solves ``||step + t d|| == radius`` for the positive root, which is the
    standard Steihaug termination.
    """
    a = space.squared_norm(direction)
    b = 2.0 * space.inner_product(step, direction)
    c = space.squared_norm(step) - radius**2
    if a == 0.0:
        return step
    discriminant = max(b * b - 4.0 * a * c, 0.0)
    t = (-b + np.sqrt(discriminant)) / (2.0 * a)
    return space.axpy(t, direction, space.copy(step))


def gauss_newton_hessian(
    operator: Operator,
    point: Any,
    /,
    *,
    weighting: LinearOperator | None = None,
) -> LinearOperator:
    """The Gauss-Newton approximation ``J* W J`` to a misfit Hessian.

    For a misfit ``phi(m) == psi(F(m))`` the exact Hessian is
    ``J* H_psi J + sum_i r_i F_i''(m)``. This is the first term alone, which is
    what "Gauss-Newton" means, and it is offered under a name that says so
    rather than as a ``hessian`` that quietly omits a term.

    A composition of the form ``J* W J`` is recognised as self-adjoint by the
    trait propagation, and as positive semidefinite when ``W`` claims to be,
    with nothing claimed here -- so it can be handed to CG directly.

    Args:
        operator: the forward map ``F``.
        point: where to linearize.
        weighting: ``W``, typically an inverse data covariance. Defaults to the
            identity on the codomain.

    Returns:
        The operator ``J* W J``, carrying the positive-semidefinite trait when
        ``W`` claims it or is omitted.

    Raises:
        ValueError: if the weighting does not act on the operator's codomain,
            or does not claim to be self-adjoint.
        NotImplementedError: from the operator, if it carries no derivative.
    """
    jacobian = operator.derivative(point)
    if weighting is None:
        return jacobian.adjoint @ jacobian
    if weighting.domain != operator.codomain:
        raise ValueError(
            f"The weighting must act on the codomain {operator.codomain!r}, "
            f"not {weighting.domain!r}."
        )
    if Traits.SELF_ADJOINT & weighting.traits != Traits.SELF_ADJOINT:
        raise ValueError(
            f"The weighting must be self-adjoint; it claims {weighting.traits!s}."
        )
    return jacobian.adjoint @ weighting @ jacobian


@dataclass(frozen=True)
class AugmentedLagrangianResult(OptimizationResult):
    """Where a constrained minimization stopped, with its multipliers.

    Beyond the fields of :class:`OptimizationResult`, whose ``iterations``
    and ``evaluations`` are summed over the inner solves and whose
    ``gradient_norm`` is that of the augmented Lagrangian at the last
    iterate, the first-order measure of how far the point is from satisfying
    the optimality conditions:

    ``multipliers``
        the Lagrange multiplier of each equality constraint, in order.
    ``inequality_multipliers``
        the multiplier of each inequality constraint, in order: zero where
        the constraint is inactive and positive where it binds.
    ``violation``
        the largest constraint violation at the minimizer, ``max |c_i|`` over
        the equalities and ``max(g_j, 0)`` over the inequalities.
    ``penalty``
        the penalty parameter in force when the method stopped.
    ``outer_iterations``
        how many multiplier updates were made.
    """

    multipliers: np.ndarray = field(default_factory=lambda: np.zeros(0))
    inequality_multipliers: np.ndarray = field(default_factory=lambda: np.zeros(0))
    violation: float = 0.0
    penalty: float = 0.0
    outer_iterations: int = 0

    def __repr__(self) -> str:
        return (
            f"AugmentedLagrangianResult(value={self.value:.6g}, "
            f"violation={self.violation:.3g}, |grad L|={self.gradient_norm:.3g}, "
            f"outer_iterations={self.outer_iterations}, "
            f"iterations={self.iterations}, converged={self.converged}: "
            f"{self.message})"
        )


class AugmentedLagrangian(Optimizer):
    """Minimize ``f(x)`` subject to ``c_i(x) == 0`` and ``g_j(x) <= 0``.

    The method of multipliers. Each outer iteration minimizes the augmented
    Lagrangian

    .. code-block:: text

        L(x) = f(x) + sum_i [ lambda_i c_i(x) + (rho / 2) c_i(x)^2 ]
                    + sum_j [ max(0, mu_j + rho g_j(x))^2 - mu_j^2 ] / (2 rho)

    without constraints, by whatever :class:`Optimizer` was handed in, warm
    started from the previous iterate; then updates the multipliers,
    ``lambda_i <- lambda_i + rho c_i(x)`` and
    ``mu_j <- max(0, mu_j + rho g_j(x))``, and grows the penalty ``rho`` only
    when the constraint violation did not fall by the factor asked of it.
    The inner solve's gradient tolerance starts loose and tightens as the
    multipliers settle, so early outer iterations, whose minimizers will be
    discarded, are not solved to an accuracy nobody will use.

    The constraints belong to the method rather than to :meth:`minimize`, so
    that the contract every optimizer shares, ``minimize(functional, x0)``,
    is kept exactly: an augmented Lagrangian built for a set of constraints
    can be handed anywhere an optimizer is wanted, as a preconditioned solver
    built for one operator can be handed anywhere a solver is. A constrained
    problem whose constraints change with the data builds one per data
    vector, which costs nothing.

    Everything is coordinate-free: the multipliers are scalars, the penalty
    terms are squares of scalars, and the only vectors are the gradients of
    ``f`` and of the constraints in the space's own metric.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        /,
        *,
        equalities: Sequence[Functional] = (),
        inequalities: Sequence[Functional] = (),
        penalty: float = 10.0,
        penalty_growth: float = 10.0,
        reduction: float = 0.25,
        tolerance: float = 1e-6,
        initial_gtol: float = 1e-2,
        gtol_reduction: float = 0.1,
        gtol: float = 1e-8,
        max_iterations: int = 30,
    ) -> None:
        """
        Args:
            optimizer: the unconstrained method for the inner solves. Its own
                iteration budget bounds each inner solve; its gradient
                tolerance is overridden by the schedule described below. It
                needs whatever the augmented Lagrangian can supply: a Newton
                method is usable when the objective and every constraint
                carry a Hessian.
            equalities: the functionals ``c_i``, each with a derivative,
                constrained to vanish.
            inequalities: the functionals ``g_j``, each with a derivative,
                constrained to be at most zero.
            penalty: the initial penalty ``rho``. Too small and the early
                inner problems are barely constrained; too large and they are
                ill-conditioned. The default suits constraints scaled so that
                a unit violation is a large one; rescale a constraint rather
                than tune this.
            penalty_growth: the factor ``rho`` grows by when an outer
                iteration fails to reduce the violation enough.
            reduction: the factor the violation must fall by, outer iteration
                to outer iteration, for the penalty to stay put.
            tolerance: the constraint violation at which the method may stop:
                ``max |c_i|`` and ``max(g_j, 0)`` are both at most this.
            initial_gtol: the gradient tolerance of the first inner solve.
            gtol_reduction: the factor the inner gradient tolerance shrinks
                by after an outer iteration that reduced the violation
                enough, down to *gtol*.
            gtol: the gradient tolerance of the augmented Lagrangian the
                final inner solve must meet. With the multipliers converged
                this is the norm of the Lagrangian's gradient, the
                first-order optimality residual.
            max_iterations: the budget of outer iterations.

        Raises:
            ValueError: if a constraint carries no derivative, if the
                constraints do not all share one domain, or if a parameter
                is outside its range.
        """
        super().__init__(gtol=gtol, max_iterations=max_iterations)
        constraints = tuple(equalities) + tuple(inequalities)
        for constraint in constraints:
            if not isinstance(constraint, Functional) or not constraint.has_derivative:
                raise ValueError(
                    "Every constraint must be a Functional with a derivative."
                )
        domains = {constraint.domain for constraint in constraints}
        if len(domains) > 1:
            raise ValueError("The constraints must all live on one space.")
        if penalty <= 0.0:
            raise ValueError(f"The penalty is positive, got {penalty}.")
        if penalty_growth <= 1.0:
            raise ValueError(f"The penalty growth exceeds one, got {penalty_growth}.")
        if not 0.0 < reduction < 1.0:
            raise ValueError(f"The reduction factor lies in (0, 1), got {reduction}.")
        if not 0.0 < gtol_reduction < 1.0:
            raise ValueError(
                f"The gradient tolerance reduction lies in (0, 1), got "
                f"{gtol_reduction}."
            )
        self._optimizer = optimizer
        self._equalities = tuple(equalities)
        self._inequalities = tuple(inequalities)
        self._penalty = float(penalty)
        self._penalty_growth = float(penalty_growth)
        self._reduction = float(reduction)
        self._tolerance = float(tolerance)
        self._initial_gtol = float(initial_gtol)
        self._gtol_reduction = float(gtol_reduction)

    @property
    def optimizer(self) -> Optimizer:
        """The unconstrained method used for the inner solves."""
        return self._optimizer

    @property
    def equalities(self) -> tuple[Functional, ...]:
        """The equality constraints, each constrained to vanish."""
        return self._equalities

    @property
    def inequalities(self) -> tuple[Functional, ...]:
        """The inequality constraints, each constrained to be at most zero."""
        return self._inequalities

    def augmented(
        self,
        functional: Functional,
        /,
        *,
        multipliers: Sequence[float],
        inequality_multipliers: Sequence[float],
        penalty: float,
    ) -> Functional:
        """The augmented Lagrangian at given multipliers and penalty.

        What each inner solve minimizes. It carries a gradient, and a Hessian
        when the objective and every constraint do, the latter being the
        constraints' Hessians weighted by their multipliers plus the rank-one
        terms ``rho grad c (x) grad c`` the penalty adds; an inactive
        inequality contributes nothing to either.

        Args:
            functional: the objective ``f``.
            multipliers: ``lambda``, one per equality constraint.
            inequality_multipliers: ``mu``, one per inequality constraint.
            penalty: ``rho``.

        Returns:
            The augmented Lagrangian, a functional on the objective's domain.
        """
        space = functional.domain
        equalities = self._equalities
        inequalities = self._inequalities
        lam = np.asarray(multipliers, dtype=float)
        mu = np.asarray(inequality_multipliers, dtype=float)
        rho = float(penalty)
        with_hessian = functional.has_hessian and all(
            constraint.has_hessian for constraint in equalities + inequalities
        )

        def value(x: Any) -> float:
            total = functional(x)
            for weight, constraint in zip(lam, equalities):
                c = constraint(x)
                total += weight * c + 0.5 * rho * c * c
            for weight, constraint in zip(mu, inequalities):
                shifted = max(0.0, weight + rho * constraint(x))
                total += (shifted * shifted - weight * weight) / (2.0 * rho)
            return float(total)

        def linearize(x: Any) -> QuadraticModel:
            base = functional.at(x)
            total = base.value
            gradient = space.copy(base.gradient)
            # (weight, model, constraint) for every term that is live at x.
            active: list[tuple[float, QuadraticModel, Functional]] = []
            for weight, constraint in zip(lam, equalities):
                model = constraint.at(x)
                c = model.value
                total += weight * c + 0.5 * rho * c * c
                active.append((weight + rho * c, model, constraint))
            for weight, constraint in zip(mu, inequalities):
                model = constraint.at(x)
                shifted = weight + rho * model.value
                if shifted > 0.0:
                    total += (shifted * shifted - weight * weight) / (2.0 * rho)
                    active.append((shifted, model, constraint))
                else:
                    total -= weight * weight / (2.0 * rho)
            for weight, model, _ in active:
                gradient = space.axpy(weight, model.gradient, gradient)

            hessian = None
            if with_hessian:
                base_hessian = (
                    base.hessian if base.has_hessian else functional.hessian(x)
                )
                terms = [
                    (
                        weight,
                        model.gradient,
                        model.hessian if model.has_hessian else constraint.hessian(x),
                    )
                    for weight, model, constraint in active
                ]

                def apply(v: Any) -> Any:
                    out = space.copy(base_hessian(v))
                    for weight, grad, curvature in terms:
                        out = space.axpy(weight, curvature(v), out)
                        out = space.axpy(rho * space.inner_product(grad, v), grad, out)
                    return out

                hessian = LinearOperator.self_adjoint(space, apply)

            return QuadraticModel(
                x,
                float(total),
                LinearFunctional.from_representer(space, gradient),
                hessian,
            )

        def hessian(x: Any) -> LinearOperator:
            return linearize(x).hessian

        return Functional.from_callables(
            space,
            value,
            linearize=linearize,
            hessian=hessian if with_hessian else None,
        )

    def minimize(self, functional: Functional, x0: Any, /) -> AugmentedLagrangianResult:
        """Minimize ``functional`` subject to the constraints, from ``x0``.

        Args:
            functional: the objective, which must carry a derivative and
                live on the constraints' space.
            x0: where to start; it need not be feasible.

        Returns:
            The result, with the multipliers and the final violation. It is
            converged when the violation is within tolerance and the last
            inner solve met the final gradient tolerance.

        Raises:
            ValueError: if the objective carries no derivative, or lives on a
                different space from the constraints.
        """
        for constraint in self._equalities + self._inequalities:
            if constraint.domain != functional.domain:
                raise ValueError(
                    f"The objective lives on {functional.domain!r} but a "
                    f"constraint on {constraint.domain!r}."
                )
        return super().minimize(functional, x0)

    def _minimize(self, functional: Functional, x0: Any) -> AugmentedLagrangianResult:
        space = functional.domain
        x = space.copy(x0)
        lam = np.zeros(len(self._equalities))
        mu = np.zeros(len(self._inequalities))
        rho = self._penalty
        omega = max(self._initial_gtol, self._gtol)
        previous_violation = np.inf
        iterations = 0
        evaluations = 0
        history: list[float] = []
        inner = None
        violation = np.inf

        def finish(
            converged: bool, message: str, outer: int
        ) -> AugmentedLagrangianResult:
            return AugmentedLagrangianResult(
                x,
                float(functional(x)),
                inner.gradient_norm if inner is not None else np.nan,
                iterations,
                evaluations + 1,
                converged,
                message,
                history,
                failure=inner.failure if inner is not None else None,
                multipliers=lam.copy(),
                inequality_multipliers=mu.copy(),
                violation=float(violation),
                penalty=rho,
                outer_iterations=outer,
            )

        for outer in range(1, self._max_iterations + 1):
            augmented = self.augmented(
                functional, multipliers=lam, inequality_multipliers=mu, penalty=rho
            )
            inner = self._optimizer.with_tolerance(gtol=omega, rtol=0.0).minimize(
                augmented, x
            )
            x = inner.minimizer
            iterations += inner.iterations
            evaluations += inner.evaluations
            history.extend(inner.history)
            if inner.failure is not None:
                violation = self._violation(x)
                return finish(
                    False, "an inner solve could not evaluate the functional", outer
                )

            equality_values = np.array([c(x) for c in self._equalities])
            inequality_values = np.array([g(x) for g in self._inequalities])
            evaluations += len(equality_values) + len(inequality_values)
            violation = max(
                [0.0]
                + [abs(v) for v in equality_values]
                + [max(v, 0.0) for v in inequality_values]
            )
            lam = lam + rho * equality_values
            mu = np.maximum(0.0, mu + rho * inequality_values)

            if violation <= self._tolerance and omega <= self._gtol and inner.converged:
                return finish(
                    True, "constraints satisfied and gradient tolerance reached", outer
                )

            if violation <= max(self._tolerance, self._reduction * previous_violation):
                omega = max(self._gtol, omega * self._gtol_reduction)
            else:
                rho *= self._penalty_growth
            previous_violation = violation

        return finish(False, "iteration limit reached", self._max_iterations)

    def _violation(self, x: Any) -> float:
        """The largest constraint violation at a point."""
        return max(
            [0.0]
            + [abs(c(x)) for c in self._equalities]
            + [max(g(x), 0.0) for g in self._inequalities]
        )
