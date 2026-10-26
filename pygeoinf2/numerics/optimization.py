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

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..algebra.operators import EvaluationError, Functional, LinearOperator, Operator
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

    def __repr__(self) -> str:
        return (
            f"OptimizationResult(value={self.value:.6g}, "
            f"|grad|={self.gradient_norm:.3g}, iterations={self.iterations}, "
            f"converged={self.converged}: {self.message})"
        )


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
                gradient test alone. The line-search methods apply it;
                :class:`TrustRegionNewton` does not.
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

    @abstractmethod
    def _minimize(self, functional: Functional, x0: Any) -> OptimizationResult:
        """Run the method, having passed validation."""

    # -- shared machinery ----------------------------------------------

    def _tolerance(self, initial_gradient_norm: float) -> float:
        return max(self._gtol, self._rtol * initial_gradient_norm)

    def _precision_limited(self, value: float, gradient_norm: float) -> bool:
        """Whether the gradient is as small as double precision allows here.

        A line search fails when no step along a descent direction improves the
        value. Near a minimizer that is not a defect: the decrease a step is
        predicted to make, ``c1 * alpha * slope``, eventually falls below the
        rounding error in the value itself, which is ``eps * |f|`` rather than
        ``eps``. Past that point the search is comparing noise.

        So a failure with the gradient already at that scale is success at the
        accuracy the arithmetic allows, and saying "line search failed" instead
        reports a wrong answer about a right one. The test is
        ``||g|| <= sqrt(eps) (1 + |f|)``.
        """
        floor = float(np.sqrt(np.finfo(float).eps)) * (1.0 + abs(value))
        return gradient_norm <= floor

    def _finish_search_failure(
        self,
        x: Any,
        value: float,
        gradient_norm: float,
        iteration: int,
        evaluations: int,
        history: list[float],
    ) -> OptimizationResult:
        """The outcome when no step improved the value."""
        limited = self._precision_limited(value, gradient_norm)
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
                    x, model.value, gradient_norm, iteration, evaluations, history
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
            "iteration limit reached",
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
            **kwargs: passed to :class:`Optimizer`.
        """
        self._forcing = forcing
        self._max_cg_iterations = max_cg_iterations
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

            direction, _ = truncated_cg(
                model.hessian,
                space.negative(model.gradient),
                rtol=min(self._forcing, np.sqrt(gradient_norm)),
                max_iterations=self._max_cg_iterations,
            )
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
                    x, model.value, gradient_norm, iteration, evaluations, history
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
            "iteration limit reached",
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
        initial_radius: float = 1.0,
        max_radius: float = 1e6,
        acceptance: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            initial_radius: the starting trust-region radius.
            max_radius: cap on the radius.
            acceptance: reject a step whose actual-to-predicted reduction ratio
                falls below this.
            **kwargs: passed to :class:`Optimizer`.
        """
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

    def _minimize(self, functional: Functional, x0: Any) -> OptimizationResult:
        space = functional.domain
        x = space.copy(x0)
        model = functional.at(x)
        gradient_norm = space.norm(model.gradient)
        tolerance = self._tolerance(gradient_norm)
        radius = self._initial_radius
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

            step, _ = truncated_cg(
                model.hessian,
                space.negative(model.gradient),
                rtol=min(0.1, np.sqrt(gradient_norm)),
                radius=radius,
            )
            predicted = -(
                space.inner_product(model.gradient, step)
                + 0.5 * space.inner_product(model.hessian(step), step)
            )
            if predicted <= 0.0:
                radius *= 0.25
                if radius < 1e-14:
                    break
                continue

            trial = space.axpy(1.0, step, space.copy(x))
            try:
                trial_value = functional(trial)
            except EvaluationError:
                # An infeasible trial is a trust region's ordinary business:
                # the model was wrong there, so shrink and try again.
                evaluations += 1
                radius *= 0.25
                if radius < 1e-14:
                    break
                continue
            evaluations += 1
            if not np.isfinite(trial_value):
                radius *= 0.25
                if radius < 1e-14:
                    break
                continue
            ratio = (model.value - trial_value) / predicted

            if ratio < 0.25:
                radius *= 0.25
            elif ratio > 0.75 and abs(space.norm(step) - radius) < 1e-8 * radius:
                radius = min(2.0 * radius, self._max_radius)

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
                x, model = trial, accepted
                evaluations += 1
                gradient_norm = space.norm(model.gradient)
                history.append(model.value)

            if radius < 1e-14:
                break

        return OptimizationResult(
            x,
            model.value,
            gradient_norm,
            self._max_iterations,
            evaluations,
            gradient_norm <= tolerance,
            "iteration limit reached",
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

    Returns:
        The step and a word saying which condition stopped it.
    """
    space = hessian.domain
    limit = max_iterations if max_iterations is not None else max(space.dim, 10)

    step = space.zero()
    residual = space.copy(rhs)
    direction = space.copy(residual)
    residual_norm = space.norm(residual)
    target = rtol * residual_norm

    if residual_norm == 0.0:
        return step, "converged"

    for _ in range(limit):
        curvature_vector = hessian(direction)
        curvature = space.inner_product(direction, curvature_vector)

        if curvature <= 0.0:
            if radius is None:
                # No region to stop at: fall back on the steepest direction.
                return (rhs if space.norm(step) == 0.0 else step), "negative curvature"
            return _to_boundary(space, step, direction, radius), "negative curvature"

        alpha = space.squared_norm(residual) / curvature
        trial = space.axpy(alpha, direction, space.copy(step))

        if radius is not None and space.norm(trial) >= radius:
            return _to_boundary(space, step, direction, radius), "boundary"

        squared = space.squared_norm(residual)
        step = trial
        residual = space.axpy(-alpha, curvature_vector, residual)
        residual_norm = space.norm(residual)
        if residual_norm <= target:
            return step, "converged"

        beta = space.squared_norm(residual) / squared
        direction = space.axpy(1.0, residual, space.scale_inplace(beta, direction))

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
