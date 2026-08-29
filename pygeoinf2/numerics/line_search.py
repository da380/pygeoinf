"""
Line searches, coordinate-free.

A line search needs two things along a direction: the value ``phi(a) ==
f(x + a p)`` and the slope ``phi'(a) == (grad f(x + a p), p)``. Both are inner
products in the space's own metric, so neither needs a component map — and the
slope being metric-correct is precisely what v1's SciPy bridge has to arrange
by hand, passing derivative components so that SciPy's Euclidean dot product
happens to give the right number.

Here the correct pairing is the only one available, because the gradient and
the direction are both vectors and the slope is their inner product.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..algebra.operators import Functional

__all__ = [
    "LineSearchResult",
    "LineSearch",
    "ArmijoLineSearch",
    "StrongWolfeLineSearch",
]


@dataclass(frozen=True)
class LineSearchResult:
    """The outcome of a line search."""

    step: float
    point: Any
    value: float
    evaluations: int
    converged: bool
    model: Any = None
    """The functional's model at :attr:`point`, when the search already had it.

    A Wolfe search evaluates the *gradient* at each trial step to test the
    curvature condition, so on success it is holding the very model the caller
    would otherwise go and recompute -- one full evaluation per outer
    iteration, spent to learn something already known. ``None`` from a
    backtracking search, which only ever needs values, and the caller then
    evaluates as before.
    """

    def __repr__(self) -> str:
        return (
            f"LineSearchResult(step={self.step:.4g}, value={self.value:.6g}, "
            f"evaluations={self.evaluations}, converged={self.converged})"
        )


class LineSearch(ABC):
    """Chooses a step length along a descent direction."""

    @abstractmethod
    def __call__(
        self,
        functional: Functional,
        point: Any,
        direction: Any,
        /,
        *,
        value: float,
        slope: float,
        initial_step: float = 1.0,
    ) -> LineSearchResult:
        """Find a step along ``direction`` from ``point``.

        Args:
            functional: the objective.
            point: where the search starts.
            direction: the search direction, which must be a descent direction.
            value: ``f(point)``, already known to the caller.
            slope: ``(grad f(point), direction)``, likewise. Negative for a
                descent direction.
            initial_step: the first trial step. Newton-type methods should pass
                one, so that the natural step is tried first.
        """


class ArmijoLineSearch(LineSearch):
    """Backtracking until sufficient decrease holds.

    Cheap and robust: it needs values only, never gradients, so it costs one
    evaluation per backtrack. Adequate for steepest descent and Newton, but not
    for quasi-Newton methods, whose curvature updates need the Wolfe curvature
    condition to keep the approximate Hessian positive definite.
    """

    def __init__(
        self,
        /,
        *,
        decrease: float = 1e-4,
        contraction: float = 0.5,
        max_backtracks: int = 50,
        min_step: float = 1e-16,
    ) -> None:
        """
        Args:
            decrease: the Armijo constant ``c1``.
            contraction: the factor each failed step is multiplied by.
            max_backtracks: give up after this many contractions.
            min_step: give up once the step falls below this.
        """
        self._decrease = decrease
        self._contraction = contraction
        self._max_backtracks = max_backtracks
        self._min_step = min_step

    def __call__(
        self,
        functional: Functional,
        point: Any,
        direction: Any,
        /,
        *,
        value: float,
        slope: float,
        initial_step: float = 1.0,
    ) -> LineSearchResult:
        space = functional.domain
        step = initial_step
        evaluation = 0
        for evaluation in range(1, self._max_backtracks + 1):
            trial = space.axpy(step, direction, space.copy(point))
            trial_value = functional(trial)
            if trial_value <= value + self._decrease * step * slope:
                return LineSearchResult(step, trial, trial_value, evaluation, True)
            step *= self._contraction
            if step < self._min_step:
                break
        # The evaluations actually spent: a stop on ``min_step`` used to be
        # reported as the full backtrack count.
        return LineSearchResult(step, point, value, evaluation, False)


class StrongWolfeLineSearch(LineSearch):
    """Bracketing and sectioning for the strong Wolfe conditions.

    The standard bracket-then-zoom scheme. The curvature condition is what
    quasi-Newton methods need: it guarantees ``(y, s) > 0``, so an L-BFGS
    update keeps its implicit Hessian positive definite and the direction it
    produces stays a descent direction.
    """

    def __init__(
        self,
        /,
        *,
        decrease: float = 1e-4,
        curvature: float = 0.9,
        max_iterations: int = 30,
        max_step: float = 1e8,
    ) -> None:
        """
        Args:
            decrease: the Armijo constant ``c1``.
            curvature: the Wolfe constant ``c2``. Must exceed ``decrease``;
                0.9 suits quasi-Newton methods and 0.1 suits nonlinear CG.
            max_iterations: cap on both the bracketing and sectioning phases.
            max_step: cap on the step during bracketing.
        """
        if not 0.0 < decrease < curvature < 1.0:
            raise ValueError("Require 0 < decrease < curvature < 1.")
        self._decrease = decrease
        self._curvature = curvature
        self._max_iterations = max_iterations
        self._max_step = max_step

    def __call__(
        self,
        functional: Functional,
        point: Any,
        direction: Any,
        /,
        *,
        value: float,
        slope: float,
        initial_step: float = 1.0,
    ) -> LineSearchResult:
        space = functional.domain
        evaluations = 0

        def evaluate(step: float) -> tuple[Any, Any, float, float]:
            """Model, value and slope at a trial step."""
            nonlocal evaluations
            evaluations += 1
            trial = space.axpy(step, direction, space.copy(point))
            model = functional.at(trial)
            return (
                trial,
                model,
                model.value,
                space.inner_product(model.gradient, direction),
            )

        previous_step, previous_value = 0.0, value
        step = initial_step

        for iteration in range(1, self._max_iterations + 1):
            trial, trial_model, trial_value, trial_slope = evaluate(step)

            if trial_value > value + self._decrease * step * slope or (
                iteration > 1 and trial_value >= previous_value
            ):
                return self._zoom(
                    functional,
                    point,
                    direction,
                    value,
                    slope,
                    previous_step,
                    step,
                    evaluations,
                )
            if abs(trial_slope) <= -self._curvature * slope:
                return LineSearchResult(
                    step, trial, trial_value, evaluations, True, trial_model
                )
            if trial_slope >= 0.0:
                return self._zoom(
                    functional,
                    point,
                    direction,
                    value,
                    slope,
                    step,
                    previous_step,
                    evaluations,
                )
            previous_step, previous_value = step, trial_value
            step = min(2.0 * step, self._max_step)
            if step >= self._max_step:
                break

        trial = space.axpy(step, direction, space.copy(point))
        return LineSearchResult(step, trial, functional(trial), evaluations, False)

    def _zoom(
        self,
        functional: Functional,
        point: Any,
        direction: Any,
        value: float,
        slope: float,
        low: float,
        high: float,
        evaluations: int,
    ) -> LineSearchResult:
        """Section the bracket until a strong Wolfe point is found.

        ``low`` is the end with the lower value; it is not necessarily the
        smaller step, which is why the interval is not kept ordered.
        """
        space = functional.domain
        # Counted: it is an evaluation of the functional like any other, and
        # leaving it out understated every zooming search by one.
        low_value = functional(space.axpy(low, direction, space.copy(point)))
        evaluations += 1

        for _ in range(self._max_iterations):
            step = 0.5 * (low + high)
            trial = space.axpy(step, direction, space.copy(point))
            model = functional.at(trial)
            evaluations += 1

            if model.value > value + self._decrease * step * slope or (
                model.value >= low_value
            ):
                high = step
                continue

            trial_slope = space.inner_product(model.gradient, direction)
            if abs(trial_slope) <= -self._curvature * slope:
                return LineSearchResult(
                    step, trial, model.value, evaluations, True, model
                )
            if trial_slope * (high - low) >= 0.0:
                high = low
            low, low_value = step, model.value

        step = 0.5 * (low + high)
        trial = space.axpy(step, direction, space.copy(point))
        return LineSearchResult(step, trial, functional(trial), evaluations, False)
