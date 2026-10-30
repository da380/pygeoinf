"""
Line searches, coordinate-free.

A line search needs two things along a direction: the value ``phi(a) ==
f(x + a p)`` and the slope ``phi'(a) == (grad f(x + a p), p)``. The slope is
the inner product of two vectors, the gradient and the direction, in the
space's own metric, so it needs no component map and is correct on any space.

An :class:`~pygeoinf.algebra.operators.EvaluationError` raised by the
functional at a trial point, or a non-finite value there, is treated as a step
that went too far, never as a failure of the search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from dataclasses import dataclass
from typing import Any, Callable

from ..algebra.operators import EvaluationError, Functional

__all__ = [
    "LineSearchResult",
    "LineSearch",
    "ArmijoLineSearch",
    "StrongWolfeLineSearch",
]

_MAX_GROWTH = 10.0
"""The most a bracketing step may grow by in one move of the Wolfe search.

The secant is exact on a quadratic and a guess elsewhere; a step that lands
a decade beyond the last one is still read against slopes measured near
it, where a thousandfold jump along an exponential forward is an overflow.
Twelve such moves span the twelve decades a first trial can be short by."""


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
    iteration. ``None`` from a backtracking search, which only ever needs
    values; the caller then evaluates the model itself.
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
            min_step: give up once the step falls below this fraction of the
                first trial step. Relative, because the step's units are the
                direction's, which the caller chose.
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
        floor = self._min_step * initial_step
        evaluation = 0
        for evaluation in range(1, self._max_backtracks + 1):
            trial = space.axpy(step, direction, space.copy(point))
            # A step into a region the functional cannot evaluate -- an
            # EvaluationError from a forward that did not converge, or a
            # non-finite value -- is a step that went too far, and the
            # response is the one already in hand: backtrack.
            try:
                trial_value = functional(trial)
            except EvaluationError:
                trial_value = np.inf
            if np.isfinite(trial_value) and (
                trial_value <= value + self._decrease * step * slope
            ):
                return LineSearchResult(step, trial, trial_value, evaluation, True)
            step *= self._contraction
            if step < floor:
                break
        # The evaluations actually spent, which a stop on ``min_step`` leaves
        # below ``max_backtracks``.
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
        max_step: float = 1e16,
    ) -> None:
        """
        Args:
            decrease: the Armijo constant ``c1``.
            curvature: the Wolfe constant ``c2``. Must exceed ``decrease``;
                0.9 suits quasi-Newton methods and 0.1 suits nonlinear CG.
            max_iterations: cap on both the bracketing and sectioning phases.
            max_step: cap on the step during bracketing, as a multiple of the
                first trial step. Relative, because the step's units are the
                direction's, which the caller chose; and large, because the
                first trial along a gradient direction is a guess that the
                bracketing phase is there to correct, by as many decades as
                it takes (see :meth:`_extrapolate`).
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
        counter = [0]

        def evaluate(step: float) -> tuple[Any, Any, float, float]:
            """Model, value and slope at a trial step."""
            counter[0] += 1
            trial = space.axpy(step, direction, space.copy(point))
            model = functional.at(trial)
            return (
                trial,
                model,
                model.value,
                space.inner_product(model.gradient, direction),
            )

        previous_step, previous_value, previous_slope = 0.0, value, slope
        step = initial_step
        cap = self._max_step * initial_step

        for iteration in range(1, self._max_iterations + 1):
            # A trial the functional cannot evaluate, or evaluates to a
            # non-finite value or slope, is treated as infinitely bad: the
            # step went too far, and the bracket it closes is sectioned like
            # any other.
            try:
                trial, trial_model, trial_value, trial_slope = evaluate(step)
            except EvaluationError:
                trial_value, trial_slope = np.inf, np.nan
            if not (np.isfinite(trial_value) and np.isfinite(trial_slope)):
                return self._zoom(
                    evaluate,
                    counter,
                    value,
                    slope,
                    previous_step,
                    previous_value,
                    previous_slope,
                    step,
                    np.inf,
                )

            if trial_value > value + self._decrease * step * slope or (
                iteration > 1 and trial_value >= previous_value
            ):
                return self._zoom(
                    evaluate,
                    counter,
                    value,
                    slope,
                    previous_step,
                    previous_value,
                    previous_slope,
                    step,
                    trial_value,
                )
            if abs(trial_slope) <= -self._curvature * slope:
                return LineSearchResult(
                    step, trial, trial_value, counter[0], True, trial_model
                )
            if trial_slope >= 0.0:
                return self._zoom(
                    evaluate,
                    counter,
                    value,
                    slope,
                    step,
                    trial_value,
                    trial_slope,
                    previous_step,
                    previous_value,
                )
            previous_step, previous_value, previous_slope = (
                step,
                trial_value,
                trial_slope,
            )
            step = min(self._extrapolate(step, trial_slope, slope), cap)
            if step >= cap:
                break

        trial = space.axpy(step, direction, space.copy(point))
        try:
            final_value = functional(trial)
        except EvaluationError:
            return LineSearchResult(0.0, point, value, counter[0], False)
        return LineSearchResult(step, trial, final_value, counter[0], False)

    @staticmethod
    def _extrapolate(step: float, step_slope: float, slope: float) -> float:
        """The next bracketing trial, beyond a step that was not far enough.

        The secant through the slopes at zero and at ``step`` puts the
        minimizer at ``-slope / (step_slope - slope)`` times the step, which
        is exact when the function along the line is quadratic; so a first
        trial that was too short by many decades is corrected in one move,
        where doubling would need a step per factor of two. The growth is
        clipped to between two and ``_MAX_GROWTH``: at least two so the
        bracket widens, at most tenfold a step so the trial cannot leave
        the region the slopes were read in by more than a decade. A slope
        that has not grown gives no estimate, and the step doubles.
        """
        change = step_slope - slope
        growth = 2.0
        if change > 0.0 and slope < 0.0:
            growth = min(max(-slope / change, 2.0), _MAX_GROWTH)
        return growth * step

    def _zoom(
        self,
        evaluate: Callable[[float], tuple[Any, Any, float, float]],
        counter: list[int],
        value: float,
        slope: float,
        low: float,
        low_value: float,
        low_slope: float,
        high: float,
        high_value: float,
    ) -> LineSearchResult:
        """Section the bracket until a strong Wolfe point is found.

        ``low`` is the end with the lower value; it is not necessarily the
        smaller step, which is why the interval is not kept ordered. The
        values at both ends and the slope at ``low`` arrive from the
        bracketing phase, so nothing is re-evaluated on entry.

        The trial step is the minimizer of the cubic through the two ends and
        the most recently discarded point, accepted when it lies at least a
        fifth of the bracket from either end; failing that, the minimizer of
        the quadratic through the two ends and the slope at ``low``, accepted
        at least a tenth of the bracket from either end. An interpolant's
        minimizer closer to an end than that is not to be trusted (Nocedal
        and Wright §3.5), and the trial is moved to the margin on that side:
        the bracket then shrinks by a factor of five or ten a step towards a
        minimizer that lies many decades inside it, where bisection would
        gain one factor of two a step. With no interpolant at all, bisection.
        On a quadratic model the interpolant is exact, so interpolation finds
        in one or two steps what bisection approaches only linearly.
        """
        rec_step: float | None = None
        rec_value = 0.0
        for iteration in range(self._max_iterations):
            step = self._section(
                low, low_value, low_slope, high, high_value, rec_step, rec_value
            )
            try:
                trial, model, trial_value, trial_slope = evaluate(step)
            except EvaluationError:
                trial_value, trial_slope = np.inf, np.nan
            if not (np.isfinite(trial_value) and np.isfinite(trial_slope)):
                # Unevaluable inside the bracket: infinitely bad, so the
                # bracket closes on it from the ``high`` side.
                rec_step, rec_value = high, high_value
                high, high_value = step, np.inf
                continue

            if trial_value > value + self._decrease * step * slope or (
                trial_value >= low_value
            ):
                rec_step, rec_value = high, high_value
                high, high_value = step, trial_value
                continue

            if abs(trial_slope) <= -self._curvature * slope:
                return LineSearchResult(
                    step, trial, trial_value, counter[0], True, model
                )
            if trial_slope * (high - low) >= 0.0:
                rec_step, rec_value = high, high_value
                high, high_value = low, low_value
            else:
                rec_step, rec_value = low, low_value
            low, low_value, low_slope = step, trial_value, trial_slope

        step = 0.5 * (low + high)
        try:
            trial, model, trial_value, _ = evaluate(step)
        except EvaluationError:
            # No point to return: a failed search's point is never consumed,
            # the caller keeps its own.
            return LineSearchResult(0.0, None, value, counter[0], False)
        return LineSearchResult(step, trial, trial_value, counter[0], False, model)

    @staticmethod
    def _section(
        low: float,
        low_value: float,
        low_slope: float,
        high: float,
        high_value: float,
        rec_step: float | None,
        rec_value: float,
    ) -> float:
        """The next trial step inside the bracket."""
        width = abs(high - low)
        lower, upper = min(low, high), max(low, high)
        if rec_step is not None:
            step = _cubic_minimizer(
                low, low_value, low_slope, high, high_value, rec_step, rec_value
            )
            margin = 0.2 * width
            if step is not None and lower + margin <= step <= upper - margin:
                return step
        step = _quadratic_minimizer(low, low_value, low_slope, high, high_value)
        margin = 0.1 * width
        if step is not None:
            return float(min(max(step, lower + margin), upper - margin))
        return 0.5 * (low + high)


def _quadratic_minimizer(
    a: float, fa: float, fpa: float, b: float, fb: float
) -> float | None:
    """Minimizer of the quadratic through ``(a, fa)`` with slope ``fpa`` and
    ``(b, fb)``, or None where it has none."""
    db = b - a
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        curvature = (fb - fa - fpa * db) / (db * db)
        step = a - fpa / (2.0 * curvature)
    return float(step) if np.isfinite(step) and curvature > 0.0 else None


def _cubic_minimizer(
    a: float, fa: float, fpa: float, b: float, fb: float, c: float, fc: float
) -> float | None:
    """Minimizer of the cubic through ``(a, fa)`` with slope ``fpa``,
    ``(b, fb)`` and ``(c, fc)``, or None where it has none."""
    db, dc = b - a, c - a
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denominator = (db * dc) ** 2 * (db - dc)
        d1 = np.array([[dc**2, -(db**2)], [-(dc**3), db**3]])
        A, B = d1 @ np.array([fb - fa - fpa * db, fc - fa - fpa * dc]) / denominator
        radical = B * B - 3.0 * A * fpa
        step = a + (-B + np.sqrt(radical)) / (3.0 * A)
    return float(step) if np.isfinite(step) else None
