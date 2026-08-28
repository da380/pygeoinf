"""
A damped solve inside a monotone scalar root find.

DESIGN.md section 18.6 names this as one numerical kernel with four users: the
feasible-set ellipsoid, the primal nested bisection, the set-inclusion test, and
the discrepancy principle. Each looks for the value of a multiplier at which
some quantity — a misfit, a norm — reaches a target, and each quantity is
monotone in its multiplier, which is what makes bisection a proof rather than a
heuristic.

Two things belong here and nowhere else.

**Bracketing at both ends.** Widening only upwards leaves the search converging
to whatever the lower end happened to be, which is a wrong answer that looks
like a converged one. And *failing* to bracket is not an error: the
non-existence of a root is itself the answer to a feasibility question, so it is
reported rather than raised, together with the endpoint the search ran out at.
Getting that endpoint wrong is how a discrepancy search returns the most
structured model for data that support no structure.

**Warm starting.** Consecutive multipliers in a bisection are close, so the
previous solution is an excellent starting point for the next solve. This is
the difference between a search that costs sixty full solves and one that costs
one solve and fifty-nine corrections. The solvers have taken an ``x0`` all
along; this is what passes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

from ..algebra.operators import LinearOperator
from ..traits import Traits
from .solvers import ConvergenceError, IterativeSolver, LinearSolver

# A probe that fails this way has reached the end of the usable range rather
# than encountered a bug: a damping small enough leaves the normal operator
# numerically singular, and no solver can be asked to go further.
_BREAKDOWN = (ConvergenceError, np.linalg.LinAlgError)

__all__ = [
    "Evaluation",
    "RootResult",
    "monotone_root",
    "DampedSolves",
]


@dataclass(frozen=True)
class Evaluation:
    """What one probe of the monotone quantity returned."""

    value: float
    """The quantity being driven to a target."""

    solution: Any = None
    """Whatever vector produced it, carried so the caller need not redo the
    solve at the answer, and so the next probe can warm-start from it."""

    iterations: int = 0
    """Inner iterations this probe cost, for the diagnostics."""


@dataclass(frozen=True)
class RootResult:
    """The multiplier, the solution there, and what it cost."""

    argument: float
    """The multiplier. When the root could not be bracketed this is the
    endpoint the search ran out at, which is the answer the caller wants in
    both saturated cases — see :attr:`exhausted`."""

    value: float
    """The quantity at :attr:`argument`."""

    solution: Any
    """The vector at :attr:`argument`."""

    evaluations: int
    """How many probes were taken."""

    inner_iterations: int
    """Total iterations across every inner solve. Zero for a direct solver or
    a closed-form family. This is what says whether warm starting worked."""

    bracket: tuple[float, float]
    """The final bracket."""

    converged: bool
    """Whether a root was bracketed and the bracket closed to tolerance."""

    exhausted: Literal["low", "high"] | None = None
    """Set when no root exists in the searched range. ``"high"`` means the
    quantity never reached the target however large the multiplier grew;
    ``"low"``, however small. In both cases :attr:`argument` is the endpoint,
    which is the extreme the caller was asking for."""

    warm_started: bool = False
    """Whether the probes were given the previous solution to start from."""


def monotone_root(
    evaluate: Callable[[float, Any], Evaluation],
    target: float,
    /,
    *,
    decreasing: bool = True,
    initial: float = 1.0,
    iterations: int = 60,
    rtol: float = 1e-6,
    atol: float = 0.0,
    expansions: int = 200,
    warm_start: bool = True,
) -> RootResult:
    """Find the positive multiplier at which a monotone quantity hits *target*.

    Args:
        evaluate: ``(multiplier, previous_solution) -> Evaluation``. The second
            argument is the solution from the last probe, or None on the first,
            and is there to be used as a starting guess.
        target: the value to reach.
        decreasing: whether the quantity falls as the multiplier rises. The
            search is written for a decreasing quantity and flips the sign of
            an increasing one, so both are the same code and neither is a
            special case.
        initial: where to start bracketing.
        iterations: bisection steps once bracketed.
        rtol, atol: the bracket is closed when its width falls below
            ``atol + rtol * (low + high)``.
        expansions: how far to widen, in factors of ten, before concluding
            that no root exists in that direction.
        warm_start: pass each probe the previous solution.

    Returns:
        A :class:`RootResult`. Check ``converged`` and ``exhausted`` — a
        search that could not bracket has still returned the useful endpoint.

    The bisection is *geometric*, taking the square root of the endpoints
    rather than their mean, because a multiplier of this kind ranges over
    orders of magnitude rather than over an interval.
    """
    if initial <= 0.0:
        raise ValueError(f"The starting multiplier must be positive, got {initial}.")
    if not 0 < iterations:
        raise ValueError(f"At least one bisection step is needed, got {iterations}.")

    sign = 1.0 if decreasing else -1.0
    goal = sign * target
    tally = {"evaluations": 0, "iterations": 0}
    previous: Any = None

    def probe(multiplier: float) -> tuple[float, Any]:
        nonlocal previous
        result = evaluate(multiplier, previous if warm_start else None)
        tally["evaluations"] += 1
        tally["iterations"] += result.iterations
        if result.solution is not None:
            previous = result.solution
        return sign * result.value, result.solution

    def finish(
        multiplier: float,
        scaled: float,
        solution: Any,
        bracket: tuple[float, float],
        converged: bool,
        exhausted: Literal["low", "high"] | None = None,
    ) -> RootResult:
        return RootResult(
            argument=float(multiplier),
            value=float(sign * scaled),
            solution=solution,
            evaluations=tally["evaluations"],
            inner_iterations=tally["iterations"],
            bracket=bracket,
            converged=converged,
            exhausted=exhausted,
            warm_started=warm_start,
        )

    def widen(
        direction: float,
        satisfied: Callable[[float], bool],
        end: str,
        start: tuple[float, float, Any],
    ) -> tuple[float, float, Any, RootResult | None]:
        """Walk one end of the bracket until the goal is straddled.

        Stops on three things: the goal being reached, the expansions running
        out, and the *solve breaking down* — which is not a failure but the
        edge of the usable range, since a small enough multiplier leaves the
        operator numerically singular. In every stopping case the last probe
        that worked is what is reported, because that is the extreme the
        caller was asking for.
        """
        multiplier, scaled, solution = start
        if satisfied(scaled):
            return multiplier, scaled, solution, None
        for _ in range(expansions):
            candidate = multiplier * direction
            try:
                scaled_candidate, solution_candidate = probe(candidate)
            except _BREAKDOWN:
                break
            multiplier, scaled, solution = (
                candidate,
                scaled_candidate,
                solution_candidate,
            )
            if satisfied(scaled):
                return multiplier, scaled, solution, None
        return (
            multiplier,
            scaled,
            solution,
            finish(multiplier, scaled, solution, (multiplier, multiplier), False, end),
        )

    # Probed once and handed to both walks, rather than once per direction:
    # the starting multiplier is common to them, and a probe is a solve.
    begin = (initial, *probe(initial))

    # Upward: the quantity must fall to or below the goal.
    high, scaled, solution, exhausted = widen(
        10.0, lambda value: value <= goal, "high", begin
    )
    if exhausted is not None:
        return exhausted

    # Downward: the quantity must rise to or above the goal.
    low, scaled_low, solution_low, exhausted = widen(
        0.1, lambda value: value >= goal, "low", begin
    )
    if exhausted is not None:
        return exhausted

    for _ in range(iterations):
        middle = float(np.sqrt(low * high))
        scaled, solution = probe(middle)
        if scaled > goal:
            low = middle
        else:
            high = middle
        if high - low <= atol + rtol * (low + high):
            return finish(middle, scaled, solution, (low, high), True)

    middle = float(np.sqrt(low * high))
    scaled, solution = probe(middle)
    # The loop returns the moment the bracket is tight enough, so arriving here
    # means it never was: the iteration cap was reached first. Reporting that
    # as convergence claimed a root to a tolerance never met — with one
    # iteration and zero tolerances it said converged with the bracket still
    # 6.8 wide. The criterion is re-evaluated rather than hard-coded False so
    # the answer stays tied to the bracket rather than to the control flow.
    converged = high - low <= atol + rtol * (low + high)
    return finish(middle, scaled, solution, (low, high), converged)


@dataclass
class DampedSolves:
    """Solve ``(base + multiplier * shift) x == b`` along a sweep of multipliers.

    The family a damped least-squares root find walks along: ``A* R^-1 A + t I``
    in the model space, ``A A* + t R`` in the data space, and the two Backus
    multipliers in their turn.

    Two costs are saved along the sweep, and they are different costs.

    **The solution is carried forward.** Every solve after the first starts from
    the last one's answer. In the bisection phase consecutive multipliers
    converge on each other, so each solve becomes a correction rather than a
    fresh problem. A direct solver cannot be warm started — it does not iterate
    — so the guess is ignored there and ``iterations`` comes back zero, which is
    the honest report that the family is being factorised afresh each step.

    **The preconditioner is kept.** A preconditioner supplied as a
    :class:`LinearSolver` is otherwise rebuilt against every member of the
    family, which for an expensive one — a Woodbury surrogate with its own
    inner factorisation — costs more than the solves it is accelerating. It is
    built once and reused while the multiplier stays within :attr:`refresh` of
    where it was built, and rebuilt when it wanders further. A preconditioner
    is an approximation, so reuse costs accuracy rather than correctness; the
    threshold is where that stops being a good trade.
    """

    base: LinearOperator
    shift: LinearOperator
    solver: LinearSolver
    traits: Traits | None = None
    assemble: Callable[[float], LinearOperator] | None = None
    """How to build the member at one multiplier, when ``base + t * shift``
    would lose something the solver needs.

    A plain sum is an anonymous operator: it knows its value and its adjoint
    and nothing else. That is enough to solve with, but not enough to
    *precondition* with, because every structure-aware preconditioner works by
    reading the factors ``A``, ``Q`` and ``R`` off the operator it is given.
    Assembling the sum therefore refused those preconditioners inside every
    discrepancy sweep, while the same solver worked on a fixed damping — so the
    library's own claim that they apply to all the point estimators held
    everywhere except where the sweep was the point. Pass a family's ``at`` and
    the member arrives with its factors intact."""

    refresh: float = 10.0
    """Rebuild the preconditioner once the multiplier has moved by more than
    this factor from where it was built. Infinity never rebuilds; one always
    does."""

    _cache: dict = field(default_factory=dict, repr=False)
    _prepared: LinearSolver | None = field(default=None, repr=False)
    _prepared_at: float | None = field(default=None, repr=False)

    def operator(self, multiplier: float) -> LinearOperator:
        """The member at one multiplier, assembled once and kept.

        :attr:`assemble` when one was given, and ``base + multiplier * shift``
        otherwise.

        Args:
            multiplier: which member of the family.

        Returns:
            The operator to solve with, carrying :attr:`traits`.
        """
        if multiplier not in self._cache:
            if self.assemble is not None:
                assembled = self.assemble(multiplier)
            else:
                assembled = self.base + multiplier * self.shift
            if self.traits is not None:
                assembled = assembled.with_traits(self.traits)
            self._cache[multiplier] = assembled
        return self._cache[multiplier]

    def _solver_for(self, multiplier: float, operator: LinearOperator) -> LinearSolver:
        """The solver to use, with any deferred preconditioner already built."""
        solver = self.solver
        if not isinstance(solver, IterativeSolver) or not isinstance(
            solver.preconditioner, LinearSolver
        ):
            return solver

        stale = self._prepared_at is None or not (
            1.0 / self.refresh <= multiplier / self._prepared_at <= self.refresh
        )
        if stale:
            self._prepared = solver.resolved_for(operator)
            self._prepared_at = multiplier
        return self._prepared

    def solve(
        self, multiplier: float, right_hand_side: Any, /, *, x0: Any = None
    ) -> Any:
        """One solve, warm-started from *x0*, returning the full result."""
        operator = self.operator(multiplier)
        solver = self._solver_for(multiplier, operator)
        return solver(operator).solve(right_hand_side, x0=x0)
