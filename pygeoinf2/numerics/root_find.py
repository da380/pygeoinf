"""
A damped solve inside a monotone scalar root find.

One numerical kernel serves the feasible-set ellipsoid, the primal nested
bisection, the set-inclusion test, the KKT solvers and the discrepancy
principle. Each looks for the value of a multiplier at which
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
one solve and fifty-nine corrections, the previous solution being passed to
each solve as its ``x0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

from ..algebra.operators import LinearOperator
from ..traits import Traits
from .solvers import (
    ConvergenceError,
    DirectSolver,
    FamilySolver,
    IterativeSolver,
    LinearSolver,
)

_UNSET: Any = object()
"""Distinguishes "not looked yet" from "looked, and there is nothing to do"."""

# A probe that fails this way has reached the end of the usable range rather
# than encountered a bug: a damping small enough leaves the normal operator
# numerically singular, and no solver can be asked to go further.
#
# ConvergenceError is deliberately *not* in here. It is raised both by a
# solver that met a singular operator and by one that merely ran out of
# iterations, and it carries nothing to tell them apart, so treating it as
# saturation would report "no root in this range" for a sweep whose
# iteration cap was simply too low. It is caught separately below and
# recorded on the result.
_BREAKDOWN = (np.linalg.LinAlgError,)

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

    breakdown: BaseException | None = None
    """The exception that ended the sweep, when one did.

    A saturated sweep and a failed inner solve both stop the widening, and
    they mean different things: the first says no root exists in the range,
    which is an answer, and the second says a probe could not be computed,
    which is not. Only a genuine singularity -- ``LinAlgError`` -- is read as
    saturation. A ``ConvergenceError`` from a strict solver that merely
    ran out of iterations lands here instead, so the caller can see that
    :attr:`exhausted` is standing in for a solve that did not finish rather
    than for an exhausted range."""


def monotone_root(
    evaluate: Callable[[float, Any], Evaluation],
    target: float,
    /,
    *,
    decreasing: bool = True,
    initial: float = 1.0,
    max_iterations: int = 60,
    rtol: float = 1e-6,
    atol: float = 0.0,
    expansions: int = 200,
    minimum: float = 0.0,
    maximum: float = float("inf"),
    warm_start: bool = True,
    limit: float | None = None,
) -> RootResult:
    """Find the positive multiplier at which a monotone quantity hits *target*.

    The search works in the *logarithm* of the multiplier, because a
    multiplier of this kind ranges over orders of magnitude rather than over
    an interval: the bracketing widens by decades, and the root is then found
    by Brent's method in ``log t``.

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
        max_iterations: Brent iterations once bracketed, each one solve.
        rtol: the bracket is closed when its width falls below
            ``atol + rtol * (low + high)``.
        atol: the absolute half of that criterion, which matters when the
            root sits near zero and a relative test never closes.
        expansions: how many times to widen an end of the bracket by a
            factor of ten before concluding that no root exists in that
            direction.
        minimum: a floor on the multiplier. The downward walk stops short of
            it and reports the range exhausted, rather than probing on: a
            damping far below any physical scale solves a numerically
            singular system at each probe and says nothing about the data.
            The discrepancy searches expose it as ``minimum_damping``. Zero,
            the default, is no floor.
        maximum: a ceiling on the multiplier, likewise for the upward walk.
        limit: the quantity's value in the limit of an infinite multiplier,
            when the caller knows it: the misfit of the zero model, for a
            damping search. When that value never reaches the target the
            search is saturated before it starts, at the endpoint the
            upward walk would have reached, with no probe taken: the
            result's value is the limit and its solution is ``None``, for
            the caller to replace by the vector the limit belongs to.
        warm_start: pass each probe the previous solution.

    Returns:
        A :class:`RootResult`. Check ``converged`` and ``exhausted`` — a
        search that could not bracket has still returned the useful endpoint —
        and ``breakdown``, which carries the exception when a probe could not
        be computed at all.

    Raises:
        ValueError: for a non-positive ``initial``, one outside the floor and
            ceiling, or a non-positive ``max_iterations``.
    """
    if initial <= 0.0:
        raise ValueError(f"The starting multiplier must be positive, got {initial}.")
    if not 0 < max_iterations:
        raise ValueError(
            f"At least one bisection step is needed, got {max_iterations}."
        )
    if not 0.0 <= minimum <= initial <= maximum:
        raise ValueError(
            f"The start must lie between the floor and the ceiling: got "
            f"minimum={minimum}, initial={initial}, maximum={maximum}."
        )

    sign = 1.0 if decreasing else -1.0
    goal = sign * target
    tally = {"evaluations": 0, "iterations": 0}
    # Records a probe that could not be computed, as distinct from a range
    # that ran out. See RootResult.breakdown.
    failure: dict[str, BaseException | None] = {"error": None}
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
            breakdown=failure["error"],
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
            if not minimum <= candidate <= maximum:
                break
            try:
                scaled_candidate, solution_candidate = probe(candidate)
            except _BREAKDOWN:
                break
            except ConvergenceError as error:
                # Not saturation: a probe that could not be computed. The
                # sweep still has to stop -- there is no value to widen from
                # -- but it says so rather than reporting an exhausted range.
                failure["error"] = error
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

    if limit is not None and sign * limit > goal:
        # No root above: the quantity never reaches the target however large
        # the multiplier grows. No probe: a cold solve at the endpoint is
        # numerically hopeless, and the limit is the answer.
        endpoint = min(maximum, initial * 10.0**expansions)
        return finish(endpoint, sign * limit, None, (initial, endpoint), False, "high")

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

    # Bracketed. Brent's method in log t, rather than geometric bisection:
    # the quantity is smooth in the multiplier, so interpolation converges
    # superlinearly where bisection halves the log-bracket once per solve,
    # and each probe is a linear solve.
    #
    # Probes are memoized by their log-multiplier so the two endpoints, already
    # solved by the bracketing, are not solved again, and the multiplier
    # Brent returns is read back rather than re-solved.
    from scipy.optimize import brentq

    seen: dict[float, tuple[float, Any]] = {
        float(np.log(low)): (scaled_low, solution_low),
        float(np.log(high)): (scaled, solution),
    }

    def residual(u: float) -> float:
        if u not in seen:
            seen[u] = probe(float(np.exp(u)))
        return seen[u][0] - goal

    # The bracket criterion ``high - low <= atol + rtol * (low + high)`` is,
    # in log t, a half-width of about ``rtol + atol / (low + high)``. Brent
    # stops with its bracket within twice ``xtol``, so aiming slightly inside
    # that lets the criterion below decide convergence on the bracket itself.
    eps = float(np.finfo(float).eps)
    xtol = max(0.9 * (rtol + atol / (low + high)), 4.0 * eps)
    root, report = brentq(
        residual,
        float(np.log(low)),
        float(np.log(high)),
        xtol=xtol,
        rtol=4.0 * eps,
        maxiter=max_iterations,
        full_output=True,
        disp=False,
    )
    scaled, solution = seen[root] if root in seen else probe(float(np.exp(root)))
    multiplier = float(np.exp(root))
    if scaled == goal:
        # Hit exactly -- as happens when a bracketing probe lands on the
        # root -- so there is no bracket to close.
        return finish(multiplier, scaled, solution, (multiplier, multiplier), True)
    # The tightest bracket among the probes: the largest multiplier still
    # above the goal and the smallest at or below it.
    above = [u for u, (value, _) in seen.items() if value > goal]
    below = [u for u, (value, _) in seen.items() if value <= goal]
    low = float(np.exp(max(above))) if above else low
    high = float(np.exp(min(below))) if below else high
    # Convergence is the bracket criterion, not the iteration count: with the
    # cap reached first, the answer is reported as what it is.
    converged = bool(report.converged) and high - low <= atol + rtol * (low + high)
    return finish(multiplier, scaled, solution, (low, high), converged)


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
    the honest report that the family is being factorized afresh each step.

    **The preconditioner is kept.** A preconditioner supplied as a
    :class:`LinearSolver` is otherwise rebuilt against every member of the
    family, which for an expensive one — a Woodbury surrogate with its own
    inner factorization — costs more than the solves it is accelerating. It is
    built once and reused while the multiplier stays within :attr:`refresh` of
    where it was built, and rebuilt when it wanders further. A preconditioner
    is an approximation, so reuse costs accuracy rather than correctness; the
    threshold is where that stops being a good trade.

    **A family-aware solver or preconditioner is asked once.** One that is a
    :class:`~pygeoinf2.numerics.solvers.FamilySolver` is handed the base and
    the shift up front and answers every multiplier from the work it did
    then, exactly rather than to within a refresh window: a spectral solver
    factorizes once, a probed diagonal adds the shift's own, a deflation
    moves its eigenvalues with the multiplier. The refresh rule applies only
    to preconditioners that cannot do this.
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
    reading the factors ``A``, ``Q`` and ``R`` off the operator it is given,
    and a plain sum would refuse them. Pass a family's ``at`` and the member
    arrives with its factors intact."""

    refresh: float = 10.0
    """Rebuild the preconditioner once the multiplier has moved by more than
    this factor from where it was built. Infinity never rebuilds; one always
    does."""

    _cache: dict = field(default_factory=dict, repr=False)
    _prepared: LinearSolver | None = field(default=None, repr=False)
    _prepared_at: float | None = field(default=None, repr=False)
    _family_matrices: Any = field(default=_UNSET, repr=False)
    _bound_solver: Any = field(default=None, repr=False)
    _bound_preconditioner: Any = field(default=None, repr=False)

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
                assembled = self._sum_member(multiplier)
            if self.traits is not None:
                assembled = assembled.with_traits(self.traits)
            self._cache[multiplier] = assembled
        return self._cache[multiplier]

    def _sum_member(self, multiplier: float) -> LinearOperator:
        """``base + multiplier * shift``, as a matrix where that pays.

        A direct solver factorizes a *matrix*, and it gets one by asking the
        member for it. Where the member cannot write its own matrix down --
        a normal operator, a composition, anything the ``_known_matrix``
        chain cannot read -- that costs ``dim`` applications, and the sweep
        pays it again at every multiplier although only the scalar changed.
        Extracting the two matrices once and adding them costs ``dim``
        applications twice, whatever the length of the sweep.

        Nothing is done when the member already knows its matrix: the sum of
        two matrix-backed operators reads its own, keeps a sparse array
        sparse, and there is nothing here to improve on.
        """
        member = self.base + multiplier * self.shift
        matrices = self._matrices()
        if matrices is None:
            return member
        base_matrix, shift_matrix, form = matrices
        return LinearOperator.from_matrix(
            member.domain,
            member.codomain,
            base_matrix + multiplier * shift_matrix,
            form=form,
            traits=member.traits,
        )

    def _matrices(self) -> tuple[np.ndarray, np.ndarray, str] | None:
        """The two matrices in the direct solver's form, extracted once.

        ``None`` when there is nothing to gain: no direct solver, or a member
        that can already read its own matrix.
        """
        if self._family_matrices is _UNSET:
            self._family_matrices = None
            if isinstance(self.solver, DirectSolver) and not isinstance(
                self.solver, FamilySolver
            ):
                # A family solver extracts what it needs in for_family, once;
                # assembling the member as a matrix as well would pay the
                # same applications a second time for a matrix never solved.
                form = type(self.solver).form
                # One probe of the chain, discarded: it says whether the
                # member reads its matrix or has to be applied for it, and
                # asking costs no applications either way.
                if (self.base + self.shift).known_matrix(form=form) is None:
                    self._family_matrices = (
                        self.base.matrix(form=form),
                        self.shift.matrix(form=form),
                        form,
                    )
        return self._family_matrices

    def _solver_for(self, multiplier: float, operator: LinearOperator) -> LinearSolver:
        """The solver to use, with any deferred preconditioner already built."""
        solver = self.solver
        if not isinstance(solver, IterativeSolver) or not isinstance(
            solver.preconditioner, LinearSolver
        ):
            return solver

        if isinstance(solver.preconditioner, FamilySolver):
            if self._bound_preconditioner is None:
                self._bound_preconditioner = solver.preconditioner.for_family(
                    self.base, self.shift
                )
            return solver.with_resolved_preconditioner(
                self._bound_preconditioner.at(multiplier, operator)
            )

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
        """One solve, warm-started from *x0*, returning the full result.

        Args:
            multiplier: which member of the family.
            right_hand_side: the vector to solve against.
            x0: a starting guess. Ignored by a direct solver, which does not
                iterate and so has nothing to start from.

        Returns:
            The solver's own result, with its diagnostics.
        """
        operator = self.operator(multiplier)
        if isinstance(self.solver, FamilySolver):
            if self._bound_solver is None:
                self._bound_solver = self.solver.for_family(self.base, self.shift)
            inverse = self._bound_solver.at(multiplier, operator)
            return inverse.solve(right_hand_side, x0=x0)
        solver = self._solver_for(multiplier, operator)
        return solver(operator).solve(right_hand_side, x0=x0)
