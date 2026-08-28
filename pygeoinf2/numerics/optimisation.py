"""
Unconstrained optimisation, coordinate-free.

Written against the space's own inner product rather than wrapped around
SciPy. That is not purism. v1's wrapper passes SciPy the **gradient**
components where the derivative components are wanted — verified: the ratio
between what is passed and ``dJ/dc`` is exactly the Gram diagonal — while
passing the Hessian correctly as the Galerkin matrix. The two are therefore in
different conventions within a single call, so a Newton-CG step solves
``H p = -G^-1 g`` instead of ``H p = -g``. The same file's ``line_search`` gets
it right, with a comment explaining why; the comment protects one function and
not its neighbour.

Here the confusion is unavailable. A gradient is a vector, a direction is a
vector, and the slope along a direction is their inner product. There is no
array to put in the wrong convention.

Two further consequences of working in the space rather than in components:

- **Convergence is mesh-independent.** Progress is judged on ``||grad f||`` in
  the space's own norm, which means the same thing under refinement, unlike the
  Euclidean norm of a component array.
- **Quasi-Newton methods become metric-aware for free.** The L-BFGS two-loop
  recursion is built entirely from inner products, so working in the space's
  metric preconditions it by the inverse Gram matrix at no cost. The component
  version is metric-blind, which is the conditioning half of DESIGN.md 5.6.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..algebra.operators import Functional, LinearOperator, Operator
from ..traits import Traits
from .line_search import (
    ArmijoLineSearch,
    LineSearch,
    StrongWolfeLineSearch,
)

__all__ = [
    "OptimisationResult",
    "Optimiser",
    "SteepestDescent",
    "NonlinearCG",
    "LBFGS",
    "NewtonCG",
    "TrustRegionNewton",
    "gauss_newton_hessian",
]


@dataclass(frozen=True)
class OptimisationResult:
    """Where an optimiser stopped, and why."""

    minimiser: Any
    value: float
    gradient_norm: float
    iterations: int
    evaluations: int
    converged: bool
    message: str
    history: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"OptimisationResult(value={self.value:.6g}, "
            f"|grad|={self.gradient_norm:.3g}, iterations={self.iterations}, "
            f"converged={self.converged}: {self.message})"
        )


class Optimiser(ABC):
    """Minimises a functional on a Hilbert space."""

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

                This stays tight, because loosening it would cost every caller
                accuracy to fix a reporting problem. The problem was real: a
                line-search method cannot reach ``1e-10`` in double precision
                on an ordinary objective, and on a 60-dimensional quadratic of
                condition number 4.9 that made L-BFGS stop at ``|g| == 1.8e-7``
                and report *failure* after 19 iterations, nonlinear CG do the
                same at ``1.1e-6``, and steepest descent spend 2000 iterations
                and 5330 value evaluations arriving at the same verdict. But
                the cure is to recognise the two ways a method legitimately
                finishes short of the gradient tolerance — see *ftol* and the
                precision-loss outcome — not to lower the bar. With those, the
                same four runs converge in 16 to 25 iterations and 40 to 119
                value evaluations.
            ftol: stop when one iteration decreases the value by no more than
                ``ftol * |f|``. The criterion that says "no further progress",
                which a gradient test alone cannot: near a minimiser the
                gradient is small everywhere the value is flat.

                Relative to ``|f|`` with no absolute floor, deliberately. A
                floor makes the test absolute wherever the minimum value is
                near zero, and there a slow method in a narrow valley — the
                Rosenbrock function is the standard example — takes steps
                smaller than the floor while still far from the minimiser, so
                the floor stops it early and calls that convergence. Set it to
                zero to rely on the gradient test alone.
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

    def minimise(self, functional: Functional, x0: Any, /) -> OptimisationResult:
        """Minimise ``functional`` starting from ``x0``."""
        if not functional.has_derivative:
            raise ValueError(
                f"{type(self).__name__} needs a functional with a derivative. "
                f"Supply one to Functional.from_callables."
            )
        return self._minimise(functional, x0)

    @abstractmethod
    def _minimise(self, functional: Functional, x0: Any) -> OptimisationResult:
        """Run the method, having passed validation."""

    # -- shared machinery ----------------------------------------------

    def _tolerance(self, initial_gradient_norm: float) -> float:
        return max(self._gtol, self._rtol * initial_gradient_norm)

    def _precision_limited(self, value: float, gradient_norm: float) -> bool:
        """Whether the gradient is as small as double precision allows here.

        A line search fails when no step along a descent direction improves the
        value. Near a minimiser that is not a defect: the decrease a step is
        predicted to make, ``c1 * alpha * slope``, eventually falls below the
        rounding error in the value itself, which is ``eps * |f|`` rather than
        ``eps``. Past that point the search is comparing noise.

        So a failure with the gradient already at that scale is success at the
        accuracy the arithmetic allows, and saying "line search failed" instead
        reports a wrong answer about a right one.
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
    ) -> OptimisationResult:
        """The outcome when no step improved the value."""
        limited = self._precision_limited(value, gradient_norm)
        return OptimisationResult(
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


class _DescentMethod(Optimiser):
    """A method that repeatedly picks a direction and searches along it.

    The loop is identical across steepest descent, nonlinear CG and L-BFGS;
    only the direction rule differs, so only that is left to subclasses.
    """

    def _minimise(self, functional: Functional, x0: Any) -> OptimisationResult:
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
                return OptimisationResult(
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
            new_model = functional.at(new_x)
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
                # only spends evaluations to confirm it, which is what turned a
                # converged steepest descent into 2000 iterations.
                return OptimisationResult(
                    x,
                    model.value,
                    gradient_norm,
                    iteration,
                    evaluations,
                    True,
                    "value tolerance reached",
                    history,
                )

        return OptimisationResult(
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
    steepest with respect to the inner product the modeller chose rather than
    with respect to an arbitrary coordinate basis. That difference is the whole
    of DESIGN.md 5.6, and it is why this converges at a mesh-independent rate
    where the component version does not.
    """

    def _direction(self, space: Any, gradient: Any, state: Any) -> tuple[Any, Any]:
        return space.negative(gradient), state

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
            **kwargs: passed to :class:`Optimiser`.
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
    discretisation happened to supply, which is the ill-conditioning that makes
    the same problem converge differently under refinement.
    """

    def __init__(self, /, *, memory: int = 10, **kwargs: Any) -> None:
        """
        Args:
            memory: how many correction pairs to keep. Five to twenty is usual;
                more helps a badly scaled problem and costs one inner product
                per pair per iteration.
            **kwargs: passed to :class:`Optimiser`.
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


class NewtonCG(Optimiser):
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
                once the residual falls below this fraction of the gradient
                norm, so early iterations are cheap and later ones exact.
            max_cg_iterations: cap on the inner iteration.
            **kwargs: passed to :class:`Optimiser`.
        """
        self._forcing = forcing
        self._max_cg_iterations = max_cg_iterations
        super().__init__(**kwargs)

    def minimise(self, functional: Functional, x0: Any, /) -> OptimisationResult:
        """Minimise, requiring a Hessian as well as a gradient."""
        if not functional.has_hessian:
            raise ValueError(
                "NewtonCG needs a functional with a Hessian. Use LBFGS when "
                "only a gradient is available, or supply a Gauss-Newton "
                "approximation with gauss_newton_hessian."
            )
        return super().minimise(functional, x0)

    def _minimise(self, functional: Functional, x0: Any) -> OptimisationResult:
        space = functional.domain
        x = space.copy(x0)
        model = functional.at(x)
        gradient_norm = space.norm(model.gradient)
        tolerance = self._tolerance(gradient_norm)
        history = [model.value]
        evaluations = 1

        for iteration in range(1, self._max_iterations + 1):
            if gradient_norm <= tolerance:
                return OptimisationResult(
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
            x = search.point
            model = functional.at(x)
            evaluations += 1
            gradient_norm = space.norm(model.gradient)
            history.append(model.value)

            decrease = previous_value - model.value
            if 0.0 <= decrease <= self._ftol * abs(model.value):
                return OptimisationResult(
                    x,
                    model.value,
                    gradient_norm,
                    iteration,
                    evaluations,
                    True,
                    "value tolerance reached",
                    history,
                )

        return OptimisationResult(
            x,
            model.value,
            gradient_norm,
            self._max_iterations,
            evaluations,
            gradient_norm <= tolerance,
            "iteration limit reached",
            history,
        )


class TrustRegionNewton(Optimiser):
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
            **kwargs: passed to :class:`Optimiser`.
        """
        self._initial_radius = initial_radius
        self._max_radius = max_radius
        self._acceptance = acceptance
        super().__init__(**kwargs)

    def minimise(self, functional: Functional, x0: Any, /) -> OptimisationResult:
        """Minimise, requiring a Hessian as well as a gradient."""
        if not functional.has_hessian:
            raise ValueError("TrustRegionNewton needs a functional with a Hessian.")
        return super().minimise(functional, x0)

    def _minimise(self, functional: Functional, x0: Any) -> OptimisationResult:
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
                return OptimisationResult(
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
            trial_value = functional(trial)
            evaluations += 1
            ratio = (model.value - trial_value) / predicted

            if ratio < 0.25:
                radius *= 0.25
            elif ratio > 0.75 and abs(space.norm(step) - radius) < 1e-8 * radius:
                radius = min(2.0 * radius, self._max_radius)

            if ratio > self._acceptance:
                x = trial
                model = functional.at(x)
                evaluations += 1
                gradient_norm = space.norm(model.gradient)
                history.append(model.value)

            if radius < 1e-14:
                break

        return OptimisationResult(
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
    met; or the iterate reaches the trust-region boundary. The last two return
    the point where the boundary is crossed, which is what makes the step
    useful rather than an error.

    Coordinate-free, like every other Krylov method here.

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
    rather than as a ``hessian`` that quietly omits a term. See DESIGN.md 5.5.

    It comes out positive semidefinite by the palindrome rule when the
    weighting is, with nothing claimed — so it can be handed to CG directly.

    Args:
        operator: the forward map ``F``.
        point: where to linearise.
        weighting: ``W``, typically an inverse data covariance. Defaults to the
            identity on the codomain.
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
