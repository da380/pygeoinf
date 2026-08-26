"""
Convex optimisation, coordinate-free where the geometry allows.

Which is further than one might expect. The proximal operators that matter in
practice — of a norm, of the indicator of a ball, of a squared distance — all
have closed forms written in terms of a *norm* and a *direction*, so they are
statements about the space's geometry rather than about a basis:

    prox of  t ||.||        x -> max(0, 1 - t/||x||) x
    projection onto a ball  x -> min(1, r/||x||) x

Both are metric-aware for free, and both mean the same thing under refinement.
Written in components they would instead shrink in whatever basis the
discretisation happened to supply.

What is *not* coordinate-free is the small dense subproblem a bundle method
solves over its cut coefficients. That lives in ``R^k`` for a handful of cuts
and is canonically Euclidean, so a SciPy-backed quadratic programme behind a
protocol is the right shape there — coordinates are not a constraint when the
space is genuinely finite-dimensional and has no metric of its own.

Ported from v1's ``convex_optimisation`` and ``convex_analysis``, both of which
are free of any dependence on the inversion layer. The KKT, Chambolle-Pock and
support-value machinery is entangled with it and is deliberately left behind.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..algebra.operators import Functional, LinearFunctional, LinearOperator
from ..algebra.spaces import HilbertSpace
from .optimisation import OptimisationResult, Optimiser

__all__ = [
    "SquaredDistance",
    "NormFunctional",
    "BallIndicator",
    "SupportFunction",
    "SubgradientDescent",
    "ProximalGradient",
    "ProximalPoint",
]


# --------------------------------------------------------------------- #
#                        Standard convex functionals                    #
# --------------------------------------------------------------------- #


class SquaredDistance(Functional):
    """``0.5 ||x - centre||^2`` in the space's own norm.

    Smooth, with gradient ``x - centre`` and Hessian the identity — so its
    condition number is one whatever the metric, which is the cleanest possible
    illustration of why the gradient is taken in the space rather than in the
    components.
    """

    def __init__(self, domain: HilbertSpace, /, *, centre: Any = None) -> None:
        """
        Args:
            domain: the space.
            centre: the point the distance is measured from. Defaults to zero.
        """
        super().__init__(domain)
        self._centre = domain.zero() if centre is None else centre

    @property
    def centre(self) -> Any:
        """The point the distance is measured from."""
        return self._centre

    def _value(self, x: Any) -> float:
        return 0.5 * self.domain.squared_norm(self.domain.subtract(x, self._centre))

    def _derivative(self, x: Any) -> LinearFunctional:
        return LinearFunctional.from_representer(
            self.domain, self.domain.subtract(x, self._centre)
        )

    @property
    def has_hessian(self) -> bool:
        """True: the Hessian is the identity."""
        return True

    def _hessian(self, x: Any) -> LinearOperator:
        from ..traits import Traits

        return LinearOperator.self_adjoint(
            self.domain, lambda v: v, traits=Traits.POSITIVE_DEFINITE
        )

    def prox(self, x: Any, step: float, /) -> Any:
        """``(x + step * centre) / (1 + step)``, in closed form."""
        space = self.domain
        shifted = space.axpy(step, self._centre, space.copy(x))
        return space.scale_inplace(1.0 / (1.0 + step), shifted)


class NormFunctional(Functional):
    """``weight * ||x||`` in the space's own norm.

    Non-smooth at the origin, which is the point of it: this is the
    Hilbert-space analogue of an L1 penalty, and its proximal operator is a
    shrinkage *along the vector*,

        ``prox(x) == max(0, 1 - t w / ||x||) x``

    which needs only a norm and a scaling. Nothing here refers to a basis.
    """

    def __init__(self, domain: HilbertSpace, /, *, weight: float = 1.0) -> None:
        """
        Args:
            domain: the space.
            weight: the multiplier, which must be positive.
        """
        if weight <= 0.0:
            raise ValueError("weight must be positive.")
        super().__init__(domain)
        self._weight = float(weight)

    @property
    def weight(self) -> float:
        """The multiplier."""
        return self._weight

    def _value(self, x: Any) -> float:
        return self._weight * self.domain.norm(x)

    @property
    def has_subgradient(self) -> bool:
        """True: the subdifferential is nonempty everywhere."""
        return True

    def subgradient(self, x: Any) -> Any:
        """``w x / ||x||`` away from the origin, and zero at it.

        Zero is a legitimate element of the subdifferential at the origin,
        which is the unit ball there.
        """
        space = self.domain
        norm = space.norm(x)
        if norm == 0.0:
            return space.zero()
        return space.scale(self._weight / norm, x)

    def prox(self, x: Any, step: float, /) -> Any:
        """Shrinkage along the vector, by ``step * weight`` in norm."""
        space = self.domain
        norm = space.norm(x)
        if norm == 0.0:
            return space.zero()
        scaling = max(0.0, 1.0 - step * self._weight / norm)
        return space.scale(scaling, x)


class BallIndicator(Functional):
    """Zero inside a closed ball, infinite outside it.

    The way a hard constraint enters a proximal method: its proximal operator
    is the projection onto the ball, ``x -> min(1, r/||x||) x``, which is again
    a statement about the norm alone.
    """

    def __init__(
        self, domain: HilbertSpace, /, *, radius: float = 1.0, centre: Any = None
    ) -> None:
        """
        Args:
            domain: the space.
            radius: the ball's radius, which must be positive.
            centre: the ball's centre. Defaults to zero.
        """
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        super().__init__(domain)
        self._radius = float(radius)
        self._centre = domain.zero() if centre is None else centre

    @property
    def radius(self) -> float:
        """The ball's radius."""
        return self._radius

    @property
    def centre(self) -> Any:
        """The ball's centre."""
        return self._centre

    def _value(self, x: Any) -> float:
        offset = self.domain.norm(self.domain.subtract(x, self._centre))
        return 0.0 if offset <= self._radius * (1.0 + 1e-12) else float("inf")

    def prox(self, x: Any, step: float, /) -> Any:
        """The projection onto the ball, which does not depend on the step."""
        space = self.domain
        offset = space.subtract(x, self._centre)
        norm = space.norm(offset)
        if norm <= self._radius:
            return space.copy(x)
        return space.axpy(self._radius / norm, offset, space.copy(self._centre))

    def conjugate(self) -> Functional:
        """The support function of the ball."""
        return SupportFunction.of_ball(
            self.domain, radius=self._radius, centre=self._centre
        )


class SupportFunction(Functional):
    """``h(y) == sup { (y, x) : x in K }`` for a convex set ``K``.

    A functional on the same space as the set, which is a dividend of Riesz
    identification: a support function is classically a function on the dual,
    and without the identification every duality argument would carry a
    transport map with it.

    Only the constructions with no dependence on the inversion layer are
    brought across: a ball, a point, a Minkowski sum, a linear image and a
    positive scaling. The algebra is closed, which is the point of having the
    class at all.
    """

    def __init__(self, domain: HilbertSpace, /) -> None:
        """
        Args:
            domain: the space the set lives in.
        """
        super().__init__(domain)

    @abstractmethod
    def _value(self, y: Any) -> float:
        """The supremum of the pairing over the set."""

    @property
    def has_subgradient(self) -> bool:
        """True: a maximiser is a subgradient."""
        return True

    def subgradient(self, y: Any) -> Any:
        """A maximiser of the pairing, which is a subgradient of the support."""
        return self._maximiser(y)

    @abstractmethod
    def _maximiser(self, y: Any) -> Any:
        """A point of the set attaining the supremum."""

    @staticmethod
    def of_ball(
        domain: HilbertSpace, /, *, radius: float = 1.0, centre: Any = None
    ) -> SupportFunction:
        """The support function of a ball: ``r ||y|| + (centre, y)``."""
        return _BallSupport(domain, radius=radius, centre=centre)

    @staticmethod
    def of_point(domain: HilbertSpace, point: Any, /) -> SupportFunction:
        """The support function of a single point: the linear functional ``(p, .)``."""
        return _PointSupport(domain, point)

    def __add__(self, other: Functional) -> Functional:
        """A sum of support functions is the support of the Minkowski sum."""
        if isinstance(other, SupportFunction) and other.domain == self.domain:
            return _MinkowskiSupport((self, other))
        return super().__add__(other)

    def __mul__(self, alpha: float) -> Functional:
        """A positive scaling scales the set."""
        if isinstance(alpha, (int, float)) and alpha > 0.0:
            return _ScaledSupport(self, float(alpha))
        return super().__mul__(alpha)

    def compose_with(self, operator: LinearOperator, /) -> SupportFunction:
        """The support function of the image ``A K``.

        For ``K`` in this space and ``A`` mapping out of it, ``A K`` lives in
        the codomain and ``h_{AK}(y) == h_K(A* y)``. So the result is a
        functional on the **codomain**, and the operator must map *out of* the
        set's space rather than into it.
        """
        if operator.domain != self.domain:
            raise ValueError(
                f"The operator must map out of {self.domain!r}, the space the "
                f"set lives in, not out of {operator.domain!r}."
            )
        return _ImageSupport(self, operator)


class _BallSupport(SupportFunction):
    """The support function of a ball."""

    def __init__(
        self, domain: HilbertSpace, /, *, radius: float = 1.0, centre: Any = None
    ) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        super().__init__(domain)
        self._radius = float(radius)
        self._centre = domain.zero() if centre is None else centre

    def _value(self, y: Any) -> float:
        return self._radius * self.domain.norm(y) + self.domain.inner_product(
            self._centre, y
        )

    def _maximiser(self, y: Any) -> Any:
        space = self.domain
        norm = space.norm(y)
        if norm == 0.0:
            return space.copy(self._centre)
        return space.axpy(self._radius / norm, y, space.copy(self._centre))


class _PointSupport(SupportFunction):
    """The support function of a single point."""

    def __init__(self, domain: HilbertSpace, point: Any, /) -> None:
        super().__init__(domain)
        self._point = point

    def _value(self, y: Any) -> float:
        return self.domain.inner_product(self._point, y)

    def _maximiser(self, y: Any) -> Any:
        return self.domain.copy(self._point)


class _MinkowskiSupport(SupportFunction):
    """The support function of a Minkowski sum, which is a sum of supports."""

    def __init__(self, parts: tuple[SupportFunction, ...], /) -> None:
        super().__init__(parts[0].domain)
        self._parts = parts

    @property
    def parts(self) -> tuple[SupportFunction, ...]:
        """The summands."""
        return self._parts

    def _value(self, y: Any) -> float:
        return float(sum(part(y) for part in self._parts))

    def _maximiser(self, y: Any) -> Any:
        space = self.domain
        result = space.zero()
        for part in self._parts:
            result = space.axpy(1.0, part.subgradient(y), result)
        return result


class _ScaledSupport(SupportFunction):
    """The support function of a positively scaled set."""

    def __init__(self, base: SupportFunction, alpha: float, /) -> None:
        super().__init__(base.domain)
        self._base = base
        self._alpha = alpha

    def _value(self, y: Any) -> float:
        return self._alpha * self._base(y)

    def _maximiser(self, y: Any) -> Any:
        return self.domain.scale(self._alpha, self._base.subgradient(y))


class _ImageSupport(SupportFunction):
    """The support function of a linear image, which is ``h(A* y)``."""

    def __init__(self, base: SupportFunction, operator: LinearOperator, /) -> None:
        super().__init__(operator.codomain)
        self._base = base
        self._operator = operator

    def _value(self, y: Any) -> float:
        return self._base(self._operator.adjoint(y))

    def _maximiser(self, y: Any) -> Any:
        return self._operator(self._base.subgradient(self._operator.adjoint(y)))


# --------------------------------------------------------------------- #
#                          Non-smooth methods                           #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class _ConvexResult:
    """Internal: the running state a convex method reports."""

    best_point: Any
    best_value: float
    history: list[float] = field(default_factory=list)


class SubgradientDescent(Optimiser):
    """Subgradient descent with a diminishing step.

    v1's implementation uses a *constant* step and says in its own docstring
    that convergence is not guaranteed — which is correct, and means it is not
    a usable method. The step rules here are the ones that do converge:

    - ``"sqrt"``: ``a / sqrt(k)``, the classical square-summable-but-not-
      summable choice, giving ``O(1/sqrt(k))`` on a Lipschitz convex function.
    - ``"inverse"``: ``a / k``, better when the function is strongly convex.
    - ``"polyak"``: ``(f(x) - f*) / ||g||^2`` when a target value ``f*`` is
      known, which is much faster and is available whenever the optimum is
      known to be zero — as it is for a feasibility problem.

    Subgradient methods are not descent methods: the value need not decrease.
    The best point seen is tracked and returned, which is what the theory
    bounds.
    """

    def __init__(
        self,
        /,
        *,
        step_size: float = 1.0,
        rule: Literal["sqrt", "inverse", "constant", "polyak"] = "sqrt",
        target_value: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            step_size: the scale ``a`` in the step rule.
            rule: which diminishing rule to use.
            target_value: the known optimal value, required by ``"polyak"``.
            **kwargs: passed to :class:`~pygeoinf2.numerics.optimisation.Optimiser`.
        """
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        if rule not in ("sqrt", "inverse", "constant", "polyak"):
            raise ValueError(f"Unknown rule {rule!r}.")
        if rule == "polyak" and target_value is None:
            raise ValueError("The polyak rule needs a target_value.")
        self._step_size = step_size
        self._rule = rule
        self._target_value = target_value
        super().__init__(**kwargs)

    def minimise(self, functional: Functional, x0: Any, /) -> OptimisationResult:
        """Minimise, requiring only a subgradient."""
        if not functional.has_subgradient:
            raise ValueError(
                "SubgradientDescent needs a functional with a subgradient. "
                "A smooth functional supplies one from its gradient."
            )
        return self._minimise(functional, x0)

    def _step(self, iteration: int, value: float, subgradient_norm: float) -> float:
        if self._rule == "constant":
            return self._step_size
        if self._rule == "sqrt":
            return self._step_size / np.sqrt(iteration)
        if self._rule == "inverse":
            return self._step_size / iteration
        gap = value - self._target_value
        if subgradient_norm == 0.0:
            return 0.0
        return max(gap, 0.0) / subgradient_norm**2

    def _minimise(self, functional: Functional, x0: Any) -> OptimisationResult:
        space = functional.domain
        x = space.copy(x0)
        best_point, best_value = space.copy(x), functional(x)
        history = [best_value]
        evaluations = 1

        for iteration in range(1, self._max_iterations + 1):
            subgradient = functional.subgradient(x)
            norm = space.norm(subgradient)
            if norm <= self._gtol:
                return OptimisationResult(
                    best_point,
                    best_value,
                    norm,
                    iteration - 1,
                    evaluations,
                    True,
                    "subgradient tolerance reached",
                    history,
                )

            step = self._step(iteration, functional(x), norm)
            evaluations += 1
            x = space.axpy(-step, subgradient, x)

            value = functional(x)
            evaluations += 1
            history.append(value)
            if value < best_value:
                best_value, best_point = value, space.copy(x)

        return OptimisationResult(
            best_point,
            best_value,
            norm,
            self._max_iterations,
            evaluations,
            False,
            "iteration limit reached",
            history,
        )


class ProximalGradient(Optimiser):
    """Proximal gradient descent for ``f + g``, optionally accelerated.

    ``f`` must be smooth and ``g`` must have a proximal operator; either may be
    absent. With acceleration this is FISTA, converging at ``O(1/k^2)`` against
    the plain method's ``O(1/k)``.

    The step is the reciprocal of a Lipschitz constant for ``grad f``, found by
    backtracking when none is given. Because the proximal operator is taken in
    the space's norm, the whole method is metric-aware: the same problem
    discretised twice takes the same number of iterations.
    """

    def __init__(
        self,
        /,
        *,
        step: float | None = None,
        accelerated: bool = True,
        backtracking: float = 0.5,
        max_backtracks: int = 40,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            step: a fixed step. When None, it is found by backtracking.
            accelerated: use the FISTA momentum sequence.
            backtracking: the factor a rejected step is multiplied by.
            max_backtracks: give up after this many contractions.
            **kwargs: passed to :class:`~pygeoinf2.numerics.optimisation.Optimiser`.
        """
        self._step = step
        self._accelerated = accelerated
        self._backtracking = backtracking
        self._max_backtracks = max_backtracks
        super().__init__(**kwargs)

    def minimise(
        self,
        smooth: Functional,
        x0: Any,
        /,
        *,
        nonsmooth: Functional | None = None,
    ) -> OptimisationResult:
        """Minimise ``smooth + nonsmooth`` from ``x0``.

        Args:
            smooth: the differentiable part ``f``.
            x0: the starting point.
            nonsmooth: the part ``g`` with a proximal operator. When absent,
                this is plain gradient descent with a backtracked step.
        """
        if not smooth.has_derivative:
            raise ValueError("The smooth part needs a gradient.")
        if nonsmooth is not None and not nonsmooth.has_prox:
            raise ValueError(
                "The non-smooth part needs a proximal operator. NormFunctional "
                "and BallIndicator provide one in closed form."
            )
        return self._run(smooth, nonsmooth, x0)

    def _minimise(self, functional: Functional, x0: Any) -> OptimisationResult:
        return self._run(functional, None, x0)

    def _run(
        self, smooth: Functional, nonsmooth: Functional | None, x0: Any
    ) -> OptimisationResult:
        space = smooth.domain

        def total(point: Any) -> float:
            return smooth(point) + (0.0 if nonsmooth is None else nonsmooth(point))

        def proximal(point: Any, step: float) -> Any:
            return point if nonsmooth is None else nonsmooth.prox(point, step)

        x = space.copy(x0)
        y = space.copy(x)
        momentum = 1.0
        step = self._step if self._step is not None else self._initial_step(smooth, x)
        history = [total(x)]
        evaluations = 1

        for iteration in range(1, self._max_iterations + 1):
            model = smooth.at(y)
            gradient = model.gradient
            evaluations += 1

            step, candidate = self._advance(
                smooth, proximal, y, model.value, gradient, step
            )
            if candidate is None:
                return OptimisationResult(
                    x,
                    total(x),
                    space.norm(gradient),
                    iteration,
                    evaluations,
                    False,
                    "backtracking failed to find a usable step",
                    history,
                )

            progress = space.norm(space.subtract(candidate, x)) / max(step, 1e-300)
            previous_x, x = x, candidate

            if self._accelerated:
                next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
                y = space.axpy(
                    (momentum - 1.0) / next_momentum,
                    space.subtract(x, previous_x),
                    space.copy(x),
                )
                momentum = next_momentum
            else:
                y = space.copy(x)

            history.append(total(x))
            if progress <= self._gtol:
                return OptimisationResult(
                    x,
                    history[-1],
                    progress,
                    iteration,
                    evaluations,
                    True,
                    "step tolerance reached",
                    history,
                )

        return OptimisationResult(
            x,
            history[-1],
            progress,
            self._max_iterations,
            evaluations,
            False,
            "iteration limit reached",
            history,
        )

    @staticmethod
    def _initial_step(smooth: Functional, x: Any) -> float:
        """A step from a two-point estimate of the gradient's Lipschitz constant.

        Since the step only ever decreases afterwards, starting from a
        reasonable estimate rather than from one matters: an initial step far
        below the stable one is never recovered.
        """
        space = smooth.domain
        gradient = smooth.gradient(x)
        norm = space.norm(gradient)
        if norm == 0.0:
            return 1.0
        offset = space.scale(1e-6 / norm, gradient)
        nearby = space.axpy(1.0, offset, space.copy(x))
        change = space.norm(space.subtract(smooth.gradient(nearby), gradient))
        lipschitz = change / space.norm(offset)
        return 1.0 / lipschitz if lipschitz > 0.0 else 1.0

    def _advance(
        self,
        smooth: Functional,
        proximal: Any,
        y: Any,
        value: float,
        gradient: Any,
        step: float,
    ) -> tuple[float, Any | None]:
        """One proximal step, backtracking on the quadratic upper bound."""
        space = smooth.domain
        if self._step is not None:
            trial = proximal(
                space.axpy(-self._step, gradient, space.copy(y)), self._step
            )
            return self._step, trial

        # The step is monotonically non-increasing, as in Beck and Teboulle.
        # Growing it each iteration looks appealing but is unsafe: as the
        # gradient vanishes the sufficient-decrease test degenerates, so any
        # step passes, the step runs away, and the proximal operator then
        # slams the iterate to the minimiser of the non-smooth part alone.
        for _ in range(self._max_backtracks):
            trial = proximal(space.axpy(-step, gradient, space.copy(y)), step)
            difference = space.subtract(trial, y)
            bound = (
                value
                + space.inner_product(gradient, difference)
                + space.squared_norm(difference) / (2.0 * step)
            )
            if smooth(trial) <= bound * (1.0 + 1e-12) + 1e-300:
                return step, trial
            step *= self._backtracking
        return step, None


class ProximalPoint(Optimiser):
    """The proximal point method: repeated proximal steps on the objective.

    Unconditionally stable for any positive step, and the conceptual parent of
    the augmented-Lagrangian and ADMM families. Practical only when the
    proximal operator is available in closed form, which is why it is offered
    alongside functionals that have one.
    """

    def __init__(self, /, *, step: float = 1.0, **kwargs: Any) -> None:
        """
        Args:
            step: the proximal parameter. Larger steps converge in fewer
                iterations and make each one harder.
            **kwargs: passed to :class:`~pygeoinf2.numerics.optimisation.Optimiser`.
        """
        if step <= 0.0:
            raise ValueError("step must be positive.")
        self._step = step
        super().__init__(**kwargs)

    def minimise(self, functional: Functional, x0: Any, /) -> OptimisationResult:
        """Minimise, requiring a proximal operator."""
        if not functional.has_prox:
            raise ValueError("ProximalPoint needs a functional with a prox.")
        return self._minimise(functional, x0)

    def _minimise(self, functional: Functional, x0: Any) -> OptimisationResult:
        space = functional.domain
        x = space.copy(x0)
        history = [functional(x)]

        for iteration in range(1, self._max_iterations + 1):
            nxt = functional.prox(x, self._step)
            movement = space.norm(space.subtract(nxt, x))
            x = nxt
            history.append(functional(x))
            if movement <= self._gtol:
                return OptimisationResult(
                    x,
                    history[-1],
                    movement,
                    iteration,
                    iteration + 1,
                    True,
                    "step tolerance reached",
                    history,
                )

        return OptimisationResult(
            x,
            history[-1],
            movement,
            self._max_iterations,
            self._max_iterations + 1,
            False,
            "iteration limit reached",
            history,
        )
