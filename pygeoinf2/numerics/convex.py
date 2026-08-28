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

import warnings

import numpy as np
from numpy.random import Generator

from ..algebra.operators import Functional, LinearFunctional, LinearOperator
from ..algebra.spaces import HilbertSpace
from .optimisation import OptimisationResult, Optimiser

__all__ = [
    "ChambollePockSolver",
    "PrimalKKTSolver",
    "KKTResult",
    "SaddlePointResult",
    "ProximalBundleMethod",
    "LevelBundleMethod",
    "BundleResult",
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
            radius: the ball's radius, which must not be negative. Zero gives
                the indicator of a single point, whose prox is the constant map
                to the centre.
            centre: the ball's centre. Defaults to zero.

        Raises:
            ValueError: if the radius is negative.
        """
        if radius < 0.0:
            raise ValueError("radius must not be negative.")
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
        """The support function of a ball: ``r ||y|| + (centre, y)``.

        Args:
            domain: the space the ball lives in.
            radius: its radius, in the space's own norm.
            centre: its centre. The origin if omitted, which makes the
                support function the norm alone.

        Returns:
            The support function.
        """
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

        Args:
            operator: ``A``, whose domain is this set's space.

        Returns:
            The support function of the image, on the codomain.

        Raises:
            ValueError: if the operator does not map out of this space -- the
                commonest way to get this wrong is to pass the adjoint.
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
        if radius < 0.0:
            raise ValueError("radius must not be negative.")
        super().__init__(domain)
        self._radius = float(radius)
        self._centre = domain.zero() if centre is None else centre

    def _value(self, y: Any) -> float:
        # At radius zero this is the point support ``(centre, y)``, which is
        # what it should be: the support function of a single point.
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
        """Minimise, requiring only a subgradient.

        Args:
            functional: convex, and able to supply a subgradient. It need not
                be differentiable, which is the point of this method.
            x0: where to start.

        Returns:
            The optimisation result.

        Raises:
            ValueError: if the functional cannot supply a subgradient.
        """
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

        Returns:
            The optimisation result.

        Raises:
            ValueError: if the two parts live on different spaces, or the
                nonsmooth one has no proximal operator -- without which this
                is not a proximal method.
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
        """Minimise, requiring a proximal operator.

        Args:
            functional: convex, with a proximal operator. A set's indicator
                is the usual case, its prox being the projection.
            x0: where to start.

        Returns:
            The optimisation result.

        Raises:
            ValueError: if the functional has no proximal operator.
        """
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


@dataclass(frozen=True)
class BundleResult:
    """The outcome of a bundle minimisation.

    The field names are
    :class:`~pygeoinf2.numerics.optimisation.OptimisationResult`'s, so a caller
    can read either without looking up which it has. ``value`` used to be
    ``minimum``, which said the same thing in a different word and made the two
    gratuitously incompatible.
    """

    value: float
    """The least value found."""

    minimiser: Any
    """Where it was found."""

    iterations: int
    """How many bundle iterations were taken."""

    evaluations: int
    """How many times the functional was evaluated."""

    converged: bool
    """Whether the gap fell below the tolerance."""

    message: str
    """Why it stopped."""

    gap: float
    """The model's predicted decrease at the last step.

    A *practical* stopping criterion, and stated as one. The cutting-plane
    model lies below the function everywhere, so the unregularised gap would
    bound ``f(centre) - f*``; this one includes the proximal term, so it is
    the decrease predicted under that regularisation and is generally smaller.
    It behaves like a bound and is the right thing to stop on, but it is not
    certified. A certified global bound is what the level bundle method's LP
    gives, and that is D-13.
    """

    def __repr__(self) -> str:
        return (
            f"BundleResult(value={self.value:.6g}, "
            f"iterations={self.iterations}, converged={self.converged})"
        )


class ProximalBundleMethod:
    """Minimise a convex function from values and subgradients alone.

    A subgradient is a *lower* bound on a convex function everywhere, not just
    near where it was taken. A bundle method keeps the ones it has seen and
    minimises their upper envelope — the best piecewise-linear model the
    information so far supports — with a proximal term to stop it running away
    from where the model is trustworthy.

    That is what makes it the right method for a support function: subgradient
    descent needs a step-size schedule chosen in advance and converges at
    ``1/sqrt(k)``, while here the model's own gap gives a stopping criterion
    that means something -- see :attr:`BundleResult.gap` for exactly how much,
    which is less than "a genuine bound" as this used to say.
    """

    def __init__(
        self,
        /,
        *,
        weight: float = 1.0,
        tolerance: float = 1e-8,
        iterations: int = 200,
        capacity: int = 40,
        descent: float = 0.1,
    ) -> None:
        """
        Args:
            weight: the proximal weight. Larger keeps steps shorter.
            tolerance: stop when the model's gap falls below this.
            iterations: the cap.
            capacity: how many cuts to keep; the oldest are dropped.
            descent: the fraction of the predicted decrease a step must
                deliver to be accepted as a serious step.
        """
        if not 0.0 < descent < 1.0:
            raise ValueError(f"The descent fraction lies in (0, 1), got {descent}.")
        self._weight = weight
        self._tolerance = tolerance
        self._iterations = iterations
        self._capacity = capacity
        self._descent = descent
        # The cuts a stored Gram matrix was built from, and the matrix. Reset
        # at the start of each minimisation.
        self._cache: tuple[list[int], np.ndarray] = ([], np.empty((0, 0)))

    def minimise(
        self,
        functional: Functional,
        start: Any,
        /,
        *,
        subgradient: Any = None,
    ) -> BundleResult:
        """Minimise a convex functional from a starting point.

        Args:
            functional: convex, and able to supply a subgradient.
            start: where to begin.
            subgradient: an override, called with a point and returning a
                subgradient. Defaults to the functional's own.

        Returns:
            The minimum, a minimiser, and the gap that certifies it.
        """
        space = functional.domain
        slope = subgradient or functional.subgradient

        self._cache = ([], np.empty((0, 0)))
        centre = space.copy(start)
        best = float(functional(centre))
        # Each cut is (gradient, value, point). The point matters: a cut taken
        # elsewhere bounds f from below everywhere, but its offset *at the
        # current centre* is f(x_i) + (g_i, c - x_i), and dropping the second
        # term makes every cut look tight and the method stop at once.
        cuts: list[tuple[Any, float, Any]] = []
        weight = self._weight
        gap = float("inf")

        # The bundle starts with the cut at the starting point; every later
        # cut is taken at the *candidate*.
        #
        # This is the part that was wrong. The subgradient used to be taken at
        # the centre at the top of each iteration, and a null step leaves the
        # centre where it was -- so a null step added a cut identical to one
        # already in the bundle and learned nothing from the trial point it had
        # just paid to evaluate. Duplicate cuts also make the model's Gram
        # matrix exactly singular, and it showed: across the Backus tests the
        # median condition number was 1e67 and the worst was infinite, which is
        # why the subproblem could not be solved to any useful tolerance.
        # Taking the cut where the candidate is, which is what makes a null
        # step informative, is the textbook arrangement.
        cuts.append((space.copy(slope(centre)), best, space.copy(centre)))
        evaluations = 1

        for iteration in range(1, self._iterations + 1):
            candidate, gap = self._solve_model(space, centre, best, cuts, weight)
            if gap <= self._tolerance * max(abs(best), 1.0):
                return BundleResult(
                    best, centre, iteration, evaluations, True,
                    "gap tolerance reached", gap,
                )

            value = float(functional(candidate))
            evaluations += 1
            cuts.append((space.copy(slope(candidate)), value, space.copy(candidate)))
            if len(cuts) > self._capacity:
                cuts.pop(0)

            if best - value >= self._descent * gap:
                centre, best = candidate, value  # a serious step
                weight = max(weight * 0.5, 1e-8)
            else:
                weight = min(weight * 2.0, 1e12)  # a null step: trust less

        return BundleResult(
            best, centre, self._iterations, evaluations, False,
            "iteration limit reached", gap,
        )

    def _gram(self, space: Any, cuts: Any) -> np.ndarray:
        """The cuts' Gram matrix, extended rather than rebuilt.

        A bundle grows by one cut per iteration and drops the oldest when it
        is full, so all but one row and column of this matrix are the same as
        last time. Rebuilding it cost ``k^2`` inner products per iteration and
        ``k^3`` over a run, on a space where an inner product is the expensive
        operation.

        Keyed on the identity of the cuts, so a dropped cut is noticed: the
        stored rows are matched against the current bundle and only the ones
        that are new are computed.
        """
        keys = [id(gradient) for gradient, _, _ in cuts]
        stored_keys, stored = self._cache
        # Where the kept cuts sit in the stored matrix. A cut that has been
        # dropped from the front simply does not appear.
        positions = {key: index for index, key in enumerate(stored_keys)}
        order = [positions.get(key) for key in keys]

        count = len(cuts)
        gram = np.empty((count, count))
        for i, source in enumerate(order):
            for j, target in enumerate(order[: i + 1]):
                if source is not None and target is not None:
                    value = stored[source, target]
                else:
                    value = space.inner_product(cuts[i][0], cuts[j][0])
                gram[i, j] = gram[j, i] = value

        self._cache = (keys, gram)
        return gram

    def _solve_model(
        self,
        space: Any,
        centre: Any,
        value: float,
        cuts: Any,
        weight: float,
    ) -> tuple[Any, float]:
        """Minimise the cutting-plane model plus a proximal term.

        The dual of that quadratic program is a simplex-constrained least
        squares in the *number of cuts*, which is small — so it is solved
        there rather than in the space, whose dimension is the data's.

        Returns the candidate and the predicted decrease, which is the model's
        gap: a genuine lower bound on how much is left to gain, and the only
        stopping criterion here that means anything.
        """
        gram = self._gram(space, cuts)

        errors = np.array(
            [
                max(
                    value
                    - taken
                    - space.inner_product(gradient, space.subtract(centre, point)),
                    0.0,
                )
                for gradient, taken, point in cuts
            ]
        )

        step_size = 1.0 / weight
        weights = _minimise_on_simplex(step_size * gram, -errors)

        combination = space.zero()
        for coefficient, (gradient, _, _) in zip(weights, cuts):
            combination = space.axpy(coefficient, gradient, combination)
        candidate = space.add(centre, space.scale(-step_size, combination))
        decrease = 0.5 * step_size * space.squared_norm(combination) + float(
            weights @ errors
        )
        return candidate, max(decrease, 0.0)


def _project_on_simplex(vector: np.ndarray) -> np.ndarray:
    """The nearest point of the unit simplex, in closed form.

    Sort, take a running mean, and find where it stops being feasible. Exact,
    and ``O(k log k)`` -- which matters because a bundle method solves a
    simplex-constrained problem at every single iteration.
    """
    size = vector.size
    ordered = np.sort(vector)[::-1]
    running = (np.cumsum(ordered) - 1.0) / np.arange(1, size + 1)
    count = int(np.nonzero(ordered - running > 0)[0][-1]) + 1
    return np.clip(vector - running[count - 1], 0.0, None)


def _minimise_on_simplex(
    quadratic: np.ndarray,
    linear: np.ndarray,
    /,
    *,
    iterations: int = 1000,
    tolerance: float = 1e-8,
    warn_above: float = 1e-4,
) -> np.ndarray:
    """Minimise ``w' Q w / 2 - l' w`` over the unit simplex.

    Accelerated projected gradient -- FISTA -- with a step from the largest
    eigenvalue. Handing this to a general nonlinear solver instead cost fifty
    seconds per bundle minimisation, almost all of it in setting up problems
    this small.

    **Accelerated, because plain projected gradient was not converging here.**
    The cuts a bundle accumulates are near-parallel by the end -- that is what
    it means for the method to be closing in -- so ``Q`` is a Gram matrix of
    near-dependent vectors and badly conditioned. Measured over 30 random
    bundles of that shape, at 400 iterations each, the accelerated method
    leaves a smaller KKT residual on 93 per cent of them, by a median factor
    of 3300, and by more than tenfold on 93 per cent.

    It is not uniformly better -- momentum is not monotone, and on 2 of the 30
    it still ends behind. Keeping the best iterate seen rather than the last
    is what took the median from 557 to 3300 and the tenfold share from 87 per
    cent to 93, and it costs nothing: the residual is already computed every
    step for the stopping test.

    Stopped on the KKT residual rather than on the iterate standing still. The
    two are not the same: a small step means the *method* has slowed, which is
    what an ill-conditioned problem does far from the optimum, while the
    residual is a statement about the answer. At the minimum the gradient is
    constant across the support and no smaller anywhere off it, so
    ``max_{w_i > 0} g_i - min_i g_i`` is zero there and is the residual used.

    Exhausting the iterations warns -- but only when the residual left is big
    enough to matter, which is what ``warn_above`` is for. It used to return
    silently, and a bundle method calls this at every step, so a subproblem
    quietly failing showed up only as an outer method that would not settle.

    The two thresholds are separate because the residual has a floor that no
    iteration count removes. A converging bundle's cuts become near-parallel --
    that is what converging means for it -- so ``Q`` is a Gram matrix of nearly
    dependent vectors: measured across the Backus tests, the median condition
    number is 1e23. The residual distribution there is 4.8e-7 at the median and
    3.5e-3 at the 99th percentile, so a single threshold either warns half the
    time or never. The accuracy answer for the hard tail is a proper QP
    backend on the ``k``-variable dual, which is D-13.

    Args:
        quadratic: the ``Q`` above, symmetric positive semidefinite.
        linear: the ``l`` above.
        iterations: the cap.
        tolerance: the KKT residual to stop at, relative to the gradient's
            scale.
        warn_above: warn when the iterations run out with the residual still
            above this. Set to infinity to silence it.

    Returns:
        The minimising weights.
    """
    size = linear.size
    step = 1.0 / max(float(np.linalg.eigvalsh(quadratic).max()), 1e-12)
    weights = np.full(size, 1.0 / size)
    # The extrapolated point the gradient is taken at, and the momentum
    # coefficient's running term.
    lookahead = weights.copy()
    momentum = 1.0
    # The best iterate seen. Acceleration is not monotone, so without this it
    # can end worse than where it passed through -- and measured over 30
    # random bundles it did end worse than the plain method on 2 of them.
    best, residual = weights.copy(), float("inf")

    for _ in range(iterations):
        gradient = quadratic @ weights - linear
        support = weights > 0.0
        current = float(gradient[support].max() - gradient.min())
        if current < residual:
            best, residual = weights.copy(), current
        if residual <= tolerance * max(float(np.abs(gradient).max()), 1.0):
            return best

        moved = _project_on_simplex(
            lookahead - step * (quadratic @ lookahead - linear)
        )
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        lookahead = moved + ((momentum - 1.0) / next_momentum) * (moved - weights)
        if np.max(np.abs(moved - weights)) < 1e-15:
            weights = moved
            break
        weights, momentum = moved, next_momentum

    gradient = quadratic @ weights - linear
    current = float(gradient[weights > 0.0].max() - gradient.min())
    if current < residual:
        best, residual = weights.copy(), current
    weights = best

    scale = max(float(np.abs(quadratic @ weights - linear).max()), 1.0)
    if residual > warn_above * scale:
        warnings.warn(
            f"The bundle subproblem did not converge in {iterations} "
            f"iterations; the KKT residual is {residual / scale:.3g} relative. "
            "The outer method will still make progress, but its gap is only "
            "as good as this solve.",
            RuntimeWarning,
            stacklevel=2,
        )
    return weights


class LevelBundleMethod:
    """Minimise a convex function, with a *certified* bound on how far off it is.

    The other bundle method here, :class:`ProximalBundleMethod`, stops on its
    model's predicted decrease, which behaves like a bound and is not one. This
    one keeps a genuine global lower bound and stops on the gap between it and
    the best value seen -- so when it says the answer is within a tolerance,
    that is a statement about the minimum and not about the method.

    The bound comes from the cutting-plane model itself. Every cut is an affine
    *under*-estimate of a convex function everywhere, so the model
    ``max_j [f_j + (g_j, x - x_j)]`` lies below ``f``, and *its* minimum lies
    below ``f*``. That minimum is a linear programme -- minimise ``t`` subject
    to every cut being at or below it -- and it is solved as one, by the
    simplex method, which is more reliable on a nearly unbounded LP than
    handing it to a QP solver.

    Each step then asks for a point that reaches a *level* between the bound
    and the best value, and is as close as possible to the stability centre:

    .. code-block:: text

        minimise    ||x - centre||^2 / 2
        subject to  f_j + (g_j, x - x_j) <= t   for every cut
                    t <= level

    with ``level == alpha * lower + (1 - alpha) * upper``. Small ``alpha`` aims
    close to the lower bound and makes fast progress when it succeeds; the QP
    can then be infeasible, because no point reaches that level. That is
    handled rather than raised: ``alpha`` is widened towards one and retried,
    and if it still fails a proximal step is taken instead, so the centre and
    the bundle always advance.

    **Needs a coordinate space.** The master problem is a QP in the domain's
    components, so unlike the proximal method -- which works through inner
    products alone -- this one cannot run on a space without a basis. In
    practice the variable is a dual one on the data space, which has one.

    A port of v1's ``LevelBundleMethod``. Its author's view on the API is to be
    sought before it is changed beyond the port.
    """

    def __init__(
        self,
        /,
        *,
        alpha: float = 0.1,
        tolerance: float = 1e-6,
        iterations: int = 500,
        capacity: int = 100,
        qp_solver: Any = None,
    ) -> None:
        """
        Args:
            alpha: where between the lower bound and the best value the level
                is set. Towards zero aims at the bound and is aggressive;
                towards one is cautious. In ``(0, 1)``.
            tolerance: stop when the gap falls to this fraction of the best
                value. **Relative**, unlike a bare number would suggest.
            iterations: the cap on oracle calls.
            capacity: how many cuts to keep. The oldest are dropped.
            qp_solver: the backend for the master problem. Defaults to
                :func:`~pygeoinf2.numerics.quadratic_programming.best_available_qp_solver`.

        Raises:
            ValueError: if *alpha* is not in ``(0, 1)``.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha lies in (0, 1), got {alpha}.")
        self._alpha = alpha
        self._tolerance = tolerance
        self._iterations = iterations
        self._capacity = capacity
        self._qp_solver = qp_solver

    @property
    def qp_solver(self) -> Any:
        """The backend the master problem is handed to."""
        if self._qp_solver is None:
            from .quadratic_programming import best_available_qp_solver

            self._qp_solver = best_available_qp_solver()
        return self._qp_solver

    @staticmethod
    def _cut_rows(
        space: Any, cuts: list[tuple[np.ndarray, float, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """The cuts as ``[g_j, -1] z <= (g_j, x_j) - f_j``.

        With ``z == [x, t]``: the constraint ``f_j + (g_j, x - x_j) <= t``
        rearranged so that every term in ``x`` and ``t`` is on the left.
        """
        rows, bounds = [], []
        for gradient, value, point in cuts:
            components = space.to_components(gradient)
            rows.append(np.append(components, -1.0))
            bounds.append(float(components @ space.to_components(point)) - value)
        return np.asarray(rows, dtype=float), np.asarray(bounds, dtype=float)

    def _lower_bound(self, space: Any, cuts: list) -> float:
        """The cutting-plane model's own minimum: a global lower bound.

        A linear programme, and solved as one. The model underestimates the
        function everywhere, so this underestimates the minimum -- which is
        what makes the gap a bound rather than an indication.

        Returns ``-inf`` when the LP is unbounded or infeasible, which is the
        honest answer early on: one cut does not bound a function below.
        """
        from scipy.optimize import linprog

        rows, bounds = self._cut_rows(space, cuts)
        size = space.dim
        objective = np.zeros(size + 1)
        objective[size] = 1.0

        outcome = linprog(
            objective,
            A_ub=rows,
            b_ub=bounds,
            bounds=[(None, None)] * (size + 1),
            method="highs",
        )
        return float(outcome.x[size]) if outcome.status == 0 else -np.inf

    def _master(
        self, space: Any, cuts: list, centre: np.ndarray, level: float | None
    ) -> np.ndarray | None:
        """The level QP, or the proximal fallback when *level* is ``None``.

        Returns the new point's components, or ``None`` if the programme was
        infeasible -- which for the level problem is expected and handled, and
        for the fallback means something has gone properly wrong.
        """
        rows, bounds = self._cut_rows(space, cuts)
        size = space.dim

        # 0.5 ||x - centre||^2 == 0.5 x'x - centre'x + const, and t is free of
        # the quadratic; the fallback adds t to the linear part instead.
        quadratic = np.zeros((size + 1, size + 1))
        quadratic[:size, :size] = np.eye(size)
        linear = np.append(-centre, 0.0 if level is not None else 1.0)

        if level is not None:
            level_row = np.zeros(size + 1)
            level_row[size] = 1.0
            rows = np.vstack([rows, level_row])
            bounds = np.append(bounds, level)

        result = self.qp_solver.solve(
            quadratic,
            linear,
            rows,
            np.full(bounds.size, -np.inf),
            bounds,
        )
        if not result.solved:
            return None
        return np.asarray(result.x[:size], dtype=float)

    def minimise(
        self,
        functional: Functional,
        start: Any,
        /,
        *,
        subgradient: Callable[[Any], Any] | None = None,
    ) -> BundleResult:
        """Minimise a convex functional from a starting point.

        Args:
            functional: convex, and able to supply a subgradient.
            start: where to begin.
            subgradient: an override, called with a point and returning a
                subgradient. Defaults to the functional's own.

        Returns:
            The minimum, a minimiser, and a gap that is a genuine bound on the
            distance to the true minimum.

        Raises:
            TypeError: if the domain has no component map.
        """
        from ..algebra.operators import require_coordinates

        space = functional.domain
        require_coordinates(space, space)
        slope = subgradient or functional.subgradient

        centre = space.copy(start)
        best_point = space.copy(start)
        upper = float(functional(centre))
        evaluations = 1
        cuts: list[tuple[Any, float, Any]] = [
            (space.copy(slope(centre)), upper, space.copy(centre))
        ]
        lower = -np.inf
        message = "iteration limit reached"

        for iteration in range(1, self._iterations + 1):
            lower = max(lower, self._lower_bound(space, cuts))
            gap = upper - lower
            if gap <= self._tolerance * max(abs(upper), 1.0):
                return BundleResult(
                    upper, best_point, iteration, evaluations, True,
                    "gap tolerance reached", gap,
                )

            centre_components = space.to_components(centre)
            candidate = None
            alpha = self._alpha
            # A level too close to the bound can be unreachable. Widen towards
            # caution and retry before giving up on the level step entirely.
            for _ in range(3):
                if not np.isfinite(lower):
                    break
                candidate = self._master(
                    space, cuts, centre_components, alpha * lower + (1.0 - alpha) * upper
                )
                if candidate is not None:
                    break
                alpha = min(alpha * 1.5, 0.9)
            if candidate is None:
                # No reachable level, or no bound yet to set one from. A
                # proximal step still improves the centre and the bundle, which
                # is what the next lower bound is built from.
                candidate = self._master(space, cuts, centre_components, None)
            if candidate is None:
                message = "the master problem could not be solved"
                break

            point = space.from_components(candidate)
            value = float(functional(point))
            evaluations += 1
            cuts.append((space.copy(slope(point)), value, space.copy(point)))
            if len(cuts) > self._capacity:
                cuts.pop(0)

            if value < upper:
                upper, best_point = value, space.copy(point)
            centre = point

        return BundleResult(
            upper, best_point, self._iterations, evaluations, False, message,
            upper - lower,
        )


@dataclass(frozen=True)
class SaddlePointResult:
    """The outcome of a primal-dual feasibility solve."""

    value: float
    """``(c, m)`` at the point found: the support value being sought."""

    model: Any
    """The maximising model."""

    discrepancy: Any
    """The data-error vector ``v`` that goes with it."""

    certificate: Any
    """The dual variable, which is the same object the dual route calls a
    certificate: a weighting of the data."""

    residual: float
    """``||G m + v - d||``, the feasibility gap. This is what convergence is
    declared on -- not the objective, which is monotone in neither variable."""

    iterations: int
    converged: bool

    def __repr__(self) -> str:
        return (
            f"SaddlePointResult(value={self.value:.6g}, "
            f"residual={self.residual:.3g}, converged={self.converged})"
        )


class ChambollePockSolver:
    """Maximise a linear functional over a feasible set, by primal-dual splitting.

    The primal form of the same question the dual route answers:

    .. code-block:: text

        maximise    (c, m)
        subject to  m in prior, v in noise, G m + v == d

    Chambolle and Pock's first-order method (2011) on the saddle-point form,
    with ``K == [G; I]``. Each step is one application of ``G``, one of its
    adjoint, and a *projection* onto each of the two sets -- so the cost per
    iteration is small and the number of them is large, which is the opposite
    trade to the bundle methods and the reason to have both.

    Converges at ``O(1/N)`` in the primal-dual gap when
    ``tau sigma ||K||^2 <= 1``; the step sizes are chosen to satisfy that from
    a power-iteration estimate of ``||G||`` unless given.

    **It projects rather than supports.** v1's version could only handle a
    ball, because it implemented the projection itself and raised for anything
    else. Here every :class:`~pygeoinf2.geometry.convex.ConvexSet` knows its
    own nearest point, so a polytope or an intersection works without a case
    for it -- and an intersection projects by Dykstra, which is iterative, so
    the cost per step is then no longer small. That is a real trade and worth
    knowing about before choosing this route for such a set.

    It is at its best when the objective ``c`` changes and the feasible set
    does not: the step sizes and the operator norm are then computed once, and
    only the linear term moves.

    A port of v1's ``ChambollePockSolver``.

    References:
        Chambolle, A. and Pock, T. (2011). A first-order primal-dual algorithm
        for convex problems with applications to imaging. *Journal of
        Mathematical Imaging and Vision* 40(1), 120-145.
    """

    def __init__(
        self,
        prior: Any,
        noise: Any,
        forward: LinearOperator,
        data: Any,
        /,
        *,
        sigma: float | None = None,
        tau: float | None = None,
        theta: float = 1.0,
        iterations: int = 1000,
        tolerance: float = 1e-6,
        rng: Generator | None = None,
    ) -> None:
        """
        Args:
            prior: the convex set the model lies in.
            noise: the convex set the data error lies in.
            forward: ``G``, from the model space to the data space.
            data: the observations.
            sigma: the dual step. Chosen with *tau* from ``||G||`` if unset.
            tau: the primal step.
            theta: the over-relaxation, one being the standard choice.
            iterations: the cap.
            tolerance: on the feasibility residual.
            rng: for the power iteration that estimates ``||G||``.

        Raises:
            ValueError: if the sets do not live in the operator's spaces, or
                the steps are not positive.
        """
        if prior.domain != forward.domain:
            raise ValueError("The prior set must live in the model space.")
        if noise.domain != forward.codomain:
            raise ValueError("The noise set must live in the data space.")
        if theta < 0.0:
            raise ValueError(f"theta must be non-negative, got {theta}.")

        self._prior = prior
        self._noise = noise
        self._forward = forward
        self._data = data
        self._theta = theta
        self._iterations = iterations
        self._tolerance = tolerance

        if sigma is None or tau is None:
            chosen_sigma, chosen_tau = self._steps(rng)
            sigma = chosen_sigma if sigma is None else sigma
            tau = chosen_tau if tau is None else tau
        if sigma <= 0.0 or tau <= 0.0:
            raise ValueError("The step sizes must be positive.")
        self._sigma, self._tau = sigma, tau

    def _steps(self, rng: Generator | None) -> tuple[float, float]:
        """Steps satisfying ``tau sigma ||K||^2 <= 0.99``.

        ``||K||^2 <= ||G||^2 + 1`` since ``K == [G; I]``, and ``||G||`` comes
        from twenty power iterations -- enough for a step size, which needs an
        estimate rather than a number.
        """
        model_space = self._forward.domain
        vector = model_space.random(rng=rng)
        norm = model_space.norm(vector)
        if norm == 0.0:  # pragma: no cover - a zero draw
            return 1.0, 1.0
        vector = model_space.scale(1.0 / norm, vector)

        estimate = 0.0
        for _ in range(20):
            image = self._forward.adjoint(self._forward(vector))
            estimate = model_space.norm(image)
            if estimate == 0.0:
                break
            vector = model_space.scale(1.0 / estimate, image)
        operator_norm = float(np.sqrt(max(estimate, 0.0)) ** 2 + 1.0)
        step = float(np.sqrt(0.99 / max(operator_norm, 1e-30)))
        return step, step

    def solve(self, objective: Any, /, *, start: Any = None) -> SaddlePointResult:
        """Maximise ``(objective, m)`` over the feasible set.

        Args:
            objective: ``c``, in the model space. Typically ``T* q``.
            start: an initial model. Defaults to zero, and is the warm start
                when sweeping directions.

        Returns:
            The result, whose ``converged`` says whether the feasibility
            residual reached the tolerance.
        """
        model_space = self._forward.domain
        data_space = self._forward.codomain

        model = model_space.zero() if start is None else model_space.copy(start)
        discrepancy = data_space.zero()
        certificate = data_space.zero()
        model_bar, discrepancy_bar = model, discrepancy
        residual = float("inf")

        for iteration in range(1, self._iterations + 1):
            # Dual ascent on the extrapolated primal point.
            gap = data_space.subtract(
                data_space.add(self._forward(model_bar), discrepancy_bar), self._data
            )
            certificate = data_space.axpy(self._sigma, gap, certificate)

            # Primal descent, then back onto each set.
            pulled = self._forward.adjoint(certificate)
            step = model_space.axpy(-self._tau, pulled, model_space.copy(model))
            step = model_space.axpy(self._tau, objective, step)
            next_model = self._prior.project(step)

            next_discrepancy = self._noise.project(
                data_space.axpy(-self._tau, certificate, data_space.copy(discrepancy))
            )

            # Over-relaxation: the extrapolation that buys the O(1/N) rate.
            # Onto *copies*: axpy writes into its third argument, so relaxing
            # into next_model would leave the iterate itself extrapolated --
            # and the method then converges to something that is not a point
            # of the feasible set at all.
            model_bar = model_space.axpy(
                self._theta,
                model_space.subtract(next_model, model),
                model_space.copy(next_model),
            )
            discrepancy_bar = data_space.axpy(
                self._theta,
                data_space.subtract(next_discrepancy, discrepancy),
                data_space.copy(next_discrepancy),
            )
            model, discrepancy = next_model, next_discrepancy

            residual = data_space.norm(
                data_space.subtract(
                    data_space.add(self._forward(model), discrepancy), self._data
                )
            )
            if residual < self._tolerance:
                return SaddlePointResult(
                    float(model_space.inner_product(objective, model)),
                    model,
                    discrepancy,
                    certificate,
                    residual,
                    iteration,
                    True,
                )

        return SaddlePointResult(
            float(model_space.inner_product(objective, model)),
            model,
            discrepancy,
            certificate,
            residual,
            self._iterations,
            False,
        )


@dataclass(frozen=True)
class KKTResult:
    """The outcome of a KKT solve for a quadratically constrained maximum."""

    value: float
    """``(c, m)`` at the maximiser."""

    model: Any
    """The maximiser."""

    multipliers: tuple[float, float]
    """``(lambda, mu)``, on the prior and the data constraint. A zero second
    multiplier means the data constraint is slack: the answer is the prior's
    own support point and the data never bit."""

    iterations: int
    converged: bool

    def __repr__(self) -> str:
        return (
            f"KKTResult(value={self.value:.6g}, "
            f"multipliers={self.multipliers}, converged={self.converged})"
        )


class PrimalKKTSolver:
    """The exact maximum over two *quadratic* sets, from the KKT conditions.

    Where :class:`ChambollePockSolver` iterates towards the answer for any
    convex sets, this writes it down -- when both sets are balls or ellipsoids.
    Then the constraints are quadratic, the stationarity condition is linear in
    the model given the two multipliers, and the whole problem collapses to a
    two-variable root find:

    .. code-block:: text

        maximise    (c, m)
        subject to  (m - m0, B (m - m0)) <= eta^2
                    (G m - d, V (G m - d)) <= r^2

    Stationarity gives ``m*(lambda, mu)`` in closed form, and the two active
    constraints give two equations in ``(lambda, mu)``, solved in log
    coordinates so both stay positive.

    **The model space is never discretised.** The closed form needs
    ``(1/mu) V^-1 + (1/lambda) G B^-1 G*``, which acts on the *data* space --
    finite-dimensional by construction -- so the Woodbury identity moves the
    only matrix that is formed onto the small side. That is the whole reason
    this route exists alongside the others: on a fine model grid it is the one
    that does not care how fine.

    **Two branches.** If the prior's own support point already satisfies the
    data constraint, the data never bit and that point is the answer, with
    ``mu == 0``. Only when both constraints are active is the root find run.

    **Where it is weak.** When the noise set is small enough that the data
    constraint is effectively an equality, the second multiplier runs away --
    it is the reciprocal of the constraint's slack -- and the root find ends
    at the clip that keeps the exponential finite. The answer is then good to
    about ``1e-3`` rather than to machine precision. Measured against
    :class:`ChambollePockSolver` over sixteen directions with a noise radius of
    0.05: fourteen agree to 1e-11 and two, where ``mu`` hit the clip, to
    2.4e-3. v1 behaves identically, down to the same multiplier, so this is
    the method's limit rather than a defect in the port -- and the primal
    route has no such trouble, which is the reason to keep both.

    A port of v1's ``PrimalKKTSolver``, and the solver behind its
    ``sphere_dli_example``.
    """

    def __init__(
        self,
        prior: Any,
        noise: Any,
        forward: LinearOperator,
        data: Any,
        /,
        *,
        tolerance: float = 1e-10,
        evaluations: int = 200,
    ) -> None:
        """
        Args:
            prior: a :class:`~pygeoinf2.geometry.convex.Ball` or
                :class:`~pygeoinf2.geometry.convex.Ellipsoid` on the model
                space. An ellipsoid must know its covariance.
            noise: likewise on the data space, and centred at the origin --
                the data itself is the offset.
            forward: ``G``.
            data: the observations.
            tolerance: for the root find on the multipliers.
            evaluations: its cap.

        Raises:
            TypeError: if either set is neither a ball nor an ellipsoid.
            ValueError: if an ellipsoid cannot supply its covariance, or the
                sets do not live in the operator's spaces.
        """
        from ..geometry.convex import Ball, Ellipsoid

        for name, given, space in (
            ("prior", prior, forward.domain),
            ("noise", noise, forward.codomain),
        ):
            if not isinstance(given, (Ball, Ellipsoid)):
                raise TypeError(
                    f"The {name} must be a Ball or an Ellipsoid for the KKT "
                    f"route, which is what makes its constraint quadratic; "
                    f"got {type(given).__name__}. Use ChambollePockSolver for "
                    "a general convex set."
                )
            if given.domain != space:
                raise ValueError(f"The {name} set must live in {space!r}.")

        self._prior = prior
        self._noise = noise
        self._forward = forward
        self._data = data
        self._tolerance = tolerance
        self._evaluations = evaluations
        # The multipliers of the last solve, which start the next. Directions
        # near each other have multipliers near each other, so this is the
        # same warm start the dual route gets from its certificate.
        self._previous: tuple[float, float] | None = None

        self._prior_parts = self._quadratic(forward.domain, prior)
        self._noise_parts = self._quadratic(forward.codomain, noise)
        self._prior_radius = self._prior_parts[2]
        self._noise_radius = self._noise_parts[2]
        self._noise_weight = self._noise_parts[0]

        # Both act on the *data* space, which is finite-dimensional by
        # construction -- so these are the only matrices formed, whatever the
        # model space is.
        self._noise_covariance_matrix = self._noise_parts[1].matrix(form="components")
        self._gram = (
            forward @ self._prior_parts[1] @ forward.adjoint
        ).matrix(form="components")

    @staticmethod
    def _quadratic(space: Any, given: Any) -> tuple[Any, Any, float, Any]:
        """A set as ``(weight, inverse weight, radius, centre)``.

        A ball's weight is the identity and its radius is its own; an
        ellipsoid's are its precision and covariance, with radius one, since
        its constraint is already normalised.
        """
        from ..geometry.convex import Ball

        identity = LinearOperator.identity(space)
        if isinstance(given, Ball):
            return identity, identity, given.radius, given.centre
        if given._covariance is None:
            raise ValueError(
                "An ellipsoid needs its covariance for the KKT route: the "
                "closed form inverts the constraint's weight."
            )
        return given._precision, given._covariance, 1.0, given.centre

    def _model(self, multipliers: tuple[float, float], objective: Any) -> Any:
        """``m*(lambda, mu)`` from stationarity, through Woodbury.

        The only linear system solved acts on the data space.
        """
        lam, mu = multipliers
        model_space = self._forward.domain
        data_space = self._forward.codomain
        prior_weight, prior_inverse, _, centre = self._prior_parts
        _, noise_inverse, _, _ = self._noise_parts

        # r = c + lambda B m0 + mu G* V d.
        right = model_space.add(objective, model_space.scale(lam, prior_weight(centre)))
        right = model_space.add(
            right,
            model_space.scale(
                mu, self._forward.adjoint(self._noise_weight(self._data))
            ),
        )

        # w == (1/lambda) B^-1 r. The 1/lambda belongs here, before G is
        # applied: Woodbury's correction carries 1/lambda *twice*, once on
        # each side of the inverse, and applying G to the unscaled vector
        # loses one of them -- which leaves a model satisfying neither
        # constraint and a root find that cannot move.
        scaled = model_space.scale(1.0 / lam, prior_inverse(right))
        if mu == 0.0:
            return scaled

        # K == (1/mu) V^-1 + (1/lambda) G B^-1 G*, on the data space.
        pulled = data_space.from_components(
            np.linalg.solve(
                self._kernel(lam, mu),
                data_space.to_components(self._forward(scaled)),
            )
        )
        correction = model_space.scale(
            1.0 / lam, prior_inverse(self._forward.adjoint(pulled))
        )
        return model_space.subtract(scaled, correction)

    def _kernel(self, lam: float, mu: float) -> np.ndarray:
        """``(1/mu) V^-1 + (1/lambda) G B^-1 G*`` as a matrix on the data space."""
        return self._noise_covariance_matrix / mu + self._gram / lam

    def solve(self, objective: Any, /) -> KKTResult:
        """Maximise ``(objective, m)`` over the two sets.

        Args:
            objective: ``c``, in the model space.

        Returns:
            The maximiser and its multipliers.
        """
        from scipy.optimize import fsolve

        model_space = self._forward.domain
        data_space = self._forward.codomain

        # The prior's own support point, which is the answer when the data
        # constraint does not bite.
        best = self._prior.support_maximiser(objective)
        residual = data_space.subtract(self._forward(best), self._data)
        if (
            float(data_space.inner_product(self._noise_weight(residual), residual))
            <= self._noise_radius**2 * (1.0 + 1e-9)
        ):
            return KKTResult(
                float(model_space.inner_product(objective, best)),
                best,
                (float("inf"), 0.0),
                1,
                True,
            )

        def residuals(logged: np.ndarray) -> np.ndarray:
            lam, mu = np.exp(np.clip(logged, -30.0, 25.0))
            model = self._model((float(lam), float(mu)), objective)
            offset = model_space.subtract(model, self._prior_parts[3])
            first = (
                float(model_space.inner_product(self._prior_parts[0](offset), offset))
                - self._prior_radius**2
            )
            gap = data_space.subtract(self._forward(model), self._data)
            second = (
                float(data_space.inner_product(self._noise_weight(gap), gap))
                - self._noise_radius**2
            )
            return np.array([first, second])

        # A physically scaled start, as v1 has. At the solution the prior
        # multiplier balances the objective against the constraint, so
        # ``|c| / eta`` is the right order -- and starting at one instead
        # leaves fsolve searching in the wrong decade, which on these problems
        # it simply fails to leave.
        weighted = float(
            np.sqrt(
                max(
                    model_space.inner_product(
                        self._prior_parts[1](objective), objective
                    ),
                    0.0,
                )
            )
        )
        physical = max(weighted / self._prior_radius, 1e-4)

        def attempt(guess: tuple[float, float]) -> tuple[np.ndarray, dict, int]:
            logged = np.log(
                np.array([max(guess[0], 1e-4), max(guess[1], 1e-8)], dtype=float)
            )
            found, info, status, _ = fsolve(
                residuals,
                logged,
                full_output=True,
                xtol=self._tolerance,
                maxfev=self._evaluations,
            )
            return found, info, status

        found, info, status = attempt(self._previous or (physical, 1e-3))
        if status != 1:
            # A two-variable root find on a pair of quadratics is not reliably
            # solved from one start, and a failed solve here is not a hard
            # failure -- it is a start in the wrong decade. v1 carries a ladder
            # of fallbacks for exactly this, and without it the method returns
            # its own starting point and disagrees with the primal route by
            # tens of per cent.
            for guess in (
                (physical, 1e-3),
                (physical, 1e-2),
                (physical, 0.5),
                (physical, 2.0),
                (0.5 * physical, 1e-2),
                (0.5 * physical, 1.0),
            ):
                found, info, status = attempt(guess)
                if status == 1:
                    break

        multipliers = tuple(float(value) for value in np.exp(np.clip(found, -30.0, 25.0)))
        # Only a *converged* solve is worth carrying into the next direction.
        # Carrying a failed one starts the next problem from a point that is
        # not a root of anything.
        self._previous = multipliers if status == 1 else None
        model = self._model(multipliers, objective)
        return KKTResult(
            float(model_space.inner_product(objective, model)),
            model,
            multipliers,
            int(info["nfev"]),
            status == 1,
        )
