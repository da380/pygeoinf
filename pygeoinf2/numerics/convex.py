"""
Convex optimization, coordinate-free where the geometry allows.

Which is further than one might expect. The proximal operators that matter in
practice — of a norm, of the indicator of a ball, of a squared distance — all
have closed forms written in terms of a *norm* and a *direction*, so they are
statements about the space's geometry rather than about a basis:

    prox of  t ||.||        x -> max(0, 1 - t/||x||) x
    projection onto a ball  x -> min(1, r/||x||) x

Both are metric-aware for free, and both mean the same thing under refinement.
Written in components they would instead shrink in whatever basis the
discretization happened to supply.

What is *not* coordinate-free is the small dense subproblem a bundle method
solves over its cut coefficients. That lives in ``R^k`` for a handful of cuts
and is canonically Euclidean, so a quadratic-programming backend behind a
protocol (:mod:`pygeoinf2.numerics.quadratic_programming`) is the right shape
there -- coordinates are not a constraint when the space is genuinely
finite-dimensional and has no metric of its own.

The module holds the standard convex functionals and support functions; the
non-smooth methods :class:`SubgradientDescent`, :class:`ProximalGradient` and
:class:`ProximalPoint`; the proximal bundle method; and the primal-dual
splitting :class:`ChambollePockSolver` for the maximum of a linear functional
over a prior set intersected with a data constraint, which the feasible
property set of :mod:`pygeoinf2.geometry.feasible` uses.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import warnings

import numpy as np
from numpy.random import Generator

from ..algebra.operators import Functional, LinearFunctional, LinearOperator
from ..algebra.spaces import HilbertSpace
from ..traits import Traits
from .optimization import OptimizationResult, Optimizer

__all__ = [
    "ChambollePockSolver",
    "SaddlePointResult",
    "ProximalBundleMethod",
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
    """``0.5 ||x - center||^2`` in the space's own norm.

    Smooth, with gradient ``x - center`` and Hessian the identity — so its
    condition number is one whatever the metric, which is the cleanest possible
    illustration of why the gradient is taken in the space rather than in the
    components.
    """

    def __init__(self, domain: HilbertSpace, /, *, center: Any = None) -> None:
        """
        Args:
            domain: the space.
            center: the point the distance is measured from. Defaults to zero.
        """
        super().__init__(domain)
        self._center = domain.zero() if center is None else center

    @property
    def center(self) -> Any:
        """The point the distance is measured from."""
        return self._center

    def _value(self, x: Any) -> float:
        return 0.5 * self.domain.squared_norm(self.domain.subtract(x, self._center))

    def _derivative(self, x: Any) -> LinearFunctional:
        return LinearFunctional.from_representer(
            self.domain, self.domain.subtract(x, self._center)
        )

    @property
    def has_hessian(self) -> bool:
        """True: the Hessian is the identity."""
        return True

    def _hessian(self, x: Any) -> LinearOperator:
        return LinearOperator.self_adjoint(
            self.domain, lambda v: v, traits=Traits.POSITIVE_DEFINITE
        )

    def prox(self, x: Any, step: float, /) -> Any:
        """``(x + step * center) / (1 + step)``, in closed form."""
        space = self.domain
        shifted = space.axpy(step, self._center, space.copy(x))
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
        self, domain: HilbertSpace, /, *, radius: float = 1.0, center: Any = None
    ) -> None:
        """
        Args:
            domain: the space.
            radius: the ball's radius, which must not be negative. Zero gives
                the indicator of a single point, whose prox is the constant map
                to the center.
            center: the ball's center. Defaults to zero.

        Raises:
            ValueError: if the radius is negative.
        """
        if radius < 0.0:
            raise ValueError("radius must not be negative.")
        super().__init__(domain)
        self._radius = float(radius)
        self._center = domain.zero() if center is None else center

    @property
    def radius(self) -> float:
        """The ball's radius."""
        return self._radius

    @property
    def center(self) -> Any:
        """The ball's center."""
        return self._center

    def _value(self, x: Any) -> float:
        offset = self.domain.norm(self.domain.subtract(x, self._center))
        return 0.0 if offset <= self._radius * (1.0 + 1e-12) else float("inf")

    def prox(self, x: Any, step: float, /) -> Any:
        """The projection onto the ball, which does not depend on the step."""
        space = self.domain
        offset = space.subtract(x, self._center)
        norm = space.norm(offset)
        if norm <= self._radius:
            return space.copy(x)
        return space.axpy(self._radius / norm, offset, space.copy(self._center))

    def conjugate(self) -> Functional:
        """The support function of the ball."""
        return SupportFunction.of_ball(
            self.domain, radius=self._radius, center=self._center
        )


class SupportFunction(Functional):
    """``h(y) == sup { (y, x) : x in K }`` for a convex set ``K``.

    A functional on the same space as the set, which is a dividend of Riesz
    identification: a support function is classically a function on the dual,
    and without the identification every duality argument would carry a
    transport map with it.

    The constructions are a ball, a point, a half-space, a hyperplane and an
    oracle; a sum of support functions is the support function of the
    Minkowski sum, a positive multiple that of the scaled set, and
    :meth:`compose_with` gives that of a linear image. The algebra is closed,
    which is the point of having the class at all.
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
        """True: a maximizer is a subgradient."""
        return True

    def subgradient(self, y: Any) -> Any:
        """A maximizer of the pairing, which is a subgradient of the support."""
        return self._maximizer(y)

    @abstractmethod
    def _maximizer(self, y: Any) -> Any:
        """A point of the set attaining the supremum."""

    @staticmethod
    def of_ball(
        domain: HilbertSpace, /, *, radius: float = 1.0, center: Any = None
    ) -> SupportFunction:
        """The support function of a ball: ``r ||y|| + (center, y)``.

        Args:
            domain: the space the ball lives in.
            radius: its radius, in the space's own norm.
            center: its center. The origin if omitted, which makes the
                support function the norm alone.

        Returns:
            The support function.
        """
        return _BallSupport(domain, radius=radius, center=center)

    @staticmethod
    def of_point(domain: HilbertSpace, point: Any, /) -> SupportFunction:
        """The support function of a single point: the linear functional ``(p, .)``."""
        return _PointSupport(domain, point)

    @staticmethod
    def of_oracle(
        domain: HilbertSpace,
        value: Callable[[Any], float],
        /,
        *,
        maximizer: Callable[[Any], Any] | None = None,
    ) -> SupportFunction:
        """A support function given by a callable, with its maximizer if known.

        The value alone gives a functional that can be evaluated and added and
        scaled like any other; the
        maximizer, ``q -> argmax (q, x)`` over the set, is what makes it a
        *subgradient*-carrying functional, which is what a bundle method
        needs of it. Without one, :attr:`has_subgradient` is false and
        :meth:`subgradient` refuses, rather than returning nothing.

        Args:
            domain: the space.
            value: ``q -> h(q)``.
            maximizer: ``q -> x*(q)`` attaining it, if available.

        Returns:
            The support function.
        """
        return _OracleSupport(domain, value, maximizer=maximizer)

    @staticmethod
    def of_half_space(
        domain: HilbertSpace,
        normal: Any,
        /,
        *,
        offset: float = 0.0,
        rtol: float = 1e-12,
    ) -> SupportFunction:
        """The support function of ``{ x : (normal, x) <= offset }``.

        Extended-real valued, as a support function of an unbounded set must
        be: ``alpha * offset`` when the direction is ``alpha * normal`` with
        ``alpha >= 0``, and ``+inf`` in every other direction, since the set
        then contains a ray along which the pairing grows without bound.

        Args:
            domain: the space.
            normal: the outward normal, which must be nonzero.
            offset: the level.
            rtol: a direction counts as parallel to the normal when what is
                left of it after removing its component along the normal is
                at most ``rtol`` times its own norm.

        Returns:
            The support function.
        """
        return _HalfSpaceSupport(domain, normal, offset=offset, rtol=rtol)

    @staticmethod
    def of_hyperplane(
        domain: HilbertSpace,
        normal: Any,
        /,
        *,
        offset: float = 0.0,
        rtol: float = 1e-12,
    ) -> SupportFunction:
        """The support function of ``{ x : (normal, x) == offset }``.

        As :meth:`of_half_space`, but finite for a direction parallel to the
        normal of *either* sign: the plane is bounded along its normal and
        unbounded along everything else.

        Args:
            domain: the space.
            normal: the normal, which must be nonzero.
            offset: the level.
            rtol: as for :meth:`of_half_space`.

        Returns:
            The support function.
        """
        return _HalfSpaceSupport(
            domain, normal, offset=offset, rtol=rtol, two_sided=True
        )

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
        self, domain: HilbertSpace, /, *, radius: float = 1.0, center: Any = None
    ) -> None:
        if radius < 0.0:
            raise ValueError("radius must not be negative.")
        super().__init__(domain)
        self._radius = float(radius)
        self._center = domain.zero() if center is None else center

    def _value(self, y: Any) -> float:
        # At radius zero this is the point support ``(center, y)``, which is
        # what it should be: the support function of a single point.
        return self._radius * self.domain.norm(y) + self.domain.inner_product(
            self._center, y
        )

    def _maximizer(self, y: Any) -> Any:
        space = self.domain
        norm = space.norm(y)
        if norm == 0.0:
            return space.copy(self._center)
        return space.axpy(self._radius / norm, y, space.copy(self._center))


class _OracleSupport(SupportFunction):
    """A support function that is whatever the oracle says."""

    def __init__(
        self,
        domain: HilbertSpace,
        value: Callable[[Any], float],
        /,
        *,
        maximizer: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__(domain)
        self._oracle = value
        self._maximizer_fn = maximizer

    @property
    def has_subgradient(self) -> bool:
        """Only with a maximizer: a support value alone gives no subgradient."""
        return self._maximizer_fn is not None

    def _value(self, y: Any) -> float:
        return float(self._oracle(y))

    def _maximizer(self, y: Any) -> Any:
        if self._maximizer_fn is None:
            raise NotImplementedError(
                "This support function was given its values only; a subgradient "
                "is a maximizer, and none was supplied. Pass maximizer= to "
                "SupportFunction.of_oracle or ConvexSet.from_support_function."
            )
        return self._maximizer_fn(y)


class _HalfSpaceSupport(SupportFunction):
    """The support function of a half-space, or of its bounding hyperplane.

    For ``{ x : (a, x) <= b }`` the pairing ``(y, x)`` is bounded above on
    the set only when ``y == alpha a`` with ``alpha >= 0``, where its supremum
    is ``alpha b``, attained on the whole boundary plane. For the hyperplane
    the sign of ``alpha`` is free. Everywhere else the value is ``+inf``.

    Whether a direction is parallel to the normal is decided on the residual
    ``y - alpha a`` with ``alpha == (y, a) / (a, a)``, formed as a vector: the
    same quantity from ``||y||^2 - alpha^2 ||a||^2`` cancels to noise exactly
    when the two are nearly parallel, which is the case being decided.

    The maximizer, when the value is finite, is not unique -- every point of
    the boundary plane attains it. The one returned is the plane's point of
    least norm, ``(b / (a, a)) a``; any other would serve a subgradient
    method equally, and this one is the canonical choice.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        normal: Any,
        /,
        *,
        offset: float = 0.0,
        rtol: float = 1e-12,
        two_sided: bool = False,
    ) -> None:
        super().__init__(domain)
        squared = domain.squared_norm(normal)
        if squared == 0.0:
            raise ValueError("The normal vector must be nonzero.")
        if rtol < 0.0:
            raise ValueError("rtol must not be negative.")
        self._normal = normal
        self._offset = float(offset)
        self._squared_norm = squared
        self._rtol = float(rtol)
        self._two_sided = two_sided

    def _multiple(self, y: Any) -> float | None:
        """``alpha`` with ``y == alpha * normal``, or ``None`` if there is none."""
        space = self.domain
        alpha = space.inner_product(y, self._normal) / self._squared_norm
        residual = space.axpy(-alpha, self._normal, space.copy(y))
        if space.norm(residual) > self._rtol * space.norm(y):
            return None
        if alpha < 0.0 and not self._two_sided:
            return None
        return float(alpha)

    def _value(self, y: Any) -> float:
        alpha = self._multiple(y)
        return float("inf") if alpha is None else alpha * self._offset

    def _maximizer(self, y: Any) -> Any:
        if self._multiple(y) is None:
            raise ValueError(
                "The support is infinite in this direction: the set is "
                "unbounded along it, and no point of the set attains the "
                "supremum. The support function is finite only for a "
                "direction parallel to the normal."
            )
        return self.domain.scale(self._offset / self._squared_norm, self._normal)


class _PointSupport(SupportFunction):
    """The support function of a single point."""

    def __init__(self, domain: HilbertSpace, point: Any, /) -> None:
        super().__init__(domain)
        self._point = point

    def _value(self, y: Any) -> float:
        return self.domain.inner_product(self._point, y)

    def _maximizer(self, y: Any) -> Any:
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

    def _maximizer(self, y: Any) -> Any:
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

    def _maximizer(self, y: Any) -> Any:
        return self.domain.scale(self._alpha, self._base.subgradient(y))


class _ImageSupport(SupportFunction):
    """The support function of a linear image, which is ``h(A* y)``."""

    def __init__(self, base: SupportFunction, operator: LinearOperator, /) -> None:
        super().__init__(operator.codomain)
        self._base = base
        self._operator = operator

    def _value(self, y: Any) -> float:
        return self._base(self._operator.adjoint(y))

    def _maximizer(self, y: Any) -> Any:
        return self._operator(self._base.subgradient(self._operator.adjoint(y)))


# --------------------------------------------------------------------- #
#                          Non-smooth methods                           #
# --------------------------------------------------------------------- #


class SubgradientDescent(Optimizer):
    """Subgradient descent with a diminishing step.

    The step rules:

    - ``"sqrt"``: ``a / sqrt(k)``, the classical choice for a Lipschitz
      convex function, giving ``O(1/sqrt(k))`` in the best value.
    - ``"inverse"``: ``a / k``, square-summable but not summable, and better
      when the function is strongly convex.
    - ``"polyak"``: ``(f(x) - f*) / ||g||^2`` when a target value ``f*`` is
      known, which is much faster and is available whenever the optimum is
      known to be zero -- as it is for a feasibility problem.
    - ``"constant"``: ``a`` at every step. Not convergent in general; it
      reaches a neighborhood of the minimum whose size grows with ``a``.

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
            rule: which step rule to use, as listed above.
            target_value: the known optimal value, required by ``"polyak"``.
            **kwargs: passed to :class:`~pygeoinf2.numerics.optimization.Optimizer`.
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

    def minimize(self, functional: Functional, x0: Any, /) -> OptimizationResult:
        """Minimize, requiring only a subgradient.

        Args:
            functional: convex, and able to supply a subgradient. It need not
                be differentiable, which is the point of this method.
            x0: where to start.

        Returns:
            The optimization result.

        Raises:
            ValueError: if the functional cannot supply a subgradient.
        """
        if not functional.has_subgradient:
            raise ValueError(
                "SubgradientDescent needs a functional with a subgradient. "
                "A smooth functional supplies one from its gradient."
            )
        return self._minimize(functional, x0)

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

    def _minimize(self, functional: Functional, x0: Any) -> OptimizationResult:
        space = functional.domain
        x = space.copy(x0)
        value = functional(x)
        best_point, best_value = space.copy(x), value
        history = [best_value]
        evaluations = 1

        for iteration in range(1, self._max_iterations + 1):
            subgradient = functional.subgradient(x)
            norm = space.norm(subgradient)
            if norm <= self._gtol:
                return OptimizationResult(
                    best_point,
                    best_value,
                    norm,
                    iteration - 1,
                    evaluations,
                    True,
                    "subgradient tolerance reached",
                    history,
                )

            # The value at x was evaluated at the end of the last iteration
            # and is reused here; only the Polyak rule reads it.
            step = self._step(iteration, value, norm)
            x = space.axpy(-step, subgradient, x)

            value = functional(x)
            evaluations += 1
            history.append(value)
            if value < best_value:
                best_value, best_point = value, space.copy(x)

        return OptimizationResult(
            best_point,
            best_value,
            norm,
            self._max_iterations,
            evaluations,
            False,
            "iteration limit reached",
            history,
        )


class ProximalGradient(Optimizer):
    """Proximal gradient descent for ``f + g``, optionally accelerated.

    ``f`` must be smooth and ``g`` must have a proximal operator; either may be
    absent. With acceleration this is FISTA, converging at ``O(1/k^2)`` against
    the plain method's ``O(1/k)``.

    The step is the reciprocal of a Lipschitz constant for ``grad f``, found by
    backtracking when none is given. Because the proximal operator is taken in
    the space's norm, the whole method is metric-aware: the same problem
    discretized twice takes the same number of iterations.
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
            **kwargs: passed to :class:`~pygeoinf2.numerics.optimization.Optimizer`.
        """
        self._step = step
        self._accelerated = accelerated
        self._backtracking = backtracking
        self._max_backtracks = max_backtracks
        super().__init__(**kwargs)

    def minimize(
        self,
        smooth: Functional,
        x0: Any,
        /,
        *,
        nonsmooth: Functional | None = None,
    ) -> OptimizationResult:
        """Minimize ``smooth + nonsmooth`` from ``x0``.

        Args:
            smooth: the differentiable part ``f``.
            x0: the starting point.
            nonsmooth: the part ``g`` with a proximal operator. When absent,
                this is plain gradient descent with a backtracked step.

        Returns:
            The optimization result.

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

    def _minimize(self, functional: Functional, x0: Any) -> OptimizationResult:
        return self._run(functional, None, x0)

    def _run(
        self, smooth: Functional, nonsmooth: Functional | None, x0: Any
    ) -> OptimizationResult:
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
                return OptimizationResult(
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
                return OptimizationResult(
                    x,
                    history[-1],
                    progress,
                    iteration,
                    evaluations,
                    True,
                    "step tolerance reached",
                    history,
                )

        return OptimizationResult(
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
        # slams the iterate to the minimizer of the non-smooth part alone.
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


class ProximalPoint(Optimizer):
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
            **kwargs: passed to :class:`~pygeoinf2.numerics.optimization.Optimizer`.
        """
        if step <= 0.0:
            raise ValueError("step must be positive.")
        self._step = step
        super().__init__(**kwargs)

    def minimize(self, functional: Functional, x0: Any, /) -> OptimizationResult:
        """Minimize, requiring a proximal operator.

        Args:
            functional: convex, with a proximal operator. A set's indicator
                is the usual case, its prox being the projection.
            x0: where to start.

        Returns:
            The optimization result.

        Raises:
            ValueError: if the functional has no proximal operator.
        """
        if not functional.has_prox:
            raise ValueError("ProximalPoint needs a functional with a prox.")
        return self._minimize(functional, x0)

    def _minimize(self, functional: Functional, x0: Any) -> OptimizationResult:
        space = functional.domain
        x = space.copy(x0)
        history = [functional(x)]

        for iteration in range(1, self._max_iterations + 1):
            nxt = functional.prox(x, self._step)
            movement = space.norm(space.subtract(nxt, x))
            x = nxt
            history.append(functional(x))
            if movement <= self._gtol:
                return OptimizationResult(
                    x,
                    history[-1],
                    movement,
                    iteration,
                    iteration + 1,
                    True,
                    "step tolerance reached",
                    history,
                )

        return OptimizationResult(
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
class BundleResult(OptimizationResult):
    """The outcome of a bundle minimization.

    An :class:`~pygeoinf2.numerics.optimization.OptimizationResult` whose
    ``gradient_norm`` is ``NaN``, a bundle method having no gradient to
    report, and whose ``history`` is the best value after each iteration;
    the fields below are the bundle's own.
    """

    gap: float = float("inf")
    """The model's predicted decrease at the last step.

    A *practical* stopping criterion, and stated as one. The cutting-plane
    model lies below the function everywhere, so the unregularized gap would
    bound ``f(center) - f*``; this one includes the proximal term, so it is
    the decrease predicted under that regularization and is generally smaller.
    It behaves like a bound and is the right thing to stop on, but it is not
    certified.
    """

    lower_bound: float = -np.inf
    """A certified lower bound on the minimum, where the method has one.

    The proximal method has no certified bound and leaves this at minus
    infinity; a method with one reports ``value - lower_bound`` as its
    ``gap``.
    """

    serious_steps: int = 0
    """How many iterations moved the stability center."""

    def __repr__(self) -> str:
        return (
            f"BundleResult(value={self.value:.6g}, "
            f"iterations={self.iterations}, converged={self.converged})"
        )


class ProximalBundleMethod(Optimizer):
    """Minimize a convex function from values and subgradients alone.

    A subgradient is a *lower* bound on a convex function everywhere, not just
    near where it was taken. A bundle method keeps the ones it has seen and
    minimizes their upper envelope — the best piecewise-linear model the
    information so far supports — with a proximal term to stop it running away
    from where the model is trustworthy.

    That is what makes it the right method for a support function: subgradient
    descent needs a step-size schedule chosen in advance and converges at
    ``1/sqrt(k)``, while here the model's own gap gives a stopping criterion
    that means something -- see :attr:`BundleResult.gap` for exactly what it
    does and does not bound.
    """

    def __init__(
        self,
        /,
        *,
        weight: float = 1.0,
        tolerance: float = 1e-8,
        max_iterations: int = 200,
        capacity: int = 40,
        descent: float = 0.1,
        qp_solver: Any = None,
    ) -> None:
        """
        Args:
            weight: the proximal weight. Larger keeps steps shorter.
            tolerance: stop when the model's gap falls below this times
                ``max(|f|, 1)``, with ``f`` the best value so far.
            max_iterations: the cap.
            capacity: how many cuts to keep; the oldest are dropped.
            descent: the fraction of the predicted decrease a step must
                deliver to be accepted as a serious step.
            qp_solver: a backend for the subproblem, the simplex-constrained
                QP in the number of cuts. ``None`` takes Clarabel or OSQP if
                either is installed and otherwise the accelerated projected
                gradient built in; ``"builtin"`` asks for the latter outright;
                a :class:`~pygeoinf2.numerics.quadratic_programming.QPSolver`
                is used as given. An exact backend is what the gap needs to be
                trusted to its last digits, and is faster than the projected
                gradient as well. SciPy's SLSQP is slower and less accurate
                than the projected gradient here, which is why it is not
                chosen by default.
        """
        if not 0.0 < descent < 1.0:
            raise ValueError(f"The descent fraction lies in (0, 1), got {descent}.")
        super().__init__(max_iterations=max_iterations)
        self._weight = weight
        self._gap_tolerance = tolerance
        self._capacity = capacity
        self._descent = descent
        self._qp_solver = qp_solver
        self._warm: np.ndarray | None = None
        # The cuts a stored Gram matrix was built from, and the matrix. Reset
        # at the start of each minimization.
        self._cache: tuple[list[int], np.ndarray] = ([], np.empty((0, 0)))

    @property
    def qp_solver(self) -> Any:
        """The subproblem backend, or ``None`` for the built-in method."""
        if self._qp_solver is None:
            from .quadratic_programming import ClarabelQPSolver, OSQPQPSolver

            self._qp_solver = "builtin"
            for backend in (ClarabelQPSolver, OSQPQPSolver):
                try:
                    self._qp_solver = backend()
                    break
                except ImportError:  # pragma: no cover - depends on the install
                    continue
        return None if self._qp_solver == "builtin" else self._qp_solver

    def minimize(
        self,
        functional: Functional,
        x0: Any,
        /,
        *,
        subgradient: Any = None,
    ) -> BundleResult:
        """Minimize a convex functional from a starting point.

        A subgradient is what the method needs, not a derivative, so the
        base class's requirement of one is not imposed here.

        Args:
            functional: convex, and able to supply a subgradient.
            x0: where to begin.
            subgradient: an override, called with a point and returning a
                subgradient. Defaults to the functional's own.

        Returns:
            The least value found, where it was found, and the model's gap at
            the last step (a practical bound, not a certified one).
        """
        return self._minimize(functional, x0, subgradient=subgradient)

    def _minimize(
        self, functional: Functional, x0: Any, *, subgradient: Any = None
    ) -> BundleResult:
        space = functional.domain
        start = x0
        slope = subgradient or functional.subgradient

        self._cache = ([], np.empty((0, 0)))
        self._warm = None
        center = space.copy(start)
        best = float(functional(center))
        # Each cut is (gradient, value, point). The point matters: a cut taken
        # elsewhere bounds f from below everywhere, but its offset *at the
        # current center* is f(x_i) + (g_i, c - x_i), and dropping the second
        # term makes every cut look tight and the method stop at once.
        cuts: list[tuple[Any, float, Any]] = []
        weight = self._weight
        gap = float("inf")

        # The bundle starts with the cut at the starting point; every later
        # cut is taken at the *candidate*, the textbook arrangement. A null
        # step leaves the center where it was, so a cut taken at the center
        # would duplicate one already in the bundle, learn nothing from the
        # trial point just evaluated, and make the model's Gram matrix
        # exactly singular.
        cuts.append((space.copy(slope(center)), best, space.copy(center)))
        evaluations = 1
        history: list[float] = []

        for iteration in range(1, self._max_iterations + 1):
            candidate, gap = self._solve_model(space, center, best, cuts, weight)
            if gap <= self._gap_tolerance * max(abs(best), 1.0):
                return BundleResult(
                    minimizer=center,
                    value=best,
                    gradient_norm=float("nan"),
                    iterations=iteration,
                    evaluations=evaluations,
                    converged=True,
                    message="gap tolerance reached",
                    history=history,
                    gap=gap,
                )

            value = float(functional(candidate))
            evaluations += 1
            cuts.append((space.copy(slope(candidate)), value, space.copy(candidate)))
            if len(cuts) > self._capacity:
                cuts.pop(0)

            if best - value >= self._descent * gap:
                center, best = candidate, value  # a serious step
                weight = max(weight * 0.5, 1e-8)
            else:
                weight = min(weight * 2.0, 1e12)  # a null step: trust less
            history.append(best)

        return BundleResult(
            minimizer=center,
            value=best,
            gradient_norm=float("nan"),
            iterations=self._max_iterations,
            evaluations=evaluations,
            converged=False,
            message="iteration limit reached",
            history=history,
            gap=gap,
        )

    def _gram(self, space: Any, cuts: Any) -> np.ndarray:
        """The cuts' Gram matrix, extended rather than rebuilt.

        A bundle grows by one cut per iteration and drops the oldest when it
        is full, so all but one row and column of this matrix are the same as
        last time. Rebuilding it would cost ``k^2`` inner products per
        iteration, on a space where an inner product is the expensive
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
        center: Any,
        value: float,
        cuts: Any,
        weight: float,
    ) -> tuple[Any, float]:
        """Minimize the cutting-plane model plus a proximal term.

        The dual of that quadratic program is a simplex-constrained least
        squares in the *number of cuts*, which is small -- so it is solved
        there rather than in the space, which may be large.

        Returns the candidate and the predicted decrease, which is the model's
        gap and the stopping criterion; :attr:`BundleResult.gap` says what it
        does and does not bound.
        """
        gram = self._gram(space, cuts)

        errors = np.array(
            [
                max(
                    value
                    - taken
                    - space.inner_product(gradient, space.subtract(center, point)),
                    0.0,
                )
                for gradient, taken, point in cuts
            ]
        )

        step_size = 1.0 / weight
        weights = self._simplex_weights(step_size * gram, errors)

        combination = space.zero()
        for coefficient, (gradient, _, _) in zip(weights, cuts):
            combination = space.axpy(coefficient, gradient, combination)
        candidate = space.add(center, space.scale(-step_size, combination))
        decrease = 0.5 * step_size * space.squared_norm(combination) + float(
            weights @ errors
        )
        return candidate, max(decrease, 0.0)

    def _simplex_weights(self, quadratic: np.ndarray, errors: np.ndarray) -> np.ndarray:
        """``argmin w' Q w / 2 + e' w`` over the unit simplex.

        Through the QP backend when one was given, warm-started from the
        previous subproblem's weights padded with a zero for the new cut, and
        through the built-in accelerated projected gradient otherwise.
        """
        size = errors.size
        backend = self.qp_solver
        if backend is None:
            return _minimize_on_simplex(quadratic, -errors)
        warm = None
        if self._warm is not None and self._warm.size in (size, size - 1):
            warm = np.zeros(size)
            warm[: self._warm.size] = self._warm
        elif self._warm is not None:
            # The bundle dropped its oldest cut: shift the weights along.
            warm = np.zeros(size)
            kept = self._warm[-(size - 1) :] if size > 1 else self._warm[:0]
            warm[: kept.size] = kept
        # sum w == 1, then 0 <= w_i.
        rows = np.vstack([np.ones((1, size)), np.eye(size)])
        lower = np.concatenate([[1.0], np.zeros(size)])
        upper = np.concatenate([[1.0], np.full(size, np.inf)])
        result = backend.solve(quadratic, errors, rows, lower, upper, x0=warm)
        if not result.solved:
            weights = _minimize_on_simplex(quadratic, -errors)
        else:
            weights = _project_on_simplex(np.asarray(result.x, dtype=float))
        self._warm = weights
        return weights


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


def _minimize_on_simplex(
    quadratic: np.ndarray,
    linear: np.ndarray,
    /,
    *,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
    warn_above: float = 1e-4,
) -> np.ndarray:
    """Minimize ``w' Q w / 2 - l' w`` over the unit simplex.

    Accelerated projected gradient -- FISTA -- with a step from the largest
    eigenvalue. Needs nothing installed, and has no setup cost on problems
    this small.

    **Accelerated, because plain projected gradient converges too slowly
    here.** The cuts a bundle accumulates are near-parallel by the end -- that
    is what it means for the method to be closing in -- so ``Q`` is a Gram
    matrix of near-dependent vectors and badly conditioned. Momentum is not
    monotone, so the best iterate seen is returned rather than the last; that
    costs nothing, the residual being computed every step for the stopping
    test anyway.

    Stopped on the KKT residual rather than on the iterate standing still. The
    two are not the same: a small step means the *method* has slowed, which is
    what an ill-conditioned problem does far from the optimum, while the
    residual is a statement about the answer. At the minimum the gradient is
    constant across the support and no smaller anywhere off it, so
    ``max_{w_i > 0} g_i - min_i g_i`` is zero there and is the residual used.

    Exhausting the iterations warns -- but only when the residual left is big
    enough to matter, which is what ``warn_above`` is for. A bundle method
    calls this at every step, so a subproblem quietly failing would show up
    only as an outer method that would not settle.

    The two thresholds are separate because, with ``Q`` as badly conditioned
    as a converging bundle makes it, the residual has a floor that no
    iteration count removes, and a single threshold would either warn half
    the time or never. :class:`ProximalBundleMethod` uses a QP backend for the
    subproblem when one is installed; this is the fallback for when none is.

    Args:
        quadratic: the ``Q`` above, symmetric positive semidefinite.
        linear: the ``l`` above.
        max_iterations: the cap.
        tolerance: the KKT residual to stop at, relative to the gradient's
            scale.
        warn_above: warn when the iterations run out with the residual still
            above this. Set to infinity to silence it.

    Returns:
        The minimizing weights.
    """
    size = linear.size
    step = 1.0 / max(float(np.linalg.eigvalsh(quadratic).max()), 1e-12)
    weights = np.full(size, 1.0 / size)
    # The extrapolated point the gradient is taken at, and the momentum
    # coefficient's running term.
    lookahead = weights.copy()
    momentum = 1.0
    # The best iterate seen. Acceleration is not monotone, so without this it
    # can end worse than where it passed through.
    best, residual = weights.copy(), float("inf")

    for _ in range(max_iterations):
        gradient = quadratic @ weights - linear
        support = weights > 0.0
        current = float(gradient[support].max() - gradient.min())
        if current < residual:
            best, residual = weights.copy(), current
        if residual <= tolerance * max(float(np.abs(gradient).max()), 1.0):
            return best

        moved = _project_on_simplex(lookahead - step * (quadratic @ lookahead - linear))
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
            f"The bundle subproblem did not converge in {max_iterations} "
            f"iterations; the KKT residual is {residual / scale:.3g} relative. "
            "The outer method will still make progress, but its gap is only "
            "as good as this solve.",
            RuntimeWarning,
            stacklevel=2,
        )
    return weights


@dataclass(frozen=True)
class SaddlePointResult(OptimizationResult):
    """The outcome of a primal-dual feasibility solve.

    An :class:`~pygeoinf2.numerics.optimization.OptimizationResult` read as
    the minimization of ``-(c, m)``: ``minimizer`` is the maximizing model
    and ``value`` is ``-(c, m)`` there, so the support value being sought
    is ``-value``. ``gradient_norm`` is ``NaN``, the method having no
    gradient to report; ``history`` is the feasibility residual after each
    iteration; the fields below are the splitting's own.
    """

    discrepancy: Any = None
    """The data-error vector ``v`` that goes with the model."""

    certificate: Any = None
    """The dual variable, which is the same object the dual route calls a
    certificate: a weighting of the data."""

    residual: float = float("nan")
    """``||G m + v - d||``, the feasibility gap. This is what convergence is
    declared on -- not the objective, which is monotone in neither variable."""

    def __repr__(self) -> str:
        return (
            f"SaddlePointResult(value={self.value:.6g}, "
            f"residual={self.residual:.3g}, converged={self.converged})"
        )


class ChambollePockSolver:
    """Maximize a linear functional over a feasible set, by primal-dual splitting.

    The primal form of the same question the dual route answers:

    .. code-block:: text

        maximize    (c, m)
        subject to  m in prior, v in noise, G m + v == d

    Chambolle and Pock's first-order method (2011) on the saddle-point form,
    with ``K == [G; I]``. Each step is one application of ``G``, one of its
    adjoint, and a *projection* onto each of the two sets -- so the cost per
    iteration is small and the number of them is large, which is the opposite
    trade to the bundle methods and the reason to have both.

    Converges at ``O(1/N)`` in the primal-dual gap when
    ``tau sigma ||K||^2 <= 1``; the step sizes are chosen to satisfy that from
    a power-iteration estimate of ``||G||`` unless given.

    **It projects rather than supports.** Every
    :class:`~pygeoinf2.geometry.convex.ConvexSet` knows its own nearest point,
    so a polytope or an intersection works without a case for it -- but an
    intersection projects by Dykstra, which is iterative, so the cost per step
    is then no longer small. That is a real trade and worth knowing about
    before choosing this route for such a set.

    It is at its best when the objective ``c`` changes and the feasible set
    does not: the step sizes and the operator norm are then computed once, and
    only the linear term moves.

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
        max_iterations: int = 1000,
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
            max_iterations: the cap.
            tolerance: on the feasibility residual.
            rng: for the power iteration that estimates ``||G||``.

        Raises:
            ValueError: if the sets do not live in the operator's spaces, the
                steps are not positive, or *theta* is negative.
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
        self._max_iterations = max_iterations
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
        """Maximize ``(objective, m)`` over the feasible set.

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
        history: list[float] = []

        for iteration in range(1, self._max_iterations + 1):
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
            history.append(residual)
            if residual < self._tolerance:
                return SaddlePointResult(
                    minimizer=model,
                    value=-float(model_space.inner_product(objective, model)),
                    gradient_norm=float("nan"),
                    iterations=iteration,
                    evaluations=iteration,
                    converged=True,
                    message="feasibility residual reached the tolerance",
                    history=history,
                    discrepancy=discrepancy,
                    certificate=certificate,
                    residual=residual,
                )

        return SaddlePointResult(
            minimizer=model,
            value=-float(model_space.inner_product(objective, model)),
            gradient_norm=float("nan"),
            iterations=self._max_iterations,
            evaluations=self._max_iterations,
            converged=False,
            message="iteration limit reached",
            history=history,
            discrepancy=discrepancy,
            certificate=certificate,
            residual=residual,
        )
