"""
Set-valued inference: a constraint set in, a constraint set out.

The feasible model set is ``S_M == S_M^0 ∩ A^-1(d - S_eta)`` and the answer is
its image ``T(S_M)``. Four routes compute that image, and they are the same set
— which is what makes them testable against each other. See DESIGN.md §18.3.

Here: route (a), the closed form for error-free data and a ball prior, and
route (b), the linear certificate, which is where Backus-Gilbert lives. Routes
(c) and (d) — the primal bisection and the dual with bundle methods — are
§18.3's expensive general cases.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np

from ..algebra.operators import LinearOperator
from ..algebra.spaces import HilbertSpace
from ..geometry.convex import Ball, ConvexSet, Ellipsoid
from ..geometry.subspaces import OrthogonalProjector
from ..numerics.solvers import CGSolver, LinearSolver
from ..traits import Traits
from .estimators import LinearPointEstimator, SetEstimator
from .problem import LinearForwardProblem

# Relative floor below which an eigenvalue of A A* is treated as zero, so
# that a null direction contributes nothing rather than dividing by noise.
_SPECTRUM_FLOOR = 1.0e-13

__all__ = ["BackusGilbert", "BackusInference", "FeasibleProperty"]


def _ball_radius(candidate: Any, name: str) -> float:
    """The radius of a ball, or a complaint naming what was wanted."""
    if not isinstance(candidate, Ball):
        raise TypeError(
            f"{name} must be a Ball for this route; got a "
            f"{type(candidate).__name__}. A general convex set needs route (d)."
        )
    return float(candidate.radius)


class BackusGilbert(LinearPointEstimator):
    """The optimally-averaged estimate of a property, with its error set.

    A *linear certificate*, in the sense of §18.3(b): the estimator is a fixed
    operator ``X`` applied to the data, chosen by minimising a quadratic
    surrogate for the width of the resulting bound,

    .. code-block:: text

        M^2 ||T - X A||_HS^2  +  D^2 ||X||_HS^2      ->      X = T A* (A A* + alpha)^-1

    with ``alpha == D^2 / M^2``. **Any** ``X`` gives a *valid* bound — that is
    weak duality, and it is why this route can never be wrong, only loose. This
    particular one is the least loose in an average sense.

    What comes out is not a number but a set. The estimate is ``X d``; the
    uncertainty is a Minkowski sum of two ellipsoids, one for what the data
    cannot resolve and one for the noise, and the two are reported separately
    because they respond to different remedies.
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        target: LinearOperator,
        prior: Ball,
        /,
        *,
        noise: Ball | None = None,
        level: float = 0.95,
        solver: LinearSolver | None = None,
    ) -> None:
        """
        Args:
            problem: the forward problem.
            target: the property operator ``T``.
            prior: a norm ball on the model space.
            noise: a norm ball on the data space. Taken from the problem's
                error if omitted, hardening a Gaussian one at ``level``.
            level: the confidence level used if a Gaussian error is hardened.
            solver: how to invert the data-space normal operator.
        """
        if target.domain != problem.model_space:
            raise ValueError("The property operator must act on the model space.")
        forward = problem.forward_operator
        model_radius = _ball_radius(prior, "The prior")

        if noise is None:
            noise = self._harden(problem, level=level)
        noise_radius = _ball_radius(noise, "The noise")

        if model_radius <= 0.0:
            raise ValueError("The prior ball must have a positive radius.")
        alpha = (noise_radius / model_radius) ** 2
        normal = forward @ forward.adjoint
        if alpha > 0.0:
            normal = normal + alpha * LinearOperator.identity(problem.data_space)
        inverse = (solver or CGSolver(rtol=1e-12))(
            normal.with_traits(Traits.POSITIVE_DEFINITE)
        )
        operator = target @ forward.adjoint @ inverse

        super().__init__(
            operator,
            forward_operator=forward,
            error=(
                problem.error
                if problem.has_error and not isinstance(problem.error, ConvexSet)
                else None
            ),
        )
        self._problem = problem
        self._target = target
        self._prior_radius = model_radius
        self._noise_radius = noise_radius

    @staticmethod
    def _harden(problem: LinearForwardProblem, /, *, level: float) -> Ball:
        """The problem's error as a ball, hardening a measure if need be."""
        if not problem.has_error:
            return Ball(problem.data_space, radius=0.0)
        if isinstance(problem.error, Ball):
            return problem.error
        if isinstance(problem.error, ConvexSet):
            raise TypeError(
                "This route needs the noise set to be a ball; a general convex "
                "one needs route (d). Pass noise= explicitly to bound it."
            )
        measure = problem.error_measure
        radius = float(
            np.sqrt(
                problem.critical_chi_squared(level=level)
                * measure.covariance.matrix(form="components").diagonal().mean()
            )
        )
        return Ball(problem.data_space, radius=radius)

    @property
    def unresolved(self) -> LinearOperator:
        """``T - X A``: what the estimate cannot see.

        The complement of the resolution, and the operator whose norm sets the
        first half of the error bound. An estimate is only as good as this is
        small.
        """
        return self._target - self.resolution

    def uncertainty(self, data: Any, /) -> ConvexSet:
        """The set of property values consistent with the data.

        The Minkowski sum ``X d + (T - X A) S_M + (-X) S_eta``, whose support
        function is
        ``(q, X d) + M ||(T - X A)* q|| + D ||X* q||`` — a resolution term and
        a noise term, added. Both are needed: shrinking one at the expense of
        the other is exactly what the choice of ``X`` trades.
        """
        centre = self(data)
        unresolved = self.unresolved
        prior_radius, noise_radius = self._prior_radius, self._noise_radius
        space = self.target_space
        operator = self.operator

        def support(direction: Any) -> float:
            resolution_term = prior_radius * self._problem.model_space.norm(
                unresolved.adjoint(direction)
            )
            noise_term = noise_radius * self._problem.data_space.norm(
                operator.adjoint(direction)
            )
            return space.inner_product(centre, direction) + resolution_term + noise_term

        return ConvexSet.from_support_function(space, support)

    def error_bars(self, data: Any, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Componentwise bounds, and the two contributions separately.

        Returns ``(estimate, resolution_half_width, noise_half_width)``. The
        split is the useful part: more data narrows the second and better
        coverage narrows the first, and a single number cannot say which is
        needed.
        """
        space = self.target_space
        estimate = self(data)
        unresolved, operator = self.unresolved, self.operator
        resolution, noise = [], []
        for index in range(space.dim):
            direction = space.basis_vector(index)
            resolution.append(
                self._prior_radius
                * self._problem.model_space.norm(unresolved.adjoint(direction))
            )
            noise.append(
                self._noise_radius
                * self._problem.data_space.norm(operator.adjoint(direction))
            )
        return estimate, np.array(resolution), np.array(noise)


class BackusInference(SetEstimator):
    """The exact feasible property set, where a closed form exists.

    Route (a) of §18.3: error-free data and a ball prior. Al-Attar (2021)
    eq. (2.84) gives the answer as an *ellipsoid*,

    .. code-block:: text

        { p : ((T P T*)^-1 (p - p~), p - p~) <= r^2 - ||m~||^2 }

    with ``m~`` the minimum-norm model fitting the data, ``p~ == T m~``, and
    ``P`` the projection onto the kernel of the forward operator. Every piece
    of that is already here: ``A A*`` is positive semidefinite by the
    palindrome rule, ``onto_kernel`` is example 18's subject, and ``Ellipsoid``
    carries the support function the bound's directional form needs.

    Costs ``dim(P) + 1`` minimum-norm solves, and nothing else.
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        target: LinearOperator,
        prior: Ball,
        /,
        *,
        solver: LinearSolver | None = None,
    ) -> None:
        """
        Args:
            problem: the forward problem. Its data are treated as exact.
            target: the property operator ``T``.
            prior: a norm ball on the model space, centred at the origin.
            solver: how to invert ``A A*`` and the property Gram.
        """
        if target.domain != problem.model_space:
            raise ValueError("The property operator must act on the model space.")
        self._problem = problem
        self._target = target
        self._radius = _ball_radius(prior, "The prior")
        self._solver = solver or CGSolver(rtol=1e-12)

        forward = problem.forward_operator
        normal = (forward @ forward.adjoint).with_traits(Traits.POSITIVE_DEFINITE)
        self._normal_inverse = self._solver(normal)
        self._kernel = OrthogonalProjector.onto_kernel(forward, solver=self._solver)
        # T P T*, the shape of the answer. Positive definite whenever the
        # property is not determined by the data alone.
        self._shape = (target @ self._kernel @ target.adjoint).with_traits(
            Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )

    @property
    def data_space(self) -> HilbertSpace:
        """The problem's data space."""
        return self._problem.data_space

    @property
    def target_space(self) -> HilbertSpace:
        """The property space."""
        return self._target.codomain

    @property
    def shape(self) -> LinearOperator:
        """``T P T*``: the shape of the answer, independent of the data.

        Only the centre and the size depend on the data. That is the same
        structure as a Gaussian estimator's data-independent covariance, and it
        arrives for the same reason.
        """
        return self._shape

    def minimum_norm_model(self, data: Any, /) -> Any:
        """``A* (A A*)^-1 d``, the smallest model fitting the data exactly."""
        return self._problem.forward_operator.adjoint(self._normal_inverse(data))

    def budget(self, data: Any, /) -> float:
        """``r^2 - ||m~||^2``: what the prior has left after fitting the data.

        Negative when no model within the prior ball fits the data at all,
        which is a statement about the data and the prior together and is
        reported rather than clipped.
        """
        model = self.minimum_norm_model(data)
        return self._radius**2 - self._problem.model_space.squared_norm(model)

    def __call__(self, data: Any) -> ConvexSet:
        """The feasible property set, as an ellipsoid."""
        budget = self.budget(data)
        if budget < 0.0:
            raise ValueError(
                f"No model within the prior ball fits these data: the smallest "
                f"one has norm {np.sqrt(self._radius**2 - budget):.4g} against "
                f"a bound of {self._radius:.4g}."
            )
        centre = self._target(self.minimum_norm_model(data))
        covariance = budget * self._shape
        return Ellipsoid(
            self.target_space,
            self._solver(covariance.with_traits(Traits.POSITIVE_DEFINITE)),
            centre=centre,
            covariance=covariance,
        )

    def inclusion_norm(self, value: Any, data: Any, /) -> float:
        """``min { ||m|| : A m == d, T m == p }``, the cost of a property value.

        Parker's joint data-property map ``C == (A, T)``: the smallest model
        reproducing *both* the data and the proposed property. A value is
        admissible exactly when this is within the prior bound, which is
        Al-Attar (2021) eq. (2.46).

        ``C C*`` acts on ``D (+) P``, so the solve is Parker's square system of
        size ``dim(D) + dim(P)`` — and it is positive semidefinite by the
        palindrome rule, so nothing has to be claimed about it.
        """
        from ..algebra.direct_sum import ColumnLinearOperator

        joint = ColumnLinearOperator([self._problem.forward_operator, self._target])
        normal = (joint @ joint.adjoint).with_traits(Traits.POSITIVE_DEFINITE)
        target = joint.codomain.from_components(
            np.concatenate(
                [
                    self.data_space.to_components(data),
                    self.target_space.to_components(value),
                ]
            )
        )
        model = joint.adjoint(self._solver(normal)(target))
        return self._problem.model_space.norm(model)

    def admits(self, value: Any, data: Any, /, *, rtol: float = 1e-8) -> bool:
        """Whether a property value is consistent with the data and the prior.

        The primal membership test, run without ever forming the feasible set.
        It agrees with ``self(data).contains(value)`` — the two are different
        computations of the same statement, and the test suite checks that they
        do.
        """
        return self.inclusion_norm(value, data) <= self._radius * (1.0 + rtol)

    def push_forward(self, operator: LinearOperator, /) -> "BackusInference":
        """The same inference about a further property of the model."""
        return BackusInference(
            self._problem,
            operator @ self._target,
            Ball(self._problem.model_space, radius=self._radius),
            solver=self._solver,
        )

    def prior_only(self) -> ConvexSet:
        """What the prior alone says, before any data.

        ``((T T*)^-1 p, p) <= r^2``, Al-Attar (2021) eq. (2.85). The bracket
        the data are supposed to improve on, and worth reporting beside the
        answer for exactly that reason.
        """
        covariance = self._radius**2 * (
            self._target @ self._target.adjoint
        ).with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE)
        return Ellipsoid(
            self.target_space,
            self._solver(covariance),
            covariance=covariance,
        )


class FeasibleProperty(SetEstimator):
    """The exact feasible property set for noisy data, by the primal route.

    Route (c) of §18.3, and the one BGP recommends when both the prior and the
    noise are norm balls. The support value in a direction is a concave
    maximisation over the intersection of two balls, and attaching multipliers
    to the two constraints turns its stationarity condition into

    .. code-block:: text

        (s I + t A* A) m == T* q + t A* d

    which is a **damped least-squares solve** — the same operation as a single
    regularised inversion, and the same primitive as §18.6. The multipliers are
    fixed by ``||m|| == M`` and ``||d - A m|| == D``, and both residuals are
    monotone in their own multiplier, so nested bisection converges.

    Two things this has that the dual route does not: it reuses solvers that
    already exist, and it produces the **extremal model** attaining each bound.
    What it lacks is generality: it is norm balls or nothing.
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        target: LinearOperator,
        prior: Ball,
        /,
        *,
        noise: Ball | None = None,
        level: float = 0.95,
        solver: LinearSolver | None = None,
        iterations: int = 60,
    ) -> None:
        """
        Args:
            problem: the forward problem.
            target: the property operator ``T``.
            prior: a norm ball on the model space.
            noise: a norm ball on the data space; taken from the problem if
                omitted.
            level: the level at which a Gaussian error is hardened.
            solver: how to invert the damped normal operator.
            iterations: bisection steps, on each of the two multipliers.
        """
        if target.domain != problem.model_space:
            raise ValueError("The property operator must act on the model space.")
        self._problem = problem
        self._target = target
        self._radius = _ball_radius(prior, "The prior")
        self._noise_radius = _ball_radius(
            BackusGilbert._harden(problem, level=level) if noise is None else noise,
            "The noise",
        )
        self._solver = solver or CGSolver(rtol=1e-12)
        self._iterations = iterations
        self._normal = problem.forward_operator.adjoint @ problem.forward_operator

    @property
    def data_space(self) -> HilbertSpace:
        """The problem's data space."""
        return self._problem.data_space

    @property
    def target_space(self) -> HilbertSpace:
        """The property space."""
        return self._target.codomain

    @cached_property
    def _data_gram(self) -> tuple[np.ndarray, np.ndarray]:
        """The eigendecomposition of ``A A*``, formed once.

        BGP §2.6's reduction, and the thing that makes the bisection
        affordable. Woodbury turns the model-space solve

        .. code-block:: text

            (s I + t A* A)^-1 == (1/s) [ I - A* (s/t I + A A*)^-1 A ]

        so every quantity the bisection tests — the model's norm and its misfit
        — becomes an ``O(dim(D))`` expression once ``A A*`` is diagonalised.
        Without it each of the four thousand bisection steps per direction
        would be a fresh Krylov solve in the model space.

        Costs ``dim(D)`` applications of the forward operator and its adjoint,
        once per estimator.
        """
        if not self.data_space.is_orthonormal:
            raise NotImplementedError(
                "The primal route reduces to the data space, which needs that "
                "space to be orthonormal. Every forward problem here has a "
                "Euclidean data space; if yours does not, use route (d)."
            )
        forward = self._problem.forward_operator
        gram = (forward @ forward.adjoint).matrix(form="components")
        values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
        return np.clip(values, 0.0, None), vectors

    def _kernel_part(self, vector: Any, forward: LinearOperator) -> float:
        """``||P_ker(A) v||^2``, from the data-space spectrum.

        Computed once per direction and at the natural scale, because the
        alternative — subtracting the range part from the whole at each
        bisection step — cancels: at a data weight of ``1e8`` the kernel term
        is ``1e-16`` of two quantities of order one, and comes out as noise.
        """
        values, vectors = self._data_gram
        image = vectors.T @ self.data_space.to_components(forward(vector))
        live = values > _SPECTRUM_FLOOR * max(values.max(initial=0.0), 1.0)
        return max(
            self._problem.model_space.squared_norm(vector)
            - float(np.sum(image[live] ** 2 / values[live])),
            0.0,
        )

    def _prepare(self, direction: Any, data: Any) -> dict:
        """Everything that does not change during the bisection."""
        space = self._problem.model_space
        forward = self._problem.forward_operator
        pulled = self._target.adjoint(direction)
        adjoint_data = forward.adjoint(data)
        components = self.data_space.to_components(data)
        values, vectors = self._data_gram
        return {
            "pulled": pulled,
            "adjoint_data": adjoint_data,
            "data": components,
            "forward_pulled": self.data_space.to_components(forward(pulled)),
            "gram_data": vectors @ (values * (vectors.T @ components)),
            "pulled_squared": space.squared_norm(pulled),
            "pulled_kernel_squared": self._kernel_part(pulled, forward),
            "cross": space.inner_product(pulled, adjoint_data),
            "adjoint_squared": space.squared_norm(adjoint_data),
        }

    def _state(self, prepared: dict, damping: float, weight: float) -> tuple:
        """``(||m*||, misfit)`` at a Tikhonov parameter and a data weight.

        Parameterised by ``gamma == s / t`` rather than by ``s``, which is what
        BGP §2.6 calls it and which is the only stable choice: with ``s``, the
        norm is a ratio of two large numbers as ``t`` grows and the whole
        expression cancels. Here

        .. code-block:: text

            m* == (gamma I + A* A)^-1 ( (1/t) T* q + A* d )

        stays bounded as ``t -> infinity``, tending to the damped least-squares
        solution. Both norms expand into inner products in the data space.
        """
        values, vectors = self._data_gram
        inverse_weight = 1.0 / weight
        forward_w = inverse_weight * prepared["forward_pulled"] + prepared["gram_data"]
        projected = vectors.T @ forward_w

        # A m* == (gamma I + A A*)^-1 A w', exactly -- the Woodbury difference
        # cancels identically here, so the misfit is stable at every damping.
        image = vectors @ (projected / (damping + values))

        # ||m*||^2 split into its kernel and range parts. Taking it from
        # (1/gamma)(w' - A* z) instead is a difference of two nearly equal
        # vectors divided by a small number, and at gamma of 1e-8 it returns
        # 2.8 for a model whose norm is 0.85.
        # A* d lies entirely in the range of A*, so the whole kernel part of
        # w' comes from (1/t) T* q -- exactly, at its own scale.
        kernel_squared = inverse_weight**2 * prepared["pulled_kernel_squared"]
        live = values > _SPECTRUM_FLOOR * max(values.max(initial=0.0), 1.0)
        model_squared = kernel_squared / damping**2 + float(
            np.sum(
                projected[live] ** 2 / (values[live] * (damping + values[live]) ** 2)
            )
        )
        residual = prepared["data"] - image
        return np.sqrt(model_squared), float(np.linalg.norm(residual))

    def _model(self, prepared: dict, damping: float, weight: float) -> Any:
        """The extremal model itself, at the cost of two adjoint applications."""
        space = self._problem.model_space
        values, vectors = self._data_gram
        inverse_weight = 1.0 / weight
        forward_w = inverse_weight * prepared["forward_pulled"] + prepared["gram_data"]
        projected = vectors.T @ forward_w
        # m* == (1/gamma) w'_ker  +  A* [ f / (Lambda (gamma + Lambda)) ],
        # which is the same vector as (1/gamma)(w' - A* z) with the
        # cancellation taken out analytically.
        adjoint = self._problem.forward_operator.adjoint
        live = values > _SPECTRUM_FLOOR * max(values.max(initial=0.0), 1.0)
        w = space.add(
            space.scale(inverse_weight, prepared["pulled"]), prepared["adjoint_data"]
        )
        pseudo = np.zeros_like(projected)
        pseudo[live] = projected[live] / values[live]
        kernel = space.subtract(
            w, adjoint(self.data_space.from_components(vectors @ pseudo))
        )
        weighted = np.zeros_like(projected)
        weighted[live] = projected[live] / (values[live] * (damping + values[live]))
        return space.add(
            space.scale(1.0 / damping, kernel),
            adjoint(self.data_space.from_components(vectors @ weighted)),
        )

    def _bisect(self, decreasing: Any, target: float) -> float:
        """The positive argument at which a decreasing function hits a target.

        Both bisections here are on a quantity that falls as its multiplier
        rises, so the same routine does both. **The bracket is widened at both
        ends**: widening only upwards leaves the search converging to whatever
        the lower end happened to be, which is a wrong answer that looks like a
        converged one.
        """
        low, high = 1.0, 1.0
        for _ in range(200):
            if decreasing(high) <= target:
                break
            high *= 10.0
        else:  # pragma: no cover - the function is unbounded above
            raise ValueError("The bisection could not bracket its target.")
        for _ in range(200):
            if decreasing(low) >= target:
                break
            low /= 10.0
        else:  # pragma: no cover
            raise ValueError("The bisection could not bracket its target.")
        for _ in range(self._iterations):
            middle = np.sqrt(low * high)
            if decreasing(middle) > target:
                low = middle
            else:
                high = middle
        return float(np.sqrt(low * high))

    def _fit_norm(self, prepared: dict, weight: float) -> float:
        """The damping at which the model's norm is the prior radius."""
        return self._bisect(
            lambda damping: self._state(prepared, damping, weight)[0], self._radius
        )

    def extremal_model(self, direction: Any, data: Any, /) -> Any:
        """The model of the feasible set furthest along a direction.

        What the dual route leaves implicit. Not generally unique — when the
        prior constraint is slack the null-space components are free — but the
        bound it attains is.
        """
        space = self._problem.model_space
        pulled = self._target.adjoint(direction)
        length = space.norm(pulled)
        if length == 0.0:
            return space.zero()

        # Prior-only: if the prior's own support point already fits the data,
        # the data constraint is slack and there is nothing to solve.
        flat = space.scale(self._radius / length, pulled)
        residual = self.data_space.subtract(data, self._problem.forward_operator(flat))
        if self.data_space.norm(residual) <= self._noise_radius:
            return flat

        prepared = self._prepare(direction, data)

        def misfit(weight: float) -> float:
            return self._state(prepared, self._fit_norm(prepared, weight), weight)[1]

        weight = self._bisect(misfit, self._noise_radius)
        return self._model(prepared, self._fit_norm(prepared, weight), weight)

    def support(self, direction: Any, data: Any, /) -> float:
        """The support value of the feasible property set in one direction."""
        model = self.extremal_model(direction, data)
        return self._problem.model_space.inner_product(
            self._target.adjoint(direction), model
        )

    def __call__(self, data: Any) -> ConvexSet:
        """The feasible property set, as a support-function oracle."""
        return ConvexSet.from_support_function(
            self.target_space,
            lambda direction: self.support(direction, data),
            maximiser=lambda direction: self._target(
                self.extremal_model(direction, data)
            ),
        )

    def push_forward(self, operator: LinearOperator, /) -> "FeasibleProperty":
        """The same inference about a further property."""
        return FeasibleProperty(
            self._problem,
            operator @ self._target,
            Ball(self._problem.model_space, radius=self._radius),
            noise=Ball(self.data_space, radius=self._noise_radius),
            solver=self._solver,
            iterations=self._iterations,
        )
