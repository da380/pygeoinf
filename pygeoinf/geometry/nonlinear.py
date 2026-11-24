"""
The feasible property set of a nonlinear inference problem, probed from inside.

The set is ``T(S)`` with ``S == {m in prior : d − F(m) in noise}``: every
value a property ``T m`` takes over the models that lie in a convex
constraint set and whose predictions ``F(m)`` fit the data to within a
convex noise set. With ``F`` linear this is the convex set of
:class:`~pygeoinf.geometry.feasible.FeasiblePropertySet`. With ``F``
nonlinear the feasible models are not a convex set, their image need be
neither convex nor connected, and no single optimization describes it.

What can always be had is a statement from inside. A model that is checked,
by evaluating ``F``, to lie in the prior and fit the data puts its property
in the set, whatever search produced it. :class:`LocalSearch` finds such
models by constrained local optimization: the farthest in a direction, the
one closest to the prior's center, the one with a given property. Each
answer is a point of the set and a bound on it from inside, and says
nothing about what lies beyond it: another start may find a part of the set
the first never reached.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..algebra.operators import (
    AffineOperator,
    Functional,
    LinearFunctional,
    LinearOperator,
    Operator,
)
from ..algebra.linearization import Linearization, QuadraticModel
from ..algebra.spaces import HilbertSpace
from ..numerics.optimization import (
    LBFGS,
    AugmentedLagrangian,
    Optimizer,
)
from ..numerics.solvers import ConvergenceError, LinearSolver
from ..parallel import parallel_map
from .convex import Ball, ConvexSet, Ellipsoid
from .feasible import (
    FeasiblePropertySet,
    FeasibleStructure,
    InclusionResult,
    SupportResult,
    _computes_support,
    _is_point,
    _require_property_coordinates,
    _support_of,
)
from .sets import SublevelSet, Subset

__all__ = ["LocalSearch", "NonlinearFeasiblePropertySet"]


def _scale_of(level: float) -> float:
    """The size a level is measured against: its own magnitude, or one at zero."""
    return abs(level) if level != 0.0 else 1.0


def _rescaled(functional: Functional, scale: float, shift: float) -> Functional:
    """``scale * f − shift``, keeping the gradient and the Hessian."""

    def value(x: Any) -> float:
        return float(functional(x)) * scale - shift

    def linearize(x: Any) -> QuadraticModel:
        local = functional.at(x)
        return QuadraticModel(
            x,
            float(local.value) * scale - shift,
            local.derivative * scale,
            None if local.hessian is None else local.hessian * scale,
        )

    return Functional.from_callables(
        functional.domain,
        value,
        linearize=linearize,
        hessian=(lambda x: functional.hessian(x) * scale)
        if functional.has_hessian
        else None,
    )


class LocalSearch:
    """Feasible models by constrained local optimization, holding tuning only.

    Each probe is a smooth constrained problem on the model space, solved by
    the method of multipliers
    (:class:`~pygeoinf.numerics.optimization.AugmentedLagrangian`) over an
    unconstrained optimizer: the two sets enter through their level
    functions, the noise set's composed with the residual ``d − F(m)``.
    Constraints are scaled by their levels, so the tolerances are relative
    and the same problem in other units stops at the same model.

    A method of multipliers stops within a tolerance of its constraints, on
    either side of them, and a model that misses the noise set by that much
    is not a model of the feasible set. So the search runs against both sets
    shrunk by ``margin``, a fraction of their levels larger than the
    tolerance, and its answer is then **checked**: the model is in the
    prior, and ``d − F(m)`` is in the noise set, by the sets' own membership
    tests at no tolerance. Only a model that passes is reported, and the
    best of the checked models met on the way is the answer, so a search
    that stalls still returns what it proved. The cost of the margin is an
    answer short of the local optimum by about that fraction.

    The answer is local. It depends on where the search starts, and a
    different start may reach a different part of the set.
    """

    name = "local"

    def __init__(
        self,
        /,
        *,
        optimizer: Optimizer | None = None,
        tolerance: float = 1e-6,
        margin: float = 1e-5,
        rtol: float = 1e-8,
        penalty: float = 10.0,
        max_iterations: int = 30,
    ) -> None:
        """
        Args:
            optimizer: the unconstrained method for the inner solves. L-BFGS
                with 500 iterations by default; a Newton method is usable
                when the forward operator carries a second derivative and
                the sets' level functions their Hessians.
            tolerance: the constraint violation at which a search may stop,
                relative to each set's level.
            margin: how far inside each set the search aims, as a fraction
                of the set's level. It must exceed the tolerance, so that a
                converged search ends strictly inside.
            rtol: the gradient tolerance the last inner solve must meet,
                relative to the terms that cancel at a stationary point.
            penalty: the augmented Lagrangian's initial penalty.
            max_iterations: the budget of multiplier updates; each inner
                solve has the optimizer's own.

        Raises:
            ValueError: if the margin does not exceed the tolerance.
        """
        if margin <= tolerance:
            raise ValueError(
                f"The margin ({margin}) must exceed the tolerance ({tolerance}): "
                "a search that stops within the tolerance of a set shrunk by "
                "less than that may stop outside the set itself."
            )
        self._optimizer = optimizer or LBFGS(max_iterations=500)
        self._tolerance = float(tolerance)
        self._margin = float(margin)
        self._rtol = float(rtol)
        self._penalty = float(penalty)
        self._max_iterations = int(max_iterations)

    @property
    def margin(self) -> float:
        """How far inside each set the search aims, as a fraction of its level."""
        return self._margin

    @property
    def tolerance(self) -> float:
        """The relative constraint violation at which a search may stop."""
        return self._tolerance

    def _method(
        self,
        /,
        *,
        equalities: Sequence[Functional] = (),
        inequalities: Sequence[Functional] = (),
    ) -> AugmentedLagrangian:
        return AugmentedLagrangian(
            self._optimizer,
            equalities=equalities,
            inequalities=inequalities,
            penalty=self._penalty,
            tolerance=self._tolerance,
            rtol=self._rtol,
            max_iterations=self._max_iterations,
        )

    # ----------------------------------------------------------------- #
    #                               Probes                               #
    # ----------------------------------------------------------------- #

    def support(
        self,
        feasible: "NonlinearFeasiblePropertySet",
        direction: Any,
        start: Any,
        /,
    ) -> SupportResult:
        """The farthest checked model in a direction that a search from ``start`` finds.

        The search opens with one step of the linearized problem: where the
        problem linearized about ``start`` has a feasible set, that set's
        extremal model in the direction is where the constrained
        optimization begins. It is the answer itself for a linear forward
        operator and near it for a nearly linear one, and it puts the
        optimization on the boundary it will work along rather than in the
        interior, where a first step is set by the objective alone.

        Args:
            feasible: the set.
            direction: the direction in the property space.
            start: the model to search from. It need not be feasible; when
                it is, the answer is at least its own property value.

        Returns:
            The result: ``inner`` is the property value, along the
            direction, of ``extremal_model``, a checked model of the
            feasible set, or minus infinity when the search met none.
            ``value``, the outer bound, is infinite: a search from inside
            bounds nothing from outside. ``converged`` says the search
            reached a constrained stationary point, which makes the answer
            a local maximum's and no more. ``certificate`` is the noise
            constraint's multiplier as a vector of the data space, the
            local counterpart of a dual certificate, and ``multipliers``
            those of the prior and the noise constraints.

        Raises:
            ValueError: if the property operator's adjoint annihilates the
                direction.
        """
        space = feasible.model_space
        pulled = feasible.property_operator.adjoint(direction)
        size = space.norm(pulled)
        if size == 0.0:
            raise ValueError("The direction is annihilated by the property operator.")
        # Minimize the negated property, in units of the prior's reach
        # along the direction where the prior can say what that is.
        reach = feasible._reach(pulled)
        objective = LinearFunctional.from_representer(
            space, space.scale(-1.0 / reach, pulled)
        )
        prior_constraint, noise_constraint = feasible._constraints(self._margin)
        stepped = feasible._linearized_step(direction, start)
        result = self._method(
            inequalities=(prior_constraint, noise_constraint)
        ).minimize(objective, start if stepped is None else stepped)

        def along(model: Any) -> float:
            return float(space.inner_product(pulled, model))

        best, inner = None, float("-inf")
        for candidate in (start, stepped, result.minimizer):
            if candidate is None:
                continue
            if feasible.is_feasible(candidate) and along(candidate) > inner:
                best, inner = candidate, along(candidate)
        at_result = best is result.minimizer
        message = result.message
        if best is None:
            message += "; no model met on the search passed the feasibility check"
        elif not at_result:
            message += "; the search did not improve on a feasible model it began with"
        return SupportResult(
            value=float("inf"),
            inner=inner,
            converged=bool(result.converged and at_result),
            algorithm=self.name,
            evaluations=int(result.evaluations),
            certificate=feasible._certificate(
                result.minimizer, float(result.inequality_multipliers[1]) * reach
            ),
            extremal_model=best,
            multipliers=tuple(
                float(multiplier) * reach
                for multiplier in result.inequality_multipliers
            ),
            message=message,
        )

    def feasibility(
        self, feasible: "NonlinearFeasiblePropertySet", start: Any, /
    ) -> InclusionResult:
        """The fitting model closest to the prior's center that a search from ``start`` finds.

        Minimizes the prior's level function over the models that fit the
        data, the nonlinear counterpart of the linear set's fitting model.

        Args:
            feasible: the set.
            start: the model to search from.

        Returns:
            The result: ``level`` is the prior's level function at
            ``fitting_model``, a model checked to fit the data, so a level
            within the prior's own proves the feasible set non-empty.
            ``level`` is infinite when the search met no model that fits,
            which proves nothing: ``lower_bound`` is minus infinity
            throughout, a local search having nothing to say from below.
        """
        return self._least_level(feasible, start, through=None)

    def inclusion(
        self,
        feasible: "NonlinearFeasiblePropertySet",
        value: Any,
        start: Any,
        /,
    ) -> InclusionResult:
        """The model with a given property closest to the prior's center, from ``start``.

        Minimizes the prior's level function over the models that fit the
        data and have the property ``value``. The property is held exactly
        rather than as a constraint: the search runs over the models
        ``m_p + N z``, with ``m_p`` the smallest model with the property and
        ``N`` the projector onto the property operator's kernel, so that
        every model it meets has the property and only the fit to the data
        is left to the method of multipliers.

        Args:
            feasible: the set.
            value: the property value.
            start: the model to search from; its projection onto the models
                with the property is where the search begins.

        Returns:
            The result: ``level`` is the prior's level function at
            ``fitting_model``, which fits the data and has the property, so
            a level within the prior's own proves the value is in the set.
            A larger or an infinite level proves nothing.

        Raises:
            TypeError: if the property space has no coordinate map.
        """
        _require_property_coordinates(feasible.domain, "Membership")
        return self._least_level(
            feasible, start, through=feasible._with_property(value)
        )

    def _least_level(
        self,
        feasible: "NonlinearFeasiblePropertySet",
        start: Any,
        /,
        *,
        through: AffineOperator | None,
    ) -> InclusionResult:
        prior_level = feasible._prior_level
        objective = _rescaled(prior_level, 1.0 / _scale_of(feasible.level), 0.0)
        _, noise_constraint = feasible._constraints(self._margin)
        if through is not None:
            objective, noise_constraint = (
                objective @ through,
                noise_constraint @ through,
            )
        result = self._method(inequalities=(noise_constraint,)).minimize(
            objective, start
        )
        best, level = None, float("inf")
        for candidate in (start, result.minimizer):
            if through is not None:
                candidate = through(candidate)
            if feasible.fits(candidate) and float(prior_level(candidate)) < level:
                best, level = candidate, float(prior_level(candidate))
        message = result.message
        if best is None:
            message += (
                "; no model met on the search passed the check, which does "
                "not show that none exists"
            )
        return InclusionResult(
            level=level,
            lower_bound=float("-inf"),
            converged=bool(result.converged and best is not None),
            algorithm=self.name,
            evaluations=int(result.evaluations),
            fitting_model=best,
            message=message,
        )

    def __repr__(self) -> str:
        return f"LocalSearch(tolerance={self._tolerance}, margin={self._margin})"


class NonlinearFeasiblePropertySet(Subset):
    """``T({m in prior : d − F(m) in noise})`` for a nonlinear ``F``, known by its points.

    The property values of every model that lies in a convex constraint set
    and fits the data, through a nonlinear forward operator, to within a
    convex noise set. The set lives in the property space and is in general
    neither convex nor connected. Its **support** in a direction is the
    supremum of the direction's inner product over the set, which is the
    support of its convex hull and describes the set only that far.

    Every probe here answers from inside, through a
    :class:`LocalSearch`: it exhibits a model, checked by evaluating the
    forward operator to lie in the prior and fit the data, and reports what
    that model proves. A support result's ``inner`` is attained and its
    ``value``, the outer bound, is infinite; an inclusion result's
    ``level`` is attained and its ``lower_bound`` is minus infinity. An
    answer is local to its start: :meth:`support_results` runs one search
    per start, and the spread of their answers is the first sign of a set
    in several parts.

    .. code-block:: python

        feasible = NonlinearFeasiblePropertySet(F, T, prior, data, noise=noise)
        feasible.fitting_model()                 # some model that fits
        feasible.support_result(direction)       # the farthest found, checked
        feasible.support_results(direction, starts)
        feasible.inclusion_result(value)         # a proof of membership, or none
        feasible.linearized(model)               # the linearized problem's convex set

    Both sets must carry a level function with a gradient, which is how they
    enter a smooth search; a ball and an ellipsoid do. Exact data are not
    taken: a noise set that is a single point leaves a search no interior to
    work in, and a small ball about it is the alternative.
    """

    def __init__(
        self,
        forward: Operator,
        property_operator: LinearOperator,
        prior: ConvexSet | SublevelSet,
        data: Any,
        /,
        *,
        noise: ConvexSet | SublevelSet,
        solver: LinearSolver | None = None,
    ) -> None:
        """
        Args:
            forward: ``F``, the forward operator, with a derivative.
            property_operator: ``T``, the property operator, acting on the
                model space.
            prior: the constraint set on the model space: a convex set, or
                a ``SublevelSet`` of a convex functional, taken as convex
                on the caller's word.
            data: the observations.
            noise: the confidence set on the data space, likewise.
            solver: how the linearized sets invert the operators they
                build; conjugate gradients by default.

        Raises:
            ValueError: if an operator or a set is on the wrong space, or
                the noise set is a single point.
            TypeError: if the forward operator carries no derivative, or a
                set is not convex or has no level function with a gradient.
        """
        super().__init__(property_operator.codomain)
        if not forward.has_derivative:
            raise TypeError("The forward operator must carry a derivative.")
        if property_operator.domain != forward.domain:
            raise ValueError("The property operator must act on the model space.")
        for subset, space, name in (
            (prior, forward.domain, "constraint"),
            (noise, forward.codomain, "noise"),
        ):
            if not isinstance(subset, (ConvexSet, SublevelSet)):
                raise TypeError(
                    f"The {name} set must be a ConvexSet, or a SublevelSet of "
                    "a convex functional."
                )
            if subset.domain != space:
                raise ValueError(f"The {name} set lies in the wrong space.")
            if not subset.has_smooth_level_function:
                raise TypeError(
                    f"The {name} set must carry a level function with a "
                    "gradient to enter a local search; a ball and an "
                    "ellipsoid do."
                )
        if _is_point(noise):
            raise ValueError(
                "Exact data are not taken: a noise set that is a single point "
                "leaves a local search no interior. Use a small ball about it."
            )
        self._forward = forward
        self._property_operator = property_operator
        self._prior = prior
        self._noise = noise
        self._data = data
        self._solver = solver
        self._prior_level = prior.level_function()
        self._noise_level = noise.level_function()
        self._cache: dict[Any, Any] = {}

    # ----------------------------------------------------------------- #
    #                              Structure                             #
    # ----------------------------------------------------------------- #

    @property
    def forward_operator(self) -> Operator:
        """``F``, the forward operator."""
        return self._forward

    @property
    def property_operator(self) -> LinearOperator:
        """``T``, the property operator."""
        return self._property_operator

    @property
    def prior(self) -> ConvexSet | SublevelSet:
        """The constraint set on the model space."""
        return self._prior

    @property
    def noise(self) -> ConvexSet | SublevelSet:
        """The confidence set on the data space."""
        return self._noise

    @property
    def data(self) -> Any:
        """The observations."""
        return self._data

    @property
    def model_space(self) -> HilbertSpace:
        """The model space, which ``F`` and ``T`` act on."""
        return self._forward.domain

    @property
    def data_space(self) -> HilbertSpace:
        """The data space."""
        return self._forward.codomain

    @property
    def level(self) -> float:
        """The prior's own level, which an inclusion result's level is read against."""
        return float(self._prior.level)

    def with_data(self, data: Any, /) -> "NonlinearFeasiblePropertySet":
        """The set for other data, on the same operators and sets."""
        return NonlinearFeasiblePropertySet(
            self._forward,
            self._property_operator,
            self._prior,
            data,
            noise=self._noise,
            solver=self._solver,
        )

    @property
    def has_membership(self) -> bool:
        """False: membership is proved by exhibiting a model, and not otherwise decided."""
        return False

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """Not decided by a search from inside.

        Args:
            x: a property value.
            rtol: unused.

        Raises:
            NotImplementedError: always. A model with the property proves
                membership and the lack of one proves nothing;
                :meth:`inclusion_result` reports which was found.
        """
        raise NotImplementedError(
            "Membership of a nonlinear feasible set is proved by a model and "
            "not decided without one; inclusion_result() reports what a local "
            "search found."
        )

    # ----------------------------------------------------------------- #
    #                         Checks by evaluation                       #
    # ----------------------------------------------------------------- #

    def residual(self, model: Any, /) -> Any:
        """``d − F(m)``, the part of the data a model leaves to the noise."""
        return self.data_space.subtract(self._data, self._forward(model))

    def fits(self, model: Any, /) -> bool:
        """Whether a model's residual lies in the noise set, by evaluating ``F``.

        Decided by the noise set's own membership test at no tolerance.
        """
        return bool(self._noise.contains(self.residual(model), rtol=0.0))

    def is_feasible(self, model: Any, /) -> bool:
        """Whether a model lies in the prior and fits the data, by evaluation.

        The check every reported model has passed: both sets' own membership
        tests at no tolerance, the noise set's on ``d − F(m)``.
        """
        return bool(self._prior.contains(model, rtol=0.0)) and self.fits(model)

    # ----------------------------------------------------------------- #
    #                         Pieces of a search                         #
    # ----------------------------------------------------------------- #

    def _reach(self, pulled: Any) -> float:
        """The prior's width along a model-space direction, or the direction's norm."""
        space = self.model_space
        if _computes_support(self._prior):
            support = _support_of(self._prior)
            width = float(support(pulled)) + float(support(space.negative(pulled)))
            if np.isfinite(width) and width > 0.0:
                return width
        return float(space.norm(pulled))

    def _constraints(self, margin: float) -> tuple[Functional, Functional]:
        """The two sets, shrunk by a margin, as functionals held at or below zero."""

        def build() -> tuple[Functional, Functional]:
            data_space = self.data_space
            prior_scale = _scale_of(self.level)
            noise_scale = _scale_of(float(self._noise.level))
            flip = AffineOperator(
                LinearOperator.identity(data_space) * (-1.0), self._data
            )
            misfit = self._noise_level @ (flip @ self._forward)
            return (
                _rescaled(
                    self._prior_level,
                    1.0 / prior_scale,
                    self.level / prior_scale - margin,
                ),
                _rescaled(
                    misfit,
                    1.0 / noise_scale,
                    float(self._noise.level) / noise_scale - margin,
                ),
            )

        key = ("constraints", margin)
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def _certificate(self, model: Any, multiplier: float) -> Any:
        """The noise constraint's multiplier as a vector of the data space.

        At a constrained stationary point ``T*q == F'(m)* λ`` plus a normal
        to the prior, with ``−λ`` a normal to the noise set at the
        residual: the multiplier times the gradient of the noise set's
        scaled level function there.
        """
        data_space = self.data_space
        gradient = self._noise_level.gradient(self.residual(model))
        return data_space.scale(
            -multiplier / _scale_of(float(self._noise.level)), gradient
        )

    def _property_rows(self) -> tuple[list[Any], list[float]]:
        """``T* e_i`` for the property space's basis, and the prior's width along each."""
        if "rows" not in self._cache:
            space = self.domain
            rows = [
                self._property_operator.adjoint(space.basis_vector(i))
                for i in range(space.dim)
            ]
            self._cache["rows"] = (rows, [self._reach(row) for row in rows])
        return self._cache["rows"]

    def _with_property(self, value: Any) -> AffineOperator:
        """``z -> m_p + N z``: the models whose property is ``value``, parameterized by the model space.

        ``m_p`` is the smallest model with the property and ``N`` the
        orthogonal projector onto the property operator's kernel, both in
        the model space's metric, through the Gram matrix of the rows
        ``T* e_i``.
        """
        space, property_space = self.model_space, self.domain
        rows, _ = self._property_rows()
        if "gram" not in self._cache:
            self._cache["gram"] = np.array(
                [[space.inner_product(a, b) for b in rows] for a in rows]
            )
        gram = self._cache["gram"]

        def combine(right: np.ndarray) -> Any:
            weights = np.linalg.lstsq(gram, right, rcond=None)[0]
            out = space.zero()
            for weight, row in zip(weights, rows):
                out = space.axpy(float(weight), row, out)
            return out

        def project(vector: Any) -> Any:
            along = np.array([space.inner_product(row, vector) for row in rows])
            return space.subtract(vector, combine(along))

        offsets = np.array(
            [
                property_space.inner_product(property_space.basis_vector(i), value)
                for i in range(property_space.dim)
            ]
        )
        return AffineOperator(
            LinearOperator.self_adjoint(space, project), combine(offsets)
        )

    def _default_start(self) -> Any:
        prior = self._prior
        if isinstance(prior, (Ball, Ellipsoid)):
            return prior.center
        return self.model_space.zero()

    # ----------------------------------------------------------------- #
    #                               Probes                               #
    # ----------------------------------------------------------------- #

    def fitting_result(
        self, /, *, start: Any = None, via: LocalSearch | None = None
    ) -> InclusionResult:
        """The fitting model closest to the prior's center that a local search finds.

        Args:
            start: the model to search from; the prior's center, or the
                zero model for a prior without one, if omitted.
            via: the search, with its own tuning; the default
                :class:`LocalSearch` if omitted.

        Returns:
            The result; see :meth:`LocalSearch.feasibility`.
        """
        default = start is None and via is None
        if default and "fitting" in self._cache:
            return self._cache["fitting"]
        result = (via or LocalSearch()).feasibility(
            self, self._default_start() if start is None else start
        )
        if default:
            self._cache["fitting"] = result
        return result

    def fitting_model(
        self, /, *, start: Any = None, via: LocalSearch | None = None
    ) -> Any:
        """A model in the prior that fits the data, found by a local search.

        Args:
            start: the model to search from; see :meth:`fitting_result`.
            via: the search; the default :class:`LocalSearch` if omitted.

        Returns:
            The model, checked to lie in the prior and fit the data.

        Raises:
            ValueError: if the search found no such model, which does not
                show the feasible set empty: another start may find one.
        """
        result = self.fitting_result(start=start, via=via)
        model = result.fitting_model
        if model is None or not self._prior.contains(model, rtol=0.0):
            raise ValueError(
                "The local search found no model inside the prior set that "
                f"fits the data ({result.message}). That does not show the "
                "feasible set empty; another start may find one."
            )
        return model

    def linearized(self, model: Any, /) -> FeasiblePropertySet:
        """The feasible property set of the problem linearized about a model.

        The convex set built on the derivative ``A == F'(m*)`` with the data
        shifted to ``d − F(m*) + A m*``, the constraint and noise sets as
        they are: exact for the linearized problem, and neither inside nor
        outside the real one.

        Args:
            model: ``m*``, the linearization point.
        """
        return self._linearized(self._forward.at(model))

    def _linearized(self, linearization: Linearization) -> FeasiblePropertySet:
        data_space = self.data_space
        derivative = linearization.derivative
        structure = FeasibleStructure(
            derivative,
            self._property_operator,
            self._prior,
            noise=self._noise,
            solver=self._solver,
        )
        shifted = data_space.add(
            data_space.subtract(self._data, linearization.value),
            derivative(linearization.point),
        )
        return FeasiblePropertySet.from_structure(structure, shifted)

    def _linearized_step(self, direction: Any, model: Any) -> Any | None:
        """The extremal model, in a direction, of the problem linearized about a model.

        None when the linearized set is empty or exhibits no model, as it
        may be about a model that does not fit.
        """
        try:
            return self.linearized(model).support_result(direction).extremal_model
        except (ValueError, ConvergenceError):
            return None

    def support_result(
        self,
        direction: Any,
        /,
        *,
        start: Any = None,
        via: LocalSearch | None = None,
    ) -> SupportResult:
        """The farthest checked model in a direction that one local search finds.

        Args:
            direction: the direction in the property space.
            start: the model to search from; :meth:`fitting_model` if
                omitted.
            via: the search; the default :class:`LocalSearch` if omitted.

        Returns:
            The result, with ``inner`` attained by ``extremal_model`` and
            ``value`` infinite; see :meth:`LocalSearch.support`.

        Raises:
            ValueError: if no start is given and no fitting model is found
                to start from.
        """
        origin = self.fitting_model() if start is None else start
        return (via or LocalSearch()).support(self, direction, origin)

    def support_results(
        self,
        direction: Any,
        starts: Sequence[Any],
        /,
        *,
        via: LocalSearch | None = None,
        n_jobs: int | None = None,
    ) -> tuple[SupportResult, ...]:
        """One local search per start, in a direction.

        The largest ``inner`` among them is the best bound from inside, and
        answers that differ by more than the searches' tolerance come from
        different local maxima, often different parts of the set.

        Args:
            direction: the direction in the property space.
            starts: the models to search from.
            via: the search; the default :class:`LocalSearch` if omitted.
            n_jobs: worker processes for the searches, which are
                independent. Serial by default.

        Returns:
            The results, in the order of the starts.
        """
        search = via or LocalSearch()
        return tuple(
            parallel_map(
                lambda start: search.support(self, direction, start),
                starts,
                n_jobs=n_jobs,
            )
        )

    def inclusion_result(
        self,
        value: Any,
        /,
        *,
        start: Any = None,
        via: LocalSearch | None = None,
    ) -> InclusionResult:
        """The model with a given property closest to the prior's center, by a local search.

        A ``level`` at or below :attr:`level` proves the value is in the
        set, by the ``fitting_model`` that has it. Anything else proves
        nothing.

        Args:
            value: the property value.
            start: the model to search from; :meth:`fitting_model` if
                omitted.
            via: the search; the default :class:`LocalSearch` if omitted.

        Returns:
            The result; see :meth:`LocalSearch.inclusion`.

        Raises:
            TypeError: if the property space has no coordinate map.
            ValueError: if no start is given and no fitting model is found.
        """
        origin = self.fitting_model() if start is None else start
        return (via or LocalSearch()).inclusion(self, value, origin)

    def __repr__(self) -> str:
        return (
            f"NonlinearFeasiblePropertySet(dim={self.domain.dim}, "
            f"prior={self._prior!r}, noise={self._noise!r})"
        )
