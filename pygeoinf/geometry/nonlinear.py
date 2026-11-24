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

A statement from outside needs a bound on how far ``F`` departs from its
linearization, and such a bound is only useful locally. On a :class:`Cell`,
a piece of the model space about a model ``m_K``, write
``R(m) == F(m) − F(m_K) − F'(m_K)(m − m_K)`` for the remainder. A
:class:`RemainderBound` claims a convex set ``Z`` of the data space that
holds ``R(m)`` for every model of the cell, and then the feasible models of
the cell lie between two feasible sets of the *linearized* problem: the one
whose noise set is grown by ``Z``, which contains them, and the one whose
noise set is shrunk by ``Z``, which they contain. Both are convex sets with
every probe of the linear theory. The claim is the caller's, certified or
estimated, and every model evaluated in a cell tests it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from numpy.random import Generator

from ..algebra.operators import (
    AffineOperator,
    Functional,
    LinearFunctional,
    LinearOperator,
    Operator,
)
from ..algebra.direct_sum import ColumnLinearOperator
from ..algebra.linearization import Linearization, QuadraticModel
from ..algebra.spaces import CoordinateSpace, EuclideanSpace, HilbertSpace, resolve_rng
from ..numerics.optimization import (
    LBFGS,
    AugmentedLagrangian,
    Optimizer,
)
from ..numerics.solvers import ConvergenceError, LinearSolver
from ..parallel import parallel_map
from .convex import Ball, ConvexSet, Ellipsoid, Zonotope
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

__all__ = [
    "Cell",
    "CellSupportResult",
    "ComponentwiseRemainder",
    "LocalSearch",
    "NonlinearFeasiblePropertySet",
    "NormRemainder",
    "RemainderBound",
    "SampledRemainder",
]


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


# --------------------------------------------------------------------- #
#                      Cells and remainder bounds                        #
# --------------------------------------------------------------------- #


class Cell:
    """A piece of the model space about a model: slabs along chosen directions.

    ``{ m : |(u_j, m − center)| <= w_j for every j, |m − center| <= radius }``.
    A cell is where a forward operator is compared with its linearization
    about the center, so it is thin along the directions ``u_j`` in which
    the operator curves and as wide as the constraint set allows in the
    others, which is what ``radius`` records. A cell is meant to be
    intersected with a constraint set, and has no boundary of its own
    across its slabs beyond the radius.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        center: Any,
        /,
        *,
        directions: Sequence[Any] = (),
        half_widths: Sequence[float] = (),
        radius: float = float("inf"),
    ) -> None:
        """
        Args:
            domain: the model space.
            center: the model the cell is about.
            directions: the vectors ``u_j`` the slabs are taken along. They
                must be linearly independent.
            half_widths: the half-width ``w_j`` of each slab, positive.
            radius: a bound on the distance of any model of the cell from
                the center; infinite if nothing is known.

        Raises:
            ValueError: if the half-widths do not match the directions in
                number, one is not positive, or the directions are
                dependent.
        """
        self._domain = domain
        self._center = center
        self._directions = tuple(directions)
        self._half_widths = np.asarray(half_widths, dtype=float).reshape(-1)
        self._radius = float(radius)
        if self._half_widths.shape != (len(self._directions),):
            raise ValueError("There must be one half-width per direction.")
        if np.any(self._half_widths <= 0.0):
            raise ValueError("Every half-width must be positive.")
        if self._radius <= 0.0:
            raise ValueError("The radius must be positive.")
        self._gram = np.array(
            [
                [domain.inner_product(a, b) for b in self._directions]
                for a in self._directions
            ]
        ).reshape(len(self._directions), len(self._directions))
        if self._directions:
            sizes = np.sqrt(np.diag(self._gram))
            if np.linalg.eigvalsh(self._gram / np.outer(sizes, sizes)).min() < 1e-10:
                raise ValueError("The slab directions must be linearly independent.")

    @property
    def domain(self) -> HilbertSpace:
        """The model space."""
        return self._domain

    @property
    def center(self) -> Any:
        """The model the cell is about."""
        return self._center

    @property
    def directions(self) -> tuple[Any, ...]:
        """The vectors the slabs are taken along."""
        return self._directions

    @property
    def half_widths(self) -> np.ndarray:
        """The half-width of each slab."""
        return self._half_widths

    @property
    def radius(self) -> float:
        """A bound on the distance of any model of the cell from the center."""
        return self._radius

    def offsets(self, model: Any, /) -> np.ndarray:
        """``(u_j, m − center)`` for each slab direction."""
        space = self._domain
        offset = space.subtract(model, self._center)
        return np.array([space.inner_product(u, offset) for u in self._directions])

    def point(self, offsets: Any, /) -> Any:
        """The model nearest the center with the given :meth:`offsets`.

        Args:
            offsets: one value per slab direction.
        """
        space = self._domain
        out = space.copy(self._center)
        if self._directions:
            weights = np.linalg.solve(self._gram, np.asarray(offsets, dtype=float))
            for weight, direction in zip(weights, self._directions):
                out = space.axpy(float(weight), direction, out)
        return out

    def contains(self, model: Any, /, *, rtol: float = 1e-9) -> bool:
        """Whether a model lies within every slab and within the radius.

        Args:
            model: a model.
            rtol: how far outside still counts as inside, relative to each
                half-width and to the radius.
        """
        space = self._domain
        if np.any(np.abs(self.offsets(model)) > self._half_widths * (1.0 + rtol)):
            return False
        distance = space.norm(space.subtract(model, self._center))
        return bool(distance <= self._radius * (1.0 + rtol))

    def half_width(self, vector: Any, /) -> float:
        """A bound on ``|(v, m − center)|`` over the cell, for any vector ``v``.

        The part of ``v`` in the span of the slab directions,
        ``sum_j a_j u_j``, is bounded by ``sum_j |a_j| w_j``, and the rest
        by its norm times the radius; the whole by the smaller of that sum
        and ``|v|`` times the radius. Exact for a slab direction itself.

        Args:
            vector: a vector of the model space.
        """
        space = self._domain
        size = float(space.norm(vector))
        whole = size * self._radius if size > 0.0 else 0.0
        if not self._directions:
            return whole
        pairings = np.array([space.inner_product(u, vector) for u in self._directions])
        weights = np.linalg.solve(self._gram, pairings)
        # What is left of |v|^2 outside the span, to the rounding of the
        # subtraction; below that the vector is in the span.
        rest = max(size**2 - float(pairings @ weights), 0.0)
        rest = 0.0 if rest <= 1e-12 * size**2 else rest**0.5
        split = float(np.abs(weights) @ self._half_widths)
        if rest > 0.0:
            split += rest * self._radius
        return min(split, whole)

    def row_operator(self) -> LinearOperator:
        """``m -> ((u_j, m))_j``, into a Euclidean space of the slabs' number."""
        space, directions = self._domain, self._directions

        def value(model: Any) -> np.ndarray:
            return np.array([space.inner_product(u, model) for u in directions])

        def adjoint(weights: np.ndarray) -> Any:
            out = space.zero()
            for weight, direction in zip(weights, directions):
                out = space.axpy(float(weight), direction, out)
            return out

        return LinearOperator.from_callables(
            space, EuclideanSpace(len(directions)), value, adjoint=adjoint
        )

    def __repr__(self) -> str:
        return f"Cell(slabs={len(self._directions)}, radius={self._radius:.3g})"


class RemainderBound(ABC):
    """A claim about how far a forward operator departs from its linearization over a cell.

    For a cell about ``m_K`` the remainder is
    ``R(m) == F(m) − F(m_K) − F'(m_K)(m − m_K)``, and :meth:`remainder_set`
    returns a convex set of the data space claimed to hold it for every
    model of the cell. The claim concerns one forward operator, which the
    bound was written for. ``certified`` says whether it is a theorem about
    that operator or an estimate; an outer bound on a feasible set is as
    good as the claim it was built on, and reports which kind that was.
    """

    def __init__(self, /, *, certified: bool) -> None:
        """
        Args:
            certified: whether the claim is proved for the operator, as
                opposed to estimated.
        """
        self._certified = bool(certified)

    @property
    def certified(self) -> bool:
        """Whether the claim is proved, as opposed to estimated."""
        return self._certified

    @abstractmethod
    def remainder_set(self, cell: Cell, /) -> ConvexSet:
        """A convex set of the data space claimed to hold the remainder over the cell.

        Args:
            cell: the cell, whose center is the linearization point.

        Returns:
            The set: a ball, or a set with a projection and a
            ``bounding_ball``.
        """


class NormRemainder(RemainderBound):
    """The remainder bounded in norm: ``|F''(m)[v, v]| <= L |v|^2`` over the cell.

    Then ``|R(m)| <= (L / 2) |m − m_K|^2``, and the remainder lies in the
    ball of radius ``(L / 2) radius^2`` about the origin of the data space,
    the cell's radius being the largest distance from its center. The
    bound takes no account of the cell's slabs: it is the right one for a
    cell that is small in every direction and of little use for one as
    wide as a constraint set on a function space.
    """

    def __init__(
        self, data_space: HilbertSpace, constant: float, /, *, certified: bool = True
    ) -> None:
        """
        Args:
            data_space: the forward operator's codomain.
            constant: ``L``, a bound on the second derivative's norm as a
                bilinear map, over every cell the bound is asked about.
            certified: whether ``L`` is proved for the operator.

        Raises:
            ValueError: if the constant is negative.
        """
        super().__init__(certified=certified)
        if constant < 0.0:
            raise ValueError("The constant must not be negative.")
        self._data_space = data_space
        self._constant = float(constant)

    def remainder_set(self, cell: Cell, /) -> Ball:
        """The ball of radius ``(L / 2) radius^2`` about the origin.

        Raises:
            ValueError: if the cell has no finite radius.
        """
        if not np.isfinite(cell.radius):
            raise ValueError(
                "A norm bound needs a cell of finite radius; give the cell "
                "one, or use a bound that reads the cell's slabs."
            )
        return Ball(self._data_space, radius=0.5 * self._constant * cell.radius**2)


class ComponentwiseRemainder(RemainderBound):
    """The remainder of ``F(m) = sum_i g_i((b_i, m)) e_i``, bounded component by component.

    A forward operator that applies a scalar function to each of a set of
    linear functionals of the model, over the data space's basis ``e_i``:
    each datum sees the model through one number ``t_i = (b_i, m)``, and
    its remainder is ``g_i''(s) (b_i, m − m_K)^2 / 2`` at some ``s`` between
    the two values of ``t_i``. With ``w_i`` the cell's half-width along
    ``b_i`` and ``[lo_i, hi_i]`` a range of ``g_i''`` over
    ``[t_i − w_i, t_i + w_i]``, the remainder's ``i``-th coefficient lies
    between ``min(lo_i, 0) w_i^2 / 2`` and ``max(hi_i, 0) w_i^2 / 2``. A
    second derivative of one sign gives a one-sided range, half the size of
    the symmetric one a norm would give, and a datum that is linear in the
    model gives none at all.
    """

    def __init__(
        self,
        data_space: CoordinateSpace,
        rows: Sequence[Any],
        curvature: Callable[[np.ndarray, np.ndarray], tuple[Any, Any]],
        /,
        *,
        certified: bool = True,
    ) -> None:
        """
        Args:
            data_space: the forward operator's codomain, whose basis the
                components are taken in.
            rows: the model-space vectors ``b_i``, one per basis vector of
                the data space.
            curvature: called with the arrays ``t − w`` and ``t + w`` and
                returning two arrays, a lower and an upper bound on each
                ``g_i''`` over its interval.
            certified: whether the ranges are proved.

        Raises:
            ValueError: if the rows do not match the data space's dimension.
        """
        super().__init__(certified=certified)
        self._data_space = data_space
        self._rows = tuple(rows)
        self._curvature = curvature
        if len(self._rows) != data_space.dim:
            raise ValueError(
                "There must be one row per basis vector of the data space."
            )

    def remainder_set(self, cell: Cell, /) -> Zonotope:
        """The box of coefficient ranges over the data space's basis.

        Raises:
            ValueError: if the cell is unbounded along a row, having neither
                a finite radius nor that row within its slabs' span.
        """
        space, data_space = cell.domain, self._data_space
        at_center = np.array(
            [space.inner_product(row, cell.center) for row in self._rows]
        )
        widths = np.array([cell.half_width(row) for row in self._rows])
        if not np.all(np.isfinite(widths)):
            raise ValueError(
                "The cell is unbounded along a row; give it a radius or a "
                "slab along every row the forward operator curves in."
            )
        low, high = self._curvature(at_center - widths, at_center + widths)
        low = np.minimum(np.asarray(low, dtype=float), 0.0)
        high = np.maximum(np.asarray(high, dtype=float), 0.0)
        return Zonotope(
            data_space,
            [data_space.basis_vector(i) for i in range(data_space.dim)],
            lower=0.5 * low * widths**2,
            upper=0.5 * high * widths**2,
        )


class SampledRemainder(RemainderBound):
    """The remainder estimated from evaluations of the forward operator: not certified.

    The remainder is computed at models of the cell, at the ends of each
    slab, at the corners where they are few, and at random points between,
    all displaced from the center within the span of the slab directions.
    The estimate is the box, over the data space's basis, of the
    coefficients seen, widened by a safety factor and taken to include the
    origin. Nothing proves the box holds the remainder elsewhere in the
    cell, and in particular nothing is learned of curvature across the
    slabs, so a bound built on this is an estimate and says so.
    """

    def __init__(
        self,
        forward: Operator,
        /,
        *,
        samples: int = 32,
        safety: float = 2.0,
        rng: Generator | None = None,
    ) -> None:
        """
        Args:
            forward: the forward operator, into a space with coordinates.
            samples: how many random models to evaluate beyond the ends of
                the slabs and their corners.
            safety: the factor the observed ranges are widened by; at least
                one.
            rng: the generator for the random models.

        Raises:
            ValueError: if the safety factor is below one.
        """
        super().__init__(certified=False)
        if safety < 1.0:
            raise ValueError("The safety factor must be at least one.")
        self._forward = forward
        self._samples = int(samples)
        self._safety = float(safety)
        self._rng = resolve_rng(rng)

    def remainder_set(self, cell: Cell, /) -> Zonotope:
        """The widened box of the remainder's coefficients over the sampled models.

        Raises:
            ValueError: if the cell has no slabs to sample along.
        """
        count = len(cell.directions)
        if count == 0:
            raise ValueError(
                "A sampled remainder needs a cell with slabs to sample along."
            )
        data_space = self._forward.codomain
        linearization = self._forward.at(cell.center)
        signs = [np.eye(count)[j] * sign for j in range(count) for sign in (-1.0, 1.0)]
        if count <= 6:
            signs += [
                np.array([1.0 if (index >> j) & 1 else -1.0 for j in range(count)])
                for index in range(2**count)
            ]
        signs += list(self._rng.uniform(-1.0, 1.0, size=(self._samples, count)))
        coefficients = []
        for sign in signs:
            model = cell.point(sign * cell.half_widths)
            coefficients.append(
                data_space.to_components(
                    _remainder(self._forward, linearization, model)
                )
            )
        coefficients = np.array(coefficients)
        return Zonotope(
            data_space,
            [data_space.basis_vector(i) for i in range(data_space.dim)],
            lower=self._safety * np.minimum(coefficients.min(axis=0), 0.0),
            upper=self._safety * np.maximum(coefficients.max(axis=0), 0.0),
        )


def _remainder(forward: Operator, linearization: Linearization, model: Any) -> Any:
    """``F(m) − F(m_K) − F'(m_K)(m − m_K)`` at a model."""
    space, data_space = forward.domain, forward.codomain
    step = linearization.derivative(space.subtract(model, linearization.point))
    return data_space.subtract(
        data_space.subtract(forward(model), linearization.value), step
    )


def _product_set(space: HilbertSpace, parts: Sequence[ConvexSet]) -> ConvexSet:
    """The product of convex sets, one per summand of a direct sum."""
    supports = [part.support_function() for part in parts]
    return ConvexSet.from_support_function(
        space,
        lambda y: float(sum(h(block) for h, block in zip(supports, y))),
        maximizer=lambda y: tuple(
            part.support_maximizer(block) for part, block in zip(parts, y)
        ),
        membership=lambda x, rtol: all(
            part.contains(block, rtol=rtol) for part, block in zip(parts, x)
        ),
    )


def _grown(noise: Ball, remainder: ConvexSet) -> ConvexSet:
    """A noise ball plus a remainder set: every error plus every remainder."""
    space = noise.domain
    if isinstance(remainder, Ball):
        return Ball(
            space,
            radius=noise.radius + remainder.radius,
            center=space.add(noise.center, remainder.center),
        )
    if not remainder.has_projection:
        raise TypeError(
            "A remainder set must be a ball or carry a projection, which is "
            "how membership of its sum with the noise ball is decided."
        )
    ball_support, set_support = noise.support_function(), remainder.support_function()

    def membership(x: Any, rtol: float) -> bool:
        offset = space.subtract(x, noise.center)
        distance = space.norm(space.subtract(offset, remainder.project(offset)))
        return bool(distance <= noise.radius * (1.0 + rtol))

    return ConvexSet.from_support_function(
        space,
        lambda y: float(ball_support(y)) + float(set_support(y)),
        maximizer=lambda y: space.add(
            noise.support_maximizer(y), remainder.support_maximizer(y)
        ),
        membership=membership,
    )


def _shrunk(noise: Ball, remainder: ConvexSet, margin: float) -> Ball | None:
    """A ball of errors that stay in the noise ball whatever remainder is added, or None.

    The remainder set lies in a ball of radius ``rho`` about ``z``; an error
    within ``radius − rho`` of ``center + z`` then stays within ``radius``
    of the center when any remainder is taken from it. A further fraction
    ``margin`` of the radius is given up, so that a model on the boundary
    to a solver's tolerance is inside.
    """
    space = noise.domain
    bound = remainder if isinstance(remainder, Ball) else remainder.bounding_ball()
    radius = noise.radius * (1.0 - margin) - bound.radius
    if radius <= 0.0:
        return None
    return Ball(space, radius=radius, center=space.add(noise.center, bound.center))


def _slab_constraints(cell: Cell, margin: float) -> tuple[Functional, ...]:
    """Each slab of a cell, shrunk by a margin, as a functional held at or below zero.

    ``((u, m − center) / w)^2 − 1 + margin``: smooth, convex, and scaled so
    that the margin is the same fraction of the level as the two sets' is.
    """
    space, center = cell.domain, cell.center

    def build(direction: Any, half_width: float) -> Functional:
        offset = float(space.inner_product(direction, center))
        scale = 1.0 / half_width**2

        def gap(model: Any) -> float:
            return float(space.inner_product(direction, model)) - offset

        return Functional.from_callables(
            space,
            lambda model: scale * gap(model) ** 2 - 1.0 + margin,
            gradient=lambda model: space.scale(2.0 * scale * gap(model), direction),
            hessian=lambda model: LinearOperator.self_adjoint(
                space,
                lambda v: space.scale(
                    2.0 * scale * float(space.inner_product(direction, v)), direction
                ),
            ),
        )

    return tuple(
        build(direction, float(half_width))
        for direction, half_width in zip(cell.directions, cell.half_widths)
    )


@dataclass(frozen=True)
class CellSupportResult(SupportResult):
    """The support of a nonlinear feasible set's part in one cell, from both sides.

    ``value`` bounds the property, along the direction, of every feasible
    model of the cell, given the remainder claim, and is minus infinity for
    a cell shown to hold none. ``inner`` is attained by ``extremal_model``,
    a model of the cell checked to lie in the prior and fit the data, and
    is minus infinity when the cell exhibited none.
    """

    certified: bool = True
    """Whether the remainder claim ``value`` rests on was certified. An
    estimated claim leaves ``value`` an estimate, however well the
    optimization behind it converged."""
    refuted: bool = False
    """Whether a model evaluated in the cell had a remainder outside the
    claimed set. The claim is then false and ``value`` bounds nothing."""

    def __repr__(self) -> str:
        return (
            f"CellSupportResult(value={self.value:.6g}, inner={self.inner:.6g}, "
            f"certified={self.certified}, refuted={self.refuted}, "
            f"converged={self.converged})"
        )


# --------------------------------------------------------------------- #
#                             Local search                               #
# --------------------------------------------------------------------- #


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
        *,
        cell: Cell | None = None,
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
            cell: a cell to keep the search in, each slab a further
                constraint, shrunk by the margin as the two sets are. The
                answer is then a model of the cell, and the linearized
                first step is not taken, since it knows nothing of the
                cell.

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
        constraints = feasible._constraints(self._margin)
        stepped = None
        if cell is None:
            stepped = feasible._linearized_step(direction, start)
        else:
            constraints += _slab_constraints(cell, self._margin)
        result = self._method(inequalities=constraints).minimize(
            objective, start if stepped is None else stepped
        )

        def along(model: Any) -> float:
            return float(space.inner_product(pulled, model))

        def checked(model: Any) -> bool:
            if cell is not None and not cell.contains(model, rtol=0.0):
                return False
            return feasible.is_feasible(model)

        best, inner = None, float("-inf")
        for candidate in (start, stepped, result.minimizer):
            if candidate is None:
                continue
            if checked(candidate) and along(candidate) > inner:
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
                for multiplier in result.inequality_multipliers[:2]
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

    # ----------------------------------------------------------------- #
    #                          The set on a cell                         #
    # ----------------------------------------------------------------- #

    def cell(
        self,
        center: Any,
        /,
        *,
        directions: Sequence[Any] = (),
        half_widths: Sequence[float] = (),
        radius: float | None = None,
    ) -> Cell:
        """A cell about a model, its radius read from the prior where the prior has one.

        Args:
            center: the model the cell is about.
            directions: the vectors the slabs are taken along.
            half_widths: the half-width of each slab.
            radius: a bound on the distance of a model of the cell from the
                center, if one is known. A ball prior gives its own, the
                farthest any of its models can be, and the smaller is kept.

        Returns:
            The cell.
        """
        space = self.model_space
        bound = float("inf") if radius is None else float(radius)
        if isinstance(self._prior, Ball):
            offset = space.norm(space.subtract(center, self._prior.center))
            bound = min(bound, self._prior.radius + offset)
        return Cell(
            space,
            center,
            directions=directions,
            half_widths=half_widths,
            radius=bound,
        )

    def _noise_ball(self) -> Ball:
        if not isinstance(self._noise, Ball):
            raise TypeError(
                "The sets on a cell are built for a noise set that is a ball; "
                f"this one is a {type(self._noise).__name__}."
            )
        return self._noise

    def _cell_set(
        self, cell: Cell, linearization: Linearization, noise: ConvexSet
    ) -> FeasiblePropertySet:
        """The linearized problem on a cell, with a given noise set."""
        data_space = self.data_space
        derivative = linearization.derivative
        shifted = data_space.add(
            data_space.subtract(self._data, linearization.value),
            derivative(linearization.point),
        )
        if not cell.directions:
            structure = FeasibleStructure(
                derivative,
                self._property_operator,
                self._prior,
                noise=noise,
                solver=self._solver,
            )
            return FeasiblePropertySet.from_structure(structure, shifted)
        # The slabs enter as data: the rows (u_j, m) observed at the
        # center's values to within the half-widths.
        rows = cell.row_operator()
        forward = ColumnLinearOperator([derivative, rows])
        count = len(cell.directions)
        slabs = Zonotope(
            rows.codomain,
            [rows.codomain.basis_vector(j) for j in range(count)],
            lower=-cell.half_widths,
            upper=cell.half_widths,
        )
        structure = FeasibleStructure(
            forward,
            self._property_operator,
            self._prior,
            noise=_product_set(forward.codomain, (noise, slabs)),
            solver=self._solver,
        )
        return FeasiblePropertySet.from_structure(
            structure, (shifted, rows(cell.center))
        )

    def outer_set(
        self, cell: Cell, remainder: RemainderBound, /
    ) -> FeasiblePropertySet:
        """A convex set holding the property of every feasible model of a cell.

        The feasible property set of the problem linearized about the
        cell's center, on the prior cut by the cell's slabs, with the noise
        set grown by the remainder set: a feasible model of the cell has a
        linearized residual that is an error plus a remainder. The set is
        as good as the remainder claim; with a certified claim it contains
        the property of every feasible model of the cell, and an empty one
        shows the cell holds none.

        Args:
            cell: the cell.
            remainder: the claim about the forward operator over the cell.

        Returns:
            The set, with every probe of a linear feasible property set.

        Raises:
            TypeError: if the noise set is not a ball.
        """
        noise = _grown(self._noise_ball(), remainder.remainder_set(cell))
        return self._cell_set(cell, self._forward.at(cell.center), noise)

    def inner_set(
        self,
        cell: Cell,
        remainder: RemainderBound,
        /,
        *,
        margin: float = 1e-5,
    ) -> FeasiblePropertySet | None:
        """A convex set of property values that feasible models of a cell attain.

        The linearized problem on the cell with the noise set shrunk by the
        remainder set: a model whose linearized residual stays in the noise
        set whatever remainder is taken from it fits the data through the
        real operator. With a true remainder claim every model of this set
        is feasible, so the set lies inside the nonlinear one; with a false
        one it need not, and a model taken from it is still to be checked.

        Args:
            cell: the cell.
            remainder: the claim about the forward operator over the cell.
            margin: a further fraction of the noise radius given up, so
                that a model on the set's boundary to a solver's tolerance
                is inside.

        Returns:
            The set, or None when the remainder set leaves nothing of the
            noise set: the cell is too large for the claim to say anything
            from inside.

        Raises:
            TypeError: if the noise set is not a ball.
        """
        noise = _shrunk(self._noise_ball(), remainder.remainder_set(cell), margin)
        if noise is None:
            return None
        return self._cell_set(cell, self._forward.at(cell.center), noise)

    def cell_support_result(
        self,
        direction: Any,
        cell: Cell,
        remainder: RemainderBound,
        /,
        *,
        via: LocalSearch | None = None,
    ) -> CellSupportResult:
        """The support, in a direction, of the feasible set's part in one cell, from both sides.

        The outer value is the support of :meth:`outer_set`. The inner value
        is attained by a checked model of the cell: the answer of a local
        search kept in the cell, begun from the outer set's extremal model,
        or that model itself where it happens to fit. Every model evaluated
        in the cell also tests the remainder claim: a remainder outside the
        claimed set marks the result refuted.

        Args:
            direction: the direction in the property space.
            cell: the cell.
            remainder: the claim about the forward operator over the cell.
            via: the local search; the default :class:`LocalSearch` if
                omitted.

        Returns:
            The result; ``value`` is minus infinity when the outer set is
            empty.

        Raises:
            TypeError: if the noise set is not a ball.
        """
        space, data_space = self.model_space, self.data_space
        ball = self._noise_ball()
        linearization = self._forward.at(cell.center)
        claimed = remainder.remainder_set(cell)
        pulled = self._property_operator.adjoint(direction)

        try:
            outer = self._cell_set(
                cell, linearization, _grown(ball, claimed)
            ).support_result(direction)
        except ValueError as error:
            if "empty" not in str(error):
                raise
            outer = None
        if outer is None:
            return CellSupportResult(
                value=float("-inf"),
                inner=float("-inf"),
                converged=True,
                algorithm="cell",
                evaluations=0,
                certified=remainder.certified,
                message="no model of the cell fits the data, given the remainder claim",
            )
        start = outer.extremal_model
        if start is None or not cell.contains(start):
            start = cell.center
        searched = (via or LocalSearch()).support(self, direction, start, cell=cell)
        best, attained, refuted = None, float("-inf"), False
        for result in (outer, searched):
            model = result.extremal_model
            if model is None:
                continue
            if not (
                cell.contains(model, rtol=0.0) and self._prior.contains(model, rtol=0.0)
            ):
                continue
            predicted = self._forward(model)
            step = linearization.derivative(space.subtract(model, cell.center))
            departure = data_space.subtract(
                data_space.subtract(predicted, linearization.value), step
            )
            # Outside the claimed set by more than the rounding of the
            # three terms the remainder is the difference of.
            excess = data_space.norm(
                data_space.subtract(departure, claimed.project(departure))
            )
            scale = max(
                data_space.norm(predicted),
                data_space.norm(linearization.value),
                data_space.norm(step),
            )
            if excess > 1e-9 * scale:
                refuted = True
            fits = self._noise.contains(
                data_space.subtract(self._data, predicted), rtol=0.0
            )
            along = float(space.inner_product(pulled, model))
            if fits and along > attained:
                best, attained = model, along
        message = outer.message
        if refuted:
            message += "; a model of the cell has a remainder outside the claimed set"
        return CellSupportResult(
            value=float(outer.value),
            inner=attained,
            converged=bool(outer.converged),
            algorithm="cell",
            evaluations=int(outer.evaluations) + int(searched.evaluations),
            certificate=outer.certificate,
            extremal_model=best,
            message=message,
            certified=remainder.certified,
            refuted=refuted,
        )

    def __repr__(self) -> str:
        return (
            f"NonlinearFeasiblePropertySet(dim={self.domain.dim}, "
            f"prior={self._prior!r}, noise={self._noise!r})"
        )
