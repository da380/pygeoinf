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
import heapq
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from numpy.random import Generator

from ..algebra.operators import (
    AffineOperator,
    EvaluationError,
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
    Dual,
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
    "BranchAndPrune",
    "BranchInclusionResult",
    "BranchSupportResult",
    "Cell",
    "CellRecord",
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
                the center; infinite if nothing is known. Slabs along a
                basis of the whole space give a bound of their own, and the
                smaller is kept.

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
            if len(self._directions) == domain.dim:
                # Slabs along a basis bound the cell on their own: the
                # squared distance is the offsets' quadratic form in the
                # inverse Gram matrix.
                reach = np.sqrt(
                    np.linalg.eigvalsh(np.linalg.inv(self._gram)).max()
                ) * np.linalg.norm(self._half_widths)
                self._radius = min(self._radius, float(reach))
        self._orthonormal = bool(
            self._directions
            and np.abs(self._gram - np.eye(len(self._directions))).max() <= 1e-10
        )

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

        See :meth:`bounds_along`, which this is for one vector. Exact for a
        slab direction itself.

        Args:
            vector: a vector of the model space.
        """
        space = self._domain
        pairings = np.array([space.inner_product(u, vector) for u in self._directions])
        size = np.array([float(space.norm(vector))])
        return float(self.bounds_along(pairings[None, :], size)[0])

    def bounds_along(self, pairings: np.ndarray, sizes: np.ndarray, /) -> np.ndarray:
        """Bounds on ``|(v, m − center)|`` over the cell, for several vectors at once.

        A vector enters through its inner products with the slab directions
        and its norm, which is all the bound reads. The part of ``v`` in
        the span of the slab directions, ``sum_j a_j u_j``, is bounded by
        ``sum_j |a_j| w_j``, and the rest by its norm times the radius; the
        whole by ``|v|`` times the radius; and the smaller is taken. Where
        the directions are orthonormal the bound is sharpened to the exact
        supremum over the slabs cut by the ball of the cell's radius,
        ``min over d of sum_j w_j (|a_j| − d_j) + radius |(rest, d)|``, in
        which a slab wider than the ball allows simply drops out.

        Args:
            pairings: one row per vector, its inner products ``(u_j, v)``
                with the slab directions.
            sizes: the vectors' norms.

        Returns:
            One bound per vector; infinite where the cell has no finite
            radius and the vector is not in the span of the slab directions.
        """
        sizes = np.asarray(sizes, dtype=float)
        radius = self._radius
        with np.errstate(invalid="ignore"):
            whole = np.where(sizes > 0.0, sizes * radius, 0.0)
        if not self._directions:
            return whole
        pairings = np.asarray(pairings, dtype=float).reshape(len(sizes), -1)
        weights = np.linalg.solve(self._gram, pairings.T).T
        # What is left of |v|^2 outside the span, to the rounding of the
        # subtraction; below that the vector is in the span.
        rest = np.maximum(sizes**2 - np.einsum("ij,ij->i", pairings, weights), 0.0)
        rest = np.where(rest <= 1e-12 * sizes**2, 0.0, np.sqrt(rest))
        split = np.abs(weights) @ self._half_widths
        with np.errstate(invalid="ignore"):
            split = split + np.where(rest > 0.0, rest * radius, 0.0)
        bound = np.minimum(split, whole)
        if self._orthonormal and np.isfinite(radius):
            # The supremum over the slabs cut by the ball, by its dual: any
            # d between zero and |a| bounds, and the fixed point is exact.
            along = np.abs(weights)
            level = sizes / radius
            for _ in range(60):
                kept = np.minimum(level[:, None] * self._half_widths[None, :], along)
                updated = np.sqrt(rest**2 + (kept**2).sum(axis=1)) / radius
                settled = np.abs(updated - level).max() <= 1e-12 * max(
                    level.max(), 1e-300
                )
                level = updated
                if settled:
                    break
            kept = np.minimum(level[:, None] * self._half_widths[None, :], along)
            cut = (along - kept) @ self._half_widths + radius * np.sqrt(
                rest**2 + (kept**2).sum(axis=1)
            )
            bound = np.minimum(bound, cut)
        return bound

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

    def slab_directions(self) -> tuple[Any, ...] | None:
        """The model-space directions the forward operator curves along, where the bound knows them.

        What a search over cells cuts its slabs along. None for a bound
        that reads only a cell's radius, or has no such directions to name.
        """
        return None

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
        self,
        data_space: HilbertSpace,
        constant: float | Callable[[Cell], float],
        /,
        *,
        certified: bool = True,
    ) -> None:
        """
        Args:
            data_space: the forward operator's codomain.
            constant: ``L``, a bound on the second derivative's norm as a
                bilinear map: a number that holds over every cell the bound
                is asked about, or a callable of the cell returning one
                that holds over that cell.
            certified: whether ``L`` is proved for the operator.

        Raises:
            ValueError: if a constant given as a number is negative.
        """
        super().__init__(certified=certified)
        if not callable(constant) and constant < 0.0:
            raise ValueError("The constant must not be negative.")
        self._data_space = data_space
        self._constant = constant

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
        constant = self._constant
        if callable(constant):
            constant = float(constant(cell))
        return Ball(self._data_space, radius=0.5 * constant * cell.radius**2)


class ComponentwiseRemainder(RemainderBound):
    """The remainder of a scalar function applied to linear functionals of the model.

    A forward operator ``F(m) = W g(B m)``: linear functionals
    ``t_q = (b_q, m)`` of the model, a scalar function ``g_q`` applied to
    each, and optionally a linear map ``W`` of the results into the data
    space; without ``W`` the results are the data, over the data space's
    basis. Point values squared are of this form, and so is a line integral
    of a function of a field, once its quadrature is written out: the
    ``b_q`` are evaluations at the nodes and ``W`` holds the weights.

    The remainder of ``g_q`` is ``g_q''(s) (b_q, m − m_K)^2 / 2`` at some
    ``s`` between the two values of ``t_q``. With ``w_q`` the cell's
    half-width along ``b_q`` and ``[lo_q, hi_q]`` a range of ``g_q''`` over
    ``[t_q − w_q, t_q + w_q]``, it lies between ``min(lo_q, 0) w_q^2 / 2``
    and ``max(hi_q, 0) w_q^2 / 2``: one-sided for a second derivative of one
    sign, and nothing at all for a linear ``g_q``. The remainder set is the
    box of those ranges over the data space's basis, taken through ``W``
    where there is one by the range of each row of its matrix, which
    contains the image of the box and is that image when no two data share
    a functional.
    """

    def __init__(
        self,
        data_space: CoordinateSpace,
        rows: Sequence[Any] | LinearOperator,
        curvature: Callable[[np.ndarray, np.ndarray], tuple[Any, Any]],
        /,
        *,
        then: LinearOperator | None = None,
        certified: bool = True,
    ) -> None:
        """
        Args:
            data_space: the forward operator's codomain, whose basis the
                remainder's components are taken in.
            rows: the functionals ``b_q``: the model-space vectors
                themselves, or the operator ``B`` into a Euclidean space
                whose components they give.
            curvature: called with the arrays ``t − w`` and ``t + w`` and
                returning two arrays, a lower and an upper bound on each
                ``g_q''`` over its interval.
            then: ``W``, from the Euclidean space of the functionals'
                values into the data space; omitted when the values are
                the data.
            certified: whether the ranges are proved.

        Raises:
            ValueError: if the rows do not match the space they feed, the
                data space or the domain of ``then``, in number.
            TypeError: if ``rows`` is an operator into a space that is not
                Euclidean.
        """
        super().__init__(certified=certified)
        self._data_space = data_space
        self._curvature = curvature
        self._then = then
        if isinstance(rows, LinearOperator):
            if not isinstance(rows.codomain, EuclideanSpace):
                raise TypeError(
                    "An operator giving the functionals must map into a "
                    "Euclidean space, whose components are their values."
                )
            self._operator: LinearOperator | None = rows
            self._rows: tuple[Any, ...] | None = None
            count = rows.codomain.dim
        else:
            self._operator = None
            self._rows = tuple(rows)
            count = len(self._rows)
        fed = data_space.dim if then is None else then.domain.dim
        if count != fed:
            raise ValueError(
                f"There are {count} functionals for a space of dimension {fed}."
            )
        self._count = count
        self._sizes: np.ndarray | None = None
        self._paired: tuple[Any, np.ndarray] | None = None
        self._matrix: np.ndarray | None = None

    def _materialized(self) -> tuple[Any, ...]:
        if self._rows is None:
            space = self._operator.codomain
            self._rows = tuple(
                self._operator.adjoint(space.basis_vector(q))
                for q in range(self._count)
            )
        return self._rows

    def slab_directions(self) -> tuple[Any, ...]:
        """The functionals ``b_q``: the only directions the forward operator sees the model along."""
        return self._materialized()

    def remainder_set(self, cell: Cell, /) -> Zonotope:
        """The box of coefficient ranges over the data space's basis.

        Raises:
            ValueError: if the cell is unbounded along a functional, having
                neither a finite radius nor that functional within its
                slabs' span.
        """
        space, data_space = cell.domain, self._data_space
        if self._operator is not None:
            at_center = np.asarray(self._operator(cell.center), dtype=float)
        else:
            at_center = np.array(
                [space.inner_product(row, cell.center) for row in self._rows]
            )
        if self._sizes is None:
            self._sizes = np.array([space.norm(row) for row in self._materialized()])
        # Cells of one search share their directions, so the functionals'
        # inner products with them are worked out once.
        if self._paired is None or self._paired[0] is not cell.directions:
            if self._operator is not None:
                columns = [
                    np.asarray(self._operator(direction), dtype=float)
                    for direction in cell.directions
                ]
                paired = np.array(columns).T.reshape(self._count, len(cell.directions))
            else:
                paired = np.array(
                    [
                        [
                            space.inner_product(direction, row)
                            for direction in cell.directions
                        ]
                        for row in self._rows
                    ]
                ).reshape(self._count, len(cell.directions))
            self._paired = (cell.directions, paired)
        widths = cell.bounds_along(self._paired[1], self._sizes)
        if not np.all(np.isfinite(widths)):
            raise ValueError(
                "The cell is unbounded along a functional; give it a radius "
                "or a slab along every functional the forward operator "
                "curves in."
            )
        low, high = self._curvature(at_center - widths, at_center + widths)
        low = 0.5 * np.minimum(np.asarray(low, dtype=float), 0.0) * widths**2
        high = 0.5 * np.maximum(np.asarray(high, dtype=float), 0.0) * widths**2
        if self._then is not None:
            if self._matrix is None:
                self._matrix = np.asarray(self._then.matrix(form="components"))
            positive = np.maximum(self._matrix, 0.0)
            negative = np.minimum(self._matrix, 0.0)
            low, high = (
                positive @ low + negative @ high,
                positive @ high + negative @ low,
            )
        return Zonotope(
            data_space,
            [data_space.basis_vector(i) for i in range(data_space.dim)],
            lower=low,
            upper=high,
        )


class SampledRemainder(RemainderBound):
    """The remainder estimated from evaluations of the forward operator: not certified.

    The remainder is computed at models of the cell and the estimate is the
    box, over the data space's basis, of the coefficients seen, widened by
    a safety factor and taken to include the origin. Nothing proves the box
    holds the remainder elsewhere in the cell, so a bound built on this is
    an estimate and says so.

    Along the slabs the models are at the ends of each slab, at the corners
    where the slabs are few, and at random points between. Across the
    slabs a cell is as wide as the constraint set, and there the models
    move by perturbations the caller supplies through ``across``, a draw
    from a prior measure say, taken with their part along the slabs removed
    and scaled out to the cell's radius: without them nothing is learned of
    how the operator curves in the directions the slabs do not span, and
    the estimate is of a cell that is thin in every direction.

    The same perturbations and the same points of the slabs are used for
    every cell, so the estimate is a fixed function of the cell, and two
    cells can be compared by it.
    """

    def __init__(
        self,
        forward: Operator,
        /,
        *,
        samples: int = 32,
        safety: float = 2.0,
        rng: Generator | None = None,
        across: Callable[[], Any] | None = None,
        within: Subset | None = None,
        shape: str = "box",
    ) -> None:
        """
        Args:
            forward: the forward operator, into a space with coordinates.
            samples: how many random models to evaluate beyond the ends of
                the slabs and their corners.
            safety: the factor the observed ranges are widened by; at least
                one.
            shape: ``"box"`` for the box of the coefficients seen over the
                data space's basis, or ``"ball"`` for the ball about the
                origin that holds the remainders seen. The box keeps a
                remainder's sign, and is the tighter where each datum's
                remainder has one; with many data and no such structure its
                corners lie far beyond any remainder seen, and the ball is
                the tighter by far.
            rng: the generator for the random points of the slabs.
            across: called with no arguments and returning a model-space
                perturbation; ``samples`` of them are drawn, once. Each
                sampled model is then also moved across the slabs by one
                of them.
            within: a set with a projection, the constraint set, that each
                sampled model is brought into, so that the models sampled
                are models the cell and the set both allow.

        Raises:
            ValueError: if the safety factor is below one, or the shape is
                neither of the two.
        """
        super().__init__(certified=False)
        if safety < 1.0:
            raise ValueError("The safety factor must be at least one.")
        if shape not in ("box", "ball"):
            raise ValueError(f'The shape is "box" or "ball", got {shape!r}.')
        self._forward = forward
        self._samples = int(samples)
        self._safety = float(safety)
        self._rng = resolve_rng(rng)
        self._within = within
        self._shape = shape
        self._across = (
            None if across is None else [across() for _ in range(self._samples)]
        )
        self._patterns: dict[int, np.ndarray] = {}

    def _slab_points(self, count: int) -> np.ndarray:
        """Where along the slabs to sample, as multiples of the half-widths; fixed per slab count."""
        if count not in self._patterns:
            points = [np.zeros(count)]
            points += [
                np.eye(count)[j] * sign for j in range(count) for sign in (-1.0, 1.0)
            ]
            if 0 < count <= 6:
                points += [
                    np.array([1.0 if (index >> j) & 1 else -1.0 for j in range(count)])
                    for index in range(2**count)
                ]
            points += list(self._rng.uniform(-1.0, 1.0, size=(self._samples, count)))
            self._patterns[count] = np.array(points).reshape(len(points), count)
        return self._patterns[count]

    def remainder_set(self, cell: Cell, /) -> ConvexSet:
        """The widened box, or ball, of the remainders at the sampled models.

        A model the forward operator refuses, by raising
        :class:`~pygeoinf.algebra.operators.EvaluationError`, is passed
        over: it is no model of the feasible set.

        Raises:
            ValueError: if the cell has neither slabs to sample along nor
                perturbations to sample across them with, has no finite
                radius to scale those perturbations to, or holds no
                sampled model the forward operator evaluates.
        """
        count = len(cell.directions)
        if count == 0 and self._across is None:
            raise ValueError(
                "A sampled remainder needs a cell with slabs to sample along, "
                "or perturbations to sample across them with."
            )
        if self._across is not None and not np.isfinite(cell.radius):
            raise ValueError(
                "Sampling across the slabs needs a cell of finite radius to "
                "scale the perturbations to."
            )
        space, data_space = self._forward.domain, self._forward.codomain
        linearization = self._forward.at(cell.center)
        coefficients, sizes = [], []
        for index, multiples in enumerate(self._slab_points(count)):
            model = cell.point(multiples * cell.half_widths)
            if self._across is not None:
                step = self._across[index % len(self._across)]
                # Its part along the slabs removed, and out to the radius.
                step = space.subtract(
                    step,
                    space.subtract(
                        cell.point(cell.offsets(space.add(cell.center, step))),
                        cell.center,
                    ),
                )
                along = space.norm(space.subtract(model, cell.center))
                room = max(cell.radius**2 - along**2, 0.0) ** 0.5
                size = space.norm(step)
                if size > 0.0 and room > 0.0:
                    model = space.axpy(room / size, step, model)
            if self._within is not None:
                model = self._within.project(model)
            try:
                departure = _remainder(self._forward, linearization, model)
            except EvaluationError:
                # Where the operator cannot evaluate there is no model to
                # fit the data, and nothing for the remainder to bound.
                continue
            coefficients.append(data_space.to_components(departure))
            sizes.append(float(data_space.norm(departure)))
        if not coefficients:
            raise ValueError(
                "The forward operator evaluated at none of the sampled models."
            )
        if self._shape == "ball":
            return Ball(data_space, radius=self._safety * max(sizes))
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


def _departs(
    forward: Operator,
    linearization: Linearization,
    claimed: ConvexSet,
    model: Any,
    predicted: Any,
) -> bool:
    """Whether a model's remainder lies outside a claimed set.

    Outside by more than the rounding of the three terms the remainder is
    the difference of: ``F(m)``, ``F(m_K)`` and ``F'(m_K)(m − m_K)``.
    """
    space, data_space = forward.domain, forward.codomain
    step = linearization.derivative(space.subtract(model, linearization.point))
    departure = data_space.subtract(
        data_space.subtract(predicted, linearization.value), step
    )
    excess = data_space.norm(data_space.subtract(departure, claimed.project(departure)))
    scale = max(
        data_space.norm(predicted),
        data_space.norm(linearization.value),
        data_space.norm(step),
    )
    return bool(excess > 1e-9 * scale)


def _independent(space: HilbertSpace, vectors: Sequence[Any]) -> list[Any]:
    """A largest independent subset of some vectors, the most significant first.

    Greedy, by what is left of each vector's squared norm once the chosen
    ones are projected out; a vector within ``1e-8`` of the span of those
    already chosen, relative to its own norm, is left out.
    """
    vectors = list(vectors)
    gram = np.array([[space.inner_product(a, b) for b in vectors] for a in vectors])
    sizes = np.diag(gram).copy()
    left = sizes.copy()
    chosen: list[int] = []
    factor = np.zeros((0, len(vectors)))
    for _ in range(len(vectors)):
        relative = np.where(sizes > 0.0, left / np.where(sizes > 0.0, sizes, 1.0), 0.0)
        relative[chosen] = 0.0
        candidates = np.where(relative > 1e-16)[0]
        if candidates.size == 0:
            break
        pick = int(candidates[np.argmax(left[candidates])])
        row = (gram[pick] - factor[:, pick] @ factor) / np.sqrt(left[pick])
        factor = np.vstack([factor, row])
        left = np.maximum(left - row**2, 0.0)
        chosen.append(pick)
    return [vectors[index] for index in chosen]


def _extent(subset: ConvexSet) -> float:
    """The size of a remainder set: its radius, as a ball or about its midpoint."""
    return float(subset.radius)


@dataclass(frozen=True)
class CellRecord:
    """One cell a search over cells examined, and what became of it."""

    cell: Cell
    value: float
    """The outer value of the feasible set's part in the cell, given the
    remainder claim; minus infinity for a cell shown to hold no model that
    fits."""
    status: str
    """``"empty"``: shown to hold no feasible model. ``"pruned"``: its outer
    value is no larger than a value already attained. ``"split"``: replaced
    by its two halves. ``"live"``: still standing when the search stopped,
    so part of the outer description."""


@dataclass(frozen=True)
class BranchSupportResult(CellSupportResult):
    """The support of a nonlinear feasible set from both sides, by a search over cells.

    ``value`` is the largest outer value over the cells that were neither
    shown empty nor split, and bounds the support of the whole set, given
    the remainder claim; ``inner`` is attained by ``extremal_model``, a
    checked model. ``converged`` says the two met to the tolerance asked;
    a search that stopped at its budget returns the bracket it had.
    """

    cells: tuple[CellRecord, ...] = ()
    """Every cell examined, in the order it was bounded."""
    history: tuple[tuple[int, float, float], ...] = ()
    """``(cells examined, outer value, inner value)`` after each split."""
    searches: int = 0
    """How many local searches were run for the inner value."""

    @property
    def live(self) -> tuple[CellRecord, ...]:
        """The cells still standing: together they hold every feasible model the search could not rule out."""
        return tuple(record for record in self.cells if record.status == "live")

    def __repr__(self) -> str:
        return (
            f"BranchSupportResult(value={self.value:.6g}, inner={self.inner:.6g}, "
            f"cells={len(self.cells)}, certified={self.certified}, "
            f"refuted={self.refuted}, converged={self.converged})"
        )


@dataclass(frozen=True)
class BranchInclusionResult(InclusionResult):
    """Whether a value is in a nonlinear feasible set, decided by a search over cells where it can be.

    Three outcomes. A ``fitting_model`` with a ``level`` at or below the
    prior's own proves the value **inside**: the model fits the data, lies
    in the prior and has the property. A ``lower_bound`` of plus infinity
    proves it **outside**, given the remainder claim: every cell of a cover
    of the prior was shown to hold no model with the property that fits.
    Otherwise the search stopped at its budget and nothing is decided, with
    ``level`` infinite and ``lower_bound`` minus infinity.
    """

    certified: bool = True
    """Whether the remainder claim an exclusion rests on was certified."""
    cells: tuple[CellRecord, ...] = ()
    """Every cell examined. ``"empty"`` here means shown to hold no fitting
    model with the property; a record's ``value`` is not used."""
    searches: int = 0
    """How many local searches were run for a model with the property."""

    def __repr__(self) -> str:
        if self.fitting_model is not None:
            verdict = "inside"
        elif self.lower_bound == float("inf"):
            verdict = "outside"
        else:
            verdict = "undecided"
        return (
            f"BranchInclusionResult({verdict}, cells={len(self.cells)}, "
            f"certified={self.certified})"
        )


class BranchAndPrune:
    """The support of a nonlinear feasible set from both sides, by splitting cells.

    The constraint set is covered by cells, each the set cut by slabs along
    the directions the forward operator curves in. A cell's outer value is
    the support of the linearized problem on it with the noise set grown by
    the cell's remainder set, so the largest over a cover bounds the whole
    set; a checked model gives a value from inside. The search starts from
    one cell as wide as the constraint set and repeatedly takes the cell
    with the largest outer value, runs a local search kept in it unless it
    already holds the best model found, and splits it in two along the slab
    whose halving shrinks its remainder set most. A cell is dropped when the linearized problem on it has no
    feasible model, or when its outer value is no larger than a value
    already attained. It stops when the largest outer value is within the
    tolerance of the best attained one, or at its budget of cells, and
    returns the bracket either way.

    The outer value is as good as the remainder claim, and the result says
    whether that was certified. A checked model beyond the outer value, or
    one whose remainder lies outside its cell's claimed set, marks the
    result refuted.

    The number of cells needed grows exponentially with the number of
    directions that are both curved enough to need narrow slabs and left
    wide by the data, so the method is for problems where those are few.
    """

    name = "branch"

    def __init__(
        self,
        remainder: RemainderBound,
        /,
        *,
        directions: Sequence[Any] | None = None,
        rtol: float = 1e-3,
        max_cells: int = 200,
        tighten: int = 0,
        search: "LocalSearch | None" = None,
    ) -> None:
        """
        Args:
            remainder: the claim about the forward operator over a cell.
            directions: the model-space directions to cut slabs along; the
                remainder bound's own, where it names them, if omitted. A
                largest independent subset is used, and a direction whose
                slab the remainder set does not depend on is left out.
            rtol: the gap between the outer and the inner value at which
                the search stops, relative to the constraint set's width
                along the direction.
            max_cells: the budget of cells to bound.
            tighten: how many sweeps of
                :meth:`NonlinearFeasiblePropertySet.tightened_cell` to give
                each cell before it is bounded: its slabs cut back to what
                its feasible models can reach, at two support solves per
                slab and sweep. None by default. It is what makes a
                problem with many slab directions tractable, where the
                data pin most of them and halving would take a cell for
                each.
            search: the local search run in a cell for the inner value; the
                default :class:`LocalSearch` if omitted.

        Raises:
            ValueError: if no directions are given and the remainder bound
                names none.
        """
        if directions is None:
            directions = remainder.slab_directions()
        if directions is None:
            raise ValueError(
                "The remainder bound names no directions to cut slabs along; "
                "pass directions=."
            )
        self._remainder = remainder
        self._directions = tuple(directions)
        self._rtol = float(rtol)
        self._max_cells = int(max_cells)
        self._tighten = int(tighten)
        self._search = search

    @property
    def remainder(self) -> RemainderBound:
        """The claim about the forward operator over a cell."""
        return self._remainder

    def root(self, feasible: "NonlinearFeasiblePropertySet", /) -> Cell:
        """The first cell: the constraint set, with a slab as wide as its reach along each direction.

        Args:
            feasible: the set.

        Returns:
            The cell, about the constraint set's center, cut along the
            directions the remainder set depends on.

        Raises:
            TypeError: if the constraint set has no support function to
                read its reach from.
            ValueError: if the constraint set is unbounded along a
                direction.
        """
        space = feasible.model_space
        center = feasible._default_start()
        chosen = _independent(space, self._directions)
        widths = [feasible._half_reach(direction, center) for direction in chosen]
        cell = feasible.cell(center, directions=chosen, half_widths=widths)
        # A slab the remainder set does not narrow with is not worth its
        # row in every cell's problem.
        base = _extent(self._remainder.remainder_set(cell))
        keep = []
        for index in range(len(chosen)):
            thin = np.array(widths, dtype=float)
            thin[index] *= 1e-3
            trial = feasible.cell(center, directions=chosen, half_widths=thin)
            if _extent(self._remainder.remainder_set(trial)) < base * (1.0 - 1e-12):
                keep.append(index)
        return feasible.cell(
            center,
            directions=[chosen[index] for index in keep],
            half_widths=[widths[index] for index in keep],
        )

    def _split_index(
        self, feasible: "NonlinearFeasiblePropertySet", cell: Cell, root: Cell
    ) -> int:
        """The slab whose halving shrinks the remainder set most; the relatively widest on a tie."""
        base = _extent(self._remainder.remainder_set(cell))
        gains = []
        for index in range(len(cell.directions)):
            halved = cell.half_widths.copy()
            halved[index] *= 0.5
            trial = feasible.cell(
                cell.center, directions=cell.directions, half_widths=halved
            )
            gains.append(base - _extent(self._remainder.remainder_set(trial)))
        if max(gains) > 1e-12 * base:
            return int(np.argmax(gains))
        return int(np.argmax(cell.half_widths / root.half_widths))

    def support(
        self,
        feasible: "NonlinearFeasiblePropertySet",
        direction: Any,
        start: Any,
        /,
    ) -> BranchSupportResult:
        """The support in a direction, from both sides.

        Args:
            feasible: the set, whose noise set must be a ball.
            direction: the direction in the property space.
            start: a model to run the first local search from, or None for
                the set's fitting model, where a search finds one.

        Returns:
            The bracket, the cells examined, and the bracket's history.

        Raises:
            TypeError: if the noise set is not a ball, or the constraint
                set has no support function.
            ValueError: if the property operator's adjoint annihilates the
                direction, or the constraint set is unbounded along a slab
                direction.
        """
        space, data_space = feasible.model_space, feasible.data_space
        forward, prior, noise = (
            feasible.forward_operator,
            feasible.prior,
            feasible.noise,
        )
        ball = feasible._noise_ball()
        pulled = feasible.property_operator.adjoint(direction)
        if space.norm(pulled) == 0.0:
            raise ValueError("The direction is annihilated by the property operator.")
        reach = feasible._reach(pulled)
        search = self._search or LocalSearch()
        # Any certificate bounds, so a cell's dual is solved only as far as
        # the gap asked of the whole search makes worth while.
        dual_tolerance = max(1e-10, 1e-2 * self._rtol)
        best, attained, refuted = None, float("-inf"), False
        evaluations, searches, unconverged = 0, 0, 0

        def offer(model: Any, entry: dict[str, Any] | None) -> None:
            """Check a model, test the claim on it where it lies in a cell, and keep it if it is the best."""
            nonlocal best, attained, refuted, evaluations
            if model is None or not prior.contains(model, rtol=0.0):
                return
            try:
                predicted = forward(model)
            except EvaluationError:
                return
            evaluations += 1
            if entry is not None and entry["cell"].contains(model, rtol=0.0):
                if _departs(
                    forward, entry["linearization"], entry["claimed"], model, predicted
                ):
                    refuted = True
            along = float(space.inner_product(pulled, model))
            fits = noise.contains(
                data_space.subtract(feasible.data, predicted), rtol=0.0
            )
            if fits and along > attained:
                best, attained = model, along

        def bound(cell: Cell, parent: dict[str, Any] | None) -> dict[str, Any] | None:
            """The outer value of a cell, or None for a cell shown empty."""
            nonlocal evaluations, unconverged
            rows = feasible._active_slabs(cell)
            # A parent's certificate starts its halves, where they have
            # the same rows and so the same space of certificates.
            warm = (
                parent["certificate"]
                if parent is not None and parent["rows"] == rows
                else None
            )
            linearization = forward.at(cell.center)
            claimed = self._remainder.remainder_set(cell)
            linear = feasible._cell_set(cell, linearization, _grown(ball, claimed))
            # A short run first: a cell with no feasible model, or none
            # worth having, shows itself in a few steps, by a certificate
            # of emptiness or a value below the one attained, and any
            # certificate bounds. Only a cell still in play is run out.
            try:
                result = linear.support_result(
                    direction,
                    via=Dual(tolerance=dual_tolerance, max_iterations=40, start=warm),
                )
                evaluations += int(result.evaluations)
                if not result.converged and result.value > attained:
                    result = linear.support_result(
                        direction,
                        via=Dual(tolerance=dual_tolerance, start=result.certificate),
                    )
                    evaluations += int(result.evaluations)
            except ValueError as error:
                if "empty" not in str(error):
                    raise
                return None
            # A dual that stopped at its budget still certifies its value,
            # by weak duality; it is only a looser bound than it might be.
            unconverged += 0 if result.converged else 1
            return {
                "cell": cell,
                "rows": rows,
                "value": float(result.value),
                "certificate": result.certificate,
                "extremal": result.extremal_model,
                "linearization": linearization,
                "claimed": claimed,
            }

        origin = start
        if origin is None:
            fitting = feasible.fitting_result().fitting_model
            if fitting is not None and prior.contains(fitting, rtol=0.0):
                origin = fitting
        if origin is not None:
            first = search.support(feasible, direction, origin)
            evaluations += int(first.evaluations)
            searches += 1
            offer(first.extremal_model, None)

        root = self.root(feasible)
        records: list[list[Any]] = []
        heap: list[tuple[float, int, int, dict[str, Any]]] = []
        history: list[tuple[int, float, float]] = []
        pruned = float("-inf")

        def admit(cell: Cell, parent: dict[str, Any] | None, cap: float) -> None:
            """Bound a cell and file it as empty, pruned or live."""
            nonlocal pruned
            narrowed = cell
            if self._tighten:
                narrowed = feasible.tightened_cell(
                    cell, self._remainder, sweeps=self._tighten, rtol=dual_tolerance
                )
            entry = None if narrowed is None else bound(narrowed, parent)
            if entry is None:
                records.append([cell, float("-inf"), "empty"])
                return
            cell = narrowed
            # A half is linearized about its own center, so its bound is
            # not nested in its parent's; both hold, and the smaller is kept.
            entry["value"] = min(entry["value"], cap)
            offer(entry["extremal"], entry)
            if entry["value"] <= attained:
                pruned = max(pruned, entry["value"])
                records.append([cell, entry["value"], "pruned"])
                return
            records.append([cell, entry["value"], "live"])
            heapq.heappush(
                heap, (-entry["value"], len(records), len(records) - 1, entry)
            )

        admit(root, None, float("inf"))
        converged = False
        message = "the budget of cells was reached"
        while True:
            if not heap:
                converged, message = True, "every cell was shown empty or pruned"
                break
            upper = max(-heap[0][0], pruned)
            if upper - attained <= self._rtol * reach:
                converged, message = True, "the outer and inner values met"
                break
            if len(records) + 2 > self._max_cells:
                break
            _, _, index, entry = heapq.heappop(heap)
            cell = entry["cell"]
            # The cell with the most to offer: look in it for a better
            # model, unless the best one so far is already its own.
            if best is None or not cell.contains(best):
                origin = entry["extremal"]
                if origin is None or not cell.contains(origin):
                    origin = cell.center
                found = search.support(feasible, direction, origin, cell=cell)
                evaluations += int(found.evaluations)
                searches += 1
                offer(found.extremal_model, entry)
            if entry["value"] <= attained:
                pruned = max(pruned, entry["value"])
                records[index][2] = "pruned"
                continue
            if not cell.directions:
                heapq.heappush(heap, (-entry["value"], index, index, entry))
                message = "the cell has no slabs to split"
                break
            split = self._split_index(feasible, cell, root)
            records[index][2] = "split"
            for sign in (-1.0, 1.0):
                offsets = np.zeros(len(cell.directions))
                offsets[split] = sign * 0.5 * cell.half_widths[split]
                halved = cell.half_widths.copy()
                halved[split] *= 0.5
                admit(
                    feasible.cell(
                        cell.point(offsets),
                        directions=cell.directions,
                        half_widths=halved,
                    ),
                    entry,
                    entry["value"],
                )
            history.append(
                (len(records), max([pruned] + [-item[0] for item in heap]), attained)
            )

        value = max([pruned] + [-item[0] for item in heap])
        if attained > value + 1e-9 * reach:
            # A checked model beyond the outer value: the claim is false.
            refuted = True
        if unconverged:
            message += (
                f"; {unconverged} cell bounds are from dual solves stopped at "
                "their budget, which bound all the same"
            )
        if refuted:
            message += "; a checked model contradicts the remainder claim"
        return BranchSupportResult(
            value=float(value),
            inner=attained,
            converged=bool(converged),
            algorithm=self.name,
            evaluations=evaluations,
            certificate=heap[0][3]["certificate"] if heap else None,
            extremal_model=best,
            message=message,
            certified=self._remainder.certified,
            refuted=refuted,
            cells=tuple(CellRecord(*record) for record in records),
            history=tuple(history),
            searches=searches,
        )

    def inclusion(
        self,
        feasible: "NonlinearFeasiblePropertySet",
        value: Any,
        start: Any,
        /,
    ) -> BranchInclusionResult:
        """Whether a property value is in the set: a model that has it, or a cover that excludes it.

        A local search looks for a fitting model with the property, first
        from ``start`` and then in each cell taken up whose remainder set
        is smaller than the noise set. A cell is dropped
        when the linearized problem on it, with the noise set grown by the
        remainder set and the property held at the value, has no feasible
        model; a cell that cannot be dropped is split. The value is proved
        inside by a model, and outside when no cell is left.

        Args:
            feasible: the set, whose noise set must be a ball.
            value: the property value.
            start: a model to run the first local search from, or None for
                the set's fitting model, where a search finds one.

        Returns:
            The verdict, with the model or the cells behind it.

        Raises:
            TypeError: if the noise set is not a ball, the constraint set
                has no support function, or the property space has no
                coordinate map.
        """
        _require_property_coordinates(feasible.domain, "Membership")
        forward, prior = feasible.forward_operator, feasible.prior
        ball = feasible._noise_ball()
        search = self._search or LocalSearch()
        dual_tolerance = max(1e-10, 1e-2 * self._rtol)
        evaluations, searches = 0, 0
        records: list[list[Any]] = []

        def finish(
            model: Any, level: float, excluded: bool, message: str
        ) -> BranchInclusionResult:
            return BranchInclusionResult(
                level=level,
                lower_bound=float("inf") if excluded else float("-inf"),
                converged=bool(model is not None or excluded),
                algorithm=self.name,
                evaluations=evaluations,
                fitting_model=model,
                message=message,
                certified=self._remainder.certified,
                cells=tuple(CellRecord(*record) for record in records),
                searches=searches,
            )

        def look(origin: Any, cell: Cell | None) -> InclusionResult | None:
            """A local search for a model with the property; the result if it proves membership."""
            nonlocal evaluations, searches
            found = search.inclusion(feasible, value, origin, cell=cell)
            evaluations += int(found.evaluations)
            searches += 1
            if found.fitting_model is not None and found.level <= feasible.level:
                return found
            return None

        def holds_none(cell: Cell) -> bool:
            """Whether the linearized problem on a cell has no model with the property."""
            nonlocal evaluations
            claimed = self._remainder.remainder_set(cell)
            linear = feasible._cell_set(
                cell, forward.at(cell.center), _grown(ball, claimed), pin=value
            )
            for budget in (40, 300):
                result = linear.feasibility_result(
                    via=Dual(tolerance=dual_tolerance, max_iterations=budget)
                )
                evaluations += int(result.evaluations)
                if result.level == float("inf"):
                    return True
                if result.converged:
                    return False
            return False

        origin = start
        if origin is None:
            fitting = feasible.fitting_result().fitting_model
            if fitting is not None and prior.contains(fitting, rtol=0.0):
                origin = fitting
        if origin is not None:
            found = look(origin, None)
            if found is not None:
                return finish(
                    found.fitting_model, found.level, False, "a model has the property"
                )

        root = self.root(feasible)
        heap: list[tuple[float, int, int, Cell]] = []

        def admit(cell: Cell) -> None:
            narrowed = cell
            if self._tighten:
                narrowed = feasible.tightened_cell(
                    cell, self._remainder, sweeps=self._tighten, rtol=dual_tolerance
                )
            if narrowed is None or holds_none(narrowed):
                records.append([cell, float("-inf"), "empty"])
                return
            cell = narrowed
            records.append([cell, float("nan"), "live"])
            extent = _extent(self._remainder.remainder_set(cell))
            heapq.heappush(heap, (-extent, len(records), len(records) - 1, cell))

        admit(root)
        while heap and len(records) + 2 <= self._max_cells:
            _, _, index, cell = heapq.heappop(heap)
            # A search in the cell is worth its cost once the remainder is
            # below the noise, where the linearized problem that could not
            # be shown empty is close to the real one.
            if _shrunk(ball, self._remainder.remainder_set(cell), 0.0) is not None:
                found = look(cell.center, cell)
                if found is not None:
                    return finish(
                        found.fitting_model,
                        found.level,
                        False,
                        "a model has the property",
                    )
            if not cell.directions:
                heapq.heappush(heap, (0.0, index, index, cell))
                break
            split = self._split_index(feasible, cell, root)
            records[index][2] = "split"
            for sign in (-1.0, 1.0):
                offsets = np.zeros(len(cell.directions))
                offsets[split] = sign * 0.5 * cell.half_widths[split]
                halved = cell.half_widths.copy()
                halved[split] *= 0.5
                admit(
                    feasible.cell(
                        cell.point(offsets),
                        directions=cell.directions,
                        half_widths=halved,
                    )
                )
        if not heap:
            return finish(
                None,
                float("inf"),
                True,
                "no cell holds a fitting model with the property, given the "
                "remainder claim",
            )
        return finish(
            None,
            float("inf"),
            False,
            "the budget of cells was reached with the value neither found nor "
            "excluded",
        )

    def __repr__(self) -> str:
        return f"BranchAndPrune(rtol={self._rtol}, max_cells={self._max_cells})"


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
        # Where the forward operator cannot be evaluated a search cannot
        # begin; the next place is tried, a cell's center last.
        origins = [stepped, start] + ([] if cell is None else [cell.center])
        result = self._begun(self._method(inequalities=constraints), objective, origins)

        def along(model: Any) -> float:
            return float(space.inner_product(pulled, model))

        def checked(model: Any) -> bool:
            if cell is not None and not cell.contains(model, rtol=0.0):
                return False
            return feasible.is_feasible(model)

        found = None if result is None else result.minimizer
        best, inner = None, float("-inf")
        for candidate in (start, stepped, found):
            if candidate is None:
                continue
            if checked(candidate) and along(candidate) > inner:
                best, inner = candidate, along(candidate)
        if result is None:
            return SupportResult(
                value=float("inf"),
                inner=inner,
                converged=False,
                algorithm=self.name,
                evaluations=0,
                extremal_model=best,
                message="the forward operator could not be evaluated where the "
                "search was to begin",
            )
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

    @staticmethod
    def _begun(
        method: AugmentedLagrangian, objective: Functional, origins: Sequence[Any]
    ) -> Any:
        """The method's result from the first origin the forward operator evaluates at, or None."""
        for origin in origins:
            if origin is None:
                continue
            try:
                return method.minimize(objective, origin)
            except EvaluationError:
                continue
        return None

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
        *,
        cell: Cell | None = None,
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
            cell: a cell to keep the search in, each slab a further
                constraint.

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
            feasible, start, through=feasible._with_property(value), cell=cell
        )

    def _least_level(
        self,
        feasible: "NonlinearFeasiblePropertySet",
        start: Any,
        /,
        *,
        through: AffineOperator | None,
        cell: Cell | None = None,
    ) -> InclusionResult:
        prior_level = feasible._prior_level
        objective = _rescaled(prior_level, 1.0 / _scale_of(feasible.level), 0.0)
        constraints = feasible._constraints(self._margin)[1:]
        if cell is not None:
            constraints += _slab_constraints(cell, self._margin)
        if through is not None:
            objective = objective @ through
            constraints = tuple(constraint @ through for constraint in constraints)
        origins = [start] + ([] if cell is None else [cell.center])
        result = self._begun(self._method(inequalities=constraints), objective, origins)
        best, level = None, float("inf")
        for candidate in (start, None if result is None else result.minimizer):
            if candidate is None:
                continue
            if through is not None:
                candidate = through(candidate)
            if cell is not None and not cell.contains(candidate, rtol=0.0):
                continue
            if feasible.fits(candidate) and float(prior_level(candidate)) < level:
                best, level = candidate, float(prior_level(candidate))
        if result is None:
            return InclusionResult(
                level=level,
                lower_bound=float("-inf"),
                converged=False,
                algorithm=self.name,
                evaluations=0,
                fitting_model=best,
                message="the forward operator could not be evaluated where the "
                "search was to begin",
            )
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
        dense_limit: int = 256,
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
            dense_limit: the largest data dimension at which the operator
                linearized on a cell is held as a matrix. Assembling it
                costs one application of the derivative's adjoint per
                datum, and a cell's dual applies the operator and its
                adjoint some hundreds of times, so with a few hundred data
                or fewer the matrix is the cheaper route.

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
        self._dense_limit = int(dense_limit)
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
            dense_limit=self._dense_limit,
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

        Decided by the noise set's own membership test at no tolerance. A
        model the forward operator cannot evaluate fits nothing.
        """
        try:
            residual = self.residual(model)
        except EvaluationError:
            return False
        return bool(self._noise.contains(residual, rtol=0.0))

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

    def _half_reach(self, vector: Any, center: Any) -> float:
        """The largest ``|(v, m − center)|`` over the prior, from its support function."""
        space = self.model_space
        if not _computes_support(self._prior):
            raise TypeError(
                "The constraint set must have a support function to read its "
                "reach along a slab direction from."
            )
        support = _support_of(self._prior)
        offset = float(space.inner_product(vector, center))
        reach = max(
            float(support(vector)) - offset,
            float(support(space.negative(vector))) + offset,
        )
        if not np.isfinite(reach) or reach <= 0.0:
            raise ValueError("The constraint set is unbounded along a slab direction.")
        return reach

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
        derivative = self._held(linearization)
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
        may be about a model that does not fit, or when the forward
        operator cannot be evaluated at the model.
        """
        try:
            return self.linearized(model).support_result(direction).extremal_model
        except (ValueError, ConvergenceError, EvaluationError):
            return None

    def support_result(
        self,
        direction: Any,
        /,
        *,
        start: Any = None,
        via: "LocalSearch | BranchAndPrune | None" = None,
    ) -> SupportResult:
        """The support in a direction: from inside by a local search, or from both sides.

        By default one :class:`LocalSearch`, whose answer is a checked
        model and no bound from outside. ``via=BranchAndPrune(remainder)``
        searches over cells instead and returns a bracket on the support
        of the whole set, as good as the remainder claim it is given.

        Args:
            direction: the direction in the property space.
            start: the model to search from; :meth:`fitting_model` if
                omitted.
            via: a :class:`LocalSearch` with its own tuning, or a
                :class:`BranchAndPrune`; the default local search if
                omitted.

        Returns:
            The result, with ``inner`` attained by ``extremal_model``.
            ``value`` is infinite from a local search, and the outer bound
            from a search over cells; see :meth:`LocalSearch.support` and
            :meth:`BranchAndPrune.support`.

        Raises:
            ValueError: if a local search is given no start and no fitting
                model is found to start from.
        """
        if isinstance(via, BranchAndPrune):
            return via.support(self, direction, start)
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
        via: "LocalSearch | BranchAndPrune | None" = None,
    ) -> InclusionResult:
        """Whether a property value is in the set: proved by a model, or excluded by a cover.

        By default one :class:`LocalSearch` for the model with the property
        closest to the prior's center: a ``level`` at or below
        :attr:`level` proves the value is in the set, by the
        ``fitting_model`` that has it, and anything else proves nothing.
        ``via=BranchAndPrune(remainder)`` can also prove the value outside,
        by a cover of the prior none of whose cells holds a fitting model
        with the property, given the remainder claim.

        Args:
            value: the property value.
            start: the model to search from; :meth:`fitting_model` if
                omitted.
            via: a :class:`LocalSearch` with its own tuning, or a
                :class:`BranchAndPrune`; the default local search if
                omitted.

        Returns:
            The result; see :meth:`LocalSearch.inclusion` and
            :meth:`BranchAndPrune.inclusion`.

        Raises:
            TypeError: if the property space has no coordinate map.
            ValueError: if a local search is given no start and no fitting
                model is found.
        """
        if isinstance(via, BranchAndPrune):
            return via.inclusion(self, value, start)
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
        half_widths: Sequence[float] | None = None,
        radius: float | None = None,
    ) -> Cell:
        """A cell about a model, its radius read from the prior where the prior has one.

        Args:
            center: the model the cell is about.
            directions: the vectors the slabs are taken along.
            half_widths: the half-width of each slab; the prior's reach
                from the center along each direction if omitted, so that
                the cell is the whole prior, cut nowhere yet.
            radius: a bound on the distance of a model of the cell from the
                center, if one is known. A ball prior gives its own, the
                farthest any of its models can be, and the smaller is kept.

        Returns:
            The cell.

        Raises:
            TypeError: if the half-widths are left to the prior and it has
                no support function to read its reach from.
            ValueError: if the prior is unbounded along a direction whose
                half-width is left to it.
        """
        space = self.model_space
        if half_widths is None:
            half_widths = [
                self._half_reach(direction, center) for direction in directions
            ]
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

    def _held(self, linearization: Linearization) -> LinearOperator:
        """A linearization's derivative, as a matrix where the data are few enough.

        Assembled through the adjoint, one application per datum; the last
        linearization's is kept, a cell's probes sharing it.
        """
        derivative = linearization.derivative
        data_space = self.data_space
        if not isinstance(data_space, CoordinateSpace) or not isinstance(
            self.model_space, CoordinateSpace
        ):
            return derivative
        if data_space.dim > self._dense_limit:
            return derivative
        kept = self._cache.get("held")
        if kept is None or kept[0] is not linearization:
            kept = (linearization, derivative.adjoint.assembled().adjoint)
            self._cache["held"] = kept
        return kept[1]

    def _active_slabs(self, cell: Cell) -> list[int]:
        """The slabs of a cell that cut the prior: those narrower than its reach along them."""
        if not _computes_support(self._prior):
            return list(range(len(cell.directions)))
        return [
            index
            for index, (direction, half_width) in enumerate(
                zip(cell.directions, cell.half_widths)
            )
            if half_width < self._half_reach(direction, cell.center)
        ]

    def _cell_set(
        self,
        cell: Cell,
        linearization: Linearization,
        noise: ConvexSet,
        *,
        pin: Any = None,
        property_operator: LinearOperator | None = None,
    ) -> FeasiblePropertySet:
        """The linearized problem on a cell, with a given noise set.

        With ``pin``, a property value, the models are further held to have
        that property exactly: the set is then empty exactly when no model
        of the linearized problem on the cell has it. With
        ``property_operator``, the set is that operator's image of the same
        models, in place of the property's.
        """
        image = (
            self._property_operator if property_operator is None else property_operator
        )
        data_space = self.data_space
        derivative = self._held(linearization)
        shifted = data_space.add(
            data_space.subtract(self._data, linearization.value),
            derivative(linearization.point),
        )
        operators, parts, observed = [derivative], [noise], [shifted]
        # The slabs enter as data: the rows (u_j, m) observed at the
        # center's values to within the half-widths. A slab the prior
        # already implies is left out.
        space = self.model_space
        active = self._active_slabs(cell)
        if active:
            chosen = [cell.directions[index] for index in active]
            frame = EuclideanSpace(len(active))

            def value(model: Any) -> np.ndarray:
                return np.array([space.inner_product(u, model) for u in chosen])

            def adjoint(weights: np.ndarray) -> Any:
                out = space.zero()
                for weight, direction in zip(weights, chosen):
                    out = space.axpy(float(weight), direction, out)
                return out

            rows = LinearOperator.from_callables(space, frame, value, adjoint=adjoint)
            operators.append(rows)
            parts.append(
                Zonotope(
                    frame,
                    [frame.basis_vector(j) for j in range(len(active))],
                    lower=-cell.half_widths[active],
                    upper=cell.half_widths[active],
                )
            )
            observed.append(rows(cell.center))
        if pin is not None:
            # The property enters as data too, observed without error.
            operators.append(self._property_operator)
            parts.append(Ball(self.domain, radius=0.0))
            observed.append(pin)
        if len(operators) == 1:
            structure = FeasibleStructure(
                derivative,
                image,
                self._prior,
                noise=noise,
                solver=self._solver,
            )
            return FeasiblePropertySet.from_structure(structure, shifted)
        forward = ColumnLinearOperator(operators)
        structure = FeasibleStructure(
            forward,
            image,
            self._prior,
            noise=_product_set(forward.codomain, parts),
            solver=self._solver,
        )
        return FeasiblePropertySet.from_structure(structure, tuple(observed))

    def tightened_cell(
        self,
        cell: Cell,
        remainder: RemainderBound,
        /,
        *,
        sweeps: int = 3,
        rtol: float = 1e-6,
    ) -> Cell | None:
        """A cell's slabs narrowed to what its feasible models can reach.

        Along each slab direction ``u_j`` the values ``(u_j, m)`` over the
        feasible models of the cell lie within the support, either way, of
        the cell's outer set seen through the slab rows; a slab wider than
        that holds nothing feasible in the excess and is cut back, the
        cell recentred on what is left. A narrower cell has a smaller
        remainder set, hence a smaller outer set, so the cut can be
        repeated. Nothing feasible is lost, given the remainder claim, and
        no cell is added: where the data pin a direction this does at once
        what halving the slab would take many cells to do.

        Args:
            cell: the cell.
            remainder: the claim about the forward operator over a cell.
            sweeps: how many times at most to go over the slabs; a sweep
                that narrows no slab by a twentieth is the last.
            rtol: the tolerance of the support solves, as a relative gap.

        Returns:
            The narrowed cell, or None when the cell's outer set is empty:
            the cell holds no feasible model, given the claim.

        Raises:
            TypeError: if the noise set is not a ball.
        """
        ball = self._noise_ball()
        count = len(cell.directions)
        if count == 0:
            return cell
        frame = EuclideanSpace(count)
        for _ in range(sweeps):
            linearization = self._forward.at(cell.center)
            linear = self._cell_set(
                cell,
                linearization,
                _grown(ball, remainder.remainder_set(cell)),
                property_operator=cell.row_operator(),
            )
            at_center = np.array(
                [
                    self.model_space.inner_product(u, cell.center)
                    for u in cell.directions
                ]
            )
            low = at_center - cell.half_widths
            high = at_center + cell.half_widths
            warm = None
            try:
                for index in range(count):
                    for sign in (1.0, -1.0):
                        result = linear.support_result(
                            frame.scale(sign, frame.basis_vector(index)),
                            via=Dual(tolerance=rtol, start=warm),
                        )
                        warm = result.certificate
                        if sign > 0.0:
                            high[index] = min(high[index], float(result.value))
                        else:
                            low[index] = max(low[index], -float(result.value))
            except ValueError as error:
                if "empty" not in str(error):
                    raise
                return None
            if np.any(low > high + 1e-9 * cell.half_widths):
                return None
            high = np.maximum(high, low)
            halves = np.maximum(0.5 * (high - low), 1e-9 * cell.half_widths)
            narrowed = float(np.min(halves / cell.half_widths))
            cell = self.cell(
                cell.point(0.5 * (high + low) - at_center),
                directions=cell.directions,
                half_widths=halves,
            )
            if narrowed > 0.95:
                break
        return cell

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
            try:
                predicted = self._forward(model)
            except EvaluationError:
                continue
            if _departs(self._forward, linearization, claimed, model, predicted):
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
