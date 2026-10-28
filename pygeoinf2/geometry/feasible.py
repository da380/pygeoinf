"""
The feasible property set of a linear inference problem, and what probes it.

The set is ``T(prior ∩ A⁻¹(d − noise))``: every value a property ``T m`` can
take over the models ``m`` that lie in a convex constraint set and whose
predictions ``A m`` fit the data to within a convex noise set. It is convex,
it lives in a finite-dimensional property space, and it is defined only
implicitly, so every question about it -- how far it reaches in a direction,
whether it contains a point, whether it is empty -- is an optimization over
the model space with the forward operator inside.

The *form* of the two sets decides how those optimizations are done. Exact
data with a ball or an ellipsoid as the prior give a closed form: the set is
an ellipsoid. A ball or an ellipsoid on each side turns every question into
a damped least-squares family and a scalar root find. Anything else is
answered through what the sets supply, a support function or a projection.
Each probe of :class:`FeasiblePropertySet` dispatches to the first algorithm
that accepts the sets' form and can be made to use another by name, for a
cross-check, or by an instance, when tuning rides along. Every answer carries
a certified bracket: a value no smaller than the true one, from a dual
certificate, and one no larger, from an attainable point.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Literal, Sequence

import numpy as np

from ..algebra.operators import Functional, LinearOperator
from ..algebra.spaces import CoordinateSpace, HilbertSpace
from ..numerics.convex import SupportFunction
from ..numerics.normal import DampedNormalOperator
from ..numerics.root_find import DampedSolves, Evaluation, RootResult, monotone_root
from ..numerics.solvers import CGSolver, ConvergenceError, LinearSolver
from ..traits import Traits
from .convex import (
    Ball,
    ConvexSet,
    Ellipsoid,
    HalfSpace,
    Polytope,
    _LevelSupport,
    _project_by_support,
    sum_of_ellipsoids,
)
from .sets import SublevelSet, Subset

__all__ = [
    "Algorithm",
    "ClosedForm",
    "Distance",
    "Dual",
    "FeasiblePropertySet",
    "FeasibleStructure",
    "InclusionResult",
    "LinearCertificate",
    "Primal",
    "Search",
    "Splitting",
    "SupportResult",
    "SweepResult",
    "linear_certificate_operator",
]

_UNBOUNDED = 1.0e6
"""How far below zero a non-converged dual must fall to be read as unbounded."""

Form = Literal["closed_form", "quadratic", "general"]


# --------------------------------------------------------------------- #
#                            What a probe returns                        #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class SupportResult:
    """The support of the set in one direction, with its certificate.

    ``value`` is the outer bound, the number that is a bound; ``inner`` is
    a value attained by a point of the set, so the truth lies in
    ``[inner, value]`` and ``gap`` is their difference. An algorithm that
    exhibits no point leaves ``inner`` at minus infinity. A direction in
    which the set is unbounded has ``value`` infinite.
    """

    value: float
    inner: float
    converged: bool
    algorithm: str
    evaluations: int
    certificate: Any = None
    """The dual variable on the data space certifying ``value``, where the
    algorithm produces one."""
    extremal_model: Any = None
    """A model of the feasible set attaining ``inner``, where the algorithm
    exhibits one."""
    multipliers: tuple[float, ...] | None = None
    message: str = ""

    @property
    def gap(self) -> float:
        """``value - inner``, the width of the certified bracket."""
        return self.value - self.inner

    def __repr__(self) -> str:
        return (
            f"SupportResult(value={self.value:.6g}, gap={self.gap:.3g}, "
            f"algorithm={self.algorithm!r}, converged={self.converged})"
        )


@dataclass(frozen=True)
class InclusionResult:
    """The prior's level function at the model fitting a property value.

    ``level`` is the smallest value the prior's level function takes over
    the models that have the given property and fit the data; the value is
    in the set exactly when ``level`` is at most the prior's own level.
    ``lower_bound`` certifies ``level`` from below, so a lower bound above
    the prior's level is a proof of exclusion. ``level`` is infinite when no
    model has the property and fits the data at all.
    """

    level: float
    lower_bound: float
    converged: bool
    algorithm: str
    evaluations: int
    fitting_model: Any = None
    """The model attaining ``level``, where the algorithm exhibits one."""
    multiplier: float | None = None
    message: str = ""

    def __repr__(self) -> str:
        return (
            f"InclusionResult(level={self.level:.6g}, "
            f"algorithm={self.algorithm!r}, converged={self.converged})"
        )


@dataclass(frozen=True)
class EllipsoidResult:
    """An ellipsoid containing the set, with how it was chosen.

    ``ellipsoid`` is certified: every point of the set lies in it. ``weight``
    is the parameter the algorithm optimized, where it has one -- for the
    S-procedure the weight ``θ`` on the prior against ``1 − θ`` on the
    data -- and ``criterion`` says what it was optimized for.
    """

    ellipsoid: Ellipsoid
    criterion: str
    converged: bool
    algorithm: str
    evaluations: int
    weight: float | None = None
    message: str = ""

    def __repr__(self) -> str:
        return (
            f"EllipsoidResult(criterion={self.criterion!r}, "
            f"algorithm={self.algorithm!r}, converged={self.converged})"
        )


@dataclass(frozen=True)
class SweepResult:
    """The support in many directions, one result each.

    What :meth:`FeasiblePropertySet.polytope` consumes: it needs to know
    which directions came back infinite.
    """

    directions: tuple[Any, ...]
    results: tuple[SupportResult, ...]

    @property
    def values(self) -> np.ndarray:
        """The outer bounds, one per direction."""
        return np.array([result.value for result in self.results])

    @property
    def finite(self) -> np.ndarray:
        """Which directions have a finite bound."""
        return np.isfinite(self.values)

    @property
    def converged(self) -> bool:
        """Whether every direction's bracket closed."""
        return all(result.converged for result in self.results)


# --------------------------------------------------------------------- #
#                      The data-independent structure                    #
# --------------------------------------------------------------------- #


def _is_point(noise: ConvexSet | None) -> bool:
    """Whether the noise set is a single point, which is exact data."""
    return noise is None or (isinstance(noise, Ball) and noise.radius == 0.0)


def _is_quadratic(subset: ConvexSet | None) -> bool:
    return isinstance(subset, (Ball, Ellipsoid))


class FeasibleStructure:
    """What every set from one estimator shares: operators, sets, solver, caches.

    The data-independent part of a feasible property set: the operators,
    the two sets, the solver, and the expensive work each algorithm does
    once -- a kernel projector's pieces, a closed form's shape, a family
    with its preconditioner -- kept under :meth:`shared`, keyed by the
    algorithm's settings, so that the sets for different data share it. An
    algorithm reads what it needs from here through the set's
    :attr:`FeasiblePropertySet.structure`.
    """

    def __init__(
        self,
        forward: LinearOperator,
        property_operator: LinearOperator,
        prior: ConvexSet,
        /,
        *,
        noise: ConvexSet | None,
        solver: LinearSolver | None,
    ) -> None:
        if property_operator.domain != forward.domain:
            raise ValueError("The property operator must act on the model space.")
        if not isinstance(prior, (ConvexSet, SublevelSet)):
            raise TypeError(
                "The constraint set must be a ConvexSet, or a SublevelSet of a "
                "convex functional."
            )
        if prior.domain != forward.domain:
            raise ValueError("The constraint set must lie in the model space.")
        if noise is not None:
            if not isinstance(noise, (ConvexSet, SublevelSet)):
                raise TypeError(
                    "The noise set must be a ConvexSet or a SublevelSet of a "
                    "convex functional, or None for exact data."
                )
            if noise.domain != forward.codomain:
                raise ValueError("The noise set must lie in the data space.")
        self.forward = forward
        self.property_operator = property_operator
        self.prior = prior
        self.noise = None if _is_point(noise) else noise
        self._solver = solver
        self.cache: dict[Any, Any] = {}
        self.lock = threading.RLock()

    @property
    def solver(self) -> LinearSolver | None:
        """The solver the caller supplied, or None for the default."""
        return self._solver

    def solver_for(self, space: HilbertSpace) -> LinearSolver:
        """The solver for systems on a space: the caller's, or conjugate gradients.

        The default is ``CGSolver(rtol=1e-10)`` with an iteration cap of at
        least 200: a tighter tolerance is below the rounding floor of an
        ill-conditioned family, and the solver's own cap of twice the
        dimension is too tight for a small one. One instance per space,
        kept, so that the families built on it share.
        """
        if self._solver is not None:
            return self._solver
        return self.shared(
            ("default_solver", space.dim), lambda: _default_solver(space)
        )

    @property
    def model_space(self) -> HilbertSpace:
        """The model space, which ``A`` and ``T`` act on."""
        return self.forward.domain

    @property
    def data_space(self) -> HilbertSpace:
        """The data space."""
        return self.forward.codomain

    @property
    def property_space(self) -> HilbertSpace:
        """The property space, where the feasible set lives."""
        return self.property_operator.codomain

    @property
    def exact(self) -> bool:
        """Whether the data are exact: the noise set is a single point."""
        return self.noise is None

    @property
    def quadratic(self) -> bool:
        """Whether the prior and the noise are each a ball or an ellipsoid."""
        return _is_quadratic(self.prior) and (self.exact or _is_quadratic(self.noise))

    @property
    def form(self) -> Form:
        """The form of the pair of sets, which decides the algorithms."""
        if self.exact and _is_quadratic(self.prior):
            return "closed_form"
        if self.quadratic:
            return "quadratic"
        return "general"

    def shared(self, key: Any, build: Callable[[], Any]) -> Any:
        """A data-independent piece, built once under the lock."""
        with self.lock:
            if key not in self.cache:
                self.cache[key] = build()
            return self.cache[key]

    def prior_covariance(self) -> LinearOperator:
        """``C`` with the prior ``{(C⁻¹(m − a), m − a) <= 1}``, in normalized units.

        A ball of radius ``r`` has ``C == r² I``; an ellipsoid its own
        covariance, or the solver's inverse of its precision when it was
        built without one.

        Raises:
            TypeError: if the prior is neither.
        """
        return _covariance_of(self.prior, self.model_space, self.solver_for)

    def prior_center(self) -> Any:
        """The prior's centre, or the origin for a set without one."""
        prior = self.prior
        if isinstance(prior, (Ball, Ellipsoid)):
            return prior.center
        return self.model_space.zero()

    def prior_precision(self) -> LinearOperator:
        """``C⁻¹``, the prior's quadratic form in normalized units.

        Raises:
            TypeError: if the prior is neither a ball nor an ellipsoid.
        """
        return _precision_of(self.prior, self.model_space, "prior")

    def noise_covariance(self) -> LinearOperator:
        """``R`` with the noise set ``{(R⁻¹(e − b), e − b) <= 1}``.

        Raises:
            TypeError: if the noise set is neither a ball nor an ellipsoid,
                or the data are exact.
        """
        return _covariance_of(self.noise, self.data_space, self.solver_for)

    def noise_precision(self) -> LinearOperator:
        """``R⁻¹``, the noise set's quadratic form in normalized units.

        Raises:
            TypeError: if the noise set is neither a ball nor an ellipsoid,
                or the data are exact.
        """
        if self.noise is None:
            raise TypeError("Exact data have no noise precision.")
        return _precision_of(self.noise, self.data_space, "noise")

    def noise_center(self) -> Any:
        """The noise set's centre, or the origin."""
        noise = self.noise
        if isinstance(noise, (Ball, Ellipsoid)):
            return noise.center
        return self.data_space.zero()

    def shifted_data(self, data: Any) -> Any:
        """``d − A a − b``: the data relative to both centres."""
        space = self.data_space
        shifted = space.subtract(data, self.forward(self.prior_center()))
        if self.noise is not None:
            shifted = space.subtract(shifted, self.noise_center())
        return shifted


def _covariance_of(
    subset: Any, space: HilbertSpace, solver_for: Callable[[HilbertSpace], LinearSolver]
) -> LinearOperator:
    """A ball's or ellipsoid's covariance, inverting an ellipsoid's precision if it has no other."""
    if isinstance(subset, Ball):
        return subset.covariance
    if isinstance(subset, Ellipsoid):
        if subset.covariance is not None:
            return subset.covariance
        return solver_for(space)(subset.precision.with_traits(Traits.POSITIVE_DEFINITE))
    raise TypeError("Only a ball or an ellipsoid has a covariance to read.")


def _precision_of(subset: Any, space: HilbertSpace, name: str) -> LinearOperator:
    """A ball's or ellipsoid's precision; a zero-radius ball has none."""
    if isinstance(subset, Ball):
        if subset.radius <= 0.0:
            raise ValueError(f"The {name} ball needs a positive radius.")
        return subset.precision
    if isinstance(subset, Ellipsoid):
        return subset.precision
    raise TypeError(f"The {name} must be a ball or an ellipsoid to have a precision.")


# --------------------------------------------------------------------- #
#                               Algorithms                               #
# --------------------------------------------------------------------- #


class _ByIdentity:
    """A cache key that holds a vector and hashes on its identity."""

    __slots__ = ("vector",)

    def __init__(self, vector: Any) -> None:
        self.vector = vector

    def __hash__(self) -> int:
        return id(self.vector)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ByIdentity) and other.vector is self.vector


def _require_property_coordinates(space: HilbertSpace, what: str) -> None:
    """Refuse a membership-side probe on a property space without coordinates.

    The support side of the feasible set is coordinate-free on every space.
    The membership side forms matrices of the property's dimension (the
    closed form's shape, the inclusion search's kernel Gram, the outer
    ellipsoids), and so needs a coordinate map on the property space.

    Raises:
        TypeError: naming the probe and the space.
    """
    if not isinstance(space, CoordinateSpace):
        raise TypeError(
            f"{what} forms matrices of the property's dimension and needs a "
            f"coordinate map on the property space; {type(space).__name__} "
            "provides none. The support side of the set (support values, "
            "brackets, polytopes, extremal models) is coordinate-free."
        )


class Algorithm:
    """A way of probing a feasible property set, holding tuning only.

    An algorithm declares the forms it accepts and answers the probes it
    can: ``support`` in a direction, ``inclusion`` of a value, and
    ``feasibility`` of the whole set. Instances carry tolerances and
    budgets and nothing else, so two instances with the same settings are
    interchangeable and the set may share work between them. A probe takes
    one by name (its ``name``) or as an instance through ``via=``. An
    algorithm with ``automatic`` false is never chosen for the caller and
    answers only when asked for by name or instance: the place for a bound
    that is cheap but not sharp.
    """

    name: str = ""
    probes: frozenset[str] = frozenset()
    automatic: bool = True

    @property
    def settings(self) -> tuple[Any, ...]:
        """The tuning this instance carries, by value; instances with equal settings share results."""
        return ()

    def accepts(self, structure: FeasibleStructure) -> bool:
        """Whether this algorithm applies to sets of this structure.

        Raises:
            NotImplementedError: on the base class.
        """
        raise NotImplementedError

    def support(
        self, feasible: "FeasiblePropertySet", direction: Any, /
    ) -> SupportResult:
        """The support of the set in a direction.

        Raises:
            NotImplementedError: if this algorithm does not compute it.
        """
        raise NotImplementedError(f"{self.name} does not compute support values.")

    def inclusion(
        self, feasible: "FeasiblePropertySet", value: Any, /
    ) -> InclusionResult:
        """The prior's level function at the model fitting a value.

        Raises:
            NotImplementedError: if this algorithm does not decide it.
        """
        raise NotImplementedError(f"{self.name} does not decide membership.")

    def feasibility(self, feasible: "FeasiblePropertySet", /) -> InclusionResult:
        """The prior's level function at the model fitting the data alone.

        Raises:
            NotImplementedError: if this algorithm does not decide it.
        """
        raise NotImplementedError(f"{self.name} does not decide feasibility.")

    def level_gradient(
        self, feasible: "FeasiblePropertySet", value: Any, /
    ) -> Any | None:
        """The gradient of the inclusion level in the value, or None where the algorithm has none."""
        return None

    def outer_ellipsoid(
        self, feasible: "FeasiblePropertySet", criterion: str, /
    ) -> EllipsoidResult:
        """An ellipsoid containing the set, chosen by a criterion.

        Raises:
            NotImplementedError: if this algorithm does not build one.
        """
        raise NotImplementedError(f"{self.name} does not build outer ellipsoids.")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class ClosedForm(Algorithm):
    """Exact data and a quadratic prior: the set is an ellipsoid, written down.

    With the prior ``{(C⁻¹(m − a), m − a) <= 1}`` and the data exact, the
    model is ``a + ũ + u₀`` with ``ũ == C A*(A C A*)⁻¹ (d − A a)`` the
    smallest fitting correction in the prior's inner product and ``u₀`` in
    the kernel of ``A``, orthogonal to ``ũ`` in that inner product. What is
    left of the prior's budget, ``1 − (C⁻¹ũ, ũ)``, bounds ``u₀``, and the
    image of that bounded kernel under ``T`` is the ellipsoid

    .. code-block:: text

        { p : (S⁻¹ (p − c), p − c) <= budget },   c == T (a + ũ),
        S == T C T* − T C A* (A C A*)⁻¹ A C T*

    Al-Attar (2021) eq. (2.84), there for a ball, where ``C == r² I``. The
    shape ``S`` costs ``dim P`` data-space solves and is shared by every data
    set; each data set costs one more solve for ``ũ``. Every probe is then
    algebra on the property space: the support in a direction, the
    membership of a value and its fitting model, the nearest point.
    """

    name = "closed_form"
    probes = frozenset({"support", "inclusion", "feasibility", "outer_ellipsoid"})

    def accepts(self, structure: FeasibleStructure) -> bool:
        """Exact data and a quadratic prior."""
        return structure.form == "closed_form"

    # -- the data-independent part ---------------------------------------

    def _pieces(self, structure: FeasibleStructure) -> dict[str, Any]:
        return structure.shared(("closed_form",), lambda: self._build(structure))

    def _build(self, structure: FeasibleStructure) -> dict[str, Any]:
        _require_property_coordinates(structure.property_space, "the closed form")
        forward, T = structure.forward, structure.property_operator
        covariance = structure.prior_covariance()
        normal = (forward @ covariance @ forward.adjoint).with_traits(
            Traits.POSITIVE_DEFINITE
        )
        normal_inverse = structure.solver_for(structure.data_space)(normal)
        space = structure.property_space
        model_space = structure.model_space
        # S e_j == T C T* e_j − T C A* (A C A*)⁻¹ A C T* e_j, one solve each.
        columns = []
        for j in range(space.dim):
            pulled = covariance(T.adjoint(space.basis_vector(j)))
            correction = covariance(forward.adjoint(normal_inverse(forward(pulled))))
            columns.append(
                space.to_components(T(model_space.subtract(pulled, correction)))
            )
        shape = np.column_stack(columns)
        shape = 0.5 * (shape + shape.T) if space.is_orthonormal else shape
        return {
            "covariance": covariance,
            "normal_inverse": normal_inverse,
            "shape": shape,
        }

    # -- the data-dependent part -----------------------------------------

    def _fit(self, feasible: "FeasiblePropertySet") -> dict[str, Any]:
        """``ũ``, the budget and the centre for these data, computed once."""
        return feasible.cached(
            ("closed_form", "fit"), lambda: self._compute_fit(feasible)
        )

    def _compute_fit(self, feasible: "FeasiblePropertySet") -> dict[str, Any]:
        structure = feasible.structure
        pieces = self._pieces(structure)
        forward, T = structure.forward, structure.property_operator
        model_space, data_space = structure.model_space, structure.data_space
        center = structure.prior_center()
        shifted = data_space.subtract(feasible.data, forward(center))
        weights = pieces["normal_inverse"](shifted)
        correction = pieces["covariance"](forward.adjoint(weights))
        # (C⁻¹ ũ, ũ) == (w, A C A* w) == (w, d′): no inverse of C needed.
        used = float(data_space.inner_product(weights, shifted))
        anchor = model_space.add(center, correction)
        return {
            "anchor": anchor,
            "used": used,
            "budget": 1.0 - used,
            "center": T(anchor),
        }

    def ellipsoid(self, feasible: "FeasiblePropertySet") -> Ellipsoid:
        """The set itself, as an ellipsoid on the property space.

        Raises:
            ValueError: if the set is empty.
        """
        fit = self._fit(feasible)
        if fit["budget"] < 0.0:
            raise ValueError(
                "No model within the prior fits these data exactly: the "
                f"smallest fitting model has level {fit['used']:.4g} against "
                "the prior's 1."
            )
        structure = feasible.structure
        space = structure.property_space
        shape = self._pieces(structure)["shape"]
        budget = max(fit["budget"], 0.0)
        covariance = LinearOperator.from_matrix(
            space,
            space,
            budget * shape,
            form="components",
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )
        precision = LinearOperator.from_matrix(
            space,
            space,
            np.linalg.inv(shape) / budget if budget > 0.0 else np.linalg.inv(shape),
            form="components",
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )
        return Ellipsoid(space, precision, center=fit["center"], covariance=covariance)

    def outer_ellipsoid(
        self, feasible: "FeasiblePropertySet", criterion: str, /
    ) -> EllipsoidResult:
        """The set itself, which no criterion can improve on.

        Raises:
            ValueError: if the set is empty.
        """
        _check_criterion(criterion)
        return EllipsoidResult(
            ellipsoid=self.ellipsoid(feasible),
            criterion=criterion,
            converged=True,
            algorithm=self.name,
            evaluations=0,
            message="exact",
        )

    def prior_only(self, structure: FeasibleStructure) -> Ellipsoid:
        """What the prior alone says: ``{ p : ((T C T*)⁻¹ (p − T a), p − T a) <= 1 }``."""
        T = structure.property_operator
        space = structure.property_space
        _require_property_coordinates(space, "the prior's ellipsoid")
        covariance = structure.prior_covariance()
        matrix = np.column_stack(
            [
                space.to_components(T(covariance(T.adjoint(space.basis_vector(j)))))
                for j in range(space.dim)
            ]
        )
        return Ellipsoid(
            space,
            LinearOperator.from_matrix(
                space,
                space,
                np.linalg.inv(matrix),
                form="components",
                traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
            ),
            center=T(structure.prior_center()),
            covariance=LinearOperator.from_matrix(
                space,
                space,
                matrix,
                form="components",
                traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
            ),
        )

    # -- probes -----------------------------------------------------------

    def feasibility(self, feasible: "FeasiblePropertySet", /) -> InclusionResult:
        """The prior's level function at ``a + ũ``: what fitting the data alone costs."""
        fit = self._fit(feasible)
        level = feasible.structure.prior.level
        return InclusionResult(
            level=fit["used"] * level,
            lower_bound=fit["used"] * level,
            converged=True,
            algorithm=self.name,
            evaluations=1,
            fitting_model=fit["anchor"],
            message="exact",
        )

    def _kernel_part(self, feasible: "FeasiblePropertySet", vector: Any) -> Any:
        """``P_K C v`` for a model vector ``v``: the kernel's share, in the prior's metric."""
        structure = feasible.structure
        pieces = self._pieces(structure)
        forward = structure.forward
        image = pieces["covariance"](vector)
        correction = pieces["covariance"](
            forward.adjoint(pieces["normal_inverse"](forward(image)))
        )
        return structure.model_space.subtract(image, correction)

    def support(
        self, feasible: "FeasiblePropertySet", direction: Any, /
    ) -> SupportResult:
        """The support of the ellipsoid, with the model attaining it.

        Raises:
            ValueError: if the set is empty.
        """
        fit = self._fit(feasible)
        structure = feasible.structure
        space = structure.property_space
        model_space = structure.model_space
        T = structure.property_operator
        if fit["budget"] < 0.0:
            raise ValueError("The feasible set is empty; it has no support.")
        pulled = T.adjoint(direction)
        kernel = self._kernel_part(feasible, pulled)
        quadratic = float(model_space.inner_product(kernel, pulled))
        value = float(space.inner_product(direction, fit["center"])) + float(
            np.sqrt(max(fit["budget"] * quadratic, 0.0))
        )
        if quadratic > 0.0 and fit["budget"] > 0.0:
            scale = float(np.sqrt(fit["budget"] / quadratic))
            extremal = model_space.axpy(scale, kernel, model_space.copy(fit["anchor"]))
        else:
            extremal = model_space.copy(fit["anchor"])
        return SupportResult(
            value=value,
            inner=value,
            converged=True,
            algorithm=self.name,
            evaluations=1,
            extremal_model=extremal,
            message="exact",
        )

    def inclusion(
        self, feasible: "FeasiblePropertySet", value: Any, /
    ) -> InclusionResult:
        """The level of the fitting model: ``(C⁻¹ũ, ũ) + (S⁻¹(p − c), p − c)``, in the prior's units."""
        fit = self._fit(feasible)
        structure = feasible.structure
        space = structure.property_space
        shape = self._pieces(structure)["shape"]
        offset = space.to_components(space.subtract(value, fit["center"]))
        # The shape is singular when the kernel of A has fewer dimensions
        # than the property: the set is then flat, and a value off it is
        # unreachable however large the model, which is a proof and not a
        # failure. The least-squares solve drops the shape's null
        # directions, and a residual there says so.
        weights = np.linalg.lstsq(shape, offset, rcond=1e-12)[0]
        size = float(np.linalg.norm(offset))
        if not np.allclose(shape @ weights, offset, rtol=1e-8, atol=1e-12 * size):
            return InclusionResult(
                level=float("inf"),
                lower_bound=float("inf"),
                converged=True,
                algorithm=self.name,
                evaluations=1,
                message="no model has this property",
            )
        level = structure.prior.level
        # (S⁻¹ (p − c), p − c) in the property space's own inner product.
        normalized = fit["used"] + float(
            space.inner_product(
                space.from_components(weights), space.subtract(value, fit["center"])
            )
        )
        T = structure.property_operator
        model_space = structure.model_space
        lifted = self._kernel_part(feasible, T.adjoint(space.from_components(weights)))
        fitting = model_space.add(fit["anchor"], lifted)
        return InclusionResult(
            level=normalized * level,
            lower_bound=normalized * level,
            converged=True,
            algorithm=self.name,
            evaluations=1,
            fitting_model=fitting,
            message="exact",
        )

    def level_gradient(self, feasible: "FeasiblePropertySet", value: Any, /) -> Any:
        """The gradient of the inclusion level in the value: ``2 S⁻¹ (p − c)``, in the prior's units."""
        fit = self._fit(feasible)
        structure = feasible.structure
        space = structure.property_space
        shape = self._pieces(structure)["shape"]
        offset = space.to_components(space.subtract(value, fit["center"]))
        return space.from_components(
            2.0 * structure.prior.level * np.linalg.solve(shape, offset)
        )


# --------------------------------------------------------------------- #
#                           The quadratic tier                           #
# --------------------------------------------------------------------- #


class _QuadraticFamily:
    """The damped family of two quadratic sets, solved along a sweep.

    ``K(γ) == A C A* + γ R`` on the data space, or ``M(γ) == γ C⁻¹ + A* R⁻¹ A``
    on the model space, with ``C`` the prior's covariance (or a restriction
    of it to the kernel of the property operator) and ``R`` the noise's.
    The same interface as a Tikhonov family -- ``solve``, ``right_hand_side``,
    ``model_from`` -- so that the misfit search runs on it unchanged, and
    members carry their factors for the preconditioners.
    """

    def __init__(
        self,
        structure: FeasibleStructure,
        /,
        *,
        formalism: Literal["model_space", "data_space"],
        solver: LinearSolver,
        prior_covariance: LinearOperator | None = None,
    ) -> None:
        self.formalism = formalism
        self.forward = structure.forward
        self.covariance = (
            structure.prior_covariance()
            if prior_covariance is None
            else prior_covariance
        )
        self.noise_covariance = structure.noise_covariance()
        self.noise_precision = structure.noise_precision()
        self.prior_precision = (
            structure.prior_precision() if prior_covariance is None else None
        )
        forward = self.forward
        if formalism == "model_space":
            if self.prior_precision is None:
                raise ValueError(
                    "The model-space formalism needs the prior's precision, "
                    "which a kernel-restricted covariance does not have."
                )
            base = forward.adjoint @ self.noise_precision @ forward
            shift = self.prior_precision
        else:
            base = forward @ self.covariance @ forward.adjoint
            shift = self.noise_covariance
        self.solves = DampedSolves(
            base.with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE),
            shift.with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE),
            solver,
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
            assemble=self.member,
        )

    def member(self, multiplier: float, /) -> DampedNormalOperator:
        """The member at one multiplier, carrying its factors."""
        return DampedNormalOperator(
            self.forward,
            self.covariance,
            self.noise_covariance,
            multiplier,
            prior_precision=self.prior_precision,
            noise_precision=self.noise_precision,
            formalism=self.formalism,
        )

    def right_hand_side(self, shifted: Any, /) -> Any:
        """The right-hand side for the shifted data: ``d′`` or ``A* R⁻¹ d′``."""
        if self.formalism == "model_space":
            return self.forward.adjoint(self.noise_precision(shifted))
        return shifted

    def model_from(self, solution: Any, /) -> Any:
        """The model (relative to the prior's centre) from a solve's solution."""
        if self.formalism == "model_space":
            return solution
        return self.covariance(self.forward.adjoint(solution))

    def solve(
        self, multiplier: float, right_hand_side: Any, /, *, x0: Any = None
    ) -> Any:
        """One member's solve.

        Args:
            multiplier: which member.
            right_hand_side: the vector to solve against.
            x0: a starting guess, usually a neighbouring member's solution.

        Returns:
            The solver's result.
        """
        return self.solves.solve(multiplier, right_hand_side, x0=x0)


def _quadratic_family(
    structure: FeasibleStructure,
    /,
    *,
    formalism: Literal["model_space", "data_space"],
    solver: LinearSolver | None,
) -> _QuadraticFamily:
    """The damped family in one formalism, shared by every algorithm that sweeps it."""
    if solver is None:
        solver = structure.solver_for(
            structure.model_space
            if formalism == "model_space"
            else structure.data_space
        )
    return structure.shared(
        ("family", formalism, id(solver)),
        lambda: _QuadraticFamily(structure, formalism=formalism, solver=solver),
    )


def _check_criterion(criterion: str) -> None:
    if criterion not in ("trace", "volume"):
        raise ValueError(f"The criterion is 'trace' or 'volume', got {criterion!r}.")


def _dual_cost(
    structure: FeasibleStructure, data: Any, direction: Any, certificate: Any
) -> float:
    """``(λ, d) + h_prior(T*q − A*λ) + h_noise(−λ)``: the outer bound a certificate gives.

    Valid for any certificate by weak duality. Closed form on quadratic
    sets; through the sets' support functions otherwise, exact data being
    the point ``{0}`` whose support function vanishes.
    """
    model_space, data_space = structure.model_space, structure.data_space
    pulled = structure.property_operator.adjoint(direction)
    residual = model_space.subtract(pulled, structure.forward.adjoint(certificate))
    if not structure.quadratic:
        prior_term = float(_support_of(structure.prior)(residual))
        noise_term = (
            0.0
            if structure.noise is None
            else float(_support_of(structure.noise)(data_space.negative(certificate)))
        )
        return (
            float(data_space.inner_product(certificate, data)) + prior_term + noise_term
        )
    prior_term = model_space.inner_product(residual, structure.prior_center()) + float(
        np.sqrt(
            max(
                model_space.inner_product(
                    residual, structure.prior_covariance()(residual)
                ),
                0.0,
            )
        )
    )
    noise_term = -data_space.inner_product(
        certificate, structure.noise_center()
    ) + float(
        np.sqrt(
            max(
                data_space.inner_product(
                    certificate, structure.noise_covariance()(certificate)
                ),
                0.0,
            )
        )
    )
    return float(data_space.inner_product(certificate, data)) + prior_term + noise_term


def _largest_feasible(
    structure: FeasibleStructure, shifted: Any, u1: Any, u2: Any
) -> Any | None:
    """``u₂ + x u₁`` at the largest ``x >= 0`` keeping both constraints, or None.

    Along ``x == 1/t`` at a fixed ``γ`` both the prior's level and the misfit
    are quadratics in ``x``, so the largest admissible ``x`` has closed-form
    roots, and the model there is exactly feasible with a pairing first
    order in the error of ``γ``. None when ``u₂`` itself violates a
    constraint, where this ray never enters the set.
    """
    model_space, data_space = structure.model_space, structure.data_space
    Q, P, A = (
        structure.prior_precision(),
        structure.noise_precision(),
        structure.forward,
    )

    def largest(f0: float, cross: float, f2: float) -> float | None:
        # f(x) == f0 + 2 x cross + x² f2 <= 1 on [0, x_max].
        if f0 > 1.0 + 1e-12:
            return None
        if f2 <= 0.0:
            return float("inf") if cross <= 0.0 else (1.0 - f0) / (2.0 * cross)
        disc = cross * cross - f2 * (f0 - 1.0)
        return max((-cross + float(np.sqrt(max(disc, 0.0)))) / f2, 0.0)

    q2, q1 = Q(u2), Q(u1)
    x_prior = largest(
        model_space.inner_product(q2, u2),
        model_space.inner_product(q2, u1),
        model_space.inner_product(q1, u1),
    )
    r2 = data_space.subtract(shifted, A(u2))
    r1 = A(u1)
    x_misfit = largest(
        data_space.inner_product(P(r2), r2),
        -data_space.inner_product(P(r2), r1),
        data_space.inner_product(P(r1), r1),
    )
    if x_prior is None or x_misfit is None:
        return None
    x = min(x_prior, x_misfit)
    if not np.isfinite(x):
        return None
    return model_space.axpy(x, u1, model_space.copy(u2))


def _scaled_inside(
    structure: FeasibleStructure, shifted: Any, anchor: Any, model: Any
) -> Any:
    """The point of the segment from *anchor* to *model* nearest *model* that lies in both sets.

    Both sets are quadratic, so each constraint along the segment is a
    quadratic in the parameter with closed-form roots; *anchor* must satisfy
    both. Relative to the prior's centre.
    """
    model_space, data_space = structure.model_space, structure.data_space
    Q, P, A = (
        structure.prior_precision(),
        structure.noise_precision(),
        structure.forward,
    )
    step = model_space.subtract(model, anchor)
    theta = 1.0

    def largest(f_anchor: float, cross: float, f_step: float) -> float:
        # f(anchor + θ step) == f_anchor + 2 θ cross + θ² f_step <= 1
        if f_step <= 0.0:
            return 1.0
        disc = cross * cross - f_step * (f_anchor - 1.0)
        if disc < 0.0:
            return 0.0
        return max((-cross + float(np.sqrt(disc))) / f_step, 0.0)

    q_anchor = model_space.inner_product(Q(anchor), anchor)
    q_step = model_space.inner_product(Q(step), step)
    q_cross = model_space.inner_product(Q(anchor), step)
    theta = min(theta, largest(q_anchor, q_cross, q_step))
    r_anchor = data_space.subtract(shifted, A(anchor))
    r_step = A(step)
    m_anchor = data_space.inner_product(P(r_anchor), r_anchor)
    m_step = data_space.inner_product(P(r_step), r_step)
    m_cross = -data_space.inner_product(P(r_anchor), r_step)
    theta = min(theta, largest(m_anchor, m_cross, m_step))
    theta = min(max(theta, 0.0), 1.0)
    return model_space.axpy(theta, step, model_space.copy(anchor))


class Search(Algorithm):
    """Membership and feasibility on two quadratic sets, by one multiplier.

    The smallest prior level over the models fitting the data within the
    noise set is a damped least-squares problem: at a multiplier ``η`` the
    minimizer of ``η (C⁻¹u, u) + (R⁻¹(d′ − Au), d′ − Au)`` is one solve of the
    family, its misfit rises with ``η``, and the multiplier at which the
    misfit reaches the noise level is a monotone root find with every solve
    warm-started from the last. The multiplier also certifies the answer
    from below, through the dual of the same problem.

    A property value restricts the model to ``ũ_p + ker T``, with ``ũ_p``
    the smallest model having the value, and the same search runs on the
    kernel: the prior's covariance restricted to it is a rank-``dim P``
    downdate ``C − C T*(T C T*)⁻¹ T C``, applied matrix-free, and the data
    are shifted by ``A ũ_p``. That search is assembled in the data space,
    where the restriction is the downdate; the formalism applies to the
    feasibility search. Either formalism serves whatever the two
    dimensions: the family is positive definite at every positive
    multiplier, the search probes no other, and a range walked to its end
    without a fit is reported as no fit.
    """

    name = "search"
    probes = frozenset({"inclusion", "feasibility"})

    def __init__(
        self,
        /,
        *,
        formalism: Literal["model_space", "data_space"] = "data_space",
        max_iterations: int = 60,
        rtol: float = 1e-6,
        solver: LinearSolver | None = None,
        floor: float = 1e-6,
    ) -> None:
        """
        Args:
            formalism: which space the feasibility search solves in.
            max_iterations: the root find's budget.
            rtol: the root find's bracket tolerance.
            solver: how the family is inverted; the set's solver if omitted.
            floor: the smallest multiplier the search probes when the
                data-space family is rank-deficient (more data than the
                model, or than the model less the property when restricted
                to a value), as a fraction of the family's scale
                ``(A C A* d, d) / (R d, d)`` on the data it fits. The misfit
                falls as the multiplier does and tends to its infimum at
                zero, where such a family is singular; a walk that reaches
                the floor without a fit reports that none exists, which is
                exact to the order of the floor. A family of full rank
                stays well conditioned as the multiplier vanishes and is
                walked without a floor.
        """
        if formalism not in ("model_space", "data_space"):
            raise ValueError(
                f"The formalism is 'model_space' or 'data_space', got {formalism!r}."
            )
        if floor < 0.0:
            raise ValueError(f"The floor must be non-negative, got {floor}.")
        self._formalism = formalism
        self._max_iterations = max_iterations
        self._rtol = rtol
        self._solver = solver
        self._floor = floor

    @property
    def settings(self) -> tuple[Any, ...]:
        """The formalism, the budget, the tolerance, the solver and the floor."""
        return (
            self._formalism,
            self._max_iterations,
            self._rtol,
            id(self._solver),
            self._floor,
        )

    def accepts(self, structure: FeasibleStructure) -> bool:
        """Two quadratic sets, or two smooth level functions; the noise not a point."""
        if structure.form == "quadratic":
            return True
        return (
            structure.noise is not None
            and _smooth_level(structure.prior) is not None
            and _smooth_level(structure.noise) is not None
        )

    def _family(
        self, structure: FeasibleStructure, *, restricted: bool
    ) -> _QuadraticFamily:
        if restricted:
            solver = (
                self._solver
                if self._solver is not None
                else structure.solver_for(structure.data_space)
            )
            return structure.shared(
                ("search", "restricted", id(solver)),
                lambda: _QuadraticFamily(
                    structure,
                    formalism="data_space",
                    solver=solver,
                    prior_covariance=_restricted_covariance(structure),
                ),
            )
        solver = (
            self._solver
            if self._solver is not None
            else structure.solver_for(
                structure.model_space
                if self._formalism == "model_space"
                else structure.data_space
            )
        )
        return structure.shared(
            ("family", self._formalism, id(solver)),
            lambda: _QuadraticFamily(
                structure, formalism=self._formalism, solver=solver
            ),
        )

    def _search(
        self,
        structure: FeasibleStructure,
        family: _QuadraticFamily,
        shifted: Any,
        *,
        restricted: bool = False,
    ) -> tuple[RootResult | None, Any, float]:
        """The misfit search on shifted data: ``(root, model, misfit)``, the root None when nothing need be solved."""
        data_space = structure.data_space
        P, A = structure.noise_precision(), structure.forward
        if data_space.inner_product(P(shifted), shifted) <= 1.0:
            return (
                None,
                structure.model_space.zero(),
                float(data_space.inner_product(P(shifted), shifted)),
            )

        def misfit(model: Any) -> float:
            residual = data_space.subtract(shifted, A(model))
            return float(data_space.inner_product(P(residual), residual))

        def evaluate(multiplier: float, previous: Any) -> Evaluation:
            result = family.solve(
                multiplier, family.right_hand_side(shifted), x0=previous
            )
            return Evaluation(
                value=misfit(family.model_from(result.solution)),
                solution=result.solution,
                iterations=result.iterations,
            )

        minimum = 0.0
        rank = structure.model_space.dim - (
            structure.property_space.dim if restricted else 0
        )
        if (
            self._floor > 0.0
            and family.formalism == "data_space"
            and data_space.dim > rank
        ):
            # A rank-deficient family is singular at a multiplier of zero.
            # The family's scale on the data it fits; the walk stops at this
            # fraction of it rather than probing a singular member.
            weight = float(
                data_space.inner_product(family.noise_covariance(shifted), shifted)
            )
            if weight > 0.0:
                image = family.forward(
                    family.covariance(family.forward.adjoint(shifted))
                )
                minimum = (
                    self._floor
                    * float(data_space.inner_product(image, shifted))
                    / weight
                )
        try:
            root = monotone_root(
                evaluate,
                1.0,
                decreasing=False,
                initial=max(1.0, 10.0 * minimum),
                max_iterations=self._max_iterations,
                rtol=self._rtol,
                atol=0.0,
                minimum=minimum,
            )
        except ConvergenceError as error:
            raise ValueError(
                f"The damped family could not be solved at its first multiplier: "
                f"{error}. A tighter solver, a preconditioner, or the dense "
                "spectral solver (solver=SpectralFamilySolver()) may be needed."
            ) from error
        model = family.model_from(root.solution)
        return root, model, float(root.value)

    def _result(
        self,
        structure: FeasibleStructure,
        root: RootResult | None,
        relative: Any,
        misfit: float,
        *,
        extra_level: float,
        offset: Any,
    ) -> InclusionResult:
        model_space = structure.model_space
        level = structure.prior.level
        Q = structure.prior_precision()
        own = float(model_space.inner_product(Q(relative), relative))
        fitting = model_space.add(
            structure.prior_center(), model_space.add(offset, relative)
        )
        if root is None:
            return InclusionResult(
                level=(extra_level + own) * level,
                lower_bound=(extra_level + own) * level,
                converged=True,
                algorithm=self.name,
                evaluations=0,
                fitting_model=fitting,
                multiplier=0.0,
                message="the data are fitted without a correction",
            )
        if root.exhausted == "low":
            return InclusionResult(
                level=float("inf"),
                lower_bound=float("inf"),
                converged=True,
                algorithm=self.name,
                evaluations=root.evaluations,
                multiplier=float(root.argument),
                message="no model fits the data",
            )
        # The dual of min f s.t. g <= 1 at multiplier 1/η: f + (g − 1)/η <= min;
        # capped at the value itself, which rounding can otherwise cross.
        lower = min(own + (misfit - 1.0) / float(root.argument), own)
        return InclusionResult(
            level=(extra_level + own) * level,
            lower_bound=(extra_level + lower) * level,
            converged=bool(root.converged),
            algorithm=self.name,
            evaluations=root.evaluations,
            fitting_model=fitting,
            multiplier=float(root.argument),
            message="misfit search converged"
            if root.converged
            else "misfit search did not converge",
        )

    # -- general sets: a convex minimization inside the one-multiplier search --

    def _optimizer(self, prior: Functional, noise: Functional) -> Any:
        from ..numerics.optimization import LBFGS, NewtonCG

        if prior.has_hessian and noise.has_hessian:
            return NewtonCG(forcing=1e-10, rtol=1e-12, gtol=0.0, ftol=1e-15)
        return LBFGS(rtol=1e-12, gtol=0.0, max_iterations=2000)

    def _general_fit(
        self,
        feasible: "FeasiblePropertySet",
        anchor: Any,
        projector: LinearOperator | None,
    ) -> InclusionResult:
        """``min f(m)`` over ``m == anchor + P v`` with ``g(d − A m) <= b``, by one multiplier.

        The misfit's infimum over the subspace is found first: above the
        level, nothing fits; at it, that minimizer is the answer; below it,
        the multiplier at which the misfit of the minimizer of ``f/η + g``
        reaches the level is bracketed by a monotone search, each probe a
        convex minimization started cold.
        """
        structure = feasible.structure
        space, data_space = structure.model_space, structure.data_space
        forward, data = structure.forward, feasible.data
        f, a = _smooth_level(structure.prior)
        g, b = _smooth_level(structure.noise)
        optimizer = self._optimizer(f, g)
        if projector is None:
            projector = LinearOperator.identity(space)

        def model_of(v: Any) -> Any:
            return space.add(anchor, projector(v))

        def misfit_point(m: Any) -> Any:
            return data_space.subtract(data, forward(m))

        def objective(multiplier: float | None) -> Functional:
            weight = 1.0 if multiplier is None else 1.0 / multiplier

            def value(v: Any) -> float:
                m = model_of(v)
                fit = float(g(misfit_point(m)))
                return fit if multiplier is None else weight * float(f(m)) + fit

            def gradient(v: Any) -> Any:
                m = model_of(v)
                pulled = forward.adjoint(g.gradient(misfit_point(m)))
                if multiplier is None:
                    return projector(space.negative(pulled))
                return projector(
                    space.axpy(-1.0, pulled, space.scale(weight, f.gradient(m)))
                )

            hessian = None
            if g.has_hessian and (multiplier is None or f.has_hessian):

                def hessian(v: Any) -> LinearOperator:
                    m = model_of(v)
                    curvature = g.hessian(misfit_point(m))
                    pulled = (
                        projector @ forward.adjoint @ curvature @ forward @ projector
                    )
                    if multiplier is None:
                        return pulled.with_traits(Traits.POSITIVE_SEMIDEFINITE)
                    own = projector @ f.hessian(m) @ projector
                    return (own * weight + pulled).with_traits(
                        Traits.POSITIVE_SEMIDEFINITE
                    )

            return Functional.from_callables(
                space, value, gradient=gradient, hessian=hessian
            )

        start = space.zero()
        limit = optimizer.minimize(objective(None), start)
        floor = float(limit.value)
        evaluations = int(limit.evaluations)
        if floor > b * (1.0 + self._rtol):
            return InclusionResult(
                level=float("inf"),
                lower_bound=float("inf"),
                converged=bool(limit.converged),
                algorithm=self.name,
                evaluations=evaluations,
                message="no model fits the data",
            )
        if floor >= b * (1.0 - self._rtol):
            model = model_of(limit.minimizer)
            level = float(f(model))
            return InclusionResult(
                level=level,
                lower_bound=level,
                converged=bool(limit.converged),
                algorithm=self.name,
                evaluations=evaluations,
                fitting_model=model,
                multiplier=float("inf"),
                message="the data constraint is tight",
            )

        def probe(multiplier: float, previous: Any) -> Evaluation:
            result = optimizer.minimize(objective(multiplier), start)
            m = model_of(result.minimizer)
            return Evaluation(
                float(g(misfit_point(m))), result.minimizer, result.evaluations
            )

        root = monotone_root(
            probe,
            b,
            decreasing=True,
            max_iterations=self._max_iterations,
            rtol=self._rtol,
            expansions=40,
            warm_start=False,
        )
        if root.breakdown is not None:
            raise root.breakdown
        model = model_of(root.solution)
        level = float(f(model))
        misfit = float(root.value)
        # The dual of min f s.t. g <= b at multiplier η: f + η (g − b) <= min.
        lower = min(level + float(root.argument) * (misfit - b), level)
        return InclusionResult(
            level=level,
            lower_bound=lower,
            converged=bool(root.converged),
            algorithm=self.name,
            evaluations=evaluations + root.inner_iterations,
            fitting_model=model,
            multiplier=float(root.argument),
            message=(
                "multiplier search converged"
                if root.converged
                else "multiplier search did not converge"
            ),
        )

    def _general_anchor(
        self, structure: FeasibleStructure, value: Any
    ) -> tuple[Any, LinearOperator]:
        """A model with the property, and the projector onto the property's kernel."""
        from .subspaces import OrthogonalProjector

        T = structure.property_operator
        space = structure.property_space
        _require_property_coordinates(space, "the inclusion search")
        solver = structure.solver_for(structure.data_space)

        def build() -> dict[str, Any]:
            gram = np.column_stack(
                [
                    space.to_components(T(T.adjoint(space.basis_vector(j))))
                    for j in range(space.dim)
                ]
            )
            return {
                "inverse": np.linalg.inv(gram),
                "projector": OrthogonalProjector.onto_kernel(T, solver=solver),
            }

        pieces = structure.shared(("search", "general_kernel"), build)
        weights = pieces["inverse"] @ space.to_components(value)
        return T.adjoint(space.from_components(weights)), pieces["projector"]

    def feasibility(self, feasible: "FeasiblePropertySet", /) -> InclusionResult:
        """The smallest prior level over the models fitting the data."""
        structure = feasible.structure
        if not structure.quadratic:
            return self._general_fit(feasible, structure.model_space.zero(), None)
        shifted = structure.shifted_data(feasible.data)
        _check_floor(structure, shifted)
        family = self._family(structure, restricted=False)
        root, model, misfit = self._search(structure, family, shifted)
        return self._result(
            structure,
            root,
            model,
            misfit,
            extra_level=0.0,
            offset=structure.model_space.zero(),
        )

    def inclusion(
        self, feasible: "FeasiblePropertySet", value: Any, /
    ) -> InclusionResult:
        """The smallest prior level over the models with this value fitting the data."""
        structure = feasible.structure
        if not structure.quadratic:
            anchor, projector = self._general_anchor(structure, value)
            return self._general_fit(feasible, anchor, projector)
        data_space, space = structure.data_space, structure.property_space
        pieces = _kernel_pieces(structure)
        offset_p = space.subtract(
            value, structure.property_operator(structure.prior_center())
        )
        weights = np.linalg.solve(pieces["gram"], space.to_components(offset_p))
        lifted = pieces["covariance"](
            structure.property_operator.adjoint(space.from_components(weights))
        )
        anchor_level = float(
            space.inner_product(space.from_components(weights), offset_p)
        )
        shifted = data_space.subtract(
            structure.shifted_data(feasible.data), structure.forward(lifted)
        )
        family = self._family(structure, restricted=True)
        root, kernel_model, misfit = self._search(
            structure, family, shifted, restricted=True
        )
        return self._result(
            structure,
            root,
            kernel_model,
            misfit,
            extra_level=anchor_level,
            offset=lifted,
        )


def _kernel_pieces(structure: FeasibleStructure) -> dict[str, Any]:
    """``S == T C T*`` as a matrix on the property space, and ``C`` itself, shared."""

    def build() -> dict[str, Any]:
        space = structure.property_space
        _require_property_coordinates(space, "the inclusion search")
        T = structure.property_operator
        covariance = structure.prior_covariance()
        gram = np.column_stack(
            [
                space.to_components(T(covariance(T.adjoint(space.basis_vector(j)))))
                for j in range(space.dim)
            ]
        )
        return {"gram": gram, "covariance": covariance}

    return structure.shared(("kernel",), build)


def _restricted_covariance(structure: FeasibleStructure) -> LinearOperator:
    """``C − C T* S⁻¹ T C``: the prior's covariance restricted to the kernel of ``T``."""
    pieces = _kernel_pieces(structure)
    covariance, T = pieces["covariance"], structure.property_operator
    space, model_space = structure.property_space, structure.model_space
    inverse = np.linalg.inv(pieces["gram"])

    def apply(x: Any) -> Any:
        image = covariance(x)
        pulled = T(image)
        back = space.from_components(inverse @ space.to_components(pulled))
        return model_space.subtract(image, covariance(T.adjoint(back)))

    return LinearOperator.self_adjoint(
        model_space, apply, traits=Traits.POSITIVE_SEMIDEFINITE
    )


def _check_floor(structure: FeasibleStructure, shifted: Any) -> None:
    """Refuse a noise set the solver cannot resolve.

    The misfit is measured on the model a solve returns, so it carries the
    solve's residual, ``rtol`` times the data in the noise metric; a noise
    radius below that is one no fit can be certified to.
    """
    import warnings

    from ..numerics.solvers import IterativeSolver

    solver = structure.solver_for(structure.data_space)
    if not isinstance(solver, IterativeSolver):
        return  # a direct solver has no residual floor to speak of
    rtol = solver.rtol
    size = float(
        np.sqrt(
            max(
                structure.data_space.inner_product(
                    structure.noise_precision()(shifted), shifted
                ),
                0.0,
            )
        )
    )
    floor = rtol * size
    if floor >= 0.1:
        raise ValueError(
            f"The solver's tolerance ({rtol:.1e}) times the data's size in the "
            f"noise metric ({size:.3g}) is {floor:.2g} against a noise level of "
            "1: the noise set is below the solver's floor and no fit to it can "
            "be certified. Tighten the solver or widen the noise set."
        )
    if floor >= 1e-3:
        warnings.warn(
            f"The solver's floor ({floor:.2g} of the noise level) is close to "
            "the noise set; the feasibility decision carries that error.",
            RuntimeWarning,
            stacklevel=3,
        )


class Primal(Algorithm):
    """The support of the set on two quadratic sets, by one multiplier.

    Attaching multipliers ``s`` and ``t`` to the prior and the data
    constraints, the stationarity condition ``(s C⁻¹ + t A* R⁻¹ A) u == T*q
    + t A* R⁻¹ d′`` depends on ``γ == s/t`` through the family ``M(γ)``
    alone, and the dual minimized over ``t`` is closed form:
    ``φ(γ) == B + sqrt(A C)`` with ``A == (c, u₁)``, ``B == (c, u₂)``,
    ``C == (A* R⁻¹ d′, u₂) + γ + 1 − (R⁻¹ d′, d′)``, from the two solves
    ``u₁ == M(γ)⁻¹ c`` and ``u₂ == M(γ)⁻¹ A* R⁻¹ d′``. The support is the
    minimum of ``φ`` over ``γ``, a scalar minimization in ``log γ`` of a
    function that is unimodal, with ``γ == ∞`` the prior-only bound (taken
    with no solve when the prior's own support point fits the data) and
    ``γ == 0`` the data-only bound, available when ``A* R⁻¹ A`` is
    invertible. In the data-space formalism the coefficients are evaluated
    through the resolvent identity, with the kernel part of ``c`` computed
    once per direction at a small multiplier ``γ₀`` -- a fraction of the
    family's own scale rather than zero, because the undamped member
    ``A C A*`` is the worst-conditioned of the family and the identity
    holds at any ``γ₀`` -- so that nothing cancels.

    At the minimum the model is ``u₁/t* + u₂`` with ``t* == sqrt(A/C)``; the
    certificate ``λ == t* R⁻¹(A u − d′)`` gives the outer bound by weak
    duality, evaluated in closed form, and the model scaled into both sets
    gives the inner one, so every answer carries its bracket. ``converged``
    reports the scalar minimization; the bracket's width is the gap.
    """

    name = "primal"
    probes = frozenset({"support"})

    def __init__(
        self,
        /,
        *,
        formalism: Literal["model_space", "data_space"] = "data_space",
        tolerance: float = 1e-4,
        max_iterations: int = 100,
        solver: LinearSolver | None = None,
        floor: float = 1e-6,
    ) -> None:
        """
        Args:
            formalism: which space the family is solved in.
            tolerance: on ``log γ`` at the minimum. The outer value is
                stationary there, so its error is second order in this; the
                inner value's is first order, which is the bracket's width.
            max_iterations: the minimization's budget of evaluations.
            solver: how the family is inverted; the set's solver if omitted.
            floor: in the data-space formalism, the multiplier at which the
                kernel part of a direction is computed, as a fraction of
                the family's scale ``(A C A* v, v) / (R v, v)``. Zero asks
                for the undamped member, which a direct or spectral solver
                inverts exactly and conjugate gradients may not; the
                result does not depend on it beyond the solver's accuracy.
                On a family with more data than model coefficients it is
                also where the walk of the multiplier stops, since the
                member is singular at zero; a slack prior then gives an
                outer value above the limit by a few hundred times the
                floor, and the model-space formalism gives the limit.
        """
        if formalism not in ("model_space", "data_space"):
            raise ValueError(
                f"The formalism is 'model_space' or 'data_space', got {formalism!r}."
            )
        if floor < 0.0:
            raise ValueError(f"The floor must be non-negative, got {floor}.")
        self._formalism = formalism
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        self._solver = solver
        self._floor = floor

    @property
    def settings(self) -> tuple[Any, ...]:
        """The formalism, the tolerance, the budget, the solver and the floor."""
        return (
            self._formalism,
            self._tolerance,
            self._max_iterations,
            id(self._solver),
            self._floor,
        )

    def accepts(self, structure: FeasibleStructure) -> bool:
        """Two quadratic sets, the noise not a point."""
        return structure.form == "quadratic"

    def _family(self, structure: FeasibleStructure) -> _QuadraticFamily:
        return _quadratic_family(
            structure, formalism=self._formalism, solver=self._solver
        )

    def support(
        self, feasible: "FeasiblePropertySet", direction: Any, /
    ) -> SupportResult:
        """The support in a direction, by minimizing ``φ(γ)``.

        Either formalism serves whatever the two dimensions: in the data
        space the kernel part of a direction is taken at the floor
        multiplier rather than at zero, where the family is singular when
        the data outnumber the model, and on such a family the walk of the
        multiplier stops at the floor too, so a slack prior gives a bound
        within the order of the floor of the limit rather than a solve of a
        singular member. The model-space formalism reaches the limit
        exactly where that matters.

        Raises:
            ValueError: if the set is empty.
        """
        structure = feasible.structure
        model_space, data_space, space = (
            structure.model_space,
            structure.data_space,
            structure.property_space,
        )
        if feasible.is_empty():
            raise ValueError("The feasible set is empty; it has no support.")
        A, T = structure.forward, structure.property_operator
        C, P, R = (
            structure.prior_covariance(),
            structure.noise_precision(),
            structure.noise_covariance(),
        )
        a = structure.prior_center()
        shifted = structure.shifted_data(feasible.data)
        c = T.adjoint(direction)
        centre_term = float(space.inner_product(direction, T(a)))
        evaluations = 0

        # Prior-only: the prior's own support point, if it fits.
        image = C(c)
        prior_norm = float(np.sqrt(max(model_space.inner_product(image, c), 0.0)))
        if prior_norm == 0.0:
            return SupportResult(
                value=centre_term,
                inner=centre_term,
                converged=True,
                algorithm=self.name,
                evaluations=0,
                extremal_model=model_space.copy(a),
                message="the property is constant",
            )
        flat = model_space.scale(1.0 / prior_norm, image)
        residual = data_space.subtract(shifted, A(flat))
        if data_space.inner_product(P(residual), residual) <= 1.0:
            certificate = data_space.zero()
            outer = _dual_cost(structure, feasible.data, direction, certificate)
            return SupportResult(
                value=outer,
                inner=centre_term + prior_norm,
                converged=True,
                algorithm=self.name,
                evaluations=0,
                certificate=certificate,
                extremal_model=model_space.add(a, flat),
                multipliers=(1.0 / prior_norm, 0.0),
                message="the data constraint is slack",
            )

        family = self._family(structure)
        rhs_data = family.right_hand_side(shifted)
        data_term = float(data_space.inner_product(P(shifted), shifted))
        cache = feasible.cached(("primal", "u2", self._formalism), dict)

        if self._formalism == "data_space":
            v = A(image)  # A C c
            # The kernel part of the direction, through the family at a
            # small multiplier γ₀: kappa₀ == (c, C c) − (v, K(γ₀)⁻¹ v), and
            # (c, C c) − (v, K(γ)⁻¹ v) == kappa₀ + (γ − γ₀)(K(γ₀)⁻¹ v, R K(γ)⁻¹ v)
            # by the resolvent identity, exactly, for any γ₀.
            weight = float(data_space.inner_product(R(v), v))
            gamma_floor = (
                self._floor
                * float(data_space.inner_product(A(C(A.adjoint(v))), v))
                / weight
                if weight > 0.0
                else 0.0
            )
            try:
                g_inv_v = family.solve(gamma_floor, v).solution
            except ConvergenceError as error:
                raise ValueError(
                    "The data-space family could not be solved at "
                    f"γ = {gamma_floor:.3g}, where a direction's kernel part is "
                    "computed; a preconditioner, the dense spectral solver "
                    "(solver=SpectralFamilySolver()) or "
                    "Primal(formalism='model_space') may be needed."
                ) from error
            kappa = max(
                float(model_space.inner_product(image, c))
                - float(data_space.inner_product(v, g_inv_v)),
                0.0,
            )
            evaluations += 1

        def solve_u2(gamma: float) -> Any:
            if gamma in cache:
                return cache[gamma]
            # Warm-start from the nearest cached multiplier, within a few
            # decades: further away the scales differ too much to help.
            nearest = (
                min(cache, key=lambda g: abs(np.log(g) - np.log(gamma)))
                if cache and gamma > 0.0
                else None
            )
            if nearest is not None and abs(np.log(nearest) - np.log(gamma)) > 6.0:
                nearest = None
            result = family.solve(
                gamma, rhs_data, x0=None if nearest is None else cache[nearest]
            )
            cache[gamma] = result.solution
            return result.solution

        def coefficients(gamma: float) -> tuple[float, float, float, Any, Any]:
            nonlocal evaluations
            evaluations += 1
            k_d = solve_u2(gamma)
            if self._formalism == "data_space":
                k_v = family.solve(gamma, v, x0=g_inv_v).solution
                a_coef = (
                    kappa
                    + (gamma - gamma_floor)
                    * float(data_space.inner_product(g_inv_v, R(k_v)))
                ) / gamma
                b_coef = float(data_space.inner_product(v, k_d))
                c_coef = 1.0 + gamma * (
                    1.0 - float(data_space.inner_product(shifted, k_d))
                )
                u1 = model_space.scale(
                    1.0 / gamma, model_space.subtract(image, C(A.adjoint(k_v)))
                )
                u2 = C(A.adjoint(k_d))
            else:
                u1 = family.solve(gamma, c).solution
                u2 = k_d
                a_coef = float(model_space.inner_product(c, u1))
                b_coef = float(model_space.inner_product(c, u2))
                c_coef = (
                    float(model_space.inner_product(rhs_data, u2))
                    + gamma
                    + 1.0
                    - data_term
                )
            return a_coef, b_coef, c_coef, u1, u2

        def phi(log_gamma: float) -> float:
            try:
                a_coef, b_coef, c_coef, _, _ = coefficients(float(np.exp(log_gamma)))
            except ConvergenceError as error:
                raise ValueError(
                    f"The family could not be solved at γ = {np.exp(log_gamma):.3g}: "
                    f"{error}. A tighter solver, a preconditioner, or the other "
                    "formalism may be needed."
                ) from error
            if c_coef <= 0.0 or a_coef < 0.0:
                return float("inf")
            return b_coef + float(np.sqrt(a_coef * c_coef))

        # The warm start is per worker: a sweep hands each thread a
        # contiguous run of directions, whose roots are neighbours.
        # On a rank-deficient family (more data than model coefficients) the
        # member is singular at a multiplier of zero, and a walk there, which
        # a slack prior asks for, returns a bracket that holds but says
        # nothing. The walk stops at the floor instead, where the kernel
        # part was taken: the outer value there is a bound, within the order
        # of the floor of the limit.
        lower = (
            float(np.log(gamma_floor))
            if self._formalism == "data_space"
            and gamma_floor > 0.0
            and data_space.dim > model_space.dim
            else None
        )
        log_gamma, value, converged = _minimize_unimodal(
            phi,
            feasible.cached(("primal", "start", threading.get_ident()), lambda: [0.0]),
            tolerance=self._tolerance,
            max_iterations=self._max_iterations,
            lower=lower,
        )
        gamma = float(np.exp(log_gamma))
        if log_gamma <= -_LIMIT + 2.0 and self._formalism == "model_space":
            # The walk reached the lower limit: the prior is slack and the
            # data alone bound the property, at γ == 0 exactly.
            try:
                a0, b0, c0, _, _ = coefficients(0.0)
            except ConvergenceError:
                a0 = c0 = -1.0
            if a0 >= 0.0 and c0 > 0.0:
                gamma = 0.0
        a_coef, b_coef, c_coef, u1, u2 = coefficients(gamma)
        if c_coef <= 0.0:
            raise ValueError("The dual is unbounded below: the feasible set is empty.")
        t_star = float(np.sqrt(a_coef / c_coef)) if a_coef > 0.0 else 0.0
        if t_star > 0.0:
            relative = model_space.axpy(1.0 / t_star, u1, model_space.copy(u2))
        else:
            relative = u2
        residual = data_space.subtract(shifted, A(relative))
        # Stationarity: T*q − A*λ == s C⁻¹ u with λ == t R⁻¹ (A u − d′).
        certificate = data_space.scale(-t_star, P(residual))
        outer = _dual_cost(structure, feasible.data, direction, certificate)
        inside = _largest_feasible(structure, shifted, u1, u2)
        if inside is None:
            anchor = model_space.subtract(feasible.fitting_model(), a)
            inside = _scaled_inside(structure, shifted, anchor, relative)
        inner = centre_term + float(model_space.inner_product(c, inside))
        if np.isfinite(prior_norm):
            outer = min(outer, centre_term + prior_norm)
        return SupportResult(
            value=outer,
            inner=inner,
            converged=converged,
            algorithm=self.name,
            evaluations=evaluations,
            certificate=certificate,
            extremal_model=model_space.add(a, inside),
            multipliers=(gamma * t_star, t_star),
            message="unimodal search converged"
            if converged
            else "evaluation budget exhausted",
        )


class SProcedure(Algorithm):
    """An outer ellipsoid on two quadratic sets, by weighting the two constraints.

    For a weight ``θ`` in ``(0, 1)`` every feasible model satisfies
    ``θ (C⁻¹u, u) + (1 − θ)(R⁻¹(A u − d′), A u − d′) <= 1`` with
    ``u == m − a``, because it satisfies each term's constraint on its own.
    That one inequality is an ellipsoid in the model space, centred at the
    damped estimate ``u_θ == C_θ (1 − θ) A* R⁻¹ d′`` with
    ``C_θ == (θ C⁻¹ + (1 − θ) A* R⁻¹ A)⁻¹``, and its image under ``T`` is

    .. code-block:: text

        { p : ((T C_θ T*)⁻¹ (p − T(a + u_θ)), p − T(a + u_θ)) <= 1 − r_θ }

    with ``r_θ`` the weighted sum at its minimum. Every ``θ`` gives a
    certified outer ellipsoid; the weight is chosen to minimize the trace
    or the volume of it, a scalar minimization in ``log γ`` with
    ``γ == θ / (1 − θ)`` of a function that is unimodal, with ``γ == ∞``
    the prior's own image and ``γ == 0`` the data-only ellipsoid where
    ``A* R⁻¹ A`` is invertible. One weight costs ``dim P + 1`` solves of
    the damped family ``A C A* + γ R`` (or ``γ C⁻¹ + A* R⁻¹ A``), the
    family the primal sweeps for support values, warm-started along the
    search. The data enter the constraint, which is what makes this tighter
    than the Minkowski sum a linear certificate gives, and it is checked
    against support values direction by direction.
    """

    name = "s_procedure"
    probes = frozenset({"outer_ellipsoid"})

    def __init__(
        self,
        /,
        *,
        formalism: Literal["model_space", "data_space"] = "data_space",
        tolerance: float = 1e-2,
        max_iterations: int = 60,
        solver: LinearSolver | None = None,
    ) -> None:
        """
        Args:
            formalism: which space the family is solved in.
            tolerance: on ``log γ`` at the minimum. The criterion is
                stationary there, so its error is second order in this,
                and the ellipsoid is certified whatever the weight.
            max_iterations: the minimization's budget of evaluations, each
                ``dim P + 1`` solves.
            solver: how the family is inverted; the set's solver if omitted.
        """
        if formalism not in ("model_space", "data_space"):
            raise ValueError(
                f"The formalism is 'model_space' or 'data_space', got {formalism!r}."
            )
        self._formalism = formalism
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        self._solver = solver

    @property
    def settings(self) -> tuple[Any, ...]:
        """The formalism, the tolerance, the budget and the solver."""
        return (
            self._formalism,
            self._tolerance,
            self._max_iterations,
            id(self._solver),
        )

    def accepts(self, structure: FeasibleStructure) -> bool:
        """Two quadratic sets, the noise not a point."""
        return structure.form == "quadratic"

    def outer_ellipsoid(
        self, feasible: "FeasiblePropertySet", criterion: str, /
    ) -> EllipsoidResult:
        """The ellipsoid of the best weight, by the criterion.

        Raises:
            ValueError: if the set is empty, or the criterion is unknown.
        """
        _require_property_coordinates(
            feasible.structure.property_space, "the outer ellipsoid"
        )
        _check_criterion(criterion)
        structure = feasible.structure
        if feasible.is_empty():
            raise ValueError("The feasible set is empty; nothing contains it.")
        model_space, data_space, space = (
            structure.model_space,
            structure.data_space,
            structure.property_space,
        )
        A, T = structure.forward, structure.property_operator
        C, R = structure.prior_covariance(), structure.noise_covariance()
        a = structure.prior_center()
        shifted = structure.shifted_data(feasible.data)
        family = _quadratic_family(
            structure, formalism=self._formalism, solver=self._solver
        )
        rhs = family.right_hand_side(shifted)
        pulls = [T.adjoint(space.basis_vector(j)) for j in range(space.dim)]
        if self._formalism == "data_space":
            images = [C(c) for c in pulls]
            columns_rhs = [A(image) for image in images]
        else:
            columns_rhs = pulls
            P = structure.noise_precision()
            data_term = float(data_space.inner_product(P(shifted), shifted))

        # The natural scale of γ: where the two terms of the family balance.
        probe = (
            shifted
            if data_space.norm(shifted) > 0.0
            else data_space.random(rng=np.random.default_rng(0))
        )
        scale = float(
            data_space.inner_product(A(C(A.adjoint(probe))), probe)
            / data_space.inner_product(R(probe), probe)
        )
        start = [float(np.log(scale)) if scale > 0.0 else 0.0]

        warm: dict[str, Any] = {"at": None, "centre": None, "columns": None}
        memo: dict[float, tuple[float, float, Any, np.ndarray]] = {}
        evaluations = [0]

        def pieces(x: float) -> tuple[float, float, Any, np.ndarray]:
            """``(θ, r_θ, u_θ, T C_θ T* in components)`` at ``log γ == x``."""
            if x in memo:
                return memo[x]
            evaluations[0] += 1
            gamma = float(np.exp(x))
            theta = gamma / (1.0 + gamma)
            # Warm-start from the last weight within a few decades: further
            # away the scales differ too much to help.
            near = warm["at"] is not None and abs(warm["at"] - x) <= 6.0
            centre = family.solve(
                gamma, rhs, x0=warm["centre"] if near else None
            ).solution
            offset = family.model_from(centre)
            if self._formalism == "data_space":
                residual = theta * float(data_space.inner_product(centre, shifted))
            else:
                residual = (1.0 - theta) * (
                    data_term - float(model_space.inner_product(rhs, centre))
                )
            previous = warm["columns"] if near else [None] * space.dim
            solutions, columns = [], []
            for j in range(space.dim):
                solution = family.solve(gamma, columns_rhs[j], x0=previous[j]).solution
                solutions.append(solution)
                if self._formalism == "data_space":
                    column = model_space.subtract(images[j], C(A.adjoint(solution)))
                    column = T(model_space.scale((1.0 + gamma) / gamma, column))
                else:
                    column = T(model_space.scale(1.0 + gamma, solution))
                columns.append(space.to_components(column))
            shape = np.column_stack(columns)
            if space.is_orthonormal:
                shape = 0.5 * (shape + shape.T)
            warm.update(at=x, centre=centre, columns=solutions)
            memo[x] = (theta, residual, offset, shape)
            return memo[x]

        def objective(x: float) -> float:
            theta, residual, _, shape = pieces(x)
            budget = 1.0 - residual
            if budget <= 0.0:
                return np.inf
            if criterion == "trace":
                return budget * float(np.trace(shape))
            sign, log_det = np.linalg.slogdet(shape)
            if sign <= 0.0:
                return np.inf
            return space.dim * float(np.log(budget)) + float(log_det)

        x, _, converged = _minimize_unimodal(
            objective,
            start,
            tolerance=self._tolerance,
            max_iterations=self._max_iterations,
        )
        theta, residual, offset, shape = pieces(x)
        budget = 1.0 - residual
        if budget <= 0.0:
            raise ValueError(
                "No model satisfies both constraints: the weighted sum's "
                f"minimum is {residual:.4g} against 1."
            )
        traits = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        ellipsoid = Ellipsoid(
            space,
            LinearOperator.from_matrix(
                space,
                space,
                np.linalg.inv(shape) / budget,
                form="components",
                traits=traits,
            ),
            center=T(model_space.add(a, offset)),
            covariance=LinearOperator.from_matrix(
                space, space, budget * shape, form="components", traits=traits
            ),
        )
        return EllipsoidResult(
            ellipsoid=ellipsoid,
            criterion=criterion,
            converged=converged,
            algorithm=self.name,
            evaluations=evaluations[0],
            weight=theta,
            message="" if converged else "the weight search did not converge",
        )


_LIMIT = 40.0
"""How far the search in ``log γ`` walks before an end is taken as the minimizer."""


def _minimize_unimodal(
    f: Callable[[float], float],
    start: list[float],
    /,
    *,
    tolerance: float,
    max_iterations: int,
    limit: float = _LIMIT,
    lower: float | None = None,
) -> tuple[float, float, bool]:
    """Minimize a unimodal function of one variable from a start, by bracketing then Brent.

    *start* is a one-element list carrying the previous minimizer, updated
    on return so that the next direction starts there. The bracket is found
    by walking downhill with a growing step; a walk that reaches ``±limit``,
    or *lower* when one is given, reports that end as the minimizer.
    """
    from scipy.optimize import minimize_scalar

    floor = -limit if lower is None else max(lower, -limit)
    x = max(start[0], floor)
    step = 0.5
    calls = [0]

    def g(value: float) -> float:
        calls[0] += 1
        return f(value)

    fx = g(x)
    lo, hi = max(x - step, floor), x + step
    flo, fhi = g(lo), g(hi)
    while not (fx <= flo and fx <= fhi):
        if flo < fx:
            hi, fhi, x, fx = x, fx, lo, flo
            lo = max(x - step, floor)
            flo = g(lo)
        else:
            lo, flo, x, fx = x, fx, hi, fhi
            hi = x + step
            fhi = g(hi)
        step *= 1.6
        if x >= limit or x <= floor:
            # The function keeps decreasing to the end of the range: that
            # end is the minimizer, as far as this search can tell.
            start[0] = x
            return x, fx, True
        if calls[0] >= max_iterations:
            start[0] = x
            return x, fx, False
    if not (fx < flo and fx < fhi):
        # Flat to rounding on one side: nothing further to resolve.
        start[0] = x
        return x, fx, True
    remaining = max(max_iterations - calls[0], 5)
    result = minimize_scalar(
        g,
        bracket=(lo, x, hi),
        method="brent",
        options=dict(xtol=tolerance, maxiter=remaining),
    )
    start[0] = float(result.x)
    return float(result.x), float(result.fun), bool(result.success)


# --------------------------------------------------------------------- #
#                             The general tier                           #
# --------------------------------------------------------------------- #


def _smooth_level(subset: Subset) -> tuple[Functional, float] | None:
    """``(f, c)`` with the set ``{ f <= c }`` and ``f`` carrying a gradient, or None."""
    if not subset.has_smooth_level_function:
        return None
    return subset.level_function(), float(subset.level)


def _computes_support(subset: Any) -> bool:
    """Whether a support function with a maximizer can be had, cheaply or by conversion."""
    if subset is None:
        return True  # exact data: the point {0}
    if isinstance(subset, ConvexSet):
        return subset.computes_support_function and subset.computes_maximizer
    return _smooth_level(subset) is not None


def _support_of(subset: Any) -> SupportFunction:
    """A set's support function, converting from a smooth level function when needed."""
    if isinstance(subset, ConvexSet):
        return subset.support_function()
    smooth = _smooth_level(subset)
    if smooth is None:
        raise TypeError(
            f"{type(subset).__name__} has no support function to convert from."
        )
    functional, level = smooth
    return _LevelSupport(subset.domain, functional, level)


def _maximizer_of(subset: Any, direction: Any) -> Any:
    if isinstance(subset, ConvexSet):
        return subset.support_maximizer(direction)
    return _support_of(subset).subgradient(direction)


def _has_projection(subset: Subset | None) -> bool:
    return subset is None or subset.has_projection


class Dual(Algorithm):
    """The support by the dual: a nonsmooth minimization over the data space.

    ``h(q) == inf over λ of (λ, d) + h_prior(T*q − A*λ) + h_noise(−λ)``,
    using only the two sets' support functions and their maximizers (the
    maximizers are the subgradients), so any convex sets with those will
    do, including ones whose support function is converted from a smooth
    level function. Every ``λ`` certifies an outer bound by weak duality,
    and the minimizer is the best certificate; the prior's maximizer at the
    residual ``T*q − A*λ`` is the model attaining the bound, so the contact
    point comes with it, and it is reported as the inner value when it fits
    the data.

    The minimization runs in the noise set's own metric when that set is
    quadratic -- the data space reweighted by its precision -- since the
    bundle's proximal term is an inner product and the Euclidean one is
    the wrong geometry for a coloured noise set. Certificates warm-start
    across a sweep: the last direction's, or the negated certificate of the
    opposite direction when that was seen and the noise set is symmetric.
    An unbounded dual is an empty feasible set, raised rather than returned.
    """

    name = "dual"
    probes = frozenset({"support", "feasibility"})

    def __init__(
        self,
        /,
        *,
        method: Any = None,
        tolerance: float = 1e-10,
        max_iterations: int = 300,
        start: Any = None,
    ) -> None:
        """
        Args:
            method: the minimizer of the dual cost: any ``Optimizer`` that
                takes a subgradient. A proximal bundle method with the
                tolerance and budget below by default.
            tolerance: the default method's gap tolerance.
            max_iterations: its budget.
            start: a certificate to start every direction from, instead of
                the sweep's warm start; for cross-seeding from another
                algorithm's certificate.
        """
        self._method = method
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        self._start = start

    @property
    def settings(self) -> tuple[Any, ...]:
        """The method, the tolerance, the budget and the start."""
        return (
            id(self._method),
            self._tolerance,
            self._max_iterations,
            id(self._start),
        )

    def accepts(self, structure: FeasibleStructure) -> bool:
        """Support functions and maximizers on both sides, cheaply or by conversion."""
        return _computes_support(structure.prior) and _computes_support(structure.noise)

    def _minimizer(self, *, weight: float = 1.0) -> Any:
        if self._method is not None:
            return self._method
        from ..numerics.convex import ProximalBundleMethod

        return ProximalBundleMethod(
            weight=weight,
            tolerance=self._tolerance,
            max_iterations=self._max_iterations,
        )

    @staticmethod
    def _scaled_start(cost: Functional, start: Any) -> tuple[Any, float]:
        """A first step along the subgradient, and the proximal weight it implies.

        A bundle's proximal weight is in the units of the certificate, and
        the certificate's natural size is anything: a tight noise set and a
        wide prior put the optimum at ``1e-9`` in the data metric, where a
        weight of one steps a billion times too far and every step is a
        null step until the weight has doubled thirty times. One scalar
        minimization along the first subgradient finds the step that the
        model would take at the right weight; the bundle starts from there
        with that weight, and nothing else about it depends on the units.
        """
        space = cost.domain
        slope = cost.subgradient(start)
        length = space.norm(slope)
        if not np.isfinite(length) or length == 0.0:
            return start, 1.0

        def along(log_step: float) -> float:
            value = cost(space.axpy(-np.exp(log_step), slope, space.copy(start)))
            return value if np.isfinite(value) else np.inf

        log_step, value, _ = _minimize_unimodal(
            along, [0.0], tolerance=0.1, max_iterations=40
        )
        if log_step >= _LIMIT - 1.0 and value < cost(start):
            raise ValueError(
                "The dual cost falls without bound along its first subgradient: "
                "no model lies both inside the prior set and within the noise "
                "set of the data, so the feasible set is empty."
            )
        step = float(np.exp(log_step))
        if value >= cost(start):
            return start, 1.0 / step
        return space.axpy(-step, slope, space.copy(start)), 1.0 / step

    def _space(self, structure: FeasibleStructure) -> tuple[HilbertSpace, Any]:
        """The space the dual runs in, and the map from the data-space gradient to its own."""

        def build() -> tuple[HilbertSpace, Any]:
            from ..algebra.spaces import MassWeightedSpace

            data_space = structure.data_space
            if structure.noise is not None and _is_quadratic(structure.noise):
                precision = structure.noise_precision()
                covariance = structure.noise_covariance()
                # The metric's shape is the noise set's covariance, which is
                # the curvature of the noise term of the dual cost in the
                # certificate; its scale is normalized away, so that a tiny
                # or a huge noise set does not reach the bundle as a tiny or
                # a huge proximal weight.
                probe = data_space.random(rng=np.random.default_rng(0))
                scale = data_space.squared_norm(probe) / float(
                    data_space.inner_product(covariance(probe), probe)
                )
                weighted = MassWeightedSpace(data_space, covariance * scale)
                return weighted, precision * (1.0 / scale)
            return data_space, None

        return structure.shared(("dual", "space"), build)

    def cost(self, feasible: "FeasiblePropertySet", direction: Any, /) -> Functional:
        """The dual cost functional of a direction, on the space the dual runs in.

        Args:
            feasible: the set.
            direction: the direction in the property space.
        """
        structure = feasible.structure
        space, to_metric = self._space(structure)
        data_space, model_space = structure.data_space, structure.model_space
        forward, data = structure.forward, feasible.data
        pulled = structure.property_operator.adjoint(direction)
        prior_support = _support_of(structure.prior)
        noise = structure.noise
        noise_support = None if noise is None else _support_of(noise)
        memo = threading.local()

        def parts(certificate: Any) -> tuple[Any, Any]:
            # A bundle asks for the value and the subgradient at one point
            # in turn; the residual and the negation are shared between them.
            last = getattr(memo, "last", None)
            if last is None or last[0] is not certificate:
                last = (
                    certificate,
                    (
                        model_space.subtract(pulled, forward.adjoint(certificate)),
                        data_space.negative(certificate),
                    ),
                )
                memo.last = last
            return last[1]

        def value(certificate: Any) -> float:
            residual, negated = parts(certificate)
            total = data_space.inner_product(certificate, data) + float(
                prior_support(residual)
            )
            if noise_support is not None:
                total += float(noise_support(negated))
            return total

        def gradient(certificate: Any) -> Any:
            residual, negated = parts(certificate)
            slope = data_space.subtract(
                data, forward(prior_support.subgradient(residual))
            )
            if noise_support is not None:
                slope = data_space.subtract(slope, noise_support.subgradient(negated))
            return slope if to_metric is None else to_metric(slope)

        return Functional.from_callables(space, value, gradient=gradient)

    def _start_for(self, feasible: "FeasiblePropertySet", direction: Any) -> Any:
        if self._start is not None:
            return self._start
        structure = feasible.structure
        space = structure.data_space
        memo = feasible.cached(("dual", "warm", threading.get_ident()), dict)
        # The opposite direction's certificate, negated, when the noise set
        # is symmetric about its centre (every quadratic set is).
        key = feasible._key(structure.property_space.negative(direction))
        opposite = memo.get(key)
        if opposite is not None and (
            structure.noise is None or _is_quadratic(structure.noise)
        ):
            return space.negative(opposite)
        last = memo.get("last")
        return space.zero() if last is None else last

    def support(
        self, feasible: "FeasiblePropertySet", direction: Any, /
    ) -> SupportResult:
        """The support in a direction, by minimizing the dual cost.

        Raises:
            ValueError: if the dual is unbounded, which is an empty set.
        """
        structure = feasible.structure
        data_space, model_space = structure.data_space, structure.model_space
        cost = self.cost(feasible, direction)
        start, weight = self._scaled_start(cost, self._start_for(feasible, direction))
        result = self._minimizer(weight=weight).minimize(cost, start)
        if not result.converged and result.value < -_UNBOUNDED:
            raise ValueError(
                f"The dual fell to {result.value:.3g} without converging: no "
                "model lies both inside the prior set and within the noise "
                "set of the data, so the feasible set is empty."
            )
        certificate = result.minimizer
        memo = feasible.cached(("dual", "warm", threading.get_ident()), dict)
        memo["last"] = certificate
        memo[feasible._key(direction)] = certificate
        # The model attaining the bound: the prior's maximizer at the
        # residual, which fits the data at the exact optimum.
        pulled = structure.property_operator.adjoint(direction)
        residual = model_space.subtract(pulled, structure.forward.adjoint(certificate))
        extremal = _maximizer_of(structure.prior, residual)
        error = data_space.subtract(feasible.data, structure.forward(extremal))
        fits = (
            data_space.norm(error) <= 1e-8 * max(data_space.norm(feasible.data), 1.0)
            if structure.noise is None
            else structure.noise.contains(error, rtol=1e-6)
        )
        inner = (
            float(model_space.inner_product(pulled, extremal))
            if fits
            else float("-inf")
        )
        return SupportResult(
            value=float(result.value),
            inner=inner,
            converged=bool(result.converged),
            algorithm=self.name,
            evaluations=int(result.evaluations),
            certificate=certificate,
            extremal_model=extremal,
            message=result.message,
        )

    def feasibility(self, feasible: "FeasiblePropertySet", /) -> InclusionResult:
        """Emptiness by the dual's boundedness in one direction; no fitting model.

        A bounded dual means a non-empty set; the level reported is zero,
        since the prior's level at a fitting model is not what this
        algorithm computes, and the message says so.
        """
        direction = feasible.domain.basis_vector(0)
        try:
            result = feasible.support_result(direction, via=self)
        except ValueError:
            return InclusionResult(
                level=float("inf"),
                lower_bound=float("inf"),
                converged=True,
                algorithm=self.name,
                evaluations=0,
                message="the dual is unbounded: no model fits",
            )
        return InclusionResult(
            level=0.0,
            lower_bound=0.0,
            converged=bool(result.converged),
            algorithm=self.name,
            evaluations=result.evaluations,
            message="the dual is bounded: some model fits; its level is not computed",
        )


class Splitting(Algorithm):
    """The support by primal-dual splitting: Chambolle-Pock on the two sets' projections.

    ``maximize (T*q, m)`` subject to ``m`` in the prior, ``d − A m`` in the
    noise set, by the first-order primal-dual method, which needs nothing of
    the sets but their projections. The primal iterate is a model of the
    set to the feasibility tolerance and gives the inner value; its dual
    iterate is a certificate, and where the sets have support functions it
    gives the outer value through the dual cost, the better of the two
    signs since either certifies. Without support functions the outer
    value is infinite, honestly: a projection cannot bound a set from
    outside.

    The outer value is a bound whatever the iterate, by weak duality. The
    inner value is not quite attained: the primal iterate lies in the set
    only to the feasibility tolerance, so ``inner`` can exceed ``value`` by
    that order, and a bracket read off this algorithm is certified to the
    tolerance on its inner side. A loose tolerance is a loose inner value.
    """

    name = "splitting"
    probes = frozenset({"support"})

    def __init__(
        self,
        /,
        *,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        rng: Any = None,
    ) -> None:
        """
        Args:
            tolerance: on the feasibility residual.
            max_iterations: the cap.
            rng: for the power iteration that sets the step sizes.
        """
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        self._rng = rng

    @property
    def settings(self) -> tuple[Any, ...]:
        """The tolerance and the budget."""
        return (self._tolerance, self._max_iterations)

    def accepts(self, structure: FeasibleStructure) -> bool:
        """Projections on both sides."""
        return _has_projection(structure.prior) and _has_projection(structure.noise)

    def _solver(self, feasible: "FeasiblePropertySet") -> Any:
        from ..numerics.convex import ChambollePockSolver

        structure = feasible.structure
        noise = structure.noise
        if noise is None:
            noise = Ball(structure.data_space, radius=0.0)
        return feasible.cached(
            ("splitting", self.settings),
            lambda: ChambollePockSolver(
                structure.prior,
                noise,
                structure.forward,
                feasible.data,
                tolerance=self._tolerance,
                max_iterations=self._max_iterations,
                rng=self._rng,
            ),
        )

    def support(
        self, feasible: "FeasiblePropertySet", direction: Any, /
    ) -> SupportResult:
        """The support in a direction, by the splitting."""
        structure = feasible.structure
        data_space = structure.data_space
        pulled = structure.property_operator.adjoint(direction)
        memo = feasible.cached(("splitting", "warm", threading.get_ident()), dict)
        result = self._solver(feasible).solve(pulled, start=memo.get("last"))
        memo["last"] = result.minimizer
        inner = -float(result.value)
        outer = float("inf")
        if _computes_support(structure.prior) and _computes_support(structure.noise):
            outer = min(
                _dual_cost(structure, feasible.data, direction, result.certificate),
                _dual_cost(
                    structure,
                    feasible.data,
                    direction,
                    data_space.negative(result.certificate),
                ),
            )
        return SupportResult(
            value=outer,
            inner=inner,
            converged=bool(result.converged),
            algorithm=self.name,
            evaluations=int(result.evaluations),
            certificate=result.certificate,
            extremal_model=result.minimizer,
            message=result.message,
        )


class Distance(Algorithm):
    """Membership from the support side: the distance to the set by the projection dual.

    ``½ d(p)² == max over q of (q, p) − h(q) − ½ ||q||²`` on the property
    space, with ``h`` the set's own support function answered by whichever
    algorithm computes it and the contact points as subgradients; a
    strongly convex problem in ``dim P`` unknowns, each evaluation one
    support solve. A point is in the set when its distance is at most the
    tolerance, relative to its size; the result carries the distance, and
    a positive one comes with a separating direction, the certificate of
    exclusion. The level reported is the prior's when inside and infinite
    when outside: this algorithm has no fitting model to measure.
    """

    name = "distance"
    probes = frozenset({"inclusion"})

    def __init__(self, /, *, tolerance: float = 1e-8, method: Any = None) -> None:
        """
        Args:
            tolerance: the relative distance under which a point counts as
                inside.
            method: the minimizer of the projection dual; a proximal bundle
                method by default.
        """
        self._tolerance = tolerance
        self._method = method

    @property
    def settings(self) -> tuple[Any, ...]:
        """The tolerance and the method."""
        return (self._tolerance, id(self._method))

    def accepts(self, structure: FeasibleStructure) -> bool:
        """Whenever the support function is finite with a contact point: both sets have one, cheaply or by conversion."""
        return _computes_support(structure.prior) and _computes_support(structure.noise)

    def inclusion(
        self, feasible: "FeasiblePropertySet", value: Any, /
    ) -> InclusionResult:
        """Membership by the distance from the set's support function."""
        space = feasible.domain
        oracle = ConvexSet.from_support_function(
            space, feasible.support, maximizer=feasible.support_maximizer
        )
        _, distance, result = _project_by_support(oracle, value, method=self._method)
        inside = distance <= self._tolerance * max(space.norm(value), 1.0)
        level = feasible.level if inside else float("inf")
        return InclusionResult(
            level=level,
            lower_bound=level,
            converged=bool(result.converged),
            algorithm=self.name,
            evaluations=int(result.evaluations),
            message=(
                f"distance {distance:.3g}: inside"
                if inside
                else f"distance {distance:.3g}: outside"
            ),
        )


def linear_certificate_operator(
    forward: LinearOperator,
    property_operator: LinearOperator,
    /,
    *,
    covariance: LinearOperator,
    error_covariance: LinearOperator | None = None,
    alpha: float = 1.0,
    solver: LinearSolver | None = None,
) -> LinearOperator:
    """``X == T C A* (A C A* + α R)⁻¹``: the weighting of the data that resolves ``T`` against the noise.

    The minimizer of the quadratic surrogate ``‖(T − X A) C^½‖² + α ‖X R^½‖²``
    for the width of a linear certificate's set: ``C`` weights how much of
    the model the certificate has to resolve, ``R`` how much noise it lets
    through, and ``α`` trades the two. Without ``R`` the second term is
    dropped and ``X`` reproduces ``T`` wherever the data allow. Two balls
    about the origin, of radii ``M`` on the model and ``D`` on the data,
    give the classical ``T A* (A A* + (D/M)² I)⁻¹``. Nothing is assembled:
    each application of ``X`` is one solve on the data space through the
    solver.

    Args:
        forward: ``A``, from the model space to the data space.
        property_operator: ``T``, from the model space to the property
            space.
        covariance: ``C``, positive definite on the model space.
        error_covariance: ``R``, positive semidefinite on the data space;
            ``None`` for exact data.
        alpha: ``α``, positive. One minimizes the sum of the two terms.
        solver: how ``A C A* + α R`` is inverted. Conjugate gradients at
            ``rtol=1e-10`` by default.

    Raises:
        ValueError: if ``alpha`` is not positive, or an operator is on the
            wrong space.
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}.")
    if property_operator.domain != forward.domain:
        raise ValueError("The property operator must act on the model space.")
    if covariance.domain != forward.domain:
        raise ValueError("The covariance must act on the model space.")
    data_space = forward.codomain
    if error_covariance is not None and error_covariance.domain != data_space:
        raise ValueError("The error covariance must act on the data space.")
    normal = forward @ covariance @ forward.adjoint
    if error_covariance is not None:
        normal = normal + alpha * error_covariance
    if solver is None:
        solver = _default_solver(data_space)
    inverse = solver(normal.with_traits(Traits.POSITIVE_DEFINITE))
    return property_operator @ covariance @ forward.adjoint @ inverse


def _default_solver(space: HilbertSpace) -> LinearSolver:
    """Conjugate gradients at ``rtol=1e-10`` with a cap of at least 200 iterations."""
    return CGSolver(rtol=1e-10, max_iterations=max(2 * space.dim, 200))


def _has_support(subset: Any) -> bool:
    """Whether a set's support value can be had: cheaply, or by conversion from a smooth level function."""
    if subset is None:
        return True  # exact data: the point {0}
    if isinstance(subset, ConvexSet):
        return subset.computes_support_function
    return _smooth_level(subset) is not None


class LinearCertificate(Algorithm):
    """An outer bound from a fixed weighting of the data, on any sets; never chosen unasked.

    A linear certificate is an operator ``X`` from the data space to the
    property space. Whatever ``X`` is, a feasible model ``m`` with data
    error ``e`` has ``T m == X d + (T − X A) m − X e``, so its property lies
    in the Minkowski sum

    .. code-block:: text

        X d + (T − X A) S_prior + (−X) S_noise

    whose support in a direction ``q`` is ``(q, X d) + h_prior((T − X A)* q)
    + h_noise(−X* q)``: the dual cost at the certificate ``λ == X* q``, an
    outer bound by weak duality. It is sharp only where ``X* q`` happens to
    be the best certificate for ``q``, so this route reports no inner
    value, and the certified algorithms are what it is loose against. Its
    price is one application of ``X*`` and two support values per
    direction, nothing solved on the model space, which is what makes it
    worth asking for when every certified value is a search.

    ``X`` is the caller's ``operator=``, evaluated against any sets with
    support functions, or, on a ball or an ellipsoid on each side, the
    minimizer of the quadratic surrogate for the width of the sum,
    :func:`linear_certificate_operator` with the sets' covariances and
    ``alpha=`` as the trade-off. On quadratic sets the two parts of the sum
    are ellipsoids and :meth:`outer_ellipsoid` bounds their sum by one.
    :meth:`terms` gives the three parts of a support value separately,
    the estimate, the unresolved reach of the prior set and the noise's,
    which is the split that says whether coverage or data quality is the
    limit.
    """

    name = "linear_certificate"
    probes = frozenset({"support", "outer_ellipsoid"})
    automatic = False

    def __init__(
        self,
        /,
        *,
        operator: LinearOperator | None = None,
        alpha: float = 1.0,
        solver: LinearSolver | None = None,
    ) -> None:
        """
        Args:
            operator: ``X``, from the data space to the property space. Built
                from the sets' covariances if omitted, which needs a ball or
                an ellipsoid on each side.
            alpha: the trade-off of the built ``X`` between resolving ``T``
                and passing noise; ignored when ``operator`` is given.
            solver: how the built ``X`` inverts its data-space operator;
                the set's solver if omitted.

        Raises:
            ValueError: if ``alpha`` is not positive.
        """
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}.")
        self._operator = operator
        self._alpha = alpha
        self._solver = solver

    @property
    def settings(self) -> tuple[Any, ...]:
        """The operator's identity, the trade-off and the solver."""
        return (
            None if self._operator is None else id(self._operator),
            self._alpha,
            id(self._solver),
        )

    def accepts(self, structure: FeasibleStructure) -> bool:
        """Quadratic sets when ``X`` is to be built; any sets with support values for a given ``X``."""
        if self._operator is None:
            return structure.quadratic
        operator = self._operator
        return (
            operator.domain == structure.data_space
            and operator.codomain == structure.property_space
            and _has_support(structure.prior)
            and _has_support(structure.noise)
        )

    def operator(self, feasible: "FeasiblePropertySet", /) -> LinearOperator:
        """``X``: as given, or built once from the sets' covariances and shared across data."""
        if self._operator is not None:
            return self._operator
        structure = feasible.structure

        def build() -> LinearOperator:
            return linear_certificate_operator(
                structure.forward,
                structure.property_operator,
                covariance=structure.prior_covariance(),
                error_covariance=(
                    None if structure.exact else structure.noise_covariance()
                ),
                alpha=self._alpha,
                solver=(
                    structure.solver_for(structure.data_space)
                    if self._solver is None
                    else self._solver
                ),
            )

        return structure.shared(
            ("linear_certificate", self._alpha, id(self._solver)), build
        )

    def _parts(
        self, feasible: "FeasiblePropertySet", direction: Any
    ) -> tuple[float, float, float, Any]:
        """The estimate, the prior's and the noise's reach, and the certificate ``X* q``."""
        structure = feasible.structure
        model_space, data_space = structure.model_space, structure.data_space
        operator = self.operator(feasible)
        certificate = operator.adjoint(direction)
        residual = model_space.subtract(
            structure.property_operator.adjoint(direction),
            structure.forward.adjoint(certificate),
        )
        estimate = float(data_space.inner_product(certificate, feasible.data))
        if structure.quadratic:
            estimate += float(
                model_space.inner_product(residual, structure.prior_center())
            )
            covariance = structure.prior_covariance()
            resolution = float(
                np.sqrt(
                    max(model_space.inner_product(residual, covariance(residual)), 0.0)
                )
            )
            noise = 0.0
            if not structure.exact:
                estimate -= float(
                    data_space.inner_product(certificate, structure.noise_center())
                )
                noise = float(
                    np.sqrt(
                        max(
                            data_space.inner_product(
                                certificate, structure.noise_covariance()(certificate)
                            ),
                            0.0,
                        )
                    )
                )
            return estimate, resolution, noise, certificate
        resolution = float(_support_of(structure.prior)(residual))
        noise = (
            0.0
            if structure.exact
            else float(_support_of(structure.noise)(data_space.negative(certificate)))
        )
        return estimate, resolution, noise, certificate

    def terms(
        self, feasible: "FeasiblePropertySet", direction: Any, /
    ) -> tuple[float, float, float]:
        """The three parts of the support value in a direction.

        ``(estimate, resolution, noise)``: the pairing of the direction with
        ``X d`` moved by the sets' centres, the reach of ``(T − X A)`` over
        the prior set about its centre, and the reach of ``−X`` over the
        noise set about its centre. Their sum is the support value. More
        data narrow the second, better data the third, and a single
        half-width cannot say which is wanted. A set without a centre
        contributes its whole support value to its term.

        Args:
            feasible: the set.
            direction: the direction in the property space.
        """
        estimate, resolution, noise, _ = self._parts(feasible, direction)
        return estimate, resolution, noise

    def support(
        self, feasible: "FeasiblePropertySet", direction: Any, /
    ) -> SupportResult:
        """The outer value ``(q, X d) + h_prior((T − X A)* q) + h_noise(−X* q)``, with no inner value."""
        estimate, resolution, noise, certificate = self._parts(feasible, direction)
        return SupportResult(
            value=estimate + resolution + noise,
            inner=-np.inf,
            converged=True,
            algorithm=self.name,
            evaluations=1,
            certificate=certificate,
            message="linear certificate: an outer value with no inner value",
        )

    def outer_ellipsoid(
        self, feasible: "FeasiblePropertySet", criterion: str, /
    ) -> EllipsoidResult:
        """One ellipsoid containing the sum of the two parts, on quadratic sets.

        The prior's part is the ellipsoid of covariance ``(T − X A) C (T − X
        A)*`` and the noise's that of ``X R X*``, both about the estimate;
        their sum is bounded by :func:`~pygeoinf2.geometry.sum_of_ellipsoids`
        at the given criterion.

        Raises:
            NotImplementedError: unless the prior and the noise are each a
                ball or an ellipsoid.
            ValueError: if the criterion is unknown.
        """
        _check_criterion(criterion)
        structure = feasible.structure
        if not structure.quadratic:
            raise NotImplementedError(
                "The linear certificate's ellipsoid needs a ball or an ellipsoid "
                "on each side."
            )
        space = structure.property_space
        operator = self.operator(feasible)
        unresolved = structure.property_operator - operator @ structure.forward
        traits = Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE
        covariances = [
            (
                unresolved @ structure.prior_covariance() @ unresolved.adjoint
            ).with_traits(traits)
        ]
        center = space.add(
            operator(feasible.data), unresolved(structure.prior_center())
        )
        if not structure.exact:
            covariances.append(
                (
                    operator @ structure.noise_covariance() @ operator.adjoint
                ).with_traits(traits)
            )
            center = space.subtract(center, operator(structure.noise_center()))
        ellipsoid = sum_of_ellipsoids(
            space, covariances, center=center, criterion=criterion
        )
        return EllipsoidResult(
            ellipsoid=ellipsoid,
            criterion=criterion,
            converged=True,
            algorithm=self.name,
            evaluations=1,
            message="one ellipsoid about the estimate containing the Minkowski sum",
        )


_DEFAULTS: dict[str, tuple[Algorithm, ...]] = {
    "support": (ClosedForm(), Primal(), Dual(), Splitting(), LinearCertificate()),
    "inclusion": (ClosedForm(), Search(), Distance()),
    "feasibility": (ClosedForm(), Search(), Dual()),
    "outer_ellipsoid": (ClosedForm(), SProcedure(), LinearCertificate()),
}
"""The algorithms for each probe, in the order ``auto`` tries them; the
linear certificate answers by request only."""


# --------------------------------------------------------------------- #
#                                 The set                                #
# --------------------------------------------------------------------- #


class _FeasibleSupport(SupportFunction):
    def __init__(self, feasible: "FeasiblePropertySet") -> None:
        super().__init__(feasible.domain)
        self._feasible = feasible

    def _value(self, y: Any) -> float:
        return self._feasible.support(y)

    def _maximizer(self, y: Any) -> Any:
        return self._feasible.support_maximizer(y)


class FeasiblePropertySet(ConvexSet):
    """``T(prior ∩ A⁻¹(d − noise))``: the property values the data cannot rule out.

    The support side (support values and brackets, extremal models,
    polytopes, the linear certificate) is coordinate-free on every space.
    The membership side (``contains``, ``inclusion_level``, ``extent``,
    ``inner_hull``, the outer ellipsoids, and the closed form throughout)
    forms matrices of the property's dimension, and needs a coordinate map
    on the property space; without one it refuses by name.

    Built from the forward operator ``A``, the property operator ``T``, a
    convex constraint set on the model space, the data, and a convex noise
    set on the data space (``None`` for exact data). Nothing is computed at
    construction; every probe is answered when asked, by the first
    algorithm that accepts the sets' :attr:`form`, and remembered.

    Two characterizations are always available. The **support side** bounds
    the set from outside, direction by direction: :meth:`support`,
    :meth:`support_values`, :meth:`polytope`; each value comes with a
    certified bracket (:meth:`support_bracket`) and, where the algorithm
    exhibits one, the contact point and the model attaining it;
    :meth:`outer_ellipsoid` bounds it from outside as a whole. The
    **level side** decides points: :meth:`contains`, :meth:`inclusion_level`
    and :meth:`level_function`, with :meth:`extent` and :meth:`inner_hull`
    bounding the set from inside. Which algorithm answers each probe is
    :attr:`algorithms`; ``via=`` on a probe names one for a cross-check, or
    passes an instance carrying tuning. ``via="linear_certificate"`` asks
    for the cheap outer bound a fixed weighting of the data gives, which
    no probe chooses on its own.
    """

    def __init__(
        self,
        forward_operator: LinearOperator,
        property_operator: LinearOperator,
        prior: ConvexSet,
        data: Any,
        /,
        *,
        noise: ConvexSet | None = None,
        solver: LinearSolver | None = None,
    ) -> None:
        """
        Args:
            forward_operator: ``A``, from the model space to the data space.
            property_operator: ``T``, from the model space to the property
                space.
            prior: the constraint set on the model space.
            data: the observations.
            noise: the set the data error lies in, on the data space;
                ``None`` for exact data.
            solver: how the algorithms invert the operators they build on
                the data space. Conjugate gradients at ``rtol=1e-12`` by
                default; a preconditioner rides on it.

        Raises:
            ValueError: if an operator or a set is on the wrong space.
            TypeError: if the prior or the noise is not a convex set.
        """
        structure = FeasibleStructure(
            forward_operator, property_operator, prior, noise=noise, solver=solver
        )
        self._attach(structure, data)

    def _attach(self, structure: FeasibleStructure, data: Any) -> None:
        super().__init__(structure.property_space)
        self._structure = structure
        self._data = data
        self._cache: dict[Any, Any] = {}
        self._lock = threading.RLock()

    @classmethod
    def from_structure(
        cls, structure: FeasibleStructure, data: Any, /
    ) -> "FeasiblePropertySet":
        """The set for these data on a structure already built, sharing its work.

        What :meth:`with_data` and an estimator use: every set built on one
        structure shares the data-independent pieces its algorithms keep
        there.

        Args:
            structure: the operators, sets and solver, with their caches.
            data: the observations.
        """
        feasible = cls.__new__(cls)
        feasible._attach(structure, data)
        return feasible

    @property
    def structure(self) -> FeasibleStructure:
        """The data-independent part: operators, sets, solver, and the algorithms' shared work."""
        return self._structure

    # -- what it holds -----------------------------------------------------

    @property
    def data(self) -> Any:
        """The observations."""
        return self._data

    @property
    def forward_operator(self) -> LinearOperator:
        """``A``."""
        return self._structure.forward

    @property
    def property_operator(self) -> LinearOperator:
        """``T``."""
        return self._structure.property_operator

    @property
    def prior(self) -> ConvexSet:
        """The constraint set on the model space."""
        return self._structure.prior

    @property
    def noise(self) -> ConvexSet | None:
        """The noise set on the data space, or ``None`` for exact data."""
        return self._structure.noise

    @property
    def form(self) -> Form:
        """``"closed_form"``, ``"quadratic"`` or ``"general"``: what the sets admit."""
        return self._structure.form

    @property
    def algorithms(self) -> dict[str, tuple[str, ...]]:
        """The algorithms that accept these sets, per probe, in the order ``auto`` tries them.

        An algorithm that answers only on request, such as the linear
        certificate, is left out; it is still reachable through ``via=``.
        """
        return {
            probe: tuple(
                a.name for a in entries if a.automatic and a.accepts(self._structure)
            )
            for probe, entries in _DEFAULTS.items()
        }

    def with_data(self, data: Any, /) -> "FeasiblePropertySet":
        """The set for other data, sharing everything that does not depend on them."""
        return FeasiblePropertySet.from_structure(self._structure, data)

    # -- dispatch ----------------------------------------------------------

    def _resolve(self, probe: str, via: Any) -> Algorithm:
        entries = _DEFAULTS[probe]
        applicable = [a for a in entries if a.accepts(self._structure)]
        if via is None:
            automatic = [a for a in applicable if a.automatic]
            if not automatic:
                raise NotImplementedError(
                    f"No algorithm answers {probe!r} for sets of form "
                    f"{self.form!r}: the prior is a {type(self.prior).__name__} "
                    f"and the noise a {type(self.noise).__name__}."
                )
            return automatic[0]
        names = [a.name for a in applicable]
        if isinstance(via, str):
            for algorithm in entries:
                if algorithm.name == via:
                    if not algorithm.accepts(self._structure):
                        raise ValueError(
                            f"{via!r} does not apply to sets of form {self.form!r}; "
                            f"the algorithms that answer {probe!r} here are {names}."
                        )
                    return algorithm
            raise ValueError(
                f"Unknown algorithm {via!r} for {probe!r}; the algorithms are "
                f"{[a.name for a in entries]}."
            )
        if isinstance(via, Algorithm):
            if probe not in via.probes:
                raise ValueError(f"{via!r} does not answer {probe!r}.")
            if not via.accepts(self._structure):
                raise ValueError(
                    f"{via!r} does not apply to sets of form {self.form!r}; the "
                    f"algorithms that answer {probe!r} here are {names}."
                )
            return via
        raise TypeError("via= takes an algorithm's name or an instance.")

    def cached(self, key: Any, build: Callable[[], Any], /) -> Any:
        """A data-dependent piece of an algorithm's work, built once for this set.

        The counterpart of :meth:`FeasibleStructure.shared` for what depends
        on the data: an algorithm keeps its fit, its solutions along a sweep
        and its results here, under a key of its own.

        Args:
            key: hashable; by convention the algorithm's name first.
            build: called once, under the set's lock, if the key is new.

        Returns:
            What was built, now or before.
        """
        with self._lock:
            if key not in self._cache:
                self._cache[key] = build()
            return self._cache[key]

    def _key(self, vector: Any) -> Any:
        """A key for a property vector: its value where the space has
        coordinates, its identity otherwise.

        The support side of the set is coordinate-free, so a property space
        without a coordinate map must not be asked for one here; the price
        is that two equal vectors held as different objects are two cache
        entries.
        """
        if isinstance(self.domain, CoordinateSpace):
            return np.ascontiguousarray(
                self.domain.to_components(vector), dtype=float
            ).tobytes()
        return _ByIdentity(vector)

    # -- the support side ---------------------------------------------------

    def support_result(self, direction: Any, /, *, via: Any = None) -> SupportResult:
        """The support in one direction, with its bracket and by-products.

        Args:
            direction: the direction in the property space.
            via: an algorithm's name or instance, to force one.

        Raises:
            ValueError: if the set is empty, or ``via`` does not apply.
            NotImplementedError: if no algorithm answers for these sets.
        """
        algorithm = self._resolve("support", via)
        return self.cached(
            ("support", algorithm.name, algorithm.settings, self._key(direction)),
            lambda: algorithm.support(self, direction),
        )

    def support(self, direction: Any, /, *, via: Any = None) -> float:
        """The support value in one direction: the outer bound, infinite where the set is unbounded.

        Args:
            direction: the direction in the property space.
            via: an algorithm's name or instance, to force one.
        """
        return self.support_result(direction, via=via).value

    def support_bracket(
        self, direction: Any, /, *, via: Any = None
    ) -> tuple[float, float]:
        """``(inner, outer)``: an attained value and the certified bound.

        Args:
            direction: the direction in the property space.
            via: an algorithm's name or instance, to force one.
        """
        result = self.support_result(direction, via=via)
        return result.inner, result.value

    def support_sweep(
        self,
        directions: Sequence[Any],
        /,
        *,
        via: Any = None,
        n_jobs: int | None = None,
    ) -> SweepResult:
        """The support in many directions, one result each.

        Args:
            directions: the directions to evaluate, in the order given;
                neighbouring directions warm-start each other where the
                algorithm can.
            via: an algorithm's name or instance, to force one.
            n_jobs: workers over the directions. Serial by default.
        """
        from ..parallel import parallel_map, resolve_jobs

        directions = tuple(directions)
        workers = max(1, min(resolve_jobs(n_jobs), len(directions)))
        if workers == 1:
            results = [self.support_result(q, via=via) for q in directions]
        else:
            # Contiguous chunks, one per worker, so that the warm start of
            # one direction serves its neighbour within a chunk.
            bounds = np.linspace(0, len(directions), workers + 1).astype(int)
            chunks = [directions[bounds[i] : bounds[i + 1]] for i in range(workers)]

            def sweep(chunk: tuple[Any, ...]) -> list[SupportResult]:
                return [self.support_result(q, via=via) for q in chunk]

            results = [
                result
                for part in parallel_map(sweep, chunks, n_jobs=workers, threads=True)
                for result in part
            ]
        return SweepResult(directions, tuple(results))

    def support_values(
        self,
        directions: Sequence[Any],
        /,
        *,
        via: Any = None,
        n_jobs: int | None = None,
    ) -> np.ndarray:
        """The support values in many directions.

        Args:
            directions: the directions to evaluate.
            via: an algorithm's name or instance, to force one.
            n_jobs: workers over the directions. Serial by default.
        """
        return self.support_sweep(directions, via=via, n_jobs=n_jobs).values

    def support_function(self) -> SupportFunction:
        """The support function, answered by :meth:`support`."""
        return _FeasibleSupport(self)

    def extremal_model(self, direction: Any, /, *, via: Any = None) -> Any:
        """A model of the feasible set attaining the inner value in a direction.

        Args:
            direction: the direction in the property space.
            via: an algorithm's name or instance, to force one.

        Raises:
            NotImplementedError: if the algorithm exhibits no model.
        """
        result = self.support_result(direction, via=via)
        if result.extremal_model is None:
            raise NotImplementedError(
                f"{result.algorithm!r} bounds the set without exhibiting a model."
            )
        return result.extremal_model

    def support_maximizer(self, direction: Any, /, *, via: Any = None) -> Any:
        """The contact point: the property of the extremal model.

        Args:
            direction: the direction in the property space.
            via: an algorithm's name or instance, to force one.

        Raises:
            NotImplementedError: if the algorithm exhibits no model.
        """
        return self.property_operator(self.extremal_model(direction, via=via))

    def certificate(self, direction: Any, /, *, via: Any = None) -> Any:
        """The dual certificate of the outer bound, where the algorithm produces one.

        Args:
            direction: the direction in the property space.
            via: an algorithm's name or instance, to force one.

        Raises:
            NotImplementedError: if it does not.
        """
        result = self.support_result(direction, via=via)
        if result.certificate is None:
            raise NotImplementedError(
                f"{result.algorithm!r} does not produce a dual certificate."
            )
        return result.certificate

    def polytope(
        self,
        directions: Sequence[Any],
        /,
        *,
        via: Any = None,
        n_jobs: int | None = None,
    ) -> Polytope:
        """A certified outer bound, one half-space per direction with a finite support.

        Args:
            directions: the directions; one in which the set is unbounded
                contributes no half-space.
            via: an algorithm's name or instance, to force one.
            n_jobs: workers over the directions.
        """
        sweep = self.support_sweep(directions, via=via, n_jobs=n_jobs)
        planes = [
            HalfSpace(self.domain, direction, offset=float(result.value))
            for direction, result in zip(sweep.directions, sweep.results)
            if np.isfinite(result.value)
        ]
        return Polytope(self.domain, planes, outer=True)

    def outside(
        self, x: Any, directions: Sequence[Any], /, *, rtol: float = 1e-9
    ) -> bool:
        """True when some direction's bound proves the point is not in the set.

        One-sided: a ``True`` is a proof, a ``False`` only the absence of
        one among the directions tried.

        Args:
            x: the point to test.
            directions: the directions to try.
            rtol: how far past a supporting hyperplane the point must lie.
        """
        for direction in directions:
            pairing = self.domain.inner_product(direction, x)
            if pairing > self.support(direction) * (1.0 + rtol) + rtol:
                return True
        return False

    def outer_ellipsoid_result(
        self,
        *,
        criterion: Literal["trace", "volume"] = "trace",
        via: Any = None,
    ) -> EllipsoidResult:
        """A certified ellipsoid containing the set, with how it was chosen.

        Args:
            criterion: what the ellipsoid is smallest in -- the sum of its
                squared semi-axes, or its volume.
            via: an algorithm's name or instance, to force one.
        """
        _check_criterion(criterion)
        algorithm = self._resolve("outer_ellipsoid", via)
        return self.cached(
            ("outer_ellipsoid", algorithm.name, algorithm.settings, criterion),
            lambda: algorithm.outer_ellipsoid(self, criterion),
        )

    def outer_ellipsoid(
        self,
        *,
        criterion: Literal["trace", "volume"] = "trace",
        via: Any = None,
    ) -> Ellipsoid:
        """A certified ellipsoid containing the set.

        On the closed form, the set itself; on two quadratic sets, the
        S-procedure ellipsoid of the best weight, which is tighter than a
        linear certificate's Minkowski sum because the data enter the
        constraint. Its support in any direction is at least the set's.

        Args:
            criterion: what the ellipsoid is smallest in -- the sum of its
                squared semi-axes, or its volume.
            via: an algorithm's name or instance, to force one.

        Raises:
            ValueError: if the set is empty.
            NotImplementedError: on the general form, where no algorithm
                builds one.
        """
        return self.outer_ellipsoid_result(criterion=criterion, via=via).ellipsoid

    # -- the level side ---------------------------------------------------

    def feasibility_result(self, *, via: Any = None) -> InclusionResult:
        """The prior's level function at the smallest model fitting the data alone.

        Args:
            via: an algorithm's name or instance, to force one.
        """
        algorithm = self._resolve("feasibility", via)
        return self.cached(
            ("feasibility", algorithm.name, algorithm.settings),
            lambda: algorithm.feasibility(self),
        )

    def is_empty(self, *, via: Any = None, rtol: float = 1e-8) -> bool:
        """Whether no model within the prior fits the data.

        Args:
            via: an algorithm's name or instance, to force one.
            rtol: how far above the prior's level still counts as fitting.
        """
        result = self.feasibility_result(via=via)
        return not result.level <= self.level * (1.0 + rtol)

    def fitting_model(self, *, via: Any = None) -> Any:
        """The smallest model, in the prior's level function, that fits the data.

        Args:
            via: an algorithm's name or instance, to force one.

        Raises:
            ValueError: if the set is empty.
        """
        if self.is_empty(via=via):
            raise ValueError(
                "No model within the prior fits the data; the set is empty."
            )
        return self.feasibility_result(via=via).fitting_model

    def inclusion_result(self, value: Any, /, *, via: Any = None) -> InclusionResult:
        """The prior's level function at the model fitting a value, with its certificate.

        Args:
            value: the property value.
            via: an algorithm's name or instance, to force one.
        """
        algorithm = self._resolve("inclusion", via)
        return self.cached(
            ("inclusion", algorithm.name, algorithm.settings, self._key(value)),
            lambda: algorithm.inclusion(self, value),
        )

    def inclusion_level(self, value: Any, /, *, via: Any = None) -> float:
        """The prior's level function at the smallest model with this value fitting the data.

        Admissible exactly when at most :attr:`level`; infinite when no model
        has the value and fits the data, which is a proof of exclusion.

        Args:
            value: the property value.
            via: an algorithm's name or instance, to force one.
        """
        return self.inclusion_result(value, via=via).level

    @property
    def level(self) -> float:
        """The prior's level, which :meth:`inclusion_level` is compared with.

        One, nominally, for a prior without a level function: there the
        inclusion level is one inside the set and infinite outside, decided
        from the support side.
        """
        try:
            return float(self.prior.level)
        except (NotImplementedError, AttributeError):
            return 1.0

    def admits(self, value: Any, /, *, rtol: float = 1e-8, via: Any = None) -> bool:
        """Whether a property value is consistent with the data and the prior.

        Args:
            value: the property value.
            rtol: how far above the level still counts.
            via: an algorithm's name or instance, to force one.
        """
        return self.inclusion_level(value, via=via) <= self.level * (1.0 + rtol)

    def contains(self, x: Any, /, *, rtol: float = 1e-8) -> bool:
        """Membership, by the inclusion level.

        Args:
            x: a property value.
            rtol: how far above the prior's level still counts.
        """
        return self.admits(x, rtol=rtol)

    def level_function(self) -> Functional:
        """``p -> inclusion_level(p)``, whose sublevel set at :attr:`level` is this set.

        Convex on the property space; carries a gradient where the
        algorithm supplies one.
        """
        algorithm = self._resolve("inclusion", None)
        gradient = None
        if algorithm.level_gradient(self, self.domain.zero()) is not None:

            def gradient(value: Any) -> Any:
                return algorithm.level_gradient(self, value)

        return Functional.from_callables(
            self.domain, self.inclusion_level, gradient=gradient
        )

    def extent(
        self, direction: Any, /, *, max_iterations: int = 40, via: Any = None
    ) -> tuple[float, float]:
        """How far the set reaches along a line through the fitting model's property.

        The values are admissible on an interval, since the set is convex,
        and its two ends are found by bracketing and bisecting :meth:`admits`.
        Returned as the pairing ``(direction, p)`` at the two ends, which
        compare directly with ``-support(-direction)`` and
        ``support(direction)``: these are inner bounds, points of the
        boundary.

        Args:
            direction: the direction of the line.
            max_iterations: bisection steps for each end.
            via: an algorithm's name or instance, to force one.

        Returns:
            ``(lower, upper)``.

        Raises:
            ValueError: if the set is empty, or appears unbounded along the line.
        """
        space = self.domain
        base = self.property_operator(self.fitting_model(via=via))
        length = space.squared_norm(direction)
        if length == 0.0:
            value = space.inner_product(direction, base)
            return value, value

        def admitted(step: float) -> bool:
            return self.admits(space.axpy(step, direction, space.copy(base)), via=via)

        def crossing(sign: float) -> float:
            step = max(1.0, abs(space.inner_product(direction, base))) / length
            inside, outside = 0.0, sign * step
            for _ in range(60):
                if not admitted(outside):
                    break
                inside, outside = outside, 2.0 * outside
            else:
                raise ValueError("The set appears unbounded along this direction.")
            for _ in range(max_iterations):
                middle = 0.5 * (inside + outside)
                if admitted(middle):
                    inside = middle
                else:
                    outside = middle
            return inside

        center = space.inner_product(direction, base)
        return center + crossing(-1.0) * length, center + crossing(1.0) * length

    def inner_hull(
        self,
        directions_or_values: Sequence[Any],
        /,
        *,
        source: Literal["contact", "admitted"] = "contact",
        via: Any = None,
    ) -> Polytope:
        """The convex hull of points of the set: a certified inner bound.

        Args:
            directions_or_values: with ``source="contact"``, directions whose
                contact points (the properties of the extremal models) are
                the hull's vertices; with ``source="admitted"``, candidate
                values of which the admissible ones are kept.
            source: which of the two.
            via: an algorithm's name or instance, to force one.

        Raises:
            ValueError: if fewer points are found than the space has
                dimensions plus one, there being no hull to take.
        """
        from scipy.spatial import ConvexHull

        space = self.domain
        if source == "contact":
            points = [
                space.to_components(self.support_maximizer(q, via=via))
                for q in directions_or_values
            ]
        elif source == "admitted":
            points = [
                space.to_components(v)
                for v in directions_or_values
                if self.admits(v, via=via)
            ]
        else:
            raise ValueError("source is 'contact' or 'admitted'.")
        if len(points) <= space.dim:
            raise ValueError(
                f"Only {len(points)} points, which is not enough to bound a hull "
                f"in {space.dim} dimensions."
            )
        hull = ConvexHull(np.stack(points))
        planes = [
            HalfSpace(
                space, space.representer(equation[:-1]), offset=-float(equation[-1])
            )
            for equation in hull.equations
        ]
        return Polytope(space, planes, outer=False)

    # -- as a convex set ---------------------------------------------------

    @property
    def has_membership(self) -> bool:
        """Whether some algorithm decides membership for these sets."""
        return bool(self.algorithms["inclusion"])

    @property
    def has_projection(self) -> bool:
        """Whether :meth:`project` is available: on the closed form."""
        return self.form == "closed_form"

    @property
    def has_support_function(self) -> bool:
        """Whether some algorithm computes support values for these sets."""
        return bool(self.algorithms["support"])

    @property
    def has_maximizer(self) -> bool:
        """Whether the support side exhibits contact points."""
        return self.has_support_function

    @property
    def has_level_function(self) -> bool:
        """Whether the level side describes the set."""
        return self.has_membership

    @cached_property
    def ellipsoid(self) -> Ellipsoid:
        """The set as an ellipsoid, on the closed form.

        Raises:
            NotImplementedError: on other forms.
            ValueError: if the set is empty.
        """
        if self.form != "closed_form":
            raise NotImplementedError(
                f"On the {self.form!r} form the set is not an ellipsoid."
            )
        return ClosedForm().ellipsoid(self)

    def project(self, x: Any, /) -> Any:
        """The nearest point, on the closed form; refused elsewhere.

        Raises:
            NotImplementedError: unless the set is an ellipsoid.
        """
        if self.form != "closed_form":
            raise NotImplementedError(
                "A feasible property set known through its probes has no "
                "projection in closed form; bound it with polytope() and "
                "project onto that."
            )
        return self.ellipsoid.project(x)

    def push_forward(self, operator: LinearOperator, /) -> "FeasiblePropertySet":
        """The same set for a further property ``operator @ T``."""
        return FeasiblePropertySet(
            self._structure.forward,
            operator @ self._structure.property_operator,
            self._structure.prior,
            self._data,
            noise=self._structure.noise,
            solver=self._structure.solver,
        )

    def __repr__(self) -> str:
        return f"FeasiblePropertySet(form={self.form!r}, dim={self.domain.dim})"
