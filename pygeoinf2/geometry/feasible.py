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
from ..algebra.spaces import HilbertSpace
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
)
from .sets import SublevelSet

__all__ = [
    "Algorithm",
    "ClosedForm",
    "Distance",
    "Dual",
    "FeasiblePropertySet",
    "FeasibleStructure",
    "InclusionResult",
    "Primal",
    "Search",
    "Splitting",
    "SupportResult",
    "SweepResult",
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
class SweepResult:
    """The support in many directions, one result each.

    What :meth:`FeasiblePropertySet.polytope` and
    :meth:`FeasiblePropertySet.outer_ellipsoid` consume: they need to know
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
            ("default_solver", space.dim),
            lambda: CGSolver(rtol=1e-10, max_iterations=max(2 * space.dim, 200)),
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
        prior = self.prior
        space = self.model_space
        if isinstance(prior, Ball):
            return (LinearOperator.identity(space) * prior.radius**2).with_traits(
                Traits.POSITIVE_DEFINITE
            )
        if isinstance(prior, Ellipsoid):
            if prior.covariance is not None:
                return prior.covariance
            return self.solver_for(space)(
                prior.precision.with_traits(Traits.POSITIVE_DEFINITE)
            )
        raise TypeError("Only a ball or an ellipsoid has a covariance to read.")

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
        noise = self.noise
        space = self.data_space
        if isinstance(noise, Ball):
            return (LinearOperator.identity(space) * noise.radius**2).with_traits(
                Traits.POSITIVE_DEFINITE
            )
        if isinstance(noise, Ellipsoid):
            if noise.covariance is not None:
                return noise.covariance
            return self.solver_for(space)(
                noise.precision.with_traits(Traits.POSITIVE_DEFINITE)
            )
        raise TypeError("Only a ball or an ellipsoid has a covariance to read.")

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


def _precision_of(subset: Any, space: HilbertSpace, name: str) -> LinearOperator:
    if isinstance(subset, Ball):
        if subset.radius <= 0.0:
            raise ValueError(f"The {name} ball needs a positive radius.")
        return (LinearOperator.identity(space) * (1.0 / subset.radius**2)).with_traits(
            Traits.POSITIVE_DEFINITE
        )
    if isinstance(subset, Ellipsoid):
        return subset.precision
    raise TypeError(f"The {name} must be a ball or an ellipsoid to have a precision.")


# --------------------------------------------------------------------- #
#                               Algorithms                               #
# --------------------------------------------------------------------- #


class Algorithm:
    """A way of probing a feasible property set, holding tuning only.

    An algorithm declares the forms it accepts and answers the probes it
    can: ``support`` in a direction, ``inclusion`` of a value, and
    ``feasibility`` of the whole set. Instances carry tolerances and
    budgets and nothing else, so two instances with the same settings are
    interchangeable and the set may share work between them. A probe takes
    one by name (its ``name``) or as an instance through ``via=``.
    """

    name: str = ""
    probes: frozenset[str] = frozenset()

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
    probes = frozenset({"support", "inclusion", "feasibility"})

    def accepts(self, structure: FeasibleStructure) -> bool:
        """Exact data and a quadratic prior."""
        return structure.form == "closed_form"

    # -- the data-independent part ---------------------------------------

    def _pieces(self, structure: FeasibleStructure) -> dict[str, Any]:
        return structure.shared(("closed_form",), lambda: self._build(structure))

    def _build(self, structure: FeasibleStructure) -> dict[str, Any]:
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

    def prior_only(self, structure: FeasibleStructure) -> Ellipsoid:
        """What the prior alone says: ``{ p : ((T C T*)⁻¹ (p − T a), p − T a) <= 1 }``."""
        T = structure.property_operator
        space = structure.property_space
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
    feasibility search.
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
    ) -> None:
        """
        Args:
            formalism: which space the feasibility search solves in.
            max_iterations: the root find's budget.
            rtol: the root find's bracket tolerance.
            solver: how the family is inverted; the set's solver if omitted.
        """
        if formalism not in ("model_space", "data_space"):
            raise ValueError(
                f"The formalism is 'model_space' or 'data_space', got {formalism!r}."
            )
        self._formalism = formalism
        self._max_iterations = max_iterations
        self._rtol = rtol
        self._solver = solver

    @property
    def settings(self) -> tuple[Any, ...]:
        """The formalism, the budget, the tolerance and the solver."""
        return (self._formalism, self._max_iterations, self._rtol, id(self._solver))

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
        self, structure: FeasibleStructure, family: _QuadraticFamily, shifted: Any
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

        try:
            root = monotone_root(
                evaluate,
                1.0,
                decreasing=False,
                max_iterations=self._max_iterations,
                rtol=self._rtol,
                atol=0.0,
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

    def _singular_by_dimension(
        self, structure: FeasibleStructure, *, restricted: bool
    ) -> bool:
        """Whether the data-space family has no full rank at ``γ == 0``.

        ``A C A*`` on the data space has rank at most the model's dimension,
        and restricted to the kernel of ``T`` at most that less the
        property's; data beyond that make the family singular as the
        multiplier vanishes, and the search runs the model-space fit
        instead.
        """
        if self._formalism != "data_space":
            return False
        rank = structure.model_space.dim
        if restricted:
            rank -= structure.property_space.dim
        return structure.data_space.dim > rank

    def feasibility(self, feasible: "FeasiblePropertySet", /) -> InclusionResult:
        """The smallest prior level over the models fitting the data."""
        structure = feasible.structure
        if not structure.quadratic or self._singular_by_dimension(
            structure, restricted=False
        ):
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
        if not structure.quadratic or self._singular_by_dimension(
            structure, restricted=True
        ):
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
        root, kernel_model, misfit = self._search(structure, family, shifted)
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
    once per direction, so that nothing cancels.

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
    ) -> None:
        """
        Args:
            formalism: which space the family is solved in.
            tolerance: on ``log γ`` at the minimum. The outer value is
                stationary there, so its error is second order in this; the
                inner value's is first order, which is the bracket's width.
            max_iterations: the minimization's budget of evaluations.
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

    def _family(self, structure: FeasibleStructure) -> _QuadraticFamily:
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

    def support(
        self, feasible: "FeasiblePropertySet", direction: Any, /
    ) -> SupportResult:
        """The support in a direction, by minimizing ``φ(γ)``.

        Raises:
            ValueError: if the set is empty, or the data-space family is
                singular and the model-space formalism is needed.
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

        if self._formalism == "data_space" and data_space.dim > model_space.dim:
            raise ValueError(
                "The data outnumber the model, so the data-space family is "
                "singular at γ = 0 and the data-only bound is out of reach; "
                "use Primal(formalism='model_space')."
            )
        family = self._family(structure)
        rhs_data = family.right_hand_side(shifted)
        data_term = float(data_space.inner_product(P(shifted), shifted))
        cache = feasible.cached(("primal", "u2", self._formalism), dict)

        if self._formalism == "data_space":
            v = A(image)  # A C c
            try:
                g_inv_v = family.solve(0.0, v).solution
            except ConvergenceError as error:
                raise ValueError(
                    "The data-space family is singular at γ = 0: the forward "
                    "operator has no full row rank; use "
                    "Primal(formalism='model_space')."
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
                a_coef = kappa / gamma + float(
                    data_space.inner_product(g_inv_v, R(k_v))
                )
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
        log_gamma, value, converged = _minimize_unimodal(
            phi,
            feasible.cached(("primal", "start", threading.get_ident()), lambda: [0.0]),
            tolerance=self._tolerance,
            max_iterations=self._max_iterations,
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
) -> tuple[float, float, bool]:
    """Minimize a unimodal function of one variable from a start, by bracketing then Brent.

    *start* is a one-element list carrying the previous minimizer, updated
    on return so that the next direction starts there. The bracket is found
    by walking downhill with a growing step; a walk that reaches ``±limit``
    reports that end as the minimizer.
    """
    from scipy.optimize import minimize_scalar

    x = start[0]
    step = 0.5
    calls = [0]

    def g(value: float) -> float:
        calls[0] += 1
        return f(value)

    fx = g(x)
    lo, hi = x - step, x + step
    flo, fhi = g(lo), g(hi)
    while not (fx <= flo and fx <= fhi):
        if flo < fx:
            hi, fhi, x, fx = x, fx, lo, flo
            lo = x - step
            flo = g(lo)
        else:
            lo, flo, x, fx = x, fx, hi, fhi
            hi = x + step
            fhi = g(hi)
        step *= 1.6
        if abs(x) >= limit:
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


def _smooth_level(subset: Any) -> tuple[Functional, float] | None:
    """``(f, c)`` with the set ``{ f <= c }`` and ``f`` carrying a gradient, or None."""
    if isinstance(subset, SublevelSet):
        functional, level = subset.functional, float(subset.level)
    elif getattr(subset, "has_level_function", False):
        functional, level = subset.level_function(), float(subset.level)
    else:
        return None
    if not getattr(functional, "has_derivative", False):
        return None
    return functional, level


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


def _has_projection(subset: Any) -> bool:
    return subset is None or bool(getattr(subset, "has_projection", False))


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

    def _minimizer(self) -> Any:
        if self._method is not None:
            return self._method
        from ..numerics.convex import ProximalBundleMethod

        return ProximalBundleMethod(
            tolerance=self._tolerance, max_iterations=self._max_iterations
        )

    def _space(self, structure: FeasibleStructure) -> tuple[HilbertSpace, Any]:
        """The space the dual runs in, and the map from the data-space gradient to its own."""

        def build() -> tuple[HilbertSpace, Any]:
            from ..algebra.spaces import MassWeightedSpace

            data_space = structure.data_space
            if structure.noise is not None and _is_quadratic(structure.noise):
                precision = structure.noise_precision()
                covariance = structure.noise_covariance()
                # The metric's shape is the noise set's; its scale is
                # normalized away, so that a tiny or a huge noise set does
                # not reach the bundle as a tiny or a huge proximal weight.
                probe = data_space.from_components(np.ones(data_space.dim))
                scale = data_space.squared_norm(probe) / float(
                    data_space.inner_product(precision(probe), probe)
                )
                weighted = MassWeightedSpace(data_space, precision * scale)
                return weighted, covariance * (1.0 / scale)
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
        result = self._minimizer().minimize(cost, self._start_for(feasible, direction))
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


_DEFAULTS: dict[str, tuple[Algorithm, ...]] = {
    "support": (ClosedForm(), Primal(), Dual(), Splitting()),
    "inclusion": (ClosedForm(), Search(), Distance()),
    "feasibility": (ClosedForm(), Search(), Dual()),
}
"""The algorithms for each probe, in the order ``auto`` tries them."""


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

    Built from the forward operator ``A``, the property operator ``T``, a
    convex constraint set on the model space, the data, and a convex noise
    set on the data space (``None`` for exact data). Nothing is computed at
    construction; every probe is answered when asked, by the first
    algorithm that accepts the sets' :attr:`form`, and remembered.

    Two characterizations are always available. The **support side** bounds
    the set from outside, direction by direction: :meth:`support`,
    :meth:`support_values`, :meth:`polytope`; each value comes with a
    certified bracket (:meth:`support_bracket`) and, where the algorithm
    exhibits one, the contact point and the model attaining it. The
    **level side** decides points: :meth:`contains`, :meth:`inclusion_level`
    and :meth:`level_function`, with :meth:`extent` and :meth:`inner_hull`
    bounding the set from inside. Which algorithm answers each probe is
    :attr:`algorithms`; ``via=`` on a probe names one for a cross-check, or
    passes an instance carrying tuning.
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
        """The algorithms that accept these sets, per probe, in the order ``auto`` tries them."""
        return {
            probe: tuple(a.name for a in entries if a.accepts(self._structure))
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
            if not applicable:
                raise NotImplementedError(
                    f"No algorithm answers {probe!r} for sets of form "
                    f"{self.form!r}: the prior is a {type(self.prior).__name__} "
                    f"and the noise a {type(self.noise).__name__}."
                )
            return applicable[0]
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

    def _key(self, vector: Any) -> bytes:
        """A value-based key for a property vector."""
        return np.ascontiguousarray(
            self.domain.to_components(vector), dtype=float
        ).tobytes()

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

    def outer_ellipsoid(self) -> Ellipsoid:
        """A certified ellipsoid containing the set.

        On the closed form, the set itself.

        Raises:
            NotImplementedError: on other forms, until the S-procedure
                ellipsoid is built.
        """
        if self.form == "closed_form":
            return self.ellipsoid
        raise NotImplementedError(
            "An outer ellipsoid is available on the closed form only, for now."
        )

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
