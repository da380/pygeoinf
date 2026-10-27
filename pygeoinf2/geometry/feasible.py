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
from ..numerics.solvers import CGSolver, LinearSolver
from ..traits import Traits
from .convex import Ball, ConvexSet, Ellipsoid, HalfSpace, Polytope

__all__ = [
    "Algorithm",
    "ClosedForm",
    "FeasiblePropertySet",
    "InclusionResult",
    "SupportResult",
    "SweepResult",
]

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


class _Structure:
    """What every set from one estimator shares: operators, sets, solver, caches.

    The expensive data-independent work of each algorithm -- a kernel
    projector's pieces, a closed form's shape -- is kept here, keyed by the
    algorithm's settings, so that sets for different data share it.
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
        if prior.domain != forward.domain:
            raise ValueError("The constraint set must lie in the model space.")
        if not isinstance(prior, ConvexSet):
            raise TypeError("The constraint set must be a ConvexSet.")
        if noise is not None:
            if not isinstance(noise, ConvexSet):
                raise TypeError(
                    "The noise set must be a ConvexSet, or None for exact data."
                )
            if noise.domain != forward.codomain:
                raise ValueError("The noise set must lie in the data space.")
        self.forward = forward
        self.property_operator = property_operator
        self.prior = prior
        self.noise = None if _is_point(noise) else noise
        self.solver = solver if solver is not None else CGSolver(rtol=1e-12)
        self.cache: dict[Any, Any] = {}
        self.lock = threading.RLock()

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
            return self.solver(prior.precision.with_traits(Traits.POSITIVE_DEFINITE))
        raise TypeError("Only a ball or an ellipsoid has a covariance to read.")

    def prior_center(self) -> Any:
        """The prior's centre, or the origin for a set without one."""
        prior = self.prior
        if isinstance(prior, (Ball, Ellipsoid)):
            return prior.center
        return self.model_space.zero()


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

    def accepts(self, structure: _Structure) -> bool:
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

    def accepts(self, structure: _Structure) -> bool:
        """Exact data and a quadratic prior."""
        return structure.form == "closed_form"

    # -- the data-independent part ---------------------------------------

    def _pieces(self, structure: _Structure) -> dict[str, Any]:
        return structure.shared(("closed_form",), lambda: self._build(structure))

    def _build(self, structure: _Structure) -> dict[str, Any]:
        forward, T = structure.forward, structure.property_operator
        covariance = structure.prior_covariance()
        normal = (forward @ covariance @ forward.adjoint).with_traits(
            Traits.POSITIVE_DEFINITE
        )
        normal_inverse = structure.solver(normal)
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
        return feasible._cached(
            ("closed_form", "fit"), lambda: self._compute_fit(feasible)
        )

    def _compute_fit(self, feasible: "FeasiblePropertySet") -> dict[str, Any]:
        structure = feasible._structure
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
        structure = feasible._structure
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

    def prior_only(self, structure: _Structure) -> Ellipsoid:
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
        level = feasible._structure.prior.level
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
        structure = feasible._structure
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
        structure = feasible._structure
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
        structure = feasible._structure
        space = structure.property_space
        shape = self._pieces(structure)["shape"]
        offset = space.to_components(space.subtract(value, fit["center"]))
        try:
            weights = np.linalg.solve(shape, offset)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(shape, offset, rcond=None)[0]
            if not np.allclose(shape @ weights, offset, rtol=1e-10, atol=1e-14):
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
        structure = feasible._structure
        space = structure.property_space
        shape = self._pieces(structure)["shape"]
        offset = space.to_components(space.subtract(value, fit["center"]))
        return space.from_components(
            2.0 * structure.prior.level * np.linalg.solve(shape, offset)
        )


_DEFAULTS: dict[str, tuple[Algorithm, ...]] = {
    "support": (ClosedForm(),),
    "inclusion": (ClosedForm(),),
    "feasibility": (ClosedForm(),),
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
        _structure: _Structure | None = None,
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
        if _structure is None:
            _structure = _Structure(
                forward_operator, property_operator, prior, noise=noise, solver=solver
            )
        super().__init__(_structure.property_space)
        self._structure = _structure
        self._data = data
        self._cache: dict[Any, Any] = {}
        self._lock = threading.RLock()

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
        return FeasiblePropertySet(
            self._structure.forward,
            self._structure.property_operator,
            self._structure.prior,
            data,
            noise=self._structure.noise,
            _structure=self._structure,
        )

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

    def _cached(self, key: Any, build: Callable[[], Any]) -> Any:
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
        return self._cached(
            ("support", algorithm.name, self._key(direction)),
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
        from ..parallel import parallel_map

        directions = tuple(directions)
        results = parallel_map(
            lambda q: self.support_result(q, via=via), directions, n_jobs=n_jobs
        )
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
        return self._cached(
            ("feasibility", algorithm.name), lambda: algorithm.feasibility(self)
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
        return self._cached(
            ("inclusion", algorithm.name, self._key(value)),
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
        """The prior's level, which :meth:`inclusion_level` is compared with."""
        return float(self.prior.level)

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
        if hasattr(algorithm, "level_gradient"):

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
