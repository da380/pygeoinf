"""
Set-valued inference: a constraint set in, a convex set out.

Given a convex constraint set ``S_M^0`` on the model space and a convex
confidence set ``S_eta`` on the data space, the feasible model set is
``S_M == S_M^0 ∩ A^-1(d - S_eta)`` -- the models the constraint allows that
fit the data to within the confidence set -- and the answer is its image
``T(S_M)`` under the property operator ``T``, a convex subset of the property
space. :class:`BackusGilbertParker` returns it as a
:class:`~pygeoinf2.geometry.feasible.FeasiblePropertySet`, which chooses
from the two sets how to answer each question put to it.

:class:`BackusGilbert` is the linear certificate: a fixed operator ``X``
applied to the data, with the set of property values the bound allows as a
Minkowski sum of the resolution and noise parts. Any ``X`` gives a valid set
by weak duality, so it is never wrong, only loose; the feasible set is what
it is loose against.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ..algebra.operators import LinearOperator
from ..algebra.spaces import HilbertSpace
from ..geometry.convex import Ball, ConvexSet, Ellipsoid, sum_of_ellipsoids
from ..geometry.feasible import FeasiblePropertySet, FeasibleStructure
from ..geometry.sets import SublevelSet
from ..numerics.convex import SupportFunction
from ..numerics.solvers import LinearSolver
from ..probability.base import ProbabilityMeasure
from ..traits import Traits
from .estimators import LinearPointEstimator, SetEstimator
from .problem import LinearForwardProblem

__all__ = [
    "BackusGilbert",
    "BackusGilbertParker",
    "BackusGilbertSet",
    "FeasiblePropertySet",
    "harden_error",
]


def harden_error(problem: LinearForwardProblem, /, *, level: float) -> Ball:
    """The problem's error as a ball, hardening a measure if need be.

    The bridge from a probabilistic error to a set-theoretic one: the
    smallest ball about the mean carrying probability *level*, which is
    :meth:`~pygeoinf2.probability.GaussianMeasure.ambient_ball`.

    The radius is a quantile of ``sum_i lambda_i Z_i^2``, with ``lambda_i``
    the covariance's eigenvalues, in the *space's* norm.

    Args:
        problem: the forward problem, whose error may be a measure or a set.
        level: the probability the ball is to carry.

    Returns:
        A ball in the data space. Radius zero when the problem has no error,
        which is the error-free case saying "exactly this".

    Raises:
        TypeError: if the error is a convex set that is not a ball.
    """
    if not problem.has_error:
        return Ball(problem.data_space, radius=0.0)
    if isinstance(problem.error, Ball):
        return problem.error
    if isinstance(problem.error, ConvexSet):
        raise TypeError(
            "The problem's error is a convex set that is not a ball, so it "
            "cannot be hardened to one; BackusGilbertParker takes any convex "
            "set as its confidence set."
        )
    return problem.error_measure.ambient_ball(level=level)


def _confidence_set(
    problem: LinearForwardProblem, noise: Any, level: float
) -> ConvexSet | SublevelSet | None:
    """The set the data error lies in: as given, from the problem, or None for exact data."""
    if noise is not None:
        if not isinstance(noise, (ConvexSet, SublevelSet)):
            raise TypeError(
                "The confidence set must be a ConvexSet, or a SublevelSet of a "
                "convex functional."
            )
        return noise
    if not problem.has_error:
        return None
    error = problem.error
    if isinstance(error, (ConvexSet, SublevelSet)):
        return error
    if isinstance(error, ProbabilityMeasure):
        return problem.error_measure.ambient_ball(level=level)
    raise TypeError(
        "The problem's error is neither a measure nor a convex set; pass "
        "noise= to say what confidence set to use."
    )


def _covariance_if_quadratic(subset: ConvexSet | None) -> LinearOperator | None:
    """The covariance of a ball or an ellipsoid carrying one; None for any other set."""
    if isinstance(subset, (Ball, Ellipsoid)):
        return subset.covariance
    return None


def _center_of(subset: Any, space: HilbertSpace) -> Any:
    """A ball's or ellipsoid's centre, or the origin for any other set."""
    if isinstance(subset, (Ball, Ellipsoid)):
        return subset.center
    return space.zero()


# --------------------------------------------------------------------- #
#                         The feasible property set                     #
# --------------------------------------------------------------------- #


class BackusGilbertParker(SetEstimator):
    """The feasible property set: what the data and a constraint let a property be.

    In go the forward problem, a convex **constraint set** on the model,
    optionally a convex **confidence set** on the data, and the property
    operator ``T``; out comes the convex set of property values of every
    model that lies in the constraint set and fits the data to within the
    confidence set. The confidence set is taken from the problem when not
    given: its own set if it has one, the credible ball at ``level`` if its
    error is a measure, and exact data if it has no error at all.

    Nothing is computed here. :meth:`__call__` returns a
    :class:`~pygeoinf2.geometry.feasible.FeasiblePropertySet`, which
    answers each question when asked -- a support value with its bracket,
    membership of a value, emptiness, an extent, an inner hull -- by the
    algorithm the two sets' form admits, and ``via=`` on any probe names
    another for a cross-check or passes an instance with its own tuning.
    Sets for different data built by one estimator share the work that does
    not depend on the data.

    .. code-block:: python

        feasible = BackusGilbertParker(problem, property_operator, prior)(data)
        feasible.is_empty()
        feasible.support(direction)          # the support-function side
        feasible.support_bracket(direction)  # inner and outer values
        feasible.contains(value)             # the level-function side
        feasible.polytope(directions)        # an outer bound
        feasible.inner_hull(directions)      # an inner bound
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        property_operator: LinearOperator,
        prior: ConvexSet | SublevelSet,
        /,
        *,
        noise: ConvexSet | SublevelSet | None = None,
        level: float = 0.95,
        solver: LinearSolver | None = None,
    ) -> None:
        """
        Args:
            problem: the forward problem.
            property_operator: ``T``, the property operator, acting on the
                model space.
            prior: the constraint set on the model space: a convex set, or a
                ``SublevelSet`` of a convex functional, taken as convex on
                the caller's word.
            noise: the confidence set on the data space, likewise. Taken
                from the problem when omitted, as described above.
            level: the probability the confidence set carries when it is
                hardened from a measure.
            solver: how the algorithms invert the operators they build.
                Conjugate gradients by default; a preconditioner rides on
                it.

        Raises:
            ValueError: if an operator or set is on the wrong space.
            TypeError: if the prior or the confidence set is neither a
                ``ConvexSet`` nor a ``SublevelSet``, or the problem's error
                is neither a measure nor a convex set.
        """
        self._problem = problem
        self._level = level
        self._structure = FeasibleStructure(
            problem.forward_operator,
            property_operator,
            prior,
            noise=_confidence_set(problem, noise, level),
            solver=solver,
        )

    @property
    def forward_problem(self) -> LinearForwardProblem:
        """The forward problem."""
        return self._problem

    @property
    def property_operator(self) -> LinearOperator:
        """The property operator ``T``."""
        return self._structure.property_operator

    @property
    def prior(self) -> ConvexSet | SublevelSet:
        """The constraint set on the model space."""
        return self._structure.prior

    @property
    def noise(self) -> ConvexSet | SublevelSet | None:
        """The confidence set on the data space as resolved, or None for exact data."""
        return self._structure.noise

    @property
    def structure(self) -> FeasibleStructure:
        """The data-independent part every set from this estimator shares."""
        return self._structure

    @property
    def form(self) -> str:
        """The form of the pair of sets, which decides the algorithms."""
        return self._structure.form

    @property
    def data_space(self) -> HilbertSpace:
        """The problem's data space."""
        return self._structure.data_space

    @property
    def property_space(self) -> HilbertSpace:
        """The property space."""
        return self._structure.property_space

    def __call__(self, data: Any) -> FeasiblePropertySet:
        """The feasible property set for these data, as an object to probe."""
        return FeasiblePropertySet.from_structure(self._structure, data)

    def push_forward(self, operator: LinearOperator, /) -> "BackusGilbertParker":
        """The same inference about a further property of the model."""
        return BackusGilbertParker(
            self._problem,
            operator @ self._structure.property_operator,
            self._structure.prior,
            noise=self._structure.noise,
            level=self._level,
            solver=self._structure.solver,
        )

    def __repr__(self) -> str:
        return (
            f"BackusGilbertParker(form={self.form!r}, prior={self.prior!r}, "
            f"noise={self.noise!r})"
        )


# --------------------------------------------------------------------- #
#                          The linear certificate                       #
# --------------------------------------------------------------------- #


class BackusGilbertSet(ConvexSet):
    """The property values a linear certificate allows: a Minkowski sum.

    ``X d + (T − X A) S_prior + (−X) S_noise``: the estimate, plus what the
    estimate cannot see of the prior set, plus the noise set as the
    certificate weights it. Its support function is the sum of the parts'
    and is exact for any convex sets; membership and the projection are
    computed from it. :attr:`estimate` is the point the sum is built about
    and :meth:`error_bars` splits the half-widths into the two causes.
    """

    def __init__(
        self,
        estimate: Any,
        resolution: SupportFunction,
        noise: SupportFunction | None,
        /,
        *,
        resolution_covariance: LinearOperator | None = None,
        noise_covariance: LinearOperator | None = None,
    ) -> None:
        """
        Args:
            estimate: ``X d`` moved by the sets' centres, in the property
                space.
            resolution: the support function of ``(T − X A)(S_prior − a)``,
                on the property space.
            noise: that of ``(−X)(S_noise − b)``, or None for exact data.
            resolution_covariance: ``(T − X A) C (T − X A)*`` when the prior
                is a ball or an ellipsoid of covariance ``C``, which makes
                :meth:`outer_ellipsoid` available.
            noise_covariance: ``X R X*`` likewise, for a quadratic noise set.
        """
        space = resolution.domain
        super().__init__(space)
        self._estimate = estimate
        self._resolution = resolution
        self._noise = noise
        self._resolution_covariance = resolution_covariance
        self._noise_covariance = noise_covariance
        total = SupportFunction.of_point(space, estimate) + resolution
        if noise is not None:
            total = total + noise
        self._support = total
        self._oracle = ConvexSet.from_support_function(
            space, total, maximizer=total.subgradient
        )

    @property
    def estimate(self) -> Any:
        """``X d``, with the sets' centres carried through: the point the set is built about."""
        return self._estimate

    @property
    def parts(self) -> tuple[SupportFunction, SupportFunction | None]:
        """The support functions of the resolution part and the noise part, about the origin."""
        return self._resolution, self._noise

    def error_bars(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Componentwise half-widths, and the two contributions separately.

        Returns ``(estimate, resolution_half_width, noise_half_width)`` as
        component arrays. The split is the useful part: more data narrow
        the second and better coverage narrows the first, and a single
        number cannot say which is needed. Each half-width is the larger
        of the part's reach on the two sides of the estimate, which is
        the reach itself for a set symmetric about its centre, as every
        ball and ellipsoid is.
        """
        space = self.domain
        resolution, noise = [], []
        for index in range(space.dim):
            direction = space.basis_vector(index)
            opposite = space.negative(direction)
            resolution.append(
                max(self._resolution(direction), self._resolution(opposite))
            )
            noise.append(
                0.0
                if self._noise is None
                else max(self._noise(direction), self._noise(opposite))
            )
        return (
            np.asarray(space.to_components(self._estimate), dtype=float),
            np.array(resolution),
            np.array(noise),
        )

    @property
    def has_membership(self) -> bool:
        """False: membership is computed from the support function."""
        return False

    @property
    def has_projection(self) -> bool:
        """False: the projection is computed from the support function."""
        return False

    @property
    def has_support_function(self) -> bool:
        """True."""
        return True

    @property
    def has_maximizer(self) -> bool:
        """True: the sum of the parts' maximizers."""
        return True

    def support_function(self) -> SupportFunction:
        """``(q, X d) + h_prior((T − X A)* q) + h_noise(−X* q)``."""
        return self._support

    def support(self, direction: Any, /) -> float:
        """The support value in a direction: how far the set reaches along it.

        Args:
            direction: a vector of the property space.
        """
        return float(self._support(direction))

    def support_maximizer(self, direction: Any, /) -> Any:
        """The point attaining the support: the sum of the parts' maximizers."""
        return self._support.subgradient(direction)

    def outer_ellipsoid(
        self, *, criterion: Literal["trace", "volume"] = "trace"
    ) -> Ellipsoid:
        """One ellipsoid about the estimate containing the sum, on quadratic sets.

        The resolution part is the ellipsoid of covariance
        ``(T − X A) C (T − X A)*`` and the noise part that of ``X R X*``;
        their sum is not an ellipsoid, and this is the smallest of the
        family ``C₁/α + C₂/(1 − α)`` that contains it
        (:func:`~pygeoinf2.geometry.sum_of_ellipsoids`).

        Args:
            criterion: what the ellipsoid is smallest in -- the sum of its
                squared semi-axes, or its volume.

        Raises:
            TypeError: if either set was not a ball or an ellipsoid.
        """
        if self._resolution_covariance is None or (
            self._noise is not None and self._noise_covariance is None
        ):
            raise TypeError(
                "An outer ellipsoid needs a ball or an ellipsoid on each side."
            )
        covariances = [self._resolution_covariance]
        if self._noise_covariance is not None:
            covariances.append(self._noise_covariance)
        return sum_of_ellipsoids(
            self.domain, covariances, center=self._estimate, criterion=criterion
        )

    def project(self, x: Any, /) -> Any:
        """The nearest point, by the dual of the nearest-point problem."""
        return self._oracle.project(x)

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """Membership, by the distance from the projection.

        Args:
            x: a vector of the property space.
            rtol: how far outside still counts as inside, relative to the
                point's size.
        """
        return self._oracle.contains(x, rtol=rtol)

    def __repr__(self) -> str:
        return f"BackusGilbertSet(dim={self.domain.dim})"


class BackusGilbert(SetEstimator):
    """The optimally-averaged estimate of a property, with the set it certifies.

    A *linear certificate*: a fixed operator ``X`` applied to the data,
    whose error has two parts. ``X A`` is not ``T``, so the estimate sees
    the model through the **resolution** ``X A`` rather than the property
    asked for, and the difference ``(T − X A) m`` is bounded by the prior
    set; and ``X`` applied to the noise is bounded by the noise set. The
    set of property values consistent with both is the Minkowski sum

    .. code-block:: text

        X d + (T − X A) S_prior + (−X) S_noise

    whose support function is the sum of the parts'. **Any** ``X`` gives a
    valid set -- that is weak duality -- so this estimator can never be
    wrong, only loose, and the feasible set of :class:`BackusGilbertParker`
    is what it is loose against.

    The automatic choice, for a ball or an ellipsoid on each side with
    covariances ``C`` and ``R`` in the sets' normalized units, minimizes
    the quadratic surrogate ``||(T − X A) C^½||² + α ||X R^½||²`` for the
    width of the sum:

    .. code-block:: text

        X = T C A* (A C A* + α R)⁻¹

    one data-space solve per property through the caller's solver. Two
    origin-centred balls of radii ``M`` and ``D`` give the classical
    ``T A* (A A* + (D/M)² I)⁻¹``. SOLA is this estimator with ``α`` chosen
    by hand and no prior set named, which is why it reports a resolution
    kernel but cannot certify a resolution error; :meth:`from_operator`
    evaluates such a kernel, or any other certificate, against named sets.
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        property_operator: LinearOperator,
        prior: ConvexSet,
        /,
        *,
        noise: ConvexSet | None = None,
        level: float = 0.95,
        solver: LinearSolver | None = None,
        alpha: float = 1.0,
    ) -> None:
        """
        Args:
            problem: the forward problem.
            property_operator: ``T``, the property operator.
            prior: a ball or an ellipsoid on the model space, centred
                anywhere.
            noise: a ball or an ellipsoid on the data space. Taken from the
                problem's error if omitted, hardening a Gaussian one at
                ``level``; exact data when the problem has no error.
            level: the confidence level used if a Gaussian error is hardened.
            solver: how to invert the data-space operator ``A C A* + α R``.
                Conjugate gradients by default.
            alpha: the trade-off between the two parts of the width. One
                minimizes their sum of squares.

        Raises:
            ValueError: if an operator or a set is on the wrong space, or
                ``alpha`` is not positive.
            TypeError: if the prior or the noise is neither a ball nor an
                ellipsoid, or the problem's error is neither a measure nor
                a convex set.
        """
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}.")
        noise = _confidence_set(problem, noise, level)
        structure = FeasibleStructure(
            problem.forward_operator,
            property_operator,
            prior,
            noise=noise,
            solver=solver,
        )
        if not structure.quadratic:
            raise TypeError(
                "The automatic certificate needs a ball or an ellipsoid on each "
                "side; evaluate a certificate of your own against general sets "
                "with from_operator()."
            )
        forward, data_space = structure.forward, structure.data_space
        covariance = structure.prior_covariance()
        normal = forward @ covariance @ forward.adjoint
        if noise is not None:
            normal = normal + alpha * structure.noise_covariance()
        inverse = structure.solver_for(data_space)(
            normal.with_traits(Traits.POSITIVE_DEFINITE)
        )
        operator = property_operator @ covariance @ forward.adjoint @ inverse
        self._attach(
            problem,
            property_operator,
            operator,
            prior,
            noise,
            level,
            prior_covariance=covariance,
            noise_covariance=(None if noise is None else structure.noise_covariance()),
        )

    def _attach(
        self,
        problem: LinearForwardProblem,
        property_operator: LinearOperator,
        operator: LinearOperator,
        prior: ConvexSet,
        noise: ConvexSet | None,
        level: float,
        *,
        prior_covariance: LinearOperator | None,
        noise_covariance: LinearOperator | None,
    ) -> None:
        self._problem = problem
        self._property_operator = property_operator
        self._prior = prior
        self._noise = noise
        self._level = level
        model_space, data_space = problem.model_space, problem.data_space
        space = property_operator.codomain
        prior_center = _center_of(prior, model_space)
        noise_center = (
            data_space.zero() if noise is None else _center_of(noise, data_space)
        )
        unresolved = property_operator - operator @ problem.forward_operator
        # The estimate carries the centres: X d + (T − X A) a − X b.
        translation = space.subtract(unresolved(prior_center), operator(noise_center))
        self._point = LinearPointEstimator(
            operator,
            translation=translation,
            forward_operator=problem.forward_operator,
            error=(
                problem.error
                if problem.has_error and isinstance(problem.error, ProbabilityMeasure)
                else None
            ),
        )
        self._resolution_support = (
            prior.support_function()
            + SupportFunction.of_point(model_space, model_space.negative(prior_center))
        ).compose_with(unresolved)
        self._noise_support = (
            None
            if noise is None
            else (
                noise.support_function()
                + SupportFunction.of_point(
                    data_space, data_space.negative(noise_center)
                )
            ).compose_with(-operator)
        )
        traits = Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE
        self._resolution_covariance = (
            None
            if prior_covariance is None
            else (unresolved @ prior_covariance @ unresolved.adjoint).with_traits(
                traits
            )
        )
        self._noise_covariance = (
            None
            if noise_covariance is None
            else (operator @ noise_covariance @ operator.adjoint).with_traits(traits)
        )

    @classmethod
    def from_operator(
        cls,
        problem: LinearForwardProblem,
        property_operator: LinearOperator,
        operator: LinearOperator,
        prior: ConvexSet,
        /,
        *,
        noise: ConvexSet | None = None,
        level: float = 0.95,
    ) -> "BackusGilbert":
        """A certificate chosen by the caller, evaluated against any sets with support functions.

        Where a SOLA kernel, a Zaroli kernel or any other fixed weighting of
        the data is assessed against a named prior: the set it certifies is
        valid whatever ``X`` is, and this says how wide.

        Args:
            problem: the forward problem.
            property_operator: ``T``.
            operator: ``X``, from the data space to the property space.
            prior: the constraint set on the model space, any convex set
                with a support function.
            noise: the confidence set on the data space, likewise. Taken
                from the problem when omitted.
            level: the confidence level used if a Gaussian error is hardened.

        Raises:
            ValueError: if an operator or a set is on the wrong space.
            TypeError: if a set is not a ``ConvexSet``.
        """
        if property_operator.domain != problem.model_space:
            raise ValueError("The property operator must act on the model space.")
        if operator.domain != problem.data_space:
            raise ValueError("The certificate must act on the data space.")
        if operator.codomain != property_operator.codomain:
            raise ValueError("The certificate must map into the property space.")
        noise = _confidence_set(problem, noise, level)
        for subset, name, space in (
            (prior, "prior", problem.model_space),
            (noise, "noise", problem.data_space),
        ):
            if subset is None:
                continue
            if not isinstance(subset, ConvexSet):
                raise TypeError(
                    f"The {name} must be a ConvexSet with a support function."
                )
            if subset.domain != space:
                raise ValueError(f"The {name} set lies on the wrong space.")
        estimator = cls.__new__(cls)
        estimator._attach(
            problem,
            property_operator,
            operator,
            prior,
            noise,
            level,
            prior_covariance=_covariance_if_quadratic(prior),
            noise_covariance=_covariance_if_quadratic(noise),
        )
        return estimator

    # -- what it holds -----------------------------------------------------

    @property
    def forward_problem(self) -> LinearForwardProblem:
        """The forward problem."""
        return self._problem

    @property
    def property_operator(self) -> LinearOperator:
        """The property operator ``T``."""
        return self._property_operator

    @property
    def prior(self) -> ConvexSet:
        """The constraint set on the model space."""
        return self._prior

    @property
    def noise(self) -> ConvexSet | None:
        """The confidence set on the data space as resolved, or None for exact data."""
        return self._noise

    @property
    def operator(self) -> LinearOperator:
        """``X``, the certificate: the weighting of the data."""
        return self._point.operator

    @property
    def point_estimator(self) -> LinearPointEstimator:
        """The estimate as an affine operator on the data, with the propagated covariance."""
        return self._point

    @property
    def resolution(self) -> LinearOperator:
        """``X A``: what the estimate reports when the truth is ``m``."""
        return self._point.resolution

    @property
    def unresolved(self) -> LinearOperator:
        """``T − X A``: what the estimate cannot see.

        The complement of the resolution, and the operator whose reach over
        the prior set is the first part of the width. An estimate is only
        as good as this is small.
        """
        return self._property_operator - self.resolution

    @property
    def data_space(self) -> HilbertSpace:
        """The problem's data space."""
        return self._problem.data_space

    @property
    def property_space(self) -> HilbertSpace:
        """The property space."""
        return self._property_operator.codomain

    # ----------------------------------------------------------------- #

    def estimate(self, data: Any, /) -> Any:
        """``X d``, with the sets' centres carried through."""
        return self._point(data)

    def __call__(self, data: Any) -> BackusGilbertSet:
        """The set of property values the certificate allows for these data."""
        return BackusGilbertSet(
            self.estimate(data),
            self._resolution_support,
            self._noise_support,
            resolution_covariance=self._resolution_covariance,
            noise_covariance=self._noise_covariance,
        )

    def push_forward(self, operator: LinearOperator, /) -> "BackusGilbert":
        """The same certificate for a further property ``operator @ T``: ``operator @ X``."""
        return BackusGilbert.from_operator(
            self._problem,
            operator @ self._property_operator,
            operator @ self.operator,
            self._prior,
            noise=self._noise,
            level=self._level,
        )

    def __repr__(self) -> str:
        return f"BackusGilbert(prior={self._prior!r}, noise={self._noise!r})"
