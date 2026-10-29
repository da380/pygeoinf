"""
Set-valued inference, and the averaging-kernel estimate it completes.

Given a convex constraint set ``S_M^0`` on the model space and a convex
confidence set ``S_eta`` on the data space, the feasible model set is
``S_M == S_M^0 ∩ A^-1(d - S_eta)`` -- the models the constraint allows that
fit the data to within the confidence set -- and the answer is its image
``T(S_M)`` under the property operator ``T``, a convex subset of the property
space. :class:`BackusGilbertParker` returns it as a
:class:`~pygeoinf2.geometry.feasible.FeasiblePropertySet`, which chooses
from the two sets how to answer each question put to it. A nonlinear
forward problem is linearized about a model of the caller's choosing, and
the set that comes back is exact for the linearized problem and a local
statement about the real one.

:class:`BackusGilbert` is the point estimator of the literature: a fixed
operator ``X`` applied to the data, chosen so that the resolution ``X A``
comes close to ``T`` against a penalty on the noise it passes. It carries
its resolution and its propagated error and states no bound; the feasible
set under a prior set is where a bound comes from, and the same ``X`` with
prior and noise sets gives a certified set through the feasible set's
``linear_certificate`` route.
"""

from __future__ import annotations

from typing import Any

from ..algebra.linearization import Linearization
from ..algebra.operators import LinearOperator
from ..algebra.spaces import HilbertSpace
from ..geometry.convex import Ball, ConvexSet, Ellipsoid
from ..geometry.feasible import (
    FeasiblePropertySet,
    FeasibleStructure,
    linear_certificate_operator,
)
from ..geometry.sets import SublevelSet
from ..numerics.solvers import LinearSolver
from ..probability.base import ProbabilityMeasure
from ..probability.gaussian import GaussianMeasure
from .estimators import LinearPointEstimator, SetEstimator
from .problem import ForwardProblem, LinearForwardProblem

__all__ = [
    "BackusGilbert",
    "BackusGilbertParker",
    "FeasiblePropertySet",
    "LinearizedFeasiblePropertySet",
]


def _confidence_set(
    problem: ForwardProblem, noise: Any, level: float
) -> ConvexSet | SublevelSet | None:
    """The set the data error lies in: as given, from the problem, or None for exact data.

    A measure is hardened to its ambient ball at *level*: the smallest ball
    about its expectation, in the data space's norm, carrying that
    probability.
    """
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


# --------------------------------------------------------------------- #
#                         The feasible property set                     #
# --------------------------------------------------------------------- #


class LinearizedFeasiblePropertySet(FeasiblePropertySet):
    """The feasible property set of a forward problem linearized about a model.

    With ``F`` the forward operator and ``m*`` the linearization point, the
    linearized problem predicts ``F(m*) + F'(m*)(m − m*)``, an affine map of
    the model itself. Its feasible property set is therefore built on the
    derivative ``A == F'(m*)`` with the data shifted to ``d − F(m*) + A m*``,
    and nothing else moves: the constraint set is the one given on the
    model, and the values are the property's own, so every probe reads as
    it does on a linear problem. The set is exact for the linearized problem
    and a local statement about the real one, good to the extent that the
    linearization holds across the constraint set. The linearization point
    need not lie in the constraint set; if it does not, the forward operator
    is linearized outside the region the set makes a claim about, and the
    local statement is weaker.

    :attr:`linearization_point` is ``m*`` and :attr:`linearization_property`
    its property ``T m*``; :attr:`observed_data` the data as observed, and
    :attr:`data` the shifted data the set is built on.
    """

    def _attach_linearization(
        self, linearization: Linearization, observed_data: Any
    ) -> None:
        self._linearization = linearization
        self._observed_data = observed_data

    @classmethod
    def from_linearization(
        cls,
        structure: FeasibleStructure,
        linearization: Linearization,
        data: Any,
        /,
    ) -> "LinearizedFeasiblePropertySet":
        """The set for these data on a structure built on the linearization's derivative.

        Args:
            structure: the operators, sets and solver, with the derivative
                ``F'(m*)`` as the forward operator.
            linearization: ``F`` at ``m*``: the point, the value and the
                derivative.
            data: the observations, as observed.

        Raises:
            ValueError: if the structure's forward operator is not the
                linearization's derivative.
        """
        if structure.forward is not linearization.derivative:
            raise ValueError(
                "The structure must be built on the linearization's derivative."
            )
        data_space = structure.data_space
        shifted = data_space.add(
            data_space.subtract(data, linearization.value),
            linearization.derivative(linearization.point),
        )
        feasible = cls.__new__(cls)
        feasible._attach(structure, shifted)
        feasible._attach_linearization(linearization, data)
        return feasible

    @property
    def linearization(self) -> Linearization:
        """The forward operator's value and derivative at the linearization point."""
        return self._linearization

    @property
    def linearization_point(self) -> Any:
        """``m*``, the model the forward operator was linearized about."""
        return self._linearization.point

    @property
    def linearization_property(self) -> Any:
        """``T m*``, the property of the linearization point."""
        return self.property_operator(self._linearization.point)

    @property
    def observed_data(self) -> Any:
        """The data as observed, before the shift to the linearized problem."""
        return self._observed_data

    @property
    def data(self) -> Any:
        """``d − F(m*) + F'(m*) m*``: the data of the linearized problem, which the set is built on."""
        return self._data

    def with_data(self, data: Any, /) -> "LinearizedFeasiblePropertySet":
        """The set for other observed data, about the same linearization point.

        Args:
            data: the observations, as observed.
        """
        return LinearizedFeasiblePropertySet.from_linearization(
            self._structure, self._linearization, data
        )

    def push_forward(
        self, operator: LinearOperator, /
    ) -> "LinearizedFeasiblePropertySet":
        """The same set for a further property ``operator @ T``."""
        structure = FeasibleStructure(
            self._structure.forward,
            operator @ self._structure.property_operator,
            self._structure.prior,
            noise=self._structure.noise,
            solver=self._structure.solver,
        )
        return LinearizedFeasiblePropertySet.from_linearization(
            structure, self._linearization, self._observed_data
        )

    def __repr__(self) -> str:
        return (
            f"LinearizedFeasiblePropertySet(form={self.form!r}, dim={self.domain.dim})"
        )


class BackusGilbertParker(SetEstimator):
    """The feasible property set: what the data and a constraint let a property be.

    In go the forward problem, a convex **constraint set** on the model,
    optionally a convex **confidence set** on the data, and the property
    operator ``T``; out comes the convex set of property values of every
    model that lies in the constraint set and fits the data to within the
    confidence set. The confidence set is taken from the problem when not
    given: its own set if it has one, the ambient ball at ``level`` if its
    error is a measure -- the smallest ball about the expectation, in the
    data space's norm, carrying that probability -- and exact data if it has
    no error at all. The ambient ball is not the credible ellipsoid whose
    misfit threshold :class:`~pygeoinf2.inference.point.MinimumNorm` fits
    to, so the smallest model fitting the data differs between the two;
    ``noise=error.credible_set(level)`` makes them the same.

    The support side of the answer is coordinate-free on every space; the
    membership side forms matrices of the property's dimension and needs a
    coordinate map on the property space, refusing by name without one.

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

    **A nonlinear forward problem** is linearized about a model, given as
    ``model=`` or as a ``linearization=`` already in hand, and the set
    returned is a :class:`LinearizedFeasiblePropertySet`: the feasible
    property set of the linearized problem, in the model's own coordinates
    and the property's own values, exact for the linearized problem and a
    local statement about the real one. The constraint set is the one given
    on the model; the linearization point need not lie in it, but if it does
    not, the forward operator is linearized outside the region the set makes
    a claim about and the local statement is weaker. A linear problem may be
    linearized anywhere and gives the same set.
    """

    def __init__(
        self,
        problem: ForwardProblem,
        property_operator: LinearOperator,
        prior: ConvexSet | SublevelSet,
        /,
        *,
        noise: ConvexSet | SublevelSet | None = None,
        level: float = 0.95,
        solver: LinearSolver | None = None,
        model: Any = None,
        linearization: Linearization | None = None,
    ) -> None:
        """
        Args:
            problem: the forward problem. A nonlinear one needs ``model=``
                or ``linearization=``.
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
            model: the model to linearize the forward operator about.
            linearization: the forward operator's value and derivative at
                the linearization point, when they are already in hand. Not
                with ``model``.

        Raises:
            ValueError: if an operator or set is on the wrong space, or both
                ``model`` and ``linearization`` are given.
            TypeError: if the prior or the confidence set is neither a
                ``ConvexSet`` nor a ``SublevelSet``, the problem's error is
                neither a measure nor a convex set, or the forward operator
                is nonlinear and no linearization point is given.
        """
        if model is not None and linearization is not None:
            raise ValueError("Give model= or linearization=, not both.")
        forward = problem.forward_operator
        if model is not None:
            linearization = forward.at(model)
        if linearization is not None:
            if linearization.derivative.domain != problem.model_space:
                raise ValueError("The linearization must act on the model space.")
            if linearization.derivative.codomain != problem.data_space:
                raise ValueError("The linearization must map into the data space.")
            forward = linearization.derivative
        elif not isinstance(forward, LinearOperator):
            raise TypeError(
                "The forward operator is nonlinear; pass model= to linearize "
                "about a model, or linearization= with a linearization in hand."
            )
        self._problem = problem
        self._level = level
        self._linearization = linearization
        self._structure = FeasibleStructure(
            forward,
            property_operator,
            prior,
            noise=_confidence_set(problem, noise, level),
            solver=solver,
        )

    @property
    def forward_problem(self) -> ForwardProblem:
        """The forward problem."""
        return self._problem

    @property
    def linearization(self) -> Linearization | None:
        """The linearization the set is built on, or ``None`` for a linear problem taken as it is."""
        return self._linearization

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
        if self._linearization is not None:
            return LinearizedFeasiblePropertySet.from_linearization(
                self._structure, self._linearization, data
            )
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
            linearization=self._linearization,
        )

    def __repr__(self) -> str:
        return (
            f"BackusGilbertParker(form={self.form!r}, prior={self.prior!r}, "
            f"noise={self.noise!r})"
        )


# --------------------------------------------------------------------- #
#                      The averaging-kernel estimate                    #
# --------------------------------------------------------------------- #


def _error_covariance(
    problem: LinearForwardProblem, given: LinearOperator | None
) -> LinearOperator | None:
    """``R``: as given, the error measure's, a ball's or an ellipsoid's, or None for exact data."""
    if given is not None:
        if given.domain != problem.data_space:
            raise ValueError("The error covariance must act on the data space.")
        return given
    if not problem.has_error:
        return None
    error = problem.error
    if isinstance(error, ProbabilityMeasure):
        return error.covariance
    if isinstance(error, (Ball, Ellipsoid)):
        covariance = error.covariance
        if covariance is None:
            raise TypeError(
                "The problem's error is an ellipsoid without a covariance; pass "
                "error_covariance=."
            )
        return covariance
    raise TypeError(
        "The problem's error is neither a measure nor a ball or an ellipsoid; "
        "pass error_covariance= to say what the noise penalty weights."
    )


class BackusGilbert(LinearPointEstimator):
    """The averaging-kernel estimate: ``X d`` with ``X A`` as close to ``T`` as the noise allows.

    A fixed operator ``X`` applied to the data, so the estimate sees the
    model through its **resolution** ``X A`` rather than the property ``T``
    asked for, and passes the data error through ``X``. The two are traded
    against each other by the choice

    .. code-block:: text

        X = T C A* (A C A* + α R)⁻¹

    which minimizes ``‖(T − X A) C^½‖² + α ‖X R^½‖²``: ``C`` weights the
    models the kernels have to resolve (the identity by default, so the
    model space's own norm), ``R`` is the data error's covariance, and
    ``α`` is the knob. A small ``α`` buys resolution with noise; a large
    one the reverse. When the data error is a measure with a nonzero
    expectation ``mu_e`` the estimate is ``X (d − mu_e)``, so that the
    noise it passes has mean zero. The estimate is affine in the data, with
    :meth:`resolution` and :meth:`propagated_covariance` from the base class
    as its error budget and :meth:`unresolved`, ``T − X A``, as what it
    cannot see.

    What it does not state is a bound on the unresolved part, because no
    bound on the model has been named. Backus and Gilbert's original
    treatment already reached for one, a norm bound on the model, and the
    feasible property set of :class:`BackusGilbertParker` is the
    completion of that thought: under a prior set and a noise set it is the
    exact set of property values the data cannot rule out. The same ``X``
    with those sets gives a certified set through the feasible set's
    ``linear_certificate`` route, which is valid for any ``X`` and loose
    against the exact one:

    .. code-block:: python

        estimator = BackusGilbert(problem, T, alpha=alpha)
        estimate = estimator(data)
        feasible = BackusGilbertParker(problem, T, prior)(data)
        bound = feasible.support(
            direction, via=LinearCertificate(operator=estimator.operator)
        )
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        property_operator: LinearOperator,
        /,
        *,
        covariance: LinearOperator | None = None,
        error_covariance: LinearOperator | None = None,
        alpha: float = 1.0,
        solver: LinearSolver | None = None,
    ) -> None:
        """
        Args:
            problem: the forward problem.
            property_operator: ``T``, the property operator.
            covariance: ``C``, positive definite on the model space,
                weighting which models the kernels are to resolve. The
                identity if omitted.
            error_covariance: ``R``, on the data space. Taken from the
                problem if omitted: the error measure's covariance, or a
                ball's or an ellipsoid's; with no error at all the noise
                penalty is dropped.
            alpha: the trade-off between resolution and noise, positive.
            solver: how ``A C A* + α R`` is inverted, one solve per
                application of ``X``. Conjugate gradients by default.

        Raises:
            ValueError: if an operator is on the wrong space, or ``alpha``
                is not positive.
            TypeError: if the problem's error is a set that is neither a
                ball nor an ellipsoid and no error covariance is given.
        """
        if property_operator.domain != problem.model_space:
            raise ValueError("The property operator must act on the model space.")
        model_space = problem.model_space
        if covariance is None:
            covariance = LinearOperator.identity(model_space)
        elif covariance.domain != model_space:
            raise ValueError("The covariance must act on the model space.")
        error_covariance = _error_covariance(problem, error_covariance)
        operator = linear_certificate_operator(
            problem.forward_operator,
            property_operator,
            covariance=covariance,
            error_covariance=error_covariance,
            alpha=alpha,
            solver=solver,
        )
        self._attach(
            problem, property_operator, operator, covariance, error_covariance, alpha
        )

    def _attach(
        self,
        problem: LinearForwardProblem,
        property_operator: LinearOperator,
        operator: LinearOperator,
        covariance: LinearOperator,
        error_covariance: LinearOperator | None,
        alpha: float,
    ) -> None:
        error = (
            problem.error
            if problem.has_error and isinstance(problem.error, ProbabilityMeasure)
            else None
        )
        translation = None
        if error is not None and not (
            isinstance(error, GaussianMeasure) and error.has_zero_expectation
        ):
            expectation = error.expectation
            space = operator.codomain

            def translation() -> Any:
                # -X mu_e, one solve, made on first use.
                return space.negative(operator(expectation))

        super().__init__(
            operator,
            translation=translation,
            lazy=True,
            forward_operator=problem.forward_operator,
            error=error,
            property_operator=property_operator,
        )
        self._problem = problem
        self._covariance = covariance
        self._error_covariance = error_covariance
        self._alpha = alpha

    @property
    def forward_problem(self) -> LinearForwardProblem:
        """The forward problem."""
        return self._problem

    @property
    def covariance(self) -> LinearOperator:
        """``C``, the weighting of the models the kernels resolve."""
        return self._covariance

    @property
    def error_covariance(self) -> LinearOperator | None:
        """``R``, the weighting of the noise, or ``None`` when the penalty is dropped."""
        return self._error_covariance

    @property
    def alpha(self) -> float:
        """The trade-off between resolution and noise."""
        return self._alpha

    def unresolved(self) -> LinearOperator:
        """``T − X A``: what the estimate cannot see.

        The complement of the resolution, and the operator whose reach over
        a constraint set is the first part of a certified width. An
        estimate is only as good as this is small.

        Returns:
            The operator from the model space to the property space.
        """
        return self.property_operator - self.resolution()

    def push_forward(self, operator: LinearOperator, /) -> "BackusGilbert":
        """The same kernels for a further property ``operator @ T``: ``operator @ X``."""
        pushed = BackusGilbert.__new__(BackusGilbert)
        pushed._attach(
            self._problem,
            operator @ self.property_operator,
            operator @ self.operator,
            self._covariance,
            self._error_covariance,
            self._alpha,
        )
        return pushed

    def __repr__(self) -> str:
        return f"BackusGilbert(alpha={self._alpha!r})"
