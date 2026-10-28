"""
Point estimators: a data vector in, one model out.

Least squares at a damping chosen by the caller (:class:`LeastSquares`); the
smallest model that fits the data, with the damping found from the data by
the discrepancy principle (:class:`MinimumNorm`); and their counterparts
confined to an affine subspace, :class:`ConstrainedLeastSquares` and
:class:`ConstrainedMinimumNorm`.

The fixed-damping estimators are built on
:class:`~pygeoinf2.numerics.tikhonov.TikhonovNormalOperator`, so they expose
the operator they invert, and :class:`LeastSquares` can be preconditioned,
given a surrogate, parameterized, data-reduced, or swept over its damping.

The distinction that runs through the module is between a damping *chosen* and
a damping *found*. With the damping fixed, the estimate is affine in the data
and the estimator is an operator, which is what
:class:`LinearPointEstimator` is for. With the damping found from the data by
the discrepancy principle, it is not: the map is genuinely non-linear, and
:class:`MinimumNorm` supplies its derivative rather than pretending
otherwise.
"""

from __future__ import annotations

from typing import Any, Callable

from ..algebra.operators import Linearization, LinearOperator, Operator
from ..algebra.spaces import HilbertSpace
from ..geometry.sets import Subset
from ..geometry.subspaces import AffineSubspace
from ..numerics.normal import Formalism
from ..numerics.normal import choose_formalism as _choose
from ..numerics.root_find import RootResult
from ..numerics.solvers import LinearSolver, SolveStep, resolve_solver
from ..numerics.tikhonov import TikhonovFamily, TikhonovNormalOperator, misfit_search
from ..probability.base import ProbabilityMeasure
from ..probability.gaussian import GaussianMeasure
from .estimators import ComposedPointEstimator, LinearPointEstimator, PointEstimator
from .problem import LinearForwardProblem

__all__ = [
    "ConstrainedLeastSquares",
    "ConstrainedMinimumNorm",
    "LeastSquares",
    "MinimumNorm",
    "choose_formalism",
]


def choose_formalism(
    problem: LinearForwardProblem, /, *, formalism: Formalism = "auto"
) -> str:
    """Which space to assemble *this problem's* normal equations in.

    The spaces-based decision of
    :func:`~pygeoinf2.numerics.normal.choose_formalism`, read off a problem.

    Args:
        problem: the forward problem.
        formalism: ``"model_space"``, ``"data_space"``, or ``"auto"``.

    Returns:
        The formalism to use.
    """
    return _choose(problem.model_space, problem.data_space, formalism=formalism)


def _error_measure(problem: LinearForwardProblem) -> Any:
    """The problem's error measure, or None when it has none."""
    return problem.error_measure if problem.has_error else None


class LeastSquares(LinearPointEstimator):
    """Tikhonov-regularized least squares: minimize ``|A u - d|^2_R + t |u|^2``.

    The estimator *is* the mapping from data to the fitted model, so it is an
    affine operator and joins the algebra. The damping is fixed here; to let
    the data choose it, see :class:`MinimumNorm`.
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        /,
        *,
        damping: float = 0.0,
        solver: LinearSolver | None = None,
        formalism: Formalism = "data_space",
    ) -> None:
        """
        Args:
            problem: the forward problem.
            damping: the Tikhonov parameter. Must be positive unless the
                normal operator is invertible without it.
            solver: how to invert the normal operator. Conjugate gradients by
                default. To precondition, pass an iterative solver carrying
                one; a preconditioner that is itself a ``LinearSolver`` is
                handed :attr:`normal_operator`, which still knows its factors.
                For one built from *other* factors, pass a callable taking the
                normal operator and returning the solver — see
                :func:`~pygeoinf2.numerics.solvers.resolve_solver`.
            formalism: which space to solve in; see :func:`choose_formalism`.
        """
        # The normal operator is built before the solver, which only inverts
        # it, so a caller may pass a factory that builds a preconditioner from
        # the operator about to be inverted.
        normal = TikhonovNormalOperator(
            problem.forward_operator,
            damping,
            error=_error_measure(problem),
            formalism=formalism,
        )
        solver = resolve_solver(solver, normal)
        inverse = solver(normal)

        if normal.formalism == "model_space":
            operator = inverse @ normal.weighted_adjoint()
        else:
            operator = problem.forward_operator.adjoint @ inverse

        translation = None
        if problem.has_error:
            expectation = problem.error_measure.expectation
            shift = operator(expectation)
            if problem.model_space.norm(shift) > 0.0:
                translation = problem.model_space.negative(shift)

        super().__init__(
            operator,
            translation=translation,
            forward_operator=problem.forward_operator,
            error=_error_measure(problem),
        )
        self._problem = problem
        self._normal = normal
        self._inverse = inverse
        self._damping = float(damping)
        self._solver = solver

    # ----------------------------------------------------------------- #
    #                       What it inverts                             #
    # ----------------------------------------------------------------- #

    @property
    def normal_operator(self) -> TikhonovNormalOperator:
        """The operator being inverted, with its factors still attached.

        What a preconditioner is built against, and what to look at when a
        solve behaves badly — its condition number is the problem's, not the
        forward operator's.
        """
        return self._normal

    @property
    def inverse_normal_operator(self) -> LinearOperator:
        """The inverse the solver produced."""
        return self._inverse

    def right_hand_side(self, data: Any, /) -> Any:
        """The right-hand side of the normal equations for this data."""
        return self._normal.right_hand_side(data)

    @property
    def formalism(self) -> str:
        """Which space the normal equations were assembled in."""
        return self._normal.formalism

    @property
    def damping(self) -> float:
        """The Tikhonov parameter used."""
        return self._damping

    @property
    def forward_problem(self) -> LinearForwardProblem:
        """The forward problem being inverted."""
        return self._problem

    @property
    def solver(self) -> LinearSolver:
        """How the normal operator is inverted."""
        return self._solver

    # ----------------------------------------------------------------- #
    #                          The same, but                            #
    # ----------------------------------------------------------------- #

    def with_solver(self, solver: Any, /) -> "LeastSquares":
        """The same estimator, solved a different way."""
        return type(self)(
            self._problem,
            damping=self._damping,
            solver=solver,
            formalism=self._normal.formalism,
        )

    def with_formalism(self, formalism: Formalism, /) -> "LeastSquares":
        """The same estimator, assembled in the other space.

        The two formalisms give the same answer, to the solver's tolerance.
        """
        return type(self)(
            self._problem,
            damping=self._damping,
            solver=self._solver,
            formalism=formalism,
        )

    def with_damping(self, damping: float, /) -> "LeastSquares":
        """The same estimator at a different damping.

        What an L-curve walks along. Building a
        :class:`~pygeoinf2.numerics.tikhonov.TikhonovFamily` directly is
        cheaper for a long sweep, because it keeps the undamped pieces and can
        warm-start each solve from the last.
        """
        return type(self)(
            self._problem,
            damping=damping,
            solver=self._solver,
            formalism=self._normal.formalism,
        )

    def family(self) -> TikhonovFamily:
        """The one-parameter family this estimator is one member of."""
        return TikhonovFamily(
            self._problem.forward_operator,
            error=_error_measure(self._problem),
            solver=self._solver,
            formalism=self._normal.formalism,
        )

    def surrogate(
        self,
        /,
        *,
        forward: LinearOperator | None = None,
        error: GaussianMeasure | None = None,
        damping: float | None = None,
        formalism: Formalism | None = None,
    ) -> TikhonovNormalOperator:
        """A cheap stand-in for this estimator's normal operator.

        Args:
            forward: a cheaper ``A``, sharing the data space.
            error: a cheaper ``R``.
            damping: a different damping.
            formalism: which space to assemble in. Inherited if omitted.

        Returns:
            The surrogate normal operator, which is the part a preconditioner
            is built from. It may live on a different model space, as long as
            the data space is shared.
        """
        return self._normal.surrogate(
            forward=forward, error=error, damping=damping, formalism=formalism
        )

    def parameterized(self, parameterization: LinearOperator, /) -> "LeastSquares":
        """The same estimator restricted to a parameter space.

        Args:
            parameterization: maps the parameter space into the model space.

        Returns:
            An estimator of the same kind on the parameterized problem.
        """
        return type(self)(
            self._problem.parameterized(parameterization),
            damping=self._damping,
            solver=self._solver,
            formalism=self._normal.formalism,
        )

    def data_reduced(
        self,
        reduction: LinearOperator,
        /,
        *,
        error: ProbabilityMeasure | Subset | None = None,
    ) -> "LeastSquares":
        """The same estimator with the data compressed by *reduction*.

        Args:
            reduction: maps the data space to the reduced one.
            error: the reduced error. Defaults to pushing the current one
                forward, as on the problem itself.

        Returns:
            An estimator of the same kind on the reduced problem.
        """
        return type(self)(
            self._problem.data_reduced(reduction, error=error),
            damping=self._damping,
            solver=self._solver,
            formalism=self._normal.formalism,
        )

    def residual_callback(
        self,
        data: Any,
        /,
        *,
        message: str = "iteration {iteration}: normal residual {residual:.3e}",
        report: Callable[[str], None] = print,
    ) -> Any:
        """A solver callback that reports the normal-equation residual.

        Wired to *these* normal equations and *this* right-hand side, so the number printed is the one the solve
        is actually driving down rather than whatever the solver happens to
        track internally.

        Args:
            data: the observations the solve is for.
            message: a format string taking ``iteration`` and ``residual``.
                The residual is *relative* to the right-hand side, which is
                what makes it comparable between problems.
            report: where the line goes. ``print`` by default; pass a logger's
                method to send it somewhere else.

        Returns:
            A callback to hand to an iterative solver.
        """
        space = self._normal.domain
        target = self.right_hand_side(data)
        scale = space.norm(target)
        if scale == 0.0:
            scale = 1.0

        def callback(step: SolveStep) -> None:
            report(
                message.format(iteration=step.iteration, residual=step.residual / scale)
            )

        return callback


def _discrepancy_search(
    family: TikhonovFamily,
    problem: LinearForwardProblem,
    data: Any,
    /,
    *,
    level: float = 0.95,
    max_iterations: int = 60,
    rtol: float = 1e-6,
    atol: float = 0.0,
    minimum_damping: float = 0.0,
) -> RootResult:
    """Find the damping at which the chi-squared misfit reaches its threshold."""
    return misfit_search(
        family,
        family.right_hand_side(data),
        lambda model: problem.chi_squared(model, data),
        problem.critical_chi_squared(level=level),
        max_iterations=max_iterations,
        rtol=rtol,
        atol=atol,
        minimum_damping=minimum_damping,
    )


def _searched_damping(
    family: TikhonovFamily,
    problem: LinearForwardProblem,
    data: Any,
    /,
    *,
    level: float = 0.95,
    max_iterations: int = 60,
    rtol: float = 1e-6,
    atol: float = 0.0,
    minimum_damping: float = 0.0,
) -> RootResult:
    """The discrepancy search, refusing data that no model fits.

    Raises:
        ValueError: when the misfit does not reach its target at any damping.
    """
    found = _discrepancy_search(
        family,
        problem,
        data,
        level=level,
        max_iterations=max_iterations,
        rtol=rtol,
        atol=atol,
        minimum_damping=minimum_damping,
    )
    if found.exhausted == "low":
        # No damping small enough brings the misfit to its target, so the
        # principle has no solution: these data cannot be fitted to this level
        # by any model this problem admits. The least-damped model is *not* a
        # fallback -- it solves a numerically singular system, and its size
        # says nothing about the data.
        target = problem.critical_chi_squared(level=level)
        raise ValueError(
            f"The data cannot be fitted to the chi-squared threshold at "
            f"level {level}: the misfit is still {found.value:.4g} against a "
            f"target of {target:.4g} at the smallest damping the normal "
            f"operator survives. Lower the level, widen the model, or use "
            f"LeastSquares with a chosen damping."
        )
    return found


class MinimumNorm(PointEstimator, Operator):
    """The smallest model that fits the data.

    When the problem carries an error measure, "fits" is the discrepancy
    principle: the chi-squared misfit is allowed up to its critical value at
    the confidence *level*, and the estimate is the smallest model whose
    misfit is no larger. That model is the Tikhonov minimizer of
    ``|A u - d|^2_R + t |u|^2`` at the largest damping ``t`` whose misfit
    still reaches the threshold, found by a one-dimensional root search since
    the misfit increases with the damping. The level is the one knob: it
    says how much of the data to attribute to noise.

    Without an error measure the data are exact, there is no threshold, and
    the estimate is the fit at zero damping: the minimum-norm solution of
    ``A u == d``.

    Two saturated cases of the search are handled explicitly, both read off
    :func:`~pygeoinf2.numerics.root_find.monotone_root` reporting which end
    of the range it ran out at:

    * the misfit stays below the threshold however large the damping, so
      every damping fits and the answer is the **largest** reached: the data
      support no structure and the model should say so;
    * the misfit stays above the threshold however small the damping, so no
      model fits and there is no answer to give. That is refused.

    The damping is found from the data, so this map is **not** affine and
    not an operator in the linear sense: two data vectors that need
    different dampings are not related by any fixed matrix. What makes it
    usable anyway is that its derivative can be written down exactly.
    Differentiating ``H(t) u == A* R^-1 d`` and the constraint
    ``chi^2(u, d) == target`` together, and using ``A* R^-1 r == -t u`` from
    the normal equations themselves:

    .. code-block:: text

        du/dd == L - h (x) w
        h == H(t)^-1 u
        w == (L* u + R^-1 r / t) / (u, h)

    with ``L`` the fixed-damping estimator and ``r`` the residual. The first
    term is what the answer would do if the damping were frozen; the second is
    the correction for the damping moving, and it is a rank-one update, which
    is why the derivative costs one extra solve rather than a new problem.
    Both directions are supplied, and the adjoint is the one an inversion
    built on top of this would actually call.

    :meth:`for_data` returns the :class:`LeastSquares` at the damping the
    data chose, which is an affine operator with the error budget of one;
    :meth:`discrepancy_search` returns the search itself with its cost.
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        /,
        *,
        level: float = 0.95,
        solver: LinearSolver | None = None,
        formalism: Formalism = "data_space",
        max_iterations: int = 60,
        rtol: float = 1e-6,
        atol: float = 0.0,
        minimum_damping: float = 0.0,
    ) -> None:
        """
        Args:
            problem: the forward problem, with an error measure or without.
            level: the confidence level setting the misfit target. Unused
                when the problem has no error.
            solver: how to invert each ``N(t)``. Conjugate gradients by
                default; see :class:`LeastSquares`.
            formalism: which space to solve in; see :func:`choose_formalism`.
            max_iterations: the discrepancy search's budget.
            rtol: the search's bracket tolerance.
            atol: the absolute half of that tolerance, for a damping near
                zero, where a relative test never closes.
            minimum_damping: a floor on the damping. Data that no model fits
                are refused once the search reaches it, rather than after
                probing ever smaller dampings on a numerically singular
                system. Zero is no floor.
        """
        super().__init__(problem.data_space, problem.model_space)
        self._problem = problem
        self._level = level
        self._solver = solver
        self._formalism = formalism
        self._max_iterations = max_iterations
        self._rtol = rtol
        self._atol = atol
        self._minimum_damping = minimum_damping
        self._family = TikhonovFamily(
            problem.forward_operator,
            error=_error_measure(problem),
            solver=self._solver,
            formalism=formalism,
        )
        self._exact_fit: LeastSquares | None = None

    # ----------------------------------------------------------------- #
    #                            Structure                              #
    # ----------------------------------------------------------------- #

    @property
    def forward_problem(self) -> LinearForwardProblem:
        """The forward problem."""
        return self._problem

    @property
    def level(self) -> float:
        """The confidence level the misfit is damped towards."""
        return self._level

    @property
    def solver(self) -> LinearSolver | None:
        """How each damped normal operator is inverted; None for the default."""
        return self._solver

    @property
    def formalism(self) -> str:
        """Which space the normal equations are assembled in."""
        return self._family.formalism

    @property
    def data_space(self) -> HilbertSpace:
        """The problem's data space."""
        return self._problem.data_space

    @property
    def property_space(self) -> HilbertSpace:
        """The problem's model space: the estimate is of the model itself."""
        return self._problem.model_space

    def family(self) -> TikhonovFamily:
        """The one-parameter family the search walks along."""
        return self._family

    def __call__(self, data: Any) -> Any:
        """The smallest model fitting the data."""
        return self._value(data)

    # ----------------------------------------------------------------- #
    #                          The same, but                            #
    # ----------------------------------------------------------------- #

    def _rebuild(
        self, *, problem: LinearForwardProblem | None = None, **changes: Any
    ) -> "MinimumNorm":
        """A copy with the given settings replaced."""
        settings: dict[str, Any] = dict(
            level=self._level,
            solver=self._solver,
            formalism=self._formalism,
            max_iterations=self._max_iterations,
            rtol=self._rtol,
            atol=self._atol,
            minimum_damping=self._minimum_damping,
        )
        settings.update(changes)
        return type(self)(self._problem if problem is None else problem, **settings)

    def with_solver(self, solver: Any, /) -> "MinimumNorm":
        """The same estimator, solved a different way."""
        return self._rebuild(solver=solver)

    def with_formalism(self, formalism: Formalism, /) -> "MinimumNorm":
        """The same estimator, assembled in the other space."""
        return self._rebuild(formalism=formalism)

    def with_level(self, level: float, /) -> "MinimumNorm":
        """The same estimator at another confidence level."""
        return self._rebuild(level=level)

    def parameterized(self, parameterization: LinearOperator, /) -> "MinimumNorm":
        """The same estimator restricted to a parameter space.

        Args:
            parameterization: maps the parameter space into the model space.

        Returns:
            The estimator on the parameterized problem.
        """
        return self._rebuild(problem=self._problem.parameterized(parameterization))

    def data_reduced(
        self,
        reduction: LinearOperator,
        /,
        *,
        error: ProbabilityMeasure | Subset | None = None,
    ) -> "MinimumNorm":
        """The same estimator with the data compressed by *reduction*.

        Args:
            reduction: maps the data space to the reduced one.
            error: the reduced error. Defaults to pushing the current one
                forward, as on the problem itself.

        Returns:
            The estimator on the reduced problem.
        """
        return self._rebuild(problem=self._problem.data_reduced(reduction, error=error))

    # ----------------------------------------------------------------- #
    #                            The search                             #
    # ----------------------------------------------------------------- #

    def _settings(
        self,
        level: float | None,
        max_iterations: int | None,
        rtol: float | None,
        atol: float | None,
        minimum_damping: float | None,
    ) -> dict[str, Any]:
        """The search's settings, the estimator's own where none is given."""
        return dict(
            level=self._level if level is None else level,
            max_iterations=(
                self._max_iterations if max_iterations is None else max_iterations
            ),
            rtol=self._rtol if rtol is None else rtol,
            atol=self._atol if atol is None else atol,
            minimum_damping=(
                self._minimum_damping if minimum_damping is None else minimum_damping
            ),
        )

    def _require_error(self, what: str) -> None:
        if not self._problem.has_error:
            raise ValueError(
                f"The problem has no error measure, so there is no misfit "
                f"threshold and no {what}: the data are exact and the estimate "
                f"is the fit at zero damping, estimator_at(0.0)."
            )

    def discrepancy_search(
        self,
        data: Any,
        /,
        *,
        level: float | None = None,
        max_iterations: int | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        minimum_damping: float | None = None,
    ) -> RootResult:
        """The damping search for this data, with its diagnostics.

        Returned rather than hidden so that the cost is visible: without the
        iteration count there is no way to tell a warm start that is working
        from one that silently is not. Saturation is reported, not raised;
        :meth:`for_data` is the route that refuses unfittable data.

        Args:
            data: the observations.
            level: the confidence level setting the misfit target. The
                estimator's own if omitted, as for every setting below.
            max_iterations: bisection steps once the root is bracketed.
            rtol: how tightly to close the bracket.
            atol: its absolute half, for a root near zero damping.
            minimum_damping: a floor on the damping; see
                :func:`~pygeoinf2.numerics.tikhonov.misfit_search`.

        Returns:
            The root result, with its bracket, evaluation count and inner
            iteration total.

        Raises:
            ValueError: if the problem has no error measure.
        """
        self._require_error("search")
        return _discrepancy_search(
            self._family,
            self._problem,
            data,
            **self._settings(level, max_iterations, rtol, atol, minimum_damping),
        )

    def search(self, data: Any, /) -> RootResult:
        """The damping search for this data with the estimator's own settings.

        Args:
            data: the observations.

        Returns:
            The root result; see :meth:`discrepancy_search`.

        Raises:
            ValueError: if the problem has no error measure.
        """
        return self.discrepancy_search(data)

    def for_data(
        self,
        data: Any,
        /,
        *,
        level: float | None = None,
        max_iterations: int | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        minimum_damping: float | None = None,
    ) -> LeastSquares:
        """The fixed-damping estimator at the damping these data choose.

        Damps as hard as the data allow: the largest damping whose misfit still
        reaches the chi-squared threshold at *level*. Data that anything fits
        give the largest damping reached and a model close to zero; data that
        nothing fits are refused. With exact data the fit is at zero damping.

        The result is affine in the data, so it is an operator and carries
        the error budget of one; applying it to the same data gives this
        estimator's answer.

        Args:
            data: the data vector.
            level: the confidence level setting the misfit target. The
                estimator's own if omitted, as for every setting below.
            max_iterations: the search's budget.
            rtol: the search's bracket tolerance.
            atol: its absolute half, for a root near zero damping.
            minimum_damping: a floor on the damping; see
                :func:`~pygeoinf2.numerics.tikhonov.misfit_search`.

        Returns:
            A :class:`LeastSquares` with its damping fixed at the value found.

        Raises:
            ValueError: when no model fits the data to the threshold.
        """
        if not self._problem.has_error:
            return self.estimator_at(0.0)
        found = _searched_damping(
            self._family,
            self._problem,
            data,
            **self._settings(level, max_iterations, rtol, atol, minimum_damping),
        )
        return self.estimator_at(found.argument)

    def estimator_at(self, damping: float, /) -> LeastSquares:
        """The fixed-damping estimator this collapses to at one damping.

        Args:
            damping: the Tikhonov parameter.

        Returns:
            The :class:`LeastSquares` with this estimator's solver and
            formalism.
        """
        return LeastSquares(
            self._problem,
            damping=damping,
            solver=self._solver,
            formalism=self._formalism,
        )

    # ----------------------------------------------------------------- #
    #                       Value and derivative                        #
    # ----------------------------------------------------------------- #

    def _exact(self) -> LeastSquares:
        """The zero-damping fit, built on first use."""
        if self._exact_fit is None:
            self._exact_fit = self.estimator_at(0.0)
        return self._exact_fit

    def _resolve(self, data: Any) -> tuple[Any, float, bool]:
        """The model, the damping, and whether the damping *moves* with the data.

        The last is what the derivative turns on. A damping found at an
        interior root is a function of the data and contributes a term; a
        damping pinned at the end of its range is not, and contributes none.
        """
        target = self._problem.critical_chi_squared(level=self._level)
        model_space = self._problem.model_space
        if self._problem.chi_squared(model_space.zero(), data) <= target:
            # The zero model already fits, so it is the smallest that does.
            return model_space.zero(), 0.0, False
        found = _searched_damping(
            self._family,
            self._problem,
            data,
            level=self._level,
            max_iterations=self._max_iterations,
            rtol=self._rtol,
            atol=self._atol,
            minimum_damping=self._minimum_damping,
        )
        return (
            self._family.model_from(found.solution),
            found.argument,
            found.converged,
        )

    def _value(self, data: Any) -> Any:
        if not self._problem.has_error:
            return self._exact()(data)
        model, _, _ = self._resolve(data)
        return model

    def _linearize(self, data: Any) -> Linearization:
        """The model and its derivative from a *single* damping search.

        Without this, ``at(data)`` would call :meth:`_value` and
        :meth:`_derivative` in turn, each running :meth:`_resolve`, so the
        root find over the damping -- the whole cost of this operator -- would
        run twice for one linearization.
        """
        if not self._problem.has_error:
            return self._exact().at(data)
        resolved = self._resolve(data)
        return Linearization(data, resolved[0], self._derivative_from(data, *resolved))

    def _derivative(self, data: Any) -> LinearOperator:
        if not self._problem.has_error:
            return self._exact().operator
        return self._derivative_from(data, *self._resolve(data))

    def _derivative_from(
        self, data: Any, model: Any, damping: float, moves: bool, /
    ) -> LinearOperator:
        """The derivative, given a search that has already been run."""
        model_space = self._problem.model_space
        data_space = self._problem.data_space
        if damping == 0.0:
            # Locally constant: the zero model fits, and still fits after a
            # small perturbation of the data.
            return LinearOperator.from_callables(
                data_space,
                model_space,
                lambda _: model_space.zero(),
                adjoint=lambda _: data_space.zero(),
            )

        fixed = self.estimator_at(damping)
        linear = fixed.operator
        if not moves:
            # The search saturated: no damping in range brings the misfit to
            # its target, so the one in force is pinned by the end of the
            # range rather than chosen by the data. It does not move when the
            # data do, and the rank-one correction for its movement would be
            # not merely unnecessary but wrong.
            return linear
        forward = self._problem.forward_operator
        error = self._problem.error_measure

        # h == H(t)^-1 u, through the identity that fits the formalism.
        if fixed.formalism == "model_space":
            h = fixed.inverse_normal_operator(model)
        else:
            # (A* R^-1 A + t I)^-1 == (1/t)[I - A* (A A* + t R)^-1 A].
            pulled = forward.adjoint(fixed.inverse_normal_operator(forward(model)))
            h = model_space.scale(1.0 / damping, model_space.subtract(model, pulled))

        shifted = data_space.subtract(data, error.expectation)
        residual = data_space.subtract(forward(model), shifted)
        weighted = error.precision(residual)
        w = data_space.scale(
            1.0 / model_space.inner_product(model, h),
            data_space.add(
                linear.adjoint(model), data_space.scale(1.0 / damping, weighted)
            ),
        )

        def value(perturbation: Any) -> Any:
            frozen = linear(perturbation)
            movement = data_space.inner_product(w, perturbation)
            return model_space.subtract(frozen, model_space.scale(movement, h))

        def adjoint(perturbation: Any) -> Any:
            pulled_back = linear.adjoint(perturbation)
            movement = model_space.inner_product(h, perturbation)
            return data_space.subtract(pulled_back, data_space.scale(movement, w))

        return LinearOperator.from_callables(
            data_space, model_space, value, adjoint=adjoint
        )

    def push_forward(self, operator: LinearOperator, /) -> ComposedPointEstimator:
        """The estimator of ``T m``: this one followed by ``T``, derivative and all.

        The composition carries no error budget, since the damping moves with
        the data. For a budget, take :meth:`for_data` at the data in hand and
        push that affine estimator forward: its resolution and propagated
        covariance are those of the fixed damping it chose, and leave out the
        damping's own dependence on the data.

        Args:
            operator: ``T``, acting on the model space.

        Returns:
            The composed estimator.
        """
        return ComposedPointEstimator(self, operator)


class ConstrainedLeastSquares(LinearPointEstimator):
    """Least squares within an affine subspace.

    For a constraint that is *exact* rather than probable — a boundary
    condition, a fixed total mass, a known mean — which a prior cannot express
    and a penalty only approximates.

    The subspace is written as ``t + range(P)``, so substituting ``u == t + P w``
    turns the constrained problem into an unconstrained one for the operator
    ``A P`` and the data ``d - A t``. The answer is affine in the data, so this
    is still an operator and still joins the algebra: the projector supplies
    the linear part and the translation the offset.
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        subspace: AffineSubspace,
        /,
        *,
        damping: float = 0.0,
        solver: LinearSolver | None = None,
        formalism: Formalism = "data_space",
    ) -> None:
        """
        Args:
            problem: the forward problem.
            subspace: an ``AffineSubspace`` of the model space.
            damping: the Tikhonov parameter, applied within the subspace.
            solver: how to invert the reduced normal operator.
            formalism: which space to solve in.
        """
        if subspace.domain != problem.model_space:
            raise ValueError("The subspace must live in the model space.")
        model_space = problem.model_space
        projector = subspace.projector
        translation = subspace.translation

        reduced = _reduced_problem(problem, subspace)
        inner = LeastSquares(
            reduced, damping=damping, solver=solver, formalism=formalism
        )
        operator = projector @ inner.operator
        offset = model_space.subtract(
            translation, operator(problem.forward_operator(translation))
        )

        super().__init__(
            operator,
            translation=offset,
            forward_operator=problem.forward_operator,
            error=_error_measure(problem),
        )
        self._problem = problem
        self._subspace = subspace
        self._inner = inner
        self._damping = damping
        self._solver = solver
        self._formalism = formalism

    @property
    def subspace(self) -> AffineSubspace:
        """The affine subspace the answer is confined to."""
        return self._subspace

    @property
    def damping(self) -> float:
        """The Tikhonov parameter used within the subspace."""
        return self._damping

    @property
    def reduced(self) -> LeastSquares:
        """The unconstrained estimator on the reduced problem."""
        return self._inner

    @property
    def normal_operator(self) -> TikhonovNormalOperator:
        """The reduced normal operator, which is the one actually inverted."""
        return self._inner.normal_operator

    @property
    def formalism(self) -> str:
        """Which space the reduced normal equations were assembled in."""
        return self._inner.formalism

    def with_solver(self, solver: Any, /) -> "ConstrainedLeastSquares":
        """The same estimator, solved a different way."""
        return type(self)(
            self._problem,
            self._subspace,
            damping=self._damping,
            solver=solver,
            formalism=self._formalism,
        )

    def with_formalism(self, formalism: Formalism, /) -> "ConstrainedLeastSquares":
        """The same estimator, assembled in the other space."""
        return type(self)(
            self._problem,
            self._subspace,
            damping=self._damping,
            solver=self._solver,
            formalism=formalism,
        )

    def parameterized(
        self, parameterization: LinearOperator, /
    ) -> "ConstrainedLeastSquares":
        """The same estimator restricted to a parameter space.

        The constraint is pulled back with the problem: ``B u == w`` in the
        model space becomes ``(B M) p == w`` in the parameter space, where
        ``M`` is the parameterization. This works because a constraint written
        as an equation says what it means about any model, including one built
        from parameters. It therefore needs the subspace to remember the
        equation it was built from, which
        :attr:`~pygeoinf2.geometry.subspaces.AffineSubspace.has_explicit_equation`
        reports.

        Args:
            parameterization: ``M``, from the parameter space into the model
                space.

        Returns:
            The estimator on the parameter space.

        Raises:
            NotImplementedError: if the subspace was built from a basis rather
                than an equation. A basis fixes the solution set but not which
                equation defines it, and any equation invented here would be a
                different one.
            ValueError: if the parameter space is too small to carry the
                constraints.
        """
        subspace = _parameterized_subspace(self._subspace, parameterization)
        return type(self)(
            self._problem.parameterized(parameterization),
            subspace,
            damping=self._damping,
            solver=self._solver,
            formalism=self._formalism,
        )

    def _parameterized_subspace(
        self, parameterization: LinearOperator, /
    ) -> AffineSubspace:
        """The constraint, read in the parameter space."""
        return _parameterized_subspace(self._subspace, parameterization)

    def data_reduced(self, *args: Any, **kwargs: Any) -> "ConstrainedLeastSquares":
        """The same estimator on a reduced set of data.

        The constraint lives in the *model* space and a data reduction does not
        touch it, so unlike :meth:`parameterized` there is nothing to pull
        back: the reduction applies to the problem alone.
        """
        return type(self)(
            self._problem.data_reduced(*args, **kwargs),
            self._subspace,
            damping=self._damping,
            solver=self._solver,
            formalism=self._formalism,
        )


def _parameterized_subspace(
    subspace: AffineSubspace, parameterization: LinearOperator, /
) -> AffineSubspace:
    """A constraint pulled back through a parameterization.

    ``B u == w`` in the model space becomes ``(B M) p == w`` in the parameter
    space. This works because a constraint written as an equation says what it
    means about any model, including one assembled from parameters.

    Args:
        subspace: the constraint, which must remember its equation.
        parameterization: ``M``, from the parameter space into the model space.

    Returns:
        The constraint in the parameter space.

    Raises:
        NotImplementedError: if the subspace was built from a basis rather than
            an equation. A basis fixes the solution set but not which equation
            defines it, and any equation invented here would be a different
            one with the same solutions.
        ValueError: if the parameter space is too small to carry the
            constraints.
    """
    if not subspace.has_explicit_equation:
        raise NotImplementedError(
            "Parameterizing a constrained inversion needs the constraint as "
            "an equation, and this subspace was built from a basis. Rebuild "
            "it with AffineSubspace.from_linear_equation, or use "
            "to_hyperplanes() for an equation with the same solution set."
        )
    constraint = subspace.constraint_operator
    if constraint.codomain.dim > parameterization.domain.dim:
        raise ValueError(
            f"The parameter space has dimension {parameterization.domain.dim}, "
            f"which cannot carry {constraint.codomain.dim} constraints."
        )
    return AffineSubspace.from_linear_equation(
        constraint @ parameterization, subspace.constraint_value
    )


def _reduced_problem(
    problem: LinearForwardProblem, subspace: AffineSubspace, /
) -> LinearForwardProblem:
    """The unconstrained problem inside an affine subspace.

    ``A P`` in place of ``A``, so that a model of the reduced problem is a set
    of coordinates within the subspace rather than a model.
    """
    return LinearForwardProblem(
        problem.forward_operator @ subspace.projector,
        error=problem.error if problem.has_error else None,
    )


class ConstrainedMinimumNorm(PointEstimator, Operator):
    """The smallest model inside an affine subspace that fits the data.

    The constrained counterpart of :class:`MinimumNorm`, and built the same
    way :class:`ConstrainedLeastSquares` is: substitute ``u == c + P w``,
    solve the unconstrained problem for ``w`` against the shifted data
    ``d - A c``, and add ``c`` back. "Fits" is the discrepancy principle
    when the problem carries an error measure and exact agreement when it
    does not, as for the unconstrained class.

    The derivative follows from the unconstrained one by the chain rule, and
    since both substitutions are affine with constant parts, it is simply the
    reduced problem's derivative evaluated at the shifted data.
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        subspace: AffineSubspace,
        /,
        *,
        level: float = 0.95,
        solver: LinearSolver | None = None,
        formalism: Formalism = "data_space",
        max_iterations: int = 60,
        rtol: float = 1e-6,
    ) -> None:
        """
        Args:
            problem: the forward problem, with an error measure or without.
            subspace: an ``AffineSubspace`` of the model space.
            level: the confidence level setting the misfit target.
            solver: how to invert each reduced ``N(t)``.
            formalism: which space to solve in.
            max_iterations: the discrepancy search's budget.
            rtol: the search's bracket tolerance.

        Raises:
            ValueError: if the subspace is not in the model space.
        """
        if subspace.domain != problem.model_space:
            raise ValueError("The subspace must live in the model space.")
        super().__init__(problem.data_space, problem.model_space)
        self._problem = problem
        self._subspace = subspace
        self._projector = subspace.projector
        self._translation = subspace.translation
        self._inner = MinimumNorm(
            _reduced_problem(problem, subspace),
            level=level,
            solver=solver,
            formalism=formalism,
            max_iterations=max_iterations,
            rtol=rtol,
        )
        self._offset = problem.forward_operator(self._translation)
        # Kept so parameterized() and data_reduced() can rebuild with the
        # same settings; the inner method does not expose all of them.
        self._level = level
        self._solver = solver
        self._formalism = formalism

    @property
    def forward_problem(self) -> LinearForwardProblem:
        """The forward problem."""
        return self._problem

    @property
    def data_space(self) -> HilbertSpace:
        """The problem's data space."""
        return self._problem.data_space

    @property
    def property_space(self) -> HilbertSpace:
        """The problem's model space: the estimate is of the model itself."""
        return self._problem.model_space

    def __call__(self, data: Any) -> Any:
        """The smallest model in the subspace fitting the data."""
        return self._value(data)

    def push_forward(self, operator: LinearOperator, /) -> ComposedPointEstimator:
        """The estimator of ``T m``: this one followed by ``T``, derivative and all.

        Args:
            operator: ``T``, acting on the model space.

        Returns:
            The composed estimator.
        """
        return ComposedPointEstimator(self, operator)

    @property
    def subspace(self) -> AffineSubspace:
        """The affine subspace the answer is confined to."""
        return self._subspace

    @property
    def reduced(self) -> MinimumNorm:
        """The unconstrained method on the reduced problem."""
        return self._inner

    def parameterized(
        self, parameterization: LinearOperator, /
    ) -> "ConstrainedMinimumNorm":
        """The same method restricted to a parameter space.

        As :meth:`ConstrainedLeastSquares.parameterized`: the constraint
        ``B u == w`` is pulled back to ``(B M) p == w``, which needs the
        subspace to remember the equation it was built from.
        """
        return type(self)(
            self._problem.parameterized(parameterization),
            _parameterized_subspace(self._subspace, parameterization),
            level=self._level,
            solver=self._solver,
            formalism=self._formalism,
        )

    def data_reduced(self, *args: Any, **kwargs: Any) -> "ConstrainedMinimumNorm":
        """The same method on a reduced set of data.

        The constraint is in the model space, so a data reduction leaves it
        alone and only the problem changes.
        """
        return type(self)(
            self._problem.data_reduced(*args, **kwargs),
            self._subspace,
            level=self._level,
            solver=self._solver,
            formalism=self._formalism,
        )

    def _shifted(self, data: Any) -> Any:
        return self._problem.data_space.subtract(data, self._offset)

    def _value(self, data: Any) -> Any:
        coordinates = self._inner(self._shifted(data))
        return self._problem.model_space.add(
            self._translation, self._projector(coordinates)
        )

    def _derivative(self, data: Any) -> LinearOperator:
        return self._projector @ self._inner.derivative(self._shifted(data))

    def constraint_value_mapping(self, data: Any, /) -> Operator:
        """How the answer moves when the constraint value does, at fixed data.

        For a constraint written as an equation ``B u == w``, this maps ``w`` to
        the constrained minimum-norm model, with the data held fixed. The
        question it answers is the one a constrained inversion invites: *how
        much does insisting on this cost, and how much does the model change if
        I insist on something else?*

        With ``B+`` the constraint's pseudo-inverse and ``D`` the unconstrained
        derivative, the chain rule gives ``(I - D A) B+`` — the direct effect of
        moving the constraint, less the part of it the data pull back.

        Args:
            data: the observations, held fixed.

        Returns:
            The operator from constraint values to models.

        Raises:
            ValueError: if the subspace carries no explicit equation. One
                defined geometrically has no ``w`` to vary.
        """
        subspace = self._subspace
        if not subspace.has_explicit_equation:
            raise ValueError(
                "This subspace was defined geometrically, so there is no "
                "constraint value to vary. Build it from an equation "
                "B u == w to use this."
            )
        pseudo_inverse = subspace.pseudo_inverse()
        forward = self._problem.forward_operator
        model_space = self._problem.model_space
        data_space = self._problem.data_space
        identity = LinearOperator.identity(model_space)
        # The *reduced* method, not the unconstrained one: its answers lie in
        # the subspace's tangent space, so adding them to a point of the
        # subspace stays on it. The unconstrained method would walk straight
        # off the constraint it was asked to respect.
        reduced = self._inner

        def shifted_for(constraint_value: Any) -> tuple[Any, Any]:
            base = pseudo_inverse(constraint_value)
            return base, data_space.subtract(data, forward(base))

        def value(constraint_value: Any) -> Any:
            base, shifted = shifted_for(constraint_value)
            return model_space.add(base, self._projector(reduced(shifted)))

        def derivative(constraint_value: Any) -> LinearOperator:
            _, shifted = shifted_for(constraint_value)
            return (
                identity - self._projector @ reduced.derivative(shifted) @ forward
            ) @ pseudo_inverse

        return Operator.from_callables(
            subspace.constraint_operator.codomain,
            model_space,
            value,
            derivative=derivative,
        )
