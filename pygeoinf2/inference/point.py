"""
Point estimators: a data vector in, one model out.

Three methods, and a fourth that is one of them with its damping chosen by the
data rather than by the caller.

Each is built on :class:`~pygeoinf2.inference.tikhonov.TikhonovNormalOperator`,
so each exposes the operator it inverts and can be preconditioned, given a
surrogate, reduced, or swept over its damping — see DESIGN.md section 24.

The distinction that runs through the module is between a damping *chosen* and
a damping *found*. With the damping fixed, the estimate is affine in the data
and the estimator is an operator, which is what
:class:`LinearPointEstimator` is for. With the damping found from the data by
the discrepancy principle, it is not: the map is genuinely non-linear, and
:class:`DiscrepancyPrinciple` supplies its derivative rather than pretending
otherwise.
"""

from __future__ import annotations

from typing import Any, Callable


from ..algebra.operators import LinearOperator, Operator
from ..geometry.sets import Subset
from ..geometry.subspaces import AffineSubspace
from ..probability.base import ProbabilityMeasure
from ..probability.gaussian import GaussianMeasure
from ..numerics.root_find import Evaluation, RootResult, monotone_root
from ..numerics.solvers import LinearSolver, resolve_solver
from .estimators import LinearPointEstimator
from .normal import Formalism
from .normal import choose_formalism as _choose
from .problem import LinearForwardProblem
from .tikhonov import TikhonovFamily, TikhonovNormalOperator

__all__ = [
    "ConstrainedLeastSquares",
    "ConstrainedMinimumNorm",
    "DiscrepancyPrinciple",
    "LeastSquares",
    "MinimumNorm",
    "choose_formalism",
]


def choose_formalism(
    problem: LinearForwardProblem, /, *, formalism: Formalism = "auto"
) -> str:
    """Which space to assemble *this problem's* normal equations in.

    The spaces-based decision of
    :func:`~pygeoinf2.inference.normal.choose_formalism`, read off a problem.
    """
    return _choose(problem.model_space, problem.data_space, formalism=formalism)


def _error_measure(problem: LinearForwardProblem) -> Any:
    """The problem's error measure, or None when it has none."""
    return problem.error_measure if problem.has_error else None


class LeastSquares(LinearPointEstimator):
    """Tikhonov-regularised least squares: minimise ``|A u - d|^2_R + t |u|^2``.

    The estimator *is* the mapping from data to the fitted model, so it is an
    affine operator and joins the algebra. The damping is fixed here; to let
    the data choose it, see :class:`DiscrepancyPrinciple`.
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
        # Built before the solver, because it never needed one: a solver only
        # inverts it. So a caller may pass a factory and precondition with
        # something derived from the operator about to be inverted.
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
    def problem(self) -> LinearForwardProblem:
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

        The two give the same answer; the test suite checks that they do.
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
        :class:`~pygeoinf2.inference.tikhonov.TikhonovFamily` directly is
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

        Returns the surrogate normal operator, which is the part a
        preconditioner is built from. It may live on a different model space,
        as long as the data space is shared.
        """
        return self._normal.surrogate(
            forward=forward, error=error, damping=damping, formalism=formalism
        )

    def parameterised(self, parameterisation: LinearOperator, /) -> "LeastSquares":
        """The same estimator restricted to a parameter space.

        Args:
            parameterisation: maps the parameter space into the model space.

        Returns:
            An estimator of the same kind on the parameterised problem.
        """
        return type(self)(
            self._problem.parameterised(parameterisation),
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

        v1's ``normal_residual_callback``. Wired to *these* normal equations
        and *this* right-hand side, so the number printed is the one the solve
        is actually driving down rather than whatever the solver happens to
        track internally.
        """
        space = self._normal.domain
        target = self.right_hand_side(data)
        scale = space.norm(target)
        if scale == 0.0:
            scale = 1.0

        def callback(iteration: int, residual: float) -> None:
            report(message.format(iteration=iteration, residual=residual / scale))

        return callback


class MinimumNorm(LeastSquares):
    """The smallest model fitting the data, at a damping chosen in advance.

    Identical to :class:`LeastSquares` in construction — the two differ in
    intent rather than in algebra, since minimising ``|A u - d|^2_R + t |u|^2``
    is trading misfit against norm either way. It is a separate name because
    the question it answers is "how small can the model be", and because
    :meth:`for_data` is what makes that question well posed.
    """

    def for_data(
        self,
        data: Any,
        /,
        *,
        level: float = 0.95,
        iterations: int = 60,
        rtol: float = 1e-6,
    ) -> "MinimumNorm":
        """The same method with the damping set by the discrepancy principle.

        Damps as hard as the data allow: the largest damping whose misfit still
        reaches the chi-squared threshold at *level*. The misfit increases with
        the damping, which is what makes the bisection a proof.

        Two saturated cases matter, and getting either wrong is worse than
        failing:

        * the misfit never reaches the threshold, so every damping fits and the
          answer is the **largest** — the data support no structure and the
          model should say so;
        * the misfit already exceeds it at any damping, so nothing fits and the
          answer is the **smallest**.

        Both fall out of :func:`~pygeoinf2.numerics.root_find.monotone_root`
        reporting which end it ran out at.

        The third case is not saturation but failure: if the misfit never
        reaches the threshold *from below*, no model fits and there is no
        answer to give. That is refused rather than answered, on the same terms
        as :class:`DiscrepancyPrinciple` — the two share one search.

        Args:
            data: the data vector.
            level: the confidence level setting the misfit target.
            iterations: the search's budget.
            rtol: the search's bracket tolerance.

        Returns:
            A :class:`MinimumNorm` with its damping fixed at the value found.

        Raises:
            ValueError: when no model fits the data to the threshold.
        """
        found = _searched_damping(
            self.family(),
            self._problem,
            data,
            level=level,
            iterations=iterations,
            rtol=rtol,
        )
        return MinimumNorm(
            self._problem,
            damping=found.argument,
            solver=self._solver,
            formalism=self._normal.formalism,
        )

    def discrepancy_search(
        self,
        data: Any,
        /,
        *,
        level: float = 0.95,
        iterations: int = 60,
        rtol: float = 1e-6,
    ) -> RootResult:
        """The search itself, with its diagnostics.

        Returned rather than hidden so that the cost is visible: without the
        iteration count there is no way to tell a warm start that is working
        from one that silently is not.
        """
        return _discrepancy_search(
            self.family(),
            self._problem,
            data,
            level=level,
            iterations=iterations,
            rtol=rtol,
        )


def _discrepancy_search(
    family: TikhonovFamily,
    problem: LinearForwardProblem,
    data: Any,
    /,
    *,
    level: float = 0.95,
    iterations: int = 60,
    rtol: float = 1e-6,
) -> RootResult:
    """Find the damping at which the misfit reaches its threshold.

    Each probe is a solve of ``N(t) w == v``, warm-started from the previous
    probe's ``w`` — which is what makes a sixty-step bisection cost roughly one
    solve rather than sixty.
    """
    target = problem.critical_chi_squared(level=level)
    right_hand_side = family.right_hand_side(data)

    def evaluate(damping: float, previous: Any) -> Evaluation:
        result = family.solve(damping, right_hand_side, x0=previous)
        model = family.model_from(result.solution)
        return Evaluation(
            value=problem.chi_squared(model, data),
            solution=result.solution,
            iterations=result.iterations,
        )

    return monotone_root(
        evaluate,
        target,
        decreasing=False,
        iterations=iterations,
        rtol=rtol,
    )


def _searched_damping(
    family: TikhonovFamily,
    problem: LinearForwardProblem,
    data: Any,
    /,
    *,
    level: float = 0.95,
    iterations: int = 60,
    rtol: float = 1e-6,
) -> RootResult:
    """The discrepancy search, refusing data that no model fits.

    Shared by :class:`DiscrepancyPrinciple` and :meth:`MinimumNorm.for_data`
    so the two cannot disagree about the saturated cases. They did: where the
    principle raised, ``for_data`` returned a damping of 1e-200 whose model had
    a chi-squared of 5e7 against a target of 9.5 — the solution of a
    numerically singular system, reported as an answer.

    Raises:
        ValueError: when the misfit does not reach its target at any damping.
    """
    found = _discrepancy_search(
        family, problem, data, level=level, iterations=iterations, rtol=rtol
    )
    if found.exhausted == "low":
        # No damping small enough brings the misfit to its target, so the
        # principle has no solution: these data cannot be fitted to this level
        # by any model this problem admits. v1 raises here too. The least-damped
        # model is *not* a fallback — it solves a numerically singular system,
        # and its size says nothing about the data.
        target = problem.critical_chi_squared(level=level)
        raise ValueError(
            f"The data cannot be fitted to the chi-squared threshold at "
            f"level {level}: the misfit is still {found.value:.4g} against a "
            f"target of {target:.4g} at the smallest damping the normal "
            f"operator survives. Lower the level, widen the model, or use "
            f"LeastSquares with a chosen damping."
        )
    return found


class DiscrepancyPrinciple(Operator):
    """Data to the smallest model that fits them, damping and all.

    The damping is found from the data, so this map is **not** affine and not
    an operator in the linear sense: two data vectors that need different
    dampings are not related by any fixed matrix. What makes it usable anyway
    is that its derivative can be written down exactly.

    Differentiating ``H(t) u == A* R^-1 d`` and the constraint
    ``chi^2(u, d) == target`` together, and using ``A* R^-1 r == -t u`` from the
    normal equations themselves:

    .. code-block:: text

        du/dd == L - h (x) w
        h == H(t)^-1 u
        w == (L* u + R^-1 r / t) / (u, h)

    with ``L`` the fixed-damping estimator and ``r`` the residual. The first
    term is what the answer would do if the damping were frozen; the second is
    the correction for the damping moving, and it is a rank-one update, which
    is why the derivative costs one extra solve rather than a new problem.

    Both directions are supplied, and the adjoint is the one an inversion built
    on top of this would actually call.
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        /,
        *,
        level: float = 0.95,
        solver: LinearSolver | None = None,
        formalism: Formalism = "data_space",
        iterations: int = 60,
        rtol: float = 1e-6,
    ) -> None:
        """
        Args:
            problem: the forward problem. It must carry an error measure —
                without one there is no misfit threshold and no principle.
            level: the confidence level setting the misfit target.
            solver: how to invert each ``N(t)``.
            formalism: which space to solve in.
            iterations, rtol: the discrepancy search's budget and tolerance.
        """
        if not problem.has_error:
            raise ValueError(
                "The discrepancy principle needs a data error measure: the "
                "threshold it damps towards is a statement about the noise. "
                "Without one, use LeastSquares with a chosen damping."
            )
        super().__init__(problem.data_space, problem.model_space)
        self._problem = problem
        self._level = level
        self._solver = solver
        self._formalism = formalism
        self._iterations = iterations
        self._rtol = rtol
        self._family = TikhonovFamily(
            problem.forward_operator,
            error=problem.error_measure,
            solver=self._solver,
            formalism=formalism,
        )

    @property
    def problem(self) -> LinearForwardProblem:
        """The forward problem."""
        return self._problem

    @property
    def level(self) -> float:
        """The confidence level the misfit is damped towards."""
        return self._level

    def search(self, data: Any, /) -> RootResult:
        """The damping search for this data, with its diagnostics."""
        return _discrepancy_search(
            self._family,
            self._problem,
            data,
            level=self._level,
            iterations=self._iterations,
            rtol=self._rtol,
        )

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
            iterations=self._iterations,
            rtol=self._rtol,
        )
        return (
            self._family.model_from(found.solution),
            found.argument,
            found.converged,
        )

    def _value(self, data: Any) -> Any:
        model, _, _ = self._resolve(data)
        return model

    def _linearise(self, data: Any) -> Any:
        """The model and its derivative from a *single* damping search.

        ``at(data)`` otherwise called :meth:`_value` and :meth:`_derivative`
        in turn, and each of those runs :meth:`_resolve` -- so the root find
        over the damping, which is the whole cost of this operator, ran twice
        for one linearisation. Nothing else about the two paths differs.
        """
        from ..algebra.operators import Linearisation

        resolved = self._resolve(data)
        return Linearisation(
            data, resolved[0], self._derivative_from(data, *resolved)
        )

    def estimator_at(self, damping: float, /) -> LeastSquares:
        """The fixed-damping estimator this collapses to at one damping."""
        return LeastSquares(
            self._problem,
            damping=damping,
            solver=self._solver,
            formalism=self._formalism,
        )

    def _derivative(self, data: Any) -> LinearOperator:
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

    def parameterised(
        self, parameterisation: LinearOperator, /, **kwargs: Any
    ) -> "ConstrainedLeastSquares":
        """The same estimator restricted to a parameter space.

        The constraint is pulled back with the problem: ``B u == w`` in the
        model space becomes ``(B M) p == w`` in the parameter space, where
        ``M`` is the parameterisation. That is v1's construction, and it works
        because a constraint written as an equation says what it means about
        any model, including one built from parameters.

        This used to refuse outright, on the grounds that the parameterisation
        alone does not determine the constraint. It does when the subspace
        remembers the equation it was built from -- which is what
        :attr:`~pygeoinf2.geometry.subspaces.AffineSubspace.has_explicit_equation`
        reports -- and only then, which is what the refusal below is for.

        Args:
            parameterisation: ``M``, from the parameter space into the model
                space.
            **kwargs: passed to the reduced problem's own parameterisation.

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
        subspace = _parameterised_subspace(self._subspace, parameterisation)
        return type(self)(
            self._problem.parameterised(parameterisation, **kwargs),
            subspace,
            damping=self._damping,
            solver=self._solver,
            formalism=self._formalism,
        )

    def _parameterised_subspace(
        self, parameterisation: LinearOperator, /
    ) -> AffineSubspace:
        """The constraint, read in the parameter space."""
        return _parameterised_subspace(self._subspace, parameterisation)


    def data_reduced(self, *args: Any, **kwargs: Any) -> "ConstrainedLeastSquares":
        """The same estimator on a reduced set of data.

        The constraint lives in the *model* space and a data reduction does not
        touch it, so unlike :meth:`parameterised` there is nothing to pull
        back: the reduction applies to the problem alone.
        """
        return type(self)(
            self._problem.data_reduced(*args, **kwargs),
            self._subspace,
            damping=self._damping,
            solver=self._solver,
            formalism=self._formalism,
        )


def _parameterised_subspace(
    subspace: AffineSubspace, parameterisation: LinearOperator, /
) -> AffineSubspace:
    """A constraint pulled back through a parameterisation.

    ``B u == w`` in the model space becomes ``(B M) p == w`` in the parameter
    space. That is v1's construction, and it works because a constraint written
    as an equation says what it means about any model, including one assembled
    from parameters.

    Args:
        subspace: the constraint, which must remember its equation.
        parameterisation: ``M``, from the parameter space into the model space.

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
            "Parameterising a constrained inversion needs the constraint as "
            "an equation, and this subspace was built from a basis. Rebuild "
            "it with AffineSubspace.from_linear_equation, or use "
            "to_hyperplanes() for an equation with the same solution set."
        )
    constraint = subspace.constraint_operator
    if constraint.codomain.dim > parameterisation.domain.dim:
        raise ValueError(
            f"The parameter space has dimension {parameterisation.domain.dim}, "
            f"which cannot carry {constraint.codomain.dim} constraints."
        )
    return AffineSubspace.from_linear_equation(
        constraint @ parameterisation, subspace.constraint_value
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


class ConstrainedMinimumNorm(Operator):
    """The smallest model inside an affine subspace that fits the data.

    The constrained counterpart of :class:`DiscrepancyPrinciple`, and built the
    same way :class:`ConstrainedLeastSquares` is: substitute ``u == c + P w``,
    solve the unconstrained problem for ``w`` against the shifted data
    ``d - A c``, and add ``c`` back.

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
        iterations: int = 60,
        rtol: float = 1e-6,
    ) -> None:
        """
        Args:
            problem: the forward problem, which must carry an error measure.
            subspace: an ``AffineSubspace`` of the model space.
            level: the confidence level setting the misfit target.
            solver: how to invert each reduced ``N(t)``.
            formalism: which space to solve in.
            iterations, rtol: the discrepancy search's budget and tolerance.
        """
        if subspace.domain != problem.model_space:
            raise ValueError("The subspace must live in the model space.")
        super().__init__(problem.data_space, problem.model_space)
        self._problem = problem
        self._subspace = subspace
        self._projector = subspace.projector
        self._translation = subspace.translation
        self._inner = DiscrepancyPrinciple(
            _reduced_problem(problem, subspace),
            level=level,
            solver=solver,
            formalism=formalism,
            iterations=iterations,
            rtol=rtol,
        )
        self._offset = problem.forward_operator(self._translation)
        # Kept so parameterised() and data_reduced() can rebuild with the
        # same settings; the inner method does not expose all of them.
        self._level = level
        self._solver = solver
        self._formalism = formalism

    @property
    def subspace(self) -> AffineSubspace:
        """The affine subspace the answer is confined to."""
        return self._subspace

    @property
    def reduced(self) -> DiscrepancyPrinciple:
        """The unconstrained method on the reduced problem."""
        return self._inner

    def parameterised(
        self, parameterisation: LinearOperator, /, **kwargs: Any
    ) -> "ConstrainedMinimumNorm":
        """The same method restricted to a parameter space.

        As :meth:`ConstrainedLeastSquares.parameterised`: the constraint
        ``B u == w`` is pulled back to ``(B M) p == w``, which needs the
        subspace to remember the equation it was built from.
        """
        return type(self)(
            self._problem.parameterised(parameterisation, **kwargs),
            _parameterised_subspace(self._subspace, parameterisation),
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

        Needs the subspace to carry an explicit equation; one defined
        geometrically has no ``w`` to vary.
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
