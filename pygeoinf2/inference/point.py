"""
Point estimators: no prior, so the answer is a single model.

Both methods here assemble the same mapping two ways. The *model-space*
formalism solves ``A* R^-1 A + damping I`` on the model space; the *data-space*
one solves ``A A* + damping R`` on the data space. Which is cheaper depends
only on which space is smaller, so the choice is computational and is made
automatically unless it cannot be — see DESIGN.md section 18.10.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ..algebra.operators import LinearOperator
from ..numerics.solvers import CholeskySolver, LinearSolver
from ..traits import Traits
from .estimators import LinearPointEstimator
from .problem import LinearForwardProblem

__all__ = [
    "ConstrainedLeastSquares",
    "LeastSquares",
    "MinimumNorm",
    "choose_formalism",
]

Formalism = Literal["auto", "model_space", "data_space"]


def choose_formalism(
    problem: LinearForwardProblem, /, *, formalism: Formalism = "auto"
) -> str:
    """Which space to assemble the normal equations in.

    ``"auto"`` takes the smaller of the two, which is the whole content of the
    choice: both assemble the same mapping. It falls back to the data space
    when a dimension is unavailable, since a coordinate-free model space is
    exactly the case where a model-space solve cannot be assembled anyway.
    """
    if formalism not in ("auto", "model_space", "data_space"):
        raise ValueError(
            f"The formalism is 'auto', 'model_space' or 'data_space', got "
            f"{formalism!r}."
        )
    if formalism != "auto":
        return formalism
    try:
        return (
            "model_space"
            if problem.model_space.dim <= problem.data_space.dim
            else "data_space"
        )
    except (AttributeError, NotImplementedError):  # pragma: no cover
        return "data_space"


def _precision(problem: LinearForwardProblem) -> LinearOperator | None:
    """``R^-1`` when there is a data error, otherwise ``None`` for the identity."""
    return problem.error_measure.precision if problem.has_error else None


class LeastSquares(LinearPointEstimator):
    """Tikhonov-regularised least squares: minimise ``|A u - d|^2_R + t |u|^2``.

    The estimator *is* the mapping from data to the fitted model, so it is an
    affine operator and joins the algebra.
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        /,
        *,
        damping: float = 0.0,
        solver: LinearSolver | None = None,
        formalism: Formalism = "auto",
    ) -> None:
        """
        Args:
            problem: the forward problem.
            damping: the Tikhonov parameter. Must be positive unless the
                normal operator is invertible without it.
            solver: how to invert the normal operator. A Cholesky factorisation
                by default, since the operator is positive definite.
            formalism: which space to solve in; see :func:`choose_formalism`.
        """
        if damping < 0.0:
            raise ValueError(f"The damping must be non-negative, got {damping}.")
        forward = problem.forward_operator
        solver = CholeskySolver() if solver is None else solver
        precision = _precision(problem)
        chosen = choose_formalism(problem, formalism=formalism)
        self._formalism = chosen

        if chosen == "model_space":
            weighted = (
                forward.adjoint if precision is None else forward.adjoint @ precision
            )
            normal = weighted @ forward
            if damping > 0.0:
                normal = normal + damping * LinearOperator.identity(problem.model_space)
            operator = solver(normal.with_traits(Traits.POSITIVE_DEFINITE)) @ weighted
        else:
            gram = forward @ forward.adjoint
            if damping > 0.0:
                covariance = (
                    problem.error_measure.covariance
                    if problem.has_error
                    else LinearOperator.identity(problem.data_space)
                )
                gram = gram + damping * covariance
            operator = forward.adjoint @ solver(
                gram.with_traits(Traits.POSITIVE_DEFINITE)
            )

        super().__init__(
            operator,
            forward_operator=forward,
            error=problem.error_measure if problem.has_error else None,
        )
        self._problem = problem
        self._damping = damping

    @property
    def formalism(self) -> str:
        """Which space the normal equations were assembled in."""
        return self._formalism

    @property
    def damping(self) -> float:
        """The Tikhonov parameter used."""
        return self._damping


class MinimumNorm(LinearPointEstimator):
    """The smallest model fitting the data to within a chosen confidence.

    The discrepancy principle: damp as hard as the data will allow. The
    damping is found by a root search, and the residual is monotone in it — one
    of the four users of the primitive in DESIGN.md section 18.6.

    The estimator is affine in the data only once the damping is fixed, so the
    damping is chosen at construction from a *target* misfit rather than from
    the data. Use :meth:`for_data` to choose it from a particular data vector.
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        /,
        *,
        damping: float = 0.0,
        solver: LinearSolver | None = None,
        formalism: Formalism = "auto",
    ) -> None:
        """
        Args:
            problem: the forward problem.
            damping: the damping to use. Zero gives the exact minimum-norm
                solution of a consistent system.
            solver: how to invert the normal operator.
            formalism: which space to solve in.
        """
        fitted = LeastSquares(
            problem, damping=damping, solver=solver, formalism=formalism
        )
        super().__init__(
            fitted.operator,
            forward_operator=problem.forward_operator,
            error=problem.error_measure if problem.has_error else None,
        )
        self._problem = problem
        self._damping = damping
        self._solver = solver
        self._formalism = formalism

    @property
    def damping(self) -> float:
        """The damping in force."""
        return self._damping

    def for_data(
        self,
        data: Any,
        /,
        *,
        level: float = 0.95,
        bracket: tuple[float, float] = (1e-12, 1e12),
        iterations: int = 60,
    ) -> "MinimumNorm":
        """The same method with the damping set by the discrepancy principle.

        Damps until the misfit reaches the chi-squared threshold at ``level``,
        by bisection — which is valid because the misfit increases with the
        damping. If even the largest damping in ``bracket`` leaves the misfit
        below the threshold, the data are fitted to within their errors by an
        arbitrarily small model, and the smallest damping is returned.

        Args:
            data: the data to fit.
            level: the confidence level for the misfit target.
            bracket: the range of dampings to search.
            iterations: bisection steps.
        """
        target = self._problem.critical_chi_squared(level=level)

        def misfit(damping: float) -> float:
            estimator = MinimumNorm(
                self._problem,
                damping=damping,
                solver=self._solver,
                formalism=self._formalism,
            )
            return self._problem.chi_squared(estimator(data), data)

        low, high = bracket
        if misfit(high) < target:
            chosen = low
        elif misfit(low) > target:
            chosen = low
        else:
            for _ in range(iterations):
                middle = np.sqrt(low * high)
                if misfit(middle) < target:
                    low = middle
                else:
                    high = middle
            chosen = np.sqrt(low * high)
        return MinimumNorm(
            self._problem,
            damping=float(chosen),
            solver=self._solver,
            formalism=self._formalism,
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
        subspace: Any,
        /,
        *,
        damping: float = 0.0,
        solver: LinearSolver | None = None,
        formalism: Formalism = "auto",
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

        reduced = LinearForwardProblem(
            problem.forward_operator @ projector,
            error=problem.error if problem.has_error else None,
        )
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
            error=problem.error_measure if problem.has_error else None,
        )
        self._problem = problem
        self._subspace = subspace
        self._damping = damping

    @property
    def subspace(self) -> Any:
        """The affine subspace the answer is confined to."""
        return self._subspace

    @property
    def damping(self) -> float:
        """The Tikhonov parameter used within the subspace."""
        return self._damping
