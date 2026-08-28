"""Quadratic programmes, and the backends that solve them.

A bundle method's subproblem is a small dense QP: minimise a quadratic over a
simplex, or over a level set, in as many variables as there are cuts. It is
tiny compared with the space the outer problem lives in, and it is solved many
times, so what matters is that it is solved *accurately* and without setup
overhead.

``pygeoinf2.numerics.convex`` solves the simplex case by an accelerated
projected gradient, which needs nothing installed and is enough for the
proximal method. The level bundle method's subproblem has a general linear
constraint and is not a simplex projection, so it wants a real QP solver --
and the proximal method's own accuracy is limited by its projected gradient on
the ill-conditioned Gram matrices a converging bundle produces, which is the
other reason these are here.

Three backends, in the order :func:`best_available_qp_solver` prefers them:

* **OSQP**, an ADMM splitting method. Fast on larger problems and warm-startable,
  which matters when the same QP is re-solved with one row added.
* **Clarabel**, an interior-point method. Slower to start and more accurate.
* **SciPy's SLSQP**, always available, and the reason nothing here is an
  optional dependency in practice.

All three take the OSQP standard form, which is the most permissive of the
three: ``l <= A x <= u`` covers equalities (``l == u``), one-sided inequalities
(an infinite bound) and box constraints without a separate case for each.

This module is a port of v1's ``convex_optimisation.py``, whose author's view
on the API is to be sought before it is changed beyond the port.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "QPResult",
    "QPSolver",
    "SciPyQPSolver",
    "OSQPQPSolver",
    "ClarabelQPSolver",
    "best_available_qp_solver",
]

# OSQP spells infinity this way, and rejects np.inf.
_OSQP_INFINITY = 1e30


@dataclass(frozen=True)
class QPResult:
    """The outcome of one quadratic programme solve."""

    x: np.ndarray
    """The minimiser, in components."""

    objective: float
    """``x' P x / 2 + q' x`` there."""

    status: str
    """``"solved"``, or the backend's own description of what went wrong.

    A string rather than a boolean because the three backends fail in
    different ways -- infeasible, unbounded, iteration limit -- and which one
    it was is the useful part. Compare against ``"solved"`` for the plain
    question.
    """

    @property
    def solved(self) -> bool:
        """Whether the backend reported success."""
        return self.status == "solved"

    def __repr__(self) -> str:
        return f"QPResult(objective={self.objective:.6g}, status={self.status!r})"


@runtime_checkable
class QPSolver(Protocol):
    """What a QP backend must provide.

    The OSQP standard form:

    .. code-block:: text

        minimise    x' P x / 2 + q' x
        subject to  l <= A x <= u

    with ``-inf`` and ``+inf`` allowed in the bounds, and ``l_i == u_i``
    meaning an equality. Everything here works in *components*: a bundle
    subproblem is finite-dimensional and Euclidean whatever the space the
    outer problem lives in, which is the whole reason it can be handed to an
    off-the-shelf solver.
    """

    def solve(
        self,
        P: np.ndarray,
        q: np.ndarray,
        A: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        /,
        *,
        x0: np.ndarray | None = None,
    ) -> QPResult:
        """Solve the programme.

        Args:
            P: symmetric positive semidefinite, ``(n, n)``.
            q: the linear cost, ``(n,)``.
            A: the constraint matrix, ``(m, n)``.
            lower: lower bounds, ``(m,)``.
            upper: upper bounds, ``(m,)``.
            x0: a warm start, when the backend can use one.

        Returns:
            The result.
        """
        ...


def _validate(
    P: np.ndarray,
    q: np.ndarray,
    A: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shapes checked once, here, rather than three times inconsistently."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    q = np.atleast_1d(np.asarray(q, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    lower = np.atleast_1d(np.asarray(lower, dtype=float))
    upper = np.atleast_1d(np.asarray(upper, dtype=float))

    size = q.size
    if P.shape != (size, size):
        raise ValueError(f"P must be ({size}, {size}), got {P.shape}.")
    if A.shape[1] != size:
        raise ValueError(f"A must have {size} columns, got {A.shape[1]}.")
    if lower.shape != (A.shape[0],) or upper.shape != (A.shape[0],):
        raise ValueError(
            f"The bounds must have {A.shape[0]} entries each, got "
            f"{lower.shape} and {upper.shape}."
        )
    if np.any(lower > upper):
        raise ValueError("Every lower bound must be at or below its upper bound.")
    return P, q, A, lower, upper


class SciPyQPSolver:
    """Sequential least squares, through ``scipy.optimize.minimize``.

    The fallback that is always installed. SLSQP is a general nonlinear method
    given a quadratic and its exact gradient, so it is slower and less accurate
    than a QP solver proper -- fine for a few dozen variables, which is what a
    bundle subproblem has, and not the thing to reach for at a few hundred.

    Warm starts are honoured: SLSQP takes a starting point like any other
    descent method.
    """

    def __init__(self, /, *, tolerance: float = 1e-9, iterations: int = 1000) -> None:
        """
        Args:
            tolerance: SLSQP's ``ftol``.
            iterations: its iteration cap.
        """
        self._tolerance = tolerance
        self._iterations = iterations

    def solve(
        self,
        P: np.ndarray,
        q: np.ndarray,
        A: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        /,
        *,
        x0: np.ndarray | None = None,
    ) -> QPResult:
        """Solve the programme with SLSQP."""
        from scipy.optimize import minimize

        P, q, A, lower, upper = _validate(P, q, A, lower, upper)
        start = np.zeros(q.size) if x0 is None else np.asarray(x0, dtype=float)

        constraints = []
        for row, low, high in zip(A, lower, upper):
            finite_low, finite_high = np.isfinite(low), np.isfinite(high)
            if finite_low and finite_high and low == high:
                constraints.append(
                    {"type": "eq", "fun": lambda x, r=row, v=low: float(r @ x) - v}
                )
                continue
            if finite_high:
                constraints.append(
                    {"type": "ineq", "fun": lambda x, r=row, v=high: v - float(r @ x)}
                )
            if finite_low:
                constraints.append(
                    {"type": "ineq", "fun": lambda x, r=row, v=low: float(r @ x) - v}
                )

        outcome = minimize(
            fun=lambda x: 0.5 * float(x @ P @ x) + float(q @ x),
            x0=start,
            jac=lambda x: P @ x + q,
            method="SLSQP",
            constraints=constraints,
            options={"ftol": self._tolerance, "maxiter": self._iterations},
        )
        return QPResult(
            x=np.asarray(outcome.x, dtype=float),
            objective=float(outcome.fun),
            status="solved" if outcome.success else str(outcome.message),
        )


class OSQPQPSolver:
    """OSQP: operator splitting, and the fastest of the three at scale.

    Takes the standard form directly, so nothing is translated. A fresh solver
    object is built per call rather than kept: OSQP's setup carries the problem
    data, and re-using an instance across problems of different shape is how
    stale state gets in. The warm start is passed separately and is the part
    worth keeping between solves of the *same* problem.

    Raises:
        ImportError: if ``osqp`` is not installed. Use
            :func:`best_available_qp_solver` to fall back rather than fail.
    """

    def __init__(
        self,
        /,
        *,
        absolute_tolerance: float = 1e-6,
        relative_tolerance: float = 1e-6,
        iterations: int = 10_000,
        polish: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            absolute_tolerance: OSQP's ``eps_abs``.
            relative_tolerance: its ``eps_rel``.
            iterations: the ADMM iteration cap.
            polish: run OSQP's polishing step, which recovers a high-accuracy
                solution from the ADMM one and is usually worth its cost on
                problems this small.
            verbose: let OSQP print.
        """
        try:
            import osqp  # noqa: F401
        except ImportError as error:  # pragma: no cover - depends on the install
            raise ImportError(
                "OSQPQPSolver needs the 'osqp' package. Install it, or use "
                "best_available_qp_solver() to fall back to what is here."
            ) from error
        self._absolute_tolerance = absolute_tolerance
        self._relative_tolerance = relative_tolerance
        self._iterations = iterations
        self._polish = polish
        self._verbose = verbose

    def solve(
        self,
        P: np.ndarray,
        q: np.ndarray,
        A: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        /,
        *,
        x0: np.ndarray | None = None,
    ) -> QPResult:
        """Solve the programme with OSQP."""
        import osqp
        import scipy.sparse as sparse

        P, q, A, lower, upper = _validate(P, q, A, lower, upper)
        # OSQP rejects np.inf and spells an absent bound this way.
        low = np.where(np.isneginf(lower), -_OSQP_INFINITY, lower)
        high = np.where(np.isposinf(upper), _OSQP_INFINITY, upper)

        problem = osqp.OSQP()
        problem.setup(
            sparse.csc_matrix(P),
            q,
            sparse.csc_matrix(A),
            low,
            high,
            verbose=self._verbose,
            eps_abs=self._absolute_tolerance,
            eps_rel=self._relative_tolerance,
            polishing=self._polish,
            max_iter=self._iterations,
            warm_starting=True,
        )
        if x0 is not None:
            problem.warm_start(x=np.asarray(x0, dtype=float))

        outcome = problem.solve()
        reported = str(outcome.info.status)
        return QPResult(
            x=np.asarray(outcome.x, dtype=float),
            objective=float(outcome.info.obj_val),
            status="solved" if "solved" in reported.lower() else reported,
        )


class ClarabelQPSolver:
    """Clarabel: an interior-point method, and the most accurate of the three.

    The standard form has to be translated into cones, because Clarabel takes
    ``b - A x`` in a cone rather than two-sided bounds. An equality becomes a
    zero cone; each finite one-sided bound becomes a row of a non-negative
    cone, so a two-sided inequality contributes *two* rows. There is no warm
    start: an interior-point method starts from its own central point, and
    handing it someone else's is not helpful.

    Raises:
        ImportError: if ``clarabel`` is not installed.
    """

    def __init__(
        self,
        /,
        *,
        absolute_tolerance: float = 1e-8,
        relative_tolerance: float = 1e-8,
        iterations: int = 200,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            absolute_tolerance: Clarabel's ``tol_gap_abs`` and ``tol_feas``.
            relative_tolerance: its ``tol_gap_rel``.
            iterations: the interior-point iteration cap.
            verbose: let Clarabel print.
        """
        try:
            import clarabel  # noqa: F401
        except ImportError as error:  # pragma: no cover - depends on the install
            raise ImportError(
                "ClarabelQPSolver needs the 'clarabel' package. Install it, "
                "or use best_available_qp_solver() to fall back."
            ) from error
        self._absolute_tolerance = absolute_tolerance
        self._relative_tolerance = relative_tolerance
        self._iterations = iterations
        self._verbose = verbose

    def solve(
        self,
        P: np.ndarray,
        q: np.ndarray,
        A: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        /,
        *,
        x0: np.ndarray | None = None,
    ) -> QPResult:
        """Solve the programme with Clarabel.

        ``x0`` is accepted and ignored, so that the three backends remain
        interchangeable. An interior-point method has no use for it.
        """
        import clarabel
        import scipy.sparse as sparse

        P, q, A, lower, upper = _validate(P, q, A, lower, upper)

        equality_rows, equality_values = [], []
        inequality_rows, inequality_values = [], []
        for row, low, high in zip(A, lower, upper):
            finite_low, finite_high = np.isfinite(low), np.isfinite(high)
            if finite_low and finite_high and low == high:
                equality_rows.append(row)
                equality_values.append(float(low))
                continue
            if finite_high:
                # A x <= u  is  u - A x >= 0.
                inequality_rows.append(row)
                inequality_values.append(float(high))
            if finite_low:
                # A x >= l  is  (-l) - (-A) x >= 0.
                inequality_rows.append(-row)
                inequality_values.append(float(-low))

        cones = []
        if equality_rows:
            cones.append(clarabel.ZeroConeT(len(equality_rows)))
        if inequality_rows:
            cones.append(clarabel.NonnegativeConeT(len(inequality_rows)))
        if not cones:
            raise ValueError("Clarabel needs at least one constraint.")

        constraint = sparse.csc_matrix(
            np.asarray(equality_rows + inequality_rows, dtype=float).reshape(
                len(equality_rows) + len(inequality_rows), q.size
            )
        )
        offsets = np.asarray(equality_values + inequality_values, dtype=float)

        settings = clarabel.DefaultSettings()
        settings.verbose = self._verbose
        settings.max_iter = self._iterations
        settings.tol_gap_abs = self._absolute_tolerance
        settings.tol_gap_rel = self._relative_tolerance
        settings.tol_feas = self._absolute_tolerance

        outcome = clarabel.DefaultSolver(
            sparse.csc_matrix(P), q, constraint, offsets, cones, settings
        ).solve()

        x = np.asarray(outcome.x, dtype=float)
        reported = str(outcome.status)
        return QPResult(
            x=x,
            # Clarabel reports its own objective on the transformed problem, so
            # this is evaluated directly to be sure it is the one asked for.
            objective=float(0.5 * x @ P @ x + q @ x),
            status="solved" if "solved" in reported.lower() else reported,
        )


def best_available_qp_solver() -> QPSolver:
    """Whichever QP backend is installed, in order of preference.

    OSQP first, for speed and its warm start; Clarabel next, for accuracy;
    SciPy's SLSQP last, because it is always there. The point of the ordering
    is that a caller need not know which of the optional packages the user has,
    and gets the best of them without asking.

    Returns:
        An instance of the best available backend.
    """
    for backend in (OSQPQPSolver, ClarabelQPSolver):
        try:
            return backend()
        except ImportError:  # pragma: no cover - depends on the install
            continue
    return SciPyQPSolver()
