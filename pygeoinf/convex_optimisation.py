"""
Convex optimisation utilities for non-smooth problems.

This module provides a minimal subgradient descent implementation suitable
for learning and experimentation. It assumes the objective is a NonLinearForm
that can provide a subgradient oracle via form.subgradient(x).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional, List, Tuple, TYPE_CHECKING, runtime_checkable, Protocol

import numpy as np
from scipy.optimize import minimize

from .nonlinear_forms import NonLinearForm
from .convex_analysis import BallSupportFunction, EllipsoidSupportFunction, PointSupportFunction

if TYPE_CHECKING:
    from .hilbert_space import HilbertSpace, Vector
    from .convex_analysis import SupportFunction
    from .linear_operators import LinearOperator


@dataclass
class SubgradientResult:
    """Result from subgradient descent optimisation.

    Attributes:
        x_best: Best point found (lowest function value).
        f_best: Best function value found.
        x_final: Final iterate (may differ from x_best).
        f_final: Final function value.
        num_iterations: Number of iterations performed.
        converged: Whether convergence criterion was met.
        function_values: History of function values at each iteration.
        iterates: Optional history of all iterates (memory intensive).
    """

    x_best: "Vector"
    f_best: float
    x_final: "Vector"
    f_final: float
    num_iterations: int
    converged: bool
    function_values: List[float]
    iterates: Optional[List["Vector"]] = None


class SubgradientDescent:
    """
    Basic subgradient descent for minimising non-smooth convex functions.

    Algorithm:
        x_{k+1} = x_k - α * g_k

    where g_k ∈ ∂f(x_k) is a subgradient (obtained via oracle.subgradient(x_k)).

    This implementation uses CONSTANT step size α for all k. Convergence is
    not guaranteed with constant step size; use for learning/testing only.

    Parameters:
        oracle: A NonLinearForm with subgradient() method returning subgradients.
        step_size: Constant step size α > 0.
        max_iterations: Maximum number of iterations.
        store_iterates: Whether to store full history (memory intensive).
        stagnation_window: Optional number of iterations without improvement
            to declare convergence.
    """

    def __init__(
        self,
        oracle: NonLinearForm,
        /,
        *,
        step_size: float,
        max_iterations: int = 500,
        store_iterates: bool = False,
        stagnation_window: Optional[int] = None,
    ) -> None:
        if not isinstance(oracle, NonLinearForm):
            raise ValueError("oracle must be a NonLinearForm")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if stagnation_window is not None and stagnation_window <= 0:
            raise ValueError("stagnation_window must be positive if provided")

        self._oracle = oracle
        self._step_size = float(step_size)
        self._max_iterations = int(max_iterations)
        self._store_iterates = bool(store_iterates)
        self._stagnation_window = stagnation_window

    @property
    def oracle(self) -> NonLinearForm:
        return self._oracle

    @property
    def domain(self) -> "HilbertSpace":
        return self._oracle.domain

    def solve(self, x0: "Vector") -> SubgradientResult:
        """Run subgradient descent from initial point x0."""
        if not self.domain.is_element(x0):
            raise ValueError("x0 must be an element of the oracle domain")
        if not self._oracle.has_subgradient:
            raise ValueError("oracle must provide a subgradient")

        x = x0
        f_best = float("inf")
        x_best = x0
        function_values: List[float] = []
        iterates: Optional[List["Vector"]] = [] if self._store_iterates else None

        no_improve = 0
        converged = False

        for _ in range(self._max_iterations):
            f_x = self._oracle(x)
            function_values.append(float(f_x))

            if f_x < f_best:
                f_best = float(f_x)
                x_best = x
                no_improve = 0
            else:
                no_improve += 1

            if self._stagnation_window is not None:
                if no_improve >= self._stagnation_window:
                    converged = True
                    break

            if iterates is not None:
                iterates.append(x)

            g = self._oracle.subgradient(x)
            step = self.domain.multiply(self._step_size, g)
            x = self.domain.subtract(x, step)

        f_final = float(self._oracle(x))

        return SubgradientResult(
            x_best=x_best,
            f_best=f_best,
            x_final=x,
            f_final=f_final,
            num_iterations=len(function_values),
            converged=converged,
            function_values=function_values,
            iterates=iterates,
        )


# =============================================================================
# Phase 1: Bundle method core data structures
# =============================================================================


@dataclass
class Cut:
    """A linearisation cut for a convex function at a point.

    A cut records the function value and a subgradient at an evaluation point,
    defining the affine lower bound: f_j + <g_j, lambda - x_j> <= f(lambda)
    for all lambda.

    Attributes:
        x: Evaluation point (a Hilbert-space vector).
        f: Function value f(x).
        g: Subgradient g in partial_f(x).
        iteration: Iteration index at which this cut was generated.
    """

    x: "Vector"
    f: float
    g: "Vector"
    iteration: int


class Bundle:
    """Collection of cutting-plane linearisations of a convex function.

    A bundle stores a list of :class:`Cut` objects and provides utilities
    for building the piecewise-linear epigraph model:
        hat_phi(lambda) = max_j [ f_j + <g_j, lambda - x_j> ]
    used by proximal and level bundle methods.

    Examples:
        >>> space = EuclideanSpace(3)
        >>> bundle = Bundle()
        >>> bundle.add_cut(Cut(x=np.zeros(3), f=1.0, g=np.ones(3), iteration=0))
        >>> len(bundle)
        1
    """

    def __init__(self) -> None:
        self._cuts: List[Cut] = []

    def add_cut(self, cut: Cut) -> None:
        """Append *cut* to the bundle.

        Args:
            cut: The :class:`Cut` to add.
        """
        self._cuts.append(cut)

    def __len__(self) -> int:
        """Return the number of cuts currently stored."""
        return len(self._cuts)

    def lower_bound(self) -> float:
        """Return a placeholder lower bound (-infinity).

        The true lower bound min_lambda hat_phi(lambda) is computed
        as the master QP/LP objective value by the outer bundle solver (Phase 2–3).
        This placeholder allows :class:`BundleResult` to report ``f_low`` before
        any master problem has been solved.

        Returns:
            ``-np.inf``
        """
        return -np.inf

    def upper_bound(self) -> float:
        """Return the best (lowest) function value seen so far.

        This is a valid upper bound on the optimum since
        f* <= f(x_j) for all recorded points x_j.

        Returns:
            min_j f_j.

        Raises:
            ValueError: If the bundle is empty.
        """
        if not self._cuts:
            raise ValueError("Bundle is empty; no upper bound available.")
        return min(cut.f for cut in self._cuts)

    def best_point(self) -> "Vector":
        """Return the evaluation point with the lowest recorded function value.

        Returns:
            The vector x_j achieving min_j f_j.

        Raises:
            ValueError: If the bundle is empty.
        """
        if not self._cuts:
            raise ValueError("Bundle is empty; no best point available.")
        return min(self._cuts, key=lambda c: c.f).x

    def linearization_matrix(
        self,
        stability_center: "Vector",
        domain: "HilbertSpace",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build the inequality constraint matrix for the epigraph QP.

        For each cut (x_j, f_j, g_j) the cutting-plane constraint is:
            f_j + <g_j, lambda - x_j> <= t
        which is equivalent to:
            g_j^T @ lambda - t <= g_j^T @ x_j - f_j

        Stacking all cuts gives the system A_ineq @ [lambda; t] <= b where row
        i is [g_i^T, -1] and b_i = g_i^T @ x_i - f_i.

        Args:
            stability_center: Current stability/proximal centre (not used to build
                the constraint data, but kept for API consistency with Phase 2).
            domain: The Hilbert space whose ``to_components`` converts vectors to
                flat numpy arrays.

        Returns:
            A tuple ``(A_ineq, b_ineq)`` where

            * ``A_ineq`` has shape ``(n_cuts, dim + 1)`` — columns are
              [lambda_1, ..., lambda_d, t].
            * ``b_ineq`` has shape ``(n_cuts,)``.

        Raises:
            ValueError: If the bundle is empty.
        """
        if not self._cuts:
            raise ValueError("Bundle is empty; cannot build linearization matrix.")
        d = domain.dim
        n = len(self._cuts)
        A = np.empty((n, d + 1))
        b = np.empty(n)
        for i, cut in enumerate(self._cuts):
            g_c = domain.to_components(cut.g)
            x_c = domain.to_components(cut.x)
            A[i, :d] = g_c
            A[i, d] = -1.0
            b[i] = float(np.dot(g_c, x_c)) - cut.f
        return A, b

    def compress(self, max_size: int) -> None:
        """Discard all but the *max_size* most recent cuts.

        Args:
            max_size: Maximum number of cuts to retain.  If the bundle already
                has fewer cuts than *max_size*, nothing is changed.
        """
        if len(self._cuts) > max_size:
            self._cuts = self._cuts[-max_size:]


# ---------------------------------------------------------------------------
# QP abstraction
# ---------------------------------------------------------------------------


@dataclass
class QPResult:
    """Result from a quadratic programme solve.

    Attributes:
        x: Solution vector (component array).
        obj: Objective value at ``x``.
        status: ``'solved'`` on success; a descriptive failure message otherwise.
    """

    x: np.ndarray
    obj: float
    status: str


@runtime_checkable
class QPSolver(Protocol):
    """Protocol for QP solvers used by bundle methods.

    Solvers must implement the OSQP standard form:
        min_x (1/2) x^T @ P @ x + q^T @ x
        subject to: l <= A @ x <= u

    Args:
        P: Symmetric positive-semi-definite Hessian, shape ``(n, n)``.
        q: Linear cost, shape ``(n,)``.
        A: Constraint matrix, shape ``(m, n)``.
        l: Lower bounds, shape ``(m,)``; use ``-np.inf`` for one-sided.
        u: Upper bounds, shape ``(m,)``; use ``+np.inf`` for one-sided.
        x0: Optional warm-start primal solution.

    Returns:
        A :class:`QPResult`.
    """

    def solve(
        self,
        P: np.ndarray,
        q: np.ndarray,
        A: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> QPResult: ...


class SciPyQPSolver:
    """QP solver backed by :func:`scipy.optimize.minimize` with ``method='SLSQP'``.

    Implements the :class:`QPSolver` protocol.  Converts the OSQP standard-form
    bounds $l ≤ Ax ≤ u$ to SLSQP inequality/equality constraints.

    Notes:
        SLSQP is a gradient-based method suitable for small to medium problems
        (up to a few hundred variables).  For large-scale bundle QPs use an OSQP-
        or Clarabel-backed solver (Phase 4).
    """

    def solve(
        self,
        P: np.ndarray,
        q: np.ndarray,
        A: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> QPResult:
        """Solve the QP and return a :class:`QPResult`.

        Args:
            P: Symmetric PSD Hessian of shape ``(n, n)``.
            q: Linear cost vector of shape ``(n,)``.
            A: Constraint matrix of shape ``(m, n)``.
            l: Lower-bound vector of shape ``(m,)``.
            u: Upper-bound vector of shape ``(m,)``.
            x0: Optional warm-start point of shape ``(n,)``.

        Returns:
            :class:`QPResult` with ``status='solved'`` on success.
        """
        n = q.shape[0]
        if x0 is None:
            x0 = np.zeros(n)

        def objective(x: np.ndarray) -> float:
            return 0.5 * float(x @ P @ x) + float(q @ x)

        def gradient(x: np.ndarray) -> np.ndarray:
            return P @ x + q

        constraints = []
        for i in range(A.shape[0]):
            row = A[i]
            li = l[i]
            ui = u[i]
            finite_l = np.isfinite(li)
            finite_u = np.isfinite(ui)

            if finite_l and finite_u and li == ui:
                # Equality constraint: A[i] @ x == li
                def _eq(x, row=row, val=li):
                    return float(row @ x) - val

                constraints.append({"type": "eq", "fun": _eq})
            else:
                if finite_u:
                    # A[i] @ x <= u[i]  →  u[i] - A[i]@x >= 0
                    def _ub(x, row=row, val=ui):
                        return val - float(row @ x)

                    constraints.append({"type": "ineq", "fun": _ub})
                if finite_l:
                    # A[i] @ x >= l[i]  →  A[i]@x - l[i] >= 0
                    def _lb(x, row=row, val=li):
                        return float(row @ x) - val

                    constraints.append({"type": "ineq", "fun": _lb})

        result = minimize(
            fun=objective,
            x0=x0,
            jac=gradient,
            method="SLSQP",
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        status = "solved" if result.success else result.message
        return QPResult(x=result.x, obj=float(result.fun), status=status)


class OSQPQPSolver:
    """QP solver using OSQP (ADMM-based). Requires ``pip install osqp``.

    Supports warm-starting via the *x0* parameter.  A fresh OSQP instance is
    created for every :meth:`solve` call to avoid stale state.

    Args:
        eps_abs: Absolute feasibility tolerance (default ``1e-6``).
        eps_rel: Relative feasibility tolerance (default ``1e-6``).
        verbose: Whether OSQP prints solver output (default ``False``).
        polish: Whether to apply polishing step for higher accuracy (default ``True``).
        max_iter: Maximum number of ADMM iterations (default ``10000``).

    Raises:
        ImportError: If the ``osqp`` package is not installed.
    """

    def __init__(
        self,
        *,
        eps_abs: float = 1e-6,
        eps_rel: float = 1e-6,
        verbose: bool = False,
        polish: bool = True,
        max_iter: int = 10000,
    ) -> None:
        try:
            import osqp as _osqp  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OSQPQPSolver requires the 'osqp' package. "
                "Install it with: pip install osqp"
            ) from exc
        self._eps_abs = eps_abs
        self._eps_rel = eps_rel
        self._verbose = verbose
        self._polish = polish
        self._max_iter = max_iter

    def solve(
        self,
        P: np.ndarray,
        q: np.ndarray,
        A: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> QPResult:
        """Solve the QP using OSQP and return a :class:`QPResult`.

        Args:
            P: Symmetric PSD Hessian of shape ``(n, n)``.
            q: Linear cost vector of shape ``(n,)``.
            A: Constraint matrix of shape ``(m, n)``.
            l: Lower-bound vector of shape ``(m,)``; ``-np.inf`` for one-sided.
            u: Upper-bound vector of shape ``(m,)``; ``+np.inf`` for one-sided.
            x0: Optional warm-start primal solution of shape ``(n,)``.

        Returns:
            :class:`QPResult` with ``status='solved'`` on success.
        """
        import osqp
        import scipy.sparse as sp

        # OSQP uses 1e30 to represent infinity.
        _INF = 1e30
        l_osqp = np.where(np.isneginf(l), -_INF, l)
        u_osqp = np.where(np.isposinf(u), _INF, u)

        P_sparse = sp.csc_matrix(P)
        A_sparse = sp.csc_matrix(A)

        prob = osqp.OSQP()
        prob.setup(
            P_sparse,
            q,
            A_sparse,
            l_osqp,
            u_osqp,
            verbose=self._verbose,
            eps_abs=self._eps_abs,
            eps_rel=self._eps_rel,
            polishing=self._polish,
            max_iter=self._max_iter,
            warm_starting=True,
        )

        if x0 is not None:
            prob.warm_start(x=x0)

        results = prob.solve()
        raw_status: str = results.info.status
        status = "solved" if "solved" in raw_status.lower() else raw_status
        return QPResult(
            x=results.x,
            obj=float(results.info.obj_val),
            status=status,
        )


class ClarabelQPSolver:
    """QP solver using Clarabel (interior-point). Requires ``pip install clarabel``.

    Converts the OSQP standard form $l ≤ Ax ≤ u$ to Clarabel's cone
    form internally.  Equality constraints ($l_i = u_i$) are handled via
    :class:`clarabel.ZeroConeT`; inequality constraints via
    :class:`clarabel.NonnegativeConeT`.

    Args:
        verbose: Whether Clarabel prints solver output (default ``False``).
        max_iter: Maximum interior-point iterations (default ``200``).
        eps_abs: Absolute convergence tolerance (default ``1e-8``).
        eps_rel: Relative convergence tolerance (default ``1e-8``).

    Raises:
        ImportError: If the ``clarabel`` package is not installed.
    """

    def __init__(
        self,
        *,
        verbose: bool = False,
        max_iter: int = 200,
        eps_abs: float = 1e-8,
        eps_rel: float = 1e-8,
    ) -> None:
        try:
            import clarabel as _clarabel  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ClarabelQPSolver requires the 'clarabel' package. "
                "Install it with: pip install clarabel"
            ) from exc
        self._verbose = verbose
        self._max_iter = max_iter
        self._eps_abs = eps_abs
        self._eps_rel = eps_rel

    def solve(
        self,
        P: np.ndarray,
        q: np.ndarray,
        A: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> QPResult:
        """Solve the QP using Clarabel and return a :class:`QPResult`.

        Args:
            P: Symmetric PSD Hessian of shape ``(n, n)``.
            q: Linear cost vector of shape ``(n,)``.
            A: Constraint matrix of shape ``(m, n)``.
            l: Lower-bound vector of shape ``(m,)``; ``-np.inf`` for one-sided.
            u: Upper-bound vector of shape ``(m,)``; ``+np.inf`` for one-sided.
            x0: Warm-start hint (ignored if API unavailable).

        Returns:
            :class:`QPResult` with ``status='solved'`` on success.

        Notes:
            Clarabel's constraint form is $Ax + s = b$, $s ∈ K$.  For
            :class:`~clarabel.NonnegativeConeT` this means $b - Ax ≥ 0$, i.e.
            $Ax ≤ b$.  Each OSQP row is therefore expanded into at most two
            Clarabel rows.
        """
        import clarabel
        import scipy.sparse as sp

        m = A.shape[0]
        eq_rows: list[np.ndarray] = []
        eq_b: list[float] = []
        ineq_rows: list[np.ndarray] = []
        ineq_b: list[float] = []

        for i in range(m):
            li, ui = l[i], u[i]
            row = A[i]
            if np.isfinite(li) and np.isfinite(ui) and li == ui:
                # Equality: A[i]x = li  →  ZeroCone
                eq_rows.append(row)
                eq_b.append(float(li))
            else:
                if np.isfinite(ui):
                    # A[i]x <= ui  →  ui - A[i]x >= 0  →  NonnegativeCone
                    ineq_rows.append(row)
                    ineq_b.append(float(ui))
                if np.isfinite(li):
                    # A[i]x >= li  →  A[i]x - li >= 0  →  -(-A[i])x - li <= 0
                    # Clarabel: b - Ax >= 0 so row is -A[i], b is -li
                    ineq_rows.append(-row)
                    ineq_b.append(float(-li))

        n_eq = len(eq_rows)
        n_ineq = len(ineq_rows)

        all_rows = eq_rows + ineq_rows
        all_b = np.array(eq_b + ineq_b, dtype=float)

        A_cl = sp.csc_matrix(
            np.array(all_rows, dtype=float).reshape(n_eq + n_ineq, A.shape[1])
        )

        cones = []
        if n_eq > 0:
            cones.append(clarabel.ZeroConeT(n_eq))
        if n_ineq > 0:
            cones.append(clarabel.NonnegativeConeT(n_ineq))

        P_sparse = sp.csc_matrix(P, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        settings = clarabel.DefaultSettings()
        settings.verbose = self._verbose
        settings.max_iter = self._max_iter
        settings.tol_gap_abs = self._eps_abs
        settings.tol_gap_rel = self._eps_rel
        settings.tol_feas = self._eps_abs

        solver = clarabel.DefaultSolver(P_sparse, q_arr, A_cl, all_b, cones, settings)
        result = solver.solve()

        x = np.asarray(result.x, dtype=float)
        raw_status = str(result.status)
        status = "solved" if "solved" in raw_status.lower() else raw_status
        obj = float(0.5 * x @ P @ x + q_arr @ x)
        return QPResult(x=x, obj=obj, status=status)


def best_available_qp_solver() -> "QPSolver":
    """Return the best available QP solver (OSQP > Clarabel > SciPy).

    Tries solvers in order of preference: OSQP (ADMM, fast for large-scale),
    then Clarabel (interior-point, high accuracy), then the SciPy SLSQP fallback.

    Returns:
        A :class:`QPSolver` instance backed by the best installed package.
    """
    try:
        return OSQPQPSolver()
    except ImportError:
        pass
    try:
        return ClarabelQPSolver()
    except ImportError:
        pass
    return SciPyQPSolver()


# ---------------------------------------------------------------------------
# BundleResult
# ---------------------------------------------------------------------------


@dataclass
class BundleResult:
    """Result from a bundle method optimisation run.

    Attributes:
        x_best: Best primal iterate found (lowest function value).
        f_best: Function value at ``x_best``; upper bound on the optimum.
        f_low: Lower bound on the optimum from the cutting-plane model.
        gap: Optimality gap ``f_best - f_low`` (non-negative for a valid lower bound).
        converged: Whether the gap tolerance was satisfied.
        num_iterations: Number of bundle loop iterations (master solves),
            excluding the initial oracle evaluation before the loop.
        num_serious_steps: Number of serious (descent) steps taken.
        function_values: History of function values at each iteration.
        iterates: Optional history of all iterates (memory intensive).
    """

    x_best: "Vector"
    f_best: float
    f_low: float
    gap: float
    converged: bool
    num_iterations: int
    num_serious_steps: int
    function_values: List[float]
    iterates: Optional[List["Vector"]] = None


# ---------------------------------------------------------------------------
# Proximal Bundle Method
# ---------------------------------------------------------------------------


def _get_value_and_subgradient(
    oracle: NonLinearForm, x: "Vector"
) -> "tuple[float, Vector]":
    """Duck-typed value + subgradient query.

    If *oracle* has a ``value_and_subgradient`` method (e.g.
    :class:`~pygeoinf.backus_gilbert.DualMasterCostFunction`) it is called
    to share computation.  Otherwise the value and subgradient are obtained
    via separate ``oracle(x)`` and ``oracle.subgradient(x)`` calls.

    Args:
        oracle: A :class:`NonLinearForm` with a subgradient oracle.
        x: The query point.

    Returns:
        ``(f, g)`` — scalar value and subgradient vector.
    """
    if (
        hasattr(oracle, "value_and_subgradient")
        and callable(oracle.value_and_subgradient)
    ):
        return oracle.value_and_subgradient(x)
    return oracle(x), oracle.subgradient(x)


class ProximalBundleMethod:
    """Proximal bundle method for minimising a non-smooth convex function.

    Solves:
        min_{lambda in D} f(lambda)

    where f is a convex function accessible through a value + subgradient
    oracle (a :class:`~pygeoinf.nonlinear_forms.NonLinearForm` with
    ``subgradient``).

    At each iteration the *master QP* is:
        min_{lambda, t} t + (rho / 2) ||lambda - lambda_hat||^2
        subject to: f_j + <g_j, lambda - x_j> <= t  for all j in bundle

    where lambda_hat is the current *stability centre* and rho > 0
    is the proximal weight.

    A *serious step* is taken whenever the new oracle value f(lambda_+) <
    f(lambda_hat); otherwise a *null step* occurs and rho is increased to
    tighten the proximal term.

    Args:
        oracle: Non-smooth convex functional with subgradient oracle.
        rho0: Initial proximal weight rho > 0.
        rho_factor: Multiplicative factor applied to rho on null steps (divide
            on serious steps).
        tolerance: Convergence tolerance; terminates when the duality gap
            f_up - f_low <= tolerance.
        max_iterations: Maximum number of oracle calls.
        bundle_size: Maximum number of cuts retained in the bundle.
        store_iterates: If ``True``, all iterates are stored in
            :attr:`BundleResult.iterates`.
        qp_solver: QP solver implementing :class:`QPSolver`.  Defaults to
            :class:`SciPyQPSolver` if ``None``.

    Examples:
        >>> from pygeoinf.hilbert_space import EuclideanSpace
        >>> from pygeoinf.nonlinear_forms import NonLinearForm
        >>> import numpy as np
        >>> domain = EuclideanSpace(1)
        >>> f = lambda x: float(x[0]**2 + 2*x[0])
        >>> g = lambda x: np.array([2*x[0] + 2.0])
        >>> oracle = NonLinearForm(domain, f, subgradient=g)
        >>> solver = ProximalBundleMethod(oracle, tolerance=1e-5)
        >>> result = solver.solve(domain.from_components(np.array([2.0])))
        >>> np.testing.assert_allclose(domain.to_components(result.x_best), [-1.0], atol=1e-3)
    """

    def __init__(
        self,
        oracle: NonLinearForm,
        /,
        *,
        rho0: float = 1.0,
        rho_factor: float = 2.0,
        tolerance: float = 1e-6,
        max_iterations: int = 500,
        bundle_size: int = 100,
        store_iterates: bool = False,
        qp_solver: Optional["QPSolver"] = None,
    ) -> None:
        if rho0 <= 0:
            raise ValueError("rho0 must be positive")
        if rho_factor <= 1.0:
            raise ValueError("rho_factor must be > 1")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if bundle_size < 1:
            raise ValueError("bundle_size must be >= 1")

        self._oracle = oracle
        self._rho0 = rho0
        self._rho_factor = rho_factor
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        self._bundle_size = bundle_size
        self._store_iterates = store_iterates
        self._qp_solver: QPSolver = qp_solver if qp_solver is not None else SciPyQPSolver()

    def _solve_master(
        self,
        bundle: Bundle,
        lam_hat: "Vector",
        rho: float,
        domain: "HilbertSpace",
        x0_comps: np.ndarray,
        t_warm: float = 0.0,
    ) -> "tuple[Vector, float]":
        """Solve the proximal bundle master QP.

        Minimises:
            t + (rho / 2) ||lambda - lambda_hat||^2

        subject to the bundle cutting-plane constraints.

        Args:
            bundle: Current bundle of cuts.
            lam_hat: Stability centre $\\hat{λ}$.
            rho: Proximal weight.
            domain: Hilbert space of the decision variable.
            x0_comps: Warm-start components for the QP solver.

        Returns:
            ``(lam_next, qp_obj)`` — optimal point (as a Hilbert-space element)
            and the QP objective value.
        """
        d = domain.dim
        lam_hat_c = domain.to_components(lam_hat)

        # Hessian: rho * I_{d} in the λ block, 0 for t
        P = np.zeros((d + 1, d + 1))
        P[:d, :d] = rho * np.eye(d)

        # Linear cost: -rho * lam_hat for λ components, 1 for t
        q_vec = np.zeros(d + 1)
        q_vec[:d] = -rho * lam_hat_c
        q_vec[d] = 1.0

        # Inequality constraints from bundle: A_ineq @ z <= b_ineq
        A_ineq, b_ineq = bundle.linearization_matrix(lam_hat, domain)
        n_cuts = A_ineq.shape[0]

        l_bounds = np.full(n_cuts, -np.inf)
        u_bounds = b_ineq

        # Warm start: use t_warm (should be feasible, e.g. f_hat) to help SLSQP
        x0 = np.append(x0_comps, t_warm)

        result = self._qp_solver.solve(P, q_vec, A_ineq, l_bounds, u_bounds, x0=x0)
        lam_next_c = result.x[:d]
        t_opt = float(result.x[d])  # the t variable at the QP solution
        lam_next = domain.from_components(lam_next_c)
        return lam_next, t_opt

    def solve(self, x0: "Vector") -> BundleResult:
        """Run the proximal bundle method starting from *x0*.

        Args:
            x0: Initial point in the domain of the oracle.

        Returns:
            A :class:`BundleResult` summarising the optimisation run.
        """
        domain = self._oracle.domain
        lam_hat = x0

        f_hat, g_hat = _get_value_and_subgradient(self._oracle, lam_hat)

        f_up = f_hat
        best_lam = lam_hat
        f_low = -np.inf

        bundle = Bundle()
        bundle.add_cut(Cut(x=lam_hat, f=f_hat, g=g_hat, iteration=0))

        rho = self._rho0
        n_serious = 0
        function_values: List[float] = [f_hat]
        iterates: Optional[List["Vector"]] = [] if self._store_iterates else None

        lam_hat_c = domain.to_components(lam_hat)

        t_warm = f_hat  # feasible warm-start for the t variable

        for k in range(self._max_iterations):
            # Solve master QP — returns (lam_next, t_opt) where t_opt = result.x[d]
            # is the cutting-plane model value at lam_next: hat_phi(lam_next) <= f(lam_next).
            lam_next, t_opt = self._solve_master(
                bundle, lam_hat, rho, domain, lam_hat_c, t_warm=t_warm
            )

            lam_next_c = domain.to_components(lam_next)

            # Oracle evaluation
            f_next, g_next = _get_value_and_subgradient(self._oracle, lam_next)

            function_values.append(f_next)
            if iterates is not None:
                iterates.append(lam_next)

            # Update best
            if f_next < f_up:
                f_up = f_next
                best_lam = lam_next

            # ------------------------------------------------------------------
            # Step classification: serious or null
            # ------------------------------------------------------------------
            if f_next < f_hat:
                # Serious step: accept lam_next as new stability centre.
                # Reset the lower bound — the old f_low was measured relative to
                # the previous stability centre and may be meaningless now.
                lam_hat = lam_next
                lam_hat_c = lam_next_c
                f_hat = f_next
                n_serious += 1
                rho = rho / self._rho_factor
                f_low = -np.inf  # reset; will be re-established after a null step
            else:
                # Null step: tighten proximal weight.
                # t_opt = hat_phi(lam_next) is a valid cutting-plane lower bound
                # relative to the current stability centre.  Update monotonically.
                rho = rho * self._rho_factor
                f_low = max(f_low, t_opt)

            # Add cut and compress bundle
            bundle.add_cut(Cut(x=lam_next, f=f_next, g=g_next, iteration=k + 1))
            bundle.compress(self._bundle_size)
            t_warm = f_hat  # always a feasible warm-start

            # ------------------------------------------------------------------
            # Convergence check (after null-step lower-bound update)
            # ------------------------------------------------------------------
            # gap = f_hat - f_low measures how tight the cutting-plane model is
            # at the stability centre.  When small, the model is nearly exact and
            # lam_hat is near-optimal (f_hat ≈ f* and 0 ∈ ∂f(lam_hat)).
            if f_low > -np.inf:
                gap = f_hat - f_low
                if gap <= self._tolerance:
                    return BundleResult(
                        x_best=best_lam,
                        f_best=f_up,
                        f_low=f_low,
                        gap=gap,
                        converged=True,
                        num_iterations=k + 1,
                        num_serious_steps=n_serious,
                        function_values=function_values,
                        iterates=iterates,
                    )

        # Max iterations reached without convergence
        gap = f_hat - f_low if f_low > -np.inf else np.inf
        return BundleResult(
            x_best=best_lam,
            f_best=f_up,
            f_low=f_low,
            gap=gap,
            converged=False,
            num_iterations=self._max_iterations,
            num_serious_steps=n_serious,
            function_values=function_values,
            iterates=iterates,
        )


# ---------------------------------------------------------------------------
# Level Bundle Method
# ---------------------------------------------------------------------------


class LevelBundleMethod:
    """Level bundle method for minimising a non-smooth convex function.

    Solves:
        min_{lambda in D} f(lambda)

    where f is a convex function accessible through a value + subgradient
    oracle (a :class:`~pygeoinf.nonlinear_forms.NonLinearForm` with
    ``subgradient``).

    At each iteration the *level master QP* is:
        min_{lambda, t} (1/2) ||lambda - lambda_hat||^2
        subject to: f_j + <g_j, lambda - x_j> <= t  for all j
                   t <= f_lev

    where the level is: f_lev = alpha * f_low + (1 - alpha) * f_up, alpha in (0,1).

    The lower bound f_low is maintained as the LP optimal value of
    the cutting-plane model:
        f_LP = min_{lambda} hat_phi(lambda)
             = min_{lambda, t} t
        subject to: f_j + <g_j, lambda - x_j> <= t

    **Infeasibility handling.**  If the level QP is infeasible (which can
    happen when f_lev < f_LP for a tight alpha), alpha is widened by a factor
    of 1.5 (capped at 0.9) for up to three attempts. If all attempts fail an
    emergency proximal step is taken (minimize t + (1/2) ||lambda - lambda_hat||^2
    over the current bundle) so the stability centre and bundle are always updated.

    Args:
        oracle: Non-smooth convex functional with subgradient oracle.
        alpha: Level parameter alpha in (0, 1) controlling how
            aggressively the level is set towards the lower bound. Smaller
            values are more aggressive (risk infeasibility); larger are more
            conservative. Defaults to ``0.1``.
        tolerance: Convergence tolerance; terminates when the duality gap
            f_up - f_low <= tolerance.
        max_iterations: Maximum number of oracle calls.
        bundle_size: Maximum number of cuts retained in the bundle.
        store_iterates: If ``True``, all iterates are stored in
            :attr:`BundleResult.iterates`.
        qp_solver: QP solver implementing :class:`QPSolver`.  Defaults to
            :class:`SciPyQPSolver` if ``None``.

    Examples:
        >>> from pygeoinf.hilbert_space import EuclideanSpace
        >>> from pygeoinf.nonlinear_forms import NonLinearForm
        >>> import numpy as np
        >>> domain = EuclideanSpace(1)
        >>> f = lambda x: float(x[0]**2 + 2*x[0])
        >>> g = lambda x: np.array([2*x[0] + 2.0])
        >>> oracle = NonLinearForm(domain, f, subgradient=g)
        >>> solver = LevelBundleMethod(oracle, tolerance=1e-5)
        >>> result = solver.solve(domain.from_components(np.array([2.0])))
        >>> np.testing.assert_allclose(domain.to_components(result.x_best), [-1.0], atol=1e-3)
    """

    def __init__(
        self,
        oracle: NonLinearForm,
        /,
        *,
        alpha: float = 0.1,
        tolerance: float = 1e-6,
        max_iterations: int = 500,
        bundle_size: int = 100,
        store_iterates: bool = False,
        qp_solver: Optional["QPSolver"] = None,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if bundle_size < 1:
            raise ValueError("bundle_size must be >= 1")

        self._oracle = oracle
        self._alpha = alpha
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        self._bundle_size = bundle_size
        self._store_iterates = store_iterates
        self._qp_solver: QPSolver = qp_solver if qp_solver is not None else SciPyQPSolver()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_lp_lower_bound(
        self,
        bundle: Bundle,
        lam_hat: "Vector",
        domain: "HilbertSpace",
        lam_hat_c: np.ndarray,
    ) -> float:
        """Compute a lower bound on f* via the bundle LP.

        Solves the cutting-plane LP:
            f_LP = min_{lambda, t} t
            subject to: f_j + <g_j, lambda - x_j> <= t
                       ||lambda - lambda_hat||_inf <= R

        where R = 1e3 * (1 + ||lambda_hat||_inf) is a large box that keeps
        the LP bounded without biasing the lower bound.

        The LP is solved via :func:`scipy.optimize.linprog` (HiGHS simplex),
        which is more reliable than ADMM-based QP solvers for this
        nearly-unbounded pure-LP subproblem.

        Since hat_phi(lambda) <= f(lambda) for all lambda, the
        LP optimum satisfies f_LP <= f*, providing a valid global lower bound.

        Args:
            bundle: Current bundle of cuts.
            lam_hat: Stability centre $\\hat{λ}$.
            domain: Hilbert space of the decision variable.
            lam_hat_c: Component array of the stability centre.

        Returns:
            Scalar lower-bound estimate f_LP, or ``-np.inf`` if
            the LP cannot be solved reliably (e.g. only a single cut present
            when the problem is genuinely unbounded).
        """
        from scipy.optimize import linprog

        d = domain.dim

        # LP variables: z = [lambda (d), t (1)]  (d+1 variables)
        # Objective: min t  →  c = [0, ..., 0, 1]
        c = np.zeros(d + 1)
        c[d] = 1.0

        # Cut constraints: A_ineq @ z <= b_ineq
        A_ub, b_ub = bundle.linearization_matrix(lam_hat, domain)

        # Box constraint: -R <= lambda_i - lam_hat_i <= R  (no constraint on t)
        R = 1e3 * (1.0 + float(np.max(np.abs(lam_hat_c))))
        lb = np.append(lam_hat_c - R, -np.inf)  # lower var bounds
        ub = np.append(lam_hat_c + R, np.inf)   # upper var bounds

        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(lb, ub)), method="highs"
        )
        if result.status == 0:
            return float(result.x[d])
        return -np.inf  # infeasible or unbounded — return conservative bound

    def _solve_level_master(
        self,
        bundle: Bundle,
        lam_hat: "Vector",
        f_lev: float,
        domain: "HilbertSpace",
        lam_hat_c: np.ndarray,
        t_warm: float = 0.0,
    ) -> QPResult:
        """Solve the level-bundle master QP.

        Minimises:
            (1/2) ||lambda - lambda_hat||^2

        subject to the bundle cutting-plane constraints AND the level
        constraint t <= f_lev.

        Decision variable: z = [lambda_1, ..., lambda_d, t].

        Args:
            bundle: Current bundle of cuts.
            lam_hat: Stability centre $\\hat{λ}$.
            f_lev: Level value; upper bound on $t$.
            domain: Hilbert space of the decision variable.
            lam_hat_c: Component array of the stability centre.
            t_warm: Warm-start value for the $t$ component.

        Returns:
            :class:`QPResult` from the QP solver.
        """
        d = domain.dim

        # Objective: 0.5 ||λ - lam_hat||^2 = 0.5 λ'λ - lam_hat'λ + const
        # P = block_diag(I_d, 0),  q = [-lam_hat_c, 0]
        P = np.zeros((d + 1, d + 1))
        P[:d, :d] = np.eye(d)

        q_vec = np.zeros(d + 1)
        q_vec[:d] = -lam_hat_c

        # Cut constraints: A_ineq @ z <= b_ineq
        A_ineq, b_ineq = bundle.linearization_matrix(lam_hat, domain)
        n_cuts = A_ineq.shape[0]

        # Level constraint: t <= f_lev (i.e. [0...0, 1] @ z <= f_lev)
        A_level = np.zeros((1, d + 1))
        A_level[0, d] = 1.0

        A_full = np.vstack([A_ineq, A_level])
        l_full = np.full(n_cuts + 1, -np.inf)
        u_full = np.append(b_ineq, f_lev)

        x0 = np.append(lam_hat_c, t_warm)
        return self._qp_solver.solve(P, q_vec, A_full, l_full, u_full, x0=x0)

    def _solve_emergency_proximal(
        self,
        bundle: Bundle,
        lam_hat: "Vector",
        domain: "HilbertSpace",
        lam_hat_c: np.ndarray,
        f_hat: float,
    ) -> "tuple[Vector, float]":
        """Emergency proximal fallback when all level QP attempts fail.

        Minimises: t + (1/2) ||lambda - lambda_hat||^2
        subject to the bundle cutting-plane constraints (no level constraint).

        Args:
            bundle: Current bundle of cuts.
            lam_hat: Stability centre.
            domain: Hilbert space of the decision variable.
            lam_hat_c: Component array of the stability centre.
            f_hat: Function value at stability centre (used for warm start).

        Returns:
            ``(lam_next, t_opt)`` — next iterate and its cutting-plane value.
        """
        d = domain.dim

        P = np.zeros((d + 1, d + 1))
        P[:d, :d] = np.eye(d)

        q_vec = np.zeros(d + 1)
        q_vec[:d] = -lam_hat_c
        q_vec[d] = 1.0  # penalise t (same as proximal with rho=1)

        A_ineq, b_ineq = bundle.linearization_matrix(lam_hat, domain)
        n_cuts = A_ineq.shape[0]
        l_bounds = np.full(n_cuts, -np.inf)

        x0 = np.append(lam_hat_c, f_hat)
        result = self._qp_solver.solve(P, q_vec, A_ineq, l_bounds, b_ineq, x0=x0)
        lam_next = domain.from_components(result.x[:d])
        t_opt = float(result.x[d])
        return lam_next, t_opt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, x0: "Vector") -> BundleResult:
        """Run the level bundle method starting from *x0*.

        Args:
            x0: Initial point in the domain of the oracle.

        Returns:
            A :class:`BundleResult` summarising the optimisation run.
        """
        domain = self._oracle.domain
        d = domain.dim

        lam_hat = x0
        f_hat, g_hat = _get_value_and_subgradient(self._oracle, lam_hat)

        f_up = f_hat
        best_lam = lam_hat
        f_low = -np.inf

        bundle = Bundle()
        bundle.add_cut(Cut(x=lam_hat, f=f_hat, g=g_hat, iteration=0))

        n_serious = 0
        function_values: List[float] = [f_hat]
        iterates: Optional[List["Vector"]] = [] if self._store_iterates else None

        lam_hat_c = domain.to_components(lam_hat)

        for k in range(self._max_iterations):
            # ------------------------------------------------------------------
            # Step 1: update LP lower bound from the cutting-plane model.
            # ------------------------------------------------------------------
            f_lp = self._compute_lp_lower_bound(bundle, lam_hat, domain, lam_hat_c)
            f_low = max(f_low, f_lp)

            # ------------------------------------------------------------------
            # Step 2: convergence check (gap between upper and lower bound).
            # ------------------------------------------------------------------
            if f_low > -np.inf:
                gap = f_up - f_low
                if gap <= self._tolerance:
                    return BundleResult(
                        x_best=best_lam,
                        f_best=f_up,
                        f_low=f_low,
                        gap=gap,
                        converged=True,
                        num_iterations=k,
                        num_serious_steps=n_serious,
                        function_values=function_values,
                        iterates=iterates,
                    )

            # ------------------------------------------------------------------
            # Step 3: solve the level QP with infeasibility recovery.
            # ------------------------------------------------------------------
            lam_next: Optional["Vector"] = None
            alpha_try = self._alpha

            for attempt in range(3):
                if f_low > -np.inf:
                    f_lev_try = alpha_try * f_low + (1.0 - alpha_try) * f_up
                else:
                    f_lev_try = f_up

                t_warm = min(f_lev_try, f_hat)
                qp_res = self._solve_level_master(
                    bundle, lam_hat, f_lev_try, domain, lam_hat_c, t_warm=t_warm
                )
                if qp_res.status == "solved":
                    lam_next = domain.from_components(qp_res.x[:d])
                    break
                # Widen level for next attempt
                alpha_try = min(alpha_try * 1.5, 0.9)

            if lam_next is None:
                # All level QP attempts failed — emergency proximal fallback.
                lam_next, _ = self._solve_emergency_proximal(
                    bundle, lam_hat, domain, lam_hat_c, f_hat
                )

            # ------------------------------------------------------------------
            # Step 5: oracle evaluation at the new candidate point.
            # ------------------------------------------------------------------
            f_next, g_next = _get_value_and_subgradient(self._oracle, lam_next)
            function_values.append(f_next)
            if iterates is not None:
                iterates.append(lam_next)

            # Update upper bound / best point.
            if f_next < f_up:
                f_up = f_next
                best_lam = lam_next

            # ------------------------------------------------------------------
            # Step 6: serious or null step classification.
            # ------------------------------------------------------------------
            lam_next_c = domain.to_components(lam_next)
            if f_next < f_hat:
                # Serious step: accept lam_next as new stability centre.
                lam_hat = lam_next
                lam_hat_c = lam_next_c
                f_hat = f_next
                n_serious += 1

            # Add cut and compress bundle (always, regardless of step type).
            bundle.add_cut(Cut(x=lam_next, f=f_next, g=g_next, iteration=k + 1))
            bundle.compress(self._bundle_size)

        # ------------------------------------------------------------------
        # Max iterations reached without convergence.
        # ------------------------------------------------------------------
        gap = f_up - f_low if f_low > -np.inf else np.inf
        return BundleResult(
            x_best=best_lam,
            f_best=f_up,
            f_low=f_low,
            gap=gap,
            converged=False,
            num_iterations=self._max_iterations,
            num_serious_steps=n_serious,
            function_values=function_values,
            iterates=iterates,
        )


# ---------------------------------------------------------------------------
# Multi-direction batch helper
# ---------------------------------------------------------------------------
def solve_support_values(
    cost,
    qs,
    solver,
    lambda0=None,
    *,
    warm_start: bool = True,
    n_jobs: int = 1,
    verbose: bool = False,
):
    """Compute support function values for multiple directions.

    Unified solver that works with both bundle methods
    (DualMasterCostFunction + ProximalBundleMethod) and the direct primal
    solver (PrimalKKTSolver built via :meth:`PrimalKKTSolver.from_cost`).

    In both cases ``cost`` is the :class:`~pygeoinf.backus_gilbert.DualMasterCostFunction`
    that defines the problem; ``solver`` is the implementation that solves it.
    The property operator T is always read from the cost, so the API is
    symmetric regardless of solver type.

    The support function evaluated at direction q is:

        h_U(q) = min_{lambda} phi(lambda; q)          [bundle method]
        h_U(q) = <c, u*>  where c = T* q              [primal KKT]

    Args:
        cost: :class:`~pygeoinf.backus_gilbert.DualMasterCostFunction`
            defining the inference problem.
        qs: Directions to evaluate; a list of property-space Vectors or an
            ``np.ndarray`` of shape ``(p, prop_dim)``.
        solver: Either a bundle-method solver (ProximalBundleMethod or
            LevelBundleMethod) or a :class:`PrimalKKTSolver` built via
            :meth:`PrimalKKTSolver.from_cost`.
        lambda0: (Bundle only) Initial data-space vector for the first
            direction. Ignored when solver is a PrimalKKTSolver.
        warm_start: (Bundle only) If ``True`` (default), each direction
            warm-starts from the previous optimal lambda. Ignored for
            PrimalKKTSolver.
        n_jobs: Number of parallel jobs. ``1`` = fully sequential.
            ``>1`` = joblib Parallel (warm-starting disabled for bundle).

    Returns:
        values: ``np.ndarray`` of shape ``(p,)``, support values h_U(q_i).
        lambdas: ``list`` of length ``p``:
            - Bundle: optimal lambda (x_best) for each direction.
            - PrimalKKT: KKT multiplier lambda (multipliers[0]).
        diagnostics: ``list`` of BundleResult or KKTResult per direction.

    Raises:
        ValueError: If solver is PrimalKKTSolver but was not built via
            ``from_cost`` (i.e. T is not stored).
        ImportError: If ``n_jobs > 1`` and ``joblib`` is not installed
            (falls back to sequential with a warning).
    """
    if type(solver).__name__ == "PrimalKKTSolver":
        if solver._T is None:
            raise ValueError(
                "PrimalKKTSolver must be constructed via PrimalKKTSolver.from_cost(cost) "
                "to be used with solve_support_values."
            )
        return _solve_support_values_primal_kkt(solver, qs, n_jobs=n_jobs, verbose=verbose)
    else:
        if lambda0 is None:
            raise ValueError(
                "lambda0 is required for bundle-method solvers."
            )
        return _solve_support_values_bundle(
            cost, qs, solver, lambda0, warm_start=warm_start, n_jobs=n_jobs
        )


def _solve_support_values_bundle(cost, qs, solver, lambda0, *, warm_start, n_jobs):
    """Bundle-method implementation used by solve_support_values."""
    if n_jobs > 1:
        try:
            import copy
            from joblib import Parallel, delayed

            def _solve_one(cost_copy, q, solver_copy, lam0):
                cost_copy.set_direction(q)
                solver_copy._oracle = cost_copy
                result = solver_copy.solve(lam0)
                return result.f_best, result.x_best, result

            results_raw = Parallel(n_jobs=n_jobs)(
                delayed(_solve_one)(
                    copy.deepcopy(cost), q, copy.deepcopy(solver), lambda0
                )
                for q in qs
            )
            values_list, lambdas, diagnostics = zip(*results_raw)
            return np.array(values_list), list(lambdas), list(diagnostics)

        except ImportError:
            import warnings
            warnings.warn(
                "joblib is not installed; falling back to sequential execution.",
                RuntimeWarning,
                stacklevel=2,
            )

    lam_current = lambda0
    values_list = []
    lambdas = []
    diagnostics = []
    for q in qs:
        cost.set_direction(q)
        result = solver.solve(lam_current if warm_start else lambda0)
        values_list.append(result.f_best)
        lambdas.append(result.x_best)
        diagnostics.append(result)
        if warm_start:
            lam_current = result.x_best
    return np.array(values_list), lambdas, diagnostics


def _solve_support_values_primal_kkt(solver, qs, *, n_jobs, verbose=False):
    """PrimalKKTSolver implementation used by solve_support_values."""
    import time
    T = solver._T
    model_space = solver._model_space
    n_total = len(qs)

    if n_jobs > 1:
        try:
            import copy
            from joblib import Parallel, delayed

            def _solve_one(solver_inst, q, T_op, model_space_inst, idx):
                t0 = time.perf_counter()
                c = T_op.adjoint(q)
                result = solver_inst.solve(c)
                h_val = float(model_space_inst.inner_product(c, result.m))
                return idx, h_val, result.multipliers[0], result, time.perf_counter() - t0

            backend = os.environ.get("PYGEOINF_KKT_PARALLEL_BACKEND")
            use_shared_solver = (
                backend == "threading"
                and getattr(solver, "_warm_start_disabled", False)
            )

            def _solver_for_task():
                if use_shared_solver:
                    return solver
                return copy.deepcopy(solver)

            parallel_kwargs = {
                "n_jobs": n_jobs,
                "return_as": "generator_unordered",
            }
            if backend:
                parallel_kwargs["backend"] = backend

            gen = Parallel(**parallel_kwargs)(
                delayed(_solve_one)(_solver_for_task(), q, T, model_space, i)
                for i, q in enumerate(qs)
            )

            results_by_idx = {}
            t_start = time.perf_counter()
            done = 0
            for idx, h_val, lam, diag, elapsed_one in gen:
                results_by_idx[idx] = (h_val, lam, diag)
                done += 1
                if verbose:
                    wall = time.perf_counter() - t_start
                    avg = wall / done
                    eta = avg * (n_total - done)
                    converged = "✓" if diag.converged else "✗"
                    print(
                        f"  [{done:2d}/{n_total}] dir {idx+1:2d}  "
                        f"h={h_val:+.4f}  {converged}  "
                        f"solve {elapsed_one:.1f}s  "
                        f"wall {wall:.0f}s  ETA {eta:.0f}s",
                        flush=True,
                    )

            ordered = [results_by_idx[i] for i in range(n_total)]
            values_list, lambdas, diagnostics = zip(*ordered)
            return np.array(values_list), list(lambdas), list(diagnostics)

        except ImportError:
            import warnings
            warnings.warn(
                "joblib is not installed; falling back to sequential execution.",
                RuntimeWarning,
                stacklevel=2,
            )

    values_list = []
    lambdas = []
    diagnostics = []
    for q in qs:
        c = T.adjoint(q)
        result = solver.solve(c)
        h_val = float(model_space.inner_product(c, result.m))
        values_list.append(h_val)
        lambdas.append(result.multipliers[0])
        diagnostics.append(result)
    return np.array(values_list), lambdas, diagnostics


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# KKT Primal Solver result
# ---------------------------------------------------------------------------


@dataclass
class KKTResult:
    """Result from :class:`PrimalKKTSolver`.

    Attributes:
        m: Optimal primal model vector $u^*$ in the model space.
        multipliers: KKT multipliers $(\\lambda^*, \\mu^*)$.  $\\lambda^*$
            enforces the model prior constraint and $\\mu^*$ enforces the
            data-fit constraint.  Both are non-negative.
        converged: ``True`` if the root-finder converged to the required
            tolerance.
        num_iterations: Number of function evaluations used by the
            root-finder (or 1 for the closed-form branch).
    """

    m: "Vector"
    multipliers: "tuple[float, float]"
    converged: bool
    num_iterations: int


# ---------------------------------------------------------------------------
# Chambolle-Pock primal-dual algorithm
# ---------------------------------------------------------------------------


@dataclass
class ChambollePockResult:
    """Result from :class:`ChambollePockSolver`.

    Attributes:
        m: Primal variable m* in B (model space).
        v: Primal variable v* in V (data space).
        mu: Dual variable mu* in D (data space). Approximates the
            optimal Lagrange multiplier for the equality constraint
            G @ m + v = d_tilde.
        primal_dual_gap: Feasibility residual ||G @ m* + v* - d_tilde||
            at termination.
        converged: ``True`` if ``primal_dual_gap < tolerance``.
        num_iterations: Number of iterations performed.
    """

    m: "Vector"
    v: "Vector"
    mu: "Vector"
    primal_dual_gap: float
    converged: bool
    num_iterations: int


class ChambollePockSolver:
    r"""Solve the primal feasibility form of the dual master via Chambolle-Pock.

    Solves the constrained maximisation:
        h_U(q) = max_{m in B, v in V} <c, m>
                 subject to: G @ m + v = d_tilde

    where c = T* @ q is the linear objective and the feasible set
    (B, V, G, d_tilde) is **fixed**, using the first-order primal-dual
    algorithm of Chambolle & Pock (2011).

    The saddle-point reformulation (with dual variable mu in D) is:
        min_{m in B, v in V} max_{mu}
            <G @ m + v - d_tilde, mu> - <c, m>

    with operator K = [G; I_D] : M x D -> D.

    **Convergence rate:** O(1/N) in the primal-dual gap when
    tau * sigma * ||K||^2 <= 1 (where ||K||^2 <= ||G||^2 + 1).

    This is particularly efficient when the objective c = T* @ q changes while
    the feasible set (B, V, G, d_tilde) remains fixed.

    Args:
        B: Support function for the model prior (B subset of M).
        V: Support function for the data error set (V subset of D).
        G: Forward operator G: M -> D.
        d_tilde: Observed data vector d_tilde in D.
        sigma: Dual step size. If ``None``, auto-selected from power iteration.
        tau: Primal step size. If ``None``, auto-selected from power iteration.
        theta: Over-relaxation parameter (default 1.0).
        max_iterations: Maximum number of iterations.
        tolerance: Convergence tolerance on feasibility residual
            ||G @ m + v - d_tilde||.

    References:
        Chambolle A. & Pock T. (2011). A First-Order Primal-Dual Algorithm for
        Convex Problems with Applications to Imaging.
        *Journal of Mathematical Imaging and Vision*, 40(1), 120–145.
        https://doi.org/10.1007/s10851-010-0251-1
    """

    def __init__(
        self,
        B: "SupportFunction",
        V: "SupportFunction",
        G: "LinearOperator",
        d_tilde: "Vector",
        /,
        *,
        sigma: Optional[float] = None,
        tau: Optional[float] = None,
        theta: float = 1.0,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ) -> None:
        self._B = B
        self._V = V
        self._G = G
        self._d_tilde = d_tilde
        self._theta = theta
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._model_space = B.primal_domain
        self._data_space = G.codomain

        # Auto-compute step sizes if not provided
        if sigma is None or tau is None:
            _sigma, _tau = self._compute_step_sizes()
            self._sigma = sigma if sigma is not None else _sigma
            self._tau = tau if tau is not None else _tau
        else:
            self._sigma = sigma
            self._tau = tau

    def _compute_step_sizes(self) -> "tuple[float, float]":
        """Estimate ||G|| via 20 steps of power iteration.

        Returns:
            (sigma, tau) satisfying tau * sigma * ||K||^2 <= 0.99
            where K = [G; I] so ||K||^2 <= ||G||^2 + 1.
        """
        model_space = self._model_space
        G = self._G

        rng = np.random.default_rng(0)
        x_m = model_space.from_components(
            rng.standard_normal(model_space.dim)
        )
        # Normalise initial vector
        n0 = model_space.norm(x_m)
        if n0 > 1e-14:
            x_m = model_space.multiply(1.0 / n0, x_m)

        eigenvalue_est = 1.0
        for _ in range(20):
            y = G(x_m)                  # data space: G x_m
            z = G.adjoint(y)            # model space: G^T G x_m
            eigenvalue_est = model_space.norm(z)   # ||G^T G x_m|| -> sigma_max^2
            if eigenvalue_est < 1e-14:
                eigenvalue_est = 0.0
                break
            x_m = model_space.multiply(1.0 / eigenvalue_est, z)

        # eigenvalue_est ≈ ||G||^2  (dominant eigenvalue of G^T G)
        G_norm_est = float(np.sqrt(max(eigenvalue_est, 0.0)))
        # ||K||^2 <= ||G||^2 + 1  (K = [G; I])
        K_norm = float(np.sqrt(G_norm_est ** 2 + 1.0)) * 1.01  # slight over-estimate
        step = 0.99 / K_norm
        return step, step

    def _proj_B(self, z: "Vector") -> "Vector":
        """Project $z$ onto the set $B$.

        Supports :class:`BallSupportFunction` (closed ball projection).
        Raises :class:`NotImplementedError` for other support function types.
        """
        if isinstance(self._B, BallSupportFunction):
            center = self._B._center
            radius = self._B._radius
            H = self._model_space
            diff = H.subtract(z, center)
            dist = H.norm(diff)
            if dist <= radius:
                return z
            # proj_B(z) = c + r * (z - c) / ||z - c||
            return H.add(center, H.multiply(radius / dist, diff))
        elif isinstance(self._B, EllipsoidSupportFunction):
            raise NotImplementedError(
                "Projection onto EllipsoidSupportFunction is not yet implemented "
                "in ChambollePockSolver. Only BallSupportFunction is supported."
            )
        else:
            raise NotImplementedError(
                f"ChambollePockSolver does not know how to project onto "
                f"{type(self._B).__name__}. Only BallSupportFunction is currently "
                "supported."
            )

    def _proj_V(self, z: "Vector") -> "Vector":
        """Project $z$ onto the set $V$.

        Supports :class:`BallSupportFunction` (closed ball projection).
        Raises :class:`NotImplementedError` for other support function types.
        """
        if isinstance(self._V, BallSupportFunction):
            center = self._V._center
            radius = self._V._radius
            H = self._data_space
            diff = H.subtract(z, center)
            dist = H.norm(diff)
            if dist <= radius:
                return z
            # proj_V(z) = c + r * (z - c) / ||z - c||
            return H.add(center, H.multiply(radius / dist, diff))
        elif isinstance(self._V, EllipsoidSupportFunction):
            raise NotImplementedError(
                "Projection onto EllipsoidSupportFunction is not yet implemented "
                "in ChambollePockSolver. Only BallSupportFunction is supported."
            )
        else:
            raise NotImplementedError(
                f"ChambollePockSolver does not know how to project onto "
                f"{type(self._V).__name__}. Only BallSupportFunction is currently "
                "supported."
            )

    def solve(
        self,
        c: "Vector",
        m0: "Optional[Vector]" = None,
    ) -> ChambollePockResult:
        r"""Run the Chambolle-Pock iterations for objective direction $c$.

        Solves

        .. math::

            h_U = \max_{m \in B,\, v \in V}\; \langle c, m \rangle
                  \quad\text{s.t.}\quad Gm + v = \tilde{d}

        Iteration (with $θ = 1$, $K = [G;\\ I]$):

        #. $μ^{n+1} = μ^n + σ(G\\bar{m}^n + \\bar{v}^n - \\tilde{d})$
        #. $m^{n+1} = \\operatorname{proj}_B\\bigl(m^n - τ G^* μ^{n+1} + τ c\\bigr)$
        #. $v^{n+1} = \\operatorname{proj}_V\\bigl(v^n - τ μ^{n+1}\\bigr)$
        #. $\\bar{m}^{n+1} = m^{n+1} + θ(m^{n+1} - m^n)$,
           v_bar^{n+1} = v^{n+1} + theta * (v^{n+1} - v^n)

        Convergence is declared when the feasibility residual
        ||G @ m + v - d_tilde|| < tolerance.

        Args:
            c: Linear objective coefficient in model space (typically c = T* @ q).
            m0: Initial model vector. Defaults to zero.

        Returns:
            :class:`ChambollePockResult` containing the primal optimisers
            $(m^*, v^*)$, dual variable $μ^*$, feasibility gap, and
            convergence diagnostics.
        """
        model_space = self._model_space
        data_space = self._data_space
        G = self._G
        d_tilde = self._d_tilde
        sigma = self._sigma
        tau = self._tau
        theta = self._theta

        m = m0 if m0 is not None else model_space.zero
        v = data_space.zero
        mu = data_space.zero
        m_bar = m
        v_bar = v

        for n in range(self._max_iterations):
            # --- Dual update --------------------------------------------------
            # mu^{n+1} = mu^n + sigma * (G m_bar + v_bar - d_tilde)
            Gm_bar = G(m_bar)
            residual = data_space.subtract(
                data_space.add(Gm_bar, v_bar), d_tilde
            )
            mu_new = data_space.add(mu, data_space.multiply(sigma, residual))

            # --- Primal update m ----------------------------------------------
            # m^{n+1} = proj_B(m - tau * G^* mu_new + tau * c)
            Gstar_mu = G.adjoint(mu_new)
            m_input = model_space.add(
                model_space.subtract(m, model_space.multiply(tau, Gstar_mu)),
                model_space.multiply(tau, c),
            )
            m_new = self._proj_B(m_input)

            # --- Primal update v ----------------------------------------------
            # v^{n+1} = proj_V(v - tau * mu_new)
            v_input = data_space.subtract(v, data_space.multiply(tau, mu_new))
            v_new = self._proj_V(v_input)

            # --- Over-relaxation ----------------------------------------------
            m_bar = model_space.add(
                m_new,
                model_space.multiply(theta, model_space.subtract(m_new, m)),
            )
            v_bar = data_space.add(
                v_new,
                data_space.multiply(theta, data_space.subtract(v_new, v)),
            )

            m, v, mu = m_new, v_new, mu_new

            # --- Convergence check -------------------------------------------
            Gm_new = G(m)
            feas = data_space.norm(
                data_space.subtract(data_space.add(Gm_new, v), d_tilde)
            )

            if feas < self._tolerance:
                return ChambollePockResult(
                    m=m,
                    v=v,
                    mu=mu,
                    primal_dual_gap=feas,
                    converged=True,
                    num_iterations=n + 1,
                )

        # Maximum iterations reached
        Gm_final = G(m)
        feas_final = data_space.norm(
            data_space.subtract(data_space.add(Gm_final, v), d_tilde)
        )
        return ChambollePockResult(
            m=m,
            v=v,
            mu=mu,
            primal_dual_gap=feas_final,
            converged=False,
            num_iterations=self._max_iterations,
        )


def solve_primal_feasibility(
    cost,
    qs: "list[Vector] | np.ndarray",
    cp_solver: ChambollePockSolver,
) -> np.ndarray:
    r"""Compute support values h_U(q_i) using the primal feasibility form.

    Solves one Chambolle-Pock problem for each direction q_i
    (using c = T* @ q_i), exploiting that the feasible set
    (B, V, G, d_tilde) is independent of q.

    The support value for direction q is:
        h_U(q) = max_{m in B, v in V} <T* @ q, m>
                 subject to: G @ m + v = d_tilde
               = <T* @ q, m*(q)>

    where m*(q) is returned by :meth:`ChambollePockSolver.solve`.

    Args:
        cost: :class:`~pygeoinf.backus_gilbert.DualMasterCostFunction` holding
            references to T, G, and the model space.
        qs: Directions to evaluate; a list of Vectors (in the property space)
            or an ``np.ndarray`` of shape ``(p, prop_dim)``.
        cp_solver: Pre-configured :class:`ChambollePockSolver` for the problem.

    Returns:
        ``np.ndarray`` of shape ``(p,)`` with h_U(q_i) for each direction.
    """
    model_space = cost._model_space
    T = cost._T

    values = []
    for q in qs:
        c = T.adjoint(q)              # c = T^* q  (in model space)
        result = cp_solver.solve(c)
        h_value = model_space.inner_product(c, result.m)
        values.append(h_value)
    return np.array(values)


# ---------------------------------------------------------------------------
# Smoothed Dual Master (Moreau-Yosida)
# ---------------------------------------------------------------------------


class SmoothedDualMaster:
    """Smooth approximation of DualMasterCostFunction using Moreau-Yosida smoothing.

    Smooths the norm-type support functions with parameter epsilon, making
    the objective differentiable. Only supports :class:`BallSupportFunction` and
    :class:`EllipsoidSupportFunction`.

    The smoothed ball support is

    .. math::

        σ_{B,\\varepsilon}(z) = ⟨ z, c ⟩ + r\\,\\sqrt{‖z‖^2 + \\varepsilon^2}

    and its gradient w.r.t. $z$ is

    .. math::

        \\nabla_z σ_{B,\\varepsilon}(z) = c + r\\,\\frac{z}{\\sqrt{‖z‖^2 + \\varepsilon^2}}

    The smoothed ellipsoid support is

    .. math::

        σ_{E,\\varepsilon}(z) = ⟨ z, c ⟩
            + r\\,\\sqrt{⟨ z,\\, A^{-1}z ⟩ + \\varepsilon^2}

    and its gradient w.r.t. $z$ is

    .. math::

        \\nabla_z σ_{E,\\varepsilon}(z) = c
            + r\\,\\frac{A^{-1}z}{\\sqrt{⟨ z,\\, A^{-1}z ⟩ + \\varepsilon^2}}

    The full smoothed objective and its gradient are

    .. math::

        \\varphi_\\varepsilon(λ) = ⟨λ, \\tilde{d}⟩
            + σ_{B,\\varepsilon}(T^*q - G^*λ)
            + σ_{V,\\varepsilon}(-λ)

    .. math::

        \\nabla_λ \\varphi_\\varepsilon(λ)
            = \\tilde{d}
            - G\\,\\nabla_{z_1}σ_{B,\\varepsilon}(z_1)
            - \\nabla_{z_2}σ_{V,\\varepsilon}(z_2)

    where $z_1 = T^*q - G^*λ$ and $z_2 = -λ$.

    Args:
        cost: :class:`~pygeoinf.backus_gilbert.DualMasterCostFunction` instance.
        epsilon: Smoothing parameter ($> 0$). Smaller values give a better
            approximation but a larger Lipschitz constant
            $L = r‖G‖^2 / \\varepsilon$.

    Raises:
        NotImplementedError: If either support function is not a
            :class:`BallSupportFunction` or :class:`EllipsoidSupportFunction`.
    """

    def __init__(self, cost: "object", epsilon: float) -> None:
        self._cost = cost
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self._epsilon = float(epsilon)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _eval_support(
        self, z: "Vector", sigma: object
    ) -> "tuple[float, Vector]":
        """Dispatch to the appropriate smoothed value-and-gradient helper.

        Args:
            z: Argument vector in the primal domain of *sigma*.
            sigma: A support function (Ball or Ellipsoid).

        Returns:
            ``(value, grad)`` — smoothed scalar value and gradient w.r.t. *z*.

        Raises:
            NotImplementedError: For unsupported support function types.
        """
        if isinstance(sigma, BallSupportFunction):
            return self._smoothed_ball_value_and_grad(z, sigma)
        if isinstance(sigma, EllipsoidSupportFunction):
            return self._smoothed_ellipsoid_value_and_grad(z, sigma)
        raise NotImplementedError(
            f"SmoothedDualMaster only supports BallSupportFunction and "
            f"EllipsoidSupportFunction; got {type(sigma).__name__}."
        )

    def _smoothed_ball_value_and_grad(
        self, z: "Vector", sigma: "BallSupportFunction"
    ) -> "tuple[float, Vector]":
        """Smoothed value and gradient of a ball support function at *z*.

        For $σ_{B,\\varepsilon}(z) = ⟨ z, c ⟩
        + r\\sqrt{‖z‖^2 + \\varepsilon^2}$:

        .. math::

            \\nabla_z σ_{B,\\varepsilon}(z)
                = c + r\\,\\frac{z}{\\sqrt{‖z‖^2 + \\varepsilon^2}}

        Args:
            z: Argument vector in the primal domain of *sigma*.
            sigma: A :class:`BallSupportFunction`.

        Returns:
            ``(value, grad)`` — smoothed scalar value and gradient.
        """
        H = sigma.primal_domain
        c = sigma._center
        r = sigma._radius
        eps = self._epsilon

        center_term = H.inner_product(z, c)
        z_norm_sq = H.inner_product(z, z)
        denom = np.sqrt(z_norm_sq + eps * eps)

        value = center_term + r * denom
        # grad = c + (r / denom) * z
        grad = H.add(c, H.multiply(r / denom, z))
        return float(value), grad

    def _smoothed_ellipsoid_value_and_grad(
        self, z: "Vector", sigma: "EllipsoidSupportFunction"
    ) -> "tuple[float, Vector]":
        """Smoothed value and gradient of an ellipsoid support function at *z*.

        For $σ_{E,\\varepsilon}(z) = ⟨ z, c ⟩
        + r\\sqrt{⟨ z, A^{-1}z ⟩ + \\varepsilon^2}$:

        .. math::

            \\nabla_z σ_{E,\\varepsilon}(z)
                = c + r\\,\\frac{A^{-1}z}{\\sqrt{⟨ z, A^{-1}z ⟩ + \\varepsilon^2}}

        Args:
            z: Argument vector in the primal domain of *sigma*.
            sigma: A :class:`EllipsoidSupportFunction`.

        Returns:
            ``(value, grad)`` — smoothed scalar value and gradient.

        Raises:
            NotImplementedError: If ``sigma._A_inv`` is ``None``.
        """
        if sigma._A_inv is None:
            raise NotImplementedError(
                "EllipsoidSupportFunction requires inverse_operator to be set "
                "for use with SmoothedDualMaster."
            )
        H = sigma.primal_domain
        c = sigma._center
        r = sigma._radius
        eps = self._epsilon

        A_inv_z = sigma._A_inv(z)
        center_term = H.inner_product(z, c)
        inner_term = H.inner_product(z, A_inv_z)  # <z, A^{-1}z>
        denom = np.sqrt(max(inner_term, 0.0) + eps * eps)

        value = center_term + r * denom
        # grad = c + (r / denom) * A^{-1}z
        grad = H.add(c, H.multiply(r / denom, A_inv_z))
        return float(value), grad

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, lam: "Vector") -> float:
        """Evaluate the smoothed objective $\\varphi_\\varepsilon(λ)$.

        Args:
            lam: Dual variable $λ ∈ D$.

        Returns:
            Smoothed objective value (float).

        Raises:
            NotImplementedError: If either support function is unsupported.
        """
        cost = self._cost
        domain = cost.domain
        model_space = cost._model_space

        term1 = domain.inner_product(lam, cost._observed_data)

        Gstar_lam = cost._G.adjoint(lam)
        z1 = model_space.subtract(cost._Tstar_q, Gstar_lam)
        z2 = domain.negative(lam)

        term2, _ = self._eval_support(z1, cost._model_prior_support)
        term3, _ = self._eval_support(z2, cost._data_error_support)

        return term1 + term2 + term3

    def gradient(self, lam: "Vector") -> "Vector":
        """Compute the gradient $\\nabla_λ \\varphi_\\varepsilon(λ)$.

        Uses the chain rule through $z_1 = T^*q - G^*λ$ and
        $z_2 = -λ$:

        .. math::

            \\nabla_λ \\varphi_\\varepsilon
                = \\tilde{d}
                    - G\\,\\nabla_{z_1}σ_{B,\\varepsilon}(z_1)
                    - \\nabla_{z_2}σ_{V,\\varepsilon}(z_2)

        Args:
            lam: Dual variable $λ ∈ D$.

        Returns:
            Gradient vector in $D$.

        Raises:
            NotImplementedError: If either support function is unsupported.
        """
        cost = self._cost
        domain = cost.domain
        model_space = cost._model_space

        Gstar_lam = cost._G.adjoint(lam)
        z1 = model_space.subtract(cost._Tstar_q, Gstar_lam)
        z2 = domain.negative(lam)

        _, grad_z1 = self._eval_support(z1, cost._model_prior_support)
        _, grad_z2 = self._eval_support(z2, cost._data_error_support)

        # Contribution from σ_B(T*q - G*λ): chain rule gives -G * grad_z1
        term2_grad = domain.negative(cost._G(grad_z1))
        # Contribution from σ_V(-λ): chain rule gives -grad_z2
        term3_grad = domain.negative(grad_z2)

        return domain.add(
            cost._observed_data,
            domain.add(term2_grad, term3_grad),
        )


# ---------------------------------------------------------------------------
# Smoothed L-BFGS-B Solver
# ---------------------------------------------------------------------------


class SmoothedLBFGSSolver:
    """L-BFGS-B optimiser with smoothing continuation for DualMasterCostFunction.

    Uses Moreau-Yosida smoothing with a geometric continuation schedule:

        epsilon_0 >> epsilon_1 >> ... >> epsilon_{L-1} ≈ tol

    where epsilon_i = epsilon_0 × 10^{-i}. Each level is
    solved with L-BFGS-B, warm-starting from the previous solution.

    Note:
        Returns :class:`BundleResult` with ``gap=np.nan`` and
        ``f_low=np.nan`` — no gap certificate is available for smoothed methods.

    Args:
        cost: :class:`~pygeoinf.backus_gilbert.DualMasterCostFunction` instance.
        epsilon0: Initial smoothing parameter (default ``1e-2``).
        n_levels: Number of continuation levels (default ``5``).
        tolerance: Target accuracy; last epsilon is
            $\\varepsilon_0 × 10^{-(n\\_levels - 1)}$.
        max_iter_per_level: Maximum L-BFGS-B iterations per level
            (default ``500``).
    """

    def __init__(
        self,
        cost: "object",
        /,
        *,
        epsilon0: float = 1e-2,
        n_levels: int = 5,
        tolerance: float = 1e-6,
        max_iter_per_level: int = 500,
    ) -> None:
        self._oracle = cost
        self._epsilon0 = float(epsilon0)
        self._n_levels = int(n_levels)
        self._tolerance = float(tolerance)
        self._max_iter_per_level = int(max_iter_per_level)

    def solve(self, lam0: "Vector") -> BundleResult:
        """Run the smoothed L-BFGS-B continuation and return the result.

        Args:
            lam0: Starting point $λ_0 ∈ D$.

        Returns:
            :class:`BundleResult` with ``gap`` and ``f_low`` set to
            ``np.nan`` (no subgradient-based lower bound is maintained).
        """
        cost = self._oracle
        domain = cost.domain
        lam_current = lam0
        total_iters = 0
        function_values: List[float] = []

        eps_schedule = [
            self._epsilon0 * (10.0 ** (-i)) for i in range(self._n_levels)
        ]

        scipy_result = None  # will hold the last scipy OptimizeResult
        smoothed = None

        for eps in eps_schedule:
            smoothed = SmoothedDualMaster(cost, eps)
            x0_np = domain.to_components(lam_current)

            def f_and_g(x_np: np.ndarray, _sm=smoothed) -> "tuple[float, np.ndarray]":
                lam = domain.from_components(x_np)
                f = _sm(lam)
                g = _sm.gradient(lam)
                g_np = domain.to_components(g)
                return float(f), g_np

            scipy_result = minimize(
                f_and_g,
                x0_np,
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": self._max_iter_per_level,
                    "ftol": 1e-15,
                    "gtol": 1e-8,
                },
            )

            total_iters += scipy_result.nit
            function_values.append(float(scipy_result.fun))
            lam_current = domain.from_components(scipy_result.x)

        # Evaluate at the final iterate using the finest smoothing level
        f_best = float(smoothed(lam_current))

        return BundleResult(
            x_best=lam_current,
            f_best=f_best,
            f_low=float("nan"),
            gap=float("nan"),
            converged=scipy_result.success if scipy_result is not None else False,
            num_iterations=total_iters,
            num_serious_steps=0,
            function_values=function_values,
            iterates=None,
        )


# ---------------------------------------------------------------------------
# PrimalKKTSolver
# ---------------------------------------------------------------------------

class PrimalKKTSolver:
    r"""Primal KKT solver via Woodbury identity in abstract Hilbert spaces.

    The solver operates on abstract Hilbert-space vectors throughout: the
    model space $H$ is **never**
    discretised.  Only the data space $D$ (which is explicitly
    finite-dimensional) is discretised — and only inside
    :meth:`_woodbury_solve`, which builds the $M \times M$ Woodbury system.

    The Woodbury reduction:

    .. math::

        u^*(\lambda,\mu) =
            \frac{1}{\lambda} A_B^{-1} r_H
            - \frac{1}{\lambda} A_B^{-1} G^* K^{-1} G A_B^{-1} r_H

    where

    .. math::

        r_H = c + \lambda A_B u_0 + \mu G^* A_V \tilde{d}, \qquad
        K   = \tfrac{1}{\mu} A_V^{-1} + \tfrac{1}{\lambda} G A_B^{-1} G^*

    The $M \times M$ matrix $P = G A_B^{-1} G^*$ is precomputed once via
    ``.matrix(dense=True)`` which probes the **data space** only.  No
    model-space ``to_components`` or ``from_components`` is ever called.

    **Ball/ball simplification** ($A_B = I_H$, $A_V = I_D$):

    .. math::

        r_H = c + \lambda u_0 + \mu G^* \tilde{d}, \qquad
        P   = G G^*, \qquad
        K   = \tfrac{1}{\mu} I_D + \tfrac{1}{\lambda} G G^*

    so the only M×M solve is $K z = G w$ and no model-space matrix is
    ever formed.

    Supports :class:`~pygeoinf.convex_analysis.BallSupportFunction` and
    :class:`~pygeoinf.convex_analysis.EllipsoidSupportFunction` for both $B$
    and $V$ (four combinations).

    Args:
        B: Model prior support function.
        V: Data error support function (ball or ellipsoid, centered at origin).
        G: Forward operator $G : H \to D$.
        d_tilde: Observed data vector in $D$.
        fsolve_tol: Tolerance for ``scipy.optimize.fsolve``.
        fsolve_maxfev: Maximum function evaluations for ``fsolve``.
    """

    def __init__(
        self,
        B: "SupportFunction",
        V: "SupportFunction",
        G: "LinearOperator",
        d_tilde: "Vector",
        /,
        *,
        fsolve_tol: float = 1e-10,
        fsolve_maxfev: int = 200,
    ) -> None:
        # --- Validate B ---
        if isinstance(B, BallSupportFunction):
            self._prior_is_ball = True
            self._A_B_op = B.primal_domain.identity_operator()
            self._A_B_inv_op = B.primal_domain.identity_operator()
        elif isinstance(B, EllipsoidSupportFunction):
            if B._A_inv is None:
                raise ValueError(
                    "EllipsoidSupportFunction B must supply inverse_operator "
                    "(A^{-1}) to be used as a PrimalKKTSolver prior."
                )
            self._prior_is_ball = False
            self._A_B_op = B._A
            self._A_B_inv_op = B._A_inv
        else:
            raise TypeError(
                f"B must be BallSupportFunction or EllipsoidSupportFunction, "
                f"got {type(B).__name__}"
            )

        # --- Validate V ---
        if isinstance(V, PointSupportFunction):
            # Equality data constraint: G u = d̃  (V = {0}, r = 0)
            self._equality_data = True
            self._data_is_ball = True   # A_V = I (unused in equality path)
            self._A_V_op = V.primal_domain.identity_operator()
        elif isinstance(V, BallSupportFunction):
            self._equality_data = False
            self._data_is_ball = True
            self._A_V_op = V.primal_domain.identity_operator()
        elif isinstance(V, EllipsoidSupportFunction):
            if V._A_inv is None:
                raise ValueError(
                    "EllipsoidSupportFunction V must supply inverse_operator "
                    "(A^{-1}) to be used as a PrimalKKTSolver data set."
                )
            self._equality_data = False
            self._data_is_ball = False
            self._A_V_op = V._A
        else:
            raise TypeError(
                f"V must be BallSupportFunction, EllipsoidSupportFunction, or "
                f"PointSupportFunction, got {type(V).__name__}"
            )

        self._B = B
        self._V = V
        self._G = G
        self._d_tilde = d_tilde
        self._fsolve_tol = fsolve_tol
        self._fsolve_maxfev = fsolve_maxfev

        self._model_space = B.primal_domain
        self._data_space = G.codomain
        M = self._data_space.dim
        self._M = M
        self._eta = float(B._radius)
        self._r = 0.0 if self._equality_data else float(V._radius)
        self._u0 = B._center

        # --- G_adj_AV operator (G.adjoint ∘ A_V : D → H) ---
        # For ball/point V: A_V = I_D, so G_adj_AV = G.adjoint
        # For ellipsoid V: G_adj_AV = G.adjoint @ V._A
        if self._data_is_ball:
            self._G_adj_AV = G.adjoint
        else:
            self._G_adj_AV = G.adjoint @ V._A

        # --- Precomputed H-vectors (used in inequality path) ---
        self._A_B_u0 = self._A_B_op(self._u0)       # H-vector
        self._G_adj_AV_d = self._G_adj_AV(d_tilde)  # H-vector

        # --- A_V^{-1} matrix (M×M, unused in equality path) ---
        if self._data_is_ball:
            self._AV_inv_mat = np.eye(M)
        else:
            self._AV_inv_mat = V._A_inv.matrix(dense=True)

        # --- P_mat = G A_B^{-1} G* (M×M, data space probing only!) ---
        if self._prior_is_ball:
            try:
                self._P_mat = self._ball_prior_gram_matrix(G)
            except (AttributeError, TypeError, ValueError, NotImplementedError):
                self._P_mat = (G @ G.adjoint).matrix(dense=True)
        else:
            self._P_mat = (G @ B._A_inv @ G.adjoint).matrix(dense=True)

        self._P_eigvals: np.ndarray | None = None
        self._P_eigvecs: np.ndarray | None = None
        if self._prior_is_ball and self._data_is_ball and not self._equality_data:
            self._precompute_ball_ball_spectrum()

        # --- Equality-constraint precomputation ---
        if self._equality_data:
            self._precompute_equality_data()

        # --- Warm-start state (inequality path only) ---
        self._lambda_prev: float = 1.0
        self._mu_prev: float = 0.0
        self._has_warm_start: bool = False
        self._warm_start_disabled: bool = False
        # Property operator T, set via from_cost for use in solve_support_values
        self._T: "LinearOperator | None" = None

    @staticmethod
    def _matmul_with_optional_setup_threads(
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        setup_threads = int(os.environ.get("PYGEOINF_KKT_SETUP_THREADS", "0"))
        if setup_threads <= 0:
            return left @ right

        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            return left @ right

        with threadpool_limits(limits=setup_threads, user_api="blas"):
            return left @ right

    @staticmethod
    def _eigh_with_optional_setup_threads(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        setup_threads = int(os.environ.get("PYGEOINF_KKT_SETUP_THREADS", "0"))
        if setup_threads <= 0:
            return np.linalg.eigh(matrix)

        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            return np.linalg.eigh(matrix)

        with threadpool_limits(limits=setup_threads, user_api="blas"):
            return np.linalg.eigh(matrix)

    @staticmethod
    def _ball_prior_gram_matrix(G: "LinearOperator") -> np.ndarray:
        r"""Build $P = G G^*$ for a Hilbert-ball prior.

        If the data space is Euclidean and the model space exposes a diagonal
        inverse mass operator, the adjoint matrix is
        $R_H^{-1} G^T$.  In that common Sobolev/mass-weighted case we can form
        $G R_H^{-1} G^T$ directly from the stored forward matrix and avoid
        materializing ``G.adjoint.matrix(dense=True)`` column by column.
        """
        G_mat = np.asarray(G.matrix(dense=True), dtype=float)

        try:
            from .hilbert_space import EuclideanSpace

            inverse_mass_operator = G.domain.inverse_mass_operator
            inverse_mass_diagonal = np.asarray(
                inverse_mass_operator.extract_diagonal(galerkin=False),
                dtype=float,
            )
            data_is_euclidean = isinstance(G.codomain, EuclideanSpace)
        except (AttributeError, TypeError, ValueError, NotImplementedError):
            data_is_euclidean = False
            inverse_mass_diagonal = None

        if (
            data_is_euclidean
            and inverse_mass_diagonal is not None
            and inverse_mass_diagonal.shape == (G.domain.dim,)
        ):
            weighted_G = G_mat * inverse_mass_diagonal[np.newaxis, :]
            return PrimalKKTSolver._matmul_with_optional_setup_threads(
                weighted_G,
                G_mat.T,
            )

        G_adj_mat = np.asarray(G.adjoint.matrix(dense=True), dtype=float)
        return PrimalKKTSolver._matmul_with_optional_setup_threads(
            G_mat,
            G_adj_mat,
        )

    def _precompute_ball_ball_spectrum(self) -> None:
        P_sym = 0.5 * (self._P_mat + self._P_mat.T)
        eigvals, eigvecs = self._eigh_with_optional_setup_threads(P_sym)
        self._P_eigvals = np.maximum(eigvals, 0.0)
        self._P_eigvecs = eigvecs

    @classmethod
    def from_cost(cls, cost, /, *, fsolve_tol: float = 1e-10, fsolve_maxfev: int = 200):
        """Construct a PrimalKKTSolver from a DualMasterCostFunction.

        Extracts the model prior B, data error V, forward operator G, and
        observed data d̃ from the cost function and stores the property
        operator T so that :func:`solve_support_values` can convert
        property-space directions q to model-space directions c = T*q.

        Args:
            cost: A :class:`~pygeoinf.backus_gilbert.DualMasterCostFunction`.
            fsolve_tol: Tolerance for ``scipy.optimize.fsolve``.
            fsolve_maxfev: Maximum function evaluations for ``fsolve``.

        Returns:
            A :class:`PrimalKKTSolver` instance with T stored internally.

        Raises:
            TypeError: If ``cost`` is not a DualMasterCostFunction.
            ValueError: If the cost's prior or data sets are not
                BallSupportFunction or EllipsoidSupportFunction.
        """
        if type(cost).__name__ != "DualMasterCostFunction":
            raise TypeError(
                f"cost must be a DualMasterCostFunction, got {type(cost).__name__}"
            )
        instance = cls(
            cost._model_prior_support,
            cost._data_error_support,
            cost._G,
            cost._observed_data,
            fsolve_tol=fsolve_tol,
            fsolve_maxfev=fsolve_maxfev,
        )
        instance._T = cost._T
        return instance

    def disable_warm_start(self) -> None:
        """Reset warm-start state to ensure fresh solves from default initial guesses.

        Useful for benchmarking or when reproducibility across sequential/parallel
        is required. After calling this, each solve() starts with default initial
        guesses (lam_phys, 1e-3) rather than previous solution. Warm-start is
        permanently disabled until a new solver is created.
        """
        self._lambda_prev = 1.0
        self._mu_prev = 0.0
        self._has_warm_start = False
        self._warm_start_disabled = True

    def _precompute_equality_data(self) -> None:
        r"""Precompute the data-mismatch correction $a_H$ for the equality path.

        With $V = \{0\}$ the data constraint is $G u = \tilde{d}$
        (equality).  Any feasible $u$ can be written as

        .. math::

            u = u_0 + a_H + n_H / \lambda

        where $a_H$ is the minimum-$A_B$-norm correction that shifts $u_0$
        onto the hyperplane $\{G u = \tilde{d}\}$:

        .. math::

            a_H = A_B^{-1} G^* P^{-1}(\tilde{d} - G u_0),
            \qquad P = G A_B^{-1} G^*.

        and $n_H / \lambda$ is the direction that maximises $\langle c, u \rangle$
        subject to the prior constraint (computed per solve in
        :meth:`_solve_equality`).
        """
        from scipy.linalg import cho_factor, cho_solve

        ms = self._model_space
        ds = self._data_space

        Gu0_D = self._G(self._u0)
        rhs_D = ds.subtract(self._d_tilde, Gu0_D)
        rhs_comps = np.asarray(ds.to_components(rhs_D), dtype=float)

        cho_P = cho_factor(self._P_mat + 1e-14 * np.eye(self._M))
        z_comps = cho_solve(cho_P, rhs_comps)
        z_D = ds.from_components(z_comps)

        Gstar_z = self._G.adjoint(z_D)
        self._a_H = self._A_B_inv_op(Gstar_z)
        self._a_H_sq = float(ms.inner_product(self._A_B_op(self._a_H), self._a_H))

    def _solve_equality(self, c: "Vector") -> "KKTResult":
        r"""Solve the equality-constrained support problem analytically.

        Computes

        .. math::

            h_U(q) = \sup_{u}\ \langle c,\, u \rangle
            \quad \text{s.t.} \quad
            \|u - u_0\|_B \le \eta,\quad G u = \tilde{d}

        in closed form via a 1-D quadratic in $t = 1/\lambda$.

        **Decomposition.**  Every feasible $u$ satisfies

        .. math::

            u - u_0 = \frac{n_H}{\lambda} + a_H

        where $a_H = A_B^{-1} G^* P^{-1}(\tilde{d} - G u_0)$ (precomputed)
        and $n_H = A_B^{-1}(c - G^* P^{-1} G A_B^{-1} c)$ is the
        null-space projection of $c$ (computed here).  Substituting into
        the prior constraint $\|u - u_0\|_{A_B}^2 = \eta^2$ yields

        .. math::

            \|n_H\|_{A_B}^2\, t^2
            + 2\langle n_H,\, a_H\rangle_{A_B}\, t
            + \bigl(\|a_H\|_{A_B}^2 - \eta^2\bigr) = 0,
            \quad t = 1/\lambda.

        The larger positive root $t^*$ maximises $\langle c, u \rangle$
        because the objective is $\langle c, u_0\rangle
        + \|n_H\|_{A_B}^2 \cdot t + \langle c, a_H\rangle$
        and $\|n_H\|_{A_B}^2 \ge 0$.

        Returns:
            :class:`KKTResult` with ``converged=True`` when the quadratic
            has a non-negative discriminant.  ``multipliers[1]`` is
            ``float("inf")`` to signal that the data multiplier is not a
            scalar (it is an $M$-dimensional vector in the equality case).
        """
        from scipy.linalg import cho_factor, cho_solve

        ms = self._model_space
        ds = self._data_space
        EPS = 1e-12

        # n_H = A_B⁻¹(c - G* P⁻¹ G A_B⁻¹ c)  [null-space component of c]
        w_H = self._A_B_inv_op(c)
        p_comps = np.asarray(ds.to_components(self._G(w_H)), dtype=float)
        cho_P = cho_factor(self._P_mat + 1e-14 * np.eye(self._M))
        q_comps = cho_solve(cho_P, p_comps)
        q_D = ds.from_components(q_comps)
        n_H = ms.subtract(w_H, self._A_B_inv_op(self._G.adjoint(q_D)))

        n_sq  = float(ms.inner_product(self._A_B_op(n_H), n_H))
        dot_na = float(ms.inner_product(self._A_B_op(n_H), self._a_H))
        a_sq  = self._a_H_sq

        # Prior-inactive branch: c entirely in range(G*) → u uniquely determined
        if n_sq < EPS * max(a_sq, 1.0):
            u_star = ms.add(self._u0, self._a_H)
            feasible = a_sq <= self._eta ** 2 * (1.0 + EPS)
            return KKTResult(
                m=u_star,
                multipliers=(0.0, float("inf")),
                converged=feasible,
                num_iterations=1,
            )

        # Quadratic: n_sq * t² + 2*dot_na * t + (a_sq - η²) = 0
        disc = dot_na ** 2 - n_sq * (a_sq - self._eta ** 2)
        if disc < 0.0:
            # Numerically infeasible (data and prior incompatible)
            u_star = ms.add(self._u0, self._a_H)
            return KKTResult(
                m=u_star,
                multipliers=(0.0, float("inf")),
                converged=False,
                num_iterations=1,
            )

        t_star = (-dot_na + np.sqrt(disc)) / n_sq
        lam_star = 1.0 / t_star if t_star > EPS else float("inf")
        u_star = ms.add(
            self._u0,
            ms.add(ms.multiply(t_star, n_H), self._a_H),
        )
        return KKTResult(
            m=u_star,
            multipliers=(lam_star, float("inf")),
            converged=True,
            num_iterations=1,
        )

    def _woodbury_solve(self, lam: float, mu: float, c: "Vector") -> "Vector":
        r"""Return $u^*(\lambda, \mu)$ entirely in abstract Hilbert-space vectors.

        Solves

        .. math::

            (\lambda A_B + \mu G^* A_V G)\, u^* =
                c + \lambda A_B u_0 + \mu G^* A_V \tilde{d}

        using the Woodbury identity.  The only data-space discretisation is the
        $M \times M$ Cholesky solve for $K z = G w$.

        Args:
            lam: KKT multiplier $\lambda > 0$.
            mu:  KKT multiplier $\mu \geq 0$.
            c:   Objective direction (primal $H$-vector).

        Returns:
            $u^*$ as an abstract $H$-vector.
        """
        from scipy.linalg import cho_factor, cho_solve

        ms = self._model_space
        ds = self._data_space

        # rhs = c + λ A_B(u0) + μ G_adj_AV(d̃)   [abstract H]
        rhs_H = ms.add(
            c,
            ms.add(
                ms.multiply(lam, self._A_B_u0),
                ms.multiply(mu, self._G_adj_AV_d),
            ),
        )

        # w = (1/λ) A_B_inv(rhs)   [abstract H]
        w_H = ms.multiply(1.0 / lam, self._A_B_inv_op(rhs_H))

        if mu == 0.0:
            return w_H

        # p = G(w)   [data-space vector]
        p_D = self._G(w_H)
        p_comps = ds.to_components(p_D)

        if self._P_eigvals is not None and self._P_eigvecs is not None:
            qt_p = self._P_eigvecs.T @ p_comps
            denom = (1.0 / mu) + (1.0 / lam) * self._P_eigvals
            z_comps = self._P_eigvecs @ (qt_p / denom)
        else:
            # K = (1/μ) A_V^{-1} + (1/λ) P_mat   [M×M]
            K_mat = (1.0 / mu) * self._AV_inv_mat + (1.0 / lam) * self._P_mat
            cho_decomp = cho_factor(K_mat)
            z_comps = cho_solve(cho_decomp, p_comps)
        z_D = ds.from_components(z_comps)

        # correction = (1/λ) A_B_inv(G*(z))   [abstract H]
        # NOTE: uses G.adjoint (plain adjoint), NOT G_adj_AV
        correction_H = ms.multiply(
            1.0 / lam, self._A_B_inv_op(self._G.adjoint(z_D))
        )

        return ms.subtract(w_H, correction_H)

    def _residuals(
        self, lam: float, mu: float, c: "Vector"
    ) -> "tuple[float, float]":
        r"""KKT residuals $(\rho_1, \rho_2)$ at $(\lambda, \mu)$.

        .. math::

            \rho_1 = \langle A_B(u^* - u_0),\, u^* - u_0 \rangle_H - \eta^2

            \rho_2 = \langle A_V(G u^* - \tilde{d}),\, G u^* - \tilde{d} \rangle_D
                     - r^2

        Args:
            lam: $\lambda > 0$.
            mu:  $\mu \geq 0$.
            c:   Objective direction.

        Returns:
            ``(rho1, rho2)`` floats.
        """
        ms = self._model_space
        ds = self._data_space

        u = self._woodbury_solve(lam, mu, c)

        diff_u = ms.subtract(u, self._u0)
        # For ball B: A_B_op = identity, so <diff_u, diff_u>_H = ||u* - u0||_H^2
        # For ellipsoid B: <A_B(diff_u), diff_u>_H
        rho1 = float(ms.inner_product(self._A_B_op(diff_u), diff_u)) - self._eta ** 2

        res_d = ds.subtract(self._G(u), self._d_tilde)
        rho2 = float(ds.inner_product(self._A_V_op(res_d), res_d)) - self._r ** 2

        return rho1, rho2

    def solve(self, c: "Vector") -> KKTResult:
        r"""Solve for $u^*$ maximising $\langle c, u \rangle$ over the feasible set.

        Uses a two-branch strategy:

        1. Compute the support point $u_{\rm ball}$ of $B$ in direction $c$.
           If the data constraint is satisfied, return immediately.
        2. Otherwise both constraints are active; solve the $2 \times 2$
           root-finding problem in log-space via ``scipy.optimize.fsolve``,
           warm-started from $(\lambda_{\rm prev}, \mu_{\rm prev})$.

        The model space is **never** discretised during this method.

        Args:
            c: Linear objective in model space $H$.

        Returns:
            :class:`KKTResult` with the optimal model vector $u^*$,
            KKT multipliers, convergence flag, and iteration count.
        """
        # ------------------------------------------------------------------
        # Equality-data branch (V = {0}): closed-form 1-D quadratic solve
        # ------------------------------------------------------------------
        if self._equality_data:
            return self._solve_equality(c)

        from scipy.optimize import fsolve

        ms = self._model_space
        ds = self._data_space

        # ------------------------------------------------------------------
        # Branch 1: prior-only (data constraint not active)
        # ------------------------------------------------------------------
        _, u_ball = self._B.value_and_support_point(c)
        if u_ball is None:
            u_ball = self._B.support_point(c)

        if u_ball is not None:
            res_data = ds.subtract(self._G(u_ball), self._d_tilde)
            data_sq = float(ds.inner_product(self._A_V_op(res_data), res_data))

            if data_sq <= self._r ** 2 * (1.0 + 1e-9):
                # Compute λ* from the active prior constraint
                if self._prior_is_ball:
                    c_norm = float(ms.norm(c))
                    lam_star = c_norm / self._eta if c_norm > 1e-14 else 1.0
                else:
                    c_norm_AB = float(
                        np.sqrt(max(float(ms.inner_product(self._B._A_inv(c), c)), 0.0))
                    )
                    lam_star = c_norm_AB / self._eta if c_norm_AB > 1e-14 else 1.0
                return KKTResult(
                    m=u_ball,
                    multipliers=(lam_star, 0.0),
                    converged=True,
                    num_iterations=1,
                )

        # ------------------------------------------------------------------
        # Branch 2: both constraints active — 2D fsolve in log space
        # ------------------------------------------------------------------
        if self._prior_is_ball:
            c_norm = float(ms.norm(c))
            lam_phys = max(c_norm / self._eta, 1e-4)
        else:
            c_norm_AB = float(
                np.sqrt(max(float(ms.inner_product(self._B._A_inv(c), c)), 0.0))
            )
            lam_phys = max(c_norm_AB / self._eta, 1e-4)

        if self._has_warm_start and not self._warm_start_disabled and self._lambda_prev > 1e-4:
            lam0 = self._lambda_prev
        else:
            lam0 = lam_phys

        if self._has_warm_start and not self._warm_start_disabled and self._mu_prev > 1e-8:
            mu0 = self._mu_prev
        else:
            mu0 = 1e-3

        def _residual_log(log_x: np.ndarray) -> np.ndarray:
            log_x_safe = np.clip(log_x, -30.0, 25.0)
            lam = float(np.exp(log_x_safe[0]))
            mu = float(np.exp(log_x_safe[1]))
            r1, r2 = self._residuals(lam, mu, c)
            return np.array([r1, r2])

        x0_log = np.array([np.log(lam0), np.log(mu0)])
        sol, info, ier, _mesg = fsolve(
            _residual_log,
            x0_log,
            full_output=True,
            xtol=self._fsolve_tol,
            maxfev=self._fsolve_maxfev,
        )

        if ier != 1:
            fallback_guesses = [
                (lam_phys, 1e-3),
                (lam_phys, 1e-2),
                (lam_phys, 0.5),
                (lam_phys, 2.0),
                (0.5 * lam_phys, 1e-2),
                (0.5 * lam_phys, 1.0),
            ]
            for alt_lam, alt_mu in fallback_guesses:
                alt_x0 = np.array([
                    np.log(max(alt_lam, 1e-4)),
                    np.log(max(alt_mu, 1e-8)),
                ])
                alt_sol, alt_info, alt_ier, _ = fsolve(
                    _residual_log,
                    alt_x0,
                    full_output=True,
                    xtol=self._fsolve_tol,
                    maxfev=self._fsolve_maxfev,
                )
                if alt_ier == 1:
                    sol, info, ier = alt_sol, alt_info, alt_ier
                    break

        if ier != 1:
            from scipy.optimize import least_squares

            prior_residual_scale = max(self._eta ** 2, 1e-30)
            data_residual_scale = max(self._r ** 2, 1e-30)

            def _residual_log_scaled(log_x: np.ndarray) -> np.ndarray:
                log_x_safe = np.clip(log_x, -30.0, 25.0)
                lam = float(np.exp(log_x_safe[0]))
                mu = float(np.exp(log_x_safe[1]))
                r1, r2 = self._residuals(lam, mu, c)
                return np.array([r1 / prior_residual_scale, r2 / data_residual_scale])

            ls_result = least_squares(
                _residual_log_scaled,
                np.clip(sol, -30.0, 25.0),
                bounds=([-30.0, -30.0], [25.0, 25.0]),
                xtol=self._fsolve_tol,
                ftol=self._fsolve_tol,
                gtol=self._fsolve_tol,
                max_nfev=self._fsolve_maxfev,
            )
            scaled_residual = _residual_log_scaled(ls_result.x)
            if ls_result.success and np.linalg.norm(scaled_residual, ord=np.inf) <= 1e-6:
                sol = ls_result.x
                info = {"nfev": ls_result.nfev}
                ier = 1

        sol_safe = np.clip(sol, -30.0, 25.0)
        lam_sol = float(np.exp(sol_safe[0]))
        mu_sol = float(np.exp(sol_safe[1]))
        converged = ier == 1

        u_star = self._woodbury_solve(lam_sol, mu_sol, c)

        if not self._warm_start_disabled:
            self._lambda_prev = lam_sol
            self._mu_prev = mu_sol
            self._has_warm_start = converged

        return KKTResult(
            m=u_star,
            multipliers=(lam_sol, mu_sol),
            converged=converged,
            num_iterations=int(info["nfev"]),
        )
