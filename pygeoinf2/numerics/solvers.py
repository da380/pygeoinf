"""
Linear solvers.

A solver turns an operator into its inverse, *as an operator*: the one piece of
v1's solver API that is straightforwardly right, and kept. What changes:

- **Preconditions are declared and checked.** A solver states the traits it
  needs and whether it needs coordinates, and ``__call__`` validates before
  doing any work. v1 writes ``assert operator.is_automorphism``, which vanishes
  under ``python -O`` and says nothing about symmetry or definiteness.

- **The iterative solvers are coordinate-free.** CG, MINRES, BiCGStab and LSQR
  are written against ``inner_product`` and ``axpy`` alone, so they run against
  a space with no component map at all. Only the direct solvers set
  ``requires_coordinates``.

- **Solvers are stateless.** v1 stores the iteration count on the solver
  object, so a solver cannot be shared and the count belongs to whichever solve
  ran last. Here diagnostics come back from ``InverseOperator.solve``, which
  returns a :class:`SolveResult`.

- **Non-square is a different operation.** A least-squares solver is a sibling
  of ``LinearSolver``, not a subclass whose ``__call__`` asserts squareness.

See DESIGN.md section 6.
"""

from __future__ import annotations

import copy

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, lu_factor, lu_solve

from ..algebra.operators import LinearOperator, require_coordinates
from ..algebra.spaces import CoordinateSpace, HilbertSpace
from ..traits import Traits, inverse_traits

__all__ = [
    "ConvergenceError",
    "SolveResult",
    "LinearSolver",
    "InverseOperator",
    "DirectSolver",
    "LUSolver",
    "CholeskySolver",
    "EigenSolver",
    "IterativeSolver",
    "CGSolver",
    "MinResSolver",
    "BiCGStabSolver",
    "LeastSquaresSolver",
    "LSQRSolver",
]


class ConvergenceError(RuntimeError):
    """An iterative solve failed to reach its tolerance.

    Raised rather than warned by default: an unconverged solve is a wrong
    answer, and returning one silently is how a wrong answer propagates into a
    published figure. Pass ``strict=False`` to downgrade it.
    """


@dataclass(frozen=True)
class SolveResult[X]:
    """The outcome of one solve, including its diagnostics."""

    solution: X
    iterations: int
    residual_norm: float
    converged: bool
    history: tuple[float, ...] = ()

    def __repr__(self) -> str:
        return (
            f"SolveResult(iterations={self.iterations}, "
            f"residual_norm={self.residual_norm:.3g}, "
            f"converged={self.converged})"
        )


class LinearSolver(ABC):
    """Turns an operator into its inverse.

    Subclasses declare what they need. ``requires`` is a set of traits the
    operator must claim; ``requires_coordinates`` says whether the domain and
    codomain must provide a component map.
    """

    requires: ClassVar[Traits] = Traits.NONE
    requires_coordinates: ClassVar[bool] = False

    def __call__(self, operator: LinearOperator) -> InverseOperator:
        """The inverse of ``operator``, as an operator."""
        self._validate(operator)
        return self._invert(operator)

    def _validate(self, operator: LinearOperator) -> None:
        if operator.domain.dim != operator.codomain.dim:
            raise ValueError(
                f"{type(self).__name__} inverts square operators; this one maps "
                f"a space of dimension {operator.domain.dim} to one of "
                f"dimension {operator.codomain.dim}. For a rectangular system "
                f"use a LeastSquaresSolver."
            )
        missing = self.requires & ~operator.traits
        if missing:
            raise ValueError(
                f"{type(self).__name__} requires {self.requires!s}, but the "
                f"operator claims only {operator.traits!s} (missing "
                f"{missing!s}). Traits are claims: attach them with "
                f"with_traits() and verify them with testing.check_traits()."
            )
        if self.requires_coordinates:
            require_coordinates(operator.domain, operator.codomain)

    @abstractmethod
    def _invert(self, operator: LinearOperator) -> InverseOperator:
        """Build the inverse, having passed validation."""


class InverseOperator[X, Y](LinearOperator[Y, X]):
    """The inverse of an operator, produced by a solver.

    Applying it solves a system. ``solve`` does the same but returns the
    diagnostics with the answer, which is where iteration counts live now that
    solvers are stateless.
    """

    def __init__(
        self,
        operator: LinearOperator[X, Y],
        solver: LinearSolver,
        solve_fn: Callable[[Y, X | None], SolveResult[X]],
        /,
        *,
        traits: Traits | None = None,
    ) -> None:
        if traits is None:
            # A pseudo-inverse of a rectangular operator is not an inverse, and
            # must not inherit the claim that it is one.
            traits = (
                inverse_traits(operator.traits)
                if operator.domain.dim == operator.codomain.dim
                else Traits.NONE
            )
        super().__init__(operator.codomain, operator.domain, traits=traits)
        self._operator = operator
        self._solver = solver
        self._solve_fn = solve_fn

    @property
    def operator(self) -> LinearOperator[X, Y]:
        """The operator being inverted."""
        return self._operator

    @property
    def solver(self) -> LinearSolver:
        """The solver that produced this inverse."""
        return self._solver

    def solve(self, y: Y, /, *, x0: X | None = None) -> SolveResult[X]:
        """Solve ``A x == y``, returning the solution and its diagnostics."""
        return self._solve_fn(y, x0)

    def _value(self, y: Y) -> X:
        return self._solve_fn(y, None).solution

    def _adjoint_value(self, x: X) -> Y:
        return self.adjoint_inverse(x)

    def adjoint_inverse(self, x: X) -> Y:
        """Apply ``(A^-1)* == (A*)^-1``."""
        cached = self.__dict__.get("_adjoint_inverse_op")
        if cached is None:
            cached = self._solver(self._operator.adjoint)
            self.__dict__["_adjoint_inverse_op"] = cached
        return cached(x)

    def _make_adjoint(self) -> LinearOperator[X, Y]:
        """``(A^-1)* == (A*)^-1``, built as an inverse rather than a wrapper."""
        result = self._solver(self._operator.adjoint)
        # Close the loop, for the same reason the sum and composition nodes do.
        result.__dict__["_adjoint_cache"] = self
        return result

    def __repr__(self) -> str:
        return f"Inverse({type(self._solver).__name__}, {self._operator!r})"


# --------------------------------------------------------------------- #
#                            Direct solvers                             #
# --------------------------------------------------------------------- #


class DirectSolver(LinearSolver):
    """A solver that factorises a matrix representation.

    These need coordinates, and they need the *right* representation. Which one
    is right follows from the operator's traits: self-adjointness shows up as
    matrix symmetry only in the Galerkin form, so a symmetric factorisation
    must be handed that one. v1 threads a ``galerkin=`` flag by hand through
    every call site instead.
    """

    requires_coordinates: ClassVar[bool] = True
    form: ClassVar[str] = "components"

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        domain: CoordinateSpace = operator.domain
        codomain: CoordinateSpace = operator.codomain
        matrix = operator.matrix(form=self.form)
        apply_inverse = self._factorise(matrix)

        def solve_fn(y, x0):
            cy = codomain.to_components(y)
            if self.form == "galerkin":
                # M c_x == G_Y c_y, since M == G_Y A_c.
                cy = codomain.apply_gram(cy)
            cx = apply_inverse(cy)
            return SolveResult(domain.from_components(cx), 0, 0.0, True)

        return InverseOperator(operator, self, solve_fn)

    @abstractmethod
    def _factorise(self, matrix: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """Factorise once, returning something that applies the inverse."""


class LUSolver(DirectSolver):
    """LU factorisation. Makes no structural demands beyond squareness."""

    form: ClassVar[str] = "components"

    def _factorise(self, matrix: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        factor = lu_factor(matrix)
        return lambda c: lu_solve(factor, c)


class CholeskySolver(DirectSolver):
    """Cholesky factorisation of the Galerkin matrix.

    Requires positive-definiteness, which by closure implies self-adjointness —
    and the Galerkin form is what makes that visible as symmetry.
    """

    requires: ClassVar[Traits] = Traits.POSITIVE_DEFINITE
    form: ClassVar[str] = "galerkin"

    def _factorise(self, matrix: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        symmetric = 0.5 * (matrix + matrix.T)
        factor = cho_factor(symmetric)
        return lambda c: cho_solve(factor, c)


class EigenSolver(DirectSolver):
    """Symmetric eigendecomposition, with a pseudo-inverse for small modes."""

    requires: ClassVar[Traits] = Traits.SELF_ADJOINT
    form: ClassVar[str] = "galerkin"

    def __init__(self, /, *, rtol: float = 1e-12) -> None:
        self._rtol = rtol

    def _factorise(self, matrix: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        symmetric = 0.5 * (matrix + matrix.T)
        values, vectors = eigh(symmetric)
        largest = np.max(np.abs(values)) if values.size else 0.0
        threshold = self._rtol * largest
        inverted = np.where(np.abs(values) > threshold, 1.0 / values, 0.0)
        return lambda c: vectors @ (inverted * (vectors.T @ c))


# --------------------------------------------------------------------- #
#                          Iterative solvers                            #
# --------------------------------------------------------------------- #


class IterativeSolver(LinearSolver):
    """A Krylov solver, written against the space's own inner product.

    Every arithmetic operation here goes through ``inner_product``, ``axpy``
    and ``scale_inplace``, so these run unchanged against a space with no
    coordinate map. That is not incidental: measuring the residual in the
    space's norm rather than in a component norm is what makes the stopping
    criterion mean the same thing under refinement.
    """

    requires_coordinates: ClassVar[bool] = False

    def __init__(
        self,
        /,
        *,
        rtol: float = 1e-10,
        atol: float = 0.0,
        maxiter: int | None = None,
        preconditioner: LinearSolver | LinearOperator | None = None,
        strict: bool = True,
        callback: Callable[[int, float], None] | None = None,
    ) -> None:
        """
        Args:
            rtol: stop when the residual norm falls below ``rtol * ||b||``.
            atol: absolute floor on the same test.
            maxiter: iteration cap; defaults to the dimension of the space.
            preconditioner: either a ready-made approximate inverse, or a
                ``LinearSolver`` from which to build one once the operator is
                known.
            strict: raise :class:`ConvergenceError` on failure to converge
                rather than warning.
            callback: called with ``(iteration, residual_norm)`` after each
                step. For watching a long solve, and for finding out *where* a
                stalled one stalled — which the final residual alone cannot
                say.
        """
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter
        self._preconditioner = preconditioner
        self._strict = strict
        self._callback = callback

    def with_preconditioner(
        self, preconditioner: LinearSolver | LinearOperator, /
    ) -> "IterativeSolver":
        """The same solver, with a preconditioner attached.

        Exists so that a caller can hand in a configured solver — tolerances,
        iteration cap, callbacks — and a library routine can still supply the
        preconditioner it knows how to build. An explicit preconditioner on the
        original is kept; the caller knows something the routine does not.
        """
        if self._preconditioner is not None:
            return self
        clone = copy.copy(self)
        clone._preconditioner = preconditioner
        return clone

    @property
    def preconditioner(self) -> "LinearSolver | LinearOperator | None":
        """What this solver preconditions with, if anything.

        A :class:`LinearSolver` here is *deferred*: it is applied to whatever
        operator the solver is asked to invert, so it is rebuilt for each one.
        A :class:`LinearOperator` is already resolved and is reused as it is.
        """
        return self._preconditioner

    def resolved_for(self, operator: LinearOperator, /) -> "IterativeSolver":
        """The same solver with a deferred preconditioner already built.

        Returns ``self`` when there is nothing to resolve. Otherwise the
        preconditioner is applied to *operator* once and the result carried as
        a fixed operator, so that a caller sweeping a family of related
        operators can build it once instead of once per member. Whether that is
        a good trade is the caller's judgement — a preconditioner is an
        approximation, so reusing one across a family costs accuracy, never
        correctness.
        """
        if self._preconditioner is None or not isinstance(
            self._preconditioner, LinearSolver
        ):
            return self
        clone = copy.copy(self)
        clone._preconditioner = self._preconditioner(operator)
        return clone

    def _resolve_preconditioner(
        self, operator: LinearOperator
    ) -> LinearOperator | None:
        if self._preconditioner is None:
            return None
        if isinstance(self._preconditioner, LinearSolver):
            return self._preconditioner(operator)
        return self._preconditioner

    def _limit(self, operator: LinearOperator) -> int:
        """The iteration cap.

        Krylov methods terminate within ``dim`` steps in exact arithmetic, but
        rounding routinely costs an extra step or two, so ``dim`` itself is too
        tight a default to be safe. A generous cap is paid for only by a solve
        that was going to fail anyway.
        """
        if self._maxiter is not None:
            return self._maxiter
        return max(2 * operator.domain.dim, 20)

    def _record(self, iteration: int, residual: float, history: list) -> None:
        """Note one step's residual, and tell the callback about it."""
        history.append(float(residual))
        if self._callback is not None:
            self._callback(iteration, float(residual))

    def _finish(self, result: SolveResult) -> SolveResult:
        if not result.converged:
            message = (
                f"{type(self).__name__} did not converge in "
                f"{result.iterations} iterations; the residual norm is "
                f"{result.residual_norm:.3g}."
            )
            if self._strict:
                raise ConvergenceError(message)
            import warnings

            warnings.warn(message, RuntimeWarning, stacklevel=3)
        return result

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        preconditioner = self._resolve_preconditioner(operator)

        def solve_fn(y, x0):
            return self._finish(self._solve(operator, preconditioner, y, x0))

        return InverseOperator(operator, self, solve_fn)

    @abstractmethod
    def _solve(
        self,
        operator: LinearOperator,
        preconditioner: LinearOperator | None,
        b: Any,
        x0: Any | None,
    ) -> SolveResult:
        """Solve ``A x == b``, returning the solution and diagnostics."""

    # -- shared helpers, all coordinate-free ---------------------------

    @staticmethod
    def _initial(operator: LinearOperator, b: Any, x0: Any | None) -> tuple[Any, Any]:
        """Return ``(x, r)`` with ``r == b - A x``."""
        domain, codomain = operator.domain, operator.codomain
        if x0 is None:
            return domain.zero(), codomain.copy(b)
        x = domain.copy(x0)
        return x, codomain.subtract(b, operator(x))

    def _tolerance(self, codomain: HilbertSpace, b: Any) -> float:
        return max(self._rtol * codomain.norm(b), self._atol)


class CGSolver(IterativeSolver):
    """Preconditioned conjugate gradients.

    Requires a positive-definite operator, which by trait closure implies
    self-adjointness. That precondition is now *checked*: the Bayesian normal
    operator ``A Q A* + R`` carries it structurally, so CG accepts it without
    anyone asserting anything, and rejects an operator that has not earned it.
    """

    requires: ClassVar[Traits] = Traits.POSITIVE_DEFINITE

    def _solve(
        self,
        operator: LinearOperator,
        preconditioner: LinearOperator | None,
        b: Any,
        x0: Any | None,
    ) -> SolveResult:
        space = operator.domain
        x, r = self._initial(operator, b, x0)
        tolerance = self._tolerance(operator.codomain, b)

        history: list[float] = []
        residual = space.norm(r)
        self._record(0, residual, history)
        if residual <= tolerance:
            return SolveResult(x, 0, residual, True, tuple(history))

        z = r if preconditioner is None else preconditioner(r)
        p = space.copy(z)
        rz = space.inner_product(r, z)

        for iteration in range(1, self._limit(operator) + 1):
            ap = operator(p)
            curvature = space.inner_product(p, ap)
            if curvature <= 0.0:
                raise ConvergenceError(
                    f"CG met a non-positive curvature direction "
                    f"((p, A p) == {curvature:g}) at iteration {iteration}. "
                    f"The operator claims POSITIVE_DEFINITE but is not; verify "
                    f"the claim with testing.check_traits()."
                )
            alpha = rz / curvature
            x = space.axpy(alpha, p, x)
            r = space.axpy(-alpha, ap, r)

            residual = space.norm(r)
            self._record(iteration, residual, history)
            if residual <= tolerance:
                return SolveResult(x, iteration, residual, True, tuple(history))

            z = r if preconditioner is None else preconditioner(r)
            rz_next = space.inner_product(r, z)
            beta = rz_next / rz
            rz = rz_next
            # p <- z + beta p
            p = space.scale_inplace(beta, p)
            p = space.axpy(1.0, z, p)

        return SolveResult(x, self._limit(operator), residual, False, tuple(history))


class FlexibleCGSolver(IterativeSolver):
    """Conjugate gradients for a preconditioner that changes between steps.

    Ordinary CG assumes a fixed preconditioner; its short recurrence relies on
    it. When the preconditioner is itself an iterative solve — or is rebuilt as
    the iteration proceeds, which is what a localised preconditioner does — the
    recurrence no longer produces conjugate directions and CG stalls.

    The fix is one term: Polak-Ribiere in place of Fletcher-Reeves, so that
    ``beta`` measures the *change* in the residual rather than assuming
    orthogonality it no longer has. That costs one extra inner product and one
    extra stored vector, which is why it is not simply the default.
    """

    requires: ClassVar[Traits] = Traits.POSITIVE_DEFINITE

    def _solve(
        self,
        operator: LinearOperator,
        preconditioner: LinearOperator | None,
        b: Any,
        x0: Any | None,
    ) -> SolveResult:
        space = operator.domain
        x, r = self._initial(operator, b, x0)
        tolerance = self._tolerance(operator.codomain, b)

        history: list[float] = []
        residual = space.norm(r)
        self._record(0, residual, history)
        if residual <= tolerance:
            return SolveResult(x, 0, residual, True, tuple(history))

        z = r if preconditioner is None else preconditioner(r)
        p = space.copy(z)
        rz = space.inner_product(r, z)

        for iteration in range(1, self._limit(operator) + 1):
            ap = operator(p)
            curvature = space.inner_product(p, ap)
            if curvature <= 0.0:
                raise ConvergenceError(
                    f"Flexible CG met a non-positive curvature direction "
                    f"((p, A p) == {curvature:g}) at iteration {iteration}."
                )
            alpha = rz / curvature
            x = space.axpy(alpha, p, x)
            previous = space.copy(r)
            r = space.axpy(-alpha, ap, r)

            residual = space.norm(r)
            self._record(iteration, residual, history)
            if residual <= tolerance:
                return SolveResult(x, iteration, residual, True, tuple(history))

            z = r if preconditioner is None else preconditioner(r)
            # Polak-Ribiere: (r_new - r_old, z_new) rather than (r_new, z_new).
            # The two agree when the preconditioner is fixed, because then the
            # residuals are conjugate; they differ exactly when it is not.
            change = space.subtract(r, previous)
            rz_next = space.inner_product(r, z)
            beta = space.inner_product(change, z) / rz
            rz = rz_next
            p = space.scale_inplace(beta, p)
            p = space.axpy(1.0, z, p)

        return SolveResult(x, self._limit(operator), residual, False, tuple(history))


class GMRESSolver(IterativeSolver):
    """Generalised minimal residual, for an operator with no symmetry at all.

    The only solver here that asks nothing of its operator. Arnoldi builds an
    orthonormal Krylov basis and the residual is minimised over it by a small
    least-squares problem, kept triangular by Givens rotations applied as each
    column arrives — so the residual norm is known at every step without
    forming the iterate.

    Restarted, because the cost and storage of a step grow with the step
    number: ``restart`` vectors are held, then the basis is discarded and the
    iteration begins again from the current residual.
    """

    requires: ClassVar[Traits] = Traits.NONE

    def __init__(self, /, *, restart: int = 30, **kwargs: Any) -> None:
        """
        Args:
            restart: how many Arnoldi vectors to keep before restarting.
            **kwargs: as for :class:`IterativeSolver`.
        """
        if restart < 1:
            raise ValueError(f"The restart length must be positive, got {restart}.")
        super().__init__(**kwargs)
        self._restart = restart

    def _solve(
        self,
        operator: LinearOperator,
        preconditioner: LinearOperator | None,
        b: Any,
        x0: Any | None,
    ) -> SolveResult:
        space = operator.domain
        x, r = self._initial(operator, b, x0)
        tolerance = self._tolerance(operator.codomain, b)
        if preconditioner is not None:
            r = preconditioner(r)
            tolerance = max(self._rtol * space.norm(preconditioner(b)), self._atol)

        history: list[float] = []
        residual = space.norm(r)
        self._record(0, residual, history)
        if residual <= tolerance:
            return SolveResult(x, 0, residual, True, tuple(history))

        limit = self._limit(operator)
        total = 0
        while total < limit:
            width = min(self._restart, limit - total)
            basis = [space.scale(1.0 / residual, r)]
            hessenberg = np.zeros((width + 1, width))
            cosines, sines = np.zeros(width), np.zeros(width)
            rhs = np.zeros(width + 1)
            rhs[0] = residual
            used = 0

            for column in range(width):
                total += 1
                used = column + 1
                w = operator(basis[column])
                if preconditioner is not None:
                    w = preconditioner(w)
                for row in range(column + 1):
                    hessenberg[row, column] = space.inner_product(basis[row], w)
                    w = space.axpy(-hessenberg[row, column], basis[row], w)
                # Kept aside: the rotations below overwrite the subdiagonal
                # entry with zero, which is the point of them, and it is still
                # needed both as the breakdown test and as the next basis
                # vector's normalisation.
                subdiagonal = space.norm(w)
                hessenberg[column + 1, column] = subdiagonal

                # Apply the rotations already accumulated, then make a new one
                # that zeroes the subdiagonal entry this column introduced.
                for row in range(column):
                    upper = hessenberg[row, column]
                    lower = hessenberg[row + 1, column]
                    hessenberg[row, column] = cosines[row] * upper + sines[row] * lower
                    hessenberg[row + 1, column] = (
                        -sines[row] * upper + cosines[row] * lower
                    )
                denominator = np.hypot(
                    hessenberg[column, column], hessenberg[column + 1, column]
                )
                if denominator == 0.0:
                    cosines[column], sines[column] = 1.0, 0.0
                else:
                    cosines[column] = hessenberg[column, column] / denominator
                    sines[column] = hessenberg[column + 1, column] / denominator
                hessenberg[column, column] = denominator
                hessenberg[column + 1, column] = 0.0
                rhs[column + 1] = -sines[column] * rhs[column]
                rhs[column] = cosines[column] * rhs[column]

                residual = abs(rhs[column + 1])
                self._record(total, residual, history)
                if residual <= tolerance or subdiagonal == 0.0:
                    break
                if total >= limit:
                    break
                basis.append(space.scale(1.0 / subdiagonal, w))

            weights = np.linalg.solve(hessenberg[:used, :used], rhs[:used])
            for index in range(used):
                x = space.axpy(weights[index], basis[index], x)

            if residual <= tolerance:
                return SolveResult(x, total, residual, True, tuple(history))
            _, r = self._initial(operator, b, x)
            if preconditioner is not None:
                r = preconditioner(r)
            residual = space.norm(r)

        return SolveResult(x, total, residual, False, tuple(history))


class MinResSolver(IterativeSolver):
    """MINRES for a self-adjoint, possibly indefinite, operator.

    The Paige-Saunders recurrences: Lanczos tridiagonalisation with Givens
    rotations applied on the fly, so the least-squares problem over the Krylov
    space is solved without storing it.
    """

    requires: ClassVar[Traits] = Traits.SELF_ADJOINT

    def _solve(
        self,
        operator: LinearOperator,
        preconditioner: LinearOperator | None,
        b: Any,
        x0: Any | None,
    ) -> SolveResult:
        if preconditioner is not None:
            raise NotImplementedError(
                "Preconditioned MINRES needs a positive-definite preconditioner "
                "and the inner product it induces; it is not implemented."
            )
        space = operator.domain
        x, r = self._initial(operator, b, x0)
        tolerance = self._tolerance(operator.codomain, b)

        beta = space.norm(r)
        if beta <= tolerance:
            return SolveResult(x, 0, beta, True)

        # Lanczos state: v_{k-1}, v_k, and beta_k.
        v_prev = space.zero()
        v = space.scale(1.0 / beta, r)

        # Givens state. c = -1, s = 0 makes the first sweep act as no rotation.
        c, s = -1.0, 0.0
        delta_bar = 0.0
        epsilon = 0.0
        phi_bar = beta

        # Solution update state: w_{k-2}, w_{k-1}.
        w_prev = space.zero()
        w = space.zero()
        residual = beta

        for iteration in range(1, self._limit(operator) + 1):
            # --- Lanczos step: extend the tridiagonalisation ----------
            p = operator(v)
            alpha = space.inner_product(v, p)
            p = space.axpy(-alpha, v, p)
            p = space.axpy(-beta, v_prev, p)
            beta_next = space.norm(p)

            # --- apply the previous rotation to the new column --------
            delta = c * delta_bar + s * alpha
            gamma_bar = s * delta_bar - c * alpha
            epsilon_next = s * beta_next
            delta_bar_next = -c * beta_next

            # --- the new rotation, annihilating beta_next -------------
            gamma = float(np.hypot(gamma_bar, beta_next))
            if gamma == 0.0:
                return SolveResult(x, iteration, residual, True)
            c = gamma_bar / gamma
            s = beta_next / gamma
            tau = c * phi_bar
            phi_bar = s * phi_bar

            # --- w_k == (v_k - delta w_{k-1} - epsilon w_{k-2}) / gamma
            w_next = space.copy(v)
            w_next = space.axpy(-delta, w, w_next)
            w_next = space.axpy(-epsilon, w_prev, w_next)
            w_next = space.scale_inplace(1.0 / gamma, w_next)
            x = space.axpy(tau, w_next, x)

            # phi_bar is the residual norm, available without forming it.
            residual = abs(phi_bar)
            if residual <= tolerance:
                return SolveResult(x, iteration, residual, True)
            if beta_next == 0.0:
                return SolveResult(x, iteration, residual, True)

            # --- roll the state ---------------------------------------
            w_prev, w = w, w_next
            v_prev, v = v, space.scale(1.0 / beta_next, p)
            beta = beta_next
            delta_bar = delta_bar_next
            epsilon = epsilon_next

        return SolveResult(x, self._limit(operator), residual, False)


class BiCGStabSolver(IterativeSolver):
    """BiCGSTAB, for an operator with no symmetry to exploit."""

    def _solve(
        self,
        operator: LinearOperator,
        preconditioner: LinearOperator | None,
        b: Any,
        x0: Any | None,
    ) -> SolveResult:
        space = operator.domain
        x, r = self._initial(operator, b, x0)
        tolerance = self._tolerance(operator.codomain, b)

        residual = space.norm(r)
        if residual <= tolerance:
            return SolveResult(x, 0, residual, True)

        r0 = space.copy(r)
        rho = 1.0
        alpha = 1.0
        omega = 1.0
        p = space.zero()
        v = space.zero()

        for iteration in range(1, self._limit(operator) + 1):
            rho_next = space.inner_product(r0, r)
            if rho_next == 0.0 or omega == 0.0:
                return SolveResult(x, iteration, residual, False)
            beta = (rho_next / rho) * (alpha / omega)
            rho = rho_next

            # p <- r + beta (p - omega v)
            p = space.axpy(-omega, v, p)
            p = space.scale_inplace(beta, p)
            p = space.axpy(1.0, r, p)

            y = p if preconditioner is None else preconditioner(p)
            v = operator(y)
            denominator = space.inner_product(r0, v)
            if denominator == 0.0:
                return SolveResult(x, iteration, residual, False)
            alpha = rho / denominator

            s = space.axpy(-alpha, v, space.copy(r))
            if space.norm(s) <= tolerance:
                x = space.axpy(alpha, y, x)
                return SolveResult(x, iteration, space.norm(s), True)

            z = s if preconditioner is None else preconditioner(s)
            t = operator(z)
            tt = space.inner_product(t, t)
            omega = 0.0 if tt == 0.0 else space.inner_product(t, s) / tt

            x = space.axpy(alpha, y, x)
            x = space.axpy(omega, z, x)
            r = space.axpy(-omega, t, s)

            residual = space.norm(r)
            if residual <= tolerance:
                return SolveResult(x, iteration, residual, True)

        return SolveResult(x, self._limit(operator), residual, False)


# --------------------------------------------------------------------- #
#                          Least squares                                #
# --------------------------------------------------------------------- #


class LeastSquaresSolver(ABC):
    """Solves a rectangular system in the least-squares sense.

    A **sibling** of :class:`LinearSolver`, not a subclass. In v1 ``LSQRSolver``
    sits under ``IterativeLinearSolver``, whose ``__call__`` asserts the
    operator is an automorphism — so the one solver written for rectangular
    systems inherits a squareness check. Solving ``A x == b`` and minimising
    ``||A x - b||`` are different operations and get different types.
    """

    requires_coordinates: ClassVar[bool] = False

    def __call__(self, operator: LinearOperator) -> LinearOperator:
        """The pseudo-inverse of ``operator``, as an operator."""
        if self.requires_coordinates:
            require_coordinates(operator.domain, operator.codomain)
        return self._invert(operator)

    @abstractmethod
    def _invert(self, operator: LinearOperator) -> LinearOperator:
        """Build the pseudo-inverse."""


class LSQRSolver(LeastSquaresSolver):
    """Golub-Kahan bidiagonalisation, applied to the normal equations implicitly.

    Coordinate-free, and damped when asked: with ``damping > 0`` it minimises
    ``||A x - b||^2 + damping^2 ||x||^2``, both norms being the spaces' own.
    """

    def __init__(
        self,
        /,
        *,
        damping: float = 0.0,
        rtol: float = 1e-10,
        maxiter: int | None = None,
        strict: bool = True,
    ) -> None:
        if damping < 0.0:
            raise ValueError("damping must be non-negative.")
        self._damping = damping
        self._rtol = rtol
        self._maxiter = maxiter
        self._strict = strict

    def _invert(self, operator: LinearOperator) -> LinearOperator:
        def solve_fn(b, x0):
            result = self._solve(operator, b)
            if not result.converged and self._strict:
                raise ConvergenceError(
                    f"LSQR did not converge in {result.iterations} iterations; "
                    f"the normal residual is {result.residual_norm:.3g}."
                )
            return result

        return InverseOperator(operator, self, solve_fn, traits=Traits.NONE)

    def _solve(self, operator: LinearOperator, b: Any) -> SolveResult:
        domain, codomain = operator.domain, operator.codomain
        limit = self._maxiter if self._maxiter is not None else 4 * max(domain.dim, 1)

        beta = codomain.norm(b)
        if beta == 0.0:
            return SolveResult(domain.zero(), 0, 0.0, True)
        u = codomain.scale(1.0 / beta, b)

        v = operator.adjoint(u)
        alpha = domain.norm(v)
        if alpha == 0.0:
            return SolveResult(domain.zero(), 0, 0.0, True)
        v = domain.scale_inplace(1.0 / alpha, v)

        x = domain.zero()
        w = domain.copy(v)
        phi_bar, rho_bar = beta, alpha
        residual_target = self._rtol * beta
        normal_residual = alpha * beta
        normal_target = self._rtol * normal_residual

        for iteration in range(1, limit + 1):
            # --- Golub-Kahan bidiagonalisation step -------------------
            u_next = operator(v)
            u_next = codomain.axpy(-alpha, u, u_next)
            beta = codomain.norm(u_next)
            if beta > 0.0:
                u = codomain.scale_inplace(1.0 / beta, u_next)
                v_next = operator.adjoint(u)
                v_next = domain.axpy(-beta, v, v_next)
                alpha = domain.norm(v_next)
                if alpha > 0.0:
                    v = domain.scale_inplace(1.0 / alpha, v_next)

            # --- damping enters as an extra row, annihilated first ----
            # With no damping rho_bar is used SIGNED: taking its magnitude
            # here would discard the sign the recurrence carries, and the
            # iteration would silently converge to the wrong point.
            if self._damping > 0.0:
                rho_bar_damped = float(np.hypot(rho_bar, self._damping))
                phi_bar = (rho_bar / rho_bar_damped) * phi_bar
            else:
                rho_bar_damped = rho_bar

            # --- the rotation annihilating beta ----------------------
            rho = float(np.hypot(rho_bar_damped, beta))
            if rho == 0.0:
                return SolveResult(x, iteration, normal_residual, True)
            c = rho_bar_damped / rho
            s = beta / rho
            theta = s * alpha
            rho_bar = -c * alpha
            phi = c * phi_bar
            phi_bar = s * phi_bar

            # --- update the solution and the search direction --------
            x = domain.axpy(phi / rho, w, x)
            w = domain.scale_inplace(-theta / rho, w)
            w = domain.axpy(1.0, v, w)

            normal_residual = abs(alpha * s * phi)
            if normal_residual <= normal_target or abs(phi_bar) <= residual_target:
                return SolveResult(x, iteration, normal_residual, True)
            if beta == 0.0 or alpha == 0.0:
                return SolveResult(x, iteration, normal_residual, True)

        return SolveResult(x, limit, normal_residual, False)
