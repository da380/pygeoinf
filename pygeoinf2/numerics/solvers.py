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
from ..traits import Traits, adjoint_traits, inverse_traits

__all__ = [
    "SolverLike",
    "resolve_solver",
    "ConvergenceError",
    "ProgressCallback",
    "SolveResult",
    "LinearSolver",
    "InverseOperator",
    "DirectSolver",
    "LUSolver",
    "CholeskySolver",
    "EigenSolver",
    "IterativeSolver",
    "CGSolver",
    "FlexibleCGSolver",
    "GMRESSolver",
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
        if operator.domain != operator.codomain:
            # Matching dimensions are not enough. Every iterative method here
            # adds the iterate to the residual, so the two must be vectors of
            # the *same* space; two spaces of equal dimension over the same
            # vectors but different metrics would give a plausible wrong
            # answer rather than an error.
            raise ValueError(
                f"{type(self).__name__} inverts an operator from a space to "
                f"itself; this one maps {operator.domain!r} to "
                f"{operator.codomain!r}. They have the same dimension but are "
                "not the same space, so the residual and the iterate do not "
                "live together."
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


SolverLike = "LinearSolver | Callable[[LinearOperator], LinearSolver]"
"""A solver, or a function that builds one once the operator is known."""


def resolve_solver(
    solver: Any,
    operator: LinearOperator,
    /,
    *,
    default: LinearSolver | None = None,
) -> LinearSolver:
    """A solver, from a solver or from a function of what it will invert.

    Preconditioning has a sequencing problem: the good preconditioners are
    built *from* the operator being inverted, and the operator does not exist
    until the problem is set up. Three things resolve it, and they suit
    different cases.

    * A preconditioner that is itself a :class:`LinearSolver` is already
      deferred — :meth:`IterativeSolver.with_preconditioner` applies it to the
      operator at solve time. That covers every generic preconditioner and the
      structure-aware ones that read their factors off the operator they are
      given, which is most of them, and it needs nothing from this function.
    * A preconditioner built from *different* factors — a surrogate on a
      coarser space — cannot be derived from the operator, so it needs the
      operator in hand first. That is what passing a callable here is for: it
      receives the assembled operator and returns the solver to use.
    * Failing both, the operator can be built on its own and inspected, since
      it never needed a solver in the first place.

    Args:
        solver: a :class:`LinearSolver`, a callable taking *operator* and
            returning one, or None for *default*.
        operator: what the solver will be asked to invert.
        default: what None means. Conjugate gradients if not given.

    Returns:
        A :class:`LinearSolver`.

    Raises:
        TypeError: if a factory returns something that is not one, or the
            argument is neither a solver, a callable, nor None.
    """
    if solver is None:
        return CGSolver() if default is None else default
    if isinstance(solver, LinearSolver):
        return solver
    if callable(solver):
        resolved = solver(operator)
        if not isinstance(resolved, LinearSolver):
            raise TypeError(
                f"A solver factory must return a LinearSolver, but this one "
                f"returned {type(resolved).__name__}."
            )
        return resolved
    raise TypeError(
        f"The solver must be a LinearSolver, or a callable taking the operator "
        f"to be inverted and returning one. Got {type(solver).__name__}."
    )


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
        adjoint_solve_fn: Callable[[X, Y | None], SolveResult[Y]] | None = None,
        known_matrix: Callable[[str], np.ndarray] | None = None,
        components_action: Callable[[np.ndarray], np.ndarray] | None = None,
        components_adjoint_action: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """
        Args:
            operator: the operator being inverted.
            solver: the solver that produced this inverse.
            solve_fn: ``(y, x0) -> SolveResult``. Note that **each application
                runs the solve again**: an ``InverseOperator`` is a recipe, not
                a stored factorisation, so applying one ``n`` times costs ``n``
                solves.
            traits: claims about the inverse. Deduced from the operator's when
                omitted.
            adjoint_solve_fn: how to solve ``A* w == x``, when the solver can
                do it without redoing its work. A direct solver can: the same
                factorisation solves the transposed system. Without it the
                adjoint has to be inverted from scratch, which for LU means a
                second matrix extraction and a second factorisation.
            known_matrix: ``form -> dense matrix`` of the *inverse*, for a
                solver holding a factorisation that can produce it -- a
                direct solver applies its factors to the identity. Lets
                ``inverse.matrix()`` and everything built on it (a
                preconditioner reading it, a composition assembling) skip
                ``dim`` solves.
            components_action: the inverse's action on components,
                ``c_y -> c_x``, for a solver that can give it -- a direct
                solver's factors applied to a component vector. Lets a
                product containing the inverse stay in coordinates across
                it; see ``LinearOperator._components_action``.
            components_adjoint_action: the same for the adjoint inverse.
        """
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
        self._adjoint_solve_fn = adjoint_solve_fn
        self._known_matrix_fn = known_matrix
        self._components_action_fn = components_action
        self._components_adjoint_action_fn = components_adjoint_action

    @property
    def operator(self) -> LinearOperator[X, Y]:
        """The operator being inverted."""
        return self._operator

    @property
    def solver(self) -> LinearSolver:
        """The solver that produced this inverse."""
        return self._solver

    def solve(self, y: Y, /, *, x0: X | None = None) -> SolveResult[X]:
        """Solve ``A x == y``, returning the solution and its diagnostics.

        Args:
            y: the right-hand side.
            x0: a starting guess. A direct solver ignores it, having nothing
                to iterate; an iterative one starts there, which is what makes
                a damping sweep cheap.

        Returns:
            The solution with its iteration count, residual and history.
        """
        return self._solve_fn(y, x0)

    def _value(self, y: Y) -> X:
        return self._solve_fn(y, None).solution

    def _adjoint_value(self, x: X) -> Y:
        return self.adjoint(x)

    def _known_matrix(self, form: str) -> np.ndarray | None:
        if self._known_matrix_fn is None:
            return None
        return self._known_matrix_fn(form)

    def _components_action(self) -> Callable[[np.ndarray], np.ndarray] | None:
        return self._components_action_fn

    def _components_adjoint_action(
        self,
    ) -> Callable[[np.ndarray], np.ndarray] | None:
        return self._components_adjoint_action_fn

    def adjoint_inverse(self, x: X) -> Y:
        """Apply ``(A^-1)* == (A*)^-1``.

        Kept as a name; it is :attr:`adjoint` applied. It used to hold a
        *second* cache of its own, so ``inv.adjoint(x)`` and this each built
        their own inverse of ``A*`` and the operator ended up inverted twice.
        """
        return self.adjoint(x)

    def _make_adjoint(self) -> LinearOperator[X, Y]:
        """``(A^-1)* == (A*)^-1``, built as an inverse rather than a wrapper.

        Reuses the solver's own transposed solve where there is one, so a
        direct factorisation is not repeated for the adjoint.
        """
        if self._adjoint_solve_fn is not None:
            known_adjoint = None
            if self._known_matrix_fn is not None:

                def known_adjoint(form: str) -> np.ndarray:
                    # Galerkin((A^-1)*) == Galerkin(A^-1)^T; its components
                    # form carries the codomain's inverse metric, as any
                    # adjoint's does.
                    transposed = np.array(self._known_matrix_fn("galerkin").T)
                    if form == "galerkin":
                        return transposed
                    return self._operator.codomain.solve_gram_to_columns(transposed)

            result = InverseOperator(
                self._operator.adjoint,
                self._solver,
                self._adjoint_solve_fn,
                traits=adjoint_traits(self.traits),
                known_matrix=known_adjoint,
                components_action=self._components_adjoint_action_fn,
                components_adjoint_action=self._components_action_fn,
            )
        else:
            result = self._solver(self._operator.adjoint)
        # Close the loop, for the same reason the sum and composition nodes do.
        result._link_adjoint(self)
        return result

    def __repr__(self) -> str:
        return f"Inverse({type(self._solver).__name__}, {self._operator!r})"


# --------------------------------------------------------------------- #
#                            Direct solvers                             #
# --------------------------------------------------------------------- #


class ProgressCallback:
    """A callback that records a solve's progress, and can report it.

    Every iterative solver here takes ``callback=(iteration, residual)``, so
    this is a few lines; it is supplied because the few lines were the same
    ones every caller was writing, and because v1 had it. Pass one and keep it:
    it is the diagnostic an inversion otherwise discards, since an estimator
    applies its inverse operator through the algebra and the ``SolveResult``
    never reaches the caller.

    A single instance can be reused across solves. It resets itself whenever a
    solve starts over at iteration zero, so the counts belong to the last one.

    .. code-block:: python

        progress = ProgressCallback()
        estimator = LinearGaussianInversion(
            problem, prior, solver=CGSolver(callback=progress)
        )
        posterior = estimator(data)
        print(progress.iterations, progress.residual)
    """

    def __init__(self, /, *, report: Callable[[str], None] | None = None) -> None:
        """
        Args:
            report: called with a one-line summary at each iteration. Nothing
                is printed unless one is given, because a library that writes
                to stdout uninvited is a nuisance inside a loop. ``print`` is
                the obvious argument.
        """
        self._report = report
        self.residuals: list[float] = []

    def __call__(self, iteration: int, residual: float) -> None:
        """Record one step."""
        if iteration == 0:
            self.residuals = []
        self.residuals.append(float(residual))
        if self._report is not None:
            self._report(f"iteration {iteration}: residual {residual:.6g}")

    @property
    def iterations(self) -> int:
        """Steps taken in the last solve, not counting the initial residual."""
        return max(len(self.residuals) - 1, 0)

    @property
    def residual(self) -> float:
        """The last residual seen, or infinity before anything has run."""
        return self.residuals[-1] if self.residuals else float("inf")

    def __repr__(self) -> str:
        return (
            f"ProgressCallback(iterations={self.iterations}, "
            f"residual={self.residual:.3g})"
        )


_Factors = tuple[
    Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray] | None
]
"""What a direct solver's factorisation yields: ``(apply_inverse,
apply_transposed)``, the second ``None`` when the factors cannot give it."""


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

    def __init__(self, /, *, n_jobs: int | None = None) -> None:
        """
        Args:
            n_jobs: workers for extracting the matrix when the operator has
                to be probed for it, one column per worker. An operator that
                knows its matrix -- built from one, or a sum or composition
                of such -- is read, and the setting is not used. Serial by
                default; see :mod:`pygeoinf2.parallel`.
        """
        self._n_jobs = n_jobs

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        domain: CoordinateSpace = operator.domain
        codomain: CoordinateSpace = operator.codomain
        matrix = operator.matrix(form=self.form, n_jobs=self._n_jobs)
        apply_inverse, apply_transposed = self._factorise(matrix)

        def solve_fn(y, x0):
            cy = codomain.to_components(y)
            if self.form == "galerkin":
                # M c_x == G_Y c_y, since M == G_Y A_c.
                cy = codomain.apply_gram(cy)
            cx = apply_inverse(cy)
            return SolveResult(domain.from_components(cx), 0, 0.0, True)

        def known_matrix(form: str) -> np.ndarray:
            # (A^-1)_c is M^-1 for M the components form, and M^-1 G_Y for M
            # the Galerkin form G_Y A_c: the factors applied to a matrix.
            if self.form == "components":
                inverse = apply_inverse(np.identity(domain.dim))
            else:
                inverse = apply_inverse(codomain.gram_matrix())
            if form == "components":
                return inverse
            return domain.apply_gram_to_columns(inverse)

        def components_action(cy: np.ndarray) -> np.ndarray:
            if self.form == "galerkin":
                cy = codomain.apply_gram(cy)
            return apply_inverse(cy)

        adjoint_solve_fn = None
        components_adjoint_action = None
        if apply_transposed is not None:

            def components_adjoint_action(cx: np.ndarray) -> np.ndarray:
                cw = apply_transposed(domain.apply_gram(cx))
                if self.form == "components":
                    cw = codomain.solve_gram(cw)
                return cw

            def adjoint_solve_fn(x, w0):
                # Solve A* w == x for w in the codomain. With M the components
                # form, A* has component matrix G_X^-1 M^T G_Y, so
                # c_w == G_Y^-1 M^-T G_X c_x; with M the Galerkin form the
                # trailing G_Y is already in M and the last solve drops out.
                cx = domain.apply_gram(domain.to_components(x))
                cw = apply_transposed(cx)
                if self.form == "components":
                    cw = codomain.solve_gram(cw)
                return SolveResult(codomain.from_components(cw), 0, 0.0, True)

        if components_adjoint_action is None and Traits.SELF_ADJOINT & operator.traits:
            # A self-adjoint operator's inverse is its own adjoint.
            components_adjoint_action = components_action
        return InverseOperator(
            operator,
            self,
            solve_fn,
            adjoint_solve_fn=adjoint_solve_fn,
            known_matrix=known_matrix,
            components_action=components_action,
            components_adjoint_action=components_adjoint_action,
        )

    @abstractmethod
    def _factorise(self, matrix: np.ndarray) -> _Factors:
        """Factorise **once**, returning ``(apply_inverse, apply_transposed)``.

        Both accept a vector or a matrix of columns. The second applies
        ``M^-T`` from the same factors and is ``None`` when the factorisation
        cannot, which means "invert the adjoint from scratch"; a symmetric
        factorisation needs no second: its operator is self-adjoint, so the
        inverse is too and its adjoint is itself.

        One call, deliberately. The first version had a second method for the
        transposed solve and it factorised again -- the O(n^3) step, twice,
        for a solver whose whole point is to do it once.
        """


class LUSolver(DirectSolver):
    """LU factorisation. Makes no structural demands beyond squareness."""

    form: ClassVar[str] = "components"

    def _factorise(self, matrix: np.ndarray) -> _Factors:
        factor = lu_factor(matrix)
        # ``trans=1`` is M^-T from the same factors: the adjoint of an LU
        # inverse costs no second factorisation.
        return (lambda c: lu_solve(factor, c), lambda c: lu_solve(factor, c, trans=1))


class CholeskySolver(DirectSolver):
    """Cholesky factorisation of the Galerkin matrix.

    Requires positive-definiteness, which by closure implies self-adjointness —
    and the Galerkin form is what makes that visible as symmetry.
    """

    requires: ClassVar[Traits] = Traits.POSITIVE_DEFINITE
    form: ClassVar[str] = "galerkin"

    def _factorise(self, matrix: np.ndarray) -> _Factors:
        symmetric = 0.5 * (matrix + matrix.T)
        factor = cho_factor(symmetric)
        return (lambda c: cho_solve(factor, c), None)


class EigenSolver(DirectSolver):
    """Symmetric eigendecomposition, with a pseudo-inverse for small modes."""

    requires: ClassVar[Traits] = Traits.SELF_ADJOINT
    form: ClassVar[str] = "galerkin"

    def __init__(self, /, *, rtol: float = 1e-12, n_jobs: int | None = None) -> None:
        """
        Args:
            rtol: eigenvalues below this fraction of the largest in magnitude
                are treated as zero and their directions dropped.
            n_jobs: workers for extracting the matrix, as for
                :class:`DirectSolver`.
        """
        super().__init__(n_jobs=n_jobs)
        self._rtol = rtol

    def _factorise(self, matrix: np.ndarray) -> _Factors:
        symmetric = 0.5 * (matrix + matrix.T)
        values, vectors = eigh(symmetric)
        largest = np.max(np.abs(values)) if values.size else 0.0
        threshold = self._rtol * largest
        inverted = np.where(np.abs(values) > threshold, 1.0 / values, 0.0)

        def apply(c: np.ndarray) -> np.ndarray:
            coefficients = vectors.T @ c
            scaled = (
                inverted[:, None] * coefficients
                if coefficients.ndim == 2
                else inverted * coefficients
            )
            return vectors @ scaled

        return (apply, None)


# --------------------------------------------------------------------- #
#                          Iterative solvers                            #
# --------------------------------------------------------------------- #


class _ComponentOperator(LinearOperator):
    """An operator seen on the component views of its spaces.

    Acts on component arrays. Where the operator can act on components
    directly it does, and a product of such operators stays in components
    end to end; otherwise the array is synthesised, the operator applied,
    and the image analysed -- which is what the operator would have cost
    anyway. Traits carry over: they are statements about the metric, and the
    view has the same one.
    """

    def __init__(
        self, operator: LinearOperator, domain: Any, codomain: Any, /
    ) -> None:
        super().__init__(domain, codomain, traits=operator.traits)
        self._operator = operator
        self._action = operator._components_action()
        self._adjoint_action = operator._components_adjoint_action()

    def _value(self, c: np.ndarray) -> np.ndarray:
        if self._action is not None:
            return np.asarray(self._action(c), dtype=float)
        operator = self._operator
        return operator.codomain.to_components(
            operator(operator.domain.from_components(c))
        )

    def _adjoint_value(self, c: np.ndarray) -> np.ndarray:
        if self._adjoint_action is not None:
            return np.asarray(self._adjoint_action(c), dtype=float)
        operator = self._operator
        return operator.domain.to_components(
            operator.adjoint(operator.codomain.from_components(c))
        )

    def _known_matrix(self, form: str) -> np.ndarray | None:
        return self._operator._known_matrix(form)

    def _known_diagonals(
        self, offsets: tuple[int, ...], form: str
    ) -> np.ndarray | None:
        return self._operator._known_diagonals(offsets, form)

    def __repr__(self) -> str:
        return f"OnComponents({self._operator!r})"


class IterativeSolver(LinearSolver):
    """A Krylov solver, written against the space's own inner product.

    Every arithmetic operation here goes through ``inner_product``, ``axpy``
    and ``scale_inplace``, so these run unchanged against a space with no
    coordinate map. That is not incidental: measuring the residual in the
    space's norm rather than in a component norm is what makes the stopping
    criterion mean the same thing under refinement.

    **On a space with coordinates the loop runs on the components.** The
    right-hand side is converted in once and the solution out once; between
    them the operator, the preconditioner and the vector algebra act on
    arrays through a :class:`~pygeoinf2.algebra.spaces.ComponentView`, whose
    inner product is the space's own, so the residual history is unchanged
    to rounding. On a sphere the vector algebra alone was seven transforms
    per iteration; an operator that can act on components then costs none,
    and one that cannot costs what it cost before.
    """

    requires_coordinates: ClassVar[bool] = False

    # Run the loop on the space's ComponentView when it has one. Off only for
    # the tests that pin the two routes against each other.
    _use_component_view: ClassVar[bool] = True

    def __init__(
        self,
        /,
        *,
        rtol: float = 1e-8,
        atol: float = 0.0,
        maxiter: int | None = None,
        preconditioner: LinearSolver | LinearOperator | None = None,
        strict: bool = True,
        callback: Callable[[int, float], None] | None = None,
    ) -> None:
        """
        Args:
            rtol: stop when the residual norm falls below ``rtol * ||b||``,
                measured in the space's own norm and relative to ``||b||``
                rather than to ``||b - A x0||``.

                The default is ``1e-8``. On a 40 000-dimensional five-point
                Laplacian with a 0.01 shift, conjugate gradients takes 147
                iterations at ``1e-5`` (v1's default), 243 at ``1e-8`` and 301
                at ``1e-10`` (v2's first choice), for relative errors in the
                solution of 4e-5, 4e-8 and 4e-10. ``1e-10`` costs a quarter
                more iterations than ``1e-8`` to reach an accuracy no inverse
                problem can use, and on an ill-conditioned normal operator it
                is often unreachable — which with ``strict=True`` turns a
                usable answer into an exception. That combination is what broke
                one of the examples during the port.
            atol: absolute floor on the same test.
            maxiter: iteration cap. Defaults to ``max(2 * dim, 20)``: a Krylov
                method terminates within ``dim`` steps in exact arithmetic, but
                rounding routinely costs a step or two more.
            preconditioner: either a ready-made approximate inverse, or a
                ``LinearSolver`` from which to build one once the operator is
                known.
            strict: raise :class:`ConvergenceError` on failure to converge
                rather than warning.
            callback: called with ``(iteration, residual_norm)`` after each
                step. For watching a long solve, and for finding out *where* a
                stalled one stalled — which the final residual alone cannot
                say. Every iterative solver here honours it, and records the
                same numbers in :attr:`SolveResult.history`.
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
        preconditioner it knows how to build.

        **Refuses to replace one that is already set.** It used to return the
        solver unchanged instead, which reads as success and is not: a caller
        writing ``solver.with_preconditioner(mine)`` on a solver that already
        had one got their preconditioner silently discarded and no way to tell.
        A library routine that means "supply one only if the caller did not"
        should ask, through :attr:`preconditioner`, and say so.

        Args:
            preconditioner: the preconditioner to attach.

        Returns:
            A copy of this solver carrying it.

        Raises:
            ValueError: if this solver already has one.
        """
        if self._preconditioner is not None:
            raise ValueError(
                f"{type(self).__name__} already has a preconditioner "
                f"({type(self._preconditioner).__name__}). Replacing it "
                "silently would discard whichever of the two the caller meant; "
                "check .preconditioner first, or build a fresh solver."
            )
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
        viewed = self._viewed(operator, preconditioner)

        def solve_fn(y, x0):
            if viewed is None:
                return self._finish(self._solve(operator, preconditioner, y, x0))
            # In components: the right-hand side and the start converted in
            # once, the solution out once, and every inner product, norm and
            # update in between an array operation with the same metric --
            # so the residual history is the one the space's norm gives.
            viewed_operator, viewed_preconditioner = viewed
            domain, codomain = operator.domain, operator.codomain
            result = self._solve(
                viewed_operator,
                viewed_preconditioner,
                codomain.to_components(y),
                None if x0 is None else domain.to_components(x0),
            )
            return self._finish(
                SolveResult(
                    domain.from_components(result.solution),
                    result.iterations,
                    result.residual_norm,
                    result.converged,
                    result.history,
                )
            )

        return InverseOperator(operator, self, solve_fn)

    def _viewed(
        self, operator: LinearOperator, preconditioner: LinearOperator | None
    ) -> tuple[LinearOperator, LinearOperator | None] | None:
        """The operator and preconditioner on the component views of their
        spaces, or ``None`` when the spaces have no components to view."""
        from ..algebra.spaces import ComponentView, CoordinateSpace

        domain, codomain = operator.domain, operator.codomain
        if not self._use_component_view:
            return None
        if not all(
            isinstance(space, CoordinateSpace) and space.uses_component_fast_paths
            for space in (domain, codomain)
        ):
            return None
        views = {domain: ComponentView(domain)}
        views.setdefault(codomain, ComponentView(codomain))
        viewed_operator = _ComponentOperator(operator, views[domain], views[codomain])
        viewed_preconditioner = (
            None
            if preconditioner is None
            else _ComponentOperator(preconditioner, views[codomain], views[domain])
        )
        return viewed_operator, viewed_preconditioner

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
        # Without a preconditioner (r, z) is (r, r), which the norm just gave.
        rz = residual**2 if preconditioner is None else space.inner_product(r, z)

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
            rz_next = (
                residual**2 if preconditioner is None else space.inner_product(r, z)
            )
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

    Preconditioning is supported, and needs a *positive-definite* one: the
    method runs in the inner product ``M`` induces, so an indefinite ``M``
    gives a negative squared norm and is refused with a message saying so
    rather than producing a NaN some iterations later. That is the only extra
    condition -- the operator itself may still be indefinite, which is the
    reason to reach for MINRES over CG in the first place.
    """

    requires: ClassVar[Traits] = Traits.SELF_ADJOINT

    def _solve(
        self,
        operator: LinearOperator,
        preconditioner: LinearOperator | None,
        b: Any,
        x0: Any | None,
    ) -> SolveResult:
        space = operator.domain
        x, r = self._initial(operator, b, x0)

        # Preconditioned MINRES runs the same recurrences in the inner product
        # the preconditioner induces: the Lanczos vectors are M-orthonormal, so
        # every norm below is ``sqrt((u, M u))`` rather than ``sqrt((u, u))``.
        # With no preconditioner M is the identity and this is the plain
        # method, term for term.
        def preconditioned(vector: Any) -> Any:
            return vector if preconditioner is None else preconditioner(vector)

        def m_norm(vector: Any, image: Any, /) -> float:
            """``sqrt((u, M u))``, from ``u`` and ``M u`` already in hand."""
            squared = space.inner_product(vector, image)
            if squared < 0.0:
                raise ValueError(
                    "MINRES needs a positive-definite preconditioner: the "
                    f"inner product it induces gave {squared:.3e} for a "
                    "vector's own norm."
                )
            return float(np.sqrt(squared))

        # Relative to the M-norm of the right-hand side, since that is the norm
        # phi_bar below is measured in.
        tolerance = max(
            self._rtol * m_norm(b, preconditioned(b)), self._atol
        )

        history: list[float] = []
        z = preconditioned(r)
        beta = m_norm(r, z)
        self._record(0, beta, history)
        if beta <= tolerance:
            return SolveResult(x, 0, beta, True, tuple(history))

        # Lanczos state: the two previous *unpreconditioned* residuals, which
        # are what the three-term recurrence runs on, and v_k == M r_k / beta.
        r_prev = space.zero()
        r_current = r
        v = space.scale(1.0 / beta, z)

        # Givens state. c = -1, s = 0 makes the first sweep act as no rotation.
        c, s = -1.0, 0.0
        delta_bar = 0.0
        epsilon = 0.0
        phi_bar = beta

        # Solution update state: w_{k-2}, w_{k-1}.
        w_prev = space.zero()
        w = space.zero()
        residual = beta
        beta_prev = 1.0

        for iteration in range(1, self._limit(operator) + 1):
            # --- Lanczos step: extend the tridiagonalisation ----------
            # The recurrence is carried on the residuals rather than on the
            # M-orthonormal vectors: p is the next unpreconditioned residual,
            # and M p is what gives both the next vector and its norm.
            p = operator(v)
            alpha = space.inner_product(v, p)
            p = space.axpy(-alpha / beta, r_current, p)
            if iteration > 1:
                p = space.axpy(-beta / beta_prev, r_prev, p)
            z_next = preconditioned(p)
            beta_next = m_norm(p, z_next)

            # --- apply the previous rotation to the new column --------
            delta = c * delta_bar + s * alpha
            gamma_bar = s * delta_bar - c * alpha
            epsilon_next = s * beta_next
            delta_bar_next = -c * beta_next

            # --- the new rotation, annihilating beta_next -------------
            gamma = float(np.hypot(gamma_bar, beta_next))
            if gamma == 0.0:
                return SolveResult(x, iteration, residual, True, tuple(history))
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
            self._record(iteration, residual, history)
            if residual <= tolerance:
                return SolveResult(x, iteration, residual, True, tuple(history))
            if beta_next == 0.0:
                return SolveResult(x, iteration, residual, True, tuple(history))

            # --- roll the state ---------------------------------------
            w_prev, w = w, w_next
            v = space.scale(1.0 / beta_next, z_next)
            r_prev, r_current = r_current, p
            beta_prev, beta = beta, beta_next
            delta_bar = delta_bar_next
            epsilon = epsilon_next

        return SolveResult(x, self._limit(operator), residual, False, tuple(history))


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

        history: list[float] = []
        residual = space.norm(r)
        self._record(0, residual, history)
        if residual <= tolerance:
            return SolveResult(x, 0, residual, True, tuple(history))

        r0 = space.copy(r)
        rho = 1.0
        alpha = 1.0
        omega = 1.0
        p = space.zero()
        v = space.zero()

        for iteration in range(1, self._limit(operator) + 1):
            rho_next = space.inner_product(r0, r)
            if rho_next == 0.0 or omega == 0.0:
                return SolveResult(x, iteration, residual, False, tuple(history))
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
                return SolveResult(x, iteration, residual, False, tuple(history))
            alpha = rho / denominator

            s = space.axpy(-alpha, v, space.copy(r))
            if space.norm(s) <= tolerance:
                x = space.axpy(alpha, y, x)
                self._record(iteration, space.norm(s), history)
                return SolveResult(x, iteration, space.norm(s), True, tuple(history))

            z = s if preconditioner is None else preconditioner(s)
            t = operator(z)
            tt = space.inner_product(t, t)
            omega = 0.0 if tt == 0.0 else space.inner_product(t, s) / tt

            x = space.axpy(alpha, y, x)
            x = space.axpy(omega, z, x)
            r = space.axpy(-omega, t, s)

            residual = space.norm(r)
            self._record(iteration, residual, history)
            if residual <= tolerance:
                return SolveResult(x, iteration, residual, True, tuple(history))

        return SolveResult(x, self._limit(operator), residual, False, tuple(history))


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
        callback: Callable[[int, float], None] | None = None,
    ) -> None:
        """
        Args:
            damping: minimise ``||A x - b||^2 + damping^2 ||x||^2`` when
                positive. See :meth:`_solve` on what this means with a warm
                start.
            rtol: relative tolerance, applied to both the residual and the
                normal residual.
            maxiter: iteration cap. Four times the domain dimension by default.
            strict: raise :class:`ConvergenceError` rather than warn when the
                cap is reached.
            callback: called as ``callback(iteration, normal_residual)`` each
                step, and the same numbers are kept in
                :attr:`SolveResult.history`.

        Raises:
            ValueError: if the damping is negative.
        """
        if damping < 0.0:
            raise ValueError("damping must be non-negative.")
        self._damping = damping
        self._rtol = rtol
        self._maxiter = maxiter
        self._strict = strict
        self._callback = callback

    def _invert(self, operator: LinearOperator) -> LinearOperator:
        def solve_fn(b, x0):
            result = self._solve(operator, b, x0)
            if not result.converged and self._strict:
                raise ConvergenceError(
                    f"LSQR did not converge in {result.iterations} iterations; "
                    f"the normal residual is {result.residual_norm:.3g}."
                )
            return result

        return InverseOperator(operator, self, solve_fn, traits=Traits.NONE)

    def _note(self, iteration: int, residual: float, history: list) -> None:
        """Note one step's normal residual, and tell the callback about it."""
        history.append(float(residual))
        if self._callback is not None:
            self._callback(iteration, float(residual))

    def _solve(self, operator: LinearOperator, b: Any, x0: Any | None) -> SolveResult:
        """The bidiagonalisation, started from *x0* when one is given.

        The warm start solves for the *correction*: the iteration runs on the
        residual ``b - A x0`` and accumulates into a copy of ``x0``, which is
        what v1 does. With no damping that is exactly equivalent to starting
        from zero, so a warm start can only save iterations. With damping it is
        not: the penalty then falls on the correction rather than on the whole
        solution, which is a different minimisation. Damped warm starts are
        therefore refused rather than quietly answering a different question.

        Raises:
            ValueError: if both a warm start and a damping are given.
        """
        domain, codomain = operator.domain, operator.codomain
        limit = self._maxiter if self._maxiter is not None else 4 * max(domain.dim, 1)
        history: list[float] = []

        if x0 is not None and self._damping > 0.0:
            raise ValueError(
                "A damped LSQR cannot be warm-started: the iteration would "
                "penalise the correction rather than the solution, which "
                "minimises something else. Start from zero, or damp by "
                "composing with the shift yourself."
            )

        start = domain.zero() if x0 is None else domain.copy(x0)
        shifted = b if x0 is None else codomain.subtract(b, operator(start))

        beta = codomain.norm(shifted)
        if beta == 0.0:
            return SolveResult(start, 0, 0.0, True)
        u = codomain.scale(1.0 / beta, shifted)

        v = operator.adjoint(u)
        alpha = domain.norm(v)
        if alpha == 0.0:
            return SolveResult(start, 0, 0.0, True)
        v = domain.scale_inplace(1.0 / alpha, v)

        x = start
        w = domain.copy(v)
        phi_bar, rho_bar = beta, alpha
        normal_residual = alpha * beta

        # Both tolerances are relative to the *data*, not to the shifted
        # residual the iteration happens to start from. For a cold start the
        # two coincide (``alpha * beta == ||A* b||`` exactly, since
        # ``u == b / beta``). For a warm start they do not: starting from the
        # solution leaves a normal residual of ~0, so a target relative to it
        # is unreachable and the iteration would run to its cap having nothing
        # left to do.
        if x0 is None:
            residual_target = self._rtol * beta
            normal_target = self._rtol * normal_residual
        else:
            residual_target = self._rtol * codomain.norm(b)
            normal_target = self._rtol * domain.norm(operator.adjoint(b))

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
                return SolveResult(x, iteration, normal_residual, True, tuple(history))
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
            self._note(iteration, normal_residual, history)
            if normal_residual <= normal_target or abs(phi_bar) <= residual_target:
                return SolveResult(x, iteration, normal_residual, True, tuple(history))
            if beta == 0.0 or alpha == 0.0:
                return SolveResult(x, iteration, normal_residual, True, tuple(history))

        return SolveResult(x, limit, normal_residual, False, tuple(history))
