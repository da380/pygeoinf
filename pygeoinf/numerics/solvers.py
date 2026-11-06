"""
Linear solvers.

A solver turns an operator into its inverse, *as an operator*:
``solver(A)`` is an :class:`InverseOperator`, and applying it solves a system.

- **Preconditions are declared and checked.** A solver states the traits it
  needs (``requires``) and whether it needs coordinates
  (``requires_coordinates``), and ``__call__`` validates the operator before
  doing any work, raising with a message that names what is missing.

- **The iterative solvers are coordinate-free.** CG, MINRES, GMRES, BiCGStab
  and LSQR are written against ``inner_product`` and ``axpy`` alone, so they
  run against a space with no component map at all. Only the direct solvers
  set ``requires_coordinates``.

- **Solvers are stateless.** Diagnostics are never stored on the solver; they
  come back from :meth:`InverseOperator.solve` as a :class:`SolveResult`, so
  one solver can be shared between any number of inversions.

- **Non-square is a different operation.** A least-squares solver is a sibling
  of ``LinearSolver``, not a subclass, and performs no squareness check.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Literal

import numpy as np
from scipy.linalg import (
    cho_factor,
    cho_solve,
    cholesky,
    eigh,
    lu_factor,
    lu_solve,
    solve_triangular,
)

from ..algebra.operators import LinearOperator, require_coordinates
from ..algebra.spaces import CoordinateSpace, HilbertSpace
from ..traits import Traits, adjoint_traits, inverse_traits

__all__ = [
    "BoundFamily",
    "FamilySolver",
    "SpectralFamilySolver",
    "SolverLike",
    "ModelSpace",
    "resolve_solver",
    "ConvergenceError",
    "ProgressCallback",
    "SolutionTrackingCallback",
    "SolveStep",
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


class SolveStep[X]:
    """One step of an iterative solve, as its callback sees it.

    ``iteration`` and ``residual`` are the numbers every solver has in hand
    after a step and records in :attr:`SolveResult.history`. ``iterate`` is
    the current approximation, and it is a *property* because not every
    solver holds one: conjugate gradients, MINRES, BiCGSTAB and LSQR update
    the solution every step, but GMRES assembles it only at the end of a
    restart cycle, from the Arnoldi basis and the triangular system it has
    built so far, at a cost of one vector operation per basis vector. Forming
    it on demand lets a callback that only counts steps pay nothing, and one
    that wants the path -- :class:`SolutionTrackingCallback` -- pay only for
    what it asks. What comes back is an independent copy, yours to keep: the
    solvers update their iterate in place, so handing out the live vector
    would give a history in which every entry is the final answer.

    Take it during the callback. The step is a view onto the solver's state
    at that moment, and the state moves on when the callback returns. A
    solve that runs on the component views of its spaces still hands out
    the iterate as a vector of the operator's domain, not as its components.
    """

    __slots__ = ("iteration", "residual", "_form")

    def __init__(self, iteration: int, residual: float, form: Callable[[], X]) -> None:
        self.iteration = iteration
        self.residual = residual
        self._form = form

    @property
    def iterate(self) -> X:
        """An independent copy of the current approximation."""
        return self._form()

    def __repr__(self) -> str:
        return f"SolveStep(iteration={self.iteration}, residual={self.residual:.3g})"


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


type SolverLike = LinearSolver | Callable[[LinearOperator], LinearSolver]
"""A solver, or a function that builds one once the operator is known."""


class ModelSpace:
    """Solve a linear problem's normal equations in the model space.

    The estimators of a linear problem, and the feasible set's algorithms,
    solve normal equations, and by default they assemble them in the data
    space: ``A Q A* + R``, of the data's dimension, which needs the
    covariances. Passing ``solver=ModelSpace(solver)`` asks for the
    model-space equations instead: ``Q^-1 + A* R^-1 A``, of the model's
    dimension, which needs the precisions and an error measure, since exact
    data cannot be written in the model space. The two give the same answer;
    the choice is only of which linear system is solved and how well
    conditioned it is.

    The model space suits an explicitly finite-dimensional problem with more
    data than unknowns and a cheap prior precision, and a low-dimensional
    surrogate, which is such a problem by construction. On a space that
    discretizes functions the data space is the one to use: the conditioning
    of a function-space prior's precision grows under refinement, and so do
    the log-determinants behind a model-space evidence.

    It is a choice of equations, not a solver, so
    :func:`resolve_solver` refuses it: it is unwrapped by whatever assembles
    the equations.
    """

    def __init__(self, solver: SolverLike | None, /) -> None:
        """
        Args:
            solver: how to invert the model-space normal operator: a
                :class:`LinearSolver`, a callable taking the operator and
                returning one, or None for the default of whatever uses it,
                as in ``ModelSpace(None)``.

        Raises:
            TypeError: if *solver* is itself a ``ModelSpace``.
        """
        if isinstance(solver, ModelSpace):
            raise TypeError(
                "ModelSpace(ModelSpace(...)) asks for the model space twice; "
                "wrap the solver once."
            )
        self._solver = solver

    @property
    def solver(self) -> SolverLike | None:
        """The wrapped solver, or None for the default."""
        return self._solver

    @staticmethod
    def split(solver: Any, /) -> tuple[Literal["model_space", "data_space"], Any]:
        """Which equations a ``solver=`` argument asks for, and the solver.

        Args:
            solver: what was passed as ``solver=``.

        Returns:
            ``("model_space", inner)`` for ``ModelSpace(inner)``, and
            ``("data_space", solver)`` for anything else, None included.
        """
        if isinstance(solver, ModelSpace):
            return "model_space", solver.solver
        return "data_space", solver

    @staticmethod
    def wrap(space: Literal["model_space", "data_space"], solver: Any, /) -> Any:
        """The ``solver=`` argument asking for *space* with *solver*: the
        inverse of :meth:`split`.

        Args:
            space: ``"model_space"`` or ``"data_space"``.
            solver: the solver, or None for the default.

        Returns:
            ``ModelSpace(solver)`` for the model space, *solver* itself for
            the data space.

        Raises:
            ValueError: if *space* is neither.
        """
        if space == "model_space":
            return ModelSpace(solver)
        if space == "data_space":
            return solver
        raise ValueError(f"The space is 'model_space' or 'data_space', got {space!r}.")

    def __repr__(self) -> str:
        return f"ModelSpace({self._solver!r})"


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
        TypeError: if a factory returns something that is not one, the
            argument is neither a solver, a callable, nor None, or it is a
            :class:`ModelSpace`, which only an assembly of normal equations
            unwraps.
    """
    if isinstance(solver, ModelSpace):
        raise TypeError(
            "ModelSpace(...) chooses which normal equations are solved, and "
            "is unwrapped only where they are assembled: as solver= to "
            "LeastSquares, MinimumNorm, their constrained forms, "
            "LinearGaussianInversion and the mixture inversion, or to a "
            "feasible-set algorithm passed through via=. It is not itself a "
            "solver; pass the solver it wraps here. A normal operator or a "
            "Tikhonov family takes formalism= instead."
        )
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
    diagnostics with the answer; solvers are stateless, so this is the only
    place the iteration count and residual are reported.
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
                a stored factorization, so applying one ``n`` times costs ``n``
                solves.
            traits: claims about the inverse. Deduced from the operator's when
                omitted.
            adjoint_solve_fn: how to solve ``A* w == x``, when the solver can
                do it without redoing its work. A direct solver can: the same
                factorization solves the transposed system. Without it the
                adjoint has to be inverted from scratch, which for LU means a
                second matrix extraction and a second factorization.
            known_matrix: ``form -> dense matrix`` of the *inverse*, for a
                solver holding a factorization that can produce it -- a
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

        A named spelling of :attr:`adjoint` applied to ``x``; the two share
        one inverse of ``A*``.
        """
        return self.adjoint(x)

    def _make_adjoint(self) -> LinearOperator[X, Y]:
        """``(A^-1)* == (A*)^-1``, built as an inverse rather than a wrapper.

        Reuses the solver's own transposed solve where there is one, so a
        direct factorization is not repeated for the adjoint.
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
        # Link the pair, so that ``self.adjoint.adjoint is self``: the
        # structural recognition in the expression nodes compares factors by
        # identity.
        result._link_adjoint(self)
        return result

    def __repr__(self) -> str:
        return f"Inverse({type(self._solver).__name__}, {self._operator!r})"


# --------------------------------------------------------------------- #
#                              Callbacks                                #
# --------------------------------------------------------------------- #


def _record_residual(
    name: str,
    iteration: int,
    residual: float,
    history: list,
    callback: Callable[[SolveStep], None] | None,
    form: Callable[[], Any],
) -> None:
    """Note one step's residual, refusing a non-finite one.

    Shared by every iterative method, LSQR included, so that a breakdown is
    reported the same way wherever it happens; see
    :meth:`IterativeSolver._record` for why it is refused regardless of
    ``strict``.

    Raises:
        ConvergenceError: for a NaN or infinite residual.
    """
    residual = float(residual)
    if not np.isfinite(residual):
        raise ConvergenceError(
            f"{name} met a non-finite residual ({residual}) at iteration "
            f"{iteration}. The operator or the preconditioner returned NaN or "
            f"inf, or the problem needs rescaling; this is a breakdown, so "
            f"strict=False does not downgrade it."
        )
    history.append(residual)
    if callback is not None:
        callback(SolveStep(iteration, residual, form))


def _finish_solve(
    name: str, result: SolveResult, strict: bool, what: str
) -> SolveResult:
    """Raise or warn on a solve that did not converge, and hand it back.

    Raises:
        ConvergenceError: when *strict* and the result is not converged.
    """
    if not result.converged:
        message = (
            f"{name} did not converge in {result.iterations} iterations; the "
            f"{what} norm is {result.residual_norm:.3g}."
        )
        if strict:
            raise ConvergenceError(message)
        import warnings

        warnings.warn(message, RuntimeWarning, stacklevel=3)
    return result


class ProgressCallback:
    """A callback that records a solve's progress, and can report it.

    Every iterative solver here calls its ``callback`` with a
    :class:`SolveStep`; this one keeps the residuals. Pass one and keep a
    reference to it: it is the only way to see the diagnostics of a solve
    inside an inversion, since an estimator applies its inverse operator
    through the algebra and the ``SolveResult`` never reaches the caller. To
    keep the iterates as well, see :class:`SolutionTrackingCallback`.

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

    def __call__(self, step: SolveStep) -> None:
        """Record one step."""
        if step.iteration == 0:
            self.residuals = []
        self.residuals.append(float(step.residual))
        if self._report is not None:
            self._report(f"iteration {step.iteration}: residual {step.residual:.6g}")

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


class SolutionTrackingCallback(ProgressCallback):
    """A callback that keeps every iterate of a solve, as well as its residuals.

    For looking at the path a solver took: how the iterates approach the
    answer, which components settle first, what a
    stalled solve was doing when it stalled. Each entry of :attr:`iterates`
    is an independent copy taken at that step, so ``iterates[0]`` is the
    starting point and ``iterates[-1]`` the solution. Like its base it
    resets itself whenever a solve starts over at iteration zero, so the
    path belongs to the last solve.

    The cost is one copy of the iterate per step, in memory as well as time
    -- ``iterations`` vectors held at once -- and on GMRES the assembly of
    each iterate from the Arnoldi basis, which the solver otherwise does once
    per restart cycle. A callback that does not ask for the iterate pays
    neither; see :class:`SolveStep`.

    .. code-block:: python

        path = SolutionTrackingCallback()
        CGSolver(callback=path)(A).solve(b)
        errors = [A.domain.norm(A.domain.subtract(x, exact)) for x in path.iterates]
    """

    def __init__(self, /, *, report: Callable[[str], None] | None = None) -> None:
        super().__init__(report=report)
        self.iterates: list[Any] = []

    def __call__(self, step: SolveStep) -> None:
        """Record one step, iterate included."""
        super().__call__(step)
        if step.iteration == 0:
            self.iterates = []
        self.iterates.append(step.iterate)

    def __repr__(self) -> str:
        return (
            f"SolutionTrackingCallback(iterations={self.iterations}, "
            f"residual={self.residual:.3g})"
        )


# --------------------------------------------------------------------- #
#                            Direct solvers                             #
# --------------------------------------------------------------------- #


_Factors = tuple[
    Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray] | None
]
"""What a direct solver's factorization yields: ``(apply_inverse,
apply_transposed)``, the second ``None`` when the factors cannot give it."""


class DirectSolver(LinearSolver):
    """A solver that factorizes a matrix representation.

    These need coordinates, and they need the *right* representation.
    Self-adjointness shows up as matrix symmetry only in the Galerkin form, so
    a symmetric factorization must be handed that one. Each solver names the
    form it factorizes in its ``form`` attribute, and the conversion between
    forms is done here, so a caller never passes one.

    The matrix is extracted and factorized on the first solve, not when the
    solver is applied to the operator, and the factors are then kept and
    shared by every later solve, by the adjoint and by ``matrix()``.
    """

    requires_coordinates: ClassVar[bool] = True
    form: ClassVar[str] = "components"
    transposes: ClassVar[bool] = False
    """Whether :meth:`_factorize` returns a transposed solve as well."""

    def __init__(self, /, *, n_jobs: int | None = None) -> None:
        """
        Args:
            n_jobs: workers for extracting the matrix when the operator has
                to be probed for it, one column per worker. An operator that
                knows its matrix -- built from one, or a sum or composition
                of such -- is read, and the setting is not used. Serial by
                default; see :mod:`pygeoinf.parallel`.
        """
        self._n_jobs = n_jobs

    @property
    def n_jobs(self) -> int | None:
        """The workers for extracting a matrix that has to be probed."""
        return self._n_jobs

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        domain: CoordinateSpace = operator.domain
        codomain: CoordinateSpace = operator.codomain

        # The matrix is extracted and factorized on the first solve, not
        # here. ``solver(A)`` is then free, and an inverse that is built and
        # never applied -- an inversion constructed to be reduced, swept or
        # made a surrogate of, a preconditioner that is compared but not
        # used -- costs nothing. Once done the factors are kept, so the
        # O(n^3) step is paid once however many solves follow.
        factors: list[_Factors] = []

        def factorized() -> _Factors:
            if not factors:
                matrix = operator.matrix(form=self.form, n_jobs=self._n_jobs)
                factors.append(self._factorize(matrix))
            return factors[0]

        def solve_fn(y, x0):
            apply_inverse, _ = factorized()
            cy = codomain.to_components(y)
            if self.form == "galerkin":
                # M c_x == G_Y c_y, since M == G_Y A_c.
                cy = codomain.apply_gram(cy)
            cx = apply_inverse(cy)
            return SolveResult(domain.from_components(cx), 0, 0.0, True)

        def known_matrix(form: str) -> np.ndarray:
            # (A^-1)_c is M^-1 for M the components form, and M^-1 G_Y for M
            # the Galerkin form G_Y A_c: the factors applied to a matrix.
            apply_inverse, _ = factorized()
            if self.form == "components":
                inverse = apply_inverse(np.identity(domain.dim))
            else:
                inverse = apply_inverse(codomain.gram_matrix())
            if form == "components":
                return inverse
            return domain.apply_gram_to_columns(inverse)

        def components_action(cy: np.ndarray) -> np.ndarray:
            apply_inverse, _ = factorized()
            if self.form == "galerkin":
                cy = codomain.apply_gram(cy)
            return apply_inverse(cy)

        # Whether the factorization will offer a transposed solve is a
        # property of the solver, known before it runs.
        adjoint_solve_fn = None
        components_adjoint_action = None
        if self.transposes:

            def components_adjoint_action(cx: np.ndarray) -> np.ndarray:
                _, apply_transposed = factorized()
                cw = apply_transposed(domain.apply_gram(cx))
                if self.form == "components":
                    cw = codomain.solve_gram(cw)
                return cw

            def adjoint_solve_fn(x, w0):
                # Solve A* w == x for w in the codomain. With M the components
                # form, A* has component matrix G_X^-1 M^T G_Y, so
                # c_w == G_Y^-1 M^-T G_X c_x; with M the Galerkin form the
                # trailing G_Y is already in M and the last solve drops out.
                _, apply_transposed = factorized()
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
    def _factorize(self, matrix: np.ndarray) -> _Factors:
        """Factorize **once**, returning ``(apply_inverse, apply_transposed)``.

        Both accept a vector or a matrix of columns. The second applies
        ``M^-T`` from the same factors and is ``None`` when the factorization
        cannot, which means "invert the adjoint from scratch"; a symmetric
        factorization needs no second: its operator is self-adjoint, so the
        inverse is too and its adjoint is itself.

        One call returns both, so that the transposed solve shares the
        factorization rather than repeating it.
        """


class LUSolver(DirectSolver):
    """LU factorization. Makes no structural demands beyond squareness."""

    form: ClassVar[str] = "components"
    transposes: ClassVar[bool] = True

    def _factorize(self, matrix: np.ndarray) -> _Factors:
        factor = lu_factor(matrix)
        # ``trans=1`` is M^-T from the same factors: the adjoint of an LU
        # inverse costs no second factorization.
        return (lambda c: lu_solve(factor, c), lambda c: lu_solve(factor, c, trans=1))


class CholeskySolver(DirectSolver):
    """Cholesky factorization of the Galerkin matrix.

    Requires positive-definiteness, which by closure implies self-adjointness —
    and the Galerkin form is what makes that visible as symmetry.
    """

    requires: ClassVar[Traits] = Traits.POSITIVE_DEFINITE
    form: ClassVar[str] = "galerkin"

    def _factorize(self, matrix: np.ndarray) -> _Factors:
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

    def _factorize(self, matrix: np.ndarray) -> _Factors:
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
#                     Solvers for a family of operators                 #
# --------------------------------------------------------------------- #


class BoundFamily(ABC):
    """A family ``base + t * shift`` with its multiplier-independent work done.

    What :meth:`FamilySolver.for_family` returns. Whatever is expensive and
    does not depend on the multiplier -- a factorization, a probed diagonal,
    a deflation -- has been computed, and :meth:`at` answers each member of
    the family from it.
    """

    @abstractmethod
    def at(self, multiplier: float, member: LinearOperator, /) -> InverseOperator:
        """The inverse of the member at one multiplier, exact or approximate.

        Args:
            multiplier: ``t``.
            member: the member itself, ``base + t * shift`` as the sweep
                assembled it. It is what the returned operator inverts, and
                it is where a preconditioner reads the factors it needs on
                the first call.

        Returns:
            The inverse, applied as an operator or through ``solve``.
        """


class FamilySolver(LinearSolver):
    """A solver, or a preconditioner, that serves a whole family at once.

    Applied to one operator it is an ordinary :class:`LinearSolver`. Handed
    a family ``base + t * shift`` through :meth:`for_family` it does its
    expensive work once and answers every multiplier from that, which is
    what a sweep over a damped family needs from it: an eigendecomposition
    reused across the sweep, a probed diagonal updated by the shift's own,
    a deflation whose eigenvalues move with the multiplier.
    :class:`~pygeoinf.numerics.root_find.DampedSolves` asks this of its
    solver and of its solver's preconditioner, and rebuilds at intervals
    the ones that cannot answer.
    """

    @abstractmethod
    def for_family(self, base: LinearOperator, shift: LinearOperator, /) -> BoundFamily:
        """Prepare for ``base + t * shift``, doing the multiplier-independent work.

        Args:
            base: the undamped part.
            shift: what the multiplier scales.

        Returns:
            The prepared family.
        """


class SpectralFamilySolver(EigenSolver, FamilySolver):
    """Factor a family ``base + t * shift`` once; solve every member by spectral division.

    With ``S == L L^T`` the Galerkin matrix of the shift and ``W == L^-1``,
    the whitened base ``W B W^T == V diag(lambda) V^T`` is diagonalized once,
    and the member at ``t`` is inverted as ``W^T V diag(1 / (lambda + t))
    V^T W``: one factorization at the cost of a dense eigendecomposition in
    the family's dimension, then a solve at any ``t`` for two matrix-vector
    products, with nothing refactorized as the multiplier moves. That is the
    right tool for a root find that visits tens of multipliers on a family
    whose dimension is small enough to factor, and the wrong one when it is
    not: it is dense in the family's dimension, and it is used only when
    passed as a solver, never assumed.

    Applied to a single operator it is an :class:`EigenSolver`. Eigenvalues
    ``lambda + t`` at or below ``rtol`` times the largest in magnitude are
    treated as zero and their directions dropped, so at ``t == 0`` a
    singular base is inverted in the pseudo-inverse sense, the minimum-norm
    solution in the shift's inner product.
    """

    def for_family(
        self, base: LinearOperator, shift: LinearOperator, /
    ) -> "_SpectralFamily":
        """Whiten by the shift and diagonalize the base, once.

        Args:
            base: the undamped part, self-adjoint.
            shift: what the multiplier scales, claiming positive definiteness.

        Returns:
            The prepared family.

        Raises:
            ValueError: if the two operators are not on one space, if the base
                does not claim self-adjointness, or if the shift does not
                claim to be positive definite.
        """
        if not (base.domain == base.codomain == shift.domain == shift.codomain):
            raise ValueError(
                "A family's base and shift must both be operators on one space."
            )
        if not Traits.SELF_ADJOINT & base.traits:
            raise ValueError(
                f"{type(self).__name__} diagonalizes the base, which must claim "
                f"self-adjointness; it claims {base.traits!s}."
            )
        if not Traits.POSITIVE_DEFINITE & shift.traits:
            raise ValueError(
                f"{type(self).__name__} whitens by the shift, which must claim "
                f"positive definiteness; it claims {shift.traits!s}."
            )
        require_coordinates(base.domain, base.codomain)
        base_matrix = base.matrix(form="galerkin", n_jobs=self._n_jobs)
        shift_matrix = shift.matrix(form="galerkin", n_jobs=self._n_jobs)
        base_matrix = 0.5 * (base_matrix + base_matrix.T)
        shift_matrix = 0.5 * (shift_matrix + shift_matrix.T)
        lower = cholesky(shift_matrix, lower=True)
        # W B W^T with W == L^-1, by two triangular solves.
        half = solve_triangular(lower, base_matrix, lower=True)
        whitened = solve_triangular(lower, half.T, lower=True)
        values, vectors = eigh(0.5 * (whitened + whitened.T))
        # Z == W^T V == L^-T V, so that the inverse is Z diag(.) Z^T G.
        basis = solve_triangular(lower, vectors, lower=True, trans="T")
        return _SpectralFamily(self, base.domain, values, basis, self._rtol)


class _SpectralFamily(BoundFamily):
    """The eigenpairs of a whitened family, and the inverse they give at any ``t``."""

    def __init__(
        self,
        solver: SpectralFamilySolver,
        space: CoordinateSpace,
        values: np.ndarray,
        basis: np.ndarray,
        rtol: float,
    ) -> None:
        self._solver = solver
        self._space = space
        self._values = values
        self._basis = basis
        self._rtol = rtol

    @property
    def eigenvalues(self) -> np.ndarray:
        """The eigenvalues of the base in the shift's inner product."""
        return self._values

    def _inverted(self, multiplier: float) -> np.ndarray:
        shifted = self._values + multiplier
        largest = float(np.max(np.abs(shifted))) if shifted.size else 0.0
        return np.where(np.abs(shifted) > self._rtol * largest, 1.0 / shifted, 0.0)

    def at(self, multiplier: float, member: LinearOperator, /) -> InverseOperator:
        """The member's inverse by spectral division.

        Args:
            multiplier: ``t``.
            member: the member at that multiplier, on the family's space.

        Returns:
            Its inverse.

        Raises:
            ValueError: if the member is not on the space the family was
                prepared on.
        """
        if member.domain != self._space or member.codomain != self._space:
            raise ValueError(
                f"The member must be an operator on {self._space!r}, the space "
                "the family was prepared on."
            )
        space, basis = self._space, self._basis
        inverted = self._inverted(multiplier)

        def components_action(cy: np.ndarray) -> np.ndarray:
            return basis @ (inverted * (basis.T @ space.apply_gram(cy)))

        def solve_fn(y, x0):
            cx = components_action(space.to_components(y))
            return SolveResult(space.from_components(cx), 0, 0.0, True)

        def known_matrix(form: str) -> np.ndarray:
            inverse = basis @ ((inverted[:, None] * basis.T) @ space.gram_matrix())
            if form == "components":
                return inverse
            return space.apply_gram_to_columns(inverse)

        return InverseOperator(
            member,
            self._solver,
            solve_fn,
            known_matrix=known_matrix,
            components_action=components_action,
            components_adjoint_action=components_action,
        )


# --------------------------------------------------------------------- #
#                          Iterative solvers                            #
# --------------------------------------------------------------------- #


class _ComponentOperator(LinearOperator):
    """An operator seen on the component views of its spaces.

    Acts on component arrays. Where the operator can act on components
    directly it does, and a product of such operators stays in components
    end to end; otherwise the array is synthesized, the operator applied,
    and the image analyzed -- which is what the operator would have cost
    anyway. Traits carry over: they are statements about the metric, and the
    view has the same one.
    """

    def __init__(self, operator: LinearOperator, domain: Any, codomain: Any, /) -> None:
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
    arrays through a :class:`~pygeoinf.algebra.spaces.ComponentView`, whose
    inner product is the space's own, so the residual history is unchanged
    to rounding. The vector algebra then costs no transforms; an operator
    that can act on components costs none either, and one that cannot is
    applied to the synthesized vector and its image analyzed, which is what
    applying it costs anyway.
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
        max_iterations: int | None = None,
        preconditioner: LinearSolver | LinearOperator | None = None,
        strict: bool = True,
        callback: Callable[[SolveStep], None] | None = None,
    ) -> None:
        """
        Args:
            rtol: stop when the residual norm falls below ``rtol * ||b||``,
                measured in the space's own norm and relative to ``||b||``
                rather than to ``||b - A x0||``. A tighter tolerance costs
                more iterations for accuracy an inverse problem rarely uses,
                and on an ill-conditioned normal operator may be unreachable,
                which with ``strict=True`` raises.
            atol: absolute floor on the same test.
            max_iterations: iteration cap. Defaults to ``max(2 * dim, 20)``: a Krylov
                method terminates within ``dim`` steps in exact arithmetic, but
                rounding routinely costs a step or two more.
            preconditioner: either a ready-made approximate inverse, or a
                ``LinearSolver`` from which to build one once the operator is
                known.
            strict: raise :class:`ConvergenceError` on failure to converge
                rather than warning.
            callback: called with a :class:`SolveStep` after each step,
                carrying the iteration count, the residual norm, and the
                iterate on request. For watching a long solve, and for
                finding out *where* a stalled one stalled — which the final
                residual alone cannot say. Every iterative solver here honors
                it, and records the same residuals in
                :attr:`SolveResult.history`. :class:`ProgressCallback` and
                :class:`SolutionTrackingCallback` are ready-made ones.
        """
        self._rtol = rtol
        self._atol = atol
        self._max_iterations = max_iterations
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

        **Refuses to replace one that is already set**, since silently keeping
        either would discard the one the caller meant. A library routine that
        means "supply one only if the caller did not" should check
        :attr:`preconditioner` first.

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
        a good trade is the caller's judgment — a preconditioner is an
        approximation, so reusing one across a family costs accuracy, never
        correctness.
        """
        if self._preconditioner is None or not isinstance(
            self._preconditioner, LinearSolver
        ):
            return self
        return self.with_resolved_preconditioner(self._preconditioner(operator))

    def with_resolved_preconditioner(
        self, preconditioner: LinearOperator, /
    ) -> "IterativeSolver":
        """A copy of this solver with a *resolved* preconditioner in place.

        For a routine that has already built the preconditioner for the
        operator at hand -- a family sweep asking a family-aware one for the
        member -- where :meth:`with_preconditioner` would refuse to replace
        the deferred one the caller attached.

        Args:
            preconditioner: the preconditioner, as an operator.

        Returns:
            The copy.
        """
        clone = copy.copy(self)
        clone._preconditioner = preconditioner
        return clone

    @property
    def rtol(self) -> float:
        """The relative residual tolerance the solver stops at."""
        return self._rtol

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
        if self._max_iterations is not None:
            return self._max_iterations
        return max(2 * operator.domain.dim, 20)

    def _record(
        self,
        iteration: int,
        residual: float,
        history: list,
        form: Callable[[], Any],
    ) -> None:
        """Note one step's residual, and tell the callback about it.

        ``form`` produces an independent copy of the current iterate, and is
        called only if the callback asks for it (see :class:`SolveStep`).

        A non-finite residual is a breakdown, not slow convergence: something
        returned NaN or inf, and every later step is arithmetic on it. It is
        refused here, once for every solver, and regardless of ``strict``,
        which downgrades a solve that was *slow* and not one that was
        poisoned. Left to the residual test, which a NaN never passes, the
        solve would run to its iteration cap and report non-convergence, with
        the cause unmentioned and the cap's worth of applications spent.
        """
        _record_residual(
            type(self).__name__, iteration, residual, history, self._callback, form
        )

    def _finish(self, result: SolveResult) -> SolveResult:
        return _finish_solve(type(self).__name__, result, self._strict, "residual")

    def _lifting(self, domain: HilbertSpace) -> "IterativeSolver":
        """This solver, with its callback handed iterates of *domain* rather
        than their components: a shallow copy, so that a solver shared
        between threads is not altered, for a solve run on the views."""
        if self._callback is None:
            return self
        callback = self._callback

        def lifted(step: SolveStep) -> None:
            callback(
                SolveStep(
                    step.iteration,
                    step.residual,
                    lambda: domain.from_components(step.iterate),
                )
            )

        solver = copy.copy(self)
        solver._callback = lifted
        return solver

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        preconditioner = self._resolve_preconditioner(operator)
        if preconditioner is not None and not (
            preconditioner.domain == operator.codomain
            and preconditioner.codomain == operator.domain
        ):
            raise ValueError(
                f"A preconditioner approximates the inverse, so it maps "
                f"{operator.codomain!r} to {operator.domain!r}; this one maps "
                f"{preconditioner.domain!r} to {preconditioner.codomain!r}."
            )
        solve_fn = self._solve_fn(operator, preconditioner)

        # The adjoint system ``A* w == x`` is preconditioned by ``P*``, from
        # the *same* resolved ``P``, and built only when first needed. Handing
        # ``A*`` back to this solver instead would precondition it with ``P``
        # itself when ``P`` is a fixed operator -- the right answer, reached
        # slowly -- and, for a deferred direct preconditioner, factorize ``A*``
        # a second time when ``P*`` is one transposed solve away.
        adjoint_solve_fn = None
        if preconditioner is not None:
            adjoint_solve = None

            def adjoint_solve_fn(x, w0):
                nonlocal adjoint_solve
                if adjoint_solve is None:
                    adjoint_solve = self._solve_fn(
                        operator.adjoint, preconditioner.adjoint
                    )
                return adjoint_solve(x, w0)

        return InverseOperator(
            operator, self, solve_fn, adjoint_solve_fn=adjoint_solve_fn
        )

    def _solve_fn(
        self, operator: LinearOperator, preconditioner: LinearOperator | None
    ) -> Callable[[Any, Any | None], SolveResult]:
        """``(y, x0) -> SolveResult`` for one operator and its preconditioner."""
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
            result = self._lifting(domain)._solve(
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

        return solve_fn

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
    self-adjointness. The requirement is checked against the operator's
    traits: the Bayesian normal operator ``A Q A* + R`` carries them by
    construction, so CG accepts it with nothing asserted, and an operator that
    does not claim them is refused. A direction of non-positive curvature met
    during the iteration raises :class:`ConvergenceError`, since it means the
    claim was false.
    """

    requires: ClassVar[Traits] = Traits.POSITIVE_DEFINITE

    @staticmethod
    def _refuse_preconditioner(value: float, iteration: int) -> None:
        """``(r, P r)`` was not positive: the preconditioner is not.

        The recurrence divides by it, and its theory needs it positive. The
        cause is the preconditioner, not the operator: a non-positive-definite
        operator is caught first, by the curvature check.
        """
        if not np.isfinite(value):
            raise ConvergenceError(
                f"CG met a non-finite preconditioned product ((r, P r) == "
                f"{value}) at iteration {iteration}. The preconditioner "
                f"returned NaN or inf, or the problem needs rescaling; this is "
                f"a breakdown, so strict=False does not downgrade it."
            )
        raise ConvergenceError(
            f"CG met a non-positive preconditioned product "
            f"((r, P r) == {value:g}) at iteration {iteration}. The "
            f"preconditioner is not positive definite, which conjugate "
            f"gradients requires of it; verify it with testing.check_traits()."
        )

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
        self._record(0, residual, history, lambda: space.copy(x))
        if residual <= tolerance:
            return SolveResult(x, 0, residual, True, tuple(history))

        z = r if preconditioner is None else preconditioner(r)
        p = space.copy(z)
        # Without a preconditioner (r, z) is (r, r), which the norm just gave.
        rz = residual**2 if preconditioner is None else space.inner_product(r, z)
        if not rz > 0.0:
            self._refuse_preconditioner(rz, 0)

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
            self._record(iteration, residual, history, lambda: space.copy(x))
            if residual <= tolerance:
                return SolveResult(x, iteration, residual, True, tuple(history))

            z = r if preconditioner is None else preconditioner(r)
            rz_next = (
                residual**2 if preconditioner is None else space.inner_product(r, z)
            )
            if rz_next < 0.0:
                self._refuse_preconditioner(rz_next, iteration)
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
    the iteration proceeds, which is what a localized preconditioner does — the
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
        self._record(0, residual, history, lambda: space.copy(x))
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
            self._record(iteration, residual, history, lambda: space.copy(x))
            if residual <= tolerance:
                return SolveResult(x, iteration, residual, True, tuple(history))

            z = r if preconditioner is None else preconditioner(r)
            # Polak-Ribiere: (r_new - r_old, z_new) rather than (r_new, z_new).
            # The two agree when the preconditioner is fixed, because then
            # successive residuals are orthogonal in the preconditioner's inner
            # product; they differ exactly when it is not.
            change = space.subtract(r, previous)
            rz_next = space.inner_product(r, z)
            beta = space.inner_product(change, z) / rz
            rz = rz_next
            p = space.scale_inplace(beta, p)
            p = space.axpy(1.0, z, p)

        return SolveResult(x, self._limit(operator), residual, False, tuple(history))


# A Krylov space has closed when the new direction left after
# orthogonalization is this small against the image it was taken from: the
# rounding level of the orthogonalization, relative, so the test does not
# depend on the units of the operator.
_CLOSURE = 1e-13


def _triangular_weights(triangle: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """``R y == g`` for the upper-triangular factor of a Krylov least squares.

    By back substitution when ``R`` is nonsingular, and by least squares
    when a diagonal entry is at rounding level against the largest entry:
    ``R`` is singular exactly when the operator is singular on the Krylov
    space, and back substitution then returns numbers of order one over
    rounding error where the least-squares solution is the minimum the
    method promises.

    Args:
        triangle: ``R``, upper triangular and square.
        rhs: ``g``.

    Returns:
        ``y``.
    """
    size = triangle.shape[0]
    largest = float(np.max(np.abs(triangle))) if size else 0.0
    smallest = float(np.min(np.abs(np.diag(triangle)))) if size else 0.0
    eps = float(np.finfo(float).eps)
    if largest > 0.0 and smallest > size * eps * largest:
        return solve_triangular(triangle, rhs)
    return np.linalg.lstsq(triangle, rhs, rcond=None)[0]


class GMRESSolver(IterativeSolver):
    """Generalized minimal residual, for an operator with no symmetry at all.

    Like :class:`BiCGStabSolver`, it asks nothing of its operator. Arnoldi
    builds an orthonormal Krylov basis and the residual is minimized over it
    by a small least-squares problem, kept triangular by Givens rotations
    applied as each column arrives -- so the residual norm is known at every
    step without forming the iterate.

    Restarted, because the cost and storage of a step grow with the step
    number: ``restart`` vectors are held, then the basis is discarded and the
    iteration begins again from the current residual.

    **Preconditioned from the right**, ``A P u == b`` with ``x == P u``, so
    the residual it minimizes, reports and stops on is the true ``b - A x``
    in the space's own norm, as for the other Krylov solvers here: ``rtol``
    means the same thing with a preconditioner as without one.

    **Convergence is declared on the true residual.** The recurrence's
    residual is what the iteration stops on, but each cycle ends by forming
    ``b - A x`` and testing that, so a cycle whose recurrence has drifted from
    the truth restarts rather than reporting success. A Krylov space that
    closes -- the new Arnoldi vector vanishing against the image it came
    from -- ends the solve with the true residual: on a singular operator
    with a right-hand side outside its range (a mask, a projector, a diagonal
    with a zero) the space closes before the residual can reach the
    tolerance, and the solve reports that rather than convergence. The small
    triangular system is solved by least squares when it is singular, which
    is the minimum over the Krylov space that GMRES promises.
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
        domain, codomain = operator.domain, operator.codomain
        x, r = self._initial(operator, b, x0)
        tolerance = self._tolerance(codomain, b)

        def lift(vector: Any) -> Any:
            """``P v``, the domain vector a Krylov vector stands for."""
            return vector if preconditioner is None else preconditioner(vector)

        history: list[float] = []
        residual = codomain.norm(r)
        self._record(0, residual, history, lambda: domain.copy(x))
        if residual <= tolerance:
            return SolveResult(x, 0, residual, True, tuple(history))

        limit = self._limit(operator)
        total = 0
        while total < limit:
            width = min(self._restart, limit - total)
            start = residual
            basis = [codomain.scale(1.0 / residual, r)]
            hessenberg = np.zeros((width + 1, width))
            cosines, sines = np.zeros(width), np.zeros(width)
            rhs = np.zeros(width + 1)
            rhs[0] = residual
            used = 0
            closed = False

            for column in range(width):
                total += 1
                used = column + 1
                w = operator(lift(basis[column]))
                image = codomain.norm(w)
                for row in range(column + 1):
                    hessenberg[row, column] = codomain.inner_product(basis[row], w)
                    w = codomain.axpy(-hessenberg[row, column], basis[row], w)
                # Kept aside: the rotations below overwrite the subdiagonal
                # entry with zero, which is the point of them, and it is still
                # needed both as the breakdown test and as the next basis
                # vector's normalization.
                subdiagonal = codomain.norm(w)
                hessenberg[column + 1, column] = subdiagonal
                # The space has closed when what is left of the image after
                # orthogonalization is rounding: relative to the image itself,
                # so the test is the same in any units.
                closed = subdiagonal <= _CLOSURE * image

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

                def form(used: int = column + 1) -> Any:
                    # The iterate GMRES would return if it stopped here: the
                    # cycle's own assembly below, on the triangular system as
                    # it stands. Later columns leave rows up to this one alone.
                    weights = _triangular_weights(hessenberg[:used, :used], rhs[:used])
                    step = codomain.zero()
                    for index in range(used):
                        step = codomain.axpy(weights[index], basis[index], step)
                    return domain.add(x, lift(step))

                self._record(total, residual, history, form)
                if residual <= tolerance or closed:
                    break
                if total >= limit:
                    break
                basis.append(codomain.scale(1.0 / subdiagonal, w))

            claimed = residual <= tolerance
            weights = _triangular_weights(hessenberg[:used, :used], rhs[:used])
            step = codomain.zero()
            for index in range(used):
                step = codomain.axpy(weights[index], basis[index], step)
            x = domain.add(x, lift(step))

            # The verdict is the true residual's, never the recurrence's.
            _, r = self._initial(operator, b, x)
            residual = codomain.norm(r)
            if residual <= tolerance:
                return SolveResult(x, total, residual, True, tuple(history))
            if (claimed or closed) and residual > 0.5 * start:
                # The recurrence stopped -- on its tolerance, or because the
                # Krylov space closed -- and the true residual did not follow
                # far. A restart would search inside the same closed space,
                # or meet the same rounding floor: only a cycle that was
                # still gaining is worth repeating, as a refinement.
                break

        return SolveResult(x, total, residual, False, tuple(history))


class MinResSolver(IterativeSolver):
    """MINRES for a self-adjoint, possibly indefinite, operator.

    The Paige-Saunders recurrences: Lanczos tridiagonalization with Givens
    rotations applied on the fly, so the least-squares problem over the Krylov
    space is solved without storing it.

    Preconditioning is supported, and needs a *positive-definite* one: the
    method runs in the inner product ``M`` induces, so an indefinite ``M``
    gives a negative squared norm and is refused with a message saying so
    rather than producing a NaN some iterations later. That is the only extra
    condition -- the operator itself may still be indefinite, which is the
    reason to reach for MINRES over CG in the first place.

    **Convergence is declared on the true residual.** The recurrence's
    residual is what the iteration stops on; the solve then forms
    ``b - A x`` and tests its ``M``-norm against the same tolerance. A
    recurrence that has drifted from the truth starts again from the current
    iterate while that still gains, and a Krylov space that closes -- the
    next Lanczos vector vanishing against the column it came from, as it does
    on a mask, a projector or a diagonal with a zero -- ends the cycle there.
    On a singular operator with a right-hand side outside its range the solve
    therefore reports the residual it reached and does not claim convergence.
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
                # A ConvergenceError, as CG reports its indefinite
                # preconditioner: the two are one kind of failure.
                raise ConvergenceError(
                    "MINRES needs a positive-definite preconditioner: the "
                    f"inner product it induces gave {squared:.3e} for a "
                    "vector's own norm."
                )
            return float(np.sqrt(squared))

        # Relative to the M-norm of the right-hand side, since that is the norm
        # phi_bar below is measured in.
        tolerance = max(self._rtol * m_norm(b, preconditioned(b)), self._atol)

        history: list[float] = []
        z = preconditioned(r)
        beta = m_norm(r, z)
        self._record(0, beta, history, lambda: space.copy(x))
        if beta <= tolerance:
            return SolveResult(x, 0, beta, True, tuple(history))

        limit = self._limit(operator)
        total = 0
        while total < limit:
            start = beta
            x, steps = self._cycle(
                operator,
                preconditioned,
                m_norm,
                x,
                r,
                z,
                beta,
                tolerance,
                limit - total,
                total,
                history,
            )
            total += steps
            # The verdict is the true residual's, never the recurrence's.
            _, r = self._initial(operator, b, x)
            z = preconditioned(r)
            beta = m_norm(r, z)
            if beta <= tolerance:
                return SolveResult(x, total, beta, True, tuple(history))
            if beta > 0.5 * start:
                # The cycle gained little, so another from here would gain
                # less: a closed Krylov space, or a recurrence that has
                # outrun what rounding lets the true residual reach.
                break

        return SolveResult(x, total, beta, False, tuple(history))

    def _cycle(
        self,
        operator: LinearOperator,
        preconditioned: Callable[[Any], Any],
        m_norm: Callable[[Any, Any], float],
        x: Any,
        r: Any,
        z: Any,
        beta: float,
        tolerance: float,
        budget: int,
        offset: int,
        history: list[float],
    ) -> tuple[Any, int]:
        """One run of the recurrences from ``x``, with ``r == b - A x``.

        Stops when the recurrence's residual reaches the tolerance, when the
        Krylov space closes, or when the budget is spent.

        Returns:
            The iterate and the steps taken.
        """
        space = operator.domain

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
        beta_prev = 1.0

        for step in range(1, budget + 1):
            # --- Lanczos step: extend the tridiagonalization ----------
            # The recurrence is carried on the residuals rather than on the
            # M-orthonormal vectors: p is the next unpreconditioned residual,
            # and M p is what gives both the next vector and its norm.
            p = operator(v)
            alpha = space.inner_product(v, p)
            p = space.axpy(-alpha / beta, r_current, p)
            if step > 1:
                p = space.axpy(-beta / beta_prev, r_prev, p)
            z_next = preconditioned(p)
            beta_next = m_norm(p, z_next)

            # The column of the tridiagonal matrix this step added, whose size
            # is the scale the closure is judged against: the space has
            # closed when the next off-diagonal is rounding against it.
            column = float(
                np.hypot(np.hypot(alpha, beta if step > 1 else 0.0), beta_next)
            )
            closed = beta_next <= _CLOSURE * column

            # --- apply the previous rotation to the new column --------
            delta = c * delta_bar + s * alpha
            gamma_bar = s * delta_bar - c * alpha
            epsilon_next = s * beta_next
            delta_bar_next = -c * beta_next

            # --- the new rotation, annihilating beta_next -------------
            gamma = float(np.hypot(gamma_bar, beta_next))
            if gamma <= _CLOSURE * column:
                # The projected system is singular on a closed space: the
                # update would divide by rounding error. The iterate stands.
                return x, step
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

            # phi_bar is the recurrence's residual norm, available without
            # forming it; the caller checks it against the true one.
            self._record(offset + step, abs(phi_bar), history, lambda: space.copy(x))
            if abs(phi_bar) <= tolerance or closed:
                return x, step

            # --- roll the state ---------------------------------------
            w_prev, w = w, w_next
            v = space.scale(1.0 / beta_next, z_next)
            r_prev, r_current = r_current, p
            beta_prev, beta = beta, beta_next
            delta_bar = delta_bar_next
            epsilon = epsilon_next

        return x, budget


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
        self._record(0, residual, history, lambda: space.copy(x))
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
                self._record(iteration, space.norm(s), history, lambda: space.copy(x))
                return SolveResult(x, iteration, space.norm(s), True, tuple(history))

            z = s if preconditioner is None else preconditioner(s)
            t = operator(z)
            tt = space.inner_product(t, t)
            omega = 0.0 if tt == 0.0 else space.inner_product(t, s) / tt

            x = space.axpy(alpha, y, x)
            x = space.axpy(omega, z, x)
            r = space.axpy(-omega, t, s)

            residual = space.norm(r)
            self._record(iteration, residual, history, lambda: space.copy(x))
            if residual <= tolerance:
                return SolveResult(x, iteration, residual, True, tuple(history))

        return SolveResult(x, self._limit(operator), residual, False, tuple(history))


# --------------------------------------------------------------------- #
#                          Least squares                                #
# --------------------------------------------------------------------- #


class LeastSquaresSolver(ABC):
    """Solves a rectangular system in the least-squares sense.

    A **sibling** of :class:`LinearSolver`, not a subclass: solving
    ``A x == b`` and minimizing ``||A x - b||`` are different operations and
    get different types, and a least-squares solver makes no squareness check.
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
    """Golub-Kahan bidiagonalization, applied to the normal equations implicitly.

    Coordinate-free, and damped when asked: with ``damping > 0`` it minimizes
    ``||A x - b||^2 + damping^2 ||x||^2``, both norms being the spaces' own.
    """

    def __init__(
        self,
        /,
        *,
        damping: float = 0.0,
        rtol: float = 1e-10,
        max_iterations: int | None = None,
        strict: bool = True,
        callback: Callable[[SolveStep], None] | None = None,
    ) -> None:
        """
        Args:
            damping: minimize ``||A x - b||^2 + damping^2 ||x||^2`` when
                positive. See :meth:`_solve` on what this means with a warm
                start.
            rtol: relative tolerance, applied to both the residual and the
                normal residual.
            max_iterations: iteration cap. Four times the domain dimension by default.
            strict: raise :class:`ConvergenceError` rather than warn with a
                ``RuntimeWarning`` when the cap is reached.
            callback: called with a :class:`SolveStep` each step, whose
                residual is the *normal* residual; the same numbers are kept
                in :attr:`SolveResult.history`.

        Raises:
            ValueError: if the damping is negative.
        """
        if damping < 0.0:
            raise ValueError("damping must be non-negative.")
        self._damping = damping
        self._rtol = rtol
        self._max_iterations = max_iterations
        self._strict = strict
        self._callback = callback

    def _invert(self, operator: LinearOperator) -> LinearOperator:
        def solve_fn(b, x0):
            # Raise or warn as the other iterative solvers do, by ``strict``.
            return _finish_solve(
                type(self).__name__,
                self._solve(operator, b, x0),
                self._strict,
                "normal residual",
            )

        return InverseOperator(operator, self, solve_fn, traits=Traits.NONE)

    def _note(
        self,
        iteration: int,
        residual: float,
        history: list,
        form: Callable[[], Any],
    ) -> None:
        """Note one step's normal residual, and tell the callback about it.

        A non-finite one is refused as a breakdown, as every other iterative
        solver refuses it, rather than being run to the iteration cap.
        """
        _record_residual(
            type(self).__name__, iteration, residual, history, self._callback, form
        )

    def _solve(self, operator: LinearOperator, b: Any, x0: Any | None) -> SolveResult:
        """The bidiagonalization, started from *x0* when one is given.

        The warm start solves for the *correction*: the iteration runs on the
        residual ``b - A x0`` and accumulates into a copy of ``x0``. With no
        damping that is exactly equivalent to starting
        from zero, so a warm start can only save iterations. With damping it is
        not: the penalty then falls on the correction rather than on the whole
        solution, which is a different minimization. Damped warm starts are
        therefore refused rather than quietly answering a different question.

        Raises:
            ValueError: if both a warm start and a damping are given.
        """
        domain, codomain = operator.domain, operator.codomain
        limit = (
            self._max_iterations
            if self._max_iterations is not None
            else 4 * max(domain.dim, 1)
        )
        history: list[float] = []

        if x0 is not None and self._damping > 0.0:
            raise ValueError(
                "A damped LSQR cannot be warm-started: the iteration would "
                "penalize the correction rather than the solution, which "
                "minimizes something else. Start from zero, or damp by "
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
            # --- Golub-Kahan bidiagonalization step -------------------
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
            self._note(iteration, normal_residual, history, lambda: domain.copy(x))
            if normal_residual <= normal_target or abs(phi_bar) <= residual_target:
                return SolveResult(x, iteration, normal_residual, True, tuple(history))
            if beta == 0.0 or alpha == 0.0:
                return SolveResult(x, iteration, normal_residual, True, tuple(history))

        return SolveResult(x, limit, normal_residual, False, tuple(history))
