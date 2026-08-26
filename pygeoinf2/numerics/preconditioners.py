"""
Preconditioners.

A preconditioner is an approximate inverse, so it is a :class:`LinearSolver`
that happens not to be exact. v1 already models it this way and it is right:
nothing else is needed, and the iterative solvers accept either a ready-made
operator or a solver to build one from.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from ..algebra.operators import LinearOperator, require_coordinates
from ..algebra.spaces import CoordinateSpace
from ..traits import Traits
from .solvers import InverseOperator, LinearSolver, SolveResult

__all__ = ["IdentityPreconditioner", "JacobiPreconditioner"]


class IdentityPreconditioner(LinearSolver):
    """Does nothing, usefully: a baseline and a null object."""

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        def solve_fn(y, x0):
            return SolveResult(y, 0, 0.0, True)

        return InverseOperator(operator, self, solve_fn, traits=Traits.NONE)


class JacobiPreconditioner(LinearSolver):
    """Inverts the diagonal of the Galerkin matrix.

    Needs coordinates, since a diagonal is a statement about a basis. The
    Galerkin form is the right one for the same reason the direct solvers use
    it: it is the representation in which a self-adjoint operator is symmetric,
    so its diagonal is the one a symmetric preconditioner wants.
    """

    requires: ClassVar[Traits] = Traits.SELF_ADJOINT
    requires_coordinates: ClassVar[bool] = True

    def __init__(self, /, *, floor: float = 1e-14) -> None:
        self._floor = floor

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        domain: CoordinateSpace = operator.domain
        codomain: CoordinateSpace = operator.codomain
        require_coordinates(domain, codomain)

        diagonal = np.diag(operator.matrix(form="galerkin"))
        safe = np.where(np.abs(diagonal) > self._floor, diagonal, 1.0)
        inverse_diagonal = 1.0 / safe

        def solve_fn(y, x0):
            cy = codomain.apply_gram(codomain.to_components(y))
            return SolveResult(
                domain.from_components(inverse_diagonal * cy), 0, 0.0, True
            )

        return InverseOperator(operator, self, solve_fn, traits=Traits.SELF_ADJOINT)
