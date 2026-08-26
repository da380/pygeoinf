"""
Preconditioners.

A preconditioner is an approximate inverse, so it is a :class:`LinearSolver`
that happens not to be exact. v1 already models it this way and it is right:
nothing else is needed, and the iterative solvers accept either a ready-made
operator or a solver to build one from.
"""

from __future__ import annotations

from typing import ClassVar, Literal, Sequence

import numpy as np
from numpy.random import Generator

from ..algebra.operators import LinearOperator, require_coordinates
from ..algebra.spaces import CoordinateSpace
from ..traits import Traits
from .solvers import InverseOperator, LinearSolver, SolveResult

__all__ = [
    "IdentityPreconditioner",
    "JacobiPreconditioner",
    "SpectralPreconditioner",
    "BandedPreconditioner",
    "BlockPreconditioner",
]


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


class SpectralPreconditioner(LinearSolver):
    """Invert the dominant eigenmodes exactly, and damp the rest.

    A randomised eigendecomposition finds the leading ``rank`` modes; those are
    inverted properly and the unresolved tail is replaced by a single scalar.
    That is the right shape for an operator whose spectrum decays: the modes
    that make it ill-conditioned are exactly the ones the decomposition finds.

    The damping defaults to the smallest resolved eigenvalue, which is the only
    value that makes the preconditioner continuous at the truncation.
    """

    requires: ClassVar[Traits] = Traits.SELF_ADJOINT
    requires_coordinates: ClassVar[bool] = False

    def __init__(
        self,
        /,
        *,
        rank: int = 20,
        damping: float | None = None,
        rng: Generator | None = None,
    ) -> None:
        """
        Args:
            rank: how many eigenmodes to resolve.
            damping: the scalar standing in for the tail. Defaults to the
                smallest resolved eigenvalue.
            rng: the generator for the randomised range finder.
        """
        if rank < 1:
            raise ValueError(f"The rank must be positive, got {rank}.")
        self._rank = rank
        self._damping = damping
        self._rng = rng

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        from .randomised import random_eig

        space = operator.domain
        low_rank = random_eig(operator, rank=self._rank, rng=self._rng)
        values = low_rank.eigenvalues
        floor = self._damping
        if floor is None:
            positive = values[values > 0.0]
            floor = float(positive.min()) if positive.size else 1.0
        if floor <= 0.0:
            raise ValueError(f"The damping must be positive, got {floor}.")

        factor = low_rank.factor
        # (1/d) I  +  U (1/lambda - 1/d) U*, which inverts the resolved modes
        # and leaves everything else divided by the damping.
        adjustment = 1.0 / np.where(values > 0.0, values, floor) - 1.0 / floor

        def solve_fn(y, x0):
            coefficients = factor.adjoint(y)
            corrected = factor(adjustment * coefficients)
            return SolveResult(
                space.add(space.scale(1.0 / floor, y), corrected), 1, 0.0, True
            )

        return InverseOperator(operator, self, solve_fn, traits=Traits.SELF_ADJOINT)


class BandedPreconditioner(LinearSolver):
    """Keep a band of the operator's matrix and factor it.

    The cheapest structural approximation there is: an operator whose
    correlations are local has a matrix that is nearly banded in a basis
    ordered by position, and the band is what a sparse factorisation can
    invert quickly.

    **Only worth it when the operator has that structure.** On a dense operator
    a tridiagonal approximation is not a poor preconditioner but a harmful one:
    measured on a random dense positive-definite operator it took conjugate
    gradients from 125 iterations to failure. Nothing here can detect that for
    you, which is why the bandwidth is a required argument rather than a
    default.
    """

    requires: ClassVar[Traits] = Traits.NONE
    requires_coordinates: ClassVar[bool] = True

    def __init__(
        self,
        bandwidth: int,
        /,
        *,
        form: Literal["auto", "components", "galerkin"] = "auto",
        probe: Literal["exact", "banded"] = "exact",
    ) -> None:
        """
        Args:
            bandwidth: sub- and super-diagonals to keep on each side. One gives
                a tridiagonal preconditioner.
            form: which matrix representation to band.
            probe: how to extract the diagonals. ``"exact"`` by default even
                though the result is an approximation either way: the fast
                probe sums the out-of-band entries into the band, and on an
                operator that is *not* banded that turns a merely unhelpful
                preconditioner into an actively harmful one. Use ``"banded"``
                when the operator really is banded, where the two agree.
        """
        if bandwidth < 0:
            raise ValueError(f"The bandwidth must be non-negative, got {bandwidth}.")
        self._bandwidth = bandwidth
        self._form = form
        self._probe = probe

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        from scipy.sparse import dia_array
        from scipy.sparse.linalg import splu

        space = operator.domain
        offsets = list(range(-self._bandwidth, self._bandwidth + 1))
        diagonals = operator.diagonals(
            offsets=offsets, form=self._form, probe=self._probe
        )
        banded = dia_array((diagonals, offsets), shape=(space.dim, space.dim)).tocsc()
        factorisation = splu(banded)
        galerkin = self._form == "galerkin" or (
            self._form == "auto" and Traits.SELF_ADJOINT & operator.traits
        )

        def solve_fn(y, x0):
            components = space.to_components(y)
            if galerkin:
                components = space.apply_gram(components)
            return SolveResult(
                space.from_components(factorisation.solve(components)), 1, 0.0, True
            )

        return InverseOperator(operator, self, solve_fn, traits=Traits.SELF_ADJOINT)


class BlockPreconditioner(LinearSolver):
    """Invert exactly within groups, and ignore everything between them.

    Given a partition of the components — from
    :meth:`~pygeoinf2.symmetric_space.sphere.Sphere.cluster_points`, or any
    other grouping that reflects locality — this factors each diagonal block
    and drops the rest. It is the natural preconditioner when the operator's
    structure is *clustered* rather than banded, which is what a real
    acquisition geometry gives.
    """

    requires: ClassVar[Traits] = Traits.NONE
    requires_coordinates: ClassVar[bool] = True

    def __init__(
        self,
        blocks: Sequence[Sequence[int]],
        /,
        *,
        form: Literal["auto", "components", "galerkin"] = "auto",
    ) -> None:
        """
        Args:
            blocks: index groups, which must partition the components.
            form: which matrix representation to take blocks of.
        """
        self._blocks = [np.asarray(block, dtype=int) for block in blocks]
        if not self._blocks:
            raise ValueError("At least one block is needed.")
        self._form = form

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        space = operator.domain
        covered = np.concatenate(self._blocks)
        if covered.size != space.dim or set(covered.tolist()) != set(range(space.dim)):
            raise ValueError(
                f"The blocks must partition all {space.dim} components; they "
                f"cover {covered.size} indices."
            )
        matrix = operator.matrix(form=self._form)
        factors = [
            np.linalg.inv(matrix[np.ix_(block, block)]) for block in self._blocks
        ]
        galerkin = self._form == "galerkin" or (
            self._form == "auto" and Traits.SELF_ADJOINT & operator.traits
        )

        def solve_fn(y, x0):
            components = space.to_components(y)
            if galerkin:
                components = space.apply_gram(components)
            result = np.zeros(space.dim)
            for block, inverse in zip(self._blocks, factors):
                result[block] = inverse @ components[block]
            return SolveResult(space.from_components(result), 1, 0.0, True)

        return InverseOperator(operator, self, solve_fn, traits=Traits.SELF_ADJOINT)
