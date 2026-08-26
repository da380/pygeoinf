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
from ..algebra.spaces import CoordinateSpace, HilbertSpace
from ..traits import Traits
from .solvers import InverseOperator, LinearSolver, SolveResult

__all__ = [
    "IdentityPreconditioner",
    "JacobiPreconditioner",
    "SpectralPreconditioner",
    "BandedPreconditioner",
    "BlockPreconditioner",
    "WoodburyPreconditioner",
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


class WoodburyPreconditioner(LinearSolver):
    """Invert a normal operator through the *other* space.

    The two normal operators of a linear Gaussian problem are

    .. code-block:: text

        N_m == Q^-1 + A* R^-1 A        on the model space
        N_d == A Q A* + R              on the data space

    and the Woodbury identity writes each inverse in terms of a solve in the
    opposite space:

    .. code-block:: text

        N_m^-1 == Q - Q A* (R + A Q A*)^-1 A Q
        N_d^-1 == R^-1 - R^-1 A (Q^-1 + A* R^-1 A)^-1 A* R^-1

    This is the same trade as :func:`~pygeoinf2.inference.point.choose_formalism`
    — solve wherever the dimension is smaller — but used as a *preconditioner*
    rather than as the solve itself, which is what makes it useful when neither
    space is small enough to settle the matter outright.

    Two things turn it from an identity into a preconditioner:

    * **The inner solve is cheap.** A few iterations, or a spectral
      approximation, in place of an exact inverse.
    * **The pieces are surrogates.** A smoother forward operator, a stationary
      prior standing in for a non-stationary one, a diagonal noise covariance
      standing in for a correlated one. Pass the cheap versions here and use
      the result on the true operator; the closer they are, the better it
      works, and correctness never depends on how close they are.

    The model form needs only *applications* of ``Q`` and ``R``, never their
    inverses, which is why it survives priors whose inverse is unbounded — a
    Sobolev covariance, say. The data form needs ``Q^-1`` and ``R^-1``, so it
    wants covariances that are already given in inverse form.

    Which identity is used is decided by the operator handed to
    :meth:`__call__`: one acts on the model space and the other on the data
    space, and their dimensions say which is which.

    Note:
        With an inexact inner solve the result is no longer exactly symmetric,
        and ordinary conjugate gradients relies on a symmetric preconditioner.
        Use :class:`~pygeoinf2.numerics.solvers.FlexibleCGSolver` when the
        inner solver is itself iterative, or tighten the inner tolerance until
        the asymmetry is below the outer one.
    """

    requires: ClassVar[Traits] = Traits.NONE
    requires_coordinates: ClassVar[bool] = False

    def __init__(
        self,
        forward: LinearOperator,
        prior_covariance: LinearOperator,
        noise_covariance: LinearOperator,
        /,
        *,
        solver: LinearSolver | None = None,
        prior_solver: LinearSolver | None = None,
        noise_solver: LinearSolver | None = None,
        prior_inverse: LinearOperator | None = None,
        noise_inverse: LinearOperator | None = None,
    ) -> None:
        """
        Args:
            forward: the forward operator ``A``, possibly a cheap surrogate.
            prior_covariance: the prior covariance ``Q`` on the model space.
            noise_covariance: the data error covariance ``R``.
            solver: inverts the inner operator. This is the one that should be
                cheap; it defaults to conjugate gradients, which assumes the
                inner operator is positive definite, as both are.
            prior_solver: inverts ``Q``, for the data form only. Defaults to
                *solver*. Ignored when *prior_inverse* is given.
            noise_solver: inverts ``R``, for the data form only. Defaults to
                *solver*. Ignored when *noise_inverse* is given.
            prior_inverse: ``Q^-1`` when it is known in closed form, which is
                usually cheaper and always better conditioned than solving.
            noise_inverse: ``R^-1`` likewise — a diagonal noise covariance
                inverts by hand.
        """
        if forward.domain.dim != prior_covariance.domain.dim:
            raise ValueError(
                f"The prior covariance acts on a space of dimension "
                f"{prior_covariance.domain.dim}, but the forward operator has "
                f"a model space of dimension {forward.domain.dim}."
            )
        if forward.codomain.dim != noise_covariance.domain.dim:
            raise ValueError(
                f"The noise covariance acts on a space of dimension "
                f"{noise_covariance.domain.dim}, but the forward operator has "
                f"a data space of dimension {forward.codomain.dim}."
            )
        self._forward = forward
        self._prior = prior_covariance
        self._noise = noise_covariance
        self._solver = solver
        self._prior_solver = prior_solver
        self._noise_solver = noise_solver
        self._prior_inverse = prior_inverse
        self._noise_inverse = noise_inverse

    @property
    def model_space(self) -> HilbertSpace:
        """The space ``N_m`` acts on."""
        return self._forward.domain

    @property
    def data_space(self) -> HilbertSpace:
        """The space ``N_d`` acts on."""
        return self._forward.codomain

    def _inner_solver(self) -> LinearSolver:
        if self._solver is not None:
            return self._solver
        from .solvers import CGSolver

        return CGSolver()

    def _resolve(
        self,
        inverse: LinearOperator | None,
        operator: LinearOperator,
        solver: LinearSolver | None,
    ) -> LinearOperator:
        if inverse is not None:
            return inverse
        chosen = solver if solver is not None else self._inner_solver()
        return chosen(operator.with_traits(Traits.POSITIVE_DEFINITE))

    def model_form(self) -> LinearOperator:
        """``Q - Q A* (R + A Q A*)^-1 A Q``, an approximate ``N_m^-1``.

        Built without ever inverting ``Q`` or ``R``.
        """
        forward, prior, noise = self._forward, self._prior, self._noise
        inner = noise + forward @ prior @ forward.adjoint
        inverse = self._inner_solver()(inner.with_traits(Traits.POSITIVE_DEFINITE))
        cross = prior @ forward.adjoint
        return (prior - cross @ inverse @ cross.adjoint).with_traits(
            Traits.SELF_ADJOINT
        )

    def data_form(self) -> LinearOperator:
        """``R^-1 - R^-1 A (Q^-1 + A* R^-1 A)^-1 A* R^-1``, an approximate ``N_d^-1``.

        Needs ``Q^-1`` and ``R^-1``, so it wants covariances given in inverse
        form; see the class docstring.
        """
        forward = self._forward
        prior_inverse = self._resolve(
            self._prior_inverse, self._prior, self._prior_solver
        )
        noise_inverse = self._resolve(
            self._noise_inverse, self._noise, self._noise_solver
        )
        inner = prior_inverse + forward.adjoint @ noise_inverse @ forward
        inverse = self._inner_solver()(inner.with_traits(Traits.POSITIVE_DEFINITE))
        cross = noise_inverse @ forward
        return (noise_inverse - cross @ inverse @ cross.adjoint).with_traits(
            Traits.SELF_ADJOINT
        )

    def _validate(self, operator: LinearOperator) -> None:
        super()._validate(operator)
        dim = operator.domain.dim
        if dim not in (self.model_space.dim, self.data_space.dim):
            raise ValueError(
                f"WoodburyPreconditioner was built for a model space of "
                f"dimension {self.model_space.dim} and a data space of "
                f"dimension {self.data_space.dim}; it was asked to invert an "
                f"operator on a space of dimension {dim}, which is neither."
            )

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        dim = operator.domain.dim
        if dim == self.model_space.dim and dim == self.data_space.dim:
            # Ambiguous by dimension alone, so ask the spaces themselves.
            model = operator.domain == self.model_space
        else:
            model = dim == self.model_space.dim
        approximate = self.model_form() if model else self.data_form()

        def solve_fn(y, x0):
            return SolveResult(approximate(y), 1, 0.0, True)

        return InverseOperator(operator, self, solve_fn, traits=Traits.SELF_ADJOINT)
