"""
Preconditioners.

A preconditioner is an approximate inverse, so it is a :class:`LinearSolver`
that happens not to be exact. v1 already models it this way and it is right:
nothing else is needed, and the iterative solvers accept either a ready-made
operator or a solver to build one from.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Iterator, Literal, Sequence

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
from numpy.random import Generator

from ..algebra.operators import LinearOperator, require_coordinates
from ..algebra.spaces import CoordinateSpace, HilbertSpace
from ..traits import Traits
from .randomized import random_diagonal, random_eig
from .solvers import (
    CGSolver,
    InverseOperator,
    LinearSolver,
    SolveResult,
)

__all__ = [
    "IdentityPreconditioner",
    "JacobiPreconditioner",
    "SpectralPreconditioner",
    "BandedPreconditioner",
    "BlockPreconditioner",
    "WoodburyPreconditioner",
    "ColumnThresholdedPreconditioner",
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

    The diagonal is read exactly by default, and can be *estimated* from a few
    random probes instead -- v1's behavior, and what makes this usable on a
    large matrix-free operator, where exact costs one application per
    component. See ``samples``.
    """

    requires: ClassVar[Traits] = Traits.SELF_ADJOINT
    requires_coordinates: ClassVar[bool] = True

    def __init__(
        self,
        /,
        *,
        floor: float = 1e-14,
        samples: int | None = None,
        rtol: float | None = None,
        max_samples: int | None = None,
        block_size: int = 20,
        rng: Generator | None = None,
        n_jobs: int | None = None,
    ) -> None:
        """
        Args:
            floor: entries smaller than this in magnitude are left alone
                rather than inverted.
            samples: estimate the diagonal from this many random probes
                instead of reading it exactly. **Exact costs one operator
                application per component** on an operator that has to be
                probed, which for a large space is the whole reason someone
                reached for a preconditioner; v1 estimated by default, from
                20 probes. Exact is kept as the default here because an
                operator that knows its own diagonal gives it up for nothing:
                one built from a matrix, a diagonal one, a sum, scaling or
                adjoint of those, or a block-diagonal arrangement of them --
                ``M + t I`` is read. A *composition* is not: ``A* A`` and
                the normal operators built on it cannot be read and pay the
                probe, and on those this is the argument to pass.
            rtol: stop adding probes once the running estimate changes by
                less than this, relative; ``samples`` is then the first
                batch, ``block_size`` by default when it is not given. v1's
                adaptive stopping, which the estimator underneath always had
                and this did not pass on.
            max_samples: the cap on the adaptive route.
            block_size: probes added per step on the adaptive route.
            rng: the generator for those probes.
            n_jobs: workers for the exact probe, where one is needed. Serial
                by default.
        """
        if samples is not None and samples < 1:
            raise ValueError(f"At least one sample is needed, got {samples}.")
        if rtol is not None and samples is None:
            samples = block_size
        self._floor = floor
        self._samples = samples
        self._rtol = rtol
        self._max_samples = max_samples
        self._block_size = block_size
        self._rng = rng
        self._n_jobs = n_jobs

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        domain: CoordinateSpace = operator.domain
        codomain: CoordinateSpace = operator.codomain
        require_coordinates(domain, codomain)

        if self._samples is None:
            diagonal = operator.diagonals(
                offsets=(0,), form="galerkin", n_jobs=self._n_jobs
            )[0]
        else:
            diagonal = random_diagonal(
                operator,
                samples=self._samples,
                rtol=self._rtol,
                max_samples=self._max_samples,
                block_size=self._block_size,
                form="galerkin",
                rng=self._rng,
                n_jobs=self._n_jobs,
            )
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

    A randomized eigendecomposition finds the leading ``rank`` modes; those are
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
        rank: int | None = 20,
        damping: float | None = None,
        rtol: float = 1e-4,
        max_rank: int | None = None,
        block_size: int = 10,
        rng: Generator | None = None,
        n_jobs: int | None = None,
    ) -> None:
        """
        Args:
            rank: how many eigenmodes to resolve. ``None`` resolves as many as
                ``rtol`` demands, growing the range ``block_size`` modes at a
                time until the residual falls below it or ``max_rank`` is
                reached: v1's ``method="variable"``, which the range finder
                underneath always offered and this did not pass on.
            damping: the scalar standing in for the tail. Defaults to the
                smallest resolved eigenvalue.
            rtol: the adaptive route's stopping tolerance.
            max_rank: its cap.
            block_size: modes added per step on the adaptive route.
            rng: the generator for the randomized range finder.
            n_jobs: workers for the range finder's probes. Serial by default.
        """
        if rank is not None and rank < 1:
            raise ValueError(f"The rank must be positive, got {rank}.")
        self._rank = rank
        self._damping = damping
        self._rtol = rtol
        self._max_rank = max_rank
        self._block_size = block_size
        self._rng = rng
        self._n_jobs = n_jobs

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        space = operator.domain
        low_rank = random_eig(
            operator,
            rank=self._rank,
            rng=self._rng,
            rtol=self._rtol,
            max_rank=self._max_rank,
            block_size=self._block_size,
            n_jobs=self._n_jobs,
        )
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


def _sparse_factorization(
    matrix: Any, /, *, incomplete: bool, drop_tol: float, fill_factor: float
) -> Any:
    """An exact or an incomplete LU of a sparse matrix, as SciPy gives them.

    Shared by the three sparse preconditioners. The incomplete factorization
    is for when even the sparse factors fill in too much; its ``drop_tol``
    and ``fill_factor`` are SciPy's own.
    """
    if incomplete:
        return sparse_linalg.spilu(matrix, drop_tol=drop_tol, fill_factor=fill_factor)
    return sparse_linalg.splu(matrix)


def _sparse_inverse_traits(operator: LinearOperator, /, *, galerkin: bool) -> Traits:
    """What a sparse approximate inverse of *operator* may honestly claim.

    :class:`BandedPreconditioner` and :class:`BlockPreconditioner` both keep a
    *symmetric pattern* of entries -- a band, or every pair inside a block --
    from one of the operator's two matrices, factor it, and solve with the
    result. The operator they return therefore has component matrix ``B^-1 G``
    when the entries came from the Galerkin matrix and ``B^-1`` when they came
    from the components matrix, and an operator on a coordinate space is
    self-adjoint exactly when ``G C`` is symmetric:

    ==================  ============  ======================================
    entries taken from  ``G C``       symmetric when
    ==================  ============  ======================================
    Galerkin            ``G B^-1 G``  ``B`` is: a symmetric pattern truncates
                                      a symmetric matrix to a symmetric one,
                                      and the Galerkin matrix is symmetric
                                      exactly when the operator is
                                      self-adjoint
    components          ``G B^-1``    ``B`` is *and* ``G`` is the identity
    ==================  ============  ======================================

    So the claim is not free, and both classes used to make it unconditionally:
    ``form="components"`` on an operator that is not self-adjoint, or on any
    space whose metric is not the identity, gave an inverse that is not
    self-adjoint while saying it was. ``check_traits`` caught it on a dense
    Gram; conjugate gradients would not have, it would merely have converged
    to something else.

    Note the claim is inherited from the operator's own, as claims are here:
    it is exact for ``probe="exact"``, and for ``probe="banded"`` it holds
    under that probe's stated precondition, that the operator really is banded
    so the fold adds nothing.

    Args:
        operator: the operator being approximated.
        galerkin: whether the entries were taken from the Galerkin matrix.

    Returns:
        ``SELF_ADJOINT`` where that is true, ``NONE`` otherwise.
    """
    domain = operator.domain
    self_adjoint = bool(Traits.SELF_ADJOINT & operator.traits) and (
        galerkin or bool(getattr(domain, "is_orthonormal", False))
    )
    return Traits.SELF_ADJOINT if self_adjoint else Traits.NONE


class BandedPreconditioner(LinearSolver):
    """Keep a band of the operator's matrix and factor it.

    The cheapest structural approximation there is: an operator whose
    correlations are local has a matrix that is nearly banded in a basis
    ordered by position, and the band is what a sparse factorization can
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
        incomplete: bool = False,
        drop_tol: float = 1e-4,
        fill_factor: float = 10.0,
        n_jobs: int | None = None,
    ) -> None:
        """
        Args:
            bandwidth: sub- and super-diagonals to keep on each side. One gives
                a tridiagonal preconditioner.
            form: which matrix representation to band. The Galerkin form is
                the one in which a self-adjoint operator is symmetric, so it
                is the only one from which the resulting inverse can claim
                self-adjointness unless the space's metric is the identity;
                see :func:`_sparse_inverse_traits`.
            probe: how to extract the diagonals. ``"exact"`` by default even
                though the result is an approximation either way: the fast
                probe sums the out-of-band entries into the band, and on an
                operator that is *not* banded that turns a merely unhelpful
                preconditioner into an actively harmful one. Use ``"banded"``
                when the operator really is banded, where the two agree.
            incomplete: factorize with an incomplete LU rather than an exact
                sparse one, for when even the banded factors fill in too much.
            drop_tol: the ILU drop tolerance, used only when *incomplete*.
            fill_factor: the ILU fill limit, used only when *incomplete*.
            n_jobs: workers for the exact probe's columns. Serial by default.
        """
        if bandwidth < 0:
            raise ValueError(f"The bandwidth must be non-negative, got {bandwidth}.")
        self._bandwidth = bandwidth
        self._form = form
        self._probe = probe
        self._incomplete = incomplete
        self._drop_tol = drop_tol
        self._fill_factor = fill_factor
        self._n_jobs = n_jobs

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        space = operator.domain
        offsets = list(range(-self._bandwidth, self._bandwidth + 1))
        diagonals = operator.diagonals(
            offsets=offsets, form=self._form, probe=self._probe, n_jobs=self._n_jobs
        )
        banded = sparse.dia_array(
            (diagonals, offsets), shape=(space.dim, space.dim)
        ).tocsc()
        factorization = _sparse_factorization(
            banded,
            incomplete=self._incomplete,
            drop_tol=self._drop_tol,
            fill_factor=self._fill_factor,
        )
        galerkin = self._form == "galerkin" or (
            self._form == "auto" and bool(Traits.SELF_ADJOINT & operator.traits)
        )

        def solve_fn(y, x0):
            components = space.to_components(y)
            if galerkin:
                components = space.apply_gram(components)
            return SolveResult(
                space.from_components(factorization.solve(components)), 1, 0.0, True
            )

        return InverseOperator(
            operator,
            self,
            solve_fn,
            traits=_sparse_inverse_traits(operator, galerkin=galerkin),
        )


def _probe_columns(
    operator: LinearOperator,
    columns: Sequence[int],
    /,
    *,
    galerkin: bool,
    n_jobs: int | None = None,
) -> "Iterator[tuple[int, np.ndarray]]":
    """One column of an operator's matrix at a time, never all of them.

    ``matrix()`` costs one application per column *and* holds the whole
    ``dim x dim`` result, which for a preconditioner that keeps a few entries
    per column is the one thing it exists to avoid: the point of a sparse
    preconditioner is that the dense matrix does not fit. v1 probed a
    matrix-free scipy operator column by column for exactly this reason.

    The applications are the same either way; only the memory differs.
    In parallel the columns go to the workers a few per worker at a time,
    so the memory stays bounded by the block rather than by the matrix.

    Args:
        operator: the operator to probe.
        columns: which columns are wanted.
        galerkin: return the Galerkin column ``G A e_j`` rather than the
            component column ``A e_j``.
        n_jobs: workers for the applications. Serial by default.

    Yields:
        ``(index, column)`` pairs.
    """
    from ..parallel import parallel_map, resolve_jobs

    domain: CoordinateSpace = operator.domain
    codomain: CoordinateSpace = operator.codomain

    def probe(index: int) -> np.ndarray:
        basis = np.zeros(domain.dim)
        basis[index] = 1.0
        image = codomain.to_components(operator(domain.from_components(basis)))
        return codomain.apply_gram(image) if galerkin else image

    wanted = list(columns)
    workers = resolve_jobs(n_jobs)
    if workers == 1:
        for index in wanted:
            yield index, probe(index)
        return
    block = 8 * workers
    for start in range(0, len(wanted), block):
        chunk = wanted[start : start + block]
        yield from zip(chunk, parallel_map(probe, chunk, n_jobs=n_jobs))


def _thresholded_matrix(
    operator: LinearOperator,
    keep: "Callable[[np.ndarray, int], np.ndarray]",
    /,
    *,
    n_jobs: int | None = None,
) -> "sparse.csc_matrix":
    """The Galerkin matrix with, in each column, only the rows *keep* names.

    One pass of column probes, never the whole matrix: at dim 2000 with 20
    entries kept per column that is 0.6 MB against 32 MB. Either column
    wanting a position is enough, which makes the pattern symmetric without
    dropping anything that was asked for; the value at a position one column
    wanted and the other did not is the Galerkin matrix's own, which is
    symmetric, so the transpose of what was kept supplies it and no second
    probe is needed.
    """
    dimension = operator.domain.dim
    kept_rows, kept_columns, kept_values = [], [], []
    for index, column in _probe_columns(
        operator, range(dimension), galerkin=True, n_jobs=n_jobs
    ):
        rows = keep(column, index)
        kept_rows.append(rows)
        kept_columns.append(np.full(rows.size, index))
        kept_values.append(column[rows])

    thresholded = sparse.coo_matrix(
        (
            np.concatenate(kept_values),
            (np.concatenate(kept_rows), np.concatenate(kept_columns)),
        ),
        shape=(dimension, dimension),
    ).tocsr()
    mirrored = thresholded.T.tocsr()
    missing = mirrored.copy()
    missing[thresholded.astype(bool)] = 0.0
    missing.eliminate_zeros()
    return (thresholded + missing).tocsc()


def sparse_approximation(
    operator: LinearOperator,
    /,
    *,
    threshold: float = 1e-3,
    max_per_column: int | None = None,
    criterion: Literal["correlation", "column"] = "correlation",
    diagonal: np.ndarray | None = None,
    n_jobs: int | None = None,
) -> LinearOperator:
    """A sparse operator approximating a self-adjoint one, never formed densely.

    Built by probing the operator's Galerkin matrix one column at a time and
    keeping, in each column, the entries that pass a test against the
    diagonal; the pattern is symmetrized and the diagonal is always kept.
    The result is a :class:`~pygeoinf2.algebra.operators.LinearOperator`
    backed by a ``scipy.sparse`` matrix in Galerkin form, which is what a
    sparse solver, a localized preconditioner or a plotting routine wants
    from a covariance whose correlations are genuinely local. It is an
    operator, not a measure: thresholding does not preserve positive
    definiteness, so nothing here claims it -- ``testing.check_traits`` can
    verify the claim if one is wanted -- and SciPy has no sparse Cholesky
    to sample from, so a measure built on it could not be drawn from.

    This is v1's ``with_sparse_approximation`` with the measure taken off
    it (DECISIONS.md D-65). The port had replaced it with a dense assembly, a
    global threshold and an ``O(N^3)`` definiteness check.

    Args:
        operator: self-adjoint, on a coordinate space.
        threshold: the entries kept. Under ``"correlation"`` those with
            ``|c_ij| / sqrt(d_i d_j) >= threshold``, the natural test for a
            covariance and v1's; under ``"column"`` those with
            ``|c_ij| >= threshold |c_jj|``, the test the thresholded
            preconditioner uses.
        max_per_column: a hard cap on entries kept per column, the largest
            by the same measure. The diagonal is always among them.
        criterion: which test.
        diagonal: the Galerkin diagonal, when the caller has it -- from
            :func:`~pygeoinf2.numerics.randomized.random_diagonal` or
            :func:`~pygeoinf2.numerics.randomized.deflated_diagonal` as an
            estimate on an operator too large to probe twice. Read exactly
            from the operator when omitted, which is free where the operator
            knows its diagonal and one application per column otherwise.
        n_jobs: workers for the column probes. Serial by default.

    Returns:
        The sparse operator, claiming self-adjointness and nothing more.

    Raises:
        ValueError: for a negative threshold, a cap below one, an unknown
            criterion, or a diagonal of the wrong length.
        TypeError: if the space has no component map.
    """
    if threshold < 0.0:
        raise ValueError(f"The threshold must be non-negative, got {threshold}.")
    if max_per_column is not None and max_per_column < 1:
        raise ValueError(
            f"At least one entry per column must be kept -- the diagonal -- "
            f"but max_per_column is {max_per_column}."
        )
    if criterion not in ("correlation", "column"):
        raise ValueError(
            f"The criterion is 'correlation' or 'column', got {criterion!r}."
        )
    _require_self_adjoint_claim(operator, "A sparse approximation")
    require_coordinates(operator.domain, operator.codomain)
    dimension = operator.domain.dim

    if criterion == "correlation":
        if diagonal is None:
            diagonal = operator.diagonals(offsets=(0,), form="galerkin", n_jobs=n_jobs)[
                0
            ]
        diagonal = np.asarray(diagonal, dtype=float)
        if diagonal.shape != (dimension,):
            raise ValueError(
                f"The diagonal has shape {diagonal.shape}; expected ({dimension},)."
            )
        scale = np.sqrt(np.where(np.abs(diagonal) < 1e-14, 1.0, np.abs(diagonal)))

    def keep(column: np.ndarray, index: int) -> np.ndarray:
        if criterion == "correlation":
            measure = np.abs(column) / (scale * scale[index])
            kept = np.flatnonzero(measure >= threshold)
        else:
            measure = np.abs(column)
            reference = measure[index]
            if reference < 1e-14:
                reference = measure.max(initial=0.0)
            kept = np.flatnonzero(measure >= threshold * reference)
        if max_per_column is not None and kept.size > max_per_column:
            masked = measure.copy()
            masked[index] = -1.0
            largest = (
                np.argpartition(masked, -(max_per_column - 1))[-(max_per_column - 1) :]
                if max_per_column > 1
                else []
            )
            kept = np.asarray(largest, dtype=int)
        return np.union1d(kept, [index])

    matrix = _thresholded_matrix(operator, keep, n_jobs=n_jobs)
    return LinearOperator.from_matrix(
        operator.domain,
        operator.domain,
        matrix,
        form="galerkin",
        traits=Traits.SELF_ADJOINT,
    )


def _require_self_adjoint_claim(operator: LinearOperator, what: str) -> None:
    if not (Traits.SELF_ADJOINT & operator.traits):
        raise ValueError(
            f"{what} needs a self-adjoint operator; this one claims "
            f"{operator.traits!s}. Attach the trait with with_traits() and "
            f"verify it with testing.check_traits()."
        )


class BlockPreconditioner(LinearSolver):
    """Invert exactly within groups, and ignore everything between them.

    Given a partition of the components — from
    :meth:`~pygeoinf2.symmetric_space.sphere.Sphere.cluster_points`, or any
    other grouping that reflects locality — this factors the entries the groups
    name and drops the rest. It is the natural preconditioner when the
    operator's structure is *clustered* rather than banded, which is what a
    real acquisition geometry gives.

    The matrix is built one column at a time and kept sparse, so the memory is
    the entries retained rather than ``dim x dim``, so it grows with the
    dimension rather than with its square: measured in blocks of 20, four times
    the dimension costs four times the memory, where the dense matrix costs
    sixteen. At ``dim`` 2000 that is 3.5 MB against 97 MB to form the dense
    matrix alone. The applications are the same either way.

    Blocks may overlap and need not cover everything. Both matter for a real
    clustering, where a point near two clusters belongs to both and a few
    points belong to none.
    """

    requires: ClassVar[Traits] = Traits.NONE
    requires_coordinates: ClassVar[bool] = True

    def __init__(
        self,
        blocks: Sequence[Sequence[int]],
        /,
        *,
        form: Literal["auto", "components", "galerkin"] = "auto",
        incomplete: bool = False,
        drop_tol: float = 1e-4,
        fill_factor: float = 10.0,
        n_jobs: int | None = None,
    ) -> None:
        """
        Args:
            blocks: index groups. They may **overlap**, and they need not cover
                every component: anything left out keeps its own diagonal
                entry, so a partial grouping degrades to Jacobi there rather
                than being refused. v2 required a partition, which rules out
                exactly the clusterings a real geometry produces -- a station
                near two clusters belongs to both.
            form: which matrix representation to take entries from. As for
                :class:`BandedPreconditioner`, only the Galerkin form lets the
                resulting inverse claim self-adjointness on a space whose
                metric is not the identity; see :func:`_sparse_inverse_traits`.
            incomplete: factorize with an incomplete LU rather than an exact
                sparse one, for when the block pattern's factors fill in too
                much -- large overlapping blocks do.
            drop_tol: the ILU drop tolerance, used only when *incomplete*.
            fill_factor: the ILU fill limit, used only when *incomplete*.
            n_jobs: workers for the column probes. Serial by default.

        Raises:
            ValueError: if no blocks are given, or an index is negative.
        """
        self._blocks = [np.asarray(block, dtype=int) for block in blocks]
        if not self._blocks:
            raise ValueError("At least one block is needed.")
        if any(block.size and block.min() < 0 for block in self._blocks):
            raise ValueError("Component indices are non-negative.")
        self._form = form
        self._incomplete = incomplete
        self._drop_tol = drop_tol
        self._fill_factor = fill_factor
        self._n_jobs = n_jobs

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        space: CoordinateSpace = operator.domain
        require_coordinates(space, operator.codomain)
        for block in self._blocks:
            if block.size and block.max() >= space.dim:
                raise ValueError(
                    f"A block names component {int(block.max())}, but the "
                    f"space has {space.dim}."
                )

        galerkin = self._form == "galerkin" or (
            self._form == "auto" and bool(Traits.SELF_ADJOINT & operator.traits)
        )

        # The sparsity pattern each block asks for: every pair within it.
        # Overlapping blocks simply name some pairs twice, and the duplicates
        # are removed rather than added -- the entries come from the operator
        # itself, not from summed block inverses, which is what makes overlap
        # free. That is v1's construction.
        #
        # Held as arrays throughout. A dict of sets says the same thing and
        # cost 11 MB at dim 2000 where the pattern itself is 1.6, which would
        # have given back most of what the sparsity bought.
        pattern_rows, pattern_columns = [], []
        for block in self._blocks:
            pattern_rows.append(np.tile(block, block.size))
            pattern_columns.append(np.repeat(block, block.size))
        # Anything no block claimed keeps its diagonal, so the preconditioner
        # stays invertible on those components.
        diagonal = np.arange(space.dim)
        pattern_rows.append(diagonal)
        pattern_columns.append(diagonal)

        rows = np.concatenate(pattern_rows)
        columns = np.concatenate(pattern_columns)
        _, unique = np.unique(columns * space.dim + rows, return_index=True)
        rows, columns = rows[unique], columns[unique]
        # np.unique sorts by the key, so the columns are already grouped and
        # the boundaries between them are one pass.
        starts = np.searchsorted(columns, np.arange(space.dim))
        starts = np.append(starts, columns.size)

        values = np.empty(rows.size)
        for index, column in _probe_columns(
            operator, np.unique(columns), galerkin=galerkin, n_jobs=self._n_jobs
        ):
            span = slice(starts[index], starts[index + 1])
            values[span] = column[rows[span]]

        assembled = sparse.coo_matrix(
            (values, (rows, columns)), shape=(space.dim, space.dim)
        ).tocsc()
        factorized = _sparse_factorization(
            assembled,
            incomplete=self._incomplete,
            drop_tol=self._drop_tol,
            fill_factor=self._fill_factor,
        )

        def solve_fn(y: Any, x0: Any) -> SolveResult:
            components = space.to_components(y)
            if galerkin:
                components = space.apply_gram(components)
            return SolveResult(
                space.from_components(factorized.solve(components)), 1, 0.0, True
            )

        return InverseOperator(
            operator,
            self,
            solve_fn,
            traits=_sparse_inverse_traits(operator, galerkin=galerkin),
        )


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
        prior_damping: float = 0.0,
    ) -> None:
        """
        Args:
            forward: the forward operator ``A``, possibly a cheap surrogate.
            prior_covariance: the prior covariance ``Q`` on the model space.
            noise_covariance: the data error covariance ``R``.
            solver: inverts the inner operator. This is the one that should be
                cheap: it defaults to conjugate gradients at ``rtol=1e-3``
                with ``strict=False``, which assumes the inner operator is
                positive definite, as both are. Loose because a preconditioner
                is an approximation and inner precision does not show in the
                answer; not strict because a strict inner solver raises on
                slow convergence and takes the outer solve down with it.
            prior_solver: inverts ``Q``, for the data form only. Defaults to
                *solver*. Ignored when *prior_inverse* is given.
            noise_solver: inverts ``R``, for the data form only. Defaults to
                *solver*. Ignored when *noise_inverse* is given.
            prior_inverse: ``Q^-1`` when it is known in closed form, which is
                usually cheaper and always better conditioned than solving.
            noise_inverse: ``R^-1`` likewise — a diagonal noise covariance
                inverts by hand.
            prior_damping: a floor for the data form's ``Q^-1``, which is
                then ``(Q + prior_damping I)^-1`` with ``I`` the identity on
                the model space. Most priors here approximate measures on
                function spaces and are singular in practice long before they
                are in theory, so ``Q^-1`` does not exist and a solve for it
                fails; damping supplies an operator that is self-adjoint,
                positive definite and close to what the identity asks for,
                which is all a preconditioner needs. The model form never
                inverts ``Q`` and is not affected. Contradicts
                *prior_inverse*, so the two cannot be given together.

        Raises:
            ValueError: if the spaces do not fit together, if the damping is
                negative, or if a damping is given alongside *prior_inverse*.
        """
        if prior_damping < 0.0:
            raise ValueError(
                f"The prior damping must be non-negative, got {prior_damping}."
            )
        if prior_damping > 0.0 and prior_inverse is not None:
            raise ValueError(
                "prior_damping regularizes the solve for Q^-1, and prior_inverse "
                "says Q^-1 is already known; give one or the other."
            )
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
        self._prior_damping = float(prior_damping)

    @classmethod
    def from_normal(
        cls,
        normal: Any,
        /,
        *,
        solver: LinearSolver | None = None,
        prior_solver: LinearSolver | None = None,
        noise_solver: LinearSolver | None = None,
        prior_damping: float = 0.0,
    ) -> "WoodburyPreconditioner":
        """Read ``A``, ``Q`` and ``R`` off a normal operator.

        Takes anything exposing ``forward``, ``prior_covariance`` and
        ``error_covariance`` — in practice a
        :class:`~pygeoinf2.inference.normal.NormalOperator`, from an
        inversion's ``normal_operator`` or, more usefully, from a
        ``surrogate`` of one. Precisions are picked up when the measures carry
        them, which is what lets the data form avoid a solve.

        This is the ordinary way to build the preconditioner: the arguments are
        exactly what an inversion already holds, so there is no reason for a
        caller to reassemble them.

        Args:
            normal: the normal operator to read the factors off.
            solver: inverts the inner operator; see the constructor.
            prior_solver: inverts ``Q``, for the data form.
            noise_solver: inverts ``R``, likewise.
            prior_damping: a floor for the data form's ``Q^-1``; see the
                constructor. When it is given, a precision the prior measure
                carries is *not* picked up, since the point of damping is that
                ``Q`` has no usable inverse of its own.

        Returns:
            The preconditioner.

        Raises:
            TypeError: if the object does not expose the three factors.
            ValueError: if they do not fit together, or the damping is
                negative.
        """
        for attribute in ("forward", "prior_covariance", "error_covariance"):
            if not hasattr(normal, attribute):
                raise TypeError(
                    f"from_normal needs an operator carrying its factors — "
                    f"{type(normal).__name__} has no {attribute!r}. Build one "
                    f"with pygeoinf2.inference.NormalOperator, or pass A, Q "
                    f"and R to the constructor directly."
                )
        error_covariance = normal.error_covariance
        if error_covariance is None:
            raise ValueError(
                "The Woodbury identity needs a data error covariance R; this "
                "problem is noise-free, so its normal operator is singular "
                "whenever the model space is the larger of the two."
            )
        return cls(
            normal.forward,
            normal.prior_covariance,
            error_covariance,
            solver=solver,
            prior_solver=prior_solver,
            noise_solver=noise_solver,
            prior_inverse=(
                None
                if prior_damping > 0.0
                else getattr(normal, "prior_precision", None)
            ),
            noise_inverse=getattr(normal, "error_precision", None),
            prior_damping=prior_damping,
        )

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
        # Loose, and not strict. A preconditioner is an approximation, so an
        # inner solve to 1e-8 is precision spent where it does not show, and a
        # *strict* inner solver turns a slow inner convergence into an
        # exception that aborts the whole outer solve -- a preconditioner
        # failing should cost iterations, never the answer.
        return CGSolver(rtol=1e-3, max_iterations=200, strict=False)

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
        form; see the class docstring. With a ``prior_damping`` the first is
        ``(Q + prior_damping I)^-1`` instead, the approximate inverse of a
        prior that has no exact one.
        """
        forward = self._forward
        prior = self._prior
        if self._prior_damping > 0.0:
            # Semidefinite plus definite is definite by the trait closure
            # rules, so the sum needs no claim of its own.
            prior = prior + self._prior_damping * LinearOperator.identity(prior.domain)
        prior_inverse = self._resolve(self._prior_inverse, prior, self._prior_solver)
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
        # The forms invert the Gaussian reading of the factors; the operator
        # in hand may be a scalar multiple of that (a Tikhonov operator in
        # the data space is ``t`` times it), and says so.
        scale = float(getattr(operator, "scale", 1.0))
        if scale != 1.0:
            approximate = (1.0 / scale) * approximate

        def solve_fn(y, x0):
            return SolveResult(approximate(y), 1, 0.0, True)

        return InverseOperator(operator, self, solve_fn, traits=Traits.SELF_ADJOINT)


class ColumnThresholdedPreconditioner(LinearSolver):
    """Keep the large entries of each column, and factorize what is left.

    Where :class:`BandedPreconditioner` assumes the significant entries sit
    near the diagonal, this one finds them: a column's entries are kept when
    they are large relative to that column's diagonal, and dropped otherwise.
    The right choice when the operator is sparse in a basis but not *banded* in
    it — a covariance over scattered points, where what couples to what is a
    matter of geometry rather than of index.

    It costs one application per column, but never holds the whole matrix: the
    columns are probed one at a time and only the retained entries are kept.
    Measured at ``dim`` 2000 with a threshold of 0.1, the whole build peaks at
    1.2 MB against 97 MB to form the dense matrix alone, and grows linearly in
    the dimension rather than quadratically. It is for the case
    where those applications are affordable and a dense factorization is not --
    which is the case it was written for, and the case that forming the dense
    matrix ruled out.

    Note:
        Thresholding column by column does not produce a symmetric matrix —
        entry ``(i, j)`` can pass its column's test while ``(j, i)`` fails its
        own — and conjugate gradients needs a symmetric preconditioner. So the
        *pattern* is symmetrized, by keeping a position when either column
        wants it, and the values are then read off the Galerkin matrix, which
        is symmetric to begin with. v1 did not do this. It is the same failure
        as the untapered truncation of
        :class:`~pygeoinf2.inference.preconditioners.InvariantDistancePreconditioner`
        and it has the same character: cheap to prevent, silent when it happens.
    """

    requires: ClassVar[Traits] = Traits.SELF_ADJOINT
    requires_coordinates: ClassVar[bool] = True

    def __init__(
        self,
        threshold: float,
        /,
        *,
        max_per_column: int | None = None,
        incomplete: bool = False,
        drop_tol: float = 1e-4,
        fill_factor: float = 10.0,
        n_jobs: int | None = None,
    ) -> None:
        """
        Args:
            threshold: entries below ``threshold * |diagonal|`` of their own
                column are dropped. Zero keeps everything.
            max_per_column: a hard cap on retained entries per column, keeping
                the largest. The diagonal is always among them.
            incomplete: factorize with an incomplete LU rather than an exact
                sparse one, for when even the sparse factors fill in too much.
            drop_tol: the ILU drop tolerance, used only when *incomplete*.
            fill_factor: the ILU fill limit, used only when *incomplete*.
            n_jobs: workers for the column probes. Serial by default.
        """
        if threshold < 0.0:
            raise ValueError(f"The threshold must be non-negative, got {threshold}.")
        if max_per_column is not None and max_per_column < 1:
            raise ValueError(
                f"At least one entry per column must be kept -- the diagonal "
                f"-- but max_per_column is {max_per_column}."
            )
        self._threshold = threshold
        self._max_per_column = max_per_column
        self._incomplete = incomplete
        self._drop_tol = drop_tol
        self._fill_factor = fill_factor
        self._n_jobs = n_jobs

    def _keep(self, column: np.ndarray, index: int) -> np.ndarray:
        """Which rows of one column survive."""
        magnitudes = np.abs(column)
        reference = magnitudes[index]
        if reference < 1e-14:
            reference = magnitudes.max(initial=0.0)
        kept = np.flatnonzero(magnitudes >= self._threshold * reference)
        cap = self._max_per_column
        if cap is not None and kept.size > cap:
            masked = magnitudes.copy()
            masked[index] = -1.0
            largest = (
                np.argpartition(masked, -(cap - 1))[-(cap - 1) :] if cap > 1 else []
            )
            kept = np.asarray(largest, dtype=int)
        return np.union1d(kept, [index])

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        domain: CoordinateSpace = operator.domain
        require_coordinates(domain, operator.codomain)
        dimension = domain.dim

        # One pass, keeping only what survives each column's own test; the
        # whole matrix is never held. Shared with sparse_approximation.
        del dimension
        thresholded = _thresholded_matrix(operator, self._keep, n_jobs=self._n_jobs)

        factorized = _sparse_factorization(
            thresholded,
            incomplete=self._incomplete,
            drop_tol=self._drop_tol,
            fill_factor=self._fill_factor,
        )

        def solve_fn(y: Any, x0: Any) -> SolveResult:
            weighted = domain.apply_gram(domain.to_components(y))
            return SolveResult(
                domain.from_components(factorized.solve(weighted)), 0, 0.0, True
            )

        return InverseOperator(operator, self, solve_fn, traits=Traits.SELF_ADJOINT)
