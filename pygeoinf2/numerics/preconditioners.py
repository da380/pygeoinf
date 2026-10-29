"""
Preconditioners.

A preconditioner is an approximate inverse, so it is a :class:`LinearSolver`
that happens not to be exact. The iterative solvers accept either a
ready-made operator or a solver to build one from; a solver here is applied
to the operator being inverted when the solve is set up.

The generic preconditioners read entries of the operator's matrix: its
diagonal (:class:`JacobiPreconditioner`), a band
(:class:`BandedPreconditioner`), groups of components
(:class:`BlockPreconditioner`) or the large entries of each column
(:class:`ColumnThresholdedPreconditioner`). :class:`SpectralPreconditioner`
resolves the leading eigenmodes, and :class:`WoodburyPreconditioner` inverts
a normal operator through the other space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterator, Literal, Sequence

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
from numpy.random import Generator

from ..algebra.operators import LinearOperator, require_coordinates
from ..algebra.spaces import (
    CoordinateSpace,
    DiagonalMetricSpace,
    EuclideanSpace,
    HilbertSpace,
    OrthonormalSpace,
)
from ..traits import Traits
from .normal import FactoredNormalOperator, NormalOperator
from .randomized import random_diagonal, random_eig
from .solvers import (
    BoundFamily,
    CGSolver,
    FamilySolver,
    InverseOperator,
    LinearSolver,
    SolveResult,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..symmetric_space.base import SymmetricSpace

__all__ = [
    "IdentityPreconditioner",
    "JacobiPreconditioner",
    "SpectralPreconditioner",
    "BandedPreconditioner",
    "BlockPreconditioner",
    "WoodburyPreconditioner",
    "ColumnThresholdedPreconditioner",
    "sparse_approximation",
    "NormalDiagonalPreconditioner",
    "LocalizedPreconditioner",
    "InvariantDistancePreconditioner",
    "gaspari_cohn",
]


class IdentityPreconditioner(LinearSolver):
    """Does nothing, usefully: a baseline and a null object."""

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        def solve_fn(y, x0):
            return SolveResult(y, 0, 0.0, True)

        return InverseOperator(operator, self, solve_fn, traits=Traits.NONE)


# Diagonal entries at or below this fraction of the largest are treated as
# zero by the thresholds that compare entries with a diagonal.
_RELATIVE_FLOOR = 1e-14


def _safe_diagonal(diagonal: np.ndarray, floor: float) -> np.ndarray:
    """A diagonal fit to divide by, whatever its units.

    Entries at or below *floor* times the largest magnitude are replaced by
    that largest magnitude, so a preconditioner built from the result holds
    those components at the scale of the rest. A zero diagonal becomes ones,
    there being no scale to take.

    Args:
        diagonal: the entries.
        floor: the fraction of the largest magnitude below which an entry
            is not used.

    Returns:
        The entries to divide by.
    """
    magnitudes = np.abs(diagonal)
    largest = float(magnitudes.max(initial=0.0))
    if largest == 0.0:
        return np.ones_like(diagonal, dtype=float)
    return np.where(magnitudes > floor * largest, diagonal, largest)


class JacobiPreconditioner(LinearSolver):
    """Inverts the diagonal of the Galerkin matrix.

    Needs coordinates, since a diagonal is a statement about a basis. The
    Galerkin form is the right one for the same reason the direct solvers use
    it: it is the representation in which a self-adjoint operator is symmetric,
    so its diagonal is the one a symmetric preconditioner wants.

    The diagonal is read exactly by default, and can be *estimated* from a few
    random probes instead, which is what makes this usable on a large
    matrix-free operator, where exact costs one application per component.
    See ``samples``.
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
            floor: relative to the diagonal's largest magnitude. Entries at
                or below this fraction of it are not inverted but replaced by
                that largest magnitude, so the preconditioner holds those
                components at the scale of the rest instead of amplifying
                them; the test is the same whatever the operator's units.
            samples: estimate the diagonal from this many random probes
                instead of reading it exactly. **Exact costs one operator
                application per component** on an operator that has to be
                probed, which on a large space is prohibitive. Exact is the
                default because an operator that knows its own diagonal gives
                it up for nothing:
                one built from a matrix, a diagonal one, a sum, scaling or
                adjoint of those, or a block-diagonal arrangement of them --
                ``M + t I`` is read. A *composition* is not: ``A* A`` and
                the normal operators built on it cannot be read and pay the
                probe, and on those this is the argument to pass.
            rtol: stop adding probes once the running estimate changes by
                less than this, relative; ``samples`` is then the first
                batch, ``block_size`` by default when it is not given.
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
        inverse_diagonal = 1.0 / _safe_diagonal(diagonal, self._floor)

        def solve_fn(y, x0):
            cy = codomain.apply_gram(codomain.to_components(y))
            return SolveResult(
                domain.from_components(inverse_diagonal * cy), 0, 0.0, True
            )

        return InverseOperator(operator, self, solve_fn, traits=Traits.SELF_ADJOINT)


class SpectralPreconditioner(FamilySolver):
    """Invert the dominant eigenmodes exactly, and damp the rest.

    A randomized eigendecomposition finds the leading ``rank`` modes; those are
    inverted properly and the unresolved tail is replaced by a single scalar.
    That is the right shape for an operator whose spectrum decays: the modes
    that make it ill-conditioned are exactly the ones the decomposition finds.

    The damping defaults to the smallest resolved eigenvalue, which is the only
    value that makes the preconditioner continuous at the truncation.

    **Along a damped family** ``base + t * shift`` the deflation is paid
    once. Handed the family through :meth:`for_family`, it deflates the base,
    takes the shift's quadratic form on each resolved mode, and at a
    multiplier ``t`` moves every resolved eigenvalue by ``t`` times that
    form and the tail's damping by ``t`` times their mean. Exact when the
    shift is a multiple of the identity, where the base's eigenvectors are
    the member's; an approximation otherwise, which is all a preconditioner
    needs.
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
                reached.
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

    def _deflate(self, operator: LinearOperator) -> tuple[LinearOperator, np.ndarray]:
        """The leading modes of an operator: ``(factor, eigenvalues)``."""
        low_rank = random_eig(
            operator,
            rank=self._rank,
            rng=self._rng,
            rtol=self._rtol,
            max_rank=self._max_rank,
            block_size=self._block_size,
            n_jobs=self._n_jobs,
        )
        return low_rank.factor, low_rank.eigenvalues

    def _floor(self, values: np.ndarray) -> float:
        floor = self._damping
        if floor is None:
            positive = values[values > 0.0]
            floor = float(positive.min()) if positive.size else 1.0
        if floor <= 0.0:
            raise ValueError(f"The damping must be positive, got {floor}.")
        return floor

    def _inverse_of(
        self,
        operator: LinearOperator,
        factor: LinearOperator,
        values: np.ndarray,
        floor: float,
    ) -> InverseOperator:
        space = operator.domain
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

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        factor, values = self._deflate(operator)
        return self._inverse_of(operator, factor, values, self._floor(values))

    def for_family(
        self, base: LinearOperator, shift: LinearOperator, /
    ) -> "_SpectralFamilyPreconditioner":
        """Deflate the base once, and take the shift's form on each resolved mode.

        Args:
            base: the undamped part, self-adjoint; what is deflated.
            shift: what the multiplier scales; applied once to each resolved
                mode, the modes shared among ``n_jobs`` workers.

        Returns:
            The prepared family.

        Raises:
            ValueError: if the two operators are not on one space, or the base
                does not claim self-adjointness.
        """
        if not (base.domain == base.codomain == shift.domain == shift.codomain):
            raise ValueError(
                "A family's base and shift must both be operators on one space."
            )
        _require_self_adjoint_claim(base, type(self).__name__)
        factor, values = self._deflate(base)
        space = base.domain
        modes = factor.domain
        resolved = factor.apply_block(
            [modes.basis_vector(i) for i in range(modes.dim)], n_jobs=self._n_jobs
        )
        images = shift.apply_block(resolved, n_jobs=self._n_jobs)
        forms = np.array(
            [space.inner_product(mode, image) for mode, image in zip(resolved, images)]
        )
        return _SpectralFamilyPreconditioner(
            self, factor, values, forms, self._floor(values)
        )


class _SpectralFamilyPreconditioner(BoundFamily):
    """A deflated base whose eigenvalues move with the multiplier."""

    def __init__(
        self,
        preconditioner: SpectralPreconditioner,
        factor: LinearOperator,
        values: np.ndarray,
        forms: np.ndarray,
        floor: float,
    ) -> None:
        self._preconditioner = preconditioner
        self._factor = factor
        self._values = values
        self._forms = forms
        self._floor = floor

    def at(self, multiplier: float, member: LinearOperator, /) -> InverseOperator:
        """The deflated inverse of the member at one multiplier."""
        values = self._values + multiplier * self._forms
        tail = float(np.mean(self._forms)) if self._forms.size else 1.0
        floor = self._floor + multiplier * tail
        # The family is the preconditioner's own companion, built only by
        # its ``for_family``, so it shares the assembly step by name.
        return self._preconditioner._inverse_of(member, self._factor, values, floor)


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

    So the claim is not free: entries taken from the components matrix give
    a self-adjoint inverse only when the operator is self-adjoint and the
    metric is the identity, and a false claim would not be caught by
    conjugate gradients, which would converge to something else.

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
    domain: CoordinateSpace = operator.domain
    self_adjoint = bool(Traits.SELF_ADJOINT & operator.traits) and (
        galerkin or domain.is_orthonormal
    )
    return Traits.SELF_ADJOINT if self_adjoint else Traits.NONE


class BandedPreconditioner(LinearSolver):
    """Keep a band of the operator's matrix and factor it.

    The cheapest structural approximation there is: an operator whose
    correlations are local has a matrix that is nearly banded in a basis
    ordered by position, and the band is what a sparse factorization can
    invert quickly.

    **Only worth it when the operator has that structure.** On a dense operator
    a tridiagonal approximation is not a poor preconditioner but a harmful
    one, and can make conjugate gradients fail where it converged without.
    Nothing here can detect that for you, which is why the bandwidth is a
    required argument rather than a default.
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
            form: which matrix representation to band. ``"auto"`` takes the
                Galerkin form when the operator claims self-adjointness and
                the components form otherwise. The Galerkin form is the one in
                which a self-adjoint operator is symmetric, so it is the only
                one from which the resulting inverse can claim
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
    preconditioner is that the dense matrix does not fit.

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


def _capped_rows(
    measure: np.ndarray, kept: np.ndarray, index: int, cap: int | None, /
) -> np.ndarray:
    """The rows kept in one column, the diagonal always among them.

    Args:
        measure: the size of each entry by the column's own test.
        kept: the rows that passed the test.
        index: the column, which is the diagonal's row.
        cap: at most this many rows, the largest by *measure*; None for
            no cap.

    Returns:
        The row indices, sorted.
    """
    if cap is not None and kept.size > cap:
        masked = measure.copy()
        masked[index] = -1.0
        largest = np.argpartition(masked, -(cap - 1))[-(cap - 1) :] if cap > 1 else []
        kept = np.asarray(largest, dtype=int)
    return np.union1d(kept, [index])


def _column_rows(
    column: np.ndarray, index: int, /, *, threshold: float, cap: int | None
) -> np.ndarray:
    """The ``"column"`` criterion: rows at least *threshold* times the diagonal.

    The column's largest entry stands in for a diagonal that is zero. Shared
    by :func:`sparse_approximation` and
    :class:`ColumnThresholdedPreconditioner`, which is this test with a
    factorization behind it.

    Args:
        column: one column of the Galerkin matrix.
        index: which column, so which row holds the diagonal.
        threshold: the fraction of the diagonal an entry must reach.
        cap: at most this many rows per column; None for no cap.

    Returns:
        The row indices, sorted.
    """
    magnitudes = np.abs(column)
    reference = magnitudes[index]
    largest = magnitudes.max(initial=0.0)
    if reference <= _RELATIVE_FLOOR * largest:
        reference = largest
    kept = np.flatnonzero(magnitudes >= threshold * reference)
    return _capped_rows(magnitudes, kept, index, cap)


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

    Args:
        operator: self-adjoint, on a coordinate space.
        threshold: the entries kept. Under ``"correlation"`` those with
            ``|c_ij| / sqrt(d_i d_j) >= threshold``, the natural test for a
            covariance; under ``"column"`` those with
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
        scale = np.sqrt(np.abs(_safe_diagonal(diagonal, _RELATIVE_FLOOR)))

    def keep(column: np.ndarray, index: int) -> np.ndarray:
        if criterion == "column":
            return _column_rows(column, index, threshold=threshold, cap=max_per_column)
        measure = np.abs(column) / (scale * scale[index])
        kept = np.flatnonzero(measure >= threshold)
        return _capped_rows(measure, kept, index, max_per_column)

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
    that of the entries retained rather than ``dim x dim``, and for blocks of
    fixed size it grows with the dimension rather than with its square. The
    applications are those of a dense assembly, one per column a block names.

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
                than being refused. A partition is not required, since a real
                clustering is rarely one: a station near two clusters belongs
                to both.
            form: which matrix representation to take entries from, with
                ``"auto"`` as for :class:`BandedPreconditioner`. Only the
                Galerkin form lets the
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
        # free.
        #
        # Held as arrays throughout: a dict of sets says the same thing at
        # several times the memory of the pattern itself.
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


class WoodburyPreconditioner(FamilySolver):
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

    This is the same trade as :func:`~pygeoinf2.numerics.normal.choose_formalism`
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

    **Along a damped family** ``A C A* + t R`` or ``t C⁻¹ + A* R⁻¹ A`` the
    multiplier only scales the noise covariance in the Gaussian reading, so
    handed the family through :meth:`for_family` the preconditioner keeps
    ``Q⁻¹``, ``R⁻¹`` and, in the data form, the kernel ``A* R⁻¹ A`` resolved
    once and refactorizes the inner operator, which lives on the surrogate's
    own space, at each multiplier it is asked for. The surrogate pieces are
    its own; only the multiplier is read off the member. A Tikhonov family
    ``A A* + t R`` reads the same way with ``C`` the identity, so the
    preconditioner for one holds the identity as its prior covariance:
    :meth:`from_normal` of the member at damping one, or the constructor
    with ``prior_inverse`` the identity too.

    Note:
        With an inexact inner solve the result is symmetric only to the inner
        tolerance, and ordinary conjugate gradients relies on a symmetric
        preconditioner. The default inner solve is tight enough for the
        default outer tolerance; with a looser inner solver of your own, use
        :class:`~pygeoinf2.numerics.solvers.FlexibleCGSolver`, or keep the
        inner tolerance below the outer one.
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
                cheap: it defaults to conjugate gradients at ``rtol=1e-10``
                and ``strict=False``, which assumes the inner operator is
                positive definite, as both are. Tight because the result
                claims self-adjointness and ordinary conjugate gradients
                needs that claim to hold to below its own tolerance, ``1e-8``
                by default: an inner solve at ``1e-3`` breaks it by a quarter
                on a four-dimensional data space, and the outer solve then
                need not converge at all. Not strict because a strict inner
                solver raises on slow convergence and takes the outer solve
                down with it. A direct solver is the usual choice where the
                inner operator is small enough to factorize.
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
        normal: FactoredNormalOperator,
        /,
        *,
        solver: LinearSolver | None = None,
        prior_solver: LinearSolver | None = None,
        noise_solver: LinearSolver | None = None,
        prior_damping: float = 0.0,
    ) -> "WoodburyPreconditioner":
        """Read ``A``, ``Q`` and ``R`` off a normal operator.

        Takes a :class:`~pygeoinf2.numerics.normal.FactoredNormalOperator`
        -- in practice a :class:`~pygeoinf2.numerics.normal.NormalOperator`,
        from an inversion's ``normal_operator`` or, more usefully, from a
        ``surrogate`` of one. Precisions are picked up when the operator
        carries them, which is what lets the data form avoid a solve.

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
            TypeError: if the operator does not carry its factors.
            ValueError: if the problem has no error covariance, if the
                factors do not fit together, or if the damping is negative.
        """
        if not isinstance(normal, FactoredNormalOperator):
            raise TypeError(
                f"from_normal needs an operator carrying its factors A, Q and "
                f"R, got {type(normal).__name__}. Build one with "
                f"pygeoinf2.numerics.NormalOperator, or pass A, Q and R to "
                f"the constructor directly."
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
            prior_inverse=None if prior_damping > 0.0 else normal.prior_precision,
            noise_inverse=normal.error_precision,
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
        # Tight, and not strict. The forms claim self-adjointness, and
        # ordinary conjugate gradients relies on the claim to below the outer
        # tolerance, so the inner one sits beneath the default outer 1e-8; a
        # *strict* inner solver would turn a slow inner convergence into an
        # exception that aborts the whole outer solve -- a preconditioner
        # failing should cost iterations, never the answer.
        return CGSolver(rtol=1e-10, strict=False)

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

    def model_form(self, *, multiplier: float = 1.0) -> LinearOperator:
        """``Q - Q A* (t R + A Q A*)^-1 A Q``, an approximate ``N_m^-1``.

        Built without ever inverting ``Q`` or ``R``.

        Args:
            multiplier: ``t``, the factor on the noise covariance; one for
                the Gaussian normal operator itself.
        """
        forward, prior, noise = self._forward, self._prior, self._noise
        inner = multiplier * noise + forward @ prior @ forward.adjoint
        inverse = self._inner_solver()(inner.with_traits(Traits.POSITIVE_DEFINITE))
        cross = prior @ forward.adjoint
        return (prior - cross @ inverse @ cross.adjoint).with_traits(
            Traits.SELF_ADJOINT
        )

    def _resolved_prior_inverse(self) -> LinearOperator:
        """``Q^-1``, or the damped ``(Q + prior_damping I)^-1``."""
        prior = self._prior
        if self._prior_damping > 0.0:
            # Semidefinite plus definite is definite by the trait closure
            # rules, so the sum needs no claim of its own.
            prior = prior + self._prior_damping * LinearOperator.identity(prior.domain)
        return self._resolve(self._prior_inverse, prior, self._prior_solver)

    def _resolved_noise_inverse(self) -> LinearOperator:
        """``R^-1``."""
        return self._resolve(self._noise_inverse, self._noise, self._noise_solver)

    def data_form(
        self,
        *,
        multiplier: float = 1.0,
        prior_inverse: LinearOperator | None = None,
        noise_inverse: LinearOperator | None = None,
        kernel: LinearOperator | None = None,
    ) -> LinearOperator:
        """``(tR)^-1 - (tR)^-1 A (Q^-1 + A* (tR)^-1 A)^-1 A* (tR)^-1``, an approximate ``N_d^-1``.

        Needs ``Q^-1`` and ``R^-1``, so it wants covariances given in inverse
        form; see the class docstring. With a ``prior_damping`` the first is
        ``(Q + prior_damping I)^-1`` instead, the approximate inverse of a
        prior that has no exact one.

        Args:
            multiplier: ``t``, the factor on the noise covariance; one for
                the Gaussian normal operator itself.
            prior_inverse: ``Q^-1`` already resolved, from a previous call
                along a family; resolved here if omitted.
            noise_inverse: ``R^-1`` likewise.
            kernel: ``A* R^-1 A`` on the model space, likewise; see
                :meth:`data_kernel`. Built from the factors if omitted.
        """
        forward = self._forward
        if prior_inverse is None:
            prior_inverse = self._resolved_prior_inverse()
        if noise_inverse is None:
            noise_inverse = self._resolved_noise_inverse()
        if kernel is None:
            kernel = forward.adjoint @ noise_inverse @ forward
        if multiplier != 1.0:
            noise_inverse = (1.0 / multiplier) * noise_inverse
            kernel = (1.0 / multiplier) * kernel
        inner = prior_inverse + kernel
        inverse = self._inner_solver()(inner.with_traits(Traits.POSITIVE_DEFINITE))
        cross = noise_inverse @ forward
        return (noise_inverse - cross @ inverse @ cross.adjoint).with_traits(
            Traits.SELF_ADJOINT
        )

    def data_kernel(
        self, *, noise_inverse: LinearOperator | None = None
    ) -> LinearOperator:
        """``A* R^-1 A`` on the model space, as a stored matrix where it can be.

        The part of the data form's inner operator that does not move with
        the multiplier. Along a family it is wanted at every member, and a
        direct inner solver would otherwise assemble it afresh each time; so
        when the factors can write its matrix down it is read once here and
        kept as a matrix-backed operator, and the inner operator at each
        multiplier is then a sum of two stored matrices. Where the matrix is
        not known the composition itself is returned, and the inner solver
        probes or iterates as it would have.

        Args:
            noise_inverse: ``R^-1`` already resolved; resolved here if
                omitted.

        Returns:
            The kernel, with its traits.
        """
        forward = self._forward
        if noise_inverse is None:
            noise_inverse = self._resolved_noise_inverse()
        kernel = forward.adjoint @ noise_inverse @ forward
        traits = Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE
        matrix = kernel.known_matrix(form="galerkin")
        if matrix is None:
            return kernel.with_traits(traits)
        return LinearOperator.from_matrix(
            forward.domain, forward.domain, matrix, form="galerkin", traits=traits
        )

    def _is_model_form(self, operator: LinearOperator) -> bool:
        dim = operator.domain.dim
        if dim == self.model_space.dim and dim == self.data_space.dim:
            # Ambiguous by dimension alone, so ask the spaces themselves.
            return operator.domain == self.model_space
        return dim == self.model_space.dim

    def _approximate_inverse(
        self,
        operator: LinearOperator,
        approximate: LinearOperator,
        *,
        scale: float | None = None,
    ) -> InverseOperator:
        # The forms invert the Gaussian reading of the factors; the operator
        # in hand may be a scalar multiple of that (a Tikhonov operator in
        # the data space is ``t`` times it), and says so. Along a family the
        # caller knows the factor from the multiplier and passes it.
        if scale is None:
            scale = (
                float(operator.scale)
                if isinstance(operator, FactoredNormalOperator)
                else 1.0
            )
        if scale != 1.0:
            approximate = (1.0 / scale) * approximate

        def solve_fn(y, x0):
            return SolveResult(approximate(y), 1, 0.0, True)

        return InverseOperator(operator, self, solve_fn, traits=Traits.SELF_ADJOINT)

    def for_family(
        self, base: LinearOperator, shift: LinearOperator, /
    ) -> "_WoodburyFamily":
        """Prepare for ``base + t * shift``: the surrogate's inverses resolved once.

        The family's members are read as Gaussian normal operators whose
        noise covariance carries the multiplier, ``A C A* + t R`` in the
        data space and ``t (C⁻¹ + A* (t R)⁻¹ A)`` in the model space, and
        each member's approximate inverse is the Woodbury form with ``t R``
        in place of ``R``, divided by ``t`` in the model space. The surrogate
        pieces are this preconditioner's, with ``C`` the prior covariance it
        holds; the base and the shift only say which space the family lives
        on, and the member's own ``scale`` is not consulted, since the
        reading above already accounts for it whether the member is a
        damped normal operator or a Tikhonov one.

        Args:
            base: the undamped part.
            shift: what the multiplier scales.

        Returns:
            The prepared family.

        Raises:
            ValueError: if the family lives on neither of the surrogate's
                spaces.
        """
        self._validate(base + shift)
        return _WoodburyFamily(self)

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
        approximate = (
            self.model_form() if self._is_model_form(operator) else self.data_form()
        )
        return self._approximate_inverse(operator, approximate)


# How many Woodbury forms a family keeps: the one in hand and the one before
# it, which is all a root find comes back to.
_KEPT_FORMS = 2


class _WoodburyFamily(BoundFamily):
    """The Woodbury form at each multiplier, the surrogate's inverses resolved once."""

    def __init__(self, preconditioner: WoodburyPreconditioner) -> None:
        self._preconditioner = preconditioner
        self._prior_inverse: LinearOperator | None = None
        self._noise_inverse: LinearOperator | None = None
        self._kernel: LinearOperator | None = None
        # The most recent forms only: each may hold an inner factorization,
        # and a sweep keeping one per multiplier would grow by that much per
        # probe.
        self._forms: dict[tuple[bool, float], LinearOperator] = {}
        self._smallest: dict[bool, float] = {}

    def at(self, multiplier: float, member: LinearOperator, /) -> InverseOperator:
        """The Woodbury form of the member at one multiplier.

        At a multiplier of zero the forms have no ``(t R)⁻¹`` to speak of
        and the surrogate's own normal operator may be singular, so the
        member is approximated by the form at the smallest positive
        multiplier asked for so far, or at one: positive definite, and as a
        preconditioner it only has to be close.
        """
        preconditioner = self._preconditioner
        model = preconditioner._is_model_form(member)
        if multiplier <= 0.0:
            multiplier = self._smallest.get(model, 1.0)
        else:
            self._smallest[model] = min(
                self._smallest.get(model, multiplier), multiplier
            )
        key = (model, float(multiplier))
        if key not in self._forms:
            while len(self._forms) >= _KEPT_FORMS:
                self._forms.pop(next(iter(self._forms)))
            if model:
                self._forms[key] = preconditioner.model_form(multiplier=multiplier)
            else:
                if self._prior_inverse is None:
                    self._prior_inverse = preconditioner._resolved_prior_inverse()
                    self._noise_inverse = preconditioner._resolved_noise_inverse()
                    self._kernel = preconditioner.data_kernel(
                        noise_inverse=self._noise_inverse
                    )
                self._forms[key] = preconditioner.data_form(
                    multiplier=multiplier,
                    prior_inverse=self._prior_inverse,
                    noise_inverse=self._noise_inverse,
                    kernel=self._kernel,
                )
        # The data form inverts ``A C A* + t R``, the member itself; the
        # model form inverts ``C⁻¹ + A* (t R)⁻¹ A``, of which the member is
        # ``t`` times.
        return preconditioner._approximate_inverse(
            member, self._forms[key], scale=multiplier if model else 1.0
        )


class ColumnThresholdedPreconditioner(LinearSolver):
    """Keep the large entries of each column, and factorize what is left.

    Where :class:`BandedPreconditioner` assumes the significant entries sit
    near the diagonal, this one finds them: a column's entries are kept when
    they are large relative to that column's diagonal, and dropped otherwise.
    The right choice when the operator is sparse in a basis but not *banded* in
    it — a covariance over scattered points, where what couples to what is a
    matter of geometry rather than of index.

    It costs one application per column, but never holds the whole matrix: the
    columns are probed one at a time and only the retained entries are kept,
    so the memory grows with the number of entries kept rather than with the
    square of the dimension. It is for the case where those applications are
    affordable and a dense matrix or its factorization is not.

    Note:
        Thresholding column by column does not produce a symmetric matrix —
        entry ``(i, j)`` can pass its column's test while ``(j, i)`` fails its
        own — and conjugate gradients needs a symmetric preconditioner. So the
        *pattern* is symmetrized, by keeping a position when either column
        wants it, and the values are then read off the Galerkin matrix, which
        is symmetric to begin with.
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
                column are dropped, the column's largest entry standing in for
                a diagonal that is zero. Zero keeps everything.
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
        return _column_rows(
            column, index, threshold=self._threshold, cap=self._max_per_column
        )

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        domain: CoordinateSpace = operator.domain
        require_coordinates(domain, operator.codomain)

        # One pass, keeping only what survives each column's own test; the
        # whole matrix is never held. Shared with sparse_approximation.
        thresholded = _thresholded_matrix(operator, self._keep, n_jobs=self._n_jobs)

        factorized = _sparse_factorization(
            thresholded,
            incomplete=self._incomplete,
            drop_tol=self._drop_tol,
            fill_factor=self._fill_factor,
        )

        def solve_fn(y: Any, x0: Any) -> SolveResult:
            weighted = domain.apply_gram(domain.to_components(y))
            # One sparse solve, counted as one iteration, as the banded and
            # block preconditioners count theirs.
            return SolveResult(
                domain.from_components(factorized.solve(weighted)), 1, 0.0, True
            )

        return InverseOperator(operator, self, solve_fn, traits=Traits.SELF_ADJOINT)


# --------------------------------------------------------------------- #
#          Preconditioners that need the normal operator's factors      #
# --------------------------------------------------------------------- #
#
# The preconditioners above take any operator and approximate it. These take
# a normal operator that still carries its factors -- a FactoredNormalOperator,
# such as an estimator's ``normal_operator`` or ``surrogate(...)`` -- and
# exploit the fact that it is ``A Q A* + R``, which an assembled operator no
# longer knows. All three are for the data-space formalism, and refuse a
# normal operator assembled in the model space: none of the identities they
# use has a model-space counterpart, because ``Q^-1 + A* R^-1 A`` has no cheap
# diagonal, the expensive part being the operator sandwiched in the middle
# rather than at the ends.


def _require_normal(operator: LinearOperator, name: str) -> FactoredNormalOperator:
    if not isinstance(operator, FactoredNormalOperator):
        raise TypeError(
            f"{name} needs the factors A, Q and R, so it takes a normal "
            f"operator that still carries them rather than an assembled one. "
            f"Build it from an estimator's .normal_operator, or from "
            f".surrogate(...). Got {type(operator).__name__}."
        )
    if operator.formalism != "data_space":
        raise ValueError(
            f"{name} is derived for the data-space normal operator "
            f"A Q A* + R. This one is assembled in the model space, where the "
            f"identity it uses has no counterpart."
        )
    return operator


def _probe_normal_diagonal(
    normal: FactoredNormalOperator,
    blocks: Sequence[Sequence[int]],
    /,
    *,
    n_jobs: int | None,
) -> list[float]:
    """``<A* v, Q A* v>`` for the normalized indicator ``v`` of each block.

    The identity ``<v, A Q A* v> == <A* v, Q A* v>``: for a singleton block
    this is one Galerkin diagonal entry of ``A Q A*``, read off the factors
    at one adjoint and one prior application rather than an application of
    the normal operator; for a larger block it is the value the block's
    members share.

    Args:
        normal: the data-space normal operator carrying its factors.
        blocks: the index groups to probe.
        n_jobs: workers, one block per worker. Serial by default.

    Returns:
        One value per block, in order.
    """
    data_space = normal.data_space
    model_space = normal.model_space
    dimension = data_space.dim
    adjoint = normal.forward.adjoint
    prior_covariance = normal.prior_covariance

    from ..parallel import parallel_map

    def probe(block: Sequence[int]) -> float:
        components = np.zeros(dimension)
        components[list(block)] = 1.0 / len(block)
        pulled = adjoint(data_space.from_components(components))
        return model_space.inner_product(pulled, prior_covariance(pulled))

    return parallel_map(probe, [list(block) for block in blocks], n_jobs=n_jobs)


class NormalDiagonalPreconditioner(FamilySolver):
    """Invert the diagonal of ``A Q A* + R``, obtained without forming it.

    The identity is ``<v, A Q A* v> == <A* v, Q A* v>``, so a diagonal entry
    costs one adjoint application and one application of the prior covariance —
    never an application of the assembled operator, and never the forward
    operator at all.

    The saving that matters is *blocks*. Given a partition of the data indices,
    the probe vector for a block is its normalized indicator, so the whole block
    shares one adjoint application and is assigned one representative variance.
    A thousand data in fifty blocks costs fifty adjoint applications rather than
    a thousand, and for data that cluster — stations in a region, rays through a
    corridor — the representative value is close to each member's own.

    With no blocks this computes the exact Galerkin diagonal, and then agrees
    with :class:`~pygeoinf2.numerics.preconditioners.JacobiPreconditioner`
    applied to the assembled operator. The Galerkin diagonal is the one to
    take on a weighted data space: a diagonal is a statement about a basis,
    and the Galerkin form is the one in which a self-adjoint operator is
    symmetric.

    **Along a damped family** ``base + t * shift`` the probes are paid once.
    Handed the family through :meth:`for_family`, it probes the diagonal of
    the base off the first member it meets -- ``scale * (A* e, Q A* e)``,
    which does not depend on the multiplier as long as the member's
    ``scale * prior_covariance`` does not, true of a Tikhonov family and of
    any family whose multiplier scales the error term -- reads the shift's
    diagonal once, and adds ``t`` times it for every member thereafter.
    """

    requires: ClassVar[Traits] = Traits.NONE
    requires_coordinates: ClassVar[bool] = True

    def __init__(
        self,
        /,
        *,
        blocks: Sequence[Sequence[int]] | None = None,
        floor: float = 1e-14,
        n_jobs: int | None = None,
    ) -> None:
        """
        Args:
            blocks: a partition of the data indices. Every index must appear
                exactly once; a partial cover is refused rather than silently
                left at zero. None means one block per index, which is exact.
            floor: relative to the diagonal's largest magnitude; entries at
                or below this fraction of it are replaced by that largest
                magnitude rather than inverted, whatever the units.
            n_jobs: workers for the probes, one adjoint application and one
                prior application per block, and for the diagonals of the
                error covariance and of a family's shift where those must be
                probed. Serial by default.
        """
        self._blocks = None if blocks is None else [list(b) for b in blocks]
        self._floor = floor
        self._n_jobs = n_jobs

    def _partition(self, dimension: int) -> list[list[int]]:
        if self._blocks is None:
            return [[index] for index in range(dimension)]
        flattened = [index for block in self._blocks for index in block]
        if len(flattened) != dimension or len(set(flattened)) != dimension:
            raise ValueError(
                f"The blocks must partition the data space exactly: expected "
                f"{dimension} distinct indices, got {len(flattened)} indices "
                f"of which {len(set(flattened))} are distinct."
            )
        if min(flattened) < 0 or max(flattened) >= dimension:
            raise ValueError(
                f"Block indices must lie in [0, {dimension}), but they run "
                f"from {min(flattened)} to {max(flattened)}."
            )
        return self._blocks

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        normal = _require_normal(operator, type(self).__name__)
        diagonal = self._base_diagonal(normal)
        error_covariance = normal.error_covariance
        if error_covariance is not None:
            diagonal = (
                diagonal
                + normal.scale
                * error_covariance.diagonals(
                    offsets=(0,), form="galerkin", n_jobs=self._n_jobs
                )[0]
            )
        return self._inverse_of(operator, normal.data_space, diagonal)

    def _base_diagonal(self, normal: FactoredNormalOperator) -> np.ndarray:
        """``scale * diag(A Q A*)``, probed through the factors.

        At a scale of zero -- a Tikhonov member at zero damping, which has no
        prior reading -- the undamped operator's own diagonal is read
        instead, at a cost of one application per datum.
        """
        data_space = normal.data_space
        require_coordinates(data_space, data_space)
        dimension = data_space.dim
        if normal.scale <= 0.0:
            return normal.diagonals(offsets=(0,), form="galerkin", n_jobs=self._n_jobs)[
                0
            ]

        blocks = self._partition(dimension)
        diagonal = np.zeros(dimension)
        for block, value in zip(
            blocks, _probe_normal_diagonal(normal, blocks, n_jobs=self._n_jobs)
        ):
            diagonal[block] = value
        # The diagonal of the Gaussian reading A Q A*; the operator's base is
        # ``normal.scale`` times that.
        return normal.scale * diagonal

    def _inverse_of(
        self, operator: LinearOperator, data_space: HilbertSpace, diagonal: np.ndarray
    ) -> InverseOperator:
        inverse_diagonal = 1.0 / _safe_diagonal(diagonal, self._floor)

        def solve_fn(y: Any, x0: Any) -> SolveResult:
            weighted = data_space.apply_gram(data_space.to_components(y))
            return SolveResult(
                data_space.from_components(inverse_diagonal * weighted), 0, 0.0, True
            )

        return InverseOperator(operator, self, solve_fn, traits=Traits.SELF_ADJOINT)

    def for_family(
        self, base: LinearOperator, shift: LinearOperator, /
    ) -> "_NormalDiagonalFamily":
        """Prepare for ``base + t * shift``: the shift's diagonal now, the base's at the first member.

        Args:
            base: the undamped part; its diagonal is probed off the first
                member's factors.
            shift: what the multiplier scales; its Galerkin diagonal is read
                once.

        Returns:
            The prepared family.
        """
        require_coordinates(shift.domain, shift.codomain)
        return _NormalDiagonalFamily(
            self,
            shift.diagonals(offsets=(0,), form="galerkin", n_jobs=self._n_jobs)[0],
        )


class _NormalDiagonalFamily(BoundFamily):
    """The probed base diagonal plus ``t`` times the shift's, for any ``t``."""

    def __init__(
        self, preconditioner: NormalDiagonalPreconditioner, shift_diagonal: np.ndarray
    ) -> None:
        self._preconditioner = preconditioner
        self._shift_diagonal = shift_diagonal
        self._base_diagonal: np.ndarray | None = None

    def at(self, multiplier: float, member: LinearOperator, /) -> InverseOperator:
        """The inverted diagonal of the member at one multiplier."""
        normal = _require_normal(member, type(self._preconditioner).__name__)
        # The family is the preconditioner's own companion, built only by
        # its ``for_family``, so it shares the probe and the assembly by name.
        if self._base_diagonal is None:
            self._base_diagonal = self._preconditioner._base_diagonal(normal)
        diagonal = self._base_diagonal + multiplier * self._shift_diagonal
        return self._preconditioner._inverse_of(member, normal.data_space, diagonal)


class LocalizedPreconditioner(LinearSolver):
    """Approximate ``A Q A* + R`` by the blocks that actually couple.

    Data that are near each other — stations in a region, rays sharing a
    corridor — have strongly correlated predictions, and data that are far apart
    have nearly none. So the normal operator is close to block-sparse in a
    sensible ordering, and a preconditioner can keep the blocks and drop the
    rest.

    Each block of ``A Q A*`` -- its Galerkin sub-block ``P (A Q A*) P*`` on
    the block's indices -- is approximated by a randomized eigendecomposition
    (:func:`~pygeoinf2.numerics.randomized.random_eig`) at the given rank
    rather than formed exactly, so a block of size 200 still costs only about
    ``rank`` applications. The blocks are assembled into a sparse matrix, the
    noise diagonal is added, and the result is factorized once with a sparse
    LU.

    Unlike :class:`NormalDiagonalPreconditioner` the blocks may **overlap** and
    need not cover every index: this is an approximation to the operator, not a
    partition of it. Where blocks overlap their estimates are **averaged**,
    which is what makes an overlapping cover behave sensibly at the seams.
    An index outside every block keeps its exact diagonal entry of
    ``A Q A*``, read off the factors at one adjoint application each as
    :class:`NormalDiagonalPreconditioner` reads it, so a partial cover is
    block-diagonal where the blocks are and diagonal elsewhere, and never
    singular.

    Note:
        Only ``A Q A*`` is treated block-wise; the error covariance contributes
        its diagonal. So a single block covering every index, at full rank,
        reproduces ``N^-1`` exactly when ``R`` is diagonal -- which is the
        usual case -- and drops ``R``'s off-diagonal otherwise. That is a real approximation and not a free one: with a
        strongly correlated ``R`` this preconditioner is describing a different
        operator, and the iteration count will say so.

        The blocks buy their saving in applications of ``A*``, which is what an
        expensive forward operator makes expensive. ``R`` is applied
        ``dim(D)`` times regardless, which is cheap precisely because it is the
        operator no one worries about.
    """

    requires: ClassVar[Traits] = Traits.NONE
    requires_coordinates: ClassVar[bool] = True

    def __init__(
        self,
        blocks: Sequence[Sequence[int]],
        /,
        *,
        rank: int = 10,
        rng: Generator | None = None,
        n_jobs: int | None = None,
    ) -> None:
        """
        Args:
            blocks: index groups that couple strongly. May overlap, and need
                not cover every index.
            rank: the rank of the eigendecomposition within each block,
                capped at the block size.
            rng: the generator for the randomized range finder.
            n_jobs: workers for the probes within each block's range finder
                and for the diagonal entries of the uncovered indices.
                Serial by default.

        Raises:
            ValueError: for no blocks at all, or a rank below one.
        """
        self._blocks = [list(block) for block in blocks]
        if not self._blocks:
            raise ValueError("At least one block is needed.")
        if rank < 1:
            raise ValueError(f"The rank must be positive, got {rank}.")
        self._rank = rank
        self._rng = rng
        self._n_jobs = n_jobs

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        normal = _require_normal(operator, type(self).__name__)
        data_space = normal.data_space
        require_coordinates(data_space, data_space)
        dimension = data_space.dim
        for block in self._blocks:
            if not block or min(block) < 0 or max(block) >= dimension:
                raise ValueError(
                    f"Block indices must be non-empty and lie in "
                    f"[0, {dimension}); got a block running from "
                    f"{min(block) if block else 'nothing'} to "
                    f"{max(block) if block else 'nothing'}."
                )

        # A Q A* alone. The noise contributes only its diagonal, added
        # afterwards -- see the class docstring on what that costs.
        core = normal.forward @ normal.prior_covariance @ normal.forward.adjoint

        rows: list[np.ndarray] = []
        columns: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for block in self._blocks:
            indices = np.asarray(block)
            size = indices.size
            local = EuclideanSpace(size)

            def restrict(vector: Any, indices: np.ndarray = indices) -> np.ndarray:
                # Galerkin, not components: the Galerkin matrix is the one in
                # which a self-adjoint operator is symmetric, and a principal
                # sub-block of a symmetric positive semidefinite matrix is
                # again one, which is what the eigendecomposition needs.
                return data_space.apply_gram(data_space.to_components(vector))[indices]

            def extend(
                components: np.ndarray,
                indices: np.ndarray = indices,
                dimension: int = dimension,
            ) -> Any:
                full = np.zeros(dimension)
                full[indices] = components
                return data_space.from_components(full)

            def block_value(
                components: np.ndarray,
                restrict: Callable[[Any], np.ndarray] = restrict,
                extend: Callable[[np.ndarray], Any] = extend,
            ) -> np.ndarray:
                return restrict(core(extend(components)))

            block_operator = LinearOperator.from_callables(
                local,
                local,
                block_value,
                adjoint=block_value,
                traits=Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE,
            )
            decomposition = random_eig(
                block_operator,
                rank=min(self._rank, size),
                rng=self._rng,
                n_jobs=self._n_jobs,
            )
            factor = decomposition.factor.matrix(form="components")
            approximation = (factor * decomposition.eigenvalues) @ factor.T

            grid_rows, grid_columns = np.meshgrid(indices, indices, indexing="ij")
            rows.append(grid_rows.ravel())
            columns.append(grid_columns.ravel())
            values.append(approximation.ravel())

        # An index no block covers would otherwise get an empty row and
        # column, filled only by the noise diagonal -- nothing at all for
        # exact data -- and the factorization would be singular. It keeps
        # its exact diagonal entry of A Q A* instead, probed off the factors.
        covered = np.zeros(dimension, dtype=bool)
        for block in self._blocks:
            covered[block] = True
        uncovered = np.flatnonzero(~covered)
        if uncovered.size:
            entries = _probe_normal_diagonal(
                normal, [[int(index)] for index in uncovered], n_jobs=self._n_jobs
            )
            rows.append(uncovered)
            columns.append(uncovered)
            values.append(np.asarray(entries, dtype=float))

        all_rows = np.concatenate(rows)
        all_columns = np.concatenate(columns)
        shape = (dimension, dimension)
        assembled = sparse.coo_matrix(
            (np.concatenate(values), (all_rows, all_columns)), shape=shape
        ).tocsr()
        # COO sums duplicates, so an entry covered by two blocks arrived twice.
        # Averaging over the blocks that cover it is what makes an overlapping
        # cover a smooth approximation rather than one whose seams are inflated
        # by their multiplicity — a doubled diagonal where two blocks meet is
        # not a better preconditioner, it is a different operator.
        multiplicity = sparse.coo_matrix(
            (np.ones(all_rows.size), (all_rows, all_columns)), shape=shape
        ).tocsr()
        multiplicity.data = 1.0 / multiplicity.data
        assembled = assembled.multiply(multiplicity).tocsr()
        error_covariance = normal.error_covariance
        if error_covariance is not None:
            assembled = assembled + sparse.diags(
                error_covariance.diagonals(offsets=(0,), form="galerkin")[0]
            )
        # Assembled as the Gaussian reading; the operator is a multiple of it.
        assembled = normal.scale * assembled
        factorized = sparse_linalg.splu(assembled.tocsc())

        def solve_fn(y: Any, x0: Any) -> SolveResult:
            weighted = data_space.apply_gram(data_space.to_components(y))
            return SolveResult(
                data_space.from_components(factorized.solve(weighted)), 0, 0.0, True
            )

        return InverseOperator(operator, self, solve_fn, traits=Traits.SELF_ADJOINT)


def _metric_diagonal(space: HilbertSpace, name: str, /) -> np.ndarray:
    """The data space's Gram as a diagonal, for a Galerkin assembly.

    A sparse kernel matrix can be turned into a Galerkin one by a row and a
    column scaling only when the metric is diagonal; a full Gram would fill it
    in and defeat the purpose of the class. Point data almost always live on a
    Euclidean or diagonally weighted space, so this is a real restriction only
    in principle.

    Args:
        space: the data space.
        name: the calling class, for the error message.

    Returns:
        The diagonal of the Gram matrix.

    Raises:
        ValueError: if the space's metric is not diagonal.
    """
    if isinstance(space, OrthonormalSpace):
        return np.ones(space.dim)
    if isinstance(space, DiagonalMetricSpace):
        return np.asarray(space.metric_values, dtype=float)
    raise ValueError(
        f"{name} needs a data space with a diagonal metric: it assembles a "
        f"sparse kernel matrix, and a full Gram matrix would fill it in. The "
        f"data space is {type(space).__name__}. Use a general preconditioner "
        f"such as JacobiPreconditioner or LocalizedPreconditioner instead."
    )


def gaspari_cohn(distances: np.ndarray, length: float, /) -> np.ndarray:
    """The Gaspari-Cohn (1999) correlation function, eq. (4.10).

    A compactly supported, positive definite function on ``[0, 2 * length]``.
    Its role here is not smoothing: truncating a covariance matrix to a
    neighborhood is not a positive definite operation, and multiplying by this
    before truncating is what makes the result positive definite again — a
    Schur product with a positive definite function, cut off exactly where the
    truncation cuts off.

    Args:
        distances: separations, in the same units as *length*.
        length: the correlation length. The function reaches zero at twice
            this and stays there.

    Returns:
        The taper, one value per distance.

    Raises:
        ValueError: if the length is not positive.
    """
    if length <= 0.0:
        raise ValueError(f"The taper length must be positive, got {length}.")
    scaled = np.asarray(distances, dtype=float) / length
    taper = np.zeros_like(scaled)

    near = scaled <= 1.0
    z = scaled[near]
    taper[near] = (
        1.0 - (5.0 / 3.0) * z**2 + (5.0 / 8.0) * z**3 + 0.5 * z**4 - 0.25 * z**5
    )

    far = (scaled > 1.0) & (scaled <= 2.0)
    z = scaled[far]
    taper[far] = (
        4.0
        - 5.0 * z
        + (5.0 / 3.0) * z**2
        + (5.0 / 8.0) * z**3
        - 0.5 * z**4
        + (1.0 / 12.0) * z**5
        - (2.0 / 3.0) / z
    )
    return taper


class InvariantDistancePreconditioner(LinearSolver):
    """``A Q A*`` written down directly, for point data and an invariant prior.

    When the forward operator is point evaluation and the prior is invariant,
    the two-point covariance depends on a pair of points only through the
    distance between them, so

    .. code-block:: text

        (A Q A*)_ij == k(d(p_i, p_j))

    with ``k`` a function of one variable. The whole matrix can then be written
    down from a table of distances, with **no** applications of the forward
    operator, its adjoint, or the prior covariance. That makes it by far the
    cheapest preconditioner in the library, and the only one whose cost does
    not scale with the model space at all.

    Entries beyond ``max_distance`` are dropped, which is what makes it sparse
    and what the sparse LU then exploits.

    Note:
        **Truncating a covariance matrix does not preserve positive
        definiteness**, and conjugate gradients needs a positive definite
        preconditioner. That is what ``taper`` is for: multiplying by a
        compactly supported positive definite function before cutting is a
        Schur product, which does preserve it. It defaults to True. Without
        a taper the preconditioner can be indefinite, and an indefinite
        preconditioner does not slow conjugate gradients down, it breaks it.

    The caller asserts two things this class cannot check: that the forward
    operator really is evaluation at *points*, in that order, and that the
    prior really is invariant. If either is false nothing fails loudly; the
    preconditioner is simply wrong. On a small case, compare it with the exact
    inverse of the normal operator before relying on it.
    """

    requires: ClassVar[Traits] = Traits.NONE
    requires_coordinates: ClassVar[bool] = True

    def __init__(
        self,
        space: "SymmetricSpace",
        points: Sequence[Any],
        max_distance: float,
        /,
        *,
        taper: bool = True,
        n_jobs: int | None = None,
    ) -> None:
        """
        Args:
            space: the symmetric space the model lives on, which supplies the
                distances and the covariance function.
            points: the observation points, in the order the data are in.
            max_distance: separations beyond which the covariance is taken to
                be zero. Zero gives a purely diagonal preconditioner, which
                needs one evaluation of ``k`` rather than a table.
            taper: multiply by a Gaspari-Cohn function of support
                ``max_distance`` before truncating. Keep this on unless you
                have checked the result is positive definite without it.
            n_jobs: workers for the error covariance's diagonal, where it
                must be probed. Serial by default.
        """
        if max_distance < 0.0:
            raise ValueError(f"The distance must be non-negative, got {max_distance}.")
        self._space = space
        self._points = list(points)
        self._max_distance = float(max_distance)
        self._taper = taper
        self._n_jobs = n_jobs

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        normal = _require_normal(operator, type(self).__name__)
        if not isinstance(normal, NormalOperator):
            raise TypeError(
                f"{type(self).__name__} reads the covariance function of the "
                f"prior *measure*, which only a NormalOperator built from a "
                f"Gaussian prior carries; a {type(normal).__name__} holds a "
                f"covariance operator and no measure. Use "
                f"NormalDiagonalPreconditioner or LocalizedPreconditioner for "
                f"it, which read the factors."
            )
        data_space = normal.data_space
        require_coordinates(data_space, data_space)
        dimension = data_space.dim
        if len(self._points) != dimension:
            raise ValueError(
                f"{len(self._points)} points were given for a data space of "
                f"dimension {dimension}. The points are the data, in order."
            )

        error_covariance = normal.error_covariance
        noise = (
            np.zeros(dimension)
            if error_covariance is None
            else error_covariance.diagonals(
                offsets=(0,), form="galerkin", n_jobs=self._n_jobs
            )[0]
        )
        # The table ``k(d_ij)`` is the covariance of the data *components*, so
        # it is neither the operator's component matrix nor its Galerkin one:
        # with the component covariance equal to ``C_c G^-1``, the table is
        # ``K == C_c G^-1`` and so ``C_gal == G K G``. Everything below is
        # assembled in the Galerkin form, which is what the solve then wants
        # (``N_gal^-1 G c_y`` gives the components of ``N^-1 y``) and what the
        # noise diagonal is already in. Without the two ``G`` factors the
        # matrix would be right only on an orthonormal data space, and the
        # Galerkin noise would be added to a non-Galerkin kernel.
        metric = _metric_diagonal(data_space, type(self).__name__)

        if self._max_distance == 0.0:
            # Every diagonal entry is k(0), the pointwise prior variance, which
            # is one number for an invariant measure.
            variance = float(
                self._space.covariance_function(normal.prior, np.zeros(1))[0]
            )
            inverse_diagonal = 1.0 / (
                normal.scale * (variance * metric * metric + noise)
            )

            def diagonal_solve(y: Any, x0: Any) -> SolveResult:
                weighted = data_space.apply_gram(data_space.to_components(y))
                return SolveResult(
                    data_space.from_components(inverse_diagonal * weighted),
                    0,
                    0.0,
                    True,
                )

            return InverseOperator(
                operator, self, diagonal_solve, traits=Traits.SELF_ADJOINT
            )

        rows, columns, distances = self._space.pairs_within_distance(
            self._points, self._max_distance, with_distances=True
        )
        values = self._space.covariance_function(normal.prior, distances)
        if self._taper:
            values = values * gaspari_cohn(distances, 0.5 * self._max_distance)

        assembled = sparse.coo_matrix(
            (values, (rows, columns)), shape=(dimension, dimension)
        )
        # G K G, with G diagonal, is a row and a column scaling: sparsity and
        # symmetry both survive it.
        scaling = sparse.diags(metric)
        assembled = (
            normal.scale * (scaling @ assembled @ scaling + sparse.diags(noise))
        ).tocsc()
        factorized = sparse_linalg.splu(assembled)

        def solve_fn(y: Any, x0: Any) -> SolveResult:
            weighted = data_space.apply_gram(data_space.to_components(y))
            return SolveResult(
                data_space.from_components(factorized.solve(weighted)), 0, 0.0, True
            )

        return InverseOperator(operator, self, solve_fn, traits=Traits.SELF_ADJOINT)
