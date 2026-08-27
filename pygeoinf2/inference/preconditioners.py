"""
Preconditioners that need the normal operator's factors, not its assembly.

The generic preconditioners in :mod:`pygeoinf2.numerics.preconditioners` take
any operator and approximate it. These take a
:class:`~pygeoinf2.inference.normal.NormalOperator` and exploit the fact that it
is ``A Q A* + R``, which is why they live here: an assembled operator no longer
knows that, and in v1 they had to be methods on the inversion class for exactly
that reason. Here they are ordinary :class:`LinearSolver` objects and can be
passed to any solver's ``with_preconditioner``.

Both are for the *data-space* formalism. Neither identity has a model-space
counterpart — ``Q^-1 + A* R^-1 A`` has no cheap diagonal, because the expensive
part is the operator sandwiched in the middle rather than at the ends.
"""

from __future__ import annotations

from typing import Any, ClassVar, Sequence

import numpy as np

from numpy.random import Generator

from ..algebra.operators import LinearOperator, require_coordinates
from ..numerics.solvers import InverseOperator, LinearSolver, SolveResult
from ..traits import Traits
from .normal import NormalOperator

__all__ = [
    "NormalDiagonalPreconditioner",
    "LocalisedPreconditioner",
    "InvariantDistancePreconditioner",
    "gaspari_cohn",
]


def _require_normal(operator: LinearOperator, name: str) -> NormalOperator:
    if not isinstance(operator, NormalOperator):
        raise TypeError(
            f"{name} needs the factors A, Q and R, so it takes a "
            f"NormalOperator rather than an assembled one. Build it from an "
            f"inversion's .normal_operator, or from .surrogate(...). Got "
            f"{type(operator).__name__}."
        )
    if operator.formalism != "data_space":
        raise ValueError(
            f"{name} is derived for the data-space normal operator "
            f"A Q A* + R. This one is assembled in the model space, where the "
            f"identity it uses has no counterpart."
        )
    return operator


class NormalDiagonalPreconditioner(LinearSolver):
    """Invert the diagonal of ``A Q A* + R``, obtained without forming it.

    The identity is ``<v, A Q A* v> == <A* v, Q A* v>``, so a diagonal entry
    costs one adjoint application and one application of the prior covariance —
    never an application of the assembled operator, and never the forward
    operator at all.

    The saving that matters is *blocks*. Given a partition of the data indices,
    the probe vector for a block is its normalised indicator, so the whole block
    shares one adjoint application and is assigned one representative variance.
    A thousand data in fifty blocks costs fifty adjoint applications rather than
    a thousand, and for data that cluster — stations in a region, rays through a
    corridor — the representative value is close to each member's own.

    With no blocks this computes the exact Galerkin diagonal, and then agrees
    with :class:`~pygeoinf2.numerics.preconditioners.JacobiPreconditioner`
    applied to the assembled operator. The test suite checks that it does,
    which is what pins the metric handling: a diagonal is a statement about a
    basis, and the Galerkin form is the one in which a self-adjoint operator is
    symmetric.
    """

    requires: ClassVar[Traits] = Traits.NONE
    requires_coordinates: ClassVar[bool] = True

    def __init__(
        self,
        /,
        *,
        blocks: Sequence[Sequence[int]] | None = None,
        floor: float = 1e-14,
    ) -> None:
        """
        Args:
            blocks: a partition of the data indices. Every index must appear
                exactly once; a partial cover is refused rather than silently
                left at zero. None means one block per index, which is exact.
            floor: diagonal entries at or below this are left uninverted.
        """
        self._blocks = None if blocks is None else [list(b) for b in blocks]
        self._floor = floor

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
        data_space = normal.data_space
        model_space = normal.model_space
        require_coordinates(data_space, data_space)

        adjoint = normal.forward.adjoint
        prior_covariance = normal.prior_covariance
        dimension = data_space.dim
        blocks = self._partition(dimension)

        diagonal = np.zeros(dimension)
        components = np.zeros(dimension)
        for block in blocks:
            components[:] = 0.0
            components[block] = 1.0 / len(block)
            pulled = adjoint(data_space.from_components(components))
            diagonal[block] = model_space.inner_product(
                pulled, prior_covariance(pulled)
            )

        error_covariance = normal.error_covariance
        if error_covariance is not None:
            diagonal = diagonal + np.diag(error_covariance.matrix(form="galerkin"))

        safe = np.where(np.abs(diagonal) > self._floor, diagonal, 1.0)
        inverse_diagonal = 1.0 / safe

        def solve_fn(y: Any, x0: Any) -> SolveResult:
            weighted = data_space.apply_gram(data_space.to_components(y))
            return SolveResult(
                data_space.from_components(inverse_diagonal * weighted), 0, 0.0, True
            )

        return InverseOperator(operator, self, solve_fn, traits=Traits.SELF_ADJOINT)


class LocalisedPreconditioner(LinearSolver):
    """Approximate ``A Q A* + R`` by the blocks that actually couple.

    Data that are near each other — stations in a region, rays sharing a
    corridor — have strongly correlated predictions, and data that are far apart
    have nearly none. So the normal operator is close to block-sparse in a
    sensible ordering, and a preconditioner can keep the blocks and drop the
    rest.

    Each block is approximated by a randomised Nystrom decomposition of
    ``P N P*`` at the given rank rather than by the exact sub-block, so a block
    of size 200 still costs only ``rank`` applications. The blocks are assembled
    into a sparse matrix, the noise diagonal is added, and the result is
    factorised once with a sparse LU.

    Unlike :class:`NormalDiagonalPreconditioner` the blocks may **overlap** and
    need not cover every index: this is an approximation to the operator, not a
    partition of it. Overlapping contributions add, which is what makes an
    overlapping cover behave sensibly at the seams.

    Note:
        Only ``A Q A*`` is treated block-wise; the error covariance contributes
        its diagonal. So a single block covering every index, at full rank,
        reproduces ``N^-1`` exactly when ``R`` is diagonal — which is the usual
        case, and what the test suite checks — and drops ``R``'s off-diagonal
        otherwise. That is a real approximation and not a free one: with a
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
    ) -> None:
        """
        Args:
            blocks: index groups that couple strongly. May overlap.
            rank: the rank of the Nystrom approximation within each block,
                capped at the block size.
            rng: the generator for the randomised range finder.
        """
        self._blocks = [list(block) for block in blocks]
        if not self._blocks:
            raise ValueError("At least one block is needed.")
        if rank < 1:
            raise ValueError(f"The rank must be positive, got {rank}.")
        self._rank = rank
        self._rng = rng

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        import scipy.sparse as sparse
        import scipy.sparse.linalg as sparse_linalg

        from ..algebra.spaces import EuclideanSpace
        from ..numerics.randomised import random_eig

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
                # which a self-adjoint operator is symmetric (DESIGN §5.6), and
                # a principal sub-block of a symmetric positive semidefinite
                # matrix is again one, which is what Nystrom needs.
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
                restrict: Any = restrict,
                extend: Any = extend,
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
                block_operator, rank=min(self._rank, size), rng=self._rng
            )
            factor = decomposition.factor.matrix(form="components")
            approximation = (factor * decomposition.eigenvalues) @ factor.T

            grid_rows, grid_columns = np.meshgrid(indices, indices, indexing="ij")
            rows.append(grid_rows.ravel())
            columns.append(grid_columns.ravel())
            values.append(approximation.ravel())

        assembled = sparse.coo_matrix(
            (
                np.concatenate(values),
                (np.concatenate(rows), np.concatenate(columns)),
            ),
            shape=(dimension, dimension),
        )
        error_covariance = normal.error_covariance
        if error_covariance is not None:
            assembled = assembled + sparse.diags(
                np.diag(error_covariance.matrix(form="galerkin"))
            )
        factorised = sparse_linalg.splu(assembled.tocsc())

        def solve_fn(y: Any, x0: Any) -> SolveResult:
            weighted = data_space.apply_gram(data_space.to_components(y))
            return SolveResult(
                data_space.from_components(factorised.solve(weighted)), 0, 0.0, True
            )

        return InverseOperator(operator, self, solve_fn, traits=Traits.SELF_ADJOINT)


def gaspari_cohn(distances: np.ndarray, length: float, /) -> np.ndarray:
    """The Gaspari-Cohn (1999) correlation function, eq. (4.10).

    A compactly supported, positive definite function on ``[0, 2 * length]``.
    Its role here is not smoothing: truncating a covariance matrix to a
    neighbourhood is not a positive definite operation, and multiplying by this
    before truncating is what makes the result positive definite again — a
    Schur product with a positive definite function, cut off exactly where the
    truncation cuts off.
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
        Schur product, which does preserve it. It defaults to True here.

        v1's version defaulted to no taper, which is the most likely reason it
        never performed as well as hoped: without one the "preconditioner" can
        be indefinite, and an indefinite preconditioner does not slow CG down,
        it breaks it. See DESIGN.md section 23.6.

    The caller asserts two things this class cannot check: that the forward
    operator really is evaluation at *points*, in that order, and that the
    prior really is invariant. Both are false quietly rather than loudly, so
    the test suite checks the assembled matrix against ``A Q A*`` directly.
    """

    requires: ClassVar[Traits] = Traits.NONE
    requires_coordinates: ClassVar[bool] = True

    def __init__(
        self,
        space: Any,
        points: Sequence[Any],
        max_distance: float,
        /,
        *,
        taper: bool = True,
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
        """
        if max_distance < 0.0:
            raise ValueError(f"The distance must be non-negative, got {max_distance}.")
        self._space = space
        self._points = list(points)
        self._max_distance = float(max_distance)
        self._taper = taper

    def _invert(self, operator: LinearOperator) -> InverseOperator:
        import scipy.sparse as sparse
        import scipy.sparse.linalg as sparse_linalg

        normal = _require_normal(operator, type(self).__name__)
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
            else np.diag(error_covariance.matrix(form="galerkin"))
        )

        if self._max_distance == 0.0:
            # Every diagonal entry is k(0), the pointwise prior variance, which
            # is one number for an invariant measure.
            variance = float(
                self._space.covariance_function(normal.prior, np.zeros(1))[0]
            )
            inverse_diagonal = 1.0 / (variance + noise)

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
        assembled = (assembled + sparse.diags(noise)).tocsc()
        factorised = sparse_linalg.splu(assembled)

        def solve_fn(y: Any, x0: Any) -> SolveResult:
            weighted = data_space.apply_gram(data_space.to_components(y))
            return SolveResult(
                data_space.from_components(factorised.solve(weighted)), 0, 0.0, True
            )

        return InverseOperator(operator, self, solve_fn, traits=Traits.SELF_ADJOINT)
