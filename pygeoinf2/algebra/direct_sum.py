"""
Direct sums of Hilbert spaces, and block operators between them.

A vector in a direct sum is a **tuple** of component vectors. The container is
fixed-length and should never be restructured, and a tuple says so; the
components themselves stay mutable, so ``axpy`` still updates in place. The
space may carry labels, which gives named access without the vector wrapper
that was rejected everywhere else — worth having chiefly for the ``(model,
data)`` case, where indexing a joint vector by position is how sign errors get
made.

Block operators are **nonlinear by default**, with the linear case as a
specialisation. That is not speculative generality: v1 already builds the joint
law of model and data as a pushforward of a product measure through a block
operator,

    mu    = model_measure (x) noise_measure        on X (+) Y
    op    = [[I, 0], [A, I]]
    joint = op @ mu                                the law of (m, A m + e)

and the same shape with a nonlinear ``F`` in place of ``A`` gives the law of
``(m, F(m) + e)`` — a samplable measure, and a linearisation
``[[I, 0], [F'(m), I]]`` that comes from the same object rather than from a
separate construction.

See DESIGN.md section 3.3.
"""

from __future__ import annotations

from typing import Any, Hashable, Sequence

import numpy as np
from numpy.random import Generator

from ..traits import Traits, close
from .operators import LinearOperator, Operator
from .spaces import CoordinateSpace, HilbertSpace

__all__ = [
    "DirectSum",
    "BlockOperator",
    "BlockLinearOperator",
    "ColumnOperator",
    "ColumnLinearOperator",
    "RowOperator",
    "RowLinearOperator",
    "BlockDiagonalOperator",
    "BlockDiagonalLinearOperator",
]


class DirectSum[V](HilbertSpace[tuple]):
    """The direct sum of Hilbert spaces, with the summed inner product.

    Instantiating this returns a coordinate-providing subclass when every
    summand provides coordinates, so that ``isinstance(X, CoordinateSpace)``
    stays a reliable answer to "can this be given to a direct solver". A sum
    containing one coordinate-free summand is itself coordinate-free, which is
    the honest answer.
    """

    def __new__(
        cls,
        spaces: Sequence[HilbertSpace],
        /,
        *,
        labels: Sequence[str] | None = None,
    ) -> DirectSum:
        if cls is DirectSum and all(isinstance(s, CoordinateSpace) for s in spaces):
            return object.__new__(_CoordinateDirectSum)
        return object.__new__(cls)

    def __init__(
        self,
        spaces: Sequence[HilbertSpace],
        /,
        *,
        labels: Sequence[str] | None = None,
    ) -> None:
        spaces = tuple(spaces)
        if not spaces:
            raise ValueError("A direct sum needs at least one summand.")
        if labels is not None:
            labels = tuple(labels)
            if len(labels) != len(spaces):
                raise ValueError(
                    f"Got {len(labels)} labels for {len(spaces)} summands."
                )
            if len(set(labels)) != len(labels):
                raise ValueError("Labels must be distinct.")
        self._spaces = spaces
        self._labels = labels
        self._dim = sum(space.dim for space in spaces)

    # ----------------------------------------------------------------- #
    #                             Structure                             #
    # ----------------------------------------------------------------- #

    @property
    def dim(self) -> int:
        """The total dimension, summed over the summands."""
        return self._dim

    @property
    def subspaces(self) -> tuple[HilbertSpace, ...]:
        """The summands, in order."""
        return self._spaces

    @property
    def labels(self) -> tuple[str, ...] | None:
        """The summand labels, or None if the sum is unlabelled."""
        return self._labels

    def __len__(self) -> int:
        return len(self._spaces)

    def shares_vectors_with(self, other: HilbertSpace, /) -> bool:
        """True when the summands do, one for one."""
        if self is other:
            return True
        if not isinstance(other, DirectSum) or len(self) != len(other):
            return False
        return all(
            mine.shares_vectors_with(theirs)
            for mine, theirs in zip(self.subspaces, other.subspaces)
        )

    def _key(self) -> Hashable:
        """Identity is the summands alone.

        Labels are deliberately excluded. Two direct sums over the same
        summands are the same space whatever their components are called, and
        making labels part of identity would mean a block operator — which
        cannot know what its user chose to call things — lands on a space that
        compares unequal to the one its vectors came from.
        """
        return self._spaces

    def index(self, key: int | str) -> int:
        """Resolve a label or index to a position.

        Args:
            key: a summand's label, or its position.

        Returns:
            The position.

        Raises:
            KeyError: if a label is given and the sum has none, or none
                matching -- with the available labels in the message, since
                that is what the caller needs next.
            IndexError: if a position is out of range.
        """
        if isinstance(key, str):
            if self._labels is None:
                raise KeyError(f"This direct sum has no labels, so no {key!r}.")
            if key not in self._labels:
                raise KeyError(f"No summand labelled {key!r}; have {self._labels}.")
            return self._labels.index(key)
        if not 0 <= key < len(self._spaces):
            raise IndexError(f"Index {key} out of range for {len(self._spaces)}.")
        return key

    def subspace(self, key: int | str) -> HilbertSpace:
        """The summand at a label or index."""
        return self._spaces[self.index(key)]

    def component(self, x: tuple, key: int | str) -> object:
        """The named or indexed component of a vector."""
        return x[self.index(key)]

    def __repr__(self) -> str:
        if self._labels is not None:
            inner = ", ".join(
                f"{label}={space!r}" for label, space in zip(self._labels, self._spaces)
            )
        else:
            inner = ", ".join(repr(space) for space in self._spaces)
        return f"DirectSum({inner})"

    # ----------------------------------------------------------------- #
    #                          Vector operations                        #
    # ----------------------------------------------------------------- #

    def zero(self) -> tuple:
        """A tuple of the summands' zero vectors."""
        return tuple(space.zero() for space in self._spaces)

    def copy(self, x: tuple) -> tuple:
        """An independent copy, component by component."""
        return tuple(space.copy(xi) for space, xi in zip(self._spaces, x))

    def inner_product(self, x: tuple, y: tuple) -> float:
        """The sum of the summands' inner products."""
        return float(
            sum(
                space.inner_product(xi, yi) for space, xi, yi in zip(self._spaces, x, y)
            )
        )

    def axpy(self, a: float, x: tuple, y: tuple) -> tuple:
        """``y += a * x`` on each component."""
        return tuple(space.axpy(a, xi, yi) for space, xi, yi in zip(self._spaces, x, y))

    def scale_inplace(self, a: float, x: tuple) -> tuple:
        """``x *= a`` on each component."""
        return tuple(space.scale_inplace(a, xi) for space, xi in zip(self._spaces, x))

    def random(self, *, rng: Generator | None = None) -> tuple:
        """An arbitrary random vector, drawn independently on each summand."""
        return tuple(space.random(rng=rng) for space in self._spaces)

    def white_noise(self, *, rng: Generator | None = None) -> tuple:
        """Independent white noise on each summand.

        Correct because the summands are orthogonal: the covariance of the
        whole is block diagonal, and each block is the identity on its summand.
        """
        return tuple(space.white_noise(rng=rng) for space in self._spaces)

    # ----------------------------------------------------------------- #
    #                            Projections                            #
    # ----------------------------------------------------------------- #

    def projection(self, key: int | str) -> LinearOperator:
        """The orthogonal projection onto one summand, as an operator.

        Memoised per index, for the same reason ``adjoint`` is: the palindrome
        rule compares factors by identity, so a rebuilt projection would make
        ``P @ C @ P.adjoint`` unrecognisable as a congruence.
        """
        i = self.index(key)
        cache = self.__dict__.setdefault("_projection_cache", {})
        if i in cache:
            return cache[i]
        space = self._spaces[i]
        spaces = self._spaces

        def value(x: tuple):
            return x[i]

        def adjoint(y):
            return tuple(
                y if j == i else other.zero() for j, other in enumerate(spaces)
            )

        operator = LinearOperator.from_callables(self, space, value, adjoint=adjoint)
        cache[i] = operator
        return operator

    def inclusion(self, key: int | str) -> LinearOperator:
        """The isometric inclusion of one summand into the sum."""
        return self.projection(key).adjoint


class _CoordinateDirectSum(DirectSum, CoordinateSpace):
    """A direct sum all of whose summands provide coordinates.

    Components are the summands' components concatenated, so the Gram matrix is
    block diagonal and every metric operation splits.
    """

    def __init__(
        self,
        spaces: Sequence[HilbertSpace],
        /,
        *,
        labels: Sequence[str] | None = None,
    ) -> None:
        DirectSum.__init__(self, spaces, labels=labels)
        bounds = np.cumsum([0] + [space.dim for space in self._spaces])
        self._slices = tuple(
            slice(int(bounds[i]), int(bounds[i + 1])) for i in range(len(self._spaces))
        )

    def to_components(self, x: tuple) -> np.ndarray:
        """The summands' components, concatenated."""
        return np.concatenate(
            [space.to_components(xi) for space, xi in zip(self._spaces, x)]
        )

    def from_components(self, c: np.ndarray) -> tuple:
        """Split the array and rebuild each summand's vector."""
        return tuple(
            space.from_components(c[s]) for space, s in zip(self._spaces, self._slices)
        )

    def apply_gram(self, c: np.ndarray) -> np.ndarray:
        """``G c``, applied blockwise. The Gram matrix is block diagonal."""
        return np.concatenate(
            [space.apply_gram(c[s]) for space, s in zip(self._spaces, self._slices)]
        )

    def solve_gram(self, c: np.ndarray) -> np.ndarray:
        """``G^-1 c``, applied blockwise."""
        return np.concatenate(
            [space.solve_gram(c[s]) for space, s in zip(self._spaces, self._slices)]
        )

    def apply_gram_to_columns(self, columns: np.ndarray, /) -> np.ndarray:
        """``G`` on every column, one summand's block of rows at a time."""
        return np.concatenate(
            [
                space.apply_gram_to_columns(columns[s])
                for space, s in zip(self._spaces, self._slices)
            ]
        )

    def gram_diagonal(self) -> np.ndarray:
        """The summands' Gram diagonals, concatenated."""
        return np.concatenate([space.gram_diagonal() for space in self._spaces])

    def solve_gram_to_columns(self, columns: np.ndarray, /) -> np.ndarray:
        """``G^-1`` on every column, one summand's block of rows at a time."""
        return np.concatenate(
            [
                space.solve_gram_to_columns(columns[s])
                for space, s in zip(self._spaces, self._slices)
            ]
        )

    def white_noise_components(self, *, rng: Generator | None = None) -> np.ndarray:
        """Independent white noise components on each summand."""
        return np.concatenate(
            [space.white_noise_components(rng=rng) for space in self._spaces]
        )

    @property
    def is_orthonormal(self) -> bool:
        """True only when every summand has an orthonormal basis."""
        return all(space.is_orthonormal for space in self._spaces)

    @property
    def has_diagonal_metric(self) -> bool:
        """True only when every summand's metric is diagonal.

        The Gram matrix is block diagonal, so it is diagonal exactly when
        every block is.
        """
        return all(space.has_diagonal_metric for space in self._spaces)


# --------------------------------------------------------------------- #
#                            Block operators                            #
# --------------------------------------------------------------------- #


def _check_grid(
    blocks: Sequence[Sequence[Operator]],
) -> tuple[tuple[HilbertSpace, ...], tuple[HilbertSpace, ...]]:
    """Validate a rectangular grid of operators; return its domains and codomains."""
    if not blocks or not blocks[0]:
        raise ValueError("A block operator needs at least one block.")
    width = len(blocks[0])
    domains = tuple(operator.domain for operator in blocks[0])
    codomains = []
    for i, row in enumerate(blocks):
        if len(row) != width:
            raise ValueError(f"Row {i} has {len(row)} blocks, but row 0 has {width}.")
        for j, operator in enumerate(row):
            if operator.domain != domains[j]:
                raise ValueError(
                    f"Block ({i}, {j}) has domain {operator.domain!r}, but "
                    f"column {j} is {domains[j]!r}."
                )
        codomain = row[0].codomain
        for j, operator in enumerate(row):
            if operator.codomain != codomain:
                raise ValueError(
                    f"Block ({i}, {j}) has codomain {operator.codomain!r}, but "
                    f"row {i} is {codomain!r}."
                )
        codomains.append(codomain)
    return domains, tuple(codomains)


def _block_traits(
    blocks: Sequence[Sequence[Operator]],
    domains: tuple[HilbertSpace, ...],
    codomains: tuple[HilbertSpace, ...],
) -> Traits:
    """A block operator is self-adjoint when its grid is its own adjoint transpose."""
    if domains != codomains or len(blocks) != len(blocks[0]):
        return Traits.NONE
    n = len(blocks)
    for i in range(n):
        for j in range(n):
            if not LinearOperator.adjoints_are_linked(blocks[i][j], blocks[j][i]):
                return Traits.NONE
    return close(Traits.SELF_ADJOINT)


def _known_block_matrix(
    blocks: Sequence[Sequence[LinearOperator]],
    domain: HilbertSpace,
    codomain: HilbertSpace,
    form: str,
) -> np.ndarray | None:
    """The dense matrix of a grid of blocks, when every block knows its own.

    Assembled in the components form, where blocks simply tile, and converted
    to the Galerkin form afterwards with the direct sum's block-diagonal
    metric. ``None`` if either side lacks coordinates or any block answers
    ``None``.
    """
    if not isinstance(domain, CoordinateSpace) or not isinstance(
        codomain, CoordinateSpace
    ):
        return None
    rows = []
    for row in blocks:
        known = [block._known_matrix("components") for block in row]
        if any(part is None for part in known):
            return None
        rows.append(known)
    matrix = np.block(rows)
    if form == "components":
        return matrix
    return codomain.apply_gram_to_columns(matrix)


def _known_block_diagonals(
    diagonal_blocks: Sequence[LinearOperator],
    domains: Sequence[HilbertSpace],
    codomains: Sequence[HilbertSpace],
    offsets: tuple[int, ...],
    form: str,
) -> np.ndarray | None:
    """The main diagonal of a block operator, from its diagonal blocks.

    Only the main diagonal: an off-diagonal of the whole crosses block
    boundaries, and only the main one is what a Jacobi or a structure-aware
    preconditioner reads. The off-diagonal blocks contribute nothing to it
    when every diagonal block is square, which is the case handled; otherwise
    ``None`` and the base class probes.
    """
    if tuple(offsets) != (0,):
        return None
    if any(d.dim != c.dim for d, c in zip(domains, codomains)):
        return None
    parts = []
    for block in diagonal_blocks:
        known = block._known_diagonals((0,), form)
        if known is None:
            return None
        parts.append(known[0])
    return np.concatenate(parts)[None, :]


class BlockOperator(Operator):
    """An operator between direct sums, given as a grid of operators.

    Nonlinear by default. The derivative is the block operator of the blocks'
    derivatives, which is what makes a linearised joint model fall out of the
    joint model itself.
    """

    def __new__(cls, blocks: Sequence[Sequence[Operator]]) -> BlockOperator:
        if cls is BlockOperator and all(
            isinstance(operator, LinearOperator) for row in blocks for operator in row
        ):
            return object.__new__(BlockLinearOperator)
        return object.__new__(cls)

    def __init__(self, blocks: Sequence[Sequence[Operator]]) -> None:
        self._blocks = tuple(tuple(row) for row in blocks)
        self._domains, self._codomains = _check_grid(self._blocks)
        Operator.__init__(self, DirectSum(self._domains), DirectSum(self._codomains))

    @property
    def blocks(self) -> tuple[tuple[Operator, ...], ...]:
        """The grid of blocks, row-major."""
        return self._blocks

    @property
    def row_dim(self) -> int:
        """The number of block rows."""
        return len(self._blocks)

    @property
    def col_dim(self) -> int:
        """The number of block columns."""
        return len(self._blocks[0])

    def block(self, i: int, j: int) -> Operator:
        """The block at row ``i``, column ``j``."""
        return self._blocks[i][j]

    @property
    def has_derivative(self) -> bool:
        """True only when every block carries a derivative."""
        return all(operator.has_derivative for row in self._blocks for operator in row)

    def _value(self, x: tuple) -> tuple:
        result = []
        for i, row in enumerate(self._blocks):
            codomain = self._codomains[i]
            y = codomain.zero()
            for j, operator in enumerate(row):
                y = codomain.axpy(1.0, operator(x[j]), y)
            result.append(y)
        return tuple(result)

    def _derivative(self, x: tuple) -> LinearOperator:
        return BlockLinearOperator(
            [
                [operator.derivative(x[j]) for j, operator in enumerate(row)]
                for row in self._blocks
            ]
        )

    def __repr__(self) -> str:
        return f"BlockOperator({self.row_dim}x{self.col_dim})"


class BlockLinearOperator(BlockOperator, LinearOperator):
    """A block operator whose every block is linear."""

    def __init__(self, blocks: Sequence[Sequence[LinearOperator]]) -> None:
        self._blocks = tuple(tuple(row) for row in blocks)
        self._domains, self._codomains = _check_grid(self._blocks)
        LinearOperator.__init__(
            self,
            DirectSum(self._domains),
            DirectSum(self._codomains),
            traits=_block_traits(self._blocks, self._domains, self._codomains),
        )

    def _adjoint_value(self, y: tuple) -> tuple:
        result = []
        for j, domain in enumerate(self._domains):
            x = domain.zero()
            for i in range(self.row_dim):
                x = domain.axpy(1.0, self._blocks[i][j].adjoint(y[i]), x)
            result.append(x)
        return tuple(result)

    def _make_adjoint(self) -> LinearOperator:
        """The adjoint of a block operator is the transposed grid of adjoints."""
        result = BlockLinearOperator(
            [
                [self._blocks[i][j].adjoint for i in range(self.row_dim)]
                for j in range(self.col_dim)
            ]
        )
        result._link_adjoint(self)
        return result

    def _known_matrix(self, form: str) -> np.ndarray | None:
        return _known_block_matrix(self._blocks, self.domain, self.codomain, form)

    def _known_diagonals(
        self, offsets: tuple[int, ...], form: str
    ) -> np.ndarray | None:
        if self.row_dim != self.col_dim:
            return None
        return _known_block_diagonals(
            [self._blocks[i][i] for i in range(self.row_dim)],
            self._domains,
            self._codomains,
            offsets,
            form,
        )

    def __repr__(self) -> str:
        return f"BlockLinearOperator({self.row_dim}x{self.col_dim})"


class ColumnOperator(Operator):
    """``x -> (F_1(x), ..., F_n(x))``, from one space into a direct sum.

    Distinct from a one-column ``BlockOperator``, whose domain would be
    ``DirectSum([X])`` rather than ``X``.
    """

    def __new__(cls, operators: Sequence[Operator]) -> ColumnOperator:
        if cls is ColumnOperator and all(
            isinstance(operator, LinearOperator) for operator in operators
        ):
            return object.__new__(ColumnLinearOperator)
        return object.__new__(cls)

    def __init__(self, operators: Sequence[Operator]) -> None:
        self._operators = tuple(operators)
        if not self._operators:
            raise ValueError("A column operator needs at least one block.")
        domain = self._operators[0].domain
        for i, operator in enumerate(self._operators):
            if operator.domain != domain:
                raise ValueError(
                    f"Block {i} has domain {operator.domain!r}, expected {domain!r}."
                )
        Operator.__init__(
            self, domain, DirectSum([op.codomain for op in self._operators])
        )

    @property
    def operators(self) -> tuple[Operator, ...]:
        """The blocks, in order."""
        return self._operators

    @property
    def has_derivative(self) -> bool:
        """True only when every block carries a derivative."""
        return all(operator.has_derivative for operator in self._operators)

    def _value(self, x: object) -> tuple:
        return tuple(operator(x) for operator in self._operators)

    def _derivative(self, x: object) -> LinearOperator:
        return ColumnLinearOperator(
            [operator.derivative(x) for operator in self._operators]
        )


class ColumnLinearOperator(ColumnOperator, LinearOperator):
    """A column operator whose every block is linear."""

    def __init__(self, operators: Sequence[LinearOperator]) -> None:
        self._operators = tuple(operators)
        domain = self._operators[0].domain
        for i, operator in enumerate(self._operators):
            if operator.domain != domain:
                raise ValueError(
                    f"Block {i} has domain {operator.domain!r}, expected {domain!r}."
                )
        LinearOperator.__init__(
            self, domain, DirectSum([op.codomain for op in self._operators])
        )

    def _adjoint_value(self, y: tuple) -> object:
        domain = self.domain
        x = domain.zero()
        for operator, yi in zip(self._operators, y):
            x = domain.axpy(1.0, operator.adjoint(yi), x)
        return x

    def _make_adjoint(self) -> LinearOperator:
        result = RowLinearOperator([op.adjoint for op in self._operators])
        result._link_adjoint(self)
        return result


class RowOperator(Operator):
    """``(x_1, ..., x_n) -> F_1(x_1) + ... + F_n(x_n)``, from a direct sum."""

    def __new__(cls, operators: Sequence[Operator]) -> RowOperator:
        if cls is RowOperator and all(
            isinstance(operator, LinearOperator) for operator in operators
        ):
            return object.__new__(RowLinearOperator)
        return object.__new__(cls)

    def __init__(self, operators: Sequence[Operator]) -> None:
        self._operators = tuple(operators)
        if not self._operators:
            raise ValueError("A row operator needs at least one block.")
        codomain = self._operators[0].codomain
        for i, operator in enumerate(self._operators):
            if operator.codomain != codomain:
                raise ValueError(
                    f"Block {i} has codomain {operator.codomain!r}, "
                    f"expected {codomain!r}."
                )
        Operator.__init__(
            self, DirectSum([op.domain for op in self._operators]), codomain
        )

    @property
    def operators(self) -> tuple[Operator, ...]:
        """The blocks, in order."""
        return self._operators

    @property
    def has_derivative(self) -> bool:
        """True only when every block carries a derivative."""
        return all(operator.has_derivative for operator in self._operators)

    def _value(self, x: tuple) -> object:
        codomain = self.codomain
        y = codomain.zero()
        for operator, xi in zip(self._operators, x):
            y = codomain.axpy(1.0, operator(xi), y)
        return y

    def _derivative(self, x: tuple) -> LinearOperator:
        return RowLinearOperator(
            [operator.derivative(xi) for operator, xi in zip(self._operators, x)]
        )


class RowLinearOperator(RowOperator, LinearOperator):
    """A row operator whose every block is linear."""

    def __init__(self, operators: Sequence[LinearOperator]) -> None:
        self._operators = tuple(operators)
        codomain = self._operators[0].codomain
        for i, operator in enumerate(self._operators):
            if operator.codomain != codomain:
                raise ValueError(
                    f"Block {i} has codomain {operator.codomain!r}, "
                    f"expected {codomain!r}."
                )
        LinearOperator.__init__(
            self, DirectSum([op.domain for op in self._operators]), codomain
        )

    def _adjoint_value(self, y: object) -> tuple:
        return tuple(operator.adjoint(y) for operator in self._operators)

    def _make_adjoint(self) -> LinearOperator:
        result = ColumnLinearOperator([op.adjoint for op in self._operators])
        result._link_adjoint(self)
        return result


class BlockDiagonalOperator(Operator):
    """``(x_i) -> (F_i(x_i))``, acting on each summand independently."""

    def __new__(cls, operators: Sequence[Operator]) -> BlockDiagonalOperator:
        if cls is BlockDiagonalOperator and all(
            isinstance(operator, LinearOperator) for operator in operators
        ):
            return object.__new__(BlockDiagonalLinearOperator)
        return object.__new__(cls)

    def __init__(self, operators: Sequence[Operator]) -> None:
        self._operators = tuple(operators)
        if not self._operators:
            raise ValueError("A block diagonal operator needs at least one block.")
        Operator.__init__(
            self,
            DirectSum([op.domain for op in self._operators]),
            DirectSum([op.codomain for op in self._operators]),
        )

    @property
    def operators(self) -> tuple[Operator, ...]:
        """The diagonal blocks, in order."""
        return self._operators

    @property
    def has_derivative(self) -> bool:
        """True only when every block carries a derivative."""
        return all(operator.has_derivative for operator in self._operators)

    def _value(self, x: tuple) -> tuple:
        return tuple(operator(xi) for operator, xi in zip(self._operators, x))

    def _derivative(self, x: tuple) -> LinearOperator:
        return BlockDiagonalLinearOperator(
            [operator.derivative(xi) for operator, xi in zip(self._operators, x)]
        )


class BlockDiagonalLinearOperator(BlockDiagonalOperator, LinearOperator):
    """A block diagonal operator whose every block is linear.

    Traits are the intersection of the blocks' own, which is exactly right: the
    whole is self-adjoint iff every block is, positive definite iff every block
    is, and so on.
    """

    def __init__(self, operators: Sequence[LinearOperator]) -> None:
        self._operators = tuple(operators)
        traits = self._operators[0].traits
        for operator in self._operators[1:]:
            traits = traits & operator.traits
        LinearOperator.__init__(
            self,
            DirectSum([op.domain for op in self._operators]),
            DirectSum([op.codomain for op in self._operators]),
            traits=close(traits),
        )

    def _adjoint_value(self, y: tuple) -> tuple:
        return tuple(operator.adjoint(yi) for operator, yi in zip(self._operators, y))

    def _make_adjoint(self) -> LinearOperator:
        result = BlockDiagonalLinearOperator([op.adjoint for op in self._operators])
        result._link_adjoint(self)
        return result

    def _known_matrix(self, form: str) -> np.ndarray | None:
        count = len(self._operators)
        grid = [
            [
                self._operators[i]
                if i == j
                else LinearOperator.zero(
                    self._operators[j].domain, codomain=self._operators[i].codomain
                )
                for j in range(count)
            ]
            for i in range(count)
        ]
        return _known_block_matrix(grid, self.domain, self.codomain, form)

    def _known_diagonals(
        self, offsets: tuple[int, ...], form: str
    ) -> np.ndarray | None:
        return _known_block_diagonals(
            self._operators,
            tuple(op.domain for op in self._operators),
            tuple(op.codomain for op in self._operators),
            offsets,
            form,
        )

    def apply_block(
        self, vectors: Sequence[Any], /, *, n_jobs: int | None = None
    ) -> list[Any]:
        """Each block applied to its summand's vectors as a block.

        Args:
            vectors: the inputs, tuples with one entry per summand.
            n_jobs: workers, passed to each block.

        Returns:
            The images, in order.
        """
        vectors = list(vectors)
        parts = [
            operator.apply_block([x[i] for x in vectors], n_jobs=n_jobs)
            for i, operator in enumerate(self._operators)
        ]
        return [tuple(part[k] for part in parts) for k in range(len(vectors))]

    def _adjoint_apply_block(
        self, vectors: Sequence[Any], /, *, n_jobs: int | None = None
    ) -> list[Any]:
        vectors = list(vectors)
        parts = [
            operator._adjoint_apply_block([y[i] for y in vectors], n_jobs=n_jobs)
            for i, operator in enumerate(self._operators)
        ]
        return [tuple(part[k] for part in parts) for k in range(len(vectors))]
