"""
Module for direct sums of Hilbert spaces and related block operators.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np

from .hilbert_space import HilbertSpace
from .operators import LinearOperator
from .forms import LinearForm


class HilbertSpaceDirectSum(HilbertSpace):
    """
    A Hilbert space formed from the direct sum of a list of other spaces.

    Vectors in this space are represented as lists of vectors, where each
    element of the list is a vector from the corresponding subspace.
    """

    def __init__(self, spaces: List[HilbertSpace]) -> None:
        """
        Initializes the direct sum space.

        Args:
            spaces (List[HilbertSpace]): A list of Hilbert spaces to combine
                in the direct sum.
        """
        self._spaces: List[HilbertSpace] = spaces
        dim = sum([space.dim for space in spaces])
        super().__init__(
            dim,
            self.__to_components,
            self.__from_components,
            self.__inner_product,
            self.__to_dual,
            self.__from_dual,
            add=self.__add,
            subtract=self.__subtract,
            multiply=self.__multiply,
            ax=self.__ax,
            axpy=self.__axpy,
            copy=self.__copy,
        )

    @property
    def subspaces(self) -> List[HilbertSpace]:
        """Returns the list of subspaces that form the direct sum."""
        return self._spaces

    @property
    def number_of_subspaces(self) -> int:
        """Returns the number of subspaces in the direct sum."""
        return len(self.subspaces)

    def subspace(self, i: int) -> HilbertSpace:
        """
        Returns the i-th subspace.

        Args:
            i (int): The index of the subspace to retrieve.
        """
        return self.subspaces[i]

    def subspace_projection(self, i: int) -> LinearOperator:
        """
        Returns the projection operator onto the i-th subspace.

        Args:
            i (int): The index of the subspace to project onto.
        """
        return LinearOperator(
            self,
            self.subspaces[i],
            lambda xs: self._subspace_projection_mapping(i, xs),
            adjoint_mapping=lambda x: self._subspace_inclusion_mapping(i, x),
        )

    def subspace_inclusion(self, i: int) -> LinearOperator:
        """
        Returns the inclusion operator from the i-th subspace into the sum.

        Args:
            i (int): The index of the subspace to include from.
        """
        return LinearOperator(
            self.subspaces[i],
            self,
            lambda x: self._subspace_inclusion_mapping(i, x),
            adjoint_mapping=lambda xs: self._subspace_projection_mapping(i, xs),
        )

    def canonical_dual_isomorphism(self, xps: List[LinearForm]) -> LinearForm:
        """
        Maps a list of dual vectors to a single dual vector on the sum space.

        This is the canonical isomorphism from the direct sum of the dual spaces
        to the dual of the direct sum space.

        Args:
            xps (List[LinearForm]): A list of dual vectors, one for each
                subspace.
        """
        if len(xps) != self.number_of_subspaces:
            raise ValueError("Incorrect number of dual vectors provided.")
        return LinearForm(
            self,
            mapping=lambda x: sum(
                xp(self.subspace_projection(i)(x)) for i, xp in enumerate(xps)
            ),
        )

    def canonical_dual_inverse_isomorphism(self, xp: LinearForm) -> List[LinearForm]:
        """
        Maps a dual vector on the sum space to a list of dual vectors.

        This is the inverse of the canonical isomorphism, projecting the action
        of a dual vector onto each subspace.

        Args:
            xp (LinearForm): A dual vector on the direct sum space.
        """
        return [
            LinearForm(space, mapping=lambda x, j=i: xp(self.subspace_inclusion(j)(x)))
            for i, space in enumerate(self.subspaces)
        ]

    def __eq__(self, other: object) -> bool:
        """
        Checks for mathematical equality with another direct sum space.

        Two direct sum spaces are equal if they are composed of the same
        number of subspaces, and each corresponding subspace is equal.
        """
        if not isinstance(other, HilbertSpaceDirectSum):
            return NotImplemented
    
        # This relies on the subspaces having their own __eq__ methods.
        return self.subspaces == other.subspaces

        
    def __to_components(self, xs: List[Any]) -> np.ndarray:
        cs = [space.to_components(x) for space, x in zip(self._spaces, xs)]
        return np.concatenate(cs, 0)

    def __from_components(self, c: np.ndarray) -> List[Any]:
        xs = []
        i = 0
        for space in self._spaces:
            j = i + space.dim
            x = space.from_components(c[i:j])
            xs.append(x)
            i = j
        return xs

    def __inner_product(self, x1s: List[Any], x2s: List[Any]) -> float:
        return sum(
            space.inner_product(x1, x2) for space, x1, x2 in zip(self._spaces, x1s, x2s)
        )

    def __to_dual(self, xs: List[Any]) -> LinearForm:
        if len(xs) != self.number_of_subspaces:
            raise ValueError("Input list has incorrect number of vectors.")
        return self.canonical_dual_isomorphism(
            [space.to_dual(x) for space, x in zip(self._spaces, xs)]
        )

    def __from_dual(self, xp: LinearForm) -> List[Any]:
        xps = self.canonical_dual_inverse_isomorphism(xp)
        return [space.from_dual(xip) for space, xip in zip(self._spaces, xps)]

    def __add(self, xs: List[Any], ys: List[Any]) -> List[Any]:
        return [space.add(x, y) for space, x, y in zip(self._spaces, xs, ys)]

    def __subtract(self, xs: List[Any], ys: List[Any]) -> List[Any]:
        return [space.subtract(x, y) for space, x, y in zip(self._spaces, xs, ys)]

    def __multiply(self, a: float, xs: List[Any]) -> List[Any]:
        return [space.multiply(a, x) for space, x in zip(self._spaces, xs)]

    def __ax(self, a: float, xs: List[Any]) -> None:
        for space, x in zip(self._spaces, xs):
            space.ax(a, x)

    def __axpy(self, a: float, xs: List[Any], ys: List[Any]) -> None:
        for space, x, y in zip(self._spaces, xs, ys):
            space.axpy(a, x, y)

    def __copy(self, xs: List[Any]) -> List[Any]:
        return [space.copy(x) for space, x in zip(self._spaces, xs)]

    def _subspace_projection_mapping(self, i: int, xs: List[Any]) -> Any:
        return xs[i]

    def _subspace_inclusion_mapping(self, i: int, x: Any) -> List[Any]:
        return [x if j == i else space.zero for j, space in enumerate(self._spaces)]


class BlockStructure(ABC):
    """
    An abstract base class for operators with a block structure.
    """

    # ... (class content is the same) ...
    def __init__(self, row_dim: int, col_dim: int) -> None:
        self._row_dim: int = row_dim
        self._col_dim: int = col_dim

    @property
    def row_dim(self) -> int:
        return self._row_dim

    @property
    def col_dim(self) -> int:
        return self._col_dim

    @abstractmethod
    def block(self, i: int, j: int) -> "LinearOperator":
        pass

    def _check_block_indices(self, i: int, j: int) -> None:
        if not (0 <= i < self.row_dim):
            raise ValueError("Row index out of range.")
        if not (0 <= j < self.col_dim):
            raise ValueError("Column index out of range.")


class BlockLinearOperator(LinearOperator, BlockStructure):
    """
    A linear operator between direct sums of Hilbert spaces, defined by a
    matrix of sub-operators (blocks).
    """

    def __init__(self, blocks: List[List[LinearOperator]]) -> None:
        """
        Initializes the block operator from a 2D list of operators.

        Args:
            blocks (List[List[LinearOperator]]): A list of lists of
                LinearOperators, ordered row-major. The domains and codomains
                of the blocks must be consistent.
        """
        if not blocks or not blocks[0]:
            raise ValueError("Block structure cannot be empty.")
        # ... (rest of the method is the same) ...
        domains = [operator.domain for operator in blocks[0]]
        codomains = []
        for row in blocks:
            if len(row) != len(domains):
                raise ValueError(
                    "All rows in the block structure must have the same length."
                )
            if not all(op.domain == dom for op, dom in zip(row, domains)):
                raise ValueError(
                    "Operators in the same column must share the same domain."
                )
            codomain = row[0].codomain
            if not all(op.codomain == codomain for op in row):
                raise ValueError(
                    "Operators in the same row must share the same codomain."
                )
            codomains.append(codomain)
        domain = HilbertSpaceDirectSum(domains)
        codomain = HilbertSpaceDirectSum(codomains)
        self._domains: List["HilbertSpace"] = domains
        self._codomains: List["HilbertSpace"] = codomains
        self._blocks: List[List[LinearOperator]] = blocks
        row_dim = len(blocks)
        col_dim = len(blocks[0])
        super().__init__(
            domain, codomain, self.__mapping, adjoint_mapping=self.__adjoint_mapping
        )
        BlockStructure.__init__(self, row_dim, col_dim)

    def block(self, i: int, j: int) -> LinearOperator:
        """Returns the operator in the (i, j)-th sub-block."""
        self._check_block_indices(i, j)
        return self._blocks[i][j]

    def __mapping(self, xs: List[Any]) -> List[Any]:
        # ... (method content is the same) ...
        ys = []
        for i in range(self.row_dim):
            codomain = self._codomains[i]
            y = codomain.zero
            for j in range(self.col_dim):
                a = self.block(i, j)
                codomain.axpy(1.0, a(xs[j]), y)
            ys.append(y)
        return ys

    def __adjoint_mapping(self, ys: List[Any]) -> List[Any]:
        # ... (method content is the same) ...
        xs = []
        for j in range(self.col_dim):
            domain = self._domains[j]
            x = domain.zero
            for i in range(self.row_dim):
                a = self.block(i, j)
                domain.axpy(1.0, a.adjoint(ys[i]), x)
            xs.append(x)
        return xs


class ColumnLinearOperator(BlockLinearOperator):
    """
    An operator that maps a single Hilbert space to a direct sum of spaces,
    structured as a column of operator blocks.
    """

    def __init__(self, operators: List[LinearOperator]) -> None:
        """
        Args:
            operators (List[LinearOperator]): A list of operators, all sharing
                a common domain, that form the rows of the column operator.
        """
        if not operators:
            raise ValueError("Operator list cannot be empty.")
        self._column_operators = operators
        domain = operators[0].domain
        if not all(op.domain == domain for op in operators):
            raise ValueError("All operators must share a common domain.")
        blocks = [[operator] for operator in operators]
        super().__init__(blocks)


class RowLinearOperator(BlockLinearOperator):
    """
    An operator that maps a direct sum of spaces to a single Hilbert space,
    structured as a row of operator blocks.
    """

    def __init__(self, operators: List[LinearOperator]) -> None:
        """
        Args:
            operators (List[LinearOperator]): A list of operators, all sharing
                a common codomain, that form the columns of the row operator.
        """
        if not operators:
            raise ValueError("Operator list cannot be empty.")
        self._row_operators = operators
        codomain = operators[0].codomain
        if not all(op.codomain == codomain for op in operators):
            raise ValueError("All operators must share a common codomain.")
        blocks = [operators]
        super().__init__(blocks)


class BlockDiagonalLinearOperator(LinearOperator, BlockStructure):
    """
    A block operator where all off-diagonal blocks are zero operators.
    """

    def __init__(self, operators: List[LinearOperator]) -> None:
        """
        Args:
            operators (List[LinearOperator]): A list of operators that will
                form the diagonal blocks.
        """
        self._operators: List[LinearOperator] = operators
        domain = HilbertSpaceDirectSum([operator.domain for operator in operators])
        codomain = HilbertSpaceDirectSum([operator.codomain for operator in operators])

        def mapping(xs: List[Any]) -> List[Any]:
            return [operator(x) for operator, x in zip(operators, xs)]

        def adjoint_mapping(ys: List[Any]) -> List[Any]:
            return [operator.adjoint(y) for operator, y in zip(operators, ys)]

        super().__init__(domain, codomain, mapping, adjoint_mapping=adjoint_mapping)
        dim = len(self._operators)
        BlockStructure.__init__(self, dim, dim)

    def block(self, i: int, j: int) -> LinearOperator:
        """
        Returns the operator in the (i, j)-th sub-block.

        If i equals j, this is one of the diagonal operators. Otherwise, it
        is a zero operator.
        """
        self._check_block_indices(i, j)
        if i == j:
            return self._operators[i]
        else:
            domain = self._operators[j].domain
            codomain = self._operators[i].codomain
            return domain.zero_operator(codomain)
