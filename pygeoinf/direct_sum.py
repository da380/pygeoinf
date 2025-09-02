"""
Implements direct sums of Hilbert spaces and corresponding block operators.

This module provides tools for constructing larger, composite Hilbert spaces and
operators from smaller ones. This is essential for problems involving multiple
coupled fields or joint inversions where a single model is constrained by
data from different experiments.

Key Classes
-----------
- `HilbertSpaceDirectSum`: A `HilbertSpace` formed by the direct sum of a
  list of other spaces. Vectors in this space are lists of vectors from the
  component subspaces.
- `BlockLinearOperator`: A `LinearOperator` acting between direct sum spaces,
  represented as a 2D grid (matrix) of sub-operators.
- `ColumnLinearOperator`: A specialized block operator mapping from a single
  space to a direct sum space.
- `RowLinearOperator`: A specialized block operator mapping from a direct sum
  space to a single space.
- `BlockDiagonalLinearOperator`: An efficient representation for block
  operators with zero off-diagonal blocks.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np
from scipy.linalg import block_diag
from joblib import Parallel, delayed

from .hilbert_space import HilbertSpace
from .operators import LinearOperator
from .linear_forms import LinearForm
from .parallel import parallel_compute_dense_matrix_from_scipy_op


class HilbertSpaceDirectSum(HilbertSpace):
    """
    A Hilbert space formed from the direct sum of a list of other spaces.

    A vector in this space is represented as a list of vectors, where the i-th
    element of the list is a vector from the i-th component subspace. The
    inner product is the sum of the inner products of the components.
    """

    def __init__(self, spaces: List[HilbertSpace]) -> None:
        """
        Initializes the direct sum space.

        Args:
            spaces (List[HilbertSpace]): A list of Hilbert spaces to combine
                in the direct sum.
        """
        self._spaces: List[HilbertSpace] = spaces
        self._dim = sum([space.dim for space in spaces])

    @property
    def dim(self) -> int:
        """Returns the dimension of the direct sum space."""
        return self._dim

    def to_components(self, xs: List[Any]) -> np.ndarray:
        cs = [space.to_components(x) for space, x in zip(self._spaces, xs)]
        return np.concatenate(cs, 0)

    def from_components(self, c: np.ndarray) -> List[Any]:
        xs = []
        i = 0
        for space in self._spaces:
            j = i + space.dim
            x = space.from_components(c[i:j])
            xs.append(x)
            i = j
        return xs

    def to_dual(self, xs: List[Any]) -> LinearForm:
        if len(xs) != self.number_of_subspaces:
            raise ValueError("Input list has incorrect number of vectors.")
        return self.canonical_dual_isomorphism(
            [space.to_dual(x) for space, x in zip(self._spaces, xs)]
        )

    def from_dual(self, xp: LinearForm) -> List[Any]:
        xps = self.canonical_dual_inverse_isomorphism(xp)
        return [space.from_dual(xip) for space, xip in zip(self._spaces, xps)]

    def add(self, xs: List[Any], ys: List[Any]) -> List[Any]:
        return [space.add(x, y) for space, x, y in zip(self._spaces, xs, ys)]

    def subtract(self, xs: List[Any], ys: List[Any]) -> List[Any]:
        return [space.subtract(x, y) for space, x, y in zip(self._spaces, xs, ys)]

    def multiply(self, a: float, xs: List[Any]) -> List[Any]:
        return [space.multiply(a, x) for space, x in zip(self._spaces, xs)]

    def ax(self, a: float, xs: List[Any]) -> None:
        for space, x in zip(self._spaces, xs):
            space.ax(a, x)

    def axpy(self, a: float, xs: List[Any], ys: List[Any]) -> None:
        for space, x, y in zip(self._spaces, xs, ys):
            space.axpy(a, x, y)

    def copy(self, xs: List[Any]) -> List[Any]:
        return [space.copy(x) for space, x in zip(self._spaces, xs)]

    def __eq__(self, other: object) -> bool:
        """
        Checks for mathematical equality with another direct sum space.

        Two direct sum spaces are equal if they are composed of the same
        number of subspaces, and each corresponding subspace is equal.
        """
        if not isinstance(other, HilbertSpaceDirectSum):
            return NotImplemented

        return self.subspaces == other.subspaces

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

        cps = [space.dual.to_components(xp) for space, xp in zip(self._spaces, xps)]
        cp = np.concatenate(cps, 0)
        return LinearForm(self, components=cp)

    def canonical_dual_inverse_isomorphism(self, xp: LinearForm) -> List[LinearForm]:
        """
        Maps a dual vector on the sum space to a list of dual vectors.

        This is the inverse of the canonical isomorphism, projecting the action
        of a dual vector onto each subspace.

        Args:
            xp (LinearForm): A dual vector on the direct sum space.
        """

        cp = self.dual.to_components(xp)
        xps = []
        i = 0
        for space in self._spaces:
            j = i + space.dim
            xp = space.dual.from_components(cp[i:j])
            xps.append(xp)
            i = j
        return xps

    def _subspace_projection_mapping(self, i: int, xs: List[Any]) -> Any:
        return xs[i]

    def _subspace_inclusion_mapping(self, i: int, x: Any) -> List[Any]:
        return [x if j == i else space.zero for j, space in enumerate(self._spaces)]


class BlockStructure(ABC):
    """
    An abstract base class for operators with a block structure.
    """

    def __init__(self, row_dim: int, col_dim: int) -> None:
        self._row_dim: int = row_dim
        self._col_dim: int = col_dim

    @property
    def row_dim(self) -> int:
        """
        Returns the number of rows in the block structure.
        """
        return self._row_dim

    @property
    def col_dim(self) -> int:
        """
        Returns the number of columns in the block structure.
        """
        return self._col_dim

    @abstractmethod
    def block(self, i: int, j: int) -> "LinearOperator":
        """
        Returns the operator in the (i, j)-th sub-block.
        """
        pass

    def _check_block_indices(self, i: int, j: int) -> None:
        if not (0 <= i < self.row_dim):
            raise ValueError("Row index out of range.")
        if not (0 <= j < self.col_dim):
            raise ValueError("Column index out of range.")


class BlockLinearOperator(LinearOperator, BlockStructure):
    """
    A linear operator between direct sum spaces, defined by a matrix of sub-operators.

    This operator acts like a matrix where each entry is itself a `LinearOperator`.
    It maps a list of input vectors `[x_1, x_2, ...]` to a list of output
    vectors `[y_1, y_2, ...]`. The constructor checks for dimensional
    consistency between the blocks.
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

    def _compute_dense_matrix(
        self, galerkin: bool, parallel: bool, n_jobs: int
    ) -> np.ndarray:
        """Overloaded method to efficiently compute the dense matrix for a block operator."""
        if not parallel:
            block_matrices = [
                [
                    self.block(i, j).matrix(
                        dense=True, galerkin=galerkin, parallel=False
                    )
                    for j in range(self.col_dim)
                ]
                for i in range(self.row_dim)
            ]
            return np.block(block_matrices)

        else:
            block_scipy_ops = [
                self.block(i, j).matrix(galerkin=galerkin)
                for i in range(self.row_dim)
                for j in range(self.col_dim)
            ]

            computed_blocks = Parallel(n_jobs=n_jobs)(
                delayed(parallel_compute_dense_matrix_from_scipy_op)(op, n_jobs=1)
                for op in block_scipy_ops
            )

            block_matrices = [
                computed_blocks[i * self.col_dim : (i + 1) * self.col_dim]
                for i in range(self.row_dim)
            ]
            return np.block(block_matrices)

    def __mapping(self, xs: List[Any]) -> List[Any]:

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

        xs = []
        for j in range(self.col_dim):
            domain = self._domains[j]
            x = domain.zero
            for i in range(self.row_dim):
                a = self.block(i, j)
                domain.axpy(1.0, a.adjoint(ys[i]), x)
            xs.append(x)
        return xs


class ColumnLinearOperator(LinearOperator, BlockStructure):
    """
    An operator that maps from a single space to a direct sum space.

    It can be visualized as a column vector of operators, `[A_1, A_2, ...]^T`.
    It takes a single input vector `x` and produces a list of output vectors
    `[A_1(x), A_2(x), ...]`. This is often used to represent a joint forward
    operator in an inverse problem.
    """

    def __init__(self, operators: List[LinearOperator]) -> None:
        """
        Args:
            operators (List[LinearOperator]): A list of operators, all sharing
                a common domain, that form the rows of the column operator.
        """
        if not operators:
            raise ValueError("Operator list cannot be empty.")

        domain = operators[0].domain
        if not all(op.domain == domain for op in operators):
            raise ValueError("All operators must share a common domain.")

        codomains = [op.codomain for op in operators]
        codomain = HilbertSpaceDirectSum(codomains)

        self._operators = operators

        def mapping(x: Any) -> List[Any]:
            """Applies each operator to x and returns a list of results."""
            return [op(x) for op in self._operators]

        def adjoint_mapping(ys: List[Any]) -> Any:
            """
            Applies the adjoint of each operator to the corresponding y_i
            and sums the results.
            """
            x = domain.zero
            for op, y in zip(self._operators, ys):
                domain.axpy(1.0, op.adjoint(y), x)
            return x

        LinearOperator.__init__(
            self, domain, codomain, mapping, adjoint_mapping=adjoint_mapping
        )
        BlockStructure.__init__(self, len(operators), 1)

    def block(self, i: int, j: int) -> LinearOperator:
        """Returns the operator in the (i, 0)-th sub-block."""
        self._check_block_indices(i, j)
        if j != 0:
            raise IndexError("Column index out of range for ColumnLinearOperator.")
        return self._operators[i]

    def _compute_dense_matrix(
        self, galerkin: bool, parallel: bool, n_jobs: int
    ) -> np.ndarray:
        """Overloaded method to efficiently compute the dense matrix for a column operator."""
        if not parallel:
            block_matrices = [
                op.matrix(dense=True, galerkin=galerkin, parallel=False)
                for op in self._operators
            ]
            return np.vstack(block_matrices)
        else:
            block_scipy_ops = [
                op.matrix(galerkin=galerkin, parallel=False) for op in self._operators
            ]

            computed_blocks = Parallel(n_jobs=n_jobs)(
                delayed(parallel_compute_dense_matrix_from_scipy_op)(op, n_jobs=1)
                for op in block_scipy_ops
            )

            return np.vstack(computed_blocks)


class RowLinearOperator(LinearOperator, BlockStructure):
    """
    An operator that maps from a direct sum space to a single space.

    It can be visualized as a row vector of operators, `[A_1, A_2, ...]`.
    It takes a list of input vectors `[x_1, x_2, ...]` and produces a single
    output vector `y = A_1(x_1) + A_2(x_2) + ...`. The adjoint of a
    `ColumnLinearOperator` is a `RowLinearOperator`.
    """

    def __init__(self, operators: List[LinearOperator]) -> None:
        """
        Args:
            operators (List[LinearOperator]): A list of operators, all sharing
                a common codomain, that form the columns of the row operator.
        """
        if not operators:
            raise ValueError("Operator list cannot be empty.")

        codomain = operators[0].codomain
        if not all(op.codomain == codomain for op in operators):
            raise ValueError("All operators must share a common codomain.")

        domains = [op.domain for op in operators]
        domain = HilbertSpaceDirectSum(domains)

        self._operators = operators

        def mapping(xs: List[Any]) -> Any:
            """
            Applies each operator to the corresponding x_i and sums the results.
            """
            y = codomain.zero
            for op, x in zip(self._operators, xs):
                codomain.axpy(1.0, op(x), y)
            return y

        def adjoint_mapping(y: Any) -> List[Any]:
            """Applies the adjoint of each operator to y and returns a list."""
            return [op.adjoint(y) for op in self._operators]

        LinearOperator.__init__(
            self, domain, codomain, mapping, adjoint_mapping=adjoint_mapping
        )
        BlockStructure.__init__(self, 1, len(operators))

    def block(self, i: int, j: int) -> LinearOperator:
        """Returns the operator in the (0, j)-th sub-block."""
        self._check_block_indices(i, j)
        if i != 0:
            raise IndexError("Row index out of range for RowLinearOperator.")
        return self._operators[j]

    def _compute_dense_matrix(
        self, galerkin: bool, parallel: bool, n_jobs: int
    ) -> np.ndarray:
        """Overloaded method to efficiently compute the dense matrix for a row operator."""
        if not parallel:
            block_matrices = [
                op.matrix(dense=True, galerkin=galerkin, parallel=False)
                for op in self._operators
            ]
            return np.hstack(block_matrices)
        else:
            block_scipy_ops = [
                op.matrix(galerkin=galerkin, parallel=False) for op in self._operators
            ]

            computed_blocks = Parallel(n_jobs=n_jobs)(
                delayed(parallel_compute_dense_matrix_from_scipy_op)(op, n_jobs=1)
                for op in block_scipy_ops
            )

            return np.hstack(computed_blocks)


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

    def _compute_dense_matrix(
        self, galerkin: bool, parallel: bool, n_jobs: int
    ) -> np.ndarray:
        """Overloaded method to efficiently compute the dense matrix for a block-diagonal operator."""
        if not parallel:
            block_matrices = [
                op.matrix(dense=True, galerkin=galerkin, parallel=False)
                for op in self._operators
            ]
            return block_diag(*block_matrices)
        else:
            block_scipy_ops = [
                op.matrix(galerkin=galerkin, parallel=False) for op in self._operators
            ]

            computed_blocks = Parallel(n_jobs=n_jobs)(
                delayed(parallel_compute_dense_matrix_from_scipy_op)(op, n_jobs=1)
                for op in block_scipy_ops
            )

            return block_diag(*computed_blocks)
