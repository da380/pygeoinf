"""
Module for direct sums of Hilbert spaces and related constructions. 
"""

from abc import ABC, abstractmethod
import numpy as np
from pygeoinf.hilbert_space import HilbertSpace, LinearOperator, LinearForm


class HilbertSpaceDirectSum(HilbertSpace):
    """
    Class for Hilbert spaces formed from a direct sum of a list of Hilbert spaces.

    Instances are formed by providing a list of HilbertSpaces. Along with the usual
    HilbertSpace methods, this class implements projection and inclusion operators
    onto the subspaces. And also the canonical injection from the direct sum of
    the dual spaces into the dual of the direct sum.
    """

    def __init__(self, spaces):
        """
        Args:
            spaces ([HilbertSpaces]): A list of Hilbert spaces whos direct sum is to be formed.
        """
        self._spaces = spaces
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
            axpy=self.__axpy,
            copy=self.__copy,
        )

    #######################################################
    #                    Public methods                   #
    #######################################################

    @property
    def subspaces(self):
        """
        Return a list of the subspaces.
        """
        return self._spaces

    @property
    def number_of_subspaces(self):
        """
        Returns the number of subspaces in the direct sum.
        """
        return len(self.subspaces)

    def subspace(self, i):
        """
        Returns the ith subspace.
        """
        return self.subspaces[i]

    def subspace_projection(self, i):
        """
        Returns the projection operator onto the ith subspace.
        """
        return LinearOperator(
            self,
            self.subspaces[i],
            lambda xs: self._subspace_projection_mapping(i, xs),
            adjoint_mapping=lambda x: self._subspace_inclusion_mapping(i, x),
        )

    def subspace_inclusion(self, i):
        """
        Returns the inclusion operator from the ith space.
        """
        return LinearOperator(
            self.subspaces[i],
            self,
            lambda x: self._subspace_inclusion_mapping(i, x),
            adjoint_mapping=lambda xs: self._subspace_projection_mapping(i, xs),
        )

    def canonical_dual_isomorphism(self, xps):
        """
        Maps a direct sum of dual-subspace vectors to the dual vector.
        """
        if len(xps) != self.number_of_subspaces:
            raise ValueError("Not a valid vector")
        return LinearForm(
            self,
            mapping=lambda x: sum(
                xp(self.subspace_projection(i)(x)) for i, xp in enumerate(xps)
            ),
        )

    def canonical_dual_inverse_isomorphism(self, xp):
        """
        Maps a dual vector to the direct sum of the dual-subspace vectors.
        """
        return [
            LinearForm(space, mapping=lambda x, j=i: xp(self.subspace_inclusion(j)(x)))
            for i, space in enumerate(self.subspaces)
        ]

    #######################################################
    #                   Private methods                   #
    #######################################################

    def __to_components(self, xs):
        # Local implementation of to component mapping.
        cs = [space.to_components(x) for space, x in zip(self._spaces, xs)]
        return np.concatenate(cs, 0)

    def __from_components(self, c):
        # Local implementation of from component mapping.
        xs = []
        i = 0
        for space in self._spaces:
            j = i + space.dim
            x = space.from_components(c[i:j])
            xs.append(x)
            i = j
        return xs

    def __inner_product(self, x1s, x2s):
        # Local implementation of inner product.
        return sum(
            space.inner_product(x1, x2) for space, x1, x2 in zip(self._spaces, x1s, x2s)
        )

    def __to_dual(self, xs):
        # Local implementation of the mapping to the dual space.
        if len(xs) != self.number_of_subspaces:
            raise ValueError("Not a valid vector")
        return self.canonical_dual_isomorphism(
            [space.to_dual(x) for space, x in zip(self._spaces, xs)]
        )

    def __from_dual(self, xp):
        # Local implementation of the mapping from the dual space.
        xps = self.canonical_dual_inverse_isomorphism(xp)
        return [space.from_dual(xip) for space, xip in zip(self._spaces, xps)]

    def __add(self, xs, ys):
        # Local implementation of add.
        return [space.add(x, y) for space, x, y in zip(self._spaces, xs, ys)]

    def __subtract(self, xs, ys):
        # Local implementation of subtract.
        return [space.subtract(x, y) for space, x, y in zip(self._spaces, xs, ys)]

    def __multiply(self, a, xs):
        # Local implementation of multiply.
        return [space.multiply(a, x) for space, x in zip(self._spaces, xs)]

    def __axpy(self, a, xs, ys):
        # Local implementation of axpy.
        return [space.axpy(a, x, y) for space, x, y in zip(self._spaces, xs, ys)]

    def __copy(self, xs):
        return [space.copy(x) for space, x in zip(self._spaces, xs)]

    def _subspace_projection_mapping(self, i, xs):
        # Implementation of the projection mapping onto ith space.
        return xs[i]

    def _subspace_inclusion_mapping(self, i, x):
        # Implementation of the inclusion mapping from ith space.
        return [x if j == i else space.zero for j, space in enumerate(self._spaces)]


class BlockStructure(ABC):
    """
    Base class for block operators.
    """

    def __init__(self, row_dim, col_dim):
        """
        Args:

            row_dim (int): Number of rows in block structure.
            col_dim (int): Number of columns in row structure.
        """
        self._row_dim = row_dim
        self._col_dim = col_dim

    @property
    def row_dim(self):
        """
        Returns the number of rows in block operator.
        """
        return self._row_dim

    @property
    def col_dim(self):
        """
        Returns the number of columns in block operator.
        """
        return self._col_dim

    @abstractmethod
    def block(self, i, j):
        """
        Return the ith block
        """

    def _check_block_indices(self, i, j):
        # Check block indices are in range.
        if not (0 <= i < self.row_dim):
            raise ValueError("Row index out of range.")
        if not (0 <= j < self.col_dim):
            raise ValueError("Column index out of range.")


class BlockLinearOperator(LinearOperator, BlockStructure):
    """
    Class for linear operators acting between direct sums of Hilbert spaces that are defined
    in a blockwise manner.

    Instances are formed from lists of list of LinearOperators which give the sub-blocks in
    a row major ordering.
    """

    def __init__(self, blocks):

        # Check and form the list of domains and codomains.
        domains = [operator.domain for operator in blocks[0]]
        codomains = []
        for row in blocks:
            assert domains == [operator.domain for operator in row]
            codomain = row[0].codomain
            assert all(operator.codomain == codomain for operator in row)
            codomains.append(codomain)

        domain = HilbertSpaceDirectSum(domains)
        codomain = HilbertSpaceDirectSum(codomains)

        self._domains = domains
        self._codomains = codomains
        self._blocks = blocks
        row_dim = len(blocks)
        col_dim = len(blocks[0])

        super().__init__(
            domain,
            codomain,
            self.__mapping,
            adjoint_mapping=self.__adjoint_mapping,
        )
        BlockStructure.__init__(self, row_dim, col_dim)

    def block(self, i, j):
        """
        Returns the operator in the (i,j)th sub-block.
        """
        assert 0 <= i < self.row_dim
        assert 0 <= j < self.col_dim
        return self._blocks[i][j]

    def __mapping(self, xs):
        ys = []
        for i in range(self.row_dim):
            codomain = self._codomains[i]
            y = codomain.zero
            for j in range(self.col_dim):
                a = self.block(i, j)
                y = codomain.axpy(1, a(xs[j]), y)
            ys.append(y)
        return ys

    def __adjoint_mapping(self, ys):
        xs = []
        for j in range(self.col_dim):
            domain = self._domains[j]
            x = domain.zero
            for i in range(self.row_dim):
                a = self.block(i, j)
                x = domain.axpy(1, a.adjoint(ys[i]), x)
            xs.append(x)
        return xs


class ColumnLinearOperator(LinearOperator, BlockStructure):
    """
    Class for linear operators acting between a space and a direct sum of spaces.
    """

    def __init__(self, operators):
        """
        Args:
            operators ([LinearOperator]): List of operators all with a common domain.
        """

        self._operators = operators

        domains = [operator.domain for operator in operators]
        codomains = [operator.codomain for operator in operators]

        domain = domains[0]
        assert all(operator.domain == domain for operator in operators)

        codomain = HilbertSpaceDirectSum(codomains)

        blocks = [[operator] for operator in operators]

        block_operator = BlockLinearOperator(blocks)

        def mapping(x):
            return block_operator([x])

        def adjoint_mapping(ys):
            return block_operator.adjoint(ys)[0]

        super().__init__(domain, codomain, mapping, adjoint_mapping=adjoint_mapping)

        row_dim = len(self._operators)
        BlockStructure.__init__(self, row_dim, 1)

    def block(self, i, j):
        """
        Returns the operator in the (i,j)th sub-block.
        """
        assert 0 <= i < self.row_dim
        assert j == 0

        return self._operators[i]


class RowLinearOperator(LinearOperator, BlockStructure):
    """
    Class for linear operators acting between a direct sum of spaces and a space.
    """

    def __init__(self, operators):
        """
        Args:
            operators ([LinearOperator]): List of operators all with a common codomain.
        """

        self._operators = operators

        domains = [operator.domain for operator in operators]
        codomains = [operator.codomain for operator in operators]

        codomain = codomains[0]
        assert all(operator.codomain == codomain for operator in operators)

        domain = HilbertSpaceDirectSum(domains)

        blocks = [operators]

        block_operator = BlockLinearOperator(blocks)

        def mapping(xs):
            return block_operator(xs)[0]

        def adjoint_mapping(y):
            return block_operator.adjoint([y])

        super().__init__(domain, codomain, mapping, adjoint_mapping=adjoint_mapping)

        col_dim = len(self._operators)
        BlockStructure.__init__(self, 1, col_dim)

    def block(self, i, j):
        """
        Returns the operator in the (i,j)th sub-block.
        """
        assert i == 0
        assert 0 <= j < self.col_dim

        return self._operators[j]


class BlockDiagonalLinearOperator(LinearOperator, BlockStructure):
    """
    Class for linear operators acting between direct sums of Hilbert spaces that are defined
    in a blockwise diagonal manner.

    Instances are formed from list of LinearOperators that define the diagonal blocks.
    """

    def __init__(self, operators):
        """
        Args:
            operators ([LinearOperator]): List of operators for the diagonal blocks.
        """
        self._operators = operators

        domain = HilbertSpaceDirectSum([operator.domain for operator in operators])
        codomain = HilbertSpaceDirectSum([operator.codomain for operator in operators])

        def mapping(xs):
            return [operator(x) for operator, x in zip(operators, xs)]

        def adjoint_mapping(ys):
            return [operator.adjoint(y) for operator, y in zip(operators, ys)]

        super().__init__(domain, codomain, mapping, adjoint_mapping=adjoint_mapping)

        row_dim = len(self._operators)
        col_dim = row_dim
        BlockStructure.__init__(self, row_dim, col_dim)

    def block(self, i, j):
        """
        Returns the operator in the (i,j)th sub-block.
        """
        assert 0 <= i < self.row_dim
        assert 0 <= j < self.col_dim

        if i == j:
            return self._operators[i]
        else:
            domain = self._operators[j].domain
            codomain = self._operators[i].codomain
            return domain.zero_operator(codomain)
