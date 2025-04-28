"""
Module for linear algebra on Hilbert spaces. 
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import (
    cho_factor,
    cho_solve,
    lu_factor,
    lu_solve,
    solve_triangular,
    eigh,
    svd,
    qr,
)
from scipy.stats import norm
from scipy.sparse.linalg import LinearOperator as ScipyLinOp
from scipy.sparse.linalg import gmres, bicgstab, cg, bicg
from scipy.sparse import diags


class HilbertSpace:
    """
    A class for real Hilbert spaces. To define an instance, the
    user needs to provide the following:

        (1) The dimension of the space, or the dimension of the
            finite-dimensional approximating space.
        (2) A mapping from elements of the space to their components.
            These components must be expressed as numpy arrays with
            shape (dim) with dim the spaces dimension.
        (3) A mapping from components back to the vectors. This
            needs to be the inverse of the mapping in (2), but
            this requirement is not automatically checked.
        (4) The inner product on the space.
        (5) The mapping from the space to its dual.
        (5) The mapping from a dual vector to its representation
            within the space.

    Note that this class does *not* define elements of the
    vector space. These must be pre-defined separately. It
    is also assumed that the usual vector operations are
    available for this latter space.
    """

    def __init__(
        self,
        dim,
        to_components,
        from_components,
        inner_product,
        to_dual,
        from_dual,
        /,
        *,
        add=None,
        subtract=None,
        multiply=None,
        axpy=None,
        copy=None,
        base=None,
    ):
        """
        Args:
            dim (int): The dimension of the space, or of the
                finite-dimensional approximating space.
            to_components (callable):  A functor that maps vectors
                to their components.
            from_components (callable): A functor that maps components
                to vectors.
            inner_product (callable): A functor the implements the inner
                product on the space.
            to_dual (callable | None): A funcator that maps a vector
                to the cannonically associated dual vector.
            from_dual (callable | None): A functor that maps a dual vector
                to its representation on the space.
            add (callable): Implements vector addition.
            subtract (callable): Implements vector subtraction.
            multiply (callable): Implements scalar multiplication.
            axpy (callable): Implements the mapping y -> a*x + y
            copy (callable): Implements deep copy of a vector.
            base (HilbertSpace | None): Set to none for an original space,
                and to the base space when forming the dual.
        """
        self._dim = dim
        self.__to_components = to_components
        self.__from_components = from_components
        self.__inner_product = inner_product
        self.__from_dual = from_dual
        self.__to_dual = to_dual
        self._base = base
        self._add = self.__add if add is None else add
        self._subtract = self.__subtract if subtract is None else subtract
        self._multiply = self.__multiply if multiply is None else multiply
        self._axpy = self.__axpy if axpy is None else axpy
        self._copy = self.__copy if copy is None else copy

    @property
    def dim(self):
        """The dimension of the space."""
        return self._dim

    @property
    def dual(self):
        """The dual of the Hilbert space."""
        if self._base is None:
            return HilbertSpace(
                self.dim,
                self._dual_to_components,
                self._dual_from_components,
                self._dual_inner_product,
                self.from_dual,
                self.to_dual,
                base=self,
            )
        else:
            return self._base

    @property
    def zero(self):
        return self.from_components(np.zeros((self.dim)))

    @property
    def coordinate_inclusion(self):
        """
        Returns the linear operator that maps coordinate vectors
        to elements of the sapce.
        """
        domain = EuclideanSpace(self.dim)

        def dual_mapping(xp):
            cp = self.dual.to_components(xp)
            return domain.to_dual(cp)

        def adjoint_mapping(y):
            yp = self.to_dual(y)
            return self.dual.to_components(yp)

        return LinearOperator(
            domain,
            self,
            self.from_components,
            dual_mapping=dual_mapping,
            adjoint_mapping=adjoint_mapping,
        )

    @property
    def coordinate_projection(self):
        """
        Returns the linear operator that maps vectors to their coordinates.
        """
        codomain = EuclideanSpace(self.dim)

        def dual_mapping(cp):
            c = codomain.from_dual(cp)
            return self.dual.from_components(c)

        def adjoint_mapping(c):
            xp = self.dual.from_components(c)
            return self.from_dual(xp)

        return LinearOperator(
            self,
            codomain,
            self.to_components,
            dual_mapping=dual_mapping,
            adjoint_mapping=adjoint_mapping,
        )

    @property
    def riesz(self):
        """
        Returns as a LinearOpeator the isomorphism from the
        dual space to the space via the Riesz representation theorem.
        """
        return LinearOperator.self_dual(self.dual, self.from_dual)

    @property
    def inverse_riesz(self):
        """
        Returns as a LinearOpeator the isomorphism from the
        space to the dual space via the Riesz representation theorem.
        """
        return LinearOperator.self_dual(self, self.to_dual)

    def inner_product(self, x1, x2):
        """Return the inner product of two vectors."""
        return self.__inner_product(x1, x2)

    def squared_norm(self, x):
        """Return the squared norm of a vector."""
        return self.inner_product(x, x)

    def norm(self, x):
        """Return the norm of a vector."""
        return np.sqrt(self.squared_norm(x))

    def to_dual(self, x):
        """Map a vector to cannonically associated dual vector."""
        return self.__to_dual(x)

    def from_dual(self, xp):
        """Map a dual vector to its representation in the space."""
        return self.__from_dual(xp)

    def _dual_inner_product(self, xp1, xp2):
        return self.inner_product(self.from_dual(xp1), self.from_dual(xp2))

    def is_element(self, x):
        """
        Returns True if the argument is a vector in the space.
        """
        return isinstance(x, type(self.zero))

    def add(self, x, y):
        """Returns x + y."""
        return self._add(x, y)

    def subtract(self, x, y):
        """Returns x - y."""
        return self._subtract(x, y)

    def multiply(self, a, x):
        """Returns a * x."""
        return self._multiply(a, x)

    def negative(self, x):
        """Returns -x."""
        return self.multiply(-1, x)

    def axpy(self, a, x, y):
        return self._axpy(a, x, y)

    def copy(self, x):
        return self._copy(x)

    def to_components(self, x):
        """Maps vectors to components."""
        return self.__to_components(x)

    def from_components(self, c):
        """Maps components to vectors."""
        return self.__from_components(c)

    def basis_vector(self, i):
        """Return the ith basis vector."""
        c = np.zeros(self.dim)
        c[i] = 1
        return self.from_components(c)

    def random(self):
        """Returns a random vector with components drwn from a standard Gaussian distribution."""
        return self.from_components(norm().rvs(size=self.dim))

    def sample_expectation(self, vectors):
        """
        Given a list of elements in the space, forms their sample variance.
        """
        n = len(vectors)
        all([self.is_element(x) for x in vectors])
        xbar = self.zero
        for x in vectors:
            xbar = self.axpy(1 / n, x, xbar)
        return xbar

    def identity_operator(self):
        """Returns identity operator on the space."""
        return LinearOperator(
            self,
            self,
            lambda x: x,
            dual_mapping=lambda yp: yp,
            adjoint_mapping=lambda y: y,
        )

    def zero_operator(self, codomain=None):
        """Returns zero operator into another space."""
        codomain = self if codomain is None else codomain
        return LinearOperator(
            self,
            codomain,
            lambda x: codomain.zero,
            dual_mapping=lambda yp: self.dual.zero,
            adjoint_mapping=lambda y: self.zero,
        )

    def _dual_to_components(self, xp):
        return xp.components

    def _dual_from_components(self, cp):
        return LinearForm(self, components=cp)

    def __add(self, x, y):
        # Default implementation of vector addition.
        return x + y

    def __subtract(self, x, y):
        # Default implementation of vector subtraction.
        return x - y

    def __multiply(self, a, x):
        # Default implementation of scalar multiplication.
        return a * x.copy()

    def __axpy(self, a, x, y):
        # Default implementation of y -> a*x+y.
        y += a * x
        return y

    def __copy(self, x):
        return x.copy()


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
        assert len(xps) == self.number_of_subspaces
        return LinearForm(
            self,
            mapping=lambda x: sum(
                [xp(self.subspace_projection(i)(x)) for i, xp in enumerate(xps)]
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
        return sum(
            [
                space.inner_product(x1, x2)
                for space, x1, x2 in zip(self._spaces, x1s, x2s)
            ]
        )

    def __to_dual(self, xs):
        # Local implementation of the mapping to the dual space.
        assert len(xs) == self.number_of_subspaces
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


class EuclideanSpace(HilbertSpace):
    """
    Euclidean space implemented as an instance of HilbertSpace."""

    def __init__(self, dim):
        """
        Args:
            dim (int): Dimension of the space.
        """

        super().__init__(
            dim,
            lambda x: x,
            lambda x: x,
            self.__inner_product,
            self.__to_dual,
            self.__from_dual,
        )

    def standard_gaussisan_measure(self, standard_deviation):
        """
        Returns a Gaussian measure on the space with covariance proportional
        to the identity operator and with zero expectation.

        Args:
            standard_deviation (float): The standard deviation for each component.
        """
        factor = standard_deviation * self.identity_operator()
        inverse_factor = (1 / standard_deviation) * self.identity_operator()
        return GaussianMeasure.from_factored_covariance(
            factor, inverse_factor=inverse_factor
        )

    def diagonal_gaussian_measure(self, standard_deviations):
        """
        Returns a Gaussian measure on the space with a diagonal
        covariance and with zero expectation.

        Args:
            standard_deviations (vector): Vector of the standard deviations.
        """
        factor = DiagonalLinearOperator(self, self, standard_deviations)
        return GaussianMeasure.from_factored_covariance(
            factor, inverse_factor=factor.inverse
        )

    def __inner_product(self, x1, x2):
        return np.dot(x1, x2)

    def __to_dual(self, x):
        return self.dual.from_components(x)

    def __from_dual(self, xp):
        cp = self.dual.to_components(xp)
        return self.from_components(cp)

    def __eq__(self, other):
        """
        Overload of equality operator for Euclidean spaces.
        """
        return isinstance(other, EuclideanSpace) and self.dim == other.dim


class Operator:
    """
    Class for operators between two Hilbert spaces.
    """

    def __init__(self, domain, codomain, mapping):
        """
        Args:
            domain (HilbertSpace): Domain of the operator.
            codomain (HilbertSpace): Codomain of the operator.
            mapping (callable): Mapping from domain to codomain.
        """
        self._domain = domain
        self._codomain = codomain
        self.__mapping = mapping

    @property
    def domain(self):
        """Domain of the operator."""
        return self._domain

    @property
    def codomain(self):
        """Codomain of the operator."""
        return self._codomain

    @property
    def is_automorphism(self):
        """True is operator maps a space into itself."""
        return self.domain == self.codomain

    @property
    def is_square(self):
        """True is operator maps a space into itself."""
        return self.domain.dim == self.codomain.dim

    @property
    def linear(self):
        """
        True is the operator is linear.
        """
        return False

    def __call__(self, x):
        """Action of the operator on a vector."""
        return self.__mapping(x)


class LinearOperator(Operator):
    """
    Class for linear operators between two Hilbert spaces.
    """

    def __init__(
        self,
        domain,
        codomain,
        mapping,
        /,
        *,
        dual_mapping=None,
        adjoint_mapping=None,
        dual_base=None,
        adjoint_base=None,
    ):
        """
        Args:
            domain (HilbertSpace): The domain of the operator.
            codomain (HilbertSpace): The codomain of the operator.
            mapping (callable): Mapping from the domain to codomain.
            dual_mapping (callable | None): Optional implementation of
                dual operator's action.
            adjoint_mapping (callable | None): Optional implementation
                of the adjoint operator's action.
            dual_base (LinearOperator) : Used internally when defining
                dual operators. Should not be set manually.
            adjoint_base (LinearOperator): Used internally when defining
                adjoint operators. Should not be set manually.
        """
        super().__init__(domain, codomain, mapping)
        self._dual_base = dual_base
        self._adjoint_base = adjoint_base
        if dual_mapping is None:
            if adjoint_mapping is None:
                self.__dual_mapping = self._dual_mapping_default
                self.__adjoint_mapping = self._adjoint_mapping_from_dual
            else:
                self.__adjoint_mapping = adjoint_mapping
                self.__dual_mapping = self._dual_mapping_from_adjoint
        else:
            self.__dual_mapping = dual_mapping
            if adjoint_mapping is None:
                self.__adjoint_mapping = self._adjoint_mapping_from_dual
            else:
                self.__adjoint_mapping = adjoint_mapping

    @staticmethod
    def self_dual(domain, mapping):
        """Returns a self-dual operator in terms of its domain and mapping."""
        return LinearOperator(domain, domain.dual, mapping, dual_mapping=mapping)

    @staticmethod
    def self_adjoint(domain, mapping):
        """Returns a self-adjoint operator in terms of its domain and mapping."""
        return LinearOperator(domain, domain, mapping, adjoint_mapping=mapping)

    @staticmethod
    def from_linear_forms(forms):
        """
        Returns a linear operator into Euclidiean space defined by the tensor
        product of a set of forms with the standard Euclidean basis vectors.

        Args:
            forms ([LinearForms]): A list of linear forms defined on a common domain.

        Returns:
            LinearOperator: The linear operator.

        Notes: The matrix components of the forms are used to define the
               matrix representation of the operator and this is stored internally.
        """
        domain = forms[0].domain
        codomain = EuclideanSpace(len(forms))
        if not all([form.domain == domain for form in forms]):
            raise ValueError("Forms need to be defined on a common domain")

        matrix = np.zeros((codomain.dim, domain.dim))
        for i, form in enumerate(forms):
            matrix[i, :] = form.components

        def mapping(x):
            cx = domain.to_components(x)
            cy = matrix @ cx
            return cy

        def dual_mapping(yp):
            cyp = codomain.dual.to_components(yp)
            cxp = matrix.T @ cyp
            return domain.dual.from_components(cxp)

        return LinearOperator(domain, codomain, mapping, dual_mapping=dual_mapping)

    @staticmethod
    def from_matrix(domain, codomain, matrix, /, *, galerkin=False):
        """
        Returns a linear operator defined by its matrix representation.
        By default the standard representation is assumed but the
        Galerkin representation can optinally be used so long as the
        domain and codomain are Hilbert spaces.

        Args:
            domain (HilbertSpace): The domain of the operator.
            codomain (HilbertSpace): The codomain of the operator.
            matrix (matrix-like): The matrix representation of the operator.
            galerkin (bool): True if the Galkerin represention is used.

        Returns:
            LinearOperator: The linear operator.
        """
        assert matrix.shape == (codomain.dim, domain.dim)

        if galerkin:

            def mapping(x):
                cx = domain.to_components(x)
                cyp = matrix @ cx
                yp = codomain.dual.from_components(cyp)
                return codomain.from_dual(yp)

            def adjoint_mapping(y):
                cy = codomain.to_components(y)
                cxp = matrix.T @ cy
                xp = domain.dual.from_components(cxp)
                return domain.from_dual(xp)

            return LinearOperator(
                domain,
                codomain,
                mapping,
                adjoint_mapping=adjoint_mapping,
            )

        else:

            def mapping(x):
                cx = domain.to_components(x)
                cy = matrix @ cx
                return codomain.from_components(cy)

            def dual_mapping(yp):
                cyp = codomain.dual.to_components(yp)
                cxp = matrix.T @ cyp
                return domain.dual.from_components(cxp)

            return LinearOperator(domain, codomain, mapping, dual_mapping=dual_mapping)

    @staticmethod
    def self_adjoint_from_matrix(domain, matrix):
        """
        Forms a self-adjoint operator from its Galerkin G matrix representation.
        """

        def mapping(x):
            cx = domain.to_components(x)
            cyp = matrix @ cx
            yp = domain.dual.from_components(cyp)
            return domain.from_dual(yp)

        return LinearOperator.self_adjoint(domain, mapping)

    @staticmethod
    def from_tensor_product(domain, codomain, vector_pairs, /, *, weights=None):
        """
        Forms a LinearOperator between Hilbert spaces from
        the tensor product of a list of pairs of vectors.

        Args:
            domain (HilbertSpace): Domain for the linear operator.
            codomain (HilbertSpace): Codomain for the linear operator.
            vector_pairs ([[codomain vector, domain vector]]): A list of pairs of vectors
                from which the tensor product is to be constructed.
            weights [float]: Optional list of weights for the terms in the tensor
               product. If none is provided default weights of one are used.
        """

        assert all([domain.is_element(vector) for _, vector in vector_pairs])
        assert all([codomain.is_element(vector) for vector, _ in vector_pairs])

        if weights is None:
            _weights = [1 for _ in vector_pairs]
        else:
            _weights = weights

        def mapping(x):
            y = codomain.zero
            for left, right, weight in zip(vector_pairs, _weights):
                product = domain.inner_product(right, x)
                y = codomain.axpy(weight * product, left, y)
            return y

        def adjoint_mapping(y):
            x = domain.zero
            for left, right, weight in zip(vector_pairs, _weights):
                product = codomain.inner_product(left, y)
                x = domain.axpy(weight * product, right, x)
            return x

        return LinearOperator(
            domain, codomain, mapping, adjoint_mapping=adjoint_mapping
        )

    @staticmethod
    def self_adjoint_from_tensor_product(domain, vectors, /, *, weights=None):
        """
        Forms a self-adjoint LinearOperator on a Hilbert space from
        the tensor product of a list of vectors.

        Args:
            domain (HilbertSpace): Domain for the linear operator.
            vectors ([domain vector]): A list of vectors from which
                the tensor product is to be constructed.
            weights [float]: Optional list of weights for the terms in the tensor
                product. If none is provided default weights of one are used.
        """

        assert all([domain.is_element(vector) for vector in vectors])

        if weights is None:
            _weights = [1 for _ in vectors]
        else:
            _weights = weights

        def mapping(x):
            y = domain.zero
            for vector, weight in zip(vectors, _weights):
                product = domain.inner_product(vector, x)
                y = domain.axpy(weight * product, vector, y)
            return y

        return LinearOperator.self_adjoint(domain, mapping)

    @property
    def linear(self):
        # Overide of method from base class.
        return True

    @property
    def dual(self):
        """The dual of the operator."""
        if self._dual_base is None:
            return LinearOperator(
                self.codomain.dual,
                self.domain.dual,
                self.__dual_mapping,
                dual_mapping=self,
                dual_base=self,
            )
        else:
            return self._dual_base

    @property
    def adjoint(self):
        """The adjoint of the operator."""
        if self._adjoint_base is None:
            return LinearOperator(
                self.codomain,
                self.domain,
                self.__adjoint_mapping,
                adjoint_mapping=self,
                adjoint_base=self,
            )
        else:
            return self._adjoint_base

    def matrix(self, /, *, dense=False, galerkin=False):
        """Return matrix representation of the operator."""
        if dense:
            return self._compute_dense_matrix(galerkin)
        else:

            # Implement matrix-vector and transposed-matrix-vector products
            if galerkin:

                def matvec(cx):
                    x = self.domain.from_components(cx)
                    y = self(x)
                    yp = self.codomain.to_dual(y)
                    return self.codomain.dual.to_components(yp)

                def rmatvec(cy):
                    y = self.codomain.from_components(cy)
                    x = self.adjoint(y)
                    xp = self.domain.to_dual(x)
                    return self.domain.dual.to_components(xp)

            else:

                def matvec(cx):
                    x = self.domain.from_components(cx)
                    y = self(x)
                    return self.codomain.to_components(y)

                def rmatvec(cyp):
                    yp = self.codomain.dual.from_components(cyp)
                    xp = self.dual(yp)
                    return self.domain.dual.to_components(xp)

            # Implement matrix-matrix and transposed-matrix-matrix products
            def matmat(xmat):
                n, k = xmat.shape
                assert n == self.domain.dim
                ymat = np.zeros((self.codomain.dim, k))
                for j in range(k):
                    cx = xmat[:, j]
                    ymat[:, j] = matvec(cx)
                return ymat

            def rmatmat(ymat):
                m, k = ymat.shape
                assert m == self.codomain.dim
                xmat = np.zeros((self.domain.dim, k))
                for j in range(k):
                    cy = ymat[:, j]
                    xmat[:, j] = rmatvec(cy)
                return xmat

            # Return the scipy LinearOperator
            return ScipyLinOp(
                (self.codomain.dim, self.domain.dim),
                matvec=matvec,
                rmatvec=rmatvec,
                matmat=matmat,
                rmatmat=rmatmat,
            )

    def random_svd(self, rank, /, *, power=0, galerkin=False):
        """
        Returns an approximate singular value decomposition of an operator using
        the randomised method of Halko et al. 2011.

        Let X and Y be the domain and codomain of the operator, A, and E be Euclidean
        space of dimension rank. The factorisation then takes the form

        A = L D R

        where the factors map:

        R : X --> E
        D : E --> E
        L : E --> Y

        with D diagonal and comprises the singular values for the operator.

        Args:

            rank (int) : rank for the decomposition.
            power (int) : exponent within the power iterations.
            galerkin (bool) : If true, use the Galerkin representation of the
                operator. Only possible if the operator maps between Hilbert spaces.

        Returns:
            LinearOperator: The left factor, L.
            DiagonalLinearOperator: The diagonal factor, D.
            LinearOperator: The right factor, R.

        """
        matrix = self.matrix(galerkin=galerkin)
        m, n = matrix.shape
        k = min(m, n)
        rank = rank if rank <= k else k
        qr_factor = fixed_rank_random_range(matrix, rank, power)
        left_factor, singular_values, right_factor_transposed = random_svd(
            matrix, qr_factor
        )

        euclidean = EuclideanSpace(rank)
        diagonal = DiagonalLinearOperator(euclidean, euclidean, singular_values)

        if galerkin:

            def right_mapping(x):
                cx = self.domain.to_components(x)
                return right_factor_transposed @ cx

            def right_mapping_adjoint(cx):
                cxp = right_factor_transposed.T @ cx
                xp = self.domain.dual.from_components(cxp)
                return self.domain.from_dual(xp)

            right = LinearOperator(
                self.domain,
                euclidean,
                right_mapping,
                adjoint_mapping=right_mapping_adjoint,
            )

            def left_mapping(cx):
                cyp = left_factor @ cx
                yp = self.codomain.dual.from_components(cyp)
                return self.codomain.from_dual(yp)

            def left_mapping_adjoint(y):
                cy = self.codomain.to_components(y)
                return left_factor.T @ cy

            left = LinearOperator(
                euclidean,
                self.codomain,
                left_mapping,
                adjoint_mapping=left_mapping_adjoint,
            )

        else:

            def right_mapping(x):
                cx = self.domain.to_components(x)
                return right_factor_transposed @ cx

            def right_mapping_dual(cp):
                c = euclidean.from_dual(cp)
                cxp = right_factor_transposed.T @ c
                return self.domain.dual.from_components(cxp)

            right = LinearOperator(
                self.domain, euclidean, right_mapping, dual_mapping=right_mapping_dual
            )

            def left_mapping(c):
                cy = left_factor @ c
                return self.codomain.from_components(cy)

            def left_mapping_dual(yp):
                cpy = self.codomain.dual.to_components(yp)
                c = left_factor.T @ cpy
                return euclidean.to_dual(c)

            left = LinearOperator(
                euclidean, self.codomain, left_mapping, dual_mapping=left_mapping_dual
            )

        # Return the factors.
        return left, diagonal, right

    def random_eig(self, rank, /, *, power=0):
        """
        Returns an approximate eigenvalue decomposition of a self-adjoint operator using
        the randomised method of Halko et al. 2011.

        Let X  the domain the operator, A, and E be Euclidean space of dimension rank.
        The factorisation then takes the form

        A = U D U*

        where the factors map:

        U : E --> X
        D : E --> E

        with D diagonal and comprises the eigenvalues of the operator.

        If the diagonal values are non-zero, we can also factor an
        approximation to the inverse mapping as:

        A^{-1} = V D^{-1} V*

        where V = I I* U with I the coordinate_inclusion mapping on the Hilbert space.


        Args:
            rank (int) : rank for the decomposition.
            power (int) : exponent within the power iterations.
            inverse (bool): If true, return the decomposition for
                the inverse operator.

        Returns:
            LinearOperator: The factor, U.
            DiagonalLinearOperator: The diagonal factor, D.
        """

        assert self.is_automorphism
        matrix = self.matrix(galerkin=True)
        m, n = matrix.shape
        k = min(m, n)
        rank = rank if rank <= k else k
        qr_factor = fixed_rank_random_range(matrix, rank, power)
        eigenvectors, eigenvalues = random_eig(matrix, qr_factor)
        euclidean = EuclideanSpace(rank)
        diagonal = DiagonalLinearOperator(euclidean, euclidean, eigenvalues)

        def mapping(c):
            cyp = eigenvectors @ c
            yp = self.domain.dual.from_components(cyp)
            return self.domain.from_dual(yp)

        def adjoint_mapping(x):
            cx = self.domain.to_components(x)
            return eigenvectors.T @ cx

        expansion = LinearOperator(
            euclidean, self.domain, mapping, adjoint_mapping=adjoint_mapping
        )

        return expansion, diagonal

    def random_cholesky(self, rank, /, *, power=0):
        """
        Returns an approximate Cholesky decomposition of a positive-definite and
        self-adjoint operator using the randomised method of Halko et al. 2011.

        Let X  the domain the operator, A, and E be Euclidean space of dimension rank.
        The factorisation then takes the form

        A = F F*

        where F : E --> X.

        Args:

            rank (int) : rank for the decomposition.
            power (int) : exponent within the power iterations.

        Returns:
            LinearOperator : The Cholesky factor, F

        """
        assert self.is_automorphism
        matrix = self.matrix(galerkin=True)
        m, n = matrix.shape
        k = min(m, n)
        rank = rank if rank <= k else k
        qr_factor = fixed_rank_random_range(matrix, rank, power)
        cholesky_factor = random_cholesky(matrix, qr_factor)

        def mapping(x):
            cyp = cholesky_factor @ x
            yp = self.codomain.dual.from_components(cyp)
            return self.codomain.from_dual(yp)

        def adjoint_mapping(y):
            x = self.codomain.to_components(y)
            return cholesky_factor.T @ x

        return LinearOperator(
            EuclideanSpace(rank), self.domain, mapping, adjoint_mapping=adjoint_mapping
        )

    def random_preconditioner(self, rank, /, *, power=0):
        """
        Returns an approximate inverse of a self-adjoint operator
        based on a random Cholesky factorisation.

        Args:
            rank (int) : rank for the decomposition.
            sigma (float | None): The shift to apply in forming the inverse.
            power (int) : exponent within the power iterations.
        """
        assert self.domain == self.codomain
        U, D = self.random_eig(rank, power=power)
        sigma = 1
        F = U @ D.sqrt
        M = F.domain.identity_operator() + (1 / sigma) * F.adjoint @ F
        N = CholeskySolver()(M)
        return (1 / sigma) * self.domain.identity_operator() - (
            1 / sigma**2
        ) * F @ N @ F.adjoint

    def _dual_mapping_default(self, yp):
        # Default implementation of the dual mapping.
        return LinearForm(self.domain, mapping=lambda x: yp(self(x)))

    def _dual_mapping_from_adjoint(self, yp):
        # Dual mapping in terms of the adjoint.
        y = self.codomain.from_dual(yp)
        x = self.__adjoint_mapping(y)
        return self.domain.to_dual(x)

    def _adjoint_mapping_from_dual(self, y):
        # Adjoing mapping in terms of the dual.
        yp = self.codomain.to_dual(y)
        xp = self.__dual_mapping(yp)
        return self.domain.from_dual(xp)

    def _compute_dense_matrix(self, galerkin=False):
        # Compute the matrix representation in dense form.
        matrix = np.zeros((self.codomain.dim, self.domain.dim))
        a = self.matrix(galerkin=galerkin)
        cx = np.zeros(self.domain.dim)
        for i in range(self.domain.dim):  # todo: parellise?
            cx[i] = 1
            matrix[:, i] = (a @ cx)[:]
            cx[i] = 0
        return matrix

    def __neg__(self):
        """negative unary"""
        domain = self.domain
        codomain = self.codomain

        def mapping(x):
            return codomain.negative(self(x))

        def dual_mapping(yp):
            return domain.dual.negative(self.dual(yp))

        def adjoint_mapping(y):
            return domain.negative(self.adjoint(y))

        return LinearOperator(
            domain,
            codomain,
            mapping,
            dual_mapping=dual_mapping,
            adjoint_mapping=adjoint_mapping,
        )

    def __mul__(self, a):
        """Multiply by a scalar."""
        domain = self.domain
        codomain = self.codomain

        def mapping(x):
            return codomain.multiply(a, self(x))

        def dual_mapping(yp):
            return domain.dual.multiply(a, self.dual(yp))

        def adjoint_mapping(y):
            return domain.multiply(a, self.adjoint(y))

        return LinearOperator(
            domain,
            codomain,
            mapping,
            dual_mapping=dual_mapping,
            adjoint_mapping=adjoint_mapping,
        )

    def __rmul__(self, a):
        """Multiply by a scalar."""
        return self * a

    def __truediv__(self, a):
        """Divide by scalar."""
        return self * (1 / a)

    def __add__(self, other):
        """Add another operator."""
        domain = self.domain
        codomain = self.codomain

        def mapping(x):
            return codomain.add(self(x), other(x))

        def dual_mapping(yp):
            return domain.dual.add(self.dual(yp), other.dual(yp))

        def adjoint_mapping(y):
            return domain.add(self.adjoint(y), other.adjoint(y))

        return LinearOperator(
            domain,
            codomain,
            mapping,
            dual_mapping=dual_mapping,
            adjoint_mapping=adjoint_mapping,
        )

    def __sub__(self, other):
        """Subtract another operator."""
        domain = self.domain
        codomain = self.codomain

        def mapping(x):
            return codomain.subtract(self(x), other(x))

        def dual_mapping(yp):
            return domain.dual.subtract(self.dual(yp), other.dual(yp))

        def adjoint_mapping(y):
            return domain.subtract(self.adjoint(y), other.adjoint(y))

        return LinearOperator(
            domain,
            codomain,
            mapping,
            dual_mapping=dual_mapping,
            adjoint_mapping=adjoint_mapping,
        )

    def __matmul__(self, other):
        """Compose with another operator."""
        domain = other.domain
        codomain = self.codomain

        def mapping(x):
            return self(other(x))

        def dual_mapping(yp):
            return other.dual(self.dual(yp))

        def adjoint_mapping(y):
            return other.adjoint(self.adjoint(y))

        return LinearOperator(
            domain,
            codomain,
            mapping,
            dual_mapping=dual_mapping,
            adjoint_mapping=adjoint_mapping,
        )

    def __str__(self):
        """Print the operator as its dense matrix representation."""
        return self.matrix(dense=True).__str__()


class BlockLinearOperator(LinearOperator):
    """
    Class for linear operators acting between direct sums of Hilbert spaces that are defined
    in a blockwise manner.

    Instances are formed from lists of list of LinearOperators. Methods to return the subblock
    operators are provided.
    """

    def __init__(self, blocks):

        # Check and form the list of domains and codomains.
        domains = [operator.domain for operator in blocks[0]]
        codomains = []
        for row in blocks:
            assert domains == [operator.domain for operator in row]
            codomain = row[0].codomain
            assert all([operator.codomain == codomain for operator in row])
            codomains.append(codomain)

        domain = HilbertSpaceDirectSum(domains)
        codomain = HilbertSpaceDirectSum(codomains)

        self._domains = domains
        self._codomains = codomains
        self._blocks = blocks
        self._row_dim = len(blocks)
        self._col_dim = len(blocks[0])

        super().__init__(
            domain, codomain, self.__mapping, adjoint_mapping=self.__adjoint_mapping
        )

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

    def block(self, i, j):
        """
        Returns the operator in the (i,j)th sub-block.
        """
        assert i >= 0 and i < self.row_dim
        assert j >= 0 and j < self.col_dim
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


class DiagonalLinearOperator(LinearOperator):
    """
    Class for Linear operators whose matrix representation is diagonal.
    """

    def __init__(self, domain, codomain, diagonal_values, /, *, galerkin=False):
        """
        Args:
            domain (HilbertSpace): The domain of the operator.
            codoomain (HilbertSpace): The codomain of the operator.
            diagonal_values (numpy vector): Diagonal values for the
                operator's matrix representation.
            galerkin (bool): true is galerkin representation is used.
        """

        assert domain.dim == codomain.dim
        assert domain.dim == len(diagonal_values)
        self._diagonal_values = diagonal_values
        matrix = diags([diagonal_values], [0])
        operator = LinearOperator.from_matrix(
            domain, codomain, matrix, galerkin=galerkin
        )
        super().__init__(
            operator.domain,
            operator.codomain,
            operator,
            dual_mapping=operator.dual,
            adjoint_mapping=operator.adjoint,
        )

    @property
    def diagonal_values(self):
        """
        Return the diagonal values.
        """
        return self._diagonal_values

    @property
    def inverse(self):
        """
        return the inverse operator. Valid only if diagonal values
        are non-zero.
        """
        assert all([val != 0 for val in self._diagonal_values])
        diagonal_values = np.reciprocal(self._diagonal_values.copy())
        return DiagonalLinearOperator(self.codomain, self.domain, diagonal_values)

    @property
    def sqrt(self):
        """
        Returns the square root of the operator. Valid only if diagonal values
        are non-negative.
        """
        assert all([val >= 0 for val in self._diagonal_values])
        return DiagonalLinearOperator(
            self.domain, self.codomain, np.sqrt(self._diagonal_values)
        )


class LinearForm:
    """
    Class for linear forms on a Hilbert space. Can be specified by its
    action or through its components.
    """

    def __init__(self, domain, /, *, mapping=None, components=None):
        """
        Args:
            domain (HilbertSpace): Domain of the linear form.
            mapping (callable | None): A functor that performs the action
                of the linear form on a vector.
            matrix (MatrixLike | None): The matrix representation of the
                form, this having shape (1,dim) with dim the dimension of
                the domain.
        """

        self._domain = domain
        if components is None:
            self._components = None
            if mapping is None:
                raise AssertionError("Neither mapping nor components specified.")
            else:
                self._mapping = mapping
        else:
            self._components = components
            if mapping is None:
                self._mapping = self._mapping_from_components
            else:
                self._mapping = mapping

    @staticmethod
    def from_linear_operator(operator):
        """
        Form a linear form from an linear operator mapping onto one-dimensional Euclidean space.
        """
        assert operator.codomain == EuclideanSpace(1)
        return LinearForm(operator.domain, mapping=lambda x: operator(x)[0])

    @property
    def domain(self):
        """Return the form the domain is defined on"""
        return self._domain

    @property
    def components_stored(self):
        """True is the form has its components stored."""
        return self._components is not None

    @property
    def components(self):
        """Return the components of the form."""
        if self.components_stored:
            return self._components
        else:
            self.store_components()
            return self.components

    def store_components(self):
        """Compute and store the forms components."""
        if not self.components_stored:
            self._components = np.zeros(self.domain.dim)
            cx = np.zeros(self.domain.dim)
            for i in range(self.domain.dim):
                cx[i] = 1
                x = self.domain.from_components(cx)
                self._components[i] = self(x)
                cx[i] = 0

    @property
    def as_linear_operator(self):
        """
        Return the linear form as a LinearOperator.
        """
        return LinearOperator(
            self.domain,
            EuclideanSpace(1),
            lambda x: np.array([self(x)]),
            dual_mapping=lambda y: y * self,
        )

    def __call__(self, x):
        """Action of the form on a vector"""
        return self._mapping(x)

    def __neg__(self):
        """negative unary"""
        if self.components_stored:
            return LinearForm(self.domain, components=-self._components)
        else:
            return LinearForm(self.domain, mapping=lambda x: -self(x))

    def __mul__(self, a):
        """Multiply by a scalar."""
        if self.components_stored:
            return LinearForm(self.domain, components=a * self._components)
        else:
            return LinearForm(self.domain, mapping=lambda x: a * self(x))

    def __rmul__(self, a):
        """Multiply by a scalar."""
        return self * a

    def __truediv__(self, a):
        """Divide by scalar."""
        return self * (1 / a)

    def __add__(self, other):
        """Add another form."""
        if self.components_stored and other.components_stored:
            return LinearForm(
                self.domain, components=self.components + other.components
            )
        else:
            return LinearForm(self.domain, mapping=lambda x: self(x) + other(x))

    def __sub__(self, other):
        """Subtract another form."""
        if self.components_stored and other.components_stored:
            return LinearForm(
                self.domain, components=self.components - other.components
            )
        else:
            return LinearForm(self.domain, mapping=lambda x: self(x) - other(x))

    def __str__(self):
        return self.components.__str__()

    def _mapping_from_components(self, x):
        # Implement the action of the form using its components.
        return np.dot(self._components, self.domain.to_components(x))


class GaussianMeasure:
    """
    Class for Gaussian measures on a Hilbert space.
    """

    def __init__(
        self,
        domain,
        covariance,
        /,
        *,
        expectation=None,
        sample=None,
        inverse_covariance=None,
    ):
        """
        Args:
            domain (HilbertSpace): The Hilbert space on which the measure is defined.
            covariance (LinearOperator): A self-adjoint and non-negative linear operator on the domain
            expectation (vector | zero): The expectation value of the measure.
            sample (callable | None): A functor that returns a random sample from the measure.
            inverse_covariance (LinearOperator | None): The inverse of the covaraiance.

        Notes:
            If the inverse of the covariance is not provided, it is computed internally
            using a matrix-free CG solver.
        """
        self._domain = domain
        self._covariance = covariance

        if expectation is None:
            self._expectation = self.domain.zero
        else:
            self._expectation = expectation

        self._sample = sample

        if inverse_covariance is None:
            self._inverse_covariance = CGSolver()(covariance)
        else:
            self._inverse_covariance = inverse_covariance

    @staticmethod
    def from_factored_covariance(factor, /, *, inverse_factor=None, expectation=None):
        """
        For a Gaussian measure whos covariance, C, is approximated in the form C = LL*,
        with L a mapping into the domain from Euclidean space.

        Args:
            factor (LinearOperator): Linear operator from Euclidean space into the
                domain of the measure.
            expectation (vector | zero): expected value of the measure.
            inverse factor (LinearOperator): An analgous factorisation of the inverse covariance.

        Returns:
            GassianMeasure: The measure with the required covariance and expectation.
        """

        def sample():
            value = factor(norm().rvs(size=factor.domain.dim))
            if expectation is None:
                return value
            else:
                return factor.codomain.add(value, expectation)

        covariance = factor @ factor.adjoint

        _inverse_covariance = (
            inverse_factor.adjoint @ inverse_factor
            if inverse_factor is not None
            else None
        )

        return GaussianMeasure(
            factor.codomain,
            covariance,
            expectation=expectation,
            sample=sample,
            inverse_covariance=_inverse_covariance,
        )

    @staticmethod
    def from_samples(domain, samples):
        """
        Forms a Gaussian measure from a set of samples based on
        the sample expectation and sample variance.

        Args:
            domain (HilbertSpace): The space the measure is defined on.
            samples ([vectors]): A list of samples.

        Notes:
            :
        """
        assert all([domain.is_element(x) for x in samples])

        n = len(samples)
        expectation = domain.zero
        for x in samples:
            expectation = domain.axpy(1 / n, x, expectation)

        if n == 1:
            covariance = LinearOperator.self_adjoint(domain, lambda x: domain.zero)

            def sample():
                return expectation

        else:
            offsets = [domain.subtract(x, expectation) for x in samples]
            covariance = LinearOperator.self_adjoint_from_tensor_product(
                domain, offsets
            ) / (n - 1)

            def sample():
                x = expectation
                randoms = np.random.randn(len(offsets))
                for y, r in zip(offsets, randoms):
                    x = domain.axpy(r / np.sqrt(n - 1), y, x)
                return x

        return GaussianMeasure(
            domain, covariance, expectation=expectation, sample=sample
        )

    @property
    def domain(self):
        """The Hilbert space the measure is defined on."""
        return self._domain

    @property
    def covariance(self):
        """The covariance operator as an instance of LinearOperator."""
        return self._covariance

    @property
    def inverse_covariance(self):
        """The covariance operator as an instance of LinearOperator."""
        return self._inverse_covariance

    @property
    def expectation(self):
        """The expectation of the measure."""
        return self._expectation

    def sample(self):
        """
        Returns a random sample drawn from the measure.
        """
        if self._sample is None:
            raise NotImplementedError("Sample method is not set.")
        else:
            return self._sample()

    def samples(self, n):
        """
        Returns a list of n samples form the measure.
        """
        assert n >= 1
        return [self.sample() for _ in range(n)]

    def sample_expectation(self, n):
        """
        Returns the sample expectation using n > 1 samples.
        """
        assert n >= 1
        expectation = self.domain.zero
        for _ in range(n):
            sample = self.sample()
            expectation = self.domain.axpy(1 / n, sample, expectation)
        return expectation

    def affine_mapping(self, /, *, operator=None, translation=None):
        """
        Returns the push forward of the measure under an affine mapping.

        Args:
            operator (LinearOperator): The operator part of the mapping.
            translation (vector): The translational part of the mapping.

        Returns:
            Gaussian Measure: The transformed measure defined on the
                codomain of the operator.

        Raises:
            ValueError: If the domain of the operator domain is not
                the domain of the measure.

        Notes:
            If operator is not set, it defaults to the identity.
            It translation is not set, it defaults to zero.
        """

        if operator is None:
            _operator = self.domain.identity_operator()
        else:
            _operator = operator

        if translation is None:
            _translation = _operator.codomain.zero
        else:
            _translation = translation

        covariance = _operator @ self.covariance @ _operator.adjoint
        expectation = _operator.codomain.add(_operator(self.expectation), _translation)

        def sample():
            return _operator.codomain.add(_operator(self.sample()), _translation)

        return GaussianMeasure(
            _operator.codomain, covariance, expectation=expectation, sample=sample
        )

    def low_rank_approximation(self, rank, /, *, power=0):
        """
        Returns an approximation to the measure with the covariance operator
        replaced by a low-rank Cholesky decomposition along with the associated
        sampling method.
        """
        F = self.covariance.random_cholesky(rank, power=power)
        return GaussianMeasure.from_factored_covariance(F, expectation=self.expectation)

    def __neg__(self):
        """Negative of the measure."""
        return GaussianMeasure(
            self.domain,
            -self.covariance,
            expectation=self.domain.negative(self.expectation),
            sample=lambda: self.domain.negative(self.sample()),
        )

    def __mul__(self, alpha):
        """Multiply the measure by a scalar."""
        return GaussianMeasure(
            self.domain,
            alpha * alpha * self.covariance,
            expectation=self.domain.multiply(alpha, self.expectation),
            sample=lambda: self.domain.multiply(alpha, self.sample()),
        )

    def __rmul__(self, alpha):
        """Multiply the measure by a scalar."""
        return self * alpha

    def __add__(self, other):
        """Add two measures on the same domain."""
        return GaussianMeasure(
            self.domain,
            self.covariance + other.covariance,
            expectation=self.domain.add(self.expectation, other.expectation),
            sample=lambda: self.domain.add(self.sample(), other.sample()),
        )

    def __sub__(self, other):
        """Subtract two measures on the same domain."""
        return GaussianMeasure(
            self.domain,
            self.covariance + other.covariance,
            expectation=self.domain.subtract(self.expectation, other.expectation),
            sample=lambda: self.domain.subtract(self.sample(), other.sample()),
        )


class LinearSolver(ABC):
    """
    Abstract base class for linear solvers.
    """


class DirectLinearSolver(LinearSolver):
    """
    Abstract base class for direct linear solvers.
    """


class LUSolver(DirectLinearSolver):
    """
    Direct Linear solver class based on LU decomposition of the
    matrix representation.
    """

    def __init__(self, /, *, galerkin=False):
        """
        Args:
            galerkin (bool): If true, the Galerkin matrix representation is used.
        """
        self._galerkin = galerkin

    def __call__(self, operator):
        """
        Returns the inverse of a LinearOperator based on the LU factorisation
        of its dense matrix representation.
        """

        assert operator.domain.dim == operator.codomain.dim
        matrix = operator.matrix(dense=True, galerkin=self._galerkin)
        factor = lu_factor(matrix, overwrite_a=True)

        def matvec(cy):
            return lu_solve(factor, cy, 0)

        def rmatvec(cx):
            return lu_solve(factor, cx, 1)

        inverse_matrix = ScipyLinOp(
            (operator.domain.dim, operator.codomain.dim),
            matvec=matvec,
            rmatvec=rmatvec,
        )

        return LinearOperator.from_matrix(
            operator.codomain, operator.domain, inverse_matrix, galerkin=self._galerkin
        )


class CholeskySolver(DirectLinearSolver):
    """
    Direct Linear solver class based on Cholesky decomposition of the
    matrix representation. It is assumed that the operator's matrix
    representation is self-adjoint and positive-definite.
    """

    def __init__(self, /, *, galerkin=False):
        """
        galerkin (bool): If true, use the Galerkin matrix representation.
        """
        self._galerkin = galerkin

    def __call__(self, operator):
        """
        Returns the inverse of a LinearOperator based on the LU factorisation
        of its dense matrix representation.
        """

        assert operator.is_automorphism

        matrix = operator.matrix(dense=True, galerkin=self._galerkin)
        factor = cho_factor(matrix, overwrite_a=False)

        def matvec(cy):
            return cho_solve(factor, cy)

        inverse_matrix = ScipyLinOp(
            (operator.domain.dim, operator.codomain.dim), matvec=matvec, rmatvec=matvec
        )

        return LinearOperator.from_matrix(
            operator.domain, operator.domain, inverse_matrix, galerkin=self._galerkin
        )


class IterativeLinearSolver(LinearSolver):
    """
    Abstract base class for direct linear solvers.
    """


class CGMatrixSolver(IterativeLinearSolver):
    """
    Linear solver for self-adjoint operators based on the application
    of the conjugate gradient algorithm to the matrix representation.

    It is assumed that the matrix representation of the operator
    is also self-adjoint. This will hold automatically when the
    Galerkin representation is used.
    """

    def __init__(
        self, /, *, galerkin=False, rtol=1.0e-5, atol=0, maxiter=None, callback=None
    ):
        """
        Args:
            galerkin (bool): True if the Galerkin matrix representation is used.
            rtol (float): relative tolerance within convergence checks.
            atol (float): absolute tolerance within convergence checks.
            maxiter (int): maximum number of iterations to allow.
            callback (callable): callable function after each iteration. This function
                takes in as argument the current solution vector.
        """
        self._galerkin = galerkin
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter
        self._callback = callback

    def __call__(self, operator, /, *, preconditioner=None):
        """
        Returns the IterativeLinearOperator corresponding to the given operator.

        Args:
            operator (LinearOperator): The operator to invert.

        """
        assert operator.is_automorphism
        domain = operator.codomain
        matrix = operator.matrix(galerkin=self._galerkin)

        if preconditioner is None:
            matrix_preconditioner = None
        else:
            matrix_preconditioner = preconditioner.matrix(galerkin=self._galerkin)

        def mapping(y):
            cy = domain.to_components(y)
            cxp = cg(
                matrix,
                cy,
                rtol=self._rtol,
                atol=self._atol,
                maxiter=self._maxiter,
                M=matrix_preconditioner,
                callback=self._callback,
            )[0]
            if self._galerkin:
                xp = domain.dual.from_components(cxp)
                return domain.from_dual(xp)
            else:
                return domain.from_components(cxp)

        return LinearOperator(domain, domain, mapping, adjoint_mapping=mapping)


class BICGMatrixSolver(IterativeLinearSolver):
    """
    Linear solver for general operators based on the application
    of the biconjugate gradient algorithm to the matrix representation.
    """

    def __init__(
        self, /, *, galerkin=False, rtol=1.0e-5, atol=0, maxiter=None, callback=None
    ):
        """
        Args:
            galerkin (bool): True if the Galerkin matrix representation is used.
            rtol (float): relative tolerance within convergence checks.
            atol (float): absolute tolerance within convergence checks.
            maxiter (int): maximum number of iterations to allow.
            callback (callable): callable function after each iteration. This function
                takes in as argument the current solution vector.
        """
        self._galerkin = galerkin
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter
        self._callback = callback

    def __call__(self, operator, /, *, preconditioner=None):
        """
        Returns the IterativeLinearOperator corresponding to the given operator.

        Args:
            operator (LinearOperator): The operator to invert.
        """
        assert operator.is_square
        domain = operator.codomain
        codomain = operator.domain
        matrix = operator.matrix(galerkin=self._galerkin)

        if preconditioner is None:
            matrix_preconditioner = None
        else:
            matrix_preconditioner = preconditioner.matrix(galerkin=self._galerkin)

        def mapping(y):
            cy = domain.to_components(y)
            cxp = bicg(
                matrix,
                cy,
                rtol=self._rtol,
                atol=self._atol,
                maxiter=self._maxiter,
                M=matrix_preconditioner,
                callback=self._callback,
            )[0]
            if self._galerkin:
                xp = codomain.dual.from_components(cxp)
                return codomain.from_dual(xp)
            else:
                return codomain.from_components(cxp)

        def adjoint_mapping(x):
            cx = codomain.to_components(x)
            cyp = bicg(
                matrix.T,
                cx,
                rtol=self._rtol,
                atol=self._atol,
                maxiter=self._maxiter,
                M=matrix_preconditioner,
                callback=self._callback,
            )[0]
            if self._galerkin:
                yp = domain.dual.from_components(cyp)
                return domain.from_dual(yp)
            else:
                return domain.from_components(cyp)

        return LinearOperator(domain, domain, mapping, adjoint_mapping=adjoint_mapping)


class BICGStabMatrixSolver(IterativeLinearSolver):
    """
    Linear solver for general operators based on the application
    of the biconjugate gradient stabilised algorithm to the matrix representation.
    """

    def __init__(
        self, /, *, galerkin=False, rtol=1.0e-5, atol=0, maxiter=None, callback=None
    ):
        """
        Args:
            galerkin (bool): True if the Galerkin matrix representation is used.
            rtol (float): relative tolerance within convergence checks.
            atol (float): absolute tolerance within convergence checks.
            maxiter (int): maximum number of iterations to allow.
            callback (callable): callable function after each iteration. This function
                takes in as argument the current solution vector.
        """
        self._galerkin = galerkin
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter
        self._callback = callback

    def __call__(self, operator, /, *, preconditioner=None):
        """
        Returns the IterativeLinearOperator corresponding to the given operator.

        Args:
            operator (LinearOperator): The operator to invert.
        """
        assert operator.is_square
        domain = operator.codomain
        codomain = operator.domain
        matrix = operator.matrix(galerkin=self._galerkin)

        if preconditioner is None:
            matrix_preconditioner = None
        else:
            matrix_preconditioner = preconditioner.matrix(galerkin=self._galerkin)

        def mapping(y):
            cy = domain.to_components(y)
            cxp = bicgstab(
                matrix,
                cy,
                rtol=self._rtol,
                atol=self._atol,
                maxiter=self._maxiter,
                M=matrix_preconditioner,
                callback=self._callback,
            )[0]
            if self._galerkin:
                xp = codomain.dual.from_components(cxp)
                return codomain.from_dual(xp)
            else:
                return codomain.from_components(cxp)

        def adjoint_mapping(x):
            cx = codomain.to_components(x)
            cyp = bicgstab(
                matrix.T,
                cx,
                rtol=self._rtol,
                atol=self._atol,
                maxiter=self._maxiter,
                M=matrix_preconditioner,
                callback=self._callback,
            )[0]
            if self._galerkin:
                yp = domain.dual.from_components(cyp)
                return domain.from_dual(yp)
            else:
                return domain.from_components(cyp)

        return LinearOperator(domain, domain, mapping, adjoint_mapping=adjoint_mapping)


class GMRESMatrixSolver(IterativeLinearSolver):
    """
    Linear solver for general operators based on the application
    of the GMRES algorithm to the matrix representation.
    """

    def __init__(
        self,
        /,
        *,
        galerkin=False,
        rtol=1.0e-5,
        atol=0,
        restart=None,
        maxiter=None,
        callback=None,
        callback_type=None,
    ):
        """
        Args:
            galerkin (bool): True if the Galerkin matrix representation is used.
            rtol (float): relative tolerance within convergence checks.
            atol (float): absolute tolerance within convergence checks.
            restart (int): Number of iterations between restarts.
            maxiter (int): maximum number of iterations to allow.
            callback (callable): callable function after each iteration. Signature
                of this function is determined by callback_type.
            callback_type ("x", "pr_norm", "legacy"): If "x" the current solution is
                passed to the callback function, if "pr_norm" it is the preconditioned
                residual norm. The default is "legacy" which means the same as "pr_norm",
                but changes the meaning of maxiter to count inner iterations instead of
                restart cycles.
        """
        self._galerkin = galerkin
        self._rtol = rtol
        self._atol = atol
        self._restart = restart
        self._maxiter = maxiter
        self._callback = callback
        self._callback_type = callback_type

    def __call__(self, operator, /, *, preconditioner=None):
        """
        Returns the IterativeLinearOperator corresponding to the given operator.

        Args:
            operator (LinearOperator): The operator to invert.
            galerkin (bool): True if the Galerkin matrix representation is used.
        """
        assert operator.is_square
        domain = operator.codomain
        codomain = operator.domain
        matrix = operator.matrix(galerkin=self._galerkin)

        if preconditioner is None:
            matrix_preconditioner = None
        else:
            matrix_preconditioner = preconditioner.matrix(galerkin=self._galerkin)

        def mapping(y):
            cy = domain.to_components(y)
            cxp = gmres(
                matrix,
                cy,
                rtol=self._rtol,
                atol=self._atol,
                restart=self._restart,
                maxiter=self._maxiter,
                M=matrix_preconditioner,
                callback=self._callback,
                callback_type=self._callback_type,
            )[0]
            if self._galerkin:
                xp = codomain.dual.from_components(cxp)
                return codomain.from_dual(xp)
            else:
                return codomain.from_components(cxp)

        def adjoint_mapping(x):
            cx = codomain.to_components(x)
            cyp = gmres(
                matrix.T,
                cx,
                rtol=self._rtol,
                atol=self._atol,
                restart=self._restart,
                maxiter=self._maxiter,
                M=matrix_preconditioner,
                callback=self._callback,
                callback_type=self._callback_type,
            )[0]
            if self._galerkin:
                yp = domain.dual.from_components(cyp)
                return domain.from_dual(yp)
            else:
                return domain.from_components(cyp)

        return LinearOperator(domain, domain, mapping, adjoint_mapping=adjoint_mapping)


class CGSolver(IterativeLinearSolver):
    """
    LinearSolver class using the conjugate gradient algorithm without
    use of the matrix representation. Can be applied to self-adjoint
    operators on a general Hilbert space.
    """

    def __init__(self, /, *, rtol=1.0e-5, atol=0, maxiter=None, callback=None):
        """
        Args:
            rtol (float): relative tolerance within convergence checks.
            atol (float): absolute tolerance within convergence checks.
            maxiter (int): maximum number of iterations to allow.
            callback (callable): callable function after each iteration. This function
                takes in as argument the current solution vector.
        """
        if rtol > 0:
            self._rtol = rtol
        else:
            raise ValueError("rtol must be positive")
        if atol >= 0:
            self._atol = atol
        else:
            raise ValueError("atol must be non-negative!")
        if maxiter is None:
            self._maxiter = maxiter
        else:
            if maxiter >= 0:
                self._maxiter = maxiter
            else:
                raise ValueError("maxiter must be None or positive")

        self._callback = callback

    def __call__(self, operator, /, *, preconditioner=None):

        assert operator.is_automorphism

        def mapping(y):

            domain = operator.domain
            x = domain.zero

            r = domain.subtract(y, operator(x))
            if preconditioner is None:
                z = domain.copy(r)
            else:
                z = preconditioner(r)
            p = domain.copy(z)

            y_squared_norm = domain.squared_norm(y)
            if y_squared_norm <= self._atol:
                return y

            tol = np.max([self._atol, self._rtol * y_squared_norm])

            if self._maxiter is None:
                maxiter = 10 * domain.dim
            else:
                maxiter = self._maxiter

            for _ in range(maxiter):

                print(domain.norm(r))

                if domain.norm(r) <= tol:
                    break

                q = operator(p)
                num = domain.inner_product(r, z)
                den = domain.inner_product(p, q)
                alpha = num / den

                x = domain.axpy(alpha, p, x)
                r = domain.axpy(-alpha, q, r)

                if preconditioner is None:
                    z = domain.copy(r)
                else:
                    z = preconditioner(r)

                den = num
                num = operator.domain.inner_product(r, z)
                beta = num / den

                p = domain.multiply(beta, p)
                p = domain.add(p, z)

                if self._callback is not None:
                    self._callback(x)

            return x

        return LinearOperator.self_adjoint(operator.domain, mapping)


###########################################################################
#        Utility functions linked to random matrix decomposition          #
###########################################################################


def fixed_rank_random_range(matrix, rank, power=0):
    """
    Forms the fixed-rank approximation to the range of a matrix using
    a random-matrix method.

    Args:
        matrix (matrix-like): (m,n)-matrix whose range is to be approximated.
        rank (int): The desired rank. Must be greater than 1.
        power (int): The exponent to use within the power iterations.

    Returns:
        matrix: A (m,rank)-matrix whose columns are orthonormal and
            whose span approximates the desired range.

    Notes:
        The input matrix can be a numpy array or a scipy LinearOperator. In the latter case,
        it requires the the matmat, and rmatmat methods have been implemented.

        This method is based on Algorithm 4.4 in Halko et. al. 2011
    """

    m, n = matrix.shape
    random_matrix = np.random.rand(n, rank)

    product_matrix = matrix @ random_matrix
    qr_factor, _ = qr(product_matrix, overwrite_a=True, mode="economic")

    for _ in range(power):
        tilde_product_matrix = matrix.T @ qr_factor
        tilde_qr_factor, _ = qr(tilde_product_matrix, overwrite_a=True, mode="economic")
        product_matrix = matrix @ tilde_qr_factor
        qr_factor, _ = qr(product_matrix, overwrite_a=True, mode="economic")

    return qr_factor


def variable_rank_random_range(matrix, rtol, /, *, rank=None, power=0):
    """
    Forms the variable-rank approximation to the range of a matrix using
    a random-matrix method.

    Args:
        matrix (matrix-like): (m,n)-matrix whose range is to be approximated.
        rtol (float): The desired relative accuracy.
        rank (int): Starting rank for the decomposition. If none, then
            determined from the dimension of the matrix.
        power (int): The exponent to use within the power iterations.

    Returns:
        matrix: A (m,rank)-matrix whose columns are orthonormal and
            whose span approximates the desired range.

    Notes:
        The input matrix can be a numpy array or a scipy LinearOperator. In the latter case,
        it requires the the matmat, and rmatmat methods have been implemented.

        This method is based on Algorithm 4.5 in Halko et. al. 2011
    """
    raise NotImplementedError


def random_svd(matrix, qr_factor):
    """
    Given a matrix, A,  and a low-rank approximation to its range, Q,
    this function returns the approximate SVD factors, (U, S, Vh)
    such that A ~ U @ S @ VT where S is diagonal.

    Based on Algorithm 5.1 of Halko et al. 2011
    """
    small_matrix = qr_factor.T @ matrix
    left_factor, diagonal_factor, right_factor_transposed = svd(
        small_matrix, full_matrices=False, overwrite_a=True
    )
    return (
        qr_factor @ left_factor,
        diagonal_factor,
        right_factor_transposed,
    )


def random_eig(matrix, qr_factor):
    """
    Given a symmetric matrix, A,  and a low-rank approximation to its range, Q,
    this function returns the approximate eigen-decomposition, (U, S)
    such that A ~ U @ S @ U.T where S is diagonal.

    Based on Algorithm 5.3 of Halko et al. 2011
    """
    m, n = matrix.shape
    assert m == n
    small_matrix = qr_factor.T @ matrix @ qr_factor
    eigenvalues, eigenvectors = eigh(small_matrix, overwrite_a=True)
    return qr_factor @ eigenvectors, eigenvalues


def random_cholesky(matrix, qr_factor):
    """
    Given a symmetric and positive-definite matrix, A,  along with a low-rank
    approximation to its range, Q, this function returns the approximate
    Cholesky factorisation A ~ F F*.

    Based on Algorithm 5.5 of Halko et al. 2011
    """
    small_matrix_1 = matrix @ qr_factor
    small_matrix_2 = qr_factor.T @ small_matrix_1
    factor, lower = cho_factor(small_matrix_2, overwrite_a=True)
    identity_operator = np.identity(factor.shape[0])
    inverse_factor = solve_triangular(
        factor, identity_operator, overwrite_b=True, lower=lower
    )
    return small_matrix_1 @ inverse_factor
