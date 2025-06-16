"""
Module defining classes for Hilbert spaces, linear operators and linear forms. 

The classes within this module have interdependencies and so cannot be split 
into separate modules. 
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator as ScipyLinOp
from scipy.sparse import diags

from pygeoinf.random_matrix import (
    fixed_rank_random_range,
    variable_rank_random_range,
    random_svd,
    random_cholesky,
    random_eig,
)


class HilbertSpace:
    """
    A class for real Hilbert spaces. To define an instance, the
    user needs to provide the following:

        (1) The dimension of the space, or the dimension of the
            finite-dimensional approximating space.
        (2) A mapping from elements of the space to their components.
            These components must be expressed as numpy arrays with
            shape (dim) with dim the space's dimension.
        (3) A mapping from components back to the vectors. This
            needs to be the inverse of the mapping in (2), but
            this requirement is not automatically checked.
        (4) The inner product on the space.
        (5) The mapping from the space to its dual.
        (6) The mapping from a dual vector to its representation
            within the space.

    Optinally, custom implementations for the basic vector operations can
    be provided.
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
        vector_multiply=None,
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
        self._vector_multiply = vector_multiply

    @property
    def dim(self):
        """The dimension of the space."""
        return self._dim

    @property
    def has_vector_multiply(self):
        """
        True if multiplication of elements is defined.
        """
        return self._vector_multiply is not None

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
        """
        Returns the zero vector for the space.
        """
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

    def gram_schmidt(self, vectors):
        """
        Given a list of elements of the space, returns a set
        of orthonormal vectors spanning the same sub-space.
        """

        if not all(self.is_element(vector) for vector in vectors):
            raise ValueError("Not all vectors are elements of the space")

        orthonormalised_vectors = []
        for i, vector in enumerate(vectors):
            for j in range(i):
                product = self.inner_product(vector, orthonormalised_vectors[j])
                vector = self.axpy(-product, orthonormalised_vectors[j], vector)
            norm = self.norm(vector)
            vector = self.multiply(1 / norm, vector)
            orthonormalised_vectors.append(vector)

        return orthonormalised_vectors

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
        """
        Sets y = y + a * x.
        """
        return self._axpy(a, x, y)

    def copy(self, x):
        """
        Returns a copy of x.
        """
        return self._copy(x)

    def vector_multiply(self, x1, x2):
        """
        Returns the product of two elements of the space.
        """
        if not self.has_vector_multiply:
            raise NotImplementedError(
                "Vector multiplication not defined on this space."
            )
        return self._vector_multiply(x1, x2)

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
        return self.from_components(np.random.randn(self.dim))

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
        adjoint_mapping=None,
        dual_mapping=None,
        thread_safe=False,
        dual_base=None,
        adjoint_base=None,
    ):
        """
        Args:
            domain (HilbertSpace): The domain of the operator.
            codomain (HilbertSpace): The codomain of the operator.
            mapping (callable): Mapping from the domain to codomain.
            dual_mapping (callable): Optional implementation of
                dual operator's action.
            adjoint_mapping (callable): Optional implementation
                of the adjoint operator's action.

            thread_safe (bool): True if the operators action can be
                safely called in parallel. Default is false.
            dual_base (LinearOperator) : Used internally when defining
                dual operators. Should not be set manually.
            adjoint_base (LinearOperator): Used internally when defining
                adjoint operators. Should not be set manually.
        """
        super().__init__(domain, codomain, mapping)
        self._dual_base = dual_base
        self._adjoint_base = adjoint_base
        self._thread_safe = thread_safe

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
    def from_formal_adjoint(domain, codomain, mapping, formal_adjoint):
        """
        Forms a LinearOperator mapping the domain to the codomain. The
        action of the mapping is provided along with that of its formal
        adjoint. The formal adjoint is a mapping from the codomain to the
        domain and is the adjoint of the operator relative to an L2 inner product.
        """

        def dual_mapping(yp):
            cyp = codomain.dual.to_components(yp)
            y = codomain.from_components(cyp)
            x = formal_adjoint(y)
            cx = domain.to_components(x)
            return domain.dual.from_components(cx)

        return LinearOperator(domain, codomain, mapping, dual_mapping=dual_mapping)

    @staticmethod
    def formally_self_adjoint(domain, mapping):
        """
        Forms a LinearOperator on the domain given the mapping
        on the assumption that the operator is formally self-adjoint.
        """
        return LinearOperator.from_formal_adjoint(domain, domain, mapping, mapping)

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
        if not all(form.domain == domain for form in forms):
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
        Forms a self-adjoint operator from its Galerkin matrix representation.
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
        Forms a LinearOperator between Hilbert spaces from the tensor product
        of a list of pairs of vectors.

        Args:
            domain (HilbertSpace): Domain for the linear operator.
            codomain (HilbertSpace): Codomain for the linear operator.
            vector_pairs ([[codomain vector, domain vector]]): A list of pairs of vectors
                from which the tensor product is to be constructed.
            weights ([float]): Optional list of weights for the terms in the tensor
               product. If none is provided default weights of one are used.
        """

        assert all(domain.is_element(vector) for _, vector in vector_pairs)
        assert all(codomain.is_element(vector) for vector, _ in vector_pairs)

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

        assert all(domain.is_element(vector) for vector in vectors)

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

    @property
    def thread_safe(self):
        """
        Returns True if operator is thread safe.
        """
        return self._thread_safe

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

    def random_svd(
        self, rank, /, *, power=0, galerkin=False, rtol=1e-3, method="fixed"
    ):
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

        if method == "fixed":
            qr_factor = fixed_rank_random_range(matrix, rank, power=power)
        elif method == "variable":
            qr_factor = variable_rank_random_range(matrix, rank, power=power, rtol=rtol)
        else:
            raise ValueError("Invalid method selected")

        left_factor, singular_values, right_factor_transposed = random_svd(
            matrix, qr_factor
        )

        euclidean = EuclideanSpace(qr_factor.shape[1])
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

    def random_eig(self, rank, /, *, power=0, rtol=1e-3, method="fixed"):
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

        if method == "fixed":
            qr_factor = fixed_rank_random_range(matrix, rank, power=power)
        elif method == "variable":
            qr_factor = variable_rank_random_range(matrix, rank, power=power, rtol=rtol)
        else:
            raise ValueError("Invalid method selected")

        eigenvectors, eigenvalues = random_eig(matrix, qr_factor)
        euclidean = EuclideanSpace(qr_factor.shape[1])
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

    def random_cholesky(self, rank, /, *, power=0, rtol=1e-3, method="fixed"):
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

        if method == "fixed":
            qr_factor = fixed_rank_random_range(matrix, rank, power=power)
        elif method == "variable":
            qr_factor = variable_rank_random_range(matrix, rank, power=power, rtol=rtol)
        else:
            raise ValueError("Invalid method selected")

        cholesky_factor = random_cholesky(matrix, qr_factor)

        def mapping(x):
            cyp = cholesky_factor @ x
            yp = self.codomain.dual.from_components(cyp)
            return self.codomain.from_dual(yp)

        def adjoint_mapping(y):
            x = self.codomain.to_components(y)
            return cholesky_factor.T @ x

        return LinearOperator(
            EuclideanSpace(qr_factor.shape[1]),
            self.domain,
            mapping,
            adjoint_mapping=adjoint_mapping,
        )

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
        for i in range(self.domain.dim):
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

        def adjoint_mapping(y):
            return domain.negative(self.adjoint(y))

        return LinearOperator(
            domain,
            codomain,
            mapping,
            adjoint_mapping=adjoint_mapping,
        )

    def __mul__(self, a):
        """Multiply by a scalar."""
        domain = self.domain
        codomain = self.codomain

        def mapping(x):
            return codomain.multiply(a, self(x))

        def adjoint_mapping(y):
            return domain.multiply(a, self.adjoint(y))

        return LinearOperator(
            domain,
            codomain,
            mapping,
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

        def adjoint_mapping(y):
            return domain.add(self.adjoint(y), other.adjoint(y))

        return LinearOperator(
            domain,
            codomain,
            mapping,
            adjoint_mapping=adjoint_mapping,
        )

    def __sub__(self, other):
        """Subtract another operator."""
        domain = self.domain
        codomain = self.codomain

        def mapping(x):
            return codomain.subtract(self(x), other(x))

        def adjoint_mapping(y):
            return domain.subtract(self.adjoint(y), other.adjoint(y))

        return LinearOperator(
            domain,
            codomain,
            mapping,
            adjoint_mapping=adjoint_mapping,
        )

    def __matmul__(self, other):
        """Compose with another operator."""
        domain = other.domain
        codomain = self.codomain

        def mapping(x):
            return self(other(x))

        def adjoint_mapping(y):
            return other.adjoint(self.adjoint(y))

        return LinearOperator(
            domain,
            codomain,
            mapping,
            adjoint_mapping=adjoint_mapping,
        )

    def __str__(self):
        """Print the operator as its dense matrix representation."""
        return self.matrix(dense=True).__str__()


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
            adjoint_mapping=operator.adjoint,
        )

    @property
    def diagonal_values(self):
        """
        Return the diagonal values.
        """
        return self._diagonal_values

    def function(self, f):
        """
        Returns a function, f, of the operator. It is assumed that the
        function is well-defined on the diagonal values.
        """
        diagonal_values = np.array([f(x) for x in self.diagonal_values])
        return DiagonalLinearOperator(self.domain, self.codomain, diagonal_values)

    @property
    def inverse(self):
        """
        return the inverse operator. Valid only if diagonal values
        are non-zero.
        """
        assert all(val != 0 for val in self.diagonal_values)
        return self.function(lambda x: 1 / x)

    @property
    def sqrt(self):
        """
        Returns the square root of the operator. Valid only if diagonal values
        are non-negative.
        """
        assert all(val >= 0 for val in self._diagonal_values)
        return self.function(np.sqrt)


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
