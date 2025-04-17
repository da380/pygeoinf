"""
This module defined the VectorSpace class along with a function
that returns n-dimensional real vector space with its standard
basis as an instance of this class.
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
from scipy.stats import norm, multivariate_normal
from scipy.sparse.linalg import LinearOperator as ScipyLinOp
from scipy.sparse.linalg import gmres, bicgstab, cg, bicg
from scipy.sparse import diags
import warnings


class VectorSpace:
    """
    A class for real vector spaces. To define an instance, the
    user needs to provide the following:

        (1) The dimension of the space, or the dimension of the
            finite-dimensional approximating space.
        (2) A mapping from elements of the space to their components.
            These components must be expressed as numpy arrays with
            shape (dim) with dim the spaces dimension.
        (3) A mapping from components back to the vectors. This
            needs to be the inverse of the mapping in (2), but
            this requirement is not automatically checked.

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
            add (callable): Implements vector addition.
            subtract (callable): Implements vector subtraction.
            multiply (callable): Implements scalar multiplication.
            axpy (callable): Implements the mapping y -> a*x + y
            copy (callable): Implements deep copy of a vector.
            base (VectorSpace | None): Set to none for an original space,
                and to the base space when forming the dual.
        """
        self._dim = dim
        self.__to_components = to_components
        self.__from_components = from_components
        self._base = base

        # Default vector operations.
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
        """The dual of the vector space."""
        if self._base is None:
            return VectorSpace(
                self.dim,
                self._dual_to_components,
                self._dual_from_components,
                base=self,
            )
        else:
            return self._base

    @property
    def zero(self):
        return self.from_components(np.zeros((self.dim)))

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

    def identity(self):
        """Returns identity operator on the space."""
        if isinstance(self, HilbertSpace):
            return LinearOperator(
                self,
                self,
                lambda x: x,
                dual_mapping=lambda yp: yp,
                adjoint_mapping=lambda y: y,
            )
        else:
            return LinearOperator(self, self, lambda x: x, dual_mapping=lambda yp: yp)

    def _dual_to_components(self, xp):
        if xp._matrix is None:
            matrix = xp.matrix(dense=True)
        else:
            matrix = xp._matrix
        return matrix.reshape(xp.domain.dim)

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


class VectorSpaceDirectSum(VectorSpace):
    """
    Class for the direct sum of vector spaces. To define an instance, the
    user provides a list of VectorSpace instances.
    """

    def __init__(self, spaces):
        """
        Args:
            spaces ([VectorSpace]) : A list of the vector spaces.
        """
        self._spaces = spaces
        dim = sum([space.dim for space in spaces])
        super().__init__(
            dim,
            self.__to_components,
            self.__from_components,
            add=self.__add,
            subtract=self.__subtract,
            multiply=self.__multiply,
            axpy=self.__axpy,
            copy=self.__copy,
        )

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

    def _number_of_spaces(self):
        # Return the total number of spaces.
        return len(self._spaces)

    def _projection_mapping(self, i, xs):
        # Implementation of the projection mapping onto ith space.
        return xs[i]

    def _dual_projection_mapping(self, i, ypi):
        # Implementation of the dual of the projection mapping onto ith space.
        return LinearForm(self, mapping=lambda xs: ypi(xs[i]))

    def _inclusion_mapping(self, i, x):
        # Implementation of the inclusion mapping from ith space.
        xs = []
        for j in range(0, i):
            xs.append(self._spaces[j].zero)
        xs.append(x)
        for j in range(i + 1, self._number_of_spaces()):
            xs.append(self._spaces[j].zero)
        return xs

    def _dual_inclusion_mapping(self, i, xp):
        # Implementation of dual of the inclusion mapping from the ith space.
        return LinearForm(
            self._spaces[i], mapping=lambda xi: xp(self._inclusion_mapping(i, xi))
        )

    def projection(self, i):
        """
        Returns the projection operator onto the ith space.
        """
        return LinearOperator(
            self,
            self._spaces[i],
            lambda xs: self._projection_mapping(i, xs),
            dual_mapping=lambda ypi: self._dual_projection_mapping(i, ypi),
        )

    def inclusion(self, i):
        """
        Returns the inclusion operator from the ith space.
        """
        return LinearOperator(
            self._spaces[i],
            self,
            lambda x: self._inclusion_mapping(i, x),
            dual_mapping=lambda xp: self._dual_inclusion_mapping(i, xp),
        )


class Operator:
    """
    Class for operators between two vector spaces.


    (1) The domain of the operator as an instance of VectorSpace.
        (2) The codomain of the operator as an instance of VectorSpace.

    To define the action of the operator they can provide either:

        (a) A functor that represents the action of the operator.
    """

    def __init__(self, domain, codomain, mapping):
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
    def hilbert_operator(self):
        """True is operator maps between Hilbert spaces."""
        return isinstance(self.domain, HilbertSpace) and isinstance(
            self.codomain, HilbertSpace
        )

    @property
    def automorphism(self):
        """True is operator maps a space into itself."""
        return self.domain == self.codomain

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
    Class for linear operators between two vector spaces. To define an
    instance, the user must provide the following:

        (1) The domain of the operator as an instance of VectorSpace.
        (2) The codomain of the operator as an instance of VectorSpace.

    To define the action of the operator they can provide either:

        (a) A functor that represents the action of the operator.

    For option (a) the dual and adjoint can be deduced internally
    but in an inefficient manner, or they can be supplied directly.

    Linear operators form an algebra over the reals in the usual way. Overloads
    for the relevant operators are provided. In all cases, these operations are
    lazily implemented.
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
            domain (VectorSpace): The domain of the operator.
            codomain (VectorSpace): The codomain of the operator.
            mapping (callable | None): A functor that implements the
                action of the operator.
            dual_mapping (callable | None): A functor that implements
                the action of the dual operator.
            adjoint_mapping (callable | None): A functor that implements
                the action of the adjoint operator.
        """
        super().__init__(domain, codomain, mapping)
        self._dual_base = dual_base
        self._adjoint_base = adjoint_base
        self._matrix = None
        if dual_mapping is None:
            if self.hilbert_operator:
                if adjoint_mapping is None:
                    self.__dual_mapping = self._dual_mapping_default
                    self.__adjoint_mapping = self._adjoint_mapping_from_dual
                else:
                    self.__adjoint_mapping = adjoint_mapping
                    self.__dual_mapping = self._dual_mapping_from_adjoint
            else:
                self.__dual_mapping = self._dual_mapping_default
                self.__adjoint_mapping = None
        else:
            self.__dual_mapping = dual_mapping
            if self.hilbert_operator:
                if adjoint_mapping is None:
                    self.__adjoint_mapping = self._adjoint_mapping_from_dual
                else:
                    self.__adjoint_mapping = adjoint_mapping
            else:
                self.__adjoint_mapping = None

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
    def from_matrix(matrix, domain, codomain, galerkin=False):
        """
        Returns a linear operator defined by its matrix representation.
        By default the standard representation is returned. But the
        Galerkin representation can be returned optionally so long as the
        domain and codomain are Hilbert spaces.

        Args:
            matrix (MatrixLike): The matrix representation of the operator.
            domain (VectorSpace): The domain of the operator.
            codomain (VectorSpace): The codomain of the operator.
            galerkin (bool): True if the Galkerin represention is used.

        Returns:
            LinearOperator: The linear operator.
        """
        assert matrix.shape == (codomain.dim, domain.dim)

        if galerkin:
            assert isinstance(domain, HilbertSpace) and isinstance(
                codomain, HilbertSpace
            )

            def mapping(x):
                cx = domain.to_components(x)
                cyp = matrix @ cx
                yp = codomain.dual.from_components(cyp)
                return codomain.from_dual(yp)

            def dual_mapping(yp):
                y = codomain.from_dual(yp)
                cyp = codomain.to_components(yp)
                cxp = matrix.T @ cyp
                return domain.dual.from_components(cxp)

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
    def from_svd_factors(domain, codomain, u, sigma, vt):
        """
        Returns a linear operator defined from low-rank SVD factors for
        it standard matrix representation.

        Args:
            domain (VectorSpace): The domain of the operator.
            codomain (VectorSpace): The codomain of the operator.
            u (np.ndarray): The left singular vectors.
            sigma (np.ndarray): The singular values.
            vt (np.ndarray): The right singular vectors.

        Returns:
            LinearOperator: The linear operator.
        """
        assert u.shape[0] == codomain.dim
        assert u.shape[1] == len(sigma)
        assert vt.shape[0] == len(sigma)
        assert vt.shape[1] == domain.dim

        # Form diagonal operator from singular values.
        s = diags([sigma], [0])

        def mapping(x):
            cx = domain.to_components(x)
            cy = u @ (s @ (vt @ cx))
            return codomain.from_components(cy)

        def dual_mapping(yp):
            cyp = codomain.dual.to_components(yp)
            cxp = vt.T @ (s @ (u.T @ cyp))
            return domain.dual.from_components(cxp)

        return LinearOperator(domain, codomain, mapping, dual_mapping=dual_mapping)

    @staticmethod
    def self_adjoint_from_eigen_factors(domain, u, lambdas, inverse=False):
        """
        Returns a self-adjoint operator defined by low rank eigen-factors
        for its Galkerin representation. The approximate inverse of the
        operator can, optionally, be returned.

        Args:
            domain (VectorSpace): The domain of the operator.
            u (np.ndarray): The matrix of eigenvectors stored columwise.
            lambdas (np.ndarray): The eigenvalues.
            inverse (bool): If true, the inverse operator is returned.

        Returns:
            LinearOperator: The linear operator.
        """
        assert isinstance(domain, HilbertSpace)
        assert u.shape[0] == domain.dim
        assert u.shape[1] == len(lambdas)
        d = (
            diags([lambdas], [0])
            if not inverse
            else diags([np.reciprocal(lambdas)], [0])
        )

        def mapping(x):
            cx = domain.to_components(x)
            cyp = u @ (d @ (u.T @ cx))
            yp = domain.dual.from_component(cyp)
            return domain.from_dual(yp)

        return LinearOperator.self_adjoint(domain, mapping)

    @staticmethod
    def self_adjoint_from_cholesky_factor(domain, F):
        """
        Returns a self-adjoint linear operator on a Hilbert space
        from its low-rank Cholesky-factors.

        Args:
            domain (VectorSpace): The domain of the operator.
            F (np.ndarray or SciPy LinOp): The Cholesky factor.

        Returns:
            LinearOperator: The linear operator F @ F.T.
        """
        assert isinstance(domain, HilbertSpace)
        assert F.shape[0] == domain.dim

        def mapping(x):
            cx = domain.to_components(x)
            cyp = F @ (F.T @ cx)
            yp = domain.dual.from_components(cyp)
            return domain.from_dual(yp)

        return LinearOperator(domain, domain, mapping)

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
        if not self.hilbert_operator:
            raise NotImplementedError("Adjoint not defined for the operator.")
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
            if galerkin:
                if not self.hilbert_operator:
                    raise NotImplementedError(
                        "Defined only for operators between Hilbert spaces."
                    )

                def matvec(cx):
                    x = self.domain.from_components(cx)
                    y = self(x)
                    yp = self.codomain.to_dual(y)
                    return self.codomain.to_components(yp)

                def rmatvec(cy):
                    y = self.codomain.from_components(cy)
                    x = self.adjoint(y)
                    xp = seld.to_dual(x)
                    return self.domain.dual_to_components(x)

            else:

                def matvec(cx):
                    x = self.domain.from_components(cx)
                    y = self(x)
                    return self.codomain.to_components(y)

                def rmatvec(cyp):
                    yp = self.codomain.dual.from_components(cyp)
                    xp = self.dual(yp)
                    return self.domain.dual.to_components(xp)

            return ScipyLinOp(
                (self.codomain.dim, self.domain.dim), matvec=matvec, rmatvec=rmatvec
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
        for i in range(self.domain.dim):  # todo: parellise?
            cx[i] = 1
            matrix[:, i] = (a @ cx)[:]
            cx[i] = 0
        return matrix

    def reduced_rank_basis(self, rank, /, *, galerkin=True, pow=0, reorth=False):
        """
        Given an operator and rank, computes an approximate basis for the range.

        Uses power iteration when q > 0 to enhance the approximation.

        Args:
            rank (int): The target rank.
            galerkin (bool): Whether to use the Galerkin matrix representation.
            q (int): The power iteration exponent (default 0, meaning no power iteration).

        Returns:
            np.ndarray: An orthonormal basis for the range.
        """
        m = self.domain.dim
        n = self.codomain.dim

        a = self.matrix(galerkin=galerkin)
        p = np.random.randn(n, rank)

        y = np.zeros((m, rank))
        for i in range(rank):  # todo: parellise?
            y[:, i] = a @ p[:, i]

        for _ in range(pow):
            y = a @ (a.T @ y)
            if reorth:
                y, _ = np.linalg.qr(y)  # todo: test this

        if not reorth:
            q, _ = np.linalg.qr(y)
        else:
            q = y

        return q

    def reduced_rank_svd(self, rank, /, *, pow=0, galerkin=False):
        """
        Computes an approximate low-rank SVD of a general (non-square) operator.

        Args:
            rank (int): The target rank.
            galerkin (bool): Whether to use the Galerkin matrix representation.
            q (int): The power iteration exponent.

        Returns:
            U (np.ndarray): Left singular vectors.
            Sigma (np.ndarray): Singular values.
            Vt (np.ndarray): Right singular vectors (transposed).
        """
        a = self.matrix(galerkin=galerkin)

        q = self.reduced_rank_basis(rank, galerkin=galerkin, pow=pow)
        b = q.T @ a

        u_tilde, sigma, vt = svd(b, full_matrices=False)
        u = q @ u_tilde
        return u, sigma, vt

    def reduced_rank_eigen_factors(self, rank, /, *, pow=0):
        """
        Given a rank, computes an approximate eigenvalue decomposition of a self-adjoint operator.

        Args:
            rank (int): The rank of the decomposition.
            q (int): The power iteration exponent.

        Returns:
            np.ndarray: The eigen factors.
            np.ndarray: The eigenvalues.
        """
        assert self.automorphism

        m = self.domain.dim
        if rank > m:
            rank = m
            warnings.warn(
                "Rank exceeds dimension of operator. Rank set to dimension of operator.",
                UserWarning,
            )

        a = self.matrix(galerkin=True)
        q = self.reduced_rank_basis(rank, galerkin=True, pow=pow)

        # Eigenvalue decomposition
        op = LinearOperator.self_adjoint(
            EuclideanSpace(rank), lambda x: q.T @ (a @ (q @ x))
        )
        b = op.matrix(dense=True)

        # Form U and the approx operator
        lambdas, v = eigh(b)
        u = q @ v

        return v, lambdas

    def reduced_rank_cholesky_factor(self, rank, /, *, pow=0):
        """
        Given a symmetric, positive-definite operator and rank, computes an approximate Cholesky factorisation.

        Args:
            rank (int): The rank of the factorisation.

        Returns:
            np.ndarray: The Cholesky factor as a matrix.
        """
        assert self.automorphism

        m = self.domain.dim
        if rank > m:
            rank = m
            warnings.warn(
                "Rank exceeds dimension of operator. Rank set to dimension of operator.",
                UserWarning,
            )

        q = self.reduced_rank_basis(rank, galerkin=True, pow=pow)
        a = self.matrix(galerkin=True)

        b1 = np.zeros((m, rank))
        for i in range(rank):  # todo: parellise?
            b1[:, i] = a @ q[:, i]
        b2 = q.T @ b1

        # Perform a cholesky factorisation of b2
        c, _ = cho_factor(b2)

        c_inv = solve_triangular(c, np.eye(rank), lower=True)
        f = b1 @ c_inv

        return f

    def low_rank_approximation(self, rank, /, *, method="svd", pow=0):
        """
        Given an operator and rank, computes an approximate low-rank operator.

        Args:
            rank (int): The rank of the approximation.
            method (str): The method to use. Options are 'cholesky', 'eigen', 'nystrom'.
            q (int): The power iteration exponent.

        Returns:
            LinearOperator: The low-rank operator.
        """

        if method == "svd":
            return self.low_rank_approximation_by_svd(rank, pow=pow)
        elif method == "cholesky":
            return self.low_rank_approximation_by_cholesky(rank, pow=pow)
        elif method == "eigen":
            return self.low_rank_approximation_by_eigen(rank, pow=pow)
        elif method == "nystrom":
            return self.low_rank_approximation_by_nystrom(rank, pow=pow)
        else:
            raise ValueError("Invalid method.")

    def low_rank_approximation_by_svd(self, rank, /, *, pow=0):
        """
        Given an operator and rank, computes an approximate low-rank operator using the SVD.

        Args:
            rank (int): The rank of the approximation.
            q (int): The power iteration exponent.

        Returns:
            LinearOperator: The low-rank operator.
        """
        # todo: optional inverse for if the operator is self-adjoint?
        u, sigma, vt = self.reduced_rank_svd(rank, pow=pow)
        return LinearOperator.from_svd_factors(self.domain, self.codomain, u, sigma, vt)

    def low_rank_approximation_by_cholesky(self, rank, /, *, pow=0):
        """
        Given a symmetric, positive-definite operator and rank, computes an approximate low-rank operator using Cholesky factorisation.

        Args:
            rank (int): The rank of the approximation.

        Returns:
            LinearOperator: The low-rank operator.
        """
        f = self.reduced_rank_cholesky_factor(rank, pow=pow)
        return LinearOperator.self_adjoint_from_cholesky_factor(self.domain, f)

    def low_rank_approximation_by_eigen(self, rank, /, *, inverse=False, pow=0):
        """
        Given a self-adjoint operator and rank, computes a low rank eigenvalue decomposition and returns as an operator.

        Args:
            rank (int): The rank of the decomposition.
            inverse (bool): If true, the inverse operator is returned.

        Returns:
            LinearOperator: The low-rank operator.
        """
        u, lambdas = self.reduced_rank_eigen_factors(rank, pow=pow)
        return LinearOperator.self_adjoint_from_eigen_factors(
            self.domain, u, lambdas, inverse=inverse
        )

    def low_rank_approximation_by_nystrom(self, rank, /, *, inverse=False, pow=0):
        """
        Given an operator and rank, computes an approximate low-rank operator using the Nyström method.

        Args:
            rank (int): The rank of the approximation.
            inverse (bool): If true, an approximate inverse operator is returned.

        Returns:
            LinearOperator: The low-rank operator.
        """

        f = self.reduced_rank_cholesky_factor(rank, pow=pow)
        return LinearOperator.self_adjoint_from_cholesky_factor(self.domain, f)

    def __neg__(self):
        """Negative of the operator."""
        domain = self.domain
        codomain = self.codomain

        def mapping(x):
            return codomain.negative(x)

        def dual_mapping(yp):
            return domain.dual.negative(self.dual(yp))

        if self.hilbert_operator:

            def adjoint_mapping(y):
                return domain.negative(self.adjoint(y))

        else:
            adjoint_mapping = None

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

        if self.hilbert_operator:

            def adjoint_mapping(y):
                return domain.multiply(a, self.adjoint(y))

        else:
            adjoint_mapping = None

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

        if self.hilbert_operator:

            def adjoint_mapping(y):
                return domain.add(self.adjoint(y), other.adjoint(y))

        else:
            adjoint_mapping = None

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

        if self.hilbert_operator:

            def adjoint_mapping(y):
                return domain.subtract(self.adjoint(y), other.adjoint(y))

        else:
            adjoint_mapping = None

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

        if self.hilbert_operator:

            def adjoint_mapping(y):
                return other.adjoint(self.adjoint(y))

        else:
            adjoint_mapping = None

        return LinearOperator(
            domain,
            codomain,
            mapping,
            dual_mapping=dual_mapping,
            adjoint_mapping=adjoint_mapping,
        )

    def __str__(self):
        """Print the operator as its dense matrix representation."""
        if self._matrix is None:
            return self.matrix(dense=True).__str__()
        else:
            return self._matrix.__str__()


# Global definition of the real numbers as a VectorSpace.
_REAL = VectorSpace(1, lambda x: np.array([x]), lambda c: c[0])


class LinearForm(LinearOperator):
    """
    Class for linear forms on a vector space.

    The vector space is represented by a VectorSpace object. Linear forms
    are (continuous) linear mappings from the to the real numbers. These
    mappings form a vector space known that is *dual* to the original.

    A linear form can be specified either:

        (1) In terms of a functor that performs its action on a vector.
        (2) Its matrix-representation relative to the basis for the space.

    Specification of a form in terms of its matrix representation offers
    computational advatages in many cases. A form specified in terms
    of a mapping can (as with a general linear operator) compute and
    store its matrix representation.
    """

    def __init__(self, domain, /, *, mapping=None, components=None):
        """
        Args:
            domain (VectorSpace): Domain of the linear form.
            mapping (callable | None): A functor that performs the action
                of the linear form on a vector.
            matrix (MatrixLike | None): The matrix representation of the
                form, this having shape (1,dim) with dim the dimension of
                the domain.
        """

        if components is None:
            super().__init__(domain, _REAL, mapping)
            self._matrix = None
        else:
            super().__init__(domain, _REAL, self._mapping_from_components)
            self._matrix = components.reshape(1, self.domain.dim)

    def _mapping_from_components(self, x):
        # Implement the action of the form using its components.
        return self.codomain.from_components(
            np.dot(self._matrix, self.domain.to_components(x))
        )

    @property
    def components_stored(self):
        """True is the form has its components stored."""
        return self._matrix is not None

    @property
    def components(self):
        """Return the components of the form."""
        if self.components_stored:
            return self._matrix.reshape(self.domain.dim)
        else:
            self.store_components()
            return self.components

    def store_components(self):
        """Compute and store the forms components."""
        if not self.components_stored:
            self._matrix = self.matrix(dense=True)


class HilbertSpace(VectorSpace):
    """
    Class for Hilbert spaces. To define an instance, the user provides:

        (1) The dimension of the space, or the dimension of the
            finite-dimensional approximating space.
        (2) A mapping from elements of the space to their components.
            These components must be expressed as numpy arrays with
            shape (dim,1) with dim the spaces dimension.
        (3) A mapping from components back to the vectors. This
            needs to be the inverse of the mapping in (2), but
            this requirement is not automatically checked.
        (4) The inner product on the space.

    Note that this space inherits from VectorSpace.

    The user can also provide either of the following:

        (a) The mapping from the space to its dual.
        (b) The mapping from a dual vector to its representation
            within the space.

    The existence of such mappings follows from the Riesz representation
    theorem. If (a) is not provided, then then cannonical form of this
    mapping is used, with a vector, x, being mapped to the dual vector
    that acts on a vector, y, by y -> (x,y).

    If (b) is not provided, the metric tensor for the space relative
    to its basis is computed along with its Cholesky factor. In this
    case the mapping in (a) is replaced by one using dual components
    that is more efficient.

    Within high-dimensional spaces, custom mappings to and from the
    dual space should be provided for the user for better efficiency.

    A HilbertSpace can be constructed directly, or from an already
    formed VectorSpace using the static method "from_vector_space".
    The latter option is often preferable as a custom definition of
    the functor "to_dual" requires that the dual space be available.
    """

    def __init__(
        self,
        dim,
        to_components,
        from_components,
        inner_product,
        /,
        *,
        to_dual=None,
        from_dual=None,
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
            base (VectorSpace | None): Set to none for an original space,
                and to the base space when forming the dual.
        """
        super().__init__(dim, to_components, from_components)
        self.__inner_product = inner_product

        if from_dual is None:
            self.__metric_tensor = self.calculate_metric_tensor()
            self.__metric_tensor_factor = cho_factor(self.metric_tensor)
            self.__from_dual = self._from_dual_default
        else:
            self.__metric_tensor = None
            self.__from_dual = from_dual

        if to_dual is None:
            if self.__metric_tensor is None:
                self.__to_dual = self._to_dual_default
            else:
                self.__to_dual = self._to_dual_default_with_metric
        else:
            self.__to_dual = to_dual

        self._base = base

    @staticmethod
    def from_vector_space(space, inner_product, /, *, to_dual=None, from_dual=None):
        """
        Form a HilbertSpace from a VectorSpace by providing additional structures.

        Args:
            space (VectorSpace): The vector space to form as base for the Hilbert space.
            inner_product (callable): A functor the implements the inner
                product on the space.
            to_dual (callable | None): A funcator that maps a vector
                to the cannonically associated dual vector.
            from_dual (callable | None): A functor that maps a dual vector
                to its representation on the space.
        """
        return HilbertSpace(
            space.dim,
            space.to_components,
            space.from_components,
            inner_product,
            to_dual=to_dual,
            from_dual=from_dual,
        )

    @property
    def vector_space(self):
        """The underlying vector space."""
        return VectorSpace(self.dim, self.to_components, self.from_components)

    @property
    def dual(self):
        """The dual of the Hilbert space."""
        if self._base is None:
            return HilbertSpace(
                self.dim,
                self._dual_to_components,
                self._dual_from_components,
                self._dual_inner_product,
                to_dual=self.from_dual,
                from_dual=self.to_dual,
                base=self,
            )
        else:
            return self._base

    @property
    def metric_tensor(self):
        """The metric tensor for the space."""
        if self.__metric_tensor is None:
            return self.calculate_metric_tensor()
        else:
            return self.__metric_tensor

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

    def calculate_metric_tensor(self):
        """Return the space's metric tensor as a numpy matrix."""
        metric_tensor = np.zeros((self.dim, self.dim))
        c1 = np.zeros(self.dim)
        c2 = np.zeros(self.dim)
        for i in range(self.dim):
            c1[i] = 1
            x1 = self.from_components(c1)
            metric_tensor[i, i] = self.inner_product(x1, x1)
            for j in range(i + 1, self.dim):
                c2[j] = 1
                x2 = self.from_components(c2)
                metric_tensor[i, j] = self.inner_product(x1, x2)
                metric_tensor[j, i] = metric_tensor[i, j]
                c2[j] = 0
            c1[i] = 0
        return metric_tensor

    def _to_dual_default(self, x):
        return LinearForm(self, mapping=lambda y: self.inner_product(x, y))

    def _to_dual_default_with_metric(self, x):
        cp = self.metric_tensor @ self.to_components(x)
        return self.dual.from_components(cp)

    def _from_dual_default(self, xp):
        cp = self.dual.to_components(xp)
        c = cho_solve(self.__metric_tensor_factor, cp)
        return self.from_components(c)

    def _dual_inner_product(self, xp1, xp2):
        return self.inner_product(self.from_dual(xp1), self.from_dual(xp2))


class EuclideanSpace(HilbertSpace):
    """
    Euclidean space implemented as an instance of HilbertSpace."""

    def __init__(self, dim, /, *, metric_tensor=None, inverse_metric_tensor=None):
        """
        Args:
            dim (int): Dimension of the space.
            metric_tensor (scipy LinearOperator): The metric tensor in the
                form of a scipy LinearOperator or an equivalent object.
            inverse_metric_tensor (scipy LinearOperator): The inverse metric tensor
                in the form of a scipy LinearOperator or an equivalent object.

        Notes:
            If the inverse metric tensor is not provided, it action is
            computed using the conjugate gradient algorithm.
        """

        if metric_tensor is None:
            super().__init__(
                dim,
                self._identity,
                self._identity,
                self._inner_product_without_metric,
                to_dual=self._to_dual_without_metric,
                from_dual=self._from_dual_without_metric,
            )
        else:
            self._metric_tensor = metric_tensor
            if inverse_metric_tensor is None:
                self._inverse_metric_tensor = self._inverse_metric_tensor_default
            else:
                self._inverse_metric_tensor = inverse_metric_tensor

            super().__init__(
                dim,
                self._identity,
                self._identity,
                self._inner_product_with_metric,
                to_dual=self._to_dual_with_metric,
                from_dual=self._from_dual_with_metric,
            )

    def standard_gaussisan_measure(self, standard_deviation):
        """
        Returns a Gaussian measure on the space with covariance proportional
        to the identity operator and with zero expectation.

        Args:
            standard_deviation (float): The standard deviation for each component.
        """
        factor = standard_deviation * self.identity()
        inverse_factor = self.identity() / standard_deviation
        return GaussianMeasure.from_factored_covariance(
            factor, inverse_factor=inverse_factor
        )

    def diagonal_covariance(self, standard_deviations):
        """
        Returns a Gaussian measure on the space with a diagonal
        covariance and with zero expectation.

        Args:
            standard_deviations (vector): Vector of the standard deviations.
        """
        matrix = diags([standard_deviations], [0])
        inverse_matrix = diags([standard_deviations.reciprocal()], [0])
        factor = LinearOperator.self_adjoint(self, lambda x: matrix @ x)
        inverse_factor = LinearOperator.self_adjoint(self, lambda x: inverse_matrix @ x)
        return GaussianMeasure.from_factored_covariance(
            factor, inverse_factor=inverse_factor
        )

    def _identity(self, x):
        return x

    def _inner_product_without_metric(self, x1, x2):
        return np.dot(x1, x2)

    def _inner_product_with_metric(self, x1, x2):
        return np.dot(self._metric_tensor @ x1, x2)

    def _to_dual_without_metric(self, x):
        return self.dual.from_components(x)

    def _to_dual_with_metric(self, x):
        return self.dual.from_components(self._metric_tensor @ x)

    def _from_dual_without_metric(self, xp):
        cp = self.dual.to_components(xp)
        return self.from_components(cp)

    def _from_dual_with_metric(self, xp):
        cp = self.dual.to_components(xp)
        c = self._inverse_metric_tensor @ cp
        return self.from_components(c)

    def _inverse_metric_tensor_default(self, x):
        return cg(self._metric_tensor, x)


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
        inverse_covariance=None,
        expectation=None,
        sample=None,
    ):
        """
        Args:
            domain (HilbertSpace): The Hilbert space on which the measure is defined.
            covariance (callable): A functor representing the covariance operator.
            inverse_covariance (callable): A functor representing the action of the inverse covariance operator.
            expectation (Vector): The expectation value of the measure. If none is provided, set equal to zero.
            sample (callable): A functor that returns a random sample from the measure.
        """
        self._domain = domain
        self._covariance = covariance
        self._inverse_covariance = inverse_covariance
        self._inverse_covariance_set = self._inverse_covariance is not None
        self._dist = None

        if expectation is None:
            self._expectation = self.domain.zero
        else:
            self._expectation = expectation

        self._sample = sample

        self._solver = CGSolver(rtol=1.0e-8)
        self._preconditioner = None

    @staticmethod
    def from_factored_covariance(factor, /, *, inverse_factor=None, expectation=None):
        """
        For a Gaussian measure hows covariance, C, is approximated in the form C = LL*,
        with L a mapping into the domain from Euclidean space.

        Args:
            factor (LinearOperator): Linear operator from Euclidean space into the
                domain of the measure.
            inverse_factor (LinearOperator): Inverse of the factor operator. Default is None.
            expectation (vector): expected value of the measure. Default is zero.

        Returns:
            GassianMeasure: The measure with the required covariance and expectation.
        """

        def sample():
            value = factor(
                norm().rvs(size=factor.domain.dim)
            )  # todo: it's already fine?
            if expectation is None:
                return value
            else:
                return value + expectation

        covariance = factor @ factor.adjoint

        if inverse_factor is None:
            inverse_covariance = None
        else:
            inverse_covariance = inverse_factor.adjoint @ inverse_factor

        return GaussianMeasure(
            factor.codomain,
            covariance,
            inverse_covariance=inverse_covariance,
            expectation=expectation,
            sample=sample,
        )

    @staticmethod
    def low_rank_measure_by_factored_covariance(measure, rank):
        """
        Returns a low-rank approximation to a Gaussian measure by forming a
        low-rank approximation of the covariance.

        Args:
            measure (GaussianMeasure): The original measure.
            rank (int): The rank of the approximation.

        Returns:
            GaussianMeasure: The low-rank approximation.
        """
        F = measure.covariance.reduced_rank_cholesky_factor(rank)
        euclidean_space = EuclideanSpace(rank)
        F_operator = LinearOperator.from_matrix(
            F, euclidean_space, measure.covariance.domain
        )
        return GaussianMeasure.from_factored_covariance(
            F_operator, expectation=measure.expectation
        )

    @property
    def domain(self):
        """The Hilbert space the measure is defined on."""
        return self._domain

    @property
    def covariance(self):
        """The covariance operator as an instance of LinearOperator."""
        return LinearOperator.self_adjoint(self.domain, self._covariance)

    @property
    def inverse_covariance(self):
        """The inverse covariance operator as an instance of LinearOperator."""
        if self._inverse_covariance_set:
            mapping = self._inverse_covariance
        else:
            if isinstance(self._solver, IterativeLinearSolver):
                mapping = self._solver(
                    self.covariance, preconditioner=self._preconditioner
                )
            else:
                mapping = self._solver(self.covariance)
        return LinearOperator.self_adjoint(self.domain, mapping)

    @property
    def expectation(self):
        """The expectation of the measure."""
        return self._expectation

    @property
    def cameron_martin_space(self):
        """
        Returns the associated Cameron-Martin space as a HilbertSpace instance.

        Args:
            inverse_covariance (LinearOperator): The inverse covariance operator. If this
                is not provided, it is computed using the provided solver.
            solver (LinearSolver): The linear solver used to invert the covariance. The
                default is to use the conjugate gradient algorithm.
            preconditioner (LinearOperator): A preconditioner for the inversion of the
                covariance. Only used if the solver method is an iterative one.

        Returns:
            HilbertSpace: The Cameron-Martin space.
        """

        def inner_product(x1, x2):
            return self.domain.inner_product(self.inverse_covariance(x1), x2)

        def to_dual(x):
            return self.domain.to_dual(self.inverse_covariance(x))

        def from_dual(xp):
            return self.covariance(self.domain.from_dual(xp))

        return HilbertSpace.from_vector_space(
            self.domain, inner_product, to_dual=to_dual, from_dual=from_dual
        )

    def sample_using_dense_matrix(self):
        """
        If sampling method is not set, use scipy algorithm based on
        dense matrix representation.
        """
        if self._sample is None:
            mean = self.domain.to_components(self.expectation)
            cov = self.covariance.matrix(dense=True, galerkin=True)
            cov = 0.5 * (cov + cov.T)
            self._dist = multivariate_normal(mean=mean, cov=cov)
        self._sample = lambda: self.domain.from_dual(
            self.domain.dual.from_components(self._dist.rvs())
        )

    def sample(self):
        """Returns a random sample drawn from the measure."""
        if self._sample is None:
            raise NotImplementedError("Sample method is not set.")
        else:
            return self._sample()

    def set_solver(self, solver):
        """Set the linear solver to be used in computing the inverse covariance."""

        self._solver = solver

    def set_preconditioner(self, preconditioner):
        """Set the preconditioner to be used in computing the inverse covariance."""
        self._preconditioner = preconditioner

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
            _operator = self.domain.identity()
        else:
            _operator = operator

        if translation is None:
            _translation = _operator.codomain.zero
        else:
            _translation = translation

        covariance = _operator @ self.covariance @ _operator.adjoint
        expectation = _operator(self.expectation) + _translation

        def sample():
            return _operator(self.sample()) + _translation

        return GaussianMeasure(
            _operator.codomain, covariance, expectation=expectation, sample=sample
        )

    def __neg__(self):
        """Negative of the measure."""

        return GaussianMeasure(
            self.domain,
            -self.covariance,
            inverse_covariance=-self.inverse_covariance,
            expectation=-self.expectation,
            sample=lambda: -self.sample(),
        )

    def __mul__(self, alpha):
        """Multiply the measure by a scalar."""
        return GaussianMeasure(
            self.domain,
            alpha * alpha * self.covariance,
            inverse_covariance=self.inverse_covariance / (alpha * alpha),
            expectation=alpha * self.expectation,
            sample=lambda: alpha * self.sample(),
        )

    def __rmul__(self, alpha):
        """Multiply the measure by a scalar."""
        return self * alpha

    def __add__(self, other):
        """Add two measures on the same domain."""
        return GaussianMeasure(
            self.domain,
            self.covariance + other.covariance,
            inverse_covariance=self.inverse_covariance + other.inverse_covariance,
            expectation=self.expectation + other.expectation,
            sample=lambda: self.sample() + other.sample(),
        )

    def __sub__(self, other):
        """Subtract two measures on the same domain."""
        return GaussianMeasure(
            self.domain,
            self.covariance + other.covariance,
            inverse_covariance=self.inverse_covariance + other.inverse_covariance,
            expectation=self.expectation - other.expectation,
            sample=lambda: self.sample() - other.sample(),
        )


class LinearSolver(ABC):
    """
    Abstract base class for linear solvers.
    """


def _inverse_operator_from_matrix_solver(operator, matrix_solver, galerkin):
    # Returns the inverse mappings relative to the matrix representation.

    domain = operator.domain
    codomain = operator.codomain

    if galerkin:

        def mapping(y):
            yp = codomain.to_dual(y)
            cyp = codomain.dual.to_components(yp)
            cx = matrix_solver(cyp, 0)
            return domain.from_components(cx)

        def dual_mapping(xp):
            cxp = domain.dual.to_components(xp)
            cy = matrix_solver(cxp, trans=1)
            y = codomain.from_components(cy, 0)
            return codomain.to_dual(y)

        if operator.hilbert_operator:

            def adjoint_mapping(x):
                xp = domain.to_dual(x)
                cxp = domain.dual.to_components(xp)
                cy = matrix_solver(cxp, 1)
                return codomain.from_components(cy)

    else:

        def mapping(y):
            cy = codomain.to_components(y)
            cx = matrix_solver(cy, 0)
            return domain.from_components(cx)

        def dual_mapping(xp):
            cxp = domain.dual.to_components(xp)
            cyp = matrix_solver(cxp, 1)
            return codomain.dual.from_components(cyp)

        if operator.hilbert_operator:

            def adjoint_mapping(x):
                xp = domain.to_dual(x)
                cxp = domain.dual.to_components(xp)
                cyp = matrix_solver(cxp, 1)
                yp = codomain.dual.from_components(cyp)
                return codomain.from_dual(yp)

    if operator.hilbert_operator:
        return LinearOperator(
            codomain,
            domain,
            mapping,
            dual_mapping=dual_mapping,
            adjoint_mapping=adjoint_mapping,
        )
    else:
        return LinearOperator(codomain, domain, mapping, dual_mapping=dual_mapping)


class DirectLinearSolver(LinearSolver):
    """
    Abstract base class for direct linear solvers.
    """

    def __init__(self, galerkin):
        self._galerkin = galerkin

    @property
    def galerkin(self):
        """True if the Galerkin matrix representation is used."""
        return self._galerkin

    @abstractmethod
    def _matrix_solver(self, matrix):
        """
        Returns the inverse operator's action in terms of its matrix representation.

        Args:
            matrix (Scipy matrix): Matrix representation of the operator.

        Returns:
            A callable object that takes arguments (cy, trans). If trans = 0,
            this function acts the inverse operators matrix representation
            on the components, cy. If trans=1, the action of the transpose
            of the inverse's matrix representation is provided.
        """

    def __call__(self, operator):
        """
        Given a linear operator, return its inverse.

        Args:
            operator (LineraOperator): The operator to be inverted.

        Returns:
            LinearOperator: A linear operator that implements the inverse
                of the input operator.
        """

        if self.galerkin and not operator.hilbert_operator:
            raise ValueError("Galerkin matrix not defined for non-Hilbert operators!")

        matrix = operator.matrix(dense=True, galerkin=self.galerkin)
        return _inverse_operator_from_matrix_solver(
            operator, self._matrix_solver(matrix), self.galerkin
        )


class LUSolver(DirectLinearSolver):
    """
    Direct Linear solver class based on LU decomposition of the
    matrix representation.
    """

    def __init__(self, /, *, galerkin=False):
        """
        Args:
            galerkin (bool): If true, the Galerkin matrix representation is used,
                otherwise the standard representation applied.
        """
        super().__init__(galerkin)

    def _matrix_solver(self, matrix):
        factor = lu_factor(matrix)
        return lambda y, trans: lu_solve(factor, y, trans)


class CholeskySolver(DirectLinearSolver):
    """
    Direct Linear solver class based on Cholesky decomposition of the
    matrix representation. This method is applicable only if the
    chosen matrix representation is symmetric and positive definite.
    """

    def __init__(self, /, *, galerkin=False):
        """
        Args:
            galerkin (bool): If true, the Galerkin matrix representation is used,
                otherwise the standard representation applied.
        """
        super().__init__(galerkin)

    def _matrix_solver(self, matrix):
        factor = cho_factor(matrix)
        return lambda y, trans: cho_solve(factor, y)


class IterativeLinearSolver(LinearSolver):
    """Base class for iterative linear solvers."""

    @abstractmethod
    def __call__(self, operator, /, *, preconditioner=None, x0=None):
        """
        Given a linear operator, return its inverse.

        Args:
            operator (LineraOperator): The operator to be inverted.
            preconditioner (LinearOperator): An optional operator for
                preconditioning the linear system.
            x0 (Vector): Optional initial guess for iterative solution,
                the default value is zero.

        Returns:
            LinearOperator: A linear operator that implements the inverse
                of the input operator.
        """


class IterativeMatrixLinearSolver(IterativeLinearSolver):
    """
    Abstract base class for iterative linear solvers based
    on matrix representations.
    """

    def __init__(self, galerkin):
        self._galerkin = galerkin

    @property
    def galerkin(self):
        """True if the Galerkin matrix representation is used."""
        return self._galerkin

    @abstractmethod
    def _matrix_solver(self, matrix, matrix_preconditioner, c0):
        """
        Returns the inverse operator's action in terms of its matrix representation.

        Args:
            matrix (Scipy LinearOperator): Matrix representation of the operator.
            matrix_preconditioner (Scipy LinearOperator): Matrix representation
                of the preconditioning operator.
            c0 (scipy vector): Initial guess in component form.

        Returns:
            A callable object that takes arguments (cy, trans). If trans = 0,
            this function acts the inverse operators matrix representation
            on the components, cy. If trans=1, the action of the transpose
            of the inverse's matrix representation is provided.
        """

    def __call__(self, operator, /, *, preconditioner=None, x0=None):
        """
        Given a linear operator, return its inverse.

        Args:
            operator (LineraOperator): The operator to be inverted.
            preconditioner (LinearOperator): An optional operator for
                preconditioning the linear system.
            x0 (Vector): Optional initial guess for iterative solution,
                the default value is zero.

        Returns:
            LinearOperator: A linear operator that implements the inverse
                of the input operator.
        """
        if self.galerkin and not operator.hilbert_operator:
            raise ValueError("Galerkin matrix not defined for non-Hilbert operators!")

        matrix = operator.matrix(galerkin=self.galerkin)
        if preconditioner is None:
            matrix_preconditioner = None
        else:
            matrix_preconditioner = preconditioner.matrix(galerkin=self.galerkin)

        if x0 is None:
            c0 = None
        else:
            c0 = operator.domain.to_components(x0)

        return _inverse_operator_from_matrix_solver(
            operator,
            self._matrix_solver(matrix, matrix_preconditioner, c0),
            self.galerkin,
        )


class CGMatrixSolver(IterativeMatrixLinearSolver):
    """Solver class using the conjugate gradient method on the matrix representation."""

    def __init__(
        self, /, *, galerkin=False, rtol=1.0e-5, atol=0, maxiter=None, callback=None
    ):
        """
        Args:
            galerkin (bool): True is the Galerkin matrix representation is used.
            rtol (float): relative tolerance within convergence checks.
            atol (float): absolute tolerance within convergence checks.
            maxiter (int): maximum number of iterations to allow.
            callback (callable): callable function after each iteration. This function
                takes in as argument the current solution vector.
        """
        super().__init__(galerkin)
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter
        self._callback = callback

    def _matrix_solver(self, matrix, matrix_preconditioner, c0):
        return lambda cy, trans: cg(
            matrix,
            cy,
            x0=c0,
            rtol=self._rtol,
            atol=self._atol,
            maxiter=self._maxiter,
            M=matrix_preconditioner,
            callback=self._callback,
        )[0]


class BICGMatrixSolver(IterativeMatrixLinearSolver):
    """LinearSolver class using the biconjugate gradient method on the matrix representation."""

    def __init__(
        self, /, *, galerkin=False, rtol=1.0e-5, atol=0, maxiter=None, callback=None
    ):
        """
        Args:
            galerkin (bool): True is the Galerkin matrix representation is used.
            rtol (float): relative tolerance within convergence checks.
            atol (float): absolute tolerance within convergence checks.
            maxiter (int): maximum number of iterations to allow.
            callback (callable): callable function after each iteration. This function
                takes in as argument the current solution vector.
        """
        super().__init__(galerkin)
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter
        self._callback = callback

    def _matrix_solver(self, matrix, matrix_preconditioner, c0):
        return lambda cy, trans: bicg(
            matrix,
            cy,
            x0=c0,
            rtol=self._rtol,
            atol=self._atol,
            maxiter=self._maxiter,
            M=matrix_preconditioner,
            callback=self._callback,
        )[0]


class BICGSTABMatrixSolver(IterativeMatrixLinearSolver):
    """LinearSolver class using the biconjugate gradient stablised method on the matrix representation."""

    def __init__(
        self, /, *, galerkin=False, rtol=1.0e-5, atol=0, maxiter=None, callback=None
    ):
        """
        Args:
            galerkin (bool): True is the Galerkin matrix representation is used.
            rtol (float): relative tolerance within convergence checks.
            atol (float): absolute tolerance within convergence checks.
            maxiter (int): maximum number of iterations to allow.
            callback (callable): callable function after each iteration. This function
                takes in as argument the current solution vector.
        """
        super().__init__(galerkin)
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter
        self._callback = callback

    def _matrix_solver(self, matrix, matrix_preconditioner, c0):
        return lambda cy, trans: bicgstab(
            matrix,
            cy,
            x0=c0,
            rtol=self._rtol,
            atol=self._atol,
            maxiter=self._maxiter,
            M=matrix_preconditioner,
            callback=self._callback,
        )[0]


class GMRESMatrixSolver(IterativeMatrixLinearSolver):
    """LinearSolver class using the GMRES method on the matrix representation."""

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
            galerkin (bool): True is the Galerkin matrix representation is used.
            rtol (float): relative tolerance within convergence checks.
            atol (float): absolute tolerance within convergence checks.
            restart (int): Number of iterations between restarts.
            maxiter (int): maximum number of iterations (restart cycles).
            callback (callable): callable function after each iteration. Signature
                of this function is determined by callback_type.
            callback_type ("x", "pr_norm", "legacy"): If "x" the current solution is
                passed to the callback function, if "pr_norm" it is the preconditioned
                residual norm. The default is "legacy" which means the same as "pr_norm",
                but changes the meaning of maxiter to count inner iterations instead of
                restart cycles.
        """
        super().__init__(galerkin)
        self._rtol = rtol
        self._atol = atol
        self._restart = restart
        self._maxiter = maxiter
        self._callback = callback
        self._callback_type = callback_type

    def _matrix_solver(self, matrix, matrix_preconditioner, c0):
        return lambda cy, trans: gmres(
            matrix,
            cy,
            x0=c0,
            rtol=self._rtol,
            atol=self._atol,
            restart=self._restart,
            maxiter=self._maxiter,
            M=matrix_preconditioner,
            callback=self._callback,
            callback_type=self._callback_type,
        )[0]


class CGSolver(IterativeLinearSolver):
    """
    LinearSolver class using the conjugate gradient algorithm within
    use of the matrix representation. Can be applied to self-adjoint
    operators on a general Hilbert space.
    """

    def __init__(self, rtol=1.0e-5, atol=0, maxiter=None, callback=None):
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

    def __call__(self, operator, /, *, preconditioner=None, x0=None):

        if operator.domain != operator.codomain:
            raise ValueError("Operator is not self-adjoint!")

        def mapping(y):

            if x0 is None:
                x = operator.domain.zero
            else:
                x = x0.copy()

            r = y - operator(x)
            if preconditioner is None:
                z = r.copy()
            else:
                z = preconditioner(r)
            p = z.copy()

            y_squared_norm = operator.domain.squared_norm(y)
            if y_squared_norm <= self._atol:
                return y

            tol = np.max([self._atol, self._rtol * y_squared_norm])

            if self._maxiter is None:
                maxiter = 10 * operator.domain.dim
            else:
                maxiter = self._maxiter

            for iteration in range(maxiter):

                if operator.domain.norm(r) <= tol:
                    break

                q = operator(p)
                num = operator.domain.inner_product(r, z)
                den = operator.domain.inner_product(p, q)
                alpha = num / den

                x += alpha * p
                r -= alpha * q

                if preconditioner is None:
                    z = r.copy()
                else:
                    z = preconditioner(r)

                den = num
                num = operator.domain.inner_product(r, z)
                beta = num / den

                p *= beta
                p += z

                if self._callback is not None:
                    self._callback(x)

            return x

        return LinearOperator.self_adjoint(operator.domain, mapping)


class PreconditioningMethod(ABC):
    """Base class for pre-conditioning methods."""

    @abstractmethod
    def __call__(self, operator):
        """
        Given an operator, constructs the associated preconditioner.
        """


class IdentityPreconditioner(PreconditioningMethod):
    """Class for use of the identity operator as a preconditioner."""

    def __call__(self, operator):
        assert operator.automorphism
        return operator.domain.identity()


class FixedRankRandomisedSVDPreconditioner(PreconditioningMethod):

    def __init__(self, rank):
        """
        Args:
            rank (int): The rank of the preconditioner.
        """
        self._rank = rank

    def __call__(self, operator):
        """
        Given an operator, constructs the associated preconditioner.

        Args:
            operator (LinearOperator): The operator to be preconditioned.
            rank (int): The rank of the preconditioner.

        Returns:
            LinearOperator: The preconditioner.
        """
        assert operator.automorphism
        Q = operator.reduced_rank_basis(self._rank)
        A = operator.matrix(galerkin=True)

        # Eigenvalue decomposition
        B = np.zeros((self._rank, self._rank))
        w = np.zeros(self._rank)
        z = np.zeros(self._rank)  # todo: delete this?
        for j in range(self._rank):
            w[j] = 1
            x = Q @ w
            y = A @ x
            z = Q.T @ y
            B[:, j] = z[:]
            w[j] = 0

        # Form U and the approx inverse
        lambdas, V = np.linalg.eig(B)
        U = Q @ V
        mapping = U @ np.diag(1 / lambdas) @ U.T

        return LinearOperator.self_adjoint(operator.domain, lambda x: mapping @ x)
