"""
Provides classes for linear operators between Hilbert spaces.

This module is the primary tool for defining and manipulating linear mappings
between `HilbertSpace` objects. It provides a powerful `LinearOperator` class
that supports a rich algebra and includes numerous factory methods for
convenient construction from matrices, forms, or tensor products.

Key Classes
-----------
- `LinearOperator`: The main workhorse for linear algebra. It represents a
  linear map `L(x) = Ax` and provides rich functionality, including composition
  (`@`), adjoints (`.adjoint`), duals (`.dual`), and matrix representations
  (`.matrix`).
- `DiagonalLinearOperator`: A specialized, efficient implementation for linear
  operators that are diagonal in their component representation, notable for
  supporting functional calculus (e.g., `.inverse`, `.sqrt`).
"""

from __future__ import annotations
from typing import Callable, List, Optional, Any, Union, Tuple, TYPE_CHECKING

import numpy as np
from scipy.sparse.linalg import LinearOperator as ScipyLinOp
from scipy.sparse import diags

# from .operators import Operator
from .nonlinear_operators import NonLinearOperator

from .random_matrix import (
    random_range,
    random_svd as rm_svd,
    random_cholesky as rm_chol,
    random_eig as rm_eig,
)

from .parallel import parallel_compute_dense_matrix_from_scipy_op

from .checks.linear_operators import LinearOperatorAxiomChecks

# This block only runs for type checkers, not at runtime
if TYPE_CHECKING:
    from .hilbert_space import HilbertSpace, EuclideanSpace
    from .linear_forms import LinearForm


class LinearOperator(NonLinearOperator, LinearOperatorAxiomChecks):
    """A linear operator between two Hilbert spaces.

    This class represents a linear map `L(x) = Ax` and provides rich
    functionality for linear algebraic operations. It specializes
    `NonLinearOperator`, correctly defining its derivative as the operator
    itself.

    Key features include operator algebra (`@`, `+`, `*`), automatic
    derivation of adjoint (`.adjoint`) and dual (`.dual`) operators, and
    multiple matrix representations (`.matrix()`) for use with numerical
    solvers.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        codomain: HilbertSpace,
        mapping: Callable[[Any], Any],
        /,
        *,
        dual_mapping: Optional[Callable[[Any], Any]] = None,
        adjoint_mapping: Optional[Callable[[Any], Any]] = None,
        thread_safe: bool = False,
        dual_base: Optional[LinearOperator] = None,
        adjoint_base: Optional[LinearOperator] = None,
    ) -> None:
        """
        Initializes the LinearOperator.

        Args:
            domain (HilbertSpace): The domain of the operator.
            codomain (HilbertSpace): The codomain of the operator.
            mapping (callable): The function defining the linear mapping.
            dual_mapping (callable, optional): The action of the dual operator.
            adjoint_mapping (callable, optional): The action of the adjoint.
            thread_safe (bool, optional): True if the mapping is thread-safe.
            dual_base (LinearOperator, optional): Internal use for duals.
            adjoint_base (LinearOperator, optional): Internal use for adjoints.
        """
        super().__init__(
            domain, codomain, self._mapping_impl, derivative=self._derivative_impl
        )
        self._mapping = mapping
        self._dual_base: Optional[LinearOperator] = dual_base
        self._adjoint_base: Optional[LinearOperator] = adjoint_base
        self._thread_safe: bool = thread_safe
        self.__adjoint_mapping: Callable[[Any], Any]
        self.__dual_mapping: Callable[[Any], Any]

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
    def self_dual(
        domain: HilbertSpace, mapping: Callable[[Any], Any]
    ) -> LinearOperator:
        """Creates a self-dual operator."""
        return LinearOperator(domain, domain.dual, mapping, dual_mapping=mapping)

    @staticmethod
    def self_adjoint(
        domain: HilbertSpace, mapping: Callable[[Any], Any]
    ) -> LinearOperator:
        """Creates a self-adjoint operator."""
        return LinearOperator(domain, domain, mapping, adjoint_mapping=mapping)

    @staticmethod
    def from_formal_adjoint(
        domain: HilbertSpace, codomain: HilbertSpace, operator: LinearOperator
    ) -> LinearOperator:
        """
        Constructs an operator on weighted spaces from one on the underlying spaces.

        This is a key method for working with `MassWeightedHilbertSpace`. It takes
        an operator `A` that is defined on the simple, unweighted underlying spaces
        and "lifts" it to be a proper operator on the mass-weighted spaces. It
        correctly defines the new operator's adjoint with respect to the
        weighted inner products.

        This method automatically handles cases where the domain and/or codomain
        are a `HilbertSpaceDirectSum`, recursively building the necessary
        block-structured mass operators.

        Args:
            domain: The (potentially) mass-weighted domain of the new operator.
            codomain: The (potentially) mass-weighted codomain of the new operator.
            operator: The original operator defined on the underlying,
                unweighted spaces.

        Returns:
            A new `LinearOperator` that acts between the mass-weighted spaces.
        """
        from .hilbert_space import MassWeightedHilbertSpace
        from .direct_sum import HilbertSpaceDirectSum, BlockDiagonalLinearOperator

        def get_properties(space: HilbertSpace):
            if isinstance(space, MassWeightedHilbertSpace):
                return (
                    space.underlying_space,
                    space.mass_operator,
                    space.inverse_mass_operator,
                )
            elif isinstance(space, HilbertSpaceDirectSum):
                properties = [get_properties(subspace) for subspace in space.subspaces]
                underlying_space = HilbertSpaceDirectSum(
                    [property[0] for property in properties]
                )
                mass_operator = BlockDiagonalLinearOperator(
                    [property[1] for property in properties]
                )
                inverse_mass_operator = BlockDiagonalLinearOperator(
                    [property[2] for property in properties]
                )
                return (
                    underlying_space,
                    mass_operator,
                    inverse_mass_operator,
                )
            else:
                return space, space.identity_operator(), space.identity_operator()

        domain_base, _, domain_inverse_mass_operator = get_properties(domain)
        codomain_base, codomain_mass_operator, _ = get_properties(codomain)

        if domain_base != operator.domain:
            raise ValueError("Domain mismatch")

        if codomain_base != operator.codomain:
            raise ValueError("Codomain mismatch")

        return LinearOperator(
            domain,
            codomain,
            operator,
            adjoint_mapping=domain_inverse_mass_operator
            @ operator.adjoint
            @ codomain_mass_operator,
        )

    @staticmethod
    def from_formally_self_adjoint(
        domain: HilbertSpace, operator: LinearOperator
    ) -> LinearOperator:
        """
        Constructs a self-adjoint operator on a weighted space.

        This method takes an operator that is formally self-adjoint on an
        underlying (unweighted) space and promotes it to a truly self-adjoint
        operator on the `MassWeightedHilbertSpace`. It automatically handles
        `HilbertSpaceDirectSum` domains.

        Args:
            domain (HilbertSpace): The domain of the operator, which can be a
                `MassWeightedHilbertSpace` or a `HilbertSpaceDirectSum`.
            operator (LinearOperator): The operator to be converted.
        """
        return LinearOperator.from_formal_adjoint(domain, domain, operator)

    @staticmethod
    def from_linear_forms(forms: List[LinearForm]) -> LinearOperator:
        """
        Creates an operator from a list of linear forms.

        The resulting operator maps from the forms' domain to an N-dimensional
        Euclidean space, where N is the number of forms.
        """
        from .hilbert_space import EuclideanSpace

        domain = forms[0].domain
        codomain = EuclideanSpace(len(forms))
        if not all(form.domain == domain for form in forms):
            raise ValueError("Forms need to be defined on a common domain")

        matrix = np.zeros((codomain.dim, domain.dim))
        for i, form in enumerate(forms):
            matrix[i, :] = form.components

        def mapping(x: Any) -> np.ndarray:
            cx = domain.to_components(x)
            cy = matrix @ cx
            return cy

        def dual_mapping(yp: Any) -> Any:
            cyp = codomain.dual.to_components(yp)
            cxp = matrix.T @ cyp
            return domain.dual.from_components(cxp)

        return LinearOperator(domain, codomain, mapping, dual_mapping=dual_mapping)

    @staticmethod
    def from_matrix(
        domain: HilbertSpace,
        codomain: HilbertSpace,
        matrix: Union[np.ndarray, ScipyLinOp],
        /,
        *,
        galerkin: bool = False,
    ) -> LinearOperator:
        """
        Creates a LinearOperator from its matrix representation.

        This factory defines a `LinearOperator` using a concrete matrix that
        acts on the component vectors of the abstract Hilbert space vectors.

        Args:
            domain: The operator's domain space.
            codomain: The operator's codomain space.
            matrix: The matrix representation (NumPy array or SciPy
                LinearOperator). Shape must be `(codomain.dim, domain.dim)`.
            galerkin: If `True`, the matrix is interpreted in its "weak form"
                or Galerkin representation (`M_ij = <basis_j, A(basis_i)>`),
                which maps a vector's components to the components of its
                *dual*. This is crucial as it ensures a self-adjoint
                operator is represented by a symmetric matrix. If `False`
                (default), it's a standard component-to-component map.

        Returns:
            A new `LinearOperator` defined by the matrix action.
        """

        assert matrix.shape == (codomain.dim, domain.dim)

        if galerkin:

            def mapping(x: Any) -> Any:
                cx = domain.to_components(x)
                cyp = matrix @ cx
                yp = codomain.dual.from_components(cyp)
                return codomain.from_dual(yp)

            def adjoint_mapping(y: Any) -> Any:
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

            def mapping(x: Any) -> Any:
                cx = domain.to_components(x)
                cy = matrix @ cx
                return codomain.from_components(cy)

            def dual_mapping(yp: Any) -> Any:
                cyp = codomain.dual.to_components(yp)
                cxp = matrix.T @ cyp
                return domain.dual.from_components(cxp)

            return LinearOperator(domain, codomain, mapping, dual_mapping=dual_mapping)

    @staticmethod
    def self_adjoint_from_matrix(
        domain: HilbertSpace, matrix: Union[np.ndarray, ScipyLinOp]
    ) -> LinearOperator:
        """Forms a self-adjoint operator from its Galerkin matrix."""

        def mapping(x: Any) -> Any:
            cx = domain.to_components(x)
            cyp = matrix @ cx
            yp = domain.dual.from_components(cyp)
            return domain.from_dual(yp)

        return LinearOperator.self_adjoint(domain, mapping)

    @staticmethod
    def from_tensor_product(
        domain: HilbertSpace,
        codomain: HilbertSpace,
        vector_pairs: List[Tuple[Any, Any]],
        /,
        *,
        weights: Optional[List[float]] = None,
    ) -> LinearOperator:
        """
        Creates an operator from a weighted sum of tensor products.

        The operator represents A(x) = sum_i( w_i * <x, v_i> * u_i ),
        where vector_pairs are (u_i, v_i).
        """
        _weights = [1.0] * len(vector_pairs) if weights is None else weights

        def mapping(x: Any) -> Any:
            y = codomain.zero
            for (left, right), weight in zip(vector_pairs, _weights):
                product = domain.inner_product(right, x)
                codomain.axpy(weight * product, left, y)
            return y

        def adjoint_mapping(y: Any) -> Any:
            x = domain.zero
            for (left, right), weight in zip(vector_pairs, _weights):
                product = codomain.inner_product(left, y)
                domain.axpy(weight * product, right, x)
            return x

        return LinearOperator(
            domain, codomain, mapping, adjoint_mapping=adjoint_mapping
        )

    @staticmethod
    def self_adjoint_from_tensor_product(
        domain: HilbertSpace,
        vectors: List[Any],
        /,
        *,
        weights: Optional[List[float]] = None,
    ) -> LinearOperator:
        """Creates a self-adjoint operator from a tensor product sum."""
        _weights = [1.0] * len(vectors) if weights is None else weights

        def mapping(x: Any) -> Any:
            y = domain.zero
            for vector, weight in zip(vectors, _weights):
                product = domain.inner_product(vector, x)
                domain.axpy(weight * product, vector, y)
            return y

        return LinearOperator.self_adjoint(domain, mapping)

    @property
    def linear(self) -> bool:
        """True, as this is a LinearOperator."""
        return True

    @property
    def dual(self) -> LinearOperator:
        """The dual of the operator."""
        if self._dual_base is None:
            return LinearOperator(
                self.codomain.dual,
                self.domain.dual,
                self.__dual_mapping,
                dual_base=self,
            )
        else:
            return self._dual_base

    @property
    def adjoint(self) -> LinearOperator:
        """The adjoint of the operator."""
        if self._adjoint_base is None:
            return LinearOperator(
                self.codomain,
                self.domain,
                self.__adjoint_mapping,
                adjoint_base=self,
            )
        else:
            return self._adjoint_base

    @property
    def thread_safe(self) -> bool:
        """True if the operator's mapping is thread-safe."""
        return self._thread_safe

    def matrix(
        self,
        /,
        *,
        dense: bool = False,
        galerkin: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> Union[ScipyLinOp, np.ndarray]:
        """Returns a matrix representation of the operator.

        This provides a concrete matrix that represents the operator's action
        on the underlying component vectors.

        Args:
            dense: If `True`, returns a dense `numpy.ndarray`. If `False`
                (default), returns a memory-efficient, matrix-free
                `scipy.sparse.linalg.LinearOperator`.
            galerkin: If `True`, the returned matrix is the Galerkin
                representation, whose `rmatvec` corresponds to the
                **adjoint** operator. If `False` (default), the `rmatvec`
                corresponds to the **dual** operator. The Galerkin form is
                essential for algorithms that rely on symmetry/self-adjointness.
            parallel: If `True` and `dense=True`, computes the matrix columns
                in parallel.
            n_jobs: Number of parallel jobs to use. `-1` uses all available cores.

        Returns:
            The matrix representation, either dense or matrix-free.
        """

        if dense:
            return self._compute_dense_matrix(galerkin, parallel, n_jobs)
        else:
            if galerkin:

                def matvec(cx: np.ndarray) -> np.ndarray:
                    x = self.domain.from_components(cx)
                    y = self(x)
                    yp = self.codomain.to_dual(y)
                    return self.codomain.dual.to_components(yp)

                def rmatvec(cy: np.ndarray) -> np.ndarray:
                    y = self.codomain.from_components(cy)
                    x = self.adjoint(y)
                    xp = self.domain.to_dual(x)
                    return self.domain.dual.to_components(xp)

            else:

                def matvec(cx: np.ndarray) -> np.ndarray:
                    x = self.domain.from_components(cx)
                    y = self(x)
                    return self.codomain.to_components(y)

                def rmatvec(cyp: np.ndarray) -> np.ndarray:
                    yp = self.codomain.dual.from_components(cyp)
                    xp = self.dual(yp)
                    return self.domain.dual.to_components(xp)

            def matmat(xmat: np.ndarray) -> np.ndarray:
                n, k = xmat.shape
                assert n == self.domain.dim
                ymat = np.zeros((self.codomain.dim, k))
                for j in range(k):
                    cx = xmat[:, j]
                    ymat[:, j] = matvec(cx)
                return ymat

            def rmatmat(ymat: np.ndarray) -> np.ndarray:
                m, k = ymat.shape
                assert m == self.codomain.dim
                xmat = np.zeros((self.domain.dim, k))
                for j in range(k):
                    cy = ymat[:, j]
                    xmat[:, j] = rmatvec(cy)
                return xmat

            return ScipyLinOp(
                (self.codomain.dim, self.domain.dim),
                matvec=matvec,
                rmatvec=rmatvec,
                matmat=matmat,
                rmatmat=rmatmat,
            )

    def random_svd(
        self,
        size_estimate: int,
        /,
        *,
        galerkin: bool = False,
        method: str = "variable",
        max_rank: int = None,
        power: int = 2,
        rtol: float = 1e-4,
        block_size: int = 10,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> Tuple[LinearOperator, DiagonalLinearOperator, LinearOperator]:
        """
        Computes an approximate SVD using a randomized algorithm.

        Args:
            size_estimate: For 'fixed' method, the exact target rank. For 'variable'
                       method, this is the initial rank to sample.
            galerkin (bool): If True, use the Galerkin representation.
            method ({'variable', 'fixed'}): The algorithm to use.
            - 'variable': (Default) Progressively samples to find the rank needed
                          to meet tolerance `rtol`, stopping at `max_rank`.
            - 'fixed': Returns a basis with exactly `size_estimate` columns.
            max_rank: For 'variable' method, a hard limit on the rank. Ignored if
                    method='fixed'. Defaults to min(m, n).
            power: Number of power iterations to improve accuracy.
            rtol: Relative tolerance for the 'variable' method. Ignored if
                method='fixed'.
            block_size: Number of new vectors to sample per iteration in 'variable'
                        method. Ignored if method='fixed'.
            parallel: Whether to use parallel matrix multiplication.
            n_jobs: Number of jobs for parallelism.

        Returns:
            left (LinearOperator): The left singular vector matrix.
            singular_values (DiagonalLinearOperator): The singular values.
            right (LinearOperator): The right singular vector matrix.

        Notes:
            The right factor is in transposed form. This means the original
            operator can be approximated as:
            A = left @ singular_values @ right
        """
        from .hilbert_space import EuclideanSpace

        matrix = self.matrix(galerkin=galerkin)
        m, n = matrix.shape
        k = min(m, n)

        qr_factor = random_range(
            matrix,
            size_estimate if size_estimate < k else k,
            method=method,
            max_rank=max_rank,
            power=power,
            rtol=rtol,
            block_size=block_size,
            parallel=parallel,
            n_jobs=n_jobs,
        )

        left_factor_mat, singular_values, right_factor_transposed = rm_svd(
            matrix, qr_factor
        )

        euclidean = EuclideanSpace(qr_factor.shape[1])
        diagonal = DiagonalLinearOperator(euclidean, euclidean, singular_values)

        if galerkin:
            right = LinearOperator.from_matrix(
                self.domain, euclidean, right_factor_transposed, galerkin=False
            )
            left = LinearOperator.from_matrix(
                euclidean, self.codomain, left_factor_mat, galerkin=True
            )
        else:
            right = LinearOperator.from_matrix(
                self.domain, euclidean, right_factor_transposed, galerkin=False
            )
            left = LinearOperator.from_matrix(
                euclidean, self.codomain, left_factor_mat, galerkin=False
            )

        return left, diagonal, right

    def random_eig(
        self,
        size_estimate: int,
        /,
        *,
        method: str = "variable",
        max_rank: int = None,
        power: int = 2,
        rtol: float = 1e-4,
        block_size: int = 10,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> Tuple[LinearOperator, DiagonalLinearOperator]:
        """
        Computes an approximate eigen-decomposition using a randomized algorithm.

        Args:
            size_estimate: For 'fixed' method, the exact target rank. For 'variable'
                       method, this is the initial rank to sample.
            method ({'variable', 'fixed'}): The algorithm to use.
            - 'variable': (Default) Progressively samples to find the rank needed
                          to meet tolerance `rtol`, stopping at `max_rank`.
            - 'fixed': Returns a basis with exactly `size_estimate` columns.
            max_rank: For 'variable' method, a hard limit on the rank. Ignored if
                    method='fixed'. Defaults to min(m, n).
            power: Number of power iterations to improve accuracy.
            rtol: Relative tolerance for the 'variable' method. Ignored if
                method='fixed'.
            block_size: Number of new vectors to sample per iteration in 'variable'
                        method. Ignored if method='fixed'.
            parallel: Whether to use parallel matrix multiplication.
            n_jobs: Number of jobs for parallelism.

        Returns:
            expansion (LinearOperator): Mapping from coefficients in eigen-basis to vectors.
            eigenvaluevalues (DiagonalLinearOperator): The eigenvalues values.

        """
        from .hilbert_space import EuclideanSpace

        assert self.is_automorphism
        matrix = self.matrix(galerkin=True)
        m, n = matrix.shape
        k = min(m, n)

        qr_factor = random_range(
            matrix,
            size_estimate if size_estimate < k else k,
            method=method,
            max_rank=max_rank,
            power=power,
            rtol=rtol,
            block_size=block_size,
            parallel=parallel,
            n_jobs=n_jobs,
        )

        eigenvectors, eigenvalues = rm_eig(matrix, qr_factor)
        euclidean = EuclideanSpace(qr_factor.shape[1])
        diagonal = DiagonalLinearOperator(euclidean, euclidean, eigenvalues)

        expansion = LinearOperator.from_matrix(
            euclidean, self.domain, eigenvectors, galerkin=True
        )

        return expansion, diagonal

    def random_cholesky(
        self,
        size_estimate: int,
        /,
        *,
        method: str = "variable",
        max_rank: int = None,
        power: int = 2,
        rtol: float = 1e-4,
        block_size: int = 10,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> LinearOperator:
        """
        Computes an approximate Cholesky decomposition for a positive-definite
        self-adjoint operator using a randomized algorithm.

        Args:
            size_estimate: For 'fixed' method, the exact target rank. For 'variable'
                       method, this is the initial rank to sample.
            method ({'variable', 'fixed'}): The algorithm to use.
            - 'variable': (Default) Progressively samples to find the rank needed
                          to meet tolerance `rtol`, stopping at `max_rank`.
            - 'fixed': Returns a basis with exactly `size_estimate` columns.
            max_rank: For 'variable' method, a hard limit on the rank. Ignored if
                    method='fixed'. Defaults to min(m, n).
            power: Number of power iterations to improve accuracy.
            rtol: Relative tolerance for the 'variable' method. Ignored if
                method='fixed'.
            block_size: Number of new vectors to sample per iteration in 'variable'
                        method. Ignored if method='fixed'.
            parallel: Whether to use parallel matrix multiplication.
            n_jobs: Number of jobs for parallelism.

        Returns:
            factor (LinearOperator): A linear operator from a Euclidean space
                into the domain of the operator.

        Notes:
            The original operator can be approximated as:
                A = factor @ factor.adjoint
        """

        from .hilbert_space import EuclideanSpace

        assert self.is_automorphism
        matrix = self.matrix(galerkin=True)
        m, n = matrix.shape
        k = min(m, n)

        qr_factor = random_range(
            matrix,
            size_estimate if size_estimate < k else k,
            method=method,
            max_rank=max_rank,
            power=power,
            rtol=rtol,
            block_size=block_size,
            parallel=parallel,
            n_jobs=n_jobs,
        )

        cholesky_factor = rm_chol(matrix, qr_factor)

        return LinearOperator.from_matrix(
            EuclideanSpace(qr_factor.shape[1]),
            self.domain,
            cholesky_factor,
            galerkin=True,
        )

    def _mapping_impl(self, x: Any) -> Any:
        return self._mapping(x)

    def _derivative_impl(self, _: Any) -> LinearOperator:
        return self

    def _dual_mapping_default(self, yp: Any) -> LinearForm:
        from .linear_forms import LinearForm

        return LinearForm(self.domain, mapping=lambda x: yp(self(x)))

    def _dual_mapping_from_adjoint(self, yp: Any) -> Any:
        y = self.codomain.from_dual(yp)
        x = self.__adjoint_mapping(y)
        return self.domain.to_dual(x)

    def _adjoint_mapping_from_dual(self, y: Any) -> Any:
        yp = self.codomain.to_dual(y)
        xp = self.__dual_mapping(yp)
        return self.domain.from_dual(xp)

    def _compute_dense_matrix(
        self, galerkin: bool, parallel: bool, n_jobs: int
    ) -> np.ndarray:

        scipy_op_wrapper = self.matrix(galerkin=galerkin)

        if not parallel:
            matrix = np.zeros((self.codomain.dim, self.domain.dim))
            cx = np.zeros(self.domain.dim)
            for i in range(self.domain.dim):
                cx[i] = 1.0
                matrix[:, i] = (scipy_op_wrapper @ cx)[:]
                cx[i] = 0.0
            return matrix
        else:
            return parallel_compute_dense_matrix_from_scipy_op(
                scipy_op_wrapper, n_jobs=n_jobs
            )

    def __neg__(self) -> LinearOperator:
        domain = self.domain
        codomain = self.codomain

        def mapping(x: Any) -> Any:
            return codomain.negative(self(x))

        def adjoint_mapping(y: Any) -> Any:
            return domain.negative(self.adjoint(y))

        return LinearOperator(
            domain, codomain, mapping, adjoint_mapping=adjoint_mapping
        )

    def __mul__(self, a: float) -> LinearOperator:
        domain = self.domain
        codomain = self.codomain

        def mapping(x: Any) -> Any:
            return codomain.multiply(a, self(x))

        def adjoint_mapping(y: Any) -> Any:
            return domain.multiply(a, self.adjoint(y))

        return LinearOperator(
            domain, codomain, mapping, adjoint_mapping=adjoint_mapping
        )

    def __rmul__(self, a: float) -> LinearOperator:
        return self * a

    def __truediv__(self, a: float) -> LinearOperator:
        return self * (1.0 / a)

    def __add__(
        self, other: NonLinearOperator | LinearOperator
    ) -> NonLinearOperator | LinearOperator:
        """Returns the sum of this operator and another.

        If `other` is also a `LinearOperator`, this performs an optimized
        addition that preserves linearity and correctly defines the new
        operator's `adjoint`. Otherwise, it delegates to the general
        implementation in the `NonLinearOperator` base class.

        Args:
            other: The operator to add to this one.

        Returns:
            A new `LinearOperator` if adding two linear operators, otherwise
            a `NonLinearOperator`.
        """

        if isinstance(other, LinearOperator):
            domain = self.domain
            codomain = self.codomain

            def mapping(x: Any) -> Any:
                return codomain.add(self(x), other(x))

            def adjoint_mapping(y: Any) -> Any:
                return domain.add(self.adjoint(y), other.adjoint(y))

            return LinearOperator(
                domain, codomain, mapping, adjoint_mapping=adjoint_mapping
            )
        else:
            return super().__add__(other)

    def __sub__(
        self, other: NonLinearOperator | LinearOperator
    ) -> NonLinearOperator | LinearOperator:
        """Returns the difference between this operator and another.

        If `other` is also a `LinearOperator`, this performs an optimized
        subtraction that preserves linearity and correctly defines the new
        operator's `adjoint`. Otherwise, it delegates to the general
        implementation in the `NonLinearOperator` base class.

        Args:
            other: The operator to subtract from this one.

        Returns:
            A new `LinearOperator` if subtracting two linear operators,
            otherwise a `NonLinearOperator`.
        """

        if isinstance(other, LinearOperator):

            domain = self.domain
            codomain = self.codomain

            def mapping(x: Any) -> Any:
                return codomain.subtract(self(x), other(x))

            def adjoint_mapping(y: Any) -> Any:
                return domain.subtract(self.adjoint(y), other.adjoint(y))

            return LinearOperator(
                domain, codomain, mapping, adjoint_mapping=adjoint_mapping
            )
        else:
            return super().__sub__(other)

    def __matmul__(
        self, other: NonLinearOperator | LinearOperator
    ) -> NonLinearOperator | LinearOperator:
        """Composes this operator with another using the @ symbol.

        The composition `(self @ other)` results in a new operator that
        first applies `other` and then applies `self`, i.e.,
        `(self @ other)(x) = self(other(x))`.

        If `other` is also a `LinearOperator`, this creates a new `LinearOperator`
        whose adjoint is correctly defined using the composition rule:
        `(L1 @ L2)* = L2* @ L1*`. Otherwise, it delegates to the general
        `NonLinearOperator` implementation.

        Args:
            other: The operator to compose with (the right-hand operator).

        Returns:
            A new `LinearOperator` if composing two linear operators,
            otherwise a `NonLinearOperator`.
        """

        if isinstance(other, LinearOperator):
            domain = other.domain
            codomain = self.codomain

            def mapping(x: Any) -> Any:
                return self(other(x))

            def adjoint_mapping(y: Any) -> Any:
                return other.adjoint(self.adjoint(y))

            return LinearOperator(
                domain, codomain, mapping, adjoint_mapping=adjoint_mapping
            )

        else:
            return super().__matmul__(other)

    def __str__(self) -> str:
        return self.matrix(dense=True).__str__()


class DiagonalLinearOperator(LinearOperator):
    """A LinearOperator that is diagonal in its component representation.

    This provides an efficient implementation for diagonal linear operators.
    Its key feature is support for **functional calculus**, allowing for the
    direct computation of operator functions like inverse (`.inverse`) or

    square root (`.sqrt`) by applying the function to the diagonal entries.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        codomain: HilbertSpace,
        diagonal_values: np.ndarray,
        /,
        *,
        galerkin: bool = False,
    ) -> None:
        """
        Initializes the DiagonalLinearOperator.

        Args:
            domain (HilbertSpace): The domain of the operator.
            codomain (HilbertSpace): The codomain of the operator.
            diagonal_values (np.ndarray): The diagonal entries of the
                operator's matrix representation.
            galerkin (bool): If True, use the Galerkin representation.
        """

        assert domain.dim == codomain.dim
        assert domain.dim == len(diagonal_values)
        self._diagonal_values: np.ndarray = diagonal_values
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
    def diagonal_values(self) -> np.ndarray:
        """The diagonal entries of the operator's matrix representation."""
        return self._diagonal_values

    def function(self, f: Callable[[float], float]) -> DiagonalLinearOperator:
        """Applies a function to the operator via functional calculus.

        This creates a new `DiagonalLinearOperator` where the function `f` has
        been applied to each of the diagonal entries. For example,
        `op.function(lambda x: 1/x)` computes the inverse.

        Args:
            f: A scalar function to apply to the diagonal values.

        Returns:
            A new `DiagonalLinearOperator` with the transformed diagonal.
        """
        diagonal_values = np.array([f(x) for x in self.diagonal_values])
        return DiagonalLinearOperator(self.domain, self.codomain, diagonal_values)

    @property
    def inverse(self) -> DiagonalLinearOperator:
        """
        The inverse of the operator, computed via functional calculus.
        Requires all diagonal values to be non-zero.
        """
        assert all(val != 0 for val in self.diagonal_values)
        return self.function(lambda x: 1 / x)

    @property
    def sqrt(self) -> DiagonalLinearOperator:
        """
        The square root of the operator, computed via functional calculus.
        Requires all diagonal values to be non-negative.
        """
        assert all(val >= 0 for val in self._diagonal_values)
        return self.function(np.sqrt)
