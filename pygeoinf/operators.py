"""
Defines a class hierarchy for operators between Hilbert spaces.

This module provides the tools for defining and manipulating mappings between
`HilbertSpace` objects. It distinguishes between general non-linear operators
and the more structured linear operators, which are the primary focus and
support a rich algebra.

Key Classes
-----------
- `Operator`: A general, potentially non-linear operator defined by a simple
  mapping function.
- `LinearOperator`: The main workhorse for linear algebra. It represents a
  linear map and provides rich functionality, including composition (`@`),
  adjoints (`.adjoint`), duals (`.dual`), and matrix representations (`.matrix`).
  It includes numerous factory methods for convenient construction.
- `DiagonalLinearOperator`: A specialized, efficient implementation for linear
  operators that are diagonal in their component representation, which supports
  functional calculus.
"""

from __future__ import annotations
from typing import Callable, List, Optional, Any, Union, Tuple, TYPE_CHECKING

import numpy as np
from scipy.sparse.linalg import LinearOperator as ScipyLinOp
from scipy.sparse import diags


from .random_matrix import (
    fixed_rank_random_range,
    variable_rank_random_range,
    random_svd as rm_svd,
    random_cholesky as rm_chol,
    random_eig as rm_eig,
)

from .parallel import parallel_compute_dense_matrix_from_scipy_op

# This block only runs for type checkers, not at runtime
if TYPE_CHECKING:
    from .hilbert_space import HilbertSpace, EuclideanSpace
    from .linear_forms import LinearForm


def _parallel_scipy_op_col(scipy_op: ScipyLinOp, j: int, domain_dim: int) -> np.ndarray:
    """
    A top-level helper that applies a scipy.LinearOperator to a basis vector.

    This function is simple and serializable ("picklable").

    Args:
        scipy_op: The SciPy LinearOperator wrapper for the matrix action.
        j: The index of the basis vector (column) to compute.
        domain_dim: The dimension of the domain space.

    Returns:
        The j-th column of the dense matrix as a NumPy array.
    """
    # Create the j-th component basis vector
    cx = np.zeros(domain_dim)
    cx[j] = 1.0

    # Apply the SciPy wrapper, which handles all necessary conversions
    return scipy_op @ cx


class Operator:
    """
    A general, potentially non-linear operator between two Hilbert spaces.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        codomain: HilbertSpace,
        mapping: Callable[[Any], Any],
    ) -> None:
        """
        Initializes the Operator.

        Args:
            domain (HilbertSpace): Domain of the operator.
            codomain (HilbertSpace): Codomain of the operator.
            mapping (callable): The function defining the mapping from the
                domain to the codomain.
        """
        self._domain: HilbertSpace = domain
        self._codomain: HilbertSpace = codomain
        self.__mapping: Callable[[Any], Any] = mapping

    @property
    def domain(self) -> HilbertSpace:
        """The domain of the operator."""
        return self._domain

    @property
    def codomain(self) -> HilbertSpace:
        """The codomain of the operator."""
        return self._codomain

    @property
    def is_automorphism(self) -> bool:
        """True if the operator maps a space into itself."""
        return self.domain == self.codomain

    @property
    def is_square(self) -> bool:
        """True if the operator's domain and codomain have the same dimension."""
        return self.domain.dim == self.codomain.dim

    @property
    def linear(self) -> bool:
        """False for a general operator. Overridden by LinearOperator."""
        return False

    def __call__(self, x: Any) -> Any:
        """Applies the operator's mapping to a vector."""
        return self.__mapping(x)


class LinearOperator(Operator):
    """
    A linear operator between two Hilbert spaces.

    This class is the primary workhorse for linear algebraic operations. An
    operator can be defined "on the fly" from a callable mapping. The class
    automatically derives the associated `adjoint` and `dual` operators,
    which are fundamental for solving linear systems and for optimization.

    It supports a rich algebra, including composition (`@`), addition (`+`),
    and scalar multiplication (`*`). Operators can also be represented as
    dense or matrix-free (`scipy`) matrices for use with numerical solvers.
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
        super().__init__(domain, codomain, mapping)
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

        This factory method allows you to define a `LinearOperator` using a
        concrete matrix (like a `numpy.ndarray`) that acts on the component
        vectors of the abstract Hilbert space vectors. The `galerkin` flag
        determines how this matrix action is interpreted.

        Args:
            domain (HilbertSpace): The operator's domain.
            codomain (HilbertSpace): The operator's codomain.
            matrix (MatrixLike): The matrix representation, which can be a dense
                NumPy array or a SciPy LinearOperator. Its shape must be
                (codomain.dim, domain.dim).
            galerkin (bool): Specifies the interpretation of the matrix.

                - **`galerkin=False` (Default): Standard Component Mapping**
                  This is the most direct interpretation. The matrix `M` maps the
                  component vector `c_x` of an input vector `x` directly to the
                  component vector `c_y` of the output vector `y`.

                - **`galerkin=True`: Galerkin (or "Weak Form") Representation**
                  This interpretation is standard in the finite element method (FEM)
                  and other variational techniques. The matrix `M` maps the component
                  vector `c_x` of an input `x` to the component vector `c_yp` of the
                  *dual* of the output vector `y`.

                  - **Matrix Entries**: The matrix elements are defined by inner
                    products with basis vectors: `M_ij = inner_product(A(b_j), b_i)`,
                    where `b_j` are domain basis vectors and `b_i` are codomain
                    basis vectors.
                  - **Use Case**: This is critically important for preserving the
                    mathematical properties of an operator. For example, if an operator
                    `A` is self-adjoint, its Galerkin matrix `M` will be **symmetric**
                    (`M.T == M`). This allows the use of highly efficient numerical
                    methods like the Conjugate Gradient solver or Cholesky
                    factorization, which rely on symmetry. The standard component
                    matrix of a self-adjoint operator is generally not symmetric
                    unless the basis is orthonormal.

        Returns:
            LinearOperator: A new `LinearOperator` instance whose action is
                defined by the provided matrix and interpretation.
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
        """
        Returns a matrix representation of the operator.

        This method provides a concrete matrix that represents the abstract
        linear operator's action on the underlying component vectors.

        Args:
            dense (bool): Determines the format of the returned matrix.
                - If `True`, this method computes and returns a dense `numpy.ndarray`.
                  Be aware that this can be very memory-intensive for
                  high-dimensional spaces.
                - If `False` (default), it returns a matrix-free
                  `scipy.sparse.linalg.LinearOperator`. This object encapsulates
                  the operator's action (`matvec`) and its transpose action
                  (`rmatvec`) without ever explicitly forming the full matrix in memory,
                  making it ideal for large-scale problems.

            galerkin (bool): Specifies the interpretation of the matrix representation. This
                flag is crucial for correctly using the matrix with numerical solvers.

                - **`galerkin=False` (Default): Standard Component Mapping**
                  The returned matrix `M` performs a standard component-to-component
                  mapping.
                  - **`matvec` action**: Takes the component vector `c_x` of an input `x`
                    and returns the component vector `c_y` of the output `y`.
                  - **`rmatvec` action**: Corresponds to the matrix of the **dual operator**, `A'`.

                - **`galerkin=True`: Galerkin (or "Weak Form") Representation**
                  The returned matrix `M` represents the operator in a weak form, mapping
                  components of a vector to components of a dual vector.
                  - **`matvec` action**: Takes the component vector `c_x` of an input `x`
                    and returns the component vector `c_yp` of the *dual* of the output `y`.
                  - **`rmatvec` action**: Corresponds to the matrix of the **adjoint operator**, `A*`.
                  - **Key Property**: This representation is designed to preserve fundamental
                    mathematical properties. For instance, if the `LinearOperator` is
                    self-adjoint, its Galerkin matrix will be **symmetric**, which is a
                    prerequisite for algorithms like the Conjugate Gradient method.

            parallel (bool): If True, use parallel computing. Defaults to False.
                This is only relevant for dense matrices.
            n_jobs (int): Number of parallel jobs. Defaults to -1.
                This is only relevant for dense matrices.

        Returns:
            Union[ScipyLinOp, np.ndarray]: The matrix representation of the
                operator, either as a dense array or a matrix-free object.
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
        rank: int,
        /,
        *,
        power: int = 0,
        galerkin: bool = False,
        rtol: float = 1e-3,
        method: str = "fixed",
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> Tuple[LinearOperator, "DiagonalLinearOperator", LinearOperator]:
        """
        Computes an approximate SVD using a randomized algorithm.

        Args:
            rank (int): The desired rank of the SVD.
            power (int): The power of the random matrix.
            galerkin (bool): If True, use the Galerkin representation.
            rtol (float): The relative tolerance for the SVD.
            method (str): The method to use for the SVD.
                - "fixed": Use a fixed rank SVD.
                - "variable": Use a variable rank SVD.
            parallel (bool): If True, use parallel computing. Defaults to False.
                Only used with fixed rank method.
            n_jobs (int): Number of parallel jobs. Defaults to -1.
                Only used with fixed rank method.

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
        rank = rank if rank <= k else k

        qr_factor: np.ndarray
        if method == "fixed":
            qr_factor = fixed_rank_random_range(
                matrix, rank, power=power, parallel=parallel, n_jobs=n_jobs
            )
        elif method == "variable":
            qr_factor = variable_rank_random_range(matrix, rank, power=power, rtol=rtol)
        else:
            raise ValueError("Invalid method selected")

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
        rank: int,
        /,
        *,
        power: int = 0,
        rtol: float = 1e-3,
        method: str = "fixed",
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> Tuple[LinearOperator, "DiagonalLinearOperator"]:
        """
        Computes an approximate eigendecomposition for a self-adjoint
        operator using a randomized algorithm.

        Args:
            rank (int): The desired rank of the eigendecomposition.
            power (int): The power of the random matrix.
            rtol (float): The relative tolerance for the eigendecomposition.
            method (str): The method to use for the eigendecomposition.
                - "fixed": Use a fixed rank eigendecomposition.
                - "variable": Use a variable rank eigendecomposition.
            parallel (bool): If True, use parallel computing. Defaults to False.
                Only used with fixed rank method.
            n_jobs (int): Number of parallel jobs. Defaults to -1.
                Only used with fixed rank method.

        Returns:
            expansion (LinearOperator): A linear operator that maps coefficients
                in the eigen-basis to the resulting vector.
            eigenvalues (DiagonalLinearOperator): The eigenvalues.

        Notes:
            The original operator can be approximated as:
                A = expansion @ eigenvalues @ expansion.adjoint

        """
        from .hilbert_space import EuclideanSpace

        assert self.is_automorphism
        matrix = self.matrix(galerkin=True)
        m, n = matrix.shape
        k = min(m, n)
        rank = rank if rank <= k else k

        qr_factor: np.ndarray
        if method == "fixed":
            qr_factor = fixed_rank_random_range(
                matrix, rank, power=power, parallel=parallel, n_jobs=n_jobs
            )
        elif method == "variable":
            qr_factor = variable_rank_random_range(matrix, rank, power=power, rtol=rtol)
        else:
            raise ValueError("Invalid method selected")

        eigenvectors, eigenvalues = rm_eig(matrix, qr_factor)
        euclidean = EuclideanSpace(qr_factor.shape[1])
        diagonal = DiagonalLinearOperator(euclidean, euclidean, eigenvalues)

        expansion = LinearOperator.from_matrix(
            euclidean, self.domain, eigenvectors, galerkin=True
        )

        return expansion, diagonal

    def random_cholesky(
        self,
        rank: int,
        /,
        *,
        power: int = 0,
        rtol: float = 1e-3,
        method: str = "fixed",
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> LinearOperator:
        """
        Computes an approximate Cholesky decomposition for a positive-definite
        self-adjoint operator using a randomized algorithm.

        Args:
            rank (int): The desired rank of the Cholesky decomposition.
            power (int): The power of the random matrix.
            rtol (float): The relative tolerance for the Cholesky decomposition.
            method (str): The method to use for the Cholesky decomposition.
                - "fixed": Use a fixed rank Cholesky decomposition.
                - "variable": Use a variable rank Cholesky decomposition.
        parallel (bool): If True, use parallel computing. Defaults to False.
            Only used with fixed rank method.
        n_jobs (int): Number of parallel jobs. Defaults to -1.
                Only used with fixed rank method.

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
        rank = rank if rank <= k else k

        qr_factor: np.ndarray
        if method == "fixed":
            qr_factor = fixed_rank_random_range(
                matrix, rank, power=power, parallel=parallel, n_jobs=n_jobs
            )
        elif method == "variable":
            qr_factor = variable_rank_random_range(matrix, rank, power=power, rtol=rtol)
        else:
            raise ValueError("Invalid method selected")

        cholesky_factor = rm_chol(matrix, qr_factor)

        return LinearOperator.from_matrix(
            EuclideanSpace(qr_factor.shape[1]),
            self.domain,
            cholesky_factor,
            galerkin=True,
        )

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

    def __add__(self, other: LinearOperator) -> LinearOperator:
        domain = self.domain
        codomain = self.codomain

        def mapping(x: Any) -> Any:
            return codomain.add(self(x), other(x))

        def adjoint_mapping(y: Any) -> Any:
            return domain.add(self.adjoint(y), other.adjoint(y))

        return LinearOperator(
            domain, codomain, mapping, adjoint_mapping=adjoint_mapping
        )

    def __sub__(self, other: LinearOperator) -> LinearOperator:
        domain = self.domain
        codomain = self.codomain

        def mapping(x: Any) -> Any:
            return codomain.subtract(self(x), other(x))

        def adjoint_mapping(y: Any) -> Any:
            return domain.subtract(self.adjoint(y), other.adjoint(y))

        return LinearOperator(
            domain, codomain, mapping, adjoint_mapping=adjoint_mapping
        )

    def __matmul__(self, other: LinearOperator) -> LinearOperator:
        domain = other.domain
        codomain = self.codomain

        def mapping(x: Any) -> Any:
            return self(other(x))

        def adjoint_mapping(y: Any) -> Any:
            return other.adjoint(self.adjoint(y))

        return LinearOperator(
            domain, codomain, mapping, adjoint_mapping=adjoint_mapping
        )

    def __str__(self) -> str:
        return self.matrix(dense=True).__str__()


class DiagonalLinearOperator(LinearOperator):
    """
    A LinearOperator that is diagonal in its component representation.
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

    def function(self, f: Callable[[float], float]) -> "DiagonalLinearOperator":
        """
        Applies a function to the operator via functional calculus.

        This creates a new DiagonalLinearOperator where each diagonal entry `d_i`
        is replaced by `f(d_i)`.

        Args:
            f: A function that maps a float to a float.
        """
        diagonal_values = np.array([f(x) for x in self.diagonal_values])
        return DiagonalLinearOperator(self.domain, self.codomain, diagonal_values)

    @property
    def inverse(self) -> "DiagonalLinearOperator":
        """
        The inverse of the operator, computed via functional calculus.
        Requires all diagonal values to be non-zero.
        """
        assert all(val != 0 for val in self.diagonal_values)
        return self.function(lambda x: 1 / x)

    @property
    def sqrt(self) -> "DiagonalLinearOperator":
        """
        The square root of the operator, computed via functional calculus.
        Requires all diagonal values to be non-negative.
        """
        assert all(val >= 0 for val in self._diagonal_values)
        return self.function(np.sqrt)
