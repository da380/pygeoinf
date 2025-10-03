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
from typing import Callable, List, Optional, Any, Union, Tuple, TYPE_CHECKING, Dict

from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator as ScipyLinOp


from joblib import Parallel, delayed

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
    from .hilbert_space import HilbertSpace
    from .linear_forms import LinearForm


class LinearOperator(NonLinearOperator, LinearOperatorAxiomChecks):
    """A linear operator between two Hilbert spaces.

    This class represents a linear map `L(x) = Ax` and provides rich
    functionality for linear algebraic operations. It specializes
    `NonLinearOperator`, with the derivative mapping taking the
    required form (i.e., the derivative is just the operator itself).

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
            dual_base (LinearOperator, optional): Internal use for duals.
            adjoint_base (LinearOperator, optional): Internal use for adjoints.

        Notes:
            If neither the dual or adjoint mappings are provided, an they are
            deduced internally using a correction but very inefficient method.
            In general this functionality should not be relied on other than
            for operators between low-dimensional spaces.
        """
        super().__init__(
            domain, codomain, self._mapping_impl, derivative=self._derivative_impl
        )
        self._mapping = mapping
        self._dual_base: Optional[LinearOperator] = dual_base
        self._adjoint_base: Optional[LinearOperator] = adjoint_base
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
        matrix: Union[np.ndarray, sp.sparray, ScipyLinOp],
        /,
        *,
        galerkin: bool = False,
    ) -> MatrixLinearOperator:
        """
        Creates the most appropriate LinearOperator from a matrix representation.

        This factory method acts as a dispatcher, inspecting the type of the
        input matrix and returning the most specialized and optimized operator
        subclass (e.g., Dense, Sparse, or DiagonalSparse). It also handles
        matrix-free `scipy.sparse.linalg.LinearOperator` objects.

        Args:
            domain: The operator's domain space.
            codomain: The operator's codomain space.
            matrix: The matrix representation (NumPy ndarray, SciPy sparray,
                    or SciPy LinearOperator).
            galerkin: If `True`, the matrix is interpreted in Galerkin form.

        Returns:
            An instance of the most appropriate MatrixLinearOperator subclass.
        """
        # The order of these checks is important: from most specific to most general.

        # 1. Check for the most specific diagonal-sparse format
        if isinstance(matrix, sp.dia_array):
            diagonals_tuple = (matrix.data, matrix.offsets)
            return DiagonalSparseMatrixLinearOperator(
                domain, codomain, diagonals_tuple, galerkin=galerkin
            )

        # 2. Check for any other modern sparse format
        elif isinstance(matrix, sp.sparray):
            return SparseMatrixLinearOperator(
                domain, codomain, matrix, galerkin=galerkin
            )

        # 3. Check for a dense NumPy array
        elif isinstance(matrix, np.ndarray):
            return DenseMatrixLinearOperator(
                domain, codomain, matrix, galerkin=galerkin
            )

        # 4. Check for a matrix-free SciPy LinearOperator
        elif isinstance(matrix, ScipyLinOp):
            # This is matrix-free, so the general MatrixLinearOperator is the correct wrapper.
            return MatrixLinearOperator(domain, codomain, matrix, galerkin=galerkin)

        # 5. Handle legacy sparse matrix formats (optional but robust)
        elif sp.issparse(matrix):
            modern_array = sp.csr_array(matrix)
            return SparseMatrixLinearOperator(
                domain, codomain, modern_array, galerkin=galerkin
            )

        # 6. Raise an error for unsupported types
        else:
            raise TypeError(f"Unsupported matrix type: {type(matrix)}")

    @staticmethod
    def self_adjoint_from_matrix(
        domain: HilbertSpace,
        matrix: Union[np.ndarray, sp.sparray, ScipyLinOp],
    ) -> MatrixLinearOperator:
        """
        Creates the most appropriate self-adjoint LinearOperator from a matrix.

        This factory acts as a dispatcher, returning the most specialized
        subclass for the given matrix type (e.g., Dense, Sparse).

        It ALWAYS assumes the provided matrix is the **Galerkin** representation
        of the operator. The user is responsible for ensuring the input matrix
        is symmetric (or self-adjoint for ScipyLinOp).

        Args:
            domain: The operator's domain and codomain space.
            matrix: The symmetric matrix representation.

        Returns:
            An instance of the most appropriate MatrixLinearOperator subclass.
        """
        # Dispatch to the appropriate subclass, always with galerkin=True
        if isinstance(matrix, sp.dia_array):
            diagonals_tuple = (matrix.data, matrix.offsets)
            return DiagonalSparseMatrixLinearOperator(
                domain, domain, diagonals_tuple, galerkin=True
            )
        elif isinstance(matrix, sp.sparray):
            return SparseMatrixLinearOperator(domain, domain, matrix, galerkin=True)
        elif isinstance(matrix, np.ndarray):
            return DenseMatrixLinearOperator(domain, domain, matrix, galerkin=True)
        elif isinstance(matrix, ScipyLinOp):
            return MatrixLinearOperator(domain, domain, matrix, galerkin=True)
        elif sp.issparse(matrix):
            modern_array = sp.csr_array(matrix)
            return SparseMatrixLinearOperator(
                domain, domain, modern_array, galerkin=True
            )
        else:
            raise TypeError(f"Unsupported matrix type: {type(matrix)}")

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
                _n, k = xmat.shape
                assert _n == self.domain.dim

                if not parallel:
                    ymat = np.zeros((self.codomain.dim, k))
                    for j in range(k):
                        ymat[:, j] = matvec(xmat[:, j])
                    return ymat
                else:
                    result_cols = Parallel(n_jobs=n_jobs)(
                        delayed(matvec)(xmat[:, j]) for j in range(k)
                    )
                    return np.column_stack(result_cols)

            def rmatmat(ymat: np.ndarray) -> np.ndarray:
                _m, k = ymat.shape
                assert _m == self.codomain.dim

                if not parallel:
                    xmat = np.zeros((self.domain.dim, k))
                    for j in range(k):
                        xmat[:, j] = rmatvec(ymat[:, j])
                    return xmat
                else:
                    result_cols = Parallel(n_jobs=n_jobs)(
                        delayed(rmatvec)(ymat[:, j]) for j in range(k)
                    )
                    return np.column_stack(result_cols)

            return ScipyLinOp(
                (self.codomain.dim, self.domain.dim),
                matvec=matvec,
                rmatvec=rmatvec,
                matmat=matmat,
                rmatmat=rmatmat,
            )

    def extract_diagonal(
        self,
        /,
        *,
        galerkin: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """
        Computes the main diagonal of the operator's matrix representation.

        This method is highly parallelizable and memory-efficient, as it
        avoids forming the full dense matrix.

        Args:
            galerkin: If True, computes the diagonal of the Galerkin matrix.
            parallel: If True, computes the entries in parallel.
            n_jobs: Number of parallel jobs to use.

        Returns:
            A NumPy array containing the diagonal entries.
        """

        dim = min(self.domain.dim, self.codomain.dim)
        jobs = n_jobs if parallel else 1

        def compute_entry(i: int) -> float:
            """Worker function to compute a single diagonal entry."""
            e_i = self.domain.basis_vector(i)
            L_e_i = self(e_i)

            if galerkin:
                return self.domain.inner_product(e_i, L_e_i)
            else:
                return self.codomain.to_components(L_e_i)[i]

        diagonal_entries = Parallel(n_jobs=jobs)(
            delayed(compute_entry)(i) for i in range(dim)
        )
        return np.array(diagonal_entries)

    def extract_diagonals(
        self,
        offsets: List[int],
        /,
        *,
        galerkin: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Computes specified diagonals of the operator's matrix representation.

        This is a memory-efficient and parallelizable method that computes
        the matrix one column at a time.

        Args:
            offsets: A list of diagonal offsets to extract (e.g., [0] for
                the main diagonal, [-1, 0, 1] for a tridiagonal matrix).
            galerkin: If True, computes the diagonals of the Galerkin matrix.
            parallel: If True, computes columns in parallel.
            n_jobs: Number of parallel jobs to use.

        Returns:
            A tuple containing:
            - A NumPy array where each row is a diagonal.
            - The list of offsets.
            This format is compatible with scipy.sparse.spdiags.
        """
        dim = min(self.domain.dim, self.codomain.dim)
        jobs = n_jobs if parallel else 1

        # Prepare a thread-safe dictionary to store results

        results: Dict[int, Dict[int, float]] = defaultdict(dict)

        def compute_column_entries(j: int) -> Dict[int, Dict[int, float]]:
            """
            Worker function to compute all needed entries for column j.
            """
            e_j = self.domain.basis_vector(j)
            L_e_j = self(e_j)

            col_results = defaultdict(dict)

            for k in offsets:
                i = j - k
                if 0 <= i < dim:
                    if galerkin:
                        e_i = self.domain.basis_vector(i)
                        val = self.domain.inner_product(e_i, L_e_j)
                    else:
                        val = self.codomain.to_components(L_e_j)[i]
                    col_results[k][i] = val
            return col_results

        # Run the computation in parallel
        column_data = Parallel(n_jobs=jobs)(
            delayed(compute_column_entries)(j) for j in range(dim)
        )

        # Aggregate results from the parallel computation
        for col_dict in column_data:
            for k, entries in col_dict.items():
                results[k].update(entries)

        # Format the results for spdiags
        # The array must have padding for shorter off-diagonals.
        diagonals_array = np.zeros((len(offsets), dim))
        for idx, k in enumerate(offsets):
            diag_entries = results[k]
            for i, val in diag_entries.items():
                j = i + k
                if 0 <= j < dim:
                    diagonals_array[idx, j] = val

        return diagonals_array, offsets

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
    ) -> Tuple[
        DenseMatrixLinearOperator,
        DiagonalSparseMatrixLinearOperator,
        DenseMatrixLinearOperator,
    ]:
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
            left (DenseMatrixLinearOperator): The left singular vector matrix.
            singular_values (DiagonalSparseMatrixLinearOperator): The singular values.
            right (DenseMatrixLinearOperator): The right singular vector matrix.

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
        diagonal = DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            euclidean, euclidean, singular_values
        )

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
    ) -> Tuple[DenseMatrixLinearOperator, DiagonalSparseMatrixLinearOperator]:
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
            expansion (DenseMatrixLinearOperator): Mapping from coefficients in eigen-basis to vectors.
            eigenvaluevalues (DiagonalSparseMatrixLinearOperator): The eigenvalues values.

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
        diagonal = DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            euclidean, euclidean, eigenvalues
        )

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
    ) -> DenseMatrixLinearOperator:
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
            factor (DenseMatrixLinearOperator): A linear operator from a Euclidean space
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


class MatrixLinearOperator(LinearOperator):
    """
    A sub-class of LinearOperator for which the operator's action is
    defined internally through its matrix representation.

    This matrix can be either a dense numpy matrix or a
    scipy LinearOperator.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        codomain: HilbertSpace,
        matrix: Union[np.ndarray, ScipyLinOp],
        /,
        *,
        galerkin=False,
    ):
        """
        Args:
            domain: The domain of the operator.
            codomain: The codomain of the operator.
            matrix: matrix representation of the linear operator in either standard
                 or Galerkin form.
            galerkin: If True, galerkin representation used. Default is false.
        """
        assert matrix.shape == (codomain.dim, domain.dim)

        self._matrix = matrix
        self._is_dense = isinstance(matrix, np.ndarray)
        self._galerkin = galerkin

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

            super().__init__(domain, codomain, mapping, adjoint_mapping=adjoint_mapping)

        else:

            def mapping(x: Any) -> Any:
                cx = domain.to_components(x)
                cy = matrix @ cx
                return codomain.from_components(cy)

            def dual_mapping(yp: Any) -> Any:
                cyp = codomain.dual.to_components(yp)
                cxp = matrix.T @ cyp
                return domain.dual.from_components(cxp)

            super().__init__(domain, codomain, mapping, dual_mapping=dual_mapping)

    @property
    def is_dense(self) -> bool:
        """
        Returns True if the matrix representation is stored internally in dense form.
        """
        return self._is_dense

    @property
    def is_galerkin(self) -> bool:
        """
        Returns True if the matrix representation is stored in Galerkin form.
        """
        return self._galerkin

    def _compute_dense_matrix(
        self, galerkin: bool, parallel: bool, n_jobs: int
    ) -> np.ndarray:
        """
        Overloaded method to efficiently compute the dense matrix.
        """

        if galerkin == self.is_galerkin and self.is_dense:
            return self._matrix
        else:
            return super()._compute_dense_matrix(galerkin, parallel, n_jobs)

    def extract_diagonal(
        self,
        /,
        *,
        galerkin: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """
        Overload for efficiency.
        """

        if galerkin == self.is_galerkin and self.is_dense:
            return self._matrix.diagonal()
        else:
            return super().extract_diagonal(
                galerkin=galerkin, parallel=parallel, n_jobs=n_jobs
            )

    def extract_diagonals(
        self,
        offsets: List[int],
        /,
        *,
        galerkin: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Overrides the base method for efficiency by extracting diagonals directly
        from the stored dense matrix when possible.
        """

        if self.is_dense and galerkin == self.is_galerkin:
            dim = self.domain.dim

            diagonals_array = np.zeros((len(offsets), dim))

            for i, k in enumerate(offsets):
                diag_k = np.diag(self._matrix, k=k)

                if k >= 0:
                    diagonals_array[i, k : k + len(diag_k)] = diag_k
                else:
                    diagonals_array[i, : len(diag_k)] = diag_k

            return diagonals_array, offsets

        else:
            return super().extract_diagonals(
                offsets, galerkin=galerkin, parallel=parallel, n_jobs=n_jobs
            )


class DenseMatrixLinearOperator(MatrixLinearOperator):
    """
    A specialisation of the MatrixLinearOperator class to instances where
    the matrix representation is always provided as a numpy array.

    This is a class provides some additional methods for component-wise access.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        codomain: HilbertSpace,
        matrix: np.ndarray,
        /,
        *,
        galerkin=False,
    ):
        """
        domain: The domain of the operator.
            codomain: The codomain of the operator.
            matrix: matrix representation of the linear operator in either standard
                 or Galerkin form.
            galerkin: If True, galerkin representation used. Default is false.
        """

        if not isinstance(matrix, np.ndarray):
            raise ValueError("Matrix must be input in dense form.")

        super().__init__(domain, codomain, matrix, galerkin=galerkin)

    @staticmethod
    def from_linear_operator(
        operator: LinearOperator,
        /,
        *,
        galerkin: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> DenseMatrixLinearOperator:
        """
        Converts a LinearOperator into a DenseMatrixLinearOperator by forming its dense matrix representation.

        Args:
            operator: The operator to be converted.
            galerkin: If True, the Galerkin representation is used. Default is False.
            parallel: If True, dense matrix calculation is done in parallel. Default is False.
            n_jobs: Number of jobs used for parallel calculations. Default is False.
        """

        if isinstance(operator, DenseMatrixLinearOperator):
            return operator

        domain = operator.domain
        codomain = operator.codomain

        matrix = operator.matrix(
            dense=True, galerkin=galerkin, parallel=parallel, n_jobs=n_jobs
        )

        return DenseMatrixLinearOperator(domain, codomain, matrix, galerkin=galerkin)

    def __getitem__(self, key: tuple[int, int] | int | slice) -> float | np.ndarray:
        """
        Provides direct, component-wise access to the underlying matrix.

        This allows for intuitive slicing and indexing, like `op[i, j]` or `op[0, :]`.
        Note: The access is on the stored matrix, which may be in either
        standard or Galerkin form depending on how the operator was initialized.
        """
        return self._matrix[key]


class SparseMatrixLinearOperator(MatrixLinearOperator):
    """
    A specialization for operators represented by a modern SciPy sparse array.

    This class requires a `scipy.sparse.sparray` object (e.g., csr_array)
    and provides optimized methods that delegate to efficient SciPy routines.

    Upon initialization, the internal array is converted to the CSR
    (Compressed Sparse Row) format to ensure consistently fast matrix-vector
    products and row-slicing operations.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        codomain: HilbertSpace,
        matrix: sp.sparray,
        /,
        *,
        galerkin: bool = False,
    ):
        """
        Args:
            domain: The domain of the operator.
            codomain: The codomain of the operator.
            matrix: The sparse array representation of the linear operator.
                    Must be a modern sparray object (e.g., csr_array).
            galerkin: If True, the matrix is in Galerkin form. Defaults to False.
        """
        # Strict check for the modern sparse array type
        if not isinstance(matrix, sp.sparray):
            raise TypeError(
                "Matrix must be a modern SciPy sparray object (e.g., csr_array)."
            )

        super().__init__(domain, codomain, matrix, galerkin=galerkin)
        self._matrix = self._matrix.asformat("csr")

    def __getitem__(self, key):
        """Provides direct component access using SciPy's sparse indexing."""
        return self._matrix[key]

    def _compute_dense_matrix(
        self, galerkin: bool, parallel: bool, n_jobs: int
    ) -> np.ndarray:
        """
        Overrides the base method to efficiently compute the dense matrix.
        """
        # ⚡️ Fast path: Use the highly optimized .toarray() method.
        if galerkin == self.is_galerkin:
            return self._matrix.toarray()

        # Fallback path for when a basis conversion is needed.
        else:
            return super()._compute_dense_matrix(galerkin, parallel, n_jobs)

    def extract_diagonal(
        self,
        /,
        *,
        galerkin: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """
        Overrides the base method to efficiently extract the main diagonal.
        """
        if galerkin == self.is_galerkin:
            return self._matrix.diagonal(k=0)
        else:
            return super().extract_diagonal(
                galerkin=galerkin, parallel=parallel, n_jobs=n_jobs
            )

    def extract_diagonals(
        self,
        offsets: List[int],
        /,
        *,
        galerkin: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Overrides the base method for efficiency by extracting diagonals
        directly from the stored sparse array.
        """
        if galerkin != self.is_galerkin:
            return super().extract_diagonals(
                offsets, galerkin=galerkin, parallel=parallel, n_jobs=n_jobs
            )

        dim = self.domain.dim
        diagonals_array = np.zeros((len(offsets), dim))

        for i, k in enumerate(offsets):
            # Use the sparse array's fast .diagonal() method
            diag_k = self._matrix.diagonal(k=k)

            # Place the raw diagonal into the padded output array
            if k >= 0:
                diagonals_array[i, k : k + len(diag_k)] = diag_k
            else:
                diagonals_array[i, : len(diag_k)] = diag_k

        return diagonals_array, offsets


class DiagonalSparseMatrixLinearOperator(SparseMatrixLinearOperator):
    """
    A highly specialized operator for matrices defined purely by a set of
    non-zero diagonals.

    This class internally stores the operator using a `scipy.sparse.dia_array`
    for maximum efficiency in storage and matrix-vector products. It provides
    extremely fast methods for extracting diagonals, as this is its native
    storage format.

    A key feature of this class is its support for **functional calculus**. It
    dynamically proxies element-wise mathematical functions (e.g., `.sqrt()`,
    `.log()`, `abs()`, `**`) to the underlying sparse array. For reasons of
    mathematical correctness, these operations are restricted to operators that
    are **strictly diagonal** (i.e., have only a non-zero main diagonal) and
    will raise a `NotImplementedError` otherwise.

    Aggregation methods that do not return a new operator (e.g., `.sum()`)
    are not restricted and can be used on any multi-diagonal operator.

    Class Methods
    -------------
    from_diagonal_values:
        Constructs a strictly diagonal operator from a 1D array of values.
    from_operator:
        Creates a diagonal approximation of another LinearOperator.

    Properties
    ----------
    offsets:
        The array of stored diagonal offsets.
    is_strictly_diagonal:
        True if the operator only has a non-zero main diagonal.
    inverse:
        The inverse of a strictly diagonal operator.
    sqrt:
        The square root of a strictly diagonal operator.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        codomain: HilbertSpace,
        diagonals: Tuple[np.ndarray, List[int]],
        /,
        *,
        galerkin: bool = False,
    ):
        """
        Args:
            domain: The domain of the operator.
            codomain: The codomain of the operator.
            diagonals: A tuple `(data, offsets)` where `data` is a 2D array
                       of diagonal values and `offsets` is a list of their
                       positions. This is the native format for a dia_array.
            galerkin: If True, the matrix is in Galerkin form. Defaults to False.
        """
        shape = (codomain.dim, domain.dim)
        dia_array = sp.dia_array(diagonals, shape=shape)

        MatrixLinearOperator.__init__(
            self, domain, codomain, dia_array, galerkin=galerkin
        )

    @classmethod
    def from_operator(
        cls, operator: LinearOperator, offsets: List[int], /, *, galerkin: bool = True
    ) -> DiagonalSparseMatrixLinearOperator:
        """
        Creates a diagonal approximation of another LinearOperator.

        This factory method works by calling the source operator's
        `.extract_diagonals()` method and using the result to construct a
        new, highly efficient DiagonalSparseMatrixLinearOperator.

        Args:
            operator: The source operator to approximate.
            offsets: The list of diagonal offsets to extract and keep.
            galerkin: Specifies which matrix representation to use.

        Returns:
            A new DiagonalSparseMatrixLinearOperator.
        """
        diagonals_data, extracted_offsets = operator.extract_diagonals(
            offsets, galerkin=galerkin
        )
        return cls(
            operator.domain,
            operator.codomain,
            (diagonals_data, extracted_offsets),
            galerkin=galerkin,
        )

    @classmethod
    def from_diagonal_values(
        cls,
        domain: HilbertSpace,
        codomain: HilbertSpace,
        diagonal_values: np.ndarray,
        /,
        *,
        galerkin: bool = False,
    ) -> "DiagonalSparseMatrixLinearOperator":
        """
        Constructs a purely diagonal operator from a 1D array of values.

        This provides a convenient way to create an operator with non-zero
        entries only on its main diagonal (offset k=0).

        Args:
            domain: The domain of the operator.
            codomain: The codomain of the operator. Must have the same dimension.
            diagonal_values: A 1D NumPy array of the values for the main diagonal.
            galerkin: If True, the operator is in Galerkin form.

        Returns:
            A new DiagonalSparseMatrixLinearOperator.
        """
        if domain.dim != codomain.dim or domain.dim != len(diagonal_values):
            raise ValueError(
                "Domain, codomain, and diagonal_values must all have the same dimension."
            )

        # Reshape the 1D array of values into the 2D `data` array format
        diagonals_data = diagonal_values.reshape(1, -1)
        offsets = [0]

        return cls(domain, codomain, (diagonals_data, offsets), galerkin=galerkin)

    @property
    def offsets(self) -> np.ndarray:
        """Returns the array of stored diagonal offsets."""
        return self._matrix.offsets

    @property
    def is_strictly_diagonal(self) -> bool:
        """
        True if the operator only has a non-zero main diagonal (offset=0).
        """
        return len(self.offsets) == 1 and self.offsets[0] == 0

    @property
    def inverse(self) -> "DiagonalSparseMatrixLinearOperator":
        """
        The inverse of the operator, computed via functional calculus.
        Requires the operator to be strictly diagonal with no zero entries.
        """
        if not self.is_strictly_diagonal:
            raise NotImplementedError(
                "Inverse is only implemented for strictly diagonal operators."
            )

        if np.any(self._matrix.diagonal(k=0) == 0):
            raise ValueError("Cannot invert an operator with zeros on the diagonal.")

        return self**-1

    @property
    def sqrt(self) -> "DiagonalSparseMatrixLinearOperator":
        """
        The square root of the operator, computed via functional calculus.
        Requires the operator to be strictly diagonal with non-negative entries.
        """

        if np.any(self._matrix.data < 0):
            raise ValueError(
                "Cannot take the square root of an operator with negative entries."
            )

        return self.__getattr__("sqrt")()

    def extract_diagonals(
        self,
        offsets: List[int],
        /,
        *,
        galerkin: bool = True,
        # parallel and n_jobs are ignored but kept for signature consistency
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Overrides the base method for extreme efficiency.

        This operation is nearly free, as it involves selecting the requested
        diagonals from the data already stored in the native format.
        """
        if galerkin != self.is_galerkin:
            return super().extract_diagonals(offsets, galerkin=galerkin)

        # Create a result array and fill it with the requested stored diagonals
        result_diagonals = np.zeros((len(offsets), self.domain.dim))

        # Create a mapping from stored offset to its data row for quick lookup
        stored_diagonals = dict(zip(self.offsets, self._matrix.data))

        for i, k in enumerate(offsets):
            if k in stored_diagonals:
                result_diagonals[i, :] = stored_diagonals[k]

        return result_diagonals, offsets

    def __getattr__(self, name: str):
        """
        Dynamically proxies method calls to the underlying dia_array.

        For element-wise mathematical functions that return a new operator,
        this method enforces that the operator must be strictly diagonal.
        """
        attr = getattr(self._matrix, name)

        if callable(attr):

            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)

                if isinstance(result, sp.sparray):
                    if not self.is_strictly_diagonal:
                        raise NotImplementedError(
                            f"Element-wise function '{name}' is only defined for "
                            "strictly diagonal operators."
                        )

                    return DiagonalSparseMatrixLinearOperator(
                        self.domain,
                        self.codomain,
                        (result.data, result.offsets),
                        galerkin=self.is_galerkin,
                    )
                else:
                    return result

            return wrapper
        else:
            return attr

    def __abs__(self):
        """Explicitly handle the built-in abs() function."""
        return self.__getattr__("__abs__")()

    def __pow__(self, power):
        """Explicitly handle the power operator (**)."""
        return self.__getattr__("__pow__")(power)


class NormalSumOperator(LinearOperator):
    """
    Represents a self-adjoint operator of the form N = A @ Q @ A.adjoint + B.

    The operators Q and B are expected to be self-adjoint for the resulting
    operator to be mathematically correct.

    Q and B are optional. If Q is None, it defaults to the identity operator.
    If B is None, it defaults to the zero operator.

    This class uses operator algebra for a concise definition and provides an
    optimized, parallelizable method for computing its dense Galerkin matrix.
    """

    def __init__(
        self,
        A: LinearOperator,
        Q: Optional[LinearOperator] = None,
        B: Optional[LinearOperator] = None,
    ) -> None:

        op_domain = A.codomain

        if Q is None:
            Q = A.domain.identity_operator()

        if B is None:
            B = op_domain.zero_operator()

        if A.domain != Q.domain:
            raise ValueError("The domain of A must match the domain of Q.")
        if op_domain != B.domain:
            raise ValueError("The domain of B must match the codomain of A.")

        self._A = A
        self._Q = Q
        self._B = B

        composite_op = self._A @ self._Q @ self._A.adjoint + self._B

        super().__init__(
            composite_op.domain,
            composite_op.codomain,
            composite_op,
            adjoint_mapping=composite_op,
        )

    def _compute_dense_matrix(
        self, galerkin: bool, parallel: bool, n_jobs: int
    ) -> np.ndarray:
        """
        Overloaded method using the matrix-free approach for Q and a cleaner
        implementation leveraging the base class's methods.
        """
        if not galerkin:
            return super()._compute_dense_matrix(galerkin, parallel, n_jobs)

        domain_Y = self._A.codomain
        dim = self.domain.dim
        jobs = n_jobs if parallel else 1

        a_star_mat = self._A.adjoint.matrix(
            dense=True, galerkin=False, parallel=parallel, n_jobs=n_jobs
        )

        v_vectors = [domain_Y.from_components(a_star_mat[:, j]) for j in range(dim)]
        w_vectors = Parallel(n_jobs=jobs)(delayed(self._Q)(v_j) for v_j in v_vectors)

        def compute_row(i: int) -> np.ndarray:
            """Computes the i-th row of the inner product matrix."""
            v_i = v_vectors[i]
            return np.array([domain_Y.inner_product(v_i, w_j) for w_j in w_vectors])

        rows = Parallel(n_jobs=jobs)(delayed(compute_row)(i) for i in range(dim))
        m_aqa_mat = np.vstack(rows)

        b_mat = self._B.matrix(
            dense=True, galerkin=True, parallel=parallel, n_jobs=n_jobs
        )

        return m_aqa_mat + b_mat

    def extract_diagonal(
        self,
        /,
        *,
        galerkin: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """Overrides base method for efficiency."""
        if not galerkin:
            return super().extract_diagonal(
                galerkin=galerkin, parallel=parallel, n_jobs=n_jobs
            )

        diag_B = self._B.extract_diagonal(
            galerkin=True, parallel=parallel, n_jobs=n_jobs
        )

        dim = self.domain.dim
        jobs = n_jobs if parallel else 1

        def compute_entry(i: int) -> float:
            e_i = self.domain.basis_vector(i)
            v_i = self._A.adjoint(e_i)
            w_i = self._Q(v_i)
            return self._A.domain.inner_product(v_i, w_i)

        diag_AQA_T = Parallel(n_jobs=jobs)(
            delayed(compute_entry)(i) for i in range(dim)
        )

        return np.array(diag_AQA_T) + diag_B

    def extract_diagonals(
        self,
        offsets: List[int],
        /,
        *,
        galerkin: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> Tuple[np.ndarray, List[int]]:
        """Overrides base method for efficiency."""
        if not galerkin:
            return super().extract_diagonals(
                offsets, galerkin=galerkin, parallel=parallel, n_jobs=n_jobs
            )

        diagonals_B, _ = self._B.extract_diagonals(
            offsets, galerkin=True, parallel=parallel, n_jobs=n_jobs
        )

        dim = self.domain.dim
        jobs = n_jobs if parallel else 1

        # Pre-compute A*e_i for all i
        v_vectors = Parallel(n_jobs=jobs)(
            delayed(self._A.adjoint)(self.domain.basis_vector(i)) for i in range(dim)
        )

        def compute_column_entries(j: int) -> Dict[int, Dict[int, float]]:
            col_results = defaultdict(dict)
            v_j = v_vectors[j]
            w_j = self._Q(v_j)

            for k in offsets:
                i = j - k
                if 0 <= i < dim:
                    v_i = v_vectors[i]
                    val = self._A.domain.inner_product(v_i, w_j)
                    col_results[k][i] = val
            return col_results

        column_data = Parallel(n_jobs=jobs)(
            delayed(compute_column_entries)(j) for j in range(dim)
        )

        results: Dict[int, Dict[int, float]] = defaultdict(dict)
        for col_dict in column_data:
            for k, entries in col_dict.items():
                results[k].update(entries)

        diagonals_array = np.zeros((len(offsets), dim))
        for idx, k in enumerate(offsets):
            diag_entries = results[k]
            for i, val in diag_entries.items():
                j = i + k
                if 0 <= j < dim:
                    diagonals_array[idx, j] = val

        return diagonals_array + diagonals_B, offsets
