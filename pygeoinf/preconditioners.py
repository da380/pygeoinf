from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from joblib import Parallel, delayed

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as splinalg


from .linear_operators import LinearOperator, DiagonalSparseMatrixLinearOperator
from .linear_solvers import LinearSolver, IterativeLinearSolver
from .random_matrix import random_diagonal

if TYPE_CHECKING:
    from .hilbert_space import Vector




class IdentityPreconditioningMethod(LinearSolver):
    """
    A trivial preconditioning method that returns the Identity operator.

    This acts as a "no-op" placeholder in the preconditioning framework,
    useful for benchmarking or default configurations.
    """

    def __call__(self, operator: LinearOperator) -> LinearOperator:
        """
        Returns the identity operator for the domain of the input operator.
        """
        return operator.domain.identity_operator()


class JacobiPreconditioningMethod(LinearSolver):
    """
    A LinearSolver wrapper that generates a Jacobi preconditioner.
    """

    def __init__(
        self,
        num_samples: Optional[int] = 20,
        method: str = "variable",
        rtol: float = 1e-2,
        block_size: int = 10,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> None:
        # Damping is removed: the operator passed to __call__ is already damped
        self._num_samples = num_samples
        self._method = method
        self._rtol = rtol
        self._block_size = block_size
        self._parallel = parallel
        self._n_jobs = n_jobs

    def __call__(self, operator: LinearOperator) -> LinearOperator:
        # Hutchinson's method or exact extraction on the damped normal operator
        if self._num_samples is not None:
            diag_values = random_diagonal(
                operator.matrix(galerkin=True),
                self._num_samples,
                method=self._method,
                rtol=self._rtol,
                block_size=self._block_size,
                parallel=self._parallel,
                n_jobs=self._n_jobs,
            )
        else:
            diag_values = operator.extract_diagonal(
                galerkin=True, parallel=self._parallel, n_jobs=self._n_jobs
            )

        inv_diag = np.where(np.abs(diag_values) > 1e-14, 1.0 / diag_values, 1.0)

        return DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            operator.domain, operator.domain, inv_diag, galerkin=True
        )


class SpectralPreconditioningMethod(LinearSolver):
    """
    A LinearSolver wrapper that generates a spectral (low-rank) preconditioner.

    This preconditioner uses a randomized eigendecomposition to invert the dominant
    modes of the operator. The unresolved tail is regularized using a damping parameter.
    """

    def __init__(
        self,
        /,
        *,
        damping: Optional[float] = None,
        rank: int = 20,
        method: str = "variable",
        max_rank: Optional[int] = None,
        power: int = 2,
        rtol: float = 1e-4,
        block_size: int = 10,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> None:
        """
        Args:
            damping: The scalar damping parameter. If None, it is estimated
                heuristically from the smallest resolved eigenvalue.
            rank: For 'fixed' method, the exact target rank. For 'variable'
                method, this is the initial rank to sample.
            method ({'variable', 'fixed'}): The randomized algorithm to use.
            max_rank: A hard limit on the rank for the 'variable' method.
            power: Number of power iterations to improve spectral accuracy.
            rtol: Relative tolerance for the 'variable' method.
            block_size: Number of new vectors to sample per iteration ('variable').
            parallel: Whether to use parallel matrix multiplication.
            n_jobs: Number of jobs for parallelism.
        """
        self._damping = damping
        self._rank = rank
        self._method = method
        self._max_rank = max_rank
        self._power = power
        self._rtol = rtol
        self._block_size = block_size
        self._parallel = parallel
        self._n_jobs = n_jobs

    def __call__(self, operator: LinearOperator) -> LinearOperator:
        """
        Generates a spectral preconditioner.
        Note: This assumes the operator provided is the data-misfit operator A*WA.
        """
        space = operator.domain

        # Use randomized eigendecomposition with full parameter control
        U, S = operator.random_eig(
            self._rank,
            method=self._method,
            max_rank=self._max_rank,
            power=self._power,
            rtol=self._rtol,
            block_size=self._block_size,
            parallel=self._parallel,
            n_jobs=self._n_jobs,
        )

        s_vals = S.extract_diagonal()

        # Heuristic: If damping is not provided, use the smallest resolved
        # eigenvalue as a proxy for the unresolved spectral tail.
        if self._damping is None:
            # Add a safety floor relative to the max eigenvalue to prevent div-by-zero
            d_sq = max(s_vals[-1], s_vals[0] * 1e-8)
        else:
            d_sq = self._damping**2

        d_vals = s_vals / (s_vals + d_sq)

        def mapping(r: Vector) -> Vector:
            ut_r = U.adjoint(r)
            d_ut_r = d_vals * ut_r
            correction = U(d_ut_r)

            diff = space.subtract(r, correction)
            return space.multiply(1.0 / d_sq, diff)

        return LinearOperator(space, space, mapping, adjoint_mapping=mapping)


class IterativePreconditioningMethod(LinearSolver):
    """
    Wraps an iterative solver to act as a preconditioner.

    This is best used with FCGSolver to handle the potential
    variability of the inner iterations.
    """

    def __init__(
        self,
        inner_solver: IterativeLinearSolver,
        max_inner_iter: int = 5,
        rtol: float = 1e-1,
    ) -> None:
        self._inner_solver = inner_solver
        self._max_iter = max_inner_iter
        self._rtol = rtol

    def __call__(self, operator: LinearOperator) -> LinearOperator:
        """
        Returns a LinearOperator whose action is 'solve the system'.
        """
        # We override the inner solver parameters for efficiency
        self._inner_solver._maxiter = self._max_iter
        self._inner_solver._rtol = self._rtol

        # The solver's __call__ returns the InverseLinearOperator
        return self._inner_solver(operator)


class BandedPreconditioningMethod(LinearSolver):
    """
    A LinearSolver wrapper that generates a symmetrically banded sparse preconditioner.

    Extracts a symmetric band of diagonals from the operator's Galerkin
    matrix representation, constructs a sparse matrix, and uses a sparse
    direct solver (exact or incomplete LU) to invert it.
    """

    def __init__(
        self,
        bandwidth: int,
        /,
        *,
        incomplete: bool = False,
        drop_tol: float = 1e-4,
        fill_factor: float = 10.0,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> None:
        """
        Args:
            bandwidth: The number of sub/super-diagonals to include on each side
                of the main diagonal. For example, `1` creates a tridiagonal matrix.
            incomplete: If True, uses an Incomplete LU (ILU) factorization to
                save memory and time. If False, uses exact LU.
            drop_tol: For ILU, the relative tolerance for dropping small elements.
            fill_factor: For ILU, the maximum allowed ratio of non-zeros in the
                factorized matrix compared to the original sparse matrix.
            parallel: If True, computes the diagonals in parallel.
            n_jobs: Number of parallel jobs to use.
        """
        if bandwidth < 0:
            raise ValueError("Bandwidth must be a non-negative integer.")

        self._bandwidth = bandwidth
        # Generate the symmetric list of offsets: [-bandwidth, ..., 0, ..., bandwidth]
        self._offsets = list(range(-bandwidth, bandwidth + 1))

        self._incomplete = incomplete
        self._drop_tol = drop_tol
        self._fill_factor = fill_factor
        self._parallel = parallel
        self._n_jobs = n_jobs

    def __call__(self, operator: LinearOperator) -> LinearOperator:
        domain = operator.domain
        codomain = operator.codomain

        if domain.dim != codomain.dim:
            raise ValueError("Banded preconditioner requires a square operator.")

        data, extracted_offsets = operator.extract_diagonals(
            self._offsets, galerkin=True, parallel=self._parallel, n_jobs=self._n_jobs
        )

        sparse_matrix = sps.spdiags(
            data, extracted_offsets, domain.dim, codomain.dim, format="csc"
        )

        if self._incomplete:
            factorization = splinalg.spilu(
                sparse_matrix, drop_tol=self._drop_tol, fill_factor=self._fill_factor
            )
        else:
            factorization = splinalg.splu(sparse_matrix)

        def mapping(x: Vector) -> Vector:
            c = domain.to_components(x)
            c_solved = factorization.solve(c)
            return codomain.from_components(c_solved)

        def adjoint_mapping(y: Vector) -> Vector:
            c = codomain.to_components(y)
            c_solved = factorization.solve(c, trans="T")
            return domain.from_components(c_solved)

        return LinearOperator(
            codomain,
            domain,
            mapping,
            adjoint_mapping=adjoint_mapping,
        )


class ExactBlockPreconditioningMethod(LinearSolver):
    """
    A LinearSolver wrapper that generates a sparse block preconditioner
    using exact matrix-vector evaluations.

    Explicitly probes the operator with basis vectors but only retains the entries
    specified by the interaction blocks. Factorizes the resulting sparse matrix
    using exact or incomplete LU.
    """

    def __init__(
        self,
        blocks: list[list[int]],
        /,
        *,
        galerkin: bool = True,
        incomplete: bool = False,
        drop_tol: float = 1e-4,
        fill_factor: float = 10.0,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> None:
        """
        Args:
            blocks: A list of lists containing the interacting indices.
            galerkin: Whether to extract the entries using the Galerkin representation.
            incomplete: If True, uses an Incomplete LU (ILU) factorization.
            drop_tol: For ILU, the relative tolerance for dropping small elements.
            fill_factor: For ILU, the maximum allowed fill-in ratio.
            parallel: If True, evaluates the operator columns in parallel.
            n_jobs: Number of parallel jobs to use.
        """
        self._blocks = blocks
        self._galerkin = galerkin
        self._incomplete = incomplete
        self._drop_tol = drop_tol
        self._fill_factor = fill_factor
        self._parallel = parallel
        self._n_jobs = n_jobs

    def __call__(self, operator: LinearOperator) -> LinearOperator:
        domain = operator.domain
        codomain = operator.codomain

        if domain.dim != codomain.dim:
            raise ValueError(
                "ExactBlockPreconditioningMethod requires a square operator."
            )

        # Map each column to the exact set of rows we need to extract for it
        col_to_rows = {j: set() for block in self._blocks for j in block}
        for block in self._blocks:
            for j in block:
                col_to_rows[j].update(block)

        # Safety Check: fill missing indices with exact diagonal
        missing_indices = set(range(domain.dim)) - set(col_to_rows.keys())
        for j in missing_indices:
            col_to_rows[j] = {j}

        def _process_column(j: int):
            e_j = domain.basis_vector(j)
            L_e_j = operator(e_j)

            rows = list(col_to_rows[j])
            vals = []

            if self._galerkin:
                for i in rows:
                    e_i = codomain.basis_vector(i)
                    vals.append(codomain.inner_product(e_i, L_e_j))
            else:
                c_vec = codomain.to_components(L_e_j)
                for i in rows:
                    vals.append(c_vec[i])

            return rows, [j] * len(rows), vals

        if self._parallel:
            results = Parallel(n_jobs=self._n_jobs)(
                delayed(_process_column)(j) for j in col_to_rows.keys()
            )
        else:
            results = [_process_column(j) for j in col_to_rows.keys()]

        I_global, J_global, V_local = [], [], []
        for rows, cols, vals in results:
            I_global.extend(rows)
            J_global.extend(cols)
            V_local.extend(vals)

        sparse_matrix = sps.coo_matrix(
            (V_local, (I_global, J_global)), shape=(codomain.dim, domain.dim)
        ).tocsc()

        if self._incomplete:
            factorization = splinalg.spilu(
                sparse_matrix, drop_tol=self._drop_tol, fill_factor=self._fill_factor
            )
        else:
            factorization = splinalg.splu(sparse_matrix)

        def mapping(x: Vector) -> Vector:
            c = domain.to_components(x)
            c_solved = factorization.solve(c)
            return codomain.from_components(c_solved)

        def adjoint_mapping(y: Vector) -> Vector:
            c = codomain.to_components(y)
            c_solved = factorization.solve(c, trans="T")
            return domain.from_components(c_solved)

        return LinearOperator(
            codomain,
            domain,
            mapping,
            adjoint_mapping=adjoint_mapping,
        )
