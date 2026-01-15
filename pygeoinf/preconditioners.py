from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import numpy as np

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
    """

    def __init__(
        self,
        damping: float,
        rank: int = 20,
        power: int = 2,
    ) -> None:
        self._damping = damping
        self._rank = rank
        self._power = power

    def __call__(self, operator: LinearOperator) -> LinearOperator:
        """
        Generates a spectral preconditioner.
        Note: This assumes the operator provided is the data-misfit operator A*WA.
        """
        space = operator.domain

        # Use randomized eigendecomposition to find dominant modes
        U, S = operator.random_eig(self._rank, power=self._power)

        s_vals = S.extract_diagonal()
        d_vals = s_vals / (s_vals + self._damping**2)

        def mapping(r: Vector) -> Vector:
            ut_r = U.adjoint(r)
            d_ut_r = d_vals * ut_r
            correction = U(d_ut_r)

            diff = space.subtract(r, correction)
            return space.multiply(1.0 / self._damping**2, diff)

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
