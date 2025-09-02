"""
Provides a collection of solvers for linear systems of equations.

This module offers a unified interface for solving linear systems `A(x) = y`,
where `A` is a `LinearOperator`. It includes both direct methods based on
matrix factorization and iterative, matrix-free methods suitable for large-scale
problems.

The solvers are implemented as callable classes. An instance of a solver can
be called with an operator to produce a new operator representing its inverse.

Key Classes
-----------
- `LUSolver`, `CholeskySolver`: Direct solvers based on matrix factorization.
- `ScipyIterativeSolver`: A general wrapper for SciPy's iterative algorithms
  (CG, GMRES, etc.) that operate on matrix representations.
- `CGSolver`: A pure, matrix-free implementation of the Conjugate Gradient
  algorithm that operates directly on abstract Hilbert space vectors.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any

import numpy as np
from scipy.sparse.linalg import LinearOperator as ScipyLinOp
from scipy.linalg import (
    cho_factor,
    cho_solve,
    lu_factor,
    lu_solve,
)
from scipy.sparse.linalg import gmres, bicgstab, cg, bicg

from .operators import LinearOperator
from .hilbert_space import Vector


class LinearSolver(ABC):
    """
    An abstract base class for linear solvers.
    """


class DirectLinearSolver(LinearSolver):
    """
    An abstract base class for direct linear solvers that rely on matrix
    factorization.
    """

    def __init__(
        self, /, *, galerkin: bool = False, parallel: bool = False, n_jobs: int = -1
    ):
        """
        Args:
            galerkin (bool): If True, the Galerkin matrix representation is used.
            parallel (bool): If True, parallel computation is used.
            n_jobs (int): Number of parallel jobs.
        """
        self._galerkin: bool = galerkin
        self._parallel: bool = parallel
        self._n_jobs: int = n_jobs


class LUSolver(DirectLinearSolver):
    """
    A direct linear solver based on the LU decomposition of an operator's
    dense matrix representation.
    """

    def __init__(
        self, /, *, galerkin: bool = False, parallel: bool = False, n_jobs: int = -1
    ) -> None:
        """
        Args:
            galerkin (bool): If True, the Galerkin matrix representation is used.
            parallel (bool): If True, parallel computation is used.
            n_jobs (int): Number of parallel jobs.
        """
        super().__init__(galerkin=galerkin, parallel=parallel, n_jobs=n_jobs)

    def __call__(self, operator: LinearOperator) -> LinearOperator:
        """
        Computes the inverse of a LinearOperator.

        Args:
            operator (LinearOperator): The operator to be inverted.

        Returns:
            LinearOperator: A new operator representing the inverse.
        """
        assert operator.is_square

        matrix = operator.matrix(
            dense=True,
            galerkin=self._galerkin,
            parallel=self._parallel,
            n_jobs=self._n_jobs,
        )
        factor = lu_factor(matrix, overwrite_a=True)

        def matvec(cy: np.ndarray) -> np.ndarray:
            return lu_solve(factor, cy, 0)

        def rmatvec(cx: np.ndarray) -> np.ndarray:
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
    A direct linear solver based on Cholesky decomposition.

    It is assumed that the operator is self-adjoint and its matrix
    representation is positive-definite.
    """

    def __init__(
        self, /, *, galerkin: bool = False, parallel: bool = False, n_jobs: int = -1
    ) -> None:
        """
        Args:
            galerkin (bool): If True, the Galerkin matrix representation is used.
            parallel (bool): If True, parallel computation is used.
            n_jobs (int): Number of parallel jobs.
        """
        super().__init__(galerkin=galerkin, parallel=parallel, n_jobs=n_jobs)

    def __call__(self, operator: LinearOperator) -> LinearOperator:
        """
        Computes the inverse of a self-adjoint LinearOperator.

        Args:
            operator (LinearOperator): The self-adjoint operator to be inverted.

        Returns:
            LinearOperator: A new operator representing the inverse.
        """
        assert operator.is_automorphism

        matrix = operator.matrix(
            dense=True,
            galerkin=self._galerkin,
            parallel=self._parallel,
            n_jobs=self._n_jobs,
        )
        factor = cho_factor(matrix, overwrite_a=False)

        def matvec(cy: np.ndarray) -> np.ndarray:
            return cho_solve(factor, cy)

        inverse_matrix = ScipyLinOp(
            (operator.domain.dim, operator.codomain.dim), matvec=matvec, rmatvec=matvec
        )

        return LinearOperator.from_matrix(
            operator.domain, operator.domain, inverse_matrix, galerkin=self._galerkin
        )


class IterativeLinearSolver(LinearSolver):
    """
    An abstract base class for iterative linear solvers.
    """

    @abstractmethod
    def solve_linear_system(
        self,
        operator: LinearOperator,
        preconditioner: Optional[LinearOperator],
        y: Vector,
        x0: Optional[Vector],
    ) -> Vector:
        """
        Solves the linear system Ax = y for x.

        Args:
            operator (LinearOperator): The operator A of the linear system.
            preconditioner (LinearOperator, optional): The preconditioner.
            y (Vector): The right-hand side vector.
            x0 (Vector, optional): The initial guess for the solution.

        Returns:
            Vector: The solution vector x.
        """

    def solve_adjoint_linear_system(
        self,
        operator: LinearOperator,
        preconditioner: Optional[LinearOperator],
        x: Vector,
        y0: Optional[Vector],
    ) -> Vector:
        """
        Solves the adjoint linear system A*y = x for y.
        """
        adjoint_preconditioner = (
            None if preconditioner is None else preconditioner.adjoint
        )
        return self.solve_linear_system(operator.adjoint, adjoint_preconditioner, x, y0)

    def __call__(
        self,
        operator: LinearOperator,
        /,
        *,
        preconditioner: Optional[LinearOperator] = None,
    ) -> LinearOperator:
        """
        Creates an operator representing the inverse of the input operator.

        Args:
            operator (LinearOperator): The operator to be inverted.
            preconditioner (LinearOperator, optional): A preconditioner to
                accelerate convergence.

        Returns:
            LinearOperator: A new operator that applies the inverse of the
                original operator.
        """
        assert operator.is_automorphism
        return LinearOperator(
            operator.codomain,
            operator.domain,
            lambda y: self.solve_linear_system(operator, preconditioner, y, None),
            adjoint_mapping=lambda x: self.solve_adjoint_linear_system(
                operator, preconditioner, x, None
            ),
        )


class ScipyIterativeSolver(IterativeLinearSolver):
    """
    A general iterative solver that wraps SciPy's iterative algorithms.

    This class provides a unified interface to SciPy's sparse iterative
    solvers like `cg`, `gmres`, `bicgstab`, etc. The specific algorithm is chosen
    during instantiation, and keyword arguments are passed directly to the
    chosen SciPy function.
    """

    _SOLVER_MAP = {
        "cg": cg,
        "bicg": bicg,
        "bicgstab": bicgstab,
        "gmres": gmres,
    }

    def __init__(
        self,
        method: str,
        /,
        *,
        galerkin: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            method (str): The name of the SciPy solver to use (e.g., 'cg', 'gmres').
            galerkin (bool): If True, use the Galerkin matrix representation.
            **kwargs: Keyword arguments to be passed directly to the SciPy solver
                (e.g., rtol, atol, maxiter, restart).
        """
        if method not in self._SOLVER_MAP:
            raise ValueError(
                f"Unknown solver method '{method}'. Available methods: {list(self._SOLVER_MAP.keys())}"
            )

        self._solver_func = self._SOLVER_MAP[method]
        self._galerkin: bool = galerkin
        self._solver_kwargs: Dict[str, Any] = kwargs

    def solve_linear_system(
        self,
        operator: LinearOperator,
        preconditioner: Optional[LinearOperator],
        y: Vector,
        x0: Optional[Vector],
    ) -> Vector:
        domain = operator.codomain
        codomain = operator.domain

        matrix = operator.matrix(galerkin=self._galerkin)
        matrix_preconditioner = (
            None
            if preconditioner is None
            else preconditioner.matrix(galerkin=self._galerkin)
        )

        cy = domain.to_components(y)
        cx0 = None if x0 is None else domain.to_components(x0)

        cxp, _ = self._solver_func(
            matrix,
            cy,
            x0=cx0,
            M=matrix_preconditioner,
            **self._solver_kwargs,
        )

        if self._galerkin:
            xp = codomain.dual.from_components(cxp)
            return codomain.from_dual(xp)
        else:
            return codomain.from_components(cxp)


def CGMatrixSolver(galerkin: bool = False, **kwargs) -> ScipyIterativeSolver:
    return ScipyIterativeSolver("cg", galerkin=galerkin, **kwargs)


def BICGMatrixSolver(galerkin: bool = False, **kwargs) -> ScipyIterativeSolver:
    return ScipyIterativeSolver("bicg", galerkin=galerkin, **kwargs)


def BICGStabMatrixSolver(galerkin: bool = False, **kwargs) -> ScipyIterativeSolver:
    return ScipyIterativeSolver("bicgstab", galerkin=galerkin, **kwargs)


def GMRESMatrixSolver(galerkin: bool = False, **kwargs) -> ScipyIterativeSolver:
    return ScipyIterativeSolver("gmres", galerkin=galerkin, **kwargs)


class CGSolver(IterativeLinearSolver):
    """
    A matrix-free implementation of the Conjugate Gradient (CG) algorithm.

    This solver operates directly on Hilbert space vectors and operator actions
    without explicitly forming a matrix. It is suitable for self-adjoint,
    positive-definite operators on a general Hilbert space.
    """

    def __init__(
        self,
        /,
        *,
        rtol: float = 1.0e-5,
        atol: float = 0.0,
        maxiter: Optional[int] = None,
        callback: Optional[Callable[[Vector], None]] = None,
    ) -> None:
        """
        Args:
            rtol (float): Relative tolerance for convergence.
            atol (float): Absolute tolerance for convergence.
            maxiter (int, optional): Maximum number of iterations.
            callback (callable, optional): User-supplied function to call
                after each iteration with the current solution vector.
        """
        if not rtol > 0:
            raise ValueError("rtol must be positive")
        self._rtol: float = rtol

        if not atol >= 0:
            raise ValueError("atol must be non-negative!")
        self._atol: float = atol

        if maxiter is not None and not maxiter >= 0:
            raise ValueError("maxiter must be None or positive")
        self._maxiter: Optional[int] = maxiter

        self._callback: Optional[Callable[[Vector], None]] = callback

    def solve_linear_system(
        self,
        operator: LinearOperator,
        preconditioner: Optional[LinearOperator],
        y: Vector,
        x0: Optional[Vector],
    ) -> Vector:
        domain = operator.domain
        x = domain.zero if x0 is None else domain.copy(x0)

        r = domain.subtract(y, operator(x))
        z = domain.copy(r) if preconditioner is None else preconditioner(r)
        p = domain.copy(z)

        y_squared_norm = domain.squared_norm(y)
        # If RHS is zero, solution is zero
        if y_squared_norm == 0.0:
            return domain.zero

        # Determine tolerance
        tol_sq = max(self._atol**2, (self._rtol**2) * y_squared_norm)

        maxiter = self._maxiter if self._maxiter is not None else 10 * domain.dim

        num = domain.inner_product(r, z)

        for _ in range(maxiter):
            # Check for convergence
            if domain.squared_norm(r) <= tol_sq:
                break

            q = operator(p)
            den = domain.inner_product(p, q)
            alpha = num / den

            domain.axpy(alpha, p, x)
            domain.axpy(-alpha, q, r)

            if preconditioner is None:
                z = domain.copy(r)
            else:
                z = preconditioner(r)

            den = num
            num = operator.domain.inner_product(r, z)
            beta = num / den

            # p = z + beta * p
            domain.ax(beta, p)
            domain.axpy(1.0, z, p)

            if self._callback is not None:
                self._callback(x)

        return x
