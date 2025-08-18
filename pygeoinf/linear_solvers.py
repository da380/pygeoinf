"""
Module for linear solvers.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable

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
from .hilbert_space import T_vec


class LinearSolver(ABC):
    """
    An abstract base class for linear solvers.
    """


class DirectLinearSolver(LinearSolver):
    """
    An abstract base class for direct linear solvers that rely on matrix
    factorization.
    """


class LUSolver(DirectLinearSolver):
    """
    A direct linear solver based on the LU decomposition of an operator's
    dense matrix representation.
    """

    def __init__(self, /, *, galerkin: bool = False) -> None:
        """
        Args:
            galerkin (bool): If True, the Galerkin matrix representation is used.
        """
        self._galerkin: bool = galerkin

    def __call__(self, operator: LinearOperator) -> LinearOperator:
        """
        Computes the inverse of a LinearOperator.

        Args:
            operator (LinearOperator): The operator to be inverted.

        Returns:
            LinearOperator: A new operator representing the inverse.
        """
        assert operator.is_square

        matrix = operator.matrix(dense=True, galerkin=self._galerkin)
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

    def __init__(self, /, *, galerkin: bool = False) -> None:
        """
        Args:
            galerkin (bool): If True, the Galerkin matrix representation is used.
        """
        self._galerkin: bool = galerkin

    def __call__(self, operator: LinearOperator) -> LinearOperator:
        """
        Computes the inverse of a self-adjoint LinearOperator.

        Args:
            operator (LinearOperator): The self-adjoint operator to be inverted.

        Returns:
            LinearOperator: A new operator representing the inverse.
        """
        assert operator.is_automorphism

        matrix = operator.matrix(dense=True, galerkin=self._galerkin)
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
        y: T_vec,
        x0: Optional[T_vec],
    ) -> T_vec:
        """
        Solves the linear system Ax = y for x.

        Args:
            operator (LinearOperator): The operator A of the linear system.
            preconditioner (LinearOperator, optional): The preconditioner.
            y (T_vec): The right-hand side vector.
            x0 (T_vec, optional): The initial guess for the solution.

        Returns:
            T_vec: The solution vector x.
        """

    def solve_adjoint_linear_system(
        self,
        operator: LinearOperator,
        preconditioner: Optional[LinearOperator],
        x: T_vec,
        y0: Optional[T_vec],
    ) -> T_vec:
        """
        Solves the adjoint linear system A*y = x for y.
        """
        # Note: Preconditioner is not used for adjoint solve in this default impl.
        return self.solve_linear_system(operator.adjoint, None, x, y0)

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


class CGMatrixSolver(IterativeLinearSolver):
    """
    Iterative solver using SciPy's Conjugate Gradient (CG) algorithm on the
    operator's matrix representation. Assumes the operator is self-adjoint.
    """

    def __init__(
        self,
        /,
        *,
        galerkin: bool = False,
        rtol: float = 1.0e-5,
        atol: float = 0.0,
        maxiter: Optional[int] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        """
        Args:
            galerkin (bool): If True, use the Galerkin matrix representation.
            rtol (float): Relative tolerance for convergence.
            atol (float): Absolute tolerance for convergence.
            maxiter (int, optional): Maximum number of iterations.
            callback (callable, optional): User-supplied function to call
                after each iteration.
        """
        self._galerkin: bool = galerkin
        self._rtol: float = rtol
        self._atol: float = atol
        self._maxiter: Optional[int] = maxiter
        self._callback: Optional[Callable[[np.ndarray], None]] = callback

    def solve_linear_system(
        self,
        operator: LinearOperator,
        preconditioner: Optional[LinearOperator],
        y: T_vec,
        x0: Optional[T_vec],
    ) -> T_vec:
        domain = operator.codomain
        matrix = operator.matrix(galerkin=self._galerkin)

        matrix_preconditioner = (
            None
            if preconditioner is None
            else preconditioner.matrix(galerkin=self._galerkin)
        )

        cx0 = None if x0 is None else domain.to_components(x0)
        cy = domain.to_components(y)

        cxp, _ = cg(
            matrix,
            cy,
            x0=cx0,
            rtol=self._rtol,
            atol=self._atol,
            maxiter=self._maxiter,
            M=matrix_preconditioner,
            callback=self._callback,
        )
        if self._galerkin:
            xp = domain.dual.from_components(cxp)
            return domain.from_dual(xp)
        else:
            return domain.from_components(cxp)


class BICGMatrixSolver(IterativeLinearSolver):
    """
    Iterative solver using SciPy's Biconjugate Gradient (BiCG) algorithm on
    the operator's matrix representation. For general square operators.
    """

    def __init__(
        self,
        /,
        *,
        galerkin: bool = False,
        rtol: float = 1.0e-5,
        atol: float = 0.0,
        maxiter: Optional[int] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        """
        Args:
            galerkin (bool): If True, use the Galerkin matrix representation.
            rtol (float): Relative tolerance for convergence.
            atol (float): Absolute tolerance for convergence.
            maxiter (int, optional): Maximum number of iterations.
            callback (callable, optional): User-supplied function to call
                after each iteration.
        """
        self._galerkin: bool = galerkin
        self._rtol: float = rtol
        self._atol: float = atol
        self._maxiter: Optional[int] = maxiter
        self._callback: Optional[Callable[[np.ndarray], None]] = callback

    def solve_linear_system(
        self,
        operator: LinearOperator,
        preconditioner: Optional[LinearOperator],
        y: T_vec,
        x0: Optional[T_vec],
    ) -> T_vec:
        domain = operator.codomain
        codomain = operator.domain
        matrix = operator.matrix(galerkin=self._galerkin)

        matrix_preconditioner = (
            None
            if preconditioner is None
            else preconditioner.matrix(galerkin=self._galerkin)
        )

        cx0 = None if x0 is None else domain.to_components(x0)
        cy = domain.to_components(y)

        cxp, _ = bicg(
            matrix,
            cy,
            x0=cx0,
            rtol=self._rtol,
            atol=self._atol,
            maxiter=self._maxiter,
            M=matrix_preconditioner,
            callback=self._callback,
        )
        if self._galerkin:
            xp = codomain.dual.from_components(cxp)
            return codomain.from_dual(xp)
        else:
            return codomain.from_components(cxp)


class BICGStabMatrixSolver(IterativeLinearSolver):
    """
    Iterative solver using SciPy's Biconjugate Gradient Stabilized (BiCGSTAB)
    algorithm on the operator's matrix representation. For general square operators.
    """

    def __init__(
        self,
        /,
        *,
        galerkin: bool = False,
        rtol: float = 1.0e-5,
        atol: float = 0.0,
        maxiter: Optional[int] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        """
        Args:
            galerkin (bool): If True, use the Galerkin matrix representation.
            rtol (float): Relative tolerance for convergence.
            atol (float): Absolute tolerance for convergence.
            maxiter (int, optional): Maximum number of iterations.
            callback (callable, optional): User-supplied function to call
                after each iteration.
        """
        self._galerkin: bool = galerkin
        self._rtol: float = rtol
        self._atol: float = atol
        self._maxiter: Optional[int] = maxiter
        self._callback: Optional[Callable[[np.ndarray], None]] = callback

    def solve_linear_system(
        self,
        operator: LinearOperator,
        preconditioner: Optional[LinearOperator],
        y: T_vec,
        x0: Optional[T_vec],
    ) -> T_vec:
        domain = operator.codomain
        codomain = operator.domain
        matrix = operator.matrix(galerkin=self._galerkin)

        matrix_preconditioner = (
            None
            if preconditioner is None
            else preconditioner.matrix(galerkin=self._galerkin)
        )

        cx0 = None if x0 is None else domain.to_components(x0)
        cy = domain.to_components(y)

        cxp, _ = bicgstab(
            matrix,
            cy,
            x0=cx0,
            rtol=self._rtol,
            atol=self._atol,
            maxiter=self._maxiter,
            M=matrix_preconditioner,
            callback=self._callback,
        )
        if self._galerkin:
            xp = codomain.dual.from_components(cxp)
            return codomain.from_dual(xp)
        else:
            return codomain.from_components(cxp)


class GMRESMatrixSolver(IterativeLinearSolver):
    """
    Iterative solver using SciPy's Generalized Minimal Residual (GMRES)
    algorithm on the operator's matrix representation. For general square operators.
    """

    def __init__(
        self,
        /,
        *,
        galerkin: bool = False,
        rtol: float = 1.0e-5,
        atol: float = 0.0,
        restart: Optional[int] = None,
        maxiter: Optional[int] = None,
        callback: Optional[Callable] = None,
        callback_type: Optional[str] = None,
    ) -> None:
        """
        Args:
            galerkin (bool): If True, use the Galerkin matrix representation.
            rtol (float): Relative tolerance for convergence.
            atol (float): Absolute tolerance for convergence.
            restart (int, optional): Number of iterations between restarts.
            maxiter (int, optional): Maximum number of iterations.
            callback (callable, optional): User-supplied function to call
                during iterations.
            callback_type (str, optional): Type of callback ("x", "pr_norm").
        """
        self._galerkin: bool = galerkin
        self._rtol: float = rtol
        self._atol: float = atol
        self._restart: Optional[int] = restart
        self._maxiter: Optional[int] = maxiter
        self._callback: Optional[Callable] = callback
        self._callback_type: Optional[str] = callback_type

    def solve_linear_system(
        self,
        operator: LinearOperator,
        preconditioner: Optional[LinearOperator],
        y: T_vec,
        x0: Optional[T_vec],
    ) -> T_vec:
        domain = operator.codomain
        codomain = operator.domain
        matrix = operator.matrix(galerkin=self._galerkin)

        matrix_preconditioner = (
            None
            if preconditioner is None
            else preconditioner.matrix(galerkin=self._galerkin)
        )

        cx0 = None if x0 is None else domain.to_components(x0)
        cy = domain.to_components(y)

        cxp, _ = gmres(
            matrix,
            cy,
            x0=cx0,
            rtol=self._rtol,
            atol=self._atol,
            restart=self._restart,
            maxiter=self._maxiter,
            M=matrix_preconditioner,
            callback=self._callback,
            callback_type=self._callback_type,
        )

        if self._galerkin:
            xp = codomain.dual.from_components(cxp)
            return codomain.from_dual(xp)
        else:
            return codomain.from_components(cxp)


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
        callback: Optional[Callable[[T_vec], None]] = None,
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

        self._callback: Optional[Callable[[T_vec], None]] = callback

    def solve_linear_system(
        self,
        operator: LinearOperator,
        preconditioner: Optional[LinearOperator],
        y: T_vec,
        x0: Optional[T_vec],
    ) -> T_vec:
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
