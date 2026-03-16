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
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve, eigh
from scipy.sparse.linalg import gmres, bicgstab, cg, bicg

from .linear_operators import LinearOperator
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


class EigenSolver(DirectLinearSolver):
    """
    A direct linear solver based on the eigendecomposition of a symmetric operator.

    This solver is robust for symmetric operators that may be singular or
    numerically ill-conditioned. In such cases, it computes a pseudo-inverse by
    regularizing the eigenvalues, treating those close to zero (relative to the largest
    eigenvalue) as exactly zero.
    """

    def __init__(
        self,
        /,
        *,
        galerkin: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
        rtol: float = 1e-12,
    ) -> None:
        """
        Args:
            galerkin (bool): If True, the Galerkin matrix representation is used.
            parallel (bool): If True, parallel computation is used.
            n_jobs (int): Number of parallel jobs.
            rtol (float): Relative tolerance for treating eigenvalues as zero.
                An eigenvalue `s` is treated as zero if
                `abs(s) < rtol * max(abs(eigenvalues))`.
        """
        super().__init__(galerkin=galerkin, parallel=parallel, n_jobs=n_jobs)
        self._rtol = rtol

    def __call__(self, operator: LinearOperator) -> LinearOperator:
        """
        Computes the pseudo-inverse of a self-adjoint LinearOperator.
        """
        assert operator.is_automorphism

        matrix = operator.matrix(
            dense=True,
            galerkin=self._galerkin,
            parallel=self._parallel,
            n_jobs=self._n_jobs,
        )

        eigenvalues, eigenvectors = eigh(matrix)

        max_abs_eigenvalue = np.max(np.abs(eigenvalues))
        if max_abs_eigenvalue > 0:
            threshold = self._rtol * max_abs_eigenvalue
        else:
            threshold = 0

        inv_eigenvalues = np.where(
            np.abs(eigenvalues) > threshold,
            np.reciprocal(eigenvalues),
            0.0,
        )

        def matvec(cy: np.ndarray) -> np.ndarray:
            z = eigenvectors.T @ cy
            w = inv_eigenvalues * z
            return eigenvectors @ w

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

    def __init__(self, /, *, preconditioning_method: LinearSolver = None) -> None:
        """
        Args:
            preconditioning_method: A LinearSolver from which to generate a preconditioner
                once the operator is known.

        Notes:
            If a preconditioner is provided to either the call or solve_linear_system
            methods, then it takes precedence over the preconditioning method.
        """
        self._preconditioning_method = preconditioning_method
        self._iterations: int = 0

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

    @property
    def iterations(self) -> int:
        """
        Returns the number of iterations within the last solve.
        The value is zero if the solver has yet to be called.
        """
        return self._iterations

    def solve_adjoint_linear_system(
        self,
        operator: LinearOperator,
        adjoint_preconditioner: Optional[LinearOperator],
        x: Vector,
        y0: Optional[Vector],
    ) -> Vector:
        """
        Solves the adjoint linear system A*y = x for y.
        """
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

        if preconditioner is None:
            if self._preconditioning_method is None:
                _preconditioner = None
                _adjoint_preconditions = None
            else:
                _preconditioner = self._preconditioning_method(operator)
        else:
            _preconditioner = preconditioner

        if _preconditioner is None:
            _adjoint_preconditioner = None
        else:
            _adjoint_preconditioner = _preconditioner.adjoint

        return LinearOperator(
            operator.codomain,
            operator.domain,
            lambda y: self.solve_linear_system(operator, _preconditioner, y, None),
            adjoint_mapping=lambda x: self.solve_adjoint_linear_system(
                operator, _adjoint_preconditioner, x, None
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
        preconditioning_method: LinearSolver = None,
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

        super().__init__(preconditioning_method=preconditioning_method)

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
        self._iterations = 0
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

        # Set up the iteration counter
        self._iterations = 0
        user_callback = self._solver_kwargs.get("callback", None)

        def iteration_counter(*args):
            self._iterations += 1
            if user_callback is not None:
                user_callback(*args)

        solver_kwargs = self._solver_kwargs.copy()
        solver_kwargs["callback"] = iteration_counter

        if self._solver_func is gmres:
            solver_kwargs.setdefault("callback_type", "pr_norm")

        cxp, _ = self._solver_func(
            matrix, cy, x0=cx0, M=matrix_preconditioner, **solver_kwargs
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
        preconditioning_method: LinearSolver = None,
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

        super().__init__(preconditioning_method=preconditioning_method)

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
        self._iterations = 0
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

        for k in range(maxiter):
            # Check for convergence
            if domain.squared_norm(r) <= tol_sq:
                self._iterations = k + 1
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

        else:
            self._iterations = maxiter

        return x


class MinResSolver(IterativeLinearSolver):
    """
    A matrix-free implementation of the MINRES algorithm.

    Suitable for symmetric, possibly indefinite or singular linear systems.
    It minimizes the norm of the residual ||r|| in each step using the
    Hilbert space's native inner product.
    """

    def __init__(
        self,
        /,
        *,
        preconditioning_method: LinearSolver = None,
        rtol: float = 1.0e-5,
        atol: float = 1.0e-8,
        maxiter: Optional[int] = None,
    ) -> None:
        super().__init__(preconditioning_method=preconditioning_method)
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter

    def solve_linear_system(
        self,
        operator: LinearOperator,
        preconditioner: Optional[LinearOperator],
        y: Vector,
        x0: Optional[Vector],
    ) -> Vector:
        self._iterations = 0
        domain = operator.domain

        x = domain.zero if x0 is None else domain.copy(x0)

        # r1 is the UNPRECONDITIONED residual
        r1 = domain.subtract(y, operator(x))
        r2 = domain.copy(r1)

        # y_vec is the PRECONDITIONED residual
        y_vec = domain.copy(r1) if preconditioner is None else preconditioner(r1)

        # beta1 is the M-norm of the initial residual
        beta1 = domain.inner_product(r1, y_vec)
        if beta1 < 0:
            raise ValueError("Preconditioner is not positive definite.")
        if beta1 == 0:
            return x

        beta1 = np.sqrt(beta1)

        # Initial Lanczos vectors
        oldb = 0.0
        beta = beta1

        # We need w vectors for the solution update (same as your w_curr, w_prev)
        w = domain.zero
        w2 = domain.zero

        # Givens rotation variables
        dbar = 0.0
        epsln = 0.0
        phibar = beta1
        cs = -1.0
        sn = 0.0

        maxiter = self._maxiter if self._maxiter is not None else 10 * domain.dim

        for k in range(1, maxiter + 1):
            s = 1.0 / beta
            v = domain.multiply(s, y_vec)  # v is the normalized preconditioned vector

            # 1. Apply Operator (Unpreconditioned)
            Av = operator(v)

            # 2. Lanczos Orthogonalization (on the unpreconditioned vectors)
            if k >= 2:
                # Av = Av - (beta / oldb) * r1
                domain.axpy(-(beta / oldb), r1, Av)

            alfa = domain.inner_product(v, Av)

            # Av = Av - (alfa / beta) * r2
            domain.axpy(-(alfa / beta), r2, Av)

            # Shift the old unpreconditioned vectors
            r1 = r2
            r2 = Av  # r2 is now the new unpreconditioned Lanczos vector

            # 3. Apply Preconditioner
            y_vec = domain.copy(r2) if preconditioner is None else preconditioner(r2)

            # 4. Calculate the new beta (M-norm)
            oldb = beta
            beta_sq = domain.inner_product(r2, y_vec)
            if beta_sq < 0:
                raise ValueError("Preconditioner is not positive definite.")
            beta = np.sqrt(beta_sq)

            # --- Givens Rotations (Exactly as in SciPy/MATLAB) ---
            oldeps = epsln
            delta = cs * dbar + sn * alfa
            gbar = sn * dbar - cs * alfa
            epsln = sn * beta
            dbar = -cs * beta

            # Compute next rotation
            gamma = max(np.linalg.norm([gbar, beta]), 1e-15)
            cs = gbar / gamma
            sn = beta / gamma
            phi = cs * phibar
            phibar = sn * phibar

            # --- Update Solution ---
            denom = 1.0 / gamma
            w1 = w2
            w2 = w

            # w = (v - oldeps*w1 - delta*w2) * denom
            w_new = domain.copy(v)
            domain.axpy(-oldeps, w1, w_new)
            domain.axpy(-delta, w2, w_new)
            domain.ax(denom, w_new)  # scale in place
            w = w_new

            # x = x + phi * w
            domain.axpy(phi, w, x)

            # --- Convergence Check ---
            if abs(phibar) < self._rtol * beta1 or abs(phibar) < self._atol:
                self._iterations = k
                break
        else:
            self._iterations = maxiter

        return x


class BICGStabSolver(IterativeLinearSolver):
    """
    A matrix-free implementation of the BiCGStab algorithm.

    Suitable for non-symmetric linear systems Ax = y. It operates directly
    on Hilbert space vectors using native inner products and arithmetic.
    """

    def __init__(
        self,
        /,
        *,
        preconditioning_method: LinearSolver = None,
        rtol: float = 1.0e-5,
        atol: float = 1.0e-8,
        maxiter: Optional[int] = None,
    ) -> None:
        super().__init__(preconditioning_method=preconditioning_method)
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter

    def solve_linear_system(
        self,
        operator: LinearOperator,
        preconditioner: Optional[LinearOperator],
        y: Vector,
        x0: Optional[Vector],
    ) -> Vector:
        self._iterations = 0
        domain = operator.domain

        x = domain.zero if x0 is None else domain.copy(x0)

        # Initial residual r = y - Ax
        r = domain.subtract(y, operator(x))
        r_hat = domain.copy(r)  # shadow residual

        norm_y = domain.norm(y)
        if norm_y == 0.0:
            return x

        # Tolerance: max(atol, rtol * norm(y)) to match SciPy logic
        atol = max(self._atol, self._rtol * norm_y)

        # Initialize dummy variables
        rho_prev = 1.0
        omega = 1.0
        alpha = 1.0

        v = domain.zero
        p = domain.zero

        maxiter = self._maxiter if self._maxiter is not None else 10 * domain.dim

        for k in range(maxiter):
            # 1. Early convergence check
            if domain.norm(r) < atol:
                self._iterations = k
                return x

            # 2. rho = <r_hat, r>
            rho = domain.inner_product(r_hat, r)
            if abs(rho) < 1e-16:  # breakdown
                self._iterations = k
                break

            # 3. Update search direction p
            if k > 0:
                if abs(omega) < 1e-16:  # breakdown
                    self._iterations = k
                    break

                beta = (rho / rho_prev) * (alpha / omega)

                # In-place equivalent of: p = r + beta * (p - omega * v)
                domain.axpy(-omega, v, p)  # p = p - omega * v
                domain.ax(beta, p)  # p = beta * p
                domain.axpy(1.0, r, p)  # p = p + r
            else:
                p = domain.copy(r)

            # 4. phat = M^-1 p
            phat = domain.copy(p) if preconditioner is None else preconditioner(p)

            # 5. v = A phat
            v = operator(phat)

            # 6. alpha = rho / <r_hat, v>
            rv = domain.inner_product(r_hat, v)
            if abs(rv) < 1e-16:  # breakdown
                self._iterations = k
                break
            alpha = rho / rv

            # 7. Update r to act as 's' (s = r - alpha * v)
            domain.axpy(-alpha, v, r)

            # 8. Early exit check on 's' (which is currently stored in r)
            if domain.norm(r) < atol:
                domain.axpy(alpha, phat, x)
                self._iterations = k + 1
                return x

            # 9. shat = M^-1 s
            shat = domain.copy(r) if preconditioner is None else preconditioner(r)

            # 10. t = A shat
            t = operator(shat)

            # 11. omega = <t, s> / <t, t>
            omega = domain.inner_product(t, r) / domain.inner_product(t, t)

            # 12. Update x = x + alpha * phat + omega * shat
            domain.axpy(alpha, phat, x)
            domain.axpy(omega, shat, x)

            # 13. Update r to true next residual (r = s - omega * t)
            domain.axpy(-omega, t, r)

            rho_prev = rho

        else:
            self._iterations = maxiter

        return x


class LSQRSolver(IterativeLinearSolver):
    """
    A matrix-free implementation of the LSQR algorithm with damping support.

    This solver is designed to solve the problem: minimize ||Ax - y||_2^2 + damping^2 * ||x||_2^2.
    """

    def __init__(
        self,
        /,
        *,
        rtol: float = 1.0e-5,
        atol: float = 1.0e-8,
        maxiter: Optional[int] = None,
    ) -> None:
        super().__init__(preconditioning_method=None)
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter

    def solve_linear_system(
        self,
        operator: LinearOperator,
        preconditioner: Optional[LinearOperator],
        y: Vector,
        x0: Optional[Vector],
        damping: float = 0.0,
    ) -> Vector:
        self._iterations = 0
        domain = operator.domain
        codomain = operator.codomain

        # Initial Setup
        x = domain.zero if x0 is None else domain.copy(x0)

        # u = y - A x
        u = codomain.subtract(y, operator(x))
        norm_y = codomain.norm(y)

        beta = codomain.norm(u)
        if beta > 0:
            codomain.ax(1.0 / beta, u)  # in-place scale

        # v = A* u
        v = operator.adjoint(u)
        alfa = domain.norm(v)
        if alfa > 0:
            domain.ax(1.0 / alfa, v)  # in-place scale

        w = domain.copy(v)

        # Variables for QR rotations and norms
        rhobar = alfa
        phibar = beta
        dampsq = damping**2

        # Residual tracking variables
        res2 = 0.0

        # Tolerances
        atol = self._atol
        rtol = self._rtol
        btol = atol + rtol * norm_y

        maxiter = (
            self._maxiter
            if self._maxiter is not None
            else 2 * max(domain.dim, codomain.dim)
        )

        for k in range(maxiter):
            # --- 1. Bidiagonalization Step ---

            # u = A v - alfa * u
            Av = operator(v)
            codomain.ax(-alfa, u)  # u = -alfa * u
            codomain.axpy(1.0, Av, u)  # u = u + Av
            beta = codomain.norm(u)
            if beta > 0:
                codomain.ax(1.0 / beta, u)

            # v = A* u - beta * v
            A_u = operator.adjoint(u)
            domain.ax(-beta, v)  # v = -beta * v
            domain.axpy(1.0, A_u, v)  # v = v + A* u
            alfa = domain.norm(v)
            if alfa > 0:
                domain.ax(1.0 / alfa, v)

            # --- 2. Plane Rotation for Damping ---
            if damping > 0:
                rhobar1 = np.sqrt(rhobar**2 + dampsq)
                cs1 = rhobar / rhobar1
                sn1 = damping / rhobar1
                psi = sn1 * phibar
                phibar = cs1 * phibar
            else:
                rhobar1 = rhobar
                psi = 0.0

            # --- 3. Plane Rotation for Subdiagonal (SciPy's sym_ortho equivalent) ---
            # This stable rotation is crucial for LSQR's numerical stability
            rho = np.sqrt(rhobar1**2 + beta**2)
            cs = rhobar1 / rho
            sn = beta / rho

            theta = sn * alfa
            rhobar = -cs * alfa
            phi = cs * phibar
            phibar = sn * phibar
            tau = sn * phi

            # --- 4. Update Solution and Search Direction ---
            t1 = phi / rho
            t2 = -theta / rho

            # x = x + t1 * w
            domain.axpy(t1, w, x)

            # w = v + t2 * w
            domain.ax(t2, w)  # w = t2 * w
            domain.axpy(1.0, v, w)  # w = w + v

            # --- 5. Convergence Check ---
            # Estimate the true residual norm accurately without calculating A*x
            res1 = phibar**2
            res2 = res2 + psi**2
            rnorm = np.sqrt(res1 + res2)

            arnorm = alfa * abs(tau)

            # Stopping criteria aligned with SciPy
            if rnorm <= btol:
                self._iterations = k + 1
                break

            # If the least-squares gradient is flat, we've found the LS minimum
            if arnorm <= atol:
                self._iterations = k + 1
                break

        else:
            self._iterations = maxiter

        return x


class FCGSolver(IterativeLinearSolver):
    """
    Flexible Conjugate Gradient (FCG) solver.

    FCG is designed to handle variable preconditioning, such as using an
    inner iterative solver to approximate the action of M^-1.
    """

    def __init__(
        self,
        /,
        *,
        rtol: float = 1.0e-5,
        atol: float = 1.0e-8,
        maxiter: Optional[int] = None,
        preconditioning_method: Optional[LinearSolver] = None,
    ) -> None:
        super().__init__(preconditioning_method=preconditioning_method)
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter

    def solve_linear_system(
        self,
        operator: LinearOperator,
        preconditioner: Optional[LinearOperator],
        y: Vector,
        x0: Optional[Vector],
    ) -> Vector:
        space = operator.domain
        x = space.zero if x0 is None else space.copy(x0)

        # Initial residual: r = y - Ax
        r = space.subtract(y, operator(x))
        norm_y = space.norm(y)

        # Default to identity if no preconditioner exists
        if preconditioner is None:
            preconditioner = space.identity_operator()

        # Initial preconditioned residual z_0 = M^-1 r_0
        z = preconditioner(r)
        p = space.copy(z)

        # Initial r.z product
        rz = space.inner_product(r, z)

        maxiter = self._maxiter if self._maxiter is not None else 2 * space.dim

        for k in range(maxiter):
            # w = A p
            ap = operator(p)
            pap = space.inner_product(p, ap)

            # Step size alpha = (r, z) / (p, Ap)
            alpha = rz / pap

            # Update solution: x = x + alpha * p
            space.axpy(alpha, p, x)

            # Update residual: r = r - alpha * ap
            space.axpy(-alpha, ap, r)

            # Convergence check
            if space.norm(r) < self._atol + self._rtol * norm_y:
                break

            # Flexible Beta update: Beta = - (z_new, Ap) / (p, Ap)
            # This ensures that p_new is A-orthogonal to p_old
            z_new = preconditioner(r)
            beta = -space.inner_product(z_new, ap) / pap

            # Update search direction: p = z_new + beta * p
            p = space.add(z_new, space.multiply(beta, p))

            # Prepare for next iteration
            z = z_new
            rz = space.inner_product(r, z)

        return x
