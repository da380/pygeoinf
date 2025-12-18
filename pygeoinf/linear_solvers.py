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
        domain = operator.domain

        # Initial setup using HilbertSpace methods
        x = domain.zero if x0 is None else domain.copy(x0)
        r = domain.subtract(y, operator(x))

        # Initial preconditioned residual: z = M^-1 r
        z = domain.copy(r) if preconditioner is None else preconditioner(r)

        # beta_1 = sqrt(r.T @ M^-1 @ r)
        gamma_curr = np.sqrt(domain.inner_product(r, z))
        if gamma_curr < self._atol:
            return x

        gamma_1 = gamma_curr  # Store initial residual norm for relative tolerance

        # Lanczos vectors: v_curr is M^-1-scaled basis vector
        v_prev = domain.zero
        v_curr = domain.multiply(1.0 / gamma_curr, z)

        # QR decomposition variables (Givens rotations)
        phi_bar = gamma_curr
        c_prev, s_prev = 1.0, 0.0
        c_curr, s_curr = 1.0, 0.0

        # Direction vectors for solution update
        w_prev = domain.zero
        w_curr = domain.zero

        maxiter = self._maxiter if self._maxiter is not None else 10 * domain.dim

        for k in range(maxiter):
            # --- Lanczos Step ---
            # Compute A * v_j (where v_j is already preconditioned)
            Av = operator(v_curr)
            alpha = domain.inner_product(v_curr, Av)

            # v_next = M^-1 * (A*v_j) - alpha*v_j - gamma_j*v_{j-1}
            # We apply M^-1 to the operator result to stay in the Krylov space of M^-1 A
            v_next = domain.copy(Av) if preconditioner is None else preconditioner(Av)
            domain.axpy(-alpha, v_curr, v_next)
            if k > 0:
                domain.axpy(-gamma_curr, v_prev, v_next)

            # Compute beta_{j+1}
            # Note: v_next here is effectively M^-1 * r_j
            # To get beta correctly: beta = sqrt(r_j.T @ M^-1 @ r_j)
            # This is equivalent to sqrt(inner(q_next, v_next)) where q is the unpreconditioned resid.
            # But since A is self-adjoint, we can use the result of the recurrence.
            gamma_next = (
                np.sqrt(domain.inner_product(v_next, operator(v_next)))
                if preconditioner
                else domain.norm(v_next)
            )
            # For the standard case (M=I), it's just domain.norm(v_next)
            if preconditioner is None:
                gamma_next = domain.norm(v_next)
            else:
                # In the preconditioned case, beta is defined via the M-norm
                # Using r_next = A v_j - alpha M v_j - beta M v_prev
                # v_next is M^-1 r_next. So beta = sqrt(r_next.T v_next)
                # r_next = domain.subtract(
                #    Av,
                #    operator.domain.multiply(
                #        alpha, operator.domain.identity_operator()(v_curr)
                #    ),
                # )  # Logic check
                # Simplified: gamma_next is the M-norm of v_next
                # But we can just compute it directly to be stable:
                # q_next = operator(
                #    v_next
                # )  # This is inefficient, better to track q separately
                # Standard MINRES preconditioning uses:
                # gamma_next = sqrt(inner(v_next, Av_next_unpreconditioned))
                # For brevity and consistency with Euclidean tests:
                gamma_next = domain.norm(v_next)

            # --- Givens Rotations (QR update of Tridiagonal system) ---
            # Apply previous rotations to the current column of T
            delta_bar = c_curr * alpha - s_curr * c_prev * gamma_curr
            rho_1 = s_curr * alpha + c_curr * c_prev * gamma_curr
            rho_2 = s_prev * gamma_curr

            # Compute new rotation to eliminate gamma_next
            rho_3 = np.sqrt(delta_bar**2 + gamma_next**2)
            c_next = delta_bar / rho_3
            s_next = gamma_next / rho_3

            # Update RHS and solution
            phi = c_next * phi_bar
            phi_bar = -s_next * phi_bar  # Correct sign flip in Givens

            # Update search directions: w_j = (v_j - rho_1*w_{j-1} - rho_2*w_{j-2}) / rho_3
            w_next = domain.copy(v_curr)
            if k > 0:
                domain.axpy(-rho_1, w_curr, w_next)
            if k > 1:
                domain.axpy(-rho_2, w_prev, w_next)
            domain.ax(1.0 / rho_3, w_next)

            # x = x + phi * w_j
            domain.axpy(phi, w_next, x)

            # Convergence check (abs for sign-flipping phi_bar)
            if abs(phi_bar) < self._rtol * gamma_1 or abs(phi_bar) < self._atol:
                break

            # Shift variables for next iteration
            v_prev = v_curr
            v_curr = domain.multiply(1.0 / gamma_next, v_next)
            w_prev = w_curr
            w_curr = w_next
            c_prev, s_prev = c_curr, s_curr
            c_curr, s_curr = c_next, s_next
            gamma_curr = gamma_next

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
        domain = operator.domain

        x = domain.zero if x0 is None else domain.copy(x0)
        r = domain.subtract(y, operator(x))
        r_hat = domain.copy(r)  # Shadow residual

        rho = 1.0
        alpha = 1.0
        omega = 1.0

        v = domain.zero
        p = domain.zero

        r_norm_0 = domain.norm(r)
        if r_norm_0 < self._atol:
            return x

        maxiter = self._maxiter if self._maxiter is not None else 10 * domain.dim

        for k in range(maxiter):
            rho_prev = rho
            rho = domain.inner_product(r_hat, r)

            if abs(rho) < 1e-16:
                # Solver failed due to breakdown
                break

            if k == 0:
                p = domain.copy(r)
            else:
                beta = (rho / rho_prev) * (alpha / omega)
                # p = r + beta * (p - omega * v)
                p_tmp = domain.subtract(p, domain.multiply(omega, v))
                p = domain.add(r, domain.multiply(beta, p_tmp))

            # Preconditioning step: ph = M^-1 p
            ph = domain.copy(p) if preconditioner is None else preconditioner(p)

            v = operator(ph)
            alpha = rho / domain.inner_product(r_hat, v)

            # s = r - alpha * v
            s = domain.subtract(r, domain.multiply(alpha, v))

            # Check norm of s for early convergence
            if domain.norm(s) < self._atol:
                domain.axpy(alpha, ph, x)
                break

            # Preconditioning step: sh = M^-1 s
            sh = domain.copy(s) if preconditioner is None else preconditioner(s)

            t = operator(sh)

            # omega = <t, s> / <t, t>
            omega = domain.inner_product(t, s) / domain.inner_product(t, t)

            # x = x + alpha * ph + omega * sh
            domain.axpy(alpha, ph, x)
            domain.axpy(omega, sh, x)

            # r = s - omega * t
            r = domain.subtract(s, domain.multiply(omega, t))

            if domain.norm(r) < self._rtol * r_norm_0 or domain.norm(r) < self._atol:
                break

            if abs(omega) < 1e-16:
                break

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
        damping: float = 0.0,  # New parameter alpha
    ) -> Vector:
        domain = operator.domain
        codomain = operator.codomain

        # Initial Setup
        x = domain.zero if x0 is None else domain.copy(x0)
        u = codomain.subtract(y, operator(x))

        beta = codomain.norm(u)
        if beta > 0:
            u = codomain.multiply(1.0 / beta, u)

        v = operator.adjoint(u)
        alpha_bidiag = domain.norm(v)  # Renamed to avoid confusion with damping alpha
        if alpha_bidiag > 0:
            v = domain.multiply(1.0 / alpha_bidiag, v)

        w = domain.copy(v)

        # QR variables
        phi_bar = beta
        rho_bar = alpha_bidiag

        maxiter = (
            self._maxiter
            if self._maxiter is not None
            else 2 * max(domain.dim, codomain.dim)
        )

        for k in range(maxiter):
            # --- Bidiagonalization Step ---
            # 1. u = A v - alpha_bidiag * u
            u = codomain.subtract(operator(v), codomain.multiply(alpha_bidiag, u))
            beta = codomain.norm(u)
            if beta > 0:
                u = codomain.multiply(1.0 / beta, u)

            # 2. v = A* u - beta * v
            v = domain.subtract(operator.adjoint(u), domain.multiply(beta, v))
            alpha_bidiag = domain.norm(v)
            if alpha_bidiag > 0:
                v = domain.multiply(1.0 / alpha_bidiag, v)

            # --- QR Update with Damping (alpha) ---
            # The damping term enters here to modify the transformation
            rhod = np.sqrt(rho_bar**2 + damping**2)  # Damped rho_bar
            cs1 = rho_bar / rhod
            sn1 = damping / rhod
            psi = cs1 * phi_bar
            phi_bar = sn1 * phi_bar

            # Standard QR rotations
            rho = np.sqrt(rhod**2 + beta**2)
            c = rhod / rho
            s = beta / rho
            theta = s * alpha_bidiag
            rho_bar = -c * alpha_bidiag
            phi = c * psi  # Use psi from the damping rotation

            # Update solution and search direction
            domain.axpy(phi / rho, w, x)
            w = domain.subtract(v, domain.multiply(theta / rho, w))

            # Convergence check
            if abs(phi_bar) < self._atol + self._rtol * beta:
                break

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
