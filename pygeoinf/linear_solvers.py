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
from typing import Callable, Optional, Dict, Any, List

import numpy as np
from scipy.sparse.linalg import LinearOperator as ScipyLinOp
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve, eigh
from scipy.sparse.linalg import gmres, bicgstab, cg, bicg

from .linear_operators import LinearOperator
from .hilbert_space import Vector, HilbertSpace


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

    def _build_inverse_operator(
        self,
        operator: LinearOperator,
        solve_func: Callable[[np.ndarray], np.ndarray],
        solve_adj_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> LinearOperator:
        """Helper to wrap array-based solve functions into a LinearOperator."""
        if solve_adj_func is None:
            solve_adj_func = solve_func

        if self._galerkin:

            def mapping(y: Vector) -> Vector:
                yp = operator.codomain.to_dual(y)
                cyp = operator.codomain.dual.to_components(yp)
                cx = solve_func(cyp)
                return operator.domain.from_components(cx)

            def adjoint_mapping(x: Vector) -> Vector:
                xp = operator.domain.to_dual(x)
                cxp = operator.domain.dual.to_components(xp)
                cy = solve_adj_func(cxp)
                return operator.codomain.from_components(cy)

            return LinearOperator(
                operator.codomain,
                operator.domain,
                mapping,
                adjoint_mapping=adjoint_mapping,
            )
        else:
            inverse_matrix = ScipyLinOp(
                (operator.domain.dim, operator.codomain.dim),
                matvec=solve_func,
                rmatvec=solve_adj_func,
            )
            return LinearOperator.from_matrix(
                operator.codomain, operator.domain, inverse_matrix, galerkin=False
            )


class LUSolver(DirectLinearSolver):
    """
    A direct linear solver based on the LU decomposition of an operator's
    dense matrix representation.
    """

    def __init__(
        self, /, *, galerkin: bool = False, parallel: bool = False, n_jobs: int = -1
    ) -> None:
        super().__init__(galerkin=galerkin, parallel=parallel, n_jobs=n_jobs)

    def __call__(self, operator: LinearOperator) -> LinearOperator:
        assert operator.is_square

        matrix = operator.matrix(
            dense=True,
            galerkin=self._galerkin,
            parallel=self._parallel,
            n_jobs=self._n_jobs,
        )
        factor = lu_factor(matrix, overwrite_a=True)

        def apply_inv(c: np.ndarray) -> np.ndarray:
            return lu_solve(factor, c, 0)

        def apply_inv_adj(c: np.ndarray) -> np.ndarray:
            return lu_solve(factor, c, 1)

        return self._build_inverse_operator(operator, apply_inv, apply_inv_adj)


class CholeskySolver(DirectLinearSolver):
    """
    A direct linear solver based on Cholesky decomposition.
    """

    def __init__(
        self, /, *, galerkin: bool = False, parallel: bool = False, n_jobs: int = -1
    ) -> None:
        super().__init__(galerkin=galerkin, parallel=parallel, n_jobs=n_jobs)

    def __call__(self, operator: LinearOperator) -> LinearOperator:
        assert operator.is_automorphism

        matrix = operator.matrix(
            dense=True,
            galerkin=self._galerkin,
            parallel=self._parallel,
            n_jobs=self._n_jobs,
        )
        factor = cho_factor(matrix, overwrite_a=False)

        def solve_galerkin(c: np.ndarray) -> np.ndarray:
            return cho_solve(factor, c)

        return self._build_inverse_operator(operator, solve_galerkin)


class EigenSolver(DirectLinearSolver):
    """
    A direct linear solver based on the eigendecomposition of a symmetric operator.
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
        super().__init__(galerkin=galerkin, parallel=parallel, n_jobs=n_jobs)
        self._rtol = rtol

    def __call__(self, operator: LinearOperator) -> LinearOperator:
        assert operator.is_automorphism

        matrix = operator.matrix(
            dense=True,
            galerkin=self._galerkin,
            parallel=self._parallel,
            n_jobs=self._n_jobs,
        )

        eigenvalues, eigenvectors = eigh(matrix)
        max_abs_eigenvalue = np.max(np.abs(eigenvalues))
        threshold = self._rtol * max_abs_eigenvalue if max_abs_eigenvalue > 0 else 0

        inv_eigenvalues = np.where(
            np.abs(eigenvalues) > threshold,
            np.reciprocal(eigenvalues),
            0.0,
        )

        def solve_galerkin(cy: np.ndarray) -> np.ndarray:
            z = eigenvectors.T @ cy
            w = inv_eigenvalues * z
            return eigenvectors @ w

        return self._build_inverse_operator(operator, solve_galerkin)


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
        op_domain = operator.domain
        op_codomain = operator.codomain

        matrix = operator.matrix(galerkin=self._galerkin)
        matrix_preconditioner = (
            None
            if preconditioner is None
            else preconditioner.matrix(galerkin=self._galerkin)
        )

        if self._galerkin:
            yp = op_codomain.to_dual(y)
            rhs = op_codomain.dual.to_components(yp)
        else:
            rhs = op_codomain.to_components(y)

        cx0 = None if x0 is None else op_domain.to_components(x0)

        self._iterations = 0
        user_callback = self._solver_kwargs.get("callback", None)

        if hasattr(user_callback, "reset") and callable(user_callback.reset):
            user_callback.reset()

        def iteration_counter(*args):
            self._iterations += 1
            if user_callback is not None:
                user_callback(*args)

        solver_kwargs = self._solver_kwargs.copy()
        solver_kwargs["callback"] = iteration_counter

        if self._solver_func is gmres:
            solver_kwargs.setdefault("callback_type", "pr_norm")

        cx, _ = self._solver_func(
            matrix, rhs, x0=cx0, M=matrix_preconditioner, **solver_kwargs
        )

        if hasattr(user_callback, "finalize") and callable(user_callback.finalize):
            user_callback.finalize()

        return op_domain.from_components(cx)


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

        if hasattr(self._callback, "reset") and callable(self._callback.reset):
            self._callback.reset()

        domain = operator.domain
        x = domain.zero if x0 is None else domain.copy(x0)

        r = domain.subtract(y, operator(x))
        z = domain.copy(r) if preconditioner is None else preconditioner(r)
        p = domain.copy(z)

        y_squared_norm = domain.squared_norm(y)

        if y_squared_norm == 0.0:
            return domain.zero

        tol_sq = max(self._atol**2, (self._rtol**2) * y_squared_norm)

        maxiter = self._maxiter if self._maxiter is not None else 10 * domain.dim

        num = domain.inner_product(r, z)

        for k in range(maxiter):
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

            domain.ax(beta, p)
            domain.axpy(1.0, z, p)

            if self._callback is not None:
                self._callback(x)

        else:
            self._iterations = maxiter

        if hasattr(self._callback, "finalize") and callable(self._callback.finalize):
            self._callback.finalize()

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

        r1 = domain.subtract(y, operator(x))
        r2 = domain.copy(r1)

        y_vec = domain.copy(r1) if preconditioner is None else preconditioner(r1)

        beta1 = domain.inner_product(r1, y_vec)
        if beta1 < 0:
            raise ValueError("Preconditioner is not positive definite.")
        if beta1 == 0:
            return x

        beta1 = np.sqrt(beta1)

        oldb = 0.0
        beta = beta1

        w = domain.zero
        w2 = domain.zero

        dbar = 0.0
        epsln = 0.0
        phibar = beta1
        cs = -1.0
        sn = 0.0

        maxiter = self._maxiter if self._maxiter is not None else 10 * domain.dim

        for k in range(1, maxiter + 1):
            s = 1.0 / beta
            v = domain.multiply(s, y_vec)

            Av = operator(v)

            if k >= 2:
                domain.axpy(-(beta / oldb), r1, Av)

            alfa = domain.inner_product(v, Av)

            domain.axpy(-(alfa / beta), r2, Av)

            r1 = r2
            r2 = Av

            y_vec = domain.copy(r2) if preconditioner is None else preconditioner(r2)

            oldb = beta
            beta_sq = domain.inner_product(r2, y_vec)
            if beta_sq < 0:
                raise ValueError("Preconditioner is not positive definite.")
            beta = np.sqrt(beta_sq)

            oldeps = epsln
            delta = cs * dbar + sn * alfa
            gbar = sn * dbar - cs * alfa
            epsln = sn * beta
            dbar = -cs * beta

            gamma = max(np.linalg.norm([gbar, beta]), 1e-15)
            cs = gbar / gamma
            sn = beta / gamma
            phi = cs * phibar
            phibar = sn * phibar

            denom = 1.0 / gamma
            w1 = w2
            w2 = w

            w_new = domain.copy(v)
            domain.axpy(-oldeps, w1, w_new)
            domain.axpy(-delta, w2, w_new)
            domain.ax(denom, w_new)
            w = w_new

            domain.axpy(phi, w, x)

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

        r = domain.subtract(y, operator(x))
        r_hat = domain.copy(r)

        norm_y = domain.norm(y)
        if norm_y == 0.0:
            return x

        atol = max(self._atol, self._rtol * norm_y)

        rho_prev = 1.0
        omega = 1.0
        alpha = 1.0

        v = domain.zero
        p = domain.zero

        maxiter = self._maxiter if self._maxiter is not None else 10 * domain.dim

        for k in range(maxiter):
            if domain.norm(r) < atol:
                self._iterations = k
                return x

            rho = domain.inner_product(r_hat, r)
            if abs(rho) < 1e-16:
                self._iterations = k
                break

            if k > 0:
                if abs(omega) < 1e-16:
                    self._iterations = k
                    break

                beta = (rho / rho_prev) * (alpha / omega)

                domain.axpy(-omega, v, p)
                domain.ax(beta, p)
                domain.axpy(1.0, r, p)
            else:
                p = domain.copy(r)

            phat = domain.copy(p) if preconditioner is None else preconditioner(p)

            v = operator(phat)

            rv = domain.inner_product(r_hat, v)
            if abs(rv) < 1e-16:
                self._iterations = k
                break
            alpha = rho / rv

            domain.axpy(-alpha, v, r)

            if domain.norm(r) < atol:
                domain.axpy(alpha, phat, x)
                self._iterations = k + 1
                return x

            shat = domain.copy(r) if preconditioner is None else preconditioner(r)

            t = operator(shat)

            omega = domain.inner_product(t, r) / domain.inner_product(t, t)

            domain.axpy(alpha, phat, x)
            domain.axpy(omega, shat, x)

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

        x = domain.zero if x0 is None else domain.copy(x0)

        u = codomain.subtract(y, operator(x))
        norm_y = codomain.norm(y)

        beta = codomain.norm(u)
        if beta > 0:
            codomain.ax(1.0 / beta, u)

        v = operator.adjoint(u)
        alfa = domain.norm(v)
        if alfa > 0:
            domain.ax(1.0 / alfa, v)

        w = domain.copy(v)

        rhobar = alfa
        phibar = beta
        dampsq = damping**2

        res2 = 0.0

        atol = self._atol
        rtol = self._rtol
        btol = atol + rtol * norm_y

        maxiter = (
            self._maxiter
            if self._maxiter is not None
            else 2 * max(domain.dim, codomain.dim)
        )

        for k in range(maxiter):
            Av = operator(v)
            codomain.ax(-alfa, u)
            codomain.axpy(1.0, Av, u)
            beta = codomain.norm(u)
            if beta > 0:
                codomain.ax(1.0 / beta, u)

            A_u = operator.adjoint(u)
            domain.ax(-beta, v)
            domain.axpy(1.0, A_u, v)
            alfa = domain.norm(v)
            if alfa > 0:
                domain.ax(1.0 / alfa, v)

            if damping > 0:
                rhobar1 = np.sqrt(rhobar**2 + dampsq)
                cs1 = rhobar / rhobar1
                sn1 = damping / rhobar1
                psi = sn1 * phibar
                phibar = cs1 * phibar
            else:
                rhobar1 = rhobar
                psi = 0.0

            rho = np.sqrt(rhobar1**2 + beta**2)
            cs = rhobar1 / rho
            sn = beta / rho

            theta = sn * alfa
            rhobar = -cs * alfa
            phi = cs * phibar
            phibar = sn * phibar
            tau = sn * phi

            t1 = phi / rho
            t2 = -theta / rho

            domain.axpy(t1, w, x)

            domain.ax(t2, w)
            domain.axpy(1.0, v, w)

            res1 = phibar**2
            res2 = res2 + psi**2
            rnorm = np.sqrt(res1 + res2)

            arnorm = alfa * abs(tau)

            if rnorm <= btol:
                self._iterations = k + 1
                break

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
        self._iterations = 0
        space = operator.domain
        x = space.zero if x0 is None else space.copy(x0)

        r = space.subtract(y, operator(x))
        norm_y = space.norm(y)

        if preconditioner is None:
            preconditioner = space.identity_operator()

        z = preconditioner(r)
        p = space.copy(z)

        rz = space.inner_product(r, z)

        maxiter = self._maxiter if self._maxiter is not None else 2 * space.dim

        for k in range(maxiter):
            ap = operator(p)
            pap = space.inner_product(p, ap)

            alpha = rz / pap

            space.axpy(alpha, p, x)

            space.axpy(-alpha, ap, r)

            if space.norm(r) < self._atol + self._rtol * norm_y:
                self._iterations += 1
                break

            z_new = preconditioner(r)
            beta = -space.inner_product(z_new, ap) / pap

            p = space.add(z_new, space.multiply(beta, p))

            z = z_new
            rz = space.inner_product(r, z)
        else:
            self._iterations = maxiter

        return x


class ProgressCallback:
    """
    A simple callback that prints the solver's iteration count.
    """

    def __init__(self, message: str = "Iteration: "):
        self.iteration = 0
        self.message = message

    def reset(self) -> None:
        """Resets the state for a new solve."""
        self.iteration = 0

    def finalize(self) -> None:
        """Called at the end of a solve to clean up the console output."""
        print()

    def __call__(self, xk: Any) -> None:
        self.iteration += 1
        print(f"{self.message}{self.iteration}", end="\r")


class SolutionTrackingCallback(ProgressCallback):
    """
    A callback that tracks the solution vector at each iteration.

    Useful for visualizing the convergence path of the solver or
    calculating diagnostics post-hoc without slowing down the inversion.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        message: str = "Iteration: ",
        print_progress: bool = True,
    ):
        super().__init__(message)
        self.print_progress = print_progress
        self.history: List[Any] = []
        self.domain = domain

    def __call__(self, xk: Any) -> None:
        self.iteration += 1
        # Copy the state to prevent overwriting if the solver mutates xk in-place
        self.history.append(self.domain.copy(xk))

        if self.print_progress:
            print(f"{self.message}{self.iteration}", end="\r")


class ResidualTrackingCallback(ProgressCallback):
    """
    A callback that computes and tracks the exact residual norm ||y - A(x)||.

    Warning: This evaluates the forward operator once per iteration. For very
    large problems, this may introduce computational overhead.
    """

    def __init__(
        self,
        operator: LinearOperator,
        y: Vector,
        print_progress: bool = True,
        message: str = "Iteration: {iter} | Residual: {res:.3e}",
    ):
        super().__init__("")
        self.operator = operator
        self.y = y
        self.print_progress = print_progress
        self.custom_message = message
        self.residuals: List[float] = []

    def __call__(self, xk: Any) -> None:
        self.iteration += 1

        # Safely handle both SciPy NumPy arrays and pygeoinf Vectors
        domain = self.operator.domain
        if isinstance(xk, np.ndarray):
            x_vec = domain.from_components(xk)
        else:
            x_vec = xk

        # Compute the exact residual: r = y - A(x)
        Ax = self.operator(x_vec)
        r = self.operator.codomain.subtract(self.y, Ax)
        res_norm = self.operator.codomain.norm(r)

        self.residuals.append(res_norm)

        if self.print_progress:
            print(
                self.custom_message.format(iter=self.iteration, res=res_norm), end="\r"
            )
