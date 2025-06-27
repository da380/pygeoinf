"""
Module for linear solvers. 
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse.linalg import LinearOperator as ScipyLinOp
from scipy.linalg import (
    cho_factor,
    cho_solve,
    lu_factor,
    lu_solve,
)
from scipy.sparse.linalg import gmres, bicgstab, cg, bicg
from pygeoinf.hilbert_space import LinearOperator


class LinearSolver(ABC):
    """
    Abstract base class for linear solvers.
    """


class DirectLinearSolver(LinearSolver):
    """
    Abstract base class for direct linear solvers.
    """


class LUSolver(DirectLinearSolver):
    """
    Direct Linear solver class based on LU decomposition of the
    matrix representation.
    """

    def __init__(self, /, *, galerkin=False):
        """
        Args:
            galerkin (bool): If true, the Galerkin matrix representation is used.
        """
        self._galerkin = galerkin

    def __call__(self, operator):
        """
        Returns the inverse of a LinearOperator based on the LU factorisation
        of its dense matrix representation.
        """

        assert operator.domain.dim == operator.codomain.dim
        matrix = operator.matrix(dense=True, galerkin=self._galerkin)
        factor = lu_factor(matrix, overwrite_a=True)

        def matvec(cy):
            return lu_solve(factor, cy, 0)

        def rmatvec(cx):
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
    Direct Linear solver class based on Cholesky decomposition of the
    matrix representation. It is assumed that the operator's matrix
    representation is self-adjoint and positive-definite.
    """

    def __init__(self, /, *, galerkin=False):
        """
        galerkin (bool): If true, use the Galerkin matrix representation.
        """
        self._galerkin = galerkin

    def __call__(self, operator):
        """
        Returns the inverse of a LinearOperator based on the LU factorisation
        of its dense matrix representation.
        """

        assert operator.is_automorphism

        matrix = operator.matrix(dense=True, galerkin=self._galerkin)
        factor = cho_factor(matrix, overwrite_a=False)

        def matvec(cy):
            return cho_solve(factor, cy)

        inverse_matrix = ScipyLinOp(
            (operator.domain.dim, operator.codomain.dim), matvec=matvec, rmatvec=matvec
        )

        return LinearOperator.from_matrix(
            operator.domain, operator.domain, inverse_matrix, galerkin=self._galerkin
        )


class IterativeLinearSolver(LinearSolver):
    """
    Abstract base class for direct linear solvers.
    """

    @abstractmethod
    def solve_linear_system(self, operator, preconditioner, y, x0):
        """
        Returns the solution of the linear system.
        """

    def solve_adjoint_linear_system(self, operator, preconditioner, x, y0):
        """
        Returns the solution of the adjoint linear system.
        """
        return self.solve_linear_system(operator.adjoint, preconditioner, x, y0)

    def __call__(self, operator, /, *, preconditioner=None):
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
    Linear solver for self-adjoint operators based on the application
    of the conjugate gradient algorithm to the matrix representation.

    It is assumed that the matrix representation of the operator
    is also self-adjoint. This will hold automatically when the
    Galerkin representation is used.
    """

    def __init__(
        self, /, *, galerkin=False, rtol=1.0e-5, atol=0, maxiter=None, callback=None
    ):
        """
        Args:
            galerkin (bool): True if the Galerkin matrix representation is used.
            rtol (float): relative tolerance within convergence checks.
            atol (float): absolute tolerance within convergence checks.
            maxiter (int): maximum number of iterations to allow.
            callback (callable): callable function after each iteration. This function
                takes in as argument the current solution vector.
        """
        self._galerkin = galerkin
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter
        self._callback = callback

    def solve_linear_system(self, operator, preconditioner, y, x0):

        domain = operator.codomain
        matrix = operator.matrix(galerkin=self._galerkin)

        if preconditioner is None:
            matrix_preconditioner = None
        else:
            matrix_preconditioner = preconditioner.matrix(galerkin=self._galerkin)

        cx0 = None if x0 is None else domain.to_components(x0)
        cy = domain.to_components(y)

        cxp = cg(
            matrix,
            cy,
            x0=cx0,
            rtol=self._rtol,
            atol=self._atol,
            maxiter=self._maxiter,
            M=matrix_preconditioner,
            callback=self._callback,
        )[0]
        if self._galerkin:
            xp = domain.dual.from_components(cxp)
            return domain.from_dual(xp)
        else:
            return domain.from_components(cxp)


class BICGMatrixSolver(IterativeLinearSolver):
    """
    Linear solver for general operators based on the application
    of the biconjugate gradient algorithm to the matrix representation.
    """

    def __init__(
        self, /, *, galerkin=False, rtol=1.0e-5, atol=0, maxiter=None, callback=None
    ):
        """
        Args:
            galerkin (bool): True if the Galerkin matrix representation is used.
            rtol (float): relative tolerance within convergence checks.
            atol (float): absolute tolerance within convergence checks.
            maxiter (int): maximum number of iterations to allow.
            callback (callable): callable function after each iteration. This function
                takes in as argument the current solution vector.
        """
        self._galerkin = galerkin
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter
        self._callback = callback

    def solve_linear_system(self, operator, preconditioner, y, x0):

        domain = operator.codomain
        codomain = operator.domain
        matrix = operator.matrix(galerkin=self._galerkin)

        if preconditioner is None:
            matrix_preconditioner = None
        else:
            matrix_preconditioner = preconditioner.matrix(galerkin=self._galerkin)

        cx0 = None if x0 is None else domain.to_components(x0)
        cy = domain.to_components(y)

        cxp = bicg(
            matrix,
            cy,
            x0=cx0,
            rtol=self._rtol,
            atol=self._atol,
            maxiter=self._maxiter,
            M=matrix_preconditioner,
            callback=self._callback,
        )[0]
        if self._galerkin:
            xp = codomain.dual.from_components(cxp)
            return codomain.from_dual(xp)
        else:
            return codomain.from_components(cxp)


class BICGStabMatrixSolver(IterativeLinearSolver):
    """
    Linear solver for general operators based on the application
    of the biconjugate gradient stabilised algorithm to the matrix representation.
    """

    def __init__(
        self, /, *, galerkin=False, rtol=1.0e-5, atol=0, maxiter=None, callback=None
    ):
        """
        Args:
            galerkin (bool): True if the Galerkin matrix representation is used.
            rtol (float): relative tolerance within convergence checks.
            atol (float): absolute tolerance within convergence checks.
            maxiter (int): maximum number of iterations to allow.
            callback (callable): callable function after each iteration. This function
                takes in as argument the current solution vector.
        """
        self._galerkin = galerkin
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter
        self._callback = callback

    def solve_linear_system(self, operator, preconditioner, y, x0):

        domain = operator.codomain
        codomain = operator.domain
        matrix = operator.matrix(galerkin=self._galerkin)

        if preconditioner is None:
            matrix_preconditioner = None
        else:
            matrix_preconditioner = preconditioner.matrix(galerkin=self._galerkin)

        cx0 = None if x0 is None else domain.to_components(x0)
        cy = domain.to_components(y)

        cxp = bicgstab(
            matrix,
            cy,
            x0=cx0,
            rtol=self._rtol,
            atol=self._atol,
            maxiter=self._maxiter,
            M=matrix_preconditioner,
            callback=self._callback,
        )[0]
        if self._galerkin:
            xp = codomain.dual.from_components(cxp)
            return codomain.from_dual(xp)
        else:
            return codomain.from_components(cxp)


class GMRESMatrixSolver(IterativeLinearSolver):
    """
    Linear solver for general operators based on the application
    of the GMRES algorithm to the matrix representation.
    """

    def __init__(
        self,
        /,
        *,
        galerkin=False,
        rtol=1.0e-5,
        atol=0,
        restart=None,
        maxiter=None,
        callback=None,
        callback_type=None,
    ):
        """
        Args:
            galerkin (bool): True if the Galerkin matrix representation is used.
            rtol (float): relative tolerance within convergence checks.
            atol (float): absolute tolerance within convergence checks.
            restart (int): Number of iterations between restarts.
            maxiter (int): maximum number of iterations to allow.
            callback (callable): callable function after each iteration. Signature
                of this function is determined by callback_type.
            callback_type ("x", "pr_norm", "legacy"): If "x" the current solution is
                passed to the callback function, if "pr_norm" it is the preconditioned
                residual norm. The default is "legacy" which means the same as "pr_norm",
                but changes the meaning of maxiter to count inner iterations instead of
                restart cycles.
        """
        self._galerkin = galerkin
        self._rtol = rtol
        self._atol = atol
        self._restart = restart
        self._maxiter = maxiter
        self._callback = callback
        self._callback_type = callback_type

    def solve_linear_system(self, operator, preconditioner, y, x0):

        domain = operator.codomain
        codomain = operator.domain
        matrix = operator.matrix(galerkin=self._galerkin)

        if preconditioner is None:
            matrix_preconditioner = None
        else:
            matrix_preconditioner = preconditioner.matrix(galerkin=self._galerkin)

        cx0 = None if x0 is None else domain.to_components(x0)
        cy = domain.to_components(y)

        cxp = gmres(
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
        )[0]

        if self._galerkin:
            xp = codomain.dual.from_components(cxp)
            return codomain.from_dual(xp)
        else:
            return codomain.from_components(cxp)


class CGSolver(IterativeLinearSolver):
    """
    LinearSolver class using the conjugate gradient algorithm without
    use of the matrix representation. Can be applied to self-adjoint
    operators on a general Hilbert space.
    """

    def __init__(self, /, *, rtol=1.0e-5, atol=0, maxiter=None, callback=None):
        """
        Args:
            rtol (float): relative tolerance within convergence checks.
            atol (float): absolute tolerance within convergence checks.
            maxiter (int): maximum number of iterations to allow.
            callback (callable): callable function after each iteration. This function
                takes in as argument the current solution vector.
        """
        if rtol > 0:
            self._rtol = rtol
        else:
            raise ValueError("rtol must be positive")
        if atol >= 0:
            self._atol = atol
        else:
            raise ValueError("atol must be non-negative!")
        if maxiter is None:
            self._maxiter = maxiter
        else:
            if maxiter >= 0:
                self._maxiter = maxiter
            else:
                raise ValueError("maxiter must be None or positive")

        self._callback = callback

    def solve_linear_system(self, operator, preconditioner, y, x0):

        domain = operator.domain
        if x0 is None:
            x = domain.zero
        else:
            x = domain.copy(x0)

        r = domain.subtract(y, operator(x))
        if preconditioner is None:
            z = domain.copy(r)
        else:
            z = preconditioner(r)
        p = domain.copy(z)

        y_squared_norm = domain.squared_norm(y)
        if y_squared_norm <= self._atol:
            return y

        tol = np.max([self._atol, self._rtol * y_squared_norm])

        if self._maxiter is None:
            maxiter = 10 * domain.dim
        else:
            maxiter = self._maxiter

        for _ in range(maxiter):

            if domain.norm(r) <= tol:
                break

            q = operator(p)
            num = domain.inner_product(r, z)
            den = domain.inner_product(p, q)
            alpha = num / den

            x = domain.axpy(alpha, p, x)
            r = domain.axpy(-alpha, q, r)

            if preconditioner is None:
                z = domain.copy(r)
            else:
                z = preconditioner(r)

            den = num
            num = operator.domain.inner_product(r, z)
            beta = num / den

            p = domain.multiply(beta, p)
            p = domain.add(p, z)

            if self._callback is not None:
                self._callback(x)

        return x
