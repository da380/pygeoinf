"""
Module for solution of non-linear inverse and inference problems based on optimisation methods. 
"""

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator as ScipyLinOp


from .hilbert_space import Vector
from .nonlinear_forms import NonLinearForm


class ScipyUnconstrainedOptimiser:
    """
    A wrapper for scipy.optimize.minimize that adapts a NonLinearForm.
    """

    _HESSIAN_METHODS = {
        "Newton-CG",
        "trust-ncg",
        "trust-krylov",
        "trust-exact",
        "dogleg",
    }

    _GRADIENT_METHODS = {"BFGS", "L-BFGS-B", "CG"}

    _DERIVATIVE_FREE_METHODS = {"Nelder-Mead", "Powell"}

    def __init__(self, method: str, /, **kwargs: Any) -> None:
        """
        Args:
            method (str): The optimization method to use (e.g., 'Newton-CG', 'BFGS').
            **kwargs: Options to be passed to scipy.optimize.minimize (e.g., tol, maxiter).
        """
        self.method = method
        self.solver_kwargs = kwargs

    def minimize(self, form: NonLinearForm, x0: Vector) -> Vector:
        """
        Finds the minimum of a NonLinearForm starting from an initial guess.

        Args:
            form (NonLinearForm): The non-linear functional to minimize.
            x0 (Vector): The initial guess in the Hilbert space.

        Returns:
            Vector: The vector that minimizes the form.
        """
        domain = form.domain

        def fun(cx: np.ndarray) -> float:
            x = domain.from_components(cx)
            return form(x)

        jac_wrapper = None
        try:

            def jac_func(cx: np.ndarray) -> np.ndarray:
                x = domain.from_components(cx)
                grad_x = form.gradient(x)
                return domain.to_components(grad_x)

            jac_wrapper = jac_func
        except NotImplementedError:
            pass

        hess_wrapper = None
        try:

            def hess_func(cx: np.ndarray) -> ScipyLinOp:
                x = domain.from_components(cx)
                hessian_op = form.hessian(x)
                return hessian_op.matrix(galerkin=True)

            hess_wrapper = hess_func
        except NotImplementedError:
            pass

        final_jac = (
            jac_wrapper if self.method not in self._DERIVATIVE_FREE_METHODS else None
        )
        final_hess = hess_wrapper if self.method in self._HESSIAN_METHODS else None

        options = self.solver_kwargs.copy()
        tol = options.pop("tol", None)

        if self.method in self._GRADIENT_METHODS:
            if tol is not None and "gtol" not in options:
                options["gtol"] = tol

        cx0 = domain.to_components(x0)

        result = minimize(
            fun=fun,
            x0=cx0,
            method=self.method,
            jac=final_jac,
            hess=final_hess,
            tol=tol,
            options=options,
        )

        c_final = result.x
        return domain.from_components(c_final)
