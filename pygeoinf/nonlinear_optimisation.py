"""
Module for solution of non-linear inverse and inference problems based on optimisation methods.
"""

from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import line_search as scipy_line_search
from scipy.sparse.linalg import LinearOperator as ScipyLinOp


from .hilbert_space import Vector
from .nonlinear_forms import NonLinearForm


class ScipyUnconstrainedOptimiser:
    """
    A wrapper for scipy.optimize.minimize that adapts a NonLinearForm.

    Note on derivative-free methods:
    Internal testing has shown that the 'Nelder-Mead' solver can be unreliable
    for some problems, failing to converge to the correct minimum while still
    reporting success. The 'Powell' method appears to be more robust. Users
    should exercise caution and verify results when using derivative-free
    methods.
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
        if form.has_gradient:

            def jac_func(cx: np.ndarray) -> np.ndarray:
                x = domain.from_components(cx)
                grad_x = form.gradient(x)
                return domain.to_components(grad_x)

            jac_wrapper = jac_func

        hess_wrapper = None
        if form.has_hessian:

            def hess_func(cx: np.ndarray) -> ScipyLinOp:
                x = domain.from_components(cx)
                hessian_op = form.hessian(x)
                return hessian_op.matrix(galerkin=True)

            hess_wrapper = hess_func

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


def line_search(
    form: NonLinearForm,
    xk: Vector,
    pk: Vector,
    gfk: Vector = None,
    old_fval: float = None,
    old_old_fval: float = None,
    c1: float = 0.0001,
    c2: float = 0.9,
    amax: float = None,
    extra_condition: Callable[[float, Vector, float, Vector], bool] = None,
    maxiter: int = 10,
):
    """
    Wrapper for the scipy line_search method for application to a non-linear form.

    Args:
        form (NonLinearForm): The non-linear functional to minimize.
        xk (Vector): The current point.
        pk (Vector): The search direction.
        gfk (Vector, optional): The gradient at x=xk. If not provided will be recalculated.
        old_fval (float, optional): The function value at x=xk. If not provided will be recalculated.
        old_old_fval (float, optional): The valur at the point proceeding x=xk.
        c1 (float, optional): Parameter for Armijo condition rule.
        c2 (float, optional): Parameter for curvature condition rule.
        amax (float, optional): Maximum step size.
        extra_condition (callable, optional): A callable of the form extra_condition(alpha, x, f, g) returning
             a boolean. Arguments are the proposed step alpha and the corresponding x, f and g values. The line
             search accepts the value of alpha only if this callable returns True. If the callable returns False
             for the step length, the algorithm will continue with new iterates. The callable is only called for
             iterates satisfying the strong Wolfe conditions.
        maxiter (int, optional): Maximum number of iterations to perform.

    Returns:
        alpha (float | None): Alpha for which x_new = x0 + alpha * pk, or None if the
             line search algorithm did not converge.
        fc (int): Number of function evaluations made.
        gc (int): Numner of gradient evaluations mades.
        new_fval (float | None): New function value f(x_new)=f(x0+alpha*pk), or
            None if the line search algorithm did not converge.
        old_fval (float): Old function value f(x0).
        new_slope (float | None): The local slope along the search direction at
             the new value <myfprime(x_new), pk>, or None if the line search algorithm
             did not converge.

    Raises:
        ValueError: If the non-linear form does not have a gradient set.
    """

    if not form.has_gradient:
        raise ValueError("NonLinearForm must provide its gradient")

    domain = form.domain

    # Wrap the function.
    def f(xc: np.ndarray) -> float:
        x = domain.from_components(xc)
        return form(x)

    # Wrap the derivative. Note that this is given in
    # terms of the components of the derivative (i.e., an element
    # of the dual space) and not the gradient, this meaning that
    # the standard Euclidean pairing with the components on the
    # descent direction will yield the correct slope.
    def myfprime(c: np.ndarray) -> np.ndarray:
        x = domain.from_components(c)
        g = form.derivative(x)
        return domain.dual.to_components(g)

    # Convert the initial vector to components.
    xkc = domain.to_components(xk)

    # Convert descent direction to components
    pkc = domain.to_components(pk)

    # If gradient provided, convert to its dual components.
    gfkc = domain.to_dual(gfk).components if gfk is not None else None

    # Wrap the extra condition, if provided.

    if extra_condition is not None:

        def _extra_condition(
            alpha: float, xc: np.ndarray, f: float, gc: np.ndarray
        ) -> bool:
            x = domain.from_components(xc)
            df = domain.dual.from_components(gc)
            g = domain.from_dual(df)
            return extra_condition(alpha, x, f, g)

    return scipy_line_search(
        f,
        myfprime,
        xkc,
        pkc,
        gfk=gfkc,
        old_fval=old_fval,
        old_old_fval=old_old_fval,
        c1=c1,
        c2=c2,
        amax=amax,
        extra_condition=_extra_condition,
        maxiter=maxiter,
    )
