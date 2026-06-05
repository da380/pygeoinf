"""
Functional calculus for abstract linear operators.

This module provides the machinery to evaluate matrix functions of the form f(A)v,
where A is a self-adjoint LinearOperator, v is a vector in a Hilbert space, and
f is a continuous function defined on the spectrum of A.

The primary mathematical engine is the Lanczos method. Given a self-adjoint
LinearOperator C, a vector v in H, and a real-valued analytic function f, it
computes the approximation:

    f(C)v  ~=  ||v||_H * V_k * f(T_k) * e_1

where T_k is the k x k symmetric tridiagonal matrix produced by k steps of
the Lanczos recurrence, V_k is the orthogonal basis of the Krylov subspace,
and e_1 is the first standard basis vector. This approach yields the optimal
degree-k polynomial approximation to the true action of the function.
"""

from __future__ import annotations

from typing import Callable, List, Tuple, Optional, Iterator, Literal

import numpy as np
from scipy.linalg import eigh_tridiagonal

from .hilbert_space import Vector
from .linear_operators import LinearOperator


_RealFunction = Callable[[np.ndarray], np.ndarray]
_BREAKDOWN_TOL = 1e-13


# =====================================================================
#                             Public Classes
# =====================================================================


class LanczosOperatorFunction(LinearOperator):
    """
    A matrix-free LinearOperator representing the action of a continuous
    function applied to a self-adjoint positive operator.

    Rather than explicitly computing and storing the dense matrix f(A), this
    class evaluates the matrix-vector product f(A)x dynamically on the fly
    using the Lanczos process. This allows for highly efficient evaluations
    in massive or infinite-dimensional Hilbert spaces.
    """

    def __init__(
        self,
        operator: LinearOperator,
        func: Callable[[np.ndarray], np.ndarray],
        size_estimate: int,
        *,
        method: Literal["variable", "fixed"] = "variable",
        max_k: Optional[int] = None,
        reorth: str = "full",
        rtol: float = 1e-3,
        atol: float = 1e-8,
        check_interval: int = 5,
    ):
        """
        Initializes the functional calculus operator.

        Args:
            operator (LinearOperator): The base self-adjoint operator A. Must be
                an automorphism (domain == codomain).
            func (Callable): A vectorized, real-valued function defined on the
                spectrum of A (e.g., numpy.exp, numpy.sqrt).
            size_estimate (int): If method is 'fixed', this sets the exact Krylov
                dimension k. If method is 'variable', this sets the baseline number
                of Lanczos iterations to perform before the first convergence check.
            method (str, optional): The strategy for determining the Krylov rank.
                'fixed' runs exactly `size_estimate` steps. 'variable' dynamically
                checks for convergence. Defaults to 'variable'.
            max_k (int, optional): The absolute maximum number of Lanczos iterations
                allowed when using the 'variable' method. If None, defaults to the
                dimension of the underlying Hilbert space.
            reorth (str, optional): The reorthogonalization strategy to combat
                floating-point drift. 'full' applies "twice-is-enough" Gram-Schmidt.
                'none' skips reorthogonalization (faster, but highly unstable for
                large k). Defaults to 'full'.
            rtol (float, optional): The relative tolerance for dynamic convergence
                checking. Defaults to 1e-3.
            atol (float, optional): The absolute tolerance for dynamic convergence
                checking. Defaults to 1e-8.
            check_interval (int, optional): The number of Lanczos iterations to
                perform between convergence evaluations. Defaults to 5.

        Raises:
            ValueError: If the operator's domain and codomain are not identical.
        """
        if operator.domain != operator.codomain:
            raise ValueError(
                "Functional calculus via Lanczos requires an automorphism (domain == codomain)."
            )

        self._base_operator = operator
        self._func = func
        self._size_estimate = size_estimate
        self._method = method
        self._max_k = max_k
        self._reorth = reorth
        self._rtol = rtol
        self._atol = atol
        self._check_interval = check_interval

        super().__init__(
            operator.domain,
            operator.domain,
            self._mapping_impl,
            adjoint_mapping=self._mapping_impl,
        )

    @property
    def base_operator(self) -> LinearOperator:
        """Returns the underlying base LinearOperator."""
        return self._base_operator

    def _mapping_impl(self, x: Vector) -> Vector:
        """
        Internal mapping implementation. Evaluates f(A)x dynamically using
        the high-level apply_operator_function routine.
        """
        return apply_operator_function(
            self._base_operator,
            x,
            self._func,
            self._size_estimate,
            method=self._method,
            max_k=self._max_k,
            reorth=self._reorth,
            rtol=self._rtol,
            atol=self._atol,
            check_interval=self._check_interval,
        )


# =====================================================================
#                 High-Level API (Functional Evaluation)
# =====================================================================


def apply_operator_function(
    operator: LinearOperator,
    v: Vector,
    func: _RealFunction,
    size_estimate: int,
    *,
    method: Literal["variable", "fixed"] = "variable",
    max_k: Optional[int] = None,
    reorth: str = "full",
    rtol: float = 1e-3,
    atol: float = 1e-8,
    check_interval: int = 5,
) -> Vector:
    """
    Computes the action of a matrix function on a vector, f(A)v, using the
    Lanczos approximation.

    This function builds a Krylov subspace from the starting vector v. It projects
    the large operator onto this subspace to form a small tridiagonal matrix T.
    The target function f is evaluated on the eigensystem of T, and the result
    is lifted back to the original full-dimensional space.

    Args:
        operator (LinearOperator): The self-adjoint positive operator A.
        v (Vector): The input vector to be multiplied by f(A).
        func (Callable): A vectorized scalar function.
        size_estimate (int): The initial or fixed number of Krylov basis vectors.
        method (str): 'variable' to stop dynamically when the relative change in
            the approximated vector falls below tolerance. 'fixed' to run an
            exact number of iterations.
        max_k (int, optional): Hard limit on Krylov dimension.
        reorth (str): 'full' for complete basis orthogonalization, 'none' otherwise.
        rtol (float): Relative tolerance for convergence checking.
        atol (float): Absolute tolerance for convergence checking.
        check_interval (int): Iterations between convergence checks.

    Returns:
        Vector: The result of f(A)v residing in the same Hilbert space as v.
    """
    space = operator.domain
    norm_v = space.norm(v)
    if norm_v == 0.0:
        return space.zero

    if method == "fixed":
        max_k = size_estimate
    elif max_k is None:
        max_k = space.dim

    old_g = None
    final_Q, final_g = None, None
    last_Q, last_T = None, None

    for i, (Q, T) in enumerate(
        iter_lanczos_tridiagonalize(operator, v, max_k, reorth=reorth)
    ):
        step = i + 1
        last_Q, last_T = Q, T

        should_check = (
            method == "variable"
            and step >= size_estimate
            and (step - size_estimate) % check_interval == 0
        )

        if should_check:
            # Extract diagonals for O(k^2) tridiagonal eigensolver
            d = np.diag(T)
            e = np.diag(T, k=1) if len(d) > 1 else np.empty(0)
            eigvals, S = eigh_tridiagonal(d, e)

            # Evaluate the function on the projected eigenvalues
            f_lambda = np.asarray(func(eigvals)).ravel()
            g = S @ (f_lambda * S[0, :])

            # Check convergence of the projected coordinate vector
            if old_g is not None:
                padded_old_g = np.zeros_like(g)
                padded_old_g[: len(old_g)] = old_g

                diff_norm = np.linalg.norm(g - padded_old_g)
                g_norm = np.linalg.norm(g)

                if diff_norm <= atol + rtol * g_norm:
                    final_Q, final_g = Q, g
                    break

            old_g = g

    # Fallback: Capture the final state if we hit max_k without meeting tolerance,
    # or if the generator broke down early due to an exact invariant subspace.
    if final_g is None and last_T is not None:
        d = np.diag(last_T)
        e = np.diag(last_T, k=1) if len(d) > 1 else np.empty(0)
        eigvals, S = eigh_tridiagonal(d, e)
        f_lambda = np.asarray(func(eigvals)).ravel()
        final_g = S @ (f_lambda * S[0, :])
        final_Q = last_Q

    # Reconstruct the full-dimensional output vector from the Krylov basis
    result = space.zero
    if final_g is not None and final_Q is not None:
        for g_i, q_i in zip(final_g, final_Q):
            if g_i != 0.0:
                result = space.add(result, space.multiply(g_i, q_i))

    return space.multiply(norm_v, result)


def operator_function_quadratic_form(
    operator: LinearOperator,
    v: Vector,
    func: _RealFunction,
    size_estimate: int,
    *,
    method: Literal["variable", "fixed"] = "variable",
    max_k: Optional[int] = None,
    reorth: str = "full",
    rtol: float = 1e-3,
    atol: float = 1e-8,
    check_interval: int = 5,
) -> float:
    """
    Computes the quadratic form <v, f(A)v> using the Lanczos approximation.

    This function evaluates the quadratic form much more efficiently than
    explicitly computing the full vector f(A)v first. It relies on the spectral
    theorem: the quadratic form is equivalent to an integral over the spectral
    measure of A. The Lanczos process naturally generates the nodes (eigenvalues)
    and weights (squared first components of eigenvectors) for a highly accurate
    Gaussian quadrature of this integral.

    Args:
        operator (LinearOperator): The self-adjoint positive operator A.
        v (Vector): The input vector.
        func (Callable): A vectorized scalar function.
        size_estimate (int): The initial or fixed number of Krylov basis vectors.
        method (str): 'variable' to check convergence dynamically, 'fixed' otherwise.
        max_k (int, optional): Hard limit on Krylov dimension.
        reorth (str): Reorthogonalization strategy ('full' or 'none').
        rtol (float): Relative tolerance for convergence checking.
        atol (float): Absolute tolerance for convergence checking.
        check_interval (int): Iterations between convergence checks.

    Returns:
        float: The scalar evaluation of the quadratic form.
    """
    space = operator.domain
    norm_v = space.norm(v)
    if norm_v == 0.0:
        return 0.0

    if method == "fixed":
        max_k = size_estimate
    elif max_k is None:
        max_k = space.dim

    old_val = None
    final_val = None
    last_T = None

    for i, (Q, T) in enumerate(
        iter_lanczos_tridiagonalize(operator, v, max_k, reorth=reorth)
    ):
        step = i + 1
        last_T = T

        should_check = (
            method == "variable"
            and step >= size_estimate
            and (step - size_estimate) % check_interval == 0
        )

        if should_check:
            d = np.diag(T)
            e = np.diag(T, k=1) if len(d) > 1 else np.empty(0)
            eigvals, S = eigh_tridiagonal(d, e)

            f_lambda = np.asarray(func(eigvals)).ravel()

            # The weights for the Gaussian quadrature are the squares of the
            # first components of the eigenvectors.
            weights = S[0, :] * S[0, :]
            val = np.dot(f_lambda, weights)

            if old_val is not None:
                diff = abs(val - old_val)
                if diff <= atol + rtol * abs(val):
                    final_val = val
                    break

            old_val = val

    # Fallback if the loop finishes or breaks early without satisfying the tolerance check
    if final_val is None and last_T is not None:
        d = np.diag(last_T)
        e = np.diag(last_T, k=1) if len(d) > 1 else np.empty(0)
        eigvals, S = eigh_tridiagonal(d, e)
        f_lambda = np.asarray(func(eigvals)).ravel()
        weights = S[0, :] * S[0, :]
        final_val = np.dot(f_lambda, weights)

    if final_val is None:
        final_val = 0.0

    return float(norm_v * norm_v * final_val)


# =====================================================================
#                 Low-Level API (Factorization Engines)
# =====================================================================


def lanczos_tridiagonalize(
    operator: LinearOperator,
    v: Vector,
    max_k: int,
    *,
    reorth: str = "full",
) -> Tuple[List[Vector], np.ndarray]:
    """
    Executes a fixed number of Lanczos iterations to tridiagonalize the operator.

    This is a convenience wrapper around the `iter_lanczos_tridiagonalize` generator.
    It runs the process to completion and returns only the final state.

    Args:
        operator (LinearOperator): The self-adjoint operator to tridiagonalize.
        v (Vector): The starting vector for the Krylov subspace.
        max_k (int): The maximum number of iterations/dimensions to compute.
        reorth (str, optional): The reorthogonalization strategy ('full' or 'none').
            Defaults to "full".

    Returns:
        Tuple[List[Vector], np.ndarray]:
            - A list of orthonormal basis vectors defining the Krylov subspace.
            - The final symmetric tridiagonal matrix T of size up to (max_k, max_k).
    """
    Q, T = [], np.zeros((0, 0))
    for Q_step, T_step in iter_lanczos_tridiagonalize(
        operator, v, max_k, reorth=reorth
    ):
        Q, T = Q_step, T_step
    return Q, T


def iter_lanczos_tridiagonalize(
    operator: LinearOperator,
    v: Vector,
    max_k: int,
    *,
    reorth: str = "full",
) -> Iterator[Tuple[List[Vector], np.ndarray]]:
    """
    Generator that lazily yields the Krylov basis and tridiagonal matrix
    at each step of the Lanczos process.

    This engine progressively builds an orthonormal basis for the Krylov
    subspace K_k(A, v) while simultaneously projecting the operator into
    that subspace to form a symmetric tridiagonal matrix T.

    It includes an early-stopping mechanism: if the starting vector v is fully
    contained within an invariant subspace of dimension less than max_k, the
    residual norm will drop to zero, and the generator will terminate cleanly
    to prevent numerical breakdown.

    Args:
        operator (LinearOperator): The self-adjoint operator A.
        v (Vector): The starting vector.
        max_k (int): The maximum number of Krylov subspace dimensions to build.
        reorth (str, optional): Reorthogonalization strategy. 'full' employs the
            "twice is enough" modified Gram-Schmidt algorithm against all previous
            basis vectors, enforcing strict numerical orthogonality. 'none' only
            orthogonalizes against the immediate two predecessors. Defaults to 'full'.

    Yields:
        Iterator[Tuple[List[Vector], np.ndarray]]: At each step k (from 1 to max_k),
            yields a tuple containing:
            - The current list of k orthonormal basis vectors.
            - The current k x k symmetric tridiagonal numpy array T.

    Raises:
        ValueError: If max_k is less than 1, if an invalid reorth strategy is
            passed, or if the starting vector v is the zero vector.
    """
    if max_k < 1:
        raise ValueError("max_k must be at least 1.")
    if reorth not in ("full", "none"):
        raise ValueError("reorth must be 'full' or 'none'.")

    space = operator.domain
    norm_v = space.norm(v)
    if norm_v == 0.0:
        raise ValueError("v must be non-zero.")

    q_curr = space.multiply(1.0 / norm_v, v)
    q_prev: Vector | None = None
    beta_prev = 0.0

    Q: List[Vector] = []
    alpha_list: List[float] = []
    beta_list: List[float] = []

    max_scale = max(norm_v, 1.0)

    for j in range(max_k):
        Q.append(q_curr)
        Cq = operator(q_curr)

        # Compute diagonal element alpha
        alpha_j = space.inner_product(Cq, q_curr)
        alpha_list.append(alpha_j)
        max_scale = max(max_scale, abs(alpha_j))

        # Compute the residual vector orthogonal to the current and previous vectors
        r = space.subtract(Cq, space.multiply(alpha_j, q_curr))
        if q_prev is not None:
            r = space.subtract(r, space.multiply(beta_prev, q_prev))

        # Apply Full Orthogonalization if requested
        if reorth == "full":
            norm_r_old = space.norm(r)

            for q_i in Q:
                proj = space.inner_product(r, q_i)
                if proj != 0.0:
                    r = space.subtract(r, space.multiply(proj, q_i))

            # The Kahan "Twice is Enough" check
            norm_r_new = space.norm(r)
            if norm_r_new < 0.707 * norm_r_old:
                for q_i in Q:
                    proj = space.inner_product(r, q_i)
                    if proj != 0.0:
                        r = space.subtract(r, space.multiply(proj, q_i))

        # Efficiently assemble the tridiagonal matrix using 1D diagonal lists
        k_eff = len(alpha_list)
        if k_eff == 1:
            T = np.array([[alpha_list[0]]])
        else:
            T = (
                np.diag(alpha_list)
                + np.diag(beta_list[: k_eff - 1], k=1)
                + np.diag(beta_list[: k_eff - 1], k=-1)
            )

        yield Q, T

        # Compute off-diagonal element beta and prepare for the next iteration
        if j < max_k - 1:
            beta_j = space.norm(r)

            # Subspace breakdown check: if beta is practically zero, the Krylov
            # subspace is fully invariant and exact up to machine precision.
            if beta_j < _BREAKDOWN_TOL * max_scale:
                break

            beta_list.append(beta_j)
            max_scale = max(max_scale, beta_j)

            q_prev = q_curr
            beta_prev = beta_j
            q_curr = space.multiply(1.0 / beta_j, r)
