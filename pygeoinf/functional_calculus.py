"""
Functional calculus for abstract linear operators.

This module provides the machinery to evaluate $f(A)v$, where $A$ is a
LinearOperator, $v$ is a vector in a Hilbert space, and $f$ is a continuous
function defined on the spectrum of $A$.
"""

from __future__ import annotations

from typing import Callable, List, Tuple, Optional, Iterator, Literal

import numpy as np

from .hilbert_space import Vector
from .linear_operators import LinearOperator


_RealFunction = Callable[[np.ndarray], np.ndarray]
_BREAKDOWN_TOL = 1e-13


class LanczosOperatorFunction(LinearOperator):
    """
    A matrix-free LinearOperator representing f(A) evaluated via the Lanczos method.
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
        Args:
            operator: The self-adjoint positive operator A.
            func: A vectorized real-valued function on the spectrum of A.
            size_estimate: For 'fixed', the exact Krylov dimension. For 'variable',
                           the initial dimension before checking convergence.
            method: {'variable', 'fixed'}. The rank-determination algorithm.
            max_k: Hard limit on the Krylov dimension for the 'variable' method.
            reorth: Reorthogonalization strategy ('full' or 'none').
            rtol: Relative tolerance for the 'variable' method.
            atol: Absolute tolerance for the 'variable' method.
            check_interval: Iterations between convergence checks in 'variable' method.
        """
        if operator.domain != operator.codomain:
            raise ValueError(
                "Functional calculus via Lanczos requires an automorphism."
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

    def _mapping_impl(self, x: Vector) -> Vector:
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

    @property
    def base_operator(self) -> LinearOperator:
        return self._base_operator


def iter_lanczos_tridiagonalize(
    operator: LinearOperator,
    v: Vector,
    max_k: int,
    *,
    reorth: str = "full",
) -> Iterator[Tuple[List[Vector], np.ndarray]]:
    """Generator yielding the Lanczos basis and tridiagonal matrix at each step."""
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

        alpha_j = space.inner_product(Cq, q_curr)
        alpha_list.append(alpha_j)
        max_scale = max(max_scale, abs(alpha_j))

        r = space.subtract(Cq, space.multiply(alpha_j, q_curr))
        if q_prev is not None:
            r = space.subtract(r, space.multiply(beta_prev, q_prev))

        if reorth == "full":
            norm_r_old = space.norm(r)

            for q_i in Q:
                proj = space.inner_product(r, q_i)
                if proj != 0.0:
                    r = space.subtract(r, space.multiply(proj, q_i))

            norm_r_new = space.norm(r)
            if norm_r_new < 0.707 * norm_r_old:
                for q_i in Q:
                    proj = space.inner_product(r, q_i)
                    if proj != 0.0:
                        r = space.subtract(r, space.multiply(proj, q_i))

        k_eff = len(alpha_list)
        T = np.zeros((k_eff, k_eff))
        np.fill_diagonal(T, alpha_list)
        if k_eff > 1:
            offdiag = np.array(beta_list[: k_eff - 1])
            idx = np.arange(k_eff - 1)
            T[idx, idx + 1] = offdiag
            T[idx + 1, idx] = offdiag

        yield Q, T

        if j < max_k - 1:
            beta_j = space.norm(r)
            if beta_j < _BREAKDOWN_TOL * max_scale:
                break

            beta_list.append(beta_j)
            max_scale = max(max_scale, beta_j)

            q_prev = q_curr
            beta_prev = beta_j
            q_curr = space.multiply(1.0 / beta_j, r)


def lanczos_tridiagonalize(
    operator: LinearOperator,
    v: Vector,
    max_k: int,
    *,
    reorth: str = "full",
) -> Tuple[List[Vector], np.ndarray]:
    """Run `max_k` steps of the Lanczos process, returning the final state.

    Args:
        operator: A self-adjoint ``LinearOperator`` on a Hilbert space $H$.
        v: Starting vector in $H$. Must be non-zero.
        max_k: Maximum Krylov dimension. Early termination occurs on
            invariant-subspace breakdown.
        reorth: Reorthogonalization strategy. One of:

            - ``"full"``: Full reorthogonalization against all previous
              Lanczos vectors using the DGKS ("Twice is Nice") criterion.
            - ``"none"``: No reorthogonalization.

    Returns:
        ``(Q, T)`` where ``Q`` is the orthonormal Krylov basis and ``T`` is
        the symmetric tridiagonal matrix.
    """
    # Consume the generator to the end
    Q, T = [], np.zeros((0, 0))
    for Q_step, T_step in iter_lanczos_tridiagonalize(
        operator, v, max_k, reorth=reorth
    ):
        Q, T = Q_step, T_step
    return Q, T


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
    """Compute $f(C) v$ approximately via Lanczos, with dynamic convergence."""
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

    for i, (Q, T) in enumerate(
        iter_lanczos_tridiagonalize(operator, v, max_k, reorth=reorth)
    ):
        step = i + 1
        is_final_step = step == max_k

        # Only check convergence if we are using the variable method, we've reached
        # the initial size_estimate, and we're on a check_interval boundary.
        should_check = (
            method == "variable"
            and step >= size_estimate
            and (step - size_estimate) % check_interval == 0
        )

        if is_final_step or should_check:
            eigvals, S = np.linalg.eigh(T)
            f_lambda = np.asarray(func(eigvals)).ravel()
            g = S @ (f_lambda * S[0, :])

            final_Q, final_g = Q, g

            if should_check:
                if old_g is not None:
                    padded_old_g = np.zeros_like(g)
                    padded_old_g[: len(old_g)] = old_g

                    diff_norm = np.linalg.norm(g - padded_old_g)
                    g_norm = np.linalg.norm(g)

                    if diff_norm <= atol + rtol * g_norm:
                        break
                old_g = g

    result = space.zero
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
    """Compute $\\langle v, f(C)\\, v \\rangle_H$ via Lanczos, with dynamic convergence."""
    space = operator.domain
    norm_v = space.norm(v)
    if norm_v == 0.0:
        return 0.0

    if method == "fixed":
        max_k = size_estimate
    elif max_k is None:
        max_k = space.dim

    old_val = None
    final_val = 0.0

    for i, (Q, T) in enumerate(
        iter_lanczos_tridiagonalize(operator, v, max_k, reorth=reorth)
    ):
        step = i + 1
        is_final_step = step == max_k
        should_check = (
            method == "variable"
            and step >= size_estimate
            and (step - size_estimate) % check_interval == 0
        )

        if is_final_step or should_check:
            eigvals, S = np.linalg.eigh(T)
            f_lambda = np.asarray(func(eigvals)).ravel()

            weights = S[0, :] * S[0, :]
            val = np.dot(f_lambda, weights)

            final_val = val

            if should_check:
                if old_val is not None:
                    diff = abs(val - old_val)
                    if diff <= atol + rtol * abs(val):
                        break
                old_val = val

    return norm_v * norm_v * final_val
