"""
Matrix-function evaluation via the Lanczos method.

Given a self-adjoint positive ``LinearOperator`` $C$ on a Hilbert space
$H$, a vector $v \\in H$, and a real-valued analytic function $f$ on the
spectrum of $C$, this module implements

$$
    f(C)\\, v \\approx \\lVert v \\rVert_H \\cdot V_k\\, f(T_k)\\, e_1,
$$

where $T_k$ is the symmetric tridiagonal matrix produced by $k$ steps of
the Lanczos recurrence with starting vector $v$, and $V_k = [q_1, \\dots,
q_k]$ is the matrix of Lanczos vectors. The Lanczos identity
$C V_k = V_k T_k + \\beta_k\\, q_{k+1}\\, e_k^\\top$ guarantees that the
restriction of $C$ to the order-$k$ Krylov subspace $\\mathcal{K}_k(C, v)$
is exactly $V_k T_k V_k^*$, so $f(T_k) e_1$ provides the optimal degree-$k$
polynomial approximation to $f(C) v / \\lVert v \\rVert$ in the Krylov
basis.

This is the matrix-free workhorse used by the function-space hardening
pipeline to evaluate the weakened gauge
$R_\\theta(m) = \\lVert C^{-\\theta/2}(m - m_0) \\rVert_H$ when an explicit
spectrum of $C$ is not available.

The mathematical derivation is in
``docs/agent-docs/theory/function-space-hardening.md`` section 6.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from .hilbert_space import HilbertSpace, Vector
from .linear_operators import LinearOperator


_RealFunction = Callable[[np.ndarray], np.ndarray]
_BREAKDOWN_TOL = 1e-13


def lanczos_tridiagonalize(
    operator: LinearOperator,
    v: Vector,
    k: int,
    *,
    reorth: str = "full",
) -> Tuple[List[Vector], np.ndarray]:
    """Run $k$ steps of the Lanczos process on a self-adjoint operator.

    Args:
        operator: A self-adjoint ``LinearOperator`` on a Hilbert space $H$.
            Only the action ``operator(v)`` is used; no matrix form is
            required.
        v: Starting vector in $H$. Must be non-zero.
        k: Maximum Krylov dimension. Early termination occurs on
            invariant-subspace breakdown.
        reorth: Reorthogonalisation strategy. One of:

            - ``"full"``: full reorthogonalisation against all previous
              Lanczos vectors at every step (default; recommended).
            - ``"none"``: no reorthogonalisation. Cheaper but accuracy
              degrades rapidly past $k \\sim 30$ in floating point.

    Returns:
        ``(Q, T)`` where:

        - ``Q`` is a Python list of orthonormal Lanczos vectors of length
          $k_{\\text{eff}} \\le k$ in the ambient Hilbert space.
        - ``T`` is a $k_{\\text{eff}} \\times k_{\\text{eff}}$ symmetric
          tridiagonal NumPy matrix.

    Raises:
        ValueError: if ``v`` has zero norm, ``k < 1``, or ``reorth`` is
            invalid.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")
    if reorth not in ("full", "none"):
        raise ValueError("reorth must be 'full' or 'none'.")

    space: HilbertSpace = operator.domain
    norm_v = float(space.norm(v))
    if norm_v == 0.0:
        raise ValueError("v must be non-zero.")

    q_curr = space.multiply(1.0 / norm_v, v)
    q_prev: Vector | None = None
    beta_prev = 0.0

    Q: List[Vector] = []
    alpha_list: List[float] = []
    beta_list: List[float] = []

    for j in range(k):
        Q.append(q_curr)
        Cq = operator(q_curr)
        alpha_j = float(space.inner_product(Cq, q_curr))
        alpha_list.append(alpha_j)

        r = space.subtract(Cq, space.multiply(alpha_j, q_curr))
        if q_prev is not None:
            r = space.subtract(r, space.multiply(beta_prev, q_prev))

        if reorth == "full":
            for q_i in Q:
                proj = float(space.inner_product(r, q_i))
                if proj != 0.0:
                    r = space.subtract(r, space.multiply(proj, q_i))

        if j < k - 1:
            beta_j = float(space.norm(r))
            if beta_j < _BREAKDOWN_TOL * max(norm_v, 1.0):
                break
            beta_list.append(beta_j)
            q_prev = q_curr
            beta_prev = beta_j
            q_curr = space.multiply(1.0 / beta_j, r)

    k_eff = len(alpha_list)
    T = np.zeros((k_eff, k_eff), dtype=float)
    np.fill_diagonal(T, alpha_list)
    if k_eff > 1:
        offdiag = np.array(beta_list[: k_eff - 1], dtype=float)
        idx = np.arange(k_eff - 1)
        T[idx, idx + 1] = offdiag
        T[idx + 1, idx] = offdiag
    return Q, T


def apply_matrix_function(
    operator: LinearOperator,
    v: Vector,
    func: _RealFunction,
    k: int,
    *,
    reorth: str = "full",
) -> Vector:
    """Compute $f(C) v$ approximately via order-$k$ Lanczos.

    Args:
        operator: A self-adjoint positive ``LinearOperator`` $C$ on $H$.
        v: A vector in $H$.
        func: A vectorised real-valued function on the spectrum of $C$.
            Receives the Ritz values (NumPy array) and must return an
            array of the same length.
        k: Krylov dimension. Larger $k$ improves accuracy at $O(k \\cdot
            \\text{cost}(C\\cdot v))$ cost per call.
        reorth: Reorthogonalisation strategy. See
            :func:`lanczos_tridiagonalize`.

    Returns:
        An approximation to $f(C) v$ in $H$.
    """
    space: HilbertSpace = operator.domain
    norm_v = float(space.norm(v))
    if norm_v == 0.0:
        return space.zero

    Q, T = lanczos_tridiagonalize(operator, v, k, reorth=reorth)
    eigvals, S = np.linalg.eigh(T)

    f_lambda = np.asarray(func(eigvals), dtype=float).ravel()
    if f_lambda.shape != eigvals.shape:
        raise ValueError(
            "func must return an array matching the Ritz-value shape "
            f"{eigvals.shape}; got {f_lambda.shape}."
        )

    # f(T_k) e_1 = S diag(f(lambda)) S^T e_1 = S @ (f_lambda * S[0, :])
    g = S @ (f_lambda * S[0, :])

    # Reassemble in the ambient space: result = norm_v * sum_i g_i * Q[i].
    result = space.zero
    for g_i, q_i in zip(g, Q):
        if g_i != 0.0:
            result = space.add(result, space.multiply(float(g_i), q_i))
    return space.multiply(norm_v, result)


def matrix_function_quadratic_form(
    operator: LinearOperator,
    v: Vector,
    func: _RealFunction,
    k: int,
    *,
    reorth: str = "full",
) -> float:
    """Compute $\\langle v, f(C)\\, v \\rangle_H$ via order-$k$ Lanczos.

    Equivalent to ``space.inner_product(v, apply_matrix_function(...))``
    but avoids re-assembling $f(C) v$ in the ambient space. Uses the
    identity

    $$
        \\langle v, f(C)\\, v\\rangle_H
        = \\lVert v\\rVert_H^2 \\cdot e_1^\\top f(T_k)\\, e_1
        = \\lVert v\\rVert_H^2 \\cdot \\sum_i f(\\theta_i)\\, S_{0,i}^2,
    $$

    where $\\theta_i, S$ are the eigenvalues and eigenvectors of the
    tridiagonal $T_k$. This is the cheapest way to evaluate Gauss-quadrature
    matrix moments (Gauss-Lanczos) and is the inner loop of the
    weakened-ellipsoid sampling radius.
    """
    space: HilbertSpace = operator.domain
    norm_v = float(space.norm(v))
    if norm_v == 0.0:
        return 0.0
    _, T = lanczos_tridiagonalize(operator, v, k, reorth=reorth)
    eigvals, S = np.linalg.eigh(T)
    f_lambda = np.asarray(func(eigvals), dtype=float).ravel()
    weights = S[0, :] * S[0, :]
    return float(norm_v * norm_v * np.dot(f_lambda, weights))
