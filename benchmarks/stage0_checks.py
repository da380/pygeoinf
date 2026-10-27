"""Two focused checks for Stage 0.1.

1. At very tight noise the model-space and data-space evaluations of φ
   differ by 2e-5 somewhere on the γ grid. Evaluate both in extended
   precision (longdouble) to see which one loses digits, and where.
2. In the overdetermined case the minimum is at γ = 0 (the prior is slack).
   Check φ is monotone down to γ = 0 and that φ(0) is the answer, and show
   what the present library route does there.
"""

import numpy as np
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from stage0_unimodality import (
    DataSpacePhi,
    make_problem,
    phi,
    prior_only,
    library_support,
)  # noqa: E402


def phi_model_ld(prob, c, gamma):
    """Model-space φ in longdouble, with a longdouble solve by Cholesky."""
    A, Q, P = (prob[k].astype(np.longdouble) for k in ("A", "Q", "P"))
    dp = (prob["d"] - prob["A"] @ prob["a"] - prob["b"]).astype(np.longdouble)
    c = c.astype(np.longdouble)
    g = np.longdouble(gamma)
    M = g * Q + A.T @ P @ A
    rhs = A.T @ (P @ dp)
    L = cholesky_ld(M)
    u1 = solve_ld(L, c)
    u2 = solve_ld(L, rhs)
    Aval, Bval, Cval = c @ u1, c @ u2, rhs @ u2 + g + 1 - dp @ (P @ dp)
    return float(Bval + np.sqrt(Aval * Cval)) if Cval > 0 else np.inf


def cholesky_ld(M):
    n = M.shape[0]
    L = np.zeros_like(M)
    for i in range(n):
        for j in range(i + 1):
            s = M[i, j] - L[i, :j] @ L[j, :j]
            L[i, j] = np.sqrt(s) if i == j else s / L[j, j]
    return L


def solve_ld(L, b):
    n = L.shape[0]
    y = np.zeros_like(b)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    x = np.zeros_like(b)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L[i + 1 :, i] @ x[i + 1 :]) / L[i, i]
    return x


def check_tight():
    rng = np.random.default_rng(0)
    prob = make_problem(rng, snr=2000.0)
    T = prob["T"]
    grid = np.logspace(-4, 9, 53)
    worst = (0.0, None)
    for k in range(6):
        q = rng.normal(size=T.shape[0])
        q /= np.linalg.norm(q)
        c = T.T @ q
        ds = DataSpacePhi(prob, c)
        for g in grid:
            vm, vd = phi(prob, c, g), ds(g)
            rel = abs(vm - vd) / max(abs(vm), 1e-12)
            if rel > worst[0]:
                worst = (rel, (k, g, vm, vd, c))
    rel, (k, g, vm, vd, c) = worst
    vl = phi_model_ld(prob, c, g)
    print(f"tight noise: worst disagreement {rel:.1e} at direction {k}, gamma {g:.1e}")
    print(f"  model-space double     {vm:.12e}")
    print(f"  data-space double      {vd:.12e}")
    print(f"  model-space longdouble {vl:.12e}")
    print(
        f"  -> model-space double off by {abs(vm - vl) / abs(vl):.1e}, data-space by {abs(vd - vl) / abs(vl):.1e}"
    )


def check_overdetermined():
    rng = np.random.default_rng(0)
    prob = make_problem(rng, n_m=20, n_d=30)
    T = prob["T"]
    for k in range(4):
        q = rng.normal(size=T.shape[0])
        q /= np.linalg.norm(q)
        c = T.T @ q
        gammas = [0.0, 1e-16, 1e-12, 1e-9, 1e-6, 1e-3, 1.0, 1e3]
        vals = [phi(prob, c, g) for g in gammas]
        try:
            lib = library_support(prob, (prob, q))
        except Exception as error:
            lib = f"{type(error).__name__}"
        print(f"direction {k}: phi(gamma) for gamma in {gammas}")
        print("   ", " ".join(f"{v:.9f}" for v in vals))
        print(f"    prior-only {prior_only(prob, c):.9f}   library {lib}")
        # a brute-force reference: maximise (c, m) over both sets with scipy SLSQP
        from scipy.optimize import minimize

        A, Q, P = prob["A"], prob["Q"], prob["P"]
        dp = prob["d"] - A @ prob["a"] - prob["b"]
        cons = [
            dict(type="ineq", fun=lambda u: 1.0 - u @ (Q @ u)),
            dict(type="ineq", fun=lambda u: 1.0 - (dp - A @ u) @ (P @ (dp - A @ u))),
        ]
        best = None
        for trial in range(5):
            u0 = rng.normal(size=A.shape[1]) * 0.01
            r = minimize(
                lambda u: -(c @ u),
                u0,
                constraints=cons,
                method="SLSQP",
                options=dict(ftol=1e-14, maxiter=2000),
            )
            if r.success and (best is None or -r.fun > best):
                best = -r.fun
        print(
            f"    SLSQP primal {best:.9f}  (phi(0) = {vals[0]:.9f}, min phi = {min(vals):.9f})"
        )


if __name__ == "__main__":
    check_tight()
    check_overdetermined()
