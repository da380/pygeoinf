"""Stage 0.1: is the one-multiplier reduction of the quadratic tier sound?

Support value in direction q of T(prior ∩ A⁻¹(d − noise)) with
prior = {(Q(m−a), m−a) ≤ 1}, noise = {(P(e−b), e−b) ≤ 1}.  With c = T*q and
d' = d − A a − b, the Lagrangian dual is

    ψ(s, t) = ½ (b_st, M_st⁻¹ b_st) + (s + t)/2 − (t/2)(P d', d'),
    b_st = c + t A*P d',   M_st = s Q + t A*P A,

and h(q) = min over s, t ≥ 0 of ψ (strong duality under Slater).  Writing
s = γ t, M_st = t M(γ) with M(γ) = γ Q + A*P A, so with

    u₁ = M(γ)⁻¹ c,   u₂ = M(γ)⁻¹ A*P d',
    A(γ) = (c, u₁),  B(γ) = (c, u₂),  C(γ) = (A*P d', u₂) + γ + 1 − (P d', d'),

ψ(γt, t) = A/(2t) + B + t C/2, minimised over t at t* = sqrt(A/C) with value

    φ(γ) = B(γ) + sqrt(A(γ) C(γ)).

So h(q) = min over γ > 0 of φ(γ): one scalar minimisation, two solves per γ.
φ(∞) is the prior-only bound sqrt((c, Q⁻¹ c)).  The question is whether φ is
unimodal in γ, so that a bracketed scalar minimiser finds the global minimum.

This script tabulates φ on a log grid for many directions and problems,
counts interior local minima, and compares min φ with (a) the library's
nested-bisection route and (b) a direct minimisation of ψ(s, t).
"""

from __future__ import annotations

import sys
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.geometry.convex import Ellipsoid
from pygeoinf2.inference import BackusGilbertParker, LinearForwardProblem
from pygeoinf2.traits import Traits


def spd(rng, n, cond):
    """A random SPD matrix with the given condition number."""
    v, _ = np.linalg.qr(rng.normal(size=(n, n)))
    w = np.logspace(0.0, np.log10(cond), n)
    return (v * w) @ v.T


def make_problem(
    rng,
    *,
    n_m=40,
    n_d=12,
    n_p=3,
    prior="ball",
    noise="ball",
    centers=False,
    prior_cond=1.0,
    snr=20.0,
):
    A = rng.normal(size=(n_d, n_m)) / np.sqrt(n_m)
    T = rng.normal(size=(n_p, n_m)) / np.sqrt(n_m)
    if prior == "ball":
        Q = np.eye(n_m)
    elif prior == "ellipsoid":
        Q = spd(rng, n_m, 30.0)
    elif prior == "illcond":
        # a quadratic form that is nearly singular relative to A*PA, the
        # L²-ball-on-H² situation: Q eigenvalues spanning prior_cond decades
        Q = spd(rng, n_m, prior_cond)
    a = rng.normal(size=n_m) if centers else np.zeros(n_m)
    # a truth strictly inside the prior
    z = rng.normal(size=n_m)
    L = np.linalg.cholesky(Q)
    m_true = a + 0.6 * np.linalg.solve(L.T, z) / np.sqrt(z @ z)
    if noise == "ball":
        P = np.eye(n_d) * (snr / np.linalg.norm(A @ (m_true - a))) ** 2 * n_d
    else:
        P = spd(rng, n_d, 10.0) * (snr / np.linalg.norm(A @ (m_true - a))) ** 2 * n_d
    b = rng.normal(size=n_d) * 0.1 if centers else np.zeros(n_d)
    # noise realisation inside the noise set, with some margin
    e = rng.normal(size=n_d)
    Lp = np.linalg.cholesky(P)
    e = b + 0.7 * np.linalg.solve(Lp.T, e) / np.sqrt(e @ e)
    d = A @ m_true + e
    return dict(A=A, T=T, Q=Q, P=P, a=a, b=b, d=d, m_true=m_true)


def phi_terms(prob, c, gamma):
    A, Q, P = prob["A"], prob["Q"], prob["P"]
    dp = prob["d"] - A @ prob["a"] - prob["b"]
    M = gamma * Q + A.T @ P @ A
    rhs = A.T @ (P @ dp)
    sol = np.linalg.solve(M, np.column_stack([c, rhs]))
    u1, u2 = sol[:, 0], sol[:, 1]
    Aval = c @ u1
    Bval = c @ u2
    Cval = rhs @ u2 + gamma + 1.0 - dp @ (P @ dp)
    return Aval, Bval, Cval


def phi(prob, c, gamma):
    Aval, Bval, Cval = phi_terms(prob, c, gamma)
    if Cval <= 0.0:
        return np.inf  # the inner minimum over t is not attained
    return Bval + np.sqrt(Aval * Cval)


class DataSpacePhi:
    """φ(γ) evaluated in the data-space formalism, as a matrix-free code would.

    With C = Q⁻¹, R = P⁻¹, G = A C A* and K(γ) = G + γ R (Woodbury), and the
    resolvent identity G⁻¹ − K⁻¹ = γ G⁻¹ R K⁻¹:

        A(γ) = κ/γ + (ACc, G⁻¹ R K⁻¹ ACc),   κ = (c, Cc) − (ACc, G⁻¹ ACc)
        B(γ) = (ACc, K⁻¹ d')
        C(γ) = 1 + γ (1 − (d', K⁻¹ d'))

    so nothing cancels: κ is the kernel part of c, computed once per
    direction at its natural scale, and every other term is a single solve
    with K(γ). Needs G invertible (the underdetermined case).
    """

    def __init__(self, prob, c):
        A, Q, P = prob["A"], prob["Q"], prob["P"]
        self.C = np.linalg.inv(Q)
        self.R = np.linalg.inv(P)
        self.G = A @ self.C @ A.T
        self.dp = prob["d"] - A @ prob["a"] - prob["b"]
        self.v = A @ (self.C @ c)  # A C c
        self.g_inv_v = np.linalg.solve(self.G, self.v)
        self.kappa = max(c @ (self.C @ c) - self.v @ self.g_inv_v, 0.0)

    def __call__(self, gamma):
        K = self.G + gamma * self.R
        k_inv_v = np.linalg.solve(K, self.v)
        k_inv_d = np.linalg.solve(K, self.dp)
        Aval = self.kappa / gamma + self.g_inv_v @ (self.R @ k_inv_v)
        Bval = self.v @ k_inv_d
        Cval = 1.0 + gamma * (1.0 - self.dp @ k_inv_d)
        if Cval <= 0.0:
            return np.inf
        return Bval + np.sqrt(Aval * Cval)


def prior_only(prob, c):
    return np.sqrt(c @ np.linalg.solve(prob["Q"], c))


def psi(prob, c, s, t):
    A, Q, P = prob["A"], prob["Q"], prob["P"]
    dp = prob["d"] - A @ prob["a"] - prob["b"]
    M = s * Q + t * A.T @ P @ A
    bb = c + t * A.T @ (P @ dp)
    return 0.5 * bb @ np.linalg.solve(M, bb) + 0.5 * (s + t) - 0.5 * t * dp @ (P @ dp)


def direct_dual(prob, c, start):
    """Minimise ψ over (log s, log t) from a start; the reference answer."""

    def f(x):
        return psi(prob, c, np.exp(x[0]), np.exp(x[1]))

    best = None
    for x0 in (
        start,
        start + np.array([2.0, 0.0]),
        start + np.array([-2.0, 0.0]),
        start + np.array([0.0, 2.0]),
        start + np.array([0.0, -2.0]),
    ):
        r = minimize(
            f,
            x0,
            method="Nelder-Mead",
            options=dict(xatol=1e-12, fatol=1e-14, maxiter=20000),
        )
        if best is None or r.fun < best.fun:
            best = r
    return best.fun


def library_support(prob, c_index_direction):
    """The present bisection route on the same problem (plus the center shift)."""
    prob_, q = c_index_direction
    A, T, Q, P = prob_["A"], prob_["T"], prob_["Q"], prob_["P"]
    n_m, n_d, n_p = A.shape[1], A.shape[0], T.shape[0]
    M, D, Pp = EuclideanSpace(n_m), EuclideanSpace(n_d), EuclideanSpace(n_p)
    Aop = LinearOperator.from_matrix(M, D, A, form="galerkin")
    Top = LinearOperator.from_matrix(M, Pp, T, form="galerkin")
    pd = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
    Qop = LinearOperator.from_matrix(M, M, Q, form="galerkin").with_traits(pd)
    Cop = LinearOperator.from_matrix(
        M, M, np.linalg.inv(Q), form="galerkin"
    ).with_traits(pd)
    Pop = LinearOperator.from_matrix(D, D, P, form="galerkin").with_traits(pd)
    Rop = LinearOperator.from_matrix(
        D, D, np.linalg.inv(P), form="galerkin"
    ).with_traits(pd)
    prior = Ellipsoid(M, Qop, center=prob_["a"], covariance=Cop)
    noise = Ellipsoid(D, Pop, center=prob_["b"], covariance=Rop)
    est = BackusGilbertParker(
        LinearForwardProblem(Aop),
        Top,
        prior,
        noise=noise,
        route="bisection",
        max_iterations=60,
    )
    return est(prob_["d"]).support(q)


def cold_bracket_minimize(f, x0, *, step=1.0, xatol=1e-10):
    """Expand geometrically from x0 in log γ until a bracket exists, then Brent.

    Returns (x_min, f_min, evaluations). If f keeps decreasing to the
    boundary of the search range, the boundary is reported.
    """
    calls = [0]

    def g(x):
        calls[0] += 1
        return f(x)

    fx = g(x0)
    lo, hi = x0 - step, x0 + step
    flo, fhi = g(lo), g(hi)
    # walk downhill until the middle is the smallest of the three
    while not (fx <= flo and fx <= fhi):
        if flo < fx:
            hi, fhi, x0, fx = x0, fx, lo, flo
            lo = x0 - step
            flo = g(lo)
        else:
            lo, flo, x0, fx = x0, fx, hi, fhi
            hi = x0 + step
            fhi = g(hi)
        step *= 1.6
        if abs(x0) > 40.0:
            return x0, fx, calls[0]
    res = minimize_scalar(
        g, bracket=(lo, x0, hi), method="brent", options=dict(xtol=xatol)
    )
    return res.x, res.fun, calls[0]


def run_case(rng, name, n_dirs=24, verbose=False, **kw):
    prob = make_problem(rng, **kw)
    T = prob["T"]
    n_p = T.shape[0]
    grid = np.logspace(-9, 9, 361)
    n_multi, worst_rel_lib, worst_rel_dual, evals, edges = 0, 0.0, 0.0, [], []
    lib_failures = []
    cold_evals, cold_misses = [], 0
    ds_multi, ds_agree, warm, warm_evals, warm_misses = [], [], [], [], 0
    t0 = time.time()
    # directions on a great circle of the property space, so neighbours are near
    e1, e2 = rng.normal(size=n_p), rng.normal(size=n_p)
    e1 /= np.linalg.norm(e1)
    e2 -= (e2 @ e1) * e1
    e2 /= np.linalg.norm(e2)
    for k in range(n_dirs):
        angle = 2.0 * np.pi * k / n_dirs
        q = np.cos(angle) * e1 + np.sin(angle) * e2
        c = T.T @ q
        shift = c @ prob["a"]  # the property of the prior's centre
        values = np.array([phi(prob, c, g) for g in grid])
        finite = np.isfinite(values)
        v = values[finite]
        # interior local minima on the grid
        interior = (v[1:-1] < v[:-2]) & (v[1:-1] <= v[2:])
        n_min = int(interior.sum())
        n_multi += n_min > 1
        i_best = int(np.argmin(v))
        g_best = grid[finite][i_best]
        if n_min > 1 and verbose:
            where = np.flatnonzero(interior) + 1
            print(
                f"   direction {k}: local minima at gamma =",
                ", ".join(
                    f"{grid[finite][i]:.2e} (phi-min={v[i] - v[i_best]:.2e})"
                    for i in where
                ),
            )
        # the real method: a cold start at gamma = 1, bracket, Brent
        x_cold, f_cold, n_cold = cold_bracket_minimize(
            lambda x: phi(prob, c, np.exp(x)), 0.0
        )
        cold_evals.append(n_cold)
        if f_cold > v[i_best] * (1.0 + 1e-8) + 1e-12:
            cold_misses += 1
        # the data-space evaluation on the same grid: is the small-gamma region clean?
        if prob["A"].shape[0] < prob["A"].shape[1]:
            ds = DataSpacePhi(prob, c)
            values_ds = np.array([ds(g) for g in grid])
            vd = values_ds[finite]
            interior_ds = (vd[1:-1] < vd[:-2]) & (vd[1:-1] <= vd[2:])
            ds_multi.append(int(interior_ds.sum()) > 1)
            # agreement where the model-space value is trustworthy (gamma >= 1e-4)
            ok = grid[finite] >= 1e-4
            ds_agree.append(
                np.max(np.abs(vd[ok] - v[ok]) / np.maximum(np.abs(v[ok]), 1e-12))
            )

            # warm start from the previous direction's gamma, loose tolerance
            def f_ds(x):
                return ds(np.exp(x))

            start = warm[0] if warm else 0.0
            x_w, f_w, n_w = cold_bracket_minimize(f_ds, start, step=0.5, xatol=1e-3)
            warm[:] = [x_w]
            warm_evals.append(n_w)
            if f_w > v[i_best] * (1.0 + 1e-6) + 1e-12:
                warm_misses += 1
        # refine with a bounded scalar minimiser in log γ around the grid minimum
        lo, hi = np.log(g_best) - 0.3, np.log(g_best) + 0.3
        res = minimize_scalar(
            lambda x: phi(prob, c, np.exp(x)),
            bounds=(lo, hi),
            method="bounded",
            options=dict(xatol=1e-10),
        )
        at_edge = i_best in (0, len(v) - 1)
        h_phi = min(res.fun, prior_only(prob, c)) + shift
        evals.append(res.nfev)
        edges.append(at_edge)
        # references
        Aval, Bval, Cval = phi_terms(prob, c, np.exp(res.x))
        t_star = np.sqrt(Aval / max(Cval, 1e-300))
        h_dual = (
            direct_dual(
                prob, c, np.array([np.log(np.exp(res.x) * t_star), np.log(t_star)])
            )
            + shift
        )
        try:
            h_lib = library_support(prob, (prob, q))
        except Exception as error:  # the present route has its own failure modes
            lib_failures.append(type(error).__name__)
            h_lib = np.nan
        scale = max(abs(h_phi), 1e-12)
        if np.isfinite(h_lib):
            worst_rel_lib = max(worst_rel_lib, abs(h_phi - h_lib) / scale)
        worst_rel_dual = max(worst_rel_dual, abs(h_phi - h_dual) / scale)
    print(
        f"{name:30s} multi {n_multi:2d}/{n_dirs} edge {sum(edges):2d}"
        f" |phi-lib| {worst_rel_lib:8.1e} |phi-dual| {worst_rel_dual:8.1e}"
        f" cold-evals {np.mean(cold_evals):5.1f} cold-miss {cold_misses:2d}"
        f" lib-fail {len(lib_failures):2d} ({time.time() - t0:4.1f}s)"
    )
    if ds_multi:
        print(
            f"{'':30s} data-space: multi {sum(ds_multi):2d}/{n_dirs}"
            f" agree(gamma>=1e-4) {max(ds_agree):8.1e}"
            f" warm-evals {np.mean(warm_evals):5.1f} warm-miss {warm_misses:2d}"
        )
    return n_multi


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    rng = np.random.default_rng(seed)
    total = 0
    cases = [
        ("two balls", dict()),
        ("two balls, off-center", dict(centers=True)),
        ("ellipsoid prior, ball noise", dict(prior="ellipsoid")),
        ("ball prior, ellipsoid noise", dict(noise="ellipsoid")),
        (
            "ellipsoids both, off-center",
            dict(prior="ellipsoid", noise="ellipsoid", centers=True),
        ),
        ("ill-conditioned prior 1e4", dict(prior="illcond", prior_cond=1e4)),
        ("ill-conditioned prior 1e8", dict(prior="illcond", prior_cond=1e8)),
        ("tight noise (snr 200)", dict(snr=200.0, verbose=True)),
        ("very tight noise (snr 2000)", dict(snr=2000.0, verbose=True)),
        ("loose noise (snr 2)", dict(snr=2.0)),
        ("overdetermined n_d > n_m", dict(n_m=20, n_d=30)),
        ("many properties n_p=15", dict(n_p=15)),
    ]
    for name, kw in cases:
        if which != "all" and which not in name:
            continue
        total += run_case(rng, name, **kw)
    print("total multimodal directions:", total)


if __name__ == "__main__":
    main()
