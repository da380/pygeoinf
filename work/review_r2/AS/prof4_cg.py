from common import *
import cProfile, pstats, io
import scipy.sparse.linalg as spl
from pygeoinf2 import LinearOperator, EuclideanSpace, DirectSum, BlockDiagonalLinearOperator, CGSolver, Traits, DiagonalLinearOperator
n = 2000
rng = np.random.default_rng(0)
R = rng.standard_normal((n, n))
S = R @ R.T / n + np.eye(n)          # cond ~ 10
b = rng.standard_normal(n)
rtol = 1e-8

# --- Euclidean: v2 CG vs scipy cg -----------------------------------
X = EuclideanSpace(n)
A = LinearOperator.from_matrix(X, X, S, form="galerkin", traits=Traits.POSITIVE_DEFINITE)
inv = CGSolver(rtol=rtol)(A)
res = inv.solve(b)
sp_it = [0]
def cb(xk): sp_it[0] += 1
def scipy_cg():
    sp_it[0] = 0
    return spl.cg(S, b, rtol=rtol, atol=0.0, callback=cb, maxiter=4*n)
r = interleave({"v2 CG": lambda: inv.solve(b), "scipy cg": scipy_cg, "matvec x it": lambda: [S @ b for _ in range(res.iterations)]}, repeats=3)
print(f"Euclid {n}: v2 {res.iterations} it {r['v2 CG']*1e3:.1f} ms; scipy {sp_it[0]} it {r['scipy cg']*1e3:.1f} ms; matvecs alone {r['matvec x it']*1e3:.1f} ms")

# --- dense metric: components-form operator ------------------------
D = dense_space(n)
G = D._gram
# a G-self-adjoint PD operator: A_c = G^-1 S  (Galerkin matrix S, SPD)
Ad = LinearOperator.from_matrix(D, D, S, form="galerkin", traits=Traits.POSITIVE_DEFINITE)
Ac = LinearOperator.from_matrix(D, D, np.linalg.solve(G, S), form="components", traits=Traits.POSITIVE_DEFINITE)
bd = rng.standard_normal(n)
cnt = Counter(); orig_apply = D.apply_gram; orig_solve = D.solve_gram
cnt2 = Counter()
D.apply_gram = cnt.wrap(orig_apply); D.solve_gram = cnt2.wrap(orig_solve)
res_c = CGSolver(rtol=rtol)(Ac).solve(bd)
print(f"dense metric, components-form: {res_c.iterations} it; apply_gram calls {cnt.n} ({cnt.n/res_c.iterations:.2f}/it), solve_gram {cnt2.n}")
cnt.n = cnt2.n = 0
res_g = CGSolver(rtol=rtol)(Ad).solve(bd)
print(f"dense metric, galerkin-form:   {res_g.iterations} it; apply_gram calls {cnt.n} ({cnt.n/res_g.iterations:.2f}/it), solve_gram {cnt2.n} ({cnt2.n/res_g.iterations:.2f}/it)")
D.apply_gram = orig_apply; D.solve_gram = orig_solve
invc = CGSolver(rtol=rtol)(Ac); invg = CGSolver(rtol=rtol)(Ad)
r = interleave({"components-form": lambda: invc.solve(bd), "galerkin-form": lambda: invg.solve(bd), "matvec x it": lambda: [S @ bd for _ in range(res_c.iterations)], "gram x 3it": lambda: [G @ bd for _ in range(3*res_c.iterations)]}, repeats=3)
print({k: f"{v*1e3:.0f} ms" for k, v in r.items()})
pr = cProfile.Profile(); pr.enable(); invg.solve(bd); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(8); print("\n".join(s.getvalue().splitlines()[:20]))

# --- saving from not recomputing (r, r) when unpreconditioned -------
# emulate: norm(r) then inner_product(r, r) == 2 Gram applications; sharing saves 1 of 3
print("per-iteration Gram applications: inner(p,Ap), norm(r), inner(r,z) = 3; with z is r the last duplicates the second.")

# --- DirectSum vs flat -----------------------------------------------
m = n // 2
X1, X2 = EuclideanSpace(m), EuclideanSpace(m)
S1 = S[:m, :m]; S2 = S[m:, m:]
Bd = BlockDiagonalLinearOperator([LinearOperator.from_matrix(X1, X1, S1, form="galerkin", traits=Traits.POSITIVE_DEFINITE),
                                  LinearOperator.from_matrix(X2, X2, S2, form="galerkin", traits=Traits.POSITIVE_DEFINITE)])
Sflat = np.block([[S1, np.zeros((m, m))], [np.zeros((m, m)), S2]])
Aflat = LinearOperator.from_matrix(X, X, Sflat, form="galerkin", traits=Traits.POSITIVE_DEFINITE)
bt = (b[:m].copy(), b[m:].copy())
inv_ds = CGSolver(rtol=rtol)(Bd); inv_fl = CGSolver(rtol=rtol)(Aflat)
rd = inv_ds.solve(bt); rf = inv_fl.solve(b)
r = interleave({"DirectSum CG": lambda: inv_ds.solve(bt), "flat CG": lambda: inv_fl.solve(b)}, repeats=3)
print(f"DirectSum(2x{m}) {rd.iterations} it {r['DirectSum CG']*1e3:.1f} ms vs flat {rf.iterations} it {r['flat CG']*1e3:.1f} ms")
# small blocks: 20 x 100
k = 20; mm = n // k
blocks = [LinearOperator.from_matrix(EuclideanSpace(mm), EuclideanSpace(mm), S[i*mm:(i+1)*mm, i*mm:(i+1)*mm], form="galerkin", traits=Traits.POSITIVE_DEFINITE) for i in range(k)]
Bk = BlockDiagonalLinearOperator(blocks)
bk = tuple(b[i*mm:(i+1)*mm].copy() for i in range(k))
inv_k = CGSolver(rtol=rtol)(Bk); rk = inv_k.solve(bk)
r = interleave({"DirectSum 20 blocks": lambda: inv_k.solve(bk)}, repeats=3)
print(f"DirectSum(20x{mm}) {rk.iterations} it {r['DirectSum 20 blocks']*1e3:.1f} ms ({r['DirectSum 20 blocks']/rk.iterations*1e6:.0f} us/it)")
pr = cProfile.Profile(); pr.enable(); inv_k.solve(bk); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(8); print("\n".join(s.getvalue().splitlines()[:20]))
