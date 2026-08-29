"""Verify the level-master dual (O4) and a warm-started simplex QP (O3) before writing them down."""
from common import *
import warnings; warnings.simplefilter("ignore")
import osqp, scipy.sparse as sparse
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.numerics.convex import LevelBundleMethod, _minimise_on_simplex, _project_on_simplex
rng = np.random.default_rng(7)
# ---- O4: level QP  min 0.5|x-c|^2  s.t. f_j + (g_j, x - x_j) <= level  ==  dual over l >= 0 of
#      0.5 |sum l_j g_j|^2 + sum l_j (level - f_j + (g_j, x_j - c))  ... minimise; x = c - sum l_j g_j
n, k = 400, 12
sp = EuclideanSpace(n)
cuts = [(rng.normal(size=n), float(rng.normal()), rng.normal(size=n)) for _ in range(k)]
c = rng.normal(size=n)
lb = LevelBundleMethod()
values = np.array([f + g @ (c - p) for g, f, p in cuts])   # model value at the centre
level = float(values.max()) - 0.5                            # reachable but active
x_primal = lb._master(sp, cuts, c, level)
Gm = np.stack([g for g, _, _ in cuts]); Q = Gm @ Gm.T
q = np.array([level - f + g @ (p - c) for g, f, p in cuts])
prob = osqp.OSQP(); prob.setup(sparse.csc_matrix(Q), q, sparse.identity(k, format="csc"), np.zeros(k), np.full(k, np.inf), verbose=False, eps_abs=1e-12, eps_rel=1e-12, polishing=True)
lam = np.clip(prob.solve().x, 0, None)
x_dual = c - Gm.T @ lam
print(f"O4 level master: |x_primal - x_dual| = {np.linalg.norm(x_primal - x_dual):.2e}, |x| = {np.linalg.norm(x_dual):.3f}, "
      f"max cut violation primal {np.max([f + g @ (x_primal - p) for g, f, p in cuts]) - level:.1e} dual {np.max([f + g @ (x_dual - p) for g, f, p in cuts]) - level:.1e}")
# proximal fallback (level=None):  min 0.5|x-c|^2 + t  s.t. cuts <= t  == simplex dual with weight 1 (the proximal method's)
x_fb = lb._master(sp, cuts, c, None)
errs = values.max() - values          # linearisation errors relative to the model's max at c
w = _minimise_on_simplex(Q, -errs, iterations=20000, tolerance=1e-14)
x_fb_dual = c - Gm.T @ w
print(f"O4 proximal fallback: |x_qp - x_simplex_dual| = {np.linalg.norm(x_fb - x_fb_dual):.2e}")
# ---- O3: warm start of the simplex FISTA across bundle updates
def fista(Q, l, w0, iterations=1000, tol=1e-8):
    size = l.size; step = 1.0 / max(float(np.linalg.eigvalsh(Q).max()), 1e-12)
    w = w0.copy(); y = w.copy(); m = 1.0
    for it in range(1, iterations + 1):
        g = Q @ w - l
        res = float(g[w > 0].max() - g.min())
        if res <= tol * max(float(np.abs(g).max()), 1.0): return w, it
        moved = _project_on_simplex(y - step * (Q @ y - l))
        m2 = 0.5 * (1 + np.sqrt(1 + 4 * m * m)); y = moved + ((m - 1) / m2) * (moved - w); w, m = moved, m2
    return w, iterations
# a growing bundle of near-parallel cuts, as at convergence
base = rng.normal(size=n); grads = [base + 0.05 * rng.normal(size=n) for _ in range(40)]
cold, warm = [], []
w_prev = None
for k in range(5, 40):
    Gk = np.stack(grads[:k]); Q = Gk @ Gk.T; l = -np.abs(rng.normal(size=k)) * 0.1
    w_c, it_c = fista(Q, l, np.full(k, 1.0 / k)); cold.append(it_c)
    w0 = np.full(k, 1.0 / k) if w_prev is None else np.append(w_prev, 0.0)   # new cut enters with weight 0
    w_w, it_w = fista(Q, l, w0); warm.append(it_w); w_prev = w_w
print(f"O3 FISTA iterations, cold start: mean {np.mean(cold):.0f} (max {max(cold)}); warm start from previous weights: mean {np.mean(warm):.0f} (max {max(warm)})")
