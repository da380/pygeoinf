"""Prototype: hand the bundle's simplex subproblem to OSQP; compare time and outer convergence."""
from common import *
import warnings; warnings.simplefilter("ignore")
from p2_bundle import setting, rng
from pygeoinf2.geometry.convex import Ball
from pygeoinf2.inference import DualFeasibleProperty
import pygeoinf2.numerics.convex as cv
from pygeoinf2.numerics.convex import ProximalBundleMethod, LevelBundleMethod
import osqp, scipy.sparse as sparse

original = cv._minimise_on_simplex
def osqp_simplex(Q, l, **kw):
    k = l.size
    A = sparse.vstack([sparse.csc_matrix(np.ones((1, k))), sparse.identity(k, format="csc")]).tocsc()
    lo = np.concatenate([[1.0], np.zeros(k)]); hi = np.concatenate([[1.0], np.full(k, np.inf)])
    p = osqp.OSQP(); p.setup(sparse.csc_matrix(Q), -l, A, lo, hi, verbose=False, eps_abs=1e-10, eps_rel=1e-10, polishing=True, max_iter=20000)
    r = p.solve()
    w = np.clip(r.x, 0.0, None); return w / w.sum()

for nm, nd in ((300, 60), (300, 2000)):
    model, data, tgt, target, problem, d = setting(nm, nd)
    dual = DualFeasibleProperty(problem, target, Ball(model, radius=3.0))
    dirs = [tgt.scale(s, tgt.basis_vector(i)) for i in range(3) for s in (1., -1.)]
    ref = dual.support_values(dirs, d, route="kkt")
    for label, fn in (("fista", original), ("osqp", osqp_simplex), ("fista", original), ("osqp", osqp_simplex)):
        cv._minimise_on_simplex = fn
        t = time.perf_counter(); vals = dual.support_values(dirs, d, route="dual"); dt = time.perf_counter() - t
        cost = dual.dual_cost(dirs[0], d)
        r = ProximalBundleMethod(tolerance=1e-10, iterations=300).minimise(cost, data.zero())
        print(f"data {nd} {label}: support_values(6) {dt:.2f}s  max rel dev from kkt {np.max(np.abs(vals-ref)/np.abs(ref)):.2e}  single run: its {r.iterations} conv {r.converged} evals {r.evaluations}")
    cv._minimise_on_simplex = original
    t = time.perf_counter(); r = LevelBundleMethod(tolerance=1e-8, iterations=300).minimise(dual.dual_cost(dirs[0], d), data.zero()); dt = time.perf_counter()-t
    print(f"data {nd} level bundle: {dt:.2f}s its {r.iterations} conv {r.converged} evals {r.evaluations} value {r.value:.6f} (kkt {ref[0]:.6f}) msg {r.message}")
