from common import *
import warnings; warnings.simplefilter("ignore")
from p2_bundle import setting, rng
from pygeoinf2.geometry.convex import Ball
from pygeoinf2.inference import DualFeasibleProperty
from pygeoinf2.numerics.convex import LevelBundleMethod, ProximalBundleMethod
from pygeoinf2.numerics.quadratic_programming import OSQPQPSolver, ClarabelQPSolver, SciPyQPSolver
model, data, tgt, target, problem, d = setting(300, 60)
dual = DualFeasibleProperty(problem, target, Ball(model, radius=3.0))
dirs = [tgt.scale(s, tgt.basis_vector(i)) for i in range(3) for s in (1., -1.)]
cost = dual.dual_cost(dirs[0], d)
for name, qp in (("osqp", OSQPQPSolver()), ("clarabel", ClarabelQPSolver()), ("scipy", SciPyQPSolver())):
    t = time.perf_counter()
    r = LevelBundleMethod(tolerance=1e-6, iterations=100, qp_solver=qp).minimise(cost, data.zero())
    print(name, f"{time.perf_counter()-t:.2f}s", r, r.message, "evaluations", r.evaluations, "gap", r.gap)
print("kkt:", dual.support_values(dirs[:1], d, route="kkt"), " primal:", dual.support_values(dirs[:1], d, route="primal"))
