"""Bundle-method runs on a moderate problem, profiled."""
from common import *
import cProfile, pstats, io, warnings
warnings.simplefilter("ignore")
from pygeoinf2.tests.conftest import make_dense_metric_space
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.geometry.convex import Ball
from pygeoinf2.inference import DualFeasibleProperty, LinearForwardProblem
from pygeoinf2.numerics.convex import LevelBundleMethod, ProximalBundleMethod
rng = np.random.default_rng(3)

def setting(nm, nd):
    from pygeoinf2.tests.conftest import DenseMetricSpace
    root = np.eye(nm) + (0.3/np.sqrt(nm)) * np.tril(rng.standard_normal((nm, nm)), -1)
    model = DenseMetricSpace(root @ root.T); data = EuclideanSpace(nd); tgt = EuclideanSpace(3)
    forward = LinearOperator.from_matrix(model, data, rng.normal(size=(nd, nm)), form="galerkin")
    target = LinearOperator.from_matrix(model, tgt, rng.normal(size=(3, nm)), form="galerkin")
    raw = model.random(rng=rng); truth = model.scale(2.0 / model.norm(raw), raw)
    noise = data.scale(0.05, data.random(rng=rng))
    d = data.add(forward(truth), noise)
    problem = LinearForwardProblem(forward, error=Ball(data, radius=1.2 * data.norm(noise)))
    return model, data, tgt, target, problem, d

def profile(fn, label, keep=("convex.py", "backus.py", "quadratic_programming", "linprog", "spaces.py", "osqp", "operators.py")):
    pr = cProfile.Profile(); pr.enable(); t = time.perf_counter(); r = fn(); dt = time.perf_counter()-t; pr.disable()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(40)
    lines = [l for l in s.getvalue().splitlines() if any(k in l for k in keep)]
    print(f"--- {label}: {dt:.2f}s"); print("\n".join(lines[:18])); return r

if __name__ == "__main__":
  for nm, nd in ((300, 60), (300, 2000)):
      model, data, tgt, target, problem, d = setting(nm, nd)
      dual = DualFeasibleProperty(problem, target, Ball(model, radius=3.0))
      dirs = [tgt.scale(s, tgt.basis_vector(i)) for i in range(3) for s in (1., -1.)]
      vals = profile(lambda: dual.support_values(dirs, d, route="dual"), f"proximal bundle via support_values, model {nm}, data {nd}")
      # LevelBundleMethod on the same dual cost
      cost = dual.dual_cost(dirs[0], d)
      lb = LevelBundleMethod(tolerance=1e-6, iterations=200)
      r = profile(lambda: lb.minimise(cost, data.zero()), f"LevelBundleMethod, data {nd}")
      print("   level result:", r, " proximal support value:", vals[0])
      pb = ProximalBundleMethod(tolerance=1e-10, iterations=300)
      r2 = profile(lambda: pb.minimise(cost, data.zero()), f"ProximalBundleMethod alone, data {nd}")
      print("   proximal result:", r2)
