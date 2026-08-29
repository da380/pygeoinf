from common import *
import warnings; warnings.simplefilter("ignore")
from pygeoinf2.tests.conftest import make_dense_metric_space
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.geometry.convex import Ball
from pygeoinf2.inference import DualFeasibleProperty, LinearForwardProblem
from pygeoinf2.numerics.convex import ProximalBundleMethod
from scipy.optimize import minimize
def run(model, label, nd=20, rprior=3.0):
    rng = np.random.default_rng(3); nm = model.dim
    data = EuclideanSpace(nd); tgt = EuclideanSpace(3)
    M = rng.normal(size=(nd, nm)); forward = LinearOperator.from_matrix(model, data, M, form="galerkin")
    Tm = rng.normal(size=(3, nm)); target = LinearOperator.from_matrix(model, tgt, Tm, form="galerkin")
    raw = model.random(rng=rng); truth = model.scale(2.0 / model.norm(raw), raw)
    noise = data.scale(0.05, data.random(rng=rng)); d = data.add(forward(truth), noise)
    radius = 1.2 * data.norm(noise)
    problem = LinearForwardProblem(forward, error=Ball(data, radius=radius))
    dual = DualFeasibleProperty(problem, target, Ball(model, radius=rprior),
                                method=ProximalBundleMethod(tolerance=1e-10, iterations=600))
    q = tgt.basis_vector(0)
    G = model.gram_matrix(); Gi = np.linalg.inv(G)
    # independent: maximise (T m, q) = q^T Tm_galerkin? target galerkin form on Euclidean codomain: T m components = Tm @ c_m
    obj = lambda c: -(Tm @ c)[0]
    cons = [{"type": "ineq", "fun": lambda c: rprior**2 - c @ G @ c},
            {"type": "ineq", "fun": lambda c: radius**2 - np.sum((M @ c - d)**2)}]
    best = -np.inf
    for s in range(3):
        x0 = np.random.default_rng(s).normal(size=nm) * 0.01
        r = minimize(obj, x0, constraints=cons, method="SLSQP", options={"maxiter": 2000, "ftol": 1e-12})
        best = max(best, -r.fun)
    out = {"slsqp": best}
    for route in ("dual", "kkt", "primal", "smoothed"):
        try: out[route] = float(dual.support_values([q], d, route=route)[0])
        except Exception as e: out[route] = f"raised {type(e).__name__}"
    print(label, {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in out.items()})
run(EuclideanSpace(60), "euclid 60")
run(make_dense_metric_space(60), "dense 60")
from pygeoinf2.tests.conftest import make_weighted_space, WeightedSpace
run(WeightedSpace(np.linspace(0.5, 4.0, 60)), "weighted 60")
