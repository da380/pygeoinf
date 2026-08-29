"""Example 21 end to end (no plotting) under cProfile, lmax 48."""
import cProfile, pstats, sys, time, io
sys.path.insert(0, "/home/david/dev/pygeoinf")
import numpy as np
from pygeoinf2.inference import LinearGaussianInversion, LinearForwardProblem
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.symmetric_space.sphere import Sobolev

def run():
    rng = np.random.default_rng(1)
    t = {}
    t0 = time.perf_counter()
    X = Sobolev(48, 2.0, 0.1)
    receivers = X.stations(count=24, rng=rng)
    sources = X.earthquakes(count=40, minimum_magnitude=5.5, rng=rng)
    paths = [(s, r) for s in sources for r in receivers]
    t["setup"] = time.perf_counter() - t0; t0 = time.perf_counter()
    forward = X.path_average_operator(paths, count=16, dense=True)
    t["path_average_operator(dense)"] = time.perf_counter() - t0; t0 = time.perf_counter()
    noise = GaussianMeasure.from_standard_deviation(forward.codomain, 0.02)
    problem = LinearForwardProblem(forward, error=noise)
    prior = X.heat_measure(0.17, pointwise_std=0.05)
    truth, data = problem.synthetic_model_and_data(prior, rng=rng)
    chi = problem.chi_squared(truth, data)
    t["prior+synthetic+chi2"] = time.perf_counter() - t0; t0 = time.perf_counter()
    estimator = LinearGaussianInversion(problem, prior)
    t["LinearGaussianInversion()"] = time.perf_counter() - t0; t0 = time.perf_counter()
    posterior = estimator(data)
    t["estimator(data)"] = time.perf_counter() - t0; t0 = time.perf_counter()
    err = X.norm(X.subtract(posterior.expectation, truth)) / X.norm(truth)
    fit = problem.chi_squared(posterior.expectation, data)
    t["error+fit"] = time.perf_counter() - t0; t0 = time.perf_counter()
    caps = X.geodesic_ball_average_operator(sources[:4], 0.15, dense=True)
    t["geodesic_ball_average_operator"] = time.perf_counter() - t0; t0 = time.perf_counter()
    property_posterior = estimator.push_forward(caps)(data)
    t["push_forward(caps)(data)"] = time.perf_counter() - t0; t0 = time.perf_counter()
    dev = np.sqrt(np.diag(property_posterior.covariance.matrix(form="components")))
    t["property covariance.matrix (4x4)"] = time.perf_counter() - t0; t0 = time.perf_counter()
    s = posterior.sample(rng=rng)
    t["posterior.sample()"] = time.perf_counter() - t0; t0 = time.perf_counter()
    return t, X.dim, len(paths), err, fit, chi

pr = cProfile.Profile()
pr.enable()
t, dim, npaths, err, fit, chi = run()
pr.disable()
print(f"dim {dim}, paths {npaths}, rel err {err:.3f}, fit {fit:.0f}, chi2 truth {chi:.0f}")
for k, v in t.items():
    print(f"  {k:36s} {v:8.3f} s")
print(f"  total {sum(t.values()):.3f} s")
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(28)
print(s.getvalue())
