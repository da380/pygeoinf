import sys, time
import numpy as np
sys.path.insert(0, "/home/david/dev/pygeoinf")
from pygeoinf2.inference import LinearGaussianInversion, LinearForwardProblem
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.symmetric_space.sphere import Sobolev
from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
from pygeoinf2.numerics.solvers import CholeskySolver
from pygeoinf2.traits import Traits
rng = np.random.default_rng(1)
X = Sobolev(48, 2.0, 0.1)
receivers = X.stations(count=24, rng=rng); sources = X.earthquakes(count=40, minimum_magnitude=5.5, rng=rng)
paths = [(s, r) for s in sources for r in receivers]
forward = X.path_average_operator(paths, count=16, dense=True)
noise = GaussianMeasure.from_standard_deviation(forward.codomain, 0.02)
problem = LinearForwardProblem(forward, error=noise)
prior = X.heat_measure(0.17, pointwise_std=0.05)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)
print("with_traits keeps DiagonalLinearOperator:", isinstance(prior.covariance.with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE), DiagonalLinearOperator))
est = LinearGaussianInversion(problem, prior)
t = time.perf_counter(); ld = est.normal_log_determinant(method="dense"); td = time.perf_counter() - t
print(f"log det dense (dim {problem.data_space.dim}): {ld.value:.3f} in {td:.2f} s")
t = time.perf_counter(); ls = est.normal_log_determinant(method="stochastic", rng=rng); ts = time.perf_counter() - t
print(f"log det stochastic (100 probes): {ls.value:.3f} +/- {ls.standard_error:.3f} in {ts:.2f} s")
t = time.perf_counter(); ls2 = est.normal_log_determinant(method="stochastic", rng=rng, sample_rtol=1e-3, max_samples=400); ts2 = time.perf_counter() - t
print(f"log det stochastic sample_rtol=1e-3: {ls2.value:.3f} +/- {ls2.standard_error:.3f} ({ls2.samples} probes) in {ts2:.2f} s")
# Cholesky-solved estimator: how long does construction (assembly+factorisation) take, and evidence after it?
t = time.perf_counter(); estc = LinearGaussianInversion(problem, prior, solver=CholeskySolver()); tc = time.perf_counter() - t
t = time.perf_counter(); estc.log_evidence(data); te = time.perf_counter() - t
t = time.perf_counter(); estc(data); tm = time.perf_counter() - t
print(f"Cholesky: construction {tc:.2f} s, log_evidence (re-assembles for the dense log det) {te:.2f} s, est(data) {tm:.4f} s")
