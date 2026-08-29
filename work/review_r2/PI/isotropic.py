import sys, time
import numpy as np
sys.path.insert(0, "/home/david/dev/pygeoinf")
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
D = EuclideanSpace(3000)
iso = GaussianMeasure.from_standard_deviation(D, 0.3)
print("covariance type:", type(iso.covariance).__name__, "| _diagonal_eigenvalues:", iso._diagonal_eigenvalues() is not None)
diag = GaussianMeasure(D, covariance_factor=DiagonalLinearOperator(D, 0.3*np.ones(3000)), precision_factor=DiagonalLinearOperator(D, np.ones(3000)/0.3))
print("diag covariance type:", type(diag.covariance).__name__, "| _diagonal_eigenvalues:", diag._diagonal_eigenvalues() is not None)
for name, mu in (("from_standard_deviation", iso), ("DiagonalLinearOperator factor", diag)):
    t = time.perf_counter(); c = mu.log_normalising_constant(); t1 = time.perf_counter() - t
    exact = -0.5*3000*np.log(2*np.pi) - 0.5*3000*np.log(0.09)
    print(f"{name:32s} log_normalising_constant {c:.3f} (exact {exact:.3f}) in {t1:.3f} s")
    t = time.perf_counter(); v = mu.nuclear_norm(); t1 = time.perf_counter() - t
    print(f"{name:32s} nuclear_norm {v:.3f} in {t1:.3f} s")
    t = time.perf_counter()
    try:
        r = mu.ambient_ball(level=0.9).radius; print(f"{name:32s} ambient_ball radius {r:.3f} in {time.perf_counter()-t:.3f} s")
    except Exception as e:
        print(f"{name:32s} ambient_ball raised {type(e).__name__}: {str(e)[:80]}")
    t = time.perf_counter()
    try:
        k = mu.kl_divergence(2.0*mu); print(f"{name:32s} kl_divergence(mu, 2mu) {k:.3f} in {time.perf_counter()-t:.3f} s")
    except Exception as e:
        print(f"{name:32s} kl_divergence(auto) raised {type(e).__name__}: {str(e)[:60]}")
