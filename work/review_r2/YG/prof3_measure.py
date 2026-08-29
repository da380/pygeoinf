"""Sobolev-measure sampling, pointwise variance, covariance function, two-point covariance."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from yg_util import TransformCounter, bench, fmt
import numpy as np
import time

from pygeoinf2.symmetric_space.sphere import Sobolev as Sob2
import pygeoinf.symmetric_space.sphere as sph1

rng = np.random.default_rng(3)

print("=== transforms per sample draw (lmax 32) ===")
X = Sob2(32, 2.0, 0.2); X1 = sph1.Sobolev(32, 2.0, 0.2)
mu = X.sobolev_measure(2.0, 0.2); mu1 = X1.sobolev_kernel_gaussian_measure(2.0, 0.2)
with TransformCounter() as c: mu.sample(rng=rng)
print("  v2 sobolev_measure.sample:", c)
with TransformCounter() as c: mu1.sample()
print("  v1 sobolev_kernel_gaussian_measure.sample:", c)
with TransformCounter() as c: mu.samples(4, rng=rng)
print("  v2 samples(4):", c)
mu_e = X.sobolev_measure(2.0, 0.2, expectation=X.heat_measure(0.3).sample(rng=rng))
with TransformCounter() as c: mu_e.sample(rng=rng)
print("  v2 with expectation:", c)
with TransformCounter() as c: X.pointwise_variance(mu.covariance.eigenvalues)
print("  v2 pointwise_variance:", c)
with TransformCounter() as c: mu.two_point_covariance(X.reference_point)
print("  v2 two_point_covariance:", c)
with TransformCounter() as c: X.covariance_function(mu, np.linspace(0, 1, 50))
print("  v2 covariance_function (50 distances):", c)
with TransformCounter() as c: X.pointwise_variance_at(mu, X.random_points(5, rng=rng))
print("  v2 pointwise_variance_at exact, 5 points:", c)
with TransformCounter() as c: X.pointwise_variance_at(mu, X.random_points(300, rng=rng), samples=20)
print("  v2 pointwise_variance_at samples=20, 300 points:", c)
with TransformCounter() as c: mu.directional_variance(X.dirac(X.reference_point).representer)
print("  v2 directional_variance(dirac representer):", c)
# correctness cross-check: closed form pointwise variance vs pointwise_variance_at exact
p = X.random_points(3, rng=rng)
exact = X.pointwise_variance_at(mu, p)
closed = X.pointwise_variance(mu.covariance.eigenvalues)
print(f"  pointwise_variance_at exact {exact} vs pointwise_variance {closed:.6g}")
# proposed closed form for a diagonal covariance, no transforms
B = X.basis_matrix(p)
prop = (B**2 * (mu.covariance.eigenvalues / X.metric_values)[None, :]).sum(axis=1)
print(f"  proposed components formula {prop}")

# covariance_function against v1 closed form (Legendre) -- on a Lebesgue space both should agree
L2 = X.with_order(0.0); L21 = sph1.Lebesgue(32)
var = L2.sobolev_symbol(-2.0, 0.2)
nu = L2.invariant_measure(var)
d = np.linspace(0.0, 2.0, 7)
try:
    cf2 = L2.covariance_function(nu, d)
except ValueError as e:
    print("  covariance_function on a Lebesgue space RAISES:", str(e)[:80])
cf1 = L21.invariant_covariance_function(var)(d)
# closed form on the Sobolev space: sum_k s_k phi_k(p) phi_k(q) / g_k with the addition theorem
nuS = X.invariant_measure(var)
cf2S = X.covariance_function(nuS, d)
from scipy.special import eval_legendre
degv = np.array([ (var/X.metric_values)[X.degrees == l][0] for l in range(X.lmax+1)])
coef = degv * (2*np.arange(X.lmax+1)+1) / (4*np.pi*X.radius**2)
closed = np.polynomial.legendre.legval(np.cos(d/X.radius), coef)
print(f"  covariance_function (Sobolev) vs closed form: max diff {np.max(np.abs(cf2S-closed)):.2e} of {np.max(np.abs(closed)):.3g}")

print("\n=== timings ===")
for lmax in (128, 256):
    X = Sob2(lmax, 2.0, 0.2); X1 = sph1.Sobolev(lmax, 2.0, 0.2)
    mu = X.sobolev_measure(2.0, 0.2); mu1 = X1.sobolev_kernel_gaussian_measure(2.0, 0.2)
    s = np.sqrt(mu.covariance.eigenvalues / X.metric_values)
    def proposed_sample():
        return X.from_components(s * rng.standard_normal(X.dim))
    pts = X.random_points(10, rng=rng)
    fns = {
        "v2 sample": lambda: mu.sample(rng=rng),
        "v1 sample": lambda: mu1.sample(),
        "proposed: one synthesis": proposed_sample,
        "v2 pointwise_variance_at exact (10 pts)": lambda: X.pointwise_variance_at(mu, pts),
        "proposed diag formula (10 pts)": lambda: (X.basis_matrix(pts)**2 * (mu.covariance.eigenvalues / X.metric_values)[None, :]).sum(axis=1),
        "v2 covariance_function (50 d)": lambda: X.covariance_function(mu, np.linspace(0, 1, 50)),
        "closed form Legendre (50 d)": lambda: np.polynomial.legendre.legval(np.cos(np.linspace(0,1,50)/X.radius), np.bincount(X.degrees, weights=mu.covariance.eigenvalues/X.metric_values/(2*X.degrees+1))*(2*np.arange(X.lmax+1)+1)/(4*np.pi*X.radius**2)),
        "v2 two_point_covariance": lambda: mu.two_point_covariance(X.reference_point),
    }
    print(f"\n-- lmax {lmax} (dim {X.dim}) --")
    print(fmt(bench(fns, reps=5)))
