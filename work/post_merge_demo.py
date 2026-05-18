"""Demo: post-merge additions on convex-analysis-pr.

Three self-contained sections; run them all or skip to any one.

    conda activate inferences3
    cd pygeoinf
    python work/post_merge_demo.py

Section A  — Exact 2D credible-set plots for Ball and Ellipsoid
             (new plot.py quadratic-slice renderer; no sampling, no grid)

Section B  — Function-space credible sets: ambient_ball and weakened_ellipsoid
             (high-dimensional Euclidean proxy for a function space)

Section C  — Exact spherical-cap integral functional
             (new sphere.spherical_cap_integral; returns a LinearForm)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from pygeoinf import GaussianMeasure
from pygeoinf.hilbert_space import EuclideanSpace

# ─────────────────────────────────────────────────────────────────────────────
# Section A: Exact 2D credible-set plots
# ─────────────────────────────────────────────────────────────────────────────
print("=== Section A: Exact 2D credible-set plots ===")

space = EuclideanSpace(2)
cov   = np.array([[4.0, 2.0],
                  [2.0, 2.0]])
mu    = GaussianMeasure.from_covariance_matrix(space, cov)

# Each call returns a typed Ellipsoid; .plot() dispatches to the exact
# quadratic-slice renderer — the boundary curve is computed analytically.
e90  = mu.credible_set(0.90)
e95  = mu.credible_set(0.95)
e99  = mu.credible_set(0.99)

fig, ax = plt.subplots(figsize=(5, 5))
for e, c in [(e99, "steelblue"), (e95, "cornflowerblue"), (e90, "lightblue")]:
    e.plot(ax=ax, alpha=0.25, show_plot=False)
ax.set_title("Exact Mahalanobis ellipsoids (no sampling)")
ax.set_aspect("equal")
ax.legend(["99%", "95%", "90%"], loc="upper left")

plt.tight_layout()
plt.savefig("work/figures/demo_A_exact_plots.png", dpi=120)
plt.close()
print("  Saved work/figures/demo_A_exact_plots.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section B: Function-space credible sets
# ─────────────────────────────────────────────────────────────────────────────
# Use a high-dimensional Euclidean space whose covariance has eigenvalues
# decaying like 1/(k*pi)^2 — an exact analogue of the inverse 1D Laplacian.
print("\n=== Section B: Function-space credible sets ===")

N = 200
k = np.arange(1, N + 1, dtype=float)
eigvals = 1.0 / (k * np.pi) ** 2          # decaying spectrum
cov_matrix = np.diag(eigvals)

space_N = EuclideanSpace(N)
mu_fs   = GaussianMeasure.from_covariance_matrix(space_N, cov_matrix)

# ambient_ball — calibrates ||f||_H <= r_p using the weighted chi-squared
# distribution of the covariance spectrum.  spectrum_size controls how many
# eigenvalues are used; the rest are approximated.
ambient = mu_fs.ambient_ball(0.95, spectrum_size=50)
print(f"  ambient_ball   radius = {ambient.radius:.4f}")

# weakened_ellipsoid — uses C^{1-theta} as the inner-product metric.
# theta=0 is the ordinary Mahalanobis ellipsoid; theta=1 is the ambient ball.
weakened = mu_fs.weakened_ellipsoid(0.95, theta=0.5, spectrum_size=50)
print(f"  weakened_ellipsoid (theta=0.5)  radius = {weakened.radius:.4f}")

# Sweep theta to see how the radius varies
print("\n  Radius vs theta (how the credible set grows as we relax the metric):")
for th in [0.1, 0.3, 0.5, 0.7, 0.9]:
    r = mu_fs.weakened_ellipsoid(0.95, theta=th, spectrum_size=50).radius
    print(f"    theta={th:.1f}  r={r:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Section C: Exact spherical-cap integral on the sphere
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section C: Exact spherical-cap integral ===")

from pygeoinf.symmetric_space.sphere import Lebesgue as SphericalLebesgue

# Earth-radius sphere; lmax=10 truncated harmonic basis (dim = (lmax+1)^2 = 121)
sphere = SphericalLebesgue(10, radius=6371.0)

center         = (0.0, 0.0)         # equatorial cap center (lat, lon) in degrees
angular_radius = np.radians(20.0)   # 20-degree angular radius

# Both return LinearForms — they are cheap to create (no sampling).
# Given any model field f ∈ sphere, calling form(f) gives the cap statistic.
cap_integral = sphere.spherical_cap_integral(center, angular_radius)
cap_average  = sphere.spherical_cap_average(center, angular_radius)

cap_area = 2.0 * np.pi * sphere.radius**2 * (1.0 - np.cos(angular_radius))
print(f"  Cap area (analytic) = {cap_area:.3e} km^2")
print(f"  cap_integral has {cap_integral.components.shape[0]} non-zero basis coefficients")
print(f"  cap_average  has {cap_average.components.shape[0]} basis coefficients")

# Demonstrate the ratio: cap_integral = cap_area * cap_average
# (component-wise, they should differ only by the area scale factor)
ratio = cap_integral.components / cap_average.components
print(f"  integral/average ratio (should be uniform ≈ {cap_area:.3e}): "
      f"min={ratio.min():.3e}  max={ratio.max():.3e}")

print("\nDone.")
