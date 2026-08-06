"""Visualising a correlated pair of Gaussian random fields on the sphere.

Sets up a joint measure for two fields (u, v) on a spherical Sobolev space
with different marginal smoothness and a correlation that decays with the
Laplacian eigenvalue, so that the fields share their large-scale structure
while their small scales are nearly independent.

Produces three figures:

1. sphere_samples.png            -- two joint draws of the pair (u, v);
2. sphere_covariance_maps.png    -- the covariance functions c_uu(x0, .),
                                    c_uv(x0, .) and c_vv(x0, .), obtained by
                                    applying the covariance blocks to the
                                    Dirac representation at a point x0;
3. sphere_covariance_profiles.png -- the same functions evaluated along a
                                    meridian through x0, as profiles in the
                                    angular separation.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from pygeoinf.symmetric_space import sphere
from pygeoinf.symmetric_space.symmetric_space import (
    CorrelatedInvariantGaussianMeasure,
)

np.random.seed(3)

# ------------------------------------------------------------------ #
# The joint measure
# ------------------------------------------------------------------ #
# A Sobolev space of order 2 > d/2, so that Dirac measures have
# representations within the space and point evaluation is bounded.
X = sphere.Sobolev(64, 2.0, 0.2)

# Marginals with a common length-scale but different smoothness. Each is
# rescaled to unit pointwise variance, so that the covariance functions
# plotted below peak at exactly one and are directly comparable. The
# pointwise variance c(x0, x0) is read off by applying the covariance to
# the Dirac representation at x0 and evaluating the result there.
point = (30.0, 40.0)
delta = X.dirac_representation(point)
evaluate_at_point = X.point_evaluation_operator([point])

mu_u = X.invariant_gaussian_measure(X.sobolev_kernel(1.5, 0.1))
mu_v = X.invariant_gaussian_measure(X.sobolev_kernel(3.0, 0.1))
for name, marginal in [("u", mu_u), ("v", mu_v)]:
    variance = np.asarray(evaluate_at_point(marginal.covariance(delta)))[0]
    if name == "u":
        mu_u = marginal * (1.0 / np.sqrt(variance))
    else:
        mu_v = marginal * (1.0 / np.sqrt(variance))


def correlation_profile(lam):
    """Strongly correlated large scales, nearly independent small ones."""
    return -0.95 * np.exp(-lam / 500.0)


mu = CorrelatedInvariantGaussianMeasure.from_invariant_measures(
    [mu_u, mu_v], correlation_profile
)

cross_correlation_at_zero = np.asarray(
    evaluate_at_point(mu.cross_covariance(0, 1)(delta))
)[0]
print(
    f"pointwise cross-correlation at zero separation: "
    f"{cross_correlation_at_zero:.2f}"
)

degrees = np.arange(X.lmax + 1)
rho_by_degree = correlation_profile(degrees * (degrees + 1) / X.radius**2)
print("correlation by spherical harmonic degree:")
for l in (2, 5, 10, 20, 40):
    print(f"  l = {l:2d}:  rho = {rho_by_degree[l]:.2f}")

# ------------------------------------------------------------------ #
# Figure 1: joint samples
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(
    2,
    2,
    figsize=(11, 6.5),
    subplot_kw={"projection": ccrs.Robinson()},
    layout="constrained",
)

for row in range(2):
    u, v = mu.sample()
    for col, (field, name) in enumerate([(u, "u (rough)"), (v, "v (smooth)")]):
        sphere.plot(field, ax=axes[row, col], symmetric=True, gridlines=False)
        axes[row, col].set_title(f"Draw {row + 1}: field {name}")

fig.suptitle("Joint draws: shared large-scale structure, independent small scales")
fig.savefig("sphere_samples.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ #
# Figure 2: covariance functions through Dirac representations
# ------------------------------------------------------------------ #
# For a point x0, the representation d of the Dirac measure satisfies
# <d, w> = w(x0) for all w, and so applying a covariance block C_ij to d
# yields the function y -> Cov(u_i(x0), u_j(y)).
covariance_functions = [
    (mu.cross_covariance(0, 0)(delta), r"$c_{uu}(x_0,\, \cdot\,)$"),
    (mu.cross_covariance(0, 1)(delta), r"$c_{uv}(x_0,\, \cdot\,)$"),
    (mu.cross_covariance(1, 1)(delta), r"$c_{vv}(x_0,\, \cdot\,)$"),
]

# extent = [
#    point[1] - 60.0,
#    point[1] + 60.0,
#    point[0] - 50.0,
#    point[0] + 50.0,
# ]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13, 4.2),
    subplot_kw={"projection": ccrs.PlateCarree()},
    layout="constrained",
)

for ax, (field, label) in zip(axes, covariance_functions):
    sphere.plot(
        field,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        #       map_extent=extent,
        gridlines=False,
        colorbar=True,
        colorbar_kwargs={"shrink": 0.9},
    )
    sphere.plot_points([point], ax=ax, color="black", marker="o", s=25)
    ax.set_title(label)

fig.suptitle("Covariance functions from the Dirac representation at $x_0$")
fig.savefig("sphere_covariance_maps.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ #
# Figure 3: profiles in angular separation along a meridian
# ------------------------------------------------------------------ #
separations = np.linspace(0.0, 60.0, 121)
profile_points = [(point[0] - t, point[1]) for t in separations]
evaluate = X.point_evaluation_operator(profile_points)

fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
for field, label in covariance_functions:
    ax.plot(separations, np.asarray(evaluate(field)), label=label)

ax.axhline(0.0, color="gray", linewidth=0.8)
ax.set_xlabel(r"angular separation from $x_0$ (degrees)")
ax.set_ylabel("covariance")
ax.set_title("Covariance profiles along a meridian through $x_0$")
ax.legend()
fig.savefig("sphere_covariance_profiles.png", dpi=150)
plt.close(fig)

print(
    "wrote sphere_samples.png, sphere_covariance_maps.png, "
    "sphere_covariance_profiles.png"
)
