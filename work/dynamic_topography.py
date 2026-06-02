import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev, plot, plot_points


# Physical constants
g = 9.8
G = 6.6743e-11
H = 2e4
b = 6.371e6
E = 9e10
nu = 0.25
rho_l = 3000
rho_m = 3400

# Flexural rigidity
D = E * H**3 / (12 * (1 - nu**2))


# Prior parameters
sigma_order = 2.0
sigma_scale = 1.0e5
sigma_std = 1e6

rho_order = 2.0
rho_scale = 1.0e6
rho_std = 1e1


space = Sobolev(128, 2.0, 0.1 * b, radius=b)

sigma_meas = space.point_value_scaled_sobolev_kernel_gaussian_measure(
    sigma_order, sigma_scale, std=sigma_std
)


rho_meas = space.point_value_scaled_sobolev_kernel_gaussian_measure(
    rho_order, rho_scale, std=rho_std
)

to_km = 1e-3


rho = rho_meas.sample()
sigma = sigma_meas.sample()


flexural_operator = space.invariant_automorphism(
    lambda k: 1 / (D * k**2 + (rho_m - rho_l) * g)
)

load = rho * g * H + sigma

w = flexural_operator(load)

plot(
    sigma,
    colorbar=True,
    symmetric=True,
    colorbar_kwargs={"label": "Normal traction (N m$^{-2}$)"},
    coasts=True,
)

plot(
    rho,
    colorbar=True,
    symmetric=True,
    colorbar_kwargs={"label": "Density perturbation (kg m$^{-3}$)"},
    coasts=True,
)

plot(
    w * to_km,
    colorbar=True,
    symmetric=True,
    colorbar_kwargs={"label": "Dynamic topography (km)"},
    coasts=True,
)


plt.show()
