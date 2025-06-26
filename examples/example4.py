import numpy as np
import matplotlib.pyplot as plt


from cartopy import crs as ccrs

from pygeoinf import (
    LinearBayesianInversion,
    CholeskySolver,
    GaussianMeasure,
    LinearForwardProblem,
)
from pygeoinf.homogeneous_space.sphere import Sobolev


# Set the model space.
X = Sobolev(64, 2, 0.25)


# Set up the prior distribution.
mu = X.heat_gaussian_measure(0.4, 1)


# Set the observation points
n = 50
points = X.random_points(n)
lats = [point[0] for point in points]
lons = [point[1] for point in points]

# Set the forward operator.
A = X.point_evaluation_operator(points)
Y = A.codomain


# Set up the error distribution.
sigma = 0.1
nu = GaussianMeasure.from_standard_deviation(Y, sigma) if sigma > 0 else None


forward_problem = LinearForwardProblem(A, nu)
inverse_problem = LinearBayesianInversion(forward_problem, mu)


# Generate synthetic data.
u, v = forward_problem.synthetic_model_and_data(mu)


solver = CholeskySolver()
pi = inverse_problem.model_posterior_measure(v, solver)

umax = np.max(np.abs(u.data))


fig, ax, im = X.plot(u, vmin=-umax, vmax=umax)
fig.colorbar(im, ax=ax, orientation="horizontal")
ax.plot(
    lons,
    lats,
    "o",
    color="k",
    markersize=4,
    transform=ccrs.PlateCarree(),
)


fig, ax, im = X.plot(pi.expectation, vmin=-umax, vmax=umax)
fig.colorbar(im, ax=ax, orientation="horizontal")
ax.plot(
    lons,
    lats,
    "o",
    color="k",
    markersize=4,
    transform=ccrs.PlateCarree(),
)


plt.show()
