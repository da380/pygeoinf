import numpy as np
import matplotlib.pyplot as plt

from cartopy import crs as ccrs

from pygeoinf import (
    GaussianMeasure,
    LinearForwardProblem,
    LinearLeastSquaresInversion,
    LinearMinimumNormInversion,
    CGSolver,
    CholeskySolver,
)

from pygeoinf.symmetric_space.sphere import Sobolev


# Set the model space.
X = Sobolev(128, 2.0, 0.4)

# Set up the prior distribution.
mu = X.sobolev_gaussian_measure(2.0, 0.1, 1)


# Set the observation points
n = 50
points = X.random_points(n)
lats = [point[0] for point in points]
lons = [point[1] for point in points]

# Set the forward operator.
A = X.point_evaluation_operator(points)
Y = A.codomain

# Set the error distribution
sigma = 0.1
nu = GaussianMeasure.from_standard_deviation(Y, sigma) if sigma > 0 else None


# Set up forward problem.
forward_problem = LinearForwardProblem(A, nu)

# Make synthetic data
u, v = forward_problem.synthetic_model_and_data(mu)


inversion = LinearMinimumNormInversion(forward_problem)
B = inversion.minimum_norm_operator(CGSolver() if sigma > 0 else CholeskySolver())
w = B(v)


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


fig, ax, im = X.plot(w, vmin=-umax, vmax=umax)
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
