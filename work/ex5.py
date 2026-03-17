import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev
import cartopy.crs as ccrs

# Set threads available for backends
inf.configure_threading(n_threads=1)

# Parameters
lmax = 128
order = 2
scale = 0.1
radius = 2

prior_order = 2.0
prior_scale = 0.2

npoints = 50
std = 0.01


# Setup Model Space and Forward Problem
model_space = Sobolev(lmax, order, scale, radius=radius)

points = model_space.random_points(npoints)
lats = [lat for lat, _ in points]
lons = [lon for _, lon in points]
forward_operator = model_space.point_evaluation_operator(points)
data_error_measure = inf.GaussianMeasure.from_standard_deviation(
    forward_operator.codomain, std
)
forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)

# Set the unconstrained prior
unconstrained_model_prior_measure = (
    model_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        prior_order, prior_scale
    )
)


# Setup Constraint
constraint_operator = model_space.to_coefficient_operator(0, lmin=0)
constraint_value = np.array([0])
constraint = inf.AffineSubspace.from_linear_equation(
    constraint_operator, constraint_value, solver=inf.CholeskySolver()
)

# Form the constrained prior
model_prior_measure = constraint.condition_gaussian_measure(
    unconstrained_model_prior_measure
)


# Setup Inversion
inversion = inf.LinearBayesianInversion(forward_problem, model_prior_measure)

# Generate Synthetic Data
model, data = forward_problem.synthetic_model_and_data(model_prior_measure)

# Plot True Model
fig1, ax1, im1 = model_space.plot(model, symmetric=True)
fig1.colorbar(im1, orientation="horizontal", shrink=0.7)
ax1.plot(lons, lats, "k^", markersize=5, transform=ccrs.PlateCarree())
ax1.set_title("True Model")

# Solve Inverse Problem
model_posterior_measure = inversion.model_posterior_measure(data, inf.CholeskySolver())
model_posterior_expectation = model_posterior_measure.expectation

# Plot Posterior Mean
fig2, ax2, im2 = model_space.plot(model_posterior_expectation, symmetric=True)
fig2.colorbar(im2, orientation="horizontal", shrink=0.7)
ax2.plot(lons, lats, "k^", markersize=5, transform=ccrs.PlateCarree())
ax2.set_title("Posterior Mean (Constrained)")


# Estimate the pointwise std from sampling
print("Sampling from posterior")
pointwise_variance = model_posterior_measure.sample_pointwise_variance(
    500, parallel=True, n_jobs=10
)
pointwise_std = pointwise_variance.copy()
pointwise_std.data = np.sqrt(pointwise_std.data)

# Plot Posterior Mean
fig3, ax3, im3 = model_space.plot(pointwise_std, symmetric=True)
fig3.colorbar(im3, orientation="horizontal", shrink=0.7)
ax3.plot(lons, lats, "k^", markersize=5, transform=ccrs.PlateCarree())
ax3.set_title("Pointwise standard deviation")

plt.show()
