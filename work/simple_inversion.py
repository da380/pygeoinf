import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from cartopy import crs as ccrs

from pygeoinf.symmetric_space.sphere import (
    Sobolev,
    plot,
    plot_geodesic_network,
)

# Ensure we import the corner plot utility
from pygeoinf.plot import plot_corner_distributions


# Set up the model space
order = 1.5
scale = 0.1
prior_scale = 0.05

# Strictly positional initialization based on our recent fixes!
model_space = Sobolev.from_heat_kernel_prior(
    prior_scale, order, scale, power_of_two=True, min_degree=64
)
print(
    f"Model space truncation degree = {model_space.degree} and dimension = {model_space.dim}"
)


# Set up the forward operator
print("Setting up the forward problem")
n_sources = 20
n_receivers = 40

# Fetch real-world geometry
print(f"Fetching {n_receivers} IRIS stations and {n_sources} USGS events...")
receivers = model_space.iris_stations(n_stations=n_receivers)
sources = model_space.random_earthquakes(n_points=n_sources, min_magnitude=6.5)

# Create all unique source-receiver pairs
paths = [(src, rec) for src in sources for rec in receivers]
print(f"Generated {len(paths)} real-world ray paths.")

forward_operator = model_space.path_average_operator(paths)


# Set up the data errors
data_space = forward_operator.codomain
data_error_measure = inf.GaussianMeasure.from_standard_deviation(data_space, 0.1)

# Set up the forward problem
forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)

# Set up the prior measure
model_prior = model_space.point_value_scaled_heat_kernel_gaussian_measure(prior_scale)

# Generate some synthetic data
model, data = forward_problem.joint_measure(model_prior).sample()

# Set up the inverse problem
inverse_problem = inf.LinearBayesianInversion(forward_problem, model_prior)


# Set up the preconditioner
print("Building the preconditioner")
surrogate_degree = model_space.degree // 6
surrogate_space = model_space.with_degree(surrogate_degree)
surrogate_operator = surrogate_space.path_average_operator(paths)
surrogate_prior = surrogate_space.point_value_scaled_heat_kernel_gaussian_measure(
    prior_scale
)
surrogate_problem = inverse_problem.surrogate_inversion(
    alternate_forward_operator=surrogate_operator,
    alternate_prior_measure=surrogate_prior,
)
precon = surrogate_problem.woodbury_data_preconditioner()


# Solve the inverse problem
print("Solving the problem")
solver = inf.CGMatrixSolver()
model_posterior = inverse_problem.model_posterior_measure(
    data, solver, preconditioner=precon
)
print(f"Solution in {solver.iterations} iterations")


# Posterior STD
print("Estimating the pointwise STD")
posterior_std = model_posterior.sample_pointwise_std(100)


# Corner Plot Section
print("Generating corner plot for degree 2 coefficients")

mapping_op = model_space.to_coefficient_operator(2, lmin=2)
prior_l2 = model_prior.affine_mapping(operator=mapping_op)
posterior_l2 = model_posterior.affine_mapping(operator=mapping_op)

true_l2 = mapping_op(model)

labels_l2 = [r"$C_{2,0}$", r"$C_{2,1}$", r"$C_{2,2}$", r"$C_{2,-1}$", r"$C_{2,-2}$"]

axes_corner = plot_corner_distributions(
    posterior_l2,
    prior_measure=prior_l2,
    true_values=true_l2,
    labels=labels_l2,
    title="Posterior Distribution: Degree 2 Coefficients",
)

# --- Spatial Plotting ---
model_out = model_posterior.expectation

vmax = np.max(np.abs(model.data))

ax1, im1 = plot(
    model,
    projection=ccrs.Robinson(),
    colorbar=True,
    coasts=True,
    vmax=vmax,
    vmin=-vmax,
    colorbar_kwargs={"label": "True model"},
)
plot_geodesic_network(paths, ax=ax1, alpha=0.05)
ax1.set_title("True Model")


ax2, im2 = plot(
    model_out,
    colorbar=True,
    projection=ccrs.Robinson(),
    coasts=True,
    vmax=vmax,
    vmin=-vmax,
    colorbar_kwargs={"label": "Posterior expectation"},
)
plot_geodesic_network(paths, ax=ax2, alpha=0.05)
ax2.set_title("Posterior Expectation")


ax3, im3 = plot(
    posterior_std,
    colorbar=True,
    projection=ccrs.Robinson(),
    coasts=True,
    cmap="Blues",
    colorbar_kwargs={"label": "Posterior Standard Deviation"},
)
plot_geodesic_network(paths, ax=ax3, alpha=0.05)
ax3.set_title("Posterior Standard Deviation")


plt.show()
