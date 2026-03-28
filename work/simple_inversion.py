import matplotlib.pyplot as plt
import pygeoinf as inf
from cartopy import crs as ccrs

from pygeoinf.symmetric_space.sphere import Sobolev, plot, plot_geodesic_network


# Set up the model space
order = 2.0
scale = 0.1
prior_scale = 0.05
model_space = Sobolev.from_heat_kernel_prior(
    prior_scale, order, scale, power_of_two=True, min_degree=64
)


# Set up the forward operator
print("Setting up the forward problem")
n_sources = 5
n_receivers = 100
paths = model_space.random_source_receiver_paths(n_sources, n_receivers)
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
model, data = forward_problem.synthetic_model_and_data(model_prior)

# Set up the inverse problem
inverse_problem = inf.LinearBayesianInversion(forward_problem, model_prior)


# Set up the preconditioner
print("Builing the preconditioner")
surrogate_degree = model_space.degree // 4
surrogate_space = model_space.with_degree(surrogate_degree)
surrogate_operator = surrogate_space.path_average_operator(paths)
surrogate_prior = surrogate_space.point_value_scaled_heat_kernel_gaussian_measure(
    prior_scale
)
surrogate_problem = inverse_problem.surrogate_inversion(
    alternate_forward_operator=surrogate_operator,
    alternate_prior_measure=surrogate_prior,
)
surrogate_normal = surrogate_problem.normal_operator
precon = inf.ColumnThresholdedPreconditioningMethod(1e-3, incomplete=True)(
    surrogate_normal
)


# Solve the inverse problem
print("Solving the problem")
solver = inf.CGMatrixSolver()
model_posterior = inverse_problem.model_posterior_measure(
    data, solver, preconditioner=precon
)
print(f"Solution in {solver.iterations} iterations")

model_out = model_posterior.expectation


# Plot the true model
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(16, 16), subplot_kw={"projection": ccrs.PlateCarree()}
)


_, im1 = plot(model, ax=ax1)
plot_geodesic_network(paths, ax=ax1, alpha=0.1)
ax1.set_title("True model")
fig.colorbar(im1)


_, im2 = plot(model_out, ax=ax2)
plot_geodesic_network(paths, ax=ax2, alpha=0.1)
ax2.set_title("Posterior expectation")
fig.colorbar(im2)


plt.show()
