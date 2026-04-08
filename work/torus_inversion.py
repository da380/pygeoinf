import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf

# Swap sphere for torus
from pygeoinf.symmetric_space.torus import (
    Sobolev,
    plot,
    plot_geodesic_network,
)

# Ensure we import the corner plot utility

# Set up the model space
order = 2
scale = 0.2
prior_scale = 0.05

# Initialize Torus Sobolev space
# Positional args: kernel_scale, order, scale
model_space = Sobolev.from_heat_kernel_prior(
    prior_scale,
    order,
    scale,
    radius_x=1.0,
    radius_y=1.0,
    power_of_two=True,
    min_degree=32,
)

print(f"Model space kmax = {model_space.kmax} and dimension = {model_space.dim}")

# Set up the forward operator
print("Setting up the forward problem")
n_sources = 20
n_receivers = 20

# Generate random paths on the Torus
print("Generating random ray paths...")
receivers = [model_space.random_point() for _ in range(n_receivers)]
sources = [model_space.random_point() for _ in range(n_sources)]

paths = [(src, rec) for src in sources for rec in receivers]
print(f"Generated {len(paths)} ray paths.")

forward_operator = model_space.path_average_operator(paths)

# Set up the data errors
data_space = forward_operator.codomain
data_error_measure = inf.GaussianMeasure.from_standard_deviation(data_space, 0.05)

# Set up the forward problem
forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)

# Set up the prior measure
model_prior = model_space.point_value_scaled_heat_kernel_gaussian_measure(prior_scale)

# Generate synthetic data

disk_model = False

if disk_model:
    center = (np.pi, np.pi)
    disk_radius = 1.0
    smoothing_scale = 0.15

    def top_hat_disk(p: tuple[float, float]) -> float:
        return 1.0 if model_space.geodesic_distance(center, p) <= disk_radius else 0.0

    raw_model = model_space.project_function(top_hat_disk)

    smoothing_kernel = model_space.heat_kernel(smoothing_scale)
    smoothing_op = model_space.invariant_automorphism(smoothing_kernel)
    model = smoothing_op(raw_model)

    data = forward_problem.data_measure_from_model(model).sample()

else:
    model, data = forward_problem.joint_measure(model_prior).sample()

# Set up the inverse problem
inverse_problem = inf.LinearBayesianInversion(forward_problem, model_prior)

# Set up the Woodbury preconditioner
print("Building the Woodbury preconditioner")
# Use a lower-resolution surrogate for the preconditioner
surrogate_space = model_space.with_degree(model_space.kmax // 6)
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
print("Solving the problem via CG...")
solver = inf.CGMatrixSolver()
model_posterior = inverse_problem.model_posterior_measure(
    data, solver, preconditioner=precon
)
print(f"Solution in {solver.iterations} iterations")

# Posterior STD
print("Estimating the pointwise STD...")
posterior_std = model_posterior.sample_pointwise_std(100)


# --- Spatial Plotting ---
model_out = model_posterior.expectation
vmax = np.max(np.abs(model))

# Plot True Model
ax1, im1 = plot(
    model_space,
    model,
    colorbar=True,
    symmetric=True,
    cmap="RdBu_r",
    colorbar_kwargs={"label": "True model"},
)
plot_geodesic_network(paths, ax=ax1, alpha=0.1, color="black")
ax1.set_title("True Model (Torus)")

# Plot Posterior Expectation
ax2, im2 = plot(
    model_space,
    model_out,
    colorbar=True,
    symmetric=True,
    cmap="RdBu_r",
    colorbar_kwargs={"label": "Posterior expectation"},
)
plot_geodesic_network(paths, ax=ax2, alpha=0.1, color="black")
ax2.set_title("Posterior Expectation")

# Plot Posterior Standard Deviation
ax3, im3 = plot(
    model_space,
    posterior_std,
    colorbar=True,
    cmap="viridis",
    colorbar_kwargs={"label": "Posterior Standard Deviation"},
)
plot_geodesic_network(paths, ax=ax3, alpha=0.1, color="white")
ax3.set_title("Posterior Standard Deviation")

plt.show()
