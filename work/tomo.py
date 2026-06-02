import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import (
    Sobolev,
    create_map_figure,
    plot,
    plot_points,
    plot_geodesic_network,
)


# Global Tomography Parameters
lmax = 128
order = 2
scale = 0.1
prior_scale = 0.005
n_sources = 30
n_receivers = n_sources

print("Global parameters initialized.")


model_space = Sobolev(
    lmax,
    order,
    scale,
)
print(f"Initialized model space with dimension {model_space.dim}")


# 1. Generate random paths (sources and receivers)
# model_space.random_point() handles 1D floats or 2D tuples automatically
receivers = model_space.random_points(n_receivers)
sources = model_space.random_points(n_sources)
paths = [(src, rec) for src in sources for rec in receivers]

# 2. Define the Forward Operator and Data Noise
forward_operator = model_space.path_average_operator(paths)
data_error_measure = inf.GaussianMeasure.from_standard_deviation(
    forward_operator.codomain, 0.05
)

# 3. Combine into a Linear Forward Problem
forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)  #

# 4. Define the Prior
model_prior = model_space.point_value_scaled_heat_kernel_gaussian_measure(prior_scale)

print(f"Forward problem set up with {len(paths)} ray paths.")


# Generate a consistent (model, data) pair
model_true, data_obs = forward_problem.joint_measure(model_prior).sample()

print("Synthetic model and data generated.")

# 1. Initialize the Bayesian Inversion
inverse_problem = inf.LinearBayesianInversion(forward_problem, model_prior)

# 2. Build the Woodbury Surrogate Preconditioner
surrogate_space = model_space.with_degree(model_space.degree // 6)
raw_surrogate_prior = surrogate_space.point_value_scaled_heat_kernel_gaussian_measure(
    prior_scale
)
woodbury_solver = inf.CholeskySolver(galerkin=True)
damped_surrogate_prior = raw_surrogate_prior.with_regularized_inverse(
    woodbury_solver, damping=1e-6
)
precon = inverse_problem.surrogate_inversion(
    alternate_forward_operator=surrogate_space.path_average_operator(paths),
    alternate_prior_measure=damped_surrogate_prior,
).woodbury_data_preconditioner(woodbury_solver)


# 3. Solve for the Posterior Expectation
solver = inf.CGMatrixSolver()
model_posterior = inverse_problem.model_posterior_measure(
    data_obs, solver, preconditioner=precon
)

print(f"Solution found in {solver.iterations} iterations.")


# Get the two-point covariance functions
x = model_space.random_point()
k_prior = model_prior.two_point_covariance(x)
k_posterior = model_posterior.two_point_covariance(x)


# Plot the results
fig1, ax1 = create_map_figure(figsize=(8, 5))


plot(
    model_true,
    ax=ax1,
    colorbar=True,
    symmetric=True,
    coasts=True,
    cmap="seismic",
    colorbar_kwargs={
        "label": "True model",
    },
)

plot_geodesic_network(paths, ax=ax1, alpha=0.1)


fig2, ax2 = create_map_figure(figsize=(8, 5))

plot(
    model_posterior.expectation,
    ax=ax2,
    colorbar=True,
    symmetric=True,
    coasts=True,
    cmap="seismic",
    colorbar_kwargs={
        "label": "Posterior expectation",
    },
)


plot_geodesic_network(paths, ax=ax2, alpha=0.1)


fig3, ax3 = create_map_figure(figsize=(8, 5))

plot(
    k_prior,
    ax=ax3,
    colorbar=True,
    symmetric=True,
    coasts=True,
    cmap="seismic",
    colorbar_kwargs={
        "label": "Prior covariance function",
    },
)


plot_points([x], ax=ax3, color="black", s=15, zorder=5)
plot_geodesic_network(paths, ax=ax3, alpha=0.1)


fig4, ax4 = create_map_figure(figsize=(8, 5))

plot(
    k_posterior,
    ax=ax4,
    colorbar=True,
    symmetric=True,
    coasts=True,
    cmap="seismic",
    colorbar_kwargs={
        "label": "Posterior covariance function",
    },
)

plot_points([x], ax=ax4, color="black", s=15, zorder=5)
plot_geodesic_network(paths, ax=ax4, alpha=0.1)


plt.show()
