import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.plane import Sobolev, plot, plot_geodesic_network

# Set threads available for backends.
inf.configure_threading(n_threads=1)

# --- Set up the model space ---
ORDER = 2.0
SCALE = 0.02
PRIOR_SCALE = 0.02

model_space = Sobolev.from_heat_kernel_prior(
    PRIOR_SCALE,
    ORDER,
    SCALE,
    power_of_two=True,
    min_degree=64,
)

print(f"Model space dimension {model_space.dim}")

# --- Set up the forward problem ---
print("Setting up forward problem...")
N_SOURCES = 10
N_RECEIVERS = 500
receivers = model_space.random_points(N_RECEIVERS)
sources = model_space.random_points(N_SOURCES)
paths = [(src, rec) for src in sources for rec in receivers]

NOISE_STD = 0.1


forward_operator = model_space.path_average_operator(paths)
data_space = forward_operator.codomain
data_error_measure = inf.GaussianMeasure.from_standard_deviation(data_space, NOISE_STD)
forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)


print(f"Data space dimension {data_space.dim}")


# visualise the adjoint mapping


i = np.random.randint(0, data_space.dim - 1)
data = data_space.basis_vector(i)
model = forward_operator.adjoint(data)

_, ax0 = plt.subplots(figsize=(16, 14))
_, im0 = plot(
    model_space,
    model,
    symmetric=True,
    ax=ax0,
    colorbar=True,
    colorbar_kwargs={"label": f"Sensitivity kernel for the {i}th path"},
)
plot_geodesic_network(paths, ax=ax0, alpha=0.05)


# --- Set up the inverse problem ---

# Set the prior.
model_prior = model_space.point_value_scaled_heat_kernel_gaussian_measure(PRIOR_SCALE)

# Generate synthetic data.
model, data = forward_problem.joint_measure(model_prior).sample()

# Set up the inverse problem.
inverse_problem = inf.LinearBayesianInversion(forward_problem, model_prior)

# --- Build the preconditioner ---

# Set up the surrogate problem
print("Setting up surrogate forward problem...")


surr_model_space = model_space.with_degree(model_space.degree // 6)
surr_forward_operator = surr_model_space.path_average_operator(paths)
raw_surr_model_prior = surr_model_space.point_value_scaled_heat_kernel_gaussian_measure(
    PRIOR_SCALE
)

woodbury_solver = inf.CholeskySolver(galerkin=True, parallel=True, n_jobs=8)
damped_surr_prior = raw_surr_model_prior.with_regularized_inverse(
    woodbury_solver, damping=1.0e-6
)

preconditioner = inverse_problem.surrogate_woodbury_data_preconditioner(
    woodbury_solver,
    alternate_forward_operator=surr_forward_operator,
    alternate_prior_measure=damped_surr_prior,
)


print(f"Surrogate model space dimension {surr_model_space.dim}")


print("Building the preconditioner...")
woodbury_solver = inf.CholeskySolver(galerkin=True)
damped_surr_prior = raw_surr_model_prior.with_regularized_inverse(
    woodbury_solver, damping=1.0e-6
)
preconditioner = inverse_problem.surrogate_woodbury_data_preconditioner(
    woodbury_solver,
    alternate_forward_operator=surr_forward_operator,
    alternate_prior_measure=damped_surr_prior,
)


# --- Solve the inverse problem ---
print("Solving for the posterior...")


solver = inf.CGSolver(rtol=0.01 * NOISE_STD)
model_posterior = inverse_problem.model_posterior_measure(
    data, solver, preconditioner=preconditioner
)


print(f"Solution in {solver.iterations} iterations")

print("Sampling to estimate pointwise std")

N_SAMPLES = 200
pointwise_std = model_posterior.sample_pointwise_std(N_SAMPLES, parallel=True, n_jobs=8)

# --- plot the results ---
print("Visualising the results...")

_, ax1 = plt.subplots(figsize=(16, 14))
_, im1 = plot(
    model_space,
    model,
    ax=ax1,
    symmetric=True,
    colorbar=True,
    colorbar_kwargs={"label": "True model"},
)
plot_geodesic_network(paths, ax=ax1, alpha=0.05)

_, ax2 = plt.subplots(figsize=(16, 14))
_, im2 = plot(
    model_space,
    model_posterior.expectation,
    ax=ax2,
    symmetric=True,
    colorbar=True,
    colorbar_kwargs={"label": "Posterior expectation"},
)
plot_geodesic_network(paths, ax=ax2, alpha=0.05)
im2.set_clim(im1.get_clim())

_, ax3 = plt.subplots(figsize=(16, 14))
plot(
    model_space,
    pointwise_std,
    ax=ax3,
    colorbar=True,
    cmap="Blues",
    colorbar_kwargs={"label": "Pointwise STD"},
)
plot_geodesic_network(paths, ax=ax3, alpha=0.05)


# Set up the property operator


centers = [(0.2, 0.2), (0.5, 0.8), (0.7, 0.3)]
widths = [0.05, 0.02, 0.1]


def make_gaussian_average(center, sigma):
    xc, yc = center
    # norm = 1.0 / (2 * np.pi * sigma**2)

    def g(p):
        x, y = p
        return np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2 * sigma**2))

    return g


weighting_functions = []
summed_weights = model_space.zero
for center, sigma in zip(centers, widths):
    func = make_gaussian_average(center, sigma)
    weighting_function = model_space.project_function(func)
    weighting_functions.append(weighting_function)
    summed_weights += weighting_function


_, ax4 = plt.subplots(figsize=(16, 14))
plot(
    model_space,
    summed_weights,
    ax=ax4,
    colorbar=True,
    cmap="Blues",
    colorbar_kwargs={"label": "Combined weights"},
)
plot_geodesic_network(paths, ax=ax4, alpha=0.05)

property_operator = model_space.l2_products_operator(weighting_functions)


property_prior = model_prior.affine_mapping(operator=property_operator)
property_posterior = model_posterior.affine_mapping(operator=property_operator)


labels = [f"Region {i+1}\n(σ={w})" for i, w in enumerate(widths)]

inf.plot_corner_distributions(
    property_posterior,
    prior_measure=property_prior,
    true_values=property_operator(model),
    labels=labels,
    parallel=True,
    n_jobs=3,
)

plt.show()
