import matplotlib.pyplot as plt
import pygeoinf as inf

from pygeoinf.symmetric_space.sphere import Sobolev, plot, plot_geodesic_network


# Set up the model space
lmax = 128
order = 2.0
scale = 0.1
model_space = Sobolev(lmax, order, scale)


# Set up the forward operator
n_sources = 2
n_receivers = 10
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
prior_scale = 0.05
model_prior = model_space.point_value_scaled_heat_kernel_gaussian_measure(prior_scale)


# Generate some synthetic data
model, data = forward_problem.synthetic_model_and_data(model_prior)

# Set up the inverse problem
# inverse_problem = inf.LinearBayesianInversion(forward_problem, model_prior)

# Solve the inverse problem
# model_posterior = inverse_problem.model_posterior_measure(data, inf.CholeskySolver())
# model_out = model_posterior.expectation

inverse_problem = inf.LinearMinimumNormInversion(forward_problem)
model_out = inverse_problem.minimum_norm_operator(inf.CGSolver())(data)

# Plot the true model
fig1, ax1, im1 = plot(model)
plot_geodesic_network(paths, ax=ax1)
fig1.colorbar(im1)

# Plot the true model
fig2, ax2, im2 = plot(model_out)
plot_geodesic_network(paths, ax=ax2)
fig2.colorbar(im2)

# Plot the pointwise std
# model_posterior_std = model_posterior.sample_pointwise_std(500)
# fig3, ax3, im3 = plot(model_posterior_std)
# plot_geodesic_network(paths, ax=ax3)
# fig3.colorbar(im3)

plt.show()
