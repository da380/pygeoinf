import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev

# Set the model space
lmax = 128
order = 2.0
scale = 0.05

model_space = Sobolev(lmax, order, scale)

# Set up the forward problem
sources = 4
receivers = 50

paths = model_space.random_source_receiver_paths(sources, receivers)
forward_operator = model_space.path_average_operator(paths)

data_space = forward_operator.codomain
data_error_measure = inf.GaussianMeasure.from_standard_deviation(data_space, 0.01)

forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)


# Set a prior and make synthetic data
prior_order = 2
prior_scale = 0.05
model_prior_measure = model_space.point_value_scaled_sobolev_kernel_gaussian_measure(
    prior_order, prior_scale
)
model, data = forward_problem.synthetic_model_and_data(model_prior_measure)

# Plot the true model
fig1, ax1, im1 = model_space.plot(model, projection=ccrs.Robinson(), coasts=True)
ax1.set_title("True model", y=1.1)
model_space.plot_geodesic_network(paths, ax=ax1)
cbar1 = fig1.colorbar(im1, ax=ax1, orientation="horizontal", pad=0.05, shrink=0.7)


# Set and solve the inverse problem
inverse_problem = inf.LinearBayesianInversion(forward_problem, model_prior_measure)
model_posterior_measure = inverse_problem.model_posterior_measure(
    data, inf.CholeskySolver()
)

# Plot the posterior expectation
fig2, ax2, im2 = model_space.plot(
    model_posterior_measure.expectation, projection=ccrs.Robinson(), coasts=True
)
ax2.set_title("Posterior expectation", y=1.1)
model_space.plot_geodesic_network(paths, ax=ax2)
cbar2 = fig2.colorbar(im2, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.7)

# Estimate the pointwise standard deviation
model_std = model_posterior_measure.sample_pointwise_std(100)

fig3, ax3, im3 = model_space.plot(model_std, projection=ccrs.Robinson(), coasts=True)
ax3.set_title("Posterior pointwise std", y=1.1)
model_space.plot_geodesic_network(paths, ax=ax3)
cbar3 = fig3.colorbar(im3, ax=ax3, orientation="horizontal", pad=0.05, shrink=0.7)

plt.show()
