import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.circle import Sobolev


# Set up the model space.
# We define the space of possible solutions. Here, we use a Sobolev space
# on a circle. This space contains functions that are twice-differentiable
# (order=2) with a characteristic length scale of 0.05. The `from_sobolev_parameters`
# method automatically determines the necessary Fourier resolution (kmax).
model_space = Sobolev.from_sobolev_parameters(2, 0.05)


# Set the sample points randomly.
# We will observe the "true" function at 20 random points on the circle.
n = 20
observation_points = model_space.random_points(n)

# Set the forward operator using a method of the Sobolev class.
# The `point_evaluation_operator` is a linear operator that takes a function
# from our model space and returns its values at the specified points.
forward_operator = model_space.point_evaluation_operator(observation_points)
data_space = forward_operator.codomain

# Set the data error measure. If standard deviation is zero, the data is
# free of observational errors.
# We assume the observations are corrupted by Gaussian noise with a
# standard deviation of 0.1. We represent this with a GaussianMeasure.
standard_deviation = 0.1
data_error_measure = (
    inf.GaussianMeasure.from_standard_deviation(data_space, standard_deviation)
    if standard_deviation > 0
    else None
)

# Set up the forward problem.
# The `LinearForwardProblem` object bundles the operator and the error model.
# This fully defines the relationship: `data = operator(model) + error`.
forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)

# Define a prior measure on the model space.
# We define a prior belief about what the "true" function looks like before
# seeing any data. Here, we use a `heat_gaussian_measure`, which generates
# smooth functions. We give it a mean of zero and a pointwise amplitude of 1.
model_prior_measure = model_space.heat_gaussian_measure(0.1, 1)

# Sample a model and corresponding data.
# To test our inversion, we first create a "true" model by drawing a random
# sample from our prior. Then, we generate the corresponding noisy data.
model, data = forward_problem.synthetic_model_and_data(model_prior_measure)

# Set up the inversion method.
# We set up the Bayesian inversion, providing the forward problem and our prior.
bayesian_inversion = inf.LinearBayesianInversion(forward_problem, model_prior_measure)

# Get the posterior distribiution.
# We solve the inversion to get the posterior distribution. The posterior
# represents our updated belief about the model after incorporating the data.
# A Cholesky solver is used for the underlying matrix inversion.
model_posterior_measure = bayesian_inversion.model_posterior_measure(
    data, inf.CholeskySolver()
)

# Estimate the pointwise variance.
# To visualize the posterior uncertainty, we create a low-rank approximation
# of the posterior, which allows us to draw samples efficiently.
low_rank_posterior_approximation = model_posterior_measure.low_rank_approximation(
    10, method="variable", rtol=1e-4
)
# We then estimate the pointwise variance by drawing many samples from this approximation.
model_pointwise_variance = low_rank_posterior_approximation.sample_pointwise_variance(
    100
)
model_pointwise_std = np.sqrt(model_pointwise_variance)


# Plot the results.
# Create the final plot, starting with the true underlying model.
fig, ax = model_space.plot(
    model, color="k", figsize=(15, 10), linestyle="--", label="True Model"
)
# Overlay the noisy data points that were used for the inversion.
ax.errorbar(
    observation_points,
    data,
    2 * standard_deviation,
    fmt="ko",
    capsize=2,
    label="Noisy Data",
)
# Plot the posterior mean, which is our best estimate of the true model.
model_space.plot(
    model_posterior_measure.expectation,
    fig=fig,
    ax=ax,
    color="b",
    label="Posterior Mean",
)
# Plot the 2-sigma uncertainty bounds around the posterior mean.
model_space.plot_error_bounds(
    model_posterior_measure.expectation,
    2 * model_pointwise_std,
    fig=fig,
    ax=ax,
    alpha=0.2,
    color="b",
    label="Posterior (2 std dev)",
)
# Add titles and labels for clarity
ax.set_title("Bayesian Inversion Results")
ax.set_xlabel("Angle (radians)")
ax.set_ylabel("Function Value")
ax.legend()
# Display the final plot.
plt.show()
