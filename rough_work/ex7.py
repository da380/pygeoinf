import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.circle import Sobolev

# Set the components of the model space
kmax = 512
model1_space = Sobolev(kmax, 2, 0.1)
model2_space = Sobolev(kmax, 2, 0.1)

# Form the model space
model_space = inf.HilbertSpaceDirectSum([model1_space, model2_space])

# Set the priors
model1_prior_measure = model1_space.point_value_scaled_heat_kernel_gaussian_measure(
    0.1, 1.0
)
model2_prior_measure = model2_space.point_value_scaled_heat_kernel_gaussian_measure(
    0.01, 1.0
)
model_prior_measure = inf.GaussianMeasure.from_direct_sum(
    [model1_prior_measure, model2_prior_measure]
)

# Set the forward operator
points = model1_space.random_points(1000)
forward_operator1 = model1_space.point_evaluation_operator(points)
forward_operator2 = model2_space.point_evaluation_operator(points)
forward_operator = inf.RowLinearOperator([forward_operator1, forward_operator2])

# Set the data space and data error measure
data_space = forward_operator.codomain
data_error_measure = inf.GaussianMeasure.from_standard_deviation(data_space, 0.1)

# Set the forward problem and generate synthetic data
forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)
model, data = forward_problem.synthetic_model_and_data(model_prior_measure)

# Set up and solve the inverse problem
inverse_problem = inf.LinearBayesianInversion(forward_problem, model_prior_measure)
model_posterior_measure = inverse_problem.model_posterior_measure(
    data, inf.CholeskySolver()
)

# Plot the posterior expectations
model_posterior_expectation = model_posterior_measure.expectation

fig1, ax1 = model1_space.plot(model[0])
model1_space.plot(model_posterior_expectation[0], fig=fig1, ax=ax1)

fig2, ax2 = model2_space.plot(model[1])
model2_space.plot(model_posterior_expectation[1], fig=fig2, ax=ax2)

fig3, ax3 = model1_space.plot(model[0] + model[1])
model1_space.plot(
    model_posterior_expectation[0] + model_posterior_expectation[1], fig=fig3, ax=ax3
)

plt.show()
