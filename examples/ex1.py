import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.circle import Sobolev, CircleHelper


# --- Setup the forward problem (same as Tutorial 1) ---
model_space = Sobolev.from_sobolev_parameters(2.0, 0.01)
n_data = 30
observation_points = model_space.random_points(n_data)


forward_operator = model_space.point_evaluation_operator(observation_points)

data_space = forward_operator.codomain
standard_deviation = 0.1
data_error_measure = inf.GaussianMeasure.from_standard_deviation(
    data_space, standard_deviation
)
forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)

# --- Generate synthetic data ---
model_prior_measure = model_space.point_value_scaled_heat_kernel_gaussian_measure(
    scale=0.1, amplitude=1.0
)
true_model, data = forward_problem.synthetic_model_and_data(model_prior_measure)

print("Forward problem and synthetic data are ready.")


# Set up the inversion object
lsq_inversion = inf.LinearLeastSquaresInversion(forward_problem)
solver = inf.CGSolver(rtol=1e-9)

# --- Solve for different damping parameters ---
damping = 1

print("Building preconditioner")
normal_operator = lsq_inversion.normal_operator(damping)
diagonal_normal_operator = normal_operator.diagonal()


print("Solving normal equations")
solution = lsq_inversion.least_squares_operator(
    damping,
    solver,  # preconditioner=diagonal_normal_operator.inverse
)(data)
# solution = (
#    diagonal_normal_operator.inverse
#    @ forward_operator.adjoint
#    @ data_error_measure.inverse_covariance
# )(data)


# --- Plot the results ---
fig, ax = model_space.plot(
    true_model, color="k", linestyle="--", label="True Model", figsize=(15, 10)
)
ax.errorbar(
    observation_points, data, 2 * standard_deviation, fmt="ko", capsize=3, label="Data"
)


model_space.plot(
    solution,
    fig=fig,
    ax=ax,
    color="r",
    linestyle="-",
)

ax.set_title("Least-Squares Solutions for Different Damping Parameters", fontsize=16)
ax.legend()
ax.grid(True, linestyle=":", alpha=0.7)
plt.show()
