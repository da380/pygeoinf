import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import (
    Sobolev,
    plot,
)


# ==========================================
# 1. Setup Model Space & Forward Problem
# ==========================================
model_space = Sobolev(32, 2.0, 0.1)

n_data = 100
observation_points = model_space.random_points(n_data)

blocks = model_space.cluster_points(observation_points, n_clusters=50)
print(f"Clustered {n_data} observation points into {len(blocks)} interacting blocks.")

forward_operator = model_space.point_evaluation_operator(observation_points)
data_space = forward_operator.codomain

standard_deviation = 0.1
data_error_measure = inf.GaussianMeasure.from_standard_deviation(
    data_space, standard_deviation
)

forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)

print(f"Model space dimension (kmax): {model_space.dim}")
print(f"Data space dimension: {data_space.dim}")

# ==========================================
# 2. Generate Synthetic Data
# ==========================================
model_prior_measure = model_space.point_value_scaled_heat_kernel_gaussian_measure(
    0.1, std=1.0
)
true_model, data = forward_problem.synthetic_model_and_data(model_prior_measure)


# ==========================================
# 4. Bayesian Inversion
# ==========================================
bayesian_inversion = inf.LinearBayesianInversion(forward_problem, model_prior_measure)


print("Building sparse localized preconditioner...")
preconditioner = bayesian_inversion.sparse_localized_preconditioner(
    interacting_blocks=blocks,
    rank=5,
    parallel=False,
)
print("Preconditioner built successfully.")

solver = inf.CGSolver()
model_posterior_measure = bayesian_inversion.model_posterior_measure(
    data, solver, preconditioner=preconditioner
)
posterior_expectation = model_posterior_measure.expectation
print(solver.iterations)

fig1, ax1, im1 = plot(true_model)
# ax1.scatter(observation_points, marker="o", color="k")

fig2, ax2, im2 = plot(posterior_expectation)
# ax2.scatter(observation_points, marker="o", color="k")

plt.show()
