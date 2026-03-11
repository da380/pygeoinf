import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.circle import (
    Sobolev,
    plot,
    plot_error_bounds,
)

# For reproducibility
np.random.seed(42)

# ==========================================
# 1. Setup Model Space & Forward Problem
# ==========================================
model_space = Sobolev.from_sobolev_parameters(2.0, 0.05)

n_data = 50
observation_points = model_space.random_points(n_data)
forward_operator = model_space.point_evaluation_operator(observation_points)
data_space = forward_operator.codomain

standard_deviation = 0.5
data_error_measure = inf.GaussianMeasure.from_standard_deviation(
    data_space, standard_deviation
)

forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)

print(f"Model space dimension (kmax): {model_space.kmax}")
print(f"Data space dimension: {data_space.dim}")

# ==========================================
# 2. Generate Synthetic Data
# ==========================================
model_prior_measure = model_space.point_value_scaled_heat_kernel_gaussian_measure(
    0.1, std=1.0
)
true_model, data = forward_problem.synthetic_model_and_data(model_prior_measure)


# ==========================================
# 3. Plotting Helper Function
# ==========================================
def plot_results(
    space: Sobolev,
    true_model: np.ndarray,
    data: np.ndarray,
    obs_points: np.ndarray,
    data_std: float,
    solution_model: np.ndarray,
    solution_label: str,
    solution_std: np.ndarray = None,
):
    """Helper function to create a consistent plot."""
    fig, ax = plot(
        space,
        true_model,
        color="k",
        linestyle="--",
        label="True Model",
        figsize=(15, 10),
    )

    # Plot the solution
    plot(space, solution_model, fig=fig, ax=ax, color="b", label=solution_label)

    # Plot uncertainty bounds if provided
    if solution_std is not None:
        plot_error_bounds(
            space,
            solution_model,
            2 * solution_std,
            fig=fig,
            ax=ax,
            alpha=0.2,
            color="b",
        )

    # Plot the noisy data points
    ax.errorbar(obs_points, data, 2 * data_std, fmt="ko", capsize=3, label="Data")

    ax.set_title("Inversion Results", fontsize=16)
    ax.set_xlabel("Angle (radians)")
    ax.set_ylabel("Function Value")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7)

    plt.show()


# ==========================================
# 4. Bayesian Inversion
# ==========================================
bayesian_inversion = inf.LinearBayesianInversion(forward_problem, model_prior_measure)

solver = inf.CholeskySolver(galerkin=True)
model_posterior_measure = bayesian_inversion.model_posterior_measure(data, solver)
posterior_mean = model_posterior_measure.expectation

# Estimate the pointwise variance from posterior samples
posterior_pointwise_variance = model_posterior_measure.sample_pointwise_variance(1000)
posterior_std = np.sqrt(posterior_pointwise_variance)

# ==========================================
# 5. Final Comparison Plot
# ==========================================
plot_results(
    model_space,
    true_model,
    data,
    observation_points,
    standard_deviation,
    solution_model=posterior_mean,
    solution_label="Posterior Mean",
    solution_std=posterior_std,
)
