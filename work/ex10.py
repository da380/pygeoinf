import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import pygeoinf as inf
from pygeoinf.symmetric_space.circle import (
    Sobolev,
    plot,
    plot_error_bounds,
)

# ==========================================
# 0. Command Line Configuration
# ==========================================
parser = argparse.ArgumentParser(
    description="Run 1D Bayesian Inversion with varying preconditioners."
)
parser.add_argument(
    "--precond",
    type=str,
    choices=["none", "block_diagonal", "sparse_localized", "distance_localized"],
    default="distance_localized",
    help="The type of preconditioner to use.",
)
parser.add_argument(
    "--ndata", type=int, default=200, help="Number of observation points."
)
parser.add_argument(
    "--nclusters",
    type=int,
    default=20,
    help="Number of clusters for block-based preconditioners.",
)


args = parser.parse_args()

PRECOND_TYPE = args.precond
n_data = args.ndata


# For reproducibility
np.random.seed(42)

# ==========================================
# 1. Setup Model Space & Forward Problem
# ==========================================
model_space = Sobolev.from_sobolev_parameters(2.0, 0.005)

observation_points = model_space.random_points(n_data)


forward_operator = model_space.point_evaluation_operator(observation_points)
data_space = forward_operator.codomain

standard_deviation = 0.01
data_error_measure = inf.GaussianMeasure.from_standard_deviation(
    data_space, standard_deviation
)

forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)

# ==========================================
# 2. Prior Measure & Synthetic Data
# ==========================================
model_prior_measure = model_space.point_value_scaled_heat_kernel_gaussian_measure(
    0.01, std=1.0
)
true_model, data = forward_problem.synthetic_model_and_data(model_prior_measure)


# ==========================================
# 3. Setup Bayesian Inversion & Preconditioner
# ==========================================
bayesian_inversion = inf.LinearBayesianInversion(forward_problem, model_prior_measure)

# Fixed parameters for this experiment
n_clusters = args.nclusters
max_cov_distance = 19.0 * model_space.scale

if PRECOND_TYPE == "block_diagonal" or PRECOND_TYPE == "sparse_localised":
    # Cluster the observation points for the block-based preconditioners
    blocks = model_space.cluster_points(observation_points, n_clusters=n_clusters)
    print(
        f"Clustered {n_data} observation points into {len(blocks)} interacting blocks."
    )

print(f"\nBuilding preconditioner: {PRECOND_TYPE}...")
t0 = time.time()

if PRECOND_TYPE == "none":
    preconditioner = None

elif PRECOND_TYPE == "block_diagonal":
    preconditioner = bayesian_inversion.diagonal_normal_preconditioner(
        blocks=blocks, parallel=False
    )

elif PRECOND_TYPE == "sparse_localized":
    preconditioner = bayesian_inversion.sparse_localized_preconditioner(
        interacting_blocks=blocks, rank=5, parallel=False
    )

elif PRECOND_TYPE == "distance_localized":
    preconditioner = model_space.distance_localized_preconditioner(
        model_prior_measure,
        observation_points,
        data_error_measure,
        max_distance=max_cov_distance,
        parallel=False,
    )
else:
    raise ValueError(f"Unknown preconditioner type: {PRECOND_TYPE}")

t_build = time.time() - t0
print(f"Preconditioner built in {t_build:.4f} seconds.")


# ==========================================
# 4. Solve the Inverse Problem
# ==========================================
print("\nSolving normal equations via Conjugate Gradient...")
solver = inf.CGSolver()  # Adjust tolerance as needed

t0 = time.time()
model_posterior_measure = bayesian_inversion.model_posterior_measure(
    data, solver, preconditioner=preconditioner
)
# Force evaluation of the expectation to realize the CG solve time
posterior_mean = model_posterior_measure.expectation
t_solve = time.time() - t0

print(f"Solve completed in {t_solve:.4f} seconds.")
print(f"Solve took {solver.iterations} iterations.")

# Estimate the pointwise variance from posterior samples
# print("Sampling posterior to estimate variance...")
# posterior_pointwise_variance = model_posterior_measure.sample_pointwise_variance(100)
# posterior_std = np.sqrt(posterior_pointwise_variance)


# ==========================================
# 5. Plotting
# ==========================================
def plot_results(space, true_m, d, obs_pts, d_std, sol_m, sol_label, sol_std=None):
    fig, ax = plot(
        space, true_m, color="k", linestyle="--", label="True Model", figsize=(12, 8)
    )
    plot(space, sol_m, fig=fig, ax=ax, color="b", label=sol_label)
    if sol_std is not None:
        plot_error_bounds(
            space, sol_m, 2 * sol_std, fig=fig, ax=ax, alpha=0.2, color="b"
        )

    ax.errorbar(obs_pts, d, 2 * d_std, fmt="ko", capsize=3, label="Data")
    ax.set_title(
        f"Inversion Results ({PRECOND_TYPE})\nSolve Time: {t_solve:.2f}s", fontsize=14
    )
    ax.set_xlabel("Angle (radians)")
    ax.set_ylabel("Function Value")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7)
    plt.show()


plot_results(
    model_space,
    true_model,
    data,
    observation_points,
    standard_deviation,
    sol_m=posterior_mean,
    sol_label="Posterior Mean",
    # sol_std=posterior_std,
)
