import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from cartopy import crs as ccrs

import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import (
    Sobolev,
    plot,
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
lmax = 128
order = 2.0
scale = 0.025
model_space = Sobolev(lmax, order, scale)

observation_points = model_space.random_points(n_data)

lats = [lat for lat, _ in observation_points]
lons = [lon for _, lon in observation_points]


forward_operator = model_space.point_evaluation_operator(observation_points)
data_space = forward_operator.codomain

standard_deviation = 0.1
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
    scale, std=1.0
)
true_model, data = forward_problem.synthetic_model_and_data(model_prior_measure)


# ==========================================
# 3. Setup Bayesian Inversion & Preconditioner
# ==========================================
bayesian_inversion = inf.LinearBayesianInversion(forward_problem, model_prior_measure)

# Fixed parameters for this experiment
n_clusters = args.nclusters
max_cov_distance = 20.0 * model_space.scale

if PRECOND_TYPE == "block_diagonal" or PRECOND_TYPE == "sparse_localized":
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
        max_cov_distance,
    )
else:
    raise ValueError(f"Unknown preconditioner type: {PRECOND_TYPE}")

t_build = time.time() - t0
print(f"Preconditioner built in {t_build:.4f} seconds.")


# ==========================================
# 4. Solve the Inverse Problem
# ==========================================
print("\nSolving normal equations via Conjugate Gradient...")
solver = inf.CGSolver()


t0 = time.time()
model_posterior_measure = bayesian_inversion.model_posterior_measure(
    data, solver, preconditioner=preconditioner
)
posterior_expectation = model_posterior_measure.expectation
t_solve = time.time() - t0

print(f"Solve completed in {t_solve:.4f} seconds.")
print(f"Solve took {solver.iterations} iterations.")

fig1, ax1, im1 = plot(true_model, projection=ccrs.Robinson())
ax1.plot(lons, lats, "k^", markersize=5, transform=ccrs.PlateCarree(), alpha=0.1)

fig2, ax2, im2 = plot(posterior_expectation, projection=ccrs.Robinson())
ax2.plot(lons, lats, "k^", markersize=5, transform=ccrs.PlateCarree(), alpha=0.1)

plt.show()
