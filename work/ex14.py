"""
Example 14: Bayesian Inversion with Surrogate-Based Preconditioning

This script demonstrates solving a 2D Bayesian inverse problem on a sphere
using the pygeoinf library. To accelerate the iterative solver, it utilizes
a "surrogate" problem—a cheaper, lower-resolution version of the physics and
prior—to construct a highly efficient preconditioner.

Available Preconditioning Strategies:
    - block:    Uses an exact matrix-vector probed block-diagonal matrix based
                on spatial clustering (KD-Tree).
    - banded:   Extracts a symmetric band of diagonals from the exact operator.
    - spectral: Uses a randomized eigendecomposition to invert the dominant
                modes of the operator.
    - none:     Runs the CG solver without any preconditioning.

Default Parameters:
    --n-data             100       (Number of observation points)
    --prior-scale        0.2       (Length scale of the prior)
    --surrogate-degree   32        (Truncation degree for the spatial surrogate)
    --seed               42        (Random seed for reproducibility)
    --precond            'block'   (Type of preconditioner to apply)
    --bandwidth          10        (Bandwidth for banded preconditioner)
    --rank               20        (Rank for spectral preconditioner)
    --incomplete         False     (Use Incomplete LU instead of exact LU)
    --threshold          5         (distance relative to prior-scale used in forming blocks)

Low-Rank Surrogate Options (applied ON TOP of the spatial surrogate):
    --lr-forward         None      (Rank for randomized SVD of the forward operator)
    --lr-prior           None      (Rank for randomized eigendecomposition of the prior)
    --lr-data-error      None      (Rank for randomized eigendecomposition of the data error)

Usage:
    Run `python ex14.py --help` to see all available command-line arguments.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import (
    Sobolev,
    plot,
)


def setup_problem_components(
    base_space: Sobolev, degree: int, obs_points: np.ndarray, prior_scale: float
):
    """
    Helper function to generate the space, forward operator, and prior measure
    for a specific truncation degree.
    """
    space = base_space.with_degree(degree)
    forward_operator = space.point_evaluation_operator(obs_points)
    prior_measure = space.point_value_scaled_heat_kernel_gaussian_measure(
        prior_scale, std=1.0
    )
    return space, forward_operator, prior_measure


def main():

    parser = argparse.ArgumentParser(
        description="Bayesian inversion with stacked surrogate preconditioning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Physics & Grid Parameters
    parser.add_argument(
        "--n-data", type=int, default=100, help="Number of observation points"
    )
    parser.add_argument(
        "--prior-scale", type=float, default=0.2, help="Length scale of the prior"
    )
    parser.add_argument(
        "--surrogate-degree",
        type=int,
        default=32,
        help="Truncation degree for the spatial surrogate problem",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Preconditioner Parameters
    parser.add_argument(
        "--precond",
        type=str,
        choices=["block", "banded", "spectral", "none"],
        default="block",
        help="Type of preconditioner to apply",
    )
    parser.add_argument(
        "--bandwidth", type=int, default=10, help="Bandwidth for banded preconditioner"
    )
    parser.add_argument(
        "--rank", type=int, default=20, help="Rank for spectral preconditioner"
    )
    parser.add_argument(
        "--incomplete",
        action="store_true",
        help="Use Incomplete LU (spilu) instead of exact LU",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5,
        help="Relative distance used in forming blocks",
    )

    # Low-Rank Surrogate Parameters
    parser.add_argument(
        "--lr-forward",
        type=int,
        default=None,
        help="Rank for low-rank forward operator surrogate",
    )
    parser.add_argument(
        "--lr-prior",
        type=int,
        default=None,
        help="Rank for low-rank prior measure surrogate",
    )
    parser.add_argument(
        "--lr-data-error",
        type=int,
        default=None,
        help="Rank for low-rank data error measure surrogate",
    )

    args = parser.parse_args()

    # For reproducibility
    np.random.seed(args.seed)

    # ==========================================
    # Setup Base Space & Data Grid
    # ==========================================
    print("Setting up grid and data spaces...")
    base_space = Sobolev(128, 2.0, args.prior_scale)

    observation_points = base_space.random_points(args.n_data)
    lats = [lat for lat, _ in observation_points]
    lons = [lon for _, lon in observation_points]

    # Dummy operator just to quickly grab the codomain size
    dummy_op = base_space.point_evaluation_operator(observation_points)
    data_space = dummy_op.codomain

    standard_deviations = np.random.uniform(0.2, 0.5, data_space.dim)
    data_error_measure = inf.GaussianMeasure.from_standard_deviations(
        data_space, standard_deviations
    )

    # ==========================================
    # 1. Exact Problem Components
    # ==========================================
    model_space, forward_operator, model_prior_measure = setup_problem_components(
        base_space, base_space.degree, observation_points, args.prior_scale
    )

    forward_problem = inf.LinearForwardProblem(
        forward_operator, data_error_measure=data_error_measure
    )

    print(f"Model space dimension (kmax): {model_space.dim}")
    print(f"Data space dimension: {data_space.dim}")

    # Generate Synthetic Data
    true_model, data = forward_problem.synthetic_model_and_data(model_prior_measure)
    bayesian_inversion = inf.LinearBayesianInversion(
        forward_problem, model_prior_measure
    )

    # ==========================================
    # 2. Surrogate Problem Components
    # ==========================================
    print(f"Building spatial surrogate problem (degree={args.surrogate_degree})...")

    # Step A: Always build the lower-degree spatial surrogate first
    _, surrogate_fwd_op, surrogate_prior = setup_problem_components(
        base_space, args.surrogate_degree, observation_points, args.prior_scale
    )

    surrogate_inv = bayesian_inversion.surrogate_inversion(
        alternate_forward_operator=surrogate_fwd_op,
        alternate_prior_measure=surrogate_prior,
    )

    # Step B: If requested, apply low-rank approximations ON TOP of the spatial surrogate
    if args.lr_forward or args.lr_prior or args.lr_data_error:
        print("Applying low-rank approximations to the spatial surrogate...")
        surrogate_inv = surrogate_inv.low_rank_surrogate(
            forward_rank=args.lr_forward,
            prior_rank=args.lr_prior,
            data_error_rank=args.lr_data_error,
        )

    surrogate_normal_operator = surrogate_inv.normal_operator

    # ==========================================
    # 3. Preconditioner Routing
    # ==========================================
    preconditioner = None
    if args.precond != "none":
        print(f"Initializing {args.precond} preconditioner...")

        if args.precond == "block":
            print("Forming blocks...")
            # Note: Clustering happens on the surrogate space points
            blocks = model_space.cluster_points(
                observation_points, threshold=args.threshold * args.prior_scale
            )
            print("Building preconditioner...")
            solver_wrapper = inf.ExactBlockPreconditioningMethod(
                blocks, incomplete=args.incomplete
            )
        elif args.precond == "banded":
            solver_wrapper = inf.BandedPreconditioningMethod(
                args.bandwidth, incomplete=args.incomplete
            )
        elif args.precond == "spectral":
            solver_wrapper = inf.SpectralPreconditioningMethod(rank=args.rank)

        preconditioner = solver_wrapper(surrogate_normal_operator)

    # ==========================================
    # 4. Bayesian Inversion & Plotting
    # ==========================================
    print("Solving linear system...")
    solver = inf.CGMatrixSolver()
    model_posterior_measure = bayesian_inversion.model_posterior_measure(
        data, solver, preconditioner=preconditioner
    )
    posterior_expectation = model_posterior_measure.expectation

    print(f"Number of CG iterations = {solver.iterations}")

    fig1, ax1, im1 = plot(true_model, projection=ccrs.Robinson(), coasts=True)
    ax1.plot(lons, lats, "k^", markersize=5, transform=ccrs.PlateCarree(), alpha=0.1)
    ax1.set_title("True Model & Observation Points")

    fig2, ax2, im2 = plot(
        posterior_expectation, projection=ccrs.Robinson(), coasts=True
    )
    ax2.plot(lons, lats, "k^", markersize=5, transform=ccrs.PlateCarree(), alpha=0.1)
    ax2.set_title("Posterior Mean")

    plt.show()


if __name__ == "__main__":
    main()
