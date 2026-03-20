"""
Example 13: Bayesian Inversion with Surrogate-Based Preconditioning

This script demonstrates solving a 1D Bayesian inverse problem on a circle
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
    --n-data             1000      (Number of observation points)
    --prior-scale        0.02      (Length scale of the prior)
    --surrogate-degree   64        (Truncation degree for the surrogate problem)
    --seed               42        (Random seed for reproducibility)
    --precond            'block'   (Type of preconditioner to apply)
    --bandwidth          100       (Bandwidth for banded preconditioner)
    --rank               50        (Rank for spectral preconditioner)
    --incomplete         False     (Use Incomplete LU instead of exact LU)

Usage:
    Run `python ex13.py --help` to see all available command-line arguments.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.circle import (
    Sobolev,
    plot,
    plot_error_bounds,
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


def plot_results(
    space: Sobolev,
    true_model: np.ndarray,
    data: np.ndarray,
    obs_points: np.ndarray,
    data_std: np.ndarray,
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

    plot(space, solution_model, fig=fig, ax=ax, color="b", label=solution_label)

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

    ax.errorbar(obs_points, data, 2 * data_std, fmt="ko", capsize=3, label="Data")
    ax.set_title("Inversion Results", fontsize=16)
    ax.set_xlabel("Angle (radians)")
    ax.set_ylabel("Function Value")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7)
    plt.show()


def main():

    parser = argparse.ArgumentParser(
        description="Bayesian inversion with surrogate preconditioning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Physics & Grid Parameters
    parser.add_argument(
        "--n-data", type=int, default=1000, help="Number of observation points"
    )
    parser.add_argument(
        "--prior-scale", type=float, default=0.02, help="Length scale of the prior"
    )
    parser.add_argument(
        "--surrogate-degree",
        type=int,
        default=64,
        help="Truncation degree for the surrogate problem",
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
        "--bandwidth", type=int, default=100, help="Bandwidth for banded preconditioner"
    )
    parser.add_argument(
        "--rank", type=int, default=50, help="Rank for spectral preconditioner"
    )
    parser.add_argument(
        "--incomplete",
        action="store_true",
        help="Use Incomplete LU (spilu) instead of exact LU",
    )

    args = parser.parse_args()

    # For reproducibility
    np.random.seed(args.seed)

    # ==========================================
    # Setup Base Space & Data Grid
    # ==========================================
    print("Setting up grid and data spaces...")
    base_space = Sobolev.from_sobolev_parameters(2.0, 0.01, power_of_two=True)

    observation_points = base_space.random_points(args.n_data)
    observation_points = np.sort(observation_points)

    # Dummy operator just to quickly grab the codomain size
    dummy_op = base_space.point_evaluation_operator(observation_points)
    data_space = dummy_op.codomain

    standard_deviations = np.random.uniform(0.01, 0.5, data_space.dim)
    data_error_measure = inf.GaussianMeasure.from_standard_deviations(
        data_space, standard_deviations
    )

    # ==========================================
    # 1. Exact Problem Components
    # ==========================================
    model_space, forward_operator, model_prior_measure = setup_problem_components(
        base_space, base_space.kmax, observation_points, args.prior_scale
    )

    forward_problem = inf.LinearForwardProblem(
        forward_operator, data_error_measure=data_error_measure
    )

    print(f"Model space dimension (kmax): {model_space.kmax}")
    print(f"Data space dimension: {data_space.dim}")

    # Generate Synthetic Data
    true_model, data = forward_problem.synthetic_model_and_data(model_prior_measure)
    bayesian_inversion = inf.LinearBayesianInversion(
        forward_problem, model_prior_measure
    )

    # ==========================================
    # 2. Surrogate Problem Components
    # ==========================================
    print(f"Building surrogate problem (degree={args.surrogate_degree})...")
    surrogate_space, surrogate_fwd_op, surrogate_prior = setup_problem_components(
        base_space, args.surrogate_degree, observation_points, args.prior_scale
    )

    surrogate_inv = bayesian_inversion.surrogate_inversion(
        alternate_forward_operator=surrogate_fwd_op,
        alternate_prior_measure=surrogate_prior,
    )
    surrogate_normal_operator = surrogate_inv.normal_operator

    # ==========================================
    # 3. Preconditioner Routing
    # ==========================================
    preconditioner = None
    if args.precond != "none":
        print(f"Initializing {args.precond} preconditioner...")

        if args.precond == "block":
            blocks = model_space.cluster_points(
                observation_points, threshold=5 * args.prior_scale
            )
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
    posterior_mean = model_posterior_measure.expectation

    print(f"Number of CG iterations = {solver.iterations}")

    plot_results(
        model_space,
        true_model,
        data,
        observation_points,
        standard_deviations,
        solution_model=posterior_mean,
        solution_label="Posterior Mean",
        solution_std=None,
    )


if __name__ == "__main__":
    main()
