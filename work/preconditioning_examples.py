"""
Bayesian Inversion with Surrogate-Based Preconditioning

This script demonstrates how to solve a large-scale, 2D Bayesian inverse problem
on a spherical manifold using the pygeoinf library.

The core challenge in high-resolution Bayesian inversion is the computational
cost of applying the dense normal operator during iterative Krylov subspace
methods (like Conjugate Gradient). To drastically accelerate convergence, this
script utilizes a "surrogate" problem—a cheaper, lower-resolution representation
of the forward physics and prior measure—to construct highly efficient preconditioners.

The script supports both uncorrelated (white) and spatially correlated data noise,
allowing for robust testing of the solver under realistic observational conditions.

Available Preconditioning Strategies:
    - none:               Runs the CG solver without any preconditioning (baseline).
    - dense:              Based on Cholesky factorisation of dense matrix normal operator
                          for the surrogate system.
    - block:              Builds an exact, block-diagonal preconditioner by clustering
                          observation points using a K-D tree. Probes the surrogate
                          operator to form the localized blocks.
    - banded:             Extracts a symmetric band of diagonals from the exact surrogate
                          normal operator.
    - spectral:           Constructs a low-rank approximation of the surrogate operator
                          using randomized eigendecomposition to invert the dominant modes.
    - distance-localized: Constructs an ultra-fast, sparse data-space preconditioner.
                          It calculates a 1D covariance function based on geodesic distance
                          and interpolates it. Can optionally apply a Gaspari-Cohn taper
                          to guarantee positive definiteness.
    - sparse:             Based on a sparse approximation of the normal operator.
    - woodbury            Based on the model-space normal operator.

Command-Line Arguments:

    [Problem & Grid Parameters]
    --n-data             (int)   Number of observation points.
    --base-degree        (int)   Truncation degree for the 'true' high-res model.
    --sobolev-order      (float) Smoothness order for the Sobolev space.
    --surrogate-degree   (int)   Truncation degree for the spatial surrogate.

    [Statistical Parameters]
    --prior-scale        (float) Non-dimensional length scale of the prior.
    --prior-std          (float) Standard deviation of the prior.
    --seed               (int)   Random seed for reproducibility.

    [Noise Parameters]
    --noise-amplitude-factor  (float) Relative amplitude (standard deviation) of the noise.
    --noise-scale-factor      (float) Relative length scale for correlated noise.

    [Preconditioner Parameters]
    --precond            (str)   Preconditioner type to apply.
    --bandwidth          (int)   Bandwidth for the 'banded' preconditioner.
    --rank               (int)   Rank for the 'spectral' preconditioner.
    --incomplete         (bool)  Use Incomplete LU (ILU) instead of exact LU for block/banded.
    --threshold          (float) Relative distance (multiplier of prior-scale) used to form
                                 blocks or determine sparsity max-distance.
    --loc-apply-taper    (bool)  Apply Gaspari-Cohn taper for 'distance-localized' precond.
    --sparse-threshold   (float) Relative cutoff for the 'sparse' preconditioner. Elements
                                 smaller than this threshold * diagonal are dropped.
    --sparse-max-nnz     (int)   Maximum number of non-zeros to retain per column for the
                                 'sparse' preconditioner.

    [Low-Rank Surrogate Options] (Applied ON TOP of the spatial surrogate)
    --lr-forward         (int)   Rank for randomized SVD of the forward operator.
    --lr-prior           (int)   Rank for randomized eigendecomposition of the prior.
    --lr-data-error      (int)   Rank for randomized eigendecomposition of the data error.

    [Plotting Parameters]
    --std-samples        (int)   Number of samples to use for plotting pointwise STD. If 0, only Expectation is plotted.

Usage Examples:
    Run with default settings (block preconditioner):
        python ex14.py

    Run with distance-localized preconditioner and correlated noise:
        python ex14.py --precond distance-localized --loc-apply-taper --noise-type correlated

    Run the full help menu to see all options:
        python ex14.py --help
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
    base_space: Sobolev,
    degree: int,
    obs_points: np.ndarray,
    prior_scale: float,
    prior_std: float,
    noise_scale: float,
    noise_std: float,
):
    """
    Helper function to generate the space, forward operator,  prior measure,
    and noise measure for a specific truncation degree.
    """
    space = base_space.with_degree(degree)
    forward_operator = space.point_evaluation_operator(obs_points)
    prior_measure = space.point_value_scaled_heat_kernel_gaussian_measure(
        prior_scale, std=prior_std
    )
    if noise_scale == 0.0:
        data_error_measure = inf.GaussianMeasure.from_standard_deviation(
            forward_operator.codomain, noise_std
        )
    else:
        spatial_noise_measure = space.point_value_scaled_heat_kernel_gaussian_measure(
            noise_scale,
            std=noise_std,
        )
        data_error_measure = spatial_noise_measure.affine_mapping(
            operator=forward_operator
        )

    return space, forward_operator, prior_measure, data_error_measure


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian inversion with stacked surrogate preconditioning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Problem & Grid Parameters ---
    grid_group = parser.add_argument_group("Problem & Grid Parameters")
    grid_group.add_argument(
        "--n-data", type=int, default=100, help="Number of observation points"
    )
    grid_group.add_argument(
        "--base-degree",
        type=int,
        default=128,
        help="Truncation degree for the 'true' high-resolution model space",
    )
    grid_group.add_argument(
        "--sobolev-order",
        type=float,
        default=2.0,
        help="Smoothness order for the Sobolev space",
    )
    grid_group.add_argument(
        "--surrogate-degree",
        type=int,
        default=32,
        help="Truncation degree for the spatial surrogate problem",
    )

    # --- Statistical Parameters ---
    stat_group = parser.add_argument_group("Statistical Parameters")
    stat_group.add_argument(
        "--prior-scale", type=float, default=0.05, help="Length scale of the prior"
    )
    stat_group.add_argument(
        "--prior-std", type=float, default=1.0, help="Standard deviation of the prior"
    )
    stat_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # --- Noise Parameters ---
    noise_group = parser.add_argument_group("Noise Parameters")

    noise_group.add_argument(
        "--noise-amplitude-factor",
        type=float,
        default=0.1,
        help="Relative amplitude of the noise",
    )
    noise_group.add_argument(
        "--noise-scale-factor",
        type=float,
        default=0.0,
        help="Relative length scale for correlated noise (zero means uncorrelated)",
    )

    # --- Preconditioner Parameters ---
    precond_group = parser.add_argument_group("Preconditioner Parameters")
    precond_group.add_argument(
        "--precond",
        type=str,
        choices=[
            "dense",
            "block",
            "banded",
            "spectral",
            "distance-localized",
            "none",
            "sparse",
            "woodbury",
        ],
        default="none",
        help="Type of preconditioner to apply",
    )
    precond_group.add_argument(
        "--bandwidth", type=int, default=10, help="Bandwidth for banded preconditioner"
    )
    precond_group.add_argument(
        "--rank", type=int, default=20, help="Rank for spectral preconditioner"
    )
    precond_group.add_argument(
        "--incomplete",
        action="store_true",
        help="Use Incomplete LU (spilu) instead of exact LU",
    )
    precond_group.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Relative distance used in forming blocks or calculating max sparse distance",
    )
    precond_group.add_argument(
        "--loc-apply-taper",
        action="store_true",
        help="Apply Gaspari-Cohn taper for the distance-localized preconditioner",
    )
    precond_group.add_argument(
        "--sparse-threshold",
        type=float,
        default=1e-3,
        help="Relative cutoff for the 'sparse' preconditioner",
    )
    precond_group.add_argument(
        "--sparse-max-nnz",
        type=int,
        default=None,
        help="Maximum number of non-zeros to retain per column for the 'sparse' preconditioner",
    )

    # --- Plotting Parameters ---
    plot_group = parser.add_argument_group("Plotting Parameters")
    plot_group.add_argument(
        "--std-samples",
        type=int,
        default=0,
        help="Number of samples to use for plotting pointwise STD. If 0, only Expectation is plotted.",
    )

    args = parser.parse_args()

    # For reproducibility
    np.random.seed(args.seed)

    # ==========================================
    # Setup Base Space & Data Grid
    # ==========================================
    print("Setting up grid and data spaces...")
    base_space = Sobolev(args.base_degree, args.sobolev_order, args.prior_scale)

    observation_points = base_space.random_points(args.n_data)
    lats = [lat for lat, _ in observation_points]
    lons = [lon for _, lon in observation_points]

    # ==========================================
    # 1. Exact Problem Components
    # ==========================================
    model_space, forward_operator, model_prior_measure, data_error_measure = (
        setup_problem_components(
            base_space,
            base_space.degree,
            observation_points,
            args.prior_scale,
            args.prior_std,
            args.noise_scale_factor * args.prior_scale,
            args.noise_amplitude_factor * args.prior_std,
        )
    )

    data_space = forward_operator.codomain

    print(f"Model space dimension: {model_space.dim}")
    print(f"Data space dimension: {data_space.dim}")

    forward_problem = inf.LinearForwardProblem(
        forward_operator, data_error_measure=data_error_measure
    )

    # Generate Synthetic Data
    true_model, data = forward_problem.synthetic_model_and_data(model_prior_measure)
    bayesian_inversion = inf.LinearBayesianInversion(
        forward_problem, model_prior_measure
    )

    # ==========================================
    # 4. Preconditioner Routing
    # ==========================================
    preconditioner = None
    if args.precond != "none":
        print(f"Initializing {args.precond} preconditioner...")

        if args.precond != "distance-localized":
            # ==========================================
            # 3. Surrogate Problem Components
            # ==========================================

            print(
                f"Building spatial surrogate problem (degree={args.surrogate_degree})..."
            )

            surrogate_space, surrogate_fwd_op, surrogate_prior, surrogate_noise = (
                setup_problem_components(
                    base_space,
                    args.surrogate_degree,
                    observation_points,
                    args.prior_scale,
                    args.prior_std,
                    0.0,
                    args.noise_amplitude_factor * args.prior_std,
                )
            )

            surrogate_inv = bayesian_inversion.surrogate_inversion(
                alternate_forward_operator=surrogate_fwd_op,
                alternate_prior_measure=surrogate_prior,
                alternate_data_error_measure=surrogate_noise,
            )

            surrogate_normal_operator = surrogate_inv.normal_operator

        if args.precond == "block":
            print("Forming blocks...")
            blocks = model_space.cluster_points(
                observation_points, threshold=args.threshold * args.prior_scale
            )
            print("Building preconditioner...")
            solver_wrapper = inf.ExactBlockPreconditioningMethod(
                blocks, incomplete=args.incomplete
            )
            preconditioner = solver_wrapper(surrogate_normal_operator)

        elif args.precond == "banded":
            solver_wrapper = inf.BandedPreconditioningMethod(
                args.bandwidth, incomplete=args.incomplete
            )
            preconditioner = solver_wrapper(surrogate_normal_operator)

        elif args.precond == "spectral":
            solver_wrapper = inf.SpectralPreconditioningMethod(
                rank=args.rank,
                method="variable",
            )
            preconditioner = solver_wrapper(surrogate_normal_operator)

        elif args.precond == "dense":
            print("Cholesky factoring surrogate normal")
            preconditioner = inf.CholeskySolver()(surrogate_normal_operator)

        elif args.precond == "sparse":
            print("Building and factoring sparse approximation")
            max_nnz = (
                args.sparse_max_nnz
                if args.sparse_max_nnz is not None
                else max(100, int(data_space.dim / 10))
            )
            preconditioner = inf.ColumnThresholdedPreconditioningMethod(
                args.sparse_threshold, max_nnz=max_nnz, incomplete=True
            )(surrogate_normal_operator)

        elif args.precond == "woodbury":
            print("Forming model-space preconditioner")
            preconditioner = surrogate_inv.woodbury_data_preconditioner()

        elif args.precond == "distance-localized":
            print("Building distance-localized sparse matrix...")
            max_distance = args.threshold * args.prior_scale
            preconditioner = model_space.distance_localized_preconditioner(
                model_prior_measure,
                observation_points,
                max_distance,
                data_error_measure=(
                    data_error_measure if args.noise_scale_factor <= 0.0 else None
                ),
                apply_taper=args.loc_apply_taper,
            )

    # ==========================================
    # 5. Bayesian Inversion & Plotting
    # ==========================================
    print("Solving linear system...")
    solver = inf.CGSolver()
    model_posterior_measure = bayesian_inversion.model_posterior_measure(
        data, solver, preconditioner=preconditioner
    )
    posterior_expectation = model_posterior_measure.expectation
    print(f"Number of CG iterations = {solver.iterations}")

    # --- Modernized Plotting Section ---

    if args.std_samples > 0:
        print(f"Sampling pointwise STD with {args.std_samples} samples...")
        posterior_std = model_posterior_measure.sample_pointwise_std(args.std_samples)

        fig, (ax1, ax2, ax3) = plt.subplots(
            1,
            3,
            figsize=(24, 7),
            subplot_kw={"projection": ccrs.Robinson()},
            layout="constrained",
        )
    else:
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(16, 7),
            subplot_kw={"projection": ccrs.Robinson()},
            layout="constrained",
        )

    _, im1 = plot(
        true_model,
        ax=ax1,
        coasts=True,
        cmap="seismic",
        symmetric=True,
        colorbar=True,
        colorbar_kwargs={
            "label": "True model",
            "orientation": "horizontal",
            "shrink": 0.8,
        },
    )

    vmin1, vmax1 = im1.get_clim()
    ax1.scatter(
        lons,
        lats,
        c=data,
        cmap="seismic",
        vmin=vmin1,
        vmax=vmax1,
        edgecolors="black",
        linewidths=0.5,
        s=8,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    ax1.set_title("True model", fontsize=14, fontweight="bold")

    _, im2 = plot(
        posterior_expectation,
        ax=ax2,
        coasts=True,
        cmap="seismic",
        symmetric=True,
        colorbar=True,
        colorbar_kwargs={
            "label": "Posterior expectation",
            "orientation": "horizontal",
            "shrink": 0.8,
        },
    )

    vmin2, vmax2 = im2.get_clim()
    ax2.scatter(
        lons,
        lats,
        c=data,
        cmap="seismic",
        vmin=vmin2,
        vmax=vmax2,
        edgecolors="black",
        linewidths=0.5,
        s=8,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    ax2.set_title("Posterior Expectation", fontsize=14, fontweight="bold")

    if args.std_samples > 0:
        _, im3 = plot(
            posterior_std,
            ax=ax3,
            coasts=True,
            cmap="viridis",
            colorbar=True,
            colorbar_kwargs={
                "label": "Standard Deviation",
                "orientation": "horizontal",
                "shrink": 0.8,
            },
        )

        ax3.scatter(
            lons,
            lats,
            color="white",
            edgecolors="black",
            marker="o",
            linewidths=0.5,
            s=8,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        ax3.set_title(
            f"Posterior Pointwise STD (N={args.std_samples})",
            fontsize=14,
            fontweight="bold",
        )

    plt.show()


if __name__ == "__main__":
    main()
