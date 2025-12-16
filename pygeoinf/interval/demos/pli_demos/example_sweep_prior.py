#!/usr/bin/env python
"""
Example: Prior parameter sweep (smoothness and length scale).

This script demonstrates how to run a 2D parameter sweep over
prior parameters: smoothness (s) and length scale. This helps
understand the sensitivity of inference to prior assumptions.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from pygeoinf.interval.demos.pli_demos import (
    PLIConfig,
    PLISweep,
    plot_parameter_comparison
)


def main():
    """Run prior parameter sweep."""

    # Create base configuration
    base_config = PLIConfig(
        # Model space
        N=100,
        N_d=50,
        N_p=20,
        K=30,

        # Prior (will be overridden)
        s=2.0,
        length_scale=0.1,
        overall_variance=1.0,
        prior_type='bessel_sobolev',

        # Noise
        noise_level=0.02,

        # Integration
        integration_order=32,
        polynomial_degree=4,

        # Outputs
        compute_model_posterior=False,
        save_figures=False,  # Disable per-run figures

        # Random seed for reproducibility
        seed=42
    )

    # Output directory
    output_dir = Path(__file__).parent / "sweep_prior_results"

    # Create sweep
    sweep = PLISweep(
        base_config=base_config,
        output_dir=output_dir,
        verbose=True
    )

    # Define parameter grid
    param_grid = {
        's': [1.0, 1.5, 2.0, 2.5, 3.0],
        'length_scale': [0.05, 0.1, 0.15, 0.2, 0.3]
    }

    n_combos = len(param_grid['s']) * len(param_grid['length_scale'])
    print(f"\nRunning prior grid sweep with {n_combos} combinations...")
    print(f"Smoothness values (s): {param_grid['s']}")
    print(f"Length scale values: {param_grid['length_scale']}")
    print(f"Output directory: {output_dir}\n")

    # Run the grid sweep
    results = sweep.sweep_grid(param_grid)

    # Print summary
    sweep.print_summary()

    # Create 2D comparison plots
    print("\nCreating analysis plots...")

    successful = results[results['status'] == 'success']

    # Property error plot
    plot_parameter_comparison(
        successful,
        x_param='s',
        y_param='length_scale',
        color_metric='property_rms_error',
        output_path=output_dir / 'prior_property_error.png'
    )

    # Model error plot
    plot_parameter_comparison(
        successful,
        x_param='s',
        y_param='length_scale',
        color_metric='model_rms_error',
        output_path=output_dir / 'prior_model_error.png'
    )

    # Uncertainty calibration plot
    plot_parameter_comparison(
        successful,
        x_param='s',
        y_param='length_scale',
        color_metric='properties_within_2sigma_pct',
        output_path=output_dir / 'prior_calibration.png'
    )

    print(f"Plots saved to {output_dir}")

    # Find best combination
    best = sweep.get_best_run('property_rms_error', minimize=True)
    print(f"\nBest prior parameters:")
    print(f"  s = {best.get('s', 'N/A')}")
    print(f"  length_scale = {best.get('length_scale', 'N/A')}")
    err = best.get('property_rms_error', 'N/A')
    print(f"  Property RMS error: {err:.6f}" if err != 'N/A' else err)

    return results


if __name__ == "__main__":
    main()
