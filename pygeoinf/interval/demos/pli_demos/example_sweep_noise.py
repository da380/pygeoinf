#!/usr/bin/env python
"""
Example: Noise level parameter sweep.

This script demonstrates how to run a parameter sweep over the
noise level in the data. This helps understand how inference
accuracy degrades with increasing measurement noise.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from pygeoinf.interval.demos.pli_demos import (
    PLIConfig,
    PLISweep,
    plot_convergence_study,
    plot_accuracy_vs_time
)


def main():
    """Run noise level parameter sweep."""

    # Create base configuration
    base_config = PLIConfig(
        # Model space
        N=100,
        N_d=50,
        N_p=20,
        K=30,

        # Prior
        s=2.0,
        length_scale=0.1,
        overall_variance=1.0,
        prior_type='bessel_sobolev',

        # Integration
        integration_order=32,
        polynomial_degree=4,

        # Outputs
        compute_model_posterior=False,
        save_figures=True,

        # Random seed for reproducibility
        seed=42
    )

    # Output directory
    output_dir = Path(__file__).parent / "sweep_noise_results"

    # Create sweep
    sweep = PLISweep(
        base_config=base_config,
        output_dir=output_dir,
        verbose=True
    )

    # Define noise levels to sweep (logarithmically spaced)
    noise_levels = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    print(f"\nRunning noise sweep with {len(noise_levels)} values...")
    print(f"Noise levels: {noise_levels}")
    print(f"Output directory: {output_dir}\n")

    # Run the sweep
    results = sweep.sweep_parameter('noise_level', noise_levels)

    # Print summary
    sweep.print_summary()

    # Find best noise level (minimum achievable error)
    best_run = sweep.get_best_run('property_rms_error', minimize=True)
    print(f"\nBest noise level: {best_run.get('noise_level', 'N/A')}")
    err = best_run.get('property_rms_error', 'N/A')
    print(f"  Property RMS error: {err:.6f}" if err != 'N/A' else err)

    # Create convergence plot
    print("\nCreating analysis plots...")

    successful = results[results['status'] == 'success']

    plot_convergence_study(
        successful,
        x_param='noise_level',
        y_metrics=['property_rms_error', 'model_rms_error'],
        output_path=output_dir / 'noise_convergence.png',
        log_scale=True
    )

    # Create accuracy vs time plot
    plot_accuracy_vs_time(
        successful,
        accuracy_metric='property_rms_error',
        time_metric='inference_total',
        label_param='noise_level',
        output_path=output_dir / 'accuracy_vs_time.png'
    )

    print(f"Plots saved to {output_dir}")

    return results


if __name__ == "__main__":
    main()
