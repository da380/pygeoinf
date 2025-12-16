#!/usr/bin/env python
"""
Example: K (KL truncation) parameter sweep.

This script demonstrates how to run a parameter sweep over the
KL truncation level K, which controls the number of modes used
in the Karhunen-Loève expansion for representing the prior.

Higher K values capture more prior variability but increase
computational cost. This sweep helps find the optimal trade-off.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from pygeoinf.interval.demos.pli_demos import (
    PLIConfig,
    PLISweep,
    plot_convergence_study
)


def main():
    """Run K parameter sweep."""

    # Create base configuration
    base_config = PLIConfig.get_standard_config()

    # Override some parameters for faster testing
    base_config = PLIConfig(
        # Model space
        N=100,
        N_d=50,
        N_p=20,

        # Prior
        s=2.0,
        length_scale=0.1,
        overall_variance=1.0,
        prior_type='bessel_sobolev',

        # Noise
        noise_level=0.02,

        # Integration (fast settings)
        integration_order=32,
        polynomial_degree=4,

        # Outputs
        compute_model_posterior=False,  # Faster
        save_figures=True,

        # Random seed for reproducibility
        seed=42
    )

    # Output directory
    output_dir = Path(__file__).parent / "sweep_K_results"

    # Create sweep
    sweep = PLISweep(
        base_config=base_config,
        output_dir=output_dir,
        verbose=True
    )

    # Define K values to sweep
    K_values = [5, 10, 15, 20, 30, 40, 50, 60]

    print(f"\nRunning K sweep with {len(K_values)} values...")
    print(f"K values: {K_values}")
    print(f"Output directory: {output_dir}\n")

    # Run the sweep
    results = sweep.sweep_parameter('K', K_values)

    # Print summary
    sweep.print_summary()

    # Find best K
    best_run = sweep.get_best_run('property_rms_error', minimize=True)
    print(f"\nBest K value: {best_run.get('K', 'N/A')}")
    print(f"  Property RMS error: {best_run.get('property_rms_error', 'N/A'):.6f}")
    print(f"  Model RMS error: {best_run.get('model_rms_error', 'N/A'):.6f}")
    print(f"  Time: {best_run.get('total', 'N/A'):.2f}s")

    # Create convergence plot
    print("\nCreating convergence plots...")

    fig = plot_convergence_study(
        results[results['status'] == 'success'],
        x_param='K',
        y_metrics=['property_rms_error', 'model_rms_error', 'total'],
        output_path=output_dir / 'K_convergence.png'
    )

    print(f"Convergence plot saved to {output_dir / 'K_convergence.png'}")

    return results


if __name__ == "__main__":
    main()
