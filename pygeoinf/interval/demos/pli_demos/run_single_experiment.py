#!/usr/bin/env python
"""
Example: Run a single PLI experiment.

This script shows the simplest way to run a single PLI experiment
with a pre-defined configuration.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from pygeoinf.interval.demos.pli_demos import PLIConfig, PLIExperiment


def main():
    """Run a single PLI experiment."""

    # Choose a configuration
    # Options: get_fast_config(), get_standard_config(), get_high_resolution_config()
    config = PLIConfig.get_fast_config()

    # Customize output directory
    config.output_dir = str(Path(__file__).parent / "single_run_output")
    config.name = "example_run"

    # Optionally override parameters
    config.noise_level = 0.02
    config.K = 30
    config.seed = 42

    print("=" * 60)
    print("PLI EXPERIMENT: Single Run Example")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model space dimension (N): {config.N}")
    print(f"  Data points (N_d): {config.N_d}")
    print(f"  Property points (N_p): {config.N_p}")
    print(f"  KL modes (K): {config.K}")
    print(f"  Prior type: {config.prior_type}")
    print(f"  Smoothness (s): {config.s}")
    print(f"  Noise level: {config.noise_level}")
    print(f"\nOutput directory: {config.output_dir}")
    print("=" * 60)

    # Create and run experiment
    experiment = PLIExperiment(config)
    results = experiment.run()

    # Print results summary
    metrics = results.get('metrics', {})
    timings = results.get('timings', {})

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nAccuracy:")
    err = metrics.get('model_rms_error', 'N/A')
    print(f"  Model RMS error: {err:.6f}" if err != 'N/A' else f"  Model RMS error: {err}")
    err = metrics.get('property_rms_error', 'N/A')
    print(f"  Property RMS error: {err:.6f}" if err != 'N/A' else f"  Property RMS error: {err}")

    print(f"\nUncertainty Calibration:")
    n = metrics.get('properties_within_2sigma', 'N/A')
    pct = metrics.get('properties_within_2sigma_pct', 'N/A')
    print(f"  Properties within ±2σ: {n}/{config.N_p} ({pct:.1f}%)" if pct != 'N/A' else f"  Properties within ±2σ: {n}")

    print(f"\nInference Improvement:")
    pct = metrics.get('data_fit_improvement_pct', 'N/A')
    print(f"  Data fit improvement: {pct:.1f}%" if pct != 'N/A' else f"  Data fit improvement: {pct}")
    pct = metrics.get('uncertainty_reduction_pct', 'N/A')
    print(f"  Uncertainty reduction: {pct:.1f}%" if pct != 'N/A' else f"  Uncertainty reduction: {pct}")

    print(f"\nTiming:")
    t = timings.get('total', 'N/A')
    print(f"  Total time: {t:.2f}s" if t != 'N/A' else f"  Total time: {t}")

    print("\n" + "=" * 60)
    print(f"Figures saved to: {config.output_dir}/figures/")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
