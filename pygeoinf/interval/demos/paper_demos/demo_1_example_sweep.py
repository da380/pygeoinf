#!/usr/bin/env python
"""
Example parameter sweeps for Demo 1.

This script demonstrates how to use the Demo 1 parameter sweep framework
to run various types of experiments. Users can uncomment and run any of
the example sweeps below.
"""

from pathlib import Path
from demo_1_config import (
    Demo1Config,
    get_fast_config,
    get_standard_config,
    get_high_resolution_config
)
from demo_1_sweep import (
    Demo1Sweep,
    create_prior_sensitivity_sweep,
    create_resolution_sweep,
    create_noise_sensitivity_sweep,
    create_kl_truncation_sweep,
    create_component_coupling_sweep,
    create_comprehensive_sweep,
    create_adaptive_resolution_sweep
)
from demo_1_analysis import generate_sweep_report
from demo_1_experiment import run_single_experiment


def example_1_single_experiment():
    """Example 1: Run a single experiment with standard configuration."""
    print("="*60)
    print("Example 1: Single Experiment")
    print("="*60)

    config = get_standard_config()
    output_dir = Path("./demo_1_experiments/single_run")

    results = run_single_experiment(config, output_dir)

    print("\nResults:")
    print(f"  vp relative L2 error: {results['metrics']['vp_rel_l2_error']:.6f}")
    print(f"  vs relative L2 error: {results['metrics']['vs_rel_l2_error']:.6f}")
    print(f"  rho relative L2 error: {results['metrics']['rho_rel_l2_error']:.6f}")
    print(f"  Total time: {results['timings']['total']:.2f}s")


def example_2_prior_sensitivity():
    """Example 2: Sweep over prior smoothness parameters for vp component."""
    print("="*60)
    print("Example 2: Prior Sensitivity (vp component)")
    print("="*60)

    output_dir = Path("./demo_1_experiments")

    sweep = create_prior_sensitivity_sweep(
        output_dir=output_dir,
        s_values=[1.5, 2.0, 2.5, 3.0],
        length_scale_values=[0.1, 0.2, 0.3, 0.4],
        component='vp'
    )

    # Run sweep
    df = sweep.run(progress_bar=True)

    # Generate analysis report
    generate_sweep_report(sweep.sweep_dir)

    # Print summary
    print("\nBest configuration:")
    best_idx = df['vp_rel_l2_error'].idxmin()
    print(f"  s_vp: {df.loc[best_idx, 's_vp']}")
    print(f"  length_scale_vp: {df.loc[best_idx, 'length_scale_vp']}")
    print(f"  vp error: {df.loc[best_idx, 'vp_rel_l2_error']:.6f}")


def example_3_resolution_study():
    """Example 3: Study convergence with increasing resolution."""
    print("="*60)
    print("Example 3: Resolution Study")
    print("="*60)

    output_dir = Path("./demo_1_experiments")

    sweep = create_resolution_sweep(
        output_dir=output_dir,
        N_values=[50, 100, 150, 200],
        N_d_values=[25, 50, 75, 100]
    )

    # Run sweep
    df = sweep.run(progress_bar=True)

    # Generate analysis
    generate_sweep_report(sweep.sweep_dir)

    # Print convergence rates
    print("\nConvergence analysis:")
    for N in df['N'].unique():
        subset = df[df['N'] == N]
        print(f"  N={N}:")
        print(f"    Mean vp error: {subset['vp_rel_l2_error'].mean():.6f}")
        print(f"    Mean vs error: {subset['vs_rel_l2_error'].mean():.6f}")
        print(f"    Mean rho error: {subset['rho_rel_l2_error'].mean():.6f}")


def example_4_noise_sensitivity():
    """Example 4: Study robustness to observation noise."""
    print("="*60)
    print("Example 4: Noise Sensitivity")
    print("="*60)

    output_dir = Path("./demo_1_experiments")

    sweep = create_noise_sensitivity_sweep(
        output_dir=output_dir,
        noise_values=[0.001, 0.005, 0.01, 0.05, 0.1]
    )

    # Run sweep
    df = sweep.run(progress_bar=True)

    # Generate analysis
    generate_sweep_report(sweep.sweep_dir)

    # Print results
    print("\nNoise sensitivity:")
    for noise in df['noise_level'].unique():
        subset = df[df['noise_level'] == noise]
        print(f"  Noise level {noise}:")
        print(f"    vp error: {subset['vp_rel_l2_error'].mean():.6f}")
        print(f"    Data fit error: {subset['data_rel_l2_error'].mean():.6f}")


def example_5_kl_truncation():
    """Example 5: Study effect of KL truncation on computational efficiency."""
    print("="*60)
    print("Example 5: KL Truncation Study")
    print("="*60)

    output_dir = Path("./demo_1_experiments")

    sweep = create_kl_truncation_sweep(
        output_dir=output_dir,
        kl_values=[10, 20, 50, 100, None]
    )

    # Run sweep
    df = sweep.run(progress_bar=True)

    # Generate analysis
    generate_sweep_report(sweep.sweep_dir)

    # Print timing comparison
    print("\nKL truncation effects:")
    for kl in df['kl_truncation_vp'].unique():
        subset = df[df['kl_truncation_vp'] == kl]
        kl_str = "None" if pd.isna(kl) else str(int(kl))
        print(f"  KL={kl_str}:")
        print(f"    Mean vp error: {subset['vp_rel_l2_error'].mean():.6f}")
        print(f"    Mean time: {subset['time_total'].mean():.2f}s")


def example_6_component_coupling():
    """Example 6: Explore coupling between components with different smoothness."""
    print("="*60)
    print("Example 6: Component Coupling Study")
    print("="*60)

    output_dir = Path("./demo_1_experiments")

    sweep = create_component_coupling_sweep(
        output_dir=output_dir,
        smoothness_values=[1.5, 2.0, 2.5]
    )

    # Run sweep
    df = sweep.run(progress_bar=True)

    # Generate analysis
    generate_sweep_report(sweep.sweep_dir)

    # Find optimal smoothness combination
    print("\nBest smoothness combination:")
    best_idx = df['vp_rel_l2_error'].idxmin()
    print(f"  s_vp: {df.loc[best_idx, 's_vp']}")
    print(f"  s_vs: {df.loc[best_idx, 's_vs']}")
    print(f"  s_rho: {df.loc[best_idx, 's_rho']}")
    print(f"  vp error: {df.loc[best_idx, 'vp_rel_l2_error']:.6f}")


def example_7_custom_sweep():
    """Example 7: Create a custom parameter sweep."""
    print("="*60)
    print("Example 7: Custom Parameter Sweep")
    print("="*60)

    # Define base configuration
    base_config = get_standard_config()

    # Define parameters to sweep
    sweep_params = {
        's_vp': [1.5, 2.0, 2.5],
        's_vs': [1.5, 2.0, 2.5],
        'noise_level': [0.01, 0.05]
    }

    # Create sweep
    output_dir = Path("./demo_1_experiments")
    sweep = Demo1Sweep(
        base_config=base_config,
        sweep_params=sweep_params,
        output_dir=output_dir,
        name="custom_sweep",
        parallel=True
    )

    # Run sweep
    df = sweep.run(progress_bar=True)

    # Generate analysis
    generate_sweep_report(sweep.sweep_dir)

    print(f"\nCompleted {len(df)} runs")
    print(f"Results saved to: {sweep.sweep_dir}")


def example_8_adaptive_resolution():
    """Example 8: Adaptive resolution sweep with coupled parameters."""
    print("="*60)
    print("Example 8: Adaptive Resolution Sweep")
    print("="*60)

    output_dir = Path("./demo_1_experiments")

    sweep = create_adaptive_resolution_sweep(
        output_dir=output_dir,
        N_values=[50, 100, 150, 200]
    )

    # Run sweep
    df = sweep.run(progress_bar=True)

    # Generate analysis
    generate_sweep_report(sweep.sweep_dir)

    # Print scaling relationships
    print("\nAdaptive resolution results:")
    for N in df['N'].unique():
        subset = df[df['N'] == N]
        N_d = subset['N_d'].iloc[0]
        N_p = subset['N_p'].iloc[0]
        print(f"  N={N}, N_d={N_d}, N_p={N_p}:")
        print(f"    vp error: {subset['vp_rel_l2_error'].mean():.6f}")
        print(f"    Time: {subset['time_total'].mean():.2f}s")


def example_9_fast_prototype():
    """Example 9: Fast prototype sweep for quick testing."""
    print("="*60)
    print("Example 9: Fast Prototype Sweep")
    print("="*60)

    # Use fast configuration for quick testing
    base_config = get_fast_config()

    # Small parameter sweep
    sweep_params = {
        's_vp': [1.5, 2.0],
        'noise_level': [0.01, 0.05]
    }

    output_dir = Path("./demo_1_experiments")
    sweep = Demo1Sweep(
        base_config=base_config,
        sweep_params=sweep_params,
        output_dir=output_dir,
        name="fast_prototype",
        parallel=False  # Sequential for quick testing
    )

    # Run sweep
    df = sweep.run(progress_bar=True)

    print("\nFast prototype complete!")
    print(f"Total runs: {len(df)}")
    print(f"Total time: {df['time_total'].sum():.2f}s")


def example_10_comprehensive():
    """Example 10: Comprehensive sweep (WARNING: many runs!)."""
    print("="*60)
    print("Example 10: Comprehensive Sweep")
    print("WARNING: This will run many experiments and take significant time!")
    print("="*60)

    response = input("Are you sure you want to proceed? (yes/no): ")
    if response.lower() != 'yes':
        print("Sweep cancelled.")
        return

    output_dir = Path("./demo_1_experiments")

    sweep = create_comprehensive_sweep(
        output_dir=output_dir,
        N_values=[50, 100],
        s_values=[1.5, 2.0, 2.5],
        noise_values=[0.01, 0.05]
    )

    print(f"\nThis will run {len(sweep.param_combinations)} experiments!")

    # Run sweep
    df = sweep.run(progress_bar=True)

    # Generate comprehensive analysis
    generate_sweep_report(sweep.sweep_dir)

    print("\nComprehensive sweep complete!")
    print(f"Total experiments: {len(df)}")
    print(f"Total time: {df['time_total'].sum():.2f}s")


if __name__ == "__main__":
    import sys
    import pandas as pd

    print("\n" + "="*60)
    print("Demo 1 Parameter Sweep Examples")
    print("="*60)
    print("\nAvailable examples:")
    print("  1. Single experiment (standard config)")
    print("  2. Prior sensitivity sweep (vp component)")
    print("  3. Resolution convergence study")
    print("  4. Noise sensitivity analysis")
    print("  5. KL truncation efficiency study")
    print("  6. Component coupling exploration")
    print("  7. Custom parameter sweep")
    print("  8. Adaptive resolution sweep")
    print("  9. Fast prototype sweep (quick test)")
    print(" 10. Comprehensive sweep (WARNING: many runs!)")
    print()

    if len(sys.argv) > 1:
        # Run specified example
        example_num = int(sys.argv[1])
        examples = {
            1: example_1_single_experiment,
            2: example_2_prior_sensitivity,
            3: example_3_resolution_study,
            4: example_4_noise_sensitivity,
            5: example_5_kl_truncation,
            6: example_6_component_coupling,
            7: example_7_custom_sweep,
            8: example_8_adaptive_resolution,
            9: example_9_fast_prototype,
            10: example_10_comprehensive
        }

        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Invalid example number: {example_num}")
    else:
        # Interactive mode
        print("Usage:")
        print("  python demo_1_example_sweep.py <example_number>")
        print("\nOr uncomment one of the examples below and run directly:")
        print()

        # Uncomment one of the following to run:

        # example_1_single_experiment()
        # example_2_prior_sensitivity()
        # example_3_resolution_study()
        # example_4_noise_sensitivity()
        # example_5_kl_truncation()
        # example_6_component_coupling()
        # example_7_custom_sweep()
        # example_8_adaptive_resolution()
        example_9_fast_prototype()  # Fast example enabled by default
        # example_10_comprehensive()
