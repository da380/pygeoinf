#!/usr/bin/env python3
"""
Quick example script for running PLI parameter sweeps.

Usage:
    python example_sweep.py
"""

from run_pli_experiments import PLISweep
import matplotlib.pyplot as plt
import seaborn as sns

# Example 1: Sweep over K values to find optimal KL truncation
def sweep_kl_truncation():
    """Test different KL expansion sizes."""
    print("="*80)
    print("EXAMPLE 1: KL Expansion Size Sweep")
    print("="*80)

    sweep = PLISweep()

    base_config = {
        'N': 100,
        'N_d': 50,
        'N_p': 20,
        'basis': 'sine',
        'alpha': 0.1,
        'noise_level': 0.1,
        'compute_model_posterior': False,
        'random_seed': 42
    }

    param_grid = {
        'K': [20, 30, 50, 100, 150, 200]
    }

    df = sweep.run(param_grid, base_config)

    # Analyze results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    df_success = df[df['status'] == 'success']
    print(df_success[['K', 'inference_total', 'property_rms_error',
                      'properties_within_2sigma_pct']].to_string(index=False))

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Timing
    axes[0, 0].plot(df_success['K'], df_success['inference_total'], 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('K (KL modes)', fontsize=12)
    axes[0, 0].set_ylabel('Inference time (s)', fontsize=12)
    axes[0, 0].set_title('Computational Cost', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(df_success['K'], df_success['property_rms_error'], 'o-',
                    color='tab:orange', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('K (KL modes)', fontsize=12)
    axes[0, 1].set_ylabel('Property RMS error', fontsize=12)
    axes[0, 1].set_title('Reconstruction Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Uncertainty calibration
    axes[1, 0].plot(df_success['K'], df_success['properties_within_2sigma_pct'], 'o-',
                    color='tab:green', linewidth=2, markersize=8)
    axes[1, 0].axhline(95, color='red', linestyle='--', label='Expected (95%)')
    axes[1, 0].set_xlabel('K (KL modes)', fontsize=12)
    axes[1, 0].set_ylabel('% within 2σ', fontsize=12)
    axes[1, 0].set_title('Uncertainty Quantification', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Data fit
    axes[1, 1].plot(df_success['K'], df_success['data_fit_improvement_pct'], 'o-',
                    color='tab:purple', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('K (KL modes)', fontsize=12)
    axes[1, 1].set_ylabel('Data fit improvement (%)', fontsize=12)
    axes[1, 1].set_title('Data Fit Improvement', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(sweep.sweep_dir / "analysis_K_sweep.png", dpi=300, bbox_inches='tight')
    print(f"\nAnalysis plot saved to: {sweep.sweep_dir / 'analysis_K_sweep.png'}")

    return sweep.sweep_dir


# Example 2: Compare different model resolutions
def sweep_resolution():
    """Test different discretization resolutions."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Resolution Comparison")
    print("="*80)

    sweep = PLISweep()

    base_config = {
        'K': 100,
        'N_p': 20,
        'basis': 'sine',
        'alpha': 0.1,
        'noise_level': 0.1,
        'compute_model_posterior': False,
        'random_seed': 42
    }

    param_grid = {
        'N': [50, 100, 200],
        'N_d': [25, 50, 100]
    }

    df = sweep.run(param_grid, base_config)

    # Analyze results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    df_success = df[df['status'] == 'success']
    print(df_success[['N', 'N_d', 'inference_total', 'property_rms_error']].to_string(index=False))

    # Create heatmap
    pivot_time = df_success.pivot(index='N', columns='N_d', values='inference_total')
    pivot_error = df_success.pivot(index='N', columns='N_d', values='property_rms_error')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(pivot_time, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Time (s)'})
    axes[0].set_title('Inference Time (seconds)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('N_d (data points)', fontsize=12)
    axes[0].set_ylabel('N (model dimension)', fontsize=12)

    sns.heatmap(pivot_error, annot=True, fmt='.4f', cmap='YlGnBu_r', ax=axes[1], cbar_kws={'label': 'RMS error'})
    axes[1].set_title('Property RMS Error', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('N_d (data points)', fontsize=12)
    axes[1].set_ylabel('N (model dimension)', fontsize=12)

    plt.tight_layout()
    plt.savefig(sweep.sweep_dir / "analysis_resolution.png", dpi=300, bbox_inches='tight')
    print(f"\nAnalysis plot saved to: {sweep.sweep_dir / 'analysis_resolution.png'}")

    return sweep.sweep_dir


# Example 3: Noise sensitivity analysis
def sweep_noise_sensitivity():
    """Test sensitivity to different noise levels."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Noise Sensitivity Analysis")
    print("="*80)

    sweep = PLISweep()

    base_config = {
        'N': 100,
        'N_d': 50,
        'N_p': 20,
        'basis': 'sine',
        'alpha': 0.1,
        'compute_model_posterior': False,
        'random_seed': 42
    }

    param_grid = {
        'noise_level': [0.01, 0.05, 0.1, 0.2, 0.5],
        'K': [50, 100, 200]
    }

    df = sweep.run(param_grid, base_config)

    # Analyze results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    df_success = df[df['status'] == 'success']
    print(df_success[['noise_level', 'K', 'snr', 'property_rms_error']].to_string(index=False))

    # Plot noise vs accuracy for different K
    plt.figure(figsize=(10, 6))
    for k in [50, 100, 200]:
        subset = df_success[df_success['K'] == k]
        plt.plot(subset['noise_level'], subset['property_rms_error'],
                'o-', label=f'K={k}', linewidth=2, markersize=8)

    plt.xlabel('Noise Level (relative to signal)', fontsize=12)
    plt.ylabel('Property RMS Error', fontsize=12)
    plt.title('Noise Sensitivity Analysis', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(sweep.sweep_dir / "analysis_noise_sensitivity.png", dpi=300, bbox_inches='tight')
    print(f"\nAnalysis plot saved to: {sweep.sweep_dir / 'analysis_noise_sensitivity.png'}")

    return sweep.sweep_dir


if __name__ == "__main__":
    import sys

    # Set plotting style
    sns.set_theme(style="whitegrid", palette="muted")

    print("\n" + "="*80)
    print("PLI PARAMETER SWEEP EXAMPLES")
    print("="*80)
    print("\nChoose an example to run:")
    print("  1. KL expansion size sweep (recommended for first run)")
    print("  2. Resolution comparison")
    print("  3. Noise sensitivity analysis")
    print("  4. Run all examples")
    print("\nOr run with argument: python example_sweep.py [1-4]")
    print("="*80)

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1-4): ").strip()

    if choice == '1':
        sweep_dir = sweep_kl_truncation()
        print(f"\n✓ Results saved to: {sweep_dir}")

    elif choice == '2':
        sweep_dir = sweep_resolution()
        print(f"\n✓ Results saved to: {sweep_dir}")

    elif choice == '3':
        sweep_dir = sweep_noise_sensitivity()
        print(f"\n✓ Results saved to: {sweep_dir}")

    elif choice == '4':
        print("\nRunning all examples...")
        dir1 = sweep_kl_truncation()
        dir2 = sweep_resolution()
        dir3 = sweep_noise_sensitivity()
        print(f"\n✓ All results saved!")
        print(f"  - KL sweep: {dir1}")
        print(f"  - Resolution: {dir2}")
        print(f"  - Noise sensitivity: {dir3}")

    else:
        print(f"\nInvalid choice: {choice}")
        sys.exit(1)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
