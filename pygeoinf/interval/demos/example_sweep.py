#!/usr/bin/env python3
"""
Quick example script for running PLI parameter sweeps.

Usage:
    python example_sweep.py
"""

from run_pli_experiments import PLISweep
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example 1: Sweep over K values to find optimal KL truncation
def sweep_kl_truncation():
    """Test different KL expansion sizes."""
    print("="*80)
    print("EXAMPLE 1: KL Expansion Size Sweep")
    print("="*80)

    sweep = PLISweep(sweep_name="kl_truncation")

    base_config = {
        'N': 100,
        'N_d': 50,
        'N_p': 20,
        'basis': 'sine',
        'alpha': 0.1,
        'noise_level': 0.1,
        'compute_model_posterior': False,
        'random_seed': 42,
        'n_jobs': 16,
        'bc_config': {'bc_type': 'dirichlet', 'left': 0, 'right': 0}
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
    print("="*80)
    print("EXAMPLE 2: Resolution Comparison")
    print("="*80)

    sweep = PLISweep(sweep_name="resolution_comparison")

    base_config = {
        'K': 100,
        'N_p': 20,
        'basis': 'sine',
        'alpha': 0.1,
        'noise_level': 0.1,
        'compute_model_posterior': False,
        'random_seed': 42,
        'n_jobs': 30,
        'bc_config': {'bc_type': 'dirichlet', 'left': 0, 'right': 0}
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

    sweep = PLISweep(sweep_name="noise_sensitivity")

    base_config = {
        'N': 100,
        'N_d': 50,
        'N_p': 20,
        'basis': 'sine',
        'alpha': 0.1,
        'compute_model_posterior': False,
        'random_seed': 42,
        'n_jobs': 30,
        'bc_config': {'bc_type': 'dirichlet', 'left': 0, 'right': 0}
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


# Example 4: Boundary condition comparison
def sweep_boundary_conditions():
    """
    Test different boundary conditions for the prior covariance.

    This example demonstrates how different boundary conditions affect:
    - Prior distribution smoothness and behavior at boundaries
    - Posterior model reconstruction
    - Property inference accuracy
    - Uncertainty quantification

    Boundary conditions tested:
    1. Dirichlet (0,0): Fixed zero at both boundaries
    2. Dirichlet (0,1): Fixed zero at left, one at right
    3. Neumann (0,0): Zero derivative (periodic-like)
    4. Neumann (1,-1): Positive slope at left, negative at right
    5. Robin (1,1): Mixed value/derivative constraint

    The prior covariance operator is InverseLaplacian, so BCs control
    the smoothness properties and boundary behavior of prior samples.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Boundary Condition Comparison")
    print("="*80)

    sweep = PLISweep(sweep_name="boundary_conditions")

    base_config = {
        'N': 100,
        'N_d': 50,
        'N_p': 20,
        'K': 100,
        'basis': None,
        'alpha': 0.1,
        'noise_level': 0.1,
        'compute_model_posterior': False,
        'random_seed': 42,
        'n_jobs': 30,
        'method': 'spectral',
        'dofs': 100
    }

    # Define different boundary condition configurations
    # bc_config format: {'bc_type': type, 'left': value, 'right': value}
    bc_configs = [
        # Dirichlet (fixed value at boundaries)
        {'bc_type': 'dirichlet', 'left': 0, 'right': 0},
        # Neumann (fixed derivative at boundaries)
        {'bc_type': 'neumann', 'left': 0, 'right': 0},
        # Mixed Dirichlet
        {'bc_type': 'mixed_dirichlet_neumann', 'left': 0, 'right': 0},
        # MMixed Neumann
        {'bc_type': 'mixed_neumann_dirichlet', 'left': 0, 'right': 0}
    ]

    # Create readable names for each BC configuration
    bc_names = [
        'Dirichlet_0_0',
        'Neumann_0_0',
        'Mixed_Dirichlet_Neumann_0_0',
        'Mixed_Neumann_Dirichlet_0_0',
    ]

    # Run experiments with different boundary conditions
    results = []
    for bc_config, bc_name in zip(bc_configs, bc_names):
        config = base_config.copy()
        config['bc_config'] = bc_config
        config['bc_name'] = bc_name  # For tracking

        # Create run directory
        run_dir = sweep.sweep_dir / f"bc_{bc_name}"

        try:
            from run_pli_experiments import PLIExperiment
            experiment = PLIExperiment(config, run_dir)
            metrics = experiment.run()

            result = config.copy()
            result.update(metrics)
            result.update(experiment.timings)
            result['run_name'] = f"bc_{bc_name}"
            result['status'] = 'success'
            results.append(result)

        except Exception as e:
            print(f"\n❌ ERROR in {bc_name}: {e}\n")
            result = config.copy()
            result['run_name'] = f"bc_{bc_name}"
            result['status'] = 'failed'
            result['error'] = str(e)
            results.append(result)

    df = pd.DataFrame(results)
    df_success = df[df['status'] == 'success']

    # Analyze results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df_success[['bc_name', 'inference_total', 'property_rms_error',
                      'properties_within_2sigma_pct']].to_string(index=False))

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Timing comparison
    axes[0, 0].barh(df_success['bc_name'], df_success['inference_total'])
    axes[0, 0].set_xlabel('Inference time (s)', fontsize=12)
    axes[0, 0].set_ylabel('Boundary Condition', fontsize=12)
    axes[0, 0].set_title('Computational Cost', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # 2. Accuracy comparison
    axes[0, 1].barh(df_success['bc_name'], df_success['property_rms_error'],
                    color='tab:orange')
    axes[0, 1].set_xlabel('Property RMS error', fontsize=12)
    axes[0, 1].set_ylabel('Boundary Condition', fontsize=12)
    axes[0, 1].set_title('Reconstruction Accuracy',
                         fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # 3. Uncertainty calibration
    axes[1, 0].barh(df_success['bc_name'],
                    df_success['properties_within_2sigma_pct'],
                    color='tab:green')
    axes[1, 0].axvline(95, color='red', linestyle='--',
                       linewidth=2, label='Expected (95%)')
    axes[1, 0].set_xlabel('% within 2σ', fontsize=12)
    axes[1, 0].set_ylabel('Boundary Condition', fontsize=12)
    axes[1, 0].set_title('Uncertainty Quantification',
                         fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # 4. Model error vs property error
    axes[1, 1].scatter(df_success['model_rms_error'],
                      df_success['property_rms_error'],
                      s=150, alpha=0.7, c=range(len(df_success)),
                      cmap='viridis', edgecolors='black', linewidths=2)
    for idx, row in df_success.iterrows():
        axes[1, 1].annotate(row['bc_name'],
                           (row['model_rms_error'],
                            row['property_rms_error']),
                           fontsize=8, ha='right', va='bottom')
    axes[1, 1].set_xlabel('Model RMS error', fontsize=12)
    axes[1, 1].set_ylabel('Property RMS error', fontsize=12)
    axes[1, 1].set_title('Model vs Property Error',
                         fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(sweep.sweep_dir / "analysis_bc_comparison.png",
                dpi=300, bbox_inches='tight')
    print(f"\nAnalysis plot saved to: "
          f"{sweep.sweep_dir / 'analysis_bc_comparison.png'}")

    # Save summary CSV
    df.to_csv(sweep.sweep_dir / "bc_summary.csv", index=False)
    print(f"Summary CSV saved to: {sweep.sweep_dir / 'bc_summary.csv'}")

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
    print("  4. Boundary condition comparison")
    print("  5. Run all examples")
    print("\nOr run with argument: python example_sweep.py [1-5]")
    print("="*80)

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1-5): ").strip()

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
        sweep_dir = sweep_boundary_conditions()
        print(f"\n✓ Results saved to: {sweep_dir}")

    elif choice == '5':
        print("\nRunning all examples...")
        dir1 = sweep_kl_truncation()
        dir2 = sweep_resolution()
        dir3 = sweep_noise_sensitivity()
        dir4 = sweep_boundary_conditions()
        print("\n✓ All results saved!")
        print(f"  - KL sweep: {dir1}")
        print(f"  - Resolution: {dir2}")
        print(f"  - Noise sensitivity: {dir3}")
        print(f"  - Boundary conditions: {dir4}")

    else:
        print(f"\nInvalid choice: {choice}")
        sys.exit(1)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
