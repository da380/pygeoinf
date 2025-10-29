#!/usr/bin/env python3
"""
Quick example script for running PLI parameter sweeps.

Usage:
    python example_sweep.py
"""

from run_pli_experiments import PLISweep, PLIExperiment
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
        'method': 'fem',
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


# Example 5: Comprehensive N_d, BC, and method comparison
def sweep_data_bc_method():
    """
    Comprehensive sweep over data points, boundary conditions, and methods.

    This example systematically tests:
    - Different numbers of data points (N_d)
    - All boundary condition types
    - Both spectral and FEM methods

    This helps understand:
    - How many data points are needed for accurate inference
    - Which boundary conditions work best for different data densities
    - Performance and accuracy trade-offs between spectral and FEM methods
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Comprehensive Data Points, BC, and Method Comparison")
    print("="*80)

    sweep = PLISweep(sweep_name="data_bc_method_comparison")

    base_config = {
        'N': 100,
        'N_p': 20,
        'K': 100,
        'basis': None,
        'alpha': 0.1,
        'noise_level': 0.1,
        'compute_model_posterior': False,
        'random_seed': 42,
        'n_jobs': 30,
        'dofs': 100
    }

    # Define boundary condition configurations
    bc_configs = [
        {'bc_type': 'dirichlet', 'left': 0, 'right': 0},
        {'bc_type': 'neumann', 'left': 0, 'right': 0},
        {'bc_type': 'mixed_dirichlet_neumann', 'left': 0, 'right': 0},
        {'bc_type': 'mixed_neumann_dirichlet', 'left': 0, 'right': 0}
    ]

    bc_names = [
        'Dirichlet',
        'Neumann',
        'Mixed_D-N',
        'Mixed_N-D'
    ]

    # Test different data point counts and methods
    n_data_values = [10, 20, 30, 50, 75, 100]
    methods = ['spectral', 'fem']

    results = []
    total_runs = len(bc_configs) * len(n_data_values) * len(methods)
    run_count = 0

    for bc_config, bc_name in zip(bc_configs, bc_names):
        for n_data in n_data_values:
            for method in methods:
                run_count += 1
                print(f"\n[{run_count}/{total_runs}] Testing: {bc_name}, "
                      f"N_d={n_data}, method={method}")

                config = base_config.copy()
                config['N_d'] = n_data
                config['bc_config'] = bc_config
                config['method'] = method
                config['bc_name'] = bc_name

                run_name = f"{bc_name}_Nd{n_data}_{method}"
                run_dir = sweep.sweep_dir / run_name

                try:
                    from run_pli_experiments import PLIExperiment
                    experiment = PLIExperiment(config, run_dir)
                    metrics = experiment.run()

                    result = config.copy()
                    result.update(metrics)
                    result.update(experiment.timings)
                    result['run_name'] = run_name
                    result['status'] = 'success'
                    results.append(result)

                    inf_time = experiment.timings['inference_total']
                    print(f"  ✓ Success: RMS error = "
                          f"{metrics['property_rms_error']:.5f}, "
                          f"time = {inf_time:.2f}s")

                except Exception as e:
                    print(f"  ✗ Failed: {e}")
                    result = config.copy()
                    result['run_name'] = run_name
                    result['status'] = 'failed'
                    result['error'] = str(e)
                    results.append(result)

    df = pd.DataFrame(results)
    df_success = df[df['status'] == 'success']

    # Save detailed results
    df.to_csv(sweep.sweep_dir / "detailed_results.csv", index=False)

    # Analysis and visualization
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    # Summary by BC and method
    summary = df_success.groupby(['bc_name', 'method']).agg({
        'property_rms_error': ['mean', 'std'],
        'inference_total': ['mean', 'std'],
        'N_d': 'count'
    }).round(4)
    print("\nSummary by BC and Method:")
    print(summary)

    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Accuracy vs N_d for each BC (spectral vs FEM)
    for idx, (bc_name, bc_data) in enumerate(df_success.groupby('bc_name')):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        for method in ['spectral', 'fem']:
            method_data = bc_data[bc_data['method'] == method]
            if len(method_data) > 0:
                ax.plot(method_data['N_d'], method_data['property_rms_error'],
                       'o-', label=method.upper(), linewidth=2, markersize=8,
                       alpha=0.7)

        ax.set_xlabel('Number of data points (N_d)', fontsize=11)
        ax.set_ylabel('Property RMS error', fontsize=11)
        ax.set_title(f'{bc_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # 2. Timing comparison (spectral vs FEM)
    ax = fig.add_subplot(gs[2, 0])
    timing_data = df_success.groupby(['method', 'N_d'])['inference_total'].mean().unstack()
    timing_data.T.plot(kind='bar', ax=ax, width=0.7, alpha=0.8)
    ax.set_xlabel('Number of data points (N_d)', fontsize=11)
    ax.set_ylabel('Average inference time (s)', fontsize=11)
    ax.set_title('Computational Cost: Spectral vs FEM',
                 fontsize=12, fontweight='bold')
    ax.legend(title='Method', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # 3. Heatmap: BC vs N_d (spectral method)
    ax = fig.add_subplot(gs[2, 1])
    spectral_data = df_success[df_success['method'] == 'spectral']
    pivot_spectral = spectral_data.pivot_table(
        values='property_rms_error',
        index='bc_name',
        columns='N_d',
        aggfunc='mean'
    )
    sns.heatmap(pivot_spectral, annot=True, fmt='.4f', cmap='YlOrRd_r',
                ax=ax, cbar_kws={'label': 'RMS error'})
    ax.set_title('Spectral Method Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of data points (N_d)', fontsize=11)
    ax.set_ylabel('Boundary Condition', fontsize=11)

    # 4. Heatmap: BC vs N_d (FEM method)
    ax = fig.add_subplot(gs[2, 2])
    fem_data = df_success[df_success['method'] == 'fem']
    pivot_fem = fem_data.pivot_table(
        values='property_rms_error',
        index='bc_name',
        columns='N_d',
        aggfunc='mean'
    )
    sns.heatmap(pivot_fem, annot=True, fmt='.4f', cmap='YlGnBu_r',
                ax=ax, cbar_kws={'label': 'RMS error'})
    ax.set_title('FEM Method Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of data points (N_d)', fontsize=11)
    ax.set_ylabel('Boundary Condition', fontsize=11)

    plt.savefig(sweep.sweep_dir / "analysis_comprehensive.png",
                dpi=300, bbox_inches='tight')
    print(f"\nComprehensive analysis plot saved to: "
          f"{sweep.sweep_dir / 'analysis_comprehensive.png'}")

    # Create additional comparison plots
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Method comparison by BC
    for idx, bc_name in enumerate(bc_names):
        ax = axes[idx // 2, idx % 2]
        bc_data = df_success[df_success['bc_name'] == bc_name]

        for method in ['spectral', 'fem']:
            method_data = bc_data[bc_data['method'] == method]
            if len(method_data) > 0:
                ax.scatter(method_data['N_d'], method_data['property_rms_error'],
                          label=method.upper(), s=100, alpha=0.6)
                # Add trend line
                z = np.polyfit(method_data['N_d'], method_data['property_rms_error'], 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(method_data['N_d'].min(),
                                      method_data['N_d'].max(), 100)
                ax.plot(x_smooth, p(x_smooth), '--', alpha=0.5)

        ax.set_xlabel('Number of data points (N_d)', fontsize=11)
        ax.set_ylabel('Property RMS error', fontsize=11)
        ax.set_title(f'{bc_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(sweep.sweep_dir / "analysis_method_comparison.png",
                dpi=300, bbox_inches='tight')
    print(f"Method comparison plot saved to: "
          f"{sweep.sweep_dir / 'analysis_method_comparison.png'}")

    # Save summary statistics
    summary_stats = df_success.groupby(['bc_name', 'method', 'N_d']).agg({
        'property_rms_error': 'mean',
        'model_rms_error': 'mean',
        'inference_total': 'mean',
        'properties_within_2sigma_pct': 'mean'
    }).reset_index()
    summary_stats.to_csv(sweep.sweep_dir / "summary_statistics.csv", index=False)
    print(f"Summary statistics saved to: "
          f"{sweep.sweep_dir / 'summary_statistics.csv'}")

    return sweep.sweep_dir


# Example 6: Bessel-Sobolev parameter sweep
def sweep_bessel_sobolev_params():
    """
    Sweep over Bessel-Sobolev parameters k and s.

    This example explores how the Bessel-Sobolev prior parameters affect
    inference quality:

    - k: Bessel parameter (correlation length scale)
      * Higher k → shorter correlation length
      * Lower k → longer correlation length
      * k controls the balance between identity and Laplacian

    - s: Sobolev order (smoothness)
      * Higher s → smoother prior samples
      * s = 1 is typical, s = 2 gives very smooth priors
      * Controls regularity of the covariance operator

    The prior covariance is C_0 = (k²I + L)^(-s) where L is the Laplacian.

    This sweep tests combinations of k and s to understand:
    1. How correlation length affects property inference accuracy
    2. How smoothness order affects reconstruction quality
    3. Trade-offs between regularization strength and data fit
    4. Optimal parameter combinations for different problem types
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Bessel-Sobolev Parameter Sweep")
    print("="*80)

    sweep = PLISweep(sweep_name="bessel_sobolev_params")

    base_config = {
        'N': 100,
        'N_d': 50,
        'N_p': 20,
        'K': 100,
        'basis': None,
        'noise_level': 0.1,
        'compute_model_posterior': False,
        'random_seed': 42,
        'n_jobs': 30,
        # Bessel-Sobolev prior (default)
        'prior_type': 'bessel_sobolev',
        'alpha': 0.1,
        'method': 'spectral',
        'dofs': 100,
        'n_samples': 2048,
        'use_fast_transforms': True,
        'bc_config': {'bc_type': 'dirichlet', 'left': 0, 'right': 0}
    }

    # Parameter grid: sweep over k and s
    param_grid = {
        'k': [0.5, 1.0, 2.0, 5.0],     # Correlation length scale
        's': [0.5, 1.0, 1.5, 2.0]      # Smoothness order
    }

    results = []
    total_runs = len(param_grid['k']) * len(param_grid['s'])
    run_idx = 0

    for k_val in param_grid['k']:
        for s_val in param_grid['s']:
            run_idx += 1
            config = base_config.copy()
            config['k'] = k_val
            config['s'] = s_val

            run_name = f"k{k_val}_s{s_val}"

            print(f"\n[{run_idx}/{total_runs}] Testing: k={k_val}, s={s_val}")

            try:
                # Create output directory
                run_dir = sweep.sweep_dir / run_name
                experiment = PLIExperiment(config, run_dir)

                # Run experiment
                metrics = experiment.run()

                # Store results
                result = config.copy()
                result.update(metrics)
                result.update(experiment.timings)
                result['run_name'] = run_name
                result['status'] = 'success'
                results.append(result)

                inf_time = experiment.timings['inference_total']
                print(f"  ✓ Success: RMS error = "
                      f"{metrics['property_rms_error']:.5f}, "
                      f"time = {inf_time:.2f}s")

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                result = config.copy()
                result['run_name'] = run_name
                result['status'] = 'failed'
                result['error'] = str(e)
                results.append(result)

    df = pd.DataFrame(results)
    df_success = df[df['status'] == 'success']

    # Save detailed results
    df.to_csv(sweep.sweep_dir / "detailed_results.csv", index=False)

    # Analysis and visualization
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    # Summary statistics
    summary = df_success.groupby(['k', 's']).agg({
        'property_rms_error': ['mean', 'std'],
        'model_rms_error': ['mean', 'std'],
        'inference_total': 'mean',
        'properties_within_2sigma_pct': 'mean'
    }).round(4)
    print("\nSummary by k and s:")
    print(summary)

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Heatmap: Property RMS error vs k and s
    ax = fig.add_subplot(gs[0, 0])
    pivot_prop_error = df_success.pivot_table(
        values='property_rms_error',
        index='s',
        columns='k',
        aggfunc='mean'
    )
    sns.heatmap(pivot_prop_error, annot=True, fmt='.4f', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': 'Property RMS error'})
    ax.set_title('Property Error: k vs s', fontsize=14, fontweight='bold')
    ax.set_xlabel('k (correlation length)', fontsize=12)
    ax.set_ylabel('s (smoothness)', fontsize=12)

    # 2. Heatmap: Model RMS error vs k and s
    ax = fig.add_subplot(gs[0, 1])
    pivot_model_error = df_success.pivot_table(
        values='model_rms_error',
        index='s',
        columns='k',
        aggfunc='mean'
    )
    sns.heatmap(pivot_model_error, annot=True, fmt='.4f', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': 'Model RMS error'})
    ax.set_title('Model Error: k vs s', fontsize=14, fontweight='bold')
    ax.set_xlabel('k (correlation length)', fontsize=12)
    ax.set_ylabel('s (smoothness)', fontsize=12)

    # 3. Heatmap: Inference time vs k and s
    ax = fig.add_subplot(gs[0, 2])
    pivot_time = df_success.pivot_table(
        values='inference_total',
        index='s',
        columns='k',
        aggfunc='mean'
    )
    sns.heatmap(pivot_time, annot=True, fmt='.2f', cmap='YlGnBu',
                ax=ax, cbar_kws={'label': 'Time (s)'})
    ax.set_title('Inference Time: k vs s', fontsize=14, fontweight='bold')
    ax.set_xlabel('k (correlation length)', fontsize=12)
    ax.set_ylabel('s (smoothness)', fontsize=12)

    # 4. Line plot: Property error vs k for different s values
    ax = fig.add_subplot(gs[1, 0])
    for s_val in sorted(df_success['s'].unique()):
        subset = df_success[df_success['s'] == s_val]
        ax.plot(subset['k'], subset['property_rms_error'],
                'o-', label=f's={s_val}', linewidth=2, markersize=8)
    ax.set_xlabel('k (correlation length)', fontsize=12)
    ax.set_ylabel('Property RMS error', fontsize=12)
    ax.set_title('Error vs k (for different s)', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 5. Line plot: Property error vs s for different k values
    ax = fig.add_subplot(gs[1, 1])
    for k_val in sorted(df_success['k'].unique()):
        subset = df_success[df_success['k'] == k_val]
        ax.plot(subset['s'], subset['property_rms_error'],
                'o-', label=f'k={k_val}', linewidth=2, markersize=8)
    ax.set_xlabel('s (smoothness)', fontsize=12)
    ax.set_ylabel('Property RMS error', fontsize=12)
    ax.set_title('Error vs s (for different k)', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 6. Scatter: Property error vs model error
    ax = fig.add_subplot(gs[1, 2])
    scatter = ax.scatter(df_success['model_rms_error'],
                        df_success['property_rms_error'],
                        c=df_success['k'], s=df_success['s']*100,
                        cmap='viridis', alpha=0.6, edgecolors='black')
    ax.set_xlabel('Model RMS error', fontsize=12)
    ax.set_ylabel('Property RMS error', fontsize=12)
    ax.set_title('Property vs Model Error\n(color=k, size=s)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('k', fontsize=11)

    # 7. Bar plot: Uncertainty calibration (% within 2σ)
    ax = fig.add_subplot(gs[2, 0])
    grouped = df_success.groupby(['k', 's'])[
        'properties_within_2sigma_pct'].mean().unstack()
    grouped.plot(kind='bar', ax=ax, width=0.8, alpha=0.8)
    ax.set_xlabel('k (correlation length)', fontsize=12)
    ax.set_ylabel('% properties within 2σ', fontsize=12)
    ax.set_title('Uncertainty Calibration', fontsize=14, fontweight='bold')
    ax.legend(title='s', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # 8. Line plot: Inference time vs k
    ax = fig.add_subplot(gs[2, 1])
    for s_val in sorted(df_success['s'].unique()):
        subset = df_success[df_success['s'] == s_val]
        ax.plot(subset['k'], subset['inference_total'],
                'o-', label=f's={s_val}', linewidth=2, markersize=8)
    ax.set_xlabel('k (correlation length)', fontsize=12)
    ax.set_ylabel('Inference time (s)', fontsize=12)
    ax.set_title('Computational Cost vs k', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 9. Best parameter identification
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    # Find best parameters
    best_accuracy = df_success.loc[
        df_success['property_rms_error'].idxmin()]
    best_speed = df_success.loc[df_success['inference_total'].idxmin()]
    best_calibration = df_success.loc[
        df_success['properties_within_2sigma_pct'].idxmax()]

    text_info = (
        "OPTIMAL PARAMETERS\n"
        "=" * 40 + "\n\n"
        f"Best Accuracy:\n"
        f"  k={best_accuracy['k']}, s={best_accuracy['s']}\n"
        f"  RMS error: {best_accuracy['property_rms_error']:.5f}\n\n"
        f"Fastest:\n"
        f"  k={best_speed['k']}, s={best_speed['s']}\n"
        f"  Time: {best_speed['inference_total']:.2f}s\n\n"
        f"Best Calibration:\n"
        f"  k={best_calibration['k']}, s={best_calibration['s']}\n"
        f"  Within 2σ: "
        f"{best_calibration['properties_within_2sigma_pct']:.1f}%\n\n"
        f"Total runs: {len(df_success)}/{len(df)}"
    )
    ax.text(0.1, 0.5, text_info, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig(sweep.sweep_dir / "analysis_bessel_sobolev.png",
                dpi=300, bbox_inches='tight')
    print(f"\nAnalysis plot saved to: "
          f"{sweep.sweep_dir / 'analysis_bessel_sobolev.png'}")

    # Save summary statistics
    summary_stats = df_success.groupby(['k', 's']).agg({
        'property_rms_error': 'mean',
        'model_rms_error': 'mean',
        'inference_total': 'mean',
        'properties_within_2sigma_pct': 'mean',
        'data_fit_improvement_pct': 'mean'
    }).reset_index()
    summary_stats.to_csv(sweep.sweep_dir / "summary_statistics.csv",
                        index=False)
    print(f"Summary statistics saved to: "
          f"{sweep.sweep_dir / 'summary_statistics.csv'}")

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
    print("  5. Comprehensive N_d, BC, and method comparison")
    print("  6. Bessel-Sobolev parameters (k and s) sweep")
    print("  7. Run all examples")
    print("\nOr run with argument: python example_sweep.py [1-7]")
    print("="*80)

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1-7): ").strip()

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
        sweep_dir = sweep_data_bc_method()
        print(f"\n✓ Results saved to: {sweep_dir}")

    elif choice == '6':
        sweep_dir = sweep_bessel_sobolev_params()
        print(f"\n✓ Results saved to: {sweep_dir}")

    elif choice == '7':
        print("\nRunning all examples...")
        dir1 = sweep_kl_truncation()
        dir2 = sweep_resolution()
        dir3 = sweep_noise_sensitivity()
        dir4 = sweep_boundary_conditions()
        dir5 = sweep_data_bc_method()
        dir6 = sweep_bessel_sobolev_params()
        print("\n✓ All results saved!")
        print(f"  - KL sweep: {dir1}")
        print(f"  - Resolution: {dir2}")
        print(f"  - Noise sensitivity: {dir3}")
        print(f"  - Boundary conditions: {dir4}")
        print(f"  - Data/BC/Method: {dir5}")
        print(f"  - Bessel-Sobolev: {dir6}")

    else:
        print(f"\nInvalid choice: {choice}")
        sys.exit(1)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
