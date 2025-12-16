"""
Analysis and visualization tools for Demo 1 parameter sweep results.

This module provides functions for analyzing sweep results, generating
summary statistics, and creating publication-quality visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json


def load_sweep_results(sweep_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load sweep results from directory.

    Args:
        sweep_dir: Directory containing sweep results

    Returns:
        Tuple of (summary DataFrame, sweep configuration dict)
    """
    df = pd.read_csv(sweep_dir / 'summary.csv')

    with open(sweep_dir / 'sweep_config.json', 'r') as f:
        config = json.load(f)

    return df, config


def plot_parameter_sensitivity(
    df: pd.DataFrame,
    param: str,
    metrics: List[str],
    output_path: Optional[Path] = None,
    title: Optional[str] = None
) -> None:
    """Plot how metrics vary with a parameter.

    Args:
        df: Summary DataFrame
        param: Parameter name to plot on x-axis
        metrics: List of metric names to plot
        output_path: Path to save figure (None = show only)
        title: Figure title (None = auto-generate)
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4*n_metrics))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        # Group by parameter and compute statistics
        grouped = df.groupby(param)[metric]
        means = grouped.mean()
        stds = grouped.std()

        # Plot with error bars
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   marker='o', markersize=8, linewidth=2, capsize=5)
        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} vs {param}', fontsize=14)
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=16, y=1.002)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_convergence_analysis(
    df: pd.DataFrame,
    resolution_param: str = 'N',
    metrics: List[str] = ['vp_rel_l2_error', 'vs_rel_l2_error', 'rho_rel_l2_error'],
    output_path: Optional[Path] = None
) -> None:
    """Plot convergence analysis showing error vs resolution.

    Args:
        df: Summary DataFrame
        resolution_param: Parameter representing resolution (e.g., 'N')
        metrics: Metrics to plot (errors)
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for metric in metrics:
        grouped = df.groupby(resolution_param)[metric]
        means = grouped.mean()
        stds = grouped.std()

        # Plot with log scale
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   marker='o', markersize=8, linewidth=2, capsize=5,
                   label=metric)

    ax.set_xlabel(resolution_param, fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title('Convergence Analysis', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_timing_analysis(
    df: pd.DataFrame,
    param: str,
    timing_keys: List[str] = ['time_setup_model_space', 'time_setup_prior',
                               'time_run_inference', 'time_plot_results'],
    output_path: Optional[Path] = None
) -> None:
    """Plot timing breakdown vs parameter.

    Args:
        df: Summary DataFrame
        param: Parameter to vary on x-axis
        timing_keys: List of timing keys to plot
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Stacked bar chart
    grouped = df.groupby(param)[timing_keys].mean()
    grouped.plot(kind='bar', stacked=True, ax=axes[0], colormap='tab10')
    axes[0].set_xlabel(param, fontsize=12)
    axes[0].set_ylabel('Time (s)', fontsize=12)
    axes[0].set_title('Timing Breakdown', fontsize=14)
    axes[0].legend(title='Component', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Total time with breakdown
    total_times = df.groupby(param)['time_total'].mean()
    axes[1].plot(total_times.index, total_times.values,
                marker='o', markersize=8, linewidth=2, label='Total')

    for timing_key in timing_keys:
        times = df.groupby(param)[timing_key].mean()
        axes[1].plot(times.index, times.values,
                    marker='s', markersize=6, linewidth=1.5,
                    alpha=0.7, label=timing_key.replace('time_', ''))

    axes[1].set_xlabel(param, fontsize=12)
    axes[1].set_ylabel('Time (s)', fontsize=12)
    axes[1].set_title('Component Times', fontsize=14)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_heatmap_2d(
    df: pd.DataFrame,
    param1: str,
    param2: str,
    metric: str,
    output_path: Optional[Path] = None,
    title: Optional[str] = None
) -> None:
    """Create heatmap for metric vs two parameters.

    Args:
        df: Summary DataFrame
        param1: First parameter (x-axis)
        param2: Second parameter (y-axis)
        metric: Metric to visualize as color
        output_path: Path to save figure
        title: Figure title
    """
    # Pivot to create 2D grid
    pivot = df.pivot_table(values=metric, index=param2, columns=param1, aggfunc='mean')

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis',
               cbar_kws={'label': metric}, ax=ax)

    ax.set_xlabel(param1, fontsize=12)
    ax.set_ylabel(param2, fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'{metric} vs {param1} and {param2}', fontsize=14)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_component_comparison(
    df: pd.DataFrame,
    param: str,
    output_path: Optional[Path] = None
) -> None:
    """Compare errors across components (vp, vs, rho) vs parameter.

    Args:
        df: Summary DataFrame
        param: Parameter to vary on x-axis
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Function components
    components = ['vp', 'vs', 'rho']
    colors = ['red', 'blue', 'green']

    # Plot 1: Relative L2 errors
    ax = axes[0, 0]
    for comp, color in zip(components, colors):
        metric = f'{comp}_rel_l2_error'
        grouped = df.groupby(param)[metric]
        means = grouped.mean()
        stds = grouped.std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   marker='o', markersize=6, linewidth=2, capsize=4,
                   color=color, label=comp)
    ax.set_xlabel(param, fontsize=11)
    ax.set_ylabel('Relative L2 Error', fontsize=11)
    ax.set_title('Relative L2 Errors', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Max errors
    ax = axes[0, 1]
    for comp, color in zip(components, colors):
        metric = f'{comp}_max_error'
        grouped = df.groupby(param)[metric]
        means = grouped.mean()
        stds = grouped.std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   marker='o', markersize=6, linewidth=2, capsize=4,
                   color=color, label=comp)
    ax.set_xlabel(param, fontsize=11)
    ax.set_ylabel('Max Pointwise Error', fontsize=11)
    ax.set_title('Max Pointwise Errors', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 3: Posterior uncertainties
    ax = axes[1, 0]
    for comp, color in zip(components, colors):
        metric = f'{comp}_avg_std'
        grouped = df.groupby(param)[metric]
        means = grouped.mean()
        stds = grouped.std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   marker='o', markersize=6, linewidth=2, capsize=4,
                   color=color, label=comp)
    ax.set_xlabel(param, fontsize=11)
    ax.set_ylabel('Average Posterior Std', fontsize=11)
    ax.set_title('Posterior Uncertainties', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Sigma errors
    ax = axes[1, 1]
    for i, sigma in enumerate(['sigma_0', 'sigma_1']):
        metric = f'{sigma}_rel_error'
        grouped = df.groupby(param)[metric]
        means = grouped.mean()
        stds = grouped.std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   marker='o', markersize=6, linewidth=2, capsize=4,
                   label=f'σ_{i}')
    ax.set_xlabel(param, fontsize=11)
    ax.set_ylabel('Relative Error', fontsize=11)
    ax.set_title('Sigma Component Errors', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_noise_sensitivity(
    df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> None:
    """Specialized plot for noise sensitivity analysis.

    Args:
        df: Summary DataFrame with 'noise_level' column
        output_path: Path to save figure
    """
    if 'noise_level' not in df.columns:
        raise ValueError("DataFrame must contain 'noise_level' column")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: All component errors vs noise
    ax = axes[0, 0]
    components = ['vp', 'vs', 'rho']
    colors = ['red', 'blue', 'green']
    for comp, color in zip(components, colors):
        metric = f'{comp}_rel_l2_error'
        grouped = df.groupby('noise_level')[metric]
        means = grouped.mean()
        ax.plot(means.index, means.values, marker='o', markersize=6,
               linewidth=2, color=color, label=comp)
    ax.set_xlabel('Noise Level', fontsize=11)
    ax.set_ylabel('Relative L2 Error', fontsize=11)
    ax.set_title('Function Component Errors', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Plot 2: Sigma errors vs noise
    ax = axes[0, 1]
    for i, sigma in enumerate(['sigma_0', 'sigma_1']):
        metric = f'{sigma}_rel_error'
        grouped = df.groupby('noise_level')[metric]
        means = grouped.mean()
        ax.plot(means.index, means.values, marker='o', markersize=6,
               linewidth=2, label=f'σ_{i}')
    ax.set_xlabel('Noise Level', fontsize=11)
    ax.set_ylabel('Relative Error', fontsize=11)
    ax.set_title('Sigma Component Errors', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Plot 3: Data fit vs noise
    ax = axes[1, 0]
    metric = 'data_rel_l2_error'
    grouped = df.groupby('noise_level')[metric]
    means = grouped.mean()
    stds = grouped.std()
    ax.errorbar(means.index, means.values, yerr=stds.values,
               marker='o', markersize=6, linewidth=2, capsize=4)
    ax.plot(means.index, means.index, 'k--', alpha=0.5, label='y=x reference')
    ax.set_xlabel('Noise Level', fontsize=11)
    ax.set_ylabel('Data Fit Error', fontsize=11)
    ax.set_title('Data Fit Quality', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Plot 4: Average posterior uncertainty vs noise
    ax = axes[1, 1]
    for comp, color in zip(components, colors):
        metric = f'{comp}_avg_std'
        grouped = df.groupby('noise_level')[metric]
        means = grouped.mean()
        ax.plot(means.index, means.values, marker='o', markersize=6,
               linewidth=2, color=color, label=comp)
    ax.set_xlabel('Noise Level', fontsize=11)
    ax.set_ylabel('Average Posterior Std', fontsize=11)
    ax.set_title('Posterior Uncertainty vs Noise', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def generate_sweep_report(
    sweep_dir: Path,
    output_dir: Optional[Path] = None
) -> None:
    """Generate comprehensive analysis report for a sweep.

    Creates multiple figures and a text summary.

    Args:
        sweep_dir: Directory containing sweep results
        output_dir: Directory for report outputs (None = use sweep_dir/analysis)
    """
    # Load results
    df, config = load_sweep_results(sweep_dir)

    # Set up output directory
    if output_dir is None:
        output_dir = sweep_dir / 'analysis'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Generating sweep report for: {config['name']}")
    print(f"Number of runs: {len(df)}")

    # Get swept parameters
    swept_params = list(config['sweep_params'].keys())
    print(f"Swept parameters: {swept_params}")

    # Generate figures for each swept parameter
    for param in swept_params:
        print(f"  Analyzing {param}...")

        # Parameter sensitivity plot
        metrics = ['vp_rel_l2_error', 'vs_rel_l2_error', 'rho_rel_l2_error',
                  'sigma_0_rel_error', 'sigma_1_rel_error']
        plot_parameter_sensitivity(
            df, param, metrics,
            output_path=output_dir / f'sensitivity_{param}.png',
            title=f'Sensitivity to {param}'
        )

        # Component comparison
        plot_component_comparison(
            df, param,
            output_path=output_dir / f'components_{param}.png'
        )

        # Timing analysis
        plot_timing_analysis(
            df, param,
            output_path=output_dir / f'timing_{param}.png'
        )

    # Special analyses
    if 'noise_level' in df.columns:
        print("  Generating noise sensitivity analysis...")
        plot_noise_sensitivity(df, output_path=output_dir / 'noise_analysis.png')

    if 'N' in df.columns:
        print("  Generating convergence analysis...")
        plot_convergence_analysis(
            df, 'N',
            output_path=output_dir / 'convergence.png'
        )

    # 2D heatmaps for pairs of parameters
    if len(swept_params) >= 2:
        print("  Generating 2D heatmaps...")
        for i, param1 in enumerate(swept_params):
            for param2 in swept_params[i+1:]:
                plot_heatmap_2d(
                    df, param1, param2, 'vp_rel_l2_error',
                    output_path=output_dir / f'heatmap_{param1}_{param2}.png',
                    title=f'vp error: {param1} vs {param2}'
                )

    # Generate text summary
    print("  Writing summary statistics...")
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write(f"Sweep Analysis Report\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Sweep name: {config['name']}\n")
        f.write(f"Number of runs: {len(df)}\n")
        f.write(f"Swept parameters: {', '.join(swept_params)}\n\n")

        # Overall statistics
        f.write("Overall Statistics\n")
        f.write("-" * 60 + "\n")
        metrics = ['vp_rel_l2_error', 'vs_rel_l2_error', 'rho_rel_l2_error',
                  'sigma_0_rel_error', 'sigma_1_rel_error', 'time_total']
        for metric in metrics:
            if metric in df.columns:
                f.write(f"{metric}:\n")
                f.write(f"  Mean: {df[metric].mean():.6f}\n")
                f.write(f"  Std:  {df[metric].std():.6f}\n")
                f.write(f"  Min:  {df[metric].min():.6f}\n")
                f.write(f"  Max:  {df[metric].max():.6f}\n")

        # Best configurations
        f.write("\n\nBest Configurations\n")
        f.write("-" * 60 + "\n")
        for metric in ['vp_rel_l2_error', 'vs_rel_l2_error', 'rho_rel_l2_error']:
            if metric in df.columns:
                best_idx = df[metric].idxmin()
                f.write(f"\nBest for {metric}:\n")
                f.write(f"  Value: {df.loc[best_idx, metric]:.6f}\n")
                f.write(f"  Parameters:\n")
                for param in swept_params:
                    f.write(f"    {param}: {df.loc[best_idx, param]}\n")

    print(f"Report complete! Results saved to: {output_dir}")


def compare_sweeps(
    sweep_dirs: List[Path],
    sweep_names: List[str],
    metric: str = 'vp_rel_l2_error',
    output_path: Optional[Path] = None
) -> None:
    """Compare results from multiple sweeps.

    Args:
        sweep_dirs: List of sweep directories
        sweep_names: Names for each sweep
        metric: Metric to compare
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for sweep_dir, name in zip(sweep_dirs, sweep_names):
        df, _ = load_sweep_results(sweep_dir)

        # Plot distribution
        ax.hist(df[metric], bins=20, alpha=0.5, label=name, density=True)

    ax.set_xlabel(metric, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Comparison of {metric}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    else:
        plt.show()

    plt.close()
