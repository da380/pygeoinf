"""
Analysis and visualization tools for PLI experiments.

This module provides functions for analyzing results from PLI experiments,
including comparison plots, summary statistics, and convergence analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json


def load_experiment_results(experiment_dir: Path) -> Dict[str, Any]:
    """Load results from a single experiment directory.

    Args:
        experiment_dir: Path to experiment output directory

    Returns:
        Dictionary with config, metrics, and timings
    """
    experiment_dir = Path(experiment_dir)

    results = {}

    # Load config
    config_path = experiment_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            results['config'] = json.load(f)

    # Load metrics
    metrics_path = experiment_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            results['metrics'] = json.load(f)

    # Load timings
    timings_path = experiment_dir / "timings.json"
    if timings_path.exists():
        with open(timings_path, 'r') as f:
            results['timings'] = json.load(f)

    results['path'] = str(experiment_dir)

    return results


def load_sweep_results(sweep_dir: Path) -> pd.DataFrame:
    """Load results from a parameter sweep directory.

    Args:
        sweep_dir: Path to sweep output directory

    Returns:
        DataFrame with all experiment results
    """
    sweep_dir = Path(sweep_dir)

    # Try to load existing summary
    summary_path = sweep_dir / "summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)

    # Otherwise, reconstruct from individual experiments
    results = []
    for run_dir in sorted(sweep_dir.glob("run_*")):
        if run_dir.is_dir():
            result = load_experiment_results(run_dir)
            if result.get('config') and result.get('metrics'):
                row = result['config'].copy()
                row.update(result['metrics'])
                row.update(result['timings'])
                row['run_name'] = run_dir.name
                row['status'] = 'success'
                results.append(row)

    return pd.DataFrame(results)


def plot_convergence_study(
    df: pd.DataFrame,
    x_param: str,
    y_metrics: List[str],
    output_path: Optional[Path] = None,
    figsize: tuple = (15, 4),
    log_scale: bool = False
) -> plt.Figure:
    """Plot convergence study for a parameter.

    Args:
        df: DataFrame with experiment results
        x_param: Parameter name for x-axis
        y_metrics: List of metric names for y-axis
        output_path: Optional path to save figure
        figsize: Figure size
        log_scale: Whether to use log scale for y-axis

    Returns:
        Matplotlib figure
    """
    n_metrics = len(y_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    df_sorted = df.sort_values(x_param)

    for ax, metric in zip(axes, y_metrics):
        ax.plot(df_sorted[x_param], df_sorted[metric], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel(x_param, fontsize=14)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
        ax.set_title(f'{metric.replace("_", " ").title()} vs {x_param}', fontsize=16)
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale('log')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def plot_parameter_comparison(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    color_metric: str,
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """Plot 2D parameter comparison with color-coded metric.

    Args:
        df: DataFrame with experiment results
        x_param: Parameter for x-axis
        y_param: Parameter for y-axis
        color_metric: Metric for color coding
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        df[x_param], df[y_param],
        c=df[color_metric],
        cmap='viridis',
        s=200,
        edgecolors='white',
        linewidths=2
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_metric.replace('_', ' ').title(), fontsize=14)

    ax.set_xlabel(x_param, fontsize=14)
    ax.set_ylabel(y_param, fontsize=14)
    ax.set_title(f'{color_metric.replace("_", " ").title()} for {x_param} vs {y_param}', fontsize=16)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def plot_timing_breakdown(
    df: pd.DataFrame,
    x_param: str,
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """Plot stacked bar chart of timing breakdown.

    Args:
        df: DataFrame with experiment results
        x_param: Parameter for x-axis
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Identify timing columns
    timing_cols = [col for col in df.columns if
                   col.startswith('setup_') or
                   col.endswith('_compute') or
                   col.endswith('_extract') or
                   col.endswith('_mapping') or
                   col == 'inference_total']

    timing_cols = [col for col in timing_cols if col in df.columns]

    if not timing_cols:
        print("No timing columns found in DataFrame")
        return None

    df_sorted = df.sort_values(x_param)

    fig, ax = plt.subplots(figsize=figsize)

    x_values = range(len(df_sorted))
    bottom = np.zeros(len(df_sorted))

    colors = plt.cm.tab10(np.linspace(0, 1, len(timing_cols)))

    for timing_col, color in zip(timing_cols, colors):
        if timing_col in df_sorted.columns:
            values = df_sorted[timing_col].fillna(0).values
            ax.bar(x_values, values, bottom=bottom, label=timing_col.replace('_', ' '), color=color)
            bottom += values

    ax.set_xticks(x_values)
    ax.set_xticklabels([f"{x_param}={v}" for v in df_sorted[x_param]], rotation=45, ha='right')
    ax.set_xlabel(x_param, fontsize=14)
    ax.set_ylabel('Time (s)', fontsize=14)
    ax.set_title('Timing Breakdown by Component', fontsize=16)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def plot_accuracy_vs_time(
    df: pd.DataFrame,
    accuracy_metric: str = 'property_rms_error',
    time_metric: str = 'inference_total',
    label_param: Optional[str] = None,
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """Plot accuracy vs computational time trade-off.

    Args:
        df: DataFrame with experiment results
        accuracy_metric: Metric for y-axis (accuracy)
        time_metric: Metric for x-axis (time)
        label_param: Optional parameter to label points
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(df[time_metric], df[accuracy_metric], s=200, alpha=0.7,
               edgecolors='white', linewidths=2)

    if label_param and label_param in df.columns:
        for idx, row in df.iterrows():
            ax.annotate(f"{label_param}={row[label_param]}",
                        (row[time_metric], row[accuracy_metric]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax.set_xlabel(time_metric.replace('_', ' ').title() + ' (s)', fontsize=14)
    ax.set_ylabel(accuracy_metric.replace('_', ' ').title(), fontsize=14)
    ax.set_title('Accuracy vs Computational Cost', fontsize=16)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics table.

    Args:
        df: DataFrame with experiment results

    Returns:
        DataFrame with summary statistics
    """
    metric_cols = [
        'model_rms_error', 'model_relative_error',
        'property_rms_error', 'property_mean_abs_error',
        'properties_within_2sigma_pct',
        'data_fit_improvement_pct', 'uncertainty_reduction_pct',
        'inference_total', 'total'
    ]

    available_cols = [col for col in metric_cols if col in df.columns]

    summary = df[available_cols].describe()

    return summary


def compare_experiments(
    experiment_dirs: List[Path],
    output_dir: Optional[Path] = None,
    figsize: tuple = (14, 10)
) -> Dict[str, Any]:
    """Compare multiple individual experiments.

    Args:
        experiment_dirs: List of experiment directories
        output_dir: Optional directory to save comparison plots
        figsize: Figure size

    Returns:
        Dictionary with comparison results
    """
    results = []
    for exp_dir in experiment_dirs:
        result = load_experiment_results(Path(exp_dir))
        if result.get('config') and result.get('metrics'):
            row = result['config'].copy()
            row.update(result['metrics'])
            row.update(result.get('timings', {}))
            row['experiment'] = Path(exp_dir).name
            results.append(row)

    df = pd.DataFrame(results)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Model error comparison
        ax = axes[0, 0]
        ax.bar(df['experiment'], df['model_rms_error'])
        ax.set_ylabel('Model RMS Error', fontsize=12)
        ax.set_title('Model Reconstruction Error', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # Property error comparison
        ax = axes[0, 1]
        ax.bar(df['experiment'], df['property_rms_error'])
        ax.set_ylabel('Property RMS Error', fontsize=12)
        ax.set_title('Property Inference Error', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # Uncertainty calibration
        ax = axes[1, 0]
        ax.bar(df['experiment'], df['properties_within_2sigma_pct'])
        ax.axhline(y=95, color='r', linestyle='--', label='Expected (95%)')
        ax.set_ylabel('% within ±2σ', fontsize=12)
        ax.set_title('Uncertainty Calibration', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Timing comparison
        ax = axes[1, 1]
        if 'total' in df.columns:
            ax.bar(df['experiment'], df['total'])
        elif 'inference_total' in df.columns:
            ax.bar(df['experiment'], df['inference_total'])
        ax.set_ylabel('Time (s)', fontsize=12)
        ax.set_title('Computational Time', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / 'comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'comparison.pdf', bbox_inches='tight')
        plt.close()

        # Save comparison table
        df.to_csv(output_dir / 'comparison.csv', index=False)

    return {
        'dataframe': df,
        'summary': generate_summary_table(df)
    }


def plot_prior_vs_posterior_comparison(
    experiment_dir: Path,
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 5)
) -> plt.Figure:
    """Plot comparison of prior vs posterior for properties.

    This requires access to the saved numpy arrays or re-running the experiment.
    For now, this is a placeholder that creates a comparison from metrics.

    Args:
        experiment_dir: Path to experiment directory
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    results = load_experiment_results(experiment_dir)
    metrics = results.get('metrics', {})
    config = results.get('config', {})

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Error comparison
    ax = axes[0]
    prior_error = metrics.get('data_misfit_prior', 0)
    post_error = metrics.get('data_misfit_posterior', 0)
    ax.bar(['Prior', 'Posterior'], [prior_error, post_error], color=['tab:orange', 'tab:blue'])
    ax.set_ylabel('Data Misfit', fontsize=14)
    ax.set_title('Data Fit: Prior vs Posterior', fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')

    # Uncertainty reduction
    ax = axes[1]
    reduction = metrics.get('uncertainty_reduction_pct', 0)
    ax.bar(['Uncertainty\nReduction'], [reduction], color='tab:green', width=0.5)
    ax.set_ylabel('Reduction (%)', fontsize=14)
    ax.set_title('Property Uncertainty Reduction', fontsize=16)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    # Coverage
    ax = axes[2]
    coverage = metrics.get('properties_within_2sigma_pct', 0)
    ax.bar(['Properties\nwithin ±2σ'], [coverage], color='tab:purple', width=0.5)
    ax.axhline(y=95, color='r', linestyle='--', label='Expected (95%)')
    ax.set_ylabel('Coverage (%)', fontsize=14)
    ax.set_title('Uncertainty Calibration', fontsize=16)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def print_experiment_summary(experiment_dir: Path) -> None:
    """Print a formatted summary of experiment results.

    Args:
        experiment_dir: Path to experiment directory
    """
    results = load_experiment_results(experiment_dir)

    config = results.get('config', {})
    metrics = results.get('metrics', {})
    timings = results.get('timings', {})

    print("=" * 70)
    print(f"EXPERIMENT SUMMARY: {config.get('name', 'Unknown')}")
    print("=" * 70)

    print("\n📊 Configuration:")
    print(f"  Model space dimension (N): {config.get('N', 'N/A')}")
    print(f"  Data points (N_d): {config.get('N_d', 'N/A')}")
    print(f"  Property points (N_p): {config.get('N_p', 'N/A')}")
    print(f"  KL modes (K): {config.get('K', 'N/A')}")
    print(f"  Prior type: {config.get('prior_type', 'N/A')}")
    print(f"  Smoothness (s): {config.get('s', 'N/A')}")
    print(f"  Length scale: {config.get('length_scale', 'N/A')}")
    print(f"  Noise level: {config.get('noise_level', 'N/A')}")

    print("\n📈 Accuracy Metrics:")
    print(f"  Model RMS error: {metrics.get('model_rms_error', 'N/A'):.6f}")
    print(f"  Model relative error: {metrics.get('model_relative_error', 'N/A'):.2%}")
    print(f"  Property RMS error: {metrics.get('property_rms_error', 'N/A'):.6f}")
    print(f"  Properties within ±2σ: {metrics.get('properties_within_2sigma', 'N/A')}/{config.get('N_p', 'N/A')} ({metrics.get('properties_within_2sigma_pct', 'N/A'):.1f}%)")

    print("\n📉 Inference Improvement:")
    print(f"  Data fit improvement: {metrics.get('data_fit_improvement_pct', 'N/A'):.1f}%")
    print(f"  Uncertainty reduction: {metrics.get('uncertainty_reduction_pct', 'N/A'):.1f}%")

    print("\n⏱️ Timing:")
    print(f"  Total time: {timings.get('total', 'N/A'):.2f}s")
    print(f"  Setup: {timings.get('setup_spaces', 0) + timings.get('create_operators', 0):.2f}s")
    print(f"  Prior setup: {timings.get('setup_prior', 'N/A'):.2f}s")
    print(f"  Inference: {timings.get('inference_total', 'N/A'):.2f}s")

    print("=" * 70)
