"""Utilities for profiling integration and plotting results.

Put profiling logic here so notebooks can call a thin wrapper.
"""
from typing import Dict, Any, Tuple
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def run_profiling(domain, func, integration_parameters: Dict[str, Any], repeats: int = 3, save_prefix: str = "integration_profile") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run profiling for the provided domain/function and integration parameters.

    Returns (df, summary):
      - df: long-form DataFrame with one row per run and columns for parameters + measurements
      - summary: grouped summary DataFrame with time_mean_s, time_std_s, count

    The function also writes files: {save_prefix}_flat.csv, {save_prefix}_summary.csv, {save_prefix}_flat.pkl
    """
    results = []
    methods = integration_parameters.get('method', [])
    n_points_list = integration_parameters.get('n_points', [])
    vectorized = integration_parameters.get('vectorized', False)

    for method in methods:
        for n_points in n_points_list:
            parameters = {
                'method': method,
                'n_points': int(n_points),
            }
            for run_idx in range(repeats):
                t0 = time.perf_counter()
                measurement_result = domain.integrate(func, method=method, n_points=n_points, vectorized=vectorized)
                t1 = time.perf_counter()
                measurements = {
                    'run': int(run_idx),
                    'time_s': float(t1 - t0),
                    'result': float(measurement_result) if np.isscalar(measurement_result) else None,
                }
                record = {'parameters': parameters, 'measurements': measurements}
                results.append(record)

    # flatten
    flat_records = []
    for rec in results:
        flat = {}
        flat.update(rec['parameters'])
        flat.update(rec['measurements'])
        flat_records.append(flat)

    df = pd.DataFrame(flat_records)

    # summary
    summary = (
        df.groupby(['method', 'n_points'])['time_s']
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .rename(columns={'mean': 'time_mean_s', 'std': 'time_std_s'})
    )

    # save artifacts
    df.to_csv(f"{save_prefix}_flat.csv", index=False)
    summary.to_csv(f"{save_prefix}_summary.csv", index=False)
    df.to_pickle(f"{save_prefix}_flat.pkl")

    return df, summary


# General-purpose plotter: plot any measurement vs any parameter
def plot_measurement_vs_parameter(df: pd.DataFrame, measurement: str, parameter: str, by: str = None, agg: str = 'mean', error: str = 'std', kind: str = 'line', log_x: bool = False, figsize: tuple = (7, 4)) -> pd.DataFrame:
    """Plot any measurement versus any parameter from dataframe `df`.

    Returns grouped DataFrame used for plotting.
    """
    if measurement not in df.columns:
        raise ValueError(f"measurement '{measurement}' not found in dataframe")
    if parameter not in df.columns:
        raise ValueError(f"parameter '{parameter}' not found in dataframe")

    group_cols = [parameter]
    if by:
        if by not in df.columns:
            raise ValueError(f"grouping column '{by}' not found in dataframe")
        group_cols.append(by)

    agg_funcs = {measurement: ['mean', 'std', 'count']}
    grouped = df.groupby(group_cols).agg(agg_funcs)
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()

    mean_col = f"{measurement}_mean"
    std_col = f"{measurement}_std"

    plt.figure(figsize=figsize)

    if kind == 'line':
        if by:
            pivot_mean = grouped.pivot(index=parameter, columns=by, values=mean_col)
            pivot_std = grouped.pivot(index=parameter, columns=by, values=std_col) if error else None
            try:
                pivot_mean = pivot_mean.sort_index()
                if pivot_std is not None:
                    pivot_std = pivot_std.reindex(pivot_mean.index)
            except Exception:
                pass
            for col in pivot_mean.columns:
                x = pivot_mean.index.values
                y = pivot_mean[col].values
                yerr = pivot_std[col].values if (pivot_std is not None and col in pivot_std.columns) else None
                plt.errorbar(x, y, yerr=yerr, marker='o', label=str(col), capsize=3)
        else:
            g = grouped.sort_values(parameter)
            x = g[parameter].values
            y = g[mean_col].values
            yerr = g[std_col].values if error else None
            plt.errorbar(x, y, yerr=yerr, marker='o', capsize=3)

        plt.xlabel(parameter)
        plt.ylabel(measurement)
        if by:
            plt.legend(title=by)
        if log_x:
            plt.xscale('log')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif kind == 'bar':
        if by:
            sns.barplot(data=df, x=parameter, y=measurement, hue=by, ci='sd' if error == 'std' else None, estimator=np.mean)
        else:
            sns.barplot(data=df, x=parameter, y=measurement, ci='sd' if error == 'std' else None, estimator=np.mean)
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("kind must be 'line' or 'bar'")

    return grouped


def geometric_range(start, stop, steps):
    """Return geometric sequence of `steps` values between start and stop (inclusive)."""
    return np.exp(np.linspace(np.log(start), np.log(stop), steps))
