"""Profiler utilities for timing interval integrations for Gram-matrix-style tasks.
This module collects long-form timing records (parameters + measurements) but does
NOT assemble the full Gram matrix. Use the notebook to call these helpers.
"""
from typing import List, Tuple, Callable
import time
import numpy as np
import pandas as pd


def geometric_range(start: int, stop: int, steps: int) -> List[int]:
    """Return an increasing list of unique integer values spaced geometrically.
    Rounds values and removes duplicates to avoid repeated n_points.
    """
    vals = np.geomspace(start, stop, num=steps)
    ints = np.unique(np.round(vals).astype(int))
    return [int(v) for v in ints]


def run_profiling(
    domain,
    provider,
    n_functions: int,
    methods: List[str],
    n_points_list: List[int],
    repeats: int = 3,
    vectorized: bool = True,
    save_prefix: str = "profile_gram_matrix",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run profiling over a set of integration parameters.

    Parameters
    - domain: IntervalDomain instance with .integrate(callable, ...)
    - provider: a function provider exposing get_function_by_index(i)
    - n_functions: number of functions to sample from provider
    - methods: list of integration method names (e.g., ['simpson','trapz','quad'])
    - n_points_list: list of integer n_points to test
    - repeats: repetitions per configuration
    - vectorized: pass-through to domain.integrate when supported
    - save_prefix: file prefix for saved artifacts

    Returns: (df, summary) where df is long-form records and summary is grouped stats
    """
    records = []
    for method in methods:
        for n_points in n_points_list:
            for rep in range(int(repeats)):
                for i in range(int(n_functions)):
                    f = provider.get_function_by_index(i)
                    product = f * f
                    integrand = lambda x: product.evaluate(x, check_domain=False)
                    t0 = time.perf_counter()
                    val = domain.integrate(
                        integrand, method=method, n_points=int(n_points), vectorized=bool(vectorized)
                    )
                    t1 = time.perf_counter()
                    records.append(
                        {
                            "function_index": int(i),
                            "method": str(method),
                            "n_points": int(n_points),
                            "vectorized": bool(vectorized),
                            "run": int(rep),
                            "time_s": float(t1 - t0),
                            "result": float(val),
                        }
                    )

    df = pd.DataFrame(records)
    summary = df.groupby(["method", "n_points"]).time_s.agg(["mean", "std", "count"]).reset_index()

    csv_path = save_prefix + "_flat.csv"
    summary_path = save_prefix + "_summary.csv"
    pkl_path = save_prefix + "_flat.pkl"

    df.to_csv(csv_path, index=False)
    summary.to_csv(summary_path, index=False)
    df.to_pickle(pkl_path)

    return df, summary


def plot_time_vs_npoints(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 4))
    sns.lineplot(data=df, x="n_points", y="time_s", hue="method", estimator="mean")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n_points")
    plt.ylabel("time (s)")
    plt.title("Integration time vs n_points")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
