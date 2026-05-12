"""Benchmark exact spherical-cap averages against Monte Carlo cap averages.

Run from the pygeoinf repository root::

    conda activate inferences3
    python work/sphere_cap_monte_carlo_benchmark.py

The benchmark treats the exact spherical-harmonic cap average as ground truth
and compares the retained Monte Carlo cap sampler across increasing sample
counts. It writes a CSV table and two figures under ``work/figures``.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf.symmetric_space.sphere import Sobolev

from sphere_dli_example import (
    DEFAULT_TARGET_LATLON,
    ORDER,
    SCALE,
    _sample_cap_points,
)


DEFAULT_N_CAPS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def _parse_n_caps(value: str) -> tuple[int, ...]:
    n_caps = tuple(
        int(item.strip()) for item in value.split(",") if item.strip()
    )
    if not n_caps:
        raise ValueError("At least one n_cap value is required.")
    if any(n_cap <= 0 for n_cap in n_caps):
        raise ValueError("All n_cap values must be positive.")
    return n_caps


def _exact_cap_rows(
    space: Sobolev,
    target_latlon: list[tuple[float, float]],
    cap_radius_rad: float,
) -> np.ndarray:
    rows = [
        space.geodesic_ball_average(
            centre,
            space.radius * cap_radius_rad,
        ).components
        for centre in target_latlon
    ]
    return np.vstack(rows)


def _monte_carlo_cap_rows(
    space: Sobolev,
    target_latlon: list[tuple[float, float]],
    cap_radius_rad: float,
    n_cap: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for centre in target_latlon:
        sample_points = _sample_cap_points(centre, cap_radius_rad, n_cap, rng)
        rows.append(
            np.mean(
                np.stack(
                    [space.dirac(point).components for point in sample_points]
                ),
                axis=0,
            )
        )
    return np.vstack(rows)


def _smooth_probe_components(
    space: Sobolev,
    n_fields: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    degrees = np.array(
        [space.integer_to_index(index)[0] for index in range(space.dim)]
    )
    laplacian = degrees * (degrees + 1.0) / space.radius**2
    spectral_decay = (1.0 + space.scale**2 * laplacian) ** (-space.order)
    components = (
        rng.normal(size=(space.dim, n_fields)) * spectral_decay[:, None]
    )
    norms = np.linalg.norm(components, axis=0, keepdims=True)
    return components / np.maximum(norms, 1e-15)


def _summarize(records: list[dict[str, float]]) -> list[dict[str, float]]:
    n_caps = sorted({int(record["n_cap"]) for record in records})
    summary = []
    for n_cap in n_caps:
        group = [record for record in records if int(record["n_cap"]) == n_cap]
        row: dict[str, float] = {"n_cap": float(n_cap)}
        for key in (
            "component_relative_l2",
            "field_relative_rmse",
            "elapsed_seconds",
            "speedup_vs_exact",
        ):
            values = np.array([record[key] for record in group], dtype=float)
            row[f"{key}_median"] = float(np.median(values))
            row[f"{key}_p10"] = float(np.percentile(values, 10.0))
            row[f"{key}_p90"] = float(np.percentile(values, 90.0))
        summary.append(row)
    return summary


def _write_records_csv(path: Path, records: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def _write_summary_csv(path: Path, summary: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary[0].keys())
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary)


def _plot_accuracy(
    path: Path,
    summary: list[dict[str, float]],
    title_suffix: str,
) -> None:
    n_caps = np.array([row["n_cap"] for row in summary], dtype=float)
    component_median = np.array(
        [row["component_relative_l2_median"] for row in summary], dtype=float
    )
    component_p10 = np.array(
        [row["component_relative_l2_p10"] for row in summary], dtype=float
    )
    component_p90 = np.array(
        [row["component_relative_l2_p90"] for row in summary], dtype=float
    )
    field_median = np.array(
        [row["field_relative_rmse_median"] for row in summary], dtype=float
    )
    field_p10 = np.array(
        [row["field_relative_rmse_p10"] for row in summary], dtype=float
    )
    field_p90 = np.array(
        [row["field_relative_rmse_p90"] for row in summary], dtype=float
    )

    reference = field_median[0] * np.sqrt(n_caps[0] / n_caps)

    fig, ax = plt.subplots(figsize=(7.1, 4.2))
    ax.fill_between(
        n_caps, component_p10, component_p90, color="tab:blue", alpha=0.16
    )
    ax.plot(
        n_caps,
        component_median,
        marker="o",
        linewidth=1.8,
        color="tab:blue",
        label="component relative L2",
    )
    ax.fill_between(
        n_caps, field_p10, field_p90, color="tab:orange", alpha=0.18
    )
    ax.plot(
        n_caps,
        field_median,
        marker="s",
        linewidth=1.8,
        color="tab:orange",
        label="field-output relative RMSE",
    )
    ax.plot(
        n_caps,
        reference,
        linestyle="--",
        linewidth=1.2,
        color="0.25",
        label="N^-1/2 reference",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Monte Carlo samples per cap")
    ax.set_ylabel("relative error vs exact cap average")
    ax.set_title(f"Monte Carlo cap-average convergence ({title_suffix})")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.45)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_cost(
    path: Path,
    summary: list[dict[str, float]],
    exact_elapsed_seconds: float,
    title_suffix: str,
) -> None:
    n_caps = np.array([row["n_cap"] for row in summary], dtype=float)
    time_median = np.array(
        [row["elapsed_seconds_median"] for row in summary], dtype=float
    )
    time_p10 = np.array(
        [row["elapsed_seconds_p10"] for row in summary], dtype=float
    )
    time_p90 = np.array(
        [row["elapsed_seconds_p90"] for row in summary], dtype=float
    )
    speedup_median = np.array(
        [row["speedup_vs_exact_median"] for row in summary], dtype=float
    )

    fig, ax = plt.subplots(figsize=(7.1, 4.2))
    ax.fill_between(n_caps, time_p10, time_p90, color="tab:green", alpha=0.18)
    ax.plot(
        n_caps,
        time_median,
        marker="o",
        linewidth=1.8,
        color="tab:green",
        label="Monte Carlo construction",
    )
    ax.axhline(
        exact_elapsed_seconds,
        color="tab:red",
        linestyle="--",
        linewidth=1.6,
        label="exact construction",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Monte Carlo samples per cap")
    ax.set_ylabel("seconds for all target caps")
    ax.set_title(f"Cap-average construction cost ({title_suffix})")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.45)

    ax_speed = ax.twinx()
    ax_speed.plot(
        n_caps,
        speedup_median,
        marker="^",
        linewidth=1.4,
        color="tab:purple",
        alpha=0.8,
        label="MC / exact time",
    )
    ax_speed.set_yscale("log")
    ax_speed.set_ylabel("Monte Carlo time / exact time")

    lines, labels = ax.get_legend_handles_labels()
    speed_lines, speed_labels = ax_speed.get_legend_handles_labels()
    ax.legend(
        lines + speed_lines,
        labels + speed_labels,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_benchmark(args: argparse.Namespace) -> dict[str, Path | float | int]:
    target_latlon = list(DEFAULT_TARGET_LATLON[: args.n_targets])
    space = Sobolev(args.lmax, ORDER, SCALE, radius=1.0, grid="DH")
    probe_components = _smooth_probe_components(
        space, args.n_fields, args.seed + 1000
    )

    exact_times = []
    exact_rows = None
    for _ in range(args.exact_repeats):
        start = time.perf_counter()
        exact_rows = _exact_cap_rows(space, target_latlon, args.cap_radius_rad)
        exact_times.append(time.perf_counter() - start)
    assert exact_rows is not None
    exact_elapsed_seconds = float(np.median(exact_times))
    exact_values = exact_rows @ probe_components
    exact_component_norm = np.linalg.norm(exact_rows, axis=1)
    exact_value_scale = np.sqrt(np.mean(exact_values**2))

    records: list[dict[str, float]] = []
    for n_cap in args.n_caps:
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            start = time.perf_counter()
            monte_carlo_rows = _monte_carlo_cap_rows(
                space,
                target_latlon,
                args.cap_radius_rad,
                n_cap,
                seed,
            )
            elapsed_seconds = time.perf_counter() - start

            row_difference = monte_carlo_rows - exact_rows
            component_relative_l2 = float(
                np.mean(
                    np.linalg.norm(row_difference, axis=1)
                    / np.maximum(exact_component_norm, 1e-15)
                )
            )
            value_difference = row_difference @ probe_components
            field_relative_rmse = float(
                np.sqrt(np.mean(value_difference**2))
                / max(exact_value_scale, 1e-15)
            )
            records.append(
                {
                    "lmax": float(args.lmax),
                    "dim": float(space.dim),
                    "n_targets": float(len(target_latlon)),
                    "cap_radius_rad": float(args.cap_radius_rad),
                    "n_cap": float(n_cap),
                    "repeat": float(repeat),
                    "seed": float(seed),
                    "component_relative_l2": component_relative_l2,
                    "field_relative_rmse": field_relative_rmse,
                    "elapsed_seconds": float(elapsed_seconds),
                    "exact_elapsed_seconds": exact_elapsed_seconds,
                    "speedup_vs_exact": float(
                        elapsed_seconds / exact_elapsed_seconds
                    ),
                }
            )

    summary = _summarize(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_csv = (
        args.output_dir / "sphere_cap_monte_carlo_benchmark_records.csv"
    )
    summary_csv = (
        args.output_dir / "sphere_cap_monte_carlo_benchmark_summary.csv"
    )
    accuracy_png = args.output_dir / "sphere_cap_monte_carlo_accuracy.png"
    cost_png = args.output_dir / "sphere_cap_monte_carlo_cost.png"

    _write_records_csv(records_csv, records)
    _write_summary_csv(summary_csv, summary)
    title_suffix = f"lmax={args.lmax}, {len(target_latlon)} caps"
    _plot_accuracy(accuracy_png, summary, title_suffix)
    _plot_cost(cost_png, summary, exact_elapsed_seconds, title_suffix)

    return {
        "lmax": args.lmax,
        "dim": space.dim,
        "n_targets": len(target_latlon),
        "exact_elapsed_seconds": exact_elapsed_seconds,
        "records_csv": records_csv,
        "summary_csv": summary_csv,
        "accuracy_png": accuracy_png,
        "cost_png": cost_png,
        "largest_n_cap": int(args.n_caps[-1]),
        "largest_n_cap_field_relative_rmse_median": summary[-1][
            "field_relative_rmse_median"
        ],
        "largest_n_cap_component_relative_l2_median": summary[-1][
            "component_relative_l2_median"
        ],
        "largest_n_cap_elapsed_seconds_median": summary[-1][
            "elapsed_seconds_median"
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmax", type=int, default=32)
    parser.add_argument("--n-targets", type=int, default=6)
    parser.add_argument("--cap-radius-rad", type=float, default=0.15)
    parser.add_argument("--n-caps", type=_parse_n_caps, default=DEFAULT_N_CAPS)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--exact-repeats", type=int, default=8)
    parser.add_argument("--n-fields", type=int, default=24)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "figures",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_benchmark(args)

    print("Sphere cap Monte Carlo benchmark complete")
    print(f"  lmax: {result['lmax']}  dim: {result['dim']}")
    print(f"  targets: {result['n_targets']}")
    print(
        f"  exact construction median: "
        f"{result['exact_elapsed_seconds']:.6f} s"
    )
    print(
        f"  n_cap={result['largest_n_cap']} median field RMSE: "
        f"{result['largest_n_cap_field_relative_rmse_median']:.6e}"
    )
    print(
        f"  n_cap={result['largest_n_cap']} median component error: "
        f"{result['largest_n_cap_component_relative_l2_median']:.6e}"
    )
    print(
        f"  n_cap={result['largest_n_cap']} median MC time: "
        f"{result['largest_n_cap_elapsed_seconds_median']:.6f} s"
    )
    print(f"  records: {result['records_csv']}")
    print(f"  summary: {result['summary_csv']}")
    print(f"  accuracy figure: {result['accuracy_png']}")
    print(f"  cost figure: {result['cost_png']}")


if __name__ == "__main__":
    main()
