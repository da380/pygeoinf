"""Sphere DLI — Discretization Convergence Study.

Sweeps over spherical-harmonic truncation degrees lmax ∈ {8, 16, 32, 64, 128, 256}
and records the DLI admissible bounds for each cap at each discretization level.

The same sources, receivers, cap centres, and random seed are used across all
runs so that differences in the bounds are attributable solely to the change in
model-space resolution.

Expectation:
    - Coarser spaces (low lmax) tend to produce **narrower** but potentially
      spurious bounds (fewer degrees of freedom to explain the data).
    - Bounds should converge as lmax → 128 (the natural energy-truncation
      threshold for these prior parameters) and remain stable at lmax = 256.

Outputs (written to ``pygeoinf/work/figures/``):
    convergence_bounds.png   — per-cap bound-vs-lmax panel plot
    convergence_width.png    — bound width (upper − lower) vs lmax per cap
    convergence_results.npz  — raw arrays for later analysis

Run::

    conda activate inferences2   # or inferences3 locally
    cd <workspace-root>
    python pygeoinf/work/sphere_dli_convergence.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf.symmetric_space.sphere import Sobolev

from sphere_dli_example import (
    DEFAULT_TARGET_LATLON,
    N_SOURCES,
    N_RECEIVERS,
    SIGMA_NOISE,
    ORDER,
    SCALE,
    PRIOR_SCALE,
    build_cap_property_operator,
    build_forward_operator,
    generate_synthetic_data,
    solve_dli,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LMAX_VALUES = [8, 16, 32, 64, 128, 256]
N_TARGET = 6
N_CAP = 40
SEED = 42
MAX_ITER = 200
TOL = 1e-4

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

TARGET_LATLON = DEFAULT_TARGET_LATLON[:N_TARGET]


# ---------------------------------------------------------------------------
# Per-lmax runner
# ---------------------------------------------------------------------------

def run_at_lmax(lmax: int) -> dict:
    """Run the full DLI pipeline with a model space truncated at *lmax*."""
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  lmax = {lmax:4d}   dim = {(lmax+1)**2:7d}")
    print(f"{'='*60}")

    # Build model space at this resolution
    model_space = Sobolev.from_heat_kernel_prior(
        PRIOR_SCALE, ORDER, SCALE,
        max_degree=lmax,
        min_degree=lmax,
        power_of_two=False,
    )
    print(f"  Model space: lmax={model_space.lmax}, dim={model_space.dim}")

    t1 = time.time()
    property_operator = build_cap_property_operator(
        model_space, TARGET_LATLON, n_cap=N_CAP, seed=SEED,
    )
    forward_operator, _ = build_forward_operator(
        model_space, n_sources=N_SOURCES, n_receivers=N_RECEIVERS, seed=SEED,
    )
    print(f"  Operators built in {time.time()-t1:.1f}s")

    t2 = time.time()
    truth_model, data_vector = generate_synthetic_data(
        model_space, forward_operator, sigma_noise=SIGMA_NOISE, seed=SEED,
    )
    print(f"  Data generated in {time.time()-t2:.1f}s")

    t3 = time.time()
    bounds = solve_dli(
        model_space,
        forward_operator,
        property_operator,
        truth_model,
        data_vector,
        sigma_noise=SIGMA_NOISE,
        max_iter=MAX_ITER,
        tol=TOL,
    )
    elapsed = time.time() - t3
    total = time.time() - t_start
    print(f"  DLI solve: {elapsed:.1f}s  (total: {total:.1f}s)")

    for i, (lo, hi, tv) in enumerate(
        zip(bounds["lower"], bounds["upper"], bounds["true_values"])
    ):
        print(f"    Cap {i+1}: [{lo:.4f}, {hi:.4f}]  true={tv:.4f}")

    return {
        "lmax": lmax,
        "dim": model_space.dim,
        "lower": bounds["lower"],
        "upper": bounds["upper"],
        "true_values": bounds["true_values"],
        "prior_lower": bounds["prior_lower"],
        "prior_upper": bounds["prior_upper"],
        "elapsed": total,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_convergence(results: list[dict]) -> None:
    """Plot DLI bounds and widths as a function of lmax."""
    lmax_arr = np.array([r["lmax"] for r in results])
    n_caps = len(results[0]["lower"])
    cap_labels = [f"Cap {i+1}" for i in range(n_caps)]

    lower_arr = np.array([r["lower"] for r in results])   # (n_lmax, n_caps)
    upper_arr = np.array([r["upper"] for r in results])
    true_arr  = np.array([r["true_values"] for r in results])
    prior_lo  = np.array([r["prior_lower"] for r in results])
    prior_hi  = np.array([r["prior_upper"] for r in results])

    # --- Figure 1: bounds per cap -------------------------------------------
    ncols = 3
    nrows = (n_caps + ncols - 1) // ncols
    fig1, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                               sharey=False)
    axes_flat = axes.flat

    for cap_idx in range(n_caps):
        ax = axes_flat[cap_idx]
        ax.fill_between(
            lmax_arr,
            prior_lo[:, cap_idx],
            prior_hi[:, cap_idx],
            alpha=0.15, color="grey", label="Prior bounds",
        )
        ax.fill_between(
            lmax_arr,
            lower_arr[:, cap_idx],
            upper_arr[:, cap_idx],
            alpha=0.4, color="tab:blue", label="DLI interval",
        )
        ax.plot(lmax_arr, lower_arr[:, cap_idx], "b-o", markersize=4, linewidth=1)
        ax.plot(lmax_arr, upper_arr[:, cap_idx], "b-o", markersize=4, linewidth=1)
        ax.plot(lmax_arr, true_arr[:, cap_idx], "r--", linewidth=1.5, label="True value")
        ax.axhline(true_arr[-1, cap_idx], color="r", linewidth=0.6, alpha=0.4)
        ax.set_xscale("log", base=2)
        ax.set_xticks(lmax_arr)
        ax.set_xticklabels([str(l) for l in lmax_arr], fontsize=8)
        ax.set_xlabel("lmax")
        ax.set_ylabel("Phase-vel. perturbation (km/s)")
        ax.set_title(cap_labels[cap_idx])
        if cap_idx == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for ax in list(axes_flat)[n_caps:]:
        ax.set_visible(False)

    fig1.suptitle(
        f"DLI bound convergence vs SH truncation degree\n"
        f"(N_sources={N_SOURCES}, N_receivers={N_RECEIVERS}, seed={SEED})",
        fontsize=11,
    )
    fig1.tight_layout()
    fig1.savefig(FIG_DIR / "convergence_bounds.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {FIG_DIR / 'convergence_bounds.png'}")

    # --- Figure 2: bound width (upper − lower) per cap ----------------------
    widths = upper_arr - lower_arr
    prior_widths = prior_hi - prior_lo

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_caps))
    for cap_idx in range(n_caps):
        ax2.plot(
            lmax_arr, widths[:, cap_idx],
            "-o", color=colors[cap_idx], markersize=5,
            label=cap_labels[cap_idx],
        )
        ax2.plot(
            lmax_arr, prior_widths[:, cap_idx],
            "--", color=colors[cap_idx], alpha=0.4,
        )
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(lmax_arr)
    ax2.set_xticklabels([str(l) for l in lmax_arr])
    ax2.set_xlabel("lmax (SH truncation degree)")
    ax2.set_ylabel("Bound width  upper − lower  (km/s)")
    ax2.set_title(
        "DLI bound width vs discretization  (solid=DLI, dashed=prior)"
    )
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "convergence_width.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {FIG_DIR / 'convergence_width.png'}")

    # --- Figure 3: timing ---------------------------------------------------
    elapsed = np.array([r["elapsed"] for r in results])
    dims = np.array([r["dim"] for r in results])
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.loglog(dims, elapsed, "ko-", markersize=6)
    for r in results:
        ax3.annotate(
            f"lmax={r['lmax']}",
            (r["dim"], r["elapsed"]),
            xytext=(5, 3), textcoords="offset points", fontsize=8,
        )
    ax3.set_xlabel("Model space dimension  (lmax+1)²")
    ax3.set_ylabel("Total wall-clock time (s)")
    ax3.set_title("Runtime vs model space dimension")
    ax3.grid(True, which="both", alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / "convergence_timing.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {FIG_DIR / 'convergence_timing.png'}")

    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Sphere DLI convergence sweep")
    print(f"lmax values: {LMAX_VALUES}")
    print(f"N_sources={N_SOURCES}, N_receivers={N_RECEIVERS}, n_target={N_TARGET}")
    print(f"seed={SEED}, max_iter={MAX_ITER}, tol={TOL}")

    results = []
    for lmax in LMAX_VALUES:
        r = run_at_lmax(lmax)
        results.append(r)

    # Save raw results
    out_path = FIG_DIR / "convergence_results.npz"
    np.savez(
        out_path,
        lmax_values=np.array([r["lmax"] for r in results]),
        dims=np.array([r["dim"] for r in results]),
        lower=np.array([r["lower"] for r in results]),
        upper=np.array([r["upper"] for r in results]),
        true_values=np.array([r["true_values"] for r in results]),
        prior_lower=np.array([r["prior_lower"] for r in results]),
        prior_upper=np.array([r["prior_upper"] for r in results]),
        elapsed=np.array([r["elapsed"] for r in results]),
    )
    print(f"\nRaw results saved: {out_path}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'lmax':>6}  {'dim':>8}  {'time':>8}  {'cap1 width':>12}  {'cap5 width':>12}")
    for r in results:
        w1 = r["upper"][0] - r["lower"][0]
        w5 = r["upper"][4] - r["lower"][4]
        print(f"{r['lmax']:6d}  {r['dim']:8d}  {r['elapsed']:7.1f}s  {w1:12.4f}  {w5:12.4f}")

    plot_convergence(results)
