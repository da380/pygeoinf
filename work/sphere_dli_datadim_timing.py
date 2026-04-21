"""Sphere DLI — Data-dimension timing study.

Sweeps over data-space dimension by varying n_sources × n_receivers with
the model space fixed at lmax=64, isolating the effect of data dimension on
solving time.

Data dimensions tested:
    dim=10   (n_sources=2,  n_receivers=5)
    dim=50   (n_sources=5,  n_receivers=10)   ← current default
    dim=100  (n_sources=10, n_receivers=10)
    dim=500  (n_sources=25, n_receivers=20)
    dim=1000 (n_sources=50, n_receivers=20)

Results are saved to disk after each completed dimension so the run can be
killed early without losing data.

Outputs (written to ``pygeoinf/work/figures/``):
    datadim_timing.png     — wall-clock time vs data dimension (log-log)
    datadim_results.npz    — raw arrays for later analysis
    datadim_partial.npz    — incremental save (overwritten after each dim)

Run::

    conda activate inferences2   # or inferences3 locally
    cd <workspace-root>
    python -u pygeoinf/work/sphere_dli_datadim_timing.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Allow running from workspace root as  python pygeoinf/work/sphere_dli_datadim_timing.py
_WORK_DIR = Path(__file__).parent
if str(_WORK_DIR) not in sys.path:
    sys.path.insert(0, str(_WORK_DIR))

from sphere_dli_example import (
    DEFAULT_TARGET_LATLON,
    SIGMA_NOISE,
    ORDER,
    SCALE,
    PRIOR_SCALE,
    build_cap_property_operator,
    build_forward_operator,
    build_model_space,
    generate_synthetic_data,
    solve_dli,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# (n_sources, n_receivers) → data_dim = n_sources * n_receivers
DATA_DIM_CONFIGS = [
    (2,  5),   # dim=10
    (5,  10),  # dim=50   (baseline)
    (10, 10),  # dim=100
    (25, 20),  # dim=500
    (50, 20),  # dim=1000
]

LMAX = 64          # fixed model-space resolution
N_TARGET = 6       # number of cap properties
N_CAP = 40         # cap sampling points
SEED = 42
MAX_ITER = 200
TOL = 1e-4

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

TARGET_LATLON = DEFAULT_TARGET_LATLON[:N_TARGET]

PARTIAL_SAVE = FIG_DIR / "datadim_partial.npz"


# ---------------------------------------------------------------------------
# Per-dimension runner
# ---------------------------------------------------------------------------

def run_at_datadim(
    model_space,
    property_operator,
    truth_model,
    n_sources: int,
    n_receivers: int,
) -> dict:
    """Run the DLI pipeline for one (n_sources, n_receivers) configuration.

    The model space, property operator, and truth model are shared across all
    runs (built once by the caller) so the measured times reflect only the
    cost attributable to the data dimension.
    """
    data_dim = n_sources * n_receivers
    print(f"\n{'='*60}")
    print(f"  data_dim = {data_dim:5d}   (n_src={n_sources}, n_rec={n_receivers})")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Build forward operator (scales with n_paths × model_dim)
    t1 = time.time()
    forward_operator, _ = build_forward_operator(
        model_space,
        n_sources=n_sources,
        n_receivers=n_receivers,
        seed=SEED,
    )
    build_time = time.time() - t1
    print(f"  Forward operator built in {build_time:.2f}s  (dim={forward_operator.codomain.dim})")
    sys.stdout.flush()

    # Generate synthetic data (uses same truth model → comparable bounds)
    t2 = time.time()
    _, data_vector = generate_synthetic_data(
        model_space,
        forward_operator,
        sigma_noise=SIGMA_NOISE,
        seed=SEED,
    )
    data_gen_time = time.time() - t2
    print(f"  Data generated in {data_gen_time:.2f}s")
    sys.stdout.flush()

    # DLI solve — this is the bottleneck
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
    solve_time = time.time() - t3
    total_time = time.time() - t1
    print(f"  DLI solve: {solve_time:.2f}s  (total incl. build: {total_time:.2f}s)")
    sys.stdout.flush()

    for i, (lo, hi, tv) in enumerate(
        zip(bounds["lower"], bounds["upper"], bounds["true_values"])
    ):
        print(f"    Cap {i+1}: [{lo:.4f}, {hi:.4f}]  true={tv:.4f}")
    sys.stdout.flush()

    return {
        "data_dim":     data_dim,
        "n_sources":    n_sources,
        "n_receivers":  n_receivers,
        "build_time":   build_time,
        "data_gen_time": data_gen_time,
        "solve_time":   solve_time,
        "total_time":   total_time,
        "lower":        bounds["lower"],
        "upper":        bounds["upper"],
        "true_values":  bounds["true_values"],
        "prior_lower":  bounds["prior_lower"],
        "prior_upper":  bounds["prior_upper"],
    }


def save_partial(results: list[dict]) -> None:
    """Save completed results to disk (overwrite)."""
    if not results:
        return
    np.savez(
        PARTIAL_SAVE,
        data_dims      = np.array([r["data_dim"]    for r in results]),
        n_sources      = np.array([r["n_sources"]   for r in results]),
        n_receivers    = np.array([r["n_receivers"]  for r in results]),
        build_times    = np.array([r["build_time"]   for r in results]),
        solve_times    = np.array([r["solve_time"]   for r in results]),
        total_times    = np.array([r["total_time"]   for r in results]),
        lower          = np.array([r["lower"]        for r in results]),
        upper          = np.array([r["upper"]        for r in results]),
        true_values    = np.array([r["true_values"]  for r in results]),
        prior_lower    = np.array([r["prior_lower"]  for r in results]),
        prior_upper    = np.array([r["prior_upper"]  for r in results]),
        lmax           = LMAX,
    )
    print(f"  [partial save → {PARTIAL_SAVE}]")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: list[dict]) -> None:
    """Produce timing and bound-width plots from the completed results."""
    data_dims   = np.array([r["data_dim"]   for r in results])
    build_times = np.array([r["build_time"]  for r in results])
    solve_times = np.array([r["solve_time"]  for r in results])
    total_times = np.array([r["total_time"]  for r in results])

    # ---- Figure 1: timing vs data dimension --------------------------------
    fig1, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.loglog(data_dims, solve_times, "o-", color="tab:blue",  label="Solve (DLI bundle)")
    ax.loglog(data_dims, build_times, "s--", color="tab:orange", label="Build fwd operator")
    ax.loglog(data_dims, total_times, "^:",  color="tab:grey",  label="Total")
    # Reference power-law slopes
    _xlim = np.array([data_dims[0] * 0.8, data_dims[-1] * 1.3])
    for exp, ls, label in [(1, ":", r"$\propto n$"), (2, "--", r"$\propto n^2$"),
                            (1.5, "-.", r"$\propto n^{1.5}$")]:
        _y = solve_times[0] * (_xlim / data_dims[0]) ** exp
        ax.loglog(_xlim, _y, color="lightgrey", linestyle=ls, linewidth=1, label=label)
    ax.set_xlabel("Data-space dimension  $n$")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(f"DLI timing vs data dimension  (lmax={LMAX})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # ---- Figure 1b: bound width vs data dimension --------------------------
    lower_arr = np.array([r["lower"] for r in results])  # (n_dims, n_caps)
    upper_arr = np.array([r["upper"] for r in results])
    widths = upper_arr - lower_arr  # (n_dims, n_caps)

    ax2 = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, widths.shape[1]))
    for cap_idx in range(widths.shape[1]):
        ax2.semilogx(
            data_dims, widths[:, cap_idx],
            "o-", color=colors[cap_idx], label=f"Cap {cap_idx+1}",
        )
    ax2.set_xlabel("Data-space dimension  $n$")
    ax2.set_ylabel("Bound width  upper − lower  (km/s)")
    ax2.set_title("DLI bound width vs data dimension")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig1.suptitle(
        f"Data-dimension scaling study  "
        f"(lmax={LMAX}, {N_TARGET} caps, seed={SEED})",
        fontsize=11,
    )
    fig1.tight_layout()
    out1 = FIG_DIR / "datadim_timing.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out1}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Sphere DLI — data-dimension timing study")
    print(f"Data-dim configs: {[ns*nr for (ns, nr) in DATA_DIM_CONFIGS]}")
    print(f"Model space: lmax={LMAX}  (fixed)")
    print(f"N_target={N_TARGET}, N_cap={N_CAP}, seed={SEED}")
    print(f"max_iter={MAX_ITER}, tol={TOL}")
    sys.stdout.flush()

    # Build shared model space and property operator once
    t0 = time.time()
    print("\nBuilding model space...")
    sys.stdout.flush()
    model_space = build_model_space(min_degree=LMAX)
    print(f"  lmax={model_space.lmax}, dim={model_space.dim}  ({time.time()-t0:.1f}s)")
    sys.stdout.flush()

    print("Building property operator (shared)...")
    sys.stdout.flush()
    t_prop = time.time()
    property_operator = build_cap_property_operator(
        model_space, TARGET_LATLON, n_cap=N_CAP, seed=SEED,
    )
    print(f"  {N_TARGET} caps built in {time.time()-t_prop:.1f}s")
    sys.stdout.flush()

    # Build a reference forward operator and truth model at baseline dim
    print("Building reference truth model at baseline dim=50...")
    sys.stdout.flush()
    ref_fwd, _ = build_forward_operator(model_space, n_sources=5, n_receivers=10, seed=SEED)
    truth_model, _ = generate_synthetic_data(model_space, ref_fwd, sigma_noise=SIGMA_NOISE, seed=SEED)
    print(f"  Truth model ready  ({time.time()-t0:.1f}s total setup)")
    sys.stdout.flush()

    # Sweep over data dimensions
    results = []
    for n_src, n_rec in DATA_DIM_CONFIGS:
        result = run_at_datadim(model_space, property_operator, truth_model, n_src, n_rec)
        results.append(result)
        save_partial(results)

    # Final save
    final_save = FIG_DIR / "datadim_results.npz"
    save_partial(results)
    import shutil
    shutil.copy(PARTIAL_SAVE, final_save)
    print(f"\nFinal results saved: {final_save}")
    sys.stdout.flush()

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  {'data_dim':>10}  {'build_time':>12}  {'solve_time':>12}  {'total_time':>12}")
    for r in results:
        print(f"  {r['data_dim']:>10}  {r['build_time']:>11.1f}s  {r['solve_time']:>11.1f}s  {r['total_time']:>11.1f}s")
    sys.stdout.flush()

    # Plots
    plot_results(results)


if __name__ == "__main__":
    main()
