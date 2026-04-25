"""Sphere DLI — Solver comparison timing study.

Sweeps over data-space dimension {10, 50, 100, 500} with model space fixed at
lmax=64, comparing the wall-clock time of four optimization methods:

    ProximalBundle  — proximal bundle method (OSQP QP master)
    LevelBundle     — level bundle method (OSQP QP master)
    SmoothedLBFGS   — Moreau-Yosida smoothing + L-BFGS-B continuation
    ChambollePock   — first-order primal-dual (Chambolle & Pock 2011)

The figure is rebuilt and saved after every single (solver, dim) data point
so it can be pulled from europa to monitor progress.

Outputs (written to ``pygeoinf/work/figures/``):
    solver_comparison.png    — timing + bound-width figure (updated live)
    solver_partial.npz       — incremental results (overwritten each point)
    solver_results.npz       — final complete results

Run::

    conda activate inferences2   # or inferences3 locally
    cd <workspace-root>
    python -u pygeoinf/work/sphere_dli_solver_comparison.py
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

_WORK_DIR = Path(__file__).parent
if str(_WORK_DIR) not in sys.path:
    sys.path.insert(0, str(_WORK_DIR))

from sphere_dli_example import (
    DEFAULT_TARGET_LATLON,
    SIGMA_NOISE,
    PRIOR_SCALE,
    build_cap_property_operator,
    build_forward_operator,
    build_model_space,
    generate_synthetic_data,
)

from pygeoinf.backus_gilbert import DualMasterCostFunction
from pygeoinf.convex_analysis import BallSupportFunction
from pygeoinf.convex_optimisation import (
    ProximalBundleMethod,
    SmoothedLBFGSSolver,
    ChambollePockSolver,
    PrimalKKTSolver,
    best_available_qp_solver,
    solve_support_values,
    solve_primal_feasibility,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# (n_sources, n_receivers) → data_dim = n_sources * n_receivers
DATA_DIM_CONFIGS = [
    (2,  5),   # dim=10
    (5,  10),  # dim=50
    (10, 10),  # dim=100
    (20, 10),  # dim=200
]

SOLVER_NAMES = ["ProximalBundle", "SmoothedLBFGS", "ChambollePock", "PrimalKKT"]

# Configs to skip — not used in this run (only 4 fast solvers)
SKIP_CONFIGS: frozenset = frozenset()

# Estimated solve times for DNF configs — not used in this run
DNF_ESTIMATED_TIMES: dict = {}

SOLVER_COLORS = {
    "ProximalBundle": "tab:blue",
    "SmoothedLBFGS":  "tab:green",
    "ChambollePock":  "tab:purple",
    "PrimalKKT":      "tab:red",
}
SOLVER_MARKERS = {
    "ProximalBundle": "o",
    "SmoothedLBFGS":  "^",
    "ChambollePock":  "D",
    "PrimalKKT":      "P",
}

LMAX = 64
N_TARGET = 6
N_CAP = 40
SEED = 42
MAX_ITER = 200
TOL = 1e-4

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

PARTIAL_SAVE = FIG_DIR / "solver_partial.npz"
FIGURE_PATH  = FIG_DIR / "solver_comparison.png"

TARGET_LATLON = DEFAULT_TARGET_LATLON[:N_TARGET]


def _format_duration(seconds: float) -> str:
    """Return a compact human-readable duration label for plot annotations."""
    hours = seconds / 3600.0
    if hours < 24.0:
        return f"~{hours:.0f}h"
    days = hours / 24.0
    if days < 10.0:
        return f"~{days:.1f}d"
    return f"~{days:.0f}d"


# ---------------------------------------------------------------------------
# DLI solve core (solver-agnostic)
# ---------------------------------------------------------------------------

def _build_dli_components(
    model_space,
    forward_operator,
    property_operator,
    truth_model,
    data_vector,
    *,
    sigma_noise: float = SIGMA_NOISE,
    prior_radius_multiplier: float = 3.0,
):
    """Build the shared DLI components (balls, cost, basis directions)."""
    data_space     = forward_operator.codomain
    property_space = property_operator.codomain

    prior_radius    = prior_radius_multiplier * model_space.norm(truth_model)
    model_ball      = BallSupportFunction(model_space, model_space.zero, prior_radius)

    data_ball_radius = 3.0 * sigma_noise * np.sqrt(data_space.dim)
    data_ball        = BallSupportFunction(data_space, data_space.zero, data_ball_radius)

    basis_dirs   = [property_space.basis_vector(i) for i in range(property_space.dim)]
    neg_basis    = [property_space.multiply(-1.0, q) for q in basis_dirs]

    observed = np.asarray(data_vector, dtype=float)

    cost = DualMasterCostFunction(
        data_space,
        property_space,
        model_space,
        forward_operator,
        property_operator,
        model_ball,
        data_ball,
        observed,
        basis_dirs[0],
    )

    true_values = np.asarray(property_operator(truth_model), dtype=float)
    prior_bounds = np.array([model_ball(property_operator.adjoint(q)) for q in basis_dirs])

    return cost, model_ball, data_ball, basis_dirs, neg_basis, true_values, prior_bounds


def solve_with_proximal_bundle(cost, basis_dirs, neg_basis, data_space):
    """Run ProximalBundleMethod via solve_support_values."""
    qp = best_available_qp_solver()
    solver = ProximalBundleMethod(
        cost, tolerance=TOL, max_iterations=MAX_ITER, qp_solver=qp,
    )
    lambda0 = data_space.zero
    upper_vals, _, _ = solve_support_values(cost, basis_dirs,  solver, lambda0, n_jobs=-1)
    lower_neg,  _, _ = solve_support_values(cost, neg_basis,   solver, lambda0, n_jobs=-1)
    return np.asarray(upper_vals), -np.asarray(lower_neg)


def solve_with_level_bundle(cost, basis_dirs, neg_basis, data_space):
    """Run LevelBundleMethod via solve_support_values.

    Uses alpha=0.5 (conservative level placement) to avoid pathological
    infeasibility cycles seen with the default alpha=0.1 at higher data dims.
    """
    qp = best_available_qp_solver()
    solver = LevelBundleMethod(
        cost, tolerance=TOL, max_iterations=MAX_ITER, qp_solver=qp, alpha=0.5,
    )
    lambda0 = data_space.zero
    upper_vals, _, _ = solve_support_values(cost, basis_dirs, solver, lambda0, n_jobs=-1)
    lower_neg,  _, _ = solve_support_values(cost, neg_basis,  solver, lambda0, n_jobs=-1)
    return np.asarray(upper_vals), -np.asarray(lower_neg)


def solve_with_smoothed_lbfgs(cost, basis_dirs, neg_basis, data_space):
    """Run SmoothedLBFGSSolver via solve_support_values."""
    solver = SmoothedLBFGSSolver(
        cost, epsilon0=1e-2, n_levels=5, tolerance=TOL, max_iter_per_level=MAX_ITER,
    )
    lambda0 = data_space.zero
    upper_vals, _, _ = solve_support_values(cost, basis_dirs, solver, lambda0, n_jobs=-1)
    lower_neg,  _, _ = solve_support_values(cost, neg_basis,  solver, lambda0, n_jobs=-1)
    return np.asarray(upper_vals), -np.asarray(lower_neg)


def solve_with_chambolle_pock(
    cost, basis_dirs, neg_basis, model_ball, data_ball, forward_operator, data_vector
):
    """Run ChambollePockSolver via solve_primal_feasibility."""
    cp_solver = ChambollePockSolver(
        model_ball,
        data_ball,
        forward_operator,
        np.asarray(data_vector, dtype=float),
        max_iterations=MAX_ITER * 10,  # CP needs more iterations (first-order)
        tolerance=TOL,
    )
    upper_vals = solve_primal_feasibility(cost, basis_dirs, cp_solver)
    lower_neg  = solve_primal_feasibility(cost, neg_basis,  cp_solver)
    return np.asarray(upper_vals), -np.asarray(lower_neg)


def solve_with_primal_kkt(
    cost, basis_dirs, neg_basis, model_ball, data_ball, forward_operator, data_vector
):
    """Run PrimalKKTSolver via direct KKT + Woodbury on each direction."""
    model_space = cost._model_space
    T = cost._T
    data_space = forward_operator.codomain

    d_tilde_vec = data_space.from_components(np.asarray(data_vector, dtype=float))
    kkt_solver = PrimalKKTSolver(
        model_ball,
        data_ball,
        forward_operator,
        d_tilde_vec,
    )

    def _solve_direction(qs):
        vals = []
        for q in qs:
            c = T.adjoint(q)
            result = kkt_solver.solve(c)
            vals.append(model_space.inner_product(c, result.m))
        return np.array(vals)

    upper_vals = _solve_direction(basis_dirs)
    lower_neg  = _solve_direction(neg_basis)
    return np.asarray(upper_vals), -np.asarray(lower_neg)


def run_one(
    solver_name: str,
    model_space,
    property_operator,
    truth_model,
    forward_operator,
    data_vector,
) -> dict:
    """Run a single (solver, dim) timing measurement."""
    data_dim = forward_operator.codomain.dim
    print(f"    [{solver_name}]  data_dim={data_dim}")
    sys.stdout.flush()

    (cost, model_ball, data_ball,
     basis_dirs, neg_basis,
     true_values, prior_bounds) = _build_dli_components(
        model_space, forward_operator, property_operator, truth_model, data_vector,
    )

    data_space = forward_operator.codomain
    t0 = time.time()

    if solver_name == "ProximalBundle":
        upper, lower = solve_with_proximal_bundle(cost, basis_dirs, neg_basis, data_space)
    elif solver_name == "SmoothedLBFGS":
        upper, lower = solve_with_smoothed_lbfgs(cost, basis_dirs, neg_basis, data_space)
    elif solver_name == "ChambollePock":
        upper, lower = solve_with_chambolle_pock(
            cost, basis_dirs, neg_basis, model_ball, data_ball,
            forward_operator, data_vector,
        )
    elif solver_name == "PrimalKKT":
        upper, lower = solve_with_primal_kkt(
            cost, basis_dirs, neg_basis, model_ball, data_ball,
            forward_operator, data_vector,
        )
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    elapsed = time.time() - t0
    print(f"      → {elapsed:.1f}s   bounds: [{lower[0]:.3f}, {upper[0]:.3f}]")
    sys.stdout.flush()

    return {
        "solver":       solver_name,
        "data_dim":     data_dim,
        "solve_time":   elapsed,
        "upper":        upper,
        "lower":        lower,
        "true_values":  true_values,
        "prior_lower":  -prior_bounds,
        "prior_upper":  prior_bounds,
    }


# ---------------------------------------------------------------------------
# Progressive figure
# ---------------------------------------------------------------------------

def plot_progress(results: list[dict]) -> None:
    """Rebuild and save the comparison figure from all completed results.

    Completed runs (solve_time > 0) are plotted normally.  DNF entries
    (solve_time == np.nan) are shown as upward-pointing open triangles at
    the estimated time value stored in ``DNF_ESTIMATED_TIMES``.
    """
    if not results:
        return

    all_dims     = sorted({r["data_dim"] for r in results})
    all_solvers  = SOLVER_NAMES  # maintain order

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Panel 1: solve time vs data dim -----------------------------------
    ax = axes[0]
    for sname in all_solvers:
        rows = [r for r in results if r["solver"] == sname]
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: r["data_dim"])
        # Split into finished and DNF entries
        finished = [r for r in rows_sorted if not np.isnan(r["solve_time"])]
        dnf      = [r for r in rows_sorted if  np.isnan(r["solve_time"])]

        xs = [r["data_dim"]   for r in finished]
        ys = [r["solve_time"] for r in finished]
        if xs:
            ax.loglog(
                xs, ys,
                color=SOLVER_COLORS[sname],
                marker=SOLVER_MARKERS[sname],
                linestyle="-",
                markersize=8,
                linewidth=1.5,
                label=sname,
            )
            ax.annotate(
                f"{ys[-1]:.0f}s",
                (xs[-1], ys[-1]),
                textcoords="offset points",
                xytext=(6, 2),
                fontsize=7,
                color=SOLVER_COLORS[sname],
            )
        else:
            ax.plot([], [], color=SOLVER_COLORS[sname], marker=SOLVER_MARKERS[sname],
                    linestyle="-", markersize=8, linewidth=1.5, label=sname)

        # DNF markers — open upward-triangle at estimated time, dashed connector
        for r in dnf:
            key = (sname, r["data_dim"])
            est_time = DNF_ESTIMATED_TIMES.get(key, None)
            if est_time is None:
                continue
            ax.plot(
                r["data_dim"], est_time,
                marker="^", markersize=12,
                markeredgecolor=SOLVER_COLORS[sname], markerfacecolor="none",
                markeredgewidth=2.0, linestyle="none", zorder=5,
            )
            if finished:
                ax.loglog(
                    [finished[-1]["data_dim"], r["data_dim"]],
                    [finished[-1]["solve_time"], est_time],
                    color=SOLVER_COLORS[sname], linestyle="--",
                    linewidth=1.0, alpha=0.5,
                )
            ax.annotate(
                f"DNF\n(est. {_format_duration(est_time)})",
                (r["data_dim"], est_time),
                textcoords="offset points", xytext=(6, -14),
                fontsize=7, color=SOLVER_COLORS[sname],
            )

    # Reference slopes anchored at dim=10, time=10s
    _xlim = np.array([7, max(150.0, max(all_dims) * 1.15)])
    _t0_ref = 10.0
    for exp_ref, ls, lbl in [(1, ":", r"$n^1$"), (1.5, "--", r"$n^{1.5}$"), (2, "-.", r"$n^2$")]:
        _y = _t0_ref * (_xlim / 10) ** exp_ref
        ax.loglog(_xlim, _y, color="lightgrey", linestyle=ls, linewidth=1)
        ax.text(_xlim[-1]*0.6, _y[-1]*0.7, lbl, fontsize=8, color="grey")

    ax.set_xlabel("Data-space dimension  $n$", fontsize=11)
    ax.set_ylabel("Solve time (s)", fontsize=11)
    ax.set_title(f"Solver timing vs data dimension  (lmax={LMAX})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # ---- Panel 2: bound width vs data dim, per solver ----------------------
    ax2 = axes[1]
    cap_colors = plt.cm.tab10(np.linspace(0, 1, N_TARGET))

    # Only plot one solver's bound widths to keep it readable; all should agree
    # Show each solver as a distinct line style, averaged across caps
    for sname in all_solvers:
        rows = [r for r in results if r["solver"] == sname]
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: r["data_dim"])
        # Only plot finished runs in bound-width panel
        rows_sorted = [r for r in rows_sorted if not np.isnan(r["solve_time"])]
        if not rows_sorted:
            continue
        xs = np.array([r["data_dim"] for r in rows_sorted])
        widths = np.array([r["upper"] - r["lower"] for r in rows_sorted])  # (n_dims, n_caps)
        mean_width = widths.mean(axis=1)
        ax2.semilogx(
            xs, mean_width,
            color=SOLVER_COLORS[sname],
            marker=SOLVER_MARKERS[sname],
            linestyle="-",
            markersize=8,
            linewidth=1.5,
            label=f"{sname} (mean)",
        )

    # Also show individual cap widths for ProximalBundle as reference
    pb_rows = sorted([r for r in results if r["solver"] == "ProximalBundle"],
                     key=lambda r: r["data_dim"])
    if pb_rows:
        xs_pb = np.array([r["data_dim"] for r in pb_rows])
        for cap_idx in range(N_TARGET):
            cap_widths = np.array([r["upper"][cap_idx] - r["lower"][cap_idx] for r in pb_rows])
            ax2.semilogx(
                xs_pb, cap_widths,
                color=cap_colors[cap_idx],
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
                label=f"Cap {cap_idx+1}" if xs_pb[0] == xs_pb[0] else "",
            )

    ax2.set_xlabel("Data-space dimension  $n$", fontsize=11)
    ax2.set_ylabel("Mean bound width  (km/s)", fontsize=11)
    ax2.set_title("DLI bound width vs data dimension", fontsize=11)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    completed = len(results)
    total = len(DATA_DIM_CONFIGS) * len(SOLVER_NAMES)
    fig.suptitle(
        f"Solver comparison  (lmax={LMAX}, n_target={N_TARGET}, seed={SEED})"
        f"  —  {completed}/{total} runs complete",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      [figure saved → {FIGURE_PATH}]")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Partial save / load
# ---------------------------------------------------------------------------

def save_partial(results: list[dict]) -> None:
    if not results:
        return
    np.savez(
        PARTIAL_SAVE,
        solvers    = np.array([r["solver"]      for r in results]),
        data_dims  = np.array([r["data_dim"]     for r in results]),
        solve_times= np.array([r["solve_time"]   for r in results]),
        upper      = np.array([r["upper"]        for r in results]),
        lower      = np.array([r["lower"]        for r in results]),
        true_values= np.array([r["true_values"]  for r in results]),
        prior_lower= np.array([r["prior_lower"]  for r in results]),
        prior_upper= np.array([r["prior_upper"]  for r in results]),
        lmax       = LMAX,
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Sphere DLI — solver comparison timing study")
    print(f"Data dims: {[ns*nr for ns,nr in DATA_DIM_CONFIGS]}")
    print(f"Solvers:   {SOLVER_NAMES}")
    print(f"lmax={LMAX}, n_target={N_TARGET}, seed={SEED}")
    print(f"max_iter={MAX_ITER}, tol={TOL}")
    sys.stdout.flush()

    # ---- Shared setup -------------------------------------------------------
    print("\nBuilding model space and shared operators...")
    sys.stdout.flush()
    t0 = time.time()
    model_space = build_model_space(min_degree=LMAX)
    print(f"  lmax={model_space.lmax}, dim={model_space.dim}  ({time.time()-t0:.1f}s)")
    sys.stdout.flush()

    property_operator = build_cap_property_operator(
        model_space, TARGET_LATLON, n_cap=N_CAP, seed=SEED,
    )
    print(f"  Property operator: {N_TARGET} caps  ({time.time()-t0:.1f}s)")
    sys.stdout.flush()

    # Truth model at baseline dim (fixed for fair comparison)
    ref_fwd, _ = build_forward_operator(model_space, n_sources=5, n_receivers=10, seed=SEED)
    from sphere_dli_example import generate_synthetic_data
    truth_model, _ = generate_synthetic_data(model_space, ref_fwd, sigma_noise=SIGMA_NOISE, seed=SEED)
    print(f"  Truth model ready  ({time.time()-t0:.1f}s total setup)")
    sys.stdout.flush()

    # Pre-build forward operators and data for all dims
    print("\nPre-building forward operators for all data dims...")
    sys.stdout.flush()
    fwd_ops = {}
    data_vecs = {}
    for n_src, n_rec in DATA_DIM_CONFIGS:
        dim = n_src * n_rec
        fwd, _ = build_forward_operator(model_space, n_sources=n_src, n_receivers=n_rec, seed=SEED)
        _, data_vec = generate_synthetic_data(model_space, fwd, sigma_noise=SIGMA_NOISE, seed=SEED)
        fwd_ops[dim]  = fwd
        data_vecs[dim] = data_vec
        print(f"  dim={dim}: built  ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

    # ---- Resume: load any already-completed runs --------------------------
    results = []
    completed_keys: set[tuple[str, int]] = set()
    if PARTIAL_SAVE.exists():
        try:
            d = np.load(PARTIAL_SAVE, allow_pickle=True)
            for solver_name, dim, t, up, lo, tv, pl, pu in zip(
                d["solvers"], d["data_dims"], d["solve_times"],
                d["upper"], d["lower"], d["true_values"],
                d["prior_lower"], d["prior_upper"],
            ):
                results.append({
                    "solver":      str(solver_name),
                    "data_dim":    int(dim),
                    "solve_time":  float(t),
                    "upper":       np.asarray(up),
                    "lower":       np.asarray(lo),
                    "true_values": np.asarray(tv),
                    "prior_lower": np.asarray(pl),
                    "prior_upper": np.asarray(pu),
                })
                completed_keys.add((str(solver_name), int(dim)))
            print(f"  Resuming: loaded {len(results)} completed runs from {PARTIAL_SAVE}")
            sys.stdout.flush()
        except Exception as exc:
            print(f"  Warning: could not load partial save ({exc}); starting fresh")
            results = []
            completed_keys = set()

    # ---- Main sweep: dim (outer) × solver (inner) --------------------------
    total = len(DATA_DIM_CONFIGS) * len(SOLVER_NAMES)

    for n_src, n_rec in DATA_DIM_CONFIGS:
        dim = n_src * n_rec
        print(f"\n{'='*60}")
        print(f"  data_dim = {dim:5d}   (n_src={n_src}, n_rec={n_rec})")
        print(f"{'='*60}")
        sys.stdout.flush()

        for solver_name in SOLVER_NAMES:
            if (solver_name, dim) in completed_keys:
                print(f"\n  --- {solver_name} (dim={dim}) --- [SKIPPED: already done]")
                sys.stdout.flush()
                continue
            if (solver_name, dim) in SKIP_CONFIGS:
                print(f"\n  --- {solver_name} (dim={dim}) --- [DNF: in SKIP_CONFIGS]")
                sys.stdout.flush()
                # Insert a NaN-time DNF entry so the figure can show the marker
                results.append({
                    "solver":       solver_name,
                    "data_dim":     dim,
                    "solve_time":   float("nan"),
                    "upper":        np.full(N_TARGET, float("nan")),
                    "lower":        np.full(N_TARGET, float("nan")),
                    "true_values":  np.full(N_TARGET, float("nan")),
                    "prior_lower":  np.full(N_TARGET, float("nan")),
                    "prior_upper":  np.full(N_TARGET, float("nan")),
                })
                completed_keys.add((solver_name, dim))
                save_partial(results)
                plot_progress([r for r in results if not (np.isnan(r["solve_time"])
                               and (r["solver"], r["data_dim"]) not in DNF_ESTIMATED_TIMES)])
                continue
            print(f"\n  --- {solver_name} (dim={dim}) ---")
            sys.stdout.flush()
            try:
                result = run_one(
                    solver_name,
                    model_space,
                    property_operator,
                    truth_model,
                    fwd_ops[dim],
                    data_vecs[dim],
                )
                results.append(result)
            except Exception as exc:
                print(f"      ERROR: {exc}")
                sys.stdout.flush()
                # Record failure so the figure still shows other solvers
                results.append({
                    "solver":       solver_name,
                    "data_dim":     dim,
                    "solve_time":   float("nan"),
                    "upper":        np.full(N_TARGET, float("nan")),
                    "lower":        np.full(N_TARGET, float("nan")),
                    "true_values":  np.full(N_TARGET, float("nan")),
                    "prior_lower":  np.full(N_TARGET, float("nan")),
                    "prior_upper":  np.full(N_TARGET, float("nan")),
                })

            # Save and redraw after every single data point
            # Include DNF entries (NaN time + in DNF_ESTIMATED_TIMES) in the figure
            save_partial(results)
            plot_progress(results)

    # Final save
    final_save = FIG_DIR / "solver_results.npz"
    import shutil
    shutil.copy(PARTIAL_SAVE, final_save)
    print(f"\nFinal results saved: {final_save}")
    sys.stdout.flush()

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY — solve times (s)")
    print("="*60)
    header = f"  {'solver':<18}" + "".join(f"  dim={d:>4}" for d in [ns*nr for ns,nr in DATA_DIM_CONFIGS])
    print(header)
    for sname in SOLVER_NAMES:
        row = f"  {sname:<18}"
        for n_src, n_rec in DATA_DIM_CONFIGS:
            dim = n_src * n_rec
            match = [r for r in results if r["solver"] == sname and r["data_dim"] == dim]
            t = match[0]["solve_time"] if match else float("nan")
            row += f"  {t:>8.1f}s" if not np.isnan(t) else "       NaN"
        print(row)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
