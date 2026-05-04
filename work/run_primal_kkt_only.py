"""Run PrimalKKT solver for dims 10, 50, 100 and merge into solver_partial.npz.

Usage::

    conda activate inferences3
    cd <workspace-root>
    python -u pygeoinf/work/run_primal_kkt_only.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import numpy as np

_WORK_DIR = Path(__file__).parent
if str(_WORK_DIR) not in sys.path:
    sys.path.insert(0, str(_WORK_DIR))

from sphere_dli_solver_comparison import (
    DATA_DIM_CONFIGS,
    N_TARGET,
    PARTIAL_SAVE,
    LMAX,
    N_CAP,
    N_REPEATS,
    SEED,
    solve_with_primal_kkt,
    save_partial,
    plot_progress,
    _build_dli_components,
)
from sphere_dli_example import (
    DEFAULT_TARGET_LATLON,
    SIGMA_NOISE,
    build_cap_property_operator,
    build_forward_operator,
    build_model_space,
    generate_synthetic_data,
)

TARGET_LATLON = DEFAULT_TARGET_LATLON[:N_TARGET]

# Only run these dims
TARGET_DIMS = {10, 50, 100}

# ---------------------------------------------------------------------------
# Load existing partial results
# ---------------------------------------------------------------------------
results: list[dict] = []
completed_keys: set[tuple[str, int]] = set()

if PARTIAL_SAVE.exists():
    d = np.load(PARTIAL_SAVE, allow_pickle=True)
    skipped = 0
    for solver_name, dim, nrep, t_mean, t_std, ts, up, lo, up_std, lo_std, tv, pl, pu in zip(
        d["solvers"], d["data_dims"], d["n_repeats"],
        d["solve_time_mean"], d["solve_time_std"], d["solve_times"],
        d["upper"], d["lower"], d["upper_std"], d["lower_std"],
        d["true_values"], d["prior_lower"], d["prior_upper"],
    ):
        sname = str(solver_name)
        # Drop entries that will be recomputed with the current PrimalKKT.
        if sname == "PrimalKKT":
            skipped += 1
            continue
        results.append({
            "solver":          sname,
            "data_dim":        int(dim),
            "n_repeats":       int(nrep),
            "solve_time_mean": float(t_mean),
            "solve_time_std":  float(t_std),
            "solve_times":     np.asarray(ts, dtype=float),
            "upper":           np.asarray(up),
            "lower":           np.asarray(lo),
            "upper_std":       np.asarray(up_std),
            "lower_std":       np.asarray(lo_std),
            "true_values":     np.asarray(tv),
            "prior_lower":     np.asarray(pl),
            "prior_upper":     np.asarray(pu),
        })
        completed_keys.add((sname, int(dim)))
    print(f"Loaded {len(results)} existing results from {PARTIAL_SAVE} (dropped {skipped} PrimalKKT entries)")
else:
    print("No partial save found — starting fresh.")

# ---------------------------------------------------------------------------
# Build model space and operators
# ---------------------------------------------------------------------------
print("\nBuilding model space and shared operators...")
t0 = time.time()
model_space = build_model_space(min_degree=LMAX)
print(f"  lmax={LMAX}, dim={model_space.dim}  ({time.time()-t0:.1f}s)")

property_operator = build_cap_property_operator(model_space, TARGET_LATLON, n_cap=N_CAP)
print(f"  Property operator: {N_TARGET} caps  ({time.time()-t0:.1f}s)")

# Truth model at fixed baseline dim (same as main script)
ref_fwd, _ = build_forward_operator(model_space, n_sources=5, n_receivers=10, seed=SEED)
truth_model, _ = generate_synthetic_data(model_space, ref_fwd, sigma_noise=SIGMA_NOISE, seed=SEED)
print(f"  Truth model ready  ({time.time()-t0:.1f}s total setup)")

print("\nPre-building forward operators for target dims...")
fwd_ops: dict[int, object] = {}
data_vecs: dict[int, np.ndarray] = {}
for n_src, n_rec in DATA_DIM_CONFIGS:
    dim = n_src * n_rec
    if dim not in TARGET_DIMS:
        continue
    t1 = time.time()
    fwd, _ = build_forward_operator(model_space, n_sources=n_src, n_receivers=n_rec, seed=SEED)
    _, data_vec = generate_synthetic_data(model_space, fwd, sigma_noise=SIGMA_NOISE, seed=SEED)
    fwd_ops[dim] = fwd
    data_vecs[dim] = data_vec
    print(f"  dim={dim}: built  ({time.time()-t1:.1f}s)")

# ---------------------------------------------------------------------------
# Run PrimalKKT for each target dim
# ---------------------------------------------------------------------------
for n_src, n_rec in DATA_DIM_CONFIGS:
    dim = n_src * n_rec
    if dim not in TARGET_DIMS:
        continue

    key = ("PrimalKKT", dim)
    if key in completed_keys:
        print(f"\n  --- PrimalKKT (dim={dim}) --- [SKIPPED: already done]")
        continue

    print(f"\n  --- PrimalKKT (dim={dim}) ---")
    sys.stdout.flush()

    (cost, model_ball, data_ball,
     basis_dirs, neg_basis,
     true_values, prior_bounds) = _build_dli_components(
        model_space, fwd_ops[dim], property_operator, truth_model, data_vecs[dim]
    )

    t0 = time.time()
    try:
        per_run_times = []
        per_run_upper = []
        per_run_lower = []
        for rep in range(N_REPEATS):
            rep_t0 = time.time()
            upper, lower = solve_with_primal_kkt(
                cost, basis_dirs, neg_basis, model_ball, data_ball,
                fwd_ops[dim], data_vecs[dim],
            )
            per_run_times.append(time.time() - rep_t0)
            per_run_upper.append(np.asarray(upper, dtype=float))
            per_run_lower.append(np.asarray(lower, dtype=float))

        times = np.asarray(per_run_times, dtype=float)
        uppers = np.asarray(per_run_upper, dtype=float)
        lowers = np.asarray(per_run_lower, dtype=float)
        elapsed = time.time() - t0
        print(f"    → {times.mean():.2f}s ± {times.std(ddof=1) if N_REPEATS > 1 else 0.0:.2f}s   bounds: [{lowers.mean(axis=0)[0]:.3f}, {uppers.mean(axis=0)[0]:.3f}]")
        results.append({
            "solver":          "PrimalKKT",
            "data_dim":        dim,
            "n_repeats":       int(N_REPEATS),
            "solve_time_mean": float(times.mean()),
            "solve_time_std":  float(times.std(ddof=1)) if N_REPEATS > 1 else 0.0,
            "solve_times":     times,
            "upper":           uppers.mean(axis=0),
            "lower":           lowers.mean(axis=0),
            "upper_std":       uppers.std(axis=0, ddof=1) if N_REPEATS > 1 else np.zeros_like(uppers[0]),
            "lower_std":       lowers.std(axis=0, ddof=1) if N_REPEATS > 1 else np.zeros_like(lowers[0]),
            "true_values":     true_values,
            "prior_lower":     -prior_bounds,
            "prior_upper":      prior_bounds,
        })
    except Exception as exc:
        print(f"    ERROR: {exc}")
        results.append({
            "solver":          "PrimalKKT",
            "data_dim":        dim,
            "n_repeats":       int(N_REPEATS),
            "solve_time_mean": float("nan"),
            "solve_time_std":  float("nan"),
            "solve_times":     np.full(N_REPEATS, float("nan")),
            "upper":           np.full(N_TARGET, float("nan")),
            "lower":           np.full(N_TARGET, float("nan")),
            "upper_std":       np.full(N_TARGET, float("nan")),
            "lower_std":       np.full(N_TARGET, float("nan")),
            "true_values":     true_values,
            "prior_lower":     np.full(N_TARGET, float("nan")),
            "prior_upper":     np.full(N_TARGET, float("nan")),
        })

    completed_keys.add(key)
    save_partial(results)
    plot_progress(results)
    print(f"    [saved → {PARTIAL_SAVE}]")
    sys.stdout.flush()

print("\nDone. PrimalKKT results merged into solver_partial.npz")
