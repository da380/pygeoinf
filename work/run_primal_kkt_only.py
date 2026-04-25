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
    for solver_name, dim, t, up, lo, tv, pl, pu in zip(
        d["solvers"], d["data_dims"], d["solve_times"],
        d["upper"], d["lower"], d["true_values"],
        d["prior_lower"], d["prior_upper"],
    ):
        sname = str(solver_name)
        # Drop ALL PrimalKKT entries (buggy bounds from previous run);
        # they will be recomputed with the corrected Gram matrix
        if sname == "PrimalKKT":
            skipped += 1
            continue
        results.append({
            "solver":      sname,
            "data_dim":    int(dim),
            "solve_time":  float(t),
            "upper":       np.asarray(up),
            "lower":       np.asarray(lo),
            "true_values": np.asarray(tv),
            "prior_lower": np.asarray(pl),
            "prior_upper": np.asarray(pu),
        })
        completed_keys.add((sname, int(dim)))
    print(f"Loaded {len(results)} existing results from {PARTIAL_SAVE} (dropped {skipped} failed PrimalKKT entries)")
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
        upper, lower = solve_with_primal_kkt(
            cost, basis_dirs, neg_basis, model_ball, data_ball,
            fwd_ops[dim], data_vecs[dim],
        )
        elapsed = time.time() - t0
        print(f"    → {elapsed:.2f}s   bounds: [{lower[0]:.3f}, {upper[0]:.3f}]")
        results.append({
            "solver":      "PrimalKKT",
            "data_dim":    dim,
            "solve_time":  elapsed,
            "upper":       upper,
            "lower":       lower,
            "true_values": true_values,
            "prior_lower": -prior_bounds,
            "prior_upper":  prior_bounds,
        })
    except Exception as exc:
        print(f"    ERROR: {exc}")
        results.append({
            "solver":       "PrimalKKT",
            "data_dim":     dim,
            "solve_time":   float("nan"),
            "upper":        np.full(N_TARGET, float("nan")),
            "lower":        np.full(N_TARGET, float("nan")),
            "true_values":  true_values,
            "prior_lower":  np.full(N_TARGET, float("nan")),
            "prior_upper":  np.full(N_TARGET, float("nan")),
        })

    completed_keys.add(key)
    save_partial(results)
    plot_progress(results)
    print(f"    [saved → {PARTIAL_SAVE}]")
    sys.stdout.flush()

print("\nDone. PrimalKKT results merged into solver_partial.npz")
