"""
Benchmark: SubgradientDescent vs ProximalBundleMethod vs LevelBundleMethod
==========================================================================

Compares three non-smooth convex optimisation algorithms on a simple 1D
function with four kinks:

    f(x) = |x - 0.3| + 2|x - 0.8| + |x - 1.4|

The weighted median is at x* = 0.8  (the weight-2 knot),
with f* = |0.8 - 0.3| + 0 + |0.8 - 1.4| = 0.5 + 0.6 = 1.1.

Output
------
Saves ``comparison.png`` in the same directory as this script.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive; remove for interactive use
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Make the package importable when running from the repo root
# ---------------------------------------------------------------------------
_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from pygeoinf.hilbert_space import EuclideanSpace  # noqa: E402
from pygeoinf.convex_optimisation import (  # noqa: E402
    SubgradientDescent,
    ProximalBundleMethod,
    LevelBundleMethod,
    best_available_qp_solver,
)
from pygeoinf.nonlinear_forms import NonLinearForm  # noqa: E402

QP_SOLVER = best_available_qp_solver()
print(f"Using QP solver: {type(QP_SOLVER).__name__}")

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
KNOTS = np.array([0.3, 0.8, 1.4])
WEIGHTS = np.array([1.0, 2.0, 1.0])

X_STAR = 0.8          # weighted median
F_STAR = np.sum(WEIGHTS * np.abs(X_STAR - KNOTS))   # = 1.1

SPACE = EuclideanSpace(1)


def f(x: np.ndarray) -> float:
    """Weighted sum of absolute values."""
    return float(np.dot(WEIGHTS, np.abs(x[0] - KNOTS)))


def g(x: np.ndarray) -> np.ndarray:
    """Subgradient of f at x (use derivative where defined; 0 at kinks)."""
    return np.array([float(np.dot(WEIGHTS, np.sign(x[0] - KNOTS)))])


oracle = NonLinearForm(SPACE, f, subgradient=g)

X0 = np.array([-1.0])   # starting point well outside the kinks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_timed(solver):
    """Return (result, wall_time_seconds)."""
    t0 = time.perf_counter()
    result = solver.solve(X0.copy())
    return result, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
MAX_ITER = 600

step_sizes = [0.05, 0.1, 0.3]
subgrad_results: dict[str, tuple] = {}

for alpha in step_sizes:
    solver = SubgradientDescent(
        oracle,
        step_size=alpha,
        max_iterations=MAX_ITER,
        store_iterates=True,
    )
    res, t = run_timed(solver)
    label = f"SubGrad α={alpha}"
    subgrad_results[label] = (res, t)

pbm_solver = ProximalBundleMethod(
    oracle,
    rho0=1.0,
    rho_factor=2.0,
    tolerance=1e-8,
    max_iterations=MAX_ITER,
    bundle_size=50,
    store_iterates=True,
    qp_solver=QP_SOLVER,
)
pbm_result, pbm_time = run_timed(pbm_solver)

lbm_solver = LevelBundleMethod(
    oracle,
    alpha=0.3,
    tolerance=1e-8,
    max_iterations=MAX_ITER,
    bundle_size=50,
    store_iterates=True,
    qp_solver=QP_SOLVER,
)
lbm_result, lbm_time = run_timed(lbm_solver)

# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"True optimum:  x* = {X_STAR:.4f},  f* = {F_STAR:.4f}")
print("=" * 60)

for label, (res, t) in subgrad_results.items():
    x_val = SPACE.to_components(res.x_best)[0]
    print(
        f"{label:<22}  "
        f"f_best={res.f_best:.6f}  "
        f"|f-f*|={abs(res.f_best - F_STAR):.2e}  "
        f"iters={res.num_iterations:<4}  "
        f"time={t*1e3:.2f} ms"
    )

for label, (res, t) in [
    ("ProxBundleMethod", (pbm_result, pbm_time)),
    ("LevelBundleMethod", (lbm_result, lbm_time)),
]:
    flag = "✓" if res.converged else " "
    x_val = SPACE.to_components(res.x_best)[0]
    print(
        f"{flag} {label:<21}  "
        f"f_best={res.f_best:.6f}  "
        f"|f-f*|={abs(res.f_best - F_STAR):.2e}  "
        f"iters={res.num_iterations:<4}  "
        f"serious={res.num_serious_steps:<4}  "
        f"time={t*1e3:.2f} ms"
    )
print("=" * 60)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
SG_PALETTE = {
    "SubGrad α=0.05": "#4C72B0",
    "SubGrad α=0.1":  "#DD8452",
    "SubGrad α=0.3":  "#55A868",
}
BUNDLE_PALETTE = {
    "ProxBundleMethod": "#C44E52",
    "LevelBundleMethod": "#8172B3",
}

# ============================================================
# Figure layout: 2×2 grid
#   [0,0] Function landscape + iterate scatter
#   [0,1] Convergence curves (f_best - f*)
#   [1,0] Iterate paths over time (x value vs iteration)
#   [1,1] Wall-clock timing bar chart + solution quality
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle(
    r"Optimiser comparison  —  $f(x)=|x-0.3|+2|x-0.8|+|x-1.4|$,  $f^*=1.1$ at $x^*=0.8$",
    fontsize=12,
    y=1.01,
)

# ---- Panel [0,0]: function landscape + iterate trails ----------------------
ax = axes[0, 0]
xs_plot = np.linspace(-1.5, 3.0, 500)
ys_plot = np.array([f(np.array([xi])) for xi in xs_plot])
ax.plot(xs_plot, ys_plot, "k-", lw=2, label="f(x)", zorder=5)
for xi in KNOTS:
    ax.axvline(xi, color="grey", ls=":", lw=0.8, alpha=0.6)
ax.axvline(X_STAR, color="red", ls="--", lw=1.2, alpha=0.8, label=f"x* = {X_STAR}")
ax.plot(X_STAR, F_STAR, "r*", ms=12, zorder=10)

# Scatter iterates for each subgradient run (subsample every 20)
for label, (res, _) in subgrad_results.items():
    if res.iterates:
        pts = np.array([SPACE.to_components(v)[0] for v in res.iterates[::20]])
        fx_pts = np.array([f(np.array([xi])) for xi in pts])
        ax.scatter(pts, fx_pts, s=10, alpha=0.45, color=SG_PALETTE[label], zorder=4)

# Scatter ProxBundleMethod serious steps
if pbm_result.iterates:
    pts = np.array([SPACE.to_components(v)[0] for v in pbm_result.iterates])
    fx_pts = np.array([f(np.array([xi])) for xi in pts])
    ax.scatter(pts, fx_pts, s=15, alpha=0.65, color=BUNDLE_PALETTE["ProxBundleMethod"],
               marker="D", zorder=6, label="Prox iterates")

if lbm_result.iterates:
    pts = np.array([SPACE.to_components(v)[0] for v in lbm_result.iterates])
    fx_pts = np.array([f(np.array([xi])) for xi in pts])
    ax.scatter(pts, fx_pts, s=15, alpha=0.65, color=BUNDLE_PALETTE["LevelBundleMethod"],
               marker="^", zorder=6, label="Level iterates")

ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Function landscape + iterates")
ax.set_xlim(-1.6, 3.1)
ax.legend(fontsize=8, loc="upper right")

# ---- Panel [0,1]: convergence (f_best - f*) --------------------------------
ax = axes[0, 1]
EPS = 1e-12   # floor for log-scale

for label, (res, _) in subgrad_results.items():
    vals = np.maximum(np.array(res.function_values) - F_STAR, EPS)
    ax.semilogy(vals, lw=1.2, color=SG_PALETTE[label], label=label)

# Bundle methods: f_best tracks the running best
for label, (res, _), col in [
    ("ProxBundleMethod", (pbm_result, pbm_time), BUNDLE_PALETTE["ProxBundleMethod"]),
    ("LevelBundleMethod", (lbm_result, lbm_time), BUNDLE_PALETTE["LevelBundleMethod"]),
]:
    fvals = np.array(res.function_values)
    running_best = np.minimum.accumulate(fvals)
    ax.semilogy(
        np.maximum(running_best - F_STAR, EPS),
        lw=2, color=col, label=label,
    )
    # Mark serious steps for bundle methods
    if res.iterates:
        ax.semilogy(
            [0, len(fvals) - 1],
            [np.maximum(running_best[0] - F_STAR, EPS),
             np.maximum(running_best[-1] - F_STAR, EPS)],
            ".", color=col, ms=0,  # invisible, just for legend coherence
        )

ax.axhline(1e-4, color="grey", ls=":", lw=0.8, label="tol=1e-4")
ax.set_xlabel("Iteration")
ax.set_ylabel(r"$f_\mathrm{best} - f^*$")
ax.set_title("Convergence (log scale)")
ax.legend(fontsize=8)
ax.set_xlim(left=0)

# ---- Panel [1,0]: iterate paths (x value over iterations) ------------------
ax = axes[1, 0]

for label, (res, _) in subgrad_results.items():
    if res.iterates:
        xs_path = np.array([SPACE.to_components(v)[0] for v in res.iterates])
        ax.plot(xs_path, lw=1.0, alpha=0.7, color=SG_PALETTE[label], label=label)

for label, (res, _), col, mkr in [
    ("ProxBundleMethod", (pbm_result, pbm_time), BUNDLE_PALETTE["ProxBundleMethod"], "D"),
    ("LevelBundleMethod", (lbm_result, lbm_time), BUNDLE_PALETTE["LevelBundleMethod"], "^"),
]:
    if res.iterates:
        xs_path = np.array([SPACE.to_components(v)[0] for v in res.iterates])
        ax.plot(xs_path, lw=2, color=col, label=label)
        ax.scatter(range(len(xs_path)), xs_path, s=25, color=col,
                   marker=mkr, zorder=5, alpha=0.8)

ax.axhline(X_STAR, color="red", ls="--", lw=1.2, label=f"x* = {X_STAR}")
for xi in KNOTS:
    ax.axhline(xi, color="grey", ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("x (iterate value)")
ax.set_title("Iterate paths")
ax.legend(fontsize=8)
ax.set_xlim(left=0)

# ---- Panel [1,1]: timing + quality bar chart --------------------------------
ax = axes[1, 1]

labels_all = list(subgrad_results.keys()) + ["ProxBundleMethod", "LevelBundleMethod"]
times_all = [t for _, (_, t) in subgrad_results.items()] + [pbm_time, lbm_time]
errors_all = (
    [abs(res.f_best - F_STAR) for _, (res, _) in subgrad_results.items()]
    + [abs(pbm_result.f_best - F_STAR), abs(lbm_result.f_best - F_STAR)]
)
colors_all = list(SG_PALETTE.values()) + list(BUNDLE_PALETTE.values())

x_pos = np.arange(len(labels_all))
width = 0.4

ax2 = ax.twinx()

bars_t = ax.bar(x_pos - width / 2, np.array(times_all) * 1e3, width,
                color=colors_all, alpha=0.8, label="Wall time (ms)")
bars_e = ax2.bar(x_pos + width / 2, errors_all, width,
                 color=colors_all, alpha=0.4, hatch="//", label=r"|f_best - f*|")

ax.set_xticks(x_pos)
ax.set_xticklabels(labels_all, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Wall-clock time (ms)", color="black")
ax2.set_ylabel(r"$|f_\mathrm{best} - f^*|$", color="dimgrey")
ax2.set_yscale("log")
ax.set_title("Timing and solution quality")

lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper left")

# ---------------------------------------------------------------------------
# Save + show
# ---------------------------------------------------------------------------
plt.tight_layout()
out_path = Path(__file__).with_name("comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {out_path}")

# Only show interactively when not using a non-interactive backend.
import matplotlib
if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
    plt.show()
