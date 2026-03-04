"""
Bundle Method Visualisation — Phase 1: Steps 1 & 2
====================================================

Generates individual PNG frames illustrating the first two steps of the
Proximal Bundle Algorithm (Bundle Algorithm, Section 9 of LecturesIntroBundle):

    Step 1. Initialise: x̂₀, y₀ = x̂₀, compute f(x̂₀) and s₀ ∈ ∂f(x̂₀),
            build initial piecewise-linear model f̂₀.
    Step 2. Compute next iterate:
            y₁ = argmin_y  f̂₀(y) + (μ/2)‖y − x̂₀‖²
            (Moreau-Yosida regularisation of f̂₀ at x̂₀).

Frames are saved in the same directory as this script:
    frame_001.png  – function curve only
    frame_002.png  – x̂₀ marked; f(x̂₀) and s₀ annotated; f̂₀ drawn
    frame_003.png  – Moreau-Yosida envelope (quadratic model) drawn
    frame_004.png  – minimiser y₁ of envelope found and marked

Run with:
    conda run -n inferences3 python bundle_visualisation/phase1_steps1_2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
C_F      = "#2c3e7a"   # true function
C_MODEL  = "#e07b39"   # piecewise-linear model f̂
C_ENV    = "#8b3a9e"   # Moreau-Yosida envelope
C_XHAT  = "#c0392b"   # stability centre x̂
C_Y     = "#27ae60"   # trial point y
C_MU    = "#7f8c8d"   # μ-parabola guide
ALPHA_BG = 0.18        # background fill alpha

# ── problem definition ────────────────────────────────────────────────────────
KNOTS   = np.array([0.3, 0.8, 1.4])
WEIGHTS = np.array([1.0, 2.0, 1.0])
X_STAR  = 0.8
F_STAR  = float(np.dot(WEIGHTS, np.abs(X_STAR - KNOTS)))   # = 1.1


def f(x: float) -> float:
    """True objective: f(x) = |x−0.3| + 2|x−0.8| + |x−1.4|."""
    return float(np.dot(WEIGHTS, np.abs(x - KNOTS)))


def subgradient(x: float) -> float:
    """Subgradient of f at x (ordinary derivative away from kinks)."""
    return float(np.dot(WEIGHTS, np.sign(x - KNOTS)))


# ── algorithm parameters ──────────────────────────────────────────────────────
X_HAT0 = 2.2     # initial stability centre x̂₀
MU     = 3.0     # proximal weight μ
M_PARAM = 0.5    # serious-step fraction m
DELTA_BAR = 1e-4

# ── plot domain ───────────────────────────────────────────────────────────────
X_MIN, X_MAX = -0.3, 2.8
XS = np.linspace(X_MIN, X_MAX, 600)
FS = np.array([f(x) for x in XS])


# ═════════════════════════════════════════════════════════════════════════════
# Helper: consistent figure layout
# ═════════════════════════════════════════════════════════════════════════════

ALGO_STEPS_TEXT = [
    ("Step 1", (
        r"Choose $\bar\delta>0$, $m\!\in\!(0,1)$, set $k=0$.",
        r"Set $\hat{x}_0$, $y_0 = \hat{x}_0$.",
        r"Compute $f(\hat{x}_0)$ and $s_0 \in \partial f(\hat{x}_0)$.",
        r"Define $\hat{f}_0(y) = f(\hat{x}_0) + \langle s_0, y - \hat{x}_0\rangle$.",
    )),
    ("Step 2", (
        r"Compute next trial point:",
        r"$y_{k+1} \in \arg\min_y\; \hat{f}_k(y) + \dfrac{\mu_k}{2}\|y-\hat{x}_k\|^2$",
        r"(Moreau-Yosida regularisation of $\hat{f}_k$ at $\hat{x}_k$)",
    )),
    ("Step 3", (
        r"Compute predicted decrease:",
        r"$\delta_k = f(\hat{x}_k) - \left[\hat{f}_k(y_{k+1}) + \dfrac{\mu_k}{2}\|y_{k+1}-\hat{x}_k\|^2\right]$",
    )),
    ("Step 4", (r"If $\delta_k < \bar\delta$, \textbf{stop}.",)),
    ("Step 5", (
        r"Compute $f(y_{k+1})$ and $s_{k+1} \in \partial f(y_{k+1})$.",
    )),
    ("Step 6", (
        r"If $f(\hat{x}_k) - f(y_{k+1}) \geq m\,\delta_k$:",
        r"  \textbf{Serious Step}: $\hat{x}_{k+1} = y_{k+1}$",
        r"Else:",
        r"  \textbf{Null Step}: $\hat{x}_{k+1} = \hat{x}_k$",
    )),
    ("Step 7", (
        r"Update model:",
        r"$\hat{f}_{k+1}(y) = \max\{\hat{f}_k(y),\; f(y_{k+1})+\langle s_{k+1}, y-y_{k+1}\rangle\}$",
    )),
    ("Step 8", (r"Set $k = k+1$, go to Step 2.",)),
]


def make_figure() -> tuple:
    """Return (fig, ax_main, ax_algo) with shared consistent layout."""
    fig = plt.figure(figsize=(13, 6.5))
    ax_main = fig.add_axes([0.05, 0.10, 0.60, 0.82])   # function plot
    ax_algo = fig.add_axes([0.68, 0.05, 0.30, 0.90])   # algorithm panel
    ax_algo.axis("off")
    return fig, ax_main, ax_algo


def draw_function(ax, label_true=True):
    ax.plot(XS, FS, color=C_F, lw=2.2, zorder=4,
            label=r"$f(x) = |x-0.3| + 2|x-0.8| + |x-1.4|$" if label_true else "")
    ax.axvline(X_STAR, color="grey", ls=":", lw=0.8, alpha=0.6)
    ax.plot(X_STAR, F_STAR, "k*", ms=9, zorder=10, label=r"$x^* = 0.8,\; f^*=1.1$")
    for ki in KNOTS:
        ax.axvline(ki, color="lightgrey", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-0.5, max(FS) * 1.15)
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$f(x)$", fontsize=13)
    ax.grid(True, alpha=0.25)


def draw_algo_panel(ax_algo, active_steps: list[int], active_lines: dict[int, list[int]] | None = None):
    """Render the algorithm steps in the right panel.

    active_steps: list of step indices (0-based) to highlight
    active_lines: {step_idx: [line_indices]} — bold-highlight specific lines
    """
    if active_lines is None:
        active_lines = {}

    y_cursor = 0.97
    dy_title = 0.055
    dy_line  = 0.042

    for idx, (title, lines) in enumerate(ALGO_STEPS_TEXT):
        is_active = idx in active_steps
        title_color  = C_XHAT  if is_active else "#aaaaaa"
        title_weight = "bold"  if is_active else "normal"
        text_color   = "#222222" if is_active else "#bbbbbb"

        ax_algo.text(0.0, y_cursor, title, transform=ax_algo.transAxes,
                     fontsize=11, color=title_color, fontweight=title_weight,
                     va="top")
        y_cursor -= dy_title
        for li, line in enumerate(lines):
            hl = (idx in active_lines) and (li in active_lines.get(idx, []))
            lc = C_ENV if hl else text_color
            lw = "bold" if hl else "normal"
            ax_algo.text(0.04, y_cursor, line, transform=ax_algo.transAxes,
                         fontsize=8.5, color=lc, fontweight=lw, va="top",
                         wrap=True)
            y_cursor -= dy_line
        y_cursor -= 0.010  # gap between steps


def annotate_point(ax, x, y, label, color, dy=0.18, dx=0.12):
    ax.plot(x, y, "o", color=color, ms=9, zorder=9)
    ax.annotate(label, xy=(x, y), xytext=(x + dx, y + dy),
                fontsize=11, color=color,
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.4, mutation_scale=14),
                zorder=10)


# ═════════════════════════════════════════════════════════════════════════════
# Pre-compute Step 1 values
# ═════════════════════════════════════════════════════════════════════════════
xhat0 = X_HAT0
f_xhat0 = f(xhat0)
s0 = subgradient(xhat0)

# f̂₀(y) = f(x̂₀) + s₀(y − x̂₀)
def fhat0(y):
    return f_xhat0 + s0 * (y - xhat0)


# Step 2: y₁ = argmin  f̂₀(y) + μ/2 (y − x̂₀)²
#         derivative = s₀ + μ(y − x̂₀) = 0  ⟹  y₁ = x̂₀ − s₀/μ
y1 = xhat0 - s0 / MU
f_y1 = f(y1)

# Moreau-Yosida envelope: ĝ₀(y) = f̂₀(y) + μ/2 (y−x̂₀)²
def envelope0(y):
    return fhat0(y) + MU / 2 * (y - xhat0) ** 2


envelopes_ys = np.linspace(X_MIN, X_MAX, 600)
fhat0_ys     = np.array([fhat0(y) for y in envelopes_ys])
env0_ys      = np.array([envelope0(y) for y in envelopes_ys])

# ═════════════════════════════════════════════════════════════════════════════
# FRAME 001 — function curve only
# ═════════════════════════════════════════════════════════════════════════════

fig, ax_m, ax_a = make_figure()
draw_function(ax_m)
ax_m.set_title(r"Bundle Method — Proximal Bundle Algorithm (BA, §9)"
               "\nInitialising…", fontsize=11)
ax_m.legend(fontsize=9, loc="upper right")
draw_algo_panel(ax_a, active_steps=[])   # nothing highlighted yet

fig.savefig(OUT_DIR / "frame_001.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ frame_001.png")

# ═════════════════════════════════════════════════════════════════════════════
# FRAME 002 — Step 1 complete: x̂₀ marked, s₀ tangent line, f̂₀ drawn
# ═════════════════════════════════════════════════════════════════════════════

fig, ax_m, ax_a = make_figure()
draw_function(ax_m)

# stability centre x̂₀
ax_m.axvline(xhat0, color=C_XHAT, ls=":", lw=0.9, alpha=0.5)
annotate_point(ax_m, xhat0, f_xhat0,
               label=rf"$\hat{{x}}_0={xhat0:.1f}$"
                     "\n" rf"$f(\hat{{x}}_0)={f_xhat0:.3f}$",
               color=C_XHAT, dy=0.25, dx=-0.9)

# subgradient arrow showing s₀
arrow_len = 0.4
ax_m.annotate(
    "", xy=(xhat0 + arrow_len, f_xhat0 + s0 * arrow_len),
    xytext=(xhat0, f_xhat0),
    arrowprops=dict(arrowstyle="-|>", color=C_Y, lw=1.8, mutation_scale=16),
)
ax_m.text(xhat0 + arrow_len + 0.05, f_xhat0 + s0 * arrow_len,
          rf"$s_0 = {s0:.2f}$", color=C_Y, fontsize=10)

# f̂₀ piecewise-linear model (restricted to visible domain)
ax_m.plot(envelopes_ys, fhat0_ys, color=C_MODEL, lw=2, ls="--", zorder=5,
          label=rf"$\hat{{f}}_0(y) = f(\hat{{x}}_0) + s_0(y-\hat{{x}}_0)$")

ax_m.set_title(r"Step 1: Initialisation"
               "\n"
               r"Set $\hat{x}_0$, compute $f(\hat{x}_0)$, $s_0\in\partial f(\hat{x}_0)$, build $\hat{f}_0$",
               fontsize=10.5)
ax_m.legend(fontsize=9, loc="upper right")
draw_algo_panel(ax_a, active_steps=[0])

fig.savefig(OUT_DIR / "frame_002.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ frame_002.png")

# ═════════════════════════════════════════════════════════════════════════════
# FRAME 003 — Step 2: Moreau-Yosida envelope drawn
# ═════════════════════════════════════════════════════════════════════════════

fig, ax_m, ax_a = make_figure()
draw_function(ax_m)

# x̂₀
ax_m.axvline(xhat0, color=C_XHAT, ls=":", lw=0.9, alpha=0.5)
ax_m.plot(xhat0, f_xhat0, "o", color=C_XHAT, ms=9, zorder=9)
ax_m.text(xhat0 - 0.08, f_xhat0 + 0.2,
          rf"$\hat{{x}}_0={xhat0:.1f}$", color=C_XHAT, fontsize=10,
          ha="right")

# f̂₀
ax_m.plot(envelopes_ys, fhat0_ys, color=C_MODEL, lw=1.8, ls="--", zorder=4,
          label=rf"$\hat{{f}}_0$ (piecewise-linear model)", alpha=0.8)

# μ/2 ‖y − x̂₀‖² parabola (guide)
parabola = MU / 2 * (envelopes_ys - xhat0) ** 2
# shift parabola to start at f(x̂₀) so it is visually readable
ax_m.plot(envelopes_ys, f_xhat0 + parabola - parabola[0],
          color=C_MU, lw=1.2, ls=":", alpha=0.5,
          label=rf"$\mu_0/2 \cdot \Vert y-\hat{{x}}_0\Vert^2$  ($\mu_0={MU}$)")

# Envelope ĝ₀ = f̂₀ + μ/2 ‖y−x̂₀‖²
# clip to sensible y-range for readability
env_clip = np.clip(env0_ys, -1, max(FS) * 1.4)
ax_m.plot(envelopes_ys, env_clip, color=C_ENV, lw=2.3, zorder=5,
          label=r"$\hat{f}_0(y) + \mu_0/2 \cdot \Vert y-\hat{x}_0\Vert^2$  (envelope)")

ax_m.set_title(r"Step 2: Moreau-Yosida envelope of $\hat{f}_0$ at $\hat{x}_0$"
               "\n"
               r"$y_1 = \arg\min_y\; [\hat{f}_0(y) + \mu_0/2 \cdot \Vert y-\hat{x}_0\Vert^2]$",
               fontsize=10.5)
ax_m.legend(fontsize=8.5, loc="upper right")
draw_algo_panel(ax_a, active_steps=[1])

fig.savefig(OUT_DIR / "frame_003.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ frame_003.png")

# ═════════════════════════════════════════════════════════════════════════════
# FRAME 004 — Step 2: minimiser y₁ found
# ═════════════════════════════════════════════════════════════════════════════

fig, ax_m, ax_a = make_figure()
draw_function(ax_m)

# x̂₀
ax_m.axvline(xhat0, color=C_XHAT, ls=":", lw=0.9, alpha=0.4)
ax_m.plot(xhat0, f_xhat0, "o", color=C_XHAT, ms=9, zorder=9)
ax_m.text(xhat0 - 0.08, f_xhat0 + 0.2,
          rf"$\hat{{x}}_0={xhat0:.1f}$", color=C_XHAT, fontsize=10, ha="right")

# f̂₀
ax_m.plot(envelopes_ys, fhat0_ys, color=C_MODEL, lw=1.8, ls="--", zorder=4,
          alpha=0.7,
          label=r"$\hat{f}_0$ (piecewise-linear model)")

# Envelope
env_clip = np.clip(env0_ys, -1, max(FS) * 1.4)
ax_m.plot(envelopes_ys, env_clip, color=C_ENV, lw=2.3, zorder=5,
          label=r"$\hat{f}_0(y) + \mu/2\,\Vert y-\hat{x}_0\Vert^2$")

# minimiser y₁
env_at_y1 = envelope0(y1)
ax_m.axvline(y1, color=C_Y, ls=":", lw=0.9, alpha=0.5)
annotate_point(ax_m, y1, env_at_y1,
               label=(rf"$y_1 = \hat{{x}}_0 - s_0/\mu$"
                      "\n"
                      rf"$\;= {y1:.3f}$"),
               color=C_Y, dy=-0.45, dx=-0.65)

# horizontal line at envelope minimum
ax_m.axhline(env_at_y1, color=C_Y, ls=":", lw=0.8, alpha=0.4)

# arrow from x̂₀ to y₁ showing the proximal step
mid_y = (f_xhat0 + env_at_y1) / 2
ax_m.annotate("", xy=(y1, env_at_y1), xytext=(xhat0, f_xhat0),
              arrowprops=dict(arrowstyle="-|>", color="#555555",
                              lw=1.5, mutation_scale=14,
                              connectionstyle="arc3,rad=0.25"))
ax_m.text((xhat0 + y1) / 2 + 0.1, mid_y + 0.15,
          "proximal\nstep", fontsize=8.5, color="#555555", ha="center")

ax_m.set_title(r"Step 2: Trial point $y_1 = \arg\min_y\; [\hat{f}_0(y) + \mu_0/2 \cdot \Vert y-\hat{x}_0\Vert^2]$"
               "\n"
               rf"Analytical minimum: $y_1 = \hat{{x}}_0 - s_0/\mu = {y1:.3f}$",
               fontsize=10.5)
ax_m.legend(fontsize=8.5, loc="upper right")
draw_algo_panel(ax_a, active_steps=[1], active_lines={1: [1]})

fig.savefig(OUT_DIR / "frame_004.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ frame_004.png")

# ── summary ───────────────────────────────────────────────────────────────────
print()
print("Phase 1 complete: 4 frames saved to", OUT_DIR)
print(f"  x̂₀ = {xhat0},  f(x̂₀) = {f_xhat0:.4f},  s₀ = {s0:.4f}")
print(f"  μ  = {MU},       y₁   = {y1:.4f},  f(y₁) = {f_y1:.4f}")
print()
print("Next: run phase2_steps3_5.py to add frames 005–008")
