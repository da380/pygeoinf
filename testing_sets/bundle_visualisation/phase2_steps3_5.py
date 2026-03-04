"""
Bundle Method Visualisation — Phase 2: Steps 3, 4 & 5
=======================================================

Generates frames 005–008 continuing the bundle algorithm visualization from
phase1_steps1_2.py.

    Step 3. Compute predicted decrease:
            δ₀ = f(x̂₀) − [f̂₀(y₁) + μ₀/2‖y₁−x̂₀‖²]
    Step 4. If δ₀ < δ̄, stop (not satisfied → continue).
    Step 5. Evaluate oracle at trial point:
            compute f(y₁) and s₁ ∈ ∂f(y₁).

Frames saved:
    frame_005.png  – δ₀ highlighted as gap between f(x̂₀) and envelope value at y₁
    frame_006.png  – δ₀ vs δ̄ threshold: δ₀ >> δ̄ → algorithm continues
    frame_007.png  – oracle called at y₁: f(y₁) and s₁ shown
    frame_008.png  – gap between f(y₁) (true) and f̂₀(y₁) (model) emphasised
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── import shared state from phase 1 ─────────────────────────────────────────
# We simply reproduce the constants; keeping phases self-contained.
OUT_DIR = Path(__file__).parent

KNOTS   = np.array([0.3, 0.8, 1.4])
WEIGHTS = np.array([1.0, 2.0, 1.0])
X_STAR  = 0.8
F_STAR  = float(np.dot(WEIGHTS, np.abs(X_STAR - KNOTS)))

def f(x):
    return float(np.dot(WEIGHTS, np.abs(x - KNOTS)))

def subgradient(x):
    return float(np.dot(WEIGHTS, np.sign(x - KNOTS)))

X_HAT0  = 2.2
MU      = 3.0
M_PARAM = 0.5
DELTA_BAR = 1e-4

X_MIN, X_MAX = -0.3, 2.8
XS = np.linspace(X_MIN, X_MAX, 600)
FS = np.array([f(x) for x in XS])

# ── palette ───────────────────────────────────────────────────────────────────
C_F     = "#2c3e7a"
C_MODEL = "#e07b39"
C_ENV   = "#8b3a9e"
C_XHAT  = "#c0392b"
C_Y     = "#27ae60"
C_MU    = "#7f8c8d"
C_DELTA = "#e74c3c"   # predicted decrease annotation

# ── pre-compute ───────────────────────────────────────────────────────────────
xhat0   = X_HAT0
f_xhat0 = f(xhat0)
s0      = subgradient(xhat0)

def fhat0(y):
    return f_xhat0 + s0 * (y - xhat0)

y1       = xhat0 - s0 / MU          # trial point from Step 2
f_y1     = f(y1)
s1       = subgradient(y1)

fhat0_y1  = fhat0(y1)
prox_term = MU / 2 * (y1 - xhat0) ** 2
Qk        = fhat0_y1 + prox_term    # value of the regulated model at y₁
delta0    = f_xhat0 - Qk            # predicted decrease δ₀

# model curve arrays
envelopes_ys = np.linspace(X_MIN, X_MAX, 600)
fhat0_ys     = np.array([fhat0(y) for y in envelopes_ys])
env0_ys      = np.clip(fhat0_ys + MU / 2 * (envelopes_ys - xhat0) ** 2,
                       -1, max(FS) * 1.4)

print(f"δ₀ = f(x̂₀) − Q₀(y₁) = {f_xhat0:.4f} − {Qk:.4f} = {delta0:.4f}")
print(f"δ̄  = {DELTA_BAR}  →  δ₀ > δ̄, algorithm continues")
print(f"f(y₁) = {f_y1:.4f},  s₁ = {s1:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers (same layout as phase 1)
# ═════════════════════════════════════════════════════════════════════════════

ALGO_STEPS_TEXT = [
    ("Step 1", (
        r"Choose $\bar\delta>0$, $m\!\in\!(0,1)$, set $k=0$.",
        r"Set $\hat{x}_0$, $y_0 = \hat{x}_0$.",
        r"Compute $f(\hat{x}_0)$ and $s_0 \in \partial f(\hat{x}_0)$.",
        r"Build $\hat{f}_0(y) = f(\hat{x}_0) + \langle s_0, y - \hat{x}_0\rangle$.",
    )),
    ("Step 2", (
        r"Compute trial point:",
        r"$y_{k+1} = \arg\min_y\; \hat{f}_k(y) + \mu_k/2\,\Vert y-\hat{x}_k\Vert^2$",
    )),
    ("Step 3", (
        r"Compute predicted decrease $\delta_k$:",
        r"$\delta_k = f(\hat{x}_k) - [\hat{f}_k(y_{k+1}) + \mu_k/2\,\Vert y_{k+1}-\hat{x}_k\Vert^2]$",
    )),
    ("Step 4", (r"If $\delta_k < \bar\delta$, \textbf{stop}.",)),
    ("Step 5", (
        r"Compute $f(y_{k+1})$ and $s_{k+1} \in \partial f(y_{k+1})$.",
    )),
    ("Step 6", (
        r"If $f(\hat{x}_k) - f(y_{k+1}) \geq m\,\delta_k$:",
        r"  Serious Step: $\hat{x}_{k+1} = y_{k+1}$",
        r"Else:",
        r"  Null Step: $\hat{x}_{k+1} = \hat{x}_k$",
    )),
    ("Step 7", (
        r"Update model: $\hat{f}_{k+1} = \max\{\hat{f}_k,\; f(y_{k+1})+\langle s_{k+1}, \cdot-y_{k+1}\rangle\}$",
    )),
    ("Step 8", (r"Set $k = k+1$, go to Step 2.",)),
]


def make_figure():
    fig = plt.figure(figsize=(13, 6.5))
    ax_main = fig.add_axes([0.05, 0.10, 0.60, 0.82])
    ax_algo = fig.add_axes([0.68, 0.05, 0.30, 0.90])
    ax_algo.axis("off")
    return fig, ax_main, ax_algo


def draw_base(ax, show_xhat0=True, show_y1=False, show_fhat0=False,
              show_envelope=False):
    """Draw the function, optionally with previous-step artefacts (dimmed)."""
    ax.plot(XS, FS, color=C_F, lw=2.2, zorder=4,
            label=r"$f(x)$")
    ax.plot(X_STAR, F_STAR, "k*", ms=9, zorder=10,
            label=r"$x^*=0.8$")
    for ki in KNOTS:
        ax.axvline(ki, color="lightgrey", ls="--", lw=0.6, alpha=0.4)

    if show_fhat0:
        ax.plot(envelopes_ys, fhat0_ys, color=C_MODEL, lw=1.6, ls="--",
                alpha=0.45, zorder=3, label=r"$\hat{f}_0$ (model)")

    if show_envelope:
        ax.plot(envelopes_ys, env0_ys, color=C_ENV, lw=1.8, alpha=0.45,
                zorder=3, label=r"$\hat{f}_0 + \mu_0/2\,\Vert\cdot-\hat{x}_0\Vert^2$")

    if show_xhat0:
        ax.plot(xhat0, f_xhat0, "o", color=C_XHAT, ms=9, zorder=9)
        ax.text(xhat0 + 0.05, f_xhat0 + 0.15,
                rf"$\hat{{x}}_0={xhat0}$", color=C_XHAT, fontsize=9)

    if show_y1:
        ax.plot(y1, f_y1, "o", color=C_Y, ms=8, zorder=9)
        ax.text(y1 + 0.05, f_y1 + 0.15,
                rf"$y_1={y1:.3f}$", color=C_Y, fontsize=9)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-0.5, max(FS) * 1.15)
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$f(x)$", fontsize=13)
    ax.grid(True, alpha=0.25)


def draw_algo_panel(ax_algo, active_steps, active_lines=None):
    if active_lines is None:
        active_lines = {}
    y_cursor = 0.97
    dy_title = 0.055
    dy_line  = 0.042
    for idx, (title, lines) in enumerate(ALGO_STEPS_TEXT):
        is_active   = idx in active_steps
        title_color = C_XHAT if is_active else "#aaaaaa"
        title_wt    = "bold" if is_active else "normal"
        text_color  = "#222222" if is_active else "#bbbbbb"
        ax_algo.text(0.0, y_cursor, title, transform=ax_algo.transAxes,
                     fontsize=11, color=title_color, fontweight=title_wt,
                     va="top")
        y_cursor -= dy_title
        for li, line in enumerate(lines):
            hl = (idx in active_lines) and (li in active_lines.get(idx, []))
            ax_algo.text(0.04, y_cursor, line, transform=ax_algo.transAxes,
                         fontsize=8.5, va="top", wrap=True,
                         color=C_ENV if hl else text_color,
                         fontweight="bold" if hl else "normal")
            y_cursor -= dy_line
        y_cursor -= 0.010


# ═════════════════════════════════════════════════════════════════════════════
# FRAME 005 — Step 3: predicted decrease δ₀ visualised as a gap
# ═════════════════════════════════════════════════════════════════════════════

fig, ax_m, ax_a = make_figure()
draw_base(ax_m, show_xhat0=True, show_y1=False,
          show_fhat0=True, show_envelope=True)

# vertical line at y₁
ax_m.axvline(y1, color=C_Y, ls=":", lw=0.9, alpha=0.5)
ax_m.plot(y1, Qk, "s", color=C_ENV, ms=9, zorder=9)
ax_m.text(y1 + 0.06, Qk - 0.3,
          rf"$Q_0(y_1)={Qk:.3f}$" "\n"
          r"$= \hat{f}_0(y_1) + \mu_0/2\,\Vert y_1-\hat{x}_0\Vert^2$",
          fontsize=8.5, color=C_ENV)

# f(x̂₀) marker
ax_m.plot(xhat0, f_xhat0, "o", color=C_XHAT, ms=10, zorder=10)
ax_m.axhline(f_xhat0, color=C_XHAT, ls=":", lw=0.8, alpha=0.45)

# δ₀ brace / double-headed arrow
ax_m.annotate("", xy=(y1 - 0.15, f_xhat0),
              xytext=(y1 - 0.15, Qk),
              arrowprops=dict(arrowstyle="<->", color=C_DELTA,
                              lw=2.0, mutation_scale=14))
ax_m.text(y1 - 0.55, (f_xhat0 + Qk) / 2,
          rf"$\delta_0 = {delta0:.3f}$",
          color=C_DELTA, fontsize=11, ha="right", va="center",
          bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_DELTA, alpha=0.85))

ax_m.set_title(r"Step 3: Predicted decrease  $\delta_0 = f(\hat{x}_0) - "
               r"[\hat{f}_0(y_1) + \mu_0/2\,\Vert y_1-\hat{x}_0\Vert^2]$"
               "\n"
               rf"$\delta_0 = {f_xhat0:.3f} - {Qk:.3f} = {delta0:.3f}$",
               fontsize=10.5)
ax_m.legend(fontsize=8.5, loc="upper right")
draw_algo_panel(ax_a, active_steps=[2], active_lines={2: [1]})

fig.savefig(OUT_DIR / "frame_005.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ frame_005.png")

# ═════════════════════════════════════════════════════════════════════════════
# FRAME 006 — Step 4: stopping test δ₀ < δ̄ (fails → do NOT stop)
# ═════════════════════════════════════════════════════════════════════════════

fig, ax_m, ax_a = make_figure()
draw_base(ax_m, show_xhat0=True, show_y1=False,
          show_fhat0=True, show_envelope=True)

ax_m.axvline(y1, color=C_Y, ls=":", lw=0.9, alpha=0.5)
ax_m.plot(y1, Qk, "s", color=C_ENV, ms=9, zorder=9)
ax_m.axhline(f_xhat0, color=C_XHAT, ls=":", lw=0.8, alpha=0.45)

# δ₀ arrow
ax_m.annotate("", xy=(y1 - 0.15, f_xhat0),
              xytext=(y1 - 0.15, Qk),
              arrowprops=dict(arrowstyle="<->", color=C_DELTA,
                              lw=2.0, mutation_scale=14))

# δ̄ threshold line
ax_m.axhline(f_xhat0 - DELTA_BAR, color="#888888", ls="-.", lw=1.2,
             label=rf"$f(\hat{{x}}_0) - \bar\delta$  (stop threshold), $\bar\delta={DELTA_BAR}$")

# verdict box
verdict = rf"$\delta_0 = {delta0:.3f} \gg \bar\delta = {DELTA_BAR}$" + "\n" + r"$\Rightarrow$ Do NOT stop, continue"
ax_m.text(0.5, 0.60, verdict, transform=ax_m.transAxes,
          fontsize=12, ha="center", va="center",
          color="#1a6b1a", fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.5", fc="#e8f8e8", ec="#27ae60",
                    alpha=0.92))

ax_m.set_title(r"Step 4: Stopping test  $\delta_0 < \bar\delta$?"
               "\n"
               rf"$\delta_0 = {delta0:.3f}$,  $\bar\delta = {DELTA_BAR}$  "
               r"$\Rightarrow$ condition is FALSE → continue",
               fontsize=10.5)
ax_m.legend(fontsize=8.5, loc="upper right")
draw_algo_panel(ax_a, active_steps=[3])

fig.savefig(OUT_DIR / "frame_006.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ frame_006.png")

# ═════════════════════════════════════════════════════════════════════════════
# FRAME 007 — Step 5: oracle evaluated at y₁; f(y₁) and s₁ shown
# ═════════════════════════════════════════════════════════════════════════════

fig, ax_m, ax_a = make_figure()
draw_base(ax_m, show_xhat0=True, show_y1=False,
          show_fhat0=True, show_envelope=False)

# mark y₁ on the TRUE function curve
ax_m.axvline(y1, color=C_Y, ls=":", lw=0.9, alpha=0.4)
ax_m.plot(y1, f_y1, "D", color=C_Y, ms=10, zorder=9, label=rf"$f(y_1)={f_y1:.4f}$")
ax_m.text(y1 + 0.07, f_y1 + 0.18,
          rf"$y_1={y1:.3f}$" "\n" rf"$f(y_1)={f_y1:.4f}$",
          color=C_Y, fontsize=9.5)

# subgradient arrow at y₁
arrow_len = 0.35
ax_m.annotate("", xy=(y1 + arrow_len, f_y1 + s1 * arrow_len),
              xytext=(y1, f_y1),
              arrowprops=dict(arrowstyle="-|>", color="#1abc9c",
                              lw=1.8, mutation_scale=16))
ax_m.text(y1 + arrow_len + 0.05, f_y1 + s1 * arrow_len,
          rf"$s_1 = {s1:.2f}$", color="#1abc9c", fontsize=10)

# new linearisation at y₁ (the new cut, not yet added to model)
lin1_ys = np.array([f_y1 + s1 * (y - y1) for y in envelopes_ys])
ax_m.plot(envelopes_ys, lin1_ys, color="#1abc9c", lw=1.6, ls="--",
          alpha=0.6, zorder=3, label=rf"$f(y_1)+\langle s_1, \cdot-y_1\rangle$ (new cut)")

ax_m.set_title(r"Step 5: Oracle evaluation at $y_1$"
               "\n"
               rf"$f(y_1) = {f_y1:.4f}$,  $s_1 \in \partial f(y_1)$,  $s_1 = {s1:.4f}$",
               fontsize=10.5)
ax_m.legend(fontsize=8.5, loc="upper right")
draw_algo_panel(ax_a, active_steps=[4])

fig.savefig(OUT_DIR / "frame_007.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ frame_007.png")

# ═════════════════════════════════════════════════════════════════════════════
# FRAME 008 — Model accuracy: f(y₁) vs f̂₀(y₁) — the linearisation error
# ═════════════════════════════════════════════════════════════════════════════

fig, ax_m, ax_a = make_figure()
draw_base(ax_m, show_xhat0=True, show_y1=False,
          show_fhat0=True, show_envelope=False)

# marks at y₁ for both model and true value
model_at_y1 = fhat0(y1)
ax_m.axvline(y1, color=C_Y, ls=":", lw=0.9, alpha=0.4)
ax_m.plot(y1, f_y1,       "D", color=C_Y,     ms=10, zorder=10)
ax_m.plot(y1, model_at_y1, "s", color=C_MODEL, ms=10, zorder=10)

# vertical gap  (linearisation error / subgradient inequality)
lin_error = f_y1 - model_at_y1
ax_m.annotate("", xy=(y1 + 0.08, f_y1),
              xytext=(y1 + 0.08, model_at_y1),
              arrowprops=dict(arrowstyle="<->", color="#888888",
                              lw=1.8, mutation_scale=14))
ax_m.text(y1 + 0.12, (f_y1 + model_at_y1) / 2,
          rf"lin. error $= {lin_error:.3f}$"
          "\n" r"$= f(y_1) - \hat{f}_0(y_1) \geq 0$",
          fontsize=8.5, color="#555555", va="center")

ax_m.text(y1 - 0.08, f_y1 + 0.12,
          rf"$f(y_1) = {f_y1:.3f}$", color=C_Y, fontsize=9, ha="right")
ax_m.text(y1 - 0.08, model_at_y1 - 0.2,
          rf"$\hat{{f}}_0(y_1) = {model_at_y1:.3f}$", color=C_MODEL,
          fontsize=9, ha="right")

ax_m.set_title(r"Subgradient inequality: $f(y_1) \geq \hat{f}_0(y_1)$"
               "\n"
               rf"Model underestimates by {lin_error:.4f}  "
               r"(linearisation error $\geq 0$)",
               fontsize=10.5)
ax_m.legend(fontsize=8.5, loc="upper right")
draw_algo_panel(ax_a, active_steps=[4])

fig.savefig(OUT_DIR / "frame_008.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ frame_008.png")

print()
print("Phase 2 complete: frames 005–008 saved to", OUT_DIR)
print(f"  δ₀={delta0:.4f}  δ̄={DELTA_BAR}  →  continue")
print(f"  f(y₁)={f_y1:.4f}  s₁={s1:.4f}")
print()
print("Next: run phase3_steps6_8.py to add frames 009–013")
