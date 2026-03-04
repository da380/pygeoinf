"""
Bundle Method Visualisation — Phase 3: Steps 6–8 + second iteration
====================================================================

Picks up immediately after phase2 (frames 005–008) and covers:

    Step 6.  Serious/Null step decision  (k=0)
    Step 7.  Model update f̂₁ = max{f̂₀, new cut}  (k=0)
    Step 8.  k→k+1, go to Step 2  (loop start k=1)
    ── iteration k=1 ──────────────────────────────
    Step 2.  New envelope, trial point y₂ found
    Step 3.  Predicted decrease δ₁
    Step 5.  Oracle at y₂, subgradient s₂
    Step 6.  Null step decision  (k=1) — centre DOES NOT move
    Step 7.  Model update f̂₂ = max{f̂₀, cut₁, cut₂}  (k=1)

Frames saved:
    frame_009  – Step 6 (k=0): serious/null test; verdict SERIOUS STEP
    frame_010  – Step 6 (k=0): stability centre moves x̂₀ → x̂₁ = y₁
    frame_011  – Step 7 (k=0): updated model f̂₁ drawn (two-cut max)
    frame_012  – Step 8 / k=1 Step 2: new envelope at x̂₁, trial point y₂
    frame_013  – k=1 Steps 3–5: δ₁ and oracle output at y₂
    frame_014  – k=1 Step 6: NULL step, centre stays at x̂₁
    frame_015  – k=1 Step 7: model f̂₂ updated with third cut
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = Path(__file__).parent

# ── problem ────────────────────────────────────────────────────────────────
KNOTS   = np.array([0.3, 0.8, 1.4])
WEIGHTS = np.array([1.0, 2.0, 1.0])
X_STAR  = 0.8
F_STAR  = float(np.dot(WEIGHTS, np.abs(X_STAR - KNOTS)))

def f(x):
    return float(np.dot(WEIGHTS, np.abs(x - KNOTS)))

def subgradient(x):
    return float(np.dot(WEIGHTS, np.sign(x - KNOTS)))

# ── algorithm parameters ───────────────────────────────────────────────────
X_HAT0    = 2.2
MU        = 3.0
M_PARAM   = 0.5
DELTA_BAR = 1e-4

X_MIN, X_MAX = -0.3, 2.8
XS = np.linspace(X_MIN, X_MAX, 600)
FS = np.array([f(x) for x in XS])
envelopes_ys = np.linspace(X_MIN, X_MAX, 600)

# ── colour palette ─────────────────────────────────────────────────────────
C_F      = "#2c3e7a"
C_MODEL  = "#e07b39"
C_ENV    = "#8b3a9e"
C_XHAT   = "#c0392b"
C_Y      = "#27ae60"
C_MU     = "#7f8c8d"
C_DELTA  = "#e74c3c"
C_CUT1   = "#1abc9c"
C_CUT2   = "#d35400"
C_XHAT1  = "#8e44ad"   # for x̂₁ (distinct from x̂₀)

# ── iteration k=0 pre-compute ──────────────────────────────────────────────
xhat0   = X_HAT0
f_xhat0 = f(xhat0)
s0      = subgradient(xhat0)

def cut0(y): return f_xhat0 + s0 * (y - xhat0)

fhat0_ys = np.array([cut0(y) for y in envelopes_ys])
env0_ys  = np.clip(fhat0_ys + MU / 2 * (envelopes_ys - xhat0) ** 2,
                   -1, max(FS) * 1.4)

y1      = xhat0 - s0 / MU
f_y1    = f(y1)
s1      = subgradient(y1)

Qk0     = cut0(y1) + MU / 2 * (y1 - xhat0) ** 2
delta0  = f_xhat0 - Qk0

#  serious step check
ss_lhs0    = f_xhat0 - f_y1
ss_thresh0 = M_PARAM * delta0
serious0   = ss_lhs0 >= ss_thresh0
xhat1      = y1          # serious step → x̂₁ = y₁

# ── iteration k=1 pre-compute ──────────────────────────────────────────────
f_xhat1 = f(y1)          # same as f_y1

def cut1(y): return f_y1 + s1 * (y - y1)

def fhat1(y): return max(cut0(y), cut1(y))

fhat1_ys = np.array([fhat1(y) for y in envelopes_ys])
env1_ys  = np.clip(fhat1_ys + MU / 2 * (envelopes_ys - xhat1) ** 2,
                   -1, max(FS) * 1.4)

y2      = xhat1 - s1 / MU
f_y2    = f(y2)
s2      = subgradient(y2)

Qk1     = fhat1(y2) + MU / 2 * (y2 - xhat1) ** 2
delta1  = f_xhat1 - Qk1

ss_lhs1    = f_xhat1 - f_y2
ss_thresh1 = M_PARAM * delta1
serious1   = ss_lhs1 >= ss_thresh1
xhat2      = xhat1       # null step → x̂₂ = x̂₁

def cut2(y): return f_y2 + s2 * (y - y2)

def fhat2(y): return max(fhat1(y), cut2(y))

fhat2_ys = np.array([fhat2(y) for y in envelopes_ys])

# ── verify ────────────────────────────────────────────────────────────────
print("k=0:")
print(f"  y₁={y1:.4f}  f(y₁)={f_y1:.4f}  s₁={s1:.4f}")
print(f"  δ₀={delta0:.4f}")
print(f"  SS test: {ss_lhs0:.4f} >= m·δ₀={ss_thresh0:.4f} → {serious0}")
print(f"  x̂₁={xhat1:.4f}  (SERIOUS STEP)")
print("k=1:")
print(f"  y₂={y2:.4f}  f(y₂)={f_y2:.4f}  s₂={s2:.4f}")
print(f"  δ₁={delta1:.4f}")
print(f"  SS test: {ss_lhs1:.4f} >= m·δ₁={ss_thresh1:.4f} → {serious1}")
print(f"  x̂₂={xhat2:.4f}  (NULL STEP)")

# ═══════════════════════════════════════════════════════════════════════════
# Algorithm panel text (identical across all phases)
# ═══════════════════════════════════════════════════════════════════════════

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


def draw_base_k0(ax, show_xhat0=True, show_y1=False,
                 show_fhat0=False, show_envelope=False,
                 show_new_cut=False, dim_xhat0=False):
    """Shared base for k=0 frames."""
    ax.plot(XS, FS, color=C_F, lw=2.2, zorder=4, label=r"$f(x)$")
    ax.plot(X_STAR, F_STAR, "k*", ms=9, zorder=10, label=r"$x^*=0.8$")
    for ki in KNOTS:
        ax.axvline(ki, color="lightgrey", ls="--", lw=0.6, alpha=0.4)

    if show_fhat0:
        alpha = 0.25 if dim_xhat0 else 0.45
        ax.plot(envelopes_ys, fhat0_ys, color=C_MODEL, lw=1.6, ls="--",
                alpha=alpha, zorder=3, label=r"$\hat{f}_0$ (cut 0, $k{=}0$)")

    if show_envelope:
        ax.plot(envelopes_ys, env0_ys, color=C_ENV, lw=1.8, alpha=0.45,
                zorder=3, label=r"$\hat{f}_0 + \mu/2\,\Vert\cdot-\hat{x}_0\Vert^2$")

    if show_new_cut:
        cut1_ys = np.array([cut1(y) for y in envelopes_ys])
        ax.plot(envelopes_ys, cut1_ys, color=C_CUT1, lw=1.6, ls="--",
                alpha=0.55, zorder=3, label=r"$f(y_1)+\langle s_1,\cdot-y_1\rangle$ (new cut)")

    if show_xhat0:
        alpha_pt = 0.35 if dim_xhat0 else 1.0
        ax.plot(xhat0, f_xhat0, "o", color=C_XHAT, ms=9, zorder=9, alpha=alpha_pt)
        if not dim_xhat0:
            ax.text(xhat0 + 0.05, f_xhat0 + 0.15,
                    rf"$\hat{{x}}_0={xhat0}$", color=C_XHAT, fontsize=9)

    if show_y1:
        ax.plot(y1, f_y1, "D", color=C_Y, ms=8, zorder=9)
        ax.text(y1 + 0.05, f_y1 + 0.18,
                rf"$y_1={y1:.3f}$", color=C_Y, fontsize=9)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-0.5, max(FS) * 1.15)
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$f(x)$", fontsize=13)
    ax.grid(True, alpha=0.25)


def draw_base_k1(ax, show_fhat1=True, show_fhat0_dim=True,
                 show_envelope1=False, show_xhat1=True,
                 show_y1=True, show_y2=False, show_cut2=False,
                 show_fhat2=False):
    """Shared base for k=1 frames."""
    ax.plot(XS, FS, color=C_F, lw=2.2, zorder=4, label=r"$f(x)$")
    ax.plot(X_STAR, F_STAR, "k*", ms=9, zorder=10, label=r"$x^*=0.8$")
    for ki in KNOTS:
        ax.axvline(ki, color="lightgrey", ls="--", lw=0.6, alpha=0.4)

    if show_fhat0_dim:
        ax.plot(envelopes_ys, fhat0_ys, color=C_MODEL, lw=1.2, ls="--",
                alpha=0.20, zorder=2)

    if show_fhat1:
        ax.plot(envelopes_ys, fhat1_ys, color=C_MODEL, lw=1.9, ls="-",
                alpha=0.75, zorder=3, label=r"$\hat{f}_1 = \max\{\hat{f}_0,\,\mathrm{cut}_1\}$")

    if show_fhat2:
        ax.plot(envelopes_ys, fhat2_ys, color="#9b59b6", lw=2.2, ls="-",
                alpha=0.85, zorder=4, label=r"$\hat{f}_2 = \max\{\hat{f}_1,\,\mathrm{cut}_2\}$")

    if show_envelope1:
        ax.plot(envelopes_ys, env1_ys, color=C_ENV, lw=1.8, alpha=0.50,
                zorder=3, label=r"$\hat{f}_1 + \mu/2\,\Vert\cdot-\hat{x}_1\Vert^2$")

    if show_cut2:
        cut2_ys = np.array([cut2(y) for y in envelopes_ys])
        ax.plot(envelopes_ys, cut2_ys, color=C_CUT2, lw=1.5, ls="--",
                alpha=0.55, zorder=3, label=r"$f(y_2)+\langle s_2,\cdot-y_2\rangle$ (new cut)")

    if show_y1:
        ax.plot(y1, f_y1, "D", color=C_Y, ms=7, zorder=8, alpha=0.45)
        ax.text(y1 + 0.05, f_y1 + 0.15,
                rf"$y_1={y1:.3f}$", color=C_Y, fontsize=8.5, alpha=0.55)

    if show_xhat1:
        ax.plot(xhat1, f_xhat1, "o", color=C_XHAT1, ms=10, zorder=10)
        ax.text(xhat1 + 0.06, f_xhat1 + 0.18,
                rf"$\hat{{x}}_1={xhat1:.3f}$", color=C_XHAT1, fontsize=9.5)

    if show_y2:
        ax.plot(y2, f_y2, "D", color=C_CUT2, ms=10, zorder=9)
        ax.text(y2 + 0.06, f_y2 + 0.18,
                rf"$y_2={y2:.3f}$", color=C_CUT2, fontsize=9.5)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-0.5, max(FS) * 1.15)
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$f(x)$", fontsize=13)
    ax.grid(True, alpha=0.25)


# ═══════════════════════════════════════════════════════════════════════════
# FRAME 009 — Step 6 (k=0): Serious step test visualised
# ═══════════════════════════════════════════════════════════════════════════

fig, ax_m, ax_a = make_figure()
draw_base_k0(ax_m, show_xhat0=True, show_y1=True,
             show_fhat0=True, show_new_cut=True)

# horizontal reference lines for f(x̂₀) and f(y₁)
ax_m.axhline(f_xhat0, color=C_XHAT, ls=":", lw=1.0, alpha=0.5)
ax_m.axhline(f_y1,    color=C_Y,    ls=":", lw=1.0, alpha=0.5)

# actual improvement arrow dagger: f(x̂₀) - f(y₁)
ax_m.annotate("", xy=(xhat0 + 0.12, f_xhat0),
              xytext=(xhat0 + 0.12, f_y1),
              arrowprops=dict(arrowstyle="<->", color=C_DELTA, lw=2.2,
                              mutation_scale=15))
ax_m.text(xhat0 + 0.17, (f_xhat0 + f_y1) / 2,
          rf"$f(\hat{{x}}_0) - f(y_1)$" "\n" rf"$= {ss_lhs0:.3f}$",
          color=C_DELTA, fontsize=9.5, va="center",
          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_DELTA, alpha=0.85))

# threshold line  m·δ₀ dashed inside the big arrow
thresh_y = f_xhat0 - ss_thresh0
ax_m.axhline(thresh_y, color="#f39c12", ls="--", lw=1.5, alpha=0.9,
             label=rf"$f(\hat{{x}}_0) - m\,\delta_0 = {thresh_y:.3f}$  (threshold)")

# verdict box
verdict = (rf"$f(\hat{{x}}_0)-f(y_1) = {ss_lhs0:.3f}$" "\n"
           rf"$m\,\delta_0 = {M_PARAM}\times{delta0:.3f} = {ss_thresh0:.3f}$" "\n"
           r"$\Rightarrow$ " + r"$\mathbf{SERIOUS\ STEP}$" "\n"
           r"$\hat{x}_1 = y_1$")
ax_m.text(0.36, 0.55, verdict, transform=ax_m.transAxes,
          fontsize=11, ha="center", va="center",
          color="#155724", fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.55", fc="#d4edda", ec="#28a745", alpha=0.93))

ax_m.set_title(r"Step 6 ($k\!=\!0$): Serious step test"
               "\n"
               r"$f(\hat{x}_k) - f(y_{k+1}) \geq m\,\delta_k$?",
               fontsize=10.5)
ax_m.legend(fontsize=8, loc="upper right")
draw_algo_panel(ax_a, active_steps=[5], active_lines={5: [0, 1]})

fig.savefig(OUT_DIR / "frame_009.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ frame_009.png")

# ═══════════════════════════════════════════════════════════════════════════
# FRAME 010 — Step 6 (k=0) conclusion: stability centre moves x̂₀ → x̂₁ = y₁
# ═══════════════════════════════════════════════════════════════════════════

fig, ax_m, ax_a = make_figure()
draw_base_k0(ax_m, show_xhat0=True, show_y1=True,
             show_fhat0=True, dim_xhat0=True)

# faded old centre annotation
ax_m.text(xhat0 + 0.05, f_xhat0 + 0.15,
          rf"$\hat{{x}}_0={xhat0}$  (old)", color=C_XHAT, fontsize=8.5, alpha=0.35)

# big horizontal shift arrow
ax_m.annotate("", xy=(xhat1 + 0.03, f_xhat0 - 0.55),
              xytext=(xhat0 - 0.03, f_xhat0 - 0.55),
              arrowprops=dict(arrowstyle="-|>", color=C_XHAT1,
                              lw=2.5, mutation_scale=18))
ax_m.text((xhat0 + xhat1) / 2, f_xhat0 - 0.9,
          r"centre shifts to $y_1$",
          color=C_XHAT1, fontsize=10, ha="center")

# new centre marker
ax_m.plot(xhat1, f_xhat1, "o", color=C_XHAT1, ms=12, zorder=11,
          label=rf"$\hat{{x}}_1 = y_1 = {xhat1:.3f}$")
ax_m.text(xhat1 + 0.06, f_xhat1 + 0.22,
          rf"$\hat{{x}}_1 = {xhat1:.3f}$", color=C_XHAT1, fontsize=11,
          fontweight="bold")

ax_m.set_title(r"Step 6 ($k\!=\!0$): Serious Step — stability centre advances"
               "\n"
               rf"$\hat{{x}}_1 \leftarrow y_1 = {xhat1:.4f}$  "
               r"(significant function decrease achieved)",
               fontsize=10.5)
ax_m.legend(fontsize=8.5, loc="upper right")
draw_algo_panel(ax_a, active_steps=[5], active_lines={5: [1]})

fig.savefig(OUT_DIR / "frame_010.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ frame_010.png")

# ═══════════════════════════════════════════════════════════════════════════
# FRAME 011 — Step 7 (k=0): model update f̂₁ = max{cut₀, cut₁}
# ═══════════════════════════════════════════════════════════════════════════

fig, ax_m, ax_a = make_figure()
draw_base_k0(ax_m, show_xhat0=True, show_y1=True,
             show_fhat0=True, show_new_cut=True, dim_xhat0=True)

# new model f̂₁
ax_m.plot(envelopes_ys, fhat1_ys, color=C_MODEL, lw=2.8, ls="-",
          alpha=0.92, zorder=5,
          label=r"$\hat{f}_1 = \max\{\hat{f}_0,\,f(y_1)\!+\!\langle s_1,\cdot-y_1\rangle\}$")

# mark intersection of the two cuts
y_int = (f_y1 - s1 * y1 - f_xhat0 + s0 * xhat0) / (s0 - s1)
f_int = cut0(y_int)
ax_m.plot(y_int, f_int, "^", color="#f39c12", ms=9, zorder=11,
          label=rf"cut intersection @ $y={y_int:.3f}$")
ax_m.text(y_int + 0.05, f_int + 0.25,
          rf"cuts meet at $y={y_int:.2f}$", color="#f39c12", fontsize=9)

# annotate new centre (dimmed x̂₀, bright x̂₁)
ax_m.plot(xhat1, f_xhat1, "o", color=C_XHAT1, ms=10, zorder=10,
          label=rf"$\hat{{x}}_1={xhat1:.3f}$")
ax_m.text(xhat1 + 0.06, f_xhat1 + 0.20,
          rf"$\hat{{x}}_1={xhat1:.3f}$", color=C_XHAT1, fontsize=9.5)

ax_m.set_title(r"Step 7 ($k\!=\!0$): Model update  $\hat{f}_1 = \max\{\hat{f}_0,\; f(y_1)+\langle s_1, \cdot-y_1\rangle\}$"
               "\n"
               r"Growing piecewise-linear lower bound on $f$",
               fontsize=10.5)
ax_m.legend(fontsize=8, loc="upper right")
draw_algo_panel(ax_a, active_steps=[6], active_lines={6: [0]})

fig.savefig(OUT_DIR / "frame_011.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ frame_011.png")

# ═══════════════════════════════════════════════════════════════════════════
# FRAME 012 — Step 8 → k=1, Step 2: new envelope at x̂₁, trial point y₂
# ═══════════════════════════════════════════════════════════════════════════

fig, ax_m, ax_a = make_figure()
draw_base_k1(ax_m, show_fhat1=True, show_envelope1=True,
             show_xhat1=True, show_y1=True, show_y2=False)

# minimiser y₂ mark on envelope
ax_m.plot(y2, fhat1(y2) + MU / 2 * (y2 - xhat1) ** 2,
          "s", color=C_ENV, ms=10, zorder=11,
          label=rf"$Q_1(y_2) = {Qk1:.3f}$")

# proximal step arrow from x̂₁ to y₂
ax_m.annotate("", xy=(y2 + 0.03, f_xhat1 - 0.5),
              xytext=(xhat1 - 0.03, f_xhat1 - 0.5),
              arrowprops=dict(arrowstyle="-|>", color=C_Y,
                              lw=2.0, mutation_scale=15))
ax_m.text((xhat1 + y2) / 2 - 0.05, f_xhat1 - 0.85,
          rf"$y_2 = \hat{{x}}_1 - s_1/\mu = {y2:.3f}$",
          color=C_Y, fontsize=9.5, ha="center")

# y₂ on envelope
ax_m.plot(y2, f_y2, "D", color=C_Y, ms=9, zorder=9)
ax_m.text(y2 - 0.24, f_y2 + 0.20,
          rf"$y_2={y2:.3f}$", color=C_Y, fontsize=9.5)

ax_m.set_title(r"Step 8 $\to$ $k{=}1$, Step 2: new envelope of $\hat{f}_1$ at $\hat{x}_1$"
               "\n"
               rf"Trial point $y_2 = {y2:.4f}$  (proximal step from $\hat{{x}}_1$)",
               fontsize=10.5)
ax_m.legend(fontsize=8, loc="upper right")
draw_algo_panel(ax_a, active_steps=[1, 7], active_lines={1: [1]})

fig.savefig(OUT_DIR / "frame_012.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ frame_012.png")

# ═══════════════════════════════════════════════════════════════════════════
# FRAME 013 — k=1, Steps 3–5: δ₁, oracle at y₂
# ═══════════════════════════════════════════════════════════════════════════

fig, ax_m, ax_a = make_figure()
draw_base_k1(ax_m, show_fhat1=True, show_envelope1=True,
             show_xhat1=True, show_y1=True, show_y2=True)

# vertical line at y₂
ax_m.axvline(y2, color=C_CUT2, ls=":", lw=0.9, alpha=0.45)

# Q₁(y₂) marker
Q1_y2_val = fhat1(y2) + MU / 2 * (y2 - xhat1) ** 2
ax_m.plot(y2, Q1_y2_val, "s", color=C_ENV, ms=9, zorder=11)
ax_m.text(y2 + 0.07, Q1_y2_val - 0.30,
          rf"$Q_1(y_2)={Q1_y2_val:.3f}$", color=C_ENV, fontsize=8.5)

# f(x̂₁) dashed reference
ax_m.axhline(f_xhat1, color=C_XHAT1, ls=":", lw=0.9, alpha=0.5)

# δ₁ arrow
ax_m.annotate("", xy=(y2 + 0.12, f_xhat1),
              xytext=(y2 + 0.12, Q1_y2_val),
              arrowprops=dict(arrowstyle="<->", color=C_DELTA,
                              lw=2.0, mutation_scale=14))
ax_m.text(y2 + 0.17, (f_xhat1 + Q1_y2_val) / 2,
          rf"$\delta_1={delta1:.3f}$", color=C_DELTA, fontsize=11,
          va="center",
          bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_DELTA, alpha=0.85))

# subgradient arrow at y₂
arrow_x = 0.28
ax_m.annotate("", xy=(y2 + arrow_x, f_y2 + s2 * arrow_x),
              xytext=(y2, f_y2),
              arrowprops=dict(arrowstyle="-|>", color=C_CUT2,
                              lw=1.8, mutation_scale=16))
ax_m.text(y2 + arrow_x + 0.04, f_y2 + s2 * arrow_x,
          rf"$s_2={s2:.1f}$", color=C_CUT2, fontsize=10)

ax_m.set_title(r"$k{=}1$, Steps 3–5: predicted decrease $\delta_1$ and oracle at $y_2$"
               "\n"
               rf"$\delta_1={delta1:.4f}$,   $f(y_2)={f_y2:.4f}$,   $s_2={s2:.4f}$",
               fontsize=10.5)
ax_m.legend(fontsize=8, loc="upper right")
draw_algo_panel(ax_a, active_steps=[2, 3, 4], active_lines={2: [1]})

fig.savefig(OUT_DIR / "frame_013.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ frame_013.png")

# ═══════════════════════════════════════════════════════════════════════════
# FRAME 014 — k=1, Step 6: NULL step — centre does NOT move
# ═══════════════════════════════════════════════════════════════════════════

fig, ax_m, ax_a = make_figure()
draw_base_k1(ax_m, show_fhat1=True, show_xhat1=True,
             show_y1=True, show_y2=True)

# horizontal references
ax_m.axhline(f_xhat1, color=C_XHAT1, ls=":", lw=0.9, alpha=0.5)
ax_m.axhline(f_y2,    color=C_CUT2,  ls=":", lw=0.9, alpha=0.5)

# actual change arrow  f(x̂₁) - f(y₂)  — negative: f(y₂) > f(x̂₁)!
ax_m.annotate("", xy=(y2 + 0.12, f_xhat1),
              xytext=(y2 + 0.12, f_y2),
              arrowprops=dict(arrowstyle="<->", color=C_DELTA, lw=2.2,
                              mutation_scale=15))
ax_m.text(y2 + 0.17, (f_xhat1 + f_y2) / 2,
          rf"$f(\hat{{x}}_1)-f(y_2)$" "\n" rf"$= {ss_lhs1:.3f}$" "\n" r"(negative!)",
          color=C_DELTA, fontsize=9, va="center",
          bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_DELTA, alpha=0.85))

# threshold
thresh_y1 = f_xhat1 - ss_thresh1
ax_m.axhline(thresh_y1, color="#f39c12", ls="--", lw=1.5, alpha=0.88,
             label=rf"$f(\hat{{x}}_1) - m\,\delta_1 = {thresh_y1:.3f}$  (threshold)")

# verdict box — null step
verdict = (rf"$f(\hat{{x}}_1)-f(y_2) = {ss_lhs1:.3f}$" "\n"
           rf"$m\,\delta_1 = {M_PARAM}\times{delta1:.3f} = {ss_thresh1:.3f}$" "\n"
           r"$\Rightarrow$ " + r"$\mathbf{NULL\ STEP}$" "\n"
           r"$\hat{x}_2 = \hat{x}_1$ (centre stays)")
ax_m.text(0.42, 0.55, verdict, transform=ax_m.transAxes,
          fontsize=11, ha="center", va="center",
          color="#721c24", fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.55", fc="#f8d7da", ec="#dc3545", alpha=0.93))

ax_m.set_title(r"$k{=}1$, Step 6: Null step test"
               "\n"
               r"$f(\hat{x}_1)-f(y_2) < m\,\delta_1$  $\Rightarrow$  centre stays at $\hat{x}_1$",
               fontsize=10.5)
ax_m.legend(fontsize=8, loc="upper right")
draw_algo_panel(ax_a, active_steps=[5], active_lines={5: [2, 3]})

fig.savefig(OUT_DIR / "frame_014.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ frame_014.png")

# ═══════════════════════════════════════════════════════════════════════════
# FRAME 015 — k=1, Step 7: model grows with third cut; f̂₂ = max{f̂₁, cut₂}
# ═══════════════════════════════════════════════════════════════════════════

fig, ax_m, ax_a = make_figure()
draw_base_k1(ax_m, show_fhat1=False, show_fhat2=True, show_cut2=True,
             show_xhat1=True, show_y1=True, show_y2=True)

# also ghost f̂₁ for comparison
ax_m.plot(envelopes_ys, fhat1_ys, color=C_MODEL, lw=1.4, ls="--",
          alpha=0.30, zorder=2, label=r"$\hat{f}_1$ (prev model, dimmed)")

# annotate x̂₂ = x̂₁ (no movement)
ax_m.plot(xhat2, f(xhat2), "o", color=C_XHAT1, ms=12, zorder=11)
ax_m.text(xhat2 + 0.06, f(xhat2) + 0.22,
          rf"$\hat{{x}}_2 = \hat{{x}}_1 = {xhat2:.3f}$" "\n" r"(Null step: no move)",
          color=C_XHAT1, fontsize=9.5)

# summary text box in lower left
summary = ("Iteration summary:\n"
           rf"  $k=0$: Serious step  $\hat{{x}}_1={xhat1:.3f}$" "\n"
           rf"  $k=1$: Null step     $\hat{{x}}_2={xhat2:.3f}$" "\n"
           r"  Model: 3 cuts, tighter lower bound")
ax_m.text(0.02, 0.37, summary, transform=ax_m.transAxes,
          fontsize=9, va="top", linespacing=1.5,
          bbox=dict(boxstyle="round,pad=0.45", fc="#f0f4ff", ec="#5b6abf", alpha=0.92))

ax_m.set_title(r"$k{=}1$, Step 7: $\hat{f}_2 = \max\{\hat{f}_1,\; f(y_2)+\langle s_2,\cdot-y_2\rangle\}$"
               "\n"
               r"Bundle grows — model tightens around $f$ near optimum",
               fontsize=10.5)
ax_m.legend(fontsize=8, loc="upper right")
draw_algo_panel(ax_a, active_steps=[6], active_lines={6: [0]})

fig.savefig(OUT_DIR / "frame_015.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ frame_015.png")

print()
print("Phase 3 complete: frames 009–015 saved to", OUT_DIR)
print(f"  k=0: SERIOUS STEP → x̂₁={xhat1:.4f}")
print(f"  k=1: NULL STEP   → x̂₂={xhat2:.4f}  (centre holds)")
print()
print("Next: run phase4_animation.py to assemble all 15 frames into video")
