"""support_function_figures.py
==========================

Recreate the support-function illustration panels using pygeoinf.

Produces:
  support_function.pdf   – panels (a) and (b) side by side
  support_function_c.pdf – panel (c): geometric meaning of h_C(q)

Usage::

    cd /home/adrian/PhD/Inferences/pygeoinf
    conda activate inferences
    python work/support_function_figures.py

Mathematical setup
------------------
Ellipsoid  E = { x ∈ R² : ((x₁ - cₓ) / a)² + ((x₂ - c_y) / b)² ≤ 1 }

This is equivalent to { x : <A(x - c), x - c> ≤ 1 } with
A = diag(1/a², 1/b²), so A⁻¹ = diag(a², b²).

Support function for unit direction q = (cos θ, sin θ):
  h_C(q) = <q, c> + sqrt(<q, A⁻¹ q>)
          = cₓ cos θ + c_y sin θ + sqrt(a² cos²θ + b² sin²θ)

Support point in direction q:
  x*(q) = c + (A⁻¹ q) / sqrt(<q, A⁻¹ q>)
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

import pygeoinf as inf

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.grid": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "support_function_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Ellipsoid parameters ──────────────────────────────────────────────────────
CX, CY = 1.0, 0.5   # centre
A_AXIS, B_AXIS = 2.0, 0.75   # semi-axes  (a horizontal, b vertical)
# h(0) = CX + A_AXIS = 3,  h(π) = -CX + A_AXIS = 1
# h(π/2) = CY + B_AXIS = 1.25,  h(3π/2) = -CY + B_AXIS = 0.25

# ── Build the pygeoinf support function ───────────────────────────────────────
H2 = inf.EuclideanSpace(2)

# Vectors in EuclideanSpace(2) are plain numpy arrays.
center = np.array([CX, CY])

A_mat = np.diag([1.0 / A_AXIS**2, 1.0 / B_AXIS**2])
Ainv_mat = np.diag([A_AXIS**2, B_AXIS**2])
# A^{-1/2}: for diag(a², b²), the sqrt is diag(a, b)
Ainv_sqrt_mat = np.diag([A_AXIS, B_AXIS])

h_C = inf.EllipsoidSupportFunction(
    H2,
    center,
    radius=1.0,
    shape_operator=inf.DenseMatrixLinearOperator(H2, H2, A_mat),
    inverse_operator=inf.DenseMatrixLinearOperator(H2, H2, Ainv_mat),
    inverse_sqrt_operator=inf.DenseMatrixLinearOperator(H2, H2, Ainv_sqrt_mat),
)


def h(theta: float) -> float:
    """Return h_C for unit direction at angle *theta* (rad)."""
    q = np.array([np.cos(theta), np.sin(theta)])
    return float(h_C(q))


def support_pt(theta: float) -> np.ndarray:
    """Return the support point of C for unit direction at angle *theta*."""
    q = np.array([np.cos(theta), np.sin(theta)])
    return h_C.support_point(q)


# ── Colour palette ────────────────────────────────────────────────────────────
ELLIPSE_EDGE = "#4a9e76"
ELLIPSE_FACE = "#d4f0e2"
LINE_TEAL = "#3a9e88"          # tangent lines near horizontal/vertical normals
LINE_BLUE = "#2a5e9a"          # tangent lines for diagonal normals
AXIS_COLOR = "#3a9e88"         # axis lines through origin

DOT_COLORS = [
    "#2ca08a",   # θ ≈ 0       right
    "#1e6e96",   # θ ≈ 2π/3   upper-left
    "#1a4c7e",   # θ = π       left
    "#1a7a7a",   # θ ≈ 4.5    lower
    "#3a7ac0",   # θ ≈ 5.5    lower-right
]

# ── Highlighted directions ────────────────────────────────────────────────────
# Five angles that give distinct support points around the ellipse.
THETA_MARKS = [0.0, 2 * np.pi / 3, np.pi, 4.5, 5.5]

# ── Ellipse boundary (for plotting) ──────────────────────────────────────────
_t = np.linspace(0, 2 * np.pi, 600)
EX = CX + A_AXIS * np.cos(_t)
EY = CY + B_AXIS * np.sin(_t)


# ─────────────────────────────────────────────────────────────────────────────
# Panel (a) – Convex set C with supporting hyperplanes
# ─────────────────────────────────────────────────────────────────────────────

def draw_panel_a(ax: plt.Axes) -> None:
    """Draw the ellipse C with supporting lines at THETA_MARKS."""
    # Ellipse fill and boundary
    ax.fill(EX, EY, color=ELLIPSE_FACE, zorder=1)
    ax.plot(EX, EY, color=ELLIPSE_EDGE, lw=1.8, zorder=2)

    EXTEND = 4.0

    for i, theta in enumerate(THETA_MARKS):
        xp = support_pt(theta)
        # unit normal (= direction θ)
        nx, ny = np.cos(theta), np.sin(theta)
        # tangent direction (perpendicular to normal)
        tx, ty = -ny, nx

        # Choose line colour: teal for near-vertical/horizontal normals, blue otherwise
        col = LINE_TEAL if (abs(ny) < 0.15) else LINE_BLUE

        # Supporting hyperplane: line through xp with direction (tx, ty)
        ax.plot(
            [xp[0] - EXTEND * tx, xp[0] + EXTEND * tx],
            [xp[1] - EXTEND * ty, xp[1] + EXTEND * ty],
            color=col, lw=0.9, ls="--", alpha=0.80, zorder=3,
        )

        # Mark support point on the ellipse (filled)
        ax.plot(xp[0], xp[1], "o", color=DOT_COLORS[i], ms=5.5, zorder=5)

        # Mark foot of perpendicular from origin to hyperplane (hollow)
        foot = h(theta) * np.array([nx, ny])
        ax.plot(foot[0], foot[1], "o", color=DOT_COLORS[i], ms=5.5,
                mfc="white", mew=1.5, zorder=5)

    # Angle annotation for THETA_MARKS[1] (θ = 2π/3)
    _theta_ann = THETA_MARKS[1]
    _foot_ann = h(_theta_ann) * np.array([np.cos(_theta_ann), np.sin(_theta_ann)])
    _col_ann = DOT_COLORS[1]
    # dotted line: origin → foot
    ax.plot([0, _foot_ann[0]], [0, _foot_ann[1]], color=_col_ann,
            lw=0.9, ls=":", zorder=4)
    # short horizontal reference line along positive x-axis
    _arc_r = 0.45
    ax.plot([0, _arc_r * 1.25], [0, 0], color=_col_ann, lw=0.9, zorder=4)
    # arc from 0 to _theta_ann
    _arc = mpatches.Arc((0, 0), 2 * _arc_r, 2 * _arc_r,
                        angle=0, theta1=0, theta2=np.degrees(_theta_ann),
                        color=_col_ann, lw=1.0, zorder=6)
    ax.add_patch(_arc)
    # θ label at mid-arc, close to the arc
    _mid = _theta_ann / 2
    ax.text(_arc_r * 1.18 * np.cos(_mid), _arc_r * 1.18 * np.sin(_mid),
            r"$\theta$", fontsize=10, color=_col_ann, ha="center", va="center")

    # Origin marker
    ax.plot(0, 0, "o", color="k", ms=4, zorder=6)
    ax.text(-0.22, -0.18, r"$0$", fontsize=11)

    # Label the set
    ax.text(
        CX, CY, r"$C$",
        ha="center", va="center", fontsize=13, color=ELLIPSE_EDGE, style="italic",
    )

    ax.set_xlim(-2.2, 3.8)
    ax.set_ylim(-1.5, 2.0)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$", rotation=0, labelpad=12)
    ax.set_title(r"(a) Convex set $C$ with supporting hyperplanes", fontsize=11)



# ─────────────────────────────────────────────────────────────────────────────
# Panel (b) – Support function h_C(θ)
# ─────────────────────────────────────────────────────────────────────────────

def draw_panel_b(ax: plt.Axes) -> None:
    """Plot the support function h_C(θ) over [0, 2π]."""
    thetas = np.linspace(0, 2 * np.pi, 1000)
    h_vals = np.array([h(t) for t in thetas])

    ax.plot(thetas, h_vals, color="#1a3c78", lw=1.8)

    for i, theta in enumerate(THETA_MARKS):
        hv = h(theta)
        ax.axvline(theta, color=DOT_COLORS[i], lw=0.7, ls="--", alpha=0.6)
        ax.plot(theta, hv, "o", color=DOT_COLORS[i], ms=5.5,
                mfc="white", mew=1.5, zorder=5)

    ax.set_xlim(-0.1, 2 * np.pi + 0.1)
    ax.set_ylim(0, 3.3)

    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(
        [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    )
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$h_C(\theta)$", rotation=0, labelpad=18)
    ax.set_title(r"(b) Support function $h_C(\theta)$", fontsize=11)


# ─────────────────────────────────────────────────────────────────────────────
# Panel (c) – Geometric meaning of h_C(q)
# ─────────────────────────────────────────────────────────────────────────────
# Shows: direction vector q, supporting hyperplane, support point, and the
# scalar value h_C(q) = <q, x*(q)> measured from the origin.

THETA_C = np.pi / 4   # direction used for the illustration (45°)


def draw_panel_c(ax: plt.Axes) -> None:
    """Draw the geometric interpretation of h_C(q)."""
    # Ellipse fill and boundary
    ax.fill(EX, EY, color=ELLIPSE_FACE, zorder=1)
    ax.plot(EX, EY, color=ELLIPSE_EDGE, lw=1.8, zorder=2)

    q_unit = np.array([np.cos(THETA_C), np.sin(THETA_C)])
    hval = h(THETA_C)
    xstar = support_pt(THETA_C)

    # Supporting hyperplane through xstar perpendicular to q_unit
    tx, ty = -q_unit[1], q_unit[0]
    EXTEND = 3.0
    ax.plot(
        [xstar[0] - EXTEND * tx, xstar[0] + EXTEND * tx],
        [xstar[1] - EXTEND * ty, xstar[1] + EXTEND * ty],
        color=LINE_BLUE, lw=1.2, ls="--", alpha=0.85, zorder=3,
    )

    # Foot of perpendicular from origin to supporting hyperplane = h_C(q) * q
    foot = hval * q_unit

    # Dashed line from origin to foot
    ax.plot([0, foot[0]], [0, foot[1]], color="#888888", lw=0.8, ls=":", zorder=3)

    # Dashed line from foot to support point (to show x* is not the foot)
    ax.plot(
        [foot[0], xstar[0]], [foot[1], xstar[1]],
        color="#aaaaaa", lw=0.8, ls=":", zorder=3,
    )

    # ── Arrows ──
    arrowstyle = dict(arrowstyle="-|>", lw=1.5, mutation_scale=14)

    # q direction arrow: from origin, length 1.1
    q_arrow_end = 1.1 * q_unit
    ax.annotate(
        "", xy=q_arrow_end, xytext=(0.0, 0.0),
        arrowprops=dict(color="#1a3c78", **arrowstyle),
        zorder=6,
    )
    ax.text(
        q_arrow_end[0] - 0.15, q_arrow_end[1] + 0.12,
        r"$\mathbf{q}$", fontsize=12, color="#1a3c78",
    )

    # h_C(q) arrow: from origin to foot of perpendicular
    ax.annotate(
        "", xy=foot, xytext=(0.0, 0.0),
        arrowprops=dict(color=LINE_TEAL, **arrowstyle),
        zorder=6,
    )
    # label placed on the CW (lower-right) side of the arrow to avoid overlap
    # with the ellipse boundary
    cw_off = 0.30 * np.array([q_unit[1], -q_unit[0]])   # 90° CW from q
    label_pos = 0.62 * foot + cw_off
    ax.text(
        label_pos[0], label_pos[1],
        r"$h_C(\mathbf{q})$", fontsize=12, color=LINE_TEAL,
        ha="center", va="center",
    )

    # ── Markers ──
    # Hollow circle at foot of perpendicular (closest point on hyperplane to origin)
    ax.plot(foot[0], foot[1], "o", color=LINE_TEAL, ms=6,
            mfc="white", mew=1.5, zorder=7)

    ax.plot(0, 0, "o", color="k", ms=4, zorder=7)
    ax.text(-0.22, -0.18, r"$0$", fontsize=11)

    ax.plot(xstar[0], xstar[1], "o", color="#1a3c78", ms=6, zorder=7)
    ax.text(
        xstar[0] + 0.10, xstar[1] + 0.12,
        r"$x^*(\mathbf{q})$", fontsize=11, color="#1a3c78",
    )

    # Label the set
    ax.text(
        CX, CY - 0.05, r"$C$",
        ha="center", va="center", fontsize=13, color=ELLIPSE_EDGE, style="italic",
    )

    ax.set_xlim(-2.5, 4.0)
    ax.set_ylim(-1.2, 2.5)
    ax.set_aspect("equal")

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$", rotation=0, labelpad=12)
    ax.set_title(
        r"(c) Geometric meaning of $h_C(\mathbf{q})$", fontsize=11
    )


# ─────────────────────────────────────────────────────────────────────────────
# Assemble and save figures
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Figure (a) ────────────────────────────────────────────────────────────
    fig_a, ax_a = plt.subplots(figsize=(5.5, 4.5))
    draw_panel_a(ax_a)
    path_a = os.path.join(OUT_DIR, "support_function_convex_set.pdf")
    fig_a.savefig(path_a, bbox_inches="tight")
    print(f"Saved  {path_a}")

    # ── Figure (b) ────────────────────────────────────────────────────────────
    fig_b, ax_b = plt.subplots(figsize=(5.5, 4.0))
    draw_panel_b(ax_b)
    path_b = os.path.join(OUT_DIR, "support_function_support_curve.pdf")
    fig_b.savefig(path_b, bbox_inches="tight")
    print(f"Saved  {path_b}")

    # ── Figure (c) ────────────────────────────────────────────────────────────
    fig_c, ax_c = plt.subplots(figsize=(5.5, 4.5))
    draw_panel_c(ax_c)
    path_c = os.path.join(OUT_DIR, "support_function_displacement.pdf")
    fig_c.savefig(path_c, bbox_inches="tight")
    print(f"Saved  {path_c}")

    plt.show()


if __name__ == "__main__":
    main()
