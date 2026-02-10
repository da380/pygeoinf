"""
visualisation.py

Visualisation tools for the Double Pendulum.
Handles 4D->2D projections, physical animations, and marginal PDF plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .. import core
from . import physics as phys


# --- 1. Marginal PDF Plots (Grid Based) ---


def plot_marginal_pdf(
    axes, grid_4d, dims=(0, 1), ax=None, title=None, labels=None, cmap="viridis"
):
    """
    Marginalizes a 4D grid down to 2 dimensions and plots on a specific axis.

    Args:
        axes: List of 4 1D arrays [th1, th2, p1, p2].
        grid_4d: The 4D probability density array.
        dims: Tuple of indices to KEEP (x_dim, y_dim).
        ax: Matplotlib Axes object. If None, a new figure is created.
        title: Plot title.
        labels: Tuple of (xlabel, ylabel). Auto-generated if None.
        cmap: Colormap name.

    Returns:
        ax: The axis object.
        contour: The contour plot object (useful for adding colorbars externally).
    """
    # 1. Marginalise using Core
    kept_axes, marginal_Z = core.marginalise_grid(grid_4d, axes, keep_indices=dims)

    # 2. Prepare Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    X, Y = np.meshgrid(kept_axes[0], kept_axes[1], indexing="ij")

    # 3. Plot Contour
    contour = ax.contourf(X, Y, marginal_Z, levels=30, cmap=cmap)

    # 4. Labels & Decoration
    default_labels = [r"$\theta_1$", r"$\theta_2$", r"$p_1$", r"$p_2$"]
    xlabel = labels[0] if labels else default_labels[dims[0]]
    ylabel = labels[1] if labels else default_labels[dims[1]]

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax, contour


def plot_both_phase_marginals(axes, grid_4d, t=None):
    """
    High-level helper: Plots the Phase Space (Theta vs P) for BOTH bobs side-by-side.
    Reuses plot_marginal_pdf internally.

    Left Panel: Bob 1 (Theta1 vs P1)
    Right Panel: Bob 2 (Theta2 vs P2)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: Bob 1 (Theta1 vs P1) ---
    # Indices: 0=Theta1, 2=P1
    _, c1 = plot_marginal_pdf(
        axes,
        grid_4d,
        dims=(0, 2),
        ax=ax1,
        title=r"Bob 1 Phase Space ($\theta_1, p_1$)",
        cmap="viridis",
    )
    fig.colorbar(c1, ax=ax1, label="Probability Density")

    # --- Right: Bob 2 (Theta2 vs P2) ---
    # Indices: 1=Theta2, 3=P2
    _, c2 = plot_marginal_pdf(
        axes,
        grid_4d,
        dims=(1, 3),
        ax=ax2,
        title=r"Bob 2 Phase Space ($\theta_2, p_2$)",
        cmap="plasma",
    )
    fig.colorbar(c2, ax=ax2, label="Probability Density")

    # Overall Title
    super_title = "Double Pendulum Phase Marginals"
    if t is not None:
        super_title += f" (t={t:.2f}s)"
    plt.suptitle(super_title, fontsize=16)

    plt.tight_layout()
    plt.show()


# --- 2. Phase Projections (Particle Based) ---


def plot_phase_projections(ensemble_trajectories, t_points, time_idx=-1):
    """
    Plots the ensemble particles projected onto (th1, p1) and (th2, p2) planes.
    """
    # Extract snapshot
    snap = ensemble_trajectories[:, :, time_idx]

    th1 = core.wrap_angle(snap[:, 0])
    th2 = core.wrap_angle(snap[:, 1])
    p1 = snap[:, 2]
    p2 = snap[:, 3]

    time = t_points[time_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Bob 1
    ax1.scatter(th1, p1, alpha=0.5, s=10, c="royalblue", label="Particles")
    ax1.set_title(f"Bob 1 Phase Space (t={time:.1f}s)")
    ax1.set_xlabel(r"$\theta_1$")
    ax1.set_ylabel(r"$p_1$")
    ax1.set_xlim([-np.pi, np.pi])
    ax1.grid(True)

    # Plot Bob 2
    ax2.scatter(th2, p2, alpha=0.5, s=10, c="firebrick", label="Particles")
    ax2.set_title(f"Bob 2 Phase Space (t={time:.1f}s)")
    ax2.set_xlabel(r"$\theta_2$")
    ax2.set_ylabel(r"$p_2$")
    ax2.set_xlim([-np.pi, np.pi])
    ax2.grid(True)

    plt.suptitle("Double Pendulum Ensemble Projections", fontsize=16)
    plt.tight_layout()
    plt.show()


# --- 3. Animations ---


def animate_pendulum(t_points, solution, L1=1.0, L2=1.0):
    """
    Physical space animation for the Double Pendulum.
    """
    x1, y1, x2, y2 = phys.get_coords(solution[0], solution[1], L1, L2)

    fig, ax = plt.subplots(figsize=(6, 6))
    lim = (L1 + L2) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Double Pendulum")

    (rods,) = ax.plot([], [], "k-", lw=2)
    (bob1,) = ax.plot([], [], "o", markersize=8, c="royalblue")
    (bob2,) = ax.plot([], [], "o", markersize=8, c="firebrick")
    (trail,) = ax.plot([], [], "-", lw=1, c="gray", alpha=0.5)

    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)
    trail_len = 100

    def update(i):
        rods.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        bob1.set_data([x1[i]], [y1[i]])
        bob2.set_data([x2[i]], [y2[i]])

        start = max(0, i - trail_len)
        trail.set_data(x2[start:i], y2[start:i])

        time_text.set_text(f"t = {t_points[i]:.1f}s")
        return rods, bob1, bob2, trail, time_text

    return FuncAnimation(fig, update, frames=len(t_points), interval=40, blit=True)
