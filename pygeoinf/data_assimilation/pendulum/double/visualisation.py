"""
visualisation.py

Visualisation tools for the Double Pendulum.
Focuses on Chaotic Sensitivity (Divergence) and Ensemble particle clouds.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Tuple, Any

from ... import core
from . import physics as phys


# --- 1. Chaos & Sensitivity ---


def plot_sensitivity_divergence(
    t_points: np.ndarray,
    traj_ref: np.ndarray,
    traj_pert: np.ndarray,
    title: str = "Sensitivity to Initial Conditions",
) -> None:
    """
    Plots the divergence between two trajectories to demonstrate chaos.
    Top panel: Time series comparison of Theta 1 & 2.
    Bottom panel: Log-scale Euclidean distance between states.

    Args:
        t_points: Time array.
        traj_ref: Reference trajectory (4, N_time).
        traj_pert: Perturbed trajectory (4, N_time).
    """
    # Calculate Euclidean distance in state space
    diff = traj_ref - traj_pert
    dist = np.linalg.norm(diff, axis=0)

    fig, (ax_ts, ax_err) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: Time Series (Theta 1 only for clarity)
    ax_ts.plot(t_points, core.wrap_angle(traj_ref[0]), "k-", label="Reference")
    ax_ts.plot(
        t_points, core.wrap_angle(traj_pert[0]), "r--", alpha=0.8, label="Perturbed"
    )
    ax_ts.set_ylabel(r"$\theta_1$")
    ax_ts.set_title(title)
    ax_ts.legend(loc="upper right")
    ax_ts.grid(True, alpha=0.3)

    # Bottom: Error Growth (Lyapunov hint)
    ax_err.semilogy(t_points, dist, "b-", lw=1.5)
    ax_err.set_ylabel("State Euclidean Distance (Log Scale)")
    ax_err.set_xlabel("Time [s]")
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.set_title("Divergence of Trajectories")

    plt.tight_layout()
    plt.show()


# --- 2. Ensemble Projections (Particle Clouds) ---


def plot_ensemble_phase_space(
    ensemble_trajectories: np.ndarray, t_points: np.ndarray, time_idx: int = -1
) -> None:
    """
    Plots the ensemble particles projected onto (th1, p1) and (th2, p2) planes.
    Uses generic core scatter tools but configured for the 4D Double Pendulum.

    Args:
        ensemble_trajectories: Shape (N_samples, 4, N_time).
        t_points: Time array.
        time_idx: Index of time to plot.
    """
    time = t_points[time_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Bob 1: Theta1 (idx 0) vs P1 (idx 2) ---
    core.plot_ensemble_scatter(
        ensemble_trajectories,
        dim_indices=(0, 2),  # th1, p1
        time_idx=time_idx,
        ax=ax1,
        c="royalblue",
        s=10,
        alpha=0.6,
    )
    ax1.set_title(f"Bob 1 Phase Space (t={time:.1f}s)")
    ax1.set_xlabel(r"$\theta_1$")
    ax1.set_ylabel(r"$p_1$")
    ax1.set_xlim([-np.pi, np.pi])

    # --- Bob 2: Theta2 (idx 1) vs P2 (idx 3) ---
    core.plot_ensemble_scatter(
        ensemble_trajectories,
        dim_indices=(1, 3),  # th2, p2
        time_idx=time_idx,
        ax=ax2,
        c="firebrick",
        s=10,
        alpha=0.6,
    )
    ax2.set_title(f"Bob 2 Phase Space (t={time:.1f}s)")
    ax2.set_xlabel(r"$\theta_2$")
    ax2.set_ylabel(r"$p_2$")
    ax2.set_xlim([-np.pi, np.pi])

    plt.suptitle("Double Pendulum Ensemble Distribution", fontsize=16)
    plt.tight_layout()
    plt.show()


# --- 3. Animations ---


def animate_pendulum(
    t_points: np.ndarray,
    solution: np.ndarray,
    L1: float = 1.0,
    L2: float = 1.0,
    trail_len: int = 50,
) -> FuncAnimation:
    """
    Physical space animation for the Double Pendulum.
    Includes a fading trail for the second bob to visualize chaotic paths.
    """
    # solution shape: (4, N_time) -> [th1, th2, p1, p2]
    x1, y1, x2, y2 = phys.get_coords(solution[0], solution[1], L1, L2)

    fig, ax = plt.subplots(figsize=(6, 6))
    lim = (L1 + L2) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Double Pendulum Motion")

    (rods,) = ax.plot([], [], "k-", lw=2)
    (bob1,) = ax.plot([], [], "o", markersize=8, c="royalblue", zorder=3)
    (bob2,) = ax.plot([], [], "o", markersize=8, c="firebrick", zorder=3)
    (trail1,) = ax.plot([], [], "-", lw=1, c="royalblue", alpha=0.4)
    (trail2,) = ax.plot([], [], "-", lw=1, c="firebrick", alpha=0.4)

    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    def update(i):
        # Update Rods
        rods.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        bob1.set_data([x1[i]], [y1[i]])
        bob2.set_data([x2[i]], [y2[i]])

        # Update Trail
        start = max(0, i - trail_len)
        trail1.set_data(x1[start:i], y1[start:i])
        trail2.set_data(x2[start:i], y2[start:i])

        time_text.set_text(f"t = {t_points[i]:.1f}s")
        return rods, bob1, bob2, trail1, trail2, time_text

    return FuncAnimation(fig, update, frames=len(t_points), interval=40, blit=True)


def animate_ensemble_phase_space(
    t_points: np.ndarray,
    ensemble_trajectories: np.ndarray,
) -> FuncAnimation:
    """
    Animates the ensemble of particles moving in the two phase planes:
    (Theta1 vs P1) and (Theta2 vs P2).

    Args:
        t_points: Time array.
        ensemble_trajectories: Array of shape (N_samples, 4, N_time).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pre-calculate limits to keep axes fixed
    p1_max = np.max(np.abs(ensemble_trajectories[:, 2, :])) * 1.1
    p2_max = np.max(np.abs(ensemble_trajectories[:, 3, :])) * 1.1

    # Setup Bob 1 Axis
    ax1.set_xlim([-np.pi, np.pi])
    ax1.set_ylim([-p1_max, p1_max])
    ax1.set_xlabel(r"$\theta_1$")
    ax1.set_ylabel(r"$p_1$")
    ax1.set_title("Bob 1 Phase Space")
    ax1.grid(True, alpha=0.3)
    scatter1 = ax1.scatter([], [], alpha=0.6, s=10, c="royalblue")

    # Setup Bob 2 Axis
    ax2.set_xlim([-np.pi, np.pi])
    ax2.set_ylim([-p2_max, p2_max])
    ax2.set_xlabel(r"$\theta_2$")
    ax2.set_ylabel(r"$p_2$")
    ax2.set_title("Bob 2 Phase Space")
    ax2.grid(True, alpha=0.3)
    scatter2 = ax2.scatter([], [], alpha=0.6, s=10, c="firebrick")

    time_text = ax1.text(0.05, 0.9, "", transform=ax1.transAxes)

    def update(i):
        # Bob 1 Data: (Theta1, P1)
        th1 = core.wrap_angle(ensemble_trajectories[:, 0, i])
        p1 = ensemble_trajectories[:, 2, i]
        scatter1.set_offsets(np.column_stack([th1, p1]))

        # Bob 2 Data: (Theta2, P2)
        th2 = core.wrap_angle(ensemble_trajectories[:, 1, i])
        p2 = ensemble_trajectories[:, 3, i]
        scatter2.set_offsets(np.column_stack([th2, p2]))

        time_text.set_text(f"t = {t_points[i]:.2f}s")
        return scatter1, scatter2, time_text

    return FuncAnimation(fig, update, frames=len(t_points), interval=50, blit=True)
