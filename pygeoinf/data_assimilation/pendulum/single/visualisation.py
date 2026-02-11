"""
visualisation.py

Visualisation tools specific to the Single Pendulum (2D system).
Handles phase portraits, physical animations, and Bayesian update plots.
"""

from typing import Optional, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection


from ... import core
from . import physics as phys


# --- Static Plots ---


def plot_bayesian_analysis(
    prior: core.ProbabilityGrid,
    likelihood: core.ProbabilityGrid,
    posterior: core.ProbabilityGrid,
    obs_val: float,
    obs_time: float,
) -> None:
    """
    Visualises the Bayesian update step for the single pendulum.
    Displays Prior, Likelihood, and Posterior side-by-side.

    Args:
        prior, likelihood, posterior: core.ProbabilityGrid objects.
        obs_val: The scalar observation value (theta).
        obs_time: The time at which observation occurred.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # Common args for the core plotter
    # We use dims=(0, 1) because the grid is 2D: (Theta, P)
    plot_args = {"dims": (0, 1), "filled": True, "levels": 30}

    # 1. Prior
    core.plot_grid_marginal(prior, ax=axes[0], cmap="viridis", **plot_args)
    axes[0].set_title(f"Prior PDF (t={obs_time:.1f}s)")
    axes[0].set_ylabel(r"$p_\theta$")

    # 2. Likelihood
    core.plot_grid_marginal(likelihood, ax=axes[1], cmap="plasma", **plot_args)
    axes[1].set_title("Likelihood")

    # Add domain-specific decoration (Observation Line)
    axes[1].axvline(
        obs_val, color="white", linestyle="--", lw=2, label=f"Obs: {obs_val:.2f}"
    )
    axes[1].legend(loc="upper right")

    # 3. Posterior
    core.plot_grid_marginal(posterior, ax=axes[2], cmap="viridis", **plot_args)
    axes[2].set_title("Posterior PDF")

    # Apply common domain labels
    for ax in axes:
        ax.set_xlabel(r"$\theta$")
        ax.set_aspect("auto")

    plt.suptitle("Bayesian Analysis Step", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_phase_portrait(
    ensemble_trajectories: np.ndarray, t_points: np.ndarray, title: Optional[str] = None
) -> None:
    """
    Plots the ensemble distribution at initial and final times.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # 1. Plot Initial (t index 0)
    core.plot_ensemble_scatter(
        ensemble_trajectories,
        dim_indices=(0, 1),
        time_idx=0,
        ax=ax1,
        c="royalblue",
        s=5,
        alpha=0.5,
    )
    ax1.set_title(f"Initial Distribution (t = {t_points[0]:.1f}s)")
    ax1.set_ylabel(r"$p_\theta$")

    # 2. Plot Final (t index -1)
    core.plot_ensemble_scatter(
        ensemble_trajectories,
        dim_indices=(0, 1),
        time_idx=-1,
        ax=ax2,
        c="firebrick",
        s=5,
        alpha=0.5,
    )
    ax2.set_title(f"Final Distribution (t = {t_points[-1]:.1f}s)")

    # 3. Domain Decoration
    p_max = np.max(np.abs(ensemble_trajectories[:, 1, :])) * 1.1

    for ax in (ax1, ax2):
        ax.set_xlabel(r"$\theta$")
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-p_max, p_max])
        ax.grid(True, alpha=0.3)

    if title:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()


def plot_trajectory_from_initial_condition(
    y0: np.ndarray,
    t_max: float,
    obs_data: Optional[Tuple] = None,
    physics_params: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
    color: str = "k",
    label: str = "Trajectory",
    dt_render: float = 0.02,
) -> plt.Axes:
    """
    Plots a smooth trajectory starting from y0, optionally overlaying observations.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if physics_params is None:
        physics_params = {"L": 1.0, "m": 1.0, "g": 1.0}

    # 1. Generate Smooth Curve (Integrate Physics)
    t_smooth = np.arange(0, t_max + dt_render / 2, dt_render)
    L, m, g = (
        physics_params.get("L", 1.0),
        physics_params.get("m", 1.0),
        physics_params.get("g", 1.0),
    )

    sol = core.solve_trajectory(phys.eom, y0, t_smooth, args=(L, m, g))
    theta_smooth = sol[0, :]

    # 2. Plot the Smooth Line
    ax.plot(t_smooth, theta_smooth, color=color, lw=2, label=label)

    # 3. Overlay Observations
    if obs_data is not None:
        t_obs, y_obs, y_std = obs_data
        ax.errorbar(
            t_obs,
            y_obs,
            yerr=y_std,
            fmt="o",
            color="red",
            alpha=0.6,
            capsize=3,
            markersize=5,
            label="Observations",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\theta$")
    ax.grid(True, alpha=0.3)

    return ax


# --- Animations ---


def animate_pendulum(
    t_points: np.ndarray, solution: np.ndarray, L: float = 1.0
) -> FuncAnimation:
    """
    Physical space animation (x, y) with a fading trail.
    """
    theta = solution[0, :]
    x, y = phys.get_coords(theta, L)

    fig, ax = plt.subplots(figsize=(6, 6))
    lim = 1.2 * L
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Single Pendulum")

    (line,) = ax.plot(
        [], [], "o-", lw=2, color="k", markerfacecolor="royalblue", markersize=8
    )
    trail = LineCollection([], linewidths=1.5, cmap="Blues")
    ax.add_collection(trail)
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    trail_len = 20

    def update(i):
        line.set_data([0, x[i]], [0, y[i]])

        start = max(0, i - trail_len)
        if i - start > 1:
            pts = np.column_stack([x[start : i + 1], y[start : i + 1]])
            segments = np.stack([pts[:-1], pts[1:]], axis=1)
            trail.set_segments(segments)
            trail.set_array(np.linspace(0, 1, len(segments)))

        time_text.set_text(f"t = {t_points[i]:.1f}s")
        return line, trail, time_text

    return FuncAnimation(fig, update, frames=len(t_points), interval=30, blit=True)


def animate_phase_portrait(
    ensemble_trajectories: np.ndarray, t_points: np.ndarray
) -> FuncAnimation:
    """
    Animates the ensemble of particles moving in phase space (Theta vs P).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate limits
    p_vals = ensemble_trajectories[:, 1, :]
    p_max = np.max(np.abs(p_vals)) * 1.1

    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-p_max, p_max])
    ax.set_xlabel(r"$\theta$ ")
    ax.set_ylabel(r"$p_\theta$")
    ax.grid(True)
    ax.set_title("Phase Space Evolution")

    scatter = ax.scatter([], [], alpha=0.5, s=10, c="royalblue")
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    def update(i):
        current_theta = core.wrap_angle(ensemble_trajectories[:, 0, i])
        current_p = ensemble_trajectories[:, 1, i]

        data = np.column_stack([current_theta, current_p])
        scatter.set_offsets(data)
        time_text.set_text(f"t = {t_points[i]:.2f}s")
        return scatter, time_text

    return FuncAnimation(fig, update, frames=len(t_points), interval=50, blit=True)


def animate_combined(
    t_points: np.ndarray, solution: np.ndarray, L: float = 1.0, stride: int = 1
) -> FuncAnimation:
    """
    Side-by-side animation:
    Left: Physical Motion (Real space)
    Right: Phase Space Trajectory (Theta vs P)
    """
    theta = solution[0, :]
    p = solution[1, :]
    x, y = phys.get_coords(theta, L)

    fig, (ax_phys, ax_phase) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Left: Physical ---
    ax_phys.set_xlim(-1.2 * L, 1.2 * L)
    ax_phys.set_ylim(-1.2 * L, 1.2 * L)
    ax_phys.set_aspect("equal")
    ax_phys.grid(True, alpha=0.3)
    ax_phys.set_title("Physical Motion")

    (line,) = ax_phys.plot(
        [], [], "o-", lw=2, color="k", markerfacecolor="firebrick", markersize=10
    )

    # --- Right: Phase Space ---
    ax_phase.set_xlim(-np.pi, np.pi)
    range_p = np.ptp(p)
    ax_phase.set_ylim(np.min(p) - 0.1 * range_p, np.max(p) + 0.1 * range_p)
    ax_phase.set_xlabel(r"$\theta$")
    ax_phase.set_ylabel(r"$p$")
    ax_phase.grid(True)
    ax_phase.set_title("Trajectory")

    # Handling wrap-around lines in phase space
    theta_wrapped = core.wrap_angle(theta)
    theta_plot = theta_wrapped.copy()
    diffs = np.abs(np.diff(theta_plot, prepend=theta_plot[0]))
    theta_plot[diffs > np.pi] = np.nan

    (trace,) = ax_phase.plot([], [], "-", color="royalblue", lw=1.5, alpha=0.6)
    (head,) = ax_phase.plot([], [], "o", color="royalblue")

    time_text = ax_phys.text(0.05, 0.9, "", transform=ax_phys.transAxes)

    def update(frame):
        # Physical
        line.set_data([0, x[frame]], [0, y[frame]])

        # Phase Space
        trace.set_data(theta_plot[: frame + 1], p[: frame + 1])
        head.set_data([theta_wrapped[frame]], [p[frame]])

        time_text.set_text(f"t = {t_points[frame]:.1f}s")
        return line, trace, head, time_text

    frames = range(0, len(t_points), stride)
    anim = FuncAnimation(fig, update, frames=frames, interval=30, blit=True)

    plt.close(fig)
    return anim


def animate_advection(
    pdf_func: Callable,
    t_points: np.ndarray,
    res: int = 100,
    x_lim: Tuple[float, float] = (-np.pi, np.pi),
    y_lim: Tuple[float, float] = (-2.5, 2.5),
    title: str = "Advection",
    t_start: float = 0.0,
    L: float = 1.0,
    m: float = 1.0,
    g: float = 1.0,
) -> FuncAnimation:
    """
    Lazy-evaluation animation of PDF advection (Liouville).
    """
    # 1. Setup Grid
    x_vals = np.linspace(x_lim[0], x_lim[1], res)
    y_vals = np.linspace(y_lim[0], y_lim[1], res)
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")

    # Stack X and Y, then flatten to 1D array
    grid_flat = np.stack([X.ravel(), Y.ravel()])
    y_grid_vectorized = grid_flat.reshape(-1)

    # 2. Define Vectorized EOM Wrapper
    def vectorized_eom(t, y_flat):
        y_reshaped = y_flat.reshape(2, -1)
        dydt = phys.eom(t, y_reshaped, L=L, m=m, g=g)
        return np.concatenate(dydt).reshape(-1)

    # 3. Setup Figure
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$p$")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")

    # Initial Plot (t=0)
    Z_0 = pdf_func(X, Y)
    mesh = ax.pcolormesh(X, Y, Z_0, cmap="viridis", shading="gouraud")

    fig.colorbar(mesh, ax=ax, label="Probability Density")
    title_text = ax.set_title(f"{title} (t={t_points[0]:.2f})")

    # 4. Update Function
    def update(frame_idx):
        current_t = t_points[frame_idx]

        if frame_idx == 0 or current_t == 0:
            Z_new = pdf_func(X, Y)
        else:
            t_span = np.array([current_t, 0.0])
            sol = core.solve_trajectory(
                vectorized_eom, y_grid_vectorized, t_span, rtol=1e-4, atol=1e-4
            )
            final_state_flat = sol[:, -1]
            origins = final_state_flat.reshape(2, res, res)
            Z_new = pdf_func(origins[0], origins[1])

        mesh.set_array(Z_new.ravel())
        title_text.set_text(f"{title} (t={t_start+current_t:.2f})")

        return mesh, title_text

    anim = FuncAnimation(fig, update, frames=len(t_points), interval=50, blit=False)
    plt.close(fig)
    return anim


def animate_linear_comparison(
    t_points: np.ndarray,
    sol_nl: np.ndarray,
    sol_l: np.ndarray,
    L: float = 1.0,
    theta0_deg: Optional[float] = None,
) -> FuncAnimation:
    """
    Side-by-side animation comparing the Non-linear (True) and Linear (Approx)
    physical motion.
    """
    fig, (ax_true, ax_lin) = plt.subplots(1, 2, figsize=(12, 6))

    for ax, title in zip([ax_true, ax_lin], ["Non-linear (True)", "Linear (Approx)"]):
        ax.set_xlim(-1.2 * L, 1.2 * L)
        ax.set_ylim(-1.2 * L, 1.2 * L)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(title)

    (line_true,) = ax_true.plot(
        [], [], "o-", lw=2, color="k", markerfacecolor="royalblue", markersize=10
    )
    (line_lin,) = ax_lin.plot(
        [], [], "o-", lw=2, color="r", markerfacecolor="firebrick", markersize=10
    )
    time_text = ax_true.text(0.05, 0.9, "", transform=ax_true.transAxes)

    def update(i):
        x_t, y_t = phys.get_coords(sol_nl[0, i], L)
        x_l, y_l = phys.get_coords(sol_l[0, i], L)

        line_true.set_data([0, x_t], [0, y_t])
        line_lin.set_data([0, x_l], [0, y_l])

        time_text.set_text(f"t = {t_points[i]:.1f}s")
        return line_true, line_lin, time_text

    title_str = "Linearisation Accuracy Comparison"
    if theta0_deg is not None:
        title_str += rf" ($\theta_{{0}} = {theta0_deg}^\circ$)"

    plt.suptitle(title_str, fontsize=14)

    anim = FuncAnimation(fig, update, frames=len(t_points), interval=50, blit=True)
    plt.close(fig)
    return anim
