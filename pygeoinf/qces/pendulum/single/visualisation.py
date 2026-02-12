"""
visualisation.py

Visualisation tools specific to the Single Pendulum (2D system).
Handles phase portraits, physical animations, and Bayesian update plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse

from scipy.stats import multivariate_normal

from .. import core
from . import physics as phys


# --- Static Plots ---


def plot_bayesian_analysis(X, Y, prior, likelihood, posterior, obs_val, obs_time):
    """
    Visualizes the Bayesian update step for the single pendulum.
    Displays Prior, Likelihood, and Posterior side-by-side.

    Args:
        X, Y: Meshgrids for Theta and Momentum.
        prior: 2D array of prior PDF values.
        likelihood: 2D array of likelihood values.
        posterior: 2D array of posterior PDF values.
        obs_val: The scalar observation value (theta).
        obs_time: The time at which observation occurred.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    def plot_panel(ax, data, title, cmap="viridis"):
        im = ax.contourf(X, Y, data, levels=30, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel(r"$\theta$ ")
        ax.grid(True, alpha=0.3)
        return im

    # 1. Prior
    plot_panel(axes[0], prior, f"Prior PDF (t={obs_time:.1f}s)")
    axes[0].set_ylabel(r"$p_\theta$")

    # 2. Likelihood
    plot_panel(axes[1], likelihood, "Likelihood", cmap="plasma")
    # Draw observation line
    axes[1].axvline(
        obs_val, color="white", linestyle="--", lw=2, label=f"Obs: {obs_val:.2f}"
    )
    axes[1].legend(loc="upper right")

    # 3. Posterior
    plot_panel(axes[2], posterior, "Posterior PDF")

    for ax in axes:
        ax.set_aspect("equal")

    plt.suptitle("Bayesian Analysis Step", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_phase_portrait(ensemble_trajectories, t_points, title=None):
    """
    Plots the ensemble distribution at initial and final times.

    Args:
        ensemble_trajectories: Array of shape (N_samples, 2, N_time).
        t_points: Time array.
    """
    # Extract states
    theta_0 = core.wrap_angle(ensemble_trajectories[:, 0, 0])
    p_0 = ensemble_trajectories[:, 1, 0]

    theta_f = core.wrap_angle(ensemble_trajectories[:, 0, -1])
    p_f = ensemble_trajectories[:, 1, -1]

    # Determine plot limits based on max momentum
    p_max = np.max(np.abs(ensemble_trajectories[:, 1, :])) * 1.1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot Initial
    ax1.scatter(theta_0, p_0, alpha=0.5, s=5, c="royalblue")
    ax1.set_title(f"Initial Distribution (t = {t_points[0]:.1f}s)")
    ax1.set_ylabel(r"$p_\theta$")

    # Plot Final
    ax2.scatter(theta_f, p_f, alpha=0.5, s=5, c="firebrick")
    ax2.set_title(f"Final Distribution (t = {t_points[-1]:.1f}s)")

    for ax in (ax1, ax2):
        ax.set_xlabel(r"$\theta$ ")
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-p_max, p_max])
        ax.grid(True, alpha=0.3)

    if title:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()


def plot_ensemble_stats(ensemble_trajectories, t_points):
    """
    Plots initial and final distributions with Mean and 2-Sigma Confidence Ellipses.
    """

    def add_confidence_ellipse(ax, states, color="black", label=None):
        mean = np.mean(states, axis=0)
        cov = np.cov(states, rowvar=False)
        vals, vecs = np.linalg.eig(cov)

        # Calculate angle of ellipse
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 4 * np.sqrt(vals)  # 2-sigma width/height

        ell = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            edgecolor=color,
            facecolor="none",
            lw=2,
            linestyle="--",
            label=label,
        )
        ax.add_patch(ell)
        ax.plot(mean[0], mean[1], "k*", markersize=12, label="Mean")

    # Data Prep
    states_0 = ensemble_trajectories[:, :, 0]
    states_f = ensemble_trajectories[:, :, -1]

    # Note: wrapping applies to scatter visualization
    theta_0 = core.wrap_angle(states_0[:, 0])
    theta_f = core.wrap_angle(states_f[:, 0])

    p_max = np.max(np.abs(ensemble_trajectories[:, 1, :])) * 1.1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Initial
    ax1.scatter(theta_0, states_0[:, 1], alpha=0.3, s=5, c="royalblue")
    add_confidence_ellipse(ax1, states_0, label=r"2$\sigma$")
    ax1.set_title(f"Initial (t={t_points[0]:.1f}s)")

    # Final
    ax2.scatter(theta_f, states_f[:, 1], alpha=0.3, s=5, c="firebrick")
    add_confidence_ellipse(ax2, states_f, label=r"2$\sigma$")
    ax2.set_title(f"Final (t={t_points[-1]:.1f}s)")

    for ax in (ax1, ax2):
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-p_max, p_max])
        ax.set_xlabel(r"$\theta$ ")
        ax.set_ylabel(r"$p_\theta$")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


def evaluate_gaussian_pdf(X_grid, Y_grid, mean, cov):
    """
    Evaluates a Multivariate Gaussian PDF using scipy.stats.
    Matches the indexing='ij' convention from the core engine.

    Args:
        X_grid (THETA): 2D meshgrid of angle values.
        Y_grid (P): 2D meshgrid of momentum values.
        mean: State vector [theta_bar, p_bar].
        cov: 2x2 covariance matrix.
    """
    # Stack grids to create an (N, M, 2) array of coordinate pairs
    # This ensures the evaluator 'sees' [theta, p] for every point
    pos = np.stack([X_grid, Y_grid], axis=-1)

    # Evaluate using scipy.stats
    return multivariate_normal.pdf(pos, mean=mean, cov=cov)


def plot_gaussian_2d(ax, mean, cov, x_lim, y_lim, res=150, title="Gaussian PDF"):
    """
    Plots a 2D Gaussian PDF aligned with the indexing='ij' convention.
    """
    from scipy.stats import multivariate_normal

    # 1. Generate axes using 'ij' to match core.advect_pdf_grid
    theta_vals = np.linspace(x_lim[0], x_lim[1], res)
    p_vals = np.linspace(y_lim[0], y_lim[1], res)
    THETA, P = np.meshgrid(theta_vals, p_vals, indexing="ij")  #

    # 2. Stack as (N, M, 2) where the last dimension is exactly [theta, p]
    pos = np.stack([THETA, P], axis=-1)

    # 3. Evaluate the PDF
    rv = multivariate_normal(mean, cov)
    Z = rv.pdf(pos)

    # 4. Plot using the ij meshgrids
    contour = ax.contourf(THETA, P, Z, levels=30, cmap="viridis")

    # 5. Fix Grid Lines: Ensure they are drawn on top of the contour
    ax.set_axisbelow(False)  # Forces grid lines to the front
    ax.grid(True, color="white", alpha=0.3, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$p$")
    return contour


def plot_pdf(
    X,
    Y,
    Z,
    ax=None,
    title="Probability Density",
    xlabel=r"$\theta$ ",
    ylabel=r"$p$ ",
    cmap="viridis",
):
    """
    Simple helper to plot a 2D PDF contour.

    Args:
        X, Y: Meshgrids of coordinates.
        Z: Density values.
        ax: Matplotlib axes to plot on. If None, creates a new figure.
        title: Plot title.
        xlabel, ylabel: Axis labels.
        cmap: Colormap.

    Returns:
        ax: The axis object.
        contour: The contour set (useful for adding a colorbar).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    contour = ax.contourf(X, Y, Z, levels=30, cmap=cmap)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    # If we created the figure, add a colorbar automatically for convenience
    if (
        ax.figure.axes[-1] == ax
    ):  # Checks if colorbar already exists usually, but simpler:
        # Only add colorbar if we created the figure context to avoid layout issues in subplots
        pass

    return ax, contour


def plot_trajectory_from_initial_condition(
    y0,
    t_max,
    obs_data=None,
    physics_params=None,
    ax=None,
    color="k",
    label="Trajectory",
    dt_render=0.02,
):
    """
    Plots a smooth trajectory starting from y0, optionally overlaying observations.

    Args:
        y0 (array): Initial state [theta, p] at t=0.
        t_max (float): End time for the plot.
        obs_data (tuple): Optional (t_obs, y_obs, y_std) for scatter plot.
        physics_params (dict): L, m, g.
        ax (matplotlib.axes): Axis to plot on.
        color (str): Color of the smooth line.
        label (str): Legend label.
        dt_render (float): Time step for the smooth curve integration.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Generate Smooth Curve (Integrate Physics)
    # We create a dense time array from 0 to t_max
    t_smooth = np.arange(0, t_max + dt_render / 2, dt_render)

    # Package args for the solver
    L, m, g = physics_params["L"], physics_params["m"], physics_params["g"]

    # Solve trajectory
    # Note: We assume y0 is at t=0. If your simulation starts elsewhere,
    # you'd need a t_start argument.
    sol = core.solve_trajectory(phys.eom, y0, t_smooth, args=(L, m, g))
    theta_smooth = sol[0, :]

    # 2. Plot the Smooth Line
    ax.plot(t_smooth, theta_smooth, color=color, lw=2, label=label)

    # 3. Overlay Observations (if provided)
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
    ax.set_ylabel(r"$\theta$ (rad)")
    ax.grid(True, alpha=0.3)

    return ax


# --- Animations ---


def animate_pendulum(t_points, solution, L=1.0):
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

    # Objects
    (line,) = ax.plot(
        [], [], "o-", lw=2, color="k", markerfacecolor="royalblue", markersize=8
    )
    trail = LineCollection([], linewidths=1.5, cmap="Blues")
    ax.add_collection(trail)
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    # Trail setup
    trail_len = 20

    def update(i):
        # Current Rod
        line.set_data([0, x[i]], [0, y[i]])

        # Trail
        start = max(0, i - trail_len)
        if i - start > 1:
            pts = np.column_stack([x[start : i + 1], y[start : i + 1]])
            segments = np.stack([pts[:-1], pts[1:]], axis=1)
            trail.set_segments(segments)
            # Fade alpha
            trail.set_array(np.linspace(0, 1, len(segments)))

        time_text.set_text(f"t = {t_points[i]:.1f}s")
        return line, trail, time_text

    return FuncAnimation(fig, update, frames=len(t_points), interval=30, blit=True)


def animate_phase_portrait(ensemble_trajectories, t_points):
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
        # Extract frame i
        current_theta = core.wrap_angle(ensemble_trajectories[:, 0, i])
        current_p = ensemble_trajectories[:, 1, i]

        data = np.column_stack([current_theta, current_p])
        scatter.set_offsets(data)
        time_text.set_text(f"t = {t_points[i]:.2f}s")
        return scatter, time_text

    return FuncAnimation(fig, update, frames=len(t_points), interval=50, blit=True)


def animate_combined(t_points, solution, L=1.0, stride=1):
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
    # We plot the full history, but we need to insert NaNs where it wraps
    # so we don't draw horizontal lines across the plot.
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
    pdf_func,
    t_points,
    res=100,
    x_lim=(-np.pi, np.pi),
    y_lim=(-2.5, 2.5),
    title="Advection",
    L=1.0,
    m=1.0,
    g=1.0,
):  # <--- Note g=1.0 default
    """
    Lazy-evaluation animation of PDF advection (Liouville).
    Computes each frame on the fly by integrating backwards from t to 0.

    Args:
        pdf_func: Callable prior f(theta, p).
        t_points: Array of time points to animate.
        res: Grid resolution.
        x_lim, y_lim: Plot limits.
        L, m, g: Physics parameters. Default g=1.0 for normalized smooth animation.
    """
    # 1. Setup Grid
    x_vals = np.linspace(x_lim[0], x_lim[1], res)
    y_vals = np.linspace(y_lim[0], y_lim[1], res)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Pre-calculate flattened grid for the solver
    # Stack X and Y, then flatten to 1D array [x0, x1... p0, p1...]
    grid_flat = np.stack([X.ravel(), Y.ravel()])
    y_grid_vectorized = grid_flat.reshape(-1)

    # 2. Define Vectorized EOM Wrapper
    # This maps the flat solver state back to (2, N) for the physics engine
    def vectorized_eom(t, y_flat):
        y_reshaped = y_flat.reshape(2, -1)
        dydt = phys.eom(t, y_reshaped, L=L, m=m, g=g)
        return np.concatenate(dydt).reshape(-1)

    # 3. Setup Figure
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel(r"$\theta$ ")
    ax.set_ylabel(r"$p$")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Initial Plot (t=0)
    Z_0 = pdf_func(X, Y)
    # shading='gouraud' creates a smooth interpolated look
    mesh = ax.pcolormesh(X, Y, Z_0, cmap="viridis", shading="gouraud")

    fig.colorbar(mesh, ax=ax, label="Probability Density")
    title_text = ax.set_title(f"{title} (t={t_points[0]:.2f})")

    # 4. Update Function (The "Lazy" Part)
    def update(frame_idx):
        current_t = t_points[frame_idx]

        # Frame 0 is always just the Prior
        if frame_idx == 0 or current_t == 0:
            Z_new = pdf_func(X, Y)
        else:
            # Backwards Integration: Solve from current_t -> 0
            t_span = [current_t, 0.0]

            sol = core.solve_trajectory(
                vectorized_eom, y_grid_vectorized, t_span, rtol=1e-4, atol=1e-4
            )

            # The solution at t=0 is the LAST column of the output
            final_state_flat = sol[:, -1]

            # Reshape back to (2, res, res) to get origin coordinates
            origins = final_state_flat.reshape(2, res, res)

            # Sample the Prior at these origin points
            Z_new = pdf_func(origins[0], origins[1])

        # Update the plot data
        mesh.set_array(Z_new.ravel())
        title_text.set_text(f"{title} (t={current_t:.2f})")

        return mesh, title_text

    # 5. Create Animation
    anim = FuncAnimation(fig, update, frames=len(t_points), interval=50, blit=False)
    plt.close(fig)
    return anim


def animate_linear_comparison(t_points, sol_nl, sol_l, L=1.0, theta0_deg=None):
    """
    Side-by-side animation comparing the Non-linear (True) and Linear (Approx)
    physical motion.

    Args:
        t_points: 1D array of time values.
        sol_nl: 2D array [theta, p] from the non-linear physics.
        sol_l: 2D array [theta, p] from the linear physics.
        L: Length of the pendulum.
        theta0_deg: Optional initial angle in degrees for the title.
    """
    fig, (ax_true, ax_lin) = plt.subplots(1, 2, figsize=(12, 6))

    # 1. Setup Axes
    for ax, title in zip([ax_true, ax_lin], ["Non-linear (True)", "Linear (Approx)"]):
        ax.set_xlim(-1.2 * L, 1.2 * L)
        ax.set_ylim(-1.2 * L, 1.2 * L)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(title)

    # 2. Create Visual Objects
    (line_true,) = ax_true.plot(
        [], [], "o-", lw=2, color="k", markerfacecolor="royalblue", markersize=10
    )
    (line_lin,) = ax_lin.plot(
        [], [], "o-", lw=2, color="r", markerfacecolor="firebrick", markersize=10
    )
    time_text = ax_true.text(0.05, 0.9, "", transform=ax_true.transAxes)

    def update(i):
        # Convert angles to Cartesian for both solutions
        x_t, y_t = phys.get_coords(sol_nl[0, i], L)
        x_l, y_l = phys.get_coords(sol_l[0, i], L)

        line_true.set_data([0, x_t], [0, y_t])
        line_lin.set_data([0, x_l], [0, y_l])

        time_text.set_text(f"t = {t_points[i]:.1f}s")
        return line_true, line_lin, time_text

    # Handle the title with doubled braces for LaTeX safety
    title_str = "Linearization Accuracy Comparison"
    if theta0_deg is not None:
        title_str += rf" ($\theta_{{0}} = {theta0_deg}^\circ$)"

    plt.suptitle(title_str, fontsize=14)

    anim = FuncAnimation(fig, update, frames=len(t_points), interval=50, blit=True)
    plt.close(fig)  # Prevent static plot from displaying
    return anim
