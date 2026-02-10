"""
assimilation.py

Single Pendulum specialization for Grid-based Data Assimilation.
Configures the generic core engine for the 2D pendulum state space.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .. import core
from . import physics as phys


def advect_pdf(
    pdf_func,
    t_final,
    eom_func=None,
    x_lim=(-np.pi, np.pi),
    y_lim=(-3, 3),
    res=100,
    **physics_params,
):
    """
    Advects a PDF forward in time for the Single Pendulum.

    Args:
        eom_func: Optional. If None, defaults to phys.eom (non-linear).
    """
    # Default to non-linear if no function is provided
    if eom_func is None:
        eom_func = phys.eom

    L = physics_params.get("L", 1.0)
    m = physics_params.get("m", 1.0)
    g = physics_params.get("g", 1.0)
    eom_args = (L, m, g)

    grid_limits = [x_lim, y_lim]

    # Call Generic Core Engine
    axes, Z_initial, Z_advected = core.advect_pdf_grid(
        eom_func=eom_func,
        pdf_func=pdf_func,
        t_final=t_final,
        grid_limits=grid_limits,
        resolution=res,
        eom_args=eom_args,
    )

    X, Y = np.meshgrid(*axes, indexing="ij")
    return X, Y, Z_initial, Z_advected


def generate_synthetic_data(
    t_obs,
    prior_mean,
    prior_cov,
    obs_noise_std,
    L=1.0,
    m=1.0,
    g=1.0,
    seed=None,
    dt_render=0.05,
):
    """
    Generates synthetic "Ground Truth" (smooth) and "Noisy Observations" (sparse).

    Args:
        dt_render (float): Time step for the high-res "Truth" curve (for plotting).
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Sample the "True" Initial State from the Prior
    y0_true = np.random.multivariate_normal(prior_mean, prior_cov)
    physics_args = (L, m, g)

    # --- Run A: Sparse Integration for Observations ---
    # We need exact states at t_obs
    t_eval_obs = np.sort(t_obs)

    # Ensure we define the span from 0 (or min time) to max time
    t_span_obs = np.concatenate(([0.0], t_eval_obs))

    # Solve
    sol_obs = core.solve_trajectory(phys.eom, y0_true, t_span_obs, args=physics_args)

    # Extract states corresponding to t_obs (skipping 0.0 unless it's in t_obs)
    # logic: searchsorted finds indices of t_obs in t_span_obs
    idx_obs = np.searchsorted(t_span_obs, t_eval_obs)
    true_state_at_obs = sol_obs[:, idx_obs]

    # Add Noise
    true_theta_obs = true_state_at_obs[0, :]
    noise = np.random.normal(0, obs_noise_std, size=len(true_theta_obs))
    observations = true_theta_obs + noise

    # --- Run B: Dense Integration for Plotting ---
    t_max = np.max(t_obs)
    t_render = np.arange(0, t_max + dt_render, dt_render)

    # Ensure t_max is included for a complete plot
    if t_render[-1] < t_max:
        t_render = np.append(t_render, t_max)

    state_render = core.solve_trajectory(phys.eom, y0_true, t_render, args=physics_args)

    return {
        # Data for the Filter
        "t_obs": t_obs,
        "observations": observations,
        # Data for Visualization (Ground Truth)
        "t_ground_truth": t_render,
        "state_ground_truth": state_render,
        "initial_truth": y0_true,
    }


def assimilate(t_obs, observations, obs_std, prior_func, grid_params, physics_params):
    """
    Single Pendulum Assimilation Wrapper.
    """
    # 1. Setup Grid & Axes
    res = grid_params.get("res", 100)
    x_vals = np.linspace(*grid_params["x_lim"], res)
    y_vals = np.linspace(*grid_params["y_lim"], res)
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")
    axes_1d = [x_vals, y_vals]

    # 2. Define Specific Callbacks
    def forecast_callback(pdf_func, dt):
        """Wraps advect_pdf to return only the final Z grid."""
        # index 3 is Z_final
        return advect_pdf(
            pdf_func,
            t_final=dt,
            res=res,
            x_lim=grid_params["x_lim"],
            y_lim=grid_params["y_lim"],
            **physics_params,
        )[3]

    def analysis_callback(Z_forecast, obs_val, obs_std):
        """Computes Likelihood (circular) and Updates."""
        diff = core.wrap_angle(X - obs_val)
        likelihood = core.gaussian_likelihood(diff, 0.0, obs_std)
        return core.bayesian_update(Z_forecast, likelihood, axes_1d)

    def grid_to_func_callback(Z_grid):
        """Wraps the core interpolator."""
        return core.get_pdf_from_grid(axes_1d, Z_grid)

    # 3. Run Core Cycle

    # Note: assimilate_cycle in the previous step expected a function for the forecast
    results = core.assimilate_cycle(
        t_obs=t_obs,
        observations=observations,
        obs_std=obs_std,
        prior_func=prior_func,
        forecast_func=forecast_callback,
        analysis_func=analysis_callback,
        grid_to_func_wrapper=grid_to_func_callback,
    )

    return results


def reanalyse_trajectory(
    final_posterior_grid, t_final, grid_params, physics_params, dt_render=0.05
):
    """
    Performs Reanalysis (Smoothing).

    1. Pulls the Final Posterior back to t=0 to get P(x_0 | y_{1:T}).
    2. Computes the mean of this smoothed initial distribution.
    3. Generates the optimal trajectory from that mean.

    Args:
        final_posterior_grid: The grid Z at t_final.
        t_final: The time at which this posterior exists.
        grid_params: Res, x_lim, y_lim.
        physics_params: L, m, g.

    Returns:
        t_smooth: Time array.
        state_smooth: Reanalyzed state trajectory.
        smoothed_initial_mean: The optimized x0.
    """
    # 1. Setup Grid at t=0
    res = grid_params.get("res", 100)
    x_vals = np.linspace(*grid_params["x_lim"], res)
    y_vals = np.linspace(*grid_params["y_lim"], res)
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")

    # Pre-calculate flat grid
    grid_flat = np.stack([X.ravel(), Y.ravel()])
    y0_vectorized = grid_flat.reshape(-1)

    # 2. Physics Helper (Same as core)
    L, m, g = physics_params["L"], physics_params["m"], physics_params["g"]

    def vectorized_eom(t, y_flat):
        y_reshaped = y_flat.reshape(2, -1)
        dydt = phys.eom(t, y_reshaped, L=L, m=m, g=g)
        return np.concatenate(dydt).reshape(-1)

    # 3. Advection: "Pull Back"
    # To find density at x_0, we integrate x_0 FORWARD to x_T and sample the final posterior.
    # This is the Liouville method for "What initial condition lands here?"

    # Solve 0 -> T
    sol = core.solve_trajectory(vectorized_eom, y0_vectorized, [0, t_final])

    # Get endpoints at T
    final_states_flat = sol[:, -1]
    final_states = final_states_flat.reshape(2, res, res)

    # Interpolate the Final Posterior Grid at these endpoints
    # We need to construct the interpolator on the Final Grid
    interp = RegularGridInterpolator(
        (x_vals, y_vals), final_posterior_grid, bounds_error=False, fill_value=0.0
    )

    # Prepare query points (N, 2)
    pts = np.stack([final_states[0].ravel(), final_states[1].ravel()], axis=1)

    # Sample to get Smoothed Initial Grid
    Z_smoothed_initial = interp(pts).reshape(res, res)

    # 4. Compute Mean of Smoothed Initial Grid
    norm = np.sum(Z_smoothed_initial)
    if norm > 1e-15:
        mx = np.sum(X * Z_smoothed_initial) / norm
        my = np.sum(Y * Z_smoothed_initial) / norm
        smoothed_mean = np.array([mx, my])
    else:
        smoothed_mean = np.array([0.0, 0.0])  # Fallback

    # 5. Generate Optimal Trajectory
    # Integrate from 0 to t_final using this optimized mean
    t_smooth = np.arange(0, t_final + dt_render, dt_render)
    state_smooth = core.solve_trajectory(
        phys.eom, smoothed_mean, t_smooth, args=(L, m, g)
    )

    return t_smooth, state_smooth, smoothed_mean


def get_smooth_filter_trajectory(results, physics_params, dt_render=0.05):
    """
    Reconstructs the continuous 'Sawtooth' trajectory of the filter.

    It calls the standard integrator between t_i and t_{i+1},
    restarting the physics engine at every new filter estimate.
    """
    t_chunks = []
    state_chunks = []

    # Physics args for the solver
    args = (physics_params["L"], physics_params["m"], physics_params["g"])

    # Loop over every interval between observations
    for i in range(len(results) - 1):
        # 1. Get the Filter's Estimate at t_i (The "Jump")
        t_start = results[i]["time"]
        y_start = results[i]["mean"]  # Restart physics here

        t_end = results[i + 1]["time"]

        # 2. Generate time points for this segment
        # Ensure we don't go past t_end
        t_eval = np.arange(t_start, t_end, dt_render)

        # 3. Call the USUAL INTEGRATOR for just this segment
        sol = core.solve_trajectory(physics.eom, y_start, t_eval, args=args)

        t_chunks.append(t_eval)
        state_chunks.append(sol)

    # Concatenate all the short physics runs into one long array
    if t_chunks:
        t_smooth = np.concatenate(t_chunks)
        state_smooth = np.concatenate(state_chunks, axis=1)
    else:
        # Handle edge case of single point
        t_smooth = np.array([results[0]["time"]])
        state_smooth = np.array([results[0]["mean"]]).T

    return t_smooth, state_smooth
