"""
Practical 1 SOLUTION: The Single Pendulum & Data Assimilation

Objective:
1.  Simulate the 'True' dynamics of a single pendulum.
2.  Manually implement the "Forecast-Analysis" Bayesian cycle.
3.  Perform a "Reanalysis" to reconstruct the initial state.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import core engine and physics
from pygeoinf import data_assimilation as da
from pygeoinf.data_assimilation.pendulum import single

# =============================================================================
# PART 1: SETUP & SYNTHETIC REALITY
# =============================================================================

# 1. Physics
L = 1.0  # Length [m]
g = 9.81  # Gravity [m/s^2]
params = (L, g)

# 2. Initial Condition (Truth)
# Start at 45 degrees (pi/4) with 0 velocity.
true_y0 = np.array([np.pi / 4, 0.0])

# 3. Create the Problem Manager
problem = da.BayesianAssimilationProblem(eom_func=single.physics.eom, eom_args=params)

# 4. Define Observations
# Observe Angle (theta) every 0.5 seconds
t_obs_points = [0.5, 1.0, 1.5, 2.0, 2.5]
obs_error_sigma = 0.15  # 0.15 radians uncertainty

R = np.array([[obs_error_sigma**2]])  # Covariance
H = np.array([[1.0, 0.0]])  # Observation Operator

# Register observations
print("Generating synthetic data...")
for t in t_obs_points:
    problem.add_observation(time=t, covariance=R, operator=H, value=None)

# 5. Generate Truth and Data
truth_data = problem.generate_synthetic_data(
    true_initial_condition=true_y0, dt_render=0.01, seed=42
)

# Visualise the Problem
t_true = truth_data["t_ground_truth"]
y_true = truth_data["state_ground_truth"]
obs_list = problem.observations

plt.figure(figsize=(10, 5))
plt.plot(t_true, y_true[0], "k-", label="True Angle")
t_vals = [t for t, _ in obs_list]
y_vals = [m.y_obs[0] for _, m in obs_list]
plt.errorbar(t_vals, y_vals, yerr=obs_error_sigma, fmt="rx", label="Observations")
plt.title("The Problem: Tracking a Pendulum with Noisy Data")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# =============================================================================
# PART 2: MANUAL ASSIMILATION LOOP (SOLUTION)
# =============================================================================

# --- Step A: Define Grid ---
bounds = [(-np.pi, np.pi), (-5, 5)]
resolution = [100, 100]

# --- Step B: Create Prior ---
# Wrong guess: 60 degrees (pi/3) instead of 45
prior_mean = [np.pi / 3, 0.0]
prior_cov = np.diag([0.5**2, 0.5**2])

prior_pdf = da.get_gaussian_pdf(prior_mean, prior_cov)
current_grid = da.ProbabilityGrid.from_bounds(bounds, resolution, prior_pdf)

# --- Step C: The Loop ---
manual_history = []
t_current = 0.0

print("Starting manual assimilation...")

for t_obs, likelihood_model in problem.observations:

    dt = t_obs - t_current

    # --- SOLUTION START ---

    # 1. FORECAST (Physics Step)
    # Advect the probability grid forward to t_obs
    if dt > 0:
        forecast_grid = current_grid.push_forward(
            problem.eom_func, dt, problem.eom_args
        )
    else:
        forecast_grid = current_grid

    # 2. ANALYSIS (Bayesian Step)
    # Evaluate Likelihood P(y|x) on the grid
    lik_grid = likelihood_model.evaluate(forecast_grid)

    # Compute Posterior P(x|y) = Prior * Likelihood / Evidence
    analysis_grid, evidence = forecast_grid.bayes_update(lik_grid)

    # --- SOLUTION END ---

    manual_history.append(
        {
            "time": t_obs,
            "forecast": forecast_grid,
            "analysis": analysis_grid,
            "evidence": evidence,
        }
    )

    current_grid = analysis_grid
    t_current = t_obs
    print(f"Assimilated observation at t={t_obs:.2f}")


# =============================================================================
# PART 3: VISUALISATION
# =============================================================================

plt.figure(figsize=(10, 6))

# Plot Truth & Obs
plt.plot(t_true, y_true[0], "k-", lw=2, label="Truth")
plt.plot(t_vals, y_vals, "rx", label="Observations")

# Plot Estimates
est_times = [step["time"] for step in manual_history]
est_means = []

for step in manual_history:
    # Marginalise to 1D PDF for Theta
    marg = step["analysis"].marginalise(keep_indices=(0,))
    # Calculate mean of that 1D grid
    m = marg.mean[0]
    est_means.append(m)

plt.plot(est_times, est_means, "b-o", label="Grid Estimate")
plt.title("Tracking Performance")
plt.legend()
plt.grid(True)
plt.show()


# =============================================================================
# PART 4: AUTOMATED REANALYSIS (SOLUTION)
# =============================================================================
print("\nRunning reanalysis...")

# 1. Get the final posterior
final_grid = manual_history[-1]["analysis"]
final_time = manual_history[-1]["time"]

# 2. Push it BACKWARDS to t=0
# --- SOLUTION START ---
smoothed_initial_grid = final_grid.push_forward(
    problem.eom_func,  # Same physics
    -final_time,  # Negative time step!
    problem.eom_args,
)
# --- SOLUTION END ---

# 3. Visualise Improvement
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# A: Prior
da.plot_grid_marginal(
    da.ProbabilityGrid.from_bounds(bounds, resolution, prior_pdf),
    dims=(0, 1),
    ax=axes[0],
    title="Initial Guess (Prior)",
)
axes[0].plot(
    true_y0[0], true_y0[1], "rx", markersize=10, markeredgewidth=2, label="True Start"
)

# B: Final Posterior
da.plot_grid_marginal(
    final_grid, dims=(0, 1), ax=axes[1], title=f"Final Posterior (t={final_time})"
)

# C: Reanalysis
da.plot_grid_marginal(
    smoothed_initial_grid, dims=(0, 1), ax=axes[2], title="Reanalysis (Smoothed t=0)"
)
axes[2].plot(
    true_y0[0], true_y0[1], "rx", markersize=10, markeredgewidth=2, label="True Start"
)
axes[2].legend()

plt.tight_layout()
plt.show()

print("Reanalysis complete. The smoothed estimate should now align with the red cross.")
