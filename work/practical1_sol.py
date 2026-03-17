import numpy as np
import matplotlib.pyplot as plt

# We import the core engine and the specific physics for the pendulum
from pygeoinf import data_assimilation as da
from pygeoinf.data_assimilation.pendulum import single

# =============================================================================
# PART 1: SETUP & SYNTHETIC REALITY
# =============================================================================
# In this section, we set up the "Truth" - the actual physics we want to track.
# We will generate a true trajectory and some noisy observations of it.

# 1. Physics: A 1-meter pendulum in standard gravity
L = 1.0  # Length [m]
g = 9.81  # Gravity [m/s^2]
params = (L, g)

# 2. Initial Condition
# We start at 45 degrees (pi/4) with 0 velocity.
true_y0 = np.array([np.pi / 4, 0.0])

# 3. Create the Assimilation Problem Manager
# We use this class to manage the "Truth" and generate data.
problem = da.BayesianAssimilationProblem(eom_func=single.physics.eom, eom_args=params)

# 4. Define Observations
# We will observe the ANGLE (theta) every 0.5 seconds.
# We do NOT observe the velocity.
t_obs_points = [0.5, 1.0, 1.5, 2.0, 2.5]
obs_error_sigma = 0.15  # 0.15 radians uncertainty (approx 8.5 degrees)

# Covariance matrix R (1x1 for a single scalar observation)
R = np.array([[obs_error_sigma**2]])
# Observation Operator H (1x2 matrix: [1, 0] selects theta from [theta, p])
H = np.array([[1.0, 0.0]])

# Register these observations with the problem
print("Generating synthetic data...")
for t in t_obs_points:
    problem.add_observation(
        time=t, covariance=R, operator=H, value=None  # Placeholder, will be filled next
    )

# 5. Generate the "Truth" and the Noisy Data
truth_data = problem.generate_synthetic_data(
    true_initial_condition=true_y0,
    dt_render=0.01,
    seed=42,  # Fixed seed so everyone gets the same noise
)

# Visualise the Problem
t_true = truth_data["t_ground_truth"]
y_true = truth_data["state_ground_truth"]
obs_list = problem.observations  # List of (time, model) tuples

plt.figure(figsize=(10, 5))
plt.plot(t_true, y_true[0], "k-", label="True Angle")
# Extract observation values for plotting
t_vals = [t for t, _ in obs_list]
y_vals = [m.y_obs[0] for _, m in obs_list]
plt.errorbar(t_vals, y_vals, yerr=obs_error_sigma, fmt="rx", label="Observations")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.title("The Problem: Tracking a Pendulum with Noisy Data")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# =============================================================================
# PART 2: MANUAL ASSIMILATION LOOP (Student Task)
# =============================================================================
# Now it is your turn. We will track this pendulum using a Probability Grid.
# You need to implement the loop that alternates between:
#   1. FORECAST: Advecting the grid forward in time (physics).
#   2. ANALYSIS: Updating the grid with the observation (bayes).

# --- Step A: Define the Grid ---
# We need to discretise the phase space (Angle vs Velocity).
# Bounds: Angle [-pi, pi], Velocity [-5, 5]
bounds = [(-np.pi, np.pi), (-5, 5)]
resolution = [100, 100]  # 100x100 points

# --- Step B: Create the Prior (Initial Belief) ---
# We have a rough guess of where it started.
# Guess: 60 degrees (pi/3) instead of 45. We are wrong!
# Uncertainty: Large (0.5 rad)
prior_mean = [np.pi / 3, 0.0]
prior_cov = np.diag([0.5**2, 0.5**2])

# Create the PDF function and the Grid object
prior_pdf = da.get_gaussian_pdf(prior_mean, prior_cov)
current_grid = da.ProbabilityGrid.from_bounds(bounds, resolution, prior_pdf)

# --- Step C: The Loop (TODO) ---

# We will store our results here
manual_history = []
t_current = 0.0

print("Starting manual assimilation...")

# Iterate through the observations available in 'problem.observations'
for t_obs, likelihood_model in problem.observations:

    # 1. Calculate time step to the next observation
    dt = t_obs - t_current

    # ============================================================
    # TODO: YOUR CODE HERE
    # ============================================================

    # Task 1: FORECAST
    # Use current_grid.push_forward(...) to evolve physics by dt.
    # Store the result in a variable called 'forecast_grid'.
    # Hint: You need problem.eom_func and problem.eom_args

    forecast_grid = current_grid.push_forward(
        problem.eom_func, dt, problem.eom_args
    )  # <--- SOLUTION

    # Task 2: ANALYSIS
    # Use likelihood_model.evaluate(...) to get the likelihood grid.
    # Use forecast_grid.bayes_update(...) to combine them.
    # Store result in 'analysis_grid' and 'evidence'.

    lik_grid = likelihood_model.evaluate(forecast_grid)  # <--- SOLUTION
    analysis_grid, evidence = forecast_grid.bayes_update(lik_grid)  # <--- SOLUTION

    # ============================================================
    # END YOUR CODE
    # ============================================================

    # Store result
    manual_history.append(
        {"time": t_obs, "forecast": forecast_grid, "analysis": analysis_grid}
    )

    # Prepare for next step
    current_grid = analysis_grid
    t_current = t_obs
    print(f"Assimilated observation at t={t_obs:.2f}")


# =============================================================================
# PART 3: VISUALISATION
# =============================================================================
# Let's see how well you did. We will plot the "marginal" distribution
# of the angle (integrating out the velocity) over time.

plt.figure(figsize=(10, 6))

# Plot Truth
plt.plot(t_true, y_true[0], "k-", lw=2, label="Truth")
plt.plot(t_vals, y_vals, "rx", label="Observations")

# Plot Your Estimates
est_times = [step["time"] for step in manual_history]
est_means = []
est_stds = []

for step in manual_history:
    # Marginalise to getting 1D PDF for Theta (dimension 0)
    # The grid object has a .marginalise() method
    marg = step["analysis"].marginalise(keep_indices=(0,))

    # Calculate mean and std from the grid axes and values
    # (Simple approximation for visualisation)
    m = marg.mean[0]  # ProbabilityGrid.mean returns a vector
    est_means.append(m)

plt.plot(est_times, est_means, "b-o", label="Grid Estimate (Posterior Mean)")
plt.title("Tracking Performance")
plt.legend()
plt.grid(True)
plt.show()


# =============================================================================
# PART 4: AUTOMATED RUN & REANALYSIS
# =============================================================================
# Now that you understand the loop, you can use the automated .run() method.
# We will use this to perform REANALYSIS (Smoothing).
#
# Task: Can we use the final state at t=2.5 to figure out the
# TRUE initial condition at t=0?

print("\nRunning automated reanalysis...")

# 1. Get the final posterior from your manual history
final_grid = manual_history[-1]["analysis"]
final_time = manual_history[-1]["time"]

# 2. Push it BACKWARDS in time to t=0
# TODO: Use push_forward with a NEGATIVE time step (-final_time)
# to create 'smoothed_initial_grid'.

smoothed_initial_grid = final_grid.push_forward(  # <--- SOLUTION
    problem.eom_func,  # <--- SOLUTION
    -final_time,  # <--- SOLUTION
    problem.eom_args,  # <--- SOLUTION
)  # <--- SOLUTION

# 3. Visualise the Improvement
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot A: The Wrong Prior we started with
da.plot_grid_marginal(
    da.ProbabilityGrid.from_bounds(bounds, resolution, prior_pdf),
    dims=(0, 1),
    ax=axes[0],
    title="Initial Guess (Prior)",
)
axes[0].plot(true_y0[0], true_y0[1], "rx", markersize=10, label="True Start")

# Plot B: The Final Posterior (after seeing all data)
da.plot_grid_marginal(
    final_grid, dims=(0, 1), ax=axes[1], title=f"Final Posterior (t={final_time})"
)

# Plot C: The Smoothed Initial Condition (Reanalysis)
da.plot_grid_marginal(
    smoothed_initial_grid, dims=(0, 1), ax=axes[2], title="Reanalysis (Smoothed t=0)"
)
axes[2].plot(true_y0[0], true_y0[1], "rx", markersize=10, label="True Start")
axes[2].legend()

plt.tight_layout()
plt.show()

print("Discussion: How does the Reanalysis (Plot C) compare to the Prior (Plot A)?")
print("Notice how the red cross (Truth) is now inside the high probability region!")
