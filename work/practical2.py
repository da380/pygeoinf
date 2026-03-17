"""
Practical 2, Part 1: The Double Pendulum & Chaos

In this first part, we explore the dynamics of the Double Pendulum.
Unlike the single pendulum, this system is chaotic: small changes in
initial conditions lead to exponentially diverging outcomes.

We will:
1. Define the physical parameters (Lengths, Masses).
2. Set an Initial Condition (State vector).
3. Integrate the equations of motion.
4. Animate the result to visualise the behavior.
"""

import numpy as np

# Import the core engine and the double pendulum module
try:
    from pygeoinf import data_assimilation as da
except ImportError:
    print("Error: pygeoinf not found. Please ensure it is installed.")

from pygeoinf.data_assimilation.pendulum import double

# =============================================================================
# 1. PHYSICAL SETUP
# =============================================================================
# We define a standard double pendulum with equal lengths and masses.
params = {"L1": 1.0, "L2": 1.0, "m1": 1.0, "m2": 1.0, "g": 1.0}

# Unpack for passing to the solver later
physics_args = (params["L1"], params["L2"], params["m1"], params["m2"], params["g"])


# =============================================================================
# 2. INITIAL CONDITIONS (STUDENT EXPERIMENT)
# =============================================================================
# The state vector is 4D: [theta1, theta2, p1, p2]
# theta: Angle from vertical (0 is down)
# p: Generalized momentum (related to angular velocity)

# --- EXPERIMENT 1: ORDER ---
# Try small angles (e.g., pi/10). The motion should look regular and repeating.
# theta1_init = np.pi / 10
# theta2_init = np.pi / 10

# --- EXPERIMENT 2: CHAOS ---
# Try large angles (e.g., pi/2 or horizontal). The motion becomes unpredictable.
# This is the "High Energy" regime.
theta1_init = np.pi / 2  # 90 degrees (Horizontal)
theta2_init = np.pi / 2  # 90 degrees (Horizontal)

# Initial momenta (usually start at rest)
p1_init = 0.0
p2_init = 0.0

# Create the state vector y0
y0 = np.array([theta1_init, theta2_init, p1_init, p2_init])

print(f"Initial State: {y0}")
print("Integration started...")


# =============================================================================
# 3. SOLVE DYNAMICS
# =============================================================================
# We integrate the system for 20 seconds.

t_start = 0.0
t_end = 20.0
dt = 0.05  # Time step for animation frames

t_eval = np.arange(t_start, t_end, dt)

# Solve using the core engine
# double.physics.eom is the Equation of Motion for the system
solution = da.solve_trajectory(
    eom_func=double.physics.eom,
    y0=y0,
    t_points=t_eval,
    args=physics_args,
    rtol=1e-10,  # High precision required for chaotic systems
    atol=1e-12,
)

print("Integration complete.")


# =============================================================================
# 4. VISUALISATION
# =============================================================================
# We use the built-in animator for the double pendulum.
# It includes a fading trail to help visualise the complexity of the path.

print("Generating Animation...")

anim = double.animate_pendulum(
    t_points=t_eval,
    solution=solution,
    L1=params["L1"],
    L2=params["L2"],
    trail_len=50,  # Length of the "tail" following the bob
)

# In a Jupyter Notebook, we would use:
# da.display_animation_html(anim)

# In a standard script, we use plt.show()
# plt.show()
