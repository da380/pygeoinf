"""
physics.py

Hamiltonian dynamics, coordinate transformations, and linearized models
specifically for the Single Pendulum (2D state space).

Includes:
1. Full Non-Linear Hamiltonian Dynamics (eom)
2. Tangent Linear Model for EKF (eom_tangent_linear)
3. Static Linearized Model about Equilibrium (eom_linear)
"""

import numpy as np

# --- 1. Hamiltonian Dynamics (Non-linear) ---


def eom(t, y, L=1.0, m=1.0, g=1.0):
    """
    Hamilton's Equations of Motion for a single pendulum.

    State vector y = [theta, p_theta]

    Args:
        t: Time (unused in autonomous system).
        y: State vector [angle, momentum].
        L, m, g: Physical parameters.

    Returns:
        dydt: [d(theta)/dt, d(p_theta)/dt]
    """
    theta, p_theta = y

    # d(theta)/dt = dH/dp = p / (m L^2)
    d_theta = p_theta / (m * L**2)

    # d(p)/dt = -dH/dtheta = -m g L sin(theta)
    d_p_theta = -m * g * L * np.sin(theta)

    return [d_theta, d_p_theta]


def get_coords(theta, L=1.0):
    """
    Converts angular state to Cartesian coordinates for visualization.
    """
    # Standard pendulum convention: 0 is down.
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    return x, y


# --- 2. Tangent Linear Model (Dynamic Linearization) ---


def get_jacobian(theta, L=1.0, m=1.0, g=1.0):
    """
    Computes the Jacobian matrix J(theta) of the system at a specific state.

    System f(y):
      f1 = p / (m L^2)
      f2 = -m g L sin(theta)

    Jacobian J = df/dy:
      [[ df1/dth, df1/dp ],
       [ df2/dth, df2/dp ]]

    Returns:
       2x2 Matrix J
    """
    # df1/dth = 0,  df1/dp = 1/(m L^2)
    a = 1.0 / (m * L**2)

    # df2/dth = -m g L cos(theta), df2/dp = 0
    b = -m * g * L * np.cos(theta)

    return np.array([[0, a], [b, 0]])


def eom_tangent_linear(t, state_aug, L=1.0, m=1.0, g=1.0):
    """
    Coupled ODE for the reference trajectory AND the tangent linear perturbation.
    Crucial for Extended Kalman Filters (EKF).

    State vector (4D): [theta, p, delta_theta, delta_p]

    1. Evolves [theta, p] using full non-linear physics.
    2. Evolves [delta_theta, delta_p] using the Jacobian at the CURRENT [theta, p].
    """
    # 1. Unpack Reference (Non-linear) and Perturbation (Linear)
    ref = state_aug[0:2]  # [theta, p]
    pert = state_aug[2:4]  # [delta_theta, delta_p]

    # 2. Compute Non-Linear Evolution for Reference (d_ref/dt)
    d_ref = eom(t, ref, L, m, g)

    # 3. Compute Linear Evolution for Perturbation (d_pert/dt = J * pert)
    J = get_jacobian(ref[0], L, m, g)
    d_pert = J @ pert

    # 4. Concatenate results
    return np.concatenate([d_ref, d_pert])


# --- 3. Static Linear Models (Equilibrium Approximation) ---


def eom_linear(t, y, L=1.0, m=1.0, g=1.0):
    """
    Linearized EOM around the stable equilibrium (theta=0, p=0).
    Here the Jacobian is constant because cos(0) = 1.
    """
    a = 1.0 / (m * L**2)
    b = -m * g * L

    A = np.array([[0, a], [b, 0]])
    return A @ y


def get_linear_propagator(t, L=1.0, m=1.0, g=1.0):
    """
    Returns the analytical propagator matrix P(t) for the linear system.
    y(t) = P(t) @ y(0)

    Useful for validating numerical solvers or standard Kalman Filters.
    """
    omega = np.sqrt(g / L)
    cos_t = np.cos(omega * t)
    sin_t = np.sin(omega * t)

    # Derived from solving the linear harmonic oscillator matrix exponential
    m01 = sin_t / (m * L**2 * omega)
    m10 = -m * g * L * sin_t / omega

    return np.array([[cos_t, m01], [m10, cos_t]])


# Alias for explicit clarity if requested by init files
eom_linear_static = eom_linear
