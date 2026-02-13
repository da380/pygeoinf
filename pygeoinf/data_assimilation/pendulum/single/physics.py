"""
physics.py

Hamiltonian dynamics, coordinate transformations, and linearised models
specifically for the Single Pendulum (2D state space).
"""

from typing import Tuple, List, Union

import numpy as np


# --- 1. Hamiltonian Dynamics (Non-linear) ---


def eom(
    t: float, y: np.ndarray, L: float = 1.0, m: float = 1.0, g: float = 1.0
) -> List[float]:
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


def get_coords(
    theta: Union[float, np.ndarray], L: float = 1.0
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Converts angular state to Cartesian coordinates for visualisation.
    Standard pendulum convention: 0 is down.
    """
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    return x, y


# --- 2. Tangent Linear Model (Dynamic Linearisation) ---


def get_jacobian(
    theta: float, L: float = 1.0, m: float = 1.0, g: float = 1.0
) -> np.ndarray:
    """
    Computes the Jacobian matrix J(theta) of the system at a specific state.

    Returns:
       2x2 Matrix J
    """
    # df1/dth = 0,  df1/dp = 1/(m L^2)
    a = 1.0 / (m * L**2)

    # df2/dth = -m g L cos(theta), df2/dp = 0
    b = -m * g * L * np.cos(theta)

    return np.array([[0, a], [b, 0]])


def eom_tangent_linear(
    t: float, state_aug: np.ndarray, L: float = 1.0, m: float = 1.0, g: float = 1.0
) -> np.ndarray:
    """
    Coupled ODE for the reference trajectory AND the tangent linear perturbation.
    Crucial for Extended Kalman Filters (EKF).

    State vector (4D): [theta, p, delta_theta, delta_p]
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


def eom_linear(
    t: float, y: np.ndarray, L: float = 1.0, m: float = 1.0, g: float = 1.0
) -> np.ndarray:
    """
    Linearised EOM around the stable equilibrium (theta=0, p=0).
    """
    a = 1.0 / (m * L**2)
    b = -m * g * L

    A = np.array([[0, a], [b, 0]])
    return A @ y


def get_linear_propagator(
    t: float, L: float = 1.0, m: float = 1.0, g: float = 1.0
) -> np.ndarray:
    """
    Returns the analytical propagator matrix P(t) for the linear system.
    y(t) = P(t) @ y(0)
    """
    omega = np.sqrt(g / L)
    cos_t = np.cos(omega * t)
    sin_t = np.sin(omega * t)

    # Derived from solving the linear harmonic oscillator matrix exponential
    m01 = sin_t / (m * L**2 * omega)
    m10 = -m * g * L * sin_t / omega

    return np.array([[cos_t, m01], [m10, cos_t]])
