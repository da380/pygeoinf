"""
physics.py

Hamiltonian dynamics, coordinate transformations, and linearised models
specifically for the Double Pendulum (4D state space).

State vector: y = [theta1, theta2, p1, p2]
Convention: 0 is vertically DOWN. Rotation is Counter-Clockwise.
"""

from typing import List, Tuple, Union, Optional

import numpy as np
from scipy.linalg import expm


# --- 1. Hamiltonian Dynamics (Non-linear) ---


def eom(
    t: float,
    y: np.ndarray,
    L1: float = 1.0,
    L2: float = 1.0,
    m1: float = 1.0,
    m2: float = 1.0,
    g: float = 1.0,
) -> List[float]:
    """
    Hamilton's Equations of Motion for the Double Pendulum.
    Compatible with core.solve_trajectory.
    """
    th1, th2, p1, p2 = y

    # Intermediate terms
    d_th = th1 - th2
    c, s = np.cos(d_th), np.sin(d_th)
    den = m1 + m2 * s**2

    # 1. Angular Velocities (dtheta/dt)
    dth1 = (L2 * p1 - L1 * p2 * c) / (L1**2 * L2 * den)
    dth2 = (L1 * (m1 + m2) * p2 - L2 * m2 * p1 * c) / (L1 * L2**2 * m2 * den)

    # 2. Forces (dp/dt)
    term1 = m2 * L1 * L2 * dth1 * dth2 * s
    term2 = (m1 + m2) * g * L1 * np.sin(th1)
    term3 = m2 * g * L2 * np.sin(th2)

    dp1 = -term1 - term2
    dp2 = term1 - term3

    return [dth1, dth2, dp1, dp2]


def get_coords(
    th1: Union[float, np.ndarray],
    th2: Union[float, np.ndarray],
    L1: float = 1.0,
    L2: float = 1.0,
) -> Tuple[Union[float, np.ndarray], ...]:
    """
    Converts angles to Cartesian coordinates for both bobs.
    Convention: (0,0) is the pivot, +y is Up, +x is Right.
    """
    x1 = L1 * np.sin(th1)
    y1 = -L1 * np.cos(th1)
    x2 = x1 + L2 * np.sin(th2)
    y2 = y1 - L2 * np.cos(th2)
    return x1, y1, x2, y2


# --- 2. Tangent Linear Model (Analytic Jacobian) ---


def get_jacobian(
    y: np.ndarray,
    L1: float = 1.0,
    L2: float = 1.0,
    m1: float = 1.0,
    m2: float = 1.0,
    g: float = 1.0,
) -> np.ndarray:
    """
    Computes the Exact Analytic Jacobian Matrix J(y) = df/dy (4x4).
    Crucial for Extended Kalman Filters (EKF).
    """
    th1, th2, p1, p2 = y

    # --- Shared Terms ---
    delta = th1 - th2
    c, s = np.cos(delta), np.sin(delta)
    den = m1 + m2 * s**2
    den2 = den * den

    # d(den)/dth1 = m2 * sin(2*delta)
    d_den_dth = m2 * np.sin(2 * delta)

    # Constants
    K1 = 1.0 / (L1**2 * L2)
    K2 = 1.0 / (L1 * L2**2 * m2)

    # --- Rows 0 & 1: d(dth)/dy ---
    # Num1 for dth1
    Num1 = L2 * p1 - L1 * p2 * c
    dth1_val = Num1 * K1 / den

    dNum1_dth1 = L1 * p2 * s
    dNum1_dth2 = -L1 * p2 * s
    dNum1_dp1 = L2
    dNum1_dp2 = -L1 * c

    # Jacobian Elements Row 0
    J00 = K1 * (dNum1_dth1 * den - Num1 * d_den_dth) / den2
    J01 = K1 * (dNum1_dth2 * den - Num1 * (-d_den_dth)) / den2
    J02 = K1 * (dNum1_dp1 * den) / den2
    J03 = K1 * (dNum1_dp2 * den) / den2

    # Num2 for dth2
    Num2 = L1 * (m1 + m2) * p2 - L2 * m2 * p1 * c
    dth2_val = Num2 * K2 / den

    dNum2_dth1 = L2 * m2 * p1 * s
    dNum2_dth2 = -L2 * m2 * p1 * s
    dNum2_dp1 = -L2 * m2 * c
    dNum2_dp2 = L1 * (m1 + m2)

    # Jacobian Elements Row 1
    J10 = K2 * (dNum2_dth1 * den - Num2 * d_den_dth) / den2
    J11 = K2 * (dNum2_dth2 * den - Num2 * (-d_den_dth)) / den2
    J12 = K2 * (dNum2_dp1 * den) / den2
    J13 = K2 * (dNum2_dp2 * den) / den2

    # --- Rows 2 & 3: d(dp)/dy ---
    C_force = m2 * L1 * L2
    G1 = (m1 + m2) * g * L1
    G2 = m2 * g * L2

    # Helper for product rule on term1 = C * dth1 * dth2 * s
    J = np.zeros((4, 4))
    J[0, :] = [J00, J01, J02, J03]
    J[1, :] = [J10, J11, J12, J13]

    def d_term1(idx, ds_dq):
        # d(uvw) = u'vw + uv'w + uvw'
        return C_force * (
            J[0, idx] * dth2_val * s
            + dth1_val * J[1, idx] * s
            + dth1_val * dth2_val * ds_dq
        )

    d_term1_dth1 = d_term1(0, c)
    d_term1_dth2 = d_term1(1, -c)
    d_term1_dp1 = d_term1(2, 0)
    d_term1_dp2 = d_term1(3, 0)

    # Row 2 (dp1)
    J[2, 0] = -d_term1_dth1 - G1 * np.cos(th1)
    J[2, 1] = -d_term1_dth2
    J[2, 2] = -d_term1_dp1
    J[2, 3] = -d_term1_dp2

    # Row 3 (dp2)
    J[3, 0] = d_term1_dth1
    J[3, 1] = d_term1_dth2 - G2 * np.cos(th2)
    J[3, 2] = d_term1_dp1
    J[3, 3] = d_term1_dp2

    return J


def eom_tangent_linear(
    t: float,
    state_aug: np.ndarray,
    L1: float = 1.0,
    L2: float = 1.0,
    m1: float = 1.0,
    m2: float = 1.0,
    g: float = 1.0,
) -> np.ndarray:
    """
    Coupled ODE for Reference (Non-linear) + Perturbation (Linear).
    State vector (8D): [y (4D), delta_y (4D)]
    """
    ref = state_aug[0:4]
    pert = state_aug[4:8]

    # 1. Non-linear Reference Evolution
    d_ref = eom(t, ref, L1, L2, m1, m2, g)

    # 2. Linear Perturbation Evolution
    J = get_jacobian(ref, L1, L2, m1, m2, g)
    d_pert = J @ pert

    return np.concatenate([d_ref, d_pert])


# --- 3. Static Linear Models (Equilibrium) ---


def get_linear_matrix(
    L1: float = 1.0, L2: float = 1.0, m1: float = 1.0, m2: float = 1.0, g: float = 1.0
) -> np.ndarray:
    """
    Returns the linearised system matrix A for the equilibrium state (0,0,0,0).
    d/dt(y) = A * y
    """
    # Mass Matrix M at equilibrium
    M = np.array([[(m1 + m2) * L1**2, m2 * L1 * L2], [m2 * L1 * L2, m2 * L2**2]])
    M_inv = np.linalg.inv(M)

    # Stiffness Matrix K
    K = np.array([[(m1 + m2) * g * L1, 0], [0, m2 * g * L2]])

    # Construct 4x4 A Matrix
    A = np.zeros((4, 4))
    A[0:2, 2:4] = M_inv
    A[2:4, 0:2] = -K

    return A


def get_linear_propagator(
    dt: float,
    L1: float = 1.0,
    L2: float = 1.0,
    m1: float = 1.0,
    m2: float = 1.0,
    g: float = 1.0,
) -> np.ndarray:
    """
    Returns the discrete-time propagator P = exp(A * dt).
    """
    A = get_static_matrix(L1, L2, m1, m2, g)
    return expm(A * dt)


def eom_linear(t: float, y: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Linearised EOM given pre-computed A matrix."""
    return A @ y
