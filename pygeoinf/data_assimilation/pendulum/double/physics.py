"""
physics.py

Hamiltonian dynamics, coordinate transformations, and linearized models
specifically for the Double Pendulum (4D state space).

--- Coordinate Convention ---
* Angles (theta) are measured in radians.
* theta = 0 corresponds to the rod hanging vertically DOWN.
* Rotation is COUNTER-CLOCKWISE positive.
* Both angles (theta1, theta2) are ABSOLUTE with respect to the vertical axis.
  (theta2 is NOT relative to theta1).

State vector: y = [theta1, theta2, p1, p2]
"""

import numpy as np
from scipy.linalg import expm

# --- 1. Hamiltonian Dynamics (Non-linear) ---


def eom(t, y, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
    """
    Hamilton's Equations of Motion for the Double Pendulum.
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


def get_coords(th1, th2, L1=1.0, L2=1.0):
    """
    Converts angles to Cartesian coordinates for both bobs.
    Convention: (0,0) is the pivot, +y is Up, +x is Right.
    """
    x1 = L1 * np.sin(th1)
    y1 = -L1 * np.cos(th1)
    x2 = x1 + L2 * np.sin(th2)
    y2 = y1 - L2 * np.cos(th2)
    return x1, y1, x2, y2


# --- 2. Tangent Linear Model (Explicit Analytic Jacobian) ---


def get_jacobian(y, L1, L2, m1, m2, g):
    """
    Computes the Exact Analytic Jacobian Matrix J(y) = df/dy.
    Calculated via explicit chain rule application (no sympy, no finite diff).
    """
    th1, th2, p1, p2 = y

    # --- Shared Terms ---
    delta = th1 - th2
    c = np.cos(delta)
    s = np.sin(delta)
    den = m1 + m2 * s**2
    den2 = den * den

    # Derivative of denominator wrt th1 (and -th2)
    # d(den)/dth1 = 2 * m2 * sin(delta) * cos(delta) = m2 * sin(2*delta)
    d_den_dth = m2 * np.sin(2 * delta)

    # Constants for denominators
    K1 = 1.0 / (L1**2 * L2)
    K2 = 1.0 / (L1 * L2**2 * m2)

    # --- Rows 0 & 1: Derivatives of dth1/dt and dth2/dt ---
    # We define Numerators (Num1, Num2) to apply quotient rule: (u/v)' = (u'v - uv')/v^2

    # dth1 = (L2*p1 - L1*p2*c) / (L1^2 * L2 * den)
    Num1 = L2 * p1 - L1 * p2 * c
    dth1_val = Num1 * K1 / den  # Recompute value for chain rule usage later

    # Partial derivatives of Num1
    dNum1_dth1 = -L1 * p2 * (-s)  # = L1 p2 s
    dNum1_dth2 = -L1 * p2 * (s)  # = -L1 p2 s
    dNum1_dp1 = L2
    dNum1_dp2 = -L1 * c

    # Jacobian Elements for Row 0 (dth1)
    J00 = K1 * (dNum1_dth1 * den - Num1 * d_den_dth) / den2
    J01 = K1 * (dNum1_dth2 * den - Num1 * (-d_den_dth)) / den2
    J02 = K1 * (dNum1_dp1 * den) / den2  # d_den_dp is 0
    J03 = K1 * (dNum1_dp2 * den) / den2

    # dth2 = (L1*(m1+m2)*p2 - L2*m2*p1*c) / (L1 * L2^2 * m2 * den)
    Num2 = L1 * (m1 + m2) * p2 - L2 * m2 * p1 * c
    dth2_val = Num2 * K2 / den  # Recompute value

    # Partial derivatives of Num2
    dNum2_dth1 = -L2 * m2 * p1 * (-s)  # = L2 m2 p1 s
    dNum2_dth2 = -L2 * m2 * p1 * (s)  # = -L2 m2 p1 s
    dNum2_dp1 = -L2 * m2 * c
    dNum2_dp2 = L1 * (m1 + m2)

    # Jacobian Elements for Row 1 (dth2)
    J10 = K2 * (dNum2_dth1 * den - Num2 * d_den_dth) / den2
    J11 = K2 * (dNum2_dth2 * den - Num2 * (-d_den_dth)) / den2
    J12 = K2 * (dNum2_dp1 * den) / den2
    J13 = K2 * (dNum2_dp2 * den) / den2

    # --- Rows 2 & 3: Derivatives of dp1/dt and dp2/dt ---
    # dp1 = -term1 - term2
    # dp2 =  term1 - term3
    # term1 = C_force * dth1 * dth2 * s
    # term2 = G1 * sin(th1)
    # term3 = G2 * sin(th2)

    C_force = m2 * L1 * L2
    G1 = (m1 + m2) * g * L1
    G2 = m2 * g * L2

    # Precompute term1 parts
    # term1 = C_force * (dth1 * dth2 * s)
    # We use Product Rule: d(uvw) = u'vw + uv'w + uvw'

    # Wrt State variable q (th1, th2, p1, p2):
    # d(term1)/dq = C_force * [ J0q * dth2 * s  +  dth1 * J1q * s  +  dth1 * dth2 * d(s)/dq ]

    # Calculate d(term1)/dq for each component
    # Note: d(s)/dth1 = c, d(s)/dth2 = -c, d(s)/dp = 0

    def d_term1(idx, ds_dq):
        return C_force * (
            J[0, idx] * dth2_val * s
            + dth1_val * J[1, idx] * s
            + dth1_val * dth2_val * ds_dq
        )

    # Initialize J
    J = np.zeros((4, 4))
    J[0, :] = [J00, J01, J02, J03]
    J[1, :] = [J10, J11, J12, J13]

    # Calculate d(term1) vector components
    d_term1_dth1 = d_term1(0, c)
    d_term1_dth2 = d_term1(1, -c)
    d_term1_dp1 = d_term1(2, 0)
    d_term1_dp2 = d_term1(3, 0)

    # Fill Row 2 (dp1) -> -d(term1) - d(term2)
    J[2, 0] = -d_term1_dth1 - G1 * np.cos(th1)  # term2 depends only on th1
    J[2, 1] = -d_term1_dth2
    J[2, 2] = -d_term1_dp1
    J[2, 3] = -d_term1_dp2

    # Fill Row 3 (dp2) -> d(term1) - d(term3)
    J[3, 0] = d_term1_dth1
    J[3, 1] = d_term1_dth2 - G2 * np.cos(th2)  # term3 depends only on th2
    J[3, 2] = d_term1_dp1
    J[3, 3] = d_term1_dp2

    return J


def eom_tangent_linear(t, state_aug, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
    """
    Coupled ODE for Reference + Perturbation using Exact Jacobian.
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


def get_static_matrix(L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
    """
    Returns the linearized system matrix A for the equilibrium state (0,0,0,0).
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


def get_static_propagator(t, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
    """Returns P(t) = exp(A*t)."""
    A = get_static_matrix(L1, L2, m1, m2, g)
    return expm(A * t)


def eom_linear_static(t, y, A):
    """Linearized EOM given pre-computed A."""
    return A @ y
