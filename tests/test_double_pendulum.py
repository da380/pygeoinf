import pytest
import numpy as np
from unittest.mock import patch

import pygeoinf.qces.pendulum.core as core
import pygeoinf.qces.pendulum.double as double

# --- Fixtures ---


@pytest.fixture
def default_params():
    return {"L1": 1.0, "L2": 1.0, "m1": 1.0, "m2": 1.0, "g": 9.81}


@pytest.fixture
def t_points():
    return np.linspace(0, 5, 100)


@pytest.fixture
def stable_equilibrium():
    return np.array([0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def random_state():
    """A generic non-trivial state (angles ~ 45 deg, some momentum)."""
    return np.array([0.5, -0.5, 0.2, -0.1])


# --- 1. Physics & Jacobian Tests ---


def test_jacobian_accuracy(random_state, default_params):
    """
    CRITICAL: Verify the manual analytic Jacobian matches Finite Differences.
    """
    args = tuple(default_params.values())  # L1, L2, m1, m2, g

    # 1. Compute Analytic Jacobian
    J_analytic = double.get_jacobian(random_state, *args)

    # 2. Compute Finite Difference Jacobian
    def func(y):
        return np.array(double.eom(0, y, *args))

    eps = 1e-7
    n = len(random_state)
    J_fd = np.zeros((n, n))

    for i in range(n):
        y_plus = random_state.copy()
        y_plus[i] += eps
        y_minus = random_state.copy()
        y_minus[i] -= eps
        J_fd[:, i] = (func(y_plus) - func(y_minus)) / (2 * eps)

    # 3. Compare
    # Tolerances: rel=1e-5 should pass easily if math is correct
    np.testing.assert_allclose(J_analytic, J_fd, rtol=1e-5, atol=1e-6)


def test_energy_conservation(default_params):
    """
    Verify Total Energy (Hamiltonian) is conserved for a chaotic trajectory.
    """
    L1, L2, m1, m2, g = default_params.values()

    # FIX: Use a state with NON-ZERO total energy.
    # Previous state [pi/2, pi/2, 0, 0] had V=0 and T=0 => E=0.
    # New state: Bob 1 at ~57 deg, Bob 2 vertical relative to grid.
    y0 = [1.0, 0.0, 0.0, 0.0]

    t_span = np.linspace(0, 5, 100)

    sol = core.solve_trajectory(double.eom, y0, t_span, args=(L1, L2, m1, m2, g))

    # Calculate Energy at each step
    energies = []
    for i in range(len(t_span)):
        state = sol[:, i]
        th1, th2, p1, p2 = state

        # To get Kinetic T, we need dth1, dth2.
        # The EOM function computes these as the first 2 return values.
        dth1, dth2, _, _ = double.eom(0, state, L1, L2, m1, m2, g)

        # Kinetic Energy T
        # T = 0.5(m1+m2)(L1 th1_dot)^2 + 0.5 m2(L2 th2_dot)^2 + m2 L1 L2 th1_dot th2_dot cos(th1-th2)
        T = (
            0.5 * (m1 + m2) * (L1 * dth1) ** 2
            + 0.5 * m2 * (L2 * dth2) ** 2
            + m2 * L1 * L2 * dth1 * dth2 * np.cos(th1 - th2)
        )

        # Potential Energy V (y is up)
        y1_pos = -L1 * np.cos(th1)
        y2_pos = y1_pos - L2 * np.cos(th2)
        V = m1 * g * y1_pos + m2 * g * y2_pos

        energies.append(T + V)

    energies = np.array(energies)

    # Energy should be constant (std dev near zero)
    # Check relative error against the mean energy
    assert np.std(energies) / np.mean(np.abs(energies)) < 1e-4


def test_static_linearization_small_angle(default_params):
    """
    Test that static linearized model matches full non-linear model for tiny angles.
    """
    args = tuple(default_params.values())
    y0 = [0.001, 0.001, 0.0, 0.0]  # Very small perturbation
    t = np.linspace(0, 1.0, 10)

    # 1. Non-linear
    sol_nl = core.solve_trajectory(double.eom, y0, t, args=args)

    # 2. Linear (Matrix Exponential)
    P = double.get_static_propagator(1.0, *args)
    y_linear_final = P @ y0

    np.testing.assert_allclose(sol_nl[:, -1], y_linear_final, rtol=1e-2, atol=1e-4)


# --- 2. Assimilation Tests ---


def test_ensemble_propagation(default_params):
    """Ensure we can propagate a batch of particles."""
    n_particles = 10
    y0_ensemble = np.random.rand(n_particles, 4) * 0.1  # Small random states
    t_points = np.linspace(0, 1, 5)

    params = default_params
    trajs = double.propagate_ensemble(y0_ensemble, t_points, **params)

    assert trajs.shape == (n_particles, 4, 5)
    # Check that they moved from initial state
    assert not np.allclose(trajs[:, :, -1], trajs[:, :, 0])


def test_advect_pdf_integration():
    """
    Test 4D grid advection conservation.
    WARNING: Keeping resolution LOW (res=8) to keep test fast.
    """

    # Simple Gaussian in 4D
    def pdf_4d(th1, th2, p1, p2):
        dist = th1**2 + th2**2 + p1**2 + p2**2
        return np.exp(-dist)

    # Coarse grid, small time
    axes, Z_in, Z_out = double.advect_pdf(
        pdf_4d, t_final=0.1, theta_lim=(-2, 2), p_lim=(-2, 2), res=8
    )

    norm_in = core.compute_normalization(Z_in, axes)
    norm_out = core.compute_normalization(Z_out, axes)

    assert np.isclose(norm_in, 1.0, atol=1e-3)
    # With crude resolution (8^4 points), tolerance must be loose
    assert np.isclose(norm_out, 1.0, atol=0.2)


# --- 3. Visualisation Smoke Tests ---


@patch("matplotlib.pyplot.show")
def test_visualisation_smoke(mock_show, t_points):
    """Ensure plotting functions run without errors."""

    # Dummy data
    ensemble = np.random.rand(5, 4, len(t_points))  # 5 particles, 4 states
    sol = np.zeros((4, len(t_points)))

    # 1. Phase Projections
    double.plot_phase_projections(ensemble, t_points)

    # 2. Animation
    anim = double.animate_pendulum(t_points, sol)
    assert anim is not None

    # 3. Marginals (Grid based)
    # Create fake 4D grid (4x4x4x4)
    ax = np.linspace(0, 1, 4)
    axes = [ax, ax, ax, ax]
    grid_4d = np.random.rand(4, 4, 4, 4)

    double.plot_marginal_pdf(axes, grid_4d, dims=(0, 1))
    double.plot_both_phase_marginals(axes, grid_4d)
