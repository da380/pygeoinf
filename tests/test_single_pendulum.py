import pytest
import numpy as np
from unittest.mock import patch

# Import the new package structure
import pygeoinf.qces.pendulum.core as core
import pygeoinf.qces.pendulum.single as single

# --- Fixtures ---


@pytest.fixture
def default_params():
    return {"L": 1.0, "m": 1.0, "g": 9.81}


@pytest.fixture
def t_points():
    return np.linspace(0, 5, 100)


@pytest.fixture
def small_angle_state():
    """State vector [theta, p] with small angle (linear regime)."""
    return np.array([0.1, 0.0])


# --- 1. Core Module Tests (Math & Solvers) ---


def test_wrap_angle():
    """Test angle wrapping to [-pi, pi]."""
    assert np.isclose(core.wrap_angle(0.1), 0.1)
    assert np.isclose(core.wrap_angle(np.pi + 0.1), -np.pi + 0.1)
    # Boundary case: 3*pi maps to -pi
    assert np.isclose(core.wrap_angle(3 * np.pi), -np.pi)


def test_compute_normalization_2d():
    """Verify integration of a 2D Gaussian sums to 1.0."""
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Standard 2D Gaussian
    sigma = 1.0
    Z = (1 / (2 * np.pi * sigma**2)) * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    norm = core.compute_normalization(Z, [x, y])
    assert np.isclose(norm, 1.0, atol=1e-3)


def test_marginalise_grid():
    """Verify we can integrate out a dimension from a 3D grid."""
    # Create 3 axes
    ax = np.linspace(0, 1, 10)
    grids = np.meshgrid(ax, ax, ax, indexing="ij")
    # Function constant=1.0. Total volume = 1*1*1 = 1.
    Z = np.ones_like(grids[0])

    # Keep axis 0 and 1, integrate out axis 2 (length 1.0)
    new_axes, marg_Z = core.marginalise_grid(Z, [ax, ax, ax], keep_indices=[0, 1])

    assert len(new_axes) == 2
    # The value should essentially be the length of the integrated dimension (1.0)
    assert np.allclose(marg_Z, 1.0, atol=0.1)


# --- 2. Single Physics Tests ---


def test_single_pendulum_period(default_params):
    """
    Test that the non-linear model approximates the linear period
    T = 2*pi*sqrt(L/g) for small angles.
    """
    L, g = default_params["L"], default_params["g"]
    expected_period = 2 * np.pi * np.sqrt(L / g)

    # Integrate for exactly one period
    t_span = np.linspace(0, expected_period, 50)

    # Use a truly small angle (0.001) to match linear theory tolerances
    y0 = [0.001, 0.0]

    # Using generic solver with specific single physics
    sol = core.solve_trajectory(single.eom, y0, t_span, args=(L, 1.0, g))
    y_final = sol[:, -1]

    # Should return to initial state
    np.testing.assert_allclose(y_final, y0, rtol=1e-2, atol=1e-4)


def test_energy_conservation(default_params):
    """
    Hamiltonian H = p^2/(2mL^2) - m g L cos(theta) should be constant.
    """
    L, m, g = default_params["L"], default_params["m"], default_params["g"]
    t_span = np.linspace(0, 10, 100)
    y0 = [np.pi / 2, 0.0]  # Start at 90 degrees (high energy)

    sol = core.solve_trajectory(single.eom, y0, t_span, args=(L, m, g))
    theta, p = sol[0], sol[1]

    # Calculate Energy over time
    H = (p**2) / (2 * m * L**2) - (m * g * L * np.cos(theta))

    # Check that standard deviation of Energy is near zero (constant)
    assert np.std(H) < 1e-3


def test_linearization_accuracy(default_params):
    """Compare exact non-linear EOM vs Linearized EOM for small angles."""
    y0 = [0.01, 0.0]
    t = np.linspace(0, 1, 10)
    args = (default_params["L"], default_params["m"], default_params["g"])

    # 1. Non-linear
    sol_nl = core.solve_trajectory(single.eom, y0, t, args=args)

    # 2. Linear Static
    sol_lin = core.solve_trajectory(single.eom_linear, y0, t, args=args)

    # Should be almost identical
    np.testing.assert_allclose(sol_nl, sol_lin, atol=1e-4)


# --- 3. Assimilation Tests ---


def test_advect_pdf_conservation():
    """Test that Liouville advection preserves total probability mass."""

    # Define a simple Gaussian PDF centered at 0
    def gauss_pdf(X, Y):
        return np.exp(-(X**2 + Y**2))

    # Use wider limits so mass doesn't leave the grid
    x_lim = (-5, 5)
    y_lim = (-5, 5)

    t_final = 0.5
    X, Y, Z_in, Z_out = single.advect_pdf(
        gauss_pdf,
        t_final=t_final,
        x_lim=x_lim,
        y_lim=y_lim,
        res=30,
        L=1.0,
        m=1.0,
        g=9.81,
    )

    # Reconstruct axes from the meshgrid limits for normalization check
    x_axis = X[:, 0]
    y_axis = Y[0, :]
    axes = [x_axis, y_axis]

    norm_in = core.compute_normalization(Z_in, axes)
    norm_out = core.compute_normalization(Z_out, axes)

    # Both should be normalized to ~1.0
    assert np.isclose(norm_in, 1.0, atol=1e-3)
    assert np.isclose(norm_out, 1.0, atol=0.1)


# --- 4. Visualisation "Smoke" Tests ---


@patch("matplotlib.pyplot.show")
def test_visualisation_smoke(mock_show, t_points):
    """Ensure plotting functions run without errors."""
    # Create dummy trajectory data
    sol = np.zeros((2, len(t_points)))
    sol[0, :] = np.sin(t_points)
    sol[1, :] = np.cos(t_points)

    # Create dummy ensemble data (samples, dim, time)
    ensemble = np.zeros((5, 2, len(t_points)))
    ensemble[:, 0, :] = np.sin(t_points)
    # FIX: Add non-zero momentum so ylim doesn't crash with [0, 0]
    ensemble[:, 1, :] = 0.5 * np.cos(t_points)

    # 1. Static Plot
    single.plot_phase_portrait(ensemble, t_points)

    # 2. Animations (just creation)
    anim = single.animate_pendulum(t_points, sol)
    assert anim is not None

    anim2 = single.animate_phase_portrait(ensemble, t_points)
    assert anim2 is not None

    # 3. Bayesian Plot
    # Create dummy grids
    grid = np.zeros((10, 10))
    X, Y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    single.plot_bayesian_analysis(X, Y, grid, grid, grid, 0.5, 1.0)
