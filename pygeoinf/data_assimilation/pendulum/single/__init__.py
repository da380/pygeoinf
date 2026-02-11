"""
Single Pendulum Submodule.

Exposes physics and visualization tools specifically for the 2D Single Pendulum.
"""

# --- Physics (Hamiltonian & Linearised) ---
from .physics import (
    eom,
    get_coords,
    eom_linear,
    eom_tangent_linear,
    get_linear_propagator,
    get_jacobian,
)

# --- Visualisation (2D Specific) ---
from .visualisation import (
    plot_phase_portrait,
    plot_bayesian_analysis,
    plot_trajectory_from_initial_condition,
    animate_pendulum,
    animate_phase_portrait,
    animate_combined,
    animate_advection,
    animate_linear_comparison,
)

__all__ = [
    # Physics
    "eom",
    "get_coords",
    "eom_linear",
    "eom_tangent_linear",
    "get_linear_propagator",
    "get_jacobian",
    # Visualisation
    "plot_phase_portrait",
    "plot_bayesian_analysis",
    "plot_trajectory_from_initial_condition",
    "animate_pendulum",
    "animate_phase_portrait",
    "animate_combined",
    "animate_advection",
    "animate_linear_comparison",
]
