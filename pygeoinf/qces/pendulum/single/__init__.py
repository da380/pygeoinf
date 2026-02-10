"""
Single Pendulum Submodule.

Exposes physics, visualization, and grid-based assimilation tools
specifically for the 2D Single Pendulum.
"""

# --- Physics (Hamiltonian & Linearized) ---
from .physics import (
    eom,
    get_coords,
    eom_linear,
    eom_linear_static,
    eom_tangent_linear,
    get_linear_propagator,
    get_jacobian,
)

# --- Data Assimilation (Grid-Based) ---
from .assimilation import (
    advect_pdf,
    generate_synthetic_data,
    assimilate,
    reanalyse_trajectory,
    get_smooth_filter_trajectory,
)

# --- Visualization (2D Specific) ---
from .visualisation import (
    plot_phase_portrait,
    plot_ensemble_stats,
    plot_bayesian_analysis,
    animate_pendulum,
    animate_phase_portrait,
    animate_combined,
    plot_pdf,
    animate_advection,
    plot_trajectory_from_initial_condition,
    plot_gaussian_2d,
    evaluate_gaussian_pdf,
)


__all__ = [
    "eom",
    "get_coords",
    "eom_linear",
    "eom_linear_static",
    "eom_tangent_linear",
    "get_linear_propagator",
    "get_jacobian",
    "advect_pdf",
    "generate_synthetic_data",
    "assimilate",
    "reanalyse_trajectory",
    "get_smooth_filter_trajectory",
    "plot_phase_portrait",
    "plot_ensemble_stats",
    "plot_bayesian_analysis",
    "animate_pendulum",
    "animate_phase_portrait",
    "animate_combined",
    "plot_pdf",
    "animate_advection",
    "plot_trajectory_from_initial_condition",
    "plot_gaussian_2d",
    "evaluate_gaussian_pdf",
]
