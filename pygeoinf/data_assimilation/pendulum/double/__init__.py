"""
Double Pendulum Submodule.

Focuses on chaotic dynamics, phase space visualisation, and
Ensemble (particle) experiments.
"""

# --- Physics (Hamiltonian & Linearised) ---
from .physics import (
    eom,
    get_coords,
    get_jacobian,
    eom_tangent_linear,
    get_linear_matrix,
    get_linear_propagator,
    eom_linear,
)

# --- Visualisation (Chaos & Ensembles) ---
from .visualisation import (
    plot_sensitivity_divergence,
    plot_ensemble_phase_space,
    animate_pendulum,
    animate_ensemble_phase_space,
)

__all__ = [
    # Physics
    "eom",
    "get_coords",
    "get_jacobian",
    "eom_tangent_linear",
    "get_linear_matrix",
    "get_linear_propagator",
    "eom_linear",
    # Visualisation
    "plot_sensitivity_divergence",
    "plot_ensemble_phase_space",
    "animate_pendulum",
    "animate_ensemble_phase_space",
]
