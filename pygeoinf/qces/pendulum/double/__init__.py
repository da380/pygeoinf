"""
Double Pendulum Submodule.

Exposes physics, visualization, and assimilation tools for the 4D chaotic pendulum.
"""

# --- Physics (Hamiltonian & Linearized) ---
from .physics import (
    eom,
    get_coords,
    eom_tangent_linear,
    get_jacobian,
    get_static_matrix,
    get_static_propagator,
    eom_linear_static,
)

# --- Data Assimilation (Ensemble & Grid) ---
from .assimilation import propagate_ensemble, advect_pdf

# --- Visualization (4D Projections & Marginals) ---
from .visualisation import (
    animate_pendulum,
    plot_phase_projections,
    plot_marginal_pdf,
    plot_both_phase_marginals,
)
