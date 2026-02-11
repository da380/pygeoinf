"""
QCES Data Assimilation Package.

A dimension-agnostic engine for numerical integration, statistical analysis,
and Bayesian inference, with specific examples for dynamical systems.
"""

# Expose Core Engine Tools
from .core import (
    # Classes
    ProbabilityGrid,
    BayesianAssimilationProblem,
    GaussianLikelihood,
    LinearGaussianLikelihood,
    # Solvers
    solve_trajectory,
    solve_ensemble,
    # Factories
    get_gaussian_pdf,
    get_independent_gaussian_pdf,
    # Math Utilities
    wrap_angle,
    # Analysis & Smoothing
    reanalyse_initial_condition,
    # Visualisation
    plot_grid_marginal,
    plot_ensemble_scatter,
    plot_1d_slice,
    display_animation_html,
)

# Expose the pendulum sub-package
from . import pendulum

__all__ = [
    # Core Classes
    "ProbabilityGrid",
    "BayesianAssimilationProblem",
    "GaussianLikelihood",
    "LinearGaussianLikelihood",
    # Core Solvers
    "solve_trajectory",
    "solve_ensemble",
    # Core Factories
    "get_gaussian_pdf",
    "get_independent_gaussian_pdf",
    # Core Utilities
    "wrap_angle",
    "reanalyse_initial_condition",
    # Core Visualisation
    "plot_grid_marginal",
    "plot_ensemble_scatter",
    "plot_1d_slice",
    "display_animation_html",
    # Sub-packages
    "pendulum",
]
