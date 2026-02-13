"""
QCES Data Assimilation Package.

A dimension-agnostic engine for numerical integration, statistical analysis,
and Bayesian inference, with specific examples for dynamical systems.
"""

# Expose Core Engine Tools
from .core import (
    # Classes
    ProbabilityGrid,
    AssimilationEngine,
    BayesianAssimilationProblem,
    LinearKalmanFilter,
    EnsembleKalmanFilter,
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
    # Visualisation (Grid & General)
    plot_grid_marginal,
    plot_ensemble_scatter,
    plot_1d_slice,
    display_animation_html,
    # Visualisation (Kalman Filters)
    plot_gaussian_ellipsoid,
    plot_kf_step,
    plot_tracker_1d,
)

# Expose the pendulum sub-package
from . import pendulum

__all__ = [
    # Core Classes
    "ProbabilityGrid",
    "AssimilationEngine",
    "BayesianAssimilationProblem",
    "LinearKalmanFilter",
    "EnsembleKalmanFilter",
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
    # Visualisation
    "plot_grid_marginal",
    "plot_ensemble_scatter",
    "plot_1d_slice",
    "display_animation_html",
    "plot_gaussian_ellipsoid",
    "plot_kf_step",
    "plot_tracker_1d",
    # Sub-packages
    "pendulum",
]
