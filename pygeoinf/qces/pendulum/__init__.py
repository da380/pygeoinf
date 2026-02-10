"""
Pendulum Data Assimilation Package.

A collection of modules for simulating and assimilating data into
pendulum systems. Common solvers and utilities from 'core' are
exposed directly here for convenience.

Modules:
    core   - Dimension-agnostic solvers, statistics, and math utilities.
    single - The 2D Single Pendulum (Hamiltonian dynamics, Grid-based DA).
    double - The 4D Double Pendulum (Chaotic dynamics, Particle-based DA).
"""

# 1. Expose Submodules
from . import core
from . import single
from . import double

# 2. Expose Common Core Utilities (Shortcuts)
from .core import (
    # Solvers
    solve_trajectory,
    solve_ensemble,
    # Statistics & Math
    wrap_angle,
    get_pdf_from_grid,
    evaluate_pdf_on_grid,
    compute_normalization,
    marginalise_grid,
    # Bayesian Inference
    get_independent_gaussian_func,
    get_gaussian_pdf_func,
    bayesian_update,
    gaussian_likelihood,
    GaussianLikelihood,
    LinearGaussianLikelihood,
    sample_from_grid,
    # Visualisation Helpers
    display_animation_html,
)
