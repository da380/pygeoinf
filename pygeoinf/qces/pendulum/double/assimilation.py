"""
assimilation.py

Data Assimilation tools for the Double Pendulum.
Includes both Particle-based propagation (efficient) and
Grid-based Liouville advection (computationally intensive).
"""

import numpy as np
from .. import core
from . import physics as phys


def propagate_ensemble(initial_conditions, t_points, **physics_params):
    """
    Propagates a particle ensemble (N_samples, 4) forward in time.
    Wrapper around core.solve_ensemble.
    """
    # Extract args in correct order for eom
    L1 = physics_params.get("L1", 1.0)
    L2 = physics_params.get("L2", 1.0)
    m1 = physics_params.get("m1", 1.0)
    m2 = physics_params.get("m2", 1.0)
    g = physics_params.get("g", 9.81)

    return core.solve_ensemble(
        phys.eom, initial_conditions, t_points, args=(L1, L2, m1, m2, g)
    )


def advect_pdf(
    pdf_func,
    t_final,
    theta_lim=(-np.pi, np.pi),
    p_lim=(-5, 5),
    res=15,
    **physics_params,
):
    """
    Advects a 4D PDF using Liouville's theorem.
    WARNING: This creates a grid of size res^4.
    Keep res low (e.g., < 25) to avoid memory overflow.

    Args:
        pdf_func: Callable(th1, th2, p1, p2) -> Density.
        theta_lim: Tuple limits for BOTH angles (th1, th2).
        p_lim: Tuple limits for BOTH momenta (p1, p2).
        res: Points per dimension. Total grid = res^4.

    Returns:
        axes: List of 4 1D arrays [th1, th2, p1, p2].
        Z_initial: 4D array of initial PDF.
        Z_advected: 4D array of final PDF.
    """
    # 1. Setup 4D Limits: [th1, th2, p1, p2]
    grid_limits = [theta_lim, theta_lim, p_lim, p_lim]

    # 2. Physics Args
    L1 = physics_params.get("L1", 1.0)
    L2 = physics_params.get("L2", 1.0)
    m1 = physics_params.get("m1", 1.0)
    m2 = physics_params.get("m2", 1.0)
    g = physics_params.get("g", 9.81)
    eom_args = (L1, L2, m1, m2, g)

    # 3. Call Generic Core
    axes, Z_in, Z_out = core.advect_pdf_grid(
        phys.eom, pdf_func, t_final, grid_limits, res, eom_args
    )

    return axes, Z_in, Z_out
