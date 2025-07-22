"""
Other Space Module

This module contains implementations for various function spaces beyond the
standard pygeoinf spaces, including:

- Sobolev spaces on intervals with spectral methods
- DOLFINx integration for finite element computations
- Inverse elliptic operators for covariance modeling

Modules:
    interval_space: Sobolev spaces on 1D intervals
    sobolev_functions: Function objects for Sobolev spaces
    interval_domain: Domain utilities for intervals
    dolfinx_bridge: Bridge between pygeoinf and DOLFINx
    inverse_elliptical: Inverse elliptic operators for PDEs

Classes:
    Sobolev: Sobolev space implementation with spectral methods
    SobolevFunction: Function living in Sobolev spaces
    DOLFINxSobolevBridge: Bridge for pygeoinf-DOLFINx integration
    InverseEllipticOperator: Inverse elliptic PDE operators
"""

# Core functionality that exists
from .l2_space import L2Space
from .l2_functions import L2Function
from .sobolev_space import Sobolev
from .interval_domain import IntervalDomain

# LaplacianInverseOperator (requires DOLFINx)
try:
    import importlib
    if importlib.util.find_spec("pygeoinf.interval.laplacian_operator"):
        LAPLACIAN_OPERATOR_AVAILABLE = True
    else:
        LAPLACIAN_OPERATOR_AVAILABLE = False
except ImportError:
    LAPLACIAN_OPERATOR_AVAILABLE = False

__all__ = [
    'L2Space',
    'L2Function',
    'Sobolev',
    'IntervalDomain',
]

if LAPLACIAN_OPERATOR_AVAILABLE:
    __all__.append('LaplacianInverseOperator')
