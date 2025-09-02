"""
Interval Function Spaces Module

This module contains implementations for various function spaces on intervals,
including:

- Sobolev spaces on intervals with spectral methods
- Native finite element computations
- Inverse elliptic operators for covariance modeling

Modules:
    interval_space: Sobolev spaces on 1D intervals
    sobolev_functions: Function objects for Sobolev spaces
    interval_domain: Domain utilities for intervals
    inverse_elliptical: Inverse elliptic operators for PDEs

Classes:
    Sobolev: Sobolev space implementation with spectral methods
    SobolevFunction: Function living in Sobolev spaces
    InverseEllipticOperator: Inverse elliptic PDE operators
"""

# Core functionality that exists
from .lebesgue_space import Lebesgue
from .functions import Function
from .sobolev_space import Sobolev
from .interval_domain import IntervalDomain
from .boundary_conditions import BoundaryConditions

# Backward compatibility alias during transition
# TODO: Remove this once all code is migrated
L2Space = Lebesgue

# LaplacianInverseOperator (native implementation)
try:
    import importlib
    if importlib.util.find_spec("pygeoinf.interval.laplacian_operator"):
        LAPLACIAN_OPERATOR_AVAILABLE = True
    else:
        LAPLACIAN_OPERATOR_AVAILABLE = False
except ImportError:
    LAPLACIAN_OPERATOR_AVAILABLE = False

__all__ = [
    'Lebesgue',
    'L2Space',  # Backward compatibility alias
    'Function',
    'Sobolev',
    'IntervalDomain',
    'BoundaryConditions',
]

if LAPLACIAN_OPERATOR_AVAILABLE:
    __all__.append('LaplacianInverseOperator')
