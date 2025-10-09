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
from .operators import SOLAOperator, Laplacian, InverseLaplacian, BesselSobolev, BesselSobolevInverse

# LaplacianInverseOperator (native implementation)

__all__ = [
    'Lebesgue',
    'Function',
    'Sobolev',
    'IntervalDomain',
    'BoundaryConditions',
    'SOLAOperator',
    'Laplacian',
    'InverseLaplacian',
    'BesselSobolev',
    'BesselSobolevInverse'
]
