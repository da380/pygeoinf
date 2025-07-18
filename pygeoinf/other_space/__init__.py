"""
Other Space Module

This module contains implementations for various function spaces beyond the standard
pygeoinf spaces, including:

- Sobolev spaces on intervals with spectral methods
- DOLFINx integration for finite element computations
- Inverse elliptic operators for covariance modeling
- Bridge classes for framework integration

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

# Core Sobolev space functionality
from .interval_space import Sobolev
from .sobolev_functions import SobolevFunction
from .interval_domain import IntervalDomain

# DOLFINx integration (optional, requires DOLFINx installation)
try:
    from .dolfinx_bridge import DOLFINxSobolevBridge, DOLFINxBridgeFactory
    DOLFINX_BRIDGE_AVAILABLE = True
except ImportError:
    DOLFINX_BRIDGE_AVAILABLE = False

# Inverse operators (may require DOLFINx for full functionality)
try:
    from .inverse_elliptical import InverseEllipticOperator
    INVERSE_OPERATORS_AVAILABLE = True
except ImportError as e:
    # This is expected if DOLFINx is not installed
    INVERSE_OPERATORS_AVAILABLE = False
    _inverse_import_error = str(e)

__all__ = [
    'Sobolev',
    'SobolevFunction',
    'IntervalDomain'
]

# Add optional imports to __all__ if available
if DOLFINX_BRIDGE_AVAILABLE:
    __all__.extend(['DOLFINxSobolevBridge', 'DOLFINxBridgeFactory'])

if INVERSE_OPERATORS_AVAILABLE:
    __all__.extend(['InverseEllipticOperator'])

# Version info
__version__ = "0.1.0"
__author__ = "Adrian-Mag"
