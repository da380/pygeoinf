"""Differential operators on interval domains.

This module provides differential operators for interval domains with various
solution methods:

- Laplacian: -Δ with spectral/finite difference methods
- InverseLaplacian: (-Δ)^(-1) with FEM solvers
- Gradient: ∇ with finite difference
- BesselSobolev: Bessel-Sobolev operators with fast transforms
- BesselSobolevInverse: Inverse Bessel-Sobolev operators
- SOLAOperator: SOLA integration operator
- SpectralOperator: Abstract base class

The operators implement proper function space mappings and support
boundary conditions through the underlying function spaces.
"""

from .base import SpectralOperator
from .gradient import Gradient
from .laplacian import Laplacian, InverseLaplacian
from .bessel import BesselSobolev, BesselSobolevInverse
from .sola import SOLAOperator
from .radial_operators import RadialLaplacian, InverseRadialLaplacian

__all__ = [
    'SpectralOperator',
    'Gradient',
    'Laplacian',
    'InverseLaplacian',
    'BesselSobolev',
    'BesselSobolevInverse',
    'SOLAOperator',
    'RadialLaplacian',
    'InverseRadialLaplacian',
]
