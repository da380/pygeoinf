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
from .lebesgue_space import Lebesgue, LebesgueSpaceDirectSum
from .functions import Function
from .sobolev_space import Sobolev, SobolevSpaceDirectSum
from .interval_domain import IntervalDomain
from .boundary_conditions import BoundaryConditions
from .linear_form_kernel import LinearFormKernel
from .KL_sampler import KLSampler
from .configs import (
    IntegrationConfig,
    LebesgueIntegrationConfig,
    ParallelConfig,
    LebesgueParallelConfig
)
from .operators import (
    SOLAOperator,
    Laplacian,
    InverseLaplacian,
    BesselSobolev,
    BesselSobolevInverse,
    SpectralOperator,
    Gradient
)
from .radial_operators import (
    RadialLaplacian,
    InverseRadialLaplacian
)
from .providers import SpectrumProvider

# LaplacianInverseOperator (native implementation)

__all__ = [
    'Lebesgue',
    'Function',
    'Sobolev',
    'SobolevSpaceDirectSum',
    'IntervalDomain',
    'BoundaryConditions',
    'SOLAOperator',
    'Laplacian',
    'InverseLaplacian',
    'BesselSobolev',
    'BesselSobolevInverse',
    'SpectralOperator',
    'SpectrumProvider',
    'Gradient',
    'LebesgueSpaceDirectSum',
    'RadialLaplacian',
    'InverseRadialLaplacian',
    'LinearFormKernel',
    'IntegrationConfig',
    'LebesgueIntegrationConfig',
    'ParallelConfig',
    'LebesgueParallelConfig',
    'KLSampler'
]
