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
from .lebesgue_space import (
    Lebesgue,
    LebesgueSpaceDirectSum,
    KnownRegion,
    PartitionedLebesgueSpace,
    RestrictedKernelProvider,
)
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
from .operators.radial_operators import (
    RadialLaplacian,
    InverseRadialLaplacian
)
from .providers import SpectrumProvider

# Sensitivity kernel modules (re-exported from paper_demos)
from .paper_demos.sensitivity_kernel_loader import (
    SensitivityKernelData,
    parse_mode_id,
    format_mode_id,
    load_kernel_file,
    parse_header,
)
from .paper_demos.sensitivity_kernel_catalog import SensitivityKernelCatalog
from .paper_demos.depth_coordinates import (
    DepthCoordinateSystem,
    EARTH_RADIUS_KM,
)
from .paper_demos.kernel_interpolator import (
    KernelInterpolator,
    compare_interpolation_methods,
)
from .paper_demos.discontinuity_kernels import DiscontinuityKernel
from .paper_demos.sensitivity_kernel_provider import SensitivityKernelProvider

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
    'KLSampler',
    # Sensitivity kernels
    'SensitivityKernelData',
    'parse_mode_id',
    'format_mode_id',
    'load_kernel_file',
    'parse_header',
    'SensitivityKernelCatalog',
    'DepthCoordinateSystem',
    'EARTH_RADIUS_KM',
    'KernelInterpolator',
    'compare_interpolation_methods',
    'DiscontinuityKernel',
    'SensitivityKernelProvider',
    # Partitioned model spaces
    'KnownRegion',
    'PartitionedLebesgueSpace',
    'RestrictedKernelProvider',
]
