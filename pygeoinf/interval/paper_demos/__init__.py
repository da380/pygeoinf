"""
Paper demos: sensitivity kernel utilities package

This subpackage collects the loader, catalog, interpolator,
discontinuity handler and provider implementations used in the
paper demos. It is intentionally a thin package that re-exports
the main classes and helpers for convenience.
"""

from .sensitivity_kernel_loader import (
    SensitivityKernelData,
    parse_mode_id,
    format_mode_id,
    load_kernel_file,
    parse_header,
)
from .sensitivity_kernel_catalog import SensitivityKernelCatalog
from .depth_coordinates import DepthCoordinateSystem, EARTH_RADIUS_KM
from .kernel_interpolator import KernelInterpolator, compare_interpolation_methods
from .discontinuity_kernels import DiscontinuityKernel
from .sensitivity_kernel_provider import SensitivityKernelProvider

__all__ = [
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
]
