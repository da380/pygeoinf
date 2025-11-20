"""
Function providers for the interval module.

This package contains providers for generating functions from various families
(orthogonal bases, bump functions, wavelets, etc.) that can be used independently
or as building blocks for operator construction.

The original monolithic function_providers.py has been split into logical
submodules for better maintainability. This __init__.py re-exports all classes
to preserve backward compatibility with existing code.
"""

# Import base classes
from .base import (
    FunctionProvider,
    IndexedFunctionProvider,
    RestrictedFunctionProvider,
    ParametricFunctionProvider,
    RandomFunctionProvider,
    NullFunctionProvider,
)

# Import concrete providers (demo-critical classes first)
from .normal_modes import NormalModesProvider
from .bump import BumpFunctionProvider
from .fourier import FourierFunctionProvider
from .sine import SineFunctionProvider
from .cosine import CosineFunctionProvider
from .hat import HatFunctionProvider
from .mixed import MixedDNFunctionProvider, MixedNDFunctionProvider
from .robin import RobinFunctionProvider
from .wavelet import WaveletFunctionProvider
from .spline import SplineFunctionProvider
from .boxcar import BoxCarFunctionProvider
from .discontinuous import DiscontinuousFunctionProvider
from .bump_gradient import BumpFunctionGradientProvider
from .kernel import KernelProvider

# All providers have been extracted - no need for dynamic loading

# Public API (same as original function_providers.py)
__all__ = [
    # Base classes
    'FunctionProvider',
    'IndexedFunctionProvider',
    'RestrictedFunctionProvider',
    'ParametricFunctionProvider',
    'RandomFunctionProvider',
    'NullFunctionProvider',
    # Concrete providers (all extracted)
    'NormalModesProvider',
    'BumpFunctionProvider',
    'FourierFunctionProvider',
    'SineFunctionProvider',
    'CosineFunctionProvider',
    'HatFunctionProvider',
    'MixedDNFunctionProvider',
    'MixedNDFunctionProvider',
    'RobinFunctionProvider',
    'WaveletFunctionProvider',
    'SplineFunctionProvider',
    'BoxCarFunctionProvider',
    'DiscontinuousFunctionProvider',
    'BumpFunctionGradientProvider',
    'KernelProvider',
]
