"""
Symmetric space function spaces and utilities.
"""

import importlib

from .symmetric_space import (
    SymmetricHilbertSpace,
    AbstractSymmetricLebesgueSpace,
    SymmetricSobolevSpace,
    InvariantLinearAutomorphism,
    InvariantGaussianMeasure,
)

__all__ = [
    "SymmetricHilbertSpace",
    "AbstractSymmetricLebesgueSpace",
    "SymmetricSobolevSpace",
    "InvariantLinearAutomorphism",
    "InvariantGaussianMeasure",
]


def __getattr__(name):
    """Lazily load submodules when requested."""
    if name in ("circle", "line", "sphere"):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Ensure IDE autocompletion still sees the lazy submodules."""
    return __all__ + ["circle", "line", "sphere"]
