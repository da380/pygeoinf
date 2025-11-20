"""
Shared utilities for the interval module.

This submodule contains common utilities used across multiple interval components,
reducing code duplication and improving maintainability.
"""

from .robin_utils import RobinRootFinder

__all__ = [
    'RobinRootFinder',
]
