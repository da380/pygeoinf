"""
Tests for the Sobolev space implementation on a sphere.
"""

import pytest
from pygeoinf.symmetric_space.sphere import Sobolev
from ..checks.hilbert_space import HilbertSpaceChecks


@pytest.fixture
def space() -> Sobolev:
    """Provides a Sobolev space instance on a sphere for the tests."""
    # Using a relatively low lmax for efficiency in testing
    return Sobolev(16, 1.0, 0.1)


class TestSphereSobolev(HilbertSpaceChecks):
    """
    Runs the standard suite of Hilbert space checks on the sphere.Sobolev class.
    """

    pass
