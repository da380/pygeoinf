"""
Tests for the Sobolev space implementation on a line.
"""

import pytest
from pygeoinf.symmetric_space.line import Sobolev
from ..checks.hilbert_space import HilbertSpaceChecks


@pytest.fixture
def space() -> Sobolev:
    """Provides a Sobolev space instance on a line for the tests."""
    # Using a relatively low kmax for efficiency in testing
    return Sobolev(16, 1.0, 0.1)


class TestLineSobolev(HilbertSpaceChecks):
    """
    Runs the standard suite of Hilbert space checks on the line.Sobolev class.
    """

    pass
