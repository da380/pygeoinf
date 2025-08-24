"""
Tests for function spaces on a circle. 
"""

import pytest
import numpy as np
from pygeoinf.symmetric_space_new.sphere import Lebesgue

from ..checks.hilbert_space import HilbertSpaceChecks

# from ..checks.linear_operator import LinearOperatorChecks


@pytest.fixture
def space() -> Lebesgue:
    """Provides a Lebesgue space instance on a circle for the tests."""
    # Using a relatively low kmax for efficiency in testing
    return Lebesgue(16)


class TestCircleLebesgue(HilbertSpaceChecks):
    """
    Runs the standard suite of Hilbert space checks on the circle.Lebesgue class.
    """
