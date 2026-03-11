"""
Tests for Sobolev function spaces on a line segment (interval).
"""

import pytest
import numpy as np

from pygeoinf.symmetric_space.line import Sobolev


@pytest.mark.parametrize("kmax, a, b, c", [(16, 0.0, 1.0, 0.1), (32, -1.5, 1.5, 0.2)])
def test_sobolev_axioms(kmax: int, a: float, b: float, c: float):
    """
    Verifies that the Sobolev space instance satisfies all Hilbert space axioms
    by calling its internal self-check method.
    """
    space = Sobolev(kmax, 2.0, 0.5, a=a, b=b, c=c)
    space.check(n_checks=5)


class TestLineSobolevSpecific:
    """
    Tests functionalities that are specific to the Sobolev nature of the space,
    particularly those dependent on order and scale.
    """

    @pytest.fixture(params=[(1.0, 0.5), (2.0, 1.0)])
    def sobolev_space(self, request) -> Sobolev:
        """Provides a Sobolev space with varying order and scale."""
        order, scale = request.param
        return Sobolev(16, order, scale, a=0.0, b=1.0, c=0.1)

    def test_dirac_functional_property(self, sobolev_space: Sobolev):
        """Tests that <δ_p, f> = f(p) for different Sobolev parameters."""
        space = sobolev_space
        test_point = 0.5

        test_func = lambda x: np.exp(-((x - 0.5) ** 2) / 0.02)

        dirac_representation = space.dirac_representation(test_point)
        func_vector = space.project_function(test_func)

        inner_product_result = space.inner_product(dirac_representation, func_vector)
        direct_evaluation_result = test_func(test_point)

        assert np.isclose(inner_product_result, direct_evaluation_result)

    def test_dirac_order_requirement(self):
        """
        Tests that the dirac functional raises an error for Sobolev
        order <= 0.5, as required by theory.
        """
        low_order_space = Sobolev(16, 0.5, 1.0, a=0.0, b=1.0, c=0.1)
        with pytest.raises(NotImplementedError):
            low_order_space.dirac(0.5)
