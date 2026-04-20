"""
Tests for the Sobolev function space Hˢ on a compact 2D Plane.
"""

import pytest
import numpy as np

from pygeoinf.symmetric_space.plane import Sobolev


@pytest.mark.parametrize(
    "kmax, ax, bx, cx, ay, by, cy", [(4, 0.0, 1.0, 0.1, 0.0, 1.0, 0.1)]
)
def test_sobolev_axioms(
    kmax: int, ax: float, bx: float, cx: float, ay: float, by: float, cy: float
):
    """
    Verifies that the Plane Sobolev space instance satisfies all Hilbert space axioms
    by calling its internal self-check method.
    """
    # Note: Order must be > 1.0 on a 2D manifold for continuous point evaluations
    space = Sobolev(kmax, 1.5, 0.5, ax=ax, bx=bx, cx=cx, ay=ay, by=by, cy=cy)
    space.check(n_checks=5)


class TestPlaneSobolevSpecific:
    """
    Tests functionalities specific to the Sobolev nature of the space on the Plane.
    """

    @pytest.fixture(params=[(1.5, 0.5), (2.0, 1.0)])
    def sobolev_space(self, request) -> Sobolev:
        """Provides a Sobolev space with varying order and scale."""
        order, scale = request.param
        return Sobolev(6, order, scale, ax=0.0, bx=2.0, cx=0.2, ay=0.0, by=2.0, cy=0.2)

    def test_dirac_functional_property(self, sobolev_space: Sobolev):
        """
        Tests that applying the Dirac functional δ_p to a function f
        correctly evaluates the function at point p, i.e., <δ_p, f> = f(p).

        The test point is placed strictly inside the computational domain
        to avoid the tapering regions where the projected function is artificially damped.
        """
        space = sobolev_space
        test_point = (1.0, 1.0)  # Dead center of [0, 2] x [0, 2]

        def test_func(p: tuple[float, float]) -> float:
            return np.cos(p[0]) + 2 * np.sin(2 * p[1])

        dirac_functional = space.dirac(test_point)
        func_vector = space.project_function(test_func)

        functional_evaluation = space.duality_product(dirac_functional, func_vector)
        direct_evaluation = test_func(test_point)

        # Allow slight leniency due to finite kmax spectral ringing
        assert np.isclose(
            functional_evaluation, direct_evaluation, rtol=1e-2, atol=1e-2
        )

    def test_dirac_order_requirement(self):
        """
        Tests that the dirac functional raises an error for Sobolev
        order <= 1.0, as required by theory for a 2D manifold.
        """
        low_order_space = Sobolev(
            6, 1.0, 1.0, ax=0.0, bx=1.0, cx=0.1, ay=0.0, by=1.0, cy=0.1
        )
        with pytest.raises(NotImplementedError):
            low_order_space.dirac((0.5, 0.5))

    def test_sobolev_coefficient_operators_axioms(self, sobolev_space: Sobolev):
        """
        Verifies that the Sobolev coefficient operators satisfy all
        LinearOperator axioms (including mass-weighted adjoints).
        """
        kmax = sobolev_space.kmax

        op_to = sobolev_space.to_coefficient_operator(kmax)
        op_to.check(n_checks=5)

        op_from = sobolev_space.from_coefficient_operator(kmax)
        op_from.check(n_checks=5)

    def test_spectral_projection_operator_axioms(self, sobolev_space: Sobolev):
        """
        Verifies that the Sobolev spectral projection satisfies LinearOperator
        axioms, meaning the adjoint correctly applies the inverse mass metric.
        """
        modes = [(0, 0), (1, 0), (0, 1), (1, 1)]
        op_proj = sobolev_space.spectral_projection_operator(modes)
        op_proj.check(n_checks=5)

    def test_degree_and_with_degree(self, sobolev_space: Sobolev):
        """Tests the unified degree property and the with_degree factory."""
        space = sobolev_space
        assert space.degree == space.kmax

        target_degree = space.kmax + 2
        new_space = space.with_degree(target_degree)

        assert new_space.degree == target_degree
        assert new_space.bounds_x == space.bounds_x
        assert new_space.bounds_y == space.bounds_y
        assert new_space.order == space.order
        assert new_space.scale == space.scale


def test_factory_methods():
    """Tests the automatic truncation degree factories for Sobolev spaces."""
    # Note: We use a larger kernel_scale (1.0) so the energy spectrum decays quickly,
    # ensuring the truncation estimation loop finishes in milliseconds.
    space = Sobolev.from_sobolev_kernel_prior(
        4.0,
        1.0,
        1.5,
        0.5,
        ax=0.0,
        bx=2.0,
        cx=0.5,
        ay=0.0,
        by=2.0,
        cy=0.5,
        min_degree=4,
        power_of_two=True,
    )
    assert isinstance(space, Sobolev)
    assert space.order == 1.5
    assert space.scale == 0.5
    assert space.bounds_x == (0.0, 2.0, 0.5)
    assert space.bounds_y == (0.0, 2.0, 0.5)
    assert space.kmax >= 4
    assert (space.kmax & (space.kmax - 1)) == 0  # Power of two check
