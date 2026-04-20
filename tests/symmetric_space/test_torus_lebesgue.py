"""
Tests for the Lebesgue function space L² on a 2D Torus.
"""

import pytest
import numpy as np

from pygeoinf.symmetric_space.torus import Lebesgue


@pytest.mark.parametrize("kmax, radius_x, radius_y", [(4, 1.0, 1.0), (6, 2.0, 1.5)])
def test_lebesgue_axioms(kmax: int, radius_x: float, radius_y: float):
    """
    Verifies that the Lebesgue space instance satisfies all Hilbert space axioms
    by calling its internal self-check method.
    """
    space = Lebesgue(kmax, radius_x=radius_x, radius_y=radius_y)
    space.check(n_checks=5)


@pytest.mark.parametrize("kmax, radius_x, radius_y", [(6, 1.0, 1.0)])
class TestTorusLebesgueSpecifics:
    """
    Tests functionalities that are unique to the Lebesgue space on the Torus.
    """

    @pytest.fixture
    def space(self, kmax: int, radius_x: float, radius_y: float) -> Lebesgue:
        """Provides a Lebesgue space instance for the specific tests."""
        return Lebesgue(kmax, radius_x=radius_x, radius_y=radius_y)

    def test_project_constant_function(self, space: Lebesgue):
        """Tests if projecting a constant function f(p) = c works correctly."""
        projected_vector = space.project_function(lambda p: 5.0)
        expected_vector = np.full_like(projected_vector, 5.0)
        assert np.allclose(projected_vector, expected_vector)

    def test_laplacian_eigenvalues(self, space: Lebesgue):
        """Tests the Laplacian eigenvalues against their analytical formula."""
        # Check a specific mode: (kx=2, ky=1)
        indices = space.wavevector_indices(2, 1)
        expected_eval = (2 / space.radius_x) ** 2 + (1 / space.radius_y) ** 2

        for idx in indices:
            assert np.isclose(space.laplacian_eigenvalue(idx), expected_eval)

    def test_transform_round_trip(self, space: Lebesgue):
        """Ensures that transforming to components and back is self-consistent."""
        original_vector = space.project_function(
            lambda p: np.cos(3 * p[0]) + np.sin(p[1])
        )
        components = space.to_components(original_vector)
        reconstructed_vector = space.from_components(components)

        assert np.allclose(original_vector, reconstructed_vector)

    def test_spectral_projection_operator(self, space: Lebesgue):
        """Tests the spectral projection operator axioms and mapping."""
        modes = [(0, 0), (1, 0), (0, 1), (1, 1)]
        op_proj = space.spectral_projection_operator(modes)

        # 1. Test Operator Axioms (This inherently proves adjoint mathematical correctness)
        op_proj.check(n_checks=5)

        # 2. Test Forward Mapping
        u_orig = space.project_function(lambda p: 2.0 + np.cos(p[0]))
        coeffs = op_proj(u_orig)
        assert len(coeffs) == op_proj.codomain.dim

        # 3. Test Adjoint Mapping.
        # NOTE: The adjoint of projection to a Euclidean space scales Torus components by M^-1.
        # For the DC mode (2.0) M=1.0, so 2.0 -> 2.0.
        # For the interior mode (1,0), M=2.0, so cos(x) -> 0.5 * cos(x).
        # Therefore, the peak value should be exactly 2.0 + 0.5 = 2.5.
        u_recon = op_proj.adjoint(coeffs)
        assert np.isclose(np.max(u_recon), 2.5)

    def test_degree_and_with_degree(self, space: Lebesgue):
        """Tests the unified degree property and the with_degree factory."""
        assert space.degree == space.kmax

        target_degree = space.kmax + 2
        new_space = space.with_degree(target_degree)

        assert new_space.degree == target_degree
        assert new_space.radius_x == space.radius_x
        assert new_space.radius_y == space.radius_y
        assert isinstance(new_space, Lebesgue)

    def test_degree_transfer_operator(self, space: Lebesgue):
        """Tests the degree transfer operator's axioms and round-trip behavior."""
        op_up = space.degree_transfer_operator(space.kmax + 2)
        op_up.check(n_checks=5)

        op_down = op_up.codomain.degree_transfer_operator(space.kmax)
        op_down.check(n_checks=5)

        # Test Round-Trip
        u_orig = space.project_function(lambda p: np.cos(2 * p[0]) + np.sin(p[1]))
        u_upsampled = op_up(u_orig)
        u_recon = op_down(u_upsampled)

        assert np.allclose(u_orig, u_recon)


def test_factory_methods():
    """Tests the automatic truncation degree factories for L2 Torus spaces."""
    # Increased kernel_scale to 1.0 so the frequency energy drops off much faster
    space_hk = Lebesgue.from_heat_kernel_prior(
        1.0, radius_x=2.0, radius_y=2.0, min_degree=4, power_of_two=True
    )
    assert isinstance(space_hk, Lebesgue)
    assert space_hk.radius_x == 2.0
    assert space_hk.kmax >= 4
    assert (space_hk.kmax & (space_hk.kmax - 1)) == 0  # Power of two check
