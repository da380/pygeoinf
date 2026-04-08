"""
Tests for the Lebesgue function space L² on a compact 2D Plane.
"""

import pytest
import numpy as np

from pygeoinf.symmetric_space.plane import Lebesgue


@pytest.mark.parametrize(
    "kmax, ax, bx, cx, ay, by, cy",
    [(4, 0.0, 1.0, 0.1, 0.0, 1.0, 0.1), (6, -1.0, 1.0, 0.5, -2.0, 2.0, 0.2)],
)
def test_lebesgue_axioms(
    kmax: int, ax: float, bx: float, cx: float, ay: float, by: float, cy: float
):
    """
    Verifies that the Plane Lebesgue space instance satisfies all Hilbert space axioms
    by calling its internal self-check method. This ensures delegation to the Torus is flawless.
    """
    space = Lebesgue(kmax, ax=ax, bx=bx, cx=cx, ay=ay, by=by, cy=cy)
    space.check(n_checks=5)


@pytest.mark.parametrize(
    "kmax, ax, bx, cx, ay, by, cy", [(6, 0.0, 2.0, 0.2, 0.0, 1.0, 0.1)]
)
class TestPlaneLebesgueSpecifics:
    """
    Tests functionalities that are unique to the Lebesgue space on the Plane,
    specifically the Fourier Continuation (tapering) boundaries.
    """

    @pytest.fixture
    def space(self, kmax, ax, bx, cx, ay, by, cy) -> Lebesgue:
        """Provides a Lebesgue space instance for the specific tests."""
        return Lebesgue(kmax, ax=ax, bx=bx, cx=cx, ay=ay, by=by, cy=cy)

    def test_project_constant_function_with_tapering(self, space: Lebesgue):
        """
        Tests if projecting a constant function f(p) = c works correctly,
        ensuring it remains constant inside the bounds and tapers to 0 outside.
        """
        constant_val = 5.0
        projected_vector = space.project_function(lambda p: constant_val)

        X_flat, Y_flat = space.points()

        # 1. Check STRICT INTERIOR (should be exactly 5.0)
        interior_mask = (
            (X_flat >= space.bounds_x[0])
            & (X_flat <= space.bounds_x[1])
            & (Y_flat >= space.bounds_y[0])
            & (Y_flat <= space.bounds_y[1])
        )
        # Flatten the 2D grid before applying the 1D boolean mask
        assert np.allclose(projected_vector.flatten()[interior_mask], constant_val)

        # 2. Check ABSOLUTE BOUNDARY (should be exactly 0.0 at the absolute edge of padding)
        ax, bx, cx = space.bounds_x
        ay, by, cy = space.bounds_y

        # Allow a tiny epsilon for floating point grid placement
        eps = 1e-10
        edge_mask = (
            (X_flat <= ax - cx + eps)
            | (X_flat >= bx + cx - eps)
            | (Y_flat <= ay - cy + eps)
            | (Y_flat >= by + cy - eps)
        )
        assert np.allclose(projected_vector.flatten()[edge_mask], 0.0)

    def test_transform_round_trip(self, space: Lebesgue):
        """Ensures that transforming to components and back is self-consistent."""
        original_vector = space.project_function(
            lambda p: np.cos(3 * p[0]) + np.sin(2 * p[1])
        )
        components = space.to_components(original_vector)
        reconstructed_vector = space.from_components(components)

        assert np.allclose(original_vector, reconstructed_vector)

    def test_spectral_projection_operator(self, space: Lebesgue):
        """Tests the spectral projection operator axioms and mapping."""
        modes = [(0, 0), (1, 0), (0, 1), (1, 1)]
        op_proj = space.spectral_projection_operator(modes)

        op_proj.check(n_checks=5)

        u_orig = space.project_function(lambda p: 2.0 + np.cos(p[0]))
        coeffs = op_proj(u_orig)
        assert len(coeffs) == op_proj.codomain.dim

    def test_degree_and_with_degree(self, space: Lebesgue):
        """Tests the unified degree property and the with_degree factory."""
        assert space.degree == space.kmax

        target_degree = space.kmax + 2
        new_space = space.with_degree(target_degree)

        assert new_space.degree == target_degree
        assert new_space.bounds_x == space.bounds_x
        assert new_space.bounds_y == space.bounds_y
        assert isinstance(new_space, Lebesgue)

    def test_degree_transfer_operator(self, space: Lebesgue):
        """Tests the degree transfer operator's axioms and round-trip behavior."""
        op_up = space.degree_transfer_operator(space.kmax + 2)
        op_up.check(n_checks=5)

        op_down = op_up.codomain.degree_transfer_operator(space.kmax)
        op_down.check(n_checks=5)

        # To avoid Nyquist-splitting loss during Fourier zero-padding, we must construct
        # a function that is strictly bandlimited (contains no Nyquist energy).
        # We do this by building a wave in a smaller space and projecting it up.
        small_space = space.with_degree(space.kmax - 2)
        u_small = small_space.project_function(
            lambda p: np.cos(2 * p[0]) + np.sin(p[1])
        )
        op_up_small = small_space.degree_transfer_operator(space.kmax)

        u_orig = op_up_small(u_small)

        # Now test the round trip!
        u_upsampled = op_up(u_orig)
        u_recon = op_down(u_upsampled)

        assert np.allclose(u_orig, u_recon)


def test_factory_methods():
    """Tests the automatic truncation degree factories for L2 Plane spaces."""
    space_hk = Lebesgue.from_heat_kernel_prior(
        0.5,
        ax=0.0,
        bx=10.0,
        cx=1.0,
        ay=0.0,
        by=10.0,
        cy=1.0,
        min_degree=4,
        power_of_two=True,
    )
    assert isinstance(space_hk, Lebesgue)
    assert space_hk.bounds_x == (0.0, 10.0, 1.0)
    assert space_hk.kmax >= 4
    assert (space_hk.kmax & (space_hk.kmax - 1)) == 0  # Power of two check
