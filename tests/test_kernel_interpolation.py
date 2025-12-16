"""
Unit Tests for Depth Coordinates and Kernel Interpolation

Tests for coordinate transformations and interpolation methods.
"""

import pytest
import numpy as np
from pathlib import Path

from pygeoinf.interval import (
    DepthCoordinateSystem,
    EARTH_RADIUS_KM,
    KernelInterpolator,
    compare_interpolation_methods
)


class TestDepthCoordinateSystem:
    """Test depth coordinate transformations."""

    def test_earth_radius_constant(self):
        """Test Earth radius constant."""
        assert EARTH_RADIUS_KM == 6371.0
        assert DepthCoordinateSystem.EARTH_RADIUS_KM == 6371.0

    def test_depth_to_normalized(self):
        """Test depth to normalized conversion."""
        assert DepthCoordinateSystem.depth_to_normalized(0) == 0.0
        assert DepthCoordinateSystem.depth_to_normalized(6371) == 1.0
        assert DepthCoordinateSystem.depth_to_normalized(3185.5) == 0.5

    def test_normalized_to_depth(self):
        """Test normalized to depth conversion."""
        assert DepthCoordinateSystem.normalized_to_depth(0.0) == 0.0
        assert DepthCoordinateSystem.normalized_to_depth(1.0) == 6371.0
        assert DepthCoordinateSystem.normalized_to_depth(0.5) == 3185.5

    def test_depth_normalized_round_trip(self):
        """Test round trip depth <-> normalized."""
        depths = np.array([0, 100, 670, 2891, 6371])
        normalized = DepthCoordinateSystem.depth_to_normalized(depths)
        depths_back = DepthCoordinateSystem.normalized_to_depth(normalized)
        np.testing.assert_allclose(depths, depths_back)

    def test_radius_to_depth(self):
        """Test radius to depth conversion."""
        assert DepthCoordinateSystem.radius_to_depth(6371) == 0.0
        assert DepthCoordinateSystem.radius_to_depth(0) == 6371.0
        assert DepthCoordinateSystem.radius_to_depth(3480) == 2891.0  # CMB

    def test_depth_to_radius(self):
        """Test depth to radius conversion."""
        assert DepthCoordinateSystem.depth_to_radius(0) == 6371.0
        assert DepthCoordinateSystem.depth_to_radius(6371) == 0.0
        assert DepthCoordinateSystem.depth_to_radius(2891) == 3480.0  # CMB

    def test_radius_depth_round_trip(self):
        """Test round trip radius <-> depth."""
        radii = np.array([6371, 5701, 3480, 1217.5, 0])
        depths = DepthCoordinateSystem.radius_to_depth(radii)
        radii_back = DepthCoordinateSystem.depth_to_radius(depths)
        np.testing.assert_allclose(radii, radii_back)

    def test_radius_to_normalized(self):
        """Test radius to normalized conversion."""
        assert DepthCoordinateSystem.radius_to_normalized(6371) == 0.0
        assert DepthCoordinateSystem.radius_to_normalized(0) == 1.0
        np.testing.assert_allclose(
            DepthCoordinateSystem.radius_to_normalized(3185.5),
            0.5
        )

    def test_normalized_to_radius(self):
        """Test normalized to radius conversion."""
        assert DepthCoordinateSystem.normalized_to_radius(0.0) == 6371.0
        assert DepthCoordinateSystem.normalized_to_radius(1.0) == 0.0
        assert DepthCoordinateSystem.normalized_to_radius(0.5) == 3185.5

    def test_validate_depth_valid(self):
        """Test depth validation with valid values."""
        assert DepthCoordinateSystem.validate_depth(0)
        assert DepthCoordinateSystem.validate_depth(3185.5)
        assert DepthCoordinateSystem.validate_depth(6371)
        assert DepthCoordinateSystem.validate_depth(
            np.array([0, 100, 6371])
        )

    def test_validate_depth_invalid(self):
        """Test depth validation with invalid values."""
        assert not DepthCoordinateSystem.validate_depth(-100)
        assert not DepthCoordinateSystem.validate_depth(7000)
        assert not DepthCoordinateSystem.validate_depth(
            np.array([0, 100, 7000])
        )

    def test_validate_depth_negative_allowed(self):
        """Test depth validation allowing negative values."""
        assert DepthCoordinateSystem.validate_depth(
            -100, allow_negative=True
        )
        assert not DepthCoordinateSystem.validate_depth(
            7000, allow_negative=True
        )

    def test_validate_radius_valid(self):
        """Test radius validation with valid values."""
        assert DepthCoordinateSystem.validate_radius(0)
        assert DepthCoordinateSystem.validate_radius(3185.5)
        assert DepthCoordinateSystem.validate_radius(6371)

    def test_validate_radius_invalid(self):
        """Test radius validation with invalid values."""
        assert not DepthCoordinateSystem.validate_radius(-100)
        assert not DepthCoordinateSystem.validate_radius(7000)

    def test_validate_normalized_valid(self):
        """Test normalized validation with valid values."""
        assert DepthCoordinateSystem.validate_normalized(0.0)
        assert DepthCoordinateSystem.validate_normalized(0.5)
        assert DepthCoordinateSystem.validate_normalized(1.0)
        # Allow small numerical errors
        assert DepthCoordinateSystem.validate_normalized(1.0 + 1e-15)

    def test_validate_normalized_invalid(self):
        """Test normalized validation with invalid values."""
        assert not DepthCoordinateSystem.validate_normalized(-0.1)
        assert not DepthCoordinateSystem.validate_normalized(1.2)

    def test_get_major_discontinuities(self):
        """Test getting major discontinuities."""
        disc = DepthCoordinateSystem.get_major_discontinuities()

        assert 'CMB' in disc
        assert 'Surface' in disc
        assert '660' in disc

        # Check CMB values
        cmb_depth, cmb_radius = disc['CMB']
        assert cmb_depth == 2891.0
        assert cmb_radius == 3480.0

    def test_get_major_layers(self):
        """Test getting major layers."""
        layers = DepthCoordinateSystem.get_major_layers()

        assert 'Mantle' in layers
        assert 'Core' in layers
        assert 'Crust' in layers

        # Check mantle range
        mantle_top, mantle_bottom = layers['Mantle']
        assert mantle_top < mantle_bottom

    def test_array_operations(self):
        """Test that operations work with arrays."""
        depths = np.linspace(0, 6371, 100)

        # All conversions should work with arrays
        normalized = DepthCoordinateSystem.depth_to_normalized(depths)
        assert normalized.shape == depths.shape

        radii = DepthCoordinateSystem.depth_to_radius(depths)
        assert radii.shape == depths.shape

        # Check bounds
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)


class TestKernelInterpolator:
    """Test kernel interpolation functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample kernel data for testing."""
        depths = np.array([0, 100, 500, 1000, 2000, 6371])
        values = np.array([0.1, 0.5, 0.8, 0.6, 0.2, 0.0])
        return depths, values

    def test_initialization_linear(self, sample_data):
        """Test initializing linear interpolator."""
        depths, values = sample_data
        interp = KernelInterpolator(depths, values, method='linear')

        assert interp.method == 'linear'
        assert len(interp.depths_km) == len(depths)
        assert len(interp.values) == len(values)

    def test_initialization_cubic(self, sample_data):
        """Test initializing cubic interpolator."""
        depths, values = sample_data
        interp = KernelInterpolator(depths, values, method='cubic')

        assert interp.method == 'cubic'

    def test_initialization_spline(self, sample_data):
        """Test initializing spline interpolator."""
        depths, values = sample_data
        interp = KernelInterpolator(depths, values, method='spline')

        assert interp.method == 'spline'

    def test_invalid_method(self, sample_data):
        """Test that invalid method raises error."""
        depths, values = sample_data
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            KernelInterpolator(depths, values, method='invalid')

    def test_mismatched_lengths(self):
        """Test that mismatched array lengths raise error."""
        depths = np.array([0, 100, 500])
        values = np.array([0.1, 0.5])  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            KernelInterpolator(depths, values)

    def test_too_few_points(self):
        """Test that too few points raise error."""
        depths = np.array([0])
        values = np.array([0.1])

        with pytest.raises(ValueError, match="at least 2 data points"):
            KernelInterpolator(depths, values)

    def test_invalid_depths(self, sample_data):
        """Test that invalid depths raise error."""
        depths, values = sample_data
        depths[0] = -100  # Invalid negative depth

        with pytest.raises(ValueError, match="must be in range"):
            KernelInterpolator(depths, values)

    def test_evaluation_at_data_points(self, sample_data):
        """Test evaluation at original data points."""
        depths, values = sample_data
        interp = KernelInterpolator(depths, values, method='cubic')

        # Evaluate at normalized depths
        depths_norm = DepthCoordinateSystem.depth_to_normalized(depths)
        y_interp = interp(depths_norm)

        # Should match original values closely
        np.testing.assert_allclose(y_interp, values, rtol=1e-5)

    def test_evaluation_between_points(self, sample_data):
        """Test evaluation between data points."""
        depths, values = sample_data
        interp = KernelInterpolator(depths, values, method='linear')

        # Evaluate at midpoint between first two points
        depth_mid = (depths[0] + depths[1]) / 2
        depth_mid_norm = DepthCoordinateSystem.depth_to_normalized(depth_mid)
        y_mid = interp(depth_mid_norm)

        # For linear, should be exactly midway
        expected = (values[0] + values[1]) / 2
        np.testing.assert_allclose(y_mid, expected, rtol=1e-10)

    def test_evaluate_at_depths(self, sample_data):
        """Test evaluate_at_depths convenience method."""
        depths, values = sample_data
        interp = KernelInterpolator(depths, values)

        # Evaluate using km depths
        y_depths = interp.evaluate_at_depths(depths)

        # Should match normalized evaluation
        depths_norm = DepthCoordinateSystem.depth_to_normalized(depths)
        y_norm = interp(depths_norm)

        np.testing.assert_allclose(y_depths, y_norm)

    def test_extrapolation_zero(self, sample_data):
        """Test zero extrapolation outside data range."""
        depths, values = sample_data
        interp = KernelInterpolator(
            depths, values,
            method='linear',
            extrapolate='zero'
        )

        # Evaluate beyond data range
        x_beyond = np.array([0.0, 1.05])  # Beyond [0, 1]
        y_beyond = interp(x_beyond)

        # Within range should be non-zero, beyond should be zero
        assert y_beyond[0] != 0.0  # At surface
        assert y_beyond[1] == 0.0  # Beyond center

    def test_integration(self, sample_data):
        """Test kernel integration."""
        depths, values = sample_data
        interp = KernelInterpolator(depths, values)

        # Integrate over full range
        integral = interp.integrate(0, 1)

        # Should be finite and reasonable
        assert np.isfinite(integral)
        assert integral != 0.0

    def test_peak_depth(self, sample_data):
        """Test finding peak depth."""
        depths, values = sample_data
        interp = KernelInterpolator(depths, values)

        peak_depth, peak_value = interp.get_peak_depth()

        # Peak should be at max value point (500 km with value 0.8)
        assert 400 <= peak_depth <= 600  # Allow interpolation effects
        assert 0.7 <= peak_value <= 0.9

    def test_sorting(self):
        """Test that unsorted data gets sorted."""
        # Create unsorted data
        depths = np.array([100, 0, 500, 1000])
        values = np.array([0.5, 0.1, 0.8, 0.3])

        interp = KernelInterpolator(depths, values)

        # Depths should be sorted
        assert np.all(np.diff(interp.depths_km) >= 0)

        # Values should be reordered correspondingly
        assert interp.values[0] == 0.1  # Value at depth 0


class TestInterpolationComparison:
    """Test comparison of interpolation methods."""

    def test_compare_methods(self):
        """Test comparing different interpolation methods."""
        depths = np.array([0, 100, 500, 1000, 2000, 6371])
        values = np.array([0.1, 0.5, 0.8, 0.6, 0.2, 0.0])

        interpolators, _ = compare_interpolation_methods(
            depths, values,
            methods=['linear', 'cubic']
        )

        assert 'linear' in interpolators
        assert 'cubic' in interpolators
        assert isinstance(interpolators['linear'], KernelInterpolator)
        assert isinstance(interpolators['cubic'], KernelInterpolator)

    def test_methods_differ(self):
        """Test that different methods give different results."""
        depths = np.array([0, 100, 500, 1000, 2000, 6371])
        values = np.array([0.1, 0.5, 0.8, 0.6, 0.2, 0.0])

        linear = KernelInterpolator(depths, values, method='linear')
        cubic = KernelInterpolator(depths, values, method='cubic')

        # Evaluate between data points
        x = np.linspace(0, 1, 100)
        y_linear = linear(x)
        y_cubic = cubic(x)

        # Results should differ (except at data points)
        assert not np.allclose(y_linear, y_cubic)

        # But both should pass through data points
        depths_norm = DepthCoordinateSystem.depth_to_normalized(depths)
        np.testing.assert_allclose(linear(depths_norm), values, rtol=1e-5)
        np.testing.assert_allclose(cubic(depths_norm), values, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
