"""
Tests for the plot module.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
from unittest.mock import Mock
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.plot import plot_1d_distributions, plot_corner_distributions

# Use a non-interactive backend for testing to prevent GUI popups and CI crashes
matplotlib.use("Agg")

# =============================================================================
# Parametrised Fixtures
# =============================================================================


@pytest.fixture
def euclidean_space_1d():
    """Provides a 1D Euclidean space for testing."""
    return EuclideanSpace(dim=1)


@pytest.fixture
def euclidean_space_2d():
    """Provides a 2D Euclidean space for testing."""
    return EuclideanSpace(dim=2)


@pytest.fixture
def euclidean_space_3d():
    """Provides a 3D Euclidean space for testing."""
    return EuclideanSpace(dim=3)


@pytest.fixture
def gaussian_measure_1d(euclidean_space_1d):
    """Provides a 1D Gaussian measure for testing."""
    mean = np.array([2.0])
    cov_matrix = np.array([[1.0]])
    return GaussianMeasure.from_covariance_matrix(
        euclidean_space_1d, cov_matrix, expectation=mean
    )


@pytest.fixture
def gaussian_measure_2d(euclidean_space_2d):
    """Provides a 2D Gaussian measure for testing."""
    mean = np.array([1.0, 2.0])
    cov_matrix = np.array([[1.0, 0.3], [0.3, 2.0]])
    return GaussianMeasure.from_covariance_matrix(
        euclidean_space_2d, cov_matrix, expectation=mean
    )


@pytest.fixture
def gaussian_measure_3d(euclidean_space_3d):
    """Provides a 3D Gaussian measure for testing."""
    mean = np.array([1.0, 2.0, 3.0])
    cov_matrix = np.array([[1.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 1.5]])
    return GaussianMeasure.from_covariance_matrix(
        euclidean_space_3d, cov_matrix, expectation=mean
    )


@pytest.fixture
def scipy_multivariate_normal_1d():
    """Provides a 1D scipy multivariate normal distribution for testing."""
    return stats.multivariate_normal(mean=[1.5], cov=[[0.8]])


@pytest.fixture
def scipy_multivariate_normal_2d():
    """Provides a 2D scipy multivariate normal distribution for testing."""
    return stats.multivariate_normal(mean=[0.5, 1.5], cov=[[1.2, 0.2], [0.2, 1.8]])


@pytest.fixture
def mock_measure_1d():
    """Provides a mock measure object for testing edge cases."""
    mock = Mock(spec=GaussianMeasure)
    mock.expectation = np.array([3.0])

    mock_cov = Mock()
    mock_cov.matrix.return_value = np.array([[2.0]])
    mock.covariance = mock_cov

    return mock


@pytest.fixture
def mock_measure_2d():
    """Provides a 2D mock measure object for testing edge cases."""
    mock = Mock(spec=GaussianMeasure)
    mock.expectation = np.array([1.0, 2.5])

    mock_cov = Mock()
    mock_cov.matrix.return_value = np.array([[1.5, 0.4], [0.4, 2.2]])
    mock.covariance = mock_cov

    return mock


# =============================================================================
# Test Suite for plot_1d_distributions
# =============================================================================


class TestPlot1DDistributions:
    """Test suite for the plot_1d_distributions function."""

    def test_single_posterior_only(self, gaussian_measure_1d):
        ax = plot_1d_distributions(gaussian_measure_1d)
        fig = ax.get_figure()

        assert fig is not None
        assert ax is not None
        assert len(ax.lines) == 1
        assert len(ax.collections) == 0  # Default is fill_density=False

        plt.close(fig)

    def test_single_posterior_with_prior(self, gaussian_measure_1d, mock_measure_1d):
        ax1, ax2 = plot_1d_distributions(
            gaussian_measure_1d,
            prior_measures=mock_measure_1d,
        )
        fig = ax1.get_figure()

        assert fig is not None
        assert ax1 is not None
        assert ax2 is not None
        assert len(ax1.lines) == 1
        assert len(ax2.lines) == 1

        plt.close(fig)

    def test_multiple_posteriors(self, gaussian_measure_1d, mock_measure_1d):
        ax = plot_1d_distributions([gaussian_measure_1d, mock_measure_1d])
        fig = ax.get_figure()

        assert fig is not None
        assert ax is not None
        assert len(ax.lines) == 2

        plt.close(fig)

    def test_multiple_priors_and_posteriors(self, gaussian_measure_1d, mock_measure_1d):
        ax1, ax2 = plot_1d_distributions(
            [gaussian_measure_1d, mock_measure_1d],
            prior_measures=[mock_measure_1d, gaussian_measure_1d],
        )
        fig = ax1.get_figure()

        assert fig is not None
        assert ax1 is not None
        assert ax2 is not None
        assert len(ax1.lines) == 2
        assert len(ax2.lines) == 2

        plt.close(fig)

    def test_with_true_value(self, gaussian_measure_1d):
        true_val = 2.5
        ax = plot_1d_distributions(gaussian_measure_1d, true_value=true_val)
        fig = ax.get_figure()

        assert fig is not None
        assert ax is not None

        vertical_lines = [
            line
            for line in ax.lines
            if hasattr(line, "_x") and np.all(line._x == true_val)
        ]
        assert len(vertical_lines) >= 1

        plt.close(fig)

    def test_scipy_distribution(self, scipy_multivariate_normal_1d):
        ax = plot_1d_distributions(scipy_multivariate_normal_1d)
        fig = ax.get_figure()

        assert fig is not None
        assert ax is not None
        assert len(ax.lines) == 1

        plt.close(fig)

    def test_custom_parameters(self, gaussian_measure_1d):
        fig, ax_in = plt.subplots(figsize=(10, 6))

        ax_out = plot_1d_distributions(
            gaussian_measure_1d,
            ax=ax_in,
            xlabel="Custom X Label",
            title="Custom Title",
        )

        assert ax_out is ax_in
        assert ax_out.get_xlabel() == "Custom X Label"
        assert ax_out.get_title() == "Custom Title"
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 6

        plt.close(fig)

    def test_fill_density_parameter(self, gaussian_measure_1d):
        """Test that setting fill_density=True adds a PolyCollection to the axes."""
        # Test with fill_density=False (default)
        ax_unfilled = plot_1d_distributions(gaussian_measure_1d, fill_density=False)
        unfilled_collections = len(ax_unfilled.collections)
        plt.close(ax_unfilled.get_figure())

        # Test with fill_density=True
        ax_filled = plot_1d_distributions(gaussian_measure_1d, fill_density=True)
        filled_collections = len(ax_filled.collections)

        assert filled_collections > unfilled_collections

        plt.close(ax_filled.get_figure())

    def test_empty_list_raises_error(self):
        with pytest.raises(ValueError):
            plot_1d_distributions([])


# =============================================================================
# Test Suite for plot_corner_distributions
# =============================================================================


class TestPlotCornerDistributions:
    """Test suite for the plot_corner_distributions function."""

    def test_1d_corner_plot(self, gaussian_measure_1d):
        axes = plot_corner_distributions(gaussian_measure_1d)
        fig = axes[0, 0].get_figure()

        assert fig is not None
        assert axes is not None
        assert axes.shape == (1, 1)

        plt.close(fig)

    def test_2d_corner_plot(self, gaussian_measure_2d):
        axes = plot_corner_distributions(gaussian_measure_2d)
        fig = axes[0, 0].get_figure()

        assert fig is not None
        assert axes is not None
        assert axes.shape == (2, 2)

        assert len(axes[0, 0].lines) >= 1
        assert len(axes[1, 1].lines) >= 1
        assert hasattr(axes[1, 0], "collections")

        # Accommodate set_visible(False) or axis('off')
        assert not axes[0, 1].get_visible() or not axes[0, 1].axison

        plt.close(fig)

    def test_3d_corner_plot(self, gaussian_measure_3d):
        axes = plot_corner_distributions(gaussian_measure_3d)
        fig = axes[0, 0].get_figure()

        assert fig is not None
        assert axes is not None
        assert axes.shape == (3, 3)

        for i in range(3):
            assert len(axes[i, i].lines) >= 1

        for i in range(3):
            for j in range(i + 1, 3):
                assert not axes[i, j].get_visible() or not axes[i, j].axison

        plt.close(fig)

    def test_with_true_values(self, gaussian_measure_2d):
        true_vals = [1.2, 2.3]
        axes = plot_corner_distributions(
            gaussian_measure_2d,
            true_values=true_vals,
        )
        fig = axes[0, 0].get_figure()

        assert fig is not None
        assert axes is not None

        for i in range(2):
            lines = axes[i, i].lines
            assert len(lines) >= 2

        plt.close(fig)

    def test_with_prior_measure(self, gaussian_measure_2d):
        """Test corner plot with a prior measure injected."""
        axes = plot_corner_distributions(
            gaussian_measure_2d, prior_measure=gaussian_measure_2d
        )
        fig = axes[0, 0].get_figure()

        assert fig is not None
        assert axes is not None

        # Verify the secondary axis exists on the diagonal plots
        for i in range(2):
            ax = axes[i, i]
            assert len(ax.child_axes) > 0
            sec_ax = ax.child_axes[0]
            assert "Prior Mean" in sec_ax.get_xlabel()

        plt.close(fig)

    def test_invalid_prior_measure_raises_error(self, gaussian_measure_2d):
        """Test that invalid prior measure object raises appropriate error."""
        invalid_prior = Mock()
        with pytest.raises(TypeError, match="prior_measure must be a GaussianMeasure"):
            plot_corner_distributions(gaussian_measure_2d, prior_measure=invalid_prior)

    def test_with_custom_labels(self, gaussian_measure_2d):
        labels = ["Parameter A", "Parameter B"]
        axes = plot_corner_distributions(gaussian_measure_2d, labels=labels)
        fig = axes[0, 0].get_figure()

        assert fig is not None
        assert axes is not None
        assert axes[1, 0].get_xlabel() == "Parameter A"
        assert axes[1, 0].get_ylabel() == "Parameter B"

        plt.close(fig)

    def test_custom_parameters(self, gaussian_measure_2d):
        axes = plot_corner_distributions(
            gaussian_measure_2d,
            title="Custom Corner Plot",
            figsize=(8, 8),
            colormap="Reds",
            contour_color="darkred",
            fill_density=True,
            num_sigmas=2,
        )
        fig = axes[0, 0].get_figure()

        assert fig is not None
        assert axes is not None
        assert fig._suptitle.get_text() == "Custom Corner Plot"
        assert fig.get_size_inches()[0] == 8
        assert fig.get_size_inches()[1] == 8

        # Verify that fill_density=True generated collections (pcolormesh/fill_between)
        assert len(axes[1, 0].collections) > 0

        plt.close(fig)

    def test_invalid_measure_raises_error(self):
        invalid_measure = Mock()

        with pytest.raises((ValueError, TypeError)):
            plot_corner_distributions(posterior_measure=invalid_measure)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPlotIntegration:
    """Integration tests for plotting functions with real pygeoinf objects."""

    def test_plotting_workflow_1d(self, euclidean_space_1d):
        prior_mean = np.array([0.0])
        prior_cov = np.array([[2.0]])
        prior_measure = GaussianMeasure.from_covariance_matrix(
            euclidean_space_1d, prior_cov, expectation=prior_mean
        )

        posterior_mean = np.array([1.0])
        posterior_cov = np.array([[0.5]])
        posterior_measure = GaussianMeasure.from_covariance_matrix(
            euclidean_space_1d, posterior_cov, expectation=posterior_mean
        )

        ax1, ax2 = plot_1d_distributions(
            posterior_measure,
            prior_measures=prior_measure,
            true_value=0.8,
        )
        fig = ax1.get_figure()

        assert fig is not None
        assert ax1 is not None
        assert ax2 is not None

        plt.close(fig)

    def test_plotting_workflow_2d(self, euclidean_space_2d):
        mean = np.array([1.0, 2.0])
        cov_matrix = np.array([[1.0, 0.3], [0.3, 2.0]])
        posterior_measure = GaussianMeasure.from_covariance_matrix(
            euclidean_space_2d, cov_matrix, expectation=mean
        )

        axes = plot_corner_distributions(
            posterior_measure,
            true_values=[0.9, 2.1],
            labels=["X coordinate", "Y coordinate"],
        )
        fig = axes[0, 0].get_figure()

        assert fig is not None
        assert axes is not None
        assert axes.shape == (2, 2)

        plt.close(fig)

    def test_mixed_measure_types(
        self, gaussian_measure_1d, scipy_multivariate_normal_1d
    ):
        # Create a fresh figure and axis to prevent global state bleed
        fig, ax_in = plt.subplots()

        ax = plot_1d_distributions(
            [gaussian_measure_1d, scipy_multivariate_normal_1d],
            ax=ax_in,  # Explicitly pass the fresh axis
        )

        assert fig is not None
        assert ax is not None
        assert len(ax.lines) == 2

        plt.close(fig)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestPlotEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_variance(self, euclidean_space_1d):
        mean = np.array([1.0])
        cov_matrix = np.array([[1e-8]])
        measure = GaussianMeasure.from_covariance_matrix(
            euclidean_space_1d, cov_matrix, expectation=mean
        )

        ax = plot_1d_distributions(measure)
        fig = ax.get_figure()

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_large_variance(self, euclidean_space_1d):
        mean = np.array([1.0])
        cov_matrix = np.array([[100.0]])
        measure = GaussianMeasure.from_covariance_matrix(
            euclidean_space_1d, cov_matrix, expectation=mean
        )

        ax = plot_1d_distributions(measure)
        fig = ax.get_figure()

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_extreme_true_value(self, gaussian_measure_1d):
        true_val = 100.0

        ax = plot_1d_distributions(gaussian_measure_1d, true_value=true_val)
        fig = ax.get_figure()

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_single_dimension_corner_plot_edge_case(self, gaussian_measure_1d):
        axes = plot_corner_distributions(gaussian_measure_1d)
        fig = axes[0, 0].get_figure()

        assert fig is not None
        assert axes is not None

        plt.close(fig)

    def test_extreme_true_value_corner(self, gaussian_measure_2d):
        """Test that the dynamic contour and span logic handles far-away true values."""
        # True values far outside the default 3.75 standard deviations
        extreme_vals = [15.0, -10.0]

        axes = plot_corner_distributions(gaussian_measure_2d, true_values=extreme_vals)
        fig = axes[0, 0].get_figure()

        assert fig is not None
        assert axes is not None

        # Ensure the axes bounds dynamically expanded to include the extreme values
        x_lim = axes[1, 0].get_xlim()
        y_lim = axes[1, 0].get_ylim()

        assert x_lim[0] <= extreme_vals[0] <= x_lim[1]
        assert y_lim[0] <= extreme_vals[1] <= y_lim[1]

        plt.close(fig)
