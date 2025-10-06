"""
Tests for the plot module.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
from unittest.mock import Mock, patch
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.plot import plot_1d_distributions, plot_corner_distributions

# Use a non-interactive backend for testing
matplotlib.use('Agg')

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
    cov_matrix = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 2.0, 0.3],
        [0.1, 0.3, 1.5]
    ])
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
    mock = Mock()
    mock.expectation = np.array([3.0])
    
    # Mock covariance object
    mock_cov = Mock()
    mock_cov.matrix.return_value = np.array([[2.0]])
    mock.covariance = mock_cov
    
    return mock

@pytest.fixture
def mock_measure_2d():
    """Provides a 2D mock measure object for testing edge cases."""
    mock = Mock()
    mock.expectation = np.array([1.0, 2.5])
    
    # Mock covariance object
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
        """Test plotting a single posterior distribution without prior."""
        fig, ax = plot_1d_distributions(
            posterior_measures=gaussian_measure_1d,
            show_plot=False
        )
        
        assert fig is not None
        assert ax is not None
        assert len(ax.lines) == 1  # One line for posterior
        assert len(ax.collections) == 1  # One fill_between for posterior
        
        plt.close(fig)

    def test_single_posterior_with_prior(self, gaussian_measure_1d, mock_measure_1d):
        """Test plotting posterior with prior distribution."""
        fig, (ax1, ax2) = plot_1d_distributions(
            posterior_measures=gaussian_measure_1d,
            prior_measures=mock_measure_1d,
            show_plot=False
        )
        
        assert fig is not None
        assert ax1 is not None
        assert ax2 is not None
        assert len(ax1.lines) == 1  # Prior line
        assert len(ax2.lines) == 1  # Posterior line
        
        plt.close(fig)

    def test_multiple_posteriors(self, gaussian_measure_1d, mock_measure_1d):
        """Test plotting multiple posterior distributions."""
        fig, ax = plot_1d_distributions(
            posterior_measures=[gaussian_measure_1d, mock_measure_1d],
            show_plot=False
        )
        
        assert fig is not None
        assert ax is not None
        assert len(ax.lines) == 2  # Two posterior lines
        assert len(ax.collections) == 2  # Two fill_between for posteriors
        
        plt.close(fig)

    def test_multiple_priors_and_posteriors(self, gaussian_measure_1d, mock_measure_1d):
        """Test plotting multiple priors and posteriors."""
        fig, (ax1, ax2) = plot_1d_distributions(
            posterior_measures=[gaussian_measure_1d, mock_measure_1d],
            prior_measures=[mock_measure_1d, gaussian_measure_1d],
            show_plot=False
        )
        
        assert fig is not None
        assert ax1 is not None
        assert ax2 is not None
        assert len(ax1.lines) == 2  # Two prior lines
        assert len(ax2.lines) == 2  # Two posterior lines
        
        plt.close(fig)

    def test_with_true_value(self, gaussian_measure_1d):
        """Test plotting with a true value line."""
        true_val = 2.5
        fig, ax = plot_1d_distributions(
            posterior_measures=gaussian_measure_1d,
            true_value=true_val,
            show_plot=False
        )
        
        assert fig is not None
        assert ax is not None
        
        # Check for vertical line indicating true value
        vertical_lines = [line for line in ax.lines if hasattr(line, '_x') and 
                         np.all(line._x == true_val)]
        assert len(vertical_lines) >= 1
        
        plt.close(fig)

    def test_scipy_distribution(self, scipy_multivariate_normal_1d):
        """Test plotting with scipy distribution objects."""
        fig, ax = plot_1d_distributions(
            posterior_measures=scipy_multivariate_normal_1d,
            show_plot=False
        )
        
        assert fig is not None
        assert ax is not None
        assert len(ax.lines) == 1
        
        plt.close(fig)

    def test_custom_parameters(self, gaussian_measure_1d):
        """Test plotting with custom parameters."""
        fig, ax = plot_1d_distributions(
            posterior_measures=gaussian_measure_1d,
            xlabel="Custom X Label",
            title="Custom Title",
            figsize=(10, 6),
            show_plot=False
        )
        
        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel() == "Custom X Label"
        assert fig._suptitle.get_text() == "Custom Title"
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 6
        
        plt.close(fig)

    def test_empty_list_raises_error(self):
        """Test that empty posterior list raises appropriate error."""
        with pytest.raises(ValueError):
            plot_1d_distributions(
                posterior_measures=[],
                show_plot=False
            )

    @patch('matplotlib.pyplot.show')
    def test_show_plot_true(self, mock_show, gaussian_measure_1d):
        """Test that plt.show is called when show_plot=True."""
        plot_1d_distributions(
            posterior_measures=gaussian_measure_1d,
            show_plot=True
        )
        mock_show.assert_called_once()


# =============================================================================
# Test Suite for plot_corner_distributions
# =============================================================================

class TestPlotCornerDistributions:
    """Test suite for the plot_corner_distributions function."""

    def test_1d_corner_plot(self, gaussian_measure_1d):
        """Test corner plot for 1D distribution."""
        fig, axes = plot_corner_distributions(
            posterior_measure=gaussian_measure_1d,
            show_plot=False
        )
        
        assert fig is not None
        assert axes is not None
        assert axes.shape == (1, 1)
        
        plt.close(fig)

    def test_2d_corner_plot(self, gaussian_measure_2d):
        """Test corner plot for 2D distribution."""
        fig, axes = plot_corner_distributions(
            posterior_measure=gaussian_measure_2d,
            show_plot=False
        )
        
        assert fig is not None
        assert axes is not None
        assert axes.shape == (2, 2)
        
        # Check diagonal plots (marginal distributions)
        assert len(axes[0, 0].lines) >= 1  # First marginal
        assert len(axes[1, 1].lines) >= 1  # Second marginal
        
        # Check off-diagonal plot (joint distribution)
        assert hasattr(axes[1, 0], 'collections')  # Should have pcolormesh
        
        # Check upper triangle has axis turned off
        assert not axes[0, 1].axison
        
        plt.close(fig)

    def test_3d_corner_plot(self, gaussian_measure_3d):
        """Test corner plot for 3D distribution."""
        fig, axes = plot_corner_distributions(
            posterior_measure=gaussian_measure_3d,
            show_plot=False
        )
        
        assert fig is not None
        assert axes is not None
        assert axes.shape == (3, 3)
        
        # Check diagonal plots exist
        for i in range(3):
            assert len(axes[i, i].lines) >= 1
        
        # Check upper triangle plots have axis turned off
        for i in range(3):
            for j in range(i + 1, 3):
                assert not axes[i, j].axison
        
        plt.close(fig)

    def test_with_true_values(self, gaussian_measure_2d):
        """Test corner plot with true values."""
        true_vals = [1.2, 2.3]
        fig, axes = plot_corner_distributions(
            posterior_measure=gaussian_measure_2d,
            true_values=true_vals,
            show_plot=False
        )
        
        assert fig is not None
        assert axes is not None
        
        # Check that true value markers are present in diagonal plots
        for i in range(2):
            lines = axes[i, i].lines
            # Should have at least posterior PDF and true value line
            assert len(lines) >= 2
        
        plt.close(fig)

    def test_with_custom_labels(self, gaussian_measure_2d):
        """Test corner plot with custom dimension labels."""
        labels = ["Parameter A", "Parameter B"]
        fig, axes = plot_corner_distributions(
            posterior_measure=gaussian_measure_2d,
            labels=labels,
            show_plot=False
        )
        
        assert fig is not None
        assert axes is not None
        assert axes[1, 0].get_xlabel() == "Parameter A"
        assert axes[1, 0].get_ylabel() == "Parameter B"
        
        plt.close(fig)

    def test_custom_parameters(self, gaussian_measure_2d):
        """Test corner plot with custom parameters."""
        fig, axes = plot_corner_distributions(
            posterior_measure=gaussian_measure_2d,
            title="Custom Corner Plot",
            figsize=(8, 8),
            colormap="Reds",
            include_sigma_contours=False,
            show_plot=False
        )
        
        assert fig is not None
        assert axes is not None
        assert fig._suptitle.get_text() == "Custom Corner Plot"
        assert fig.get_size_inches()[0] == 8
        assert fig.get_size_inches()[1] == 8
        
        plt.close(fig)

    def test_invalid_measure_raises_error(self):
        """Test that invalid measure object raises appropriate error."""
        invalid_measure = Mock()
        # Missing required attributes
        
        with pytest.raises((ValueError, TypeError)):
            plot_corner_distributions(
                posterior_measure=invalid_measure,
                show_plot=False
            )

    @patch('matplotlib.pyplot.show')
    def test_show_plot_true(self, mock_show, gaussian_measure_2d):
        """Test that plt.show is called when show_plot=True."""
        plot_corner_distributions(
            posterior_measure=gaussian_measure_2d,
            show_plot=True
        )
        mock_show.assert_called_once()


# =============================================================================
# Integration Tests
# =============================================================================

class TestPlotIntegration:
    """Integration tests for plotting functions with real pygeoinf objects."""

    def test_plotting_workflow_1d(self, euclidean_space_1d):
        """Test complete 1D plotting workflow with pygeoinf objects."""
        # Create prior and posterior measures
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
        
        # Test 1D plotting
        fig, (ax1, ax2) = plot_1d_distributions(
            posterior_measures=posterior_measure,
            prior_measures=prior_measure,
            true_value=0.8,
            show_plot=False
        )
        
        assert fig is not None
        assert ax1 is not None
        assert ax2 is not None
        
        plt.close(fig)

    def test_plotting_workflow_2d(self, euclidean_space_2d):
        """Test complete 2D plotting workflow with pygeoinf objects."""
        # Create a 2D posterior measure
        mean = np.array([1.0, 2.0])
        cov_matrix = np.array([[1.0, 0.3], [0.3, 2.0]])
        posterior_measure = GaussianMeasure.from_covariance_matrix(
            euclidean_space_2d, cov_matrix, expectation=mean
        )
        
        # Test corner plotting
        fig, axes = plot_corner_distributions(
            posterior_measure=posterior_measure,
            true_values=[0.9, 2.1],
            labels=["X coordinate", "Y coordinate"],
            show_plot=False
        )
        
        assert fig is not None
        assert axes is not None
        assert axes.shape == (2, 2)
        
        plt.close(fig)

    def test_mixed_measure_types(self, gaussian_measure_1d, scipy_multivariate_normal_1d):
        """Test plotting with mixed measure types."""
        fig, ax = plot_1d_distributions(
            posterior_measures=[gaussian_measure_1d, scipy_multivariate_normal_1d],
            show_plot=False
        )
        
        assert fig is not None
        assert ax is not None
        assert len(ax.lines) == 2  # Two different measure types
        
        plt.close(fig)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestPlotEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_variance(self, euclidean_space_1d):
        """Test plotting with very small variance."""
        mean = np.array([1.0])
        cov_matrix = np.array([[1e-8]])  # Very small variance
        measure = GaussianMeasure.from_covariance_matrix(
            euclidean_space_1d, cov_matrix, expectation=mean
        )
        
        fig, ax = plot_1d_distributions(
            posterior_measures=measure,
            show_plot=False
        )
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)

    def test_large_variance(self, euclidean_space_1d):
        """Test plotting with large variance."""
        mean = np.array([1.0])
        cov_matrix = np.array([[100.0]])  # Large variance
        measure = GaussianMeasure.from_covariance_matrix(
            euclidean_space_1d, cov_matrix, expectation=mean
        )
        
        fig, ax = plot_1d_distributions(
            posterior_measures=measure,
            show_plot=False
        )
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)

    def test_extreme_true_value(self, gaussian_measure_1d):
        """Test plotting with true value far from distribution mean."""
        # True value very far from the distribution
        true_val = 100.0  # Distribution mean is around 2.0
        
        fig, ax = plot_1d_distributions(
            posterior_measures=gaussian_measure_1d,
            true_value=true_val,
            show_plot=False
        )
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)

    def test_single_dimension_corner_plot_edge_case(self, gaussian_measure_1d):
        """Test corner plot with single dimension (edge case)."""
        fig, axes = plot_corner_distributions(
            posterior_measure=gaussian_measure_1d,
            show_plot=False
        )
        
        assert fig is not None
        assert axes is not None
        # Should handle 1D case gracefully
        
        plt.close(fig)