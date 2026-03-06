"""
Tests for the plot module.
"""

import pytest
import numpy as np
import matplotlib
# Set non-interactive backend before importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats
from unittest.mock import Mock, patch
import warnings
import pygeoinf
import pygeoinf.plot
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.plot import plot_1d_distributions, plot_corner_distributions, SubspaceSlicePlotter, plot_slice
from pygeoinf.subspaces import AffineSubspace
from pygeoinf.subsets import Ball, HalfSpace, PolyhedralSet

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
        fig, ax = plot_1d_distributions(gaussian_measure_1d, show_plot=False)

        assert fig is not None
        assert ax is not None
        assert len(ax.lines) == 1  # One line for posterior
        assert len(ax.collections) == 1  # One fill_between for posterior

        plt.close(fig)

    def test_single_posterior_with_prior(self, gaussian_measure_1d, mock_measure_1d):
        """Test plotting posterior with prior distribution."""
        fig, (ax1, ax2) = plot_1d_distributions(
            gaussian_measure_1d,
            prior_measures=mock_measure_1d,
            show_plot=False,
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
            [gaussian_measure_1d, mock_measure_1d], show_plot=False
        )

        assert fig is not None
        assert ax is not None
        assert len(ax.lines) == 2  # Two posterior lines
        assert len(ax.collections) == 2  # Two fill_between for posteriors

        plt.close(fig)

    def test_multiple_priors_and_posteriors(self, gaussian_measure_1d, mock_measure_1d):
        """Test plotting multiple priors and posteriors."""
        fig, (ax1, ax2) = plot_1d_distributions(
            [gaussian_measure_1d, mock_measure_1d],
            prior_measures=[mock_measure_1d, gaussian_measure_1d],
            show_plot=False,
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
            gaussian_measure_1d, true_value=true_val, show_plot=False
        )

        assert fig is not None
        assert ax is not None

        # Check for vertical line indicating true value
        vertical_lines = [
            line
            for line in ax.lines
            if hasattr(line, "_x") and np.all(line._x == true_val)
        ]
        assert len(vertical_lines) >= 1

        plt.close(fig)

    def test_scipy_distribution(self, scipy_multivariate_normal_1d):
        """Test plotting with scipy distribution objects."""
        fig, ax = plot_1d_distributions(scipy_multivariate_normal_1d, show_plot=False)

        assert fig is not None
        assert ax is not None
        assert len(ax.lines) == 1

        plt.close(fig)

    def test_custom_parameters(self, gaussian_measure_1d):
        """Test plotting with custom parameters."""
        fig, ax = plot_1d_distributions(
            gaussian_measure_1d,
            xlabel="Custom X Label",
            title="Custom Title",
            figsize=(10, 6),
            show_plot=False,
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
            plot_1d_distributions([], show_plot=False)

    @patch("matplotlib.pyplot.show")
    def test_show_plot_true(self, mock_show, gaussian_measure_1d):
        """Test that plt.show is called when show_plot=True."""
        plot_1d_distributions(gaussian_measure_1d, show_plot=True)
        mock_show.assert_called_once()


# =============================================================================
# Test Suite for plot_corner_distributions
# =============================================================================


class TestPlotCornerDistributions:
    """Test suite for the plot_corner_distributions function."""

    def test_1d_corner_plot(self, gaussian_measure_1d):
        """Test corner plot for 1D distribution."""
        fig, axes = plot_corner_distributions(gaussian_measure_1d, show_plot=False)

        assert fig is not None
        assert axes is not None
        assert axes.shape == (1, 1)

        plt.close(fig)

    def test_2d_corner_plot(self, gaussian_measure_2d):
        """Test corner plot for 2D distribution."""
        fig, axes = plot_corner_distributions(gaussian_measure_2d, show_plot=False)

        assert fig is not None
        assert axes is not None
        assert axes.shape == (2, 2)

        # Check diagonal plots (marginal distributions)
        assert len(axes[0, 0].lines) >= 1  # First marginal
        assert len(axes[1, 1].lines) >= 1  # Second marginal

        # Check off-diagonal plot (joint distribution)
        assert hasattr(axes[1, 0], "collections")  # Should have pcolormesh

        # Check upper triangle has axis turned off
        assert not axes[0, 1].axison

        plt.close(fig)

    def test_3d_corner_plot(self, gaussian_measure_3d):
        """Test corner plot for 3D distribution."""
        fig, axes = plot_corner_distributions(gaussian_measure_3d, show_plot=False)

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
            gaussian_measure_2d,
            true_values=true_vals,
            show_plot=False,
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
            gaussian_measure_2d, labels=labels, show_plot=False
        )

        assert fig is not None
        assert axes is not None
        assert axes[1, 0].get_xlabel() == "Parameter A"
        assert axes[1, 0].get_ylabel() == "Parameter B"

        plt.close(fig)

    def test_custom_parameters(self, gaussian_measure_2d):
        """Test corner plot with custom parameters."""
        fig, axes = plot_corner_distributions(
            gaussian_measure_2d,
            title="Custom Corner Plot",
            figsize=(8, 8),
            colormap="Reds",
            include_sigma_contours=False,
            show_plot=False,
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
                posterior_measure=invalid_measure, show_plot=False
            )

    @patch("matplotlib.pyplot.show")
    def test_show_plot_true(self, mock_show, gaussian_measure_2d):
        """Test that plt.show is called when show_plot=True."""
        plot_corner_distributions(gaussian_measure_2d, show_plot=True)
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
            posterior_measure,
            prior_measures=prior_measure,
            true_value=0.8,
            show_plot=False,
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
            posterior_measure,
            true_values=[0.9, 2.1],
            labels=["X coordinate", "Y coordinate"],
            show_plot=False,
        )

        assert fig is not None
        assert axes is not None
        assert axes.shape == (2, 2)

        plt.close(fig)

    def test_mixed_measure_types(
        self, gaussian_measure_1d, scipy_multivariate_normal_1d
    ):
        """Test plotting with mixed measure types."""
        fig, ax = plot_1d_distributions(
            [gaussian_measure_1d, scipy_multivariate_normal_1d],
            show_plot=False,
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

        fig, ax = plot_1d_distributions(measure, show_plot=False)

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

        fig, ax = plot_1d_distributions(measure, show_plot=False)

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_extreme_true_value(self, gaussian_measure_1d):
        """Test plotting with true value far from distribution mean."""
        # True value very far from the distribution
        true_val = 100.0  # Distribution mean is around 2.0

        fig, ax = plot_1d_distributions(
            gaussian_measure_1d, true_value=true_val, show_plot=False
        )

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_single_dimension_corner_plot_edge_case(self, gaussian_measure_1d):
        """Test corner plot with single dimension (edge case)."""
        fig, axes = plot_corner_distributions(gaussian_measure_1d, show_plot=False)

        assert fig is not None
        assert axes is not None
        # Should handle 1D case gracefully

        plt.close(fig)


# =============================================================================
# Phase 1 Baseline: Import & SubspaceSlicePlotter Tests
# =============================================================================


class TestVisualizationModuleImports:
    """Phase 1 baseline: verify pygeoinf.plot is importable and key symbols exist."""

    def test_visualization_module_imports(self):
        """pygeoinf.plot must be importable with no errors."""
        import importlib
        mod = importlib.import_module("pygeoinf.plot")
        assert mod is not None
        assert hasattr(mod, "plot_1d_distributions")
        assert hasattr(mod, "plot_corner_distributions")
        assert hasattr(mod, "SubspaceSlicePlotter")

    def test_subspace_slice_plotter_imports(self):
        """SubspaceSlicePlotter must be importable from pygeoinf.plot and pygeoinf."""
        # Direct import already tested via module-level import; verify the class is callable
        assert callable(SubspaceSlicePlotter)
        # Also verify exposed via the top-level package
        assert hasattr(pygeoinf, "SubspaceSlicePlotter")
        assert pygeoinf.SubspaceSlicePlotter is SubspaceSlicePlotter


class TestSubspaceSlicePlotter2D:
    """Phase 1 baseline: 2D slice plotting for Ball and PolyhedralSet."""

    @pytest.fixture
    def domain_2d(self):
        return EuclideanSpace(dim=2)

    @pytest.fixture
    def subspace_2d(self, domain_2d):
        """2D affine subspace spanning all of R^2 (identity slice)."""
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return AffineSubspace.from_tangent_basis(domain_2d, [e1, e2])

    def test_subspace_slice_plotter_ball_2d(self, domain_2d, subspace_2d):
        """SubspaceSlicePlotter renders a 2D ball slice without error."""
        center = np.zeros(2)
        ball = Ball(domain_2d, center, radius=0.5, open_set=False)

        plotter = SubspaceSlicePlotter(ball, subspace_2d, grid_size=10)
        fig, ax, payload = plotter.plot(
            bounds=(-1.0, 1.0, -1.0, 1.0), show_plot=False
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        # payload is the boolean membership mask for oracle-based (Ball) path
        assert isinstance(payload, np.ndarray)
        assert payload.shape == (10, 10)
        assert payload.dtype == bool
        # The grid point closest to the origin must lie inside the ball (radius=0.5)
        grid_coords = np.linspace(-1.0, 1.0, 10)
        i0 = int(np.argmin(np.abs(grid_coords)))
        assert payload[i0, i0]

        plt.close(fig)

    def test_subspace_slice_plotter_polyhedral_2d(self, domain_2d, subspace_2d):
        """SubspaceSlicePlotter renders a 2D polyhedral (box) slice without error."""
        # Box: -0.5 <= x1 <= 0.5, -0.5 <= x2 <= 0.5  (centred at origin, side length 1)
        a1 = np.array([1.0, 0.0])
        a2 = np.array([0.0, 1.0])
        half_spaces = [
            HalfSpace(domain_2d, a1, 0.5, "<="),
            HalfSpace(domain_2d, -a1, 0.5, "<="),
            HalfSpace(domain_2d, a2, 0.5, "<="),
            HalfSpace(domain_2d, -a2, 0.5, "<="),
        ]
        poly = PolyhedralSet(domain_2d, half_spaces)

        plotter = SubspaceSlicePlotter(poly, subspace_2d, grid_size=10)
        fig, ax, payload = plotter.plot(
            bounds=(-1.0, 1.0, -1.0, 1.0), show_plot=False
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        # payload for PolyhedralSet is the polygon vertex array
        assert isinstance(payload, np.ndarray)
        # Polyhedral path: payload is vertices array with shape (n_vertices, 2)
        assert payload.ndim == 2
        assert payload.shape[1] == 2

        plt.close(fig)


# =============================================================================
# Phase 2: plot_slice() convenience wrapper
# =============================================================================


class TestPlotSliceWrapper:
    """Phase 2: tests for the top-level plot_slice() convenience wrapper."""

    def test_plot_slice_exported(self):
        """plot_slice must be importable from pygeoinf.plot and pygeoinf."""
        import pygeoinf
        import pygeoinf.plot

        assert hasattr(pygeoinf.plot, "plot_slice")
        assert callable(pygeoinf.plot.plot_slice)
        assert hasattr(pygeoinf, "plot_slice")
        assert pygeoinf.plot_slice is pygeoinf.plot.plot_slice

    def test_plot_slice_ball_2d_returns_figure(self):
        """plot_slice on a 2D Ball returns (fig, ax, payload) matching SubspaceSlicePlotter.plot."""
        domain = EuclideanSpace(dim=2)
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            subspace = AffineSubspace.from_tangent_basis(domain, [e1, e2])

        ball = Ball(domain, np.zeros(2), radius=0.5, open_set=False)

        fig, ax, payload = plot_slice(
            ball, subspace, bounds=(-1.0, 1.0, -1.0, 1.0), grid_size=10, show_plot=False
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        assert isinstance(payload, np.ndarray)
        assert payload.shape == (10, 10)
        assert payload.dtype == bool

        plt.close(fig)

    def test_plot_slice_polyhedral_2d_returns_vertices(self):
        """plot_slice on a 2D PolyhedralSet returns vertex array (n_vertices, 2)."""
        domain = EuclideanSpace(dim=2)
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            subspace = AffineSubspace.from_tangent_basis(domain, [e1, e2])

        half_spaces = [
            HalfSpace(domain, e1, 0.5, "<="),
            HalfSpace(domain, -e1, 0.5, "<="),
            HalfSpace(domain, e2, 0.5, "<="),
            HalfSpace(domain, -e2, 0.5, "<="),
        ]
        poly = PolyhedralSet(domain, half_spaces)

        fig, ax, payload = plot_slice(
            poly, subspace, bounds=(-1.0, 1.0, -1.0, 1.0), grid_size=10, show_plot=False
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        assert isinstance(payload, np.ndarray)
        assert payload.ndim == 2
        assert payload.shape[1] == 2

        plt.close(fig)

    def test_plot_slice_ball_1d_returns_mask(self):
        """plot_slice on a 1D subspace of a 2D domain returns (fig, ax, mask)."""
        domain = EuclideanSpace(dim=2)
        e1 = np.array([1.0, 0.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            subspace = AffineSubspace.from_tangent_basis(domain, [e1])

        ball = Ball(domain, np.zeros(2), radius=0.5, open_set=False)

        fig, ax, payload = plot_slice(
            ball, subspace, bounds=(-1.0, 1.0), grid_size=20, show_plot=False
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        assert isinstance(payload, np.ndarray)
        assert payload.shape == (20,)
        assert payload.dtype == bool
        assert payload.any()  # Points near origin lie inside the ball

        plt.close(fig)

    def test_plot_slice_3d_raises_not_implemented(self):
        """plot_slice must raise NotImplementedError for 3D subspaces (honest API)."""
        domain = EuclideanSpace(dim=3)
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        e3 = np.array([0.0, 0.0, 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            subspace = AffineSubspace.from_tangent_basis(domain, [e1, e2, e3])

        ball = Ball(domain, np.zeros(3), radius=0.5, open_set=False)

        with pytest.raises(NotImplementedError, match="3D"):
            plot_slice(ball, subspace, bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), show_plot=False)


# =============================================================================
# Phase 3: Subset.plot() entry point (cross-module integration)
# =============================================================================


class TestSubsetPlotEntryPoint:
    """Phase 3: verify Subset.plot() is available and delegates to plot_slice()."""

    def test_ball_2d_plot_method_exists(self):
        """Ball instances must have a .plot() method."""
        domain = EuclideanSpace(dim=2)
        ball = Ball(domain, np.zeros(2), radius=0.5, open_set=False)
        assert callable(getattr(ball, "plot", None))

    def test_ball_plot_with_explicit_subspace(self):
        """Ball.plot() with explicit on_subspace returns (fig, ax, payload) tuple."""
        domain = EuclideanSpace(dim=2)
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            subspace = AffineSubspace.from_tangent_basis(domain, [e1, e2])

        ball = Ball(domain, np.zeros(2), radius=0.5, open_set=False)
        result = ball.plot(subspace, bounds=(-1.0, 1.0, -1.0, 1.0), grid_size=10, show_plot=False)

        assert isinstance(result, tuple)
        assert len(result) == 3
        fig, ax, payload = result
        assert isinstance(fig, matplotlib.figure.Figure)
        assert payload.shape == (10, 10)
        plt.close(fig)

    def test_ball_plot_auto_default_2d(self):
        """Ball.plot() on 2D EuclideanSpace without on_subspace succeeds with auto-default."""
        domain = EuclideanSpace(dim=2)
        ball = Ball(domain, np.zeros(2), radius=0.5, open_set=False)
        fig, ax, payload = ball.plot(show_plot=False, grid_size=8)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert payload.shape == (8, 8)
        plt.close(fig)

    def test_ball_plot_requires_subspace_for_3d(self):
        """Ball.plot() on 3D domain without on_subspace raises ValueError."""
        domain = EuclideanSpace(dim=3)
        ball = Ball(domain, np.zeros(3), radius=0.5, open_set=False)
        with pytest.raises(ValueError, match="on_subspace"):
            ball.plot(show_plot=False)
