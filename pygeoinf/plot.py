"""
Plotting module for pygeoinf measures and distributions.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.axes import Axes
import matplotlib.figure
import matplotlib.axes
import numpy as np
import scipy.stats as stats
import scipy.optimize
import scipy.spatial
from typing import Union, List, Optional, Tuple, Any, TYPE_CHECKING

from .gaussian_measure import GaussianMeasure
from .hilbert_space import EuclideanSpace
from .subsets import Subset, PolyhedralSet

if TYPE_CHECKING:
    from .subspaces import AffineSubspace

# Define a flexible type alias for the 1D plotting function, which allows
# both our custom GaussianMeasures and scipy's frozen multivariate normals
MeasureType = Union[GaussianMeasure, Any]


def plot_1d_distributions(
    posterior_measures: Union[MeasureType, List[MeasureType]],
    /,
    *,
    prior_measures: Optional[Union[MeasureType, List[MeasureType]]] = None,
    true_value: Optional[float] = None,
    ax: Optional[Axes] = None,
    xlabel: str = "Property Value",
    title: str = "Prior and Posterior Probability Distributions",
    prior_labels: Optional[Union[str, List[str]]] = None,
    posterior_labels: Optional[Union[str, List[str]]] = None,
    width_scaling: float = 6.0,
    legend_position: tuple = (0.95, 0.95),
    fill_density: bool = False,
    **kwargs,
) -> Union[Axes, Tuple[Axes, Axes]]:
    """
    Plot 1D probability distributions for prior and posterior measures using dual y-axes.

    Args:
        posterior_measures: Single measure or list of measures for posterior distributions
        prior_measures: Single measure or list of measures for prior distributions (optional)
        true_value: True value to mark with a vertical line (optional)
        ax: An existing Matplotlib Axes object. If None, plots to the current active axes.
        xlabel: Label for x-axis
        title: Title for the plot
        prior_labels: Manual labels for prior distributions (optional)
        posterior_labels: Manual labels for posterior distributions (optional)
        width_scaling: Width scaling factor in standard deviations (default: 6.0)
        legend_position: Position of legend as (x, y) tuple (default: (0.95, 0.95))
        fill_density: Whether to fill the area under the PDF curves (default: False)
        **kwargs: Additional kwargs (e.g., `figsize`) safely ignored or forwarded.

    Returns:
        ax1 (if no priors) or (ax1, ax2) if dual axes are used.
    """
    kwargs.pop("figsize", None)

    # Convert single inputs to lists for uniform handling
    if not isinstance(posterior_measures, list):
        posterior_measures = [posterior_measures]
    if prior_measures is not None and not isinstance(prior_measures, list):
        prior_measures = [prior_measures]
    if prior_labels is not None and not isinstance(prior_labels, list):
        prior_labels = [prior_labels]
    if posterior_labels is not None and not isinstance(posterior_labels, list):
        posterior_labels = [posterior_labels]

    # Define color sequences
    prior_colors = [
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    posterior_colors = [
        "blue",
        "red",
        "darkgreen",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
    ]

    # Helper function to extract stats strictly
    def extract_stats(measures: List[MeasureType]) -> List[Tuple[float, float]]:
        stats_list = []
        for measure in measures:
            if isinstance(measure, GaussianMeasure):
                # Validated pygeoinf measure
                mean = measure.expectation[0]
                var = measure.covariance.matrix(dense=True)[0, 0]
                std = np.sqrt(var)
            elif hasattr(measure, "mean") and hasattr(measure, "cov"):
                # Fallback for scipy.stats multivariate normal distributions
                mean = measure.mean[0]
                std = np.sqrt(measure.cov[0, 0])
            else:
                raise TypeError(
                    f"Expected a GaussianMeasure or scipy.stats distribution, "
                    f"but got an instance of {type(measure).__name__}."
                )
            stats_list.append((mean, std))
        return stats_list

    # Calculate statistics for all distributions
    posterior_stats = extract_stats(posterior_measures)
    prior_stats = extract_stats(prior_measures) if prior_measures else []
    all_stats = posterior_stats + prior_stats

    # --- Smart Span Calculation ---
    max_z_score = width_scaling
    if true_value is not None:
        for mean, std in all_stats:
            if std > 0:
                z = abs(true_value - mean) / std
                max_z_score = max(max_z_score, z * 1.05)  # 5% visual buffer

    # Calculate x-axis bounds using the dynamic max_z_score
    x_min = min([mean - max_z_score * std for mean, std in all_stats if std > 0])
    x_max = max([mean + max_z_score * std for mean, std in all_stats if std > 0])

    # --- Dynamic Grid Resolution ---
    span_width = x_max - x_min
    valid_stds = [std for _, std in all_stats if std > 0]

    if valid_stds:
        min_std = min(valid_stds)
        # Ensure we have at least 25 points per standard deviation of the narrowest peak
        required_points = int((span_width / min_std) * 25)
        n_pts = min(10000, max(1000, required_points))
    else:
        n_pts = 1000

    x_axis = np.linspace(x_min, x_max, n_pts)

    # Get or create axes
    ax1 = plt.gca() if ax is None else ax

    # Plot priors on the first axis (left y-axis) if provided
    if prior_measures is not None:
        color1 = prior_colors[0] if len(prior_measures) > 0 else "green"
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("Prior Probability Density", color=color1)

        for i, (measure, (mean, std)) in enumerate(zip(prior_measures, prior_stats)):
            color = prior_colors[i % len(prior_colors)]
            pdf_values = stats.norm.pdf(x_axis, loc=mean, scale=std)

            if prior_labels is not None and i < len(prior_labels):
                label = prior_labels[i]
            elif len(prior_measures) == 1:
                label = f"Prior PDF (Mean: {mean:.5f})"
            else:
                label = f"Prior {i+1} (Mean: {mean:.5f})"

            ax1.plot(x_axis, pdf_values, color=color, lw=2, linestyle=":", label=label)
            if fill_density:
                ax1.fill_between(x_axis, pdf_values, color=color, alpha=0.15)

        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True, linestyle="--")
    else:
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("Probability Density")
        ax1.grid(True, linestyle="--")

    # Create second y-axis for posteriors (or use first if no priors)
    if prior_measures is not None:
        ax2 = ax1.twinx()
        color2 = posterior_colors[0] if len(posterior_measures) > 0 else "blue"
        ax2.set_ylabel("Posterior Probability Density", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.grid(False)
        plot_ax = ax2
    else:
        plot_ax = ax1

    # Plot posteriors
    for i, (measure, (mean, std)) in enumerate(
        zip(posterior_measures, posterior_stats)
    ):
        color = posterior_colors[i % len(posterior_colors)]
        pdf_values = stats.norm.pdf(x_axis, loc=mean, scale=std)

        if posterior_labels is not None and i < len(posterior_labels):
            label = posterior_labels[i]
        elif len(posterior_measures) == 1:
            label = f"Posterior PDF (Mean: {mean:.5f})"
        else:
            label = f"Posterior {i+1} (Mean: {mean:.5f})"

        plot_ax.plot(x_axis, pdf_values, color=color, lw=2, label=label)
        if fill_density:
            plot_ax.fill_between(x_axis, pdf_values, color=color, alpha=0.2)

    # Plot true value if provided
    if true_value is not None:
        ax1.axvline(
            true_value,
            color="black",
            linestyle="-",
            lw=2,
            label=f"True Value: {true_value:.5f}",
        )

    # Create combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    if prior_measures is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        all_handles = handles1 + handles2
        all_labels = [h.get_label() for h in all_handles]
    else:
        all_handles = handles1
        all_labels = [h.get_label() for h in all_handles]

    ax1.legend(
        all_handles, all_labels, loc="upper right", bbox_to_anchor=legend_position
    )
    ax1.set_title(title, fontsize=16, pad=15)

    return (ax1, ax2) if prior_measures is not None else ax1



class SubspaceSlicePlotter:
    """
    Plotter for visualizing subsets sliced along 1D, 2D, or 3D affine subspaces.

    **Fully implemented for 1D, 2D, and 3D subspaces** via two rendering paths:

    - ``PolyhedralSet`` → exact affine slice via ``scipy.spatial.HalfspaceIntersection``
      + convex hull; payload is vertex array (exact for all three dimensions).
    - All other sets → raster oracle sampling on a ``grid_size^n`` grid; payload is
      boolean membership mask.  For 3D, the mask is rendered as filled voxels using
      Matplotlib's ``mpl_toolkits.mplot3d`` backend (``Axes3D.voxels()``).

    Architecture:

    - Common methods (``parse_bounds``, ``embed_point``, ``sample_membership``) work
      for 1D, 2D, and 3D.
    - Dimension-specific ``_render_*()`` methods handle visualization.
    """

    # ===========================
    # Constructor & Initialization
    # ===========================

    def __init__(
        self,
        subset: Subset,
        on_subspace: "AffineSubspace",
        grid_size: int = 200,
        rtol: float = 1e-6,
        alpha: float = 0.5,
        bar_pixel_height: int = 6,
    ) -> None:
        """
        Initialize the plotter with geometry and subset.

        Args:
            subset: The Subset to visualize (domain must be EuclideanSpace)
            on_subspace: The AffineSubspace to slice along (1D, 2D, or 3D)
            grid_size: Number of samples per dimension
                - 1D: Total sample count
                - 2D: grid_size per axis (grid_size² total points)
                - 3D: grid_size per axis (grid_size³ total points)
            rtol: Relative tolerance for subset.is_element() oracle
            alpha: Transparency for visualization (0.0–1.0)
            bar_pixel_height: Visual thickness of 1D bars in pixels (positive int)

        Raises:
            TypeError: If subset.domain is not EuclideanSpace
            ValueError: If subspace dimension is not 1, 2, or 3
        """
        self.subset = subset
        self.subspace = on_subspace
        self.domain = subset.domain
        self.grid_size = grid_size
        self.rtol = rtol
        self.alpha = alpha

        # Extract tangent basis and translation from subspace
        self.tangent_basis = on_subspace.get_tangent_basis()
        self.translation = on_subspace.translation
        self.dimension = len(self.tangent_basis)

        # Validation
        if not isinstance(self.domain, EuclideanSpace):
            raise TypeError(
                f"SubspaceSlicePlotter requires EuclideanSpace domain, "
                f"got {type(self.domain).__name__}."
            )

        if self.dimension not in (1, 2, 3):
            raise ValueError(
                f"Subspace dimension must be 1, 2, or 3, got {self.dimension}D."
            )

        # Additional parameter validation
        if not isinstance(grid_size, int) or grid_size <= 0:
            raise ValueError(f"grid_size must be positive int, got {grid_size}.")

        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}.")

        if rtol <= 0:
            raise ValueError(f"rtol must be positive, got {rtol}.")

        # Validate and store pixel-based bar height for 1D rendering
        if not isinstance(bar_pixel_height, int) or bar_pixel_height <= 0:
            raise ValueError(f"bar_pixel_height must be a positive int, got {bar_pixel_height}.")
        self.bar_pixel_height = bar_pixel_height

        # Warn about 3D sampling cost: grid_size³ membership oracle calls.
        # At grid_size=200 that is 8 million calls — almost always unintended.
        # PolyhedralSet bypasses sampling entirely, so skip the warning for it.
        _3D_GRID_WARN_THRESHOLD = 30
        if self.dimension == 3 and grid_size > _3D_GRID_WARN_THRESHOLD and not isinstance(subset, PolyhedralSet):
            import warnings as _warnings
            _warnings.warn(
                f"3D sampled rendering will evaluate {grid_size**3:,} membership oracle "
                f"calls (grid_size={grid_size}). Consider grid_size ≤ {_3D_GRID_WARN_THRESHOLD} "
                "for interactive use, or use a PolyhedralSet which takes the fast exact path.",
                UserWarning,
                stacklevel=3,
            )

    # ===========================
    # Common Methods (All Dims)
    # ===========================

    def parse_bounds(
        self,
        bounds: Optional[Union[tuple, List]]
    ) -> tuple:
        """
        Parse and validate bounds for current dimension.

        Flexible input format handling:
        - None: Use default [-1, 1] per dimension
        - 1D: (u_min, u_max)
        - 2D: (u_min, u_max, v_min, v_max) OR ((u_min, u_max), (v_min, v_max))
        - 3D: (u_min, u_max, v_min, v_max, w_min, w_max) OR
              ((u_min, u_max), (v_min, v_max), (w_min, w_max))

        Args:
            bounds: User-provided bounds or None

        Returns:
            Normalized tuple:
            - 1D: (u_min, u_max)
            - 2D: (u_min, u_max, v_min, v_max)
            - 3D: (u_min, u_max, v_min, v_max, w_min, w_max)

        Raises:
            ValueError: If bounds format doesn't match dimension
        """
        if bounds is None:
            # Default: [-1, 1] per dimension
            return tuple([-1.0, 1.0] * self.dimension)

        bounds_tuple = tuple(bounds) if not isinstance(bounds, tuple) else bounds

        if self.dimension == 1:
            if len(bounds_tuple) == 2 and isinstance(bounds_tuple[0], (int, float)):
                return tuple(float(b) for b in bounds_tuple)  # type: ignore
            else:
                raise ValueError(
                    f"1D bounds must be (u_min, u_max), got {bounds}."
                )

        elif self.dimension == 2:
            if len(bounds_tuple) == 4:
                # (u_min, u_max, v_min, v_max)
                return tuple(float(b) for b in bounds_tuple)  # type: ignore
            elif len(bounds_tuple) == 2 and all(isinstance(b, (tuple, list)) for b in bounds_tuple):
                # ((u_min, u_max), (v_min, v_max))
                (u_min, u_max), (v_min, v_max) = bounds_tuple  # type: ignore
                return (float(u_min), float(u_max), float(v_min), float(v_max))
            else:
                raise ValueError(
                    f"2D bounds must be (u_min, u_max, v_min, v_max) or "
                    f"((u_min, u_max), (v_min, v_max)), got {bounds}."
                )

        # This handles dimension == 3 (ensures all code paths return)
        if len(bounds_tuple) == 6:
            # (u_min, u_max, v_min, v_max, w_min, w_max)
            return tuple(float(b) for b in bounds_tuple)  # type: ignore
        elif len(bounds_tuple) == 3 and all(isinstance(b, (tuple, list)) for b in bounds_tuple):
            # ((u_min, u_max), (v_min, v_max), (w_min, w_max))
            (u_min, u_max), (v_min, v_max), (w_min, w_max) = bounds_tuple  # type: ignore
            return (float(u_min), float(u_max), float(v_min), float(v_max), float(w_min), float(w_max))
        else:
            raise ValueError(
                f"3D bounds format error: {bounds}."
            )

    def embed_point(
        self,
        params: Union[float, Tuple[float, ...], List[float]]
    ) -> object:
        """
        Map parameter(s) to ambient point using tangent basis.

        Universal formula (works for any dimension):
        x = translation + sum(params[i] * tangent_basis[i])

        Args:
            params:
            - 1D: Single float
            - 2D: 2-tuple (u, v)
            - 3D: 3-tuple (u, v, w)

        Returns:
            Ambient point as Vector
        """
        # Handle single float for 1D
        if isinstance(params, (int, float)):
            params = [params]
        else:
            params = list(params)

        if len(params) != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} parameter(s), got {len(params)}."
            )

        # Start with translation
        result = self.translation

        # Add weighted tangent vectors
        for param, basis_vec in zip(params, self.tangent_basis):
            result = self.domain.add(result, self.domain.multiply(param, basis_vec))

        return result

    def _generate_param_grid(
        self,
        bounds: tuple
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Generate parameter grid for sampling.

        Args:
            bounds: Normalized bounds tuple
                - 1D: (u_min, u_max)
                - 2D: (u_min, u_max, v_min, v_max)
                - 3D: (u_min, u_max, v_min, v_max, w_min, w_max)

        Returns:
            Parameter grid:
            - 1D: 1D numpy array of shape (grid_size,)
            - 2D: Tuple of two 2D arrays from meshgrid, shapes (grid_size, grid_size)
            - 3D: Tuple of three 3D arrays from meshgrid, shapes (grid_size³)
        """
        if self.dimension == 1:
            u_min, u_max = bounds
            return np.linspace(u_min, u_max, self.grid_size)

        elif self.dimension == 2:
            u_min, u_max, v_min, v_max = bounds
            u = np.linspace(u_min, u_max, self.grid_size)
            v = np.linspace(v_min, v_max, self.grid_size)
            U, V = np.meshgrid(u, v, indexing='xy')
            return (U, V)

        elif self.dimension == 3:
            u_min, u_max, v_min, v_max, w_min, w_max = bounds
            u = np.linspace(u_min, u_max, self.grid_size)
            v = np.linspace(v_min, v_max, self.grid_size)
            w = np.linspace(w_min, w_max, self.grid_size)
            # indexing='ij': U[i,j,k]=u[i], V[i,j,k]=v[j], W[i,j,k]=w[k]
            # so mask[i,j,k] = membership at (u[i], v[j], w[k]), matching
            # parameter coords (Local Param 1=u, 2=v, 3=w).
            U, V, W = np.meshgrid(u, v, w, indexing='ij')
            return (U, V, W)

    def _pixel_to_data_height(self, ax: matplotlib.axes.Axes, pixels: int) -> float:
        """Convert a vertical pixel distance to data units for the given Axes.

        Uses the axes window extent and inverted data transform to compute how
        many data units correspond to a vertical distance of `pixels`. This
        produces consistent visual thickness across different figure sizes
        and axis limits.
        """
        fig = ax.figure
        # Ensure renderer is available; draw if necessary
        try:
            renderer = fig.canvas.get_renderer()
        except Exception:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

        bbox = ax.get_window_extent(renderer=renderer)
        # Use center of axes to avoid edge artifacts
        x_disp = bbox.x0 + bbox.width * 0.5
        y_disp0 = bbox.y0 + bbox.height * 0.5
        y_disp1 = y_disp0 + pixels

        inv = ax.transData.inverted()
        y_data0 = inv.transform((x_disp, y_disp0))[1]
        y_data1 = inv.transform((x_disp, y_disp1))[1]
        return abs(y_data1 - y_data0)

    def sample_membership(
        self,
        param_grid: Union[np.ndarray, Tuple[np.ndarray, ...]]
    ) -> np.ndarray:
        """
        Evaluate subset membership on parameter grid.

        For each grid point, converts parameter coordinates to ambient space
        via embed_point(), then tests membership using subset.is_element().

        Args:
            param_grid: Parameter grid from _generate_param_grid()

        Returns:
            Boolean mask array:
            - 1D: shape (grid_size,)
            - 2D: shape (grid_size, grid_size)
            - 3D: shape (grid_size, grid_size, grid_size)
        """
        if self.dimension == 1:
            u = param_grid
            mask = np.zeros(len(u), dtype=bool)
            for i in range(len(u)):
                point = self.embed_point(float(u[i]))
                mask[i] = self.subset.is_element(point, rtol=self.rtol)
            return mask

        elif self.dimension == 2:
            U, V = param_grid
            shape = U.shape
            mask = np.zeros(shape, dtype=bool)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    point = self.embed_point((float(U[i, j]), float(V[i, j])))
                    mask[i, j] = self.subset.is_element(point, rtol=self.rtol)
            return mask

        elif self.dimension == 3:
            U, V, W = param_grid
            shape = U.shape
            mask = np.zeros(shape, dtype=bool)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        point = self.embed_point((float(U[i, j, k]),
                                                  float(V[i, j, k]),
                                                  float(W[i, j, k])))
                        mask[i, j, k] = self.subset.is_element(point, rtol=self.rtol)
            return mask

    # ===========================
    # Rendering Methods
    # ===========================

    def _render_1d(
        self,
        param_grid: np.ndarray,
        mask: np.ndarray,
        bounds: tuple,
        color: str,
        show_plot: bool,
        ax: Optional[matplotlib.axes.Axes],
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, np.ndarray]:
        """
        Render 1D slice as horizontal colored segment bars.

        Visualizes contiguous membership regions as horizontal bars.

        Args:
            param_grid: 1D parameter array from _generate_param_grid()
            mask: 1D boolean mask from sample_membership()
            bounds: (u_min, u_max) from parse_bounds()
            color: Color for membership bars (e.g., 'steelblue')
            show_plot: Whether to call plt.show()
            ax: Optional existing Axes

        Returns:
            (fig, ax, mask) tuple
        """
        u_min, u_max = bounds
        u = param_grid

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 2))
        else:
            fig = ax.figure

        # Convert pixel height to data coordinates (consistent visual thickness)
        pixel_height = self.bar_pixel_height
        try:
            height_data = self._pixel_to_data_height(ax, pixel_height)
        except Exception:
            # If something unexpected happens, fall back to a small fraction of axis range
            fig.canvas.draw()
            height_data = self._pixel_to_data_height(ax, pixel_height)

        # Find contiguous membership regions
        changes = np.diff(mask.astype(int), prepend=0, append=0)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        # Draw horizontal bars for each membership region
        for start, end in zip(starts, ends):
            u_start = u[min(start, len(u) - 1)]
            u_end = u[min(end - 1, len(u) - 1)]
            width = u_end - u_start
            ax.barh(0, width, left=u_start, height=height_data, color=color,
                    alpha=self.alpha, edgecolor='black', linewidth=1.5)

        # Formatting
        ax.set_xlim(u_min, u_max)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("Line Parameter (Local Coord 1)")
        ax.set_title(f"1D Slice: {self.subset.__class__.__name__}")
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')

        if show_plot:
            plt.show()

        return fig, ax, mask

    def _render_2d(
        self,
        param_grid: Tuple[np.ndarray, np.ndarray],
        mask: np.ndarray,
        bounds: tuple,
        cmap: str,
        show_plot: bool,
        ax: Optional[matplotlib.axes.Axes],
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, np.ndarray]:
        """
        Render 2D slice using contourf and contour lines.

        Preserves existing 2D visualization logic.

        Args:
            param_grid: (U, V) meshgrids from _generate_param_grid()
            mask: 2D boolean mask from sample_membership()
            bounds: (u_min, u_max, v_min, v_max) from parse_bounds()
            cmap: Colormap name (e.g., 'Blues', 'Reds')
            show_plot: Whether to call plt.show()
            ax: Optional existing Axes

        Returns:
            (fig, ax, mask) tuple
        """
        u_min, u_max, v_min, v_max = bounds
        U, V = param_grid

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        # Contourf for filled regions, contour for boundary
        ax.contourf(U, V, mask.astype(float), levels=[0.5, 1.5],
                    cmap=cmap, alpha=self.alpha)
        ax.contour(U, V, mask.astype(float), levels=[0.5],
                   colors='k', linewidths=1.0)

        # Labels and formatting
        ax.set_xlabel("Local Param 1")
        ax.set_ylabel("Local Param 2")
        ax.set_title(f"2D Slice: {self.subset.__class__.__name__}")
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(u_min, u_max)
        ax.set_ylim(v_min, v_max)

        if show_plot:
            plt.show()

        return fig, ax, mask

    def _render_3d(
        self,
        param_grid: Tuple[np.ndarray, np.ndarray, np.ndarray],
        mask: np.ndarray,
        bounds: tuple,
        cmap: str,
        show_plot: bool,
        ax: Optional[matplotlib.axes.Axes],
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, np.ndarray]:
        """
        Render 3D slice as filled voxels using Matplotlib's ``mplot3d`` backend.

        Uses ``Axes3D.voxels()`` to display the boolean membership mask as a
        set of coloured cubes.  Each voxel corresponds to one cell of the
        parameter grid; voxels whose centre lies inside the subset are filled.

        Args:
            param_grid: (U, V, W) meshgrids from ``_generate_param_grid()``.
            mask: 3D boolean membership mask from ``sample_membership()``;
                shape ``(grid_size, grid_size, grid_size)``.
            bounds: ``(u_min, u_max, v_min, v_max, w_min, w_max)``.
            cmap: Colormap name used to derive the voxel face colour.
            show_plot: Whether to call ``plt.show()``.
            ax: Optional existing ``Axes3D``; if *None* a new figure is created.

        Returns:
            ``(fig, ax3, mask)`` where *ax3* is an ``Axes3D`` instance and
            *mask* is the same boolean array passed in (payload).
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection

        u_min, u_max, v_min, v_max, w_min, w_max = bounds

        if ax is None:
            fig = plt.figure(figsize=(7, 6))
            ax3 = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure
            ax3 = ax

        facecolor = plt.get_cmap(cmap)(0.6)
        r, g, b, _ = facecolor

        # Build edge arrays so voxels are positioned in parameter coordinates
        # rather than raw voxel-index space.
        # mask[i,j,k] = membership at (u[i], v[j], w[k]) (indexing='ij').
        # voxels(x_edges, y_edges, z_edges, filled) expects (grid_size+1,)
        # 1-D arrays for uniform grids (broadcast form).
        n = self.grid_size
        u_edges = np.linspace(u_min, u_max, n + 1)
        v_edges = np.linspace(v_min, v_max, n + 1)
        w_edges = np.linspace(w_min, w_max, n + 1)
        X_e, Y_e, Z_e = np.meshgrid(u_edges, v_edges, w_edges, indexing='ij')
        # Bake alpha into the RGBA tuple to avoid masked-array broadcast issues.
        ax3.voxels(X_e, Y_e, Z_e, mask, facecolors=(r, g, b, self.alpha))

        ax3.set_xlabel("Local Param 1 (u)")
        ax3.set_ylabel("Local Param 2 (v)")
        ax3.set_zlabel("Local Param 3 (w)")
        ax3.set_title(f"3D Slice: {self.subset.__class__.__name__}")

        if show_plot:
            plt.show()

        return fig, ax3, mask

    # ===========================
    # Main Dispatcher
    # ===========================

    def _resolve_backend(self, backend: str) -> str:
        """Resolve the effective rendering backend for the current dimension.

        Plotly is only used for 3D plots.  For 1D/2D the result is always
        ``"matplotlib"``.  For 3D:

        - ``"matplotlib"`` → ``"matplotlib"``
        - ``"plotly"``     → ``"plotly"`` (raises ``ImportError`` if not installed)
        - ``"auto"``       → ``"plotly"`` when plotly is importable, otherwise
          ``"matplotlib"`` with a ``UserWarning``.
        """
        if self.dimension != 3:
            return "matplotlib"

        if backend == "matplotlib":
            return "matplotlib"

        if backend == "plotly":
            try:
                import plotly.graph_objects  # noqa: F401
            except ImportError:
                raise ImportError(
                    "backend='plotly' requires the 'plotly' package. "
                    "Install it with: pip install pygeoinf[interactive]"
                ) from None
            return "plotly"

        if backend == "auto":
            try:
                import plotly.graph_objects  # noqa: F401
                return "plotly"
            except ImportError:
                import warnings as _w
                _w.warn(
                    "Plotly is not installed; falling back to Matplotlib for 3D "
                    "rendering.  Install it with: pip install pygeoinf[interactive]",
                    UserWarning,
                    stacklevel=4,
                )
                return "matplotlib"

        raise ValueError(
            f"backend must be 'auto', 'matplotlib', or 'plotly', got {backend!r}"
        )

    def _render_3d_plotly(
        self,
        param_grid: Tuple[np.ndarray, np.ndarray, np.ndarray],
        mask: np.ndarray,
        bounds: tuple,
        show_plot: bool,
    ) -> tuple:
        """Render 3D sampled membership mask as an interactive Plotly isosurface.

        Returns:
            ``(fig, None, mask)`` where *fig* is a ``plotly.graph_objects.Figure``
            and the second element is ``None`` (no Matplotlib Axes is created).
        """
        import plotly.graph_objects as go

        U, V, W = param_grid
        # Render the 0/1 membership field at the boundary level 0.5.
        # A tight band around 0.5 avoids the empty-scene case that occurs
        # when Plotly centers a single surface at value 1.0 for boolean data.
        iso_level = 0.5
        iso_half_width = 1e-3
        fig = go.Figure(data=go.Isosurface(
            x=U.ravel(),
            y=V.ravel(),
            z=W.ravel(),
            value=mask.astype(float).ravel(),
            isomin=iso_level - iso_half_width,
            isomax=iso_level + iso_half_width,
            surface_count=1,
            colorscale="Blues",
            opacity=self.alpha,
            showscale=False,
        ))
        u_min, u_max, v_min, v_max, w_min, w_max = bounds
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="Local Param 1 (u)", range=[u_min, u_max]),
                yaxis=dict(title="Local Param 2 (v)", range=[v_min, v_max]),
                zaxis=dict(title="Local Param 3 (w)", range=[w_min, w_max]),
            ),
            title=f"3D Slice: {self.subset.__class__.__name__}",
        )
        if show_plot:
            fig.show()
        return fig, None, mask

    def _render_3d_polyhedral_plotly(
        self,
        pts: np.ndarray,
        bounds: tuple,
        show_plot: bool,
    ) -> tuple:
        """Render 3D polyhedral exact vertices as an interactive Plotly mesh.

        Args:
            pts: Vertex array of shape ``(n_vertices, 3)`` in parameter space.
            bounds: ``(u_min, u_max, v_min, v_max, w_min, w_max)``.

        Returns:
            ``(fig, None, pts)``.
        """
        import plotly.graph_objects as go

        hull = scipy.spatial.ConvexHull(pts)
        fig = go.Figure(data=go.Mesh3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            i=hull.simplices[:, 0],
            j=hull.simplices[:, 1],
            k=hull.simplices[:, 2],
            opacity=self.alpha,
            color="lightblue",
            flatshading=True,
        ))
        u_min, u_max, v_min, v_max, w_min, w_max = bounds
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="Local Param 1", range=[u_min, u_max]),
                yaxis=dict(title="Local Param 2", range=[v_min, v_max]),
                zaxis=dict(title="Local Param 3", range=[w_min, w_max]),
            ),
            title=f"3D Slice (Exact): {self.subset.__class__.__name__}",
        )
        if show_plot:
            fig.show()
        return fig, None, pts

    def plot(
        self,
        bounds: Optional[Union[tuple, List]] = None,
        cmap: str = "Blues",
        color: str = "steelblue",
        show_plot: bool = True,
        ax: Optional[matplotlib.axes.Axes] = None,
        backend: str = "auto",
    ) -> tuple:
        """
        Main plotting method. Orchestrates bounds parsing, grid generation,
        membership sampling, and dimension-specific rendering.

        Args:
            bounds: Plot bounds (format depends on dimension)
            cmap: Colormap for 2D/3D (ignored for 1D)
            color: Color for 1D (ignored for 2D/3D)
            show_plot: Whether to display the plot
            ax: Optional existing Matplotlib Axes (must be ``None`` when
                ``backend='plotly'``).
            backend: Rendering backend — ``"auto"`` (default), ``"matplotlib"``,
                or ``"plotly"``.  ``"auto"`` selects Plotly for 3D when it is
                installed and falls back to Matplotlib otherwise.  1D/2D always
                use Matplotlib regardless of the backend value.

        Returns:
            ``(fig, ax, payload)`` tuple.

            When the Matplotlib backend is used *fig* is a
            ``matplotlib.figure.Figure`` and *ax* is a Matplotlib Axes.
            When the Plotly backend is used *fig* is a
            ``plotly.graph_objects.Figure`` and *ax* is ``None``.

            *payload* semantics are independent of backend:

            - Sampled path (non-``PolyhedralSet``): boolean membership mask.
            - Exact path (``PolyhedralSet``): vertex array in parameter coords.

        Raises:
            ValueError: If ``ax`` is not ``None`` when ``backend='plotly'``.
        """
        # Resolve effective backend (handles fallback + import check)
        effective_backend = self._resolve_backend(backend)

        # Reject a Matplotlib ax when Plotly is the active backend
        if effective_backend == "plotly" and ax is not None:
            raise ValueError(
                "ax must be None when backend='plotly'; pass ax=None or use "
                "backend='matplotlib'."
            )

        # Parse bounds for this dimension
        parsed_bounds = self.parse_bounds(bounds)

        # Fast path: exact affine slice of polyhedral sets.
        # This is dramatically faster and avoids rasterization artifacts.
        if isinstance(self.subset, PolyhedralSet):
            return self._plot_polyhedral_exact(
                parsed_bounds,
                cmap=cmap,
                color=color,
                show_plot=show_plot,
                ax=ax,
                backend=effective_backend,
            )

        # Generate parameter grid
        param_grid = self._generate_param_grid(parsed_bounds)

        # Sample membership on grid
        mask = self.sample_membership(param_grid)

        # Dispatch to dimension-specific renderer
        if self.dimension == 1:
            return self._render_1d(param_grid, mask, parsed_bounds, color,
                                   show_plot, ax)
        elif self.dimension == 2:
            return self._render_2d(param_grid, mask, parsed_bounds, cmap,
                                   show_plot, ax)
        elif self.dimension == 3:
            if effective_backend == "plotly":
                return self._render_3d_plotly(param_grid, mask, parsed_bounds,
                                              show_plot)
            return self._render_3d(param_grid, mask, parsed_bounds, cmap,
                                   show_plot, ax)

    # ===========================
    # Polyhedral Fast Path
    # ===========================

    def _polyhedral_inequalities_in_params(self, bounds: tuple) -> tuple[np.ndarray, np.ndarray]:
        """Build linear inequalities A u <= b for the polyhedral slice within plot bounds.

        For each ambient halfspace <a, x> <= off (or >=), with x = x0 + sum_j u_j v_j,
        we get a^T V u <= off - <a, x0>. Bound constraints are added to ensure a bounded
        intersection (required by halfspace-intersection routines).
        """
        assert isinstance(self.subset, PolyhedralSet)

        k = self.dimension
        A_rows: list[np.ndarray] = []
        b_rows: list[float] = []

        x0 = self.translation
        V = self.tangent_basis

        for hs in self.subset.half_spaces:
            a = hs.normal_vector
            off = hs.offset

            # Reduce to parameter space: a_param[j] = <a, v_j>
            a_param = np.array([self.domain.inner_product(a, vj) for vj in V], dtype=float)
            b_param = float(off - self.domain.inner_product(a, x0))

            # Convert >= to <= by multiplying by -1
            if hs.inequality_type == ">=":
                a_param = -a_param
                b_param = -b_param

            A_rows.append(a_param)
            b_rows.append(b_param)

        # Add bounding box in parameter coordinates so the slice is bounded.
        if k == 1:
            (u_min, u_max) = bounds
            A_rows.extend([np.array([1.0]), np.array([-1.0])])
            b_rows.extend([float(u_max), float(-u_min)])
        elif k == 2:
            (u_min, u_max, v_min, v_max) = bounds
            A_rows.extend(
                [
                    np.array([1.0, 0.0]),
                    np.array([-1.0, 0.0]),
                    np.array([0.0, 1.0]),
                    np.array([0.0, -1.0]),
                ]
            )
            b_rows.extend([float(u_max), float(-u_min), float(v_max), float(-v_min)])
        else:
            (u_min, u_max, v_min, v_max, w_min, w_max) = bounds
            A_rows.extend(
                [
                    np.array([1.0, 0.0, 0.0]),
                    np.array([-1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, -1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]),
                    np.array([0.0, 0.0, -1.0]),
                ]
            )
            b_rows.extend(
                [
                    float(u_max),
                    float(-u_min),
                    float(v_max),
                    float(-v_min),
                    float(w_max),
                    float(-w_min),
                ]
            )

        A = np.vstack(A_rows).astype(float)
        b = np.array(b_rows, dtype=float)
        return A, b

    def _chebyshev_center(self, A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute a strictly interior point via Chebyshev center LP.

        Maximizes radius r such that a_i^T x + r ||a_i|| <= b_i.
        Returns (x, r). If r <= 0, the feasible region may be empty or lower-dimensional.
        """
        k = A.shape[1]
        norms = np.linalg.norm(A, axis=1)
        c = np.zeros(k + 1)
        c[-1] = -1.0  # maximize r -> minimize -r
        A_ub = np.hstack([A, norms[:, None]])
        b_ub = b
        lp_bounds = [(None, None)] * k + [(0.0, None)]
        res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=lp_bounds, method="highs")
        if not res.success:
            raise ValueError(f"Failed to find an interior point for polyhedral slice: {res.message}")
        x = np.array(res.x[:k], dtype=float)
        r = float(res.x[-1])
        return x, r

    def _plot_polyhedral_exact(
        self,
        bounds: tuple,
        *,
        cmap: str,
        color: str,
        show_plot: bool,
        ax: Optional[matplotlib.axes.Axes],
        backend: str = "matplotlib",
    ) -> tuple:
        """Exact plotting for PolyhedralSet slices in 1D/2D/3D (within bounds)."""
        A, b = self._polyhedral_inequalities_in_params(bounds)
        k = self.dimension

        if k == 1:
            (u_min, u_max) = bounds
            lo = float(u_min)
            hi = float(u_max)
            eps = 1e-14
            for ai, bi in zip(A, b):
                a0 = float(ai[0])
                if abs(a0) < eps:
                    if bi < 0.0:
                        raise ValueError("Polyhedral slice is empty within bounds.")
                    continue
                val = float(bi / a0)
                if a0 > 0:
                    hi = min(hi, val)
                else:
                    lo = max(lo, val)
            if lo > hi:
                raise ValueError("Polyhedral slice is empty within bounds.")

            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 2))
            else:
                fig = ax.figure

            # Draw as a single interval bar (convex intersection with a line).
            try:
                height_data = self._pixel_to_data_height(ax, self.bar_pixel_height)
            except Exception:
                fig.canvas.draw()
                height_data = self._pixel_to_data_height(ax, self.bar_pixel_height)

            ax.barh(
                0,
                hi - lo,
                left=lo,
                height=height_data,
                color=color,
                alpha=self.alpha,
                edgecolor="black",
                linewidth=1.5,
            )
            ax.set_xlim(u_min, u_max)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel("Line Parameter (Local Coord 1)")
            ax.set_title(f"1D Slice (Exact): {self.subset.__class__.__name__}")
            ax.set_yticks([])
            ax.grid(True, alpha=0.3, axis="x")

            if show_plot:
                plt.show()

            return fig, ax, np.array([lo, hi], dtype=float)

        # 2D / 3D use halfspace intersection (bounded by the added box constraints)
        interior, radius = self._chebyshev_center(A, b)
        if radius <= 1e-10:
            # Lower-dimensional intersection or numerical degeneracy; fall back to oracle.
            # (This keeps behavior usable even for thin/degenerate slices.)
            param_grid = self._generate_param_grid(bounds)
            mask = self.sample_membership(param_grid)
            if k == 2:
                return self._render_2d(param_grid, mask, bounds, cmap, show_plot, ax)
            raise NotImplementedError(
                "3D polyhedral slice appears lower-dimensional; exact rendering is ambiguous. "
                "Try widening bounds or use 2D slicing, or implement a dedicated degeneracy handler."
            )

        halfspaces = np.hstack([A, -b[:, None]])  # a^T x - b <= 0
        hs = scipy.spatial.HalfspaceIntersection(halfspaces, interior)
        pts = np.asarray(hs.intersections, dtype=float)
        if pts.size == 0:
            raise ValueError("Polyhedral slice has no vertices within bounds.")

        facecolor = plt.get_cmap(cmap)(0.6)

        if k == 2:
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 6))
            else:
                fig = ax.figure

            hull = scipy.spatial.ConvexHull(pts)
            verts = pts[hull.vertices]
            centroid = verts.mean(axis=0)
            angles = np.arctan2(verts[:, 1] - centroid[1], verts[:, 0] - centroid[0])
            verts = verts[np.argsort(angles)]

            from matplotlib.patches import Polygon

            poly = Polygon(
                verts,
                closed=True,
                facecolor=facecolor,
                edgecolor="k",
                linewidth=1.0,
                alpha=self.alpha,
            )
            ax.add_patch(poly)

            (u_min, u_max, v_min, v_max) = bounds
            ax.set_xlim(u_min, u_max)
            ax.set_ylim(v_min, v_max)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Local Param 1")
            ax.set_ylabel("Local Param 2")
            ax.set_title(f"2D Slice (Exact): {self.subset.__class__.__name__}")

            if show_plot:
                plt.show()

            return fig, ax, verts

        # k == 3
        if backend == "plotly":
            return self._render_3d_polyhedral_plotly(pts, bounds, show_plot)

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if ax is None:
            fig = plt.figure(figsize=(7, 6))
            ax3 = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure
            ax3 = ax

        hull = scipy.spatial.ConvexHull(pts)
        triangles = pts[hull.simplices]
        poly3d = Poly3DCollection(triangles, alpha=self.alpha, facecolor=facecolor, edgecolor="k", linewidths=0.5)
        ax3.add_collection3d(poly3d)

        (u_min, u_max, v_min, v_max, w_min, w_max) = bounds
        ax3.set_xlim(u_min, u_max)
        ax3.set_ylim(v_min, v_max)
        ax3.set_zlim(w_min, w_max)
        ax3.set_xlabel("Local Param 1")
        ax3.set_ylabel("Local Param 2")
        ax3.set_zlabel("Local Param 3")
        ax3.set_title(f"3D Slice (Exact): {self.subset.__class__.__name__}")

        if show_plot:
            plt.show()

        return fig, ax3, pts


def plot_slice(
    subset: Subset,
    on_subspace: "AffineSubspace",
    bounds=None,
    grid_size: int = 200,
    rtol: float = 1e-6,
    alpha: float = 0.5,
    cmap: str = "Blues",
    color: str = "steelblue",
    show_plot: bool = True,
    ax=None,
    backend: str = "auto",
) -> Tuple[Any, Optional[matplotlib.axes.Axes], np.ndarray]:
    """
    Convenience wrapper: slice a subset along a 1D, 2D, or 3D affine subspace and plot.

    Thin wrapper over `SubspaceSlicePlotter`. See that class for full documentation
    on the ``bounds`` format and return-value semantics.

    Args:
        subset: The `Subset` to visualize (domain must be `EuclideanSpace`).
        on_subspace: A 1D, 2D, or 3D `AffineSubspace` to slice along.
        bounds: Plot bounds — passed directly to `SubspaceSlicePlotter.plot()`.
        grid_size: Samples per axis (passed to `SubspaceSlicePlotter`).
        rtol: Oracle tolerance (passed to `SubspaceSlicePlotter`).
        alpha: Fill transparency (passed to `SubspaceSlicePlotter`).
        cmap: Colormap for 2D/3D plots.
        color: Color string for 1D plots.
        show_plot: Whether to call ``plt.show()``.
        ax: Optional existing ``Axes`` (or ``Axes3D``) to draw into.
        backend: Rendering backend — ``"auto"`` (default), ``"matplotlib"``,
            or ``"plotly"``. ``"auto"`` prefers Plotly for 3D when it is
            installed and warns then falls back to Matplotlib otherwise;
            1D/2D always use Matplotlib.

    Returns:
        ``(fig, ax, payload)`` — identical to ``SubspaceSlicePlotter.plot()``.

        *payload* semantics depend on set type and dimension:

        - **Sampled path** (non-``PolyhedralSet``): boolean membership mask.

          - 1D: shape ``(grid_size,)``
          - 2D: shape ``(grid_size, grid_size)``
          - 3D: shape ``(grid_size, grid_size, grid_size)`` —
            ``mask[i, j, k]`` is ``True`` when the point at local parameter
            coordinates ``(u[i], v[j], w[k])`` lies inside the subset.

        - **Exact path** (``PolyhedralSet``): vertex array in parameter
          coordinates.

          - 1D: ``np.array([u_lo, u_hi])`` — interval endpoints
          - 2D: shape ``(n_vertices, 2)`` — polygon vertices
          - 3D: shape ``(n_vertices, 3)`` — polytope vertices

        For 3D subspaces using ``backend='matplotlib'`` (or ``backend='auto'``
        when Plotly is not installed), ``fig`` is a
        ``matplotlib.figure.Figure`` and ``ax`` is an ``Axes3D`` instance.
        For 3D subspaces using ``backend='plotly'`` (or ``backend='auto'``
        when Plotly *is* installed), ``fig`` is a
        ``plotly.graph_objects.Figure`` and ``ax`` is ``None``.

    Raises:
        TypeError: If ``subset.domain`` is not an ``EuclideanSpace``.
        ValueError: If bounds format is incompatible with the subspace dimension,
            or if ``grid_size``, ``rtol``, or ``alpha`` are out of range.
    """
    plotter = SubspaceSlicePlotter(
        subset, on_subspace, grid_size=grid_size, rtol=rtol, alpha=alpha
    )
    return plotter.plot(bounds=bounds, cmap=cmap, color=color, show_plot=show_plot, ax=ax, backend=backend)


def plot_corner_distributions(
    posterior_measure: GaussianMeasure,
    /,
    *,
    prior_measure: Optional[GaussianMeasure] = None,
    true_values: Optional[Union[List[float], np.ndarray]] = None,
    labels: Optional[List[str]] = None,
    title: str = "Joint Posterior Distribution",
    figsize: Optional[tuple] = None,
    colormap: str = "Blues",
    contour_color: str = "darkblue",
    parallel: bool = False,
    n_jobs: int = -1,
    width_scaling: float = 3.75,
    legend_position: tuple = (0.9, 0.95),
    fill_density: bool = False,
    num_sigmas: int = 3,
) -> np.ndarray:
    """
    Create a professional corner plot for multi-dimensional posterior distributions.

    Args:
        posterior_measure: Multi-dimensional posterior measure (pygeoinf GaussianMeasure)
        prior_measure: Optional prior measure to plot secondary axes showing prior standard deviations.
        true_values: True values for each dimension (optional)
        labels: Labels for each dimension (optional)
        title: Title for the plot
        figsize: Figure size tuple (if None, calculated based on dimensions)
        colormap: Colormap for 2D plots (used when fill_density=True)
        contour_color: Uniform color for the 2D contour lines (used when fill_density=False)
        parallel: Compute dense covariance matrix in parallel, default False.
        n_jobs: Number of cores to use in parallel calculations, default -1.
        width_scaling: Width scaling factor in standard deviations for default boundaries (default: 3.75)
        legend_position: Position of legend as (x, y) tuple (default: (0.9, 0.95))
        fill_density: Whether to fill the 2D contour background with color. False is recommended for sparse truth values.
        num_sigmas: Minimum number of standard deviation contours to draw (dynamically scales up to enclose true values).

    Returns:
        axes: An N x N NumPy array of Matplotlib Axes objects.
    """
    if not isinstance(posterior_measure, GaussianMeasure):
        raise TypeError(
            f"posterior_measure must be an instance of GaussianMeasure, "
            f"but got {type(posterior_measure).__name__}."
        )

    mean_posterior = posterior_measure.expectation
    cov_posterior = posterior_measure.covariance.matrix(
        dense=True, parallel=parallel, n_jobs=n_jobs
    )
    std_posterior = np.sqrt(np.diag(cov_posterior))

    if prior_measure is not None:
        if not isinstance(prior_measure, GaussianMeasure):
            raise TypeError("prior_measure must be a GaussianMeasure.")
        mean_prior = prior_measure.expectation
        cov_prior = prior_measure.covariance.matrix(
            dense=True, parallel=parallel, n_jobs=n_jobs
        )

    n_dims = len(mean_posterior)

    if labels is None:
        labels = [f"Dimension {i+1}" for i in range(n_dims)]

    if figsize is None:
        figsize = (3 * n_dims, 3 * n_dims)

    # --- Smart Contour Level Calculation (Mahalanobis Distance) ---
    effective_num_sigmas = num_sigmas
    if true_values is not None:
        max_dist = 0.0
        if n_dims > 1:
            # Check the mathematical 2D distance for every plot pair
            for i in range(n_dims):
                for j in range(i):
                    diff = np.array(
                        [
                            true_values[j] - mean_posterior[j],
                            true_values[i] - mean_posterior[i],
                        ]
                    )
                    cov_2d = np.array(
                        [
                            [cov_posterior[j, j], cov_posterior[j, i]],
                            [cov_posterior[i, j], cov_posterior[i, i]],
                        ]
                    )
                    # Add tiny epsilon to prevent singular matrix errors in perfectly correlated edge cases
                    cov_2d += np.eye(2) * 1e-12
                    inv_cov = np.linalg.inv(cov_2d)
                    dist = np.sqrt(diff.T @ inv_cov @ diff)
                    max_dist = max(max_dist, dist)
        else:
            # Fallback for 1D edge cases
            max_dist = np.abs(true_values[0] - mean_posterior[0]) / std_posterior[0]

        # Ensure we draw enough contours to swallow the furthest point, capped at 15 to prevent memory crashes
        effective_num_sigmas = min(15, max(num_sigmas, int(np.ceil(max_dist))))

    # --- Smart Span Calculation ---
    display_spans = np.zeros(n_dims)
    eval_spans = np.zeros(n_dims)

    for idx in range(n_dims):
        z_score = 0.0
        if true_values is not None:
            z_score = (
                np.abs(true_values[idx] - mean_posterior[idx]) / std_posterior[idx]
            )

        # Display window must contain the default width OR the true value with a 5% visual buffer
        display_spans[idx] = max(width_scaling, z_score * 1.05)
        # Math evaluation grid must be at least as wide as the display OR the dynamically calculated contours
        eval_spans[idx] = max(display_spans[idx], effective_num_sigmas + 1.0)

    fig, axes = plt.subplots(
        n_dims,
        n_dims,
        figsize=figsize,
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
        layout="constrained",
    )
    fig.suptitle(title, fontsize=16)

    if n_dims == 1:
        axes = np.array([[axes]])
    elif n_dims == 2:
        axes = axes.reshape(2, 2)

    pcm = None

    for i in range(n_dims):
        for j in range(n_dims):
            ax = axes[i, j]

            # --- DIAGONALS (1D PDFs) ---
            if i == j:
                mu = mean_posterior[i]
                sigma = std_posterior[i]

                i_eval = eval_spans[i]
                i_disp = display_spans[i]

                # Scale grid resolution, but cap it to prevent memory issues for extreme true values
                n_pts_1d = min(5000, max(200, int(50 * i_eval)))

                x = np.linspace(mu - i_eval * sigma, mu + i_eval * sigma, n_pts_1d)
                pdf = stats.norm.pdf(x, mu, sigma)

                ax.plot(x, pdf, "darkblue", label="Posterior PDF")

                if fill_density:
                    ax.fill_between(x, pdf, color="lightblue", alpha=0.6)

                if true_values is not None:
                    true_val = true_values[i]
                    ax.axvline(
                        true_val,
                        color="black",
                        linestyle="-",
                        label=f"True: {true_val:.2f}",
                    )

                ax.set_xlim(mu - i_disp * sigma, mu + i_disp * sigma)

                if prior_measure is not None:
                    prior_mu = mean_prior[i]
                    prior_sigma = np.sqrt(cov_prior[i, i])

                    def make_forward(p_mu, p_sig):
                        return lambda val: (val - p_mu) / p_sig

                    def make_inverse(p_mu, p_sig):
                        return lambda stds: stds * p_sig + p_mu

                    sec_ax = ax.secondary_xaxis(
                        "top",
                        functions=(
                            make_forward(prior_mu, prior_sigma),
                            make_inverse(prior_mu, prior_sigma),
                        ),
                    )
                    sec_ax.set_xlabel(
                        r"Distance from Prior Mean ($\sigma_{prior}$)",
                        fontsize=10,
                        color="darkgreen",
                    )
                    sec_ax.tick_params(axis="x", colors="darkgreen")

                if i == n_dims - 1:
                    ax.set_xlabel(labels[i])
                else:
                    ax.tick_params(labelbottom=False)

                ax.set_yticks([])
                if i == 0:
                    ax.set_ylabel("Density")
                else:
                    ax.set_ylabel("")

            # --- OFF-DIAGONALS (2D Contours) ---
            elif i > j:
                mean_2d = np.array([mean_posterior[j], mean_posterior[i]])
                cov_2d = np.array(
                    [
                        [cov_posterior[j, j], cov_posterior[j, i]],
                        [cov_posterior[i, j], cov_posterior[i, i]],
                    ]
                )

                sigma_j = std_posterior[j]
                sigma_i = std_posterior[i]

                j_eval = eval_spans[j]
                i_eval = eval_spans[i]

                # Scale grid resolution, cap to max 500x500 to prevent severe slowdowns
                n_pts_j = min(500, max(100, int(25 * j_eval)))
                n_pts_i = min(500, max(100, int(25 * i_eval)))

                x_range = np.linspace(
                    mean_2d[0] - j_eval * sigma_j,
                    mean_2d[0] + j_eval * sigma_j,
                    n_pts_j,
                )
                y_range = np.linspace(
                    mean_2d[1] - i_eval * sigma_i,
                    mean_2d[1] + i_eval * sigma_i,
                    n_pts_i,
                )

                X, Y = np.meshgrid(x_range, y_range)
                pos = np.dstack((X, Y))

                rv = stats.multivariate_normal(mean_2d, cov_2d)
                Z = rv.pdf(pos)

                peak_density = rv.pdf(mean_2d)

                # Values are sorted ascending (lowest density/outermost ring first, to highest density/innermost last)
                sigma_levels = sorted(
                    [
                        peak_density * np.exp(-0.5 * s**2)
                        for s in range(1, effective_num_sigmas + 1)
                    ]
                )

                if fill_density:
                    pcm = ax.pcolormesh(X, Y, Z, shading="auto", cmap=colormap)
                    ax.contour(X, Y, Z, colors="black", linewidths=0.5, alpha=0.6)
                    if effective_num_sigmas >= 1:
                        ax.contour(
                            X,
                            Y,
                            Z,
                            levels=[peak_density * np.exp(-0.5)],
                            colors="red",
                            linewidths=1,
                            linestyles="--",
                            alpha=0.8,
                        )
                else:
                    if sigma_levels:
                        # Extract the base RGB components of our chosen contour color
                        base_rgba = colors.to_rgba(contour_color)

                        # Build an array of opacities from faint (outer) to solid (inner)
                        min_alpha = 0.2
                        max_alpha = 0.9
                        if effective_num_sigmas == 1:
                            level_colors = [
                                (base_rgba[0], base_rgba[1], base_rgba[2], max_alpha)
                            ]
                        else:
                            # np.linspace aligns perfectly with the sorted sigma_levels:
                            # index 0 is outermost ring (gets min_alpha), last index is innermost ring (gets max_alpha)
                            alpha_array = np.linspace(
                                min_alpha, max_alpha, effective_num_sigmas
                            )
                            level_colors = [
                                (base_rgba[0], base_rgba[1], base_rgba[2], a)
                                for a in alpha_array
                            ]

                        ax.contour(
                            X,
                            Y,
                            Z,
                            levels=sigma_levels,
                            colors=level_colors,
                            linewidths=1.5,
                        )

                ax.plot(
                    mean_posterior[j],
                    mean_posterior[i],
                    "r+",
                    markersize=10,
                    mew=2,
                    label="Posterior Mean",
                )

                if true_values is not None:
                    ax.plot(
                        true_values[j],
                        true_values[i],
                        "kx",
                        markersize=10,
                        mew=2,
                        label="True Value",
                    )

                ax.set_xlim(
                    mean_2d[0] - display_spans[j] * sigma_j,
                    mean_2d[0] + display_spans[j] * sigma_j,
                )
                ax.set_ylim(
                    mean_2d[1] - display_spans[i] * sigma_i,
                    mean_2d[1] + display_spans[i] * sigma_i,
                )

                if i == n_dims - 1:
                    ax.set_xlabel(labels[j])
                else:
                    ax.tick_params(labelbottom=False)

                if j == 0:
                    ax.set_ylabel(labels[i])
                else:
                    ax.tick_params(labelleft=False)

            else:
                ax.set_visible(False)

    fig.align_labels()

    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    if n_dims > 1:
        handles2, labels2 = axes[1, 0].get_legend_handles_labels()
        handles.extend(handles2)
        labels_leg.extend(labels2)

    cleaned_labels = [label.split(":")[0] for label in labels_leg]
    unique_legend = dict(zip(cleaned_labels, handles))

    fig.legend(
        unique_legend.values(),
        unique_legend.keys(),
        loc="upper right",
        bbox_to_anchor=legend_position,
    )

    if n_dims > 1 and pcm is not None and fill_density:
        cbar = fig.colorbar(pcm, ax=axes, shrink=0.7, aspect=30, pad=0.02)
        cbar.set_label("Probability Density", size=12)

    return axes
