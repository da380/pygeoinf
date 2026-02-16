import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.figure
import matplotlib.axes
import numpy as np
import scipy.stats as stats
import scipy.optimize
import scipy.spatial
from typing import Union, List, Optional, Tuple, TYPE_CHECKING

from .hilbert_space import EuclideanSpace
from .subsets import Subset, PolyhedralSet

if TYPE_CHECKING:
    from .subspaces import AffineSubspace


def plot_1d_distributions(
    posterior_measures: Union[object, List[object]],
    /,
    *,
    prior_measures: Optional[Union[object, List[object]]] = None,
    true_value: Optional[float] = None,
    xlabel: str = "Property Value",
    title: str = "Prior and Posterior Probability Distributions",
    figsize: tuple = (12, 7),
    show_plot: bool = True,
):
    """
    Plot 1D probability distributions for prior and posterior measures using dual y-axes.

    Args:
        posterior_measures: Single measure or list of measures for posterior distributions
        prior_measures: Single measure or list of measures for prior distributions (optional)
        true_value: True value to mark with a vertical line (optional)
        xlabel: Label for x-axis
        title: Title for the plot
        figsize: Figure size tuple
        show_plot: Whether to display the plot

    Returns:
        fig, (ax1, ax2): Figure and axes objects
    """

    # Convert single measures to lists for uniform handling
    if not isinstance(posterior_measures, list):
        posterior_measures = [posterior_measures]

    if prior_measures is not None and not isinstance(prior_measures, list):
        prior_measures = [prior_measures]

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

    # Calculate statistics for all distributions
    posterior_stats = []
    for measure in posterior_measures:
        if hasattr(measure, "expectation") and hasattr(measure, "covariance"):
            # For pygeoinf measures
            mean = measure.expectation[0]
            var = measure.covariance.matrix(dense=True)[0, 0]
            std = np.sqrt(var)
        else:
            # For scipy distributions
            mean = measure.mean[0]
            std = np.sqrt(measure.cov[0, 0])
        posterior_stats.append((mean, std))

    prior_stats = []
    if prior_measures is not None:
        for measure in prior_measures:
            if hasattr(measure, "expectation") and hasattr(measure, "covariance"):
                # For pygeoinf measures
                mean = measure.expectation[0]
                var = measure.covariance.matrix(dense=True)[0, 0]
                std = np.sqrt(var)
            else:
                # For scipy distributions
                mean = measure.mean[0]
                std = np.sqrt(measure.cov[0, 0])
            prior_stats.append((mean, std))

    # Determine plot range to include all distributions
    all_means = [stat[0] for stat in posterior_stats]
    all_stds = [stat[1] for stat in posterior_stats]

    if prior_measures is not None:
        all_means.extend([stat[0] for stat in prior_stats])
        all_stds.extend([stat[1] for stat in prior_stats])

    if true_value is not None:
        all_means.append(true_value)
        all_stds.append(0)  # No std for true value

    # Calculate x-axis range (6 sigma coverage)
    x_min = min([mean - 6 * std for mean, std in zip(all_means, all_stds) if std > 0])
    x_max = max([mean + 6 * std for mean, std in zip(all_means, all_stds) if std > 0])

    # Add some padding around true value if needed
    if true_value is not None:
        range_size = x_max - x_min
        x_min = min(x_min, true_value - 0.1 * range_size)
        x_max = max(x_max, true_value + 0.1 * range_size)

    x_axis = np.linspace(x_min, x_max, 1000)

    # Create the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot priors on the first axis (left y-axis) if provided
    if prior_measures is not None:
        color1 = prior_colors[0] if len(prior_measures) > 0 else "green"
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("Prior Probability Density", color=color1)

        for i, (measure, (mean, std)) in enumerate(zip(prior_measures, prior_stats)):
            color = prior_colors[i % len(prior_colors)]

            # Calculate PDF values using scipy.stats
            pdf_values = stats.norm.pdf(x_axis, loc=mean, scale=std)

            # Determine label
            if len(prior_measures) == 1:
                label = f"Prior PDF (Mean: {mean:.5f})"
            else:
                label = f"Prior {i+1} (Mean: {mean:.5f})"

            ax1.plot(x_axis, pdf_values, color=color, lw=2, linestyle=":", label=label)
            ax1.fill_between(x_axis, pdf_values, color=color, alpha=0.15)

        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True, linestyle="--")
    else:
        # If no priors, use the left axis for posteriors
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
        color2 = posterior_colors[0] if len(posterior_measures) > 0 else "blue"

    # Plot posteriors
    for i, (measure, (mean, std)) in enumerate(
        zip(posterior_measures, posterior_stats)
    ):
        color = posterior_colors[i % len(posterior_colors)]

        # Calculate PDF values using scipy.stats
        pdf_values = stats.norm.pdf(x_axis, loc=mean, scale=std)

        # Determine label
        if len(posterior_measures) == 1:
            label = f"Posterior PDF (Mean: {mean:.5f})"
        else:
            label = f"Posterior {i+1} (Mean: {mean:.5f})"

        plot_ax.plot(x_axis, pdf_values, color=color, lw=2, label=label)
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

    fig.legend(all_handles, all_labels, loc="upper right", bbox_to_anchor=(0.9, 0.9))
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show_plot:
        plt.show()

    if prior_measures is not None:
        return fig, (ax1, ax2)
    return fig, ax1
    return None



class SubspaceSlicePlotter:
    """
    Unified plotter for visualizing subsets on 1D, 2D, and 3D affine subspaces.

    This class encapsulates the logic for:
    - Extracting and validating subspace geometry (tangent basis, translation)
    - Parsing flexible bounds formats for any dimension
    - Generating parameter grids (1D/2D/3D)
    - Embedding parameter space points into ambient space
    - Sampling subset membership via oracle evaluation
    - Dimension-specific visualization

    Architecture:
    - Common methods (parse_bounds, embed_point, sample_membership) work for all dimensions
    - Dimension-specific _render_*() methods handle visualization
    - This class becomes the primary implementation (replaces old plot_subset_oracle function)
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
            U, V, W = np.meshgrid(u, v, w, indexing='xy')
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
        Render 3D slice using isosurfaces (stub, future implementation).

        Args:
            param_grid: (U, V, W) meshgrids from _generate_param_grid()
            mask: 3D boolean mask from sample_membership()
            bounds: (u_min, u_max, v_min, v_max, w_min, w_max)
            cmap: Colormap name
            show_plot: Whether to display
            ax: Optional 3D axes

        Returns:
            (fig, ax, mask) tuple
        """
        raise NotImplementedError(
            "3D rendering not yet implemented. "
            "Options: ax.voxels(), isosurface via mayavi/vispy, or sliced 2D views."
        )

    # ===========================
    # Main Dispatcher
    # ===========================

    def plot(
        self,
        bounds: Optional[Union[tuple, List]] = None,
        cmap: str = "Blues",
        color: str = "steelblue",
        show_plot: bool = True,
        ax: Optional[matplotlib.axes.Axes] = None,
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, np.ndarray]:
        """
        Main plotting method. Orchestrates bounds parsing, grid generation,
        membership sampling, and dimension-specific rendering.

        Args:
            bounds: Plot bounds (format depends on dimension)
            cmap: Colormap for 2D/3D (ignored for 1D)
            color: Color for 1D (ignored for 2D/3D)
            show_plot: Whether to display the plot
            ax: Optional existing Axes

        Returns:
            (fig, ax, payload) tuple

            By default, payload is the boolean membership mask evaluated on the
            parameter grid.

            For `PolyhedralSet`, a fast exact method is used (no grid sampling)
            and the payload is instead geometric:
            - 1D: array([u_lo, u_hi]) interval endpoints
            - 2D: array of polygon vertices with shape (n_vertices, 2)
            - 3D: array of polytope vertices with shape (n_vertices, 3)
        """
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
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, np.ndarray]:
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


def plot_corner_distributions(
    posterior_measure: object,
    /,
    *,
    true_values: Optional[Union[List[float], np.ndarray]] = None,
    labels: Optional[List[str]] = None,
    title: str = "Joint Posterior Distribution",
    figsize: Optional[tuple] = None,
    show_plot: bool = True,
    include_sigma_contours: bool = True,
    colormap: str = "Blues",
    parallel: bool = False,
    n_jobs: int = -1,
):
    """
    Create a corner plot for multi-dimensional posterior distributions.

    Args:
        posterior_measure: Multi-dimensional posterior measure (pygeoinf object)
        true_values: True values for each dimension (optional)
        labels: Labels for each dimension (optional)
        title: Title for the plot
        figsize: Figure size tuple (if None, calculated based on dimensions)
        show_plot: Whether to display the plot
        include_sigma_contours: Whether to include 1-sigma contour lines
        colormap: Colormap for 2D plots
        parallel: Compute dense covariance matrix in parallel, default False.
        n_jobs: Number of cores to use in parallel calculations, default -1.

    Returns:
        fig, axes: Figure and axes array
    """

    # Extract statistics from the measure
    if hasattr(posterior_measure, "expectation") and hasattr(
        posterior_measure, "covariance"
    ):
        mean_posterior = posterior_measure.expectation
        cov_posterior = posterior_measure.covariance.matrix(
            dense=True, parallel=parallel, n_jobs=n_jobs
        )
    else:
        raise ValueError(
            "posterior_measure must have 'expectation' and 'covariance' attributes"
        )

    n_dims = len(mean_posterior)

    # Set default labels if not provided
    if labels is None:
        labels = [f"Dimension {i+1}" for i in range(n_dims)]

    # Set figure size based on dimensions if not provided
    if figsize is None:
        figsize = (3 * n_dims, 3 * n_dims)

    # Create subplots
    fig, axes = plt.subplots(n_dims, n_dims, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # Ensure axes is always 2D array
    if n_dims == 1:
        axes = np.array([[axes]])
    elif n_dims == 2:
        axes = axes.reshape(2, 2)

    # Initialize pcm variable for colorbar
    pcm = None

    for i in range(n_dims):
        for j in range(n_dims):
            ax = axes[i, j]

            if i == j:  # Diagonal plots (1D marginal distributions)
                mu = mean_posterior[i]
                sigma = np.sqrt(cov_posterior[i, i])

                # Create x-axis range
                x = np.linspace(mu - 3.75 * sigma, mu + 3.75 * sigma, 200)
                pdf = stats.norm.pdf(x, mu, sigma)

                # Plot the PDF
                ax.plot(x, pdf, "darkblue", label="Posterior PDF")
                ax.fill_between(x, pdf, color="lightblue", alpha=0.6)

                # Add true value if provided
                if true_values is not None:
                    true_val = true_values[i]
                    ax.axvline(
                        true_val,
                        color="black",
                        linestyle="-",
                        label=f"True: {true_val:.2f}",
                    )

                ax.set_xlabel(labels[i])
                ax.set_ylabel("Density" if i == 0 else "")
                ax.set_yticklabels([])

            elif i > j:  # Lower triangle: 2D joint distributions
                # Extract 2D mean and covariance
                mean_2d = np.array([mean_posterior[j], mean_posterior[i]])
                cov_2d = np.array(
                    [
                        [cov_posterior[j, j], cov_posterior[j, i]],
                        [cov_posterior[i, j], cov_posterior[i, i]],
                    ]
                )

                # Create 2D grid
                sigma_j = np.sqrt(cov_posterior[j, j])
                sigma_i = np.sqrt(cov_posterior[i, i])

                x_range = np.linspace(
                    mean_2d[0] - 3.75 * sigma_j, mean_2d[0] + 3.75 * sigma_j, 100
                )
                y_range = np.linspace(
                    mean_2d[1] - 3.75 * sigma_i, mean_2d[1] + 3.75 * sigma_i, 100
                )

                X, Y = np.meshgrid(x_range, y_range)
                pos = np.dstack((X, Y))

                # Calculate PDF values
                rv = stats.multivariate_normal(mean_2d, cov_2d)
                Z = rv.pdf(pos)

                # Create filled contour plot using pcolormesh like the original
                pcm = ax.pcolormesh(
                    X,
                    Y,
                    Z,
                    shading="auto",
                    cmap=colormap,
                    norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
                )

                # Add contour lines
                ax.contour(X, Y, Z, colors="black", linewidths=0.5, alpha=0.6)

                # Add 1-sigma contour if requested
                if include_sigma_contours:
                    # Calculate 1-sigma level (approximately 39% of peak for 2D Gaussian)
                    sigma_level = rv.pdf(mean_2d) * np.exp(-0.5)
                    ax.contour(
                        X,
                        Y,
                        Z,
                        levels=[sigma_level],
                        colors="red",
                        linewidths=1,
                        linestyles="--",
                        alpha=0.8,
                    )

                # Plot mean point
                ax.plot(
                    mean_posterior[j],
                    mean_posterior[i],
                    "r+",
                    markersize=10,
                    mew=2,
                    label="Posterior Mean",
                )

                # Plot true value if provided
                if true_values is not None:
                    ax.plot(
                        true_values[j],
                        true_values[i],
                        "kx",
                        markersize=10,
                        mew=2,
                        label="True Value",
                    )

                ax.set_xlabel(labels[j])
                ax.set_ylabel(labels[i])

            else:  # Upper triangle: hide these plots
                ax.axis("off")

    # Create legend similar to the original
    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    if n_dims > 1:
        handles2, labels2 = axes[1, 0].get_legend_handles_labels()
        handles.extend(handles2)
        labels_leg.extend(labels2)

    # Clean up labels by removing values after colons
    cleaned_labels = [label.split(":")[0] for label in labels_leg]

    fig.legend(handles, cleaned_labels, loc="upper right", bbox_to_anchor=(0.9, 0.95))

    # Adjust main plot layout to make room on the right for the colorbar
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])

    # Add a colorbar if we have 2D plots
    if n_dims > 1 and pcm is not None:
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(pcm, cax=cbar_ax)
        cbar.set_label("Probability Density", size=12)

    if show_plot:
        plt.show()

    return fig, axes
