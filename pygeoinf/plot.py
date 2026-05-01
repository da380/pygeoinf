"""
Plotting module for pygeoinf measures and distributions.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.axes import Axes
import numpy as np
import scipy.stats as stats
from typing import Union, List, Optional, Tuple, Any

from .gaussian_measure import GaussianMeasure

# Define a flexible type alias for the 1D plotting function, which allows
# both our custom GaussianMeasures and scipy's frozen multivariate normals
MeasureType = Union[GaussianMeasure, Any]


def plot_1d_distributions(
    posterior_measures: Union[MeasureType, List[MeasureType]],
    /,
    *,
    prior_measures: Optional[Union[MeasureType, List[MeasureType]]] = None,
    true_value: Optional[float] = None,
    show_true_value_in_legend: bool = False,
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

        label_text = (
            f"True Value: {true_value:.5f}"
            if show_true_value_in_legend
            else "True Value"
        )
        ax1.axvline(true_value, color="black", linestyle="-", lw=2, label=label_text)

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


def plot_corner_distributions(
    posterior_measure: GaussianMeasure,
    /,
    *,
    prior_measure: Optional[GaussianMeasure] = None,
    true_values: Optional[Union[List[float], np.ndarray]] = None,
    show_true_value_in_legend: bool = False,
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

                    label_text = (
                        f"True: {true_val:.2f}"
                        if show_true_value_in_legend
                        else "True Value"
                    )

                    ax.axvline(
                        true_val,
                        color="black",
                        linestyle="-",
                        label=label_text,
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
