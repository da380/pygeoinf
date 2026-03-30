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
    ax: Optional[Axes] = None,
    xlabel: str = "Property Value",
    title: str = "Prior and Posterior Probability Distributions",
    prior_labels: Optional[Union[str, List[str]]] = None,
    posterior_labels: Optional[Union[str, List[str]]] = None,
    width_scaling: float = 6.0,
    legend_position: tuple = (0.95, 0.95),
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

    # Determine plot range to include all distributions
    all_means = [stat[0] for stat in posterior_stats] + [
        stat[0] for stat in prior_stats
    ]
    all_stds = [stat[1] for stat in posterior_stats] + [stat[1] for stat in prior_stats]

    if true_value is not None:
        all_means.append(true_value)
        all_stds.append(0)

    # Calculate x-axis range
    x_min = min(
        [
            mean - width_scaling * std
            for mean, std in zip(all_means, all_stds)
            if std > 0
        ]
    )
    x_max = max(
        [
            mean + width_scaling * std
            for mean, std in zip(all_means, all_stds)
            if std > 0
        ]
    )

    if true_value is not None:
        range_size = x_max - x_min
        x_min = min(x_min, true_value - 0.1 * range_size)
        x_max = max(x_max, true_value + 0.1 * range_size)

    x_axis = np.linspace(x_min, x_max, 1000)

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


def plot_corner_distributions(
    posterior_measure: GaussianMeasure,
    /,
    *,
    prior_measure: Optional[GaussianMeasure] = None,
    true_values: Optional[Union[List[float], np.ndarray]] = None,
    labels: Optional[List[str]] = None,
    title: str = "Joint Posterior Distribution",
    figsize: Optional[tuple] = None,
    include_sigma_contours: bool = True,
    colormap: str = "Blues",
    parallel: bool = False,
    n_jobs: int = -1,
    width_scaling: float = 3.75,
    legend_position: tuple = (0.9, 0.95),
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
        include_sigma_contours: Whether to include 1-sigma contour lines
        colormap: Colormap for 2D plots
        parallel: Compute dense covariance matrix in parallel, default False.
        n_jobs: Number of cores to use in parallel calculations, default -1.
        width_scaling: Width scaling factor in standard deviations (default: 3.75)
        legend_position: Position of legend as (x, y) tuple (default: (0.9, 0.95))

    Returns:
        axes: An N x N NumPy array of Matplotlib Axes objects.
    """
    # Strict type validation ensuring it's an authentic GaussianMeasure
    if not isinstance(posterior_measure, GaussianMeasure):
        raise TypeError(
            f"posterior_measure must be an instance of GaussianMeasure, "
            f"but got {type(posterior_measure).__name__}."
        )

    mean_posterior = posterior_measure.expectation
    cov_posterior = posterior_measure.covariance.matrix(
        dense=True, parallel=parallel, n_jobs=n_jobs
    )

    # Pre-compute prior matrices if provided
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

    # Tight grid spacing using the modern layout engine
    fig, axes = plt.subplots(
        n_dims,
        n_dims,
        figsize=figsize,
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
        layout="constrained",
    )
    fig.suptitle(title, fontsize=16)

    # Ensure axes is always 2D array
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
                sigma = np.sqrt(cov_posterior[i, i])
                x = np.linspace(
                    mu - width_scaling * sigma, mu + width_scaling * sigma, 200
                )
                pdf = stats.norm.pdf(x, mu, sigma)

                ax.plot(x, pdf, "darkblue", label="Posterior PDF")
                ax.fill_between(x, pdf, color="lightblue", alpha=0.6)

                if true_values is not None:
                    true_val = true_values[i]
                    ax.axvline(
                        true_val,
                        color="black",
                        linestyle="-",
                        label=f"True: {true_val:.2f}",
                    )

                # Inject prior secondary axis if requested
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

                # X-axis logic
                if i == n_dims - 1:
                    ax.set_xlabel(labels[i])
                else:
                    ax.tick_params(labelbottom=False)

                # Y-axis logic: Hide all ticks, only label the very first plot
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

                sigma_j = np.sqrt(cov_posterior[j, j])
                sigma_i = np.sqrt(cov_posterior[i, i])

                x_range = np.linspace(
                    mean_2d[0] - width_scaling * sigma_j,
                    mean_2d[0] + width_scaling * sigma_j,
                    100,
                )
                y_range = np.linspace(
                    mean_2d[1] - width_scaling * sigma_i,
                    mean_2d[1] + width_scaling * sigma_i,
                    100,
                )

                X, Y = np.meshgrid(x_range, y_range)
                pos = np.dstack((X, Y))

                rv = stats.multivariate_normal(mean_2d, cov_2d)
                Z = rv.pdf(pos)

                pcm = ax.pcolormesh(
                    X,
                    Y,
                    Z,
                    shading="auto",
                    cmap=colormap,
                    norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
                )

                ax.contour(X, Y, Z, colors="black", linewidths=0.5, alpha=0.6)

                if include_sigma_contours:
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

                # X-axis logic
                if i == n_dims - 1:
                    ax.set_xlabel(labels[j])
                else:
                    ax.tick_params(labelbottom=False)

                # Y-axis logic
                if j == 0:
                    ax.set_ylabel(labels[i])
                else:
                    ax.tick_params(labelleft=False)

            # --- EMPTY UPPER TRIANGLE ---
            else:
                ax.set_visible(False)

    # Force Matplotlib to align the outer axis labels so they don't stagger
    fig.align_labels()

    # Extract legend handles safely from the first available plots
    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    if n_dims > 1:
        handles2, labels2 = axes[1, 0].get_legend_handles_labels()
        handles.extend(handles2)
        labels_leg.extend(labels2)

    cleaned_labels = [label.split(":")[0] for label in labels_leg]

    # Avoid duplicate legends by dict conversion trick
    unique_legend = dict(zip(cleaned_labels, handles))

    # Let constrained_layout automatically make room for the legend on the right
    fig.legend(
        unique_legend.values(),
        unique_legend.keys(),
        loc="upper right",
        bbox_to_anchor=legend_position,
    )

    # Let constrained_layout handle the colorbar natively
    if n_dims > 1 and pcm is not None:
        cbar = fig.colorbar(pcm, ax=axes, shrink=0.7, aspect=30, pad=0.02)
        cbar.set_label("Probability Density", size=12)

    return axes
