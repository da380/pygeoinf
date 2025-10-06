import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import scipy.stats as stats
from typing import Union, List, Optional

def plot_1d_distributions(
    posterior_measures: Union[object, List[object]],
    prior_measures: Optional[Union[object, List[object]]] = None,
    true_value: Optional[float] = None,
    xlabel: str = "Property Value",
    title: str = "Prior and Posterior Probability Distributions",
    figsize: tuple = (12, 7),
    show_plot: bool = True
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
    prior_colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    posterior_colors = ['blue', 'red', 'darkgreen', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Calculate statistics for all distributions
    posterior_stats = []
    for measure in posterior_measures:
        if hasattr(measure, 'expectation') and hasattr(measure, 'covariance'):
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
            if hasattr(measure, 'expectation') and hasattr(measure, 'covariance'):
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
        color1 = prior_colors[0] if len(prior_measures) > 0 else 'green'
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Prior Probability Density', color=color1)
        
        for i, (measure, (mean, std)) in enumerate(zip(prior_measures, prior_stats)):
            color = prior_colors[i % len(prior_colors)]
            
            # Calculate PDF values using scipy.stats
            pdf_values = stats.norm.pdf(x_axis, loc=mean, scale=std)
            
            # Determine label
            if len(prior_measures) == 1:
                label = f'Prior PDF (Mean: {mean:.5f})'
            else:
                label = f'Prior {i+1} (Mean: {mean:.5f})'
            
            ax1.plot(x_axis, pdf_values, color=color, lw=2, linestyle=':', label=label)
            ax1.fill_between(x_axis, pdf_values, color=color, alpha=0.15)
        
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, linestyle='--')
    else:
        # If no priors, use the left axis for posteriors
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Probability Density')
        ax1.grid(True, linestyle='--')
    
    # Create second y-axis for posteriors (or use first if no priors)
    if prior_measures is not None:
        ax2 = ax1.twinx()
        color2 = posterior_colors[0] if len(posterior_measures) > 0 else 'blue'
        ax2.set_ylabel('Posterior Probability Density', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.grid(False)
        plot_ax = ax2
    else:
        plot_ax = ax1
        color2 = posterior_colors[0] if len(posterior_measures) > 0 else 'blue'
    
    # Plot posteriors
    for i, (measure, (mean, std)) in enumerate(zip(posterior_measures, posterior_stats)):
        color = posterior_colors[i % len(posterior_colors)]
        
        # Calculate PDF values using scipy.stats
        pdf_values = stats.norm.pdf(x_axis, loc=mean, scale=std)
        
        # Determine label
        if len(posterior_measures) == 1:
            label = f'Posterior PDF (Mean: {mean:.5f})'
        else:
            label = f'Posterior {i+1} (Mean: {mean:.5f})'
        
        plot_ax.plot(x_axis, pdf_values, color=color, lw=2, label=label)
        plot_ax.fill_between(x_axis, pdf_values, color=color, alpha=0.2)
    
    # Plot true value if provided
    if true_value is not None:
        ax1.axvline(true_value, color='black', linestyle='-', lw=2, 
                   label=f'True Value: {true_value:.5f}')
    
    # Create combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    
    if prior_measures is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        all_handles = handles1 + handles2
        all_labels = [h.get_label() for h in all_handles]
    else:
        all_handles = handles1
        all_labels = [h.get_label() for h in all_handles]
    
    fig.legend(all_handles, all_labels, loc='upper right', bbox_to_anchor=(0.9, 0.9))
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    if show_plot:
        plt.show()
    
    if prior_measures is not None:
        return fig, (ax1, ax2)
    else:
        return fig, ax1


def plot_corner_distributions(
    posterior_measure: object,
    true_values: Optional[Union[List[float], np.ndarray]] = None,
    labels: Optional[List[str]] = None,
    title: str = "Joint Posterior Distribution",
    figsize: Optional[tuple] = None,
    show_plot: bool = True,
    include_sigma_contours: bool = True,
    colormap: str = "Blues"
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
        
    Returns:
        fig, axes: Figure and axes array
    """
    
    # Extract statistics from the measure
    if hasattr(posterior_measure, 'expectation') and hasattr(posterior_measure, 'covariance'):
        mean_posterior = posterior_measure.expectation
        cov_posterior = posterior_measure.covariance.matrix(dense=True, parallel=True)
    else:
        raise ValueError("posterior_measure must have 'expectation' and 'covariance' attributes")
    
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
                x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
                pdf = stats.norm.pdf(x, mu, sigma)
                
                # Plot the PDF
                ax.plot(x, pdf, "darkblue", label="Posterior PDF")
                ax.fill_between(x, pdf, color="lightblue", alpha=0.6)
                
                # Add true value if provided
                if true_values is not None:
                    true_val = true_values[i]
                    ax.axvline(true_val, color="black", linestyle="-", 
                              label=f"True: {true_val:.2f}")
                
                ax.set_xlabel(labels[i])
                ax.set_ylabel("Density" if i == 0 else "")
                ax.set_yticklabels([])
            
            elif i > j:  # Lower triangle: 2D joint distributions
                # Extract 2D mean and covariance
                mean_2d = np.array([mean_posterior[j], mean_posterior[i]])
                cov_2d = np.array([
                    [cov_posterior[j, j], cov_posterior[j, i]],
                    [cov_posterior[i, j], cov_posterior[i, i]]
                ])
                
                # Create 2D grid
                sigma_j = np.sqrt(cov_posterior[j, j])
                sigma_i = np.sqrt(cov_posterior[i, i])
                
                x_range = np.linspace(mean_2d[0] - 3.5 * sigma_j, 
                                    mean_2d[0] + 3.5 * sigma_j, 100)
                y_range = np.linspace(mean_2d[1] - 3.5 * sigma_i, 
                                    mean_2d[1] + 3.5 * sigma_i, 100)
                
                X, Y = np.meshgrid(x_range, y_range)
                pos = np.dstack((X, Y))
                
                # Calculate PDF values
                rv = stats.multivariate_normal(mean_2d, cov_2d)
                Z = rv.pdf(pos)
                
                # Create filled contour plot using pcolormesh like the original
                pcm = ax.pcolormesh(
                    X, Y, Z, shading="auto", cmap=colormap,
                    norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max())
                )
                
                # Add contour lines
                ax.contour(X, Y, Z, colors="black", linewidths=0.5, alpha=0.6)
                
                # Add 1-sigma contour if requested
                if include_sigma_contours:
                    # Calculate 1-sigma level (approximately 39% of peak for 2D Gaussian)
                    sigma_level = rv.pdf(mean_2d) * np.exp(-0.5)
                    ax.contour(X, Y, Z, levels=[sigma_level], colors="red", 
                              linewidths=1, linestyles="--", alpha=0.8)
                
                # Plot mean point
                ax.plot(mean_posterior[j], mean_posterior[i], "r+", 
                       markersize=10, mew=2, label="Posterior Mean")
                
                # Plot true value if provided
                if true_values is not None:
                    ax.plot(true_values[j], true_values[i], "kx", 
                           markersize=10, mew=2, label="True Value")
                
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
    
    fig.legend(
        handles, cleaned_labels,
        loc="upper right", 
        bbox_to_anchor=(0.9, 0.95)
    )
    
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