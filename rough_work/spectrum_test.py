import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev


def plot_2d_power_distribution(
    all_degree_stats, title="", min_power=1e-4, max_power=10, bins_y=100
):
    """

    all_degree_stats: A dictionary where keys are degrees (int)

    and values are the 'powers' arrays from the previous function.

    title: Title of the plot.

    """

    degrees = sorted(all_degree_stats.keys())

    # Flatten the data for histogramming
    # x: repeat the degree for every sample it has
    # y: the actual power values

    x_data = []
    y_data = []

    for deg in degrees:

        samples = all_degree_stats[deg]

        x_data.extend([deg] * len(samples))

        y_data.extend(samples)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Define bins

    # X bins are centered on integers (degrees)

    x_bins = np.arange(min(degrees) - 0.5, max(degrees) + 1.5, 1)

    # Y bins are usually better on a log scale for power

    y_bins = np.logspace(np.log10(min_power), np.log10(max_power), bins_y)

    # Calculate 2D Histogram
    H, xedges, yedges = np.histogram2d(x_data, y_data, bins=[x_bins, y_bins])

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # We transpose H because histogram2d follows (x, y) but pcolormesh wants (y, x)
    pc = ax.pcolormesh(xedges, yedges, H.T, cmap="inferno", norm=LogNorm())

    ax.set_yscale("log")

    ax.set_xlabel("Spherical Harmonic Degree ($l$)")

    ax.set_ylabel("Power (km$^2$)")

    ax.set_title(title)

    fig.colorbar(pc, label="Sample Density")

    return fig, ax


# Set threads available for backends
inf.configure_threading(n_threads=4)

lmax = 64
X = Lebesgue(lmax)

mu = X.heat_kernel_gaussian_measure(0.2)

powers = X.sample_power_measure(mu, 1000, parallel=True, n_jobs=10)

powers = np.array(powers)

power_dict = {l: powers[:, l] for l in range(lmax)}

plot_2d_power_distribution(power_dict)

plt.show()
