import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pygeoinf.symmetric_space.sphere import Sobolev


def plot_synthetic_ray_paths(
    n_earthquakes: int = 3, min_magnitude: float = 6.5, n_stations: int = None
):
    """
    Plots a random subset of IRIS GSN stations, a random set of earthquakes,
    and the great-circle paths connecting them.
    """
    # 1. Initialize the manifold (passed STRICTLY positionally)
    # Sobolev(kmax, order, scale, /, *, radius=1.0, ...)
    space = Sobolev(16, 1.5, 0.1)

    # 2. Fetch the data using our new convenient static methods
    print("Loading geometry...")
    stations = space.iris_stations(n_stations=n_stations)
    earthquakes = space.random_earthquakes(
        n_points=n_earthquakes, min_magnitude=min_magnitude
    )

    # 3. Set up the map projection
    plt.figure(figsize=(14, 8))
    ax = plt.axes(projection=ccrs.Robinson())

    # Add some visual context
    ax.stock_img()
    ax.coastlines(linewidth=0.5, color="gray")

    # 4. Plot the paths (Great Circles) FIRST so they sit underneath the markers
    geodetic = ccrs.Geodetic()

    print(f"Drawing {len(stations) * n_earthquakes} great-circle paths...")
    for q_lat, q_lon in earthquakes:
        for s_lat, s_lon in stations:
            ax.plot(
                [s_lon, q_lon],
                [s_lat, q_lat],
                color="white",
                linewidth=0.3,
                alpha=0.4,
                transform=geodetic,
            )

    # 5. Extract coordinates for scatter plotting
    stat_lons = [lon for lat, lon in stations]
    stat_lats = [lat for lat, lon in stations]

    quake_lons = [lon for lat, lon in earthquakes]
    quake_lats = [lat for lat, lon in earthquakes]

    # 6. Plot the stations and earthquakes
    pc = ccrs.PlateCarree()

    ax.scatter(
        stat_lons,
        stat_lats,
        color="cyan",
        marker="^",
        s=40,
        edgecolor="black",
        linewidth=0.5,
        label=f"IRIS GSN Stations ({len(stations)})",
        transform=pc,
        zorder=5,
    )

    ax.scatter(
        quake_lons,
        quake_lats,
        color="red",
        marker="*",
        s=200,
        edgecolor="black",
        linewidth=1.0,
        label=f"Earthquakes (M > {min_magnitude})",
        transform=pc,
        zorder=6,
    )

    plt.title(
        f"Synthetic Seismic Coverage\n{len(stations)} Stations to {n_earthquakes} Events",
        fontsize=16,
        pad=15,
    )
    plt.legend(loc="lower left", scatterpoints=1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test a sparser network with 50 stations and 5 earthquakes
    plot_synthetic_ray_paths(n_earthquakes=10, n_stations=20)
