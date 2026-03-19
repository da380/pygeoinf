import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Import the spherical Lebesgue space
from pygeoinf.symmetric_space.sphere import Lebesgue

# 1. Initialize the function space on the sphere
# lmax doesn't affect point generation, but is required for initialization
space = Lebesgue(15, radius=1.0)

# 2. Generate random points on the manifold
n_points = 2000
points = space.random_points(n_points)

# 3. Cluster the points using the newly added method
# Let's group them into 8 distinct geographic regions
n_clusters = 20
blocks = space.cluster_points(points, n_clusters=n_clusters)

# 4. Visualization setup using Cartopy
fig = plt.figure(figsize=(12, 6))
# Using Mollweide projection for a nice global view
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
ax.set_global()
ax.coastlines(color="gray", alpha=0.5)
ax.gridlines(linestyle="--", alpha=0.5)

# Use a distinct colormap for the clusters
cmap = plt.get_cmap("tab10")

# 5. Plot each clustered block
for i, block_indices in enumerate(blocks):
    # Extract the (latitude, longitude) tuples for this specific cluster
    cluster_coords = [points[idx] for idx in block_indices]

    # Unpack into separate lists of lats and lons for scatter plotting
    lats, lons = zip(*cluster_coords)

    # Scatter plot onto the map using the Geodetic transform
    ax.scatter(
        lons,
        lats,
        transform=ccrs.Geodetic(),
        color=cmap(i % 10),
        s=40,
        edgecolor="black",
        linewidth=0.5,
        label=f"Cluster {i+1} ({len(block_indices)} pts)",
    )

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
plt.title(f"Geodesic Clustering on the Sphere ({n_clusters} Clusters)")
plt.tight_layout()
plt.show()
