import numpy as np


def great_circle_quadrature(p1, p2, n_points, radius=1.0):
    """
    Generates quadrature points and weights along a great circle arc between two points.

    Parameters:
    - p1, p2: Tuples of (lat, lon) in degrees.
    - n_points: Integer, order of the Gauss-Legendre quadrature.
    - radius: Radius of the sphere (default 1.0).
              Weights are scaled by this radius (ds = R * d_angle).

    Returns:
    - points: List of (lat, lon) tuples in degrees.
    - weights: List of integration weights.
    """

    # 1. Coordinate Transforms (Degrees -> Radians -> Unit Vectors)
    def to_vector(lat, lon):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        return np.array([x, y, z])

    def to_latlon(vec):
        # Normalize just in case interpolation drifted slightly
        vec = vec / np.linalg.norm(vec)
        lat_rad = np.arcsin(vec[2])
        lon_rad = np.arctan2(vec[1], vec[0])
        return (np.degrees(lat_rad), np.degrees(lon_rad))

    v1 = to_vector(*p1)
    v2 = to_vector(*p2)

    # 2. Calculate Central Angle (Omega)
    # Clamp dot product to [-1, 1] to avoid numerical errors at 1.0
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    omega = np.arccos(dot_product)

    # Handle edge case: Identical points
    if omega < 1e-10:
        return [p1], [0.0]

    # Handle edge case: Antipodal points (Path is undefined/non-unique)
    # In a real pipeline, you might want to raise an error or choose a specific path (e.g., via North Pole)
    if np.abs(omega - np.pi) < 1e-10:
        raise ValueError("Points are antipodal; great circle path is not unique.")

    # 3. Generate Gauss-Legendre Nodes and Weights
    # Nodes x are in [-1, 1], weights w are for integral on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n_points)

    # 4. Map Nodes to Path Parameter t in [0, 1]
    # Transformation: t = (x + 1) / 2
    t_vals = (x + 1) / 2

    # Scale weights:
    # Original integral is dx. We want ds.
    # ds = radius * d_angle
    # Total angle is omega.
    # Variable change x [-1, 1] -> angle [0, omega] implies factor of omega / 2.
    scaled_weights = w * (omega / 2.0) * radius

    # 5. Spherical Linear Interpolation (SLERP) for each t
    # Formula: v(t) = (sin((1-t)omega)/sin(omega))*v1 + (sin(t*omega)/sin(omega))*v2
    sin_omega = np.sin(omega)

    quadrature_points_latlon = []

    for t in t_vals:
        coeff1 = np.sin((1 - t) * omega) / sin_omega
        coeff2 = np.sin(t * omega) / sin_omega
        v_interp = coeff1 * v1 + coeff2 * v2
        quadrature_points_latlon.append(to_latlon(v_interp))

    return quadrature_points_latlon, scaled_weights


# --- Usage Example ---

# Define two points: London (approx 51.5N, 0) and New York (approx 40.7N, -74.0W)
start_pt = (51.5, 0.0)
end_pt = (40.7, -74.0)

# Use 5 quadrature points
pts, wts = great_circle_quadrature(start_pt, end_pt, n_points=5)

print(f"Start: {start_pt}")
print(f"End:   {end_pt}")
print("-" * 40)
print(f"{'Lat':<10} {'Lon':<10} {'Weight':<10}")
print("-" * 40)

for (lat, lon), weight in zip(pts, wts):
    print(f"{lat:<10.4f} {lon:<10.4f} {weight:<10.4f}")

# Verification: The sum of weights should equal the arc length (in radians if R=1)
total_arc_length_est = np.sum(wts)
print("-" * 40)
print(f"Sum of weights (Arc Length): {total_arc_length_est:.4f}")
