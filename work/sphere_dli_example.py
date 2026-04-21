"""Sphere DLI Example.

Phase-velocity perturbation DLI on the two-sphere using:
- Sobolev model space (order 1.5, pyshtools spherical-harmonic backend)
- Real IRIS GSN stations and randomly sampled USGS earthquakes
- Great-circle path integrals normalised to true path averages
- Spherical-cap averaging target properties
- Norm-ball support functions for the model prior and data-confidence set
- DualMasterCostFunction + ProximalBundleMethod for the DLI solve

Entry points:
  run_example(...)  — builds and solves the full problem, returns a result dict.
  plot_results(...) — produces three diagnostic figures from the result dict.

Run as a script::

    conda activate inferences3
    cd pygeoinf
    python work/sphere_dli_example.py
"""

from __future__ import annotations

import random
from typing import Iterable

import numpy as np

from pygeoinf.linear_forms import LinearForm
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.symmetric_space.sphere import Sobolev


ORDER = 1.5
SCALE = 0.1
PRIOR_SCALE = 0.05
N_TARGET = 6
N_SOURCES = 5
N_RECEIVERS = 10
SIGMA_NOISE = 0.02
DEFAULT_TARGET_LATLON = [
    (-60.0, 0.0),     # Cap 1 — South Atlantic (intentionally uncovered by rays)
    (5.0, 143.0),     # Cap 2 — north of Papua New Guinea
    (0.0, -120.0),    # Cap 3 — central Pacific
    (46.0, 104.0),    # Cap 4 — Mongolia
    (49.0, -123.0),   # Cap 5 — near Vancouver
    (70.0, 150.0),    # Cap 6 — northern Siberia
]


def build_model_space(min_degree: int = 64) -> Sobolev:
    """Build the Sobolev model space used by the sphere DLI example."""
    return Sobolev.from_heat_kernel_prior(
        PRIOR_SCALE,
        ORDER,
        SCALE,
        power_of_two=True,
        min_degree=min_degree,
    )


def _latlon_to_unit_xyz(centre_latlon: tuple[float, float]) -> np.ndarray:
    lat_rad, lon_rad = np.radians(centre_latlon)
    return np.array(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ]
    )


def _unit_xyz_to_latlon(points_xyz: np.ndarray) -> list[tuple[float, float]]:
    lat = np.degrees(np.arcsin(np.clip(points_xyz[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(points_xyz[:, 1], points_xyz[:, 0]))
    return list(zip(lat.tolist(), lon.tolist()))


def _sample_cap_points(
    centre_latlon: tuple[float, float],
    cap_radius_rad: float,
    n_cap: int,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Return points sampled approximately uniformly inside a spherical cap."""
    centre_xyz = _latlon_to_unit_xyz(centre_latlon)
    cos_cap = np.cos(cap_radius_rad)
    cap_points: list[list[float]] = []

    while len(cap_points) < n_cap:
        batch = rng.standard_normal((max(4 * n_cap, 8), 3))
        batch /= np.linalg.norm(batch, axis=1, keepdims=True)
        inside = batch[batch @ centre_xyz >= cos_cap]
        if inside.size:
            cap_points.extend(inside.tolist())

    return _unit_xyz_to_latlon(np.array(cap_points[:n_cap]))


def build_cap_property_operator(
    model_space: Sobolev,
    target_latlon_list: Iterable[tuple[float, float]] = DEFAULT_TARGET_LATLON,
    cap_radius_rad: float = 0.15,
    n_cap: int = 40,
    seed: int = 42,
) -> LinearOperator:
    """Build a spherical-cap averaging operator on the Sobolev model space.

    Each property is the empirical average of `n_cap` Dirac functionals sampled
    uniformly inside a spherical cap of radius `cap_radius_rad` centered on the
    corresponding target location.
    """
    rng = np.random.default_rng(seed)
    forms: list[LinearForm] = []
    target_points = list(target_latlon_list)

    for centre in target_points:
        sample_points = _sample_cap_points(centre, cap_radius_rad, n_cap, rng)
        components = np.mean(
            np.stack([model_space.dirac(point).components for point in sample_points]),
            axis=0,
        )
        forms.append(LinearForm(model_space, components=components))

    return LinearOperator.from_linear_forms(forms)


def build_forward_operator(
    model_space: Sobolev,
    *,
    n_sources: int = N_SOURCES,
    n_receivers: int = N_RECEIVERS,
    seed: int = 0,
) -> tuple[LinearOperator, list[tuple[tuple[float, float], tuple[float, float]]]]:
    """Build a normalized path-average forward operator on the sphere.

    Uses real IRIS station positions and randomly sampled USGS earthquakes.
    Each geodesic integral is divided by the arc length of the ray path so that
    the operator returns true path averages rather than raw integrals.

    Args:
        model_space: Sobolev space on the sphere.
        n_sources: Number of earthquake source points.
        n_receivers: Number of receiver stations.
        seed: Random seed for reproducible geometry selection.

    Returns:
        The normalized forward operator together with the source-receiver paths
        used to assemble it.
    """
    random_state = random.getstate()
    random.seed(seed)
    try:
        receivers = model_space.iris_stations(n_stations=n_receivers)
        sources = model_space.random_earthquakes(
            n_points=n_sources,
            min_magnitude=6.5,
        )
    finally:
        random.setstate(random_state)

    paths = [(source, receiver) for source in sources for receiver in receivers]

    forms: list[LinearForm] = []
    for point_1, point_2 in paths:
        raw_form = model_space.geodesic_integral(point_1, point_2)
        arc_length = model_space.geodesic_distance(point_1, point_2)
        if arc_length < 1e-10:
            normalized_components = raw_form.components.copy()
        else:
            normalized_components = raw_form.components / arc_length
        forms.append(LinearForm(model_space, components=normalized_components))

    return LinearOperator.from_linear_forms(forms), paths


def generate_synthetic_data(
    model_space: Sobolev,
    forward_operator: LinearOperator,
    *,
    sigma_noise: float = SIGMA_NOISE,
    seed: int = 42,
) -> tuple[object, np.ndarray]:
    r"""Sample a synthetic truth model and generate noisy observations.

    The truth is drawn from the heat-kernel Gaussian prior. Gaussian noise with
    standard deviation ``sigma_noise`` is added and clipped to ``\pm 3\sigma``
    so the resulting residual remains inside the ball confidence model used by
    later DLI phases.

    Args:
        model_space: Sobolev space on the sphere.
        forward_operator: Path-average operator built by ``build_forward_operator``.
        sigma_noise: Observation noise standard deviation.
        seed: RNG seed for reproducibility.

    Returns:
        The sampled truth model and a plain numpy data vector.
    """
    rng = np.random.default_rng(seed)
    prior = model_space.point_value_scaled_heat_kernel_gaussian_measure(PRIOR_SCALE)

    legacy_state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31)))
    try:
        truth_model = prior.sample()
    finally:
        np.random.set_state(legacy_state)

    noise_free_data = np.asarray(forward_operator(truth_model), dtype=float)
    noise = rng.normal(0.0, sigma_noise, size=noise_free_data.shape)
    noise = np.clip(noise, -3.0 * sigma_noise, 3.0 * sigma_noise)
    data_vector = noise_free_data + noise

    return truth_model, np.asarray(data_vector, dtype=float)


def solve_dli(
    model_space: Sobolev,
    forward_operator: LinearOperator,
    property_operator: LinearOperator,
    truth_model: object,
    data_vector: np.ndarray,
    *,
    sigma_noise: float = SIGMA_NOISE,
    prior_radius_multiplier: float = 3.0,
    max_iter: int = 300,
    tol: float = 1e-5,
) -> dict[str, np.ndarray]:
    """Solve deterministic linear inference bounds for the target properties.

    Builds zero-centered norm balls for the model prior set and the data-error
    set, then minimises the dual master cost for each property basis direction
    and its negation to obtain upper and lower admissible bounds.

    Args:
        model_space: Sobolev space on the sphere.
        forward_operator: Normalized path-average forward operator.
        property_operator: Spherical-cap property operator.
        truth_model: True model used to report the true property values.
        data_vector: Observed data in the forward-operator codomain.
        sigma_noise: Noise scale used to size the data-confidence ball.
        prior_radius_multiplier: Multiplier applied to ``||truth_model||`` to set
            the prior-ball radius.
        max_iter: Maximum proximal bundle iterations per support-value solve.
        tol: Bundle-method stopping tolerance.

    Returns:
        Dictionary with ``lower``, ``upper``, and ``true_values`` arrays, each of
        shape ``(property_operator.codomain.dim,)``.
    """
    from pygeoinf.backus_gilbert import DualMasterCostFunction
    from pygeoinf.convex_analysis import BallSupportFunction
    from pygeoinf.convex_optimisation import (
        ProximalBundleMethod,
        best_available_qp_solver,
        solve_support_values,
    )

    data_space = forward_operator.codomain
    property_space = property_operator.codomain

    prior_radius = prior_radius_multiplier * model_space.norm(truth_model)
    model_ball = BallSupportFunction(model_space, model_space.zero, prior_radius)

    data_ball_radius = 3.0 * sigma_noise * np.sqrt(data_space.dim)
    data_ball = BallSupportFunction(data_space, data_space.zero, data_ball_radius)

    basis_directions = [property_space.basis_vector(i) for i in range(property_space.dim)]
    negative_basis_directions = [property_space.multiply(-1.0, q) for q in basis_directions]

    observed_data = np.asarray(data_vector, dtype=float)
    cost = DualMasterCostFunction(
        data_space,
        property_space,
        model_space,
        forward_operator,
        property_operator,
        model_ball,
        data_ball,
        observed_data,
        basis_directions[0],
    )
    solver = ProximalBundleMethod(
        cost,
        tolerance=tol,
        max_iterations=max_iter,
        qp_solver=best_available_qp_solver(),
    )

    lambda0 = data_space.zero
    upper_values, _, _ = solve_support_values(
        cost, basis_directions, solver, lambda0, n_jobs=-1
    )
    lower_negated_values, _, _ = solve_support_values(
        cost,
        negative_basis_directions,
        solver,
        lambda0,
        n_jobs=-1,
    )

    lower = -np.asarray(lower_negated_values, dtype=float)
    upper = np.asarray(upper_values, dtype=float)
    true_values = np.asarray(property_operator(truth_model), dtype=float)

    # Prior bounds: support of model ball mapped through property operator
    prior_bounds = np.array(
        [model_ball(property_operator.adjoint(q)) for q in basis_directions]
    )

    return {
        "lower": lower,
        "upper": upper,
        "true_values": true_values,
        "prior_lower": -prior_bounds,
        "prior_upper": prior_bounds,
    }


def run_example(
    *,
    min_degree: int = 32,
    n_sources: int = N_SOURCES,
    n_receivers: int = N_RECEIVERS,
    n_target: int = N_TARGET,
    n_cap: int = 40,
    sigma_noise: float = SIGMA_NOISE,
    seed: int = 42,
) -> dict:
    """Run the full sphere-DLI example and return the assembled results.

    Args:
        min_degree: Minimum spherical-harmonic degree for the model space.
        n_sources: Number of earthquake source points.
        n_receivers: Number of receiver stations.
        n_target: Number of target spherical-cap properties.
        n_cap: Number of Dirac samples per spherical cap.
        sigma_noise: Data-noise standard deviation.
        seed: Global random seed used for geometry and data generation.

    Returns:
        Dictionary containing DLI bounds together with the model space, forward
        operator, property operator, truth model, data vector, ray paths, and
        selected target locations.
    """
    target_latlon = DEFAULT_TARGET_LATLON[:n_target]

    model_space = build_model_space(min_degree=min_degree)
    property_operator = build_cap_property_operator(
        model_space,
        target_latlon,
        n_cap=n_cap,
        seed=seed,
    )
    forward_operator, paths = build_forward_operator(
        model_space,
        n_sources=n_sources,
        n_receivers=n_receivers,
        seed=seed,
    )
    truth_model, data_vector = generate_synthetic_data(
        model_space,
        forward_operator,
        sigma_noise=sigma_noise,
        seed=seed,
    )
    bounds = solve_dli(
        model_space,
        forward_operator,
        property_operator,
        truth_model,
        data_vector,
        sigma_noise=sigma_noise,
    )

    return {
        **bounds,
        "model_space": model_space,
        "forward_operator": forward_operator,
        "property_operator": property_operator,
        "truth_model": truth_model,
        "data_vector": data_vector,
        "paths": paths,
        "target_latlon": target_latlon,
    }


def _plot_target_caps(
    ax,
    target_latlon: list[tuple[float, float]],
    cap_radius_rad: float = 0.15,
) -> None:
    """Overlay target spherical-cap regions on a Cartopy GeoAxes.

    Each cap is drawn as a circle of the correct angular radius using
    Cartopy's ``Geodetic`` CRS so the outline follows the sphere.
    """
    import cartopy.crs as ccrs
    import matplotlib.patches as mpatches

    cap_deg = np.degrees(cap_radius_rad)
    for index, (lat, lon) in enumerate(target_latlon):
        # Build a circle of geodetic points around the cap centre
        theta = np.linspace(0, 2 * np.pi, 361)
        # Angular offsets in lat/lon (small-angle; refined by ccrs.Geodetic)
        lats = lat + cap_deg * np.cos(theta)
        lons = lon + cap_deg / max(np.cos(np.radians(lat)), 1e-6) * np.sin(theta)
        ax.plot(
            lons,
            lats,
            transform=ccrs.PlateCarree(),
            color="tab:orange",
            linewidth=1.5,
            alpha=0.8,
        )
        ax.text(
            lon,
            lat,
            str(index + 1),
            transform=ccrs.PlateCarree(),
            fontsize=8,
            ha="center",
            va="center",
            color="tab:orange",
            fontweight="bold",
        )


def plot_results(result: dict) -> None:
    """Produce the Phase 4 example figures from a ``run_example`` result."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    from pygeoinf.symmetric_space.sphere import create_map_figure, plot, plot_geodesic_network

    paths = result["paths"]
    truth_model = result["truth_model"]
    target_latlon = result["target_latlon"]
    lower = np.asarray(result["lower"], dtype=float)
    upper = np.asarray(result["upper"], dtype=float)
    true_values = np.asarray(result["true_values"], dtype=float)
    prior_lower = np.asarray(result["prior_lower"], dtype=float)
    prior_upper = np.asarray(result["prior_upper"], dtype=float)

    # Create figures directory if needed
    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig1, ax1 = create_map_figure(figsize=(10, 5.5))
    ax1.set_global()
    ax1.coastlines(linewidth=0.8)
    plot_geodesic_network(
        paths,
        ax=ax1,
        alpha=0.08,
        linewidth=0.7,
        source_kwargs={"marker": "*", "color": "gold", "s": 140, "edgecolor": "black"},
        receiver_kwargs={"marker": "^", "color": "tab:blue", "s": 55, "edgecolor": "white"},
    )
    _plot_target_caps(ax1, target_latlon)
    fig1.suptitle("Ray network: stations, epicentres, great-circle paths, and target caps")
    fig1.savefig(fig_dir / "fig1_ray_network.png", dpi=150, bbox_inches="tight")

    fig2, ax2 = create_map_figure(figsize=(10, 5.5))
    ax2.set_global()
    plot(
        truth_model,
        ax=ax2,
        coasts=True,
        colorbar=True,
        symmetric=True,
        colorbar_kwargs={"label": "Phase-velocity perturbation (km/s)"},
    )
    _plot_target_caps(ax2, target_latlon)
    ax2.set_title("True phase-velocity perturbation field with target caps")
    fig2.savefig(fig_dir / "fig2_truth_model.png", dpi=150, bbox_inches="tight")

    n_properties = len(lower)
    fig3, ax3 = plt.subplots(figsize=(7, max(3.0, 0.7 * n_properties + 1.0)))
    y = np.arange(n_properties)
    ax3.barh(
        y,
        prior_upper - prior_lower,
        left=prior_lower,
        height=0.55,
        alpha=0.2,
        color="tab:grey",
        label="Prior bounds",
    )
    ax3.barh(
        y,
        upper - lower,
        left=lower,
        height=0.55,
        alpha=0.5,
        color="tab:blue",
        label="DLI admissible interval",
    )
    ax3.scatter(true_values, y, color="red", zorder=5, label="True value")
    ax3.set_yticks(y)
    ax3.set_yticklabels([f"Cap {index + 1}" for index in range(n_properties)])
    ax3.set_xlabel("Phase-velocity perturbation (km/s)")
    ax3.set_title("DLI admissible bounds")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(fig_dir / "fig3_dli_bounds.png", dpi=150, bbox_inches="tight")

    print(f"\nFigures saved to: {fig_dir}")
    plt.show()


if __name__ == "__main__":
    import os
    import matplotlib

    # Use non-interactive backend when running headless (e.g. on europa)
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")

    result = run_example()

    print("\nDLI Results:")
    for index, (lower, upper, true_value) in enumerate(
        zip(result["lower"], result["upper"], result["true_values"])
    ):
        print(f"  Cap {index + 1}: [{lower:.4f}, {upper:.4f}]  true={true_value:.4f}")

    plot_results(result)
