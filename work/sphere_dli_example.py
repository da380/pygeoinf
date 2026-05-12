r"""Sphere DLI Example.

Relative phase-velocity perturbation ($\delta \ln c$) DLI on the two-sphere using:
- Sobolev model space (order 1.5, pyshtools spherical-harmonic backend)
- Real IRIS GSN stations and randomly sampled USGS earthquakes
- A smooth synthetic reference phase-velocity field $c_0(x)$
- Great-circle path integrals weighted by $1 / c_0(x)$
    (or arc-length-normalised weighted path averages)
- Spherical-cap averaging target properties
- Norm-ball support functions for the model prior and data-confidence set
- PrimalKKTSolver for the DLI solve

Entry points:
  run_example(...)  — builds and solves the full problem, returns a result dict.
  plot_results(...) — produces three diagnostic figures from the result dict.

Forward map note:
    The physically motivated linearised phase-delay/travel-time model is a
    reference-velocity-weighted path integral along each ray, i.e. each row is
    of the form

        d_i = \int_{\gamma_i} m(x) / c_0(x) ds

    where ``m`` is the perturbation field, ``c_0`` is the reference phase
    velocity, and ``\gamma_i`` is a geodesic path.

    For numerical comparability across different path lengths, this script can
    alternatively use arc-length-normalised rows

        d_i = (1 / L_i) \int_{\gamma_i} m(x) / c_0(x) ds,

    where ``L_i`` is path length. This rescales each row only; it does not
    change the underlying geometric sensitivity pattern along each path.

Run as a script::

    conda activate inferences3
    cd pygeoinf
    python work/sphere_dli_example.py
"""

from __future__ import annotations

import random
from typing import Callable, Iterable, Literal

import numpy as np

from pygeoinf.linear_forms import LinearForm
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.symmetric_space.sphere import Sobolev


ORDER = 1.5
SCALE = 0.1
PRIOR_SCALE = 0.05
REFERENCE_VELOCITY_BASE = 1.0
REFERENCE_VELOCITY_LATITUDE_PERTURBATION = 0.12
REFERENCE_VELOCITY_LONGITUDE_PERTURBATION = 0.08
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

CAP_OPERATOR_METHODS = ("quadrature", "exact", "monte_carlo")
CapOperatorMethod = Literal["quadrature", "exact", "monte_carlo"]


def build_model_space(min_degree: int = 64) -> Sobolev:
    """Build the Sobolev model space used by the sphere DLI example."""
    return Sobolev.from_heat_kernel_prior(
        PRIOR_SCALE,
        ORDER,
        SCALE,
        power_of_two=True,
        min_degree=min_degree,
    )


def reference_phase_velocity(point: tuple[float, float]) -> float:
    """Return a smooth positive synthetic reference phase-velocity field.

    The example uses a dimensionless background field with mild large-scale
    latitudinal and longitudinal variation so the linearized travel-time
    operator can apply the required ``1 / c_0`` weighting along each ray.
    """
    lat, lon = point
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    return (
        REFERENCE_VELOCITY_BASE
        + REFERENCE_VELOCITY_LATITUDE_PERTURBATION * np.sin(lat_rad) ** 2
        + REFERENCE_VELOCITY_LONGITUDE_PERTURBATION
        * np.cos(lat_rad) ** 2
        * np.cos(2.0 * lon_rad)
    )


def _weighted_geodesic_integral_form(
    model_space: Sobolev,
    point_1: tuple[float, float],
    point_2: tuple[float, float],
    *,
    reference_velocity: Callable[[tuple[float, float]], float],
    normalize_by_arclength: bool,
) -> LinearForm:
    """Build a geodesic linear form weighted by the inverse reference speed."""
    _, trial_weights = model_space.geodesic_quadrature(point_1, point_2, n_points=2)
    trial_arc_length = float(np.sum(trial_weights))
    n_points = max(2, int(np.ceil((trial_arc_length / model_space.scale) * 2.0)))

    points, weights = model_space.geodesic_quadrature(point_1, point_2, n_points=n_points)
    arc_length = float(np.sum(weights))

    components = np.zeros(model_space.dim)
    for point, weight in zip(points, weights):
        ref_velocity = float(reference_velocity(point))
        if ref_velocity <= 0.0:
            raise ValueError("Reference velocity must stay strictly positive along every ray.")
        components += (weight / ref_velocity) * model_space.dirac(point).components

    if normalize_by_arclength and arc_length >= 1e-10:
        components /= arc_length

    return LinearForm(model_space, components=components)


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
    method: CapOperatorMethod = "exact",
    exact: bool | None = None,
) -> LinearOperator:
    """Build a spherical-cap averaging operator on the Sobolev model space.

    By default, each property is assembled from the sphere's deterministic
    geodesic-ball quadrature rule using ``n_cap`` points. Set ``method="exact"``
    to use the analytical spherical-harmonic cap average, or
    ``method="monte_carlo"`` to retain the empirical random cap average.

    The legacy ``exact`` flag is still accepted for compatibility:
    ``exact=True`` selects ``method="exact"`` and ``exact=False`` selects
    ``method="monte_carlo"``.
    """
    resolved_method = method
    if exact is not None:
        legacy_method: CapOperatorMethod = "exact" if exact else "monte_carlo"
        if method != "quadrature" and method != legacy_method:
            raise ValueError("Conflicting target-operator method and exact flag.")
        resolved_method = legacy_method
    if resolved_method not in CAP_OPERATOR_METHODS:
        raise ValueError(
            "method must be one of 'quadrature', 'exact', or 'monte_carlo'."
        )

    rng = np.random.default_rng(seed)
    forms: list[LinearForm] = []
    target_points = list(target_latlon_list)
    cap_radius = model_space.radius * cap_radius_rad

    for centre in target_points:
        if resolved_method == "quadrature":
            forms.append(
                model_space.geodesic_ball_average(
                    centre,
                    cap_radius,
                    n_points=n_cap,
                )
            )
        elif resolved_method == "exact":
            forms.append(
                model_space.geodesic_ball_average(
                    centre,
                    cap_radius,
                )
            )
        else:
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
    normalize_by_arclength: bool = True,
    reference_velocity: Callable[[tuple[float, float]], float] = reference_phase_velocity,
) -> tuple[LinearOperator, list[tuple[tuple[float, float], tuple[float, float]]]]:
    """Build a reference-weighted path-integral forward operator on the sphere.

    Uses real IRIS station positions and randomly sampled USGS earthquakes.
    Each row integrates the perturbation field against ``1 / c_0`` along a
    geodesic ray. If ``normalize_by_arclength`` is true (default), each row is
    divided by path length to return weighted path averages instead of raw
    integrals.

    Args:
        model_space: Sobolev space on the sphere.
        n_sources: Number of earthquake source points.
        n_receivers: Number of receiver stations.
        seed: Random seed for reproducible geometry selection.
        normalize_by_arclength: If true, use
            ``(1 / L_i) * integral_{gamma_i} m / c_0 ds`` per path. If false,
            use raw ``integral_{gamma_i} m / c_0 ds`` rows.
        reference_velocity: Smooth positive reference phase velocity ``c_0``
            sampled along each geodesic.

    Returns:
        The assembled forward operator together with the source-receiver paths
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
        forms.append(
            _weighted_geodesic_integral_form(
                model_space,
                point_1,
                point_2,
                reference_velocity=reference_velocity,
                normalize_by_arclength=normalize_by_arclength,
            )
        )

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
        forward_operator: Reference-weighted path-average operator built by
            ``build_forward_operator``.
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
    set, then solves primal KKT systems in a basis-free way for each property
    basis direction and its negation to obtain upper and lower admissible
    bounds.

    Args:
        model_space: Sobolev space on the sphere.
        forward_operator: Reference-weighted normalized path-average forward
            operator.
        property_operator: Spherical-cap property operator.
        truth_model: True model used to report the true property values.
        data_vector: Observed data in the forward-operator codomain.
        sigma_noise: Noise scale used to size the data-confidence ball.
        prior_radius_multiplier: Multiplier applied to ``||truth_model||`` to set
            the prior-ball radius.
        max_iter: Unused compatibility argument retained for API stability.
        tol: Unused compatibility argument retained for API stability.

    Returns:
        Dictionary with ``lower``, ``upper``, and ``true_values`` arrays, each of
        shape ``(property_operator.codomain.dim,)``.
    """
    from pygeoinf.convex_analysis import BallSupportFunction
    from pygeoinf.convex_optimisation import PrimalKKTSolver

    data_space = forward_operator.codomain
    property_space = property_operator.codomain

    prior_radius = prior_radius_multiplier * model_space.norm(truth_model)
    model_ball = BallSupportFunction(model_space, model_space.zero, prior_radius)

    data_ball_radius = 3.0 * sigma_noise * np.sqrt(data_space.dim)
    data_ball = BallSupportFunction(data_space, data_space.zero, data_ball_radius)

    basis_directions = [property_space.basis_vector(i) for i in range(property_space.dim)]
    negative_basis_directions = [property_space.multiply(-1.0, q) for q in basis_directions]

    observed_data = np.asarray(data_vector, dtype=float)
    d_tilde = data_space.from_components(observed_data)
    kkt_solver = PrimalKKTSolver(
        model_ball,
        data_ball,
        forward_operator,
        d_tilde,
    )

    def _solve_directional_values(q_list: list[np.ndarray]) -> np.ndarray:
        vals = []
        for q in q_list:
            c = property_operator.adjoint(q)
            result = kkt_solver.solve(c)
            vals.append(model_space.inner_product(c, result.m))
        return np.asarray(vals, dtype=float)

    upper_values = _solve_directional_values(basis_directions)
    lower_negated_values = _solve_directional_values(negative_basis_directions)

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
    normalize_by_arclength: bool = True,
    target_operator_method: CapOperatorMethod = "exact",
    exact_cap_average: bool | None = None,
) -> dict:
    """Run the full sphere-DLI example and return the assembled results.

    Args:
        min_degree: Minimum spherical-harmonic degree for the model space.
        n_sources: Number of earthquake source points.
        n_receivers: Number of receiver stations.
        n_target: Number of target spherical-cap properties.
        n_cap: Number of cap quadrature points, or Monte Carlo samples when
            ``target_operator_method='monte_carlo'``.
        sigma_noise: Data-noise standard deviation.
        seed: Global random seed used for geometry and data generation.
        normalize_by_arclength: Controls whether forward rows are divided by
            path length. True gives reference-weighted path averages; False
            gives raw reference-weighted path integrals.
        target_operator_method: Target-cap assembly method. ``'quadrature'``
            uses the deterministic spherical-cap quadrature rule, ``'exact'``
            uses the analytical spherical-harmonic cap average, and
            ``'monte_carlo'`` uses the retained empirical random sampler.
        exact_cap_average: Legacy compatibility flag. ``True`` selects
            ``target_operator_method='exact'`` and ``False`` selects
            ``target_operator_method='monte_carlo'``.

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
        method=target_operator_method,
        exact=exact_cap_average,
    )
    forward_operator, paths = build_forward_operator(
        model_space,
        n_sources=n_sources,
        n_receivers=n_receivers,
        seed=seed,
        normalize_by_arclength=normalize_by_arclength,
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
        "normalize_by_arclength": normalize_by_arclength,
        "target_operator_method": (
            ("exact" if exact_cap_average else "monte_carlo")
            if exact_cap_average is not None
            else target_operator_method
        ),
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
            linewidth=0.8,
            alpha=0.8,
            antialiased=True,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        ax.text(
            lon,
            lat,
            str(index + 1),
            transform=ccrs.PlateCarree(),
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

    # ------------------------------------------------------------------
    # Two-column paper style
    # Full text width ~6.9 in, single column ~3.35 in.
    # Font sizes (8 pt labels, 7 pt ticks) match ~10 pt body text at
    # typical journal print size.
    # ------------------------------------------------------------------
    TWO_COL_W = 6.9   # inches
    ONE_COL_W = 3.35  # inches
    _PAPER_RC = {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 0.8,
    }

    with plt.rc_context(_PAPER_RC):
        fig1, ax1 = create_map_figure(figsize=(TWO_COL_W, 3.8))
        ax1.set_global()
        ax1.coastlines(linewidth=0.5)
        plot_geodesic_network(
            paths,
            ax=ax1,
            alpha=0.08,
            linewidth=0.5,
            antialiased=True,
            solid_capstyle="round",
            solid_joinstyle="round",
            source_kwargs={"marker": "*", "color": "gold", "s": 25, "edgecolor": "black"},
            receiver_kwargs={"marker": "^", "color": "tab:blue", "s": 15, "edgecolor": "white"},
        )
        _plot_target_caps(ax1, target_latlon)
        fig1.suptitle("Ray network: stations, epicentres, great-circle paths, and target caps")
        fig1.savefig(fig_dir / "fig1_ray_network.png", dpi=300, bbox_inches="tight")

        fig2, ax2 = create_map_figure(figsize=(TWO_COL_W, 3.8))
        ax2.set_global()
        plot(
            truth_model,
            ax=ax2,
            coasts=True,
            colorbar=True,
            symmetric=True,
            colorbar_kwargs={"label": "Relative phase-velocity perturbation ($\\delta \\ln c$)"},
        )
        _plot_target_caps(ax2, target_latlon)
        ax2.set_title("True relative phase-velocity perturbation field ($\\delta \\ln c$)")
        fig2.savefig(fig_dir / "fig2_truth_model.png", dpi=300, bbox_inches="tight")

        n_properties = len(lower)
        fig3, ax3 = plt.subplots(
            figsize=(ONE_COL_W, max(2.5, 0.45 * n_properties + 0.9))
        )
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
        ax3.scatter(true_values, y, color="red", s=15, zorder=5, label="True value")
        ax3.set_yticks(y)
        ax3.set_yticklabels([f"Cap {index + 1}" for index in range(n_properties)])
        ax3.set_xlabel("Relative phase-velocity perturbation ($\\delta \\ln c$)")
        ax3.set_title("DLI admissible bounds")
        ax3.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            frameon=True,
        )
        fig3.tight_layout()
        fig3.savefig(fig_dir / "fig3_dli_bounds.png", dpi=300, bbox_inches="tight")

    print(f"\nFigures saved to: {fig_dir}")
    plt.show()


if __name__ == "__main__":
    import os
    import matplotlib

    # Use non-interactive backend when running headless (e.g. on europa)
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")

    result = run_example()

    print("\nDLI Results (relative perturbation, $\\delta \\ln c$):")
    for index, (lower, upper, true_value) in enumerate(
        zip(result["lower"], result["upper"], result["true_values"])
    ):
        print(f"  Cap {index + 1}: [{lower:.4f}, {upper:.4f}]  true={true_value:.4f}")

    plot_results(result)
