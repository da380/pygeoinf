import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf

# ==================================================================== #
#                          CONFIGURATION                               #
# ==================================================================== #

# Choose from: "Sphere", "Torus", "Plane", "Circle", "Line"
GEOMETRY = "Plane"
DISK_MODEL = False  # True for a top-hat anomaly, False for random field

order = 2
scale = 0.2
prior_scale = 0.1
n_sources = 3
n_receivers = 30

# ==================================================================== #
#                       GEOMETRY INITIALIZATION                        #
# ==================================================================== #

if GEOMETRY == "Torus":
    from pygeoinf.symmetric_space.torus import Sobolev, plot, plot_geodesic_network

    model_space = Sobolev.from_heat_kernel_prior(
        prior_scale,
        order,
        scale,
        radius_x=1.0,
        radius_y=1.0,
        power_of_two=True,
        min_degree=32,
    )

elif GEOMETRY == "Plane":
    from pygeoinf.symmetric_space.plane import Sobolev, plot, plot_geodesic_network

    model_space = Sobolev.from_heat_kernel_prior(
        prior_scale,
        order,
        scale,
        ax=0.0,
        bx=6.0,
        cx=0.5,
        ay=0.0,
        by=6.0,
        cy=0.5,
        power_of_two=True,
        min_degree=32,
    )

elif GEOMETRY == "Sphere":
    from pygeoinf.symmetric_space.sphere import (
        Sobolev,
        plot,
        plot_geodesic_network,
        create_map_figure,
    )

    model_space = Sobolev.from_heat_kernel_prior(
        prior_scale, order, scale, radius=1.0, power_of_two=True, min_degree=32
    )

elif GEOMETRY == "Circle":
    from pygeoinf.symmetric_space.circle import Sobolev, plot

    model_space = Sobolev.from_heat_kernel_prior(
        prior_scale, order, scale, radius=1.0, power_of_two=True, min_degree=32
    )
    plot_geodesic_network = None  # 1D doesn't use the network plotter

elif GEOMETRY == "Line":
    from pygeoinf.symmetric_space.line import Sobolev, plot

    model_space = Sobolev.from_heat_kernel_prior(
        prior_scale, order, scale, a=0.0, b=10.0, power_of_two=True, min_degree=32
    )
    plot_geodesic_network = None

else:
    raise ValueError(f"Unknown geometry: {GEOMETRY}")

print(f"\n--- Running Tomography on a {GEOMETRY} ---")
print(f"Model space dimension = {model_space.dim}")

# ==================================================================== #
#                      AGNOSTIC INVERSION PIPELINE                     #
# ==================================================================== #

# Coordinates are floats for 1D, tuples for 2D. random_point() handles this.
receivers = [model_space.random_point() for _ in range(n_receivers)]
sources = [model_space.random_point() for _ in range(n_sources)]
paths = [(src, rec) for src in sources for rec in receivers]

forward_operator = model_space.path_average_operator(paths)
data_error_measure = inf.GaussianMeasure.from_standard_deviation(
    forward_operator.codomain, 0.05
)
forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)
model_prior = model_space.point_value_scaled_heat_kernel_gaussian_measure(prior_scale)

# --- Generate Synthetic Data ---
if DISK_MODEL:
    center = model_space.random_point()
    # Disk radius in 2D, or interval half-width in 1D
    radius = 1.0

    def top_hat_anomaly(p) -> float:
        return 1.0 if model_space.geodesic_distance(center, p) <= radius else 0.0

    model = model_space.invariant_automorphism(model_space.heat_kernel(0.15))(
        model_space.project_function(top_hat_anomaly)
    )
    data = forward_problem.data_measure_from_model(model).sample()
else:
    model, data = forward_problem.joint_measure(model_prior).sample()

# --- Solve the Inverse Problem ---
inverse_problem = inf.LinearBayesianInversion(forward_problem, model_prior)


surrogate_space = model_space.with_degree(model_space.degree // 4)
precon = inverse_problem.surrogate_inversion(
    alternate_forward_operator=surrogate_space.path_average_operator(paths),
    alternate_prior_measure=surrogate_space.point_value_scaled_heat_kernel_gaussian_measure(
        prior_scale
    ),
).woodbury_data_preconditioner()

solver = inf.CGMatrixSolver()
model_posterior = inverse_problem.model_posterior_measure(
    data, solver, preconditioner=precon
)
print(f"Solution found in {solver.iterations} iterations")

posterior_std = model_posterior.sample_pointwise_std(100)

# ==================================================================== #
#                         PLOTTING WRAPPER                             #
# ==================================================================== #


def plot_unified(field, title, label, cmap="RdBu_r", symmetric=False):
    """Wraps API differences: 1D (Line/Circle) vs 2D (Torus/Plane) vs Sphere."""
    if GEOMETRY == "Sphere":
        fig, ax = create_map_figure(figsize=(8, 5))
    else:
        fig, ax = plt.subplots(figsize=(8, 5))

    if GEOMETRY == "Sphere":
        # Sphere plot uses pyshtools grids directly
        plot(field, ax=ax, colorbar=True, symmetric=symmetric, cmap=cmap)
        if plot_geodesic_network:
            plot_geodesic_network(paths, ax=ax, alpha=0.1, color="black")

    elif GEOMETRY == "Circle":
        # 1D Circle: Native 1D plot
        plot(model_space, field, ax=ax, color="blue", label=label)
        ax.set_xlabel("Angle (radians)")
        ax.set_xlim(0, 2 * np.pi)

    elif GEOMETRY == "Line":
        # 1D Line: Includes the 'full' flag to show/hide padding
        # Set full=True here if you want to see the Fourier Continuation tapering
        plot(model_space, field, ax=ax, color="green", label=label, full=False)
        ax.set_xlabel("Coordinate (x)")

    else:
        # 2D Torus/Plane
        plot(model_space, field, ax=ax, colorbar=True, symmetric=symmetric, cmap=cmap)
        if plot_geodesic_network:
            plot_geodesic_network(paths, ax=ax, alpha=0.1, color="black")

    ax.set_title(title)
    return ax


# --- Execute Main Plots ---
plot_unified(model, f"True Model ({GEOMETRY})", "Truth", symmetric=True)
plot_unified(
    model_posterior.expectation, "Posterior Expectation", "Mean", symmetric=True
)

# ==================================================================== #
#                    SPECIFIC ERROR BOUND PLOTTING                     #
# ==================================================================== #

if GEOMETRY == "Circle":
    from pygeoinf.symmetric_space.circle import plot_error_bounds

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_error_bounds(
        model_space,
        model_posterior.expectation,
        posterior_std,
        ax=ax,
        alpha=0.3,
        color="blue",
    )
    plot(model_space, model_posterior.expectation, ax=ax, color="black", label="Mean")
    ax.set_title("Circle Posterior with Error Bounds")
    ax.set_xlim(0, 2 * np.pi)
    ax.legend()

elif GEOMETRY == "Line":
    from pygeoinf.symmetric_space.line import plot_error_bounds

    fig, ax = plt.subplots(figsize=(8, 5))
    # 'full=False' crops the uncertainty plot to the physical [a, b] domain
    plot_error_bounds(
        model_space,
        model_posterior.expectation,
        posterior_std,
        ax=ax,
        alpha=0.3,
        color="green",
        full=False,
    )
    plot(
        model_space,
        model_posterior.expectation,
        ax=ax,
        color="black",
        label="Mean",
        full=False,
    )
    ax.set_title("Line Posterior with Error Bounds")
    ax.legend()

else:
    # Standard 2D Uncertainty Map
    plot_unified(posterior_std, "Posterior STD", "Standard Deviation", cmap="viridis")

plt.show()
