import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev, plot, plot_points


def setup_problem_components(
    base_space: Sobolev,
    degree: int,
    obs_points: np.ndarray,
    prior_scale: float,
    prior_std: float,
    noise_scale: float,
    noise_std: float,
    surrogate=False,
):
    """Helper function to generate spaces and measures for a specific degree."""
    space = base_space.with_degree(degree)
    forward_operator = space.point_evaluation_operator(obs_points)
    prior_measure = space.point_value_scaled_heat_kernel_gaussian_measure(
        prior_scale, std=prior_std
    )

    if noise_scale == 0.0:
        data_error_measure = inf.GaussianMeasure.from_standard_deviation(
            forward_operator.codomain, noise_std
        )
    else:
        spatial_noise_measure = space.point_value_scaled_heat_kernel_gaussian_measure(
            noise_scale, std=noise_std
        )
        data_error_measure = spatial_noise_measure.affine_mapping(
            operator=forward_operator
        )

        if surrogate:
            data_error_measure = data_error_measure.with_sparse_approximation()

    return space, forward_operator, prior_measure, data_error_measure


def main():
    # --- Hardcoded Parameters for Minimal Example ---
    n_data = 5000
    base_degree = 128
    surrogate_degree = 64
    sobolev_order = 2.0
    prior_scale = 0.05
    prior_std = 1.0
    noise_std = 0.3 * prior_std  # 30% noise amplitude

    print("Setting up base grid and exact problem...")
    base_space = Sobolev(base_degree, sobolev_order, prior_scale)
    observation_points = base_space.random_points(n_data)

    # 1. Exact Problem Setup
    model_space, forward_operator, model_prior_measure, data_error_measure = (
        setup_problem_components(
            base_space,
            base_degree,
            observation_points,
            prior_scale,
            prior_std,
            noise_scale=0.0,
            noise_std=noise_std,
        )
    )

    forward_problem = inf.LinearForwardProblem(
        forward_operator, data_error_measure=data_error_measure
    )

    # 2. Generate Synthetic True Model and Data
    true_model, data = forward_problem.synthetic_model_and_data(model_prior_measure)
    bayesian_inversion = inf.LinearBayesianInversion(
        forward_problem, model_prior_measure
    )

    # 3. Surrogate Problem Setup (for Preconditioning)
    print(f"Building spatial surrogate problem (degree={surrogate_degree})...")
    surrogate_space, surrogate_fwd_op, surrogate_prior, surrogate_noise = (
        setup_problem_components(
            base_space,
            surrogate_degree,
            observation_points,
            prior_scale,
            prior_std,
            noise_scale=0.0,
            noise_std=noise_std,
            surrogate=True,
        )
    )

    surrogate_inv = bayesian_inversion.surrogate_inversion(
        alternate_forward_operator=surrogate_fwd_op,
        alternate_prior_measure=surrogate_prior,
        alternate_data_error_measure=surrogate_noise,
    )

    # 4. Woodbury Preconditioner Setup
    print("Forming model-space Woodbury preconditioner...")
    woodbury_solver = inf.EigenSolver(galerkin=True)
    preconditioner = surrogate_inv.woodbury_data_preconditioner(woodbury_solver)

    # Blending the preconditioner
    alpha = 0.1
    # preconditioner = (
    #    1 - alpha
    # ) * preconditioner + alpha * surrogate_noise.inverse_covariance

    # ================================================================#

    I = bayesian_inversion.data_space.identity_operator()
    N = bayesian_inversion.normal_operator
    E = I - preconditioner @ N

    data_space = forward_operator.codomain

    M_vectors = 30  # Number of Hutchinson random vectors
    stochastic_logdet_sum = 0.0

    print(f"Running stochastic trace estimator with {M_vectors} vectors...")

    for m in range(M_vectors):
        # 1. Draw a random vector
        z = data_space.random()

        # 2. Initialize accumulators
        y = z * 0.0  # Creates a zero vector in the same space
        v = z

        k = 1
        err = 1.0

        # 3. The Maclaurin Expansion Loop
        while err > 1e-3:

            w = E(v)

            # Accumulate the next term in the series
            y = y + (w / k)

            # Update for the next iteration
            # v = data_space.copy(w)
            v = w

            # Check convergence
            err = data_space.norm(w / k)
            print(k, err)
            k += 1

            # Safety break just in case the preconditioner is struggling
            if k > 50:
                print(
                    f"Vector {m} hit max iterations. Spectral radius of E might be near 1."
                )
                break

        print("-" * 20)

        # 4. Compute the trace contribution for this vector
        # Note: the Maclaurin series for ln(I-E) has a negative sign
        trace_contribution = -data_space.inner_product(z, y)
        stochastic_logdet_sum += trace_contribution

    # 5. Average over all M vectors
    estimated_logdet_residual = stochastic_logdet_sum / M_vectors

    print(f"Estimated ln(det(P^-1 * Gamma)): {estimated_logdet_residual}")

    """


    # 5. Solve Inverse Problem
    print("Solving linear system with CG...")
    solver = inf.CGSolver(rtol=0.01 * 0.3)  # Based on 30% noise amplitude
    model_posterior_measure = bayesian_inversion.model_posterior_measure(
        data, solver, preconditioner=preconditioner
    )
    posterior_expectation = model_posterior_measure.expectation
    print(f"Number of CG iterations = {solver.iterations}")

    # 6. Plot True Model and Posterior Expectation
    print("Plotting results...")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        subplot_kw={"projection": ccrs.Robinson()},
        layout="constrained",
    )

    shared_vmax = 1.2 * max(np.nanmax(np.abs(true_model.data)), np.nanmax(np.abs(data)))

    # --- Plot 1: True Model with Observation Locations ---
    plot(
        true_model,
        ax=axes[0],
        coasts=True,
        cmap="seismic",
        vmin=-shared_vmax,
        vmax=shared_vmax,
        colorbar=True,
        colorbar_kwargs={
            "label": "True Model",
            "orientation": "horizontal",
            "shrink": 0.8,
        },
    )
    plot_points(observation_points, ax=axes[0], color="black", s=15, zorder=5)
    axes[0].set_title("True Model & Obs Locations", fontsize=14, fontweight="bold")

    # --- Plot 2: Posterior Expectation with Observation Locations ---
    plot(
        posterior_expectation,
        ax=axes[1],
        coasts=True,
        cmap="seismic",
        vmin=-shared_vmax,
        vmax=shared_vmax,
        colorbar=True,
        colorbar_kwargs={
            "label": "Posterior Expectation",
            "orientation": "horizontal",
            "shrink": 0.8,
        },
    )
    plot_points(observation_points, ax=axes[1], color="black", s=15, zorder=5)
    axes[1].set_title(
        "Posterior Expectation & Obs Locations", fontsize=14, fontweight="bold"
    )

    plt.show()

"""


if __name__ == "__main__":
    main()
