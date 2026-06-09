import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev, plot

# --- Physical constants ---
g = 9.8
G = 6.6743e-11
H = 2e4
b = 6.371e6
E = 9e10
nu = 0.25
rho_l = 3000
rho_m = 3400

D_ocean = E * H**3 / (12 * (1 - nu**2))

# --- Truncation & Prior parameters ---
lmax_full = 128
lmax_surrogate = 32
lmax_geoid = 64
lmax_topo = 96

space_order, space_scale = 2.0, 0.1 * b
traction_order, traction_scale, traction_std = 1.25, 100.0e3, 2.0e6
density_order, density_scale, density_std = 2.0, 500.0e3, 10.0


def build_operators(field_space: Sobolev, rigidity_field):
    """
    Builds the physics operators and discrete observation operators.
    """
    # 1. Step 1: [density, traction] -> [density, flexure]
    traction_operator = inf.RowLinearOperator(
        [-g * H * field_space.identity_operator(), field_space.identity_operator()]
    )

    flexure_solver = (
        None
        if isinstance(rigidity_field, float)
        else inf.CGSolver(
            callback=inf.ProgressCallback(message="   Inner Flexure CG: ")
        )
    )

    flexure_operator = field_space.inverse_flexural_operator(
        rigidity_field,
        nu,
        (rho_m - rho_l) * g,
        baseline_rigidity=D_ocean,
        solver=flexure_solver,
    )

    step1_operator = inf.ColumnLinearOperator(
        [
            inf.RowLinearOperator(
                [field_space.identity_operator(), field_space.zero_operator()]
            ),
            flexure_operator @ traction_operator,
        ]
    )

    # 2. Step 2: [density, flexure] -> [potential, flexure]
    density_to_pot = (
        inf.symmetric_space.InvariantLinearAutomorphism.from_index_function(
            field_space, lambda k: -4 * np.pi * G * b * H / (2 * k[0] + 1)
        )
    )
    flexure_to_pot = (
        inf.symmetric_space.InvariantLinearAutomorphism.from_index_function(
            field_space,
            lambda k: -4
            * np.pi
            * G
            * b
            * rho_l
            * (1 + (rho_m / rho_l - 1) * (1 - H / b) ** (k[0] + 2))
            / (2 * k[0] + 1),
        )
    )

    step2_operator = inf.BlockLinearOperator(
        [
            [density_to_pot, flexure_to_pot],
            [field_space.zero_operator(), field_space.identity_operator()],
        ]
    )

    physics_operator = step2_operator @ step1_operator

    # 3. Discrete Observation Operator
    geoid_obs_op = (-1.0 / g) * field_space.to_coefficient_operator(lmax_geoid)
    topo_obs_op = field_space.to_coefficient_operator(lmax_topo)
    observation_operator = inf.BlockDiagonalLinearOperator([geoid_obs_op, topo_obs_op])

    forward_operator = observation_operator @ physics_operator

    return physics_operator, forward_operator


def build_prior(field_space: Sobolev):
    """
    Builds the model prior measure.
    """
    density_prior = field_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        density_order, density_scale, std=density_std
    )
    traction_prior = field_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        traction_order, traction_scale, std=traction_std
    )
    return inf.GaussianMeasure.from_direct_sum([density_prior, traction_prior])


def build_noise_measure(
    field_space: Sobolev,
    model_prior: inf.GaussianMeasure,
    SNR: float = 20.0,
    noise_order: float = 1.0,
    noise_scale: float = 0.05 * b,
):
    """
    Builds the degree-dependent diagonal noise measures directly from the
    eigenvalues (spectral variances) of the invariant spatial noise measures.
    """
    # 1. Invariant Approximation for Signal Amplitude
    physics_inv, forward_inv = build_operators(field_space, D_ocean)

    clean_fields_measure = model_prior.affine_mapping(operator=physics_inv)

    geoid_proj = inf.RowLinearOperator(
        [(-1.0 / g) * field_space.identity_operator(), field_space.zero_operator()]
    )
    topo_proj = inf.RowLinearOperator(
        [field_space.zero_operator(), field_space.identity_operator()]
    )

    geoid_measure = clean_fields_measure.affine_mapping(operator=geoid_proj)
    topo_measure = clean_fields_measure.affine_mapping(operator=topo_proj)

    # Evaluate exact pointwise standard deviation
    rand_pt = field_space.random_point()
    dirac_rep = field_space.dirac_representation(rand_pt)

    var_geoid = geoid_measure.directional_variance(dirac_rep)
    var_topo = topo_measure.directional_variance(dirac_rep)

    target_geoid_std = np.sqrt(var_geoid) / SNR
    target_topo_std = np.sqrt(var_topo) / SNR

    print(
        f"   -> Expected Signal Pointwise STD | Geoid: {np.sqrt(var_geoid):.2e} m, Topo: {np.sqrt(var_topo):.2e} m"
    )
    print(
        f"   -> Target Noise Pointwise STD    | Geoid: {target_geoid_std:.2e} m, Topo: {target_topo_std:.2e} m"
    )

    # 2. Create Rough Spatial Noise Fields (Invariant Colored Noise)
    raw_geoid_spatial_noise = (
        field_space.point_value_scaled_sobolev_kernel_gaussian_measure(
            noise_order, noise_scale, std=target_geoid_std
        )
    )
    raw_topo_spatial_noise = (
        field_space.point_value_scaled_sobolev_kernel_gaussian_measure(
            noise_order, noise_scale, std=target_topo_std
        )
    )

    # 3. Extract exact coefficient variances natively (Instantaneous O(N))
    # Coefficient Variance = Spectral Variance / Squared Norm
    geoid_coeff_vars = (
        raw_geoid_spatial_noise.spectral_variances / field_space.metric_values
    )
    topo_coeff_vars = (
        raw_topo_spatial_noise.spectral_variances / field_space.metric_values
    )

    # Truncate to the data space dimensions
    geoid_size = (lmax_geoid + 1) ** 2
    topo_size = (lmax_topo + 1) ** 2

    geoid_vars = geoid_coeff_vars[:geoid_size]
    topo_vars = topo_coeff_vars[:topo_size]

    # 4. Form the exact diagonal data noise measure
    data_spaces = forward_inv.codomain.subspaces
    geoid_error_measure = inf.GaussianMeasure.from_standard_deviations(
        data_spaces[0], np.sqrt(np.maximum(geoid_vars, 0.0))
    )
    topo_error_measure = inf.GaussianMeasure.from_standard_deviations(
        data_spaces[1], np.sqrt(np.maximum(topo_vars, 0.0))
    )

    return inf.GaussianMeasure.from_direct_sum(
        [geoid_error_measure, topo_error_measure]
    )


# =========================================================
# Main Execution
# =========================================================

print("1. Initializing Spaces and Rigidity Field...")
full_space = Sobolev(lmax_full, space_order, space_scale, radius=b)
surrogate_space = Sobolev(lmax_surrogate, space_order, space_scale, radius=b)

D_base = full_space.project_function(lambda _: D_ocean)
D_raw = D_base * (1.0 + 10.0 * full_space.domain_mask())
smoothing_op = full_space.heat_kernel_gaussian_measure(0.05 * b).covariance
D_field_full = smoothing_op(D_raw)
# D_field_full = D_ocean

print("2. Building Full System Operators & Measures (Variable Rigidity)...")
physics_op, forward_op = build_operators(full_space, D_field_full)
prior = build_prior(full_space)
data_error_measure = build_noise_measure(
    full_space, prior, SNR=20.0, noise_order=1.0, noise_scale=0.05 * b
)

print("\n3. Building Surrogate System Operators & Measures (Uniform Rigidity)...")
_, surrogate_forward_op = build_operators(surrogate_space, D_ocean)
surrogate_prior = build_prior(surrogate_space)


print("\n4. Generating Synthetic Truth and Colored Noise Data...")
forward_problem = inf.LinearForwardProblem(
    forward_op, data_error_measure=data_error_measure
)
model_true, data_obs = forward_problem.joint_measure(prior).sample()

print("5. Building Fast Woodbury Preconditioner...")
woodbury_solver = inf.CholeskySolver(galerkin=True)
damped_prior_s = surrogate_prior.with_regularized_inverse(woodbury_solver, damping=1e-6)

inverse_problem = inf.LinearBayesianInversion(
    forward_problem, prior, formalism="data_space"
)

precon = inverse_problem.surrogate_inversion(
    alternate_forward_operator=surrogate_forward_op,
    alternate_prior_measure=damped_prior_s,
).woodbury_data_preconditioner(woodbury_solver)

print("6. Solving Bayesian Inversion...")

solver = inf.CGSolver(callback=inf.ProgressCallback(message="   CG: "))

model_posterior = inverse_problem.model_posterior_measure(
    data_obs, solver, preconditioner=precon
)

print(f"\nInversion complete in {solver.iterations} iterations.")

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
print("7. Rendering Results...")
density_true, traction_true = model_true
density_post, traction_post = model_posterior.expectation

potential_true, flexure_true = physics_op(model_true)
potential_post, flexure_post = physics_op(model_posterior.expectation)

vmax_rho = np.max(np.abs(density_true.data))
vmax_trac = np.max(np.abs(traction_true.data))
vmax_geoid = np.max(np.abs(potential_true.data / g))
vmax_flex = np.max(np.abs(flexure_true.data))

# ==========================================
# Figure 1: Model Recovery (Density & Traction)
# ==========================================
fig1, axs1 = plt.subplots(
    2,
    2,
    figsize=(14, 10),
    subplot_kw={"projection": ccrs.PlateCarree()},
    layout="constrained",
)
fig1.suptitle("Model Recovery: True State vs. Posterior Expectation", fontsize=16)

plot(
    density_true,
    ax=axs1[0, 0],
    coasts=True,
    colorbar=True,
    vmin=-vmax_rho,
    vmax=vmax_rho,
)
axs1[0, 0].set_title("True Density Perturbation (kg/m³)")
plot(
    density_post,
    ax=axs1[0, 1],
    coasts=True,
    colorbar=True,
    vmin=-vmax_rho,
    vmax=vmax_rho,
)
axs1[0, 1].set_title("Posterior Density Perturbation (kg/m³)")

plot(
    traction_true,
    ax=axs1[1, 0],
    coasts=True,
    colorbar=True,
    vmin=-vmax_trac,
    vmax=vmax_trac,
)
axs1[1, 0].set_title("True Basal Traction (Pa)")
plot(
    traction_post,
    ax=axs1[1, 1],
    coasts=True,
    colorbar=True,
    vmin=-vmax_trac,
    vmax=vmax_trac,
)
axs1[1, 1].set_title("Posterior Basal Traction (Pa)")

# ==========================================
# Figure 2: Data Comparison (Geoid & Topography)
# ==========================================
fig2, axs2 = plt.subplots(
    2,
    2,
    figsize=(14, 10),
    subplot_kw={"projection": ccrs.PlateCarree()},
    layout="constrained",
)
fig2.suptitle("Data Comparison: True Fields vs. Posterior Predictions", fontsize=16)

plot(
    -1 * potential_true / g,
    ax=axs2[0, 0],
    coasts=True,
    colorbar=True,
    cmap="PiYG",
    vmin=-vmax_geoid,
    vmax=vmax_geoid,
)
axs2[0, 0].set_title("True Geoid (m)")
plot(
    -1 * potential_post / g,
    ax=axs2[0, 1],
    coasts=True,
    colorbar=True,
    cmap="PiYG",
    vmin=-vmax_geoid,
    vmax=vmax_geoid,
)
axs2[0, 1].set_title("Predicted Posterior Geoid (m)")

plot(
    flexure_true,
    ax=axs2[1, 0],
    coasts=True,
    colorbar=True,
    cmap="coolwarm",
    vmin=-vmax_flex,
    vmax=vmax_flex,
)
axs2[1, 0].set_title("True Topography (m)")
plot(
    flexure_post,
    ax=axs2[1, 1],
    coasts=True,
    colorbar=True,
    cmap="coolwarm",
    vmin=-vmax_flex,
    vmax=vmax_flex,
)
axs2[1, 1].set_title("Predicted Posterior Topography (m)")

plt.show()
