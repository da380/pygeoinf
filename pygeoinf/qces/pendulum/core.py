"""
core.py

A dimension-agnostic engine for numerical integration, statistical analysis,
and Bayesian inference. This module handles N-dimensional grids and
generic differential equation solving.
"""

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
from IPython.display import HTML

# --- Math Utilities ---


def wrap_angle(theta):
    """Wraps an angle or array of angles to the interval [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


# --- Numerical Solvers (Dimension Agnostic) ---


def solve_trajectory(
    eom_func, y0, t_points, args=(), rtol=1e-9, atol=1e-12, method="RK45"
):
    """Integrates a single ODE trajectory over time."""
    t_span = (t_points[0], t_points[-1])
    sol = solve_ivp(
        eom_func,
        t_span,
        y0,
        t_eval=t_points,
        method=method,
        rtol=rtol,
        atol=atol,
        args=args,
    )
    return sol.y


def solve_ensemble(eom_func, initial_conditions, t_points, args=(), **solver_kwargs):
    """Propagates an ensemble of particles forward in time."""
    n_samples, n_dim = initial_conditions.shape
    n_times = len(t_points)
    trajectories = np.zeros((n_samples, n_dim, n_times))

    print(f"Propagating {n_samples} particles ({n_dim}D system)...")

    for i in range(n_samples):
        trajectories[i] = solve_trajectory(
            eom_func, initial_conditions[i], t_points, args=args, **solver_kwargs
        )

    return trajectories


# --- N-Dimensional Statistical Tools ---


def compute_normalization(grid_values, grid_axes):
    """Computes the total integral (volume) of an N-dimensional grid."""
    integral = grid_values
    for i, axis_vals in enumerate(reversed(grid_axes)):
        current_dim = len(grid_axes) - 1 - i
        integral = trapezoid(integral, x=axis_vals, axis=current_dim)
    return integral


def evaluate_pdf_on_grid(pdf_func, grid_limits, resolution):
    """
    Evaluates a callable PDF function onto a discretized meshgrid.

    Args:
        pdf_func: Callable f(x1, x2, ...) -> density.
        grid_limits: List of tuples [(min, max), ...] for each dimension.
        resolution: Int (same for all dims) or List of Ints.

    Returns:
        axes: List of 1D arrays defining the grid coordinates.
        grid_values: N-dimensional array of PDF values.
    """
    n_dim = len(grid_limits)

    # Handle resolution being scalar or list
    if np.isscalar(resolution):
        res_list = [resolution] * n_dim
    else:
        res_list = resolution
        if len(res_list) != n_dim:
            raise ValueError("Resolution list length must match dimension of limits.")

    # Create axes for each dimension
    axes = [np.linspace(l[0], l[1], res) for l, res in zip(grid_limits, res_list)]

    # Create the N-dimensional mesh
    # indexing='ij' ensures matrix indexing (Row, Col, Depth...) rather than Cartesian
    grids = np.meshgrid(*axes, indexing="ij")

    # Evaluate the function
    # We unpack *grids so pdf_func(X, Y, Z...) gets arguments correctly
    grid_values = pdf_func(*grids)

    return axes, grid_values


def marginalise_grid(grid_values, grid_axes, keep_indices):
    """Computes the marginal PDF by integrating out all axes NOT in keep_indices."""
    ndim = len(grid_axes)
    all_indices = set(range(ndim))
    keep_set = set(keep_indices)
    integrate_indices = sorted(list(all_indices - keep_set), reverse=True)

    current_values = grid_values.copy()

    for i in integrate_indices:
        current_values = trapezoid(current_values, x=grid_axes[i], axis=i)

    new_axes = [grid_axes[i] for i in sorted(keep_indices)]
    return new_axes, current_values


def get_pdf_from_grid(grid_axes, grid_values, fill_value=0.0):
    """Creates a callable PDF function from a discrete N-dimensional grid."""
    interp = RegularGridInterpolator(
        grid_axes,
        grid_values,
        bounds_error=False,
        fill_value=fill_value,
        method="linear",
    )

    def pdf_func(*coords):
        if len(coords) != len(grid_axes):
            raise ValueError(f"PDF expects {len(grid_axes)} coordinates.")
        broadcasted = np.broadcast_arrays(*coords)
        result_shape = broadcasted[0].shape
        points = np.column_stack([b.ravel() for b in broadcasted])
        values = interp(points)
        return values.reshape(result_shape)

    return pdf_func


def advect_pdf_grid(eom_func, pdf_func, t_final, grid_limits, resolution, eom_args=()):
    """Generic Liouville Advection on a hyper-grid."""

    # 1. Evaluate Initial PDF on Grid (Refactored to use common logic)
    axes, initial = evaluate_pdf_on_grid(pdf_func, grid_limits, resolution)

    # Need grids for advection calculation
    grids = np.meshgrid(*axes, indexing="ij")
    n_dim = len(axes)

    # 2. Back-propagate Grid Points to t=0
    flat_state = np.stack([g.ravel() for g in grids])
    y0_vectorized = flat_state.reshape(-1)

    def vectorized_eom(t, y_flat):
        y_reshaped = y_flat.reshape(n_dim, -1)
        dydt = eom_func(t, y_reshaped, *eom_args)
        return np.concatenate(dydt).reshape(-1)

    # Integrate BACKWARDS: t_final -> 0
    t_span = [t_final, 0.0]

    # We only care about the FINAL state of this backward integration (which is the origin at t=0)
    sol = solve_trajectory(
        vectorized_eom, y0_vectorized, t_span, args=(), rtol=1e-9, atol=1e-9
    )

    origins_flat = sol[:, -1]
    grid_shape = [len(ax) for ax in axes]
    origins = origins_flat.reshape(n_dim, *grid_shape)

    # 3. Evaluate PDF at the origins (Liouville's Theorem: f(x_t, t) = f(x_0, 0))
    # Note: This assumes divergence-free flow (conservative system).
    advected = pdf_func(*origins)

    # 4. Normalize
    norm_const = compute_normalization(initial, axes)
    if norm_const == 0:
        norm_const = 1.0

    return axes, initial / norm_const, advected / norm_const


# --- Bayesian Tools ---


def get_gaussian_pdf_func(mean, cov):
    """
    Returns a callable PDF function that handles full covariance matrices.
    """
    from scipy.stats import multivariate_normal

    # Pre-calculate the distribution object for speed
    dist = multivariate_normal(mean=mean, cov=cov)

    def pdf_func(*args):
        # args will be (THETA_grid, P_grid, ...)
        # We need to stack them into (..., N_dim) to match scipy's expectations
        pos = np.stack(args, axis=-1)
        return dist.pdf(pos)

    return pdf_func


def get_independent_gaussian_func(means, stds):
    """
    Factory that returns a callable PDF function for independent Gaussians.
    """
    means = np.array(means)
    stds = np.array(stds)

    def pdf_func(*args):
        total_prob = 1.0
        for i, val in enumerate(args):
            mu = means[i]
            sigma = stds[i]
            prob = np.exp(-0.5 * ((val - mu) / sigma) ** 2)
            total_prob *= prob
        return total_prob

    return pdf_func


class GaussianLikelihood:
    """
    Handles Gaussian likelihoods for generic non-linear observation operators.
    Likelihood L(x) = P(y_obs | x) ~ N(y_obs; H(x), R)
    """

    def __init__(
        self, observation_value, observation_covariance, obs_operator_func=None
    ):
        """
        Args:
            observation_value: The observed data vector 'y_obs'.
            observation_covariance: The observation error covariance matrix 'R'.
            obs_operator_func: Callable H(x) -> y.
                               Must accept shape (..., StateDim) and return (..., ObsDim).
                               If None, assumes Direct Observation H(x)=x.
        """
        self.y_obs = np.atleast_1d(observation_value)
        self.R = np.atleast_2d(observation_covariance)

        # Precompute precision matrix (inverse covariance) for speed
        try:
            self.R_inv = np.linalg.inv(self.R)
            self.norm_factor = 1.0 / np.sqrt(
                (2 * np.pi) ** len(self.y_obs) * np.linalg.det(self.R)
            )
        except np.linalg.LinAlgError:
            raise ValueError("Observation covariance matrix R must be invertible.")

        # Default to identity operator if none provided
        if obs_operator_func is None:
            self.H = lambda x: x
        else:
            self.H = obs_operator_func

    def evaluate(self, state_grid):
        """
        Computes likelihood on a grid of states.

        Args:
            state_grid: Array of shape (..., StateDim).
                        Can be the 'grids' tuple from meshgrid if stacked properly,
                        or a raw array of state vectors.

        Returns:
            Likelihood grid of shape (...)
        """
        # Ensure input is a single array (..., StateDim)
        if isinstance(state_grid, (tuple, list)):
            # Assume it's a meshgrid tuple -> stack them
            state_grid = np.stack(state_grid, axis=-1)

        # 1. Map State Space -> Observation Space: y_pred = H(x)
        # We expect H to vectorize over the leading dimensions
        y_pred = self.H(state_grid)

        # 2. Compute Residual: d = y_obs - H(x)
        # Broadcasting: (ObsDim,) - (..., ObsDim) -> (..., ObsDim)
        residual = self.y_obs - y_pred

        # 3. Mahalanobis Distance: d^T * R^-1 * d
        # We need to do this computation per grid point.
        # einsum is perfect here:
        # '...i, ij, ...j -> ...' means:
        # for every grid point (...), take row vector res (i), dot with R_inv (ij), dot with col vector res (j)
        mahalanobis = np.einsum("...i, ij, ...j -> ...", residual, self.R_inv, residual)

        return self.norm_factor * np.exp(-0.5 * mahalanobis)


class LinearGaussianLikelihood(GaussianLikelihood):
    """
    Specialized class for Linear Observation Operators: y = Hx
    """

    def __init__(self, observation_value, observation_covariance, observation_matrix):
        """
        Args:
            observation_matrix: The observation matrix (ObsDim, StateDim).
        """
        # Define the callable wrapper for the parent class, but we won't use it directly in evaluate
        # to save overhead.
        self.H_mat = np.atleast_2d(observation_matrix)

        def linear_op(x):
            # x is (..., N), H is (M, N). Result (..., M)
            return x @ self.H_mat.T

        super().__init__(observation_value, observation_covariance, linear_op)

    def evaluate(self, state_grid):
        """Optimized evaluation for linear H."""
        if isinstance(state_grid, (tuple, list)):
            state_grid = np.stack(state_grid, axis=-1)

        # Optimization: Linear algebra is often faster than generic function calls
        # (..., N) @ (N, M) -> (..., M)
        y_pred = state_grid @ self.H_mat.T

        residual = self.y_obs - y_pred
        mahalanobis = np.einsum("...i, ij, ...j -> ...", residual, self.R_inv, residual)

        return self.norm_factor * np.exp(-0.5 * mahalanobis)


def gaussian_likelihood(x_grid, observation_value, obs_std):
    """Computes Gaussian likelihood on a grid."""
    return norm.pdf(x_grid, loc=observation_value, scale=obs_std)


def bayesian_update(prior_grid, likelihood_grid, grid_axes):
    """Posterior = (Likelihood * Prior) / Evidence"""
    posterior_unnorm = likelihood_grid * prior_grid
    evidence = compute_normalization(posterior_unnorm, grid_axes)
    if evidence == 0:
        return posterior_unnorm, 0.0
    return posterior_unnorm / evidence, evidence


def assimilate_cycle(
    t_obs,
    observations,
    obs_std,
    prior_func,
    forecast_func,
    analysis_func,
    grid_to_func_wrapper,
):
    """Generic Grid-Based Assimilation Cycle."""
    results = []
    current_pdf_func = prior_func
    t_prev = 0.0

    for i, t_now in enumerate(t_obs):
        obs_val = observations[i]
        dt = t_now - t_prev

        # 1. FORECAST
        forecast = forecast_func(current_pdf_func, dt)

        # 2. ANALYSIS
        posterior, evidence = analysis_func(forecast, obs_val, obs_std)

        # 3. Store
        results.append({"time": t_now, "posterior": posterior, "evidence": evidence})

        # 4. Prepare Next Step
        current_pdf_func = grid_to_func_wrapper(posterior)
        t_prev = t_now

    return results


def sample_from_grid(grid_axes, grid_values, n_samples=1000):
    """
    Efficiently samples from an N-dimensional grid-based PDF.

    Strategy:
    1. Treat grid cells as discrete bins with probability mass ~ PDF value.
    2. Sample grid indices based on these probabilities.
    3. Add uniform jitter (dithering) within the cell size to make it continuous.

    Args:
        grid_axes: List of 1D arrays defining the grid coordinates.
        grid_values: N-dimensional array of PDF values.
        n_samples: Number of samples to generate.

    Returns:
        samples: Array of shape (n_samples, n_dim)
    """
    # 1. Compute Probability Mass Function (PMF)
    # We assume uniform grid spacing per dimension (dx, dy, etc.)
    # So Mass ~ Density (we can ignore the dx*dy factor for normalization)

    flat_pdf = grid_values.ravel()

    # Clip negative values (numerical errors) and normalize
    flat_pdf = np.maximum(flat_pdf, 0)
    total_mass = flat_pdf.sum()

    if total_mass == 0:
        raise ValueError("PDF grid has zero total probability mass.")

    pmf = flat_pdf / total_mass

    # 2. Discrete Sampling of Indices
    # Returns flat indices into the grid array
    rng = np.random.default_rng()
    flat_indices = rng.choice(len(pmf), size=n_samples, p=pmf)

    # Convert flat indices to N-dimensional coordinates (i, j, k...)
    # tuple of arrays, one per dimension
    multi_indices = np.unravel_index(flat_indices, grid_values.shape)

    # 3. Convert Indices to Continuous Coordinates + Dithering
    n_dim = len(grid_axes)
    samples = np.zeros((n_samples, n_dim))

    for d in range(n_dim):
        axis = grid_axes[d]
        indices = multi_indices[d]

        # Grid spacing (assuming regular grid)
        # Handle edge case where len(axis) == 1
        if len(axis) > 1:
            dx = axis[1] - axis[0]
        else:
            dx = 0.0

        # Base coordinate (grid point center)
        coords = axis[indices]

        # Add Jitter: Uniform noise [-dx/2, +dx/2]
        # This smoothes the distribution so samples don't align on lines
        jitter = rng.uniform(-dx / 2, dx / 2, size=n_samples)

        samples[:, d] = coords + jitter

    return samples


# --- Generic Visualization Helpers ---


def display_animation_html(anim):
    """Standard helper to render animations in notebooks."""
    print("Rendering animation...")
    return HTML(anim.to_jshtml())
