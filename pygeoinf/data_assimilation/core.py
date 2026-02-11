"""
core.py

A dimension-agnostic engine for numerical integration, statistical analysis,
Bayesian inference, and generic visualisation.
"""

from typing import Callable, List, Tuple, Union, Optional, Any, Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import multivariate_normal
from IPython.display import HTML


# --- Math Utilities ---


def wrap_angle(theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Wraps an angle or array of angles to the interval [-pi, pi].

    Args:
        theta: Input angle(s) in radians.

    Returns:
        The wrapped angle(s) in [-pi, pi].
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


# --- Numerical Solvers (Dimension Agnostic) ---


def solve_trajectory(
    eom_func: Callable[[float, np.ndarray, Any], np.ndarray],
    y0: np.ndarray,
    t_points: np.ndarray,
    args: Tuple = (),
    rtol: float = 1e-9,
    atol: float = 1e-12,
    method: str = "RK45",
) -> np.ndarray:
    """
    Integrates a single ODE trajectory over time.

    Args:
        eom_func: The Equation of Motion function f(t, y, *args) -> dy/dt.
        y0: Initial state vector of shape (n_dim,).
        t_points: Array of time points to evaluate at.
        args: Tuple of extra arguments to pass to eom_func.
        rtol: Relative tolerance for solver.
        atol: Absolute tolerance for solver.
        method: Integration method (e.g., 'RK45', 'DOP853').

    Returns:
        Solution array of shape (n_dim, n_times).
    """
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


def solve_ensemble(
    eom_func: Callable[[float, np.ndarray, Any], np.ndarray],
    initial_conditions: np.ndarray,
    t_points: np.ndarray,
    args: Tuple = (),
    **solver_kwargs,
) -> np.ndarray:
    """
    Propagates an ensemble of particles forward in time.

    Args:
        eom_func: The Equation of Motion function.
        initial_conditions: Array of shape (n_samples, n_dim).
        t_points: Array of time points.
        args: Physics arguments passed to eom_func.
        **solver_kwargs: Extra kwargs for solve_trajectory (rtol, atol, etc).

    Returns:
        Trajectories array of shape (n_samples, n_dim, n_times).
    """
    n_samples, n_dim = initial_conditions.shape
    n_times = len(t_points)
    trajectories = np.zeros((n_samples, n_dim, n_times))

    print(f"Propagating {n_samples} particles ({n_dim}D system)...")

    for i in range(n_samples):
        trajectories[i] = solve_trajectory(
            eom_func, initial_conditions[i], t_points, args=args, **solver_kwargs
        )

    return trajectories


# --- Distribution Factories ---


def get_gaussian_pdf(mean: np.ndarray, cov: np.ndarray) -> Callable[..., np.ndarray]:
    """
    Returns a callable PDF function for an N-dimensional multivariate Gaussian.
    The returned function is compatible with ProbabilityGrid factories.

    Args:
        mean: Mean vector of shape (N,).
        cov: Covariance matrix of shape (N, N).

    Returns:
        A function pdf(*coordinates) -> density_array.
    """
    # Create the frozen distribution object
    dist = multivariate_normal(mean=mean, cov=cov)

    def pdf_func(*coords):
        # coords is a tuple of meshgrids (X, Y, Z, ...), each shape (Ni, Nj, Nk...)
        # We stack them into shape (..., N_dim) for scipy
        pos = np.stack(coords, axis=-1)
        return dist.pdf(pos)

    return pdf_func


def get_independent_gaussian_pdf(
    means: Union[List[float], np.ndarray], stds: Union[List[float], np.ndarray]
) -> Callable[..., np.ndarray]:
    """
    Returns a callable PDF for N independent Gaussians (diagonal covariance).
    Faster and simpler than the full multivariate version.

    Args:
        means: List or array of means for each dimension.
        stds: List or array of standard deviations for each dimension.

    Returns:
        A function pdf(*coordinates) -> density_array.
    """
    means = np.asarray(means)
    stds = np.asarray(stds)

    def pdf_func(*coords):
        if len(coords) != len(means):
            raise ValueError(
                f"Number of coordinates ({len(coords)}) does not match dimension of means ({len(means)})."
            )

        # Calculate product of 1D Gaussians
        result = 1.0
        for i, val in enumerate(coords):
            mu = means[i]
            sigma = stds[i]
            # Normalisation factor for this dimension
            norm_const = 1.0 / (sigma * np.sqrt(2 * np.pi))
            exponent = -0.5 * ((val - mu) / sigma) ** 2
            result *= norm_const * np.exp(exponent)

        return result

    return pdf_func


# --- The Probability Grid Class ---


class ProbabilityGrid:
    """
    Encapsulates an N-dimensional probability density function discretised
    on a rectilinear grid. Handles normalisation, marginalisation, and
    Bayesian updates.
    """

    def __init__(self, axes: List[np.ndarray], values: np.ndarray):
        """
        Args:
            axes: List of 1D arrays defining the grid coordinates for each dimension.
            values: N-dimensional array of density values matching the shape of axes.
        """
        self.axes = axes
        self.values = values
        self.ndim = len(axes)
        self.shape = values.shape

        # Validation
        expected_shape = tuple(len(ax) for ax in self.axes)
        if self.values.shape != expected_shape:
            raise ValueError(
                f"Grid shape {self.values.shape} mismatch with axes {expected_shape}"
            )

    @classmethod
    def from_bounds(
        cls,
        bounds: List[Tuple[float, float]],
        resolution: Union[int, List[int]],
        pdf_func: Optional[Callable[..., np.ndarray]] = None,
    ) -> "ProbabilityGrid":
        """
        Factory: Creates a grid from bounds [(min, max), ...] and resolution.
        Optionally evaluates a pdf_func on that grid immediately.

        Args:
            bounds: List of (min, max) tuples for each dimension.
            resolution: Number of points per dimension (int or list of ints).
            pdf_func: Optional callable f(x1, x2...) -> density to evaluate on initialisation.

        Returns:
            A new ProbabilityGrid instance.
        """
        ndim = len(bounds)
        # Handle scalar vs list resolution
        if np.isscalar(resolution):
            res_list = [int(resolution)] * ndim  # type: ignore
        else:
            res_list = list(resolution)  # type: ignore

        # Create axes
        axes = [np.linspace(b[0], b[1], r) for b, r in zip(bounds, res_list)]

        if pdf_func:
            # Create mesh and evaluate (indexing='ij' for matrix order)
            mesh = np.meshgrid(*axes, indexing="ij")
            values = pdf_func(*mesh)
        else:
            # Create empty grid
            shape = tuple(len(a) for a in axes)
            values = np.zeros(shape)

        return cls(axes, values)

    @property
    def total_mass(self) -> float:
        """Computes the total integral (volume) of the grid using the trapezoidal rule."""
        integral = self.values
        # Integrate over each dimension
        for i, axis_vals in enumerate(reversed(self.axes)):
            current_dim = self.ndim - 1 - i
            integral = trapezoid(integral, x=axis_vals, axis=current_dim)
        return float(integral)

    @property
    def mean(self) -> np.ndarray:
        """
        Computes the expected value vector E[x] of the distribution.
        """
        # Ensure we are working with a normalised distribution for the expectation
        grid_norm = self.normalise()
        means = []

        # Calculate mean for each dimension by marginalising to 1D
        for i in range(self.ndim):
            marginal = grid_norm.marginalise(keep_indices=(i,))
            # E[x] = integral(x * p(x) dx)
            expected_val = trapezoid(
                marginal.axes[0] * marginal.values, marginal.axes[0]
            )
            means.append(expected_val)

        return np.array(means)

    def normalise(self) -> "ProbabilityGrid":
        """
        Returns a NEW ProbabilityGrid that sums to 1.0.
        If mass is zero, returns the original grid to avoid division errors.
        """
        mass = self.total_mass
        if mass == 0:
            return ProbabilityGrid(self.axes, self.values)
        return ProbabilityGrid(self.axes, self.values / mass)

    def marginalise(self, keep_indices: Tuple[int, ...]) -> "ProbabilityGrid":
        """
        Integrates out all axes NOT in keep_indices.

        Args:
            keep_indices: Tuple of dimension indices to retain (e.g., (0, 1)).

        Returns:
            A lower-dimensional ProbabilityGrid.
        """
        keep_set = set(keep_indices)
        all_indices = set(range(self.ndim))
        integrate_indices = sorted(list(all_indices - keep_set), reverse=True)

        current_values = self.values.copy()

        for i in integrate_indices:
            current_values = trapezoid(current_values, x=self.axes[i], axis=i)

        new_axes = [self.axes[i] for i in sorted(keep_indices)]
        return ProbabilityGrid(new_axes, current_values)

    def to_interpolator(self, fill_value: float = 0.0) -> Callable[..., np.ndarray]:
        """
        Returns a callable function f(x1, x2, ...) backed by this grid.
        Uses linear interpolation.
        """
        interp = RegularGridInterpolator(
            self.axes,
            self.values,
            bounds_error=False,
            fill_value=fill_value,
            method="linear",
        )

        def pdf_func(*coords):
            # Wrapper to handle broadcasting for meshgrids
            if len(coords) != self.ndim:
                raise ValueError(f"PDF expects {self.ndim} coordinates.")
            broadcasted = np.broadcast_arrays(*coords)
            result_shape = broadcasted[0].shape
            points = np.column_stack([b.ravel() for b in broadcasted])
            values = interp(points)
            return values.reshape(result_shape)

        return pdf_func

    def sample(self, n_samples: int = 1000) -> np.ndarray:
        """
        Efficiently samples from the grid using PMF approximation + jitter.

        Args:
            n_samples: Number of samples to draw.

        Returns:
            Array of shape (n_samples, n_dim) containing the samples.
        """
        flat_pdf = self.values.ravel()
        flat_pdf = np.maximum(flat_pdf, 0)  # Clip negative noise
        total_mass = flat_pdf.sum()

        if total_mass == 0:
            raise ValueError("PDF grid has zero total probability mass.")

        pmf = flat_pdf / total_mass

        # Discrete Sampling
        rng = np.random.default_rng()
        flat_indices = rng.choice(len(pmf), size=n_samples, p=pmf)
        multi_indices = np.unravel_index(flat_indices, self.shape)

        # Dithering (Jitter) to make continuous
        samples = np.zeros((n_samples, self.ndim))

        for d in range(self.ndim):
            axis = self.axes[d]
            indices = multi_indices[d]

            if len(axis) > 1:
                dx = axis[1] - axis[0]
            else:
                dx = 0.0

            coords = axis[indices]
            jitter = rng.uniform(-dx / 2, dx / 2, size=n_samples)
            samples[:, d] = coords + jitter

        return samples

    def push_forward(
        self,
        eom_func: Callable,
        t_final: float,
        eom_args: Tuple = (),
    ) -> "ProbabilityGrid":
        """
        Performs Liouville Advection (Method of Characteristics).
        Propagates the probability density forward in time by 't_final'.

        Args:
            eom_func: The differential equation f(t, y, *args).
            t_final: The time duration to propagate forward.
            eom_args: Physics arguments for the eom_func.

        Returns:
            A NEW ProbabilityGrid at t=t_final (normalised).
        """
        # 1. Create interpolator for current state
        initial_pdf_func = self.to_interpolator()

        # 2. Setup target mesh (Eulerian: we use the same grid definition)
        grids = np.meshgrid(*self.axes, indexing="ij")

        # 3. Back-propagate Grid Points to t=0
        flat_state = np.stack([g.ravel() for g in grids])
        y0_vectorized = flat_state.reshape(-1)

        def vectorized_eom(t, y_flat):
            y_reshaped = y_flat.reshape(self.ndim, -1)
            dydt = eom_func(t, y_reshaped, *eom_args)
            return np.concatenate(dydt).reshape(-1)

        # Integrate BACKWARDS: t_final -> 0
        sol = solve_trajectory(
            vectorized_eom,
            y0_vectorized,
            np.array([t_final, 0.0]),
            rtol=1e-5,
            atol=1e-5,
        )

        origins_flat = sol[:, -1]
        origins = origins_flat.reshape(self.ndim, *self.shape)

        # 4. Evaluate initial PDF at the back-propagated origins
        advected_values = initial_pdf_func(*origins)

        # 5. Return new normalised object
        new_grid = ProbabilityGrid(self.axes, advected_values)
        return new_grid.normalise()

    def __mul__(
        self, other: Union["ProbabilityGrid", np.ndarray, float]
    ) -> "ProbabilityGrid":
        """
        Element-wise multiplication.
        Supports: Grid * Grid, Grid * Scalar, Grid * Array.
        """
        if isinstance(other, ProbabilityGrid):
            if self.shape != other.shape:
                raise ValueError("Grids must have same shape for multiplication.")
            new_values = self.values * other.values
        else:
            # Assume numpy array or scalar
            new_values = self.values * other

        return ProbabilityGrid(self.axes, new_values)

    def __truediv__(self, scalar: float) -> "ProbabilityGrid":
        """
        Devide the grid values by a scalar
        """
        return self * (1 / scalar)

    def bayes_update(
        self, likelihood: Union["ProbabilityGrid", np.ndarray, float]
    ) -> Tuple["ProbabilityGrid", float]:
        """
        Calculates Posterior = (Prior * Likelihood) / Evidence.

        Args:
            likelihood: Can be a ProbabilityGrid, a numpy array (same shape),
                        or a scalar representing P(y|x).

        Returns:
            posterior (ProbabilityGrid): Normalised posterior density.
            evidence (float): The normalisation constant (integral of Prior * Likelihood).
        """
        # 1. Compute Unnormalised Posterior (Prior * Likelihood)
        unnormalised = self * likelihood

        # 2. Compute Evidence (Total Probability)
        evidence = unnormalised.total_mass

        # 3. Handle Zero Probability (Numerical collapse)
        if evidence == 0:
            print("Warning: Evidence is zero. Posterior is undefined.")
            return unnormalised, 0.0

        # 4. Normalise
        # We manually create normalised grid to avoid re-integrating inside normalise()
        posterior = ProbabilityGrid(self.axes, unnormalised.values / evidence)

        return posterior, evidence


# --- Bayesian Observation Models ---


class GaussianLikelihood:
    """
    Handles Gaussian likelihoods for generic non-linear observation operators.
    Likelihood L(x) = P(y_obs | x) ~ N(y_obs; H(x), R)
    """

    def __init__(
        self,
        observation_value: Optional[np.ndarray],
        observation_covariance: np.ndarray,
        obs_operator_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Args:
            observation_value: The observed data vector 'y_obs' (ObsDim,).
                               Can be None if generating synthetic data.
            observation_covariance: The observation error covariance matrix 'R' (ObsDim, ObsDim).
            obs_operator_func: Callable H(x) -> y.
                               Must accept shape (..., StateDim) and return (..., ObsDim).
                               If None, assumes Direct Observation H(x)=x.
        """
        if observation_value is not None:
            self.y_obs = np.atleast_1d(observation_value)
        else:
            self.y_obs = None

        self.R = np.atleast_2d(observation_covariance)

        # Precompute precision matrix (inverse covariance) for speed
        try:
            self.R_inv = np.linalg.inv(self.R)
            # Normalisation factor for the Gaussian
            # Use covariance dimension for normalization constant
            k = self.R.shape[0]
            det_R = np.linalg.det(self.R)
            self.norm_factor = 1.0 / np.sqrt(((2 * np.pi) ** k) * det_R)
        except np.linalg.LinAlgError:
            raise ValueError("Observation covariance matrix R must be invertible.")

        if obs_operator_func is None:
            self.H = lambda x: x
        else:
            self.H = obs_operator_func

    def evaluate(self, prob_grid: ProbabilityGrid) -> ProbabilityGrid:
        """
        Evaluates the likelihood on the given ProbabilityGrid.
        Returns a new ProbabilityGrid instance containing the Likelihood values.
        """
        if self.y_obs is None:
            raise ValueError(
                "Cannot evaluate likelihood: Observation value is not set."
            )

        # 1. Generate State Vectors
        mesh = np.meshgrid(*prob_grid.axes, indexing="ij")
        state_vectors = np.stack(mesh, axis=-1)

        # 2. Map State Space -> Observation Space: y_pred = H(x)
        y_pred = self.H(state_vectors)

        # 3. Compute Residual: d = y_obs - H(x)
        residual = self.y_obs - y_pred

        # 4. Mahalanobis Distance: d^T * R^-1 * d
        mahalanobis = np.einsum("...i, ij, ...j -> ...", residual, self.R_inv, residual)

        # 5. Compute Gaussian
        likelihood_values = self.norm_factor * np.exp(-0.5 * mahalanobis)

        return ProbabilityGrid(prob_grid.axes, likelihood_values)

    def sample(self, true_state: np.ndarray) -> np.ndarray:
        """
        Generates a noisy observation y given a true state x.
        y = H(x) + N(0, R)

        Args:
            true_state: The true state vector x.

        Returns:
            The noisy observation vector y.
        """
        # 1. Apply Observation Operator H(x)
        # Handle shape: input (N,) -> output (M,)
        y_clean = self.H(true_state)
        y_clean = np.atleast_1d(y_clean)

        # 2. Add Noise ~ N(0, R)
        noise = np.random.multivariate_normal(np.zeros(len(y_clean)), self.R)

        return y_clean + noise


class LinearGaussianLikelihood(GaussianLikelihood):
    """
    Optimised subclass for Linear Observation Operators: y = Hx
    Avoids generic function calls in favour of matrix multiplication.
    """

    def __init__(
        self,
        observation_value: Optional[np.ndarray],
        observation_covariance: np.ndarray,
        observation_matrix: np.ndarray,
    ):
        """
        Args:
            observation_value: The observed data vector 'y_obs' (ObsDim,).
            observation_covariance: The observation error covariance matrix 'R'.
            observation_matrix: The observation matrix H (ObsDim, StateDim).
        """
        self.H_mat = np.atleast_2d(observation_matrix)

        def linear_op(x):
            return x @ self.H_mat.T

        super().__init__(observation_value, observation_covariance, linear_op)

    def evaluate(self, prob_grid: ProbabilityGrid) -> ProbabilityGrid:
        """Optimised evaluation for linear H."""
        if self.y_obs is None:
            raise ValueError(
                "Cannot evaluate likelihood: Observation value is not set."
            )

        # 1. Generate State Vectors
        mesh = np.meshgrid(*prob_grid.axes, indexing="ij")
        state_vectors = np.stack(mesh, axis=-1)

        # 2. Linear Algebra Prediction: (..., N) @ (N, M) -> (..., M)
        y_pred = state_vectors @ self.H_mat.T

        # 3. Standard Gaussian Logic
        residual = self.y_obs - y_pred
        mahalanobis = np.einsum("...i, ij, ...j -> ...", residual, self.R_inv, residual)
        likelihood_values = self.norm_factor * np.exp(-0.5 * mahalanobis)

        return ProbabilityGrid(prob_grid.axes, likelihood_values)


# --- Bayesian Assimilation Manager ---


class BayesianAssimilationProblem:
    """
    Manages the definition and execution of a Bayesian assimilation cycle.
    Stores the system dynamics and a sequence of observations, managing the
    Forecast-Analysis loop automatically.
    """

    def __init__(
        self,
        eom_func: Callable[[float, np.ndarray, Any], np.ndarray],
        eom_args: Tuple = (),
    ):
        """
        Args:
            eom_func: The system dynamics ODE function.
            eom_args: Arguments (constants/parameters) for the ODE function.
        """
        self.eom_func = eom_func
        self.eom_args = eom_args
        self.observations: List[Tuple[float, Any]] = (
            []
        )  # Stores (time, likelihood_model)

    def add_observation(
        self,
        time: float,
        covariance: np.ndarray,
        value: Optional[np.ndarray] = None,
        operator: Optional[Union[np.ndarray, Callable]] = None,
    ):
        """
        Adds a Gaussian observation to the problem.

        Args:
            time: The time at which the observation is made.
            covariance: The observation error covariance matrix 'R'.
            value: The observed data vector 'y_obs'. If None, placeholder for synthetic data.
            operator: The observation operator H. Can be a Matrix (for linear)
                      or a Callable (for non-linear). If None, assumes Identity.
        """
        # Auto-detect Linear vs Non-linear to choose optimised class
        if operator is None or callable(operator):
            model = GaussianLikelihood(value, covariance, operator)
        else:
            # Assume it's a matrix
            model = LinearGaussianLikelihood(value, covariance, operator)

        self.observations.append((time, model))
        # Ensure sequential order
        self.observations.sort(key=lambda x: x[0])

    def generate_synthetic_data(
        self,
        true_initial_condition: np.ndarray,
        dt_render: float = 0.05,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs the physics from t=0, samples noisy observations at registered times,
        updates the internal Likelihood models with these values, and returns ground truth.

        Args:
            true_initial_condition: The starting state (x0).
            dt_render: Time step for the returned ground truth trajectory.
            seed: Random seed for reproducibility.

        Returns:
            Dict containing 't_ground_truth', 'state_ground_truth', 'initial_truth'.
        """
        if seed is not None:
            np.random.seed(seed)

        # 1. Identify observation times
        # Note: self.observations is a list of tuples (time, model)
        obs_times = np.array([t for t, _ in self.observations])

        if len(obs_times) == 0:
            raise ValueError("No observations registered to generate data for.")

        # 2. Run High-Res Simulation (Ground Truth)
        # We ensure we cover the full range from 0 to last obs
        t_max = obs_times[-1]
        t_render = np.arange(0, t_max + dt_render, dt_render)
        if t_render[-1] < t_max:
            t_render = np.append(t_render, t_max)

        # We must also ensure we hit exactly the observation times to sample accurately
        # Strategy: Solve for unique sorted times of (t_render U t_obs)
        all_times = np.unique(np.concatenate([t_render, obs_times]))

        sol_all = solve_trajectory(
            self.eom_func, true_initial_condition, all_times, args=self.eom_args
        )

        # 3. Iterate through registered observations and update them

        for i, (t_obs, model) in enumerate(self.observations):
            # Find the state at this exact time
            # np.isclose is safer than equality for floats
            idx = np.where(np.isclose(all_times, t_obs))[0][0]
            true_state_at_t = sol_all[:, idx]

            # Sample noisy observation
            noisy_val = model.sample(true_state_at_t)

            # UPDATE the model stored in the problem list with the sampled value
            model.y_obs = noisy_val

        return {
            "t_ground_truth": all_times,
            "state_ground_truth": sol_all,
            "initial_truth": true_initial_condition,
        }

    def run(
        self, initial_prior: ProbabilityGrid, t_final: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes the assimilation cycle starting from t=0.

        Args:
            initial_prior: The ProbabilityGrid at t=0.
            t_final: Optional end time. If > last observation, forecasts to this time.

        Returns:
            A list of dictionaries, one per step, containing:
            {'time', 'forecast', 'analysis', 'evidence'}
        """
        history = []
        current_grid = initial_prior
        t_current = 0.0

        for t_obs, lik_model in self.observations:
            dt = t_obs - t_current

            if dt < 0:
                raise ValueError(
                    f"Observation at t={t_obs} is in the past relative to t={t_current}."
                )

            # 1. Forecast Step (Advection)
            if dt > 0:
                # Evolve from t_current to t_obs
                forecast_grid = current_grid.push_forward(
                    self.eom_func, dt, self.eom_args
                )
            else:
                # Observation is immediate (e.g. at t=0)
                forecast_grid = current_grid

            # 2. Analysis Step (Bayesian Update)
            # Evaluate likelihood on the forecasted grid
            lik_grid = lik_model.evaluate(forecast_grid)
            # Update
            analysis_grid, evidence = forecast_grid.bayes_update(lik_grid)

            # Store Result
            history.append(
                {
                    "time": t_obs,
                    "forecast": forecast_grid,
                    "analysis": analysis_grid,
                    "evidence": evidence,
                }
            )

            # Advance
            current_grid = analysis_grid
            t_current = t_obs

        # Optional: Final forecast to t_final
        if t_final is not None and t_final > t_current:
            dt = t_final - t_current
            final_grid = current_grid.push_forward(self.eom_func, dt, self.eom_args)
            history.append(
                {
                    "time": t_final,
                    "forecast": final_grid,
                    "analysis": final_grid,  # No observation
                    "evidence": 1.0,  # No update
                }
            )

        return history


# --- Post-Processing / Reanalysis Tools ---


def reanalyse_initial_condition(
    final_posterior: ProbabilityGrid,
    t_final: float,
    eom_func: Callable,
    eom_args: Tuple,
) -> Tuple[ProbabilityGrid, np.ndarray]:
    """
    Performs Reanalysis (Smoothing) by pulling the final posterior back to t=0.

    Args:
        final_posterior: The ProbabilityGrid at t=t_final.
        t_final: The time of the posterior.
        eom_func: Physics ODE.
        eom_args: Physics constants.

    Returns:
        smoothed_initial_grid: ProbabilityGrid at t=0.
        smoothed_initial_mean: The mean vector of the smoothed distribution.
    """
    # Use reverse advection (negative time)
    # This maps density f(x, T) back to f(x, 0) via Liouville
    smoothed_grid = final_posterior.push_forward(eom_func, -t_final, eom_args)

    return smoothed_grid, smoothed_grid.mean


# --- Generic Visualisation Tools ---


def plot_grid_marginal(
    prob_grid: ProbabilityGrid,
    dims: Tuple[int, int] = (0, 1),
    ax: Optional[plt.Axes] = None,
    filled: bool = True,
    **kwargs,
) -> Tuple[plt.Axes, Any]:
    """
    Marginalises an N-dimensional ProbabilityGrid down to 2 dimensions
    and plots the result as a contour.

    Args:
        prob_grid: core.ProbabilityGrid instance.
        dims: Tuple of (x_dim_index, y_dim_index) to keep.
        ax: Matplotlib Axes object. If None, creates a new figure.
        filled: Boolean, whether to use contourf (filled) or contour (lines).
        **kwargs: Passed directly to ax.contourf/ax.contour (e.g., levels, cmap).

    Returns:
        ax: The matplotlib Axis.
        contour: The contour plot object.
    """
    # 1. Marginalise
    marginal_grid = prob_grid.marginalise(keep_indices=dims)

    # 2. Extract data
    x_axis = marginal_grid.axes[0]
    y_axis = marginal_grid.axes[1]
    Z = marginal_grid.values

    # 3. Setup Plot
    if ax is None:
        fig, ax = plt.subplots()

    # 4. Create Meshgrid for plotting
    X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")

    # 5. Plot
    if "levels" not in kwargs:
        kwargs["levels"] = 30

    if filled:
        contour = ax.contourf(X, Y, Z, **kwargs)
    else:
        contour = ax.contour(X, Y, Z, **kwargs)

    ax.grid(True, alpha=0.3, linestyle="--")

    return ax, contour


def plot_ensemble_scatter(
    trajectories: np.ndarray,
    dim_indices: Tuple[int, int] = (0, 1),
    time_idx: int = -1,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Generic scatter plot of ensemble particles at a specific time snapshot.

    Args:
        trajectories: Array of shape (n_samples, n_dim, n_times).
        dim_indices: Tuple of (x_dim, y_dim) indices to plot.
        time_idx: Integer index of the time step to plot.
        ax: Matplotlib Axes object.
        **kwargs: Passed to ax.scatter (e.g., c, s, alpha, label).

    Returns:
        The matplotlib Axis.
    """
    # 1. Extract Snapshot
    snap = trajectories[:, :, time_idx]

    # 2. Select Dimensions
    x_data = snap[:, dim_indices[0]]
    y_data = snap[:, dim_indices[1]]

    # 3. Setup Plot
    if ax is None:
        fig, ax = plt.subplots()

    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.5
    if "s" not in kwargs:
        kwargs["s"] = 10

    # 4. Plot
    ax.scatter(x_data, y_data, **kwargs)
    ax.grid(True, alpha=0.3)

    return ax


def plot_1d_slice(
    prob_grid: ProbabilityGrid, dim: int = 0, ax: Optional[plt.Axes] = None, **kwargs
) -> plt.Axes:
    """
    Marginalises down to a single dimension and plots a line graph (PDF).

    Args:
        prob_grid: core.ProbabilityGrid instance.
        dim: Index of the dimension to keep.
        ax: Matplotlib Axes object.
        **kwargs: Passed to ax.plot (e.g., color, linewidth).

    Returns:
        The matplotlib Axis.
    """
    marginal_grid = prob_grid.marginalise(keep_indices=(dim,))

    x_axis = marginal_grid.axes[0]
    y_values = marginal_grid.values

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x_axis, y_values, **kwargs)
    ax.grid(True, alpha=0.3)

    return ax


def display_animation_html(anim: Any) -> HTML:
    """
    Standard helper to render animations in notebooks.

    Args:
        anim: The Matplotlib FuncAnimation object.

    Returns:
        IPython HTML object for display.
    """
    print("Rendering animation...")
    return HTML(anim.to_jshtml())
