"""
core.py

A dimension-agnostic engine for numerical integration, statistical analysis,
Bayesian inference, and generic visualisation.
"""

from typing import Callable, List, Tuple, Union, Optional, Any, Dict
from abc import ABC, abstractmethod

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


# --- Abstract Base Class for Assimilation ---


class AssimilationEngine(ABC):
    """
    Abstract Base Class defining the interface for all assimilation methods
    (Grid, KF, EnKF). Enforces a consistent 'setup -> observe -> run' workflow.
    """

    def __init__(self):
        self.observations: List[Tuple[float, Any]] = []

    def add_observation(
        self,
        time: float,
        covariance: np.ndarray,
        value: Optional[np.ndarray] = None,
        operator: Optional[Union[np.ndarray, Callable]] = None,
    ):
        """
        Registers an observation to be assimilated during the run.
        """
        # Default implementation for Gaussian/Linear obs (can be overridden)
        if operator is None or callable(operator):
            model = GaussianLikelihood(value, covariance, operator)
        else:
            model = LinearGaussianLikelihood(value, covariance, operator)

        self.observations.append((time, model))
        self.observations.sort(key=lambda x: x[0])

    @abstractmethod
    def run(
        self, initial_state: Any, t_final: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes the assimilation cycle.
        Must return a history list of dicts containing 'time', 'forecast', 'analysis'.
        """
        pass

    @abstractmethod
    def reanalyse_initial_condition(self, history: List[Dict[str, Any]]) -> Any:
        """
        Performs Reanalysis (Smoothing) to estimate the state at t=0
        given all observations up to t_final.
        """
        pass


# --- Bayesian Assimilation Manager ---


class BayesianAssimilationProblem(AssimilationEngine):
    """
    Manages the definition and execution of a Grid-based Bayesian assimilation cycle.
    Inherits observation management from AssimilationEngine.
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
        super().__init__()  # Initialises self.observations = []
        self.eom_func = eom_func
        self.eom_args = eom_args

    def generate_synthetic_data(
        self,
        true_initial_condition: np.ndarray,
        dt_render: float = 0.05,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs the physics from t=0, samples noisy observations at registered times,
        updates the internal Likelihood models with these values, and returns ground truth.
        """
        if seed is not None:
            np.random.seed(seed)

        # 1. Identify observation times
        obs_times = np.array([t for t, _ in self.observations])

        if len(obs_times) == 0:
            raise ValueError("No observations registered to generate data for.")

        # 2. Run High-Res Simulation (Ground Truth)
        t_max = obs_times[-1]
        t_render = np.arange(0, t_max + dt_render, dt_render)
        if t_render[-1] < t_max:
            t_render = np.append(t_render, t_max)

        all_times = np.unique(np.concatenate([t_render, obs_times]))

        sol_all = solve_trajectory(
            self.eom_func, true_initial_condition, all_times, args=self.eom_args
        )

        # 3. Iterate through registered observations and update them
        for i, (t_obs, model) in enumerate(self.observations):
            # Find the state at this exact time
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
        self, initial_state: ProbabilityGrid, t_final: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes the assimilation cycle starting from t=0.

        Args:
            initial_state: The ProbabilityGrid at t=0 (Prior).
            t_final: Optional end time.

        Returns:
            A list of dictionaries containing {'time', 'forecast', 'analysis', 'evidence'}.
        """
        history = []
        current_grid = initial_state
        t_current = 0.0

        for t_obs, lik_model in self.observations:
            dt = t_obs - t_current

            if dt < 0:
                raise ValueError(
                    f"Observation at t={t_obs} is in the past relative to t={t_current}."
                )

            # 1. Forecast Step (Advection)
            if dt > 0:
                forecast_grid = current_grid.push_forward(
                    self.eom_func, dt, self.eom_args
                )
            else:
                forecast_grid = current_grid

            # 2. Analysis Step (Bayesian Update)
            lik_grid = lik_model.evaluate(forecast_grid)
            analysis_grid, evidence = forecast_grid.bayes_update(lik_grid)

            history.append(
                {
                    "time": t_obs,
                    "forecast": forecast_grid,
                    "analysis": analysis_grid,
                    "evidence": evidence,
                }
            )

            current_grid = analysis_grid
            t_current = t_obs

        # Optional: Final forecast
        if t_final is not None and t_final > t_current:
            dt = t_final - t_current
            final_grid = current_grid.push_forward(self.eom_func, dt, self.eom_args)
            history.append(
                {
                    "time": t_final,
                    "forecast": final_grid,
                    "analysis": final_grid,
                    "evidence": 1.0,
                }
            )

        return history

    def reanalyse_initial_condition(
        self, history: List[Dict[str, Any]]
    ) -> ProbabilityGrid:
        """
        Pulls the final posterior back to t=0 using inverse advection.
        Returns the smoothed ProbabilityGrid at t=0.
        """
        final_posterior = history[-1]["analysis"]
        t_final = history[-1]["time"]

        # Inverse advection (negative time)
        smoothed_grid = final_posterior.push_forward(
            self.eom_func, -t_final, self.eom_args
        )
        return smoothed_grid


# ---- Linear KF class ----


class LinearKalmanFilter(AssimilationEngine):
    """
    Deterministic Linear Kalman Filter.
    Assumes perfect physics (Process Noise Q = 0).

    System Model:
      x_k = F_k * x_{k-1}  (Deterministic)
      y_k = H * x_k + v_k,  v_k ~ N(0, R)

    Because the physics are deterministic, we can exactly reconstruct the
    smoothed initial condition by inverting the final state.
    """

    def __init__(
        self,
        transition_matrix_func: Callable[[float], np.ndarray],
    ):
        """
        Args:
            transition_matrix_func: Function f(dt) -> F matrix (StateDim, StateDim).
                                    Returns the propagator for a time step of dt.
        """
        super().__init__()
        self.get_F = transition_matrix_func

    def _predict(
        self, mean: np.ndarray, cov: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction Step: x = Fx, P = FPF'
        (No process noise Q added)
        """
        F = self.get_F(dt)

        # 1. Predict Mean
        mean_pred = F @ mean

        # 2. Predict Covariance
        cov_pred = F @ cov @ F.T

        return mean_pred, cov_pred

    def _update(
        self, mean: np.ndarray, cov: np.ndarray, likelihood: LinearGaussianLikelihood
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analysis Step: Standard Kalman Update.
        """
        H = likelihood.H_mat
        y_obs = likelihood.y_obs
        R = likelihood.R

        if y_obs is None:
            raise ValueError("Observation value cannot be None during KF update.")

        # 1. Innovation
        y_pred = H @ mean
        y_residual = y_obs - y_pred

        # 2. Innovation Covariance
        S = H @ cov @ H.T + R

        # 3. Kalman Gain
        try:
            # K = P H' S^-1
            K_transposed = np.linalg.solve(S, H @ cov)
            K = K_transposed.T
        except np.linalg.LinAlgError:
            print("Warning: Singular innovation covariance. Using pseudo-inverse.")
            K = cov @ H.T @ np.linalg.pinv(S)

        # 4. Update
        mean_new = mean + K @ y_residual

        # 5. Covariance Update
        I = np.eye(len(mean))
        cov_new = (I - K @ H) @ cov

        return mean_new, cov_new

    def run(
        self,
        initial_state: Tuple[np.ndarray, np.ndarray],
        t_final: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes the Deterministic KF.
        Args:
            initial_state: Tuple (mean_0, cov_0)
        """
        history = []
        current_mean, current_cov = initial_state
        t_current = 0.0

        for t_obs, lik_model in self.observations:
            if not isinstance(lik_model, LinearGaussianLikelihood):
                raise TypeError(
                    "LinearKalmanFilter only supports LinearGaussianLikelihood."
                )

            dt = t_obs - t_current

            # 1. Forecast
            if dt > 0:
                f_mean, f_cov = self._predict(current_mean, current_cov, dt)
            else:
                f_mean, f_cov = current_mean, current_cov

            # 2. Analysis
            a_mean, a_cov = self._update(f_mean, f_cov, lik_model)

            history.append(
                {
                    "time": t_obs,
                    "forecast": (f_mean, f_cov),
                    "analysis": (a_mean, a_cov),
                }
            )

            current_mean, current_cov = a_mean, a_cov
            t_current = t_obs

        # Final Forecast
        if t_final is not None and t_final > t_current:
            dt = t_final - t_current
            f_mean, f_cov = self._predict(current_mean, current_cov, dt)
            history.append(
                {
                    "time": t_final,
                    "forecast": (f_mean, f_cov),
                    "analysis": (f_mean, f_cov),
                }
            )

        return history

    def reanalyse_initial_condition(
        self, history: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Back-propagates the final estimate to t=0 by inverting the physics.
        Exact for deterministic systems.
        """
        final_mean, final_cov = history[-1]["analysis"]
        t_final = history[-1]["time"]

        # Compute propagator for the full duration: x_T = F(T) * x_0
        F_total = self.get_F(t_final)

        # Invert State: x_0 = F(T)^-1 * x_T
        smoothed_mean = np.linalg.solve(F_total, final_mean)

        # Invert Covariance: P_0 = F^-1 * P_T * F^-T
        F_inv = np.linalg.inv(F_total)
        smoothed_cov = F_inv @ final_cov @ F_inv.T

        return smoothed_mean, smoothed_cov


# ---- Ensemble Kalman filter class ----


class EnsembleKalmanFilter(AssimilationEngine):
    """
    Stochastic Ensemble Kalman Filter (EnKF).
    Suitable for non-linear dynamics. Uses the "Perturbed Observation" method.

    System Model:
      x_k = f(x_{k-1})  (Deterministic or Stochastic)
      y_k = H(x_k) + v_k
    """

    def __init__(
        self,
        eom_func: Callable[[float, np.ndarray, Any], np.ndarray],
        eom_args: Tuple = (),
        n_ensemble: int = 50,
    ):
        """
        Args:
            eom_func: The non-linear physics ODE function.
            eom_args: Arguments for the physics.
            n_ensemble: Number of particles to simulate.
        """
        super().__init__()
        self.eom_func = eom_func
        self.eom_args = eom_args
        self.n_ensemble = n_ensemble

    def _predict(self, ensemble: np.ndarray, dt: float) -> np.ndarray:
        """
        Propagates the full ensemble forward by dt using the non-linear physics.
        """
        # solve_ensemble returns shape (N_samples, N_dim, N_time)
        # We simulate from 0 to dt
        traj = solve_ensemble(
            self.eom_func, ensemble, np.array([0, dt]), args=self.eom_args
        )
        # Return the state at the end time (index -1)
        return traj[:, :, -1]

    def _update(
        self, ensemble: np.ndarray, likelihood: GaussianLikelihood
    ) -> np.ndarray:
        """
        Analysis Step: EnKF Update with perturbed observations.
        """
        n_ens, n_dim = ensemble.shape
        y_obs = likelihood.y_obs
        R = likelihood.R

        if y_obs is None:
            raise ValueError("Observation value cannot be None during EnKF update.")

        # 1. Perturb Observations
        # We generate a cloud of noisy observations centered on the actual data
        # v ~ N(0, R)
        noise = np.random.multivariate_normal(np.zeros(len(y_obs)), R, size=n_ens)
        Y_perturbed = y_obs + noise  # Shape (N_ens, ObsDim)

        # 2. Forecast Observations (H(x))
        # Map every particle to observation space (handles non-linear H)
        H_x = likelihood.H(ensemble)  # Shape (N_ens, ObsDim)

        # 3. Calculate Sample Covariances (via Anomalies)
        # Mean centering
        x_mean = np.mean(ensemble, axis=0)
        Hx_mean = np.mean(H_x, axis=0)

        # Anomaly matrices (StateDim x N_ens) and (ObsDim x N_ens)
        A = (ensemble - x_mean).T
        Y = (H_x - Hx_mean).T

        # 4. Kalman Gain Construction
        # P_xy = 1/(N-1) * A @ Y.T
        # P_yy = 1/(N-1) * Y @ Y.T
        # K = P_xy * (P_yy + R)^-1

        # Note: The denominator in EnKF is typically (P_yy + R).
        # Since we use perturbed observations, R is implicitly handled if we just use
        # the covariance of the innovations, but the standard robust form adds R explicitly.

        factor = 1.0 / (n_ens - 1)
        P_yy = factor * (Y @ Y.T)
        S = P_yy + R

        P_xy = factor * (A @ Y.T)

        # Solve K = P_xy @ S^-1
        try:
            K_transposed = np.linalg.solve(S, P_xy.T)
            K = K_transposed.T
        except np.linalg.LinAlgError:
            K = P_xy @ np.linalg.pinv(S)

        # 5. Update State
        # Innovation for each particle: y_perturbed_i - H(x_i)
        innovations = (Y_perturbed - H_x).T  # (ObsDim, N_ens)

        # Update: x_new = x_old + K * innovation
        update = K @ innovations  # (StateDim, N_ens)

        return ensemble + update.T

    def run(
        self,
        initial_state: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
        t_final: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes the EnKF.

        Args:
            initial_state: Either a tuple (mean, cov) to sample from,
                           OR an explicit ensemble array (N_ens, N_dim).
        """
        # 1. Initialize Ensemble
        if isinstance(initial_state, tuple):
            mean, cov = initial_state
            current_ensemble = np.random.multivariate_normal(
                mean, cov, size=self.n_ensemble
            )
        else:
            current_ensemble = initial_state

        history = []
        t_current = 0.0

        for t_obs, lik_model in self.observations:
            dt = t_obs - t_current

            # 1. Forecast
            if dt > 0:
                f_ens = self._predict(current_ensemble, dt)
            else:
                f_ens = current_ensemble

            # 2. Analysis
            a_ens = self._update(f_ens, lik_model)

            # Store Statistics (Mean, Cov) for history, rather than full ensemble
            # (Saves memory and keeps interface consistent with KF)
            f_stats = (np.mean(f_ens, axis=0), np.cov(f_ens, rowvar=False))
            a_stats = (np.mean(a_ens, axis=0), np.cov(a_ens, rowvar=False))

            history.append(
                {
                    "time": t_obs,
                    "forecast": f_stats,
                    "analysis": a_stats,
                    "ensemble_analysis": a_ens,  # Keep full final ensemble if needed
                }
            )

            current_ensemble = a_ens
            t_current = t_obs

        # Final Forecast
        if t_final is not None and t_final > t_current:
            dt = t_final - t_current
            f_ens = self._predict(current_ensemble, dt)
            f_stats = (np.mean(f_ens, axis=0), np.cov(f_ens, rowvar=False))

            history.append(
                {
                    "time": t_final,
                    "forecast": f_stats,
                    "analysis": f_stats,
                    "ensemble_analysis": f_ens,
                }
            )

        return history

    def reanalyse_initial_condition(
        self, history: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reanalysis for deterministic physics.
        Takes the MEAN of the final analysis ensemble and integrates it BACKWARDS.
        """
        # Get final ensemble mean
        final_mean, final_cov = history[-1]["analysis"]
        t_final = history[-1]["time"]

        # Define Reverse Physics
        def reverse_eom(t, y, *args):
            return -1.0 * np.array(self.eom_func(t, y, *args))

        # Integrate backwards: x_T -> x_0
        sol = solve_trajectory(
            self.eom_func,  # We use normal physics but negative time in solve_trajectory logic
            final_mean,
            np.array([t_final, 0.0]),  # Backwards time span
            args=self.eom_args,
        )

        smoothed_mean = sol[:, -1]

        # Note: Back-propagating covariance for EnKF is complex.
        # For this simple course, we simply return the propagated mean
        # and a placeholder covariance (or the initial covariance if available).
        # We return Zeros for covariance to indicate it wasn't computed.
        smoothed_cov = np.zeros_like(final_cov)

        return smoothed_mean, smoothed_cov


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


# --- Visualisation (Kalman Filters) ---


def plot_gaussian_ellipsoid(
    mean: np.ndarray,
    cov: np.ndarray,
    ax: plt.Axes,
    n_std: float = 2.0,
    dims: Tuple[int, int] = (0, 1),
    **kwargs,
):
    """
    Plots a 2D covariance ellipse representing the Gaussian distribution.

    Args:
        mean: Mean vector (N,).
        cov: Covariance matrix (N, N).
        ax: Matplotlib axes.
        n_std: Number of standard deviations for the ellipse radius.
        dims: Tuple of (x_dim, y_dim) indices.
        **kwargs: Passed to ax.plot (color, linestyle, etc).
    """
    # Extract 2D slice
    idx = list(dims)
    mean_2d = mean[idx]
    cov_2d = cov[np.ix_(idx, idx)]

    # Calculate Eigenvalues and Eigenvectors
    vals, vecs = np.linalg.eigh(cov_2d)

    # Order them by magnitude (largest first)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # Calculate Angle of Rotation
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and Height (eigenvalues are variance, so sqrt for std)
    width, height = 2 * n_std * np.sqrt(vals)

    # Create Ellipse Patch
    from matplotlib.patches import Ellipse

    ell = Ellipse(
        xy=mean_2d, width=width, height=height, angle=theta, fill=False, **kwargs
    )

    ax.add_patch(ell)
    # Plot center mean
    ax.plot(mean_2d[0], mean_2d[1], marker="+", markersize=10, **kwargs)


def plot_kf_step(
    forecast_stats: Tuple[np.ndarray, np.ndarray],
    analysis_stats: Tuple[np.ndarray, np.ndarray],
    dims: Tuple[int, int] = (0, 1),
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
):
    """
    Visualises a single Kalman Filter step (Forecast vs Analysis).
    Plots both Gaussian ellipsoids (2-sigma).

    Args:
        forecast_stats: (mean, cov) tuple for the prior.
        analysis_stats: (mean, cov) tuple for the posterior.
        dims: Dimensions to plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Unpack
    f_mean, f_cov = forecast_stats
    a_mean, a_cov = analysis_stats

    # Plot Forecast (Blue, Dashed)
    plot_gaussian_ellipsoid(
        f_mean,
        f_cov,
        ax,
        n_std=2.0,
        dims=dims,
        color="blue",
        linestyle="--",
        label="Forecast (Prior)",
    )

    # Plot Analysis (Red, Solid)
    plot_gaussian_ellipsoid(
        a_mean,
        a_cov,
        ax,
        n_std=2.0,
        dims=dims,
        color="red",
        linestyle="-",
        label="Analysis (Posterior)",
    )

    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(f"State Dimension {dims[0]}")
    ax.set_ylabel(f"State Dimension {dims[1]}")
    if title:
        ax.set_title(title)

    return ax


def plot_tracker_1d(
    history: List[Dict[str, Any]],
    dim: int = 0,
    ground_truth: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Plots the time-evolution of a single state variable with uncertainty bounds.

    Args:
        history: The output list from kf.run().
        dim: Index of the state dimension to plot.
        ground_truth: Optional dict from generate_synthetic_data to compare against.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Extract Time Series
    times = np.array([step["time"] for step in history])

    # Extract Analysis Means and StdDevs
    means = np.array([step["analysis"][0][dim] for step in history])
    covs = [step["analysis"][1] for step in history]
    stds = np.array([np.sqrt(c[dim, dim]) for c in covs])

    # Plot Estimate
    ax.plot(times, means, "k-", label="KF Estimate")

    # Plot Uncertainty (2-sigma shaded region)
    ax.fill_between(
        times,
        means - 2 * stds,
        means + 2 * stds,
        color="gray",
        alpha=0.3,
        label=r"$\pm 2\sigma$ Uncertainty",
    )

    # Plot Ground Truth if available
    if ground_truth:
        t_true = ground_truth["t_ground_truth"]
        x_true = ground_truth["state_ground_truth"][dim]
        ax.plot(t_true, x_true, "g--", lw=1, label="Ground Truth")

    ax.set_xlabel("Time")
    ax.set_ylabel(f"State Dimension {dim}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax
