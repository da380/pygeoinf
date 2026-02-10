"""
This module implements a simple 1D wave equation defined on a circle.

The solver uses a spectral method for homogeneous material properties (constant
density and rigidity), which is highly efficient and accurate. For spatially
varying properties, it falls back to a pseudo-spectral method coupled with a
high-precision numerical integrator from SciPy.
"""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy import pi

from IPython.display import HTML

from scipy.sparse.linalg import LinearOperator as ScipyOperator
from scipy.integrate import solve_ivp
from scipy.fft import rfft, irfft


from .utils import is_sorted, is_1d


class CircleWave:
    """
    Class for solving the wave equation on a circle using spectral or
    pseudo-spectral methods.

    The state of the system is represented by a vector containing the
    displacement and momentum at each point on the spatial grid.
    """

    def __init__(
        self,
        kmax: int,
        /,
        *,
        radius: float = 1.0,
        density: float | Callable[[float], float] = 1.0,
        rigidity: float | Callable[[float], float] = 1.0,
    ):
        """
        Args:
            kmax: Maximum degree for Fourier expansions.
            radius: Radius of the circle. Default is 1.0.
            density: Density of the material. Can be a constant float or a
                function of the angle theta. Default is 1.0.
            rigidity: Rigidity of the material. Can be a constant float or a
                function of the angle theta. Default is 1.0.
        """

        self._kmax = kmax
        self._npoints: int = 2 * kmax
        self._angles: np.ndarray = np.linspace(0, 2 * pi, self.npoints, endpoint=False)
        self._radius: float = radius

        self._density: float | Callable[[float], float] = density
        self._rigidity: float | Callable[[float], float] = rigidity

        self._density_constant = isinstance(self._density, float | int)
        self._rigidity_constant = isinstance(self._rigidity, float | int)

        if self.density_constant:
            self._density_values = np.full((self.npoints), self._density)
        else:
            self._density_values = np.fromiter(
                (self._density(angle) for angle in self.angles), dtype=float
            )

        if self.rigidity_constant:
            self._rigidity_values = np.full((self.npoints), self._rigidity)
        else:
            self._rigidity_values = np.fromiter(
                (self._rigidity(angle) for angle in self.angles), dtype=float
            )

        self._wavenumbers = np.fromiter(
            (k for k in range(self.npoints // 2 + 1)), dtype=int
        )

        self._derivative_scaling = 1j * self._wavenumbers / self._radius
        self._laplacian_scaling = -self._wavenumbers**2 / self._radius**2

        # Pre-compute angular frequencies for the homogeneous case
        self._omega = None
        self._omega_inv = None
        if self.density_and_rigidity_constant:
            rho = self._density
            tau = self._rigidity
            self._omega = self._wavenumbers / self.radius * np.sqrt(tau / rho)

            self._omega_inv = np.zeros_like(self._omega)
            non_zero_mask = self._omega != 0
            self._omega_inv[non_zero_mask] = 1.0 / self._omega[non_zero_mask]

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def npoints(self) -> int:
        """Returns the number of spatial points."""
        return self._npoints

    @property
    def radius(self) -> float:
        """Returns the radius of the circle."""
        return self._radius

    @property
    def angles(self) -> np.ndarray:
        """Returns the angles for the computational mesh."""
        return self._angles

    @property
    def density_constant(self) -> bool:
        """Returns true if the density is constant."""
        return self._density_constant

    @property
    def rigidity_constant(self) -> bool:
        """Returns true if the rigidity is constant."""
        return self._rigidity_constant

    @property
    def density_and_rigidity_constant(self) -> bool:
        """Returns true if both density and rigidity are constant."""
        return self.density_constant and self.rigidity_constant

    @property
    def density_values(self) -> np.ndarray:
        """Returns the density at the angles used in the computational mesh."""
        return self._density_values

    @property
    def rigidity_values(self) -> np.ndarray:
        """Returns the rigidity at the angles used in the computational mesh."""
        return self._rigidity_values

    # --------------------------------------------------------------------------
    # Public Methods
    # --------------------------------------------------------------------------

    def zero_state_vector(self) -> np.ndarray:
        """Returns a state vector of zeros."""
        return np.zeros(2 * self.npoints)

    def project_displacement_and_momentum(
        self, displacement: Callable[[float], float], momentum: Callable[[float], float]
    ) -> np.ndarray:
        """
        Projects continuous displacement and momentum functions onto the
        discrete state vector.
        """
        displacement_vector = np.fromiter(
            (displacement(angle) for angle in self.angles), dtype=float
        )
        momentum_vector = np.fromiter(
            (momentum(angle) for angle in self.angles), dtype=float
        )
        return np.concatenate((displacement_vector, momentum_vector))

    def plot_state(self, state_vector: np.ndarray, /, *, fig=None, ax=None):
        """
        Given a state vector, plots the displacement and momentum on separate axes.
        """
        displacement = state_vector[: self.npoints]
        momentum = state_vector[self.npoints :]

        if ax is None:
            if fig is None:
                fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            else:
                ax = fig.subplots(2, 1, sharex=True)
        else:
            fig = ax[0].get_figure()

        ax[0].plot(self.angles, displacement, color="royalblue")
        ax[0].set_ylabel("Displacement")
        ax[0].grid(True, linestyle=":")

        ax[1].plot(self.angles, momentum, color="coral")
        ax[1].set_ylabel("Momentum")
        ax[1].set_xlabel("Angle (radians)")
        ax[1].grid(True, linestyle=":")

        fig.tight_layout()
        return fig, ax

    def compute_derivative(self, state_vector: np.ndarray) -> np.ndarray:
        """
        Computes the time derivative of the state vector (dq/dt, dp/dt).

        This method uses a pure spectral approach for homogeneous materials
        and a pseudo-spectral approach for inhomogeneous materials.
        """
        displacement = state_vector[: self.npoints]
        momentum = state_vector[self.npoints :]

        if self.density_constant:
            displacement_dot = momentum / self._density
        else:
            displacement_dot = momentum / self._density_values

        displacement_fourier = rfft(displacement)
        if self.rigidity_constant:
            momentum_dot_fourier = (
                self._rigidity * self._laplacian_scaling * displacement_fourier
            )
        else:
            grad_disp_fourier = self._derivative_scaling * displacement_fourier
            stress = irfft(grad_disp_fourier, n=self.npoints) * self.rigidity_values
            stress_fourier = rfft(stress)
            momentum_dot_fourier = self._derivative_scaling * stress_fourier

        momentum_dot = irfft(momentum_dot_fourier, n=self.npoints)

        return np.concatenate((displacement_dot, momentum_dot))

    def integrate(self, initial_state: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Integrates the wave equation to find the state at specified times.

        For homogeneous materials, this method uses the highly efficient and
        accurate analytical solution in the Fourier domain. For inhomogeneous
        materials, it falls back to a high-precision numerical integrator.

        Args:
            initial_state: The state vector at the first time point.
            times: A 1D sorted array of time points to evaluate the solution at.

        Returns:
            An array where each column is the state vector at the corresponding time.
        """
        if not is_1d(times):
            raise ValueError("Times must be input as a 1D array")
        if not is_sorted(times):
            raise ValueError("Times must be sorted")

        if self.density_and_rigidity_constant:
            results = np.zeros((2 * self.npoints, len(times)))
            t0 = times[0]

            q0, p0 = initial_state[: self.npoints], initial_state[self.npoints :]
            q0_hat, p0_hat = rfft(q0), rfft(p0)

            for i, t in enumerate(times):
                dt = t - t0
                q_hat, p_hat = self._evolve_fourier(q0_hat, p0_hat, dt)

                q = irfft(q_hat, n=self.npoints)
                p = irfft(p_hat, n=self.npoints)
                results[:, i] = np.concatenate((q, p))

            return results
        else:
            integrator_options = {"method": "DOP853", "rtol": 1e-12, "atol": 1e-12}
            t0, t1 = times[0], times[-1]
            sol = solve_ivp(
                lambda t, z: self.compute_derivative(z),
                (t0, t1),
                initial_state,
                t_eval=times,
                **integrator_options,
            )
            return sol.y

    def propagator(self, t0: float, t1: float) -> ScipyOperator:
        """
        Returns the propagator as a SciPy LinearOperator.

        This operator evolves the state vector from time t0 to t1. For
        homogeneous materials, this method uses an exact analytical solution.
        For inhomogeneous materials, it uses a high-precision numerical integrator.
        """
        shape = (2 * self.npoints, 2 * self.npoints)

        if self.density_and_rigidity_constant:

            def analytical_propagate(z_start, time_step):
                q0, p0 = z_start[: self.npoints], z_start[self.npoints :]
                q0_hat, p0_hat = rfft(q0), rfft(p0)
                q1_hat, p1_hat = self._evolve_fourier(q0_hat, p0_hat, time_step)
                q1 = irfft(q1_hat, n=self.npoints)
                p1 = irfft(p1_hat, n=self.npoints)
                return np.concatenate((q1, p1))

            def matvec(z0):
                return analytical_propagate(z0, t1 - t0)

            def rmatvec(z0):
                q0, p0 = z0[: self.npoints], z0[self.npoints :]
                J_inv_z0 = np.concatenate((-p0, q0))
                z_backward = analytical_propagate(J_inv_z0, t0 - t1)
                q_b, p_b = z_backward[: self.npoints], z_backward[self.npoints :]
                return np.concatenate((p_b, -q_b))

            return ScipyOperator(shape, matvec=matvec, rmatvec=rmatvec, dtype=float)

        else:
            integrator_options = {"method": "DOP853", "rtol": 1e-12, "atol": 1e-12}

            def matvec(z0):
                sol = solve_ivp(
                    lambda t, z: self.compute_derivative(z),
                    (t0, t1),
                    z0,
                    t_eval=[t1],
                    **integrator_options,
                )
                return sol.y[:, 0]

            def rmatvec(z0):
                q0, p0 = z0[: self.npoints], z0[self.npoints :]
                Jz0 = np.concatenate((p0, -q0))
                sol = solve_ivp(
                    lambda t, z: self.compute_derivative(z),
                    (t1, t0),
                    Jz0,
                    t_eval=[t0],
                    **integrator_options,
                )
                z1 = sol.y[:, 0]
                q1, p1 = z1[: self.npoints], z1[self.npoints :]
                return np.concatenate((-p1, q1))

            return ScipyOperator(shape, matvec=matvec, rmatvec=rmatvec, dtype=float)

    def animate_solution(
        self,
        initial_state: np.ndarray,
        t_span: tuple[float, float],
        /,
        *,
        n_frames: int = 200,
        slowdown_factor: float = 1.0,
    ) -> FuncAnimation:
        """
        Computes and creates an animation object of the wave's evolution.
        """
        times = np.linspace(t_span[0], t_span[1], n_frames)
        dt = times[1] - times[0]
        solution_history = self.integrate(initial_state, times)

        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.tight_layout(pad=3.0)

        disp_history = solution_history[: self.npoints, :]
        mom_history = solution_history[self.npoints :, :]

        (line_q,) = ax[0].plot(self.angles, disp_history[:, 0], color="royalblue")
        (line_p,) = ax[1].plot(self.angles, mom_history[:, 0], color="coral")
        time_text = ax[0].text(
            0.02, 0.95, "", transform=ax[0].transAxes, ha="left", va="top", fontsize=12
        )

        q_min, q_max = disp_history.min(), disp_history.max()
        p_min, p_max = mom_history.min(), mom_history.max()
        q_margin = 0.1 * (q_max - q_min) if (q_max - q_min) > 1e-9 else 0.1
        p_margin = 0.1 * (p_max - p_min) if (p_max - p_min) > 1e-9 else 0.1

        ax[0].set_ylim(q_min - q_margin, q_max + q_margin)
        ax[1].set_ylim(p_min - p_margin, p_max + p_margin)

        ax[0].set_ylabel("Displacement")
        ax[0].grid(True, linestyle=":")
        ax[1].set_ylabel("Momentum")
        ax[1].set_xlabel("Angle (radians)")
        ax[1].grid(True, linestyle=":")

        def update(frame):
            line_q.set_ydata(disp_history[:, frame])
            line_p.set_ydata(mom_history[:, frame])
            time_text.set_text(f"Time = {times[frame]:.2f} s")
            return line_q, line_p, time_text

        interval_ms = dt * 1000 * slowdown_factor
        anim = FuncAnimation(
            fig, update, frames=n_frames, interval=interval_ms, blit=True
        )

        plt.close(fig)
        return anim

    def animate_ensemble(
        self,
        initial_states: np.ndarray,
        t_span: tuple[float, float],
        /,
        *,
        n_frames: int = 200,
        slowdown_factor: float = 1.0,
        alpha: float = 0.1,
        color_q: str = "royalblue",
        color_p: str = "coral",
    ) -> FuncAnimation:
        """
        Computes and creates an animation of a wave ensemble's evolution.

        Each initial state in the list is integrated, and all solutions are
        plotted simultaneously with transparency to visualize the distribution.

        Args:
            initial_states: A 2D numpy array where each row is an initial
                state vector for a member of the ensemble.
            t_span: The start and end time for the simulation, e.g., (0.0, 10.0).
            n_frames: The number of frames to generate for the animation.
            slowdown_factor: A factor to slow down the animation playback.
            alpha: The transparency level for the plotted lines.
            color_q: The color for the displacement lines.
            color_p: The color for the momentum lines.

        Returns:
            A matplotlib FuncAnimation object.
        """

        if initial_states.ndim != 2 or initial_states.shape[1] != 2 * self.npoints:
            raise ValueError(
                f"initial_states must be a 2D array of shape (n_samples, {2 * self.npoints})"
            )

        n_samples = initial_states.shape[0]
        times = np.linspace(t_span[0], t_span[1], n_frames)
        dt = times[1] - times[0]

        # --- 1. Solve for all trajectories ---
        print(f"Integrating {n_samples} trajectories for the ensemble animation...")
        all_solutions = []
        for i in range(n_samples):
            solution_history = self.integrate(initial_states[i, :], times)
            all_solutions.append(solution_history)

        # --- 2. Set up the plot ---
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.tight_layout(pad=3.0)

        # Find global min/max for y-limits across all solutions and times
        all_displacements = np.array([sol[: self.npoints, :] for sol in all_solutions])
        all_momenta = np.array([sol[self.npoints :, :] for sol in all_solutions])

        q_min, q_max = all_displacements.min(), all_displacements.max()
        p_min, p_max = all_momenta.min(), all_momenta.max()
        q_margin = 0.1 * (q_max - q_min) if (q_max - q_min) > 1e-9 else 0.1
        p_margin = 0.1 * (p_max - p_min) if (p_max - p_min) > 1e-9 else 0.1
        ax[0].set_ylim(q_min - q_margin, q_max + q_margin)
        ax[1].set_ylim(p_min - p_margin, p_max + p_margin)

        ax[0].set_ylabel("Displacement")
        ax[0].grid(True, linestyle=":")
        ax[1].set_ylabel("Momentum")
        ax[1].set_xlabel("Angle (radians)")
        ax[1].grid(True, linestyle=":")

        # Explicitly set the x-axis to show the full circle
        ax[1].set_xlim(0, 2 * np.pi)

        time_text = ax[0].text(
            0.02, 0.95, "", transform=ax[0].transAxes, ha="left", va="top", fontsize=12
        )

        # Create a line for each member of the ensemble
        lines_q = [
            ax[0].plot([], [], color=color_q, alpha=alpha, lw=1)[0]
            for _ in range(n_samples)
        ]
        lines_p = [
            ax[1].plot([], [], color=color_p, alpha=alpha, lw=1)[0]
            for _ in range(n_samples)
        ]

        # --- 3. Define the animation update function ---
        def update(frame):
            # Update each line in the ensemble
            for i in range(n_samples):
                disp_history = all_solutions[i][: self.npoints, :]
                mom_history = all_solutions[i][self.npoints :, :]
                lines_q[i].set_data(self.angles, disp_history[:, frame])
                lines_p[i].set_data(self.angles, mom_history[:, frame])

            time_text.set_text(f"Time = {times[frame]:.2f} s")
            # Return all artists that have changed
            return lines_q + lines_p + [time_text]

        # --- 4. Create and return the animation object ---
        interval_ms = dt * 1000 * slowdown_factor
        anim = FuncAnimation(
            fig, update, frames=n_frames, interval=interval_ms, blit=True
        )

        plt.close(fig)
        return anim

    def get_animation_html(self, anim: FuncAnimation) -> HTML:
        """
        Converts an animation object to an HTML5 video for embedding in
        a Jupyter notebook.
        """
        return HTML(anim.to_html5_video())

    # --------------------------------------------------------------------------
    # Private Methods
    # --------------------------------------------------------------------------

    def _evolve_fourier(
        self, q0_hat: np.ndarray, p0_hat: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evolves the Fourier coefficients of the state by a time step dt.
        """
        rho = self._density

        cos_vals = np.cos(self._omega * dt)
        sin_vals = np.sin(self._omega * dt)

        q_hat = cos_vals * q0_hat + sin_vals * self._omega_inv / rho * p0_hat
        p_hat = -sin_vals * self._omega * rho * q0_hat + cos_vals * p0_hat

        if self.npoints > 0:
            q_hat[0] = q0_hat[0] + dt / rho * p0_hat[0]
            p_hat[0] = p0_hat[0]

        return q_hat, p_hat
