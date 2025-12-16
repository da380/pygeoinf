"""
Experiment runner for Demo 1 multi-component inference.

This module implements the Demo1Experiment class that automates running
the complete inference pipeline from demo_1.ipynb, including model setup,
data generation, prior construction, inference, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import json
import time
from datetime import datetime

from pygeoinf.interval import (
    Interval,
    LebesgueSpace,
    EuclideanSpace,
    HilbertSpaceDirectSum,
    BesselSobolevPrior,
    GaussianMeasure,
    RowLinearOperator,
    LinearBayesian,
)
from pygeoinf.linear_operators import LinearOperator

from .demo_1_config import Demo1Config


class Demo1Experiment:
    """Run a single instance of the Demo 1 multi-component inference experiment.

    This class encapsulates all steps from demo_1.ipynb:
    1. Set up model space (3 function spaces + 2 Euclidean spaces)
    2. Create forward operator G and target operator T
    3. Generate synthetic data from true parameters
    4. Set up prior (5 independent Gaussian measures)
    5. Run Bayesian inference
    6. Compute metrics and generate visualizations
    7. Save all results to disk

    The experiment can be configured via a Demo1Config object, allowing
    systematic parameter sweeps and reproducibility.
    """

    def __init__(self, config: Demo1Config, output_dir: Path):
        """Initialize experiment with configuration.

        Args:
            config: Demo1Config object specifying all parameters
            output_dir: Directory to save results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed if specified
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        # Storage for results
        self.timings: Dict[str, float] = {}
        self.metrics: Dict[str, Any] = {}

        # Model components (set during setup)
        self.interval: Optional[Interval] = None
        self.M_vp: Optional[LebesgueSpace] = None
        self.M_vs: Optional[LebesgueSpace] = None
        self.M_rho: Optional[LebesgueSpace] = None
        self.M_sigma_0: Optional[EuclideanSpace] = None
        self.M_sigma_1: Optional[EuclideanSpace] = None
        self.M_model: Optional[HilbertSpaceDirectSum] = None
        self.M_data: Optional[EuclideanSpace] = None

        # Operators
        self.G: Optional[LinearOperator] = None
        self.T: Optional[LinearOperator] = None

        # Data
        self.u_true: Optional[np.ndarray] = None
        self.d_obs: Optional[np.ndarray] = None
        self.d_true: Optional[np.ndarray] = None

        # Prior
        self.prior: Optional[GaussianMeasure] = None

        # Posterior
        self.posterior: Optional[GaussianMeasure] = None
        self.posterior_mean: Optional[np.ndarray] = None
        self.posterior_variance: Optional[np.ndarray] = None

    def setup_model_space(self) -> None:
        """Set up the multi-component model space."""
        start = time.time()

        # Create interval
        self.interval = Interval(0.0, 1.0)

        # Create function spaces for vp, vs, rho
        self.M_vp = LebesgueSpace(
            self.interval,
            N=self.config.N,
            integration_N=self.config.integration_N,
            parallel=self.config.parallel,
            n_jobs=self.config.n_jobs
        )

        self.M_vs = LebesgueSpace(
            self.interval,
            N=self.config.N,
            integration_N=self.config.integration_N,
            parallel=self.config.parallel,
            n_jobs=self.config.n_jobs
        )

        self.M_rho = LebesgueSpace(
            self.interval,
            N=self.config.N,
            integration_N=self.config.integration_N,
            parallel=self.config.parallel,
            n_jobs=self.config.n_jobs
        )

        # Create Euclidean spaces for sigma_0, sigma_1
        self.M_sigma_0 = EuclideanSpace(1)
        self.M_sigma_1 = EuclideanSpace(1)

        # Create direct sum for model space
        self.M_model = HilbertSpaceDirectSum([
            [self.M_vp, self.M_vs, self.M_rho],
            [self.M_sigma_0, self.M_sigma_1]
        ])

        # Create data space
        self.M_data = EuclideanSpace(self.config.N_d)

        self.timings['setup_model_space'] = time.time() - start

    def create_operators(self) -> None:
        """Create forward operator G and target operator T."""
        start = time.time()

        # Create x_p points for evaluation
        x_p = np.linspace(0.0, 1.0, self.config.N_p)

        # Forward operator G: evaluates functions at x_p
        G_vp = self.M_vp.point_evaluation_operator(x_p)
        G_vs = self.M_vs.point_evaluation_operator(x_p)
        G_rho = self.M_rho.point_evaluation_operator(x_p)

        # Zero operators for sigmas
        G_sigma_0 = LinearOperator(
            domain=self.M_sigma_0,
            codomain=EuclideanSpace(self.config.N_p),
            matrix_operator=lambda x: np.zeros(self.config.N_p)
        )
        G_sigma_1 = LinearOperator(
            domain=self.M_sigma_1,
            codomain=EuclideanSpace(self.config.N_p),
            matrix_operator=lambda x: np.zeros(self.config.N_p)
        )

        self.G = RowLinearOperator([
            [G_vp, G_vs, G_rho],
            [G_sigma_0, G_sigma_1]
        ])

        # Target operator T: similar structure
        T_vp = self.M_vp.point_evaluation_operator(x_p)
        T_vs = self.M_vs.point_evaluation_operator(x_p)
        T_rho = self.M_rho.point_evaluation_operator(x_p)

        T_sigma_0 = LinearOperator(
            domain=self.M_sigma_0,
            codomain=EuclideanSpace(self.config.N_p),
            matrix_operator=lambda x: np.zeros(self.config.N_p)
        )
        T_sigma_1 = LinearOperator(
            domain=self.M_sigma_1,
            codomain=EuclideanSpace(self.config.N_p),
            matrix_operator=lambda x: np.zeros(self.config.N_p)
        )

        self.T = RowLinearOperator([
            [T_vp, T_vs, T_rho],
            [T_sigma_0, T_sigma_1]
        ])

        self.timings['create_operators'] = time.time() - start

    def generate_data(self) -> None:
        """Generate synthetic data from true parameters."""
        start = time.time()

        # Create true parameters (simple functions for each component)
        vp_true = self.M_vp.from_function(lambda x: np.sin(2 * np.pi * x))
        vs_true = self.M_vs.from_function(lambda x: np.cos(2 * np.pi * x))
        rho_true = self.M_rho.from_function(lambda x: np.exp(-((x - 0.5) ** 2) / 0.1))
        sigma_0_true = np.array([0.5])
        sigma_1_true = np.array([0.3])

        # Combine into model parameter
        self.u_true = self.M_model.concatenate([
            [vp_true, vs_true, rho_true],
            [sigma_0_true, sigma_1_true]
        ])

        # Generate clean data
        self.d_true = self.G @ self.u_true

        # Add noise
        noise = np.random.normal(0, self.config.noise_level, self.config.N_d)
        self.d_obs = self.d_true + noise

        self.timings['generate_data'] = time.time() - start

    def setup_prior(self) -> None:
        """Set up the prior distribution."""
        start = time.time()

        # Create Bessel-Sobolev priors for function components
        prior_vp = BesselSobolevPrior(
            self.M_vp,
            s=self.config.s_vp,
            k=self.config.k_vp,
            alpha=self.config.alpha_vp,
            kl_truncation=self.config.kl_truncation_vp,
            parallel=self.config.parallel,
            n_jobs=self.config.n_jobs
        )

        prior_vs = BesselSobolevPrior(
            self.M_vs,
            s=self.config.s_vs,
            k=self.config.k_vs,
            alpha=self.config.alpha_vs,
            kl_truncation=self.config.kl_truncation_vs,
            parallel=self.config.parallel,
            n_jobs=self.config.n_jobs
        )

        prior_rho = BesselSobolevPrior(
            self.M_rho,
            s=self.config.s_rho,
            k=self.config.k_rho,
            alpha=self.config.alpha_rho,
            kl_truncation=self.config.kl_truncation_rho,
            parallel=self.config.parallel,
            n_jobs=self.config.n_jobs
        )

        # Create diagonal Gaussian priors for sigma components
        prior_sigma_0 = GaussianMeasure.from_covariance_matrix(
            self.M_sigma_0,
            np.array([[self.config.sigma_0_variance]]),
            can_sample=False
        )

        prior_sigma_1 = GaussianMeasure.from_covariance_matrix(
            self.M_sigma_1,
            np.array([[self.config.sigma_1_variance]]),
            can_sample=False
        )

        # Combine into direct sum prior
        self.prior = GaussianMeasure.direct_sum([
            [prior_vp, prior_vs, prior_rho],
            [prior_sigma_0, prior_sigma_1]
        ])

        self.timings['setup_prior'] = time.time() - start

    def run_inference(self) -> None:
        """Run Bayesian inference."""
        start = time.time()

        # Create noise covariance
        Gamma = self.config.noise_level ** 2 * np.eye(self.config.N_d)

        # Set up inference problem
        inference = LinearBayesian(
            G=self.G,
            prior=self.prior,
            Gamma=Gamma,
            parallel=self.config.parallel,
            n_jobs=self.config.n_jobs
        )

        # Compute posterior
        if self.config.compute_model_posterior:
            # Full posterior with sampling capability
            self.posterior = inference.posterior(self.d_obs)
            self.posterior_mean = self.posterior.mean
            # Compute variance from posterior samples or covariance
            if hasattr(self.posterior, 'covariance_matrix'):
                cov_matrix = self.posterior.covariance_matrix()
                self.posterior_variance = np.diag(cov_matrix)
            else:
                # Estimate from samples
                samples = np.array([self.posterior.sample() for _ in range(100)])
                self.posterior_variance = np.var(samples, axis=0)
        else:
            # Fast computation: mean and variance only
            self.posterior_mean = inference.posterior_mean(self.d_obs)
            self.posterior_variance = inference.posterior_variance()

        self.timings['run_inference'] = time.time() - start

    def compute_metrics(self) -> None:
        """Compute various error metrics and statistics."""
        start = time.time()

        # Unpack true and posterior parameters
        u_true_parts = self.M_model.unpack(self.u_true)
        u_post_parts = self.M_model.unpack(self.posterior_mean)

        vp_true, vs_true, rho_true = u_true_parts[0]
        sigma_0_true, sigma_1_true = u_true_parts[1]

        vp_post, vs_post, rho_post = u_post_parts[0]
        sigma_0_post, sigma_1_post = u_post_parts[1]

        # Compute L2 errors for function components
        self.metrics['vp_l2_error'] = self.M_vp.norm(vp_true - vp_post)
        self.metrics['vs_l2_error'] = self.M_vs.norm(vs_true - vs_post)
        self.metrics['rho_l2_error'] = self.M_rho.norm(rho_true - rho_post)

        # Compute relative L2 errors
        self.metrics['vp_rel_l2_error'] = self.metrics['vp_l2_error'] / self.M_vp.norm(vp_true)
        self.metrics['vs_rel_l2_error'] = self.metrics['vs_l2_error'] / self.M_vs.norm(vs_true)
        self.metrics['rho_rel_l2_error'] = self.metrics['rho_l2_error'] / self.M_rho.norm(rho_true)

        # Compute pointwise max errors
        x_eval = np.linspace(0, 1, 1000)
        vp_true_eval = self.M_vp.evaluate(vp_true, x_eval)
        vp_post_eval = self.M_vp.evaluate(vp_post, x_eval)
        self.metrics['vp_max_error'] = np.max(np.abs(vp_true_eval - vp_post_eval))

        vs_true_eval = self.M_vs.evaluate(vs_true, x_eval)
        vs_post_eval = self.M_vs.evaluate(vs_post, x_eval)
        self.metrics['vs_max_error'] = np.max(np.abs(vs_true_eval - vs_post_eval))

        rho_true_eval = self.M_rho.evaluate(rho_true, x_eval)
        rho_post_eval = self.M_rho.evaluate(rho_post, x_eval)
        self.metrics['rho_max_error'] = np.max(np.abs(rho_true_eval - rho_post_eval))

        # Compute errors for sigma components
        self.metrics['sigma_0_error'] = np.abs(sigma_0_true[0] - sigma_0_post[0])
        self.metrics['sigma_1_error'] = np.abs(sigma_1_true[0] - sigma_1_post[0])

        # Compute relative errors for sigmas
        self.metrics['sigma_0_rel_error'] = self.metrics['sigma_0_error'] / np.abs(sigma_0_true[0])
        self.metrics['sigma_1_rel_error'] = self.metrics['sigma_1_error'] / np.abs(sigma_1_true[0])

        # Data fit metrics
        d_post = self.G @ self.posterior_mean
        self.metrics['data_l2_error'] = np.linalg.norm(self.d_obs - d_post)
        self.metrics['data_rel_l2_error'] = self.metrics['data_l2_error'] / np.linalg.norm(self.d_obs)
        self.metrics['data_max_error'] = np.max(np.abs(self.d_obs - d_post))

        # Uncertainty metrics (average posterior standard deviation)
        var_parts = self.M_model.unpack(self.posterior_variance)
        vp_var, vs_var, rho_var = var_parts[0]
        sigma_0_var, sigma_1_var = var_parts[1]

        self.metrics['vp_avg_std'] = np.sqrt(np.mean(vp_var))
        self.metrics['vs_avg_std'] = np.sqrt(np.mean(vs_var))
        self.metrics['rho_avg_std'] = np.sqrt(np.mean(rho_var))
        self.metrics['sigma_0_std'] = np.sqrt(sigma_0_var[0])
        self.metrics['sigma_1_std'] = np.sqrt(sigma_1_var[0])

        # Prior comparison (distance from prior mean)
        u_prior_mean = self.prior.mean
        self.metrics['prior_posterior_distance'] = self.M_model.norm(self.posterior_mean - u_prior_mean)

        self.timings['compute_metrics'] = time.time() - start

    def plot_results(self, save: bool = True) -> None:
        """Generate all visualization figures."""
        start = time.time()

        # Create figures directory
        fig_dir = self.output_dir / 'figures'
        fig_dir.mkdir(exist_ok=True)

        # Unpack parameters
        u_true_parts = self.M_model.unpack(self.u_true)
        u_post_parts = self.M_model.unpack(self.posterior_mean)
        var_parts = self.M_model.unpack(self.posterior_variance)

        vp_true, vs_true, rho_true = u_true_parts[0]
        sigma_0_true, sigma_1_true = u_true_parts[1]

        vp_post, vs_post, rho_post = u_post_parts[0]
        sigma_0_post, sigma_1_post = u_post_parts[1]

        vp_var, vs_var, rho_var = var_parts[0]
        sigma_0_var, sigma_1_var = var_parts[1]

        # Evaluation points
        x_eval = np.linspace(0, 1, 1000)

        # Figure 1: All function components together
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        # vp
        vp_true_eval = self.M_vp.evaluate(vp_true, x_eval)
        vp_post_eval = self.M_vp.evaluate(vp_post, x_eval)
        vp_std = np.sqrt(vp_var)
        vp_post_std_eval = self.M_vp.evaluate(vp_std, x_eval)

        axes[0].plot(x_eval, vp_true_eval, 'k-', label='True', linewidth=2)
        axes[0].plot(x_eval, vp_post_eval, 'b-', label='Posterior mean', linewidth=2)
        axes[0].fill_between(x_eval,
                             vp_post_eval - 2*vp_post_std_eval,
                             vp_post_eval + 2*vp_post_std_eval,
                             alpha=0.3, label='95% CI')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('vp')
        axes[0].set_title(f'vp (L2 error: {self.metrics["vp_l2_error"]:.4f})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # vs
        vs_true_eval = self.M_vs.evaluate(vs_true, x_eval)
        vs_post_eval = self.M_vs.evaluate(vs_post, x_eval)
        vs_std = np.sqrt(vs_var)
        vs_post_std_eval = self.M_vs.evaluate(vs_std, x_eval)

        axes[1].plot(x_eval, vs_true_eval, 'k-', label='True', linewidth=2)
        axes[1].plot(x_eval, vs_post_eval, 'b-', label='Posterior mean', linewidth=2)
        axes[1].fill_between(x_eval,
                             vs_post_eval - 2*vs_post_std_eval,
                             vs_post_eval + 2*vs_post_std_eval,
                             alpha=0.3, label='95% CI')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('vs')
        axes[1].set_title(f'vs (L2 error: {self.metrics["vs_l2_error"]:.4f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # rho
        rho_true_eval = self.M_rho.evaluate(rho_true, x_eval)
        rho_post_eval = self.M_rho.evaluate(rho_post, x_eval)
        rho_std = np.sqrt(rho_var)
        rho_post_std_eval = self.M_rho.evaluate(rho_std, x_eval)

        axes[2].plot(x_eval, rho_true_eval, 'k-', label='True', linewidth=2)
        axes[2].plot(x_eval, rho_post_eval, 'b-', label='Posterior mean', linewidth=2)
        axes[2].fill_between(x_eval,
                             rho_post_eval - 2*rho_post_std_eval,
                             rho_post_eval + 2*rho_post_std_eval,
                             alpha=0.3, label='95% CI')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('rho')
        axes[2].set_title(f'rho (L2 error: {self.metrics["rho_l2_error"]:.4f})')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            plt.savefig(fig_dir / 'all_components.png', dpi=150, bbox_inches='tight')
            plt.savefig(fig_dir / 'all_components.pdf', bbox_inches='tight')
        plt.close()

        # Figure 2: Sigma components
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # sigma_0
        axes[0].bar(['True', 'Posterior'],
                   [sigma_0_true[0], sigma_0_post[0]],
                   color=['black', 'blue'],
                   alpha=0.7)
        axes[0].errorbar(['Posterior'], [sigma_0_post[0]],
                        yerr=[2*np.sqrt(sigma_0_var[0])],
                        fmt='none', color='red', capsize=10, linewidth=2,
                        label='95% CI')
        axes[0].set_ylabel('σ₀')
        axes[0].set_title(f'σ₀ (error: {self.metrics["sigma_0_error"]:.4f})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # sigma_1
        axes[1].bar(['True', 'Posterior'],
                   [sigma_1_true[0], sigma_1_post[0]],
                   color=['black', 'blue'],
                   alpha=0.7)
        axes[1].errorbar(['Posterior'], [sigma_1_post[0]],
                        yerr=[2*np.sqrt(sigma_1_var[0])],
                        fmt='none', color='red', capsize=10, linewidth=2,
                        label='95% CI')
        axes[1].set_ylabel('σ₁')
        axes[1].set_title(f'σ₁ (error: {self.metrics["sigma_1_error"]:.4f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        if save:
            plt.savefig(fig_dir / 'sigma_components.png', dpi=150, bbox_inches='tight')
            plt.savefig(fig_dir / 'sigma_components.pdf', bbox_inches='tight')
        plt.close()

        # Figure 3: Data fit
        fig, ax = plt.subplots(figsize=(10, 6))

        x_d = np.linspace(0, 1, self.config.N_d)
        d_post = self.G @ self.posterior_mean

        ax.plot(x_d, self.d_true, 'k-', label='True data', linewidth=2)
        ax.plot(x_d, self.d_obs, 'ro', label='Observed data', markersize=4, alpha=0.6)
        ax.plot(x_d, d_post, 'b-', label='Posterior prediction', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('d')
        ax.set_title(f'Data fit (L2 error: {self.metrics["data_l2_error"]:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            plt.savefig(fig_dir / 'data_fit.png', dpi=150, bbox_inches='tight')
            plt.savefig(fig_dir / 'data_fit.pdf', bbox_inches='tight')
        plt.close()

        # Figure 4: Error distributions
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        # Pointwise errors for each component
        vp_error = np.abs(vp_true_eval - vp_post_eval)
        axes[0].plot(x_eval, vp_error, 'r-', linewidth=2)
        axes[0].axhline(self.metrics['vp_max_error'], color='k', linestyle='--',
                       label=f'Max: {self.metrics["vp_max_error"]:.4f}')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('|error|')
        axes[0].set_title('vp pointwise error')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        vs_error = np.abs(vs_true_eval - vs_post_eval)
        axes[1].plot(x_eval, vs_error, 'r-', linewidth=2)
        axes[1].axhline(self.metrics['vs_max_error'], color='k', linestyle='--',
                       label=f'Max: {self.metrics["vs_max_error"]:.4f}')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('|error|')
        axes[1].set_title('vs pointwise error')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        rho_error = np.abs(rho_true_eval - rho_post_eval)
        axes[2].plot(x_eval, rho_error, 'r-', linewidth=2)
        axes[2].axhline(self.metrics['rho_max_error'], color='k', linestyle='--',
                       label=f'Max: {self.metrics["rho_max_error"]:.4f}')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('|error|')
        axes[2].set_title('rho pointwise error')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            plt.savefig(fig_dir / 'pointwise_errors.png', dpi=150, bbox_inches='tight')
            plt.savefig(fig_dir / 'pointwise_errors.pdf', bbox_inches='tight')
        plt.close()

        # Figure 5: Uncertainty quantification
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        axes[0].plot(x_eval, vp_post_std_eval, 'b-', linewidth=2)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('std')
        axes[0].set_title(f'vp posterior std (avg: {self.metrics["vp_avg_std"]:.4f})')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x_eval, vs_post_std_eval, 'b-', linewidth=2)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('std')
        axes[1].set_title(f'vs posterior std (avg: {self.metrics["vs_avg_std"]:.4f})')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(x_eval, rho_post_std_eval, 'b-', linewidth=2)
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('std')
        axes[2].set_title(f'rho posterior std (avg: {self.metrics["rho_avg_std"]:.4f})')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            plt.savefig(fig_dir / 'posterior_uncertainty.png', dpi=150, bbox_inches='tight')
            plt.savefig(fig_dir / 'posterior_uncertainty.pdf', bbox_inches='tight')
        plt.close()

        self.timings['plot_results'] = time.time() - start

    def save_results(self) -> None:
        """Save configuration, metrics, and timings to JSON files."""
        # Save configuration
        self.config.save(self.output_dir / 'config.json')

        # Save metrics
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # Save timings
        total_time = sum(self.timings.values())
        self.timings['total'] = total_time
        with open(self.output_dir / 'timings.json', 'w') as f:
            json.dump(self.timings, f, indent=2)

    def run(self) -> Dict[str, Any]:
        """Run the complete experiment pipeline.

        Returns:
            Dictionary containing metrics and timings
        """
        print(f"Running experiment: {self.config.name}")
        print(f"Output directory: {self.output_dir}")

        # Run all steps
        print("  1. Setting up model space...")
        self.setup_model_space()

        print("  2. Creating operators...")
        self.create_operators()

        print("  3. Generating data...")
        self.generate_data()

        print("  4. Setting up prior...")
        self.setup_prior()

        print("  5. Running inference...")
        self.run_inference()

        print("  6. Computing metrics...")
        self.compute_metrics()

        print("  7. Generating plots...")
        self.plot_results()

        print("  8. Saving results...")
        self.save_results()

        print(f"Experiment complete! Total time: {sum(self.timings.values()):.2f}s")
        print(f"Key metrics:")
        print(f"  vp L2 error: {self.metrics['vp_rel_l2_error']:.4f}")
        print(f"  vs L2 error: {self.metrics['vs_rel_l2_error']:.4f}")
        print(f"  rho L2 error: {self.metrics['rho_rel_l2_error']:.4f}")
        print(f"  σ₀ error: {self.metrics['sigma_0_error']:.4f}")
        print(f"  σ₁ error: {self.metrics['sigma_1_error']:.4f}")

        return {
            'metrics': self.metrics,
            'timings': self.timings,
            'config': self.config.to_dict()
        }


def run_single_experiment(config: Demo1Config, output_dir: Path) -> Dict[str, Any]:
    """Convenience function to run a single experiment.

    Args:
        config: Demo1Config object
        output_dir: Directory to save results

    Returns:
        Dictionary containing results
    """
    experiment = Demo1Experiment(config, output_dir)
    return experiment.run()
