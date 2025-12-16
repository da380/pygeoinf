"""
Experiment runner for PLI (Probabilistic Linear Inference).

This module implements the PLIExperiment class that automates running
the complete PLI inference pipeline from pli.ipynb, including model setup,
data generation, prior construction, inference, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
from datetime import datetime

from pygeoinf.interval.function_providers import NormalModesProvider, BumpFunctionProvider
from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval import Lebesgue
from pygeoinf.interval.configs import (
    IntegrationConfig, ParallelConfig,
    LebesgueIntegrationConfig, LebesgueParallelConfig
)
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.interval.functions import Function
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.interval.boundary_conditions import BoundaryConditions
from pygeoinf.interval.operators import (
    InverseLaplacian, Laplacian, BesselSobolevInverse, SOLAOperator
)
from pygeoinf.forward_problem import LinearForwardProblem
from pygeoinf.linear_bayesian import LinearBayesianInference
from pygeoinf.linear_solvers import CholeskySolver
from pygeoinf.interval.KL_sampler import KLSampler

from .pli_config import PLIConfig


class PLIExperiment:
    """Run a single instance of the PLI inference experiment.

    This class encapsulates all steps from pli.ipynb:
    1. Set up model, data, and property spaces
    2. Create forward operator G (sensitivity kernels) and target operator T
    3. Generate synthetic data from true model
    4. Set up Gaussian prior with Bessel-Sobolev covariance
    5. Run Bayesian inference
    6. Compute metrics and generate visualizations
    7. Save all results to disk

    The experiment can be configured via a PLIConfig object, allowing
    systematic parameter sweeps and reproducibility.
    """

    def __init__(self, config: PLIConfig, output_dir: Path):
        """Initialize experiment with configuration.

        Args:
            config: PLIConfig object specifying all parameters
            output_dir: Directory to save results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        # Set random seed if specified
        np.random.seed(config.random_seed)

        # Storage for results
        self.timings: Dict[str, float] = {}
        self.metrics: Dict[str, Any] = {}

        # Model components (set during setup)
        self.function_domain: Optional[IntervalDomain] = None
        self.M: Optional[Lebesgue] = None  # Model space
        self.D: Optional[EuclideanSpace] = None  # Data space
        self.P: Optional[EuclideanSpace] = None  # Property space

        # Operators
        self.G = None  # Forward operator (sensitivity kernels)
        self.T = None  # Target operator (property kernels)

        # Data
        self.m_bar: Optional[Function] = None  # True model
        self.d_bar = None  # Clean data
        self.d_tilde = None  # Noisy observations

        # Prior
        self.M_prior: Optional[GaussianMeasure] = None
        self.sampler: Optional[KLSampler] = None

        # Posterior
        self.m_tilde: Optional[Function] = None  # Posterior mean model
        self.p_tilde = None  # Posterior property mean
        self.cov_P_matrix = None  # Property posterior covariance

        # Integration and parallel configs
        self._setup_configs()

        # Set plotting style
        sns.set_theme(style="whitegrid", palette="muted", color_codes=True)

    def _setup_configs(self):
        """Setup integration and parallel configurations."""
        self.integration_cfg = IntegrationConfig(
            method=self.config.integration_method,
            n_points=self.config.integration_n_points
        )
        self.parallel_cfg = ParallelConfig(
            enabled=self.config.parallel_enabled,
            n_jobs=self.config.parallel_n_jobs
        )
        self.lebesgue_integration_cfg = LebesgueIntegrationConfig(
            inner_product=self.integration_cfg,
            dual=self.integration_cfg,
            general=self.integration_cfg
        )

    def setup_spaces(self) -> None:
        """Set up the function domain and Hilbert spaces."""
        start = time.time()

        # Create function domain
        self.function_domain = IntervalDomain(
            self.config.domain_a,
            self.config.domain_b
        )

        # Create model space (Lebesgue L²)
        self.M = Lebesgue(
            self.config.N,
            self.function_domain,
            basis=self.config.basis,
            integration_config=self.lebesgue_integration_cfg,
            parallel_config=self.parallel_cfg
        )

        # Create data space
        self.D = EuclideanSpace(self.config.N_d)

        # Create property space
        self.P = EuclideanSpace(self.config.N_p)

        self.timings['setup_spaces'] = time.time() - start

    def create_operators(self) -> None:
        """Create forward operator G and target operator T."""
        start = time.time()

        # Forward operator (sensitivity kernels via normal modes)
        normal_modes_provider = NormalModesProvider(
            self.M,
            n_modes_range=self.config.n_modes_range,
            coeff_range=self.config.coeff_range,
            gaussian_width_percent_range=self.config.gaussian_width_percent_range,
            freq_range=self.config.freq_range,
            random_state=self.config.random_seed,
        )
        self.G = SOLAOperator(
            self.M, self.D, normal_modes_provider,
            integration_config=self.integration_cfg
        )

        # Target operator (bump functions for property extraction)
        target_provider = BumpFunctionProvider(
            self.M,
            centers=self.config.target_centers,
            default_width=self.config.target_width
        )
        self.T = SOLAOperator(
            self.M, self.P, target_provider,
            integration_config=self.integration_cfg
        )

        self.timings['create_operators'] = time.time() - start

    def generate_data(self) -> None:
        """Generate synthetic data from true model."""
        start = time.time()

        # Create true model (oscillating Gaussian envelope + linear trend)
        center = self.config.domain_center
        self.m_bar = Function(
            self.M,
            evaluate_callable=lambda x: (
                np.exp(-((x - center) / 0.5) ** 2) * np.sin(5 * np.pi * x) + x
            )
        )

        # Generate clean observations
        self.d_bar = self.G(self.m_bar)

        # Add noise
        noise_std = self.config.noise_level * np.max(np.abs(self.d_bar))
        self.d_tilde = self.d_bar + np.random.normal(0, noise_std, self.d_bar.shape)

        # Create data noise measure
        noise_variance = noise_std ** 2
        C_D_matrix = noise_variance * np.eye(self.config.N_d)
        self.gaussian_D_noise = GaussianMeasure.from_covariance_matrix(
            self.D, C_D_matrix, expectation=np.zeros(self.config.N_d)
        )

        # Store metrics
        self.metrics['noise_std'] = noise_std
        self.metrics['snr'] = np.max(np.abs(self.d_bar)) / noise_std

        self.timings['generate_data'] = time.time() - start

    def setup_prior(self) -> None:
        """Set up the Gaussian prior measure."""
        start = time.time()

        # Setup boundary conditions
        bc = BoundaryConditions(
            bc_type=self.config.bc_type,
            left=self.config.bc_left,
            right=self.config.bc_right
        )

        # Create prior covariance operator
        if self.config.prior_type == 'inverse_laplacian':
            # C_0 = (αL)^(-1)
            C_0 = InverseLaplacian(
                self.M, bc, self.config.alpha_computed,
                method=self.config.method, dofs=self.config.dofs
            )
        elif self.config.prior_type == 'bessel_sobolev':
            # C_0 = (k²I + αL)^(-s)
            L = Laplacian(
                self.M, bc, self.config.alpha_computed,
                method=self.config.method, dofs=self.config.dofs,
                integration_config=self.integration_cfg
            )
            C_0 = BesselSobolevInverse(
                self.M, self.M, self.config.k, self.config.s, L,
                dofs=self.config.dofs, n_samples=self.config.n_samples,
                use_fast_transforms=self.config.use_fast_transforms
            )
        else:
            raise ValueError(f"Unknown prior_type: {self.config.prior_type}")

        # Prior mean (linear trend)
        m_0 = Function(self.M, evaluate_callable=lambda x: x)

        # Create KL sampler and Gaussian measure
        self.sampler = KLSampler(C_0, mean=m_0, n_modes=self.config.K)
        self.M_prior = GaussianMeasure(
            covariance=C_0,
            expectation=m_0,
            sample=self.sampler.sample
        )

        self.timings['setup_prior'] = time.time() - start

    def run_inference(self) -> None:
        """Run Bayesian inference."""
        start = time.time()

        # Setup inference
        forward_problem = LinearForwardProblem(
            self.G, data_error_measure=self.gaussian_D_noise
        )
        bayesian_inference = LinearBayesianInference(
            forward_problem, self.M_prior, self.T
        )
        solver = CholeskySolver(
            parallel=self.config.parallel_enabled,
            n_jobs=self.config.parallel_n_jobs
        )

        if self.config.compute_model_posterior:
            # Workflow 1: Full model posterior
            t1 = time.time()
            posterior_model = bayesian_inference.model_posterior_measure(
                self.d_tilde, solver
            )
            t2 = time.time()
            self.timings['model_posterior_compute'] = t2 - t1

            # Extract dense covariance matrix
            t3 = time.time()
            C_M_matrix = posterior_model.covariance.matrix(
                dense=True, galerkin=True,
                parallel=self.config.parallel_enabled,
                n_jobs=self.config.parallel_n_jobs
            )
            t4 = time.time()
            self.timings['model_covariance_extract'] = t4 - t3

            # Create sampling-capable measure
            self.mu_M = GaussianMeasure.from_covariance_matrix(
                self.M, C_M_matrix, expectation=posterior_model.expectation
            )
            self.m_tilde = self.mu_M.expectation

            # Push to property space
            t5 = time.time()
            property_posterior = self.mu_M.affine_mapping(operator=self.T)
            self.p_tilde = property_posterior.expectation
            self.cov_P_matrix = property_posterior.covariance.matrix(
                dense=True, galerkin=True,
                parallel=self.config.parallel_enabled,
                n_jobs=self.config.parallel_n_jobs
            )
            t6 = time.time()
            self.timings['property_posterior_compute'] = t6 - t5

        else:
            # Workflow 2: Direct property posterior (faster)
            t1 = time.time()
            posterior_model = bayesian_inference.model_posterior_measure(
                self.d_tilde, solver
            )
            self.m_tilde = posterior_model.expectation
            t2 = time.time()
            self.timings['model_posterior_compute'] = t2 - t1

            # Push to property space
            t3 = time.time()
            property_posterior = posterior_model.affine_mapping(operator=self.T)
            self.p_tilde = property_posterior.expectation
            t4 = time.time()
            self.timings['property_mapping'] = t4 - t3

            # Extract property covariance (small matrix)
            t5 = time.time()
            self.cov_P_matrix = property_posterior.covariance.matrix(
                dense=True, galerkin=True,
                parallel=self.config.parallel_enabled,
                n_jobs=self.config.parallel_n_jobs
            )
            t6 = time.time()
            self.timings['property_covariance_extract'] = t6 - t5

        self.timings['inference_total'] = time.time() - start

    def compute_metrics(self) -> None:
        """Compute error metrics and statistics."""
        start = time.time()

        x = np.linspace(self.config.domain_a, self.config.domain_b, 1000)

        # Model reconstruction error
        m_true_eval = self.m_bar.evaluate(x)
        m_post_eval = self.m_tilde.evaluate(x)
        model_error = np.sqrt(np.mean((m_post_eval - m_true_eval) ** 2))

        self.metrics['model_rms_error'] = model_error
        self.metrics['model_relative_error'] = model_error / np.std(m_true_eval)
        self.metrics['model_max_error'] = np.max(np.abs(m_post_eval - m_true_eval))

        # Property errors
        true_props = self.T(self.m_bar)
        property_errors = np.abs(self.p_tilde - true_props)
        std_P_post = np.sqrt(np.diag(self.cov_P_matrix))

        self.metrics['property_mean_abs_error'] = np.mean(property_errors)
        self.metrics['property_rms_error'] = np.sqrt(np.mean(property_errors ** 2))
        self.metrics['property_max_error'] = np.max(property_errors)
        self.metrics['properties_within_2sigma'] = int(np.sum(property_errors <= 2 * std_P_post))
        self.metrics['properties_within_2sigma_pct'] = (
            100 * np.sum(property_errors <= 2 * std_P_post) / len(true_props)
        )
        self.metrics['avg_property_uncertainty'] = np.mean(std_P_post)

        # Data fit
        d_post = self.G(self.m_tilde)
        d_prior = self.G(self.M_prior.expectation)

        self.metrics['data_misfit_posterior'] = np.linalg.norm(d_post - self.d_tilde)
        self.metrics['data_misfit_prior'] = np.linalg.norm(d_prior - self.d_tilde)
        self.metrics['data_fit_improvement_pct'] = 100 * (
            1 - self.metrics['data_misfit_posterior'] / self.metrics['data_misfit_prior']
        )

        # Prior-to-posterior comparison
        prior_prop = self.T(self.M_prior.expectation)
        prior_P = self.M_prior.affine_mapping(operator=self.T)
        std_P_prior = np.sqrt(np.diag(prior_P.covariance.matrix(
            dense=True, galerkin=True,
            parallel=self.config.parallel_enabled,
            n_jobs=self.config.parallel_n_jobs
        )))
        self.metrics['uncertainty_reduction_pct'] = 100 * (
            1 - np.mean(std_P_post) / np.mean(std_P_prior)
        )

        self.timings['compute_metrics'] = time.time() - start

    def plot_results(self, save: bool = True) -> None:
        """Generate all visualization figures."""
        start = time.time()

        x = np.linspace(self.config.domain_a, self.config.domain_b, 1000)

        # 1. Sensitivity kernels
        self._plot_sensitivity_kernels(x, save)

        # 2. Target kernels
        self._plot_target_kernels(x, save)

        # 3. True model
        self._plot_true_model(x, save)

        # 4. Synthetic observations
        self._plot_observations(save)

        # 5. Prior measure
        self._plot_prior(x, save)

        # 6. Property prior
        self._plot_property_prior(save)

        # 7. Model posterior mean
        self._plot_model_posterior(x, save)

        # 8. Property inference results
        self._plot_property_posterior(save)

        self.timings['plot_results'] = time.time() - start

    def _plot_sensitivity_kernels(self, x, save):
        """Plot sensitivity kernels."""
        plt.figure(figsize=(12, 4), dpi=200)
        for i in range(self.config.N_d):
            plt.plot(x, self.G.get_kernel(i).evaluate(x),
                     color='tab:blue', alpha=0.35, linewidth=1.2)
        plt.title(r"Sensitivity Kernels $K_i(x)$", fontsize=18)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel("Kernel Value", fontsize=16)
        plt.grid(True, linestyle=':', alpha=0.4)
        sns.despine()
        plt.tight_layout()
        if save:
            plt.savefig(self.figures_dir / "sensitivity_kernels.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.figures_dir / "sensitivity_kernels.pdf", bbox_inches='tight')
        plt.close()

    def _plot_target_kernels(self, x, save):
        """Plot target kernels."""
        plt.figure(figsize=(12, 4), dpi=200)
        for i in range(self.config.N_p):
            plt.plot(x, self.T.get_kernel(i).evaluate(x),
                     color='tab:orange', alpha=0.6, linewidth=1.5)
        plt.title(r"Target Kernels $T_i(x)$", fontsize=18)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel("Kernel Value", fontsize=16)
        plt.grid(True, linestyle=':', alpha=0.4)
        sns.despine()
        plt.tight_layout()
        if save:
            plt.savefig(self.figures_dir / "target_kernels.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.figures_dir / "target_kernels.pdf", bbox_inches='tight')
        plt.close()

    def _plot_true_model(self, x, save):
        """Plot true model."""
        plt.figure(figsize=(12, 4), dpi=200)
        plt.plot(x, self.m_bar.evaluate(x), color='tab:red', linewidth=2.5, label=r'$\bar{m}(x)$')
        plt.title(r"True Model $\bar{m}(x)$", fontsize=18)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel("Model Value", fontsize=16)
        plt.legend(fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.4)
        sns.despine()
        plt.tight_layout()
        if save:
            plt.savefig(self.figures_dir / "true_model.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.figures_dir / "true_model.pdf", bbox_inches='tight')
        plt.close()

    def _plot_observations(self, save):
        """Plot synthetic observations."""
        plt.figure(figsize=(12, 4), dpi=200)
        data_indices = np.arange(len(self.d_bar))

        # Connection lines
        for i in range(len(self.d_bar)):
            plt.plot([i, i], [self.d_bar[i], self.d_tilde[i]],
                     color='gray', alpha=0.3, linewidth=0.8)

        plt.scatter(data_indices, self.d_tilde, label='Noisy Observations',
                    color='tab:blue', alpha=0.7, marker='o', s=25,
                    edgecolors='white', linewidths=0.5)
        plt.scatter(data_indices, self.d_bar, label='True Data',
                    color='tab:red', alpha=0.8, marker='x', s=30, linewidths=1.5)
        plt.xlabel('Observation Index', fontsize=16)
        plt.ylabel('Data Value', fontsize=16)
        plt.title('Synthetic Observations: Truth vs. Noisy Measurements', fontsize=18)
        plt.legend(fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.4)
        sns.despine()
        plt.tight_layout()
        if save:
            plt.savefig(self.figures_dir / "synthetic_observations.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.figures_dir / "synthetic_observations.pdf", bbox_inches='tight')
        plt.close()

    def _plot_prior(self, x, save):
        """Plot prior measure with samples."""
        plt.figure(figsize=(12, 5), dpi=200)

        # Prior samples
        num_samples = 15
        for i in range(num_samples):
            sample = self.M_prior.sample()
            plt.plot(x, sample.evaluate(x), color='tab:blue', alpha=0.25, linewidth=1,
                     label='Prior Samples' if i == 0 else "")

        # Prior mean
        mean_values = self.M_prior.expectation.evaluate(x)
        plt.plot(x, mean_values, color='tab:green', linewidth=3,
                 label='Prior Mean', zorder=10)

        # Uncertainty band (from KL sampler)
        variance_func = self.sampler.variance_function()
        std_values = np.sqrt(variance_func.evaluate(x))
        plt.fill_between(x, mean_values - 2*std_values, mean_values + 2*std_values,
                         color='tab:blue', alpha=0.15, label='±2σ Band')
        plt.plot(x, mean_values + 2*std_values, color='tab:blue', linestyle='--',
                 alpha=0.7, linewidth=1.5)
        plt.plot(x, mean_values - 2*std_values, color='tab:blue', linestyle='--',
                 alpha=0.7, linewidth=1.5)

        # True model
        plt.plot(x, self.m_bar.evaluate(x), color='tab:red', linewidth=2.5,
                 label='True Model', linestyle='--', zorder=9)

        plt.title("Prior Measure on Model Space", fontsize=18)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel("Model Value", fontsize=16)
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.4)
        sns.despine()
        plt.tight_layout()
        if save:
            plt.savefig(self.figures_dir / "prior_measure.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.figures_dir / "prior_measure.pdf", bbox_inches='tight')
        plt.close()

    def _plot_property_prior(self, save):
        """Plot property prior distribution."""
        prior_P = self.M_prior.affine_mapping(operator=self.T)
        mat = prior_P.covariance.matrix(
            dense=True, galerkin=True,
            parallel=self.config.parallel_enabled,
            n_jobs=self.config.parallel_n_jobs
        )
        if not isinstance(mat, np.ndarray):
            mat = np.asarray(mat)
        std_P = np.sqrt(np.diag(mat))

        centers = self.config.target_centers
        mean_prop = self.T(self.M_prior.expectation)
        true_props = self.T(self.m_bar)

        plt.figure(figsize=(12, 5), dpi=200)
        plt.errorbar(centers, mean_prop, yerr=2*std_P, fmt='o', color='tab:blue',
                     alpha=0.7, capsize=4, capthick=2, markersize=6,
                     label='Property Prior (mean ±2σ)')
        plt.fill_between(centers, mean_prop - 2*std_P, mean_prop + 2*std_P,
                         color='tab:blue', alpha=0.15)
        plt.scatter(centers, true_props, label='True Properties',
                    color='tab:red', marker='x', s=100, alpha=0.9, linewidths=3, zorder=10)
        plt.xlabel('Target Location', fontsize=16)
        plt.ylabel('Property Value', fontsize=16)
        plt.title('Property Prior: Beliefs Before Data', fontsize=18)
        plt.legend(fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.4)
        sns.despine()
        plt.tight_layout()
        if save:
            plt.savefig(self.figures_dir / "property_prior.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.figures_dir / "property_prior.pdf", bbox_inches='tight')
        plt.close()

    def _plot_model_posterior(self, x, save):
        """Plot model posterior mean."""
        workflow_label = "Full Posterior" if self.config.compute_model_posterior else "Fast Workflow"

        plt.figure(figsize=(12, 6), dpi=200)

        # If full posterior, add samples
        if self.config.compute_model_posterior and hasattr(self, 'mu_M'):
            num_samples = 20
            for i in range(num_samples):
                sample = self.mu_M.sample()
                plt.plot(x, sample.evaluate(x), color='tab:blue', alpha=0.2, linewidth=1,
                         label='Posterior Samples' if i == 0 else "")

        plt.plot(x, self.m_tilde.evaluate(x), color='tab:blue', linewidth=3,
                 label='Posterior Mean', zorder=10)
        plt.plot(x, self.m_bar.evaluate(x), color='tab:red', linestyle='--',
                 linewidth=3, label='True Model', zorder=10)
        plt.plot(x, self.M_prior.expectation.evaluate(x), color='tab:green',
                 linestyle=':', linewidth=2, alpha=0.8, label='Prior Mean', zorder=5)

        plt.title(f"Model Posterior Mean ({workflow_label})", fontsize=18)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel("Model Value", fontsize=16)
        plt.legend(fontsize=14, loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.4)
        sns.despine()
        plt.tight_layout()
        if save:
            plt.savefig(self.figures_dir / "model_posterior.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.figures_dir / "model_posterior.pdf", bbox_inches='tight')
        plt.close()

    def _plot_property_posterior(self, save):
        """Plot property posterior results."""
        std_P_post = np.sqrt(np.diag(self.cov_P_matrix))
        true_props = self.T(self.m_bar)
        centers = self.config.target_centers

        workflow_label = "Full Posterior" if self.config.compute_model_posterior else "Fast Workflow"

        plt.figure(figsize=(12, 6), dpi=200)
        plt.errorbar(centers, self.p_tilde, yerr=2*std_P_post, fmt='o',
                     color='tab:blue', alpha=0.8, capsize=4, capthick=2,
                     markersize=8, linewidth=2, label='Posterior Properties (±2σ)')
        plt.fill_between(centers, self.p_tilde - 2*std_P_post,
                         self.p_tilde + 2*std_P_post, color='tab:blue', alpha=0.2)
        plt.scatter(centers, true_props, label='True Properties',
                    color='tab:red', marker='x', s=120, alpha=0.9, linewidths=4, zorder=10)

        # Prior for comparison
        mean_prop_prior = self.T(self.M_prior.expectation)
        plt.plot(centers, mean_prop_prior, 'o--', color='tab:green', alpha=0.6,
                 markersize=6, linewidth=2, label='Prior Properties')

        plt.xlabel('Target Location', fontsize=16)
        plt.ylabel('Property Value', fontsize=16)
        plt.title(f'Property Inference Results ({workflow_label})', fontsize=18)
        plt.legend(fontsize=14, loc='best')
        plt.grid(True, linestyle=':', alpha=0.4)
        sns.despine()
        plt.tight_layout()
        if save:
            plt.savefig(self.figures_dir / "property_posterior.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.figures_dir / "property_posterior.pdf", bbox_inches='tight')
        plt.close()

    def save_results(self) -> None:
        """Save configuration, metrics, and timings to JSON files."""
        # Save configuration
        self.config.save(self.output_dir / "config.json")

        # Save metrics
        with open(self.output_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # Save timings
        total_time = sum(self.timings.values())
        self.timings['total'] = total_time
        with open(self.output_dir / "timings.json", 'w') as f:
            json.dump(self.timings, f, indent=2)

    def run(self) -> Dict[str, Any]:
        """Run the complete experiment pipeline.

        Returns:
            Dictionary containing metrics and timings
        """
        print(f"\n{'='*80}")
        print(f"Running PLI experiment: {self.config.name}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}")

        # Run all steps
        print("\n  1. Setting up spaces...")
        self.setup_spaces()

        print("  2. Creating operators...")
        self.create_operators()

        print("  3. Generating synthetic data...")
        self.generate_data()

        print("  4. Setting up prior...")
        self.setup_prior()

        print("  5. Running Bayesian inference...")
        self.run_inference()

        print("  6. Computing metrics...")
        self.compute_metrics()

        print("  7. Generating plots...")
        self.plot_results()

        print("  8. Saving results...")
        self.save_results()

        print(f"\n{'='*80}")
        print(f"Experiment complete! Total time: {sum(self.timings.values()):.2f}s")
        print(f"\nKey metrics:")
        print(f"  Model RMS error: {self.metrics['model_rms_error']:.4f}")
        print(f"  Property RMS error: {self.metrics['property_rms_error']:.4f}")
        print(f"  Properties within ±2σ: {self.metrics['properties_within_2sigma']}/{self.config.N_p}")
        print(f"  Data fit improvement: {self.metrics['data_fit_improvement_pct']:.1f}%")
        print(f"  Uncertainty reduction: {self.metrics['uncertainty_reduction_pct']:.1f}%")
        print(f"{'='*80}\n")

        return {
            'metrics': self.metrics,
            'timings': self.timings,
            'config': self.config.to_dict()
        }


def run_single_experiment(config: PLIConfig, output_dir: Path) -> Dict[str, Any]:
    """Convenience function to run a single experiment.

    Args:
        config: PLIConfig object
        output_dir: Directory to save results

    Returns:
        Dictionary containing results
    """
    experiment = PLIExperiment(config, output_dir)
    return experiment.run()
