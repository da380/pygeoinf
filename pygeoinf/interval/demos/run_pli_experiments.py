"""
Automated Parameter Sweep for Probabilistic Linear Inference (PLI)

This script runs the PLI inference pipeline with various parameter
configurations and saves all results (images, metadata, timings) in an
organized directory structure.

Configuration Parameters:
- N: Model space dimension
- N_d: Number of data points
- N_p: Number of property points
- K: KL expansion modes for prior representation
- basis: Basis type ('sine', 'cosine', etc.)
- noise_level: Data noise level (relative to signal)
- compute_model_posterior: Whether to compute full model posterior (slower)
- random_seed: Random seed for reproducibility
- n_jobs: Number of parallel workers for computations (default: 8)
- bc_config: Boundary conditions for prior covariance (dict)

Prior Covariance Operator:
- prior_type: Type of prior covariance operator (default: 'bessel_sobolev')
  * 'inverse_laplacian': C_0 = (αL)^(-1)
  * 'bessel_sobolev': C_0 = (k²I + L)^(-s)

Bessel-Sobolev Parameters (used when prior_type='bessel_sobolev'):
- k: Bessel parameter k² (default: 1.0)
- s: Sobolev order s (default: 1.0)
- alpha: Laplacian scaling parameter (default: 0.1)
- method: Method for Laplacian ('spectral' or 'fd', default: 'spectral')
- dofs: Degrees of freedom (default: 100)
- n_samples: Number of samples for fast transforms (default: 1024)
- use_fast_transforms: Use fast spectral transforms (default: True)

Inverse Laplacian Parameters (used when prior_type='inverse_laplacian'):
- alpha: Prior regularization parameter (smoothness)
- method: Method for inverse Laplacian ('spectral' or 'fem')
- dofs: Degrees of freedom (default: 100)

Boundary Condition Formats:
1. Dirichlet (fixed value):
   {'bc_type': 'dirichlet', 'left': value, 'right': value}
   Example: {'bc_type': 'dirichlet', 'left': 0, 'right': 0}

2. Neumann (fixed derivative):
   {'bc_type': 'neumann', 'left': value, 'right': value}
   Example: {'bc_type': 'neumann', 'left': 0, 'right': 0}

3. Robin (linear combination):
   {'bc_type': 'robin', 'left': {'alpha': a, 'beta': b},
    'right': {'alpha': a, 'beta': b}}
   where a*u + b*u' = 0 at boundaries
   Example: {'bc_type': 'robin', 'left': {'alpha': 1, 'beta': 1},
             'right': {'alpha': 1, 'beta': 1}}

Directory structure:
experiments/
└── sweep_YYYYMMDD_HHMMSS/
    ├── sweep_config.json          # Overall sweep configuration
    ├── summary.csv                 # Summary of all runs
    └── run_001_K100_N100_Nd50/
        ├── config.json             # Run-specific configuration
        ├── timings.json            # Execution timings
        ├── metrics.json            # Error metrics, data fit, etc.
        └── figures/
            ├── sensitivity_kernels.png
            ├── target_kernels.png
            ├── true_model.png
            ├── synthetic_observations.png
            ├── data_likelihood_distribution.png
            ├── prior_measure_on_model_space.png
            ├── property_prior_distribution.png
            ├── model_posterior_mean.png
            └── property_inference_results.png
"""

import os
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import all necessary modules
from pygeoinf.interval.function_providers import NormalModesProvider, BumpFunctionProvider
from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval import Lebesgue
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.interval.operators import SOLAOperator
from pygeoinf.interval.functions import Function
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.interval.boundary_conditions import BoundaryConditions
from pygeoinf.interval.operators import (
    InverseLaplacian, Laplacian, BesselSobolevInverse
)
from pygeoinf.forward_problem import LinearForwardProblem
from pygeoinf.linear_bayesian import LinearBayesianInference
from pygeoinf.linear_solvers import CholeskySolver
from pygeoinf.interval.KL_sampler import KLSampler


class PLIExperiment:
    """Run a single PLI experiment with given parameters."""

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """
        Initialize experiment with configuration.

        Args:
            config: Dictionary with experiment parameters
            output_dir: Directory to save all outputs
        """
        self.config = config
        self.output_dir = output_dir
        self.figures_dir = output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Extract parameters
        self.N = config['N']  # Model space dimension
        self.N_d = config['N_d']  # Number of data points
        self.N_p = config['N_p']  # Number of property points
        self.K = config['K']  # KL expansion modes
        self.basis = config['basis']  # Basis type
        self.noise_level = config['noise_level']  # Data noise level
        self.compute_model_posterior = config.get(
            'compute_model_posterior', False)
        self.random_seed = config.get('random_seed', 42)

        # Parallelization parameter
        self.n_jobs = config.get('n_jobs', 8)  # Default to 8 workers

        # Boundary condition parameters (default to Dirichlet)
        # Format: {'bc_type': 'dirichlet', 'left': 0, 'right': 0}
        # or: {'bc_type': 'neumann', 'left': 0, 'right': 0}
        # or: {'bc_type': 'robin', 'left': {...}, 'right': {...}}
        default_bc = {'bc_type': 'dirichlet', 'left': 0, 'right': 0}
        self.bc_config = config.get('bc_config', default_bc)

        # Prior covariance operator type
        # 'inverse_laplacian': C_0 = (αL)^(-1)
        # 'bessel_sobolev': C_0 = (k²I + L)^(-s)
        self.prior_type = config.get('prior_type', 'bessel_sobolev')

        # Bessel-Sobolev parameters (only used if prior_type='bessel_sobolev')
        self.k = config.get('k', 1.0)  # Bessel parameter k²
        self.s = config.get('s', 1.0)  # Sobolev order s

        # Laplacian parameters
        self.alpha = config.get('alpha', 0.1)  # Laplacian scaling
        self.method = config.get('method', 'spectral')  # 'spectral' or 'fd'
        self.dofs = config.get('dofs', 100)  # Degrees of freedom
        self.n_samples = config.get('n_samples', 1024)  # For fast transforms
        self.use_fast_transforms = config.get('use_fast_transforms', True)

        # Storage for results
        self.timings = {}
        self.metrics = {}

        # Set plotting style
        sns.set_theme(style="whitegrid", palette="muted", color_codes=True)

    def run(self):
        """Execute the full PLI experiment pipeline."""
        print(f"\n{'='*80}")
        print(f"Running experiment: {self.output_dir.name}")
        print(f"  N={self.N}, N_d={self.N_d}, N_p={self.N_p}, K={self.K}")
        print(f"  Basis: {self.basis}, α={self.alpha}")
        print(f"{'='*80}\n")

        overall_start = time.time()

        # Run pipeline steps
        self._setup_spaces()
        self._create_operators()
        self._generate_synthetic_data()
        self._setup_prior()
        self._run_inference()
        self._compute_metrics()

        total_time = time.time() - overall_start
        self.timings['total'] = total_time

        # Save all results
        self._save_results()

        print(f"\n{'='*80}")
        print(f"Experiment complete! Total time: {total_time:.2f}s")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*80}\n")

        return self.metrics

    def _setup_spaces(self):
        """Create function domain and spaces."""
        t0 = time.time()

        self.function_domain = IntervalDomain(0, 1)
        self.M = Lebesgue(self.N, self.function_domain, basis=self.basis)
        self.D = EuclideanSpace(self.N_d)
        self.P = EuclideanSpace(self.N_p)

        self.timings['setup_spaces'] = time.time() - t0

    def _create_operators(self):
        """Create forward and property operators."""
        t0 = time.time()

        # Forward operator (sensitivity kernels)
        normal_modes_provider = NormalModesProvider(
            self.M,
            n_modes_range=(1, 50),
            coeff_range=(-5, 5),
            gaussian_width_percent_range=(1, 5),
            freq_range=(0.1, 20),
            random_state=self.random_seed,
        )
        self.G = SOLAOperator(self.M, self.D, normal_modes_provider)

        # Property operator (target kernels)
        width = 0.2
        centers = np.linspace(
            self.function_domain.a + width / 2,
            self.function_domain.b - width / 2,
            self.N_p
        )
        target_provider = BumpFunctionProvider(self.M, centers=centers, default_width=width)
        self.T = SOLAOperator(self.M, self.P, target_provider)

        self.timings['create_operators'] = time.time() - t0

        # Save kernel visualizations
        self._plot_kernels()

    def _plot_kernels(self):
        """Plot sensitivity and target kernels."""
        x = np.linspace(self.function_domain.a, self.function_domain.b, 1000)

        # Sensitivity kernels
        plt.figure(figsize=(12, 4), dpi=200)
        for i in range(self.N_d):
            plt.plot(x, self.G.get_kernel(i).evaluate(x),
                    color='tab:blue', alpha=0.35, linewidth=1.2)
        plt.title(r"Sensitivity Kernels $K_i(x)$", fontsize=18)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel("Kernel Value", fontsize=16)
        plt.grid(True, linestyle=':', alpha=0.4)
        sns.despine()
        plt.tight_layout()
        plt.savefig(self.figures_dir / "sensitivity_kernels.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Target kernels
        plt.figure(figsize=(12, 4), dpi=200)
        for i in range(self.N_p):
            plt.plot(x, self.T.get_kernel(i).evaluate(x),
                    color='tab:orange', alpha=0.6, linewidth=1.5)
        plt.title(r"Target Kernels $T_i(x)$", fontsize=18)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel("Kernel Value", fontsize=16)
        plt.grid(True, linestyle=':', alpha=0.4)
        sns.despine()
        plt.tight_layout()
        plt.savefig(self.figures_dir / "target_kernels.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_synthetic_data(self):
        """Generate synthetic true model and noisy observations."""
        t0 = time.time()

        # True model
        self.m_bar = Function(
            self.M,
            evaluate_callable=lambda x: np.exp(-((x - self.function_domain.center)/0.5)**2)
                                      * np.sin(5 * np.pi * x) + x
        )

        # Generate observations
        self.d_bar = self.G(self.m_bar)
        noise_std = self.noise_level * np.max(self.d_bar)
        np.random.seed(self.random_seed)
        self.d_tilde = self.d_bar + np.random.normal(0, noise_std, self.d_bar.shape)

        # Data noise covariance
        noise_variance = noise_std**2
        C_D_matrix = noise_variance * np.eye(self.N_d)
        self.gaussian_D_noise = GaussianMeasure.from_covariance_matrix(
            self.D, C_D_matrix, expectation=np.zeros(self.N_d)
        )

        self.timings['generate_data'] = time.time() - t0

        # Store metrics
        self.metrics['snr'] = np.max(self.d_bar) / noise_std
        self.metrics['noise_std'] = noise_std

        # Plot true model and data
        self._plot_true_model_and_data()

    def _plot_true_model_and_data(self):
        """Plot true model and synthetic observations."""
        x = np.linspace(self.function_domain.a, self.function_domain.b, 1000)

        # True model
        plt.figure(figsize=(12, 4), dpi=200)
        plt.plot(x, self.m_bar.evaluate(x), color='tab:red', linewidth=2.5, label=r'$\bar{m}(x)$')
        plt.title(r"True Model $\bar{m}(x)$", fontsize=18)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel("Model Value", fontsize=16)
        plt.legend(fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.4)
        sns.despine()
        plt.tight_layout()
        plt.savefig(self.figures_dir / "true_model.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Synthetic observations
        plt.figure(figsize=(12, 4), dpi=200)
        data_indices = np.arange(len(self.d_bar))
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
        plt.savefig(self.figures_dir / "synthetic_observations.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _setup_prior(self):
        """Set up prior measure on model space."""
        t0 = time.time()

        # Setup boundary conditions
        bc_type = self.bc_config['bc_type']
        bc_left = self.bc_config['left']
        bc_right = self.bc_config['right']
        bc = BoundaryConditions(
            bc_type=bc_type, left=bc_left, right=bc_right)

        # Create prior covariance operator based on prior_type
        if self.prior_type == 'inverse_laplacian':
            # C_0 = (αL)^(-1) - simple inverse Laplacian
            C_0 = InverseLaplacian(
                self.M, bc, self.alpha,
                method=self.method, dofs=self.dofs
            )
        elif self.prior_type == 'bessel_sobolev':
            # C_0 = (k²I + L)^(-s) - Bessel-Sobolev inverse
            # First create the Laplacian operator L
            L = Laplacian(
                self.M, bc, self.alpha,
                method=self.method, dofs=self.dofs,
                n_samples=self.n_samples
            )
            # Then create the Bessel-Sobolev inverse
            C_0 = BesselSobolevInverse(
                self.M, self.M, self.k, self.s, L,
                dofs=self.dofs, n_samples=self.n_samples,
                use_fast_transforms=self.use_fast_transforms
            )
        else:
            raise ValueError(
                f"Unknown prior_type: {self.prior_type}. "
                f"Use 'inverse_laplacian' or 'bessel_sobolev'"
            )

        # Prior mean
        m_0 = Function(self.M, evaluate_callable=lambda x: x)

        # Create Gaussian measure with KL expansion using KLSampler
        # This matches the approach in the pli.ipynb notebook
        sampler = KLSampler(C_0, mean=m_0, n_modes=self.K)
        self.M_prior = GaussianMeasure(
            covariance=C_0,
            expectation=m_0,
            sample=sampler.sample
        )

        self.timings['setup_prior'] = time.time() - t0

        # Plot prior
        self._plot_prior()

    def _plot_prior(self):
        """Plot prior measure and property prior."""
        x = np.linspace(self.function_domain.a, self.function_domain.b, 1000)

        # Model prior
        plt.figure(figsize=(12, 5), dpi=200)
        num_samples = 15
        for i in range(num_samples):
            sample = self.M_prior.sample()
            plt.plot(x, sample.evaluate(x), color='tab:blue', alpha=0.25,
                    linewidth=1, label='Prior Samples' if i == 0 else "")

        mean_values = self.M_prior.expectation.evaluate(x)
        plt.plot(x, mean_values, color='tab:green', linewidth=3,
                label='Prior Mean', zorder=10)

        plt.title("Prior Gaussian Measure: Samples and Mean", fontsize=18)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel("Model Value", fontsize=16)
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.4)
        sns.despine()
        plt.tight_layout()
        plt.savefig(self.figures_dir / "prior_measure_on_model_space.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Property prior
        prior_P = self.M_prior.affine_mapping(operator=self.T)
        std_P = np.sqrt(np.diag(prior_P.covariance.matrix(
            dense=True, parallel=True, n_jobs=self.n_jobs)))
        mean_prop = self.T(self.M_prior.expectation)
        true_props = self.T(self.m_bar)

        centers = np.linspace(0.1, 0.9, self.N_p)
        plt.figure(figsize=(12, 5), dpi=200)
        plt.errorbar(centers, mean_prop, yerr=2*std_P, fmt='o', color='tab:blue',
                    alpha=0.7, capsize=4, capthick=2, markersize=6,
                    label='Property Prior (mean ±2σ)')
        plt.scatter(centers, true_props, label='True Properties',
                   color='tab:red', marker='x', s=100, alpha=0.9, linewidths=3)
        plt.xlabel('Target Location', fontsize=16)
        plt.ylabel('Property Value', fontsize=16)
        plt.title('Property Prior: Beliefs Before Data', fontsize=18)
        plt.legend(fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.4)
        sns.despine()
        plt.tight_layout()
        plt.savefig(self.figures_dir / "property_prior_distribution.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _run_inference(self):
        """Run Bayesian inference."""
        print("\nRunning Bayesian inference...")
        t0 = time.time()

        # Setup
        forward_problem = LinearForwardProblem(self.G, data_error_measure=self.gaussian_D_noise)
        bayesian_inference = LinearBayesianInference(forward_problem, self.M_prior, self.T)
        solver = CholeskySolver(parallel=True, n_jobs=self.n_jobs)

        if self.compute_model_posterior:
            # Workflow 1: Full model posterior
            t1 = time.time()
            posterior_model = bayesian_inference.model_posterior_measure(self.d_tilde, solver)
            t2 = time.time()
            self.timings['model_posterior_compute'] = t2 - t1

            t3 = time.time()
            C_M_matrix = posterior_model.covariance.matrix(
                dense=True, parallel=True, n_jobs=self.n_jobs)
            t4 = time.time()
            self.timings['model_covariance_extract'] = t4 - t3

            mu_M = GaussianMeasure.from_covariance_matrix(
                self.M, C_M_matrix, expectation=posterior_model.expectation
            )
            self.m_tilde = mu_M.expectation

            t5 = time.time()
            property_posterior = mu_M.affine_mapping(operator=self.T)
            self.p_tilde = property_posterior.expectation
            cov_P_matrix = property_posterior.covariance.matrix(
                dense=True, parallel=True, n_jobs=self.n_jobs)
            t6 = time.time()
            self.timings['property_posterior_compute'] = t6 - t5

        else:
            # Workflow 2: Property posterior directly
            t1 = time.time()
            posterior_model = bayesian_inference.model_posterior_measure(self.d_tilde, solver)
            self.m_tilde = posterior_model.expectation
            t2 = time.time()
            self.timings['model_posterior_compute'] = t2 - t1

            t3 = time.time()
            property_posterior = posterior_model.affine_mapping(operator=self.T)
            self.p_tilde = property_posterior.expectation
            t4 = time.time()
            self.timings['property_mapping'] = t4 - t3

            t5 = time.time()
            cov_P_matrix = property_posterior.covariance.matrix(
                dense=True, parallel=True, n_jobs=self.n_jobs)
            t6 = time.time()
            self.timings['property_covariance_extract'] = t6 - t5

        self.cov_P_matrix = cov_P_matrix
        self.timings['inference_total'] = time.time() - t0

        print(f"  Inference completed in {self.timings['inference_total']:.2f}s")

        # Plot results
        self._plot_posterior()

    def _plot_posterior(self):
        """Plot posterior results."""
        x = np.linspace(self.function_domain.a, self.function_domain.b, 1000)

        # Model posterior mean
        plt.figure(figsize=(12, 6), dpi=200)
        plt.plot(x, self.m_tilde.evaluate(x), color='tab:blue', linewidth=3,
                label='Posterior Mean', zorder=10)
        plt.plot(x, self.m_bar.evaluate(x), color='tab:red', linestyle='--',
                linewidth=3, label='True Model', zorder=10)
        plt.plot(x, self.M_prior.expectation.evaluate(x), color='tab:green',
                linestyle=':', linewidth=2, alpha=0.8, label='Prior Mean', zorder=5)

        workflow_label = "Workflow 1" if self.compute_model_posterior else "Workflow 2 (Fast)"
        plt.title(f"Model Posterior Mean ({workflow_label})", fontsize=18)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel("Model Value", fontsize=16)
        plt.legend(fontsize=14, loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.4)
        sns.despine()
        plt.tight_layout()
        plt.savefig(self.figures_dir / "model_posterior_mean.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Property inference results
        std_P_post = np.sqrt(np.diag(self.cov_P_matrix))
        true_props = self.T(self.m_bar)
        centers = np.linspace(0.1, 0.9, self.N_p)

        plt.figure(figsize=(12, 6), dpi=200)
        plt.errorbar(centers, self.p_tilde, yerr=2*std_P_post, fmt='o',
                    color='tab:blue', alpha=0.8, capsize=4, capthick=2,
                    markersize=8, linewidth=2, label='Posterior Properties (±2σ)')
        plt.fill_between(centers, self.p_tilde - 2*std_P_post,
                        self.p_tilde + 2*std_P_post, color='tab:blue', alpha=0.2)
        plt.scatter(centers, true_props, label='True Properties',
                   color='tab:red', marker='x', s=120, alpha=0.9, linewidths=4, zorder=10)

        plt.xlabel('Target Location', fontsize=16)
        plt.ylabel('Property Value', fontsize=16)
        plt.title(f'Property Inference Results ({workflow_label})', fontsize=18)
        plt.legend(fontsize=14, loc='best')
        plt.grid(True, linestyle=':', alpha=0.4)
        sns.despine()
        plt.tight_layout()
        plt.savefig(self.figures_dir / "property_inference_results.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _compute_metrics(self):
        """Compute error metrics and data fit."""
        x = np.linspace(self.function_domain.a, self.function_domain.b, 1000)

        # Model reconstruction error
        model_error = np.sqrt(np.mean((self.m_tilde.evaluate(x) - self.m_bar.evaluate(x))**2))
        self.metrics['model_rms_error'] = model_error
        self.metrics['model_relative_error'] = model_error / np.std(self.m_bar.evaluate(x))

        # Property errors
        true_props = self.T(self.m_bar)
        property_errors = np.abs(self.p_tilde - true_props)
        std_P_post = np.sqrt(np.diag(self.cov_P_matrix))

        self.metrics['property_mean_abs_error'] = np.mean(property_errors)
        self.metrics['property_rms_error'] = np.sqrt(np.mean(property_errors**2))
        self.metrics['property_max_error'] = np.max(property_errors)
        self.metrics['properties_within_2sigma'] = int(np.sum(property_errors <= 2*std_P_post))
        self.metrics['properties_within_2sigma_pct'] = 100 * np.sum(property_errors <= 2*std_P_post) / len(true_props)

        # Data fit
        data_misfit_post = np.linalg.norm(self.G(self.m_tilde) - self.d_tilde)
        data_misfit_prior = np.linalg.norm(self.G(self.M_prior.expectation) - self.d_tilde)
        self.metrics['data_misfit_posterior'] = data_misfit_post
        self.metrics['data_misfit_prior'] = data_misfit_prior
        self.metrics['data_fit_improvement_pct'] = 100 * (1 - data_misfit_post / data_misfit_prior)

    def _save_results(self):
        """Save configuration, timings, and metrics to JSON files."""
        # Save configuration
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)

        # Save timings
        with open(self.output_dir / "timings.json", 'w') as f:
            json.dump(self.timings, f, indent=2)

        # Save metrics
        with open(self.output_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)


class PLISweep:
    """Run a parameter sweep of PLI experiments."""

    def __init__(self, base_dir: str = "experiments", sweep_name: Optional[str] = None):
        """
        Initialize parameter sweep.

        Args:
            base_dir: Base directory for all experiments
            sweep_name: Optional descriptive name for the sweep (e.g., 'kl_truncation')
                       If provided, directory will be named '{sweep_name}_{timestamp}'
                       If not provided, directory will be named 'sweep_{timestamp}'
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        # Create sweep directory with optional name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if sweep_name:
            self.sweep_dir = self.base_dir / f"{sweep_name}_{timestamp}"
        else:
            self.sweep_dir = self.base_dir / f"sweep_{timestamp}"
        self.sweep_dir.mkdir()

        print(f"Created sweep directory: {self.sweep_dir}")

    def run(self, param_grid: Dict[str, List[Any]],
            base_config: Optional[Dict[str, Any]] = None):
        """
        Run experiments for all parameter combinations.

        Args:
            param_grid: Dictionary of parameter names to lists of values
            base_config: Base configuration (fixed parameters)

        Returns:
            DataFrame with summary of all runs
        """
        if base_config is None:
            base_config = {}

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        import itertools
        combinations = list(itertools.product(*param_values))

        print(f"\n{'='*80}")
        print(f"PARAMETER SWEEP: {len(combinations)} experiments")
        print(f"{'='*80}")
        print(f"Parameters being varied: {param_names}")
        print(f"Base configuration: {base_config}")
        print(f"{'='*80}\n")

        # Save sweep configuration
        sweep_config = {
            'param_grid': param_grid,
            'base_config': base_config,
            'n_experiments': len(combinations),
            'timestamp': datetime.now().isoformat()
        }
        with open(self.sweep_dir / "sweep_config.json", 'w') as f:
            json.dump(sweep_config, f, indent=2)

        # Run all experiments
        results = []
        for idx, values in enumerate(combinations, 1):
            # Create configuration for this run
            config = base_config.copy()
            for name, value in zip(param_names, values):
                config[name] = value

            # Create run directory
            run_name = f"run_{idx:03d}"
            for name, value in zip(param_names, values):
                run_name += f"_{name}{value}"
            run_dir = self.sweep_dir / run_name

            # Run experiment
            try:
                experiment = PLIExperiment(config, run_dir)
                metrics = experiment.run()

                # Store results
                result = config.copy()
                result.update(metrics)
                result.update(experiment.timings)
                result['run_name'] = run_name
                result['status'] = 'success'
                results.append(result)

            except Exception as e:
                print(f"\n❌ ERROR in {run_name}: {e}\n")
                result = config.copy()
                result['run_name'] = run_name
                result['status'] = 'failed'
                result['error'] = str(e)
                results.append(result)

        # Create summary DataFrame
        df = pd.DataFrame(results)

        # Save summary
        df.to_csv(self.sweep_dir / "summary.csv", index=False)

        # Print summary
        print(f"\n{'='*80}")
        print(f"SWEEP COMPLETE!")
        print(f"{'='*80}")
        print(f"Results saved to: {self.sweep_dir}")
        print(f"Successful runs: {(df['status'] == 'success').sum()}/{len(df)}")
        print(f"{'='*80}\n")

        return df


def example_sweep_K_values():
    """Example: Sweep over different K (KL expansion) values."""
    sweep = PLISweep(sweep_name="example_K_values")

    # Fixed parameters - using Bessel-Sobolev prior by default
    base_config = {
        'N': 100,
        'N_d': 50,
        'N_p': 20,
        'basis': 'sine',
        'noise_level': 0.1,
        'compute_model_posterior': False,
        'random_seed': 42,
        # Bessel-Sobolev prior parameters (default)
        'prior_type': 'bessel_sobolev',
        'k': 1.0,
        's': 1.0,
        'alpha': 0.1,
        'method': 'spectral',
        'dofs': 100,
        'n_samples': 1024,
        'use_fast_transforms': True
    }

    # Varying parameters
    param_grid = {
        'K': [20, 50, 100, 150, 200]
    }

    df = sweep.run(param_grid, base_config)

    # Plot comparison
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    df_success = df[df['status'] == 'success']

    axes[0].plot(df_success['K'], df_success['inference_total'], 'o-')
    axes[0].set_xlabel('K (KL modes)')
    axes[0].set_ylabel('Total inference time (s)')
    axes[0].set_title('Computational Cost vs K')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_success['K'], df_success['property_rms_error'], 'o-')
    axes[1].set_xlabel('K (KL modes)')
    axes[1].set_ylabel('Property RMS error')
    axes[1].set_title('Accuracy vs K')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df_success['K'], df_success['properties_within_2sigma_pct'], 'o-')
    axes[2].set_xlabel('K (KL modes)')
    axes[2].set_ylabel('% properties within 2σ')
    axes[2].set_title('Uncertainty Quantification vs K')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(sweep.sweep_dir / "K_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Comparison plot saved!")

    return df


def example_sweep_multiple_params():
    """Example: Sweep over multiple parameters."""
    sweep = PLISweep(sweep_name="multiple_params")

    base_config = {
        'basis': 'sine',
        'noise_level': 0.1,
        'compute_model_posterior': False,
        'random_seed': 42,
        # Bessel-Sobolev prior parameters (default)
        'prior_type': 'bessel_sobolev',
        'k': 1.0,
        's': 1.0,
        'alpha': 0.1,
        'method': 'spectral',
        'dofs': 100,
        'n_samples': 1024,
        'use_fast_transforms': True
    }

    param_grid = {
        'N': [50, 100],
        'N_d': [25, 50],
        'N_p': [10, 20],
        'K': [50, 100]
    }

    df = sweep.run(param_grid, base_config)
    return df


if __name__ == "__main__":
    # Run example sweep
    print("Running example K-value sweep...")
    df = example_sweep_K_values()

    print("\nResults summary:")
    print(df[['K', 'inference_total', 'property_rms_error',
             'properties_within_2sigma_pct']].to_string(index=False))
