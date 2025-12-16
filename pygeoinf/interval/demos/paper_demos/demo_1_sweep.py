"""
Parameter sweep orchestrator for Demo 1 experiments.

This module implements the Demo1Sweep class that manages running multiple
experiments with different parameter configurations, similar to PLISweep
but adapted for the multi-component structure of Demo 1.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import itertools
from datetime import datetime
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .demo_1_config import Demo1Config
from .demo_1_experiment import Demo1Experiment


class Demo1Sweep:
    """Orchestrate parameter sweeps for Demo 1 experiments.

    This class manages running multiple experiments with different parameter
    configurations, collecting results, and generating summary statistics.
    It supports:
    - Cartesian product sweeps over multiple parameters
    - Conditional parameter sweeps
    - Parallel execution
    - Comprehensive result aggregation
    """

    def __init__(
        self,
        base_config: Demo1Config,
        sweep_params: Dict[str, List[Any]],
        output_dir: Path,
        name: str = "demo_1_sweep",
        parallel: bool = False,
        max_workers: Optional[int] = None
    ):
        """Initialize parameter sweep.

        Args:
            base_config: Base configuration to start from
            sweep_params: Dictionary mapping parameter names to lists of values
                         e.g., {'s_vp': [1.5, 2.0, 2.5], 'N': [50, 100, 200]}
            output_dir: Base directory for all sweep results
            name: Name for this sweep
            parallel: Whether to run experiments in parallel
            max_workers: Maximum number of parallel workers (None = all cores)
        """
        self.base_config = base_config
        self.sweep_params = sweep_params
        self.output_dir = Path(output_dir)
        self.name = name
        self.parallel_execution = parallel
        self.max_workers = max_workers

        # Create sweep directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sweep_dir = self.output_dir / f"{name}_{timestamp}"
        self.sweep_dir.mkdir(parents=True, exist_ok=True)

        # Generate all parameter combinations
        self.param_combinations = self._generate_combinations()

        # Storage for results
        self.results: List[Dict[str, Any]] = []

    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations using Cartesian product.

        Returns:
            List of dictionaries, each containing one parameter combination
        """
        # Get parameter names and value lists
        param_names = list(self.sweep_params.keys())
        param_values = [self.sweep_params[name] for name in param_names]

        # Generate Cartesian product
        combinations = []
        for values in itertools.product(*param_values):
            combo = dict(zip(param_names, values))
            combinations.append(combo)

        return combinations

    def _create_config(self, params: Dict[str, Any], run_id: int) -> Demo1Config:
        """Create a configuration for a specific parameter combination.

        Args:
            params: Dictionary of parameter values
            run_id: Unique ID for this run

        Returns:
            Demo1Config with specified parameters
        """
        # Start with base config
        config_dict = self.base_config.to_dict()

        # Remove derived parameters
        for key in ['k_vp', 'k_vs', 'k_rho', 'alpha_vp', 'alpha_vs', 'alpha_rho']:
            config_dict.pop(key, None)

        # Update with sweep parameters
        config_dict.update(params)

        # Update name and description
        param_str = "_".join([f"{k}={v}" for k, v in params.items()])
        config_dict['name'] = f"run_{run_id:03d}_{param_str}"
        config_dict['description'] = f"Sweep run {run_id} with {param_str}"

        return Demo1Config(**config_dict)

    def _run_single(self, run_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment with specified parameters.

        Args:
            run_id: Unique ID for this run
            params: Dictionary of parameter values

        Returns:
            Dictionary containing results
        """
        # Create configuration
        config = self._create_config(params, run_id)

        # Create output directory for this run
        run_dir = self.sweep_dir / f"run_{run_id:03d}"
        run_dir.mkdir(exist_ok=True)

        # Run experiment
        experiment = Demo1Experiment(config, run_dir)
        results = experiment.run()

        # Add parameter values to results
        results['params'] = params
        results['run_id'] = run_id

        return results

    def run(self, progress_bar: bool = True) -> pd.DataFrame:
        """Run all experiments in the sweep.

        Args:
            progress_bar: Whether to show progress bar

        Returns:
            DataFrame containing all results
        """
        n_runs = len(self.param_combinations)
        print(f"Starting sweep: {self.name}")
        print(f"Number of runs: {n_runs}")
        print(f"Output directory: {self.sweep_dir}")
        print(f"Parallel execution: {self.parallel_execution}")

        # Save sweep configuration
        sweep_config = {
            'name': self.name,
            'n_runs': n_runs,
            'base_config': self.base_config.to_dict(),
            'sweep_params': {k: [str(v) for v in vals]
                           for k, vals in self.sweep_params.items()},
            'timestamp': datetime.now().isoformat()
        }
        with open(self.sweep_dir / 'sweep_config.json', 'w') as f:
            json.dump(sweep_config, f, indent=2)

        if self.parallel_execution:
            # Parallel execution
            self.results = []
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all jobs
                futures = {
                    executor.submit(self._run_single, i, params): i
                    for i, params in enumerate(self.param_combinations)
                }

                # Collect results with progress bar
                if progress_bar:
                    iterator = tqdm(as_completed(futures), total=n_runs,
                                  desc="Running experiments")
                else:
                    iterator = as_completed(futures)

                for future in iterator:
                    try:
                        result = future.result()
                        self.results.append(result)
                    except Exception as e:
                        run_id = futures[future]
                        print(f"\nError in run {run_id}: {e}")
        else:
            # Sequential execution
            iterator = enumerate(self.param_combinations)
            if progress_bar:
                iterator = tqdm(iterator, total=n_runs, desc="Running experiments")

            self.results = []
            for run_id, params in iterator:
                try:
                    result = self._run_single(run_id, params)
                    self.results.append(result)
                except Exception as e:
                    print(f"\nError in run {run_id}: {e}")

        # Create summary DataFrame
        df = self._create_summary_dataframe()

        # Save summary
        df.to_csv(self.sweep_dir / 'summary.csv', index=False)

        print(f"\nSweep complete!")
        print(f"Results saved to: {self.sweep_dir}")

        return df

    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Create a summary DataFrame from all results.

        Returns:
            DataFrame with one row per run
        """
        rows = []
        for result in self.results:
            row = {}

            # Add run info
            row['run_id'] = result['run_id']

            # Add parameters
            row.update(result['params'])

            # Add metrics
            row.update(result['metrics'])

            # Add timings
            for key, val in result['timings'].items():
                row[f'time_{key}'] = val

            rows.append(row)

        return pd.DataFrame(rows)

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze sweep results and compute summary statistics.

        Returns:
            Dictionary containing analysis results
        """
        df = self._create_summary_dataframe()

        analysis = {}

        # Compute statistics for each swept parameter
        for param in self.sweep_params.keys():
            param_analysis = {}

            # Group by parameter value
            grouped = df.groupby(param)

            # Compute mean and std for key metrics
            metrics = ['vp_rel_l2_error', 'vs_rel_l2_error', 'rho_rel_l2_error',
                      'sigma_0_rel_error', 'sigma_1_rel_error',
                      'data_rel_l2_error', 'time_total']

            for metric in metrics:
                if metric in df.columns:
                    param_analysis[f'{metric}_mean'] = grouped[metric].mean().to_dict()
                    param_analysis[f'{metric}_std'] = grouped[metric].std().to_dict()

            analysis[param] = param_analysis

        # Find best configuration for each metric
        best_configs = {}
        metrics = ['vp_rel_l2_error', 'vs_rel_l2_error', 'rho_rel_l2_error']
        for metric in metrics:
            if metric in df.columns:
                best_idx = df[metric].idxmin()
                best_configs[metric] = {
                    'run_id': int(df.loc[best_idx, 'run_id']),
                    'value': float(df.loc[best_idx, metric]),
                    'params': {k: df.loc[best_idx, k] for k in self.sweep_params.keys()}
                }

        analysis['best_configurations'] = best_configs

        # Overall statistics
        analysis['overall'] = {
            'n_runs': len(df),
            'mean_total_time': float(df['time_total'].mean()),
            'std_total_time': float(df['time_total'].std()),
            'min_total_time': float(df['time_total'].min()),
            'max_total_time': float(df['time_total'].max()),
        }

        # Save analysis
        with open(self.sweep_dir / 'analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)

        return analysis


def create_prior_sensitivity_sweep(
    output_dir: Path,
    s_values: List[float] = [1.5, 2.0, 2.5, 3.0],
    length_scale_values: List[float] = [0.1, 0.2, 0.3, 0.4],
    component: str = 'vp'
) -> Demo1Sweep:
    """Create a sweep exploring prior sensitivity for one component.

    Args:
        output_dir: Directory for sweep results
        s_values: Smoothness values to sweep
        length_scale_values: Length scale values to sweep
        component: Component to sweep ('vp', 'vs', or 'rho')

    Returns:
        Configured Demo1Sweep object
    """
    from .demo_1_config import get_standard_config

    base_config = get_standard_config()

    sweep_params = {
        f's_{component}': s_values,
        f'length_scale_{component}': length_scale_values
    }

    return Demo1Sweep(
        base_config=base_config,
        sweep_params=sweep_params,
        output_dir=output_dir,
        name=f"prior_sensitivity_{component}",
        parallel=True
    )


def create_resolution_sweep(
    output_dir: Path,
    N_values: List[int] = [50, 100, 150, 200],
    N_d_values: List[int] = [25, 50, 75, 100]
) -> Demo1Sweep:
    """Create a sweep exploring resolution effects.

    Args:
        output_dir: Directory for sweep results
        N_values: Basis function counts to sweep
        N_d_values: Data point counts to sweep

    Returns:
        Configured Demo1Sweep object
    """
    from .demo_1_config import get_standard_config

    base_config = get_standard_config()

    sweep_params = {
        'N': N_values,
        'N_d': N_d_values
    }

    return Demo1Sweep(
        base_config=base_config,
        sweep_params=sweep_params,
        output_dir=output_dir,
        name="resolution_sweep",
        parallel=True
    )


def create_noise_sensitivity_sweep(
    output_dir: Path,
    noise_values: List[float] = [0.001, 0.005, 0.01, 0.05, 0.1]
) -> Demo1Sweep:
    """Create a sweep exploring noise sensitivity.

    Args:
        output_dir: Directory for sweep results
        noise_values: Noise levels to sweep

    Returns:
        Configured Demo1Sweep object
    """
    from .demo_1_config import get_standard_config

    base_config = get_standard_config()

    sweep_params = {
        'noise_level': noise_values
    }

    return Demo1Sweep(
        base_config=base_config,
        sweep_params=sweep_params,
        output_dir=output_dir,
        name="noise_sensitivity",
        parallel=True
    )


def create_kl_truncation_sweep(
    output_dir: Path,
    kl_values: List[Optional[int]] = [10, 20, 50, 100, None]
) -> Demo1Sweep:
    """Create a sweep exploring KL truncation effects.

    Args:
        output_dir: Directory for sweep results
        kl_values: KL truncation values to sweep (None = no truncation)

    Returns:
        Configured Demo1Sweep object
    """
    from .demo_1_config import get_standard_config

    base_config = get_standard_config()

    sweep_params = {
        'kl_truncation_vp': kl_values,
        'kl_truncation_vs': kl_values,
        'kl_truncation_rho': kl_values
    }

    return Demo1Sweep(
        base_config=base_config,
        sweep_params=sweep_params,
        output_dir=output_dir,
        name="kl_truncation",
        parallel=True
    )


def create_component_coupling_sweep(
    output_dir: Path,
    smoothness_values: List[float] = [1.5, 2.0, 2.5]
) -> Demo1Sweep:
    """Create a sweep exploring different smoothness for each component.

    This creates a large sweep exploring different combinations of smoothness
    parameters for vp, vs, and rho to study component coupling effects.

    Args:
        output_dir: Directory for sweep results
        smoothness_values: Smoothness values for all components

    Returns:
        Configured Demo1Sweep object
    """
    from .demo_1_config import get_standard_config

    base_config = get_standard_config()

    sweep_params = {
        's_vp': smoothness_values,
        's_vs': smoothness_values,
        's_rho': smoothness_values
    }

    return Demo1Sweep(
        base_config=base_config,
        sweep_params=sweep_params,
        output_dir=output_dir,
        name="component_coupling",
        parallel=True
    )


def create_comprehensive_sweep(
    output_dir: Path,
    N_values: List[int] = [50, 100],
    s_values: List[float] = [1.5, 2.0, 2.5],
    noise_values: List[float] = [0.01, 0.05]
) -> Demo1Sweep:
    """Create a comprehensive sweep exploring multiple parameters.

    Warning: This creates N_values × s_values³ × noise_values experiments,
    which can be very large (e.g., 2 × 3³ × 2 = 108 runs).

    Args:
        output_dir: Directory for sweep results
        N_values: Resolution values
        s_values: Smoothness values (applied to all components)
        noise_values: Noise level values

    Returns:
        Configured Demo1Sweep object
    """
    from .demo_1_config import get_standard_config

    base_config = get_standard_config()

    sweep_params = {
        'N': N_values,
        's_vp': s_values,
        's_vs': s_values,
        's_rho': s_values,
        'noise_level': noise_values
    }

    return Demo1Sweep(
        base_config=base_config,
        sweep_params=sweep_params,
        output_dir=output_dir,
        name="comprehensive",
        parallel=True
    )


class ConditionalSweep(Demo1Sweep):
    """Extended sweep with conditional parameter relationships.

    This allows creating sweeps where some parameters depend on others,
    e.g., N_d = N/2, or length_scale = 1/(2*N).
    """

    def __init__(
        self,
        base_config: Demo1Config,
        sweep_params: Dict[str, List[Any]],
        derived_params: Dict[str, Callable[[Dict[str, Any]], Any]],
        output_dir: Path,
        name: str = "conditional_sweep",
        parallel: bool = False,
        max_workers: Optional[int] = None
    ):
        """Initialize conditional sweep.

        Args:
            base_config: Base configuration
            sweep_params: Primary parameters to sweep
            derived_params: Dictionary mapping parameter names to functions
                          that compute their values from sweep parameters
            output_dir: Output directory
            name: Sweep name
            parallel: Whether to run in parallel
            max_workers: Maximum parallel workers
        """
        self.derived_params = derived_params
        super().__init__(base_config, sweep_params, output_dir, name,
                        parallel, max_workers)

    def _create_config(self, params: Dict[str, Any], run_id: int) -> Demo1Config:
        """Create configuration with derived parameters."""
        # Compute derived parameters
        for param_name, param_func in self.derived_params.items():
            params[param_name] = param_func(params)

        # Use parent method
        return super()._create_config(params, run_id)


def create_adaptive_resolution_sweep(
    output_dir: Path,
    N_values: List[int] = [50, 100, 150, 200]
) -> ConditionalSweep:
    """Create sweep where N_d and N_p scale with N.

    Uses relationships: N_d = N/2, N_p = N/5

    Args:
        output_dir: Directory for sweep results
        N_values: Basis function counts to sweep

    Returns:
        Configured ConditionalSweep object
    """
    from .demo_1_config import get_standard_config

    base_config = get_standard_config()

    sweep_params = {
        'N': N_values
    }

    derived_params = {
        'N_d': lambda p: p['N'] // 2,
        'N_p': lambda p: p['N'] // 5
    }

    return ConditionalSweep(
        base_config=base_config,
        sweep_params=sweep_params,
        derived_params=derived_params,
        output_dir=output_dir,
        name="adaptive_resolution",
        parallel=True
    )
