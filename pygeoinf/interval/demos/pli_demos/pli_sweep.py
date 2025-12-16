"""
Parameter sweep functionality for PLI experiments.

This module provides classes and functions for running systematic parameter
sweeps over PLI configurations.
"""

import numpy as np
import pandas as pd
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import fields
from itertools import product
import logging

from .pli_config import PLIConfig
from .pli_experiment import PLIExperiment


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PLISweep:
    """Run parameter sweeps over PLI experiments.

    This class provides functionality to systematically explore the parameter
    space of PLI experiments, with support for grid search, parallel execution,
    and result aggregation.

    Attributes:
        base_config: Base configuration to use for all experiments
        output_dir: Directory to save sweep results
        results: DataFrame containing all experiment results

    Example:
        >>> base_config = PLIConfig.get_standard_config()
        >>> sweep = PLISweep(base_config, output_dir=Path("./sweep_results"))
        >>> sweep.sweep_parameter('K', [10, 20, 30, 40, 50])
        >>> sweep.results  # Access DataFrame with all results
    """

    def __init__(
        self,
        base_config: PLIConfig,
        output_dir: Optional[Path] = None,
        verbose: bool = True
    ):
        """Initialize the sweep.

        Args:
            base_config: Base configuration to use
            output_dir: Directory to save results
            verbose: Whether to print progress
        """
        self.base_config = base_config
        self.verbose = verbose
        self.results_list: List[Dict[str, Any]] = []
        self.results: Optional[pd.DataFrame] = None

        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"./pli_sweep_{timestamp}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save base config
        base_config.save(self.output_dir / "base_config.json")

        if verbose:
            logger.info(f"PLI Sweep initialized. Output: {self.output_dir}")

    def _create_config(self, **param_overrides) -> PLIConfig:
        """Create a new config with parameter overrides.

        Args:
            **param_overrides: Parameters to override

        Returns:
            New PLIConfig with overrides applied
        """
        # Get base config as dict
        config_dict = {}
        for field in fields(self.base_config):
            config_dict[field.name] = getattr(self.base_config, field.name)

        # Apply overrides
        config_dict.update(param_overrides)

        return PLIConfig(**config_dict)

    def _run_single_experiment(
        self,
        config: PLIConfig,
        run_name: str
    ) -> Dict[str, Any]:
        """Run a single experiment and return results.

        Args:
            config: Configuration for this run
            run_name: Name for this run

        Returns:
            Dictionary with config, metrics, timings, and status
        """
        run_dir = self.output_dir / run_name
        config.output_dir = str(run_dir)
        config.name = run_name

        result = {
            'run_name': run_name,
            'status': 'success'
        }

        # Add config parameters
        for field in fields(config):
            value = getattr(config, field.name)
            if not isinstance(value, (list, dict, Path)):
                result[field.name] = value

        try:
            experiment = PLIExperiment(config)
            output = experiment.run()

            # Add metrics
            result.update(output.get('metrics', {}))
            result.update(output.get('timings', {}))

        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            if self.verbose:
                logger.error(f"Run {run_name} failed: {e}")

        return result

    def sweep_parameter(
        self,
        param_name: str,
        param_values: List[Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> pd.DataFrame:
        """Sweep over a single parameter.

        Args:
            param_name: Name of parameter to sweep
            param_values: List of values to try
            progress_callback: Optional callback(current, total) for progress

        Returns:
            DataFrame with results
        """
        if self.verbose:
            logger.info(f"Sweeping {param_name} over {len(param_values)} values: {param_values}")

        n_runs = len(param_values)

        for i, value in enumerate(param_values):
            if progress_callback:
                progress_callback(i, n_runs)
            elif self.verbose:
                logger.info(f"  Run {i+1}/{n_runs}: {param_name}={value}")

            run_name = f"run_{param_name}_{i:03d}"
            config = self._create_config(**{param_name: value})
            result = self._run_single_experiment(config, run_name)
            self.results_list.append(result)

        self.results = pd.DataFrame(self.results_list)
        self._save_results()

        return self.results

    def sweep_grid(
        self,
        param_grid: Dict[str, List[Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> pd.DataFrame:
        """Sweep over a grid of parameters.

        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            progress_callback: Optional callback(current, total) for progress

        Returns:
            DataFrame with results
        """
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        n_runs = len(combinations)

        if self.verbose:
            logger.info(f"Grid sweep: {len(param_names)} parameters, {n_runs} combinations")
            for name, values in param_grid.items():
                logger.info(f"  {name}: {values}")

        for i, combo in enumerate(combinations):
            if progress_callback:
                progress_callback(i, n_runs)
            elif self.verbose:
                param_str = ", ".join(f"{k}={v}" for k, v in zip(param_names, combo))
                logger.info(f"  Run {i+1}/{n_runs}: {param_str}")

            run_name = f"run_{i:04d}"
            overrides = dict(zip(param_names, combo))
            config = self._create_config(**overrides)
            result = self._run_single_experiment(config, run_name)
            self.results_list.append(result)

        self.results = pd.DataFrame(self.results_list)
        self._save_results()

        return self.results

    def sweep_configs(
        self,
        configs: List[PLIConfig],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> pd.DataFrame:
        """Run experiments with a list of pre-defined configs.

        Args:
            configs: List of PLIConfig objects
            progress_callback: Optional callback(current, total) for progress

        Returns:
            DataFrame with results
        """
        n_runs = len(configs)

        if self.verbose:
            logger.info(f"Running {n_runs} pre-defined configurations")

        for i, config in enumerate(configs):
            if progress_callback:
                progress_callback(i, n_runs)
            elif self.verbose:
                logger.info(f"  Run {i+1}/{n_runs}: {config.name}")

            run_name = f"run_{i:04d}_{config.name}" if config.name else f"run_{i:04d}"
            result = self._run_single_experiment(config, run_name)
            self.results_list.append(result)

        self.results = pd.DataFrame(self.results_list)
        self._save_results()

        return self.results

    def _save_results(self) -> None:
        """Save current results to disk."""
        if self.results is not None:
            # Save as CSV
            self.results.to_csv(self.output_dir / "summary.csv", index=False)

            # Save as JSON for more precision
            self.results.to_json(
                self.output_dir / "summary.json",
                orient='records',
                indent=2
            )

            if self.verbose:
                logger.info(f"Results saved to {self.output_dir}")

    def get_best_run(
        self,
        metric: str = 'property_rms_error',
        minimize: bool = True
    ) -> Dict[str, Any]:
        """Get the run with the best metric value.

        Args:
            metric: Metric to optimize
            minimize: Whether to minimize (True) or maximize (False)

        Returns:
            Dictionary with best run's parameters and results
        """
        if self.results is None or self.results.empty:
            return {}

        df = self.results[self.results['status'] == 'success']

        if df.empty:
            return {}

        if minimize:
            idx = df[metric].idxmin()
        else:
            idx = df[metric].idxmax()

        return df.loc[idx].to_dict()

    def print_summary(self) -> None:
        """Print a summary of the sweep results."""
        if self.results is None or self.results.empty:
            print("No results available.")
            return

        df = self.results

        print("\n" + "=" * 70)
        print("PLI SWEEP SUMMARY")
        print("=" * 70)

        print(f"\nTotal runs: {len(df)}")
        print(f"Successful: {(df['status'] == 'success').sum()}")
        print(f"Failed: {(df['status'] == 'failed').sum()}")

        # Metrics summary for successful runs
        successful = df[df['status'] == 'success']

        if not successful.empty:
            print("\nMetric Statistics (successful runs):")

            metrics = [
                'model_rms_error', 'property_rms_error',
                'properties_within_2sigma_pct', 'total'
            ]

            for metric in metrics:
                if metric in successful.columns:
                    values = successful[metric].dropna()
                    if not values.empty:
                        print(f"\n  {metric}:")
                        print(f"    Min:  {values.min():.6f}")
                        print(f"    Max:  {values.max():.6f}")
                        print(f"    Mean: {values.mean():.6f}")
                        print(f"    Std:  {values.std():.6f}")

            # Best run
            best = self.get_best_run('property_rms_error', minimize=True)
            if best:
                print(f"\nBest run (by property_rms_error): {best.get('run_name', 'N/A')}")
                print(f"  Property RMS error: {best.get('property_rms_error', 'N/A'):.6f}")
                print(f"  Model RMS error: {best.get('model_rms_error', 'N/A'):.6f}")

        print("\n" + "=" * 70)


def create_K_sweep_configs(
    base_config: PLIConfig,
    K_values: Optional[List[int]] = None
) -> List[PLIConfig]:
    """Create configs for sweeping KL truncation level K.

    Args:
        base_config: Base configuration
        K_values: List of K values to try (default: [5, 10, 20, 30, 40, 50, 60])

    Returns:
        List of PLIConfig objects
    """
    if K_values is None:
        K_values = [5, 10, 20, 30, 40, 50, 60]

    configs = []
    for K in K_values:
        config_dict = {}
        for field in fields(base_config):
            config_dict[field.name] = getattr(base_config, field.name)

        config_dict['K'] = K
        config_dict['name'] = f"K_{K}"
        configs.append(PLIConfig(**config_dict))

    return configs


def create_noise_sweep_configs(
    base_config: PLIConfig,
    noise_levels: Optional[List[float]] = None
) -> List[PLIConfig]:
    """Create configs for sweeping noise level.

    Args:
        base_config: Base configuration
        noise_levels: List of noise levels to try

    Returns:
        List of PLIConfig objects
    """
    if noise_levels is None:
        noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    configs = []
    for noise in noise_levels:
        config_dict = {}
        for field in fields(base_config):
            config_dict[field.name] = getattr(base_config, field.name)

        config_dict['noise_level'] = noise
        config_dict['name'] = f"noise_{noise:.3f}"
        configs.append(PLIConfig(**config_dict))

    return configs


def create_resolution_sweep_configs(
    base_config: PLIConfig,
    N_values: Optional[List[int]] = None
) -> List[PLIConfig]:
    """Create configs for sweeping model space resolution.

    Args:
        base_config: Base configuration
        N_values: List of N values to try

    Returns:
        List of PLIConfig objects
    """
    if N_values is None:
        N_values = [50, 100, 150, 200, 300]

    configs = []
    for N in N_values:
        config_dict = {}
        for field in fields(base_config):
            config_dict[field.name] = getattr(base_config, field.name)

        config_dict['N'] = N
        config_dict['name'] = f"N_{N}"
        configs.append(PLIConfig(**config_dict))

    return configs


def create_prior_sweep_configs(
    base_config: PLIConfig,
    s_values: Optional[List[float]] = None,
    length_scale_values: Optional[List[float]] = None
) -> List[PLIConfig]:
    """Create configs for sweeping prior parameters.

    Args:
        base_config: Base configuration
        s_values: List of smoothness values
        length_scale_values: List of length scale values

    Returns:
        List of PLIConfig objects
    """
    if s_values is None:
        s_values = [1.0, 1.5, 2.0, 2.5, 3.0]

    if length_scale_values is None:
        length_scale_values = [0.05, 0.1, 0.2, 0.3]

    configs = []
    for s in s_values:
        for length_scale in length_scale_values:
            config_dict = {}
            for field in fields(base_config):
                config_dict[field.name] = getattr(base_config, field.name)

            config_dict['s'] = s
            config_dict['length_scale'] = length_scale
            config_dict['name'] = f"s_{s:.1f}_ls_{length_scale:.2f}"
            configs.append(PLIConfig(**config_dict))

    return configs


def run_standard_sweeps(
    output_base: Path,
    base_config: Optional[PLIConfig] = None
) -> Dict[str, pd.DataFrame]:
    """Run a suite of standard parameter sweeps.

    Args:
        output_base: Base directory for outputs
        base_config: Base configuration (default: get_fast_config())

    Returns:
        Dictionary mapping sweep name to results DataFrame
    """
    if base_config is None:
        base_config = PLIConfig.get_fast_config()

    output_base = Path(output_base)
    results = {}

    # K sweep
    logger.info("Running K sweep...")
    sweep = PLISweep(base_config, output_dir=output_base / "sweep_K")
    results['K'] = sweep.sweep_parameter('K', [10, 20, 30, 40, 50])
    sweep.print_summary()

    # Noise sweep
    logger.info("Running noise sweep...")
    sweep = PLISweep(base_config, output_dir=output_base / "sweep_noise")
    results['noise'] = sweep.sweep_parameter('noise_level', [0.01, 0.02, 0.05, 0.1])
    sweep.print_summary()

    # Resolution sweep
    logger.info("Running resolution sweep...")
    sweep = PLISweep(base_config, output_dir=output_base / "sweep_N")
    results['N'] = sweep.sweep_parameter('N', [50, 100, 150])
    sweep.print_summary()

    return results
