"""
Demo 1 Parameter Sweep Framework

A comprehensive system for running systematic parameter sweeps on the
multi-component Bayesian inference problem demonstrated in demo_1.ipynb.

Main components:
- Demo1Config: Configuration dataclass for all parameters
- Demo1Experiment: Single experiment runner
- Demo1Sweep: Parameter sweep orchestrator
- Analysis tools: Visualization and statistical analysis

Quick start:
    >>> from demo_1_config import get_standard_config
    >>> from demo_1_experiment import run_single_experiment
    >>> from pathlib import Path
    >>>
    >>> config = get_standard_config()
    >>> results = run_single_experiment(config, Path("./test_run"))

See README_DEMO1_SWEEP.md for full documentation.
"""

from .demo_1_config import (
    Demo1Config,
    get_fast_config,
    get_standard_config,
    get_high_resolution_config,
    get_posterior_sampling_config,
)

from .demo_1_experiment import (
    Demo1Experiment,
    run_single_experiment,
)

from .demo_1_sweep import (
    Demo1Sweep,
    ConditionalSweep,
    create_prior_sensitivity_sweep,
    create_resolution_sweep,
    create_noise_sensitivity_sweep,
    create_kl_truncation_sweep,
    create_component_coupling_sweep,
    create_comprehensive_sweep,
    create_adaptive_resolution_sweep,
)

from .demo_1_analysis import (
    load_sweep_results,
    plot_parameter_sensitivity,
    plot_convergence_analysis,
    plot_timing_analysis,
    plot_heatmap_2d,
    plot_component_comparison,
    plot_noise_sensitivity,
    generate_sweep_report,
    compare_sweeps,
)

__all__ = [
    # Config
    'Demo1Config',
    'get_fast_config',
    'get_standard_config',
    'get_high_resolution_config',
    'get_posterior_sampling_config',

    # Experiment
    'Demo1Experiment',
    'run_single_experiment',

    # Sweep
    'Demo1Sweep',
    'ConditionalSweep',
    'create_prior_sensitivity_sweep',
    'create_resolution_sweep',
    'create_noise_sensitivity_sweep',
    'create_kl_truncation_sweep',
    'create_component_coupling_sweep',
    'create_comprehensive_sweep',
    'create_adaptive_resolution_sweep',

    # Analysis
    'load_sweep_results',
    'plot_parameter_sensitivity',
    'plot_convergence_analysis',
    'plot_timing_analysis',
    'plot_heatmap_2d',
    'plot_component_comparison',
    'plot_noise_sensitivity',
    'generate_sweep_report',
    'compare_sweeps',
]

__version__ = '1.0.0'
