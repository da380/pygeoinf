"""
PLI (Property-Level Inference) Demos Package.

This package provides modular components for running PLI experiments:

- pli_config: Configuration dataclass for experiment parameters
- pli_experiment: Main experiment runner
- pli_analysis: Analysis and visualization utilities
- pli_sweep: Parameter sweep functionality

Example usage:

    from pygeoinf.interval.demos.pli_demos import (
        PLIConfig, PLIExperiment, PLISweep
    )

    # Quick experiment
    config = PLIConfig.get_standard_config()
    experiment = PLIExperiment(config)
    results = experiment.run()

    # Parameter sweep
    sweep = PLISweep(config)
    df = sweep.sweep_parameter('K', [10, 20, 30, 40, 50])
"""

from .pli_config import PLIConfig
from .pli_experiment import PLIExperiment
from .pli_analysis import (
    load_experiment_results,
    load_sweep_results,
    plot_convergence_study,
    plot_parameter_comparison,
    plot_timing_breakdown,
    plot_accuracy_vs_time,
    generate_summary_table,
    compare_experiments,
    print_experiment_summary
)
from .pli_sweep import (
    PLISweep,
    create_K_sweep_configs,
    create_noise_sweep_configs,
    create_resolution_sweep_configs,
    create_prior_sweep_configs,
    run_standard_sweeps
)

__all__ = [
    # Config
    'PLIConfig',

    # Experiment
    'PLIExperiment',

    # Analysis
    'load_experiment_results',
    'load_sweep_results',
    'plot_convergence_study',
    'plot_parameter_comparison',
    'plot_timing_breakdown',
    'plot_accuracy_vs_time',
    'generate_summary_table',
    'compare_experiments',
    'print_experiment_summary',

    # Sweep
    'PLISweep',
    'create_K_sweep_configs',
    'create_noise_sweep_configs',
    'create_resolution_sweep_configs',
    'create_prior_sweep_configs',
    'run_standard_sweeps',
]
