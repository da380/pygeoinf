# PLI Automated Experiments

This directory contains tools for running automated parameter sweeps of the Probabilistic Linear Inference (PLI) pipeline.

## Quick Start

```python
from run_pli_experiments import PLISweep

# Create a parameter sweep
sweep = PLISweep(base_dir="experiments")

# Define fixed parameters
base_config = {
    'N': 100,          # Model space dimension
    'N_d': 50,         # Number of data points
    'N_p': 20,         # Number of properties
    'basis': 'sine',   # Basis type
    'alpha': 0.1,      # Prior regularization
    'noise_level': 0.1,  # Data noise (relative to signal)
    'compute_model_posterior': False,  # Use fast workflow
    'random_seed': 42
}

# Define parameters to vary
param_grid = {
    'K': [20, 50, 100, 150, 200]  # KL expansion modes
}

# Run sweep
results_df = sweep.run(param_grid, base_config)

# Results are saved automatically in experiments/sweep_YYYYMMDD_HHMMSS/
```

## Example Use Cases

### 1. Test different KL expansion sizes

```python
from run_pli_experiments import example_sweep_K_values

# Run predefined K-value sweep
df = example_sweep_K_values()

# Analyze results
print(df[['K', 'inference_total', 'property_rms_error']])
```

### 2. Compare different discretizations

```python
sweep = PLISweep()

base_config = {
    'K': 100,
    'basis': 'sine',
    'alpha': 0.1,
    'noise_level': 0.1,
    'compute_model_posterior': False,
    'random_seed': 42
}

param_grid = {
    'N': [50, 100, 200],      # Model resolution
    'N_d': [25, 50, 100],     # Data density
    'N_p': [10, 20, 40]       # Property resolution
}

df = sweep.run(param_grid, base_config)
```

### 3. Test different basis functions

```python
param_grid = {
    'basis': ['sine', 'cosine', 'fourier'],
    'N': [50, 100]
}

df = sweep.run(param_grid, base_config)
```

### 4. Noise sensitivity analysis

```python
param_grid = {
    'noise_level': [0.01, 0.05, 0.1, 0.2, 0.5],
    'K': [50, 100, 200]
}

df = sweep.run(param_grid, base_config)

# Plot noise vs accuracy
import matplotlib.pyplot as plt
for k in [50, 100, 200]:
    subset = df[df['K'] == k]
    plt.plot(subset['noise_level'], subset['property_rms_error'],
             'o-', label=f'K={k}')
plt.xlabel('Noise Level')
plt.ylabel('Property RMS Error')
plt.legend()
plt.savefig('noise_sensitivity.png')
```

## Output Structure

Each sweep creates a timestamped directory:

```
experiments/
└── sweep_20251028_143025/
    ├── sweep_config.json          # Overall sweep configuration
    ├── summary.csv                 # Results from all runs
    ├── K_comparison.png           # Comparison plots (if generated)
    │
    ├── run_001_K20/
    │   ├── config.json            # Run-specific parameters
    │   ├── timings.json           # Execution timings
    │   ├── metrics.json           # Error metrics, data fit
    │   └── figures/
    │       ├── sensitivity_kernels.png
    │       ├── target_kernels.png
    │       ├── true_model.png
    │       ├── synthetic_observations.png
    │       ├── prior_measure_on_model_space.png
    │       ├── property_prior_distribution.png
    │       ├── model_posterior_mean.png
    │       └── property_inference_results.png
    │
    ├── run_002_K50/
    │   └── ...
    │
    └── run_003_K100/
        └── ...
```

## Available Metrics

Each run computes and saves:

### Timings
- `total`: Total experiment time
- `setup_spaces`: Space creation time
- `create_operators`: Operator setup time
- `generate_data`: Data generation time
- `setup_prior`: Prior setup time
- `inference_total`: Total inference time
- `model_posterior_compute`: Model posterior computation
- `property_covariance_extract`: Property covariance extraction

### Accuracy Metrics
- `model_rms_error`: Model reconstruction RMS error
- `model_relative_error`: Relative model error
- `property_mean_abs_error`: Mean absolute property error
- `property_rms_error`: Property RMS error
- `property_max_error`: Maximum property error
- `properties_within_2sigma`: Count within uncertainty bounds
- `properties_within_2sigma_pct`: Percentage within bounds

### Data Fit
- `data_misfit_posterior`: Posterior data fit
- `data_misfit_prior`: Prior data fit
- `data_fit_improvement_pct`: Percentage improvement
- `snr`: Signal-to-noise ratio
- `noise_std`: Noise standard deviation

## Configuration Parameters

### Required Parameters
- `N`: Model space dimension (int)
- `N_d`: Number of data points (int)
- `N_p`: Number of properties to infer (int)
- `K`: Number of KL expansion modes (int)
- `basis`: Basis function type (str: 'sine', 'cosine', 'fourier')
- `alpha`: Prior regularization parameter (float)
- `noise_level`: Relative noise level (float, e.g., 0.1 = 10%)

### Optional Parameters
- `compute_model_posterior`: Use workflow 1 (True) or 2 (False). Default: False
- `random_seed`: Random seed for reproducibility. Default: 42

## Advanced Usage

### Single Experiment

```python
from run_pli_experiments import PLIExperiment
from pathlib import Path

config = {
    'N': 100,
    'N_d': 50,
    'N_p': 20,
    'K': 100,
    'basis': 'sine',
    'alpha': 0.1,
    'noise_level': 0.1,
    'compute_model_posterior': False,
    'random_seed': 42
}

experiment = PLIExperiment(config, Path("my_experiment"))
metrics = experiment.run()
print(metrics)
```

### Custom Analysis

```python
import pandas as pd
import json

# Load results from a previous sweep
sweep_dir = "experiments/sweep_20251028_143025"
df = pd.read_csv(f"{sweep_dir}/summary.csv")

# Load detailed metrics from a specific run
with open(f"{sweep_dir}/run_001_K20/metrics.json") as f:
    metrics = json.load(f)

# Load timings
with open(f"{sweep_dir}/run_001_K20/timings.json") as f:
    timings = json.load(f)
```

## Tips

1. **Start small**: Test with a single parameter variation before large sweeps
2. **Use fast workflow**: Set `compute_model_posterior=False` for faster experiments
3. **Parallel runs**: The code uses parallelization internally, but you can run multiple sweeps on different machines
4. **Check failed runs**: The summary.csv includes a `status` column
5. **Compare visually**: Each run saves all figures for visual comparison

## Examples in Repository

See the bottom of `run_pli_experiments.py` for complete examples:
- `example_sweep_K_values()`: K-value sweep with comparison plots
- `example_sweep_multiple_params()`: Multi-parameter sweep
