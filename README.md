# pygeoinf: A Python Library for Geophysical Inference

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com)
[![PyPI version](https://img.shields.io/pypi/v/pygeoinf.svg)](https://pypi.org/project/pygeoinf/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation Status](https://readthedocs.org/projects/pygeoinf/badge/?version=latest)](https://pygeoinf.readthedocs.io/en/latest/?badge=latest)

**pygeoinf** is a Python library for solving geophysical inference and inverse problems in a coordinate-free, abstract framework. It leverages the mathematics of Hilbert spaces to provide a robust and flexible foundation for Bayesian and optimisation-based inference.

## Overview

The core philosophy of `pygeoinf` is to separate the abstract mathematical structure of an inverse problem from its concrete numerical implementation. Instead of manipulating NumPy arrays directly, you work with high-level objects like `HilbertSpace`, `LinearOperator`, and `GaussianMeasure`. This allows you to write code that is more readable, less error-prone, and closer to the underlying mathematics.

The library is built on a few key concepts:

* **`HilbertSpace`**: The foundational class. It represents a vector space with an inner product, but it abstracts away the specific representation of vectors (e.g., NumPy arrays, `pyshtools` grids).
* **`LinearOperator`**: Represents linear mappings between Hilbert spaces. These are the workhorses of the library, supporting composition, adjoints, and matrix representations.
* **`GaussianMeasure`**: Generalizes the multivariate normal distribution to abstract Hilbert spaces, providing a way to define priors and noise models.
* **`ForwardProblem`**: Encapsulates the mathematical model `d = A(u) + e`, linking the unknown model `u` to the observed data `d`.
* **Inversion Classes**: High-level classes like `LinearBayesianInversion` and `LinearLeastSquaresInversion` provide ready-to-use algorithms for solving the inverse problem.

## Key Features

* **Abstract Coordinate-Free Formulation**: Write elegant code that mirrors the mathematics of inverse problems.
* **Bayesian Inference**: Solve inverse problems in a probabilistic framework to obtain posterior distributions over models.
* **Optimisation Methods**: Includes Tikhonov-regularized least-squares and minimum-norm solutions.
* **Probabilistic Modelling**: Define priors and noise models using `GaussianMeasure` objects on abstract spaces.
* **Randomized Algorithms**: Utilizes randomized SVD and Cholesky decompositions for efficient low-rank approximations of large operators.
* **Specialized Operator Variants**: Efficient implementations for `DiagonalSparseMatrixLinearOperator` and `NormalSumOperator`.
* **Application-Specific Spaces**: Provides concrete `HilbertSpace` implementations for functions on a **line**, **circle**, and the **two-sphere**.
* **High-Quality Visualisation**: Built-in plotting methods for functions on symmetric spaces, including map projections via `cartopy`.

## Advanced Features

* **Block Operators**: Construct complex operators from smaller components using `BlockLinearOperator`, `ColumnLinearOperator`, and `RowLinearOperator`. This is ideal for coupled inverse problems.
* **Parallelisation**: Many expensive operations are parallelized with `joblib`, including dense matrix construction and randomized algorithms.

## Installation

The package can be installed directly using pip. By default, this will perform a minimal installation.

```bash
# Minimal installation
pip install pygeoinf
```

To include the functionality for functions on the sphere, you can install the `sphere` extra. This provides support for `pyshtools` and `Cartopy`.

```bash
# Installation with sphere-related features
pip install pygeoinf[sphere]
```

For development, you can clone the repository and install using Poetry:

```bash
git clone https://github.com/da380/pygeoinf.git
cd pygeoinf
poetry install
```

You can install all optional dependencies for development—including tools for running the test suite, 
building the documentation, and running the Jupyter tutorials—by using the ```--with``` flag and specifying the ```dev``` group.

```bash
# Install all development dependencies (for tests, docs, and tutorials)
poetry install --with dev
```



## Documentation

The full documentation for the library, including the API reference and tutorials, is available at **[pygeoinf.readthedocs.io](https://pygeoinf.readthedocs.io)**.



## Tutorials

You can run the interactive tutorials directly in Google Colab to get started with the core concepts of the library.


| Tutorial Name                 | Link to Colab                                                                                                                                                                                                                                    |
| :---------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tutorial 1 - A first example  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/docs/source/tutorials/tutorial1.ipynb)                                                                                  |
| Tutorial 2 - Hilbert spaces   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/docs/source/tutorials/tutorial2.ipynb)                                                                                  |
| Tutorial 3 - Dual spaces      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/docs/source/tutorials/tutorial3.ipynb)                                                                                  |
| Tutorial 4 - Linear operators | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/docs/source/tutorials/tutorial4.ipynb)                                                                                  |
| Tutorial 5 - Linear solvers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/docs/source/tutorials/tutorial5.ipynb)                                                                                  |
| Tutorial 6 - Gaussian measures| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/docs/source/tutorials/tutorial6.ipynb)                                                                                  |
| Tutorial 7 - Minimum norm inversions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/docs/source/tutorials/tutorial7.ipynb)                                                                          |
| Tutorial 8 - Bayesian inversions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/docs/source/tutorials/tutorial8.ipynb)                                                                              |
| Tutorial 9 - Direct sums          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/docs/source/tutorials/tutorial9.ipynb)                                                                              |
| Tutorial 10 - Symmetric spaces        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/docs/source/tutorials/tutorial10.ipynb)                                                                              |



## Quick Start: Bayesian Inference on a Circle
Here is a simple example of performing Bayesian inference on a function defined on a circle. We will define a prior, observe the function at a few noisy points, and compute the posterior mean and variance.

```python
import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.circle import Sobolev


# Set up the model space.
# We define the space of possible solutions. Here, we use a Sobolev space
# on a circle. This space contains functions that are twice-differentiable
# (order=2) with a characteristic length scale of 0.05. The `from_sobolev_parameters`
# method automatically determines the necessary Fourier resolution (kmax).
model_space = Sobolev.from_sobolev_parameters(2, 0.05)


# Set the sample points randomly.
# We will observe the "true" function at 20 random points on the circle.
n = 20
observation_points = model_space.random_points(n)

# Set the forward operator using a method of the Sobolev class.
# The `point_evaluation_operator` is a linear operator that takes a function
# from our model space and returns its values at the specified points.
forward_operator = model_space.point_evaluation_operator(observation_points)
data_space = forward_operator.codomain

# Set the data error measure. If standard deviation is zero, the data is
# free of observational errors.
# We assume the observations are corrupted by Gaussian noise with a
# standard deviation of 0.1. We represent this with a GaussianMeasure.
standard_deviation = 0.1
data_error_measure = (
    inf.GaussianMeasure.from_standard_deviation(data_space, standard_deviation)
    if standard_deviation > 0
    else None
)

# Set up the forward problem.
# The `LinearForwardProblem` object bundles the operator and the error model.
# This fully defines the relationship: `data = operator(model) + error`.
forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)

# Define a prior measure on the model space.
# We define a prior belief about what the "true" function looks like before
# seeing any data. Here, we use a `heat_gaussian_measure`, which generates
# smooth functions. We give it a mean of zero and a pointwise amplitude of 1.
model_prior_measure = model_space.heat_gaussian_measure(0.1, 1)

# Sample a model and corresponding data.
# To test our inversion, we first create a "true" model by drawing a random
# sample from our prior. Then, we generate the corresponding noisy data.
model, data = forward_problem.synthetic_model_and_data(model_prior_measure)

# Set up the inversion method.
# We set up the Bayesian inversion, providing the forward problem and our prior.
bayesian_inversion = inf.LinearBayesianInversion(forward_problem, model_prior_measure)

# Get the posterior distribiution.
# We solve the inversion to get the posterior distribution. The posterior
# represents our updated belief about the model after incorporating the data.
# A Cholesky solver is used for the underlying matrix inversion.
model_posterior_measure = bayesian_inversion.model_posterior_measure(
    data, inf.CholeskySolver()
)

# Estimate the pointwise variance.
# To visualize the posterior uncertainty, we create a low-rank approximation
# of the posterior, which allows us to draw samples efficiently.
low_rank_posterior_approximation = model_posterior_measure.low_rank_approximation(
    10, method="variable", rtol=1e-4
)
# We then estimate the pointwise variance by drawing many samples from this approximation.
model_pointwise_variance = low_rank_posterior_approximation.sample_pointwise_variance(
    100
)
model_pointwise_std = np.sqrt(model_pointwise_variance)


# Plot the results.
# Create the final plot, starting with the true underlying model.
fig, ax = model_space.plot(
    model, color="k", figsize=(15, 10), linestyle="--", label="True Model"
)
# Overlay the noisy data points that were used for the inversion.
ax.errorbar(
    observation_points,
    data,
    2 * standard_deviation,
    fmt="ko",
    capsize=2,
    label="Noisy Data",
)
# Plot the posterior mean, which is our best estimate of the true model.
model_space.plot(
    model_posterior_measure.expectation,
    fig=fig,
    ax=ax,
    color="b",
    label="Posterior Mean",
)
# Plot the 2-sigma uncertainty bounds around the posterior mean.
model_space.plot_error_bounds(
    model_posterior_measure.expectation,
    2 * model_pointwise_std,
    fig=fig,
    ax=ax,
    alpha=0.2,
    color="b",
    label="Posterior (2 std dev)",
)
# Add titles and labels for clarity
ax.set_title("Bayesian Inversion Results")
ax.set_xlabel("Angle (radians)")
ax.set_ylabel("Function Value")
ax.legend()
# Display the final plot.
plt.show()
```

The output of the above script will look similar to the following figure:

![Example of Bayesian Inference on a Circle](docs/source/figures/fig1.png)

## Dependencies

* NumPy
* SciPy
* Matplotlib
* joblib

### Optional (`sphere`)

* pyshtools
* Cartopy

## Recent Updates

`pygeoinf` is under active development, and recent updates have expanded its capabilities to address a broader range of geophysical problems. Key improvements include:

* **Non-Linear Inverse Problems**: The library now supports non-linear inverse problems through the `NonLinearOperator` and `NonLinearForm` classes. This allows you to define and solve problems where the relationship between model parameters and data is non-linear. The framework includes support for Fréchet derivatives, enabling the use of gradient-based optimisation methods.

* **Advanced Optimisation Algorithms**: To complement the non-linear framework, `pygeoinf` now includes a suite of optimisation algorithms. You can use the custom `GradientDescent` solver or leverage the power of SciPy's optimization library with methods like 'BFGS', 'L-BFGS-B', and 'Newton-CG'. Derivative-free methods such as 'Powell' and 'Nelder-Mead' are also supported for problems where gradients are unavailable.

* **Backus-Gilbert Methods**: The foundations for the Backus-Gilbert method have been implemented. While this feature is still evolving, the core components are in place for you to begin exploring this powerful inference technique.

## Future Plans

Future development will focus on the following areas:

* **Non-linear Bayesian Inference**: We plan to develop methods for non-linear Bayesian problems, including techniques for linearizing the problem around the maximum a posteriori (MAP) solution to estimate posterior uncertainty. This will also involve constructing efficient proposal distributions for Markov chain Monte Carlo (MCMC) sampling methods.

* **New Geophysical Hilbert Spaces**: We will be adding more `HilbertSpace` implementations for specific geophysical applications. A key focus will be on creating spaces for functions defined within a **spherical annulus** (spherical shell), which is crucial for problems in global seismology and mantle tomography.


## Contributing
Contributions are welcome! If you would like to contribute, please feel free to fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the BSD-3-Clause License - see the LICENSE file for details.
