# pygeoinf: A Python Library for Geophysical Inference

[![CI](https://github.com/da380/pygeoinf/actions/workflows/ci.yml/badge.svg)](https://github.com/da380/pygeoinf/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/pygeoinf.svg)](https://pypi.org/project/pygeoinf/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation Status](https://readthedocs.org/projects/pygeoinf/badge/?version=latest)](https://pygeoinf.readthedocs.io/en/latest/?badge=latest)

**pygeoinf** is a Python library for solving geophysical inference and inverse problems in a coordinate-free, abstract framework. It leverages the mathematics of Hilbert spaces to provide a robust and flexible foundation for Bayesian, optimisation-based, and set-based inference.

## Overview

The core philosophy of `pygeoinf` is to separate the abstract mathematical structure of an inverse problem from its concrete numerical implementation. Instead of manipulating NumPy arrays directly, you work with high-level objects such as `HilbertSpace`, `LinearOperator`, and `GaussianMeasure`. This allows you to write code that is more readable, less error-prone, and closer to the underlying mathematics.

The library is built around a small number of key ideas:

* **`HilbertSpace`**: The foundational class. It represents a real vector space with an inner product, while abstracting away the concrete representation of its vectors (NumPy arrays, `pyshtools` grids, and so on). A space is defined mathematically by its Riesz map, so that the pairing between a space and its dual is always explicit.
* **`LinearOperator` and `NonLinearOperator`**: Mappings between Hilbert spaces. Linear operators support composition, addition, adjoints, duals, and matrix representations, and can be defined either through a matrix or purely through their action on vectors. Non-linear operators carry their Fréchet derivative as a linear operator.
* **`LinearForm` / `NonLinearForm`**: Functionals on a space, including gradients and, where appropriate, Hessians and subgradients.
* **`GaussianMeasure`**: Generalises the multivariate normal distribution to abstract Hilbert spaces, providing a natural language for priors, noise models, and posterior distributions.
* **Sets and subspaces**: `Ball`, `Ellipsoid`, `LinearSubspace`, `AffineSubspace`, and their unions, intersections, and complements, used to express prior bounds and confidence regions geometrically.
* **`ForwardProblem` / `LinearForwardProblem`**: Encapsulates the model `d = A(u) + e`, linking an unknown model `u` to observed data `d` through a forward operator and a data error measure.
* **Inversion and inference classes**: An *inversion* estimates the model itself (`LinearBayesianInversion`, `LinearLeastSquaresInversion`, `LinearMinimumNormInversion`); an *inference* estimates a chosen property of the model, defined by a property operator mapping the model space into a property space.

## Quick Start

A complete Bayesian inversion for a function on a circle, observed at a set of points:

```python
import numpy as np
import pygeoinf as inf
from pygeoinf.symmetric_space.circle import Sobolev

# 1. The model space: functions on a circle, within a Sobolev space of order 2.
model_space = Sobolev(256, 2.0, 0.1)

# 2. The prior: a Gaussian measure with a smooth, translation-invariant covariance.
model_prior_measure = model_space.heat_kernel_gaussian_measure(0.05)

# 3. The forward operator: point measurements at twenty locations.
points = np.random.uniform(0.0, 2 * np.pi, 20)
forward_operator = model_space.point_evaluation_operator(points)

# 4. The forward problem, d = A(u) + e, with independent Gaussian errors.
data_error_measure = inf.GaussianMeasure.from_standard_deviation(
    forward_operator.codomain, 0.1
)
forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)

# 5. A synthetic model drawn from the prior, along with its noisy data.
model, data = forward_problem.synthetic_model_and_data(model_prior_measure)

# 6. The Bayesian solution: a posterior measure on the model space.
inversion = inf.LinearBayesianInversion(forward_problem, model_prior_measure)
model_posterior_measure = inversion.model_posterior_measure(data, inf.CGSolver())

# 7. Point estimates and uncertainties.
model_estimate = model_posterior_measure.expectation
pointwise_std = model_posterior_measure.sample_pointwise_std(100)
```

Note that nothing in steps 4 to 7 refers to the circle. The same code solves the
corresponding problem on the sphere, on a torus, or on any other Hilbert space,
simply by changing the model space and forward operator.

## Key Features

### Spaces and operators

* **Abstract coordinate-free formulation**: Write code that mirrors the mathematics of inverse problems, independent of how vectors are stored.
* **Dual spaces**: `DualHilbertSpace` lets linear functionals be treated as vectors in their own right, with the Riesz map handled consistently throughout.
* **Specialised space types**: `EuclideanSpace` for R^n, `OrthogonalHilbertSpace` and `OrthonormalHilbertSpace` for spaces with known bases, and `MassWeightedHilbertSpace` for Galerkin and finite-element style representations.
* **Coupled systems**: Build joint problems from separate ones using `HilbertSpaceDirectSum` together with `BlockLinearOperator`, `ColumnLinearOperator`, `RowLinearOperator`, and `BlockDiagonalLinearOperator`.
* **Efficient operator variants**: `DenseMatrixLinearOperator`, `SparseMatrixLinearOperator`, and `DiagonalSparseMatrixLinearOperator` where a matrix representation is available, alongside fully matrix-free operators where it is not.
* **Affine operators**: `AffineOperator` for mappings of the form `x -> A(x) + b`, including the propagation of covariances and inverse covariances.

### Probability and uncertainty quantification

* **Gaussian measures on abstract spaces**: Construct measures from standard deviations, covariance matrices, covariance factors, samples, or direct sums, then push them forward through affine mappings.
* **Posterior statistics**: Posterior expectations, exact posterior sampling by randomise-then-optimise, pointwise variances and standard deviations, two-point covariances, and directional statistics.
* **Credible sets**: `GaussianMeasure.credible_set` returns probability-calibrated regions in several geometries: the classical Mahalanobis ellipsoid, the Cameron–Martin unit ball, an ambient norm ball, and weakened-covariance ellipsoids. Radii are calibrated using accurate weighted chi-square quantiles, computed by saddlepoint or Imhof methods, so that the sets remain meaningful in high and infinite dimensions.
* **Measure comparison and model selection**: KL divergence, nuclear and Hilbert–Schmidt norms, and the log evidence of a linear Bayesian problem, with stochastic log-determinant estimation for large problems.

### Solvers and preconditioners

* **Direct solvers**: LU, Cholesky, and eigendecomposition-based solvers for problems with an explicit matrix representation.
* **Matrix-free iterative solvers**: CG, MINRES, BiCGStab, and flexible CG, written directly against abstract vectors, along with wrappers for the corresponding SciPy matrix solvers. Callbacks are provided for progress reporting and for tracking solutions and residuals.
* **Preconditioners**: A suite of general strategies (Jacobi, spectral, banded, exact block, column-thresholded, and iterative) plus preconditioners tailored to Bayesian normal equations, including diagonal, sparse localised, and Woodbury data-space forms.
* **Randomised algorithms**: Randomised SVD, eigendecomposition, and Cholesky factorisation for low-rank approximation of large operators, together with stochastic estimators of traces and diagonals.
* **Functional calculus**: `LanczosOperatorFunction` evaluates `f(A)v` for a self-adjoint operator `A` without forming `A`, which underlies square roots, inverse square roots, and logarithms of covariance operators.

### Inversion and inference

* **Bayesian inversion**: `LinearBayesianInversion` returns the full posterior measure, not merely a point estimate.
* **Optimisation methods**: Tikhonov-regularised least-squares and minimum-norm inversions, each available in a constrained form where the solution is restricted to an affine subspace or a convex set.
* **Backus–Gilbert inference**: Given a property operator, a prior norm bound, and a significance level, the `BackusInference` and `DualMasterCostFunction` classes of `pygeoinf.backus_gilbert` compute bounds on properties of the model by dual-level-set methods, rather than estimating the model itself.
* **Non-linear problems**: `ScipyUnconstrainedOptimiser` adapts a `NonLinearForm` for use with `scipy.optimize`, with derivative information supplied through the form's gradient and Hessian.

### Function spaces on symmetric domains

* **Concrete spaces**: Lebesgue and Sobolev spaces of functions on the **line**, **circle**, **plane**, **torus**, and **two-sphere**. Spherical harmonic expansions use `pyshtools`; the plane and torus use non-uniform FFTs through `finufft`.
* **Invariant Gaussian measures**: Translation- and rotation-invariant priors specified through their spectrum, with convenient heat kernel and Sobolev kernel forms, and norm-scaled or point-value-scaled variants for setting amplitudes directly. `CorrelatedInvariantGaussianMeasure` extends this to several correlated fields sharing a Karhunen–Loève expansion.
* **Space construction from a prior**: Factory methods such as `from_heat_kernel_prior` and `from_sobolev_parameters` choose a truncation degree automatically, given the prior and a relative tolerance.
* **Point evaluation and localised observations**: Dirac functionals, point evaluation operators, and spherical cap averages provide the usual link between a continuous field and discrete measurements.
* **Visualisation**: Plotting routines for each domain, including `cartopy` map projections on the sphere, together with one-dimensional distribution plots, corner plots for joint posteriors, and slice plots through high-dimensional sets and measures.

### Performance and verification

* **Parallelisation**: Expensive operations, including dense matrix construction and the randomised algorithms, are parallelised with `joblib`; `configure_threading` controls the underlying BLAS thread count when doing so.
* **Axiom checks**: Spaces and operators provide a `check()` method that runs randomised tests of the relevant axioms — inner product properties, Riesz identities, adjoint definitions, and finite-difference tests of derivatives. This makes it straightforward to validate a new user-defined space or operator.
* **Datasets**: Helpers for loading Global Seismographic Network station locations and for downloading and sampling USGS earthquake catalogues, for use in realistic test problems.

## Installation

The package can be installed directly using pip. By default, this performs a minimal installation.

```bash
# Minimal installation
pip install pygeoinf
```

To include functionality for functions on the sphere, install the `sphere` extra. This adds support for `pyshtools`, `Cartopy` and `shapely`.

```bash
# Installation with sphere-related features
pip install pygeoinf[sphere]
```

The full set of optional extras is:

| Extra | Installs | Enables |
| --- | --- | --- |
| `sphere` | `pyshtools`, `Cartopy`, `shapely` | Function spaces on the two-sphere, with geospatial plotting |
| `osqp` | `osqp` | The OSQP quadratic programming backend |
| `clarabel` | `clarabel` | The Clarabel quadratic programming backend |
| `interactive` | `plotly` | Interactive plotting |

Extras can be combined, for example `pip install pygeoinf[sphere,interactive]`. Everything else in the library works with the minimal installation.

For development, clone the repository and install using Poetry:

```bash
git clone https://github.com/da380/pygeoinf.git
cd pygeoinf
poetry install
```

The `dev` group provides the tools for running the test suite, building the documentation, and running the Jupyter tutorials. The extras are needed as well, since parts of the test suite and documentation cover the optional features:

```bash
# Install all development dependencies (for tests, docs, and tutorials)
poetry install --with dev --all-extras
```

## Documentation

The full documentation for the library, including the API reference and tutorials, is available at **[pygeoinf.readthedocs.io](https://pygeoinf.readthedocs.io)**.

## Tutorials

You can run the interactive tutorials directly in Google Colab to get started with the core concepts of the library.

| Tutorial Name | Link to Colab |
| :--- | :--- |
| Tutorial 1 - A first example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial1.ipynb) |
| Tutorial 2 - Hilbert spaces | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial2.ipynb) |
| Tutorial 3 - Dual spaces | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial3.ipynb) |
| Tutorial 4 - Linear operators | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial4.ipynb) |
| Tutorial 5 - Linear solvers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial5.ipynb) |
| Tutorial 6 - Gaussian measures | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial6.ipynb) |
| Tutorial 7 - Minimum norm inversions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial7.ipynb) |
| Tutorial 8 - Bayesian inversions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial8.ipynb) |
| Tutorial 9 - Direct sums | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial9.ipynb) |
| Tutorial 10 - Symmetric spaces | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial10.ipynb) |

## Contributing

Contributions are welcome! If you would like to contribute, please feel free to fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the BSD-3-Clause License - see the LICENSE file for details.
