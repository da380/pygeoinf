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
* **Extensive Solver Suite**: Choose from direct factorisations or matrix-free iterative solvers (e.g., CG, MINRES, LSQR, BiCGStab, FCG) designed for abstract vectors.
* **Coupled Systems**: Effortlessly build block matrices and direct sum spaces using `HilbertSpaceDirectSum` for joint inversions.
* **Geometric Constraints**: Constrain your inversions using affine subspaces, bounds, and convex sublevel sets.
* **Probabilistic Modelling**: Define priors and noise models using `GaussianMeasure` objects on abstract spaces.
* **Randomized Algorithms**: Utilizes randomized SVD and Cholesky decompositions for efficient low-rank approximations of large operators.
* **Specialized Operator Variants**: Efficient implementations for `SparseMatrixLinearOperator` and `DiagonalSparseMatrixLinearOperator`
* **Application-Specific Spaces**: Provides concrete `HilbertSpace` implementations for functions on a **line**, **circle**, and the **two-sphere**.
* **High-Quality Visualisation**: Built-in plotting methods for functions on symmetric spaces, now featuring multi-dimensional corner plots for joint posterior distributions.

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

You can install all optional dependencies for development—including tools for running the test suite, building the documentation, and running the Jupyter tutorials—by using the `--with` flag and specifying the `dev` group.

```bash
    # Install all development dependencies (for tests, docs, and tutorials)
    poetry install --with dev
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