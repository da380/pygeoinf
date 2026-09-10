.. pygeoinf documentation master file, created by
   sphinx-quickstart on Fri Aug 15 14:06:35 2025.

pygeoinf: A Python Library for Geophysical Inference
=====================================================

**pygeoinf** is a Python library for solving geophysical inference and inverse problems in a coordinate-free, abstract framework. It leverages the mathematics of Hilbert spaces to provide a robust and flexible foundation for Bayesian and optimisation-based inference.

Overview
========

The core philosophy of `pygeoinf` is to separate the abstract mathematical structure of an inverse problem from its concrete numerical implementation. Instead of manipulating NumPy arrays directly, you work with high-level objects like `HilbertSpace`, `LinearOperator`, and `GaussianMeasure`. This allows you to write code that is more readable, less error-prone, and closer to the underlying mathematics.

The library is built on a few key concepts:

* **`HilbertSpace`**: The foundational class. It represents a vector space with an inner product, but it abstracts away the specific representation of vectors (e.g., NumPy arrays, `pyshtools` grids).
* **`LinearOperator`**: Represents linear mappings between Hilbert spaces. These are the workhorses of the library, supporting composition, adjoints, and matrix representations.
* **`GaussianMeasure`**: Generalizes the multivariate normal distribution to abstract Hilbert spaces, providing a way to define priors and noise models.
* **`ForwardProblem`**: Encapsulates the mathematical model `d = A(u) + e`, linking the unknown model `u` to the observed data `d`.
* **Inversion Classes**: High-level classes like `LinearBayesianInversion` and `LinearLeastSquaresInversion` provide ready-to-use algorithms for solving the inverse problem.
 
Key Features
============

* **Abstract Coordinate-Free Formulation**: Write elegant code that mirrors the mathematics of inverse problems.
* **Bayesian Inference**: Solve inverse problems in a probabilistic framework to obtain posterior distributions over models.
* **Optimisation Methods**: Includes Tikhonov-regularized least-squares and minimum-norm solutions.
* **Probabilistic Modelling**: Define priors and noise models using `GaussianMeasure` objects on abstract spaces.
* **Randomized Algorithms**: Utilizes randomized SVD and Cholesky decompositions for efficient low-rank approximations of large operators.
* **Application-Specific Spaces**: Provides concrete `HilbertSpace` implementations for functions on a **line**, **circle**, and the **two-sphere**.
* **High-Quality Visualisation**: Built-in plotting methods for functions on symmetric spaces, including map projections via `cartopy`.

Tutorials
=========

You can run the interactive tutorials directly in Google Colab to get started with the core concepts of the library.

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Tutorial Name
     - Link to Colab
   * - Tutorial 1 - A first example
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial1.ipynb
          :alt: Open Tutorial 1 in Colab
   * - Tutorial 2 - Hilbert spaces
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial2.ipynb
          :alt: Open Tutorial 2 in Colab
   * - Tutorial 3 - Dual spaces
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial3.ipynb
          :alt: Open Tutorial 3 in Colab
   * - Tutorial 4 - Linear operators
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial4.ipynb
          :alt: Open Tutorial 4 in Colab
   * - Tutorial 5 - Linear solvers
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial5.ipynb
          :alt: Open Tutorial 5 in Colab
   * - Tutorial 6 - Gaussian measures
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial6.ipynb
          :alt: Open Tutorial 6 in Colab
   * - Tutorial 7 - Minimum norm inversions
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial7.ipynb
          :alt: Open Tutorial 7 in Colab
   * - Tutorial 8 - Bayesian inversions
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial8.ipynb
          :alt: Open Tutorial 8 in Colab
   * - Tutorial 9 - Direct sums
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial9.ipynb
          :alt: Open Tutorial 9 in Colab
   * - Tutorial 10 - Symmetric spaces
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/tutorial10.ipynb
          :alt: Open Tutorial 10 in Colab


.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:


.. toctree::
   :maxdepth: 4
   :caption: API Reference
   :hidden:

   modules


   