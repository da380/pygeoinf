.. pygeoinf documentation master file, created by
   sphinx-quickstart on Fri Aug 15 14:06:35 2025.

pygeoinf: A Python Library for Geophysical Inference
=====================================================


The core philosophy of `pygeoinf` is to separate the abstract mathematical structure of an inverse problem from its concrete numerical implementation. Instead of manipulating NumPy arrays directly, you work with high-level objects like `HilbertSpace`, `LinearOperator`, and `GaussianMeasure`.

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   tutorials

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/hilbert_space
   api/operators
   api/forms
   api/gaussian_measure
   api/direct_sum
   api/linear_solvers
   api/forward_problem
   api/inversion
   api/linear_optimisation
   api/linear_bayesian
   api/random_matrix
   api/symmetric_spaces

