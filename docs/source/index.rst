pygeoinf
========

Inference and inverse problems on Hilbert spaces, linear and nonlinear,
written without coordinates. A model is a vector in a space with an inner
product the problem chooses; a forward operator maps between such spaces,
linear with its adjoint or nonlinear with its derivative and, where the
code supplies one, its second derivative; a prior is a Gaussian measure on
the model space; and the answer to an inverse problem comes in one of
three kinds, each the same thing whatever the discretization:

* **a point**: a least-squares, minimum-norm, maximum a posteriori or
  Backus-Gilbert estimate for a linear problem, and the maximum a
  posteriori model or Occam's inversion for a nonlinear one, found by
  Newton or quasi-Newton methods that work in the space's own metric;
* **a measure**: the posterior, exactly for a linear problem and as a
  Laplace approximation at the mode otherwise, to be drawn from, pushed
  forward to a property, compared by its evidence, or hardened into a
  credible region;
* **a set**: the models consistent with bounded errors and a bounded
  prior, and the range of any property over them, with no probability
  assumed at all; linearized about a model when the forward operator is
  nonlinear.

The same code runs on a function on a circle, a sphere, a disk or a ball,
on a layered planet, on a finite element space, or on a vector type a C++
code owns, because nothing in it reaches past the space's own arithmetic.

Installation
------------

.. code-block:: bash

   pip install pygeoinf
   pip install "pygeoinf[sphere,planetmodel]"

Python 3.12 or later. The extras: ``sphere`` for functions on the
two-sphere, ``planetmodel`` for intervals, radial profiles, disks, balls
and layered planets, ``plot3d`` for three-dimensional sets, ``osqp`` and
``clarabel`` for alternative convex backends, and ``mfem`` for the finite
element mini-app on Linux.

Learning the library
--------------------

The `tutorials <https://github.com/da380/pygeoinf/tree/main/tutorials>`_
are an ordered course of scripts, from a first inversion to nonlinear
problems, coupled fields and writing a backend; each opens as a notebook and
runs in Google Colab from its badge. The
`examples <https://github.com/da380/pygeoinf/tree/main/examples>`_ are
short scripts, one idea each. The
`mini-apps <https://github.com/da380/pygeoinf/tree/main/miniapps>`_ are
complete applications: a C++ vector backend, a finite element backend, the
Earth's radial density, a field on the sphere from ocean data, ray
tomography in a layered planet, ray theory against finite-frequency
travel-time kernels, and acoustic waveform inversion.

The reference below documents every public class and function, with the
contract each keeps.

.. toctree::
   :maxdepth: 4
   :caption: API Reference
   :hidden:

   modules
