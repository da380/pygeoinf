# pygeoinf

[![CI](https://github.com/da380/pygeoinf/actions/workflows/ci.yml/badge.svg)](https://github.com/da380/pygeoinf/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/pygeoinf.svg)](https://pypi.org/project/pygeoinf/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation Status](https://readthedocs.org/projects/pygeoinf/badge/?version=latest)](https://pygeoinf.readthedocs.io/en/latest/?badge=latest)

Inference and inverse problems on Hilbert spaces, linear and nonlinear,
written without coordinates. A model is a vector in a space with an inner
product the problem chooses; a forward operator maps between such spaces,
linear with its adjoint or nonlinear with its derivative and, where the
code supplies one, its second derivative; a prior is a Gaussian measure on
the model space; and the answer to an inverse problem comes in one of
three kinds, each the same thing whatever the discretization:

- **a point**: a least-squares, minimum-norm, maximum a posteriori or
  Backus-Gilbert estimate for a linear problem, and the maximum a
  posteriori model or Occam's inversion for a nonlinear one, found by
  Newton or quasi-Newton methods that work in the space's own metric;
- **a measure**: the posterior, exactly for a linear problem and as a
  Laplace approximation at the mode otherwise, to be drawn from, pushed
  forward to a property, compared by its evidence, or hardened into a
  credible region;
- **a set**: the models consistent with bounded errors and a bounded
  prior, and the range of any property over them, with no probability
  assumed at all; linearized about a model when the forward operator is
  nonlinear.

The same code runs on a function on a circle, a sphere, a disk or a ball,
on a layered planet, on a finite element space, or on a vector type a C++
code owns, because nothing in it reaches past the space's own arithmetic.
Refining the grid changes the answer only as the discretization does, not
the number of iterations it takes.

## Installation

```
pip install pygeoinf
```

Python 3.12 or later. The core needs NumPy, SciPy, Matplotlib, joblib and
finufft. The extras add function spaces and backends:

| extra | what it brings |
|---|---|
| `sphere` | functions on the two-sphere, with spherical harmonics and map plotting |
| `planetmodel` | spaces on intervals, radial profiles, disks and balls whose correlation length varies, including layered planets |
| `plot3d` | three-dimensional sets drawn as figures that can be turned by hand |
| `osqp`, `clarabel` | alternative backends for the convex solvers |
| `mfem` | PyMFEM, for the finite element mini-app, on Linux |

```
pip install "pygeoinf[sphere,planetmodel]"
```

## A first inversion

A function on a circle from fifteen noisy point values, under a smooth
prior: the posterior is a measure on the function space, and its pointwise
spread is read off it.

```python
import numpy as np
from pygeoinf import GaussianMeasure, LinearForwardProblem, LinearGaussianInversion
from pygeoinf.symmetric_space.circle import Sobolev

rng = np.random.default_rng(7)
X = Sobolev(128, 2.0, 0.2)                      # H^2 functions on the circle
prior = X.heat_measure(0.3, pointwise_std=1.0)  # smooth, with unit spread

points = np.sort(rng.uniform(0.0, 2.0 * np.pi, 15))
forward = X.point_evaluation_operator(points)
noise = GaussianMeasure.from_standard_deviation(forward.codomain, 0.1)
problem = LinearForwardProblem(forward, error=noise)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)

posterior = LinearGaussianInversion(problem, prior)(data)
mean = posterior.expectation
std = np.sqrt(X.pointwise_variance_at(posterior, X.grid_axes[0]))
```

Nothing above is a matrix: the forward operator is applied, its adjoint is
derived, and the solve behind the posterior is conjugate gradients on the
data space. The [first tutorial](tutorials/01_first_inversion.py) continues
from here and draws the result.

## Where to go next

- **[Tutorials](tutorials/README.md)**: an ordered course of scripts, from
  the first inversion to nonlinear problems, coupled fields and writing a
  backend. Each runs top to bottom, opens as a notebook, and has a Colab
  badge.
- **[Examples](examples/README.md)**: short scripts, one idea each, from
  spaces and operators to feasible sets and planet models.
- **[Mini-apps](miniapps/README.md)**: complete applications, each a
  directory with real or realistic data, several features composed, and a
  run of a few minutes: a C++ vector backend, a finite element backend, the
  Earth's radial density, a field on the sphere from ocean data, ray
  tomography in a layered planet, ray theory against finite-frequency
  travel-time kernels, and acoustic waveform inversion with an MPI-split
  solver.
- **[API reference](https://pygeoinf.readthedocs.io/)**: every public class
  and function, with the contract each keeps.

## Development

```
git clone https://github.com/da380/pygeoinf.git
cd pygeoinf
poetry install --with dev --all-extras
poetry run pytest tests
```

The fast suite runs in about five minutes; `-m slow` adds the parity checks
and Monte Carlo estimates that take minutes each. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the hooks and the release process.

## Licence

BSD 3-Clause. Written by David Al-Attar, Dan Heathcote and Adrian Mag.
