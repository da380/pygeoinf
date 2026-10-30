# Examples

Scripts meant to be read in order. The early ones are short, one idea each;
the later ones are worked problems on real geometries, longer because the
problem is. Every one runs on its own:

```
poetry run python examples/05_derivative_and_gradient.py
```

They are executed by the test suite, so an example that stops working is a
failing test rather than a surprise later.

## The sequence

**Spaces and operators**

| | | |
|---|---|---|
| 1 | `spaces` | vectors are raw backend objects; the space does the arithmetic |
| 2 | `coordinates` | a basis is a capability, not a requirement; the Gram matrix |
| 3 | `operators` | adjoints, and where the metric enters |
| 4 | `traits` | structure that survives the algebra |
| 5 | `derivative_and_gradient` | **the one to read twice** |
| 6 | `nonlinear` | `at()`, and why value and derivative come together |
| 7 | `no_coordinates` | a Hilbert space whose vectors are Python lists: conjugate gradients and a Gaussian run on it, and what needs coordinates refuses it by name |

**Numerics**

| | | |
|---|---|---|
| 8 | `solvers` | coordinate-free Krylov; declared preconditions |
| 9 | `functional_calculus` | `f(A)` without a matrix |
| 10 | `randomized` | low-rank factorization and trace estimation |
| 11 | `optimization` | why working in the metric changes the iteration count |
| 12 | `evaluation_failure` | a forward that cannot converge raises, a line search backs out, and an optimizer with nowhere to retreat returns its state instead of losing it |

**Measures**

| | | |
|---|---|---|
| 13 | `measures` | Gaussians, sampling, pushforward |
| 14 | `joint_law` | direct sums, and the joint law of model and data, linear and nonlinear |
| 15 | `correlated_product` | two fields tied through their latent noise: marginals unchanged, the cross-covariance `R_ij L_i L_j*`, and a field never measured informed through `R` alone |

**Sets**

| | | |
|---|---|---|
| 16 | `sets` | a convex set, its indicator and its support function as one object; proximal operators as geometry |
| 17 | `subspaces` | projectors, kernels, and linear constraints |
| 18 | `sets_in_three_dimensions` | convex sets drawn solid: an inscribed hull of supporting points, voxel rasters, and a 3-D slice of a 4-D ellipsoid checked against the exact one |

**Function spaces meet data**

| | | |
|---|---|---|
| 19 | `fields` | circles, tori, boxes, spheres |
| 20 | `observation` | where a function space meets an instrument |
| 21 | `flexure` | a coefficient that varies in space, and a picture of it |

**Inference: a point, a measure, a set**

| | | |
|---|---|---|
| 22 | `forward_problem` | the forward problem and its posterior, end to end on a small problem |
| 23 | `point_estimates` | least squares, the minimum-norm model and Backus-Gilbert, each once, as tools inside the methods that quantify uncertainty |
| 24 | `tomography` | a Bayesian inversion on the sphere, end to end |
| 25 | `low_rank_posterior` | the posterior as the prior corrected in the few directions the data inform: the eigenvalues, the rank, and a draw at the cost of a prior draw |
| 26 | `distributions` | looking at the answer: marginals and corner plots |
| 27 | `coupled_fields` | two unknowns, one shared physical chain |
| 28 | `preconditioning` | making a large solve finish, with a surrogate |
| 29 | `mixture` | a prior that cannot make up its mind, and a bimodal posterior |
| 30 | `feasible_set` | the third kind of answer: what the data cannot rule out, under a prior measure hardened into a ball and into an ellipsoid |
| 31 | `nonlinear_map` | when the forward map is not linear: the MAP estimator, and the Laplace approximation on it, full and low-rank |

**Spaces whose structure varies**

| | | |
|---|---|---|
| 32 | `varying_length_scale` | a correlation length that varies along an interval of the line, the padding that makes it possible, and a prior calibrated by a field |
| 33 | `disk` | a field on an annulus, drawn whole, and the variance surface data remove |
| 34 | `mantle` | a field in a mantle, cut three ways to be seen: a shell, a section by any plane, a profile; and what surface data do not reach |
| 35 | `density_profile` | a function of radius under `r² dr`: a planet's density from its mass and moment of inertia, and why they say nothing of the centre |
| 36 | `layered_planet` | fields that are smooth in each layer and jump between them: a direct sum of spaces that knows its layers are adjacent, data of three kinds at once, and continuity by conditioning |

## The ones with optional dependencies

Each skips without what it needs. The list is the one `tests/test_examples.py`
enforces, so it cannot drift from what actually runs.

| needs | examples |
|---|---|
| `pyshtools` | 20, 26, 28, 34 |
| `cartopy` | 21, 24, 27, 30, 34 |
| `planetmodel` | 32, 33, 34, 35, 36 |

```
poetry install --extras sphere
poetry install --extras planetmodel
```

Examples 21, 24, 27 and 30 draw coastlines, which makes cartopy fetch a
Natural Earth shapefile the first time; after that it is cached.

## Finite elements

A finite element space is the case the design was built for -- the mass
matrix *is* the Gram matrix, an assembled bilinear form *is* a Galerkin matrix,
and an assembled linear form *is* a derivative rather than a gradient. Through
MFEM, it is a mini-app rather than an example: `miniapps/mfem/`
wraps an MFEM function space as a Hilbert space, with and without
coordinates, and wraps an MFEM solve as an operator on the library's own
periodic box.

## If you read only one

Number 5. The distinction between a derivative and a gradient is the reason
most of the rest of the design looks the way it does, and it is the mistake the
library exists to make hard.
