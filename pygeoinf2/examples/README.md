# Examples

Short scripts, one new idea each, meant to be read in order. Every one runs on
its own:

```
poetry run python pygeoinf2/examples/05_derivative_and_gradient.py
```

They are executed by the test suite, so an example that stops working is a
failing test rather than a surprise later.

## The sequence

| | | |
|---|---|---|
| 1 | `spaces` | vectors are raw backend objects; the space does the arithmetic |
| 2 | `coordinates` | a basis is a capability, not a requirement; the Gram matrix |
| 3 | `operators` | adjoints, and where the metric enters |
| 4 | `traits` | structure that survives the algebra |
| 5 | `derivative_and_gradient` | **the one to read twice** |
| 6 | `nonlinear` | `at()`, and why value and derivative come together |
| 7 | `solvers` | coordinate-free Krylov; declared preconditions |
| 8 | `functional_calculus` | `f(A)` without a matrix |
| 9 | `randomised` | low-rank factorisation and trace estimation |
| 10 | `measures` | Gaussians, sampling, pushforward |
| 11 | `direct_sums` | the joint model, linear and nonlinear |
| 12 | `optimisation` | why working in the metric changes the iteration count |
| 13 | `convex` | proximal operators as geometry |
| 14 | `fields` | circles, tori, boxes, spheres |
| 15 | `worked_example` | all of it, on a small inverse problem |
| 16 | `mfem_backend` | a finite element space, where the mass matrix is the metric |
| 17 | `sets` | a convex set, its indicator and its support function as one object |
| 18 | `subspaces` | projectors, kernels, and linear constraints |
| 19 | `observation` | where a function space meets an instrument |
| 20 | `flexure` | a coefficient that varies in space, and a picture of it |
| 21 | `tomography` | a Bayesian inversion, end to end |
| 22 | `coupled_fields` | two unknowns, one shared physical chain |
| 23 | `feasible_set` | the third kind of answer: what the data cannot rule out |
| 24 | `preconditioning` | making a large solve finish, with a surrogate |
| 25 | `distributions` | looking at the answer: marginals and corner plots |
| 26 | `mixture` | a prior that cannot make up its mind, and a bimodal posterior |
| 27 | `mfem_inverse` | an inverse problem on a finite element space, with boundary conditions |
| 28 | `nonlinear_map` | when the forward map is not linear: the mode, and a Gaussian on it |

## The ones with optional dependencies

Number 16 needs `mfem`, 19 needs `pyshtools`, and 20 to 23 need `cartopy`.
Each skips without them.

```
poetry install --extras mfem
poetry install --extras sphere
```

It answers the question the rest of the suite cannot: whether the
coordinate-free design really holds up when the vectors belong to somebody
else. A finite element space is the case the design was built for -- the mass
matrix *is* the Gram matrix, an assembled bilinear form *is* a Galerkin matrix,
and an assembled linear form *is* a derivative rather than a gradient.

## If you read only one

Number 5. The distinction between a derivative and a gradient is the reason
most of the rest of the design looks the way it does, and it is the mistake the
library exists to make hard.
