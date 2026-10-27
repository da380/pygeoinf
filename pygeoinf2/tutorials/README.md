# Tutorials

An ordered course through the library, meant to be read from the first to
the last. Where the [examples](../examples/README.md) show one idea each, in
as few lines as possible, the tutorials explain: each builds on the ones
before it and says why as well as how.

Every tutorial is a Python script divided into cells. Run one top to bottom,

```
poetry run python pygeoinf2/tutorials/01_first_inversion.py
```

or open it in Jupyter or VS Code, which read the `# %%` markers as notebook
cells and the `# %% [markdown]` ones as text. They are executed by the test
suite, so a tutorial that stops working is a failing test.

Each script also has a notebook generated from it under `notebooks/`, which
runs in Jupyter or, with nothing installed, in Google Colab through the
badges below. The scripts are the source: edit one and run
`make_notebooks.py`, which rewrites its notebook, and a test checks that
the two agree.

## The course

| | | | Colab |
|---|---|---|---|
| 1 | `01_first_inversion.py` | a function on a circle from fifteen noisy values, end to end, before any theory | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/pygeoinf2/tutorials/notebooks/01_first_inversion.ipynb) |
| 2 | `02_spaces.py` | vectors belong to the backend and the space does the arithmetic; the inner product is not the dot product; coordinates are a capability; there are no dual spaces | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/pygeoinf2/tutorials/notebooks/02_spaces.ipynb) |
| 3 | `03_operators.py` | the adjoint is not the transpose, a derivative is not a gradient, and a matrix says whether it is in components or Galerkin form | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/pygeoinf2/tutorials/notebooks/03_operators.ipynb) |
| 4 | `04_solvers.py` | matrix-free Krylov solvers, strictness, preconditioners, and what a stochastic `Estimate` means | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/pygeoinf2/tutorials/notebooks/04_solvers.ipynb) |
| 5 | `05_gaussian_measures.py` | Gaussian measures on function spaces: invariant, Sobolev, heat and power priors, sampling, push-forwards, and what a density refuses to be | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/pygeoinf2/tutorials/notebooks/05_gaussian_measures.ipynb) |
| 6 | `06_linear_forward_problem.py` | the forward problem, the posterior as a measure with its density and evidence, and credible sets on properties | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/pygeoinf2/tutorials/notebooks/06_linear_forward_problem.ipynb) |
| 7 | `07_properties.py` | asking about properties rather than models: Backus-Gilbert, and the feasible set as the third kind of answer | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/pygeoinf2/tutorials/notebooks/07_properties.ipynb) |
| 8 | `08_regularization.py` | answers without a prior: least squares, minimum norm, the Tikhonov family and the discrepancy principle, and what each costs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/pygeoinf2/tutorials/notebooks/08_regularization.ipynb) |
| 9 | `09_nonlinear.py` | nonlinear forward problems: linearisation, the MAP model, the Laplace posterior, and what happens when the forward cannot be evaluated | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/pygeoinf2/tutorials/notebooks/09_nonlinear.ipynb) |
| 10 | `10_coupled_problems.py` | several unknowns at once: direct sums, block operators, joint priors, and fields that jump between layers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/pygeoinf2/tutorials/notebooks/10_coupled_problems.ipynb) |
| 11 | `11_varying_structure.py` | spaces whose correlation length varies: an interval, a function of radius, a ball, and a layered planet | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/pygeoinf2/tutorials/notebooks/11_varying_structure.ipynb) |
| 12 | `12_writing_a_backend.py` | the contract a backend meets, the ladder of checks that verifies it, and a finite element space as the worked case | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/pygeoinf2/tutorials/notebooks/12_writing_a_backend.ipynb) |

## What they need

Tutorials 1 to 9 need only the core library. Tutorials 10 and 11 need
`planetmodel` (`poetry install --extras planetmodel`), tutorial 10 for its
last section only, and tutorial 12 needs MFEM
(`poetry install --extras mfem`); each says so at the top.
