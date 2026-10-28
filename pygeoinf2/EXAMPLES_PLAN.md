# The documentation programme for 2.0

Approved by David on 2026-10-24 as the place to start; a working document
for the docs phase, deleted when the phase completes. Three tiers.

## Tier 1: reference examples (`examples/`)

The existing 33 stay as they are, one idea each. The pre-release work adds
four in the same style:

- a 3-D set sliced and drawn, with the restored slice plotter;
- `EvaluationError`: a forward that fails, a line search that survives it;
- `gradient=` against `derivative=` in `from_callables` (a coda to 5);
- a `CheckReport` walk for backend authors (a coda to 16 and 29).

## Tier 2: tutorials (scripts, replacing `tutorials/*.ipynb`)

An ordered course, narrative, runnable top to bottom, written against the
v2 names from the start.

> **Status, 2026-10-27: thirteen drafted (13 is the ray-tomography showcase
> of D-156), notebooks generated from each by `make_notebooks.py` with
> Colab badges and checked current by `tests/test_tutorials.py`.**
> First drafts are in `pygeoinf2/tutorials/` as percent-cell scripts
> (`# %%` / `# %% [markdown]`, so Jupyter and VS Code open them as notebooks
> and the two kept notebooks can be converted from them), with a README and
> `tests/test_tutorials.py`, which runs each one as `test_examples.py` runs
> the examples; `tutorials` joins `examples` in `test_code_practice`'s
> exemptions. Tutorials 10 (last section) and 11 need `planetmodel`, 12
> needs MFEM. Tutorial 7 uses `BackusGilbertParker`'s automatic choice,
> the bracket and the inner hull (D-143).
> Writing them turned up one library bug, fixed with a regression test:
> `GaussianMeasure.marginal` refused the marginal of any posterior on a
> direct sum (`_block_of` lost the covariance's claimed traits). And one
> wart, not fixed: a posterior on a direct sum lives on the forward
> operator's domain, which `RowLinearOperator` builds without the prior's
> labels, so its marginals are reached by position, not by name.

1. **First inversion** — a function on a circle from point observations,
   twenty lines end to end; the taste before any theory.
2. **Spaces** — vectors as backend objects, inner products, the Gram
   matrix, coordinates as a capability; why there are no dual spaces
   (`f.adjoint(1.0)`).
3. **Operators** — adjoint is not transpose, derivative is not gradient,
   taught slowly on a non-diagonal metric; Galerkin against component
   matrices (D-25).
4. **Solvers** — matrix-free Krylov, strictness, preconditioners, what an
   `Estimate` means.
5. **Gaussian measures on function spaces** — invariant, Sobolev, heat and
   power measures, sampling, push-forwards; what `log_density` refuses on
   a function space and why (the marker, taught as a feature).
6. **The linear forward problem** — error measures, the posterior as a
   measure with its density and evidence, credible sets on properties.
7. **Properties, not models** — property operators, Backus-Gilbert, the
   feasible set as the third kind of answer.
8. **Regularisation without priors** — the Tikhonov family, discrepancy
   principle, minimum norm; when and what they cost. No L-curve, by
   David's instruction.
9. **Nonlinear problems** — `at()`, linearization, MAP, the Laplace
   posterior, the failure protocol.
10. **Coupled problems** — direct sums, block operators, joint priors;
    `Layered` the worked case.
11. **Spaces whose structure varies** — the spectral-element spaces from interval to layered
    planet: varying length scales, `r² dr`, balls and sections.
12. **Writing a backend** — the contract of CURRENT_STATE §6, the
    `check_*` ladder with `CheckReport`, MFEM the worked case, the
    lessons list.
13. **Nonlinear ray tomography on a function space** (added 2026-10-27,
    on D-156's operator) — the travel-time operator and the rays it
    bends, its derivative and the sensitivity kernels of the adjoint, the
    Hessian action with the ray's response inside, the inversion whitened
    by the prior's factor and solved by L-BFGS and by Newton-CG against
    straight-ray tomography, the Laplace spread. Runs on the plane in
    under a minute.
14. **One problem, three answers** (added 2026-10-28, D-163) — one linear
    problem answered as a point, a set and a measure: the linear estimate
    with its propagated standard deviation; its bias against a known
    truth, larger than that; the feasible set under a prior set and the
    posterior under a prior measure as the two legitimate completions; the
    same three pushed to a property. Then the nonlinear coda on the ray
    problem: MAP, the Laplace approximation, Occam, the linearised feasible
    set. The proof that the inference pattern holds.

Kept as notebooks, exactly two: a Colab quickstart (tutorial 1 in notebook
form, with the bootstrap cell) and a visual tour of the plotting layer,
where inline output earns the format.

> **Superseded 2026-10-27 (David):** every tutorial gets a notebook,
> generated from its script by `tutorials/make_notebooks.py` into
> `tutorials/notebooks/` (the pyslfp pattern), with a Colab install cell
> and a badge per tutorial in the tutorials README, so that the library can
> be tried without installing it. The scripts stay the source and a test
> checks the notebooks are current. Examples and mini-apps get no
> notebooks.

## Tier 3: mini-apps (`pygeoinf2/miniapps/`)

MFEM-spirit: each a self-contained directory with a README stating the
problem, realistic data (real geometry where we have it, synthetic fields
where we do not, said plainly in the first paragraph), shaped like a small
Python project without packaging of its own: small reusable modules (the
problem, the data, shared plotting) and one short executable script per
part with a fixed seed and a saved figure, a run-all script, never one
monolithic file (David, 2026-10-28); and a test in the suite that imports
the modules and runs each part at a quick scale (slow where it takes
minutes). The model is **one problem, many solutions**: each
app applies the range of methods that fit its problem, point, set and
measure, in parts that build up from the simplest.

Rewritten with David on 2026-10-28; the earlier list (great-circle
tomography, a planet's density, flexure under load, an interface from
gravity, two fields one chain, sampling, evidence against truncation,
making a large solve finish) is in git before this commit. Flexure under
load is dropped as contrived; the evidence and the large-solve ideas fold
into 4 and 1.

1. **Point data on the sphere** (`point_data/`, first; was `oceans/` until 2026-10-28). A field on the sphere with a
   Sobolev prior, observed only at ocean points (the Natural Earth land
   polygons through cartopy and shapely, already in the sphere extra),
   synthetic data from a truth drawn from the prior, at a scale where the
   naive solve is painful: degree a few hundred, observations in the tens
   of thousands. Parts: the linear algebra (the one solve behind
   every answer, in the data-space formalism: capped unpreconditioned CG
   against the Woodbury surrogate on a coarse sphere, iteration and
   application counts in a table: the need for preconditioning at real
   scale; the spectral deflation was dropped on 2026-10-28, David: the
   Woodbury is good even at this scale, the deflation is not); the Bayesian answer (posterior mean,
   the standard deviation returning to the prior's under the continents,
   samples, properties pushed forward); the point answers (`LeastSquares`,
   `MinimumNorm` with their resolution and propagated error, the bias
   against the truth); the set answer (`BackusGilbertParker` on a few
   regional averages, the brackets against the posterior intervals); a
   correlated-in-space prior through a linear transformation as an
   extension.
2. **A planet's density.** Radial density with mass and moment of inertia
   as the data, on a `Layered` radial space with the inner core, outer
   core and mantle as layers. The reference is built in the app, not
   taken from PREM's density: an Adams–Williamson model integrated
   downward from PREM's velocities (the seismic parameter
   `Φ = Vp² − 4Vs²/3`, `dρ/dr = −ρ g / Φ`, `g` from the density being
   built, planetmodel for PREM and the radial gravity integral), with the
   jumps at the CMB and ICB (and the 660 if wanted) put in by hand; not
   contemporary research, and that is not the point (David, 2026-10-28).
   The prior is then played with as the parts: effectively non-informative
   (a huge variance about the reference, so the posterior shows only what
   two numbers say, a two-dimensional constraint through a function space,
   with the feasible set under a loose norm bound making the same point
   from the other side); tight about Adams–Williamson, where the data
   adjust the jumps and little else; a mixture over two references, with
   and without a density jump at the 660, weighed by the evidence; each
   reported on properties (the mean density of the core, of the lower
   mantle) with the posterior, the point estimates and the feasible set
   side by side. Then boundary positions as unknowns, a direct sum of the
   field and a small Euclidean space, where the forward is nonlinear and
   MAP, Laplace and Occam earn their place. A variant for fun (David,
   2026-10-28): parameterise the density *through* the Adams–Williamson
   integration, the unknowns being the few density jumps (and the surface
   density) together with the velocity profile as a field with its own
   Gaussian uncertainty about PREM, the density derived; the forward to
   mass and moment of inertia is then nonlinear through the integration,
   its derivative the tangent linear of that small ODE (a radial kernel
   for the velocity, a scalar per jump), which gives a low-dimensional
   nonlinear problem with a function-valued part, ready for the sampler
   once it exists. Extension, optional and heavy: normal-mode
   eigenfrequencies through specnm, with its radial sensitivity kernels as
   the derivative.
3. **Ray tomography in a spherical Earth** (later). PREM reference rays by
   the radial integrals, in the mini-app's own module, deliberately not
   library code: the demonstration of how a user brings a forward problem
   and wraps it as a path-integral operator on the `ball` space through
   point evaluation at the ray nodes, the adjoint the library's own. A toy
   catalogue of a few hundred paths, a 3-D slowness perturbation, the
   posterior and its resolution, then the point and set answers. Extension:
   nonlinear tomography in the same geometry, rays bent in the perturbed
   model, at a toy number of paths.
4. **Realistic priors by evidence maximisation** (after 3; pyslfp-inspired,
   independent of it). Priors for ice-sheet thickness learned from
   synthetic climate-model fields: the Bayesian evidence
   (`GaussianPosterior.log_normalizer`) as a function of a few
   hyperparameters, maximised; non-stationarity through push-forwards of a
   stationary measure on the sphere by multiplication operators (amplitude
   following where ice is, a second transformation for the length scale);
   the sphere with the library's own operators first, an ice-sheet mesh
   through MFEM as the extension.
5. **Sea-level problems with pyslfp** (post-release). Joint inversions and
   composite data spaces from pyslfp, which depends on this library; built
   once pyslfp has moved to 2.0, as an optional import that skips when
   absent.
6. **Sampling the posterior** (post-release, the student's): pCN against
   the Laplace Gaussian where the Gaussian is visibly wrong.
7. **MFEM as a mini-app, not a backend** (David, 2026-10-28, to be
   decided). The MFEM backend now in the core (`backends/mfem`, D-96) would
   become a mini-app in the spirit of 9: a mesh, a bilinear form and a
   finite-element inverse problem, linking to the PyMFEM Python library
   first, and possibly a second variant showing how to link to native MFEM
   code through nanobind as 9 does for a bare vector type. Open: what the
   core keeps (the `backends` contract of CURRENT_STATE §6 and the no-array
   test of D-169 are what the library promises; the MFEM-specific code is
   a user of it, not part of it), which tests move with it, and whether the
   `mfem` extra stays in `pyproject.toml`.
8. **A parallel (MPI) forward code** (David, 2026-10-28, new). A mini-app
   whose forward operator runs under MPI, something the library has not
   looked at: the library sees one operator, the vectors it holds are the
   handles of distributed data (as 9 holds a C++ vector), and the inner
   product and the applications reduce across ranks. Candidates to settle:
   mpi4py with a toy distributed PDE solve, or PETSc through petsc4py
   (deferred from 2.0 by D-112, which this would revisit as a mini-app
   rather than a backend). To be planned after 1 to 3.
9. **A vector backend in C++** (DONE 2026-10-28, D-169): `cpp_backend/`,
   the vector type and sparse operators in C++ behind nanobind, a
   finite-element mass matrix as the Gram, the check ladder and the inverse
   problem end to end with a tally of every member the library called; its
   regression form is `tests/test_no_array_backend.py`. Kept as 9 so the
   decisions that name it stay right.

Seeds in `work/` (deleted at the release): `sphere_dli_example.py` (real
stations and earthquakes, cap averages, the feasible set) and `tomo.py` for
3; `evidence.py` for 4; `preconditioning_examples.py` for 1's solve part;
`correlated.py` for 1's extension.

## A nonlinear ray operator (David, 2026-10-27)

The linear ray-tomography operators exist; a nonlinear one would be a
small, real nonlinear operator for the examples and the tests. The forward
map is Hamiltonian ray tracing in the symmetric space's own domain
(sphere, plane, torus; it belongs in `symmetric_space`), with a root find
for the two-point problem; the derivative is Fermat's principle about the
reference ray, which is the existing path integral along it, so the
adjoint is the linear operator already built; applied adjoints and the
second-order term give a Hessian. Built 2026-10-27 as D-156
(`symmetric_space/rays.py`); an example and the tomography mini-app's
nonlinear half are to be written on it.

## Sequencing

Tier 2 and mini-apps 1, 2 and 9 are docs-phase work before the release;
3 and 4 may trail it, 5 and 6 follow it; 7 and 8 are unsequenced (added
2026-10-28). Coverage: every module of the package map appears except
the deeper convex machinery (deliberately quiet pending the redesign);
`datasets` is carried inside mini-app 1.
