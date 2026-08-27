# pygeoinf2 review: packaging, API surface, backends, tests, examples, docs

> **Note (2026-08-27):** the decisions recorded in `pygeoinf2/REVIEW.md` §11 (D-1 … D-13) override the Must/Should/Consider ranking below wherever they conflict — in particular D-1 (sphere vectors are `SHGrid`, `sampling=1` default), D-2 (points in `(lat, lon)` degrees), D-3 (per-geometry submodules with `Lebesgue`/`Sobolev` subclasses), D-4 (`from_matrix(..., form=)`), D-6 (parallel hooks around operators), D-12 (path *integral* operator), D-13 (convex solvers restored).

Scope: `/home/david/dev/pygeoinf/pygeoinf2/` (23.2k source lines, 12.5k test lines, 962 tests), against v1 `pygeoinf/` (36.3k lines) and DESIGN.md / V1_CATALOGUE.md. Nothing was modified.

## 1. Public API

**What `import pygeoinf2 as gi` gives.** 51 names (`pygeoinf2/__init__.py:71-132`) vs v1's 107: algebra, geometry, the five Fourier/box spaces, probability, `Traits`. Measured with `hasattr`:

| attribute | reachable | why |
|---|---|---|
| `gi.inference` | **no** | never imported by `__init__` |
| `gi.plotting` | **no** | never imported |
| `gi.backends`, `gi.testing`, `gi.compat` | no | by design (optional deps / scaffolding) |
| `gi.numerics` | yes, *by accident* | pulled in transitively by `symmetric_space`/`probability`; not a stated export |

So the two packages a user needs to *solve* anything are the two that are not there. DESIGN §25.4 (line 4306) records this as "worth a decision rather than a drift"; no decision has been taken.

**Against real downstream usage** (`/home/david/dev/pyslfp`, `import pygeoinf as inf`): of its 25 distinct `inf.X` names, these have no top-level v2 equivalent: `inf.LinearForwardProblem`, `inf.LinearBayesianInversion` (now `inference.LinearGaussianInversion`), `inf.CGSolver`/`CholeskySolver`/`LUSolver` (`numerics`), `inf.plot_corner_distributions` (now `plotting.plot_corner`), `inf.ProgressCallback` (gone; replaced by a `callback` kwarg, `numerics/solvers.py:356-364`). `from pygeoinf2.symmetric_space.sphere import Lebesgue, Sobolev` works. `HilbertSpaceDirectSum`→`DirectSum`, `RowLinearOperator` etc. are exported.

**Export gaps inside subpackages** (public in the module, absent from the package `__init__`):
- `numerics/solvers.py:570 FlexibleCGSolver`, `:639 GMRESSolver` — not in `solvers.__all__` nor `numerics/__init__.py`. V1_CATALOGUE line 137 still says "GMRES itself is missing"; it exists but is unreachable by the documented path.
- `numerics/randomised.py:502 deflated_diagonal` — not in `__all__`.
- `numerics/convex.py:787 ProximalBundleMethod`, `root_find.py:105 monotone_root`, `quadratic_forms.py:64/112 weighted_chi2_*`, `solvers.py:134 resolve_solver` — in module `__all__`, not re-exported by `numerics/__init__.py`.
- `backends/mfem.py:49 __all__` lists 3 of 8 public names; `essential_dofs_of`, `solver_from_bilinear_form`, `operator_from_linear_forms`, `white_noise_load`, `matern_measure` are missing, though all are used by examples 16/27.
- `inference/normal.py:85 FactoredNormalOperator`, `algebra/spaces.py:432 HilbertModule`, `:477 require_module` — not re-exported.

**Naming consistency.** Class names are consistent (`*Solver`, `*Preconditioner`, `*Inversion`, `*Estimator`). Parameter names are not: the iteration cap is `maxiter` in `numerics/solvers.py:361`, `max_iterations` in `functional_calculus.py`, `optimisation.py`, `line_search.py`, `backends/mfem.py:478`, and `iterations` in `root_find.py`, `inference/point.py`, `inference/backus.py`, `numerics/convex.py`; tolerance is `rtol` almost everywhere but `tolerance` in `quadratic_forms.py` and `convex.py`. `rng` is uniform (1018 uses). Estimator classes are bare nouns (`MinimumNorm`, `LeastSquares`) while the Gaussian ones are `Linear*Inversion` — acceptable, but worth one line in the docs.

**Layout.** The eight-package layering is sound and matches DESIGN §2.2; the import graph is acyclic and `numerics`/`algebra`/`geometry` do not import `inference` (confirmed by DESIGN §25.4 and by the import-time trace). The problem is only the top-level curation, not the structure.

**`compat.py` (345 lines).** Imported only by `tests/test_compat.py:18`. DESIGN §11.3 (line 1381) says delete it "when the concrete spaces are rewritten" — they have been (`symmetric_space/` is native). It is now the *only* v1-vs-v2 parity check (posterior mean to 1e-10, `test_compat.py:225-250`), which is worth keeping until the rename, then deleting with its test.

**V1_CATALOGUE.md is stale.** At least 13 rows carry status "Ported" while the text says "Not ported" (lines 173, 276, 366, 410, 414, 441, 444, 460, 463, 477, 481, 495, 513-517). Grepping the code: most now exist (`deflated_diagonal`, `two_point_covariance`, `degree_multiplicity`, `gaussian_curvature`, `order_inclusion`, `inverse_flexural_operator`, `from_tangent_basis`, `pseudo_inverse`, `with_translation`), but three do not: `CorrelatedInvariantGaussianMeasure` (line 276, marked "needed feature"), `norm_scaled_*_gaussian_measure` (477, "worth it"), `invariant_covariance_function` (481, "need a reason to drop"). `configure_threading` and `ProgressCallback` are dropped by agreed decision.

## 2. Backends

**MFEM backend (`backends/mfem.py`, 850 lines).** Design is right: mass matrix as Gram, `apply_gram` sparse (`:325`), `solve_gram` via a cached `scipy.sparse.linalg.factorized` (`:314-322`), essential dofs as a subspace (`:143-235`), `solver_from_bilinear_form` (`:471-554`) wrapping MFEM's own CG with the mass multiply on the way in, `white_noise_load` via MFEM's integrator (`:638-668`), and `matern_measure` supplying covariance, factor and precision (`:671-806`). `check_space/coordinates/operator/traits` pass on it (`tests/test_backends.py`).

**It is not usable for a real FE problem, and DESIGN misstates this.** `MfemSpace.restrict` (`mfem.py:240-255`) does `_to_scipy(matrix).toarray()` unconditionally, so:
- `operator_from_bilinear_form` (`:396-424`) hands a **dense** `dim x dim` array to `from_derivative_matrix`;
- `solver_from_bilinear_form` (`:520`) does `_to_mfem(sp.csr_matrix(space.restrict(form.SpMat())))` — dense, then re-sparsified, then rebuilt as an MFEM matrix by a **Python loop over every nonzero** (`_to_mfem`, `:427-449`).

DESIGN §33.4 (line 5023) claims "Sparsity now survives `restrict` and `operator_from_bilinear_form` as well"; the git history shows the dense `restrict` was introduced in 516a840 and never changed. At 1e5 dofs that is an 80 GB allocation. `from_derivative_matrix` already accepts sparse input (`algebra/operators.py:45-55`, `_as_matrix` leaves sparse alone), so the fix is local: slice the CSR (`full[free][:, free]`, as `_scipy_mass` at `:307-312` already does) and convert to MFEM via `mfem.SparseMatrix` from CSR arrays rather than `Add` per entry.

**Memory-safety workarounds** are well documented in place: `_to_scipy` refuses unfinalised matrices (`:74-97`), `to_components` and `restrict_vector` copy (`:257-266`, `:287-300`), `_default_mfem_solver` keeps the smoother alive via `solver._pygeoinf_keepalive` (`:453-470`), `_white_noise_integrator` patches PyMFEM's broken constructor (`:613-635`), and the integrator is bound to a name before `Assemble` (`:655-663`). DESIGN §33.5/§34.4 record them. The keepalive relies on setting an attribute on a SWIG proxy; a comment saying this is intentional would prevent someone "cleaning it up".

**PETSc.** Genuinely gone: no code, no extra, no example. Remaining mentions are prose — `algebra/spaces.py:6,53`, `examples/01_spaces.py:5`, V1_CATALOGUE:136, DESIGN §14 line 1994 ("the two foreign backends") and §11.9. Harmless, but §14 is now wrong.

**No generic backend recipe.** `backends/__init__.py` is 8 lines; DESIGN §15 describes MFEM, not the contract. What a new backend must implement is only discoverable from `CoordinateSpace`'s abstract methods (`dim`, `zero`, `copy`, `inner_product`, `axpy`, `scale_inplace`, `random`, `to_components`, `from_components`, `apply_gram`, `solve_gram`, `white_noise_components`, `_key`) plus the `testing.check_*` sequence. The v1 tutorial 3 §3-4 ("subclass HilbertSpace and check it") has no v2 counterpart either.

**Packaging of the extra.** `pyproject.toml` `mfem = ["mfem>=4.8.0"]`; CI installs `--all-extras` on ubuntu/windows/macos and runs `pytest` from the root, so `pygeoinf2/tests/test_backends.py` runs in CI wherever the PyMFEM wheel installs — worth confirming that this passes on Windows/macOS rather than assuming.

## 3. Tests

**Structure.** `pygeoinf2/tests/conftest.py`: one `rng` fixture (seed `20260826`, function-scoped, `:17-22`), `make_weighted_space()` (diagonal metric, `:25-44`) and `make_dense_metric_space()` (dense `G = R R^T`, dim 3, `:47-87`). `doubles.py`: `Opaque`, `OpaqueSpace`, `StrictSpace`, `CallCounter`. `CallCounter` is recorded but **never asserted on**; `doubles.py:15-16` claims it tests `at()` evaluating once, but that test uses a plain dict (`test_nonlinear.py:247`).

**The non-diagonal-Gram house rule is applied in 6 of 30 files** — `test_spaces.py:27,98,113,146`, `test_operators.py:41,272,278,383`, `test_functional_calculus.py:278,354`, `test_probability.py:68`, `test_plotting.py:152`, `test_phase_four.py:933`. Every `SymmetricSpace` inherits `DiagonalMetricSpace` (`symmetric_space/base.py:47-49`), so sphere/Fourier tests do not count. Areas that **never** see a non-diagonal metric outside the optional MFEM file:
- iterative solvers and all preconditioners (`test_solvers.py:86,147-181`; `test_phase_four.py:709-860,1082-1335`; `test_normal.py:58-72` keeps the model space Euclidean);
- every inference class — `LinearGaussianInversion`, `MinimumNorm`, `LeastSquares`, Backus, mixture (`test_inference.py:36-46`, `test_backus.py:27-41`, `test_point.py:200`, `test_mixture.py:147`);
- KL divergence (`test_spectral.py:160-215`), randomised methods (`test_randomised.py`, mostly `EuclideanSpace`), direct sums (`test_direct_sum.py:40`), geometry projections (`test_geometry.py:33`), gradient-vs-derivative checks (`test_nonlinear.py:55-71`), optimisation and convex.
This is exactly the class of code DESIGN §30.2 ("the metric, a fifth time") admits was got wrong repeatedly on diagonal-only tests.

**Coverage by module.** Thin or single-file: `plotting/sphere.py` (`plot_points`, `plot_paths`: zero tests), `algebra/nodes.py` (only via private `_Sum`/`_Composition` in `test_operators.py:158-215`), `inference/preconditioners.py` (test_normal only), `inference/tikhonov.py` (`TikhonovFamily` never mentioned), `numerics/root_find.py`, `numerics/quadratic_forms.py` (`test_phase_four.py:861-919` only), `probability/mixture.py`. Never-tested public names include `LinearOperator.from_vectors`, `from_tensor_product`, `DirectSolver`, `resolve_solver`, `FactoredNormalOperator`, `NormalOperator.posterior_covariance`, `GaussianMeasure.from_product`, `AffineSubspace.with_translation`, `Sphere.domain_mask`, `MfemSpace.restrict_vector`.

**Behaviour vs implementation.** Mostly behavioural and well motivated (`test_spaces.py:120-133`, `test_spectral.py:177-192`, `test_phase_four.py:983-1000`, `test_backends.py:106-121`). Brittle spots: private attributes in `test_sphere_transform.py:71-195`, `test_observation.py:112,200,225`, `test_fourier_spaces.py:316,351`, `test_backus.py:80`; monkeypatching `JacobiPreconditioner._invert` in `test_point.py:160-184`; identity assertions (`is A`) in `test_operators.py:165-207`.

**Performance pins.** Relative iteration counts (`test_solvers.py:257,278,288`; `test_optimisation.py:130`; `test_phase_four.py:709-860`) and "no coordinates touched" guards via `StrictSpace` (`test_solvers.py:147-181,291-297`; `test_functional_calculus.py:313-332`; `test_randomised.py:250-252`; `test_coordinate_free.py:130-176`). No test pins that an operation stays sparse or bounds memory — which is how the MFEM densification survived.

**Runtime and slow markers.** `pyproject.toml:86-99` sets `addopts = "-m 'not slow'"`; 15 explicit `slow` marks plus examples 22 and 24 (`test_examples.py:79-82`). DESIGN §32 reports 2m57 fast / 9m09 full. Unmarked but expensive in the default run: Monte Carlo at 20k-40k draws in `test_spaces.py:146,160`, `test_coordinate_free.py:50` (40k pure-Python `Opaque` draws), `test_probability.py:18` (`SAMPLES = 30000`), `test_compat.py:90,268,425,440-460`, `test_direct_sum.py:342,388`, `test_backends.py:415,440` (20k MFEM form assemblies); function-scoped `Sobolev(24)` rebuilt for 43 tests in `test_observation.py:27-34`.

**Flakiness.** Seeding is disciplined (fixture plus inline `default_rng(k)`). Exceptions: global `np.random.seed` at `test_compat.py:97,438` and unseeded v1 `random_points` at `:271,279` (order-dependent; assertions hold for any points). Tight statistical margins: `test_phase_four.py:954-963` (`abs=0.03`, ~5 SE), `test_mixture.py:78`, `test_backends.py:492,532,551`.

**Silent shrinkage.** Without `pyshtools`, five files (~150 tests) skip at module level (`test_bounded_and_sphere.py:18`, `test_observation.py:19`, `test_spectral.py:25`, `test_flexure.py:21`, `test_sphere_transform.py:19`). `test_examples.py:12-14` imports matplotlib unconditionally (collection error if absent).

**Examples are tested** (`test_examples.py:110`, `runpy.run_path` in-process, Agg backend, all 27 collected, README listing enforced at `:114-122`) — but the assertion is only "printed something" (`:111`).

**`test_code_practice.py`** enforces, per file via `ast`: docstring **presence** on public top-level defs and direct class methods (`:62-68`), keyword-only optionals (`:70-90`), annotations on every param and return (`:92-106`), and a no-shadowing check on space classes (`:145-170`). It skips `tests/` and `examples/` only (`:22`), so `compat.py`, `backends/`, `testing.py` are covered. It does **not** check docstring content, module docstrings, nested functions, or `*args/**kwargs` annotations.

**Tests live inside the package** (`pygeoinf2/tests/`), so on promotion they ship in the wheel unless excluded.

## 4. Examples and tutorials

27 flat scripts, all seeded, none with `main()`. **01-18 are genuine tutorials** (one idea, prose on *why*, a `check_*` or printed boolean making the point; `05_derivative_and_gradient.py:40-78` is the standout). **19-27 are ported `work/` scripts** with good docstrings but unexplained constants (`22_coupled_fields.py:39-44,109-114`; `23_feasible_set.py:42-57`; `27_mfem_inverse.py:59-64`) and tables of numbers.

**Workflow gaps:**
- **No minimum-norm / least-squares / Tikhonov example.** `LeastSquares`, `MinimumNorm`, `DiscrepancyPrinciple`, `Constrained*`, `TikhonovFamily` are imported by no script, though `inference/__init__.py:12` leads with `MinimumNorm(problem)`. v1 tutorial 7 has no counterpart.
- **No nonlinear inverse problem.** `ForwardProblem` (`inference/problem.py:32`) is never used by an example; LBFGS/NewtonCG only minimise a quadratic (12).
- **No pointwise posterior std of a field**, no 1-D/circle plot (15 makes no figure; `plotting/fourier.py` unused), no inversion on torus/box/interval, no "wrong prior" comparison (v1 tutorial 8), no user-written `HilbertSpace` subclass (v1 tutorial 3), no `from_formal_adjoint` demo (v1 tutorial 4 §4).
- Unexercised: `LocalisedPreconditioner`, `InvariantDistancePreconditioner`, `DualFeasibleProperty`, `BackusInference`, `LUSolver`.

**Defects in the scripts:** `15_worked_example.py:51` computes `X.norm(truth)/X.norm(truth)` (always 1.0) as the "prior error"; `16_mfem_backend.py:22` teaches a private `_to_scipy`; conclusions are hard-coded in `print` (`26_mixture.py:141-152`, `27_mfem_inverse.py:287-297`, `16_mfem_backend.py:107-108`) so a drifting number makes the tutorial lie while the test passes; 21-27 import `GaussianMeasure`/`LinearOperator` from deep paths despite top-level exports; figures are drawn but never shown or saved (six scripts).

**README** (`examples/README.md`): lists all 27; optional-dependency section (line 47) is wrong (27 needs mfem; 24, 25 need pyshtools; 20-23 need pyshtools too); lines 55-59 are an orphaned paragraph about example 16; no mention that 22/24 are slow-marked or that 20-23 skip without cached coastlines.

**v1 notebooks.** `tutorials/tutorial1..10.ipynb` still target v1; nothing in DESIGN plans their v2 replacement beyond §20.4's brief note. The README and `docs/source/index.rst` both link them.

## 5. Documentation infrastructure

- **Sphinx documents v1 only.** `.readthedocs.yaml` runs `sphinx-apidoc -f -o docs/source/ pygeoinf/`; `docs/source/conf.py` uses `autodoc` + `napoleon` (numpy-style hint, but v2 is Google-style where structured); no rst mentions `pygeoinf2`. `index.rst:35-49` embeds a Markdown table in RST (renders as literal text). `nbsphinx` is a dev dependency but no notebook is included.
- **Module docstrings**: present in all 55 v2 modules (checked by AST).
- **Docstring style**: 115 Google-style (`Args:`), 0 numpy-style, 846 freeform, 28 absent (all `__init__`s whose class docstring carries the description, e.g. `direct_sum.py:74`, `operators.py:355,941,1143,1229`, `spaces.py:501`).
- **Content, over all 857 public functions/methods** (script in scratchpad): 54% are one-liners; of 547 that take parameters, **21% have an Args section**; 2% have Returns; **0 have Raises**. By file: 0% in `algebra/spaces.py` (48 functions), `algebra/direct_sum.py`, `probability/base.py`, `traits.py`; 6% `algebra/operators.py`; 9% `inference/gaussian.py`; versus 100% `plotting/distributions.py`, 75% `inference/preconditioners.py`, 71% `numerics/preconditioners.py`.
- A random sample of 40 parameterised functions: 16 had an Args section, 2 a Returns section, 0 Raises. Good: `numerics/solvers.py:356` (`IterativeSolver.__init__`, every kwarg explained), `numerics/root_find.py:105` (Args + Returns, 26 lines), `backends/mfem.py:100`. Bad: `algebra/spaces.py:161 gram_schmidt` says "Raises if the vectors are dependent" without naming `rtol` or the exception; `testing.py:599 check_second_derivative` is one line for six parameters; `symmetric_space/sphere.py:496 accumulate(weights, points, *, eps)` never mentions any parameter; `geometry/convex.py:589 HalfSpace.contains(x, *, rtol)`: "True when the inequality holds to tolerance".
- The prose docstrings are frequently *better* than a template would be (they explain why), but they omit the mechanical facts an autodoc reader needs — arguments, return type, exceptions — and the exception behaviour is undocumented everywhere.

## 6. Code practice

- **Rule 1 (docstrings)**: enforced for presence; followed (28 `__init__` exemptions). Content is not enforced and is thin (see §5).
- **Rule 2 (type hints)**: enforced and followed at the level checked; 43 unannotated params are all in nested closures (`solve_fn` in `numerics/solvers.py:280,478,966`, `preconditioners.py:45-507`; `direct_sum.py:216-219`), which the checker does not visit.
- **Rule 3 (keyword-only optionals)**: enforced; two real violations slip past the dunder exemption — `algebra/operators.py:941 Functional.__init__(domain, codomain=...)` and `probability/mixture.py:58 GaussianMixture.__init__(..., weights=...)`; the other nine are closure default-bindings (`_retained`, `_sample`).
- **Import time**: `import pygeoinf2` 0.32 s / 694 modules vs `import pygeoinf` 0.78 s / 1277 modules (3 runs each). v2 does not import matplotlib, pyshtools, finufft or joblib at import; `scipy.linalg` (0.16 s) is the bulk.
- **Global mutable state**: none found beyond `traits.py:53 _IMPLICATIONS` (a constant table). No `os.environ`, no thread-count control (dropped by agreement), no module-level caches.
- **Logging vs print**: no `print` and no `logging` in library code. Progress is by `callback` only (`solvers.py:455-459`).
- **Warnings**: two sites — `numerics/solvers.py:470-472` (`RuntimeWarning` on non-convergence when `strict=False`, with a function-level `import warnings`) and `compat.py:218`. Consistent; the function-level import should be hoisted per DESIGN §25.4.

## Packaging facts (cross-cutting)

- `pyproject.toml` `[tool.poetry] packages = [{ include = "pygeoinf" }]` — **pygeoinf2 is not in the wheel**, but `include` lists `pygeoinf2/data/*.csv`, so a stray data file ships without its package. Version is 1.8.9; `dist/` holds stale 1.8.4 artefacts.
- No `py.typed` in either package despite full annotations.
- `v2/` is a 48-file skeleton of **entirely empty** files (plus a typo `v2/geometry/shapes/__init__.oy`), committed in b97486c; DESIGN §20.3 mentions it only as a naming precedent.
- `work/` (12 scripts) imports v1 only; its `__pycache__` is committed.
- `ruff` per-file ignores reference `pygeoinf/symmetric_space_new/` and `pygeoinf/rough_work/`, which do not exist.
- CI runs `pytest --cov=pygeoinf` from the root, so v2 tests run (under `-m 'not slow'`) but coverage is measured on v1.

## Recommendations

**Must (before promotion/release)**
1. Make `MfemSpace.restrict` return a sparse matrix: replace `_to_scipy(matrix).toarray()` at `backends/mfem.py:251` with CSR row/column slicing (mirror `_scipy_mass`, `:307-312`), and rewrite `_to_mfem` (`:427-449`) to build the `mfem.SparseMatrix` from CSR arrays instead of a per-entry `Add` loop. Then correct DESIGN §33.4 line 5023. Add a test that `operator_from_bilinear_form(...)` never allocates a dense `dim x dim` array (e.g. build on a 100x100 mesh and assert `scipy.sparse.issparse` on the matrix passed to `from_derivative_matrix`, or check peak memory).
2. Decide and implement the top-level namespace: in `pygeoinf2/__init__.py` add `from . import inference, numerics, plotting` (so `gi.inference.X` works) **and** re-export the workflow names v1 users reach for: `LinearForwardProblem`, `ForwardProblem`, `LinearGaussianInversion`, `MinimumNorm`, `LeastSquares`, `CGSolver`, `CholeskySolver`, `LUSolver`, `MinResSolver`, `plot`, `plot_corner`. Update DESIGN §25.4 with the decision.
3. Fix the export lists: add `GMRESSolver`, `FlexibleCGSolver` to `numerics/solvers.py:43 __all__` and `numerics/__init__.py`; add `deflated_diagonal` to `randomised.py:35 __all__`; add the five missing names to `backends/mfem.py:49 __all__`; add `ProximalBundleMethod`, `monotone_root`, `weighted_chi2_*`, `FactoredNormalOperator` to their package `__init__`s.
4. Apply the non-diagonal-Gram rule to the inference and solver layers: parametrise the model-space fixtures in `tests/test_inference.py:36`, `tests/test_point.py:200`, `tests/test_solvers.py:86`, `tests/test_normal.py:58-72`, `tests/test_spectral.py:160` (KL), `tests/test_randomised.py`, `tests/test_direct_sum.py:40` and `tests/test_geometry.py:33` over `make_dense_metric_space()` in addition to the weighted space.
5. Packaging: either add `{ include = "pygeoinf2" }` to `[tool.poetry] packages` for the dev period or remove the `pygeoinf2/data/*.csv` include line; add `pygeoinf2/tests` to the exclude list for the wheel; add `py.typed`; delete `dist/`.
6. Delete the empty `v2/` tree and `work/__pycache__/`; drop the nonexistent `symmetric_space_new`/`rough_work` entries from `[tool.ruff.lint.per-file-ignores]`.
7. Fix `examples/15_worked_example.py:51` (`prior_error` should be `X.norm(X.subtract(prior.expectation, truth)) / X.norm(truth)` or simply 1.0 with an explanation) and remove the private `_to_scipy` import at `examples/16_mfem_backend.py:22`.
8. Fix `examples/README.md` line 47 (dependency list — copy from `tests/test_examples.py:32-42`) and delete or relocate the orphaned paragraph at lines 55-59.

**Should**
9. Add a "least squares / minimum norm / Tikhonov" example (v1 tutorial 7) and a nonlinear inverse-problem example using `ForwardProblem` + `LBFGS`/`NewtonCG`; add a pointwise-std band and a figure to example 15.
10. Turn the hard-coded prose conclusions in examples 16, 26, 27 into `assert`s (or computed booleans that are printed), so `test_examples.py` fails when the numbers drift.
11. Unify the iteration-cap parameter name (`maxiter` vs `max_iterations` vs `iterations`) and `tolerance` vs `rtol` across `numerics/` and `inference/`; pick one and rename with a deprecation shim.
12. Extend `test_code_practice.py` to check docstring content for public functions with parameters: require an `Args:` section naming every non-`self` parameter (allow an explicit `# noqa: docstring` escape), and require `Raises:` wherever the body contains `raise`. Then raise `algebra/spaces.py`, `algebra/operators.py`, `probability/base.py`, `direct_sum.py` and `inference/gaussian.py` from 0-9% Args coverage. Also make the checker visit nested functions for rule 2.
13. Fix the two rule-3 violations: `algebra/operators.py:941` (`codomain` keyword-only) and `probability/mixture.py:58` (`weights` keyword-only).
14. Mark the heavy unmarked Monte Carlo tests `slow` or reduce their sample counts: `tests/test_spaces.py:146,160`, `tests/test_coordinate_free.py:50`, `tests/test_probability.py:18`, `tests/test_compat.py:90,268,425,440-460`, `tests/test_direct_sum.py:342,388`, `tests/test_backends.py:415,440`; make the `Sobolev(24)` fixture in `tests/test_observation.py:27-34` module-scoped.
15. Write a backend recipe (a page in `docs/` or a section in `backends/__init__.py`): the 13 `CoordinateSpace` methods a backend implements, the memory-ownership rules learned from MFEM, and the `check_space` → `check_coordinates` → `check_white_noise` → `check_operator` → `check_traits` sequence. Add a small "custom HilbertSpace from scratch" example (v1 tutorial 3 §3-4).
16. Sphinx for v2: point `sphinx-apidoc` at `pygeoinf2/` (or both), set `napoleon_google_docstring = True` / `napoleon_numpy_docstring = False`, convert the Markdown table in `docs/source/index.rst:35-49` to RST, and decide whether the ten v1 notebooks are ported (nbsphinx is already a dev dependency) or replaced by the example scripts rendered via `sphinx-gallery`.
17. Refresh V1_CATALOGUE.md: reconcile the "Ported / Not ported" rows (lines 173-517) against the code, and either port or explicitly drop `CorrelatedInvariantGaussianMeasure`, `norm_scaled_*_gaussian_measure`, `invariant_covariance_function`. Update DESIGN §14 line 1994 ("two foreign backends") and §11.9 for PETSc's removal.
18. Add tests for `plotting/sphere.py` (`plot_points`, `plot_paths`, cartopy-gated), `TikhonovFamily`, `DirectSolver`, `resolve_solver`, `FactoredNormalOperator`, `GaussianMeasure.from_product`, `MfemSpace.restrict_vector`; either assert on `CallCounter` somewhere or delete it and fix `doubles.py:15-16`.
19. Delete `compat.py` and `tests/test_compat.py` at the rename, as DESIGN §11.3 specifies; until then keep them as the parity check.

**Consider**
20. Change CI to `pytest --cov=pygeoinf --cov=pygeoinf2` and confirm the MFEM extra actually installs and passes on the Windows/macOS matrix entries.
21. Hoist the function-level `import warnings` at `numerics/solvers.py:470`; add a comment on `solver._pygeoinf_keepalive` (`backends/mfem.py:469`) saying the attribute is load-bearing.
22. Give the six plotting examples a `SAVE_DIR`/`SHOW` environment switch so a reader running them from a shell sees the figures; the test already forces `Agg`.
23. Make the examples in 21-27 import from `pygeoinf2` top level rather than deep module paths once item 2 is done, so the tutorials model the intended API.
24. Add a small performance-regression harness (a `slow`-marked test that records iteration counts and peak memory on a fixed sphere/MFEM problem against stored baselines) — currently only relative iteration comparisons exist and nothing guards sparsity or memory.
