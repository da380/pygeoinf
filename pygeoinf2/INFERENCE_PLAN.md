# The inference pattern: implementation plan

A working document for the implementing sessions, deleted when the work is
done. The design is fixed by `DECISIONS.md` D-163 (settled with David on
2026-10-28), with D-161 (the `sem1d` split) and D-162 (the degree names)
as the mechanical jobs that precede it; read those first, then D-140 to
D-143 and D-148 for the feasible set, then this file. The implementer
records any further sub-decision as D-164 onward (never renumber), runs
the fast suite between stages, and never commits: David commits.

## The binding rules

- **Data in, a point, a set or a measure out.** Every inference class is
  an `Estimator`; the kind of answer follows the description of
  uncertainty (D-163).
- **A point estimator never turns its error budget into a measure.**
  `resolution`, `propagated_covariance` and `bias` are the budget; the
  posterior and the feasible set are the two legitimate completions.
- **Matrix-free by default**; a dense factorisation is a solver the user
  passes. Rank algorithms by scaling, not toy timings.
- **No decision citations in code** (D-10 as revised, enforced by
  `tests/test_code_practice.py`): docstrings self-contained, no history, no
  v1 talk.
- House rules enforced by the same test: every optional argument
  keyword-only (D-104), American spelling in identifiers (D-100),
  `max_iterations` for budgets (D-99), every documented method's Args
  complete, public definitions documented.
- Metric rule: anything touching inner products is tested on
  `tests/conftest.py`'s `make_dense_metric_space`.
- Independent loops take `n_jobs=`.
- v1 names unless there is a strong reason; read v1 (`pygeoinf/`) before
  inventing (`inversion.py`, `linear_bayesian.py`, `backus_gilbert.py`,
  `nonlinear_optimisation.py`).
- `MPLBACKEND=Agg` for anything that plots; never `show()`.
- Before handing back a stage: `poetry run ruff check pygeoinf2/`,
  `ruff format` on the changed files, the stage's tests plus
  `tests/test_code_practice.py`; the coordinator runs the full fast suite
  (`MPLBACKEND=Agg poetry run pytest pygeoinf2 -q -m "not slow"`).
- Tutorials are scripts; after editing one, run
  `pygeoinf2/tutorials/make_notebooks.py` (the suite checks the notebooks
  are current).

## Stage 0 — the mechanical jobs (D-161, D-162)

- **DONE 2026-10-28.** D-162: `estimate_truncation_degree` is the walk
  past the truncation (was `sufficient_degree`), `power_degree` the capped
  question (`symmetric_space/base.py`, three test files; 912 tests pass).
- **DONE 2026-10-28 (624 tests in its group; the full suite not yet rerun).** D-161: `sem1d/` became
  `spectral_element/` (core: `SpectralElementSpace`, `AxisSpace`, the mesh,
  `Boundary`, `LengthScale`, the shell transform, the `Layered`
  implementation), `interval/` (`Lebesgue`, `Sobolev`, `Layered`;
  `interval.radial` the same three under `r² dr`), `ball/` (`Lebesgue`,
  `Sobolev`, `Layered`). Each geometry's `Layered` is a thin subclass of
  the core class taking that geometry's boundaries. `plotting/sem1d.py`
  splits into `plotting/interval.py` and `plotting/ball.py`; the tests
  `test_sem1d_*.py` become `test_interval.py`, `test_radial.py`,
  `test_ball.py`, `test_layered.py`, `test_spectral_element_plotting.py`;
  examples 30-33 and tutorials 10-11 follow; `pygeoinf2/__init__.py`
  imports the three on demand (planetmodel extra). `sem1d/` is deleted, no
  alias.

Then: CURRENT_STATE §2's row, the memory of the package name, full suite,
David commits.

## Stages 1 to 3 run in parallel, by file ownership

Shared file: `inference/__init__.py` (minimal line edits, re-read before
each edit). Nobody edits `pygeoinf2/__init__.py` or `inference/estimators.py`
except Stage 1; the coordinator fixes the top-level exports afterwards.

## Stage 1 — point estimators

Files: `inference/estimators.py`, `inference/point.py`, `compat.py`,
`tests/test_point.py`, `tests/test_inference.py`, `tests/test_compat.py`,
`tutorials/08_regularization.py`.

1. `PointEstimator` becomes the real base of every point-valued class.
   `LinearPointEstimator(PointEstimator, AffineOperator)` (drop the
   `PointEstimator.register` call); the renamed `MinimumNorm` and
   `ConstrainedMinimumNorm` subclass `PointEstimator` and `Operator`.
   `Estimator`'s abstract members (`data_space`, `property_space`,
   `__call__`, `push_forward`) satisfied on each; `push_forward` on the
   nonlinear-in-data ones composes the property operator so the
   derivative survives. Mind the MRO and `__init__` chains.
2. One minimum-norm class. `DiscrepancyPrinciple` → `MinimumNorm`; the
   fixed-damping `MinimumNorm(LeastSquares)` is deleted. `for_data` (the
   `LeastSquares` at the searched damping) and `discrepancy_search` become
   methods of the new class, which already has `estimator_at` and
   `search`. Docstring: the discrepancy principle is its method when the
   problem has an error, the misfit `level` its knob, exact data the fit at
   zero damping. `compat.py`'s v1 aliases updated.
3. `as_measure` deleted. `bias(model, /)` added: `resolution(model) -
   model`, documented as the term the propagated covariance leaves out;
   the class docstring states the budget: the error is `(B A - I) m + B e`,
   bounding the first term takes a statement about `m`, which is the
   posterior under a prior measure or the feasible set under a prior set.
   `resolution` and `propagated_covariance` docstrings say what they are
   not. Whoever wants the propagated Gaussian pushes the error measure
   through the estimator with the general machinery; not advertised.
4. Parity test: `MinimumNorm` on a linear problem with a Gaussian error
   against `FeasiblePropertySet.fitting_model()` from `BackusGilbertParker`
   with a non-binding ball prior and the problem's credible ball at the
   same level, on `make_dense_metric_space`; they share `misfit_search`.
5. Grep the package for `DiscrepancyPrinciple`, `as_measure`,
   `MinimumNorm.for_data`; report callers outside the stage's files.

## Stage 2 — nonlinear estimators

Files: `inference/laplace.py`, new `inference/minimization.py`,
`numerics/optimization.py`, `numerics/__init__.py`, `tests/test_laplace.py`,
`tests/test_optimization.py`, `tests/test_absolutely_continuous.py` (its
MAP uses), new tests, `tutorials/09_nonlinear.py`,
`tutorials/13_ray_tomography.py`, `examples/28_nonlinear_map.py`, the
README lines for those.

1. `MaximumAPosteriori(PointEstimator)`: `__call__(data, *, start=None)`
   returns the mode. `objective`, `log_posterior`, `linearized_normal`
   stay public (the sampling work needs the objective builder). The
   optimisation record stays one call away (`solve(data, *, start=None)`
   returning what `LaplaceResult` holds, or `LaplaceResult` under a better
   name).
2. `LaplaceApproximation(MeasureEstimator)` on a `MaximumAPosteriori`:
   `__call__(data, *, start=None)` returns the `GaussianPosterior` about
   the mode that `MaximumAPosteriori.at` builds today; `push_forward`
   conjugates. `GaussianPosterior`'s `AbsolutelyContinuousMeasure`
   behaviour preserved. **Settled (David, 2026-10-28):** it wraps a
   `MaximumAPosteriori` instance (plus the solver for the linearised
   normal operator), which states what Laplace is and lets any later point
   estimator with a Hessian at its solution be wrapped the same way; a
   classmethod `from_problem(problem, prior, ...)` builds the MAP inside
   for the common case. Not both as overloads of one constructor.
3. `Minimization(PointEstimator)` in `inference/minimization.py`: an
   `objective` callable (data → `Functional` on the model space), an
   `optimizer` (L-BFGS default), a `start` (vector, callable of the data,
   or the zero), the two spaces. `__call__` returns the model, `solve` the
   record. Derivative with respect to the data by the implicit function
   theorem at the solution when the Hessian and a `mixed_derivative=` are
   supplied; otherwise none. **Settled (David, 2026-10-28):** MAP is
   rebuilt on `Minimization`: its objective is the negative log posterior,
   its default start the prior's expectation, its Hessian the objective's;
   `log_posterior`, `linearized_normal` and the objective builder sit on
   the subclass. One optimisation loop. The Hessian plumbing is the thing
   to watch: MAP builds it from the problem's second derivative, and
   `Minimization` must take it from the objective's functional in general.
   Test: the rebuilt MAP gives the same mode and Gaussian as before on
   tutorial 9 and the ray problem.
4. `NonlinearLeastSquares(Minimization)`: a `ForwardProblem` with an error
   measure (`potential` is the misfit), Tikhonov damping about a reference
   model. Parity: on a `LinearForwardProblem` it agrees with `LeastSquares`
   at the same damping.
5. `AugmentedLagrangian` in `numerics/optimization.py`: `f(x)` subject to
   equalities `c_i(x) = 0` and inequalities `g_j(x) <= 0`, each a
   `Functional` with a derivative; standard multiplier updates and penalty
   growth; the inner unconstrained minimisation by any `Optimizer` passed
   in. Interface keeps the base `minimize` contract valid. Result reports
   multipliers and the final violation. Tests: a quadratic with a linear
   equality against the closed form; a ball-constrained quadratic against
   projection; on `make_dense_metric_space`.
6. `ConstrainedMinimization(Minimization)` on it; `Occam` is the smallest
   `|m - m0|` subject to the misfit potential at most a `level`. Not a root
   search over the damping (the misfit at the damped minimiser need not be
   monotone once the forward is nonlinear); a damping sweep only as an
   opt-in heuristic, and only if cheap. Parity: on a linear problem Occam
   agrees with `MinimumNorm` at the same level; on the ray problem the
   constraint is active at the solution and the norm is below the MAP's.
7. Tutorials 9 and 13 and example 28 to the new API, narratives kept.

## Stage 3 — set estimators

Files: `inference/backus.py`, `geometry/feasible.py`, `geometry/__init__.py`,
`tests/test_backus.py`, `tests/test_feasible.py`,
`examples/23_feasible_set.py`, `tutorials/07_properties.py`, README lines,
`plotting/sets.py` only if a removed type is referenced there.

1. The linear certificate becomes a route in the feasible property set: an
   `Algorithm` subclass in `geometry/feasible.py` (suggested
   `LinearCertificate`; its `via=` name must not be `certificate`, which
   already means the dual certificate) whose `support(direction)` returns
   the Minkowski sum's support value as an outer value with no inner, with
   `X` from the quadratic surrogate on quadratic sets or a user-supplied
   `operator=`. Accepts any structure whose sets have support functions;
   never the automatic choice. `BackusGilbertSet` survives internally or is
   folded; it is no longer a public estimator result.
2. `BackusGilbert` becomes the point estimator: a `LinearPointEstimator`
   with `X = T C A* (A C A* + α R)^-1` for a forward problem, a property
   operator, a model-space covariance or shape (identity by default), the
   error covariance, `α` a knob. Resolution and propagated error from the
   base, no prior bound. Docstring: the averaging-kernel estimate of the
   literature; the original paper already reached for norm bounds and the
   feasible property set is the completion of that thought; the same `X`
   with prior and noise sets gives a certified set through the
   linear-certificate route. The term SOLA appears nowhere.
3. The linearised feasible set: `BackusGilbertParker` accepts a nonlinear
   `ForwardProblem` with a model to linearise about (or a `Linearization`),
   builds the set on the derivative, documented as exact for the
   linearised problem and local for the real one. **Settled (David,
   2026-10-28): everything the user sees is absolute.** The constraint set
   is given on the model and the answer is read in the property's own
   values; the perturbation coordinates `δm = m − m*` stay inside. Three
   shifts happen together: the data by `−A(m*)`, the constraint set by
   `−m*`, the answer by `+T(m*)`. The set's shift must preserve its form
   (a ball or ellipsoid about `c` becomes one about `c − m*`, still
   quadratic, and the algorithms carry centres; a sublevel set composes its
   level function with the shift, still smooth): check that `ConvexSet`
   has such a translation, and if it only wraps as an oracle set, add a
   form-preserving one in `geometry/convex.py` first. The result exposes
   `m*` and `T(m*)`. The docstring says `m*` need not lie in the constraint
   set, but if it does not the linearisation is taken outside the region
   the set claims and the local statement is weaker. Tests: a linear
   problem linearised anywhere reproduces the linear set exactly, with an
   off-centre ball prior (the shift of the constraint set is the easy one
   to forget); on the ray problem the set contains the linearisation
   point's own property when the data are its own.
4. Example 23 and tutorial 7 to the new API; grep for `BackusGilbertSet`,
   `BackusGilbert(`, `from_operator` and report callers outside the stage.

## Stage 4 — the proof

Tutorial 14: one linear problem answered as a point, a set and a measure.
The linear estimate with its propagated standard deviation; the bias
against a known truth, larger than it; the feasible set under a prior set
and the posterior under a prior measure as the two completions; the same
three answers pushed to a property. Then the nonlinear coda on the ray
problem: MAP, Laplace, Occam, the linearised set.

## Afterwards

- `pygeoinf2/__init__.py` exports; `inference/__init__.py` reconciled.
- CURRENT_STATE §2 (package map), §4 (the estimator table), §8 if a
  caller-visible name changed, §10 (items struck).
- EXAMPLES_PLAN: the C++ vector-backend mini-app added (a small C++
  vector type with its inner product and a sparse operator, wrapped with
  pybind11, no numpy inside the vector, run through every `check_*` and
  every estimator; the minimal non-numpy `HilbertSpace` under the suite is
  its regression form); tutorial 14 added to tier 2.
- Memory notes; this file deleted; full suite; David commits.

## Open

Nothing: the three questions were settled with David on 2026-10-28 and
are recorded in Stages 2 and 3 above.
