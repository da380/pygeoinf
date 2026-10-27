# Code practice scan, 2026-10-27

**Scope.** Every `.py` file under `pygeoinf2/` (72 modules), excluding `tests/`, `examples/` and `tutorials/`. This is a read-only review: no code was changed.

**Method.** AST scans for:

- `x._name` where `x` is not `self` or `cls` (this includes `self._a._b`);
- `_private` names imported from another module;
- `getattr`, `hasattr`, `setattr` and `__dict__`;
- mutable defaults, bare or broad `except`, module-level mutable state, and attribute stores on objects other than `self`;
- private definitions that nothing references, and unreachable statements;
- British spellings in identifiers, and positional optional parameters.

`ruff` was also run with F401, F811, F841, B006, B008, E722, B904, B018 and B023. Its only hits were B023 closures in `numerics/solvers.py`, which are the documented lazy `SolveStep.iterate` and so are intended. Every finding below was read in context.

Items already in `CODE_ISSUES.md` are not repeated. These include:

- `PrimalKKTSolver._quadratic` reading `Ellipsoid._covariance` and `_precision`;
- `parallel._context_is_set`;
- the `except Exception` in `plotting/sets.py`;
- `Polytope.contains`'s `hasattr`;
- the `AttributeError`s from `_OracleSet` and `Ellipsoid`;
- the empty `if TYPE_CHECKING: pass` block in `algebra/operators.py`;
- `MinResSolver`'s `ValueError`.

**Paths and line numbers.** Paths are relative to `pygeoinf2/`, and line numbers are at HEAD `62bae97`. Four files were being edited while the scan ran: `geometry/feasible.py`, `numerics/normal.py`, `numerics/__init__.py` and `parallel.py`. Their line numbers refer to HEAD (`git show 62bae97:pygeoinf2/<path>`). New hits in the live `geometry/feasible.py` are listed at the end, by symbol name.

## Summary

| category | findings |
|---|---|
| A: private access across class boundaries | 22 (plus 7 groups of same-class or same-hierarchy access, lower priority) |
| B: duck-typing and private probing | 8 |
| C: mutability and exception hygiene | 5 |
| D: dead or duplicated code, inconsistent exceptions | 9 |
| E: public API inconsistency | 5 |
| Live `geometry/feasible.py` (provisional) | 4 |

## A. Private access across class boundaries

### A.1 Across classes, modules or subpackages

- `algebra/spaces.py:780`: `CoordinateSpace.coordinate_selection` writes `operator._components_action_fn` and `_components_adjoint_action_fn` onto whatever `LinearOperator.from_callables` returns. `LinearOperator.from_formal_adjoint` does the same at `algebra/operators.py:1578` and `:1587`.
  - **Why it matters:** this relies on the factory returning a `_CallableLinearOperator`, and on that class's private field names. `InverseOperator` already takes `components_action=` and `components_adjoint_action=`; `from_callables` does not.
  - **Fix:** add those two keywords to `from_callables` and `_CallableLinearOperator.__init__`, and pass them.
- `algebra/spaces.py:1066`: `_CoordinateMassWeightedSpace._mass_on_components` calls `operator._components_action()`. `symmetric_space/base.py:1058` (`SymmetricSpace.correlated_measure`) does the same.
  - **Why it matters:** a space reaches into an operator's extension hook, and no public accessor exists.
  - **Fix:** add a public read-only `LinearOperator.components_action()` that returns `None` when unavailable, and keep `_components_action` as the override point.
- `backends/mfem.py:43`: imports `_resolve_rng` from `algebra.spaces`. So do `backends/mfem_hilbert.py:58` and `compat.py:49`.
  - **Why it matters:** one private helper is shared across three subpackages, and a second `_resolve_rng` with different behaviour exists (see D).
  - **Fix:** keep one public helper for resolving `rng=None`.
- `backends/mfem.py:694`: `_white_noise_integrator` imports `mfem._ser._lininteg` and calls SWIG internals, to work around a PyMFEM `NameError`.
  - **Why it matters:** it is tied to the serial build's private module layout.
  - **Fix:** try the public `WhiteGaussianNoiseDomainLFIntegrator(seed)` first and fall back only on `NameError`, so a fixed PyMFEM takes the public path.
- `geometry/feasible.py:397` (HEAD): `ClosedForm` reads `feasible._structure` and calls `feasible._cached(...)` at `:397, 402, 433, 488, 501, 519, 551, 592`. The live file has 13 such sites, now including `Search` and `Primal`.
  - **Why it matters:** `Algorithm` is public and is meant to be passed as an instance through `via=`. A user-written algorithm would therefore need the same private access.
  - **Fix:** give `FeasiblePropertySet` a public `structure` (with `_Structure` made public) and a public `cached(key, build)`, or pass both into the `Algorithm` methods.
- `geometry/subspaces.py:322`: `AffineSubspace.from_linear_equation`, and `LinearSubspace.from_kernel` at `:634`, call `OrthogonalProjector._onto_kernel(operator, inverse)`.
  - **Why it matters:** this is another class's private classmethod (same module).
  - **Fix:** add an `inverse=` keyword to the public `OrthogonalProjector.onto_kernel` (`:167`), or move the body into a module function.
- `inference/backus.py:1363`: `_DualRoute._smoothed_support` reads `given._covariance`, at `:1363` and `:1365`.
  - **Why it matters:** `Ellipsoid.covariance` is a public property returning the same object. This is a different site from the `PrimalKKTSolver` one already in CODE_ISSUES.
  - **Fix:** use `given.covariance`.
- `inference/backus.py:2139`: `FeasiblePropertySet` reads `BackusGilbertParker` private members:
  - `_inclusion` at `:2139` and `:2165`;
  - `_need_inclusion` at `:2359`;
  - `_prior_level` at `:2371`;
  - `_likelihood_engine` at `:2464`.
  - **Why it matters:** the set's behaviour depends on the estimator's internals, although `estimator.membership` and `estimator.prior.level` already expose two of these publicly.
  - **Fix:** use `membership is not None` and `prior.level`, and give the estimator one public membership-engine accessor for the rest.
- `inference/backus.py:2294`: `FeasiblePropertySet.extremal_model` reads `self._algorithm._kernel`. That is the `_ClosedFormRoute` kernel projector, reached through the public `estimator.algorithm`.
  - **Why it matters:** it is a private field of the route object.
  - **Fix:** add a `kernel_projector` property, or an extremal-model method, on `_ClosedFormRoute`.
- `inference/gaussian.py:42`: imports `_combine_log_normalizer` from `probability.absolutely_continuous`.
  - **Why it matters:** a module-private helper is used from another subpackage.
  - **Fix:** it only combines `Estimate`s, so make it a public function next to `Estimate` in `numerics/randomized.py`.
- `numerics/functional_calculus.py:37`: imports `_REORTHOGONALISATION_THRESHOLD` from `algebra.spaces`.
  - **Why it matters:** a private constant is shared across subpackages, and its name is in British spelling (see E).
  - **Fix:** make it a public constant (`REORTHOGONALIZATION_THRESHOLD`).
- `numerics/functional_calculus.py:804`: `log_determinant` calls `operator._known_matrix("galerkin")`. So do `numerics/root_find.py:490` (`DampedSolves._matrices`) and `probability/gaussian.py:784` (`GaussianMeasure.hilbert_schmidt_norm`).
  - **Why it matters:** three subpackages need "the matrix if it is free", and all three use the protected hook to get it.
  - **Fix:** add a public `LinearOperator.known_matrix(*, form)` that returns `None` instead of probing.
- `numerics/preconditioners.py:337`: `_SpectralFamilyPreconditioner.at` calls `self._preconditioner._inverse_of(...)`, and `_NormalDiagonalFamily.at` calls `_base_diagonal` and `_inverse_of` at `:1466` and `:1468`.
  - **Why it matters:** helper classes call private methods of the preconditioner that owns them (same module).
  - **Fix:** pass the bound methods they need into the family's constructor.
- `numerics/randomized.py:166`: calls `space._orthonormalize_columns` (`:166`, `:414`) and `codomain._orthogonalize_against` (`:347`).
  - **Why it matters:** numerics depends on private methods of `CoordinateSpace` and `HilbertSpace`, and the public `orthonormal_basis` and `gram_schmidt` take no `against=`.
  - **Fix:** make both public.
- `numerics/root_find.py:511`: `DampedSolves._solver_for` calls `solver._carrying(...)` on an `IterativeSolver`.
  - **Why it matters:** the public siblings do not cover "carry this prebuilt preconditioner". `with_preconditioner` refuses to replace an existing one, and `resolved_for` builds its own.
  - **Fix:** make `_carrying` public, for example as `with_resolved_preconditioner`.
- `plotting/sphere.py:142`: the registered `plot` for `Sphere` reads `space._core_values(field)`, `space._core_colatitudes` and `space._core_longitudes` (`:142`–`146`).
  - **Why it matters:** plotting depends on how `Sphere` lays out its padding (the wrap column and the pole row).
  - **Fix:** add a public `Sphere` method returning the transform grid's colatitudes, longitudes and values.
- `probability/absolutely_continuous.py:176`: `relative_to` returns `self._relative_log_density - other._relative_log_density`. This is the instance that was reported.
  - **Why it matters:** strictly it is same-class access, but it bypasses the public `relative_log_density` property for no reason, and a subclass that overrides the property would be silently skipped. The same module's `_combine_log_normalizer` is also imported across subpackages (see `inference/gaussian.py:42` above).
  - **Fix:** use `other.relative_log_density`.
- `sem1d/layered.py:45`: `Layered` subclasses `algebra.direct_sum._CoordinateDirectSum`, and its `__hash__` at `:281` rebuilds `HilbertSpace.__hash__`'s formula, `hash((_CoordinateDirectSum, self._key()))`.
  - **Why it matters:** it couples to a private class from another subpackage and to a private hashing scheme. If `HilbertSpace.__hash__` changes, `Layered` stops hashing equal to the plain direct sum it compares equal to.
  - **Fix:** make the coordinate direct sum public, and let it declare its hash key once.
- `sem1d/layered.py:463`: `Layered._require_point_evaluation` calls `self.layers[i]._require_point_evaluation(...)`.
  - **Why it matters:** a direct sum calls a private hook on its summands. That hook has three different signatures: `(what)` on `SymmetricSpace`, `(what, points)` on `SpectralElementSpace`, and `(what, points, side)` here.
  - **Fix:** keep a private shared signature, or make it a public check method on the sem1d spaces.
- `symmetric_space/base.py:2809`: `_FlexureOperator._analyze` calls `space._own_grid_values(grid)`.
  - **Why it matters:** an operator class uses the space's no-copy constructor, which is safe only for arrays the space itself allocated.
  - **Fix:** give `from_grid_values` a public `copy=False` option, or keep the call and say in a comment why the array is unaliased.

### A.2 Same class or same hierarchy (acceptable Python idiom; lower priority)

- `algebra/nodes.py` (26 sites), `algebra/direct_sum.py:419, 450, 596, 693, 769, 837, 890`, `algebra/operators.py:376, 421, 1192, 1222, 1901`, `numerics/normal.py:171–195` (HEAD), `numerics/solvers.py:403, 936–937, 956, 961`, `symmetric_space/base.py:2713` and `probability/base.py:405`: subclasses of `LinearOperator`, `Operator` or `ProbabilityMeasure` call the extension hooks on other instances. The hooks are `_known_matrix`, `_known_diagonals`, `_components_action`, `_adjoint_apply_block`, `_link_adjoint` and `_combine_*`. These are the extension protocol, called within its own hierarchy. **Fix:** none needed. Optionally, state in `LinearOperator`'s docstring that these hooks are protected.
- `algebra/operators.py:600` and `numerics/solvers.py:1068, 1106`: `clone._traits` and `clone._preconditioner` are set on copies of `self`. These are fine.
- `algebra/spaces.py:156`, `sem1d/layered.py:278` and `backends/mfem_hilbert.py:531`: `_key()` is called on another space of the same hierarchy. These are fine.
- `geometry/subspaces.py:325–327, 494–496, 519–521, 635–637`: `_equation`, `_solver` and `_inverse` are written onto a freshly built instance after construction. This is same class, but the same three-line block repeats four times (see D).
- `probability/gaussian.py:1071, 1118, 1124, 1212–1217, 1232–1233`: these read another `GaussianMeasure`'s `_diagonal_eigenvalues()`, `_symmetric_matrix()`, `_weighted_squared()`, `_precision`, `_precision_factor` and `_require_covariance`. Public `precision` and `precision_factor` exist, so use them at `1212–1215` and `1232–1233`.
- `sem1d/base.py:345` and `symmetric_space/base.py:174`: `other._coordinate_key()` is called after an `isinstance` guard. Fine.
- `symmetric_space/fourier.py:885` and `symmetric_space/sphere.py:2297`: `target._component_index()` and `target._packing` are read in `degree_transfer_operator`. Fine.

## B. Duck-typing and private probing

- `algebra/nodes.py:741`: `getattr(self.codomain, "is_orthonormal", False)`. The same pattern appears at `numerics/preconditioners.py:395`, and at `geometry/convex.py:112` as `getattr(self.level_function(), "has_derivative", False)`.
  - **Why it matters:** the base classes (`HilbertSpace`, `Operator`) declare these attributes, so the default can never apply and would only hide a typo.
  - **Fix:** read the attribute directly.
- `geometry/feasible.py:1072` (HEAD): `level_function` checks `hasattr(algorithm, "level_gradient")`.
  - **Why it matters:** this capability is not declared on `Algorithm`, so a subclass learns about it only by reading this code.
  - **Fix:** declare `level_gradient` on `Algorithm`, raising `NotImplementedError`, and advertise it in `probes`.
- `geometry/sets.py:293`: `SublevelSet` declares none of `has_level_function`, `has_support_function`, `has_maximizer` or `has_membership`, and spells its level function as the property `functional` rather than `level_function()`.
  - **Why it matters:** that is why `inference/backus.py:1751–1753`, `2606–2616`, `numerics/convex.py:2588–2597` and `plotting/sets.py:214–221, 250` probe with `getattr(..., default)` or `isinstance(SublevelSet)`.
  - **Fix:** declare the capability flags (default `False`) and `level_function()` on `Subset`, implement them on `SublevelSet`, and drop the probes.
- `inference/backus.py:2387`: `inclusion_norm` checks `hasattr(engine, "inclusion_norm")` to decide whether the prior is quadratic.
  - **Why it matters:** this is a capability test on private route classes.
  - **Fix:** test the flag the estimator already computes (a quadratic prior), or declare the method on every engine.
- `numerics/convex.py:2588`: `_level_function_of` recognises a `SublevelSet` by `hasattr(subset, "functional") and not hasattr(subset, "level_function")`.
  - **Why it matters:** this is fragile duck typing. `numerics` may import `geometry`, and the same function already imports `geometry.convex` locally.
  - **Fix:** use `isinstance(subset, SublevelSet)`, or better the `Subset` capability above.
- `numerics/preconditioners.py:1026`: `WoodburyPreconditioner.from_normal` probes `hasattr` for `forward`, `prior_covariance` and `error_covariance`, then `getattr(..., None)` for `prior_precision` and `error_precision` (`:1051–1053`).
  - **Why it matters:** `_require_normal` in the same module (`:1282`) checks the same thing with `isinstance(operator, FactoredNormalOperator)`.
  - **Fix:** use `_require_normal`, or the same `isinstance` check.
- `probability/gaussian.py:810`: `_spectral_slices` recognises a block-spectral operator by `getattr(..., "slices")`. In the same way, `numerics/randomized.py:786, 961` look for `getattr(factor, "columns")` and `numerics/preconditioners.py:1148` for `getattr(operator, "scale", 1.0)`.
  - **Why it matters:** fast paths keyed on an attribute name will misfire on any operator that happens to have an attribute of that name with another meaning.
  - **Fix:** declare these as `None`-returning hooks on `LinearOperator`, as `_known_matrix` already is.
- `probability/gaussian.py:2524`: `_combine_affine` imports `algebra.nodes._Identity` and tests `isinstance(operator, _Identity)`.
  - **Why it matters:** this dispatches on a private class from another subpackage.
  - **Fix:** add a public `LinearOperator.is_identity` property (true on `_Identity`).

## C. Mutability and exception hygiene

- `algebra/spaces.py:40`: `_DEFAULT_RNG = default_rng()` is a single module-level generator, used whenever `rng=None` in `algebra/spaces.py`, `compat.py` and `backends/*`.
  - **Why it matters:** this is hidden shared state, and it differs from the rest of the package, which builds a fresh generator on each call (see D).
  - **Fix:** choose one convention and apply it everywhere.
- `backends/mfem.py:518`: sets `solver._pygeoinf_keepalive = (...)` on SWIG solver objects. `backends/mfem_hilbert.py:354, 370` do the same, including on the object the caller's `make_solver` returned.
  - **Why it matters:** it writes onto a third-party object. If a user's factory returns the same solver for two systems, the attribute is overwritten and the earlier references are dropped.
  - **Fix:** hold the keep-alive references on the pygeoinf object that owns the solver.
- `geometry/convex.py:560`: `_LevelSupport._solve` reads `self._last` three times, as a check-then-act: `self._last is not None and self._last[0] is direction`, then `return self._last[1]`.
  - **Why it matters:** if the support function is shared between threads (the live `parallel.py` adds `threads=`), a write between the check and the return hands one direction's answer to another.
  - **Fix:** read it once with `last = self._last` and test that.
- `geometry/feasible.py:187` (HEAD): `_Structure.__init__` compares `prior.domain` before its `isinstance(prior, ConvexSet)` check at `:191`.
  - **Why it matters:** a non-set prior raises `AttributeError` instead of the `TypeError` the docstring promises.
  - **Fix:** do the type check first.
- `plotting/fourier.py:175`: sets `ax.center_line` on the caller's Axes. `plotting/sphere.py:182, 208` set `ax.contour_set` and `ax.gridliner`.
  - **Why it matters:** these are ad hoc attributes on a third-party object. They are present only on some code paths, and a second call overwrites them.
  - **Fix:** return them in the result (for example, a small named tuple).

## D. Dead or duplicated code, inconsistent exceptions

- `geometry/feasible.py:238` (HEAD): `_Structure.shared(key, build)` and `FeasiblePropertySet._cached(key, build)` at `:775` are the same memo under an `RLock`.
  - **Why it matters:** two copies of the same logic can drift.
  - **Fix:** use one small cache class for both.
- `geometry/subspaces.py:325`: the block `_equation`, `_solver`, `_inverse` is assigned after construction in four places (`:325, 494, 519, 635`).
  - **Why it matters:** it is duplicated, and an instance built through `__init__` alone is in a different state.
  - **Fix:** use one private helper (for example `_with_equation(operator, value, solver, inverse)`) or a private constructor keyword.
- `geometry/subspaces.py:462`: several places raise `AttributeError` for "this ingredient is missing":
  - the properties `constraint_operator` and `constraint_value` (`:462`, `:480`);
  - `inference/estimators.py:138` (the `resolution` property) and `:158` (`propagated_covariance()`);
  - `inference/problem.py:94, 111, 131` (`error`, `error_measure`, `error_set`).
  - **Why it matters:** `GaussianMeasure._require_covariance` raises `ValueError` for the same kind of failure. An `AttributeError` raised from a property is also swallowed by `hasattr` and by `getattr(..., default)`.
  - **Fix:** raise `ValueError` consistently (CODE_ISSUES covers only the `convex.py` sites).
- `inference/backus.py:596`: "a `Ball` means `r² I` and `I / r²`, an `Ellipsoid` means its covariance (or `solver(precision)`), plus its centre" is written out five times:
  - `_Quadratic` (`:596–616`);
  - `_DualRoute._smoothed_support` (`:1348–1375`);
  - `numerics/convex.py:2185–2192`;
  - `geometry/feasible.py:261–277` (HEAD);
  - `_precision_of` in the live `feasible.py`.
  - **Why it matters:** five copies of the same logic can drift.
  - **Fix:** give `Ball` `covariance` and `precision` properties like `Ellipsoid` (or a shared quadratic-form accessor), and delete the chains.
- `inference/backus.py:1743`: `_level_function_of` also exists at `numerics/convex.py:2586`. The two copies detect `SublevelSet` differently and raise different messages. A third and fourth check for "differentiable level function" are `_has_level_function` (`:2599`) and `ConvexSet._level_function_is_differentiable` (`geometry/convex.py:109`).
  - **Why it matters:** the copies already disagree in detection and messages.
  - **Fix:** one helper in `geometry`, once B's `Subset` capability exists.
- `inference/backus.py:2037`: `_FeasibleSupport` is defined again at `geometry/feasible.py:614` (HEAD), and `_is_point` (`geometry/feasible.py:156`, HEAD) duplicates `BackusGilbertParker._exact` (`backus.py:2815`).
  - **Why it matters:** this is migration duplication.
  - **Fix:** drop the `inference` copies when the migration completes.
- `inference/backus.py:2820`: the property `BackusGilbertParker._balls` is never referenced. **Fix:** delete it.
- `probability/base.py:28`: `if TYPE_CHECKING: pass` is an empty block, and `TYPE_CHECKING` is imported only for it. **Fix:** delete both (CODE_ISSUES lists only the `algebra/operators.py` copy).
- `probability/mixture.py:46`: a second `_resolve_rng`, which returns a fresh generator, shares its name with `algebra/spaces.py:47`, which returns the module singleton.
  - **Why it matters:** about 30 more sites spell the same thing inline, as `default_rng() if rng is None else rng`. These are in `numerics/quadratic_forms.py`, `numerics/randomized.py`, `probability/base.py`, `sem1d/{ball,interval,radial}.py`, `symmetric_space/{box,fourier,sphere}.py` and `testing.py` (13 sites).
  - **Fix:** one public helper, used everywhere.

## E. Public API inconsistency

- `__init__.py:93`: `pygeoinf2.FeasiblePropertySet` is the `inference.backus` class, whose constructor is `(estimator, data)`. `pygeoinf2.geometry.FeasiblePropertySet` (`geometry/__init__.py:17`) is a different class, whose constructor is `(A, T, prior, data, *, noise, solver)`.
  - **Why it matters:** the same public name refers to two incompatible classes, depending on the import path.
  - **Fix:** rename one, or stop exporting one, until the migration completes.
- `geometry/feasible.py:657` (HEAD): the public `FeasiblePropertySet.__init__` takes a private keyword `_structure=`, and the public `Algorithm.accepts(structure: _Structure)` (`:294`) is typed with a private class.
  - **Why it matters:** private types leak into public signatures.
  - **Fix:** move the shared-structure path into a private classmethod (`_from_structure`), and make the structure type public if `Algorithm` stays public.
- `numerics/normal.py:389` (HEAD) and `numerics/tikhonov.py:236`: `NormalOperator.assembled` and `TikhonovNormalOperator.assembled` are properties, overriding the method `LinearOperator.assembled(*, n_jobs)` at `algebra/operators.py:1140`.
  - **Why it matters:** generic code calling `op.assembled()` gets a `TypeError` on these subclasses (it calls the returned operator with no argument), and `n_jobs` cannot be passed. This breaks substitutability.
  - **Fix:** rename the property (for example `plain_operator`), or make it a method with the base signature.
- `numerics/preconditioners.py:1471`: the public class `LocalisedPreconditioner` (exported at `__init__.py:105, 173`) uses British spelling.
  - **Fix:** rename it `LocalizedPreconditioner`, and keep the old name as an alias if needed.
- `algebra/spaces.py:44`, `plotting/distributions.py:35–36` and `sem1d/_transform.py:76, 98`: the private identifiers `_REORTHOGONALISATION_THRESHOLD`, `_PRIOR_COLOURS` and `_POSTERIOR_COLOURS`, and the method names `ShellTransform.synthesise` and `analyse`, use British spelling. The last two mirror planetmodel's `SpectralBasis` names, which may be deliberate.
  - **Fix:** use American spellings, or note the mirroring.

## Live working copy of `geometry/feasible.py` (provisional; being edited, lines shifting)

- `_check_floor`: `getattr(structure.solver, "_rtol", None)` probes a private attribute of the user's solver. Any solver without `_rtol`, such as a direct solver or a user's own, silently skips the check. **Fix:** add a public `IterativeSolver.rtol` and test `isinstance(solver, IterativeSolver)`.
- `Search.feasibility`, `Search.inclusion`, `Primal.support`: these add new `feasible._structure` and `feasible._cached` sites (see A.1, `geometry/feasible.py:397`).
- `_precision_of`: a fifth Ball/Ellipsoid chain (see D, `inference/backus.py:596`).
- `Primal.support`: the local `centre_term` uses British spelling; rename it `center_term`.

## Scanned and clean

`algebra/__init__.py`, `algebra/diagonal.py`, `algebra/linearization.py`, `backends/__init__.py`, `data/__init__.py`, `datasets.py`, `geometry/__init__.py`, `inference/__init__.py`, `inference/laplace.py`, `inference/mixture.py`, `inference/point.py`, `numerics/__init__.py`, `numerics/line_search.py`, `numerics/optimization.py`, `numerics/quadratic_programming.py`, `parallel.py` (apart from the CODE_ISSUES item), `plotting/__init__.py`, `plotting/base.py`, `plotting/sem1d.py`, `probability/__init__.py`, `sem1d/__init__.py`, `sem1d/axis.py`, `symmetric_space/__init__.py`, `symmetric_space/circle.py`, `symmetric_space/line.py`, `symmetric_space/plane.py`, `symmetric_space/torus.py`, `traits.py`.

## Resolved the same day, in the implementing session (2026-10-27)

The findings in code written that day were fixed before the scan was handed
over, so they need no action: `FeasiblePropertySet` exposes `structure` and
`cached(key, build)` and `FeasiblePropertySet.from_structure(...)` replaces
the private `_structure=` keyword, so an `Algorithm` reaches nothing private
(`_Structure` is `FeasibleStructure`, exported); `_check_floor` reads the
public `IterativeSolver.rtol` and skips direct solvers by `isinstance`;
`IterativeSolver._carrying` is the public `with_resolved_preconditioner`;
`_LevelSupport`'s memo is thread-local; `level_function` asks the
`Algorithm.level_gradient` hook rather than probing with `hasattr`;
`FeasibleStructure.__init__` tests the prior's type before its domain. The
two public `FeasiblePropertySet` classes (E) and the duplicated
`_FeasibleSupport`, `_is_point` and the Ball/Ellipsoid chain (D) are
transitional: Stage 5 of `CONVEX_PLAN.md` deletes the inference copies.
