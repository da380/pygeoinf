# Implementing D-134: the convex algorithms move into the sets

A working document for the implementing session, deleted when the work is
done. The design is fixed by `DECISIONS.md` D-134 (read it first, then
D-79/D-80/D-82 and §7 of `CURRENT_STATE.md`); this plan turns it into
stages. The implementer records sub-decisions as D-135 onward (never
renumber), runs the fast suite between stages, and never commits — David
commits.

> **Status, 2026-10-24: not started; the design is reopened.** The first
> implementing session read this plan against the code and talked it through
> with David, question by question, before writing anything. The scope grew
> past what the plan covers: matrix-free algorithms are now part of the
> initial work, with dense methods a special case, and the algorithms become
> public objects applied to sets. David judged that this needs a design pass
> with library-wide context before implementation. The record of that
> discussion is the last section, *Session of 2026-10-24*; where it
> conflicts with the stages below, it wins. No code was changed.

## The binding rules

- **Do not change the numerics of any solver.** The bundle, KKT,
  Chambolle-Pock and bisection numerics were deliberately restored to v1's
  in September (D-13 and git history); this job moves interfaces, not
  mathematics. If a numerical change seems forced, stop and record the
  question instead.
- **The parity figures are the regression bar.** The dual, primal, smoothed
  and KKT routes agreed to between 1e-3 and 1e-11 over sixteen directions
  (CURRENT_STATE §4). The cross-route tests in `tests/test_backus.py` and
  `tests/test_quadratic_programming.py` must survive, re-spelled for the
  new probe-level forcing, with the same tolerances.
- House rules enforced by `tests/test_code_practice.py`: every optional
  argument keyword-only (D-104), American spelling in identifiers (D-100),
  `max_iterations` for budgets (D-99), every documented method's Args
  complete. Run `poetry run ruff check pygeoinf2/` and
  `ruff format --check pygeoinf2/` over the WHOLE package before handing
  back, plus the full fast suite
  (`MPLBACKEND=Agg poetry run pytest pygeoinf2 -q -m "not slow"`; it must
  come back all green, ~3050 tests).
- Metric rule: anything touching inner products is tested on
  `tests/conftest.py`'s `make_dense_metric_space`, not only on Euclidean
  spaces.
- Update `CURRENT_STATE.md` §2 (package map), §4 (the route tables) and
  §10 at the end to match what was built.

## Stage 0 — unify the solver results (numerics only, no behavior change)

Today four result types coexist: `OptimizationResult`
(`numerics/optimization.py`) and the convex module's `BundleResult`,
`KKTResult`, `SaddlePointResult`. D-134(4) folds them.

1. Make `ProximalBundleMethod` and `LevelBundleMethod` subclasses of
   `Optimizer` with the standard `minimize(functional, x0)` entry where
   their problem shape allows it; where they genuinely need more than
   `(functional, x0)` (the level method's constraint set), keep the richer
   entry point but return the unified result. Read their current call
   sites in `inference/backus.py` and the tests first; adapt interfaces,
   not internals.
2. One result shape: everything returns `OptimizationResult` or a subclass
   adding route-specific diagnostics as extra frozen fields (e.g. a
   `BundleOptimizationResult` carrying the gap and the cut count; the KKT
   and saddle solvers likewise). `converged`, `iterations`, `message`,
   `value` mean the same thing everywhere. The old names can remain as
   aliases of the subclasses during this stage and die in stage 2 if
   nothing external needs them (pre-release: no deprecation machinery).
3. `tests/test_convex.py` has a test asserting `BundleResult` "reads like
   an OptimizationResult" by field comparison — that test's reason
   disappears; replace it with an `isinstance` assertion.

Record the result-shape choice as a D-entry.

## Stage 1 — the dispatch machinery, in the geometry layer

New module `geometry/probing.py` (name open to the implementer; NOT inside
`inference/backus.py` — D-134(3) requires the machinery to live in
geometry from day one).

1. **Forms.** The dispatch predicate vocabulary already exists as
   `BackusGilbertParker`'s private properties `_exact`, `_balls`,
   `_quadratic`, `_differentiable` (backus.py ~line 2766) over the
   `ConvexSet` capability flags (`has_membership`, `has_projection`,
   `has_support_function`, `has_maximizer`, level functions). Lift this
   into a small, explicit description of "the form a probe sees" — do not
   build a taxonomy enum; a predicate on the participating sets is enough
   and stays extensible.
2. **The table.** Per probe kind (`support`, `membership`, at minimum), an
   ordered registry of entries: a name (the current route names:
   `"closed_form"`, `"bisection"`, `"dual"`, `"primal"`, `"kkt"`,
   `"smoothed"`; `"reduced"`, `"likelihood"` on the membership side), an
   `accepts(form)` predicate, and a rank. `auto` = first accepting entry
   in rank order; the initial ranking reproduces today's `_choose` and
   `_choose_membership` exactly, so behavior is unchanged by construction.
   D-82 (should KKT outrank the bundle on quadratic sets?) is answered by
   TIMING it once the table exists — do the measurement, record the
   result and the chosen rank as a D-entry either way.
3. **Forcing.** `via=` on a probe accepts the entry's name (string) or an
   algorithm *instance* (for tuning: tolerances, budgets, a solver); a
   request the form does not admit is refused with a message naming which
   entries do — the same courtesy the estimator's `route=` gives today.
4. Keep the machinery exactly as small as `FeasiblePropertySet` needs. No
   adoption by `Intersection`/images/preimages in this job (D-134(3):
   incremental, on demand).

## Stage 2 — rewire `inference/backus.py`

1. The four private route classes (`_ClosedFormRoute`, `_BisectionRoute`,
   `_DualRoute`, `_LikelihoodRoute`) and the membership routes become the
   table's algorithm objects. They keep their mathematics untouched; they
   gain the `accepts`/name/rank clothing. The dual algorithm's cost
   (`dual_cost`) and per-route diagnostics become public on the algorithm
   objects — the honest home D-134(5) asks for, replacing the
   "for diagnostics" `algorithm` property back door.
2. `BackusGilbertParker.__init__` loses `route=`, `membership=`,
   `method=`, `optimizer=` (pre-release; no deprecation). It keeps
   `problem, property_operator, prior, /, *, noise=, level=, solver=`
   — `solver=` stays because hardening and the closed form need a default
   linear solver and it is estimator-wide, not probe-specific; probe-level
   tuning goes through `via=` instances. `max_iterations` moves to the
   algorithm instances. Record this signature as a D-entry.
3. `FeasiblePropertySet` probes gain `via=`: `support(d, via=...)`,
   `support_values(..., via=...)`, `contains(p, via=...)`, `extent`,
   `inner_hull`, `is_empty` — each resolves through the stage-1 table at
   call time, caching what it computes as today. Capability flags
   (`has_level_function` etc.) derive from the table: "some entry
   accepts this form".
4. `FeasiblePropertySet` becomes constructible directly from sets, without
   a `ForwardProblem`: a constructor or classmethod taking the forward
   operator, the property operator, the prior set, the data and the
   confidence set. The estimator's `__call__` goes through it.
5. Tests: re-spell the existing route-forcing tests as probe-level `via=`;
   keep every cross-route parity assertion and tolerance; add tests that
   `auto` picks the documented entry for each form (exact+ball →
   closed_form; two quadratic sets → bisection; general → dual), that a
   refused `via=` names the admissible entries, and that an algorithm
   instance with tuning is honored. The example `23_feasible_set.py` and
   anything else using `route=`/`membership=` must be updated to the new
   spelling (grep examples/ and work through `tests/test_examples.py`).

## Stage 3 — reconcile the documents

- `CURRENT_STATE.md` §4: the two route tables stay (they describe the
  algorithms), but the sentence naming `route=` changes to the probe
  spelling; §2 map rows for `geometry` and `numerics` mention the probing
  module and the unified results; §7's D-134 entry notes implementation
  landed; §10 strikes the item.
- `DECISIONS.md`: D-135+ for the sub-decisions above; edit D-82 in place
  with the measurement's verdict.
- Delete this file when everything above is done, or leave a short list of
  what remains if stopping at a stage boundary.

## Stopping rule

Each stage ends with the full fast suite green and documents consistent.
If a stage turns out to need a design decision this plan does not cover,
record the question in this file, finish the stage cleanly if possible,
and stop rather than improvise architecture — the reviewing session picks
it up.

---

## Session of 2026-10-24: the discussion, for the design pass

The plan was read against `inference/backus.py` (all of it), the bundle,
KKT and Chambolle-Pock solvers and their result types in
`numerics/convex.py`, `OptimizationResult` and `Optimizer`, D-134 and
D-79 to D-82, and the tests that force routes. Skimmed only, and needing a
proper read in the design pass: `inference/normal.py`, `inference/tikhonov.py`,
`inference/point.py` (`misfit_search`), `inference/preconditioners.py`,
`numerics/root_find.py` (`DampedSolves`), `numerics/solvers.py`.

Baseline before anything: `3053 passed, 32 deselected` in 2 min 47 s.

### Corrections to the plan above

- **Stage 0.1.** `LevelBundleMethod` takes no constraint set. Both bundle
  methods take `minimize(functional, start, /, *, subgradient=None)`, so
  both fit `Optimizer.minimize(functional, x0)` with `subgradient=` as an
  extra keyword, as `ProximalGradient` carries `nonsmooth=`. Two traps when
  they subclass `Optimizer`: their `self._tolerance` attribute (a float)
  would shadow `Optimizer._tolerance()` (a method), and
  `ProximalBundleMethod.qp_solver` has dead code after its `return`.
- **D-134(4) and (5)** are cited but do not exist: D-134 numbers (1) to
  (3), then has an unnumbered "Underneath" paragraph (the solver zoo
  unifying). The "honest home for diagnostics" has no text in D-134.
- **D-82's question as framed here is stale.** `auto` already prefers
  bisection over the bundle on quadratic sets, and KKT over the dual on
  differentiable non-quadratic ones (`_choose`, backus.py ~2796). The live
  question was KKT against bisection; see the settled Q3 below.
- **"Reproduce `_choose` exactly"** would reproduce a crash (next section)
  and today's preference for routes that are dense in the data dimension.
- **The dual gap certificate** (D-134 mentions it) is deferred to Mag by
  D-113; assumed out of scope.
- **Example 23 uses neither `route=` nor `membership=`.** The re-spelling
  falls on `test_backus`, `test_quadratic_programming`,
  `test_parallel_knobs`, `test_inference` and `test_convex`.

### A bug found, not fixed

An ellipsoid prior with exact data crashes at construction:

```python
I = LinearOperator.identity(M)
BackusGilbertParker(LinearForwardProblem(A), T, Ellipsoid(M, I * 0.25, covariance=I * 4.0))
# ValueError: The noise needs a positive radius for this route.
```

`_choose("auto")` takes `bisection` whenever both sets are quadratic,
without excluding exact data, while a forced `bisection` refuses exact
data and `_Quadratic` refuses a ball of radius zero. `_choose_membership`
has the same mismatch for `reduced`. With one `accepts` per entry, used by
`auto` and by forcing alike, that form falls through to the dual for
support and has no membership. Left for the redesign rather than patched.

### Settled with David

**Q1, what an algorithm is.** David: "on a philosophical level each
algorithm is something we could apply to any convex set … methods that
work on a given convex set that has the right properties exposed … the
current implementations take in a problem, that was a choice and it can
change." Agreed shape:

- An algorithm is a public object holding **tuning only**, with a `name`,
  an `accepts(set)` and probe methods that take the set:
  `Bisection(max_iterations=80).support(feasible, q)` is
  `feasible.support(q, via=Bisection(max_iterations=80))`. Diagnostics are
  ordinary methods, `Dual().dual_cost(feasible, q)`,
  `ClosedForm().budget(feasible)`, replacing the `algorithm` back door.
- The set **exposes its structure**, `T(prior ∩ A⁻¹(d − noise))`:
  `forward_operator`, `property_operator`, `prior`, `noise`, `data`, and
  `accepts` is a predicate on those. A future `Image`/`Intersection`/
  `Preimage` composite exposing the same structure is accepted unchanged,
  which is D-134(3)'s incremental adoption.
- The engines stop taking a `LinearForwardProblem`; they take the
  operators and the sets.
- **Expensive data-independent work is stored with the set's
  data-independent part**, keyed by the algorithm's settings by value
  (two `Bisection(max_iterations=80)` share). The estimator owns one such
  part and hands it to every set it returns, so `estimator(d1)` and
  `estimator(d2)` share as today. `feasible.with_data(d2)` would give a
  directly built set the same sharing; only if wanted.
- **Names are bare:** `ClosedForm`, `Bisection`, `Dual`, `Primal`, `KKT`,
  `Smoothed` on the support side, `Reduced`, `Likelihood` on the
  membership side (`ClosedForm` in both tables; `Bisection` and `Reduced`
  share one engine). David: "the suggestions are fine. At most namespace
  via the convex module." Layering obstacle: neither `geometry.convex` nor
  `numerics.convex` may import `inference` (D-111), and the bisection's
  two-ball fit and the dual's feasibility test use inference's
  `misfit_search` over a `TikhonovFamily`. The fallback was to define them
  in `backus.py` and export them from `pygeoinf2.inference`; moving that
  search down a layer so the algorithms can live with the sets, as D-134
  intends, is for the design pass.

**Q2, where `auto`'s choice shows.** `feasible.route` and
`feasible.membership` stay as read-only names of what `auto` picks,
derived from the table; `BackusGilbertParker` loses `route`, `membership`
and `algorithm`. Capability flags derive from "some entry accepts".

**Matrix-free first.** David: "Matrix-free methods should be the starting
point, and so part of the initial work. Dense methods provide a special
case. Broadly, we need to provide such methods with a linear solver, and
potentially a preconditioner or preconditioning family. What is supplied
determines how things are done internally." And: "a chunk of the original
work on this aspect leant too much into dense methods. Again, don't rule
them out, but don't assume them." This overrides the first binding rule
for the dense routes: they stay, numerics untouched, as special cases and
as the parity references for the new matrix-free paths.

**Q3, ranking.** David: "I would go with scaling, but this is a question
that can be played with later. And in real applications users will likely
pick directly or measure — auto is convenience." Theoretical scaling
(even rough) outranks timings on test problems. The reading proposed, not
contested: KKT below bisection, which is D-82's verdict (same system,
weaker root find; the dense `PrimalKKTSolver` is worse by scaling and has
the 1e-3 floor). Once bisection is matrix-free it and the dual are the
same order in applications, tens of warm-started Krylov solves against
hundreds of bundle iterations, so the tie is broken by what each gives —
bisection an extremal model and an exact root, the dual a certificate —
and bisection stays first on quadratic sets.

**No Woodbury baked in.** David: Woodbury is used "only in quite specific
cases", for example "a preconditioner based on a surrogate model with a
model space of vastly reduced dimension". Instead the algorithms that
solve the damped family copy the `formalism=` flag, "allowing the user to
pick a preference, with the default being you solve in whichever of the
two is smaller".

### Proposed, not yet confirmed

- **The solver decides the internals**, through machinery that exists:
  the damped family is solved by `DampedSolves`, which warm-starts along a
  root find and keeps a preconditioner until the multiplier moves past
  `refresh` (factor 10), then rebuilds it — `refresh=inf` freezes it,
  `refresh=1` recomputes it. The preconditioner rides on the solver,
  `CGSolver(...).with_preconditioner(P)`, as everywhere else. An iterative
  solver gives the matrix-free path; a `DirectSolver` a dense
  factorization per multiplier (the two matrices extracted once); an
  `EigenSolver` the spectral reduction, which is today's code. Default:
  CG, unpreconditioned. Each algorithm takes `solver=`; the set's and the
  estimator's `solver=` is the default for any algorithm given none; the
  model-space minimizations (likelihood, `LevelKKTSolver`) keep
  `optimizer=`. David's Woodbury refinement answered part of this; the
  rest was not explicitly confirmed.
- **What each formalism needs from the sets** is the reverse of the
  Gaussian case. Model space, `(B + τ G* V G) m = …`, needs the
  precisions, which a ball has trivially and an `Ellipsoid` always has.
  Data space, `(G C G* + τ⁻¹ V⁻¹) w = …`, needs the covariances, and an
  `Ellipsoid`'s is optional: without it every application is an inner
  solve, as `_Quadratic` does today. Exact data has no model-space
  formalism (`NormalOperator`'s rule), and the closed form and its
  membership are minimum-norm solves, data-space by nature (`A* A` is
  singular). So `formalism=` applies to bisection, reduced membership,
  `PrimalKKTSolver`, and the fitting and `is_empty` search on quadratic
  sets.
- **Reusing the Gaussian normal operator.** The family has its shape with
  the sets' shapes in place of `Q` and `R`, which would bring the
  formalism handling and the structure-aware preconditioners. But
  `NormalOperator` takes `GaussianMeasure`s, so the reuse is not
  straightforward; look before deciding.
- **Assumptions stated early and not discussed:** the result classes keep
  the names `BundleResult`, `KKTResult`, `SaddlePointResult` as
  `OptimizationResult` subclasses (no aliases to kill), `gradient_norm`
  NaN where a method has none; probe kinds support, maximizer,
  certificate, membership, feasibility, with `is_empty` in today's order
  closed_form, reduced, likelihood, dual; `via=` on every probe that
  dispatches, not only the six listed in stage 2.3; `support_values`'
  tuning keywords (`tolerance`, `max_iterations`, `epsilon`, `optimizer`)
  move into instances while `warm_start=` and `n_jobs=` stay on the probe;
  the direct constructor is `FeasiblePropertySet(forward_operator,
  property_operator, prior, data, /, *, noise=None, solver=None)` with
  `noise=None` meaning exact data, and `.estimator` goes.

### Open

- **The `formalism=` default against D-74.** `choose_formalism`'s
  docstring says `"auto"` is "not the default anywhere" (D-57, D-74,
  D-78), for two reasons: the model-space route needs a precision a
  function-space prior often lacks — reversed here, sets carry their
  precision — and comparing dimensions reads a discretization's size as
  the problem's, so a refinement can switch the algebra — which still
  applies. Either `"auto"` is the default here, recorded as an exception
  to D-74, or `"data_space"` is, as elsewhere. David asked for the smaller
  before the D-74 point was raised; unanswered.
- **Q4, the maximizing solvers' result.** KKT and Chambolle-Pock maximize
  `(c, m)`. Under one result shape they inherit `minimizer` and `value`,
  holding the maximizer and the maximum. Recommended: document that and
  drop their `.model`; the alternative keeps `.model` as a second name.
- **Where the algorithms live** (the layering obstacle under Q1).
- **How `DampedSolves` serves two multipliers** — moot if the reduction
  below holds, since it leaves a one-parameter family.

### The dense inventory

With `n = dim(D)` and `p = dim(P)`:

| piece | dense work in the data space | per direction or test |
|---|---|---|
| bisection support and extremal model | `n` applications of `A C A*`, `O(n³)` eigendecomposition | about 3600 nested bisection steps, each `O(n²)` |
| reduced membership | the same, on the projected Gram | one `O(n²)` projection, then about 60 steps of `O(n)` |
| closed-form membership | joint spectrum of `(A; T)`: `n + p` applications, `O((n+p)³)` | `O((n+p)²)` |
| `PrimalKKTSolver` | two `n × n` matrices | an `O(n³)` solve per root-find evaluation, tens of them |
| bisection with ellipsoid noise | Cholesky of the noise precision matrix | — |
| fitting and `is_empty`, ellipsoid on either side | the spectral reduction | — |
| `LevelBundleMethod` | master QP in `n + 1` components | intrinsic to the method: stays a special case, never `auto` |

Already matrix-free: closed-form support (CG on `A A*`), the dual,
smoothed and primal routes, `LevelKKTSolver`, likelihood membership, and
fitting and `is_empty` for two balls (`misfit_search`, D-81).
CURRENT_STATE §5 needs correcting in stage 3: it says `LevelKKTSolver`
forms data-space matrices, which it does not, and it omits the
bisection's, the reduced membership's and the closed form's spectra.

### The matrix-free reduction of the quadratic routes

KKT stationarity is `(λ B + μ G* V G) m = c + λ B m0 + μ G* V d`; the
bisection's `(s Q + t A* P A) u = T* q + t A* P d'` is the same system
with `λ = s`, `μ = t`. The two differ only in the root find on the pair:
`fsolve` in log coordinates against nested monotone bisections in
`γ = s/t` and `t`. Dividing by `λ`,

```
(B + τ G* V G) m = c/λ + B m0 + τ G* V d,      τ = μ/λ = 1/γ
```

so the operator depends on `τ` alone — a Tikhonov family, in data-space
form `G C G* + τ⁻¹ V⁻¹` — and `λ` enters only the right-hand side. At
fixed `τ` the model is affine in `1/λ`: one solve against `c` and one
against `B m0 + τ G* V d` give it for every `λ`, and the prior constraint
is a quadratic in `1/λ` with a closed-form root. What remains is a
one-dimensional root find in `τ` at two warm-started solves per probe,
against about 3600 dense steps per direction today. Unverified: that the
misfit is monotone in `τ` once `λ` is eliminated, and that a positive
root in `1/λ` exists at every `τ` probed (it need not when the data part
alone exceeds the prior). If monotonicity fails, nested root finds with
warm starts remain.

The same machinery serves the rest: `DampedSolves`' docstring already
names "the two Backus multipliers" as a client, `TikhonovFamily` and
`misfit_search` are built on it, and the structure-aware preconditioners
in `inference/preconditioners.py` read their factors off the family's
members. Closed-form membership becomes a minimum-norm solve for
`(A; T)` by CG or LSQR, infinite when the residual cannot reach tolerance
(today an exact spectral cut at 1e-13, so the iterative test is a
tolerance judgement). Reduced membership is the same damped search inside
`ker T`. Ellipsoid fitting is `misfit_search` weighted by the ellipsoid.

### Suggested next step

A design pass with library-wide context before any code: where the
algorithms live, the normal-operator reuse, the formalism default, the
reduction above checked, then the stages rewritten. Stage 0 is still
routine but is best left to the same pass, since a matrix-free KKT may
change what its result carries.
