# Review of the geometry layer

Scope: `pygeoinf2/geometry/` (`sets.py`, `convex.py`, `subspaces.py`,
`feasible.py`, `__init__.py`) and what the top level re-exports from it.
Every "verified" finding was run against the tree at `4b27d35`; the scripts
are in
`/tmp/claude-1000/-home-david-dev-pygeoinf/4bec2589-8482-4f4c-843c-b27f331a8905/scratchpad/`
(each imports `common.py` from there, which builds the operators on the
tests' `make_dense_metric_space` / `WeightedSpace`). Run them with the
poetry environment's interpreter from that directory.

## Must fix before 2.0

### The default outer ellipsoid is not certified when the data outnumber the model
`pygeoinf2/geometry/feasible.py:2142` — `SProcedure` in its default
`formalism="data_space"` builds each column of `T C_θ T*` as
`T((1+γ)/γ (C c − C A* K(γ)⁻¹ A C c))`. When the data outnumber the
model coefficients and the prior is slack, the weight search walks to small
`γ`, where `K(γ) = A C A* + γR` is singular to within `γ` and the
subtraction cancels; the trace or volume it minimizes is then evaluated
wrongly, and the "certified" ellipsoid it returns does not contain the set.
`F.outer_ellipsoid()` with no arguments takes this route on every quadratic
set. The model-space formalism on the same problems is right to `1e-13`.
`Primal` avoids the same cancellation through the resolvent identity and a
floor on the walk (D-153, D-174); `SProcedure` has neither. The class
docstring (`feasible.py:2014`) and CURRENT_STATE §4 say the ellipsoid "is
checked against support values direction by direction"; nothing in the code
checks it. Verified by running; the mechanism (cancellation and an
ill-conditioned family) is my reading of why.
Reproduction: `e8b.py` (model dense dim 3, data dim 8, `Ball(radius=100)`
prior, noise ball 0.5): the set's own default bracket in `e0` is
`(0.43887, 0.43951)`, the default ellipsoid's support `0.41090`; a contact
point of the set has Mahalanobis² `59.2` in that ellipsoid. `e8c.py`
(dims 20 and 40, eight directions, worst `(h_E − attained)/|attained|`):
prior radius 3 → `−1.8e-3`, radius 10 → `−2.4e+4`, radius 30 → `−0.74`;
`SProcedure(formalism="model_space")` → at worst `−4e-9`.
Proposed fix: evaluate the columns and the residual through the resolvent
identity with the kernel part taken at a floor multiplier, stop the walk at
that floor on a rank-deficient family as `Primal` does, and before returning
compare the ellipsoid's support with the set's in a few directions, raising
or inflating when it falls short, so the docstring's claim becomes true.

### A flat closed-form set gives a wrong projection and a wrong ellipsoid
`pygeoinf2/geometry/feasible.py:653` — when the kernel of `A` has fewer
dimensions than the property, the closed form's shape `S` is singular (the
set is a segment or a point). `ClosedForm.inclusion` handles this with a
least-squares solve and says so, but `ClosedForm.ellipsoid` inverts `S`
with `np.linalg.inv`, producing a precision with an eigenvalue of `−1.2e16`
under a `POSITIVE_DEFINITE` claim. `FeasiblePropertySet.ellipsoid`,
`project`, `outer_ellipsoid()` (returned as certified) and
`level_function().gradient` (`np.linalg.solve` at `feasible.py:817`) all
read it. `project` returns a point that is not in the set, because the
indefinite form calls it "inside". Also at zero budget the covariance is the
zero matrix with a `POSITIVE_DEFINITE` claim. Verified by running.
Reproduction: `e2_flat.py` (model dense dim 4, data dim 3, property dim 2,
exact data, `Ball(radius=4)`): covariance eigenvalues `[−1e-16, 2.195]`,
precision eigenvalues `[−1.2e16, 0]`; a point `1e-3` off the segment:
`F.contains → False`, `F.ellipsoid.contains → True`; `F.project(0) →
[0, 0]`, where the nearest point of the set is `[−0.386, 0.325]`.
Proposed fix: test the rank of `S` once (the test `inclusion` already
makes) and, for a flat set, refuse `ellipsoid`, `project` and the closed
form's `outer_ellipsoid` with a `ValueError` that says the set is flat, and
take `level_gradient` through the same least-squares solve.

### A zero-radius noise ball with a centre loses its centre
`pygeoinf2/geometry/feasible.py:254` — `_is_point` (`:205`) treats
`Ball(radius=0, center=b)` as exact data and the structure stores
`noise = None`, so `shifted_data` never subtracts `b`: the set is computed
for the data `d` rather than `d − b`, silently. A known, exact offset in
the data (a datum, a static correction) is exactly what such a set says.
Verified by running.
Reproduction: `e20_pointnoise.py`: data `d = A m + b`;
`FeasiblePropertySet(A, T, Ball(model, radius=3), d, noise=Ball(data,
radius=0, center=b)).support(e0) → 3.1861`, while the same set built on
`d − b` with exact data gives `3.2273`.
Proposed fix: when the noise collapses to a point, fold its centre into the
data (keep `b` on the structure and subtract it in `shifted_data` and in
the closed form's fit), or refuse a zero-radius ball with a nonzero centre.

### `outside()` proves points of the set to be outside it when the support is negative
`pygeoinf2/geometry/feasible.py:3531` and `pygeoinf2/geometry/convex.py:468`
— the test is `pairing > h * (1 + rtol) + rtol`. For `h < 0` the
relative term moves the threshold below `h`, so a contact point, which
attains `h` exactly, is reported outside, and the docstring says a `True`
"is a proof". Verified by running.
Reproduction: `e1_outside.py`: on a closed-form set whose support in `−e0`
is `−8.311`, the support maximizer is admitted by `F.contains` and
`F.outside(x, [q]) → True`; on `ConvexSet.from_support_function` of a unit
ball about `(−10, 0)`, the point `(−9, 0)` is in the ball and
`outside(p, [e0]) → True`.
Proposed fix: `pairing > h + rtol * max(abs(h), scale)`, with the scale
named (or an `atol=`), in both places.

### The dual's emptiness answer misreads a prior whose level is not positive
`pygeoinf2/geometry/feasible.py:2593` — `Dual.feasibility` reports a
stand-in level of `0.0` for a non-empty set, and `is_empty`
(`feasible.py:3602`) compares it with the prior's own level. A prior
written with a negative level (`SublevelSet(f, level=-1)`, a `HalfSpace`
with a negative offset) is then declared empty, `fitting_model()` raises and
`extent` with it. The same comparison, `level * (1 + rtol)`, tightens rather
than loosens for a negative level in `is_empty`, `admits` and
`Search._general_fit`. `Dual.feasibility` also turns any `ValueError` from
the support solve into "no model fits". Verified by running.
Reproduction: `e19_level.py`: the unit ball written as
`SublevelSet(‖m‖² − 2, level=−1)` with a support-only noise set (form
`general`, feasibility by `dual`): `is_empty() → True`; the same ball
written with level `+1`: `is_empty() → False`.
Proposed fix: report `feasible.level` (as `Distance` does) for a bounded
dual, catch only the unboundedness the dual raises, and write the
tolerance as `level + rtol * max(abs(level), 1)`.

### The dual declares a set empty by an absolute threshold in property units
`pygeoinf2/geometry/feasible.py:2536` — a non-converged dual whose value is
below `−_UNBOUNDED = −1e6` raises "the feasible set is empty". The value is
in the units of the property, so a property of size `1e7` and a dual that
merely ran out of budget (CURRENT_STATE §7 records a dual that does not
converge in 1 500 iterations) gives a false certificate of emptiness, and
`is_empty(via="dual")` returns `True`. Verified by running.
Reproduction: `e4_dual_units.py`, `e4b.py`: the same problem scaled by
`1e7` (support `−1.38e6`): `Dual(max_iterations=3)` raises "no model lies
both inside the prior set and within the noise set"; `F.is_empty() → False`,
`F.is_empty(via=Dual(max_iterations=3)) → True`; at unit scale the same
budget returns `−0.137` unconverged.
Proposed fix: make the threshold relative to the problem's own scale (the
prior-only bound `|h_prior(T* q)|`, or the first value of the dual cost), or
use the divergence test D-149 describes rather than a level.

## Should fix

### Degenerate priors fail on every support algorithm, with messages that point elsewhere
`pygeoinf2/geometry/feasible.py:405` and `:1767` — D-138 admits a
semidefinite `Ellipsoid` prior and D-147 promises a data-space refusal
that names `formalism="model_space"`. Instead `_covariance_of` inverts the
precision under a false `POSITIVE_DEFINITE` claim, and: the default probes
fail with "a tighter solver, a preconditioner, or the dense spectral solver
may be needed"; `Primal(formalism="model_space")` fails too, because
`Primal.support` (and `SProcedure.outer_ellipsoid`, `:2076`) first call
`feasible.is_empty()` with the default data-space `Search` and the set's
default solver, ignoring the instance's formalism and solver; `Splitting`
fails because `_dual_cost` inverts the precision for its outer value. With
the emptiness check answered by a model-space search, the model-space primal
is right. The same detour means `Primal(solver=SpectralFamilySolver())`
still runs CG first. (The dual's affine constraint is §7 and not reported.)
Verified by running.
Reproduction: `e16_degenerate.py` (model dim 5, the Gram with one direction
removed, `A` seeing that direction): default `is_empty`, default support,
and `Primal(formalism="model_space")` raise `ValueError: The damped family
could not be solved at its first multiplier: CG met a non-positive curvature
direction`; `Splitting` raises `ConvergenceError`; with the feasibility
entry seeded from `Search(formalism="model_space")` the model-space primal
gives `(12.4573784, 12.4573895)` against `12.4573895` from SLSQP.
Proposed fix: have `_covariance_of` refuse an unbounded ellipsoid by name,
pointing at `formalism="model_space"`; run the emptiness check inside
`Primal` and `SProcedure` with a `Search` of the same formalism and solver.

### A parallel sweep's results stay in the workers
`pygeoinf2/geometry/feasible.py:3416` — `support_sweep(n_jobs=k)` returns
the results but does not put them in the parent's cache, so a following
`polytope(directions)`, `support_bracket` or `extremal_model` on the same
directions solves them again; and each worker starts from an empty
structure cache (`:264`, whose comment calls it "this process's warm
starts"), so the data-independent pieces D-145 assigns to the structure
(the closed form's shape, the kernel Gram, the family) are rebuilt per
worker. The point-data mini-app's part E does exactly sweep-then-polytope
and pays the sweep twice. Verified by running (the per-worker rebuild by
reading).
Reproduction: `e6_sweep.py`: after `support_sweep(dirs, n_jobs=2)`,
`polytope(dirs)` costs 2 178 forward applications in the parent; a second
`polytope(dirs)` costs 0.
Proposed fix: write the returned results into the parent's cache under the
keys `support_result` uses, and ship the picklable data-independent pieces
with the structure rather than dropping the whole cache.

### Independent loops without `n_jobs=`
`pygeoinf2/geometry/feasible.py:3746`, `:3693`, `:2138`, `:585` —
`inner_hull` (one support solve or one inclusion search per point),
`extent` (two independent ends, each up to ~100 searches), `outer_ellipsoid`
(the `dim P` solves per weight in `SProcedure`), the closed form's `dim P`
shape solves and `outside` are serial. D-151 names the first three. Read
only.
Reproduction: the signatures: `inner_hull(directions_or_values, /, *,
source, via)`, `extent(direction, /, *, max_iterations, via)`,
`outer_ellipsoid(*, criterion, via)`.
Proposed fix: add `n_jobs=` through `parallel_map` as `support_sweep` does.

### The quadratic tier's level function has no gradient
`pygeoinf2/geometry/feasible.py:1099` — D-137 says the search's multiplier
"is the gradient of the feasible set's level function in `p`, which
`level_function()` carries", and D-143 lists "`level_function()` with
gradient". `Search` implements no `level_gradient`, so on the quadratic
tier, the main one, `level_function()` has no derivative and
`has_smooth_level_function` is `False`; only the closed form has one.
Verified by running.
Reproduction: `e15_levelgrad.py`: form `quadratic`, `level_function()
.has_derivative → False`, while `inclusion_result(p).multiplier → 47.2`.
Proposed fix: implement `Search.level_gradient` from the fit (the
multiplier on `T m = p`, one `dim P` solve with the kernel Gram it already
holds), or amend D-137 and D-143.

### `via=` reaches probes the algorithm does not answer
`pygeoinf2/geometry/feasible.py:3717`, `:3617` — `extent(q, via=...)`
passes `via` to `fitting_model`, a feasibility probe, as well as to
`admits`, so an inclusion-only algorithm is refused; and
`fitting_model(via="dual")` returns `None` silently, where
`extremal_model` raises `NotImplementedError` for an algorithm that
exhibits no model. Verified by running.
Reproduction: `e5_via.py`: `F.admits(T(truth), via="distance") → True`;
`F.extent(q, via="distance") → ValueError: Unknown algorithm 'distance'
for 'feasibility'`; `F.fitting_model(via="dual") → None`.
Proposed fix: `extent` uses `via` for `admits` only; `fitting_model`
raises `NotImplementedError` when the result carries no model.

### Membership tolerances are not scale-free, though every docstring says so
`pygeoinf2/geometry/convex.py:1635`, `:1745`, `:139` — `HalfSpace`,
`Hyperplane`, `Polytope` and the default `ConvexSet.contains` (so the
subspaces) floor their scale at `1.0`; their docstrings say "Scaled by the
set's own size, so it means the same thing whatever the units are". Below
unit scale the tolerance becomes absolute; `Polytope.contains` is what the
mini-apps use to check the truth against the brackets. Verified by running.
Reproduction: `e3_units.py`: the half-space `x ≤ s` and the point `2s`:
at `s = 1e-9`, `HalfSpace`, `Polytope`, `Hyperplane` and `LinearSubspace`
all return `True`; at `s = 1e-6` and `1`, `False`.
Proposed fix: drop the floor of one (use `|offset|` and `|x| |n|` alone,
with an explicit `atol=` for the origin case), or correct the docstrings.

### `Polytope & other` breaks unless the other is a polytope
`pygeoinf2/geometry/convex.py:1148` — `Polytope.__and__` assumes its
argument has `is_outer`, so `poly & ball` and `poly & halfspace` raise
`AttributeError`, while `ball & poly` and `poly.intersect(ball)` return a
`ConvexIntersection`. Verified by running.
Reproduction: `e18_and.py`: `poly & Ball(X) → AttributeError: 'Ball'
object has no attribute 'is_outer'`.
Proposed fix: return `NotImplemented`/`super().__and__(other)` for a
non-polytope, and append a `HalfSpace` to the constraints.

### The algorithms are not exported where a caller looks for them
`pygeoinf2/geometry/__init__.py:16` — D-143 makes an algorithm instance the
way to tune a probe, and CURRENT_STATE §2 lists `ClosedForm`, `Primal`,
`Search`, `Dual`, `Splitting`, `Distance`, `SProcedure` as living in
`geometry`; but `geometry` exports `ClosedForm` (which has no tuning) and
`LinearCertificate` and none of the others, which the tests and the
point-data app import from `geometry.feasible`. `feasible.__all__` omits
`SProcedure` and `EllipsoidResult` (returned by `outer_ellipsoid_result`),
while `geometry.__all__` has `EllipsoidResult`. The top level re-exports
`LinearCertificate` alone of the algorithms. Verified by running.
Reproduction: `hasattr(pygeoinf2.geometry, "Primal") → False`;
`"SProcedure" in pygeoinf2.geometry.feasible.__all__ → False`.
Proposed fix: export every algorithm and result type from
`geometry.feasible.__all__` and `geometry`, and state one rule for which of
them the top level carries.

### `Ellipsoid` does not check its covariance
`pygeoinf2/geometry/convex.py:1882` — the constructor checks the
precision's and the factor's spaces but takes a covariance on any space, or
one that is not the precision's inverse, and the support function, the
maximizer and every algorithm reading `prior_covariance()` then use it.
D-195 added a one-probe check of `B B⁻¹ x == x` for `ambient_ball`'s
`inverse=`; the class itself has none. Verified by running.
Reproduction: `e14_refusals.py`: `Ellipsoid(P, I, covariance=identity on
EuclideanSpace(3))` is accepted; `Ellipsoid(P, I, covariance=4 I)` is
accepted and its support in `e0` is `2.0` where the set's is `1`.
Proposed fix: refuse a covariance not on the domain, and apply one probe of
`P C x == x` as `ambient_ball` does.

### Docstrings that misstate the code
`pygeoinf2/geometry/feasible.py:3163` — `FeasiblePropertySet`'s `solver`
argument says "Conjugate gradients at `rtol=1e-12` by default"; the default
is `rtol=1e-10` (`_default_solver`, and `solver_for`'s own docstring).
`pygeoinf2/geometry/convex.py:2495` — `ConvexIntersection.support_function`
sends the caller to "the inference.backus routes for the true value": the
routes are gone and the feasible set is in `geometry`. Verified by reading
and, for the first, by `_default_solver(EuclideanSpace(3)).rtol → 1e-10`.
Reproduction: as stated.
Proposed fix: correct both.

## Could fix

### A semidefinite ellipsoid's Hessian claims to be invertible
`pygeoinf2/geometry/convex.py:1993` — `Ellipsoid.level_function()`'s
Hessian is `(2P).with_traits(POSITIVE_DEFINITE)` whatever `P` claims, so a
semidefinite ellipsoid's carries `POSITIVE_DEFINITE|INVERTIBLE`, and the
`precision` argument is documented as "a positive definite operator"
(`:1851`) though semidefinite ones are admitted. Verified by running.
Reproduction: `e11_semidef.py`: `check_traits(E.level_function()
.hessian(0))` fails "the claimed INVERTIBLE operator has full rank. rank 3
of 4"; the conversion along the flat direction also emits an overflow
`RuntimeWarning` on the way to `inf`.
Proposed fix: carry the precision's traits (times two) into the Hessian;
say "positive semidefinite" in the argument.

### `ClosedForm.prior_only` is dead while five apps write it by hand
`pygeoinf2/geometry/feasible.py:677` — no caller in the library, tests,
examples or apps. Meanwhile the point-data, tomography, MFEM and C++
backend apps (and planet density) each define a `prior_alone` helper for
"what the prior set alone admits along `T* q`". Read only.
Reproduction: `grep -rn prior_only pygeoinf2` finds only the definition.
Proposed fix: delete it, or expose the prior-only bound as a probe of the
set and let the apps use it.

### The C++ backend app works around a radius the ellipsoid now carries
`pygeoinf2/miniapps/cpp_backend/sets.py:68` — `radius_of` recovers the
radius from `(B u, u) / (P u, u)`, and its IMPLEMENTATION.md says the
ellipsoid carries "its precision `B / r^2` and its covariance, not `r`";
D-195 gave `Ellipsoid.radius` for exactly this. Read only.
Reproduction: as cited.
Proposed fix: use `prior_set.radius` and drop the helper and the sentence.

### `AffineSubspace` has no support function
`pygeoinf2/geometry/subspaces.py:217` — `Hyperplane` has one (finite only
along its normal), but an affine subspace, including one built
`from_hyperplanes`, has none, though the module docstring says the
subspaces inherit "a support function where one exists". Read only.
Reproduction: `LinearSubspace.from_basis(X, [v]).support_function()` raises
`NotImplementedError`.
Proposed fix: `(t, q)` when `q` lies in the complement, `+inf` otherwise,
with the projector deciding.

### Cache keys that ignore part of an instance's tuning
`pygeoinf2/geometry/feasible.py:2646`, `:1814` — `Splitting.settings`
omits `rng`, so instances with different generators share cached
results; `Primal`'s `u₂` cache is keyed by the formalism alone, so
instances with different solvers share solutions at equal `γ`, against
D-145's "solved to the tolerance in force". Read only; I found no case
where it changed an answer.
Proposed fix: include `id(rng)` in the settings; key the `u₂` cache by the
solver as the family already is.

### A `log(0)` warning from the primal's warm start
`pygeoinf2/geometry/feasible.py:1853` — after a model-space walk reaches
`γ = 0` the cache holds the key `0.0`, and the next direction's
nearest-multiplier search takes `np.log(0)`, which reaches the user as
`RuntimeWarning: divide by zero encountered in log`. Verified by running
(`e8d.py`).
Proposed fix: skip a zero key in the nearest-multiplier search.

### Thread wording left after the move to processes
`pygeoinf2/geometry/feasible.py:1910`, `:264` — the primal's comment says
"a sweep hands each thread a contiguous run", and the warm starts are keyed
by `threading.get_ident()`, after D-195 made the sweep's workers processes;
the structure's `__getstate__` comment calls its cache warm starts. The
point-data app's comments ("before the threads share them", "worker
threads") say the same. Read only.
Proposed fix: reword; the keys can stay.

### `inclusion_level` calls an infinite level a proof
`pygeoinf2/geometry/feasible.py:3632` — "infinite when no model has the
value and fits the data, which is a proof of exclusion"; on a
rank-deficient family the search stops at its floor and reports infinity
for a boundary point of the set, which `Search`'s own docstring calls
"exact to the order of the floor". Verified by running.
Reproduction: `e8b.py`: the contact point of the set in `e0` (model dim 3,
data dim 8) has inclusion level `inf`, "no model fits the data"; the point
`0.1%` inside it has level `4.485` against `10 000`.
Proposed fix: qualify the sentence by the floor.

### Small docstring and naming points
`pygeoinf2/geometry/sets.py:96` says a feasible property set's level
function is "the minimum norm of a model with that property"; it is the
prior's level function at the smallest fitting model with that property.
`pygeoinf2/geometry/convex.py:219` ("The Backus-Gilbert-Parker engines use
it") describes the retired design; only `Distance` uses it now. Local
identifiers `centre`, `centre_term` (`feasible.py:1778`, `:2126`;
`convex.py:1260`) against D-100. Read only.
Proposed fix: reword; rename.

## What is sound

Exercised by running, on the dense-metric and weighted fixtures:

- `check_projection`, `check_convexity` and `check_gradient` pass on
  `Ball`, `Ellipsoid` (dense metric), `HalfSpace`, `Hyperplane`,
  `Polytope`, `ConvexIntersection`, `AffineSubspace.from_linear_equation`,
  `LinearSubspace.from_basis` and a translated ellipsoid; the kernel
  projector passes `check_traits` and `check_operator`; `dimension()` and
  `to_hyperplanes()` are right.
- `Polytope.inscribed_ellipsoid` lies inside and touches its polytope,
  `enclosing_ellipsoid` encloses its points, and `sum_of_ellipsoids` at
  both criteria contains the Minkowski sum, all on a dense metric.
- The feasible set on a dense-metric property space: the closed form's
  support equals its ellipsoid's, every maximizer has level `1.00000000`,
  the level function's gradient passes `check_gradient`, its projection
  passes `check_projection`; on the quadratic tier the primal (both
  formalisms) and the dual brackets overlap, contact points are admitted,
  and `extent` lies within the brackets.
- `Primal`'s default data-space brackets on problems with twice as many
  data as model coefficients overlap the model-space ones at every prior
  size tried (`e8d.py`); only the outer ellipsoid fails there.
- `LinearCertificate`: its value lies above the certified bracket, `terms`
  sums to it, its ellipsoid contains the S-procedure one, with ellipsoids
  off centre on both sides.
- Dispatch: `form`, `algorithms`, `via=` by name and by instance, refusal
  of an inapplicable or unknown name, the hidden linear certificate; the
  existing suites `test_convex.py`, `test_geometry.py` (216 tests) and
  `test_feasible.py` (90) pass.
- Refusals that work: a negative radius, a property or constraint set on
  the wrong space, `push_forward` with a mismatched operator, the
  membership side on a property space without coordinates.

## Fixed

Fix package C, 2026-10-29. Every test named here fails on the reviewed tree
(`4b27d35`) and passes after; the new ones are in `tests/test_feasible.py`
(feasible sets) and `tests/test_geometry.py` (convex sets), on the
dense-metric and weighted fixtures.

### Must fix

1. **The outer ellipsoid.** `SProcedure` (`feasible.py`) now defaults to
   `formalism="model_space"`; `formalism="data_space"` stays, its argument
   docstring stating where it cancels (data outnumbering the model, a slack
   prior, small `γ`). Both routes build the ellipsoid from the shape's
   eigen-decomposition in the property metric (a precision positive
   definite by construction), refuse a flat or indefinite shape, and check
   it in `SProcedure._checked_budget` against the set's support along the
   longest and the shortest principal axis, both ways, with `Primal` in the
   instance's formalism and solver: a shortfall within `_COVER_RTOL = 1e-6`
   of the half-width is inflated away, a larger one raises a `ValueError`
   naming the model-space formalism. The result's `message` says how many
   directions were checked and by how much it was inflated. The class and
   `FeasiblePropertySet.outer_ellipsoid` docstrings now describe the check
   the code makes. On the reviewer's `e8c` problems the default inflates by
   `1e-13` to `7e-10` and never falls below an attained support; the
   data-space route raises at every prior radius there.
   Tests: `TestTheOuterEllipsoidIsChecked::test_it_never_falls_below_an_attained_support`
   (dims 20 and 40, radii 3, 10, 30); `TestTheOuterEllipsoid::test_it_contains_the_set_and_the_formalisms_agree`
   now compares the default with the data-space route.
2. **A flat closed-form set.** `ClosedForm._build` decomposes the shape `S`
   in the property metric once and marks the set flat when an axis is null
   to `1e-12` of the prior's own `T C T*` (negative included). `ellipsoid`,
   `project` (new `ClosedForm.project`) and the closed form's
   `outer_ellipsoid` raise a `ValueError` saying the set is flat;
   `has_projection` is false for it; `inclusion` and `level_gradient` go
   through one pseudo-inverse (`_pseudo_solve`), the on-the-flat test
   relative to the larger of the value's and the center's norms. A zero
   budget raises "a single point" from `ellipsoid` instead of a zero
   covariance claiming positive definiteness, and `project` returns the
   point. Tests: `TestAFlatSet` (both tests).
3. **A zero-radius noise ball with a center.** `FeasibleStructure.offset`
   keeps the center and `FeasibleStructure.observed` takes it from the data;
   `shifted_data`, the closed form's fit, `_dual_cost`, `Dual.cost` and its
   fit test, `Splitting`, `LinearCertificate` and the general search read
   the data through it, and `push_forward` keeps it. Test:
   `TestAPointNoiseSetOffTheOrigin::test_its_center_is_an_offset_in_the_data`
   (closed form, dual, linear certificate, push-forward against the set on
   `d − b`).
4. **`outside()` at negative support.** `pairing > h + rtol * max(|h|,
   ||q|| ||x||)`, the scale named in the docstrings, in
   `FeasiblePropertySet.outside` and `_OracleSet.outside` (`convex.py`).
   Tests: `TestOutsideWhateverTheSign` (contact points of a set with negative
   support, and a point `1e-3` beyond each), `TestSupportOnlySets` (the unit
   ball about `(−10, 0)`).
5. **The dual's emptiness answer.** `Dual.feasibility` reports
   `feasible.level` with lower bound `−∞` for a bounded dual and catches
   only `_Unbounded`, the private `ValueError` subclass the dual raises for
   a certified empty set; every other failure propagates.
   `FeasiblePropertySet._within_level` writes `level + rtol * max(|level|,
   1)` for `is_empty` and `admits`, and `Search._general_fit` uses the same
   form for the noise level. `fitting_model` raises `NotImplementedError`
   when the algorithm exhibits no model (the dual). Tests:
   `TestTheDualsEmptiness::test_a_prior_written_with_a_negative_level_is_not_empty`,
   `::test_a_failing_support_function_is_raised_not_read_as_empty`,
   `::test_a_noise_set_written_with_a_negative_level_fits_the_same`.
6. **The dual's absolute threshold.** `_UNBOUNDED` is gone. Emptiness is
   decided by a certificate (`_empty_along`): a ray `r` with `(r, d) +
   h_prior(−A* r) + h_noise(−r) < 0` beyond `1e-8` of its terms, which no
   feasible model allows; the rays tried are the first step, and for a run
   at its budget the displacement and the last descent direction. A run
   without one reports itself not converged. A first step that falls to the
   end of its range without a certificate now starts the bundle from where
   it was, where it used to declare the set empty. The class docstring says
   so. Test: `TestTheDualsEmptiness::test_a_short_budget_is_not_converged_whatever_the_units`
   (units `1` and `1e7`, three iterations: not converged and not empty; the
   same problem with far data empty at both scales).

### Should fix

- **Degenerate priors (in part).** `Primal.support` and
  `SProcedure.outer_ellipsoid` decide emptiness, and the primal takes its
  fallback fitting model, with `Search(formalism=…, solver=…)` of their own
  instance, so `Primal(formalism="model_space")` answers on a semidefinite
  prior without seeding (`(12.4573784, 12.4573895)` on the reviewer's
  `e16`, against SLSQP's `12.4573895`), and `Primal(solver=…)` no longer
  runs the set's CG first. On an unbounded prior the S-procedure balances
  its first weight from the precision rather than the covariance, so the
  default outer ellipsoid answers there too. The data-space family and the
  inclusion search's kernel pieces refuse an unbounded ellipsoid prior by
  name, pointing at the model-space formalism (`_require_bounded_prior`),
  where the default probes used to fail inside CG with a message about
  preconditioners. Test: `TestADegeneratePrior`.
- **A parallel sweep's results.** `support_sweep(n_jobs=)` writes the
  workers' results into the parent's cache under `support_result`'s keys
  and sends only the directions the set does not hold. Test:
  `TestSweepsAndWorkers::test_a_parallel_sweep_is_kept_by_the_set`.
- **The ±q duplication** (CURRENT_STATE §7, finding 6). `Primal._kernel_solve`
  keeps `(γ₀, K(γ₀)⁻¹ v, κ₀)` per direction and negates the opposite
  direction's, so a sweep over `±q` makes the floor solve once per pair.
  `FeasiblePropertySet._key` maps `−0.0` to `0.0` so that a direction and
  its negation's negation share a key. Test:
  `TestSweepsAndWorkers::test_a_direction_and_its_opposite_share_the_kernel_solve`.
- **`n_jobs=`** on `inner_hull` (the contact sweep, or the admitted values),
  `extent` (its two ends), `outer_ellipsoid`/`outer_ellipsoid_result` (the
  S-procedure's `dim P` solves per weight and its check's sweep, through a
  new keyword on `Algorithm.outer_ellipsoid`) and `outside`. Test:
  `TestSweepsAndWorkers::test_the_probes_take_workers_and_agree_with_the_serial_ones`.
- **The quadratic tier's level gradient.** `Search.level_gradient`:
  `2 S⁻¹ (p − T a − t T C A* R⁻¹ r)`, `t` the inverse of the search's
  multiplier, one solve with the kernel Gram the search holds; it raises at
  an infinite level. `level_function()` asks the algorithm
  (`_level_gradient_available`) instead of evaluating a gradient at zero.
  Test: `TestTheLevelSideOnQuadraticSets::test_the_level_function_has_the_searchs_gradient`
  (against central differences of a tight search).
- **`via=` forwarding.** `extent` uses `via` for `admits` only and takes its
  base point from the default feasibility algorithm; `fitting_model`
  raises `NotImplementedError` for an algorithm that exhibits no model.
  Test: `TestTheLevelSideOnQuadraticSets::test_via_reaches_only_the_probe_it_answers`.
- **Membership tolerances.** `HalfSpace.contains` and `Hyperplane.contains`
  scale by `max(|offset|, ||normal|| ||x||)`, the default
  `ConvexSet.contains` (so the subspaces) and `_OracleSet.contains` by the
  larger of the point's and its projection's norms, and `Distance` likewise;
  no floor of one, the scale named in each docstring. `Polytope` inherits
  the half-space rule. Test: `TestScaleFreeMembership` (scales `1`, `1e-9`,
  `1e9`).
- **`Polytope & other`.** A non-polytope gives the general intersection, as
  `intersect` does. Test: `TestPolytopeIntersection`.
- **Exports.** `SProcedure` and `EllipsoidResult` join
  `geometry.feasible.__all__`; `geometry` exports every name of it
  (`Distance`, `Dual`, `FeasibleStructure`, `Primal`, `SProcedure`,
  `Search`, `Splitting`, `linear_certificate_operator` added). Test:
  `test_the_algorithms_are_exported_with_the_set`.
- **`Ellipsoid` checks its covariance.** A covariance on another space is
  refused, and one probe (`_check_inverse`: `(P y, y) == (x, y)` for
  `y == C x`, to `1e-6`) refuses one that is not the precision's inverse;
  `translate` carries the checked covariance without probing again. Test:
  `TestTheEllipsoidsOperators::test_a_covariance_must_be_the_precisions_inverse`.
- **Docstrings.** The set's default solver tolerance (`1e-10`);
  `ConvexIntersection.support_function` points at the feasible set.

### Could fix, done on the way

- A semidefinite ellipsoid's Hessian claims what its precision claims, and
  the `precision` argument says semidefinite
  (`TestTheEllipsoidsOperators::test_a_semidefinite_hessian_claims_only_semidefiniteness`).
- The primal's warm start skips a zero multiplier (no `log(0)` warning);
  its `u₂` cache is keyed by the solver; `Splitting.settings` carries the
  generator.
- Thread wording in the primal and the structure; `inclusion_level`
  qualifies "proof" by the floor; `sets.py`'s level-function sentence and
  `from_support_function`'s retired-engine sentence; `centre`, `neighbour`
  and `coloured` spelled American in `feasible.py` and `convex.py`.
- `tests/test_backus.py::TestTheLevelSide::test_a_gaussian_error_taken_as_its_ellipsoid`
  read a flat set's ellipsoid for its center; it now asserts the refusal
  and takes the center from the fitting model.

## Skipped

- **Degenerate priors, the rest.** `_covariance_of` still hands the
  model-space primal a CG inverse of a semidefinite precision claiming
  positive definiteness, which it applies only to vectors in the
  precision's range (the prior-only bound and `_dual_cost`); refusing there
  would break the one route that answers. Writing the model-space primal's
  prior-only bound and dual cost from the precision alone, and `Splitting`'s
  outer value, is a design change.
- **Shipping the structure's data-independent pieces to workers.** Which
  cached pieces cross the process boundary (solvers, families with their
  assembled members, preconditioners) needs a picklability contract and a
  decision on what is worth shipping; the cache is still dropped.
- **`n_jobs=` on the closed form's `dim P` shape solves.** The shape is
  built lazily by whichever probe first needs it, which has no `n_jobs` to
  forward: an argument on the structure or the constructor is an API
  decision.
- **Appending a `HalfSpace` to a polytope's constraints.** Whether the
  result is outer or inner is a design question; `poly & halfspace` gives
  the general intersection instead.
- **The top level's rule for algorithm exports.** `pygeoinf2/__init__.py`
  is outside this package; it still re-exports `LinearCertificate` alone of
  the algorithms.
- **`ClosedForm.prior_only`** (delete, or expose as a probe for the five
  apps' `prior_alone`), **the C++ app's `radius_of`** (outside the package)
  and **`AffineSubspace.support_function`** (a feature): left for a
  decision.
