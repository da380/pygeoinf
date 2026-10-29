# Review: the inference layer

Scope: everything under `pygeoinf2/inference/` (`problem`, `estimators`,
`point`, `gaussian` with `LowRankGaussianPosterior`, `laplace`,
`minimization`, `backus`, `mixture`) and its re-exports at the top level.
Every claim marked "verified" was run; the scripts are in
`/tmp/claude-1000/-home-david-dev-pygeoinf/4bec2589-8482-4f4c-843c-b27f331a8905/scratchpad/inference_review/`
(`inf_t1.py` … `inf_t23.py`, helper `infcommon.py`), all on the tests'
`make_dense_metric_space` model and data spaces unless stated otherwise.
The six inference test files pass (408 tests, 15 s).

## Must fix before 2.0

### Three linear estimators ignore the data error's expectation
`inference/gaussian.py:1008`, `inference/point.py:950-957`,
`inference/backus.py:546-564` — The library supports a noise measure with
a nonzero expectation, and `LeastSquares`, `MinimumNorm`,
`ConstrainedMinimumNorm`, `LinearGaussianInversion`, `MaximumAPosteriori`,
`NonlinearLeastSquares`, `Occam` and `potential` all subtract it. Three
estimators do not:
- `LowRankGaussianPosterior.from_linear_problem` forms the residual as
  `d - A m_0`, without `- mu_e`. Its mean is wrong, and so is its
  evidence, because `J` is evaluated at that mean.
- `ConstrainedLeastSquares` builds its offset from the inner
  `LeastSquares.operator` and throws away the inner estimator's
  `translation` (`-B mu_e`).
- `BackusGilbert` is `X d` with no translation. Its budget then
  misses a term: `E[X d] - T m` is `bias(m) + X mu_e`, not `bias(m)`.

These are wrong answers, and nothing reports them. Verified.
Reproduction: `inf_t1.py`: at full rank (4 data), with the noise mean
nonzero, the low-rank mean differs from `LinearGaussianInversion`'s by
1.4e-2 relative, and the normalizer is -17.63 against -16.56. With a
zero-mean noise the difference is 1e-12. `inf_t13.py` compares each
estimator built on a biased noise `N(mu, R)` with data `A m + mu` against
the same estimator built on `N(0, R)` with data `A m`:
```
LeastSquares             4.09e-15
ConstrainedLeastSquares  1.59e-01
ConstrainedMinimumNorm   8.29e-17
BackusGilbert            1.50e-01
```
Proposed fix:
- In `from_linear_problem`, subtract `error.expectation` from the residual.
- In `ConstrainedLeastSquares`, add `projector(inner.translation)` to the
  offset.
- Give `BackusGilbert` the translation `-X mu_e`, built lazily as
  `LinearGaussianInversion` builds its own.
- Add one parametrized test that runs every linear estimator under a biased
  noise.

### `LaplaceApproximation` always uses the Gauss-Newton curvature, so `gauss_newton=` cannot change it
`inference/laplace.py:481-495`, `inference/laplace.py:99-103` — D-180 says
that the estimators' `hessian` methods use the objective's Hessian
whenever it carries one, "and the derivative with respect to the data and
the Laplace approximation follow it". `MaximumAPosteriori`'s docstring for
`gauss_newton=` says the same: "It is then also the Hessian … behind the
Laplace approximation". The code disagrees. `LaplaceApproximation.at`
inverts `estimator.linearized_normal(model)`, which is always
`P + J* R^-1 J`. So:
- with a forward that supplies its second derivative, the Laplace
  covariance is not the inverse of the objective's Hessian, although the
  MAP's own derivative is built from that Hessian;
- `gauss_newton=True` changes nothing in the Laplace result.

Verified on an overdetermined problem (3 parameters, 10 data, a quadratic
forward whose first and second derivatives pass `check_derivative` and
`check_second_derivative` to 1e-10). The low-rank route builds its
eigenpairs from `J` alone, so it is Gauss-Newton by construction too.
Reproduction: `inf_t2.py`:
```
Laplace cov vs inverse of exact Hessian/2: 0.0995   (relative)
Laplace cov vs inverse of Gauss-Newton normal: 5.5e-16
MAP derivative check: worst=9.35e-09   (uses the exact Hessian)
```
Proposed fix: make the full route take its covariance from
`estimator.hessian(data, model) / 2`, which is positive definite at a
strict minimizer and is what D-180 records. Its precision is that same
operator. If Gauss-Newton is the intended Laplace curvature, correct D-180
and the `gauss_newton` docstring instead, and say on the class that the
curvature is Gauss-Newton whatever the forward supplies.

### An exact-data posterior carries a unit-noise density
`inference/problem.py:240-306` (line 290), `inference/gaussian.py:530` —
`ForwardProblem.potential` on a problem with no error measure returns
`|d - F(m)|^2 / 2`, which is the likelihood of noise with identity
covariance on the data space. `LinearGaussianInversion.__call__` puts
that potential into the `GaussianPosterior` as `relative_log_density`. The
Gaussian moments, however, are the exact conditioning on `A m == d`. The
class docstring says that for a linear problem "the two readings describe
the same measure exactly", and here they do not. An exact-data posterior
is singular with respect to the prior and has no density at all. A
sampler given this object as its target would sample the unit-noise
posterior. D-154 keeps exact data to unit tests, which limits the reach,
but the object is still wrong. Verified.
Reproduction: `inf_t19.py`. The posterior mean fits the data to 3.9e-15.
`relative_log_density` equals `-potential` of the unit-noise problem to
3e-45. The mean the density implies differs from the Gaussian's by
9.8e-3 relative.
Proposed fix:
- `potential` raises `ValueError` without an error measure: exact data
  have no likelihood.
- `LinearGaussianInversion.__call__` returns a plain `GaussianMeasure` (or
  a `GaussianPosterior` with no density) for exact data.
- `potential`'s docstring says so.

### The error budget's two operators are reached two different ways
`inference/estimators.py:199-266` — D-163 defines the point estimator's
budget as two operators, `resolution` and `propagated_covariance`.
`resolution` is a property and `propagated_covariance()` is a method.
Neither takes an argument, and both return a `LinearOperator`. For
comparison, `BackusGilbert.unresolved` and `GaussianEstimator.covariance`
are properties. The mini-apps already write both forms side by side, for
example `pushed.propagated_covariance()` next to
`chosen.push_forward(factor).resolution(...)` in
`miniapps/point_data/point.py:104-128`. Only an API change can align
them, so it has to happen now or never. If you see this as a free choice,
drop it.
Reproduction: read only.
Proposed fix: make `propagated_covariance` a property, and update its
callers: five apps and tutorials 7 and 14.

## Should fix

### `chi_squared` bypasses the function-space gate on an argument that fails for observed data
`inference/problem.py:194-199` — D-131 lets `chi_squared` pass
`unsafe=True` on the grounds that a misfit "is defined whenever the
residual lies in the error's Cameron-Martin space". This contradicts that
decision, for the following reason. The residual of observed data contains
a draw of the noise, and on a data space that discretizes a function space
that draw almost surely lies outside the Cameron-Martin space. A
likelihood ratio needs only `F(m)` to lie there, and its value carries an
`m`-independent constant that diverges. So `chi_squared`,
`chi_squared_test` and `potential` give numbers that grow without bound
under refinement. Meanwhile `consistency_set`, on the same problem, refuses
through `credible_set`. Verified on a Sobolev circle observed whole
(`A = I`, noise `sobolev_measure(2, 0.2)`, trace class).
Reproduction: `inf_t8.py`, `inf_t17.py`:
```
dim   16: Phi(truth) =   6.75   critical chi2 =  26.3   (noise.log_density refuses)
dim   64: Phi(truth) =  30.97   critical chi2 =  83.7
dim  256: Phi(truth) = 132.54   critical chi2 = 294.3
chi_squared_test: True;  consistency_set: ValueError (credible set refused)
```
Proposed fix: pass `unsafe=True` only when
`not data_space.is_function_space`. On a function-space data space,
refuse `chi_squared`, `potential` and `critical_chi_squared` unless
`unsafe=True` is given, the same pattern as the density family. Correct
D-131's sentence to match.

### The model-space evidence volume diverges under refinement on the stochastic route
`inference/gaussian.py:686-705` — In the model-space formalism,
`normal_log_determinant` adds three log-determinants: `log|N_m|`, `log|Q|`
and `log|R|`. On a function-space model, `log|Q|` goes to minus infinity
and `log|N_m|` to plus infinity while their sum stays finite. On the
stochastic route each term carries a standard error proportional to its
own size, so the sum's error grows with the discretization. The mixture
docstring recommends exactly this route (`evidence={"method":
"stochastic"}`) for keeping the weights matrix-free. Verified on a Sobolev
circle with 12 point data.
Reproduction: `inf_t22.py`, 30 probes:
```
dim  64: data_space -7.63 +/- 0.83 (dense -8.09)   model_space -35.25 +/- 17.86
dim 256: data_space -7.63 +/- 0.83                 model_space -39.03 +/- 45.82
```
Proposed fix: in the model-space formalism, take the volume as
`log det(I + L* A* R^-1 A L)` through the prior's covariance factor. This
operator is identity plus trace class, its log-determinant is bounded
under refinement, and `LowRankGaussianPosterior` already builds it.
Without a factor, refuse the stochastic route there or say in the
docstring that it is not dimension robust.

### `LaplaceApproximation`'s two routes carry different things, and the docstring describes only one
`inference/laplace.py:321-329`, `inference/laplace.py:481-495`,
`inference/laplace.py:376-409` — The full route returns a Gaussian with no
sampler and a normalizer of `None`. The `rank=` route returns one that
samples and carries the Laplace evidence. The class docstring says "The
normalizer is `None`; a Laplace estimate of the evidence is a separate
claim this does not make", which D-198's `rank=` route contradicts. The
same docstring calls the result "the pair a sampler starts from", yet the
full-route Gaussian cannot be drawn from. Randomize-then-optimize at the
mode, which needs a prior draw, a noise draw and the linearized gain, is
exactly what `LinearGaussianInversion` already does.
`LaplaceApproximation.from_problem` also takes no `gauss_newton=`, so the
waveform app builds the MAP by hand (`miniapps/waveform/IMPLEMENTATION.md:293`).
Verified.
Reproduction: `inf_t14.py`:
```
LinearGaussianInversion posterior can_sample: True
Laplace (full) can_sample: False    sample() -> NotImplementedError
Laplace (rank) can_sample: True
normalizer full: None   rank: -10.7982   linear: -10.7982
```
Proposed fix:
- Give the full route the randomize-then-optimize sampler at the mode,
  with the linearized gain through the estimator's solver.
- Either give the full route the Laplace evidence (one `log_determinant`)
  or state the difference in the docstring.
- Add `gauss_newton=` to `from_problem`.

### The linear point estimators fix the norm and the reference, and every mini-app works around it
`inference/point.py:74-135`, `inference/point.py:412-508` — `LeastSquares`
and `MinimumNorm` minimize `|Au - d|^2_R + t|u|^2`: the model space's own
norm, about zero. Five apps want the size in the prior's norm, and each
rebuilds the same pattern by hand: parameterize by `L`, push forward
through `properties @ L`, map back with `factor(...)`, and whiten the
truth for `bias`. They whiten in four different ways:
- `factor.inverse` in `miniapps/point_data/point.py:69`;
- `factor.adjoint @ prior.precision` in `miniapps/mfem/box_inverse.py:117`;
- `prior.precision_factor` in `miniapps/cpp_backend/point.py:36`;
- a `whiten` helper in `miniapps/tomography/point.py:99`.

CURRENT_STATE §7(5) records the symptom: a measure with no precision
factor breaks the pattern. The cause is that the estimator has no knob.
Planet density also subtracts the reference's prediction from the data by
hand, because neither class takes a `reference=`, while
`NonlinearLeastSquares` and `Occam` do. The layer already has the
weighting: `BackusGilbert(problem, I, covariance=Q, alpha=1)` equals the
Kalman operator exactly.
Reproduction: `inf_t18.py`: `BackusGilbert(I, covariance=Q, alpha=1)`
against `kalman_operator` gives a relative difference of 0.0, and
`BackusGilbert(I, alpha=t)` against `LeastSquares(damping=t)` gives 0.0.
The workaround sites are listed above (read).
Proposed fix: give `LeastSquares`, `MinimumNorm` and their constrained
forms `covariance=` (`C`, applied and never inverted in the data-space
formalism, which minimizes `t (u - u0, C^-1 (u - u0))`) and `reference=`
(`u0`). The Tikhonov family needs `C`, as the feasible set's damped
family already has. `Occam` would want the same `covariance=` for its
norm.

### `ConstrainedMinimumNorm` and `ConstrainedLeastSquares` lag their unconstrained forms, and lose settings
`inference/point.py:1148-1264`, `inference/point.py:912-1074` —
`ConstrainedMinimumNorm.parameterized` and `.data_reduced` rebuild with
the class defaults for `max_iterations` and `rtol`, because those are
never stored. That bug is verified. The class also lacks:
- `atol` and `minimum_damping`. The reduced family `A P` is rank deficient
  by construction, which is D-174's case for a floor.
- `with_solver`, `with_formalism`, `with_level`, `level`, `for_data`,
  `estimator_at` and the search. D-164 says it "follows" `MinimumNorm`.

Without `for_data` there is no affine estimator, and so no error budget.
`ConstrainedLeastSquares` lacks `forward_problem` and `with_damping`.
Reproduction: `inf_t3.py`:
```
inner settings: 7 0.01
after parameterized: 60 1e-06
after data_reduced: 60 1e-06
ConstrainedMinimumNorm ['data_reduced', 'forward_problem', 'parameterized', 'reduced', 'subspace']
MinimumNorm ['data_reduced', 'discrepancy_search', 'estimator_at', 'family', 'for_data', 'formalism', ...]
```
Proposed fix: store every setting and rebuild through one `_rebuild`, as
`MinimumNorm` does. Give the constrained classes the unconstrained
surface; `for_data` returns a `ConstrainedLeastSquares` at the found
damping.

### `LeastSquares` solves at construction, and assembles with a direct solver
`inference/point.py:122-127`, `inference/point.py:955-957` — To decide
whether the noise mean shifts the estimate, `LeastSquares.__init__`
applies the estimator to `error.expectation`. With a direct solver that
application is the factorization, paid even for a zero-mean error. It is
paid again by every `with_damping`, `with_formalism`, `parameterized`,
`data_reduced` and `ConstrainedLeastSquares`, which solves for its offset
at construction as well. `LinearGaussianInversion` defers the same solve
for this reason, and its comment says so. Verified by counting
applications (40 parameters, 30 data, zero-mean error).
Reproduction: `inf_t10.py`:
```
CG:       LeastSquares {'fwd': 0, 'adj': 1}    LinearGaussianInversion {'fwd': 1, 'adj': 0}
Cholesky: LeastSquares {'fwd': 30, 'adj': 31}  LinearGaussianInversion {'fwd': 1, 'adj': 0}
```
Proposed fix: pass the translation as a thunk (`AffineOperator` accepts
one), and skip it when `error.has_zero_expectation`.

### `Minimization.push_forward` returns a private copy of `ComposedPointEstimator`
`inference/minimization.py:302-359`; also `inference/laplace.py:530-575`
and `inference/mixture.py:183-233` — D-164 says `ComposedPointEstimator`
is what `push_forward` returns for an estimator that is not affine in the
data. `Minimization` (and with it `MaximumAPosteriori`,
`NonlinearLeastSquares` and `Occam`) returns `_PushedPointEstimator`
instead. That class duplicates the logic, has no `estimator` or
`property_operator`, and is not an instance of the public class. On the
measure side, `_PushedMeasureEstimator` and `_PushedMixture` are two more
private copies, and `_PushedMixture` does not check the operator's domain
while the other two do. Verified.
Reproduction: `inf_t9.py`:
`type(MaximumAPosteriori(...).push_forward(T))` is
`_PushedPointEstimator`, and its `isinstance(..., ComposedPointEstimator)`
is `False`. Its derivative is correct (`check_derivative` 2e-9).
Proposed fix: let `ComposedPointEstimator.__call__` forward keyword
arguments such as `start=`, and return it from `Minimization.push_forward`.
Fold the two measure-side wrappers into one checked class.

### `log_evidence` drops the standard error of a stochastic estimate
`inference/gaussian.py:707-729` — `normal_log_determinant` returns an
`Estimate`, and the posterior's `log_normalizer` is an `Estimate`.
`evidence_terms` and `log_evidence`, which answer the same question, return
floats even on the stochastic route. The mixture weights are then computed
from numbers whose uncertainty has been discarded. Verified.
Reproduction: `inf_t12.py`:
`log_evidence(data, method="stochastic", samples=20)` returns a `float64`
of -64.17, while `normal_log_determinant` in the same call is
`Estimate(40.14 +/- 4.86)`.
Proposed fix: return an `Estimate` from `log_evidence` (and its volume
from `evidence_terms`), as D-51 asks of anything stochastic.

### `harden_error` is public, only the tests call it, and it disagrees with what the estimator does
`inference/backus.py:53-108` — `harden_error` is in `backus.__all__`. No
library code calls it; only `tests/test_backus.py` does. Separately,
`BackusGilbertParker` resolves its confidence set with `_confidence_set`,
which accepts any convex set, while `harden_error` refuses every set that
is not a ball. The `ambient_ball` docstring
(`probability/gaussian.py:1863-1865`) says `harden_error` is what "every
Backus route whose error is a Gaussian" calls, which is false. Verified:
`harden_error` refuses an `Ellipsoid` error that `BackusGilbertParker`
accepts (`inf_t11.py`).
Proposed fix: delete `harden_error` and its tests, or make
`_confidence_set` call it so that there is one behavior. Fix the
`ambient_ball` sentence either way.

### The mixture's per-component loops take no `n_jobs=`
`inference/mixture.py:89-94`, `inference/mixture.py:142-147`,
`inference/mixture.py:178-181`, `inference/mixture.py:223-229` — Each of
the `K` components is an independent solve and log-determinant, and none
of the loops takes `n_jobs=`, against the principle. The same applies,
more mildly, to `LowRankGaussianPosterior.modes` and
`pointwise_variance_reduction` (`gaussian.py:1044-1068`), one factor
application per kept direction. Read only.
Proposed fix: add `n_jobs=` to the mixture inversion and run the
component loops through `parallel_map`.

## Could fix

### The same fault, two error classes
`inference/point.py:1101-1107` against `inference/point.py:1302-1307` — A
subspace without an explicit equation raises `NotImplementedError` from
`parameterized` and `ValueError` from `constraint_value_mapping`. Read
only.
Proposed fix: use one class for both, `ValueError` with the rebuild hint.

### A dead private method
`inference/point.py:1055-1059` — `ConstrainedLeastSquares._parameterized_subspace`
has no caller; `parameterized` calls the module function directly.
Proposed fix: delete it.

### Docstrings that misdescribe the code
Read; the first one verified (`inf_t5.py`).
- `problem.py:526-529`: `data_reduced` says a set error raises
  `ValueError`; it raises `TypeError`.
- `backus.py:242`: "the credible ball at `level`" is `ambient_ball`, the
  ball in the space's norm.
- `estimators.py:85, 89, 395`: "target space" where the API says
  `property_space`.
- `gaussian.py:625`: "Densely when the space is small enough to afford
  it" misstates `log_determinant`'s `auto` rule, which counts
  applications.
- `laplace.py:141-142`: the refusal suggests building the prior "from
  standard deviations", which is refused on function spaces.
- `laplace.py:326`: a stray backtick ends the role.
- `problem.py:308-325`: `consistency_set` has no `Raises` section (a set
  error, or a function-space data space).

Proposed fix: correct the wording in each.

### Export lists
`inference/gaussian.py:57`, `inference/__init__.py:20-26` —
- `LowRankGaussianPosterior` is missing from `gaussian.__all__`.
- `inference` re-exports `NormalDiagonalPreconditioner`,
  `LocalizedPreconditioner` and `InvariantDistancePreconditioner` but not
  `WoodburyPreconditioner`. That is the structure-aware preconditioner the
  apps use, and CURRENT_STATE §2 says the structure-aware ones are
  re-exported.
- The top level exports `ComposedPointEstimator` but none of its bases
  (`PointEstimator`, `LinearPointEstimator`, …).

Read only.
Proposed fix: make the lists consistent.

### `MinimumNorm` evaluates the zero model's misfit twice per call
`inference/point.py:791` and `inference/point.py:364` — `_resolve` checks
`chi_squared(0, d)` and then passes the same value again as `limit=`
inside `_discrepancy_search`. That is one wasted forward application per
call. Read only.
Proposed fix: compute it once and pass it down.

### The parity test D-163 asks for is missing
No test compares `MinimumNorm` with the feasible set's `fitting_model`,
which D-163 says "must agree in tests". They do agree to 1.3e-11 when the
noise set is the credible ellipsoid (`inf_t23.py`). With the default
ambient-ball hardening (D-82) they differ by 19% on the same problem. That
is worth one sentence in `BackusGilbertParker`'s docstring, since a reader
would expect the two "fits" to coincide.
Proposed fix: add the parity test with `noise=error.credible_set(level)`.

### `alpha` and `damping` name one quantity
`BackusGilbert(problem, I, alpha=t)` is `LeastSquares(problem, damping=t)`
exactly (`inf_t18.py`, difference 0.0). D-165 chose `alpha` after the
literature. Raised only so the overlap is visible; the decision stands
unless you want one name.

### A test reaches a private attribute
`tests/test_inference.py:311-323` calls `estimator._centered_sample`. A
public route exists: `estimator(data).sample(rng=)` less the mean.
Proposed fix: use the public route.

## What is sound

Exercised on dense-metric model and data spaces, with a nonzero noise
mean wherever it matters:
- `ForwardProblem.potential` and the MAP objective pass `check_gradient`
  (1e-8, 6e-10).
- `LinearGaussianInversion` in both formalisms:
  - `check_affine` on the expectation operator, and `check_traits` on
    the posterior covariance;
  - `check_measure` on the randomize-then-optimize sampler;
  - `log_evidence` against a dense reference to 2e-14;
  - the posterior's `log_normalizer` agrees with `log_evidence` up to the
    likelihood's constant, exactly;
  - the property posterior agrees with the Laplace one on a linear
    problem to 1e-14.
- `MinimumNorm`'s derivative matches finite differences to 1e-10 in both
  formalisms, and its adjoint passes `check_operator` (2e-15).
  `for_data(d)(d)` equals `MinimumNorm(d)`, including the saturated case.
- `ConstrainedMinimumNorm`'s derivative and `constraint_value_mapping`
  match finite differences (1e-9, 1e-6).
- `Occam` matches `MinimumNorm` on a linear problem to 6e-12.
  `NonlinearLeastSquares` matches `LeastSquares` to 7e-16, and its
  derivative matches the `LeastSquares` operator.
- The MAP derivative with an exact Hessian passes `check_derivative`.
- `LowRankGaussianPosterior` at full rank, with zero-mean noise, matches
  the full posterior: covariance to 1e-15, precision to 5e-13, normalizer
  exactly. It passes `check_measure` and `check_traits`.
- `LinearizedFeasiblePropertySet` gives the plain set's bracket for a
  linear problem linearized anywhere, and so do its `push_forward` and
  `with_data`.
- `BackusGilbert` refuses a non-positive `alpha`, passes `check_affine`,
  and its propagated covariance passes `check_traits`.
- Refusals by type for set-valued errors are consistent (`TypeError`
  throughout) and name the alternative.
- No decision numbers, history or planning talk appear in the layer's
  docstrings or comments.

## Fixed

Package D, 2026-10-29. Every test named below fails on the code before
the fix and passes after; the new ones are built on dense-metric model and
data spaces or on a Sobolev circle.

### Must fix

**Three linear estimators ignore the data error's expectation.**
`inference/gaussian.py` `LowRankGaussianPosterior.from_linear_problem`
subtracts `error.expectation` from the residual.
`inference/point.py` `ConstrainedLeastSquares._attach` builds the offset
`c - P B (A c + mu_e)`, formed on first use. `inference/backus.py`
`BackusGilbert._attach` gives the estimate the translation `-X mu_e`, formed
on first use and carried by `push_forward`. Tests:
`test_inference.py::TestANoiseExpectationIsRemoved::test_it_answers_as_its_zero_mean_twin`
(eight estimators, each against its zero-mean twin on shifted data, to
`1e-12`; three of the eight fail before), `::test_the_bias_is_the_resolution_less_the_truth`,
`test_low_rank_posterior.py::TestAgainstTheFullPosterior::test_a_noise_expectation_is_removed`,
`test_backus.py::TestBackusGilbert::test_a_noise_expectation_is_removed_and_pushed_forward`.

**The low-rank linear mean keeps the prior in the dropped directions.**
`inference/gaussian.py` `from_linear_problem` forms
`m_0 + L V diag(1/(1+lambda)) V* L* A* R^-1 (d - A m_0 - mu_e)`; the class
docstring and the method's say that at a truncated rank the mean omits the
update in the dropped directions, and that the update is the full one
projected in the prior's Cameron-Martin norm. Tests:
`test_low_rank_posterior.py::TestAgainstTheFullPosterior::test_at_full_rank_it_is_the_posterior`
(existing, `1e-8`) and `::test_below_full_rank_the_mean_omits_the_dropped_update`
(rank 2 of 6 with dropped eigenvalues above 100: the norm stays below twice
the full mean's, and the Cameron-Martin norm below the full one's; the old
formula gave 150 to 320 times the full mean's norm).

**`LaplaceApproximation` always uses the Gauss-Newton curvature.**
`inference/laplace.py` `LaplaceApproximation.at`: the curvature is half the
MAP objective's Hessian, as the MAP's derivative uses it. When the
objective carries an exact Hessian that differs from Gauss-Newton (a
nonlinear forward operator with a second derivative, `gauss_newton=False`)
the covariance inverts `MaximumAPosteriori.hessian(data, model) / 2`;
otherwise it inverts `linearized_normal(model)` as before, which keeps the
normal operator's factors for structure-aware preconditioners.
`linearized_normal` itself keeps its meaning, the Gauss-Newton normal
operator (it takes no data, and the exact Hessian needs them), and its
docstring now says it is the Laplace curvature only when the Hessian is
Gauss-Newton. `from_problem` gains `gauss_newton=`. The `rank=` route
(`_low_rank`) takes the eigenpairs of the exact misfit Hessian when asked
(`_information_eigenpairs` now takes any misfit Hessian); the choice made
for its possible indefiniteness is to **clip** negative eigenvalues at
zero, so a direction in which the operator's curvature would widen the
prior keeps the prior, and the class docstring says so. Tests:
`test_laplace.py::TestTheCurvatureFollowsTheEstimator`
(`test_the_covariance_inverts_the_estimators_hessian` and
`test_the_low_rank_route_takes_the_same_curvature`, each for
`gauss_newton` False and True, on the reviewer's quadratic forward with a
dense model metric: covariance against the exact-Hessian inverse or the
Gauss-Newton inverse to `1e-10` and `1e-8`, and at least `1e-2` from the
other).

**An exact-data posterior carries a unit-noise density.** The refusal:
`inference/problem.py` `ForwardProblem.potential` raises `ValueError`
without an error measure, and `inference/gaussian.py`
`LinearGaussianInversion.__call__` returns a plain `GaussianMeasure` (no
reference, density or normalizer) for exact data. No test depended on the
density. Tests: `test_inference.py::TestExactDataHaveNoDensity`.

**`resolution` a property, `propagated_covariance()` a method.** Both
methods now: `inference/estimators.py` `LinearPointEstimator.resolution()`,
and with it `inference/backus.py` `BackusGilbert.unresolved()`, the third
operator of the same budget (an extension of the decision, flagged for the
caller). `bias` is no longer written through `resolution` (see below).
Callers updated: `tests/test_inference.py`, `test_point.py`,
`test_backus.py`, `test_no_array_backend.py`, `tutorials/07_properties.py`
and its regenerated notebook, `miniapps/point_data/point.py`,
`miniapps/tomography/point.py` and the two apps' `IMPLEMENTATION.md`.
Tests: `test_inference.py::TestPointEstimators::test_the_resolution_operator_is_available`,
`test_backus.py::TestBackusGilbert` (four tests).

### Should fix

**`ConstrainedMinimumNorm` and `ConstrainedLeastSquares` lag their
unconstrained forms.** `inference/point.py`: `ConstrainedMinimumNorm` takes
and stores `atol=` and `minimum_damping=` beside `max_iterations` and
`rtol`, rebuilds through one `_rebuild`, and gains `level`, `solver`,
`formalism`, `with_solver`, `with_formalism`, `with_level`,
`discrepancy_search`, `search`, `for_data` and `estimator_at`, the last two
returning a `ConstrainedLeastSquares` (built around the reduced family's
member, so no second factorization) with the error budget of one; its
linearization runs one search, not two. `ConstrainedLeastSquares` gains
`forward_problem`, `with_damping` and a `_rebuild`; its dead
`_parameterized_subspace` method is gone. Tests:
`test_point.py::TestConstrained::test_the_settings_survive_every_rebuild`,
`::test_for_data_gives_the_affine_estimator_with_its_budget`,
`test_inference.py::TestConstrainedLeastSquares::test_it_reaches_its_problem_and_rebuilds_at_another_damping`.

**`LeastSquares` solves at construction.** `inference/point.py`
`LeastSquares.__init__` passes its offset `-B mu_e` as a callable, and none
at all when the error is a Gaussian built without an expectation;
`LinearPointEstimator.push_forward` keeps an unevaluated offset
unevaluated. Test: `test_point.py::TestConstructionAppliesNothing`
(Cholesky, with and without a noise mean: zero forward and adjoint
applications through construction and `with_damping`).

**`Minimization.push_forward` returns a private copy.**
`inference/minimization.py`: it returns `ComposedPointEstimator`, whose call
now forwards keyword arguments such as `start=`; `_PushedPointEstimator` is
deleted. The two measure-side wrappers are folded into one checked
`_PushedMeasureEstimator` in `inference/estimators.py`, used by
`LaplaceApproximation.push_forward` and
`LinearGaussianMixtureInversion.push_forward`, which now refuses an
operator on the wrong space. Tests:
`test_minimization.py::TestMinimization::test_push_forward_is_the_public_composition`,
`test_mixture.py::TestMixtureInversion::test_a_property_operator_on_another_space_is_refused`.

**`log_evidence` drops the standard error.** `inference/gaussian.py`:
`log_evidence` returns an `Estimate` (zero error on the dense route) and
`evidence_terms` returns the volume as one; `inference/mixture.py`
`log_evidence` returns an `Estimate` whose error combines the components'
in quadrature weighted by the posterior weights, while
`log_evidence_terms` still returns the values that the weights are built
from. Callers updated: `tutorials/06_linear_forward_problem.py` and its
regenerated notebook, `examples/26_mixture.py`,
`miniapps/planet_density/src/answers.py`, `miniapps/planet_density/wide.py`,
and the tests. Tests: `test_inference.py::TestEvidence::test_it_matches_scipy`,
`test_mixture.py::TestMixtureInversion::test_a_stochastic_evidence_carries_its_error_and_ignores_the_job_count`.

**`harden_error` is public and only the tests call it.** Deleted from
`inference/backus.py`, with its export; `_confidence_set`'s docstring says
what it does with a measure. Its two tests now test the confidence set
`BackusGilbertParker` resolves. Test:
`test_backus.py::TestHardeningTheError` (three tests).

**The mixture's loops take no `n_jobs=`.** `inference/mixture.py`
`LinearGaussianMixtureInversion(n_jobs=)`: each component's posterior and
evidence run through `parallel_map` in one pass, the volume terms a worker
computes are carried back to the component's inversion, and a generator in
`evidence=` is spawned into one child per component so that a parallel run
gives a serial run's numbers. `LowRankGaussianPosterior.modes(n_jobs=)`
and `pointwise_variance_reduction(n_jobs=)` apply the factor as one block.
Tests: `test_mixture.py::TestMixtureInversion::test_the_answer_does_not_depend_on_the_job_count`,
`test_low_rank_posterior.py::TestAgainstTheFullPosterior::test_the_modes_do_not_depend_on_the_job_count`.

**The full-rank Laplace Gaussian has no sampler and no evidence.**
`inference/laplace.py`: the Gauss-Newton Gaussian samples by
randomize-then-optimize with the linearized gain through the solver
(`_linearized_sampler`); the exact-Hessian one has no such route and says
so. Both carry the Laplace evidence `-J(m) - log det(I + H) / 2` as a lazy
normalizer through `log_determinant`, in the data space
(`log |F' Q F'* + R| - log |R|`) for Gauss-Newton and by Sylvester's
identity (`log |K| + log |Q|`) for the exact Hessian. The class docstring
is rewritten for both routes. Tests:
`test_laplace.py::TestTheFullGaussianSamplesAndCarriesTheEvidence` (on a
linear problem the normalizer is the linear posterior's to `1e-8`;
`check_measure` on the draws), `test_laplace.py::TestTheCurvatureFollowsTheEstimator::test_the_evidence_is_the_laplace_estimate`
(against a dense reference to `1e-10`, the full and low-rank routes
agreeing); `test_laplace.py` and `test_absolutely_continuous.py` no longer
assert a `None` normalizer, and `test_no_array_backend.py` now samples.

**Decision 2, the consistency set.** `inference/problem.py`
`consistency_set` passes `unsafe=True` to `credible_set`, as the misfit
does, so it is the chi-squared test's sublevel set on every data space;
`chi_squared`, `potential` and `consistency_set` say that a limit under
refinement needs the error measure to be trace class. Test:
`test_inference.py::TestTheConsistencySetIsTheMisfitsSublevelSet`.

### Could fix

- The dead `ConstrainedLeastSquares._parameterized_subspace` (above).
- Docstrings: `data_reduced`'s set error is `TypeError`; `BackusGilbertParker`
  names the ambient ball and says when its fitting model is
  `MinimumNorm`'s; "target space" is "property space"; the
  `normal_log_determinant` default counts applications; the Laplace
  refusal no longer suggests standard deviations; the stray backtick;
  `consistency_set` has a `Raises` section.
- `LowRankGaussianPosterior` is in `gaussian.__all__`;
  `WoodburyPreconditioner` is re-exported from `pygeoinf2.inference`.
- `MinimumNorm` evaluates the zero model's misfit once per call.
- The parity test: `test_point.py::TestMinimumNorm::test_it_is_the_feasible_sets_fitting_model`
  (with the credible ellipsoid, to `1e-6`, the feasible route's tolerance).
- `test_inference.py` reaches the sampler through `posterior.sample`, not
  `_centered_sample`.

### Found on the way

- `MinimumNorm.estimator_at` (and so `for_data`) returned a `LeastSquares`
  whose recorded solver was the family's member solver at that damping, so
  every rebuild (`with_damping`, `parameterized`, `data_reduced`, `family`)
  kept answering at the first damping: `estimator_at(1.0).with_damping(5.0)`
  was 9% from `LeastSquares(damping=5.0)`. `LeastSquares._member` keeps the
  member for the estimator's own inverse and records the family's solver
  for rebuilds. Test: `test_point.py::TestMinimumNorm::test_an_estimator_at_a_damping_rebuilds_at_another`.
- `bias()` was `(B A - T) m`, which is the average error only when the
  translation is `-B mu_e`: for `ConstrainedLeastSquares` on a subspace off
  the origin it was a hundred times the true bias. It is now
  `B (A m + mu_e) + t - T m` for every affine estimator, the same value for
  the unconstrained ones. Test:
  `test_inference.py::TestConstrainedLeastSquares::test_the_bias_is_the_average_error_on_an_affine_subspace`.
- `test_minimization.py` used `gi.testing` without importing
  `pygeoinf2.testing`, and two of its tests failed when the file ran alone.

## Skipped

- **`covariance=` and `reference=` on `LeastSquares` and `MinimumNorm`.**
  `covariance=` needs the Tikhonov normal operator and family
  (`numerics/tikhonov.py`, outside this package) to carry a model-space
  weighting: `A C A* + t R` in the data space, and `C^-1`, which a
  covariance without a precision cannot give, in the model space; and
  `MinimumNorm`'s closed-form derivative rederived for the weighted norm.
  That is a design change, not a local fix. `reference=` alone is local
  (`u_0 + B (d - A u_0 - mu_e)`, and `u_0 + MinimumNorm(d - A u_0)` for the
  searched estimator) but leaves an API question open (a model-space
  reference has no meaning after `parameterized`), and by itself removes
  none of the apps' whitening; left for a decision with `covariance=`.
- **`chi_squared` bypasses the function-space gate.** Decision 2 keeps the
  bypass; only the consistency set and the docstrings changed (above).
- **The model-space evidence volume diverges under refinement.** Decision 2;
  the wording on the model-space formalism is package A's.
- **Could fix: one fault, two error classes** (`NotImplementedError` from
  `parameterized`, `ValueError` from `constraint_value_mapping`): changing an
  exception class is an API change the tests pin, left as it is.
- **Could fix: `alpha` and `damping`.** The decision stands.
- **Could fix: the top level exports `ComposedPointEstimator` but none of
  its bases.** `pygeoinf2/__init__.py` is outside this package.
- Not asked, noted: `BackusGilbert` with a ball or ellipsoid error ignores
  the set's center, as before; only a measure's expectation is removed.
