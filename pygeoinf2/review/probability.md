# Review: probability

Layer: everything under `pygeoinf2/probability/` (`base`, `gaussian`,
`absolutely_continuous`, `mixture`), and the measures the spaces build
through it: `sobolev_measure`, `invariant_measure`, `power_measure`,
`heat_measure`, `correlated_measure` on the symmetric spaces;
`spectral_measure`, `sobolev_measure` and `Layered.sobolev_measure` on the
spectral-element spaces (interval, radial, disk, ball); `from_product` with
and without `correlation=`; `ambient_ball` with and without `operator=`;
`credible_set`; the density family's refusals. Every bug claim below was
reproduced by running it (scripts under the session scratchpad); "read only"
marks the few that were not. The existing suites `test_probability.py`
(138 passed), `test_measures.py`, `test_mixture.py` and
`test_absolutely_continuous.py` (87 passed) are green: none of the defects
below is covered by a test.

## Must fix before 2.0

### Derived samplers are not centred: affine images, translations, sums and products of a sampler-only Gaussian draw about a multiple of the mean
`pygeoinf2/probability/gaussian.py:2701`, `:2773`, `:613` — `GaussianMeasure.sample` adds the expectation to whatever the `sample` callable returns (`:892`), so a supplied sampler must draw the centred fluctuation. Three internal sites hand the constructor an *uncentred* sampler: `_combine_affine` returns `A(self.sample(rng)) + b` (`:2701`), `_combine_add` returns `self.sample(rng) + other.sample(rng)` (`:2773`), and `from_product` returns each factor's full draw (`:613`); each is rebuilt with the new expectation, which `sample` then adds again. Any measure that samples through a callable rather than a factor is hit: a conditioned measure, a `GaussianPosterior` (its sampler is the randomize-then-optimize one), and the sum of any two Gaussians whose covariances do not add to a diagonal one, even when both have factors. So `T @ posterior`, `posterior.translate(v)`, `prior_a + prior_b`, `problem.joint_measure(posterior)` and `data_measure_from_model_measure` on a non-zero-mean prior all sample about the wrong point (twice the mean, three times for `joint_measure(posterior)`), while their `expectation` is right, so nothing but a sample check notices; `testing.check_measure` fails all four. Zero-mean tests hide it. The constructor's docstring for `sample=` (`:254`) does not say the callable must be centred, which is how three sites got it wrong. Verified by running.
Reproduction:
```python
E3, E1 = EuclideanSpace(3), EuclideanSpace(1)
mu = gi.GaussianMeasure.from_covariance_matrix(E3, np.eye(3) + 0.5, form="galerkin", expectation=np.array([10.0, 0, 0]))
A = gi.LinearOperator.from_matrix(E3, E1, np.array([[0, 0, 1.0]]), form="components")
post = mu.condition(A, np.array([5.0]), noise=gi.GaussianMeasure.from_standard_deviation(E1, 0.1))
# expectation vs sample_expectation(4000):
post                         expectation [11.66  1.66  4.97]  sample mean [11.64  1.66  4.97]
post.translate(1)            expectation [12.66  2.66  5.97]  sample mean [25.32  5.34 11.94]
A @ post                     expectation [4.97]               sample mean [9.94]
mu + mu                      expectation [20. 0. 0.]          sample mean [40. 0.02 0.01]
from_product([post, mu])[0]  expectation [11.66  1.66  4.97]  sample mean [23.31  3.3   9.93]
```
and on a circle `Sobolev(32, 1.0, 0.3)` with four point evaluations: for a posterior (`LinearGaussianInversion`, data all 2) `problem.joint_measure(post)` gives a data sample mean of 5.99 against `A(post.expectation)` = 2.0; for an order-2 Sobolev prior with a constant mean of 3, `problem.data_measure_from_model_measure(prior)` has expectation 1.197 at every point and sample mean 2.386 (the sum of the pushed prior and the noise goes through `_combine_add`). `synthetic_model_and_data`, which samples `joint_measure`, is right for a prior with a factor (every Sobolev and invariant measure) and wrong for a posterior, a conditioned prior or a sum.
Proposed fix: make the three samplers return the centred draw (subtract each source measure's expectation, drop the translation), state in the `sample=` docstring that the callable returns a centred draw, and add `check_measure` tests of these four operations on a measure with a non-zero mean and no factor.

### The identity covariance on a function space is refused by two constructors and accepted by seven; tutorial 9 teaches it
`pygeoinf2/probability/gaussian.py:376`, `:258`, `:705`; `pygeoinf2/symmetric_space/base.py:832`, `:1009`, `:1200`; `pygeoinf2/spectral_element/base.py:800` — the refusal D-190 asked for lives only in `sobolev_measure` (order <= d/2) and `from_standard_deviation`. Every other public route builds `sigma^2 I` on a function space without `unsafe=`: `from_standard_deviations` with equal deviations (the plural of the refused constructor), the constructor itself (which even special-cases an identity covariance as its own factor, `:258`), `from_covariance_matrix(eye)`, `invariant_measure(np.ones(dim))`, `spectral_measure(np.ones(dim))`, `correlated_measure` with identity slices, and `power_measure` with a flat `L2` power (on `H^1` that is worse than the identity: the covariance eigenvalues grow with the Sobolev weight, from 1 to 48 across a 32-point circle). This contradicts D-190's scope, which I argue is too narrow: the principle is "refuses otherwise without `unsafe=True`", and while a general spectrum cannot be judged in the limit from a truncation, the literal identity (a constant spectrum, a constant deviation, an `is_identity` covariance or factor) can, cheaply. The gap is not hypothetical: `tutorials/09_nonlinear.py:109` and `:182` build "white noise on the model space: equal variance in every mode of `H^2`" with `invariant_measure(np.ones(X.dim), pointwise_std=0.8)`, and `tests/test_laplace.py:595` copies it; D-190's sweep missed both because nothing refused. Verified by running.
Reproduction (circle `Sobolev(32, 1.0, 0.3)`, `is_function_space` True; `(Cx, x)/(x, x)` on a random `x`):
```
from_standard_deviation:                 refused TypeError
from_standard_deviations(ones):          ACCEPTED, 1.0
GaussianMeasure(covariance=I):           ACCEPTED, 1.0
GaussianMeasure(covariance_factor=I):    ACCEPTED, 1.0
invariant_measure(ones):                 ACCEPTED, 1.0
sobolev_measure(0.0):                    refused ValueError
from_covariance_matrix(eye, components): ACCEPTED, 1.0
correlated_measure(identity slices):     ACCEPTED, 1.0
interval.Sobolev(24,1,0.2).spectral_measure(ones): ACCEPTED, 1.0000000000000009
```
Proposed fix: give `from_standard_deviations` the same `is_function_space` gate and `unsafe=` as the singular form; refuse a constant spectrum (and an `is_identity` or scaled-identity covariance or factor) on a function space without `unsafe=` in `invariant_measure`, `spectral_measure`, `correlated_measure`, `power_measure`, `from_covariance_matrix` and the constructor; move tutorial 9 and `test_laplace.tutorial_nine_problem` to a Sobolev measure with the prior covariance as Newton-CG's preconditioner, the D-168 pattern.

### A correlated product's precision is wrong when a factor's precision factor is not the inverse of its covariance factor
`pygeoinf2/probability/gaussian.py:692` — the correlated precision factor is `S^-1 blockdiag(Li_i)`. That is a square root of `C^-1` only if each `Li_i` is exactly `L_i^-1`; the constructor's contract (`C^-1 == Li* Li`) allows any `Li_i = U_i L_i^-1` with `U_i` unitary, and the cross blocks then carry `U_i* U_j != I`. The library produces such pairs itself: `-1.0 * a` keeps `Li/|alpha|` with factor `-L`, so `U = -I`. The wrong precision then feeds every density, `mahalanobis_squared` and the model-space formalism, and `check_measure` cannot see it (it never touches the precision). Verified by running.
Reproduction:
```python
a, b = (gi.GaussianMeasure.from_covariance_matrix(E3, spd(), form="components") for _ in range(2))
joint = gi.GaussianMeasure.from_product([-1.0 * a, b], correlation=np.array([[1, .6], [.6, 1.]]))
# max |P C v - v| = 2.53
# LinearGaussianInversion on a two-block row operator, same data:
# data_space  [ 0.22   0.337 -0.091 -0.353 -0.601  0.029]
# model_space [-0.346 -0.196 -0.196 -0.462 -0.455 -0.258]
```
The same holds with a hand-built factor pair (Cholesky factor, `Q R^-1` precision factor): `max |P C v - v|` = 3.34.
Proposed fix: build the latent whitening from quantities whose relation is fixed, `L_i^-1 == L_i* P_i` (the factor's adjoint after the precision), so the precision factor is `S^-1 blockdiag(L_i* P_i)` whichever root `Li_i` is; or build it only when `Li_i L_i == I` is known.

### `from_covariance_matrix` silently symmetrizes, so the representation mistake `form=` exists to catch becomes a different covariance
`pygeoinf2/probability/gaussian.py:779` — the docstring promises a `ValueError` for a matrix "not symmetric positive semidefinite in that representation", and `form=` has no default because "getting it wrong is a silent error of one factor of the Gram matrix". But the Galerkin matrix is replaced by `0.5 (M + M^T)` before anything is checked, so a non-symmetric matrix is accepted. The common mistake, a symmetric component matrix `S` on a weighted space passed as `form="components"`, gives the Galerkin matrix `G S`, which is not symmetric (the operator is not self-adjoint and cannot be a covariance), and the measure built is `0.5 (G S + S G)` instead of a refusal. This is the metric mistake the design exists to make hard (CURRENT_STATE §1). Verified by running.
Reproduction:
```python
W = make_weighted_space(); S = spd_matrix(rng, 4)
GaussianMeasure.from_covariance_matrix(W, S + 3 * np.triu(np.ones((4, 4)), 1), form="galerkin")  # accepted; asymmetry 3.0 against entries of 11.5
GaussianMeasure.from_covariance_matrix(W, S, form="components")  # accepted, though G S is not symmetric
```
Proposed fix: refuse when `||M - M^T|| > tol ||M||` (round-off scale, say `1e-8`) with a message naming the likely wrong `form=`, and symmetrize only below that.

## Should fix

### `marginal` drops the sampler and the precision for products, layered priors and posteriors
`pygeoinf2/probability/gaussian.py:2321` (reasoning at `:2356`) — a factor is kept only when the covariance is a `BlockLinearOperator` with a diagonal block. A product's covariance is `L L*`, a composition, so the block comes through projections and the marginal has neither factor nor precision: the marginal of `from_product([s1, s2])` of two invariant measures, of every `Layered.sobolev_measure`, and of a `GaussianPosterior` on a direct sum cannot be sampled, though the joint can and `push_forward(domain.projection(key))` can. The comment's reason is wrong: `P_i L` is a factor of `C_ii` (from the joint latent space), it just is not `L_ii`. The docstring says the marginal "is the pushforward under the projection", and D-61 says marginalised measures keep the fast draw; neither holds. Verified by running.
Reproduction:
```
from_product([s1, s2]).push_forward(projection(1)).can_sample   True
from_product([s1, s2]).marginal(1).can_sample                   False   (covariance agrees with s2)
correlated from_product marginal(0).can_sample                  False
interval.Layered([0,.5,1], 12, ...).sobolev_measure(1.5).marginal(0)   can_sample False, precision None
LinearGaussianInversion(problem, joint)(data).marginal(1).can_sample  False (posterior itself True)
```
Proposed fix: take the block of a block-diagonal factor (and of a block-diagonal precision) when there is one, otherwise `projection @ L` as the factor, and carry a projected sampler when there is no factor.

### `from_product` never keeps a precision factor; the branch that would is dead
`pygeoinf2/probability/gaussian.py:599`–`:606` — the precision factor is built in an `elif` after `if all(m.precision is not None ...)`, but the constructor derives a precision from every precision factor, so the first branch always wins and the factors' `Li` are dropped. Products then lose what the correlated product, `kl_divergence`'s similarity, the low-rank posterior's `Li` route and the mini-apps' whitening read (distinct from CURRENT_STATE §7 (5), which is about invariant measures that never had one). Verified by running: two `from_covariance_matrix` factors, both with precision factors, give a product with `precision` and `precision_factor is None`.
Proposed fix: build the block-diagonal precision factor whenever every factor has one, independently of the precision.

### Scaling a sampler-only Gaussian loses the sampler
`pygeoinf2/probability/gaussian.py:2794` — `_combine_scale` passes no `sample=` to `_rebuild`, unlike `_combine_affine` and `_combine_add`. So `2.0 * post` and `post.rescale_directional_variance(...)` of a conditioned measure or posterior cannot be sampled. Verified by running: `post.can_sample` True, `(2.0 * post).can_sample` False, `post.rescale_directional_variance(e1, 1.0).can_sample` False.
Proposed fix: carry a centred scaled sampler, `alpha * (draw - m)`, when there is no factor.

### `AbsolutelyContinuousMeasure.can_sample` is True and `sample` always raises
`pygeoinf2/probability/absolutely_continuous.py:178`, `pygeoinf2/probability/base.py:74` — the base docstring says `can_sample` is True "because `sample` is abstract and so every concrete measure implements it"; this class implements it as a refusal. Push-forwards and products inherit the lie (`(2.0 * acm).can_sample` is True), and `GaussianMixture.from_parameter_samples`, `plotting/distributions.py:72` and the inference layer's sampler checks (`inference/gaussian.py:742`) branch on it. Verified by running.
Proposed fix: override `can_sample` to return False here, and correct the base docstring.

### `credible_set` claims positive definiteness on the caller's behalf
`pygeoinf2/probability/gaussian.py:1760` — without a precision the covariance is given `POSITIVE_DEFINITE` (and so `INVERTIBLE`) by `with_traits` before CG inverts it. A semidefinite covariance (`from_samples` with few draws, a low-rank approximation, an exact conditioning) builds a set whose first membership test fails deep in CG with "The operator claims POSITIVE_DEFINITE but is not; verify the claim", a claim the user never made. `kl_divergence` refuses the same situation at the door unless the trait is claimed (`:1484`), and `log_normalizing_constant` (`:2531`) and `condition` (`:2264`, on `A C A*` when the constraint is exact) also grant it silently. Verified by running: `from_samples(X, two draws).credible_set(level=0.9)` builds, `contains(mean)` is True, `contains(random)` raises `ConvergenceError`.
Proposed fix: refuse as `_kl_similarity` does when the covariance does not claim `POSITIVE_DEFINITE`, saying a singular measure has no credible ellipsoid in the whole space.

### Stochastic answers returned as bare, unseeded floats: `hilbert_schmidt_norm`, `log_normalizing_constant` and the mixture density
`pygeoinf2/probability/gaussian.py:1049`, `:2534`; `pygeoinf2/probability/mixture.py:344` — `kl_divergence` sets the library's rule ("auto refuses rather than going stochastic silently: an estimate reported as a bare number is the one way this quantity misleads"), and the same class breaks it twice. `hilbert_schmidt_norm()` on a space without coordinates silently takes the Hutchinson route (the docstring's "auto" does not mention it), while `nuclear_norm()` on the same space refuses. `log_normalizing_constant` calls `log_determinant(...).value`, which is stochastic above 4000 dimensions or without coordinates, with no `rng`, and caches it, so `GaussianMixture.log_density` and `marginal_probabilities` differ between two builds of the same mixture. Verified by running on the suite's `ListSpace` (dim 6): exact HS 3.162 against 3.053 and 3.239 at two seeds; `nuclear_norm()` raises `TypeError`; exact log normalizer -5.278 against -5.155 and -5.098 on two instances; a two-component mixture's `log_density` at one point -7.987 and -7.951.
Proposed fix: one rule across the class: exact routes or refusal on "auto", stochastic only by name and returned as an `Estimate` (an `_estimate` twin as for KL), with `rng=` where it samples.

### `method=` names: "spectral" means two different routes, and one route has two names
`pygeoinf2/probability/gaussian.py:1265`, `:1888`, `:1900`, `:1003` — in `kl_divergence` `"spectral"` is the exact read of two diagonal spectra; in `ambient_ball` `"spectral"` is a randomized eigendecomposition truncated to `rank`, a lower bound, and the exact diagonal read is called `"diagonal"`, as it is in `hilbert_schmidt_norm` and `nuclear_norm`. A caller who learns `method="spectral"` on one method gets a different algorithm on the next. Read only.
Proposed fix: `"diagonal"` for the exact spectral read everywhere (KL included), and `"randomized"` for the truncated eigendecomposition, the word `random_eig` already uses.

### `from_standard_deviation` refuses with `TypeError`; every other `unsafe=` gate raises `ValueError`
`pygeoinf2/probability/gaussian.py:351` — the same predicate (`is_function_space`) raises `ValueError` in `_require_density`, the trace-class refusal of `sobolev_measure` is `ValueError`, and so is point evaluation's. A caller catching the refusal class has to know which constructor it called. Verified by running.
Proposed fix: `ValueError`, as the other gates.

### Dense assemblies ignore the `n_jobs` the method was given
`pygeoinf2/probability/gaussian.py:1039`, `:1131`, `:1179`, `:2103`, `:2116` — `covariance.matrix(...)` costs one application per column and takes `n_jobs` (`with_dense_covariance` passes it, `:1638`), but the dense routes of `hilbert_schmidt_norm`, `nuclear_norm`, `kl_divergence` (through `_symmetric_matrix`) and `ambient_ball` (covariance and operator) call it serially although each method takes `n_jobs` for its other routes. `log_normalizing_constant` takes no `n_jobs` and passes none to `log_determinant`. Read only.
Proposed fix: pass the method's `n_jobs` to every `matrix()` call; give `log_normalizing_constant` the knobs `log_determinant` has.

### Sampling routes hold every draw in memory
`pygeoinf2/probability/gaussian.py:2042`, `:1812` — `ambient_ball(method="sampling")` (the "auto" choice above `dense_limit`, 10 000 draws by default) and `sample_pointwise_variance` materialize the whole `samples(n)` list before reducing it to one number per draw or one running sum, and the operator form's `(B (x - m), x - m)` loop then runs serially. Peak memory is one field per draw: measured with `tracemalloc` on a 4096-point circle, 3.7 MB at 100 draws and 13.8 MB at 400 for `sample_pointwise_variance`, 3.6 and 13.7 MB for `ambient_ball`; at the default 10 000 draws a 10^5-coefficient field is 8 GB. Verified by running.
Proposed fix: reduce inside the parallel map (the squared norm, or chunked running sums), never returning the draws.

### A diagonal covariance passed to the constructor gets no factor or precision, though the algebra gives it both
`pygeoinf2/probability/gaussian.py:258` — the constructor derives a factor only for an exact `is_identity` covariance. `GaussianMeasure(E, covariance=DiagonalLinearOperator(E, [1, 2, 3]))` cannot be sampled and has no precision, nor can `covariance=0.25 * I`; yet adding a zero diagonal measure to it returns one with both, through `_diagonal_inverse`. D-61 says a diagonal covariance yields its square-root factor and, when positive, its precision. Verified by running.
Proposed fix: apply `_diagonal_inverse` in the constructor when no factor or precision is supplied.

### `as_multivariate_normal` hands over the density the density family refuses
`pygeoinf2/probability/gaussian.py:2173` — on a function space `log_density`, `log_normalizing_constant` and the rest refuse without `unsafe=True` (D-131), but `as_multivariate_normal().logpdf` returns the component density with no gate (-0.0143 for a draw of an order-2 Sobolev measure on a circle). It is dense by name, but it is also the one-line way round the refusal, and its docstring does not mention it. Verified by running.
Proposed fix: the same `is_function_space` gate with `unsafe=`, or at least a sentence in the docstring saying the density it carries is the truncation's.

### `kl_divergence` on a function space grows with the truncation for measures that are not equivalent, unrefused and undocumented
`pygeoinf2/probability/gaussian.py:1240` — between two Sobolev priors of different scale (mutually singular in the limit) the divergence is an artefact of the truncation exactly as `log_density` is: 45.6, 265.5, 1161.5, 4749.9 at 32, 128, 512, 2048 points on a circle (verified by running). It has a limit only for equivalent measures, a posterior against its prior being the common case. Nothing says so.
Proposed fix: say it in the docstring; consider the `unsafe=` gate on a function space unless one measure is the other's reference (`AbsolutelyContinuousMeasure.reference`).

### `check_measure` never checks the precision, its factor or the covariance factor against the covariance, and takes no `atol`
`pygeoinf2/testing.py:1040` — it compares draws with the declared mean and covariance only, so a measure whose precision is wrong passes: the correlated product above, with `max |P C v - v|` = 2.5, passes at 4000 draws. The brief's contract for checks includes `atol=`; this one has none. Verified by running.
Proposed fix: when a precision exists, check `P C u == u` on the probes (and `Li* Li == P`, `L L* == C` for the factors) to `rtol`/`atol`.

## Could fix

### `GaussianMixture.has_covariance` raises instead of answering
`pygeoinf2/probability/mixture.py:238` — the base contract is that `covariance` returns `None` when unavailable and `has_covariance` answers with a bool; the mixture's `covariance` raises `ValueError` for a precision-only component, so `has_covariance` raises too. Verified by running.
Proposed fix: return `None` from `covariance` in that case and keep the message for callers who use it directly.

### A mixture plus a Gaussian degrades to the private `_IndependentSum`
`pygeoinf2/probability/mixture.py` (no `_combine_add`), `pygeoinf2/probability/base.py:478` — the sum of a mixture and an independent Gaussian is the mixture of the sums, with closed-form density and weights, but `mix + g` is an `_IndependentSum` with no density, and a private class name reaches the user. This is the data predictive of a mixture prior (`data_measure_from_model_measure`). Verified by running.
Proposed fix: `GaussianMixture._combine_add` for a Gaussian summand, and a public name (or a documented return type) for the generic sum.

### The two product kinds name their marginal differently
`pygeoinf2/probability/base.py:564` — `product()` returns a `GaussianMeasure` with `marginal(key)` and no `factor`, or a `ProductMeasure` with `factor(key)` and no `marginal`, so code on `product()`'s result must branch on type. Verified by running.
Proposed fix: give `ProductMeasure` a `marginal` (it is the factor).

### `-mu` is not defined, though `mu - nu` is
`pygeoinf2/probability/base.py:412` — `__sub__` goes through `-1.0 * other`, but there is no `__neg__` (v1 had one): `-mu` raises `TypeError: bad operand type for unary -`. Verified by running.
Proposed fix: `__neg__` as `-1.0 * self`.

### `product()` refuses a precision-only Gaussian
`pygeoinf2/probability/gaussian.py:595` — "Every factor needs a covariance or a covariance factor", though the block-diagonal precision is well defined and the generic `ProductMeasure` would at least hold it. Verified by running.
Proposed fix: build the precision-only product, or fall back to `ProductMeasure`.

### `ambient_ball` validates `level` only on some routes, and after drawing
`pygeoinf2/probability/gaussian.py:2042` — the diagonal and dense routes refuse `level` outside `(0, 1)`; the sampling route returns the smallest or largest draw for 0 and 1 and refuses 1.5 only after taking every draw. Verified by running (`level=0.0` gives 0.40, `level=1.0` gives 3.00 on `EuclideanSpace(3)`).
Proposed fix: validate `level` first, as `credible_set` does.

### `affine_map` promises a translation check it does not make
`pygeoinf2/probability/base.py:319` — the docstring raises `ValueError` "if the translation does not live in its codomain"; nothing checks, and on an array space the translation broadcasts: mapping `EuclideanSpace(3)` to `EuclideanSpace(1)` with a 3-vector translation gives a measure on a 1-dimensional space whose mean and draws have shape `(3,)`. Verified by running.
Proposed fix: drop the claim, or check the shape where the codomain has components.

### `combine_log_normalizer` is public with one caller in another layer
`pygeoinf2/probability/absolutely_continuous.py:39`, `:198` — in the module's `__all__`, not the package's, used only by `inference/gaussian.py`. Read only.
Proposed fix: make it private or export it; at present it is half public.

### Tests reach private members because no public reading exists
`pygeoinf2/tests/test_layered.py:334`, `test_spectral.py:679`, `:746`, `test_probability.py:1014` — `_diagonal_eigenvalues()` is the only way to read the spectrum of a product of diagonal measures (D-61's `covariance.eigenvalues` exists only when the covariance is itself diagonal, not for `L L*` of a block-diagonal factor), and `_quantile_method_for` is tested directly. Read only.
Proposed fix: either accept these as white-box tests or expose the spectrum where it is known.

### Missing-ingredient errors use two classes
`pygeoinf2/probability/gaussian.py:2528` against `:1150` — a measure without a covariance gets `NotImplementedError` from `log_normalizing_constant` and `ValueError` from `nuclear_norm`, `directional_variance` and every `_require_covariance` site; a missing precision gets `NotImplementedError`. The base reserves `NotImplementedError` for a measure *type* without a density; for an instance missing an operator it should be one class throughout. Verified by running.
Proposed fix: `ValueError` for a missing operator on an instance.

### History in three docstrings and comments
`pygeoinf2/probability/gaussian.py:1155` ("what came back was a `TypeError`..."), `:1369` ("quietly became an O(n^3) one"), `:2327` ("the alternative was slicing..."). Read only.
Proposed fix: state the reason without the past.

### Docstring gaps
`pygeoinf2/probability/gaussian.py:231` — the constructor has no Raises section and does not say a supplied `sample` must be centred (see the first finding); `:1571` — `low_rank_approximation`'s `rank` is adaptive when omitted, not stated; `:1859` — `ambient_ball`'s "the ball is always the larger" is true of volume only, and reads as containment, which D-64 says does not hold; `:2593` — `has_log_density` is True on a function space where `log_density` refuses; `pygeoinf2/spectral_element/layered.py:736` — `Layered.sobolev_measure` lists only the wrong-length error, not the order refusal it passes through. Read only.
Proposed fix: a sentence each.

### One meaning, two keyword names
`pygeoinf2/probability/gaussian.py:983`, `:1190` — "stop sampling at this relative standard error" is `rtol=` on `hilbert_schmidt_norm` and `nuclear_norm` and `sample_rtol=` on `kl_divergence` (where `rtol` goes to Lanczos through `**kwargs`, which the exact routes then ignore silently, a misspelt keyword included); the draw count is `samples` almost everywhere but `count` in `GaussianMixture.from_parameter_samples`. Read only.
Proposed fix: `sample_rtol` wherever an inner `rtol` can also exist, or everywhere; `samples` for the count.

## What is sound

- **The trace-class refusal**: `sobolev_measure` refuses at and below half the eigenvalue dimension and accepts just above, on the circle (1/2), sphere (1), interval (1/2), radial profile (1/2, through `_eigenvalue_dimension`), disk (1) and ball (3/2); `Layered.sobolev_measure` passes the refusal and `unsafe=` through; `from_standard_deviation` refuses on a layered space as well as a symmetric one.
- **The density family's gates** (D-131): `log_density`, `grad_log_density`, `mahalanobis_squared`, `log_normalizing_constant`, `credible_set`, and the mixture's `log_density` and `marginal_probabilities` all refuse on a function space and run with `unsafe=True`.
- **Moments against draws** (`check_measure`, 3000 to 6000 draws, all pass): `from_covariance_matrix` in both forms on the dense-metric and weighted spaces, affine images and scalings of factored measures, the independent product, the correlated product of Sobolev measures, `correlated_measure`, mixtures, `from_samples`, `low_rank_approximation`, `with_dense_covariance` (it rebuilds with a factor, so it also cures the first finding), Sobolev measures on sphere, circle, interval, radial, disk and ball, the pointwise-calibrated spectral-element measure, and layered priors independent and correlated.
- **`ambient_ball`** on an 8-dimensional dense metric: the dense, sampling (40 000 draws) and randomized routes agree, 15.814 / 15.817 / 15.814 without an operator and 21.755 / 21.846 / 21.755 with one; coverage of the dense ball and ellipsoid by 4000 draws 0.907 and 0.905 at level 0.9; the operator and inverse checks refuse what they should.
- **`credible_set`** on a dense metric: coverage 0.901 through the precision and 0.906 through CG at level 0.9.
- **The density's constant** is the volume density in the space's metric: `log_density + log_normalizing_constant` equals scipy's component density shifted by `-log det G / 2` on the weighted (det 9) and dense (det 1.17) metrics.
- **KL**: spectral and dense agree to 1e-14 on a sphere; stochastic within 1.5 standard errors of dense on a dense metric; a covariance-only reference without a definiteness claim is refused with a message that says what to do; "auto" refuses rather than estimate.
- **Sampling machinery**: `samples(n_jobs=2)` gives the same draws as serial for one seed under the process backend; `sample_pointwise_variance(n_jobs=2)` runs; `condition`'s Matheron draws centre on the posterior mean.
- **Coordinate-free**: on the suite's `ListSpace`, `credible_set`, `ambient_ball` and `nuclear_norm(method="stochastic")` work and the coordinate routes refuse by name.
- **`from_product(domain=)`** seats a layered prior on the layered space, keeping `sample_pointwise_std`; `_latent_mixing` passes `check_operator` and `check_traits` (worst 8e-16) between two weighted spaces of different metrics; `spectral_correlations` and `cross_covariance` read the correlated measure exactly.
- No decision numbers in the layer's code; no British spelling in identifiers; v1 renames (`affine_map`, `precision`, `from_product`) are those D-99 settled.

## Fixed

Fix package A, 2026-10-29. Tests named are in `pygeoinf2/tests/`; every new
one was run against the tree at `HEAD` and fails there (the positive
controls excepted).

### Must fix

1. **Derived samplers draw about their own mean.**
   `probability/gaussian.py`: new `GaussianMeasure._fluctuation` (one
   centered draw, through `sample`, so a subclass's override is honored);
   the samplers built by `_combine_affine`, `_combine_add` and
   `from_product` now return the centered fluctuation and carry no
   translation; the constructor's `sample=` docstring states that a
   supplied callable returns the centered fluctuation `x - m`, to which
   `sample` adds the expectation. Tests:
   `test_probability.py::TestDerivedSamplersDrawAboutTheirOwnMean` —
   translate, push-forward, sum, product, scale and marginal of a
   conditioned measure with a nonzero mean on a dense metric, each through
   `check_measure` and a sample mean; `problem.joint_measure(posterior)` and
   `data_measure_from_model_measure` of a nonzero-mean Sobolev prior on a
   circle.
2. **The correlated product's precision.** `probability/gaussian.py`: new
   `_latent_whitening`; `_correlated_product` builds the latent whitening
   from the factor's precision factor only when one probe on the latent
   space confirms `Li L xi == xi`, otherwise from `L* P` (the factor's
   adjoint after the precision) when that probe passes, otherwise gives the
   product no precision. So invariant measures, which carry a precision and
   no precision factor, now give a correlated product a precision too.
   `testing.py::check_measure` now applies the precision to the
   covariance's image of each probe (`P C u == u`), and checks `L L* u ==
   C u` and `Li* Li u == P u` where those operators exist, before drawing,
   to `atol + rtol * scale`; it gains `atol=`. Tests:
   `test_probability.py::TestTheCorrelatedPrecisionWhateverTheRoot` (after
   `-1.0 * a`, and with a hand-built Cholesky factor whose precision factor
   carries a QR rotation, on a dense metric; a non-invertible factor gives
   no precision) and `::TestTheCheckSeesThePrecision` (a wrong precision, a
   mismatched factor, a mismatched precision factor, `atol`).
3. **`from_covariance_matrix` refuses an asymmetric matrix.** Refused when
   `||M - M^T|| > 1e-8 ||M||` after conversion to Galerkin form, with a
   message naming the likely wrong `form=` (a symmetric component matrix on
   a non-identity metric: pass `form='galerkin'`; an asymmetric Galerkin
   matrix: pass `form='components'`); symmetrized below that. Tests:
   `test_probability.py::TestACovarianceMatrixIsNotSymmetrizedIntoAnotherOne`.
4. **The identity covariance on a function space, every route.** One
   helper, `probability/gaussian.py::_refuse_identity_covariance` (with
   `_is_constant` and `_is_scaled_identity`), refusing with `ValueError` a
   constant spectrum or deviation, or a covariance, factor, precision or
   precision factor known to be a multiple of the identity, on a space with
   `is_function_space`, without `unsafe=True`. Applied in the
   `GaussianMeasure` constructor (new `unsafe=`), `from_standard_deviation`
   (now `ValueError`, not `TypeError`), `from_standard_deviations` and
   `from_covariance_matrix` (new `unsafe=`); in
   `symmetric_space/base.py` `invariant_measure`, `heat_measure` (zero
   length), `correlated_measure` (a field whose variances are the same at
   every mode), `correlated_measure_from_correlations` and `power_measure`
   (a power the same at every degree, or the same per component), each
   with a new `unsafe=`, and `sobolev_measure` passing its `unsafe` through;
   in `spectral_element/base.py` `spectral_measure` (new `unsafe=`), with
   `sobolev_measure` (and so `Layered.sobolev_measure`) passing it through.
   Each docstring says that the trace-class condition beyond the literal
   identity is the caller's. Measures the library derives from others (the
   algebra through `_rebuild`, products, marginals, conditioning, the dense
   and low-rank rebuilds, `from_samples`) are not judged again. Tests:
   `test_measures.py::TestTheIdentityIsRefusedByEveryRoute` (17 routes,
   refused and then built with `unsafe=True`; decaying spectra accepted on
   each route; Euclidean spaces untouched; the algebra on an unsafe measure
   not refused).
   - `tutorials/09_nonlinear.py` re-posed on `X.sobolev_measure(1.0, 0.2,
     pointwise_std=0.8)` with `MaximumAPosteriori(..., gauss_newton=True,
     optimizer=NewtonCG(preconditioner=prior.covariance))`; the optimizer
     argument rewritten against what it prints: Newton-CG converges in 10
     iterations at 128, 256 and 512 points, L-BFGS in 158, 151 and 180;
     the failure protocol is now shown on the L-BFGS run of the same
     problem, which overshoots (8 failed evaluations) where the
     preconditioned Newton steps never do, and reaches the same mode
     (agreement 1.6e-06).
   - `tutorials/13_ray_tomography.py`: the whitened identity prior and
     `L @` removed; MAP on the problem itself, Newton-CG preconditioned by
     the prior covariance (12 iterations, converged, error 0.425) against
     L-BFGS capped at 100 (not converged, error 0.589, against 0.547 for
     straight rays); the text says so. The Laplace spread is taken from a
     `gauss_newton=True` estimator placed on Newton's mode with CG
     preconditioned by the prior covariance: with the inference package's
     exact-Hessian Laplace each covariance application of this problem
     costs about 28 times the Gauss-Newton one, 2 min 40 s for the 36
     points, against 35 s for the whole tutorial now.
   - `tests/test_laplace.py`: `tutorial_nine_problem` and `ray_problem`
     re-posed the same way; the recorded answers re-recorded (the ray
     problem's objective and misfit agree with the old whitened recording
     to 1e-12 and 1e-8, the same mode in another variable), the class
     docstring stripped of its history. `test_minimization.py`'s Occam ray
     test preconditions its MAP.
   - Broken by the gate and repaired: `examples/10_measures.py` and
     `tutorials/08_regularization.py` catch `ValueError`; the
     `TypeError` expectations in `test_measures.py` and `test_probability.py`;
     a precision-only test that used the identity (`test_probability.py`);
     four `test_spectral.py` tests and two `test_space_parity.py` power
     tests that used a flat spectrum as a convenience now use a decaying
     one.
   - Notebooks regenerated (`make_notebooks.py`: 08, 09, 13).

### Should fix

- **`marginal`** (`gaussian.py::GaussianMeasure.marginal`): the factor is
  the diagonal block's root, else the block of a block-diagonal factor,
  else `P L` from the joint factor; the precision or precision factor is
  the block of a block-diagonal one; a joint measure sampling through a
  callable gives a marginal sampling its projected fluctuation. Tests:
  `test_probability.py::TestMarginalsKeepWhatTheJointHas` (independent
  product of invariant measures, correlated product, layered prior,
  product of matrix measures) and the `marginal` case of item 1.
- **`from_product` keeps the precision factor**, built whenever every
  factor has one, independently of the precision. Test:
  `TestMarginalsKeepWhatTheJointHas::test_a_product_keeps_its_precision_factor`.
- **Scaling a sampler-only measure keeps its sampler**
  (`_combine_scale`, centered). Test: the `scale` case of item 1.
- **`AbsolutelyContinuousMeasure.can_sample` is False**; the base
  docstring no longer claims every concrete measure samples. Test:
  `TestSmallerShouldFixes::test_a_density_cannot_be_sampled_and_says_so`
  (also its scaling and push-forward).
- **Stochastic answers not returned as exact**: `hilbert_schmidt_norm`'s
  `"auto"` refuses on a space without coordinates (`TypeError`, as
  `nuclear_norm`) instead of falling back to Hutchinson;
  `log_normalizing_constant` is exact or refused: the diagonal spectrum,
  else the dense factorization on a coordinate space up to a new
  `dense_limit=` (4000), with a new `n_jobs=` for the assembly, else
  `ValueError`, so the mixture's densities and weights are deterministic.
  Tests: `TestSmallerShouldFixes::test_a_hilbert_schmidt_norm_without_coordinates_is_not_guessed`,
  `::test_the_normalizing_constant_is_exact_or_refused`;
  `test_no_array_backend.py`'s test that `"auto"` took the stochastic route
  now asserts the refusal.
- **One name per route**: `kl_divergence`'s exact spectral read is
  `method="diagonal"`, as in the norms and `ambient_ball`; `ambient_ball`'s
  truncated randomized eigendecomposition is `method="randomized"`. Callers
  updated: `test_spectral.py`, `test_probability.py`,
  `test_miniapp_cpp_backend.py`, `miniapps/cpp_backend/sets.py` and its
  `IMPLEMENTATION.md`. Test: `TestSmallerShouldFixes::test_one_name_per_route`.
- **Sampling routes no longer hold every draw**: new
  `ProbabilityMeasure._map_draws` (a function of each draw, run where the
  draw is made, on the streams `samples` uses; `samples` is built on it);
  `ambient_ball(method="sampling")` reduces each draw to its squared norm
  (or `(B (x - m), x - m)`) in the worker; `sample_pointwise_variance` sums
  blocks of `16 * n_jobs` squared deviations. The draws are those of
  `samples` for the same seed. Tests:
  `TestSmallerShouldFixes::test_the_sampling_routes_never_hold_the_draws`
  (with `samples` patched to fail), `::test_the_sampled_pointwise_variance_streams_the_same_draws`.
- **`from_standard_deviation` raises `ValueError`** (also `spaces.md`,
  "two exception classes"). Tests: the identity-gate tests above.
- **Dense assemblies take the method's `n_jobs`**: `hilbert_schmidt_norm`
  and `nuclear_norm` (`"dense"`), `kl_divergence` (`"dense"`, through
  `_symmetric_matrix(n_jobs=)`), `ambient_ball` (`"dense"`, covariance and
  operator), `log_normalizing_constant`. Covered by the existing parity
  tests of those routes.
- **A diagonal covariance given to the constructor** gets its root and,
  for a positive spectrum, its precision (`_diagonal_inverse`, extended to
  any operator with a known scaling of components on a coordinate space
  with a diagonal metric, a multiple of the identity among them). Test:
  `TestSmallerShouldFixes::test_a_diagonal_covariance_brings_its_factor_and_precision`.
- **`as_multivariate_normal`**: the docstring says its density is the
  truncation's on a function space and carries no gate (the gate itself
  skipped, below).
- **`kl_divergence` on a function space**: the docstring says the
  divergence has a limit only for equivalent measures and grows with the
  truncation otherwise (the gate skipped, below).
- **`check_measure` takes `atol=`** and checks the precision and both
  factors (item 2).

### Could fix, done on the way

- `-mu` (`ProbabilityMeasure.__neg__`). Test: `::test_a_measure_can_be_negated`.
- `ambient_ball` validates `level` before any route or draw. Test:
  `::test_a_level_is_validated_before_any_draw`.
- History removed from the three docstrings and comments named
  (`_require_covariance`, the KL spectral comment, `marginal`).
- Docstring gaps: the constructor's Raises and centered sampler,
  `low_rank_approximation`'s adaptive rank, `ambient_ball`'s "larger"
  (now: neither contains the other, the ball has the larger volume),
  `has_log_density` on a function space, `Layered.sobolev_measure`'s order
  refusal.

## Skipped

- **`credible_set` claiming definiteness.** Refusing a covariance that
  does not claim `POSITIVE_DEFINITE` breaks tutorial 6, whose property
  posterior and prior (push-forwards, which claim only semidefiniteness)
  build credible sets this way. A decision is needed: verify definiteness
  densely on a small space, require the caller's claim, or keep the claim
  and name it in the error CG raises. `condition`'s claim on `A C A*` for
  an exact constraint is the same question.
- **Decision 2, the wording outside the probability files.** The brief
  keeps this package's edits to the probability files. `consistency_set`
  is already consistent with the potential (it passes `unsafe=True`, and
  both docstrings state the trace-class condition), made so by the
  inference package. Still to place, proposed text:
  - `numerics/normal.py::choose_formalism`, for "and the model-space
    formalism is kept for when there *is* one — an overdetermined problem
    with a cheap precision, or a surrogate.": "and the model-space
    formalism is kept for when there *is* one: an explicitly
    finite-dimensional problem, overdetermined and with a cheap precision,
    or a low-dimensional surrogate. It is not for a model space that
    discretizes a function space, where its evidence adds log-determinants
    of the prior covariance and of the normal operator that diverge in
    opposite directions under refinement, and on the stochastic route their
    standard errors grow with the discretization."
  - `inference/gaussian.py::LinearGaussianInversion.__init__`, `formalism`:
    append "The model-space formalism is for explicitly finite-dimensional
    problems and low-dimensional surrogates; see
    :func:`~pygeoinf2.numerics.normal.choose_formalism`."
  - `inference/mixture.py::LinearGaussianMixtureInversion.__init__`,
    `evidence`: for "as a dict -- ``{"method": "stochastic", "samples":
    200}`` to keep the weight update matrix-free on a large data space.":
    "as a dict. The direct determinant, taken by default wherever the data
    space can be assembled, is exact and is the one to prefer: the weights
    are then exact. ``{"method": "stochastic", "samples": 200}`` keeps the
    weight update matrix-free on a data space too large for that, at the
    cost of weights that carry the estimate's error."
- **`as_multivariate_normal` and `kl_divergence` gates (`unsafe=`).** New
  refusals on existing public methods; the docstrings now state the
  behavior, the gates need a decision.
- **Could fix not taken**: `GaussianMixture.has_covariance` raising, the
  mixture plus a Gaussian degrading to `_IndependentSum`,
  `ProductMeasure.marginal`, a precision-only `product()`, the
  `affine_map` translation check, `combine_log_normalizer`'s half-public
  status, tests reading `_diagonal_eigenvalues`, the two error classes for
  a missing operator, and the `rtol`/`sample_rtol`/`count` keyword names:
  each an API or design choice.
